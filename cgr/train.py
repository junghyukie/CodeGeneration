"""LearnTask (Algorithm 2): affinity prior, GMM registration, tau calibration, Phase A (compose), Phase B (specialise)."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from cgr.config import CGRConfig
from cgr.data import Example, TaskData
from cgr.gmm import fit_diag_gmm, within_scatter
from cgr.state import CGRState
from utils.data.data_collator import DataCollator


def _collate(state: CGRState, task: str, examples: List[Example]):
    col = DataCollator(state.tokenizer, padding="longest", max_prompt_len=state.cfg.prompt_len(task),
                       max_ans_len=state.cfg.ans_len(task), pad_to_multiple_of=8, inference=False)
    return col([{"prompt": e.prompt, "answer": e.answer} for e in examples])


def _nll_sum(state: CGRState, batch, row: int):
    """Masked NLL summed over target tokens, with coefficients c^(l) = row `row` of every C^(l)."""
    dev = state.encoder.device
    ids, mask, labels = batch["input_ids"].to(dev), batch["attention_mask"].to(dev), batch["labels"].to(dev)
    with state.bank.use(state.composition.task_coefficients(row)):
        hidden = state.model.get_decoder()(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    tgt = labels[:, 1:]
    valid = tgt != -100
    logits = state.model.get_output_embeddings()(hidden[:, :-1][valid]).float()
    return F.cross_entropy(logits, tgt[valid], reduction="sum"), int(valid.sum())


@torch.no_grad()
def heldout_loss(state: CGRState, task: str, heldout: List[Example], row: int) -> float:
    """Token-mean NLL on H_k in eval mode (dropout off)."""
    was = state.model.training
    state.model.eval()
    tot, n = 0.0, 0
    mb = state.cfg.micro_batch_size
    for s in range(0, len(heldout), mb):
        loss, ntok = _nll_sum(state, _collate(state, task, heldout[s:s + mb]), row)
        tot += float(loss)
        n += ntok
    state.model.train(was)
    return tot / max(n, 1)


def _cyclic_schedule(n: int, steps: int, bs: int, rng: random.Random) -> List[List[int]]:
    """Phase A: `steps` batches drawn from successive shuffled passes over the data."""
    out, pool = [], []
    for _ in range(steps):
        batch = []
        while len(batch) < bs:
            if not pool:
                pool = list(range(n))
                rng.shuffle(pool)
            batch.append(pool.pop())
        out.append(batch)
    return out


def _epoch_schedule(n: int, epochs: int, bs: int, rng: random.Random) -> List[List[int]]:
    """Phase B: `epochs` shuffled passes, batch size bs (last batch of an epoch may be smaller)."""
    out = []
    for _ in range(epochs):
        order = list(range(n))
        rng.shuffle(order)
        out += [order[s:s + bs] for s in range(0, n, bs)]
    return out


def run_phase(state: CGRState, task: TaskData, row: int, groups: List[dict], lr_lambdas: List[Callable],
              schedule: List[List[int]], log: Callable, phase: str) -> None:
    """One optimizer over the whole schedule; each step is one batch of `cfg.batch_size` examples."""
    cfg = state.cfg
    params = [p for g in groups for p in g["params"]]
    for p in params:
        p.grad = None
    opt = torch.optim.AdamW(groups, betas=cfg.adam_betas, eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambdas)
    state.model.train()
    mb = cfg.micro_batch_size
    t0 = time.time()
    for step, idx in enumerate(schedule, start=1):
        batches = [_collate(state, task.name, [task.train[i] for i in idx[s:s + mb]]) for s in range(0, len(idx), mb)]
        if cfg.loss_normalization == "token":
            denom = sum(int((b["labels"][:, 1:] != -100).sum()) for b in batches)
        else:
            denom = len(idx)
        total = 0.0
        for b in batches:
            loss_sum, _ = _nll_sum(state, b, row)
            (loss_sum / max(denom, 1)).backward()
            total += float(loss_sum.detach())
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step == 1 or step % 10 == 0:
            log({"phase": phase, "step": step, "of": len(schedule), "train_loss": total / max(denom, 1),
                 "sec": round(time.time() - t0, 1)})
        if phase == "B" and step % cfg.heldout_log_interval == 0 and task.heldout:
            log({"phase": "B", "step": step, "heldout_loss": heldout_loss(state, task.name, task.heldout, row)})


def phase_b(state: CGRState, task: TaskData, row: int, atom: int, beta_row: Optional[int], log: Callable,
            rng: random.Random) -> None:
    """Train atom `atom` (AdamW 1e-4, wd 0.01, 1% warmup, constant) and optionally beta row `beta_row` (2e-5, no wd)."""
    cfg = state.cfg
    params = state.bank.atom_parameters(atom)
    for p in params:
        p.requires_grad_(True)
    schedule = _epoch_schedule(len(task.train), cfg.phase_b_epochs, cfg.batch_size, rng)
    warm = max(1, math.ceil(cfg.phase_b_warmup_ratio * len(schedule)))
    groups = [{"params": params, "lr": cfg.phase_b_lr, "weight_decay": cfg.phase_b_weight_decay}]
    lambdas = [lambda s: min(1.0, (s + 1) / warm)]
    if beta_row is not None:
        groups.append({"params": [state.composition.trainable_row(beta_row)], "lr": cfg.phase_b_beta_lr,
                       "weight_decay": 0.0})
        lambdas.append(lambda s: 1.0)
    run_phase(state, task, row, groups, lambdas, schedule, log, "B")


def _cuda_peak_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2**30, 2)


def learn_task(k: int, task: TaskData, state: CGRState, cfg: CGRConfig, log_dir: Optional[str] = None,
               on_phase_a_end: Optional[Callable[[int], None]] = None) -> Dict:
    """Algorithm 2 for task index k (0-based). Uses only the current task's data."""
    records: List[dict] = []

    def log(rec: dict):
        rec = {"task": k + 1, **rec}
        records.append(rec)
        print(json.dumps(rec), flush=True)

    t_start = time.time()
    torch.manual_seed(cfg.seed + k)
    rng = random.Random(cfg.seed + k)

    # Routing embeddings of up to N_max training prompts (adapters disabled).
    sample = list(range(len(task.train)))
    random.Random(cfg.gmm_seed + k).shuffle(sample)
    Z = state.encoder.embed([task.train[i].prompt for i in sample[:cfg.gmm_max_samples]])

    # Affinity prior and beta initialisation (router state before registering k).
    state.composition.add_row()
    if k >= 1:
        a = state.router.affinity(Z, cfg.affinity_temperature)
        log({"affinity": a.tolist()})
        if cfg.beta_init == "affinity":
            state.composition.init_row(k, a)

    # Fit and register GMM k; discard Z.
    gmm, resp = fit_diag_gmm(Z, cfg.gmm_components, cfg.em_max_iters, cfg.em_tol, cfg.var_floor, cfg.gmm_seed + k)
    state.router.register(gmm, within_scatter(Z, gmm, resp), Z.shape[0])
    log({"n_train": len(task.train), "n_heldout": len(task.heldout), "N_k": int(Z.shape[0]),
         "gmm_em_iters": len(gmm.ll_history), "gmm_ll": gmm.ll_history[-1]})
    del Z, resp

    if k >= 1 and task.heldout:
        tau_k = state.router.calibrate_tau(state.encoder.embed([e.prompt for e in task.heldout]), k)
        log({"tau_k": tau_k, "tau": state.router.tau})

    state.bank.add_atom()
    learn_beta = k >= 1 and not state.composition.identity

    # Phase A: beta_k only, B_k held at 0.
    if learn_beta and not cfg.skip_phase_a:
        beta = state.composition.trainable_row(k)
        schedule = _cyclic_schedule(len(task.train), cfg.phase_a_steps, cfg.batch_size, rng)
        run_phase(state, task, k, [{"params": [beta], "lr": cfg.phase_a_lr, "weight_decay": 0.0}],
                  [lambda s: 1.0], schedule, log, "A")
        state.composition.freeze_row(k)
        if task.heldout:
            log({"phase": "A", "end": True, "heldout_loss": heldout_loss(state, task.name, task.heldout, k),
                 "peak_gb": _cuda_peak_gb()})
    if on_phase_a_end is not None:
        on_phase_a_end(k)

    # Phase B: atom k (and beta_k).
    if not cfg.phase_a_only:
        if task.heldout:
            log({"phase": "B", "step": 0, "heldout_loss": heldout_loss(state, task.name, task.heldout, k)})
        phase_b(state, task, k, k, k if learn_beta else None, log, rng)
        if task.heldout:
            log({"phase": "B", "end": True, "heldout_loss": heldout_loss(state, task.name, task.heldout, k),
                 "peak_gb": _cuda_peak_gb()})

    # Freeze everything belonging to task k.
    state.bank.freeze_atom(k)
    state.composition.freeze_row(k)
    state.model.eval()
    log({"learn_task_sec": round(time.time() - t_start, 1), "beta_mean_row": state.composition.matrix()[:, k, :k]
         .mean(dim=0).tolist()})
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(log_dir) / f"learn_task_{k + 1}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    return {"records": records}
