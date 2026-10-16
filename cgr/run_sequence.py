"""RunSequence (Algorithm 4): learn tasks in order; after each, generate on the test prompts of every seen task.

  python -m cgr.run_sequence --benchmark codetask --ablation A3 --output_dir runs/cgr_codetask_A3
  python -m cgr.run_sequence --benchmark codetask --method seqlora --output_dir runs/seqlora   # forgetting baseline
  python -m cgr.run_sequence --benchmark codetask --method base --output_dir runs/base         # FWT reference
  python -m cgr.eval.score --run_dir runs/cgr_codetask_A3                                      # scoring, afterwards

Methods other than "cgr" are baselines/references and do not obey the CGR invariants by design (seqlora overwrites
its single adapter on every task).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from cgr.config import CGRConfig, apply_ablation, benchmark_defaults
from cgr.data import TaskRegistry, TaskSource, hf_source
from cgr.eval.generate import (base_predict_fn, cgr_predict_fn, generate_tasks, oracle_predict_fn,
                               seqlora_predict_fn)
from cgr.state import CGRState
from cgr.train import learn_task, phase_b


@contextmanager
def preserved_rng():
    """Evaluation inside training (Phase-A snapshot) must not change the training RNG stream."""
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def _evaluate_step(k: int, st: CGRState, view, out: Path, eval_mode: str, method: str) -> None:
    """After learning tasks 0..k: a_{k+1, j} for j <= k ("full"), or only j = k and the final row ("diag_last")."""
    K = len(st.cfg.tasks)
    if eval_mode == "none" or (eval_mode == "diag_last" and k < K - 1):
        ids = [] if eval_mode == "none" else [k]
    else:
        ids = list(range(k + 1))
    if method == "seqlora":
        fn = seqlora_predict_fn(st)
        generate_tasks(f"step_{k + 1}", k, ids, lambda j: fn, view, st, out)
    else:
        fn = cgr_predict_fn(st)
        generate_tasks(f"step_{k + 1}", k, ids, lambda j: fn, view, st, out)
        if st.cfg.ablation == "A8":
            generate_tasks(f"oracle_step_{k + 1}", k, ids, lambda j: oracle_predict_fn(st, j), view, st, out)


def run(cfg: CGRConfig, model, tokenizer, source: TaskSource, out: Path, eval_mode: str = "full",
        eval_phase_a: bool = False, resume: bool = False) -> CGRState:
    out.mkdir(parents=True, exist_ok=True)
    state_dir = out / "state"
    if resume and (state_dir / "manifest.json").exists() and cfg.method == "cgr":
        st = CGRState.load(state_dir, cfg, model, tokenizer)
    else:
        st = CGRState.create(cfg, model, tokenizer)
    cfg.save(out / "config.json")
    (out / "progress.json").write_text(json.dumps({"tasks": cfg.tasks, "benchmark": cfg.benchmark,
                                                    "method": cfg.method, "ablation": cfg.ablation,
                                                    "seed": cfg.seed, "n_samples": cfg.n_samples}, indent=2))
    registry = TaskRegistry(cfg.tasks, source, cfg.heldout_frac, cfg.heldout_max, cfg.data_seed)

    if cfg.method == "base":
        registry.resume(len(cfg.tasks))  # no training; evaluation-only access to every test split
        fn = base_predict_fn(st)
        generate_tasks("base", 0, range(len(cfg.tasks)), lambda j: fn, registry.evaluation_view(), st, out)
        return st

    start = st.K if cfg.method == "cgr" else 0
    registry.resume(start)
    if start > 0:  # finish an evaluation that was interrupted after the last checkpoint (existing files are kept)
        _evaluate_step(start - 1, st, registry.evaluation_view(), out, eval_mode, cfg.method)
    timing = json.loads((out / "timing.json").read_text()) if resume and (out / "timing.json").exists() else {}
    for k in range(start, len(cfg.tasks)):
        registry.begin_task(k)
        task = registry.training_data(k)
        t0 = time.time()
        if cfg.method == "cgr":
            hook = None
            if eval_phase_a:
                def hook(kk, _view=registry.evaluation_view()):
                    if kk == 0 or st.composition.identity or cfg.skip_phase_a:
                        return  # no Phase A was run for this task
                    with preserved_rng():
                        fn = cgr_predict_fn(st)
                        generate_tasks("phase_a", kk, [kk], lambda j: fn, _view, st, out)
            learn_task(k, task, st, cfg, log_dir=str(out / "logs"), on_phase_a_end=hook)
        else:  # seqlora: one shared atom, retrained on every task with the Phase B recipe
            if k == 0:
                st.bank.add_atom()
                st.composition.add_row()
            log = lambda rec, kk=k: print(json.dumps({"task": kk + 1, **rec}), flush=True)
            torch.manual_seed(cfg.seed + k)
            phase_b(st, task, 0, 0, None, log, random.Random(cfg.seed + k))
            st.bank.freeze_atom(0)
        registry.end_task(k)
        del task  # Invariant I4
        if cfg.method == "cgr":
            st.save(state_dir)
            (out / "logs").mkdir(exist_ok=True)
            (out / "logs" / f"beta_mean_after_{k + 1}.json").write_text(json.dumps(st.composition.beta_mean()))
            (out / "logs" / f"tau_after_{k + 1}.json").write_text(json.dumps({"tau_k": st.router.taus,
                                                                             "tau": st.router.tau}))
        t1 = time.time()
        _evaluate_step(k, st, registry.evaluation_view(), out, eval_mode, cfg.method)
        timing[cfg.tasks[k]] = {"train_sec": round(t1 - t0, 1), "eval_sec": round(time.time() - t1, 1)}
        (out / "timing.json").write_text(json.dumps(timing, indent=2))
    return st


def build_config(args) -> CGRConfig:
    cfg = apply_ablation(benchmark_defaults(args.benchmark), args.ablation, args.shrinkage)
    cfg.method = args.method
    cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.tasks:
        cfg.tasks = args.tasks.split(",")
    for kv in args.override:
        key, val = kv.split("=", 1)
        if not hasattr(cfg, key):
            raise ValueError(f"unknown config key {key}")
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass
        setattr(cfg, key, val)
    cfg.__post_init__()
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["codetask", "executable"], default="codetask")
    ap.add_argument("--method", choices=["cgr", "seqlora", "base"], default="cgr")
    ap.add_argument("--ablation", default="A3", help="A0..A9 (Sec. 8.1); ignored by seqlora/base")
    ap.add_argument("--shrinkage", type=float, default=None, help="lambda override (A4 sweep)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tasks", default=None, help="comma-separated override of the task order")
    ap.add_argument("--override", nargs="*", default=[], help="CGRConfig overrides key=value (JSON values)")
    ap.add_argument("--eval_mode", choices=["full", "diag_last", "none"], default="full",
                    help="full: all seen tasks after every task (FM/BWT matrix); diag_last: only the new task, "
                         "plus all tasks after the last one")
    ap.add_argument("--eval_phase_a", action="store_true", help="also score task k right after Phase A (pure reuse)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--prefetch_only", action="store_true",
                    help="download the backbone and every task split into the HF cache, then exit")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    from utils.utils import load_hf_tokenizer, set_random_seed

    cfg = build_config(args)
    if args.prefetch_only:
        load_hf_tokenizer(cfg.model_name_or_path, fast_tokenizer=True)
        AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path)
        src = hf_source(cfg.benchmark, cfg.data_seed, cfg.max_train_examples, cfg.max_test_examples)
        for t in cfg.tasks:
            print(f"[prefetch] {t}: train={len(src(t, 'train'))} test={len(src(t, 'test'))}", flush=True)
        return
    set_random_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} method={cfg.method} ablation={cfg.ablation} lambda={cfg.shrinkage} "
          f"seed={cfg.seed} tasks={cfg.tasks}", flush=True)
    tokenizer = load_hf_tokenizer(cfg.model_name_or_path, fast_tokenizer=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path,
                                                 torch_dtype=getattr(torch, cfg.backbone_dtype)).to(device)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    source = hf_source(cfg.benchmark, cfg.data_seed, cfg.max_train_examples, cfg.max_test_examples)
    run(cfg, model, tokenizer, source, Path(cfg.output_dir), args.eval_mode, args.eval_phase_a, args.resume)


if __name__ == "__main__":
    main()
