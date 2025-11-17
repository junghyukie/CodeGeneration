from __future__ import annotations
import os
import math
import logging
import collections
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from smoothbleu import compute_smooth_bleu
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from datasets import load_dataset

def set_up_logger(log_filepath: str) -> logging.Logger:
    logger = logging.getLogger(os.path.basename(log_filepath) or "t5_trainer_fullft_ewc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    os.makedirs(os.path.dirname(log_filepath) or ".", exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_filepath, encoding="utf-8")
    ch = logging.StreamHandler()
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

class T5Dataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.task_list = ["CodeTrans", "CodeSearchNet", "BFP", "CONCODE"]
        self.text_key = {"CONCODE": "nl", "CodeTrans": "java", "CodeSearchNet": "code", "BFP": "buggy"}
        self.label_key = {"CONCODE": "code", "CodeTrans": "cs", "CodeSearchNet": "docstring", "BFP": "fixed"}
        self.task_instructions = {
            "CONCODE": "generate java: ",
            "CodeTrans": "translate java-to-csharp: ",
            "CodeSearchNet": "summarize ruby: ",
            "BFP": "refactor java: ",
        }

    def select_subset_ds(self, ds, k=2000, seed=0):
        if k == -1:
            return ds
        rng = np.random.default_rng(seed)
        num_samples = min(k, ds.shape[0])
        idx_total = rng.choice(ds.shape[0], num_samples, replace=False)
        return ds.select(idx_total)

    def _preprocess_batch(self, examples, task: str, max_length: int = 512):
        if task not in self.task_list:
            raise ValueError(f"Unknown task name: {task}")
        tk = self.tokenizer
        text_col = self.text_key[task]
        label_col = self.label_key[task]
        instr = self.task_instructions[task]
        src_texts = [(instr + str(t)).strip() for t in examples[text_col]]
        tgt_texts = [str(t) for t in examples[label_col]]
        src = tk(src_texts, padding="max_length", truncation=True, max_length=max_length)
        with tk.as_target_tokenizer():
            tgt = tk(tgt_texts, padding="max_length", truncation=True, max_length=max_length)
        labels = []
        for ids, mask in zip(tgt["input_ids"], tgt["attention_mask"]):
            labels.append([tok if m == 1 else -100 for tok, m in zip(ids, mask)])
        return {"input_ids": src["input_ids"], "attention_mask": src["attention_mask"], "labels": labels}

    def get_final_ds(self, task, split, batch_size, k=-1, seed=0, return_test=False, max_length=512):
        if task == "CONCODE":
            dataset = load_dataset("AhmedSSoliman/CodeXGLUE-CONCODE", split=split)
        elif task == "CodeTrans":
            dataset = load_dataset("CM/codexglue_codetrans", split=split)
        elif task == "CodeSearchNet":
            dataset = load_dataset("semeru/code-text-ruby", split=split)
        elif task == "BFP":
            dataset = load_dataset("ayeshgk/code_x_glue_cc_code_refinement_annotated", split=split)
        else:
            raise ValueError(f"Unknown task: {task}")
        if k != -1:
            dataset = self.select_subset_ds(dataset, k=k, seed=seed)
        else:
            dataset = dataset.shuffle(seed=seed)
        map_fn = lambda batch: self._preprocess_batch(batch, task, max_length=max_length)
        if not return_test:
            enc = dataset.map(map_fn, batched=True, remove_columns=dataset.column_names)
            enc.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            return DataLoader(enc, batch_size=batch_size, shuffle=True)
        else:
            N = len(dataset)
            ds_val = dataset.select(range(0, N // 2))
            ds_test = dataset.select(range(N // 2, N))
            outs = []
            for ds in (ds_val, ds_test):
                enc = ds.map(map_fn, batched=True, remove_columns=ds.column_names)
                enc.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
                outs.append(DataLoader(enc, batch_size=batch_size, shuffle=False))
            return outs[0], outs[1]

@dataclass
class TrainConfig:
    task_list: List[str]
    log_filepath: str
    output_dir_prefix: str
    model_name: str = "Salesforce/codet5-small"
    batch_size: int = 8
    seq_len: int = 256
    target_seq_len: int = 128
    training_size: int = -1
    val_size: int = -1
    lr: float = 5e-5
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    epochs: int = 3
    num_beams: int = 5
    length_penalty: float = 1.0
    min_new_tokens: int = 0
    device: Optional[str] = None
    log_every_n_steps: int = 25
    eval_on_all_tasks: bool = False  
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = True
    gradient_checkpointing: bool = False
    seed: int = 42
    ewc: bool = True
    ewc_lambda: float = 0.4
    fisher_samples: int = 256
    fisher_batch_size: int = 8

class T5ContinualLearner:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.logger = set_up_logger(cfg.log_filepath)
        set_seed(cfg.seed)
        self.device = torch.device(cfg.device) if cfg.device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = T5ForConditionalGeneration.from_pretrained(cfg.model_name)
        if cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        os.makedirs(cfg.output_dir_prefix, exist_ok=True)
        self.ds_helper = T5Dataset(self.tokenizer)
        self.tasks_data: Dict[str, Dict[str, DataLoader]] = self._build_all_dataloaders()
        Tn = len(cfg.task_list)
        self.R_val = [[None for _ in range(Tn)] for _ in range(Tn)]
        self.tasks = cfg.task_list
        try:
            from torch.amp import GradScaler
            self.scaler = GradScaler('cuda', enabled=(cfg.fp16 and self.device.type == "cuda"))
        except ImportError:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler(enabled=(cfg.fp16 and self.device.type == "cuda"))
        self.ewc_snapshots: List[Dict[str, Dict[str, torch.Tensor]]] = []

    def _build_all_dataloaders(self):
        data = {}
        for task in self.cfg.task_list:
            train_dl = self.ds_helper.get_final_ds(task, "train", self.cfg.batch_size, k=self.cfg.training_size, max_length=self.cfg.seq_len)
            val_dl, test_dl = self.ds_helper.get_final_ds(task, "test", self.cfg.batch_size, k=self.cfg.val_size, return_test=True, max_length=self.cfg.seq_len)
            data[task] = {"train": train_dl, "val": val_dl, "test": test_dl}
        return data

    @staticmethod
    def _normalize_text(s: str) -> str:
        import re, string
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
        def white_space_fix(text):
            return " ".join(text.split())
        def remove_punc(text):
            return "".join(ch for ch in text if ch not in set(string.punctuation))
        s = s.lower().replace("<pad>", "").replace("</s>", "")
        return white_space_fix(remove_articles(remove_punc(s)))

    def compute_exact_match(self, pred: str, gold: str) -> int:
        return int(self._normalize_text(pred) == self._normalize_text(gold))

    def _get_ngrams(self, toks: List[str], max_order: int):
        c = collections.Counter()
        for o in range(1, max_order + 1):
            for i in range(0, len(toks) - o + 1):
                c[tuple(toks[i : i + o])] += 1
        return c

    def compute_bleu(self, refs: List[List[str]], hyps: List[str], max_order=4, smooth=True):
        matches = [0] * max_order
        possibles = [0] * max_order
        ref_len = 0
        hyp_len = 0
        for rlist, hyp in zip(refs, hyps):
            r_tokens_list = [r.split() for r in rlist]
            h = hyp.split()
            ref_len += min(len(r) for r in r_tokens_list)
            hyp_len += len(h)
            merged = collections.Counter()
            for r in r_tokens_list:
                merged |= self._get_ngrams(r, max_order)
            h_counts = self._get_ngrams(h, max_order)
            overlap = h_counts & merged
            for ng in overlap:
                matches[len(ng) - 1] += overlap[ng]
            for o in range(1, max_order + 1):
                p = len(h) - o + 1
                if p > 0:
                    possibles[o - 1] += p
        prec = [0] * max_order
        for i in range(max_order):
            if smooth:
                prec[i] = (matches[i] + 1.0) / (possibles[i] + 1.0)
            else:
                prec[i] = (matches[i] / possibles[i]) if possibles[i] > 0 else 0.0
        geo = math.exp(sum((1.0 / max_order) * math.log(p) for p in prec)) if min(prec) > 0 else 0.0
        ratio = float(hyp_len) / max(1, ref_len)
        bp = 1.0 if ratio > 1.0 else math.exp(1 - 1.0 / max(ratio, 1e-9))
        return geo * bp

    def _decode_labels(self, labels_tensor: torch.Tensor) -> List[str]:
        arr = labels_tensor.clone().cpu().numpy().tolist()
        for seq in arr:
            for i, tok in enumerate(seq):
                if tok == -100:
                    seq[i] = self.tokenizer.pad_token_id
        return self.tokenizer.batch_decode(torch.tensor(arr), skip_special_tokens=True)

    @torch.no_grad()
    def evaluate_split(self, task: str, split: str) -> Dict[str, float]:
        self.model.eval()
        dl = self.tasks_data[task][split]
        preds, golds = [], []
        for batch in dl:
            inp = batch["input_ids"].to(self.device)
            attn = batch["attention_mask"].to(self.device)
            gen = self.model.generate(
                input_ids=inp,
                attention_mask=attn,
                max_new_tokens=self.cfg.target_seq_len,
                num_beams=self.cfg.num_beams,
                do_sample=False,
                length_penalty=self.cfg.length_penalty,
                min_new_tokens=self.cfg.min_new_tokens,
                early_stopping=True,
            )
            preds.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
            golds.extend(self._decode_labels(batch["labels"]))
        em = sum(self.compute_exact_match(p, g) for p, g in zip(preds, golds)) / max(1, len(golds))
        if task == "CodeSearchNet":
            bleu = compute_smooth_bleu([[g] for g in golds], preds, smooth=1)
        else:
            bleu = self.compute_bleu([[g] for g in golds], preds)
        return {"EM": em, "BLEU": bleu, "N": float(len(golds))}

    def _snapshot_params(self) -> Dict[str, torch.Tensor]:
        snap = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                snap[n] = p.detach().clone().to(self.device)
        return snap

    def _zero_like_params(self) -> Dict[str, torch.Tensor]:
        zeros = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                zeros[n] = torch.zeros_like(p, device=self.device)
        return zeros

    @torch.no_grad()
    def _normalize_fisher_(self, fisher: Dict[str, torch.Tensor], count: int):
        for n in fisher:
            fisher[n].div_(max(1, count))

    def _accumulate_square_grads(self, fisher: Dict[str, torch.Tensor]):
        for n, p in self.model.named_parameters():
            if p.grad is None or n not in fisher:
                continue
            fisher[n] += (p.grad.detach() ** 2)

    def _compute_fisher(self, task: str) -> Dict[str, torch.Tensor]:
        self.model.eval()
        train_dl = self.tasks_data[task]["train"]
        fisher = self._zero_like_params()
        seen = 0
        max_samples = self.cfg.fisher_samples
        bs = self.cfg.fisher_batch_size
        it = iter(train_dl)
        while seen < max_samples:
            try:
                batch = next(it)
            except StopIteration:
                break
            for k in batch:
                if isinstance(batch[k], torch.Tensor) and batch[k].size(0) > bs:
                    batch[k] = batch[k][:bs]
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.model.zero_grad(set_to_none=True)
            out = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])  # NLL
            loss = out.loss
            loss.backward()
            self._accumulate_square_grads(fisher)
            seen += batch["input_ids"].size(0)
        self._normalize_fisher_(fisher, seen)
        return fisher

    def _ewc_penalty(self) -> torch.Tensor:
        if not self.ewc_snapshots:
            return torch.tensor(0.0, device=self.device)
        penalty = torch.tensor(0.0, device=self.device)
        for snap in self.ewc_snapshots:
            theta_star = snap["params"]
            fisher = snap["fisher"]
            for n, p in self.model.named_parameters():
                if n in theta_star:
                    diff = p - theta_star[n]
                    penalty = penalty + (fisher[n] * (diff ** 2)).sum()
        return (self.cfg.ewc_lambda / 2.0) * penalty

    def train_one_task(self, task: str):
        self.model.train()
        train_dl = self.tasks_data[task]["train"]
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        steps_per_epoch = len(train_dl)
        total_steps = max(1, steps_per_epoch * self.cfg.epochs)
        warm = int(total_steps * self.cfg.warmup_ratio)
        sch = get_linear_schedule_with_warmup(opt, warm, total_steps)

        K = max(1, int(self.cfg.log_every_n_steps))
        accum = max(1, int(self.cfg.grad_accum_steps))
        for ep in range(self.cfg.epochs):
            opt.zero_grad(set_to_none=True)
            running = 0.0
            for i, batch in enumerate(train_dl, start=1):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # Use compatible autocast for both PyTorch 2.0+ and older versions
                try:
                    from torch.amp import autocast
                    amp_context = autocast('cuda', enabled=(self.cfg.fp16 and self.device.type == "cuda"))
                except ImportError:
                    from torch.cuda.amp import autocast
                    amp_context = autocast(enabled=(self.cfg.fp16 and self.device.type == "cuda"))
                
                with amp_context:
                    out = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])  # CE loss
                    ce_loss = out.loss
                    if self.cfg.ewc and self.ewc_snapshots:
                        reg = self._ewc_penalty()
                        loss = ce_loss / accum
                        if self.cfg.ewc:
                            loss = loss + self._ewc_penalty()
                    else:
                        reg = torch.tensor(0.0, device=self.device)
                        loss = ce_loss / accum
                self.scaler.scale(loss).backward()

                if (i % accum) == 0:
                    self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(opt)
                    self.scaler.update()
                    sch.step()
                    opt.zero_grad(set_to_none=True)
                running += ce_loss.detach().item()

                if (i % K) == 0:
                    if self.cfg.ewc and self.ewc_snapshots:
                        self.logger.info(f"[TRAIN] {task} | epoch {ep+1}/{self.cfg.epochs} | step {i}/{steps_per_epoch} | loss={ce_loss.item():.4f} | ewc={reg.item():.4f}")
                    else:
                        self.logger.info(f"[TRAIN] {task} | epoch {ep+1}/{self.cfg.epochs} | step {i}/{steps_per_epoch} | loss={ce_loss.item():.4f}")

        save_dir = os.path.join(self.cfg.output_dir_prefix, f"fullft_{task}")
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        if self.cfg.ewc:
            self.logger.info(f"[EWC] Computing Fisher for task {task} ...")
            fisher = self._compute_fisher(task)
            theta_star = self._snapshot_params()
            self.ewc_snapshots.append({"params": theta_star, "fisher": fisher})
            self.logger.info(f"[EWC] Stored snapshot #{len(self.ewc_snapshots)} for {task}")

    def compute_forgetting_from_R(self):
        T = len(self.tasks)
        f = []
        for j in range(T):
            hist = [x for x in (self.R_val[j][:T-1]) if x is not None]
            if not hist:
                f.append(0.0)
                continue
            best_prev = max(hist)
            final = self.R_val[j][T-1]
            if final is None:
                f.append(0.0)
            else:
                f.append(max(0.0, best_prev - final))
        avg = sum(f[:-1]) / max(1, (T - 1))
        return f, avg

    def train_continual(self):
        T = len(self.tasks)
        for i, task in enumerate(self.tasks):
            self.logger.info(f"===== Train on {task} (task {i+1}/{T}) =====")
            self.train_one_task(task)
            for j, eval_task in enumerate(self.tasks):
                m = self.evaluate_split(eval_task, split="val")
                self.R_val[j][i] = m["BLEU"]
                self.logger.info(f"[VAL] after task {task} -> on {eval_task}: BLEU={m['BLEU']:.4f} EM={m['EM']:.4f} N={int(m['N'])}")

        per_task_forget, avg_forget = self.compute_forgetting_from_R()
        for name, f in zip(self.tasks, per_task_forget):
            self.logger.info(f"Forgetting(dev) {name}: {f:.4f}")
        self.logger.info(f"Average Forgetting(dev): {avg_forget:.4f}")

        test_bleu = {}
        for task in self.tasks:
            m = self.evaluate_split(task, split="test")
            test_bleu[task] = m["BLEU"]
            self.logger.info(f"[TEST] {task}: BLEU={m['BLEU']:.4f} EM={m['EM']:.4f} N={int(m['N'])}")

        final_val_bleus = []
        for j in range(T):
            val_last = self.R_val[j][-1]
            final_val_bleus.append(0.0 if val_last is None else float(val_last))
        avg_val_bleu = (sum(final_val_bleus) / max(1, len(final_val_bleus))) if final_val_bleus else 0.0
        avg_test_bleu = (sum(test_bleu[t] for t in self.tasks) / max(1, len(self.tasks))) if self.tasks else 0.0

        self.logger.info("==== SUMMARY (dev/test) ====")
        for task_name, val_bleu, f in zip(self.tasks, final_val_bleus, per_task_forget):
            self.logger.info(f"Task {task_name} | BLEU<Val_final>={val_bleu:.4f} | Forgetting(dev)={f:.4f} | BLEU<Test>={avg_test_bleu:.4f}")
        self.logger.info(f"Average Forgetting(dev): {avg_forget:.4f}")
        self.logger.info(f"Average BLEU<Val_final>: {avg_val_bleu:.4f}")
        self.logger.info(f"Average BLEU<Test>: {avg_test_bleu:.4f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--task_list", nargs="+", required=True)
    p.add_argument("--log_filepath", type=str, required=True)
    p.add_argument("--output_dir_prefix", type=str, default="outputs_fullft_ewc")
    p.add_argument("--model_name", type=str, default="Salesforce/codet5-small")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--target_seq_len", type=int, default=128)
    p.add_argument("--training_size", type=int, default=-1)
    p.add_argument("--val_size", type=int, default=-1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num_beams", type=int, default=5)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--min_new_tokens", type=int, default=0)
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])  # type: ignore
    p.add_argument("--log_every_n_steps", type=int, default=25)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ewc", action="store_true")
    p.add_argument("--ewc_lambda", type=float, default=0.4)
    p.add_argument("--fisher_samples", type=int, default=256)
    p.add_argument("--fisher_batch_size", type=int, default=8)
    args = p.parse_args()

    cfg = TrainConfig(
        task_list=args.task_list,
        log_filepath=args.log_filepath,
        output_dir_prefix=args.output_dir_prefix,
        model_name=args.model_name,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        target_seq_len=args.target_seq_len,
        training_size=args.training_size,
        val_size=args.val_size,
        lr=args.lr,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        min_new_tokens=args.min_new_tokens,
        device=args.device,
        log_every_n_steps=args.log_every_n_steps,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        fp16=bool(args.fp16),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        seed=args.seed,
        ewc=bool(args.ewc),
        ewc_lambda=args.ewc_lambda,
        fisher_samples=args.fisher_samples,
        fisher_batch_size=args.fisher_batch_size,
    )

    learner = T5ContinualLearner(cfg)
    learner.train_continual()
