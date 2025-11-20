from __future__ import annotations
import numpy as np
import os
import math
import logging
import collections
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig
from datasets import load_dataset
from smoothbleu import compute_smooth_bleu

def set_up_logger(log_filepath: str) -> logging.Logger:
    logger = logging.getLogger(os.path.basename(log_filepath) or "t5_trainer")
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
            "CONCODE": "Generate Java code from the following English description: ",
            "CodeTrans": "Translate the following Java code into C#: ",
            "CodeSearchNet": "Summarize the following Ruby code into English.\n\n<language>Ruby</language>\n<code>\n",
            "BFP": "Refactor or improve the following Java code: ",
        }

    def select_subset_ds(self, ds, k=2000, seed=0):
        np.random.seed(seed)
        num_samples = min(k, ds.shape[0])
        idx_total = np.random.choice(np.arange(ds.shape[0]), num_samples, replace=False)
        return ds.select(idx_total)

    def _preprocess_batch(self, examples, task: str, max_length: int = 512):
        if task not in self.task_list:
            raise ValueError(f"Unknown task name: {task}")
        tk = self.tokenizer
        text_col = self.text_key[task]
        label_col = self.label_key[task]
        instr = self.task_instructions[task]
        if task == "CodeSearchNet":
            src_texts = [(instr + str(t) + "\n</code>").strip() for t in examples[text_col]]
        else:
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
    lora_dir_path: str
    model_name: str = "Salesforce/codet5-small"
    batch_size: int = 16
    seq_len: int = 256
    target_seq_len: int = 128
    training_size: int = -1
    val_size: int = -1
    lr: float = 1e-4
    warmup_ratio: float = 0.01
    weight_decay: float = 0
    epochs: int = 3
    num_beams: int = 5
    repetition_penalty: float = 1.2
    generator_early_stopping: bool = True
    device: Optional[str] = None
    output_dir_prefix: str = "outputs"
    r_task: int = 16
    lora_alpha_task: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q", "v", "k")
    bias: str = "none"
    eval_on_all_tasks: bool = True
    log_every_n_steps: int = 25
    lambda_orth: float = 1e-4

class OrthoBank:
    def __init__(self):
        self.bank: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

    @torch.no_grad()
    def add(self, model: torch.nn.Module, adapter_name: str):
        for m in model.modules():
            if hasattr(m, "lora_A") and hasattr(m, "lora_B") and adapter_name in m.lora_A:
                A = m.lora_A[adapter_name].weight  
                B = m.lora_B[adapter_name].weight 
                if A.numel() == 0 or B.numel() == 0: continue
                QA, _ = torch.linalg.qr(A.T, mode="reduced")  
                QB, _ = torch.linalg.qr(B,   mode="reduced")  
                self.bank.setdefault(id(m), []).append((QA.detach().cpu(), QB.detach().cpu()))

    def orth_loss(self, model: torch.nn.Module, adapter_name: str, device: torch.device) -> torch.Tensor:
        loss = torch.zeros((), device=device)
        for m in model.modules():
            if not (hasattr(m, "lora_A") and hasattr(m, "lora_B") and adapter_name in getattr(m, "lora_A", {})):
                continue
            A = m.lora_A[adapter_name].weight
            B = m.lora_B[adapter_name].weight
            if A.numel() == 0 or B.numel() == 0: continue
            if torch.norm(B) < 1e-6:
                continue
            QA_curr, _ = torch.linalg.qr(A.T, mode="reduced")  
            QB_curr, _ = torch.linalg.qr(B,   mode="reduced")  
            for (QA_prev, QB_prev) in self.bank.get(id(m), []):
                QA_prev = QA_prev.to(A.device)
                QB_prev = QB_prev.to(B.device)
                loss = loss + (QA_curr.T @ QA_prev).pow(2).sum() + (QB_curr.T @ QB_prev).pow(2).sum()
        return loss


@torch.no_grad()
def _resolve_lora_scale(m, adapter_name: str):
    scale = getattr(m, "scaling", None)
    if isinstance(scale, dict):
        s = scale.get(adapter_name, None)
        if s is not None:
            if hasattr(s, "item"):
                try: return float(s.item())
                except Exception: pass
            try: return float(s)
            except Exception: pass
    if scale is not None and not isinstance(scale, dict):
        try: return float(scale.item() if hasattr(scale, "item") else scale)
        except Exception: pass
    alpha = getattr(m, "lora_alpha", None)
    r = getattr(m, "r", None)
    if alpha is not None and r is not None:
        try: return float(alpha)/float(r)
        except Exception: pass
    return 1.0

@torch.no_grad()
def merge_task_into_base(model: torch.nn.Module, adapter_name: str):
    for m in model.modules():
        if not (hasattr(m, "lora_A") and hasattr(m, "lora_B") and adapter_name in m.lora_A):
            continue
        A = m.lora_A[adapter_name].weight  
        B = m.lora_B[adapter_name].weight  
        if A.numel() == 0 or B.numel() == 0: continue
        scale = _resolve_lora_scale(m, adapter_name)
        delta = (B @ A) * scale  
        base = getattr(m, 'weight', None)
        if base is None:
            continue
        base.data.add_(delta.to(base.dtype))
        A.zero_(); B.zero_()


class T5ContinualLearner:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.logger = set_up_logger(cfg.log_filepath)
        self.device = torch.device(cfg.device) if cfg.device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = T5ForConditionalGeneration.from_pretrained(cfg.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.bank = OrthoBank() 
        self.ds_helper = T5Dataset(self.tokenizer)
        self.tasks_data: Dict[str, Dict[str, DataLoader]] = self._build_all_dataloaders()
        Tn = len(cfg.task_list)
        self.R_val = [[None for _ in range(Tn)] for _ in range(Tn)]
        self.tasks = cfg.task_list

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

    def compute_bleu(self, refs: List[List[str]], hyps: List[str], max_order=4, smooth=False):
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
                do_sample=(self.cfg.num_beams == 1),
                repetition_penalty=self.cfg.repetition_penalty,
                early_stopping=self.cfg.generator_early_stopping,
            )
            preds.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
            golds.extend(self._decode_labels(batch["labels"]))
        em = sum(self.compute_exact_match(p, g) for p, g in zip(preds, golds)) / max(1, len(golds))
        if task == "CodeSearchNet":
            bleu = compute_smooth_bleu([[g] for g in golds], preds, smooth=1)
        else:
            bleu = self.compute_bleu([[g] for g in golds], preds)
        return {"EM": em, "BLEU": bleu, "N": float(len(golds))}

    def _make_task_lora_cfg(self) -> LoraConfig:
        return LoraConfig(
            r=self.cfg.r_task,
            lora_alpha=self.cfg.lora_alpha_task,
            target_modules=list(self.cfg.target_modules),
            lora_dropout=self.cfg.lora_dropout,
            bias=self.cfg.bias,
            task_type="SEQ_2_SEQ_LM",
            inference_mode=False,
        )

    def _ensure_task_adapter(self, task: str) -> str:
        name = f"task_{task}"
        lcfg = self._make_task_lora_cfg()
        try:
            self.model.add_adapter(lcfg, adapter_name=name)
            self.logger.info(f"[LoRA] Added adapter: {name}")
        except Exception:
            pass
        try:
            self.model.set_adapter(name)
            self.model.enable_adapters()
        except Exception:
            pass
        return name

    def train_one_task(self, task: str):
        adapter_name = self._ensure_task_adapter(task)
        self.model.train()
        train_dl = self.tasks_data[task]["train"]
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        steps = max(1, len(train_dl) * self.cfg.epochs)
        warm = int(steps * self.cfg.warmup_ratio)
        sch = get_linear_schedule_with_warmup(opt, warm, steps)
        K = max(1, int(self.cfg.log_every_n_steps))
        last_loss = 0.0
        for ep in range(self.cfg.epochs):
            for i, batch in enumerate(train_dl, start=1):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                loss_ce = out.loss
                loss_orth = self.bank.orth_loss(self.model, adapter_name, self.device)
                loss = loss_ce + self.cfg.lambda_orth * loss_orth
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
                last_loss = float(loss_ce.detach().cpu())
                if (i % K) == 0:
                    self.logger.info(f"[TRAIN] {task} | epoch {ep+1} | step {i}/{len(train_dl)} | loss_ce={last_loss:.4f} | loss_orth={float(loss_orth.detach().cpu()):.6f}")
        self.bank.add(self.model, adapter_name)
        self.logger.info(f"[LoRA] Merged '{adapter_name}' into base and cached (Q_A,Q_B).")
        merge_task_into_base(self.model, adapter_name)
        self.logger.info(f"[LoRA] Merged '{adapter_name}' into base weights.")

    def compute_forgetting_from_R(self):
        Tn = len(self.tasks)
        f = []
        for j in range(Tn):
            hist = [x for x in (self.R_val[j][: Tn - 1]) if x is not None]
            if not hist:
                f.append(0.0)
                continue
            best_prev = max(hist)
            final = self.R_val[j][Tn - 1]
            if final is None:
                f.append(0.0)
            else:
                f.append(max(0.0, best_prev - final))
        avg = sum(f[:-1]) / max(1, (Tn - 1))
        return f, avg

    def train_continual(self):
        Tn = len(self.tasks)
        for i, task in enumerate(self.tasks):
            self.logger.info(f"===== Train on {task} (task {i+1}/{Tn}) =====")
            self.train_one_task(task)
            for j, eval_task in enumerate(self.tasks):
                m = self.evaluate_split(eval_task, split="val")
                self.R_val[j][i] = m["BLEU"]
                self.logger.info(f"[VAL] after task {task} -> on {eval_task}: BLEU={m['BLEU']:.4f} EM={m['EM']:.4f} N={int(m['N'])}")
        per_task_forget, avg_forget = self.compute_forgetting_from_R()
        test_bleu = {}
        for task in self.tasks:
            m = self.evaluate_split(task, split="test")
            test_bleu[task] = m["BLEU"]
            self.logger.info(f"[TEST] {task}: BLEU={m['BLEU']:.4f} EM={m['EM']:.4f} N={int(m['N'])}")
        final_val_bleus = []
        for j in range(Tn):
            val_last = self.R_val[j][-1]
            final_val_bleus.append(0.0 if val_last is None else float(val_last))
        avg_val_bleu = (sum(final_val_bleus) / max(1, len(final_val_bleus))) if final_val_bleus else 0.0
        avg_test_bleu = (sum(test_bleu[t] for t in self.tasks) / max(1, len(self.tasks))) if self.tasks else 0.0
        self.logger.info("==== SUMMARY (dev/test) ====")
        for task_name, val_bleu, f in zip(self.tasks, final_val_bleus, per_task_forget):
            self.logger.info(f"Task {task_name} | BLEU<Val_final>={val_bleu:.4f} | Forgetting(dev)={f:.4f} | BLEU<Test>={test_bleu[task_name]:.4f}")
        self.logger.info(f"Average Forgetting(dev): {avg_forget:.4f}")
        self.logger.info(f"Average BLEU<Val_final>: {avg_val_bleu:.4f}")
        self.logger.info(f"Average BLEU<Test>: {avg_test_bleu:.4f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--task_list", nargs="+", required=True)
    p.add_argument("--log_filepath", type=str, required=True)
    p.add_argument("--lora_dir_path", type=str, default="lora")
    p.add_argument("--model_name", type=str, default="Salesforce/codet5-small")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--target_seq_len", type=int, default=128)
    p.add_argument("--training_size", type=int, default=-1)
    p.add_argument("--val_size", type=int, default=-1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_beams", type=int, default=5)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])  # type: ignore
    p.add_argument("--log_every_n_steps", type=int, default=25)
    p.add_argument("--r_task", type=int, default=16)
    p.add_argument("--lora_alpha_task", type=int, default=32)
    p.add_argument("--lambda_orth", type=float, default=1e-4)
    args = p.parse_args()

    cfg = TrainConfig(
        task_list=args.task_list,
        log_filepath=args.log_filepath,
        lora_dir_path=args.lora_dir_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        target_seq_len=args.target_seq_len,
        training_size=args.training_size,
        val_size=args.val_size,
        lr=args.lr,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        device=args.device,
        log_every_n_steps=args.log_every_n_steps,
        r_task=args.r_task,
        lora_alpha_task=args.lora_alpha_task,
        lambda_orth=args.lambda_orth,
    )

    learner = T5ContinualLearner(cfg)
    learner.train_continual()
