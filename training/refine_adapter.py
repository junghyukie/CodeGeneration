#!/usr/bin/env python
"""
refine_adapter.py — fine-tune a per-task LoRA adapter using failed-execution feedback.

For each sample in a calibration execution-output file (produced by
infer_calibration_split.sh + execution evaluation):
  prompt : instruction + first failed prediction + its traceback (trimmed to fit)
  answer : first passing prediction, or ground-truth solution

Template applied to the 'prompt' field (DataCollator prepends "input: " automatically):

    Fix the {language} code below. It fails with the given error.

    ### Problem
    {instruction}

    ### Buggy Code
    ```{language}
    {buggy_code}
    ```

    ### Error
    {traceback_tail}

    ### Fixed Code

Usage:
    bash scripts/refine_adapter.sh
    LANGUAGE=python bash scripts/refine_adapter.sh
"""

import sys
sys.dont_write_bytecode = True

import argparse
import datetime
import json
import math
import os

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    get_constant_schedule_with_warmup,
    SchedulerType,
)
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from peft import PeftModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_collator import DataCollator
from utils.utils import (
    get_all_reduce_mean,
    get_optimizer_grouped_parameters,
    load_hf_tokenizer,
    print_rank_0,
    set_random_seed,
    to_device,
)
from utils.ds_utils import get_train_ds_config
from utils.model.model_utils import create_hf_model

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    _HF_HUB = True
except ImportError:
    _HF_HUB = False


# ---------------------------------------------------------------------------
# Logging (mirrors main_anamoe.py)
# ---------------------------------------------------------------------------

class TeeLogger:
    def __init__(self, filepath):
        self._terminal = sys.__stdout__
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self._log = open(filepath, "a", buffering=1)
        self._log.write(f"\n{'='*60}\n")
        self._log.write(f"Refinement started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._log.write(f"{'='*60}\n")
        self._log.flush()

    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        self._log.close()

    def __del__(self):
        try:
            self._log.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_results_file(results_dir: str, language: str,
                        source: str = "local", repo_type: str = "dataset") -> list:
    """Return the predictions list from calibration_{language}.json."""
    filename = f"calibration_{language}.json"
    if source == "hf_hub":
        if not _HF_HUB:
            raise ImportError("pip install huggingface-hub")
        local_path = hf_hub_download(repo_id=results_dir, filename=filename, repo_type=repo_type)
    else:
        local_path = os.path.join(results_dir, filename)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Calibration file not found: {local_path}")
    with open(local_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("predictions", data) if isinstance(data, dict) else data


def extract_repair_samples(predictions: list) -> list:
    """Extract (instruction, buggy_code, traceback, fixed_code) dicts.

    Skipping rules:
    - No failed prediction with a non-empty traceback → skip
    - No label (no passing prediction AND no ground-truth) → skip
    """
    samples, skipped_no_fail, skipped_no_label = [], 0, 0
    for pred in predictions:
        instruction = pred.get("source", "")
        passed = pred.get("passed", [])
        stderr = pred.get("stderr", [])
        candidates = pred.get("prediction", [])
        gt = pred.get("ground-truth") or ""

        # First failed prediction that has a real traceback
        buggy_code = traceback_text = None
        for p, e, code in zip(passed, stderr, candidates):
            if p == 0 and isinstance(e, str) and e.strip():
                buggy_code, traceback_text = code, e.strip()
                break
        if buggy_code is None:
            skipped_no_fail += 1
            continue

        # Label: first passing prediction, else ground-truth
        fixed_code = None
        for p, code in zip(passed, candidates):
            if p == 1:
                fixed_code = code
                break
        if fixed_code is None and gt:
            fixed_code = gt
        if not fixed_code:
            skipped_no_label += 1
            continue

        samples.append(dict(
            instruction=instruction,
            buggy_code=buggy_code,
            traceback=traceback_text,
            fixed_code=fixed_code,
        ))

    print(
        f"[repair] extracted {len(samples)} pairs "
        f"(skipped: {skipped_no_fail} no-fail-traceback, {skipped_no_label} no-label)"
    )
    return samples


# ---------------------------------------------------------------------------
# Prompt building + traceback trimming
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = (
    "Fix the {language} code below. It fails with the given error.\n\n"
    "### Problem\n{instruction}\n\n"
    "### Buggy Code\n```{language}\n{buggy_code}\n```\n\n"
    "### Error\n{traceback}\n\n"
    "### Fixed Code"
)


def _build_prompt(language: str, instruction: str, buggy_code: str, traceback: str) -> str:
    return _PROMPT_TEMPLATE.format(
        language=language,
        instruction=instruction,
        buggy_code=buggy_code,
        traceback=traceback,
    )


def _trim_traceback(
    traceback: str,
    tokenizer,
    language: str,
    instruction: str,
    buggy_code: str,
    max_prompt_len: int,
) -> str:
    """Trim traceback (keeping its END) so prompt + DataCollator overhead ≤ max_prompt_len.

    DataCollator prepends 'input: ' and appends '\\noutput: ' (~10 tokens overhead),
    so we leave a 12-token safety buffer.
    """
    prompt_no_tb = _build_prompt(language, instruction, buggy_code, "")
    prefix_ids = tokenizer(prompt_no_tb, add_special_tokens=False).input_ids
    available = max_prompt_len - len(prefix_ids) - 12  # DataCollator overhead + buffer

    if available <= 0:
        return ""  # instruction + code already fill the context

    tb_ids = tokenizer(traceback, add_special_tokens=False).input_ids
    if len(tb_ids) <= available:
        return traceback  # fits as-is

    # Keep the END of the traceback — the error type/message is on the last line
    trimmed = tokenizer.decode(tb_ids[-available:], skip_special_tokens=True)
    return trimmed


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RepairDataset(Dataset):
    """Wraps repair pairs into {'prompt': ..., 'answer': ...} for DataCollator."""

    def __init__(self, samples: list, tokenizer, language: str, max_prompt_len: int):
        self.items = []
        for s in samples:
            tb = _trim_traceback(
                s["traceback"], tokenizer, language,
                s["instruction"], s["buggy_code"], max_prompt_len,
            )
            self.items.append({
                "prompt": _build_prompt(language, s["instruction"], s["buggy_code"], tb),
                "answer": s["fixed_code"],
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a LoRA adapter with failed-execution feedback"
    )

    # ── Data ────────────────────────────────────────────────────────────────
    parser.add_argument("--language", type=str, required=True,
                        help="Programming language (python, cpp, java, …).")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Local dir or HF Hub repo ID containing calibration_{lang}.json files.")
    parser.add_argument("--results_source", type=str, default="local",
                        choices=["local", "hf_hub"])
    parser.add_argument("--results_repo_type", type=str, default="model",
                        choices=["dataset", "model", "space"])

    # ── Adapter ─────────────────────────────────────────────────────────────
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Local adapter directory (full path to adapter_config.json parent) "
                             "or HF Hub repo ID. For HF Hub, subfolder {language}/0 is used automatically.")

    # ── Model ───────────────────────────────────────────────────────────────
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-Coder-1.5B",
                        required=True)

    # ── Output ──────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fine-tuned adapter.")

    # ── Training ────────────────────────────────────────────────────────────
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--max_prompt_len", type=int, default=1024)
    parser.add_argument("--max_ans_len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=5)

    # ── System ──────────────────────────────────────────────────────────────
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--offload", action="store_true",
                        help="Enable ZeRO CPU offload.")
    parser.add_argument("--zero_stage", type=int, default=0,
                        help="DeepSpeed ZeRO stage. Use 0 for adapter refinement.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_dropout", action="store_true")

    # ── Logging ─────────────────────────────────────────────────────────────
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--group_name", type=str, default="refine_adapter")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--enable_tensorboard", action="store_true")
    parser.add_argument("--tensorboard_path", type=str, default="refine_tensorboard")

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.run_name is None:
        args.run_name = f"refine_{args.language}"

    # ── Distributed init ────────────────────────────────────────────────────
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()
    args.global_rank = torch.distributed.get_rank()

    # ── File logging (rank 0 only) ──────────────────────────────────────────
    if args.global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = os.path.join(args.output_dir, "refine.log")
        sys.stdout = TeeLogger(log_path)
        print(f"Logging to {log_path}")
        print(f"Args: {args}")

    # ── WandB ───────────────────────────────────────────────────────────────
    if args.enable_wandb and args.global_rank == 0:
        if not _WANDB_AVAILABLE:
            print("[warn] wandb not installed — skipping WandB logging.")
        else:
            wandb.init(
                project="CL4Code",
                group=args.group_name,
                job_type="refine",
                name=f"{args.group_name}_{args.run_name}",
                config=vars(args),
            )

    # ── DeepSpeed config ────────────────────────────────────────────────────
    ds_config = get_train_ds_config(
        offload=args.offload,
        stage=args.zero_stage,
        enable_tensorboard=args.enable_tensorboard,
        tb_path=args.tensorboard_path,
        tb_name="refine_sft",
    )
    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    ds_config["train_batch_size"] = (
        args.per_device_train_batch_size
        * torch.distributed.get_world_size()
        * args.gradient_accumulation_steps
    )

    set_random_seed(args.seed)
    torch.distributed.barrier()

    # ── Tokenizer ───────────────────────────────────────────────────────────
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    assert tokenizer.padding_side == "left"
    assert tokenizer.truncation_side == "left"

    # ── Base model ──────────────────────────────────────────────────────────
    base_model = create_hf_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        tokenizer,
        ds_config=ds_config,
        disable_dropout=args.disable_dropout,
    )

    # ── Load adapter (trainable) ─────────────────────────────────────────────
    if os.path.isdir(args.adapter_path):
        model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=True)
        print_rank_0(f"[refine] Loaded adapter from local: {args.adapter_path}", args.global_rank)
    else:
        subfolder = f"{args.language}/0"
        model = PeftModel.from_pretrained(
            base_model, args.adapter_path, subfolder=subfolder, is_trainable=True
        )
        print_rank_0(
            f"[refine] Loaded adapter from HF Hub: {args.adapter_path}/{subfolder}",
            args.global_rank,
        )
    model.print_trainable_parameters()

    # ── Training data ───────────────────────────────────────────────────────
    predictions = _load_results_file(
        args.results_dir, args.language, args.results_source, args.results_repo_type
    )
    repair_samples = extract_repair_samples(predictions)
    print_rank_0(
        f"[refine] {len(repair_samples)} repair pairs for language={args.language}",
        args.global_rank,
    )

    if len(repair_samples) == 0:
        print_rank_0("[refine] No training pairs found — exiting.", args.global_rank)
        return

    train_dataset = RepairDataset(repair_samples, tokenizer, args.language, args.max_prompt_len)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)

    data_collator = DataCollator(
        tokenizer,
        padding="longest",
        max_prompt_len=args.max_prompt_len,
        max_ans_len=args.max_ans_len,
        pad_to_multiple_of=8,
        inference=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
    )

    # ── Optimizer + scheduler ────────────────────────────────────────────────
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay)
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(
        optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95)
    )
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=args.num_warmup_steps
    )

    # ── DeepSpeed init ──────────────────────────────────────────────────────
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ── Training loop ────────────────────────────────────────────────────────
    total_steps = len(train_dataloader) * args.num_train_epochs
    print_rank_0(
        f"***** Running adapter refinement *****\n"
        f"  language      = {args.language}\n"
        f"  train pairs   = {len(train_dataset)}\n"
        f"  epochs        = {args.num_train_epochs}\n"
        f"  steps/epoch   = {len(train_dataloader)}\n"
        f"  total steps   = {total_steps}\n"
        f"  batch/device  = {args.per_device_train_batch_size}\n"
        f"  grad accum    = {args.gradient_accumulation_steps}\n"
        f"  lr            = {args.learning_rate}",
        args.global_rank,
    )

    global_step = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            del batch["sources"]  # added by DataCollator, not consumed by model
            batch = to_device(batch, device)

            outputs = model(**batch)
            loss = outputs.loss
            model.backward(loss)
            model.step()
            global_step += 1

            if global_step % args.logging_steps == 0:
                loss_val = get_all_reduce_mean(loss.detach()).item()
                print_rank_0(
                    f"[epoch {epoch+1}/{args.num_train_epochs}]"
                    f" step {step+1}/{len(train_dataloader)}"
                    f" global_step {global_step}"
                    f" loss {loss_val:.4f}",
                    args.global_rank,
                )
                if args.enable_wandb and args.global_rank == 0 and _WANDB_AVAILABLE:
                    wandb.log({"train/loss": loss_val, "global_step": global_step})

        print_rank_0(f"[refine] Epoch {epoch+1}/{args.num_train_epochs} complete.", args.global_rank)

    # ── Save ────────────────────────────────────────────────────────────────
    if args.zero_stage == 3:
        print_rank_0(
            "[refine] ZeRO stage 3 is not supported for adapter saving. "
            "Re-run with --zero_stage 0.",
            args.global_rank,
        )
        return

    torch.distributed.barrier()
    if args.global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        peft_model = model.module  # unwrap DeepSpeed engine
        peft_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"[refine] Saved fine-tuned adapter to {args.output_dir}")
        if args.enable_wandb and _WANDB_AVAILABLE:
            wandb.finish()


if __name__ == "__main__":
    main()
