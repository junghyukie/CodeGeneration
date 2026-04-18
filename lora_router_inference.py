from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from t5_trainer1 import T5Dataset


DEFAULT_TASKS = [
    "CodeTrans",
    "CodeSearchNet",
    "BFP",
    "CONCODE",
    "KodCode",
    "RunBugRun",
    "CoST",
]


def set_up_file_logger(log_filepath: str) -> logging.Logger:
    logger = logging.getLogger("lora_router_inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    os.makedirs(os.path.dirname(log_filepath) or ".", exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_filepath, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


class HFEncoderEmbedder:
    """
    Lightweight embedding wrapper for plain Hugging Face encoder models
    (e.g., microsoft/unixcoder-base) with an `encode` method compatible
    with SentenceTransformer usage in this script.
    """

    def __init__(self, model_name: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.vocab_size = int(getattr(self.model.config, "vocab_size", 0) or 0)

    @torch.inference_mode()
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = False,
    ):
        outputs: List[torch.Tensor] = []
        total = len(texts)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = list(texts[start:end])
            toks = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )

            # Validate token ids on CPU first to avoid opaque CUDA device-side asserts.
            if self.vocab_size > 0 and "input_ids" in toks:
                ids = toks["input_ids"]
                min_id = int(ids.min().item())
                max_id = int(ids.max().item())
                if min_id < 0 or max_id >= self.vocab_size:
                    raise ValueError(
                        f"Token id out of range for embedder vocab. "
                        f"min_id={min_id}, max_id={max_id}, vocab_size={self.vocab_size}"
                    )

            toks = {k: v.to(self.device) for k, v in toks.items()}
            model_out = self.model(**toks)

            # Mean pooling over valid tokens only.
            token_embeddings = model_out.last_hidden_state
            attention_mask = toks["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / counts

            if normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            outputs.append(embeddings.detach().cpu())

        all_embeddings = torch.cat(outputs, dim=0)
        if convert_to_numpy:
            return all_embeddings.numpy()
        return all_embeddings


def build_embedder(model_name: str, device: str, backend: str = "auto"):
    """
    Build embedder with explicit backend control.
    - auto: prefer HF encoder for UniXcoder, else try SentenceTransformer then HF.
    - sentence_transformers: force SentenceTransformer path.
    - hf: force plain Hugging Face encoder path.
    """
    if backend not in {"auto", "sentence_transformers", "hf"}:
        raise ValueError("backend must be one of: {'auto', 'sentence_transformers', 'hf'}")

    if backend == "hf":
        return HFEncoderEmbedder(model_name=model_name, device=device)

    if backend == "sentence_transformers":
        return SentenceTransformer(model_name, device=device)

    # auto mode
    if "unixcoder" in model_name.lower():
        return HFEncoderEmbedder(model_name=model_name, device=device)

    try:
        return SentenceTransformer(model_name, device=device)
    except Exception:
        return HFEncoderEmbedder(model_name=model_name, device=device)


def _get_cache_paths(cache_dir: str, task_name: str, split: str, max_samples: int) -> Tuple[str, str]:
    suffix = f"{task_name}_{split}_k{max_samples}"
    emb_path = os.path.join(cache_dir, f"{suffix}_embeddings.npy")
    proto_path = os.path.join(cache_dir, f"{suffix}_prototype.npy")
    return emb_path, proto_path


def load_task_split_like_training(
    dataset_helper: T5Dataset,
    task: str,
    split: str,
    max_samples: int = -1,
    seed: int = 0,
):
    """
    Load dataset exactly like training logic (from Hugging Face only),
    without tokenization so we can build routing embeddings and fused eval data.
    """
    if task == "CONCODE":
        dataset = load_dataset("AhmedSSoliman/CodeXGLUE-CONCODE", split=split)
    elif task == "CodeTrans":
        dataset = load_dataset("CM/codexglue_codetrans", split=split)
    elif task == "CodeSearchNet":
        dataset = load_dataset("semeru/code-text-ruby", split=split)
    elif task == "BFP":
        dataset = load_dataset("ayeshgk/code_x_glue_cc_code_refinement_annotated", split=split)
    elif task == "TheVault_Csharp":
        if split == "train":
            dataset = load_dataset(
                "Fsoft-AIC/the-vault-function",
                cache_dir="dataset/theVault",
                languages=["c_sharp"],
                split_set="train/small",
            )
        else:
            dataset = load_dataset(
                "Fsoft-AIC/the-vault-function",
                cache_dir="dataset/theVault",
                languages=["c_sharp"],
                split_set=split,
            )
    elif task == "KodCode":
        dataset = load_dataset("KodCode/KodCode-V1-SFT-R1", split="train")
    elif task == "RunBugRun":
        dataset = load_dataset("ASSERT-KTH/RunBugRun-Final", split="train")
        dataset = dataset.filter(lambda ex: ex.get("language", None) == "ruby")
    elif task == "CoST":
        dataset = load_dataset("dongg18/CoST", split=split)
    else:
        raise ValueError(f"Unknown task: {task}")

    if task in dataset_helper.train_only_tasks:
        dataset = dataset_helper._split_train_only(dataset, task, split, split_seed=42)

    if max_samples != -1:
        k = min(max_samples, len(dataset))
        dataset = dataset_helper.select_subset_ds(dataset, k=k, seed=seed)
    else:
        dataset = dataset.shuffle(seed=seed)
    return dataset


def format_training_input_text(dataset_helper: T5Dataset, task: str, example: Dict[str, Any]) -> str:
    text_col = dataset_helper.text_key[task]
    instruction = dataset_helper.task_instructions[task]
    return (instruction + str(example[text_col])).strip()


def format_training_target_text(dataset_helper: T5Dataset, task: str, example: Dict[str, Any]) -> str:
    label_col = dataset_helper.label_key[task]
    target = str(example[label_col])
    if task == "CodeSearchNet":
        target = dataset_helper._extract_first_paragraph(target)
    return target


def compute_prototypes(
    embedder,
    task_to_samples: Dict[str, Sequence[str]],
    split: str,
    max_samples: int,
    batch_size: int = 64,
    cache_dir: str | None = None,
    force_recompute: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Compute one prototype vector per task by averaging sentence embeddings.

    prototype_t = mean(embeddings_of_task_t)
    """
    prototypes: Dict[str, torch.Tensor] = {}

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    for task_name, samples in task_to_samples.items():
        if not samples:
            raise ValueError(f"Task '{task_name}' has no training samples.")

        emb_cache_path, proto_cache_path = (None, None)
        if cache_dir:
            emb_cache_path, proto_cache_path = _get_cache_paths(
                cache_dir,
                task_name,
                split=split,
                max_samples=max_samples,
            )

        if (
            cache_dir
            and not force_recompute
            and proto_cache_path
            and os.path.exists(proto_cache_path)
        ):
            proto_np = np.load(proto_cache_path)
            prototypes[task_name] = torch.from_numpy(proto_np).float()
            continue

        if (
            cache_dir
            and not force_recompute
            and emb_cache_path
            and os.path.exists(emb_cache_path)
        ):
            embeddings = np.load(emb_cache_path)
        else:
            embeddings = embedder.encode(
                list(samples),
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
                normalize_embeddings=False,
            )
            if cache_dir and emb_cache_path:
                np.save(emb_cache_path, embeddings)

        prototype = embeddings.mean(axis=0)
        if cache_dir and proto_cache_path:
            np.save(proto_cache_path, prototype)

        prototypes[task_name] = torch.from_numpy(prototype).float()

    return prototypes


def build_fused_eval_dataset(
    dataset_helper: T5Dataset,
    tasks: Sequence[str],
    split: str,
    max_samples_per_task: int,
    seed: int,
) -> Dataset:
    """
    Build a single balanced/limited evaluation dataset by fusing all tasks.
    Uses the same HF data loading and task-specific text formatting as training.
    """
    per_task_rows: Dict[str, List[Dict[str, str]]] = {}
    for task in tasks:
        raw_ds = load_task_split_like_training(
            dataset_helper=dataset_helper,
            task=task,
            split=split,
            max_samples=max_samples_per_task,
            seed=seed,
        )
        task_rows: List[Dict[str, str]] = []
        for ex in raw_ds:
            task_rows.append(
                {
                    "task": task,
                    "input_text": format_training_input_text(dataset_helper, task, ex),
                    "target_text": format_training_target_text(dataset_helper, task, ex),
                }
            )
        per_task_rows[task] = task_rows

    # Interleave tasks in round-robin order so early examples cover all tasks.
    rows: List[Dict[str, str]] = []
    max_len = max((len(v) for v in per_task_rows.values()), default=0)
    for i in range(max_len):
        for task in tasks:
            task_rows = per_task_rows.get(task, [])
            if i < len(task_rows):
                rows.append(task_rows[i])

    if not rows:
        raise ValueError("Fused evaluation dataset is empty.")
    return Dataset.from_list(rows)


def compute_similarity(
    input_embedding: torch.Tensor,
    prototypes: Dict[str, torch.Tensor],
    metric: str = "cosine",
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Compute task weights from similarities and normalize with softmax.

    w_t = softmax(sim(x, prototype_t) / temperature)
    """
    if metric not in {"cosine", "dot"}:
        raise ValueError("metric must be one of: {'cosine', 'dot'}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    task_names = list(prototypes.keys())
    proto_stack = torch.stack([prototypes[t] for t in task_names], dim=0).float()
    x = input_embedding.float().view(1, -1)

    if metric == "cosine":
        sims = F.cosine_similarity(x, proto_stack, dim=1)
    else:
        sims = (x * proto_stack).sum(dim=1)

    weights = torch.softmax(sims / temperature, dim=0)
    return {task: float(weights[i].item()) for i, task in enumerate(task_names)}


def merge_lora_weights(
    peft_model: PeftModel,
    adapter_names: List[str],
    weights: List[float],
    merged_adapter_name: str = "runtime_weighted",
) -> str:
    """
    Build a weighted LoRA adapter using PEFT's adapter algebra and activate it.
    This avoids reloading the base model or all adapters at inference time.
    """
    if len(adapter_names) != len(weights):
        raise ValueError("adapter_names and weights must have the same length")

    existing = set(peft_model.peft_config.keys())
    if merged_adapter_name in existing:
        if hasattr(peft_model, "delete_adapter"):
            peft_model.delete_adapter(merged_adapter_name)
        else:
            merged_adapter_name = f"runtime_weighted_{int(time.time() * 1000)}"

    peft_model.add_weighted_adapter(
        adapters=adapter_names,
        weights=weights,
        adapter_name=merged_adapter_name,
        combination_type="linear",
    )
    peft_model.set_adapter(merged_adapter_name)
    if hasattr(peft_model, "enable_adapter_layers"):
        peft_model.enable_adapter_layers()
    return merged_adapter_name


@torch.inference_mode()
def inference(
    peft_model: PeftModel,
    tokenizer,
    embedder,
    prototypes: Dict[str, torch.Tensor],
    input_text: str,
    metric: str = "cosine",
    temperature: float = 0.1,
    max_new_tokens: int = 128,
    num_beams: int = 4,
    device: str = "cuda",
) -> Tuple[str, Dict[str, float], str]:
    """
    1) Route input to tasks using similarity to prototypes.
    2) Construct weighted LoRA adapter from routing weights.
    3) Run generation with the weighted adapter active.
    """
    emb_np = embedder.encode(
        [input_text],
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )[0]
    input_embedding = torch.from_numpy(emb_np).float()

    task_weights = compute_similarity(
        input_embedding=input_embedding,
        prototypes=prototypes,
        metric=metric,
        temperature=temperature,
    )

    adapter_names = list(task_weights.keys())
    weights = [task_weights[name] for name in adapter_names]
    merged_name = merge_lora_weights(
        peft_model=peft_model,
        adapter_names=adapter_names,
        weights=weights,
        merged_adapter_name="runtime_weighted",
    )

    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    out_ids = peft_model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    output_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return output_text, task_weights, merged_name


@torch.inference_mode()
def inference_batch(
    peft_model: PeftModel,
    tokenizer,
    embedder,
    prototypes: Dict[str, torch.Tensor],
    input_texts: Sequence[str],
    metric: str = "cosine",
    temperature: float = 0.1,
    max_new_tokens: int = 128,
    num_beams: int = 4,
    device: str = "cuda",
    embedding_batch_size: int = 64,
) -> List[Tuple[str, Dict[str, float], str]]:
    """
    Batched routing (embedding+similarity) with weighted-adapter generation.
    Generation still runs per sample because each sample has unique routing weights.
    """
    if not input_texts:
        return []

    emb_np = embedder.encode(
        list(input_texts),
        batch_size=embedding_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )

    results: List[Tuple[str, Dict[str, float], str]] = []
    for text, emb in zip(input_texts, emb_np):
        input_embedding = torch.from_numpy(emb).float()
        task_weights = compute_similarity(
            input_embedding=input_embedding,
            prototypes=prototypes,
            metric=metric,
            temperature=temperature,
        )

        adapter_names = list(task_weights.keys())
        weights = [task_weights[name] for name in adapter_names]
        merged_name = merge_lora_weights(
            peft_model=peft_model,
            adapter_names=adapter_names,
            weights=weights,
            merged_adapter_name="runtime_weighted",
        )

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        out_ids = peft_model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        output_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        results.append((output_text, task_weights, merged_name))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prototype-based routing + weighted LoRA inference"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Salesforce/codet5p-770m",
        help="Base seq2seq model path/name",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="microsoft/unixcoder-base",
        help="Embedding model for routing (supports SentenceTransformer or plain HF encoders like UniXcoder)",
    )
    parser.add_argument(
        "--embedding_backend",
        type=str,
        default="auto",
        choices=["auto", "sentence_transformers", "hf"],
        help="Embedding backend selection. Use 'hf' for UniXcoder stability.",
    )
    parser.add_argument(
        "--lora_root",
        type=str,
        required=True,
        default="lora",
        help="Root dir containing lora_<task_name>/ folders",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=DEFAULT_TASKS,
        help="Task names to include (must match training task names)",
    )
    parser.add_argument(
        "--prototype_split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split used to construct task prototypes",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split used to construct fused evaluation set",
    )
    parser.add_argument(
        "--max_samples_per_task",
        type=int,
        default=500,
        help="Max eval samples per task for fused evaluation dataset",
    )
    parser.add_argument(
        "--max_prototype_samples_per_task",
        type=int,
        default=9000,
        help="Max train samples per task for prototypes (-1 = all)",
    )
    parser.add_argument(
        "--similarity_metric",
        type=str,
        default="cosine",
        choices=["cosine", "dot"],
        help="Similarity metric for routing",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Softmax temperature for routing weights",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="prototype_cache",
        help="Where to cache embeddings/prototypes",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=64,
        help="Batch size for embedding model encoding",
    )
    parser.add_argument(
        "--force_recompute_prototypes",
        action="store_true",
        help="Recompute cached embeddings/prototypes",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        help="Run inference over fused multi-task evaluation dataset",
    )
    parser.add_argument(
        "--max_eval_examples",
        type=int,
        default=100000,
        help="How many fused eval examples to run when --run_eval is set",
    )
    parser.add_argument(
        "--eval_output_jsonl",
        type=str,
        default="routed_eval_outputs.jsonl",
        help="Path to JSONL file for per-sample eval outputs (set empty string to disable)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="Batch size for eval routing/inference when --run_eval is set",
    )
    parser.add_argument(
        "--log_filepath",
        type=str,
        default="logs/lora_router_inference.log",
        help="File path for inference logs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default=None,
        help="Optional one-shot inference text. If omitted, interactive mode is used.",
    )

    args = parser.parse_args()
    tasks = args.tasks
    logger = set_up_file_logger(args.log_filepath)

    def log_info(msg: str) -> None:
        logger.info(msg)

    print("[1/5] Loading embedding model...")
    log_info("[1/5] Loading embedding model...")
    embedder = build_embedder(
        args.embedding_model,
        device=args.device,
        backend=args.embedding_backend,
    )

    print("[2/5] Loading tokenizer + training data helper...")
    log_info("[2/5] Loading tokenizer + training data helper...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    dataset_helper = T5Dataset(tokenizer)

    unknown_tasks = [t for t in tasks if t not in dataset_helper.task_list]
    if unknown_tasks:
        raise ValueError(f"Unknown task names: {unknown_tasks}")

    print("[3/5] Building task prototypes from Hugging Face datasets...")
    log_info("[3/5] Building task prototypes from Hugging Face datasets...")
    task_to_samples: Dict[str, List[str]] = {}
    for task_name in tasks:
        raw_ds = load_task_split_like_training(
            dataset_helper=dataset_helper,
            task=task_name,
            split=args.prototype_split,
            max_samples=args.max_prototype_samples_per_task,
            seed=42,
        )
        samples = [format_training_input_text(dataset_helper, task_name, ex) for ex in raw_ds]
        if not samples:
            raise ValueError(f"No samples found for task '{task_name}' from HF split '{args.prototype_split}'")
        task_to_samples[task_name] = samples
        print(f"  - {task_name}: {len(samples)} prototype samples")
        log_info(f"prototype_samples task={task_name} count={len(samples)}")

    prototypes = compute_prototypes(
        embedder=embedder,
        task_to_samples=task_to_samples,
        split=args.prototype_split,
        max_samples=args.max_prototype_samples_per_task,
        batch_size=args.embedding_batch_size,
        cache_dir=args.cache_dir,
        force_recompute=args.force_recompute_prototypes,
    )

    print("[4/5] Loading base model + all task LoRA adapters once...")
    log_info("[4/5] Loading base model + all task LoRA adapters once...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model).to(args.device)

    task_names = list(tasks)
    if not task_names:
        raise ValueError("No tasks were provided.")

    first_task = task_names[0]
    first_lora_path = os.path.join(args.lora_root, f"lora_{first_task}")
    if not os.path.isdir(first_lora_path):
        raise FileNotFoundError(f"Missing adapter directory: {first_lora_path}")

    peft_model = PeftModel.from_pretrained(
        base_model,
        first_lora_path,
        adapter_name=first_task,
        is_trainable=False,
    )

    for task_name in task_names[1:]:
        lora_path = os.path.join(args.lora_root, f"lora_{task_name}")
        if not os.path.isdir(lora_path):
            raise FileNotFoundError(f"Missing adapter directory: {lora_path}")
        peft_model.load_adapter(lora_path, adapter_name=task_name, is_trainable=False)

    peft_model.to(args.device)
    peft_model.eval()

    print("[5/5] Inference ready.")
    log_info("[5/5] Inference ready.")

    def run_one(text: str) -> None:
        output, task_weights, merged_name = inference(
            peft_model=peft_model,
            tokenizer=tokenizer,
            embedder=embedder,
            prototypes=prototypes,
            input_text=text,
            metric=args.similarity_metric,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            device=args.device,
        )
        sorted_weights = sorted(task_weights.items(), key=lambda x: x[1], reverse=True)
        print(f"\nActive merged adapter: {merged_name}")
        print("Routing weights (high -> low):")
        for task_name, w in sorted_weights:
            print(f"  {task_name:20s} {w:.4f}")
        print("\nModel output:")
        print(output)

    if args.input_text is not None:
        run_one(args.input_text)
        if not args.run_eval:
            return

    if args.run_eval:
        print("\nBuilding fused evaluation dataset from all tasks...")
        log_info("Building fused evaluation dataset from all tasks...")
        eval_dataset = build_fused_eval_dataset(
            dataset_helper=dataset_helper,
            tasks=task_names,
            split=args.eval_split,
            max_samples_per_task=args.max_samples_per_task,
            seed=7,
        )
        print(f"Fused eval dataset size: {len(eval_dataset)}")
        log_info(f"fused_eval_size={len(eval_dataset)}")
        task_counts = {t: 0 for t in task_names}
        for row in eval_dataset:
            task_counts[row["task"]] = task_counts.get(row["task"], 0) + 1
        print("Fused eval task distribution:")
        for t in task_names:
            print(f"  {t:20s} {task_counts.get(t, 0)}")
            log_info(f"fused_eval_task_count task={t} count={task_counts.get(t, 0)}")

        limit = len(eval_dataset)
        if args.max_eval_examples > 0:
            limit = min(limit, args.max_eval_examples)
        print(f"Running routed inference for {limit} fused eval samples...")
        log_info(
            f"Running routed inference for {limit} fused eval samples with eval_batch_size={args.eval_batch_size}"
        )

        jsonl_output_path = args.eval_output_jsonl.strip() or "routed_eval_outputs.jsonl"
        output_dir = os.path.dirname(jsonl_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        jsonl_fp = open(jsonl_output_path, "w", encoding="utf-8", buffering=1)
        print(f"Logging per-sample eval outputs to: {jsonl_output_path}")
        log_info(f"Logging per-sample eval outputs to: {jsonl_output_path}")

        try:
            batch_size = max(1, int(args.eval_batch_size))
            processed = 0
            for start in range(0, limit, batch_size):
                end = min(start + batch_size, limit)
                batch_rows = [eval_dataset[idx] for idx in range(start, end)]
                batch_inputs = [row["input_text"] for row in batch_rows]

                batch_results = inference_batch(
                    peft_model=peft_model,
                    tokenizer=tokenizer,
                    embedder=embedder,
                    prototypes=prototypes,
                    input_texts=batch_inputs,
                    metric=args.similarity_metric,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    device=args.device,
                    embedding_batch_size=args.embedding_batch_size,
                )

                for row, (output, task_weights, _) in zip(batch_rows, batch_results):
                    top_task = max(task_weights.items(), key=lambda x: x[1])[0]
                    record = {
                        "task": row["task"],
                        "input": row["input_text"],
                        "target": row["target_text"],
                        "prediction": output,
                        "routed_top": top_task,
                        "weight_routing": task_weights,
                    }
                    jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

                jsonl_fp.flush()

                processed = end
                log_info(f"eval_progress processed={processed}/{limit}")
        finally:
            jsonl_fp.close()
            log_info("Closed eval output JSONL file.")

        if args.input_text is not None:
            return

    print("\nInteractive mode. Type input text and press Enter. Type 'exit' to quit.")
    while True:
        user_in = input("\nInput> ").strip()
        if not user_in:
            continue
        if user_in.lower() in {"exit", "quit"}:
            break
        run_one(user_in)


if __name__ == "__main__":
    main()
