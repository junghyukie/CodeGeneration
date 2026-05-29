## 1) Cài đặt & import thư viện

# Imports
import os
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoModel, AutoTokenizer

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

try:
    import plotly.express as px
except Exception as e:
    px = None
    print("plotly not available:", e)

# Optional: make HF hub warnings quieter
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

print("torch:", torch.__version__)

## 2) Load RouterConfig + chọn device
@dataclass
class TSNEConfig:
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B"
    feature_layers: int = 4
    routing_dim: int = 128
    max_length: int = 512
    batch_size: int = 16
    seed: int = 42

    # Executable dataset
    executable_dataset_name: str = "ankhanhtran02/CL4Code-executable-datasets"
    executable_languages: Tuple[str, ...] = (
        "python",
        "cpp",
        "swift",
        "rust",
        "csharp",
        "java",
        "php",
        "typescript",
        "shell",
    )

    # CodeTask/T5 tasks
    t5_tasks: Tuple[str, ...] = (
        "CONCODE",
        "CodeTrans",
        "CodeSearchNet",
        "BFP",
        "KodCode",
        "RunBugRun",
        # "TheVault_Csharp",
    )

    # Samples per class (keep small for t-SNE speed)
    n_train_per_class: int = 400
    n_eval_per_class: int = 300

    # Cache
    out_dir: str = "outputs/tsne"
    feature_cache_dir: str = "outputs/tsne/features"

cfg = TSNEConfig()
Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.feature_cache_dir).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(cfg.seed)
device = get_device()
print("device:", device)
print("model:", cfg.model_name)

## 3) Load tokenizer (giống `T5RoutingFeatureExtractor`)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

# Ensure pad token exists
if tokenizer.pad_token_id is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        # last resort
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

print("pad_token_id:", tokenizer.pad_token_id)
print("eos_token_id:", tokenizer.eos_token_id)

## 4) Dataloader cho Executable dataset (theo language/task)
def t5_pad_collate(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
    label_pad_id: int = -100,
) -> Dict[str, torch.Tensor]:
    """Dynamic pad collate for input_ids/attention_mask (+ labels if present)."""
    max_input_len = max(x["input_ids"].numel() for x in batch)

    input_ids = []
    attention_mask = []

    for x in batch:
        ids = x["input_ids"].long()
        mask = x["attention_mask"].long()
        pad_len = max_input_len - ids.numel()

        input_ids.append(torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)]))

    out = {
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
    }

    if "labels" in batch[0]:
        max_label_len = max(x["labels"].numel() for x in batch)
        labels = []
        for x in batch:
            y = x["labels"].long()
            pad_len = max_label_len - y.numel()
            labels.append(torch.cat([y, torch.full((pad_len,), label_pad_id, dtype=torch.long)]))
        out["labels"] = torch.stack(labels, dim=0)

    return out


def _load_split(dataset_name: str, split: str) -> Dataset:
    return load_dataset(dataset_name, split=split)


def _limit_dataset(dataset: Dataset, max_samples: int, seed: int = 0) -> Dataset:
    if max_samples is None or max_samples == -1:
        return dataset.shuffle(seed=seed)
    max_samples = min(max_samples, len(dataset))
    return dataset.shuffle(seed=seed).select(range(max_samples))


# def _load_training_dataset(dataset_name: str, language: str, max_train_samples: int, seed: int = 0) -> Dataset:
#     split_datasets = []
#     for split in ["train_OSS_Instruct", "train_McEval_Instruct"]:
#         ds = _load_split(dataset_name, split)
#         ds = ds.filter(lambda row: row["language"] == language and row.get("solution") is not None)
#         split_datasets.append(ds)
#     train_dataset = split_datasets[0] if len(split_datasets) == 1 else concatenate_datasets(split_datasets)
#     train_dataset = _limit_dataset(train_dataset, max_train_samples, seed)

#     # keep only 2 columns
#     ds2 = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in ("instruction", "solution")])
#     ds2 = ds2.rename_column("instruction", "prompt")
#     ds2 = ds2.rename_column("solution", "answer")
#     return ds2


# def _load_eval_dataset(dataset_name: str, language: str, max_eval_samples: int, seed: int = 0) -> Dataset:
#     ds = _load_split(dataset_name, "test_McEval")
#     ds = ds.filter(lambda row: row["language"] == language and row.get("test") is not None)
#     ds = _limit_dataset(ds, max_eval_samples, seed)

#     ds2 = ds.remove_columns([c for c in ds.column_names if c not in ("instruction", "solution")])
#     ds2 = ds2.rename_column("instruction", "prompt")
#     ds2 = ds2.rename_column("solution", "answer")
#     return ds2


# def build_executable_dataloader(
#     tokenizer,
#     dataset_name: str,
#     language: str,
#     split: str,
#     batch_size: int,
#     k: int,
#     seed: int,
#     max_length: int,
# ) -> DataLoader:
#     if split == "train":
#         dataset = _load_training_dataset(dataset_name, language, k, seed)
#     elif split in {"validation", "eval"}:
#         dataset = _load_eval_dataset(dataset_name, language, k, seed)
#     elif split == "test":
#         dataset = _load_eval_dataset(dataset_name, language, k, seed)
#     else:
#         raise ValueError(f"Unknown executable split: {split}")

#     def preprocess_batch(examples):
#         src_texts = [str(t).strip() for t in examples["prompt"]]
#         return tokenizer(src_texts, padding="max_length", truncation=True, max_length=max_length)

#     enc = dataset.map(preprocess_batch, batched=True, remove_columns=dataset.column_names)
#     enc.set_format(type="torch", columns=["input_ids", "attention_mask"])

#     pad_id = tokenizer.pad_token_id or 0
#     return DataLoader(
#         enc,
#         batch_size=batch_size,
#         shuffle=(split == "train"),
#         collate_fn=lambda b: t5_pad_collate(b, pad_token_id=pad_id, label_pad_id=-100),
#     )

# print("executable languages:", cfg.executable_languages)

## 5) Trích xuất feature z cho executable tasks (cache theo file .pt)
class RoutingFeatureExtractor:
    """Frozen encoder feature extractor (prefix pooled + full pooled), then random orthonormal projection."""

    def __init__(self, model_name: str, feature_layers: int, routing_dim: int, device: torch.device, seed: int):
        self.model_name = model_name
        self.feature_layers = feature_layers
        self.routing_dim = routing_dim
        self.device = device

        self.tokenizer = tokenizer

        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        hidden_size = getattr(self.model.config, "d_model", None) or getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(f"Cannot infer hidden size from model config for {model_name}")

        proj_in_dim = hidden_size * 2
        self.P = self._make_row_orthonormal_projection(p=routing_dim, d=proj_in_dim, seed=seed, device=device)

    @staticmethod
    def _make_row_orthonormal_projection(p: int, d: int, seed: int, device: torch.device) -> torch.Tensor:
        if p > d:
            raise ValueError(f"routing_dim p={p} must be <= hidden_size d={d}")
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        A = torch.randn(d, p, generator=gen) / math.sqrt(p)
        Q, _ = torch.linalg.qr(A, mode="reduced")
        P = Q.T.contiguous().to(device)
        return P

    @torch.no_grad()
    def encode_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        layer_idx = min(cfg.feature_layers, len(out.hidden_states) - 1)
        H = out.hidden_states[layer_idx]  # [B, T, D]

        def masked_mean_pool(x: torch.Tensor, mask_2d: torch.Tensor) -> torch.Tensor:
            m = mask_2d.unsqueeze(-1).to(x.dtype)
            denom = m.sum(dim=1).clamp_min(1.0)
            return (x * m).sum(dim=1) / denom

        prefix_len = min(64, H.shape[1])
        h_prefix = masked_mean_pool(H[:, :prefix_len, :], attention_mask[:, :prefix_len])
        h_full = masked_mean_pool(H, attention_mask)

        pooled = torch.cat([h_prefix, h_full], dim=-1)
        pooled = F.layer_norm(pooled.float(), normalized_shape=(pooled.shape[-1],))

        z = pooled @ self.P.T.float()
        return z.detach().cpu()

    @torch.no_grad()
    def extract_features(self, dataloader: Iterable[Dict[str, torch.Tensor]], desc: str) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for batch in tqdm(dataloader, desc=desc):
            chunks.append(self.encode_batch(batch))
        if not chunks:
            raise RuntimeError(f"No features extracted for {desc}")
        return torch.cat(chunks, dim=0).float()


extractor = RoutingFeatureExtractor(
    model_name=cfg.model_name,
    feature_layers=cfg.feature_layers,
    routing_dim=cfg.routing_dim,
    device=device,
    seed=cfg.seed,
)


# def feature_cache_path_executable(language: str, split: str, k: int) -> Path:
#     safe_lang = language.replace("/", "_")
#     return Path(cfg.feature_cache_dir) / f"executable_{safe_lang}_{split}_k{k}_L{cfg.feature_layers}_p{cfg.routing_dim}.pt"


# def get_or_extract_features_executable(language: str, split: str, k: int) -> torch.Tensor:
#     path = feature_cache_path_executable(language, split, k)
#     path.parent.mkdir(parents=True, exist_ok=True)

#     if path.exists():
#         return torch.load(path, map_location="cpu").float()

#     dl = build_executable_dataloader(
#         tokenizer=tokenizer,
#         dataset_name=cfg.executable_dataset_name,
#         language=language,
#         split=split,
#         batch_size=cfg.batch_size,
#         k=k,
#         seed=cfg.seed,
#         max_length=cfg.max_length,
#     )
#     z = extractor.extract_features(dl, desc=f"extract executable {language}/{split}")
#     torch.save(z, path)
#     return z

## 6) t-SNE 2D cho executable dataset (scatter theo task/language)
def run_tsne(X: np.ndarray, n_components: int, seed: int = 42, perplexity: float = 30.0) -> np.ndarray:
    # t-SNE is sensitive; keep init='pca' for stability
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        max_iter=1500,
    )
    return tsne.fit_transform(X)


# # Extract and merge
# zs = []
# ys = []
# meta = []

# for lang in cfg.executable_languages:
#     z = get_or_extract_features_executable(lang, split="train", k=cfg.n_train_per_class)
#     zs.append(z)
#     ys.extend([lang] * len(z))
#     meta.extend(list(range(len(z))))

# Z = torch.cat(zs, dim=0).numpy()
# labels = np.array(ys)
# print("Z shape:", Z.shape)

# # Downsample if too big
# max_plot = 4000
# if Z.shape[0] > max_plot:
#     idx = np.random.RandomState(cfg.seed).choice(Z.shape[0], size=max_plot, replace=False)
#     Zp = Z[idx]
#     labels_p = labels[idx]
# else:
#     Zp = Z
#     labels_p = labels

# perplexity = min(40.0, max(5.0, (len(Zp) - 1) / 3.0))
# print("t-SNE perplexity:", perplexity)

# emb2 = run_tsne(Zp, n_components=2, seed=cfg.seed, perplexity=perplexity)

# df2 = pd.DataFrame({"x": emb2[:, 0], "y": emb2[:, 1], "label": labels_p})

# plt.figure(figsize=(10, 8))
# for lab in sorted(df2["label"].unique()):
#     d = df2[df2["label"] == lab]
#     plt.scatter(d["x"], d["y"], s=8, alpha=0.7, label=lab)
# plt.title("Executable dataset t-SNE 2D (train)")
# plt.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
# plt.tight_layout()
# out_png = Path(cfg.out_dir) / "tsne2d_executable_train.png"
# plt.savefig(out_png, dpi=200)
# print("saved:", out_png)
# plt.show()

# ## 7) t-SNE 3D cho executable dataset (plotly)
# if px is None:
#     print("Install plotly to view 3D plot: pip install plotly")
# else:
#     emb3 = run_tsne(Zp, n_components=3, seed=cfg.seed, perplexity=perplexity)
#     df3 = pd.DataFrame({"x": emb3[:, 0], "y": emb3[:, 1], "z": emb3[:, 2], "label": labels_p})

#     fig = px.scatter_3d(
#         df3,
#         x="x",
#         y="y",
#         z="z",
#         color="label",
#         opacity=0.7,
#         title="Executable dataset t-SNE 3D (train)",
#     )
#     out_html = Path(cfg.out_dir) / "tsne3d_executable_train.html"
#     fig.write_html(str(out_html), include_plotlyjs="cdn")
#     print("saved:", out_html)
#     fig.show()

## 8) Dataloader cho “CodeTask/T5 tasks” (7 tasks trong `t5_data.py`)
from t5_data import T5Dataset

# instantiate dataset helper
# NOTE: This will load different HF datasets for tasks. If TheVault is slow/error, skip it for visualization.
t5_ds = T5Dataset(tokenizer)


def build_t5_dataloader(task: str, split: str, k: int) -> DataLoader:
    # max_length is only input max len; target handled inside T5Dataset
    return t5_ds.get_final_ds(
        task=task,
        split=split,
        batch_size=cfg.batch_size,
        k=k,
        seed=cfg.seed,
        return_test=False,
        max_length=cfg.max_length,
    )

## 9) Trích xuất feature z cho 7 tasks (cache)
def feature_cache_path_t5(task: str, split: str, k: int) -> Path:
    safe_task = task.replace("/", "_")
    return Path(cfg.feature_cache_dir) / f"t5_{safe_task}_{split}_k{k}_L{cfg.feature_layers}_p{cfg.routing_dim}.pt"


def get_or_extract_features_t5(task: str, split: str, k: int) -> torch.Tensor:
    path = feature_cache_path_t5(task, split, k)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return torch.load(path, map_location="cpu").float()

    dl = build_t5_dataloader(task, split=split, k=k)
    z = extractor.extract_features(dl, desc=f"extract t5 {task}/{split}")
    torch.save(z, path)
    return z


# Extract per task
zs_t5 = []
ys_t5 = []

for task in cfg.t5_tasks:
    try:
        z = get_or_extract_features_t5(task, split="train", k=cfg.n_train_per_class)
    except Exception as e:
        print(f"[skip] task={task} due to error:", e)
        continue

    zs_t5.append(z)
    ys_t5.extend([task] * len(z))

Zt5 = torch.cat(zs_t5, dim=0).numpy() if zs_t5 else np.zeros((0, cfg.routing_dim), dtype=np.float32)
labels_t5 = np.array(ys_t5)
print("Zt5 shape:", Zt5.shape)

## 10) t-SNE 2D cho 7 tasks
if Zt5.shape[0] == 0:
    print("No T5-task features extracted.")
else:
    max_plot = 4000
    if Zt5.shape[0] > max_plot:
        idx = np.random.RandomState(cfg.seed).choice(Zt5.shape[0], size=max_plot, replace=False)
        Zp2 = Zt5[idx]
        labels_p2 = labels_t5[idx]
    else:
        Zp2 = Zt5
        labels_p2 = labels_t5

    perplexity2 = min(40.0, max(5.0, (len(Zp2) - 1) / 3.0))
    emb2_t5 = run_tsne(Zp2, n_components=2, seed=cfg.seed, perplexity=perplexity2)

    df2_t5 = pd.DataFrame({"x": emb2_t5[:, 0], "y": emb2_t5[:, 1], "label": labels_p2})

    plt.figure(figsize=(10, 8))
    for lab in sorted(df2_t5["label"].unique()):
        d = df2_t5[df2_t5["label"] == lab]
        plt.scatter(d["x"], d["y"], s=8, alpha=0.7, label=lab)
    plt.title("T5 tasks t-SNE 2D (train)")
    plt.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_png = Path(cfg.out_dir) / "tsne2d_t5tasks_train123.png"
    plt.savefig(out_png, dpi=200)
    print("saved:", out_png)
    plt.show()

## 11) t-SNE 3D cho 7 tasks
if px is None:
    print("Install plotly to view 3D plot: pip install plotly")
elif Zt5.shape[0] == 0:
    print("No T5-task features extracted.")
else:
    max_plot = 4000
    if Zt5.shape[0] > max_plot:
        idx = np.random.RandomState(cfg.seed).choice(Zt5.shape[0], size=max_plot, replace=False)
        Zp2 = Zt5[idx]
        labels_p2 = labels_t5[idx]
    else:
        Zp2 = Zt5
        labels_p2 = labels_t5

    perplexity2 = min(40.0, max(5.0, (len(Zp2) - 1) / 3.0))
    emb3_t5 = run_tsne(Zp2, n_components=3, seed=cfg.seed, perplexity=perplexity2)

    df3_t5 = pd.DataFrame({"x": emb3_t5[:, 0], "y": emb3_t5[:, 1], "z": emb3_t5[:, 2], "label": labels_p2})
    fig = px.scatter_3d(df3_t5, x="x", y="y", z="z", color="label", opacity=0.7, title="T5 tasks t-SNE 3D (train)")
    out_html = Path(cfg.out_dir) / "tsne3d_t5tasks_train.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print("saved:", out_html)
    fig.show()


from datasets import get_dataset_config_names

name = "Fsoft-AIC/the-vault-function"

try:
    cfgs = get_dataset_config_names(name)
    print("config names:", cfgs[:20], "... total:", len(cfgs))
except Exception as e:
    print("Cannot list config names:", e)

print("\nTry load minimal split via standard API...")

# Try the standard split name. This may fail if dataset script is blocked by your datasets version.
try:
    ds = load_dataset(name, split="train")
    print("loaded train rows:", len(ds))
    print("columns:", ds.column_names)
    print("sample:", ds[0])
except Exception as e:
    print("load_dataset failed:", e)

