from __future__ import annotations

"""
RC-GMM: Risk-Controlled Continual Density Routing for expanding LoRA expert registries.

This file rewrites the original append-only GMM router around the revised method:

  1) Frozen routing representation: fixed backbone + fixed orthonormal projection.
  2) Independent frozen per-task diagonal GMM anchors A_k.
  3) Every new expert k>0 receives a selective admission gate G_k(z).
  4) G_k is optimized to maximize new-task exposure while respecting a separate
     false-admission budget for every previous task.
  5) Budgets use a summable lifetime schedule E_j / [r(r+1)].
  6) The new expert is completely absent from the routing normalization when G_k(z)=0,
     so old posterior weights are exactly unchanged outside the admission region.
  7) Gate fitting and gate certification are separated. Certification uses fresh
     pseudo-samples from frozen old GMM anchors and a held-out current-task split.

The implementation provides:
  - full multi-constraint likelihood-ratio gate (default),
  - conservative pairwise intersection gate (ablation),
  - ordinary ungated GMM routing (baseline),
  - Clopper-Pearson exact one-sided binomial certificates (default),
  - Hoeffding certificates matching the simple theorem in the research note,
  - ASR-avg / ASR-worst / NDA and routing-change diagnostics.

This file only implements the router. The returned soft routing weights can be consumed
by the LoRA composition/generation code.
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    from t5_data import T5Dataset  # noqa: F401
except ImportError:
    T5Dataset = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class RouterConfig:
    model_name: str = "SalesForce/codet5-small"
    output_dir: str = "./rc_gmm_ckpt"
    feature_cache_dir: Optional[str] = None
    dataset_source: str = "codetask"
    executable_dataset_name: str = "ankhanhtran02/CL4Code-executable-datasets"

    tasks: Tuple[str, ...] = (
        "CONCODE",
        "CodeTrans",
        "CodeSearchNet",
        "BFP",
        "KodCode",
        "RunBugRun",
        "TheVault_Csharp",
        "CoST",
    )

    # Frozen routing representation.
    feature_layers: int = 4
    routing_dim: int = 256
    max_length: int = 512
    batch_size: int = 16
    train_k: int = 2000
    eval_k: int = 1000
    seed: int = 42

    # Frozen per-task GMM anchors.
    gmm_components: int = 4
    em_iters: int = 50
    em_tol: float = 1e-4
    variance_floor: float = 1e-4
    eps: float = 1e-8

    # RC-GMM admission.
    # dual      : multi-constraint likelihood-ratio gate p_n > sum_j lambda_j p_j
    # pairwise  : conservative intersection of pairwise LR tests
    # none      : ordinary ungated GMM router baseline
    gate_mode: str = "dual"

    # Lifetime perturbation budget E_j. For a new task n and old task j,
    # epsilon_{n,j} = E_j / [(n-j)(n-j+1)].
    lifetime_risk_budget: float = 0.10

    # Target lower confidence bound P_n(G_n=1) >= min_new_admission.
    min_new_admission: float = 0.80

    # Current task routing features are split into fit / optimize / certify-new.
    # Optimize is used only for diagnostics / model-selection checks; certification
    # is strictly independent of fitting and gate optimization.
    fit_fraction: float = 0.70
    opt_fraction: float = 0.15

    # Monte-Carlo optimization and certification sizes.
    gate_opt_samples_per_density: int = 6000
    gate_cert_samples_per_old: int = 20000
    auto_scale_cert_samples: bool = True
    max_gate_cert_samples_per_old: int = 1000000
    cert_batch_size: int = 8192
    dual_maxiter: int = 400
    dual_ftol: float = 1e-10

    # Leave population-risk slack during optimization so independent certification
    # has room to pass. Must be in (0, 1].
    optimization_budget_fraction: float = 0.90

    # Simultaneous confidence budget over all old-risk and new-power certificates.
    lifetime_confidence_budget: float = 0.05
    cert_bound: str = "clopper_pearson"  # clopper_pearson | hoeffding

    # Routing.
    temperature: float = 1.0
    eval_split: str = "test"

    # I/O.
    save_features: bool = True
    force_recompute_features: bool = False
    continue_on_unsafe: bool = False


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def router_pad_collate(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
    label_pad_id: int = -100,
):
    max_input_len = max(x["input_ids"].numel() for x in batch)
    input_ids = []
    attention_mask = []

    for x in batch:
        ids = x["input_ids"].long()
        mask = x["attention_mask"].long()
        pad_len = max_input_len - ids.numel()

        input_ids.append(
            torch.cat(
                [ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]
            )
        )
        attention_mask.append(
            torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
        )

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
            labels.append(
                torch.cat(
                    [y, torch.full((pad_len,), label_pad_id, dtype=torch.long)]
                )
            )
        out["labels"] = torch.stack(labels, dim=0)

    return out


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_tasks(raw: Optional[str], default: Tuple[str, ...]) -> Tuple[str, ...]:
    if raw is None or raw.strip() == "":
        return default
    return tuple(t.strip() for t in raw.split(",") if t.strip())


def safe_name(s: str) -> str:
    return (
        s.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def split_task_features(
    z: torch.Tensor,
    fit_fraction: float,
    opt_fraction: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split current-task routing features into Fit / Optimize / Cert-New."""
    if not (0.0 < fit_fraction < 1.0):
        raise ValueError("fit_fraction must be in (0,1)")
    if not (0.0 <= opt_fraction < 1.0):
        raise ValueError("opt_fraction must be in [0,1)")
    if fit_fraction + opt_fraction >= 1.0:
        raise ValueError("fit_fraction + opt_fraction must be < 1")

    n = len(z)
    if n < 6:
        raise ValueError(
            f"Need at least 6 routing samples to create Fit/Optimize/Cert-New splits; got {n}."
        )

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    z = z[perm]

    n_fit = max(2, int(round(n * fit_fraction)))
    n_opt = max(1, int(round(n * opt_fraction)))
    if n_fit + n_opt >= n:
        n_opt = max(1, n - n_fit - 1)
    n_cert = n - n_fit - n_opt
    if n_cert <= 0:
        raise RuntimeError("Invalid feature split; certification set is empty")

    return z[:n_fit], z[n_fit : n_fit + n_opt], z[n_fit + n_opt :]


def lifetime_epsilon(E_j: float, new_task_id: int, old_task_id: int) -> float:
    """Telescoping lifetime schedule E_j / [r(r+1)], r=n-j >= 1."""
    r = int(new_task_id - old_task_id)
    if r <= 0:
        raise ValueError("new_task_id must be strictly greater than old_task_id")
    return float(E_j) / float(r * (r + 1))


def finite_uniform_beta(cfg: RouterConfig) -> float:
    """
    Bonferroni allocation over the finite task sequence supplied to this run.

    Checks = all old-risk certificates + one new-power certificate for each task n>0.
    Their total failure probability is <= lifetime_confidence_budget.
    """
    K = len(cfg.tasks)
    old_checks = K * (K - 1) // 2
    new_checks = max(K - 1, 0)
    total = max(old_checks + new_checks, 1)
    return float(cfg.lifetime_confidence_budget) / float(total)


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------


def _load_split(dataset_name: str, split: str) -> Dataset:
    return load_dataset(dataset_name, split=split)


def _limit_dataset(dataset: Dataset, max_samples: int, seed: int = 0) -> Dataset:
    if max_samples is None or max_samples == -1:
        return dataset.shuffle(seed=seed)
    max_samples = min(max_samples, len(dataset))
    if max_samples < 0:
        raise ValueError(
            f"max_samples must be -1 or non-negative, got {max_samples}"
        )
    return dataset.shuffle(seed=seed).select(range(max_samples))


def _load_training_dataset(
    dataset_name: str,
    language: str,
    max_train_samples: int,
    seed: int = 0,
) -> Dataset:
    split_datasets = []
    for split in ["train_OSS_Instruct", "train_McEval_Instruct"]:
        dataset = _load_split(dataset_name, split)
        dataset = dataset.filter(
            lambda row: row["language"] == language and row["solution"] is not None
        )
        split_datasets.append(dataset)

    if not split_datasets:
        raise ValueError("No training splits were loaded.")

    train_dataset = (
        split_datasets[0]
        if len(split_datasets) == 1
        else concatenate_datasets(split_datasets)
    )
    train_dataset = _limit_dataset(train_dataset, max_train_samples, seed)
    dataset = train_dataset.remove_columns(
        [
            c
            for c in train_dataset.column_names
            if c not in ("instruction", "solution")
        ]
    )
    dataset = dataset.rename_column("instruction", "prompt")
    dataset = dataset.rename_column("solution", "answer")
    if len(dataset) > 0:
        print("[train] Sample:")
        print(json.dumps(dataset[0], ensure_ascii=False, indent=2))
    return dataset


def _load_eval_dataset(
    dataset_name: str,
    language: str,
    max_eval_samples: int,
    seed: int = 0,
) -> Dataset:
    dataset = _load_split(dataset_name, "test_McEval")
    dataset = dataset.filter(
        lambda row: row["language"] == language and row["test"] is not None
    )
    dataset = _limit_dataset(dataset, max_eval_samples, seed)
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in ("instruction", "solution")]
    )
    dataset = dataset.rename_column("instruction", "prompt")
    dataset = dataset.rename_column("solution", "answer")
    if len(dataset) == 0:
        raise ValueError(
            f"No evaluation samples found in split=test_McEval for language={language}."
        )
    print("[eval] Sample:")
    print(json.dumps(dataset[0], ensure_ascii=False, indent=2))
    return dataset


def build_dataloader(
    tokenizer,
    task: str,
    split: str,
    batch_size: int,
    k: int,
    seed: int,
    max_length: int,
):
    """Load CODETASK_with_instruction_pool, matching the original router code."""
    hf_split = "validation" if split in {"eval", "validation"} else split
    dataset = load_dataset(
        "dongg18/CODETASK_with_instruction_pool",
        data_files={hf_split: f"{task}/{hf_split}-*.parquet"},
        split=hf_split,
    )
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in ("input", "output")]
    )
    dataset = dataset.rename_column("input", "prompt")
    dataset = dataset.rename_column("output", "answer")

    if k != -1:
        dataset = dataset.shuffle(seed=seed).select(range(min(k, len(dataset))))
    else:
        dataset = dataset.shuffle(seed=seed)

    def preprocess_batch(examples):
        src_texts = [str(t).strip() for t in examples["prompt"]]
        return tokenizer(
            src_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    enc = dataset.map(preprocess_batch, batched=True, remove_columns=dataset.column_names)
    enc.set_format(type="torch", columns=["input_ids", "attention_mask"])

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    return DataLoader(
        enc,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=lambda batch: router_pad_collate(
            batch, pad_token_id=pad_token_id, label_pad_id=-100
        ),
    )


def build_executable_dataloader(
    tokenizer,
    dataset_name: str,
    language: str,
    split: str,
    batch_size: int,
    k: int,
    seed: int,
    max_length: int,
):
    if split == "train":
        dataset = _load_training_dataset(dataset_name, language, k, seed)
    elif split in {"validation", "eval", "test"}:
        dataset = _load_eval_dataset(dataset_name, language, k, seed)
    else:
        raise ValueError(f"Unknown executable split: {split}")

    def preprocess_batch(examples):
        src_texts = [str(t).strip() for t in examples["prompt"]]
        return tokenizer(
            src_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    enc = dataset.map(preprocess_batch, batched=True, remove_columns=dataset.column_names)
    enc.set_format(type="torch", columns=["input_ids", "attention_mask"])

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    return DataLoader(
        enc,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=lambda batch: router_pad_collate(
            batch, pad_token_id=pad_token_id, label_pad_id=-100
        ),
    )


# -----------------------------------------------------------------------------
# Frozen routing features
# -----------------------------------------------------------------------------


class RoutingFeatureExtractor:
    """Frozen T5/encoder feature extractor with one fixed orthonormal projection."""

    def __init__(
        self,
        model_name: str,
        feature_layers: int,
        routing_dim: int,
        device: torch.device,
        seed: int,
    ):
        self.model_name = model_name
        self.feature_layers = feature_layers
        self.routing_dim = routing_dim
        self.device = device
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        hidden_size = getattr(self.model.config, "d_model", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(f"Cannot infer hidden size from model config for {model_name}")

        self.P = self._make_row_orthonormal_projection(
            p=routing_dim,
            d=hidden_size,
            seed=seed,
            device=device,
        )

    @staticmethod
    def _make_row_orthonormal_projection(
        p: int,
        d: int,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        if p > d:
            raise ValueError(f"routing_dim p={p} must be <= hidden_size d={d}")
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        A = torch.randn(d, p, generator=gen) / math.sqrt(p)
        Q, _ = torch.linalg.qr(A, mode="reduced")
        return Q.T.contiguous().to(device)

    def save_projection(self, output_dir: str | Path) -> None:
        output_dir = ensure_dir(output_dir)
        torch.save(self.P.detach().cpu(), output_dir / "projection_P.pt")

    def load_projection(self, path: str | Path) -> None:
        self.P = torch.load(path, map_location=self.device).to(self.device)

    @torch.no_grad()
    def encode_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        if hasattr(self.model, "encoder"):
            enc = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            enc = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        layer_idx = min(self.feature_layers, len(enc.hidden_states) - 1)
        H = enc.hidden_states[layer_idx]  # [B,T,D]
        mask = attention_mask.unsqueeze(-1).to(H.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (H * mask).sum(dim=1) / denom

        # Affine-free LayerNorm.
        h = F.layer_norm(pooled.float(), normalized_shape=(pooled.shape[-1],))
        z = h @ self.P.T.float()
        return z.detach().cpu().float()

    @torch.no_grad()
    def extract_features(
        self,
        dataloader: Iterable[Dict[str, torch.Tensor]],
        desc: str,
    ) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for batch in tqdm(dataloader, desc=desc):
            chunks.append(self.encode_batch(batch))
        if not chunks:
            raise RuntimeError(f"No features extracted for {desc}")
        return torch.cat(chunks, dim=0).float()


# -----------------------------------------------------------------------------
# Diagonal GMM anchor
# -----------------------------------------------------------------------------


@dataclass
class DiagonalGMMState:
    pi: torch.Tensor
    mu: torch.Tensor
    var: torch.Tensor


class WeightedDiagonalGMM:
    def __init__(
        self,
        n_components: int,
        variance_floor: float = 1e-4,
        eps: float = 1e-8,
    ):
        self.n_components = n_components
        self.variance_floor = variance_floor
        self.eps = eps
        self.state: Optional[DiagonalGMMState] = None

    @staticmethod
    def _log_diag_gaussian(
        z: torch.Tensor,
        mu: torch.Tensor,
        var: torch.Tensor,
    ) -> torch.Tensor:
        z_exp = z[:, None, :]
        mu_exp = mu[None, :, :]
        var_exp = var[None, :, :]
        log_det = torch.log(var_exp).sum(dim=-1)
        quad = ((z_exp - mu_exp) ** 2 / var_exp).sum(dim=-1)
        p = z.shape[-1]
        return -0.5 * (p * math.log(2.0 * math.pi) + log_det + quad)

    def _init_params(
        self,
        z: torch.Tensor,
        weights: torch.Tensor,
        seed: int,
    ) -> DiagonalGMMState:
        N, _ = z.shape
        M = min(self.n_components, N)
        if M < self.n_components:
            print(
                f"[warn] n_components reduced from {self.n_components} to {M} because N={N}"
            )
            self.n_components = M

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        prob = weights.clamp_min(self.eps)
        prob = prob / prob.sum()
        idx = torch.multinomial(
            prob,
            num_samples=M,
            replacement=False,
            generator=gen,
        )
        mu = z[idx].clone()
        global_var = torch.var(z, dim=0, unbiased=False).clamp_min(
            self.variance_floor
        )
        var = global_var.unsqueeze(0).repeat(M, 1).clone()
        pi = torch.ones(M, dtype=z.dtype) / M
        return DiagonalGMMState(pi=pi, mu=mu, var=var)

    def fit(
        self,
        z: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        em_iters: int = 50,
        tol: float = 1e-4,
        seed: int = 42,
    ) -> "WeightedDiagonalGMM":
        z = z.float().cpu()
        N, _ = z.shape
        if N == 0:
            raise ValueError("Cannot fit GMM with zero samples")

        if sample_weights is None:
            w = torch.ones(N, dtype=torch.float32)
        else:
            w = sample_weights.float().cpu().clamp_min(self.eps)

        state = self._init_params(z, w, seed=seed)
        prev_ll = None

        for _ in range(em_iters):
            log_prob_comp = self._log_diag_gaussian(z, state.mu, state.var)
            log_joint = (
                torch.log(state.pi.clamp_min(self.eps))[None, :] + log_prob_comp
            )
            log_norm = torch.logsumexp(log_joint, dim=1)
            resp = torch.exp(log_joint - log_norm[:, None])

            wr = w[:, None] * resp
            Nk = wr.sum(dim=0).clamp_min(self.eps)
            omega = w.sum().clamp_min(self.eps)

            pi = Nk / omega
            mu = (wr.T @ z) / Nk[:, None]
            diff = z[:, None, :] - mu[None, :, :]
            var = (wr[:, :, None] * diff.pow(2)).sum(dim=0) / Nk[:, None]
            var = var.clamp_min(self.variance_floor)
            state = DiagonalGMMState(pi=pi, mu=mu, var=var)

            weighted_ll = (w * log_norm).sum() / omega
            if prev_ll is not None and abs(float(weighted_ll - prev_ll)) < tol:
                break
            prev_ll = weighted_ll.detach()

        self.state = state
        return self

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("GMM is not fitted")
        z = z.float().cpu()
        log_prob_comp = self._log_diag_gaussian(z, self.state.mu, self.state.var)
        log_joint = (
            torch.log(self.state.pi.clamp_min(self.eps))[None, :] + log_prob_comp
        )
        return torch.logsumexp(log_joint, dim=1)

    def sample(self, n: int, seed: int) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("GMM is not fitted")
        if n <= 0:
            return torch.empty((0, self.state.mu.shape[1]), dtype=torch.float32)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        comp = torch.multinomial(
            self.state.pi.clamp_min(self.eps),
            num_samples=n,
            replacement=True,
            generator=gen,
        )
        mu = self.state.mu[comp]
        std = torch.sqrt(self.state.var[comp].clamp_min(self.variance_floor))
        noise = torch.randn(mu.shape, generator=gen, dtype=mu.dtype)
        return (mu + std * noise).float()

    def to_dict(self) -> Dict[str, torch.Tensor]:
        if self.state is None:
            raise RuntimeError("GMM is not fitted")
        return {
            "pi": self.state.pi,
            "mu": self.state.mu,
            "var": self.state.var,
            "n_components": torch.tensor(self.n_components),
            "variance_floor": torch.tensor(self.variance_floor),
            "eps": torch.tensor(self.eps),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, torch.Tensor]) -> "WeightedDiagonalGMM":
        obj = cls(
            n_components=int(d["n_components"]),
            variance_floor=float(d["variance_floor"]),
            eps=float(d["eps"]),
        )
        obj.state = DiagonalGMMState(
            pi=d["pi"].float(),
            mu=d["mu"].float(),
            var=d["var"].float(),
        )
        return obj


# -----------------------------------------------------------------------------
# Finite-sample certificates
# -----------------------------------------------------------------------------


def one_sided_upper_bound(
    successes: int,
    total: int,
    beta: float,
    method: str,
) -> float:
    if total <= 0:
        raise ValueError("total must be positive")
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be in (0,1)")

    phat = successes / total
    if method == "hoeffding":
        radius = math.sqrt(math.log(1.0 / beta) / (2.0 * total))
        return min(1.0, phat + radius)
    if method == "clopper_pearson":
        if successes >= total:
            return 1.0
        return float(beta_dist.ppf(1.0 - beta, successes + 1, total - successes))
    raise ValueError(f"Unknown certificate method: {method}")



def minimum_zero_hit_samples_for_upper_bound(epsilon: float, beta: float, method: str) -> int:
    """
    Smallest approximate sample size such that a zero-hit one-sided upper bound can
    fall at or below epsilon. This is used only to size pseudo-certification draws.
    """
    epsilon = float(epsilon)
    beta = float(beta)
    if epsilon <= 0.0:
        return 10**18
    if epsilon >= 1.0:
        return 1
    if method == "clopper_pearson":
        # For k=0, CP upper bound is 1 - beta^(1/m).
        denom = math.log1p(-epsilon)
        return max(1, int(math.ceil(math.log(beta) / denom)))
    if method == "hoeffding":
        return max(1, int(math.ceil(math.log(1.0 / beta) / (2.0 * epsilon * epsilon))))
    raise ValueError(f"Unknown certificate method: {method}")


def one_sided_lower_bound(
    successes: int,
    total: int,
    beta: float,
    method: str,
) -> float:
    if total <= 0:
        raise ValueError("total must be positive")
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be in (0,1)")

    phat = successes / total
    if method == "hoeffding":
        radius = math.sqrt(math.log(1.0 / beta) / (2.0 * total))
        return max(0.0, phat - radius)
    if method == "clopper_pearson":
        if successes <= 0:
            return 0.0
        return float(beta_dist.ppf(beta, successes, total - successes + 1))
    raise ValueError(f"Unknown certificate method: {method}")


# -----------------------------------------------------------------------------
# Admission gates
# -----------------------------------------------------------------------------


@dataclass
class AdmissionGateState:
    mode: str
    old_task_ids: List[int]
    epsilon_by_old: Dict[int, float]
    admitted: bool = True

    # dual gate: p_new(z) > sum_j lambda_j p_j(z)
    lambdas: Optional[torch.Tensor] = None

    # pairwise gate: log p_new - log p_j > threshold_j for every old j
    pairwise_thresholds: Optional[torch.Tensor] = None

    optimization: Dict[str, object] = field(default_factory=dict)
    certification: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "old_task_ids": list(self.old_task_ids),
            "epsilon_by_old": {str(k): float(v) for k, v in self.epsilon_by_old.items()},
            "admitted": bool(self.admitted),
            "lambdas": None if self.lambdas is None else self.lambdas.cpu(),
            "pairwise_thresholds": (
                None
                if self.pairwise_thresholds is None
                else self.pairwise_thresholds.cpu()
            ),
            "optimization": self.optimization,
            "certification": self.certification,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "AdmissionGateState":
        return cls(
            mode=str(d["mode"]),
            old_task_ids=[int(x) for x in d.get("old_task_ids", [])],
            epsilon_by_old={
                int(k): float(v) for k, v in d.get("epsilon_by_old", {}).items()
            },
            admitted=bool(d.get("admitted", True)),
            lambdas=(
                None
                if d.get("lambdas") is None
                else torch.as_tensor(d["lambdas"]).float()
            ),
            pairwise_thresholds=(
                None
                if d.get("pairwise_thresholds") is None
                else torch.as_tensor(d["pairwise_thresholds"]).float()
            ),
            optimization=dict(d.get("optimization", {})),
            certification=dict(d.get("certification", {})),
        )


@dataclass
class TaskRouter:
    task_name: str
    task_id: int
    gmm: WeightedDiagonalGMM
    gate: Optional[AdmissionGateState] = None

    @property
    def admitted(self) -> bool:
        if self.task_id == 0:
            return True
        return self.gate is not None and self.gate.admitted


@dataclass
class AdmissionResult:
    task_id: int
    task_name: str
    admitted: bool
    epsilon_by_old: Dict[int, float]
    old_risk_ucb: Dict[int, float]
    old_risk_empirical: Dict[int, float]
    new_admission_empirical: float
    new_admission_lcb: float
    new_admission_opt: float
    beta_per_check: float
    optimizer_success: bool
    optimizer_message: str

    def to_jsonable(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "admitted": self.admitted,
            "epsilon_by_old": {str(k): v for k, v in self.epsilon_by_old.items()},
            "old_risk_ucb": {str(k): v for k, v in self.old_risk_ucb.items()},
            "old_risk_empirical": {
                str(k): v for k, v in self.old_risk_empirical.items()
            },
            "new_admission_empirical": self.new_admission_empirical,
            "new_admission_lcb": self.new_admission_lcb,
            "new_admission_opt": self.new_admission_opt,
            "beta_per_check": self.beta_per_check,
            "optimizer_success": self.optimizer_success,
            "optimizer_message": self.optimizer_message,
        }


# -----------------------------------------------------------------------------
# RC-GMM router
# -----------------------------------------------------------------------------


class RCGMMRouter:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self.tasks: List[TaskRouter] = []

    # ------------------------------ scores ----------------------------------

    def _all_log_probs(self, z: torch.Tensor) -> torch.Tensor:
        if not self.tasks:
            raise RuntimeError("No tasks available")
        return torch.stack([tr.gmm.log_prob(z) for tr in self.tasks], dim=1)

    def _task_by_id(self, task_id: int) -> TaskRouter:
        for tr in self.tasks:
            if tr.task_id == task_id:
                return tr
        raise KeyError(f"Unknown task_id={task_id}")

    # ------------------------------ gates -----------------------------------

    @staticmethod
    def _dual_gate_eval_from_logps(
        new_logp: torch.Tensor,
        old_logps: torch.Tensor,
        lambdas: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate 1{p_new > sum_j lambda_j p_j} in log-space."""
        if old_logps.ndim != 2:
            raise ValueError("old_logps must be [N,K]")
        if old_logps.shape[1] == 0:
            return torch.ones_like(new_logp, dtype=torch.bool)

        lam = lambdas.float().cpu().clamp_min(0.0)
        log_lam = torch.full_like(lam, float("-inf"))
        positive = lam > 0
        log_lam[positive] = torch.log(lam[positive])
        rhs_log = torch.logsumexp(old_logps + log_lam[None, :], dim=1)
        return new_logp > rhs_log

    def _evaluate_gate_for_task(self, tr: TaskRouter, z: torch.Tensor) -> torch.Tensor:
        if tr.task_id == 0:
            return torch.ones(len(z), dtype=torch.bool)
        if tr.gate is None or not tr.gate.admitted:
            return torch.zeros(len(z), dtype=torch.bool)

        gate = tr.gate
        old_tasks = [self._task_by_id(i) for i in gate.old_task_ids]
        new_logp = tr.gmm.log_prob(z)

        if gate.mode == "dual":
            old_logps = torch.stack([x.gmm.log_prob(z) for x in old_tasks], dim=1)
            return self._dual_gate_eval_from_logps(
                new_logp,
                old_logps,
                gate.lambdas if gate.lambdas is not None else torch.zeros(len(old_tasks)),
            )

        if gate.mode == "pairwise":
            thresholds = gate.pairwise_thresholds
            if thresholds is None:
                raise RuntimeError("Pairwise gate missing thresholds")
            opens = torch.ones(len(z), dtype=torch.bool)
            for idx, old in enumerate(old_tasks):
                ratio = new_logp - old.gmm.log_prob(z)
                opens &= ratio > float(thresholds[idx])
            return opens

        if gate.mode == "none":
            return torch.ones(len(z), dtype=torch.bool)

        raise ValueError(f"Unknown gate mode: {gate.mode}")

    def eligibility_mask(self, z: torch.Tensor) -> torch.Tensor:
        """[N,K] support mask. Task 0 is always a fallback expert."""
        if not self.tasks:
            raise RuntimeError("No tasks available")
        masks = [self._evaluate_gate_for_task(tr, z) for tr in self.tasks]
        return torch.stack(masks, dim=1)

    # ---------------------------- prediction --------------------------------

    def predict_scores(self, z: torch.Tensor) -> torch.Tensor:
        logp = self._all_log_probs(z)
        mask = self.eligibility_mask(z)
        scores = logp.clone()
        scores[~mask] = float("-inf")
        return scores

    def predict_weights(self, z: torch.Tensor, temperature: Optional[float] = None) -> torch.Tensor:
        """Soft expert weights for parameter-space LoRA composition."""
        T = float(self.cfg.temperature if temperature is None else temperature)
        if T <= 0:
            raise ValueError("temperature must be > 0")
        scores = self.predict_scores(z)
        return torch.softmax(scores / T, dim=1)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        scores = self.predict_scores(z)
        pred_local = scores.argmax(dim=1)
        task_ids = torch.tensor([tr.task_id for tr in self.tasks], dtype=torch.long)
        return task_ids[pred_local]

    # -------------------------- gate optimization ----------------------------

    def _fit_dual_gate(
        self,
        new_gmm: WeightedDiagonalGMM,
        old_tasks: Sequence[TaskRouter],
        epsilon_by_old: Dict[int, float],
        seed: int,
    ) -> Tuple[AdmissionGateState, bool, str]:
        K = len(old_tasks)
        if K == 0:
            return (
                AdmissionGateState(
                    mode="dual",
                    old_task_ids=[],
                    epsilon_by_old={},
                    lambdas=torch.empty(0),
                    admitted=True,
                ),
                True,
                "no old tasks",
            )

        # Importance-sampling proposal q = average(A_new, A_1, ..., A_K).
        densities = [tr.gmm for tr in old_tasks] + [new_gmm]
        per_density = int(self.cfg.gate_opt_samples_per_density)
        if per_density <= 0:
            raise ValueError("gate_opt_samples_per_density must be positive")

        samples = []
        groups: List[Tuple[int, int]] = []
        start = 0
        for idx, gmm in enumerate(densities):
            s = gmm.sample(per_density, seed + 1009 * (idx + 1))
            samples.append(s)
            groups.append((start, start + len(s)))
            start += len(s)
        zq = torch.cat(samples, dim=0)

        # log p_i(z), i = old tasks then new task.
        logps = torch.stack([g.log_prob(zq) for g in densities], dim=1).double()
        logq = torch.logsumexp(logps, dim=1) - math.log(len(densities))
        ratios = torch.exp(logps - logq[:, None]).cpu().numpy().astype(np.float64)
        r_old = ratios[:, :K]
        r_new = ratios[:, K]

        eps_target = np.array(
            [epsilon_by_old[tr.task_id] for tr in old_tasks], dtype=np.float64
        )
        eps_opt = np.clip(
            eps_target * float(self.cfg.optimization_budget_fraction),
            1e-12,
            1.0,
        )

        def objective_and_grad(lam: np.ndarray):
            lam = np.maximum(lam, 0.0)
            h = r_new - r_old.dot(lam)
            active = h > 0.0
            obj = float(np.dot(lam, eps_opt) + np.maximum(h, 0.0).mean())
            if np.any(active):
                grad = eps_opt - r_old[active].sum(axis=0) / len(h)
            else:
                grad = eps_opt.copy()
            return obj, grad.astype(np.float64)

        x0 = np.ones(K, dtype=np.float64) / max(K, 1)
        result = minimize(
            fun=lambda x: objective_and_grad(x)[0],
            x0=x0,
            jac=lambda x: objective_and_grad(x)[1],
            method="L-BFGS-B",
            bounds=[(0.0, None)] * K,
            options={
                "maxiter": int(self.cfg.dual_maxiter),
                "ftol": float(self.cfg.dual_ftol),
                "maxls": 50,
            },
        )
        lam = np.maximum(result.x, 0.0)
        lambdas = torch.tensor(lam, dtype=torch.float32)

        # Optimization-sample diagnostics by source density.
        old_risk_opt = {}
        for j_idx, tr in enumerate(old_tasks):
            a, b = groups[j_idx]
            z_old = zq[a:b]
            new_lp = new_gmm.log_prob(z_old)
            old_lps = torch.stack([x.gmm.log_prob(z_old) for x in old_tasks], dim=1)
            opens = self._dual_gate_eval_from_logps(new_lp, old_lps, lambdas)
            old_risk_opt[tr.task_id] = float(opens.float().mean())

        gate = AdmissionGateState(
            mode="dual",
            old_task_ids=[tr.task_id for tr in old_tasks],
            epsilon_by_old=epsilon_by_old,
            lambdas=lambdas,
            admitted=True,
            optimization={
                "dual_objective": float(result.fun),
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "old_risk_opt": {str(k): v for k, v in old_risk_opt.items()},
                "lambda": lambdas.tolist(),
                "optimization_budget_fraction": float(
                    self.cfg.optimization_budget_fraction
                ),
            },
        )
        return gate, bool(result.success), str(result.message)

    def _fit_pairwise_gate(
        self,
        new_gmm: WeightedDiagonalGMM,
        old_tasks: Sequence[TaskRouter],
        epsilon_by_old: Dict[int, float],
        seed: int,
    ) -> Tuple[AdmissionGateState, bool, str]:
        thresholds = []
        risk_opt = {}
        m = int(self.cfg.gate_opt_samples_per_density)

        for idx, old in enumerate(old_tasks):
            z_old = old.gmm.sample(m, seed + 1237 * (idx + 1))
            ratio = new_gmm.log_prob(z_old) - old.gmm.log_prob(z_old)
            eps_target = epsilon_by_old[old.task_id] * float(
                self.cfg.optimization_budget_fraction
            )
            eps_target = min(max(eps_target, 0.0), 1.0)

            # Pick a conservative empirical upper-tail threshold. Gate uses strict >.
            sorted_ratio, _ = torch.sort(ratio)
            allowed = int(math.floor(eps_target * m))
            if allowed <= 0:
                tau = float(sorted_ratio[-1].item())
            elif allowed >= m:
                tau = float("-inf")
            else:
                tau = float(sorted_ratio[m - allowed - 1].item())
            thresholds.append(tau)

            # Pairwise risk alone. The full intersection can only be smaller.
            risk_opt[old.task_id] = float((ratio > tau).float().mean())

        gate = AdmissionGateState(
            mode="pairwise",
            old_task_ids=[tr.task_id for tr in old_tasks],
            epsilon_by_old=epsilon_by_old,
            pairwise_thresholds=torch.tensor(thresholds, dtype=torch.float32),
            admitted=True,
            optimization={
                "old_pairwise_risk_opt": {
                    str(k): v for k, v in risk_opt.items()
                },
                "thresholds": thresholds,
            },
        )
        return gate, True, "pairwise thresholds calibrated"

    def _gate_on_external_gmms(
        self,
        gate: AdmissionGateState,
        new_gmm: WeightedDiagonalGMM,
        old_tasks: Sequence[TaskRouter],
        z: torch.Tensor,
    ) -> torch.Tensor:
        if gate.mode == "none":
            return torch.ones(len(z), dtype=torch.bool)
        new_lp = new_gmm.log_prob(z)
        if gate.mode == "dual":
            old_lps = torch.stack([x.gmm.log_prob(z) for x in old_tasks], dim=1)
            return self._dual_gate_eval_from_logps(
                new_lp,
                old_lps,
                gate.lambdas if gate.lambdas is not None else torch.zeros(len(old_tasks)),
            )
        if gate.mode == "pairwise":
            thresholds = gate.pairwise_thresholds
            if thresholds is None:
                raise RuntimeError("Pairwise gate missing thresholds")
            opens = torch.ones(len(z), dtype=torch.bool)
            for idx, old in enumerate(old_tasks):
                opens &= (new_lp - old.gmm.log_prob(z)) > float(thresholds[idx])
            return opens
        raise ValueError(f"Unknown gate mode {gate.mode}")

    # --------------------------- certification ------------------------------

    def _certify_gate(
        self,
        gate: AdmissionGateState,
        new_gmm: WeightedDiagonalGMM,
        old_tasks: Sequence[TaskRouter],
        z_opt_new: torch.Tensor,
        z_cert_new: torch.Tensor,
        beta_per_check: float,
        seed: int,
    ) -> AdmissionResult:
        old_ucb: Dict[int, float] = {}
        old_emp: Dict[int, float] = {}

        cert_sample_counts: Dict[int, int] = {}
        for idx, old in enumerate(old_tasks):
            budget = float(gate.epsilon_by_old[old.task_id])
            total_target = int(self.cfg.gate_cert_samples_per_old)
            if self.cfg.auto_scale_cert_samples:
                needed = minimum_zero_hit_samples_for_upper_bound(
                    budget, beta_per_check, self.cfg.cert_bound
                )
                total_target = max(total_target, needed)
            if total_target > int(self.cfg.max_gate_cert_samples_per_old):
                print(
                    f"[warn] certification for old task {old.task_id} would need about "
                    f"{total_target} pseudo-samples to resolve budget={budget:.3e} even at zero hits; "
                    f"capping at {self.cfg.max_gate_cert_samples_per_old}. The certificate may be impossible to pass."
                )
                total_target = int(self.cfg.max_gate_cert_samples_per_old)

            successes = 0
            produced = 0
            batch_idx = 0
            while produced < total_target:
                bsz = min(int(self.cfg.cert_batch_size), total_target - produced)
                z_cert_old = old.gmm.sample(
                    bsz,
                    seed + 2003 * (idx + 1) + 1000003 * batch_idx,
                )
                opens = self._gate_on_external_gmms(
                    gate, new_gmm, old_tasks, z_cert_old
                )
                successes += int(opens.sum().item())
                produced += bsz
                batch_idx += 1

            total = produced
            cert_sample_counts[old.task_id] = total
            old_emp[old.task_id] = successes / max(total, 1)
            old_ucb[old.task_id] = one_sided_upper_bound(
                successes,
                total,
                beta_per_check,
                self.cfg.cert_bound,
            )

        # Independent real current-task certification split.
        opens_new = self._gate_on_external_gmms(
            gate, new_gmm, old_tasks, z_cert_new
        )
        new_hits = int(opens_new.sum().item())
        new_total = int(len(opens_new))
        new_emp = new_hits / max(new_total, 1)
        new_lcb = one_sided_lower_bound(
            new_hits,
            new_total,
            beta_per_check,
            self.cfg.cert_bound,
        )

        # Optimization-split new-domain power is diagnostic only.
        opens_opt = self._gate_on_external_gmms(gate, new_gmm, old_tasks, z_opt_new)
        new_opt = float(opens_opt.float().mean()) if len(opens_opt) > 0 else float("nan")

        risk_ok = all(
            old_ucb[old.task_id] <= gate.epsilon_by_old[old.task_id]
            for old in old_tasks
        )
        power_ok = new_lcb >= float(self.cfg.min_new_admission)
        admitted = bool(risk_ok and power_ok)

        gate.admitted = admitted
        gate.certification = {
            "bound": self.cfg.cert_bound,
            "beta_per_check": beta_per_check,
            "old_risk_empirical": {str(k): v for k, v in old_emp.items()},
            "old_risk_ucb": {str(k): v for k, v in old_ucb.items()},
            "old_cert_samples": {str(k): v for k, v in cert_sample_counts.items()},
            "new_admission_empirical": new_emp,
            "new_admission_lcb": new_lcb,
            "new_admission_opt": new_opt,
            "min_new_admission": float(self.cfg.min_new_admission),
            "risk_ok": risk_ok,
            "power_ok": power_ok,
        }

        opt_success = bool(gate.optimization.get("optimizer_success", True))
        opt_msg = str(gate.optimization.get("optimizer_message", "n/a"))
        return AdmissionResult(
            task_id=-1,
            task_name="",
            admitted=admitted,
            epsilon_by_old=gate.epsilon_by_old,
            old_risk_ucb=old_ucb,
            old_risk_empirical=old_emp,
            new_admission_empirical=new_emp,
            new_admission_lcb=new_lcb,
            new_admission_opt=new_opt,
            beta_per_check=beta_per_check,
            optimizer_success=opt_success,
            optimizer_message=opt_msg,
        )

    # ---------------------------- task addition -----------------------------

    def fit_new_task(
        self,
        task_name: str,
        task_id: int,
        z_train: torch.Tensor,
        beta_per_check: float,
    ) -> Tuple[TaskRouter, Optional[AdmissionResult]]:
        z_fit, z_opt, z_cert = split_task_features(
            z_train,
            fit_fraction=self.cfg.fit_fraction,
            opt_fraction=self.cfg.opt_fraction,
            seed=self.cfg.seed + 7919 * (task_id + 1),
        )

        new_gmm = WeightedDiagonalGMM(
            n_components=self.cfg.gmm_components,
            variance_floor=self.cfg.variance_floor,
            eps=self.cfg.eps,
        )
        new_gmm.fit(
            z_fit,
            em_iters=self.cfg.em_iters,
            tol=self.cfg.em_tol,
            seed=self.cfg.seed + task_id,
        )

        # Task 0 is the unconditional fallback expert.
        if task_id == 0:
            tr = TaskRouter(task_name=task_name, task_id=task_id, gmm=new_gmm, gate=None)
            self.tasks.append(tr)
            print(
                f"[fit] task={task_id}:{task_name} N={len(z_train)} "
                f"fit={len(z_fit)} opt={len(z_opt)} cert={len(z_cert)} fallback=True"
            )
            return tr, None

        old_tasks = list(self.tasks)
        epsilon_by_old = {
            old.task_id: lifetime_epsilon(
                self.cfg.lifetime_risk_budget,
                new_task_id=task_id,
                old_task_id=old.task_id,
            )
            for old in old_tasks
        }

        if self.cfg.gate_mode == "dual":
            gate, opt_success, opt_msg = self._fit_dual_gate(
                new_gmm,
                old_tasks,
                epsilon_by_old,
                seed=self.cfg.seed + 100_003 * (task_id + 1),
            )
        elif self.cfg.gate_mode == "pairwise":
            gate, opt_success, opt_msg = self._fit_pairwise_gate(
                new_gmm,
                old_tasks,
                epsilon_by_old,
                seed=self.cfg.seed + 100_003 * (task_id + 1),
            )
        elif self.cfg.gate_mode == "none":
            gate = AdmissionGateState(
                mode="none",
                old_task_ids=[x.task_id for x in old_tasks],
                epsilon_by_old=epsilon_by_old,
                admitted=True,
                optimization={"optimizer_success": True, "optimizer_message": "ungated baseline"},
            )
            # Baseline is intentionally not risk-certified; all future experts are exposed.
            admission = AdmissionResult(
                task_id=task_id,
                task_name=task_name,
                admitted=True,
                epsilon_by_old=epsilon_by_old,
                old_risk_ucb={x.task_id: 1.0 for x in old_tasks},
                old_risk_empirical={x.task_id: 1.0 for x in old_tasks},
                new_admission_empirical=1.0,
                new_admission_lcb=1.0,
                new_admission_opt=1.0,
                beta_per_check=beta_per_check,
                optimizer_success=True,
                optimizer_message="ungated baseline",
            )
            tr = TaskRouter(task_name=task_name, task_id=task_id, gmm=new_gmm, gate=gate)
            self.tasks.append(tr)
            return tr, admission
        else:
            raise ValueError(f"Unknown gate_mode={self.cfg.gate_mode}")

        admission = self._certify_gate(
            gate,
            new_gmm,
            old_tasks,
            z_opt_new=z_opt,
            z_cert_new=z_cert,
            beta_per_check=beta_per_check,
            seed=self.cfg.seed + 200_003 * (task_id + 1),
        )
        admission.task_id = task_id
        admission.task_name = task_name
        admission.optimizer_success = opt_success
        admission.optimizer_message = opt_msg

        tr = TaskRouter(task_name=task_name, task_id=task_id, gmm=new_gmm, gate=gate)
        self.tasks.append(tr)

        print(
            f"[fit] task={task_id}:{task_name} N={len(z_train)} "
            f"fit={len(z_fit)} opt={len(z_opt)} cert={len(z_cert)} "
            f"admitted={admission.admitted} new_LCB={admission.new_admission_lcb:.4f}"
        )
        for old in old_tasks:
            j = old.task_id
            print(
                f"  [risk] old={j}:{old.task_name:<18s} "
                f"emp={admission.old_risk_empirical[j]:.6f} "
                f"UCB={admission.old_risk_ucb[j]:.6f} "
                f"budget={admission.epsilon_by_old[j]:.6f}"
            )

        return tr, admission

    # ------------------------------- I/O ------------------------------------

    def save(self, output_dir: str | Path, step: int) -> None:
        output_dir = ensure_dir(output_dir)
        payload = {
            "cfg": asdict(self.cfg),
            "step": step,
            "tasks": [
                {
                    "task_name": tr.task_name,
                    "task_id": tr.task_id,
                    "gmm": tr.gmm.to_dict(),
                    "gate": None if tr.gate is None else tr.gate.to_dict(),
                }
                for tr in self.tasks
            ],
        }
        torch.save(payload, output_dir / f"router_step{step}.pt")

    @classmethod
    def load(cls, path: str | Path) -> "RCGMMRouter":
        payload = torch.load(path, map_location="cpu")
        known = set(RouterConfig.__dataclass_fields__.keys())
        cfg_dict = {k: v for k, v in payload["cfg"].items() if k in known}
        if "tasks" in cfg_dict and isinstance(cfg_dict["tasks"], list):
            cfg_dict["tasks"] = tuple(cfg_dict["tasks"])
        cfg = RouterConfig(**cfg_dict)
        router = cls(cfg)
        for item in payload["tasks"]:
            router.tasks.append(
                TaskRouter(
                    task_name=item["task_name"],
                    task_id=int(item["task_id"]),
                    gmm=WeightedDiagonalGMM.from_dict(item["gmm"]),
                    gate=(
                        None
                        if item.get("gate") is None
                        else AdmissionGateState.from_dict(item["gate"])
                    ),
                )
            )
        return router


# -----------------------------------------------------------------------------
# Feature cache
# -----------------------------------------------------------------------------


def feature_cache_path(
    output_dir: str | Path,
    dataset_source: str,
    task: str,
    split: str,
    k: int,
    feature_layers: int,
    routing_dim: int,
    model_name: str,
    projection_seed: int,
) -> Path:
    return (
        Path(output_dir)
        / "features"
        / (
            f"{safe_name(dataset_source)}_{safe_name(model_name)}_{safe_name(task)}_"
            f"{split}_k{k}_L{feature_layers}_p{routing_dim}_seed{projection_seed}.pt"
        )
    )


def get_or_extract_features(
    extractor: RoutingFeatureExtractor,
    cfg: RouterConfig,
    task: str,
    split: str,
    k: int,
) -> torch.Tensor:
    cache_root = cfg.feature_cache_dir or cfg.output_dir
    path = feature_cache_path(
        cache_root,
        cfg.dataset_source,
        task,
        split,
        k,
        cfg.feature_layers,
        cfg.routing_dim,
        cfg.model_name,
        cfg.seed,
    )
    ensure_dir(path.parent)

    if cfg.save_features and path.exists() and not cfg.force_recompute_features:
        return torch.load(path, map_location="cpu").float()

    if cfg.dataset_source == "codetask":
        dl = build_dataloader(
            tokenizer=extractor.tokenizer,
            task=task,
            split=split,
            batch_size=cfg.batch_size,
            k=k,
            seed=cfg.seed,
            max_length=cfg.max_length,
        )
    elif cfg.dataset_source == "executable":
        dl = build_executable_dataloader(
            tokenizer=extractor.tokenizer,
            dataset_name=cfg.executable_dataset_name,
            language=task,
            split=split,
            batch_size=cfg.batch_size,
            k=k,
            seed=cfg.seed,
            max_length=cfg.max_length,
        )
    else:
        raise ValueError(f"Unknown dataset_source: {cfg.dataset_source}")

    z = extractor.extract_features(dl, desc=f"extract {task}/{split}")
    if cfg.save_features:
        torch.save(z, path)
    return z


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


@dataclass
class EvalResult:
    overall_acc: float
    per_task_acc: Dict[str, float]
    confusion: List[List[int]]
    eligibility_rate: List[List[float]]
    self_admission: Dict[str, float]
    asr_avg: float
    asr_worst: float
    asr_worst_pair: Optional[Tuple[str, str]]


def evaluate_seen_tasks(
    router: RCGMMRouter,
    extractor: RoutingFeatureExtractor,
    cfg: RouterConfig,
    seen_tasks: List[str],
    split: str,
) -> EvalResult:
    K = len(seen_tasks)
    confusion = torch.zeros(K, K, dtype=torch.long)
    eligibility = torch.zeros(K, K, dtype=torch.float64)
    correct_total = 0
    n_total = 0
    per_task_acc: Dict[str, float] = {}
    self_admission: Dict[str, float] = {}
    pair_rates: List[Tuple[float, int, int]] = []

    for true_id, task in enumerate(seen_tasks):
        z = get_or_extract_features(
            extractor=extractor,
            cfg=cfg,
            task=task,
            split=split,
            k=cfg.eval_k,
        )
        pred = router.predict(z)
        mask = router.eligibility_mask(z)
        y = torch.full_like(pred, fill_value=true_id)

        correct = int((pred == y).sum().item())
        total = int(y.numel())
        correct_total += correct
        n_total += total
        per_task_acc[task] = correct / max(total, 1)

        for expert_idx in range(min(K, mask.shape[1])):
            rate = float(mask[:, expert_idx].float().mean())
            eligibility[true_id, expert_idx] = rate
            if expert_idx > true_id:
                pair_rates.append((rate, true_id, expert_idx))

        # Own gate acceptance. Task 0 is the unconditional fallback and therefore 1.
        if true_id < mask.shape[1]:
            self_admission[task] = float(mask[:, true_id].float().mean())
        else:
            self_admission[task] = 0.0

        for yt, yp in zip(y.tolist(), pred.tolist()):
            if 0 <= yt < K and 0 <= yp < K:
                confusion[yt, yp] += 1

    asr_avg = float(np.mean([x[0] for x in pair_rates])) if pair_rates else 0.0
    if pair_rates:
        worst_rate, j, n = max(pair_rates, key=lambda x: x[0])
        worst_pair = (seen_tasks[j], seen_tasks[n])
        asr_worst = float(worst_rate)
    else:
        worst_pair = None
        asr_worst = 0.0

    return EvalResult(
        overall_acc=correct_total / max(n_total, 1),
        per_task_acc=per_task_acc,
        confusion=confusion.tolist(),
        eligibility_rate=eligibility.tolist(),
        self_admission=self_admission,
        asr_avg=asr_avg,
        asr_worst=asr_worst,
        asr_worst_pair=worst_pair,
    )


def print_eval(step: int, seen_tasks: List[str], result: EvalResult) -> None:
    print("\n" + "=" * 100)
    print(f"[eval] step={step} seen_tasks={seen_tasks}")
    print(f"[eval] overall routing acc = {result.overall_acc:.4f}")
    print(f"[eval] ASR-avg   = {result.asr_avg:.6f}")
    print(f"[eval] ASR-worst = {result.asr_worst:.6f} pair={result.asr_worst_pair}")
    for task, acc in result.per_task_acc.items():
        print(
            f"  - {task:<18s}: acc={acc:.4f} "
            f"self-admission={result.self_admission.get(task, float('nan')):.4f}"
        )

    print("[eval] confusion rows=true, cols=pred")
    header = "true\\pred" + "".join(
        [f"\t{i}:{t[:8]}" for i, t in enumerate(seen_tasks)]
    )
    print(header)
    for i, row in enumerate(result.confusion):
        print(f"{i}:{seen_tasks[i][:8]}" + "".join([f"\t{v}" for v in row]))

    print("[eval] eligibility rows=true-task, cols=expert")
    print(header)
    for i, row in enumerate(result.eligibility_rate):
        print(
            f"{i}:{seen_tasks[i][:8]}"
            + "".join([f"\t{100.0 * v:.2f}%" for v in row])
        )
    print("=" * 100 + "\n")


def measure_old_hard_routing_change(
    router_before_preds: Dict[str, torch.Tensor],
    router_after: RCGMMRouter,
    extractor: RoutingFeatureExtractor,
    cfg: RouterConfig,
    old_tasks: Sequence[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for task in old_tasks:
        z = get_or_extract_features(
            extractor,
            cfg,
            task,
            split=cfg.eval_split,
            k=cfg.eval_k,
        )
        after = router_after.predict(z)
        before = router_before_preds[task]
        if len(before) != len(after):
            raise RuntimeError("Prediction length changed unexpectedly")
        out[task] = float((before != after).float().mean())
    return out


# -----------------------------------------------------------------------------
# Continual run
# -----------------------------------------------------------------------------


def run(cfg: RouterConfig, resume_from: Optional[str] = None) -> None:
    set_seed(cfg.seed)
    output_dir = ensure_dir(cfg.output_dir)

    if cfg.gate_mode not in {"dual", "pairwise", "none"}:
        raise ValueError("gate_mode must be one of: dual, pairwise, none")
    if cfg.cert_bound not in {"clopper_pearson", "hoeffding"}:
        raise ValueError("cert_bound must be clopper_pearson or hoeffding")
    if not (0.0 < cfg.optimization_budget_fraction <= 1.0):
        raise ValueError("optimization_budget_fraction must be in (0,1]")
    if not (0.0 <= cfg.lifetime_risk_budget <= 1.0):
        raise ValueError("lifetime_risk_budget must be in [0,1]")
    if not (0.0 <= cfg.min_new_admission <= 1.0):
        raise ValueError("min_new_admission must be in [0,1]")

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = get_device()
    beta_per_check = finite_uniform_beta(cfg)
    print(f"[setup] device={device}")
    print(f"[setup] dataset_source={cfg.dataset_source}")
    print(f"[setup] tasks={list(cfg.tasks)}")
    print(f"[setup] gate_mode={cfg.gate_mode}")
    print(f"[setup] lifetime_risk_budget={cfg.lifetime_risk_budget}")
    print(f"[setup] min_new_admission={cfg.min_new_admission}")
    print(
        f"[setup] cert_bound={cfg.cert_bound} "
        f"beta_per_check={beta_per_check:.3e} "
        f"(lifetime confidence <= {cfg.lifetime_confidence_budget})"
    )

    extractor = RoutingFeatureExtractor(
        model_name=cfg.model_name,
        feature_layers=cfg.feature_layers,
        routing_dim=cfg.routing_dim,
        device=device,
        seed=cfg.seed,
    )

    projection_path = output_dir / "projection_P.pt"
    if resume_from is not None and projection_path.exists():
        extractor.load_projection(projection_path)
        print(f"[resume] loaded frozen projection from {projection_path}")
    else:
        extractor.save_projection(output_dir)

    if resume_from is not None:
        print(f"[resume] loading router from {resume_from}")
        router = RCGMMRouter.load(resume_from)
        # Runtime CLI config is authoritative only for non-structural evaluation/I/O fields.
        # For safety, keep gate/model parameters from checkpoint.
        router.cfg.output_dir = cfg.output_dir
        router.cfg.feature_cache_dir = cfg.feature_cache_dir
        router.cfg.eval_k = cfg.eval_k
        router.cfg.eval_split = cfg.eval_split
        router.cfg.save_features = cfg.save_features
        router.cfg.force_recompute_features = cfg.force_recompute_features
        cfg = router.cfg
        start_task_id = len(router.tasks)
        print(f"[resume] resuming from task_id={start_task_id}")
    else:
        router = RCGMMRouter(cfg)
        start_task_id = 0

    results_path = output_dir / "routing_results.json"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            all_results: Dict[str, Dict] = json.load(f)
    else:
        all_results = {}

    admission_path = output_dir / "admission_results.json"
    if admission_path.exists():
        with open(admission_path, encoding="utf-8") as f:
            all_admissions: Dict[str, Dict] = json.load(f)
    else:
        all_admissions = {}

    for task_id, task in enumerate(cfg.tasks):
        if task_id < start_task_id:
            print(f"[continual] skip task {task_id}: {task} (already in checkpoint)")
            continue

        print("\n" + "#" * 100)
        print(f"[continual] learn task {task_id}: {task}")
        print("#" * 100)

        # Predictions immediately before adding the new expert, for actual hard-routing
        # change diagnostics on old held-out tasks.
        old_tasks = list(cfg.tasks[:task_id])
        before_preds: Dict[str, torch.Tensor] = {}
        if task_id > 0:
            for old_task in old_tasks:
                z_old_eval = get_or_extract_features(
                    extractor,
                    cfg,
                    old_task,
                    split=cfg.eval_split,
                    k=cfg.eval_k,
                )
                before_preds[old_task] = router.predict(z_old_eval)

        z_train = get_or_extract_features(
            extractor=extractor,
            cfg=cfg,
            task=task,
            split="train",
            k=cfg.train_k,
        )

        tr, admission = router.fit_new_task(
            task_name=task,
            task_id=task_id,
            z_train=z_train,
            beta_per_check=beta_per_check,
        )
        router.save(output_dir, step=task_id)

        if admission is not None:
            all_admissions[f"step{task_id}"] = admission.to_jsonable()
            with open(admission_path, "w", encoding="utf-8") as f:
                json.dump(all_admissions, f, indent=2)

        # Evaluate all tasks learned so far, including a task that failed safe admission.
        seen_tasks = list(cfg.tasks[: task_id + 1])
        result = evaluate_seen_tasks(
            router=router,
            extractor=extractor,
            cfg=cfg,
            seen_tasks=seen_tasks,
            split=cfg.eval_split,
        )
        print_eval(step=task_id, seen_tasks=seen_tasks, result=result)

        hard_change = {}
        if task_id > 0:
            hard_change = measure_old_hard_routing_change(
                before_preds,
                router,
                extractor,
                cfg,
                old_tasks,
            )
            print("[eval] actual old-task hard-routing change after this addition")
            for old_task, rate in hard_change.items():
                print(f"  - {old_task:<18s}: {rate:.6f}")

        all_results[f"step{task_id}"] = {
            "seen_tasks": seen_tasks,
            "overall_acc": result.overall_acc,
            "per_task_acc": result.per_task_acc,
            "confusion": result.confusion,
            "eligibility_rate": result.eligibility_rate,
            "self_admission": result.self_admission,
            "asr_avg": result.asr_avg,
            "asr_worst": result.asr_worst,
            "asr_worst_pair": result.asr_worst_pair,
            "old_hard_routing_change": hard_change,
            "safe_admitted": tr.admitted,
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        if admission is not None and not admission.admitted:
            msg = (
                f"Task {task_id}:{task} is UNSAFE under the current routing representation "
                f"and risk/power budgets. Its expert is stored but its gate is inactive."
            )
            print(f"[unsafe] {msg}")
            if not cfg.continue_on_unsafe:
                print(
                    "[unsafe] stopping strict continual run. Relax the lifetime risk budget, "
                    "lower min_new_admission, increase routing data/certification samples, "
                    "or improve the routing representation before retrying."
                )
                break

    print(f"[done] saved checkpoints/results to: {output_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RC-GMM risk-controlled continual density router"
    )
    p.add_argument("--model_name", type=str, default=RouterConfig.model_name)
    p.add_argument("--output_dir", type=str, default=RouterConfig.output_dir)
    p.add_argument(
        "--feature_cache_dir",
        type=str,
        default=None,
        help="Optional shared feature-cache root. Defaults to <output_dir>/features.",
    )
    p.add_argument(
        "--dataset_source",
        type=str,
        default=RouterConfig.dataset_source,
        choices=["codetask", "executable"],
    )
    p.add_argument(
        "--executable_dataset_name",
        type=str,
        default=RouterConfig.executable_dataset_name,
    )
    p.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task list. For executable data, tasks are language names.",
    )

    p.add_argument("--feature_layers", type=int, default=RouterConfig.feature_layers)
    p.add_argument("--routing_dim", type=int, default=RouterConfig.routing_dim)
    p.add_argument("--max_length", type=int, default=RouterConfig.max_length)
    p.add_argument("--batch_size", type=int, default=RouterConfig.batch_size)
    p.add_argument("--train_k", type=int, default=RouterConfig.train_k)
    p.add_argument("--eval_k", type=int, default=RouterConfig.eval_k)
    p.add_argument("--seed", type=int, default=RouterConfig.seed)

    p.add_argument("--gmm_components", type=int, default=RouterConfig.gmm_components)
    p.add_argument("--em_iters", type=int, default=RouterConfig.em_iters)
    p.add_argument("--em_tol", type=float, default=RouterConfig.em_tol)
    p.add_argument("--variance_floor", type=float, default=RouterConfig.variance_floor)
    p.add_argument("--eps", type=float, default=RouterConfig.eps)

    p.add_argument(
        "--gate_mode",
        type=str,
        default=RouterConfig.gate_mode,
        choices=["dual", "pairwise", "none"],
        help="dual=RC-GMM, pairwise=conservative ablation, none=ordinary GMM baseline",
    )
    p.add_argument(
        "--lifetime_risk_budget",
        type=float,
        default=RouterConfig.lifetime_risk_budget,
        help="Per-old-task lifetime perturbation budget E_j.",
    )
    p.add_argument(
        "--min_new_admission",
        type=float,
        default=RouterConfig.min_new_admission,
        help="Required one-sided lower confidence bound for P_new(G_new=1).",
    )
    p.add_argument("--fit_fraction", type=float, default=RouterConfig.fit_fraction)
    p.add_argument("--opt_fraction", type=float, default=RouterConfig.opt_fraction)
    p.add_argument(
        "--gate_opt_samples_per_density",
        type=int,
        default=RouterConfig.gate_opt_samples_per_density,
    )
    p.add_argument(
        "--gate_cert_samples_per_old",
        type=int,
        default=RouterConfig.gate_cert_samples_per_old,
        help="Minimum pseudo-certification samples per old anchor.",
    )
    p.add_argument(
        "--no_auto_scale_cert_samples",
        action="store_true",
        help="Disable automatic increase of pseudo-certification samples for small risk budgets.",
    )
    p.add_argument(
        "--max_gate_cert_samples_per_old",
        type=int,
        default=RouterConfig.max_gate_cert_samples_per_old,
    )
    p.add_argument(
        "--cert_batch_size",
        type=int,
        default=RouterConfig.cert_batch_size,
    )
    p.add_argument("--dual_maxiter", type=int, default=RouterConfig.dual_maxiter)
    p.add_argument("--dual_ftol", type=float, default=RouterConfig.dual_ftol)
    p.add_argument(
        "--optimization_budget_fraction",
        type=float,
        default=RouterConfig.optimization_budget_fraction,
    )
    p.add_argument(
        "--lifetime_confidence_budget",
        type=float,
        default=RouterConfig.lifetime_confidence_budget,
    )
    p.add_argument(
        "--cert_bound",
        type=str,
        default=RouterConfig.cert_bound,
        choices=["clopper_pearson", "hoeffding"],
    )
    p.add_argument("--temperature", type=float, default=RouterConfig.temperature)

    p.add_argument(
        "--eval_split",
        type=str,
        default=RouterConfig.eval_split,
        choices=["validation", "test"],
    )
    p.add_argument("--no_save_features", action="store_true")
    p.add_argument("--force_recompute_features", action="store_true")
    p.add_argument(
        "--continue_on_unsafe",
        action="store_true",
        help="Continue after a task fails safe admission. Strict RC-GMM should normally stop.",
    )
    p.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to router_stepN.pt checkpoint.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    default_cfg = RouterConfig()
    default_tasks = default_cfg.tasks if args.dataset_source == "codetask" else ("swift",)

    cfg = RouterConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        feature_cache_dir=args.feature_cache_dir,
        dataset_source=args.dataset_source,
        executable_dataset_name=args.executable_dataset_name,
        tasks=parse_tasks(args.tasks, default_tasks),
        feature_layers=args.feature_layers,
        routing_dim=args.routing_dim,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_k=args.train_k,
        eval_k=args.eval_k,
        seed=args.seed,
        gmm_components=args.gmm_components,
        em_iters=args.em_iters,
        em_tol=args.em_tol,
        variance_floor=args.variance_floor,
        eps=args.eps,
        gate_mode=args.gate_mode,
        lifetime_risk_budget=args.lifetime_risk_budget,
        min_new_admission=args.min_new_admission,
        fit_fraction=args.fit_fraction,
        opt_fraction=args.opt_fraction,
        gate_opt_samples_per_density=args.gate_opt_samples_per_density,
        gate_cert_samples_per_old=args.gate_cert_samples_per_old,
        auto_scale_cert_samples=not args.no_auto_scale_cert_samples,
        max_gate_cert_samples_per_old=args.max_gate_cert_samples_per_old,
        cert_batch_size=args.cert_batch_size,
        dual_maxiter=args.dual_maxiter,
        dual_ftol=args.dual_ftol,
        optimization_budget_fraction=args.optimization_budget_fraction,
        lifetime_confidence_budget=args.lifetime_confidence_budget,
        cert_bound=args.cert_bound,
        temperature=args.temperature,
        eval_split=args.eval_split,
        save_features=not args.no_save_features,
        force_recompute_features=args.force_recompute_features,
        continue_on_unsafe=args.continue_on_unsafe,
    )
    run(cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
