"""GMM traceback router.

Trains a per-task Gaussian Mixture Model whose input is the *error traceback*
(stderr) from failed code executions rather than the original instruction.

Data source: execution output files produced by infer_calibration_split.sh,
stored either locally or in a HuggingFace Hub repo.  Each file has the schema:

    {
      "metrics": {...},
      "predictions": [
        {
          "source": "...",
          "prediction": ["...", ...],
          "passed": [0, 1, ...],
          "stderr": ["Traceback ...", "", ...],
          ...
        }, ...
      ]
    }

Training mechanic is identical to gmm.py (ResidualFitGMMRouter with MAP scoring)
but the feature vectors come from encoding tracebacks, not prompts.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from gmm import (
    ResidualFitGMMRouter,
    RouterConfig,
    T5RoutingFeatureExtractor,
    ensure_dir,
    get_device,
    set_seed,
)

try:
    from huggingface_hub import hf_hub_download
    _HF_HUB = True
except ImportError:
    _HF_HUB = False


EXECUTABLE_LANGUAGES: Tuple[str, ...] = (
    "python", "cpp", "swift", "rust", "csharp", "java", "php", "typescript", "shell"
)


@dataclass
class TracebackRouterConfig:
    # Encoder / feature settings (mirrors RouterConfig fields used by ResidualFitGMMRouter)
    model_name: str = "Salesforce/codet5-small"
    output_dir: str = "./router_gmm_traceback_ckpt"
    tasks: Tuple[str, ...] = EXECUTABLE_LANGUAGES
    feature_layers: int = 4
    routing_dim: int = 128       # lower than code (256) — tracebacks occupy a smaller manifold
    max_length: int = 256        # most tracebacks fit; Java/C++ may truncate
    batch_size: int = 32
    seed: int = 42
    # GMM fitting
    gmm_components: int = 4     # ~4 error clusters: assertion, runtime, syntax, resource
    em_iters: int = 100         # more iterations compensate for smaller post-dedup datasets
    em_tol: float = 1e-4
    variance_floor: float = 1e-3  # higher than code (1e-4) to prevent component collapse
    eps: float = 1e-8
    save_features: bool = True
    force_recompute_features: bool = False
    # Traceback-specific
    results_source: str = "local"       # "local" or "hf_hub"
    results_dir: str = "./calibration_results"  # local dir or HF Hub repo ID
    results_repo_type: str = "dataset"  # "dataset", "model", or "space" (hf_hub only)
    truncate_side: str = "left"         # keep END of traceback (error line); "right" keeps start
    min_traceback_length: int = 10      # discard empty / near-empty stderr entries
    max_tracebacks: int = 0             # cap on deduped tracebacks per task (0 = unlimited)
    train_tracebacks: int = 500         # deduped samples for training; remainder used for eval (0 = all for training)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_calibration_file(cfg: TracebackRouterConfig, language: str) -> List[Dict]:
    """Return the predictions list from calibration_{language}.json."""
    filename = f"calibration_{language}.json"
    if cfg.results_source == "hf_hub":
        if not _HF_HUB:
            raise ImportError(
                "huggingface_hub is required for results_source='hf_hub'. "
                "Run: pip install huggingface-hub"
            )
        local_path = hf_hub_download(
            repo_id=cfg.results_dir,
            filename=filename,
            repo_type=cfg.results_repo_type,
        )
    else:
        local_path = os.path.join(cfg.results_dir, filename)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Calibration file not found: {local_path}")

    with open(local_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support bare list (old format) and {"metrics": ..., "predictions": [...]}
    if isinstance(data, list):
        return data
    return data.get("predictions", [])


# ---------------------------------------------------------------------------
# Traceback extraction and deduplication
# ---------------------------------------------------------------------------

def extract_tracebacks(predictions: List[Dict], min_length: int = 10) -> List[str]:
    """Return all non-trivial stderr strings from failed candidates."""
    out: List[str] = []
    for pred in predictions:
        passed = pred.get("passed", [])
        stderr = pred.get("stderr", [])
        for p, e in zip(passed, stderr):
            if p == 0 and isinstance(e, str) and len(e.strip()) >= min_length:
                out.append(e.strip())
    return out


def deduplicate_tracebacks(tracebacks: List[str]) -> List[str]:
    """Exact-string dedup, preserving first-occurrence order."""
    seen: Set[str] = set()
    result: List[str] = []
    for tb in tracebacks:
        if tb not in seen:
            seen.add(tb)
            result.append(tb)
    return result


def _error_type(traceback: str) -> str:
    last = traceback.split("\n")[-1].strip()
    return last.split(":")[0] if ":" in last else last[:50]


def print_traceback_stats(language: str, raw: List[str], deduped: List[str]) -> None:
    reduction = 100.0 * (1 - len(deduped) / max(len(raw), 1))
    print(f"\n[traceback stats] language={language}")
    print(f"  raw failed stderr entries : {len(raw)}")
    print(f"  unique after dedup        : {len(deduped)}")
    print(f"  dedup reduction           : {reduction:.1f}%")

    counts: Dict[str, int] = {}
    for tb in deduped:
        etype = _error_type(tb)
        counts[etype] = counts.get(etype, 0) + 1
    top = sorted(counts.items(), key=lambda x: -x[1])[:8]
    if top:
        print("  top error types (deduped):")
        for etype, cnt in top:
            print(f"    {cnt:4d}  {etype}")


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def _feature_cache_path(cfg: TracebackRouterConfig, task: str) -> Path:
    safe_src = cfg.results_dir.replace("/", "_")
    safe_task = task.replace("/", "_")
    trunc = cfg.truncate_side[0]
    name = (
        f"tb_{safe_src}_{safe_task}"
        f"_L{cfg.feature_layers}_p{cfg.routing_dim}"
        f"_ml{cfg.max_length}_{trunc}_min{cfg.min_traceback_length}.pt"
    )
    return Path(cfg.output_dir) / "features" / name


def encode_tracebacks(
    tracebacks: List[str],
    extractor: T5RoutingFeatureExtractor,
    cfg: TracebackRouterConfig,
) -> torch.Tensor:
    """Tokenize tracebacks and extract encoder features, returning shape [N, routing_dim]."""
    tokenizer = extractor.tokenizer
    original_trunc_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = cfg.truncate_side

    encoded = tokenizer(
        tracebacks,
        padding=True,
        truncation=True,
        max_length=cfg.max_length,
        return_tensors="pt",
    )
    tokenizer.truncation_side = original_trunc_side

    ds = TensorDataset(encoded["input_ids"], encoded["attention_mask"])
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    chunks: List[torch.Tensor] = []
    for input_ids, attention_mask in dl:
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        chunks.append(extractor.encode_batch(batch))

    return torch.cat(chunks, dim=0).float()


def _load_or_encode(
    task: str,
    tracebacks: List[str],
    extractor: T5RoutingFeatureExtractor,
    cfg: TracebackRouterConfig,
) -> Optional[torch.Tensor]:
    """Return encoded features for `tracebacks`, using cache when available."""
    if not tracebacks:
        return None
    cache = _feature_cache_path(cfg, task)
    ensure_dir(cache.parent)
    if cfg.save_features and cache.exists() and not cfg.force_recompute_features:
        print(f"[cache] Loading features: {cache}")
        return torch.load(cache, map_location="cpu").float()
    print(f"[encode] Encoding {len(tracebacks)} unique tracebacks for {task} ...")
    z = encode_tracebacks(tracebacks, extractor, cfg)
    if cfg.save_features:
        torch.save(z, cache)
        print(f"[cache] Saved features: {cache}")
    return z


# ---------------------------------------------------------------------------
# Bridge: TracebackRouterConfig → RouterConfig (for ResidualFitGMMRouter)
# ---------------------------------------------------------------------------

def _to_router_cfg(cfg: TracebackRouterConfig) -> RouterConfig:
    return RouterConfig(
        model_name=cfg.model_name,
        output_dir=cfg.output_dir,
        tasks=cfg.tasks,
        feature_layers=cfg.feature_layers,
        routing_dim=cfg.routing_dim,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        gmm_components=cfg.gmm_components,
        em_iters=cfg.em_iters,
        em_tol=cfg.em_tol,
        variance_floor=cfg.variance_floor,
        eps=cfg.eps,
        save_features=cfg.save_features,
        force_recompute_features=cfg.force_recompute_features,
    )


# ---------------------------------------------------------------------------
# Main training / evaluation loop
# ---------------------------------------------------------------------------

def run(cfg: TracebackRouterConfig) -> None:
    set_seed(cfg.seed)
    output_dir = ensure_dir(cfg.output_dir)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = get_device()
    print(f"[setup] device={device}")
    print(f"[setup] results_source={cfg.results_source!r}  results_dir={cfg.results_dir!r}")
    print(f"[setup] tasks={list(cfg.tasks)}")
    print(f"[setup] truncate_side={cfg.truncate_side!r}  min_traceback_length={cfg.min_traceback_length}")
    print(f"[setup] train_tracebacks={cfg.train_tracebacks}  (0 = all for training, no held-out eval)")

    extractor = T5RoutingFeatureExtractor(
        model_name=cfg.model_name,
        feature_layers=cfg.feature_layers,
        routing_dim=cfg.routing_dim,
        device=device,
        seed=cfg.seed,
    )
    extractor.save_projection(output_dir)

    router = ResidualFitGMMRouter(_to_router_cfg(cfg))

    seen_tasks: List[str] = []
    # Stores full deduped feature tensor per task; sliced into train/eval at split_cache[task]
    features_cache: Dict[str, torch.Tensor] = {}
    split_cache: Dict[str, int] = {}  # task -> number of training samples

    all_results: Dict[str, Dict] = {}

    for task in cfg.tasks:
        print("\n" + "#" * 90)
        print(f"[continual] traceback task {len(seen_tasks)}: {task}")
        print("#" * 90)

        try:
            predictions = _load_calibration_file(cfg, task)
        except FileNotFoundError as exc:
            print(f"[warn] {exc} — skipping {task}")
            continue

        raw_tracebacks = extract_tracebacks(predictions, min_length=cfg.min_traceback_length)
        deduped = deduplicate_tracebacks(raw_tracebacks)
        if cfg.max_tracebacks > 0 and len(deduped) > cfg.max_tracebacks:
            print(f"[cap] Capping {len(deduped)} → {cfg.max_tracebacks} tracebacks for {task}")
            deduped = deduped[: cfg.max_tracebacks]
        print_traceback_stats(task, raw_tracebacks, deduped)

        if not deduped:
            print(f"[warn] No usable tracebacks for {task} — skipping")
            continue

        n_train = min(cfg.train_tracebacks, len(deduped)) if cfg.train_tracebacks > 0 else len(deduped)
        n_eval = len(deduped) - n_train
        print(f"  train split: {n_train} | eval split: {n_eval}")

        if n_train == 0:
            print(f"[warn] No training tracebacks for {task} — skipping")
            continue

        # Encode full deduped set (cached), then slice into train / eval tensors.
        # Truncate to len(deduped) in case the cache was built with a larger uncapped set.
        z_all = _load_or_encode(task, deduped, extractor, cfg)[:len(deduped)]
        features_cache[task] = z_all
        split_cache[task] = n_train
        z_train = z_all[:n_train]

        actual_task_id = len(seen_tasks)
        seen_tasks.append(task)
        router.fit_new_task(task_name=task, task_id=actual_task_id, z_train=z_train)
        router.save(output_dir, step=actual_task_id)

        # --- Routing accuracy + confusion matrix over all seen tasks ---
        K = len(seen_tasks)
        confusion = [[0] * K for _ in range(K)]
        correct_total = n_total = 0
        per_task_acc: Dict[str, float] = {}

        print(f"\n[eval] Routing accuracy ({K} seen task(s)):")
        for true_id, eval_task in enumerate(seen_tasks):
            z_full = features_cache.get(eval_task)
            sp = split_cache.get(eval_task, 0)
            z_eval = z_full[sp:] if z_full is not None and sp < len(z_full) else None

            if z_eval is None or z_eval.shape[0] == 0:
                per_task_acc[eval_task] = float("nan")
                print(f"  - {eval_task:<18s}: n/a (no eval samples)")
                continue

            pred = router.predict(z_eval)
            y = torch.full_like(pred, fill_value=true_id)
            correct = int((pred == y).sum().item())
            total = int(y.numel())
            correct_total += correct
            n_total += total
            per_task_acc[eval_task] = correct / max(total, 1)

            for yt, yp in zip(y.tolist(), pred.tolist()):
                if 0 <= yt < K and 0 <= yp < K:
                    confusion[yt][yp] += 1

        overall = correct_total / max(n_total, 1)
        print(f"[eval] overall routing acc = {overall:.4f}")
        for t, acc in per_task_acc.items():
            if isinstance(acc, float) and acc != acc:
                print(f"  - {t:<18s}: n/a")
            else:
                print(f"  - {t:<18s}: {acc:.4f}")

        print("[eval] confusion matrix (rows=true, cols=pred):")
        header = "true\\pred" + "".join([f"\t{i}:{t[:8]}" for i, t in enumerate(seen_tasks)])
        print(header)
        for i, row in enumerate(confusion):
            print(f"{i}:{seen_tasks[i][:8]}" + "".join([f"\t{v}" for v in row]))

        all_results[f"step{actual_task_id}"] = {
            "seen_tasks": seen_tasks[:],
            "overall_acc": overall,
            "per_task_acc": per_task_acc,
            "confusion": confusion,
            "raw_tracebacks": len(raw_tracebacks),
            "deduped_tracebacks": len(deduped),
            "train_tracebacks": n_train,
            "eval_tracebacks": n_eval,
        }
        with open(output_dir / "traceback_routing_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n[done] saved traceback router checkpoints and results to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train GMM traceback router from calibration execution results")
    d = TracebackRouterConfig()

    p.add_argument("--model_name", type=str, default=d.model_name)
    p.add_argument("--output_dir", type=str, default=d.output_dir)
    p.add_argument("--tasks", type=str, default=None,
                   help="Comma-separated language list (default: all 9 executable languages)")
    p.add_argument("--feature_layers", type=int, default=d.feature_layers)
    p.add_argument("--routing_dim", type=int, default=d.routing_dim)
    p.add_argument("--max_length", type=int, default=d.max_length)
    p.add_argument("--batch_size", type=int, default=d.batch_size)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--gmm_components", type=int, default=d.gmm_components)
    p.add_argument("--em_iters", type=int, default=d.em_iters)
    p.add_argument("--em_tol", type=float, default=d.em_tol)
    p.add_argument("--variance_floor", type=float, default=d.variance_floor)
    p.add_argument("--eps", type=float, default=d.eps)
    p.add_argument("--no_save_features", action="store_true")
    p.add_argument("--force_recompute_features", action="store_true")
    # Traceback-specific
    p.add_argument("--results_source", type=str, default=d.results_source,
                   choices=["local", "hf_hub"])
    p.add_argument("--results_dir", type=str, default=d.results_dir,
                   help="Local directory or HF Hub repo ID containing calibration_<lang>.json files")
    p.add_argument("--results_repo_type", type=str, default=d.results_repo_type,
                   choices=["dataset", "model", "space"])
    p.add_argument("--truncate_side", type=str, default=d.truncate_side,
                   choices=["left", "right"],
                   help="left = keep end of traceback (error type/message); right = keep start")
    p.add_argument("--min_traceback_length", type=int, default=d.min_traceback_length)
    p.add_argument("--max_tracebacks", type=int, default=d.max_tracebacks,
                   help="Cap on total deduped tracebacks per task before the train/eval split (0 = unlimited)")
    p.add_argument("--train_tracebacks", type=int, default=d.train_tracebacks,
                   help="Number of deduped tracebacks used for training; remainder held out for eval (0 = all for training)")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    d = TracebackRouterConfig()
    tasks = d.tasks
    if args.tasks:
        tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())

    cfg = TracebackRouterConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        tasks=tasks,
        feature_layers=args.feature_layers,
        routing_dim=args.routing_dim,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
        gmm_components=args.gmm_components,
        em_iters=args.em_iters,
        em_tol=args.em_tol,
        variance_floor=args.variance_floor,
        eps=args.eps,
        save_features=not args.no_save_features,
        force_recompute_features=args.force_recompute_features,
        results_source=args.results_source,
        results_dir=args.results_dir,
        results_repo_type=args.results_repo_type,
        truncate_side=args.truncate_side,
        min_traceback_length=args.min_traceback_length,
        max_tracebacks=args.max_tracebacks,
        train_tracebacks=args.train_tracebacks,
    )
    run(cfg)


if __name__ == "__main__":
    main()
