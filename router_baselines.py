"""Centroid, k-NN, and oracle baselines for the CodeTask routing ablation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from gmm import (
    EvalResult,
    RouterConfig,
    RoutingFeatureExtractor,
    ensure_dir,
    get_device,
    get_or_extract_features,
    print_eval,
    set_seed,
)


@dataclass
class BaselineTask:
    task_name: str
    task_id: int
    centroid: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None


class BaselineRouter:
    """Euclidean centroid or distance-weighted k-NN router."""

    def __init__(self, cfg: RouterConfig, method: str, knn_k: int = 5):
        if method not in {"centroid", "knn"}:
            raise ValueError(f"Unsupported baseline method: {method}")
        if knn_k < 1:
            raise ValueError("knn_k must be >= 1")
        self.cfg = cfg
        self.method = method
        self.knn_k = knn_k
        self.tasks: List[BaselineTask] = []

    def fit_new_task(self, task_name: str, task_id: int, z_train: torch.Tensor) -> None:
        z_train = z_train.detach().float().cpu()
        if z_train.numel() == 0:
            raise ValueError(f"No training features for task {task_name}")
        task = BaselineTask(task_name=task_name, task_id=task_id)
        if self.method == "centroid":
            task.centroid = z_train.mean(dim=0)
        else:
            # float16 halves checkpoint size; distances are evaluated in float32.
            task.features = z_train.half()
        self.tasks.append(task)
        print(f"[fit] method={self.method} task={task_id}:{task_name} N={len(z_train)}")

    def predict_scores(self, z: torch.Tensor) -> torch.Tensor:
        z = z.detach().float().cpu()
        if self.method == "centroid":
            centroids = torch.stack([task.centroid.float() for task in self.tasks])
            return -torch.cdist(z, centroids, p=2).pow(2)

        features = torch.cat([task.features.float() for task in self.tasks], dim=0)
        labels = torch.cat([
            torch.full((len(task.features),), idx, dtype=torch.long)
            for idx, task in enumerate(self.tasks)
        ])
        k = min(self.knn_k, len(features))
        score_chunks = []
        for z_chunk in z.split(128):
            distances, indices = torch.topk(
                torch.cdist(z_chunk, features, p=2), k=k, dim=1, largest=False
            )
            neighbor_labels = labels[indices]
            weights = 1.0 / distances.clamp_min(1e-8)
            votes = torch.zeros(len(z_chunk), len(self.tasks), dtype=torch.float32)
            votes.scatter_add_(1, neighbor_labels, weights)
            score_chunks.append(torch.log(votes.clamp_min(1e-12)))
        return torch.cat(score_chunks, dim=0)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        local_ids = self.predict_scores(z).argmax(dim=1)
        task_ids = torch.tensor([task.task_id for task in self.tasks], dtype=torch.long)
        return task_ids[local_ids]

    def save(self, output_dir: str | Path, step: int) -> None:
        payload = {
            "format": "baseline_router_v1",
            "method": self.method,
            "knn_k": self.knn_k,
            "cfg": asdict(self.cfg),
            "step": step,
            "tasks": [
                {
                    "task_name": task.task_name,
                    "task_id": task.task_id,
                    "centroid": task.centroid,
                    "features": task.features,
                }
                for task in self.tasks
            ],
        }
        torch.save(payload, ensure_dir(output_dir) / f"router_step{step}.pt")

    @classmethod
    def load(cls, path: str | Path) -> "BaselineRouter":
        payload = torch.load(path, map_location="cpu")
        if payload.get("format") != "baseline_router_v1":
            raise ValueError(f"Not a baseline-router checkpoint: {path}")
        known = set(RouterConfig.__dataclass_fields__)
        cfg = RouterConfig(**{k: v for k, v in payload["cfg"].items() if k in known})
        router = cls(cfg, payload["method"], int(payload.get("knn_k", 5)))
        router.tasks = [
            BaselineTask(
                task_name=item["task_name"],
                task_id=int(item["task_id"]),
                centroid=item.get("centroid"),
                features=item.get("features"),
            )
            for item in payload["tasks"]
        ]
        return router


def evaluate(router: BaselineRouter, extractor: RoutingFeatureExtractor, cfg: RouterConfig) -> EvalResult:
    task_count = len(router.tasks)
    confusion = torch.zeros(task_count, task_count, dtype=torch.long)
    per_task_acc: Dict[str, float] = {}
    correct_total = 0
    sample_total = 0

    for true_id, task in enumerate(router.tasks):
        z_eval = get_or_extract_features(extractor, cfg, task.task_name, cfg.eval_split, cfg.eval_k)
        predicted = router.predict(z_eval)
        labels = torch.full_like(predicted, true_id)
        correct = int((predicted == labels).sum())
        total = len(labels)
        correct_total += correct
        sample_total += total
        per_task_acc[task.task_name] = correct / max(total, 1)
        for true_label, predicted_label in zip(labels.tolist(), predicted.tolist()):
            confusion[true_label, predicted_label] += 1

    return EvalResult(
        overall_acc=correct_total / max(sample_total, 1),
        per_task_acc=per_task_acc,
        confusion=confusion.tolist(),
    )


def oracle_results(cfg: RouterConfig) -> None:
    output_dir = ensure_dir(cfg.output_dir)
    results = {}
    for step, task in enumerate(cfg.tasks):
        seen = list(cfg.tasks[: step + 1])
        confusion = [[cfg.eval_k if i == j else 0 for j in range(step + 1)] for i in range(step + 1)]
        result = EvalResult(1.0, {name: 1.0 for name in seen}, confusion)
        print_eval(step, seen, result)
        results[f"step{step}"] = {
            "seen_tasks": seen,
            "overall_acc": 1.0,
            "per_task_acc": result.per_task_acc,
            "confusion": confusion,
        }
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump({**asdict(cfg), "router_method": "oracle"}, file, indent=2)
    with open(output_dir / "routing_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"[done] saved oracle routing results to: {output_dir}")


def run(cfg: RouterConfig, method: str, knn_k: int) -> None:
    set_seed(cfg.seed)
    if method == "oracle":
        oracle_results(cfg)
        return

    output_dir = ensure_dir(cfg.output_dir)
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump({**asdict(cfg), "router_method": method, "knn_k": knn_k}, file, indent=2)

    extractor = RoutingFeatureExtractor(
        model_name=cfg.model_name,
        feature_layers=cfg.feature_layers,
        routing_dim=cfg.routing_dim,
        device=get_device(),
        seed=cfg.seed,
    )
    extractor.save_projection(output_dir)
    router = BaselineRouter(cfg, method=method, knn_k=knn_k)
    all_results = {}

    for task_id, task_name in enumerate(cfg.tasks):
        z_train = get_or_extract_features(extractor, cfg, task_name, "train", cfg.train_k)
        router.fit_new_task(task_name, task_id, z_train)
        router.save(output_dir, task_id)
        result = evaluate(router, extractor, cfg)
        seen = list(cfg.tasks[: task_id + 1])
        print_eval(task_id, seen, result)
        all_results[f"step{task_id}"] = {
            "seen_tasks": seen,
            "overall_acc": result.overall_acc,
            "per_task_acc": result.per_task_acc,
            "confusion": result.confusion,
        }
        with open(output_dir / "routing_results.json", "w", encoding="utf-8") as file:
            json.dump(all_results, file, indent=2)

    print(f"[done] saved {method} checkpoints and results to: {output_dir}")


def parse_tasks(raw: str):
    return tuple(task.strip() for task in raw.split(",") if task.strip())


def main() -> None:
    defaults = RouterConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router_method", choices=["centroid", "knn", "oracle"], required=True)
    parser.add_argument("--model_name", default=defaults.model_name)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_cache_dir", default=None)
    parser.add_argument("--tasks", default=",".join(defaults.tasks))
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--train_k", type=int, default=defaults.train_k)
    parser.add_argument("--eval_k", type=int, default=defaults.eval_k)
    parser.add_argument("--routing_dim", type=int, default=defaults.routing_dim)
    parser.add_argument("--feature_layers", type=int, default=defaults.feature_layers)
    parser.add_argument("--max_length", type=int, default=defaults.max_length)
    parser.add_argument("--eval_split", choices=["validation", "test"], default=defaults.eval_split)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--knn_k", type=int, default=5)
    args = parser.parse_args()

    cfg = RouterConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        feature_cache_dir=args.feature_cache_dir,
        dataset_source="codetask",
        tasks=parse_tasks(args.tasks),
        batch_size=args.batch_size,
        train_k=args.train_k,
        eval_k=args.eval_k,
        routing_dim=args.routing_dim,
        feature_layers=args.feature_layers,
        max_length=args.max_length,
        eval_split=args.eval_split,
        seed=args.seed,
    )
    run(cfg, args.router_method, args.knn_k)


if __name__ == "__main__":
    main()
