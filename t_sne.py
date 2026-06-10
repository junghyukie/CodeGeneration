from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

from gmm import (
    RouterConfig,
    T5RoutingFeatureExtractor,
    get_device,
    get_or_extract_features,
    parse_tasks,
    set_seed,
)

try:
    import plotly.express as px
except ImportError:
    px = None


def run_tsne(
    features: np.ndarray,
    n_components: int,
    seed: int,
    perplexity: float,
) -> np.ndarray:
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        max_iter=1500,
    )
    return tsne.fit_transform(features)


def balanced_downsample(
    features: np.ndarray,
    labels: np.ndarray,
    max_plot_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_plot_samples == -1 or len(features) <= max_plot_samples:
        return features, labels

    rng = np.random.RandomState(seed)
    unique_labels = np.unique(labels)
    per_task = max(1, max_plot_samples // len(unique_labels))
    selected: List[np.ndarray] = []

    for label in unique_labels:
        indices = np.flatnonzero(labels == label)
        selected.append(rng.choice(indices, size=min(per_task, len(indices)), replace=False))

    selected_indices = np.concatenate(selected)
    if len(selected_indices) > max_plot_samples:
        selected_indices = rng.choice(selected_indices, size=max_plot_samples, replace=False)
    rng.shuffle(selected_indices)
    return features[selected_indices], labels[selected_indices]


def save_2d_plot(
    embedding: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    split: str,
    routing_dim: int,
) -> None:
    frame = pd.DataFrame({"x": embedding[:, 0], "y": embedding[:, 1], "task": labels})
    plt.figure(figsize=(12, 9))
    for task in sorted(frame["task"].unique()):
        rows = frame[frame["task"] == task]
        plt.scatter(rows["x"], rows["y"], s=10, alpha=0.7, label=task)

    plt.title(f"CodeTask feature distribution ({split}, projected to {routing_dim}D)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"[saved] {output_path}")


def save_3d_plot(
    embedding: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    split: str,
    routing_dim: int,
) -> None:
    if px is None:
        print("[skip] plotly is not installed; skipping the 3D plot.")
        return

    frame = pd.DataFrame(
        {"x": embedding[:, 0], "y": embedding[:, 1], "z": embedding[:, 2], "task": labels}
    )
    figure = px.scatter_3d(
        frame,
        x="x",
        y="y",
        z="z",
        color="task",
        opacity=0.7,
        title=f"CodeTask feature distribution ({split}, projected to {routing_dim}D)",
    )
    figure.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"[saved] {output_path}")


def parse_args() -> argparse.Namespace:
    defaults = RouterConfig()
    parser = argparse.ArgumentParser(
        description="Visualize the exact CodeTask data/features used by gmm.py."
    )
    parser.add_argument("--model_name", default=defaults.model_name)
    parser.add_argument("--tasks", default=None, help="Comma-separated CodeTask names.")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--samples_per_task", type=int, default=400)
    parser.add_argument("--max_plot_samples", type=int, default=4000)
    parser.add_argument("--feature_layers", type=int, default=defaults.feature_layers)
    dimension_group = parser.add_mutually_exclusive_group()
    dimension_group.add_argument(
        "--routing_dims",
        default="128,256",
        help="Comma-separated projection dimensions to compare (default: 128,256).",
    )
    dimension_group.add_argument(
        "--routing_dim",
        type=int,
        help="Run one projection dimension; retained for compatibility.",
    )
    parser.add_argument("--max_length", type=int, default=defaults.max_length)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--out_dir", default="outputs/tsne/codetask")
    parser.add_argument("--force_recompute_features", action="store_true")
    parser.add_argument("--skip_3d", action="store_true")
    return parser.parse_args()


def parse_routing_dims(args: argparse.Namespace) -> Tuple[int, ...]:
    dimensions = (args.routing_dim,) if args.routing_dim is not None else parse_tasks(args.routing_dims, ())
    try:
        parsed = tuple(dict.fromkeys(int(dimension) for dimension in dimensions))
    except ValueError as error:
        raise ValueError(f"Invalid --routing_dims value: {args.routing_dims}") from error
    if not parsed or any(dimension <= 0 for dimension in parsed):
        raise ValueError("Projection dimensions must be positive integers.")
    return parsed


def visualize_dimension(
    args: argparse.Namespace,
    tasks: Tuple[str, ...],
    output_dir: Path,
    device: torch.device,
    routing_dim: int,
) -> None:
    print(f"\n[projection] running with routing_dim={routing_dim}")

    cfg = RouterConfig(
        model_name=args.model_name,
        output_dir=str(output_dir),
        dataset_source="codetask",
        tasks=tasks,
        feature_layers=args.feature_layers,
        routing_dim=routing_dim,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
        save_features=True,
        force_recompute_features=args.force_recompute_features,
    )

    extractor = T5RoutingFeatureExtractor(
        model_name=cfg.model_name,
        feature_layers=cfg.feature_layers,
        routing_dim=cfg.routing_dim,
        device=device,
        seed=cfg.seed,
    )

    feature_chunks: List[torch.Tensor] = []
    label_chunks: List[str] = []
    for task in tasks:
        features = get_or_extract_features(
            extractor=extractor,
            cfg=cfg,
            task=task,
            split=args.split,
            k=args.samples_per_task,
        )
        print(f"[features] {task}: {tuple(features.shape)}")
        feature_chunks.append(features)
        label_chunks.extend([task] * len(features))

    features = torch.cat(feature_chunks, dim=0).numpy()
    labels = np.asarray(label_chunks)
    features, labels = balanced_downsample(
        features,
        labels,
        max_plot_samples=args.max_plot_samples,
        seed=cfg.seed,
    )
    if len(features) < 2:
        raise ValueError("t-SNE needs at least two samples.")

    perplexity = min(40.0, max(1.0, (len(features) - 1) / 3.0))
    print(f"[tsne] samples={len(features)}, dimensions={features.shape[1]}, perplexity={perplexity:.2f}")

    stem = f"codetask_{args.split}_k{args.samples_per_task}_p{routing_dim}"
    embedding_2d = run_tsne(features, n_components=2, seed=cfg.seed, perplexity=perplexity)
    save_2d_plot(
        embedding_2d,
        labels,
        output_dir / f"tsne2d_{stem}.png",
        args.split,
        routing_dim,
    )

    if not args.skip_3d:
        embedding_3d = run_tsne(features, n_components=3, seed=cfg.seed, perplexity=perplexity)
        save_3d_plot(
            embedding_3d,
            labels,
            output_dir / f"tsne3d_{stem}.html",
            args.split,
            routing_dim,
        )


def main() -> None:
    args = parse_args()
    defaults = RouterConfig()
    tasks = parse_tasks(args.tasks, defaults.tasks)
    routing_dims = parse_routing_dims(args)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_device()
    print(f"[setup] device={device}")
    print(f"[setup] model={args.model_name}")
    print(f"[setup] CodeTask tasks={list(tasks)}")
    print(f"[setup] split={args.split}, samples_per_task={args.samples_per_task}")
    print(f"[setup] projection dimensions={list(routing_dims)}")

    for routing_dim in routing_dims:
        visualize_dimension(args, tasks, output_dir, device, routing_dim)


if __name__ == "__main__":
    main()
