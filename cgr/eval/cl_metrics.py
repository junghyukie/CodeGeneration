"""Continual-learning metrics (Sec. 8): FM, BWT, FWT, composition gain, routing diagnostics.

a[i, j] (0-based) is the score on task j after training tasks 0..i; only the lower triangle is used.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def fm(a: np.ndarray) -> float:
    """FM_K = 1/(K-1) sum_{j<K} max_{i<K} (a_ij - a_Kj), with i ranging over steps where task j was seen.
    Steps that were not evaluated (NaN, eval_mode=diag_last) are skipped, which makes FM a lower bound."""
    K = a.shape[0]
    return float(np.mean([np.nanmax(a[j:K - 1, j]) - a[K - 1, j] for j in range(K - 1)]))


def bwt(a: np.ndarray) -> float:
    K = a.shape[0]
    return float(np.mean([a[K - 1, j] - a[j, j] for j in range(K - 1)]))


def fwt(a: np.ndarray, ref: Sequence[float]) -> float:
    """FWT_K = 1/(K-1) sum_{j>=2} (a_jj - ref_j); ref_j = pretrained-backbone score (thesis a_{0,j})."""
    K = a.shape[0]
    return float(np.mean([a[j, j] - ref[j] for j in range(1, K)]))


def routing_diagnostics(true_idx: Sequence[int], k_hat: Sequence[int], alphas: Sequence[Sequence[float]], K: int) -> Dict:
    conf = np.zeros((K, K))
    for t, p in zip(true_idx, k_hat):
        conf[t, p] += 1
    rows = conf.sum(axis=1, keepdims=True)
    A = np.asarray(alphas, dtype=np.float64)
    ent = -np.sum(np.where(A > 0, A * np.log(np.clip(A, 1e-300, None)), 0.0), axis=1)
    return {
        "confusion": (conf / np.maximum(rows, 1)).tolist(),
        "per_task_accuracy": [float(conf[j, j] / max(rows[j, 0], 1)) for j in range(K)],
        "accuracy": float(np.trace(conf) / max(conf.sum(), 1)),
        "mean_alpha_entropy": float(ent.mean()) if len(ent) else 0.0,
    }


def summarize(a: np.ndarray, ref: Optional[Sequence[float]] = None) -> Dict:
    K = a.shape[0]
    out = {"avg_final": float(np.mean(a[K - 1])), "avg_diag": float(np.mean(np.diag(a)))}
    if K > 1:
        out.update({"BWT": bwt(a), "FM": fm(a)})
        if ref is not None:
            out["FWT"] = fwt(a, ref)
    return out
