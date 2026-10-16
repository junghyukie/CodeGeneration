"""Evaluation-time generation (Algorithm 4, inner loop). Test prompts go through a predict function; the task label
is attached to the written rows only after generation. Files: <run>/predictions/<tag>/results-<step>-<task>.json"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np

from cgr.data import Example
from cgr.infer import Prediction, Predictor, generate
from cgr.state import CGRState

PredictFn = Callable[[List[str], int], List[Prediction]]


def _batched(st: CGRState, prompts: List[str], n: int, alpha_fn: Callable[[int], np.ndarray]) -> List[Prediction]:
    out = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    bs = st.cfg.infer_batch_size
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        alpha = alpha_fn(len(idx))
        samples = generate(st, [prompts[i] for i in idx], n, alpha=alpha)
        for j, i in enumerate(idx):
            out[i] = Prediction(samples[j], [] if alpha is None else alpha[j].tolist(), -1)
    return out


def cgr_predict_fn(st: CGRState) -> PredictFn:
    return Predictor(st).predict


def seqlora_predict_fn(st: CGRState) -> PredictFn:
    """Baseline: the single shared atom with coefficient 1 for every query (no routing)."""
    return lambda prompts, n: _batched(st, prompts, n, lambda b: np.ones((b, 1)))


def base_predict_fn(st: CGRState) -> PredictFn:
    """Reference: pretrained backbone, all atoms disabled (FWT reference a~_j)."""
    return lambda prompts, n: _batched(st, prompts, n, lambda b: None)


def oracle_predict_fn(st: CGRState, j: int) -> PredictFn:
    """A8 analysis only: alpha = e_true. Uses the task label, so it is never a method."""
    def fn(prompts, n):
        def alpha(b):
            a = np.zeros((b, st.router.K))
            a[:, j] = 1.0
            return a
        return _batched(st, prompts, n, alpha)
    return fn


def write_predictions(path: Path, tasks: Sequence[str], j: int, test: Sequence[Example], preds: List[Prediction],
                      seconds: float) -> None:
    rows = [{"source": e.prompt, "ground-truth": e.answer, "prediction": p.samples, "alpha": p.alpha,
             "k_hat": p.k_hat} for e, p in zip(test, preds)]
    payload = {"eval": {}, "task": tasks[j], "task_index": j, "generation_sec": round(seconds, 1), "predictions": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_tasks(tag: str, step: int, task_ids: Sequence[int], predict_fn_for: Callable[[int], PredictFn], view,
                   st: CGRState, out_dir: Path) -> None:
    cfg = st.cfg
    for j in task_ids:
        path = out_dir / "predictions" / tag / f"results-{step}-{cfg.tasks[j]}.json"
        if path.exists():  # resumable
            continue
        test = view.test_split(j)
        t0 = time.time()
        preds = predict_fn_for(j)([e.prompt for e in test], cfg.n_samples)
        write_predictions(path, cfg.tasks, j, test, preds, time.time() - t0)
        print(f"[eval] {tag} task {j + 1}/{len(cfg.tasks)} {cfg.tasks[j]}: {len(test)} prompts, "
              f"{time.time() - t0:.0f}s", flush=True)
