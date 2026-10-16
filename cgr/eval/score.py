"""Evaluation harness: reads generation files after generation has finished; writes only under <run>/eval/.

CodeTask: thesis BLEU/CodeBLEU (evaluator/compute_metrics.py): BLEU for CodeSearchNet and TheVault_Csharp,
CodeBLEU otherwise. Executable: thesis execution harness (executable_dataset/calc_metrics.py) run as a subprocess
on copies of the generation files.
Usage: python -m cgr.eval.score --run_dir runs/fast_codetask/A3_s1234
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from cgr.eval.cl_metrics import routing_diagnostics, summarize

REPO = Path(__file__).resolve().parents[2]
TEXT_OUTPUT_TASKS = {"CodeSearchNet", "TheVault_Csharp"}


def score_codetask_file(path: Path) -> float:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from evaluator.compute_metrics import DATASET_TO_OUTPUT_LANG, compute_metrics
    payload = json.loads(path.read_text(encoding="utf-8"))
    task = payload["task"]
    preds = [r["prediction"][0] for r in payload["predictions"]]
    refs = [r["ground-truth"] for r in payload["predictions"]]
    code = task not in TEXT_OUTPUT_TASKS
    m = compute_metrics(preds, refs, calc_codebleu=code, language=DATASET_TO_OUTPUT_LANG.get(task))
    return float(m["codebleu"] if code else m["bleu"])


def score_executable_file(path: Path, eval_dir: Path, num_samples: int, metric: str) -> float:
    eval_dir.mkdir(parents=True, exist_ok=True)
    copy = eval_dir / path.name
    if not copy.exists() or "pass_at_1" not in json.loads(copy.read_text(encoding="utf-8")).get("eval", {}):
        shutil.copy2(path, copy)
        subprocess.run([sys.executable, "calc_metrics.py", "--local-file", str(copy.resolve()),
                        "--num-samples", str(num_samples), "--ks", "1", "5"],
                       cwd=REPO / "executable_dataset", check=True)
    return 100.0 * float(json.loads(copy.read_text(encoding="utf-8"))["eval"][metric])


def score_run(run: Path, metric: str = "pass_at_1") -> dict:
    progress = json.loads((run / "progress.json").read_text())
    tasks, benchmark, n_samples = progress["tasks"], progress["benchmark"], progress["n_samples"]
    K = len(tasks)
    cache_path = run / "eval" / "scores_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def score(f: Path) -> float:
        key = f"{f.relative_to(run)}:{metric}"
        if key not in cache:
            if benchmark == "codetask":
                cache[key] = score_codetask_file(f)
            else:
                cache[key] = score_executable_file(f, run / "eval" / f.parent.name, n_samples, metric)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1))
        return cache[key]

    def vector(tag: str) -> list:
        v = [None] * K
        for f in sorted((run / "predictions" / tag).glob("results-*.json")):
            v[json.loads(f.read_text(encoding="utf-8"))["task_index"]] = score(f)
        return v

    out = {"run": str(run), **progress, "metric": "BLEU/CodeBLEU" if benchmark == "codetask" else metric}
    pred_dir = run / "predictions"
    if (pred_dir / "base").exists():
        out["base"] = vector("base")
    steps = sorted(int(p.name.split("_")[1]) for p in pred_dir.glob("step_*")) if pred_dir.exists() else []
    if steps:
        a = np.full((max(steps), K), np.nan)
        routing = {}
        for i in steps:
            true_idx, k_hat, alphas = [], [], []
            for f in sorted((pred_dir / f"step_{i}").glob("results-*.json")):
                payload = json.loads(f.read_text(encoding="utf-8"))
                j = payload["task_index"]
                a[i - 1, j] = score(f)
                for r in payload["predictions"]:
                    if r["k_hat"] >= 0:
                        true_idx.append(j); k_hat.append(r["k_hat"]); alphas.append(r["alpha"])
            if true_idx:
                routing[f"after_task_{i}"] = routing_diagnostics(true_idx, k_hat, alphas, i)
        out["matrix"] = [[None if np.isnan(x) else float(x) for x in row] for row in a]
        out["routing"] = routing
        if a.shape[0] == K:
            out["summary"] = summarize(a)
    if (pred_dir / "phase_a").exists():
        out["phase_a"] = vector("phase_a")
    oracle = sorted(pred_dir.glob("oracle_step_*")) if pred_dir.exists() else []
    if oracle:
        out["oracle_final"] = vector(oracle[-1].name)
    (run / "eval").mkdir(exist_ok=True)
    (run / "eval" / "results.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", nargs="+", required=True)
    ap.add_argument("--metric", default="pass_at_1", help="executable only: pass_at_1 or pass_at_5")
    args = ap.parse_args()
    for r in args.run_dir:
        res = score_run(Path(r), args.metric)
        print(r, json.dumps(res.get("summary", res.get("base")), indent=1))


if __name__ == "__main__":
    main()
