"""Compare scored runs (each needs eval/results.json from cgr.eval.score) and write a markdown report.

  python -m cgr.eval.report --runs runs/fast_codetask/*/ --base runs/fast_codetask/BASE --out report.md

Groups runs by (method, ablation) and averages over seeds. Composition gain Delta_j = a_jj(A3) - a_jj(A1) uses runs
with the same seed (methodology Sec. 8: CGR vs C = I).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cgr.eval.cl_metrics import summarize

LABELS = {("cgr", "A0"): "A0 GMM-Router (C=I, lambda=0)", ("cgr", "A1"): "A1 routing only (C=I, lambda=1)",
          ("cgr", "A2"): "A2 transfer only (lambda=0)", ("cgr", "A3"): "A3 full CGR",
          ("cgr", "A5"): "A5 Phase A only", ("cgr", "A6"): "A6 no Phase A", ("cgr", "A7"): "A7 beta init 0"}


def _fmt(xs):
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    if not xs:
        return "-"
    return f"{np.mean(xs):.2f}" + (f" ± {np.std(xs, ddof=1):.2f}" if len(xs) > 1 else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--base", default=None, help="run dir of --method base (FWT reference)")
    ap.add_argument("--treatment", default="A3")
    ap.add_argument("--control", default="A1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = None
    base_file = Path(args.base) / "eval" / "results.json" if args.base else None
    if base_file is not None and base_file.exists() and None not in json.loads(base_file.read_text()).get("base", [None]):
        ref = json.loads(base_file.read_text())["base"]
    elif args.base:
        print(f"[warn] no complete base reference at {args.base}; FWT is omitted")
    runs = []
    for r in args.runs:
        p = Path(r) / "eval" / "results.json"
        if p.exists():
            res = json.loads(p.read_text())
            if "matrix" in res and all(v is not None for v in np.diag(np.array(res["matrix"], dtype=float))):
                runs.append(res)
    if not runs:
        raise SystemExit("no scored runs found")
    tasks = runs[0]["tasks"]
    K = len(tasks)

    groups = defaultdict(list)
    for res in runs:
        key = (res["method"], res["ablation"] if res["method"] == "cgr" else "-")
        a = np.array(res["matrix"], dtype=float)
        s = summarize(a, ref)
        acc = list(res.get("routing", {}).values())
        s["routing_acc_final"] = acc[-1]["accuracy"] * 100 if acc else None
        groups[key].append({"res": res, "a": a, "s": s})

    lines = [f"# CGR fast experiment report\n", f"Tasks (in order): {', '.join(tasks)}  ",
             f"Metric: {runs[0]['metric']} (x100), mean ± std over seeds\n",
             "## Continual-learning summary\n",
             "| Method | seeds | Avg final ↑ | Avg a_jj ↑ | BWT ↑ | FM ↓ | FWT ↑ | Routing acc (%) |",
             "|---|---|---|---|---|---|---|---|"]
    for key, g in sorted(groups.items()):
        name = LABELS.get(key, "SeqLoRA (shared LoRA, sequential)" if key[0] == "seqlora" else str(key))
        col = lambda m: _fmt([x["s"].get(m) for x in g])
        lines.append(f"| {name} | {len(g)} | {col('avg_final')} | {col('avg_diag')} | {col('BWT')} | {col('FM')} | "
                     f"{col('FWT')} | {col('routing_acc_final')} |")

    lines += ["\n## Per-task scores\n", "| Method | " + " | ".join(tasks) + " |", "|---|" + "---|" * K]
    if ref:
        lines.append("| Pretrained backbone (a~_j) | " + " | ".join(f"{x:.2f}" for x in ref) + " |")
    for key, g in sorted(groups.items()):
        name = LABELS.get(key, "SeqLoRA" if key[0] == "seqlora" else str(key))
        lines.append(f"| {name}: a_jj (just learned) | " +
                     " | ".join(_fmt([x["a"][j, j] for x in g]) for j in range(K)) + " |")
        lines.append(f"| {name}: a_Kj (after all tasks) | " +
                     " | ".join(_fmt([x["a"][K - 1, j] for x in g]) for j in range(K)) + " |")
        pa = [x["res"].get("phase_a") for x in g if x["res"].get("phase_a")]
        if pa:
            lines.append(f"| {name}: Phase-A-only (pure reuse) | " +
                         " | ".join(_fmt([v[j] for v in pa]) for j in range(K)) + " |")

    treat = {x["res"]["seed"]: x["a"] for x in groups.get(("cgr", args.treatment), [])}
    ctrl = {x["res"]["seed"]: x["a"] for x in groups.get(("cgr", args.control), [])}
    seeds = sorted(set(treat) & set(ctrl))
    if seeds:
        lines += [f"\n## Composition gain Δ_j = a_jj({args.treatment}) − a_jj({args.control}), paired by seed\n",
                  "| | " + " | ".join(tasks) + " | mean (tasks 2..K) |", "|---|" + "---|" * (K + 1)]
        d = np.array([[treat[s][j, j] - ctrl[s][j, j] for j in range(K)] for s in seeds])
        lines.append("| Δ_j | " + " | ".join(_fmt(list(d[:, j])) for j in range(K)) +
                     f" | {_fmt(list(d[:, 1:].mean(axis=1)))} |")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
