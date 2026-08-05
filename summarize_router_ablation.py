"""Collect router accuracy and downstream metrics from an ablation run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = ("centroid", "single_gaussian", "gmm_m4", "knn", "oracle")
TASKS = (
    "CONCODE",
    "CodeTrans",
    "CodeSearchNet",
    "BFP",
    "KodCode",
    "RunBugRun",
    "TheVault_Csharp",
    "CoST",
)


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router_root", type=Path, required=True)
    parser.add_argument("--result_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for method in METHODS:
        routing = read_json(args.router_root / method / "routing_results.json") or {}
        final_step = routing.get("step7", {})
        downstream = {}
        for task in TASKS:
            result = read_json(args.result_root / method / f"results-7-{task}.json")
            if result is not None:
                downstream[task] = result.get("eval", {})
        summary[method] = {
            "routing_accuracy": final_step.get("overall_acc"),
            "routing_per_task": final_step.get("per_task_acc", {}),
            "downstream_metrics": downstream,
        }

    json_path = args.output_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "final_routing_accuracy"])
        for method in METHODS:
            writer.writerow([method, summary[method]["routing_accuracy"]])

    print(f"[summary] JSON: {json_path}")
    print(f"[summary] CSV:  {csv_path}")


if __name__ == "__main__":
    main()
