#!/usr/bin/env bash
# Score every run of Experiment 1 (BLEU/CodeBLEU, CPU only) and write $OUT/report.md.
# Safe to re-run: per-file scores are cached in <run>/eval/scores_cache.json; unfinished runs are skipped.
# Usage: bash scripts/cgr/report_fast_codetask.sh [runs/fast_codetask]
set -euo pipefail
OUT=${1:-runs/fast_codetask}
RUNS=()
for d in "$OUT"/*/; do
  [ -f "$d/progress.json" ] && RUNS+=("${d%/}")
done
python -m cgr.eval.score --run_dir "${RUNS[@]}"
python -m cgr.eval.report --runs "${RUNS[@]}" --base "$OUT/BASE" --treatment A3 --control A1 --out "$OUT/report.md"
echo "[report] written to $OUT/report.md"
