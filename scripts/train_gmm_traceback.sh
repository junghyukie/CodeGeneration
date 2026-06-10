#!/bin/bash
# Train a GMM traceback router on calibration execution results.
#
# Loads calibration_{language}.json files (produced by infer_calibration_split.sh
# + execution evaluation), extracts failed-prediction stderr (tracebacks),
# deduplicates them, encodes each traceback with a frozen T5 encoder, and fits
# a per-task Gaussian Mixture Model — identical to the prompt-based GMM router
# in gmm.py but using error text as input.
#
# Environment variables (all optional, defaults shown):
#   MODEL             - T5 encoder for feature extraction   (default: Salesforce/codet5-small)
#   RESULTS_SOURCE    - "local" or "hf_hub"                 (default: local)
#   RESULTS_DIR       - local dir with calibration_*.json   (default: ./calibration_results)
#                       OR HF Hub repo ID when RESULTS_SOURCE=hf_hub
#   RESULTS_REPO_TYPE - HF repo type (dataset|model|space)  (default: dataset)
#   OUTPUT_DIR        - where router checkpoints are saved   (default: ./router_gmm_traceback_ckpt)
#   TASKS             - comma-separated language list        (default: all 9 languages)
#   MAX_TRACEBACKS    - max deduped tracebacks per task      (default: 0 = unlimited)
#
# Usage:
#   # Local calibration results (default):
#   bash scripts/train_gmm_traceback.sh
#
#   # Load from HF Hub:
#   RESULTS_SOURCE=hf_hub RESULTS_DIR=my-org/calib-results bash scripts/train_gmm_traceback.sh
#
#   # Single language, smaller model:
#   TASKS=python bash scripts/train_gmm_traceback.sh

export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

: "${MODEL:=Salesforce/codet5-small}"
: "${RESULTS_SOURCE:=local}"
: "${RESULTS_DIR:=./calibration_results}"
: "${RESULTS_REPO_TYPE:=dataset}"
: "${OUTPUT_DIR:=./router_gmm_traceback_ckpt}"
: "${TASKS:=python,cpp,swift,rust,csharp,java,php,typescript,shell}"
: "${MAX_TRACEBACKS:=0}"

python gmm_traceback.py \
  --model_name            "$MODEL" \
  --results_source        "$RESULTS_SOURCE" \
  --results_dir           "$RESULTS_DIR" \
  --results_repo_type     "$RESULTS_REPO_TYPE" \
  --output_dir            "$OUTPUT_DIR" \
  --tasks                 "$TASKS" \
  --feature_layers        4 \
  --routing_dim           128 \
  --max_length            256 \
  --batch_size            32 \
  --gmm_components        4 \
  --em_iters              100 \
  --em_tol                1e-4 \
  --variance_floor        1e-3 \
  --eps                   1e-8 \
  --omega_min             0.05 \
  --kappa                 0.0 \
  --tau_n                 1.0 \
  --truncate_side         left \
  --min_traceback_length  10 \
  --max_tracebacks        "$MAX_TRACEBACKS" \
  --seed                  42
