#!/usr/bin/env bash
# Fit, calibrate, certify, and evaluate PBC-GMM on executable-language tasks.
#
# Full run:
#   bash scripts/executable/run_gmm_pbc.sh
#
# Small smoke run:
#   TASKS=python,cpp TRAIN_K=200 EVAL_K=100 \
#     OLD_PSEUDO_FIT_SAMPLES=1000 OLD_PSEUDO_CERT_SAMPLES=2000 \
#     BOOTSTRAP_REPLICATES=500 bash scripts/executable/run_gmm_pbc.sh
#
# Resume (projection_P.pt must be beside the checkpoint):
#   RESUME_FROM=./router/router_gmm_pbc_executable/router_step3.pt \
#     bash scripts/executable/run_gmm_pbc.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-1.5B}
EXECUTABLE_DATASET_NAME=${EXECUTABLE_DATASET_NAME:-ankhanhtran02/CL4Code-executable-datasets}
OUTPUT_DIR=${OUTPUT_DIR:-./router/router_gmm_pbc_executable}
LOG_DIR=${LOG_DIR:-./logs/gmm_pbc_executable}
FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR:-./router/pbc_executable_feature_cache}
CACHE_TAG=${CACHE_TAG:-v1}
RESUME_FROM=${RESUME_FROM:-}

TASKS=${TASKS:-python,cpp,swift,rust,csharp,java,php,typescript,shell}
IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"
FINAL_STEP=$((${#TASK_ARRAY[@]} - 1))

BATCH_SIZE=${BATCH_SIZE:-8}
TRAIN_K=${TRAIN_K:-5000}
EVAL_K=${EVAL_K:--1}
SEED=${SEED:-42}

FEATURE_LAYERS=${FEATURE_LAYERS:-4}
ROUTING_DIM=${ROUTING_DIM:-256}
MAX_LENGTH=${MAX_LENGTH:-2048}
GMM_COMPONENTS=${GMM_COMPONENTS:-4}
EM_ITERS=${EM_ITERS:-50}
EM_TOL=${EM_TOL:-1e-4}
VARIANCE_FLOOR=${VARIANCE_FLOOR:-0.02}
EPS=${EPS:-1e-8}

GMM_FIT_FRACTION=${GMM_FIT_FRACTION:-0.70}
BOUNDARY_FIT_FRACTION=${BOUNDARY_FIT_FRACTION:-0.15}
OLD_PSEUDO_FIT_SAMPLES=${OLD_PSEUDO_FIT_SAMPLES:-6000}
OLD_PSEUDO_CERT_SAMPLES=${OLD_PSEUDO_CERT_SAMPLES:-20000}
BOOTSTRAP_REPLICATES=${BOOTSTRAP_REPLICATES:-2000}
BOOTSTRAP_ALPHA=${BOOTSTRAP_ALPHA:-0.05}
MARGIN_THRESHOLD=${MARGIN_THRESHOLD:-0.5}
EVAL_SPLIT=${EVAL_SPLIT:-test}

# Boolean switches: 1=enabled, 0=disabled.
INSTALL_DEPS=${INSTALL_DEPS:-0}
SAVE_FEATURES=${SAVE_FEATURES:-1}
FORCE_RECOMPUTE_FEATURES=${FORCE_RECOMPUTE_FEATURES:-0}
DRY_RUN=${DRY_RUN:-0}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[pbc-gmm-exec] Python executable not found: $PYTHON_BIN" >&2
  echo "[pbc-gmm-exec] Set PYTHON_BIN to the Python containing project dependencies." >&2
  exit 127
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install --requirement requirements.txt
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$FEATURE_CACHE_DIR"
LOG_FILE="$LOG_DIR/routing_$(date +%Y%m%d_%H%M%S).log"

CMD=(
  "$PYTHON_BIN" gmm_pbc.py
  --model_name "$MODEL"
  --output_dir "$OUTPUT_DIR"
  --feature_cache_dir "$FEATURE_CACHE_DIR"
  --cache_tag "$CACHE_TAG"
  --dataset_source executable
  --executable_dataset_name "$EXECUTABLE_DATASET_NAME"
  --tasks "$TASKS"
  --batch_size "$BATCH_SIZE"
  --train_k "$TRAIN_K"
  --eval_k "$EVAL_K"
  --seed "$SEED"
  --feature_layers "$FEATURE_LAYERS"
  --routing_dim "$ROUTING_DIM"
  --max_length "$MAX_LENGTH"
  --gmm_components "$GMM_COMPONENTS"
  --em_iters "$EM_ITERS"
  --em_tol "$EM_TOL"
  --variance_floor "$VARIANCE_FLOOR"
  --eps "$EPS"
  --gmm_fit_fraction "$GMM_FIT_FRACTION"
  --boundary_fit_fraction "$BOUNDARY_FIT_FRACTION"
  --old_pseudo_fit_samples "$OLD_PSEUDO_FIT_SAMPLES"
  --old_pseudo_cert_samples "$OLD_PSEUDO_CERT_SAMPLES"
  --bootstrap_replicates "$BOOTSTRAP_REPLICATES"
  --bootstrap_alpha "$BOOTSTRAP_ALPHA"
  --margin_threshold "$MARGIN_THRESHOLD"
  --eval_split "$EVAL_SPLIT"
)

if [[ -n "$RESUME_FROM" ]]; then
  CMD+=(--resume_from "$RESUME_FROM")
fi
if [[ "$SAVE_FEATURES" == "0" ]]; then
  CMD+=(--no_save_features)
fi
if [[ "$FORCE_RECOMPUTE_FEATURES" == "1" ]]; then
  CMD+=(--force_recompute_features)
fi

echo "[pbc-gmm-exec] GPU:              $CUDA_VISIBLE_DEVICES"
echo "[pbc-gmm-exec] Model:            $MODEL"
echo "[pbc-gmm-exec] Dataset:          $EXECUTABLE_DATASET_NAME"
echo "[pbc-gmm-exec] Tasks:            $TASKS"
echo "[pbc-gmm-exec] Split:            $GMM_FIT_FRACTION / $BOUNDARY_FIT_FRACTION / cert remainder"
echo "[pbc-gmm-exec] Bootstrap:        B=$BOOTSTRAP_REPLICATES alpha=$BOOTSTRAP_ALPHA"
echo "[pbc-gmm-exec] Margin threshold: $MARGIN_THRESHOLD"
echo "[pbc-gmm-exec] Output:           $OUTPUT_DIR"
echo "[pbc-gmm-exec] Log:              $LOG_FILE"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[pbc-gmm-exec] Command: '
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo "[pbc-gmm-exec] Complete"
echo "[pbc-gmm-exec] Routing metrics:  $OUTPUT_DIR/routing_results.json"
echo "[pbc-gmm-exec] PBC diagnostics:  $OUTPUT_DIR/boundary_calibration_results.json"
echo "[pbc-gmm-exec] Final checkpoint: $OUTPUT_DIR/router_step${FINAL_STEP}.pt"
