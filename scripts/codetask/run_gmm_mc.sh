#!/usr/bin/env bash
# Fit, optimize, certify, and evaluate MC-GMM on CodeTask.
#
# Full run:
#   bash scripts/codetask/run_gmm_mc.sh
#
# Small smoke run:
#   TASKS=CONCODE,CodeTrans TRAIN_K=200 EVAL_K=100 \
#     OLD_PSEUDO_OPT_SAMPLES=500 OLD_PSEUDO_CERT_SAMPLES=5000 \
#     BOOTSTRAP_REPLICATES=500 bash scripts/codetask/run_gmm_mc.sh
#
# Resume (projection_P.pt must be beside the checkpoint):
#   RESUME_FROM=./router/router_gmm_mc_codetask/router_step3.pt \
#     bash scripts/codetask/run_gmm_mc.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-1.5B}
OUTPUT_DIR=${OUTPUT_DIR:-./router/router_gmm_mc_codetask}
LOG_DIR=${LOG_DIR:-./logs/gmm_mc_codetask}
FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR:-./router/mc_codetask_feature_cache}
CACHE_TAG=${CACHE_TAG:-v1}
RESUME_FROM=${RESUME_FROM:-}

TASKS=${TASKS:-CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST}
IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"
FINAL_STEP=$((${#TASK_ARRAY[@]} - 1))

BATCH_SIZE=${BATCH_SIZE:-16}
TRAIN_K=${TRAIN_K:-5000}
EVAL_K=${EVAL_K:-1000}
SEED=${SEED:-42}
FEATURE_LAYERS=${FEATURE_LAYERS:-4}
ROUTING_DIM=${ROUTING_DIM:-256}
MAX_LENGTH=${MAX_LENGTH:-512}
GMM_COMPONENTS=${GMM_COMPONENTS:-4}
EM_ITERS=${EM_ITERS:-50}
EM_TOL=${EM_TOL:-1e-4}
VARIANCE_FLOOR=${VARIANCE_FLOOR:-0.02}
EPS=${EPS:-1e-8}

GMM_FIT_FRACTION=${GMM_FIT_FRACTION:-0.70}
CORRECTION_OPT_FRACTION=${CORRECTION_OPT_FRACTION:-0.15}
OLD_PSEUDO_OPT_SAMPLES=${OLD_PSEUDO_OPT_SAMPLES:-6000}
OLD_PSEUDO_CERT_SAMPLES=${OLD_PSEUDO_CERT_SAMPLES:-20000}
HISTORICAL_MARGIN=${HISTORICAL_MARGIN:-0.5}
NEW_MARGIN=${NEW_MARGIN:-0.0}
OLD_MARGIN=${OLD_MARGIN:-0.0}
C_NEW=${C_NEW:-1.0}
C_OLD=${C_OLD:-1.0}
SOLVER_MAX_ITER=${SOLVER_MAX_ITER:-1000}
SOLVER_TOLERANCE=${SOLVER_TOLERANCE:-1e-7}
BOOTSTRAP_REPLICATES=${BOOTSTRAP_REPLICATES:-2000}
CONFIDENCE_ALPHA=${CONFIDENCE_ALPHA:-0.05}
OLD_DISTURBANCE_BUDGET=${OLD_DISTURBANCE_BUDGET:-0.02}
EVAL_SPLIT=${EVAL_SPLIT:-test}

# Boolean switches: 1=enabled, 0=disabled.
INSTALL_DEPS=${INSTALL_DEPS:-0}
SAVE_FEATURES=${SAVE_FEATURES:-1}
FORCE_RECOMPUTE_FEATURES=${FORCE_RECOMPUTE_FEATURES:-0}
DRY_RUN=${DRY_RUN:-0}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[mc-gmm] Python executable not found: $PYTHON_BIN" >&2
  echo "[mc-gmm] Set PYTHON_BIN to the Python containing project dependencies." >&2
  exit 127
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install --requirement requirements.txt
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$FEATURE_CACHE_DIR"
LOG_FILE="$LOG_DIR/routing_$(date +%Y%m%d_%H%M%S).log"

CMD=(
  "$PYTHON_BIN" gmm_mc.py
  --model_name "$MODEL"
  --output_dir "$OUTPUT_DIR"
  --feature_cache_dir "$FEATURE_CACHE_DIR"
  --cache_tag "$CACHE_TAG"
  --dataset_source codetask
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
  --correction_opt_fraction "$CORRECTION_OPT_FRACTION"
  --old_pseudo_opt_samples "$OLD_PSEUDO_OPT_SAMPLES"
  --old_pseudo_cert_samples "$OLD_PSEUDO_CERT_SAMPLES"
  --historical_margin "$HISTORICAL_MARGIN"
  --new_margin "$NEW_MARGIN"
  --old_margin "$OLD_MARGIN"
  --c_new "$C_NEW"
  --c_old "$C_OLD"
  --solver_max_iter "$SOLVER_MAX_ITER"
  --solver_tolerance "$SOLVER_TOLERANCE"
  --bootstrap_replicates "$BOOTSTRAP_REPLICATES"
  --confidence_alpha "$CONFIDENCE_ALPHA"
  --old_disturbance_budget "$OLD_DISTURBANCE_BUDGET"
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

echo "[mc-gmm] GPU:              $CUDA_VISIBLE_DEVICES"
echo "[mc-gmm] Model:            $MODEL"
echo "[mc-gmm] Tasks:            $TASKS"
echo "[mc-gmm] Split:            $GMM_FIT_FRACTION / $CORRECTION_OPT_FRACTION / cert remainder"
echo "[mc-gmm] QP:               C_new=$C_NEW C_old=$C_OLD eta=$NEW_MARGIN kappa=$OLD_MARGIN"
echo "[mc-gmm] Certification:    B=$BOOTSTRAP_REPLICATES alpha=$CONFIDENCE_ALPHA old_budget=$OLD_DISTURBANCE_BUDGET"
echo "[mc-gmm] Output:           $OUTPUT_DIR"
echo "[mc-gmm] Log:              $LOG_FILE"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[mc-gmm] Command: '
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo "[mc-gmm] Complete"
echo "[mc-gmm] Routing metrics:  $OUTPUT_DIR/routing_results.json"
echo "[mc-gmm] MC diagnostics:   $OUTPUT_DIR/minimum_change_results.json"
echo "[mc-gmm] Final checkpoint: $OUTPUT_DIR/router_step${FINAL_STEP}.pt"
