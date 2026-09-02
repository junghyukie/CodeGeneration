#!/usr/bin/env bash
# Fit, certify, and evaluate the RC-GMM router only (no code generation/inference).
#
# Run from anywhere:
#   bash scripts/codetask/run_rc_gmm_router.sh
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES=1 TRAIN_K=3000 OUTPUT_DIR=./router/my_rc_gmm \
#     bash scripts/codetask/run_rc_gmm_router.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

# Set PYTHON_BIN explicitly when Bash/WSL does not inherit the Windows Python PATH.
# Example (Git Bash):
#   PYTHON_BIN=/c/Users/<user>/anaconda3/python.exe bash scripts/codetask/run_rc_gmm_router.sh
PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[rc-gmm] Python executable not found: $PYTHON_BIN" >&2
  echo "[rc-gmm] Set PYTHON_BIN to the Python that has this project's dependencies." >&2
  echo "[rc-gmm] Example: PYTHON_BIN=/c/Users/<user>/anaconda3/python.exe bash $0" >&2
  exit 127
fi

MODEL=${MODEL:-Qwen/Qwen2.5-Coder-1.5B}
OUTPUT_DIR=${OUTPUT_DIR:-./router/router_rc_gmm_codetask_dual}
LOG_DIR=${LOG_DIR:-./logs/rc_gmm_codetask_dual}
FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR:-}
RESUME_FROM=${RESUME_FROM:-}

TASKS=${TASKS:-CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST}
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

GATE_MODE=${GATE_MODE:-dual}
LIFETIME_RISK_BUDGET=${LIFETIME_RISK_BUDGET:-0.10}
MIN_NEW_ADMISSION=${MIN_NEW_ADMISSION:-0.80}
FIT_FRACTION=${FIT_FRACTION:-0.70}
OPT_FRACTION=${OPT_FRACTION:-0.15}
GATE_OPT_SAMPLES_PER_DENSITY=${GATE_OPT_SAMPLES_PER_DENSITY:-6000}
GATE_CERT_SAMPLES_PER_OLD=${GATE_CERT_SAMPLES_PER_OLD:-20000}
MAX_GATE_CERT_SAMPLES_PER_OLD=${MAX_GATE_CERT_SAMPLES_PER_OLD:-1000000}
CERT_BATCH_SIZE=${CERT_BATCH_SIZE:-8192}
DUAL_MAXITER=${DUAL_MAXITER:-400}
DUAL_FTOL=${DUAL_FTOL:-1e-10}
OPTIMIZATION_BUDGET_FRACTION=${OPTIMIZATION_BUDGET_FRACTION:-0.90}
LIFETIME_CONFIDENCE_BUDGET=${LIFETIME_CONFIDENCE_BUDGET:-0.05}
CERT_BOUND=${CERT_BOUND:-clopper_pearson}
TEMPERATURE=${TEMPERATURE:-1.0}
EVAL_SPLIT=${EVAL_SPLIT:-test}

# Boolean switches use 1=enabled and 0=disabled.
AUTO_SCALE_CERT_SAMPLES=${AUTO_SCALE_CERT_SAMPLES:-1}
SAVE_FEATURES=${SAVE_FEATURES:-1}
FORCE_RECOMPUTE_FEATURES=${FORCE_RECOMPUTE_FEATURES:-0}
CONTINUE_ON_UNSAFE=${CONTINUE_ON_UNSAFE:-0}

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

CMD=(
  "$PYTHON_BIN" rc_gmm_router.py
  --model_name "$MODEL"
  --output_dir "$OUTPUT_DIR"
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
  --gate_mode "$GATE_MODE"
  --lifetime_risk_budget "$LIFETIME_RISK_BUDGET"
  --min_new_admission "$MIN_NEW_ADMISSION"
  --fit_fraction "$FIT_FRACTION"
  --opt_fraction "$OPT_FRACTION"
  --gate_opt_samples_per_density "$GATE_OPT_SAMPLES_PER_DENSITY"
  --gate_cert_samples_per_old "$GATE_CERT_SAMPLES_PER_OLD"
  --max_gate_cert_samples_per_old "$MAX_GATE_CERT_SAMPLES_PER_OLD"
  --cert_batch_size "$CERT_BATCH_SIZE"
  --dual_maxiter "$DUAL_MAXITER"
  --dual_ftol "$DUAL_FTOL"
  --optimization_budget_fraction "$OPTIMIZATION_BUDGET_FRACTION"
  --lifetime_confidence_budget "$LIFETIME_CONFIDENCE_BUDGET"
  --cert_bound "$CERT_BOUND"
  --temperature "$TEMPERATURE"
  --eval_split "$EVAL_SPLIT"
)

if [[ -n "$FEATURE_CACHE_DIR" ]]; then
  CMD+=(--feature_cache_dir "$FEATURE_CACHE_DIR")
fi
if [[ -n "$RESUME_FROM" ]]; then
  CMD+=(--resume_from "$RESUME_FROM")
fi
if [[ "$AUTO_SCALE_CERT_SAMPLES" == "0" ]]; then
  CMD+=(--no_auto_scale_cert_samples)
fi
if [[ "$SAVE_FEATURES" == "0" ]]; then
  CMD+=(--no_save_features)
fi
if [[ "$FORCE_RECOMPUTE_FEATURES" == "1" ]]; then
  CMD+=(--force_recompute_features)
fi
if [[ "$CONTINUE_ON_UNSAFE" == "1" ]]; then
  CMD+=(--continue_on_unsafe)
fi

echo "[rc-gmm] GPU:        $CUDA_VISIBLE_DEVICES"
echo "[rc-gmm] Model:      $MODEL"
echo "[rc-gmm] Tasks:      $TASKS"
echo "[rc-gmm] Gate mode:  $GATE_MODE"
echo "[rc-gmm] Output:     $OUTPUT_DIR"
echo "[rc-gmm] Routing only; infer_gmm.py will not be run."

"${CMD[@]}" 2>&1 | tee "$LOG_DIR/routing.log"

echo "[rc-gmm] Routing metrics: $OUTPUT_DIR/routing_results.json"
echo "[rc-gmm] Gate certificates: $OUTPUT_DIR/admission_results.json"
echo "[rc-gmm] Log: $LOG_DIR/routing.log"
