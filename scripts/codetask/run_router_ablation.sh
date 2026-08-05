#!/usr/bin/env bash
# Fit and evaluate centroid, single-Gaussian, GMM(M=4), k-NN, and oracle routers.
# Installs requirements by default, then runs routing evaluation and generation.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

export PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL=${MODEL:-Qwen/Qwen2.5-Coder-1.5B}
BASE_PATH=${BASE_PATH:-dongg18/anamoe}
OUTPUT_ROOT=${OUTPUT_ROOT:-./router/router_ablation}
RESULT_ROOT=${RESULT_ROOT:-./inference_results/router_ablation}
LOG_ROOT=${LOG_ROOT:-./logs/router_ablation}
FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR:-$OUTPUT_ROOT/shared_feature_cache}
TRAIN_K=${TRAIN_K:-5000}
EVAL_K=${EVAL_K:-1000}
BATCH_SIZE=${BATCH_SIZE:-16}
KNN_K=${KNN_K:-5}
INSTALL_DEPS=${INSTALL_DEPS:-1}
RUN_INFERENCE=${RUN_INFERENCE:-1}

TASKS="CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST"
ADAPTERS="CONCODE/0,CodeTrans/0,CodeSearchNet/0,BFP/0,KodCode/0,RunBugRun/0,TheVault_Csharp/0,CoST/0"
PROMPT_LENS="320,320,256,130,512,256,256,256"
ANSWER_LENS="150,256,128,120,300,128,128,128"

mkdir -p "$OUTPUT_ROOT" "$RESULT_ROOT" "$LOG_ROOT" "$FEATURE_CACHE_DIR"
MASTER_LOG="$LOG_ROOT/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python command not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
  if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    "$PYTHON_BIN" -m ensurepip --upgrade
  fi
  "$PYTHON_BIN" -m pip install --requirement requirements.txt
fi

run_logged() {
  local log_file=$1
  shift
  "$@" 2>&1 | tee "$log_file"
}

COMMON_TRAIN_ARGS=(
  --model_name "$MODEL"
  --feature_cache_dir "$FEATURE_CACHE_DIR"
  --tasks "$TASKS"
  --batch_size "$BATCH_SIZE"
  --train_k "$TRAIN_K"
  --eval_k "$EVAL_K"
  --routing_dim 256
  --feature_layers 4
  --eval_split test
  --max_length 512
)

echo "[ablation] Fitting GMM (M=4)"
run_logged "$LOG_ROOT/gmm_m4_train.log" \
  "$PYTHON_BIN" gmm.py \
  --output_dir "$OUTPUT_ROOT/gmm_m4" \
  --dataset_source codetask \
  --gmm_components 4 \
  --variance_floor 0.02 \
  "${COMMON_TRAIN_ARGS[@]}"

echo "[ablation] Fitting single Gaussian (M=1)"
run_logged "$LOG_ROOT/single_gaussian_train.log" \
  "$PYTHON_BIN" gmm.py \
  --output_dir "$OUTPUT_ROOT/single_gaussian" \
  --dataset_source codetask \
  --gmm_components 1 \
  --variance_floor 0.02 \
  "${COMMON_TRAIN_ARGS[@]}"

for method in centroid knn oracle; do
  echo "[ablation] Fitting ${method}"
  run_logged "$LOG_ROOT/${method}_train.log" \
    "$PYTHON_BIN" router_baselines.py \
    --router_method "$method" \
    --output_dir "$OUTPUT_ROOT/$method" \
    --knn_k "$KNN_K" \
    "${COMMON_TRAIN_ARGS[@]}"
done

if [[ "$RUN_INFERENCE" == "1" ]]; then
  for method in centroid single_gaussian gmm_m4 knn oracle; do
    case "$method" in
      gmm_m4) infer_method=gmm ;;
      *) infer_method=$method ;;
    esac

    echo "[ablation] Downstream inference: ${method}"
    run_logged "$LOG_ROOT/${method}_inference.log" \
      "$PYTHON_BIN" infer_gmm.py \
      --router_method "$infer_method" \
      --model_name_or_path "$MODEL" \
      --base_path "$BASE_PATH" \
      --inference_model_path "$ADAPTERS" \
      --router_weight_path "$OUTPUT_ROOT/$method" \
      --benchmark non-executable \
      --routing_mode hard \
      --inference_batch 1 \
      --inference_tasks "$TASKS" \
      --max_prompt_len "$PROMPT_LENS" \
      --max_ans_len "$ANSWER_LENS" \
      --repetition_penalty 1.0 \
      --seed 42 \
      --inference_output_path "$RESULT_ROOT/$method"
  done
fi

"$PYTHON_BIN" summarize_router_ablation.py \
  --router_root "$OUTPUT_ROOT" \
  --result_root "$RESULT_ROOT" \
  --output_dir "$RESULT_ROOT"

echo "[ablation] Complete"
echo "[ablation] Master log: $MASTER_LOG"
echo "[ablation] Summary: $RESULT_ROOT/summary.json"
