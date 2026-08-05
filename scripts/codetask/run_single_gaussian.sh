#!/usr/bin/env bash
# Single-Gaussian router (GMM with M=1) for all eight CodeTask tasks.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-1.5B}
BASE_PATH=${BASE_PATH:-dongg18/anamoe}
ROUTER_DIR=${ROUTER_DIR:-./router/router_single_gaussian_m1}
RESULT_DIR=${RESULT_DIR:-./inference_results/single_gaussian_m1/step_7}
LOG_DIR=${LOG_DIR:-./logs/single_gaussian_m1}
RUN_INFERENCE=${RUN_INFERENCE:-1}

TASKS="CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST"
ADAPTERS="CONCODE/0,CodeTrans/0,CodeSearchNet/0,BFP/0,KodCode/0,RunBugRun/0,TheVault_Csharp/0,CoST/0"

mkdir -p "$ROUTER_DIR" "$RESULT_DIR" "$LOG_DIR"

echo "[single-gaussian] Fitting router with M=1"
"$PYTHON_BIN" gmm.py \
  --model_name "$MODEL" \
  --output_dir "$ROUTER_DIR" \
  --dataset_source codetask \
  --tasks "$TASKS" \
  --batch_size 16 \
  --train_k 5000 \
  --eval_k 1000 \
  --routing_dim 256 \
  --gmm_components 1 \
  --feature_layers 4 \
  --eval_split test \
  --max_length 512 \
  --variance_floor 0.02 \
  2>&1 | tee "$LOG_DIR/train.log"

if [[ "$RUN_INFERENCE" == "1" ]]; then
  echo "[single-gaussian] Running downstream inference"
  "$PYTHON_BIN" infer_gmm.py \
    --router_method single_gaussian \
    --model_name_or_path "$MODEL" \
    --base_path "$BASE_PATH" \
    --inference_model_path "$ADAPTERS" \
    --router_weight_path "$ROUTER_DIR" \
    --benchmark non-executable \
    --routing_mode hard \
    --inference_batch 1 \
    --inference_tasks "$TASKS" \
    --max_prompt_len 320,320,256,130,512,256,256,256 \
    --max_ans_len 150,256,128,120,300,128,128,128 \
    --repetition_penalty 1.0 \
    --seed 42 \
    --inference_output_path "$RESULT_DIR" \
    2>&1 | tee "$LOG_DIR/inference.log"
fi

echo "[single-gaussian] Router:  $ROUTER_DIR"
echo "[single-gaussian] Results: $RESULT_DIR"
echo "[single-gaussian] Logs:    $LOG_DIR"
