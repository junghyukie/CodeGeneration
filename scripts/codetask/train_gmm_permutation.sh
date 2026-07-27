#!/usr/bin/env bash
# Fit the GMM router continually on one CodeTask ordering.
# Usage: bash scripts/codetask/train_gmm_permutation.sh <scenario>
# Scenarios: original, permutation_1, permutation_2, permutation_3, permutation_4

set -euo pipefail

SCENARIO=${1:?"Usage: $0 <original|permutation_1|permutation_2|permutation_3|permutation_4>"}

# The GMM router uses a single visible GPU when CUDA is available.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL=${MODEL:-Qwen/Qwen2.5-Coder-1.5B}
OUTPUT_ROOT=${OUTPUT_ROOT:-./router/gmm_permutations}
TRAIN_K=${TRAIN_K:-5000}
EVAL_K=${EVAL_K:-1000}
BATCH_SIZE=${BATCH_SIZE:-16}
PYTHON_BIN=${PYTHON_BIN:-python}

# Task IDs from the experiment design:
# A=CONCODE, B=CodeTrans, C=CodeSearchNet, D=BFP,
# E=KodCode, F=RunBugRun, G=TheVault_Csharp, H=CoST.
case "$SCENARIO" in
  original)
    TASKS="CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST"
    ;;
  permutation_1)
    TASKS="CONCODE,CodeTrans,CoST,CodeSearchNet,TheVault_Csharp,BFP,RunBugRun,KodCode"
    ;;
  permutation_2)
    TASKS="CodeSearchNet,BFP,CodeTrans,KodCode,CONCODE,RunBugRun,CoST,TheVault_Csharp"
    ;;
  permutation_3)
    TASKS="KodCode,RunBugRun,BFP,TheVault_Csharp,CodeSearchNet,CoST,CodeTrans,CONCODE"
    ;;
  permutation_4)
    TASKS="TheVault_Csharp,CoST,RunBugRun,CONCODE,KodCode,CodeTrans,BFP,CodeSearchNet"
    ;;
  *)
    echo "Unknown scenario: $SCENARIO" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_ROOT}/${SCENARIO}"

echo "Scenario: ${SCENARIO}"
echo "Task order: ${TASKS}"
echo "Output: ${OUTPUT_DIR}"

# These defaults intentionally match scripts/codetask/router_codetask.sh.
"$PYTHON_BIN" gmm.py \
  --model_name "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_source codetask \
  --tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --train_k "$TRAIN_K" \
  --eval_k "$EVAL_K" \
  --routing_dim 256 \
  --gmm_components 4 \
  --feature_layers 4 \
  --eval_split test \
  --force_recompute_features \
  --max_length 512 \
  --variance_floor 0.02
