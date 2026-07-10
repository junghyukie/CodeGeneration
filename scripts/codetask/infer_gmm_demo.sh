#!/usr/bin/env bash
# Inference with the GMM router on a small local CodeTask demo file — one
# sample per task drawn from the test split of "dongg18/CODETASK_with_instruction_pool"
# (see demo_data/codetask/). Same router/adapters as infer_gmm_codetask.sh
# (final continual-learning step i=7, all 8 tasks), but reads
# --demo_dataset_path instead of pulling the full HF test split.
#
# Usage:
#   bash scripts/codetask/infer_gmm_demo.sh [soft|hard]

set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES

ROUTING_MODE=${1:-hard}

: "${MODEL:=Qwen/Qwen2.5-Coder-1.5B}"
: "${BASE_PATH:=dongg18/anamoe}"
: "${ROUTER_PATH:=ankhanhtran02/router_gmm_codetask_vf0.02_dim_256_comp_4_layer_4}"
: "${DEVICE:=cpu}"
: "${DEMO_DATASET_PATH:=demo_data/codetask/mixed.json}"
: "${OUTPUT_PATH:=./inference_results/demo_pipeline/codetask}"

python infer_gmm.py \
  --model_name_or_path    "$MODEL" \
  --base_path             "$BASE_PATH" \
  --inference_model_path  "CONCODE/0","CodeTrans/0","CodeSearchNet/0","BFP/0","KodCode/0","RunBugRun/0","TheVault_Csharp/0","CoST/0" \
  --router_weight_path    "$ROUTER_PATH" \
  --benchmark             non-executable \
  --routing_mode          "${ROUTING_MODE}" \
  --routing_temperature   1.0 \
  --inference_batch       1 \
  --inference_tasks       CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST \
  --max_prompt_len        320,320,256,130,512,256,256,256 \
  --max_ans_len           150,256,128,120,300,128,128,128 \
  --repetition_penalty    1.0 \
  --seed                  42 \
  --device                "$DEVICE" \
  --demo_dataset_path     "$DEMO_DATASET_PATH" \
  --inference_output_path "$OUTPUT_PATH"
