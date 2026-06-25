#!/usr/bin/env bash
# Inference with the GMM router on the CodeTask benchmark — step 7 (CoST).
# i = 7  →  loads router_step7.pt, evaluates all 8 tasks.
#
# Usage:
#   bash scripts/codetask/infer_gmm_CoST.sh [soft|hard]

set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export CUDA_VISIBLE_DEVICES

ROUTING_MODE=${1:-hard}

python infer_gmm.py \
  --model_name_or_path   Qwen/Qwen2.5-Coder-1.5B \
  --base_path            dongg18/anamoe \
  --inference_model_path "CONCODE/0","CodeTrans/0","CodeSearchNet/0","BFP/0","KodCode/0","RunBugRun/0","TheVault_Csharp/0","CoST/0" \
  --router_weight_path   router/router_gmm_codetask_vf0.02_dim_256_comp_4_layer_4 \
  --benchmark            non-executable \
  --routing_mode         "${ROUTING_MODE}" \
  --routing_temperature  1.0 \
  --inference_batch      1 \
  --inference_tasks      CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST \
  --max_prompt_len       320,320,256,130,512,256,256,256 \
  --max_ans_len          150,256,128,120,300,128,128,128 \
  --repetition_penalty   1.0 \
  --seed                 42 \
  --inference_output_path ./inference_results/gmm_codetask_router_gmm_codetask_vf0.02_dim_256_comp_4_layer_4/step_7
