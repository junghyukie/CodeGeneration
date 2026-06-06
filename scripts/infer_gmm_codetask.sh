#!/usr/bin/env bash
# Inference with the GMM router on the executable benchmark.
# Tasks follow the same order as train1.sh:
#   python, cpp, swift, rust, csharp, java, php, typescript, shell  (9 tasks)
#
# Usage:
#   bash scripts/infer_gmm_executable.sh [soft|hard]
#
# The first argument selects the routing mode (default: soft).

set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export CUDA_VISIBLE_DEVICES

ROUTING_MODE=${1:-soft}

python infer_gmm.py \
  --model_name_or_path   Qwen/Qwen2.5-Coder-1.5B \
  --base_path            dongg18/anamoe \
  --inference_model_path "CONCODE/0","CodeTrans/0","CodeSearchNet/0","BFP/0","KodCode/0","RunBugRun/0","TheVault_Csharp/0","CoST/0" \
  --router_weight_path   ./router_gmm_ckpt_codetask \
  --benchmark            non-executable \
  --routing_mode         "${ROUTING_MODE}" \
  --routing_temperature  1.0 \
  --inference_batch      1 \
  --inference_tasks      CONCODE,CodeTrans \
  --repetition_penalty   1.0 \
  --seed                 42 \
  --inference_output_path ./inference_results/gmm_codetask_"${ROUTING_MODE}"
