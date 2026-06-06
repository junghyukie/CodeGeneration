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

ROUTING_MODE=${1:-soft}

python infer_gmm.py \
  --model_name_or_path   Qwen/Qwen2.5-Coder-1.5B \
  --base_path            dongg18/anamoe \
  --inference_model_path python,cpp,swift,rust,csharp,java,php,typescript,shell \
  --router_weight_path   ./router_gmm_ckpt \
  --benchmark            executable \
  --routing_mode         "${ROUTING_MODE}" \
  --routing_temperature  1.0 \
  --max_prompt_len       320,320,256,130,512,256,256,256,256 \
  --max_ans_len          150,256,128,120,300,128,128,128,128 \
  --inference_batch      1 \
  --inference_tasks      all \
  --do_sample \
  --temperature          0.2 \
  --top_p                0.95 \
  --top_k                0 \
  --num_return_sequences 5 \
  --repetition_penalty   1.0 \
  --seed                 42 \
  --inference_output_path ./inference_results/gmm_executable_"${ROUTING_MODE}"
