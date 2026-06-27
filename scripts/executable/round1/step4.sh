#!/usr/bin/env bash
# Round-1 GMM-router inference: step 4.
# Infers all tasks seen through step 4: python,cpp,swift,rust,csharp
#
# Usage:
#   bash scripts/executable/round1/step4.sh [soft|hard]
#
# The first argument selects the routing mode (default: soft).

set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export CUDA_VISIBLE_DEVICES

ROUTING_MODE=${1:-soft}

python infer_gmm.py \
  --model_name_or_path   Qwen/Qwen2.5-Coder-1.5B \
  --base_path            ankhanhtran02/lora-per-task-executable-start-4 \
  --inference_model_path "python/0","cpp/0","swift/0","rust/0","csharp/0" \
  --router_weight_path   router/ckpt_executable_dim256_comp4_vf0.001_mean \
  --benchmark            executable \
  --routing_mode         "${ROUTING_MODE}" \
  --routing_temperature  1.0 \
  --max_prompt_len       2048,2048,2048,2048,2048 \
  --max_ans_len          2048,2048,2048,2048,2048 \
  --inference_batch      1 \
  --inference_tasks      python,cpp,swift,rust,csharp \
  --do_sample \
  --temperature          0.2 \
  --top_p                0.95 \
  --routing_top_p        1.0 \
  --top_k                0 \
  --num_return_sequences 5 \
  --repetition_penalty   1.0 \
  --seed                 42 \
  --inference_output_path inference_results/gmm_exe_vf0.02_dim256_comp4_omega1.0_soft_temp_1.0/step_4
