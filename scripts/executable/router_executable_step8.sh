#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES

OUTPUT_DIR="router/ckpt_executable_dim256_comp4_vf0.001_mean"

python gmm.py \
  --model_name      Qwen/Qwen2.5-Coder-1.5B \
  --output_dir      "${OUTPUT_DIR}" \
  --dataset_source  executable \
  --tasks           python,cpp,swift,rust,csharp,java,php,typescript,shell \
  --batch_size      8 \
  --train_k         -1 \
  --eval_k          -1 \
  --variance_floor  0.001 \
  --routing_dim     256 \
  --gmm_components  4 \
  --feature_layers  4 \
  --max_length      2048 \
  --seed            42 \
  --eval_split      test \
  --resume_from     "${OUTPUT_DIR}/router_step7.pt"
