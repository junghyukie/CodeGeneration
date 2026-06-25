#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES

python gmm.py \
  --model_name      Qwen/Qwen2.5-Coder-1.5B \
  --output_dir      router/ckpt_executable_dim256_comp4_vf0.001_mean \
  --dataset_source  executable \
  --tasks           python,cpp,swift,rust,csharp,java,php,typescript,shell \
  --batch_size      16 \
  --train_k         5000 \
  --eval_k          -1 \
  --variance_floor  0.02 \
  --routing_dim     256 \
  --gmm_components  4 \
  --feature_layers  4 \
  --eval_split      test \
  --force_recompute_features \
  --max_length      1024

