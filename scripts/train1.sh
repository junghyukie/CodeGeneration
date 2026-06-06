#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

python gmm.py \
  --model_name Qwen/Qwen2.5-Coder-1.5B \
  --output_dir ./router_gmm_ckpt \
  --dataset_source executable \
  --tasks python,cpp,swift,rust,csharp,java,php,typescript,shell \
  --batch_size 16 \
  --train_k -1 \
  --eval_k -1 \
  --routing_dim 256 \
  --gmm_components 4 \
  --feature_layers 4 \
  --eval_split test \
  --force_recompute_features \
  --max_length 1024
