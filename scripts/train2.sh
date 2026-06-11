#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES

python gmm2.py \
  --model_name      Qwen/Qwen2.5-Coder-1.5B \
  --output_dir      ./router_exe/router_gmm_exe_vf0.02_dim_256_comp_4_omega_0.1_layer_4 \
  --dataset_source  executable \
  --tasks           python,cpp,swift,rust,csharp,java,php,typescript,shell \
  --batch_size      16 \
  --train_k         5000 \
  --variance_floor  0.02 \
  --eval_k          1000 \
  --routing_dim     256 \
  --gmm_components  4 \
  --omega_min       0.1 \
  --feature_layers  4 \
  --eval_split      test \
  --force_recompute_features

