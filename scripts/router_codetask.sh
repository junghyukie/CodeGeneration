#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

python gmm.py \
  --model_name Qwen/Qwen2.5-Coder-1.5B \
  --output_dir router/router_gmm_codetask_vf0.02_dim_256_comp_4_layer_4 \
  --dataset_source codetask \
  --tasks CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST \
  --batch_size 16 \
  --train_k 5000 \
  --eval_k 1000 \
  --routing_dim 256 \
  --gmm_components 4 \
  --feature_layers 4 \
  --eval_split test \
  --force_recompute_features \
  --max_length 512 \
  --variance_floor 0.02 \
