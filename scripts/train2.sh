#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export CUDA_VISIBLE_DEVICES

python gmm.py \
  --model_name Qwen/Qwen2.5-Coder-1.5B \
  --output_dir router/router_gmm_ckpt_codetask_no_novelty \
  --dataset_source codetask \
  --tasks CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST \
  --batch_size 16 \
  --train_k 5000 \
  --eval_k 2000 \
  --routing_dim 256 \
  --gmm_components 8 \
  --feature_layers 4 \
  --eval_split test \
  --omega_min 0.15 \
  --force_recompute_features
