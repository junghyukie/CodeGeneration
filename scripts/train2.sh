#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES

python gmm2.py \
  --model_name      Qwen/Qwen2.5-Coder-1.5B \
  --output_dir      ./router_gmm_test_2 \
  --dataset_source  codetask \
  --tasks           CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST \
  --batch_size      16 \
  --train_k         5000 \
  --variance_floor  0.02 \
  --eval_k          1000 \
  --routing_dim     256 \
  --gmm_components  4 \
  --omega_min       1.0 \
  --feature_layers  4 \
  --eval_split      test \
  --force_recompute_features

