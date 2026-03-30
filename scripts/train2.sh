#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

export CUDA_VISIBLE_DEVICES=1

python t5_trainer1.py \
  --task_list TheVault_Csharp\
  --log_filepath logs/TheVault_Csharp.log\
  --shared_adapter_name lora_TheVault_Csharp

python t5_trainer1.py \
  --task_list CoST\
  --log_filepath logs/CoST.log\
  --shared_adapter_name lora_CoST

python t5_trainer1.py \
  --task_list KodCode\
  --log_filepath logs/KodCode.log\
  --shared_adapter_name lora_KodCode