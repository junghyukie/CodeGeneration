#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

export CUDA_VISIBLE_DEVICES=1
python t5_trainer1.py \
  --task_list Codetrans\
  --log_filepath logs/Codetrans.log \
  --shared_adapter_name lora_Codetrans
  
python t5_trainer1.py \
  --task_list CONCODE\
  --log_filepath logs/CONCODE.log \
  --shared_adapter_name lora_CONCODE

python t5_trainer1.py \
  --task_list Codetrans\
  --log_filepath logs/Codetrans.log \
  --shared_adapter_name lora_Codetrans

python t5_trainer1.py \
  --task_list CodeSearchNet\
  --log_filepath logs/CodeSearchNet.log \
  --shared_adapter_name lora_CodeSearchNet

python t5_trainer1.py \
  --task_list BFP\
  --log_filepath logs/BFP.log \
  --shared_adapter_name lora_BFP 

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