#!/bin/bash
export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

export CUDA_VISIBLE_DEVICES=0
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