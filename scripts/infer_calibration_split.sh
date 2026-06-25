#!/bin/bash
# Run calibration_MBPP inference for each executable language using a trained anamoe adapter.
#
# Each language's adapter is loaded locally from:
#   $ADAPTER_BASE_DIR/<language>/0/
# or from a HuggingFace Hub repo when ADAPTER_BASE_DIR is a repo ID (no leading ./):
#   $ADAPTER_BASE_DIR  (with subfolder <language>/0 inferred automatically)
#
# Environment variables (all optional, defaults shown):
#   MODEL            - base model path or HF repo  (default: Qwen/Qwen2.5-Coder-1.5B)
#   ADAPTER_BASE_DIR - root dir or HF repo ID      (default: ./output_models/lora_per_task_executable_start_4)
#   OUTPUT_DIR       - where calibration_<lang>.json files are written  (default: ./calibration_results)
#   CUDA_DEVICES     - which GPUs to expose        (default: 0,1,2)
#   NUM_GPUS         - number of GPUs to use       (default: number of CUDA_DEVICES)
#   ZERO_STAGE       - DeepSpeed ZeRO stage        (default: 0 — fastest for inference)
#
# Usage:
#   bash scripts/infer_calibration_split.sh
#   CUDA_DEVICES=0 NUM_GPUS=1 bash scripts/infer_calibration_split.sh

export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

: "${MODEL:=Qwen/Qwen2.5-Coder-1.5B}"
: "${ADAPTER_BASE_DIR:=ankhanhtran02/lora-per-task-executable-start-4}"
: "${OUTPUT_DIR:=./calibration_results}"
: "${CUDA_DEVICES:=1,2,3,4}"
: "${ZERO_STAGE:=0}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

# Count the number of GPUs from CUDA_DEVICES if NUM_GPUS is not set
if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS=$(echo "$CUDA_DEVICES" | tr ',' '\n' | wc -l)
fi

set -euo pipefail

mkdir -p "$OUTPUT_DIR"
port=$(shuf -i25000-30000 -n1)

for language in python cpp swift rust csharp java php typescript shell; do
  adapter_dir="${ADAPTER_BASE_DIR}/${language}/0"

  # For local adapters, verify the directory exists before proceeding
  if [[ "$ADAPTER_BASE_DIR" == ./* || "$ADAPTER_BASE_DIR" == /* ]]; then
    if [ ! -d "$adapter_dir" ]; then
      echo "[calibration] WARNING: adapter not found at $adapter_dir — skipping $language"
      continue
    fi
    adapter_arg="$adapter_dir"
  else
    # HF Hub repo ID — pass the repo; run_calibration_inference adds the subfolder
    adapter_arg="$ADAPTER_BASE_DIR"
  fi

  echo "[calibration] === $language (${NUM_GPUS} GPU(s)) ==="

  deepspeed --master_port "$port" --num_gpus "$NUM_GPUS" \
    training/main_anamoe.py \
      --model_name_or_path  "$MODEL" \
      --benchmark           executable \
      --CL_method           anamoe \
      --dataset_name        "$language" \
      --max_prompt_len      2048 \
      --max_ans_len         2048 \
      --num_train           1 \
      --num_eval            -1 \
      --num_test            1 \
      --seed                1234 \
      --per_device_eval_batch_size 8 \
      --do_sample \
      --num_return_sequences 5 \
      --temperature         0.2 \
      --top_p               0.95 \
      --repetition_penalty  1.0 \
      --infer_calibration \
      --adapter_path        "$adapter_arg" \
      --inference_output_path "$OUTPUT_DIR" \
      --zero_stage          "$ZERO_STAGE" \
      --deepspeed \
      --run_name            "calib_${language}" \
      --group_name          "calibration_executable" \
      --num_train_epochs 0

  echo "[calibration] Done: ${OUTPUT_DIR}/calibration_${language}.json"
done

echo "[calibration] All languages complete. Results in $OUTPUT_DIR"
