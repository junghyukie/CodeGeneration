#!/bin/bash
# Fine-tune a per-task LoRA adapter using failed-execution feedback.
#
# Loads calibration_{language}.json (from infer_calibration_split.sh + execution evaluation),
# extracts (instruction, failed prediction, traceback) → fixed code pairs, and continues
# training the adapter with DeepSpeed.
#
# Environment variables (all optional, defaults shown):
#   MODEL            - base model path or HF repo         (default: Qwen/Qwen2.5-Coder-1.5B)
#   LANGUAGE         - programming language to refine      (default: python)
#   ADAPTER_PATH     - local adapter dir or HF Hub repo   (default: ankhanhtran02/lora-per-task-executable-start-4)
#                      For local paths, provide the full adapter directory (containing adapter_config.json).
#                      For HF Hub, the subfolder {language}/0 is resolved automatically.
#   RESULTS_DIR      - dir with calibration_*.json files  (default: ./calibration_results)
#   RESULTS_SOURCE   - "local" or "hf_hub"                (default: local)
#   OUTPUT_DIR       - where to save the refined adapter   (default: ./refined_adapters/{language})
#   CUDA_DEVICES     - GPU indices to expose               (default: 0,1,2,3)
#   NUM_GPUS         - number of GPUs                      (default: auto from CUDA_DEVICES)
#   ZERO_STAGE       - DeepSpeed ZeRO stage (use 0)        (default: 0)
#
# Usage:
#   bash scripts/refine_adapter.sh
#   LANGUAGE=cpp ADAPTER_PATH=./my_adapters/cpp/0 bash scripts/refine_adapter.sh
#   LANGUAGE=python RESULTS_SOURCE=hf_hub RESULTS_DIR=my-org/calib-results bash scripts/refine_adapter.sh

export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

: "${MODEL:=Qwen/Qwen2.5-Coder-1.5B}"
: "${LANGUAGE:=python}"
: "${ADAPTER_PATH:=ankhanhtran02/lora-per-task-executable-start-4}"
: "${RESULTS_SOURCE:=hf_hub}"
: "${RESULTS_DIR:=ankhanhtran02/executed_calibration_results}"
: "${OUTPUT_DIR:=./refined_adapters/${LANGUAGE}}"
: "${CUDA_DEVICES:=4,5,6}"
: "${ZERO_STAGE:=0}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS=$(echo "$CUDA_DEVICES" | tr ',' '\n' | wc -l)
fi

set -euo pipefail

mkdir -p "$OUTPUT_DIR"
port=$(shuf -i25000-30000 -n1)

echo "[refine] ============================================"
echo "[refine] Language       : $LANGUAGE"
echo "[refine] Adapter source : $ADAPTER_PATH"
echo "[refine] Results dir    : $RESULTS_DIR ($RESULTS_SOURCE)"
echo "[refine] Output dir     : $OUTPUT_DIR"
echo "[refine] GPUs           : $NUM_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_DEVICES)"
echo "[refine] ZeRO stage     : $ZERO_STAGE"
echo "[refine] ============================================"

deepspeed --master_port "$port" --num_gpus "$NUM_GPUS" \
  training/refine_adapter.py \
    --model_name_or_path          "$MODEL" \
    --language                    "$LANGUAGE" \
    --adapter_path                "$ADAPTER_PATH" \
    --results_dir                 "$RESULTS_DIR" \
    --results_source              "$RESULTS_SOURCE" \
    --output_dir                  "$OUTPUT_DIR" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs            3 \
    --learning_rate               5e-5 \
    --weight_decay                0.01 \
    --max_prompt_len              1024 \
    --max_ans_len                 1024 \
    --zero_stage                  "$ZERO_STAGE" \
    --deepspeed \
    --run_name                    "refine_${LANGUAGE}" \
    --group_name                  "refine_adapter" \
    --logging_steps               5 \
    --seed                        42

echo "[refine] Done. Adapter saved to $OUTPUT_DIR"
