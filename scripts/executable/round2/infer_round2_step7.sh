#!/bin/bash
# Round-2 inference (disagree_explore) for step_7: tasks up to and including step 7.
#
# Samples with ≥1 correct prediction are passed through unchanged.
# Hard samples (all failed) are re-generated with greedy decoding;
# predictions are padded to 5 entries (1 real + 4 empty strings).
#
# Environment variables (all optional, defaults shown):
#   MODEL               - base LLM path or HF repo         (default: Qwen/Qwen2.5-Coder-1.5B)
#   BASE_PATH           - LoRA adapter repo or local dir    (default: ankhanhtran02/lora-per-task-executable-start-4)
#   ROUTER_PATH         - GMM input-router checkpoint dir   (default: router/ckpt_executable_dim256_comp4_vf0.001_mean)
#   PREV_RESULTS_DIR    - HF Hub repo with round-1 results  (default: ankhanhtran02/gmm_exe_vf0.02_dim256_comp4_omega1.0_soft_temp_1.0_executed)
#   TB_ROUTER_PATH      - traceback router checkpoint dir   (default: router/router_gmm_traceback_ckpt)
#   OUTPUT_DIR          - where round-2 results are saved   (default: ./inference_results/round2_disagree_explore_routing_topp_1.0/step_7)
#   TASKS               - comma-separated language list     (default: python,cpp,swift,rust,csharp,java,php,typescript)
#   CUDA_DEVICE         - GPU index                         (default: 1)
#   ROUND_NUM           - round number for output filenames (default: 2)
#
# Usage:
#   bash scripts/executable/round2/infer_round2_step7.sh

export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

: "${MODEL:=Qwen/Qwen2.5-Coder-1.5B}"
: "${BASE_PATH:=ankhanhtran02/lora-per-task-executable-start-4}"
: "${ROUTER_PATH:=router/ckpt_executable_dim256_comp4_vf0.001_mean}"
: "${PREV_RESULTS_DIR:=ankhanhtran02/gmm_exe_vf0.02_dim256_comp4_omega1.0_soft_temp_1.0_executed}"
: "${PREV_RESULTS_SOURCE:=hf_hub}"
: "${TB_ROUTER_PATH:=router/router_gmm_traceback_ckpt}"
: "${OUTPUT_DIR:=./inference_results/round2_disagree_explore_routing_topp_1.0/step_7}"
: "${TASKS:=python,cpp,swift,rust,csharp,java,php,typescript}"
: "${CUDA_DEVICE:=1}"
: "${ROUND_NUM:=2}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

set -euo pipefail

mkdir -p "$OUTPUT_DIR"

ADAPTER_PATHS=$(echo "$TASKS" | tr ',' '\n' | awk '{print $1"/0"}' | paste -sd ',' -)
MAX_PROMPT_LENS="4096,4096,4096,4096,4096,4096,4096,4096"
MAX_ANS_LENS="2048,2048,2048,2048,2048,2048,2048,2048"

echo "[round2_step7] ============================================"
echo "[round2_step7] Model          : $MODEL"
echo "[round2_step7] Base adapter   : $BASE_PATH"
echo "[round2_step7] Router         : $ROUTER_PATH"
echo "[round2_step7] Prev results   : $PREV_RESULTS_DIR ($PREV_RESULTS_SOURCE)"
echo "[round2_step7] TB router      : $TB_ROUTER_PATH"
echo "[round2_step7] Output dir     : $OUTPUT_DIR"
echo "[round2_step7] Tasks          : $TASKS"
echo "[round2_step7] ============================================"

python infer_gmm.py \
  --model_name_or_path    "$MODEL" \
  --base_path             "$BASE_PATH" \
  --inference_model_path  "$ADAPTER_PATHS" \
  --router_weight_path    "$ROUTER_PATH" \
  --benchmark             executable \
  --inference_output_path "$OUTPUT_DIR" \
  --inference_tasks       "$TASKS" \
  --routing_mode          soft \
  --routing_temperature   1.0 \
  --max_prompt_len        "$MAX_PROMPT_LENS" \
  --max_ans_len           "$MAX_ANS_LENS" \
  --inference_batch       1 \
  --num_return_sequences  1 \
  --prev_results_dir      "$PREV_RESULTS_DIR" \
  --prev_results_source   "$PREV_RESULTS_SOURCE" \
  --prev_results_subfolder step_7 \
  --round_num             "$ROUND_NUM" \
  --traceback_router_path "$TB_ROUTER_PATH" \
  --pass_through_correct \
  --pad_predictions_to    5

echo "[round2_step7] Done. Results saved to $OUTPUT_DIR"

HF_REPO="ankhanhtran02/$(basename "$(dirname "$OUTPUT_DIR")")"
echo "[round2_step7] Uploading $OUTPUT_DIR → $HF_REPO"
python upload_output_to_hf.py --output-dir "$OUTPUT_DIR" --repo-id "$HF_REPO"
