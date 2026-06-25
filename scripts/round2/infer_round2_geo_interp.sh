#!/bin/bash
# Round-2 inference: geometric interpolation (geo_interp) routing.
#
# Blends input-router and traceback-router via geometric interpolation
# controlled by alpha (0 = pure input router, 1 = pure traceback router).
# Samples with ≥1 correct prediction are passed through unchanged.
# Hard samples (all failed) are re-generated with greedy decoding;
# predictions are padded to 5 entries (1 real + 4 empty strings).
#
# Environment variables (all optional, defaults shown):
#   MODEL               - base LLM path or HF repo         (default: Qwen/Qwen2.5-Coder-1.5B)
#   BASE_PATH           - LoRA adapter repo or local dir    (default: ankhanhtran02/lora-per-task-executable-start-4)
#   ROUTER_PATH         - GMM input-router checkpoint dir   (default: ./router_exe/...)
#   PREV_RESULTS_DIR    - HF Hub repo with round-1 results  (default: ankhanhtran02/gmm_exe_vf0.02_dim256_comp4_omega1.0_soft_temp_1.0_executed)
#   TB_ROUTER_PATH      - traceback router checkpoint dir   (default: ./router_gmm_traceback_ckpt)
#   OUTPUT_DIR          - where round-2 results are saved   (default: ./inference_results/round2_geo_interp)
#   TASKS               - comma-separated language list     (default: all 9 languages)
#   CUDA_DEVICE         - GPU index                         (default: 0)
#   ROUND_NUM           - round number for output filenames (default: 2)
#   ROUND2_ALPHA        - interpolation weight for TB router (default: 0.3)
#
# Usage:
#   bash scripts/round2/infer_round2_geo_interp.sh
#   ROUND2_ALPHA=0.5 bash scripts/round2/infer_round2_geo_interp.sh

export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

: "${MODEL:=Qwen/Qwen2.5-Coder-1.5B}"
: "${BASE_PATH:=ankhanhtran02/lora-per-task-executable-start-4}"
: "${ROUTER_PATH:=router/ckpt_executable_dim256_comp4_vf0.001_mean}"
: "${PREV_RESULTS_DIR:=ankhanhtran02/gmm_exe_vf0.02_dim256_comp4_omega1.0_soft_temp_1.0_executed}"
: "${PREV_RESULTS_SOURCE:=hf_hub}"
: "${TB_ROUTER_PATH:=./router_gmm_traceback_ckpt}"
: "${OUTPUT_DIR:=./inference_results/round2_geo_interp}"
: "${TASKS:=python,cpp,swift,rust,csharp,java,php,typescript,shell}"
: "${CUDA_DEVICE:=3}"
: "${ROUND_NUM:=2}"
: "${ROUND2_ALPHA:=0.3}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

set -euo pipefail

mkdir -p "$OUTPUT_DIR"

ADAPTER_PATHS=$(echo "$TASKS" | tr ',' '\n' | awk '{print $1"/0"}' | paste -sd ',' -)
MAX_PROMPT_LENS="4096,4096,4096,4096,4096,4096,4096,4096,4096"
MAX_ANS_LENS="2048,2048,2048,2048,2048,2048,2048,2048,2048"

echo "[round2_geo_interp] ============================================"
echo "[round2_geo_interp] Model          : $MODEL"
echo "[round2_geo_interp] Base adapter   : $BASE_PATH"
echo "[round2_geo_interp] Router         : $ROUTER_PATH"
echo "[round2_geo_interp] Prev results   : $PREV_RESULTS_DIR ($PREV_RESULTS_SOURCE)"
echo "[round2_geo_interp] TB router      : $TB_ROUTER_PATH"
echo "[round2_geo_interp] Output dir     : $OUTPUT_DIR"
echo "[round2_geo_interp] Tasks          : $TASKS"
echo "[round2_geo_interp] Alpha          : $ROUND2_ALPHA"
echo "[round2_geo_interp] ============================================"

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
  --prev_results_subfolder step_8 \
  --round_num             "$ROUND_NUM" \
  --traceback_router_path "$TB_ROUTER_PATH" \
  --round2_routing_method geo_interp \
  --round2_alpha          "$ROUND2_ALPHA" \
  --pass_through_correct \
  --pad_predictions_to    5

echo "[round2_geo_interp] Done. Results saved to $OUTPUT_DIR"

HF_REPO="ankhanhtran02/$(basename "$OUTPUT_DIR")"
echo "[round2_geo_interp] Uploading $OUTPUT_DIR → $HF_REPO"
python upload_output_to_hf.py --output-dir "$OUTPUT_DIR" --repo-id "$HF_REPO"
