#!/bin/bash
# Round-2 inference: hard traceback-only (hard_tb_only) routing with refined adapters.
#
# Same as infer_round2_hard_tb_only.sh but uses locally refined adapters from
# ./refined_adapters instead of the default HF Hub base adapter.
#
# Routes exclusively to argmax of the traceback-router scores, ignoring the
# input router entirely. Useful as a pure traceback-signal baseline.
# Samples with ≥1 correct prediction are passed through unchanged.
# Hard samples (all failed) are re-generated with greedy decoding;
# predictions are padded to 5 entries (1 real + 4 empty strings).
#
# Environment variables (all optional, defaults shown):
#   MODEL               - base LLM path or HF repo         (default: Qwen/Qwen2.5-Coder-1.5B)
#   BASE_PATH           - LoRA adapter repo or local dir    (default: ./refined_adapters)
#   ROUTER_PATH         - GMM input-router checkpoint dir   (default: ./router_exe/...)
#   PREV_RESULTS_DIR    - HF Hub repo with round-1 results  (default: ankhanhtran02/gmm_exe_vf0.02_dim256_comp4_omega1.0_soft_temp_1.0_executed)
#   TB_ROUTER_PATH      - traceback router checkpoint dir   (default: ./router_gmm_traceback_ckpt)
#   OUTPUT_DIR          - where round-2 results are saved   (default: /inference_results/round2_hard_tb_only_routing_topp_1.0/step_8)
#   TASKS               - comma-separated language list     (default: all 9 languages)
#   CUDA_DEVICE         - GPU index                         (default: 7)
#   ROUND_NUM           - round number for output filenames (default: 2)
#
# Usage:
#   bash scripts/round2/infer_round2_hard_tb_only_refined.sh

export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

: "${MODEL:=Qwen/Qwen2.5-Coder-1.5B}"
: "${BASE_PATH:=./refined_adapters}"
: "${ROUTER_PATH:=router/ckpt_executable_dim256_comp4_vf0.001_mean}"
: "${PREV_RESULTS_DIR:=ankhanhtran02/gmm_exe_vf0.02_dim256_comp4_omega1.0_soft_temp_1.0_executed}"
: "${PREV_RESULTS_SOURCE:=hf_hub}"
: "${TB_ROUTER_PATH:=./router_gmm_traceback_ckpt}"
: "${OUTPUT_DIR:=/inference_results/round2_hard_tb_only_refined_routing_topp_1.0/step_8}"
: "${TASKS:=python,cpp,swift,rust,csharp,java,php,typescript,shell}"
: "${CUDA_DEVICE:=1}"
: "${ROUND_NUM:=2}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

set -euo pipefail

mkdir -p "$OUTPUT_DIR"

ADAPTER_PATHS=$(echo "$TASKS" | tr ',' '\n' | paste -sd ',' -)
MAX_PROMPT_LENS="4096,4096,4096,4096,4096,4096,4096,4096,4096"
MAX_ANS_LENS="2048,2048,2048,2048,2048,2048,2048,2048,2048"

echo "[round2_hard_tb_only_refined] ============================================"
echo "[round2_hard_tb_only_refined] Model          : $MODEL"
echo "[round2_hard_tb_only_refined] Base adapter   : $BASE_PATH"
echo "[round2_hard_tb_only_refined] Router         : $ROUTER_PATH"
echo "[round2_hard_tb_only_refined] Prev results   : $PREV_RESULTS_DIR ($PREV_RESULTS_SOURCE)"
echo "[round2_hard_tb_only_refined] TB router      : $TB_ROUTER_PATH"
echo "[round2_hard_tb_only_refined] Output dir     : $OUTPUT_DIR"
echo "[round2_hard_tb_only_refined] Tasks          : $TASKS"
echo "[round2_hard_tb_only_refined] ============================================"

python infer_gmm.py \
  --model_name_or_path    "$MODEL" \
  --base_path             "$BASE_PATH" \
  --inference_model_path  "$ADAPTER_PATHS" \
  --router_weight_path    "$ROUTER_PATH" \
  --benchmark             executable \
  --inference_output_path "$OUTPUT_DIR" \
  --inference_tasks       "$TASKS" \
  --routing_mode          hard \
  --routing_temperature   1.0 \
  --routing_top_p         1 \
  --max_prompt_len        "$MAX_PROMPT_LENS" \
  --max_ans_len           "$MAX_ANS_LENS" \
  --inference_batch       1 \
  --num_return_sequences  1 \
  --prev_results_dir      "$PREV_RESULTS_DIR" \
  --prev_results_source   "$PREV_RESULTS_SOURCE" \
  --prev_results_subfolder step_8 \
  --round_num             "$ROUND_NUM" \
  --traceback_router_path "$TB_ROUTER_PATH" \
  --round2_routing_method hard_tb_only \
  --pass_through_correct \
  --pad_predictions_to    5

echo "[round2_hard_tb_only_refined] Done. Results saved to $OUTPUT_DIR"

HF_REPO="ankhanhtran02/$(basename "$(dirname "$OUTPUT_DIR")")"
echo "[round2_hard_tb_only_refined] Uploading $OUTPUT_DIR → $HF_REPO"
python upload_output_to_hf.py --output-dir "$OUTPUT_DIR" --repo-id "$HF_REPO"
