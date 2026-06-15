#!/bin/bash
# Round-2 inference: confidence-gate (conf_gate) routing.
#
# Uses the traceback router when its confidence (max(p) - entropy(p)) exceeds
# a threshold; falls back to the input router otherwise.
# Samples with ≥1 correct prediction are passed through unchanged.
# Hard samples (all failed) are re-generated with greedy decoding;
# predictions are padded to 5 entries (1 real + 4 empty strings).
#
# Environment variables (all optional, defaults shown):
#   MODEL               - base LLM path or HF repo         (default: Qwen/Qwen2.5-Coder-1.5B)
#   BASE_PATH           - LoRA adapter repo or local dir    (default: ankhanhtran02/lora-per-task-executable-start-4)
#   ROUTER_PATH         - GMM input-router checkpoint dir   (default: ./router_exe/...)
#   PREV_RESULTS_DIR    - HF Hub repo with round-1 results  (default: ankhanhtran02/executed_calibration_results)
#   TB_ROUTER_PATH      - traceback router checkpoint dir   (default: ./router_gmm_traceback_ckpt)
#   OUTPUT_DIR          - where round-2 results are saved   (default: ./inference_results/round2_conf_gate)
#   TASKS               - comma-separated language list     (default: all 9 languages)
#   CUDA_DEVICE         - GPU index                         (default: 0)
#   ROUND_NUM           - round number for output filenames (default: 2)
#   CONF_GATE_THRESH    - confidence threshold (default: 0.1; range: ~-0.5 to 0.5)
#
# Usage:
#   bash scripts/round2/infer_round2_conf_gate.sh
#   CONF_GATE_THRESH=0.2 bash scripts/round2/infer_round2_conf_gate.sh

export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

: "${MODEL:=Qwen/Qwen2.5-Coder-1.5B}"
: "${BASE_PATH:=ankhanhtran02/lora-per-task-executable-start-4}"
: "${ROUTER_PATH:=./router_exe/router_gmm_exe_vf0.02_dim_256_comp_4_omega_1.0_layer_4}"
: "${PREV_RESULTS_DIR:=ankhanhtran02/executed_calibration_results}"
: "${PREV_RESULTS_SOURCE:=hf_hub}"
: "${TB_ROUTER_PATH:=./router_gmm_traceback_ckpt}"
: "${OUTPUT_DIR:=./inference_results/round2_conf_gate}"
: "${TASKS:=python,cpp,swift,rust,csharp,java,php,typescript,shell}"
: "${CUDA_DEVICE:=2}"
: "${ROUND_NUM:=2}"
: "${CONF_GATE_THRESH:=0.1}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

set -euo pipefail

mkdir -p "$OUTPUT_DIR"

ADAPTER_PATHS=$(echo "$TASKS" | tr ',' '\n' | awk '{print $1"/0"}' | paste -sd ',' -)
MAX_PROMPT_LENS="320,320,256,130,512,256,256,256,256"
MAX_ANS_LENS="150,256,128,120,300,128,128,128,128"

echo "[round2_conf_gate] ============================================"
echo "[round2_conf_gate] Model          : $MODEL"
echo "[round2_conf_gate] Base adapter   : $BASE_PATH"
echo "[round2_conf_gate] Router         : $ROUTER_PATH"
echo "[round2_conf_gate] Prev results   : $PREV_RESULTS_DIR ($PREV_RESULTS_SOURCE)"
echo "[round2_conf_gate] TB router      : $TB_ROUTER_PATH"
echo "[round2_conf_gate] Output dir     : $OUTPUT_DIR"
echo "[round2_conf_gate] Tasks          : $TASKS"
echo "[round2_conf_gate] Conf threshold : $CONF_GATE_THRESH"
echo "[round2_conf_gate] ============================================"

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
  --round_num             "$ROUND_NUM" \
  --traceback_router_path "$TB_ROUTER_PATH" \
  --round2_routing_method conf_gate \
  --conf_gate_threshold   "$CONF_GATE_THRESH" \
  --pass_through_correct \
  --pad_predictions_to    5

echo "[round2_conf_gate] Done. Results saved to $OUTPUT_DIR"
