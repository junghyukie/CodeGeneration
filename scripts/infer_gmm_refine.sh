#!/bin/bash
# Iterative refinement for GMM-router inference on the executable benchmark.
#
# Round 1 (standard inference, saves input_router_scores per sample):
#   bash scripts/infer_gmm_refine.sh
#
# Round 2+ (refine samples where ALL round-N-1 predictions failed):
#   PREV_RESULTS_DIR=./inference_results ROUND_NUM=2 bash scripts/infer_gmm_refine.sh
#   PREV_RESULTS_DIR=./inference_results ROUND_NUM=3 bash scripts/infer_gmm_refine.sh
#
# Environment variables (all optional, defaults shown):
#   MODEL              - base LLM path or HF repo        (default: Qwen/Qwen2.5-Coder-1.5B)
#   BASE_PATH          - LoRA adapter repo or local dir   (default: ankhanhtran02/lora-per-task-executable-start-4)
#   ROUTER_PATH        - GMM input-router checkpoint dir  (default: ./router_gmm_ckpt)
#   ROUTER_STEP        - router_step{N}.pt to load        (default: last step = len(tasks)-1)
#   OUTPUT_DIR         - where results-{i}-{task}.json go (default: ./inference_results)
#   TASKS              - comma-separated task list        (default: all 9 languages)
#   ROUTING_MODE       - hard | soft                      (default: hard)
#   ROUTING_TEMP       - softmax temperature (soft mode)  (default: 1.0)
#   NUM_RETURN         - predictions per sample (round 1) (default: 5)
#   DO_SAMPLE          - 1 = enable sampling (round 1)    (default: 0)
#   TEMPERATURE        - sampling temperature             (default: 0.2)
#   TOP_P              - nucleus sampling p               (default: 0.95)
#   CUDA_DEVICE        - GPU index                        (default: 0)
#
# Round-2+ only variables:
#   PREV_RESULTS_DIR   - dir with previous-round results  (required for round 2+)
#   ROUND_NUM          - which round to generate           (default: 2)
#   TB_ROUTER_PATH     - traceback router checkpoint dir   (required for round 2+)
#   TB_ROUTER_STEP     - router_step{N}.pt in TB router   (default: last step)
#   ROUND2_METHOD      - routing method (see routing.md)  (default: conf_linear)
#                        choices: poe | conf_linear | disagree_explore |
#                                 geo_interp | tb_mask | hard_poe | conf_gate
#   ROUND2_T_INPUT     - [poe] input-router temperature    (default: 1.0)
#   ROUND2_T_TRACE     - [poe] traceback-router temperature(default: 1.0)
#   ROUND2_ALPHA       - [geo_interp] alpha ∈ [0,1]       (default: 0.3)
#   CONF_GATE_THRESH   - [conf_gate] confidence threshold  (default: 0.1)

export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

: "${MODEL:=Qwen/Qwen2.5-Coder-1.5B}"
: "${BASE_PATH:=ankhanhtran02/lora-per-task-executable-start-4}"
: "${ROUTER_PATH:=./router_gmm_ckpt}"
: "${OUTPUT_DIR:=./inference_results}"
: "${TASKS:=python,cpp,swift,rust,csharp,java,php,typescript,shell}"
: "${ROUTING_MODE:=hard}"
: "${ROUTING_TEMP:=1.0}"
: "${NUM_RETURN:=5}"
: "${DO_SAMPLE:=0}"
: "${TEMPERATURE:=0.2}"
: "${TOP_P:=0.95}"
: "${CUDA_DEVICE:=0}"
# Round-2+ defaults
: "${ROUND_NUM:=2}"
: "${ROUND2_METHOD:=conf_linear}"
: "${ROUND2_T_INPUT:=1.0}"
: "${ROUND2_T_TRACE:=1.0}"
: "${ROUND2_ALPHA:=0.3}"
: "${CONF_GATE_THRESH:=0.1}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

set -euo pipefail

mkdir -p "$OUTPUT_DIR"

# Build inference_model_path from TASKS
# Assumes subfolder layout: {language}/0 inside BASE_PATH
ADAPTER_PATHS=$(echo "$TASKS" | tr ',' '\n' | awk '{print $1"/0"}' | paste -sd ',' -)

# Build max_prompt_len and max_ans_len aligned with TASKS
# Default values match infer_gmm.py defaults for the 9 executable languages
# (python cpp swift rust csharp java php typescript shell)
MAX_PROMPT_LENS="320,320,256,130,512,256,256,256,256"
MAX_ANS_LENS="150,256,128,120,300,128,128,128,128"

SAMPLE_FLAG=""
[ "$DO_SAMPLE" = "1" ] && SAMPLE_FLAG="--do_sample"

echo "[infer_gmm_refine] ============================================"
echo "[infer_gmm_refine] Model          : $MODEL"
echo "[infer_gmm_refine] Base adapter   : $BASE_PATH"
echo "[infer_gmm_refine] Router         : $ROUTER_PATH"
echo "[infer_gmm_refine] Output dir     : $OUTPUT_DIR"
echo "[infer_gmm_refine] Tasks          : $TASKS"
echo "[infer_gmm_refine] Routing mode   : $ROUTING_MODE"
if [ -n "${PREV_RESULTS_DIR:-}" ]; then
  echo "[infer_gmm_refine] *** Round $ROUND_NUM mode ***"
  echo "[infer_gmm_refine] Prev results   : $PREV_RESULTS_DIR"
  echo "[infer_gmm_refine] TB router      : ${TB_ROUTER_PATH:-NOT SET}"
  echo "[infer_gmm_refine] Round2 method  : $ROUND2_METHOD"
fi
echo "[infer_gmm_refine] ============================================"

# ── Build common args ──────────────────────────────────────────────────────────

COMMON_ARGS=(
  --model_name_or_path   "$MODEL"
  --base_path            "$BASE_PATH"
  --inference_model_path "$ADAPTER_PATHS"
  --router_weight_path   "$ROUTER_PATH"
  --benchmark            executable
  --inference_output_path "$OUTPUT_DIR"
  --inference_tasks      "$TASKS"
  --routing_mode         "$ROUTING_MODE"
  --routing_temperature  "$ROUTING_TEMP"
  --max_prompt_len       "$MAX_PROMPT_LENS"
  --max_ans_len          "$MAX_ANS_LENS"
  --inference_batch      1
  --temperature          "$TEMPERATURE"
  --top_p                "$TOP_P"
  --num_return_sequences "$NUM_RETURN"
  ${SAMPLE_FLAG}
)

# ── Round-1 or round-2+ ────────────────────────────────────────────────────────

if [ -n "${PREV_RESULTS_DIR:-}" ]; then
  # Round 2+ — requires TB_ROUTER_PATH
  if [ -z "${TB_ROUTER_PATH:-}" ]; then
    echo "[ERROR] TB_ROUTER_PATH must be set for round-2+ inference."
    exit 1
  fi

  # Method-specific args
  METHOD_ARGS=()
  case "$ROUND2_METHOD" in
    poe)
      METHOD_ARGS+=(--round2_T_input "$ROUND2_T_INPUT" --round2_T_trace "$ROUND2_T_TRACE")
      ;;
    geo_interp)
      METHOD_ARGS+=(--round2_alpha "$ROUND2_ALPHA")
      ;;
    conf_gate)
      METHOD_ARGS+=(--conf_gate_threshold "$CONF_GATE_THRESH")
      ;;
  esac

  TB_STEP_ARG=()
  [ -n "${TB_ROUTER_STEP:-}" ] && TB_STEP_ARG+=(--traceback_router_step "$TB_ROUTER_STEP")

  python infer_gmm.py \
    "${COMMON_ARGS[@]}" \
    --prev_results_dir     "$PREV_RESULTS_DIR" \
    --round_num            "$ROUND_NUM" \
    --traceback_router_path "$TB_ROUTER_PATH" \
    "${TB_STEP_ARG[@]}" \
    --round2_routing_method "$ROUND2_METHOD" \
    "${METHOD_ARGS[@]}"

else
  # Round 1 — standard inference
  python infer_gmm.py "${COMMON_ARGS[@]}"
fi

echo "[infer_gmm_refine] Done. Results saved to $OUTPUT_DIR"
