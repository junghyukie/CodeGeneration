#!/bin/bash

export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

: "${MODEL:=Qwen/Qwen2.5-Coder-1.5B}"
: "${BASE_PATH:=ankhanhtran02/lora-per-task-executable-start-4}"
: "${ROUTER_PATH:=ankhanhtran02/router_ckpt_executable_dim256_comp4_vf0.001_mean}"
: "${TB_ROUTER_PATH:=ankhanhtran02/router_gmm_traceback_ckpt}"
: "${DEVICE:=cpu}"
# python cpp swift rust csharp java php typescript shell
: "${LANGUAGES:=python cpp swift rust csharp java php typescript shell}"
: "${DEMO_DATASET_PATH:=demo_data/executable/mixed.json}"
: "${OUTPUT_ROOT:=./inference_results/demo_pipeline}"
: "${NUM_RETURN_SEQUENCES:=5}"
: "${KS:=1 5}"
: "${MAX_WORKERS:=1}"

TASKS="python,cpp,swift,rust,csharp,java,php,typescript,shell"

set -euo pipefail

if [ ! -f "$DEMO_DATASET_PATH" ]; then
  echo "[demo_pipeline] ERROR: demo dataset not found at $DEMO_DATASET_PATH" >&2
  exit 1
fi

ADAPTER_PATHS=$(echo "$TASKS" | tr ',' '\n' | awk '{print $1"/0"}' | paste -sd ',' -)
MAX_PROMPT_LENS="4096,4096,4096,4096,4096,4096,4096,4096,4096"
MAX_ANS_LENS="2048,2048,2048,2048,2048,2048,2048,2048,2048"

# i = index of the final continual-learning step = (number of tasks) - 1.
# Router/adapter checkpoints are keyed by this step, so all 9 languages must
# stay in the adapter/router list even though only the demo language(s) run.
NUM_TASKS=$(echo "$TASKS" | tr ',' '\n' | wc -l)
STEP_I=$((NUM_TASKS - 1))

ROUND1_DIR="$OUTPUT_ROOT/round1"
ROUND2_DIR="$OUTPUT_ROOT/round2"
mkdir -p "$ROUND1_DIR" "$ROUND2_DIR"

echo "[demo_pipeline] ================================================"
echo "[demo_pipeline] Languages       : $LANGUAGES"
echo "[demo_pipeline] Demo dataset    : $DEMO_DATASET_PATH"
echo "[demo_pipeline] Device          : $DEVICE"
echo "[demo_pipeline] ================================================"

echo
echo "[demo_pipeline] ---- Step 1/4: round-1 inference ----"
python infer_gmm.py \
  --model_name_or_path    "$MODEL" \
  --base_path             "$BASE_PATH" \
  --inference_model_path  "$ADAPTER_PATHS" \
  --router_weight_path    "$ROUTER_PATH" \
  --benchmark             executable \
  --inference_output_path "$ROUND1_DIR" \
  --inference_tasks       "$TASKS" \
  --routing_mode          soft \
  --routing_temperature   1.0 \
  --max_prompt_len        "$MAX_PROMPT_LENS" \
  --max_ans_len           "$MAX_ANS_LENS" \
  --inference_batch       1 \
  --do_sample \
  --temperature           0.2 \
  --top_p                 0.95 \
  --num_return_sequences  "$NUM_RETURN_SEQUENCES" \
  --repetition_penalty    1.0 \
  --seed                  42 \
  --device                "$DEVICE" \
  --demo_dataset_path     "$DEMO_DATASET_PATH"

echo
echo "[demo_pipeline] ---- Step 2/4: evaluate round-1 results ----"
for LANG in $LANGUAGES; do
  ROUND1_FILE="$ROUND1_DIR/results-${STEP_I}-${LANG}.json"
  if [ ! -f "$ROUND1_FILE" ]; then
    echo "[demo_pipeline] Skipping $LANG: $ROUND1_FILE not found (no matching rows in demo dataset)."
    continue
  fi
  echo "[demo_pipeline] Evaluating $LANG -> $ROUND1_FILE"
  python executable_dataset/calc_metrics.py \
    --local-file        "$ROUND1_FILE" \
    --local-test-source "$DEMO_DATASET_PATH" \
    --num-samples       "$NUM_RETURN_SEQUENCES" \
    --ks $KS \
    --max-workers       "$MAX_WORKERS"
done

echo
echo "[demo_pipeline] ---- Step 3/4: round-2 disagree_explore inference ----"
python infer_gmm.py \
  --model_name_or_path    "$MODEL" \
  --base_path             "$BASE_PATH" \
  --inference_model_path  "$ADAPTER_PATHS" \
  --router_weight_path    "$ROUTER_PATH" \
  --benchmark             executable \
  --inference_output_path "$ROUND2_DIR" \
  --inference_tasks       "$TASKS" \
  --routing_mode          soft \
  --routing_temperature   1.0 \
  --max_prompt_len        "$MAX_PROMPT_LENS" \
  --max_ans_len           "$MAX_ANS_LENS" \
  --inference_batch       1 \
  --num_return_sequences  1 \
  --device                "$DEVICE" \
  --prev_results_dir      "$ROUND1_DIR" \
  --prev_results_source   local \
  --round_num             2 \
  --traceback_router_path "$TB_ROUTER_PATH" \
  --pass_through_correct \
  --pad_predictions_to    "$NUM_RETURN_SEQUENCES"

echo
echo "[demo_pipeline] ---- Step 4/4: evaluate round-2 results ----"
for LANG in $LANGUAGES; do
  ROUND2_FILE="$ROUND2_DIR/results-${STEP_I}-${LANG}-round2.json"
  if [ ! -f "$ROUND2_FILE" ]; then
    echo "[demo_pipeline] Skipping $LANG: $ROUND2_FILE not found (no round-1 results to refine)."
    continue
  fi
  echo "[demo_pipeline] Evaluating $LANG -> $ROUND2_FILE"
  python executable_dataset/calc_metrics.py \
    --local-file        "$ROUND2_FILE" \
    --local-test-source "$DEMO_DATASET_PATH" \
    --num-samples       "$NUM_RETURN_SEQUENCES" \
    --ks $KS \
    --max-workers       "$MAX_WORKERS"
done

echo
echo "[demo_pipeline] ================================================"
echo "[demo_pipeline] Done. Intermediate files:"
for LANG in $LANGUAGES; do
  echo "[demo_pipeline]   $LANG round-1 : $ROUND1_DIR/results-${STEP_I}-${LANG}.json"
  echo "[demo_pipeline]   $LANG round-2 : $ROUND2_DIR/results-${STEP_I}-${LANG}-round2.json"
done
echo "[demo_pipeline] ================================================"
