#!/bin/bash
# Round-2 demo: JSD-gated posterior + uniform (disagree_explore) routing on a
# single real round-1 sample, instead of the full executable test set.
#
# Runs on CPU by default — no GPU required. This is a quick, single-sample
# sanity check of the disagree_explore routing pipeline (input router +
# traceback router), not a full benchmark run.
#
# How "single sample only" works (no changes needed to infer_gmm.py):
# infer_gmm.py's round-2 path (prediction_round2) reads a previous round's
# results file, selects samples where every prior prediction failed, and
# regenerates just those. This script writes a LOCAL "previous round"
# results file containing exactly one row — a real hard round-1 sample
# (Rust task, all 5 predictions failed on a rustc compile error) — under the
# filename infer_gmm.py expects (results-<step>-<language>.json). No results
# file exists for the other 8 languages, so infer_gmm.py skips them
# entirely and only this one sample gets regenerated.
#
# Environment variables (all optional, defaults shown):
#   MODEL               - base LLM path or HF repo         (default: Qwen/Qwen2.5-Coder-1.5B)
#   BASE_PATH           - LoRA adapter repo or local dir    (default: ankhanhtran02/lora-per-task-executable-start-4)
#   ROUTER_PATH         - GMM input-router HF repo          (default: ankhanhtran02/router_ckpt_executable_dim256_comp4_vf0.001_mean)
#   TB_ROUTER_PATH      - traceback router HF repo          (default: ankhanhtran02/router_gmm_traceback_ckpt)
#   OUTPUT_DIR          - where the demo result is saved    (default: ./inference_results/round2_disagree_explore_demo/step_8)
#   SAMPLE_LANGUAGE     - language of the demo sample       (default: rust)
#   DEVICE              - torch device for infer_gmm.py     (default: cpu)
#   ROUND_NUM           - round number for output filenames (default: 2)
#
# Usage:
#   bash scripts/executable/round2/infer_round2_disagree_explore_demo.sh
#   DEVICE=cuda bash scripts/executable/round2/infer_round2_disagree_explore_demo.sh   # to use a GPU instead

export HF_HOME=./.cache
export HF_DATASETS_CACHE=./.cache

: "${MODEL:=Qwen/Qwen2.5-Coder-1.5B}"
: "${BASE_PATH:=ankhanhtran02/lora-per-task-executable-start-4}"
: "${ROUTER_PATH:=ankhanhtran02/router_ckpt_executable_dim256_comp4_vf0.001_mean}"
: "${TB_ROUTER_PATH:=ankhanhtran02/router_gmm_traceback_ckpt}"
: "${OUTPUT_DIR:=./inference_results/round2_disagree_explore_demo/step_8}"
: "${SAMPLE_LANGUAGE:=rust}"
: "${DEVICE:=cpu}"
: "${ROUND_NUM:=2}"

TASKS="python,cpp,swift,rust,csharp,java,php,typescript,shell"

set -euo pipefail

if ! echo ",$TASKS," | grep -q ",$SAMPLE_LANGUAGE,"; then
  echo "[round2_disagree_explore_demo] ERROR: SAMPLE_LANGUAGE='$SAMPLE_LANGUAGE' not in TASKS list ($TASKS)" >&2
  exit 1
fi

PREV_RESULTS_DIR="$OUTPUT_DIR/demo_prev_results"
mkdir -p "$OUTPUT_DIR" "$PREV_RESULTS_DIR"

# i = index of the final continual-learning step = (number of tasks) - 1.
# Router/adapter checkpoints are keyed by this step, so all 9 languages must
# stay in the adapter/router list even though only one sample is generated.
NUM_TASKS=$(echo "$TASKS" | tr ',' '\n' | wc -l)
STEP_I=$((NUM_TASKS - 1))

ADAPTER_PATHS=$(echo "$TASKS" | tr ',' '\n' | awk '{print $1"/0"}' | paste -sd ',' -)
MAX_PROMPT_LENS="4096,4096,4096,4096,4096,4096,4096,4096,4096"
MAX_ANS_LENS="2048,2048,2048,2048,2048,2048,2048,2048,2048"

# Real round-1 "hard" sample (all 5 predictions failed with a rustc compile
# error) — this is the only file infer_gmm.py's round-2 path will find.
cat > "$PREV_RESULTS_DIR/results-${STEP_I}-${SAMPLE_LANGUAGE}.json" <<'JSON_EOF'
{
  "predictions": [
    {
      "source": "Write a Rust function `fn generate_dividing_sequence(initial_value: i32) -> Vec<i32>` to solve the following problem:\nGenerates a sequence of integers where each subsequent integer is half of the previous one, rounded down, starting from an initial value.\n\n  Arguments:\n  - initial_value: An integer representing the starting value of the sequence.\n\n  Returns:\n  - A vector of integers representing the sequence where each element is half of its predecessor, stopping when the value becomes less than 1.\n\n  Example:\n  - generate_dividing_sequence(20) returns [20, 10, 5, 2, 1]\n  - generate_dividing_sequence(5) returns [5, 2, 1]\n  - generate_dividing_sequence(1) returns [1]",
      "ground-truth": null,
      "prediction": [
        "fn generate_dividing_sequence(initial_value: i32) -> Vec<i32> {\n    let mut sequence = Vec::new();\n    let mut current_value = initial_value;\n\n    while current_value >= 1 {\n        sequence.push(current_value);\n        current_value = current_value / 2;\n    }\n\n    sequence\n}",
        "fn generate_dividing_sequence(initial_value: i32) -> Vec<i32> {\n    let mut sequence = Vec::new();\n    let mut current_value = initial_value;\n\n    while current_value >= 1 {\n        sequence.push(current_value);\n        current_value = current_value / 2;\n    }\n\n    sequence\n}",
        "fn generate_dividing_sequence(initial_value: i32) -> Vec<i32> {\n    let mut sequence = Vec::new();\n    let mut current_value = initial_value;\n\n    while current_value >= 1 {\n        sequence.push(current_value);\n        current_value /= 2;\n    }\n\n    sequence\n}",
        "fn generate_dividing_sequence(initial_value: i32) -> Vec<i32> {\n    let mut sequence = Vec::new();\n    let mut current_value = initial_value;\n\n    while current_value >= 1 {\n        sequence.push(current_value);\n        current_value = current_value / 2;\n    }\n\n    sequence\n}",
        "fn generate_dividing_sequence(initial_value: i32) -> Vec<i32> {\n    let mut sequence = Vec::new();\n    let mut current_value = initial_value;\n\n    while current_value >= 1 {\n        sequence.push(current_value);\n        current_value /= 2;\n    }\n\n    sequence\n}"
      ],
      "moe_id": {
        "python": 0.0,
        "cpp": 1.0,
        "swift": 0.0,
        "rust": 0.0,
        "csharp": 0.0,
        "java": 0.0,
        "php": 0.0,
        "typescript": 0.0,
        "shell": 0.0
      },
      "input_router_scores": [
        202.45773315429688,
        221.02719116210938,
        184.7416534423828,
        191.91542053222656,
        156.8226318359375,
        179.98880004882812,
        141.3209686279297,
        155.555908203125,
        64.8255386352539
      ],
      "test": "   \n#[cfg(test)]\nmod tests {\n    use super::*;\n \n    #[test]\n    fn main() {\n        assert_eq!(generate_dividing_sequence(20), vec![20, 10, 5, 2, 1]);\n        assert_eq!(generate_dividing_sequence(5), vec![5, 2, 1]);\n        assert_eq!(generate_dividing_sequence(1), vec![1]);\n        assert_eq!(generate_dividing_sequence(15), vec![15, 7, 3, 1]);        \n    }\n    \n\n}\n ",
      "passed": [
        0,
        0,
        0,
        0,
        0
      ],
      "stderr": [
        "error[E0601]: `main` function not found in crate `main`\n  --> main.rs:28:2\n   |\n28 | }\n   |  ^ consider adding a `main` function to `main.rs`\n\nerror: aborting due to 1 previous error\n\nFor more information about this error, try `rustc --explain E0601`.\n",
        "error[E0601]: `main` function not found in crate `main`\n  --> main.rs:28:2\n   |\n28 | }\n   |  ^ consider adding a `main` function to `main.rs`\n\nerror: aborting due to 1 previous error\n\nFor more information about this error, try `rustc --explain E0601`.\n",
        "error[E0601]: `main` function not found in crate `main`\n  --> main.rs:28:2\n   |\n28 | }\n   |  ^ consider adding a `main` function to `main.rs`\n\nerror: aborting due to 1 previous error\n\nFor more information about this error, try `rustc --explain E0601`.\n",
        "error[E0601]: `main` function not found in crate `main`\n  --> main.rs:28:2\n   |\n28 | }\n   |  ^ consider adding a `main` function to `main.rs`\n\nerror: aborting due to 1 previous error\n\nFor more information about this error, try `rustc --explain E0601`.\n",
        "error[E0601]: `main` function not found in crate `main`\n  --> main.rs:28:2\n   |\n28 | }\n   |  ^ consider adding a `main` function to `main.rs`\n\nerror: aborting due to 1 previous error\n\nFor more information about this error, try `rustc --explain E0601`.\n"
      ],
      "num_passed": 0
    }
  ]
}
JSON_EOF

echo "[round2_disagree_explore_demo] ============================================"
echo "[round2_disagree_explore_demo] Model          : $MODEL"
echo "[round2_disagree_explore_demo] Base adapter   : $BASE_PATH"
echo "[round2_disagree_explore_demo] Router         : $ROUTER_PATH"
echo "[round2_disagree_explore_demo] TB router      : $TB_ROUTER_PATH"
echo "[round2_disagree_explore_demo] Device         : $DEVICE"
echo "[round2_disagree_explore_demo] Sample language: $SAMPLE_LANGUAGE"
echo "[round2_disagree_explore_demo] Prev results   : $PREV_RESULTS_DIR (local, 1 sample)"
echo "[round2_disagree_explore_demo] Output dir     : $OUTPUT_DIR"
echo "[round2_disagree_explore_demo] ============================================"

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
  --device                "$DEVICE" \
  --prev_results_dir      "$PREV_RESULTS_DIR" \
  --prev_results_source   local \
  --round_num             "$ROUND_NUM" \
  --traceback_router_path "$TB_ROUTER_PATH" \
  --pass_through_correct \
  --pad_predictions_to    5

echo "[round2_disagree_explore_demo] Done. Result saved to $OUTPUT_DIR/results-${STEP_I}-${SAMPLE_LANGUAGE}-round${ROUND_NUM}.json"
