#!/usr/bin/env bash
# Experiment 1 (fast CodeTask check): does CGR reduce forgetting and enable knowledge transfer?
# One run per GPU, all runs in parallel. See README.md, "Experiment 1".
#
# Usage:  bash scripts/cgr/fast_codetask.sh            # uses GPUs 0..6
#         GPUS="0 1 2 3 4 5 6" OUT=runs/fast_codetask MAX_TRAIN=5000 MAX_TEST=500 bash scripts/cgr/fast_codetask.sh
#         MICRO_BATCH=4 INFER_BATCH=16 bash scripts/cgr/fast_codetask.sh   # if the GPUs are shared and tight
set -euo pipefail

export HF_HOME=${HF_HOME:-./.cache}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-./.cache}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # GPU numbers in GPUS match nvidia-smi

OUT=${OUT:-runs/fast_codetask}
MAX_TRAIN=${MAX_TRAIN:-5000}   # training examples per task (5% of them become the held-out slice H_k)
MAX_TEST=${MAX_TEST:-500}      # test examples per task (CoST has only 410)
MICRO_BATCH=${MICRO_BATCH:-8}  # training micro-batch (batch 32 by accumulation); halves itself automatically on OOM
INFER_BATCH=${INFER_BATCH:-32} # generation batch; splits itself automatically on OOM
MIN_FREE_MIB=${MIN_FREE_MIB:-30000}  # estimated peak per run is ~23-28 GB (KodCode training)
read -r -a G <<< "${GPUS:-0 1 2 3 4 5 6}"
if [ "${#G[@]}" -lt 7 ]; then echo "need 7 GPUs in GPUS (got ${#G[@]})" >&2; exit 1; fi

if command -v nvidia-smi >/dev/null 2>&1; then
  for g in "${G[@]:0:7}"; do
    free=$(nvidia-smi --id="$g" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    echo "[fast_codetask] GPU $g: ${free} MiB free"
    if [ "$free" -lt "$MIN_FREE_MIB" ]; then
      echo "[fast_codetask] WARNING: GPU $g has < ${MIN_FREE_MIB} MiB free; consider MICRO_BATCH=4 INFER_BATCH=16" >&2
    fi
  done
fi

COMMON=(--benchmark codetask --override "max_train_examples=${MAX_TRAIN}" "max_test_examples=${MAX_TEST}"
        "micro_batch_size=${MICRO_BATCH}" "infer_batch_size=${INFER_BATCH}")
mkdir -p "$OUT/logs"

echo "[fast_codetask] prefetching model and datasets into $HF_HOME"
python -m cgr.run_sequence "${COMMON[@]}" --output_dir "$OUT/prefetch" --prefetch_only

launch() {  # launch <gpu> <name> <args...>
  local gpu=$1 name=$2; shift 2
  echo "[fast_codetask] GPU $gpu -> $name"
  CUDA_VISIBLE_DEVICES=$gpu python -m cgr.run_sequence "${COMMON[@]}" --output_dir "$OUT/$name" "$@" \
    > "$OUT/logs/$name.log" 2>&1
}

launch "${G[0]}" A3_s1234 --method cgr --ablation A3 --seed 1234 --eval_phase_a &
launch "${G[1]}" A1_s1234 --method cgr --ablation A1 --seed 1234 &
launch "${G[2]}" SEQ_s1234 --method seqlora --seed 1234 &
launch "${G[3]}" A3_s42   --method cgr --ablation A3 --seed 42 --eval_phase_a &
launch "${G[4]}" A1_s42   --method cgr --ablation A1 --seed 42 &
launch "${G[5]}" A0_s1234 --method cgr --ablation A0 --seed 1234 &
( launch "${G[6]}" BASE --method base --seed 1234 && launch "${G[6]}" SEQ_s42 --method seqlora --seed 42 ) &

echo "[fast_codetask] all runs started; follow progress with: tail -f $OUT/logs/*.log"
wait
echo "[fast_codetask] generation finished; scoring"
bash scripts/cgr/report_fast_codetask.sh "$OUT"
