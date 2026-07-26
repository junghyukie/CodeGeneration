#!/usr/bin/env bash
# Fit the GMM router for the baseline order and four requested permutations.
# Usage: bash scripts/codetask/train_gmm_all_permutations.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCENARIOS=(original permutation_1 permutation_2 permutation_3 permutation_4)

for scenario in "${SCENARIOS[@]}"; do
  echo "================================================================"
  echo "Starting ${scenario}"
  echo "================================================================"
  bash "$SCRIPT_DIR/train_gmm_permutation.sh" "$scenario"
done
