#!/usr/bin/env bash
# Install dependencies, then fit the GMM router for the baseline order and
# four requested permutations.
# Usage: bash scripts/codetask/train_gmm_all_permutations.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export PYTHON_BIN=${PYTHON_BIN:-python}
SCENARIOS=(original permutation_1 permutation_2 permutation_3 permutation_4)

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python command not found: $PYTHON_BIN" >&2
  echo "Install Python 3, or set PYTHON_BIN to its executable path." >&2
  exit 1
fi

cd "$REPO_ROOT"

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  echo "pip is missing; bootstrapping it with ensurepip..."
  "$PYTHON_BIN" -m ensurepip --upgrade
fi

echo "Installing Python dependencies from requirements.txt..."
"$PYTHON_BIN" -m pip install --requirement requirements.txt

for scenario in "${SCENARIOS[@]}"; do
  echo "================================================================"
  echo "Starting ${scenario}"
  echo "================================================================"
  bash "$SCRIPT_DIR/train_gmm_permutation.sh" "$scenario"
done
