# GMM router task-order experiments

This directory contains two Bash launchers for continual GMM-router fitting on
the eight CodeTask datasets. Each run fits one router task-by-task in the
specified order; it is not eight independent model-training runs.

## Requirements

- Linux, WSL, or another environment with Bash.
- Python with PyTorch. A CUDA GPU is recommended; `gmm.py` will use one
  visible GPU when available and otherwise falls back to CPU.
- Python dependencies installed from the repository root:

  ```bash
  pip install -r requirements.txt
  ```

- Access to Hugging Face to download the default base model
  `Qwen/Qwen2.5-Coder-1.5B` and the CodeTask datasets. Log in with
  `huggingface-cli login` if your environment requires authentication.
- Enough disk space for the Hugging Face cache and five independent output
  directories when running every scenario.

Run commands from the repository root so `training/main_anamoe.py` resolves
correctly.

## Task key

| ID | Task | Dataset name used by the launcher |
| --- | --- | --- |
| A | Java code generation | `CONCODE` |
| B | Java → C# translation | `CodeTrans` |
| C | Ruby summarization | `CodeSearchNet` |
| D | Java refinement | `BFP` |
| E | Python code generation | `KodCode` |
| F | Ruby refinement | `RunBugRun` |
| G | C# summarization | `TheVault_Csharp` |
| H | C++ → C# translation | `CoST` |

## Run one ordering

Use `train_gmm_permutation.sh` with one of these scenario names:

| Scenario | Task order |
| --- | --- |
| `original` | A → B → C → D → E → F → G → H |
| `permutation_1` | A → B → H → C → G → D → F → E |
| `permutation_2` | C → D → B → E → A → F → H → G |
| `permutation_3` | E → F → D → G → C → H → B → A |
| `permutation_4` | G → H → F → A → E → B → D → C |

For example:

```bash
bash scripts/codetask/train_gmm_permutation.sh permutation_1
```

The script applies each task's own prompt/answer token lengths after changing
the ordering, so lengths always remain aligned to the dataset rather than to
its position in the sequence.

## Run all five orderings

```bash
bash scripts/codetask/train_gmm_all_permutations.sh
```

The scenarios run sequentially in this order: `original`, `permutation_1`,
`permutation_2`, `permutation_3`, then `permutation_4`. The launcher stops at
the first failed scenario because it uses `set -e`.

## Outputs and configuration

By default, every scenario writes independently to:

```text
router/gmm_permutations/<scenario>/
```

Each directory contains `router_step0.pt` through `router_step7.pt`,
`routing_results.json`, and the extracted features. This avoids router
checkpoints and logs from different task orders overwriting one another. The
common launcher accepts these environment-variable overrides:

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL=/models/Qwen2.5-Coder-1.5B \
OUTPUT_ROOT=/mnt/experiments/gmm_orders \
bash scripts/codetask/train_gmm_permutation.sh permutation_2
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU visible to `gmm.py` |
| `MODEL` | `Qwen/Qwen2.5-Coder-1.5B` | Hugging Face model ID or local model path |
| `OUTPUT_ROOT` | `./router/gmm_permutations` | Parent directory for scenario outputs |
| `TRAIN_K` | `5000` | Training examples extracted per task |
| `EVAL_K` | `1000` | Test examples evaluated per seen task |
| `BATCH_SIZE` | `16` | Feature-extraction batch size |

Do not launch multiple runs with the same `OUTPUT_ROOT` and scenario name
unless overwriting/resuming that run is intended.
