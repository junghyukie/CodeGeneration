# GMM-Router

Parameter-efficient continual code generation via per-task LoRA adapters and a Gaussian Mixture Model instruction router. Each new task trains an isolated LoRA adapter and fits a GMM on routing embeddings extracted from a frozen backbone; no earlier adapters or backbone weights are modified. At inference, the router scores each task against the query embedding and either selects the top expert (hard routing) or merges adapters weighted by the routing distribution (soft routing).

Two benchmark tracks are supported:

- **CodeTask** — non-executable tasks: CONCODE, CodeTrans, CodeSearchNet, BFP, KodCode, RunBugRun, TheVault\_Csharp, CoST
- **Executable** — nine programming languages: Python, C++, Swift, Rust, C#, Java, PHP, TypeScript, Shell

For the CodeTask benchmark the pipeline uses Steps 1a, 1b, and 4 only. For the Executable benchmark all five steps run, with Steps 2–3 adding calibration-based adapter refinement and a traceback GMM trained on runtime error messages.

---

## Installation

```bash
pip install -r requirements.txt
```

A GPU with at least 24 GB VRAM is required for single-GPU runs; multi-GPU training uses DeepSpeed ZeRO-2 and is configured via `CUDA_VISIBLE_DEVICES`.

---

## Pipeline

The full pipeline is illustrated in the Methodology chapter (Chapter 4). Steps 1a and 1b run in parallel. After Step 1, two branches execute concurrently: the right branch runs Round-1 inference (Step 4) directly; the left branch generates calibration predictions (Step 2), then refines the adapter (Step 3a) and fits the traceback GMM (Step 3b) in parallel. Round-2 inference (Step 5) runs after both branches complete and re-predicts only the samples where all Round-1 candidates failed.

```
Step 1a  Train LoRA adapter  ─┬─────────────────────────────► Step 4  Round-1 inference
Step 1b  Fit instruction GMM ─┘         │
                                        ▼
                               Step 2  Calibration inference   (executable only)
                                 ├─► Step 3a  Refine adapter   (executable only)
                                 └─► Step 3b  Fit traceback GMM (executable only)
                                        │
                                        ▼
                               Step 5  Round-2 inference        (executable only)
```

---

## CodeTask benchmark

### Step 1a — Train LoRA adapters

Trains one LoRA adapter per task sequentially on eight CodeTask datasets. Requires DeepSpeed and at least 6 GPUs.

```bash
bash scripts/codetask/train_anamoe_codetask.sh
```

Adapters are saved to `./output_models/anamoe/<task>/` and uploaded to HuggingFace Hub at `dongg18/anamoe`.

### Step 1b — Fit instruction GMM router

Extracts routing embeddings from the frozen backbone and fits a per-task diagonal GMM. Steps 1a and 1b are independent and can be run in parallel once adapters exist.

```bash
bash scripts/codetask/router_codetask.sh
```

The router checkpoint is saved to `router/router_gmm_codetask_vf0.02_dim_256_comp_4_layer_4/`.

### Step 4 — Round-1 inference

Runs inference on all eight CodeTask tasks using the instruction router. Defaults to hard routing.

```bash
# All eight tasks in one run
bash scripts/codetask/infer_gmm_codetask.sh [soft|hard]

# Or task-by-task (one adapter added per script, in task order)
bash scripts/codetask/infer_gmm_CONCODE.sh
bash scripts/codetask/infer_gmm_CodeTrans.sh
bash scripts/codetask/infer_gmm_CodeSearchNet.sh
bash scripts/codetask/infer_gmm_BFP.sh
bash scripts/codetask/infer_gmm_KodCode.sh
bash scripts/codetask/infer_gmm_RunBugRun.sh
bash scripts/codetask/infer_gmm_TheVault_Csharp.sh
bash scripts/codetask/infer_gmm_CoST.sh
```

Results are written to `./inference_results/gmm_codetask_*/`.

---

## Executable benchmark

### Step 1a — Train LoRA adapters

Trains one LoRA adapter per language on the executable benchmark. Requires 3 GPUs.

```bash
bash scripts/executable/train_anamoe_executable.sh
```

Adapters are saved to `./output_models/lora_per_task_executable_start_4/<language>/` and uploaded to HuggingFace Hub at `ankhanhtran02/lora-per-task-executable-start-4`.

### Step 1b — Fit instruction GMM router

```bash
# Full training run (recommended)
bash scripts/executable/router_executable.sh

# Resume from a checkpoint (e.g. after step 7)
bash scripts/executable/router_executable_step8.sh
```

The router checkpoint is saved to `router/ckpt_executable_dim256_comp4_vf0.001_mean/`.

### Step 4 — Round-1 inference

```bash
# All nine languages
bash scripts/executable/infer_gmm_executable.sh [soft|hard]

# Top-p 0.9 routing variant
bash scripts/executable/infer_gmm_executable_topp0.9.sh [soft|hard]

# Incremental (one language added per step, useful for ablation)
bash scripts/executable/round1/step0.sh   # python only
bash scripts/executable/round1/step1.sh   # python, cpp
# ... up to step8 (all nine languages)
bash scripts/executable/round1/step8.sh
```

Results are written to `./inference_results/gmm_exe_*/`.

### Step 2 — Calibration inference (executable only)

Runs each language adapter on the `calibration_MBPP` split and writes one JSON file per language containing source, predictions, ground-truth, and unit tests for execution-based evaluation.

```bash
bash scripts/executable/infer_calibration_split.sh
```

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-Coder-1.5B` | Base model |
| `ADAPTER_BASE_DIR` | `ankhanhtran02/lora-per-task-executable-start-4` | Adapter repo or local dir |
| `OUTPUT_DIR` | `./calibration_results` | Output directory |
| `CUDA_DEVICES` | `0,1,2,3` | GPU indices |
| `ZERO_STAGE` | `0` | DeepSpeed ZeRO stage |

After this step, run the execution harness (outside this repo) to populate the pass/fail and traceback fields in each JSON file before proceeding to Steps 3a and 3b.

### Step 3a — Refine LoRA adapter (executable only)

Fine-tunes each adapter on repair pairs extracted from failed calibration predictions. Each pair is `(instruction + buggy code + traceback) → ground-truth`.

```bash
bash scripts/executable/refine_adapter.sh
```

| Variable | Default | Description |
|---|---|---|
| `LANG_ID` | `swift,rust,csharp,java,php,typescript,shell` | Comma-separated languages |
| `ADAPTER_PATH` | `ankhanhtran02/lora-per-task-executable-start-4` | Source adapter |
| `RESULTS_DIR` | `ankhanhtran02/executed_calibration_results` | Executed calibration JSON |
| `OUTPUT_BASE_DIR` | `./refined_adapters` | Output directory |

### Step 3b — Fit traceback GMM router (executable only)

Encodes runtime error messages from failed calibration predictions and fits a per-language GMM. Runs in parallel with Step 3a.

```bash
bash scripts/executable/train_gmm_traceback.sh
```

The checkpoint is saved to `router/router_gmm_traceback_ckpt_eval/`.

### Step 5 — Round-2 inference (executable only)

For samples where all Round-1 predictions failed, combines instruction and traceback router scores via a JSD-gated rule and generates a new prediction.

```bash
# All nine languages (full round-2 run)
bash scripts/executable/round2/infer_round2_disagree_explore.sh

# Refined-adapter variant
bash scripts/executable/round2/infer_round2_disagree_explore_refined.sh

# Incremental (one language per step)
bash scripts/executable/round2/infer_round2_step0.sh  # python
# ... up to step7
bash scripts/executable/round2/infer_round2_step7.sh
```

Results are written to `./inference_results/round2_*/`.

---

## Environment variables

Most scripts expose the following variables with sensible defaults:

| Variable | Description |
|---|---|
| `CUDA_VISIBLE_DEVICES` | GPU indices visible to the process |
| `HF_HOME` | HuggingFace cache root (default: `./.cache`) |
| `HF_DATASETS_CACHE` | HuggingFace datasets cache (default: `./.cache`) |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
