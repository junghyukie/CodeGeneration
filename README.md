# Compositional GMM-Router (CGR)

Continual code generation with per-task LoRA experts, input-only routing, and learned cross-task transfer.
The full specification is [methodology.tex](methodology.tex). The implementation is the [`cgr/`](cgr/) package.

**Method in one paragraph.** A frozen backbone (Qwen2.5-Coder-1.5B) learns tasks one after another. For each new task:

1. The router measures how similar the task's prompts are to earlier tasks. It uses one diagonal GMM per task, fitted on frozen layer-4 features, and scores all tasks under one shared covariance.
2. That similarity sets the starting values of one lateral weight β per earlier LoRA and per block.
3. **Phase A** trains only the β weights, so the new expert is first built from the frozen earlier LoRAs.
4. **Phase B** trains a fresh LoRA "atom" (and β) for whatever the earlier LoRAs cannot express.
5. Everything that belongs to the task is then frozen.

At inference, the routing weights come from the prompt alone. They pass through the learned weights, c = Cᵀα, and every LoRA is merged into one adapter for a single generation round. Nothing in the method uses task labels at inference, data from earlier tasks, a compiler, or test cases.

CGR is the thesis GMM-Router with three changes:
- **Removed:** the thesis's leaky refinement pipeline (calibration inference, traceback router, adapter refinement, Round-2).
- **Added: a shared-covariance router.** All tasks are scored under one shared covariance (shrinkage λ), with a calibrated temperature τ.
- **Added: the composition matrix C,** the learned lateral weights that let a new task reuse earlier LoRAs.

---

## 1. Setup

```bash
conda create -n cgr python=3.11 -y && conda activate cgr
# Install a CUDA build of PyTorch that matches your driver first, e.g.:
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt          # transformers 4.47.1, datasets, scipy, safetensors, tree-sitter (CodeBLEU), ...
export HF_HOME=./.cache HF_DATASETS_CACHE=./.cache
# huggingface-cli login                  # only if your HF account needs it for the datasets
```

Check the install. The acceptance tests (T1–T10 of methodology.tex Sec. 10, plus an end-to-end smoke test) run on CPU with a tiny random model in about 10 s:

```bash
python -m unittest cgr.tests.test_acceptance -v
```

**Hardware.** Each run uses **one GPU**. Training a single run across several GPUs is not implemented. On a multi-GPU node, the fast experiment runs 7 independent runs in parallel, one per GPU. On CodeTask, one run is expected to peak well below 46 GB:
- **Training:** micro-batch 8 with gradient accumulation to batch 32. The longest task, KodCode, is 512+300 tokens per example.
- **Generation:** batch 32.

The `peak_gb` field at the end of Phase A and Phase B in each run's log reports the real peak. If a run goes out of memory, add `--override micro_batch_size=4` or `gradient_checkpointing=true`.

---

## 2. Experiment 1 — fast CodeTask check (forgetting and transfer)

**Question.** On the 8-task CodeTask sequence, does CGR
- **(H1)** keep forgetting near zero, compared with a shared LoRA trained sequentially on the same data and budget?
- **(H2)** gain from knowledge transfer, i.e. score higher on each new task than the same system without composition (C = I)?

### Design

All runs use the same data subsets (`data_seed=1234`), the same backbone, the same LoRA recipe and the same decoding. Only the method differs.

| Run (dir) | GPU | What it is | Used for |
|---|---|---|---|
| `A3_s1234`, `A3_s42` | 0, 3 | Full CGR: learned C, λ=1, Phase A + Phase B; also scores each task right after Phase A | treatment for H1, H2 |
| `A1_s1234`, `A1_s42` | 1, 4 | Ablation A1: C = I (no composition, no Phase A), λ=1; otherwise identical to A3 | control for H2 (Δ_j) |
| `SEQ_s1234`, `SEQ_s42` | 2, 6 | Baseline SeqLoRA: one LoRA (same placement and rank) retrained on every task with the Phase-B recipe; no routing | control for H1 (forgetting) |
| `A0_s1234` | 5 | Ablation A0: C = I, λ=0, the thesis GMM-Router Round-1 | sanity check; effect of the shared covariance on routing |
| `BASE` | 6 (first) | Pretrained backbone, no adapters, no training | FWT reference ã_j |

Seeds 1234 and 42 change only the training randomness: atom initialisation, batch order, dropout and sampling.

### Data size

Per task, **5,000 training examples** are drawn at random, about 5–50% of each task's full training set. 5% of them (250) become the held-out slice H_k, used for τ calibration and loss logging. Each task also uses **500 test examples**; CoST has only 410.

This size is chosen so that each LoRA still gets about 450 optimizer steps (3 epochs × 4,750 / 32), enough to learn each task's output format and content. It is small enough that a full 8-task run with complete per-step evaluation should finish in a few hours on one GPU. The small-data regime is also where transfer from earlier tasks matters most. To scale up, change `MAX_TRAIN` / `MAX_TEST`; the thesis used up to 100k / 5k.

Task order (thesis): CONCODE → CodeTrans → CodeSearchNet → BFP → KodCode → RunBugRun → TheVault_Csharp → CoST.

### Parameters (all runs; values from methodology.tex Sec. 7 unless noted)

| Group | Value |
|---|---|
| Backbone | Qwen2.5-Coder-1.5B, bf16, frozen |
| Routing features | layer ℓ* = 4 (`hidden_states[4]`), masked mean pool, parameter-free LN (ε=1e-8), fixed orthonormal projection to p = 256 (seed 0), right truncation at 2048 tokens, prompt template `input: {x}\noutput: ` |
| GMM per task | M = 4 diagonal components, EM ≤ 50 iterations, tol 1e-4, variance floor 0.02, ≤ 5,000 embeddings |
| Router | shrinkage λ = 1 (A3, A1, SEQ, BASE) or 0 (A0), jitter ε_Σ = 1e-4, affinity temperature 1, τ calibrated on H_k over {2⁻²,…,2⁸}; **hard routing** on CodeTask |
| LoRA atoms | rank 16, alpha 32 (κ = 2), dropout 0.1, `q_proj` + `v_proj` of blocks 4–27, fp32 |
| Phase A (A3 only, tasks 2–8) | β only, 300 steps, batch 32, AdamW lr 5e-3, no weight decay, β initialised from router affinity |
| Phase B (all trained runs) | atom: AdamW lr 1e-4, wd 0.01, 1% linear warmup then constant, 3 epochs, batch 32 (micro-batch 8 × 4); β (A3): lr 2e-5, no wd; AdamW betas (0.9, 0.95), grad clip 1.0 (thesis) |
| Training truncation (thesis, per task) | prompt 320/320/256/130/512/256/256/256, answer 150/256/128/120/300/128/128/128 tokens |
| Decoding | 1 sample, temperature 0.2, top-p 0.95, top-k off, ≤ 512 new tokens, prompt ≤ 2048 tokens, batch 32 |

### Metrics reported (`runs/fast_codetask/report.md`)

- **Per-task score a_{i,j}**: the score on task j after training tasks 1..i. It is CodeBLEU for code-output tasks and smoothed BLEU for CodeSearchNet and TheVault_Csharp, using the thesis harness `evaluator/compute_metrics.py`, ×100. After every task, all seen tasks are evaluated, giving the full lower-triangular matrix.
- **Avg final** = mean_j a_{K,j}, and **Avg a_jj** = mean score of each task right after it is learned.
- **BWT** = mean_{j<K} (a_{K,j} − a_{j,j}). Negative means forgetting.
- **FM** = mean_{j<K} max_{i<K} (a_{i,j} − a_{K,j}). Lower is better.
- **FWT** = mean_{j≥2} (a_{j,j} − ã_j), where ã_j is the pretrained backbone's score (`BASE`).
- **Composition gain Δ_j** = a_{j,j}(A3) − a_{j,j}(A1), paired by seed. This is the transfer that comes from C alone.
- **Phase-A-only score** (A3): task j's score right after Phase A, before its own atom is trained. It measures pure reuse of earlier tasks; compare it with `BASE`.
- **Routing diagnostics**:
  - Hard-routing accuracy and the row-normalised confusion matrix after every task. Drops in a task's routing accuracy show routing drift.
  - Mean entropy of α.
  - The learned β averaged over blocks (`<run>/logs/beta_mean_after_k.json`).
  - The calibrated τ (`<run>/logs/tau_after_k.json`).

### How to read the result

- **H1 holds** if A3's BWT is close to 0 and its FM is small, while SeqLoRA's BWT is clearly negative and its FM is large. Both are compared over the same two seeds. In CGR, forgetting can only come from routing drift, so a negative BWT should line up with a drop in routing accuracy for the affected task.
- **H2 holds** if the mean Δ_j over tasks 2–8 is positive for both seeds, and A3's FWT is above A1's.
  - The Phase-A-only scores show which tasks the earlier atoms already partly solve.
  - The β matrices show which earlier tasks were reused.
  - Differences smaller than the seed-to-seed spread (shown as ±) should not be claimed. This fast experiment has 2 seeds; methodology.tex asks for ≥ 3 seeds for A0–A3 in the final study.

### Run it

```bash
bash scripts/cgr/fast_codetask.sh                      # GPUs 0..6; override with GPUS="..." OUT=... MAX_TRAIN=... MAX_TEST=...
tail -f runs/fast_codetask/logs/A3_s1234.log           # JSON log lines: phase, step, train_loss, heldout_loss, tau, peak_gb
cat runs/fast_codetask/A3_s1234/timing.json            # train/eval seconds per task (use task 1 to project the total)
```

The script does the following:
1. Prefetches the model and datasets.
2. Starts the 7 GPU jobs. GPU 6 runs `BASE` first, then `SEQ_s42`.
3. Waits for all of them.
4. Scores everything and writes the report.

Scoring alone, any time (including on partial results), runs on CPU only:

```bash
bash scripts/cgr/report_fast_codetask.sh runs/fast_codetask    # -> runs/fast_codetask/report.md
```

A stopped CGR run resumes from its last finished task. Run the same command with `--resume` added; existing prediction files are skipped. SeqLoRA and BASE runs restart from scratch.

**Runtime.** These are rough estimates, not measurements. A3 does 300 Phase-A steps plus about 450 Phase-B steps per task, plus 36 test-set generations of up to 500 prompts, which is expected to take about 3–5 h. A1 and SEQ have no Phase A and should take about 2–4 h; BASE takes under 1 h. Check `timing.json` after the first task.

---

## 3. Other runs

All options live in [cgr/config.py](cgr/config.py). Any field can be overridden with `--override key=value` (values parsed as JSON).

```bash
# One run, any ablation of methodology.tex Sec. 8.1 (A0..A7; A8 = oracle routing and A9 = fixed tau are analysis only)
python -m cgr.run_sequence --benchmark codetask --ablation A3 --seed 1234 --output_dir runs/A3
python -m cgr.run_sequence --benchmark codetask --ablation A4 --shrinkage 0.5 --output_dir runs/A4_l0.5
python -m cgr.run_sequence --benchmark codetask --ablation A3 --override routing_mode=soft --output_dir runs/A3_soft

# Full-size CodeTask (thesis sizes: up to 100k train / 5k test per task)
python -m cgr.run_sequence --benchmark codetask --ablation A3 --output_dir runs/A3_full

# Executable benchmark (9 languages; soft routing, n = 5 samples, <= 2048 new tokens, micro-batch 1)
python -m cgr.run_sequence --benchmark executable --ablation A3 --output_dir runs/exe_A3
python -m cgr.eval.score --run_dir runs/exe_A3 --metric pass_at_1     # runs the execution harness (Docker or local)

# Cheaper evaluation: only the new task after each step, all tasks after the last (BWT ok, FM approximate)
python -m cgr.run_sequence ... --eval_mode diag_last
```

| Ablation | Composition | λ | Purpose |
|---|---|---|---|
| A0 | C = I | 0 | thesis GMM-Router Round-1 |
| A1 | C = I | 1 | routing only |
| A2 | learned | 0 | transfer only |
| A3 | learned | 1 | full CGR |
| A4 | learned | `--shrinkage` | shrinkage sweep |
| A5 | Phase A only | 1 | pure reuse |
| A6 | no Phase A | 1 | value of compose-first training |
| A7 | β initialised at 0 | 1 | value of the router prior |
| A8 | oracle α = e_true (written to `predictions/oracle_step_*`) | 1 | upper bound, analysis only |
| A9 | `--override tau_override=<τ>` | 1 | temperature sensitivity, analysis only |

### Output layout of a run

```
runs/<name>/
  config.json, progress.json, timing.json
  state/                         # CGR checkpoint (methodology Sec. 9): projection_P.npy, router/gmm_k.npz,
                                 #   router/scatter.npz, router/tau.json, atoms/atom_k.safetensors,
                                 #   composition/beta_k.npy, manifest.json (sha256 of every frozen artifact)
  logs/learn_task_k.jsonl        # affinity, EM, tau_k, Phase A/B losses, held-out losses, peak GPU memory
  logs/beta_mean_after_k.json, logs/tau_after_k.json
  predictions/step_i/results-<i-1>-<task>.json   # source, ground-truth, prediction(s), alpha, k_hat
  predictions/phase_a/, predictions/base/        # Phase-A-only and pretrained-reference generations
  eval/results.json, eval/scores_cache.json      # written by cgr.eval.score only
```

### Code map

| File | Role (methodology section) |
|---|---|
| `cgr/features.py` | routing encoder z(x), projection P (4.1) |
| `cgr/gmm.py` | diagonal EM and within-task scatter (4.2, Alg. 1) |
| `cgr/router.py` | shared-covariance scores, routing weights, affinity, τ calibration (4.3, 4.5) |
| `cgr/atoms.py` | atom bank: per-example coefficients without weight merging (4.4) |
| `cgr/composition.py` | lower-triangular C per block, β initialisation (4.4, 4.5) |
| `cgr/train.py` | LearnTask: Phase A, Phase B, freezing (4.6, Alg. 2) |
| `cgr/infer.py` | `Predictor.predict(prompts, n_samples)`, the only inference entry point (4.7, Alg. 3) |
| `cgr/data.py` | task registry with the no-replay guard; thesis data sources and prompt template |
| `cgr/state.py` | persistent state, checkpoint, sha256 manifest |
| `cgr/run_sequence.py` | RunSequence (Alg. 4), SeqLoRA baseline, pretrained reference |
| `cgr/eval/` | generation for evaluation, BLEU/CodeBLEU and Pass@k scoring, CL metrics, report |
| `cgr/tests/` | acceptance tests T1–T10 and an end-to-end smoke test |

---

## 4. Choices where the implementation goes beyond methodology.tex

These are logged here as the spec requires for changed defaults and for values tagged *[assumed]*.

- **Training loss is the mean over target tokens** of the effective batch. This is the thesis's Hugging Face loss and is needed for A0 to reproduce thesis training. The spec's literal per-sequence form is available as `--override loss_normalization=sequence`.
- **Learning-rate schedule:** 1% linear warmup for the atom, then a constant rate, as in the thesis code's constant schedule. AdamW betas (0.9, 0.95) and gradient clip 1.0 also come from the thesis DeepSpeed setup.
- **Routing input:** the routing encoder uses the generation template `input: … output: ` with right truncation at 2048 tokens. The old `gmm.py` routed the raw instruction, truncated at 512 or 1024 tokens.
- **CodeTask decoding** follows the spec and thesis text: T=0.2, top-p 0.95, 512 tokens. The thesis inference scripts actually decoded greedily with per-task output lengths. CGR uses one output limit for all queries, so that the length budget does not reveal the task. Executable decoding (T=0.2, top-p 0.95) matches the thesis scripts.
- **ℓ\* convention:** `hidden_states[4]`, the output of blocks 0–3, as in the thesis code (`feature_layers=4`).
- **FWT reference ã_j:** the pretrained backbone's score, following the thesis's definition of a_{0,j}.
- **`predict` is a method, `Predictor(state).predict(prompts, n_samples)`,** so its only parameters are the prompts and n, as T8 requires.
- **SeqLoRA (`--method seqlora`) and BASE (`--method base`) are reference runs added for Experiment 1.** They are not part of CGR.
- **Not implemented:**
  - Training one run across several GPUs (DDP).
  - Bootstrap confidence intervals for the executable benchmark.
  - The optional variant that freezes Σ̄ after the first three tasks.

---

## 5. Legacy thesis pipeline (not used by CGR)

The thesis GMM-Router code is kept for reference: `gmm.py`, `infer_gmm.py`, `training/main_anamoe.py`, and `scripts/codetask/`, `scripts/executable/`.

Its executable-benchmark Steps 2, 3a, 3b and 5 pass ground-truth task identity to later stages through the compiler and test results. Those steps are calibration inference, adapter refinement, the traceback GMM (`gmm_traceback.py`, `training/refine_adapter.py`) and Round-2 with JSD gating. They are therefore not a leak-free baseline.

None of this code is reachable from `cgr/`; test T8 checks the imports statically. CGR's ablation A0 is the leak-free counterpart of thesis Round-1. The execution harness in `executable_dataset/` is reused only by `cgr/eval/score.py`.

## License

MIT, see [LICENSE](LICENSE).
