# GMM router task-order experiments

Chạy GMM router (`gmm.py`) theo 5 thứ tự task khác nhau.

## Requirements

Chạy từ thư mục gốc của repository, trong môi trường có Python/PyTorch và
Bash (Linux hoặc WSL):

```bash
pip install -r requirements.txt
```

GPU CUDA được khuyến nghị; mặc định dùng GPU `0`.

## Task order

| Scenario | Order |
| --- | --- |
| `original` | A → B → C → D → E → F → G → H |
| `permutation_1` | A → B → H → C → G → D → F → E |
| `permutation_2` | C → D → B → E → A → F → H → G |
| `permutation_3` | E → F → D → G → C → H → B → A |
| `permutation_4` | G → H → F → A → E → B → D → C |

`A=CONCODE`, `B=CodeTrans`, `C=CodeSearchNet`, `D=BFP`, `E=KodCode`,
`F=RunBugRun`, `G=TheVault_Csharp`, `H=CoST`.

## Run

Chạy một scenario:

```bash
bash scripts/codetask/train_gmm_permutation.sh permutation_1
```

Chạy cả baseline và bốn permutation. Script này tự cài dependencies từ
`requirements.txt` trước khi chạy:

```bash
bash scripts/codetask/train_gmm_all_permutations.sh
```

Nếu Python không phải lệnh `python`, chỉ định nó qua `PYTHON_BIN`:

```bash
PYTHON_BIN=python3 bash scripts/codetask/train_gmm_all_permutations.sh
```

Output tách riêng tại `router/gmm_permutations/<scenario>/`, nên các
permutation không ghi đè kết quả của nhau.

Đổi GPU hoặc output directory:

```bash
CUDA_VISIBLE_DEVICES=1 OUTPUT_ROOT=/mnt/gmm_results \
bash scripts/codetask/train_gmm_permutation.sh permutation_1
```

## Router ablation

So sánh centroid, Single Gaussian (`M=1`), GMM (`M=4`), distance-weighted
k-NN (`k=5`) và Oracle task routing:

```bash
bash scripts/codetask/run_router_ablation.sh
```

Script tự cài dependencies, fit/evaluate router và chạy inference cho cả 5
method. Checkpoint nằm trong `router/router_ablation/`, kết quả trong
`inference_results/router_ablation/`, còn log trong `logs/router_ablation/`.
Hai file tổng hợp cuối cùng là `summary.json` và `summary.csv`.

Các tùy chọn thường dùng:

```bash
# Bỏ qua cài dependencies hoặc chỉ đánh giá routing, không generate code
INSTALL_DEPS=0 RUN_INFERENCE=0 bash scripts/codetask/run_router_ablation.sh
```

## PBC-GMM router

`gmm_pbc.py` implements Selective Pairwise Boundary Calibration on top of
calibrated per-task GMM scores. It uses disjoint GMM-fit, boundary-fit, and
boundary-certification splits. Candidate boundaries are certified with a
one-sided stratified paired bootstrap before they can affect low-margin top-2
decisions.

Install the pinned project dependencies in a clean environment first:

```bash
python -m pip install -r requirements.txt
```

Small two-task smoke run:

```bash
python gmm_pbc.py \
  --model_name SalesForce/codet5-small \
  --tasks CONCODE,CodeTrans \
  --train_k 200 \
  --eval_k 100 \
  --batch_size 8 \
  --old_pseudo_fit_samples 1000 \
  --old_pseudo_cert_samples 2000 \
  --bootstrap_replicates 500 \
  --output_dir ./router/pbc_smoke
```

Full CodeTask run with the method defaults (`70% / 15% / 15%`, `B=2000`,
one-sided `alpha=0.05`, and `delta=0.5`):

```bash
CUDA_VISIBLE_DEVICES=0 python gmm_pbc.py \
  --model_name Qwen/Qwen2.5-Coder-1.5B \
  --tasks CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST \
  --train_k 5000 \
  --eval_k 1000 \
  --batch_size 16 \
  --routing_dim 256 \
  --gmm_components 4 \
  --variance_floor 0.02 \
  --old_pseudo_fit_samples 6000 \
  --old_pseudo_cert_samples 20000 \
  --bootstrap_replicates 2000 \
  --bootstrap_alpha 0.05 \
  --margin_threshold 0.5 \
  --output_dir ./router/router_gmm_pbc_codetask
```

Resume from a checkpoint. Keep `projection_P.pt` beside the checkpoint; method
parameters are restored from the checkpoint, while output/cache/evaluation
settings may be changed:

```bash
python gmm_pbc.py \
  --tasks CONCODE,CodeTrans,CodeSearchNet,BFP,KodCode,RunBugRun,TheVault_Csharp,CoST \
  --output_dir ./router/router_gmm_pbc_codetask \
  --resume_from ./router/router_gmm_pbc_codetask/router_step3.pt
```

The output directory contains `config.json`, `representation_manifest.json`,
`projection_P.pt`, `router_stepN.pt`, `boundary_calibration_results.json`, and
`routing_results.json`. The latter reports both the calibrated-GMM baseline
(`b=0`) and PBC-GMM accuracy. Feature caches are namespaced by the model,
tokenizer, projection hash, seed, and data/representation settings; reuse a
shared cache only through `--feature_cache_dir`.
For mutable local model files or an upstream dataset revision that changed
without changing its name, also change `--cache_tag` (for example,
`--cache_tag dataset-2026-09-01`) to force a new namespace. Local model
directories and tokenizer vocabularies are content-hashed automatically.

Run the focused unit tests with:

```bash
python -m pytest -q tests/test_gmm_pbc.py
```
