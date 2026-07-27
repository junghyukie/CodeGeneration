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
