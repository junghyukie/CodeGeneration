# executable_dataset_2

This folder contains a local-first evaluator for running solution snippets against inline tests across multiple languages.

## Supported Languages

- `cpp`
- `python`
- `swift`
- `rust`
- `csharp`
- `java`
- `php`
- `typescript`
- `shell`

## Docker Setup

Docker is only needed for languages whose toolchains are not installed locally.

### 1. Build the image

Open Docker and run from the repository root:

```bash
docker build -t cl4code-executable-dataset-2:latest -f executable_dataset_2/Dockerfile executable_dataset_2
```

## Optional Environment Variables

You can override the default image and container names:

```bash
export EXECUTABLE_DATASET_2_DOCKER_IMAGE=cl4code-executable-dataset-2:latest
export EXECUTABLE_DATASET_2_DOCKER_CONTAINER=cl4code-executable-dataset-2-runner
```

## Dependency Installation
```bash
conda create -n codecl python=3.12
conda activate codecl
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Fine-Tuning

For the current `executable_dataset_2/train.py`, this script fine-tunes
`Qwen/Qwen2.5-Coder-1.5B` on Python using the default training splits
`train_OSS_Instruct` and `train_McEval_Instruct`, then evaluates on `test_McEval`.

Example:

```bash
mkdir -p executable_dataset_2/output/qwen2_5_coder_1_5b_lora_python

python3 executable_dataset_2/train.py \
  --language python \
  --output-dir executable_dataset_2/output/qwen2_5_coder_1_5b_lora_python \
  --results-json executable_dataset_2/output/qwen2_5_coder_1_5b_lora_python/eval_results.json \
  --run-name qwen2_5_coder_1_5b_python \
  --epochs 3 \
  --per-device-train-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.01 \
  --weight-decay 0.01 \
  --max-new-token 1024 \
  --temperature 0.2 \
  --top-p 0.95 \
  --repetition-penalty 1.2 \
  --early-stopping \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1 \
  --logging-steps 10 \
  --max-train-samples 5000 \
  --fp16 \
  2>&1 | tee executable_dataset_2/output/qwen2_5_coder_1_5b_lora_python/train.log
```

## Inference on Base Model
```bash
python3 executable_dataset_2/infer.py \
  --batch-size 8 \
  --max-new-tokens 1024
```

## Build the `calibration_MBPP` Split

`executable_dataset_2/build_calibration_mbpp.py` builds the `calibration_MBPP`
split for all 9 languages and pushes it to the
`ankhanhtran02/CL4Code-executable-datasets` dataset on the Hugging Face Hub. It
follows the repo schema (`index`, `language`, `instruction`, `solution`,
`test`) and the lowercase `language` convention.

Sources:

- **python** — `Muennighoff/mbpp`, config `sanitized`. `solution` is the
  reference `code`; the function signature (the last `def` line of the code) is
  spliced into the instruction inside backticks; the `test` is a
  `def check(<fn>): ... check(<fn>)` harness built from `test_imports` +
  `test_list`, matching the `test_McEval` Python format.
- **other languages** — `nuprl/MultiPL-E`, configs `mbpp-<ext>` (e.g.
  `mbpp-cpp`, `mbpp-rs`). `solution` is `null`; the documentation comment and
  function signature are extracted from the prompt; the `test` is the MultiPL-E
  `tests` with the single leading `}` removed so the harness can follow a
  complete solution.

> Note: Python MBPP problems whose function is itself named `check` are skipped,
> because that name collides with the `check(<fn>)` test harness and would
> produce a non-runnable test. (In the current `sanitized` set this is task 56,
> so the Python subset holds 426 of 427 problems.)

The script loads the script-based source datasets from their datasets-server
auto-converted parquet files, so it works even with recent `datasets` releases
that dropped loading-script support.

Example (set `HF_TOKEN` in your environment before uploading):

```bash
# Build the split, write executable_dataset_2/output/calibration_MBPP.jsonl, and
# upload only the calibration_MBPP split to the Hub.
python3 executable_dataset_2/build_calibration_mbpp.py \
  --repo-id ankhanhtran02/CL4Code-executable-datasets

# Build and inspect the JSONL locally without touching the Hub.
python3 executable_dataset_2/build_calibration_mbpp.py --dry-run
```

## Preprocess

`executable_dataset_2/preprocess.py` is the CLI entrypoint for exact deduplication and near deduplication.

### 1. Run Exact Deduplication

Use exact mode to find instructions that are identical after exact normalization and SHA-256 hashing.

Example:

```bash
python3 executable_dataset_2/preprocess.py \
  --mode exact \
  --datasets OSS-Instruct McEval-Instruct McEval \
  --duplicates-output executable_dataset_2/output/exact_duplicates.jsonl
```


### 2. Run Near Deduplication

Use near mode to compare a training set against a test set with normalization, shingling, MinHash, LSH, and exact Jaccard scoring.

Example:

```bash
python3 executable_dataset_2/preprocess.py \
  --mode near \
  --train-datasets OSS-Instruct McEval-Instruct \
  --test-datasets McEval \
  --threshold 0.8 \
  --shingle-size 5 \
  --num-perm 128 \
  --lsh-threshold 0.75 \
  --near-duplicates-output executable_dataset_2/output/near_duplicates.jsonl
```

## Near-Dedup Normalization Pipeline

The function `normalize_instruction_text_for_near_dedup()` in `preprocess.py` applies the following steps.

### 1. Unicode normalization

Convert visually equivalent Unicode text into a consistent form.

Input:

```text
Use ｐｙｔｈｏｎ and café
```

Output:

```text
Use python and café
```

### 2. Remove leading and trailing whitespace

Trim whitespace at the beginning and end of the full instruction.

Input:

```text

  solve the task

```

Output:

```text
solve the task
```

### 3. Lowercase

Lowercase the entire instruction before later normalization steps.

Input:

```text
Implement The NeuralNetwork Class
```

Output:

```text
implement the neuralnetwork class
```

### 4. Separate prose and code

Split fenced code blocks from surrounding natural-language text so prose and code can be normalized differently.

Input:

````text
implement `add_layer`

```python
def add_layer(x):
    pass
```
````

Output:

```text
prose: implement `add_layer`
code: def add_layer(x):
    pass
```

### 5. Normalize code fences and replace placeholder comments

Represent code blocks in a stable way and convert placeholder comments such as `# Your code here` or `TODO` into `MISSING_CODE`.

Input:

````text
```Python
def add_layer(x):
    # Your code here
```
````

Output:

```text
code python
def add_layer(x):
MISSING_CODE
```

### 6. Normalize identifiers

Abstract class, function, and argument names into placeholder tokens such as `CLASS_1`, `FUNC_1`, and `ARG_1`.

Input:

````text
```python
class neuralnetwork:
    def add_layer(self, units):
        return units
```
````

Output:

```text
code python
class CLASS_1
def FUNC_1(self, ARG_1)
return ARG_1
```

### 7. Collapse repeated whitespace and blank lines

Convert repeated spaces and multiple consecutive line breaks into a compact form.

Input:

```text
implement    the method


with details
```

Output:

```text
implement the method
with details
```

### 8. Normalize imports

Turn import statements into a compact import summary.

Input:

````text
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
```
````

Output:

```text
imports tensorflow tf tensorflow keras models sequential tensorflow keras layers dense activation
```

### 9. Drop low-value syntax punctuation

Remove `(`, `)`, `:`, `.`, `,`, and `=` from non-import code to reduce superficial variation.

Input:

```text
def add_layer(units=32):
    model.add(units, x=1)
```

Output:

```text
def add_layer units 32
model add units x 1
```

## Final End-to-End Example

Input:

````text
  Your task is to complete the `NeuralNetwork` class by implementing `add_layer`.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential

class NeuralNetwork:
    def add_layer(self, units):
        # Your code here
```

````

Output:

```text
your task is to complete the neuralnetwork class by implementing add layer
code python
imports tensorflow tf tensorflow keras models sequential
class CLASS_1
def FUNC_1 self ARG_1
MISSING_CODE
```
