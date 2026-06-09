# T5 Continual Learning with LoRA 

A comprehensive implementation of continual learning methods for code-related tasks using T5 models with LoRA (Low-Rank Adaptation) and various catastrophic forgetting mitigation techniques.
##  Overview

This repository implements several continual learning strategies for code generation, translation, and refinement tasks:

- **Full Fine-tuning**: Standard fine-tuning on sequential tasks
- **Full Fine-tuning with EWC**: Elastic Weight Consolidation to prevent catastrophic forgetting
- **LoRA per Task**: Parameter-efficient fine-tuning with separate LoRA adapters
- **O-LoRA (Orthogonal LoRA)**: LoRA with orthogonality constraints to reduce task interference
- **LoRA with EWC**: LoRA with Elastic Weight Consolidation

## Supported Tasks
The framework supports four code-related tasks from CodeXGLUE benchmark:
1. **CodeTrans**: Java to C# code translation
2. **CodeSearchNet**: Ruby code summarization
3. **BFP**: Bug fixing/code refinement
4. **CONCODE**: Natural language to Java code generation

## 🛠️ Installation

### Install Dependencies
pip install -r requirements.txt

##  Quick Start

### 1. Full Fine-tuning with EWC
```bash
python t5_fullfinetune.py \
  --task_list CONCODE CodeTrans CodeSearchNet BFP \
  --log_filepath logs/fullft_ewc.log \
```
### 3. LoRA per Task

```bash
python t5_trainer1.py \
  --task_list CONCODE CodeTrans CodeSearchNet BFP\
  --log_filepath logs/lora_pertask.log \
```

### 4. O-LoRA (Orthogonal LoRA)

```bash
python t5_olora.py \
  --task_list CONCODE CodeTrans CodeSearchNet BFP \
  --log_filepath logs/olora.log \
```

### 5. Continual Learning with EWC

```bash
python t5_continual_ewc.py \
  --task_list CONCODE CodeTrans CodeSearchNet BFP \
  --log_filepath logs/ewc_training.log \
```

## Calibration Set Inference (Executable Benchmark)

After training per-task anamoe adapters, evaluate them on the `calibration_MBPP` split of
[`ankhanhtran02/CL4Code-executable-datasets`](https://huggingface.co/datasets/ankhanhtran02/CL4Code-executable-datasets).
The script runs multi-GPU inference (via DeepSpeed) and writes one JSON file per language containing
the source instruction, model predictions (up to `num_return_sequences` samples for pass@k), the
reference solution, and the unit-test string needed for execution-based evaluation.

### Quick start

```bash
# Train adapters first (produces ./output_models/lora_per_task_executable_start_4/<lang>/0/)
bash scripts/train_anamoe_executable.sh

# Run calibration inference (default: 3 GPUs, ZERO_STAGE=0)
bash scripts/infer_calibration_split.sh
```

### Configuration

| Variable          | Default                                                     | Description                          |
|-------------------|-------------------------------------------------------------|--------------------------------------|
| `MODEL`           | `Qwen/Qwen2.5-Coder-1.5B`                                  | Base model path or HF repo ID        |
| `ADAPTER_BASE_DIR`| `./output_models/lora_per_task_executable_start_4`          | Root dir containing per-language adapters (`<lang>/0/`) or an HF Hub repo ID |
| `OUTPUT_DIR`      | `./calibration_results`                                     | Directory for output JSON files      |
| `CUDA_DEVICES`    | `0,1,2`                                                     | GPU indices exposed to the script    |
| `NUM_GPUS`        | (auto-detected from `CUDA_DEVICES`)                         | Number of GPUs to use                |
| `ZERO_STAGE`      | `0`                                                         | DeepSpeed ZeRO stage (0 = fastest for inference) |

### Output format

Each `calibration_<language>.json` matches the `_save_generation_predictions` format:

```json
{
  "metrics": {},
  "predictions": [
    {
      "source": "Write a function that ...",
      "ground-truth": "def solve(...):\n    ...",
      "prediction": ["def solve(...):\n    ...", "..."],
      "test": "assert solve(...) == ..."
    }
  ]
}
```

`prediction` is a list when `--num_return_sequences > 1` (for pass@k).

### Single-GPU example

```bash
CUDA_DEVICES=0 NUM_GPUS=1 \
  ADAPTER_BASE_DIR=./output_models/lora_per_task_executable_start_4 \
  OUTPUT_DIR=./calibration_results \
  bash scripts/infer_calibration_split.sh
```

### Loading adapters from HuggingFace Hub

```bash
ADAPTER_BASE_DIR=ankhanhtran02/lora-per-task-executable-start-4 \
  bash scripts/infer_calibration_split.sh
```

The script will pass the repo ID to the trainer; the subfolder `<language>/0` is resolved
automatically by `model.load_adapter`.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Salesforce for the CodeT5 model
- Microsoft for the CodeXGLUE benchmark
- Hugging Face for the transformers and PEFT libraries
- The open-source community for continual learning research

**⭐ Star this repo if you find it helpful!**
