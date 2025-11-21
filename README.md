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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Salesforce for the CodeT5 model
- Microsoft for the CodeXGLUE benchmark
- Hugging Face for the transformers and PEFT libraries
- The open-source community for continual learning research

**⭐ Star this repo if you find it helpful!**
