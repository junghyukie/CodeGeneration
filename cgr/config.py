"""CGRConfig: every hyperparameter of methodology.tex Section 7, saved as config.json."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CODETASK_TASKS = ["CONCODE", "CodeTrans", "CodeSearchNet", "BFP", "KodCode", "RunBugRun", "TheVault_Csharp", "CoST"]
EXECUTABLE_TASKS = ["python", "cpp", "swift", "rust", "csharp", "java", "php", "typescript", "shell"]

# Training-time truncation lengths from the thesis training scripts (task identity is known while training).
_CODETASK_PROMPT_LEN = dict(zip(CODETASK_TASKS, [320, 320, 256, 130, 512, 256, 256, 256]))
_CODETASK_ANS_LEN = dict(zip(CODETASK_TASKS, [150, 256, 128, 120, 300, 128, 128, 128]))


@dataclass
class CGRConfig:
    # run
    benchmark: str = "codetask"  # "codetask" | "executable"
    tasks: List[str] = field(default_factory=lambda: list(CODETASK_TASKS))
    output_dir: str = "runs/cgr"
    method: str = "cgr"                # "cgr" | "seqlora" (baseline) | "base" (pretrained reference)
    seed: int = 1234                   # training randomness: atom init, batch order, dropout, sampling
    data_seed: int = 1234              # train/test subset and held-out split; keep fixed across methods/seeds
    ablation: str = "A3"

    # backbone
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-1.5B"
    backbone_dtype: str = "bfloat16"

    # routing encoder (Sec. 4.1)
    split_layer: int = 4               # l*; z is read from output_hidden_states[l*] (thesis code: feature_layers=4)
    routing_dim: int = 256             # p
    projection_seed: int = 0           # s_P
    routing_max_length: int = 2048     # right truncation
    ln_eps: float = 1e-8
    embed_batch_size: int = 16

    # GMM (Sec. 4.2)
    gmm_components: int = 4
    em_max_iters: int = 50
    em_tol: float = 1e-4
    var_floor: float = 0.02
    gmm_max_samples: int = 5000
    gmm_seed: int = 42

    # router (Sec. 4.3)
    shrinkage: float = 1.0             # lambda
    cov_jitter: float = 1e-4           # eps_Sigma
    affinity_temperature: float = 1.0  # tau_aff
    tau_grid_exponents: Tuple[int, int] = (-2, 8)
    heldout_frac: float = 0.05
    heldout_max: int = 500
    routing_mode: str = "hard"         # "hard" | "soft"
    tau_override: Optional[float] = None  # A9 only (analysis)

    # atoms (Sec. 4.4)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj")

    # composition / ablations
    composition: str = "learned"       # "learned" | "identity" (C = I)
    beta_init: str = "affinity"        # "affinity" | "zero" (A7)
    skip_phase_a: bool = False         # A6
    phase_a_only: bool = False         # A5

    # training (Sec. 4.6)
    batch_size: int = 32
    micro_batch_size: int = 8
    phase_a_steps: int = 300
    phase_a_lr: float = 5e-3
    phase_b_epochs: int = 3
    phase_b_lr: float = 1e-4
    phase_b_weight_decay: float = 0.01
    phase_b_warmup_ratio: float = 0.01
    phase_b_beta_lr: float = 2e-5
    adam_betas: Tuple[float, float] = (0.9, 0.95)  # thesis FusedAdam setting
    grad_clip: float = 1.0                         # thesis DeepSpeed config
    loss_normalization: str = "token"  # "token" (thesis HF loss) | "sequence" (literal spec formula)
    gradient_checkpointing: bool = False
    heldout_log_interval: int = 500
    max_train_examples: int = 100_000
    max_eval_examples: int = 2_000
    max_test_examples: int = 5_000
    max_prompt_len_train: Dict[str, int] = field(default_factory=dict)
    max_ans_len_train: Dict[str, int] = field(default_factory=dict)

    # inference (Sec. 4.7)
    max_prompt_len_infer: int = 2048
    max_new_tokens: int = 512
    do_sample: bool = True
    temperature: float = 0.2
    top_p: float = 0.95
    n_samples: int = 1
    infer_batch_size: int = 32

    def __post_init__(self):
        self.tasks = list(self.tasks)
        self.target_modules = tuple(self.target_modules)
        self.tau_grid_exponents = tuple(self.tau_grid_exponents)
        self.adam_betas = tuple(self.adam_betas)

    @property
    def tau_grid(self) -> List[float]:
        lo, hi = self.tau_grid_exponents
        return [2.0 ** e for e in range(lo, hi + 1)]

    def prompt_len(self, task: str) -> int:
        return int(self.max_prompt_len_train.get(task, 2048))

    def ans_len(self, task: str) -> int:
        return int(self.max_ans_len_train.get(task, 2048))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CGRConfig":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


def benchmark_defaults(benchmark: str) -> CGRConfig:
    """Benchmark presets from Section 7 (decoding, routing mode, task order)."""
    if benchmark == "codetask":
        return CGRConfig(
            benchmark="codetask", tasks=list(CODETASK_TASKS), routing_mode="hard",
            max_new_tokens=512, n_samples=1,
            max_prompt_len_train=dict(_CODETASK_PROMPT_LEN), max_ans_len_train=dict(_CODETASK_ANS_LEN),
        )
    if benchmark == "executable":
        return CGRConfig(
            benchmark="executable", tasks=list(EXECUTABLE_TASKS), routing_mode="soft",
            max_new_tokens=2048, n_samples=5, micro_batch_size=1, infer_batch_size=8,
            max_prompt_len_train={t: 2048 for t in EXECUTABLE_TASKS},
            max_ans_len_train={t: 2048 for t in EXECUTABLE_TASKS},
        )
    raise ValueError(f"unknown benchmark {benchmark}")


def apply_ablation(cfg: CGRConfig, ablation: str, shrinkage: Optional[float] = None) -> CGRConfig:
    """Ablation grid of Section 8.1. A8/A9 are analysis-only (see cgr/eval/oracle.py, tau_override)."""
    a = ablation.upper()
    cfg.ablation = a
    presets = {
        "A0": dict(composition="identity", shrinkage=0.0),
        "A1": dict(composition="identity", shrinkage=1.0),
        "A2": dict(composition="learned", shrinkage=0.0),
        "A3": dict(composition="learned", shrinkage=1.0),
        "A4": dict(composition="learned"),
        "A5": dict(composition="learned", shrinkage=1.0, phase_a_only=True),
        "A6": dict(composition="learned", shrinkage=1.0, skip_phase_a=True),
        "A7": dict(composition="learned", shrinkage=1.0, beta_init="zero"),
        "A8": dict(composition="learned", shrinkage=1.0),
        "A9": dict(composition="learned", shrinkage=1.0),
    }
    if a not in presets:
        raise ValueError(f"unknown ablation {ablation}")
    if a in ("A8", "A9"):
        print(f"[warn] {a} uses test-set information: analysis only, never a method (methodology Sec. 8.1)")
    for k, v in presets[a].items():
        setattr(cfg, k, v)
    if shrinkage is not None:
        cfg.shrinkage = float(shrinkage)
    return cfg
