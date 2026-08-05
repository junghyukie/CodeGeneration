"""
GMM-router inference pipeline.

Architecture (matches gmm_methodology.tex §2–6 and the training setup in train1.sh):
- Layers 0–3 of the generation LLM act as the frozen routing feature extractor.
  LoRA adapters are disabled for this stage (matches gmm.py which uses a fresh
  copy of the same model for feature extraction, with no LoRA).
- The layer-4 hidden state is mean-pooled (full sequence), LN-normalised, and
  projected to the GMM routing space via the saved projection matrix P.
- GMM scores s_k(z) are computed and converted to soft routing weights α_k via
  temperature-scaled softmax (hard routing = τ → 0, i.e. argmax).
- Layers 4+ run with the selected/merged LoRA adapter for generation.

This avoids loading the backbone twice by reusing the PEFT generation model for
routing (adapters disabled via model.disable_adapter()).
"""

import os
import argparse
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import json
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

from huggingface_hub import hf_hub_download

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import PeftModel, PeftConfig

# Older PEFT versions lack a local-file fallback in _get_peft_type; patch it in.
_orig_get_peft_type = PeftConfig._get_peft_type.__func__

@classmethod  # type: ignore[misc]
def _patched_get_peft_type(cls, model_id, subfolder=None, **kwargs):
    config_path = os.path.join(model_id, subfolder or "", "adapter_config.json")
    if os.path.isfile(config_path):
        return cls.from_json_file(config_path)["peft_type"]
    return _orig_get_peft_type(cls, model_id, subfolder=subfolder, **kwargs)

PeftConfig._get_peft_type = _patched_get_peft_type

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_codetask_dataset, create_executable_dataset
from utils.utils import to_device, set_random_seed, load_hf_tokenizer
AllDatasetName = [
    "CONCODE", "CodeTrans", "CodeSearchNet", "BFP",
    "KodCode", "RunBugRun", "TheVault_Csharp", "CoST",
]
AllDatasetNameExecutable = [
    'python', 'cpp', 'swift', 'rust', 'csharp',
    'java', 'php', 'typescript', 'shell',
]
from evaluator.compute_metrics import compute_metrics, DATASET_TO_OUTPUT_LANG
from router_baselines import BaselineRouter


# ── GMM router classes ────────────────────────────────────────────────────────
# Self-contained subset of gmm.py needed for inference (no dataset loading).

@dataclass
class RouterConfig:
    model_name: str = "Salesforce/codet5-small"   # overridden by saved config
    output_dir: str = "./router_gmm_ckpt"
    dataset_source: str = "t5"
    executable_dataset_name: str = "ankhanhtran02/CL4Code-executable-datasets"
    tasks: Tuple[str, ...] = (
        "CONCODE", "CodeTrans", "CodeSearchNet", "BFP",
        "KodCode", "RunBugRun", "TheVault_Csharp",
    )
    feature_layers: int = 4    # hidden-state index used for routing (after decoder layer 3)
    routing_dim: int = 256
    max_length: int = 512      # tokenisation max-length for routing
    batch_size: int = 16
    train_k: int = 2000
    eval_k: int = 1000
    seed: int = 42
    gmm_components: int = 4
    em_iters: int = 50
    em_tol: float = 1e-4
    variance_floor: float = 1e-4
    eps: float = 1e-8
    eval_split: str = "test"
    save_features: bool = True
    force_recompute_features: bool = False


@dataclass
class DiagonalGMMState:
    pi: torch.Tensor
    mu: torch.Tensor
    var: torch.Tensor


class WeightedDiagonalGMM:
    def __init__(self, n_components: int, variance_floor: float = 1e-4, eps: float = 1e-8):
        self.n_components = n_components
        self.variance_floor = variance_floor
        self.eps = eps
        self.state: Optional[DiagonalGMMState] = None

    @staticmethod
    def _log_diag_gaussian(z: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        z_exp = z[:, None, :]
        mu_exp = mu[None, :, :]
        var_exp = var[None, :, :]
        log_det = torch.log(var_exp).sum(dim=-1)
        quad = ((z_exp - mu_exp) ** 2 / var_exp).sum(dim=-1)
        p = z.shape[-1]
        return -0.5 * (p * math.log(2.0 * math.pi) + log_det + quad)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("GMM is not fitted")
        z = z.float().cpu()
        log_prob_comp = self._log_diag_gaussian(z, self.state.mu, self.state.var)
        log_joint = torch.log(self.state.pi.clamp_min(self.eps))[None, :] + log_prob_comp
        return torch.logsumexp(log_joint, dim=1)

    @classmethod
    def from_dict(cls, d: Dict[str, torch.Tensor]) -> "WeightedDiagonalGMM":
        obj = cls(
            n_components=int(d["n_components"]),
            variance_floor=float(d["variance_floor"]),
            eps=float(d["eps"]),
        )
        obj.state = DiagonalGMMState(
            pi=d["pi"].float(), mu=d["mu"].float(), var=d["var"].float()
        )
        return obj


@dataclass
class TaskRouter:
    task_name: str
    task_id: int
    gmm: WeightedDiagonalGMM


class ResidualFitGMMRouter:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self.tasks: List[TaskRouter] = []

    def predict_scores(self, z: torch.Tensor) -> torch.Tensor:
        """MAP log-posterior scores s_k(z) = log p_k(z) + log(1/K). Returns [N, K]."""
        K = len(self.tasks)
        log_prior = math.log(1.0 / K)
        scores = []
        for tr in self.tasks:
            logp = tr.gmm.log_prob(z)
            scores.append(logp + log_prior)
        return torch.stack(scores, dim=1)

    @classmethod
    def load(cls, path: str) -> "ResidualFitGMMRouter":
        payload = torch.load(path, map_location="cpu")
        known = {f.name for f in RouterConfig.__dataclass_fields__.values()}
        cfg = RouterConfig(**{k: v for k, v in payload["cfg"].items() if k in known})
        router = cls(cfg)
        for item in payload["tasks"]:
            router.tasks.append(TaskRouter(
                task_name=item["task_name"],
                task_id=int(item["task_id"]),
                gmm=WeightedDiagonalGMM.from_dict(item["gmm"]),
            ))
        return router


# ── Routing feature extraction helpers ───────────────────────────────────────

@contextmanager
def _disable_lora(model):
    """Context manager that disables all LoRA adapters during routing feature extraction.

    This matches gmm.py's training behaviour where a fresh copy of the backbone
    (no LoRA) was used for feature extraction — ensuring train/inference consistency.
    """
    # model is a PeftModel; model.base_model is a LoraModel (PEFT's BaseTuner subclass)
    # which owns disable_adapter_layers / enable_adapter_layers.
    #
    # We must NOT call model.disable_adapters() — that resolves to HuggingFace
    # Transformers' PeftAdapterMixin.disable_adapters() which requires adapters
    # loaded via HF's own load_adapter(), not via PeftModel.from_pretrained().
    if hasattr(model.base_model, "disable_adapter_layers"):
        model.base_model.disable_adapter_layers()
        try:
            yield
        finally:
            model.base_model.enable_adapter_layers()
    else:
        raise RuntimeError("Cannot find disable_adapter_layers on model.base_model")


def _masked_mean_pool(H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(H.dtype)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (H * m).sum(dim=1) / denom


@torch.no_grad()
def extract_routing_embedding(
    model,
    tokenizer,
    text: str,
    feature_layers: int,
    projection: torch.Tensor,   # [p, 2*hidden_size]
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute the routing embedding z ∈ ℝ^p for a single input string.

    Replicates the gmm.py feature pipeline (§2 of gmm_methodology.tex):
      tokenise (no left-padding) → frozen backbone → layer feature_layers hidden state
      → mean-pool (full sequence) → LN → projection P → z

    The model's LoRA adapters are disabled for this call so the routing features
    come from the BASE model, matching the training setup.
    """
    # Tokenise with right-padding (or no padding for a single sample) to match
    # the training feature extraction, which did NOT use left-padding.
    enc = tokenizer(
        [text],
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with _disable_lora(model):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    # hidden_states[0] = embedding output; hidden_states[k] = after decoder layer k-1.
    # feature_layers=4 → hidden_states[4] = output after decoder layers 0-3 (the shared
    # feature extraction layers). Layers 4+ are reserved for LoRA-adapted generation.
    layer_idx = min(feature_layers, len(outputs.hidden_states) - 1)
    H = outputs.hidden_states[layer_idx]   # [1, T, D]

    h_full = _masked_mean_pool(H, attention_mask)  # [1, D]
    pooled = h_full
    h = F.layer_norm(pooled.float(), (pooled.shape[-1],))   # normalise
    z = h @ projection.T.float()                            # [1, p]
    return z.cpu()


# ── Refine prompt (same template as training/refine_adapter.py) ───────────────

_REFINE_PROMPT_TEMPLATE = (
    "Fix the {language} code below. It fails with the given error.\n\n"
    "### Problem\n{instruction}\n\n"
    "### Buggy Code\n```{language}\n{buggy_code}\n```\n\n"
    "### Error\n{traceback}\n\n"
    "### Fixed Code"
)


# ── Routing combination helpers ───────────────────────────────────────────────

def _align_traceback_scores(
    tb_scores: torch.Tensor,
    tb_task_names: List[str],
    input_task_names: List[str],
    neutral: float = 0.0,
) -> torch.Tensor:
    """Expand traceback router scores [K_tb] to input router task order [K_in].

    Tasks absent from the traceback router receive the neutral z-score (0.0).
    """
    aligned = torch.full((len(input_task_names),), neutral, dtype=torch.float32)
    tb_idx = {name: idx for idx, name in enumerate(tb_task_names)}
    for in_idx, name in enumerate(input_task_names):
        if name in tb_idx:
            aligned[in_idx] = tb_scores[tb_idx[name]]
    return aligned


def _soft_disagree_explore(
    s_input: torch.Tensor,
    s_trace: torch.Tensor,
    w_input: torch.Tensor,
    w_trace: torch.Tensor,
) -> torch.Tensor:
    """JSD-gated posterior + uniform exploration.

    When the two routers agree (low JSD), trust their combined posterior.
    When they disagree strongly (high JSD), blend toward the uniform to
    avoid committing to either router's noisy signal.
    """
    m = 0.5 * (w_input + w_trace)
    eps = 1e-8
    kl_pm = (w_input * (w_input.clamp_min(eps) / m.clamp_min(eps)).log()).sum()
    kl_qm = (w_trace * (w_trace.clamp_min(eps) / m.clamp_min(eps)).log()).sum()
    js = float((0.5 * (kl_pm + kl_qm)).clamp(0.0, 1.0).item())
    K = w_input.shape[0]
    w_posterior = F.softmax(s_input + s_trace, dim=0)
    w_uniform = torch.ones(K, dtype=torch.float32) / K
    return (1.0 - js) * w_posterior + js * w_uniform




# ── Traceback feature extractor ───────────────────────────────────────────────

class _TracebackFeatureExtractor:
    """Frozen T5 encoder for traceback routing features.

    Loads the T5 model and projection matrix from a saved traceback router
    checkpoint directory (produced by gmm_traceback.py).  The encode/score
    interface mirrors RoutingFeatureExtractor in gmm.py.
    """

    def __init__(
        self,
        router: ResidualFitGMMRouter,
        router_path: str,
        device: torch.device,
    ):
        cfg = router.cfg
        self._feature_layers = cfg.feature_layers
        self._max_length = cfg.max_length
        self._device = device
        self.router = router

        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModel.from_pretrained(cfg.model_name)
        self._model.to(device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False

        if os.path.isdir(router_path):
            proj_file = os.path.join(router_path, "projection_P.pt")
        else:
            proj_file = hf_hub_download(
                repo_id=router_path, filename="projection_P.pt", repo_type="model"
            )
        self._projection = torch.load(proj_file, map_location=device).float()

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """Return routing embedding z ∈ ℝ^p for a traceback string → [1, p]."""
        orig_trunc = getattr(self._tokenizer, "truncation_side", "right")
        self._tokenizer.truncation_side = "left"   # keep the END (error line)
        try:
            enc = self._tokenizer(
                [text],
                padding=False,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
        finally:
            self._tokenizer.truncation_side = orig_trunc

        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        if hasattr(self._model, "encoder"):
            out = self._model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            out = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        layer_idx = min(self._feature_layers, len(out.hidden_states) - 1)
        H = out.hidden_states[layer_idx]          # [1, T, D]
        h = _masked_mean_pool(H, attention_mask)  # [1, D]
        h = F.layer_norm(h.float(), (h.shape[-1],))
        z = h @ self._projection.T               # [1, p]
        return z.cpu()

    def predict_scores(self, text: str) -> torch.Tensor:
        """Return z-scored GMM log-probs [K_tb] for a traceback string."""
        z = self.encode(text)
        return self.router.predict_scores(z)[0]


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    def list_of_strings(arg):
        return arg.split(',')

    parser = argparse.ArgumentParser(
        description="GMM-router inference for continual-learning LoRA models"
    )
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument(
        '--router_weight_path', type=str, default='',
        help='HF Hub repo-id OR local directory containing router_step{i}.pt and projection_P.pt',
    )
    parser.add_argument(
        '--router_method',
        choices=['gmm', 'single_gaussian', 'centroid', 'knn', 'oracle'],
        default='gmm',
        help='Routing method. single_gaussian loads a GMM checkpoint trained with M=1; '
             'oracle selects the adapter matching the evaluation task.',
    )
    parser.add_argument('--data_output_path', type=str, default='/tmp/data_files/')
    parser.add_argument(
        '--benchmark', type=str, choices=['executable', 'non-executable'],
        default='non-executable',
    )
    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help='Base LLM used for both generation and routing feature extraction '
                             '(e.g. Qwen/Qwen2.5-Coder-1.5B). Must match the model trained in gmm.py.')
    parser.add_argument('--base_path', type=str, required=True,
                        help='HF Hub repo-id or local dir containing all LoRA adapter sub-folders')
    parser.add_argument('--inference_model_path', type=str, nargs='+', required=True,
                        help='Comma-separated sub-folder paths for each task LoRA adapter')
    # GMM-specific
    parser.add_argument(
        '--routing_mode', type=str, choices=['hard', 'soft'], default='hard',
        help='hard=argmax of GMM scores (§4.4); soft=adapter merging weighted by softmax(s_k/τ) (§5)',
    )
    parser.add_argument(
        '--routing_temperature', type=float, default=1.0,
        help='Softmax temperature τ for soft routing (ignored in hard mode)',
    )
    parser.add_argument(
        '--routing_top_p', type=float, default=1.0,
        help='Top-p truncation for soft routing weights: keep fewest top tasks whose '
             'cumulative weight <= routing_top_p, then renormalize. 1.0 = no truncation.',
    )
    # Shared with infer_anyssr_total.py
    parser.add_argument('--max_prompt_len', type=list_of_strings,
                        default='320,320,256,130,512,256,256,256')
    parser.add_argument('--max_ans_len', type=list_of_strings,
                        default='150,256,128,120,300,128,128,128')
    parser.add_argument('--inference_batch', type=int, default=1,
                        help='Batch size (recommend 1 for soft routing)')
    parser.add_argument('--inference_tasks', type=list_of_strings, default='all')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--inference_output_path', type=str, default=None)
    parser.add_argument('--CL_method', default=None)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--num_return_sequences', type=int, default=5)
    parser.add_argument('--device', type=str, default='auto')

    # ── Iterative refinement (round 2+) ──────────────────────────────────────
    parser.add_argument(
        '--prev_results_dir', type=str, default=None,
        help='Directory containing executed results from the previous round '
             '(results-{i}-{task}.json for round 2, or results-{i}-{task}-round{N-1}.json '
             'for round N). When set, skips round-1 inference and runs round-2 generation '
             'on samples where ALL predictions failed.',
    )
    parser.add_argument(
        '--round_num', type=int, default=2,
        help='Round number for the output filename suffix (default: 2). '
             'Output will be results-{i}-{task}-round{round_num}.json.',
    )
    parser.add_argument(
        '--traceback_router_path', type=str, default=None,
        help='Local directory or HF Hub repo containing the traceback router '
             'checkpoint (router_step{N}.pt + projection_P.pt).',
    )
    parser.add_argument(
        '--traceback_router_step', type=int, default=None,
        help='Which router_step{N}.pt to load from traceback_router_path. '
             'Defaults to the last continual step (i).',
    )
    parser.add_argument(
        '--prev_results_source', type=str, default='local', choices=['local', 'hf_hub'],
        help='Where to load previous-round results: "local" (default) or "hf_hub".',
    )
    parser.add_argument(
        '--prev_results_repo_type', type=str, default='model',
        help='HF Hub repo type for --prev_results_dir when --prev_results_source=hf_hub (default: model).',
    )
    parser.add_argument(
        '--prev_results_subfolder', type=str, default=None,
        help='Subfolder within the HF Hub repo (or local dir) that contains the previous-round result files.',
    )
    parser.add_argument(
        '--pass_through_correct', action='store_true',
        help='Include samples with ≥1 correct prediction unchanged in the round-2 output, '
             'preserving total sample count relative to the previous round.',
    )
    parser.add_argument(
        '--pad_predictions_to', type=int, default=5,
        help='Pad re-generated prediction lists to this length with empty strings '
             '(default: 5). Set to 0 to disable padding.',
    )
    parser.add_argument(
        '--demo_dataset_path', type=str, default=None,
        help='Path to a local JSON file of {instruction, solution, test, language} rows '
             'for single/few-sample round-1 demo inference. When set, replaces the full '
             'HF test split for round-1 (non round-2) inference; tasks with no matching '
             'rows are skipped.',
    )

    return parser.parse_args()


def resolve_device(args) -> torch.device:
    if args.device != "auto":
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Use --device cpu.")
        return torch.device(args.device)
    if torch.cuda.is_available():
        if args.local_rank is not None and args.local_rank >= 0:
            return torch.device(f"cuda:{args.local_rank}")
        return torch.device("cuda")
    raise RuntimeError("No CUDA device visible. Use --device cpu.")


def _load_router_file(router_weight_path: str, filename: str) -> str:
    """Return a local path for a router file, downloading from HF Hub if needed."""
    if os.path.isdir(router_weight_path):
        return os.path.join(router_weight_path, filename)
    return hf_hub_download(repo_id=router_weight_path, filename=filename, repo_type="model")


def _load_demo_dataset(path: str, language: str) -> List[Dict[str, str]]:
    """Load a small local JSON file of {instruction, solution, test, language} rows
    for single/few-sample round-1 demo inference, bypassing the full HF test split.

    Returns a list of {"prompt": ..., "answer": ...} dicts (DataCollator's expected
    instance format), filtered to the given language. Empty if none match.
    """
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if isinstance(rows, dict):
        rows = [rows]
    return [
        {"prompt": row["instruction"], "answer": row.get("solution") or ""}
        for row in rows
        if row.get("language") == language
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = resolve_device(args)
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(device)}")

    if args.routing_mode == "soft" and args.inference_batch > 1:
        print("[WARN] Soft routing merges adapters per sample; routing is computed "
              "per-sample regardless of inference_batch.")

    if args.inference_tasks[0] == "all":
        inference_tasks = (
            AllDatasetName if args.benchmark == "non-executable"
            else AllDatasetNameExecutable
        )
    else:
        inference_tasks = args.inference_tasks

    # Infer only at the last continual step (same convention as infer_anyssr_total.py)
    i = len(inference_tasks) - 1
    inference_model_path = args.inference_model_path[0].split(',')

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else None,
        top_p=args.top_p if args.do_sample else None,
        repetition_penalty=args.repetition_penalty,
    )

    # ── Load base LLM + LoRA adapters ─────────────────────────────────────────
    # This single model serves BOTH purposes:
    #   1. Routing feature extraction (layers 0–3, adapters disabled)
    #   2. Task-conditioned generation   (layers 4+, adapter selected/merged)
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=model_dtype,
    )

    # PeftModel.from_pretrained wraps the base model so that add_weighted_adapter,
    # set_adapter, delete_adapter, and disable_adapters are all available.
    # For local directories, older PEFT versions do not resolve the subfolder in
    # _get_peft_type, so we join base_path + subfolder into a single path.
    _is_local = os.path.isdir(args.base_path)

    def _adapter_path(sub):
        return os.path.join(args.base_path, sub) if _is_local else args.base_path

    def _subfolder_kwarg(sub):
        return {} if _is_local else {"subfolder": sub}

    model = PeftModel.from_pretrained(
        base_model,
        _adapter_path(inference_model_path[0]),
        adapter_name="0",
        **_subfolder_kwarg(inference_model_path[0]),
    )
    for lora_id, lora_path in enumerate(inference_model_path[1: i + 1], start=1):
        model.load_adapter(
            _adapter_path(lora_path),
            adapter_name=str(lora_id),
            **_subfolder_kwarg(lora_path),
        )
    print(f"[INFO] Loaded adapters: {list(model.peft_config.keys())}")

    model.to(device)
    model.eval()

    # ── Load the selected routing method ───────────────────────────────────────
    if args.router_method == 'oracle':
        router = ResidualFitGMMRouter(RouterConfig(
            model_name=args.model_name_or_path,
            tasks=tuple(inference_tasks),
        ))
        router.tasks = [
            TaskRouter(task_name=task, task_id=task_id, gmm=None)
            for task_id, task in enumerate(inference_tasks)
        ]
        projection = None
    else:
        router_pt = _load_router_file(args.router_weight_path, f"router_step{i}.pt")
        proj_pt = _load_router_file(args.router_weight_path, "projection_P.pt")
        if args.router_method in {'centroid', 'knn'}:
            router = BaselineRouter.load(router_pt)
            if router.method != args.router_method:
                raise ValueError(
                    f"Requested {args.router_method} but checkpoint contains {router.method}"
                )
        else:
            router = ResidualFitGMMRouter.load(router_pt)
            if args.router_method == 'single_gaussian':
                components = {task.gmm.n_components for task in router.tasks}
                if components != {1}:
                    raise ValueError(
                        "single_gaussian requires a checkpoint trained with --gmm_components 1; "
                        f"found component counts {sorted(components)}"
                    )
        projection = torch.load(proj_pt, map_location="cpu").float().to(device)

    print(f"[INFO] Router method: {args.router_method}; {len(router.tasks)} tasks "
          f"{[tr.task_name for tr in router.tasks]}")
    print(f"[INFO] Routing model: {router.cfg.model_name}  "
          f"feature_layers={router.cfg.feature_layers}  "
          f"routing_dim={router.cfg.routing_dim}")

    if router.cfg.model_name != args.model_name_or_path:
        print(f"[WARN] Router was trained with model '{router.cfg.model_name}' but "
              f"inference uses '{args.model_name_or_path}'. "
              f"Routing features may not match training.")

    if len(router.tasks) != i + 1:
        print(f"[WARN] Router has {len(router.tasks)} tasks but "
              f"{i + 1} adapters were loaded — verify alignment.")

    router_task_names = [task.task_name for task in router.tasks]
    if router_task_names != list(inference_tasks):
        raise ValueError(
            "Router task order must match --inference_tasks and adapter order. "
            f"router={router_task_names}, inference={list(inference_tasks)}"
        )

    adapter_names = [str(tr.task_id) for tr in router.tasks]

    # ── Routing helpers ────────────────────────────────────────────────────────

    def route_sample(text: str, oracle_task: Optional[str] = None) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Compute routing for a single source string.

        Feature extraction uses layers 0–(feature_layers-1) of the generation
        model (LoRA disabled), matching the training pipeline in gmm.py.

        Returns (argmax_task_id, alpha [K], scores [K]):
          hard mode: alpha is one-hot at argmax
          soft mode: alpha = softmax(s_k / τ)  (gmm_methodology.tex §5)
          scores are the raw z-scored GMM log-probs, saved for reuse in later rounds.
        """
        if args.router_method == 'oracle':
            task_names = [task.task_name for task in router.tasks]
            if oracle_task not in task_names:
                raise ValueError(f"Oracle task '{oracle_task}' is not present in {task_names}")
            k_hat = task_names.index(oracle_task)
            alpha = torch.zeros(len(router.tasks), dtype=torch.float32)
            alpha[k_hat] = 1.0
            return k_hat, alpha, alpha.clone()

        z = extract_routing_embedding(
            model=model,
            tokenizer=tokenizer,
            text=text,
            feature_layers=router.cfg.feature_layers,
            projection=projection,
            max_length=router.cfg.max_length,
            device=device,
        )                                                    # [1, p]
        scores = router.predict_scores(z)[0]                 # [K]

        if args.routing_mode == "hard":
            k_hat = int(scores.argmax().item())
            alpha = torch.zeros(len(router.tasks))
            alpha[k_hat] = 1.0
        else:
            alpha = F.softmax(scores / args.routing_temperature, dim=0)
            if args.routing_top_p < 1.0:
                sorted_alpha, sorted_idx = torch.sort(alpha, descending=True)
                # If the top task alone already meets the threshold, keep only it
                if sorted_alpha[0].item() >= args.routing_top_p:
                    mask = torch.zeros_like(alpha)
                    mask[sorted_idx[0]] = 1.0
                    alpha = mask
                else:
                    cumsum = torch.cumsum(sorted_alpha, dim=0)
                    # Keep fewest tasks whose cumulative weight <= routing_top_p
                    n_keep = int((cumsum <= args.routing_top_p).sum().item())
                    n_keep = max(n_keep, 1)
                    keep_idx = sorted_idx[:n_keep]
                    mask = torch.zeros_like(alpha)
                    mask[keep_idx] = alpha[keep_idx]
                    alpha = mask / mask.sum()
            k_hat = int(alpha.argmax().item())

        return k_hat, alpha, scores

    # ── Generation helpers ─────────────────────────────────────────────────────

    def _generate_hard(input_ids, attention_mask, k_hat, pad_token_id,
                       max_ans_len, gen_cfg) -> torch.Tensor:
        """Select the argmax adapter for layers 4+ and generate."""
        model.set_adapter(str(router.tasks[k_hat].task_id))
        with torch.no_grad():
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_ans_len,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                generation_config=gen_cfg,
                use_cache=True,
            )

    def _generate_soft(input_ids, attention_mask, alpha, pad_token_id,
                       max_ans_len, gen_cfg) -> torch.Tensor:
        """Merge adapters with GMM weights α_k and generate.

        Implements weight-level mixing Δ W = Σ_k α_k B_k A_k (§6 of
        gmm_methodology.tex). For linear LoRA this is equivalent to
        distribution-level mixing since Σ_k α_k (B_k A_k x) = (Σ_k α_k B_k A_k) x.
        """
        merged = "__merged_gmm__"
        model.add_weighted_adapter(
            adapter_names,
            alpha.tolist(),
            merged,
            combination_type="cat",
        )
        model.set_adapter(merged)
        try:
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_ans_len,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=gen_cfg,
                    use_cache=True,
                )
        finally:
            model.delete_adapter(merged)
            if adapter_names:
                model.set_adapter(adapter_names[0])   # restore a clean base adapter
        return out

    # ── Prediction loop (mirrors infer_anyssr_total.py structure) ─────────────

    def prediction(model, tokenizer, task, test_dataloader, device,
                   generation_config, max_ans_len=None):
        model.eval()
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []
        moe_ids = []
        router_scores_list = []

        if max_ans_len is None:
            max_ans_len = 256

        is_executable = args.benchmark != "non-executable"
        if is_executable:
            num_return_sequences = int(getattr(args, "num_return_sequences", 5))
            top_k = int(getattr(args, "top_k", 0))
            gc_dict = generation_config.to_dict()
            gc_dict.update({"num_return_sequences": num_return_sequences, "top_k": top_k})
            gen_cfg = GenerationConfig(**gc_dict)
        else:
            num_return_sequences = 1
            gen_cfg = generation_config

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id

        progress_bar = tqdm(total=len(test_dataloader), leave=True)

        for step, batch in enumerate(test_dataloader):
            sources = batch['sources']
            sources_sequences += sources

            if 'gts' in batch:
                ground_truths += batch['gts']
                del batch['gts']
            elif 'labels' in batch:
                for row in batch['labels']:
                    valid_ids = row[row != -100].detach().cpu().tolist()
                    ground_truths.append(
                        tokenizer.decode(valid_ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
                    )
                del batch['labels']
            else:
                ground_truths += [''] * len(sources)

            del batch['sources']
            batch = to_device(batch, device)
            prompt_len = batch['input_ids'].shape[1]   # left-padded, same for all in batch
            batch_size  = batch['input_ids'].shape[0]

            for b in range(batch_size):
                # Routing: re-tokenise source text without left-padding so that the
                # prefix pool (first 64 tokens) captures actual prompt content,
                # matching the training feature extraction in gmm.py.
                k_hat, alpha, in_scores = route_sample(sources[b], task)

                single_ids  = batch['input_ids'][b:b+1]
                single_mask = batch['attention_mask'][b:b+1]

                if args.routing_mode == "soft":
                    # moe_id: dict mapping task_name → weight for interpretability
                    moe_id = {
                        router.tasks[k].task_name: round(float(alpha[k]), 4)
                        for k in range(len(router.tasks))
                    }
                    alpha_str = "  ".join(f"{n}:{w:.4f}" for n, w in moe_id.items())
                    print(f"[step {step}:{b}] soft routing → {alpha_str}")
                    generate_ids = _generate_soft(
                        single_ids, single_mask, alpha, pad_token_id, max_ans_len, gen_cfg,
                    )
                else:
                    moe_id = str(router.tasks[k_hat].task_id)
                    print(f"[step {step}:{b}] hard routing → "
                          f"task {moe_id} ({router.tasks[k_hat].task_name})")
                    generate_ids = _generate_hard(
                        single_ids, single_mask, k_hat, pad_token_id, max_ans_len, gen_cfg,
                    )

                seqs = tokenizer.batch_decode(
                    generate_ids[:, prompt_len:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                if is_executable and num_return_sequences > 1:
                    predicted_sequences.append(seqs)
                else:
                    predicted_sequences.extend(seqs)
                moe_ids.append(moe_id)
                router_scores_list.append(in_scores.tolist())

            progress_bar.update(1)
            progress_bar.set_description(f"step {step}", refresh=False)

        return sources_sequences, predicted_sequences, ground_truths, moe_ids, router_scores_list

    # ── Eval / save helpers (identical to infer_anyssr_total.py) ──────────────

    def _task_eval(task, sources, preds, gts):
        calc_codebleu = task not in ['CodeSearchNet', 'TheVault_Csharp']
        return compute_metrics(
            preds, gts,
            calc_codebleu=calc_codebleu,
            language=DATASET_TO_OUTPUT_LANG.get(task, None),
        )

    def save_results(evaluation_result, sources_sequences, predicted_sequences,
                     ground_truths, moe_ids, router_scores_list, i_task, task,
                     filename=None, test_cases=None, tb_scores_list=None, prompts_list=None):
        """Save inference results in the standard format.

        router_scores_list: raw z-scored GMM log-probs per sample — saved so
            subsequent rounds can reuse them without re-running the input router.
        test_cases: optional list of test strings to carry forward for execution.
        tb_scores_list: traceback router scores (round 2+), saved for analysis.
        prompts_list: refine prompts used for generation (round 2+); None for pass-through samples.
        """
        os.makedirs(args.inference_output_path, exist_ok=True)
        n = len(predicted_sequences)

        def _pad(lst, default=None):
            return (lst + [default] * n)[:n] if lst else [default] * n

        moe_ids_adj     = _pad(moe_ids)
        scores_adj      = _pad(router_scores_list)
        tb_scores_adj   = _pad(tb_scores_list) if tb_scores_list else [None] * n
        test_cases_adj  = _pad(test_cases) if test_cases else [None] * n
        prompts_adj     = _pad(prompts_list) if prompts_list else [None] * n

        rows = []
        for src, gt, pred, mid, sc, tb_sc, tc, prompt in zip(
            sources_sequences, ground_truths, predicted_sequences,
            moe_ids_adj, scores_adj, tb_scores_adj, test_cases_adj, prompts_adj,
        ):
            row = {
                "source": src,
                "ground-truth": gt,
                "prediction": pred,
                "moe_id": mid,
                "input_router_scores": sc,
            }
            if tb_sc is not None:
                row["traceback_router_scores"] = tb_sc
            if tc is not None:
                row["test"] = tc
            if prompt is not None:
                row["instruction"] = prompt
            rows.append(row)

        out = {"eval": evaluation_result, "predictions": rows}
        if filename is None:
            filename = f"results-{i_task}-{task}.json"
        out_file = os.path.join(args.inference_output_path, filename)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"[INFO] Saved results to {out_file}", flush=True)

    # ── Round-2 setup: argument validation + traceback router loading ──────────

    is_round2_mode = args.prev_results_dir is not None

    if is_round2_mode:
        if args.traceback_router_path is None:
            raise ValueError("--traceback_router_path is required when --prev_results_dir is set.")
        if args.benchmark != "executable":
            raise ValueError("Iterative refinement only supports --benchmark executable.")

        tb_step = args.traceback_router_step if args.traceback_router_step is not None else i
        tb_router_file = _load_router_file(args.traceback_router_path, f"router_step{tb_step}.pt")
        tb_router = ResidualFitGMMRouter.load(tb_router_file)
        tb_extractor = _TracebackFeatureExtractor(
            router=tb_router,
            router_path=args.traceback_router_path,
            device=device,
        )
        tb_task_names = [tr.task_name for tr in tb_router.tasks]
        in_task_names = [tr.task_name for tr in router.tasks]
        print(f"[round2] Traceback router: {len(tb_router.tasks)} tasks {tb_task_names}")
        print(f"[round2] Method: disagree_explore  Round: {args.round_num}")

    # ── Round-2 generation helper ──────────────────────────────────────────────

    def prediction_round2(prev_predictions: List[Dict], task_idx: int, task: str):
        """Generate refined predictions for samples where ALL round-N-1 predictions failed.

        Reads saved input_router_scores from prev_predictions (no re-routing),
        passes first traceback through the traceback router, combines distributions
        via the chosen method, and generates with greedy decoding.

        Returns (sources, preds, gts, moe_ids, in_scores_list, tb_scores_list).
        """
        hard_rows = [
            row for row in prev_predictions
            if all(p == 0 for p in row.get("passed", []))
        ]
        if not hard_rows:
            print(f"[round2] No hard samples for {task}.")
            return [], [], [], [], [], [], []

        print(f"[round2] {len(hard_rows)} hard samples for {task} "
              f"(all predictions failed in previous round)")

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        greedy_cfg = GenerationConfig(do_sample=False, repetition_penalty=args.repetition_penalty)

        out_sources, out_preds, out_gts = [], [], []
        out_moe_ids, out_in_scores, out_tb_scores, out_prompts = [], [], [], []

        for row in tqdm(hard_rows, desc=f"round2/{task}"):
            src = row["source"]
            gt  = row.get("ground-truth", "")

            # Reuse saved input router scores — no re-routing
            saved_scores = row.get("input_router_scores")
            if saved_scores is not None:
                in_scores = torch.tensor(saved_scores, dtype=torch.float32)
            else:
                # Fallback: re-run input router (slower, for back-compat)
                _, _, in_scores = route_sample(src, task)

            # Extract first non-empty traceback from failed predictions
            candidates  = row.get("prediction", [])
            passed_list = row.get("passed", [])
            stderr_list = row.get("stderr", [])
            first_tb   = ""
            first_buggy = candidates[0] if candidates else ""
            for p, e, code in zip(passed_list, stderr_list, candidates):
                if p == 0 and isinstance(e, str) and e.strip():
                    first_tb   = e.strip()
                    first_buggy = code
                    break

            # Traceback router scores (zero vector if no traceback)
            if first_tb:
                tb_scores_raw = tb_extractor.predict_scores(first_tb)
                tb_scores = _align_traceback_scores(tb_scores_raw, tb_task_names, in_task_names)
            else:
                tb_scores = torch.zeros(len(router.tasks), dtype=torch.float32)

            # Compute combined weights via JSD-gated posterior + uniform
            w_input = F.softmax(in_scores / args.routing_temperature, dim=0)
            w_trace = F.softmax(tb_scores, dim=0)
            w_combined = _soft_disagree_explore(in_scores, tb_scores, w_input, w_trace)

            # Build refine prompt and tokenise
            prompt = _REFINE_PROMPT_TEMPLATE.format(
                language=task,
                instruction=src,
                buggy_code=first_buggy,
                traceback=first_tb,
            )
            enc = tokenizer(
                [prompt],
                padding="longest",
                truncation=True,
                max_length=int(args.max_prompt_len[task_idx]),
                return_tensors="pt",
            )
            input_ids   = enc["input_ids"].to(device)
            attn_mask   = enc["attention_mask"].to(device)
            prompt_len  = input_ids.shape[1]

            gen_ids = _generate_soft(
                input_ids, attn_mask, w_combined, pad_token_id,
                int(args.max_ans_len[task_idx]), greedy_cfg,
            )
            k_hat = int(w_combined.argmax().item())
            moe_id = {
                router.tasks[k].task_name: round(float(w_combined[k]), 4)
                for k in range(len(router.tasks))
            }

            seqs = tokenizer.batch_decode(
                gen_ids[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            out_sources.append(src)
            out_preds.append(seqs)
            out_gts.append(gt)
            out_moe_ids.append(moe_id)
            out_in_scores.append(in_scores.tolist())
            out_tb_scores.append(tb_scores.tolist())
            out_prompts.append(prompt)

        return out_sources, out_preds, out_gts, out_moe_ids, out_in_scores, out_tb_scores, out_prompts

    # ── Evaluate all tasks seen at step i ─────────────────────────────────────

    cur_inference_tasks = inference_tasks[: i + 1]

    for task_idx, task in enumerate(cur_inference_tasks):

        if is_round2_mode:
            # ── Round-2+ mode: read previous results, refine hard samples ────
            if args.round_num == 2:
                prev_filename = f"results-{i}-{task}.json"
            else:
                prev_filename = f"results-{i}-{task}-round{args.round_num - 1}.json"

            if args.prev_results_source == "hf_hub":
                try:
                    prev_path = hf_hub_download(
                        repo_id=args.prev_results_dir,
                        filename=prev_filename,
                        repo_type=args.prev_results_repo_type,
                        subfolder=args.prev_results_subfolder,
                    )
                except Exception as e:
                    print(f"[round2] Skipping {task}: {prev_filename} not found on HF Hub ({e}).")
                    continue
            else:
                base_dir = (
                    os.path.join(args.prev_results_dir, args.prev_results_subfolder)
                    if args.prev_results_subfolder else args.prev_results_dir
                )
                prev_path = os.path.join(base_dir, prev_filename)
                if not os.path.exists(prev_path):
                    print(f"[round2] Skipping {task}: {prev_path} not found.")
                    continue

            with open(prev_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            prev_preds = prev_data.get("predictions", prev_data) if isinstance(prev_data, dict) else prev_data

            print(f"\n***** Round {args.round_num} step {i}: task {task} "
                  f"[disagree_explore routing] *****")
            hard_src, hard_pred, hard_gt, hard_moe, hard_in_sc, hard_tb_sc, hard_prompts = prediction_round2(
                prev_preds, task_idx, task
            )

            # Pad hard-sample predictions to a fixed width with empty strings
            if args.pad_predictions_to > 0 and hard_pred:
                hard_pred = [
                    (seqs + [""] * args.pad_predictions_to)[:args.pad_predictions_to]
                    for seqs in hard_pred
                ]

            sources, preds, gts, moe_ids, in_scores_list, tb_scores_list, prompts_list = (
                hard_src, hard_pred, hard_gt, hard_moe, hard_in_sc, hard_tb_sc, hard_prompts
            )

            # Merge easy (pass-through) samples with re-generated hard samples
            if args.pass_through_correct:
                easy_rows = [r for r in prev_preds if any(p != 0 for p in r.get("passed", []))]
                if easy_rows:
                    easy_src     = [r["source"]                 for r in easy_rows]
                    easy_pred    = [r["prediction"]              for r in easy_rows]
                    easy_gt      = [r.get("ground-truth", "")    for r in easy_rows]
                    easy_moe     = [r.get("moe_id")              for r in easy_rows]
                    easy_in_sc   = [r.get("input_router_scores") for r in easy_rows]
                    easy_tb_sc   = [None] * len(easy_rows)
                    easy_prompts = [None] * len(easy_rows)

                    all_src     = hard_src     + easy_src
                    all_pred    = hard_pred    + easy_pred
                    all_gt      = hard_gt      + easy_gt
                    all_moe     = hard_moe     + easy_moe
                    all_in_sc   = hard_in_sc   + easy_in_sc
                    all_tb_sc   = hard_tb_sc   + easy_tb_sc
                    all_prompts = hard_prompts + easy_prompts

                    src_to_idx = {r["source"]: idx for idx, r in enumerate(prev_preds)}
                    order = sorted(
                        range(len(all_src)),
                        key=lambda j: src_to_idx.get(all_src[j], len(prev_preds)),
                    )
                    sources        = [all_src[j]     for j in order]
                    preds          = [all_pred[j]     for j in order]
                    gts            = [all_gt[j]       for j in order]
                    moe_ids        = [all_moe[j]      for j in order]
                    in_scores_list = [all_in_sc[j]    for j in order]
                    tb_scores_list = [all_tb_sc[j]    for j in order]
                    prompts_list   = [all_prompts[j]  for j in order]

            if not sources:
                continue

            # Carry test cases from previous results for re-execution
            test_map = {row["source"]: row.get("test") for row in prev_preds}
            test_cases = [test_map.get(src) for src in sources]

            out_filename = f"results-{i}-{task}-round{args.round_num}.json"
            print("***** Saving round-2 results *****")
            save_results(
                {}, sources, preds, gts, moe_ids,
                in_scores_list, i, task,
                filename=out_filename,
                test_cases=test_cases,
                tb_scores_list=tb_scores_list,
                prompts_list=prompts_list,
            )

        else:
            # ── Round-1 mode: normal inference ────────────────────────────────
            if args.demo_dataset_path:
                infer_dataset = _load_demo_dataset(args.demo_dataset_path, task)
                if not infer_dataset:
                    print(f"[demo] Skipping {task}: no rows for this language in "
                          f"{args.demo_dataset_path}.")
                    continue
            elif args.benchmark == "non-executable":
                _, _, infer_dataset = create_codetask_dataset(task, args.seed, -1, -1, -1)
            else:
                _, _, infer_dataset = create_executable_dataset(task, args.seed, -1, -1, -1)

            inf_data_collator = DataCollator(
                tokenizer,
                model=model,
                padding="longest",
                max_prompt_len=int(args.max_prompt_len[task_idx]),
                max_ans_len=int(args.max_ans_len[task_idx]),
                pad_to_multiple_of=8,
                inference=True,
            )
            infer_dataloader = DataLoader(
                infer_dataset,
                collate_fn=inf_data_collator,
                sampler=SequentialSampler(infer_dataset),
                batch_size=args.inference_batch,
            )

            assert tokenizer.padding_side == 'left'
            assert tokenizer.truncation_side == 'left'

            print(f"\n***** Inference step {i}: task {task} [{args.routing_mode} routing] *****")
            sources, preds, gts, moe_ids, in_scores_list = prediction(
                model, tokenizer, task, infer_dataloader, device, generation_config,
                max_ans_len=int(args.max_ans_len[task_idx]),
            )

            evaluation_result = (
                _task_eval(task, sources, preds, gts)
                if args.benchmark == "non-executable"
                else {}
            )

            print("***** Saving results *****")
            save_results(
                evaluation_result, sources, preds, gts, moe_ids,
                in_scores_list, i, task,
            )


if __name__ == "__main__":
    main()
