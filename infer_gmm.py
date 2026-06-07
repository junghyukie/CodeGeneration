"""
GMM-router inference pipeline.

Architecture (matches gmm_methodology.tex §2–6 and the training setup in train1.sh):
- Layers 0–3 of the generation LLM act as the frozen routing feature extractor.
  LoRA adapters are disabled for this stage (matches gmm.py which uses a fresh
  copy of the same model for feature extraction, with no LoRA).
- The layer-4 hidden state is dual-pooled (prefix-64 + full), LN-normalised, and
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
    AutoModelForCausalLM,
    GenerationConfig,
)
from peft import PeftModel

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
    omega_min: float = 0.05
    kappa: float = 0.0
    tau_n: float = 1.0
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
    a: float   # weighted-mean log-likelihood on training set
    b: float   # weighted-std  log-likelihood on training set


class ResidualFitGMMRouter:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self.tasks: List[TaskRouter] = []

    def predict_scores(self, z: torch.Tensor) -> torch.Tensor:
        """Normalised GMM scores s_k(z) for all tasks. Returns [N, K]."""
        scores = []
        for tr in self.tasks:
            logp = tr.gmm.log_prob(z)
            s = (logp - tr.a) / (tr.b + self.cfg.eps)
            scores.append(s)
        return torch.stack(scores, dim=1)

    @classmethod
    def load(cls, path: str) -> "ResidualFitGMMRouter":
        payload = torch.load(path, map_location="cpu")
        cfg = RouterConfig(**payload["cfg"])
        router = cls(cfg)
        for item in payload["tasks"]:
            router.tasks.append(TaskRouter(
                task_name=item["task_name"],
                task_id=int(item["task_id"]),
                gmm=WeightedDiagonalGMM.from_dict(item["gmm"]),
                a=float(item["a"]),
                b=float(item["b"]),
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
      → dual-pool (prefix-64 + full) → LN → projection P → z

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

    prefix_len = min(64, H.shape[1])
    h_prefix = _masked_mean_pool(H[:, :prefix_len, :], attention_mask[:, :prefix_len])
    h_full   = _masked_mean_pool(H, attention_mask)

    pooled = torch.cat([h_prefix, h_full], dim=-1)          # [1, 2D]
    h = F.layer_norm(pooled.float(), (pooled.shape[-1],))   # normalise
    z = h @ projection.T.float()                            # [1, p]
    return z.cpu()


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
    model = PeftModel.from_pretrained(
        base_model,
        args.base_path,
        adapter_name="0",
        subfolder=inference_model_path[0],
    )
    for lora_id, lora_path in enumerate(inference_model_path[1: i + 1], start=1):
        model.load_adapter(
            args.base_path,
            adapter_name=str(lora_id),
            subfolder=lora_path,
        )
    print(f"[INFO] Loaded adapters: {list(model.peft_config.keys())}")

    model.to(device)
    model.eval()

    # ── Load GMM router checkpoint ─────────────────────────────────────────────
    router_pt = _load_router_file(args.router_weight_path, f"router_step{i}.pt")
    proj_pt   = _load_router_file(args.router_weight_path, "projection_P.pt")

    router = ResidualFitGMMRouter.load(router_pt)
    projection = torch.load(proj_pt, map_location="cpu").float().to(device)  # [p, 2*hidden]
    print(f"[INFO] GMM router: {len(router.tasks)} tasks "
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

    adapter_names = [str(tr.task_id) for tr in router.tasks]

    # ── Routing helpers ────────────────────────────────────────────────────────

    def route_sample(text: str) -> Tuple[int, torch.Tensor]:
        """Compute routing for a single source string.

        Feature extraction uses layers 0–(feature_layers-1) of the generation
        model (LoRA disabled), matching the training pipeline in gmm.py.

        Returns (argmax_task_id, alpha [K]):
          hard mode: alpha is one-hot at argmax
          soft mode: alpha = softmax(s_k / τ)  (gmm_methodology.tex §5)
        """
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
            k_hat = int(alpha.argmax().item())

        return k_hat, alpha

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
            combination_type="linear",
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
                k_hat, alpha = route_sample(sources[b])

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

            progress_bar.update(1)
            progress_bar.set_description(f"step {step}", refresh=False)

        return sources_sequences, predicted_sequences, ground_truths, moe_ids

    # ── Eval / save helpers (identical to infer_anyssr_total.py) ──────────────

    def _task_eval(task, sources, preds, gts):
        calc_codebleu = task not in ['CodeSearchNet', 'TheVault_Csharp']
        return compute_metrics(
            preds, gts,
            calc_codebleu=calc_codebleu,
            language=DATASET_TO_OUTPUT_LANG.get(task, None),
        )

    def save_results(evaluation_result, sources_sequences, predicted_sequences,
                     ground_truths, moe_ids, i_task, task):
        os.makedirs(args.inference_output_path, exist_ok=True)
        if len(moe_ids) != len(predicted_sequences):
            moe_ids_adj = (moe_ids + [None] * len(predicted_sequences))[:len(predicted_sequences)]
        else:
            moe_ids_adj = moe_ids
        rows = [
            {"source": src, "ground-truth": gt, "prediction": pred, "moe_id": mid}
            for src, gt, pred, mid in zip(
                sources_sequences, ground_truths, predicted_sequences, moe_ids_adj
            )
        ]
        out = {"eval": evaluation_result, "predictions": rows}
        out_file = os.path.join(args.inference_output_path, f"results-{i_task}-{task}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"[INFO] Saved results to {out_file}", flush=True)

    # ── Evaluate all tasks seen at step i ─────────────────────────────────────

    cur_inference_tasks = inference_tasks[: i + 1]

    for task_idx, task in enumerate(cur_inference_tasks):
        if args.benchmark == "non-executable":
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
        sources, preds, gts, moe_ids = prediction(
            model, tokenizer, task, infer_dataloader, device, generation_config,
            max_ans_len=int(args.max_ans_len[task_idx]),
        )

        evaluation_result = (
            _task_eval(task, sources, preds, gts)
            if args.benchmark == "non-executable"
            else {}
        )

        print("***** Saving results *****")
        save_results(evaluation_result, sources, preds, gts, moe_ids, i, task)


if __name__ == "__main__":
    main()
