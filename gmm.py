from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from t5_data import T5Dataset


@dataclass
class RouterConfig:
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B"
    output_dir: str = "./router_gmm_ckpt"

    tasks: Tuple[str, ...] = (
        "CONCODE",
        "CodeTrans",
        "CodeSearchNet",
        "BFP",
        "KodCode",
        "RunBugRun",
    )

    feature_layers: int = 4
    routing_dim: int = 128
    max_length: int = 512
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

def t5_pad_collate(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0, label_pad_id: int = -100):
    max_input_len = max(x["input_ids"].numel() for x in batch)
    input_ids = []
    attention_mask = []

    for x in batch:
        ids = x["input_ids"].long()
        mask = x["attention_mask"].long()

        pad_len = max_input_len - ids.numel()

        input_ids.append(
            torch.cat([
                ids,
                torch.full((pad_len,), pad_token_id, dtype=torch.long)
            ])
        )

        attention_mask.append(
            torch.cat([
                mask,
                torch.zeros(pad_len, dtype=torch.long)
            ])
        )

    out = {
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
    }

    if "labels" in batch[0]:
        max_label_len = max(x["labels"].numel() for x in batch)

        labels = []
        for x in batch:
            y = x["labels"].long()
            pad_len = max_label_len - y.numel()

            labels.append(
                torch.cat([
                    y,
                    torch.full((pad_len,), label_pad_id, dtype=torch.long)
                ])
            )

        out["labels"] = torch.stack(labels, dim=0)

    return out

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_tasks(raw: Optional[str], default: Tuple[str, ...]) -> Tuple[str, ...]:
    if raw is None or raw.strip() == "":
        return default
    return tuple(t.strip() for t in raw.split(",") if t.strip())

class T5RoutingFeatureExtractor:

    def __init__(self, model_name: str, feature_layers: int, routing_dim: int, device: torch.device, seed: int):
        self.model_name = model_name
        self.feature_layers = feature_layers
        self.routing_dim = routing_dim
        self.device = device
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        hidden_size = self.model.config.d_model
        self.P = self._make_row_orthonormal_projection(
            p=routing_dim,
            d=hidden_size,
            seed=seed,
            device=device,
        )

    @staticmethod
    def _make_row_orthonormal_projection(p: int, d: int, seed: int, device: torch.device) -> torch.Tensor:
        if p > d:
            raise ValueError(f"routing_dim p={p} must be <= hidden_size d={d}")

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        A = torch.randn(d, p, generator=gen) / math.sqrt(p)
        Q, _ = torch.linalg.qr(A, mode="reduced")
        P = Q.T.contiguous().to(device)
        return P

    def save_projection(self, output_dir: str | Path) -> None:
        output_dir = ensure_dir(output_dir)
        torch.save(self.P.detach().cpu(), output_dir / "projection_P.pt")

    def load_projection(self, path: str | Path) -> None:
        self.P = torch.load(path, map_location=self.device).to(self.device)

    @torch.no_grad()
    def encode_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        enc = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        layer_idx = min(self.feature_layers, len(enc.hidden_states) - 1)
        H = enc.hidden_states[layer_idx]  # [B, T, D]

        mask = attention_mask.unsqueeze(-1).to(H.dtype)  # [B, T, 1]
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (H * mask).sum(dim=1) / denom  # [B, D]

        h = F.layer_norm(pooled.float(), normalized_shape=(pooled.shape[-1],))
        z = h @ self.P.T.float()  # [B, p]
        return z.detach().cpu()

    @torch.no_grad()
    def extract_features(self, dataloader: Iterable[Dict[str, torch.Tensor]], desc: str) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for batch in tqdm(dataloader, desc=desc):
            chunks.append(self.encode_batch(batch))
        if not chunks:
            raise RuntimeError(f"No features extracted for {desc}")
        return torch.cat(chunks, dim=0).float()

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
        log_det = torch.log(var_exp).sum(dim=-1)  # [1, M]
        quad = ((z_exp - mu_exp) ** 2 / var_exp).sum(dim=-1)  # [N, M]
        p = z.shape[-1]
        return -0.5 * (p * math.log(2.0 * math.pi) + log_det + quad)

    def _init_params(self, z: torch.Tensor, weights: torch.Tensor, seed: int) -> DiagonalGMMState:
        N, p = z.shape
        M = min(self.n_components, N)
        if M < self.n_components:
            print(f"[warn] n_components reduced from {self.n_components} to {M} because N={N}")
            self.n_components = M

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        prob = weights.clamp_min(self.eps)
        prob = prob / prob.sum()
        idx = torch.multinomial(prob, num_samples=M, replacement=False, generator=gen)
        mu = z[idx].clone()

        global_var = torch.var(z, dim=0, unbiased=False).clamp_min(self.variance_floor)
        var = global_var.unsqueeze(0).repeat(M, 1).clone()
        pi = torch.ones(M, dtype=z.dtype) / M
        return DiagonalGMMState(pi=pi, mu=mu, var=var)

    def fit(self, z: torch.Tensor, sample_weights: Optional[torch.Tensor] = None, em_iters: int = 50, tol: float = 1e-4, seed: int = 42) -> "WeightedDiagonalGMM":
        z = z.float().cpu()
        N, _ = z.shape
        if N == 0:
            raise ValueError("Cannot fit GMM with zero samples")

        if sample_weights is None:
            w = torch.ones(N, dtype=torch.float32)
        else:
            w = sample_weights.float().cpu().clamp_min(self.eps)

        state = self._init_params(z, w, seed=seed)
        prev_ll = None

        for _ in range(em_iters):
            log_prob_comp = self._log_diag_gaussian(z, state.mu, state.var)  # [N, M]
            log_joint = torch.log(state.pi.clamp_min(self.eps))[None, :] + log_prob_comp
            log_norm = torch.logsumexp(log_joint, dim=1)  # [N]
            resp = torch.exp(log_joint - log_norm[:, None])  # [N, M]

            wr = w[:, None] * resp
            Nk = wr.sum(dim=0).clamp_min(self.eps)  # [M]
            omega = w.sum().clamp_min(self.eps)

            pi = Nk / omega
            mu = (wr.T @ z) / Nk[:, None]

            diff = z[:, None, :] - mu[None, :, :]
            var = (wr[:, :, None] * diff.pow(2)).sum(dim=0) / Nk[:, None]
            var = var.clamp_min(self.variance_floor)

            state = DiagonalGMMState(pi=pi, mu=mu, var=var)

            weighted_ll = (w * log_norm).sum() / omega
            if prev_ll is not None and abs(float(weighted_ll - prev_ll)) < tol:
                break
            prev_ll = weighted_ll.detach()

        self.state = state
        return self

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("GMM is not fitted")
        z = z.float().cpu()
        log_prob_comp = self._log_diag_gaussian(z, self.state.mu, self.state.var)
        log_joint = torch.log(self.state.pi.clamp_min(self.eps))[None, :] + log_prob_comp
        return torch.logsumexp(log_joint, dim=1)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        if self.state is None:
            raise RuntimeError("GMM is not fitted")
        return {
            "pi": self.state.pi,
            "mu": self.state.mu,
            "var": self.state.var,
            "n_components": torch.tensor(self.n_components),
            "variance_floor": torch.tensor(self.variance_floor),
            "eps": torch.tensor(self.eps),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, torch.Tensor]) -> "WeightedDiagonalGMM":
        obj = cls(
            n_components=int(d["n_components"]),
            variance_floor=float(d["variance_floor"]),
            eps=float(d["eps"]),
        )
        obj.state = DiagonalGMMState(pi=d["pi"].float(), mu=d["mu"].float(), var=d["var"].float())
        return obj

@dataclass
class TaskRouter:
    task_name: str
    task_id: int
    gmm: WeightedDiagonalGMM
    a: float
    b: float


class ResidualFitGMMRouter:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self.tasks: List[TaskRouter] = []

    def _normalized_scores_for_existing_tasks(self, z: torch.Tensor) -> torch.Tensor:
        if not self.tasks:
            raise RuntimeError("No old tasks available")

        scores = []
        for tr in self.tasks:
            logp = tr.gmm.log_prob(z)
            s = (logp - tr.a) / (tr.b + self.cfg.eps)
            scores.append(s)
        return torch.stack(scores, dim=1)  # [N, K]

    def novelty_weights(self, z: torch.Tensor) -> torch.Tensor:
        if not self.tasks:
            return torch.ones(z.shape[0], dtype=torch.float32)

        old_scores = self._normalized_scores_for_existing_tasks(z)
        coverage = old_scores.max(dim=1).values
        novelty = 1.0 - torch.sigmoid((coverage - self.cfg.kappa) / self.cfg.tau_n)
        novelty = torch.clamp(novelty, min=self.cfg.omega_min, max=1.0)
        return novelty.float()

    def fit_new_task(self, task_name: str, task_id: int, z_train: torch.Tensor) -> TaskRouter:
        w = self.novelty_weights(z_train)

        gmm = WeightedDiagonalGMM(
            n_components=self.cfg.gmm_components,
            variance_floor=self.cfg.variance_floor,
            eps=self.cfg.eps,
        )
        gmm.fit(
            z_train,
            sample_weights=w,
            em_iters=self.cfg.em_iters,
            tol=self.cfg.em_tol,
            seed=self.cfg.seed + task_id,
        )

        logp = gmm.log_prob(z_train)
        omega = w.sum().clamp_min(self.cfg.eps)
        a = float((w * logp).sum() / omega)
        b = float(torch.sqrt((w * (logp - a).pow(2)).sum() / omega + self.cfg.eps))

        tr = TaskRouter(task_name=task_name, task_id=task_id, gmm=gmm, a=a, b=b)
        self.tasks.append(tr)

        print(
            f"[fit] task={task_id}:{task_name} "
            f"N={len(z_train)} omega_mean={float(w.mean()):.4f} "
            f"a={a:.4f} b={b:.4f}"
        )
        return tr

    def predict_scores(self, z: torch.Tensor) -> torch.Tensor:
        return self._normalized_scores_for_existing_tasks(z)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        scores = self.predict_scores(z)
        pred_local = scores.argmax(dim=1)
        task_ids = torch.tensor([tr.task_id for tr in self.tasks], dtype=torch.long)
        return task_ids[pred_local]

    def save(self, output_dir: str | Path, step: int) -> None:
        output_dir = ensure_dir(output_dir)
        payload = {
            "cfg": asdict(self.cfg),
            "step": step,
            "tasks": [
                {
                    "task_name": tr.task_name,
                    "task_id": tr.task_id,
                    "a": tr.a,
                    "b": tr.b,
                    "gmm": tr.gmm.to_dict(),
                }
                for tr in self.tasks
            ],
        }
        torch.save(payload, output_dir / f"router_step{step}.pt")

    @classmethod
    def load(cls, path: str | Path) -> "ResidualFitGMMRouter":
        payload = torch.load(path, map_location="cpu")
        cfg = RouterConfig(**payload["cfg"])
        router = cls(cfg)
        for item in payload["tasks"]:
            router.tasks.append(
                TaskRouter(
                    task_name=item["task_name"],
                    task_id=int(item["task_id"]),
                    gmm=WeightedDiagonalGMM.from_dict(item["gmm"]),
                    a=float(item["a"]),
                    b=float(item["b"]),
                )
            )
        return router


def build_dataloader(dataset_builder: T5Dataset, task: str, split: str, batch_size: int, k: int, seed: int, max_length: int):
    old_dl = dataset_builder.get_final_ds(
        task=task,
        split=split,
        batch_size=batch_size,
        k=k,
        seed=seed,
        return_test=False,
        max_length=max_length,
    )

    pad_token_id = dataset_builder.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0

    return DataLoader(
        old_dl.dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=lambda batch: t5_pad_collate(
            batch,
            pad_token_id=pad_token_id,
            label_pad_id=-100,
        ),
    )

def feature_cache_path(output_dir: str | Path, task: str, split: str, k: int, feature_layers: int, routing_dim: int) -> Path:
    safe_task = task.replace("/", "_")
    return Path(output_dir) / "features" / f"{safe_task}_{split}_k{k}_L{feature_layers}_p{routing_dim}.pt"


def get_or_extract_features(
    extractor: T5RoutingFeatureExtractor,
    dataset_builder: T5Dataset,
    cfg: RouterConfig,
    task: str,
    split: str,
    k: int,
) -> torch.Tensor:
    path = feature_cache_path(cfg.output_dir, task, split, k, cfg.feature_layers, cfg.routing_dim)
    ensure_dir(path.parent)

    if cfg.save_features and path.exists() and not cfg.force_recompute_features:
        return torch.load(path, map_location="cpu").float()

    dl = build_dataloader(
        dataset_builder=dataset_builder,
        task=task,
        split=split,
        batch_size=cfg.batch_size,
        k=k,
        seed=cfg.seed,
        max_length=cfg.max_length,
    )
    z = extractor.extract_features(dl, desc=f"extract {task}/{split}")

    if cfg.save_features:
        torch.save(z, path)
    return z


@dataclass
class EvalResult:
    overall_acc: float
    per_task_acc: Dict[str, float]
    confusion: List[List[int]]


def evaluate_seen_tasks(
    router: ResidualFitGMMRouter,
    extractor: T5RoutingFeatureExtractor,
    dataset_builder: T5Dataset,
    cfg: RouterConfig,
    seen_tasks: List[str],
    split: str,
) -> EvalResult:
    K = len(seen_tasks)
    confusion = torch.zeros(K, K, dtype=torch.long)
    correct_total = 0
    n_total = 0
    per_task_acc: Dict[str, float] = {}

    for true_id, task in enumerate(seen_tasks):
        z = get_or_extract_features(
            extractor=extractor,
            dataset_builder=dataset_builder,
            cfg=cfg,
            task=task,
            split=split,
            k=cfg.eval_k,
        )
        pred = router.predict(z)
        y = torch.full_like(pred, fill_value=true_id)

        correct = int((pred == y).sum().item())
        total = int(y.numel())
        correct_total += correct
        n_total += total
        per_task_acc[task] = correct / max(total, 1)

        for yt, yp in zip(y.tolist(), pred.tolist()):
            if 0 <= yt < K and 0 <= yp < K:
                confusion[yt, yp] += 1

    overall_acc = correct_total / max(n_total, 1)
    return EvalResult(
        overall_acc=overall_acc,
        per_task_acc=per_task_acc,
        confusion=confusion.tolist(),
    )


def print_eval(step: int, seen_tasks: List[str], result: EvalResult) -> None:
    print("\n" + "=" * 90)
    print(f"[eval] step={step} seen_tasks={seen_tasks}")
    print(f"[eval] overall routing acc = {result.overall_acc:.4f}")
    for task, acc in result.per_task_acc.items():
        print(f"  - {task:<18s}: {acc:.4f}")

    print("[eval] confusion rows=true, cols=pred")
    header = "true\\pred" + "".join([f"\t{i}:{t[:8]}" for i, t in enumerate(seen_tasks)])
    print(header)
    for i, row in enumerate(result.confusion):
        print(f"{i}:{seen_tasks[i][:8]}" + "".join([f"\t{v}" for v in row]))
    print("=" * 90 + "\n")


def run(cfg: RouterConfig) -> None:
    set_seed(cfg.seed)
    output_dir = ensure_dir(cfg.output_dir)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = get_device()
    print(f"[setup] device={device}")
    print(f"[setup] tasks={list(cfg.tasks)}")

    extractor = T5RoutingFeatureExtractor(
        model_name=cfg.model_name,
        feature_layers=cfg.feature_layers,
        routing_dim=cfg.routing_dim,
        device=device,
        seed=cfg.seed,
    )
    extractor.save_projection(output_dir)

    dataset_builder = T5Dataset(extractor.tokenizer)
    router = ResidualFitGMMRouter(cfg)

    all_results: Dict[str, Dict] = {}

    for task_id, task in enumerate(cfg.tasks):
        print("\n" + "#" * 90)
        print(f"[continual] learn task {task_id}: {task}")
        print("#" * 90)

        z_train = get_or_extract_features(
            extractor=extractor,
            dataset_builder=dataset_builder,
            cfg=cfg,
            task=task,
            split="train",
            k=cfg.train_k,
        )

        router.fit_new_task(task_name=task, task_id=task_id, z_train=z_train)
        router.save(output_dir, step=task_id)

        seen_tasks = list(cfg.tasks[: task_id + 1])
        result = evaluate_seen_tasks(
            router=router,
            extractor=extractor,
            dataset_builder=dataset_builder,
            cfg=cfg,
            seen_tasks=seen_tasks,
            split=cfg.eval_split,
        )
        print_eval(step=task_id, seen_tasks=seen_tasks, result=result)

        all_results[f"step{task_id}"] = {
            "seen_tasks": seen_tasks,
            "overall_acc": result.overall_acc,
            "per_task_acc": result.per_task_acc,
            "confusion": result.confusion,
        }
        with open(output_dir / "routing_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    print(f"[done] saved router checkpoints and results to: {output_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default=RouterConfig.model_name)
    p.add_argument("--output_dir", type=str, default=RouterConfig.output_dir)
    p.add_argument("--tasks", type=str, default=None, help="Comma-separated task list. Default uses all T5Dataset tasks.")

    p.add_argument("--feature_layers", type=int, default=RouterConfig.feature_layers)
    p.add_argument("--routing_dim", type=int, default=RouterConfig.routing_dim)
    p.add_argument("--max_length", type=int, default=RouterConfig.max_length)
    p.add_argument("--batch_size", type=int, default=RouterConfig.batch_size)

    p.add_argument("--train_k", type=int, default=RouterConfig.train_k)
    p.add_argument("--eval_k", type=int, default=RouterConfig.eval_k)
    p.add_argument("--seed", type=int, default=RouterConfig.seed)

    p.add_argument("--gmm_components", type=int, default=RouterConfig.gmm_components)
    p.add_argument("--em_iters", type=int, default=RouterConfig.em_iters)
    p.add_argument("--em_tol", type=float, default=RouterConfig.em_tol)
    p.add_argument("--variance_floor", type=float, default=RouterConfig.variance_floor)
    p.add_argument("--eps", type=float, default=RouterConfig.eps)

    p.add_argument("--omega_min", type=float, default=RouterConfig.omega_min)
    p.add_argument("--kappa", type=float, default=RouterConfig.kappa)
    p.add_argument("--tau_n", type=float, default=RouterConfig.tau_n)

    p.add_argument("--eval_split", type=str, default=RouterConfig.eval_split, choices=["validation", "test"])
    p.add_argument("--no_save_features", action="store_true")
    p.add_argument("--force_recompute_features", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    default_cfg = RouterConfig()
    cfg = RouterConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        tasks=parse_tasks(args.tasks, default_cfg.tasks),
        feature_layers=args.feature_layers,
        routing_dim=args.routing_dim,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_k=args.train_k,
        eval_k=args.eval_k,
        seed=args.seed,
        gmm_components=args.gmm_components,
        em_iters=args.em_iters,
        em_tol=args.em_tol,
        variance_floor=args.variance_floor,
        eps=args.eps,
        omega_min=args.omega_min,
        kappa=args.kappa,
        tau_n=args.tau_n,
        eval_split=args.eval_split,
        save_features=not args.no_save_features,
        force_recompute_features=args.force_recompute_features,
    )
    run(cfg)


if __name__ == "__main__":
    main()
