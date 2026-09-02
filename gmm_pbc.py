from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

@dataclass
class RouterConfig:
    model_name: str = "SalesForce/codet5-small"
    output_dir: str = "./router_gmm_pbc_ckpt"
    feature_cache_dir: Optional[str] = None
    cache_tag: str = "v1"
    dataset_source: str = "codetask"
    executable_dataset_name: str = "ankhanhtran02/CL4Code-executable-datasets"
    tasks: Tuple[str, ...] = (
        "CONCODE",
        "CodeTrans",
        "CodeSearchNet",
        "BFP",
        "KodCode",
        "RunBugRun",
        "TheVault_Csharp",
        "CoST",
    )
    feature_layers: int = 4
    routing_dim: int = 256
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
    eval_split: str = "test"
    save_features: bool = True
    force_recompute_features: bool = False
    gmm_fit_fraction: float = 0.70
    boundary_fit_fraction: float = 0.15
    old_pseudo_fit_samples: int = 6000
    old_pseudo_cert_samples: int = 20000
    bootstrap_replicates: int = 2000
    bootstrap_alpha: float = 0.05
    margin_threshold: float = 0.5


def build_feature_cache_manifest(cfg: RouterConfig, extractor) -> Dict[str, object]:
    projection = extractor.P.detach().float().cpu().contiguous()
    projection_hasher = hashlib.sha256()
    projection_hasher.update(str(tuple(projection.shape)).encode("utf-8"))
    projection_hasher.update(str(projection.dtype).encode("utf-8"))
    projection_hasher.update(projection.numpy().tobytes())

    model_config = getattr(extractor.model, "config", None)
    model_config_values = (
        model_config.to_dict()
        if model_config is not None and hasattr(model_config, "to_dict")
        else {}
    )
    model_config_hash = hashlib.sha256(
        json.dumps(
            model_config_values,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    tokenizer_init = getattr(extractor.tokenizer, "init_kwargs", {}) or {}
    try:
        tokenizer_vocab = extractor.tokenizer.get_vocab()
    except (AttributeError, NotImplementedError):
        tokenizer_vocab = None
    tokenizer_vocab_hash = (
        hashlib.sha256(
            json.dumps(
                sorted(tokenizer_vocab.items()),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if tokenizer_vocab is not None
        else None
    )
    try:
        extractor_source = inspect.getsource(type(extractor))
    except (OSError, TypeError):
        extractor_source = (
            f"{type(extractor).__module__}.{type(extractor).__qualname__}"
        )

    def hash_local_artifacts(model_name: str) -> Optional[str]:
        model_path = Path(model_name)
        if not model_path.exists():
            return None
        files = [model_path] if model_path.is_file() else sorted(
            path for path in model_path.rglob("*") if path.is_file()
        )
        artifact_hasher = hashlib.sha256()
        for file_path in files:
            relative_name = (
                file_path.name
                if model_path.is_file()
                else file_path.relative_to(model_path).as_posix()
            )
            artifact_hasher.update(relative_name.encode("utf-8"))
            artifact_hasher.update(str(file_path.stat().st_size).encode("utf-8"))
            with open(file_path, "rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    artifact_hasher.update(chunk)
        return artifact_hasher.hexdigest()

    manifest: Dict[str, object] = {
        "schema_version": 1,
        "cache_tag": cfg.cache_tag,
        "model_name": cfg.model_name,
        "model_revision": getattr(model_config, "_commit_hash", None),
        "model_config_sha256": model_config_hash,
        "local_model_artifacts_sha256": hash_local_artifacts(cfg.model_name),
        "model_class": type(extractor.model).__name__,
        "tokenizer_name": getattr(extractor.tokenizer, "name_or_path", None),
        "tokenizer_revision": tokenizer_init.get("_commit_hash"),
        "tokenizer_vocab_sha256": tokenizer_vocab_hash,
        "tokenizer_class": type(extractor.tokenizer).__name__,
        "feature_extractor_code_sha256": hashlib.sha256(
            extractor_source.encode("utf-8")
        ).hexdigest(),
        "torch_version": str(torch.__version__),
        "feature_layers": cfg.feature_layers,
        "routing_dim": cfg.routing_dim,
        "max_length": cfg.max_length,
        "seed": cfg.seed,
        "projection_shape": list(projection.shape),
        "projection_sha256": projection_hasher.hexdigest(),
        "dataset_source": cfg.dataset_source,
        "codetask_dataset": "dongg18/CODETASK_with_instruction_pool",
        "executable_dataset_name": cfg.executable_dataset_name,
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return manifest


def prepare_feature_cache_config(
    cfg: RouterConfig,
    extractor,
) -> Tuple[RouterConfig, Dict[str, object], Path]:
    manifest = build_feature_cache_manifest(cfg, extractor)
    cache_root = (
        Path(cfg.feature_cache_dir)
        if cfg.feature_cache_dir
        else Path(cfg.output_dir) / "pbc_feature_cache"
    )
    namespace = ensure_dir(cache_root / str(manifest["fingerprint"]))
    manifest_path = namespace / "cache_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as handle:
            stored_manifest = json.load(handle)
        if stored_manifest != manifest:
            raise RuntimeError(
                f"Feature-cache manifest mismatch at {manifest_path}; "
                "use a clean cache root or recompute features."
            )
    else:
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
    return (
        replace(cfg, feature_cache_dir=str(namespace)),
        manifest,
        namespace,
    )


def validate_feature_tensor(
    features: torch.Tensor,
    cfg: RouterConfig,
    context: str,
) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] != cfg.routing_dim:
        raise ValueError(
            f"{context} feature tensor must have routing_dim={cfg.routing_dim}; "
            f"got shape={tuple(features.shape)}"
        )
    if not torch.isfinite(features).all():
        raise ValueError(f"{context} feature tensor must contain only finite values")
    return features


def validate_representation_manifest(
    stored_manifest: Optional[Dict[str, object]],
    current_manifest: Dict[str, object],
) -> None:
    if stored_manifest is None:
        raise ValueError(
            "Checkpoint has no representation manifest and cannot be resumed safely"
        )
    if stored_manifest != current_manifest:
        raise ValueError(
            "Checkpoint representation does not match the current model/tokenizer/"
            "projection/data configuration. Resume with the original representation."
        )


def get_pbc_features(
    extractor,
    cfg: RouterConfig,
    task: str,
    split: str,
    k: int,
) -> torch.Tensor:
    from gmm import get_or_extract_features

    features = get_or_extract_features(
        extractor=extractor,
        cfg=cfg,
        task=task,
        split=split,
        k=k,
    )
    try:
        return validate_feature_tensor(features, cfg, f"{task}/{split}")
    except ValueError:
        if cfg.force_recompute_features or not cfg.save_features:
            raise
        print(
            f"[cache] invalid cached feature tensor for {task}/{split}; recomputing"
        )
        recompute_cfg = replace(cfg, force_recompute_features=True)
        features = get_or_extract_features(
            extractor=extractor,
            cfg=recompute_cfg,
            task=task,
            split=split,
            k=k,
        )
        return validate_feature_tensor(features, cfg, f"{task}/{split}")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@dataclass
class DiagonalGMMState:
    pi: torch.Tensor
    mu: torch.Tensor
    var: torch.Tensor


class WeightedDiagonalGMM:
    def __init__(
        self,
        n_components: int,
        variance_floor: float = 1e-4,
        eps: float = 1e-8,
    ):
        self.n_components = n_components
        self.variance_floor = variance_floor
        self.eps = eps
        self.state: Optional[DiagonalGMMState] = None

    @staticmethod
    def _log_diag_gaussian(
        z: torch.Tensor,
        mu: torch.Tensor,
        var: torch.Tensor,
    ) -> torch.Tensor:
        z_expanded = z[:, None, :]
        mu_expanded = mu[None, :, :]
        var_expanded = var[None, :, :]
        log_det = torch.log(var_expanded).sum(dim=-1)
        quadratic = (
            (z_expanded - mu_expanded).pow(2) / var_expanded
        ).sum(dim=-1)
        dimensions = z.shape[-1]
        return -0.5 * (
            dimensions * math.log(2.0 * math.pi) + log_det + quadratic
        )

    def _init_params(
        self,
        z: torch.Tensor,
        weights: torch.Tensor,
        seed: int,
    ) -> DiagonalGMMState:
        n_samples = len(z)
        component_count = min(self.n_components, n_samples)
        if component_count < self.n_components:
            print(
                f"[warn] n_components reduced from {self.n_components} to "
                f"{component_count} because N={n_samples}"
            )
            self.n_components = component_count

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        probabilities = weights.clamp_min(self.eps)
        probabilities = probabilities / probabilities.sum()
        indices = torch.multinomial(
            probabilities,
            num_samples=component_count,
            replacement=False,
            generator=generator,
        )
        means = z[indices].clone()
        global_variance = torch.var(z, dim=0, unbiased=False).clamp_min(
            self.variance_floor
        )
        variances = global_variance.unsqueeze(0).repeat(component_count, 1)
        mixture_weights = torch.ones(component_count, dtype=z.dtype) / component_count
        return DiagonalGMMState(
            pi=mixture_weights,
            mu=means,
            var=variances,
        )

    def fit(
        self,
        z: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        em_iters: int = 50,
        tol: float = 1e-4,
        seed: int = 42,
    ) -> "WeightedDiagonalGMM":
        z = z.float().cpu()
        if len(z) == 0:
            raise ValueError("Cannot fit GMM with zero samples")
        weights = (
            torch.ones(len(z), dtype=torch.float32)
            if sample_weights is None
            else sample_weights.float().cpu().clamp_min(self.eps)
        )
        state = self._init_params(z, weights, seed)
        previous_likelihood = None

        for _ in range(em_iters):
            component_log_prob = self._log_diag_gaussian(z, state.mu, state.var)
            log_joint = (
                torch.log(state.pi.clamp_min(self.eps))[None, :]
                + component_log_prob
            )
            log_normalizer = torch.logsumexp(log_joint, dim=1)
            responsibilities = torch.exp(log_joint - log_normalizer[:, None])
            weighted_responsibilities = weights[:, None] * responsibilities
            effective_counts = weighted_responsibilities.sum(dim=0).clamp_min(
                self.eps
            )
            total_weight = weights.sum().clamp_min(self.eps)
            mixture_weights = effective_counts / total_weight
            means = (weighted_responsibilities.T @ z) / effective_counts[:, None]
            differences = z[:, None, :] - means[None, :, :]
            variances = (
                weighted_responsibilities[:, :, None] * differences.pow(2)
            ).sum(dim=0) / effective_counts[:, None]
            state = DiagonalGMMState(
                pi=mixture_weights,
                mu=means,
                var=variances.clamp_min(self.variance_floor),
            )
            weighted_likelihood = (weights * log_normalizer).sum() / total_weight
            if (
                previous_likelihood is not None
                and abs(float(weighted_likelihood - previous_likelihood)) < tol
            ):
                break
            previous_likelihood = weighted_likelihood.detach()

        self.state = state
        return self

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("GMM is not fitted")
        z = z.float().cpu()
        component_log_prob = self._log_diag_gaussian(
            z, self.state.mu, self.state.var
        )
        log_joint = (
            torch.log(self.state.pi.clamp_min(self.eps))[None, :]
            + component_log_prob
        )
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
    def from_dict(cls, values: Dict[str, torch.Tensor]) -> "WeightedDiagonalGMM":
        required = {"pi", "mu", "var", "n_components", "variance_floor", "eps"}
        if not isinstance(values, dict) or not required.issubset(values):
            raise ValueError("GMM checkpoint state is incomplete")
        pi = values["pi"]
        mu = values["mu"]
        var = values["var"]
        if not all(isinstance(value, torch.Tensor) for value in (pi, mu, var)):
            raise ValueError("GMM checkpoint parameters must be tensors")
        n_components = int(values["n_components"])
        variance_floor = float(values["variance_floor"])
        eps = float(values["eps"])
        if n_components <= 0 or not math.isfinite(variance_floor) or variance_floor <= 0:
            raise ValueError("GMM checkpoint hyperparameters are invalid")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("GMM checkpoint epsilon is invalid")
        if (
            pi.ndim != 1
            or mu.ndim != 2
            or var.ndim != 2
            or mu.shape != var.shape
            or len(pi) != n_components
            or mu.shape[0] != n_components
            or mu.shape[1] == 0
        ):
            raise ValueError("GMM checkpoint tensor shapes are incompatible")
        if (
            not torch.isfinite(pi).all()
            or not torch.isfinite(mu).all()
            or not torch.isfinite(var).all()
            or (pi < 0).any()
            or (var <= 0).any()
            or not torch.isclose(pi.sum(), torch.tensor(1.0), atol=1e-4, rtol=1e-4)
        ):
            raise ValueError("GMM checkpoint tensors contain invalid values")
        gmm = cls(
            n_components=n_components,
            variance_floor=variance_floor,
            eps=eps,
        )
        gmm.state = DiagonalGMMState(
            pi=pi.float(),
            mu=mu.float(),
            var=var.float(),
        )
        return gmm


@dataclass
class TaskRouter:
    task_name: str
    task_id: int
    gmm: WeightedDiagonalGMM
    calibration_mean: float
    calibration_std: float

    def calibrated_score(self, z: torch.Tensor, eps: float) -> torch.Tensor:
        return (self.gmm.log_prob(z) - self.calibration_mean) / (
            self.calibration_std + eps
        )


@dataclass(frozen=True)
class PairwiseBoundaryRecord:
    new_task_name: str
    new_task_id: int
    old_task_name: str
    old_task_id: int
    candidate_boundary: float
    stored_boundary: float
    accepted: bool
    fit_balanced_accuracy: float
    baseline_fit_balanced_accuracy: float
    fit_gain: float
    cert_observed_gain: float
    lower_confidence_bound: float
    bootstrap_replicates: int
    alpha: float
    new_fit_samples: int
    new_cert_samples: int
    old_fit_samples: int
    old_cert_samples: int


def sample_diagonal_gmm(
    gmm: WeightedDiagonalGMM,
    n_samples: int,
    seed: int,
) -> torch.Tensor:
    if gmm.state is None:
        raise RuntimeError("Cannot sample an unfitted GMM")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    state = gmm.state
    component_ids = torch.multinomial(
        state.pi.float().cpu(),
        num_samples=n_samples,
        replacement=True,
        generator=generator,
    )
    noise = torch.randn(
        n_samples,
        state.mu.shape[1],
        generator=generator,
        dtype=torch.float32,
    )
    means = state.mu.float().cpu()[component_ids]
    stds = state.var.float().cpu()[component_ids].sqrt()
    return means + noise * stds


class PBCGMMRouter:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self.tasks: List[TaskRouter] = []
        self.boundaries: Dict[Tuple[int, int], float] = {}
        self.boundary_records: List[PairwiseBoundaryRecord] = []
        self.representation_manifest: Optional[Dict[str, object]] = None
        self._validate_config()

    def _validate_config(self) -> None:
        float_controls = {
            "em_tol": self.cfg.em_tol,
            "variance_floor": self.cfg.variance_floor,
            "eps": self.cfg.eps,
            "gmm_fit_fraction": self.cfg.gmm_fit_fraction,
            "boundary_fit_fraction": self.cfg.boundary_fit_fraction,
            "bootstrap_alpha": self.cfg.bootstrap_alpha,
            "margin_threshold": self.cfg.margin_threshold,
        }
        non_finite = [
            name for name, value in float_controls.items() if not math.isfinite(value)
        ]
        if non_finite:
            raise ValueError(
                "Floating-point controls must be finite: " + ", ".join(non_finite)
            )
        if not self.cfg.model_name.strip():
            raise ValueError("model_name must be non-empty")
        if not self.cfg.cache_tag.strip():
            raise ValueError("cache_tag must be non-empty")
        if self.cfg.dataset_source not in {"codetask", "executable"}:
            raise ValueError("dataset_source must be codetask or executable")
        if not self.cfg.tasks or any(not task.strip() for task in self.cfg.tasks):
            raise ValueError("tasks must contain non-empty task names")
        if len(set(self.cfg.tasks)) != len(self.cfg.tasks):
            raise ValueError("tasks must be unique")
        if self.cfg.feature_layers < 0:
            raise ValueError("feature_layers must be non-negative")
        if self.cfg.routing_dim <= 0:
            raise ValueError("routing_dim must be positive")
        if self.cfg.max_length <= 0 or self.cfg.batch_size <= 0:
            raise ValueError("max_length and batch_size must be positive")
        if self.cfg.train_k != -1 and self.cfg.train_k < 4:
            raise ValueError("train_k must be -1 or at least 4")
        if self.cfg.eval_k != -1 and self.cfg.eval_k <= 0:
            raise ValueError("eval_k must be -1 or positive")
        if self.cfg.gmm_components <= 0:
            raise ValueError("gmm_components must be positive")
        if self.cfg.em_iters <= 0:
            raise ValueError("em_iters must be positive")
        if self.cfg.em_tol < 0.0:
            raise ValueError("em_tol must be non-negative")
        if self.cfg.variance_floor <= 0.0:
            raise ValueError("variance_floor must be positive")
        if self.cfg.eps <= 0.0:
            raise ValueError("eps must be positive")
        if not 0.0 < self.cfg.gmm_fit_fraction < 1.0:
            raise ValueError("gmm_fit_fraction must be in (0, 1)")
        if not 0.0 < self.cfg.boundary_fit_fraction < 1.0:
            raise ValueError("boundary_fit_fraction must be in (0, 1)")
        if self.cfg.gmm_fit_fraction + self.cfg.boundary_fit_fraction >= 1.0:
            raise ValueError(
                "gmm_fit_fraction + boundary_fit_fraction must be less than 1"
            )
        if self.cfg.old_pseudo_fit_samples <= 0:
            raise ValueError("old_pseudo_fit_samples must be positive")
        if self.cfg.old_pseudo_cert_samples <= 0:
            raise ValueError("old_pseudo_cert_samples must be positive")
        if self.cfg.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")
        if not 0.0 < self.cfg.bootstrap_alpha < 1.0:
            raise ValueError("bootstrap_alpha must be in (0, 1)")
        if self.cfg.margin_threshold < 0.0:
            raise ValueError("margin_threshold must be non-negative")

    def fit_new_task(
        self,
        task_name: str,
        task_id: int,
        z_train: torch.Tensor,
    ) -> TaskRouter:
        if task_id != len(self.tasks):
            raise ValueError(
                f"Tasks must be appended sequentially: expected task_id={len(self.tasks)}, "
                f"got {task_id}"
            )
        if z_train.ndim != 2 or z_train.shape[1] != self.cfg.routing_dim:
            raise ValueError(
                f"z_train must have shape [N, {self.cfg.routing_dim}], "
                f"got {tuple(z_train.shape)}"
            )
        if not torch.isfinite(z_train).all():
            raise ValueError("z_train must contain only finite features")
        z_gmm_fit, z_boundary_fit, z_boundary_cert = split_task_features(
            z_train.float().cpu(),
            gmm_fit_fraction=self.cfg.gmm_fit_fraction,
            boundary_fit_fraction=self.cfg.boundary_fit_fraction,
            seed=self.cfg.seed + 1009 * task_id,
        )
        gmm = WeightedDiagonalGMM(
            n_components=self.cfg.gmm_components,
            variance_floor=self.cfg.variance_floor,
            eps=self.cfg.eps,
        )
        gmm.fit(
            z_gmm_fit,
            em_iters=self.cfg.em_iters,
            tol=self.cfg.em_tol,
            seed=self.cfg.seed + task_id,
        )
        fit_log_prob = gmm.log_prob(z_gmm_fit)
        if not torch.isfinite(fit_log_prob).all():
            raise RuntimeError("Fitted GMM produced non-finite calibration scores")
        calibration_mean = float(fit_log_prob.mean())
        calibration_std = float(fit_log_prob.std(unbiased=False))
        if not math.isfinite(calibration_mean) or not math.isfinite(calibration_std):
            raise RuntimeError("Calibration statistics must be finite")
        task_router = TaskRouter(
            task_name=task_name,
            task_id=task_id,
            gmm=gmm,
            calibration_mean=calibration_mean,
            calibration_std=calibration_std,
        )

        pending_boundaries: Dict[Tuple[int, int], float] = {}
        pending_records: List[PairwiseBoundaryRecord] = []
        for old_task in self.tasks:
            pair_seed = (
                self.cfg.seed
                + 1_000_003 * task_id
                + 10_009 * old_task.task_id
            )
            old_boundary_fit = sample_diagonal_gmm(
                old_task.gmm,
                self.cfg.old_pseudo_fit_samples,
                seed=pair_seed + 17,
            )
            old_boundary_cert = sample_diagonal_gmm(
                old_task.gmm,
                self.cfg.old_pseudo_cert_samples,
                seed=pair_seed + 29,
            )

            fit_new_margins = task_router.calibrated_score(
                z_boundary_fit, self.cfg.eps
            ) - old_task.calibrated_score(z_boundary_fit, self.cfg.eps)
            fit_old_margins = task_router.calibrated_score(
                old_boundary_fit, self.cfg.eps
            ) - old_task.calibrated_score(old_boundary_fit, self.cfg.eps)
            fit_result = find_optimal_boundary(
                fit_new_margins,
                fit_old_margins,
            )

            cert_new_margins = task_router.calibrated_score(
                z_boundary_cert, self.cfg.eps
            ) - old_task.calibrated_score(z_boundary_cert, self.cfg.eps)
            cert_old_margins = task_router.calibrated_score(
                old_boundary_cert, self.cfg.eps
            ) - old_task.calibrated_score(old_boundary_cert, self.cfg.eps)
            certification = certify_boundary(
                cert_new_margins,
                cert_old_margins,
                candidate_boundary=fit_result.boundary,
                bootstrap_replicates=self.cfg.bootstrap_replicates,
                alpha=self.cfg.bootstrap_alpha,
                seed=pair_seed + 43,
            )
            stored_boundary = fit_result.boundary if certification.accepted else 0.0
            if certification.accepted:
                pending_boundaries[(task_id, old_task.task_id)] = stored_boundary

            record = PairwiseBoundaryRecord(
                new_task_name=task_name,
                new_task_id=task_id,
                old_task_name=old_task.task_name,
                old_task_id=old_task.task_id,
                candidate_boundary=fit_result.boundary,
                stored_boundary=stored_boundary,
                accepted=certification.accepted,
                fit_balanced_accuracy=fit_result.balanced_accuracy,
                baseline_fit_balanced_accuracy=fit_result.baseline_balanced_accuracy,
                fit_gain=fit_result.fit_gain,
                cert_observed_gain=certification.observed_gain,
                lower_confidence_bound=certification.lower_confidence_bound,
                bootstrap_replicates=self.cfg.bootstrap_replicates,
                alpha=self.cfg.bootstrap_alpha,
                new_fit_samples=len(z_boundary_fit),
                new_cert_samples=len(z_boundary_cert),
                old_fit_samples=len(old_boundary_fit),
                old_cert_samples=len(old_boundary_cert),
            )
            pending_records.append(record)
            status = "accepted" if record.accepted else "fallback-zero"
            print(
                f"[pbc] pair=({task_id}:{task_name},{old_task.task_id}:"
                f"{old_task.task_name}) b*={record.candidate_boundary:.6f} "
                f"fit_gain={record.fit_gain:.6f} cert_gain="
                f"{record.cert_observed_gain:.6f} LCB={record.lower_confidence_bound:.6f} "
                f"status={status}"
            )

        self.boundaries.update(pending_boundaries)
        self.boundary_records.extend(pending_records)
        self.tasks.append(task_router)
        print(
            f"[fit] task={task_id}:{task_name} N={len(z_train)} "
            f"gmm_fit={len(z_gmm_fit)} boundary_fit={len(z_boundary_fit)} "
            f"boundary_cert={len(z_boundary_cert)}"
        )
        return task_router

    def predict_scores(self, z: torch.Tensor) -> torch.Tensor:
        if not self.tasks:
            raise RuntimeError("No fitted tasks available")
        return torch.stack(
            [task.calibrated_score(z, self.cfg.eps) for task in self.tasks],
            dim=1,
        )

    def predict_baseline(self, z: torch.Tensor) -> torch.Tensor:
        scores = self.predict_scores(z)
        local_predictions = scores.argmax(dim=1)
        task_ids = torch.tensor(
            [task.task_id for task in self.tasks], dtype=torch.long
        )
        return task_ids[local_predictions]

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return apply_selective_pairwise_boundaries(
            scores=self.predict_scores(z),
            task_ids=[task.task_id for task in self.tasks],
            boundaries=self.boundaries,
            margin_threshold=self.cfg.margin_threshold,
        )

    def save(self, output_dir: str | Path, step: int) -> Path:
        output_dir = ensure_dir(output_dir)
        path = output_dir / f"router_step{step}.pt"
        payload = {
            "format_version": 1,
            "cfg": asdict(self.cfg),
            "step": step,
            "representation_manifest": self.representation_manifest,
            "tasks": [
                {
                    "task_name": task.task_name,
                    "task_id": task.task_id,
                    "gmm": task.gmm.to_dict(),
                    "calibration_mean": task.calibration_mean,
                    "calibration_std": task.calibration_std,
                }
                for task in self.tasks
            ],
            "boundaries": [
                {
                    "new_task_id": new_task_id,
                    "old_task_id": old_task_id,
                    "boundary": boundary,
                }
                for (new_task_id, old_task_id), boundary in sorted(
                    self.boundaries.items()
                )
            ],
            "boundary_records": [
                asdict(record) for record in self.boundary_records
            ],
        }
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PBCGMMRouter":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("Unsupported or malformed PBC-GMM checkpoint")
        for required_key in ("cfg", "tasks", "boundaries", "boundary_records"):
            if required_key not in payload:
                raise ValueError(f"Checkpoint is missing required key: {required_key}")
        known = set(RouterConfig.__dataclass_fields__)
        cfg_values = {
            key: value for key, value in payload["cfg"].items() if key in known
        }
        if "tasks" in cfg_values:
            cfg_values["tasks"] = tuple(cfg_values["tasks"])
        router = cls(RouterConfig(**cfg_values))
        task_ids = [int(item["task_id"]) for item in payload["tasks"]]
        if task_ids != list(range(len(task_ids))):
            raise ValueError("Checkpoint task IDs must be sequential from zero")
        for item in payload["tasks"]:
            calibration_mean = float(item["calibration_mean"])
            calibration_std = float(item["calibration_std"])
            if (
                not math.isfinite(calibration_mean)
                or not math.isfinite(calibration_std)
                or calibration_std < 0.0
            ):
                raise ValueError("Checkpoint calibration statistics are invalid")
            gmm = WeightedDiagonalGMM.from_dict(item["gmm"])
            if gmm.state is None or gmm.state.mu.shape[1] != router.cfg.routing_dim:
                raise ValueError("GMM checkpoint dimension does not match routing_dim")
            router.tasks.append(
                TaskRouter(
                    task_name=item["task_name"],
                    task_id=int(item["task_id"]),
                    gmm=gmm,
                    calibration_mean=calibration_mean,
                    calibration_std=calibration_std,
                )
            )
        valid_task_ids = set(task_ids)
        for item in payload["boundaries"]:
            new_task_id = int(item["new_task_id"])
            old_task_id = int(item["old_task_id"])
            boundary = float(item["boundary"])
            if (
                new_task_id not in valid_task_ids
                or old_task_id not in valid_task_ids
                or new_task_id <= old_task_id
                or not math.isfinite(boundary)
            ):
                raise ValueError("Checkpoint contains an invalid pairwise boundary")
            router.boundaries[(new_task_id, old_task_id)] = boundary
        records: List[PairwiseBoundaryRecord] = []
        seen_record_pairs = set()
        task_name_by_id = {task.task_id: task.task_name for task in router.tasks}
        for item in payload["boundary_records"]:
            try:
                record = PairwiseBoundaryRecord(**item)
            except (TypeError, ValueError) as error:
                raise ValueError("Checkpoint contains a malformed boundary record") from error
            pair = (record.new_task_id, record.old_task_id)
            numeric_values = (
                record.candidate_boundary,
                record.stored_boundary,
                record.fit_balanced_accuracy,
                record.baseline_fit_balanced_accuracy,
                record.fit_gain,
                record.cert_observed_gain,
                record.lower_confidence_bound,
                record.alpha,
            )
            if (
                pair in seen_record_pairs
                or record.new_task_id <= record.old_task_id
                or record.new_task_id not in valid_task_ids
                or record.old_task_id not in valid_task_ids
                or record.new_task_name != task_name_by_id[record.new_task_id]
                or record.old_task_name != task_name_by_id[record.old_task_id]
                or not all(math.isfinite(value) for value in numeric_values)
                or not 0.0 <= record.fit_balanced_accuracy <= 1.0
                or not 0.0 <= record.baseline_fit_balanced_accuracy <= 1.0
                or not -1.0 <= record.cert_observed_gain <= 1.0
                or not -1.0 <= record.lower_confidence_bound <= 1.0
                or not 0.0 < record.alpha < 1.0
                or record.bootstrap_replicates <= 0
                or min(
                    record.new_fit_samples,
                    record.new_cert_samples,
                    record.old_fit_samples,
                    record.old_cert_samples,
                )
                <= 0
                or record.accepted != (record.lower_confidence_bound > 0.0)
                or not math.isclose(
                    record.fit_gain,
                    record.fit_balanced_accuracy
                    - record.baseline_fit_balanced_accuracy,
                    abs_tol=1e-6,
                )
                or not math.isclose(
                    record.stored_boundary,
                    record.candidate_boundary if record.accepted else 0.0,
                    abs_tol=1e-8,
                )
            ):
                raise ValueError("Checkpoint contains an invalid boundary record")
            seen_record_pairs.add(pair)
            records.append(record)
        expected_record_pairs = {
            (new_task_id, old_task_id)
            for new_task_id in valid_task_ids
            for old_task_id in valid_task_ids
            if new_task_id > old_task_id
        }
        if seen_record_pairs != expected_record_pairs:
            raise ValueError("Checkpoint boundary record set is incomplete")
        expected_boundaries = {
            (record.new_task_id, record.old_task_id): record.stored_boundary
            for record in records
            if record.accepted
        }
        if router.boundaries != expected_boundaries:
            raise ValueError(
                "Checkpoint accepted boundaries do not match boundary records"
            )
        router.boundary_records = records
        manifest = payload.get("representation_manifest")
        if manifest is not None and not isinstance(manifest, dict):
            raise ValueError("Checkpoint representation manifest must be a mapping")
        router.representation_manifest = manifest
        return router


def apply_selective_pairwise_boundaries(
    scores: torch.Tensor,
    task_ids: list[int],
    boundaries: dict[tuple[int, int], float],
    margin_threshold: float,
) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [N, K]")
    if scores.shape[1] != len(task_ids) or len(task_ids) == 0:
        raise ValueError("task_ids must match the non-empty score columns")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be unique")
    if margin_threshold < 0.0:
        raise ValueError("margin_threshold must be non-negative")

    ids = torch.tensor(task_ids, dtype=torch.long)
    if scores.shape[1] == 1:
        return ids.repeat(scores.shape[0])

    top_values, top_local = torch.topk(scores, k=2, dim=1)
    predictions = ids[top_local[:, 0]].clone()
    low_margin_rows = torch.nonzero(
        top_values[:, 0] - top_values[:, 1] <= margin_threshold,
        as_tuple=False,
    ).flatten()
    column_by_task_id = {task_id: col for col, task_id in enumerate(task_ids)}

    for row in low_margin_rows.tolist():
        first_id = int(ids[top_local[row, 0]])
        second_id = int(ids[top_local[row, 1]])
        new_id, old_id = max(first_id, second_id), min(first_id, second_id)
        boundary = float(boundaries.get((new_id, old_id), 0.0))
        pairwise_margin = float(
            scores[row, column_by_task_id[new_id]]
            - scores[row, column_by_task_id[old_id]]
        )
        predictions[row] = new_id if pairwise_margin > boundary else old_id

    return predictions


@dataclass(frozen=True)
class BoundaryFitResult:
    boundary: float
    balanced_accuracy: float
    baseline_balanced_accuracy: float
    fit_gain: float


@dataclass(frozen=True)
class BoundaryCertification:
    observed_gain: float
    lower_confidence_bound: float
    accepted: bool


def certify_boundary(
    new_task_margins: torch.Tensor,
    old_task_margins: torch.Tensor,
    candidate_boundary: float,
    bootstrap_replicates: int,
    alpha: float,
    seed: int,
) -> BoundaryCertification:
    new_task_margins = new_task_margins.detach().flatten().float().cpu()
    old_task_margins = old_task_margins.detach().flatten().float().cpu()
    if len(new_task_margins) == 0 or len(old_task_margins) == 0:
        raise ValueError("Both certification strata must be non-empty")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not torch.isfinite(new_task_margins).all() or not torch.isfinite(
        old_task_margins
    ).all():
        raise ValueError("Certification margins must be finite")
    if not math.isfinite(candidate_boundary):
        raise ValueError("candidate_boundary must be finite")

    new_deltas = (
        (new_task_margins > candidate_boundary).float()
        - (new_task_margins > 0.0).float()
    )
    old_deltas = (
        (old_task_margins <= candidate_boundary).float()
        - (old_task_margins <= 0.0).float()
    )
    observed_gain = 0.5 * (
        float(new_deltas.mean()) + float(old_deltas.mean())
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def bootstrap_means(deltas: torch.Tensor) -> torch.Tensor:
        chunks = []
        remaining = bootstrap_replicates
        while remaining:
            chunk_size = min(256, remaining)
            indices = torch.randint(
                len(deltas),
                (chunk_size, len(deltas)),
                generator=generator,
            )
            chunks.append(deltas[indices].mean(dim=1))
            remaining -= chunk_size
        return torch.cat(chunks)

    bootstrap_gains = 0.5 * (
        bootstrap_means(new_deltas) + bootstrap_means(old_deltas)
    )
    lower_confidence_bound = float(torch.quantile(bootstrap_gains, alpha))
    return BoundaryCertification(
        observed_gain=observed_gain,
        lower_confidence_bound=lower_confidence_bound,
        accepted=lower_confidence_bound > 0.0,
    )


def find_optimal_boundary(
    new_task_margins: torch.Tensor,
    old_task_margins: torch.Tensor,
) -> BoundaryFitResult:
    new_task_margins = new_task_margins.detach().flatten().float().cpu()
    old_task_margins = old_task_margins.detach().flatten().float().cpu()
    if len(new_task_margins) == 0 or len(old_task_margins) == 0:
        raise ValueError("Both task margin samples must be non-empty")
    if not torch.isfinite(new_task_margins).all() or not torch.isfinite(
        old_task_margins
    ).all():
        raise ValueError("Task margins must be finite")

    all_margins = torch.cat([new_task_margins, old_task_margins])
    unique_values, inverse = torch.unique(
        all_margins,
        sorted=True,
        return_inverse=True,
    )
    new_counts = torch.zeros(len(unique_values), dtype=torch.float32)
    old_counts = torch.zeros(len(unique_values), dtype=torch.float32)
    new_counts.scatter_add_(
        0,
        inverse[: len(new_task_margins)],
        torch.ones(len(new_task_margins)),
    )
    old_counts.scatter_add_(
        0,
        inverse[len(new_task_margins) :],
        torch.ones(len(old_task_margins)),
    )

    span = max(float(unique_values[-1] - unique_values[0]), 1.0)
    low_boundary = unique_values[0] - span
    high_boundary = unique_values[-1] + span
    if len(unique_values) == 1:
        representative_boundaries = torch.stack([low_boundary, high_boundary])
    else:
        midpoints = (unique_values[:-1] + unique_values[1:]) * 0.5
        representative_boundaries = torch.cat(
            [low_boundary.reshape(1), midpoints, high_boundary.reshape(1)]
        )

    cumulative_new = torch.cumsum(new_counts, dim=0)
    cumulative_old = torch.cumsum(old_counts, dim=0)
    state_new_correct = torch.cat(
        [
            torch.tensor([float(len(new_task_margins))]),
            len(new_task_margins) - cumulative_new,
        ]
    )
    state_old_correct = torch.cat(
        [torch.tensor([0.0]), cumulative_old]
    )
    state_balanced_accuracy = 0.5 * (
        state_new_correct / len(new_task_margins)
        + state_old_correct / len(old_task_margins)
    )

    baseline_accuracy = 0.5 * (
        float((new_task_margins > 0.0).float().mean())
        + float((old_task_margins <= 0.0).float().mean())
    )
    boundaries = torch.cat(
        [representative_boundaries, torch.tensor([0.0])]
    ).float()
    balanced_accuracy = torch.cat(
        [state_balanced_accuracy, torch.tensor([baseline_accuracy])]
    )

    best_accuracy = balanced_accuracy.max()
    best_indices = torch.nonzero(
        torch.isclose(balanced_accuracy, best_accuracy), as_tuple=False
    ).flatten()
    best_index = best_indices[
        torch.argmin(boundaries[best_indices].abs())
    ]
    boundary = float(boundaries[best_index])
    fitted_accuracy = float(best_accuracy)
    return BoundaryFitResult(
        boundary=boundary,
        balanced_accuracy=fitted_accuracy,
        baseline_balanced_accuracy=baseline_accuracy,
        fit_gain=fitted_accuracy - baseline_accuracy,
    )


def split_task_features(
    z: torch.Tensor,
    gmm_fit_fraction: float,
    boundary_fit_fraction: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split task features into GMM-fit, boundary-fit, and boundary-cert sets."""
    if not 0.0 < gmm_fit_fraction < 1.0:
        raise ValueError("gmm_fit_fraction must be in (0, 1)")
    if not 0.0 < boundary_fit_fraction < 1.0:
        raise ValueError("boundary_fit_fraction must be in (0, 1)")
    if gmm_fit_fraction + boundary_fit_fraction >= 1.0:
        raise ValueError(
            "gmm_fit_fraction + boundary_fit_fraction must be less than 1"
        )
    if len(z) < 4:
        raise ValueError("Need at least four features to create three non-empty splits")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shuffled = z[torch.randperm(len(z), generator=generator)]

    n_gmm_fit = max(2, int(round(len(z) * gmm_fit_fraction)))
    n_boundary_fit = max(1, int(round(len(z) * boundary_fit_fraction)))
    if n_gmm_fit + n_boundary_fit >= len(z):
        n_gmm_fit = len(z) - 2
        n_boundary_fit = 1

    return (
        shuffled[:n_gmm_fit],
        shuffled[n_gmm_fit : n_gmm_fit + n_boundary_fit],
        shuffled[n_gmm_fit + n_boundary_fit :],
    )


def parse_tasks(
    raw: Optional[str],
    default: Tuple[str, ...],
) -> Tuple[str, ...]:
    if raw is None or raw.strip() == "":
        return default
    return tuple(task.strip() for task in raw.split(",") if task.strip())


def merge_resume_config(
    checkpoint_cfg: RouterConfig,
    runtime_cfg: RouterConfig,
    loaded_task_names: Tuple[str, ...],
) -> RouterConfig:
    if checkpoint_cfg.tasks[: len(loaded_task_names)] != loaded_task_names:
        raise ValueError(
            "Checkpoint task metadata is inconsistent with its fitted tasks: "
            f"configured={checkpoint_cfg.tasks[:len(loaded_task_names)]}, "
            f"fitted={loaded_task_names}"
        )
    if runtime_cfg.tasks[: len(loaded_task_names)] != loaded_task_names:
        raise ValueError(
            "Configured task order does not match the checkpoint prefix: "
            f"configured={runtime_cfg.tasks[:len(loaded_task_names)]}, "
            f"checkpoint={loaded_task_names}"
        )
    return replace(
        checkpoint_cfg,
        tasks=runtime_cfg.tasks,
        output_dir=runtime_cfg.output_dir,
        feature_cache_dir=runtime_cfg.feature_cache_dir,
        batch_size=runtime_cfg.batch_size,
        eval_k=runtime_cfg.eval_k,
        eval_split=runtime_cfg.eval_split,
        save_features=runtime_cfg.save_features,
        force_recompute_features=runtime_cfg.force_recompute_features,
    )


@dataclass
class EvalResult:
    overall_acc: float
    baseline_overall_acc: float
    per_task_acc: Dict[str, float]
    baseline_per_task_acc: Dict[str, float]
    confusion: List[List[int]]
    baseline_confusion: List[List[int]]
    correction_rate: float


def evaluate_seen_tasks(
    router: PBCGMMRouter,
    extractor,
    cfg: RouterConfig,
    seen_tasks: List[str],
    split: str,
) -> EvalResult:
    task_count = len(seen_tasks)
    confusion = torch.zeros(task_count, task_count, dtype=torch.long)
    baseline_confusion = torch.zeros(task_count, task_count, dtype=torch.long)
    correct_total = 0
    baseline_correct_total = 0
    correction_total = 0
    sample_total = 0
    per_task_acc: Dict[str, float] = {}
    baseline_per_task_acc: Dict[str, float] = {}

    for true_id, task_name in enumerate(seen_tasks):
        features = get_pbc_features(
            extractor=extractor,
            cfg=cfg,
            task=task_name,
            split=split,
            k=cfg.eval_k,
        )
        predictions = router.predict(features)
        baseline_predictions = router.predict_baseline(features)
        labels = torch.full_like(predictions, fill_value=true_id)

        task_correct = int((predictions == labels).sum())
        baseline_task_correct = int((baseline_predictions == labels).sum())
        task_total = int(labels.numel())
        correct_total += task_correct
        baseline_correct_total += baseline_task_correct
        correction_total += int((predictions != baseline_predictions).sum())
        sample_total += task_total
        per_task_acc[task_name] = task_correct / max(task_total, 1)
        baseline_per_task_acc[task_name] = (
            baseline_task_correct / max(task_total, 1)
        )

        for label, prediction, baseline_prediction in zip(
            labels.tolist(),
            predictions.tolist(),
            baseline_predictions.tolist(),
        ):
            if 0 <= prediction < task_count:
                confusion[label, prediction] += 1
            if 0 <= baseline_prediction < task_count:
                baseline_confusion[label, baseline_prediction] += 1

    return EvalResult(
        overall_acc=correct_total / max(sample_total, 1),
        baseline_overall_acc=baseline_correct_total / max(sample_total, 1),
        per_task_acc=per_task_acc,
        baseline_per_task_acc=baseline_per_task_acc,
        confusion=confusion.tolist(),
        baseline_confusion=baseline_confusion.tolist(),
        correction_rate=correction_total / max(sample_total, 1),
    )


def print_eval(step: int, seen_tasks: List[str], result: EvalResult) -> None:
    print("\n" + "=" * 90)
    print(f"[eval] step={step} seen_tasks={seen_tasks}")
    print(f"[eval] calibrated-GMM baseline acc = {result.baseline_overall_acc:.4f}")
    print(f"[eval] PBC-GMM routing acc        = {result.overall_acc:.4f}")
    print(f"[eval] prediction correction rate = {result.correction_rate:.4f}")
    for task_name in seen_tasks:
        print(
            f"  - {task_name:<18s}: PBC={result.per_task_acc[task_name]:.4f} "
            f"baseline={result.baseline_per_task_acc[task_name]:.4f}"
        )

    print("[eval] PBC confusion rows=true, cols=pred")
    header = "true\\pred" + "".join(
        f"\t{i}:{task[:8]}" for i, task in enumerate(seen_tasks)
    )
    print(header)
    for index, row in enumerate(result.confusion):
        print(
            f"{index}:{seen_tasks[index][:8]}"
            + "".join(f"\t{value}" for value in row)
        )
    print("=" * 90 + "\n")


def run(cfg: RouterConfig, resume_from: Optional[str] = None) -> None:
    from gmm import RoutingFeatureExtractor, get_device, set_seed

    if resume_from is None:
        router = PBCGMMRouter(cfg)
        start_task_id = 0
        checkpoint_path = None
    else:
        checkpoint_path = Path(resume_from)
        router = PBCGMMRouter.load(checkpoint_path)
        loaded_task_names = tuple(task.task_name for task in router.tasks)
        cfg = merge_resume_config(router.cfg, cfg, loaded_task_names)
        router.cfg = cfg
        router._validate_config()
        start_task_id = len(router.tasks)

    set_seed(cfg.seed)
    output_dir = ensure_dir(cfg.output_dir)
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2)

    device = get_device()
    print(f"[setup] device={device}")
    print(f"[setup] dataset_source={cfg.dataset_source}")
    if cfg.dataset_source == "executable":
        print(f"[setup] executable_dataset_name={cfg.executable_dataset_name}")
    print(f"[setup] tasks={list(cfg.tasks)}")
    print(
        f"[setup] splits=gmm:{cfg.gmm_fit_fraction:.3f} "
        f"boundary-fit:{cfg.boundary_fit_fraction:.3f} "
        f"boundary-cert:{1.0 - cfg.gmm_fit_fraction - cfg.boundary_fit_fraction:.3f}"
    )
    print(
        f"[setup] bootstrap B={cfg.bootstrap_replicates} "
        f"alpha={cfg.bootstrap_alpha} delta={cfg.margin_threshold}"
    )

    extractor = RoutingFeatureExtractor(
        model_name=cfg.model_name,
        feature_layers=cfg.feature_layers,
        routing_dim=cfg.routing_dim,
        device=device,
        seed=cfg.seed,
    )

    if checkpoint_path is None:
        extractor.save_projection(output_dir)
    else:
        projection_path = checkpoint_path.parent / "projection_P.pt"
        if not projection_path.exists():
            raise FileNotFoundError(
                f"Resume requires the original frozen projection: {projection_path}"
        )
        extractor.load_projection(projection_path)
        extractor.save_projection(output_dir)
        print(
            f"[resume] loaded {resume_from}; continuing from task_id={start_task_id}"
        )

    data_cfg, representation_manifest, cache_namespace = (
        prepare_feature_cache_config(cfg, extractor)
    )
    if checkpoint_path is None:
        router.representation_manifest = representation_manifest
    else:
        validate_representation_manifest(
            router.representation_manifest,
            representation_manifest,
        )
    with open(
        output_dir / "representation_manifest.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(representation_manifest, handle, indent=2, sort_keys=True)
    print(f"[cache] representation namespace={cache_namespace}")

    results_path = output_dir / "routing_results.json"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as handle:
            all_results: Dict[str, Dict] = json.load(handle)
    else:
        all_results = {}

    for task_id, task_name in enumerate(cfg.tasks):
        if task_id < start_task_id:
            print(
                f"[continual] skip task {task_id}: {task_name} "
                "(already in checkpoint)"
            )
            continue

        print("\n" + "#" * 90)
        print(f"[continual] learn task {task_id}: {task_name}")
        print("#" * 90)
        train_features = get_pbc_features(
            extractor=extractor,
            cfg=data_cfg,
            task=task_name,
            split="train",
            k=cfg.train_k,
        )
        router.fit_new_task(task_name, task_id, train_features)
        checkpoint_path = router.save(output_dir, step=task_id)
        print(f"[checkpoint] {checkpoint_path}")

        with open(
            output_dir / "boundary_calibration_results.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                [asdict(record) for record in router.boundary_records],
                handle,
                indent=2,
            )

        seen_tasks = [task.task_name for task in router.tasks]
        result = evaluate_seen_tasks(
            router=router,
            extractor=extractor,
            cfg=data_cfg,
            seen_tasks=seen_tasks,
            split=cfg.eval_split,
        )
        print_eval(task_id, seen_tasks, result)
        all_results[f"step{task_id}"] = {
            "seen_tasks": seen_tasks,
            "overall_acc": result.overall_acc,
            "baseline_overall_acc": result.baseline_overall_acc,
            "per_task_acc": result.per_task_acc,
            "baseline_per_task_acc": result.baseline_per_task_acc,
            "confusion": result.confusion,
            "baseline_confusion": result.baseline_confusion,
            "correction_rate": result.correction_rate,
            "accepted_boundaries": len(router.boundaries),
            "tested_boundaries": len(router.boundary_records),
        }
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2)

    print(f"[done] saved PBC-GMM checkpoints/results to: {output_dir}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Selective Pairwise Boundary Calibration for continual diagonal-GMM routing"
        )
    )
    parser.add_argument("--model_name", default=RouterConfig.model_name)
    parser.add_argument("--output_dir", default=RouterConfig.output_dir)
    parser.add_argument("--feature_cache_dir", default=None)
    parser.add_argument(
        "--cache_tag",
        default=RouterConfig.cache_tag,
        help=(
            "User-controlled cache version. Change it whenever mutable local model "
            "artifacts or upstream datasets change."
        ),
    )
    parser.add_argument(
        "--dataset_source",
        default=RouterConfig.dataset_source,
        choices=["codetask", "executable"],
    )
    parser.add_argument(
        "--executable_dataset_name",
        default=RouterConfig.executable_dataset_name,
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task list; executable tasks are language names.",
    )
    parser.add_argument("--feature_layers", type=int, default=RouterConfig.feature_layers)
    parser.add_argument("--routing_dim", type=int, default=RouterConfig.routing_dim)
    parser.add_argument("--max_length", type=int, default=RouterConfig.max_length)
    parser.add_argument("--batch_size", type=int, default=RouterConfig.batch_size)
    parser.add_argument("--train_k", type=int, default=RouterConfig.train_k)
    parser.add_argument("--eval_k", type=int, default=RouterConfig.eval_k)
    parser.add_argument("--seed", type=int, default=RouterConfig.seed)
    parser.add_argument(
        "--gmm_components", type=int, default=RouterConfig.gmm_components
    )
    parser.add_argument("--em_iters", type=int, default=RouterConfig.em_iters)
    parser.add_argument("--em_tol", type=float, default=RouterConfig.em_tol)
    parser.add_argument(
        "--variance_floor", type=float, default=RouterConfig.variance_floor
    )
    parser.add_argument("--eps", type=float, default=RouterConfig.eps)
    parser.add_argument(
        "--gmm_fit_fraction",
        type=float,
        default=RouterConfig.gmm_fit_fraction,
    )
    parser.add_argument(
        "--boundary_fit_fraction",
        type=float,
        default=RouterConfig.boundary_fit_fraction,
    )
    parser.add_argument(
        "--old_pseudo_fit_samples",
        type=int,
        default=RouterConfig.old_pseudo_fit_samples,
    )
    parser.add_argument(
        "--old_pseudo_cert_samples",
        type=int,
        default=RouterConfig.old_pseudo_cert_samples,
    )
    parser.add_argument(
        "--bootstrap_replicates",
        type=int,
        default=RouterConfig.bootstrap_replicates,
    )
    parser.add_argument(
        "--bootstrap_alpha",
        type=float,
        default=RouterConfig.bootstrap_alpha,
    )
    parser.add_argument(
        "--margin_threshold",
        type=float,
        default=RouterConfig.margin_threshold,
    )
    parser.add_argument(
        "--eval_split",
        default=RouterConfig.eval_split,
        choices=["validation", "test"],
    )
    parser.add_argument("--no_save_features", action="store_true")
    parser.add_argument("--force_recompute_features", action="store_true")
    parser.add_argument(
        "--resume_from",
        default=None,
        help="Path to a PBC router_stepN.pt checkpoint.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    default_tasks = (
        RouterConfig.tasks if args.dataset_source == "codetask" else ("swift",)
    )
    cfg = RouterConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        feature_cache_dir=args.feature_cache_dir,
        cache_tag=args.cache_tag,
        dataset_source=args.dataset_source,
        executable_dataset_name=args.executable_dataset_name,
        tasks=parse_tasks(args.tasks, default_tasks),
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
        eval_split=args.eval_split,
        save_features=not args.no_save_features,
        force_recompute_features=args.force_recompute_features,
        gmm_fit_fraction=args.gmm_fit_fraction,
        boundary_fit_fraction=args.boundary_fit_fraction,
        old_pseudo_fit_samples=args.old_pseudo_fit_samples,
        old_pseudo_cert_samples=args.old_pseudo_cert_samples,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_alpha=args.bootstrap_alpha,
        margin_threshold=args.margin_threshold,
    )
    run(cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
