from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from gmm_pbc import (
    WeightedDiagonalGMM,
    build_feature_cache_manifest,
    sample_diagonal_gmm,
    validate_feature_tensor,
    validate_representation_manifest,
)


@dataclass
class RouterConfig:
    model_name: str = "SalesForce/codet5-small"
    output_dir: str = "./router_gmm_mc_ckpt"
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

    # D_t = D_t^fit union D_t^opt union D_t^cert.
    gmm_fit_fraction: float = 0.70
    correction_opt_fraction: float = 0.15

    # Independent model-based historical memories, per old task.
    old_pseudo_opt_samples: int = 6000
    old_pseudo_cert_samples: int = 20000
    historical_margin: float = 0.5

    # Minimum-change QP controls.
    new_margin: float = 0.0
    old_margin: float = 0.0
    c_new: float = 1.0
    c_old: float = 1.0
    solver_max_iter: int = 1000
    solver_tolerance: float = 1e-7

    # Independent deployment certification.
    bootstrap_replicates: int = 2000
    confidence_alpha: float = 0.05
    old_disturbance_budget: float = 0.02


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def prepare_feature_cache_config(
    cfg: RouterConfig,
    extractor,
) -> Tuple[RouterConfig, Dict[str, object], Path]:
    manifest = build_feature_cache_manifest(cfg, extractor)
    cache_root = (
        Path(cfg.feature_cache_dir)
        if cfg.feature_cache_dir
        else Path(cfg.output_dir) / "mc_feature_cache"
    )
    namespace = ensure_dir(cache_root / str(manifest["fingerprint"]))
    manifest_path = namespace / "cache_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as handle:
            stored_manifest = json.load(handle)
        if stored_manifest != manifest:
            raise RuntimeError(
                f"Feature-cache manifest mismatch at {manifest_path}; "
                "use a clean cache root or change --cache_tag."
            )
    else:
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
    return replace(cfg, feature_cache_dir=str(namespace)), manifest, namespace


def get_mc_features(
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
        print(f"[cache] invalid cached features for {task}/{split}; recomputing")
        recompute_cfg = replace(cfg, force_recompute_features=True)
        features = get_or_extract_features(
            extractor=extractor,
            cfg=recompute_cfg,
            task=task,
            split=split,
            k=k,
        )
        return validate_feature_tensor(features, cfg, f"{task}/{split}")


def component_responsibilities(
    gmm: WeightedDiagonalGMM,
    z: torch.Tensor,
) -> torch.Tensor:
    """Return gamma_k(z), the posterior GMM-component responsibilities."""
    if gmm.state is None:
        raise RuntimeError("GMM is not fitted")
    z = z.detach().float().cpu()
    state = gmm.state
    component_log_prob = gmm._log_diag_gaussian(z, state.mu, state.var)
    log_joint = torch.log(state.pi.clamp_min(gmm.eps))[None, :] + component_log_prob
    responsibilities = torch.softmax(log_joint, dim=1)
    if not torch.isfinite(responsibilities).all():
        raise RuntimeError("GMM produced non-finite component responsibilities")
    return responsibilities


@dataclass
class TaskRouter:
    task_name: str
    task_id: int
    gmm: WeightedDiagonalGMM
    correction: torch.Tensor


@dataclass(frozen=True)
class QPSolution:
    correction: torch.Tensor
    objective: float
    success: bool
    message: str
    iterations: int
    projected_gradient: float
    new_slack_sum: float
    old_slack_sum: float


@dataclass(frozen=True)
class DeploymentCertification:
    baseline_new_accuracy: float
    candidate_new_accuracy: float
    observed_new_gain: float
    new_gain_lcb: float
    historical_disturbance: float
    historical_disturbance_ucb: float


@dataclass(frozen=True)
class CorrectionRecord:
    task_name: str
    task_id: int
    attempted: bool
    accepted: bool
    status: str
    candidate_correction: List[float]
    deployed_correction: List[float]
    correction_norm: float
    qp_objective: float
    solver_success: bool
    solver_message: str
    solver_iterations: int
    solver_projected_gradient: float
    new_slack_sum: float
    old_slack_sum: float
    gmm_fit_samples: int
    new_opt_samples: int
    new_cert_samples: int
    old_opt_generated: int
    old_opt_protected: int
    old_cert_samples: int
    baseline_new_accuracy: float
    candidate_new_accuracy: float
    observed_new_gain: float
    new_gain_lcb: float
    historical_disturbance: float
    historical_disturbance_ucb: float
    old_disturbance_budget: float
    bootstrap_replicates: int
    confidence_alpha: float


def split_task_features(
    z: torch.Tensor,
    gmm_fit_fraction: float,
    correction_opt_fraction: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic, disjoint GMM-fit, correction-opt, and cert splits."""
    if not 0.0 < gmm_fit_fraction < 1.0:
        raise ValueError("gmm_fit_fraction must be in (0, 1)")
    if not 0.0 < correction_opt_fraction < 1.0:
        raise ValueError("correction_opt_fraction must be in (0, 1)")
    if gmm_fit_fraction + correction_opt_fraction >= 1.0:
        raise ValueError(
            "gmm_fit_fraction + correction_opt_fraction must be less than 1"
        )
    if len(z) < 4:
        raise ValueError("Need at least four features to create three non-empty splits")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shuffled = z.detach().float().cpu()[
        torch.randperm(len(z), generator=generator)
    ]
    n_gmm = max(2, int(round(len(z) * gmm_fit_fraction)))
    n_opt = max(1, int(round(len(z) * correction_opt_fraction)))
    if n_gmm + n_opt >= len(z):
        n_gmm = len(z) - 2
        n_opt = 1
    return (
        shuffled[:n_gmm],
        shuffled[n_gmm : n_gmm + n_opt],
        shuffled[n_gmm + n_opt :],
    )


def solve_minimum_change_qp(
    new_responsibilities: torch.Tensor,
    new_deficits: torch.Tensor,
    old_responsibilities: torch.Tensor,
    old_upper_bounds: torch.Tensor,
    c_new: float,
    c_old: float,
    max_iter: int,
    tolerance: float,
) -> QPSolution:
    """Solve the exact slack-QP through its smooth box-constrained dual."""
    import numpy as np
    from scipy.optimize import minimize

    new_gamma = new_responsibilities.detach().double().cpu()
    old_gamma = old_responsibilities.detach().double().cpu()
    new_deficits = new_deficits.detach().flatten().double().cpu()
    old_upper_bounds = old_upper_bounds.detach().flatten().double().cpu()
    if new_gamma.ndim != 2 or len(new_gamma) == 0:
        raise ValueError("new_responsibilities must be a non-empty matrix")
    component_count = new_gamma.shape[1]
    if old_gamma.ndim != 2 or old_gamma.shape[1] != component_count:
        raise ValueError("old_responsibilities must have the same component width")
    if len(new_deficits) != len(new_gamma) or len(old_upper_bounds) != len(old_gamma):
        raise ValueError("QP constraints and bounds have incompatible lengths")
    if not all(
        torch.isfinite(value).all()
        for value in (new_gamma, old_gamma, new_deficits, old_upper_bounds)
    ):
        raise ValueError("QP inputs must be finite")
    if c_new <= 0.0 or c_old <= 0.0:
        raise ValueError("c_new and c_old must be positive")
    if max_iter <= 0 or tolerance <= 0.0:
        raise ValueError("Solver controls must be positive")

    # A u >= c - xi. Old upper constraints are negated into this form.
    constraint_matrix = torch.cat([new_gamma, -old_gamma], dim=0).numpy()
    rhs = torch.cat([new_deficits, -old_upper_bounds], dim=0).numpy()
    box = np.concatenate(
        [
            np.full(len(new_gamma), c_new, dtype=np.float64),
            np.full(len(old_gamma), c_old, dtype=np.float64),
        ]
    )

    def objective(dual):
        correction = constraint_matrix.T @ dual
        value = 0.5 * float(correction @ correction) - float(rhs @ dual)
        gradient = constraint_matrix @ correction - rhs
        return value, gradient

    result = minimize(
        objective,
        np.zeros(len(rhs), dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, float(limit)) for limit in box],
        options={
            "maxiter": max_iter,
            "ftol": tolerance,
            "gtol": tolerance,
            "maxls": 50,
        },
    )
    dual = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, box)
    correction_np = constraint_matrix.T @ dual
    _, gradient = objective(dual)
    projected = gradient.copy()
    projected[(dual <= tolerance) & (gradient > 0.0)] = 0.0
    projected[(dual >= box - tolerance) & (gradient < 0.0)] = 0.0
    projected_gradient = float(np.max(np.abs(projected))) if len(projected) else 0.0
    success = bool(result.success) or projected_gradient <= max(1e-5, 10 * tolerance)

    correction = torch.from_numpy(correction_np).float()
    new_slack = torch.relu(new_deficits.float() - new_gamma.float() @ correction)
    old_slack = torch.relu(old_gamma.float() @ correction - old_upper_bounds.float())
    primal_objective = (
        0.5 * float(correction.square().sum())
        + c_new * float(new_slack.sum())
        + c_old * float(old_slack.sum())
    )
    if not torch.isfinite(correction).all() or not math.isfinite(primal_objective):
        raise RuntimeError("QP solver produced a non-finite correction")
    return QPSolution(
        correction=correction,
        objective=primal_objective,
        success=success,
        message=str(result.message),
        iterations=int(result.nit),
        projected_gradient=projected_gradient,
        new_slack_sum=float(new_slack.sum()),
        old_slack_sum=float(old_slack.sum()),
    )


def paired_bootstrap_gain_lcb(
    baseline_correct: torch.Tensor,
    candidate_correct: torch.Tensor,
    bootstrap_replicates: int,
    alpha: float,
    seed: int,
) -> Tuple[float, float]:
    baseline = baseline_correct.detach().flatten().float().cpu()
    candidate = candidate_correct.detach().flatten().float().cpu()
    if len(baseline) == 0 or len(candidate) != len(baseline):
        raise ValueError("Paired certification vectors must have the same non-zero length")
    if bootstrap_replicates <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("Invalid bootstrap controls")
    deltas = candidate - baseline
    observed = float(deltas.mean())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bootstrap_means = []
    remaining = bootstrap_replicates
    while remaining:
        chunk_size = min(256, remaining)
        indices = torch.randint(
            len(deltas),
            (chunk_size, len(deltas)),
            generator=generator,
        )
        bootstrap_means.append(deltas[indices].mean(dim=1))
        remaining -= chunk_size
    lcb = float(torch.quantile(torch.cat(bootstrap_means), alpha))
    return observed, lcb


def hoeffding_upper_bound(observations: torch.Tensor, alpha: float) -> float:
    values = observations.detach().flatten().float().cpu()
    if len(values) == 0:
        raise ValueError("Historical certification observations must be non-empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Historical disturbance observations must lie in [0, 1]")
    radius = math.sqrt(math.log(1.0 / alpha) / (2.0 * len(values)))
    return min(1.0, float(values.mean()) + radius)


def certify_candidate(
    baseline_new_correct: torch.Tensor,
    candidate_new_correct: torch.Tensor,
    historical_disturbance: torch.Tensor,
    bootstrap_replicates: int,
    alpha: float,
    seed: int,
) -> DeploymentCertification:
    observed_gain, lcb = paired_bootstrap_gain_lcb(
        baseline_new_correct,
        candidate_new_correct,
        bootstrap_replicates=bootstrap_replicates,
        alpha=alpha,
        seed=seed,
    )
    historical = historical_disturbance.detach().flatten().float().cpu()
    return DeploymentCertification(
        baseline_new_accuracy=float(baseline_new_correct.float().mean()),
        candidate_new_accuracy=float(candidate_new_correct.float().mean()),
        observed_new_gain=observed_gain,
        new_gain_lcb=lcb,
        historical_disturbance=float(historical.mean()),
        historical_disturbance_ucb=hoeffding_upper_bound(historical, alpha),
    )


class MinimumChangeGMMRouter:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self.tasks: List[TaskRouter] = []
        self.correction_records: List[CorrectionRecord] = []
        self.representation_manifest: Optional[Dict[str, object]] = None
        self._validate_config()

    def _validate_config(self) -> None:
        finite_controls = {
            "em_tol": self.cfg.em_tol,
            "variance_floor": self.cfg.variance_floor,
            "eps": self.cfg.eps,
            "gmm_fit_fraction": self.cfg.gmm_fit_fraction,
            "correction_opt_fraction": self.cfg.correction_opt_fraction,
            "historical_margin": self.cfg.historical_margin,
            "new_margin": self.cfg.new_margin,
            "old_margin": self.cfg.old_margin,
            "c_new": self.cfg.c_new,
            "c_old": self.cfg.c_old,
            "solver_tolerance": self.cfg.solver_tolerance,
            "confidence_alpha": self.cfg.confidence_alpha,
            "old_disturbance_budget": self.cfg.old_disturbance_budget,
        }
        bad = [name for name, value in finite_controls.items() if not math.isfinite(value)]
        if bad:
            raise ValueError("Floating-point controls must be finite: " + ", ".join(bad))
        if not self.cfg.model_name.strip() or not self.cfg.cache_tag.strip():
            raise ValueError("model_name and cache_tag must be non-empty")
        if self.cfg.dataset_source not in {"codetask", "executable"}:
            raise ValueError("dataset_source must be codetask or executable")
        if not self.cfg.tasks or any(not task.strip() for task in self.cfg.tasks):
            raise ValueError("tasks must contain non-empty task names")
        if len(set(self.cfg.tasks)) != len(self.cfg.tasks):
            raise ValueError("tasks must be unique")
        if self.cfg.feature_layers < 0 or self.cfg.routing_dim <= 0:
            raise ValueError("feature_layers/routing_dim are invalid")
        if self.cfg.max_length <= 0 or self.cfg.batch_size <= 0:
            raise ValueError("max_length and batch_size must be positive")
        if self.cfg.train_k != -1 and self.cfg.train_k < 4:
            raise ValueError("train_k must be -1 or at least 4")
        if self.cfg.eval_k != -1 and self.cfg.eval_k <= 0:
            raise ValueError("eval_k must be -1 or positive")
        if self.cfg.gmm_components <= 0 or self.cfg.em_iters <= 0:
            raise ValueError("GMM component/iteration counts must be positive")
        if self.cfg.em_tol < 0.0 or self.cfg.variance_floor <= 0.0 or self.cfg.eps <= 0.0:
            raise ValueError("GMM numerical controls are invalid")
        if not 0.0 < self.cfg.gmm_fit_fraction < 1.0:
            raise ValueError("gmm_fit_fraction must be in (0, 1)")
        if not 0.0 < self.cfg.correction_opt_fraction < 1.0:
            raise ValueError("correction_opt_fraction must be in (0, 1)")
        if self.cfg.gmm_fit_fraction + self.cfg.correction_opt_fraction >= 1.0:
            raise ValueError("GMM-fit and correction-opt fractions must sum to less than 1")
        if self.cfg.old_pseudo_opt_samples <= 0 or self.cfg.old_pseudo_cert_samples <= 0:
            raise ValueError("Historical pseudo-sample counts must be positive")
        if self.cfg.historical_margin < 0.0 or self.cfg.new_margin < 0.0:
            raise ValueError("historical_margin and new_margin must be non-negative")
        if self.cfg.old_margin < 0.0 or self.cfg.c_new <= 0.0 or self.cfg.c_old <= 0.0:
            raise ValueError("old_margin and QP penalties are invalid")
        if self.cfg.solver_max_iter <= 0 or self.cfg.solver_tolerance <= 0.0:
            raise ValueError("Solver controls must be positive")
        if self.cfg.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")
        if not 0.0 < self.cfg.confidence_alpha < 1.0:
            raise ValueError("confidence_alpha must be in (0, 1)")
        if not 0.0 <= self.cfg.old_disturbance_budget <= 1.0:
            raise ValueError("old_disturbance_budget must be in [0, 1]")

    @staticmethod
    def _task_score(task: TaskRouter, z: torch.Tensor) -> torch.Tensor:
        gamma = component_responsibilities(task.gmm, z)
        return task.gmm.log_prob(z) + gamma @ task.correction

    def predict_scores(self, z: torch.Tensor) -> torch.Tensor:
        if not self.tasks:
            raise RuntimeError("No fitted tasks available")
        return torch.stack([self._task_score(task, z) for task in self.tasks], dim=1)

    def predict_baseline_scores(self, z: torch.Tensor) -> torch.Tensor:
        if not self.tasks:
            raise RuntimeError("No fitted tasks available")
        return torch.stack([task.gmm.log_prob(z) for task in self.tasks], dim=1)

    def _predictions_from_scores(self, scores: torch.Tensor) -> torch.Tensor:
        local = scores.argmax(dim=1)
        task_ids = torch.tensor([task.task_id for task in self.tasks], dtype=torch.long)
        return task_ids[local]

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self._predictions_from_scores(self.predict_scores(z))

    def predict_baseline(self, z: torch.Tensor) -> torch.Tensor:
        return self._predictions_from_scores(self.predict_baseline_scores(z))

    def _draw_historical_probes(self, samples_per_task: int, seed: int) -> torch.Tensor:
        return torch.cat(
            [
                sample_diagonal_gmm(
                    task.gmm,
                    samples_per_task,
                    seed=seed + 10_007 * task.task_id,
                )
                for task in self.tasks
            ],
            dim=0,
        )

    def fit_new_task(
        self,
        task_name: str,
        task_id: int,
        z_train: torch.Tensor,
    ) -> TaskRouter:
        if task_id != len(self.tasks):
            raise ValueError(
                f"Tasks must be appended sequentially: expected {len(self.tasks)}, got {task_id}"
            )
        if z_train.ndim != 2 or z_train.shape[1] != self.cfg.routing_dim:
            raise ValueError(
                f"z_train must have shape [N, {self.cfg.routing_dim}], got {tuple(z_train.shape)}"
            )
        if not torch.isfinite(z_train).all():
            raise ValueError("z_train must contain only finite features")

        z_gmm, z_opt, z_cert = split_task_features(
            z_train,
            gmm_fit_fraction=self.cfg.gmm_fit_fraction,
            correction_opt_fraction=self.cfg.correction_opt_fraction,
            seed=self.cfg.seed + 1009 * task_id,
        )
        gmm = WeightedDiagonalGMM(
            n_components=self.cfg.gmm_components,
            variance_floor=self.cfg.variance_floor,
            eps=self.cfg.eps,
        )
        gmm.fit(
            z_gmm,
            em_iters=self.cfg.em_iters,
            tol=self.cfg.em_tol,
            seed=self.cfg.seed + task_id,
        )
        if not torch.isfinite(gmm.log_prob(z_gmm)).all():
            raise RuntimeError("Fitted GMM produced non-finite log probabilities")
        component_count = gmm.n_components
        zero = torch.zeros(component_count, dtype=torch.float32)
        new_task = TaskRouter(task_name, task_id, gmm, zero.clone())

        if not self.tasks:
            self.tasks.append(new_task)
            self.correction_records.append(
                CorrectionRecord(
                    task_name=task_name,
                    task_id=task_id,
                    attempted=False,
                    accepted=False,
                    status="initial-zero",
                    candidate_correction=zero.tolist(),
                    deployed_correction=zero.tolist(),
                    correction_norm=0.0,
                    qp_objective=0.0,
                    solver_success=True,
                    solver_message="first task uses zero correction",
                    solver_iterations=0,
                    solver_projected_gradient=0.0,
                    new_slack_sum=0.0,
                    old_slack_sum=0.0,
                    gmm_fit_samples=len(z_gmm),
                    new_opt_samples=len(z_opt),
                    new_cert_samples=len(z_cert),
                    old_opt_generated=0,
                    old_opt_protected=0,
                    old_cert_samples=0,
                    baseline_new_accuracy=1.0,
                    candidate_new_accuracy=1.0,
                    observed_new_gain=0.0,
                    new_gain_lcb=0.0,
                    historical_disturbance=0.0,
                    historical_disturbance_ucb=0.0,
                    old_disturbance_budget=self.cfg.old_disturbance_budget,
                    bootstrap_replicates=self.cfg.bootstrap_replicates,
                    confidence_alpha=self.cfg.confidence_alpha,
                )
            )
            print(
                f"[fit] task={task_id}:{task_name} N={len(z_train)} "
                f"gmm={len(z_gmm)} opt={len(z_opt)} cert={len(z_cert)} "
                "correction=initial-zero"
            )
            return new_task

        base_seed = self.cfg.seed + 1_000_003 * task_id
        historical_opt = self._draw_historical_probes(
            self.cfg.old_pseudo_opt_samples,
            seed=base_seed + 17,
        )
        historical_scores = self.predict_scores(historical_opt)
        historical_winners = historical_scores.argmax(dim=1)
        if historical_scores.shape[1] == 1:
            protected_mask = torch.ones(len(historical_opt), dtype=torch.bool)
        else:
            top_two = torch.topk(historical_scores, k=2, dim=1).values
            protected_mask = (
                top_two[:, 0] - top_two[:, 1] >= self.cfg.historical_margin
            )
        protected = historical_opt[protected_mask]
        protected_winners = historical_winners[protected_mask]
        protected_winner_scores = historical_scores[
            protected_mask, protected_winners
        ]

        new_opt_gamma = component_responsibilities(gmm, z_opt)
        historical_envelope = self.predict_scores(z_opt).max(dim=1).values
        new_deficits = historical_envelope - gmm.log_prob(z_opt) + self.cfg.new_margin
        if len(protected):
            old_gamma = component_responsibilities(gmm, protected)
            old_upper = (
                protected_winner_scores
                - gmm.log_prob(protected)
                - self.cfg.old_margin
            )
        else:
            old_gamma = torch.empty((0, component_count), dtype=torch.float32)
            old_upper = torch.empty(0, dtype=torch.float32)

        solution = solve_minimum_change_qp(
            new_responsibilities=new_opt_gamma,
            new_deficits=new_deficits,
            old_responsibilities=old_gamma,
            old_upper_bounds=old_upper,
            c_new=self.cfg.c_new,
            c_old=self.cfg.c_old,
            max_iter=self.cfg.solver_max_iter,
            tolerance=self.cfg.solver_tolerance,
        )
        candidate = solution.correction

        old_scores_on_new_cert = self.predict_scores(z_cert)
        new_raw_on_cert = gmm.log_prob(z_cert)
        new_candidate_on_cert = (
            new_raw_on_cert + component_responsibilities(gmm, z_cert) @ candidate
        )
        baseline_new_correct = (
            torch.cat([old_scores_on_new_cert, new_raw_on_cert[:, None]], dim=1)
            .argmax(dim=1)
            .eq(len(self.tasks))
        )
        candidate_new_correct = (
            torch.cat(
                [old_scores_on_new_cert, new_candidate_on_cert[:, None]], dim=1
            )
            .argmax(dim=1)
            .eq(len(self.tasks))
        )

        historical_cert = self._draw_historical_probes(
            self.cfg.old_pseudo_cert_samples,
            seed=base_seed + 29,
        )
        old_cert_scores = self.predict_scores(historical_cert)
        historical_cert_winners = old_cert_scores.argmax(dim=1)
        new_candidate_on_old = (
            gmm.log_prob(historical_cert)
            + component_responsibilities(gmm, historical_cert) @ candidate
        )
        candidate_old_winners = torch.cat(
            [old_cert_scores, new_candidate_on_old[:, None]], dim=1
        ).argmax(dim=1)
        disturbance = candidate_old_winners.ne(historical_cert_winners)
        certification = certify_candidate(
            baseline_new_correct=baseline_new_correct,
            candidate_new_correct=candidate_new_correct,
            historical_disturbance=disturbance,
            bootstrap_replicates=self.cfg.bootstrap_replicates,
            alpha=self.cfg.confidence_alpha,
            seed=base_seed + 43,
        )
        accepted = (
            solution.success
            and certification.new_gain_lcb > 0.0
            and certification.historical_disturbance_ucb
            <= self.cfg.old_disturbance_budget
        )
        deployed = candidate if accepted else zero
        new_task.correction = deployed.clone()
        self.tasks.append(new_task)

        if not solution.success:
            status = "fallback-solver"
        elif certification.new_gain_lcb <= 0.0:
            status = "fallback-new-lcb"
        elif certification.historical_disturbance_ucb > self.cfg.old_disturbance_budget:
            status = "fallback-old-ucb"
        else:
            status = "accepted"
        record = CorrectionRecord(
            task_name=task_name,
            task_id=task_id,
            attempted=True,
            accepted=accepted,
            status=status,
            candidate_correction=candidate.tolist(),
            deployed_correction=deployed.tolist(),
            correction_norm=float(candidate.norm()),
            qp_objective=solution.objective,
            solver_success=solution.success,
            solver_message=solution.message,
            solver_iterations=solution.iterations,
            solver_projected_gradient=solution.projected_gradient,
            new_slack_sum=solution.new_slack_sum,
            old_slack_sum=solution.old_slack_sum,
            gmm_fit_samples=len(z_gmm),
            new_opt_samples=len(z_opt),
            new_cert_samples=len(z_cert),
            old_opt_generated=len(historical_opt),
            old_opt_protected=len(protected),
            old_cert_samples=len(historical_cert),
            baseline_new_accuracy=certification.baseline_new_accuracy,
            candidate_new_accuracy=certification.candidate_new_accuracy,
            observed_new_gain=certification.observed_new_gain,
            new_gain_lcb=certification.new_gain_lcb,
            historical_disturbance=certification.historical_disturbance,
            historical_disturbance_ucb=certification.historical_disturbance_ucb,
            old_disturbance_budget=self.cfg.old_disturbance_budget,
            bootstrap_replicates=self.cfg.bootstrap_replicates,
            confidence_alpha=self.cfg.confidence_alpha,
        )
        self.correction_records.append(record)
        print(
            f"[mc] task={task_id}:{task_name} ||u*||={record.correction_norm:.6f} "
            f"gain={record.observed_new_gain:.6f} LCB={record.new_gain_lcb:.6f} "
            f"old_disturbance={record.historical_disturbance:.6f} "
            f"UCB={record.historical_disturbance_ucb:.6f} status={status}"
        )
        print(
            f"[fit] task={task_id}:{task_name} N={len(z_train)} "
            f"gmm={len(z_gmm)} opt={len(z_opt)} cert={len(z_cert)} "
            f"protected={len(protected)}/{len(historical_opt)}"
        )
        return new_task

    def save(self, output_dir: str | Path, step: int) -> Path:
        output_dir = ensure_dir(output_dir)
        path = output_dir / f"router_step{step}.pt"
        payload = {
            "format_version": 1,
            "method": "minimum_change_historical_envelope",
            "cfg": asdict(self.cfg),
            "step": step,
            "representation_manifest": self.representation_manifest,
            "tasks": [
                {
                    "task_name": task.task_name,
                    "task_id": task.task_id,
                    "gmm": task.gmm.to_dict(),
                    "correction": task.correction,
                }
                for task in self.tasks
            ],
            "correction_records": [
                asdict(record) for record in self.correction_records
            ],
        }
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MinimumChangeGMMRouter":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, dict)
            or payload.get("format_version") != 1
            or payload.get("method") != "minimum_change_historical_envelope"
        ):
            raise ValueError("Unsupported or malformed MC-GMM checkpoint")
        for key in ("cfg", "tasks", "correction_records"):
            if key not in payload:
                raise ValueError(f"Checkpoint is missing required key: {key}")
        known = set(RouterConfig.__dataclass_fields__)
        cfg_values = {key: value for key, value in payload["cfg"].items() if key in known}
        if "tasks" in cfg_values:
            cfg_values["tasks"] = tuple(cfg_values["tasks"])
        router = cls(RouterConfig(**cfg_values))

        for expected_id, item in enumerate(payload["tasks"]):
            gmm = WeightedDiagonalGMM.from_dict(item["gmm"])
            correction = item.get("correction")
            if not isinstance(correction, torch.Tensor):
                raise ValueError("Checkpoint correction must be a tensor")
            correction = correction.detach().flatten().float().cpu()
            if (
                int(item["task_id"]) != expected_id
                or gmm.state is None
                or gmm.state.mu.shape[1] != router.cfg.routing_dim
                or len(correction) != gmm.n_components
                or not torch.isfinite(correction).all()
            ):
                raise ValueError("Checkpoint contains invalid task/correction state")
            router.tasks.append(
                TaskRouter(
                    task_name=str(item["task_name"]),
                    task_id=expected_id,
                    gmm=gmm,
                    correction=correction,
                )
            )
        if len({task.task_name for task in router.tasks}) != len(router.tasks):
            raise ValueError("Checkpoint task names must be unique")

        try:
            records = [CorrectionRecord(**item) for item in payload["correction_records"]]
        except (TypeError, ValueError) as error:
            raise ValueError("Checkpoint contains malformed correction records") from error
        if len(records) != len(router.tasks):
            raise ValueError("Checkpoint correction record set is incomplete")
        for task, record in zip(router.tasks, records):
            deployed = torch.tensor(record.deployed_correction, dtype=torch.float32)
            if (
                record.task_id != task.task_id
                or record.task_name != task.task_name
                or len(deployed) != len(task.correction)
                or not torch.isfinite(deployed).all()
                or not torch.allclose(deployed, task.correction)
            ):
                raise ValueError("Checkpoint correction record is inconsistent")
        router.correction_records = records
        manifest = payload.get("representation_manifest")
        if manifest is not None and not isinstance(manifest, dict):
            raise ValueError("Checkpoint representation manifest must be a mapping")
        router.representation_manifest = manifest
        return router


def parse_tasks(raw: Optional[str], default: Tuple[str, ...]) -> Tuple[str, ...]:
    if raw is None or raw.strip() == "":
        return default
    return tuple(task.strip() for task in raw.split(",") if task.strip())


def merge_resume_config(
    checkpoint_cfg: RouterConfig,
    runtime_cfg: RouterConfig,
    loaded_task_names: Tuple[str, ...],
) -> RouterConfig:
    if checkpoint_cfg.tasks[: len(loaded_task_names)] != loaded_task_names:
        raise ValueError("Checkpoint task metadata is inconsistent with fitted tasks")
    if runtime_cfg.tasks[: len(loaded_task_names)] != loaded_task_names:
        raise ValueError("Configured task order does not match the checkpoint prefix")
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
    router: MinimumChangeGMMRouter,
    extractor,
    cfg: RouterConfig,
    seen_tasks: List[str],
    split: str,
) -> EvalResult:
    task_count = len(seen_tasks)
    confusion = torch.zeros(task_count, task_count, dtype=torch.long)
    baseline_confusion = torch.zeros_like(confusion)
    correct_total = baseline_correct_total = correction_total = sample_total = 0
    per_task_acc: Dict[str, float] = {}
    baseline_per_task_acc: Dict[str, float] = {}
    for true_id, task_name in enumerate(seen_tasks):
        features = get_mc_features(extractor, cfg, task_name, split, cfg.eval_k)
        predictions = router.predict(features)
        baseline_predictions = router.predict_baseline(features)
        labels = torch.full_like(predictions, true_id)
        total = int(labels.numel())
        correct = int((predictions == labels).sum())
        baseline_correct = int((baseline_predictions == labels).sum())
        correct_total += correct
        baseline_correct_total += baseline_correct
        correction_total += int((predictions != baseline_predictions).sum())
        sample_total += total
        per_task_acc[task_name] = correct / max(total, 1)
        baseline_per_task_acc[task_name] = baseline_correct / max(total, 1)
        for label, prediction, baseline_prediction in zip(
            labels.tolist(), predictions.tolist(), baseline_predictions.tolist()
        ):
            confusion[label, prediction] += 1
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
    print(f"[eval] raw GMM baseline acc       = {result.baseline_overall_acc:.4f}")
    print(f"[eval] MC-GMM routing acc         = {result.overall_acc:.4f}")
    print(f"[eval] prediction correction rate= {result.correction_rate:.4f}")
    for task_name in seen_tasks:
        print(
            f"  - {task_name:<18s}: MC={result.per_task_acc[task_name]:.4f} "
            f"baseline={result.baseline_per_task_acc[task_name]:.4f}"
        )
    print("[eval] MC confusion rows=true, cols=pred")
    print("true\\pred" + "".join(f"\t{i}:{task[:8]}" for i, task in enumerate(seen_tasks)))
    for index, row in enumerate(result.confusion):
        print(f"{index}:{seen_tasks[index][:8]}" + "".join(f"\t{x}" for x in row))
    print("=" * 90 + "\n")


def run(cfg: RouterConfig, resume_from: Optional[str] = None) -> None:
    from gmm import RoutingFeatureExtractor, get_device, set_seed

    if resume_from is None:
        router = MinimumChangeGMMRouter(cfg)
        start_task_id = 0
        checkpoint_path = None
    else:
        checkpoint_path = Path(resume_from)
        router = MinimumChangeGMMRouter.load(checkpoint_path)
        loaded_names = tuple(task.task_name for task in router.tasks)
        cfg = merge_resume_config(router.cfg, cfg, loaded_names)
        router.cfg = cfg
        router._validate_config()
        start_task_id = len(router.tasks)

    set_seed(cfg.seed)
    output_dir = ensure_dir(cfg.output_dir)
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2)
    device = get_device()
    print(f"[setup] device={device} dataset_source={cfg.dataset_source}")
    if cfg.dataset_source == "executable":
        print(f"[setup] executable_dataset_name={cfg.executable_dataset_name}")
    print(f"[setup] tasks={list(cfg.tasks)}")
    print(
        f"[setup] splits=gmm:{cfg.gmm_fit_fraction:.3f} "
        f"opt:{cfg.correction_opt_fraction:.3f} "
        f"cert:{1.0 - cfg.gmm_fit_fraction - cfg.correction_opt_fraction:.3f}"
    )
    print(
        f"[setup] QP C_new={cfg.c_new} C_old={cfg.c_old} "
        f"eta={cfg.new_margin} kappa={cfg.old_margin} tau={cfg.historical_margin}"
    )
    print(
        f"[setup] certification B={cfg.bootstrap_replicates} "
        f"alpha={cfg.confidence_alpha} old_budget={cfg.old_disturbance_budget}"
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
        print(f"[resume] loaded {resume_from}; continuing from task_id={start_task_id}")

    data_cfg, manifest, cache_namespace = prepare_feature_cache_config(cfg, extractor)
    if checkpoint_path is None:
        router.representation_manifest = manifest
    else:
        validate_representation_manifest(router.representation_manifest, manifest)
    with open(
        output_dir / "representation_manifest.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"[cache] representation namespace={cache_namespace}")

    results_path = output_dir / "routing_results.json"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as handle:
            all_results: Dict[str, Dict] = json.load(handle)
    else:
        all_results = {}

    for task_id, task_name in enumerate(cfg.tasks):
        if task_id < start_task_id:
            print(f"[continual] skip task {task_id}:{task_name} (in checkpoint)")
            continue
        print("\n" + "#" * 90)
        print(f"[continual] learn task {task_id}: {task_name}")
        print("#" * 90)
        train_features = get_mc_features(
            extractor, data_cfg, task_name, "train", cfg.train_k
        )
        router.fit_new_task(task_name, task_id, train_features)
        saved = router.save(output_dir, step=task_id)
        print(f"[checkpoint] {saved}")
        with open(
            output_dir / "minimum_change_results.json", "w", encoding="utf-8"
        ) as handle:
            json.dump(
                [asdict(record) for record in router.correction_records],
                handle,
                indent=2,
            )

        seen_tasks = [task.task_name for task in router.tasks]
        result = evaluate_seen_tasks(
            router, extractor, data_cfg, seen_tasks, cfg.eval_split
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
            "accepted_corrections": sum(
                record.accepted for record in router.correction_records
            ),
            "tested_corrections": sum(
                record.attempted for record in router.correction_records
            ),
        }
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2)
    print(f"[done] saved MC-GMM checkpoints/results to: {output_dir}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimum-Change Historical Envelope Preservation for GMM routing"
    )
    parser.add_argument("--model_name", default=RouterConfig.model_name)
    parser.add_argument("--output_dir", default=RouterConfig.output_dir)
    parser.add_argument("--feature_cache_dir", default=None)
    parser.add_argument("--cache_tag", default=RouterConfig.cache_tag)
    parser.add_argument(
        "--dataset_source",
        default=RouterConfig.dataset_source,
        choices=["codetask", "executable"],
    )
    parser.add_argument(
        "--executable_dataset_name", default=RouterConfig.executable_dataset_name
    )
    parser.add_argument("--tasks", default=None, help="Comma-separated task list")
    parser.add_argument("--feature_layers", type=int, default=RouterConfig.feature_layers)
    parser.add_argument("--routing_dim", type=int, default=RouterConfig.routing_dim)
    parser.add_argument("--max_length", type=int, default=RouterConfig.max_length)
    parser.add_argument("--batch_size", type=int, default=RouterConfig.batch_size)
    parser.add_argument("--train_k", type=int, default=RouterConfig.train_k)
    parser.add_argument("--eval_k", type=int, default=RouterConfig.eval_k)
    parser.add_argument("--seed", type=int, default=RouterConfig.seed)
    parser.add_argument("--gmm_components", type=int, default=RouterConfig.gmm_components)
    parser.add_argument("--em_iters", type=int, default=RouterConfig.em_iters)
    parser.add_argument("--em_tol", type=float, default=RouterConfig.em_tol)
    parser.add_argument("--variance_floor", type=float, default=RouterConfig.variance_floor)
    parser.add_argument("--eps", type=float, default=RouterConfig.eps)
    parser.add_argument("--gmm_fit_fraction", type=float, default=RouterConfig.gmm_fit_fraction)
    parser.add_argument(
        "--correction_opt_fraction",
        type=float,
        default=RouterConfig.correction_opt_fraction,
    )
    parser.add_argument(
        "--old_pseudo_opt_samples",
        type=int,
        default=RouterConfig.old_pseudo_opt_samples,
    )
    parser.add_argument(
        "--old_pseudo_cert_samples",
        type=int,
        default=RouterConfig.old_pseudo_cert_samples,
    )
    parser.add_argument("--historical_margin", type=float, default=RouterConfig.historical_margin)
    parser.add_argument("--new_margin", type=float, default=RouterConfig.new_margin)
    parser.add_argument("--old_margin", type=float, default=RouterConfig.old_margin)
    parser.add_argument("--c_new", type=float, default=RouterConfig.c_new)
    parser.add_argument("--c_old", type=float, default=RouterConfig.c_old)
    parser.add_argument("--solver_max_iter", type=int, default=RouterConfig.solver_max_iter)
    parser.add_argument(
        "--solver_tolerance", type=float, default=RouterConfig.solver_tolerance
    )
    parser.add_argument(
        "--bootstrap_replicates",
        type=int,
        default=RouterConfig.bootstrap_replicates,
    )
    parser.add_argument(
        "--confidence_alpha", type=float, default=RouterConfig.confidence_alpha
    )
    parser.add_argument(
        "--old_disturbance_budget",
        type=float,
        default=RouterConfig.old_disturbance_budget,
    )
    parser.add_argument(
        "--eval_split", default=RouterConfig.eval_split, choices=["validation", "test"]
    )
    parser.add_argument("--no_save_features", action="store_true")
    parser.add_argument("--force_recompute_features", action="store_true")
    parser.add_argument(
        "--resume_from", default=None, help="Path to an MC-GMM router_stepN.pt checkpoint"
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    default_tasks = RouterConfig.tasks if args.dataset_source == "codetask" else ("swift",)
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
        correction_opt_fraction=args.correction_opt_fraction,
        old_pseudo_opt_samples=args.old_pseudo_opt_samples,
        old_pseudo_cert_samples=args.old_pseudo_cert_samples,
        historical_margin=args.historical_margin,
        new_margin=args.new_margin,
        old_margin=args.old_margin,
        c_new=args.c_new,
        c_old=args.c_old,
        solver_max_iter=args.solver_max_iter,
        solver_tolerance=args.solver_tolerance,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_alpha=args.confidence_alpha,
        old_disturbance_budget=args.old_disturbance_budget,
    )
    run(cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
