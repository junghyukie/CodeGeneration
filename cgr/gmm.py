"""Per-task diagonal GMM (thesis Algorithm 1) and within-task scatter (Sec. 4.2, Algorithm 1)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class DiagGMM:
    pi: np.ndarray   # [M]
    mu: np.ndarray   # [M, p]
    var: np.ndarray  # [M, p]
    ll_history: List[float] = field(default_factory=list, compare=False)

    @property
    def n_components(self) -> int:
        return int(self.pi.shape[0])

    def arrays(self) -> dict:
        return {"pi": self.pi, "mu": self.mu, "var": self.var}


def diag_log_joint(Z: np.ndarray, pi: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """log pi_m + log N(z; mu_m, diag var_m) for every (i, m). Returns [N, M] float64."""
    Z = np.asarray(Z, dtype=np.float64)
    p = Z.shape[1]
    out = np.empty((Z.shape[0], pi.shape[0]), dtype=np.float64)
    for m in range(pi.shape[0]):
        d = Z - mu[m]
        quad = np.sum(d * d / var[m], axis=1)
        out[:, m] = math.log(pi[m]) - 0.5 * (p * LOG_2PI + np.sum(np.log(var[m])) + quad)
    return out


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    amax = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(amax, axis=axis) + np.log(np.sum(np.exp(a - amax), axis=axis))


def fit_diag_gmm(Z: np.ndarray, M: int, t_max: int, tol: float, var_floor: float,
                 seed: int) -> Tuple[DiagGMM, np.ndarray]:
    """EM for a diagonal GMM. Returns (gmm, responsibilities [N, M]) from a final E-step."""
    Z = np.asarray(Z, dtype=np.float64)
    N, _ = Z.shape
    if N < M:
        raise ValueError(f"need at least M={M} embeddings, got {N}")
    rng = np.random.default_rng(seed)
    pi = np.full(M, 1.0 / M)
    mu = Z[rng.choice(N, size=M, replace=False)].copy()
    var = np.tile(np.maximum(Z.var(axis=0), var_floor), (M, 1))

    history: List[float] = []
    prev = -np.inf
    for _ in range(t_max):
        log_joint = diag_log_joint(Z, pi, mu, var)
        log_norm = _logsumexp(log_joint, axis=1)
        resp = np.exp(log_joint - log_norm[:, None])
        ll = float(log_norm.mean())
        history.append(ll)

        Nm = np.maximum(resp.sum(axis=0), 1e-12)
        pi = Nm / N
        mu = (resp.T @ Z) / Nm[:, None]
        var = np.empty_like(mu)
        for m in range(M):
            d = Z - mu[m]
            var[m] = (resp[:, m:m + 1] * d * d).sum(axis=0) / Nm[m]
        var = np.maximum(var, var_floor)

        if abs(ll - prev) < tol:
            break
        prev = ll

    log_joint = diag_log_joint(Z, pi, mu, var)
    resp = np.exp(log_joint - _logsumexp(log_joint, axis=1)[:, None])
    return DiagGMM(pi=pi, mu=mu, var=var, ll_history=history), resp


def within_scatter(Z: np.ndarray, gmm: DiagGMM, resp: np.ndarray) -> np.ndarray:
    """S_k = sum_i sum_m r_im (z_i - mu_m)(z_i - mu_m)^T, accumulated in float64. Returns [p, p]."""
    Z = np.asarray(Z, dtype=np.float64)
    p = Z.shape[1]
    S = np.zeros((p, p), dtype=np.float64)
    for m in range(gmm.n_components):
        d = Z - gmm.mu[m]
        S += (d * resp[:, m:m + 1]).T @ d
    return S
