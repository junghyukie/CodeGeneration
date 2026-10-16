"""Shared-covariance GMM router: scores, routing weights, affinity prior, tau calibration (Sec. 4.3, 4.5)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.linalg import solve_triangular

from cgr.gmm import LOG_2PI, DiagGMM, _logsumexp, diag_log_joint


class Router:
    def __init__(self, p: int, shrinkage: float, cov_jitter: float, tau_grid: List[float]):
        self.p = p
        self.shrinkage = float(shrinkage)
        self.cov_jitter = float(cov_jitter)
        self.tau_grid = list(tau_grid)
        self.gmms: List[DiagGMM] = []
        self.scatters: List[np.ndarray] = []
        self.counts: List[int] = []
        self.S = np.zeros((p, p), dtype=np.float64)
        self.n = 0
        self.taus: Dict[int, float] = {}   # task number (1-based) -> tau_k
        self.tau = 1.0
        self._chol: List[List[np.ndarray]] = []
        self._logdet: List[List[float]] = []

    @property
    def K(self) -> int:
        return len(self.gmms)

    @property
    def sigma_bar(self) -> np.ndarray:
        return self.S / max(self.n, 1)

    # ---- registration -------------------------------------------------------
    def register(self, gmm: DiagGMM, scatter: np.ndarray, n_k: int) -> None:
        self.gmms.append(gmm)
        self.scatters.append(np.asarray(scatter, dtype=np.float64))
        self.counts.append(int(n_k))
        self.S = self.S + self.scatters[-1]
        self.n += int(n_k)
        self._refresh()

    def _refresh(self) -> None:
        """Recompute all K*M Cholesky factors (Sigma_bar changes on every registration)."""
        self._chol, self._logdet = [], []
        if self.shrinkage == 0.0:
            return
        lam = self.shrinkage
        shared = self.sigma_bar + self.cov_jitter * np.eye(self.p)
        for g in self.gmms:
            Ls, lds = [], []
            for m in range(g.n_components):
                cov = (1.0 - lam) * np.diag(g.var[m]) + lam * shared
                L = np.linalg.cholesky(cov)
                Ls.append(L)
                lds.append(2.0 * float(np.sum(np.log(np.diag(L)))))
            self._chol.append(Ls)
            self._logdet.append(lds)

    # ---- scoring ------------------------------------------------------------
    def scores(self, Z: np.ndarray) -> np.ndarray:
        """s_k(z) = logsumexp_m[log pi_km + log N(z; mu_km, Sigma_km(lambda))] + log(1/K). Returns [B, K]."""
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z[None]
        K = self.K
        out = np.empty((Z.shape[0], K), dtype=np.float64)
        for k, g in enumerate(self.gmms):
            if self.shrinkage == 0.0:
                lj = diag_log_joint(Z, g.pi, g.mu, g.var)
            else:
                lj = np.empty((Z.shape[0], g.n_components))
                for m in range(g.n_components):
                    y = solve_triangular(self._chol[k][m], (Z - g.mu[m]).T, lower=True, check_finite=False)
                    maha = np.sum(y * y, axis=0)
                    lj[:, m] = math.log(g.pi[m]) - 0.5 * (self.p * LOG_2PI + self._logdet[k][m] + maha)
            out[:, k] = _logsumexp(lj, axis=1) + math.log(1.0 / K)
        return out

    def weights(self, Z: np.ndarray, hard: bool, tau: Optional[float] = None) -> np.ndarray:
        s = self.scores(Z)
        if hard:
            w = np.zeros_like(s)
            w[np.arange(s.shape[0]), s.argmax(axis=1)] = 1.0
            return w
        t = self.tau if tau is None else tau
        a = s / t
        return np.exp(a - _logsumexp(a, axis=1)[:, None])

    def affinity(self, Z_new: np.ndarray, tau_aff: float = 1.0) -> np.ndarray:
        """a_kj: mean posterior of the new task's embeddings over the K existing tasks (call before register)."""
        a = self.scores(Z_new) / tau_aff
        post = np.exp(a - _logsumexp(a, axis=1)[:, None])
        return post.mean(axis=0)

    def calibrate_tau(self, Z_heldout: np.ndarray, k: int) -> float:
        """tau_k = argmin_tau NLL of the true task index k (0-based) on H_k; tau = geometric mean over k >= 2."""
        s = self.scores(Z_heldout)
        nll = []
        for t in self.tau_grid:
            a = s / t
            nll.append(-float(np.mean(a[:, k] - _logsumexp(a, axis=1))))
        tau_k = self.tau_grid[int(np.argmin(nll))]
        self.taus[k + 1] = tau_k
        self.tau = float(math.exp(np.mean([math.log(v) for v in self.taus.values()])))
        return tau_k

    # ---- persistence --------------------------------------------------------
    def save(self, router_dir: str | Path) -> None:
        d = Path(router_dir)
        d.mkdir(parents=True, exist_ok=True)
        for i, g in enumerate(self.gmms, start=1):
            path = d / f"gmm_{i}.npz"
            if not path.exists():  # frozen once written
                np.savez(path, **g.arrays())
        np.savez(d / "scatter.npz", S=self.S, n=np.array(self.n), counts=np.array(self.counts),
                 **{f"S_{i}": s for i, s in enumerate(self.scatters, start=1)})
        with open(d / "tau.json", "w", encoding="utf-8") as f:
            json.dump({"tau_k": {str(k): v for k, v in self.taus.items()}, "tau": self.tau}, f, indent=2)

    @classmethod
    def load(cls, router_dir: str | Path, p: int, shrinkage: float, cov_jitter: float,
             tau_grid: List[float]) -> "Router":
        d = Path(router_dir)
        r = cls(p, shrinkage, cov_jitter, tau_grid)
        sc = np.load(d / "scatter.npz")
        counts = [int(c) for c in sc["counts"]]
        for i in range(1, len(counts) + 1):
            g = np.load(d / f"gmm_{i}.npz")
            r.gmms.append(DiagGMM(pi=g["pi"], mu=g["mu"], var=g["var"]))
            r.scatters.append(sc[f"S_{i}"])
        r.counts, r.S, r.n = counts, sc["S"], int(sc["n"])
        with open(d / "tau.json", encoding="utf-8") as f:
            t = json.load(f)
        r.taus = {int(k): float(v) for k, v in t["tau_k"].items()}
        r.tau = float(t["tau"])
        r._refresh()
        return r
