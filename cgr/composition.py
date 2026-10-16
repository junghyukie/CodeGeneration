"""Per-block lower-triangular composition matrices C^(l) with C_kk = 1, C_kj = beta_kj (Sec. 4.4-4.5)."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn


class Composition(nn.Module):
    def __init__(self, n_blocks: int, identity: bool = False, device=None):
        super().__init__()
        self.n_blocks = n_blocks
        self.device = torch.device(device or "cpu")
        self.identity = identity        # ablations A0/A1: C = I, beta never trained
        self.rows = nn.ParameterList()  # row k (0-based) has shape [n_blocks, k]

    @property
    def K(self) -> int:
        return len(self.rows)

    def add_row(self) -> int:
        k = self.K
        self.rows.append(nn.Parameter(torch.zeros(self.n_blocks, k, device=self.device), requires_grad=False))
        return k

    def matrix(self) -> torch.Tensor:
        """C [n_blocks, K, K], differentiable w.r.t. any trainable row."""
        K = self.K
        dev = self.device
        out = []
        for k, row in enumerate(self.rows):
            parts = [row, torch.ones(self.n_blocks, 1, device=dev), torch.zeros(self.n_blocks, K - k - 1, device=dev)]
            out.append(torch.cat(parts, dim=1))
        return torch.stack(out, dim=1)

    @torch.no_grad()
    def init_row(self, k: int, affinity: np.ndarray) -> None:
        """beta_k^(l) <- sum_{i<k} a_ki C^(l)_{i, :k}  (affinity-weighted merge of earlier effective adapters)."""
        if k == 0 or self.identity:
            return
        a = torch.as_tensor(np.asarray(affinity), dtype=torch.float32, device=self.rows[k].device)
        C_prev = self.matrix()[:, :k, :k]
        self.rows[k].copy_(torch.einsum("i,lij->lj", a, C_prev))

    def trainable_row(self, k: int) -> nn.Parameter:
        row = self.rows[k]
        row.requires_grad_(True)
        return row

    def freeze_row(self, k: int) -> None:
        self.rows[k].requires_grad_(False)

    def coefficients(self, alpha: torch.Tensor) -> torch.Tensor:
        """c^(l) = C^(l)T alpha for every block. alpha [B, K] -> [B, n_blocks, K]."""
        C = self.matrix().to(alpha.device)
        return torch.einsum("bk,lki->bli", alpha.to(C.dtype), C)

    def task_coefficients(self, k: int) -> torch.Tensor:
        """Coefficients that execute task k: row k of every C^(l). Returns [1, n_blocks, K]."""
        return self.matrix()[:, k, :].unsqueeze(0)

    def beta_mean(self) -> List[List[float]]:
        """C averaged over blocks, for routing diagnostics."""
        return self.matrix().mean(dim=0).detach().cpu().tolist()
