"""Routing encoder: hidden state after blocks 0..l*-1, masked mean pool, parameter-free LN, projection P (Sec. 4.1)."""
from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np
import torch

from cgr.atoms import AtomBank, decoder_layers
from cgr.data import format_prompt


def make_projection(d: int, p: int, seed: int) -> np.ndarray:
    """P = Q^T with QR = qr(G), G in R^{d x p}, G_ij ~ N(0, 1). Generated once, saved, never regenerated."""
    G = np.random.default_rng(seed).standard_normal((d, p))
    Q, _ = np.linalg.qr(G, mode="reduced")
    return Q.T.copy()


class _Stop(Exception):
    pass


def pool_ln_project(H: torch.Tensor, mask: torch.Tensor, P: torch.Tensor, eps: float) -> torch.Tensor:
    """z = P * LN(mean_t m_t H_t), all in float32. H [B, T, d], mask [B, T] -> [B, p]."""
    H = H.to(torch.float32)
    m = mask.to(torch.float32).unsqueeze(-1)
    h = (H * m).sum(dim=1) / m.sum(dim=1)
    mean = h.mean(dim=-1, keepdim=True)
    std = h.std(dim=-1, unbiased=False, keepdim=True)  # population std
    h = (h - mean) / (std + eps)
    return h @ P.to(torch.float32).T


class RoutingEncoder:
    def __init__(self, model, tokenizer, split_layer: int, projection: np.ndarray, max_length: int,
                 ln_eps: float, batch_size: int, bank: Optional[AtomBank] = None):
        self.model = model
        self.split_layer = split_layer
        self.max_length = max_length
        self.ln_eps = ln_eps
        self.batch_size = batch_size
        self.bank = bank
        self.P_np = projection
        self.tok = copy.deepcopy(tokenizer)
        self.tok.padding_side = "right"
        self.tok.truncation_side = "right"

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def hidden_at_split(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """H = output_hidden_states[l*], computed with adapters disabled and the forward stopped after block l*-1."""
        captured = {}

        def hook(_module, _inputs, output):
            captured["H"] = output[0] if isinstance(output, tuple) else output
            raise _Stop

        handle = decoder_layers(self.model)[self.split_layer - 1].register_forward_hook(hook)
        ctx = self.bank.disabled() if self.bank is not None else _null()
        try:
            with ctx:
                self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except _Stop:
            pass
        finally:
            handle.remove()
        return captured["H"]

    def tokenize(self, prompts: List[str]):
        return self.tok([format_prompt(x) for x in prompts], add_special_tokens=False, truncation=True,
                        max_length=self.max_length, padding=True, return_tensors="pt")

    @torch.no_grad()
    def embed(self, prompts: List[str]) -> np.ndarray:
        """[B, p] float32 routing embeddings; prompt tokens only, same template as generation."""
        P = torch.as_tensor(self.P_np, dtype=torch.float32, device=self.device)
        out = []
        for s in range(0, len(prompts), self.batch_size):
            enc = self.tokenize(prompts[s:s + self.batch_size])
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            H = self.hidden_at_split(ids, mask)
            out.append(pool_ln_project(H, mask, P, self.ln_eps).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.P_np.shape[0]), np.float32)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
