"""LoRA atom bank: y = W x + kappa * sum_i c_i B_i A_i drop(x), without merging weights (Sec. 4.4)."""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn


class AtomBankLinear(nn.Module):
    def __init__(self, base: nn.Linear, bank: "AtomBank", block_pos: int, r: int, scaling: float, dropout: float):
        super().__init__()
        self.base = base
        self.block_pos = block_pos
        self.r = r
        self.scaling = scaling
        self.drop = nn.Dropout(dropout)
        self.A = nn.ParameterList()
        self.B = nn.ParameterList()
        object.__setattr__(self, "_bank", bank)  # plain reference, not a submodule

    def add_atom(self) -> None:
        dev = self.base.weight.device
        A = torch.empty(self.r, self.base.in_features, device=dev, dtype=torch.float32)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))  # PEFT default
        B = torch.zeros(self.base.out_features, self.r, device=dev, dtype=torch.float32)
        # Created frozen: only Phase B unfreezes the new atom (no stray gradients during Phase A).
        self.A.append(nn.Parameter(A, requires_grad=False))
        self.B.append(nn.Parameter(B, requires_grad=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        bank = self._bank
        if bank.disabled_flag or len(self.A) == 0:
            return y
        c = bank.coefficients
        if c is None:
            raise RuntimeError("atom-bank coefficients are not set")
        c = c[:, self.block_pos, :]                       # [Bc, K]
        if c.shape[0] != x.shape[0]:                      # generate() expands each query n times
            c = c.repeat_interleave(x.shape[0] // c.shape[0], dim=0)
        A_cat = torch.cat(list(self.A), dim=0)            # [K r, d_in]
        B_cat = torch.cat(list(self.B), dim=1)            # [d_out, K r]
        u = self.drop(x).to(torch.float32) @ A_cat.T      # [B, T, K r]
        scale = c.to(torch.float32).repeat_interleave(self.r, dim=-1)  # [B, K r]
        u = u * scale.unsqueeze(1)
        return y + (self.scaling * (u @ B_cat.T)).to(y.dtype)


class AtomBank:
    """Owns every AtomBankLinear; holds the per-example coefficients c [B, n_blocks, K]."""

    def __init__(self, layers: List[AtomBankLinear], blocks: Sequence[int], modules: Sequence[str]):
        self.layers = layers
        self.blocks = list(blocks)
        self.modules = list(modules)
        self.coefficients: Optional[torch.Tensor] = None
        self.disabled_flag = False
        self.K = 0

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def add_atom(self) -> int:
        for layer in self.layers:
            layer.add_atom()
        self.K += 1
        return self.K - 1

    def atom_parameters(self, i: int) -> List[nn.Parameter]:
        return [p for layer in self.layers for p in (layer.A[i], layer.B[i])]

    def freeze_atom(self, i: int) -> None:
        for p in self.atom_parameters(i):
            p.requires_grad_(False)

    def set_coefficients(self, c: Optional[torch.Tensor]) -> None:
        self.coefficients = c

    @contextmanager
    def use(self, c: torch.Tensor):
        prev = self.coefficients
        self.coefficients = c
        try:
            yield
        finally:
            self.coefficients = prev

    @contextmanager
    def disabled(self):
        prev = self.disabled_flag
        self.disabled_flag = True
        try:
            yield
        finally:
            self.disabled_flag = prev

    def atom_state_dict(self, i: int) -> Dict[str, torch.Tensor]:
        out = {}
        for layer, (blk, mod) in zip(self.layers, self._keys()):
            out[f"{blk}.{mod}.A"] = layer.A[i].detach().cpu().contiguous()
            out[f"{blk}.{mod}.B"] = layer.B[i].detach().cpu().contiguous()
        return out

    def load_atom(self, i: int, sd: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for layer, (blk, mod) in zip(self.layers, self._keys()):
                layer.A[i].copy_(sd[f"{blk}.{mod}.A"])
                layer.B[i].copy_(sd[f"{blk}.{mod}.B"])

    def _keys(self):
        return [(b, m) for b in self.blocks for m in self.modules]


def decoder_layers(model) -> nn.ModuleList:
    return model.get_decoder().layers


def install_atom_bank(model, split_layer: int, modules: Sequence[str], r: int, lora_alpha: int,
                      dropout: float) -> AtomBank:
    """Install atom banks on `modules` of blocks split_layer..L-1 only, so blocks < split_layer stay adapter-free."""
    layers = decoder_layers(model)
    blocks = list(range(split_layer, len(layers)))
    bank_layers: List[AtomBankLinear] = []
    bank = AtomBank(bank_layers, blocks, modules)
    for pos, b in enumerate(blocks):
        attn = layers[b].self_attn
        for mod in modules:
            wrapped = AtomBankLinear(getattr(attn, mod), bank, pos, r, lora_alpha / r, dropout)
            setattr(attn, mod, wrapped)
            bank_layers.append(wrapped)
    return bank
