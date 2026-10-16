"""Persistent CGR state (Sec. 5 table) and the checkpoint layout of Sec. 9."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from cgr.atoms import AtomBank, install_atom_bank
from cgr.composition import Composition
from cgr.config import CGRConfig
from cgr.features import RoutingEncoder, make_projection
from cgr.router import Router


def _h(*arrays) -> str:
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        m.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return m.hexdigest()


@dataclass
class CGRState:
    cfg: CGRConfig
    model: torch.nn.Module
    tokenizer: object
    P: np.ndarray
    router: Router
    bank: AtomBank
    composition: Composition
    encoder: RoutingEncoder

    @property
    def K(self) -> int:
        return self.router.K

    @classmethod
    def create(cls, cfg: CGRConfig, model, tokenizer) -> "CGRState":
        for p in model.parameters():
            p.requires_grad_(False)
        d = model.config.hidden_size
        P = make_projection(d, cfg.routing_dim, cfg.projection_seed)
        router = Router(cfg.routing_dim, cfg.shrinkage, cfg.cov_jitter, cfg.tau_grid)
        bank = install_atom_bank(model, cfg.split_layer, cfg.target_modules, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout)
        comp = Composition(bank.n_blocks, identity=(cfg.composition == "identity"),
                           device=next(model.parameters()).device)
        enc = RoutingEncoder(model, tokenizer, cfg.split_layer, P, cfg.routing_max_length, cfg.ln_eps,
                             cfg.embed_batch_size, bank)
        return cls(cfg, model, tokenizer, P, router, bank, comp, enc)

    # ---- hashes (test T7) -----------------------------------------------------
    def content_hashes(self) -> Dict[str, str]:
        out = {"projection_P": _h(self.P)}
        for i, g in enumerate(self.router.gmms, start=1):
            out[f"gmm_{i}"] = _h(g.pi, g.mu, g.var)
        for i in range(self.bank.K):
            sd = self.bank.atom_state_dict(i)
            out[f"atom_{i + 1}"] = _h(*[sd[k].numpy() for k in sorted(sd)])
        for i, row in enumerate(self.composition.rows, start=1):
            out[f"beta_{i}"] = _h(row.detach().cpu().numpy())
        return out

    # ---- checkpoint ---------------------------------------------------------
    def save(self, state_dir: str | Path) -> None:
        d = Path(state_dir)
        (d / "atoms").mkdir(parents=True, exist_ok=True)
        (d / "composition").mkdir(parents=True, exist_ok=True)
        self.cfg.save(d / "config.json")
        if not (d / "projection_P.npy").exists():
            np.save(d / "projection_P.npy", self.P)
        self.router.save(d / "router")
        for i in range(self.bank.K):
            path = d / "atoms" / f"atom_{i + 1}.safetensors"
            if not path.exists():
                save_file(self.bank.atom_state_dict(i), str(path))
        for i, row in enumerate(self.composition.rows, start=1):
            path = d / "composition" / f"beta_{i}.npy"
            if not path.exists():
                np.save(path, row.detach().cpu().numpy())
        with open(d / "manifest.json", "w", encoding="utf-8") as f:
            json.dump({"content_sha256": self.content_hashes()}, f, indent=2)

    @classmethod
    def load(cls, state_dir: str | Path, cfg: CGRConfig, model, tokenizer) -> "CGRState":
        d = Path(state_dir)
        st = cls.create(cfg, model, tokenizer)
        st.P = np.load(d / "projection_P.npy")
        st.encoder.P_np = st.P
        st.router = Router.load(d / "router", cfg.routing_dim, cfg.shrinkage, cfg.cov_jitter, cfg.tau_grid)
        for i in range(1, st.router.K + 1):
            k = st.bank.add_atom()
            st.bank.load_atom(k, load_file(str(d / "atoms" / f"atom_{i}.safetensors")))
            st.bank.freeze_atom(k)
            st.composition.add_row()
            with torch.no_grad():
                st.composition.rows[k].copy_(torch.from_numpy(np.load(d / "composition" / f"beta_{i}.npy")))
        with open(d / "manifest.json", encoding="utf-8") as f:
            expected = json.load(f)["content_sha256"]
        if st.content_hashes() != expected:
            raise RuntimeError(f"manifest mismatch while loading {d}")
        return st
