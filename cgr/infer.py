"""Predict (Algorithm 3): embed -> score -> route -> compose -> generate once. The ONLY inference entry point."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from transformers import GenerationConfig

from cgr.state import CGRState
from utils.data.data_collator import DataCollator


@dataclass
class Prediction:
    samples: List[str]
    alpha: List[float]
    k_hat: int   # 0-based task index chosen by argmax of s_k(z); -1 when there is no router (baselines)


class Predictor:
    def __init__(self, state: CGRState):
        self._state = state

    def predict(self, prompts: List[str], n_samples: int) -> List[Prediction]:
        """Accepts prompt strings only (Invariant I1); one generation round, no execution (I2, I3)."""
        st, cfg = self._state, self._state.cfg
        prompts = list(prompts)
        out: List[Optional[Prediction]] = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # batch similar lengths together
        for s in range(0, len(order), cfg.infer_batch_size):
            idx = order[s:s + cfg.infer_batch_size]
            chunk = [prompts[i] for i in idx]
            Z = st.encoder.embed(chunk)
            k_hat = st.router.scores(Z).argmax(axis=1)
            alpha = st.router.weights(Z, hard=(cfg.routing_mode == "hard"), tau=cfg.tau_override)
            samples = generate(st, chunk, n_samples, alpha=alpha)
            for j, i in enumerate(idx):
                out[i] = Prediction(samples[j], alpha[j].tolist(), int(k_hat[j]))
        return out


@torch.no_grad()
def generate(st: CGRState, prompts: List[str], n_samples: int, alpha: Optional[np.ndarray] = None) -> List[List[str]]:
    """Internal generation with c^(l) = C^(l)T alpha for every adapted block; alpha=None disables all atoms
    (pretrained backbone). Not a public entry point: methods call Predictor.predict."""
    cfg, tok, model = st.cfg, st.tokenizer, st.model
    col = DataCollator(tok, padding="longest", max_prompt_len=cfg.max_prompt_len_infer, max_ans_len=cfg.max_new_tokens,
                       pad_to_multiple_of=8, inference=True)
    batch = col([{"prompt": p, "answer": ""} for p in prompts])
    dev = st.encoder.device
    ids, mask = batch["input_ids"].to(dev), batch["attention_mask"].to(dev)
    gen = GenerationConfig(do_sample=cfg.do_sample, temperature=cfg.temperature if cfg.do_sample else None,
                           top_p=cfg.top_p if cfg.do_sample else None, top_k=0 if cfg.do_sample else None,
                           num_return_sequences=n_samples, max_new_tokens=cfg.max_new_tokens,
                           eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    model.eval()
    if alpha is None:
        with st.bank.disabled():
            seq = model.generate(input_ids=ids, attention_mask=mask, generation_config=gen, use_cache=True)
    else:
        c = st.composition.coefficients(torch.as_tensor(alpha, dtype=torch.float32, device=dev))
        with st.bank.use(c):
            seq = model.generate(input_ids=ids, attention_mask=mask, generation_config=gen, use_cache=True)
    text = tok.batch_decode(seq[:, ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [text[i * n_samples:(i + 1) * n_samples] for i in range(len(prompts))]
