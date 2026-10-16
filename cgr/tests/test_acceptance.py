"""Acceptance tests T1-T10 (methodology.tex Sec. 10). Run: python -m unittest cgr.tests.test_acceptance -v"""
from __future__ import annotations

import ast
import copy
import inspect
import itertools
import math
import unittest
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from cgr.config import CGRConfig
from cgr.data import Example, ReplayError, TaskData, TaskRegistry, format_prompt
from cgr.features import pool_ln_project
from cgr.gmm import LOG_2PI, fit_diag_gmm, within_scatter
from cgr.infer import Predictor
from cgr.router import Router
from cgr.state import CGRState
from cgr.train import learn_task
from utils.data.data_collator import DataCollator

CGR_DIR = Path(__file__).resolve().parents[1]


def tiny_tokenizer():
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
    vocab["<|endoftext|>"] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    t = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    t.padding_side, t.truncation_side = "left", "left"
    return t


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=264, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    return Qwen2ForCausalLM(cfg).float().eval()


def tiny_cfg(**kw):
    base = dict(tasks=["add", "rev"], split_layer=2, routing_dim=16, gmm_components=2, gmm_max_samples=64,
                lora_r=4, lora_alpha=8, batch_size=8, micro_batch_size=4, phase_a_steps=3, phase_b_epochs=1,
                max_new_tokens=6, do_sample=False, infer_batch_size=4, heldout_frac=0.2, heldout_max=8,
                max_prompt_len_infer=64, max_prompt_len_train={"add": 32, "rev": 32},
                max_ans_len_train={"add": 8, "rev": 8}, routing_mode="soft")
    base.update(kw)
    return CGRConfig(**base)


def synth_task(name, n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        if name == "add":
            a, b = rng.integers(0, 50, 2)
            out.append(Example(f"add {a} {b}", str(a + b)))
        else:
            w = "".join(rng.choice(list("abcdef"), 4))
            out.append(Example(f"reverse the word {w} please", w[::-1]))
    return out


def synth_source(name, split):
    return synth_task(name) if split == "train" else synth_task(name, 8, seed=1)


class T1AdapterIndependentFeatures(unittest.TestCase):
    def test_features(self):
        cfg = tiny_cfg()
        st = CGRState.create(cfg, tiny_model(), tiny_tokenizer())
        for _ in range(2):
            k = st.bank.add_atom()
            st.composition.add_row()
            for layer in st.bank.layers:
                torch.nn.init.normal_(layer.B[k], std=0.5)
        prompts = ["add 1 2", "reverse the word abcd please"]
        z_enc = st.encoder.embed(prompts)
        enc = st.encoder.tokenize(prompts)
        c = torch.randn(2, st.bank.n_blocks, 2)
        with st.bank.use(c), torch.no_grad():
            out = st.model(**enc, output_hidden_states=True)
            H = out.hidden_states[cfg.split_layer]
        z_full = pool_ln_project(H, enc["attention_mask"], torch.as_tensor(st.P), cfg.ln_eps).numpy()
        self.assertLess(np.abs(z_full - z_enc).max(), 1e-4)

    def test_routing_uses_generation_template(self):
        tok = tiny_tokenizer()
        col = DataCollator(tok, padding="longest", max_prompt_len=64, max_ans_len=8, pad_to_multiple_of=1, inference=True)
        ids = col([{"prompt": "add 1 2", "answer": ""}])["input_ids"][0].tolist()
        self.assertEqual(ids, tok(format_prompt("add 1 2"), add_special_tokens=False)["input_ids"])


class T2EMReproduction(unittest.TestCase):
    def test_em(self):
        rng = np.random.default_rng(0)
        mus = np.array([[-6.0] * 4, [0.0] * 4, [6.0] * 4])
        Z = np.concatenate([rng.normal(m, 1.0, size=(400, 4)) for m in mus])
        g, _ = fit_diag_gmm(Z, 3, 100, 1e-8, 1e-3, seed=1)
        best = min(np.abs(g.mu[list(p)] - mus).max() for p in itertools.permutations(range(3)))
        self.assertLess(best, 0.3)
        self.assertTrue(np.all(np.diff(g.ll_history) >= -1e-9))


class T3Scatter(unittest.TestCase):
    def test_pooled_within_class(self):
        rng = np.random.default_rng(0)
        r = Router(5, 1.0, 1e-4, [1.0])
        Zs = [rng.normal(c, s, size=(n, 5)) for c, s, n in [(0, 1, 50), (3, 2, 80), (-2, 0.5, 30)]]
        for Z in Zs:
            g, resp = fit_diag_gmm(Z, 1, 50, 1e-4, 1e-6, seed=0)
            r.register(g, within_scatter(Z, g, resp), len(Z))
        direct = sum((Z - Z.mean(0)).T @ (Z - Z.mean(0)) for Z in Zs) / sum(len(Z) for Z in Zs)
        np.testing.assert_allclose(r.sigma_bar, direct, rtol=1e-6)


class T4ThesisRouterEquivalence(unittest.TestCase):
    def test_lambda0(self):
        rng = np.random.default_rng(0)
        r = Router(6, 0.0, 1e-4, [1.0])
        for c in (0.0, 2.0, -2.0):
            Z = rng.normal(c, 1, size=(60, 6))
            g, resp = fit_diag_gmm(Z, 2, 50, 1e-4, 0.02, seed=0)
            r.register(g, within_scatter(Z, g, resp), len(Z))
        Q = rng.normal(0, 2, size=(10, 6))
        ref = np.zeros((10, 3))
        for k, g in enumerate(r.gmms):
            for i, z in enumerate(Q):
                terms = [math.log(g.pi[m]) - 0.5 * (6 * LOG_2PI + np.log(g.var[m]).sum() + ((z - g.mu[m]) ** 2 / g.var[m]).sum())
                         for m in range(2)]
                ref[i, k] = np.logaddexp.reduce(terms) + math.log(1 / 3)
        np.testing.assert_allclose(r.scores(Q), ref, atol=1e-8, rtol=0)


class T5MergeEquivalence(unittest.TestCase):
    def test_merge(self):
        st = CGRState.create(tiny_cfg(lora_dropout=0.0), tiny_model(), tiny_tokenizer())
        K = 3
        for _ in range(K):
            k = st.bank.add_atom()
            st.composition.add_row()
            for layer in st.bank.layers:
                torch.nn.init.normal_(layer.B[k], std=0.3)
        with torch.no_grad():
            for k in range(1, K):
                st.composition.rows[k].normal_()
        alpha = torch.softmax(torch.randn(2, K), -1)
        c = st.composition.coefficients(alpha)
        C = st.composition.matrix()
        layer = st.bank.layers[3]
        x = torch.randn(2, 5, layer.base.in_features)
        with st.bank.use(c), torch.no_grad():
            y = layer(x)
        kappa = layer.scaling
        for b in range(2):
            dW = sum(alpha[b, k] * sum(C[layer.block_pos, k, i] * layer.B[i] @ layer.A[i] for i in range(K)) for k in range(K))
            ref = layer.base(x[b]) + kappa * x[b] @ dW.T
            torch.testing.assert_close(y[b], ref, rtol=1e-4, atol=1e-5)


def _run_two_tasks(seed=0, **kw):
    cfg = tiny_cfg(**kw)
    st = CGRState.create(cfg, tiny_model(seed), tiny_tokenizer())
    reg = TaskRegistry(cfg.tasks, synth_source, cfg.heldout_frac, cfg.heldout_max, 0)
    logs, hashes = [], []
    for k in range(2):
        hashes.append(st.content_hashes())
        reg.begin_task(k)
        logs.append(learn_task(k, reg.training_data(k), st, cfg))
        reg.end_task(k)
    return st, reg, logs, hashes


class T6T7PhaseContinuityFreeze(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st, cls.reg, cls.logs, cls.hashes = _run_two_tasks()

    def test_phase_continuity(self):
        recs = self.logs[1]["records"]
        a_end = [r["heldout_loss"] for r in recs if r.get("phase") == "A" and r.get("end")][0]
        b0 = [r["heldout_loss"] for r in recs if r.get("phase") == "B" and r.get("step") == 0][0]
        self.assertLess(abs(a_end - b0), 1e-5)

    def test_new_atom_gets_no_phase_a_gradient(self):
        # Atom k is created frozen, so Phase A cannot leave stale gradients for Phase B.
        st = CGRState.create(tiny_cfg(), tiny_model(), tiny_tokenizer())
        k = st.bank.add_atom()
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in st.bank.atom_parameters(k)))

    def test_freeze_integrity(self):
        before, after = self.hashes[1], self.st.content_hashes()
        for key in ("projection_P", "gmm_1", "atom_1", "beta_1"):
            self.assertEqual(before[key], after[key], key)
        self.assertTrue(all(not p.requires_grad for p in self.st.bank.atom_parameters(1)))

    def test_checkpoint_roundtrip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.st.save(d)
            st2 = CGRState.load(d, self.st.cfg, tiny_model(), tiny_tokenizer())
            self.assertEqual(st2.content_hashes(), self.st.content_hashes())


class T8LeakageGuard(unittest.TestCase):
    def test_predict_signature(self):
        params = list(inspect.signature(Predictor.predict).parameters)
        self.assertEqual(params, ["self", "prompts", "n_samples"])

    def test_static_imports(self):
        # Execution harness / test-case loaders, and the thesis refinement code paths (calibration inference,
        # adapter refinement, traceback GMM, Round-2) that methodology Sec. 1 requires to be unreachable from CGR.
        banned = {"subprocess", "evaluate", "calc_metrics", "executable_dataset", "gmm", "gmm_traceback",
                  "infer_gmm", "training", "model"}
        for f in CGR_DIR.rglob("*.py"):
            rel = f.relative_to(CGR_DIR)
            if rel.parts[0] in ("eval", "tests"):
                continue
            for node in ast.walk(ast.parse(f.read_text())):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                        [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                for n in names:
                    self.assertNotIn(n.split(".")[0], banned, f"{rel} imports {n}")


class T9NoReplayGuard(unittest.TestCase):
    def test_registry(self):
        reg = TaskRegistry(["add", "rev", "add"], synth_source, 0.1, 5, 0)
        reg.begin_task(0); reg.training_data(0); reg.end_task(0)
        reg.begin_task(1)
        with self.assertRaises(ReplayError):
            reg.training_data(0)
        reg.training_data(1)
        view = reg.evaluation_view()
        self.assertIsInstance(view.test_split(0), tuple)
        with self.assertRaises(ReplayError):
            view.test_split(2)
        reg.end_task(1)
        with self.assertRaises(ReplayError):
            reg.training_data(1)
        with self.assertRaises(ReplayError):
            reg.begin_task(0)


class T10Determinism(unittest.TestCase):
    def test_determinism(self):
        outs = []
        for _ in range(2):
            st, _, _, _ = _run_two_tasks()
            preds = Predictor(st).predict(["add 3 4", "reverse the word abcd please"], 1)
            outs.append(([p.alpha for p in preds], [p.samples for p in preds],
                         [r.detach().clone() for r in st.composition.rows]))
        self.assertEqual(outs[0][0], outs[1][0])
        self.assertEqual(outs[0][1], outs[1][1])
        for a, b in zip(outs[0][2], outs[1][2]):
            self.assertTrue(torch.equal(a, b))


class EndToEndSmoke(unittest.TestCase):
    """run_sequence.run -> eval.score -> eval.report for cgr (A3, A1), seqlora and base on two BLEU-scored tasks."""

    def test_pipeline(self):
        import json
        import tempfile
        from cgr.eval.score import score_run
        from cgr.run_sequence import run

        names = ["CodeSearchNet", "TheVault_Csharp"]

        def source(name, split):
            return synth_source("add" if name == names[0] else "rev", split)

        with tempfile.TemporaryDirectory() as d:
            dirs = {}
            for tag, method, abl in [("A3", "cgr", "A3"), ("A1", "cgr", "A1"), ("SEQ", "seqlora", "A3"),
                                     ("BASE", "base", "A3")]:
                from cgr.config import apply_ablation
                cfg = apply_ablation(tiny_cfg(tasks=names, max_prompt_len_train={n: 32 for n in names},
                                              max_ans_len_train={n: 8 for n in names}, routing_mode="hard"), abl)
                cfg.method = method
                out = Path(d) / tag
                run(cfg, tiny_model(), tiny_tokenizer(), source, out, eval_mode="full", eval_phase_a=True)
                res = score_run(out)
                dirs[tag] = out
                if method == "base":
                    self.assertEqual(len(res["base"]), 2)
                else:
                    self.assertEqual(len(res["matrix"]), 2)
                    self.assertIn("BWT", res["summary"])
            self.assertTrue((dirs["A3"] / "predictions" / "phase_a").exists())
            self.assertFalse((dirs["A1"] / "predictions" / "phase_a").exists())
            self.assertTrue((dirs["A3"] / "state" / "manifest.json").exists())
            # Resuming a finished CGR run reloads the checkpoint and verifies the manifest.
            cfg = CGRConfig.load(dirs["A3"] / "config.json")
            st = run(cfg, tiny_model(), tiny_tokenizer(), source, dirs["A3"], eval_mode="none", resume=True)
            self.assertEqual(st.K, 2)
            import subprocess, sys
            rep = subprocess.run([sys.executable, "-m", "cgr.eval.report", "--runs", *map(str, dirs.values()),
                                  "--base", str(dirs["BASE"])], capture_output=True, text=True, check=True)
            self.assertIn("Composition gain", rep.stdout)
            self.assertIn("SeqLoRA", rep.stdout)


if __name__ == "__main__":
    unittest.main()
