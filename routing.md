# Round-2 Routing Methods

After round 1, samples where **all k predictions failed** are re-generated in a second
pass.  The input router computed weights in round 1 (saved as `input_router_scores` per
sample); a separate traceback router processes the first error traceback from round 1 and
returns its own score vector.  The two signals are combined via one of the methods below.

---

## Soft routing methods

These produce a weight vector **w ∈ ℝ^K** (sums to 1) used for adapter merging:
`W_eff = W_base + Σ_k w[k] · ΔW_k`

---

### `poe` — Product of Experts

```
w = softmax(s_input / T_in + s_trace / T_tr)
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `--round2_T_input` | — (required) | Temperature applied to input-router raw scores |
| `--round2_T_trace` | — (required) | Temperature applied to traceback-router raw scores |

**Effect of hyperparameters**

- `T_input` **lower → sharper input distribution** (emphasises the input router's top task).
  `T_input=0.5` makes the input router nearly hard; `T_input=2.0` flattens it.
- `T_trace` **lower → sharper traceback distribution** (gives the traceback router more
  control).  `T_trace=0.5` lets a confident traceback signal dominate.

**When to use:** Both routers are reliable and you want a principled product-of-experts
fusion.  Good starting point.  
**Typical values:** `T_input=1.0, T_trace=1.0` to start; decrease T_trace (e.g. `0.5`)
if the traceback router is more accurate, increase it (e.g. `2.0`) to dampen noisy signals.

---

### `conf_linear` — Confidence-Weighted Linear Blend

```
conf_input = 1 / (H(w_input) + ε)
conf_trace = 1 / (H(w_trace) + ε)
α = conf_trace / (conf_input + conf_trace)
w = (1 - α) * w_input + α * w_trace
```
where H is Shannon entropy (in nats).

**Arguments:** none (self-calibrating).

**Intuition:** Whichever router is more "sure" (lower entropy / more peaked distribution)
gets more weight automatically.  If both are equally uncertain, α ≈ 0.5.

**When to use:** You don't want to tune temperature but need the more confident router
to lead.  Robust when one router is occasionally unreliable.

---

### `disagree_explore` — JSD-Gated Posterior + Uniform Exploration

```
JSD = Jensen-Shannon divergence(w_input, w_trace)  ∈ [0, 1]
w_posterior = softmax(s_input + s_trace)
w = (1 - JSD) * w_posterior + JSD * uniform(K)
```

**Arguments:** none.

**Intuition:** When the two routers **agree** (low JSD), their combined posterior is
trusted.  When they **disagree** strongly (high JSD ≈ 1), the combined weight moves
toward the uniform — acknowledging that neither router is reliable and exploring all
adapters.

**When to use:** You want the model to be cautious when the two routing signals
conflict, rather than committing to either.  Good for out-of-distribution inputs.

---

### `geo_interp` — Geometric Interpolation

```
log_w = (1 - α) * log(w_input) + α * log(w_trace)
w = softmax(log_w)
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `--round2_alpha` | — (required) | Weight α ∈ [0, 1] given to the traceback router |

**Intuition:** Equivalent to a Bayesian product-of-experts with exponents `(1-α)` and
`α`.  Adapters that get low probability from **either** router are penalised
multiplicatively, unlike linear blending where a high score from one router can save a
low-probability adapter.

- `α = 0` → pure input router.
- `α = 0.5` → equal geometric weight.
- `α = 1` → pure traceback router.

**When to use:** You know the relative reliability of each router and want explicit,
interpretable control.  
**Typical values:** start at `α=0.3` (trust traceback router moderately), increase to
`α=0.5`–`0.7` if traceback accuracy is higher.

---

### `tb_mask` — Traceback-Guided Masking

```
mask[k] = (w_trace[k] > 1 / (2K))   # adapter is "plausible" according to traceback router
w = w_input * mask
w /= sum(w)
```

**Arguments:** none.

**Intuition:** Hard exclusion of adapters that the traceback router considers unlikely
(below the "flat" threshold 1/(2K)).  The input router's distribution is renormalised
over the remaining adapters.

**When to use:** The traceback router is reliable enough to veto adapters outright.
Simpler and more decisive than soft blending.  Can fail if the traceback router is wrong
and masks the correct adapter.

---

## Hard routing methods

These select a **single adapter** (argmax) for generation.

---

### `hard_poe` — Hard Product of Experts

```
k = argmax(s_input + s_trace)
```

**Arguments:** none.

**Intuition:** Sum the raw z-scored GMM log-probs from both routers and pick the top
task.  Equivalent to PoE with T_input = T_trace = 1.0 but using argmax instead of
softmax.

**When to use:** Fast, no adapter merging overhead.  Good baseline for hard-routing mode.

---

### `conf_gate` — Confidence Gate

```
conf = max(p_trace) - H(p_trace)
if conf > threshold:
    k = argmax(s_trace)   # traceback router is confident → use it
else:
    k = argmax(s_input)   # traceback router is uncertain → fall back to input router
```
where p_trace = softmax(s_trace).

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `--conf_gate_threshold` | — (required) | Confidence threshold for switching to the traceback router |

**Intuition of `conf`:** `max(p_trace)` is high when the traceback router has a
dominant preferred task; `H(p_trace)` is low when the distribution is peaked.  Their
difference is a concise "confidence" measure.

- **`conf > 0`**: the distribution is more peaked than a single-component distribution
  with the same max.  A reasonable starting point.
- **`conf > 0.3`**: stricter — only use the traceback router when it is very sure.
- **`conf < 0`** (i.e. very flat distribution): the traceback router essentially never
  takes over (fallback always).

**When to use:** You want the traceback router to override only when it has a clear
signal, without affecting ambiguous cases.  Good when the traceback router is accurate
but noisy (sometimes flat, sometimes confident).  
**Typical values:** start at `conf_gate_threshold=0.1`, increase to `0.3` to require
higher confidence.

---

## Choosing a method

| Scenario | Recommended method |
|---|---|
| Both routers are calibrated, want principled fusion | `poe` (T=1.0/1.0) |
| Unknown relative reliability, no tuning budget | `conf_linear` |
| Routers often disagree, want safe exploration | `disagree_explore` |
| Know relative router quality, want explicit control | `geo_interp` |
| Traceback router reliable enough to veto adapters | `tb_mask` |
| Hard routing, want fastest inference | `hard_poe` |
| Hard routing, want traceback to override selectively | `conf_gate` |

---

## Iterative use

Each round reads the executed results from the previous round and refines only the
samples where **all** predictions failed again.  The `input_router_scores` field saved
in each results file avoids re-running the input router.

```bash
# Round 1: standard inference
bash scripts/infer_gmm.sh

# (external) execute round-1 predictions → results-8-python.json now has passed/stderr

# Round 2: refine hard samples
bash scripts/infer_gmm_refine.sh --round_num 2

# (external) execute round-2 predictions → results-8-python-round2.json has passed/stderr

# Round 3: refine remaining hard samples
bash scripts/infer_gmm_refine.sh \
  --prev_results_dir ./inference_results \
  --round_num 3
```
