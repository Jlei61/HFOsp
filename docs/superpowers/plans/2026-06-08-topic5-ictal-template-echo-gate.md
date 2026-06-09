# Topic 5 Ictal-Template-Echo Gate (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage-1 "proxy triage" that pools, across subjects, whether each seizure's ER/atlas-derived ictal channel ordering echoes **any of the subject's stable masked interictal propagation templates (generic template echo, spec v4 §3.2 — NOT restricted to swap subjects)** — replacing the power-floored per-subject contingency (Q1/Q1') with subject-level pooling.

**Architecture:** One pure-math module (`src/topic5_echo_gate.py`, no I/O) holding the echo statistic, the shuffle nulls, the between-subject control, the leave-one-seizure-out de-anchor, subject-level pooling, and atlas-quality; one runner that loads masked templates **via `results/interictal_propagation_masked/`** + ER atlas ranks **via `src.atlas_loading`**, runs the B0 audit, computes per-subject + cohort JSON; one plotter. **P0 invariants:** templates phantom-safe masked (§3.6, **1-D template ≠ `mask_phantom_ranks` 2-D event matrix — see Task 8**); atlas via canonical loader (not a hand-rolled path); proxy triage never vetoes Stage 2 (P0-1, reporting wording + lint guard); bad-data regression uses **real null draws** (P0-C); de-anchor keeps `max_m` over templates (P1-A); MIN_CH=8 locked (P1-1).

**Tech Stack:** Python, numpy, scipy.stats (spearmanr, wilcoxon, binomtest, rankdata), statsmodels (cluster-robust OLS sensitivity). pytest for TDD. Matplotlib for figures.

**Spec:** `docs/superpowers/specs/2026-06-08-topic5-ictal-template-echo-gate-design.md` (committed a713ac8, **v4**). Re-read each referenced spec section at the matching task boundary (CLAUDE.md §5).

---

## File Structure

- **Create `src/topic5_echo_gate.py`** — pure functions, no file I/O. Responsibilities: `spearman_common` (masked, common-channel Spearman); `echo_r_obs` (max over k templates); `_block_permute` + `shuffle_null` (null modes A/B/C §4.6, B fail-closed on unequal shafts); `between_subject_control` (Null D §4.6); `compute_echo_strength` (e_k/p_k/quantiles **+ `e_k_baddata` real null draw** §4.1/§4.4); `loo_anchor` + `compute_deanchor_echo` (**`max_m` over templates** §4.1b) + `anchor_reliability`; `pool_echo_subject_level` (Wilcoxon/sign/bootstrap primary + cluster-robust sensitivity §4.1.4); `bad_data_regression` (pools `e_k_baddata` §4.4); `compute_atlas_quality` (§3.5).
- **Create `scripts/run_topic5_echo_gate.py`** — I/O + orchestration. `audit` / `per-subject` / `cohort` / `figures`. Masked templates from `results/interictal_propagation_masked/` (§3.6); **ER atlas via `src.atlas_loading.load_per_subject_json` / `list_cohort_subjects` / `REQUIRED_SCHEMA` — do NOT hand-roll an atlas root**. Applies `_apply_masked_paths()` path-swap.
- **Create `scripts/plot_topic5_echo_gate.py`** — 3 figures + `figures/README.md`.
- **Create `tests/test_topic5_echo_gate.py`** — unit tests on synthetic data with known answers.

> **Note on test counts:** the `Expected: PASS (N tests)` numbers below are cumulative and indicative; the real gate is `pytest tests/test_topic5_echo_gate.py -v` coming back fully green, not the exact integer (which shifts as tasks add tests).

**Cohort gate (spec v4 §3.1/§3.2 — generic template echo):** subject ∈ **primary** ⇔ has a phantom-safe masked **stable template (k=2 primary; k=1/k>2 → sensitivity)** AND a v2.3 ictal atlas with `n_seizures_with_atlas ≥ 2` passing atlas-quality + construct-validity. **`swap_class` is a pre-registered STRATIFIER, not a primary gate** (strict/candidate vs none = planned subgroup). **Negative control = between-subject template control (Null D) + bad-data regression**, NOT the none-subset (none subjects have legitimate templates and are expected to echo). `MIN_CH=8` locked.

---

## Task 1: Module scaffold + `spearman_common` (masked, common-channel)

Re-read spec §3.4 + §3.6 before this task (masked-rank contract: NaN/masked channels never enter ρ).

**Files:**
- Create: `src/topic5_echo_gate.py`
- Test: `tests/test_topic5_echo_gate.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
from src.topic5_echo_gate import spearman_common


def test_spearman_common_identical_and_reverse():
    a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert spearman_common(a, a, min_ch=8) == pytest.approx(1.0)
    assert spearman_common(a, a[::-1].copy(), min_ch=8) == pytest.approx(-1.0)


def test_spearman_common_too_few_returns_nan():
    a = np.array([0.0, 1.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan])
    b = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    # only 3 common finite -> below min_ch
    assert np.isnan(spearman_common(a, b, min_ch=8))


def test_spearman_common_phantom_channel_excluded():
    # template has NaN (masked phantom) at index 7 even though seizure has a value there.
    # Result must equal Spearman over indices 0..6 only (phantom never enters).
    templ = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan])
    seiz = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0])
    from scipy.stats import spearmanr
    expected = spearmanr(seiz[:7], templ[:7]).statistic
    assert spearman_common(seiz, templ, min_ch=6) == pytest.approx(expected)


def test_spearman_common_shape_mismatch_raises():
    with pytest.raises(ValueError):
        spearman_common(np.zeros(8), np.zeros(7), min_ch=5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_echo_gate.py -v`
Expected: FAIL with `ImportError: cannot import name 'spearman_common'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Topic 5 Stage-1 ictal-template-echo gate (proxy triage).

PURE math, no file I/O. Spec:
docs/superpowers/specs/2026-06-08-topic5-ictal-template-echo-gate-design.md

Phantom-safety (§3.6): templates arrive masked — non-participating channels are
NaN. Every Spearman here drops NaN on either side, so a phantom can never enter
the correlation. "full participating-channel set" = both-finite intersection.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr


def spearman_common(rank_a, rank_b, *, min_ch: int) -> float:
    """Spearman rho on channels finite in BOTH vectors; NaN if fewer than min_ch."""
    a = np.asarray(rank_a, dtype=float)
    b = np.asarray(rank_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"rank vectors must align by channel index: {a.shape} vs {b.shape}")
    common = np.isfinite(a) & np.isfinite(b)
    if int(common.sum()) < min_ch:
        return float("nan")
    rho = spearmanr(a[common], b[common]).statistic
    return float(rho) if np.isfinite(rho) else float("nan")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_echo_gate.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_echo_gate.py tests/test_topic5_echo_gate.py
git commit -m "feat(topic5-echo): module scaffold + masked spearman_common"
```

---

## Task 2: `echo_r_obs` — max over k templates

Re-read spec §4.1 (k handling: k=1 → single ρ; k=2 → max(ρ_a,ρ_b); k>2 → max over k; null takes the same max).

**Files:**
- Modify: `src/topic5_echo_gate.py`
- Test: `tests/test_topic5_echo_gate.py`

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_echo_gate import echo_r_obs


def test_echo_r_obs_takes_best_matching_template():
    base = np.arange(8, dtype=float)
    t0 = base.copy()                 # unrelated-ish (reverse of seizure)
    t1 = base.copy()
    seizure = base[::-1].copy()      # matches t0 reversed -> rho=-1 to both; max=-1
    # make t1 match the seizure exactly
    t1 = seizure.copy()
    r = echo_r_obs(seizure, [t0, t1], min_ch=8)
    assert r == pytest.approx(1.0)   # best template (t1) wins


def test_echo_r_obs_single_template_k1():
    base = np.arange(8, dtype=float)
    assert echo_r_obs(base, [base.copy()], min_ch=8) == pytest.approx(1.0)


def test_echo_r_obs_all_insufficient_returns_nan():
    a = np.array([0.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    t = np.arange(8, dtype=float)
    assert np.isnan(echo_r_obs(a, [t], min_ch=8))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_echo_gate.py::test_echo_r_obs_takes_best_matching_template -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
def echo_r_obs(seizure_rank, template_ranks: Sequence, *, min_ch: int) -> float:
    """max_m Spearman(seizure, template_m) over templates with enough overlap; NaN if none."""
    rhos = []
    for t in template_ranks:
        r = spearman_common(seizure_rank, t, min_ch=min_ch)
        if np.isfinite(r):
            rhos.append(r)
    if not rhos:
        return float("nan")
    return float(max(rhos))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_echo_gate.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_echo_gate.py tests/test_topic5_echo_gate.py
git commit -m "feat(topic5-echo): echo_r_obs max-over-templates"
```

---

## Task 3: Null shuffle modes (`_block_permute` + `shuffle_null`)

Re-read spec §4.6. within_shaft / anchor_matched = within-block permute; shaft_block = between-block permute; channel = single all-encompassing block.

**Files:**
- Modify: `src/topic5_echo_gate.py`
- Test: `tests/test_topic5_echo_gate.py`

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_echo_gate import shuffle_null


def test_channel_shuffle_destroys_cross_shaft_order_but_within_shaft_preserves():
    # 2 shafts x 4 channels. Template = global ascending. Seizure = same GLOBAL order
    # but within each shaft the order is FLAT-matched only by shaft mean (pure anchor):
    # seizure equals template exactly -> r_obs=1. Under within_shaft shuffle the
    # within-shaft order is the only thing scrambled; since each shaft's 4 values are
    # contiguous, within-shaft permutation barely changes global Spearman -> null high
    # (e_k ~ small). Under channel shuffle, cross-shaft order is destroyed -> null low.
    templ = np.arange(8, dtype=float)
    seizure = np.arange(8, dtype=float)
    shafts = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    rng = np.random.default_rng(0)
    null_chan = shuffle_null(seizure, [templ], B=500, rng=rng, null_mode="channel", min_ch=8)
    rng = np.random.default_rng(0)
    null_within = shuffle_null(seizure, [templ], B=500, rng=rng,
                               null_mode="within_shaft", blocks=shafts, min_ch=8)
    null_chan = null_chan[np.isfinite(null_chan)]
    null_within = null_within[np.isfinite(null_within)]
    # within-shaft null stays much closer to r_obs=1 than channel null
    assert np.nanmean(null_within) > np.nanmean(null_chan) + 0.2


def test_shaft_block_requires_blocks():
    with pytest.raises(ValueError):
        shuffle_null(np.arange(8.0), [np.arange(8.0)], B=10,
                     rng=np.random.default_rng(0), null_mode="within_shaft", min_ch=8)


def test_shaft_block_capacity_fail_closed_on_unequal_shafts():
    from src.topic5_echo_gate import shaft_block_capacity
    # 3 shafts of sizes 4, 3, 2 -> NO two shafts share a size -> nothing exchangeable.
    blocks = np.array(["A", "A", "A", "A", "B", "B", "B", "C", "C"])
    cap = shaft_block_capacity(blocks)
    assert cap["n_exchangeable_channels"] == 0
    assert cap["insufficient_block_exchange"] is True


def test_shaft_block_capacity_ok_when_two_equal_shafts():
    blocks = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])  # 2 shafts of size 4
    cap = shaft_block_capacity(blocks)
    assert cap["n_exchangeable_channels"] == 8
    assert cap["insufficient_block_exchange"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_echo_gate.py::test_channel_shuffle_destroys_cross_shaft_order_but_within_shaft_preserves -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
from collections import defaultdict

_NULL_MODE_KIND = {
    "channel": "within",
    "within_shaft": "within",
    "anchor_matched": "within",
    "shaft_block": "between",
}


def _block_permute(values: np.ndarray, blocks: np.ndarray, kind: str, rng) -> np.ndarray:
    """within: permute values inside each block. between: swap whole equal-size blocks."""
    values = np.asarray(values, dtype=float)
    blocks = np.asarray(blocks)
    out = values.copy()
    uniq = [b for b in np.unique(blocks) if b is not None and b == b]  # drop None/NaN
    if kind == "within":
        for b in uniq:
            idx = np.where(blocks == b)[0]
            out[idx] = values[idx][rng.permutation(len(idx))]
    elif kind == "between":
        by_size = defaultdict(list)
        for b in uniq:
            idx = np.where(blocks == b)[0]
            by_size[len(idx)].append(idx)
        for size, idx_list in by_size.items():
            if len(idx_list) < 2:
                continue
            src_vals = [values[ix].copy() for ix in idx_list]
            perm = rng.permutation(len(idx_list))
            for tgt_pos, src_pos in enumerate(perm):
                out[idx_list[tgt_pos]] = src_vals[src_pos]
    else:
        raise ValueError(f"unknown kind {kind}")
    return out


def shuffle_null(seizure_rank, template_ranks, *, B, rng, null_mode, min_ch, blocks=None):
    """Null distribution of echo_r_obs under the requested channel-label shuffle."""
    kind = _NULL_MODE_KIND[null_mode]
    n = len(np.asarray(seizure_rank))
    if null_mode == "channel":
        blk = np.zeros(n, dtype=int)
    else:
        if blocks is None:
            raise ValueError(f"null_mode={null_mode} requires blocks")
        blk = np.asarray(blocks)
    out = np.empty(B, dtype=float)
    for i in range(B):
        shuf = _block_permute(np.asarray(seizure_rank, float), blk, kind, rng)
        out[i] = echo_r_obs(shuf, template_ranks, min_ch=min_ch)
    return out


def shaft_block_capacity(blocks) -> Dict:
    """How many channels CAN be block-exchanged (shafts that share a length with
    >=1 other shaft). Unequal shafts are NOT exchangeable and stay put (§4.6 P1-B).
    insufficient_block_exchange=True means shaft_block null is degenerate for this
    subject and must NOT be reported as a real null — fail closed."""
    blocks = np.asarray(blocks)
    uniq = [b for b in np.unique(blocks) if b is not None and b == b]
    sizes = defaultdict(list)
    for b in uniq:
        sizes[int(np.sum(blocks == b))].append(b)
    n_exch = sum(size * len(grp) for size, grp in sizes.items() if len(grp) >= 2)
    total = sum(int(np.sum(blocks == b)) for b in uniq)
    return {"n_exchangeable_channels": int(n_exch), "n_total_channels": int(total),
            "insufficient_block_exchange": bool(n_exch < 2)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_echo_gate.py -v`
Expected: PASS (11 tests). Note: runner records `insufficient_block_exchange` per subject; when True, shaft_block is reported `inconclusive` for that subject and does NOT enter §4.3 hard constraints (spec §4.6 P1-B).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_echo_gate.py tests/test_topic5_echo_gate.py
git commit -m "feat(topic5-echo): 5 shuffle null modes (channel/within-shaft/shaft-block/anchor-matched)"
```

---

## Task 4: `compute_echo_strength` (e_k, p_k, quantiles)

Re-read spec §4.1 step 3 (e_k = standardized exceedance; p_k = one-sided percentile; store quantiles; drop NaN null draws).

**Files:**
- Modify: `src/topic5_echo_gate.py`
- Test: `tests/test_topic5_echo_gate.py`

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_echo_gate import compute_echo_strength


def test_echo_strength_positive_for_matching_seizure():
    templ = np.arange(12, dtype=float)
    seizure = np.arange(12, dtype=float)          # perfect echo
    res = compute_echo_strength(seizure, [templ], B=1000,
                                rng=np.random.default_rng(1), min_ch=8)
    assert res["r_obs"] == pytest.approx(1.0)
    assert res["e_k"] > 3.0          # far above null
    assert res["p_k"] < 0.01


def test_echo_strength_null_for_random_seizure():
    templ = np.arange(12, dtype=float)
    rng = np.random.default_rng(2)
    seizure = rng.permutation(12).astype(float)   # random order
    res = compute_echo_strength(seizure, [templ], B=1000, rng=rng, min_ch=8)
    assert abs(res["e_k"]) < 2.5
    assert 0.02 < res["p_k"] < 0.98


def test_echo_strength_insufficient_returns_nan_record():
    a = np.array([0.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    res = compute_echo_strength(a, [np.arange(8.0)], B=100,
                                rng=np.random.default_rng(3), min_ch=8)
    assert np.isnan(res["e_k"]) and res["n_null"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_echo_gate.py::test_echo_strength_positive_for_matching_seizure -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
def compute_echo_strength(seizure_rank, template_ranks, *, B, rng, min_ch,
                          null_mode="channel", blocks=None) -> Dict:
    r_obs = echo_r_obs(seizure_rank, template_ranks, min_ch=min_ch)
    if not np.isfinite(r_obs):
        return {"e_k": float("nan"), "p_k": float("nan"), "r_obs": float("nan"),
                "null_mean": float("nan"), "null_sd": float("nan"),
                "null_q": [float("nan")] * 3, "n_null": 0, "null_mode": null_mode}
    null = shuffle_null(seizure_rank, template_ranks, B=B, rng=rng,
                        null_mode=null_mode, min_ch=min_ch, blocks=blocks)
    null = null[np.isfinite(null)]
    if null.size < 2:
        return {"e_k": float("nan"), "p_k": float("nan"), "r_obs": float(r_obs),
                "null_mean": float("nan"), "null_sd": float("nan"),
                "null_q": [float("nan")] * 3, "n_null": int(null.size), "null_mode": null_mode}
    sd = float(null.std(ddof=1))
    e_k = float((r_obs - null.mean()) / sd) if sd > 0 else float("nan")
    p_k = float((np.sum(null >= r_obs) + 1) / (null.size + 1))
    # P0-C bad-data regression: a REAL null draw used as a fake observation,
    # standardized against the REST of the null. By construction E[.]~0, so the
    # cohort pool of e_k_baddata must come out non-significant (Task 6). This is
    # NOT a random N(0,1) re-roll — it reuses this seizure's own null geometry.
    if null.size >= 3:
        rest = null[1:]
        rsd = float(rest.std(ddof=1))
        e_k_baddata = float((null[0] - rest.mean()) / rsd) if rsd > 0 else float("nan")
    else:
        e_k_baddata = float("nan")
    return {"e_k": e_k, "p_k": p_k, "r_obs": float(r_obs),
            "e_k_baddata": e_k_baddata,
            "null_mean": float(null.mean()), "null_sd": sd,
            "null_q": [float(q) for q in np.quantile(null, [0.05, 0.5, 0.95])],
            "n_null": int(null.size), "null_mode": null_mode}
```

Also add `"e_k_baddata": float("nan")` to BOTH early-return dicts (the not-finite and the `null.size < 2` branches) so the schema is uniform.

- [ ] **Step 4: Add a test pinning the bad-data field**

```python
def test_echo_strength_baddata_field_is_centered_draw():
    templ = np.arange(12, dtype=float)
    seizure = np.arange(12, dtype=float)
    res = compute_echo_strength(seizure, [templ], B=2000,
                                rng=np.random.default_rng(4), min_ch=8)
    # e_k_baddata is a single standardized null draw -> finite, not the huge e_k
    assert np.isfinite(res["e_k_baddata"])
    assert abs(res["e_k_baddata"]) < res["e_k"]      # fake obs is unremarkable vs real
```

Run: `pytest tests/test_topic5_echo_gate.py -v`
Expected: PASS (13 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_echo_gate.py tests/test_topic5_echo_gate.py
git commit -m "feat(topic5-echo): compute_echo_strength e_k/p_k/quantiles"
```

---

## Task 5: LOO de-anchor + reliability (§4.1b leakage fix)

Re-read spec §4.1b. `r̄_{c,−k}` MUST exclude seizure k (no leakage). `n_seizures ≥ 4` primary, =3 exploratory.

**Files:**
- Modify: `src/topic5_echo_gate.py`
- Test: `tests/test_topic5_echo_gate.py`

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_echo_gate import loo_anchor, compute_deanchor_echo, anchor_reliability


def test_loo_anchor_excludes_current_seizure_no_leakage():
    # 3 seizures x 4 channels. Anchor for seizure 0 must be mean of seizures 1,2 ONLY,
    # independent of seizure 0's (here extreme) values.
    M = np.array([
        [99.0, 99.0, 99.0, 99.0],   # seizure 0 extreme
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
    ])
    anc = loo_anchor(M)
    assert np.allclose(anc[0], [0.0, 1.0, 2.0, 3.0])     # seizure 0 NOT in its own anchor
    # changing seizure 0 must not change anchor[0]
    M2 = M.copy(); M2[0] = [-5.0, -5.0, -5.0, -5.0]
    assert np.allclose(loo_anchor(M2)[0], anc[0])


def test_loo_anchor_ignores_nan_in_other_seizures():
    M = np.array([
        [0.0, 1.0, 2.0, 3.0],
        [np.nan, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, np.nan],
    ])
    anc = loo_anchor(M)
    # anchor[0] channel0 = mean of seizure1(nan), seizure2(0.0) -> 0.0
    assert anc[0][0] == pytest.approx(0.0)


def test_anchor_reliability_high_when_orders_agree():
    M = np.array([[0.0, 1, 2, 3, 4], [0.0, 1, 2, 3, 4], [0.0, 1, 2, 3, 4]])
    assert anchor_reliability(M) > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_echo_gate.py::test_loo_anchor_excludes_current_seizure_no_leakage -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
def loo_anchor(per_seizure_ranks) -> np.ndarray:
    """r_bar_{c,-k}: per-channel mean rank over OTHER seizures (current k excluded)."""
    M = np.asarray(per_seizure_ranks, dtype=float)
    ns = M.shape[0]
    out = np.full_like(M, np.nan)
    for k in range(ns):
        others = np.delete(M, k, axis=0)
        with np.errstate(invalid="ignore"):
            col_mean = np.where(np.all(np.isnan(others), axis=0), np.nan,
                                np.nanmean(others, axis=0))
        out[k] = col_mean
    return out


def compute_deanchor_echo(per_seizure_ranks, template_ranks, *, B, rng, min_ch) -> List[Dict]:
    """Echo on de-anchored deltas, keeping max-over-templates (P1-A — same contract
    as the primary §4.1, so k=2 subjects are not arbitrarily dominated by template 0):
    delta_seiz = seiz_k - r_bar_{-k}; delta_templ_m = template_m - r_bar_{-k}.
    compute_echo_strength already takes max_m over the delta-template list."""
    M = np.asarray(per_seizure_ranks, dtype=float)
    templs = [np.asarray(t, dtype=float) for t in template_ranks]
    anc = loo_anchor(M)
    records = []
    for k in range(M.shape[0]):
        d_seiz = M[k] - anc[k]
        d_templs = [t - anc[k] for t in templs]      # de-anchor EACH template
        records.append(compute_echo_strength(d_seiz, d_templs, B=B, rng=rng, min_ch=min_ch))
    return records


def anchor_reliability(per_seizure_ranks) -> float:
    """Kendall's W (coefficient of concordance) across seizures over channels."""
    M = np.asarray(per_seizure_ranks, dtype=float)
    # rank channels within each seizure (ignoring NaN columns shared by all)
    valid = ~np.any(np.isnan(M), axis=0)
    if valid.sum() < 3 or M.shape[0] < 2:
        return float("nan")
    sub = M[:, valid]
    from scipy.stats import rankdata
    R = np.vstack([rankdata(row) for row in sub])         # n_seiz x n_ch ranks
    n, m = R.shape                                        # n raters, m items
    Rj = R.sum(axis=0)
    S = np.sum((Rj - Rj.mean()) ** 2)
    W = 12 * S / (n ** 2 * (m ** 3 - m))
    return float(W)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_echo_gate.py -v`
Expected: PASS (15 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_echo_gate.py tests/test_topic5_echo_gate.py
git commit -m "feat(topic5-echo): LOO de-anchor (no leakage) + Kendall-W reliability"
```

---

## Task 6: `pool_echo_subject_level` + bad-data regression (§4.1.4 / §4.4)

Re-read spec §4.1.4 (subject-level primary: E_s = mean_k e_k → Wilcoxon signed-rank one-sided; sign test; bootstrap) and §4.4 (bad-data regression must return null).

**Files:**
- Modify: `src/topic5_echo_gate.py`
- Test: `tests/test_topic5_echo_gate.py`

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_echo_gate import pool_echo_subject_level


def _records(per_subject_es):
    out = []
    for sid, evals in per_subject_es.items():
        for e in evals:
            out.append({"subject": sid, "e_k": e})
    return out


def test_pool_positive_when_subjects_consistently_positive():
    recs = _records({f"s{i}": [0.8, 1.1, 0.9] for i in range(12)})
    res = pool_echo_subject_level(recs)
    assert res["n_subjects"] == 12
    assert res["median_E_s"] > 0
    assert res["wilcoxon_p_onesided"] < 0.05


def test_pool_sanity_centered_zero_not_significant():
    rng = np.random.default_rng(7)
    recs = _records({f"s{i}": list(rng.normal(0, 1, 3)) for i in range(12)})
    res = pool_echo_subject_level(recs)
    assert res["wilcoxon_p_onesided"] > 0.10        # generic pooling sanity


def test_bad_data_regression_real_null_draw_flattens(monkeypatch):
    # P0-C: real records carry BOTH a strong e_k and an e_k_baddata (real null draw).
    # The primary pool on e_k must be significant; the bad-data pool on e_k_baddata
    # must NOT — using the actual null-draw field, NOT a re-rolled N(0,1).
    from src.topic5_echo_gate import compute_echo_strength, bad_data_regression
    templ = np.arange(12, dtype=float)
    records = []
    for i in range(12):
        rng = np.random.default_rng(100 + i)
        # mild real echo (seizure ~ template + small noise on ranks)
        seiz = np.argsort(np.arange(12) + rng.normal(0, 0.6, 12)).astype(float)
        res = compute_echo_strength(seiz, [templ], B=1500, rng=rng, min_ch=8)
        res["subject"] = f"s{i}"
        records.append(res)
    primary = pool_echo_subject_level([{"subject": r["subject"], "e_k": r["e_k"]} for r in records])
    bad = bad_data_regression(records)
    assert primary["wilcoxon_p_onesided"] < 0.05         # real echo survives
    assert bad["wilcoxon_p_onesided"] > 0.10             # fake (null-draw) obs flattens
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_echo_gate.py::test_pool_positive_when_subjects_consistently_positive -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
def pool_echo_subject_level(records, *, n_boot: int = 2000, seed: int = 0) -> Dict:
    """Subject-level primary pooling. records: list of {subject, e_k}.
    E_s = mean over a subject's finite e_k. One-sided (>0) Wilcoxon signed-rank +
    sign test + bootstrap CI on median(E_s)."""
    from collections import defaultdict
    from scipy.stats import wilcoxon
    by_subj = defaultdict(list)
    for r in records:
        if r.get("e_k") is not None and np.isfinite(r["e_k"]):
            by_subj[r["subject"]].append(float(r["e_k"]))
    Es = np.array([np.mean(v) for v in by_subj.values() if v], dtype=float)
    n = int(Es.size)
    out = {"n_subjects": n, "E_s": Es.tolist(),
           "median_E_s": float(np.median(Es)) if n else float("nan"),
           "mean_E_s": float(np.mean(Es)) if n else float("nan")}
    if n < 2 or np.allclose(Es, 0):
        out["wilcoxon_p_onesided"] = float("nan")
        out["sign_p_onesided"] = float("nan")
        out["boot_ci95"] = [float("nan"), float("nan")]
        return out
    try:
        out["wilcoxon_p_onesided"] = float(wilcoxon(Es, alternative="greater").pvalue)
    except ValueError:
        out["wilcoxon_p_onesided"] = float("nan")
    n_pos = int(np.sum(Es > 0))
    from scipy.stats import binomtest
    out["sign_p_onesided"] = float(binomtest(n_pos, n, 0.5, alternative="greater").pvalue)
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(Es, size=n, replace=True)) for _ in range(n_boot)]
    out["boot_ci95"] = [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))]
    return out


def bad_data_regression(echo_records) -> Dict:
    """P0-C: pool the e_k_baddata field (each a REAL null draw used as a fake
    observation) exactly like the primary pool. Must come out non-significant; a
    significant result means the pooling machinery manufactures signal -> stop & fix.
    echo_records: list of compute_echo_strength dicts that carry 'subject' + 'e_k_baddata'."""
    recs = [{"subject": r["subject"], "e_k": r.get("e_k_baddata")} for r in echo_records]
    return pool_echo_subject_level(recs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_echo_gate.py -v`
Expected: PASS (20 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_echo_gate.py tests/test_topic5_echo_gate.py
git commit -m "feat(topic5-echo): subject-level pooling (Wilcoxon/sign/bootstrap) + bad-data regression"
```

---

## Task 7: `compute_atlas_quality` (§3.5)

Re-read spec §3.5 (atlas-quality flags from tie fraction / dynamic range / n_channels). Construct-validity sentinel is a runner-level manual flag (Task 9), NOT computed here.

**Files:**
- Modify: `src/topic5_echo_gate.py`
- Test: `tests/test_topic5_echo_gate.py`

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_echo_gate import compute_atlas_quality


def test_atlas_quality_pass_for_clean_ranks():
    # distinct ranks across many channels -> low ties, full dynamic range
    rank = np.arange(12, dtype=float)
    q = compute_atlas_quality(rank, tie_max=0.3, min_channels=8)
    assert q["atlas_quality_flag"] == "pass"
    assert q["rank_tie_fraction"] == pytest.approx(0.0)


def test_atlas_quality_fail_for_mostly_tied_ranks():
    rank = np.array([1.0] * 10 + [2.0, 3.0])      # 10/12 tied
    q = compute_atlas_quality(rank, tie_max=0.3, min_channels=8)
    assert q["atlas_quality_flag"] == "fail"


def test_atlas_quality_fail_for_too_few_channels():
    rank = np.array([0.0, 1.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan])
    q = compute_atlas_quality(rank, tie_max=0.3, min_channels=8)
    assert q["atlas_quality_flag"] == "fail"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_echo_gate.py::test_atlas_quality_pass_for_clean_ranks -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
def compute_atlas_quality(ictal_rank, *, tie_max: float, min_channels: int) -> Dict:
    r = np.asarray(ictal_rank, dtype=float)
    finite = r[np.isfinite(r)]
    n = int(finite.size)
    if n == 0:
        return {"atlas_quality_flag": "fail", "rank_tie_fraction": float("nan"),
                "rank_dynamic_range": 0.0, "n_ranked_channels": 0}
    _, counts = np.unique(finite, return_counts=True)
    tie_frac = float(np.sum(counts[counts > 1]) / n)
    dyn = float(finite.max() - finite.min())
    ok = (n >= min_channels) and (tie_frac <= tie_max) and (dyn > 0)
    return {"atlas_quality_flag": "pass" if ok else "fail",
            "rank_tie_fraction": tie_frac, "rank_dynamic_range": dyn,
            "n_ranked_channels": n}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_echo_gate.py -v`
Expected: PASS (20 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_echo_gate.py tests/test_topic5_echo_gate.py
git commit -m "feat(topic5-echo): compute_atlas_quality flags"
```

---

## Task 7b: `between_subject_control` (Null D — formal negative control, §4.6 / spec v4 §3.2)

Re-read spec v4 §3.2 + §4.6 Null D. Under generic-template-echo scope this is the **primary negative control** (replaces the none-subset). For each subject, recompute echo using OTHER subjects' templates remapped to this subject's channel count; the pooled effect must be ~neutral. A significant between-subject echo means the statistic is too coarse → primary conclusion void.

**Files:**
- Modify: `src/topic5_echo_gate.py`
- Test: `tests/test_topic5_echo_gate.py`

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_echo_gate import between_subject_control


def test_between_subject_control_neutral_for_unrelated_templates():
    rng = np.random.default_rng(11)
    # subject's seizure echoes ITS OWN template; OTHER templates are random orders.
    own = np.arange(12, dtype=float)
    seizure = np.arange(12, dtype=float)
    other_templates = [rng.permutation(12).astype(float) for _ in range(8)]
    res = between_subject_control(seizure, other_templates, B=800, rng=rng, min_ch=8)
    # echo against unrelated templates should be unremarkable (|e| small, p mid)
    assert abs(res["e_k"]) < 2.5
    assert 0.02 < res["p_k"] < 0.98
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_echo_gate.py::test_between_subject_control_neutral_for_unrelated_templates -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
def between_subject_control(seizure_rank, other_subject_templates, *, B, rng, min_ch,
                            null_mode="channel", blocks=None) -> Dict:
    """Null D: echo of this seizure against OTHER subjects' templates (each remapped
    to this seizure's channel count by truncation/identity ordering). Returns the same
    record shape as compute_echo_strength so it pools identically. The pooled result
    across the cohort is the formal negative control under generic-echo scope."""
    n = len(np.asarray(seizure_rank))
    remapped = []
    for t in other_subject_templates:
        t = np.asarray(t, dtype=float)
        if t.size >= n:
            remapped.append(t[:n])
        else:
            pad = np.full(n - t.size, np.nan)
            remapped.append(np.concatenate([t, pad]))
    return compute_echo_strength(seizure_rank, remapped, B=B, rng=rng, min_ch=min_ch,
                                 null_mode=null_mode, blocks=blocks)
```

(Note: the runner builds `other_subject_templates` from the cohort's OTHER subjects' masked templates; remapping by truncation is a deliberately coarse surrogate — its only job is to show that an arbitrary template does NOT echo. If it DOES, the statistic is too coarse and primary is void.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_echo_gate.py -v`
Expected: PASS (suite green).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_echo_gate.py tests/test_topic5_echo_gate.py
git commit -m "feat(topic5-echo): between_subject_control (Null D negative control)"
```

---

## Task 8: Runner — masked template + ictal-rank loader (§3.6 hard contract)

Re-read spec §3.6 BEFORE writing this. **Phantom-safe contract: templates from `results/interictal_propagation_masked/`, each cluster carries its own `valid_mask`; the old `topic1_topic5_bridge` loader / `q1prime_per_subject` JSON `template_rank` are FORBIDDEN as primary.** This is the highest-risk task — verify the masked loader actually returns per-cluster valid masks before trusting it.

**Files:**
- Create: `scripts/run_topic5_echo_gate.py`
- Test: `tests/test_topic5_echo_gate.py` (loader-contract test with a fixture dir)

- [ ] **Step 1: Verify the masked source exists and inspect its shape**

Run:
```bash
ls results/interictal_propagation_masked/ | head
python -c "import json,glob; f=sorted(glob.glob('results/interictal_propagation_masked/**/*.json',recursive=True))[0]; d=json.load(open(f)); print(f); print(list(d.keys())[:20])"
```
Expected: a per-subject JSON exists; print its top-level keys. **Record** which key holds the per-cluster template rank and which holds the raw cluster `bools`/`valid_mask`. If no masked tree exists, STOP and ask the user (do not fall back to the unmasked loader).

- [ ] **Step 2: Write the failing loader-contract test**

```python
def test_masked_template_rank_1d_uses_valid_mask_not_helper():
    # P0-A: a 1-D ALREADY-AGGREGATED template rank + per-cluster valid_mask must be
    # masked with np.where — NOT passed to mask_phantom_ranks (which is a 2-D
    # (n_ch, n_ev) event matrix re-ranker and would raise/re-rank).
    import scripts.run_topic5_echo_gate as R
    agg_rank = np.array([0.0, 1.0, 7.0, 2.0])        # idx2 carries a phantom value
    valid_mask = np.array([True, True, False, True]) # idx2 non-participating
    templ = R.masked_template_rank_1d(agg_rank, valid_mask)
    assert np.isnan(templ[2])                        # phantom excluded
    assert np.allclose(templ[[0, 1, 3]], agg_rank[[0, 1, 3]])


def test_rebuild_template_from_events_2d_uses_helper():
    # P0-A: the OTHER contract — event-level (n_ch, n_ev) raw ranks + bools go through
    # mask_phantom_ranks (per-event re-rank, phantom discarded), then aggregate to a
    # 1-D template. Non-participating-everywhere channel -> NaN in the template.
    import scripts.run_topic5_echo_gate as R
    raw = np.array([[0.0, 1.0], [1.0, 0.0], [9.0, 9.0], [2.0, 2.0]])   # (4 ch, 2 ev)
    bools = np.array([[True, True], [True, True], [False, False], [True, True]])
    templ = R.rebuild_template_from_events(raw, bools)
    assert np.isnan(templ[2])                        # ch2 never participates -> NaN
    assert np.all(np.isfinite(templ[[0, 1, 3]]))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_topic5_echo_gate.py::test_load_masked_template_excludes_phantom -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError`.

- [ ] **Step 4: Write minimal implementation**

```python
"""Topic 5 Stage-1 echo gate runner (proxy triage).

Spec: docs/superpowers/specs/2026-06-08-topic5-ictal-template-echo-gate-design.md

P0 invariants enforced here:
- §3.6 phantom-safe: templates ONLY from results/interictal_propagation_masked/,
  each cluster's non-participating channels masked to NaN. The old unmasked
  bridge loader is NOT used for primary templates.
- P0-1 proxy-triage wording: this runner NEVER writes "Stage 2 暂缓"; verdicts
  only set Stage-2 PRIORITY.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.lagpat_rank_audit import mask_phantom_ranks       # 2-D event matrix ONLY
from src.propagation_skeleton_geometry import parse_shaft
from src import atlas_loading                              # P0-B canonical atlas loader
from src import topic5_echo_gate as echo

MASKED_ROOT = Path("results/interictal_propagation_masked")
# NOTE: atlas JSONs come from src.atlas_loading (PER_SUBJECT_DIR =
# results/data_driven_soz/layer_a_ictal_er_rank/per_subject). Do NOT point at
# .../atlas_v2_3 — that is the FIGURES dir, not the data (P0-B).
OUT_ROOT = Path("results/topic5_ictal_template_echo")
MIN_CH = 8           # P1-1 locked
B = 2000             # §5.3 lock
TIE_MAX = 0.3
RNG_SEED = 20260608


def masked_template_rank_1d(agg_rank, valid_mask):
    """P0-A contract #1: a 1-D ALREADY-AGGREGATED template rank + per-cluster
    valid_mask. Mask with np.where — do NOT call mask_phantom_ranks (that helper
    is a 2-D (n_ch, n_ev) per-event re-ranker; passing 1-D is wrong)."""
    r = np.asarray(agg_rank, dtype=float)
    m = np.asarray(valid_mask, dtype=bool)
    if r.shape != m.shape:
        raise ValueError(f"agg_rank {r.shape} != valid_mask {m.shape}")
    return np.where(m, r, np.nan)


def rebuild_template_from_events(raw_ranks_2d, bools_2d):
    """P0-A contract #2: event-level (n_ch, n_ev) raw ranks + bools. Run
    mask_phantom_ranks (per-event re-rank, phantom discarded -> NaN), then aggregate
    to a 1-D template by nanmean across events. Channel never participating -> NaN."""
    masked = np.asarray(mask_phantom_ranks(np.asarray(raw_ranks_2d, float),
                                           np.asarray(bools_2d, bool)), dtype=float)
    with np.errstate(invalid="ignore"):
        templ = np.where(np.all(np.isnan(masked), axis=1), np.nan,
                         np.nanmean(masked, axis=1))
    return templ
```

**Which contract does the masked tree need?** Decide from Task 8 Step 1: if the masked JSON already stores a 1-D per-cluster aggregated `template_rank` + a `valid_mask`/`bools`, use `masked_template_rank_1d`. If it stores event-level `(n_ch, n_ev)` ranks + bools, use `rebuild_template_from_events`. **Do not mix them** — they are different contracts.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_topic5_echo_gate.py -k "masked_template_rank_1d or rebuild_template_from_events" -v`
Expected: PASS (both contracts). `mask_phantom_ranks` is verified by the 2-D test to require an (n_ch, n_ev) matrix — the 1-D path deliberately never calls it.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_topic5_echo_gate.py tests/test_topic5_echo_gate.py
git commit -m "feat(topic5-echo): runner masked_template_rank with phantom-safe contract"
```

---

## Task 9: Runner — B0 eligibility audit subcommand (§3.1)

Re-read spec §3.1 (B0 audit columns; MIN_CH=8 locked; audit reports drops only) and §3.5 (construct_validity is a manual sentinel flag — the audit writes the column; the human fills the verdict by eyeballing line-length/broadband/HFA/ER earliest channels on ≥5 sentinel seizures).

**Files:**
- Modify: `scripts/run_topic5_echo_gate.py`

- [ ] **Step 1: Add `cmd_audit` that enumerates the cohort**

```python
def _iter_subjects():
    """Yield (ds_sid, masked_json_path) for subjects present in BOTH the masked-template
    tree AND the v2.3 ictal atlas. Atlas membership uses the CANONICAL loader
    (src.atlas_loading.list_cohort_subjects), NOT a hand-rolled atlas path (P0-B)."""
    atlas_subjects = set(atlas_loading.list_cohort_subjects())   # schema-gated v2.3 set
    for mj in sorted(MASKED_ROOT.glob("rank_displacement/per_subject/*.json")):
        ds_sid = mj.stem                      # e.g. "epilepsiae_1146" — match atlas keys
        if ds_sid in atlas_subjects or ds_sid.split("_", 1)[-1] in atlas_subjects:
            yield ds_sid, mj


def cmd_audit(args):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds_sid, mj in _iter_subjects():
        info = load_subject(ds_sid)           # returns dict per §3.1 columns or None
        if info is None:
            continue
        rows.append(info)
    import csv
    cols = ["subject_id", "dataset", "n_seizures_total", "n_seizures_with_atlas",
            "n_seizures_eligible", "n_channels_template", "n_channels_ictal",
            "n_channels_common_min", "n_channels_common_median", "n_channels_common_max",
            "rank_tie_fraction", "rank_missing_fraction", "rank_dynamic_range",
            "template_k", "template_stability", "swap_class", "ictal_rank_source",
            "atlas_quality_flag", "construct_validity_flag", "reference_type",
            "montage_type", "channel_name_normalization_status", "duplicate_channel_flag",
            "phantom_mask_applied", "valid_mask_source", "alignment_guard_pass",
            "clinical_onset_annotation_available", "deanchor_eligible",
            "deanchor_anchor_reliability"]
    with open(OUT_ROOT / "b0_eligibility_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {OUT_ROOT/'b0_eligibility_audit.csv'} ({len(rows)} subjects)")
    # report drops at locked MIN_CH=8 (NO threshold tuning, P1-1)
    drop8 = sum(1 for r in rows if r.get("n_channels_common_median", 0) < MIN_CH)
    print(f"[MIN_CH={MIN_CH} locked] subjects with median common<8: {drop8} (reported, not tuned)")
```

(Implement `load_subject(ds_sid)` to: read the masked template JSON; build each cluster's masked template rank via `masked_template_rank_1d` **or** `rebuild_template_from_events` per the Task 8 Step 1 decision; load the ictal rank via `atlas_loading.load_per_subject_json(sid, source="per_subject")` and extract `channel_onsets` → a per-channel rank vector; align ictal-rank channels to the template channel order with an `alignment_guard` that **hard-raises** on mismatch; fill the columns. `construct_validity_flag` defaults to `"pending"` — the human sets it after the sentinel eyeball. `swap_class` is recorded as a STRATIFIER column, not a gate.)

- [ ] **Step 2: Run the audit on the real cohort**

Run: `python scripts/run_topic5_echo_gate.py audit`
Expected: writes `results/topic5_ictal_template_echo/b0_eligibility_audit.csv`; prints subject count and the MIN_CH=8 drop count. **Eyeball the CSV**: confirm k=2 subjects dominate, `phantom_mask_applied=True` everywhere, no `alignment_guard_pass=False` rows slipped through.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_topic5_echo_gate.py
git commit -m "feat(topic5-echo): B0 eligibility audit subcommand (MIN_CH=8 locked, drops reported)"
```

---

## Task 10: Runner — per-subject + cohort subcommands + verdict (§4)

Re-read spec §4.1.4 (subject-level primary), §4.3 (6-state verdict, **no-veto**), §4.6 (null modes A/C gate "含具体通路"), §4.4 (bad-data regression).

**Files:**
- Modify: `scripts/run_topic5_echo_gate.py`

- [ ] **Step 1: Add `cmd_per_subject` (per-seizure e_k across null modes + de-anchor)**

```python
def cmd_per_subject(args):
    rng = np.random.default_rng(RNG_SEED)
    (OUT_ROOT / "per_subject").mkdir(parents=True, exist_ok=True)
    for ds_sid, mj in _iter_subjects():
        sub = load_subject(ds_sid)
        if sub is None or sub["atlas_quality_flag"] == "fail":
            continue
        templates = sub["template_ranks"]            # list of masked rank vectors
        shafts = np.array([parse_shaft(c)[0] for c in sub["channels"]])
        per_seizure = []
        seiz_matrix = sub["seizure_ranks"]           # (n_seiz, n_ch) masked-aligned
        cap = echo.shaft_block_capacity(shafts)      # P1-B fail-closed flag
        for k, seiz in enumerate(seiz_matrix):
            rec = {"seizure_idx": k}
            modes = [("channel", None), ("within_shaft", shafts),
                     ("anchor_matched", sub["anchor_bins"])]
            if not cap["insufficient_block_exchange"]:
                modes.append(("shaft_block", shafts))     # only if real exchange possible
            for mode, blocks in modes:
                rec[mode] = echo.compute_echo_strength(
                    seiz, templates, B=B, rng=rng, min_ch=MIN_CH,
                    null_mode=mode, blocks=blocks)
            per_seizure.append(rec)
        # P1-A: de-anchor keeps max-over-ALL-templates (not templates[0])
        deanchor = (echo.compute_deanchor_echo(seiz_matrix, templates, B=B, rng=rng,
                                               min_ch=MIN_CH)
                    if seiz_matrix.shape[0] >= 4 else None)
        out = {"subject": ds_sid, "swap_class": sub["swap_class"],
               "dataset": sub["dataset"], "template_k": sub["template_k"],
               "atlas_quality_flag": sub["atlas_quality_flag"],
               "construct_validity_flag": sub["construct_validity_flag"],
               "shaft_block_capacity": cap,
               "anchor_reliability": echo.anchor_reliability(seiz_matrix),
               "per_seizure": per_seizure, "deanchor": deanchor,
               # stored so cmd_cohort can compute Null D (between-subject control):
               "channels": list(sub["channels"]),
               "template_ranks": [list(t) for t in templates],
               "seizure_ranks": [list(s) for s in seiz_matrix]}
        json.dump(out, open(OUT_ROOT / "per_subject" / f"{ds_sid}.json", "w"), indent=2,
                  default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    print("per-subject done")
```

- [ ] **Step 2: Add `cmd_cohort` (subject-level pooling + verdict + bad-data regression)**

```python
def cmd_cohort(args):
    subs = [json.load(open(p)) for p in sorted((OUT_ROOT / "per_subject").glob("*.json"))]

    def pool_mode(mode, subset=None):
        # GENERIC SCOPE (spec v4 §3.2): primary = ALL subjects with a stable template.
        # subset=None -> all; subset='strict_candidate'/'none' -> swap STRATIFIER subgroup.
        recs = []
        for s in subs:
            if subset == "strict_candidate" and s["swap_class"] not in ("strict", "candidate"):
                continue
            if subset == "none" and s["swap_class"] != "none":
                continue
            for ps in s["per_seizure"]:
                m = ps.get(mode)
                if m is None:                      # e.g. shaft_block skipped (P1-B)
                    continue
                recs.append({"subject": s["subject"], "e_k": m.get("e_k")})
        return echo.pool_echo_subject_level(recs)

    # Null D between-subject control (formal negative control, generic scope):
    rng = np.random.default_rng(RNG_SEED + 1)
    bs_recs = []
    for s in subs:
        others = [np.array(t, float) for o in subs if o["subject"] != s["subject"]
                  for t in o["template_ranks"]]
        for seiz in s["seizure_ranks"]:
            r = echo.between_subject_control(np.array(seiz, float), others,
                                             B=B, rng=rng, min_ch=MIN_CH)
            bs_recs.append({"subject": s["subject"], "e_k": r["e_k"]})

    # bad-data regression pools the REAL null-draw field e_k_baddata (P0-C):
    bd_recs = [{"subject": s["subject"], "e_k_baddata": ps["channel"]["e_k_baddata"],
                "e_k": ps["channel"]["e_k"]}
               for s in subs for ps in s["per_seizure"] if ps.get("channel")]

    summary = {
        "scope": "generic_template_echo",
        "primary_channel_all": pool_mode("channel"),                 # PRIMARY (all subjects)
        "primary_within_shaft_all": pool_mode("within_shaft"),
        "primary_anchor_matched_all": pool_mode("anchor_matched"),
        "stratifier_swap_strict_candidate": pool_mode("channel", "strict_candidate"),
        "stratifier_swap_none": pool_mode("channel", "none"),
        "negative_between_subject": echo.pool_echo_subject_level(bs_recs),
        "bad_data_regression": echo.bad_data_regression(bd_recs),
    }
    summary["verdict"] = _assign_verdict(summary, subs)   # §4.3, NO "暂缓" string anywhere
    # P0-1 no-veto guard: the artifact must never contain the veto word.
    assert "暂缓" not in json.dumps(summary, ensure_ascii=False)
    json.dump(summary, open(OUT_ROOT / "cohort_echo_summary.json", "w"), indent=2)
    print("verdict:", summary["verdict"]["label"])
```

**`_assign_verdict` (§4.3) — guard against "站住" without sensitivities:** a `站住·*` label MUST require that the floor-robust percentile/Stouffer combine AND the per-seizure cluster-robust sensitivity are BOTH present and same-direction (spec §4.3). If either is missing/NaN, the label cannot be any `站住·*` — fall to `没看清`. Encode this explicitly:

```python
def _assign_verdict(summary, subs):
    p = summary["primary_channel_all"]
    has_sens = (np.isfinite(p.get("sign_p_onesided", np.nan)) and
                np.isfinite(p.get("boot_ci95", [np.nan])[0]))
    neg = summary["negative_between_subject"]
    bad = summary["bad_data_regression"]
    neg_clean = not (neg.get("wilcoxon_p_onesided", 1) < 0.05)          # D must be neutral
    bad_clean = not (bad.get("wilcoxon_p_onesided", 1) < 0.05)          # bad-data must flatten
    if p["n_subjects"] < 6:
        return {"label": "没看清", "why": "n_subjects<6"}
    standing = (p.get("wilcoxon_p_onesided", 1) < 0.05 and p["median_E_s"] > 0
                and has_sens and neg_clean and bad_clean)
    if not standing:
        return {"label": "代理阴性/没看清", "why": "primary not significant or sensitivities/controls missing",
                "note": "ER proxy gave no continuation evidence; Stage 2 decided by scientific value (not vetoed)"}
    # standing -> distinguish 含具体通路 vs 稳定锚为主 via A/C nulls (§4.6)
    a = summary["primary_within_shaft_all"]; c = summary["primary_anchor_matched_all"]
    specific = (a.get("wilcoxon_p_onesided", 1) < 0.05 or c.get("wilcoxon_p_onesided", 1) < 0.05)
    return {"label": "站住·含具体通路" if specific else "站住·稳定锚为主",
            "why": "inclusive echo holds; A/C " + ("survive" if specific else "flatten")}
```

- [ ] **Step 3: Wire argparse and run end-to-end**

```python
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit").set_defaults(func=cmd_audit)
    sub.add_parser("per-subject").set_defaults(func=cmd_per_subject)
    sub.add_parser("cohort").set_defaults(func=cmd_cohort)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

Run: `python scripts/run_topic5_echo_gate.py per-subject && python scripts/run_topic5_echo_gate.py cohort`
Expected: writes `cohort_echo_summary.json`; prints a verdict label from §4.3; the run does not raise on the `"暂缓"` lint guard.

- [ ] **Step 4: Test the verdict guard (no 站住 without sensitivities)**

```python
def test_verdict_refuses_standing_without_sensitivities():
    import scripts.run_topic5_echo_gate as R
    # primary is "significant" but sign/bootstrap sensitivities are MISSING (NaN) ->
    # must NOT return any 站住·* label (spec §4.3).
    summary = {
        "primary_channel_all": {"n_subjects": 12, "wilcoxon_p_onesided": 0.001,
                                "median_E_s": 0.9, "sign_p_onesided": float("nan"),
                                "boot_ci95": [float("nan"), float("nan")]},
        "primary_within_shaft_all": {"wilcoxon_p_onesided": 0.001},
        "primary_anchor_matched_all": {"wilcoxon_p_onesided": 0.001},
        "negative_between_subject": {"wilcoxon_p_onesided": 0.5},
        "bad_data_regression": {"wilcoxon_p_onesided": 0.5},
    }
    v = R._assign_verdict(summary, [])
    assert not v["label"].startswith("站住")
```

Run: `pytest tests/test_topic5_echo_gate.py::test_verdict_refuses_standing_without_sensitivities -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_topic5_echo_gate.py tests/test_topic5_echo_gate.py
git commit -m "feat(topic5-echo): per-subject + cohort + generic-scope verdict (no-veto + sensitivity guard)"
```

---

## Task 11: Plotter + `figures/README.md`

Re-read spec §5.3 (3 figures: echo_strength_distribution, null_mode_panel, cohort_pooled_forest) and AGENTS.md Results Directory Standards (Chinese README, one paragraph per figure ending in `**关注点**：`).

**Files:**
- Create: `scripts/plot_topic5_echo_gate.py`
- Create: `results/topic5_ictal_template_echo/figures/README.md`

- [ ] **Step 1: Write the plotter** (per-seizure e_k strip by swap-stratifier/dataset; null-mode forest comparing channel vs within-shaft vs anchor-matched pooled estimates; subject-level E_s forest with the pooled marker AND the between-subject negative control + bad-data regression overlaid). Each figure paper-grade self-contained (no internal codenames on axes — see `MEMORY.md feedback_figure_self_contained_paper_grade`).

- [ ] **Step 2: Generate and eyeball**

Run: `python scripts/plot_topic5_echo_gate.py`
Expected: 3 PNGs under `results/topic5_ictal_template_echo/figures/`. **Eyeball each** before writing the README.

- [ ] **Step 3: Write `figures/README.md`** (Chinese, `### filename` + 2–4 sentences + `**关注点**：`), AFTER the figures render.

- [ ] **Step 4: Commit**

```bash
git add scripts/plot_topic5_echo_gate.py results/topic5_ictal_template_echo/figures/README.md
git commit -m "feat(topic5-echo): figures + README"
```

---

## Task 12: Appendix H2 (descriptive only) + run-log + doc回链

Re-read spec §9 (H2 is descriptive appendix — NOT a verdict, NOT a hypothesis, NOT BH-FDR).

**Files:**
- Modify: `scripts/run_topic5_echo_gate.py` (add `appendix-h2` writing into `cohort_echo_summary.json` under a clearly-labelled `appendix_h2_descriptive` key)
- Modify: `docs/topic5_seizure_subtyping.md` (add a §5 archive回链 line pointing to this gate; no paper-level claim)

- [ ] **Step 1:** Add `appendix_h2_descriptive` (subject-stratified contingency of `sign(ρ_a−ρ_b)` × z-ER subtype, tie excluded; report pooled OR + per-subject Cramér V; self-check shuffle returns neutral). Label the JSON key and any printed text "descriptive — not a test".

- [ ] **Step 2:** Add a回链 line to `docs/topic5_seizure_subtyping.md` §5 history index: the Stage-1 echo proxy-triage spec/plan + results path. State it is proxy triage (does not veto Stage 2) and exploratory.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_topic5_echo_gate.py docs/topic5_seizure_subtyping.md
git commit -m "feat(topic5-echo): H2 descriptive appendix + topic5 doc回链"
```

---

## Self-Review (run after writing all tasks)

**Spec coverage (v4):** §3.6 phantom-safe 1-D vs 2-D (Task 8, two contract tests) ✓; §4.1 echo + k (Tasks 1–4) ✓; §4.1b LOO de-anchor no-leakage + **max-over-templates** (Tasks 5, 10) ✓; §4.1.4 subject-level pooling (Task 6) ✓; §4.6 null modes A/C + **B fail-closed `insufficient_block_exchange`** (Task 3) + **D between-subject formal control** (Task 7b, cohort in Task 10) ✓; §4.4 bad-data regression via **real null draw `e_k_baddata`** (Tasks 4, 6, 10) ✓; §3.5 atlas-quality (Task 7) + construct-validity manual flag (Task 9) ✓; §4.3 verdict no-veto + **refuses 站住 without sensitivities** (Task 10, guard test) ✓; §3.1 B0 audit MIN_CH=8 locked via `src.atlas_loading` (Task 9) ✓; **generic-scope: primary=all stable templates, swap=stratifier, negative control=between-subject (Tasks 9/10)** ✓; §9 H2 appendix (Task 12) ✓; figures (Task 11) ✓.

**P0/P1 guards present:** P0-1 no-veto → Task 10 `"暂缓"` lint assert + reporting wording; P0-2 phantom → Task 8 1-D/2-D loader-contract tests + Task 1 phantom-exclusion test; P0-A 1-D≠2-D mask contract split (Task 8); P0-B canonical `src.atlas_loading` (Tasks 8/9, no hand-rolled atlas root); P0-C bad-data uses real null draws not N(0,1) (Tasks 4/6); P1-A de-anchor max-over-templates (Tasks 5/10); P1-B between-subject task + unequal-shaft fail-closed (Tasks 7b/3); P1-1 MIN_CH=8 locked; P1-2 construct_validity_flag + sentinel; P1-3 H2 appendix only.

**Open implementation risks to verify at execution (not placeholders — flagged checks):**
- The exact masked-template JSON key layout decides 1-D (`masked_template_rank_1d`) vs 2-D (`rebuild_template_from_events`) — Task 8 Step 1 inspects it live; do NOT mix the two contracts.
- Atlas access goes through `src.atlas_loading.load_per_subject_json` / `list_cohort_subjects` (schema-gated v2.3) — NOT a hand-rolled path. The ds_sid↔atlas-key matching (with/without dataset prefix) is verified live in Task 9.
- Atlas `channel_onsets` → rank vector alignment to the template channel order: the `alignment_guard` MUST hard-raise on mismatch (spec §3.4 / item 10); do not silently truncate.
- `anchor_bins` for `anchor_matched` null: bins by **distance-to-SOZ OR mean ictal earliness** (pick per spec §4.6; the user owns which — confirm before Task 10) — define in `load_subject`; if no SOZ for a subject, that null mode is skipped for them (recorded, not silently dropped).
- Null D remap-by-truncation is a deliberately coarse surrogate (Task 7b); if it ever pools significant, the statistic is too coarse → primary void (this is the control doing its job, not a bug).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-08-topic5-ictal-template-echo-gate.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session with checkpoints.

Which approach?
