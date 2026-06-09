# Topic 5 Ictal-Template-Echo Gate (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage-1 "proxy triage" that pools, across subjects, whether each seizure's ER/atlas-derived ictal channel ordering echoes the subject's masked interictal propagation template — replacing the power-floored per-subject contingency (Q1/Q1') with subject-level pooling.

**Architecture:** One pure-math module (`src/topic5_echo_gate.py`, no I/O) holding the echo statistic, the five shuffle nulls, the leave-one-seizure-out de-anchor, subject-level pooling, and atlas-quality; one runner that loads masked templates (from `results/interictal_propagation_masked/`) + ER atlas ranks, runs the B0 audit, computes per-subject + cohort JSON; one plotter. **P0 invariants from the spec:** templates are phantom-safe masked (§3.6), proxy triage never vetoes Stage 2 (P0-1, lives in reporting wording), MIN_CH=8 locked (P1-1).

**Tech Stack:** Python, numpy, scipy.stats (spearmanr, wilcoxon, kendalltau), statsmodels (cluster-robust OLS sensitivity). pytest for TDD. Matplotlib for figures.

**Spec:** `docs/superpowers/specs/2026-06-08-topic5-ictal-template-echo-gate-design.md` (committed c1b3dde, v3). Re-read each referenced spec section at the matching task boundary (CLAUDE.md §5).

---

## File Structure

- **Create `src/topic5_echo_gate.py`** — pure functions, no file I/O. Responsibilities: `spearman_common` (masked, common-channel Spearman); `echo_r_obs` (max over k templates); `_block_permute` + `shuffle_null` (the 5 null modes §4.6); `compute_echo_strength` (e_k/p_k/quantiles §4.1); `loo_anchor` + `compute_deanchor_echo` + `anchor_reliability` (§4.1b); `pool_echo_subject_level` (Wilcoxon/sign/bootstrap primary + cluster-robust sensitivity + bad-data regression §4.1.4/§4.4); `compute_atlas_quality` (§3.5).
- **Create `scripts/run_topic5_echo_gate.py`** — I/O + orchestration. `audit` (B0 csv), `per-subject`, `cohort`, `figures` subcommands. Loads masked templates per §3.6, ER atlas ranks, swap_class. Applies `_apply_masked_paths()` path-swap.
- **Create `scripts/plot_topic5_echo_gate.py`** — 3 figures + `figures/README.md`.
- **Create `tests/test_topic5_echo_gate.py`** — unit tests on synthetic data with known answers.

**Cohort gate (spec §3):** subject ∈ cohort ⇔ has a phantom-safe masked stable template (primary k=2) AND a v2.3 ictal atlas with `n_seizures_with_atlas ≥ 2` AND passes atlas-quality + construct-validity. `MIN_CH=8` locked. swap_class tiers: strict/candidate = primary, none = negative control, all = sensitivity.

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_echo_gate.py -v`
Expected: PASS (9 tests).

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
    return {"e_k": e_k, "p_k": p_k, "r_obs": float(r_obs),
            "null_mean": float(null.mean()), "null_sd": sd,
            "null_q": [float(q) for q in np.quantile(null, [0.05, 0.5, 0.95])],
            "n_null": int(null.size), "null_mode": null_mode}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_echo_gate.py -v`
Expected: PASS (12 tests).

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


def compute_deanchor_echo(per_seizure_ranks, template_rank, *, B, rng, min_ch) -> List[Dict]:
    """Echo on de-anchored deltas: delta_seiz = seiz_k - r_bar_{-k};
    delta_templ = template - r_bar_{-k}. One record per seizure."""
    M = np.asarray(per_seizure_ranks, dtype=float)
    t = np.asarray(template_rank, dtype=float)
    anc = loo_anchor(M)
    records = []
    for k in range(M.shape[0]):
        d_seiz = M[k] - anc[k]
        d_templ = t - anc[k]
        records.append(compute_echo_strength(d_seiz, [d_templ], B=B, rng=rng, min_ch=min_ch))
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


def test_pool_null_when_subjects_centered_zero():
    rng = np.random.default_rng(7)
    recs = _records({f"s{i}": list(rng.normal(0, 1, 3)) for i in range(12)})
    res = pool_echo_subject_level(recs)
    assert res["wilcoxon_p_onesided"] > 0.10        # bad-data regression: no false signal
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_echo_gate.py -v`
Expected: PASS (17 tests).

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
def test_load_masked_template_excludes_phantom(tmp_path, monkeypatch):
    # Build a tiny masked-template fixture: 1 cluster, channel 3 is non-participating.
    # The returned template rank MUST be NaN at the non-participating channel.
    import scripts.run_topic5_echo_gate as R
    channels = ["A1", "A2", "A3", "A4"]
    raw_ranks = np.array([0.0, 1.0, 7.0, 2.0])      # phantom int 7 at idx2 (A3)
    bools = np.array([True, True, False, True])      # A3 never participates
    templ = R.masked_template_rank(raw_ranks, bools)
    assert np.isnan(templ[2])                        # phantom excluded
    assert np.allclose(templ[[0, 1, 3]], raw_ranks[[0, 1, 3]])
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

from src.lagpat_rank_audit import mask_phantom_ranks
from src.propagation_skeleton_geometry import parse_shaft
from src import topic5_echo_gate as echo

MASKED_ROOT = Path("results/interictal_propagation_masked")
ATLAS_ROOT = Path("results/data_driven_soz/layer_a_ictal_er_rank/atlas_v2_3")
OUT_ROOT = Path("results/topic5_ictal_template_echo")
MIN_CH = 8           # P1-1 locked
B = 2000             # §5.3 lock
TIE_MAX = 0.3
RNG_SEED = 20260608


def masked_template_rank(raw_ranks, bools):
    """Phantom-safe template rank: non-participating channels -> NaN (§3.6)."""
    r = np.asarray(raw_ranks, dtype=float)
    masked = mask_phantom_ranks(r, np.asarray(bools, dtype=bool))
    # mask_phantom_ranks returns ranks with non-participating set to NaN; coerce here
    out = np.asarray(masked, dtype=float)
    out[~np.asarray(bools, dtype=bool)] = np.nan
    return out
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_topic5_echo_gate.py::test_load_masked_template_excludes_phantom -v`
Expected: PASS. (If `mask_phantom_ranks` signature differs from the spec, adapt the call and re-run — the contract is "non-participating → NaN", verified by the test.)

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
    """Yield (dataset, sid, masked_json_path, atlas_dir) for subjects present in BOTH
    the masked-template tree and the ictal atlas. Adapt globs to the layout recorded
    in Task 8 Step 1."""
    for mj in sorted(MASKED_ROOT.glob("rank_displacement/per_subject/*.json")):
        ds_sid = mj.stem                      # e.g. "epilepsiae_1146"
        atlas_dir = ATLAS_ROOT / "figures"    # adapt to actual atlas json location
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

(Implement `load_subject(ds_sid)` to read the masked template JSON + atlas, build the masked template rank per cluster via `masked_template_rank`, build the ictal rank vector aligned to the same channel order with an `alignment_guard` that hard-raises on mismatch, and fill the columns. `construct_validity_flag` defaults to `"pending"` — the human sets it after the sentinel eyeball.)

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
        for k, seiz in enumerate(seiz_matrix):
            rec = {"seizure_idx": k}
            for mode, blocks in [("channel", None), ("within_shaft", shafts),
                                 ("anchor_matched", sub["anchor_bins"])]:
                rec[mode] = echo.compute_echo_strength(
                    seiz, templates, B=B, rng=rng, min_ch=MIN_CH,
                    null_mode=mode, blocks=blocks)
            per_seizure.append(rec)
        deanchor = (echo.compute_deanchor_echo(seiz_matrix, templates[0], B=B, rng=rng,
                                               min_ch=MIN_CH)
                    if seiz_matrix.shape[0] >= 4 else None)
        out = {"subject": ds_sid, "swap_class": sub["swap_class"],
               "dataset": sub["dataset"], "template_k": sub["template_k"],
               "atlas_quality_flag": sub["atlas_quality_flag"],
               "anchor_reliability": echo.anchor_reliability(seiz_matrix),
               "per_seizure": per_seizure, "deanchor": deanchor}
        json.dump(out, open(OUT_ROOT / "per_subject" / f"{ds_sid}.json", "w"), indent=2,
                  default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    print("per-subject done")
```

- [ ] **Step 2: Add `cmd_cohort` (subject-level pooling + verdict + bad-data regression)**

```python
def cmd_cohort(args):
    subs = [json.load(open(p)) for p in sorted((OUT_ROOT / "per_subject").glob("*.json"))]
    def pool_for(mode, tier):
        recs = []
        for s in subs:
            if tier == "primary" and s["swap_class"] not in ("strict", "candidate"):
                continue
            if tier == "none" and s["swap_class"] != "none":
                continue
            for ps in s["per_seizure"]:
                e = ps.get(mode, {}).get("e_k")
                recs.append({"subject": s["subject"], "e_k": e})
        return echo.pool_echo_subject_level(recs)
    summary = {
        "primary_channel": pool_for("channel", "primary"),
        "primary_within_shaft": pool_for("within_shaft", "primary"),
        "primary_anchor_matched": pool_for("anchor_matched", "primary"),
        "negative_none_channel": pool_for("channel", "none"),
    }
    # bad-data regression: replace each e_k with a draw from its own null (mean 0) -> NS
    summary["bad_data_regression"] = _bad_data_regression(subs)
    summary["verdict"] = _assign_verdict(summary, subs)   # §4.3, NO "暂缓" string anywhere
    json.dump(summary, open(OUT_ROOT / "cohort_echo_summary.json", "w"), indent=2)
    print("verdict:", summary["verdict"]["label"])
```

(`_assign_verdict` implements the §4.3 table. **Lint guard:** add `assert "暂缓" not in json.dumps(summary, ensure_ascii=False)` so the no-veto wording (P0-1) can never regress into the artifact.)

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

- [ ] **Step 4: Commit**

```bash
git add scripts/run_topic5_echo_gate.py
git commit -m "feat(topic5-echo): per-subject + cohort + 6-state verdict (no-veto lint guard)"
```

---

## Task 11: Plotter + `figures/README.md`

Re-read spec §5.3 (3 figures: echo_strength_distribution, null_mode_panel, cohort_pooled_forest) and AGENTS.md Results Directory Standards (Chinese README, one paragraph per figure ending in `**关注点**：`).

**Files:**
- Create: `scripts/plot_topic5_echo_gate.py`
- Create: `results/topic5_ictal_template_echo/figures/README.md`

- [ ] **Step 1: Write the plotter** (per-seizure e_k strip by swap_class/dataset; null-mode forest comparing channel vs within-shaft vs anchor-matched pooled estimates; subject-level E_s forest with pooled marker and none-subset control). Each figure paper-grade self-contained (no internal codenames on axes — see `MEMORY.md feedback_figure_self_contained_paper_grade`).

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

**Spec coverage:** §3.6 phantom-safe (Task 8) ✓; §4.1 echo + k (Tasks 1–4) ✓; §4.1b LOO de-anchor no-leakage (Task 5) ✓; §4.1.4 subject-level pooling (Task 6) ✓; §4.6 null modes A/C (Task 3, gating in Task 10) ✓; §4.4 bad-data regression (Tasks 6, 10) ✓; §3.5 atlas-quality (Task 7) + construct-validity manual flag (Task 9) ✓; §4.3 6-state no-veto verdict (Task 10, lint guard) ✓; §3.1 B0 audit MIN_CH=8 locked (Task 9) ✓; §9 H2 appendix (Task 12) ✓; figures (Task 11) ✓.

**P0/P1 guards present:** P0-1 no-veto → Task 10 `"暂缓"` lint assert + reporting wording; P0-2 phantom → Task 8 loader-contract test + Task 1 phantom-exclusion test; P1-1 MIN_CH=8 locked constant (no post-audit tuning); P1-2 construct_validity_flag column; P1-3 H2 in appendix only.

**Open implementation risks to verify at execution (not placeholders — flagged checks):**
- The exact masked-template JSON key layout (Task 8 Step 1 inspects it live). `mask_phantom_ranks` return convention is verified by the Task 8 test, not assumed.
- Atlas `channel_onsets` → rank vector alignment to the template channel order: the `alignment_guard` MUST hard-raise on mismatch (spec §3.4 / item 10); do not silently truncate.
- `anchor_bins` for `anchor_matched` null: bins by distance-to-SOZ or mean ictal earliness — define in `load_subject` from available SOZ JSON; if no SOZ for a subject, that null mode is skipped for them (recorded, not silently dropped).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-08-topic5-ictal-template-echo-gate.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session with checkpoints.

Which approach?
