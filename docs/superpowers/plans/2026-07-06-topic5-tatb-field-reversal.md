# TA/TB Field-Reversal Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Formally test, per broad-substrate subject, whether the two interictal propagation templates' smoothed fields (TA_field, TB_field) are signed-anti-correlated beyond a within-shaft permutation null — plus a denoising supplement (field vs contact axis reproducibility).

**Architecture:** One new pure-math module `src/topic5_field_reversal.py` (reversal metric + null gate + contact head-to-head + random-split contrast + LOO reproducibility + cohort binomial), driven by one runner and one plotter. Reuses the existing contact-plane smoother, the A-line within-shaft null, and the event-resolved broad loader. No new data-loading primitives; broad substrate only.

**Tech Stack:** Python 3, numpy, scipy.stats (spearmanr), matplotlib (Agg). pytest for TDD.

## Global Constraints

Copied verbatim from spec `docs/superpowers/specs/2026-07-06-topic5-tatb-field-reversal-design.md`. Every task's requirements implicitly include these:

- **Substrate: broad only.** `load_event_labels_ranks(broad=False)` raises `NotImplementedError`; narrow is a separate future plan. (spec §8)
- **Shared frame (P0):** TA and TB fields MUST be built on ONE frame = the reference template `t_a`'s normalized readout plane (`GEOM_BROAD/{ds_sid}_t_a.json`). Never correlate a field on `plane_a` against a field on `plane_b`. (spec §3.1)
- **Value (P1):** per-contact value = `class_aggregate_contact_values(bundle, label)[name]["value"]` (masked normalized-rank class mean). NEVER display `_rank01`. (spec §3.2)
- **Smoother (P1):** stat fields use `field_from_contact_values(...) → R_smooth_rank` with `sigma = class_template_sigma(plane_ref)` (= median-nn) and `s_thresh = S_THRESH = 0.15`. FORBIDDEN: `_smooth_rank_field_mm`, `VIS_SIGMA_MULT=2.5`, `VIS_SIGMA_MIN_MM=6.0`, `VIS_MASK_REL=0.02`. (spec §10/§12)
- **Metric:** signed Pearson via `_support_corr` (identity orientation only, NO y-mirror), `overlap_min = OVERLAP_MIN = 25`, tested on the **negative** tail. (spec §3.2)
- **Primary null:** within-shaft permutation of TB **values only** (coords/support/names fixed); `effective_shuffle_n` weak-null guard computed on the channels ACTUALLY entering the stat (finite TB value on `plane_ref`), not all channels; `degenerate_null` subjects excluded from inferential count, still reported. (spec §4/§12)
- **pass** = null-left-tail `percentile < 5` AND `signed_corr < 0` AND not `degenerate_null`. Cohort = binomial on pass count vs 0.05. One reversal corr per subject (no multi-template folding). (spec §4)
- **Reporting red-line:** observed corr always reported next to its within-shaft null band; never a bare corr. Even on PASS, only "TA/TB are opposite traversal directions of the same spatial scaffold" — NOT directional replay / seizure-polarity / "field proves real direction". (spec §2)
- **Constants** (from `src/propagation_contact_plane_readout.py`): `S_THRESH=0.15`, `OVERLAP_MIN=25`, `GRID_N=81`, `MIN_PLANE_CONTACTS=3`.
- **Results root:** `results/topic5_ictal_recruitment/field_reversal/`. Figures dir needs a Chinese `README.md` (AGENTS.md results standards).

---

## File Structure

- **Create** `src/topic5_field_reversal.py` — all pure-math (no file IO, no argparse, no matplotlib). One responsibility: given per-subject class-aggregate values + a reference plane, produce the reversal gate / contact head-to-head / random-split / LOO reproducibility results.
- **Create** `tests/test_topic5_field_reversal.py` — TDD for every invariant in Global Constraints.
- **Create** `scripts/run_topic5_field_reversal.py` — broad data loading (reuses the `run_topic5_event_resolved_alignment.py` loading pattern), per-subject JSON + `cohort_summary.json`, bandwidth sweep. Refuses implicit cohort runs.
- **Create** `scripts/plot_topic5_field_reversal.py` — per-subject panels, cohort null-forest, head-to-head, supplement, 1146 case; writes `figures/README.md`.
- **Create** `results/topic5_ictal_recruitment/field_reversal/` (runner output; gitignored data).

### Reused interfaces (exact signatures — do NOT re-implement)

From `src/propagation_contact_plane_readout.py`:
- `make_plane_grid(n=81) -> (X, Y)`
- `smooth_field(record, X, Y, sigma_xy=None, scalar="rank", s_thresh=0.15) -> {"T","S","U","mask","sigma_xy"}`; `record = {"channels":[{"name","x_norm","y_norm","typical_rank","support"}]}`
- `R_smooth_rank(rec, X, Y, sigma_xy, s_thresh) -> field`
- `_support_corr(F1, F2, S1, S2, s_thresh) -> (corr, n)` — signed Pearson on `(S1>=t)&(S2>=t)&finite` pixels; `(nan, n)` if n<2
- `placement_in_distribution(value, dist) -> {"percentile","robust_z","n"}` — `percentile = (dist < value).mean()*100` (left-tail)
- `S_THRESH=0.15`, `OVERLAP_MIN=25`

From `src/topic5_axis_alignment.py`:
- `within_shaft_shuffle(values: np.ndarray, names: Sequence[str], rng) -> np.ndarray` — permute values within each shaft, multiset preserved, never cross shafts
- `channel_shuffle(values: np.ndarray, rng) -> np.ndarray`
- `effective_shuffle_n(names, anchor, kind, n_bins=4) -> int` — permutable-channel count; `kind="within_shaft"`, `anchor=None`

From `src/topic5_event_resolved_alignment.py`:
- `load_event_labels_ranks(dataset, subject, broad=True) -> bundle` where `bundle = {"masked"(n_ch,n_ev),"bools","ranks_raw","labels"(n_ev,),"valid_ev","event_abs_times","block_ids","channel_names"(n_ch,),"n_blocks","cluster_template_ranks"{0:list,1:list}}`; raises `ValueError` (C1) / `NotImplementedError` (narrow) / `FileNotFoundError`
- `map_clusters_to_templates(c0, c1, ta_rank, tb_rank, margin=0.30) -> {"map":{0,1}->{"t_a","t_b"}|None,"diag_minus_offdiag","ambiguous","corr_matrix"}`
- `class_aggregate_contact_values(bundle, label) -> {name:{"value","support"}}` — value = masked-rank class mean (NaN non-participating), support = participation fraction
- `field_from_contact_values(plane_record, values_by_name, support_by_name=None, *, sigma, X, Y, s_thresh=0.15) -> field|None` (None if `< MIN_PLANE_CONTACTS=3`)
- `class_template_sigma(plane_record, *, X, Y, s_thresh=0.15) -> float` (median-nn; stat-safe)
- `build_plane_xy(plane_record) -> {name:(x_norm,y_norm)}`

### New module public API (define in Task order; later tasks consume these exact names)

```python
# src/topic5_field_reversal.py
signed_reversal_corr(field0, field1, s_thresh=S_THRESH, overlap_min=OVERLAP_MIN) -> dict
build_reversal_fields(plane_ref, cav0, cav1, *, X, Y, sigma=None, s_thresh=S_THRESH) -> dict
within_shaft_reversal_gate(plane_ref, cav0, cav1, *, X, Y, sigma=None, n_perm=1000, rng, min_eff=6, s_thresh=S_THRESH, overlap_min=OVERLAP_MIN) -> dict
channel_floor(plane_ref, cav0, cav1, *, X, Y, sigma, n_perm, rng, s_thresh=S_THRESH, overlap_min=OVERLAP_MIN) -> dict
random_split_contrast(bundle, plane_ref, *, X, Y, sigma, n_split=200, rng, s_thresh=S_THRESH, overlap_min=OVERLAP_MIN) -> dict
contact_reversal_gate(cav0, cav1, *, n_perm=1000, rng, min_eff=6) -> dict
loo_reproducibility(bundle, plane_ref, *, n_split=50, rng, sigma) -> dict
cohort_binomial(pass_flags: Sequence[bool]) -> dict
```

---

### Task 1: Signed no-mirror reversal metric

**Files:**
- Create: `src/topic5_field_reversal.py`
- Test: `tests/test_topic5_field_reversal.py`

**Interfaces:**
- Consumes: `_support_corr`, `S_THRESH`, `OVERLAP_MIN` (propagation_contact_plane_readout)
- Produces: `signed_reversal_corr(field0, field1, s_thresh=S_THRESH, overlap_min=OVERLAP_MIN) -> {"signed_corr":float|None,"n_overlap":int,"insufficient_overlap":bool}`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
from src.topic5_field_reversal import signed_reversal_corr

def _grid(n=81):
    yy, xx = np.mgrid[0:n, 0:n]
    return xx.astype(float), yy.astype(float)

def test_detects_perfect_reversal():
    xx, yy = _grid()
    S = np.ones_like(xx)
    f0 = {"T": xx + yy, "S": S}
    f1 = {"T": -(xx + yy), "S": S}          # exact reversal
    out = signed_reversal_corr(f0, f1)
    assert out["signed_corr"] < -0.99
    assert not out["insufficient_overlap"]

def test_no_y_mirror_is_applied():
    # F1 = x - y : corr(F0,F1)=0, but flip_y(F1) -> x+y -> corr +1.
    # A mirror-invariant impl would wrongly return +1; the no-mirror stat must return ~0.
    xx, yy = _grid()
    S = np.ones_like(xx)
    f0 = {"T": xx + yy, "S": S}
    f1 = {"T": xx - yy, "S": S}
    out = signed_reversal_corr(f0, f1)
    assert abs(out["signed_corr"]) < 0.05

def test_insufficient_overlap_flagged():
    xx, yy = _grid()
    S0 = np.zeros_like(xx); S0[:2, :2] = 1.0     # tiny support
    S1 = np.zeros_like(xx); S1[:2, :2] = 1.0
    out = signed_reversal_corr({"T": xx + yy, "S": S0}, {"T": -(xx + yy), "S": S1})
    assert out["insufficient_overlap"] is True
    assert out["n_overlap"] < 25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_field_reversal.py -k "reversal or mirror or overlap" -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'signed_reversal_corr'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Topic 5 — TA/TB interictal propagation FIELD reversal gate (broad substrate).

Signed, no-mirror test of whether the two interictal templates' smoothed fields are
anti-correlated beyond a within-shaft permutation null, on a subject-fixed shared frame
(the t_a readout plane). See docs/superpowers/specs/2026-07-06-topic5-tatb-field-reversal-design.md
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from src.propagation_contact_plane_readout import _support_corr, S_THRESH, OVERLAP_MIN


def signed_reversal_corr(field0: dict, field1: dict,
                         s_thresh: float = S_THRESH,
                         overlap_min: int = OVERLAP_MIN) -> dict:
    """Signed (identity-orientation, NO y-mirror) support-gated Pearson between two fields.

    field{0,1} = {"T","S"} on the SAME grid/frame. Negative => reversed pair. Returns
    signed_corr (None if unusable), n_overlap, insufficient_overlap (overlap<overlap_min).
    """
    corr, n = _support_corr(field0["T"], field1["T"], field0["S"], field1["S"], s_thresh)
    insufficient = (n < overlap_min) or (not np.isfinite(corr))
    return {"signed_corr": (float(corr) if np.isfinite(corr) else None),
            "n_overlap": int(n), "insufficient_overlap": bool(insufficient)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_field_reversal.py -k "reversal or mirror or overlap" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/topic5_field_reversal.py tests/test_topic5_field_reversal.py
git commit -m "feat(topic5): signed no-mirror field-reversal metric"
```

---

### Task 2: Shared-frame dual-field builder

**Files:**
- Modify: `src/topic5_field_reversal.py`
- Test: `tests/test_topic5_field_reversal.py`

**Interfaces:**
- Consumes: `field_from_contact_values`, `class_template_sigma`, `make_plane_grid` (event_resolved_alignment / plane_readout)
- Produces: `build_reversal_fields(plane_ref, cav0, cav1, *, X, Y, sigma=None, s_thresh=S_THRESH) -> {"field0","field1","sigma":float,"names_used":list[str]}`. `names_used` = channels finite in **cav1** AND present with finite coords on `plane_ref` (the channels that actually enter TB's field — used later for effective_n and the null). Both fields built on `plane_ref` (P0). `sigma=None` -> `class_template_sigma(plane_ref)`.

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_field_reversal import build_reversal_fields
from src.propagation_contact_plane_readout import make_plane_grid

def _toy_plane(names, xy):
    return {"channels": [{"name": n, "x_norm": xy[n][0], "y_norm": xy[n][1],
                          "typical_rank": 0.0, "support": 1.0} for n in names]}

def test_both_fields_on_same_frame_and_sigma():
    names = [f"A{i}" for i in range(1, 7)] + [f"B{i}" for i in range(1, 7)]
    xy = {n: (0.1 * i, 0.0) for i, n in enumerate(names)}
    plane_ref = _toy_plane(names, xy)
    cav0 = {n: {"value": float(i), "support": 1.0} for i, n in enumerate(names)}
    cav1 = {n: {"value": float(len(names) - i), "support": 1.0} for i, n in enumerate(names)}  # reversed
    X, Y = make_plane_grid()
    out = build_reversal_fields(plane_ref, cav0, cav1, X=X, Y=Y)
    assert out["field0"] is not None and out["field1"] is not None
    # same sigma used for both (single float returned)
    assert out["field0"]["sigma_xy"] == out["field1"]["sigma_xy"] == out["sigma"]
    assert set(out["names_used"]) == set(names)

def test_membership_mismatch_names_used_is_cav1_on_plane():
    names = [f"A{i}" for i in range(1, 7)]
    xy = {n: (0.1 * i, 0.0) for i, n in enumerate(names)}
    plane_ref = _toy_plane(names, xy)
    cav0 = {n: {"value": 1.0, "support": 1.0} for n in names}
    cav1 = {n: {"value": 1.0, "support": 1.0} for n in names[:5]}     # A6 absent in cav1
    cav1["A6"] = {"value": np.nan, "support": 0.0}
    X, Y = make_plane_grid()
    out = build_reversal_fields(plane_ref, cav0, cav1, X=X, Y=Y)
    assert "A6" not in out["names_used"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_field_reversal.py -k "same_frame or membership" -v`
Expected: FAIL with `ImportError: cannot import name 'build_reversal_fields'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/topic5_field_reversal.py` (add imports at top):

```python
from src.topic5_event_resolved_alignment import (
    field_from_contact_values, class_template_sigma, build_plane_xy)


def _finite_on_plane(cav: dict, plane_xy: dict) -> list:
    return [n for n, d in cav.items()
            if n in plane_xy and d.get("value") is not None and np.isfinite(d["value"])]


def build_reversal_fields(plane_ref: dict, cav0: dict, cav1: dict, *,
                          X, Y, sigma: Optional[float] = None,
                          s_thresh: float = S_THRESH) -> dict:
    """Build TA (cav0) and TB (cav1) fields on the SAME reference plane (P0). Raw class-mean
    values (P1) with per-class participation support; single median-nn sigma for both."""
    if sigma is None:
        sigma = class_template_sigma(plane_ref, X=X, Y=Y, s_thresh=s_thresh)
    plane_xy = build_plane_xy(plane_ref)
    v0 = {n: d["value"] for n, d in cav0.items()}
    s0 = {n: d["support"] for n, d in cav0.items()}
    v1 = {n: d["value"] for n, d in cav1.items()}
    s1 = {n: d["support"] for n, d in cav1.items()}
    field0 = field_from_contact_values(plane_ref, v0, support_by_name=s0,
                                       sigma=sigma, X=X, Y=Y, s_thresh=s_thresh)
    field1 = field_from_contact_values(plane_ref, v1, support_by_name=s1,
                                       sigma=sigma, X=X, Y=Y, s_thresh=s_thresh)
    return {"field0": field0, "field1": field1, "sigma": float(sigma),
            "names_used": _finite_on_plane(cav1, plane_xy)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_field_reversal.py -k "same_frame or membership" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/topic5_field_reversal.py tests/test_topic5_field_reversal.py
git commit -m "feat(topic5): shared-frame dual-field builder (P0 frame + P1 raw value/sigma)"
```

---

### Task 3: Within-shaft reversal gate (primary null)

**Files:**
- Modify: `src/topic5_field_reversal.py`
- Test: `tests/test_topic5_field_reversal.py`

**Interfaces:**
- Consumes: `within_shaft_shuffle`, `effective_shuffle_n` (topic5_axis_alignment), `placement_in_distribution`, `build_reversal_fields`, `signed_reversal_corr`
- Produces: `within_shaft_reversal_gate(plane_ref, cav0, cav1, *, X, Y, sigma=None, n_perm=1000, rng, min_eff=6, s_thresh=S_THRESH, overlap_min=OVERLAP_MIN) -> {"signed_corr","n_overlap","effective_n","degenerate_null":bool,"null_corrs":list,"null_p95":float,"percentile":float,"passed":bool,"sigma":float}`. Null = permute cav1 **values** within shaft (support/coords/names fixed), rebuild TB field, recompute corr. `degenerate_null` if `effective_n < min_eff` (computed on `names_used`). `passed = (not degenerate) and percentile<5 and signed_corr<0`.

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_field_reversal import within_shaft_reversal_gate

def _two_shaft_plane():
    # two shafts A (x=0 column) and B (x=1 column), 6 contacts each, along-shaft y gradient
    names, xy = [], {}
    for sh, x in (("A", 0.0), ("B", 1.0)):
        for i in range(6):
            n = f"{sh}{i+1}"; names.append(n); xy[n] = (x, 0.15 * i)
    return {"channels": [{"name": n, "x_norm": xy[n][0], "y_norm": xy[n][1],
                          "typical_rank": 0.0, "support": 1.0} for n in names]}, names

def test_along_shaft_reversal_beats_within_shaft_null():
    plane, names = _two_shaft_plane()
    # cav0 rises along y; cav1 is the along-shaft reverse -> anti-correlated fields
    cav0 = {n: {"value": float(n[1:]), "support": 1.0} for n in names}
    cav1 = {n: {"value": float(7 - int(n[1:])), "support": 1.0} for n in names}
    rng = np.random.default_rng(0)
    out = within_shaft_reversal_gate(plane, cav0, cav1, X=None, Y=None, sigma=None,
                                     n_perm=200, rng=rng, overlap_min=10)
    assert out["signed_corr"] < 0
    assert out["percentile"] < 5.0          # observed below within-shaft null
    assert out["passed"] is True
    assert out["degenerate_null"] is False

def test_singleton_shafts_flagged_degenerate():
    # every contact on its own shaft -> nothing permutable within-shaft
    names = [f"S{i}" for i in range(8)]
    plane = {"channels": [{"name": n, "x_norm": 0.1 * i, "y_norm": 0.0,
                           "typical_rank": 0.0, "support": 1.0} for i, n in enumerate(names)]}
    cav0 = {n: {"value": float(i), "support": 1.0} for i, n in enumerate(names)}
    cav1 = {n: {"value": float(8 - i), "support": 1.0} for i, n in enumerate(names)}
    rng = np.random.default_rng(0)
    out = within_shaft_reversal_gate(plane, cav0, cav1, X=None, Y=None, sigma=None,
                                     n_perm=50, rng=rng, min_eff=6, overlap_min=10)
    assert out["degenerate_null"] is True
    assert out["passed"] is False
```

Note: `X=None, Y=None` -> the gate builds its own grid via `make_plane_grid()` when X is None (see impl). `overlap_min=10` for the toy 12-contact plane.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_field_reversal.py -k "along_shaft or singleton" -v`
Expected: FAIL with `ImportError: cannot import name 'within_shaft_reversal_gate'`

- [ ] **Step 3: Write minimal implementation**

Add imports + function to `src/topic5_field_reversal.py`:

```python
from src.propagation_contact_plane_readout import make_plane_grid, placement_in_distribution
from src.topic5_axis_alignment import within_shaft_shuffle, channel_shuffle, effective_shuffle_n


def _perm_cav(cav: dict, names_used: Sequence[str], perm_values: np.ndarray) -> dict:
    """cav with values on names_used replaced by perm_values (support/others untouched)."""
    out = dict(cav)
    for n, v in zip(names_used, perm_values):
        out[n] = {"value": float(v), "support": cav[n]["support"]}
    return out


def within_shaft_reversal_gate(plane_ref, cav0, cav1, *, X, Y, sigma=None,
                               n_perm=1000, rng, min_eff=6,
                               s_thresh=S_THRESH, overlap_min=OVERLAP_MIN) -> dict:
    if X is None or Y is None:
        X, Y = make_plane_grid()
    built = build_reversal_fields(plane_ref, cav0, cav1, X=X, Y=Y, sigma=sigma, s_thresh=s_thresh)
    names_used = built["names_used"]
    obs = signed_reversal_corr(built["field0"], built["field1"], s_thresh, overlap_min)
    eff = int(effective_shuffle_n(names_used, None, "within_shaft"))
    degenerate = eff < min_eff
    base = {"signed_corr": obs["signed_corr"], "n_overlap": obs["n_overlap"],
            "insufficient_overlap": obs["insufficient_overlap"],
            "effective_n": eff, "degenerate_null": bool(degenerate),
            "sigma": built["sigma"], "null_corrs": [], "null_p95": float("nan"),
            "percentile": float("nan"), "passed": False}
    if degenerate or obs["insufficient_overlap"] or obs["signed_corr"] is None:
        return base
    vals1 = np.array([cav1[n]["value"] for n in names_used], float)
    null = []
    for _ in range(n_perm):
        perm = within_shaft_shuffle(vals1, names_used, rng)
        cav1p = _perm_cav(cav1, names_used, perm)
        fp = build_reversal_fields(plane_ref, cav0, cav1p, X=X, Y=Y,
                                   sigma=built["sigma"], s_thresh=s_thresh)
        r = signed_reversal_corr(fp["field0"], fp["field1"], s_thresh, overlap_min)
        if r["signed_corr"] is not None:
            null.append(r["signed_corr"])
    null = np.asarray(null, float)
    place = placement_in_distribution(obs["signed_corr"], null)   # percentile = %(null < obs)
    base.update({"null_corrs": null.tolist(),
                 "null_p95": float(np.nanpercentile(null, 95)) if null.size else float("nan"),
                 "percentile": place["percentile"],
                 "passed": bool(place["percentile"] < 5.0 and obs["signed_corr"] < 0.0)})
    return base
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_field_reversal.py -k "along_shaft or singleton" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/topic5_field_reversal.py tests/test_topic5_field_reversal.py
git commit -m "feat(topic5): within-shaft reversal gate + effective_n degeneracy guard"
```

---

### Task 4: Channel-shuffle floor + random-split descriptive contrast

**Files:**
- Modify: `src/topic5_field_reversal.py`
- Test: `tests/test_topic5_field_reversal.py`

**Interfaces:**
- Produces:
  - `channel_floor(plane_ref, cav0, cav1, *, X, Y, sigma, n_perm, rng, ...) -> {"percentile","null_p95","null_corrs"}` (same shape as gate's null section, using `channel_shuffle`).
  - `random_split_contrast(bundle, plane_ref, *, X, Y, sigma, n_split=200, rng, ...) -> {"split_corrs":list,"split_median":float,"observed_ab_corr":float,"note":"non_inferential"}` — split events into 2 random balanced halves ignoring labels; aggregate each half's per-contact masked-rank mean; build both fields on `plane_ref`; corr. `observed_ab_corr` = the true class-0-vs-class-1 corr.

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_field_reversal import random_split_contrast

def _bundle_two_clusters(plane_names):
    # cluster 0: rank rises A1->B6 ; cluster 1: reversed. masked (n_ch, n_ev).
    n_ch = len(plane_names)
    rise = np.linspace(0, 1, n_ch)
    ev0 = np.tile(rise[:, None], (1, 40)) + 0.01
    ev1 = np.tile((1 - rise)[:, None], (1, 40)) + 0.01
    masked = np.hstack([ev0, ev1])
    labels = np.array([0] * 40 + [1] * 40)
    return {"masked": masked, "labels": labels, "channel_names": list(plane_names),
            "bools": np.isfinite(masked)}

def test_random_split_centers_positive_observed_negative():
    plane, names = _two_shaft_plane()
    bundle = _bundle_two_clusters(names)
    X, Y = make_plane_grid()
    rng = np.random.default_rng(1)
    out = random_split_contrast(bundle, plane, X=X, Y=Y, sigma=None, n_split=100,
                                rng=rng, overlap_min=10)
    assert out["observed_ab_corr"] < 0                 # true A/B reversed
    assert out["split_median"] > out["observed_ab_corr"]  # random halves not reversed
    assert out["note"] == "non_inferential"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_field_reversal.py -k "random_split" -v`
Expected: FAIL with `ImportError: cannot import name 'random_split_contrast'`

- [ ] **Step 3: Write minimal implementation**

```python
def _aggregate_over_events(masked: np.ndarray, names: Sequence[str], cols: np.ndarray) -> dict:
    """Per-contact masked-rank mean over the given event columns -> {name:{value,support}}."""
    sub = masked[:, cols]
    with np.errstate(invalid="ignore"):
        val = np.where(np.all(np.isnan(sub), axis=1), np.nan, np.nanmean(sub, axis=1))
    sup = np.isfinite(sub).mean(axis=1)
    return {n: {"value": float(val[i]), "support": float(sup[i])} for i, n in enumerate(names)}


def channel_floor(plane_ref, cav0, cav1, *, X, Y, sigma, n_perm, rng,
                  s_thresh=S_THRESH, overlap_min=OVERLAP_MIN) -> dict:
    built = build_reversal_fields(plane_ref, cav0, cav1, X=X, Y=Y, sigma=sigma, s_thresh=s_thresh)
    names = built["names_used"]
    obs = signed_reversal_corr(built["field0"], built["field1"], s_thresh, overlap_min)
    vals1 = np.array([cav1[n]["value"] for n in names], float)
    null = []
    for _ in range(n_perm):
        cav1p = _perm_cav(cav1, names, channel_shuffle(vals1, rng))
        fp = build_reversal_fields(plane_ref, cav0, cav1p, X=X, Y=Y, sigma=built["sigma"], s_thresh=s_thresh)
        r = signed_reversal_corr(fp["field0"], fp["field1"], s_thresh, overlap_min)
        if r["signed_corr"] is not None:
            null.append(r["signed_corr"])
    null = np.asarray(null, float)
    place = placement_in_distribution(obs["signed_corr"], null) if null.size else {"percentile": float("nan")}
    return {"null_corrs": null.tolist(), "percentile": place["percentile"],
            "null_p95": float(np.nanpercentile(null, 95)) if null.size else float("nan")}


def random_split_contrast(bundle, plane_ref, *, X, Y, sigma, n_split=200, rng,
                          s_thresh=S_THRESH, overlap_min=OVERLAP_MIN) -> dict:
    masked = bundle["masked"]; names = list(bundle["channel_names"]); n_ev = masked.shape[1]
    if sigma is None:
        sigma = class_template_sigma(plane_ref, X=X, Y=Y, s_thresh=s_thresh)
    # observed A/B (true labels)
    labels = np.asarray(bundle["labels"])
    cav0 = _aggregate_over_events(masked, names, np.where(labels == 0)[0])
    cav1 = _aggregate_over_events(masked, names, np.where(labels == 1)[0])
    b = build_reversal_fields(plane_ref, cav0, cav1, X=X, Y=Y, sigma=sigma, s_thresh=s_thresh)
    obs = signed_reversal_corr(b["field0"], b["field1"], s_thresh, overlap_min)["signed_corr"]
    splits = []
    for _ in range(n_split):
        perm = rng.permutation(n_ev); half = n_ev // 2
        ch = _aggregate_over_events(masked, names, perm[:half])
        cl = _aggregate_over_events(masked, names, perm[half:])
        fb = build_reversal_fields(plane_ref, ch, cl, X=X, Y=Y, sigma=sigma, s_thresh=s_thresh)
        r = signed_reversal_corr(fb["field0"], fb["field1"], s_thresh, overlap_min)["signed_corr"]
        if r is not None:
            splits.append(r)
    return {"split_corrs": splits, "split_median": float(np.median(splits)) if splits else float("nan"),
            "observed_ab_corr": (float(obs) if obs is not None else float("nan")),
            "note": "non_inferential"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_field_reversal.py -k "random_split" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/topic5_field_reversal.py tests/test_topic5_field_reversal.py
git commit -m "feat(topic5): channel-floor null + non-inferential random-split contrast"
```

---

### Task 5: Contact-level head-to-head gate

**Files:**
- Modify: `src/topic5_field_reversal.py`
- Test: `tests/test_topic5_field_reversal.py`

**Interfaces:**
- Consumes: `scipy.stats.spearmanr`, `within_shaft_shuffle`, `effective_shuffle_n`, `placement_in_distribution`
- Produces: `contact_reversal_gate(cav0, cav1, *, n_perm=1000, rng, min_eff=6) -> {"signed_spearman","effective_n","degenerate_null","percentile","null_p95","passed"}` — NO geometry: signed Spearman between the two per-contact value vectors (over contacts finite in both), within-shaft null on cav1 values.

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_field_reversal import contact_reversal_gate

def test_contact_gate_detects_reversal():
    names = [f"A{i}" for i in range(1, 7)] + [f"B{i}" for i in range(1, 7)]
    cav0 = {n: {"value": float(i), "support": 1.0} for i, n in enumerate(names)}
    cav1 = {n: {"value": float(len(names) - i), "support": 1.0} for i, n in enumerate(names)}
    rng = np.random.default_rng(2)
    out = contact_reversal_gate(cav0, cav1, n_perm=200, rng=rng)
    assert out["signed_spearman"] < -0.9
    assert out["percentile"] < 5.0
    assert out["passed"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_field_reversal.py -k "contact_gate" -v`
Expected: FAIL with `ImportError: cannot import name 'contact_reversal_gate'`

- [ ] **Step 3: Write minimal implementation**

```python
from scipy.stats import spearmanr


def contact_reversal_gate(cav0, cav1, *, n_perm=1000, rng, min_eff=6) -> dict:
    common = [n for n in cav0 if n in cav1
              and np.isfinite(cav0[n]["value"]) and np.isfinite(cav1[n]["value"])]
    v0 = np.array([cav0[n]["value"] for n in common], float)
    v1 = np.array([cav1[n]["value"] for n in common], float)
    eff = int(effective_shuffle_n(common, None, "within_shaft"))
    degenerate = eff < min_eff or len(common) < 3
    obs = float(spearmanr(v0, v1).correlation) if len(common) >= 3 else float("nan")
    base = {"signed_spearman": obs, "effective_n": eff, "degenerate_null": bool(degenerate),
            "percentile": float("nan"), "null_p95": float("nan"), "passed": False}
    if degenerate or not np.isfinite(obs):
        return base
    null = np.array([spearmanr(v0, within_shaft_shuffle(v1, common, rng)).correlation
                     for _ in range(n_perm)], float)
    place = placement_in_distribution(obs, null)
    base.update({"percentile": place["percentile"],
                 "null_p95": float(np.nanpercentile(null, 95)),
                 "passed": bool(place["percentile"] < 5.0 and obs < 0.0)})
    return base
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_field_reversal.py -k "contact_gate" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/topic5_field_reversal.py tests/test_topic5_field_reversal.py
git commit -m "feat(topic5): contact-level signed-Spearman reversal head-to-head"
```

---

### Task 6: LOO field-vs-contact reproducibility supplement

**Files:**
- Modify: `src/topic5_field_reversal.py`
- Test: `tests/test_topic5_field_reversal.py`

**Interfaces:**
- Produces: `loo_reproducibility(bundle, plane_ref, *, n_split=50, rng, sigma) -> {"field_rho":float,"contact_rho":float,"n_contacts_common":int}`. Per class, per split: train-half raw per-contact mean rank; held-out-half per-contact mean rank; **contact** prediction = train raw value; **field** prediction = LOO kernel regression at each contact's location from OTHER train contacts (target contact excluded). Spearman(pred, held) over contacts finite in **all three** (train, held, LOO-field). Fold A/B, mean over splits.

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_field_reversal import loo_reproducibility

def test_field_beats_contact_when_neighbors_more_reliable():
    # smooth spatial gradient + heavy per-contact per-event noise -> a contact's own train
    # estimate is noisy, but its neighbors pin the true value (field LOO should win).
    plane, names = _two_shaft_plane()
    n_ch = len(names)
    true = np.linspace(0, 1, n_ch)
    rng0 = np.random.default_rng(7)
    n_ev = 60
    masked = true[:, None] + rng0.normal(0, 0.6, (n_ch, n_ev))     # noisy per event
    bundle = {"masked": masked, "labels": np.array([0] * 30 + [1] * 30),
              "channel_names": list(names), "bools": np.isfinite(masked)}
    out = loo_reproducibility(bundle, plane, n_split=25, rng=np.random.default_rng(3), sigma=None)
    assert out["field_rho"] > out["contact_rho"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_field_reversal.py -k "field_beats_contact" -v`
Expected: FAIL with `ImportError: cannot import name 'loo_reproducibility'`

- [ ] **Step 3: Write minimal implementation**

```python
def _loo_field_predict(names, plane_xy, values, support, sigma):
    """LOO kernel regression at each contact: value <- support-weighted mean of OTHER contacts."""
    pts = np.array([plane_xy[n] for n in names], float)
    v = np.array([values[n] for n in names], float)
    sup = np.array([support[n] for n in names], float)
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    W = sup[None, :] * np.exp(-d2 / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0.0)                      # LOO: exclude self
    den = W.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pred = np.where(den > 1e-12, (W @ v) / den, np.nan)
    return pred


def _class_split_rhos(masked, names, plane_xy, cols, sigma, rng):
    perm = rng.permutation(cols); half = perm.size // 2
    if half < 1:
        return None
    a, b = perm[:half], perm[half:]
    with np.errstate(invalid="ignore"):
        train = np.array([np.nanmean(masked[c, a]) if np.any(np.isfinite(masked[c, a])) else np.nan
                          for c in range(len(names))])
        held = np.array([np.nanmean(masked[c, b]) if np.any(np.isfinite(masked[c, b])) else np.nan
                         for c in range(len(names))])
    on_plane = [i for i, n in enumerate(names) if n in plane_xy]
    idx = np.array(on_plane, int)
    tv = {names[i]: train[i] for i in idx if np.isfinite(train[i])}
    sup = {names[i]: 1.0 for i in tv}
    order = list(tv.keys())
    loo = _loo_field_predict(order, plane_xy, tv, sup, sigma)
    loo_by = {n: loo[j] for j, n in enumerate(order)}
    common = [n for n in order if np.isfinite(held[names.index(n)]) and np.isfinite(loo_by[n])]
    if len(common) < 3:
        return None
    hv = np.array([held[names.index(n)] for n in common])
    cv = np.array([tv[n] for n in common])
    fv = np.array([loo_by[n] for n in common])
    return (float(spearmanr(cv, hv).correlation), float(spearmanr(fv, hv).correlation), len(common))


def loo_reproducibility(bundle, plane_ref, *, n_split=50, rng, sigma) -> dict:
    X, Y = make_plane_grid()
    if sigma is None:
        sigma = class_template_sigma(plane_ref, X=X, Y=Y)
    masked = bundle["masked"]; names = list(bundle["channel_names"])
    labels = np.asarray(bundle["labels"]); plane_xy = build_plane_xy(plane_ref)
    c_rhos, f_rhos, ncs = [], [], []
    for g in (0, 1):
        cols = np.where(labels == g)[0]
        for _ in range(n_split):
            r = _class_split_rhos(masked, names, plane_xy, cols, sigma, rng)
            if r is not None:
                c_rhos.append(r[0]); f_rhos.append(r[1]); ncs.append(r[2])
    return {"contact_rho": float(np.nanmean(c_rhos)) if c_rhos else float("nan"),
            "field_rho": float(np.nanmean(f_rhos)) if f_rhos else float("nan"),
            "n_contacts_common": int(np.median(ncs)) if ncs else 0}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_field_reversal.py -k "field_beats_contact" -v`
Expected: PASS (if flaky at the boundary, raise noise to 0.8 or n_split to 40 — the inequality is the invariant)

- [ ] **Step 5: Commit**

```bash
git add src/topic5_field_reversal.py tests/test_topic5_field_reversal.py
git commit -m "feat(topic5): LOO field-vs-contact reproducibility supplement"
```

---

### Task 7: Cohort binomial

**Files:**
- Modify: `src/topic5_field_reversal.py`
- Test: `tests/test_topic5_field_reversal.py`

**Interfaces:**
- Consumes: `scipy.stats.binomtest`
- Produces: `cohort_binomial(pass_flags) -> {"n":int,"k":int,"p_binom":float}` — n = non-degenerate subjects, k = passes, one-sided binomial vs 0.05.

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_field_reversal import cohort_binomial

def test_cohort_binomial():
    out = cohort_binomial([True, True, True, False, False, False, False, False])
    assert out["n"] == 8 and out["k"] == 3
    assert out["p_binom"] < 0.05        # 3/8 >> 5% expected
    assert cohort_binomial([])["n"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_field_reversal.py -k "cohort_binomial" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
from scipy.stats import binomtest


def cohort_binomial(pass_flags: Sequence[bool]) -> dict:
    flags = [bool(x) for x in pass_flags]
    n = len(flags); k = int(sum(flags))
    if n == 0:
        return {"n": 0, "k": 0, "p_binom": float("nan")}
    return {"n": n, "k": k, "p_binom": float(binomtest(k, n, 0.05, alternative="greater").pvalue)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_field_reversal.py -k "cohort_binomial" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/topic5_field_reversal.py tests/test_topic5_field_reversal.py
git commit -m "feat(topic5): cohort binomial over non-degenerate reversal passes"
```

---

### Task 8: Runner (broad loading, per-subject + cohort, bandwidth sweep)

**Files:**
- Create: `scripts/run_topic5_field_reversal.py`
- Test: `tests/test_topic5_field_reversal.py` (eligibility gate unit test only; full run is manual)

**Interfaces:**
- Consumes: all Task 1-7 functions; `load_event_labels_ranks`, `map_clusters_to_templates`, `class_aggregate_contact_values`, `make_plane_grid`. Reuses the exact loading sequence from `scripts/run_topic5_event_resolved_alignment.py:89-114` (planes → bundle → cluster map → reference plane), MINUS the ictal block.
- Produces: `results/topic5_ictal_recruitment/field_reversal/per_subject/{ds_sid}.json` + `cohort_summary.json`.

**Per-subject flow (broad):**
1. Require `GEOM_BROAD/{ds_sid}_t_a.json` and `_t_b.json`; else `status="no_broad_planes"`.
2. `bundle = load_event_labels_ranks(dataset, subject)` (catch `ValueError`→`c1_violation`, `NotImplementedError`→`narrow_unsupported`, `FileNotFoundError`→`load_error`).
3. `ta_rank/tb_rank` from each plane's `typical_rank` in `bundle["channel_names"]` order; `cmap = map_clusters_to_templates(c0, c1, ta_rank, tb_rank)`; if `cmap["ambiguous"]` → `status="cluster_map_ambiguous"`.
4. **Reference frame = the plane mapped to `t_a`** (P0): `plane_ref = {plane_a if cmap["map"][k]=="t_a"}`. Concretely `plane_ref = plane_a if "t_a" in cmap["map"].values() with plane_a` — since `plane_of={"t_a":plane_a,"t_b":plane_b}`, `plane_ref = plane_of["t_a"] = plane_a`. Label→class: `cav_for_ta = class_aggregate_contact_values(bundle, [k for k,v in cmap["map"].items() if v=="t_a"][0])`, `cav_for_tb` similarly. `cav0 := cav_for_ta`, `cav1 := cav_for_tb`.
5. Gate at primary sigma: `within_shaft_reversal_gate(plane_ref, cav0, cav1, X, Y, sigma=None, n_perm, rng, min_eff)`.
6. `channel_floor(...)`, `random_split_contrast(bundle, plane_ref, ...)`, `contact_reversal_gate(cav0, cav1, ...)`, `loo_reproducibility(bundle, plane_ref, ...)`.
7. **Bandwidth sweep:** re-run only `within_shaft_reversal_gate` at `sigma ∈ {0.5,1,2} × class_template_sigma(plane_ref)` → `sweep` dict. Primary result = the `1.0×` entry (identical to step 5).
8. Write per-subject JSON with all of the above + `cluster_map`, `n_channels`, `status="ok"`.

**Cohort:** collect `passed` from non-degenerate `ok` subjects → `cohort_binomial`; also field_rho vs contact_rho paired Wilcoxon (`scipy.stats.wilcoxon`) over `ok` subjects → `cohort_summary.json`.

**CLI:** mutually-exclusive required `--subjects ds_sid...` XOR `--cohort` (explicit gate; no implicit run). `--n-perm` (default 1000), `--n-split` (200), `--loo-split` (50), `--min-eff` (6), `--out`.

- [ ] **Step 1: Write the failing test (eligibility gate is pure enough to unit-test)**

```python
def test_reference_frame_is_ta_plane(monkeypatch):
    # cluster 1 maps to t_a -> plane_ref must be plane_a and cav0 must be class-1 aggregate
    from scripts import run_topic5_field_reversal as R
    cmap = {"map": {0: "t_b", 1: "t_a"}, "ambiguous": False}
    plane_a = {"channels": [{"name": "A1"}]}; plane_b = {"channels": [{"name": "B1"}]}
    ref, ta_label, tb_label = R.pick_reference(cmap, plane_a, plane_b)
    assert ref is plane_a and ta_label == 1 and tb_label == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_field_reversal.py -k "reference_frame" -v`
Expected: FAIL with `ModuleNotFoundError: scripts.run_topic5_field_reversal`

- [ ] **Step 3: Write the runner**

Mirror `scripts/run_topic5_event_resolved_alignment.py` for `_ROOT`/sys.path, `GEOM_BROAD`, `_vec_in_order`. Implement `pick_reference(cmap, plane_a, plane_b)`:

```python
def pick_reference(cmap, plane_a, plane_b):
    """P0 reference frame = the plane mapped to t_a. Returns (plane_ref, ta_label, tb_label)."""
    inv = {v: k for k, v in cmap["map"].items()}          # {"t_a":label, "t_b":label}
    plane_of = {"t_a": plane_a, "t_b": plane_b}
    return plane_of["t_a"], inv["t_a"], inv["t_b"]
```

Then `_run_subject(ds_sid, ...)` per the flow above (steps 1-8), and `main()` with the CLI. Write per-subject JSON. `--cohort` iterates the discovered broad subjects (glob `GEOM_BROAD/*_t_a.json`).

- [ ] **Step 4: Run test + a real single-subject smoke run**

Run: `pytest tests/test_topic5_field_reversal.py -k "reference_frame" -v` → PASS
Run: `python scripts/run_topic5_field_reversal.py --subjects epilepsiae_1077 --n-perm 200`
Expected: writes `results/topic5_ictal_recruitment/field_reversal/per_subject/epilepsiae_1077.json` with `status` in {`ok`,`cluster_map_ambiguous`,`no_broad_planes`,...}; if `ok`, prints signed_corr + percentile + degenerate_null.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_topic5_field_reversal.py tests/test_topic5_field_reversal.py
git commit -m "feat(topic5): field-reversal runner (broad, per-subject + cohort + sigma sweep)"
```

---

### Task 9: Plotter + figures README

**Files:**
- Create: `scripts/plot_topic5_field_reversal.py`
- Create: `results/topic5_ictal_recruitment/field_reversal/figures/README.md`

**Interfaces:**
- Consumes: per-subject JSONs + `cohort_summary.json` from Task 8. For the field panels, mirror `scripts/plot_topic5_event_resolved_fields.py` (shared-frame display via `_subject_display_frame`; **display-only** smoothing is fine here — figures may use VIS constants, stats do not).

**Panels (CLAUDE.md §7 — one question per panel, no redundancy):**
1. **Per-subject** (`per_subject/{ds_sid}.png`): TA_field | TB_field (shared frame) | observed signed_corr marked on its within-shaft null histogram + channel-floor null + random-split contrast strip. One figure answers "is this subject's TA/TB reversed beyond within-shaft?".
2. **Cohort null-forest** (`field_reversal_null_forest.png`): per subject, observed signed_corr vs its within-shaft null 5th-pctile, sorted; black = passed; footnote = `cohort_binomial` k/n + p. (mirror `field_concordance_null_forest`.)
3. **Head-to-head** (`field_vs_contact_headtohead.png`): field passes vs contact passes (paired), answering "does the field buy robustness?".
4. **Supplement** (`loo_reproducibility.png`): per-subject field_rho vs contact_rho paired, Wilcoxon in footnote.
5. **1146 case** (`case_1146_mechanism.png`): left = raw contact values + fitted direction; right = shared-frame field + candidate physical-axis readout. Wording: "candidate physical-axis readout", NOT "true axis". **Verify the phenomenon on real 1146 data first; if unsupported, switch subject or drop this panel** (spec §9).

- [ ] **Step 1** Write `figures/README.md` (Chinese, per-figure "展示什么 / 关注点", per AGENTS.md format) — AFTER the figures render, not before.
- [ ] **Step 2** Implement the plotter; render on the cohort output.
- [ ] **Step 3** Eyeball every figure (feedback: paper-grade self-contained; shared legend; no §X/bracket axis labels).
- [ ] **Step 4: Commit**

```bash
git add scripts/plot_topic5_field_reversal.py results/topic5_ictal_recruitment/field_reversal/figures/README.md
git commit -m "feat(topic5): field-reversal figures + README"
```

---

### Task 10: Docs — FIGURE_INDEX + topic5 pointer

**Files:**
- Modify: `results/FIGURE_INDEX.md` (append the new figure dir)
- Modify: `docs/topic5_seizure_subtyping.md` (one pointer line under §3.0 network-axis: "TA/TB 场反向门 = A-line 上游 gate；broad-only；结果见 archive")
- Create: `docs/archive/topic5/field_reversal_<result-date>.md` (results archive — filled after the cohort run, with plain-language 测了什么/怎么测/揭示了什么 per CLAUDE.md §8; leave a `preliminary, pending review` tag until the user signs off)

- [ ] **Step 1** Append FIGURE_INDEX entry.
- [ ] **Step 2** Add the topic5 pointer line.
- [ ] **Step 3** Write the archive doc skeleton (numbers filled post-run).
- [ ] **Step 4: Commit**

```bash
git add results/FIGURE_INDEX.md docs/topic5_seizure_subtyping.md docs/archive/topic5/field_reversal_*.md
git commit -m "docs(topic5): index + pointer for TA/TB field-reversal gate"
```

---

## Self-Review

**1. Spec coverage:**
- §2 claim tiers/red-lines → archive doc (Task 10) + reporting-rule comments; §3.1 shared frame (P0) → Task 2 + Task 8 `pick_reference` + TDD. §3.2 raw value/no-mirror → Tasks 1-2 TDD. §4 within-shaft primary + effective_n + degenerate + pass rule → Task 3. channel floor + random-split → Task 4. §5 contact head-to-head → Task 5 + plot Task 9. §6 LOO supplement (common-support, LOO exclude) → Task 6. §7 bandwidth sweep → Task 8 step 7. §8 broad-only + no-ictal eligibility → Task 8 flow. §9 1146 wording/verify-first → Task 9 panel 5. §10 no-VIS smoother → satisfied by reusing `class_template_sigma`/`field_from_contact_values` (Task 2). §12 TDD list → covered across Tasks 1-7. **No gaps.**
- **Deferred (out of v1 scope, stated in spec §8):** narrow substrate (loader raises `NotImplementedError`).

**2. Placeholder scan:** stat Tasks 1-7 carry full test+impl code. Tasks 8-9 (IO/plot) specify exact loader calls + panel contract and reference the mirror scripts — acceptable for a skilled dev; no "TBD/handle edge cases".

**3. Type consistency:** `signed_reversal_corr`→`signed_corr`; gate returns `signed_corr`/`percentile`/`passed`/`degenerate_null`; contact gate returns `signed_spearman`; supplement returns `field_rho`/`contact_rho`; cohort returns `n`/`k`/`p_binom`. `build_reversal_fields` returns `field0`/`field1`/`sigma`/`names_used`, consumed consistently by Tasks 3-4. Consistent across tasks.

---

## Execution Handoff

(Filled by the writing-plans skill's handoff step in chat.)
