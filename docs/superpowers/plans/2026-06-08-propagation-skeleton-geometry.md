# Propagation Skeleton Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the per-subject 3D geometric skeleton of the interictal HFO propagation template — source/sink core compactness, source–sink axis length, perpendicular spread, and the along-axis temporal-stereotypy profile — as descriptive model-input numbers (not a hypothesis test, not a SOZ localizer).

**Architecture:** One pure-geometry module (`src/propagation_skeleton_geometry.py`, no I/O), one runner that loads each subject (reusing existing propagation-event + coord + rank-displacement loaders), routes swap subjects, applies the numeric eligibility gate, and writes per-subject + cohort JSON, and one plotter. All geometry is computed in an axis-coordinate frame that separates along-propagation distance from off-axis (patch-boundary) distance.

**Tech Stack:** Python, numpy, scipy.stats; reuse `src/interictal_propagation.load_subject_propagation_events`, `src/lagpat_rank_audit.mask_phantom_ranks/build_masked_kmeans_features`, `src/low_rate_template_stability.align_template_events`, `src/seeg_coord_loader.load_subject_coords`, `src/sef_itp_phase1.{pairwise_3d_euclidean,_mean_pairwise_distance}`. Matplotlib for figures. pytest for TDD.

**Spec:** `docs/superpowers/specs/2026-06-08-sef-hfo-propagation-skeleton-geometry-design.md` (committed 35ed111). Re-read each spec section at the matching task boundary (CLAUDE.md §5).

---

## File Structure

- **Create `src/propagation_skeleton_geometry.py`** — pure functions, no file I/O. Responsibilities: parse shaft id; build phantom-safe source/sink cores + eligibility tier; build the axis-coordinate frame (centroids, axis, per-channel along/off coordinates); core radii (RMS / MEB / max-pairwise); perpendicular spread (RMS/p75/p90) + participation sweep; sampling-geometry classifier (1D vs distributed); per-channel stereotypy + event-size-matched null; along-axis profile binning.
- **Create `scripts/run_propagation_skeleton_geometry.py`** — per-subject loading (propagation events + coords + swap status), swap routing (dominant-cluster axis), tier gating, calls the module, writes `results/topic4_sef_hfo/skeleton_geometry/per_subject/{ds}_{subj}.json` + `cohort_summary.json`.
- **Create `scripts/plot_propagation_skeleton_geometry.py`** — 3 figures + `figures/README.md`.
- **Create `tests/test_propagation_skeleton_geometry.py`** — unit tests on synthetic geometry with known answers.

**Cohort (from spec §2, verified):** 11 Yuquan (SEEG) + 20 Epilepsiae (SEEG+ECoG) have both a phantom-safe endpoint file and 3D coords. `n_eff` = participating (template non-NaN) AND coord-mapped channels. Tiers: `n_eff≥7`→k=3 primary; `n_eff∈{5,6}`→k=2 fallback (not pooled); `n_eff<5` or empty interior or degenerate axis → descriptive-only (excluded from cohort stats).

---

## Task 1: Module scaffold + shaft parsing

**Files:**
- Create: `src/propagation_skeleton_geometry.py`
- Test: `tests/test_propagation_skeleton_geometry.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
from src.propagation_skeleton_geometry import parse_shaft


def test_parse_shaft_depth_and_grid_and_junk():
    assert parse_shaft("D13") == ("D", 13)
    assert parse_shaft("FLA2") == ("FLA", 2)
    assert parse_shaft("GA1") == ("GA", 1)
    assert parse_shaft("A1'") == ("A'", 1)   # prime-marked shaft
    assert parse_shaft("EKG") == (None, None)
    assert parse_shaft("") == (None, None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_propagation_skeleton_geometry.py::test_parse_shaft_depth_and_grid_and_junk -v`
Expected: FAIL with `ImportError` / `cannot import name 'parse_shaft'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Interictal propagation skeleton geometry (axis-frame, descriptive model-input).

Spec: docs/superpowers/specs/2026-06-08-sef-hfo-propagation-skeleton-geometry-design.md

All functions are pure (no file I/O). Phantom-safety: source/sink cores are
derived from a template axis whose non-participating channels are NaN, so a
phantom can never enter a core (NaN sorts out of nanargsort).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_NAME_RE = re.compile(r"^([A-Za-z]+'?)\s*(\d+)$")


def parse_shaft(channel_name: str) -> Tuple[Optional[str], Optional[int]]:
    """(shaft_prefix, ordinal) from a channel name; (None, None) if unparseable."""
    m = _NAME_RE.match(str(channel_name).strip())
    if not m:
        return (None, None)
    return (m.group(1), int(m.group(2)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_propagation_skeleton_geometry.py::test_parse_shaft_depth_and_grid_and_junk -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/propagation_skeleton_geometry.py tests/test_propagation_skeleton_geometry.py
git commit -m "feat(skeleton-geom): module scaffold + shaft parsing"
```

---

## Task 2: Endpoint cores + numeric eligibility gate

Re-read spec §2 (tiers) and §3 (phantom-safe cores) before this task.

**Files:**
- Modify: `src/propagation_skeleton_geometry.py`
- Test: `tests/test_propagation_skeleton_geometry.py`

- [ ] **Step 1: Write the failing test**

```python
from src.propagation_skeleton_geometry import build_endpoint_cores


def _axis(n, participating):
    """template axis: ascending ranks on participating idx, NaN elsewhere."""
    ax = np.full(n, np.nan)
    ax[participating] = np.arange(len(participating), dtype=float)
    return ax


def test_cores_k3_primary_tier():
    # 8 participating + coord-mapped channels -> n_eff=8 -> k=3
    ax = _axis(8, np.arange(8))
    eligible = np.ones(8, bool)
    r = build_endpoint_cores(ax, eligible, k_primary=3)
    assert r["tier"] == "primary"
    assert r["k_used"] == 3
    assert r["n_eff"] == 8
    assert sorted(r["source_idx"]) == [0, 1, 2]      # lowest ranks
    assert sorted(r["sink_idx"]) == [5, 6, 7]        # highest ranks
    assert len(r["interior_idx"]) == 2               # 3,4
    assert set(r["source_idx"]).isdisjoint(r["sink_idx"])


def test_cores_phantom_never_in_core():
    # channel 0 is NON-participating (NaN) and would be index-0; must NOT be source
    n = 9
    ax = np.full(n, np.nan)
    ax[1:8] = np.arange(7, dtype=float)   # participating = idx 1..7 (n_eff=7)
    eligible = ~np.isnan(ax)
    r = build_endpoint_cores(ax, eligible, k_primary=3)
    assert 0 not in r["source_idx"]
    assert 8 not in r["sink_idx"]         # idx 8 is NaN too
    assert all(eligible[i] for i in r["source_idx"] + r["sink_idx"])


def test_cores_fallback_k2_when_neff_5_or_6():
    ax = _axis(6, np.arange(6))
    r = build_endpoint_cores(ax, np.ones(6, bool), k_primary=3)
    assert r["tier"] == "fallback"
    assert r["k_used"] == 2
    assert len(r["interior_idx"]) == 2


def test_cores_descriptive_only_when_neff_below_5():
    ax = _axis(4, np.arange(4))
    r = build_endpoint_cores(ax, np.ones(4, bool), k_primary=3)
    assert r["tier"] == "descriptive_only"
    assert r["source_idx"] == [] and r["sink_idx"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_propagation_skeleton_geometry.py -k cores -v`
Expected: FAIL with `cannot import name 'build_endpoint_cores'`.

- [ ] **Step 3: Write minimal implementation**

```python
def build_endpoint_cores(
    template_axis: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    k_primary: int = 3,
) -> Dict[str, object]:
    """Source/sink cores from a phantom-safe template axis + eligibility tier.

    eligible_mask = participating (template non-NaN) AND coord-mapped.
    Cores are the k lowest / k highest template-axis channels AMONG eligible.
    Tier gate (spec §2): n_eff>=7 -> k=3 primary; n_eff in {5,6} -> k=2
    fallback; else descriptive_only (no cores, excluded from cohort stats).
    """
    template_axis = np.asarray(template_axis, dtype=float)
    eligible_mask = np.asarray(eligible_mask, dtype=bool)
    eligible_idx = np.where(eligible_mask & ~np.isnan(template_axis))[0]
    n_eff = int(eligible_idx.size)

    base = {
        "n_eff": n_eff,
        "k_used": 0,
        "tier": "descriptive_only",
        "source_idx": [],
        "sink_idx": [],
        "interior_idx": [],
    }
    if n_eff >= 7:
        k, tier = k_primary, "primary"
    elif n_eff in (5, 6):
        k, tier = 2, "fallback"
    else:
        return base
    # cores must be disjoint with a non-empty interior: 2k < n_eff
    if 2 * k >= n_eff:
        return base

    order = eligible_idx[np.argsort(template_axis[eligible_idx], kind="stable")]
    source = sorted(int(i) for i in order[:k])
    sink = sorted(int(i) for i in order[-k:])
    interior = sorted(int(i) for i in order[k:-k])
    base.update(k_used=k, tier=tier, source_idx=source, sink_idx=sink,
                interior_idx=interior)
    return base
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_propagation_skeleton_geometry.py -k cores -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(skeleton-geom): phantom-safe endpoint cores + numeric eligibility gate"
```

---

## Task 3: Axis-coordinate frame (centroids, axis, along/off per channel)

Re-read spec §1 (axis frame) before this task.

**Files:**
- Modify: `src/propagation_skeleton_geometry.py`
- Test: `tests/test_propagation_skeleton_geometry.py`

- [ ] **Step 1: Write the failing test**

```python
from src.propagation_skeleton_geometry import compute_axis_frame


def test_axis_frame_along_and_off_are_orthogonal_decomposition():
    # source core at x=0, sink core at x=10 (along +x). A channel at (5, 3, 0)
    # must have along=5 (projection) and off=3 (perp distance).
    coords = np.array([
        [0., 0., 0.], [0., 0., 0.],   # source core (centroid x=0)
        [10., 0., 0.], [10., 0., 0.],  # sink core (centroid x=10)
        [5., 3., 0.],                  # test channel
    ])
    fr = compute_axis_frame(coords, source_idx=[0, 1], sink_idx=[2, 3])
    assert np.allclose(fr["axis_length"], 10.0)
    assert np.allclose(fr["along_axis"][4], 5.0)
    assert np.allclose(fr["off_axis"][4], 3.0)
    # on-axis channel -> off == 0
    assert np.allclose(fr["off_axis"][0], 0.0)


def test_axis_frame_degenerate_axis_flagged():
    coords = np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
    fr = compute_axis_frame(coords, source_idx=[0, 1], sink_idx=[2, 3])
    assert fr["degenerate_axis"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_propagation_skeleton_geometry.py -k axis_frame -v`
Expected: FAIL (`cannot import name 'compute_axis_frame'`).

- [ ] **Step 3: Write minimal implementation**

```python
def compute_axis_frame(
    coords: np.ndarray,
    source_idx: Sequence[int],
    sink_idx: Sequence[int],
) -> Dict[str, object]:
    """Axis frame from source/sink core centroids.

    along_axis[c] = (p_c - source_centroid) . unit_axis   (0 at source, L at sink)
    off_axis[c]   = || (p_c - source_centroid) - along*unit_axis ||
    Channels with NaN coords get NaN along/off. Coincident centroids ->
    degenerate_axis=True and along/off all NaN.
    """
    coords = np.asarray(coords, dtype=float)
    src_c = np.nanmean(coords[list(source_idx)], axis=0)
    snk_c = np.nanmean(coords[list(sink_idx)], axis=0)
    axis = snk_c - src_c
    L = float(np.linalg.norm(axis))
    n = coords.shape[0]
    along = np.full(n, np.nan)
    off = np.full(n, np.nan)
    degenerate = L < 1e-9
    if not degenerate:
        u = axis / L
        rel = coords - src_c                      # (n,3)
        along = rel @ u                            # (n,)
        perp_vec = rel - np.outer(along, u)        # (n,3)
        off = np.linalg.norm(perp_vec, axis=1)
        bad = np.isnan(coords).any(axis=1)
        along[bad] = np.nan
        off[bad] = np.nan
    return {
        "source_centroid": src_c.tolist(),
        "sink_centroid": snk_c.tolist(),
        "axis_length": L,
        "along_axis": along,
        "off_axis": off,
        "degenerate_axis": bool(degenerate),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_propagation_skeleton_geometry.py -k axis_frame -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(skeleton-geom): axis-coordinate frame (along/off decomposition)"
```

---

## Task 4: Core radii (RMS / MEB / max-pairwise) — bimodal-core guard

Re-read spec §4 deliverable 1 + advisor #3 (bimodal core).

**Files:**
- Modify: `src/propagation_skeleton_geometry.py`
- Test: `tests/test_propagation_skeleton_geometry.py`

- [ ] **Step 1: Write the failing test**

```python
from src.propagation_skeleton_geometry import core_radii


def test_core_radii_compact_vs_split():
    centroid = np.array([0., 0., 0.])
    compact = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
    r = core_radii(compact, centroid)
    assert r["rms_mm"] == pytest.approx(1.0, abs=1e-9)
    assert r["max_pairwise_mm"] == pytest.approx(2.0, abs=1e-6)   # [1,0,0]-[-1,0,0]
    assert r["meb_mm"] == pytest.approx(1.0, abs=1e-6)            # right triangle, hyp=2

    # split core: two contacts 20mm apart -> centroid in the gap.
    split = np.array([[10., 0., 0.], [-10., 0., 0.]])
    split_centroid = split.mean(axis=0)
    rs = core_radii(split, split_centroid)
    assert rs["max_pairwise_mm"] == pytest.approx(20.0, abs=1e-6)
    assert rs["meb_mm"] == pytest.approx(10.0, abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_propagation_skeleton_geometry.py -k core_radii -v`
Expected: FAIL (`cannot import name 'core_radii'`).

- [ ] **Step 3: Write minimal implementation**

```python
def _meb_radius(points: np.ndarray) -> float:
    """Exact min-enclosing-ball radius for small point sets (k<=3 exact;
    k>=4 Ritter upper bound). Cores here are k in {2,3}."""
    pts = np.asarray(points, dtype=float)
    pts = pts[~np.isnan(pts).any(axis=1)]
    m = pts.shape[0]
    if m == 0:
        return float("nan")
    if m == 1:
        return 0.0
    if m == 2:
        return float(np.linalg.norm(pts[0] - pts[1]) / 2.0)
    if m == 3:
        a, b, c = pts
        # MEB of 3 points: longest-side diameter if triangle non-acute,
        # else circumradius.
        sides = np.array([np.linalg.norm(b - c),
                          np.linalg.norm(a - c),
                          np.linalg.norm(a - b)])
        longest = sides.max()
        s = sides
        # non-acute (obtuse/right) at the vertex opposite the longest side:
        i = int(np.argmax(sides))
        others = [s[j] for j in range(3) if j != i]
        if longest ** 2 >= others[0] ** 2 + others[1] ** 2:
            return float(longest / 2.0)
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        if area < 1e-12:
            return float(longest / 2.0)
        return float((s[0] * s[1] * s[2]) / (4.0 * area))
    # Ritter upper bound (defensive; not expected for k in {2,3})
    center = pts.mean(axis=0)
    return float(np.max(np.linalg.norm(pts - center, axis=1)))


def core_radii(core_coords: np.ndarray, centroid: np.ndarray) -> Dict[str, object]:
    """RMS-to-centroid (primary), MEB, and max-pairwise of a core point set.

    RMS-to-centroid can read a split core's gap as a fake-large 'compact
    radius'; MEB + max-pairwise expose that (advisor #3).
    """
    pts = np.asarray(core_coords, dtype=float)
    valid = pts[~np.isnan(pts).any(axis=1)]
    centroid = np.asarray(centroid, dtype=float)
    if valid.shape[0] == 0:
        return {"rms_mm": float("nan"), "meb_mm": float("nan"),
                "max_pairwise_mm": float("nan")}
    rms = float(np.sqrt(np.mean(np.sum((valid - centroid) ** 2, axis=1))))
    if valid.shape[0] < 2:
        maxpair = 0.0
    else:
        diff = valid[:, None, :] - valid[None, :, :]
        maxpair = float(np.linalg.norm(diff, axis=-1).max())
    return {"rms_mm": rms, "meb_mm": _meb_radius(valid),
            "max_pairwise_mm": maxpair}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_propagation_skeleton_geometry.py -k core_radii -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(skeleton-geom): core radii RMS+MEB+max-pairwise (bimodal-core guard)"
```

---

## Task 5: Perpendicular spread + sampling-geometry classifier

Re-read spec §4 deliverable 3 + advisor #1 (width = participation lower bound) + the 1D-degeneracy flag.

**Files:**
- Modify: `src/propagation_skeleton_geometry.py`
- Test: `tests/test_propagation_skeleton_geometry.py`

- [ ] **Step 1: Write the failing test**

```python
from src.propagation_skeleton_geometry import perp_spread, classify_sampling_geometry


def test_perp_spread_rms_p75_p90():
    off = np.array([0., 0., 4., 8., np.nan])
    participating = np.array([True, True, True, True, False])
    s = perp_spread(off, participating)
    assert s["rms_mm"] == pytest.approx(np.sqrt((0 + 0 + 16 + 64) / 4.0), abs=1e-9)
    assert s["p75_mm"] == pytest.approx(np.percentile([0, 0, 4, 8], 75), abs=1e-9)
    assert s["p90_mm"] == pytest.approx(np.percentile([0, 0, 4, 8], 90), abs=1e-9)
    assert s["n"] == 4


def test_classify_1d_single_shaft():
    names = ["A1", "A2", "A3", "A4"]
    part = np.ones(4, bool)
    off = np.array([0., 0.1, 0.0, 0.2])     # all near-axis
    g = classify_sampling_geometry(names, part, off, spacing_mm=3.5)
    assert g["geometry"] == "1D"
    assert g["n_shafts"] == 1
    assert g["measurable"] is False


def test_classify_distributed_multi_shaft():
    names = ["A1", "A2", "B1", "C3"]
    part = np.ones(4, bool)
    off = np.array([0., 1., 9., 12.])
    g = classify_sampling_geometry(names, part, off, spacing_mm=3.5)
    assert g["geometry"] == "distributed"
    assert g["n_shafts"] == 3
    assert g["measurable"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_propagation_skeleton_geometry.py -k "perp_spread or classify" -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
def perp_spread(off_axis: np.ndarray, participating_mask: np.ndarray) -> Dict[str, object]:
    """RMS (primary) + p75/p90 (robust sensitivity) of off-axis distance over
    PARTICIPATING channels. NOTE (advisor #1): off-axis spread of participating
    channels is a participation+sampling-bounded LOWER BOUND, not 'patch width'.
    Report as 'participating-channel perpendicular spread'.
    """
    off = np.asarray(off_axis, dtype=float)
    part = np.asarray(participating_mask, dtype=bool)
    vals = off[part & ~np.isnan(off)]
    if vals.size == 0:
        return {"rms_mm": float("nan"), "p75_mm": float("nan"),
                "p90_mm": float("nan"), "n": 0}
    return {
        "rms_mm": float(np.sqrt(np.mean(vals ** 2))),
        "p75_mm": float(np.percentile(vals, 75)),
        "p90_mm": float(np.percentile(vals, 90)),
        "n": int(vals.size),
    }


def classify_sampling_geometry(
    channel_names: Sequence[str],
    participating_mask: np.ndarray,
    off_axis: np.ndarray,
    *,
    spacing_mm: float = 3.5,
) -> Dict[str, object]:
    """1D (single-shaft / collinear) vs distributed sampling. Width is NOT
    measurable for 1D subjects (axis length still is). spec §4."""
    part = np.asarray(participating_mask, dtype=bool)
    off = np.asarray(off_axis, dtype=float)
    names = [channel_names[i] for i in np.where(part)[0]]
    shafts = {parse_shaft(nm)[0] for nm in names} - {None}
    vals = off[part & ~np.isnan(off)]
    p90 = float(np.percentile(vals, 90)) if vals.size else 0.0
    one_d = (len(shafts) <= 1) or (p90 < spacing_mm)
    return {
        "geometry": "1D" if one_d else "distributed",
        "n_shafts": int(len(shafts)),
        "p90_off_mm": p90,
        "measurable": not one_d,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_propagation_skeleton_geometry.py -k "perp_spread or classify" -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(skeleton-geom): perpendicular spread (RMS/p75/p90) + 1D/distributed classifier"
```

---

## Task 6: Perpendicular-spread participation sweep (advisor #1 control)

Re-read advisor #1: width has no control unless we show how it moves with the participation threshold.

**Files:**
- Modify: `src/propagation_skeleton_geometry.py`
- Test: `tests/test_propagation_skeleton_geometry.py`

- [ ] **Step 1: Write the failing test**

```python
from src.propagation_skeleton_geometry import perp_spread_participation_sweep


def test_participation_sweep_tightens_with_threshold():
    # far channels (large off) have FEW events; raising the event-count
    # threshold drops them and shrinks the spread.
    off = np.array([0., 1., 2., 30.])
    full_count = np.array([100., 90., 80., 5.])   # far channel = 5 events
    sweep = perp_spread_participation_sweep(off, full_count, thresholds=[1, 10])
    assert sweep[0]["threshold"] == 1 and sweep[0]["n"] == 4
    assert sweep[1]["threshold"] == 10 and sweep[1]["n"] == 3   # 30mm dropped
    assert sweep[1]["rms_mm"] < sweep[0]["rms_mm"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_propagation_skeleton_geometry.py -k participation_sweep -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
def perp_spread_participation_sweep(
    off_axis: np.ndarray,
    full_count: np.ndarray,
    *,
    thresholds: Sequence[int] = (1, 5, 10, 20),
) -> List[Dict[str, object]]:
    """Off-axis spread as the per-channel event-count threshold rises. Shows
    whether 'width' is just a participation/rate artifact (advisor #1)."""
    off = np.asarray(off_axis, dtype=float)
    cnt = np.asarray(full_count, dtype=float)
    out = []
    for t in thresholds:
        keep = (cnt >= t) & ~np.isnan(off)
        s = perp_spread(off, keep)
        out.append({"threshold": int(t), "n": s["n"], "rms_mm": s["rms_mm"],
                    "p90_mm": s["p90_mm"]})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_propagation_skeleton_geometry.py -k participation_sweep -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(skeleton-geom): perp-spread participation sweep (width-vs-rate control)"
```

---

## Task 7: Per-channel stereotypy + event-size-matched null

Re-read spec §5 (along-axis temporal layer, participation-matched null).

**Files:**
- Modify: `src/propagation_skeleton_geometry.py`
- Test: `tests/test_propagation_skeleton_geometry.py`

- [ ] **Step 1: Write the failing test**

```python
from src.propagation_skeleton_geometry import (
    channel_stereotypy, channel_stereotypy_excess,
)


def test_stereotypy_reproducible_high_random_low():
    # masked[c, e] = normalized within-event rank, NaN = not participating.
    # ch0 always at 0.0 (perfectly reproducible); ch1 jumps 0/1 (anti-stereotyped).
    masked = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ])
    s = channel_stereotypy(masked)
    assert s[0] == pytest.approx(1.0, abs=1e-9)     # std 0 -> 1
    assert s[1] < s[0]


def test_stereotypy_excess_z_separates_signal_from_chance():
    # 10 channels all participating -> event size 10, so the integer-position
    # matched null is continuous-enough to match the observed ranks (the null's
    # event sizes MUST match the data or the z is meaningless).
    rng = np.random.default_rng(0)
    n_ev, n_ch = 200, 10
    masked = rng.random((n_ch, n_ev))               # filler channels = uniform
    masked[0] = 0.1 + rng.normal(0, 0.01, n_ev)     # reproducible (front)
    masked[1] = rng.random(n_ev)                     # random = chance
    bools = np.ones((n_ch, n_ev), bool)
    z = channel_stereotypy_excess(masked, bools, rng=np.random.default_rng(1),
                                  n_null=200)
    assert z[0] > 3.0        # clearly above chance
    assert abs(z[1]) < 2.0   # ~chance
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_propagation_skeleton_geometry.py -k stereotypy -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
def channel_stereotypy(masked: np.ndarray) -> np.ndarray:
    """Per-channel stereotypy = 1 - 2*std(normalized within-event rank) over
    participating events. 1 = perfectly reproducible position; ~0.42 = chance
    (uniform). NaN if <2 participating events."""
    masked = np.asarray(masked, dtype=float)
    n_ch = masked.shape[0]
    out = np.full(n_ch, np.nan)
    for c in range(n_ch):
        vals = masked[c][~np.isnan(masked[c])]
        if vals.size >= 2:
            out[c] = 1.0 - 2.0 * float(np.std(vals))
    return out


def channel_stereotypy_excess(
    masked: np.ndarray,
    bools: np.ndarray,
    *,
    rng: np.random.Generator,
    n_null: int = 200,
) -> np.ndarray:
    """Event-size-matched null z-score of per-channel stereotypy.

    For each channel c with participating events E_c, the null draws, per
    event e in E_c, a uniform random within-event rank position
    (integer in [0, m_e-1] normalized by m_e-1, m_e = #participants in e),
    recomputes stereotypy, repeats n_null times. z = (obs - null_mean)/null_std.
    Fewer events -> wider null -> smaller z (participation control).
    """
    masked = np.asarray(masked, dtype=float)
    bools = np.asarray(bools, dtype=bool)
    obs = channel_stereotypy(masked)
    n_ch, n_ev = masked.shape
    ev_sizes = bools.sum(axis=0).astype(float)        # m_e per event
    z = np.full(n_ch, np.nan)
    for c in range(n_ch):
        ev_idx = np.where(~np.isnan(masked[c]))[0]
        if ev_idx.size < 2:
            continue
        m = ev_sizes[ev_idx]
        denom = np.maximum(m - 1.0, 1.0)
        null_vals = np.empty(n_null)
        for j in range(n_null):
            draws = rng.integers(0, np.maximum(m.astype(int), 1)) / denom
            null_vals[j] = 1.0 - 2.0 * float(np.std(draws))
        mu, sd = float(null_vals.mean()), float(null_vals.std())
        if sd > 1e-9:
            z[c] = (obs[c] - mu) / sd
    return z
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_propagation_skeleton_geometry.py -k stereotypy -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(skeleton-geom): per-channel stereotypy + event-size-matched null z"
```

---

## Task 8: Along-axis stereotypy profile binning

Re-read spec §5 (along-axis profile is this round's temporal deliverable; off-axis stays descriptive).

**Files:**
- Modify: `src/propagation_skeleton_geometry.py`
- Test: `tests/test_propagation_skeleton_geometry.py`

- [ ] **Step 1: Write the failing test**

```python
from src.propagation_skeleton_geometry import axis_stereotypy_profile


def test_axis_profile_bins_by_along_axis():
    along = np.array([0.0, 2.0, 4.0, 9.0, np.nan])
    excess = np.array([3.0, 2.0, 1.0, -0.5, 5.0])   # last is NaN-along -> dropped
    prof = axis_stereotypy_profile(along, excess, edges=[0, 5, 10])
    assert len(prof) == 2
    assert prof[0]["n"] == 3 and prof[0]["mean_excess"] == pytest.approx(2.0)
    assert prof[1]["n"] == 1 and prof[1]["mean_excess"] == pytest.approx(-0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_propagation_skeleton_geometry.py -k axis_profile -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
def axis_stereotypy_profile(
    along_axis: np.ndarray,
    stereotypy_excess: np.ndarray,
    *,
    edges: Sequence[float],
) -> List[Dict[str, object]]:
    """Mean stereotypy-excess binned by along-axis coordinate."""
    a = np.asarray(along_axis, dtype=float)
    z = np.asarray(stereotypy_excess, dtype=float)
    ok = ~np.isnan(a) & ~np.isnan(z)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = ok & (a >= lo) & (a < hi)
        vals = z[sel]
        out.append({
            "a_lo": float(lo), "a_hi": float(hi), "n": int(vals.size),
            "mean_excess": float(vals.mean()) if vals.size else float("nan"),
            "sd_excess": float(vals.std()) if vals.size else float("nan"),
        })
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_propagation_skeleton_geometry.py -k axis_profile -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(skeleton-geom): along-axis stereotypy-excess profile binning"
```

---

## Task 9: Runner — load, swap-route, tier-gate, write per-subject + cohort JSON

Re-read spec §3 (swap routing, template source), §6 (cross-dataset mm), §7 (SOZ descriptive), §8 (outputs).

**Files:**
- Create: `scripts/run_propagation_skeleton_geometry.py`
- Test: `tests/test_propagation_skeleton_geometry.py` (one integration smoke test guarded by data availability)

- [ ] **Step 1: Write the failing integration smoke test**

```python
import os, json, subprocess, sys, pathlib


def test_runner_smoke_two_subjects(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    if not (root / "results/interictal_propagation_masked/rank_displacement/"
            "per_subject/epilepsiae_253.json").exists():
        pytest.skip("masked rank-displacement artifacts not present")
    out = tmp_path / "skeleton_geometry"
    r = subprocess.run(
        [sys.executable, str(root / "scripts/run_propagation_skeleton_geometry.py"),
         "--subjects", "epilepsiae:253", "yuquan:chengshuai",
         "--out", str(out)],
        capture_output=True, text=True, cwd=str(root))
    assert r.returncode == 0, r.stderr
    cohort = json.loads((out / "cohort_summary.json").read_text())
    # phantom-safe cores: zero violations reported
    assert cohort["phantom_core_violations"] == 0
    # dataset stratification present
    assert "by_dataset" in cohort and set(cohort["by_dataset"]) <= {"yuquan", "epilepsiae"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_propagation_skeleton_geometry.py -k runner_smoke -v`
Expected: FAIL (script does not exist) or SKIP if artifacts absent. If it SKIPs, you must still implement Step 3 and verify via Step 4's manual run.

- [ ] **Step 3: Write the runner**

```python
#!/usr/bin/env python3
"""Per-subject interictal propagation skeleton geometry (descriptive model-input).

Spec: docs/superpowers/specs/2026-06-08-sef-hfo-propagation-skeleton-geometry-design.md
Outputs: results/topic4_sef_hfo/skeleton_geometry/{per_subject/{ds}_{subj}.json, cohort_summary.json}
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse, json, sys, warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")

from sklearn.cluster import KMeans

from src.interictal_propagation import load_subject_propagation_events
from src.lagpat_rank_audit import mask_phantom_ranks, build_masked_kmeans_features
from src.low_rate_template_stability import align_template_events
from src.seeg_coord_loader import load_subject_coords
from src import propagation_skeleton_geometry as G

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
RANKDISP = _ROOT / "results/interictal_propagation_masked/rank_displacement/per_subject"
SOZ_JSON = {ds: _ROOT / f"results/{ds}_soz_core_channels.json" for ds in ("yuquan", "epilepsiae")}
ALL_SUBJECTS = None  # default cohort discovered from RANKDISP + coord availability


def _subject_dir(ds, subj):
    return YUQUAN_ROOT / subj if ds == "yuquan" else EPILEPSIAE_ROOT / subj / "all_recs"


def _swap_class(ds, subj):
    f = RANKDISP / f"{ds}_{subj}.json"
    if not f.exists():
        return "none"
    d = json.loads(f.read_text())
    pairs = d.get("pairs") or [{}]
    return (((pairs[0].get("swap_sweep") or {}).get("swap_class")) or "none")


def _cluster_axis(masked, labels, cluster):
    sel = labels == cluster
    sub = masked[:, sel]
    return np.array([np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in sub])


def _soz_set(ds, subj):
    f = SOZ_JSON[ds]
    if not f.exists():
        return set()
    d = json.loads(f.read_text())
    entry = d.get(subj) or d.get(str(subj)) or {}
    return set(entry.get("core_channels", entry if isinstance(entry, list) else []))


def process_subject(ds, subj):
    ev = load_subject_propagation_events(_subject_dir(ds, subj))
    if not ev["channel_names"] or np.asarray(ev["ranks"]).size == 0:
        return {"dataset": ds, "subject": subj, "status": "no_events"}
    names = list(ev["channel_names"])
    ranks = np.asarray(ev["ranks"], float)
    bools = np.asarray(ev["bools"]) > 0
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    labels = KMeans(n_clusters=2, n_init=5, random_state=0).fit_predict(
        build_masked_kmeans_features(ranks, bools))
    aligned, _ = align_template_events(masked, labels)

    swap_class = _swap_class(ds, subj)
    if swap_class in ("strict", "candidate"):
        dom = 0 if (labels == 0).sum() >= (labels == 1).sum() else 1
        template_axis = _cluster_axis(masked, labels, dom)
        template_source = f"dominant_cluster_{dom}"
    else:
        template_axis = np.array(
            [np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in aligned])
        template_source = "full_recording"

    cr = load_subject_coords(ds, subj, names)
    coords = np.asarray(cr.coords_array_in_requested_order, float)
    mapped = np.asarray(cr.mapped_mask_in_requested_order, bool)
    eligible = (~np.isnan(template_axis)) & mapped

    cores = G.build_endpoint_cores(template_axis, eligible, k_primary=3)
    rec = {
        "dataset": ds, "subject": subj, "status": "ok",
        "swap_class": swap_class, "template_source": template_source,
        "coord_space": cr.coord_space, "n_eff": cores["n_eff"],
        "k_used": cores["k_used"], "eligibility_tier": cores["tier"],
        "channel_names": names,
        "missing_coords": [m.channel for m in cr.missing],
    }
    # phantom-safe core assertion (acceptance §9.1)
    rec["phantom_core_violation"] = bool(
        any(not eligible[i] for i in cores["source_idx"] + cores["sink_idx"]))

    if cores["tier"] == "descriptive_only":
        return rec

    fr = G.compute_axis_frame(coords, cores["source_idx"], cores["sink_idx"])
    rec["degenerate_axis"] = fr["degenerate_axis"]
    if fr["degenerate_axis"]:
        rec["eligibility_tier"] = "descriptive_only"
        return rec

    full_count = bools.sum(axis=1).astype(float)
    z = G.channel_stereotypy_excess(masked, bools, rng=np.random.default_rng(0))
    samp = G.classify_sampling_geometry(
        names, eligible, fr["off_axis"],
        spacing_mm=3.5 if ds == "yuquan" else 4.6)
    rec.update({
        "source_radius": G.core_radii(coords[cores["source_idx"]],
                                      np.array(fr["source_centroid"])),
        "sink_radius": G.core_radii(coords[cores["sink_idx"]],
                                    np.array(fr["sink_centroid"])),
        "axis_length_mm": fr["axis_length"],
        "perp_spread": G.perp_spread(fr["off_axis"], eligible),
        "perp_spread_participation_sweep":
            G.perp_spread_participation_sweep(fr["off_axis"], full_count),
        "sampling_geometry": samp,
        "perp_width_measurable": samp["measurable"],
        "along_axis_profile": G.axis_stereotypy_profile(
            fr["along_axis"], z,
            edges=list(np.linspace(0.0, max(fr["axis_length"], 1e-6), 5))),
        "soz_relation": {
            "source_core": [names[i] for i in cores["source_idx"]],
            "sink_core": [names[i] for i in cores["sink_idx"]],
            "soz_channels_in_montage":
                sorted(_soz_set(ds, subj) & set(names)),
        },
    })
    return rec


def discover_cohort():
    subs = []
    for f in sorted(RANKDISP.glob("*.json")):
        ds, subj = f.stem.split("_", 1)
        subs.append((ds, subj))
    return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="ds:subj tokens; default = full cohort")
    ap.add_argument("--out", default=str(
        _ROOT / "results/topic4_sef_hfo/skeleton_geometry"))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "per_subject").mkdir(parents=True, exist_ok=True)

    if args.subjects:
        cohort = [tuple(s.split(":", 1)) for s in args.subjects]
    else:
        cohort = discover_cohort()

    recs = []
    for ds, subj in cohort:
        try:
            rec = process_subject(ds, subj)
        except Exception as e:  # noqa: BLE001 — record, don't crash the cohort
            rec = {"dataset": ds, "subject": subj, "status": f"error: {e}"}
        (out / "per_subject" / f"{ds}_{subj}.json").write_text(
            json.dumps(rec, indent=2, default=float))
        recs.append(rec)

    ok = [r for r in recs if r.get("status") == "ok"]
    tiers = {}
    for r in ok:
        tiers[r["eligibility_tier"]] = tiers.get(r["eligibility_tier"], 0) + 1
    by_dataset = {}
    for r in ok:
        d = by_dataset.setdefault(r["dataset"], {"n": 0, "axis_length_mm": []})
        d["n"] += 1
        if r.get("eligibility_tier") in ("primary", "fallback") and "axis_length_mm" in r:
            d["axis_length_mm"].append(r["axis_length_mm"])
    cohort_summary = {
        "n_processed": len(recs), "n_ok": len(ok),
        "phantom_core_violations":
            int(sum(bool(r.get("phantom_core_violation")) for r in ok)),
        "tiers": tiers,
        "sampling_geometry": {
            g: int(sum(r.get("sampling_geometry", {}).get("geometry") == g for r in ok))
            for g in ("1D", "distributed")},
        "swap_tiers": {
            sc: int(sum(r.get("swap_class") == sc for r in ok))
            for sc in ("none", "candidate", "strict")},
        "by_dataset": by_dataset,
    }
    (out / "cohort_summary.json").write_text(json.dumps(cohort_summary, indent=2, default=float))
    print(json.dumps(cohort_summary, indent=2, default=float))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test (or manual run if SKIP) to verify it passes**

Run: `pytest tests/test_propagation_skeleton_geometry.py -k runner_smoke -v`
If SKIPPED, run manually:
`python scripts/run_propagation_skeleton_geometry.py --subjects epilepsiae:253 yuquan:chengshuai --out /tmp/skel_test`
Expected: exit 0; `/tmp/skel_test/cohort_summary.json` shows `phantom_core_violations: 0` and a `by_dataset` block.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(skeleton-geom): runner — load/swap-route/tier-gate, per-subject + cohort JSON"
```

---

## Task 10: Full cohort run + acceptance verification

Re-read spec §9 (acceptance gates) before this task.

**Files:**
- Run only (writes `results/topic4_sef_hfo/skeleton_geometry/`).

- [ ] **Step 1: Run the full cohort**

Run: `python scripts/run_propagation_skeleton_geometry.py`
Expected: prints cohort_summary; `results/topic4_sef_hfo/skeleton_geometry/per_subject/` has ~31 files.

- [ ] **Step 2: Verify the spec §9 acceptance gates with a check script**

```bash
python - <<'PY'
import json, glob
from pathlib import Path
base = Path("results/topic4_sef_hfo/skeleton_geometry")
cohort = json.loads((base / "cohort_summary.json").read_text())
print("tiers:", cohort["tiers"])
print("phantom_core_violations:", cohort["phantom_core_violations"])
print("sampling_geometry:", cohort["sampling_geometry"])
print("swap_tiers:", cohort["swap_tiers"])
assert cohort["phantom_core_violations"] == 0, "ACCEPTANCE §9.1 FAIL: phantom in a core"
recs = [json.loads(Path(f).read_text()) for f in glob.glob(str(base/"per_subject/*.json"))]
prim = [r for r in recs if r.get("eligibility_tier") == "primary"]
# §9.3 width discipline present on every primary subject
for r in prim:
    assert "perp_spread_participation_sweep" in r, f"{r['subject']}: no participation sweep"
    assert "perp_width_measurable" in r
    if r["sampling_geometry"]["geometry"] == "1D":
        assert r["perp_width_measurable"] is False
# §9.4 cross-dataset stratification kept (no pooled mm)
assert set(cohort["by_dataset"]) <= {"yuquan", "epilepsiae"}
print("ACCEPTANCE GATES PASS: phantom-safe cores, width discipline, dataset stratification")
PY
```
Expected: prints `ACCEPTANCE GATES PASS`.

- [ ] **Step 3: Commit results**

```bash
git add results/topic4_sef_hfo/skeleton_geometry
git commit -m "feat(skeleton-geom): full-cohort skeleton geometry results + acceptance gates pass"
```

---

## Task 11: Figures + figures/README.md

Re-read spec §8 (figures) and CLAUDE.md §7 (one question per panel) + AGENTS.md (figures/README.md required, Chinese).

**Files:**
- Create: `scripts/plot_propagation_skeleton_geometry.py`
- Create: `results/topic4_sef_hfo/skeleton_geometry/figures/README.md`

- [ ] **Step 1: Write the plotter**

```python
#!/usr/bin/env python3
"""Figures for propagation skeleton geometry. Spec §8."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
BASE = _ROOT / "results/topic4_sef_hfo/skeleton_geometry"
FIG = BASE / "figures"; FIG.mkdir(parents=True, exist_ok=True)
recs = [json.loads(p.read_text()) for p in sorted((BASE/"per_subject").glob("*.json"))]
ok = [r for r in recs if r.get("eligibility_tier") in ("primary", "fallback")
      and "axis_length_mm" in r]

# Fig 1: cohort geometric scalars, stratified by dataset (Epilepsiae = MNI-mm)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, key, lab in zip(
        axes,
        ["axis_length_mm", None, None, None],
        ["source-sink axis length (mm)", "source radius RMS (mm)",
         "sink radius RMS (mm)", "perpendicular spread RMS (mm)"]):
    for i, (ds, color) in enumerate([("yuquan", "tab:blue"), ("epilepsiae", "tab:orange")]):
        if lab.startswith("source-sink"):
            vals = [r["axis_length_mm"] for r in ok if r["dataset"] == ds]
        elif lab.startswith("source radius"):
            vals = [r["source_radius"]["rms_mm"] for r in ok if r["dataset"] == ds]
        elif lab.startswith("sink radius"):
            vals = [r["sink_radius"]["rms_mm"] for r in ok if r["dataset"] == ds]
        else:
            vals = [r["perp_spread"]["rms_mm"] for r in ok
                    if r["dataset"] == ds and r.get("perp_width_measurable")]
        ax.scatter(np.full(len(vals), i) + np.random.uniform(-.08, .08, len(vals)),
                   vals, c=color, alpha=.7,
                   label=f"{ds}{' (MNI mm)' if ds=='epilepsiae' else ' (native mm)'}")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Yuquan\nSEEG", "Epilepsiae\nSEEG+ECoG"])
    ax.set_ylabel(lab); ax.legend(fontsize=7)
fig.suptitle("Interictal propagation skeleton — geometric scalars (per dataset; "
             "Epilepsiae in MNI-space mm, not pooled with Yuquan native mm)")
fig.tight_layout(); fig.savefig(FIG / "skeleton_scalars_by_dataset.png", dpi=140); plt.close(fig)

# Fig 2: pooled along-axis stereotypy-excess profile (within-subject z, poolable)
fig, ax = plt.subplots(figsize=(7, 5))
xs, ys = [], []
for r in ok:
    for b in r.get("along_axis_profile", []):
        if b["n"] > 0 and np.isfinite(b["mean_excess"]):
            xs.append((b["a_lo"] + b["a_hi"]) / 2.0); ys.append(b["mean_excess"])
ax.axhline(0, ls=":", c="grey", label="chance (matched null)")
ax.scatter(xs, ys, alpha=.4, c="tab:purple")
ax.set_xlabel("along-axis distance from source (mm)")
ax.set_ylabel("stereotypy excess (matched-null z)")
ax.set_title("Temporal stereotypy along the propagation axis (pooled, descriptive)")
ax.legend(); fig.tight_layout()
fig.savefig(FIG / "along_axis_stereotypy_profile.png", dpi=140); plt.close(fig)

# Fig 3: per-subject axis-coordinate scatter for a few example subjects
ex = [r for r in ok if r.get("perp_width_measurable")][:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, r in zip(axes.ravel(), ex):
    ax.set_title(f"{r['dataset']}:{r['subject']}  L={r['axis_length_mm']:.0f}mm")
    ax.set_xlabel("along-axis (mm)"); ax.set_ylabel("off-axis (mm)")
    ax.axhline(0, c="k", lw=.5)
for ax in axes.ravel()[len(ex):]:
    ax.axis("off")
fig.suptitle("Per-subject axis-coordinate frame (along vs off-axis)")
fig.tight_layout(); fig.savefig(FIG / "axis_frame_examples.png", dpi=140); plt.close(fig)
print("wrote 3 figures to", FIG)
```

- [ ] **Step 2: Run the plotter**

Run: `python scripts/plot_propagation_skeleton_geometry.py`
Expected: "wrote 3 figures"; 3 PNGs in `results/topic4_sef_hfo/skeleton_geometry/figures/`.

- [ ] **Step 3: Eyeball the figures**

Open each PNG. Confirm: dataset-stratified scalars (no pooled mm), along-axis profile with chance line, axis-frame examples render. Per memory `feedback_figure_self_contained_paper_grade`: tight axes, self-contained labels, no internal codenames. Fix and re-render before writing README.

- [ ] **Step 4: Write figures/README.md (Chinese, per AGENTS.md)**

```markdown
# 间期传播几何骨架 — 图说明

### skeleton_scalars_by_dataset.png
四联：源-汇轴长、源半径RMS、汇半径RMS、参与通道垂向铺展RMS，**按数据集分层**（Yuquan native mm / Epilepsiae MNI-space mm，**不跨集 pool**）。每点一个被试。垂向铺展只画"可测"（非单杆 1D）被试。

**关注点**：Epilepsiae 标的是 MNI-mm（缩放到模板），与 Yuquan 真解剖 mm **不可直接比绝对数**；看的是各自数据集内的尺度分布与量级。

### along_axis_stereotypy_profile.png
沿传播轴的时序刻板性：横轴=离源沿轴距离(mm)，纵轴=刻板性 excess（参与配对 null 的 z，0=随机）。pooled（within-subject z 后可 pool），描述性。

**关注点**：沿轴方向刻板性部分是模板**定义**出来的（rank≈沿轴位置），所以这是延展/边界 sanity，不是独立检验；离轴衰减（真正的耦合宽度）留到下一轮。

### axis_frame_examples.png
几个被试的轴坐标系散点（横=沿轴、纵=离轴）。展示传播骨架在每个被试里长什么样。

**关注点**：点沿横轴铺开=沿传播方向；纵轴铺开=离轴（斑块宽度）。单杆被试纵轴会压成一条线（1D 采样，垂宽不可测）。
```

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_propagation_skeleton_geometry.py results/topic4_sef_hfo/skeleton_geometry/figures
git commit -m "feat(skeleton-geom): cohort figures (dataset-stratified) + figures/README"
```

---

## Task 12: Docs — archive results + index pointer

Re-read memory `feedback_update_docs_after_each_exploration` (update archive doc after each key exploration).

**Files:**
- Create: `docs/archive/topic4/sef_hfo/propagation_skeleton_geometry_2026-06-08.md`
- Modify: `docs/archive/topic4/INDEX.md`

- [ ] **Step 1: Write the archive doc**

Write a 1-2 page archive doc with the three-part plain-language frame (CLAUDE.md §8 / hfosp-plain-language-recap): 测了什么 (propagation skeleton geometry: source/sink compactness, axis length, perpendicular spread, along-axis stereotypy), 怎么测的 (axis-coordinate frame, phantom-safe cores, matched null, tier gate), 揭示了什么 (the per-dataset scalar ranges + 1D-vs-distributed split, framed as descriptive model-input — NOT localization, NOT yet stability). Include: cohort tier counts from cohort_summary.json, the four mm scalars stratified by dataset, sampling-geometry split, and the honest caveats (width = participation lower bound; Epilepsiae MNI-mm; exploratory, no held-out; stability-vs-rate is next round). Link to spec + plan + `[[project_topic4_soz_localization_plan]]`.

- [ ] **Step 2: Add the INDEX pointer**

Add to `docs/archive/topic4/INDEX.md` under the existing "数据侧（paper-A）" section a bullet pointing to the new archive doc with a one-line honest summary (descriptive geometric skeleton for model input; exploratory).

- [ ] **Step 3: Commit**

```bash
git add docs/archive/topic4/sef_hfo/propagation_skeleton_geometry_2026-06-08.md docs/archive/topic4/INDEX.md
git commit -m "docs(skeleton-geom): archive results + INDEX pointer (descriptive, exploratory)"
```

---

## Out of scope (this plan)

- Geometry-vs-rate split-half **stability** comparison → next round (spec §10).
- Off-axis temporal **coherence width** as a gated cohort scalar → next round (this round: descriptive profile only).
- Cross-subject absolute-coordinate pooling; promoting any number to topic4 main-doc paper tier (exploratory, archive-only).
