# SEF-HFO 虚拟 SEEG 观测层 — Increment 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the model-free virtual-SEEG observation layer and prove, against a known-direction toy wave, that a synthetic source's propagation direction reads back correctly through virtual contacts into a real-pipeline-compatible artifact — the "已知方向玩具波合同门".

**Architecture:** A new module `src/sef_hfo_observation.py` of small pure functions (montage → envelope sampler → lag extractor → direction estimators → legacy-key artifact writer/validator), plus `src/sef_hfo_toywave.py` (three analytic sources: traveling wave, centered radial, synchronous-amplitude). A runner produces the gate verdict + figures. No model dynamics in Increment 1.

**Tech Stack:** Python, numpy, pytest. Reuses `src.sef_hfo_field._grid` conventions (mm coords, centered) and writes artifacts the real `src.interictal_propagation.load_subject_propagation_events` loader can read.

**Spec:** `docs/superpowers/specs/2026-06-06-sef-hfo-virtual-seeg-observation-layer-design.md`. This plan implements Increment 1 (§5). Increment 2 (SNN slice, §6) is a separate plan.

**Locked constants (spec §3.5/§4.1/§10):** `k_dir=3`; participation gate `n_participating ≥ 2·k_dir+1 = 7`; `timing_frac f=0.5` (per-contact, relative to own in-event peak); `τ_pass=0.9` (Spearman); `τ_fail=0.3`; `ε_deg = 0.5×contact pitch`; tie tolerance `Δt` = 1 sample (= `dt`).

---

### Task 1: VirtualMontage + parametric shaft builder + real-geometry loud-fail stub

**Files:**
- Create: `src/sef_hfo_observation.py`
- Test: `tests/test_sef_hfo_observation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sef_hfo_observation.py
"""TDD for src/sef_hfo_observation — Increment 1 virtual-SEEG observation layer.

All tests are model-free: synthetic analytic sources (src.sef_hfo_toywave) sampled
through virtual contacts, then read back. Locks the spec
docs/superpowers/specs/2026-06-06-sef-hfo-virtual-seeg-observation-layer-design.md §5.
"""
import json

import numpy as np
import pytest

from src.sef_hfo_observation import (
    VirtualMontage,
    build_shaft,
    merge_montages,
    from_real_geometry,
)


def test_build_shaft_geometry_and_names():
    m = build_shaft(angle_rad=0.0, pitch=2.0, n_contacts=5, origin=(0.0, 0.0),
                    name_prefix="A")
    assert m.contacts.shape == (5, 2)
    # 5 contacts, pitch 2mm, centered on origin -> x in [-4,-2,0,2,4], y==0
    np.testing.assert_allclose(m.contacts[:, 0], [-4, -2, 0, 2, 4])
    np.testing.assert_allclose(m.contacts[:, 1], 0.0)
    assert m.names == ["A0", "A1", "A2", "A3", "A4"]
    assert not m.spans_2d()           # single shaft is 1-D


def test_merge_two_nonparallel_shafts_spans_2d():
    a = build_shaft(0.0, 2.0, 4, (0.0, 0.0), "A")
    b = build_shaft(np.pi / 2, 2.0, 4, (1.0, 0.0), "B")
    m = merge_montages([a, b])
    assert m.contacts.shape == (8, 2)
    assert m.names[:4] == ["A0", "A1", "A2", "A3"]
    assert m.names[4:] == ["B0", "B1", "B2", "B3"]
    assert m.spans_2d()               # two non-parallel shafts span the plane


def test_from_real_geometry_loud_fails():
    with pytest.raises(NotImplementedError):
        from_real_geometry(np.zeros((3, 3)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_observation.py -k "shaft or real_geometry" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.sef_hfo_observation'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sef_hfo_observation.py
"""Virtual-SEEG observation layer (Increment 1, spec 2026-06-06).

Small pure functions: montage -> envelope sampler -> lag extractor -> direction
estimators -> legacy-key artifact writer/validator. Model adapters live next to
the models, NOT here. No model dynamics in this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VirtualMontage:
    contacts: np.ndarray   # (n_contact, 2) coords in model frame, mm
    names: list            # length n_contact
    provenance: str

    def spans_2d(self, tol: float = 1e-6) -> bool:
        c = self.contacts - self.contacts.mean(axis=0, keepdims=True)
        return int(np.linalg.matrix_rank(c, tol=tol)) >= 2


def build_shaft(angle_rad, pitch, n_contacts, origin=(0.0, 0.0),
                name_prefix="A") -> VirtualMontage:
    """One linear shaft: n_contacts evenly spaced (pitch mm) along angle_rad,
    centered on origin. Contacts named <prefix>0..<prefix>(n-1)."""
    d = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    offs = (np.arange(n_contacts) - (n_contacts - 1) / 2.0) * pitch
    contacts = np.asarray(origin, float)[None, :] + offs[:, None] * d[None, :]
    names = [f"{name_prefix}{i}" for i in range(n_contacts)]
    return VirtualMontage(contacts, names, provenance="parametric_shaft")


def merge_montages(montages) -> VirtualMontage:
    """Combine shafts into one montage (the ≥2-non-parallel-shaft 2D read-out)."""
    contacts = np.vstack([m.contacts for m in montages])
    names = [nm for m in montages for nm in m.names]
    return VirtualMontage(contacts, names, provenance="parametric_multi_shaft")


def from_real_geometry(*args, **kwargs):
    """Layer-2 stub (spec §7): 3D real SEEG coords -> 2D model frame. Loud-fail
    until the per-patient heterogeneity round builds it."""
    raise NotImplementedError(
        "real-geometry montage (3D->2D registration) is layer 2; see spec §7")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_observation.py -k "shaft or real_geometry" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_observation.py tests/test_sef_hfo_observation.py
git commit -m "feat(topic4 obs): VirtualMontage + parametric shaft + real-geometry loud-fail stub"
```

---

### Task 2: Grid coords + envelope sampler (distance footprint over arbitrary contacts)

**Files:**
- Modify: `src/sef_hfo_observation.py`
- Test: `tests/test_sef_hfo_observation.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sef_hfo_observation.py
from src.sef_hfo_observation import grid_coords, sample_envelopes


def test_grid_coords_match_field_grid():
    # mirrors src.sef_hfo_field._grid (indexing="ij", centered, spacing L/n)
    coords = grid_coords(n=4, L=8.0)
    assert coords.shape == (16, 2)
    xs = np.unique(coords[:, 0])
    np.testing.assert_allclose(xs, [-4, -2, 0, 2])   # (arange(4)-2)*2


def test_sample_envelope_tracks_a_localized_blob():
    # A static blob centered at (+10,0); a contact at (+10,0) must read a larger
    # envelope than a contact at (-10,0).
    n, L = 64, 64.0
    coords = grid_coords(n, L)
    X = coords[:, 0]
    Y = coords[:, 1]
    blob = np.exp(-(((X - 10) ** 2 + Y ** 2) / (2 * 4.0 ** 2)))
    frames = blob[None, :].repeat(3, axis=0)          # (3, n*n) constant in time
    m = merge_montages([build_shaft(0.0, 4.0, 1, (10.0, 0.0), "near"),
                        build_shaft(0.0, 4.0, 1, (-10.0, 0.0), "far")])
    env = sample_envelopes(frames, coords, m, kernel_width=3.0)
    assert env.shape == (2, 3)
    assert env[0, 0] > env[1, 0]                      # near-contact reads more
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_observation.py -k "grid_coords or sample_envelope" -v`
Expected: FAIL with `ImportError: cannot import name 'grid_coords'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/sef_hfo_observation.py
def grid_coords(n: int, L: float) -> np.ndarray:
    """Flattened (n*n, 2) mm coords matching src.sef_hfo_field._grid (ij-indexed,
    centered, spacing L/n). Row-major .ravel() order aligns with a field reshaped
    via field.reshape(n_time, -1)."""
    x = (np.arange(n) - n // 2) * (L / n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()])


def sample_envelopes(source_frames, grid_xy, montage, kernel_width,
                     Rr=None) -> np.ndarray:
    """Per-contact activity envelope = distance-weighted average of source over
    nearby grid pixels (generalizes engine/lfp.py to arbitrary contact coords).

    source_frames : (n_time, n_pix) — a field time series flattened per frame.
    grid_xy       : (n_pix, 2) mm coords (from grid_coords).
    Gaussian footprint exp(-d^2 / 2 kernel_width^2), normalized per contact.
    Optional Rr (mm) hard cutoff; None = all pixels.
    Returns (n_contact, n_time).
    """
    frames = np.asarray(source_frames, float)
    out = np.empty((len(montage.contacts), frames.shape[0]))
    for ci, c in enumerate(montage.contacts):
        d = np.linalg.norm(grid_xy - c[None, :], axis=1)
        mask = np.ones(d.shape, bool) if Rr is None else (d <= Rr)
        dd = d[mask]
        w = np.exp(-(dd ** 2) / (2.0 * kernel_width ** 2))
        w = w / max(w.sum(), 1e-12)
        out[ci] = frames[:, mask] @ w
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_observation.py -k "grid_coords or sample_envelope" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_observation.py tests/test_sef_hfo_observation.py
git commit -m "feat(topic4 obs): grid_coords + distance-footprint envelope sampler"
```

---

### Task 3: Toy-wave analytic sources (traveling / radial / synchronous-amplitude)

**Files:**
- Create: `src/sef_hfo_toywave.py`
- Test: `tests/test_sef_hfo_toywave.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sef_hfo_toywave.py
"""TDD for src/sef_hfo_toywave — Increment 1 analytic sources (no model)."""
import numpy as np

from src.sef_hfo_toywave import (
    traveling_wave,
    radial_source,
    synchronous_amplitude_source,
)


def test_traveling_wave_peak_moves_along_n_hat():
    src = traveling_wave(n=64, L=64.0, angle_rad=0.0, c=0.5, dt=0.25,
                         t_max=200.0, width=8.0)
    frames, coords, window, n_hat = (src["frames"], src["grid_xy"],
                                     src["window"], src["n_hat"])
    assert frames.shape[1] == coords.shape[0]
    np.testing.assert_allclose(n_hat, [1.0, 0.0], atol=1e-9)
    # peak-position projection on n_hat increases over time (wave moves +x)
    proj = coords @ n_hat
    early = proj[frames[10].argmax()]
    late = proj[frames[-10].argmax()]
    assert late > early
    assert window[1] > window[0]


def test_radial_source_is_centered_and_isotropic():
    src = radial_source(n=64, L=64.0, c=0.4, dt=0.25, t_max=150.0, width=6.0)
    frames, coords = src["frames"], src["grid_xy"]
    # centroid of activity stays ~at origin (no preferred axis)
    f = frames[frames.shape[0] // 2]
    centroid = (coords * f[:, None]).sum(0) / f.sum()
    np.testing.assert_allclose(centroid, [0.0, 0.0], atol=1.0)


def test_synchronous_amplitude_source_has_no_arrival_gradient():
    src = synchronous_amplitude_source(n=64, L=64.0, dt=0.25, t_max=120.0,
                                       width=10.0, ramp_axis_rad=0.0)
    frames = src["frames"]
    # every pixel peaks at the SAME time frame (h(t) shared) -> no arrival order
    peak_frames = frames.argmax(axis=0)
    active = frames.max(axis=0) > 0.1 * frames.max()
    assert np.ptp(peak_frames[active]) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_toywave.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.sef_hfo_toywave'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sef_hfo_toywave.py
"""Analytic synthetic sources for the Increment-1 known-direction contract gate.

Each returns dict(frames=(n_time, n_pix), grid_xy=(n_pix,2), window=(t_on,t_off) ms,
n_hat=(2,) or None, dt, pitch_hint). No model dynamics; coords match
src.sef_hfo_observation.grid_coords(n, L).
"""
from __future__ import annotations

import numpy as np

from src.sef_hfo_observation import grid_coords


def _times(dt, t_max):
    return np.arange(0.0, t_max, dt)


def traveling_wave(n, L, angle_rad, c, dt, t_max, width, t0=None):
    """Smooth bell a(x,t)=exp(-(s - c(t-t0))^2 / 2 width^2), s = x·n_hat."""
    coords = grid_coords(n, L)
    n_hat = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    s = coords @ n_hat
    t = _times(dt, t_max)
    if t0 is None:
        t0 = 0.0
    s0 = s.min() - 2 * width                     # start just outside the sheet
    front = s0 + c * (t - t0)
    frames = np.exp(-((s[None, :] - front[:, None]) ** 2) / (2.0 * width ** 2))
    return dict(frames=frames, grid_xy=coords, window=(0.0, t_max), n_hat=n_hat,
                dt=dt, pitch_hint=L / n)


def radial_source(n, L, c, dt, t_max, width):
    """Centered expanding ring a(x,t)=exp(-(r - c t)^2 / 2 width^2), r=||x||.
    Real arrival gradient (outward) but NO preferred axis -> direction must be no-axis."""
    coords = grid_coords(n, L)
    r = np.linalg.norm(coords, axis=1)
    t = _times(dt, t_max)
    frames = np.exp(-((r[None, :] - (c * t)[:, None]) ** 2) / (2.0 * width ** 2))
    return dict(frames=frames, grid_xy=coords, window=(0.0, t_max), n_hat=None,
                dt=dt, pitch_hint=L / n)


def synchronous_amplitude_source(n, L, dt, t_max, width, ramp_axis_rad=0.0):
    """a(x,t)=b(x)·h(t): all pixels rise/fall together (shared bell h(t)); only
    spatial amplitude b(x) (linear ramp along ramp_axis) differs. NO arrival
    gradient -> tests whether electrode layout + threshold fabricate fake order."""
    coords = grid_coords(n, L)
    axis = np.array([np.cos(ramp_axis_rad), np.sin(ramp_axis_rad)])
    proj = coords @ axis
    b = 0.5 + 0.5 * (proj - proj.min()) / max(np.ptp(proj), 1e-12)  # in [0.5,1]
    t = _times(dt, t_max)
    h = np.exp(-((t - t_max / 2.0) ** 2) / (2.0 * width ** 2))
    frames = h[:, None] * b[None, :]
    return dict(frames=frames, grid_xy=coords, window=(0.0, t_max), n_hat=None,
                dt=dt, pitch_hint=L / n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_toywave.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_toywave.py tests/test_sef_hfo_toywave.py
git commit -m "feat(topic4 obs): toy-wave analytic sources (traveling/radial/synchronous-amplitude)"
```

---

### Task 4: Lag extractor — participation gate + per-contact first-crossing timing + masked ranks

**Files:**
- Modify: `src/sef_hfo_observation.py`
- Test: `tests/test_sef_hfo_observation.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sef_hfo_observation.py
from src.sef_hfo_observation import LagPatArtifact, extract_lagpat


def test_extract_lagpat_orders_by_first_crossing_and_masks_nonparticipants():
    # 3 contacts: A peaks earliest, B later, C never participates (tiny amplitude).
    dt = 0.25
    nt = 400
    t = np.arange(nt) * dt
    envA = np.exp(-((t - 20) ** 2) / (2 * 5.0 ** 2))
    envB = np.exp(-((t - 40) ** 2) / (2 * 5.0 ** 2))
    envC = 0.001 * np.ones(nt)
    env = np.vstack([envA, envB, envC])
    art = extract_lagpat(env, dt, event_windows=[(0.0, nt * dt)],
                         participation_floor=0.0, participation_margin=0.05,
                         timing_frac=0.5, tie_tol=dt)
    assert art.bools[:, 0].tolist() == [True, True, False]
    # A crosses 0.5*peak before B -> rank(A) < rank(B)
    assert art.ranks[0, 0] < art.ranks[1, 0]
    # non-participant C: NaN rank and NaN lag (no phantom finite rank)
    assert np.isnan(art.ranks[2, 0])
    assert np.isnan(art.lag_raw[2, 0])


def test_extract_lagpat_ties_within_tol_get_equal_rank():
    dt = 0.25
    nt = 400
    t = np.arange(nt) * dt
    # A and B identical timing (synchronous) -> tied ranks
    envAB = np.exp(-((t - 30) ** 2) / (2 * 5.0 ** 2))
    env = np.vstack([envAB, envAB])
    art = extract_lagpat(env, dt, event_windows=[(0.0, nt * dt)],
                         participation_floor=0.0, participation_margin=0.05,
                         timing_frac=0.5, tie_tol=dt)
    assert art.ranks[0, 0] == art.ranks[1, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_observation.py -k "extract_lagpat" -v`
Expected: FAIL with `ImportError: cannot import name 'extract_lagpat'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/sef_hfo_observation.py
@dataclass
class LagPatArtifact:
    bools: np.ndarray          # (n_contact, n_event) bool
    ranks: np.ndarray          # (n_contact, n_event) float; non-participant = NaN
    lag_raw: np.ndarray        # (n_contact, n_event) float ms; non-participant = NaN
    contact_coords: np.ndarray # (n_contact, 2)
    names: list
    event_rel_times: np.ndarray      # (n_event,) ms, event onset
    event_rel_end_times: np.ndarray  # (n_event,) ms, event end


def _dense_ranks_with_tie_tol(lags, tie_tol):
    """Dense ranks (0,1,2,...) over a 1-D lag array; lags within tie_tol of a
    group's first value share a rank."""
    order = np.argsort(lags, kind="mergesort")
    sorted_lags = lags[order]
    grp = np.zeros(len(lags), dtype=float)
    start = sorted_lags[0]
    g = 0.0
    for i in range(1, len(lags)):
        if sorted_lags[i] - start > tie_tol:
            g += 1.0
            start = sorted_lags[i]
        grp[i] = g
    ranks = np.empty(len(lags))
    ranks[order] = grp
    return ranks


def extract_lagpat(envelopes, dt, event_windows, participation_floor,
                   participation_margin, timing_frac=0.5, tie_tol=None) -> LagPatArtifact:
    """Per event window: participation (max env > floor+margin) sets bools; among
    participants, activation time = first crossing of timing_frac * own in-window
    peak (spec §4.1 timing 阈, per-contact relative); ranks = tie-tolerant dense
    rank of those times. Non-participants -> NaN rank/lag (no phantom finite rank).

    contact_coords/names are filled by the caller via attach_geometry()."""
    env = np.asarray(envelopes, float)
    n_c = env.shape[0]
    n_ev = len(event_windows)
    if tie_tol is None:
        tie_tol = dt
    bools = np.zeros((n_c, n_ev), bool)
    ranks = np.full((n_c, n_ev), np.nan)
    lag_raw = np.full((n_c, n_ev), np.nan)
    bar = participation_floor + participation_margin
    ev_on = np.array([w[0] for w in event_windows], float)
    ev_off = np.array([w[1] for w in event_windows], float)
    for ev, (t_on, t_off) in enumerate(event_windows):
        s = int(round(t_on / dt))
        e = int(round(t_off / dt))
        seg = env[:, s:e]                      # (n_c, win)
        if seg.shape[1] == 0:
            continue
        peak = seg.max(axis=1)
        part = peak > bar
        bools[part, ev] = True
        for ci in np.flatnonzero(part):
            thr = timing_frac * peak[ci]
            crossings = np.flatnonzero(seg[ci] >= thr)
            lag_raw[ci, ev] = t_on + crossings[0] * dt   # first-crossing time (ms)
        idx = np.flatnonzero(part)
        if idx.size:
            ranks[idx, ev] = _dense_ranks_with_tie_tol(lag_raw[idx, ev], tie_tol)
    return LagPatArtifact(bools=bools, ranks=ranks, lag_raw=lag_raw,
                          contact_coords=np.zeros((n_c, 2)), names=[""] * n_c,
                          event_rel_times=ev_on, event_rel_end_times=ev_off)


def attach_geometry(artifact: LagPatArtifact, montage: VirtualMontage) -> LagPatArtifact:
    """Fill coords/names from the montage (order = contact order). Asserts length."""
    assert len(montage.names) == artifact.bools.shape[0]
    artifact.contact_coords = np.asarray(montage.contacts, float)
    artifact.names = list(montage.names)
    return artifact
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_observation.py -k "extract_lagpat" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_observation.py tests/test_sef_hfo_observation.py
git commit -m "feat(topic4 obs): lag extractor (participation gate + per-contact first-crossing + masked ranks)"
```

---

### Task 5: Direction estimators — rank-vs-projection Spearman + endpoint-centroid axis

**Files:**
- Modify: `src/sef_hfo_observation.py`
- Test: `tests/test_sef_hfo_observation.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sef_hfo_observation.py
from src.sef_hfo_observation import (
    rank_vs_projection_spearman,
    endpoint_centroid_axis,
    axis_angle_error_deg,
)


def _ranks_along(coords, n_hat):
    # helper: monotone ranks increasing along n_hat (a perfect read-out)
    proj = coords @ n_hat
    order = np.argsort(proj)
    r = np.empty(len(proj))
    r[order] = np.arange(len(proj), dtype=float)
    return r


def test_spearman_is_one_when_ranks_follow_projection():
    coords = np.column_stack([np.linspace(-5, 5, 9), np.zeros(9)])
    n_hat = np.array([1.0, 0.0])
    ranks = _ranks_along(coords, n_hat)
    bools = np.ones(9, bool)
    rho = rank_vs_projection_spearman(ranks, bools, coords, n_hat)
    assert rho > 0.99


def test_endpoint_axis_tracks_imposed_direction_within_tolerance():
    # contacts on a 2D grid; ranks increase along 30deg
    g = np.linspace(-6, 6, 5)
    XX, YY = np.meshgrid(g, g)
    coords = np.column_stack([XX.ravel(), YY.ravel()])     # 25 contacts, 2D
    theta = np.deg2rad(30.0)
    n_hat = np.array([np.cos(theta), np.sin(theta)])
    ranks = _ranks_along(coords, n_hat)
    bools = np.ones(len(coords), bool)
    axis = endpoint_centroid_axis(ranks, bools, coords, k_dir=3, eps_deg=1.0)
    assert axis is not None
    assert axis_angle_error_deg(axis, theta) < 25.0


def test_endpoint_axis_degenerate_returns_none():
    # all participants tied (rank 0) -> early/late centroids coincide -> no-axis
    coords = np.random.default_rng(0).normal(size=(9, 2))
    ranks = np.zeros(9)
    bools = np.ones(9, bool)
    assert endpoint_centroid_axis(ranks, bools, coords, k_dir=3, eps_deg=1.0) is None


def test_endpoint_axis_insufficient_participants_returns_none():
    coords = np.random.default_rng(1).normal(size=(6, 2))   # < 2*k_dir+1 = 7
    ranks = np.arange(6, dtype=float)
    bools = np.ones(6, bool)
    assert endpoint_centroid_axis(ranks, bools, coords, k_dir=3, eps_deg=0.1) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_observation.py -k "spearman or endpoint_axis" -v`
Expected: FAIL with `ImportError: cannot import name 'rank_vs_projection_spearman'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/sef_hfo_observation.py
def _rankdata_avg(a):
    """Average ranks (ties share mean rank), pure numpy."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def rank_vs_projection_spearman(ranks_ev, bools_ev, coords, n_hat) -> float:
    """Increment-1 main gate: Spearman between per-event participant ranks and
    their projection onto the known direction n_hat. NaN if <2 participants or
    no variance (e.g. all tied)."""
    idx = np.flatnonzero(np.asarray(bools_ev, bool))
    if idx.size < 2:
        return float("nan")
    r = np.asarray(ranks_ev, float)[idx]
    proj = (np.asarray(coords, float)[idx] @ np.asarray(n_hat, float))
    if np.ptp(r) == 0 or np.ptp(proj) == 0:
        return float("nan")            # no order to compare (e.g. synchronous source)
    rr = _rankdata_avg(r)
    rp = _rankdata_avg(proj)
    return float(np.corrcoef(rr, rp)[0, 1])


def endpoint_centroid_axis(ranks_ev, bools_ev, coords, k_dir=3, eps_deg=None):
    """Increment-2 main estimator (also Increment-1 C1 no-axis check). Axis =
    centroid(k_dir earliest-rank contacts) -> centroid(k_dir latest). Returns a
    unit 2-vector, or None when: <2*k_dir+1 participants, endpoint sets overlap,
    or ||axis|| < eps_deg (degenerate / no-axis)."""
    idx = np.flatnonzero(np.asarray(bools_ev, bool))
    if idx.size < 2 * k_dir + 1:
        return None
    r = np.asarray(ranks_ev, float)[idx]
    xy = np.asarray(coords, float)[idx]
    order = np.argsort(r, kind="mergesort")
    early = order[:k_dir]
    late = order[-k_dir:]
    if set(early.tolist()) & set(late.tolist()):
        return None
    vec = xy[late].mean(0) - xy[early].mean(0)
    norm = float(np.linalg.norm(vec))
    if eps_deg is not None and norm < eps_deg:
        return None
    if norm < 1e-9:
        return None
    return vec / norm


def axis_angle_error_deg(axis, theta_ref) -> float:
    """Undirected (mod 180°) angle between a 2-vector axis and a reference angle."""
    a = np.arctan2(axis[1], axis[0])
    diff = np.rad2deg(a - theta_ref) % 180.0
    return float(min(diff, 180.0 - diff))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_observation.py -k "spearman or endpoint_axis" -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_observation.py tests/test_sef_hfo_observation.py
git commit -m "feat(topic4 obs): direction estimators (rank-vs-projection Spearman + endpoint-centroid axis)"
```

---

### Task 6: Artifact writers (legacy NPZ + montage manifest + packedTimes) + validator

**Files:**
- Modify: `src/sef_hfo_observation.py`
- Test: `tests/test_sef_hfo_observation.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sef_hfo_observation.py
from pathlib import Path
from src.sef_hfo_observation import (
    write_legacy_npz,
    write_montage_manifest,
    write_packed_times,
    validate_artifact,
)


def _toy_artifact():
    env = np.array([[0.0, 1.0, 0.2, 0.0],
                    [0.0, 0.2, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0]])
    art = extract_lagpat(env, dt=1.0, event_windows=[(0.0, 4.0)],
                         participation_floor=0.0, participation_margin=0.05,
                         timing_frac=0.5, tie_tol=1.0)
    m = merge_montages([build_shaft(0.0, 2.0, 2, (0.0, 0.0), "A"),
                        build_shaft(np.pi / 2, 2.0, 1, (0.0, 3.0), "B")])
    return attach_geometry(art, m)


def test_validate_artifact_flags_phantom_rank():
    art = _toy_artifact()
    validate_artifact(art)                       # clean: non-participant rank is NaN
    art.ranks[2, 0] = 7.0                         # inject a phantom finite rank
    with pytest.raises(AssertionError):
        validate_artifact(art)


def test_legacy_npz_loads_via_real_loader(tmp_path):
    # Round-trip: write legacy-key files, then the REAL loader must read them.
    from src.interictal_propagation import load_subject_propagation_events
    art = _toy_artifact()
    rec = "synthrec"
    npz = tmp_path / f"{rec}_lagPat_withFreqCent.npz"
    write_legacy_npz(art, npz)
    write_packed_times(art, tmp_path / f"{rec}_packedTimes_withFreqCent.npy")
    write_montage_manifest(art, tmp_path / f"{rec}_montage.json")
    loaded = load_subject_propagation_events(str(tmp_path))
    assert list(loaded["channel_names"]) == art.names
    assert loaded["bools"].shape[0] == len(art.names)
    np.testing.assert_array_equal(loaded["bools"].sum(axis=0) > 0,
                                  art.bools.sum(axis=0) > 0)
    manifest = json.loads((tmp_path / f"{rec}_montage.json").read_text())
    assert manifest["chn_names"] == art.names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_observation.py -k "validate_artifact or legacy_npz" -v`
Expected: FAIL with `ImportError: cannot import name 'write_legacy_npz'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/sef_hfo_observation.py (add `import json` at top of file)
def validate_artifact(artifact: LagPatArtifact) -> None:
    """Assert the spec invariants before writing: 2D montage, name/coord order
    match, and non-participant entries carry NO finite rank (phantom-mask discipline)."""
    b = artifact.bools
    assert artifact.ranks.shape == b.shape == artifact.lag_raw.shape
    assert artifact.contact_coords.shape == (b.shape[0], 2)
    assert len(artifact.names) == b.shape[0]
    assert VirtualMontage(artifact.contact_coords, artifact.names, "x").spans_2d(), \
        "read-out montage must span 2D (>=2 non-parallel shafts) — spec D6"
    nonpart = ~b
    bad = np.isfinite(artifact.ranks[nonpart])
    assert not bad.any(), "non-participating contacts must have NaN rank (no phantom)"
    bad_lag = np.isfinite(artifact.lag_raw[nonpart])
    assert not bad_lag.any(), "non-participating contacts must have NaN lag"


def write_legacy_npz(artifact: LagPatArtifact, path) -> None:
    """Write the *_lagPat_withFreqCent.npz with the EXACT keys the real loader
    reads (src/interictal_propagation.py L344-353): lagPatRank / eventsBool /
    lagPatRaw / chnNames / start_t."""
    path = str(path)
    assert path.endswith("_lagPat_withFreqCent.npz"), \
        "filename must end with _lagPat_withFreqCent.npz for the loader's glob"
    validate_artifact(artifact)
    np.savez(
        path,
        lagPatRank=artifact.ranks.astype(float),
        eventsBool=artifact.bools.astype(np.int8),
        lagPatRaw=artifact.lag_raw.astype(float),
        chnNames=np.array(artifact.names, dtype=object),
        start_t=np.array(0.0),
    )


def write_packed_times(artifact: LagPatArtifact, path) -> None:
    """Companion *_packedTimes_withFreqCent.npy: (n_event, 2) rel start/end ms.
    The model gives these directly (D1: no packer, but the loader needs them)."""
    path = str(path)
    assert path.endswith("_packedTimes_withFreqCent.npy")
    packed = np.column_stack([artifact.event_rel_times, artifact.event_rel_end_times])
    np.save(path, packed.astype(float))


def write_montage_manifest(artifact: LagPatArtifact, path) -> None:
    """Sidecar JSON carrying contact coords (legacy lagPat has none; D6 needs them).
    Order is asserted == chnNames."""
    payload = {"contact_coords": artifact.contact_coords.tolist(),
               "chn_names": list(artifact.names)}
    from pathlib import Path as _P
    _P(str(path)).write_text(json.dumps(payload))
```

Add `import json` to the module's imports if not already present.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_observation.py -k "validate_artifact or legacy_npz" -v`
Expected: PASS (2 tests) — the round-trip proves no `KeyError` from the real loader.

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_observation.py tests/test_sef_hfo_observation.py
git commit -m "feat(topic4 obs): legacy-key artifact writer + montage manifest + packedTimes + validator (real-loader round-trip)"
```

---

### Task 7: Increment-1 contract gate — toy wave PASS + C1/C2 must-fail (the §5 gate)

**Files:**
- Modify: `src/sef_hfo_observation.py` (add the gate orchestrator)
- Test: `tests/test_sef_hfo_observation.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sef_hfo_observation.py
from src.sef_hfo_toywave import (
    traveling_wave, radial_source, synchronous_amplitude_source,
)
from src.sef_hfo_observation import read_direction_from_source


def _two_shaft():
    # ≥2 non-parallel shafts spanning the sheet (D6); enough contacts for k_dir gate.
    # Distinct origins so no contact coincides (both-through-(0,0) would duplicate a point).
    a = build_shaft(np.deg2rad(10.0), 4.0, 9, (0.0, 0.0), "A")
    b = build_shaft(np.deg2rad(100.0), 4.0, 9, (0.0, 4.0), "B")
    return merge_montages([a, b])


def test_gate_traveling_wave_reads_correct_direction():
    for deg in (30.0, 60.0):
        src = traveling_wave(64, 64.0, np.deg2rad(deg), c=0.4, dt=0.25,
                             t_max=200.0, width=8.0)
        out = read_direction_from_source(src, _two_shaft(), kernel_width=3.0)
        assert out["spearman"] >= 0.9                      # τ_pass
        assert out["axis"] is not None                     # wave has a real axis
        assert axis_angle_error_deg(out["axis"], np.deg2rad(deg)) < 25.0


def test_gate_C1_radial_source_reads_no_axis():
    src = radial_source(64, 64.0, c=0.35, dt=0.25, t_max=160.0, width=6.0)
    out = read_direction_from_source(src, _two_shaft(), kernel_width=3.0)
    # radial: no preferred axis -> endpoint axis degenerate, projection Spearman low
    assert out["axis"] is None or abs(out["spearman_vs_shaft"]) < 0.3


def test_gate_C2_synchronous_amplitude_makes_no_fake_order():
    # aligned shaft + amplitude ramp along it; per-contact-relative timing -> no order
    src = synchronous_amplitude_source(64, 64.0, dt=0.25, t_max=120.0,
                                       width=10.0, ramp_axis_rad=np.deg2rad(10.0))
    out = read_direction_from_source(src, _two_shaft(), kernel_width=3.0)
    assert np.isnan(out["spearman_vs_shaft"]) or abs(out["spearman_vs_shaft"]) < 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sef_hfo_observation.py -k "gate_" -v`
Expected: FAIL with `ImportError: cannot import name 'read_direction_from_source'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/sef_hfo_observation.py
def read_direction_from_source(source, montage, kernel_width,
                               participation_frac=0.5, timing_frac=0.5,
                               k_dir=3) -> dict:
    """End-to-end Increment-1 read-out: sample the analytic source through the
    montage, extract one event's lagPat, and report both direction metrics.

    Returns dict with:
      spearman          : rank-vs-(true n_hat)-projection Spearman (None-source -> nan)
      spearman_vs_shaft : rank-vs-(shaft long-axis)-projection Spearman (for C1/C2)
      axis              : endpoint-centroid unit axis or None (degenerate/no-axis)
    """
    frames = source["frames"]
    coords = source["grid_xy"]
    dt = source["dt"]
    env = sample_envelopes(frames, coords, montage, kernel_width)
    # participation bar = TEMPORAL noise floor + margin (spec §4.1) — NOT a
    # cross-contact peak percentile (which collapses to ~max when every contact is
    # swept by the bell, giving a knife-edge bar). floor = global quiescent baseline
    # (min over time, ~0 for the noiseless toy); margin scales to the global event
    # peak so every bell-crossed contact robustly participates. Anti-fake-order: the
    # TIMING threshold (inside extract_lagpat) is per-contact relative to own peak.
    floor = float(env.min())
    margin = participation_frac * (float(env.max()) - floor)
    art = extract_lagpat(env, dt, event_windows=[source["window"]],
                         participation_floor=floor, participation_margin=margin,
                         timing_frac=timing_frac, tie_tol=dt)
    art = attach_geometry(art, montage)
    ranks0, bools0 = art.ranks[:, 0], art.bools[:, 0]
    pitch = source.get("pitch_hint", 1.0)
    axis = endpoint_centroid_axis(ranks0, bools0, art.contact_coords,
                                  k_dir=k_dir, eps_deg=0.5 * pitch)
    n_hat = source.get("n_hat")
    spearman = (rank_vs_projection_spearman(ranks0, bools0, art.contact_coords, n_hat)
                if n_hat is not None else float("nan"))
    # shaft long-axis = principal axis of the montage contacts (for C1/C2 no-fake test)
    cc = art.contact_coords - art.contact_coords.mean(0)
    shaft_axis = np.linalg.svd(cc, full_matrices=False)[2][0]
    spearman_vs_shaft = rank_vs_projection_spearman(ranks0, bools0,
                                                    art.contact_coords, shaft_axis)
    return dict(spearman=spearman, spearman_vs_shaft=spearman_vs_shaft, axis=axis,
                artifact=art)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sef_hfo_observation.py -k "gate_" -v`
Expected: PASS (3 tests). If `test_gate_traveling_wave` fails on Spearman, the sampler/extractor has a bug — do NOT relax τ_pass; debug the chain (spec: thresholds locked, never tuned to pass). If C1's `spearman_vs_shaft` comes in high, the diagnosis is participation/centering (the radial source must be centered within the montage footprint), NOT a reason to loosen τ_fail.

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_observation.py tests/test_sef_hfo_observation.py
git commit -m "feat(topic4 obs): Increment-1 contract gate — toy wave PASS + C1/C2 must-fail"
```

---

### Task 8: Full module test pass + DRY review

**Files:**
- Test: `tests/test_sef_hfo_observation.py`, `tests/test_sef_hfo_toywave.py`

- [ ] **Step 1: Run the whole observation + toywave suite**

Run: `pytest tests/test_sef_hfo_observation.py tests/test_sef_hfo_toywave.py -v`
Expected: ALL PASS (≈14 tests).

- [ ] **Step 2: Run the existing SEF-HFO suite to confirm no regression**

Run: `pytest tests/test_sef_hfo_events.py tests/test_sef_hfo_field.py tests/test_sef_hfo_lif.py -q`
Expected: PASS (unchanged — this plan adds new modules, touches none).

- [ ] **Step 3: Commit (only if Step 1/2 surfaced a fix)**

```bash
git add -A && git commit -m "test(topic4 obs): full Increment-1 suite green + no SEF-HFO regression"
```

---

### Task 9: Runner + results + figures/README.md

**Files:**
- Create: `scripts/run_sef_hfo_obs_increment1.py`
- Create: `results/topic4_sef_hfo/observation_layer/figures/README.md` (after figures exist)

- [ ] **Step 1: Write the runner**

```python
# scripts/run_sef_hfo_obs_increment1.py
"""Increment-1 known-direction contract gate: run toy wave (30/60/0/90/135 deg) +
C1 (radial) + C2 (synchronous-amplitude) through virtual contacts; emit verdict
JSON + figures. Thresholds are LOCKED (spec §10): τ_pass=0.9, τ_fail=0.3.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.sef_hfo_observation import (
    build_shaft, merge_montages, read_direction_from_source, axis_angle_error_deg,
    write_legacy_npz, write_packed_times, write_montage_manifest,
)
from src.sef_hfo_toywave import (
    traveling_wave, radial_source, synchronous_amplitude_source,
)

TAU_PASS = 0.9
TAU_FAIL = 0.3
OUT = Path("results/topic4_sef_hfo/observation_layer/increment1_toywave")


def _montage():
    a = build_shaft(np.deg2rad(10.0), 4.0, 9, (0.0, 0.0), "A")
    b = build_shaft(np.deg2rad(100.0), 4.0, 9, (0.0, 4.0), "B")
    return merge_montages([a, b])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT.parent / "figures").mkdir(parents=True, exist_ok=True)
    montage = _montage()
    verdict = {"tau_pass": TAU_PASS, "tau_fail": TAU_FAIL, "waves": {}, "controls": {}}

    for deg in (0.0, 30.0, 60.0, 90.0, 135.0):
        src = traveling_wave(64, 64.0, np.deg2rad(deg), c=0.4, dt=0.25,
                             t_max=200.0, width=8.0)
        out = read_direction_from_source(src, montage, kernel_width=3.0)
        err = (axis_angle_error_deg(out["axis"], np.deg2rad(deg))
               if out["axis"] is not None else None)
        verdict["waves"][f"{deg:g}deg"] = {
            "spearman": out["spearman"], "axis_err_deg": err,
            "pass": bool(out["spearman"] >= TAU_PASS)}

    c1 = read_direction_from_source(radial_source(64, 64.0, 0.35, 0.25, 160.0, 6.0),
                                    montage, 3.0)
    verdict["controls"]["C1_radial"] = {
        "axis_is_none": c1["axis"] is None,
        "spearman_vs_shaft": c1["spearman_vs_shaft"],
        "must_fail_ok": bool(c1["axis"] is None or abs(c1["spearman_vs_shaft"]) < TAU_FAIL)}

    c2 = read_direction_from_source(
        synchronous_amplitude_source(64, 64.0, 0.25, 120.0, 10.0, np.deg2rad(10.0)),
        montage, 3.0)
    s2 = c2["spearman_vs_shaft"]
    verdict["controls"]["C2_synchronous"] = {
        "spearman_vs_shaft": s2,
        "must_fail_ok": bool(np.isnan(s2) or abs(s2) < TAU_FAIL)}

    verdict["GATE_PASS"] = bool(
        all(w["pass"] for w in verdict["waves"].values() if w["spearman"] == w["spearman"])
        and verdict["controls"]["C1_radial"]["must_fail_ok"]
        and verdict["controls"]["C2_synchronous"]["must_fail_ok"])

    (OUT / "gate_verdict.json").write_text(json.dumps(verdict, indent=2,
                                                      default=lambda o: None))

    # Figure: read-out axis vs imposed angle for the waves + control markers
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    degs = [0, 30, 60, 90, 135]
    errs = [verdict["waves"][f"{d:g}deg"]["axis_err_deg"] or np.nan for d in degs]
    rhos = [verdict["waves"][f"{d:g}deg"]["spearman"] for d in degs]
    ax[0].bar([str(d) for d in degs], rhos)
    ax[0].axhline(TAU_PASS, ls="--", color="k")
    ax[0].set_title("Toy-wave read-out Spearman vs imposed angle")
    ax[0].set_ylabel("Spearman rho"); ax[0].set_xlabel("imposed direction (deg)")
    ax[1].bar([str(d) for d in degs], errs)
    ax[1].axhline(25.0, ls="--", color="k")
    ax[1].set_title("Endpoint-axis angle error"); ax[1].set_ylabel("deg")
    fig.tight_layout()
    fig.savefig(OUT.parent / "figures" / "increment1_gate.png", dpi=130)
    plt.close(fig)

    # Persist one example artifact (30deg) to prove the legacy write path runs
    ex = read_direction_from_source(
        traveling_wave(64, 64.0, np.deg2rad(30.0), 0.4, 0.25, 200.0, 8.0),
        montage, 3.0)["artifact"]
    write_legacy_npz(ex, OUT / "example30_lagPat_withFreqCent.npz")
    write_packed_times(ex, OUT / "example30_packedTimes_withFreqCent.npy")
    write_montage_manifest(ex, OUT / "example30_montage.json")
    print("GATE_PASS =", verdict["GATE_PASS"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the runner**

Run: `python scripts/run_sef_hfo_obs_increment1.py`
Expected: prints `GATE_PASS = True`; writes `gate_verdict.json`, `figures/increment1_gate.png`, and the example artifact triplet.

- [ ] **Step 3: Eyeball the figure**

Open `results/topic4_sef_hfo/observation_layer/figures/increment1_gate.png`. Confirm: all five wave bars ≥ 0.9 line; angle errors < 25° line. If a wave fails, debug the chain — do NOT relax thresholds.

- [ ] **Step 4: Write figures/README.md (Chinese, per AGENTS.md)**

```markdown
# results/topic4_sef_hfo/observation_layer/figures

### increment1_gate.png

增量 1「已知方向玩具波合同门」结果。左图：对 0/30/60/90/135° 五个已知传播方向的合成
平滑行波，虚拟电极读出的通道先后顺序与真方向投影顺序的 Spearman（虚线 = 通过线 0.9）。
右图：endpoint-centroid 轴与真方向的夹角误差（虚线 = 25° 容差）。两图都越过线 = 观测
层能把已知方向干净读回来。负对照（C1 居中径向 = 无轴、C2 同起同落幅度差 = 无假序）的
裁决在 `../increment1_toywave/gate_verdict.json`。

**关注点**：五个 wave 必须全部 Spearman ≥ 0.9 且角误差 < 25°；任何一个不过 = 观测层
被几何污染，停下来调链路（绝不放宽阈值）。
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_sef_hfo_obs_increment1.py \
        results/topic4_sef_hfo/observation_layer/
git commit -m "feat(topic4 obs): Increment-1 gate runner + verdict JSON + figure + README"
```

---

## Self-Review notes (for the implementer)

- **Spec coverage:** Tasks 1–6 build the module units (montage / sampler / extractor / estimators / artifact) named in spec §2; Task 6 round-trips the real loader (§3 legacy-key contract); Task 5 locks the §3.5 estimator (k_dir=3, ε_deg, mod-180°, degenerate→None); Task 4 implements §4.1 (per-contact relative timing); Task 7 is the §5 gate with C1 (centered radial) + C2 (synchronous-amplitude). Deferred (spec §7) — Increment 2 SNN slice, rate-field parity, real-geometry montage, two-end swap — are intentionally NOT in this plan.
- **Thresholds are locked, not tuned:** τ_pass=0.9, τ_fail=0.3, k_dir=3, f=0.5 appear as constants. If a test fails, fix the chain, never the threshold (spec §10 + acceptance-gate discipline).
- **`n_event_min` / SNN-rate-dependent values** are Increment-2 concerns (not in this plan).
- **Phantom-mask discipline:** `validate_artifact` asserts non-participant ranks are NaN; `write_legacy_npz` calls it before saving. Downstream uses `bools` to mask, so NaN is correct (non-phantom) behavior.
