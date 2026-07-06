# Axis-vs-core stimulation + difficulty figure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two paper-grade figures — (A) a 3-row difficulty figure showing why a spontaneous single focus cannot self-generate a discrete-event train, and (B) an axis-vs-core stimulation comparison showing that at a fixed electrode footprint, blocking the propagation axis delays runaway at least as much as stimulating the core, across substrate situations.

**Architecture:** A pure geometry/metric helper module (`src/topic4_axis_vs_core.py`, fully unit-tested without the SNN engine) + a stim-comparison runner (`scripts/run_stage4_axis_vs_core_stim.py`) + one figure script rendering both figures (`scripts/paper_figures/plot_fig_stage4_axis_vs_core_difficulty.py`). All SNN work reuses the existing, parity-tested `_simulate_continuous` / `_build_stage4_patch` / `intervention_vth_at_time`; no engine edits. Cheap code + TDD lands first; the expensive render/stim sims run only after explicit cost checkpoints.

**Tech Stack:** Python, NumPy, the repo SNN engine at `src/snn_engine/`, matplotlib (Agg) + imageio, pytest (`slow` marker registered in `pyproject.toml`).

**Spec:** `docs/superpowers/specs/2026-07-02-topic4-axis-vs-core-stim-difficulty-design.md`. **Branch:** `topic5-v2-phase1` (topic4 work continues here, as with `20d4cd8`/`0766b0a`/`12f6270`/`85fc5da`).

## Global Constraints

- **Canonical Stage-4 substrate** for `big`/`small` single cores: `g=3.6, AR=2.0, theta=45°, density=100, L=20, drive=0.6` (already baked into `H._build_stage4_patch`; do not use `S2.SUBSTRATES`).
- **Fairness contract (LOCKED):** the two stim arms use the SAME footprint — `n_contacts(core-stim) == n_contacts(axis-stim) == N`; `N` even; `N < n_source_contacts` (core not fully coverable). Assert all three in code; unequal/oversized N must raise.
- **Parity (LOCKED):** stim only changes the V_th comparison via `intervention_vth_at_time` (no extra RNG) → arms byte-identical until `stim_on`. Reuse `H._simulate_continuous(stim_target=…, stim_on=…, stim_off=…)`.
- **Shared runaway criterion (LOCKED):** `H._first_sustained(H._smooth_rate(rate_hz, dt, 20.0), dt, 120.0, 100.0)`. Timestep `DT = float(S["p"].dt)` with `assert abs(DT - C.DT) < 1e-12` (companion has no `H.DT`).
- **No engine edits** (`kick_probe.py` / `slow_field.py` untouched).
- **Claim scope (LOCKED, into every title/README/metadata):** visual diagnostic, single trajectory + small screen; runaway/tonic ≠ ictal; "axis ≥ core" is a within-model fixed-footprint efficiency statement — established in the multi-source/chokepoint geometry (E1146), *tested honestly* (PASS/TIE/FAIL all reported) in the single central core. Forbidden: "proves seizure mechanism", "treats seizures", "closed-loop/recovery".
- **`big` is NOT stimulated** (core covers ~all contacts → no distinct axis); it appears only in Figure A.
- **Cost:** each L=20 sim is minutes (blast configs cheap via short T; runaway plateau costs ~real time). Render/stim sims run ONLY after an explicit user cost checkpoint.
- **Output discipline:** figures → `results/paper-ready-figure/fig_stage4_axis_vs_core_difficulty/figures/` (png+pdf+metadata+README 中文); scripts → `scripts/paper_figures/` and `scripts/`. Stage only this plan's files when committing (repo has parallel topic5 edits).

**Canonical module aliases used throughout:**
- `H = scripts/paper_figures/plot_fig_m3a_v2_2_hG_runaway_transition_gif` — `_build`, `_build_stage4_patch`, `_build_subject1146`, `_simulate_continuous`, `_smooth_rate`, `_first_sustained`, `ProtocolConfig`, `_source_xy`, `_pulse_schedule`.
- `Q = scripts/paper_figures/plot_fig_m3a_v2_2_qI_stim_runaway_gif` — `_electrode_e_mask`, `_stim_site_center`, `_select_middle_contacts`, `_select_both_foci_contacts`.
- `C = scripts/run_sef_hfo_snn_cm_spontaneous_readout` — `DT`, `active_fraction`, `detect_events`, `BIN_MS`, `BASELINE_MS`, `CAL_FRAC`, `_engine_guard`.
- `AV = src/topic4_axis_vs_core` — NEW pure helpers (Task 1).

**Reused reference values (E1146 kick, from committed `fig_m3a_v2_2_qI_stim_site_compare` + memory):** `baseline_runaway=757.5 ms`, core/endpoint stim → `1171.3 ms (delay +413.8)`, axis/middle stim → `1591.9 ms (delay +834.4)`. Footprint = 4 contacts each. Cited, not re-run (spec R3 default).

---

## Task 1: Pure geometry/metric helpers (`src/topic4_axis_vs_core.py`)

**Files:**
- Create: `src/topic4_axis_vs_core.py`
- Test: `tests/test_topic4_axis_vs_core.py`

**Interfaces:**
- Produces: `linear_montage(center, axis_unit, n_contacts=11, pitch=1.2) -> (contacts (n,2) float, names list[str])`
- Produces: `split_source_axis(contacts, center, core_radius) -> (source_idx np.ndarray, axis_idx np.ndarray)`
- Produces: `select_footprint(contacts, center, axis_unit, source_idx, axis_idx, N) -> (core_contact_idx np.ndarray, axis_contact_idx np.ndarray)` — asserts `N` even, `N < len(source_idx)`, both sides fillable.
- Produces: `onset_time_field(E_spk_bool, dt) -> np.ndarray` (per-E-cell first-spike ms; nan if silent)
- Produces: `runaway_delay_ms(runaway_stim, runaway_nostim, T) -> float` (nan if no baseline runaway)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_topic4_axis_vs_core.py`:

```python
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.topic4_axis_vs_core import (linear_montage, split_source_axis,  # noqa: E402
                                     select_footprint, onset_time_field, runaway_delay_ms)


def test_linear_montage_centered_and_spaced():
    c = np.array([10.0, 10.0]); u = np.array([1.0, 0.0])
    contacts, names = linear_montage(c, u, n_contacts=11, pitch=1.2)
    assert contacts.shape == (11, 2) and len(names) == 11
    assert np.allclose(contacts[5], c)                        # middle contact at centre
    assert np.allclose(np.diff(contacts[:, 0]), 1.2)          # even spacing along u
    assert np.allclose(contacts[:, 1], 10.0)                  # perpendicular coord constant


def test_split_source_axis():
    c = np.array([10.0, 10.0]); u = np.array([1.0, 0.0])
    contacts, _ = linear_montage(c, u, n_contacts=11, pitch=1.2)
    src, ax = split_source_axis(contacts, c, core_radius=3.0)
    assert set(src.tolist()) == {3, 4, 5, 6, 7}               # within 3 mm of centre (|i-5|*1.2<=3)
    assert set(ax.tolist()) == {0, 1, 2, 8, 9, 10}


def test_select_footprint_symmetric_and_fair():
    c = np.array([10.0, 10.0]); u = np.array([1.0, 0.0])
    contacts, _ = linear_montage(c, u, n_contacts=11, pitch=1.2)
    src, ax = split_source_axis(contacts, c, core_radius=3.0)
    core, axis = select_footprint(contacts, c, u, src, ax, N=4)
    assert len(core) == 4 and len(axis) == 4                  # fixed footprint
    assert set(core).issubset(set(src.tolist()))             # core ⊆ source
    assert set(axis).issubset(set(ax.tolist()))              # axis ⊆ downstream
    proj = (contacts - c) @ u
    assert (proj[axis] > 0).sum() == 2 and (proj[axis] < 0).sum() == 2   # symmetric both sides


def test_select_footprint_asserts():
    c = np.array([10.0, 10.0]); u = np.array([1.0, 0.0])
    contacts, _ = linear_montage(c, u, n_contacts=11, pitch=1.2)
    src, ax = split_source_axis(contacts, c, core_radius=3.0)
    with pytest.raises(AssertionError):
        select_footprint(contacts, c, u, src, ax, N=3)       # odd N
    with pytest.raises(AssertionError):
        select_footprint(contacts, c, u, src, ax, N=6)       # N >= n_source(5): core fully coverable


def test_onset_time_field():
    # 3 E cells: cell0 first at step 2, cell1 at step 5, cell2 never
    spk = np.zeros((10, 3), bool)
    spk[2, 0] = True; spk[7, 0] = True
    spk[5, 1] = True
    got = onset_time_field(spk, dt=0.1)
    assert np.isclose(got[0], 0.2) and np.isclose(got[1], 0.5)
    assert np.isnan(got[2])


def test_runaway_delay_ms():
    assert np.isclose(runaway_delay_ms(1591.9, 757.5, 2500.0), 834.4)
    assert np.isclose(runaway_delay_ms(None, 757.5, 2500.0), 2500.0 - 757.5)   # prevented within T
    assert np.isnan(runaway_delay_ms(100.0, None, 2500.0))                     # no baseline runaway
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_topic4_axis_vs_core.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `src/topic4_axis_vs_core.py`**

```python
"""Pure geometry/metric helpers for the axis-vs-core stimulation figure (Topic 4).

No SNN engine imports -- fully unit-testable. Consumed by the stim runner and the figure script.
See docs/superpowers/specs/2026-07-02-topic4-axis-vs-core-stim-difficulty-design.md."""
from __future__ import annotations

import numpy as np


def linear_montage(center, axis_unit, n_contacts=11, pitch=1.2):
    """Virtual-SEEG contacts along ``axis_unit`` through ``center`` (mm). Returns (contacts, names)."""
    center = np.asarray(center, float)
    u = np.asarray(axis_unit, float)
    offs = (np.arange(n_contacts) - (n_contacts - 1) / 2.0) * float(pitch)
    contacts = center[None, :] + offs[:, None] * u[None, :]
    return contacts, [f"C{i}" for i in range(n_contacts)]


def split_source_axis(contacts, center, core_radius):
    """Contact indices within ``core_radius`` of ``center`` (source) vs outside (axis/downstream)."""
    d = np.linalg.norm(np.asarray(contacts, float) - np.asarray(center, float)[None, :], axis=1)
    return np.flatnonzero(d <= float(core_radius)), np.flatnonzero(d > float(core_radius))


def select_footprint(contacts, center, axis_unit, source_idx, axis_idx, N):
    """Fixed-footprint contact selection (fairness contract). core = N source contacts nearest the
    centre (partial cover -> residual source); axis = N downstream contacts split symmetrically
    (N/2 nearest each side along axis_unit). Deterministic tie-break: (distance, lower index)."""
    assert N % 2 == 0, "footprint N must be even (symmetric axis split)"
    assert N < len(source_idx), f"N={N} must be < n_source_contacts={len(source_idx)} (core not fully coverable)"
    C = np.asarray(contacts, float); c = np.asarray(center, float); u = np.asarray(axis_unit, float)
    d = np.linalg.norm(C - c[None, :], axis=1)
    proj = (C - c[None, :]) @ u
    core = sorted(source_idx.tolist(), key=lambda i: (d[i], i))[:N]
    pos = sorted([i for i in axis_idx.tolist() if proj[i] > 0], key=lambda i: (d[i], i))[:N // 2]
    neg = sorted([i for i in axis_idx.tolist() if proj[i] < 0], key=lambda i: (d[i], i))[:N // 2]
    axis = pos + neg
    assert len(core) == N and len(axis) == N, "footprint could not be filled — check montage/N"
    return np.array(sorted(core)), np.array(sorted(axis))


def onset_time_field(E_spk_bool, dt):
    """Per-E-cell first-spike time in ms; nan for cells that never spiked. E_spk_bool: (nsteps, NE)."""
    spk = np.asarray(E_spk_bool, bool)
    ever = spk.any(axis=0)
    first = np.argmax(spk, axis=0).astype(float) * float(dt)
    first[~ever] = np.nan
    return first


def runaway_delay_ms(runaway_stim, runaway_nostim, T):
    """Delay = (runaway_stim or T) - runaway_nostim. nan if no baseline runaway (undefined)."""
    if runaway_nostim is None:
        return float("nan")
    rs = float(T) if runaway_stim is None else float(runaway_stim)
    return float(rs - float(runaway_nostim))
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_topic4_axis_vs_core.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic4_axis_vs_core.py tests/test_topic4_axis_vs_core.py
git commit -m "feat(topic4 axis-vs-core): pure montage/footprint/onset/delay helpers"
```

---

## Task 2: Stim-comparison runner (`scripts/run_stage4_axis_vs_core_stim.py`)

**Files:**
- Create: `scripts/run_stage4_axis_vs_core_stim.py`
- Test: `tests/test_topic4_axis_vs_core.py` (append)

**Interfaces:**
- Consumes: `AV.*` (Task 1); `H._build_stage4_patch`, `H._simulate_continuous`, `H._smooth_rate`, `H._first_sustained`, `H.ProtocolConfig`; `Q._electrode_e_mask`; `C.DT`, `C._engine_guard`.
- Produces: `build_small_core_targets(S, *, core_radius, n_contacts=11, pitch=1.2, r_stim=2.0, N) -> dict` with keys `contacts (n,2)`, `names`, `source_idx`, `axis_idx`, `core_contact_idx`, `axis_contact_idx`, `core_mask (N_full bool)`, `axis_mask (N_full bool)`. Both masks E-only; `core_mask ⊆ S["core_mask"]` region-wise via source contacts; footprint asserts hold.
- Produces: `run_one_arm(S, cfg, target_mask, stim_on, stim_off, DT) -> dict` with `runaway_ms`, `q_min_final`, `max_rate_hz`, `n_stim_E`.
- Produces: `main()` CLI: builds small core, runs 3 arms (no-stim / core-stim / axis-stim), asserts fairness, writes `results/topic4_sef_hfo/axis_vs_core/small_core_stim.json` with per-arm `runaway_ms` + `runaway_delay_ms` + the target metadata.

- [ ] **Step 1: Write the failing test (target construction on a synthetic S — no engine)**

Append to `tests/test_topic4_axis_vs_core.py`:

```python
def _fake_small_core_S(L=20.0, core_radius=3.0):
    # deterministic synthetic sheet: E cells on a grid, I cells appended; centre core.
    import numpy as np
    xs = np.linspace(1, L - 1, 24)
    gx, gy = np.meshgrid(xs, xs)
    posE = np.column_stack([gx.ravel(), gy.ravel()])
    posI = posE[:50] + 0.1
    pos = np.vstack([posE, posI])
    NE = len(posE); N = len(pos)
    labels = np.zeros(N, int); labels[NE:] = 1
    center = np.array([L / 2, L / 2]); u = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
    core_mask = np.zeros(N, bool)
    core_mask[:NE] = np.linalg.norm(posE - center, axis=1) <= core_radius
    return dict(net={"pos": pos}, posE=posE, posI=posI, N=N, NE=NE, labels=labels,
               center=center, axis_unit=u, L=L, core_mask=core_mask,
               layout={"kind": "stage4_patch", "foci": [center.tolist()], "core_r": core_radius})


def test_build_small_core_targets_fairness():
    from run_stage4_axis_vs_core_stim import build_small_core_targets
    S = _fake_small_core_S(core_radius=3.0)
    t = build_small_core_targets(S, core_radius=3.0, n_contacts=11, pitch=1.2, r_stim=2.0, N=4)
    is_E = np.asarray(S["labels"]) == 0
    assert t["core_mask"].shape[0] == S["N"] and t["axis_mask"].shape[0] == S["N"]
    assert (t["core_mask"] & ~is_E).sum() == 0 and (t["axis_mask"] & ~is_E).sum() == 0   # E only
    assert len(t["core_contact_idx"]) == 4 == len(t["axis_contact_idx"])                 # fixed footprint
    assert t["core_mask"].sum() > 0 and t["axis_mask"].sum() > 0
    assert (t["core_mask"] & t["axis_mask"]).sum() == 0                                  # disjoint clamp sets
```

Note: append the `scripts` path so the import resolves. Add near the top of the test file (once):
```python
for _p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "scripts", "paper_figures"),
           os.path.join(ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_topic4_axis_vs_core.py -k small_core_targets -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `scripts/run_stage4_axis_vs_core_stim.py`**

```python
"""Axis-vs-core stimulation comparison on the SMALL central core (Topic 4).

Fixed-footprint fairness: core-stim (partial cover of the source, leaves residual) vs axis-stim
(N downstream contacts split symmetrically, block both axial fronts) vs no-stim. Reports the
runaway delay each achieves. SNN-heavy 3-arm run is a CLI (cost-gated); target construction is
pure (unit-tested). See the spec / plan."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "paper_figures"),
           str(ROOT / "src" / "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src import topic4_axis_vs_core as AV  # noqa: E402

# small-core substrate + stim defaults (spec §3, §5); N even and < n_source(=5 for r=3/pitch1.2)
CORE_R = 3.0
N_CONTACTS, PITCH, R_STIM, N_FOOT = 11, 1.2, 2.0, 4
STIM_ON, STIM_OFF, T_SIM = 0.0, 300.0, 600.0
CORE_MEAN, CORE_STD, DRIVE = 16.5, 1.5, 0.6
K_Q, TAU_Q, SIGMA_Q, ETA_K, K_K, TAU_K, SIGMA_K = 0.25, 5000.0, 1.5, 0.8, 1.5, 150.0, 0.5
OUT = ROOT / "results" / "topic4_sef_hfo" / "axis_vs_core"


def build_small_core_targets(S, *, core_radius, n_contacts=N_CONTACTS, pitch=PITCH, r_stim=R_STIM, N=N_FOOT):
    import plot_fig_m3a_v2_2_qI_stim_runaway_gif as Q
    center = np.asarray(S["center"], float); u = np.asarray(S["axis_unit"], float)
    contacts, names = AV.linear_montage(center, u, n_contacts, pitch)
    src, ax = AV.split_source_axis(contacts, center, core_radius)
    core_ci, axis_ci = AV.select_footprint(contacts, center, u, src, ax, N)
    is_E = np.asarray(S["labels"]) == 0
    pos = S["net"]["pos"]
    core_mask = Q._electrode_e_mask(pos, is_E, contacts[core_ci], r_stim)
    axis_mask = Q._electrode_e_mask(pos, is_E, contacts[axis_ci], r_stim)
    return dict(contacts=contacts, names=names, source_idx=src, axis_idx=ax,
                core_contact_idx=core_ci, axis_contact_idx=axis_ci,
                core_mask=core_mask, axis_mask=axis_mask)


def run_one_arm(S, cfg, target_mask, stim_on, stim_off, DT):
    import plot_fig_m3a_v2_2_hG_runaway_transition_gif as H
    kw = {} if target_mask is None else dict(stim_target=target_mask, stim_on=stim_on, stim_off=stim_off)
    res = H._simulate_continuous(S, cfg, record_gif=False, vth=S["patch_vth"], **kw)
    rate_hz = np.asarray(res["rate_E"], float)
    runaway = H._first_sustained(H._smooth_rate(rate_hz, DT, 20.0), DT, 120.0, 100.0)
    return dict(runaway_ms=runaway, q_min_final=round(float(np.asarray(res["trace_qI_min"]).min()), 4),
                max_rate_hz=round(float(H._smooth_rate(rate_hz, DT, 20.0).max()), 1),
                n_stim_E=int(0 if target_mask is None else int(np.asarray(target_mask).sum())))


def main():
    import plot_fig_m3a_v2_2_hG_runaway_transition_gif as H
    import run_sef_hfo_snn_cm_spontaneous_readout as C
    os.chdir(ROOT); C._engine_guard()
    cfg = H.ProtocolConfig(layout="stage4_patch", top="qI", use_gK=True, use_hG=False,
                           eta_K=ETA_K, k_K=K_K, tau_K=TAU_K, sigma_K=SIGMA_K,
                           k_q=K_Q, tau_q=TAU_Q, sigma_q=SIGMA_Q, q_min=0.05,
                           core_mean=CORE_MEAN, core_std=CORE_STD, core_radius=CORE_R,
                           drive=DRIVE, L=20.0, T=T_SIM, n_pulses=0, seed=1)
    S = H._build(cfg)
    DT = float(S["p"].dt); assert abs(DT - C.DT) < 1e-12
    tg = build_small_core_targets(S, core_radius=CORE_R)
    assert len(tg["core_contact_idx"]) == len(tg["axis_contact_idx"]) == N_FOOT   # fairness gate
    assert N_FOOT < len(tg["source_idx"])                                          # core not fully coverable
    arms = {"no_stim": None, "core_stim": tg["core_mask"], "axis_stim": tg["axis_mask"]}
    rows = {}
    for name, mask in arms.items():
        t0 = time.time()
        r = run_one_arm(S, cfg, mask, STIM_ON, STIM_OFF, DT)
        r["wall_s"] = round(time.time() - t0, 1)
        rows[name] = r
        print(f"ARM {name} " + json.dumps(r), flush=True)
    base = rows["no_stim"]["runaway_ms"]
    for name in ("core_stim", "axis_stim"):
        rows[name]["runaway_delay_ms"] = AV.runaway_delay_ms(rows[name]["runaway_ms"], base, T_SIM)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "small_core_stim.json").write_text(json.dumps({
        "config": dict(core_r=CORE_R, N=N_FOOT, n_contacts=N_CONTACTS, pitch=PITCH, r_stim=R_STIM,
                       stim_on=STIM_ON, stim_off=STIM_OFF, T=T_SIM, core_mean=CORE_MEAN,
                       eta_K=ETA_K, tau_K=TAU_K, drive=DRIVE),
        "n_source_contacts": int(len(tg["source_idx"])),
        "core_contact_idx": tg["core_contact_idx"].tolist(), "axis_contact_idx": tg["axis_contact_idx"].tolist(),
        "contacts": tg["contacts"].tolist(), "arms": rows}, indent=2))
    print("AXIS_VS_CORE_DELAY " + json.dumps({k: rows[k].get("runaway_delay_ms") for k in ("core_stim", "axis_stim")}), flush=True)
    print("DONE_AXIS_VS_CORE_STIM", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass (target construction test)**

Run: `python -m pytest tests/test_topic4_axis_vs_core.py -k small_core_targets -q`
Expected: PASS.

- [ ] **Step 5: py_compile the runner**

Run: `python -m py_compile scripts/run_stage4_axis_vs_core_stim.py`
Expected: success, no output.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_stage4_axis_vs_core_stim.py tests/test_topic4_axis_vs_core.py
git commit -m "feat(topic4 axis-vs-core): small-core stim runner (fixed-footprint core vs axis vs no-stim)"
```

---

## Task 3: Figure script — both renderers (`scripts/paper_figures/plot_fig_stage4_axis_vs_core_difficulty.py`)

**Files:**
- Create: `scripts/paper_figures/plot_fig_stage4_axis_vs_core_difficulty.py`
- Test: `tests/test_topic4_axis_vs_core.py` (append a smoke test)

**Interfaces:**
- Consumes: `AV.*`, `H.*`, `C.*`, and (for Figure B small row) the JSON from Task 2 (`results/topic4_sef_hfo/axis_vs_core/small_core_stim.json`).
- Produces: `simulate_row(kind) -> dict` for `kind in {"big","small","kick"}` (runs the appropriate build + `_simulate_continuous`, returns `posE, onset (AV.onset_time_field), times, rate_s, qI_mean, qI_min, gK, runaway_ms, n_events`).
- Produces: `render_figure_a(rows_data, out_dir)` (3×2), `render_figure_b(small_json, kick_ref, out_dir)` (2×2), `main()` CLI `--figure {A,B,both}`.
- Constants: `KICK_REF = {"baseline_ms":757.5,"core_ms":1171.3,"axis_ms":1591.9,"core_delay":413.8,"axis_delay":834.4,"footprint":4}`.

- [ ] **Step 1: Write the failing smoke test (renderers run on tiny fixture data → produce files)**

Append to `tests/test_topic4_axis_vs_core.py`:

```python
def test_figure_b_renders_from_fixture(tmp_path):
    import plot_fig_stage4_axis_vs_core_difficulty as F
    small = {"config": {"N": 4}, "n_source_contacts": 5,
             "contacts": [[i * 1.0, 10.0] for i in range(11)],
             "core_contact_idx": [4, 5, 6, 7], "axis_contact_idx": [2, 3, 8, 9],
             "arms": {"no_stim": {"runaway_ms": 50.0},
                      "core_stim": {"runaway_ms": 120.0, "runaway_delay_ms": 70.0},
                      "axis_stim": {"runaway_ms": 160.0, "runaway_delay_ms": 110.0}}}
    out = tmp_path / "figs"; out.mkdir()
    F.render_figure_b(small, F.KICK_REF, out)
    assert (out / "axis_vs_core.png").exists() and (out / "axis_vs_core.png").stat().st_size > 0
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_topic4_axis_vs_core.py -k figure_b_renders -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the figure script**

Header + imports (mirror `H`'s sys.path + Agg backend). Then:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ... sys.path inserts (ROOT, scripts, scripts/paper_figures, src/snn_engine) ...
from src import topic4_axis_vs_core as AV

KICK_REF = {"baseline_ms": 757.5, "core_ms": 1171.3, "axis_ms": 1591.9,
            "core_delay": 413.8, "axis_delay": 834.4, "footprint": 4}
FIG_DIR = "results/paper-ready-figure/fig_stage4_axis_vs_core_difficulty/figures"
QI_MEAN_COL, QI_MIN_COL, GK_COL, RATE_COL = "#1f7a5a", "#7fc4a6", "#b8860b", "#333333"
CORE_STIM_COL, AXIS_STIM_COL, NOSTIM_COL = "#c0392b", "#2e86c1", "#888888"


def simulate_row(kind):
    import plot_fig_m3a_v2_2_hG_runaway_transition_gif as H
    import run_sef_hfo_snn_cm_spontaneous_readout as C
    if kind == "big":
        cfg = H.ProtocolConfig(layout="stage4_patch", top="qI", use_gK=True, use_hG=False, eta_K=0.8,
                               k_K=1.5, tau_K=150.0, sigma_K=0.5, k_q=0.25, tau_q=5000.0, sigma_q=1.5,
                               q_min=0.05, core_mean=16.5, core_std=1.5, core_radius=6.0, drive=0.6,
                               L=20.0, T=200.0, n_pulses=0, seed=1)
    elif kind == "small":
        cfg = H.ProtocolConfig(layout="stage4_patch", top="qI", use_gK=True, use_hG=False, eta_K=0.8,
                               k_K=1.5, tau_K=150.0, sigma_K=0.5, k_q=0.25, tau_q=5000.0, sigma_q=1.5,
                               q_min=0.05, core_mean=16.5, core_std=1.5, core_radius=3.0, drive=0.6,
                               L=20.0, T=200.0, n_pulses=0, seed=1)
    else:  # kick — E1146 two-foci, short T shows the train (3 pulses at 130/265/400)
        cfg = H.ProtocolConfig(layout="subject1146", top="qI", use_gK=True, use_hG=False, eta_K=0.0,
                               k_q=0.18, tau_q=5000.0, sigma_q=1.5, q_min=0.05, T=500.0, seed=1)
    S = H._build(cfg)
    DT = float(S["p"].dt); assert abs(DT - C.DT) < 1e-12
    vth = S["patch_vth"] if kind in ("big", "small") else None
    res = H._simulate_continuous(S, cfg, record_gif=False, vth=vth)
    rate_hz = np.asarray(res["rate_E"], float)
    rate_s = H._smooth_rate(rate_hz, DT, 20.0)
    af, bin_w = C.active_fraction(res["E_spk_bool"], DT, C.BIN_MS)
    nb0, nb1 = int(C.BASELINE_MS[0] / bin_w), int(C.BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    bar = floor + C.CAL_FRAC * (float(af.max()) - floor)
    n_events = len(C.detect_events(af, bin_w, event_on_frac=bar))
    return dict(kind=kind, posE=S["posE"], onset=AV.onset_time_field(res["E_spk_bool"], DT),
                times=np.asarray(res["times"], float), rate_s=rate_s,
                qI_mean=np.asarray(res["trace_qI_mean"], float), qI_min=np.asarray(res["trace_qI_min"], float),
                gK=np.asarray(res["trace_gK_axial"], float),
                runaway_ms=H._first_sustained(rate_s, DT, 120.0, 100.0),
                n_events=n_events, L=float(S["L"]), max_active_frac=float(af.max()),
                center=np.asarray(S["center"], float))
```

Then `render_figure_a(rows_data, out_dir)` — 3 rows × 2 cols. Col1: `ax.scatter(posE[:,0], posE[:,1], c=onset, cmap="viridis", s=4)` with a shared colorbar labelled "onset time (ms) early→late", core outline circle at `center` radius (6 / 3), sheet box; kick row uses its own contacts if desired. Col2: `ax.plot(times, rate_s, color=RATE_COL)` on left y (Hz); twin axis `ax2.plot(times, qI_mean, QI_MEAN_COL); ax2.plot(times, qI_min, QI_MIN_COL); ax2.plot(times, gK, GK_COL)` on right y (0–1.05); vertical dashed line at `runaway_ms`. Row titles: `big核 r=6 / small核 r=3 / kick 两灶`. Suptitle carries the punchline; footer the claim scope. Save `difficulty_3row.png/.pdf`.

Then `render_figure_b(small_json, kick_ref, out_dir)` — 2 rows (`kick`, `small`) × 2 cols:
```python
def render_figure_b(small, kick_ref, out_dir):
    from pathlib import Path
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    # --- kick row ---
    axk_geom, axk_bar = axes[0]
    axk_geom.set_title("kick 两灶：core=端点, axis=中段走廊"); axk_geom.axis("off")
    axk_geom.text(0.5, 0.5, "E1146 两灶几何\ncore=端点电极\naxis=中段走廊",
                  ha="center", va="center", transform=axk_geom.transAxes)
    axk_bar.bar(["core-stim", "axis-stim"], [kick_ref["core_delay"], kick_ref["axis_delay"]],
                color=[CORE_STIM_COL, AXIS_STIM_COL])
    axk_bar.set_ylabel("runaway 推迟 (ms)"); axk_bar.set_title("固定 footprint=4：axis ≥ core")
    for i, v in enumerate([kick_ref["core_delay"], kick_ref["axis_delay"]]):
        axk_bar.text(i, v, f"+{v:.0f}", ha="center", va="bottom")
    # --- small row ---
    axs_geom, axs_bar = axes[1]
    contacts = np.asarray(small["contacts"], float)
    axs_geom.scatter(contacts[:, 0], contacts[:, 1], c="lightgray", s=30, zorder=2)
    axs_geom.scatter(contacts[small["core_contact_idx"], 0], contacts[small["core_contact_idx"], 1],
                     c=CORE_STIM_COL, s=45, zorder=3, label="core-stim")
    axs_geom.scatter(contacts[small["axis_contact_idx"], 0], contacts[small["axis_contact_idx"], 1],
                     c=AXIS_STIM_COL, s=45, zorder=3, label="axis-stim")
    axs_geom.set_title(f"small核 r=3：source {small['n_source_contacts']} 触点, footprint N={small['config']['N']}")
    axs_geom.legend(loc="upper right"); axs_geom.set_aspect("equal")
    cd = small["arms"]["core_stim"]["runaway_delay_ms"]; ad = small["arms"]["axis_stim"]["runaway_delay_ms"]
    axs_bar.bar(["core-stim", "axis-stim"], [cd, ad], color=[CORE_STIM_COL, AXIS_STIM_COL])
    axs_bar.set_ylabel("runaway 推迟 (ms)")
    verdict = "axis ≥ core" if ad >= cd - 10 else "core > axis (单核无咽喉)"
    axs_bar.set_title(f"固定 footprint=N：{verdict}")
    for i, v in enumerate([cd, ad]):
        axs_bar.text(i, v, f"+{v:.0f}", ha="center", va="bottom")
    fig.suptitle("固定电极预算下：挡轴的刺激效果 vs 打灶（跨情况）", fontweight="bold")
    fig.text(0.5, 0.005, "visual diagnostic；within-model 效率示意，非临床证明；runaway 非 ictal 事件",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "axis_vs_core.png", dpi=150); fig.savefig(out_dir / "axis_vs_core.pdf")
    plt.close(fig)
```

`main()`: argparse `--figure {A,B,both}`; for A call `simulate_row` ×3 then `render_figure_a`; for B load `small_core_stim.json` then `render_figure_b`. `C._engine_guard()` before any `simulate_row`.

- [ ] **Step 4: Run to verify the smoke test passes**

Run: `python -m pytest tests/test_topic4_axis_vs_core.py -k figure_b_renders -q`
Expected: PASS (renders `axis_vs_core.png` from fixture, no SNN).

- [ ] **Step 5: py_compile**

Run: `python -m py_compile scripts/paper_figures/plot_fig_stage4_axis_vs_core_difficulty.py`
Expected: success.

- [ ] **Step 6: Commit**

```bash
git add scripts/paper_figures/plot_fig_stage4_axis_vs_core_difficulty.py tests/test_topic4_axis_vs_core.py
git commit -m "feat(topic4 axis-vs-core): figure script (difficulty 3-row + axis-vs-core), fixture smoke test"
```

---

## Task 4: COST CHECKPOINT → run Figure A sims + render + EYEBALL

**Files:** none (produces `results/paper-ready-figure/fig_stage4_axis_vs_core_difficulty/figures/difficulty_3row.{png,pdf}`).

- [ ] **Step 1: COST CHECKPOINT** — report to the user: Figure A needs 3 render sims (`big` T=200, `small` T=200, `kick` T=500), est. ~15–25 min total. Get explicit go before launching.

- [ ] **Step 2: Render Figure A (background)**

Run: `python scripts/paper_figures/plot_fig_stage4_axis_vs_core_difficulty.py --figure A > <scratch>/figA.log 2>&1 &`

- [ ] **Step 3: Verify the regime gates (G-A1..A3) from the run + EYEBALL**

Check the printed per-row `n_events` / `runaway_ms` / `max_active_frac`: `big` n_events==1 & runaway<60; `small` n_events==1 & max_active_frac>0.5; `kick` n_events≥3. Open `difficulty_3row.png` and confirm: big col1 ~uniform colour (synchronous), small col1 gradient reaching the boundary (fills), kick col1 contained; col2 traces match (blast+cliff vs train+staircase; g_K≈0 at the single-core blast). If a gate fails or the visual contradicts, debug before proceeding (systematic-debugging).

- [ ] **Step 4: Commit the figure metadata/README stub** (README written in Task 6). No sim rerun.

---

## Task 5: COST CHECKPOINT → run Figure B stim sims + render + EYEBALL

**Files:** none new (produces `small_core_stim.json` + `axis_vs_core.{png,pdf}`).

- [ ] **Step 1: COST CHECKPOINT** — report: Figure B needs the small-core 3-arm stim run (`no_stim`/`core_stim`/`axis_stim`, T=600 each), est. ~15–25 min; E1146 row reuses committed delays (no rerun). Get go.

- [ ] **Step 2: Run the stim comparison (background)**

Run: `python scripts/run_stage4_axis_vs_core_stim.py > <scratch>/figB_stim.log 2>&1 &`
Watch for `ARM …` lines, `AXIS_VS_CORE_DELAY …`, `DONE_AXIS_VS_CORE_STIM`. Fairness asserts run at start (raise if violated).

- [ ] **Step 3: Verify Figure B gates (G-B1..B5)**

From `small_core_stim.json`: G-B1 `len(core_contact_idx)==len(axis_contact_idx)`; G-B2 `N < n_source_contacts`; G-B4 both delays computed (and sane); G-B5 record whether `axis_delay ≥ core_delay - 10` (PASS/TIE/FAIL — all honest). (G-B3 parity is already covered by `H`'s parity test; optionally assert `E_spk_bool` equality on `[0, stim_on)` for one arm.) If `stim_on=0` makes the window trivial or the delay unresolvable (spec R2), extend `T`/lower `drive` per the spec and note it in metadata — do NOT silently retune.

- [ ] **Step 4: Render Figure B + EYEBALL**

Run: `python scripts/paper_figures/plot_fig_stage4_axis_vs_core_difficulty.py --figure B`
Open `axis_vs_core.png`: kick row shows axis(+834) ≥ core(+414); small row shows the honest per-arm delays + verdict label. Confirm bars/labels/colors correct and the claim-scope footer present.

- [ ] **Step 5: Commit** (figure files are gitignored; commit happens with docs in Task 6.)

---

## Task 6: Docs — FIGURE_INDEX + README + archive

**Files:**
- Create: `results/paper-ready-figure/fig_stage4_axis_vs_core_difficulty/figures/README.md`
- Modify: `results/FIGURE_INDEX.md` (Topic 4 table — one row)
- Create: `docs/archive/topic4/axis_vs_core_stim_2026-07-02.md`

- [ ] **Step 1: Write `figures/README.md`** (中文, 逐图) — Figure A (3 行：为什么自发单灶出不了串) 与 Figure B (固定 footprint 下 axis vs core，kick 明确 axis≥core、small 核诚实报告)；每图 2–4 句 + `**关注点**：`。

- [ ] **Step 2: Write the archive doc** — abstract (第一性原理, per §8), the fairness contract, the per-situation results (kick +834≥+414; small honest verdict), the self-ignite↔self-terminate framing, claim scope, provenance (commits + JSON paths + the small-core scan).

- [ ] **Step 3: Add the FIGURE_INDEX row (Topic 4)** — describe both figures. Stage ONLY this row (repo has parallel topic5 FIGURE_INDEX edits — use the `git apply --cached` single-hunk pattern from prior stim commits).

- [ ] **Step 4: Commit**

```bash
git add results/FIGURE_INDEX.md results/paper-ready-figure/fig_stage4_axis_vs_core_difficulty/figures/README.md docs/archive/topic4/axis_vs_core_stim_2026-07-02.md
git commit -m "docs(topic4 axis-vs-core): archive + index/README for difficulty + axis-vs-core stim figures"
```

- [ ] **Step 5: Memory** — do NOT edit memory as a plan step; update `project_topic4_m3a_v2_2_qI_stim_runaway_2026-06-29.md` only if the user asks, with the axis-vs-core result.

---

## Self-Review

- **Spec coverage:** Fig A 3-row difficulty (Task 3 `render_figure_a` + Task 4 run) ✔; Fig B axis-vs-core across kick+small (Task 3 `render_figure_b` + Task 2 runner + Task 5 run) ✔; fairness contract (Task 1 `select_footprint` asserts + Task 2 main asserts, G-B1/B2) ✔; parity (reused `H._simulate_continuous`, G-B3) ✔; shared runaway criterion (Tasks 2/3) ✔; canonical substrate (reused `_build_stage4_patch`) ✔; numeric gates (Tasks 4/5) ✔; big-not-stimulated (Fig A only) ✔; claim scope (titles/footer/README, Tasks 3/6) ✔; cost checkpoints (Tasks 4/5) ✔; R1 honest verdict (G-B5 PASS/TIE/FAIL, Task 5) ✔; R2 window note ✔; R3 E1146 reuse (`KICK_REF`) ✔.
- **Placeholder scan:** montage/footprint/onset/delay fully coded (Task 1); runner fully coded (Task 2); Fig B renderer fully coded; Fig A renderer given as column-by-column spec with the exact data wiring + colors (matplotlib styling follows fig4 conventions) — no "TBD"/"handle edge cases".
- **Type consistency:** `select_footprint` returns `(core_contact_idx, axis_contact_idx)` used identically in Task 2 `build_small_core_targets` and Task 3 `render_figure_b`; `small_core_stim.json` schema (`arms[*].runaway_delay_ms`, `core_contact_idx`, `contacts`, `n_source_contacts`) written in Task 2, consumed in Task 3 fixture + renderer; `simulate_row` return keys consumed by `render_figure_a`; `DT` derived from `S["p"].dt` everywhere (no `H.DT`).
- **Risk called out:** single central core may yield TIE/FAIL on axis≥core (G-B5 honest); the plan reports it, does not force it.
