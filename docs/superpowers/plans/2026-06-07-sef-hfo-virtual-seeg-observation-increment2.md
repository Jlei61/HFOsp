# SEF-HFO 虚拟 SEEG 观测层 — Increment 2 Implementation Plan (SNN 低驱动竖切)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
> **STATUS: draft 2026-06-07 — written for USER REVIEW. Do NOT execute / let subagents write code until the user ratifies §0.**

**Goal:** Read the SNN's single-end-kicked self-limited event through **virtual SEEG contacts + the real propagation pipeline**, and show the recovered template direction tracks the E→E connectivity axis (`θ_EE`) — not the electrode geometry — with the locked must-fail controls failing.

**Architecture:** Reuse the Increment-1 observation module verbatim (`montage / sample_envelopes / extract_lagpat / direction estimators`). Add ONE thin SNN adapter (`spikes → per-contact envelope`) + ONE backward-compatible engine kwarg (`kick_center`). The connectivity rotation + isotropic control already exist in the engine (`build_connectivity_rot`). The source-space oracle (`anisotropy_front.principal_axis`, already PASSES) is kept as a parity upper-bound, NOT the gate.

**Engine interface map (provenance):** `docs/.../tasks/wrryxksbh` workflow (3 readers over `results/topic4_sef_hfo/lif_snn/engine/`). Key facts cited inline below.

**Spec:** `docs/superpowers/specs/2026-06-06-sef-hfo-virtual-seeg-observation-layer-design.md` (Increment 2 = §6; controls = §6 + pathology-mapping spec §7 Track 连接).

---

## §0. Opening contract — the five locked items (USER must ratify before any code)

> All five answer "how exactly" so the implementation cannot drift. Threshold values marked **(proposed)** lock at review.

### Item 1 — SNN low-drive interface → how the per-contact envelope is taken

- **Operating point LOCKED**: `nu_ext_ratio = 0.6` (interictal quiet-excitable; `Params(nu_ext_ratio=0.6)`). Default 1.0 = oscillatory/seizure-ward — NOT used. `T` (duration) is set on `Params` (no `t_max` arg); `dt=0.1 ms`.
- **Envelope source LOCKED = smoothed E-spike density (path A, zero current-recording)**. The kick path `fresh_run`/`simulate_kick` saves `E_spk_bool (nsteps, NE)` aligned to `posE = net['pos'][:NE]`; **per-neuron synaptic currents are never stored**, and the kick loop does not wire `LFPRecorder`. So: bin+smooth `E_spk_bool` to a per-neuron rate field, then sample at montage contacts with the Increment-1 Gaussian footprint (`sample_envelopes(source_frames, grid_xy=posE, montage, kernel_width)`). This **reuses Increment-1 verbatim** and needs **zero engine edits** for the envelope.
  - **Why path A, not the current-LFP power-law (path B)**: path B (lfp.py `|I_E|+|I_I|`, power-law `f(r)`) is the more literal "synaptic-current proxy" the spec mentions, but it requires *extending the engine kick loop to record currents per step* + generalizing `make_grid` to montage coords — an engine edit to gitignored coworker code. For the **direction discriminator**, the read-out uses rank **ORDER** only; a discrete-contact, spatially-integrated, real-pipeline read of smoothed spike density carries that order faithfully. Path B (exact current-LFP waveform) is recorded as a **deferred fidelity upgrade** (needed only when we later match real-data event *waveforms*, not for the order-based direction test). **REVIEW HOOK: if you want the literal current-LFP observable now, say so and we bump to path B (+ engine edit + lfp.py `sites=` generalization).**
  - **Reporting boundary (LOCKED)**: path A reads a **firing-density envelope**, NOT a synaptic-current LFP. It validates the **direction-ORDER read-out** only. It may **NOT** be reported as "the LFP observation layer is validated"; any claim about LFP *waveform* amplitude/shape (e.g. matching real-data event envelopes) requires path B. Increment-2's verdict is strictly "template direction tracks θ_EE through virtual contacts + real pipeline", nothing about waveform.
- **Oracle parity (not a gate)**: keep `anisotropy_front.principal_axis` (true-position + spike-onset PCA; already PASSES err<1.2°, iso ratio 1.13). The virtual-electrode read should agree in *direction* (track θ_EE); it may be noisier. Report both; the oracle is the upper bound, the virtual read is the observable being validated.

### Item 2 — single-end kick + event window

- **Single-end kick LOCKED**: the engine kick disk is hardcoded to sheet center `[L/2, L/2]` (`kick_probe.py:93`). Add a **backward-compatible** `kick_center` kwarg (default `[L/2, L/2]` → identical to today). Place the kick at one END of the θ_EE axis: `kick_center = [L/2, L/2] + r_end · (cos θ_EE, sin θ_EE)`, `r_end` **(proposed 0.6·L/2)** so the `R_KICK=0.15 mm` disk sits inside the sheet and the wave transits the montage. Kick params `R_KICK=0.15`, `T_KICK=150 ms`, `DUR_KICK=18 ms`, `KICK_BOOST=2·nu_theta` unchanged.
  - **Engine-patch caveat**: the engine lives under `results/topic4_sef_hfo/lif_snn/engine/` which is **gitignored** (coworker code). The `kick_center` kwarg is a documented additive patch (Task 1) applied to the working copy; the exact diff is recorded in this plan for reproducibility since it can't be committed here.
- **Event window LOCKED = `detect_events` on the aggregate envelope with a recalibrated per-operating-point bar.** The kick is 18 ms but the wave transits ~150 ms, so a fixed short window truncates the front (engine currently uses fixed `PEAK_WINDOW=(150,300)` — rejected). Aggregate activity = mean over montage contacts of the per-contact envelope. Bar via `calibrate_detector(ref_series=[KICK_BOOST=0 run aggregate], kick_series=[KICK_BOOST run aggregate])` → `event_on_frac_from_refs` (σ-floor + kick-peak midpoint, anti-circular). `detect_events` shape constants (`MIN_DUR_MS=8`, `MERGE_GAP_MS=12`, `RETURN_FRAC=0.2`, `SETTLE_MS=50`) unchanged. Emit `(t_on, t_off)` for `extract_lagpat`.

### Item 3 — 2D montage: placement + rotation

- **LOCKED**: ≥2 non-parallel shafts via Increment-1 `build_shaft` + `merge_montages`, **centered at sheet center `(L/2, L/2)`**, in the engine's mm frame (`pos` is mm on `[0,L]²`). Sheet is small (`L=2–3 mm`), so contact pitch **(proposed 0.3 mm)**, `n_contacts` per shaft = **8 (EVEN — locked, not 7)** → 16 contacts spanning the sheet, well above the ≥7 participation gate. Two shafts at **(proposed 10° and 100°)**, both through the center. **Even count is required** so neither shaft has a contact exactly at the shared center → no coincident contact (two contacts at the same coord = duplicate signal + spurious ties; this is the exact Increment-1 lesson, build_shaft centers offsets at `±pitch·(2k-1)/2`, so even-n skips the origin).
- **Rotate-montage control**: rebuild all shafts at `+Δ` (rotate the whole array); θ_EE fixed → recovered direction must be invariant.
- **Angle convention LOCKED = `atan2(dy, dx) mod 180°`** (x-axis = 0°), matching the engine oracle (verified: rot θ=0→0°, θ=90→90°). The virtual read MUST use this or the vs-θ_EE comparison is meaningless.

### Item 4 — the three hard controls + how each is judged

| Control | Setup | Judge (Increment-1 estimators) | PASS criterion |
|---|---|---|---|
| **C-track: direction tracks connectivity** | `build_connectivity_rot(theta_EE=deg, AR=2.0)` for θ_EE ∈ {0,45,90}; montage FIXED | recovered `endpoint_centroid_axis` vs θ_EE via `axis_angle_error_deg` (mod 180°), over **direction-readable** seeds | mean axis error **< 25°** (proposed) for each θ_EE |
| **shaft-invariance** | θ_EE **FIXED at 45°**; rotate the whole 2D montage by Δ ∈ {0,30,60} | recovered axis vs the **fixed θ_EE (45°)** — NOT a difference between rotations | **each** montage rotation's mean axis error **< 25°** vs the fixed 45° (proposed). This directly encodes "recovered axis stays at θ_EE, invariant to electrode rotation"; comparing *differences* of errors is rejected (all-rotations-60°-off would falsely pass). |
| **must-fail: isotropic + aligned shaft** | `build_connectivity_rot(AR=1.0)` (isotropic; no separate iso flag) + shaft aligned to a reference axis | judged on **event-valid** seeds (event exists + ≥7 contacts participate); **does NOT require a direction axis** — "no readable axis" is the desired outcome. `direction_readability` (max-axis Spearman; NaN-when-fully-tied → treated as 0 = unreadable) | mean `direction_readability` over event-valid seeds **< τ_fail = 0.3** (proposed) — events occur but NO stable readable direction |

- `build_connectivity_rot` keeps E→I/I→E/I→I bit-identical and **mass fixed by `C_EE=800` (without replacement) + constant `w_EE`** — Increment 2 introduces **no** kernel/distance weight scaling (pathology-mapping spec normalization contract; already satisfied). `AR=2.0 ↔ rho_EE=0.6` (≈45°/ratio1.85, empirical match). Pre-verify any (θ,AR) offline via `connectivity_rot._partner_cov_axis`.
- **Grid/montage-contamination guard** (reader-C pitfall): C-track + shaft-invariance *together* prove the recovered axis follows connectivity, not the montage's own geometry. Both required.

### Item 5 — two-layer seed validity (event-valid vs direction-readable) + counts

Two **orthogonal** per-seed flags (the iso must-fail breaks without the split):
- **event-valid** = a self-limited event window was found **AND** `n_participating ≥ 2·k_dir+1 = 7` (k_dir=3 lock). "There was a real, well-sampled event."
- **direction-readable** = event-valid **AND** `endpoint_centroid_axis` returns a non-degenerate axis. "A direction could be read off it." For isotropic, the *desired* outcome is **event-valid but NOT direction-readable**.

Per condition, run `n_seed` **(proposed 3)** seeds:
- **Tracking conditions** (C-track, shaft-invariance): need **≥ `n_seed_min = 2`** (proposed) **direction-readable** seeds, else INSUFFICIENT (loud, never forced). Verdict = mean axis-error over readable seeds < 25°.
- **Must-fail condition** (iso): need **≥ `n_seed_min` event-valid** seeds (so we know events DID occur), else INSUFFICIENT. Verdict = mean `direction_readability` over those event-valid seeds < τ_fail. **A condition where events don't even occur (event-valid < n_seed_min) is INSUFFICIENT, NOT a must-fail PASS** — "no events" must never be read as "events with no readable direction".
- Report mean ± sd across seeds; mirrors the oracle's 3-seed protocol.

**Observation-knob freeze (LOCKED discipline):** the read-out geometry/smoothing knobs — `kernel_width`, `bin_ms`, `smooth_ms`, montage `pitch`/`n_contacts`, `R_END_FRAC` — are calibrated ONCE in the Task-5 smoke (on a SINGLE condition, e.g. θ_EE=45°), then **frozen and recorded in verdict.json BEFORE** the formal three-control run. They are read-geometry, not verdict bars; tuning them after seeing the three-control results = tuning the virtual electrode to the answer (forbidden). The verdict thresholds (25° / τ_fail / k_dir / n_seed_min) are never tuned at all.

> **Acceptance gate (encode the conclusion, not existence):** Increment-2 PASS = C-track each θ_EE mean-err < 25° (over ≥ n_seed_min direction-readable seeds) **AND** shaft-invariance each montage rotation mean-err < 25° vs the **fixed** θ_EE=45° **AND** iso must-fail mean readability < τ_fail over ≥ n_seed_min **event-valid** seeds. Any condition below its seed floor (tracking: direction-readable; iso: event-valid) is **INSUFFICIENT** — surfaced, never silently dropped, and "no events" is never a must-fail PASS. Thresholds lock at review; never tuned to pass.

---

## File structure

- **Create** `src/sef_hfo_snn_adapter.py` — pure functions: bin+smooth `E_spk_bool` → per-neuron rate field; reuse `sample_envelopes(grid_xy=posE)`; build the aggregate series. (tracked, unit-testable with synthetic spikes)
- **Patch** `results/topic4_sef_hfo/lif_snn/engine/kick_probe.py` — add `kick_center` kwarg (gitignored; diff documented in Task 1).
- **Create** `scripts/run_sef_hfo_obs_increment2.py` — the SNN discriminator runner (build_network_rot × θ_EE/iso × seeds → adapter → extract_lagpat → direction; verdict JSON + figure + README + oracle parity).
- **Test** `tests/test_sef_hfo_snn_adapter.py` — synthetic-spike-front unit tests (fast, no real SNN).
- **Reuse unchanged**: `src/sef_hfo_observation.py` (montage/sampler/extractor/estimators), `src/sef_hfo_events.py` (detect_events/calibrate_detector).

---

### Task 1: Engine patch — `kick_center` kwarg (single-end kick), backward-compatible

**Files:** Patch `results/topic4_sef_hfo/lif_snn/engine/kick_probe.py` (gitignored — diff recorded here).

- [ ] **Step 1: Apply the additive patch** (the kick disk currently centers on `[L/2,L/2]` at kick_probe.py:~93). Change the two functions' signatures + the `kick_mask` center:

```python
# simulate_kick(...) and fresh_run(...) gain a kwarg:  kick_center=None
# inside simulate_kick, where the kick_mask is built:
center = np.array([p.L / 2, p.L / 2]) if kick_center is None else np.asarray(kick_center, float)
dist_c = np.linalg.norm(pos - center, axis=1)
kick_mask = is_E & (dist_c <= R_KICK)
# fresh_run forwards kick_center: return simulate_kick(p, net, KICK_BOOST, kick_center=kick_center, ...)
```

Default `None` → `[L/2,L/2]` → byte-identical to today (backward-compatible; existing kick scripts unaffected).

- [ ] **Step 2: Fast regression test of the engine** (tiny net, ~seconds):

Run a tiny `Params(L=1.0, density=400.0, T=200.0, nu_ext_ratio=0.6, seed=1)` build + `fresh_run(p, net, KICK_BOOST=2*nu_theta, kick_center=[0.8,0.5])`; assert the inside-disk E neurons are those within `R_KICK` of `[0.8,0.5]` (read off `E_spk_bool` peak-activity centroid early in the kick window ≈ `[0.8,0.5]`, not `[0.5,0.5]`). Also assert `kick_center=None` reproduces the central-disk behavior (centroid ≈ `[0.5,0.5]`).

- [ ] **Step 3: Record the diff** in this plan + a `engine_patches/kick_center.patch` note under `scripts/` (since the engine isn't git-tracked here). Commit only the tracked patch-note + test:

```bash
git add scripts/engine_patches/kick_center.patch tests/test_sef_hfo_snn_adapter.py
git commit -m "feat(topic4 obs): engine kick_center kwarg (single-end kick) + patch note"
```

---

### Task 2: SNN envelope adapter (`spikes → per-contact envelope`), unit-tested on a synthetic spike front

**Files:** Create `src/sef_hfo_snn_adapter.py`; Test `tests/test_sef_hfo_snn_adapter.py`.

- [ ] **Step 1: Write the failing test** — a synthetic spike front sweeping along a known direction must read back correctly through the adapter + Increment-1 chain (NO real SNN):

```python
# tests/test_sef_hfo_snn_adapter.py
"""TDD for src/sef_hfo_snn_adapter — Increment 2 spike->envelope adapter.
Synthetic spike fronts (no real SNN) verify the adapter feeds the Increment-1
observation chain so direction reads back. Real-SNN runs live in the runner."""
import numpy as np

from src.sef_hfo_snn_adapter import snn_event_envelope
from src.sef_hfo_observation import (
    build_shaft, merge_montages, extract_lagpat, attach_geometry,
    rank_vs_projection_spearman,
)


def _synthetic_front(n_neuron=2000, L=2.0, angle_rad=np.deg2rad(30.0), dt=0.1,
                     t_max=200.0, c=0.01, width_ms=15.0, seed=0):
    """E neurons uniform on [0,L]^2; each fires a short burst whose onset time
    increases with projection along angle_rad (a traveling spike front)."""
    rng = np.random.default_rng(seed)
    posE = rng.uniform(0, L, size=(n_neuron, 2))
    n_hat = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    s = posE @ n_hat
    onset = 50.0 + (s - s.min()) / c          # ms, increases along n_hat
    nsteps = int(t_max / dt)
    t = np.arange(nsteps) * dt
    spk = np.zeros((nsteps, n_neuron), bool)
    for j in range(n_neuron):                  # 1 spike at each neuron's onset (+jitter)
        k = int((onset[j] + rng.normal(0, 1.0)) / dt)
        if 0 <= k < nsteps:
            spk[k, j] = True
    return spk, posE, n_hat, dt


def test_adapter_envelope_preserves_front_direction():
    spk, posE, n_hat, dt = _synthetic_front()
    montage = merge_montages([build_shaft(np.deg2rad(10.0), 0.3, 7, (1.0, 1.0), "A"),
                              build_shaft(np.deg2rad(100.0), 0.3, 7, (1.0, 1.0), "B")])
    env, frame_dt, agg = snn_event_envelope(spk, posE, montage, dt,
                                            bin_ms=2.0, smooth_ms=5.0, kernel_width=0.25)
    assert env.shape[0] == len(montage.names)
    # whole synthetic window is one event
    art = extract_lagpat(env, frame_dt, event_windows=[(0.0, env.shape[1] * frame_dt)],
                         participation_floor=float(env.min()),
                         participation_margin=0.5 * (float(env.max()) - float(env.min())),
                         timing_frac=0.5, tie_tol=frame_dt)
    art = attach_geometry(art, montage)
    rho = rank_vs_projection_spearman(art.ranks[:, 0], art.bools[:, 0],
                                      art.contact_coords, n_hat)
    assert rho >= 0.9           # the adapter carries the front order into the read-out
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest tests/test_sef_hfo_snn_adapter.py -q` → `ModuleNotFoundError: src.sef_hfo_snn_adapter`.

- [ ] **Step 3: Implement the adapter**:

```python
# src/sef_hfo_snn_adapter.py
"""SNN -> observation adapter (Increment 2). Pure functions: bin + temporally smooth
per-neuron E spikes into a rate field, then sample at virtual-electrode contacts using
the Increment-1 Gaussian footprint. The ONLY model-specific piece; everything downstream
(extract_lagpat -> real pipeline -> direction estimators) is the shared Increment-1 chain.
"""
from __future__ import annotations

import numpy as np

from src.sef_hfo_observation import sample_envelopes


def _bin_and_smooth(E_spk_bool, dt, bin_ms, smooth_ms):
    """(nsteps, NE) bool -> (n_frame, NE) smoothed per-neuron rate (Gaussian in time)."""
    nsteps, NE = E_spk_bool.shape
    bin_steps = max(1, int(round(bin_ms / dt)))
    n_frame = nsteps // bin_steps
    binned = (E_spk_bool[: n_frame * bin_steps]
              .reshape(n_frame, bin_steps, NE).sum(axis=1).astype(float))  # spikes/bin
    # Gaussian temporal smoothing (sigma in frames); separable, per neuron
    sig = max(1e-6, smooth_ms / bin_ms)
    half = int(np.ceil(3 * sig))
    x = np.arange(-half, half + 1)
    k = np.exp(-(x ** 2) / (2 * sig ** 2)); k /= k.sum()
    sm = np.apply_along_axis(lambda col: np.convolve(col, k, mode="same"), 0, binned)
    return sm, bin_ms


def snn_event_envelope(E_spk_bool, posE, montage, dt, bin_ms=2.0, smooth_ms=5.0,
                       kernel_width=0.25):
    """Per-contact activity envelope from per-neuron E spikes.

    Returns (envelopes (n_contact, n_frame), frame_dt_ms, aggregate (n_frame,)).
    Reuses Increment-1 sample_envelopes with grid_xy = posE (E-neuron coords) and
    source_frames = smoothed per-neuron rate. aggregate = mean over contacts (feeds
    detect_events for the event window)."""
    rate, frame_dt = _bin_and_smooth(np.asarray(E_spk_bool, bool), dt, bin_ms, smooth_ms)
    env = sample_envelopes(rate, np.asarray(posE, float), montage, kernel_width)
    aggregate = env.mean(axis=0)
    return env, frame_dt, aggregate
```

- [ ] **Step 4: Run to verify it passes** — `python -m pytest tests/test_sef_hfo_snn_adapter.py -q` → PASS. If the Spearman < 0.9, debug smoothing/kernel_width (a sharp front + too-wide footprint washes order) — do NOT relax the 0.9.

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_snn_adapter.py tests/test_sef_hfo_snn_adapter.py
git commit -m "feat(topic4 obs): SNN spike->envelope adapter (reuses Increment-1 chain; synthetic-front unit test)"
```

---

### Task 3: Event window from the aggregate via `detect_events` + per-op recalibration

**Files:** Modify `scripts/run_sef_hfo_obs_increment2.py` (helper) ; Test `tests/test_sef_hfo_snn_adapter.py`.

- [ ] **Step 1: Write the failing test** — a synthetic aggregate (quiet → bump → quiet) yields exactly one self-terminating window via the recalibrated bar:

```python
# append to tests/test_sef_hfo_snn_adapter.py
from src.sef_hfo_events import calibrate_detector, detect_events


def test_event_window_from_recalibrated_aggregate():
    frame_dt = 2.0
    n = 150
    t = np.arange(n) * frame_dt
    ref = 0.01 * np.abs(np.sin(t))                 # quiet no-kick reference (KICK_BOOST=0)
    kick = ref.copy()
    bump = np.exp(-((t - 150) ** 2) / (2 * 20.0 ** 2))   # one event ~150ms, returns
    kick = kick + bump
    cal = calibrate_detector([ref], kick, frac=0.5)       # σ-floor + kick-peak midpoint
    evs = detect_events(kick, frame_dt, event_on_frac=cal["event_on_frac"])
    assert len(evs) == 1
    assert evs[0]["returned"] is True                      # self-terminated
    assert 100 < evs[0]["t_on"] < 175
```

- [ ] **Step 2: Run to verify it fails** if the helper that wires this is missing; otherwise this test exercises the locked `calibrate_detector`/`detect_events` directly (they already exist) — it documents the Item-2 window contract. Run: `python -m pytest tests/test_sef_hfo_snn_adapter.py::test_event_window_from_recalibrated_aggregate -q`.

- [ ] **Step 3: Implement the runner helper** `event_window_for_run(agg_kick, agg_ref, frame_dt)` in `scripts/run_sef_hfo_obs_increment2.py`:

```python
from src.sef_hfo_events import calibrate_detector, detect_events

def event_window_for_run(agg_kick, agg_ref, frame_dt):
    """Per-operating-point event window (Item 2): recalibrate the bar from the no-kick
    reference + this kick run, then detect the single self-limited event. Returns
    (t_on, t_off) or None if no self-terminating event (INSUFFICIENT)."""
    cal = calibrate_detector([agg_ref], agg_kick, frac=0.5)
    evs = [e for e in detect_events(agg_kick, frame_dt, event_on_frac=cal["event_on_frac"])
           if e["returned"]]
    if not evs:
        return None
    e = max(evs, key=lambda d: d["peak_ext"])      # the kick-evoked event
    return (e["t_on"], e["t_off"])
```

- [ ] **Step 4: Run** the test → PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_sef_hfo_obs_increment2.py tests/test_sef_hfo_snn_adapter.py
git commit -m "feat(topic4 obs): event window from recalibrated aggregate (detect_events + calibrate_detector)"
```

---

### Task 4: SNN discriminator runner (C-track + shaft-invariance + iso must-fail + oracle parity)

**Files:** `scripts/run_sef_hfo_obs_increment2.py` (main); `results/topic4_sef_hfo/observation_layer/increment2_snn_slice/` (outputs); `.../figures/README.md`.

- [ ] **Step 1: Write the runner** (orchestration only — every numeric judgement reuses Increment-1 estimators). Sketch with the exact engine calls from the map:

```python
# scripts/run_sef_hfo_obs_increment2.py  (main; helpers from Task 3 above)
import sys, json
from pathlib import Path
import numpy as np

ENGINE = Path("results/topic4_sef_hfo/lif_snn/engine")
sys.path.insert(0, str(ENGINE))
from params import Params, compute_nu_theta          # noqa: E402
from connectivity import place_neurons                # noqa: E402
from connectivity_rot import build_connectivity_rot   # noqa: E402
from kick_probe import simulate_kick, R_KICK          # noqa: E402 (patched: kick_center kwarg)

from src.sef_hfo_observation import (build_shaft, merge_montages, extract_lagpat,
    attach_geometry, endpoint_centroid_axis, direction_readability, axis_angle_error_deg)
from src.sef_hfo_snn_adapter import snn_event_envelope

# LOCKED params (§0)
L, DENSITY, T, DT = 2.0, 4000.0, 350.0, 0.1
DRIVE = 0.6
PITCH, NC, SHAFTS = 0.3, 8, (10.0, 100.0)   # NC EVEN -> no center coincident contact (Item 3)
KDIR, NSEED, NSEED_MIN = 3, 3, 2
AXIS_ERR_MAX, TAU_FAIL = 25.0, 0.3
R_END_FRAC = 0.6
OUT = Path("results/topic4_sef_hfo/observation_layer/increment2_snn_slice")


def _montage(rot_deg=0.0, center=(L/2, L/2)):
    a = build_shaft(np.deg2rad(SHAFTS[0] + rot_deg), PITCH, NC, center, "A")
    b = build_shaft(np.deg2rad(SHAFTS[1] + rot_deg), PITCH, NC, center, "B")
    return merge_montages([a, b])


def _build_net(p, theta_EE, AR, rng):
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta_EE, AR=AR)
    return net, NE


def _read_one(theta_EE, AR, seed, montage):
    """One seed: build net, single-end kick at the θ_EE axis end, read direction."""
    p = Params(L=L, density=DENSITY, T=T, dt=DT, nu_ext_ratio=DRIVE, seed=seed)
    rng = np.random.default_rng(seed)
    net, NE = _build_net(p, theta_EE, AR, rng)
    posE = net["pos"][:NE]
    nu_theta = compute_nu_theta(p)[0]
    end = np.array([L/2, L/2]) + R_END_FRAC * (L/2) * np.array([np.cos(theta_EE), np.sin(theta_EE)])
    net["rng"] = np.random.default_rng(seed)
    res_kick = simulate_kick(p, net, KICK_BOOST=2*nu_theta, kick_center=end)
    net["rng"] = np.random.default_rng(seed)
    res_ref = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=end)
    env_k, fdt, agg_k = snn_event_envelope(res_kick["E_spk_bool"], posE, montage, DT)
    _, _, agg_r = snn_event_envelope(res_ref["E_spk_bool"], posE, montage, DT)
    win = event_window_for_run(agg_k, agg_r, fdt)        # Task 3 helper
    if win is None:
        return {"event_valid": False, "direction_readable": False,
                "reason": "no_self_limited_event"}
    art = extract_lagpat(env_k, fdt, event_windows=[win],
                         participation_floor=float(env_k.min()),
                         participation_margin=0.5*(float(env_k.max())-float(env_k.min())),
                         timing_frac=0.5, tie_tol=fdt)
    art = attach_geometry(art, montage)
    r0, b0 = art.ranks[:, 0], art.bools[:, 0]
    n_part = int(b0.sum())
    axis = endpoint_centroid_axis(r0, b0, art.contact_coords, k_dir=KDIR, eps_deg=0.5*PITCH)
    read = direction_readability(r0, b0, art.contact_coords)
    event_valid = n_part >= 2*KDIR+1                      # event exists + enough contacts (win!=None here)
    return {"event_valid": bool(event_valid),
            "direction_readable": bool(event_valid and axis is not None),
            "n_part": n_part, "axis": None if axis is None else axis.tolist(),
            "readability": float(read)}                   # may be nan (fully tied = unreadable)


def _tracking_condition(theta_EE, AR, montage):
    """C-track / shaft-invariance: recovered axis must TRACK theta_EE.
    Needs >= n_seed_min DIRECTION-READABLE seeds."""
    seeds = [_read_one(theta_EE, AR, s, montage) for s in range(1, NSEED+1)]
    readable = [s for s in seeds if s.get("direction_readable")]
    if len(readable) < NSEED_MIN:
        return {"status": "INSUFFICIENT", "n_readable": len(readable),
                "n_event_valid": sum(bool(s.get("event_valid")) for s in seeds), "seeds": seeds}
    errs = [axis_angle_error_deg(np.array(s["axis"]), theta_EE) for s in readable]
    mean_err = float(np.mean(errs))
    return {"status": "OK", "n_readable": len(readable),
            "axis_err_mean": mean_err, "axis_err_sd": float(np.std(errs)),
            "pass": bool(mean_err < AXIS_ERR_MAX), "seeds": seeds}


def _mustfail_condition(theta_EE, AR, montage):
    """Isotropic must-fail: events must OCCUR (event-valid) but direction must NOT be
    readable. Judged on event-valid seeds' readability; 'no events' = INSUFFICIENT, NOT pass."""
    seeds = [_read_one(theta_EE, AR, s, montage) for s in range(1, NSEED+1)]
    ev = [s for s in seeds if s.get("event_valid")]
    if len(ev) < NSEED_MIN:
        return {"status": "INSUFFICIENT", "n_event_valid": len(ev), "seeds": seeds}
    reads = [0.0 if s["readability"] != s["readability"] else s["readability"] for s in ev]  # NaN->0
    mean_read = float(np.mean(reads))
    return {"status": "OK", "n_event_valid": len(ev), "readability_mean": mean_read,
            "pass": bool(mean_read < TAU_FAIL), "seeds": seeds}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT.parent / "figures").mkdir(parents=True, exist_ok=True)
    verdict = {"locked": {"drive": DRIVE, "axis_err_max": AXIS_ERR_MAX, "tau_fail": TAU_FAIL,
                          "k_dir": KDIR, "n_seed": NSEED, "n_seed_min": NSEED_MIN,
                          "obs_knobs": {"pitch": PITCH, "n_contacts": NC, "shafts": SHAFTS,
                                        "r_end_frac": R_END_FRAC}}}  # frozen after smoke (Item 5)
    m0 = _montage()
    # C-track: rotate connectivity, montage FIXED -> axis must track theta_EE
    verdict["C_track"] = {f"{d}deg": _tracking_condition(np.deg2rad(d), 2.0, m0) for d in (0, 45, 90)}
    # shaft-invariance: connectivity FIXED at 45deg, rotate montage -> axis must STILL be 45deg
    verdict["shaft_invariance"] = {f"rot{r}": _tracking_condition(np.deg2rad(45.0), 2.0, _montage(rot_deg=r))
                                   for r in (0, 30, 60)}
    # must-fail: isotropic (AR=1) + aligned shaft -> events occur but NO readable direction
    verdict["iso_mustfail"] = _mustfail_condition(np.deg2rad(0.0), 1.0, m0)

    def ok(cond): return cond.get("status") == "OK"
    ct, si, iso = verdict["C_track"], verdict["shaft_invariance"], verdict["iso_mustfail"]
    # each tracking condition must itself track theta_EE within 25deg (NOT a diff-of-errors)
    c_track_pass = all(ok(ct[k]) and ct[k]["pass"] for k in ct)
    si_pass = all(ok(si[k]) and si[k]["pass"] for k in si)   # each montage rot: axis_err<25 vs fixed 45
    iso_pass = ok(iso) and iso["pass"]
    verdict["GATE_PASS"] = bool(c_track_pass and si_pass and iso_pass)
    verdict["insufficient"] = [k for d in (ct, si) for k in d if not ok(d[k])] + ([] if ok(iso) else ["iso"])
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=lambda o: None))
    print("GATE_PASS =", verdict["GATE_PASS"], "| insufficient:", verdict["insufficient"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Figure** — a 3-panel summary: (i) C-track axis-error vs θ_EE (bars + 25° line), (ii) shaft-invariance recovered-axis vs montage rotation (should be flat), (iii) iso readability vs τ_fail line. Plus an oracle-parity panel (Task 6). Save `figures/increment2_snn.png`.

- [ ] **Step 3: Commit the runner** (no run yet):

```bash
git add scripts/run_sef_hfo_obs_increment2.py
git commit -m "feat(topic4 obs): Increment-2 SNN discriminator runner (C-track + shaft-invariance + iso must-fail)"
```

---

### Task 5: Small-scale SNN smoke (fast config) — confirm the whole chain runs end-to-end

**Files:** none new (a smoke invocation + assertion).

- [ ] **Step 1: Run a fast smoke** — temporarily `L=1.0, density=1000.0, T=250.0, NSEED=1` (tiny net ~1000 neurons, seconds) for one θ_EE=45° condition. Confirm: `build_connectivity_rot` runs, `simulate_kick(kick_center=end)` returns `E_spk_bool`, the adapter yields a non-degenerate envelope, `event_window_for_run` finds a self-limited window, and `_read_one` returns `valid=True` with `n_part ≥ 7`. This de-risks the full run.

- [ ] **Step 2: If the smoke's window is None or n_part < 7** — diagnose (kick too weak / montage off the front / smoothing): adjust the observation knobs `R_END_FRAC / kernel_width / bin_ms / smooth_ms / pitch / n_contacts` (NOT the verdict bars τ_fail / 25° / k_dir / n_seed_min). Tune on the smoke's SINGLE condition (θ_EE=45°) only.

- [ ] **Step 3: FREEZE the observation knobs (Item-5 discipline).** Once the smoke gives a self-limited window + n_part ≥ 7, RECORD the final knob values into the runner constants + `verdict.json::locked.obs_knobs`, and DO NOT change them again. The formal three-control run (Task 6) uses these frozen values — tuning knobs after seeing the three-control results = tuning the electrode to the answer (forbidden). The smoke config itself is throwaway (no commit), but the frozen knob values ARE committed as the runner constants.

---

### Task 6: Full discriminator run + oracle parity + verdict + README + eyeball

**Files:** `results/topic4_sef_hfo/observation_layer/increment2_snn_slice/`, `.../figures/README.md`.

- [ ] **Step 1: Run** `PYTHONPATH=. python scripts/run_sef_hfo_obs_increment2.py` at locked params. Capture `GATE_PASS` + `insufficient`.

- [ ] **Step 2: Oracle parity** — for one θ_EE (e.g. 45°, one seed), also run the engine's `anisotropy_front.principal_axis` on the same `E_spk_bool` (source-space oracle) and record its axis-error. Assert the virtual-electrode recovered axis agrees in direction (both track θ_EE; virtual may be noisier). Add to verdict.json as `oracle_parity`.

- [ ] **Step 3: Eyeball the figure** — confirm C-track bars under 25°, shaft-invariance flat, iso readability under τ_fail. If a condition is INSUFFICIENT, that is a legitimate honest outcome (report it), NOT a reason to loosen thresholds or hand-pick seeds.

- [ ] **Step 4: Write `figures/README.md`** (Chinese, per AGENTS.md) describing the three controls + the oracle-parity panel + 关注点.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_sef_hfo_obs_increment2.py
git commit -m "feat(topic4 obs): Increment-2 full SNN discriminator run + oracle parity + verdict"
```

---

## Self-Review notes (for the implementer)

- **TDD reality:** the real SNN is slow (~minutes at paper scale), so the *direction logic* is unit-tested on a synthetic spike front (Task 2) and the *event-window* on a synthetic aggregate (Task 3); the real SNN enters only in the runner (Tasks 5–6) where a smoke de-risks before the full run. This matches the engine's cost (cannot fast-unit-test a 16k-neuron sim).
- **Reused, not reinvented:** montage / sample_envelopes / extract_lagpat / endpoint_centroid_axis / direction_readability / axis_angle_error_deg / detect_events / calibrate_detector are all Increment-1 / `sef_hfo_events` — Increment 2 adds only the spike→envelope adapter + the `kick_center` kwarg + the runner.
- **Thresholds locked, never tuned:** 25° / τ_fail=0.3 / k_dir=3 / n_seed_min=2. Observation knobs (`R_END_FRAC`, `kernel_width`, `bin_ms`, `smooth_ms`, pitch/n_contacts) MAY be calibrated in the Task-5 smoke (single condition) then are **FROZEN + recorded before the formal run** (Item 5) — read geometry, not the verdict bar, and not tunable to the three-control results.
- **must-fail logic (the subtle one):** isotropic is judged on **event-valid** seeds' `direction_readability < τ_fail`, NOT on having a valid axis — "no readable axis" is the desired PASS. A condition with too few events is **INSUFFICIENT**, never a must-fail PASS. event-valid (event + ≥7 contacts) and direction-readable (axis exists) are separate per-seed flags.
- **shaft-invariance:** each montage rotation must itself read within 25° of the **fixed** θ_EE=45° — not a difference-of-errors (which would let "all rotations 60° off" falsely pass).
- **Oracle ≠ gate:** `anisotropy_front` is the upper-bound parity reference, not the acceptance criterion. The gate is the virtual-electrode + real-pipeline read.
- **Open decisions for the user (review §0):** (1) path A (spike-density envelope, locked) vs path B (current-LFP forward model, deferred) — bump to B if you want the literal observable; (2) all **(proposed)** numbers; (3) the engine `kick_center` patch touches gitignored coworker code — acceptable, or prefer a tracked wrapper?
