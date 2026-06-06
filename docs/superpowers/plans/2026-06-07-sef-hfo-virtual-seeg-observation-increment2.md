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
- **Oracle parity (not a gate)**: keep `anisotropy_front.principal_axis` (true-position + spike-onset PCA; already PASSES err<1.2°, iso ratio 1.13). The virtual-electrode read should agree in *direction* (track θ_EE); it may be noisier. Report both; the oracle is the upper bound, the virtual read is the observable being validated.

### Item 2 — single-end kick + event window

- **Single-end kick LOCKED**: the engine kick disk is hardcoded to sheet center `[L/2, L/2]` (`kick_probe.py:93`). Add a **backward-compatible** `kick_center` kwarg (default `[L/2, L/2]` → identical to today). Place the kick at one END of the θ_EE axis: `kick_center = [L/2, L/2] + r_end · (cos θ_EE, sin θ_EE)`, `r_end` **(proposed 0.6·L/2)** so the `R_KICK=0.15 mm` disk sits inside the sheet and the wave transits the montage. Kick params `R_KICK=0.15`, `T_KICK=150 ms`, `DUR_KICK=18 ms`, `KICK_BOOST=2·nu_theta` unchanged.
  - **Engine-patch caveat**: the engine lives under `results/topic4_sef_hfo/lif_snn/engine/` which is **gitignored** (coworker code). The `kick_center` kwarg is a documented additive patch (Task 1) applied to the working copy; the exact diff is recorded in this plan for reproducibility since it can't be committed here.
- **Event window LOCKED = `detect_events` on the aggregate envelope with a recalibrated per-operating-point bar.** The kick is 18 ms but the wave transits ~150 ms, so a fixed short window truncates the front (engine currently uses fixed `PEAK_WINDOW=(150,300)` — rejected). Aggregate activity = mean over montage contacts of the per-contact envelope. Bar via `calibrate_detector(ref_series=[KICK_BOOST=0 run aggregate], kick_series=[KICK_BOOST run aggregate])` → `event_on_frac_from_refs` (σ-floor + kick-peak midpoint, anti-circular). `detect_events` shape constants (`MIN_DUR_MS=8`, `MERGE_GAP_MS=12`, `RETURN_FRAC=0.2`, `SETTLE_MS=50`) unchanged. Emit `(t_on, t_off)` for `extract_lagpat`.

### Item 3 — 2D montage: placement + rotation

- **LOCKED**: ≥2 non-parallel shafts via Increment-1 `build_shaft` + `merge_montages`, **centered at sheet center `(L/2, L/2)`**, in the engine's mm frame (`pos` is mm on `[0,L]²`). Sheet is small (`L=2–3 mm`), so contact pitch **(proposed 0.3 mm)**, `n_contacts` per shaft **(proposed 7)** → 14 contacts spanning the sheet, comfortably above the ≥7 participation gate. Two shafts at **(proposed 10° and 100°)** (non-parallel, span 2D, even count avoids an exact-origin coincident contact — see Increment-1).
- **Rotate-montage control**: rebuild all shafts at `+Δ` (rotate the whole array); θ_EE fixed → recovered direction must be invariant.
- **Angle convention LOCKED = `atan2(dy, dx) mod 180°`** (x-axis = 0°), matching the engine oracle (verified: rot θ=0→0°, θ=90→90°). The virtual read MUST use this or the vs-θ_EE comparison is meaningless.

### Item 4 — the three hard controls + how each is judged

| Control | Setup | Judge (Increment-1 estimators) | PASS criterion |
|---|---|---|---|
| **C-track: direction tracks connectivity** | `build_connectivity_rot(theta_EE=deg, AR=2.0)` for θ_EE ∈ {0,45,90}; montage FIXED | recovered `endpoint_centroid_axis` vs θ_EE via `axis_angle_error_deg` (mod 180°) | mean axis error **< 25°** (proposed) for each θ_EE |
| **shaft-invariance** | θ_EE FIXED; rotate the whole 2D montage by Δ ∈ {0,30,60} | recovered axis vs the θ_EE=fixed reference | axis shift **< 25°** (proposed) across montage rotations (direction follows connectivity, not electrodes) |
| **must-fail: isotropic + aligned shaft** | `build_connectivity_rot(AR=1.0)` (isotropic; no separate iso flag) + shaft aligned to a reference axis | `direction_readability` (max-axis Spearman) | **< τ_fail = 0.3** (proposed) — NO stable readable direction |

- `build_connectivity_rot` keeps E→I/I→E/I→I bit-identical and **mass fixed by `C_EE=800` (without replacement) + constant `w_EE`** — Increment 2 introduces **no** kernel/distance weight scaling (pathology-mapping spec normalization contract; already satisfied). `AR=2.0 ↔ rho_EE=0.6` (≈45°/ratio1.85, empirical match). Pre-verify any (θ,AR) offline via `connectivity_rot._partner_cov_axis`.
- **Grid/montage-contamination guard** (reader-C pitfall): C-track + shaft-invariance *together* prove the recovered axis follows connectivity, not the montage's own geometry. Both required.

### Item 5 — valid-event count + ≥7-contact threshold

- One **single self-limited event per seed**. Per event: require `n_participating ≥ 2·k_dir+1 = 7` (k_dir=3, Increment-1 lock) for the endpoint axis, else that event = **INSUFFICIENT for direction** (not counted; not forced).
- Per condition (each θ_EE value; each montage rotation; iso): run `n_seed` **(proposed 3)** seeds; require **≥ `n_seed_min = 2`** (proposed) events with a valid direction read, else that condition = **INSUFFICIENT** (loud — reported, never a forced PASS/FAIL). Mirrors the oracle's 3-seed protocol.
- Report **mean axis error ± sd across seeds** per θ_EE (C-track) and per montage rotation (shaft-invariance); `direction_readability` mean ± sd for the iso must-fail.

> **Acceptance gate (encode the conclusion, not existence):** Increment-2 PASS = C-track all three θ_EE mean-err < 25° **AND** shaft-invariance shift < 25° **AND** iso readability < τ_fail **AND** every reported condition has ≥ n_seed_min valid events (else that condition is INSUFFICIENT, surfaced, not silently dropped). Thresholds lock at review; never tuned to pass.

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
PITCH, NC, SHAFTS = 0.3, 7, (10.0, 100.0)
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
        return {"valid": False, "reason": "no_self_limited_event"}
    art = extract_lagpat(env_k, fdt, event_windows=[win],
                         participation_floor=float(env_k.min()),
                         participation_margin=0.5*(float(env_k.max())-float(env_k.min())),
                         timing_frac=0.5, tie_tol=fdt)
    art = attach_geometry(art, montage)
    r0, b0 = art.ranks[:, 0], art.bools[:, 0]
    n_part = int(b0.sum())
    axis = endpoint_centroid_axis(r0, b0, art.contact_coords, k_dir=KDIR, eps_deg=0.5*PITCH)
    read = direction_readability(r0, b0, art.contact_coords)
    return {"valid": axis is not None and n_part >= 2*KDIR+1,
            "n_part": n_part, "axis": None if axis is None else axis.tolist(),
            "readability": read}


def _condition(theta_EE, AR, montage):
    seeds = [_read_one(theta_EE, AR, s, montage) for s in range(1, NSEED+1)]
    valid = [s for s in seeds if s["valid"]]
    if len(valid) < NSEED_MIN:
        return {"status": "INSUFFICIENT", "n_valid": len(valid), "seeds": seeds}
    errs = [axis_angle_error_deg(np.array(s["axis"]), theta_EE) for s in valid]
    reads = [s["readability"] for s in valid]
    return {"status": "OK", "n_valid": len(valid),
            "axis_err_mean": float(np.mean(errs)), "axis_err_sd": float(np.std(errs)),
            "readability_mean": float(np.nanmean(reads)), "seeds": seeds}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT.parent / "figures").mkdir(parents=True, exist_ok=True)
    verdict = {"locked": {"drive": DRIVE, "axis_err_max": AXIS_ERR_MAX, "tau_fail": TAU_FAIL,
                          "k_dir": KDIR, "n_seed": NSEED, "n_seed_min": NSEED_MIN}}
    m0 = _montage()
    # C-track: rotate connectivity, montage fixed
    verdict["C_track"] = {f"{d}deg": _condition(np.deg2rad(d), 2.0, m0) for d in (0, 45, 90)}
    # shaft-invariance: connectivity fixed (45deg), rotate montage
    verdict["shaft_invariance"] = {f"rot{r}": _condition(np.deg2rad(45.0), 2.0, _montage(rot_deg=r))
                                   for r in (0, 30, 60)}
    # must-fail: isotropic (AR=1) + aligned shaft
    verdict["iso_mustfail"] = _condition(np.deg2rad(0.0), 1.0, m0)

    def ok(cond): return cond.get("status") == "OK"
    ct = verdict["C_track"]
    si = verdict["shaft_invariance"]
    iso = verdict["iso_mustfail"]
    c_track_pass = all(ok(ct[k]) and ct[k]["axis_err_mean"] < AXIS_ERR_MAX for k in ct)
    # shaft-invariance: recovered axis must not move with the montage (compare to rot0)
    si_pass = all(ok(si[k]) for k in si) and ok(si["rot0"]) and all(
        abs(si[k]["axis_err_mean"] - si["rot0"]["axis_err_mean"]) < AXIS_ERR_MAX for k in si)
    iso_pass = ok(iso) and iso["readability_mean"] < TAU_FAIL
    verdict["GATE_PASS"] = bool(c_track_pass and si_pass and iso_pass)
    verdict["insufficient"] = [k for grp, d in (("C", ct), ("S", si)) for k in d
                               if not ok(d[k])] + ([] if ok(iso) else ["iso"])
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

- [ ] **Step 2: If the smoke's window is None or n_part < 7** — diagnose (kick too weak / montage off the front / smoothing): adjust `R_END_FRAC`, `kernel_width`, `bin_ms` (these are observation knobs, not locked thresholds). Do NOT touch τ_fail / 25° / k_dir.

- [ ] **Step 3: No commit** (smoke is throwaway config).

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
- **Thresholds locked, never tuned:** 25° / τ_fail=0.3 / k_dir=3 / n_seed_min=2. Observation knobs that MAY be tuned in the smoke (Task 5): `R_END_FRAC`, `kernel_width`, `bin_ms`, `smooth_ms`, pitch/n_contacts — these set the read geometry, not the verdict bar.
- **Oracle ≠ gate:** `anisotropy_front` is the upper-bound parity reference, not the acceptance criterion. The gate is the virtual-electrode + real-pipeline read.
- **Open decisions for the user (review §0):** (1) path A (spike-density envelope, locked) vs path B (current-LFP forward model, deferred) — bump to B if you want the literal observable; (2) all **(proposed)** numbers; (3) the engine `kick_center` patch touches gitignored coworker code — acceptable, or prefer a tracked wrapper?
