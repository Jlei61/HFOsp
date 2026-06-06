# SEF-HFO 虚拟 SEEG 观测层 — Increment 2 Implementation Plan (SNN 低驱动竖切)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development / executing-plans. Steps use `- [ ]`.
> **STATUS: v2 2026-06-07 — reframe RATIFIED by user. Direction claim = UNDIRECTED axis (see §-1).**

**Goal:** Read the SNN's kick-evoked self-limited event through **virtual SEEG contacts + the real pipeline**, and show the recovered **undirected propagation axis** tracks the E→E connectivity axis `θ_EE` — not the kick position, not the electrode geometry — with the isotropic must-fail failing.

**Architecture:** Reuse the Increment-1 observation module (`montage / sample_envelopes / extract_lagpat`). Add: (a) a thin SNN adapter (`spikes → per-contact envelope`); (b) a NEW shared estimator `onset_front_axis` (undirected principal axis + elongation ratio of the onset-front contacts — the oracle's own measure, on virtual contacts); (c) ONE backward-compatible engine kwarg `kick_center`. Connectivity rotation + isotropic = engine's `build_connectivity_rot(theta_EE, AR)`. The source-space oracle (`anisotropy_front.principal_axis`, PASSES err<1.2°, iso 1.13) is a parity upper-bound, NOT the gate.

**Spec:** `docs/superpowers/specs/2026-06-06-sef-hfo-virtual-seeg-observation-layer-design.md`. **Engine map provenance:** workflow `wrryxksbh`.

---

## §-1. The reframe (RATIFIED) — what Increment 2 claims

**Connectivity sets the UNDIRECTED propagation axis; the seed (where the noise kick lands) sets the SIGN (source→sink).** This is the H1/H2 framework: forward/reverse templates share one connectivity-determined spatial axis; which end is source vs sink is seed-determined. Therefore:

- **PASS criterion = "the undirected propagation axis (mod 180°) tracks θ_EE."** Increment 2 makes **no** claim that source→sink direction is connectivity-set (it is seed-set — that IS the H2 forward/reverse phenomenon).
- **Estimator = undirected onset-front principal axis + elongation ratio** (`onset_front_axis`), NOT the endpoint-centroid source→sink axis. Reasons: (1) it is the connectivity-determined quantity; (2) it is well-defined for a CENTER kick's bidirectional wave (endpoint-centroid is degenerate there); (3) `ratio < 1.3` is a robust isotropic must-fail; (4) apples-to-apples with the oracle parity.
- **Endpoint-centroid (directed source→sink) → secondary descriptor**, read from single-END kicks (the forward/reverse / H2 bonus), archive-only, **not the gate**.
- This was a claim-level change; user ratified it 2026-06-07 before this rewrite.

---

## §0. The five locked items (the opening contract)

### Item 1 — SNN low-drive interface → per-contact envelope + direction estimate

- **Operating point LOCKED**: `nu_ext_ratio = 0.6` (interictal quiet-excitable). `T` on `Params` (no `t_max` arg), `dt=0.1 ms`.
- **Envelope LOCKED = smoothed E-spike density (path A)**: kick path saves `E_spk_bool (nsteps, NE)` aligned to `posE`; currents are never stored. Bin+smooth `E_spk_bool` → per-neuron rate; sample at montage via Increment-1 `sample_envelopes(source_frames, grid_xy=posE, montage, kernel_width)`. Zero engine edits for the envelope. **Reporting boundary**: this is a *firing-density* envelope, NOT a synaptic-current LFP — it validates the **direction-axis read** only; cannot be reported as "LFP observation layer validated" (waveform amplitude/shape → path B current-LFP, deferred). **REVIEW HOOK: bump to path B if you want the literal current observable.**
- **Direction estimate LOCKED = `onset_front_axis` (undirected)**: among participating contacts, take the onset-front subset (first-crossing lag within `front_ms` of the earliest), compute the covariance principal axis (`angle mod 180°`) + elongation `ratio = sqrt(λmax/λmin)`. This is the oracle's measure on virtual contacts. **Montage extent must exceed the event footprint** (or the read restricts to the onset front) so the axis reflects the wave, not the montage shape — flag for the Task-5 smoke.
  - **GEOMETRY CONTRACT (verified the hard way in WF-A — ties the center-kick choice to the estimator):** `onset_front_axis` recovers θ_EE **only when the event is an elongated LOBE** (center-origin anisotropic spread → onset front elongated ALONG θ_EE). For a **unidirectional planar front** (edge/line kick) the onset isochrone is a band PERPENDICULAR to propagation and the estimator returns that perpendicular (≈ propagation ± 90°). This is a *second* reason the kick must be CENTER (Item 2), not just the no-confound reason: the center kick produces the lobe geometry the estimator needs. All synthetic tests (Tasks 2–3) use a center-origin anisotropic-lobe lag, NOT a unidirectional ramp.
- **Oracle parity (not a gate)**: `anisotropy_front.principal_axis` on the same `E_spk_bool` — virtual read should agree in axis; may be noisier.

### Item 2 — kick placement + event window (the confound fix)

- **Kick disentangles connectivity from seed position** (the ratified fix; engine kick is hardcoded to center → add backward-compatible `kick_center` kwarg, default `[L/2,L/2]`):
  - **C-track + isotropic must-fail → CENTER kick** (`kick_center = [L/2,L/2]`). Symmetric seed → NO kick-position direction injected. Anisotropic + center kick → bidirectional elongation along θ_EE → clean undirected axis. Isotropic + center kick → circular footprint → ratio≈1 regardless of montage (this is the ONLY way to kill the "off-center kick fakes a radial direction" problem — that fake direction is intrinsic to off-center kicks on isotropic tissue; centering is the structural defense).
  - **kick-track (negative control) → OFF-CENTER kick, varied position** (`kick_center` at e.g. {+x, +y, +xy} offsets), θ_EE FIXED. The recovered undirected axis must STAY ≈ θ_EE, NOT follow the kick. A genuine test — if the axis follows the kick, the read is contaminated or anisotropy is too weak.
  - **single-END kick → secondary only** (forward/reverse descriptor; not a control).
- Kick constants `R_KICK=0.15`, `T_KICK=150 ms`, `DUR_KICK=18 ms`, `KICK_BOOST=2·nu_theta` unchanged.
- **Event window LOCKED**: `detect_events` on the aggregate envelope (mean over contacts) with a per-operating-point recalibrated bar via `calibrate_detector(ref=[KICK_BOOST=0 run aggregate], kick=[KICK_BOOST run aggregate])`. Captures the full ~150ms transit (fixed windows truncate). Emits `(t_on, t_off)`.

### Item 3 — 2D montage: placement + rotation

- **LOCKED**: ≥2 non-parallel shafts via `build_shaft` + `merge_montages`, **centered at sheet center `(L/2,L/2)`**, in the engine mm frame. `n_contacts` per shaft = **8 (EVEN)** so neither shaft has a contact exactly at the shared center (no coincident contact — Increment-1 lesson). Two shafts at (proposed) 10°/100°. Pitch (proposed) 0.3 mm. **Montage extent (≈ `n_contacts·pitch`) must EXCEED the event footprint** (Item 1) — sized in the smoke (`L=2–3 mm` sheet).
- **Rotate-montage** = rebuild all shafts at `+Δ`. **Angle convention = `atan2(dy,dx) mod 180°`** (match the oracle: rot θ=0→0°, θ=90→90°).

### Item 4 — the four controls + how each is judged (estimator = `onset_front_axis`)

| Control | Setup | Judge | PASS criterion |
|---|---|---|---|
| **C-track** (connectivity → axis) | **CENTER kick**; `build_connectivity_rot(theta_EE=deg, AR=2.0)`, θ_EE∈{0,45,90}; montage FIXED | onset-front axis vs θ_EE + ratio, over direction-readable seeds | mean axis err **< 25°** AND mean ratio **> 1.3** for each θ_EE |
| **kick-track** (seed-position confound) | **OFF-CENTER kick**, varied position; θ_EE **FIXED 45°**; montage FIXED | onset-front axis vs the fixed 45° | mean axis err **< 25°** vs 45° for each kick position (axis follows connectivity, NOT the kick) |
| **shaft-invariance** (electrode-geometry confound) | CENTER kick; θ_EE FIXED 45°; rotate montage Δ∈{0,30,60} | onset-front axis vs the fixed 45° | each rotation mean axis err **< 25°** vs 45° |
| **iso must-fail** (is there even an axis) | **CENTER kick**; `build_connectivity_rot(AR=1.0)`; montage FIXED | onset-front ratio over **event-valid** seeds | mean ratio **< 1.3** (no axis). Fizzle below participation gate → INSUFFICIENT (reported; independently supports "anisotropy necessary", consistent with spiking_gt_validation) |

- `build_connectivity_rot` keeps E→I/I→E/I→I bit-identical; **mass fixed by `C_EE=800` (without replacement) + constant `w_EE`** — Increment 2 adds NO weight scaling. `AR=2 ↔ rho_EE=0.6`. Pre-verify (θ,AR) offline via `connectivity_rot._partner_cov_axis`.
- **The four controls jointly isolate the four confound sources**: connectivity (C-track) / seed position (kick-track) / electrode geometry (shaft-invariance) / axis-existence (iso). C-track + kick-track + shaft-invariance together = "axis is from connectivity, not from where we kicked or where we put electrodes."

### Item 5 — seed validity + estimator/knob freeze

- Two orthogonal per-seed flags:
  - **event-valid** = self-limited event window found AND `n_participating ≥ 2·k_dir+1 = 7`.
  - **direction-readable** = event-valid AND `onset_front_axis` returns a defined axis (≥3 onset-front contacts, finite ratio).
- Per condition, `n_seed` (proposed 3); tracking conditions (C-track/kick-track/shaft-invariance) need ≥ `n_seed_min=2` **direction-readable** seeds else INSUFFICIENT; iso must-fail needs ≥ `n_seed_min` **event-valid** seeds else INSUFFICIENT (a condition with too few events is INSUFFICIENT, NEVER a must-fail PASS).
- **Estimator pre-lock (advisor discipline — before the smoke)**: the SNN estimator MUST satisfy, on AR=2 CENTER-kick at θ_EE∈{0,45,90}: ratio>1.3 AND axis within 25° of θ_EE; AND AR=1 CENTER-kick: ratio<1.3; AND axis-tracks-θ_EE-not-montage under montage rotation. **If it cannot, the virtual read is contaminated — STOP** (do not pick the estimator to fit results). Smoke verifies on θ_EE=45° only, then FREEZE.
- **Observation-knob freeze**: `kernel_width / bin_ms / smooth_ms / pitch / n_contacts / front_ms / R_OFF (kick-track offset)` calibrated ONCE in the Task-5 smoke, then frozen + recorded in `verdict.json` BEFORE the formal runs. Verdict bars (25° / 1.3 / k_dir / n_seed_min) never tuned.

> **Acceptance gate:** Increment-2 PASS = C-track each θ_EE (err<25° AND ratio>1.3) **AND** kick-track each kick position (err<25° vs fixed 45°) **AND** shaft-invariance each rotation (err<25° vs fixed 45°) **AND** iso must-fail mean ratio<1.3. Any condition below its seed floor = INSUFFICIENT (surfaced, never a forced verdict; "no events" is never a must-fail PASS). Plus oracle-parity sanity (Task 6).

---

## File structure

- **Create** `src/sef_hfo_snn_adapter.py` — `snn_event_envelope` (spikes→per-contact envelope; pure, synthetic-spike unit-testable).
- **Modify** `src/sef_hfo_observation.py` — add `onset_front_axis` (+ `angle_error_deg` helper) alongside the Increment-1 estimators (shared; toy waves keep using endpoint-centroid).
- **Patch** `results/topic4_sef_hfo/lif_snn/engine/kick_probe.py` — `kick_center` kwarg (gitignored; diff in Task 1).
- **Create** `scripts/run_sef_hfo_obs_increment2.py` — discriminator runner (4 controls + oracle parity).
- **Test** `tests/test_sef_hfo_snn_adapter.py` (synthetic spike front), `tests/test_sef_hfo_observation.py` (add `onset_front_axis` cases).

---

### Task 1: Engine patch — `kick_center` kwarg, backward-compatible

**Files:** Patch `results/topic4_sef_hfo/lif_snn/engine/kick_probe.py` (gitignored — diff recorded; tracked patch-note under `scripts/engine_patches/`).

- [ ] **Step 1: Apply** — in `simulate_kick` (and `fresh_run` forwards it), add `kick_center=None`; build the mask center from it:

```python
center = np.array([p.L/2, p.L/2]) if kick_center is None else np.asarray(kick_center, float)
dist_c = np.linalg.norm(pos - center, axis=1)
kick_mask = is_E & (dist_c <= R_KICK)
```

Default `None` → byte-identical to today.

- [ ] **Step 2: Fast regression** (tiny `Params(L=1.0, density=400.0, T=200.0, nu_ext_ratio=0.6, seed=1)`): `fresh_run(..., kick_center=[0.8,0.5])` → early-kick-window active-E centroid ≈ `[0.8,0.5]`; `kick_center=None` → ≈ `[0.5,0.5]`. (in `tests/test_sef_hfo_snn_adapter.py`, guarded to skip if engine import unavailable.)

- [ ] **Step 3: Commit** the patch-note + test (engine itself is gitignored):

```bash
git add scripts/engine_patches/kick_center.patch tests/test_sef_hfo_snn_adapter.py
git commit -m "feat(topic4 obs): engine kick_center kwarg (single-end/off-center kick) + patch note"
```

---

### Task 2: `onset_front_axis` estimator (undirected) — TDD on synthetic onset fronts

**Files:** Modify `src/sef_hfo_observation.py`; Test `tests/test_sef_hfo_observation.py`.

- [ ] **Step 1: Write the failing test**:

```python
# append to tests/test_sef_hfo_observation.py
from src.sef_hfo_observation import onset_front_axis, angle_error_deg


def test_onset_front_axis_tracks_anisotropic_lobe():
    # CENTER-origin anisotropic spread (the real center-kick geometry): lag grows FASTER
    # across theta_EE than along it -> onset front is a LOBE elongated ALONG theta_EE.
    # (A unidirectional planar ramp is WRONG: its onset isochrone is a band PERPENDICULAR
    # to propagation, so onset_front_axis would return ~perp. See estimator GEOMETRY CONTRACT.)
    g = np.linspace(-3, 3, 9)
    XX, YY = np.meshgrid(g, g)
    coords = np.column_stack([XX.ravel(), YY.ravel()])
    th = np.deg2rad(30.0)
    par = coords @ np.array([np.cos(th), np.sin(th)])        # along theta_EE
    perp = coords @ np.array([-np.sin(th), np.cos(th)])      # across theta_EE
    lag = 3.0 * np.abs(perp) + 1.0 * np.abs(par)             # faster along theta_EE -> lobe
    bools = np.ones(len(coords), bool)
    angle, ratio, n = onset_front_axis(lag, bools, coords, front_ms=2.0)
    assert angle is not None and ratio > 1.3
    assert angle_error_deg(angle, 30.0) < 25.0


def test_onset_front_axis_radial_has_no_axis():
    # onset lag ~ radius from center -> onset front is a ring -> no principal axis
    g = np.linspace(-3, 3, 7)
    XX, YY = np.meshgrid(g, g)
    coords = np.column_stack([XX.ravel(), YY.ravel()])
    lag = np.linalg.norm(coords, axis=1)
    bools = np.ones(len(coords), bool)
    angle, ratio, n = onset_front_axis(lag, bools, coords, front_ms=1.0)
    assert (ratio is None) or (ratio < 1.3)        # ring -> ratio ~ 1
```

- [ ] **Step 2: Run → fail** (`ImportError: onset_front_axis`).

- [ ] **Step 3: Implement** (append to `src/sef_hfo_observation.py`):

```python
def onset_front_axis(lag_raw_ev, bools_ev, coords, front_ms):
    """UNDIRECTED onset-front principal axis (the oracle's measure, on virtual contacts;
    Increment-2 main estimator, spec reframe 2026-06-07). Onset-front = participating
    contacts whose first-crossing lag is within front_ms of the earliest. Returns
    (angle_deg mod 180, elongation ratio = sqrt(lmax/lmin), n_front), or (None, None, n)
    if < 3 front contacts. Unlike endpoint_centroid_axis it has NO source/sink sign and
    is well-defined for a bidirectional (center-kick) wave."""
    idx = np.flatnonzero(np.asarray(bools_ev, bool))
    if idx.size < 3:
        return None, None, int(idx.size)
    lag = np.asarray(lag_raw_ev, float)[idx]
    finite = np.isfinite(lag)
    idx, lag = idx[finite], lag[finite]
    if idx.size < 3:
        return None, None, int(idx.size)
    front = idx[lag <= lag.min() + front_ms]
    if front.size < 3:
        return None, None, int(front.size)
    xy = np.asarray(coords, float)[front]
    c = xy - xy.mean(0)
    cov = (c.T @ c) / len(c)
    evals, evecs = np.linalg.eigh(cov)            # ascending
    ratio = float("inf") if evals[0] <= 1e-12 else float(np.sqrt(evals[1] / evals[0]))
    major = evecs[:, 1]
    angle = float(np.rad2deg(np.arctan2(major[1], major[0])) % 180.0)
    return angle, ratio, int(front.size)


def angle_error_deg(angle_deg, ref_deg) -> float:
    """Undirected (mod 180) error between two angles in degrees."""
    d = abs((angle_deg - ref_deg) % 180.0)
    return float(min(d, 180.0 - d))
```

- [ ] **Step 4: Run → pass.** **Step 5: Commit**:

```bash
git add src/sef_hfo_observation.py tests/test_sef_hfo_observation.py
git commit -m "feat(topic4 obs): onset_front_axis undirected estimator (Increment-2 reframe) + angle_error_deg"
```

---

### Task 3: SNN spike→envelope adapter — TDD on a synthetic spike front

**Files:** Create `src/sef_hfo_snn_adapter.py`; Test `tests/test_sef_hfo_snn_adapter.py`.

- [ ] **Step 1: Write the failing test** — a synthetic traveling spike front, read through the adapter + `onset_front_axis`, recovers the direction:

```python
# tests/test_sef_hfo_snn_adapter.py
import numpy as np
from src.sef_hfo_snn_adapter import snn_event_envelope
from src.sef_hfo_observation import (build_shaft, merge_montages, extract_lagpat,
    attach_geometry, onset_front_axis, angle_error_deg)


def _front(n=2000, L=2.0, ang=np.deg2rad(30.0), dt=0.1, t_max=200.0, c=0.05, seed=0):
    # CENTER-origin anisotropic lobe (real center-kick geometry, NOT a unidirectional
    # ramp): onset grows with the anisotropic distance from the sheet center, faster
    # along `ang` -> onset front elongated ALONG ang (matches onset_front_axis contract).
    rng = np.random.default_rng(seed)
    posE = rng.uniform(0, L, size=(n, 2))
    d = posE - np.array([L / 2, L / 2])
    par = d @ np.array([np.cos(ang), np.sin(ang)])
    perp = d @ np.array([-np.sin(ang), np.cos(ang)])
    onset = 50.0 + (3.0 * np.abs(perp) + 1.0 * np.abs(par)) / c
    nsteps = int(t_max / dt); spk = np.zeros((nsteps, n), bool)
    for j in range(n):
        k = int((onset[j] + rng.normal(0, 1.0)) / dt)
        if 0 <= k < nsteps: spk[k, j] = True
    return spk, posE, ang, dt


def test_adapter_front_direction():
    spk, posE, ang, dt = _front()
    m = merge_montages([build_shaft(np.deg2rad(10.0), 0.3, 8, (1.0, 1.0), "A"),
                        build_shaft(np.deg2rad(100.0), 0.3, 8, (1.0, 1.0), "B")])
    env, fdt, agg = snn_event_envelope(spk, posE, m, dt, bin_ms=2.0, smooth_ms=5.0, kernel_width=0.25)
    art = extract_lagpat(env, fdt, event_windows=[(0.0, env.shape[1]*fdt)],
                         participation_floor=float(env.min()),
                         participation_margin=0.5*(float(env.max())-float(env.min())),
                         timing_frac=0.5, tie_tol=fdt)
    art = attach_geometry(art, m)
    angle, ratio, n = onset_front_axis(art.lag_raw[:, 0], art.bools[:, 0], art.contact_coords, front_ms=8.0)
    assert angle is not None and ratio > 1.3
    assert angle_error_deg(angle, 30.0) < 25.0
```

- [ ] **Step 2: Run → fail.** **Step 3: Implement** (the adapter from the engine map — bin+smooth `E_spk_bool` → `sample_envelopes(grid_xy=posE)`):

```python
# src/sef_hfo_snn_adapter.py
"""SNN -> observation adapter (Increment 2). Pure: bin + temporally smooth per-neuron E
spikes into a rate field, sample at virtual contacts via the Increment-1 Gaussian
footprint. Only model-specific piece; downstream is the shared chain."""
from __future__ import annotations
import numpy as np
from src.sef_hfo_observation import sample_envelopes


def _bin_and_smooth(E_spk_bool, dt, bin_ms, smooth_ms):
    nsteps, NE = E_spk_bool.shape
    bs = max(1, int(round(bin_ms / dt))); nf = nsteps // bs
    binned = E_spk_bool[: nf*bs].reshape(nf, bs, NE).sum(axis=1).astype(float)
    sig = max(1e-6, smooth_ms / bin_ms); half = int(np.ceil(3*sig))
    x = np.arange(-half, half+1); k = np.exp(-(x**2)/(2*sig**2)); k /= k.sum()
    sm = np.apply_along_axis(lambda col: np.convolve(col, k, mode="same"), 0, binned)
    return sm, bin_ms


def snn_event_envelope(E_spk_bool, posE, montage, dt, bin_ms=2.0, smooth_ms=5.0, kernel_width=0.25):
    """Returns (envelopes (n_contact, n_frame), frame_dt_ms, aggregate (n_frame,))."""
    rate, fdt = _bin_and_smooth(np.asarray(E_spk_bool, bool), dt, bin_ms, smooth_ms)
    env = sample_envelopes(rate, np.asarray(posE, float), montage, kernel_width)
    return env, fdt, env.mean(axis=0)
```

- [ ] **Step 4: Run → pass.** **Step 5: Commit**:

```bash
git add src/sef_hfo_snn_adapter.py tests/test_sef_hfo_snn_adapter.py
git commit -m "feat(topic4 obs): SNN spike->envelope adapter (synthetic-front unit test, onset-front read)"
```

---

### Task 4: Event-window helper (`detect_events` + recalibrated bar) — TDD synthetic aggregate

**Files:** `scripts/run_sef_hfo_obs_increment2.py` (helper) ; Test `tests/test_sef_hfo_snn_adapter.py`.

- [ ] **Step 1: Failing test** — quiet→bump→quiet aggregate yields one self-terminating window (as in the prior plan version): `calibrate_detector([ref], kick)` → `detect_events` → exactly one `returned=True` window with `t_on∈(100,175)`.
- [ ] **Step 2: Run → fail** (helper missing). **Step 3: Implement** `event_window_for_run(agg_kick, agg_ref, frame_dt)` (recalibrate bar; return the highest-peak self-terminating event's `(t_on,t_off)`, else None). **Step 4: pass. Step 5: commit.**

(Code identical to the prior plan version's Task 3 `event_window_for_run`.)

---

### Task 5: Small-scale SNN smoke + estimator/knob FREEZE

**Files:** none new (smoke invocation).

- [ ] **Step 1: Run a fast smoke** — `Params(L=2.0, density=1000.0, T=300.0, nu_ext_ratio=0.6, seed=1)`, ONE condition θ_EE=45° AR=2, **CENTER kick**. Confirm the chain: `build_connectivity_rot(45°,2)` → `simulate_kick(kick_center=[L/2,L/2])` → adapter → window found → `onset_front_axis` gives **ratio>1.3 AND axis within 25° of 45°**. Also a quick AR=1 center-kick → **ratio<1.3** (or fizzle→INSUFFICIENT). **This is the estimator pre-lock check (Item 5).**
- [ ] **Step 2: Tune ONLY observation knobs** (`kernel_width / bin_ms / smooth_ms / pitch / n_contacts / front_ms / R_OFF`) so the smoke passes — never the verdict bars (25°/1.3). **Verify montage extent > event footprint** (else all contacts participate → reading montage shape).
- [ ] **Step 3: FREEZE** — record final knob values into the runner constants + `verdict.json::locked.obs_knobs`; do NOT change them for the formal run. If the estimator cannot pass the pre-lock on the smoke condition → **STOP** (the virtual read is contaminated; escalate to the user, do not proceed to the formal run). Smoke config is throwaway (no commit beyond the frozen constants).

---

### Task 6: Full discriminator (4 controls) + oracle parity + verdict + README

**Files:** `scripts/run_sef_hfo_obs_increment2.py` (main); `results/topic4_sef_hfo/observation_layer/increment2_snn_slice/`; `.../figures/README.md`.

- [ ] **Step 1: Write the runner main** — `onset_front_axis`-based, four controls. Sketch (engine calls from the map; helpers from Tasks 3–4):

```python
# scripts/run_sef_hfo_obs_increment2.py (main)
import sys, json; from pathlib import Path; import numpy as np
ENGINE = Path("results/topic4_sef_hfo/lif_snn/engine"); sys.path.insert(0, str(ENGINE))
from params import Params, compute_nu_theta
from connectivity import place_neurons
from connectivity_rot import build_connectivity_rot
from kick_probe import simulate_kick                       # patched: kick_center
from src.sef_hfo_observation import (build_shaft, merge_montages, extract_lagpat,
    attach_geometry, onset_front_axis, angle_error_deg)
from src.sef_hfo_snn_adapter import snn_event_envelope

L, DENSITY, T, DT, DRIVE = 2.0, 4000.0, 350.0, 0.1, 0.6
PITCH, NC, SHAFTS = 0.3, 8, (10.0, 100.0)      # NC even (no center coincident contact)
KDIR, NSEED, NSEED_MIN = 3, 3, 2
AXIS_ERR_MAX, RATIO_MIN, FRONT_MS = 25.0, 1.3, 8.0
R_OFF = 0.6 * (L/2)                            # kick-track off-center offset (frozen in smoke)
OUT = Path("results/topic4_sef_hfo/observation_layer/increment2_snn_slice")
CENTER = np.array([L/2, L/2])

def _montage(rot_deg=0.0):
    return merge_montages([build_shaft(np.deg2rad(SHAFTS[0]+rot_deg), PITCH, NC, tuple(CENTER), "A"),
                           build_shaft(np.deg2rad(SHAFTS[1]+rot_deg), PITCH, NC, tuple(CENTER), "B")])

def _read(theta_EE, AR, seed, montage, kick_center):
    p = Params(L=L, density=DENSITY, T=T, dt=DT, nu_ext_ratio=DRIVE, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta_EE, AR=AR)
    posE = net["pos"][:NE]; nu_theta = compute_nu_theta(p)[0]
    net["rng"] = np.random.default_rng(seed)
    rk = simulate_kick(p, net, KICK_BOOST=2*nu_theta, kick_center=kick_center)
    net["rng"] = np.random.default_rng(seed)
    rr = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=kick_center)
    env, fdt, agg = snn_event_envelope(rk["E_spk_bool"], posE, montage, DT)
    _, _, aggr = snn_event_envelope(rr["E_spk_bool"], posE, montage, DT)
    win = event_window_for_run(agg, aggr, fdt)
    if win is None:
        return {"event_valid": False, "direction_readable": False}
    art = extract_lagpat(env, fdt, event_windows=[win], participation_floor=float(env.min()),
                         participation_margin=0.5*(float(env.max())-float(env.min())),
                         timing_frac=0.5, tie_tol=fdt)
    art = attach_geometry(art, montage)
    n_part = int(art.bools[:,0].sum())
    angle, ratio, nf = onset_front_axis(art.lag_raw[:,0], art.bools[:,0], art.contact_coords, FRONT_MS)
    ev = n_part >= 2*KDIR+1
    return {"event_valid": bool(ev), "direction_readable": bool(ev and angle is not None),
            "n_part": n_part, "angle": angle, "ratio": ratio}

def _track(theta_EE, AR, montage, kick_center):
    seeds = [_read(theta_EE, AR, s, montage, kick_center) for s in range(1, NSEED+1)]
    rd = [s for s in seeds if s["direction_readable"]]
    if len(rd) < NSEED_MIN:
        return {"status": "INSUFFICIENT", "n_readable": len(rd), "seeds": seeds}
    errs = [angle_error_deg(s["angle"], np.rad2deg(theta_EE)) for s in rd]
    rats = [s["ratio"] for s in rd]
    return {"status": "OK", "axis_err_mean": float(np.mean(errs)), "ratio_mean": float(np.mean(rats)),
            "pass": bool(np.mean(errs) < AXIS_ERR_MAX and np.mean(rats) > RATIO_MIN), "seeds": seeds}

def _mustfail(theta_EE, AR, montage, kick_center):
    seeds = [_read(theta_EE, AR, s, montage, kick_center) for s in range(1, NSEED+1)]
    ev = [s for s in seeds if s["event_valid"]]
    if len(ev) < NSEED_MIN:
        return {"status": "INSUFFICIENT", "n_event_valid": len(ev), "seeds": seeds}
    rats = [s["ratio"] if (s.get("ratio") is not None) else 1.0 for s in ev]   # no-axis -> ratio~1
    return {"status": "OK", "ratio_mean": float(np.mean(rats)),
            "pass": bool(np.mean(rats) < RATIO_MIN), "seeds": seeds}

def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT.parent/"figures").mkdir(parents=True, exist_ok=True)
    m0 = _montage()
    off = {f"off{int(np.rad2deg(a))}": (CENTER + R_OFF*np.array([np.cos(a), np.sin(a)]))
           for a in (0.0, np.pi/2, np.pi/4)}
    V = {"locked": {"drive": DRIVE, "axis_err_max": AXIS_ERR_MAX, "ratio_min": RATIO_MIN,
                    "k_dir": KDIR, "n_seed": NSEED, "n_seed_min": NSEED_MIN, "front_ms": FRONT_MS,
                    "obs_knobs": {"pitch": PITCH, "n_contacts": NC, "r_off": R_OFF}}}
    V["C_track"] = {f"{d}deg": _track(np.deg2rad(d), 2.0, m0, CENTER) for d in (0,45,90)}      # CENTER kick
    V["kick_track"] = {k: _track(np.deg2rad(45.0), 2.0, m0, c) for k, c in off.items()}        # θ fixed, kick off-center
    V["shaft_invariance"] = {f"rot{r}": _track(np.deg2rad(45.0), 2.0, _montage(rot_deg=r), CENTER) for r in (0,30,60)}
    V["iso_mustfail"] = _mustfail(np.deg2rad(0.0), 1.0, m0, CENTER)                             # CENTER kick AR=1
    ok = lambda c: c.get("status") == "OK"
    def all_pass(grp): return all(ok(grp[k]) and grp[k]["pass"] for k in grp)
    V["GATE_PASS"] = bool(all_pass(V["C_track"]) and all_pass(V["kick_track"])
                          and all_pass(V["shaft_invariance"]) and ok(V["iso_mustfail"]) and V["iso_mustfail"]["pass"])
    V["insufficient"] = [f"{g}:{k}" for g in ("C_track","kick_track","shaft_invariance")
                         for k in V[g] if not ok(V[g][k])] + ([] if ok(V["iso_mustfail"]) else ["iso"])
    (OUT/"verdict.json").write_text(json.dumps(V, indent=2, default=lambda o: None))
    print("GATE_PASS =", V["GATE_PASS"], "| insufficient:", V["insufficient"])

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Oracle parity** — for θ_EE=45°, one seed, also run `anisotropy_front.principal_axis` on the same `E_spk_bool`; record its axis error; assert the virtual read agrees in axis (both track 45°). Add `V["oracle_parity"]`.
- [ ] **Step 3: Figure** — 4 panels: C-track axis-err+ratio vs θ_EE; kick-track axis vs kick position (flat at 45°); shaft-invariance axis vs montage rotation (flat at 45°); iso ratio vs 1.3 line. Save `figures/increment2_snn.png`.
- [ ] **Step 4: Run** `PYTHONPATH=. python scripts/run_sef_hfo_obs_increment2.py`; capture GATE_PASS + insufficient. **Eyeball**: an INSUFFICIENT condition is an honest outcome (report it), NOT a reason to loosen thresholds or hand-pick seeds.
- [ ] **Step 5: README** (Chinese, AGENTS.md) describing the 4 controls + oracle parity + 关注点. **Step 6: Commit** the runner.

---

## Self-Review notes

- **Reframe (RATIFIED):** Increment-2 claims the UNDIRECTED axis tracks θ_EE; estimator = `onset_front_axis` (oracle's measure on virtual contacts). Endpoint-centroid = secondary forward/reverse descriptor (single-END kick; archive, not gate).
- **Confound isolation:** C-track (center kick, no seed-direction) proves connectivity sets the axis; kick-track (off-center, θ_EE fixed) proves the axis does NOT follow the seed; shaft-invariance proves it does NOT follow electrodes; iso (center, AR=1) proves there must BE a connectivity axis to read.
- **Center kick is load-bearing**: it removes the seed-position confound (symmetric) AND makes the iso must-fail clean (circular → ratio≈1). Off-center kick on isotropic tissue genuinely fabricates a radial direction — no estimator is immune; centering is the structural defense.
- **Montage extent > event footprint** (else reads montage shape) — verified in the smoke.
- **Thresholds + estimator pre-locked, never tuned**: 25° / ratio 1.3 / k_dir=3 / n_seed_min=2 fixed; estimator must pass the Item-5 pre-lock on the smoke or STOP. Observation knobs frozen after smoke.
- **TDD reality:** estimator (Task 2) + adapter (Task 3) + window (Task 4) unit-tested on synthetics (fast); real SNN only in smoke (Task 5) + full run (Task 6).
- **Open for user:** path A (firing-density, locked) vs path B (current-LFP, deferred); the gitignored-engine `kick_center` patch.
