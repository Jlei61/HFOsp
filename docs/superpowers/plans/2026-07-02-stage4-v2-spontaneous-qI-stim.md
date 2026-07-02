# Stage-4 v2: Spontaneous Big-Focus q_I/g_K → Stim GIF — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find a spontaneous (no external kick) big-focus working point on the Stage-4 `extended_patch` substrate where a TRAIN of discrete, self-terminating interictal-like events slowly depletes the q_I inhibitory-resource field until the sheet tips into runaway — then render a stimulation-vs-no-stimulation comparison GIF that tests whether clamping the core **suppresses the event generator / the q_I build-up-to-runaway** on that working point.

**Architecture:** Two phases sharing one build/sim infrastructure. Phase 1 searches for the working point with hard cost controls (early-abort + a SEPARATE short-T fast-reject CLI stage → an explicitly-triggered long-T confirm CLI stage). Phase 2 renders the stim GIF on the found working point. The pilot already proved the naive regime (pure q_I, hot big core) gives an **instant one-shot burst at ~30 ms, no train** (`results/topic4_sef_hfo/stage4_spontaneous_qI_pilot/screen.json`, 6/6 `one_shot_burst`). The scientific bet of Phase 1: **coupling g_K as a FAST per-event brake** (short `tau_K`, `eta_K>0`) discretizes the spontaneous train (each nucleation self-terminates via fatigue), while **q_I does the SLOW across-event buildup** (long `tau_q`) to runaway — a two-timescale hypothesis that may fail. Phase 1 ends in an explicit go/no-go gate; if no working point is found, STOP and report the negative (do NOT force Phase 2).

**Tech Stack:** Python, NumPy, the repo SNN engine at `src/snn_engine/` (`kick_probe.simulate_kick`, `slow_field.SpatialSlowField`, `connectivity`/`connectivity_rot`, `params.Params`), `src/sef_hfo_heterogeneity.sample_core_field`, matplotlib + imageio for the GIF, pytest (`slow` marker registered in `pyproject.toml`).

## Global Constraints

- **Spontaneous means KICK_BOOST=0 / no pulse schedule.** Events are OU/Poisson-background-driven only. In `_simulate_continuous` this is `n_pulses=0` (empty `_pulse_schedule`); nothing external triggers events.
- **Both slow variables are in play (user instruction).** q_I (slow, `tau_q≈5000` ms) = across-event buildup carrier; g_K (fast, small `tau_K`, `eta_K>0` = COUPLED, not just visualized) = per-event self-termination. This differs from the kick-driven `fig_m3a_v2_2_qI_runaway_transition` where g_K was visualized only (`eta_K=0`).
- **ONE shared runaway criterion (LOCKED), used by early-abort AND the classifier AND the screen summary.** Definition: smooth the per-step E-rate with a 20 ms boxcar (`_smooth_rate`), then `_first_sustained(smoothed_hz, dt, threshold_hz=RUNAWAY_HZ, dur_ms=RUNAWAY_DUR_MS)` where `RUNAWAY_HZ=120.0`, `RUNAWAY_DUR_MS=100.0` (the existing `_first_sustained` uses the 80%-of-window rule). Early-abort periodically re-runs this exact helper on the rate accumulated so far and breaks on the first non-None onset — so abort and post-hoc detection agree by construction. Do NOT introduce a second, instantaneous-rate abort rule.
- **`aborted_ms` IS runaway evidence.** An abort fires only because the shared criterion detected sustained runaway. Classification uses `effective_runaway = runaway_ms if runaway_ms is not None else aborted_ms`; only when BOTH are None may a run be called `train_no_runaway`/`silent`.
- **Cost is a hard constraint: each L=20 runaway sim at full `T=2500` ms ≈ 1 hour.** The fast-reject stage (short T + early-abort) is the default and the only thing that runs without explicit user go. The long-confirm stage is a SEPARATE CLI invocation the user triggers after seeing the fast survivors + a wall-clock estimate. Never auto-chain fast→confirm in one process.
- **Do NOT edit the engine loop (`kick_probe.py`) or `slow_field.py`.** The slow-variable slot in `simulate_kick` is the integration point; the GIF loop is the vendored `_simulate_continuous`. All additions go in paper-figure / pilot / screener scripts.
- **Known pre-existing bug (do NOT rely on the runner path):** `run_sef_hfo_snn_cm_spontaneous_readout.py::build_lesion_vth` extended_patch branch passes `elongation`/`axis_unit` to `sample_core_field`, whose current signature (`src/sef_hfo_heterogeneity.py:442`) does NOT accept them. Build the big core DIRECTLY with `sample_core_field(pos, is_E, center, core_r, rng, core_mean=, core_std=, base_mean=18.0)` (isotropic == elongation 1.0). Do not fix the runner as part of this plan.
- **Phase 2 claim scope (LOCKED).** On a single big focus there is NO A/B propagation axis, so clamping the core is **external preventive suppression of the event generator / the build-up-to-runaway driver** — NOT a propagation-barrier / midline-block claim. Figure/README/metadata must say exactly that. A propagation-barrier question (middle/corridor stim) is a DIFFERENT experiment and belongs to the kick-driven two-focus figures (`fig_m3a_v2_2_qI_stim_*`), not here.
- **Scientific-honesty framing (locked):** every figure/README/metadata says "visual diagnostic, single trajectory; NOT a treatment/recovery/closed-loop claim; runaway/tonic is never an ictal-like event." Working-point search results are exploratory screens, not cohort claims.
- **Docs discipline:** durable findings (working point or negative) go to `docs/archive/topic4/`. Do NOT edit the agent memory files as a plan step — update memory only if the user explicitly asks.
- **Reuse, do not re-invent:** `_simulate_continuous`, `_smooth_rate`, `_first_sustained`, `_draw_arm`, `_zlfp`, `_activity_fields`, `_render_frames`, `_electrode_e_mask`, `intervention_vth_at_time` already exist. New code adds the single-big-core build, the `vth`/`n_pulses=0`/early-abort params, the pulse-mask/`_source_xy` single-focus fix, the screener, and single-core render tweaks.

**Canonical paths:**
- Companion GIF machinery: `scripts/paper_figures/plot_fig_m3a_v2_2_hG_runaway_transition_gif.py` (owns `_simulate_continuous`, `_build`, `_build_subject1146`, `_two_core_vth`, `_source_xy`, `_pulse_schedule`, `_draw_contacts`, `_axis_ellipse`, `_style_spatial`, `_activity_metrics`, `_smooth_rate`, `_first_sustained`, color consts; imports `run_m3a_v2_step2_qI` as `S2`, `sample_core_field`, `region_masks`, engine builders).
- Stim comparison runner: `scripts/paper_figures/plot_fig_m3a_v2_2_qI_stim_runaway_gif.py` (owns `_draw_arm`, `_render_frames`, `_zlfp`, `_activity_fields`, `_electrode_e_mask`).
- Existing feasibility pilot (superseded by the screener): `scripts/pilot_stage4_spontaneous_qI.py`.

---

## Phase 1 — Working-point search

### Task 1: Spontaneous single-big-core build + single-focus safety + `vth`/early-abort on `_simulate_continuous`

**Files:**
- Modify: `scripts/paper_figures/plot_fig_m3a_v2_2_hG_runaway_transition_gif.py`
- Test: `tests/test_stage4_v2_spontaneous.py`

**Interfaces:**
- Produces: `_build_stage4_patch(cfg) -> S` (S has `p, net, NE, NI, posE, posI, N, labels, axis_unit, center, L, masks`, `layout={"kind":"stage4_patch","label":"Stage-4 big focus","foci":[center.tolist()],"core_r":cfg.core_radius,"axis_unit":...,"L":L}`, plus `S["core_mask"]` (full-N bool) and `S["patch_vth"]` (full-N float)).
- Produces: `_source_xy(S, source)` returns `foci[0]` when the layout has a single focus (`len(foci)==1`) OR `kind=="stage4_patch"`, for ANY `source` (no `foci[1]` access).
- Produces: `_simulate_continuous(..., vth=None, abort_on_runaway=False, abort_check_every=25)` — `vth` overrides `_two_core_vth`; the pulse-mask loop only builds masks for sources that actually occur in `pulses` (so `n_pulses=0` never touches tempA/tempB); with `abort_on_runaway=True` the loop breaks at the first shared-criterion runaway onset, truncates all per-step arrays, and sets `res["aborted_ms"]` (float) else None.
- Consumes: engine `place_neurons`, `build_connectivity_rot`, `Params`; `sample_core_field`; `S2` (`region_masks`, `N_GRID`, `SUBSTRATES`).

- [ ] **Step 1: Write the failing test for the single-big-core build**

Create `tests/test_stage4_v2_spontaneous.py`:

```python
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "src", "snn_engine"), os.path.join(ROOT, "scripts"),
           os.path.join(ROOT, "scripts", "paper_figures"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plot_fig_m3a_v2_2_hG_runaway_transition_gif as H  # noqa: E402


@pytest.mark.slow
def test_build_stage4_patch_single_core():
    cfg = H.ProtocolConfig(layout="stage4_patch", top="qI", use_gK=True, eta_K=0.5,
                           tau_K=200.0, core_mean=17.0, core_std=1.5, core_radius=6.0,
                           T=150.0, n_pulses=0, seed=1)
    S = H._build(cfg)
    assert S["layout"]["kind"] == "stage4_patch"
    assert len(S["layout"]["foci"]) == 1                         # ONE big focus
    assert S["core_mask"].shape[0] == S["N"]
    assert int(S["core_mask"][:S["NE"]].sum()) >= 200            # r=6 mm disk on L=20 is large
    assert S["patch_vth"].shape[0] == S["N"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_stage4_v2_spontaneous.py::test_build_stage4_patch_single_core -v`
Expected: FAIL — `_build` does not handle `layout="stage4_patch"`.

- [ ] **Step 3: Implement `_build_stage4_patch`, wire into `_build`, add `ProtocolConfig` fields if missing**

Add to `ProtocolConfig` (both are ABSENT — verified): `drive: float = 0.6` and `L: float = 20.0` (the subject1146/stage5 builders set L internally; only stage4_patch reads `cfg.L`). Add near `_build_subject1146`.

**Substrate params are the CANONICAL Stage-4 runner values, NOT `S2.SUBSTRATES`.** Source of truth (`run_sef_hfo_snn_cm_spontaneous_readout.py:520-525`): `Params(g=3.6, ...)` hardcoded; CLI defaults `AR=2.0`, `theta=45deg`, `density=100`, `drive=0.6`, `_ee_over={}` (Params E→E defaults). `S2.SUBSTRATES["primary"]` is `g=8.0/AR=4.0` — the v2 kick-driven substrate, a DIFFERENT regime; using it would silently screen the wrong base. This matches the pilot (g=3.6, AR=2.0) except the pilot used `theta=0`; the canonical runner default is `theta=45deg`, which we adopt.

```python
def _build_stage4_patch(cfg: "ProtocolConfig"):
    """ONE large isotropic excitable disk at the sheet centre (Stage-4 extended_patch),
    spontaneous (no kick). Built directly via sample_core_field (the runner's build_lesion_vth
    extended_patch path is broken against the current sample_core_field signature).
    Substrate = canonical Stage-4 spontaneous runner (g=3.6, AR=2.0, theta=45deg, density=100,
    drive=0.6), NOT S2.SUBSTRATES (g=8.0/AR=4.0 = the v2 kick substrate)."""
    L = float(cfg.L)
    theta_rad = np.deg2rad(45.0)                      # canonical Stage-4 theta (runner CLI default)
    axis_unit = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    center = np.array([L / 2.0, L / 2.0])
    p = Params(g=3.6, L=L, density=100.0, T=cfg.T, dt=0.1, nu_ext_ratio=cfg.drive, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta_rad, AR=2.0, verbose=False)
    pos = net["pos"]
    is_E = np.zeros(NE + NI, bool); is_E[:NE] = True
    cf = sample_core_field(pos, is_E, center, cfg.core_radius, np.random.default_rng(cfg.seed + 7),
                           core_mean=cfg.core_mean, core_std=cfg.core_std, base_mean=18.0)
    layout = {"kind": "stage4_patch", "label": "Stage-4 big focus", "foci": [center.tolist()],
              "core_r": float(cfg.core_radius), "axis_unit": axis_unit.tolist(), "L": L}
    S = dict(p=p, net=net, NE=NE, NI=NI, posE=pos[:NE], posI=pos[NE:], N=NE + NI, labels=labels,
             axis_unit=axis_unit, center=center, L=L, layout=layout,
             core_mask=cf["core_mask"], patch_vth=cf["vth"])
    S["masks"] = region_masks(L, S2.N_GRID, center, axis_unit, S2.CORRIDOR_HW)
    return S
```

In `_build`, add the dispatch:

```python
def _build(cfg: ProtocolConfig):
    if cfg.layout == "subject1146":
        return _build_subject1146(cfg)
    if cfg.layout == "stage4_patch":
        return _build_stage4_patch(cfg)
    S = S2.build(S2.SUBSTRATES[cfg.substrate], cfg.seed, T=cfg.T)
    S["masks"] = region_masks(S["L"], S2.N_GRID, S["center"], S["axis_unit"], S2.CORRIDOR_HW)
    return S
```

- [ ] **Step 4: Run the build test to verify it passes**

Run: `python -m pytest tests/test_stage4_v2_spontaneous.py::test_build_stage4_patch_single_core -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test for single-focus safety (P0: no tempB access)**

```python
@pytest.mark.slow
def test_spontaneous_single_focus_never_touches_tempB():
    # stage4_patch has ONE focus; _source_xy must not index foci[1], and n_pulses=0 must build no masks
    cfg = H.ProtocolConfig(layout="stage4_patch", top="qI", use_gK=True, eta_K=0.0,
                           core_mean=16.5, core_std=1.5, core_radius=6.0,
                           T=120.0, n_pulses=0, seed=1)
    S = H._build(cfg)
    assert np.allclose(H._source_xy(S, "tempA"), H._source_xy(S, "tempB"))   # both -> the one focus
    # the sim runs to completion (no IndexError from a missing second focus)
    res = H._simulate_continuous(S, cfg, record_gif=False, vth=S["patch_vth"])
    assert res["E_spk_bool"].shape[0] == int(round(cfg.T / S["p"].dt))
```

- [ ] **Step 6: Run it to verify it fails**

Run: `python -m pytest tests/test_stage4_v2_spontaneous.py::test_spontaneous_single_focus_never_touches_tempB -v`
Expected: FAIL — `_source_xy(S, "tempB")` raises IndexError (`foci[1]`), and/or `_simulate_continuous` has no `vth` kwarg.

- [ ] **Step 7: Fix `_source_xy` (single-focus) and the pulse-mask loop; add `vth` to `_simulate_continuous`**

`_source_xy`:

```python
def _source_xy(S: dict, source: str) -> np.ndarray:
    if "layout" in S and "foci" in S["layout"]:
        foci = S["layout"]["foci"]
        if S["layout"].get("kind") == "stage4_patch" or len(foci) == 1:
            return np.asarray(foci[0], float)             # single focus: any source -> the one core
        return np.asarray(foci[0 if source == "tempA" else 1], float)
    sign = -1.0 if source == "tempA" else 1.0
    return np.asarray(S["center"], float) + sign * 0.6 * (float(S["L"]) / 2.0) * np.asarray(S["axis_unit"], float)
```

In `_simulate_continuous`, replace the unconditional two-source mask loop with one that follows the actual pulses (so `n_pulses=0` builds nothing), and add the `vth` override to the signature + body:

```python
    pulses = _pulse_schedule(cfg)
    masks = {}
    for source in sorted({pl["source"] for pl in pulses}):     # only sources that actually fire
        c = _source_xy(S, source)
        masks[source] = is_E & (np.linalg.norm(pos - c, axis=1) <= cfg.r_kick)
```

Signature: add `vth=None` (keyword). Body: `vth = _two_core_vth(S, cfg) if vth is None else np.asarray(vth, float)`.

- [ ] **Step 8: Run the single-focus test to verify it passes**

Run: `python -m pytest tests/test_stage4_v2_spontaneous.py::test_spontaneous_single_focus_never_touches_tempB -v`
Expected: PASS.

- [ ] **Step 9: Write the failing test for early-abort (shared runaway criterion)**

```python
@pytest.mark.slow
def test_early_abort_uses_shared_runaway_criterion():
    cfg = H.ProtocolConfig(layout="stage4_patch", top="qI", use_gK=True, eta_K=0.0,
                           core_mean=16.5, core_std=1.5, core_radius=6.0,
                           T=600.0, n_pulses=0, seed=1)
    S = H._build(cfg)
    res = H._simulate_continuous(S, cfg, record_gif=False, vth=S["patch_vth"], abort_on_runaway=True)
    assert res["aborted_ms"] is not None and res["aborted_ms"] < 400.0     # hot core bursts, abort fires
    n = res["E_spk_bool"].shape[0]
    assert n <= int(round(res["aborted_ms"] / S["p"].dt)) + 1              # arrays truncated at abort
    # the SAME shared criterion, run post-hoc on the (truncated) rate, agrees an onset exists
    rate_hz = np.asarray(res["rate_E"], float)
    assert H._first_sustained(H._smooth_rate(rate_hz, S["p"].dt, 20.0), S["p"].dt, 120.0, 100.0) is not None
```

- [ ] **Step 10: Run it to verify it fails**

Run: `python -m pytest tests/test_stage4_v2_spontaneous.py::test_early_abort_uses_shared_runaway_criterion -v`
Expected: FAIL — no `abort_on_runaway` kwarg.

- [ ] **Step 11: Implement early-abort via the shared helper**

In `_simulate_continuous`: add `abort_on_runaway: bool = False, abort_check_every: int = 25` to the signature. Before the loop: `aborted_step = None`. Note `rate_E[t]` holds spike COUNTS; convert to Hz for the shared helper. Inside the loop, AFTER `rate_E[t] = spk[:NE].sum()`:

```python
        if abort_on_runaway and t >= abort_check_every and (t % abort_check_every == 0):
            _rate_hz = rate_E[:t + 1] / NE / dt * 1e3
            if _first_sustained(_smooth_rate(_rate_hz, dt, 20.0), dt, 120.0, 100.0) is not None:
                aborted_step = t
                break
```

After the loop, if `aborted_step is not None`, truncate to `k = aborted_step + 1` every per-step array that was pre-allocated to `nsteps` (`times, rate_E, E_spk_bool`, `lfp_trace` if not None, `qI_min_trace`, `gK_axial_trace`, `stim_active`) before computing the returned `rate_E`/`rate_I` Hz arrays from the truncated counts. Add `res["aborted_ms"] = (aborted_step * dt) if aborted_step is not None else None`. The slow-field traces (`slow.trace_qI_mean`, `trace_qI_min` via `qI_min_trace`) are already per-executed-step length; make sure the returned `trace_qI_min`/`trace_gK_axial` are the truncated views.

- [ ] **Step 12: Run the abort test to verify it passes**

Run: `python -m pytest tests/test_stage4_v2_spontaneous.py::test_early_abort_uses_shared_runaway_criterion -v`
Expected: PASS.

- [ ] **Step 13: Parity guard — abort-off + vth-None unchanged**

```python
@pytest.mark.slow
def test_abort_off_and_vth_none_is_unchanged():
    cfg = H.ProtocolConfig(layout="subject1146", top="qI", use_gK=True, eta_K=0.0,
                           use_hG=False, T=150.0, n_pulses=0, seed=1)
    S = H._build(cfg)
    a = H._simulate_continuous(S, cfg, record_gif=False)
    b = H._simulate_continuous(S, cfg, record_gif=False, vth=None, abort_on_runaway=False)
    assert np.array_equal(a["E_spk_bool"], b["E_spk_bool"])
    assert b.get("aborted_ms") is None
```

Run: `python -m pytest tests/test_stage4_v2_spontaneous.py -m slow -v`
Expected: all PASS.

- [ ] **Step 14: Commit**

```bash
git add scripts/paper_figures/plot_fig_m3a_v2_2_hG_runaway_transition_gif.py tests/test_stage4_v2_spontaneous.py
git commit -m "feat(stage4-v2): single-big-core spontaneous build + single-focus safety + vth/early-abort on _simulate_continuous"
```

---

### Task 2: Working-point classifier (pure, fast; aborted_ms == runaway)

**Files:**
- Create: `scripts/run_stage4_v2_workpoint_search.py`
- Test: `tests/test_stage4_v2_spontaneous.py`

**Interfaces:**
- Produces: `classify_workpoint(event_ons, runaway_ms, aborted_ms, T, *, min_train=3, early_ms=200.0) -> str` in `{"silent","one_shot_burst","train_then_runaway","train_no_runaway","few_events_then_runaway"}`. Uses `effective_runaway = runaway_ms if runaway_ms is not None else aborted_ms`.
- Produces: `is_working_point(verdict) -> bool` (True only for `"train_then_runaway"`).

- [ ] **Step 1: Write the failing tests (including the abort-as-runaway case)**

```python
from run_stage4_v2_workpoint_search import classify_workpoint, is_working_point  # noqa: E402


def test_classify_one_shot_burst():
    assert classify_workpoint([30.0], runaway_ms=32.0, aborted_ms=180.0, T=2500.0) == "one_shot_burst"


def test_classify_abort_counts_as_runaway():
    # abort fired at 250 ms with only 1 prior event -> still a burst, NOT train_no_runaway
    assert classify_workpoint([40.0], runaway_ms=None, aborted_ms=250.0, T=2500.0) == "one_shot_burst"


def test_classify_train_then_runaway():
    v = classify_workpoint([300.0, 700.0, 1100.0, 1500.0], runaway_ms=1800.0, aborted_ms=None, T=2500.0)
    assert v == "train_then_runaway" and is_working_point(v)


def test_classify_train_then_runaway_via_abort():
    # a real working point can also end via abort (sustained runaway detected online)
    v = classify_workpoint([300.0, 700.0, 1100.0, 1500.0], runaway_ms=None, aborted_ms=1800.0, T=2500.0)
    assert v == "train_then_runaway"


def test_classify_train_no_runaway():
    v = classify_workpoint([300.0, 800.0, 1400.0, 2100.0], runaway_ms=None, aborted_ms=None, T=2500.0)
    assert v == "train_no_runaway" and not is_working_point(v)


def test_classify_silent():
    assert classify_workpoint([], runaway_ms=None, aborted_ms=None, T=2500.0) == "silent"
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_stage4_v2_spontaneous.py -k classify -v`
Expected: FAIL — module/functions missing.

- [ ] **Step 3: Implement the classifier (no stray indentation)**

Create `scripts/run_stage4_v2_workpoint_search.py` (full header docstring per Task 3) containing:

```python
def classify_workpoint(event_ons, runaway_ms, aborted_ms, T, *, min_train=3, early_ms=200.0):
    """Verdict for a spontaneous big-focus run. A working point = a TRAIN of >= min_train discrete
    events with a DELAYED runaway (not an immediate all-or-nothing burst). An abort counts AS a
    runaway (it only fires on the shared sustained-runaway criterion); `aborted_ms` is the DETECTION
    time (inflated ~100 ms over onset by the sustained-window rule), so a lone ignition event that
    itself becomes the runaway shows up as n_pre==1 -- that is still a burst, not a train."""
    eff = runaway_ms if runaway_ms is not None else aborted_ms
    end = eff if eff is not None else T
    n_pre = sum(1 for t in event_ons if t < (end - 20.0))
    if eff is None:
        return "train_no_runaway" if n_pre >= min_train else "silent"
    if n_pre >= min_train:
        return "train_then_runaway"
    if eff <= early_ms or n_pre <= 1:
        return "one_shot_burst"
    return "few_events_then_runaway"                 # 2 .. min_train-1 events then runaway (near-miss)


def is_working_point(verdict):
    return verdict == "train_then_runaway"
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_stage4_v2_spontaneous.py -k classify -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_stage4_v2_workpoint_search.py tests/test_stage4_v2_spontaneous.py
git commit -m "feat(stage4-v2): working-point classifier (abort==runaway, train_then_runaway gate)"
```

---

### Task 3: Screener with SEPARATE `--stage fast` / `--stage confirm` (no auto long-run)

**Files:**
- Modify: `scripts/run_stage4_v2_workpoint_search.py`

**Interfaces:**
- Consumes: Task 1 build/sim (`H._build`, `H._simulate_continuous`, `H.ProtocolConfig`, `H._smooth_rate`, `H._first_sustained`), Task 2 classifier, and `run_sef_hfo_snn_cm_spontaneous_readout` (`C`) for `active_fraction`, `detect_events`, `BIN_MS`, `BASELINE_MS`, `CAL_FRAC`, `DT`, `_engine_guard`.
- **Timestep is LOCKED, not `H.DT`.** The companion file has NO top-level `DT` constant (verified: it always reads `S["p"].dt`), so referencing `H.DT` would crash. `run_one` derives `DT = float(S["p"].dt)` and asserts `abs(DT - C.DT) < 1e-12`, then uses that local `DT` for `active_fraction`, `_smooth_rate`, and `_first_sustained`.
- Produces: `run_one(...) -> dict`; CLI `--stage {fast,confirm}` (default `fast`), `--survivor-json PATH` (required for confirm), `--max-confirm N` (default 4). `fast` writes `results/topic4_sef_hfo/stage4_v2_workpoint/screen_fast.json` and EXITS after printing survivors + a wall-clock estimate; `confirm` reads that JSON and writes `screen_confirm.json` + `working_points`.

- [ ] **Step 1: Implement `run_one` (spontaneous, g_K-coupled, early-abort, shared criterion)**

```python
def run_one(*, L, core_mean, core_std, core_r, drive, k_q, tau_q, sigma_q,
            eta_K, k_K, tau_K, sigma_K, T, seed, abort=True):
    cfg = H.ProtocolConfig(layout="stage4_patch", top="qI", use_gK=True, use_hG=False,
                           eta_K=eta_K, k_K=k_K, tau_K=tau_K, sigma_K=sigma_K,
                           k_q=k_q, tau_q=tau_q, sigma_q=sigma_q, q_min=0.05,
                           core_mean=core_mean, core_std=core_std, core_radius=core_r,
                           drive=drive, L=L, T=T, n_pulses=0, seed=seed)
    S = H._build(cfg)
    DT = float(S["p"].dt)
    assert abs(DT - C.DT) < 1e-12                          # timestep LOCKED (companion has no H.DT)
    t0 = time.time()
    res = H._simulate_continuous(S, cfg, record_gif=False, vth=S["patch_vth"], abort_on_runaway=abort)
    spk = res["E_spk_bool"]
    rate_hz = np.asarray(res["rate_E"], float)
    af, bin_w = C.active_fraction(spk, DT, C.BIN_MS)
    nb0, nb1 = int(C.BASELINE_MS[0] / bin_w), int(C.BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    bar = floor + C.CAL_FRAC * (float(af.max()) - floor)
    event_ons = [float(e["t_on"]) for e in C.detect_events(af, bin_w, event_on_frac=bar)]
    runaway = H._first_sustained(H._smooth_rate(rate_hz, DT, 20.0), DT, 120.0, 100.0)   # shared criterion
    aborted = res.get("aborted_ms")
    eff = runaway if runaway is not None else aborted
    verdict = classify_workpoint(event_ons, runaway, aborted, cfg.T)
    return dict(L=L, core_mean=core_mean, core_r=core_r, drive=drive, k_q=k_q, tau_q=tau_q,
                eta_K=eta_K, k_K=k_K, tau_K=tau_K, seed=seed, T=T,
                n_events=len(event_ons),
                n_pre=sum(1 for t in event_ons if eff is None or t < eff - 20.0),
                runaway_ms=runaway, aborted_ms=aborted, effective_runaway_ms=eff,
                q_min_final=round(float(np.asarray(res["trace_qI_min"]).min()), 4),
                verdict=verdict, wall_s=round(time.time() - t0, 1))
```

- [ ] **Step 2: Implement the g_K-discretization grid + the two CLI stages**

```python
FAST_T = 900.0
FULL_T = 2500.0

GRID = [dict(core_mean=cm, eta_K=ek, tau_K=tk)
        for cm in (16.5, 17.0) for ek in (0.3, 0.5, 0.8) for tk in (150.0, 400.0)]
BASE = dict(L=20.0, core_std=1.5, core_r=6.0, drive=0.6, k_q=0.25, tau_q=5000.0,
            sigma_q=1.5, k_K=1.5, sigma_K=0.5, seed=1)
OUT = ROOT / "results" / "topic4_sef_hfo" / "stage4_v2_workpoint"


def _stage_fast():
    rows = []
    for gk in GRID:
        r = run_one(**{**BASE, **gk, "T": FAST_T, "abort": True})
        rows.append(r); print("FAST " + json.dumps(r), flush=True)
    survivors = [r for r in rows if r["verdict"] in
                 ("train_then_runaway", "train_no_runaway", "few_events_then_runaway")
                 and (r["aborted_ms"] is None or r["aborted_ms"] > 300.0)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen_fast.json").write_text(json.dumps({"base": BASE, "grid": GRID,
        "fast": rows, "survivors": survivors}, indent=2))
    est_hr = round(min(len(survivors), 4) * FULL_T / 900.0 * 0.33, 1)   # rough: ~1 hr per full-T run
    print(f"SURVIVORS {len(survivors)} / {len(rows)}", flush=True)
    print(f"CONFIRM_ESTIMATE up to {min(len(survivors),4)} runs ~ {est_hr} h "
          f"(run: --stage confirm --survivor-json {OUT/'screen_fast.json'})", flush=True)
    return 0 if survivors else 2


def _stage_confirm(survivor_json, max_confirm):
    data = json.loads(Path(survivor_json).read_text())
    survivors = data["survivors"][:max_confirm]
    rows = []
    for r in survivors:
        gk = dict(core_mean=r["core_mean"], eta_K=r["eta_K"], tau_K=r["tau_K"])
        c = run_one(**{**BASE, **gk, "T": FULL_T, "abort": True})
        rows.append(c); print("CONFIRM " + json.dumps(c), flush=True)
    working = [c for c in rows if is_working_point(c["verdict"])]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen_confirm.json").write_text(json.dumps({"base": BASE,
        "confirm": rows, "working_points": working}, indent=2))
    print("WORKING_POINTS " + json.dumps(working), flush=True)
    return 0 if working else 2


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fast", "confirm"], default="fast")
    ap.add_argument("--survivor-json", default=str(OUT / "screen_fast.json"))
    ap.add_argument("--max-confirm", type=int, default=4)
    a = ap.parse_args()
    os.chdir(ROOT)
    C._engine_guard()
    return _stage_fast() if a.stage == "fast" else _stage_confirm(a.survivor_json, a.max_confirm)


if __name__ == "__main__":
    raise SystemExit(main())
```

Header/imports at the top of the file: `import json, os, sys, time` + `from pathlib import Path`, `import numpy as np`, the same `sys.path` inserts as the pilot, `import plot_fig_m3a_v2_2_hG_runaway_transition_gif as H`, `import run_sef_hfo_snn_cm_spontaneous_readout as C`. **Do NOT reference `H.DT`** — the companion file has no top-level `DT` (verified: it always reads `S["p"].dt`). `run_one` derives `DT = float(S["p"].dt)` and asserts `abs(DT - C.DT) < 1e-12`; use that local `DT` everywhere.

- [ ] **Step 3: `py_compile`**

Run: `python -m py_compile scripts/run_stage4_v2_workpoint_search.py`
Expected: success, no output.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_stage4_v2_workpoint_search.py
git commit -m "feat(stage4-v2): fast/confirm CLI stages (default fast-only; no auto long confirm)"
```

---

### Task 4: Run FAST stage, then (only on user go) CONFIRM stage — COST CHECKPOINTS

**Files:** none (produces `screen_fast.json`, then `screen_confirm.json`).

- [ ] **Step 1: Launch the fast-reject stage in the background**

Run: `python scripts/run_stage4_v2_workpoint_search.py --stage fast > <scratch>/stage4_v2_fast.log 2>&1 &` (12 configs at `FAST_T=900` with early-abort; burst configs abort ~230 ms, train/silent are low-activity → cheap).

- [ ] **Step 2: Read `FAST ...` lines + `SURVIVORS n / 12` + `CONFIRM_ESTIMATE ...`**

If `SURVIVORS 0` → **GATE FAIL (fast).** STOP Phase 1. Report: the g_K-discretization bet did not yield any discrete-train candidate; the substrate stays burst-or-silent. Do NOT run confirm. Write the negative to `docs/archive/topic4/` and offer the fallbacks below.

- [ ] **Step 3: COST CHECKPOINT — get explicit user go before confirm**

Show the survivors + the printed `CONFIRM_ESTIMATE` (~1 hr per full-T run, up to 4). Do NOT launch confirm until the user says go. If the user wants fewer/other survivors, pass `--max-confirm N`.

- [ ] **Step 4: Launch the confirm stage (after go)**

Run: `python scripts/run_stage4_v2_workpoint_search.py --stage confirm --survivor-json results/topic4_sef_hfo/stage4_v2_workpoint/screen_fast.json --max-confirm <N> > <scratch>/stage4_v2_confirm.log 2>&1 &`

- [ ] **Step 5: Read `CONFIRM ...` + `WORKING_POINTS ...`; apply the go/no-go gate**

If `WORKING_POINTS []` → **GATE FAIL (confirm).** STOP, report negative, write archive, offer fallbacks. Otherwise select the working point with the most pre-runaway events and a runaway well after the first event; record its exact config (`core_mean, eta_K, tau_K, k_q, drive, L, core_r, seed`). This feeds Phase 2.

**If no working point (either gate fails):** honest outcome — the spontaneous big focus cannot produce the interictal train in the accessible range even with a g_K brake (consistent with the repo's "no robust spontaneous discrete-event regime" lesson). Fallbacks to OFFER (do not auto-pick): (b) keep the kick-driven `fig_m3a_v2_2_qI_stim_*` figures as the model's story; (c) small-core + weak-background variant (not a "big" focus); (d) revisit if a self-terminating E→E mechanism lands. Write the negative to `docs/archive/topic4/stage4_v2_workpoint_<date>.md`. Phase 2 does NOT run.

---

## Phase 2 — Stim GIF on the working point (only if Task 4 found one)

### Task 5: Stage-4 spontaneous stim-vs-no-stim comparison GIF (generator suppression)

**Files:**
- Create: `scripts/paper_figures/plot_fig_stage4_v2_spontaneous_stim_gif.py`
- Test: `tests/test_stage4_v2_spontaneous.py`

**Claim scope (LOCKED):** the stim clamps the single big-focus core = **external preventive suppression of the event generator / the q_I build-up-to-runaway driver.** This is NOT a propagation-barrier / midline-block claim (a single big focus has no A/B axis). All titles/README/metadata state exactly this; propagation-barrier questions belong to the kick-driven two-focus figures.

**Interfaces:**
- Consumes: `H._build`, `H._simulate_continuous` (`vth=S["patch_vth"]`, `n_pulses=0`, `stim_target=`, `stim_on=`, `stim_off=`), `H._activity_metrics`; from `Q = plot_fig_m3a_v2_2_qI_stim_runaway_gif`: `Q._zlfp`, `Q._activity_fields`, `Q._electrode_e_mask`.
- Produces: `_patch_contacts(S) -> (contacts, names)` (a short line of virtual SEEG contacts across the core so the readout has traces); `_stage4_stim_target(S, radius=2.0, n_contacts=4) -> (idx, contacts, mask)` where `mask` ⊆ `S["core_mask"]` and E-only (clamps the generator).

- [ ] **Step 1: Write the failing test for the stim target (core-subset, E-only)**

```python
@pytest.mark.slow
def test_stage4_stim_target_is_core_subset():
    import plot_fig_stage4_v2_spontaneous_stim_gif as G
    cfg = H.ProtocolConfig(layout="stage4_patch", top="qI", use_gK=True, eta_K=0.5, tau_K=200.0,
                           core_mean=17.0, core_std=1.5, core_radius=6.0, T=100.0, n_pulses=0, seed=1)
    S = H._build(cfg)
    idx, contacts, mask = G._stage4_stim_target(S, radius=2.0)
    is_E = np.asarray(S["labels"]) == 0
    assert mask.dtype == bool and mask.shape[0] == S["N"]
    assert (mask & ~is_E).sum() == 0                          # E cells only
    assert mask.sum() > 0 and (mask & ~S["core_mask"]).sum() == 0    # subset of the generator core
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_stage4_v2_spontaneous.py -k stage4_stim_target -v`
Expected: FAIL — module/function missing.

- [ ] **Step 3: Implement `_patch_contacts` + `_stage4_stim_target`**

```python
def _patch_contacts(S, n=11, pitch=None):
    """A short horizontal virtual-SEEG line through the sheet centre spanning the core, so the
    readout panel has traces. Names C0..C{n-1}."""
    L = float(S["L"]); c = np.asarray(S["center"], float)
    pitch = pitch if pitch is not None else (2.0 * float(S["layout"]["core_r"]) / max(n - 1, 1))
    xs = c[0] + (np.arange(n) - (n - 1) / 2.0) * pitch
    contacts = np.column_stack([xs, np.full(n, c[1])])
    names = [f"C{i}" for i in range(n)]
    return contacts, names


def _stage4_stim_target(S, radius=2.0, n_contacts=4):
    contacts, names = _patch_contacts(S)
    center = np.asarray(S["center"], float)
    idx = np.sort(np.argsort(np.linalg.norm(contacts - center, axis=1))[:n_contacts])
    is_E = np.asarray(S["labels"]) == 0
    mask = Q._electrode_e_mask(S["net"]["pos"], is_E, contacts[idx], radius) & S["core_mask"]
    return idx, contacts[idx], mask
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_stage4_v2_spontaneous.py -k stage4_stim_target -v`
Expected: PASS.

- [ ] **Step 5: Implement the two-arm run at the working-point config (defaults from Task 4)**

Hard-code the Task-4 working-point config as CLI defaults (all overridable). Build S once; run baseline `H._simulate_continuous(S, cfg, record_gif=True, vth=S["patch_vth"])` and stim `H._simulate_continuous(S, cfg, record_gif=True, vth=S["patch_vth"], stim_target=mask, stim_on=ON, stim_off=OFF)`. Pick `[ON, OFF]` to cover the q_I danger zone (after ~2-3 spontaneous events, before the baseline runaway from Task 4). Both arms are byte-identical until `stim_on` (clamp only changes the threshold comparison — no extra rng).

- [ ] **Step 6: Implement the single-core render**

Write `_draw_arm_single(fig, row_spec, S, res, metrics, cfg, qi, frame_steps, q_frames, activity_field, zlfp, activity_vmax, *, row_title, tm_cursor, stim_contacts=None, stim_on=None, stim_off=None, baseline_qmean=None)` — a copy of `Q._draw_arm`'s three columns (`permissivity(1-q_I)` | `2D activity` | (`q_I/g_K trace` | `readout`)) but drawing ONE core circle (label "C") and NO A/B axis ellipse. Reuse `Q._zlfp`, `Q._activity_fields`, and the `_render_frames` frame loop structure (2 rows: no-stim / stim). Output dir `results/paper-ready-figure/fig_stage4_v2_spontaneous_stim/figures/`. Footer states the generator-suppression claim scope.

- [ ] **Step 7: `py_compile` + fast tests**

Run: `python -m py_compile scripts/paper_figures/plot_fig_stage4_v2_spontaneous_stim_gif.py && python -m pytest tests/test_stage4_v2_spontaneous.py -m "not slow" -q`
Expected: compile OK; fast tests PASS.

- [ ] **Step 8: COST CHECKPOINT + generate the GIF + EYEBALL**

Two full-T spontaneous arms ≈ 1-2 hr — report the estimate and get user go before launching (background). Then read metrics; open the final PNG and verify: baseline shows a spontaneous TRAIN → q_I decline → runaway; stim arm holds q_I higher during the window and delays/prevents runaway; arms identical before `stim_on`. If the visual contradicts the metrics, debug before claiming success.

- [ ] **Step 9: Commit**

```bash
git add scripts/paper_figures/plot_fig_stage4_v2_spontaneous_stim_gif.py tests/test_stage4_v2_spontaneous.py
git commit -m "feat(stage4-v2): spontaneous big-focus q_I/g_K stim-vs-no-stim GIF (generator suppression)"
```

---

### Task 6: Docs + archive

**Files:**
- Modify: `results/FIGURE_INDEX.md`, `scripts/paper_figures/README.md`
- Create: `results/paper-ready-figure/fig_stage4_v2_spontaneous_stim/figures/README.md` (auto-written by the script)
- Create: `docs/archive/topic4/stage4_v2_workpoint_<date>.md` (working point + screen results, OR the negative)

- [ ] **Step 1: Write the archive doc** — the fast/confirm screen results, the chosen working point (config + why), or the negative outcome + which fallback was recommended.

- [ ] **Step 2: Add the FIGURE_INDEX row (Topic 4) + a paper_figures/README section** — describe the working point, the two-timescale mechanism (g_K discretizes per-event, q_I builds up across events), and the stim result with the generator-suppression claim scope. Stage ONLY these rows (the repo has parallel topic5 FIGURE_INDEX edits — use the `git apply --cached` single-hunk pattern from the prior stim commits).

- [ ] **Step 3: Commit**

```bash
git add results/FIGURE_INDEX.md scripts/paper_figures/README.md docs/archive/topic4/stage4_v2_workpoint_<date>.md
git commit -m "docs(stage4-v2): archive + index/README for spontaneous big-focus stim GIF"
```

- [ ] **Step 4: Memory** — do NOT edit memory as a plan step. If the user asks, update `project_topic4_m3a_v2_2_qI_stim_runaway_2026-06-29.md` with the working point / negative.

---

## Self-Review

- **Spec coverage:** working-point search (Tasks 2-4) ✔; q_I + g_K both in play, g_K coupled as fast brake (Global Constraints + Task 3 grid) ✔; spontaneous no-kick (Task 1 `n_pulses=0` + pulse-mask/`_source_xy` single-focus fix) ✔; final output = stim GIF (Task 5) ✔; cost control (early-abort Task 1 + fast/confirm split Task 3 + checkpoints Task 4) ✔; go/no-go with honest fallback (Task 4) ✔; runner-bug avoidance (Task 1 direct build) ✔.
- **P0 fixed:** Task 1 Steps 5-8 add a dedicated test + fix so `stage4_patch` (single focus, `n_pulses=0`) never accesses `foci[1]` — via `_source_xy` single-focus return AND a pulse-driven mask loop.
- **P1 cost gate fixed:** Task 3 splits `--stage fast` (default, exits after estimate) from `--stage confirm` (separate invocation); Task 4 has explicit user-go checkpoints. No auto fast→confirm.
- **P1 unified runaway:** one criterion (`_smooth_rate` + `_first_sustained`, `RUNAWAY_HZ=120`, `RUNAWAY_DUR_MS=100`) used by abort (Task 1 Step 11), classifier (Task 2), and `run_one` (Task 3).
- **P1 abort==runaway:** classifier uses `effective_runaway = runaway_ms or aborted_ms` (Task 2), with a dedicated test.
- **P1 DT source LOCKED:** companion file has no `H.DT`; `run_one` derives `DT = float(S["p"].dt)` and asserts `abs(DT - C.DT) < 1e-12` (Task 3 interface + body + header). No `H.DT` anywhere.
- **P1 Phase 2 wording:** claim scope LOCKED to generator/build-up suppression, NOT propagation barrier (Global Constraints + Task 5 header + footer).
- **Engineering fixes:** classifier has no stray indentation; Task 1 build test has no dangling placeholder (final asserts inline); Task 6 writes archive, not memory (memory only on explicit user ask).
- **Type consistency:** `_build` returns `patch_vth`/`core_mask` (Task 1) consumed by Task 3 `run_one` + Task 5 target/render; `classify_workpoint`/`is_working_point` signatures match across Tasks 2-4; `_simulate_continuous` kwargs (`vth`, `abort_on_runaway`, `abort_check_every`) defined in Task 1, used identically in Tasks 3 & 5.
- **Risk called out:** Phase 1 may find no working point; the plan stops at the gate and does not fabricate Phase 2.
