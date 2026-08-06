# FCXR Stage D — Frozen Fast-Branch Map (D1) + Mode Analysis (D2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Any task whose deliverable is a function with multi-clause prose invariants (D1.3 injection, D1.5 classifier, D1.6 ε_c, D2.11 fold/Hopf classifier) MUST invoke `hfosp-deep-contract-verify` before writing the body.

**Goal:** Determine, under the accepted FCXR-RC1 substrate with all slow variables frozen, whether the fast E–I system possesses a finite, stable, repeatably-enterable high-activity branch (fixed point or bounded orbit) along the frozen failure coordinate D — and if so, characterize its bifurcation/mode structure. Stop for review before any dynamic Z/M/X.

**Architecture:** Two mutually-checking evidence layers. **(D1) SNN empirical branch map (load-bearing):** at each frozen `z_i(D)=clip(1−D·p_i,0,1)`, run the real 40k-neuron RC1 SNN from two initial conditions — native-low and a deterministic kicked-then-released high state — and classify the outcome (8-label, two-layer). Coexistence across D = the fold signature; ε_c only bisected at candidate D. **(D2) Reduced-operator analysis (mechanism hypothesis):** at the three landmark states, reuse the existing coarse rate-field Jacobian (`build_jacobian_dense`/`rate_eigenpairs`/`numerical_abscissa`/finite-time gain) as one lens and the SNN-connectivity sech² operator (`effective_jacobian_modes`) as a complementary lens; name both distinctly; do not over-claim. D1 is the answer, D2 the explanation.

**Tech Stack:** Python 3, numpy/scipy; existing blessed SNN engine `src/snn_engine/kick_probe.py` (untouched, no re-bless), non-blessed slow plugin `src/snn_engine/mz_slow_vars.py`; reuse `src/topic4_mz_conductance.py` (observables), `src/topic4_mz_fcxr_modes.py` (IPR/effective operator), `src/topic4_m3b_spectral_phase.py` + `src/topic4_criticality.py` + `src/topic4_state_conditioned_susceptibility.py` (rate-field operators). Tests: pytest.

## Global Constraints (RC1 accepted base — copied verbatim; every task inherits)

- **Model = FCXR-RC1 arm C + Stage-C saturation:** external additive FF AMPA (`c_E·I_E^ff`, current) + recurrent E→E conductance toward `E_E=58` + recurrent-only smooth `tanh` saturation. Two-step equation (report §Stage-D verbatim):
  `g_rec_raw = c_E·I_E^rec/(E_E−V_match)` ; `g_rec_eff = g_sat·tanh(g_rec_raw/g_sat)` ; `τ_m V̇ = −V + c_E·I_E^ff + g_rec_eff·(E_E−V) + g_I(z)(E_I−V) + g_M(m)(E_K−V)`. Multiply by `(E_E−V)` **once** (g_rec_raw is already a conductance).
- **`g_sat = 21.6` LOCKED** (arm-C active-cell median peak `gErec`; not tuned). Sensitivity band ±20% = 17.3 / 25.9.
- **`dt = 0.05` fixed for all Stage D** (report mandate: dt-robust at *relative-to-same-dt* workpoint, but absolute event stats are NOT dt-converged; re-anchor at dt=0.05).
- **Seeds: primary `seed1`, confirmation `seed3`.** `seed4` = stress only, never in a primary denominator.
- **Substrate LOCKED:** `epilepsiae_1146`, montage `narrow`, `L=20.0`, `density=100.0`, `N=40000` (NE=32000), `drive=0.6`, `g=3.6`, `E_E=58.0`, `V_match=18.0`, `E_I=0.0`, `gaba_gain=1.125`. `use_m=False`, `use_phi=False`, `use_x=False` (x≡1), `use_z=False` (z held at the frozen field, see D1.3).
- **Off-by-default + no re-bless:** all new sim is gated by `--confirm-run`; the 6 blessed engine files (`kick_probe.py, params.py, model.py, connectivity.py, connectivity_rot.py, lfp.py`) are NOT edited (all new code lives in `mz_slow_vars.py` [non-blessed], new module, runner). The bless test `test_engine_blessed_fcxr` must stay green.
- **Resource discipline:** `OMP_NUM_THREADS=1`; ≤2 workers (1 preferred), RAM-gated; launcher `flock`; **no full `T×N` array on disk** — online spatial bins / O(N) per-cell summaries / ~4000-pt downsampled traces only (reuse `_small_trace`).
- **Results root:** `results/topic4_sef_hfo/mz_full_conductance_spatial_relay/fast_slow_dynamics/`. `runs/<ISO8601Z>_<pid>_<gitshort>_<label>/`. `figures/README.md` in Chinese, `### <file>` + 2–4 sentences + trailing `**关注点**：`.
- **Commit style:** `feat|docs|test(topic4): FCXR-RC1 Stage D — <plain finding>`, end with the Co-Authored-By trailer. Off-by-default stated inline.

## Reconciliation with existing design-spec Stage 1

This plan IS "Stage 1" of `docs/superpowers/specs/2026-07-20-topic4-mz-full-conductance-spatial-relay-design.md` (§5). The FCXR-D1 message refines it; where they differ, the newer FCXR-D1 message wins, but these three locked spec details are carried in:
1. **Phenotype early-stop DISABLED** in the branch map (`early_stop_runaway=False`). The 120Hz/100ms flag is an *operational* runaway label, not numerical divergence; leaving it on would kill the finite-high state we are trying to detect. Only nonfinite / conductance-clip / `tau_eff` / memory stops remain, surfaced as `NUMERICAL_UNSAFE`.
2. **`x_clamp` sweep is deferred to D4** (D1 is `x=1` only). D1 asks "does a high branch exist at all?"; D4.3 later asks "what `x` kills it?".
3. **8-label classifier is a strict superset of the spec's 5** (low only→LOW_ONLY; finite high→FINITE_HIGH_FIXED; bounded orbit→FINITE_HIGH_ORBIT; bistable→BISTABLE; ceiling saturation→REFRACTORY_CEILING; new: METASTABLE_TRANSIENT, NUMERICAL_UNSAFE, UNRESOLVED).

We skip the spec's "first run natural Z-only trajectory" step because the RC1 workpoint is already established as interictal (only returning events); D0.2 re-confirms this once at dt=0.05, then D1 goes straight to the frozen-D controlled field.

## Decisions & Flags (my resolutions to plan ambiguities — VETOABLE at review)

- **[F1] Two-window dwell test — resolved to correct physics.** A true high attractor stays high to the window end → dwell tracks the window (grows with T). A metastable transient decays at τ_d → dwell saturates below the window. So: `FINITE_HIGH` ⇔ high state **present at window end at BOTH T1 and T2**; `METASTABLE_TRANSIENT` ⇔ had a high excursion but dwell **saturates** (`dwell(T2) ≤ (1+ε_dwell)·dwell(T1)`) and is absent at end. (The plan prose "dwell 不随观察窗增长 → finite high" reads inverted; I encode the physically-correct discriminator and flag it here.)
- **[F2] D2 operator = two complementary named lenses, not one grafted operator.** Coarse rate-field Jacobian (saturation implicit via `op.gE`) + SNN-connectivity sech² operator (`effective_jacobian_modes`). I do NOT graft `sech²` onto the coarse dense Jacobian (would double-count `op.gE`). D2 is a reduced-model lens on D1; it does not "prove" the SNN mechanism. (Deviates from the literal "new operator must add sech²" wording; §6.2 layer-match rationale in scratchpad.)
- **[F3] `p_i` per-seed self-consistent.** seed1 branch map uses the seed_1 snapshot's onset-depletion; seed3 confirmation uses seed_3's — each seed keeps a self-consistent substrate+field pair (mirrors how RC1 validated seed1 & seed3 independently). Primary snapshot set = `zA_q75_tz5000`; sensitivity = `zA_q50_tz10000`.
- **[F4] Windows (LOCKED, grounded, vetoable):** post-kick base window `T1_post = 4000 ms`, candidate re-run `T2_post = 8000 ms`. Kick at `t_kick=120 ms`, `DUR_KICK=18 ms` (engine constant), so total run = `t_kick + DUR_KICK + T_post`. `analysis_start_ms = t_kick + DUR_KICK = 138 ms`.
- **[F5] Classifier thresholds (LOCKED, relative to the dt=0.05 baseline from D0.2 — robust to dt):** see D1.5. All are relative to `baseline_rate / baseline_sigma / baseline_af_q95`, so they inherit dt-robustness.

## Reconciliation with the unsaturated slow-fast-transition line (2026-07-20)

The parallel line `codex/topic4-mz-slow-fast-transition` (baseline `codex/topic4-mz-onset-dynamics`, **current-mode membrane — NO recurrent conductance, NO smooth saturation**) froze the real per-neuron z_i/m_i and swept the failure coordinate on the 40k SNN. Result: below D≈0.08 the frozen fast system never runs away and no kick up to 0.20 ignites it; at **D≈0.087** it flips to spontaneous runaway (P=1, ε_c=0) — a **sharp position-controlled transition, finite-amplitude escape EXCLUDED**, on the *unsaturated* membrane. Its counterfactual: m does not move the boundary (only z/D) → justifies M-off, z-only here.

Implications (Stage D runs the *saturated* RC1 base = the complementary test):
- **D1 asks precisely:** does the RC1 smooth saturation convert that unbounded past-transition runaway into a **bounded** FINITE_HIGH branch/orbit, or merely delay it / pin at REFRACTORY_CEILING?
- **D grid refined near the transition:** `D = [0, 0.05, 0.075, 0.085, 0.09, 0.10, 0.125, 0.15]` (my D scalar ≈ mean-depletion, so it shares their coordinate; 0.075↔0.10 straddle D≈0.087).
- **ε_c caveat:** their ε_c = global threshold-lowering probe; mine = external kick amplitude — not directly comparable; only the qualitative "no reachable high below the transition" transfers.
- **Verdict discipline inherited:** operational runaway (120Hz/100ms) ≠ seizure; conservative result-neutral labels; no over-claim.

## Sprint staging (user choice 2026-07-20): PILOT-FIRST, INLINE

Build D0.1→D1.5 (TDD), then run a **3-point saturated pilot** (D≈0.075 below / 0.10 just above / 0.125 well above; seed1; native-low + kicked-high; early-stop OFF; T1 window). Evaluate the gate. Only if the pilot shows a bounded FINITE_HIGH above the transition do we proceed to the full grid (D1.7/D1.8) + ε_c (D1.6) + seed3 + D2. Execution is inline in-session with per-phase review checkpoints.

## HARD GATE

If the completed D1 grid yields **no `FINITE_HIGH_*` and no `BISTABLE`** cell on seed1 (i.e. all cells are `LOW_ONLY`, `REFRACTORY_CEILING`, `METASTABLE_TRANSIENT`, or `NUMERICAL_UNSAFE`), the mechanism is a **clean NO-GO**: do NOT open dynamic Z/M/X. D2 still runs (it documents *why* — α₁ stays negative / only ceiling jumps), and the sprint closes with a NO-GO verdict. A clean NO-GO is a legitimate, complete deliverable.

---

## File Structure

- **Create** `src/topic4_mz_fcxr_dynamics.py` — pure-logic + orchestration: `load_onset_depletion_pi`, `assert_field_substrate_aligned`, `frozen_z_field`, `branch_run_observables`, `classify_branch_run`, `classify_branch_D`, `epsilon_c_bisect`, `coarse_landmark_operator`, `snn_landmark_sech2`, `classify_landmark_dynamics`. (SNN cell execution is a thin wrapper around `simulate_kick`.)
- **Modify** `src/snn_engine/mz_slow_vars.py` (non-blessed) — add `z_frozen_E: np.ndarray|None = None` to `MZSlowVarsConfig` and honor it at `MZSlowVars.__init__` (initialize `self.z[:NE]=field`, no evolution when `use_z=False`). Byte-identical when `None`.
- **Create** `scripts/run_topic4_mz_fcxr_stage_d.py` — CLI: `branch-map` (D1 grid), `epsilon-c` (bisection at candidate D), `modes` (D2), plus `--confirm-run`, `--seed`, `--snapshot`, `--workers`, `--dry-run`. Reuses `build_substrate`, `_fc_cfg`, launcher flock, `_assert_engine_blessed`.
- **Create** `tests/test_topic4_mz_fcxr_dynamics.py` — unit tests (synthetic observable rows for the classifier; tiny-net sims for injection/runner).
- **Outputs** under `results/topic4_sef_hfo/mz_full_conductance_spatial_relay/fast_slow_dynamics/`: `branch_map.json`, `ignition_thresholds.json`, `mode_summary.json`, `figures/{README.md, frozen_branch_map.png, eigenmode_transition.png, ignition_and_recovery.png}`.
- **Archive** `docs/archive/topic4/sef_hfo/mz_fcxr_stage_d_branch_map_2026-07-20.md`; update the relay `STATUS.md` + `run_manifest.json` `stage_ledger`.

---

## Phase D0 — Re-anchor + alignment gate (fast; no branch map yet)

### Task D0.1: Locked `p_i` loader + substrate-alignment contract

**Files:** Create `src/topic4_mz_fcxr_dynamics.py`; Test `tests/test_topic4_mz_fcxr_dynamics.py`.

**Interfaces — Produces:**
- `load_onset_depletion_pi(snapshot_npz: str) -> dict` → `{p_i: (NE,) float64 mean-1 normalized, pos_E: (NE,2), vth_E: (NE,), src_xy, snk_xy, axis_unit, L}`. `p_i = dep/mean(dep)` where `dep = 1 - z_E[labels.index('onset')]`.
- `assert_field_substrate_aligned(pi_pack: dict, S: dict, *, atol_pos=1e-4, atol_vth=1e-4) -> None` — raises `ValueError` unless `pi_pack['pos_E'] ≈ S['posE']` and `pi_pack['vth_E'] ≈ S['vth'][:NE]` elementwise (§6 paired-key discipline: the field must map neuron-for-neuron onto the RC1 substrate, not by index luck).

- [ ] **Step 1 — failing test (loader):**
```python
import numpy as np, pytest
from src.topic4_mz_fcxr_dynamics import load_onset_depletion_pi
SNAP = "results/topic4_sef_hfo/state_conditioned_susceptibility/snapshots/zA_q75_tz5000/seed_1.npz"
def test_pi_is_mean_one_and_nonneg():
    pk = load_onset_depletion_pi(SNAP)
    assert pk["p_i"].shape == (32000,)
    assert np.isclose(pk["p_i"].mean(), 1.0, atol=1e-6)   # mean-depletion normalization
    assert (pk["p_i"] >= 0).all()
    assert pk["pos_E"].shape == (32000, 2)
```
- [ ] **Step 2 — run, expect FAIL** (`ImportError`). Run: `OMP_NUM_THREADS=1 python -m pytest tests/test_topic4_mz_fcxr_dynamics.py::test_pi_is_mean_one_and_nonneg -q`
- [ ] **Step 3 — implement `load_onset_depletion_pi`:**
```python
import numpy as np
def load_onset_depletion_pi(snapshot_npz):
    z = np.load(snapshot_npz, allow_pickle=True)
    labels = list(z["snapshot_labels"])
    onset = z["z_E"][labels.index("onset")].astype(np.float64)   # (NE,)
    dep = np.clip(1.0 - onset, 0.0, None)
    m = float(np.mean(dep))
    if not (m > 0):
        raise ValueError("onset depletion has non-positive mean; snapshot has no failure signal")
    return dict(p_i=dep / m, pos_E=z["pos_E"].astype(np.float64), vth_E=z["vth_E"].astype(np.float64),
                src_xy=z["src_xy"], snk_xy=z["snk_xy"], axis_unit=z["axis_unit"], L=float(z["L"]))
```
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — failing test (alignment gate) + tiny helper:** build the RC1 substrate and assert alignment; also assert a deliberately-shuffled field RAISES.
```python
from scripts.run_m4_phaseplane import build_substrate  # PP.build_substrate
from src.topic4_mz_fcxr_dynamics import assert_field_substrate_aligned
def test_pi_aligns_to_rc1_substrate_and_shuffle_is_rejected():
    pk = load_onset_depletion_pi(SNAP); S = build_substrate(seed=1)
    assert_field_substrate_aligned(pk, S)                       # must not raise
    bad = dict(pk); bad["pos_E"] = pk["pos_E"][::-1].copy()
    with pytest.raises(ValueError):
        assert_field_substrate_aligned(bad, S)
```
- [ ] **Step 6 — run, expect FAIL** (`assert_field_substrate_aligned` undefined).
- [ ] **Step 7 — implement:**
```python
def assert_field_substrate_aligned(pi_pack, S, *, atol_pos=1e-4, atol_vth=1e-4):
    NE = int(S["NE"]); posE = np.asarray(S["posE"], float)[:NE]; vthE = np.asarray(S["vth"], float)[:NE]
    if pi_pack["pos_E"].shape[0] != NE:
        raise ValueError(f"NE mismatch: field {pi_pack['pos_E'].shape[0]} vs substrate {NE}")
    if not np.allclose(pi_pack["pos_E"], posE, atol=atol_pos):
        raise ValueError("onset-depletion field pos_E does not match RC1 substrate posE (mis-registered field)")
    if not np.allclose(pi_pack["vth_E"], vthE, atol=atol_vth):
        raise ValueError("onset-depletion field vth_E does not match RC1 substrate vth (mis-registered field)")
```
- [ ] **Step 8 — run both alignment tests, expect PASS.** (If Step 8 fails on the real substrate, the snapshot is NOT RC1-substrate-consistent → **STOP and escalate**; the whole frozen-Z-from-snapshot approach needs a different field source. Do not weaken `atol` to force a pass.)
- [ ] **Step 9 — commit:** `git add -A && git commit -m "feat(topic4): FCXR-RC1 Stage D — locked onset-depletion p_i loader + substrate-alignment gate"` (+ trailer).

### Task D0.2: Re-anchor the dt=0.05 slow-off baseline reference

**Files:** `scripts/run_topic4_mz_fcxr_stage_d.py` (new, `baseline` subcommand); output `.../fast_slow_dynamics/baseline_ref.json`.

**Interfaces — Produces:** `baseline_ref.json` per seed with `{baseline_rate, baseline_sigma, baseline_af_q95, n_returning, duration_median, participation_median, peak_rate_median, dt, g_sat, af_bin_ms}` — the anchor every D1 classifier threshold is relative to.

**Rationale:** The report's P1-1 established the dt=0.05 workpoint holds *relative to its own reference* but absolute event stats drift (n_ret 28→12). So the classifier baseline MUST be the dt=0.05 slow-off reference, not the dt=0.1 one.

- [ ] **Step 1 — dry-run test:** the subcommand refuses without `--confirm-run` (mirror `test_runner_refuses_sim_without_confirm_run`).
```python
import subprocess, sys
def test_stage_d_baseline_refuses_without_confirm():
    p = subprocess.run([sys.executable, "scripts/run_topic4_mz_fcxr_stage_d.py", "baseline", "--seed", "1", "--T", "10"],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "confirm-run" in (p.stdout + p.stderr)
```
- [ ] **Step 2 — run, expect FAIL** (script missing).
- [ ] **Step 3 — implement the `baseline` subcommand:** build substrate (seed), `_fc_cfg(1.0, ff_conductance=False, rec_conductance=True, fail_on_clip=False, rec_sat_g=21.6)`, `dt=0.05`, slow-off event bar, no kick, `T` (default 8000 ms). Compute via reuse: `OLD.extract_run_metrics` / `compute_baseline_ref` / `slowoff_event_bar` (from `run_topic4_mz_slowvars.py`) and `active_fraction`/`peak_active_fraction` for `baseline_af_q95`. Persist `baseline_ref.json`. Assert `--confirm-run`, `_assert_engine_blessed()`, launcher flock, `OMP_NUM_THREADS=1`.
- [ ] **Step 4 — run the refuse test, expect PASS.**
- [ ] **Step 5 — REAL run (confirm):** `OMP_NUM_THREADS=1 python scripts/run_topic4_mz_fcxr_stage_d.py baseline --seed 1 --confirm-run --T 8000` then `--seed 3`. Verify against the report's dt=0.1 bands *directionally* (seed1: n_ret ≈ 12 at dt=0.05 per P1-1; participation and peak stay physiological). **Verify:** `baseline_ref.json` exists for seed1+seed3 with finite `baseline_rate/sigma/af_q95`.
- [ ] **Step 6 — commit:** `docs/feat(topic4): FCXR-RC1 Stage D — dt=0.05 slow-off baseline re-anchor (seed1+seed3)`.

---

## Phase D1 — Frozen fast-branch map (the load-bearing gate)

### Task D1.3: Frozen-Z field injection into the RC1 conductance run (non-blessed)

**Files:** Modify `src/snn_engine/mz_slow_vars.py`; `src/topic4_mz_fcxr_dynamics.py` (`frozen_z_field`); Test.

**Interfaces:**
- Produces `frozen_z_field(p_i, D) -> (NE,) = np.clip(1 - D*p_i, 0, 1)`.
- Adds `MZSlowVarsConfig.z_frozen_E: np.ndarray | None = None`; when set, `MZSlowVars.__init__` does `self.z[:self.NE] = z_frozen_E` (validated length NE, values in [0,1]); with `use_z=False` the field never evolves.

> **CONTRACT (invoke `hfosp-deep-contract-verify`):** (a) `z_frozen_E=None` ⇒ byte-identical to today (default path untouched — the existing 60 tests + parity fixture must stay green). (b) length==NE and 0≤z≤1 or raise. (c) `use_z=False` required (frozen, not evolving) — if `use_z=True` and `z_frozen_E` given, raise (ambiguous). (d) applies to the E block only (`self.z[:NE]`), I-block z untouched.

- [ ] **Step 1 — failing test (field math):**
```python
from src.topic4_mz_fcxr_dynamics import frozen_z_field
import numpy as np
def test_frozen_z_field_clips():
    p = np.array([0.0, 1.0, 2.0, 10.0]); z = frozen_z_field(p, 0.15)
    assert np.allclose(z, np.clip(1 - 0.15*p, 0, 1))
    assert z[0] == 1.0 and z[-1] == 0.0            # p_i=0 -> no depletion; large p_i -> full
```
- [ ] **Step 2 — run FAIL → Step 3 implement `frozen_z_field` → Step 4 PASS.**
- [ ] **Step 5 — failing test (injection byte-parity + freeze):** on a tiny net (`N=6, NE=4`, mirror `test_mz_full_conductance_spatial_relay._net`), assert `z_frozen_E=None` gives byte-identical `membrane_terms` output to today, and that a provided field is present on `self.z[:NE]` and unchanged after `step()` (use_z=False).
```python
def test_z_frozen_is_byte_identical_when_none_and_held_when_set():
    off = _mk_slow(z_frozen_E=None); on = _mk_slow(z_frozen_E=np.array([0.2,0.4,0.6,0.8]))
    assert np.array_equal(off.membrane_terms(*_inputs()), _reference_membrane_terms())  # None == today
    assert np.allclose(on.z[:4], [0.2,0.4,0.6,0.8])
    on.step(); assert np.allclose(on.z[:4], [0.2,0.4,0.6,0.8])                          # frozen
    with pytest.raises(ValueError): _mk_slow(z_frozen_E=np.array([0.2,0.4,0.6,0.8]), use_z=True)
```
- [ ] **Step 6 — run FAIL → Step 7 implement the `MZSlowVarsConfig`/`__init__` change (guard clauses per contract) → Step 8 PASS.**
- [ ] **Step 9 — regression:** rerun the full baseline suite; **all 60 must stay green** incl. `test_engine_blessed_fcxr` (we only touched the non-blessed plugin). Run: `OMP_NUM_THREADS=1 python -m pytest -q tests/test_fcxr_workpoint_gate.py tests/test_mz_conductance.py tests/test_topic4_mz_conductance.py tests/test_mz_full_conductance_spatial_relay.py`
- [ ] **Step 10 — commit:** `feat(topic4): FCXR-RC1 Stage D — frozen-Z field injection (off-by-default, byte-identical when None)`.

### Task D1.4: Single branch-map cell runner (frozen-Z + IC fork)

**Files:** `src/topic4_mz_fcxr_dynamics.py` (`run_branch_cell`); `scripts/run_topic4_mz_fcxr_stage_d.py`; Test (tiny net).

**Interfaces — Produces:** `run_branch_cell(S, *, D, pi, ic, kick_boost, T_post_ms, seed, g_sat=21.6, dt=0.05) -> dict` where `ic ∈ {"low","high"}`. Builds `_fc_cfg(1.0, ff_conductance=False, rec_conductance=True, fail_on_clip=False, rec_sat_g=g_sat)`, sets `z_frozen_E=frozen_z_field(pi, D)`, `dt=0.05`. Calls `simulate_kick(p, net, KICK_BOOST=(kick_boost if ic=="high" else 0.0), slow=slow, kick_center=S["src_xy"], r_kick=R_KICK, t_kick=(120.0 if ic=="high" else 1e9), early_stop_runaway=False)` (kick-then-release; **early-stop OFF**). Returns online observables only (no `E_spk_bool` persisted): `rate_E` (downsampled), `active_fraction` (bin_ms=5), `clip_frac_max`, `tau_eff_min`, `nonfinite`, `analysis_start_ms`, plus per-cell `g_raw` snapshot at the high plateau (O(NE), for D2).

> Note: kick location = source core (`src_xy`), radius `R_KICK` (engine constant 0.3) — the nucleation site; a fixed short pulse (`DUR_KICK=18ms`) then full release. `early_stop_runaway=False` per design-spec Stage 1.

- [ ] **Step 1 — failing test (tiny net, smoke):** `run_branch_cell` on `N=6`-scale substrate returns finite `rate_E`, an `active_fraction` array, and `analysis_start_ms==138.0`; `ic="low"` uses no kick (`t_kick` huge). Keep `T_post_ms` tiny (e.g. 100) for speed.
- [ ] **Step 2 — FAIL → Step 3 implement (thin wrapper; reuse `_fc_cfg`, `_make_slow`, `simulate_kick`, `active_fraction`, `_small_trace`) → Step 4 PASS.**
- [ ] **Step 5 — commit:** `feat(topic4): FCXR-RC1 Stage D — frozen-Z branch-map cell runner (kick-then-release, early-stop off)`.

### Task D1.5: Two-layer 8-label branch classifier (pure logic — TDD on synthetic rows)

**Files:** `src/topic4_mz_fcxr_dynamics.py` (`branch_run_observables`, `classify_branch_run`, `classify_branch_D`); Test.

> **CONTRACT (invoke `hfosp-deep-contract-verify`; §6.3 two-layer pronoun discipline):**
> **Per-run label** from one `(D, ic)` trajectory, computed from `oscillation_metrics(...)` + ceiling/clip checks + the two-window dwell field:
> - `NUMERICAL_UNSAFE` — `nonfinite` OR `clip_frac_max>0` OR `tau_eff_min < TAU_EFF_MIN` (=0.15 ms) OR memory stop. (Checked FIRST.)
> - `REFRACTORY_CEILING` — high AND `af_tail ≥ CEIL_FRAC` (=0.90) AND `modulation < MOD_CEIL` (=0.10): pinned, not an attractor.
> - `DECAYS_TO_LOW` — `tail_rate_band` True (last 2000 ms back inside `baseline_rate ± 1.5σ`).
> - `FINITE_HIGH_ORBIT` — persist-at-end AND `oscillatory_candidate` True.
> - `FINITE_HIGH_FIXED` — persist-at-end AND not orbit AND not ceiling.
> - `METASTABLE_TRANSIENT` — had a high excursion (`high_duration_ms ≥ MIN_HIGH_MS` =300) but NOT persist-at-end AND dwell saturates (`dwell_T2 ≤ (1+ε_dwell)·dwell_T1`, ε_dwell=0.25). [needs T1 and T2 runs → only decided for candidates; single-window rows get `DECAYS_TO_LOW`/`FINITE_HIGH_*` provisionally.]
> - persist-at-end := last-`END_WIN`(=500 ms) mean `rate > baseline_rate + K_HIGH·σ` (K_HIGH=4) AND `af_tail > baseline_af_q95`.
> **Per-D label** from `low` + the (≥2) `high` runs at one D:
> - any run `NUMERICAL_UNSAFE` → `NUMERICAL_UNSAFE`.
> - high runs disagree, or `high` plateaus differ by `> PLATEAU_TOL` (=0.20 rel) → `UNRESOLVED`.
> - `low→DECAYS_TO_LOW` AND `high→FINITE_HIGH_*` (concordant plateau) → `BISTABLE`.
> - all ICs (incl low) → `FINITE_HIGH_*` → `FINITE_HIGH` (monostable high).
> - all → `DECAYS_TO_LOW` → `LOW_ONLY`. high→`REFRACTORY_CEILING` → `REFRACTORY_CEILING`. high→`METASTABLE_TRANSIENT` → `METASTABLE_TRANSIENT`.
> Return `{per_run_label, evidence:{high_duration_ms, af_tail, modulation, tail_mean_hz, dwell_ms, oscillatory_candidate, clip_frac_max, tau_eff_min}}` and per-D `{D, D_label, low_label, high_labels, plateau_rel_spread}`. Every locked constant lives in a module-level `THRESHOLDS` dict with the values above.

- [ ] **Step 1 — failing tests (synthetic observable rows; NO sim):** one test per label, each a hand-built observable dict, asserting the returned label. Include: (i) a `settled` high row that persists at end at both windows → `FINITE_HIGH_FIXED`; (ii) same + oscillatory_candidate → `FINITE_HIGH_ORBIT`; (iii) af_tail 0.95 + modulation 0.02 → `REFRACTORY_CEILING`; (iv) tail_rate_band True → `DECAYS_TO_LOW`; (v) high excursion 400ms, absent at end, dwell_T2≈dwell_T1 → `METASTABLE_TRANSIENT`; (vi) clip_frac_max 0.01 → `NUMERICAL_UNSAFE`; (vii) per-D: low decays + two concordant high plateaus → `BISTABLE`; (viii) per-D: two high plateaus differing 40% → `UNRESOLVED`.
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement `classify_branch_run` + `classify_branch_D`** (reuse `oscillation_metrics`; the constants above in `THRESHOLDS`). Write `branch_run_observables(cell_result, baseline_ref, dwell_of_second_window=None)` to reduce a `run_branch_cell` result to the observable row `classify_branch_run` consumes.
- [ ] **Step 4 — run, expect PASS (all label tests).**
- [ ] **Step 5 — commit:** `feat(topic4): FCXR-RC1 Stage D — two-layer 8-label frozen-branch classifier (thresholds locked)`.

### Task D1.6: ε_c kick-amplitude bisection (candidate D only)

**Files:** `src/topic4_mz_fcxr_dynamics.py` (`epsilon_c_bisect`); Test.

> **CONTRACT (invoke `hfosp-deep-contract-verify`):** bisect `KICK_BOOST ∈ [lo, hi]` for the minimal amplitude whose released state is `FINITE_HIGH_*` (via `classify_branch_run`). `lo=0` (known low), `hi=kick_hi` (the D1 amplitude known to reach high). If `hi` does NOT reach high → return `{status:"no_high_at_hi", epsilon_c:None}` (do not extrapolate). Fixed `N_BISECT=6` iterations. Same RNG/seed each probe (deterministic). Returns `{epsilon_c, bracket:[lo,hi], n_bisect, status}`. **Only called at D cells labeled `BISTABLE`/`FINITE_HIGH`** — never a kick-amplitude grid over all D.

- [ ] **Step 1 — failing test with a FAKE `run_fn`:** inject a monotone threshold responder (`high` iff `kick ≥ 1.37`); assert `epsilon_c` brackets 1.37 within `2^-6·(hi-lo)`; and a responder that never goes high → `status=="no_high_at_hi"`.
- [ ] **Step 2 — FAIL → Step 3 implement (pure bisection over an injected `run_fn` classifier) → Step 4 PASS.**
- [ ] **Step 5 — commit:** `feat(topic4): FCXR-RC1 Stage D — ε_c kick bisection at candidate D (fail-closed when hi stays low)`.

### Task D1.7: `branch-map` CLI + dry-run plan

**Files:** `scripts/run_topic4_mz_fcxr_stage_d.py` (`branch-map` subcommand); Test.

**Behavior:** for `D in [0, 0.05, 0.075, 0.085, 0.09, 0.10, 0.125, 0.15]` × `ic in {low, high×2}`: `run_branch_cell` → `branch_run_observables` → write per-cell JSON + `_small_trace` npz under `runs/<stamp>/`; then per-D `classify_branch_D`; candidates (BISTABLE/FINITE_HIGH) get a `T2_post` re-run (dwell) + `epsilon_c_bisect`; landmark cells get seed3 confirmation. Assert `--confirm-run`, `_assert_engine_blessed`, flock, `OMP_NUM_THREADS=1`, `--workers` (default 1, max 2), RAM guard. `--dry-run` prints the cell plan + estimated cost and writes nothing.

- [ ] **Step 1 — failing test:** `branch-map --dry-run --seed 1` exits 0, prints 6 D values and 18 base cells, writes no files; without `--confirm-run` a real run refuses.
- [ ] **Step 2 — FAIL → Step 3 implement (reuse the mz_fcxr launcher flock + stamping + `_apply` patterns) → Step 4 PASS.**
- [ ] **Step 5 — commit:** `feat(topic4): FCXR-RC1 Stage D — branch-map CLI (D grid × IC fork, dry-run, OOM-safe)`.

### Task D1.8: LAUNCH the real D1 grid (background) + collect

- [ ] **Step 1 — pre-launch dry-run** to confirm the cell plan + cost estimate.
- [ ] **Step 2 — launch** `OMP_NUM_THREADS=1 python scripts/run_topic4_mz_fcxr_stage_d.py branch-map --seed 1 --confirm-run --workers 2 --T1 4000 --T2 8000` in the background (RAM-gated; hours). Poll; do NOT block.
- [ ] **Step 3 — seed3 confirmation** of any FINITE_HIGH/BISTABLE landmark D only.
- [ ] **Step 4 — aggregate** `branch_map.json` (per-D labels, evidence) + `ignition_thresholds.json` (ε_c per candidate). **Verify:** every D cell has a label; spot-check ≥2 cells' traces confirm the classifier read the real shape (not a threshold artifact).
- [ ] **Step 5 — GATE EVALUATION:** if no FINITE_HIGH/BISTABLE on seed1 → record clean NO-GO (proceed to D2 for the *why*, then close). Else record the finite-high landmark(s).
- [ ] **Step 6 — commit:** `docs(topic4): FCXR-RC1 Stage D — D1 frozen branch map result (<one-line verdict>)`.

---

## Phase D2 — Bifurcation + mode analysis (mechanism lens; cheap/offline)

### Task D2.9: Coarse rate-field lens across the D grid

**Files:** `src/topic4_mz_fcxr_dynamics.py` (`coarse_landmark_operator`); Test.

**Interfaces — Produces:** `coarse_landmark_operator(pi_pack, D, cfg) -> dict` = bin `frozen_z_field(pi,D)` to the coarse grid via `bin_neuron_state_to_grid` → `zbar_to_q` → `state_operator(zbar_field, grid, scaffold, ...)` (reuse `topic4_state_conditioned_susceptibility`) → `solve_branches` (low/high fork) → per branch: `build_jacobian_dense` → `rate_eigenpairs` (α₁=`eigenvalues[0].real`, left/right, `status`) → `numerical_abscissa(J)` → `directional_finite_time_gain_curve` → `ipr(mode_e_field(right[:,0], grid))`. Returns `{D, branches:[{reason, alpha1, num_abscissa, ft_gain_curve, lead_ipr, participation}]}`.

- [ ] **Step 1 — failing test:** `coarse_landmark_operator` at `D=0` returns a resolved low branch with finite `alpha1` and `num_abscissa ≥ alpha1` (Hermitian-part bound). Use the existing susceptibility grid config (`_CRIT_OP_GRID_N`).
- [ ] **Step 2 — FAIL → Step 3 implement (pure reuse; no new operator math) → Step 4 PASS.**
- [ ] **Step 5 — commit:** `feat(topic4): FCXR-RC1 Stage D — coarse rate-field lens over D grid (α₁/abscissa/gain/IPR)`.

### Task D2.10: SNN-connectivity sech² lens at landmarks

**Files:** `src/topic4_mz_fcxr_dynamics.py` (`snn_landmark_sech2`); Test.

**Interfaces — Produces:** `snn_landmark_sech2(net, NE, g_raw, g_sat=21.6) -> dict` = `build_W_EE(net,NE)` → `leading_modes(W_EE)` (raw IPR) + `effective_jacobian_modes(W_EE, g_raw, g_sat)` (sech² IPR). Returns `{raw_lead_ipr, eff_lead_ipr, sech2_min, sech2_mean, eff_eig_vals_abs}`. `g_raw` = the per-cell `g_raw` snapshot recorded by D1.4 at the landmark plateau.

- [ ] **Step 1 — failing test:** on a small synthetic sparse `W_EE` + `g_raw`, both raw and eff leading IPR are in `[1/N, 1]` and `sech2_mean ∈ (0,1]`. (Mirrors the P1-2 finding shape; no sim.)
- [ ] **Step 2 — FAIL → Step 3 implement (thin reuse of `topic4_mz_fcxr_modes`) → Step 4 PASS.**
- [ ] **Step 5 — commit:** `feat(topic4): FCXR-RC1 Stage D — SNN sech² connectivity lens at landmarks`.

### Task D2.11: Landmark dynamics classifier (fold / Hopf / non-normal / graded)

**Files:** `src/topic4_mz_fcxr_dynamics.py` (`classify_landmark_dynamics`); Test.

> **CONTRACT (invoke `hfosp-deep-contract-verify`):** combine the D1 per-D coexistence structure with the D2 coarse-operator readout — the label is a *mechanism hypothesis in the reduced model*, never a claim about the SNN:
> - `saddle_node_fold_candidate` — D1 shows low/high coexistence (BISTABLE across ≥1 D) AND ε_c falls approaching the fold AND a real eigenvalue → 0.
> - `hopf_candidate` — complex conjugate pair crosses Re=0 with nonzero Im AND FINITE_HIGH_ORBIT with amplitude growing continuously from small.
> - `non_normal_transient` — eigenvalues stable (α₁<0) BUT finite-time singular gain ≫1 AND the SNN high-IC returns to the same low (METASTABLE_TRANSIENT).
> - `graded_excitability` — no coexistence, no ceiling jump, α₁ rises monotonically without crossing (Z only shifts frequency, not basin).
> - `ceiling_cliff` — only LOW↔REFRACTORY_CEILING, no finite branch.
> Return `{label, reduced_model_evidence, snn_evidence, caveat:"reduced-model lens; D1 SNN is the load-bearing answer"}`.

- [ ] **Step 1 — failing tests (synthetic operator+branch summaries):** one per label from hand-built `{alpha1, eig_pair_im, ft_gain, D1_label, epsilon_c_trend}` dicts.
- [ ] **Step 2 — FAIL → Step 3 implement → Step 4 PASS.**
- [ ] **Step 5 — run D2 for real** on the D1 landmarks + write `mode_summary.json`. **Verify:** labels are self-consistent with `branch_map.json` (a `fold_candidate` requires a BISTABLE D in D1).
- [ ] **Step 6 — commit:** `feat(topic4): FCXR-RC1 Stage D — landmark dynamics classifier (fold/Hopf/non-normal/graded, reduced-model tier)`.

### Task D2.12: Figures + Chinese README

**Files:** `scripts/run_topic4_mz_fcxr_stage_d.py` (`modes --figures` or a small plotter); `.../fast_slow_dynamics/figures/{frozen_branch_map.png, eigenmode_transition.png, ignition_and_recovery.png, README.md}`.

**Panels (each answers one question — §7 discipline):**
- `frozen_branch_map.png` — per-D low vs high plateau rate (the SNN branch map) with per-D label + ε_c marks. Q: does a finite high branch exist across D?
- `eigenmode_transition.png` — α₁ and numerical abscissa vs D (coarse lens) + raw-vs-eff leading IPR at landmarks (SNN lens). Q: what does the operator say and is the spatial mode preserved?
- `ignition_and_recovery.png` — representative low-IC vs high-IC `rate_E` + `active_fraction` traces at the pivotal D (kick, release, persist-or-decay, tail band). Q: is entry finite and does release persist?

- [ ] **Step 1 — render the three PNGs from `branch_map.json`/`mode_summary.json`/traces.**
- [ ] **Step 2 — eyeball each** (paper-grade, self-contained: no §X/code-word axis labels; shared legend outside; per feedback memory). Fix and re-render if needed.
- [ ] **Step 3 — write `figures/README.md`** (Chinese, `### <file>` + 2–4 sentences + `**关注点**：`).
- [ ] **Step 4 — commit:** `docs(topic4): FCXR-RC1 Stage D — branch-map + mode figures + README`.

---

## Phase D-close — Verdict, archive, STOP for review

### Task D.13: Aggregate, archive, STATUS/manifest, verdict

- [ ] **Step 1 — write** `docs/archive/topic4/sef_hfo/mz_fcxr_stage_d_branch_map_2026-07-20.md` (§0 abstract in plain language per §8; §D0 re-anchor; §D1 branch map table + gate verdict; §D2 mode lens with the reduced-model caveat; resources).
- [ ] **Step 2 — update** the relay `STATUS.md` + `run_manifest.json` `stage_ledger.seizure_lifecycle`: from `NOT TESTED` to the D1 verdict (`finite_high_branch: FOUND@D=… | CLEAN NO-GO`), keeping the RC1 base ACCEPT + original NO-GO tiers intact. Update `head` to the new HEAD.
- [ ] **Step 3 — update memory** (agent-facing) with the Stage D outcome + branch name; append the results dir to `results/FIGURE_INDEX.md`.
- [ ] **Step 4 — full targeted regression** green (the 60 baseline + new `test_topic4_mz_fcxr_dynamics.py`); `test_engine_blessed_fcxr` green (engine untouched).
- [ ] **Step 5 — commit + STOP.** `docs(topic4): FCXR-RC1 Stage D — <verdict>; STOP for review (no dynamic Z/M/X)`. Do NOT proceed to D3+ without user sign-off.

---

## Self-Review

- **Spec coverage:** D0 (dt re-anchor + P1 closure cited) ✓; D1 frozen branch map with 8-label two-layer classifier + ε_c ✓; D2 eigen/non-normal via existing operators + sech² lens ✓; hard gate ✓; STOP-for-review ✓; x_clamp/Z-release/M/X explicitly deferred to D3+ ✓ (not in this sprint).
- **Placeholder scan:** classifier thresholds are concrete (`THRESHOLDS`); windows concrete (F4); no "TBD".
- **Type consistency:** `frozen_z_field`/`load_onset_depletion_pi`/`classify_branch_run`/`classify_branch_D`/`epsilon_c_bisect`/`coarse_landmark_operator`/`snn_landmark_sech2`/`classify_landmark_dynamics` names are used consistently across tasks; `p_i` is mean-1 throughout; `early_stop_runaway=False` everywhere in the branch map.
- **Biggest residual risk:** D0.1 Step 8 (snapshot↔RC1 substrate alignment). If it fails, STOP — the field source is wrong; do not weaken atol.
