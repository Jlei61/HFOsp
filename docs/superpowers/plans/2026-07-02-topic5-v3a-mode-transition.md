# Topic 5 V3a — Axis→Non-axis Mode-Transition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether, from late-preictal (P3) to early-ictal (I1), the seizure's most-amplifiable direction and activation flow move OFF the fixed interictal HFO axis onto non-axis contacts (H3a axial weakening — supportive; H3b non-axis flux amplification — primary; H3c mode transition — primary).

**Architecture:** New `src/topic5_v3_mode_transition.py` (pure math) + `scripts/run_topic5_v3_*.py` (orchestration) + `config/topic5_v3.yaml`. READ-ONLY reuse of V2 `scripts/_topic5_v2_crit_io.py` (`load_context`, cache, `load_subject_preictal`-style loaders) and `src/topic5_v2_criticality.py` (`var1_ridge`, `spectral_radius`, `cv_one_step_r2`, `block_shuffle_surrogate`, `phase_randomize_surrogate`, `contact_susceptibility`, `activations_from_z`, `avalanche_atm`, `branching_ratio`) + `src.interictal_propagation.load_subject_propagation_events` (interictal HFO participation). V3a builds its OWN nulls (shaft-spatial / rate-preserving / label). Own results subdir.

**Tech Stack:** Python, numpy, scipy (linalg svd/logm, stats.spearmanr), PyYAML, pytest, pandas/pyarrow.

## Global Constraints (copied from spec rev2 — every task inherits these)

- **EXPLORATORY tier.** No forecasting. No stand-alone critical-mode claim.
- **Primary transition contrast = P3→I1** (`Δ = median(I1) − median(P3)`). **O (±10 s) is buffer/descriptive/sensitivity ONLY — never in a primary Δ.**
- **eeg-onset anchored:** every window uses each seizure's `eeg_onset_rel` / `eeg_offset_rel` (NOT cache `relt=0`).
- **Primary metrics (one per hypothesis):** H3a `Δβ_axis_strength` (line-length-rate) <0, **supportive-only**; H3b `Δnet_offaxis_flux_surplus` >0 (**primary**); H3c `Δmode_shift` >0 (**primary**). **Support = H3b OR H3c significant; H3a alone can never define support.**
- **Non-normal:** H3c uses the dominant right **singular vector** of `A_lowrank^{k*}` (k*=3), NOT an eigenvector. Discrete reactivity = `σ_max(A^k)`; continuous only if `logm(A)` stable (renamed `reactivity_continuous_approx`).
- **Non-axis = pure interictal HFO participation** (data-blind to ictal outcome); H3c uses subspace projectors `P_A`/`P_N`, not a single `e_nonaxis`.
- **Dynamics 3-layer:** direct 2D VAR = operator primary (`λ_surplus`/gains/reactivity); low-rank all-clean VAR carries the H3c subspace mode-shift; full ridge-VAR = sensitivity only.
- **Avalanche H3b uses null-corrected `net_offaxis_flux_surplus`** (not raw); ATM excludes self-transitions (`i≠j`); lag0 common-drive control as sensitivity.
- **λ never reported raw** — always `λ_surplus` = obs − surrogate median.
- **Subject is the unit;** window→seizure→subject median. **narrow = primary cohort, broad = replication, NEVER pooled.**
- **Self-built nulls:** shaft-constrained-spatial / rate-preserving / axis-nonaxis-label. Empirical `p = (1+#exceed)/(1+n_perm)`; alignment two-sided, direction/trend one-sided.
- **Verdict = tier 0–5** (§Task 11); `state_v3_supported = tier ≥ 3`; V3a max tier 4.
- **`geometry_insufficient` (n_axis<5 OR n_nonaxis<3 OR no shaft with both) → flagged, NOT negative.**
- **onset jitter:** primary must hold under eeg_onset and ±10 s (±15 s stress). `k*=3`, DMD rank, participation/β thresholds are pilot-locked (Task 1).
- **Real-data scripts `@pytest.mark.integration` + `--outdir`; pure-fn tests plain pytest; default `n_perm_smoke`.**

---

## File Structure

- `config/topic5_v3.yaml` — phases, geometry, dynamics, avalanche, nulls, cohorts.
- `src/topic5_v3_mode_transition.py` — ALL pure math (event windows, geometry β/classify/subspace, surrogates, dynamics 2D/low-rank/singular-gain/reactivity, avalanche compartment flux).
- `scripts/run_topic5_v3_feasibility.py` — Task 1 pilot QC (gate).
- `scripts/run_topic5_v3_avalanche.py` — Task 6 (H3b).
- `scripts/run_topic5_v3_dynamics.py` — Task 8 (H3c + λ/gain/reactivity).
- `scripts/run_topic5_v3_susceptibility.py` — Task 9 (H3a supportive).
- `scripts/run_topic5_v3_summary.py` — Task 10 (tier).
- `scripts/plot_topic5_v3_summary.py` — Task 11.
- `tests/test_topic5_v3_mode_transition.py` (pure) + `tests/test_topic5_v3_integration.py` (`@pytest.mark.integration`).
- Outputs: `results/topic5_ictal_recruitment/v3_mode_transition/{narrow,broad}/`.

**Reuse note (DRY):** the per-subject preictal+ictal envelope loader mirrors V2 `_topic5_v2_crit_io.load_subject_preictal`; extend it to also return the ictal/offset segments (not just `relt<0`) via a `span=('pre_ictal_post')` option, OR add `load_subject_full_span()` in the io helper. Do NOT re-implement `load_context`/cache reading.

---

## Task 0: Config + module skeleton

**Files:** Create `config/topic5_v3.yaml`, `src/topic5_v3_mode_transition.py`; Test `tests/test_topic5_v3_mode_transition.py`.
**Interfaces:** Produces `load_v3_config(path=None)->dict`.

- [ ] **Step 1: Write config**

```yaml
# config/topic5_v3.yaml
phases:
  window_sec: 10.0
  step_sec: 5.0
  hop_sec: 0.1
  span_pre_sec: 120.0
  span_post_sec: 60.0
  P3_rel: [-30.0, -10.0]
  O_rel: [-10.0, 10.0]            # buffer only, never primary
  I1_rel: [10.0, 30.0]
  I1_min_usable_ictal_sec: 35.0
  I1_fallback_frac: 0.25          # I1=[+10, min(+30, +frac*duration)]
geometry:
  state_band: legacy_bb_1_45
  nonaxis_hfo_participation_max: 0.10   # OPEN pilot (Task 1)
  beta_axis_reliability_min: 0.20       # OPEN pilot
  min_n_axis: 5
  min_n_nonaxis: 3
dynamics:
  var_ridge_alpha: 1.0
  lowrank: 6                      # OPEN pilot (SVD/DMD rank)
  finite_horizon_k: 3             # k* primary
  finite_horizon_profile: [1, 2, 3, 5]
  surrogates: [phase_randomize, block_shuffle]
  block_len_sec: 2.0
avalanche:
  z_threshold: 2.0
  bin_sec: 0.1
nulls: {n_perm_smoke: 20, n_perm_dev: 100, n_perm_final: 1000, seed: 20260702, alpha: 0.05}
cohorts: {primary: narrow, replication: broad, never_pool: true}
jitter_sec: [5.0, 10.0, 15.0]
tier: exploratory
```

- [ ] **Step 2: Failing loader test**

```python
from src.topic5_v3_mode_transition import load_v3_config
def test_v3_config_keys():
    c = load_v3_config()
    assert c["phases"]["P3_rel"] == [-30.0, -10.0] and c["phases"]["I1_rel"] == [10.0, 30.0]
    assert c["dynamics"]["finite_horizon_k"] == 3
    assert c["cohorts"]["primary"] == "narrow" and c["cohorts"]["never_pool"] is True
    assert c["tier"] == "exploratory"
```

- [ ] **Step 3: Run fail. Step 4: Implement loader**

```python
from __future__ import annotations
from pathlib import Path
import numpy as np, yaml
_ROOT = Path(__file__).resolve().parents[1]
def load_v3_config(path=None) -> dict:
    with open(path or _ROOT / "config/topic5_v3.yaml") as fh:
        return yaml.safe_load(fh)
```

- [ ] **Step 5: Run pass. Step 6: Commit** — `git add config/topic5_v3.yaml src/topic5_v3_mode_transition.py tests/test_topic5_v3_mode_transition.py && git commit -m "feat(topic5-v3a): config + module skeleton"`

---

## Task 1: Feasibility pilot (gate — narrow cohort viable?)

**Files:** Create `scripts/run_topic5_v3_feasibility.py`; Test `tests/test_topic5_v3_integration.py`.
**Interfaces:** Produces `results/.../v3_mode_transition/feasibility.csv` cols: `subject, cohort, n_seizures, eeg_onset_rel_median, eeg_offset_rel_median, duration_median, usable_pre_sec, usable_ictal_sec, n_contacts_all_clean, n_axis, n_nonaxis, n_windows_P3, n_windows_I1, i1_eligible, geometry_sufficient`.
**Contract:** for each subject in `SUBJECTS_BY_SUB[cohort]`, use `load_context` + cache + `load_subject_propagation_events`; classify axis/non-axis at the config threshold; count P3/I1 windows per the phase design. **Print a cohort summary: how many narrow subjects are `geometry_sufficient AND i1_eligible`.** This gates whether narrow is a viable primary cohort and lets us pilot-lock `nonaxis_hfo_participation_max`, `beta_axis_reliability_min`, `lowrank`, `finite_horizon_k`.

- [ ] **Step 1: Failing integration test** (`--cohort narrow --outdir tmp`): CSV exists, has `geometry_sufficient` + `i1_eligible` columns, ≥1 row.
- [ ] **Step 2–4:** fail → implement (reuse io helper + Task 2/3 once they exist — for the pilot, inline minimal counts are OK; **re-run after Tasks 2–3 land to use the frozen definitions**) → pass.
- [ ] **Step 5: Commit** — `git commit -am "feat(topic5-v3a): feasibility pilot (geometry/time gate)"`
- [ ] **Step 6 (DECISION GATE):** inspect `feasibility.csv`; **pilot-lock** `nonaxis_hfo_participation_max`, `beta_axis_reliability_min`, `lowrank`, and confirm `finite_horizon_k=3`. Record the locked values in the config + this plan. If <4 narrow subjects are `geometry_sufficient AND i1_eligible`, STOP and report (narrow may not be a viable primary cohort → escalate).

---

## Task 2: Event windows (eeg-onset anchored; P3/O/I1; jitter) — pure

**Files:** Modify `src/topic5_v3_mode_transition.py`; Test same.
**Interfaces:**
- `i1_range(eeg_onset_rel, eeg_offset_rel, duration, cfg) -> (lo, hi, i1_eligible)` — `[onset+10, onset+30]` if `usable_ictal>=I1_min_usable_ictal_sec`, else fallback `[onset+10, onset+min(30, frac*duration)]`; `i1_eligible=False` if <1 full window fits.
- `phase_bin_range(relt, eeg_onset_rel, eeg_offset_rel, duration, phase, cfg, onset_shift=0.0) -> (start, stop) | None` — half-open sample indices for `phase in {P0,P1,P2,P3,O,I1,I2,I3,Post}`, anchored on `eeg_onset_rel+onset_shift` (P0..O,I1 relative to onset; I2/I3/Post relative to onset+ictal-fraction / offset).
- `sliding_windows(relt, start, stop, window_sec, step_sec) -> list[(ws, we)]` — 10 s / 5 s tiling with ≥3-sample guard.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v3_mode_transition import i1_range, phase_bin_range, sliding_windows, load_v3_config
def test_phase_bins_anchor_on_eeg_onset_not_relt_zero():
    cfg = load_v3_config()
    relt = np.round(np.arange(-120, 60.001, 0.1), 3)      # cache-frame axis
    onset, offset = -3.75, 202.0                           # eeg onset != relt 0 (139-like)
    p3 = phase_bin_range(relt, onset, offset, 205.0, "P3", cfg)
    i1r = phase_bin_range(relt, onset, offset, 205.0, "I1", cfg)
    assert relt[p3[0]] >= onset - 30 - 1e-6 and relt[p3[1]-1] <= onset - 10 + 1e-6
    assert relt[i1r[0]] >= onset + 10 - 1e-6                # anchored on onset, not 0
    lo, hi, ok = i1_range(onset, offset, 205.0, cfg)
    assert abs(lo - (onset + 10)) < 1e-6 and ok is True
    # jitter shifts the bin
    p3s = phase_bin_range(relt, onset, offset, 205.0, "P3", cfg, onset_shift=10.0)
    assert relt[p3s[0]] >= onset + 10 - 30 - 1e-6
    assert len(sliding_windows(relt, p3[0], p3[1], 10.0, 5.0)) >= 1
```

- [ ] **Step 2: Run fail. Step 3: Implement** (map each phase's `[lo_rel, hi_rel]` to `relt` indices via `onset_rel+onset_shift`; I2/I3 use ictal-fraction of `[onset, offset]`; guard empty). **Step 4: pass. Step 5: Commit** — `git commit -am "feat(topic5-v3a): eeg-onset-anchored phase windows + I1 + jitter"`

---

## Task 3: Geometry — signed β_axis + interictal-HFO non-axis + subspaces — pure

**Files:** Modify `src/topic5_v3_mode_transition.py`; Test same.
**Interfaces:**
- `rank_forward(ta_rank) -> {name: float}` — scale interictal `typical_rank` to −1 (early) … +1 (late).
- `beta_axis(metric_by_name, rank_forward) -> float` — signed Spearman on axis contacts (NaN if `<4`).
- `classify_contacts(all_clean, axis_template_names, hfo_participation, thresh) -> {is_axis, is_nonaxis_strict, n_axis, n_nonaxis}` — axis = in interictal HFO template (finite typical_rank / participation ≥ thresh); non-axis strict = clean, not-in-template, participation `< thresh`. **Data-blind to ictal.**
- `subspace_projectors(names, axis_names, nonaxis_names) -> (P_A, P_N)` — diagonal 0/1 selection matrices (n×n) onto axis / non-axis contacts.
- `axis_nonaxis_vectors(names, rank_forward, nonaxis_participation_map) -> (e_axis, e_nonaxis)` — `e_axis` from axis contacts weighted by `rank_forward`; `e_nonaxis` = non-axis participation-topography component, Gram-Schmidt-orthogonalized against `e_axis`; both unit. (2D viz + reduced VAR only.)
- `geometry_sufficient(n_axis, n_nonaxis, shafts_with_both, cfg) -> (bool, reason)`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v3_mode_transition import (rank_forward, beta_axis, classify_contacts,
    subspace_projectors, axis_nonaxis_vectors, geometry_sufficient, load_v3_config)
def test_beta_axis_signed_and_nonaxis_data_blind():
    cfg = load_v3_config()
    ta = {f"c{i}": float(i) for i in range(6)}          # early..late
    rf = rank_forward(ta); assert rf["c0"] == -1.0 and rf["c5"] == 1.0
    vals_late = {n: rf[n] for n in ta}                  # metric tracks late end
    assert beta_axis(vals_late, rf) > 0.9
    part = {**{n: 0.5 for n in ta}, "n0": 0.0, "n1": 0.02, "n2": 0.0}   # 3 non-axis
    cl = classify_contacts(list(part), list(ta), part, cfg["geometry"]["nonaxis_hfo_participation_max"])
    assert cl["n_axis"] == 6 and cl["n_nonaxis"] == 3
    PA, PN = subspace_projectors(list(part), [n for n in part if cl_is_axis(cl,n)], ["n0","n1","n2"])
    v = np.ones(len(part)); assert np.isclose((PN@v)@(PN@v), 3.0)
    ok, _ = geometry_sufficient(cl["n_axis"], cl["n_nonaxis"], 1, cfg)
    assert ok is True
```
*(helper `cl_is_axis` inline in test from the returned boolean map.)*

- [ ] **Step 2: Run fail. Step 3: Implement** (all pure; `subspace_projectors` returns `np.diag(mask)` per set; orthonormalize `e_nonaxis`). **Step 4: pass. Step 5: Commit** — `git commit -am "feat(topic5-v3a): signed beta_axis + interictal-HFO non-axis set + P_A/P_N subspaces"`

---

## Task 4: Self-built nulls — pure

**Files:** Modify `src/topic5_v3_mode_transition.py`; Test same.
**Interfaces:**
- `shaft_constrained_permute(values_by_name, shaft_by_name, rng) -> perm_values` — permute values within each shaft.
- `rate_preserving_shuffle(active_bool, rng) -> shuffled` — **preserve each contact's activation count/rate**, shuffle which TIME BINS it fires in (per-row independent permutation of the boolean row) → destroys cross-contact timing while keeping per-contact rate (the命门 null for flux).
- `label_permute(axis_names, nonaxis_names, shaft_by_name, rng) -> (perm_axis, perm_nonaxis)` — swap axis/non-axis labels matched within shaft + count.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v3_mode_transition import rate_preserving_shuffle, shaft_constrained_permute
def test_rate_preserving_keeps_row_counts():
    rng = np.random.default_rng(0)
    a = np.array([[1,1,0,0,0],[0,0,0,1,0],[1,1,1,0,0]], bool)
    s = rate_preserving_shuffle(a, rng)
    assert (s.sum(1) == a.sum(1)).all()                  # per-contact rate preserved
    assert s.shape == a.shape
def test_shaft_permute_stays_within_shaft():
    vals = {"A1":1.,"A2":2.,"A3":3.,"B1":9.}
    sh = {"A1":"A","A2":"A","A3":"A","B1":"B"}
    p = shaft_constrained_permute(vals, sh, np.random.default_rng(0))
    assert sorted([p["A1"],p["A2"],p["A3"]]) == [1.,2.,3.] and p["B1"] == 9.
```

- [ ] **Step 2: Run fail. Step 3: Implement.** **Step 4: pass. Step 5: Commit** — `git commit -am "feat(topic5-v3a): shaft-spatial / rate-preserving / label nulls"`

---

## Task 5: Avalanche compartment flux — pure

**Files:** Modify `src/topic5_v3_mode_transition.py`; Test same. **Reuse** V2 `activations_from_z`, `avalanche_atm`, `branching_ratio`.
**Interfaces:**
- `atm_offdiag(active_bool) -> ATM` — row-normalized ATM with `i≠j` (self-transitions zeroed) — wraps V2 `avalanche_atm` then zeros the diagonal and renormalizes.
- `atm_lag0(active_bool) -> M` — same-time coactivation `P(j@t | i@t)`, `i≠j` (common-drive control).
- `compartment_flux(atm, axis_idx, nonaxis_idx) -> {flux_A2N, flux_N2A, flux_A2A, flux_N2N}` — summed transition mass between/within compartments.
- `net_offaxis_flux(atm, axis_idx, nonaxis_idx) -> float` = `flux_A2N − flux_N2A`.
- `flux_surplus(obs, null_samples) -> (surplus, z)` — `obs − median(null)`, `(obs−median)/MAD`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v3_mode_transition import atm_offdiag, compartment_flux, net_offaxis_flux
def test_net_offaxis_flux_directional():
    # 0,1 axis ; 2,3 nonaxis ; drive axis->nonaxis
    z = np.zeros((4, 8))
    for t,(a,b) in enumerate([(0,2),(1,3),(0,2),(1,3)]): z[a,2*t]=3; z[b,2*t+1]=3
    from src.topic5_v2_criticality import activations_from_z
    atm = atm_offdiag(activations_from_z(z, 2.0))
    assert np.allclose(np.diag(atm), 0.0)
    assert net_offaxis_flux(atm, [0,1], [2,3]) > 0        # A->N dominates
```

- [ ] **Step 2: Run fail. Step 3: Implement. Step 4: pass. Step 5: Commit** — `git commit -am "feat(topic5-v3a): avalanche compartment flux (i!=j) + lag0 control + surplus"`

---

## Task 6: Avalanche run script (H3b primary) — integration

**Files:** Create `scripts/run_topic5_v3_avalanche.py`; Test integration.
**Interfaces:** Produces `v3_avalanche_subject.csv` cols: `subject, cohort, status, skip_reason, geometry_sufficient, n_axis, n_nonaxis, net_offaxis_flux_raw_P3, net_offaxis_flux_raw_I1, net_offaxis_flux_surplus_P3, net_offaxis_flux_surplus_I1, delta_net_offaxis_flux_surplus, net_offaxis_flux_z, p_rate, p_spatial, p_label, leak_index_delta, branching_N_delta, lag0_control_delta, onset_jitter_pass, n_seizures, tier`.
**Contract:** state-band z over P3 & I1 bins (eeg-onset anchored); threshold → activations → `atm_offdiag`; `compartment_flux`/`net_offaxis_flux` per bin per seizure → **median over seizures** (subject unit). **H3b primary = `delta_net_offaxis_flux_surplus` (surplus, rate-preserving-null-corrected, NOT raw), expect >0.** Nulls: rate-preserving (primary) + shaft-spatial + label. `onset_jitter_pass` from re-running at ±10 s. `geometry_insufficient` → `status=skipped` (not negative). broad + narrow separate.

- [ ] **Step 1: Failing integration test** (`--cohort narrow --subjects epilepsiae_958 --n-perm 20 --outdir tmp`): CSV has `delta_net_offaxis_flux_surplus`, `p_rate`, `geometry_sufficient`, `status`.
- [ ] **Step 2–4:** fail → implement → pass (narrow + broad). **Step 5: Commit** — `git commit -am "feat(topic5-v3a): avalanche off-axis flux run (H3b, rate-preserving null)"`

---

## Task 7: Dynamics pure — direct 2D VAR + low-rank + singular finite-time gain + reactivity

**Files:** Modify `src/topic5_v3_mode_transition.py`; Test same. **Reuse** V2 `var1_ridge`, `spectral_radius`.
**Interfaces:**
- `project_2d(X, e_axis, e_nonaxis) -> Z` — `Z = Q^T X`, `Q=[e_axis,e_nonaxis]`, shape `(2, n_t)`.
- `direct_2d_var(Z, alpha) -> B` — ridge VAR(1) on the 2×n_t series (reuse `var1_ridge`).
- `lowrank_var(X, rank, alpha) -> (A_lowrank, U)` — SVD `X≈U S V^T` top-`rank`; fit VAR in reduced coords; `A_lowrank` (rank×rank), `U` (n×rank) for map-back.
- `finite_time_gain(A, k) -> float` — `σ_max(A^k)` (largest singular value).
- `dominant_right_singular_vector(A, k) -> u1` — right singular vector of `A^k` for max σ.
- `subspace_mode_shift(u1_contact_space, P_N, P_A) -> float` = `‖P_N u1‖² − ‖P_A u1‖²` (u1 L2-normalized in contact space).
- `discrete_reactivity(A) -> {one_step_gain, ...}`; `continuous_reactivity_approx(A, dt) -> (val, logm_ok)` — `logm(A)/dt`, guard non-finite/complex-branch → `logm_ok=False`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v3_mode_transition import (project_2d, direct_2d_var, finite_time_gain,
    dominant_right_singular_vector, subspace_mode_shift, continuous_reactivity_approx)
def test_singular_gain_and_mode_shift_nonnormal():
    # non-normal A: stable eigenvalues but transient growth
    A = np.array([[0.5, 5.0],[0.0, 0.5]])
    assert max(abs(np.linalg.eigvals(A))) < 1.0            # eig-stable
    assert finite_time_gain(A, 1) > 1.0                    # yet σ_max>1 (transient amplify)
    u1 = dominant_right_singular_vector(A, 1); assert u1.shape == (2,)
    PN = np.diag([0.,1.]); PA = np.diag([1.,0.])
    ms = subspace_mode_shift(u1/np.linalg.norm(u1), PN, PA); assert -1.0 <= ms <= 1.0
    val, ok = continuous_reactivity_approx(A, 0.1); assert isinstance(ok, bool)
```

- [ ] **Step 2: Run fail. Step 3: Implement. Step 4: pass. Step 5: Commit** — `git commit -am "feat(topic5-v3a): 2D VAR + low-rank VAR + singular finite-time gain + reactivity"`

---

## Task 8: Dynamics run script (H3c primary + λ/gain/reactivity) — integration

**Files:** Create `scripts/run_topic5_v3_dynamics.py`; Test integration.
**Interfaces:** Produces `v3_dynamics_subject.csv` cols: `subject, cohort, status, skip_reason, geometry_sufficient, dynamics_primary_model, dynamics_support_model, rank_used, k_star, mode_shift_P3, mode_shift_I1, delta_mode_shift, mode_shift_2D_consistency, lambda_surplus_P3, lambda_surplus_I1, gain_axis_delta, gain_nonaxis_delta, reactivity_cont_available, logm_quality_flag, p_phase, p_block, onset_jitter_pass, cv_r2, var_meaningful_flag, n_ch_fit, n_seizures, tier`.
**Contract:** all-clean-contacts × time over P3 & I1 (eeg-onset anchored). **H3c primary = `delta_mode_shift` (from low-rank `A_lowrank^{k*}`, k*=3, dominant right singular vector → contact space → `subspace_mode_shift`), expect >0**, tested with phase+block surrogate. `λ_surplus`/gains from the direct 2D VAR (operator primary); `mode_shift_2D_consistency` = the 2D-VAR singular mode-shift as a cross-check. full ridge-VAR only if `--full-var-sensitivity`. `var_meaningful_flag = cv_r2 > 0`. median over seizures; narrow primary. `geometry_insufficient`/underpowered-window → skip-logged.

- [ ] **Step 1: Failing integration test** (`--cohort narrow --subjects epilepsiae_958 --n-perm 20 --outdir tmp`): CSV has `delta_mode_shift`, `lambda_surplus_I1`, `p_phase`, `mode_shift_2D_consistency`.
- [ ] **Step 2–4:** fail → implement → pass (narrow + broad). **Step 5: Commit** — `git commit -am "feat(topic5-v3a): dynamics run (H3c subspace mode-shift + lambda_surplus + surrogate nulls)"`

---

## Task 9: Susceptibility run script (H3a supportive) — integration

**Files:** Create `scripts/run_topic5_v3_susceptibility.py`; Test integration. **Reuse** V2 `contact_susceptibility`.
**Interfaces:** Produces `v3_susceptibility_subject.csv` cols: `subject, cohort, status, skip_reason, K_primary_metric, beta_axis_P3, beta_axis_I1, beta_axis_P3_reliable, delta_beta_axis_strength, beta_axis_delta_null_z, K_nonaxis_contrast_delta, p_spatial, p_label, onset_jitter_pass, n_seizures, tier`.
**Contract:** primary metric = **bb-envelope line-length rate**; `β_axis = beta_axis(metric_by_name, rank_forward)` on axis contacts in P3 & I1; **H3a primary = `delta_beta_axis_strength = |β_axis|_I1 − |β_axis|_P3`, expect <0, SUPPORTIVE-ONLY.** `beta_axis_P3_reliable = |β_axis|_P3 >= beta_axis_reliability_min` (else H3a not interpretable). Nulls: shaft-spatial + label. median over seizures; narrow primary.

- [ ] **Step 1: Failing integration test** (`--cohort narrow --subjects epilepsiae_958 --outdir tmp`): CSV has `delta_beta_axis_strength`, `beta_axis_P3_reliable`, `K_primary_metric=="line_length_rate"`.
- [ ] **Step 2–4:** fail → implement → pass. **Step 5: Commit** — `git commit -am "feat(topic5-v3a): susceptibility beta_axis run (H3a supportive + reliability gate)"`

---

## Task 10: Summary + tier verdict — integration

**Files:** Create `scripts/run_topic5_v3_summary.py`; Test integration.
**Interfaces:** Joins the 3 subject CSVs → `v3_summary_subject.csv` + `v3_cohort_tier.json` per cohort.
**Contract:**
- `subject_support = (H3b_sig OR H3c_sig) AND direction_correct AND matching_null_passed AND onset_jitter_pass AND (not single-contact driven) AND (axis-only cannot explain)`. **H3a significant only strengthens; never sole support.** `geometry_insufficient` → excluded from denominator, flagged.
- `tier`: 0 none / 1 descriptive-direction-only / 2 ≥1 subject support no cohort direction / 3 **narrow cohort primary support** (Wilcoxon signed-rank on subject-median H3b or H3c, direction correct, p<alpha) / 4 narrow + broad same-direction replication / (5 reserved for V3b). `state_v3_supported = tier>=3`. Cohort JSON: per-cohort subject-median H3b/H3c effect + sign test + bootstrap CI + n_support + tier + denominator (geometry-sufficient subjects). **narrow + broad never pooled.**

- [ ] **Step 1: Failing integration test** (summary CSV + cohort JSON have `tier`, `state_v3_supported`, `n_support`, per-cohort separation). **Step 2–4:** fail → implement → pass. **Step 5: Commit** — `git commit -am "feat(topic5-v3a): summary + tier 0-5 verdict (narrow primary, broad replication)"`

---

## Task 11: Result figure — integration

**Files:** Create `scripts/plot_topic5_v3_summary.py`; Test integration (smoke: PNG written).
**Contract:** 2–3 panels, each one independent question (CLAUDE.md §7), paper-grade self-contained (per `docs/figure_style_guide.md`; render→eyeball→fix): (A) per-subject P3→I1 Δ for H3b (`net_offaxis_flux_surplus`) and H3c (`mode_shift`), narrow vs broad, cohort-median bars, zero line; (B) trajectory P0→…→I3 of `mode_shift` / `net_offaxis_flux_surplus` (axis-weakening vs non-axis-amplification over phases; O shaded as buffer); optional (C) `λ_surplus` vs onset — showing surplus (not raw). Write `figures/README.md` (中文, 关注点). Append a line to `results/FIGURE_INDEX.md`.

- [ ] **Steps:** build; render; **eyeball the PNG**; fix; smoke test asserts PNG + README exist. **Commit** — `git commit -am "feat(topic5-v3a): result figure + README + FIGURE_INDEX"`

---

## Final run + Hard QC

- [ ] **Run (narrow primary first, then broad), after Task 1 pilot-lock:**

```bash
for ax in narrow broad; do
  python scripts/run_topic5_v3_feasibility.py     --cohort $ax
  python scripts/run_topic5_v3_avalanche.py       --cohort $ax --n-perm 1000
  python scripts/run_topic5_v3_dynamics.py        --cohort $ax --n-perm 1000
  python scripts/run_topic5_v3_susceptibility.py  --cohort $ax --n-perm 1000
  python scripts/run_topic5_v3_summary.py         --cohort $ax
done
python scripts/plot_topic5_v3_summary.py
pytest tests/test_topic5_v3_mode_transition.py -v
pytest -m integration tests/test_topic5_v3_integration.py -v
```

- [ ] **Hard QC:**
  - All windows eeg-onset anchored; **primary Δ is P3→I1; O never in a primary Δ** (only trajectory/sensitivity).
  - **Support = H3b OR H3c** (surplus-flux / subspace-mode-shift); **H3a never sole support**; H3a only interpretable if `beta_axis_P3_reliable`.
  - H3c uses **singular vector of `A_lowrank^{k*}`**, not eigenvector; subspace `P_N/P_A`, not single `e_nonaxis`.
  - Avalanche primary = **surplus** (rate-preserving-null-corrected), `i≠j`; lag0 control reported.
  - λ reported as **`λ_surplus`** only; discrete reactivity = `σ_max(A^k)`; continuous only if `logm_quality_flag`.
  - Nulls self-built; subject unit; **narrow primary, broad replication, never pooled**; `geometry_insufficient` flagged (≠ negative); jitter ±10 s stable.
  - Verdict = tier; `state_v3_supported = tier>=3`; V3a max tier 4.
  - All prose EXPLORATORY; no forecasting; no stand-alone critical-mode claim (needs V3b + Phase-1).

---

## Self-Review

1. **Spec coverage:** H3a→Task9; H3b→Task5/6; H3c→Task7/8; eeg-onset/P3→I1/O-buffer→Task2; β_axis + non-axis subspace→Task3; self-built nulls→Task4; surplus-flux→Task5/6; singular gain + discrete reactivity + low-rank→Task7/8; tier + narrow-primary→Task10; geometry-insufficient→Task1/3/10; jitter→Task2/6/8/9; figure→Task11; pilot-lock (thresholds/rank/k*)→Task1. **Covered.**
2. **Placeholder scan:** pure fns have literal tests+impl paths; run scripts give exact columns + contracts + commands; OPEN values are explicitly pilot-locked in Task 1 (not left vague). **OK.**
3. **Type consistency:** `rank_forward`→`beta_axis` (Task3)→Task9; `classify_contacts`/`subspace_projectors` (Task3)→Task7/8/10; `rate_preserving_shuffle` (Task4)→Task6; `atm_offdiag`/`net_offaxis_flux` (Task5)→Task6; `lowrank_var`/`dominant_right_singular_vector`/`subspace_mode_shift` (Task7)→Task8; reused V2 `var1_ridge`/`contact_susceptibility`/`avalanche_atm`/surrogates. **Consistent.**

---

## Execution Handoff

**Subagent-Driven (recommended)** — fresh subagent per task; **Task 1 feasibility is a pilot-lock gate (STOP if <4 narrow subjects geometry-sufficient+i1-eligible).** Order: 0→1(gate)→2→3→4→5→6(H3b)→7→8(H3c)→9(H3a)→10→11. **Dedicated worktree** (via `superpowers:using-git-worktrees`) recommended so V3a develops isolated from the other active worktrees.
