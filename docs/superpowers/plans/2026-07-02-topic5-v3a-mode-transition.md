# Topic 5 V3a — Axis→Non-axis Mode-Transition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **rev2 (post plan-review):** (1) feasibility pilot moved to run AFTER the frozen window+geometry modules (Task 3, not Task 1) — pilot-lock must see frozen definitions; (2) I1 short-seizure fallback fixed (offset-based, not `0.25·dur`); (3) H3c `mode_shift` **density-normalized** (÷ subspace rank) + **must pass label null** (dimension-bias fix); (4) low-rank singular vector → contact-space **map-back written explicitly**; (5) all VAR/DMD **demean within window** (mean-energy rise must not masquerade as coupling), no standardize; (6) `e_nonaxis` = **uniform mean** on non-axis contacts (participation topography ≈ 0 → unstable); (7) H3b/H3c primary **p-values computed on the Δ(I1−P3) null**, not per-phase-null-then-subtract; (8) source-normalized flux; (9) `single_contact_driven` + `axis_only_cannot_explain` **implemented as columns** (leave-one-contact + axis-only control); (10) H3b/H3c = two co-primary endpoints, **Holm-corrected at cohort level**; (11) contacts three-class (axis / non-axis-strict / ambiguous-HFO); (12) lag0 common-drive → `lag1_specific` downgrade; (13) **tier assigned only in summary** (module CSVs emit `module_support_flag` etc.); (14) two-tier integration tests (skipped-ok + eligible-runs); (15) `git add <files>` for new-file commits.

**Goal:** Test whether, from late-preictal (P3) to early-ictal (I1), the seizure's most-amplifiable direction and activation flow move OFF the fixed interictal HFO axis onto non-axis contacts (H3a axial weakening — supportive; H3b non-axis flux amplification — primary; H3c mode transition — primary).

**Architecture:** New `src/topic5_v3_mode_transition.py` (pure math) + `scripts/run_topic5_v3_*.py` (orchestration) + `config/topic5_v3.yaml`. READ-ONLY reuse of V2 `scripts/_topic5_v2_crit_io.py` (`load_context`, cache) and `src/topic5_v2_criticality.py` (`var1_ridge`, `spectral_radius`, `cv_one_step_r2`, `block_shuffle_surrogate`, `phase_randomize_surrogate`, `contact_susceptibility`, `activations_from_z`, `avalanche_atm`, `branching_ratio`) + `src.interictal_propagation.load_subject_propagation_events` (interictal HFO participation). V3a builds its OWN nulls. Own results subdir.

**Tech Stack:** Python, numpy, scipy (linalg svd/logm, stats.spearmanr/wilcoxon), PyYAML, pytest, pandas/pyarrow.

## Global Constraints (spec rev2 — every task inherits these)

- **EXPLORATORY tier.** No forecasting. No stand-alone critical-mode claim (needs V3b + Phase-1).
- **Primary contrast = P3→I1** (`Δ = median(I1) − median(P3)`). **O (±10 s) is buffer/descriptive/sensitivity ONLY.**
- **eeg-onset anchored** windows (each seizure's `eeg_onset_rel`/`eeg_offset_rel`, NOT cache `relt=0`).
- **Primary metrics:** H3a `Δβ_axis_strength` (line-length-rate) <0, **supportive-only**; H3b `Δnet_offaxis_flux_surplus` >0 (**co-primary**); H3c `Δmode_shift_density` >0 (**co-primary**). **Support = H3b OR H3c; H3a alone never supports.**
- **H3b/H3c p-values are computed on the Δ(I1−P3) permutation null** (not per-phase-null then subtract).
- **H3c mode_shift is density-normalized** (`‖P_N u‖²/rank(P_N) − ‖P_A u‖²/rank(P_A)`) AND **must pass the label null** (dimension-count control). `u` = dominant right **singular vector** of `A_lowrank^{k*}` (k*=3) mapped to contact space — NOT an eigenvector.
- **Non-axis = pure interictal HFO participation** (data-blind); contacts are **three-class** (axis / non-axis-strict / ambiguous-HFO); `P_A/P_N` use axis + non-axis-strict only.
- **All VAR/DMD demean within window** (`demean_within_window: true`), no standardize. Discrete reactivity = `σ_max(A^k)`; continuous only if `logm(A)` stable. λ never raw — always `λ_surplus`.
- **Avalanche flux source-normalized** (per-source mean) + null-corrected **surplus**; ATM `i≠j`; `lag1_specific` (lag1−lag0) downgrade for common drive.
- **Subject is the unit;** window→seizure→subject median. **narrow = primary cohort, broad = replication, NEVER pooled.**
- **Self-built nulls:** shaft-spatial / rate-preserving / label. `p=(1+#exceed)/(1+n_perm)`; alignment two-sided, direction/trend one-sided. **H3b/H3c are two co-primary endpoints → Holm-corrected at cohort level.**
- **Verdict = tier 0–5** (Task 10 only); `state_v3_supported = tier≥3`; V3a max tier 4. **Module scripts emit `module_support_flag/module_direction_correct/module_null_pass`, NOT tier.**
- **`geometry_insufficient` → flagged, NOT negative.** onset jitter ±10 s must hold. k*/rank/thresholds pilot-locked (Task 3).
- **New-file commits use `git add <files>` (not `-am`).** Real-data scripts `@pytest.mark.integration` + `--outdir`.

---

## File Structure

- `config/topic5_v3.yaml`; `src/topic5_v3_mode_transition.py` (all pure math); `scripts/run_topic5_v3_{feasibility,avalanche,dynamics,susceptibility,summary}.py`; `scripts/plot_topic5_v3_summary.py`; `tests/test_topic5_v3_mode_transition.py` (pure) + `tests/test_topic5_v3_integration.py`.
- Outputs: `results/topic5_ictal_recruitment/v3_mode_transition/{narrow,broad}/`.
- **Reuse (DRY):** extend the V2 io helper with `load_subject_full_span()` (pre-ictal + ictal + post) instead of re-reading `load_context`/cache.

**Execution order:** 0 → 1 → 2 → **3 (feasibility, after frozen 1+2)** → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11.

---

## Task 0: Config + module skeleton

**Files:** Create `config/topic5_v3.yaml`, `src/topic5_v3_mode_transition.py`; Test `tests/test_topic5_v3_mode_transition.py`.
**Interfaces:** `load_v3_config(path=None)->dict`.

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
  O_rel: [-10.0, 10.0]              # buffer only
  I1_rel: [10.0, 30.0]
  I1_min_duration_sec: 30.0         # primary I1 requires duration >= this
  I1_post_guard_sec: 2.0            # short fallback: [+10, offset - post_guard]
geometry:
  state_band: legacy_bb_1_45
  nonaxis_hfo_participation_max: 0.10   # OPEN pilot (Task 3)
  beta_axis_reliability_min: 0.20       # OPEN pilot
  min_n_axis: 5
  min_n_nonaxis: 3
dynamics:
  var_ridge_alpha: 1.0
  demean_within_window: true            # mean-energy rise must not become coupling
  standardize_within_window: false      # would erase real variance/gain
  lowrank: 6                            # OPEN pilot
  finite_horizon_k: 3                   # k* primary
  finite_horizon_profile: [1, 2, 3, 5]
  mode_shift_normalization: density     # density (÷rank) primary
  h3c_require_label_null: true
  surrogates: [phase_randomize, block_shuffle]
  block_len_sec: 2.0
avalanche:
  z_threshold: 2.0
  bin_sec: 0.1
  flux_normalization: source_mean       # per-source mean, not summed
  require_lag1_specific_positive: false # downgrade (not hard gate) initially
statistics:
  primary_pvalue_on_delta_null: true
  co_primary_correction: holm
  primary_endpoints: [H3b_delta_net_offaxis_flux_surplus, H3c_delta_mode_shift_density]
  single_contact_energy_frac_max: 0.50
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
    assert c["phases"]["I1_rel"] == [10.0, 30.0] and c["dynamics"]["finite_horizon_k"] == 3
    assert c["dynamics"]["demean_within_window"] is True and c["dynamics"]["mode_shift_normalization"] == "density"
    assert c["avalanche"]["flux_normalization"] == "source_mean"
    assert c["statistics"]["co_primary_correction"] == "holm" and c["cohorts"]["primary"] == "narrow"
```

- [ ] **Step 3: Run fail. Step 4: Implement loader** (`yaml.safe_load` of `_ROOT/config/topic5_v3.yaml`). **Step 5: pass. Step 6: Commit** — `git add config/topic5_v3.yaml src/topic5_v3_mode_transition.py tests/test_topic5_v3_mode_transition.py && git commit -m "feat(topic5-v3a): config + module skeleton"`

---

## Task 1: Event windows (eeg-onset anchored; P3/O/I1; jitter) — pure

**Files:** Modify `src/topic5_v3_mode_transition.py`; Test same.
**Interfaces:**
- `i1_range(eeg_onset_rel, eeg_offset_rel, duration, cfg) -> (lo, hi, i1_eligible)` — **primary `[onset+10, onset+30]` when `duration >= I1_min_duration_sec`**; else short fallback `[onset+10, offset − I1_post_guard]`; `i1_eligible = (hi−lo) >= window_sec` (≥1 full 10 s window). **Never `0.25·dur`.** Normalized I1 (ictal 0–25%) is a separate sensitivity path, not the fallback.
- `phase_bin_range(relt, eeg_onset_rel, eeg_offset_rel, duration, phase, cfg, onset_shift=0.0) -> (start, stop) | None` — half-open indices, anchored on `eeg_onset_rel+onset_shift`. P0..O,I1 relative to onset (I1 via `i1_range`); I2/I3 via ictal-fraction of `[onset, offset]`; Post relative to offset.
- `sliding_windows(relt, start, stop, window_sec, step_sec) -> list[(ws, we)]` — ≥3-sample guard.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v3_mode_transition import i1_range, phase_bin_range, sliding_windows, load_v3_config
def test_i1_short_seizure_fallback_is_usable():
    cfg = load_v3_config()
    lo, hi, ok = i1_range(0.0, 22.0, 22.0, cfg)            # 22 s seizure, offset=+22
    assert lo == 10.0 and hi == 20.0 and ok is True        # [+10, offset-2], one 10 s window
    lo2, hi2, ok2 = i1_range(0.0, 18.0, 18.0, cfg)         # offset-2=16 < lo+10 -> no window
    assert ok2 is False
    lo3, hi3, ok3 = i1_range(0.0, 205.0, 205.0, cfg)       # long -> primary [+10,+30]
    assert (lo3, hi3, ok3) == (10.0, 30.0, True)
def test_phase_bins_anchor_on_eeg_onset():
    cfg = load_v3_config(); relt = np.round(np.arange(-120, 60.001, 0.1), 3)
    p3 = phase_bin_range(relt, -3.75, 202.0, 205.0, "P3", cfg)
    assert relt[p3[0]] >= -3.75 - 30 - 1e-6 and relt[p3[1]-1] <= -3.75 - 10 + 1e-6
    p3j = phase_bin_range(relt, -3.75, 202.0, 205.0, "P3", cfg, onset_shift=10.0)
    assert relt[p3j[0]] >= -3.75 + 10 - 30 - 1e-6          # jitter shifts anchor
```

- [ ] **Step 2: Run fail. Step 3: Implement. Step 4: pass. Step 5: Commit** — `git commit -am "feat(topic5-v3a): eeg-onset phase windows + fixed I1 fallback + jitter"`

---

## Task 2: Geometry — signed β_axis + three-class contacts + subspaces — pure

**Files:** Modify `src/topic5_v3_mode_transition.py`; Test same.
**Interfaces:**
- `rank_forward(ta_rank) -> {name: float}` — interictal `typical_rank` scaled −1 (early)…+1 (late).
- `beta_axis(metric_by_name, rank_forward) -> float` — signed Spearman on axis contacts (NaN if `<4`).
- `classify_contacts(all_clean, axis_template_names, hfo_participation, thresh) -> {is_axis, is_nonaxis_strict, is_ambiguous_hfo, n_axis, n_nonaxis, n_ambiguous}` — **three-class**: axis = finite typical_rank OR in `axis_partition` source/mid/end; non-axis-strict = clean ∧ ¬axis ∧ `participation < thresh`; **ambiguous_hfo = clean ∧ ¬axis ∧ `participation ≥ thresh`** (in all-clean VAR X, but NOT in P_A/P_N).
- `subspace_projectors(names, axis_names, nonaxis_names) -> (P_A, P_N)` — `np.diag` 0/1 onto axis / non-axis-strict only.
- `axis_nonaxis_vectors(names, rank_forward, axis_names, nonaxis_names) -> (e_axis_mean, e_axis_grad, e_nonaxis_mean)` — **uniform means** (not participation topography): `e_axis_mean`/`e_nonaxis_mean` = unit-normalized uniform indicators; `e_axis_grad` = axis contacts weighted by `rank_forward`; `e_nonaxis_mean` Gram-Schmidt-orthogonalized against `e_axis_mean`.
- `geometry_sufficient(n_axis, n_nonaxis, shafts_with_both, cfg) -> (bool, reason)`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v3_mode_transition import classify_contacts, axis_nonaxis_vectors, subspace_projectors, load_v3_config
def test_three_class_and_uniform_nonaxis_vector():
    cfg = load_v3_config(); thr = cfg["geometry"]["nonaxis_hfo_participation_max"]
    part = {"a0":.5,"a1":.5,"a2":.5,"a3":.5,"a4":.5,"n0":.0,"n1":.02,"n2":.0,"amb":.4}
    cl = classify_contacts(list(part), ["a0","a1","a2","a3","a4"], part, thr)
    assert cl["n_axis"] == 5 and cl["n_nonaxis"] == 3 and cl["n_ambiguous"] == 1   # 'amb' high part, no rank
    names = list(part)
    e_am, e_ag, e_nm = axis_nonaxis_vectors(names, {n:0. for n in names},
                                            ["a0","a1","a2","a3","a4"], ["n0","n1","n2"])
    assert np.isclose(np.linalg.norm(e_nm), 1.0) and abs(e_am @ e_nm) < 1e-9   # unit + orthogonal
    assert np.allclose(e_nm[[names.index(n) for n in ["n0","n1","n2"]]], e_nm[names.index("n0")])  # uniform, not part-weighted
```

- [ ] **Step 2: Run fail. Step 3: Implement. Step 4: pass. Step 5: Commit** — `git commit -am "feat(topic5-v3a): beta_axis + three-class contacts + P_A/P_N + uniform non-axis vector"`

---

## Task 3: Feasibility pilot (gate — after frozen Task 1+2; pilot-lock)

**Files:** Create `scripts/run_topic5_v3_feasibility.py`; Test `tests/test_topic5_v3_integration.py`.
**Interfaces:** Produces `feasibility.csv` cols: `subject, cohort, n_seizures, eeg_onset_rel_median, eeg_offset_rel_median, duration_median, usable_pre_sec, usable_ictal_sec, n_contacts_all_clean, n_axis, n_nonaxis, n_ambiguous, n_windows_P3, n_windows_I1, i1_eligible, geometry_sufficient`.
**Contract:** uses the **frozen** Task-1 windows + Task-2 geometry (NOT temporary counts). Per subject: `load_context` + cache + `load_subject_propagation_events`; classify; count P3/I1 windows.

- [ ] **Step 1: Failing integration test** (`--cohort narrow --outdir tmp`): CSV has `geometry_sufficient`, `i1_eligible`, `n_ambiguous`; ≥1 row.
- [ ] **Step 2–4:** fail → implement (import frozen Task1/2 fns) → pass.
- [ ] **Step 5: Commit** — `git add scripts/run_topic5_v3_feasibility.py tests/test_topic5_v3_integration.py && git commit -m "feat(topic5-v3a): feasibility pilot (frozen defs, geometry/time gate)"`
- [ ] **Step 6 (DECISION GATE):** inspect `feasibility.csv`; **pilot-lock** `nonaxis_hfo_participation_max`, `beta_axis_reliability_min`, `lowrank`, `single_contact_energy_frac_max`; confirm `finite_horizon_k=3`; record locked values in config + this plan. **Auto-select one `geometry_sufficient AND i1_eligible` subject per cohort as the downstream integration subject.** If <4 narrow subjects qualify → STOP + report (narrow may be non-viable).

---

## Task 4: Self-built nulls — pure

**Files:** Modify `src/topic5_v3_mode_transition.py`; Test same.
**Interfaces:** `shaft_constrained_permute(values_by_name, shaft_by_name, rng)`; `rate_preserving_shuffle(active_bool, rng)` (per-row independent time-bin permutation → preserves each contact's rate); `label_permute(axis_names, nonaxis_names, shaft_by_name, rng)`.

- [ ] **Step 1: Failing test** (rate-preserving keeps `sum(1)` per row; shaft-permute stays within shaft — bodies as in spec). **Step 2–4:** implement. **Step 5: Commit** — `git commit -am "feat(topic5-v3a): shaft-spatial / rate-preserving / label nulls"`

---

## Task 5: Avalanche compartment flux (source-normalized) — pure

**Files:** Modify `src/topic5_v3_mode_transition.py`; Test same. **Reuse** V2 `activations_from_z`, `avalanche_atm`, `branching_ratio`.
**Interfaces:**
- `atm_offdiag(active_bool) -> ATM` (V2 `avalanche_atm` then zero diagonal + renormalize).
- `atm_lag0(active_bool) -> M` — same-time `P(j@t|i@t)`, `i≠j`.
- `compartment_flux(atm, axis_idx, nonaxis_idx, normalization) -> {flux_A2N, flux_N2A, flux_A2N_sum, flux_N2A_sum}` — **`source_mean` = mean over source i of Σ_j target ATM[i,j]** (primary); sum kept as descriptive.
- `net_offaxis_flux(atm, axis_idx, nonaxis_idx, normalization) -> float` = `flux_A2N − flux_N2A`.

- [ ] **Step 1: Failing test** (`net_offaxis_flux` >0 for a scripted A→N cascade; diagonal zero; source_mean invariant to non-axis count). **Step 2–4:** implement. **Step 5: Commit** — `git commit -am "feat(topic5-v3a): avalanche source-normalized compartment flux (i!=j) + lag0"`

---

## Task 6: Avalanche run (H3b co-primary; Δ-null p + gate columns) — integration

**Files:** Create `scripts/run_topic5_v3_avalanche.py`; Test integration.
**Interfaces:** `v3_avalanche_subject.csv` cols: `subject, cohort, status, skip_reason, geometry_sufficient, n_axis, n_nonaxis, n_ambiguous,`
`net_offaxis_flux_P3, net_offaxis_flux_I1, delta_net_offaxis_flux_raw, delta_net_offaxis_flux_surplus, net_offaxis_flux_z,`
`p_rate_delta, p_spatial_delta, p_label_delta, lag1_specific_delta, common_drive_sensitive,`
`max_source_contact_contribution, leave_one_contact_min_delta, leave_one_contact_pass, axis_only_flux_delta, axis_only_control_pass,`
`onset_jitter_pass, n_seizures, module_support_flag, module_direction_correct, module_null_pass`. **(no `tier` — Task 10 only.)**
**Contract:** flux per bin (source-normalized) per seizure → **median over seizures**. **`obs_delta = flux_I1 − flux_P3`; per perm compute `delta_perm = flux_I1_perm − flux_P3_perm`; `p_rate_delta = P(delta_perm ≥ obs_delta)`; `delta_surplus = obs_delta − median(delta_perm)`** (rate-preserving primary; spatial + label likewise). `lag1_specific_delta = (lag1−lag0)_I1 − (lag1−lag0)_P3`; `common_drive_sensitive = lag1_specific_delta <= 0`. `leave_one_contact_pass` = sign of `delta_surplus` survives dropping any single contact; `max_source_contact_contribution` for single-contact check; `axis_only_control` = flux recomputed with non-axis relabeled to axis. `module_support_flag = direction_correct ∧ p_rate_delta<alpha ∧ p_label_delta<alpha`. `geometry_insufficient`→`status=skipped`. narrow + broad separate; onset ±10 s jitter.

- [ ] **Step 1: Failing integration test (two-tier, per item 15):** `test_avalanche_writes_csv_even_if_skipped` (columns present) + `test_avalanche_runs_on_eligible_subject` (the Task-3 auto-selected subject → `status==ok`, `delta_net_offaxis_flux_surplus` finite).
- [ ] **Step 2–4:** fail → implement → pass (narrow + broad). **Step 5: Commit** — `git add scripts/run_topic5_v3_avalanche.py && git commit -m "feat(topic5-v3a): avalanche H3b (Δ-null surplus + leave-one-contact + axis-only + lag1-specific)"`

---

## Task 7: Dynamics pure — 2D VAR + low-rank map-back + singular gain + density mode-shift

**Files:** Modify `src/topic5_v3_mode_transition.py`; Test same. **Reuse** V2 `var1_ridge`, `spectral_radius`.
**Interfaces:**
- `demean_window(X) -> Xd` — subtract per-contact within-window mean (+ linear detrend); **no standardize**.
- `project_2d(X, e_axis, e_nonaxis) -> Z` (`Q^T X`, 2×n_t).
- `direct_2d_var(Z, alpha) -> B` (reuse `var1_ridge`).
- `lowrank_var(X, rank, alpha) -> (B_r, U_r)` — **`Xc=demean_window(X)`; `U,S,Vt=svd(Xc, full_matrices=False)`; `U_r=U[:,:rank]`; `q=U_r.T@Xc`; `B_r=var1_ridge(q, alpha)`** (latent = orthonormal `U_r` coords).
- `map_lowrank_vector_to_contacts(u_r, U_r) -> u_c` — `u_c = U_r @ u_r`; L2-normalize.
- `finite_time_gain(A, k) -> float` = `σ_max(A^k)`; `dominant_right_singular_vector(A, k) -> u1`.
- `subspace_mode_shift(u_contact, P_N, P_A, normalization) -> float` — **density**: `‖P_N u‖²/rank(P_N) − ‖P_A u‖²/rank(P_A)`; raw kept as descriptive.
- `discrete_reactivity(A)`; `continuous_reactivity_approx(A, dt) -> (val, logm_ok)`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v3_mode_transition import (lowrank_var, dominant_right_singular_vector,
    map_lowrank_vector_to_contacts, subspace_mode_shift, finite_time_gain, load_v3_config)
def test_lowrank_maps_to_contacts_and_density_mode_shift():
    rng = np.random.default_rng(0); X = rng.standard_normal((8, 300))
    B_r, U_r = lowrank_var(X, rank=3, alpha=1.0)
    u_r = dominant_right_singular_vector(B_r, k=3)
    u_c = map_lowrank_vector_to_contacts(u_r, U_r)
    assert u_c.shape == (8,) and np.isclose(np.linalg.norm(u_c), 1.0)
    PN = np.diag([0,0,0,0,0,1,1,1.]); PA = np.diag([1,1,1,1,1,0,0,0.])
    ms = subspace_mode_shift(u_c, PN, PA, "density")
    assert -1.0 <= ms <= 1.0
def test_singular_gain_nonnormal():
    A = np.array([[0.5, 5.0],[0.0, 0.5]])
    assert max(abs(np.linalg.eigvals(A))) < 1.0 and finite_time_gain(A, 1) > 1.0
```

- [ ] **Step 2: Run fail. Step 3: Implement. Step 4: pass. Step 5: Commit** — `git commit -am "feat(topic5-v3a): 2D VAR + low-rank map-back + singular gain + density mode-shift + demean"`

---

## Task 8: Dynamics run (H3c co-primary; Δ-null + label null + single-contact) — integration

**Files:** Create `scripts/run_topic5_v3_dynamics.py`; Test integration.
**Interfaces:** `v3_dynamics_subject.csv` cols: `subject, cohort, status, skip_reason, geometry_sufficient, dynamics_primary_model, dynamics_support_model, rank_used, k_star,`
`mode_shift_density_P3, mode_shift_density_I1, delta_mode_shift_density, mode_shift_raw_delta, mode_shift_2D_consistency,`
`p_phase, p_block, p_label, mode_shift_label_z,`
`lambda_surplus_P3, lambda_surplus_I1, gain_axis_delta, gain_nonaxis_delta, reactivity_cont_available, logm_quality_flag,`
`top_contact_energy_fraction, single_contact_driven, leave_one_contact_mode_shift_pass, axis_only_mode_shift_control, axis_only_control_pass,`
`onset_jitter_pass, cv_r2, var_meaningful_flag, n_ch_fit, n_seizures, module_support_flag, module_direction_correct, module_null_pass`. **(no `tier`.)**
**Contract:** all-clean X (incl. ambiguous) **demeaned within window**; `A_lowrank`+`U_r` per bin; `u_c` (k*=3) → **density** `mode_shift`. **`obs_delta = mode_shift_I1 − mode_shift_P3`; p on the Δ null for phase, block, AND label** (label null controls A/N contact counts). `H3c module_support` requires `direction ∧ p_phase<alpha ∧ p_block<alpha ∧ p_label<alpha`. `λ_surplus`/gains from the direct 2D VAR (`[e_axis_mean, e_nonaxis_mean]` primary; `e_axis_grad` sensitivity); `mode_shift_2D_consistency` = 2D-VAR singular mode-shift. `top_contact_energy_fraction = max(u_c²)/Σu_c²`; `single_contact_driven = > single_contact_energy_frac_max`. `axis_only_mode_shift_control` = mode_shift with N relabeled axis. median over seizures; narrow primary; skip-logged.

- [ ] **Step 1: Failing integration test (two-tier):** skipped-ok + eligible-runs (Task-3 subject → `delta_mode_shift_density`, `p_phase`, `p_label`, `mode_shift_2D_consistency` present + finite on ok).
- [ ] **Step 2–4:** fail → implement → pass. **Step 5: Commit** — `git add scripts/run_topic5_v3_dynamics.py && git commit -m "feat(topic5-v3a): dynamics H3c (density mode-shift, Δ-null phase/block/label, single-contact + axis-only)"`

---

## Task 9: Susceptibility run (H3a supportive) — integration

**Files:** Create `scripts/run_topic5_v3_susceptibility.py`; Test integration. **Reuse** V2 `contact_susceptibility`.
**Interfaces:** `v3_susceptibility_subject.csv` cols: `subject, cohort, status, skip_reason, K_primary_metric, beta_axis_P3, beta_axis_I1, beta_axis_P3_reliable, delta_beta_axis_strength, beta_axis_delta_null_z, p_spatial_delta, p_label_delta, onset_jitter_pass, n_seizures, module_support_flag(=False by construction — H3a supportive-only), module_direction_correct, module_null_pass`. **(no `tier`.)**
**Contract:** metric = **bb-envelope line-length rate**; `β_axis` on axis contacts P3 & I1; **H3a primary = `delta_beta_axis_strength = |β|_I1 − |β|_P3` (Δ-null p), expect <0, SUPPORTIVE-ONLY** (`module_support_flag` always False — H3a can never define support). `beta_axis_P3_reliable = |β|_P3 >= beta_axis_reliability_min`; if False → H3a not interpretable. median over seizures; narrow primary.

- [ ] **Step 1: Failing integration test (two-tier).** **Step 2–4:** implement. **Step 5: Commit** — `git add scripts/run_topic5_v3_susceptibility.py && git commit -m "feat(topic5-v3a): susceptibility beta_axis (H3a supportive + reliability gate + Δ-null)"`

---

## Task 10: Summary + tier verdict (Holm co-primary) — integration

**Files:** Create `scripts/run_topic5_v3_summary.py`; Test integration.
**Interfaces:** Joins 3 CSVs → `v3_summary_subject.csv` + `v3_cohort_tier.json` per cohort. **tier assigned ONLY here.**
**Contract:**
- `subject_support = (H3b.module_support_flag OR H3c.module_support_flag) AND onset_jitter_pass AND (not single_contact_driven) AND axis_only_control_pass AND (not common_drive_sensitive→downgrade)`. H3a significant only strengthens; never sole. `geometry_insufficient` excluded from denominator (flagged).
- **cohort-level: Holm-correct H3b and H3c p-values (2 co-primary endpoints) within each cohort;** narrow tier-3 needs Holm-corrected H3b OR H3c passing (Wilcoxon signed-rank on subject-median Δ, direction correct) + subject-support count.
- `tier`: 0 none / 1 descriptive-direction-only / 2 ≥1 subject support, no cohort direction / 3 **narrow cohort primary (Holm-passed)** / 4 narrow + broad same-direction replication / (5 = V3b). `state_v3_supported = tier>=3`. **narrow + broad never pooled.**

- [ ] **Step 1: Failing integration test** (summary + cohort JSON have `tier`, `state_v3_supported`, Holm-corrected p, per-cohort separation, denominator = geometry-sufficient count). **Step 2–4:** implement. **Step 5: Commit** — `git add scripts/run_topic5_v3_summary.py && git commit -m "feat(topic5-v3a): summary + tier 0-5 (Holm co-primary, narrow primary)"`

---

## Task 11: Result figure — integration

**Files:** Create `scripts/plot_topic5_v3_summary.py`; Test integration.
**Contract:** 2–3 independent-question panels (CLAUDE.md §7), paper-grade self-contained (`docs/figure_style_guide.md`; render→eyeball→fix): (A) per-subject P3→I1 Δ for H3b `net_offaxis_flux_surplus` and H3c `mode_shift_density`, narrow vs broad, cohort-median bars, zero line; (B) phase trajectory P0…I3 of `mode_shift_density` / `net_offaxis_flux` (O shaded buffer); (C optional) `λ_surplus` (not raw) vs onset. Write `figures/README.md` (中文, 关注点) + append `results/FIGURE_INDEX.md`.

- [ ] **Steps:** build; render; **eyeball PNG**; fix; smoke test (PNG + README exist). **Commit** — `git add scripts/plot_topic5_v3_summary.py && git commit -m "feat(topic5-v3a): result figure + README + FIGURE_INDEX"`

---

## Final run + Hard QC

- [ ] **Run (narrow primary first), after Task 3 pilot-lock:**

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

- [ ] **Hard QC:** windows eeg-onset anchored; **primary Δ = P3→I1** (O never in primary); **support = H3b OR H3c**, H3a never sole + only interpretable if `beta_axis_P3_reliable`; H3c = **density-normalized** singular-vector mode-shift **passing the label null**; H3b = source-normalized **surplus** with **Δ-null p**, `i≠j`, `lag1_specific` shown; VAR **demeaned within window**, no standardize; λ reported as **`λ_surplus`** only; `single_contact_driven` + `axis_only_control` + `leave_one_contact` computed (not asserted); **H3b/H3c Holm-corrected** at cohort level; tier only in summary; narrow primary / broad replication / never pooled; `geometry_insufficient` ≠ negative; jitter ±10 s stable; all prose EXPLORATORY.

---

## Self-Review

1. **Spec + review coverage:** feasibility-after-frozen-defs→order 0→1→2→3; I1 fallback→Task1; three-class→Task2; density mode-shift + label null→Task2/7/8; low-rank map-back→Task7; VAR demean→Task0/7; uniform e_nonaxis→Task2; Δ-null p→Task6/8; source-normalized flux→Task5; single-contact/axis-only→Task6/8/10; Holm co-primary→Task10; lag1-specific→Task6; tier-only-in-summary→Task6/8/9/10; two-tier integration→Task3/6/8/9; `git add` new-file→all. **Covered.**
2. **Placeholder scan:** pure fns literal tests+impl; run scripts exact columns+contracts; OPEN values pilot-locked at Task 3. **OK.**
3. **Type consistency:** `rank_forward`/`beta_axis`/`classify_contacts`/`subspace_projectors`/`axis_nonaxis_vectors` (Task2)→Task3/6/7/8/9; `rate_preserving_shuffle` (Task4)→Task6; `compartment_flux`/`net_offaxis_flux` (Task5)→Task6; `lowrank_var`/`map_lowrank_vector_to_contacts`/`dominant_right_singular_vector`/`subspace_mode_shift`/`demean_window` (Task7)→Task8. **Consistent.**

---

## Execution Handoff

**Subagent-Driven (recommended)** — fresh subagent per task; **Task 3 feasibility is a pilot-lock gate (STOP if <4 narrow subjects qualify).** Order: 0→1→2→**3(gate)**→4→5→6(H3b)→7→8(H3c)→9(H3a)→10→11. **Dedicated worktree** (via `superpowers:using-git-worktrees`) so V3a develops isolated from the other active worktrees.
