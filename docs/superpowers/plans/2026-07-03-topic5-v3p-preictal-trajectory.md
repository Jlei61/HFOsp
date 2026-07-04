# Topic 5 V3p — Preictal-Only Non-Axial Trajectory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether, in the clean pre-ictal buildup (`P0..P3` = −120~−10 s, eeg-onset anchored, no ictal), the non-axis avalanche flux and the most-amplifiable direction rise gradually toward seizure onset (Theil-Sen slope > 0), **concentrated on the true non-axis contacts** (label-null adjudicated), not merely a global pre-ictal warm-up.

**Architecture:** New `src/topic5_v3p_preictal_trajectory.py` (pure math: trend/residual/null-orchestration/per-window atom) + `scripts/run_topic5_v3p_*.py` (orchestration) + `config/topic5_v3p.yaml`. **READ-ONLY reuse** of the whole V3a stack — `src.topic5_v3_mode_transition` (windows, geometry, nulls, avalanche, dynamics) and `scripts._topic5_v3_io` (`classify_subject_contacts`, `load_subject_phase_envelopes`) — plus V2 `src.topic5_v2_criticality` primitives V3a already wraps. **V3p adds only new files; it never edits a V3a/V2 file** (keeps the merge with the still-open V3a branch clean). Own results subdir.

**Tech Stack:** Python, numpy, scipy (`stats.theilslopes`, `stats.spearmanr`, `linalg.lstsq`), PyYAML, pytest, pandas.

## Global Constraints (spec rev0 — every task inherits these; values copied verbatim)

- **EXPLORATORY tier. NO forecasting/prediction** — no lead-time, AUC, forward classifier, "pre-warn" claim. V3p = trend description + null adjudication, not a predictor.
- **Preictal-only.** Primary time leg uses `P0/P1/P2/P3` ONLY; never `O/I1/I2/I3/Post`. **eeg-onset anchored** (each seizure's `eeg_onset_rel`, NOT cache `relt=0`).
- **(rev1) onset-guard two-track span.** Every co-primary metric is computed on BOTH `full=[−120,−10]` (headline) AND `guard=[−120,−20]` (`guard_sec=max(10, jitter=10)`). **Strong support requires BOTH tracks to pass same-direction;** full-only pass → `near_onset_dependent=True` (tier capped at 2). `proximal=[−60,−10]` is a sensitivity slope.
- **Trend = slope, not two-window Δ.** Per-seizure **Theil-Sen** slope over preictal windows (primary effect size); Spearman ρ(metric, t) companion; OLS alt. **Subject value = median over seizures.** (Empirically ~17–18 windows/seizure both cohorts → `min_windows_for_slope=8` non-binding.)
- **Co-primary = H3p-b `net_offaxis_flux_surplus_slope` + H3p-c `mode_shift_density_surplus_slope`** (`surplus_slope = obs_slope − median(label-null slope)`); **Holm-corrected at cohort level.** H3p-a `beta_axis_strength_slope` < 0 is **supportive-only** (`module_support_flag_a` always False). H3p-d (burden/self-sustain/gain) secondary. **Support = H3p-b OR H3p-c.**
- **(rev1) Hardened module gates.** `module_support_flag_b = direction ∧ p_label_slope_b<α ∧ p_rate_slope_b<α ∧ lag1_specific_slope>0` (rate + lag0-common-drive are HARD gates, not secondary). `module_support_flag_c = direction ∧ p_label_slope_c<α ∧ p_phase_slope_c<α ∧ p_block_slope_c<α` (**strong**; label+one-temporal = **weak**, not support). `lag1_specific = lag1_net_offaxis_flux − lag0_net_offaxis_flux`.
- **label-null-of-slope is the PRIMARY adjudicator** for "non-axis-specific vs global pre-ictal rise" (within-shaft axis/non-axis label permute → recompute whole trajectory → refit slope). Regression residualization (`*_slope_resid`, vs `global_energy(t)` + `axial_energy(t)`) is **sensitivity only** — conservative floor (collinearity may over-strip); `*_slope_resid ≈ 0` does NOT overturn a label-null positive.
- **(rev1) rate-preserving null is PER-WINDOW.** For the H3p-b rate null, `rate_preserving_shuffle` is applied **within each window independently** (preserves per-contact per-window activation rate, destroys only within-window lagged pairing) — NOT across the whole preictal span (which would flatten the burden trajectory and cause false positives).
- **Geometry/dynamics inherited from V3a, already pilot-locked** in `config/topic5_v3.yaml`: `nonaxis_hfo_participation_max`, `beta_axis_reliability_min=0.20`, `lowrank=6`, `finite_horizon_k=3` (k*), `single_contact_energy_frac_max=0.50`, `z_threshold=2.0`. **Do not re-pilot these.** Non-axis = pure interictal HFO participation, **data/ictal/preictal-blind** (anti-circularity). Three-class contacts; `P_A/P_N` = axis + non-axis-strict only.
- **VAR/DMD demean within window, no standardize** (`demean_window`). λ never raw — always `λ_surplus`. mode-shift uses the **singular vector**, density-normalized (÷ subspace rank). ATM `i≠j`.
- **Subject is the unit;** window→seizure(Theil-Sen)→subject(median). **narrow (7) = primary cohort; broad = replication; NEVER pooled.** **(rev2 option b) broad = `broad_expanded` (`broad_core` 9 + axis-quality-gate-admitted candidates); `broad_core` is ALWAYS reported alongside (expansion must not flip the curated-subset verdict). `yuquan` admitted candidates = a separate cross-dataset supplement, NEVER pooled with epilepsiae (descriptive, 3 sz each). Axis-quality gate calibrated so it never rejects the curated roster (Task 1).**
- **Self-built nulls** (inherited): `label_permute` (shaft-constrained), `rate_preserving_shuffle`, `shaft_constrained_permute`. `p=(1+#exceed)/(1+n_perm)`; trend one-sided (H3p-a other tail). Cohort aggregation on `slope_label_z=(obs−median(null))/MAD(null)`; Wilcoxon signed-rank; H3p-b/H3p-c Holm-corrected.
- **(rev1) time-order null** (secondary sensitivity): per seizure circularly shift/shuffle window order, keep metric values + labels, refit slope → `time_order_p_{b,c}` (tests "closer-to-onset = stronger" order-dependence; not a hard gate).
- **(rev1) QC columns (emit, mostly non-gating):** H3p-c singular-vector stability `mode_singular_gap_median=median(σ1/σ2 of A^{k*})`, `mode_vector_stable`, `cv_r2`; H3p-b sparsity `n_activation_events_pre`, `n_active_windows_pre`, `h3b_activation_sufficient` (sparse activations → 0 flux NOT treated as negative); label-null power `n_label_permutable_shafts`, `n_label_permutable_{axis,nonaxis}`, `n_unique_label_permutations_est`, `label_null_entropy`, `label_null_underpowered` (<100 effective perms). **Empirically narrow 1146 shaftBoth=1, 1096/1125=2 → underpowered candidates.**
- **(rev1) H3p-d relative/hardened metrics:** gain leg uses `gain_shift_slope = slope(gain_nonaxis − gain_axis)` (keep `gain_nonaxis_surplus_slope` for reference); self-sustain uses `N_self_sustain_lag1_specific_slope = lag1_slope − lag0_slope` (≤0 → "synchronous co-activation" not "self-sustain chain"); burden reports raw + label-surplus + resid.
- **Verdict = tier 0–5** (Task 9 only); `state_v3p_supported = tier≥3`; V3p max tier 4. Module scripts emit `module_support_flag/module_direction_correct/module_null_pass`, NOT tier.
- **`geometry_insufficient` → flagged, NOT negative.** Short pre-onset recording (< `min_windows_for_slope` all seizures) → feasibility-insufficient, NOT negative. onset jitter ±10 s must hold. **(rev1) `near_onset_dependent` (full-only pass) → tier cap 2; `label_null_underpowered` → excluded from strong-positive denominator.** **Pre-registered negative acceptable; never rescue 1125** (single subject → descriptive case-series only).
- **New-file commits use `git add <files>` (not `-am`).** Real-data scripts `@pytest.mark.integration` + `--outdir`.

## File Structure

- `config/topic5_v3p.yaml` — preictal span, trend estimator, `min_windows_for_slope`, residualization covariates, co-primary endpoints, nulls, cohorts, tier. (Geometry/dynamics thresholds stay in `topic5_v3.yaml`; v3p loads both.)
- `src/topic5_v3p_preictal_trajectory.py` — ALL new pure math: `load_v3p_config`, `theil_sen_slope`, `spearman_trend`, `slope_over_windows`, `within_compartment_flux`, `global_axial_energy`, `residualize_slope`, `extract_window_metrics`, `null_slope_distribution`, `surplus_and_p`.
- `scripts/run_topic5_v3p_feasibility.py` — preictal window/geometry pilot + `min_windows_for_slope` lock gate.
- `scripts/run_topic5_v3p_trajectory.py` — one preictal-window loop → all metric trajectories → slopes → nulls → residualization → gates → `v3p_trajectory_subject.csv` + `v3p_window_detail.csv`.
- `scripts/run_topic5_v3p_summary.py` — Holm co-primary + tier → `v3p_summary_subject.csv` + `v3p_cohort_tier.json`.
- `scripts/plot_topic5_v3p_summary.py` — 2–3 independent-question panels.
- `tests/test_topic5_v3p_preictal_trajectory.py` (pure) + `tests/test_topic5_v3p_integration.py` (integration).
- Outputs: `results/topic5_ictal_recruitment/v3p_preictal_trajectory/{narrow,broad}/`.

**Execution order:** 0 → **1 (feasibility gate, geometry/dynamics already locked)** → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10.

---

## Task 0: Config + module skeleton

**Files:** Create `config/topic5_v3p.yaml`, `src/topic5_v3p_preictal_trajectory.py`; Test `tests/test_topic5_v3p_preictal_trajectory.py`.
**Interfaces:**
- Produces: `load_v3p_config(path=None) -> dict`.

- [ ] **Step 1: Write config**

```yaml
# config/topic5_v3p.yaml
preictal:
  phases: [P0, P1, P2, P3]
  span_full_rel: [-120.0, -10.0]         # headline primary (rev1: user kept)
  span_guard_rel: [-120.0, -20.0]        # rev1: jitter-safe onset guard (also co-primary)
  span_proximal_rel: [-60.0, -10.0]      # sensitivity slope span
  guard_sec: 10.0                        # guard_end = onset - max(10, jitter)
  min_windows_for_slope: 8               # empirically non-binding (~17-18/sz both cohorts)
trend:
  estimator: theil_sen                   # primary
  companions: [spearman, ols]
  slope_direction:                       # expected sign per leg
    net_offaxis_flux: greater
    mode_shift_density: greater
    nonaxis_activation_burden: greater
    N_self_sustain_lag1_specific: greater
    gain_shift: greater
    beta_axis_strength: less             # H3p-a supportive
gates:                                   # rev1 hardened module gates
  h3b_require: [p_label, p_rate, lag1_specific_positive]
  h3c_require: [p_label, p_phase, p_block]   # all -> strong; label+1 temporal -> weak
  require_both_spans: true               # strong support needs full AND guard
residualization:
  covariates: [global_energy, axial_energy]
  primary_adjudicator: label_null_slope   # regression residual is sensitivity only
nulls_v3p:                               # rev1 new null / QC knobs
  rate_null_per_window: true             # preserve per-window rate, shuffle within-window only
  time_order_null: true
  label_null_min_unique_perms: 100       # < -> label_null_underpowered
  mode_singular_gap_min: 1.2             # OPEN pilot; < -> mode_vector_stable=False
  h3b_min_activation_events: 20          # OPEN pilot; < -> h3b_activation_sufficient=False
co_primary:
  endpoints: [net_offaxis_flux_surplus_slope, mode_shift_density_surplus_slope]
  correction: holm
inherit_v3_config: topic5_v3             # geometry+dynamics+avalanche+nulls locked there
nulls: {n_perm_smoke: 20, n_perm_dev: 100, n_perm_final: 1000, seed: 20260703, alpha: 0.05}
cohorts: {primary: narrow, replication: broad, never_pool: true}
cohort_expansion:                        # rev2 (option b): expand broad via an axis-quality gate
  broad_core: [epilepsiae_139, epilepsiae_253, epilepsiae_1077, epilepsiae_1096, epilepsiae_1125,
               epilepsiae_1150, epilepsiae_620, epilepsiae_635, epilepsiae_916]   # curated (swap-positive 8 + E916)
  candidates_epilepsiae: [epilepsiae_1084, epilepsiae_583, epilepsiae_590, epilepsiae_922]
  candidates_yuquan: [yuquan_xuxinyi, yuquan_zhangkexuan]   # cross-dataset supplement, NEVER pooled with epilepsiae
  axis_quality_gate:                     # a candidate is ADMITTED iff all pass (thresholds pilot-locked at Task 1)
    require_geometry_sufficient: true
    axis_rank_min_distinct: 5            # >=5 distinct interictal typical_rank among axis contacts (real early->late gradient, not tied)
    axis_participation_gap_min: 0.15     # min(axis participation) - max(nonaxis-strict participation) >= gap (clean high/low split)
    require_rank_displacement_json: true # candidate went through the masked interictal-propagation pipeline
    calibrate_on_roster: true            # lock thresholds so the CURATED roster is NOT rejected (sanity)
  broad_analysis: broad_expanded         # broad_core + admitted candidates_epilepsiae; broad_core ALWAYS reported alongside
jitter_sec: [5.0, 10.0, 15.0]
tier: exploratory
```

- [ ] **Step 2: Failing loader test**

```python
from src.topic5_v3p_preictal_trajectory import load_v3p_config
def test_v3p_config_keys():
    c = load_v3p_config()
    assert c["preictal"]["phases"] == ["P0", "P1", "P2", "P3"]
    assert c["preictal"]["span_full_rel"] == [-120.0, -10.0]
    assert c["preictal"]["span_guard_rel"] == [-120.0, -20.0]      # rev1 onset guard
    assert c["trend"]["estimator"] == "theil_sen"
    assert c["gates"]["h3b_require"] == ["p_label", "p_rate", "lag1_specific_positive"]
    assert c["gates"]["h3c_require"] == ["p_label", "p_phase", "p_block"]
    assert c["gates"]["require_both_spans"] is True
    assert c["nulls_v3p"]["rate_null_per_window"] is True and c["nulls_v3p"]["time_order_null"] is True
    assert c["residualization"]["primary_adjudicator"] == "label_null_slope"
    assert c["co_primary"]["correction"] == "holm"
    assert c["co_primary"]["endpoints"] == ["net_offaxis_flux_surplus_slope", "mode_shift_density_surplus_slope"]
    assert c["inherit_v3_config"] == "topic5_v3" and c["cohorts"]["primary"] == "narrow"
```

- [ ] **Step 3: Run fail** — `pytest tests/test_topic5_v3p_preictal_trajectory.py::test_v3p_config_keys -v` → FAIL (module/function missing).
- [ ] **Step 4: Implement loader**

```python
"""Topic 5 V3p — preictal-only non-axial trajectory (pure math).

READ-ONLY reuse of the V3a stack (src.topic5_v3_mode_transition,
scripts._topic5_v3_io). V3p adds only the preictal-restriction + slope +
residualization + N->N self-sustain layer. See
docs/superpowers/plans/2026-07-03-topic5-v3p-preictal-trajectory.md.
Exploratory; no forecasting.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Mapping, Sequence
import numpy as np
import yaml
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CFG = _ROOT / "config" / "topic5_v3p.yaml"

def load_v3p_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path is not None else _DEFAULT_CFG
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, Mapping):
        raise ValueError(f"V3p config must be a mapping: {cfg_path}")
    return dict(cfg)
```

- [ ] **Step 5: Run pass.** **Step 6: Commit** — `git add config/topic5_v3p.yaml src/topic5_v3p_preictal_trajectory.py tests/test_topic5_v3p_preictal_trajectory.py && git commit -m "feat(topic5-v3p): config + module skeleton"`

---

## Task 1: Feasibility pilot (gate — geometry/dynamics already locked; lock min_windows_for_slope)

**Files:** Create `scripts/run_topic5_v3p_feasibility.py`; Test `tests/test_topic5_v3p_integration.py`.
**Interfaces:**
- Consumes: V3a `scripts._topic5_v3_io.classify_subject_contacts`, `load_subject_phase_envelopes`; V3a `src.topic5_v3_mode_transition.sliding_windows`, `geometry_sufficient`; Task-0 `load_v3p_config`; V3a `load_v3_config`.
- Produces: `feasibility.csv` cols: `subject, cohort, roster_status(roster|candidate), n_seizures, n_windows_P0, n_windows_P1, n_windows_P2, n_windows_P3, n_windows_full_total, n_windows_guard_total, usable_pre_sec, n_contacts_all_clean, n_axis, n_nonaxis, n_ambiguous, n_shaft_with_axis_and_nonaxis, n_unique_label_permutations_est, label_null_underpowered, geometry_sufficient, axis_rank_distinct, axis_participation_gap, has_rank_displacement_json, axis_quality_gate_pass, admitted, n_seizures_ge_min_windows, cohort_viable`. **(rev1: guard window total + label-null permutability — flags narrow 1146/1096/1125. rev2: `axis_quality_gate_pass`/`admitted` for the expansion candidates.)**

**Contract:** per subject: `classify_subject_contacts` (V3a; source of truth for pool + axis/non-axis) → `load_subject_phase_envelopes(ds_sid, cohort, v3cfg, phases=["P0","P1","P2","P3"])`. For each seizure sum `sliding_windows` counts over the four preictal phase envelopes; `n_seizures_ge_min_windows` = seizures whose preictal window total ≥ `min_windows_for_slope`. `geometry_sufficient` via V3a helper (min_n_axis=5, min_n_nonaxis=3, ≥1 shaft-with-both). `cohort_viable` = ≥4 subjects `geometry_sufficient AND n_seizures_ge_min_windows≥1`.
- **(rev2) axis-quality gate** — run over roster (`roster_status=roster`, for calibration) AND expansion candidates (`roster_status=candidate`, from `cohort_expansion.candidates_*`, loaded under broad context / yuquan under its own context): `axis_rank_distinct` = # distinct finite interictal `typical_rank` among axis contacts (from `classify_subject_contacts`'s ctx `ta_rank`); `axis_participation_gap = min(participation[axis]) − max(participation[nonaxis_strict])`; `has_rank_displacement_json` = the candidate's `interictal_propagation_masked/rank_displacement/per_subject/<ds_sid>.json` exists. `axis_quality_gate_pass = geometry_sufficient AND axis_rank_distinct≥axis_rank_min_distinct AND axis_participation_gap≥axis_participation_gap_min AND has_rank_displacement_json`. `admitted` = `roster_status=="roster"` (grandfathered) OR (`roster_status=="candidate"` AND `axis_quality_gate_pass`).

- [ ] **Step 1: Failing integration test**

```python
import pytest, pandas as pd
@pytest.mark.integration
def test_v3p_feasibility_writes_csv(tmp_path):
    from scripts.run_topic5_v3p_feasibility import main
    out = tmp_path / "feasibility.csv"
    main(["--cohort", "narrow", "--outdir", str(tmp_path)])
    df = pd.read_csv(out)
    for col in ["geometry_sufficient", "n_windows_P3", "n_seizures_ge_min_windows", "n_nonaxis",
                "roster_status", "axis_quality_gate_pass", "admitted", "n_unique_label_permutations_est"]:
        assert col in df.columns
    assert len(df) >= 1
```

- [ ] **Step 2: Run fail.** **Step 3: Implement** (import the frozen V3a io + window helpers; `main(argv)` with argparse `--cohort`, `--outdir`, optional `--subjects`, `--include-candidates` [rev2: also probe `cohort_expansion.candidates_*` under broad/yuquan context, compute the axis-quality gate + `admitted`]). **Step 4: Run pass.**
- [ ] **Step 5: Commit** — `git add scripts/run_topic5_v3p_feasibility.py tests/test_topic5_v3p_integration.py && git commit -m "feat(topic5-v3p): feasibility pilot (preictal window + geometry counts)"`
- [ ] **Step 6 (DECISION GATE):** run `--cohort narrow` and `--cohort broad` (with `--include-candidates`); inspect `feasibility.csv`. **(a) Lock `min_windows_for_slope`** (default 8 — empirically ~17–18/sz so non-binding; keep 8 unless a cohort drops <4 viable). **(b) Calibrate the axis-quality gate:** confirm `axis_quality_gate_pass=True` for ALL curated roster subjects (`roster_status=roster`); if any curated subject fails, LOOSEN `axis_rank_min_distinct`/`axis_participation_gap_min` until the roster passes (the gate must never reject the curated cohort), record locked values in config. **(c) Lock the admitted expansion set** = candidates with `admitted=True`; `broad_expanded = broad_core ∪ admitted_epilepsiae`; admitted yuquan → the `yuquan` supplement (never pooled). Record the final rosters in config + this line. **Auto-select one `admitted AND n_seizures_ge_min_windows≥1` subject per cohort as the integration subject** (expected: 253). **If <4 narrow qualify → STOP + report (narrow non-viable).**

---

## Task 2: Trend estimators — pure

**Files:** Modify `src/topic5_v3p_preictal_trajectory.py`; Test same.
**Interfaces:**
- Produces: `theil_sen_slope(y, t) -> float`; `spearman_trend(y, t) -> float`; `slope_over_windows(values, centers, estimator) -> float`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v3p_preictal_trajectory import theil_sen_slope, spearman_trend, slope_over_windows
def test_theil_sen_and_spearman_trend():
    t = np.arange(20.0)
    y = 0.3 * t + 1.0
    assert abs(theil_sen_slope(y, t) - 0.3) < 1e-9
    assert spearman_trend(y, t) > 0.999
    y_out = y.copy(); y_out[7] = 500.0                       # one wild outlier window
    assert abs(theil_sen_slope(y_out, t) - 0.3) < 0.05       # robust
    y_flat = np.zeros(20)
    assert abs(theil_sen_slope(y_flat, t)) < 1e-9 and np.isnan(spearman_trend(y_flat, t))
def test_slope_over_windows_nan_safe_and_dispatch():
    t = np.arange(12.0); y = 0.5 * t
    y2 = y.copy(); y2[3] = np.nan; t2 = t.copy()
    assert abs(slope_over_windows(y2, t2, "theil_sen") - 0.5) < 1e-9   # drops the NaN window
    assert abs(slope_over_windows(y, t, "ols") - 0.5) < 1e-9
    assert np.isnan(slope_over_windows(y[:1], t[:1], "theil_sen"))      # <2 points
```

- [ ] **Step 2: Run fail. Step 3: Implement**

```python
def _finite_pairs(y, t):
    y = np.asarray(y, float); t = np.asarray(t, float)
    m = np.isfinite(y) & np.isfinite(t)
    return y[m], t[m]

def theil_sen_slope(y, t) -> float:
    y, t = _finite_pairs(y, t)
    if y.size < 2 or np.unique(t).size < 2:
        return float("nan")
    return float(stats.theilslopes(y, t)[0])

def spearman_trend(y, t) -> float:
    y, t = _finite_pairs(y, t)
    if y.size < 3 or np.unique(y).size < 2 or np.unique(t).size < 2:
        return float("nan")
    return float(stats.spearmanr(y, t).correlation)

def slope_over_windows(values, centers, estimator) -> float:
    if estimator == "theil_sen":
        return theil_sen_slope(values, centers)
    if estimator == "spearman":
        return spearman_trend(values, centers)
    if estimator == "ols":
        y, t = _finite_pairs(values, centers)
        if y.size < 2 or np.unique(t).size < 2:
            return float("nan")
        return float(np.polyfit(t, y, 1)[0])
    raise ValueError(f"unknown estimator: {estimator!r}")
```

- [ ] **Step 4: Run pass. Step 5: Commit** — `git commit -am "feat(topic5-v3p): theil-sen / spearman / dispatch slope estimators (nan-safe)"`

---

## Task 3: N→N self-sustain flux + global/axial energy — pure

**Files:** Modify `src/topic5_v3p_preictal_trajectory.py`; Test same. **Reuse** V3a `compartment_flux` diag-free precondition pattern.
**Interfaces:**
- Produces: `within_compartment_flux(atm, idx, normalization="source_mean") -> float`; `global_axial_energy(env_win, axis_rows) -> (float, float)`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v3p_preictal_trajectory import within_compartment_flux, global_axial_energy
def test_within_compartment_flux_self_sustain():
    # 4 contacts, N = {2,3}; strong 2->3 and 3->2 mass, diagonal already zero
    atm = np.zeros((4, 4))
    atm[2, 3] = 0.6; atm[3, 2] = 0.4; atm[0, 1] = 0.9   # axis-internal, ignored by N block
    val = within_compartment_flux(atm, np.array([2, 3]), "source_mean")
    assert abs(val - 0.5) < 1e-9                          # mean of active N-source outgoing-into-N mass (0.6, 0.4)
    assert within_compartment_flux(atm, np.array([2, 3]) , "source_mean") == \
           within_compartment_flux(np.pad(atm, ((0,1),(0,1))), np.array([2, 3]), "source_mean")  # padding never-active row invariant
def test_global_axial_energy():
    env = np.array([[1.0, -1.0], [2.0, -2.0], [0.0, 0.0]])   # 3 contacts x 2 t; mean|.| rows = 1,2,0
    g, a = global_axial_energy(env, np.array([0, 1]))
    assert abs(g - 1.0) < 1e-9 and abs(a - 1.5) < 1e-9        # global mean over all rows; axial over rows 0,1
```

- [ ] **Step 2: Run fail. Step 3: Implement**

```python
def within_compartment_flux(atm, idx, normalization="source_mean") -> float:
    """Self-sustain: mean over ACTIVE sources i in idx of that source's
    outgoing mass into the SAME set idx. Mirrors V3a compartment_flux
    source_mean but for the N x N block. Requires a diagonal-free ATM."""
    mat = np.asarray(atm, float)
    if not np.allclose(np.diag(mat), 0.0):
        raise ValueError("within_compartment_flux requires a diagonal-free ATM")
    idx = np.asarray(idx, int)
    if idx.size == 0:
        return 0.0
    active = mat[idx].sum(axis=1) > 0.0
    if not np.any(active):
        return 0.0
    block_mass = mat[np.ix_(idx, idx)].sum(axis=1)   # into the same set (diag already 0)
    if normalization == "source_mean":
        return float(block_mass[active].mean())
    if normalization == "sum":
        return float(block_mass.sum())
    raise ValueError(f"unknown normalization: {normalization!r}")

def global_axial_energy(env_win, axis_rows) -> tuple[float, float]:
    """Per-window energy scalars: mean over rows of the within-window mean
    |envelope|. global = all rows; axial = axis rows only (0.0 if none)."""
    env = np.asarray(env_win, float)
    row_energy = np.nanmean(np.abs(env), axis=1)
    g = float(np.nanmean(row_energy)) if row_energy.size else float("nan")
    axis_rows = np.asarray(axis_rows, int)
    a = float(np.nanmean(row_energy[axis_rows])) if axis_rows.size else 0.0
    return g, a
```

- [ ] **Step 4: Run pass. Step 5: Commit** — `git commit -am "feat(topic5-v3p): N->N self-sustain flux + global/axial energy scalars"`

---

## Task 4: Regression residualization — pure

**Files:** Modify `src/topic5_v3p_preictal_trajectory.py`; Test same.
**Interfaces:**
- Consumes: Task-2 `slope_over_windows`.
- Produces: `residualize_slope(values, centers, covariates, estimator) -> float`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v3p_preictal_trajectory import residualize_slope
def test_residualize_strips_global_and_is_conservative_under_collinearity():
    t = np.arange(20.0)
    glob = 0.4 * t                                   # global energy IS collinear with time
    # non-axis metric = 2*global + a genuinely orthogonal rise
    orth = np.where(t % 2 == 0, 1.0, -1.0) * 0.1     # zero net slope, orthogonal to t
    vals = 2.0 * glob + orth
    resid_slope = residualize_slope(vals, t, [glob], "theil_sen")
    assert abs(resid_slope) < 0.02                   # collinear global stripped -> conservative ~0 (documented floor)
def test_residualize_keeps_orthogonal_trend():
    t = np.arange(20.0)
    glob = np.sin(t)                                 # global uncorrelated with linear time
    vals = 0.3 * t + 2.0 * glob
    resid_slope = residualize_slope(vals, t, [glob], "ols")
    assert abs(resid_slope - 0.3) < 0.05             # orthogonal non-axis trend survives
```

- [ ] **Step 2: Run fail. Step 3: Implement**

```python
def residualize_slope(values, centers, covariates, estimator) -> float:
    """Slope of the residual of `values` after OLS-regressing on
    `covariates` (each an array aligned to `values`). Conservative: if a
    covariate is collinear with time, the shared trend is absorbed and the
    residual slope shrinks toward 0 — this is the documented floor (spec
    Sec 7), NOT evidence the non-axis rise is absent. NaN-safe: windows
    with any non-finite value/covariate are dropped; rank-deficient design
    or <2 surviving windows -> nan."""
    y = np.asarray(values, float); t = np.asarray(centers, float)
    cov = [np.asarray(c, float) for c in covariates]
    m = np.isfinite(y) & np.isfinite(t)
    for c in cov:
        m &= np.isfinite(c)
    if m.sum() < 3:
        return float("nan")
    X = np.column_stack([np.ones(m.sum())] + [c[m] for c in cov])
    try:
        beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    resid = y[m] - X @ beta
    return slope_over_windows(resid, t[m], estimator)
```

- [ ] **Step 4: Run pass. Step 5: Commit** — `git commit -am "feat(topic5-v3p): regression residualize_slope (global+axial, conservative collinearity floor)"`

---

## Task 5: Per-window metric atom — pure

**Files:** Modify `src/topic5_v3p_preictal_trajectory.py`; Test same. **Reuse** V3a `atm_offdiag`, `net_offaxis_flux`, `demean_window`, `lowrank_var`, `dominant_right_singular_vector`, `map_lowrank_vector_to_contacts`, `subspace_mode_shift`, `project_2d`, `direct_2d_var`, `beta_axis`; V2 `activations_from_z`; Task-3 `within_compartment_flux`, `global_axial_energy`. **Do NOT reuse V2 `contact_susceptibility`** — its question is a late−early **Δ** between two sub-windows, not a within-window roughness **level** (CLAUDE.md §6.1 helper-question mismatch); the per-contact line-length rate is computed inline (see Contract).
**Interfaces:**
- Consumes: `geom` dict = `{names, axis_idx, nonaxis_idx, P_A, P_N, e_axis_mean, e_nonaxis_mean, rank_forward}` (built once per subject by the runner from V3a `subspace_projectors`/`axis_nonaxis_vectors`/`rank_forward`); `v3cfg` = inherited `load_v3_config()` (for `lowrank`, `finite_horizon_k`, `var_ridge_alpha`, `z_threshold`).
- Produces: `extract_window_metrics(env_win, geom, v3cfg) -> dict` with keys `net_offaxis_flux_lag1, net_offaxis_flux_lag0, mode_shift_density, mode_singular_gap, nonaxis_activation_rate, n_activation_events, global_energy, axial_energy, N_self_sustain_lag1, N_self_sustain_lag0, gain_axis, gain_nonaxis, beta_axis_strength`. **(rev1: lag1 AND lag0 for both flux legs — the runner forms `lag1_specific = lag1 − lag0`; `mode_singular_gap` + `n_activation_events` are QC.)**

**Contract:** `env_win` = `(n_all_clean, n_t)` bb-envelope slice (rows ordered by `all_clean`). All dynamics on `demean_window(env_win)`. Activations via `active = activations_from_z(env_win, z_threshold)`; `n_activation_events = int(active.sum())`. **Flux both lags:** `atm1=atm_offdiag(active)` (lag1), `atm0=atm_lag0(active)` (lag0 same-time co-activation); `net_offaxis_flux_lag1=net_offaxis_flux(atm1, axis_idx, nonaxis_idx, "source_mean")`, `net_offaxis_flux_lag0=net_offaxis_flux(atm0, axis_idx, nonaxis_idx, "source_mean")`. `mode_shift_density`: `A_lowrank,U_r=lowrank_var(env_win, lowrank, alpha)`; `sv=svd(matrix_power(A_lowrank,k*), compute_uv=False)`, `mode_singular_gap = sv[0]/sv[1]` (nan if <2 sv or sv[1]==0); `u_r=dominant_right_singular_vector(A_lowrank, k*)`; `u_c=map_lowrank_vector_to_contacts(u_r,U_r)`; `subspace_mode_shift(u_c, P_N, P_A, "density")`. **Self-sustain both lags:** `N_self_sustain_lag1=within_compartment_flux(atm1, nonaxis_idx)`, `N_self_sustain_lag0=within_compartment_flux(atm0, nonaxis_idx)`. gains: `B=direct_2d_var(project_2d(demean_window(env_win), e_axis_mean, e_nonaxis_mean), alpha)`, `gain_axis=‖B[:,0]‖`, `gain_nonaxis=‖B[:,1]‖` (runner forms `gain_shift=gain_nonaxis−gain_axis`). `nonaxis_activation_rate=active[nonaxis_idx].mean()`. `global_energy,axial_energy=global_axial_energy(env_win, axis_idx)`. `beta_axis_strength=abs(beta_axis({names[i]: float(np.nanmean(np.abs(np.diff(env_win[i])))) for i in range(len(names))}, rank_forward))` — per-contact within-window **line-length rate (roughness)** inline (mean absolute successive difference); NOT V2 `contact_susceptibility` (§6.1 mismatch). Any degenerate window → the affected key is `nan` (never raises).

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v3_mode_transition import subspace_projectors, axis_nonaxis_vectors, rank_forward, load_v3_config
from src.topic5_v3p_preictal_trajectory import extract_window_metrics
def test_extract_window_metrics_keys_and_flux_sign():
    rng = np.random.default_rng(0); names = [f"A{i}" for i in range(6)]
    axis = names[:3]; nonaxis = names[3:]
    P_A, P_N = subspace_projectors(names, axis, nonaxis)
    rf = rank_forward({n: float(i) for i, n in enumerate(axis)})
    e_am, e_ag, e_nm = axis_nonaxis_vectors(names, rf, axis, nonaxis)
    # scripted axis->non-axis cascade: axis fires at t, non-axis at t+1
    n_t = 200; env = rng.standard_normal((6, n_t)) * 0.1
    env[:3, :-1] += 4.0 * (rng.random((3, n_t - 1)) > 0.6)
    env[3:, 1:] += env[:3, :-1]                       # non-axis echoes axis one step later
    geom = {"names": names, "axis_idx": np.array([0,1,2]), "nonaxis_idx": np.array([3,4,5]),
            "P_A": P_A, "P_N": P_N, "e_axis_mean": e_am, "e_nonaxis_mean": e_nm, "rank_forward": rf}
    m = extract_window_metrics(env, geom, load_v3_config())
    for k in ["net_offaxis_flux_lag1","net_offaxis_flux_lag0","mode_shift_density","mode_singular_gap",
              "nonaxis_activation_rate","n_activation_events","global_energy","axial_energy",
              "N_self_sustain_lag1","N_self_sustain_lag0","gain_axis","gain_nonaxis","beta_axis_strength"]:
        assert k in m
    # lag1 (delayed A->N flow) exceeds lag0 (same-time co-activation) for a scripted one-step cascade
    assert np.isfinite(m["net_offaxis_flux_lag1"]) and m["net_offaxis_flux_lag1"] > m["net_offaxis_flux_lag0"]
```

- [ ] **Step 2: Run fail. Step 3: Implement** (per Contract; import the V3a/V2 helpers at module top; wrap each estimator in a local finite-guard returning `nan` on empty index / degenerate SVD). **Step 4: Run pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(topic5-v3p): per-window metric atom (flux/mode-shift/burden/self-sustain/gain/beta) reusing V3a"`

---

## Task 6: Null-slope orchestration + surplus/p — pure

**Files:** Modify `src/topic5_v3p_preictal_trajectory.py`; Test same.
**Interfaces:**
- Consumes: Task-2 `slope_over_windows`.
- Produces: `null_slope_distribution(resample_traj_fn, estimator, n_perm, rng) -> np.ndarray`; `surplus_and_p(obs_slope, null_slopes, direction) -> dict`.

**Contract:** `resample_traj_fn(rng) -> list[(values, centers)]` is a closure the caller (Task 7) builds once per null TYPE and returns the recomputed per-seizure `(metric_values, centers)` pairs under one draw. **rev1: four closure types** — (a) **label** (`label_permute` re-index of axis/non-axis → rebuild `geom_perm` → recompute atom), (b) **rate-per-window** (`rate_preserving_shuffle` applied WITHIN each window's `active` independently, preserving per-window rate, then recompute flux), (c) **spatial** (`shaft_constrained_permute`), (d) **time-order** (circularly shift/shuffle the window ORDER, keeping each window's metric value + labels — permutes `centers`↔`values` pairing to test order-dependence). `null_slope_distribution` fits `slope_over_windows` per seizure, takes the **median over seizures** = one null subject-slope, repeats `n_perm` times. `surplus_and_p`: `surplus = obs − median(null)`; `p = (1+#{null ≥ obs})/(1+n_perm)` for `direction="greater"`, `(1+#{null ≤ obs})/(1+n_perm)` for `"less"`; `z = (obs − median)/MAD` (NaN-safe; MAD via `1.4826*median(|null−median|)`, 0→nan).

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v3p_preictal_trajectory import null_slope_distribution, surplus_and_p
def test_null_slope_distribution_and_surplus_p():
    rng = np.random.default_rng(1); t = np.arange(15.0)
    # resample returns flat-trend seizures (null slope ~ 0) for two seizures
    def resample(r):
        return [(r.standard_normal(15), t), (r.standard_normal(15), t)]
    null = null_slope_distribution(resample, "theil_sen", n_perm=200, rng=rng)
    assert null.shape == (200,) and abs(np.median(null)) < 0.05
    res = surplus_and_p(0.5, null, "greater")               # obs strongly positive vs ~0 null
    assert res["surplus"] > 0.4 and res["p"] < 0.02 and res["z"] > 3
    res_neg = surplus_and_p(-0.5, null, "less")
    assert res_neg["p"] < 0.02
```

- [ ] **Step 2: Run fail. Step 3: Implement**

```python
def null_slope_distribution(resample_traj_fn, estimator, n_perm, rng) -> np.ndarray:
    out = np.empty(int(n_perm), float)
    for p in range(int(n_perm)):
        per_sz = resample_traj_fn(rng)
        slopes = [slope_over_windows(v, c, estimator) for v, c in per_sz]
        slopes = [s for s in slopes if np.isfinite(s)]
        out[p] = float(np.median(slopes)) if slopes else float("nan")
    return out

def surplus_and_p(obs_slope, null_slopes, direction) -> dict:
    null = np.asarray(null_slopes, float)
    null = null[np.isfinite(null)]
    n = null.size
    if n == 0 or not np.isfinite(obs_slope):
        return {"surplus": float("nan"), "p": float("nan"), "z": float("nan")}
    med = float(np.median(null))
    if direction == "greater":
        p = (1 + int(np.sum(null >= obs_slope))) / (1 + n)
    elif direction == "less":
        p = (1 + int(np.sum(null <= obs_slope))) / (1 + n)
    else:
        raise ValueError(f"unknown direction: {direction!r}")
    mad = 1.4826 * float(np.median(np.abs(null - med)))
    z = (obs_slope - med) / mad if mad > 0 else float("nan")
    return {"surplus": float(obs_slope - med), "p": float(p), "z": float(z)}
```

- [ ] **Step 4: Run pass. Step 5: Commit** — `git commit -am "feat(topic5-v3p): null-slope distribution + surplus/p/z (label/rate/spatial via resample callback)"`

---

## Task 7: Trajectory runner — co-primary H3p-b + H3p-c — integration

**Files:** Create `scripts/run_topic5_v3p_trajectory.py`; Test `tests/test_topic5_v3p_integration.py`.
**Interfaces:**
- Consumes: Tasks 2–6 pure fns; V3a io `load_subject_phase_envelopes`/`classify_subject_contacts`; V3a `subspace_projectors`, `axis_nonaxis_vectors`, `rank_forward`, `label_permute`, `rate_preserving_shuffle`, `shaft_constrained_permute`; V2 `shaft_of`.
- Produces: `v3p_trajectory_subject.csv` (co-primary cols below) + `v3p_window_detail.csv` (`subject, cohort, seizure_idx, span, phase, t_center, net_offaxis_flux_lag1, net_offaxis_flux_lag0, mode_shift_density, mode_singular_gap, nonaxis_activation_rate, n_activation_events, global_energy, axial_energy, N_self_sustain_lag1, N_self_sustain_lag0, gain_axis, gain_nonaxis`).

**Subject CSV co-primary cols:** `subject, cohort, status, skip_reason, geometry_sufficient, n_axis, n_nonaxis, n_ambiguous, n_seizures_used,`
`net_offaxis_flux_slope_raw, net_offaxis_flux_surplus_slope, net_offaxis_flux_slope_resid, net_offaxis_flux_slope_z, p_label_slope_b, p_rate_slope_b, p_spatial_slope_b, proximal_flux_slope, spearman_rho_flux, leave_one_contact_flux_pass, axis_only_flux_control_pass, onset_jitter_pass_b, module_support_flag_b, module_direction_correct_b, module_null_pass_b,`
`mode_shift_density_slope_raw, mode_shift_density_surplus_slope, mode_shift_density_slope_resid, mode_shift_density_slope_z, p_label_slope_c, p_phase_slope_c, p_block_slope_c, mode_shift_2D_consistency_slope, top_contact_energy_fraction, single_contact_driven, leave_one_contact_mode_pass, axis_only_mode_control_pass, onset_jitter_pass_c, rank_used, k_star, spearman_rho_mode, module_support_flag_c, module_direction_correct_c, module_null_pass_c,`
**rev1 co-primary additions** — naming convention: the **unsuffixed** `*_surplus_slope`/`*_slope_z`/`p_label_slope_{b,c}` = the **`full` (headline)** span; each also gets a **`_guard`** companion (`net_offaxis_flux_surplus_slope_guard, net_offaxis_flux_slope_z_guard, p_label_slope_b_guard, mode_shift_density_surplus_slope_guard, mode_shift_density_slope_z_guard, p_label_slope_c_guard`). Plus: `near_onset_dependent_b, near_onset_dependent_c, lag1_specific_slope, common_drive_sensitive, mode_singular_gap_median, mode_vector_stable, cv_r2, n_activation_events_pre, n_active_windows_pre, h3b_activation_sufficient, h3c_support_grade, time_order_p_b, time_order_p_c, n_label_permutable_shafts, n_unique_label_permutations_est, label_null_underpowered,`
`trend_estimator, slope_span`. **(no `tier`.)**

**Contract:**
- Build `geom` once per subject (`subspace_projectors`, `axis_nonaxis_vectors`, `rank_forward` from `classify_subject_contacts`; `shaft_by_name` via `shaft_of`).
- **Observed (rev1: two spans):** per seizure, per preictal window (`load_subject_phase_envelopes` phases `P0..P3`), call `extract_window_metrics` → per-seizure metric trajectories `(values, centers=t_center)`. **Compute every co-primary slope + null on BOTH `full=[−120,−10]` AND `guard=[−120,−20]` spans** (windows filtered by `t_center` range). `slope_over_windows(..., "theil_sen")` per seizure; **subject obs = median over seizures**. Only seizures with ≥ `min_windows_for_slope` preictal windows count.
- **label-null (PRIMARY):** `resample_traj_fn` = closure that (a) draws `label_permute(axis_names, nonaxis_names, shaft_by_name, rng)`, (b) rebuilds `geom_perm` (P_A/P_N/idx from permuted labels), (c) recomputes each seizure's flux & mode-shift trajectory via `extract_window_metrics(env, geom_perm, v3cfg)`. `null_slope_distribution(...)` → `surplus_and_p(obs, null, "greater")` → `net_offaxis_flux_surplus_slope`/`mode_shift_density_surplus_slope`, `p_label_slope_{b,c}`, `_slope_z`.
- **nulls (rev1 HARD, not secondary):** flux `p_rate_slope_b` (**rate-per-window**: `rate_preserving_shuffle` applied within each window's `active` independently) + `p_spatial_slope_b` (`shaft_constrained_permute`); `lag1_specific_slope = slope(net_offaxis_flux_lag1 − net_offaxis_flux_lag0)`, `common_drive_sensitive = lag1_specific_slope <= 0`. mode-shift `p_phase_slope_c` + `p_block_slope_c` (phase-randomize + block-shuffle of `env_win`) — **both HARD**. `time_order_p_{b,c}` via the time-order closure (sensitivity).
- **residualization (sensitivity):** `net_offaxis_flux_slope_resid`/`mode_shift_density_slope_resid` = `residualize_slope(metric_traj, centers, [global_energy_traj, axial_energy_traj], "theil_sen")` per seizure → median over seizures.
- **label-null power QC (rev1):** `n_label_permutable_shafts` = #shafts with both axis & non-axis-strict; `n_unique_label_permutations_est = exp(Σ_shaft log C(n_shaft, k_axis_shaft))`; `label_null_underpowered = n_unique_label_permutations_est < label_null_min_unique_perms (100)`.
- **QC (rev1):** `mode_singular_gap_median = median` over windows of `mode_singular_gap`; `mode_vector_stable = gap_median ≥ mode_singular_gap_min`; `n_activation_events_pre`/`n_active_windows_pre`; `h3b_activation_sufficient = n_activation_events_pre ≥ h3b_min_activation_events`.
- **gates:** `top_contact_energy_fraction=max(u_c²)/Σu_c²`; `single_contact_driven = > single_contact_energy_frac_max`; `leave_one_contact_*_pass`; `axis_only_*_control_pass`; `onset_jitter_pass_*` stable under ±10 s.
- **rev1 hardened module flags** (unsuffixed p = full/headline span): `module_support_flag_b = direction_correct ∧ p_label_slope_b<α ∧ p_label_slope_b_guard<α ∧ p_rate_slope_b<α ∧ (lag1_specific_slope>0)`; `h3c_support_grade = "strong" if (p_label_slope_c<α ∧ p_label_slope_c_guard<α ∧ p_phase_slope_c<α ∧ p_block_slope_c<α) else "weak" if (p_label_slope_c<α ∧ (p_phase_slope_c<α ∨ p_block_slope_c<α)) else "none"`, `module_support_flag_c = direction_correct ∧ h3c_support_grade=="strong"`. `near_onset_dependent_{b,c} = (full passes) ∧ ¬(guard passes)`. `geometry_insufficient`/`<min_windows` all seizures → `status=skipped`. narrow + broad separate; never pooled. `proximal_flux_slope` on `[-60,-10]` (sensitivity).

- [ ] **Step 1: Failing integration test (two-tier)**

```python
import pytest, pandas as pd, numpy as np
@pytest.mark.integration
def test_v3p_trajectory_writes_csv_even_if_skipped(tmp_path):
    from scripts.run_topic5_v3p_trajectory import main
    main(["--cohort", "narrow", "--outdir", str(tmp_path), "--n-perm", "20", "--subjects", "__none__"])
    df = pd.read_csv(tmp_path / "v3p_trajectory_subject.csv")
    for c in ["net_offaxis_flux_surplus_slope","mode_shift_density_surplus_slope","p_label_slope_b","p_label_slope_c","status"]:
        assert c in df.columns
@pytest.mark.integration
def test_v3p_trajectory_runs_on_eligible_subject(tmp_path):
    from scripts.run_topic5_v3p_trajectory import main
    main(["--cohort", "narrow", "--outdir", str(tmp_path), "--n-perm", "50", "--subjects", "253"])
    df = pd.read_csv(tmp_path / "v3p_trajectory_subject.csv")
    row = df[df.subject.astype(str) == "253"].iloc[0]
    if row["status"] == "ok":
        assert np.isfinite(row["net_offaxis_flux_surplus_slope"])
        assert np.isfinite(row["mode_shift_density_surplus_slope"])
        assert 0.0 <= row["p_label_slope_b"] <= 1.0
```

- [ ] **Step 2: Run fail. Step 3: Implement** (argparse `--cohort/--outdir/--n-perm/--subjects`; `--subjects __none__` → write header-only CSV for the skipped-path test). **Step 4: Run pass (narrow + broad, `--n-perm 50` smoke).**
- [ ] **Step 5: Commit** — `git add scripts/run_topic5_v3p_trajectory.py && git commit -m "feat(topic5-v3p): trajectory runner co-primary (flux+mode-shift slopes, label-null primary + secondary nulls + residual + gates)"`

---

## Task 8: Supportive H3p-a + secondary H3p-d columns — integration

**Files:** Modify `scripts/run_topic5_v3p_trajectory.py`; Test `tests/test_topic5_v3p_integration.py`.
**Interfaces:**
- Produces: additional subject-CSV cols: `K_primary_metric(=line_length_rate), beta_axis_strength_slope, beta_axis_reliable, beta_axis_slope_z, p_label_slope_a, module_support_flag_a(=False),` `nonaxis_activation_burden_slope_raw, nonaxis_activation_burden_slope_label_surplus, nonaxis_activation_burden_slope_resid, burden_slope_z, p_label_burden,` `N_self_sustain_lag1_slope, N_self_sustain_lag0_slope, N_self_sustain_lag1_specific_slope, N_self_sustain_slope_z, p_label_selfsustain,` `gain_axis_slope, gain_nonaxis_slope, gain_shift_slope, gain_nonaxis_surplus_slope, gain_shift_slope_z`.

**Contract:** reuse the SAME per-seizure window loop already computing the atom (do not re-load). H3p-a `beta_axis_strength` trajectory → Theil-Sen slope, expected `< 0`, `p_label_slope_a` via `surplus_and_p(obs, null, "less")`; `beta_axis_reliable = median(|β_axis|) ≥ beta_axis_reliability_min` (else not interpretable); **`module_support_flag_a` hard-coded False** (supportive-only). H3p-d (rev1): burden reports `_raw` + `_label_surplus` (obs − label-null median) + `_resid` (vs `global_activation_rate(t)`); **`N_self_sustain_lag1_specific_slope = slope(N_self_sustain_lag1) − slope(N_self_sustain_lag0)`** (≤0 → synchronous co-activation, NOT self-sustain — record but don't claim self-sustain); **gain leg primary = `gain_shift_slope = slope(gain_nonaxis − gain_axis)`** (relative; `gain_nonaxis_surplus_slope` kept for reference). All with `_z` from `surplus_and_p`; directions `greater` except beta_axis.

- [ ] **Step 1: Failing integration test** (eligible subject → `beta_axis_strength_slope`, `nonaxis_activation_burden_slope_resid`, `N_self_sustain_slope`, `gain_nonaxis_surplus_slope`, `module_support_flag_a==False` present + `p_label_slope_a` in [0,1]). **Step 2: Run fail. Step 3: Implement. Step 4: Run pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(topic5-v3p): supportive beta_axis slope (H3p-a) + secondary burden/self-sustain/gain slopes (H3p-d)"`

---

## Task 9: Summary + tier verdict (Holm co-primary) — integration

**Files:** Create `scripts/run_topic5_v3p_summary.py`; Test `tests/test_topic5_v3p_integration.py`.
**Interfaces:**
- Consumes: `v3p_trajectory_subject.csv` (narrow + broad).
- Produces: `v3p_summary_subject.csv` + `v3p_cohort_tier.json` per cohort. **tier assigned ONLY here.**

**Contract:**
- `subject_support = (module_support_flag_b OR module_support_flag_c) AND onset_jitter_pass AND (not single_contact_driven) AND leave_one_contact_pass AND axis_only_control_pass AND (not near_onset_dependent_of_the_supporting_leg) AND (not label_null_underpowered)`. H3p-a significant only strengthens; never sole. `geometry_insufficient`/`skipped` excluded from denominator. **(rev1: `near_onset_dependent` → that leg does not count as strong support; `label_null_underpowered` subject excluded from the strong-positive denominator.)**
- **cohort-level: Holm-correct the two co-primary p-values** — take the per-subject `slope_label_z` for H3p-b and H3p-c, Wilcoxon signed-rank (one-sided, direction correct) on subject-median across the cohort → two raw p; Holm-adjust the pair. narrow tier-3 needs a Holm-passed H3p-b OR H3p-c + subject-support count ≥ (config threshold, default ≥2).
- `tier`: 0 none / 1 descriptive-direction-only / 2 ≥1 subject support, no cohort direction / 3 **narrow cohort co-primary (Holm-passed)** / 4 narrow + broad same-direction replication / (5 = model-side, out of scope). `state_v3p_supported = tier>=3`. **narrow + broad never pooled.** Emit `pre_registered_negative` flag = True when tier ≤ 1 (honest-negative path). **(rev2) broad replication = `broad_expanded`; ALSO compute + emit the `broad_core`-only verdict (`tier_broad_core`) — tier 4 requires the direction to hold on `broad_core` too (expansion adds power, never rescues a curated-subset null). `yuquan` supplement reported descriptively, never pooled.**

- [ ] **Step 1: Failing integration test** (summary + cohort JSON have `tier`, `state_v3p_supported`, Holm-corrected `p_holm_b`/`p_holm_c`, per-cohort separation, denominator = geometry-sufficient count, `pre_registered_negative`). **Step 2: Run fail. Step 3: Implement. Step 4: Run pass.**
- [ ] **Step 5: Commit** — `git add scripts/run_topic5_v3p_summary.py && git commit -m "feat(topic5-v3p): summary + tier 0-5 (Holm co-primary, narrow primary, honest-negative flag)"`

---

## Task 10: Result figure — integration

**Files:** Create `scripts/plot_topic5_v3p_summary.py`; Test `tests/test_topic5_v3p_integration.py`.
**Contract:** 2–3 independent-question panels (CLAUDE.md §7), paper-grade self-contained (`docs/figure_style_guide.md`; render→eyeball→fix): **(A)** per-subject co-primary surplus-slopes — H3p-b `net_offaxis_flux_surplus_slope` and H3p-c `mode_shift_density_surplus_slope`, narrow vs broad, cohort-median bars, zero line, label-null-significant subjects marked; **(B)** preictal phase trajectory `P0→P3` of `mode_shift_density` and `net_offaxis_flux` (mean ± IQR band across seizures/subjects, from `v3p_window_detail.csv`); **(C optional)** per-subject `slope_label_z` (H3p-b, H3p-c) with `±1.96` guides. Write `figures/README.md` (中文, 每图 2–4 句 + `**关注点**：`) + append `results/FIGURE_INDEX.md`.

- [ ] **Steps:** build; render; **eyeball PNG**; fix; smoke test (PNG + README exist). **Commit** — `git add scripts/plot_topic5_v3p_summary.py && git commit -m "feat(topic5-v3p): result figure + README + FIGURE_INDEX"`

---

## Final run + Hard QC

- [ ] **Run (narrow primary first), after Task 1 min_windows lock:**

```bash
for ax in narrow broad; do
  python scripts/run_topic5_v3p_feasibility.py  --cohort $ax
  python scripts/run_topic5_v3p_trajectory.py   --cohort $ax --n-perm 1000
  python scripts/run_topic5_v3p_summary.py      --cohort $ax
done
python scripts/plot_topic5_v3p_summary.py
pytest tests/test_topic5_v3p_preictal_trajectory.py -v
pytest -m integration tests/test_topic5_v3p_integration.py -v
```

- [ ] **Hard QC:** windows eeg-onset anchored + **preictal-only** (no O/I1/I2/I3); trend = **Theil-Sen slope** (median over seizures); **support = H3p-b OR H3p-c**, H3p-a `module_support_flag_a` always False + only interpretable if `beta_axis_reliable`; **label-null-of-slope is the primary p** (surplus = obs − null median); residualization reported but **never gates** (collinearity floor documented); VAR **demeaned within window**; λ reported as `λ_surplus`; mode-shift = density-normalized singular-vector; ATM `i≠j`; `single_contact_driven`/`axis_only_control`/`leave_one_contact` computed (not asserted); **H3p-b/H3p-c Holm-corrected** at cohort level; tier only in summary; narrow primary / broad replication / **never pooled**; `geometry_insufficient` ≠ negative; jitter ±10 s stable; **no forecasting** language anywhere; **1125 (or any single subject) → descriptive case only, never rescues a cohort negative.** **(rev1) two-span (full+guard) both required for strong support; `near_onset_dependent` (full-only) → tier cap 2; H3p-b HARD gates `p_rate_slope_b` + `lag1_specific_slope>0`; rate null is PER-WINDOW; H3p-c HARD gates `p_phase_slope_c` + `p_block_slope_c` (strong vs weak grade); `gain_shift = gain_nonaxis − gain_axis`; `N_self_sustain_lag1_specific` (lag1−lag0); QC `mode_singular_gap_median`/`mode_vector_stable`/`h3b_activation_sufficient`/`label_null_underpowered` emitted, underpowered → excluded from strong-positive denominator.**

---

## Self-Review

1. **Spec coverage:** preictal-only P0..P3 → Task 1/7; Theil-Sen slope → Task 2; co-primary flux+mode-shift → Task 7; label-null primary adjudicator → Task 6/7; regression residual sensitivity → Task 4/7; N→N self-sustain → Task 3/8; H3p-a supportive (`module_support_flag_a=False`) → Task 8; H3p-d secondary → Task 8; Holm co-primary + tier → Task 9; geometry/dynamics inherited-locked → Global Constraints + Task 5; read-only V3a reuse / new-files-only → File Structure; feasibility gate → Task 1; onset jitter → Task 7; figure → Task 10; no-forecasting/negative-acceptable → Global Constraints + Task 9. **rev1 coverage:** onset-guard two-span (full+guard, `near_onset_dependent`) → Global Constraints + config + Task 7 (Observed/flags); H3p-b HARD rate + lag1_specific → Task 5 (lag0/lag1 flux) + Task 7 flags; rate-null-per-window → Task 6 closure + Task 7; H3p-c HARD phase+block + strong/weak grade → Task 7; singular-gap/`mode_vector_stable`/`cv_r2` → Task 5 (`mode_singular_gap`) + Task 7; `h3b_activation_sufficient`/`n_activation_events_pre` → Task 5 + Task 7; `gain_shift_slope` → Task 5 (gain_axis/nonaxis) + Task 8; `N_self_sustain_lag1_specific` → Task 5 (lag0/lag1) + Task 8; label-perm QC + `label_null_underpowered` → Task 1 + Task 7 + Task 9 denominator; time-order null → Task 6 closure + Task 7. **Covered.**
2. **Placeholder scan:** pure fns (Tasks 2–6) literal tests + full impl; runners (Tasks 1,7,8,9,10) exact columns + contract + null/gate logic as code; OPEN `min_windows_for_slope` pilot-locked at Task 1. **OK.**
3. **Type consistency:** `slope_over_windows` (Task 2) → 4/6/7/8; `within_compartment_flux`/`global_axial_energy` (Task 3) → 5; `residualize_slope` (Task 4) → 7/8; `extract_window_metrics(env_win, geom, v3cfg)` (Task 5) → 7/8; `null_slope_distribution(resample_traj_fn,...)`/`surplus_and_p(obs, null, direction)` (Task 6) → 7/8; subject-CSV col names (Task 7/8) → 9 join; `slope_label_z` (Task 7/8) → 9 Wilcoxon. **Consistent.**

---

## Execution Handoff

**Subagent-Driven (recommended)** — fresh subagent per task; **Task 1 feasibility is a min_windows-lock gate (STOP if <4 narrow subjects qualify).** Order: 0→**1(gate)**→2→3→4→5→6→7(H3p-b/c)→8(H3p-a/d)→9→10. Worktree already created (`.worktrees/topic5-v3p-preictal-trajectory`, off V3a HEAD `ac042f3`, cache symlinked). **V3p adds only new files — never edit a V3a/V2 file** (keeps the still-open V3a branch merge clean).
