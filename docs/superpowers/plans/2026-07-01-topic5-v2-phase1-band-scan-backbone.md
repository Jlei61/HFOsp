# Topic 5 V2 Phase 1 (1a–1c) — Frequency-Scan Backbone Implementation Plan (rev2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **rev2 (post plan-review patches A–I):** (A) config splits `power.spectrogram_*` vs `epoch.field_*`; (B) legacy CSV per axis_set + repro bands separated from analysis bands; (C) primary bands half-open `[lo,hi)`; (D) residual features WIRED into alignment/null/gate via new Tasks 10b/11b/12a; (E) null orchestration emits a **permutation-level long table** so max-over-bands null is computable; (F) `order_null_rank_pair` rebuilds A/B; (G) signed metric uses a **pre-fixed orientation** (post-hoc template choice demoted to sensitivity); (H) confound is a **deterministic adjustment table**, not a permutation `null_type`; (I) real-data scripts are `@pytest.mark.integration` + `--outdir`. Plus: subject-fixed contact mask across bands, spatial-null fallback + strength, explicit gate-pass thresholds, artifact/bad-channel QC, loader factored (Task 6a).

**Goal:** Build a reproducible, gate-driven frequency-scan backbone that answers: does early-ictal band-power field align with the interictal HFO timing geometry (`G_HFO`) beyond spatial + HFO-rate-order nulls (Gate A), beyond broadband recruitment via leave-one-band-out common-field residual (Gate B), and specifically in ripple 80–250 Hz after aperiodic 1/f control (Gate C)?

**Architecture:** Reuse the existing `align_maxab` pipeline (`scripts/run_topic5_ictal_field_dynamics.py::load_context`/`window_maxab`) as the legacy backbone; first prove reproduction of the committed `align_maxab` numbers (per axis_set), then extend to config-driven multi-band masked cache, contact-level metrics, three residual feature streams (raw / common-field-LOBO / aperiodic-residual), a three-layer null with a permutation-level long output, and a Gate A/B/C decision table with explicit thresholds. Pure math in `src/topic5_v2_band_scan.py`; orchestration in `scripts/run_topic5_v2_*.py` driven by `config/topic5_v2_phase1.yaml`.

**Tech Stack:** Python, numpy, scipy.stats, pandas/pyarrow (parquet), PyYAML, pytest. Reuses `src.topic5_ictal_recruitment` (`_spectrogram_on_hop`, `baseline_robust_z`), `scripts.run_topic5_ictal_field_dynamics` (`load_context`, `window_maxab`, `_slice`, `_zmean_by_name`), `src.propagation_contact_plane_readout` (`R_smooth_rank`, `make_field_record`), `src.interictal_propagation` (`load_subject_propagation_events`).

## Global Constraints

- **Reference geometry `G_HFO` = interictal HFO `typical_rank`; FIXED** and always **candidate** mode (§1.1) until the evidence ladder passes. Never rebuild from ictal data (order-null preserves participation counts only).
- **Subject is the unit of cohort inference.** Fixed aggregation: window → seizure median → subject median → cohort median. No window/seizure-level p-values.
- **`broad`/`narrow` axis sets are NEVER pooled.** One row per `axis_set`; cohort summaries per `axis_set`.
- **Every permutation null replicates the FULL statistic** (per-window metric → A/B max → window→seizure→subject) AND emits per-permutation per-band subject-level values (for max-over-bands).
- **Two window scales are distinct:** `power.spectrogram_win_sec` (1.0s, power estimation) ≠ `epoch.field_window_sec` (10s, field alignment window). Never conflate.
- **Line-noise harmonics (50 Hz; 50/100/150/200/250) excluded at FFT-bin level BEFORE integration**; every band row carries `band_eff_frac`; rows `< min_effective_bandwidth_frac` are excluded from PRIMARY.
- **512 Hz subjects (`epilepsiae_139`,`epilepsiae_253`) get `fs_edge_flag` for hi>220.** Ripple double-reported: `ripple_safe_80_220` (full-cohort) vs `ripple_full_80_250` (1024-subset); never pooled.
- **Subject-fixed contact mask across bands:** `analysis_channels[subject,axis_set]` = channels finite/QC-good across ALL primary bands; PRIMARY metrics use this fixed mask (band-wise mask only as sensitivity).
- **Phantom-rank discipline:** any rank rebuilt from `lagPatRank` uses `eventsBool` participation masking.
- **Denominator = `phase1_cohort_manifest` (Task 0), NOT globbed result files** (P1-a). Every summary/null/gate's `n_subjects_valid` = manifest `included` count per axis_set; artifact presence ≠ analysis eligibility.
- **Null strengths gate formal tiers** (P1-c): only `spatial_null_strength=='within_shaft_strong'` and `order_null_strength!='weak_downgrade'` support formal Gate A; weaker nulls → descriptive/sensitivity only.
- **Signed direction uses a PRE-FIXED template orientation** (never post-hoc argmax); post-hoc-max signed is sensitivity only.
- **Confound residualization is a deterministic ADJUSTMENT** (its own table), not a permutation `null_type`; if p-values on residualized `G_HFO` are wanted, spatial/order nulls are re-run on the adjusted rank.
- **Real-data scripts are `@pytest.mark.integration` and accept `--outdir`** (default a `_test` dir in tests). Pure-function tests are plain pytest. Tests default to `n_perm_smoke`.
- **TDD, DRY, YAGNI, frequent commits. No FOOOF dependency.**
- **Out of scope (follow-on plan, gated on Gate A):** Phase 1d phenotyping, 1e PAC, Phase 2, fast-ripple confirmatory.

---

## File Structure

- `config/topic5_v2_phase1.yaml` — bands (half-open primary / closed composite), repro bands (separate), power vs field windows, line-noise, edge, artifact QC, metrics, cohorts, common-field LOBO, null params (smoke/dev/final), gate alpha, tolerances.
- `src/topic5_v2_band_scan.py` — pure math (all functions below).
- `tests/test_topic5_v2_band_scan.py` — pure-function TDD.
- `tests/test_topic5_v2_integration.py` — `@pytest.mark.integration` smoke over real data with `--outdir`.
- `scripts/run_topic5_v2_legacy_repro.py` — Task 4.
- `scripts/build_topic5_v2_band_cache.py` — Task 6 (reuses Task-6a loader).
- `scripts/run_topic5_v2_alignment.py` — Task 7 (raw + wires residual alignment in 10b/11b via `--feature`).
- `scripts/build_topic5_v2_confound_maps.py` — Task 12a.
- `scripts/run_topic5_v2_nulls.py` — Task 13 (two-layer output).
- `scripts/run_topic5_v2_gates.py` — Task 14 (thresholds + max-over-bands).

Outputs under `results/topic5_ictal_recruitment/v2_band_scan/{axis_set}/`.

---

# PHASE 1a — Config + Legacy Reproduction

### Task 0: Cohort manifest (denominator truth — P1-a)

**Files:** Create `scripts/build_topic5_v2_cohort_manifest.py`; Test `tests/test_topic5_v2_integration.py`.
**Interfaces:** Produces `results/topic5_ictal_recruitment/v2_band_scan/phase1_cohort_manifest.{csv,json}`.

**Why (P1-a):** there is NOT one denominator. `field_dynamics` = broad 9 / narrow 7; `ictal_field_long_cache` ≈ 16; `propagation_geometry_broad` ≈ 23. A script that "scans available files" would silently conflate **artifact presence** with **analysis eligibility**. The manifest is the SINGLE source of truth; every summary/null/gate references it (`n_subjects_valid` = manifest `included` count, per axis_set).

**Contract:** for each `subject × axis_set` write: `subject, axis_set, has_long_cache (ictal_field_long_cache npz), has_axis_record (geo t_a/t_b), has_order_event_data (lagPat eventsBool via load_subject_propagation_events), fs, included (bool), exclusion_reason ('' | no_long_cache | no_axis_record | ...)`. Include a `cohort_manifest_hash` (hash of the sorted included set) written to the JSON. `included` requires `has_long_cache AND has_axis_record`; `has_order_event_data=False` does NOT exclude (it downgrades the order-null to `weak_downgrade`, Task 9). The candidate subject lists come from `SUBJECTS_BY_SUB` (both axis sets), NOT from globbing result files.

- [ ] **Step 1: Write failing integration test**

```python
# in tests/test_topic5_v2_integration.py
import pytest, subprocess, sys, csv, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
@pytest.mark.integration
def test_cohort_manifest_denominator(tmp_path):
    r = subprocess.run([sys.executable, "scripts/build_topic5_v2_cohort_manifest.py",
                        "--outdir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(open(tmp_path / "phase1_cohort_manifest.csv")))
    axes = {x["axis_set"] for x in rows}
    assert {"broad", "narrow"} <= axes
    for x in rows:
        assert x["included"] in ("True", "False") and "exclusion_reason" in x
    meta = json.load(open(tmp_path / "phase1_cohort_manifest.json"))
    assert "cohort_manifest_hash" in meta
```

- [ ] **Step 2: Run to verify fail** — `pytest -m integration tests/test_topic5_v2_integration.py -k cohort_manifest -v`.
- [ ] **Step 3: Implement** — iterate `SUBJECTS_BY_SUB[axis]`; probe each artifact path; classify; hash the included set; write CSV+JSON+`--outdir`.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `git add scripts/build_topic5_v2_cohort_manifest.py tests/test_topic5_v2_integration.py && git commit -m "feat(topic5-v2): cohort manifest (single denominator truth)"`

---

### Task 1: Phase-1 config (rev2)

**Files:** Create `config/topic5_v2_phase1.yaml`; Test `tests/test_topic5_v2_band_scan.py`.
**Interfaces:** `load_phase1_config(path=None) -> dict`.

- [ ] **Step 1: Write the config**

```yaml
# config/topic5_v2_phase1.yaml  (rev2)
power:
  spectrogram_win_sec: 1.0
  spectrogram_hop_sec: 0.1
epoch:
  field_window_sec: 10.0
  field_step_sec: 5.0
  main_rel: [0.0, 20.0]
  ictal_fraction_min: 0.5
bands:
  primary:                                   # HALF-OPEN [lo, hi)
    - [delta_HYP_slow, 1, 4]
    - [theta_preictal_PAC, 4, 8]
    - [alpha_sharp_leq13, 8, 13]
    - [beta_LVFA_low, 13, 30]
    - [gamma_LVFA, 30, 80]
    - [hg_low_ripple, 80, 150]
    - [ripple_high, 150, 250]
  composites:                                # CLOSED [lo, hi]
    - [low_HYP_1_13, 1, 13]
    - [LVFA_13_80, 13, 80]
    - [ripple_full_80_250, 80, 250]
    - [ripple_safe_80_220, 80, 220]
  primary_interval: half_open
  composite_interval: closed
repro_bands:                                 # legacy reproduction ONLY (from OLD cache, not v2-recomputed)
  bb: legacy_bb_1_45
  hfa: legacy_hfa_60_100
line_noise: {mains_hz: 50.0, harmonics_hz: [50,100,150,200,250], halfwidth_hz: 2.0, min_effective_bandwidth_frac: 0.5}
edge: {fs512_hi_safe_hz: 220.0}
artifact: {flatline_mad_eps: 1.0e-9, saturation_abs_z: 12.0, saturation_frac: 0.02}
metrics:
  primary: align_abs_maxab
  signed_fixed_orientation: align_signed_oriented
  sensitivity: [align_signed_a, align_signed_b, align_signed_posthoc_max, align_spearman_a, align_spearman_b]
cohorts: {axis_sets: [broad, narrow], never_pool_axis_sets: true}
common_field: {broadband_band: [1, 250], leave_one_band_out: true}
nulls:
  n_perm_smoke: 20
  n_perm_dev: 100
  n_perm_final: 1000
  seed: 20260701
  alpha: 0.05
  spatial: within_shaft
  min_group_for_shaft: 4
  order_null_min_corr_to_geo: 0.90
  confound_covariates: [hfo_rate, baseline_band_power, broadband_1_250, shaft_position]
tolerances: {legacy_subject_median_abs: 0.02}
```

- [ ] **Step 2: Failing loader test**

```python
from src.topic5_v2_band_scan import load_phase1_config
def test_config_rev2_keys():
    c = load_phase1_config()
    assert c["power"]["spectrogram_win_sec"] == 1.0
    assert c["epoch"]["field_window_sec"] == 10.0
    assert c["bands"]["primary_interval"] == "half_open"
    assert c["repro_bands"]["hfa"] == "legacy_hfa_60_100"
    assert c["nulls"]["n_perm_smoke"] == 20 and c["common_field"]["leave_one_band_out"] is True
```

- [ ] **Step 3: Run fail** — `pytest tests/test_topic5_v2_band_scan.py::test_config_rev2_keys -v`.
- [ ] **Step 4: Create module + loader**

```python
# src/topic5_v2_band_scan.py
from __future__ import annotations
from pathlib import Path
import numpy as np, yaml
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CFG = _ROOT / "config/topic5_v2_phase1.yaml"
def load_phase1_config(path=None) -> dict:
    with open(path or _DEFAULT_CFG) as fh:
        return yaml.safe_load(fh)
```

- [ ] **Step 5: Run pass. Step 6: Commit** — `git add config/topic5_v2_phase1.yaml src/topic5_v2_band_scan.py tests/test_topic5_v2_band_scan.py && git commit -m "feat(topic5-v2): rev2 config (window split, half-open bands, repro/analysis split)"`

---

### Task 2: Line-noise mask + effective bandwidth (half-open aware)

**Files:** Modify `src/topic5_v2_band_scan.py`; Test same.
**Interfaces:** `line_noise_bin_mask(freqs, harmonics_hz, halfwidth_hz)->bool[]`; `band_bin_selection(freqs, lo, hi, line_mask, half_open=False)->(band_mask, eff_frac, n_band_bins)` where `half_open` uses `freqs<hi` (primary) else `freqs<=hi` (composite).

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v2_band_scan import line_noise_bin_mask, band_bin_selection
def test_half_open_bands_do_not_share_boundary_bin():
    f = np.arange(0, 251, 1.0); lm = line_noise_bin_mask(f, [50,100,150,200,250], 2.0)
    # delta [1,4) and theta [4,8): 4 Hz belongs to theta only
    dmask,_,_ = band_bin_selection(f, 1, 4, lm, half_open=True)
    tmask,_,_ = band_bin_selection(f, 4, 8, lm, half_open=True)
    assert not dmask[f==4].any() and tmask[f==4].all()
    # composite closed keeps hi
    cmask,_,_ = band_bin_selection(f, 80, 250, lm, half_open=False)
    assert not cmask[f==250].any()             # 250 is line harmonic -> masked anyway
    _, eff_bb, _ = band_bin_selection(f, 1, 45, lm, half_open=False)
    assert eff_bb == 1.0
```

- [ ] **Step 2: Run fail. Step 3: Implement**

```python
def line_noise_bin_mask(freqs, harmonics_hz, halfwidth_hz):
    freqs = np.asarray(freqs, float); m = np.zeros(freqs.shape, bool)
    for h in harmonics_hz: m |= np.abs(freqs - float(h)) <= float(halfwidth_hz)
    return m
def band_bin_selection(freqs, lo, hi, line_mask, half_open=False):
    freqs = np.asarray(freqs, float)
    in_band = (freqs >= float(lo)) & ((freqs < float(hi)) if half_open else (freqs <= float(hi)))
    n_band = int(in_band.sum())
    band_mask = in_band & ~np.asarray(line_mask, bool)
    return band_mask, float(band_mask.sum()) / max(n_band, 1), n_band
```

- [ ] **Step 4: Run pass. Step 5: Commit** — `git commit -am "feat(topic5-v2): half-open-aware line-noise mask + effective bandwidth"`

---

### Task 3: Masked band-power trace + robust-z + artifact/bad-channel flags

**Files:** Modify `src/topic5_v2_band_scan.py`; Test same.
**Interfaces:**
- `masked_band_power_trace(signal, fs, lo, hi, spec_win_sec, spec_hop_sec, harmonics_hz, halfwidth_hz, fs512_hi_safe, half_open=False)->dict{logpower,t,eff_frac,fs_edge_flag,n_band_bins}` (Nyquist-gated).
- `channel_artifact_flags(logpower, z, sat_abs_z, sat_frac, flatline_mad_eps)->dict{flatline: bool[], saturation: bool[], bad_channel: bool[]}` — flatline: baseline MAD≈0 (all-NaN z row); saturation: fraction of `|z|>sat_abs_z` exceeds `sat_frac`; bad = flatline | saturation.
- `robust_z_with_flags(logpower, baseline_idx, hop_sec, min_baseline_valid_sec)->(z, low_baseline_flag: bool[])` (unchanged wrapper).

**Note (issue #19):** first grep `src/preprocessing.py`, `src/hfo_detector.py` for an existing per-channel quality/bad-channel mask; if present, REUSE it and only add saturation/flatline on top.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v2_band_scan import masked_band_power_trace, robust_z_with_flags, channel_artifact_flags
def test_band_power_flags_and_edge():
    rng=np.random.default_rng(0); fs=1024.0; sig=rng.standard_normal((4,int(fs*40)))
    out=masked_band_power_trace(sig, fs, 80, 250, 1.0, 0.1, [50,100,150,200,250], 2.0, 220.0)
    assert out["fs_edge_flag"] is False and 0<out["eff_frac"]<1
    out512=masked_band_power_trace(sig[:,:int(512*40)], 512.0, 80, 250, 1.0, 0.1, [50,100,150,200,250], 2.0, 220.0)
    assert out512["fs_edge_flag"] is True
    n_t=out["logpower"].shape[1]; z,low=robust_z_with_flags(out["logpower"], (0,n_t//3), 0.1, 1.0)
    zz=z.copy(); zz[1,:]=50.0                                        # saturate ch1
    fl=channel_artifact_flags(out["logpower"], zz, 12.0, 0.02, 1e-9)
    assert fl["saturation"][1] and fl["bad_channel"][1]
```

- [ ] **Step 2: Run fail. Step 3: Implement**

```python
def masked_band_power_trace(signal, fs, lo, hi, spec_win_sec, spec_hop_sec,
                            harmonics_hz, halfwidth_hz, fs512_hi_safe, half_open=False):
    from src.topic5_ictal_recruitment import _spectrogram_on_hop
    nyq=float(fs)/2.0
    if hi>=nyq: raise ValueError(f"band hi {hi} >= Nyquist {nyq} for fs={fs}")
    f,t,Sxx=_spectrogram_on_hop(signal, fs, spec_win_sec, spec_hop_sec)
    lm=line_noise_bin_mask(f, harmonics_hz, halfwidth_hz)
    bmask,eff,n_band=band_bin_selection(f, lo, hi, lm, half_open=half_open)
    if not bmask.any(): raise ValueError(f"no bins in ({lo},{hi}) after line mask")
    power=Sxx[:,bmask,:].sum(axis=1)
    return {"logpower":np.log(np.maximum(power,1e-30)),"t":t,"eff_frac":eff,
            "fs_edge_flag":bool(float(fs)<=512.0 and float(hi)>float(fs512_hi_safe)),"n_band_bins":n_band}
def robust_z_with_flags(logpower, baseline_idx, hop_sec, min_baseline_valid_sec):
    from src.topic5_ictal_recruitment import baseline_robust_z
    z=baseline_robust_z(logpower, baseline_idx, hop_sec=hop_sec, min_baseline_valid_sec=min_baseline_valid_sec)
    return z, np.all(~np.isfinite(z), axis=1)
def channel_artifact_flags(logpower, z, sat_abs_z, sat_frac, flatline_mad_eps):
    z=np.asarray(z,float); flat=np.all(~np.isfinite(z),axis=1)
    with np.errstate(invalid="ignore"):
        sat=np.nanmean(np.abs(z)>float(sat_abs_z), axis=1) > float(sat_frac)
    sat=np.where(np.isfinite(sat), sat, False)
    bad=flat|sat
    return {"flatline":flat, "saturation":sat, "bad_channel":bad}
```

- [ ] **Step 4: Run pass. Step 5: Commit** — `git commit -am "feat(topic5-v2): masked band power + robust-z + artifact/bad-channel flags"`

---

### Task 4: Legacy bb/hfa reproduction QC (per axis_set, hard gate)

**Files:** Create `scripts/run_topic5_v2_legacy_repro.py`; Test `tests/test_topic5_v2_integration.py`.
**Interfaces:** Produces `results/topic5_ictal_recruitment/v2_band_scan/{axis_set}/phase1_qc_legacy_reproduction.csv` cols `subject, axis_set, band(bb|hfa), n_seizures, n_windows, old_subject_median, new_subject_median, delta`. Exits non-zero if any `|delta|>tol`.

**Patch B:** `LEGACY_CSV_BY_AXIS = {"broad": field_dynamics/per_seizure_metrics.csv, "narrow": field_dynamics_narrow/per_seizure_metrics.csv}`. Reproduction recomputes bb/hfa via the EXISTING `ictal_field_long_cache` (`bb_zt`/`hfa_zt`, unmasked, as the old pipeline) + `window_maxab` — NOT via the v2 masked cache. `n_seizures`/`n_windows` emitted.

**P1-d — two SEPARATE QCs (do not conflate unmasked reproduction with fixed-mask v2):**
- **QC-1 (HARD GATE):** old-pipeline **unmasked, all-channel** reproduction — new-legacy bb/hfa vs committed `align_maxab` per subject, `|delta| <= legacy_subject_median_abs`. This uses the OLD channel set (no fixed mask). Exit non-zero on failure.
- **QC-2 (RECORD-ONLY):** v2 **fixed-mask** (`analysis_channels`) `legacy_bb_1_45` vs QC-1 new-legacy bb → `fixed_mask_delta` column, and `n_channels_dropped_by_fixed_mask`. **This delta is NOT a fail gate** — a nonzero delta is expected when the fixed mask drops bad channels. It only becomes a gate IF `n_channels_dropped_by_fixed_mask==0` for that subject (then it must also match). So Task-7's fixed-mask legacy cross-check (Step 1) is RECORD-ONLY unless zero channels were dropped.

- [ ] **Step 1: Failing integration test**

```python
# tests/test_topic5_v2_integration.py
import subprocess, sys, csv
import pytest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
@pytest.mark.integration
@pytest.mark.parametrize("axis", ["broad", "narrow"])
def test_legacy_reproduction_within_tolerance(axis, tmp_path):
    r = subprocess.run([sys.executable, "scripts/run_topic5_v2_legacy_repro.py",
                        "--substrate", axis, "--outdir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(open(tmp_path / axis / "phase1_qc_legacy_reproduction.csv")))
    assert rows and all("n_seizures" in x for x in rows)
    for x in rows: assert abs(float(x["delta"])) <= 0.02, f"{x['subject']} {x['band']} {x['delta']}"
```

- [ ] **Step 2: Run fail** — `pytest -m integration tests/test_topic5_v2_integration.py -k legacy -v`.
- [ ] **Step 3: Implement** — as rev1 Task 4 but with `LEGACY_CSV_BY_AXIS[substrate]`, `--outdir`, and `n_seizures`/`n_windows` columns (count seizures/windows entering each subject median). The `_new_align` recomputes from the EXISTING `CACHE` bb/hfa (unchanged from rev1).
- [ ] **Step 4: Run pass** (broad AND narrow). **HARD GATE: if any subject over tol, STOP — debug orchestration before Task 6.**
- [ ] **Step 5: Commit** — `git add scripts/run_topic5_v2_legacy_repro.py tests/test_topic5_v2_integration.py && git commit -m "feat(topic5-v2): per-axis legacy reproduction QC gate"`

---

# PHASE 1b — Multi-band cache + alignment + Gate-A nulls

### Task 5: Contact alignment metrics (fixed-orientation signed + per-template)

**Files:** Modify `src/topic5_v2_band_scan.py`; Test same.
**Interfaces (Patch G):** `contact_alignment(vals_by_name, rank_a_by_name, rank_b_by_name, oriented_template='a') -> dict` with:
- `signed_pearson_a, signed_spearman_a, signed_pearson_b, signed_spearman_b` (per template, NaN if `<4` shared);
- `align_signed_oriented` = signed Spearman against `oriented_template` (PRE-FIXED, the mechanistic direction);
- `align_signed_posthoc_max` = signed Spearman of the larger-|.| template (SENSITIVITY only);
- `align_abs_maxab_contact` = max(|spearman_a|,|spearman_b|) (contact-level abs, supportive);
- `n_contacts_a, n_contacts_b`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v2_band_scan import contact_alignment
def test_signed_orientation_is_fixed_not_posthoc():
    names=[f"c{i}" for i in range(8)]
    ra={n:float(i) for i,n in enumerate(names)}; rb={n:float(7-i) for i,n in enumerate(names)}
    vals={n:float(7-i) for i,n in enumerate(names)}     # tracks B (anti-A)
    out=contact_alignment(vals, ra, rb, oriented_template="a")
    assert out["signed_spearman_a"] < -0.9              # anti-correlated with A
    assert out["align_signed_oriented"] == out["signed_spearman_a"]   # fixed to A regardless
    assert out["align_signed_posthoc_max"] > 0.9        # posthoc would pick B (positive)
```

- [ ] **Step 2: Run fail. Step 3: Implement**

```python
def contact_alignment(vals_by_name, rank_a_by_name, rank_b_by_name, oriented_template="a"):
    from scipy.stats import spearmanr, pearsonr
    def _one(rank_by):
        names=[n for n in vals_by_name if n in rank_by
               and np.isfinite(vals_by_name[n]) and np.isfinite(rank_by[n])]
        if len(names)<4: return None
        v=np.array([vals_by_name[n] for n in names]); r=np.array([rank_by[n] for n in names])
        if np.std(v)==0 or np.std(r)==0: return None
        return {"sp":float(spearmanr(v,r).statistic),"pe":float(pearsonr(v,r)[0]),"n":len(names)}
    a,b=_one(rank_a_by_name),_one(rank_b_by_name)
    def g(o,k,d=float("nan")): return o[k] if o else d
    posthoc=max([o for o in (a,b) if o], key=lambda o:abs(o["sp"]), default=None)
    return {"signed_pearson_a":g(a,"pe"),"signed_spearman_a":g(a,"sp"),
            "signed_pearson_b":g(b,"pe"),"signed_spearman_b":g(b,"sp"),
            "align_signed_oriented":(g(a,"sp") if oriented_template=="a" else g(b,"sp")),
            "align_signed_posthoc_max":(posthoc["sp"] if posthoc else float("nan")),
            "align_abs_maxab_contact":max([abs(o["sp"]) for o in (a,b) if o], default=float("nan")),
            "n_contacts_a":g(a,"n",0),"n_contacts_b":g(b,"n",0)}
```

- [ ] **Step 4: Run pass. Step 5: Commit** — `git commit -am "feat(topic5-v2): fixed-orientation signed + per-template alignment"`

---

### Task 6a: Factor out the seizure-window loader (removes module-name ambiguity, issue #7)

**Files:** Modify `scripts/build_topic5_ictal_field_long_cache.py`; Test `tests/test_topic5_v2_integration.py`.
**Interfaces:** Add `iter_subject_seizure_windows(ds_sid, substrate) -> Iterator[(idx, sw, eeg_rel)]` factored from the existing per-seizure loop (same `post_sec`/`span>600` drops, same baseline resolution inputs). Export `HOP, GUARD_SEC, MIN_BASELINE_SEC`. Existing long-cache behavior must be unchanged (regression: rebuild one subject, assert `bb_auc` identical).

- [ ] **Step 1:** Read `scripts/build_topic5_ictal_field_long_cache.py`; identify the per-seizure loop that produces `sw`/`eeg_rel`.
- [ ] **Step 2: Failing test** — `@pytest.mark.integration`: import `iter_subject_seizure_windows`; assert it yields ≥1 tuple for `epilepsiae_139`.
- [ ] **Step 3: Implement** the factor-out (pure move; no logic change).
- [ ] **Step 4: Regression** — rebuild `epilepsiae_139` long cache; assert `bb_auc__*` unchanged vs committed.
- [ ] **Step 5: Commit** — `git commit -am "refactor(topic5-v2): factor out seizure-window loader for reuse"`

---

### Task 6: Multi-band masked band-power cache + subject-fixed mask

**Files:** Create `scripts/build_topic5_v2_band_cache.py`; Test `tests/test_topic5_v2_integration.py`.
**Interfaces:**
- Consumes: `iter_subject_seizure_windows` (Task 6a), `masked_band_power_trace`, `robust_z_with_flags`, `channel_artifact_flags`, `resolve_baseline_window`.
- Produces: `.../v2_band_scan/cache/{ds_sid}.npz` (per band `B`, seizure `idx`: `f"{B}__zt__{idx}"`, `f"{B}__relt__{idx}"`; `channels`) + sidecar JSON with per-`(B,idx)` `{eff_frac, fs_edge_flag, n_band_bins, low_baseline_channels, bad_channels}` and a subject-level `analysis_channels` = channel names finite & not-bad across ALL primary bands (issue #8). Half-open flag from `bands.primary_interval` for primary bands.

- [ ] **Step 1: Failing integration test** (`--subjects epilepsiae_139 --bands legacy_bb_1_45 ripple_full_80_250 --outdir tmp`): assert npz + sidecar exist; sidecar `ripple_full_80_250` `eff_frac<1`, `fs_edge_flag True`; sidecar has `analysis_channels` list.
- [ ] **Step 2: Run fail. Step 3: Implement** (loop `iter_subject_seizure_windows`; per config band call `masked_band_power_trace` with `half_open = name in primary`; robust-z; artifact flags; accumulate per-band good channels; write `analysis_channels = ∩ over primary bands of (finite ∧ ¬bad)`).
- [ ] **Step 4: Run pass**, then `--substrate broad`/`narrow` full.
- [ ] **Step 5: Commit** — `git add scripts/build_topic5_v2_band_cache.py && git commit -m "feat(topic5-v2): multi-band masked cache + subject-fixed analysis mask"`

---

### Task 7: Raw alignment tables (subject-fixed mask; window→seizure→subject; `--feature`)

**Files:** Create `scripts/run_topic5_v2_alignment.py`; Test `tests/test_topic5_v2_integration.py`.
**Interfaces:**
- `--feature {raw|common_resid|aperiodic_resid}` (default `raw`); `--outdir`. Reads the v2 cache (or residual caches from 10b/11b).
- Produces per axis_set + feature: `phase1_alignment_{feature}_window_long.csv`, `..._seizure_summary.csv`, `..._subject_summary.csv`. Window cols: `subject, axis_set, seizure, band, feature, win_start_rel, win_end_rel, win_center_rel, ictal_fraction, strict_onset, align_abs_maxab, align_signed_oriented, align_signed_posthoc_max, signed_spearman_a, signed_spearman_b, n_contacts, band_eff_frac, fs_edge_flag, used_fixed_mask`.

**Epoch rule (fixed):** valid early window iff `win_end_rel>0 and win_start_rel<20 and ictal_fraction>=field ictal_fraction_min`; `strict_onset = win_start_rel>=0 and win_end_rel<=20` (recorded, not primary). **PRIMARY uses `analysis_channels` fixed mask** (assert `analysis_channels ⊆ ctx["mapped"] ∩ cache_channels`); band-wise mask runs only under `--sensitivity`.

- [ ] **Step 1: Failing integration test** (`--feature raw --substrate broad --subjects epilepsiae_139 --outdir tmp`): assert subject_summary has `ripple_full_80_250` + `legacy_bb_1_45` rows, `axis_set=broad`, `used_fixed_mask=True`. **P1-d: the fixed-mask `legacy_bb` vs Task-4 unmasked median is RECORD-ONLY** (write `fixed_mask_delta`); assert equality within `legacy_subject_median_abs` **only for subjects with `n_channels_dropped_by_fixed_mask==0`**.
- [ ] **Step 2: Run fail. Step 3: Implement** (`align_abs_maxab = window_maxab(ctx, zmn_fixed_mask)`; `contact_alignment(zmn, ta_rank, tb_rank, oriented_template)`; aggregate `np.median`).
- [ ] **Step 4: Run pass**, then broad/narrow.
- [ ] **Step 5: Commit** — `git add scripts/run_topic5_v2_alignment.py && git commit -m "feat(topic5-v2): alignment tables (fixed mask, feature-parameterized)"`

---

### Task 8: Spatial constrained-permutation null with fallback + strength

**Files:** Modify `src/topic5_v2_band_scan.py`; Test same.
**Interfaces (Patch/issue #10):** `spatial_constrained_permute(names, values_by_name, shaft_by_name, coord_by_name, rng, mode, min_group)->(perm_values, strength_dict)` where `strength_dict = {spatial_null_strength, n_effectively_permutable, n_singleton_groups}`. Group by shaft if `len(finite group)>=min_group`; else fall back to nearest-distance bin; else subject-wide (flag `subject_wide_weak`).

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v2_band_scan import spatial_constrained_permute
def test_spatial_fallback_reports_strength():
    names=["A1","A2","A3","A4","B1"]                      # B has 1 -> singleton
    vals={n:float(i) for i,n in enumerate(names)}
    shaft={"A1":"A","A2":"A","A3":"A","A4":"A","B1":"B"}
    coord={n:(float(i),0.0) for i,n in enumerate(names)}
    perm,st=spatial_constrained_permute(names,vals,shaft,coord,np.random.default_rng(0),"within_shaft",4)
    assert sorted(perm[n] for n in ["A1","A2","A3","A4"])==[0.0,1.0,2.0,3.0]
    assert st["n_singleton_groups"]>=1 and "spatial_null_strength" in st
```

- [ ] **Step 2: Run fail. Step 3: Implement** (within-shaft permute for groups `>=min_group`; singletons/small groups → distance-bin or subject-wide pooled permute; record strength).
- [ ] **Step 4: Run pass. Step 5: Commit** — `git commit -am "feat(topic5-v2): spatial null with fallback + strength reporting"`

---

### Task 9: HFO-rate-preserving order null PAIR (A/B) + dependency check

**Files:** Modify `src/topic5_v2_band_scan.py`; Create `scripts/run_topic5_v2_order_null_depcheck.py`; Test `tests/test_topic5_v2_band_scan.py` + integration.
**Interfaces (Patch F):**
- `rebuild_typical_rank(events_bool, event_lag, agg='mean')->rank[n_ch]` (participation from `events_bool`; non-participants NaN).
- `order_null_rank_pair(events_a, lag_a, events_b, lag_b, rng)->(rank_a_null, rank_b_null)` — independently rebuild A and B from their own event tables with within-event lag permutation preserving per-channel counts. (If B events unavailable, `rank_b_null = None` and downstream A/B-max uses A only, flagged.)
- dep-check script → `phase1_order_null_depcheck.csv` cols `subject, axis_set, has_event_data_a, has_event_data_b, corr_rebuilt_vs_geo_a, corr_rebuilt_vs_geo_b, order_null_strength(strong|weak_downgrade|missing)`.

- [ ] **Step 1: Failing pure test**

```python
import numpy as np
from src.topic5_v2_band_scan import rebuild_typical_rank, order_null_rank_pair
def test_order_null_pair_preserves_counts_both_templates():
    eb=np.array([[1,1,1,0],[1,1,1,1],[1,0,1,1]],bool); lag=np.array([[0,1,2,np.nan],[0,1,2,3],[0,np.nan,1,2]],float)
    r=rebuild_typical_rank(eb,lag); assert np.nanargmin(r)==0 and np.nanargmax(r)==3
    ra,rb=order_null_rank_pair(eb,lag,eb,lag,np.random.default_rng(0))
    assert (eb.sum(0)>0).tolist()==np.isfinite(ra).tolist()==np.isfinite(rb).tolist()
```

- [ ] **Step 2: Run fail. Step 3: Implement** (`rebuild_typical_rank` as rev1; `order_null_rank_pair` calls the within-event permute+rebuild on A and B separately; handle `events_b is None`).
- [ ] **Step 4: Integration test (relaxed, issue #12):** run dep-check `--substrate broad --outdir tmp`; assert CSV exists, has required cols, and every `order_null_strength ∈ {strong, weak_downgrade, missing}`. **Do NOT assert "≥1 strong" here** (that is an integration-QC print, exit 0 with a warning if none strong).
- [ ] **Step 5: Commit** — `git add src/topic5_v2_band_scan.py scripts/run_topic5_v2_order_null_depcheck.py && git commit -m "feat(topic5-v2): order-null A/B pair + dependency check"`

---

# PHASE 1c — Residual features wired in + Gate A/B/C

### Task 10: Common-field residual (all-band + leave-one-band-out)

**Files:** Modify `src/topic5_v2_band_scan.py`; Test same.
**Interfaces (Patch/issue #14):** `common_field_residual(band_vals_by_name, common_field_vals_by_name)->{name:resid}` (OLS residual of band on the common field). The caller supplies TWO common fields: all-band `1–250` AND leave-one-band-out (`1–250` excluding the target band's bins). Both computed; **Gate B uses leave-one-band-out** (fair), all-band reported as very-conservative.

- [ ] **Step 1: Failing test** (collinear band ⇒ all-band residual ≈ 0; a band-specific bump ⇒ LOBO residual retains it). **Step 2–4:** implement `np.polyfit(cf, band, 1)` residual. **Step 5: Commit** — `git commit -am "feat(topic5-v2): common-field residual (all-band + LOBO)"`

### Task 10b: Common-field residual band cache + alignment (WIRES Gate B)

**Files:** Create `scripts/build_topic5_v2_common_resid_cache.py`; run alignment via Task 7 `--feature common_resid`; Test integration.
**Interfaces:** Build per-window per-contact `common_field_residual_band_z` (LOBO) from the v2 cache (broadband field = sum over `1–250` bands minus target band). Store as a residual cache with the SAME keys as the band cache. Then `run_topic5_v2_alignment.py --feature common_resid` produces `phase1_alignment_common_resid_{window,seizure,subject}.csv`.

- [ ] Steps: build LOBO broadband field per window; residualize each target band; write cache; run alignment; smoke test asserts `common_resid` subject summary exists. Commit `feat(topic5-v2): common-field residual alignment (Gate B input)`.

### Task 11: Aperiodic residual (renamed) — pure

**Files:** Modify `src/topic5_v2_band_scan.py`; Test same.
**Interfaces (Patch/issue #15):** `aperiodic_corrected_excess_power(freqs, psd_ch, lo, hi, line_mask, fit_lo=1, fit_hi=200, min_r2=0.5, half_open=False)->{excess_power, fit_r2, slope, offset, ok}` (log-log guarded fit; name avoids "oscillatory"). Same body as rev1 but renamed key `excess_power`.

- [ ] Steps: rename + test bump-detection. Commit `feat(topic5-v2): aperiodic-corrected excess power (renamed, guarded)`.

### Task 11b: Aperiodic-residual band cache + robust-z + alignment (WIRES Gate C)

**Files:** Create `scripts/build_topic5_v2_aperiodic_cache.py`; alignment via `--feature aperiodic_resid`; Test integration.
**Interfaces:** Per subject/seizure/window/channel, compute the window PSD (reuse `_spectrogram_on_hop` averaged over the field window), `aperiodic_corrected_excess_power` per band → a per-(band,window) excess-power trace → **baseline-robust-z against the same baseline window** → `aperiodic_residual_band_z` cache (same keys). Then `run_topic5_v2_alignment.py --feature aperiodic_resid`. Rows with `ok=False`/`fit_r2<min_r2` → NaN (excluded). This closes Gate C's data path (previously a bare function).

- [ ] Steps: build excess-power cache; robust-z; alignment; smoke test. Commit `feat(topic5-v2): aperiodic-residual cache + alignment (Gate C input)`.

### Task 12: Confound residual rank — pure

**Files:** Modify `src/topic5_v2_band_scan.py`; Test same.
**Interfaces:** `confound_residual_rank(rank_by_name, covariate_maps, overfit_min_ratio=3)->{single:{cov:{name:resid}}, combined:{name:resid}|None}` (combined only if `n_contacts >= overfit_min_ratio*len(covs)+3`). Unchanged from rev1 intent.

- [ ] Steps: test single always / combined guarded. Commit `feat(topic5-v2): confound residual rank (overfit-guarded)`.

### Task 12a: Confound covariate maps builder (issue #13)

**Files:** Create `scripts/build_topic5_v2_confound_maps.py`; Test integration.
**Interfaces:** Produces per subject `phase1_confound_maps.json` with per-contact maps: `hfo_rate` (interictal event count per channel from `load_subject_propagation_events` `events_bool.sum`), `baseline_band_power` (mean baseline log-power per channel from cache baseline window), `broadband_1_250` (baseline broadband power), `shaft_position` (index along shaft from geo `along_axis_mm`), and `soz`/`resection` ONLY if a reliable label file exists (else omitted, logged). These feed Task 12/13.

- [ ] Steps: build maps; smoke test asserts `hfo_rate`,`baseline_band_power`,`shaft_position` present for one subject. Commit `feat(topic5-v2): confound covariate maps`.

---

### Task 13: Null orchestration — TWO-LAYER output (Patch E/H), subject unit

**Files:** Create `scripts/run_topic5_v2_nulls.py`; Test integration.
**Interfaces:**
- `--feature {raw|common_resid|aperiodic_resid}`, `--n-perm`, `--outdir`.
- Produces per axis_set + feature:
  - `phase1_null_perm_subject_long.parquet` — cols `subject, axis_set, feature, null_type(spatial|order), band, perm_id, perm_subject_median` (+ `obs` rows with `perm_id=-1`). **This is the permutation-level table required for max-over-bands.**
  - `phase1_null_subject_summary.csv` — derived per (subject,band,null_type): `obs_subject_median, null_median, null_mad, null_z, delta, empirical_p, spatial_null_strength, order_null_strength`.
  - `phase1_confound_adjusted_subject.csv` (Patch H) — per (subject,band,covariate): `obs_align_to_G_HFO_resid` (deterministic; NOT a null). Optionally, spatial+order null re-run on the adjusted rank → `resid_null_z`/`resid_empirical_p` when `--confound-null`.

**Contract:** each permutation recomputes the FULL statistic. Spatial null: `spatial_constrained_permute` the ictal per-contact values → rebuild field via `make_field_record`+`R_smooth_rank` → `window_maxab` → window→seizure→subject median (one row per perm/band). Order null: replace `ctx["F_inter_a"]/F_inter_b` with fields built from `order_null_rank_pair`-rebuilt A/B ranks → recompute. Uses `analysis_channels` fixed mask. Seed + `n_perm` logged in a sidecar meta JSON. **Precompute per (subject,seizure,window,band) the `vals_by_name` once (issue #16) so perms only re-map, not re-slice.**

- [ ] **Step 1: Failing integration test** (`--feature raw --n-perm 20 --subjects epilepsiae_139 --substrate broad --outdir tmp`): assert `phase1_null_perm_subject_long.parquet` has `perm_id` for `spatial` and `order`, and `phase1_null_subject_summary.csv` has `null_z, empirical_p, order_null_strength`.
- [ ] **Step 2: Run fail. Step 3: Implement.**
- [ ] **Step 4: Run pass** (smoke), then dev (`--n-perm 100`), then final (`--n-perm 1000`) for broad+narrow, raw feature first; residual features after 10b/11b.
- [ ] **Step 5: Commit** — `git add scripts/run_topic5_v2_nulls.py && git commit -m "feat(topic5-v2): two-layer null output (perm-long + summary) + confound adjustment"`

---

### Task 14: Gate A/B/C decision — explicit thresholds + max-over-bands null

**Files:** Modify `src/topic5_v2_band_scan.py` (gate logic); Create `scripts/run_topic5_v2_gates.py`; Test both.
**Interfaces (Patch/issue #17):**
- Pure (P1-b/P1-c: **every variable used in a rule is in the signature**; `gate_A` = spatial AND order; null-strengths gate formal tiers):
  `gate_pass_flags(spatial_p, spatial_delta, spatial_strength, order_p, order_delta, order_strength, common_resid_p, common_resid_delta, aperiodic_p, aperiodic_delta, band_max_over_bands_p, band, fs_subset, alpha) -> {gate_A_spatial, gate_A_order, gate_A, gate_B_freq_specific, gate_C_hfo_specific}` with rules:
  - `gate_A_spatial = spatial_p<alpha and spatial_delta>0 and spatial_strength=='within_shaft_strong'`  ← P1-c: `subject_wide_weak`/`distance_bin_fallback` CANNOT pass a formal Gate A (only descriptive/sensitivity)
  - `gate_A_order   = order_p<alpha and order_delta>0 and order_strength!='weak_downgrade'`
  - `gate_A         = gate_A_spatial and gate_A_order`  ← P1-b: gate_A explicitly requires BOTH
  - `gate_B_freq_specific = gate_A and common_resid_p<alpha and common_resid_delta>0 and band_max_over_bands_p<alpha`
  - `gate_C_hfo_specific  = gate_B_freq_specific and band in {ripple_safe_80_220 (full-cohort), ripple_full_80_250 (fs1024)} and aperiodic_p<alpha and aperiodic_delta>0`
- Pure: `gate_tier(flags, band)->tier` (rev1 tier map; `broadband_recruitment` when A passes but B fails).
- Script: `max_over_bands_p` from `phase1_null_perm_subject_long.parquet` (`P(max_band cohort_perm_delta >= observed_max_band_delta)`), per axis_set + feature. Produces `phase1_gate_summary.csv` cols: `axis_set, cohort, band, feature, gate_A_spatial_pass, gate_A_order_pass, gate_B_frequency_specific_pass, gate_C_HFO_specific_pass, cohort_delta, cohort_null_z, cohort_empirical_p, max_over_bands_p, n_subjects_valid, interpretation_tier`.

- [ ] **Step 1: Failing test** for `gate_pass_flags`: (spatial p=0.01,delta>0,strength=`within_shaft_strong` ⇒ `gate_A_spatial` True); (same but strength=`subject_wide_weak` ⇒ `gate_A_spatial` False — P1-c); (order `weak_downgrade` ⇒ `gate_A_order` False); (`gate_A` True only when both spatial+order pass — P1-b); (`gate_B` needs `common_resid_delta>0` AND `band_max_over_bands_p<alpha`). And `gate_tier` (A only ⇒ broadband_recruitment; A+B+C ripple ⇒ strongest; A fail ⇒ weak_negative). Add `spatial_null_strength` to the gate CSV cols.
- [ ] **Step 2: Run fail. Step 3: Implement** both pure fns.
- [ ] **Step 4: Run pass**, then write the script computing `max_over_bands_p` from the perm-long parquet + joining summary/common_resid/aperiodic tables; run broad+narrow; integration smoke asserts the gate CSV columns + `max_over_bands_p` present.
- [ ] **Step 5: Commit** — `git add src/topic5_v2_band_scan.py scripts/run_topic5_v2_gates.py && git commit -m "feat(topic5-v2): Gate A/B/C thresholds + max-over-bands null"`

---

## Final integration run + QC checklist

- [ ] **Run (both axis sets), features raw → common_resid → aperiodic_resid:**

```bash
for ax in broad narrow; do
  python scripts/run_topic5_v2_legacy_repro.py --substrate $ax            # HARD GATE (exit 0)
  python scripts/build_topic5_v2_band_cache.py --substrate $ax
  python scripts/build_topic5_v2_confound_maps.py --substrate $ax
  python scripts/run_topic5_v2_order_null_depcheck.py --substrate $ax
  python scripts/build_topic5_v2_common_resid_cache.py --substrate $ax
  python scripts/build_topic5_v2_aperiodic_cache.py --substrate $ax
  for feat in raw common_resid aperiodic_resid; do
    python scripts/run_topic5_v2_alignment.py --substrate $ax --feature $feat
    python scripts/run_topic5_v2_nulls.py --substrate $ax --feature $feat --n-perm 1000
  done
  python scripts/run_topic5_v2_gates.py --substrate $ax
done
pytest tests/test_topic5_v2_band_scan.py -v            # pure fns (fast)
pytest -m integration tests/test_topic5_v2_integration.py -v   # real-data smoke
```

- [ ] **Hard QC gates (all must hold before interpreting `phase1_gate_summary.csv`):**
  - Legacy reproduction all `|delta|<=0.02`, per axis_set.
  - `analysis_channels` ⊆ `ctx["mapped"] ∩ cache_channels`; PRIMARY uses the fixed mask; same mask across bands.
  - Every band row carries `band_eff_frac`; `<0.5` flagged + excluded from PRIMARY (kept in sensitivity).
  - 512 Hz subjects `fs_edge_flag=True` on `ripple_full_80_250`/`ripple_high`; ripple primary = `ripple_safe_80_220` full-cohort + `ripple_full_80_250` 1024-subset (both reported, never pooled).
  - Null CSV: `window_maxab` inside the perm loop; subject is the unit; `perm-long` parquet present; seed/n_perm logged.
  - Gate B uses **leave-one-band-out** common field; Gate B/C p include **max-over-bands** null.
  - Confound = deterministic adjustment table (not a `null_type`); confound-null only if `--confound-null`.
  - `order_null_strength=weak_downgrade` subjects excluded from `strongest`.
  - `broad`/`narrow` never pooled; artifact/bad-channel flagged channels excluded.

---

## Self-Review

1. **Patch coverage:** A (window rename)→Task1/3; B (per-axis legacy + repro/analysis split)→Task1/4; C (half-open)→Task2/6; D (residual wired)→Tasks10b/11b/12a; E (perm-long null)→Task13; F (order A/B pair)→Task9; G (fixed-orientation signed)→Task5; H (confound adjustment table)→Task13; I (integration marks + outdir)→all real-data tests. Issues #7 (loader)→Task6a; #8 (fixed mask)→Task6/7; #10 (spatial fallback)→Task8; #12 (relaxed dep test)→Task9; #17 (gate thresholds)→Task14; #19 (artifact QC)→Task3/6. **P1-a (cohort manifest denominator)→Task0; P1-b (gate signature=all rule vars + gate_A=spatial AND order)→Task14; P1-c (spatial_null_strength gates formal tier)→Task14/Global; P1-d (legacy unmasked hard-gate vs fixed-mask record-only)→Task4/7. Covered.**
2. **Placeholder scan:** pure fns full test+impl; wiring tasks (10b/11b/12a) give exact I/O + inputs (no bare functions); no "handle edge cases". **OK.**
3. **Type consistency:** `contact_alignment` keys (Task5) consumed in Tasks7/13; `masked_band_power_trace` dict (Task3) in Task6; `order_null_rank_pair->(rank_a_null,rank_b_null)` (Task9) in Task13; `gate_pass_flags`/`gate_tier` keys (Task14) match gate CSV; `phase1_null_perm_subject_long.parquet` schema (Task13) consumed by Task14 max-over-bands. **Consistent.**

---

## Execution Handoff

1. **Subagent-Driven (recommended)** — fresh subagent per task; **Task 4 legacy reproduction (QC-1 unmasked) is a hard gate (STOP if over tol before Task 6).** Order: **0 (manifest)** → 1→2→3→4 (gate) → 6a→6→7→(5,8,9) → 10,10b,11,11b,12,12a → 13 → 14.
2. **Inline Execution** — checkpoint per phase (1a/1b/1c).

Recommend a clean worktree (current branch `codex/topic4-m3a-v2-2` is entangled with topic5) via `superpowers:using-git-worktrees` before Task 1.
