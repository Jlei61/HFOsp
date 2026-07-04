"""Tests for Topic 5 V2 Phase-1-v2 workstreams.

Task 2.1 (W2 "who?"): continuous per-subject phenotype artifact.
Later phase1-v2 tasks (2.2, 2.3, W1, W3) will extend this file.

TDD contract: unit tests use a fully hand-computable synthetic subject; the
`@pytest.mark.integration` smoke test loads the real Phase-1 CSVs and asserts
the bad-data-regression acceptance gate (Task 2.1 brief, "Acceptance gate").
"""
from __future__ import annotations

from pathlib import Path
import statistics
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import entropy as scipy_entropy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_topic5_v2_subject_phenotype import (  # noqa: E402
    PRIMARY_BANDS,
    LOW_BAND,
    LVFA_BAND,
    HFA_RIPPLE,
    subject_profile,
    load_substrate_frames,
    build_phenotype_table,
    assign_tier,
    assign_band_profile_group,
    spearman_phenotype_gate,
)

# Task 3.1 (W3 "when?"): peri-ictal epoch grid + hard pre-window gate.
import yaml  # noqa: E402
from scripts.run_topic5_v2_alignment import (  # noqa: E402
    _epoch_grid, _epoch_region, _window_admitted, _ictal_fraction_ok, _window_frames,
    _ictal_fraction_eeg)
from scripts.run_topic5_ictal_field_dynamics import _ictal_fraction  # noqa: E402
from src.topic5_v2_band_scan import load_phase1_config  # noqa: E402

# Task 3.2 (W3 "when?"): peri-ictal scaffold-score trajectory + subject-level sign-flip.
from scripts.run_topic5_v2_trajectory import (  # noqa: E402
    BIN_ORDER, CONTRASTS, compute_subject_bins, subject_bin_wide,
    signflip_test, compute_contrasts)

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_PHASE1_CFG = _CONFIG_DIR / "topic5_v2_phase1.yaml"
_PERI_CFG = _CONFIG_DIR / "topic5_v2_phase1_v2_periictal.yaml"

# ---------------------------------------------------------------------------
# Synthetic single-subject fixtures with KNOWN, hand-computable feature values
# ---------------------------------------------------------------------------

# Chosen spatial deltas per primary band (subject "S1", anchor -> clin_onset).
_SYNTH_DELTA = {
    "delta_HYP_slow": 0.30,     # low_band,  positive, sig  (p=0.01)
    "theta_preictal_PAC": 0.20, # low_band,  positive, sig  (p=0.04)
    "alpha_sharp_leq13": 0.10,  # low_band,  positive, NOT sig (p=0.20)
    "beta_LVFA_low": -0.05,     # LVFA_band, NEGATIVE (p=0.01 but delta<0 -> not sig)
    "gamma_LVFA": 0.05,         # LVFA_band, positive, NOT sig (p=0.60)
    "hg_low_ripple": 0.40,      # HFA,       positive, sig  (p=0.02)
    "ripple_high": 0.60,        # HFA,       positive, sig  (p=0.001) -> largest delta
}
_SYNTH_P = {
    "delta_HYP_slow": 0.01,
    "theta_preictal_PAC": 0.04,
    "alpha_sharp_leq13": 0.20,
    "beta_LVFA_low": 0.01,
    "gamma_LVFA": 0.60,
    "hg_low_ripple": 0.02,
    "ripple_high": 0.001,
}
# align-subject align_abs_maxab per primary band -> median over 7 = 0.40
_SYNTH_MAXAB = {
    "delta_HYP_slow": 0.10,
    "theta_preictal_PAC": 0.20,
    "alpha_sharp_leq13": 0.30,
    "beta_LVFA_low": 0.40,
    "gamma_LVFA": 0.50,
    "hg_low_ripple": 0.60,
    "ripple_high": 0.70,
}


def _synth_null_df(subject="S1", deltas=None, ps=None):
    """One (subject x band x null_type) null frame, plus decoy composite/order rows."""
    deltas = deltas if deltas is not None else _SYNTH_DELTA
    ps = ps if ps is not None else _SYNTH_P
    rows = []
    for band in PRIMARY_BANDS:
        # spatial (primary endpoint)
        rows.append(dict(subject=subject, axis_set="narrow", feature="raw", band=band,
                         null_type="spatial", delta=deltas[band], empirical_p=ps[band],
                         spatial_null_strength="within_shaft_strong",
                         order_null_strength="strong"))
        # order null row: strongly "significant" -> MUST be ignored by spatial filter
        rows.append(dict(subject=subject, axis_set="narrow", feature="raw", band=band,
                         null_type="order", delta=0.99, empirical_p=0.0001,
                         spatial_null_strength="within_shaft_strong",
                         order_null_strength="strong"))
    # decoy composite band with huge sig spatial delta -> MUST be excluded from 7-band profile
    rows.append(dict(subject=subject, axis_set="narrow", feature="raw", band="low_HYP_1_13",
                     null_type="spatial", delta=5.0, empirical_p=0.0001,
                     spatial_null_strength="within_shaft_strong",
                     order_null_strength="strong"))
    return pd.DataFrame(rows)


def _synth_align_df(subject="S1", n_seizures=3, n_contacts=12, maxab=None):
    maxab = maxab if maxab is not None else _SYNTH_MAXAB
    rows = []
    for band in PRIMARY_BANDS:
        rows.append(dict(subject=subject, axis_set="narrow", band=band, feature="raw",
                         n_seizures=n_seizures, align_abs_maxab=maxab[band],
                         n_contacts=n_contacts))
    # decoy composite band -> ignored
    rows.append(dict(subject=subject, axis_set="narrow", band="low_HYP_1_13", feature="raw",
                     n_seizures=n_seizures, align_abs_maxab=0.99, n_contacts=n_contacts))
    return pd.DataFrame(rows)


def _synth_seizure_df(subject="S1", per_seizure_center=(0.60, 0.50, 0.40)):
    """Each seizure gets 7 primary bands whose median == the chosen center."""
    rows = []
    for sz, center in enumerate(per_seizure_center):
        # symmetric spread around center -> median over 7 bands == center
        offsets = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        for band, off in zip(PRIMARY_BANDS, offsets):
            rows.append(dict(subject=subject, axis_set="narrow", seizure=sz, band=band,
                             feature="raw", align_abs_maxab=center + off, n_contacts=12.0))
        # decoy composite band per seizure -> ignored in per-seizure median
        rows.append(dict(subject=subject, axis_set="narrow", seizure=sz, band="low_HYP_1_13",
                         feature="raw", align_abs_maxab=0.0, n_contacts=12.0))
    return pd.DataFrame(rows)


def test_subject_profile_features():
    """Every derived feature matches hand-computed values for the synthetic subject."""
    prof = subject_profile(_synth_null_df(), _synth_align_df(), _synth_seizure_df(),
                           PRIMARY_BANDS)

    # identity / provenance
    assert prof["subject"] == "S1"
    assert prof["anchor"] == "clin_onset"
    assert prof["n_sz"] == 3
    assert prof["n_contacts"] == 12
    assert prof["spatial_strength"] == "within_shaft_strong"
    assert prof["order_strength"] == "strong"

    # 7-band spatial delta summaries
    # deltas = [0.30, 0.20, 0.10, -0.05, 0.05, 0.40, 0.60], sum = 1.60
    assert prof["mean_delta_7bands"] == pytest.approx(1.60 / 7)
    assert prof["median_delta_7bands"] == pytest.approx(0.20)
    assert prof["n_positive_delta_7bands"] == 6      # all but beta_LVFA_low(-0.05)
    assert prof["n_sig_7bands"] == 4                 # delta>0 & p<0.05: HYP, theta, hg, ripple
    assert prof["band_genericity_index"] == pytest.approx(6 / 7)

    # band-group medians
    assert prof["low_band_score"] == pytest.approx(0.20)   # median{0.30,0.20,0.10}
    assert prof["LVFA_band_score"] == pytest.approx(0.0)   # median{-0.05,0.05}
    assert prof["HFA_ripple_score"] == pytest.approx(0.50) # median{0.40,0.60}
    assert prof["HF_minus_low"] == pytest.approx(0.30)     # 0.50 - 0.20

    # ripple_high (0.60) is the single largest -> descending rank 1
    assert prof["ripple_rank"] == 1

    # profile_entropy: Shannon entropy of normalized positive part / ln(7)
    pos = np.array([max(_SYNTH_DELTA[b], 0.0) for b in PRIMARY_BANDS])
    expected_entropy = float(scipy_entropy(pos) / np.log(7))  # scipy: independent oracle
    assert prof["profile_entropy"] == pytest.approx(expected_entropy)

    # maxab_primary: median align_abs_maxab over 7 primary (align-subject df) = 0.40
    assert prof["maxab_primary"] == pytest.approx(0.40)

    # within_subject_seizure_consistency:
    # per-seizure medians = [0.60, 0.50, 0.40]; median M=0.50,
    # IQR (linear) = 0.55 - 0.45 = 0.10; consistency = 1 - 0.10/0.50 = 0.80
    assert prof["within_subject_seizure_consistency"] == pytest.approx(0.80)


def test_anchor_rule_eeg_onset_subjects():
    """xuxinyi and zhangkexuan use eeg_onset; everyone else clin_onset."""
    for subj in ("xuxinyi", "zhangkexuan"):
        prof = subject_profile(_synth_null_df(subject=subj), _synth_align_df(subject=subj),
                               _synth_seizure_df(subject=subj), PRIMARY_BANDS)
        assert prof["anchor"] == "eeg_onset"
    prof = subject_profile(_synth_null_df(subject="1077"), _synth_align_df(subject="1077"),
                           _synth_seizure_df(subject="1077"), PRIMARY_BANDS)
    assert prof["anchor"] == "clin_onset"


def test_profile_entropy_all_nonpositive_is_zero():
    """If every primary spatial delta <= 0, profile_entropy is defined as 0.0."""
    neg = {b: -0.1 for b in PRIMARY_BANDS}
    ps = {b: 0.9 for b in PRIMARY_BANDS}
    prof = subject_profile(_synth_null_df(deltas=neg, ps=ps), _synth_align_df(),
                           _synth_seizure_df(), PRIMARY_BANDS)
    assert prof["profile_entropy"] == 0.0
    assert prof["n_positive_delta_7bands"] == 0
    assert prof["n_sig_7bands"] == 0


def test_seizure_consistency_single_seizure_is_nan():
    """n_seizures < 2 -> consistency is NaN (cannot form an IQR)."""
    prof = subject_profile(_synth_null_df(), _synth_align_df(n_seizures=1),
                           _synth_seizure_df(per_seizure_center=(0.55,)), PRIMARY_BANDS)
    assert np.isnan(prof["within_subject_seizure_consistency"])


def test_seizure_consistency_zero_median_is_nan():
    """|median| ~ 0 across seizures -> consistency is NaN (division guard)."""
    prof = subject_profile(_synth_null_df(), _synth_align_df(n_seizures=3),
                           _synth_seizure_df(per_seizure_center=(0.0, 0.0, 0.0)),
                           PRIMARY_BANDS)
    assert np.isnan(prof["within_subject_seizure_consistency"])


def test_n_sz_counts_distinct_seizures_not_align_n_seizures():
    """n_sz = distinct seizures in the per-seizure frame (robust to HFA bands
    dropping seizures), NOT align_df's per-band n_seizures."""
    rows = []
    # seizures 0,1: all 7 primary bands; seizures 2,3,4: only low bands (HFA dropped)
    for sz in (0, 1):
        for band in PRIMARY_BANDS:
            rows.append(dict(subject="S1", axis_set="narrow", seizure=sz, band=band,
                             feature="raw", align_abs_maxab=0.5, n_contacts=12.0))
    for sz in (2, 3, 4):
        for band in LOW_BAND:
            rows.append(dict(subject="S1", axis_set="narrow", seizure=sz, band=band,
                             feature="raw", align_abs_maxab=0.5, n_contacts=12.0))
    seizure_df = pd.DataFrame(rows)
    # align_df deliberately reports n_seizures=2 to prove n_sz ignores it.
    prof = subject_profile(_synth_null_df(), _synth_align_df(n_seizures=2), seizure_df,
                           PRIMARY_BANDS)
    assert prof["n_sz"] == 5  # distinct {0,1,2,3,4}, not align n_seizures=2


def test_non_raw_rows_are_excluded_from_align_and_seizure():
    """feature != 'raw' rows must not leak into align/seizure aggregates."""
    align = _synth_align_df()
    decoy_a = align.iloc[[0]].copy()
    decoy_a["feature"] = "adjusted"
    decoy_a["align_abs_maxab"] = 99.0  # would blow up maxab_primary if not filtered
    align = pd.concat([align, decoy_a], ignore_index=True)

    seiz = _synth_seizure_df()
    decoy_s = seiz.iloc[[0]].copy()
    decoy_s["feature"] = "adjusted"
    decoy_s["seizure"] = 999          # would inflate n_sz to 4 if not filtered
    seiz = pd.concat([seiz, decoy_s], ignore_index=True)

    prof = subject_profile(_synth_null_df(), align, seiz, PRIMARY_BANDS)
    assert prof["maxab_primary"] == pytest.approx(0.40)
    assert prof["n_sz"] == 3


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_empty_align_returns_nan_maxab_without_warning():
    """Zero align rows -> maxab_primary is NaN with no RuntimeWarning."""
    empty_align = _synth_align_df().iloc[0:0]  # same columns, zero rows
    prof = subject_profile(_synth_null_df(), empty_align, _synth_seizure_df(),
                           PRIMARY_BANDS)
    assert np.isnan(prof["maxab_primary"])


# ---------------------------------------------------------------------------
# Real-data smoke test: bad-data-regression acceptance gate (brief "Acceptance gate")
# ---------------------------------------------------------------------------

_V2_ROOT = Path(__file__).resolve().parents[1] / "results" / "topic5_ictal_recruitment" / "v2_band_scan"


@pytest.mark.integration
def test_acceptance_gate_reproduces_phase1_baseline():
    """narrow n_sig median==2 & count>=4==6 (of 20); broad median==3 & count>=4==7 (of 17);
    {1146,1150,384} in both-pool >=4 set. Baseline is LOCKED; code is wrong if this fails."""
    if not (_V2_ROOT / "narrow" / "phase1_null_raw_subject_summary.csv").exists():
        pytest.skip("real Phase-1 v2_band_scan artifacts not present")

    tables = {}
    for substrate in ("narrow", "broad"):
        null_df, align_df, seizure_df = load_substrate_frames(_V2_ROOT / substrate)
        tables[substrate] = build_phenotype_table(null_df, align_df, seizure_df,
                                                  substrate, PRIMARY_BANDS)

    narrow = tables["narrow"]
    broad = tables["broad"]
    assert len(narrow) == 20
    assert len(broad) == 17

    # Median convention pinned in ALL THREE forms so no downstream task (2.2/2.3) can
    # read a phantom regression under either convention: narrow is even-n (20) ->
    # np.median 2.5 while statistics.median_low 2 (the Phase-1 archive "2"); broad is
    # odd-n (17) -> both 3. count(n_sig>=4) is the convention-free, load-bearing stat.
    n_nar = narrow["n_sig_7bands"].astype(int).tolist()
    n_brd = broad["n_sig_7bands"].astype(int).tolist()
    assert statistics.median_low(n_nar) == 2
    assert np.median(n_nar) == pytest.approx(2.5)
    assert sum(v >= 4 for v in n_nar) == 6
    assert statistics.median_low(n_brd) == 3
    assert np.median(n_brd) == pytest.approx(3.0)
    assert sum(v >= 4 for v in n_brd) == 7

    ge4_nar = set(narrow.loc[narrow["n_sig_7bands"] >= 4, "subject"].astype(str))
    ge4_brd = set(broad.loc[broad["n_sig_7bands"] >= 4, "subject"].astype(str))
    assert len(ge4_nar) == 6
    assert len(ge4_brd) == 7
    assert (ge4_nar & ge4_brd) == {"1146", "1150", "384"}


def test_band_group_membership_is_disjoint_partition():
    """low/LVFA/HFA groups partition the 7 primary bands exactly."""
    assert LOW_BAND + LVFA_BAND + HFA_RIPPLE  # non-empty
    assert set(LOW_BAND) | set(LVFA_BAND) | set(HFA_RIPPLE) == set(PRIMARY_BANDS)
    assert len(LOW_BAND) + len(LVFA_BAND) + len(HFA_RIPPLE) == len(PRIMARY_BANDS) == 7


# ---------------------------------------------------------------------------
# Task 2.2: three-tier subject label (thresholds + precedence LOCKED, brief §"The task")
# ---------------------------------------------------------------------------

def test_three_tier_labels():
    """Boundary values for the three tiers; exact labels, strong precedence over directional."""
    assert assign_tier(n_sig_7bands=4, n_positive_delta_7bands=4) == "strong"
    assert assign_tier(n_sig_7bands=3, n_positive_delta_7bands=6) == "directional"
    assert assign_tier(n_sig_7bands=3, n_positive_delta_7bands=4) == "weak_absent"
    assert assign_tier(n_sig_7bands=5, n_positive_delta_7bands=5) == "strong"  # strong wins
    assert assign_tier(n_sig_7bands=0, n_positive_delta_7bands=4) == "weak_absent"


# ---------------------------------------------------------------------------
# Task 2.3: descriptive band-gradient buckets + Spearman phenotype-hunt gate
# (both DESCRIPTIVE; cutoffs are bucketing labels, the gate is a report filter --
#  NOT a KMeans/statistical subtype claim, NOT a formal test threshold; brief §Step1/2)
# ---------------------------------------------------------------------------

def test_band_profile_group_buckets():
    """The 4 descriptive buckets + weak_absent precedence + the ±0.05 / 0.6 cutoffs.

    Cutoffs are DESCRIPTIVE bucketing labels (not thresholds of any test).
    weak_absent tier is assigned FIRST regardless of the signed gradient.
    """
    # weak_absent precedence: overrides both an HFA-leaning and a low-leaning gradient
    assert assign_band_profile_group(hf_minus_low=+0.20, band_genericity_index=1.0,
                                     tier="weak_absent") == "weak_absent"
    assert assign_band_profile_group(hf_minus_low=-0.20, band_genericity_index=0.0,
                                     tier="weak_absent") == "weak_absent"

    # low_leaning : HF_minus_low < -0.05  (non-weak_absent tier)
    assert assign_band_profile_group(hf_minus_low=-0.10, band_genericity_index=0.71,
                                     tier="directional") == "low_leaning"
    # hf_leaning : HF_minus_low > +0.05
    assert assign_band_profile_group(hf_minus_low=+0.10, band_genericity_index=0.57,
                                     tier="strong") == "hf_leaning"
    # flat_generic : |HF_minus_low| <= 0.05 AND band_genericity_index >= 0.6
    assert assign_band_profile_group(hf_minus_low=0.0, band_genericity_index=0.80,
                                     tier="strong") == "flat_generic"

    # boundary: exactly -0.05 is NOT low_leaning (strict <); with high genericity -> flat_generic
    assert assign_band_profile_group(hf_minus_low=-0.05, band_genericity_index=0.86,
                                     tier="directional") == "flat_generic"
    # boundary: exactly +0.05 is NOT hf_leaning (strict >); with high genericity -> flat_generic
    assert assign_band_profile_group(hf_minus_low=+0.05, band_genericity_index=0.86,
                                     tier="strong") == "flat_generic"


def test_spearman_phenotype_gate():
    """LOCKED phenotype gate: flag "predicts" ONLY if |r| > 0.4 AND p < 0.05.

    Returns (r, p, n, passes). A strong monotonic feature passes; a near-zero-r
    feature does NOT; a high-|r| but non-significant (small-n) feature does NOT
    (the AND is load-bearing); a constant feature yields NaN and does NOT pass.
    """
    target = [1, 2, 3, 4, 5, 6]

    # strong monotonic predictor: r == 1.0, p ~ 0  -> passes
    r, p, n, passes = spearman_phenotype_gate([10, 20, 30, 40, 50, 60], target)
    assert n == 6
    assert r == pytest.approx(1.0)
    assert p < 0.05
    assert passes is True

    # near-zero-r predictor: |r| ~ 0.26 < 0.4  -> does NOT pass (this is the expected case)
    r, p, n, passes = spearman_phenotype_gate([1, 6, 2, 5, 3, 4], target)
    assert abs(r) < 0.4
    assert passes is False

    # high |r| (0.8) but p >= 0.05 at n=4: AND semantics -> does NOT pass
    r, p, n, passes = spearman_phenotype_gate([1, 2, 4, 3], [1, 2, 3, 4])
    assert abs(r) > 0.4
    assert p >= 0.05
    assert passes is False

    # constant predictor -> undefined correlation (NaN) -> does NOT pass, no crash
    r, p, n, passes = spearman_phenotype_gate([5, 5, 5, 5], [1, 2, 3, 4])
    assert np.isnan(r)
    assert passes is False

    # NaN pairs are dropped before correlating (n counts finite pairs only)
    r, p, n, passes = spearman_phenotype_gate([10, 20, 30, 40, 50, np.nan],
                                              [1, 2, 3, 4, 5, 6])
    assert n == 5
    assert passes is True


# ---------------------------------------------------------------------------
# Task 3.1 (W3 "when?"): peri-ictal epoch grid (-100->+20s) + hard pre-window gate.
# The peri-ictal admission + ictal-fraction EXEMPTION + region partition are OPT-IN
# (epoch.admit_pre_ictal_windows); the DEFAULT [0,20] window set must stay unchanged (C1).
# ---------------------------------------------------------------------------

def test_periictal_grid_pre_windows_survive():
    """Peri-ictal config (main_rel=[-100,20], window=10, step=5): pre-ictal windows (EEG-frame
    win_end_rel_eeg<=0) are ADMITTED and SURVIVE the ictal-fraction floor (exempt, icf~0 by
    construction, legitimately pre-ictal); the grid reaches EEG -100; deep pre-ictal windows carry a
    pre-region label. Synthetic seizure: eeg_onset_rel=0 (so EEG frame == relt frame), relt covers
    [-100, 20]. Routes through the production _window_frames so it exercises the real frame mapping."""
    cfg = load_phase1_config(_PERI_CFG)
    assert cfg["epoch"]["admit_pre_ictal_windows"] is True
    grid, r1 = _epoch_grid(cfg)
    r0 = float(cfg["epoch"]["main_rel"][0])
    ictal_min = float(cfg["epoch"]["ictal_fraction_min"])
    admit_pre = True

    relt = np.round(np.arange(-100.0, 20.0 + 1e-9, 0.1), 3)   # covers [-100, 20]
    off, on = 15.0, 0.0                                       # ictal = [0, 15]; EEG onset at relt=0

    surviving = []
    for st, en in grid:                                       # st,en are EEG-frame bounds (peri)
        if not _window_admitted(st, en, r0, r1, admit_pre):
            continue
        st_relt, en_relt, wcr, wcr_eeg = _window_frames(st, en, on, admit_pre)
        if ((relt >= st_relt) & (relt <= en_relt)).sum() == 0:   # recording-boundary drop (zmv is None)
            continue
        icf = _ictal_fraction(relt, st_relt, en_relt, off)
        if not _ictal_fraction_ok(icf, ictal_min, en, admit_pre):   # en = EEG-frame end (grid=EEG)
            continue
        surviving.append(dict(win_center_rel_eeg=wcr_eeg, win_end_rel_eeg=en,
                              ictal_fraction=icf, epoch_region=_epoch_region(wcr_eeg)))

    centers = [w["win_center_rel_eeg"] for w in surviving]
    assert min(centers) <= -90                               # grid reaches EEG -100 (first center -95)

    pre = [w for w in surviving if w["win_end_rel_eeg"] <= 0]   # windows entirely before EEG onset
    assert len(pre) > 0                                      # pre-windows survive (not all filtered)
    assert all(w["ictal_fraction"] < ictal_min for w in pre) # they are icf-exempt (icf~0), NOT filtered

    deep_pre = [w for w in surviving if w["win_center_rel_eeg"] <= -15]   # EEG center well before onset
    assert len(deep_pre) > 0
    assert all(w["epoch_region"] in {"far_pre", "mid_pre", "near_pre"} for w in deep_pre)


def test_periictal_grid_eeg_frame_nonzero_onset():
    """The peri-ictal grid is the EEG-ONSET frame (not clinical/relt). With eeg_onset_rel != 0:
    win_center_rel_eeg tiles a CLEAN [-100,20] (EEG frame) so epoch_region always bins to one of the
    5 named regions (no out-of-partition); win_center_rel == win_center_rel_eeg + eeg_onset_rel (relt
    frame); and the relt slice bounds shift by eeg_onset_rel (so the cache is read at the physically
    correct time). This is the fix for large EEG-vs-clinical gaps (down to -86s in cohort)."""
    cfg = load_phase1_config(_PERI_CFG)
    grid, r1 = _epoch_grid(cfg)
    admit_pre = True
    on = -40.0                                               # EEG onset 40s BEFORE relt=0 (clinical)
    for st, en in grid:                                      # EEG-frame bounds
        st_relt, en_relt, wcr, wcr_eeg = _window_frames(st, en, on, admit_pre)
        assert -100.0 <= wcr_eeg <= 20.0                    # EEG-frame center is clean, always in-range
        assert wcr_eeg == pytest.approx((st + en) / 2.0)    # == grid center (grid IS the EEG frame)
        assert wcr == pytest.approx(wcr_eeg + on)           # relt center = EEG center + eeg_onset_rel
        assert st_relt == pytest.approx(st + on)            # relt slice shifted by eeg_onset_rel
        assert en_relt == pytest.approx(en + on)
        assert _epoch_region(wcr_eeg) in {"far_pre", "mid_pre", "near_pre", "peri_onset", "early_post"}

    # DEFAULT mode: grid IS the relt frame; EEG center = relt center - eeg_onset_rel (may leave range)
    st, en = grid[0]
    sr, er, wcr_d, wcr_eeg_d = _window_frames(st, en, on, admit_pre=False)
    assert (sr, er) == pytest.approx((st, en))              # relt slice == grid (no shift)
    assert wcr_d == pytest.approx((st + en) / 2.0)
    assert wcr_eeg_d == pytest.approx(wcr_d - on)


def test_eeg_anchor_icf_recovers_post_onset_windows():
    """EEG-anchored icf fix (Task 3.2 Step 0). Large EEG-clinical gap (eeg_onset_rel=-50): the EEG
    post-onset windows (win_center_rel_eeg in (0,20] -- inside the EEG seizure, yet their relt center is
    still < 0 = before CLINICAL onset) SURVIVE the ictal-fraction floor under the EEG anchor (ictal =
    [eeg_onset_rel, eeg_offset_rel]); under the OLD clinical-anchored icf (ictal = [0, eeg_offset_rel])
    those SAME windows have icf=0 and are FILTERED (the load-bearing Task-3.1 bug). Their EEG-frame end
    en>0, so they are NOT the admit_pre pre-window exemption -- the ONLY reason they survive is the
    EEG-anchored icf floor. Routes through the production frame/icf helpers."""
    cfg = load_phase1_config(_PERI_CFG)
    grid, r1 = _epoch_grid(cfg)
    r0 = float(cfg["epoch"]["main_rel"][0])
    ictal_min = float(cfg["epoch"]["ictal_fraction_min"])
    admit_pre = True
    on, off = -50.0, 30.0                                     # EEG onset 50s before clinical; EEG offset relt=30
    relt = np.round(np.arange(-100.0, 40.0 + 1e-9, 0.1), 3)   # spans the whole large-gap seizure

    recovered = []               # EEG post-onset windows the EEG anchor KEEPS but the clinical icf DROPS
    for st, en in grid:
        if not _window_admitted(st, en, r0, r1, admit_pre):
            continue
        st_relt, en_relt, wcr, wcr_eeg = _window_frames(st, en, on, admit_pre, anchor="eeg")
        if not (0.0 < wcr_eeg <= 20.0):                       # only EEG post-onset windows
            continue
        if ((relt >= st_relt) & (relt <= en_relt)).sum() == 0:   # recording-boundary drop
            continue
        icf_eeg = _ictal_fraction_eeg(relt, st_relt, en_relt, on, off)   # EEG-anchored ictal [on, off] (fixed)
        icf_clin = _ictal_fraction(relt, st_relt, en_relt, off)          # clinical-anchored ictal [0, off] (bug)
        keeps_eeg = _ictal_fraction_ok(icf_eeg, ictal_min, en, admit_pre)
        keeps_clin = _ictal_fraction_ok(icf_clin, ictal_min, en, admit_pre)
        assert en > 0                                         # EEG-frame end > 0 -> NOT pre-window exempt
        if keeps_eeg and not keeps_clin:
            recovered.append(dict(wcr_eeg=wcr_eeg, icf_eeg=icf_eeg, icf_clin=icf_clin,
                                  region=_epoch_region(wcr_eeg)))

    assert len(recovered) > 0                                 # the fix recovers >=1 post-EEG-onset window
    assert all(w["icf_eeg"] >= ictal_min for w in recovered) # genuinely ictal under the EEG anchor
    assert all(w["icf_clin"] == 0.0 for w in recovered)      # icf=0 under the clinical anchor -> filtered
    assert all(w["region"] in {"peri_onset", "early_post"} for w in recovered)


def test_default_config_unchanged_no_pre_windows():
    """DEFAULT config (main_rel=[0,20], admit_pre_ictal_windows absent): backward-compat (C1). The
    [0,20] admitted-window SET is exactly the locked Phase-1 set and NO admitted window is a pre-ictal
    window (win_end_rel<=0) -- the peri-ictal exemption must NOT leak into default mode."""
    cfg = load_phase1_config()                               # canonical phase1.yaml
    assert cfg["epoch"].get("admit_pre_ictal_windows", False) is False
    grid, r1 = _epoch_grid(cfg)
    r0 = float(cfg["epoch"]["main_rel"][0])
    admit_pre = False
    admitted = [(st, en) for st, en in grid if _window_admitted(st, en, r0, r1, admit_pre)]
    assert admitted
    assert all(en > 0 for st, en in admitted)                # no pre-window (win_end_rel<=0) admitted
    assert {(round(st, 3), round(en, 3)) for st, en in admitted} == {
        (-5.0, 5.0), (0.0, 10.0), (5.0, 15.0), (10.0, 20.0), (15.0, 25.0)}


def test_periictal_config_matches_phase1_nonepoch():
    """Drift guard: the peri-ictal yaml is identical to phase1.yaml in EVERY section except `epoch`."""
    base = yaml.safe_load(_PHASE1_CFG.read_text())
    peri = yaml.safe_load(_PERI_CFG.read_text())
    assert base.pop("epoch") != peri.pop("epoch")            # epoch DID change
    assert base == peri                                      # everything else byte-identical


# ---------------------------------------------------------------------------
# Task 3.2 (W3 "when?"): peri-ictal scaffold-score trajectory + subject-level SIGN-FLIP.
# Aggregation LOCK (window->seizure->subject->cohort): per-window scaffold = median align_abs_maxab
# over the 7 PRIMARY bands; per (subject,seizure,bin) = median over that bin's windows; per
# (subject,bin) = median over that subject's seizures. Contrasts are per-subject PAIRED diffs; a
# subject missing either bin is EXCLUDED from THAT contrast. Main test = subject-level sign-flip.
# ---------------------------------------------------------------------------

def _synth_traj_long():
    """Peri-ictal window-long frame with hand-set per-(subject,bin) scaffold scores.

    Per (subject, bin): 3 seizures with per-seizure targets base+{-0.02, 0, +0.10} -> the MEDIAN
    over seizures is the middle value = base (proves seizure-level median, robust to the +0.10
    outlier; a mean would give base+0.027). Per (seizure, bin): 2 windows, each window's 7 primary
    bands are symmetric around the seizure target (band offsets sum to 0 -> median over bands ==
    seizure target; two identical windows -> median over windows == seizure target). Decoys per
    window: a composite (non-primary) band with a wild value, and a feature!='raw' row -- both MUST
    be excluded. S4 has NO far_pre windows (paired-exclusion for the two far_pre contrasts)."""
    band_off = [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06]     # sum 0 -> median==target
    sz_off = [-0.02, 0.0, 0.10]                                 # median over 3 == middle == +0.0
    bases = {
        "S1": {"far_pre": 0.30, "mid_pre": 0.32, "near_pre": 0.42, "peri_onset": 0.50, "early_post": 0.70},
        "S2": {"far_pre": 0.32, "mid_pre": 0.34, "near_pre": 0.40, "peri_onset": 0.52, "early_post": 0.68},
        "S3": {"far_pre": 0.28, "mid_pre": 0.30, "near_pre": 0.44, "peri_onset": 0.48, "early_post": 0.72},
        "S4": {"mid_pre": 0.31, "near_pre": 0.41, "peri_onset": 0.49, "early_post": 0.69},  # NO far_pre
    }
    rows = []
    for subj, binmap in bases.items():
        for bi, (binname, base) in enumerate(binmap.items()):
            for sz, so in enumerate(sz_off):
                seiz_target = base + so
                for wi in range(2):                            # 2 windows per seizure per bin
                    ws = 100.0 * bi + wi                       # unique bounds within (subject,seizure)
                    for band in PRIMARY_BANDS:
                        rows.append(dict(subject=subj, seizure=sz, band=band, feature="raw",
                                         win_start_rel=ws, win_end_rel=ws + 10.0,
                                         epoch_region=binname,
                                         align_abs_maxab=seiz_target + band_off[PRIMARY_BANDS.index(band)]))
                    # decoy composite band (non-primary) -> excluded from the 7-band median
                    rows.append(dict(subject=subj, seizure=sz, band="low_HYP_1_13", feature="raw",
                                     win_start_rel=ws, win_end_rel=ws + 10.0, epoch_region=binname,
                                     align_abs_maxab=0.99))
                    # decoy non-raw feature -> excluded
                    rows.append(dict(subject=subj, seizure=sz, band="delta_HYP_slow", feature="adjusted",
                                     win_start_rel=ws, win_end_rel=ws + 10.0, epoch_region=binname,
                                     align_abs_maxab=99.0))
    return pd.DataFrame(rows)


def test_trajectory_contrasts():
    """window->seizure->subject aggregation + the 3 paired contrasts + directions + paired exclusion.

    subject_bin[bin] recovers the injected base (median over seizures / windows / 7 primary bands,
    ignoring the composite + non-raw decoys). Contrasts are per-subject paired diffs; S4 (no far_pre)
    drops out of the two far_pre contrasts (n=3) but stays in post_minus_near_pre (n=4). All three
    cohort median diffs are POSITIVE with the hand-computed values."""
    df = _synth_traj_long()
    subj = compute_subject_bins(df, PRIMARY_BANDS)
    wide = subject_bin_wide(subj)

    # subject_bin recovers the injected base (decoys ignored)
    assert wide.loc["S1", "far_pre"] == pytest.approx(0.30)
    assert wide.loc["S1", "early_post"] == pytest.approx(0.70)
    assert wide.loc["S3", "near_pre"] == pytest.approx(0.44)
    assert wide.loc["S2", "peri_onset"] == pytest.approx(0.52)
    assert np.isnan(wide.loc["S4", "far_pre"])                  # S4 has no far_pre

    # n_seizures per (subject, bin) counts the 3 distinct seizures
    s1_far = subj[(subj.subject == "S1") & (subj.epoch_region == "far_pre")].iloc[0]
    assert int(s1_far.n_seizures) == 3

    con = compute_contrasts(wide)

    def _row(name):
        return con[con.contrast == name].iloc[0]

    c_nf = _row("near_pre_minus_far_pre")            # diffs S1=.12 S2=.08 S3=.16 -> median .12
    assert int(c_nf.n_subjects) == 3                 # S4 excluded (no far_pre)
    assert c_nf.cohort_median_diff == pytest.approx(0.12)

    c_pf = _row("post_minus_far_pre")                # diffs S1=.40 S2=.36 S3=.44 -> median .40
    assert int(c_pf.n_subjects) == 3                 # S4 excluded (no far_pre)
    assert c_pf.cohort_median_diff == pytest.approx(0.40)

    c_pn = _row("post_minus_near_pre")               # diffs all = .28; S4 = .69-.41 = .28 -> median .28
    assert int(c_pn.n_subjects) == 4                 # S4 INCLUDED (has both near_pre & early_post)
    assert c_pn.cohort_median_diff == pytest.approx(0.28)

    assert (con["cohort_median_diff"] > 0).all()     # every contrast is a rise (pre < post)
    assert set(con["contrast"]) == {n for n, _, _ in CONTRASTS}


def test_subject_signflip():
    """Subject-level sign-flip permutation (the MAIN test). Statistic = median(d_i); null = symmetric
    about 0; two-sided. Exact enumeration of 2^n for n<=14, random n_perm otherwise. All-positive
    diffs -> small p; symmetric-about-0 diffs -> p==1; a single subject cannot reject (p==1); the
    random path is deterministic under a fixed seed."""
    # all-positive, n=10 -> exact enumeration. obs median = 0.55; 32/1024 perms reach |median|>=0.55.
    res = signflip_test(np.array([0.1 * i for i in range(1, 11)]))
    assert res["n"] == 10
    assert res["method"] == "exact"
    assert res["median_diff"] == pytest.approx(0.55)
    assert res["p_signflip"] == pytest.approx(32 / 1024)
    assert res["p_signflip"] < 0.05

    # symmetric about 0, n=6 -> exact. obs median = 0 -> every sign-flip has |median|>=0 -> p==1.
    res = signflip_test(np.array([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]))
    assert res["method"] == "exact"
    assert res["median_diff"] == pytest.approx(0.0)
    assert res["p_signflip"] == pytest.approx(1.0)

    # a single subject cannot reject (both sign flips give |median|==|obs|) -> p==1.
    res = signflip_test(np.array([0.5]))
    assert res["n"] == 1
    assert res["p_signflip"] == pytest.approx(1.0)

    # random path (n=20 > 14), strongly positive -> tiny p; identical across calls (deterministic).
    d = np.arange(1, 21) / 100.0
    r1 = signflip_test(d, n_perm=20000, seed=0)
    r2 = signflip_test(d, n_perm=20000, seed=0)
    assert r1["method"] == "random"
    assert r1["p_signflip"] == r2["p_signflip"]          # deterministic under fixed seed
    assert r1["p_signflip"] < 0.001
