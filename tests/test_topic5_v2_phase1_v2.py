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
)

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
