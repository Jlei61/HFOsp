"""Tests for src/topic1_topic5_bridge.py."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import topic1_topic5_bridge as bridge


def test_locked_constants():
    """Sanity check: locked constants match spec §4."""
    assert bridge.ALPHA_WITHIN == pytest.approx(0.0167)
    assert bridge.EFFECT_MIN == pytest.approx(0.10)
    assert bridge.WINDOWS_MIN == [(-15.0, -1.0), (-30.0, -1.0), (-60.0, -1.0)]
    assert len(bridge.COHORT_GAMMA) == 10
    assert "442" not in bridge.COHORT_GAMMA  # 442 is Q1b sentinel only


def test_load_topic5_subtype_labels_442():
    """442 gamma_ER: 17 seizures, 16 with subtype 0 + 1 outlier (-1)."""
    out = bridge.load_topic5_subtype_labels(
        subject="442",
        band="gamma_ER",
        results_root=Path("/home/honglab/leijiaxin/HFOsp/results"),
    )
    assert isinstance(out["seizure_id_to_subtype"], dict)
    assert len(out["seizure_id_to_subtype"]) == 17
    n_outliers = sum(1 for v in out["seizure_id_to_subtype"].values() if v == -1)
    assert n_outliers == 1
    assert out["n_subtypes"] == 1
    assert out["status"] == "ok"


def test_load_topic5_subtype_labels_548():
    """548 gamma_ER has 5 subtypes per audit-rerun cohort."""
    out = bridge.load_topic5_subtype_labels(
        subject="548",
        band="gamma_ER",
        results_root=Path("/home/honglab/leijiaxin/HFOsp/results"),
    )
    assert out["n_subtypes"] == 5
    assert len(out["seizure_id_to_subtype"]) == 19


def test_load_seizure_onsets_442():
    """442 has 17 ok seizures; clin_onset preferred, fallback eeg_onset."""
    out = bridge.load_seizure_onsets(
        subject="442",
        results_root=Path("/home/honglab/leijiaxin/HFOsp/results"),
    )
    assert isinstance(out, dict)
    # 442 has 17 seizures kept by topic5; the inventory has all of them
    assert len(out) >= 17
    # clin_onset_epoch is float seconds since epoch
    sample_id, sample_t = next(iter(out.items()))
    assert isinstance(sample_t, float)
    assert sample_t > 1e9, "epoch seconds, not relative"
    # No NaN values returned (NaN sources should be flagged via missing key)
    assert all(np.isfinite(v) for v in out.values())


def test_load_topic1_events_with_templates_442_alignment():
    """442 has n_events_total=6556, all valid (n_valid_events=6556).
    event_abs_times must be epoch seconds; labels must align 1-to-1.
    """
    out = bridge.load_topic1_events_with_templates(
        subject="442",
        results_root=Path("/home/honglab/leijiaxin/HFOsp/results"),
        artifact_root=Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns"),
    )
    assert out["n_valid_events"] == 6556
    assert out["event_abs_times"].shape == (6556,)
    assert out["template_labels"].shape == (6556,)
    # epoch seconds, not relative
    assert np.nanmin(out["event_abs_times"]) > 1e9
    # Label values match cluster_id set {0, 1} from spec
    assert set(np.unique(out["template_labels"]).tolist()) == {0, 1}
    # T0/T1 fractions match topic1 JSON
    n_t0 = int((out["template_labels"] == out["t0_template_id"]).sum())
    expected_t0_fraction = 0.5167785234899329  # cluster_id=1 has fraction 0.517
    assert n_t0 / 6556 == pytest.approx(expected_t0_fraction, abs=1e-6)


def test_freeze_bridge_setup_idempotent(tmp_path):
    """Running freeze_bridge_setup twice produces byte-identical bridge_setup.json."""
    setup_path = tmp_path / "bridge_setup.json"
    bridge.freeze_bridge_setup(
        cohort=["442"],
        results_root=Path("/home/honglab/leijiaxin/HFOsp/results"),
        artifact_root=Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns"),
        out_path=setup_path,
    )
    first = setup_path.read_bytes()
    bridge.freeze_bridge_setup(
        cohort=["442"],
        results_root=Path("/home/honglab/leijiaxin/HFOsp/results"),
        artifact_root=Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns"),
        out_path=setup_path,
    )
    second = setup_path.read_bytes()
    assert first == second


def test_freeze_bridge_setup_audit_rerun_marker(tmp_path):
    """bridge_setup.json must include audit-rerun completion marker."""
    setup_path = tmp_path / "bridge_setup.json"
    bridge.freeze_bridge_setup(
        cohort=["442"],
        results_root=Path("/home/honglab/leijiaxin/HFOsp/results"),
        artifact_root=Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns"),
        out_path=setup_path,
    )
    with setup_path.open() as fh:
        d = json.load(fh)
    assert "audit_rerun_marker_log_line" in d
    assert "cohort_summary.csv" in d["audit_rerun_marker_log_line"]


def test_compute_pre_ictal_fingerprint_basic():
    """6 events in [-30, -1] min with template ids [0,0,1,1,0,1]; T0=0.
    Expect frac_T0=0.5, switch_rate=3/5=0.6, last_template=1, n_events=6.
    Pairs: (0,0)(0,1)(1,1)(1,0)(0,1) → switches at 3 of 5 positions.
    """
    onset = 1_000_000.0  # arbitrary epoch
    # event times in seconds; -30 min = -1800 s, -1 min = -60 s
    event_times = onset + np.array([-1500, -1200, -900, -600, -300, -100], dtype=float)
    event_template_ids = np.array([0, 0, 1, 1, 0, 1], dtype=int)
    out = bridge.compute_pre_ictal_fingerprint(
        event_times=event_times,
        event_template_ids=event_template_ids,
        seizure_clinical_onset=onset,
        window_min_min=-30.0,
        window_min_max=-1.0,
        t0_template_id=0,
    )
    assert out["n_events"] == 6
    assert out["frac_T0"] == pytest.approx(0.5)
    assert out["switch_rate"] == pytest.approx(3 / 5)
    assert out["last_template"] == 1
    assert out["dropped_reason"] is None


def test_compute_pre_ictal_fingerprint_zero_events_drop():
    """0 events in window → dropped, fingerprint values are NaN."""
    onset = 1_000_000.0
    event_times = onset + np.array([-3600, +60], dtype=float)  # outside window
    event_template_ids = np.array([0, 1], dtype=int)
    out = bridge.compute_pre_ictal_fingerprint(
        event_times=event_times,
        event_template_ids=event_template_ids,
        seizure_clinical_onset=onset,
        window_min_min=-30.0,
        window_min_max=-1.0,
        t0_template_id=0,
    )
    assert out["n_events"] == 0
    assert out["dropped_reason"] == "no_events_in_window"
    assert math.isnan(out["frac_T0"])
    assert math.isnan(out["switch_rate"])
    assert out["last_template"] is None


def test_compute_pre_ictal_fingerprint_one_event_switch_nan():
    """1 event → switch_rate is NaN (no transitions defined)."""
    onset = 1_000_000.0
    event_times = onset + np.array([-600], dtype=float)
    event_template_ids = np.array([0], dtype=int)
    out = bridge.compute_pre_ictal_fingerprint(
        event_times=event_times,
        event_template_ids=event_template_ids,
        seizure_clinical_onset=onset,
        window_min_min=-30.0,
        window_min_max=-1.0,
        t0_template_id=0,
    )
    assert out["n_events"] == 1
    assert out["frac_T0"] == pytest.approx(1.0)
    assert math.isnan(out["switch_rate"])
    assert out["last_template"] == 0


def test_build_subject_fingerprint_table_442():
    """442: 17 seizures, 1 outlier (sz_id ending '00102' is index 8 in seizure_ids_kept).
    All seizures should have fingerprint computed for [-30, -1] min window;
    expect a tall DataFrame with columns including subtype_label and dropped_reason.
    """
    df = bridge.build_subject_fingerprint_table(
        subject="442",
        band="gamma_ER",
        results_root=Path("/home/honglab/leijiaxin/HFOsp/results"),
        artifact_root=Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns"),
        windows_min=[(-30.0, -1.0)],
    )
    assert isinstance(df, pd.DataFrame)
    assert {
        "subject", "band", "window_min_min", "window_min_max",
        "seizure_id", "subtype_label",
        "n_events", "frac_T0", "switch_rate", "last_template", "dropped_reason",
    }.issubset(df.columns)
    # 17 seizures × 1 window = 17 rows
    assert len(df) == 17
    # subtype labels: 16x 0 + 1x -1
    assert (df["subtype_label"] == 0).sum() == 16
    assert (df["subtype_label"] == -1).sum() == 1


def test_compute_pre_ictal_fingerprint_window_boundary_inclusive_left_exclusive_right():
    """Convention: t ∈ [onset + window_min*60, onset + window_max*60).
    A point exactly at the right edge is excluded; at the left edge is included.
    A point just inside the right edge (onset + (-60.001) < win_hi) is included.
    """
    onset = 1_000_000.0
    # window [-30, -1] min → [onset-1800, onset-60) s
    # -60.001 s from onset is just inside the window (< win_hi = onset-60)
    # -60.0 s from onset is exactly at win_hi, so excluded
    event_times = onset + np.array([-1800.0, -60.0, -60.001], dtype=float)
    event_template_ids = np.array([0, 1, 1], dtype=int)
    out = bridge.compute_pre_ictal_fingerprint(
        event_times=event_times,
        event_template_ids=event_template_ids,
        seizure_clinical_onset=onset,
        window_min_min=-30.0,
        window_min_max=-1.0,
        t0_template_id=0,
    )
    assert out["n_events"] == 2  # -1800 included, -60.001 included, -60.0 excluded
    assert out["last_template"] == 1  # the -60.001 event (closer to onset)


# ---------------------------------------------------------------------------
# Task 8: per-feature statistical helpers
# ---------------------------------------------------------------------------

def test_mann_whitney_with_rank_biserial():
    """Two groups, clear separation → small p, large |effect|."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 11.0, 12.0, 13.0])
    p, eff = bridge._mann_whitney_with_effect(a, b)
    assert p < 0.05
    assert abs(eff) > 0.9


def test_kruskal_wallis_with_eps2():
    """Three groups, large between-group variance → small p, large ε²."""
    g0 = np.array([1.0, 1.5, 2.0])
    g1 = np.array([5.0, 5.5, 6.0])
    g2 = np.array([10.0, 10.5, 11.0])
    p, eff = bridge._kruskal_wallis_with_effect([g0, g1, g2])
    assert p < 0.05
    assert eff > 0.5


def test_fisher_with_cramer_v_2x2():
    """2x2 contingency, perfect separation → p<0.05, V≈1."""
    contingency = np.array([[5, 0], [0, 5]], dtype=int)
    p, eff = bridge._fisher_or_chi2_with_cramer_v(contingency)
    assert p < 0.05
    assert eff > 0.9


def test_fisher_with_cramer_v_3x2():
    """3 subtypes × 2 last_template levels."""
    contingency = np.array([[4, 0], [0, 4], [2, 2]], dtype=int)
    p, eff = bridge._fisher_or_chi2_with_cramer_v(contingency)
    assert 0 <= eff <= 1
