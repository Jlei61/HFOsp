"""Tests for src/topic1_topic5_bridge.py."""

from __future__ import annotations

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
