"""TDD tests for atlas pure-data helpers (no I/O, no plotting)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.plot_ictal_er_atlas import (
    REQUIRED_SCHEMA,
    _build_onset_matrix,
    _channel_order,
    _channel_role,
    _select_sort_band,
    _sort_band_unreliable,
)


# ---------------------------------------------------------------------------
# sort_band rule (spec §5.2)


def test_sort_band_prefers_stable_over_moderate():
    ps = {"producer_health": {"gamma_ER": "moderate", "broad_ER": "stable"}}
    assert _select_sort_band(ps) == "broad_ER"


def test_sort_band_prefers_moderate_over_unstable():
    ps = {"producer_health": {"gamma_ER": "unstable", "broad_ER": "moderate"}}
    assert _select_sort_band(ps) == "broad_ER"


def test_sort_band_tie_defaults_gamma():
    ps = {"producer_health": {"gamma_ER": "stable", "broad_ER": "stable"}}
    assert _select_sort_band(ps) == "gamma_ER"


def test_sort_band_unreliable_when_both_unstable():
    ps = {"producer_health": {"gamma_ER": "unstable", "broad_ER": "unstable"}}
    assert _sort_band_unreliable(ps) is True
    # Still falls back to gamma per spec
    assert _select_sort_band(ps) == "gamma_ER"


def test_sort_band_unreliable_false_when_one_moderate():
    ps = {"producer_health": {"gamma_ER": "moderate", "broad_ER": "unstable"}}
    assert _sort_band_unreliable(ps) is False
    assert _select_sort_band(ps) == "gamma_ER"


def test_sort_band_unreliable_when_both_insufficient():
    ps = {"producer_health": {"gamma_ER": "insufficient", "broad_ER": "insufficient"}}
    assert _sort_band_unreliable(ps) is True


# ---------------------------------------------------------------------------
# Channel role (spec §5.3)


def test_channel_role_focal_returns_soz():
    assert _channel_role("HL3", focal_set={"HL3", "HL4"}) == "soz"


def test_channel_role_non_focal_returns_other():
    assert _channel_role("HRA1", focal_set={"HL3", "HL4"}) == "other"


def test_channel_role_empty_focal_set_returns_other():
    assert _channel_role("HL3", focal_set=set()) == "other"


# ---------------------------------------------------------------------------
# Channel ordering (sort by sort_band r_sz asc; None at end)


def test_channel_order_sorts_by_r_sz_asc_with_none_at_end():
    ps = {
        "producer_health": {"gamma_ER": "stable", "broad_ER": "moderate"},
        "per_er": {
            "gamma_ER": {
                "r_sz": {"HL3": 2.5, "HL2": 1.5, "TBA1": 5.0,
                         "HRA1": None, "GA1": 4.0},
            },
            "broad_ER": {"r_sz": {}},
        },
    }
    chs, _ = _channel_order(ps, "gamma_ER")
    # Sorted ascending by r_sz, None at end
    assert chs == ["HL2", "HL3", "GA1", "TBA1", "HRA1"]


def test_channel_order_handles_all_none():
    ps = {
        "producer_health": {"gamma_ER": "insufficient",
                            "broad_ER": "insufficient"},
        "per_er": {
            "gamma_ER": {"r_sz": {"a": None, "b": None}},
            "broad_ER": {"r_sz": {}},
        },
    }
    chs, _ = _channel_order(ps, "gamma_ER")
    assert set(chs) == {"a", "b"}
    assert len(chs) == 2


# ---------------------------------------------------------------------------
# Onset matrix construction


def test_build_onset_matrix_shape_and_values():
    per_er = {
        "seizure_records": [
            {
                "seizure_idx": 0,
                "seizure_id": "sz0",
                "status": "ok",
                "channel_onsets": {
                    "HL3": {"frame_idx": 1500, "t_onset_sec": -45.0},
                    "HL2": {"frame_idx": 1700, "t_onset_sec": -25.0},
                    "TBA1": {"frame_idx": None, "t_onset_sec": None},
                },
            },
            {
                "seizure_idx": 1,
                "seizure_id": "sz1",
                "status": "ok",
                "channel_onsets": {
                    "HL3": {"frame_idx": 1600, "t_onset_sec": -35.0},
                    "HL2": {"frame_idx": None, "t_onset_sec": None},
                    "TBA1": {"frame_idx": 1900, "t_onset_sec": -5.0},
                },
            },
        ]
    }
    onset, statuses, sids = _build_onset_matrix(per_er, ["HL3", "HL2", "TBA1"])
    assert onset.shape == (3, 2)
    # HL3: -45 / -35
    assert onset[0, 0] == pytest.approx(-45.0)
    assert onset[0, 1] == pytest.approx(-35.0)
    # HL2: -25 / NaN
    assert onset[1, 0] == pytest.approx(-25.0)
    assert np.isnan(onset[1, 1])
    # TBA1: NaN / -5
    assert np.isnan(onset[2, 0])
    assert onset[2, 1] == pytest.approx(-5.0)
    assert list(statuses) == ["ok", "ok"]
    assert sids == ["sz0", "sz1"]


def test_build_onset_matrix_baseline_invalid_yields_nan_column():
    """seizure with no channel_onsets (e.g. baseline_invalid) → all NaN."""
    per_er = {
        "seizure_records": [
            {"seizure_idx": 0, "seizure_id": "sz0",
             "status": "baseline_invalid"},  # no channel_onsets key
        ]
    }
    onset, statuses, _ = _build_onset_matrix(per_er, ["HL3", "HL2"])
    assert onset.shape == (2, 1)
    assert np.all(np.isnan(onset))
    assert statuses[0] == "baseline_invalid"


def test_build_onset_matrix_unknown_channel_yields_nan():
    """A channel not present in channel_onsets stays NaN."""
    per_er = {
        "seizure_records": [
            {
                "seizure_idx": 0,
                "seizure_id": "sz0",
                "status": "ok",
                "channel_onsets": {
                    "HL3": {"frame_idx": 1500, "t_onset_sec": -45.0},
                },
            },
        ]
    }
    onset, _, _ = _build_onset_matrix(per_er, ["HL3", "GHOST"])
    assert onset[0, 0] == pytest.approx(-45.0)
    assert np.isnan(onset[1, 0])


def test_required_schema_constant_matches_spec():
    """Hard-fail if anyone bumps the schema name without updating both files."""
    assert REQUIRED_SCHEMA == "pr_t3_1_layer_a_v2_3_timing"
