"""TDD tests for atlas pure-data helpers (no I/O, no plotting)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.plot_ictal_er_atlas import (
    REQUIRED_SCHEMA,
    _build_onset_matrix,
    _channel_order,
    _channel_role,
    _clip_display_window_to_signal,
    _display_window_around_eeg,
    _heatmap_order_from_display_entries,
    _lagpat_json_to_display_clusters,
    _ordered_display_rows,
    _row_order_per_seizure,
    _select_sort_band,
    _sort_band_unreliable,
    _select_bg_traces_in_window,
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


# ---------------------------------------------------------------------------
# Per-seizure row ordering (spec §4.1)


def test_row_order_per_seizure_soz_first_then_by_onset():
    chs = ["HRA1", "HL3", "HL2", "TBA1", "GA1", "HL5"]
    focal = {"HL3", "HL2", "TBA1"}
    onsets = {
        "HRA1": +5.0,    # non-SOZ, t=5
        "HL3":  -45.0,   # SOZ, t=-45
        "HL2":  -25.0,   # SOZ, t=-25
        "TBA1": -10.0,   # SOZ, t=-10
        "GA1":  None,    # non-SOZ, no onset
        "HL5":  -8.0,    # non-SOZ, t=-8
    }
    order = _row_order_per_seizure(chs, focal, onsets)
    ordered_chs = [chs[i] for i in order]
    # SOZ tier first (HL3 -45 < HL2 -25 < TBA1 -10), then non-SOZ
    # (HL5 -8 < HRA1 +5 < GA1 NaN)
    assert ordered_chs == ["HL3", "HL2", "TBA1", "HL5", "HRA1", "GA1"]


def test_row_order_per_seizure_no_focal_pure_onset_sort():
    chs = ["A", "B", "C", "D"]
    onsets = {"A": +20.0, "B": -10.0, "C": None, "D": -50.0}
    order = _row_order_per_seizure(chs, focal_set=set(), onsets=onsets)
    # All same tier; sort by onset asc, NaN at end: D (-50), B (-10), A (+20), C
    assert [chs[i] for i in order] == ["D", "B", "A", "C"]


def test_row_order_per_seizure_returns_permutation():
    chs = ["a", "b", "c", "d", "e"]
    order = _row_order_per_seizure(chs, focal_set={"b"}, onsets={})
    # Output is a permutation of [0..4]
    assert sorted(order) == [0, 1, 2, 3, 4]
    # 'b' (focal) must be first
    assert chs[order[0]] == "b"


# ---------------------------------------------------------------------------
# Per-seizure display controls


def test_display_window_centers_on_eeg_onset_when_available():
    assert _display_window_around_eeg(-35.0) == pytest.approx((-125.0, 55.0))


def test_display_window_falls_back_to_zero_for_yuquan_reference():
    assert _display_window_around_eeg(None) == pytest.approx((-90.0, 90.0))


def test_clip_window_keeps_sane_eeg_centered_window():
    # legitimate large eeg-to-clinical offset (-150s) stays eeg-centered
    assert _clip_display_window_to_signal(
        (-240.0, -60.0), pre_sec=300.0, post_sec=100.0
    ) == pytest.approx((-240.0, -60.0))


def test_clip_window_falls_back_to_zero_zoom_for_bogus_eeg_onset():
    # bogus eeg_onset thousands of s off -> window outside signal -> ±90 around 0,
    # NOT the full [-pre, post] span (would defeat the zoom).
    assert _clip_display_window_to_signal(
        (-11329.0, -11149.0), pre_sec=300.0, post_sec=100.0
    ) == pytest.approx((-90.0, 90.0))


def test_alignment_reference_prefers_eeg_else_clinical():
    from scripts.plot_ictal_er_atlas import _alignment_reference
    # Yuquan already eeg/ref-aligned
    assert _alignment_reference("yuquan", None, pre_sec=300, post_sec=30) == (0.0, True)
    # Epilepsiae: sane eeg offset -> align to eeg onset
    assert _alignment_reference("epilepsiae", -5.0, pre_sec=300, post_sec=30) == (-5.0, True)
    # bogus eeg far outside signal -> fall back to clinical onset
    assert _alignment_reference("epilepsiae", -11239.0, pre_sec=300, post_sec=30) == (0.0, False)
    assert _alignment_reference("epilepsiae", None, pre_sec=300, post_sec=30) == (0.0, False)


def test_select_sequence_rows_picks_onset_channels_ordered_by_onset():
    from scripts.plot_ictal_er_atlas import _select_sequence_rows
    ch = ["A", "B", "C", "D", "E"]
    t = np.linspace(-50.0, 50.0, 11)
    z = np.zeros((5, 11))
    z[0, 5] = 5.0    # A peak 5 (focal)
    z[1, 5] = 4.0    # B peak 4
    z[2, 5] = 3.0    # C peak 3
    z[3, 5] = 6.0    # D peak 6 but NO onset
    z[4, 5] = 99.0   # E not High-HI -> ignored
    onsets = {"A": -10.0, "B": -30.0, "C": 5.0, "D": None}
    rows = _select_sequence_rows(
        z, t, ch,
        high_hi_upper={"A", "B", "C", "D"},
        focal_upper={"A"},
        valid_mask=np.array([True] * 5),
        display_window=(-50.0, 50.0),
        onsets=onsets, align_ref_sec=0.0, max_ch=3,
    )
    # onset-bearing channels beat the strong no-onset D; ordered by onset asc
    assert [r["channel"] for r in rows] == ["B", "A", "C"]
    roles = {r["channel"]: r["role"] for r in rows}
    assert roles["A"] == "high_hi_ictal"   # focal
    assert roles["B"] == "high_hi_index"
    assert rows[0]["onset_disp"] == pytest.approx(-30.0)


def test_select_sequence_rows_caps_to_max_ch():
    from scripts.plot_ictal_er_atlas import _select_sequence_rows
    ch = [f"C{i}" for i in range(12)]
    t = np.linspace(-50.0, 50.0, 11)
    z = np.zeros((12, 11))
    for i in range(12):
        z[i, 5] = float(i + 1)
    onsets = {c: -float(i) for i, c in enumerate(ch)}
    rows = _select_sequence_rows(
        z, t, ch,
        high_hi_upper={c.upper() for c in ch},
        focal_upper=set(),
        valid_mask=np.array([True] * 12),
        display_window=(-50.0, 50.0),
        onsets=onsets, align_ref_sec=0.0, max_ch=8,
    )
    assert len(rows) == 8


def test_ordered_display_rows_groups_by_role_and_counts():
    entries = [
        {"idx": 7, "role": "other", "channel": "O1"},
        {"idx": 1, "role": "high_hi_index", "channel": "X1"},
        {"idx": 3, "role": "ictal", "channel": "I1"},
        {"idx": 4, "role": "high_hi_ictal", "channel": "H1"},
    ]
    rows, counts = _ordered_display_rows(entries)
    assert [r["idx"] for r in rows] == [4, 1, 3, 7]
    assert [r["channel"] for r in rows] == ["H1", "X1", "I1", "O1"]
    assert counts == {
        "high_hi_ictal": 1, "high_hi_index": 1, "ictal": 1, "other": 1,
    }


def test_select_bg_traces_uses_display_window_and_excludes_high_hi():
    z = np.zeros((4, 5), dtype=float)
    t = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
    # Channel C is strongest inside the displayed [-80, 20] window.
    z[1, 1:3] = 3.0
    z[2, 4] = 99.0
    z[3, 1:3] = 2.0
    sel = _select_bg_traces_in_window(
        z,
        t,
        ["A", "B", "C", "D"],
        {"A"},
        np.array([True, True, True, True]),
        (-80.0, 20.0),
        n_bg=2,
    )
    assert sel.tolist() == [1, 3]


def test_heatmap_order_uses_raw_panel_entries_not_all_other_channels():
    entries = [
        {"idx": 7, "role": "other"},
        {"idx": 1, "role": "high_hi_index"},
        {"idx": 2, "role": "other"},
        {"idx": 3, "role": "ictal"},
        {"idx": 4, "role": "high_hi_ictal"},
    ]
    order, counts = _heatmap_order_from_display_entries(entries)
    assert order.tolist() == [4, 1, 3, 7, 2]
    assert counts == {
        "high_hi_ictal": 1,
        "high_hi_index": 1,
        "ictal": 1,
        "other": 2,
    }


def test_lagpat_json_to_display_clusters_aligns_rank_by_channel():
    d = {
        "channel_names": ["A1", "A2", "A3"],
        "adaptive_cluster": {
            "clusters": [
                {
                    "cluster_id": 1,
                    "n_events": 10,
                    "fraction": 0.5,
                    "template_rank": [2, None, 0],
                }
            ]
        },
    }
    channels, clusters = _lagpat_json_to_display_clusters("yuquan/fake", d)
    assert channels == ["A1", "A2", "A3"]
    assert clusters[0]["rank_by_channel"] == {"A1": 2, "A2": None, "A3": 0}
