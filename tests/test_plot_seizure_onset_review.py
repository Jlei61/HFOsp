"""Unit tests for the pure helpers of scripts/plot_seizure_onset_review.py.

Only the silently-wrong-prone math is tested: marker relative times (sign),
window margins (must bracket every marker), and robust normalization. Plotting
and EDF I/O are not tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.plot_seizure_onset_review import (
    _detail_windows,
    activity_envelope,
    duration_bits,
    reference_shift,
    robust_normalize_mad,
    seizure_marker_times,
    soz_channel_flags,
    window_margins,
)


# --- seizure_marker_times ---------------------------------------------------
def test_epilepsiae_markers_relative_to_clin_onset():
    row = {
        "clin_onset_epoch": "1000.0",
        "clin_offset_epoch": "1170.0",
        "eeg_onset_epoch": "994.0",     # eeg precedes clinical → negative t_rel
        "eeg_offset_epoch": "1175.0",
    }
    markers, t0 = seizure_marker_times(row, "epilepsiae")
    assert t0 == 1000.0
    by_label = {m["label"]: m for m in markers}
    assert by_label["clin onset"]["t_rel"] == 0.0
    assert by_label["clin offset"]["t_rel"] == pytest.approx(170.0)
    assert by_label["eeg onset"]["t_rel"] == pytest.approx(-6.0)
    assert by_label["eeg offset"]["t_rel"] == pytest.approx(175.0)
    assert by_label["clin onset"]["kind"] == "onset"
    assert by_label["clin offset"]["kind"] == "offset"


def test_yuquan_markers_relative_to_eeg_onset_only():
    row = {"eeg_onset_epoch": "500.0", "eeg_offset_epoch": "718.9"}
    markers, t0 = seizure_marker_times(row, "yuquan")
    assert t0 == 500.0
    assert {m["ann"] for m in markers} == {"eeg"}
    by_label = {m["label"]: m for m in markers}
    assert by_label["eeg onset"]["t_rel"] == 0.0
    assert by_label["eeg offset"]["t_rel"] == pytest.approx(218.9)


def test_missing_offset_is_omitted_not_zeroed():
    row = {"clin_onset_epoch": "1000.0", "clin_offset_epoch": ""}
    markers, _ = seizure_marker_times(row, "epilepsiae")
    labels = {m["label"] for m in markers}
    assert "clin onset" in labels
    assert "clin offset" not in labels


def test_missing_onset_reference_raises():
    with pytest.raises(ValueError):
        seizure_marker_times({"clin_offset_epoch": "1.0"}, "epilepsiae")


# --- window_margins ---------------------------------------------------------
def test_window_brackets_earliest_onset_and_latest_offset():
    markers = [
        {"t_rel": 0.0, "kind": "onset"},
        {"t_rel": -6.0, "kind": "onset"},   # eeg precedes clinical
        {"t_rel": 170.0, "kind": "offset"},
        {"t_rel": 175.0, "kind": "offset"},
    ]
    pre, post = window_margins(markers, pre_margin=30.0, post_margin=30.0,
                               default_post=120.0)
    # pre must reach back past the eeg onset at -6 plus the 30s margin.
    assert pre == pytest.approx(36.0)
    # post must reach past the latest offset (175) plus the 30s margin.
    assert post == pytest.approx(205.0)


def test_window_uses_default_post_when_no_offset():
    markers = [{"t_rel": 0.0, "kind": "onset"}]
    pre, post = window_margins(markers, pre_margin=20.0, post_margin=15.0,
                               default_post=120.0)
    assert pre == pytest.approx(20.0)
    assert post == pytest.approx(135.0)


# --- robust_normalize_mad ---------------------------------------------------
def test_mad_normalization_unit_scale_on_gaussian():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 5.0, size=(3, 20000))
    z = robust_normalize_mad(x)
    # 1.4826 * MAD ≈ sigma, so normalized std ≈ 1.
    assert np.allclose(np.std(z, axis=-1), 1.0, atol=0.05)
    assert np.allclose(np.median(z, axis=-1), 0.0, atol=0.05)


def test_mad_normalization_flat_channel_no_divzero():
    x = np.zeros((2, 100))
    z = robust_normalize_mad(x)
    assert np.all(np.isfinite(z))


# --- _detail_windows --------------------------------------------------------
def test_short_seizure_gets_single_full_span_panel():
    t = np.linspace(-30, 90, 1000)               # 120 s span < 160
    markers = [{"t_rel": 0.0, "kind": "onset"}, {"t_rel": 80.0, "kind": "offset"}]
    wins = _detail_windows(t, markers, post_margin=30.0)
    assert len(wins) == 1
    assert wins[0][1] == pytest.approx(-30.0)
    assert wins[0][2] == pytest.approx(90.0)


def test_long_seizure_splits_into_onset_and_end_zoom():
    t = np.linspace(-30, 650, 5000)              # 680 s span > 160
    markers = [{"t_rel": 0.0, "kind": "onset"}, {"t_rel": 623.0, "kind": "offset"}]
    wins = _detail_windows(t, markers, post_margin=30.0)
    assert [w[0] for w in wins] == ["onset zoom", "end zoom"]
    # onset zoom brackets the onset at high resolution (anchored on earliest onset)
    assert wins[0][1] == pytest.approx(-20.0)    # earliest onset (0) - 20
    assert wins[0][2] == pytest.approx(70.0)
    # end zoom is centered on the offset (623) and stays within loaded data
    assert wins[1][1] == pytest.approx(553.0)
    assert wins[1][2] == pytest.approx(650.0)   # min(650, 623+30)


def test_onset_zoom_anchors_on_earliest_onset_when_eeg_precedes_clin():
    # eeg (electrographic) onset 90 s before clinical onset → onset zoom must
    # reach back to include the earlier eeg onset, not crop it.
    t = np.linspace(-150, 400, 6000)
    markers = [
        {"t_rel": -90.0, "kind": "onset"},   # eeg
        {"t_rel": 0.0, "kind": "onset"},     # clin
        {"t_rel": 300.0, "kind": "offset"},
    ]
    wins = _detail_windows(t, markers, post_margin=30.0)
    onset_lo, onset_hi = wins[0][1], wins[0][2]
    assert onset_lo == pytest.approx(-110.0)         # earliest onset (-90) - 20
    assert onset_lo <= -90.0 <= onset_hi             # eeg onset visible
    assert onset_lo <= 0.0 <= onset_hi               # clin onset also visible


def test_long_seizure_offblock_offset_uses_recording_end_zoom():
    t = np.linspace(-30, 400, 4000)              # offset annotated past loaded data
    markers = [{"t_rel": 0.0, "kind": "onset"}, {"t_rel": 600.0, "kind": "offset"}]
    wins = _detail_windows(t, markers, post_margin=30.0)
    assert wins[1][0] == "recording-end zoom"
    assert wins[1][2] == pytest.approx(400.0)    # ends at loaded-data end


# --- reference_shift --------------------------------------------------------
def test_reference_shift_eeg_recenters_on_electrographic_onset():
    # markers in loader (clin) frame: eeg onset precedes clin by 3.5 s
    markers = [
        {"label": "clin onset", "t_rel": 0.0, "kind": "onset", "ann": "clin"},
        {"label": "eeg onset", "t_rel": -3.5, "kind": "onset", "ann": "eeg"},
        {"label": "clin offset", "t_rel": 86.5, "kind": "offset", "ann": "clin"},
    ]
    shift, label = reference_shift(markers, "eeg")
    assert shift == pytest.approx(-3.5)   # subtracting this puts eeg onset at 0
    assert label == "eeg onset"
    # after shifting, eeg onset -> 0, clin onset -> +3.5
    shifted = {m["label"]: m["t_rel"] - shift for m in markers}
    assert shifted["eeg onset"] == pytest.approx(0.0)
    assert shifted["clin onset"] == pytest.approx(3.5)


def test_reference_shift_falls_back_to_clin_when_no_eeg_onset():
    markers = [{"label": "clin onset", "t_rel": 0.0, "kind": "onset", "ann": "clin"}]
    assert reference_shift(markers, "eeg") == (0.0, "clin onset")


def test_reference_shift_clin_is_noop():
    markers = [{"label": "eeg onset", "t_rel": -3.5, "kind": "onset", "ann": "eeg"}]
    assert reference_shift(markers, "clin") == (0.0, "clin onset")


# --- duration_bits ----------------------------------------------------------
def test_duration_bits_are_true_per_source_durations():
    # eeg onset -3.5 / eeg offset 76.8  -> eeg dur 80.3; clin 0 / 86.5 -> 86.5
    markers = [
        {"t_rel": 0.0, "kind": "onset", "ann": "clin"},
        {"t_rel": 86.5, "kind": "offset", "ann": "clin"},
        {"t_rel": -3.5, "kind": "onset", "ann": "eeg"},
        {"t_rel": 76.8, "kind": "offset", "ann": "eeg"},
    ]
    bits = duration_bits(markers)
    assert "clin dur≈86.5s" in bits
    assert "eeg dur≈80.3s" in bits


# --- soz_channel_flags ------------------------------------------------------
def test_soz_flags_single_montage_exact_contact():
    names = ["HL7", "HL8", "GA1", "GA2"]
    soz = {"HL7", "HL8", "HL9", "HL10"}        # normalized contact set
    assert soz_channel_flags(names, soz) == [True, True, False, False]


def test_soz_flags_bipolar_any_constituent_contact():
    names = ["A6-A7", "B1-B2", "A9-A10"]
    soz = {"A7", "A10"}                         # bipolar channel SOZ if a contact hits
    assert soz_channel_flags(names, soz) == [True, False, True]


def test_soz_flags_empty_set_all_false():
    assert soz_channel_flags(["A1", "A2"], set()) == [False, False]


# --- activity_envelope ------------------------------------------------------
def test_envelope_is_nonnegative_and_length_preserving():
    rng = np.random.default_rng(1)
    sig = rng.normal(size=(5, 1000))
    env = activity_envelope(sig, fs=100.0, smooth_sec=0.5)
    assert env.shape == (1000,)
    assert np.all(env >= 0)
