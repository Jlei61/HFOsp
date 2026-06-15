"""TDD for src.topic5_t0_features (pure early-ictal activation windowing/summaries)."""
import numpy as np
import pytest

from src.topic5_t0_features import (
    onset_window_indices, activation_mean, ramp_slope,
    window_indices, window_activation, AXIS_WINDOWS)

HOP = 0.1


def test_onset_window_indices_maps_post_onset_window():
    # pre=120s -> onset at frame 1200; [0,10]s post-onset -> frames [1200, 1300)
    n = 1351  # 135s window at hop 0.1
    idx = onset_window_indices(n, pre_sec=120.0, hop_sec=HOP, t0_sec=0.0, t1_sec=10.0)
    assert idx[0] == 1200
    assert idx[-1] == 1299
    assert idx.size == 100


def test_onset_window_clipped_when_window_short():
    # only 1250 frames -> [0,10]s would need 1300; clip to available
    idx = onset_window_indices(1250, pre_sec=120.0, hop_sec=HOP)
    assert idx[0] == 1200 and idx[-1] == 1249


def test_activation_mean_per_channel():
    # ch0 flat 2.0, ch1 flat 0.0 over the window
    z = np.zeros((2, 50)); z[0, :] = 2.0
    win = np.arange(10, 40)
    out = activation_mean(z, win)
    assert np.isclose(out[0], 2.0) and np.isclose(out[1], 0.0)


def test_activation_mean_tolerates_nan_frames():
    z = np.full((1, 20), np.nan); z[0, 5:10] = 4.0  # only some frames finite
    out = activation_mean(z, np.arange(0, 20))
    assert np.isclose(out[0], 4.0)               # nanmean over the finite frames
    out2 = activation_mean(z, np.arange(15, 20)) # window has only NaN
    assert np.isnan(out2[0])


def test_ramp_slope_recovers_known_slope():
    # z rises 0.5 z-units/sec; hop 0.1 -> per-frame +0.05
    win = np.arange(0, 100)
    z = (win * 0.05)[None, :].astype(float)      # 1 channel, slope 0.5/s
    out = ramp_slope(z, win, hop_sec=HOP)
    assert np.isclose(out[0], 0.5, atol=1e-6)


def test_ramp_slope_nan_when_insufficient_points():
    z = np.full((1, 100), np.nan); z[0, 3] = 1.0
    out = ramp_slope(z, np.arange(0, 100), hop_sec=HOP)
    assert np.isnan(out[0])


# --- relt-driven window slicing (the multi-window cache, v2) ------------------
# rel_t = each bin's time RELATIVE to onset (= band_power_trace times - pre_sec).
# Spans the full extracted window [-pre, +post]; all 6 axis windows are slices of it.
REL_T = np.round(np.arange(-130.0, 30.0, 0.1), 6)   # [-130, +30) at 0.1s, onset=0


def test_window_indices_post_0_10_inclusive_bounds():
    idx = window_indices(REL_T, 0.0, 10.0)
    assert np.all(REL_T[idx] >= 0.0) and np.all(REL_T[idx] <= 10.0)
    # inclusive of both edges: rel=0.0 and rel=10.0 bins are present
    assert np.isclose(REL_T[idx][0], 0.0) and np.isclose(REL_T[idx][-1], 10.0)
    assert idx.size == 101                                  # 0.0 .. 10.0 inclusive at 0.1


def test_window_indices_distal_pre_m120_m90():
    idx = window_indices(REL_T, -120.0, -90.0)
    assert np.all(REL_T[idx] >= -120.0) and np.all(REL_T[idx] <= -90.0)
    assert np.isclose(REL_T[idx][0], -120.0) and np.isclose(REL_T[idx][-1], -90.0)
    # the distal window is ENTIRELY before onset (load-bearing negative control)
    assert np.all(REL_T[idx] < 0.0)


def test_window_indices_post_5_10_excludes_below_5():
    idx = window_indices(REL_T, 5.0, 10.0)
    assert np.all(REL_T[idx] >= 5.0)                        # no bin < 5 leaks in
    assert not np.any(np.isclose(REL_T[idx], 4.9))


def test_pre_windows_are_before_onset_post_windows_after():
    for key in ("pre_prox_m10_0", "pre_distal_m120_m90"):
        a, b = AXIS_WINDOWS[key]
        idx = window_indices(REL_T, a, b)
        assert np.all(REL_T[idx] <= 0.0)                    # pre = onset and earlier (rel<=0)
    for key in ("post_0_5", "post_5_10", "post_0_10", "post_0_20"):
        a, b = AXIS_WINDOWS[key]
        idx = window_indices(REL_T, a, b)
        assert np.all(REL_T[idx] >= 0.0)                    # post = onset and later (rel>=0)


def test_window_activation_per_channel_mean_matches_hand_calc():
    # 3 channels, known constant-per-channel values so the window mean is exact
    rel_t = np.round(np.arange(-5.0, 15.0, 0.1), 6)
    z = np.zeros((3, rel_t.size))
    z[0, :] = 2.0; z[1, :] = -1.0; z[2, :] = 7.5
    out = window_activation(z, rel_t, 0.0, 10.0)
    assert np.allclose(out, [2.0, -1.0, 7.5])
    # distinct window picks the same per-channel constant (mean invariant under constant)
    out2 = window_activation(z, rel_t, 5.0, 10.0)
    assert np.allclose(out2, [2.0, -1.0, 7.5])


def test_window_activation_reuses_activation_mean_nan_semantics():
    # same NaN contract as activation_mean: nanmean over finite bins; all-NaN window -> NaN
    rel_t = np.round(np.arange(0.0, 20.0, 0.1), 6)
    z = np.full((1, rel_t.size), np.nan)
    finite = (rel_t >= 2.0) & (rel_t <= 4.0)
    z[0, finite] = 4.0
    out = window_activation(z, rel_t, 0.0, 10.0)            # window has some finite bins
    assert np.isclose(out[0], 4.0)
    out2 = window_activation(z, rel_t, 12.0, 18.0)          # window all-NaN
    assert np.isnan(out2[0])


def test_axis_windows_constant_has_six_named_windows():
    assert set(AXIS_WINDOWS) == {
        "post_0_5", "post_5_10", "post_0_10", "post_0_20",
        "pre_prox_m10_0", "pre_distal_m120_m90"}
    assert AXIS_WINDOWS["post_0_20"] == (0, 20)
    assert AXIS_WINDOWS["pre_distal_m120_m90"] == (-120, -90)
