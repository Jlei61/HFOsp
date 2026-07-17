import numpy as np

from src.topic5_energy_timing import (
    band_energy_timing,
    centered_window_hit_profile,
    detect_centered_window_enhancement,
    detect_multiband_recruitment_onset,
    detect_sustained_enhancement,
    distal_baseline_delta,
    max_upward_transition,
    spatial_quantile_trace,
)


def test_distal_baseline_delta_is_per_contact() -> None:
    t = np.arange(-120.0, 21.0, 1.0)
    z = np.vstack([np.ones(t.size) * 3.0, np.ones(t.size) * -2.0])
    z[:, t >= -10.0] += np.array([[4.0], [1.0]])
    out = distal_baseline_delta(z, t, baseline=(-120.0, -90.0))
    assert np.allclose(np.nanmedian(out[:, (t >= -120) & (t < -90)], axis=1), 0.0)
    assert np.allclose(out[:, t == 0.0].ravel(), [4.0, 1.0])


def test_spatial_quantile_uses_fixed_contact_axis() -> None:
    z = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    got = spatial_quantile_trace(z, q=0.75)
    assert np.allclose(got, [2.25, 3.25])


def test_detector_rejects_short_spike_and_finds_sustained_rise() -> None:
    t = np.arange(-120.0, 20.1, 0.1)
    y = np.zeros(t.size)
    y[(t >= -40.0) & (t < -39.5)] = 5.0
    y[(t >= -12.0) & (t < -8.0)] = 4.0
    got = detect_sustained_enhancement(
        y,
        t,
        baseline=(-120.0, -90.0),
        search=(-60.0, 20.0),
        baseline_quantile=0.99,
        sustain_sec=2.0,
    )
    assert got.detected is True
    assert abs(got.rise_sec - (-12.0)) <= 0.11
    assert got.longest_above_sec >= 3.9


def test_detector_marks_no_sustained_enhancement() -> None:
    t = np.arange(-120.0, 20.1, 0.1)
    y = np.zeros(t.size)
    y[(t >= -4.0) & (t < -3.5)] = 2.0
    got = detect_sustained_enhancement(y, t, sustain_sec=2.0)
    assert got.detected is False
    assert np.isnan(got.rise_sec)
    assert abs(got.peak_sec - (-4.0)) <= 0.11


def test_full_pipeline_detects_distributed_contact_gain() -> None:
    t = np.arange(-120.0, 20.1, 0.1)
    z = np.zeros((8, t.size))
    z[:4, (t >= -9.0) & (t < -3.0)] = 6.0
    trace, got = band_energy_timing(z, t, smooth_sec=0.0, sustain_sec=2.0)
    assert trace.shape == t.shape
    assert got.detected is True
    assert abs(got.rise_sec - (-9.0)) <= 0.11


def test_max_upward_transition_finds_late_sharp_gain_not_early_plateau() -> None:
    t = np.arange(-120.0, 20.1, 0.1)
    y = np.zeros(t.size)
    y[(t >= -40.0) & (t < -10.0)] = 1.0
    y[t >= -2.0] = 7.0
    step, got = max_upward_transition(y, t, flank_sec=2.0)
    assert step.shape == t.shape
    assert got.detected is True
    assert abs(got.transition_sec - (-2.0)) <= 0.11
    assert got.step_delta > 5.0


def test_max_upward_transition_can_identify_true_early_episode() -> None:
    t = np.arange(-120.0, 20.1, 0.1)
    y = np.zeros(t.size)
    y[(t >= -48.0) & (t < -35.0)] = 5.0
    y[(t >= -2.0) & (t < 8.0)] = 2.0
    _, got = max_upward_transition(y, t, flank_sec=2.0)
    assert got.detected is True
    assert abs(got.transition_sec - (-48.0)) <= 0.11


def test_centered_window_enhancement_distinguishes_true_and_shifted_onset() -> None:
    t = np.arange(-120.0, 20.1, 0.1)
    y = np.zeros(t.size)
    y[(t >= -2.0) & (t < 4.0)] = 5.0
    true = detect_centered_window_enhancement(y, t, center_sec=0.0)
    shifted = detect_centered_window_enhancement(y, t, center_sec=-30.0)
    assert true.detected is True
    assert shifted.detected is False


def test_centered_window_hit_profile_preserves_center_order() -> None:
    t = np.arange(-120.0, 20.1, 0.1)
    y = np.zeros(t.size)
    y[(t >= -32.0) & (t < -26.0)] = 5.0
    got = centered_window_hit_profile(y, t, np.array([0.0, -30.0, -10.0]))
    assert got.tolist() == [False, True, False]


def test_multiband_recruitment_onset_finds_confirmed_change_point() -> None:
    t = np.arange(-120.0, 5.1, 0.1)
    traces = np.zeros((5, t.size))
    traces[:4, t >= -12.0] = 5.0
    got = detect_multiband_recruitment_onset(traces, t, majority_required=3)
    assert got.detected is True
    assert abs(got.onset_sec - (-12.0)) <= 0.11
    assert got.n_band_post_sustained == 4


def test_multiband_recruitment_onset_rejects_short_transient() -> None:
    t = np.arange(-120.0, 5.1, 0.1)
    traces = np.zeros((5, t.size))
    traces[:4, (t >= -12.0) & (t < -11.5)] = 5.0
    got = detect_multiband_recruitment_onset(traces, t, majority_required=3)
    assert got.detected is False
    assert got.consensus_post_sustained is False
