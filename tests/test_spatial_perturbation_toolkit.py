"""Unit tests for the model-agnostic spatial perturbation toolbox."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.spatial_perturbation_toolkit import (  # noqa: E402
    finite_time_operator_svd, operator_gain_envelope, linear_response_timecourse,
    response_gain_curve, region_response_curve,
    cumulative_response_ratio, first_arrival_times, fit_arrival_time_distance,
    normalized_field_overlap,
)


def test_finite_time_svd_respects_declared_input_and_readout_spaces():
    J = np.diag([-0.1, -0.2, -0.3])
    B = np.eye(3)[:, :2]
    out = finite_time_operator_svd(J, 10.0, B, output_indices=slice(0, 2))
    assert np.isclose(out["sigma1"], np.exp(-1.0))
    assert out["optimal_input_coordinates"].shape == (2,)
    assert out["optimal_output"].shape == (2,)
    t, g = operator_gain_envelope(J, [0.0, 10.0], B, output_indices=slice(0, 2))
    assert np.allclose(t, [0, 10]) and np.allclose(g, [1.0, np.exp(-1.0)])


def test_linear_response_keeps_the_same_input_and_known_decay():
    J = -0.1 * np.eye(3)
    b = np.array([1.0, 0.0, 0.0])
    ev = linear_response_timecourse(J, b, [0.0, 10.0, 20.0])
    t, g = response_gain_curve(ev)
    assert np.allclose(t, [0, 10, 20])
    assert np.allclose(g, np.exp(-0.1 * t))


def test_region_curve_uses_fixed_mask_and_no_sign_cancellation():
    ev = {0.0: np.array([[1.0, -1.0], [0.0, 0.0]]),
          1.0: np.array([[2.0, -2.0], [0.0, 0.0]])}
    _, rms = region_response_curve(ev, np.array([[True, True], [False, False]]))
    assert np.allclose(rms, [1.0, 2.0])


def test_cumulative_ratio_is_stable_when_instantaneous_source_crosses_zero():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    source = np.array([1.0, 0.0, -1.0, 0.0])
    sink = np.array([0.1, 0.2, 0.3, 0.4])
    ratio = cumulative_response_ratio(sink, source, t)
    assert np.all(np.isfinite(ratio)) and np.nanmax(ratio) < 2.0


def test_first_arrival_and_velocity_recover_synthetic_wavefront():
    times = np.arange(0.0, 11.0)
    positions = np.arange(0.0, 5.0)
    K = np.zeros((times.size, positions.size))
    for j in range(positions.size):
        K[times >= 2.0 * positions[j], j] = 1.0
    arrival, threshold = first_arrival_times(K, times, threshold_fraction=0.5)
    fit = fit_arrival_time_distance(positions, arrival, source_position=0.0, sink_position=4.0)
    assert threshold == 0.5 and np.allclose(arrival, 2.0 * positions)
    assert fit["eligible"] and abs(fit["slope_ms_per_unit"] - 2.0) < 1e-12
    assert abs(fit["velocity_unit_per_ms"] - 0.5) < 1e-12 and fit["r2"] > 0.999


def test_arrival_fit_fails_closed_when_too_few_positions_cross():
    fit = fit_arrival_time_distance([0, 1, 2], [0, np.nan, np.nan],
                                    source_position=0, sink_position=2, min_points=2)
    assert not fit["eligible"] and fit["velocity_unit_per_ms"] is None


def test_mode_overlap_is_scale_invariant():
    a = np.array([[1.0, 2.0], [0.0, 0.0]])
    assert np.isclose(normalized_field_overlap(a, 3.0 * a), 1.0)
    assert normalized_field_overlap(a, np.rot90(a)) < 1.0
