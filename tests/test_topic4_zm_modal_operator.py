import numpy as np
import pytest

from src.topic4_zm_modal_operator import (
    analyze_discrete_operator,
    equal_energy_perturbations,
    evaluate_operator_prediction,
    fit_discrete_operator,
    infer_linearity_range,
    mode_axis_angle_deg,
    route_source_temporal_class,
    route_operator_tool,
)


def test_equal_energy_spatial_perturbations_are_distinct_and_matched():
    modes = equal_energy_perturbations(
        12, theta_deg=25.0, energy=0.04, random_seed=7
    )
    assert set(modes) == {"axial", "transverse", "isotropic", "core", "random"}
    flattened = {key: value.ravel() for key, value in modes.items()}
    for value in flattened.values():
        assert np.isclose(np.sum(value ** 2), 0.04)
        assert abs(np.mean(value)) < 1e-12
    assert abs(np.dot(flattened["axial"], flattened["transverse"])) < 1e-8
    assert not np.allclose(flattened["core"], flattened["isotropic"])


def test_carrier_type_routes_to_valid_tool_and_periodic_mean_jacobian_is_forbidden():
    assert route_source_temporal_class("global_periodic_candidate") == "periodic"
    assert route_source_temporal_class("phase_staggered_periodic_candidate") == "periodic"
    assert route_source_temporal_class("asynchronous_or_irregular_candidate") == "stochastic"
    with pytest.raises(ValueError, match="insufficient"):
        route_source_temporal_class("insufficient_active_cells")
    assert route_operator_tool("fixed") == "eigen"
    assert route_operator_tool("periodic") == "stroboscopic_floquet"
    assert route_operator_tool("stochastic") == "dmd_finite_time_gain"
    with pytest.raises(ValueError, match="periodic"):
        route_operator_tool("periodic", requested_tool="eigen")


def test_fitted_operator_predicts_held_out_states_and_recovers_rotated_soft_mode():
    theta = np.deg2rad(32.0)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    truth = rotation @ np.diag([0.96, 0.62]) @ rotation.T
    rng = np.random.default_rng(4)
    x = rng.normal(size=(400, 2))
    y = x @ truth.T
    fit = fit_discrete_operator(x[:300], y[:300], ridge=1e-10)
    pred = x[300:] @ fit["operator"].T
    error = np.linalg.norm(pred - y[300:]) / np.linalg.norm(y[300:])
    assert error < 1e-8
    heldout = evaluate_operator_prediction(fit["operator"], x[300:], y[300:])
    assert heldout["heldout_relative_error"] < 1e-8

    summary = analyze_discrete_operator(fit["operator"], dt_ms=2.0, horizon_ms=20.0)
    angle = mode_axis_angle_deg(summary["leading_right_mode"], [1.0, 0.0])
    assert abs(angle - 32.0) < 0.5
    assert summary["spectral_radius"] < 1.0


def test_finite_time_gain_detects_nonnormal_amplification():
    operator = np.array([[0.90, 0.80], [0.0, 0.80]])
    summary = analyze_discrete_operator(operator, dt_ms=1.0, horizon_ms=8.0)
    assert summary["spectral_radius"] < 1.0
    assert summary["finite_time_gain"] > 1.0
    assert len(summary["optimal_input_mode"]) == 2


def test_complex_oscillatory_modes_are_phase_aligned_not_rejected():
    theta = 0.35
    operator = 0.97 * np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    summary = analyze_discrete_operator(operator, dt_ms=2.0, horizon_ms=20.0)
    assert abs(summary["leading_eigenvalue_imag"]) > 0
    assert np.isfinite(mode_axis_angle_deg(summary["leading_right_mode"], [1, 0]))


def test_linearity_range_is_largest_contiguous_passing_amplitude():
    out = infer_linearity_range(
        [
            {"amplitude": 0.001, "heldout_relative_error": 0.05},
            {"amplitude": 0.003, "heldout_relative_error": 0.08},
            {"amplitude": 0.010, "heldout_relative_error": 0.24},
        ],
        max_relative_error=0.15,
    )
    assert out["status"] == "identified"
    assert out["maximum_linear_amplitude"] == 0.003
    assert out["n_passing"] == 2

    failed = infer_linearity_range(
        [{"amplitude": 0.001, "heldout_relative_error": 0.5}],
        max_relative_error=0.15,
    )
    assert failed["status"] == "no_valid_linear_range"
