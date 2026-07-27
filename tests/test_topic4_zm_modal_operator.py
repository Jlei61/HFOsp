import numpy as np
import pytest

from src.topic4_zm_modal_operator import (
    analyze_discrete_operator,
    apply_voltage_perturbation,
    assemble_central_propagator,
    equal_energy_perturbations,
    evaluate_operator_prediction,
    fit_discrete_operator,
    infer_linearity_range,
    mode_axis_angle_deg,
    project_ei_grid,
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


def test_voltage_probe_has_matched_rms_and_touches_only_selected_population():
    state = {"V": np.linspace(5.0, 10.0, 6)}
    posE = np.array([[0.1, 0.1], [0.8, 0.1], [0.2, 0.8], [0.8, 0.8]])
    posI = np.array([[0.2, 0.2], [0.7, 0.7]])
    field = np.array([[-1.0, 0.5], [0.0, 1.0]])
    out, delta = apply_voltage_perturbation(
        state, field, posE, posI, L=1.0, population="E",
        rms_amplitude_mv=0.2, sign=1,
    )
    assert np.isclose(np.sqrt(np.mean(delta ** 2)), 0.2)
    assert abs(np.mean(delta)) < 1e-12
    assert np.allclose(out["V"][:4] - state["V"][:4], delta)
    assert np.array_equal(out["V"][4:], state["V"][4:])


def test_central_pairs_recover_finite_time_propagator_and_require_matched_noise():
    truth = np.array([[0.8, 0.2], [-0.1, 0.6]])
    rows = []
    for j, name in enumerate(("axial_E", "transverse_E")):
        for amplitude in (0.01, 0.03):
            for sign in (-1, 1):
                x = np.zeros(2)
                x[j] = sign * amplitude
                rows.append(
                    {
                        "input_mode": name,
                        "amplitude": amplitude,
                        "sign": sign,
                        "bank_sha": f"bank-{amplitude}",
                        "response": (truth @ x).tolist(),
                    }
                )
    out = assemble_central_propagator(
        rows, input_order=("axial_E", "transverse_E"), amplitude=0.01
    )
    assert np.allclose(out["operator"], truth)

    rows[0]["bank_sha"] = "mismatch"
    with pytest.raises(ValueError, match="noise"):
        assemble_central_propagator(
            rows, input_order=("axial_E", "transverse_E"), amplitude=0.01
        )


def test_ei_grid_projection_uses_same_registered_spatial_basis():
    modes = {
        "axial": np.array([[-1.0, 0.0], [0.0, 1.0]]),
        "core": np.array([[1.0, -1.0], [-1.0, 1.0]]),
    }
    E = 2.0 * modes["axial"] + 0.5 * modes["core"]
    I = -0.3 * modes["axial"] + 1.2 * modes["core"]
    out = project_ei_grid(E, I, modes, mode_order=("axial", "core"))
    assert out["coordinate_order"] == ["axial_E", "core_E", "axial_I", "core_I"]
    assert np.allclose(out["coordinates"], [2.0, 0.5, -0.3, 1.2])
