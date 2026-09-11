import numpy as np
import pytest
from scipy import sparse

from src.topic4_dual_core_spatial_z import (
    DualCoreSpatialZMap,
    path_parameter_derivative,
    path_state,
    pseudo_arclength_spatial_z,
    solve_spatial_z_fixed_point,
    spatial_z_dynamic_jacobian,
    spatial_z_jacobian,
    spatial_z_residual,
)
from src.topic4_patient_zm_meanfield import (
    dynamic_jacobian,
    homogeneous_one_cell_model,
    transfer_rates,
)
from src.topic4_dual_core_spatial_z_delay import (
    CoarseDelayOperators,
    delayed_growth_rate,
    delayed_leading_eigenvalues,
    delayed_step_matrix,
    simulate_delayed_ou_trajectory,
)


def one_cell_surround_map():
    return DualCoreSpatialZMap(
        core_a_fraction_e=np.asarray([0.0]),
        core_b_fraction_e=np.asarray([0.0]),
        centers_mm=np.asarray([[0.25, 0.25], [0.75, 0.75]]),
        selected_count_per_core=np.asarray([0, 0]),
    )


def test_spatial_z_path_is_exact_regional_mixture():
    z_map = DualCoreSpatialZMap(
        core_a_fraction_e=np.asarray([1.0, 0.0, 0.25]),
        core_b_fraction_e=np.asarray([0.0, 1.0, 0.25]),
        centers_mm=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        selected_count_per_core=np.asarray([10, 11]),
    )
    observed = z_map.z_field(z_a=0.6, z_b=0.8, z_surround=1.0)
    np.testing.assert_allclose(observed, [0.6, 0.8, 0.85])
    observed_second = z_map.z_second_moment_field(
        z_a=0.6, z_b=0.8, z_surround=1.0)
    np.testing.assert_allclose(observed_second, [0.36, 0.64, 0.75])


def test_spatial_jacobian_and_path_derivative_match_finite_difference():
    model = homogeneous_one_cell_model(ratio=0.6)
    z_map = one_cell_surround_map()
    rates = np.asarray([0.06, 0.08])
    _za, _zb, _zs, field = path_state(
        z_map, 0.18, surround_weight=1.0)
    analytic = spatial_z_jacobian(model, rates, z_field=field)
    numerical = np.empty_like(analytic)
    h = 1e-6
    for column in range(rates.size):
        step = np.zeros(rates.size)
        step[column] = h
        numerical[:, column] = (
            spatial_z_residual(model, rates + step, z_field=field)
            - spatial_z_residual(model, rates - step, z_field=field)
        ) / (2.0 * h)
    assert analytic == pytest.approx(numerical, rel=2e-4, abs=2e-5)

    analytic_parameter = path_parameter_derivative(
        model, rates, z_field=field, depletion_profile=np.asarray([1.0]))
    plus = path_state(z_map, 0.18 + h, surround_weight=1.0)[-1]
    minus = path_state(z_map, 0.18 - h, surround_weight=1.0)[-1]
    numerical_parameter = (
        spatial_z_residual(model, rates, z_field=plus)
        - spatial_z_residual(model, rates, z_field=minus)
    ) / (2.0 * h)
    assert analytic_parameter == pytest.approx(
        numerical_parameter, rel=2e-4, abs=2e-5)


def test_mixed_cell_second_moment_path_derivative_matches_finite_difference():
    model = homogeneous_one_cell_model(ratio=0.6)
    z_map = DualCoreSpatialZMap(
        core_a_fraction_e=np.asarray([0.25]),
        core_b_fraction_e=np.asarray([0.25]),
        centers_mm=np.asarray([[0.25, 0.25], [0.75, 0.75]]),
        selected_count_per_core=np.asarray([1, 1]),
    )
    rates = np.asarray([0.06, 0.08])
    weights = (1.0, 0.5, 0.7)
    parameter = 0.18
    z_a, z_b, z_surround, z = path_state(
        z_map, parameter, core_a_weight=weights[0],
        core_b_weight=weights[1], surround_weight=weights[2])
    z2 = z_map.z_second_moment_field(
        z_a=z_a, z_b=z_b, z_surround=z_surround)
    profile = z_map.depletion_profile(
        core_a_weight=weights[0], core_b_weight=weights[1],
        surround_weight=weights[2])
    dz2 = -2.0 * (
        z_map.core_a_fraction_e * weights[0] * z_a
        + z_map.core_b_fraction_e * weights[1] * z_b
        + z_map.surround_fraction_e * weights[2] * z_surround)
    analytic = path_parameter_derivative(
        model, rates, z_field=z, z_second_moment=z2,
        depletion_profile=profile, z_second_moment_derivative=dz2)
    h = 1e-6
    fields = []
    for shifted in (parameter + h, parameter - h):
        a, b, surround, mean = path_state(
            z_map, shifted, core_a_weight=weights[0],
            core_b_weight=weights[1], surround_weight=weights[2])
        second = z_map.z_second_moment_field(
            z_a=a, z_b=b, z_surround=surround)
        fields.append((mean, second))
    numerical = (
        spatial_z_residual(
            model, rates, z_field=fields[0][0],
            z_second_moment=fields[0][1])
        - spatial_z_residual(
            model, rates, z_field=fields[1][0],
            z_second_moment=fields[1][1])
    ) / (2.0 * h)
    assert analytic == pytest.approx(numerical, rel=2e-4, abs=2e-5)


def test_spatial_pseudo_arclength_crosses_known_homogeneous_fold():
    model = homogeneous_one_cell_model(ratio=0.6)
    z_map = one_cell_surround_map()
    initial = np.asarray([0.30, 0.32])
    first = solve_spatial_z_fixed_point(
        model, z_map, parameter=0.16, initial_rates=initial,
        surround_weight=1.0)
    second = solve_spatial_z_fixed_point(
        model, z_map, parameter=0.15, initial_rates=first.rates,
        surround_weight=1.0)
    points = pseudo_arclength_spatial_z(
        model, z_map, first, second, surround_weight=1.0,
        step_size=0.01, n_steps=24)
    valid = [point for point in points if point.solution.converged]
    tangents = np.asarray([point.tangent_parameter for point in valid])
    parameters = np.asarray([point.solution.parameter for point in valid])
    assert np.any(tangents > 0.0)
    assert np.any(tangents < 0.0)
    assert 0.111 < parameters.min() < 0.115


def test_spatial_dynamic_jacobian_reduces_to_homogeneous_q_case():
    model = homogeneous_one_cell_model(ratio=0.6)
    rates = np.asarray([0.06, 0.08])
    observed = spatial_z_dynamic_jacobian(
        model, rates, z_field=np.asarray([0.82])).toarray()
    expected = dynamic_jacobian(model, rates, q=0.82).toarray()
    np.testing.assert_allclose(observed, expected)


def test_delay_aware_growth_assay_recovers_stable_uncoupled_system():
    model = homogeneous_one_cell_model(ratio=0.6)
    zero = sparse.csr_matrix((1, 1))
    operators = CoarseDelayOperators(
        dt_ms=0.1, max_delay_steps=1,
        w_ee_history=zero, w_ei_history=zero,
        w_ie_history=zero, w_ii_history=zero)
    result = delayed_growth_rate(
        model, operators, np.asarray([0.01, 0.02]),
        z_field=np.asarray([1.0]), n_steps=1200, burn_in_steps=600,
        seeds=(101, 102))
    assert result["classification"] == "stable"
    assert max(result["per_seed_growth_rate_per_ms"]) < -0.04


def test_exact_delay_spectrum_matches_uncoupled_membrane_mode():
    model = homogeneous_one_cell_model(ratio=0.6)
    zero = sparse.csr_matrix((1, 1))
    operators = CoarseDelayOperators(
        dt_ms=0.1, max_delay_steps=2,
        w_ee_history=sparse.hstack([zero, zero], format="csr"),
        w_ei_history=sparse.hstack([zero, zero], format="csr"),
        w_ie_history=sparse.hstack([zero, zero], format="csr"),
        w_ii_history=sparse.hstack([zero, zero], format="csr"))
    spectrum = delayed_leading_eigenvalues(
        model, operators, np.asarray([0.01, 0.02]),
        z_field=np.asarray([1.0]), k=2)
    expected = np.log(1.0 - 0.1 / model.tau_mem_e_ms) / 0.1
    assert spectrum[0]["growth_rate_per_ms"] == pytest.approx(
        expected, rel=1e-7, abs=1e-9)
    assert spectrum[0]["frequency_hz"] == pytest.approx(0.0, abs=1e-8)


def test_delay_tangent_includes_instantaneous_variance_gain():
    model = homogeneous_one_cell_model(ratio=0.6)
    z_map = one_cell_surround_map()
    parameter = 0.10
    solution = solve_spatial_z_fixed_point(
        model, z_map, parameter=parameter,
        initial_rates=np.asarray([0.01, 0.03]), surround_weight=1.0)
    operators = CoarseDelayOperators(
        dt_ms=0.1, max_delay_steps=1,
        w_ee_history=sparse.csr_matrix(model.w_ee),
        w_ei_history=sparse.csr_matrix(model.w_ei),
        w_ie_history=sparse.csr_matrix(model.w_ie),
        w_ii_history=sparse.csr_matrix(model.w_ii))
    z = path_state(z_map, parameter, surround_weight=1.0)[-1]
    matrix = delayed_step_matrix(
        model, operators, solution.rates, z_field=z).toarray()
    rate_e, rate_i = solution.rate_e, solution.rate_i
    syn = np.asarray([
        model.tau_mem_e_ms * (model.w_ee @ rate_e),
        model.tau_mem_e_ms * (model.w_ei @ rate_i),
        model.tau_mem_i_ms * (model.w_ie @ rate_e),
        model.tau_mem_i_ms * (model.w_ii @ rate_i),
    ])

    def rate_step(current):
        current_e, current_i = np.split(np.asarray(current, float), 2)
        mu_e = (syn[0] - z * syn[1]
                + model.tau_mem_e_ms * model.j_ext_e_mv
                * model.nu_ext_per_ms)
        mu_i = (syn[2] - syn[3]
                + model.tau_mem_i_ms * model.j_ext_i_mv
                * model.nu_ext_per_ms)
        sigma_e = np.sqrt(model.tau_mem_e_ms * (
            model.v_ee @ current_e + z ** 2 * (model.v_ei @ current_i)
            + model.j_ext_e_mv ** 2 * model.nu_ext_per_ms))
        sigma_i = np.sqrt(model.tau_mem_i_ms * (
            model.v_ie @ current_e + model.v_ii @ current_i
            + model.j_ext_i_mv ** 2 * model.nu_ext_per_ms))
        phi_e, phi_i = transfer_rates(
            model, mu_e, sigma_e, mu_i, sigma_i)
        return np.r_[
            current_e + 0.1 * (-current_e + phi_e)
            / model.tau_mem_e_ms,
            current_i + 0.1 * (-current_i + phi_i)
            / model.tau_mem_i_ms,
        ]

    numerical = np.empty((2, 2), float)
    h = 1e-6
    for column in range(2):
        step = np.zeros(2)
        step[column] = h
        numerical[:, column] = (
            rate_step(solution.rates + step)
            - rate_step(solution.rates - step)) / (2.0 * h)
    np.testing.assert_allclose(matrix[:2, :2], numerical,
                               rtol=3e-4, atol=3e-6)


def test_nonlinear_delayed_assay_preserves_a_fixed_point_without_ou():
    model = homogeneous_one_cell_model(ratio=0.6)
    z_map = one_cell_surround_map()
    solution = solve_spatial_z_fixed_point(
        model, z_map, parameter=0.10,
        initial_rates=np.asarray([0.001, 0.004]), surround_weight=1.0)
    operators = CoarseDelayOperators(
        dt_ms=0.1, max_delay_steps=1,
        w_ee_history=sparse.csr_matrix(model.w_ee),
        w_ei_history=sparse.csr_matrix(model.w_ei),
        w_ie_history=sparse.csr_matrix(model.w_ie),
        w_ii_history=sparse.csr_matrix(model.w_ii))
    trajectory = simulate_delayed_ou_trajectory(
        model, operators, solution.rates,
        z_field=path_state(
            z_map, 0.10, surround_weight=1.0)[-1],
        ou_rate_e=np.zeros((100, 1)))
    np.testing.assert_allclose(
        trajectory["final_rates"], solution.rates, rtol=1e-7, atol=1e-10)
    assert np.max(trajectory["rms_rate_deviation_from_initial_hz"]) < 1e-7


def test_dynamic_m_delayed_assay_preserves_its_adapted_fixed_point():
    model = homogeneous_one_cell_model(ratio=0.6)
    z_map = one_cell_surround_map()
    eta_m = 0.004
    tau_m_ms = 500.0
    solution = solve_spatial_z_fixed_point(
        model, z_map, parameter=0.10,
        initial_rates=np.asarray([0.001, 0.004]), surround_weight=1.0,
        eta_m=eta_m, tau_m_slow_ms=tau_m_ms)
    assert solution.converged and solution.physical
    operators = CoarseDelayOperators(
        dt_ms=0.1, max_delay_steps=1,
        w_ee_history=sparse.csr_matrix(model.w_ee),
        w_ei_history=sparse.csr_matrix(model.w_ei),
        w_ie_history=sparse.csr_matrix(model.w_ie),
        w_ii_history=sparse.csr_matrix(model.w_ii))
    trajectory = simulate_delayed_ou_trajectory(
        model, operators, solution.rates,
        z_field=path_state(z_map, 0.10, surround_weight=1.0)[-1],
        ou_rate_e=np.zeros((100, 1)), eta_m=eta_m,
        tau_m_slow_ms=tau_m_ms)
    np.testing.assert_allclose(
        trajectory["final_rates"], solution.rates, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(
        trajectory["final_adaptation_state"],
        tau_m_ms * solution.rate_e, rtol=1e-7, atol=1e-10)
    assert np.max(trajectory["rms_rate_deviation_from_initial_hz"]) < 1e-7


def test_delay_tangent_contains_dynamic_m_feedback_blocks():
    model = homogeneous_one_cell_model(ratio=0.6)
    z_map = one_cell_surround_map()
    eta_m = 0.004
    tau_m_ms = 500.0
    solution = solve_spatial_z_fixed_point(
        model, z_map, parameter=0.10,
        initial_rates=np.asarray([0.001, 0.004]), surround_weight=1.0,
        eta_m=eta_m, tau_m_slow_ms=tau_m_ms)
    zero = sparse.csr_matrix((1, 1))
    operators = CoarseDelayOperators(
        dt_ms=0.1, max_delay_steps=1,
        w_ee_history=zero, w_ei_history=zero,
        w_ie_history=zero, w_ii_history=zero)
    z = path_state(z_map, 0.10, surround_weight=1.0)[-1]
    matrix = delayed_step_matrix(
        model, operators, solution.rates, z_field=z,
        eta_m=eta_m, tau_m_slow_ms=tau_m_ms).toarray()
    # Block order is rE,rI,sEE,sEI,sIE,sII,hE,hI,M for the one-cell model.
    assert matrix.shape == (9, 9)
    assert matrix[8, 0] == pytest.approx(0.1)
    assert matrix[8, 8] == pytest.approx(1.0 - 0.1 / tau_m_ms)
    assert matrix[0, 8] < 0.0


def test_delayed_trajectory_can_resume_without_resetting_hidden_state():
    model = homogeneous_one_cell_model(ratio=0.6)
    z_map = one_cell_surround_map()
    operators = CoarseDelayOperators(
        dt_ms=0.1, max_delay_steps=2,
        w_ee_history=sparse.hstack([
            sparse.csr_matrix(model.w_ee), sparse.csr_matrix((1, 1))]),
        w_ei_history=sparse.hstack([
            sparse.csr_matrix(model.w_ei), sparse.csr_matrix((1, 1))]),
        w_ie_history=sparse.hstack([
            sparse.csr_matrix(model.w_ie), sparse.csr_matrix((1, 1))]),
        w_ii_history=sparse.hstack([
            sparse.csr_matrix(model.w_ii), sparse.csr_matrix((1, 1))]))
    rates = np.asarray([0.01, 0.02])
    z = path_state(z_map, 0.10, surround_weight=1.0)[-1]
    whole = simulate_delayed_ou_trajectory(
        model, operators, rates, z_field=z,
        ou_rate_e=np.zeros((200, 1)), eta_m=0.004,
        tau_m_slow_ms=500.0)
    first = simulate_delayed_ou_trajectory(
        model, operators, rates, z_field=z,
        ou_rate_e=np.zeros((100, 1)), eta_m=0.004,
        tau_m_slow_ms=500.0)
    second = simulate_delayed_ou_trajectory(
        model, operators, first["final_rates"], z_field=z,
        ou_rate_e=np.zeros((100, 1)), eta_m=0.004,
        tau_m_slow_ms=500.0,
        initial_m=first["final_adaptation_state"],
        initial_synapses=first["final_synapses"],
        initial_history_e=first["final_history_e"],
        initial_history_i=first["final_history_i"])
    np.testing.assert_allclose(second["final_rates"], whole["final_rates"],
                               rtol=0, atol=1e-13)
    np.testing.assert_allclose(
        second["final_adaptation_state"], whole["final_adaptation_state"],
        rtol=0, atol=1e-13)
