import numpy as np
import pytest

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
