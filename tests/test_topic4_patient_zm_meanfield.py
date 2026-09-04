import numpy as np
import pytest
from scipy import sparse

from src.sef_hfo_lif import _ms, lif_rate, nu_theta_pop
from src.topic4_patient_zm_meanfield import (
    _aggregate_pathway,
    fixed_point_jacobian,
    fixed_point_q_derivative,
    fixed_point_residual,
    grouped_threshold_support,
    homogeneous_one_cell_model,
    lif_rate_gauss_legendre,
    load_patient_coarse_model,
    moments,
    pseudo_arclength_continue,
    save_patient_coarse_model,
    solve_fixed_point,
    spatial_cell_index,
    transfer_rates,
)


def test_gauss_legendre_transfer_matches_canonical_siegert():
    probes = [(-5.0, 3.0, 18.0), (10.0, 7.0, 17.2),
              (20.0, 5.0, 18.0), (70.0, 15.0, 19.0)]
    for mu, sigma, threshold in probes:
        observed = lif_rate_gauss_legendre(
            mu, sigma, tau_mem_ms=20.0, tau_ref_ms=2.0,
            v_threshold_mv=threshold, quadrature_order=16)
        expected = lif_rate(
            mu, sigma, 20.0, 2.0, v_th=threshold)
        assert float(observed) == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_homogeneous_fixture_reduces_to_canonical_moments_and_transfer():
    model = homogeneous_one_cell_model(ratio=1.0)
    rate_e, rate_i = 0.003, 0.007
    observed = moments(
        model, np.asarray([rate_e]), np.asarray([rate_i]), q=1.0)
    expected = _ms(rate_e, rate_i, nu_theta_pop())
    # ``moments`` follows transfer-function argument order
    # (mu_E, sigma_E, mu_I, sigma_I); the legacy helper returns
    # (mu_E, mu_I, sigma_E, sigma_I).
    assert [float(value[0]) for value in observed] == pytest.approx(
        [expected[0], expected[2], expected[1], expected[3]])
    phi_e, phi_i = transfer_rates(model, *observed)
    assert phi_e[0] == pytest.approx(
        lif_rate(expected[0], expected[2], 20.0, 2.0), rel=1e-9)
    assert phi_i[0] == pytest.approx(
        lif_rate(expected[1], expected[3], 10.0, 1.0), rel=1e-9)


def test_grouped_threshold_support_preserves_cell_means():
    values = np.asarray([15.0, 17.0, 18.0, 18.0, 19.0, 20.0])
    cells = np.asarray([0, 0, 0, 1, 1, 1])
    nodes, weights = grouped_threshold_support(
        values, cells, n_groups=2, n_cells=2)
    reconstructed = np.sum(nodes * weights, axis=1)
    assert reconstructed == pytest.approx([values[:3].mean(), values[3:].mean()])


def test_spatial_index_is_row_major_and_clips_right_boundary():
    positions = np.asarray([[0.0, 0.0], [0.75, 0.25], [1.0, 1.0]])
    assert spatial_cell_index(
        positions, n_grid=2, sheet_l_mm=1.0).tolist() == [0, 1, 3]


def test_pathway_aggregation_restores_physical_weights_and_orientation():
    # Jump matrix rows are targets, columns are sources.  A target-specific
    # rise conversion of 0.5 turns [2,4,6] into physical [1,2,3].
    matrix = sparse.csc_matrix((
        np.asarray([2.0, 4.0, 6.0]),
        (np.asarray([0, 1, 2]), np.asarray([0, 1, 0]))), shape=(3, 2))
    target_cells = np.asarray([0, 0, 1])
    source_cells = np.asarray([0, 1])
    mean, variance = _aggregate_pathway(
        [matrix], target_cells=target_cells, source_cells=source_cells,
        target_mask=lambda rows: rows < 2, n_cells=2,
        target_counts=np.asarray([2, 1]), physical_factor=0.5)
    np.testing.assert_allclose(mean, [[0.5, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(variance, [[0.5, 2.0], [0.0, 0.0]])


def test_fixed_point_jacobian_matches_finite_difference():
    model = homogeneous_one_cell_model(ratio=0.6)
    rates = np.asarray([0.06, 0.08])
    analytic = fixed_point_jacobian(model, rates, q=0.82, eta_m=0.02)
    numerical = np.empty_like(analytic)
    h = 1e-6
    for column in range(2):
        step = np.zeros(2); step[column] = h
        numerical[:, column] = (
            fixed_point_residual(model, rates + step, q=0.82, eta_m=0.02)
            - fixed_point_residual(model, rates - step, q=0.82, eta_m=0.02)
        ) / (2.0 * h)
    assert analytic == pytest.approx(numerical, rel=2e-4, abs=2e-5)


def test_fixed_point_q_derivative_matches_finite_difference():
    model = homogeneous_one_cell_model(ratio=0.6)
    rates = np.asarray([0.06, 0.08])
    analytic = fixed_point_q_derivative(
        model, rates, q=0.82, eta_m=0.02)
    h = 1e-6
    numerical = (
        fixed_point_residual(model, rates, q=0.82 + h, eta_m=0.02)
        - fixed_point_residual(model, rates, q=0.82 - h, eta_m=0.02)
    ) / (2.0 * h)
    assert analytic == pytest.approx(numerical, rel=2e-4, abs=2e-5)


def test_solver_and_numeric_archive_round_trip(tmp_path):
    model = homogeneous_one_cell_model(ratio=0.6)
    record = save_patient_coarse_model(tmp_path / "model.npz", model)
    assert len(record["sha256"]) == 64
    restored = load_patient_coarse_model(record["path"])
    assert restored.w_ee == pytest.approx(model.w_ee)
    solution = solve_fixed_point(
        restored, q=1.0, initial_rates=np.asarray([0.005, 0.008]))
    assert solution.converged
    assert solution.physical
    assert solution.residual_inf < 1e-9


def test_pseudo_arclength_crosses_known_homogeneous_fold():
    model = homogeneous_one_cell_model(ratio=0.6)
    first = solve_fixed_point(
        model, q=0.84, initial_rates=np.asarray([0.30, 0.32]))
    second = solve_fixed_point(
        model, q=0.85, initial_rates=first.rates)
    points = pseudo_arclength_continue(
        model, first, second, step_size=0.01, n_steps=24)
    valid = [point for point in points if point.solution.converged]
    tangents = np.asarray([point.tangent_q for point in valid])
    q_values = np.asarray([point.solution.q for point in valid])
    assert np.any(tangents > 0.0)
    assert np.any(tangents < 0.0)
    assert 0.885 < q_values.max() < 0.889
