import numpy as np
import pytest

from src.topic4_patient_zm_delay import (
    PatientCoarseDelayOperator,
    delayed_discrete_linear_map,
    load_patient_coarse_delay_operator,
    pathway_variance_matrix,
    pathway_weight_matrix,
    save_patient_coarse_delay_operator,
    stationary_delay_mode_vector,
)
from src.topic4_patient_zm_meanfield import (
    dynamic_jacobian,
    homogeneous_one_cell_model,
    solve_fixed_point,
)


def _one_cell_operator(model, delay_dt_ms=0.001):
    values = {}
    for name in ("ee", "ei", "ie", "ii"):
        values[f"delay_step_{name}"] = np.asarray([1], np.int32)
        values[f"target_{name}"] = np.asarray([0], np.int32)
        values[f"source_{name}"] = np.asarray([0], np.int32)
        values[f"weight_{name}"] = np.asarray(
            [float(getattr(model, f"w_{name}")[0, 0])])
        values[f"variance_weight_{name}"] = np.asarray(
            [float(getattr(model, f"v_{name}")[0, 0])])
    return PatientCoarseDelayOperator(
        n_grid=1, sheet_l_mm=1.0, source_delay_dt_ms=delay_dt_ms,
        max_delay_steps=1, **values)


def test_delay_archive_round_trip_and_pathway_conservation(tmp_path):
    model = homogeneous_one_cell_model(ratio=0.6)
    operator = _one_cell_operator(model)
    record = save_patient_coarse_delay_operator(
        tmp_path / "delay_operator.npz", operator)
    assert len(record["sha256"]) == 64
    restored = load_patient_coarse_delay_operator(record["path"])
    for name in ("ee", "ei", "ie", "ii"):
        observed = pathway_weight_matrix(restored, name).toarray()
        assert observed == pytest.approx(getattr(model, f"w_{name}"))
        observed_variance = pathway_variance_matrix(restored, name).toarray()
        assert observed_variance == pytest.approx(getattr(model, f"v_{name}"))


def test_explicit_history_map_converges_to_zero_delay_jacobian():
    model = homogeneous_one_cell_model(ratio=0.6)
    solution = solve_fixed_point(
        model, q=0.82, eta_m=0.02,
        initial_rates=np.asarray([0.30, 0.32]))
    assert solution.converged
    dt = 0.001
    operator = _one_cell_operator(model, delay_dt_ms=dt)
    delayed, metadata = delayed_discrete_linear_map(
        model, operator, solution.rates, q=0.82, eta_m=0.02,
        tau_m_slow_ms=12.5, history_dt_ms=dt,
        variance_closure="frozen_variance")
    multipliers = np.linalg.eigvals(delayed.toarray())
    exponents = np.log(multipliers.astype(complex)) / dt
    observed = max(exponents.real)
    expected = max(np.linalg.eigvals(dynamic_jacobian(
        model, solution.rates, q=0.82, eta_m=0.02,
        tau_m_slow_ms=12.5).toarray()).real)
    assert metadata["maximum_lag_steps"] == 1
    assert observed == pytest.approx(expected, abs=2e-4)


def test_self_consistent_variance_closure_has_fixed_point_zero_gain():
    """At multiplier one, the delay map reproduces the full FP Jacobian."""
    from src.topic4_patient_zm_meanfield import fixed_point_jacobian

    model = homogeneous_one_cell_model(ratio=0.6)
    solution = solve_fixed_point(
        model, q=0.82, eta_m=0.02,
        initial_rates=np.asarray([0.30, 0.32]))
    assert solution.converged
    dt = 0.01
    operator = _one_cell_operator(model, delay_dt_ms=dt)
    matrix, _metadata = delayed_discrete_linear_map(
        model, operator, solution.rates, q=0.82, eta_m=0.02,
        tau_m_slow_ms=12.5, history_dt_ms=dt,
        variance_closure="self_consistent_variance")
    # Eliminating every non-rate block at multiplier one gives the derivative
    # of r-Phi(r).  Compare its singularity invariant through determinants;
    # both vanish at exactly the same stationary fold.
    dense = matrix.toarray()
    rate = np.arange(2)
    auxiliary = np.arange(2, dense.shape[0])
    identity = np.eye(dense.shape[0])
    system = identity - dense
    reduced = (system[np.ix_(rate, rate)]
               - system[np.ix_(rate, auxiliary)]
               @ np.linalg.solve(system[np.ix_(auxiliary, auxiliary)],
                                 system[np.ix_(auxiliary, rate)]))
    fp = fixed_point_jacobian(
        model, solution.rates, q=0.82, eta_m=0.02,
        tau_m_slow_ms=12.5)
    rate_scaling = np.asarray([
        1.0 - np.exp(-dt / model.tau_mem_e_ms),
        1.0 - np.exp(-dt / model.tau_mem_i_ms),
    ])
    assert reduced / rate_scaling[:, None] == pytest.approx(
        fp, rel=3e-3, abs=3e-4)

    direction = np.asarray([0.7, -0.3])
    lifted = stationary_delay_mode_vector(
        model, operator, direction, tau_m_slow_ms=12.5,
        history_dt_ms=dt)
    map_residual = lifted - matrix @ lifted
    expected = rate_scaling * (fp @ direction)
    assert map_residual[:2] == pytest.approx(expected, rel=3e-3, abs=3e-4)
    assert map_residual[2:] == pytest.approx(0.0, abs=1e-12)
