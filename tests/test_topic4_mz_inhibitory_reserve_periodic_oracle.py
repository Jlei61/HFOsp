import numpy as np
import pytest

from scripts.run_topic4_mz_inhibitory_reserve_periodic_oracle import (
    _status,
    exact_constant_use_step,
    exact_periodic_hold,
    extract_piecewise_constant_window,
)


def test_exact_window_uses_partial_zoh_intervals_without_interpolation():
    duration, use, boundaries = extract_piecewise_constant_window(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([1.0, 2.0, 3.0]),
        0.25,
        1.50,
    )
    np.testing.assert_allclose(boundaries, [0.25, 1.0, 1.5])
    np.testing.assert_allclose(duration, [0.75, 0.50])
    np.testing.assert_allclose(use, [1.0, 2.0])


def test_constant_use_step_matches_closed_form_equilibrium_and_integral():
    q_rest = 0.9
    q_reserve = 0.8
    tau_recovery = 20000.0
    tau_depletion = 200.0
    use = 0.2
    duration = 17.0
    q_initial = 0.87
    q_final, integral, alpha, beta = exact_constant_use_step(
        q_initial,
        duration,
        use,
        q_rest=q_rest,
        q_reserve=q_reserve,
        tau_recovery_ms=tau_recovery,
        tau_depletion_ms=tau_depletion,
    )
    decay = 1.0 / tau_recovery + use / tau_depletion
    q_inf = (q_rest / tau_recovery + q_reserve * use / tau_depletion) / decay
    expected_alpha = np.exp(-decay * duration)
    expected_final = q_inf + (q_initial - q_inf) * expected_alpha
    expected_integral = q_inf * duration + (q_initial - q_inf) * (1.0 - expected_alpha) / decay
    assert q_final == pytest.approx(expected_final)
    assert integral == pytest.approx(expected_integral)
    assert alpha == pytest.approx(expected_alpha)
    assert alpha * q_initial + beta == pytest.approx(q_final)


def test_periodic_constant_use_converges_to_exact_fixed_hold_and_reports_return_rho():
    durations = np.full(8, 10.0)
    use = np.full(8, 0.2)
    q_rest = 0.9
    q_reserve = 0.8
    tau_recovery = 20000.0
    tau_depletion = 200.0
    result = exact_periodic_hold(
        durations,
        use,
        q_rest=q_rest,
        q_reserve=q_reserve,
        tau_recovery_ms=tau_recovery,
        tau_depletion_ms=tau_depletion,
        integrated_returns=8,
        initial_q=0.9,
        convergence_tolerance=1.0e-13,
        maximum_iterations=10000,
    )
    decay = 1.0 / tau_recovery + 0.2 / tau_depletion
    fixed = (q_rest / tau_recovery + q_reserve * 0.2 / tau_depletion) / decay
    assert result["q_min"] == pytest.approx(fixed, abs=1.0e-11)
    assert result["q_max"] == pytest.approx(fixed, abs=1.0e-11)
    assert result["q_mean"] == pytest.approx(fixed, abs=1.0e-11)
    assert result["per_period_rho"] == pytest.approx(np.exp(-decay * 10.0))
    assert result["window_closure_error"] <= 1.0e-12
    assert result["stroboscopic_convergence_error"] <= 1.0e-13


@pytest.mark.parametrize("passed", [True, False])
def test_status_always_preserves_entry_ordering_no_go(passed):
    status = _status(passed)
    assert "BUT_ENTRY_ORDERING_NO_GO_PERSISTS" in status
    assert ("SUPPORTED" in status) == passed
