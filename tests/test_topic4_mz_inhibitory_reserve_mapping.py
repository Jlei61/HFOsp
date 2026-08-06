import numpy as np
import pytest

from scripts.run_topic4_mz_inhibitory_reserve_mapping import (
    integrate_event_q,
    period_average,
)


def test_event_q_replay_stays_at_rest_without_use():
    time = np.arange(0.0, 101.0, 1.0)
    q = integrate_event_q(
        time, np.zeros_like(time), q_rest=0.9, q_reserve=0.83,
        tau_recovery_ms=20000.0, tau_depletion_ms=500.0,
    )
    np.testing.assert_array_equal(q, 0.9)


def test_event_q_replay_depletes_toward_reserve_and_recovers():
    time = np.arange(0.0, 1001.0, 1.0)
    use = np.zeros_like(time)
    use[100:200] = 1.0
    q = integrate_event_q(
        time, use, q_rest=0.9, q_reserve=0.83,
        tau_recovery_ms=20000.0, tau_depletion_ms=500.0,
    )
    assert q[200] < q[100]
    assert q[-1] > q[200]
    assert np.min(q) >= 0.83


def test_event_q_replay_rejects_misaligned_or_negative_sensor():
    with pytest.raises(ValueError, match="aligned"):
        integrate_event_q(
            np.arange(3.0), np.arange(2.0), q_rest=0.9, q_reserve=0.83,
            tau_recovery_ms=20000.0, tau_depletion_ms=500.0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        integrate_event_q(
            np.arange(3.0), np.asarray([0.0, -1.0, 0.0]),
            q_rest=0.9, q_reserve=0.83,
            tau_recovery_ms=20000.0, tau_depletion_ms=500.0,
        )


def test_period_average_interpolates_exact_return_endpoints():
    time = np.arange(0.0, 11.0, 1.0)
    values = 2.0 * time + 1.0
    mean, dose = period_average(time, values, 1.25, 8.75)
    assert mean == pytest.approx(11.0)
    assert dose == pytest.approx(82.5)


def test_period_average_rejects_endpoints_outside_saved_trace():
    with pytest.raises(ValueError, match="inside"):
        period_average(np.arange(3.0), np.ones(3), -1.0, 2.0)
