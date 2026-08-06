import numpy as np
import pytest

from src.topic4_mz_inhibitory_reserve import (
    InhibitoryReserveParameters,
    interval_passes_gate,
    reserve_floor_for_hold,
    safe_q_intervals,
)


def test_effective_q_and_rhs_preserve_the_registered_bounds():
    params = InhibitoryReserveParameters(
        q_rest=0.9, q_reserve=0.8, tau_recovery_ms=20000.0,
        tau_depletion_ms=1000.0,
    )
    np.testing.assert_allclose(params.effective_q([0.0, 0.5, 1.0]), [0.8, 0.85, 0.9])
    assert params.fraction_rhs(1.0, 1.0) < 0.0
    assert params.fraction_rhs(0.0, 0.0) > 0.0
    assert params.q_rhs(0.9, 1.0) < 0.0
    assert params.q_rhs(0.8, 0.0) > 0.0


def test_q_nullcline_is_above_floor_and_inverts_to_the_same_floor():
    params = InhibitoryReserveParameters(
        q_rest=0.9, q_reserve=0.8, tau_recovery_ms=20000.0,
        tau_depletion_ms=1000.0,
    )
    mean_use = 0.25
    hold = float(params.q_nullcline(mean_use))
    assert 0.8 < hold < 0.9
    recovered = reserve_floor_for_hold(
        hold, mean_use, q_rest=0.9, tau_recovery_ms=20000.0,
        tau_depletion_ms=1000.0,
    )
    assert float(recovered) == pytest.approx(0.8)


def test_invalid_floor_or_use_is_rejected_instead_of_clipped():
    with pytest.raises(ValueError, match="q_reserve"):
        InhibitoryReserveParameters(q_rest=0.9, q_reserve=0.9).validate()
    with pytest.raises(ValueError, match="strictly positive"):
        reserve_floor_for_hold(
            0.84, 0.0, q_rest=0.9, tau_recovery_ms=20000.0,
            tau_depletion_ms=1000.0,
        )


def test_safe_intervals_do_not_bridge_failed_or_unresolved_gaps():
    nodes = [0.855, 0.8525, 0.85, 0.8475, 0.845, 0.8425]
    accepted = [True, True, False, True, True, True]
    intervals = safe_q_intervals(nodes, accepted, maximum_spacing=0.0025)
    assert intervals == [[0.8425, 0.845, 0.8475], [0.8525, 0.855]]
    assert interval_passes_gate(intervals[0], minimum_width=0.005, minimum_nodes=3)
    assert not interval_passes_gate(intervals[1], minimum_width=0.005, minimum_nodes=3)
