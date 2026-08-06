import numpy as np
import pytest

from src.topic4_mz_persistence_feasibility import (
    causal_sustained_onset_ms,
    classify_leverage_race,
    compact_smoothstep,
    integrate_bounded_effector,
    integrate_lowpass,
    required_additive_from_fold,
    unopposed_z,
)


def test_causal_sustained_onset_uses_trailing_history_and_component_start():
    rate = np.r_[np.zeros(10), np.full(20, 4.0), np.zeros(5)]
    onset, envelope = causal_sustained_onset_ms(
        rate,
        dt_ms=1.0,
        envelope_ms=2.0,
        threshold_hz=3.0,
        minimum_duration_ms=5.0,
    )
    assert onset == 11.0
    assert envelope[10] == 2.0
    assert envelope[11] == 4.0


def test_causal_sustained_onset_rejects_short_excursion():
    with pytest.raises(RuntimeError, match="no qualifying"):
        causal_sustained_onset_ms(
            np.r_[np.zeros(5), np.ones(3), np.zeros(5)],
            dt_ms=1.0,
            envelope_ms=1.0,
            threshold_hz=0.5,
            minimum_duration_ms=4.0,
        )


def test_compact_smoothstep_has_exact_dead_zone_and_saturation():
    observed = compact_smoothstep(
        np.array([-1.0, 0.0, 0.5, 1.0, 2.0]), low=0.0, high=1.0
    )
    np.testing.assert_array_equal(observed, [0.0, 0.0, 0.5, 1.0, 1.0])


def test_integrate_lowpass_step_response_starts_at_causal_boundary():
    observed = integrate_lowpass([0.0, 0.0, 1.0, 1.0], dt_ms=1.0, tau_ms=2.0)
    np.testing.assert_array_equal(observed, [0.0, 0.0, 0.5, 0.75])


def test_bounded_effector_stays_zero_below_gate_then_accumulates_across_it():
    state, gate = integrate_bounded_effector(
        [-1.0, 0.0, 0.5, 1.0, 2.0],
        dt_ms=1.0,
        gate_low=0.0,
        gate_high=1.0,
        tau_up_ms=4.0,
        tau_down_ms=5.0,
        unsafe_decay_fraction=0.5,
    )
    np.testing.assert_array_equal(gate, [0.0, 0.0, 0.5, 1.0, 1.0])
    np.testing.assert_array_equal(state, [0.0, 0.0, 0.125, 0.34375, 0.5078125])


def test_latched_effector_keeps_building_after_first_activation():
    memoryless, memoryless_gate = integrate_bounded_effector(
        [0.0, 1.0, 0.0, 0.0], dt_ms=1.0, gate_low=0.5, gate_high=0.5,
        tau_up_ms=4.0, tau_down_ms=10.0, unsafe_decay_fraction=0.0,
    )
    latched, latched_gate = integrate_bounded_effector(
        [0.0, 1.0, 0.0, 0.0], dt_ms=1.0, gate_low=0.5, gate_high=0.5,
        tau_up_ms=4.0, tau_down_ms=10.0, unsafe_decay_fraction=0.0,
        latch_after_first_activation=True,
    )
    np.testing.assert_array_equal(memoryless_gate, [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(latched_gate, [0.0, 1.0, 1.0, 1.0])
    assert latched[-1] > memoryless[-1]


def test_unopposed_z_depletes_monotonically_toward_equilibrium():
    observed = unopposed_z(
        [0.0, 10.0, 20.0, 40.0],
        z_start=1.0,
        depletion_occupancy=0.4,
        tau_z_ms=20.0,
    )
    assert observed[0] == 1.0
    assert np.all(np.diff(observed) < 0.0)
    assert np.all(observed > 0.6)


def test_required_additive_interpolates_monotonically_on_unsorted_surface():
    observed = required_additive_from_fold(
        [0.0, 0.5, 1.0, 1.5, 2.0],
        fold_z=[1.0, 0.0, 2.0],
        fold_additive_mv=[2.0, 0.0, 4.0],
    )
    np.testing.assert_array_equal(observed, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.all(np.diff(observed) >= 0.0)


@pytest.mark.parametrize("requested_z", ([-0.01], [2.01]))
def test_required_additive_rejects_fold_extrapolation(requested_z):
    with pytest.raises(ValueError, match="outside the locked fold surface"):
        required_additive_from_fold(
            requested_z,
            fold_z=[0.0, 1.0, 2.0],
            fold_additive_mv=[0.0, 2.0, 4.0],
        )


@pytest.mark.parametrize(
    ("available", "expected_status", "expected_crossing_ms"),
    [
        (
            [0.0, 0.5, 1.0, 1.1, 1.2, 1.3],
            "too_early_or_prevention_risk",
            20.0,
        ),
        (
            [0.0, 0.5, 0.9, 1.0, 1.1, 1.2],
            "timing_leverage_feasible",
            30.0,
        ),
        (
            [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
            "insufficient_leverage",
            None,
        ),
    ],
)
def test_classify_leverage_race_three_registered_outcomes(
    available, expected_status, expected_crossing_ms
):
    result = classify_leverage_race(
        [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        available,
        np.ones(6),
        minimum_cycles=3.0,
        maximum_cycles=5.0,
        cycle_period_ms=10.0,
    )

    assert result["status"] == expected_status
    assert result["first_crossing_ms"] == expected_crossing_ms
    assert result["first_crossing_cycles"] == (
        None if expected_crossing_ms is None else expected_crossing_ms / 10.0
    )
    assert result["registered_window_ms"] == [30.0, 50.0]
