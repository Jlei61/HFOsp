import numpy as np
import pytest

from scripts.run_topic4_mz_spatial_autonomous_latch import _classify
from src.topic4_mz_spatial_autonomous_latch import (
    Pulse,
    RegionalSlowParameters,
    pulse_drive,
    regional_slow_rhs,
    smooth_gate,
    update_regional_latch,
)
from src.topic4_mz_spatial_patch import (
    LOCAL_FIELDS,
    PatchKernels,
    PatchParameters,
    pack_patch_state,
    prepare_patch_rhs,
    uniform_patch_state,
)
from src.topic4_spatial_slowfast_stage0c import equilibrium_state


def _state(s_ei, r_fast, z=0.88, persistence=(0.2, 0.2, 0.0), m=0.05):
    local = {name: np.zeros(3) for name in LOCAL_FIELDS}
    local["sEI"] = np.asarray(s_ei, dtype=float)
    local["rE_fast"] = np.asarray(r_fast, dtype=float)
    local["z"].fill(z)
    local["p"] = np.asarray(persistence, dtype=float)
    local["m"] = np.asarray([m, m, 0.0])
    return pack_patch_state(local, mu_g=0.0, s_g=0.0)


def test_compact_slow_gate_has_exact_zero_and_one_arms():
    observed = smooth_gate(np.asarray([0.0, 0.5, 1.0, 1.5]), 0.5, 0.5)
    np.testing.assert_array_equal(observed[[0, 1]], 0.0)
    np.testing.assert_array_equal(observed[[2, 3]], 1.0)


def test_regional_slow_rhs_masks_bath_z_and_uses_hybrid_latch_for_m():
    arm = RegionalSlowParameters(
        z_rest=0.9, inhibitory_use_threshold_khz=0.004,
        inhibitory_use_width_khz=0.004, occupancy_threshold_khz=0.020,
        occupancy_width_khz=0.010,
    )
    baseline = np.asarray([0.006, 0.006, 0.006])
    joint = _state([0.020, 0.020, 0.006], [0.040, 0.040, 0.001])
    focal = _state([0.020, 0.020, 0.006], [0.040, 0.001, 0.001])
    rhs, sensors = regional_slow_rhs(
        np.asarray([joint, focal]), [arm, arm], inhibitory_baseline_khz=baseline,
        recruitment_kernel=np.eye(3), patch_weights=np.asarray([0.25, 0.25, 0.5]),
        latch_state=np.asarray([[True, True, False], [False, False, False]]),
    )
    assert np.all(rhs[0, 21:23] < 0.0)
    assert rhs[0, 21] == pytest.approx(rhs[0, 22])
    assert rhs[0, 23] > 0.0
    assert sensors["z_use"][0, 2] == 0.0
    assert rhs[0, 27] > 0.0 and rhs[0, 28] > 0.0
    assert rhs[0, 27] == pytest.approx(rhs[0, 28])
    assert rhs[1, 27] < 0.0 and rhs[1, 28] < 0.0
    np.testing.assert_array_equal(sensors["occupancy"][0], [1.0, 1.0, 0.0])
    assert rhs[0, 29] == 0.0


def test_primary_resource_pool_keeps_core_annulus_z_on_registered_coordinate():
    arm = RegionalSlowParameters(z_rest=0.9)
    state = _state([0.020, 0.012, 0.020], [0.001, 0.001, 0.001])
    rhs, sensors = regional_slow_rhs(
        state[None, :], [arm], inhibitory_baseline_khz=np.full(3, 0.006),
        recruitment_kernel=np.eye(3), patch_weights=np.asarray([0.3, 0.2, 0.5]),
        latch_state=np.zeros((1, 3), dtype=bool),
    )
    assert sensors["z_use"][0, 0] == pytest.approx(sensors["z_use"][0, 1])
    assert sensors["z_use"][0, 2] == 0.0
    assert rhs[0, 21] == pytest.approx(rhs[0, 22])


def test_default_resource_rhs_preserves_the_original_m_independent_equation():
    arm = RegionalSlowParameters(
        z_rest=0.9, tau_z_recovery_ms=20000.0,
        tau_z_depletion_ms=4000.0,
    )
    state = _state([0.020, 0.012, 0.006], [0.001, 0.001, 0.001], z=0.88, m=0.75)
    rhs, sensors = regional_slow_rhs(
        state[None, :], [arm], inhibitory_baseline_khz=np.full(3, 0.006),
        recruitment_kernel=np.eye(3), patch_weights=np.asarray([0.3, 0.2, 0.5]),
        latch_state=np.zeros((1, 3), dtype=bool),
    )
    expected = (
        (arm.z_rest - 0.88) / arm.tau_z_recovery_ms
        - sensors["z_use"][0] * 0.88 / arm.tau_z_depletion_ms
    )
    np.testing.assert_array_equal(rhs[0, 21:24], expected)


def test_m_gated_reserve_uses_pooled_dimensionless_m_and_keeps_bath_fixed():
    arm = RegionalSlowParameters(
        z_rest=0.9, tau_z_recovery_ms=90000.0,
        tau_z_fast_recovery_ms=20000.0, tau_z_depletion_ms=280.0,
        q_reserve=0.8415, enable_m_gated_z_recovery=True,
    )
    state = _state([0.020, 0.012, 0.020], [0.001, 0.001, 0.001], z=0.88, m=0.4)
    rhs, sensors = regional_slow_rhs(
        state[None, :], [arm], inhibitory_baseline_khz=np.full(3, 0.006),
        recruitment_kernel=np.eye(3), patch_weights=np.asarray([0.3, 0.2, 0.5]),
        latch_state=np.zeros((1, 3), dtype=bool),
    )
    recovery_rate = 0.6 / 90000.0 + 0.4 / 20000.0
    expected = (
        recovery_rate * (0.9 - 0.88)
        - sensors["z_use"][0, 0] * (0.88 - 0.8415) / 280.0
    )
    assert rhs[0, 21] == pytest.approx(expected)
    assert rhs[0, 22] == pytest.approx(expected)
    assert rhs[0, 23] == 0.0


def test_m_gated_reserve_parameters_fail_closed():
    with pytest.raises(ValueError, match="paired"):
        RegionalSlowParameters(enable_m_gated_z_recovery=True).validate()
    with pytest.raises(ValueError, match="q_reserve"):
        RegionalSlowParameters(
            q_reserve=0.9, tau_z_fast_recovery_ms=20000.0,
            enable_m_gated_z_recovery=True,
        ).validate()
    with pytest.raises(ValueError, match="tau_fast"):
        RegionalSlowParameters(
            tau_z_recovery_ms=20000.0, q_reserve=0.84,
            tau_z_fast_recovery_ms=20000.0, enable_m_gated_z_recovery=True,
        ).validate()
    with pytest.raises(ValueError, match="require"):
        RegionalSlowParameters(q_reserve=0.84).validate()

    arm = RegionalSlowParameters(
        tau_z_recovery_ms=90000.0, q_reserve=0.84,
        tau_z_fast_recovery_ms=20000.0, enable_m_gated_z_recovery=True,
    )
    invalid = _state([0.006] * 3, [0.001] * 3)
    invalid[27:30] = [1.2, 0.0, 0.0]
    with pytest.raises(ValueError, match="dimensionless m"):
        regional_slow_rhs(
            invalid[None, :], [arm], inhibitory_baseline_khz=np.full(3, 0.006),
            recruitment_kernel=np.eye(3), patch_weights=np.asarray([0.3, 0.2, 0.5]),
            latch_state=np.zeros((1, 3), dtype=bool),
        )


def test_latch_requires_persistence_and_neighbor_recruitment_then_safe_resets():
    arm = RegionalSlowParameters(persistence_on=0.15, recruitment_on=0.60)
    kernel = np.asarray([
        [0.80, 0.20, 0.00],
        [0.20, 0.60, 0.20],
        [0.00, 0.01, 0.99],
    ])
    baseline = np.full(3, 0.006)
    joint = _state([0.006] * 3, [0.040, 0.040, 0.001])
    focal = _state([0.006] * 3, [0.040, 0.001, 0.001])
    batch = np.asarray([joint, focal])
    empty = np.zeros((2, 3), dtype=bool)
    _, sensors = regional_slow_rhs(
        batch, [arm, arm], inhibitory_baseline_khz=baseline,
        recruitment_kernel=kernel, patch_weights=np.asarray([0.25, 0.25, 0.5]),
        latch_state=empty,
    )
    latch, set_now, reset_now = update_regional_latch(batch, [arm, arm], sensors, empty)
    np.testing.assert_array_equal(set_now, [True, False])
    np.testing.assert_array_equal(reset_now, False)
    np.testing.assert_array_equal(latch[0], [True, True, False])
    np.testing.assert_array_equal(latch[1], False)

    recovered = _state(
        [0.006] * 3, [0.001, 0.001, 0.001], z=0.89,
        persistence=(0.01, 0.01, 0.0),
    )
    active = np.asarray([[True, True, False]])
    _, recovered_sensors = regional_slow_rhs(
        recovered[None, :], [arm], inhibitory_baseline_khz=baseline,
        recruitment_kernel=kernel, patch_weights=np.asarray([0.25, 0.25, 0.5]),
        latch_state=active,
    )
    latch, set_now, reset_now = update_regional_latch(
        recovered[None, :], [arm], recovered_sensors, active
    )
    assert not set_now[0] and reset_now[0]
    np.testing.assert_array_equal(latch, False)


def test_pulse_schedule_is_patch_local_and_half_open():
    pulses = [Pulse(10.0, 5.0, 2.0, (1.0, 0.25, 0.0))]
    np.testing.assert_array_equal(pulse_drive(9.99, pulses, 2), 0.0)
    expected = np.asarray([[2.0, 0.5, 0.0], [2.0, 0.5, 0.0]])
    np.testing.assert_array_equal(pulse_drive(10.0, pulses, 2), expected)
    np.testing.assert_array_equal(pulse_drive(14.99, pulses, 2), expected)
    np.testing.assert_array_equal(pulse_drive(15.0, pulses, 2), 0.0)


def test_pulse_rejects_profiles_that_can_broadcast_across_space():
    with pytest.raises(ValueError, match="exactly three"):
        Pulse(10.0, 5.0, 2.0, (1.0,)).validate()
    with pytest.raises(ValueError, match="exactly three"):
        Pulse(10.0, 5.0, 2.0, (1.0, 0.0, 0.0, 0.0)).validate()


def test_four_return_low_tail_is_pending_recovery_not_complete_lifecycle():
    time_ms = np.asarray([0.0, 16000.0, 17000.0])
    trace = np.full((3, 1, 3), 0.001, dtype=float)
    z = np.full((3, 1, 3), 0.84, dtype=float)
    result = {
        "time_ms": time_ms,
        "rE": trace,
        "z": z,
        "p": np.zeros_like(trace),
        "m": np.zeros_like(trace),
        "return_times_ms": [[[11100.0, 12000.0, 13000.0, 14000.0],
                             [11110.0, 12010.0, 13010.0, 14010.0], []]],
        "first_support_failure_ms": np.asarray([np.nan]),
        "first_bound_failure_ms": np.asarray([np.nan]),
        "first_nonfinite_ms": np.asarray([np.nan]),
        "latch_set_times_ms": [[12500.0]],
        "latch_reset_times_ms": [[16500.0]],
        "active_at_end": np.asarray([True]),
        "finite": np.asarray([True]),
        "support_violation_count": np.zeros((1, 3), dtype=int),
        "state_bound_violation_count": np.zeros((1, 3), dtype=int),
    }
    cfg = {
        "background_event_challenge": {"pulse_free_analysis_start_ms": 11035.0},
        "known_fast_boundaries": {"regional_entry_fold_z": 0.8558315843},
        "classification": {
            "low_tail_start_ms": 16000.0,
            "low_tail_max_hz": 5.0,
            "minimum_pulse_free_returns": 4,
        },
        "slow_common": {"z_rest": 0.90},
    }
    row = _classify(result, 0, "synthetic", 0.125, cfg)
    assert row["outcome"] == "finite_low_tail_after_four_returns_pending_recovery"
    assert "complete" not in row["outcome"]


def test_integrator_refuses_trace_allocation_above_budget():
    params = PatchParameters(additive_max_mv=1.6)
    prepared = prepare_patch_rhs(PatchKernels.identity(3), params)
    initial = uniform_patch_state(
        equilibrium_state((0.001, 0.006)), n_patches=3,
        z=0.9, additive_mv=0.0, parameters=params,
    )
    with pytest.raises(MemoryError, match="above max_trace_bytes"):
        from src.topic4_mz_spatial_autonomous_latch import integrate_autonomous_latch_batch

        integrate_autonomous_latch_batch(
            initial[None, :], prepared, object(), [RegionalSlowParameters()], [],
            inhibitory_baseline_khz=np.full(3, 0.006), dt_ms=0.125,
            duration_ms=1000.0, save_dt_ms=1.0, max_trace_bytes=1,
        )


def test_integrator_fail_closes_an_unsupported_fork_without_nan_propagation():
    from src.topic4_mz_spatial_autonomous_latch import integrate_autonomous_latch_batch

    class UnsupportedTransfer:
        @staticmethod
        def rate(mu, sigma, pop):
            return np.zeros_like(np.asarray(mu, dtype=float))

        @staticmethod
        def support_mask(mu, sigma):
            return np.zeros_like(np.asarray(mu, dtype=float), dtype=bool)

    params = PatchParameters(additive_max_mv=1.6)
    prepared = prepare_patch_rhs(PatchKernels.identity(3), params)
    initial = uniform_patch_state(
        equilibrium_state((0.001, 0.006)), n_patches=3,
        z=0.9, additive_mv=0.0, parameters=params,
    )
    result = integrate_autonomous_latch_batch(
        initial[None, :], prepared, UnsupportedTransfer(), [RegionalSlowParameters()], [],
        inhibitory_baseline_khz=initial[9:12], dt_ms=0.125,
        duration_ms=1.0, save_dt_ms=0.125,
    )
    assert not result["active_at_end"][0]
    assert result["first_support_failure_ms"][0] == 0.0
    assert np.isnan(result["first_nonfinite_ms"][0])
    assert np.all(np.isfinite(result["rE"]))


def test_integrator_accepts_only_regional_initial_latch_state():
    from src.topic4_mz_spatial_autonomous_latch import integrate_autonomous_latch_batch

    class UnsupportedTransfer:
        @staticmethod
        def rate(mu, sigma, pop):
            return np.zeros_like(np.asarray(mu, dtype=float))

        @staticmethod
        def support_mask(mu, sigma):
            return np.zeros_like(np.asarray(mu, dtype=float), dtype=bool)

    params = PatchParameters(additive_max_mv=1.6)
    prepared = prepare_patch_rhs(PatchKernels.identity(3), params)
    initial = uniform_patch_state(
        equilibrium_state((0.001, 0.006)), n_patches=3,
        z=0.84, additive_mv=0.0, parameters=params,
    )
    with pytest.raises(ValueError, match="fork-by-patch"):
        integrate_autonomous_latch_batch(
            initial[None, :], prepared, UnsupportedTransfer(),
            [RegionalSlowParameters(enable_z=False)], [],
            inhibitory_baseline_khz=initial[9:12], dt_ms=0.125,
            duration_ms=1.0, save_dt_ms=0.125,
            initial_latch_state=np.asarray([[True, True]]),
        )
    result = integrate_autonomous_latch_batch(
        initial[None, :], prepared, UnsupportedTransfer(),
        [RegionalSlowParameters(enable_z=False)], [],
        inhibitory_baseline_khz=initial[9:12], dt_ms=0.125,
        duration_ms=1.0, save_dt_ms=0.125,
        initial_latch_state=np.asarray([[True, True, False]]),
    )
    np.testing.assert_array_equal(result["latch"][0, 0], [1, 1, 0])
    np.testing.assert_array_equal(result["final_latch_state"], [[True, True, False]])
