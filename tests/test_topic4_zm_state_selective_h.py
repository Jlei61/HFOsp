from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))

from src.snn_engine.slow_field import (
    SpatialSlowField,
    SpatialSlowFieldConfig,
    local_rate_field_hz,
    recover_i2e_resource,
    zero_baseline_sigmoid,
)
from kick_probe import persistent_exc_membrane_step
from src.topic4_zm_checkpoint import capture_slow, restore_slow


def _slow(**updates):
    values = dict(
        use_qI=False,
        use_gK=False,
        use_SG=True,
        alpha_G=1.0,
        use_z=True,
        use_m=True,
        I_th_EI=1.0,
        eta_m=0.001,
        use_mode_H=True,
        rho_mode_H=2.0,
        tau_mode_H=250.0,
        theta_mode_H_hz=0.0,
        half_mode_H_hz=1.0,
        z_mode_base=1.0,
        z_mode_susceptible=0.5,
        zeta_mode_center=0.5,
        zeta_mode_slope=0.1,
        m_mode_half=45.0,
        m_mode_power=4.0,
    )
    values.update(updates)
    cfg = SpatialSlowFieldConfig(**values)
    posE = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]])
    posI = np.array([[1.5, 1.5], [2.5, 2.5]])
    return SpatialSlowField(6, 18.0, posE, posI, 4.0, cfg=cfg)


def test_zeta_gate_is_exactly_zero_at_healthy_baseline():
    out = zero_baseline_sigmoid(np.array([0.0, 0.25, 0.5, 1.0]), 0.5, 0.1)
    assert out[0] == 0.0
    assert np.all(np.diff(out) > 0.0)
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_mode_h_sensor_converts_native_per_step_counts_to_hz():
    # Four E neurons on a 2x2 grid means one neuron/cell.  A native field of
    # 0.004 spikes per 0.1-ms step is therefore exactly 40 Hz.
    native = np.full((2, 2), 0.004)
    hz = local_rate_field_hz(native, dt_ms=0.1, n_population=4, n_grid=2)
    np.testing.assert_allclose(hz, 40.0)


def test_local_h_opens_with_z_depletion_and_closes_with_m():
    slow = _slow()
    slow.S_G = 0.25
    slow.mode_H[:] = 1.0
    slow.z[:4] = np.array([1.0, 0.75, 0.5, 0.5])
    slow.m[:4] = np.array([0.0, 0.0, 0.0, 90.0])
    I_E = np.full(6, 10.0)
    I_I = np.full(6, 2.0)
    I_rec = np.full(6, 4.0)
    out = slow.apply_currents(I_E, I_I, I_E_rec=I_rec)
    # Healthy zeta=0 receives no H contribution.  Depleted/low-M opens it;
    # depleted/high-M closes it again.
    gain = slow.mode_H_gain_at_E()
    assert gain[0] == 0.0
    assert gain[2] > gain[1] > 0.0
    assert gain[3] < 0.1 * gain[2]
    assert out[2] > out[1]


def test_mode_h_common_subtraction_rejects_uniform_gain_and_keeps_hotspot():
    slow = _slow(mode_H_common_subtraction=1.0)
    slow.z[:4] = 0.5
    slow.m[:4] = 0.0
    slow.mode_H[:] = 0.5
    np.testing.assert_array_equal(slow.mode_H_gain_at_E(), np.zeros(4))
    slow.mode_H[slow._iyE[0], slow._ixE[0]] = 1.0
    gain = slow.mode_H_gain_at_E()
    assert gain[0] > 0.0
    assert np.count_nonzero(gain) == 1


def test_mode_h_common_subtraction_zero_is_exact_legacy_gain():
    explicit = _slow(mode_H_common_subtraction=0.0)
    legacy = _slow()
    for slow in (explicit, legacy):
        slow.z[:4] = 0.5
        slow.m[:4] = 0.0
        slow.mode_H[slow._iyE, slow._ixE] = np.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_array_equal(
        explicit.mode_H_gain_at_E(), legacy.mode_H_gain_at_E()
    )


def test_sensor_only_rho_zero_is_membrane_identical_to_mode_h_off():
    sensor = _slow(rho_mode_H=0.0)
    off = _slow(use_mode_H=False, rho_mode_H=0.0)
    for obj in (sensor, off):
        obj.S_G = 0.2
        obj.z[:4] = 0.6
        obj.m[:4] = 10.0
    I_E = np.linspace(5.0, 10.0, 6)
    I_I = np.linspace(1.0, 2.0, 6)
    I_rec = np.linspace(0.5, 3.0, 6)
    np.testing.assert_array_equal(
        sensor.apply_currents(I_E, I_I, I_E_rec=I_rec),
        off.apply_currents(I_E, I_I, I_E_rec=I_rec),
    )


def test_persistent_mode_h_conductance_uses_same_z_m_gate_and_survives_zero_ampa():
    slow = _slow(rho_mode_H=0.0, mode_H_persistent_g_max=0.1)
    slow.mode_H[:] = 1.0
    slow.z[:4] = np.array([1.0, 0.5, 0.5, 0.5])
    slow.m[:4] = np.array([0.0, 0.0, 45.0, 90.0])
    g = slow.mode_H_persistent_g_at_E()
    assert g[0] == 0.0
    assert g[1] > g[2] > g[3] > 0.0
    V = np.full(6, 11.0)
    I_net = np.full(6, 11.0)
    out = persistent_exc_membrane_step(
        V, I_net, np.full(6, 0.99), 4, g, e_exc=60.0
    )
    assert out[1] > out[2] > out[3] > out[0]
    np.testing.assert_array_equal(out[4:], V[4:])


def test_persistent_mode_h_zero_scale_is_exact_membrane_parity():
    V = np.linspace(10.0, 12.0, 6)
    I_net = np.linspace(9.0, 14.0, 6)
    decay = np.linspace(0.98, 0.99, 6)
    expected = I_net + (V - I_net) * decay
    actual = persistent_exc_membrane_step(
        V, I_net, decay, 4, np.zeros(4), e_exc=60.0
    )
    np.testing.assert_array_equal(actual, expected)


def test_mode_h_is_causal_local_state_and_round_trips_checkpoint():
    slow = _slow(rho_mode_H=0.0)
    slow.apply_currents(np.ones(6), np.zeros(6), I_E_rec=np.zeros(6))
    spk = np.array([True, False, False, False, False, False])
    slow.step(spk, None, 1.0)
    assert slow.mode_H.max() > 0.0
    assert slow.mode_H.min() >= 0.0
    state = capture_slow(slow)
    restored = _slow(rho_mode_H=0.0)
    restore_slow(restored, state)
    np.testing.assert_array_equal(restored.mode_H, slow.mode_H)


def test_mode_h_rejects_missing_native_coordinates():
    with pytest.raises(ValueError, match="requires the native Z and M"):
        _slow(use_z=False)


def test_collective_m_divisor_suppresses_base_recurrent_e_without_h():
    slow = _slow(
        use_mode_H=False,
        rho_mode_H=0.0,
        use_mode_M_divisive=True,
        kappa_mode_M=2.0,
        m_mode_div_ref=10.0,
        m_mode_div_power=2.0,
    )
    slow.S_G = 0.0
    slow.m[:4] = 10.0
    assert slow.mode_M_raw_pool() == pytest.approx(1.0)
    assert slow.mode_M_pool() == pytest.approx(0.5)
    out = slow.apply_currents(
        np.full(6, 10.0), np.zeros(6), I_E_rec=np.full(6, 6.0)
    )
    # Hill activation is 0.5 at the reference, so recurrent component is
    # 6/(1 + 2*0.5) = 3 mV; the non-recurrent 4 mV remains outside the
    # divisor, giving 7 mV before the already-existing
    # eta_m*m = 0.01-mV additive M current.
    np.testing.assert_allclose(out[:4], 6.99)
    assert slow.trace_mode_M_divisor[-1] == pytest.approx(2.0)


def test_asymmetric_h_and_collective_m_memory_create_separate_timescales():
    slow = _slow(
        tau_mode_H=10.0,
        tau_mode_H_down=1000.0,
        use_mode_M_divisive=True,
        kappa_mode_M=2.0,
        m_mode_div_ref=10.0,
        m_mode_div_power=2.0,
        use_mode_M_memory=True,
        tau_mode_M_memory_up=1000.0,
        tau_mode_M_memory_down=5000.0,
        tau_adp=1e12,
    )
    slow.mode_H[:] = 1.0
    slow.m[:4] = 10.0
    slow.apply_currents(np.ones(6), np.zeros(6), I_E_rec=np.ones(6))
    slow.step(np.zeros(6, dtype=bool), None, 1000.0)
    # H used the slow decay arm and retained a finite memory across silence.
    np.testing.assert_allclose(slow.mode_H, np.exp(-1.0), rtol=2e-3)
    # M memory rose toward the instantaneous Hill drive (approximately 0.5),
    # but remained a distinct slow state and round-trips in checkpoints.
    assert 0.25 < slow.mode_M_memory < 0.40
    state = capture_slow(slow)
    restored = _slow(
        use_mode_M_divisive=True,
        use_mode_M_memory=True,
    )
    restore_slow(restored, state)
    assert restored.mode_M_memory == pytest.approx(slow.mode_M_memory)


def test_quenched_i2e_recovery_heterogeneity_is_reproducible_and_zero_cv_is_exact():
    resource = np.array([0.2, 0.5, 0.8])
    scalar = recover_i2e_resource(resource, 1.0, 300.0)
    vector_same = recover_i2e_resource(resource, 1.0, np.full(3, 300.0))
    np.testing.assert_array_equal(scalar, vector_same)
    a = _slow(i2e_tau_cv=0.3, i2e_tau_seed=7)
    b = _slow(i2e_tau_cv=0.3, i2e_tau_seed=7)
    np.testing.assert_array_equal(a.i2e_tau_recovery, b.i2e_tau_recovery)
    assert np.std(a.i2e_tau_recovery) > 0.0
    assert np.mean(a.i2e_tau_recovery) == pytest.approx(a.cfg.tau_i2e_depression)
