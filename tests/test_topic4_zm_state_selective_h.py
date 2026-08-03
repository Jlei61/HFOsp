import numpy as np
import pytest

from src.snn_engine.slow_field import (
    SpatialSlowField,
    SpatialSlowFieldConfig,
    local_rate_field_hz,
    zero_baseline_sigmoid,
)
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
    assert slow.mode_M_pool() == pytest.approx(1.0)
    out = slow.apply_currents(
        np.full(6, 10.0), np.zeros(6), I_E_rec=np.full(6, 6.0)
    )
    # recurrent component is 6/(1 + 2*1) = 2 mV; the non-recurrent 4 mV
    # remains outside the divisor, giving 6 mV before the already-existing
    # eta_m*m = 0.01-mV additive M current.
    np.testing.assert_allclose(out[:4], 5.99)
    assert slow.trace_mode_M_divisor[-1] == pytest.approx(3.0)
