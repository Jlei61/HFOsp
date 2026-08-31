from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_topic4_spatial_zm_qigk_canary import (
    _frozen_endpoint_contact_centers,
)
from src.snn_engine.slow_field import SpatialSlowField, SpatialSlowFieldConfig
from src.topic4_spatial_zm_qigk import (
    SpatialZMQIGKConfig,
    SpatialZMQIGKSlowVars,
    periodic_endpoint_field,
    thresholded_hill_saturation,
)


def _positions():
    pos_e = np.asarray([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
    pos_i = np.asarray([[0.75, 0.75], [1.25, 1.25]])
    return pos_e, pos_i


def test_thresholded_hill_n1_matches_historical_qi_saturation():
    values = np.asarray([0.01, 0.03, 0.04, 0.08])
    expected = np.maximum(values - 0.03, 0.0)
    expected = expected / (0.05 + expected)
    actual = thresholded_hill_saturation(values, 0.03, 0.05, 1.0)
    assert np.allclose(actual, expected)


def test_thresholded_hill_sharpens_event_gate_without_changing_bounds():
    values = np.asarray([0.031, 0.04, 0.06, 0.10])
    smooth = thresholded_hill_saturation(values, 0.03, 0.02, 1.0)
    steep = thresholded_hill_saturation(values, 0.03, 0.02, 4.0)
    assert steep[0] < smooth[0]
    assert steep[-1] > smooth[-1]
    assert np.all((steep >= 0.0) & (steep <= 1.0))


def test_zero_m_and_h_gains_match_existing_qi_field_dynamics():
    pos_e, pos_i = _positions()
    cfg = SpatialZMQIGKConfig(
        n_grid=4,
        sigma_r_mm=0.35,
        tau_rate_ms=20.0,
        field_update_ms=0.1,
        tau_q_ms=5000.0,
        k_q_per_ms=0.1,
        q_min=0.05,
        sigma_q_mm=0.7,
        eta_m=0.0,
        h_smooth_sigma_mm=0.5,
        trace_stride_steps=1,
    )
    hybrid = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0, np.ones(4), cfg=cfg)
    reference = SpatialSlowField(
        6, 18.0, pos_e, pos_i, 2.0,
        cfg=SpatialSlowFieldConfig(
            n_grid=4,
            sigma_r=0.35,
            tau_a=20.0,
            use_qI=True,
            tau_q=5000.0,
            k_q=0.1,
            q_min=0.05,
            sigma_q=0.7,
            eta_E=0.3,
            eta_I=1.0,
            a0_q=0.0,
            a50_q=1.0,
            use_gK=False,
            k_K=0.0,
            sigma_K=0.35,
        ),
    )
    labels = np.asarray([0, 0, 0, 0, 1, 1])
    spike_rows = [
        np.asarray([1, 0, 0, 0, 1, 0], bool),
        np.asarray([0, 1, 0, 0, 0, 1], bool),
        np.asarray([0, 0, 1, 1, 0, 0], bool),
    ]
    for spikes in spike_rows:
        hybrid.step(spikes, labels, 0.1)
        reference.step(spikes, labels, 0.1)
        np.testing.assert_allclose(hybrid.q_I, reference.q_I, atol=1e-14)


def test_patient_field_modulates_parameters_without_changing_their_mean():
    pos_e, pos_i = _positions()
    cfg = SpatialZMQIGKConfig(
        n_grid=4,
        sigma_q_mm=0.7,
        h_smooth_sigma_mm=0.5,
        k_q_per_ms=0.02,
        k_q_h_gain=0.5,
        eta_m=0.04,
        eta_m_h_gain=-0.4,
        q_floor_h_gain=0.2,
    )
    slow = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0,
        np.asarray([0.0, 0.2, 0.8, 1.0]), cfg=cfg)
    assert np.ptp(slow.k_q_grid) > 0.0
    assert np.ptp(slow.eta_m_E) > 0.0
    assert np.ptp(slow.q_floor_grid) > 0.0
    assert np.isclose(np.mean(slow.k_q_grid), cfg.k_q_per_ms)
    assert np.isclose(np.mean(slow.eta_m_E), cfg.eta_m)
    assert np.max(slow.k_q_grid) / np.min(slow.k_q_grid) < 3.0
    assert np.max(slow.eta_m_E) / np.min(slow.eta_m_E) < 3.0


def test_membrane_current_is_spatial_qi_plus_per_neuron_m():
    pos_e, pos_i = _positions()
    cfg = SpatialZMQIGKConfig(
        n_grid=4,
        sigma_q_mm=0.7,
        h_smooth_sigma_mm=0.5,
        eta_m=0.5,
    )
    slow = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0, np.ones(4), cfg=cfg)
    slow.q_I[:] = 0.75
    slow.m[:4] = np.asarray([0.0, 1.0, 2.0, 3.0])
    exc = np.full(6, 10.0)
    inh = np.full(6, 4.0)
    got = slow.apply_currents(exc, inh)
    np.testing.assert_allclose(got[:4], [7.0, 6.5, 6.0, 5.5])
    np.testing.assert_allclose(got[4:], [6.0, 6.0])


def test_m_current_threshold_spares_low_state_and_activates_above_gate():
    pos_e, pos_i = _positions()
    cfg = SpatialZMQIGKConfig(
        n_grid=4,
        sigma_q_mm=0.7,
        h_smooth_sigma_mm=0.5,
        eta_m=0.5,
        m_current_threshold=2.0,
    )
    slow = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0, np.ones(4), cfg=cfg)
    slow.q_I[:] = 0.75
    slow.m[:4] = np.asarray([0.0, 1.0, 2.0, 3.0])
    got = slow.apply_currents(np.full(6, 10.0), np.full(6, 4.0))
    np.testing.assert_allclose(got[:4], [7.0, 7.0, 7.0, 6.5])


def test_bounded_m_current_gate_has_fixed_maximum_current():
    pos_e, pos_i = _positions()
    cfg = SpatialZMQIGKConfig(
        n_grid=4,
        sigma_q_mm=0.7,
        h_smooth_sigma_mm=0.5,
        eta_m=10.0,
        m_current_threshold=2.0,
        m_current_saturation_width=1.0,
        m_current_hill_n=2.0,
    )
    slow = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0, np.ones(4), cfg=cfg)
    slow.q_I[:] = 0.75
    slow.m[:4] = np.asarray([0.0, 2.0, 3.0, 1e6])
    got = slow.apply_currents(np.full(6, 10.0), np.full(6, 4.0))
    np.testing.assert_allclose(got[:3], [7.0, 7.0, 2.0])
    assert got[3] > -3.0 - 1e-6
    assert got[3] < -2.999


def test_bounded_gk_state_never_exceeds_its_ceiling():
    pos_e, pos_i = _positions()
    cfg = SpatialZMQIGKConfig(
        n_grid=4,
        sigma_q_mm=0.7,
        h_smooth_sigma_mm=0.5,
        tau_m_ms=1000.0,
        m_state_ceiling=1.0,
    )
    slow = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0, np.ones(4), cfg=cfg)
    spikes = np.ones(6, dtype=bool)
    labels = np.zeros(6, dtype=int)
    for _ in range(20):
        slow.step(spikes, labels, dt=0.1)
    assert np.all(slow.m[:4] <= 1.0)
    assert np.all(slow.m[:4] > 0.99)


def test_m_build_gain_separates_gk_build_rate_from_current_coupling():
    pos_e, pos_i = _positions()
    cfg = SpatialZMQIGKConfig(
        n_grid=4,
        sigma_q_mm=0.7,
        h_smooth_sigma_mm=0.5,
        tau_m_ms=1000.0,
        m_build_gain=0.25,
        eta_m=7.0,
    )
    slow = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0, np.ones(4), cfg=cfg)
    spikes = np.asarray([1, 0, 0, 0, 0, 0], bool)
    labels = np.zeros(6, dtype=int)
    slow.step(spikes, labels, dt=0.1)
    assert slow.m[0] == 0.25
    np.testing.assert_allclose(slow.m[1:4], 0.0)
    assert slow._m_current_E()[0] == 1.75


def test_q_floor_prevents_complete_disinhibition():
    pos_e, pos_i = _positions()
    cfg = SpatialZMQIGKConfig(
        n_grid=4,
        sigma_q_mm=0.7,
        h_smooth_sigma_mm=0.5,
        k_q_per_ms=10.0,
        q_min=0.2,
        q_floor_h_gain=0.3,
        eta_m=0.0,
    )
    slow = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0,
        np.asarray([0.0, 0.2, 0.8, 1.0]), cfg=cfg)
    labels = np.asarray([0, 0, 0, 0, 1, 1])
    spikes = np.ones(6, bool)
    for _ in range(20):
        slow.step(spikes, labels, 0.1)
    assert np.all(slow.q_I >= slow.q_floor_grid)
    assert np.min(slow.q_I) >= cfg.q_min


def test_frozen_q_clamp_is_unchanged_by_sustained_activity():
    pos_e, pos_i = _positions()
    cfg = SpatialZMQIGKConfig(
        n_grid=4,
        sigma_q_mm=0.7,
        h_smooth_sigma_mm=0.5,
        k_q_per_ms=10.0,
        q_min=0.6,
        q_init=0.7,
        freeze_q=True,
    )
    slow = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0, np.ones(4), cfg=cfg)
    spikes = np.ones(6, dtype=bool)
    labels = np.zeros(6, dtype=int)
    for _ in range(20):
        slow.step(spikes, labels, dt=0.1)
    np.testing.assert_allclose(slow.q_I, 0.7)


def test_patient_field_can_seed_nonhomogeneous_frozen_q_without_randomness():
    pos_e, pos_i = _positions()
    cfg = SpatialZMQIGKConfig(
        n_grid=4,
        sigma_q_mm=0.7,
        h_smooth_sigma_mm=0.5,
        q_min=0.4,
        q_init=0.7,
        q_init_h_gain=0.2,
        freeze_q=True,
    )
    h_e = np.asarray([0.0, 0.2, 0.8, 1.0])
    slow_a = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0, h_e, cfg=cfg)
    slow_b = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0, h_e, cfg=cfg)
    assert np.ptp(slow_a.q_I) > 0.0
    np.testing.assert_array_equal(slow_a.q_I, slow_b.q_I)
    assert np.isclose(np.mean(slow_a.q_I), cfg.q_init)
    high_h = np.unravel_index(np.argmax(slow_a.h_grid), slow_a.h_grid.shape)
    low_h = np.unravel_index(np.argmin(slow_a.h_grid), slow_a.h_grid.shape)
    assert slow_a.q_I[high_h] < slow_a.q_I[low_h]


def test_periodic_endpoint_field_has_all_declared_foci():
    field = periodic_endpoint_field(
        4, 2.0, np.asarray([[0.25, 0.25], [1.25, 1.25]]), 0.2)
    assert field.shape == (4, 4)
    assert field[0, 0] == pytest.approx(1.0)
    assert field[2, 2] == pytest.approx(1.0)
    assert field[1, 1] < field[0, 0]


def test_zero_endpoint_gain_exactly_recovers_previous_q_initialization():
    pos_e, pos_i = _positions()
    cfg = SpatialZMQIGKConfig(
        n_grid=4,
        sigma_q_mm=0.7,
        h_smooth_sigma_mm=0.5,
        q_min=0.4,
        q_init=0.7,
        q_endpoint_gain=0.0,
        freeze_q=True,
    )
    without_centers = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0, np.ones(4), cfg=cfg)
    with_centers = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0, np.ones(4),
        endpoint_centers_xy=np.asarray([[0.25, 0.25], [1.25, 1.25]]),
        cfg=cfg)
    np.testing.assert_array_equal(
        without_centers.q_init_grid, with_centers.q_init_grid)


def test_positive_endpoint_gain_lowers_q_at_both_frozen_endpoints():
    pos_e, pos_i = _positions()
    cfg = SpatialZMQIGKConfig(
        n_grid=4,
        sigma_q_mm=0.7,
        h_smooth_sigma_mm=0.5,
        q_min=0.1,
        q_init=0.7,
        q_endpoint_gain=0.4,
        q_endpoint_sigma_mm=0.2,
        freeze_q=True,
    )
    slow = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0, np.ones(4),
        endpoint_centers_xy=np.asarray([[0.25, 0.25], [1.25, 1.25]]),
        cfg=cfg)
    assert slow.q_init_grid[0, 0] < slow.q_init_grid[1, 1]
    assert slow.q_init_grid[2, 2] < slow.q_init_grid[1, 1]
    assert np.isclose(np.mean(slow.q_init_grid), cfg.q_init)


def test_nonzero_endpoint_gain_requires_frozen_centers():
    pos_e, pos_i = _positions()
    cfg = SpatialZMQIGKConfig(
        n_grid=4,
        sigma_q_mm=0.7,
        h_smooth_sigma_mm=0.5,
        q_endpoint_gain=0.2,
    )
    with pytest.raises(ValueError, match="requires frozen endpoint"):
        SpatialZMQIGKSlowVars(
            6, 18.0, pos_e, pos_i, 2.0, np.ones(4), cfg=cfg)


def test_source_and_sink_q_gains_can_differ_on_frozen_contact_sets():
    pos_e, pos_i = _positions()
    cfg = SpatialZMQIGKConfig(
        n_grid=4,
        sigma_q_mm=0.7,
        h_smooth_sigma_mm=0.5,
        q_min=0.1,
        q_init=0.7,
        q_source_gain=0.3,
        q_sink_gain=0.1,
        q_endpoint_sigma_mm=0.2,
        freeze_q=True,
    )
    slow = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0, np.ones(4),
        source_centers_xy=np.asarray([[0.25, 0.25], [0.25, 0.75]]),
        sink_centers_xy=np.asarray([[1.25, 1.25], [1.25, 1.75]]),
        cfg=cfg)
    source_q = np.mean([slow.q_init_grid[0, 0], slow.q_init_grid[1, 0]])
    sink_q = np.mean([slow.q_init_grid[2, 2], slow.q_init_grid[3, 2]])
    assert source_q < sink_q
    assert np.isclose(np.mean(slow.q_init_grid), cfg.q_init)


def test_runner_selects_frozen_source_or_sink_contacts_without_refitting():
    substrate = SimpleNamespace(
        contact_names=["S2", "K1", "S1", "K2"],
        contact_xy=np.asarray([[2.0, 0.0], [3.0, 0.0],
                               [1.0, 0.0], [4.0, 0.0]]),
        extras={"placement": {
            "source_names": ["S1", "S2"],
            "sink_names": ["K1", "K2"],
        }},
    )
    source_names, source_xy = _frozen_endpoint_contact_centers(
        substrate, side="source")
    sink_names, sink_xy = _frozen_endpoint_contact_centers(
        substrate, side="sink")
    assert source_names == ["S1", "S2"]
    assert sink_names == ["K1", "K2"]
    np.testing.assert_array_equal(source_xy[:, 0], [1.0, 2.0])
    np.testing.assert_array_equal(sink_xy[:, 0], [3.0, 4.0])


def test_spatial_q_initialization_never_starts_below_local_floor():
    pos_e, pos_i = _positions()
    cfg = SpatialZMQIGKConfig(
        n_grid=4,
        sigma_q_mm=0.7,
        h_smooth_sigma_mm=0.5,
        q_min=0.4,
        q_init=0.5,
        q_init_h_gain=0.4,
        q_floor_h_gain=0.3,
        freeze_q=True,
    )
    slow = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0,
        np.asarray([0.0, 0.2, 0.8, 1.0]), cfg=cfg)
    assert np.all(slow.q_init_grid >= slow.q_floor_grid)
    np.testing.assert_array_equal(slow.q_I, slow.q_init_grid)


def test_one_ms_field_hold_does_not_delay_per_neuron_m():
    pos_e, pos_i = _positions()
    cfg = SpatialZMQIGKConfig(
        n_grid=4,
        sigma_q_mm=0.7,
        h_smooth_sigma_mm=0.5,
        field_update_ms=1.0,
        eta_m=0.5,
        tau_m_ms=10.0,
    )
    slow = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0, np.ones(4), cfg=cfg)
    labels = np.asarray([0, 0, 0, 0, 1, 1])
    spikes = np.asarray([1, 0, 0, 0, 0, 0], bool)
    q0 = slow.q_I.copy()
    slow.step(spikes, labels, 0.1)
    assert slow.m[0] == 1.0
    np.testing.assert_array_equal(slow.q_I, q0)
    for _ in range(9):
        slow.step(np.zeros(6, bool), labels, 0.1)
    assert np.any(slow.q_I < q0)


def test_spatial_m_mix_turns_private_spike_into_shared_local_drive():
    pos_e = np.asarray([
        [0.50, 0.50], [0.52, 0.50], [1.50, 1.50], [1.52, 1.50],
    ])
    pos_i = np.asarray([[0.75, 0.75], [1.25, 1.25]])
    cfg = SpatialZMQIGKConfig(
        n_grid=4,
        sigma_q_mm=0.7,
        sigma_m_mm=0.35,
        h_smooth_sigma_mm=0.5,
        field_update_ms=0.1,
        m_spatial_mix=1.0,
        eta_m=0.5,
    )
    slow = SpatialZMQIGKSlowVars(
        6, 18.0, pos_e, pos_i, 2.0, np.ones(4), cfg=cfg)
    labels = np.asarray([0, 0, 0, 0, 1, 1])
    spikes = np.asarray([1, 0, 0, 0, 0, 0], bool)
    slow.step(spikes, labels, 0.1)
    assert slow.m[0] > 0.0
    assert slow.m[0] == slow.m[1]
    assert slow.m[0] > slow.m[2]
