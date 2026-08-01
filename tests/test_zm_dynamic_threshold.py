"""TDD for the Phase-D E-only dynamic-threshold increment."""
from __future__ import annotations

import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, "src/snn_engine")
from kick_probe import (  # noqa: E402
    i2e_depression_apply,
    scatter_i2e_emissions_at_spike_time,
)
from lfp import LFPRecorder  # noqa: E402
from src.snn_engine.slow_field import (
    SpatialSlowField,
    SpatialSlowFieldConfig,
    deplete_i2e_resource,
    recover_i2e_resource,
)
from src.topic4_zm_checkpoint import capture_slow, restore_slow
from src.topic4_zm_fork_state import FreezePolicy, FreezeWrapper


def _field(*, use_phi=True, tau_phi=100.0, delta_phi=2.0, core=True):
    nE, nI = 4, 2
    posE = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
    posI = np.array([[0.75, 0.75], [1.25, 1.25]])
    cfg = SpatialSlowFieldConfig(
        n_grid=4,
        use_qI=False,
        use_gK=False,
        use_phi=use_phi,
        tau_phi=tau_phi,
        delta_phi=delta_phi,
    )
    core_mask = np.array([True, True, False, False]) if core else None
    return SpatialSlowField(
        nE + nI,
        18.0,
        posE,
        posI,
        2.0,
        core_mask_E=core_mask,
        cfg=cfg,
    )


def test_phi_config_is_fail_closed():
    with pytest.raises(ValueError, match="tau_phi"):
        SpatialSlowFieldConfig(use_phi=True, tau_phi=0.0).validate()
    with pytest.raises(ValueError, match="delta_phi"):
        SpatialSlowFieldConfig(use_phi=True, delta_phi=-0.1).validate()


def test_disabled_phi_is_literal_threshold_passthrough():
    slow = _field(use_phi=False)
    base = np.array([16.0, 17.0, 18.0, 19.0, 18.0, 18.0])
    got = slow.threshold(base)
    assert got is base
    assert np.array_equal(slow.phi_increment, np.zeros(6))


def test_single_spike_increments_only_the_e_cell():
    slow = _field()
    spikes = np.array([True, False, False, False, True, False])
    slow.step(spikes, None, 0.1)
    np.testing.assert_array_equal(
        slow.phi_increment, np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )


def test_exact_decay_then_increment_equation():
    slow = _field(tau_phi=80.0, delta_phi=1.5)
    slow.phi_increment[0] = 3.0
    spikes = np.array([True, False, False, False, False, False])
    slow.step(spikes, None, 0.2)
    expected = 3.0 * np.exp(-0.2 / 80.0) + 1.5
    assert slow.phi_increment[0] == pytest.approx(expected, abs=1e-14)
    slow.step(np.zeros(6, bool), None, 0.2)
    assert slow.phi_increment[0] == pytest.approx(
        expected * np.exp(-0.2 / 80.0), abs=1e-14
    )


def test_phi_is_an_increment_on_the_heterogeneous_base():
    slow = _field()
    slow.phi_increment[:4] = [0.0, 0.5, 1.0, 1.5]
    base = np.array([15.5, 16.0, 18.0, 19.0, 18.0, 18.0])
    got = slow.threshold(base)
    np.testing.assert_array_equal(
        got, np.array([15.5, 16.5, 19.0, 20.5, 18.0, 18.0])
    )
    scalar = slow.threshold(18.0)
    np.testing.assert_array_equal(
        scalar, np.array([18.0, 18.5, 19.0, 19.5, 18.0, 18.0])
    )


def test_checkpoint_roundtrip_preserves_phi_exactly():
    source = _field()
    source.phi_increment[:4] = [0.1, 0.2, 0.3, 0.4]
    state = capture_slow(source)
    assert "slow.phi_increment" in state
    target = _field()
    restore_slow(target, state)
    np.testing.assert_array_equal(
        target.phi_increment, source.phi_increment
    )


def test_frozen_zm_wrapper_leaves_phi_dynamic():
    slow = _field()
    slow.z[:4] = 0.5
    slow.m[:4] = 2.0
    wrapper = FreezeWrapper(
        slow,
        FreezePolicy(
            freeze_z=True,
            freeze_m=True,
            freeze_sg_family=True,
        ),
    )
    z0, m0 = slow.z.copy(), slow.m.copy()
    wrapper.step(
        np.array([False, True, False, False, False, False]), None, 0.1
    )
    np.testing.assert_array_equal(slow.z, z0)
    np.testing.assert_array_equal(slow.m, m0)
    assert slow.phi_increment[1] == 2.0


def test_phi_traces_exist_only_when_enabled_and_are_core_aware():
    enabled = _field()
    enabled.step(
        np.array([True, False, True, False, False, False]), None, 0.1
    )
    assert enabled.trace_phi_mean == [1.0]
    assert enabled.trace_phi_max == [2.0]
    assert enabled.trace_phi_core_mean == [1.0]
    assert enabled.trace_phi_surround_mean == [1.0]

    disabled = _field(use_phi=False)
    disabled.step(np.zeros(6, bool), None, 0.1)
    assert disabled.trace_phi_mean == []
    assert disabled.trace_phi_max == []
    assert disabled.trace_phi_core_mean == []
    assert disabled.trace_phi_surround_mean == []


def test_inhibitory_state_config_rejects_nonphysical_parameters():
    with pytest.raises(ValueError, match="U_i2e_depression"):
        SpatialSlowFieldConfig(
            use_i2e_depression=True, U_i2e_depression=1.0
        ).validate()
    with pytest.raises(ValueError, match="d_i2e_min"):
        SpatialSlowFieldConfig(
            use_i2e_depression=True, d_i2e_min=0.0
        ).validate()
    with pytest.raises(ValueError, match="tau_i_adaptation"):
        SpatialSlowFieldConfig(
            use_i_adaptation=True, tau_i_adaptation=0.0
        ).validate()


def test_i2e_resource_recovery_use_and_floor_are_exact():
    recovered = recover_i2e_resource(np.array([0.2, 0.7]), 0.1, 100.0)
    expected = 1.0 - (1.0 - np.array([0.2, 0.7])) * np.exp(-0.1 / 100.0)
    np.testing.assert_allclose(recovered, expected, atol=0, rtol=1e-15)
    used = deplete_i2e_resource(recovered, 0.95, 0.20)
    np.testing.assert_array_equal(used, np.array([0.20, 0.20]))


def test_i_adaptation_changes_only_i_thresholds():
    slow = _field(use_phi=False)
    slow.cfg.use_i_adaptation = True
    slow.cfg.tau_i_adaptation = 100.0
    slow.cfg.delta_i_adaptation = 1.25
    slow.step(np.array([True, False, False, False, True, False]), None, 0.1)
    np.testing.assert_array_equal(slow.i_adaptation_increment[:4], np.zeros(4))
    np.testing.assert_array_equal(slow.i_adaptation_increment[4:], [1.25, 0.0])
    np.testing.assert_array_equal(
        slow.threshold(np.full(6, 18.0)), [18, 18, 18, 18, 19.25, 18]
    )


def test_i2e_depression_scales_only_edges_onto_e():
    weights = np.array([2.0, 3.0, 5.0, 7.0])
    targets = np.array([0, 3, 4, 5])
    resource = np.array([0.4, 0.6, 0.2, 0.8])
    got = i2e_depression_apply(weights, targets, resource, NE=4)
    np.testing.assert_allclose(got, [0.8, 1.8, 5.0, 7.0], rtol=0, atol=1e-15)


def test_i2e_delay_ring_freezes_resource_at_each_emission_time():
    """Two spikes from one I source keep their launch amplitudes in flight."""
    ring = np.zeros((12, 2), dtype=float)
    weights = np.array([10.0, 7.0])
    targets = np.array([0, 1])  # one E and one I target from the same I source
    first = scatter_i2e_emissions_at_spike_time(
        ring, np.array([3, 3]), targets, weights,
        np.array([0.8, 0.8]), NE=1,
    )
    second = scatter_i2e_emissions_at_spike_time(
        ring, np.array([7, 7]), targets, weights,
        np.array([0.4, 0.4]), NE=1,
    )
    current_resource_at_arrival = 0.1

    np.testing.assert_array_equal(first, [8.0, 7.0])
    np.testing.assert_array_equal(second, [4.0, 7.0])
    np.testing.assert_array_equal(ring[3], [8.0, 7.0])
    np.testing.assert_array_equal(ring[7], [4.0, 7.0])
    assert current_resource_at_arrival not in ring


def test_lfp_component_readout_sums_exactly_to_legacy_total():
    p = SimpleNamespace(Rr=10.0, rx=1.0)
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    labels = np.array([0, 0, 1])
    rec = LFPRecorder(p, pos, labels, sites=np.array([[0.5, 0.0]]))
    I_E = np.array([2.0, -4.0, 99.0])
    I_I = np.array([-3.0, 5.0, 99.0])
    exc, inh = rec.sample_components(I_E, I_I)
    np.testing.assert_allclose(rec.sample(I_E, I_I), exc + inh, rtol=1e-15, atol=1e-15)


def test_checkpoint_roundtrip_preserves_both_inhibitory_states():
    source = _field(use_phi=False)
    source.i2e_resource[:] = [0.3, 0.8]
    source.i_adaptation_increment[4:] = [0.7, 1.1]
    state = capture_slow(source)
    assert "slow.i2e_resource" in state
    assert "slow.i_adaptation_increment" in state
    target = _field(use_phi=False)
    restore_slow(target, state)
    np.testing.assert_array_equal(target.i2e_resource, source.i2e_resource)
    np.testing.assert_array_equal(
        target.i_adaptation_increment, source.i_adaptation_increment
    )
