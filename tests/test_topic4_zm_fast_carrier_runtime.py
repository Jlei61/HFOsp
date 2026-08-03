"""Observation-wrapper tests; no full SNN is constructed here."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from src.topic4_zm_fast_carrier_runtime import (
    DiagnosticSlowWrapper,
    FrozenAllNoStepWrapper,
    rescale_i2e_delay_bins,
)


class _CurrentInner:
    def __init__(self):
        self.nE = 2
        self.S_G = 0.5
        self.H = 0.0
        self.z = np.array([0.5, 1.0, 1.0])
        self.cfg = SimpleNamespace(
            use_SG=True,
            alpha_G=2.0,
            use_H=False,
            alpha_H=0.0,
            beta_SG=0.0,
            use_z=True,
            cond_tau_m_E=20.0,
        )

    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        frac = 0.5
        out = np.asarray(I_E, float) - np.asarray(I_I, float)
        out[:2] = I_E[:2] - I_E_rec[:2] * frac - self.z[:2] * I_I[:2]
        return out


def test_current_diagnostic_delegates_and_separates_effective_charge():
    wrapped = DiagnosticSlowWrapper(_CurrentInner())
    I_E = np.array([6.0, 8.0, 4.0])
    I_I = np.array([2.0, 4.0, 1.0])
    I_rec = np.array([2.0, 2.0, 1.0])
    expected = wrapped.inner.apply_currents(I_E, I_I, None, I_rec)
    got = wrapped.apply_currents(I_E, I_I, None, I_rec)
    np.testing.assert_array_equal(got, expected)
    summary = wrapped.diagnostic_summary()
    assert summary["median_vinf_mv"] == np.median(expected[:2])
    expected_ratio = np.mean([1.0, 4.0]) / np.mean([5.0, 7.0])
    assert summary["effective_inhibitory_to_excitatory_charge_ratio"] == expected_ratio


def test_wrapper_forwards_state_writes_to_inner():
    wrapped = DiagnosticSlowWrapper(_CurrentInner())
    wrapped.S_G = 0.25
    assert wrapped.inner.S_G == 0.25


def test_zero_copy_freeze_all_skips_step_but_keeps_reads_live():
    inner = _CurrentInner()
    inner.cfg.use_phi = False
    calls = []
    inner.step = lambda *args: calls.append(args)
    diagnostic = DiagnosticSlowWrapper(inner)
    frozen = FrozenAllNoStepWrapper(diagnostic)
    frozen.step(np.array([True]), np.array([0]), 0.1)
    assert calls == []
    got = frozen.apply_currents(
        np.array([6.0, 8.0, 4.0]),
        np.array([2.0, 4.0, 1.0]),
        None,
        np.array([2.0, 2.0, 1.0]),
    )
    assert got.shape == (3,)


def test_zero_copy_freeze_refuses_dynamic_phi():
    inner = _CurrentInner()
    inner.cfg.use_phi = True
    with np.testing.assert_raises_regex(ValueError, "dynamic threshold"):
        FrozenAllNoStepWrapper(inner)


def test_dynamic_diagnostic_step_delegates_and_advances_counter():
    inner = _CurrentInner()
    inner.cfg.use_phi = False
    calls = []
    inner.step = lambda *args: calls.append(args)
    diagnostic = DiagnosticSlowWrapper(inner)
    diagnostic.step(np.array([True]), np.array([0]), 0.1)
    assert len(calls) == 1
    assert diagnostic._step_index == 1


def test_i2e_delay_rescaling_moves_only_e_targets_and_preserves_inflight_offsets():
    zero_a = sparse.csc_matrix((3, 2))
    zero_g = sparse.csc_matrix((3, 1))
    gaba_d1 = sparse.csc_matrix(([2.0, 5.0], ([0, 2], [0, 0])), shape=(3, 1))
    net = {
        "ampa_by_delay": [zero_a.copy() for _ in range(3)],
        "gaba_by_delay": [zero_g.copy(), gaba_d1, zero_g.copy()],
        "max_delay_steps": 2,
    }
    old_ring = np.arange(9.0).reshape(3, 3)
    state = {"t": np.asarray(1), "ring_sE": old_ring, "ring_sI": old_ring + 10}
    new_net, new_state, receipt = rescale_i2e_delay_bins(
        net, state, n_e=2, scale=3.0
    )
    assert new_net["max_delay_steps"] == 3
    assert new_net["gaba_by_delay"][1][2, 0] == 5.0  # I target unchanged
    assert new_net["gaba_by_delay"][1][0, 0] == 0.0
    assert new_net["gaba_by_delay"][3][0, 0] == 2.0  # E target delayed
    np.testing.assert_array_equal(new_state["ring_sE"][1], old_ring[1])
    np.testing.assert_array_equal(new_state["ring_sE"][2], old_ring[2])
    np.testing.assert_array_equal(new_state["ring_sE"][3], old_ring[0])
    assert receipt["edges_unchanged"]
