"""Observation-wrapper tests; no full SNN is constructed here."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.topic4_zm_fast_carrier_runtime import DiagnosticSlowWrapper


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
