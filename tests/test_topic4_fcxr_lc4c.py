from __future__ import annotations

import numpy as np
import pytest

from src.topic4_fcxr_lc4c import adjudicate_entry, derive_aligned_candidate


def _base():
    return dict(
        name="base", n=4.0, tau_adp_ms=1000.0, tau_a_on_ms=100.0,
        tau_a_off_ms=10000.0, deadzone=4.0, K=2.0, g_m_max=10.0,
        matched_ictal_current=20.0, calibration={"old": True},
    )


def _entry(onset=11000.0):
    return dict(
        onset_ms=onset, n_returning_before_onset=9, theta_h_lc2=1.7,
        no_kick=True, no_reset=True, no_parameter_step=True,
        numerical=dict(finite=True, numerical_unsafe=False, clip_frac_max=0.0),
    )


def test_candidate_uses_locked_theta_and_exact_observed_dose_transfer():
    c = derive_aligned_candidate(_base(), _entry(), [0.0, 0.25, 0.5], [0.0, 2.0, 4.0])
    assert c["theta_h_lc2"] == 1.7
    assert c["g_m_max"] == 40.0
    assert c["g_m_max"] * c["calibration"]["closed_high_a_mean_max"] == 20.0
    assert c["calibration"]["interictal_activation_max"] == 0.0


@pytest.mark.parametrize("onset", [None, 7000.0, 16000.0])
def test_invalid_entry_anchor_is_rejected(onset):
    row = _entry(11000.0)
    row["onset_ms"] = onset
    with pytest.raises(ValueError, match="ENTRY_OFFSET_REPAIR_NOT_IDENTIFIABLE"):
        derive_aligned_candidate(_base(), row, [0.5], [0.0])


def _events(n=4):
    # The canonical LC3/LC4 event contract stores onset in milliseconds as ``t_on``.
    # Keeping the real producer schema here prevents a fixture-only alias from masking drift.
    return [dict(returned=True, t_on=1000.0 + i * 1000.0) for i in range(n)]


def test_entry_gate_accepts_aligned_safe_probe_with_zero_prefix():
    out = adjudicate_entry(
        regimes=["INTERICTAL"] * 10 + ["ICTAL"] * 5,
        win_ms=1000.0, events=_events(), current_trace=np.r_[np.zeros(400), np.ones(1100)],
        current_dt_ms=10.0, numerical_safe=True, refractory_fraction=0.0)
    assert out["verdict"] == "C1_ENTRY_ALIGNED"
    assert out["passed"]
    assert all(out["clauses"].values())


def test_entry_gate_rejects_early_entry_and_pre_entry_leak():
    early = adjudicate_entry(
        regimes=["INTERICTAL"] * 7 + ["ICTAL"] * 8,
        win_ms=1000.0, events=_events(), current_trace=np.zeros(1500),
        current_dt_ms=10.0, numerical_safe=True, refractory_fraction=0.0)
    assert early["verdict"] == "C1_ENTRY_TOO_EARLY"
    leaking = adjudicate_entry(
        regimes=["INTERICTAL"] * 10 + ["ICTAL"] * 5,
        win_ms=1000.0, events=_events(), current_trace=np.ones(1500),
        current_dt_ms=10.0, numerical_safe=True, refractory_fraction=0.0)
    assert leaking["verdict"] == "C1_ENTRY_GATE_FAILED"
    assert not leaking["clauses"]["first_four_seconds_zero_current"]


def test_entry_gate_rejects_missing_bout():
    out = adjudicate_entry(
        regimes=["INTERICTAL"] * 15, win_ms=1000.0, events=_events(),
        current_trace=np.zeros(1500), current_dt_ms=10.0,
        numerical_safe=True, refractory_fraction=0.0)
    assert out["verdict"] == "C1_NO_ENTRY"
    assert not out["passed"]


def test_entry_gate_rejects_noncanonical_event_onset_key():
    with pytest.raises(KeyError, match="t_on"):
        adjudicate_entry(
            regimes=["INTERICTAL"] * 10 + ["ICTAL"] * 5,
            win_ms=1000.0,
            events=[dict(returned=True, t_on_ms=1000.0)],
            current_trace=np.zeros(1500), current_dt_ms=10.0,
            numerical_safe=True, refractory_fraction=0.0)
