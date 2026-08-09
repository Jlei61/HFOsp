from __future__ import annotations

import numpy as np
import pytest

from src.topic4_fcxr_lc4d import adjudicate_latency_screen, derive_latency_candidate


def _base():
    return dict(
        name="lc4c", n=4.0, tau_adp_ms=1000.0, tau_a_on_ms=100.0,
        tau_a_off_ms=10000.0, deadzone=4.0, K=2.0, g_m_max=10.0,
        matched_ictal_current=20.0, calibration={"old": True},
    )


def _entry():
    return dict(gate=dict(passed=True))


def _nominal():
    return dict(
        no_kick=True, no_reset=True, no_parameter_step=True,
        nominal_gate=dict(onset_ms=11000.0, offset_ms=66000.0),
        numerical=dict(finite=True, numerical_unsafe=False, clip_frac_max=0.0),
    )


def test_candidate_uses_exact_onset_plus_four_second_sample():
    a = np.zeros(7000)
    a[1500] = 0.25
    current = np.zeros(7000)
    c = derive_latency_candidate(
        _base(), _entry(), _nominal(), a, 10.0, [0.0, 2.0, 4.0], current, 10.0)
    assert c["g_m_max"] == 80.0
    assert c["g_m_max"] * c["calibration"]["a_mean_at_align"] == 20.0
    assert c["calibration"]["align_time_ms"] == 15000.0
    assert c["calibration"]["interictal_activation_max"] == 0.0


def test_candidate_rejects_missing_exact_sample_and_prefix_leak():
    with pytest.raises(ValueError, match="missing exact alignment sample"):
        derive_latency_candidate(
            _base(), _entry(), _nominal(), np.ones(1499), 10.0,
            [0.0], np.zeros(7000), 10.0)
    a = np.ones(7000)
    with pytest.raises(ValueError, match="prefix"):
        derive_latency_candidate(
            _base(), _entry(), _nominal(), a, 10.0,
            [0.0], np.ones(7000), 10.0)


def _events(n=4):
    return [dict(returned=True, t_on=1000.0 + i * 2000.0) for i in range(n)]


def _screen(**overrides):
    kw = dict(
        regimes=["INTERICTAL"] * 10 + ["ICTAL"] * 4 + ["SILENT"] * 2
                + ["INTERICTAL"] * 2,
        win_ms=1000.0,
        events=_events(),
        current_trace=np.r_[np.zeros(400), np.ones(1400)],
        current_dt_ms=10.0,
        numerical_safe=True,
        refractory_fraction=0.0,
        pre_rate_hz=4.0,
        post_rate_hz=0.5,
    )
    kw.update(overrides)
    return adjudicate_latency_screen(**kw)


def test_latency_screen_accepts_fresh_entry_timely_offset_and_guard():
    out = _screen()
    assert out["verdict"] == "L1_ENTRY_OFFSET_ALIGNED"
    assert out["passed"]
    assert out["onset_ms"] == 10000.0
    assert out["offset_ms"] == 14000.0
    assert all(out["clauses"].values())


def test_latency_screen_rejects_no_bout_and_record_end_bout():
    no_bout = _screen(regimes=["INTERICTAL"] * 18)
    assert no_bout["verdict"] == "TERMINATOR_PREVENTS_QUALIFYING_ENTRY"
    late = _screen(regimes=["INTERICTAL"] * 10 + ["ICTAL"] * 8)
    assert late["verdict"] == "OFFSET_LATENCY_REPAIR_INSUFFICIENT"
    assert not late["passed"]


def test_latency_screen_rejects_relapse_or_unsuppressed_post_rate():
    relapse = _screen(regimes=["INTERICTAL"] * 10 + ["ICTAL"] * 3
                      + ["INTERICTAL", "ICTAL"] + ["INTERICTAL"] * 3)
    assert relapse["verdict"] == "SHORT_POSTICTAL_PROTECTION_INSUFFICIENT"
    unsuppressed = _screen(post_rate_hz=5.0)
    assert unsuppressed["verdict"] == "SHORT_POSTICTAL_PROTECTION_INSUFFICIENT"


def test_latency_screen_rejects_current_leak_and_noncanonical_event_key():
    leak = _screen(current_trace=np.ones(1800))
    assert not leak["clauses"]["first_four_seconds_zero_current"]
    with pytest.raises(KeyError, match="t_on"):
        _screen(events=[dict(returned=True, t_on_ms=1000.0)])
