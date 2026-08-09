from __future__ import annotations

import numpy as np
import pytest

from src.topic4_fcxr_lc4_gate import (
    baseline_gate,
    force_matched_candidates,
    onset_surface_gate,
    select_candidate,
    summarize_returning_events,
)


def _separation():
    return {"by_tau": {"1000": {
        "K_midgap": 45.0,
        "hill": {
            "4": {"ictal_mean": 0.8, "interictal_mean": 0.005},
            "6": {"ictal_mean": 0.9, "interictal_mean": 0.0008},
            "8": {"ictal_mean": 0.95, "interictal_mean": 0.0001},
        },
    }}}


def test_candidates_hold_the_executed_ictal_current_fixed():
    rows = force_matched_candidates(_separation(), recurrent_scale=100.0, dose_frac=0.2)
    assert [r["n"] for r in rows] == [6.0, 8.0]
    assert [r["g_m_max"] * r["ictal_activation"] for r in rows] == pytest.approx([16.0, 16.0])
    assert rows[1]["g_m_max"] < rows[0]["g_m_max"]


def test_event_summary_uses_only_returning_events_inside_the_matched_window():
    ev = [
        dict(t_on=1000, dur_ms=9, peak_ext=.04, returned=True),
        dict(t_on=3000, dur_ms=10, peak_ext=.05, returned=True),
        dict(t_on=5000, dur_ms=12, peak_ext=.06, returned=False),
        dict(t_on=7000, dur_ms=11, peak_ext=.05, returned=True),
    ]
    got = summarize_returning_events(ev, start_ms=2000, end_ms=8000)
    assert got["n_returning"] == 2
    assert got["event_rate_hz"] == pytest.approx(2 / 6)
    assert got["median_duration_ms"] == 10.5


def _summary(rate=2.0, cv=.6, dur=10.0, part=.05, n=10):
    return dict(n_returning=n, event_rate_hz=rate, iei_cv=cv,
                median_duration_ms=dur, median_participation=part)


def test_baseline_gate_passes_only_the_joint_functional_neighbourhood():
    got = baseline_gate(_summary(rate=1.8, cv=.7, dur=11, part=.052), _summary(),
                        numerical_safe=True, sustained_bout=False, max_current=.1,
                        recurrent_scale=200.0)
    assert got["passed"]
    assert all(got["clauses"].values())


@pytest.mark.parametrize("change", [
    dict(rate=.79), dict(cv=.39), dict(dur=14.0), dict(part=.07), dict(n=2),
])
def test_each_functional_baseline_clause_can_fail(change):
    kw = dict(rate=2.0, cv=.6, dur=10.0, part=.05, n=10); kw.update(change)
    got = baseline_gate(_summary(**kw), _summary(), numerical_safe=True,
                        sustained_bout=False, max_current=.1, recurrent_scale=200.0)
    assert not got["passed"]


def test_numerics_bout_and_leakage_are_hard_baseline_gates():
    for kwargs in (
        dict(numerical_safe=False, sustained_bout=False, max_current=.1),
        dict(numerical_safe=True, sustained_bout=True, max_current=.1),
        dict(numerical_safe=True, sustained_bout=False, max_current=.3),
    ):
        assert not baseline_gate(_summary(), _summary(), recurrent_scale=200.0, **kwargs)["passed"]


def test_candidate_selection_prefers_n6_when_both_pass():
    rows = [dict(candidate=dict(n=8), gate=dict(passed=True)),
            dict(candidate=dict(n=6), gate=dict(passed=True))]
    assert select_candidate(rows)["candidate"]["n"] == 6
    assert select_candidate([dict(candidate=dict(n=6), gate=dict(passed=False))]) is None


def test_onset_surface_needs_all_three_scientific_clauses():
    rows = [
        dict(role="positive_control", d_label="D10", departed=True),
        dict(role="candidate", d_label="D_healthy", departed=False),
        dict(role="candidate", d_label="D10", departed=False),
        dict(role="candidate", d_label="D30", departed=True),
        dict(role="candidate", d_label="D50", departed=True),
    ]
    got = onset_surface_gate(rows)
    assert got["passed"] and got["first_departing_field"] == "D30"
    rows[0]["departed"] = False
    assert onset_surface_gate(rows)["verdict"] == "ONSET_INSTRUMENT_INVALID"


def test_onset_surface_does_not_extend_past_the_registered_fields():
    rows = [dict(role="positive_control", d_label="D10", departed=True),
            dict(role="candidate", d_label="D_healthy", departed=False)]
    rows += [dict(role="candidate", d_label=x, departed=False) for x in ("D10", "D30", "D50")]
    got = onset_surface_gate(rows)
    assert got["verdict"] == "ONSET_SURFACE_UNREACHABLE_IN_TESTED_RANGE"
    assert not got["passed"]
