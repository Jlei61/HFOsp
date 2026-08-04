"""Contracts for the guard that licenses reading a re-simulated ledger.

The regression that motivated this module is the last test: a truncated tail
event made a bit-identical run report one event too many.
"""
import pytest

from src.topic4_fcxr_lc3_reproduction import (
    comparable_margin_ms,
    events_reproduce,
)

CUT = 20000.0


def _rec(n, t0=100.0, step=1000.0, dur=10.0):
    return [dict(t_on_ms=t0 + i * step, t_off_ms=t0 + i * step + dur,
                 dur_ms=dur, peak_ext=0.05 + 0.001 * i) for i in range(n)]


def _fresh(recorded):
    return [dict(t_on=e["t_on_ms"], t_off=e["t_off_ms"], dur_ms=e["dur_ms"],
                 peak_ext=e["peak_ext"]) for e in recorded]


def test_identical_runs_reproduce():
    rec = _rec(12)
    out = events_reproduce(_fresh(rec), rec, cut_ms=CUT)
    assert out["reproduces"] and out["n_compared"] == 12


def test_a_drifted_field_is_refused_and_named():
    rec = _rec(12)
    fresh = _fresh(rec)
    fresh[7]["peak_ext"] += 1e-4
    out = events_reproduce(fresh, rec, cut_ms=CUT)
    assert not out["reproduces"] and "event 7 peak_ext" in out["detail"]


def test_a_drifted_onset_is_refused():
    rec = _rec(12)
    fresh = _fresh(rec)
    fresh[3]["t_on"] += 0.05
    out = events_reproduce(fresh, rec, cut_ms=CUT)
    assert not out["reproduces"] and "event 3 t_on" in out["detail"]


def test_a_missing_event_is_refused():
    rec = _rec(12)
    out = events_reproduce(_fresh(rec)[:-1], rec, cut_ms=CUT)
    assert not out["reproduces"] and "11 events against 12" in out["detail"]


def test_events_the_short_run_never_saw_are_excluded_from_both_sides():
    rec = _rec(12) + _rec(5, t0=CUT + 500.0)
    out = events_reproduce(_fresh(_rec(12)), rec, cut_ms=CUT)
    assert out["reproduces"] and out["n_compared"] == 12


def test_the_margin_is_the_longest_event_either_run_produced():
    rec = _rec(3, dur=10.0) + [dict(t_on_ms=5000.0, t_off_ms=5045.0, dur_ms=45.0,
                                    peak_ext=0.06)]
    assert comparable_margin_ms(_fresh(rec), rec) == 45.0


def test_a_missing_duration_key_fails_loudly_rather_than_comparing_nothing():
    rec = _rec(3)
    fresh = _fresh(rec)
    del fresh[1]["t_off"]
    with pytest.raises(KeyError, match="t_off"):
        events_reproduce(fresh, rec, cut_ms=CUT)


def test_a_tail_event_truncated_by_the_cut_is_not_a_mismatch():
    """The measured regression on seed 401.

    The short run stops mid-event and records 19985-19999; the long run has the
    same event whole as 19985-20014. A plain "ends before the cut" filter keeps
    the truncated copy and drops the whole one, so a bit-identical run reports
    one event too many.
    """
    whole = dict(t_on_ms=19985.0, t_off_ms=20014.0, dur_ms=30.0, peak_ext=0.33381)
    truncated = dict(t_on=19985.0, t_off=19999.0, dur_ms=15.0, peak_ext=0.33381)
    body = _rec(8, t0=100.0, step=2000.0, dur=30.0)
    out = events_reproduce(_fresh(body) + [truncated], body + [whole], cut_ms=CUT)
    assert out["reproduces"], out["detail"]
    assert out["n_compared"] == 8
    assert out["comparable_until_ms"] == CUT - 30.0
