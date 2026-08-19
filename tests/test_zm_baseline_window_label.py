"""The word `baseline` is earned, not assigned.

The 2 s checkpoint was called baseline before it was measured, and it turned out
to be elevated on all three seeds. These tests pin the rule that replaced it: the
label is `baseline` only when EVERY seed finds a qualifying window and they all
agree on which one. Anything else reports `early transition vs pre-ictal`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_baseline_discovery import find_baseline_window  # noqa: E402

BIN = 1.0


def _flat(value, n=4000):
    return np.full(n, value, float)


def _trace(n=400, span=4000.0):
    return np.linspace(0.0, span, n)


def test_a_quiet_window_with_quiet_slow_state_is_found():
    out = find_baseline_window(
        _flat(0.02), BIN, rate_q95=40.0,
        z_trace=np.full(400, 0.99), m_trace=np.full(400, 0.5),
        zm_time_ms=_trace(), z_q95=1.0, m_q95=1.0,
        burn_in_ms=500.0, window_ms=500.0)
    assert out["found"] is True
    assert out["window_ms"] == [500.0, 1000.0]


def test_a_quiet_rate_with_a_loaded_slow_state_is_refused():
    """This is the failure the two slow clauses exist to catch: the rate looks
    like baseline while z and m have already moved."""
    out = find_baseline_window(
        _flat(0.02), BIN, rate_q95=40.0,
        z_trace=np.full(400, 0.99), m_trace=np.full(400, 9.0),
        zm_time_ms=_trace(), z_q95=1.0, m_q95=1.0,
        burn_in_ms=500.0, window_ms=500.0)
    assert out["found"] is False
    assert "early transition vs pre-ictal" in out["consequence"]
    assert all(a["clauses"]["rate_within_zm_off_support"] for a in out["attempts"])
    assert not any(a["clauses"]["m_within_support"] for a in out["attempts"])


def test_an_elevated_rate_is_refused_even_with_a_quiet_slow_state():
    out = find_baseline_window(
        _flat(0.20), BIN, rate_q95=40.0,
        z_trace=np.full(400, 0.999), m_trace=np.full(400, 0.1),
        zm_time_ms=_trace(), z_q95=1.0, m_q95=1.0,
        burn_in_ms=500.0, window_ms=500.0)
    assert out["found"] is False


def test_burn_in_is_skipped_not_searched():
    """A quiet first 500 ms must not be picked; burn-in is excluded by rule."""
    series = np.concatenate([_flat(0.001, 500), _flat(0.02, 3500)])
    out = find_baseline_window(
        series, BIN, rate_q95=40.0,
        z_trace=np.full(400, 0.99), m_trace=np.full(400, 0.5),
        zm_time_ms=_trace(), z_q95=1.0, m_q95=1.0,
        burn_in_ms=500.0, window_ms=500.0)
    assert out["found"] is True
    assert out["window_ms"][0] >= 500.0


def test_every_attempt_is_recorded_so_a_near_miss_is_visible():
    out = find_baseline_window(
        _flat(0.02), BIN, rate_q95=40.0,
        z_trace=np.full(400, 0.99), m_trace=np.full(400, 9.0),
        zm_time_ms=_trace(), z_q95=1.0, m_q95=1.0,
        burn_in_ms=500.0, window_ms=500.0)
    assert len(out["attempts"]) >= 6
    for attempt in out["attempts"]:
        assert set(attempt["clauses"]) == {"rate_within_zm_off_support",
                                           "z_within_support", "m_within_support"}
