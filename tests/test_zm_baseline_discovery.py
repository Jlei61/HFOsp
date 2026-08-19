"""A baseline must be found against a distribution, never asserted by clock."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_baseline_discovery import (  # noqa: E402
    ema_rate_hz, find_baseline_window, window_medians, zm_off_support)


def test_window_medians_are_non_overlapping_and_numerous():
    series = np.arange(4000, dtype=float)
    out = window_medians(series, bin_ms=1.0, window_ms=500.0)
    assert len(out) == 8
    assert out[0] < out[-1]


def test_support_pools_windows_across_reference_runs():
    rng = np.random.default_rng(0)
    refs = [np.abs(rng.normal(0.02, 0.005, 20000)) for _ in range(3)]
    out = zm_off_support(refs, bin_ms=1.0)
    assert out["n_windows"] == 3 * 40          # forty windows per 20 s run
    assert out["q95"] > out["median"]


def test_a_quiet_run_finds_a_baseline_window():
    quiet = np.full(20000, 0.01)
    out = find_baseline_window(
        quiet, bin_ms=1.0, rate_q95=50.0,
        z_trace=np.full(200, 0.99), m_trace=np.full(200, 0.05),
        zm_time_ms=np.linspace(0, 20000, 200),
        disinhibition_q95=0.05, m_q95=0.2)
    assert out["found"] is True
    assert out["window_ms"][0] >= 500.0          # after burn-in


def test_an_already_elevated_run_reports_not_found_rather_than_relaxing():
    """The Joint arm's real situation: nothing after burn-in is inside support."""
    busy = np.full(20000, 0.9)
    out = find_baseline_window(
        busy, bin_ms=1.0, rate_q95=30.0,
        z_trace=np.full(200, 0.90), m_trace=np.full(200, 0.9),
        zm_time_ms=np.linspace(0, 20000, 200),
        disinhibition_q95=0.05, m_q95=0.2)
    assert out["found"] is False
    assert "early transition vs pre-ictal" in out["consequence"]
    assert all(not a["pass"] for a in out["attempts"])


def test_every_clause_is_reported_not_only_the_failing_one():
    out = find_baseline_window(
        np.full(6000, 0.9), bin_ms=1.0, rate_q95=30.0,
        z_trace=np.full(60, 0.99), m_trace=np.full(60, 0.01),
        zm_time_ms=np.linspace(0, 6000, 60),
        disinhibition_q95=0.05, m_q95=0.2)
    first = out["attempts"][0]
    assert set(first["clauses"]) == {"rate_within_zm_off_support",
                                     "disinhibition_within_support",
                                     "m_within_support"}
    assert first["clauses"]["disinhibition_within_support"] is True   # only rate fails
    assert first["clauses"]["rate_within_zm_off_support"] is False


def test_ema_matches_the_engine_convention():
    out = ema_rate_hz(np.full(100, 0.01), bin_ms=1.0, tau_ms=20.0)
    assert np.isclose(out[-1], 10.0, rtol=0.02)     # 0.01 / 1 ms = 10 Hz
