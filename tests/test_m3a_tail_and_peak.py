"""M3A-A2 science-decided helpers: absolute tail_to_baseline (A) + real activity peak (C).

Pure functions, synthetic input, no SNN. They pin the user's 2026-06-27 decisions:
  A -- "absolute ruler": return-to-baseline is tail activity vs a FIXED quiet baseline
       window (BASELINE_MS), NEVER the event's own peak; returned if ratio <= 1.5.
  C -- "take several peaks": the canonical event peak is the real activity-fraction
       peak inside the event window (not the window midpoint placeholder).
"""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.sef_hfo_a2 import tail_to_baseline_absolute, event_peak_ms  # noqa: E402


def _rate_with(baseline, event_peak, tail, T=500):
    """rate[ms]: ~baseline in [5,50], a spike to event_peak at [100,200], tail level in [200,500]."""
    r = np.full(T, float(baseline))
    r[100:200] = float(event_peak)
    r[200:] = float(tail)
    return r


def test_tail_absolute_returned_when_tail_near_baseline():
    rate = _rate_with(baseline=1.0, event_peak=20.0, tail=1.1)
    ratio, returned = tail_to_baseline_absolute(rate, dt_ms=1.0, t_off_ms=200.0)
    assert returned is True
    assert ratio < 1.5


def test_tail_absolute_not_returned_when_tail_stays_high():
    rate = _rate_with(baseline=1.0, event_peak=20.0, tail=8.0)
    ratio, returned = tail_to_baseline_absolute(rate, dt_ms=1.0, t_off_ms=200.0)
    assert returned is False
    assert ratio > 1.5


def test_tail_absolute_denominator_is_fixed_baseline_not_event_peak():
    # Two events with the SAME baseline and SAME tail but very different peaks must give
    # the SAME ratio -- proving the ruler is against the fixed baseline, not the peak.
    r_small = _rate_with(baseline=1.0, event_peak=5.0, tail=2.0)
    r_big = _rate_with(baseline=1.0, event_peak=50.0, tail=2.0)
    ratio_small, _ = tail_to_baseline_absolute(r_small, 1.0, 200.0)
    ratio_big, _ = tail_to_baseline_absolute(r_big, 1.0, 200.0)
    assert ratio_small == pytest.approx(ratio_big)


def test_event_peak_is_the_activity_fraction_max_in_window():
    af = np.zeros(500)
    af[100:200] = 0.2
    af[150] = 0.9            # the true activity peak
    t_peak = event_peak_ms(af, bin_w=1.0, t_on_ms=100.0, t_off_ms=200.0)
    assert t_peak == pytest.approx(150.0)


def test_event_peak_stays_inside_the_event_window():
    af = np.zeros(500)
    af[300] = 5.0            # a bigger bump OUTSIDE the event window must be ignored
    af[120:140] = 0.3
    t_peak = event_peak_ms(af, bin_w=1.0, t_on_ms=100.0, t_off_ms=200.0)
    assert 100.0 <= t_peak <= 200.0
