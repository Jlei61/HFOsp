"""The noise scale must be measured from the trace, not assumed.

The claim this supports -- that the 200 ms probe endpoint carries a noise term
the size of one spontaneous event -- is only worth anything if that size comes
from the run's own activity rather than from an idealised peak-times-duration
figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_topic4_zm_probe_snr import spontaneous_event_scale  # noqa: E402

N_E = 32000


def test_a_silent_trace_has_no_event_scale():
    out = spontaneous_event_scale(np.zeros(1000), 1.0, N_E, window_ms=200.0)
    assert out["max_spikes_in_window"] == 0.0
    assert out["median_spikes_in_window"] == 0.0


def test_one_planted_burst_sets_the_scale():
    trace = np.zeros(1000)
    trace[400:420] = 0.14                     # 20 ms at 14 % of the population
    out = spontaneous_event_scale(trace, 1.0, N_E, window_ms=200.0)
    assert out["max_spikes_in_window"] == pytest.approx(0.14 * N_E * 20)
    # a window that misses the burst sees nothing, so the median stays at zero
    assert out["median_spikes_in_window"] == 0.0


def test_the_scale_grows_with_the_window_it_is_measured_over():
    trace = np.full(1000, 0.001)
    short = spontaneous_event_scale(trace, 1.0, N_E, window_ms=50.0)
    long = spontaneous_event_scale(trace, 1.0, N_E, window_ms=200.0)
    assert long["median_spikes_in_window"] == pytest.approx(
        4.0 * short["median_spikes_in_window"])


def test_a_window_longer_than_the_trace_is_refused_rather_than_silently_empty():
    with pytest.raises(ValueError):
        spontaneous_event_scale(np.zeros(10), 1.0, N_E, window_ms=200.0)
