"""State characterization must recover a synthetic burst train exactly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_state_characterization import (  # noqa: E402
    band_proxy, characterize_state, interictal_reference)


def _burst_train(dt=0.1, period_ms=86.0, active_ms=22.0, total_ms=500.0,
                 high=300.0, low=0.0):
    n = int(round(total_ms / dt))
    t = np.arange(n) * dt
    phase = np.mod(t, period_ms)
    return np.where(phase < active_ms, high, low)


def test_recovers_burst_geometry():
    rate = _burst_train()
    out = characterize_state(rate, dt_ms=0.1, window_ms=(0.0, 500.0),
                             silence_threshold_hz=1.0)
    assert np.isclose(np.median(out["active_durations_ms"]), 22.0, atol=0.2)
    assert np.isclose(np.median(out["silent_durations_ms"]), 64.0, atol=0.2)
    assert np.isclose(out["burst_interval_ms"], 86.0, atol=0.5)
    assert np.isclose(out["reignition_rate_hz"], 1000.0 / 86.0, atol=0.2)
    assert np.isclose(out["peak_rate_hz"], 300.0)


def test_zero_spike_window_fraction_matches_the_silent_duty_cycle():
    rate = _burst_train()
    out = characterize_state(rate, dt_ms=0.1, window_ms=(0.0, 500.0),
                             silence_threshold_hz=1.0, zero_window_ms=20.0)
    assert 0.3 <= out["zero_spike_window_fraction"] <= 0.55


def test_a_continuous_state_has_no_reignition():
    rate = np.full(5000, 250.0)
    out = characterize_state(rate, dt_ms=0.1, window_ms=(0.0, 500.0),
                             silence_threshold_hz=1.0)
    assert out["zero_spike_window_fraction"] == 0.0
    assert out["n_bursts"] == 1
    assert np.isnan(out["burst_interval_ms"])


def test_band_proxy_finds_a_planted_frequency():
    dt, total = 0.1, 500.0
    t = np.arange(int(total / dt)) * dt
    rate = 50.0 + 20.0 * np.sin(2 * np.pi * 45.0 * t / 1000.0)
    out = band_proxy(rate, dt_ms=dt, band_hz=(30.0, 80.0))
    assert np.isclose(out["peak_frequency_hz"], 45.0, atol=2.5)
    assert out["frequency_resolution_hz"] <= 2.5
    assert out["n_cycles_at_band_low"] >= 10.0


def test_interictal_reference_is_length_matched_and_supplies_the_threshold():
    rng = np.random.default_rng(0)
    rate = np.abs(rng.normal(20.0, 5.0, 20000))
    out = interictal_reference(rate, dt_ms=0.1, window_ms=(1000.0, 1500.0))
    assert out["window_ms"] == (1000.0, 1500.0)
    assert out["n_steps"] == 5000
    assert 25.0 < out["percentile_95_hz"] < 40.0
