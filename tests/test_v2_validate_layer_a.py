import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v2_validate_layer_a import (
    compute_dur_in_band_frac,
    compute_peak_side_ratio,
    compute_threshold_margin,
    _window_starts_for_record,
    CHUNK_SEC,
    N_WINDOWS_PER_RECORD,
)

def test_dur_in_band_all_pass():
    # all events 100 ms — well within [50, 200]
    events = np.array([[0.0, 0.1], [0.5, 0.6], [1.0, 1.1]])
    assert compute_dur_in_band_frac(events, 50.0, 200.0) == 1.0

def test_dur_in_band_partial():
    # one event 30 ms (too short), one event 150 ms (in band)
    events = np.array([[0.0, 0.030], [0.5, 0.65]])
    assert compute_dur_in_band_frac(events, 50.0, 200.0) == 0.5

def test_peak_side_ratio_basic():
    fs = 1024.0
    env = np.ones(int(fs) * 2) * 5.0
    env[1024:1024+102] = 20.0
    events = np.array([[1.0, 1.1]])
    ratios = compute_peak_side_ratio(env, events, fs)
    assert len(ratios) == 1
    assert ratios[0] == pytest.approx(4.0, rel=1e-3)

def test_window_starts_short_recording_falls_back():
    # 400s recording (< 600s) → only the first window
    starts = _window_starts_for_record(400.0)
    assert starts == [0.0]

def test_window_starts_long_recording_evenly_spaced():
    # 3600s recording, 3 windows: 0, mid, last_start
    starts = _window_starts_for_record(3600.0)
    assert len(starts) == N_WINDOWS_PER_RECORD == 3
    last_start = 3600.0 - CHUNK_SEC
    assert starts[0] == 0.0
    assert starts[-1] == pytest.approx(last_start)
    assert starts[1] == pytest.approx(last_start / 2)

def test_window_starts_no_overlap_at_minimum_dur():
    # exactly 3 * CHUNK_SEC → first/middle/last must not all collapse to 0
    starts = _window_starts_for_record(3 * CHUNK_SEC)
    assert starts[0] == 0.0
    assert starts[-1] > 0.0

def test_threshold_margin_zero_width_event_returns_nan():
    fs = 1024.0
    env = np.ones(2048) * 5.0
    # zero-width event: t0 == t1
    events = np.array([[1.0, 1.0]])
    margins = compute_threshold_margin(env, events, fs, threshold=4.0)
    assert len(margins) == 1
    assert np.isnan(margins[0])

def test_peak_side_ratio_event_at_recording_start_returns_nan():
    fs = 1024.0
    env = np.ones(2048) * 5.0
    env[100:202] = 20.0
    # event at t0=0 (or near enough that pre-side has zero samples)
    events = np.array([[0.0, 0.1]])
    ratios = compute_peak_side_ratio(env, events, fs)
    assert len(ratios) == 1
    assert np.isnan(ratios[0])  # one-sided edge → NaN per the new contract

def test_peak_side_ratio_event_at_recording_end_returns_nan():
    fs = 1024.0
    n = 2048
    env = np.ones(n) * 5.0
    # event at end: t0 = (n - 50)/fs, t1 = n/fs → post-side has zero samples
    t0 = (n - 50) / fs
    t1 = n / fs
    events = np.array([[t0, t1]])
    ratios = compute_peak_side_ratio(env, events, fs)
    assert len(ratios) == 1
    assert np.isnan(ratios[0])
