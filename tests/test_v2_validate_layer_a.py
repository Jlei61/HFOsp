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
