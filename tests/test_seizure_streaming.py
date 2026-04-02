"""Unit tests for detect_seizure_streaming and match_seizure_intervals."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.preprocessing import (
    _flag_to_runs,
    _merge_close_runs,
    _robust_z,
    match_seizure_intervals,
    detect_seizure_onsets_from_data,
)


class TestFlagToRuns:
    def test_empty(self):
        assert _flag_to_runs(np.array([], dtype=bool)) == []

    def test_single_run(self):
        flag = np.array([False, True, True, True, False])
        assert _flag_to_runs(flag) == [(1, 4)]

    def test_multiple_runs(self):
        flag = np.array([True, True, False, True, False, False, True])
        runs = _flag_to_runs(flag)
        assert runs == [(0, 2), (3, 4), (6, 7)]

    def test_trailing_run(self):
        flag = np.array([False, False, True, True])
        assert _flag_to_runs(flag) == [(2, 4)]


class TestMergeCloseRuns:
    def test_no_runs(self):
        assert _merge_close_runs([], np.array([]), 1.0) == []

    def test_far_apart(self):
        t = np.arange(100, dtype=np.float64)
        runs = [(0, 5), (50, 55)]
        result = _merge_close_runs(runs, t, 10.0)
        assert result == [(0, 5), (50, 55)]

    def test_close_merge(self):
        t = np.arange(100, dtype=np.float64)
        runs = [(0, 5), (8, 15)]
        result = _merge_close_runs(runs, t, 5.0)
        assert result == [(0, 15)]


class TestMatchSeizureIntervals:
    def test_empty_both(self):
        m = match_seizure_intervals([], [])
        assert m["recall"] == 1.0
        assert m["tp"] == []

    def test_all_fn(self):
        m = match_seizure_intervals([(0, 10), (20, 30)], [])
        assert len(m["fn"]) == 2
        assert m["recall"] == 0.0

    def test_all_fp(self):
        m = match_seizure_intervals([], [(5, 15)])
        assert len(m["fp"]) == 1
        assert m["precision"] == 0.0

    def test_perfect_match(self):
        m = match_seizure_intervals([(10, 20)], [(10, 20)])
        assert len(m["tp"]) == 1
        assert m["recall"] == 1.0
        assert m["precision"] == 1.0
        assert m["tp"][0]["onset_err"] == 0.0
        assert m["tp"][0]["offset_err"] == 0.0

    def test_overlap_match(self):
        m = match_seizure_intervals([(10, 20)], [(15, 25)])
        assert len(m["tp"]) == 1
        assert m["tp"][0]["onset_err"] == pytest.approx(5.0)
        assert m["tp"][0]["offset_err"] == pytest.approx(5.0)

    def test_no_overlap(self):
        m = match_seizure_intervals([(10, 20)], [(30, 40)])
        assert len(m["fn"]) == 1
        assert len(m["fp"]) == 1

    def test_multi_event(self):
        manual = [(10, 20), (50, 60), (100, 110)]
        detected = [(12, 22), (55, 65)]
        m = match_seizure_intervals(manual, detected)
        assert len(m["tp"]) == 2
        assert len(m["fn"]) == 1
        assert len(m["fp"]) == 0


class TestDetectSeizureFromSynthetic:
    """Integration test with synthetic data fed to detect_seizure_onsets_from_data."""

    def _make_seizure_data(self, sfreq=500, dur_sec=120, sz_on=50, sz_off=70, n_ch=10):
        n = int(dur_sec * sfreq)
        rng = np.random.RandomState(42)
        data = rng.randn(n_ch, n) * 0.5
        on_samp = int(sz_on * sfreq)
        off_samp = int(sz_off * sfreq)
        data[:, on_samp:off_samp] += rng.randn(n_ch, off_samp - on_samp) * 10.0
        return data, sfreq

    def test_detects_large_seizure(self):
        data, sfreq = self._make_seizure_data()
        res = detect_seizure_onsets_from_data(data, sfreq, ll_k=4.0, rms_k=4.0)
        assert len(res["onsets_sec"]) >= 1
        assert res["onsets_sec"][0] < 55.0
        assert res["offsets_sec"][-1] > 65.0

    def test_no_false_alarm_on_noise(self):
        rng = np.random.RandomState(99)
        data = rng.randn(10, 60000) * 0.5
        res = detect_seizure_onsets_from_data(data, 500.0, ll_k=6.0, rms_k=6.0)
        assert len(res["onsets_sec"]) == 0
