"""Tests for Yuquan synchronous-count group window packing."""

from __future__ import annotations

import numpy as np

from src.group_event_analysis import build_windows_from_detections


def _ev(starts: list[float], dur: float = 0.05) -> np.ndarray:
    s = np.asarray(starts, dtype=np.float64)
    if s.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack([s, s + dur], axis=1)


def test_two_channels_coactive_yields_centered_window() -> None:
    dets = {
        "A1": _ev([1.0]),
        "B1": _ev([1.02]),
    }
    wins = build_windows_from_detections(
        dets,
        window_sec=0.5,
        chns_thr=0.5,
        time_axis_hz=500.0,
        t_max_sec=10.0,
    )
    assert len(wins) == 1
    assert abs((wins[0].start + wins[0].end) * 0.5 - 1.01) < 0.05


def test_non_overlapping_separate_runs() -> None:
    dets = {
        "A1": _ev([0.0, 2.0]),
        "B1": _ev([0.02, 2.02]),
    }
    wins = build_windows_from_detections(
        dets,
        window_sec=0.5,
        chns_thr=0.5,
        t_max_sec=10.0,
    )
    assert len(wins) == 2


def test_overlapping_packed_windows_drop_both_like_legacy() -> None:
    dets = {
        "A1": _ev([1.00, 1.20]),
        "B1": _ev([1.02, 1.22]),
    }
    wins = build_windows_from_detections(
        dets,
        window_sec=0.5,
        chns_thr=0.5,
        t_max_sec=10.0,
    )
    assert len(wins) == 0
