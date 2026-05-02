"""TDD tests for `refine_packed_windows_by_all_bool` (legacy
`refine_packedEvents_byAllBool` semantic — synchronized-noise rejector).

Plan reference: docs/archive/epilepsiae_lagpat/legacy_replication_audit_2026-05-03.md
Δ1 — fills a previously-missing legacy step in the new pipeline.
Reference impl: ReplayIED/inter_events/epilepsiae_interictal/hfo_net.py:559.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.group_event_analysis import EventWindow, refine_packed_windows_by_all_bool


def _det(times: list[float], dur: float = 0.05) -> np.ndarray:
    arr = np.array(times, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack([arr, arr + dur], axis=1)


def test_drops_window_when_all_channels_have_events() -> None:
    """A window where 100% of channels fire should be rejected."""
    windows = [EventWindow(1.00, 1.50, event_id=0), EventWindow(5.00, 5.50, event_id=1)]
    # Window 0: all 5 channels have events; Window 1: only channel A has an event.
    dets = {
        "A": _det([1.10, 5.10]),
        "B": _det([1.10]),
        "C": _det([1.20]),
        "D": _det([1.30]),
        "E": _det([1.05]),
    }
    out = refine_packed_windows_by_all_bool(windows, dets, fs=500.0, thresh=0.7)

    assert len(out) == 1
    assert out[0].start == 5.00
    assert out[0].event_id == 0  # reindexed from 0


def test_keeps_window_when_below_threshold_fraction() -> None:
    """5 channels, thresh 0.7 -> require >= 0.7*5 = 3.5 -> i.e. 4+ channels.
    A window with exactly 3 channels firing is kept."""
    windows = [EventWindow(2.00, 2.50, event_id=0)]
    dets = {
        "A": _det([2.10]),
        "B": _det([2.20]),
        "C": _det([2.30]),
        "D": np.zeros((0, 2)),
        "E": np.zeros((0, 2)),
    }
    out = refine_packed_windows_by_all_bool(windows, dets, fs=500.0, thresh=0.7)

    assert len(out) == 1
    assert out[0].start == 2.00


def test_threshold_boundary_70pct_drops_at_or_above() -> None:
    """10 channels, thresh 0.7 -> drop windows with >= 7 active channels."""
    windows = [EventWindow(3.00, 3.50, event_id=0), EventWindow(8.00, 8.50, event_id=1)]
    # window 0: exactly 7 channels fire -> should be dropped (>= thresh)
    # window 1: only 6 channels fire -> kept
    dets = {f"CH{i}": _det([3.10, 8.10]) for i in range(6)}
    dets["CH6"] = _det([3.10])  # only contributes to window 0
    dets["CH7"] = np.zeros((0, 2))
    dets["CH8"] = np.zeros((0, 2))
    dets["CH9"] = np.zeros((0, 2))

    out = refine_packed_windows_by_all_bool(windows, dets, fs=500.0, thresh=0.7)

    # Window 0 had 7 active (>= 0.7*10 = 7.0 -> dropped).
    # Window 1 had 6 active (< 7.0 -> kept).
    assert len(out) == 1
    assert out[0].start == 8.00


def test_empty_windows_returns_empty() -> None:
    dets = {"A": _det([1.0]), "B": _det([1.0])}
    out = refine_packed_windows_by_all_bool([], dets, fs=500.0)
    assert out == []


def test_empty_detections_keeps_all_windows() -> None:
    """Sanity: no all-channel info -> can't reject anything."""
    windows = [EventWindow(1.0, 1.5, event_id=0), EventWindow(2.0, 2.5, event_id=1)]
    out = refine_packed_windows_by_all_bool(windows, {}, fs=500.0, thresh=0.7)
    assert len(out) == 2


def test_thresh_outside_range_raises() -> None:
    windows = [EventWindow(1.0, 1.5, event_id=0)]
    dets = {"A": _det([1.1])}
    with pytest.raises(ValueError, match="thresh"):
        refine_packed_windows_by_all_bool(windows, dets, fs=500.0, thresh=0.0)
    with pytest.raises(ValueError, match="thresh"):
        refine_packed_windows_by_all_bool(windows, dets, fs=500.0, thresh=1.5)


def test_reindexes_kept_windows_consecutively() -> None:
    """After dropping, surviving windows must have index 0..N-1 (legacy contract)."""
    windows = [
        EventWindow(1.0, 1.5, event_id=7),  # original index 7
        EventWindow(3.0, 3.5, event_id=8),
        EventWindow(5.0, 5.5, event_id=9),
    ]
    # Make middle window all-active (gets dropped); outer ones partial.
    dets = {
        "A": _det([1.1, 3.1, 5.1]),
        "B": _det([3.1]),
        "C": _det([3.1]),
        "D": _det([3.1]),
    }
    out = refine_packed_windows_by_all_bool(windows, dets, fs=500.0, thresh=0.7)
    assert len(out) == 2
    assert [w.event_id for w in out] == [0, 1]
    assert out[0].start == 1.0
    assert out[1].start == 5.0


def test_matches_legacy_synrefine_contract() -> None:
    """Anchored to the legacy ReplayIED contract: a packed window where every
    one of N channels has at least one event during it must be dropped at
    thresh=0.7."""
    n = 8
    windows = [EventWindow(1.0, 1.4, event_id=0)]
    dets = {f"E{i}": _det([1.05 + 0.01 * i]) for i in range(n)}
    out = refine_packed_windows_by_all_bool(windows, dets, fs=500.0, thresh=0.7)
    assert len(out) == 0
