from __future__ import annotations

import numpy as np

from src.group_event_analysis import (
    legacy_refine_channels_from_detections,
    legacy_refine_counts_from_detection_sets,
)


def _mk_dets(starts: list[float], dur: float = 0.05) -> np.ndarray:
    arr = np.array(starts, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack([arr, arr + float(dur)], axis=1)


def test_legacy_refine_closed_loop_adds_context_channel() -> None:
    detections = {
        # strong channels used to build provisional windows
        "A1": _mk_dets([1.00, 2.00, 3.00, 4.00]),
        "B1": _mk_dets([1.02, 2.02, 3.02, 4.02]),
        # lower global count but consistently overlaps provisional windows
        "C1": _mk_dets([1.01, 2.01, 3.01]),
        # background channel mostly outside provisional windows
        "D1": _mk_dets([10.0, 11.0]),
    }

    out = legacy_refine_channels_from_detections(
        detections,
        window_sec=0.20,
        initial_method="top_n",
        initial_top_n=2,
        refine_method="top_n",
        refine_top_n=3,
        min_count=1,
        provisional_min_channels=1,
    )

    assert out["initial_selected_channels"] == ["A1", "B1"]
    assert out["provisional_windows"].shape[0] == 4
    # Closed-loop recount should recover C1 (not in initial pick).
    assert out["refined_channels"] == ["A1", "B1", "C1"]


def test_legacy_refine_is_deterministic() -> None:
    detections = {
        "A1": _mk_dets([0.0, 1.0, 2.0]),
        "B1": _mk_dets([0.02, 1.02, 2.02]),
        "C1": _mk_dets([5.0]),
    }
    out1 = legacy_refine_channels_from_detections(
        detections,
        window_sec=0.2,
        initial_method="top_n",
        initial_top_n=2,
        refine_method="top_n",
        refine_top_n=2,
    )
    out2 = legacy_refine_channels_from_detections(
        detections,
        window_sec=0.2,
        initial_method="top_n",
        initial_top_n=2,
        refine_method="top_n",
        refine_top_n=2,
    )
    assert out1["refined_channels"] == out2["refined_channels"]
    assert np.array_equal(out1["refined_counts"], out2["refined_counts"])


def test_subject_level_legacy_refine_counts_from_detection_sets() -> None:
    dets1 = {
        "A1": _mk_dets([1.00, 2.00, 3.00]),
        "B1": _mk_dets([1.02, 2.02, 3.02]),
        "C1": _mk_dets([1.01, 2.01, 3.01]),
        "D1": _mk_dets([10.0]),
    }
    dets2 = {
        "A1": _mk_dets([4.00, 5.00, 6.00]),
        "B1": _mk_dets([4.02, 5.02, 6.02]),
        "C1": _mk_dets([4.01, 5.01]),
        "D1": _mk_dets([11.0]),
    }
    out = legacy_refine_counts_from_detection_sets(
        [dets1, dets2],
        ["A1", "B1", "C1", "D1"],
        pick_k=0.0,
        refine_window_sec=0.3,
    )
    assert out["initial_selected_channels"] == ["A1", "B1", "C1"]
    assert out["refined_channels"] == ["A1", "B1"]
    assert out["refined_counts"].tolist() == [6, 6, 5, 0]


def test_subject_level_legacy_refine_honors_initial_counts_override() -> None:
    dets = {
        "A1": _mk_dets([1.00, 2.00]),
        "B1": _mk_dets([1.02, 2.02]),
        "C1": _mk_dets([1.01, 2.01]),
    }
    out = legacy_refine_counts_from_detection_sets(
        [dets],
        ["A1", "B1", "C1"],
        initial_counts_override=[20, 20, 1],
        pick_k=0.0,
        refine_window_sec=0.3,
    )
    assert out["initial_counts"].tolist() == [2, 2, 2]
    assert out["initial_counts_for_pick"].tolist() == [20, 20, 1]
    assert out["initial_selected_channels"] == ["A1", "B1"]
