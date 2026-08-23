import numpy as np

from src.topic4_node_dualmode import (
    assign_detector_fragments_to_cascades,
    cascade_event_windows,
    spatiotemporal_cascade_labels,
)


def _fragment(start, stop):
    return {"t_on": float(start), "t_off": float(stop)}


def test_travelling_activity_across_adjacent_bins_is_one_cascade():
    counts = np.zeros((5, 5, 5), np.uint16)
    for frame in range(5):
        counts[frame, 2, frame] = 3
    result = spatiotemporal_cascade_labels(
        counts, minimum_active_neurons=2,
    )
    assert len(result["components"]) == 1
    assert result["components"][0]["duration_frames"] == 5


def test_spatially_disconnected_simultaneous_origins_remain_distinct():
    counts = np.zeros((4, 8, 8), np.uint16)
    counts[:, 1, 1] = 3
    counts[:, 6, 6] = 3
    result = spatiotemporal_cascade_labels(
        counts, minimum_active_neurons=2,
    )
    assert len(result["components"]) == 2
    assignment = assign_detector_fragments_to_cascades(
        counts, result["labels"], [_fragment(0, 6)],
        frame_ms=2.0, minimum_dominance=0.7,
    )
    assert assignment[0]["compound"] is True
    assert assignment[0]["dominant_activity_fraction"] == 0.5


def test_detector_dip_does_not_split_one_connected_cascade():
    counts = np.zeros((8, 6, 6), np.uint16)
    counts[0:3, 2, 1:4] = 3
    counts[2:6, 2, 3:5] = 3
    counts[5:8, 2, 4:6] = 3
    result = spatiotemporal_cascade_labels(
        counts, minimum_active_neurons=2,
    )
    assignments = assign_detector_fragments_to_cascades(
        counts, result["labels"], [_fragment(0, 4), _fragment(10, 14)],
        frame_ms=2.0, minimum_dominance=0.7,
    )
    assert len(result["components"]) == 1
    assert assignments[0]["dominant_cascade_id"] == assignments[1]["dominant_cascade_id"]
    assert not any(row["compound"] for row in assignments)


def test_cascade_windows_merge_only_fragments_with_shared_spatiotemporal_ancestry():
    components = [
        {"cascade_id": 7, "start_frame": 2, "stop_frame": 8,
         "activity_mass": 30.0, "active_bin_frames": 7},
        {"cascade_id": 9, "start_frame": 20, "stop_frame": 24,
         "activity_mass": 20.0, "active_bin_frames": 5},
    ]
    assignments = [
        {"detector_fragment_index": 0, "dominant_cascade_id": 7,
         "dominant_activity_fraction": 0.9, "compound": False},
        {"detector_fragment_index": 1, "dominant_cascade_id": 7,
         "dominant_activity_fraction": 0.8, "compound": False},
        {"detector_fragment_index": 2, "dominant_cascade_id": 9,
         "dominant_activity_fraction": 0.6, "compound": True},
    ]
    events, compounds = cascade_event_windows(
        components, assignments,
        [_fragment(4, 8), _fragment(12, 16), _fragment(40, 48)],
        frame_ms=2.0, total_ms=100.0,
    )
    assert len(events) == 1
    assert events[0]["detector_fragment_indices"] == [0, 1]
    assert (events[0]["t_on"], events[0]["t_off"]) == (4.0, 18.0)
    assert compounds == [{
        "detector_fragment_indices": [2], "t_on": 40.0, "t_off": 48.0,
        "compound": True, "dominant_activity_fraction": 0.6,
    }]
