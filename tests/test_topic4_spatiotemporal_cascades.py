import numpy as np

from src.topic4_node_dualmode import (
    assign_detector_fragments_to_directed_lineages,
    assign_detector_fragments_to_cascades,
    cascade_event_windows,
    directed_lineage_onset_maps,
    directed_spatiotemporal_lineages,
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
    assert (events[0]["trigger_t_on"], events[0]["trigger_t_off"]) == (4.0, 16.0)
    assert (events[0]["t_on"], events[0]["t_off"]) == (4.0, 18.0)
    assert compounds == [{
        "detector_fragment_indices": [2], "t_on": 40.0, "t_off": 48.0,
        "compound": True, "dominant_activity_fraction": 0.6,
    }]


def test_directed_lineage_preserves_one_travelling_root():
    counts = np.zeros((5, 5, 5), np.uint16)
    for frame in range(5):
        counts[frame, 2, frame] = 3
    result = directed_spatiotemporal_lineages(
        counts, minimum_active_neurons=2,
    )
    assert len(result["components"]) == 1
    assert set(np.unique(result["labels"])) == {0, 1}
    assert not np.any(result["collision_mask"])


def test_persistent_lineage_bridges_a_short_local_fast_state_gap():
    counts = np.zeros((7, 5, 5), np.uint16)
    counts[0:2, 2, 1] = 3
    counts[4:7, 2, 2] = 3
    immediate = directed_spatiotemporal_lineages(
        counts, minimum_active_neurons=2, maximum_parent_gap_frames=1,
    )
    persistent = directed_spatiotemporal_lineages(
        counts, minimum_active_neurons=2, maximum_parent_gap_frames=3,
    )
    assert len(immediate["components"]) == 2
    assert len(persistent["components"]) == 1
    assert persistent["components"][0]["resumed_after_gap_count"] == 1
    assert persistent["components"][0]["maximum_parent_gap_frames_observed"] == 3


def test_persistent_lineage_starts_a_new_root_after_fast_state_memory_expires():
    counts = np.zeros((8, 5, 5), np.uint16)
    counts[0:2, 2, 1] = 3
    counts[6:8, 2, 2] = 3
    result = directed_spatiotemporal_lineages(
        counts, minimum_active_neurons=2, maximum_parent_gap_frames=3,
    )
    assert len(result["components"]) == 2


def test_persistent_lineage_does_not_join_spatially_separate_reactivations():
    counts = np.zeros((7, 9, 9), np.uint16)
    counts[0:2, 1, 1] = 3
    counts[4:7, 7, 7] = 3
    result = directed_spatiotemporal_lineages(
        counts, minimum_active_neurons=2, maximum_parent_gap_frames=5,
        parent_neighborhood_bins=1,
    )
    assert len(result["components"]) == 2


def test_persistent_lineage_uses_nearest_parent_frame_not_a_stale_root():
    counts = np.zeros((5, 7, 7), np.uint16)
    counts[0, 3, 2] = 3
    counts[2, 3, 4] = 3
    counts[3, 3, 3] = 3
    result = directed_spatiotemporal_lineages(
        counts, minimum_active_neurons=2, maximum_parent_gap_frames=4,
        parent_neighborhood_bins=1,
    )
    assert result["labels"][3, 3, 3] == result["labels"][2, 3, 4]
    assert result["labels"][3, 3, 3] != result["labels"][0, 3, 2]


def test_directed_lineage_does_not_merge_roots_that_later_collide():
    counts = np.zeros((4, 7, 7), np.uint16)
    counts[0, 3, [0, 6]] = 3
    counts[1, 3, [1, 5]] = 3
    counts[2, 3, [2, 4]] = 3
    counts[3, 3, 3] = 6
    undirected = spatiotemporal_cascade_labels(
        counts, minimum_active_neurons=2,
    )
    directed = directed_spatiotemporal_lineages(
        counts, minimum_active_neurons=2,
    )
    assert len(undirected["components"]) == 1
    assert len(directed["components"]) == 2
    assert directed["labels"][3, 3, 3] == -1
    assert directed["collision_mask"][3, 3, 3]


def test_directed_fragment_counts_collision_mass_against_dominance():
    counts = np.zeros((3, 5, 5), np.uint16)
    labels = np.zeros_like(counts, np.int32)
    counts[0, 2, 1] = 4
    labels[0, 2, 1] = 1
    counts[1, 2, 2] = 4
    labels[1, 2, 2] = 2
    counts[2, 2, 3] = 4
    labels[2, 2, 3] = -1
    assignment = assign_detector_fragments_to_directed_lineages(
        counts, labels, [_fragment(0, 5)],
        frame_ms=2.0, minimum_dominance=0.7,
    )[0]
    assert assignment["compound"] is True
    assert np.isclose(assignment["dominant_activity_fraction"], 1 / 3)
    assert np.isclose(assignment["collision_activity_fraction"], 1 / 3)


def test_directed_lineage_onset_map_tracks_first_arrival_per_sheet_bin():
    labels = np.zeros((4, 3, 4), np.int32)
    labels[0, 1, 0] = 7
    labels[1, 1, 0:2] = 7
    labels[2, 1, 2] = 7
    labels[3, 1, 3] = 7
    result = directed_lineage_onset_maps(
        labels, [{"cascade_id": 7}], frame_ms=2.0,
    )
    assert result["evaluable"].tolist() == [True]
    assert result["onset_maps_ms"][0, 1].tolist() == [0.0, 2.0, 4.0, 6.0]


def test_cascade_window_does_not_call_nonreturning_fragment_returned():
    components = [{
        "cascade_id": 3, "start_frame": 1, "stop_frame": 3,
        "activity_mass": 10.0, "active_bin_frames": 3,
    }]
    assignments = [{
        "detector_fragment_index": 0, "dominant_cascade_id": 3,
        "dominant_activity_fraction": 1.0, "compound": False,
    }]
    fragment = {"t_on": 2.0, "t_off": 6.0, "returned": False}
    events, _ = cascade_event_windows(
        components, assignments, [fragment], frame_ms=2.0, total_ms=100.0,
    )
    assert events[0]["returned"] is False
