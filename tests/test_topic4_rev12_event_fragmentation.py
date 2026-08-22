import numpy as np

from scripts.audit_topic4_rev12_event_fragmentation import (
    fragmentation_counts,
    merge_returned_event_indices,
)
from src.topic4_node_dualmode import merge_detected_event_fragments


def test_merge_uses_returned_segments_and_includes_frozen_boundary():
    onset = np.asarray([0.0, 30.0, 80.0, 130.0])
    offset = np.asarray([10.0, 40.0, 90.0, 140.0])
    returned = np.asarray([True, False, True, True])
    assert merge_returned_event_indices(onset, offset, returned, 40.0) == [
        [0], [2, 3]
    ]


def test_fragmentation_counts_opposite_labels_inside_episode():
    groups = [[2, 3], [5], [8, 9, 10]]
    returned_indices = np.asarray([2, 3, 5, 8, 9, 10])
    labels = np.asarray([0, 1, 0, 1, 1, 0])
    result = fragmentation_counts(groups, labels, returned_indices)
    assert result["n_returned_fragments"] == 6
    assert result["n_merged_episodes"] == 3
    assert result["n_multifragment_episodes"] == 2
    assert result["n_close_adjacent_pairs"] == 3
    assert result["n_close_opposite_label_pairs"] == 2


def test_no_events_returns_empty_groups():
    assert merge_returned_event_indices(
        np.asarray([0.0]), np.asarray([10.0]), np.asarray([False]), 50.0,
    ) == []


def test_episode_merge_turns_close_detector_fragments_into_one_event():
    events = [
        {"t_on": 100.0, "t_off": 130.0, "dur_ms": 31.0,
         "peak_ext": 0.1, "returned": True},
        {"t_on": 158.0, "t_off": 190.0, "dur_ms": 33.0,
         "peak_ext": 0.2, "returned": True},
        {"t_on": 251.0, "t_off": 270.0, "dur_ms": 20.0,
         "peak_ext": 0.15, "returned": True},
    ]
    merged = merge_detected_event_fragments(
        events, maximum_gap_ms=50.0, sample_dt_ms=1.0,
    )
    assert len(merged) == 2
    assert merged[0]["t_on"] == 100.0
    assert merged[0]["t_off"] == 190.0
    assert merged[0]["dur_ms"] == 91.0
    assert merged[0]["peak_ext"] == 0.2
    assert merged[0]["fragment_count"] == 2
    assert merged[0]["detector_fragment_indices"] == [0, 1]


def test_episode_merge_propagates_nonreturning_fragment():
    events = [
        {"t_on": 0.0, "t_off": 10.0, "dur_ms": 11.0,
         "peak_ext": 0.1, "returned": True},
        {"t_on": 20.0, "t_off": 30.0, "dur_ms": 11.0,
         "peak_ext": 0.2, "returned": False},
    ]
    merged = merge_detected_event_fragments(
        events, maximum_gap_ms=50.0, sample_dt_ms=1.0,
    )
    assert merged[0]["returned"] is False
