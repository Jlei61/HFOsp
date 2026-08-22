import numpy as np

from scripts.run_topic4_rev12_node_intervention import (
    _rank_distance,
    _select_branch_event,
)


def test_branch_event_is_nearest_returned_event_within_frozen_window():
    events = [
        {"t_on": 10.0, "returned": False},
        {"t_on": 35.0, "returned": True},
        {"t_on": 80.0, "returned": True},
    ]
    assert _select_branch_event(events, 40.0, 20.0)["t_on"] == 35.0
    assert _select_branch_event(events, 120.0, 20.0) is None


def test_rank_distance_is_zero_for_identical_missing_contact_pattern():
    ranks = np.asarray([0.0, 1.0, np.nan, 2.0])
    names = np.asarray(["ICL1", "ICL2", "SCL1", "SCL2"])
    assert _rank_distance(ranks, ranks, names) == 0.0


def test_rank_distance_penalizes_recruitment_and_order_change():
    left = np.asarray([0.0, 1.0, np.nan, 2.0])
    right = np.asarray([2.0, 1.0, 0.0, np.nan])
    names = np.asarray(["ICL1", "ICL2", "SCL1", "SCL2"])
    assert _rank_distance(left, right, names) > 0.0
