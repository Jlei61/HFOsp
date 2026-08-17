import numpy as np

from src.topic5_event_innovation_bootstrap_v3_0 import (
    moving_block_resamples,
    observable_gain_sufficient_statistics,
    standardized_propagation_gain,
)
from src.topic5_event_innovation_response_v3_0 import observable_propagation_gain
from src.topic5_event_innovation_v3_0 import RankStateBasis


def test_moving_block_resample_preserves_group_counts_and_membership():
    group = np.array([0, 0, 0, 0, 1, 1, 1])
    event = np.array([10, 11, 12, 13, 30, 31, 32])
    draws = list(moving_block_resamples(
        group, event, block_length=2, draws=5, seed=4
    ))
    assert len(draws) == 5
    for selected in draws:
        assert len(selected) == len(group)
        assert np.sum(group[selected] == 0) == 4
        assert np.sum(group[selected] == 1) == 3


def test_moving_blocks_do_not_bridge_missing_event_indices():
    group = np.zeros(4, dtype=int)
    event = np.array([10, 11, 20, 21])
    for selected in moving_block_resamples(
        group, event, block_length=3, draws=10, seed=9
    ):
        # Each two-row contiguous segment is resampled independently, so each
        # side of the gap keeps exactly two selected rows.
        assert np.sum(selected < 2) == 2
        assert np.sum(selected >= 2) == 2


def test_sufficient_statistics_reproduce_full_observable_gain():
    basis = RankStateBasis(
        backbone=np.zeros(3), loadings=np.eye(3), singular_values=np.ones(3)
    )
    ranks = np.array([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]])
    participation = np.ones_like(ranks, dtype=bool)
    observed = ranks.copy()
    support = np.ones_like(ranks)
    windows = [np.array([0]), np.array([1])]
    automatic = np.zeros((2, 3))
    driven = np.array([[0.0, 0.4, 0.9], [0.9, 0.4, 0.0]])
    exact = observable_propagation_gain(
        basis,
        observed,
        support,
        windows,
        ranks,
        participation,
        ranks,
        automatic,
        driven,
    )
    statistics = observable_gain_sufficient_statistics(
        basis,
        observed,
        support,
        windows,
        ranks,
        participation,
        ranks,
        automatic,
        driven,
    )
    reproduced = standardized_propagation_gain(
        statistics,
        np.arange(2),
        rank_scale=1.0,
        pair_scale=1.0,
    )
    assert np.isclose(reproduced, exact["propagation_gain"])
