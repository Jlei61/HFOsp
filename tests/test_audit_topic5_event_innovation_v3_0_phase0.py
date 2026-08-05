from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_topic5_event_innovation_v3_0_phase0 import (
    chronological_split_indices,
)


def test_chronological_split_is_ordered_disjoint_and_complete():
    eligible = np.arange(1000, dtype=np.int64)
    split = chronological_split_indices(eligible, [0.6, 0.2, 0.2], minimum_events=120)
    assert [len(split[key]) for key in ("train", "validation", "test")] == [600, 200, 200]
    assert split["train"][-1] < split["validation"][0] < split["test"][0]
    assert np.array_equal(
        np.concatenate([split["train"], split["validation"], split["test"]]),
        eligible,
    )


def test_chronological_split_adjusts_cut_without_breaking_minimum():
    eligible = np.arange(360, dtype=np.int64)
    split = chronological_split_indices(eligible, [0.8, 0.1, 0.1], minimum_events=100)
    assert [len(split[key]) for key in ("train", "validation", "test")] == [160, 100, 100]


def test_chronological_split_rejects_nonchronological_or_too_small_input():
    with pytest.raises(ValueError):
        chronological_split_indices(np.asarray([0, 2, 1]), [0.6, 0.2, 0.2], minimum_events=1)
    with pytest.raises(ValueError):
        chronological_split_indices(np.arange(100), [0.6, 0.2, 0.2], minimum_events=40)

