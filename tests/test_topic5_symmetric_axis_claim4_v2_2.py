from __future__ import annotations

import numpy as np
import pytest

from src.topic5_symmetric_axis_claim4_v2_2 import (
    SOURCE_LEFT,
    SOURCE_MIDDLE,
    SOURCE_RIGHT,
    event_source_projection,
    meets_claim4_event_thresholds,
    partition_source_sides,
    side_event_indices,
)


def _data() -> tuple[np.ndarray, np.ndarray]:
    coords = np.column_stack(
        [np.arange(8, dtype=float), np.zeros(8), np.zeros(8)]
    )
    source_contact = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 7])
    groups = np.full((len(source_contact), 8), -1, dtype=np.int64)
    for event, source in enumerate(source_contact):
        groups[event, source] = 0
        groups[event, (source + 1) % 8] = 1
    return groups, coords


def test_source_projection_uses_first_rank_set_centroid() -> None:
    groups, coords = _data()
    groups[0, 0] = 0
    groups[0, 2] = 0
    projection = event_source_projection(
        groups=groups,
        coords=coords,
        axis=np.array([1.0, 0.0, 0.0]),
    )
    assert projection[0] == pytest.approx(1.0 - 3.5)


def test_heldout_labels_use_train_only_thresholds() -> None:
    groups, coords = _data()
    train = np.arange(8)
    heldout = np.arange(8, 10)
    partition = partition_source_sides(
        groups=groups,
        coords=coords,
        axis=np.array([1.0, 0.0, 0.0]),
        train_indices=train,
        heldout_indices=heldout,
    )
    assert partition.status == "ok"
    assert partition.q25 == pytest.approx(-1.75)
    assert partition.q75 == pytest.approx(1.75)
    assert partition.heldout_side.tolist() == [SOURCE_LEFT, SOURCE_RIGHT]
    left = side_event_indices(
        partition_indices=heldout,
        side=partition.heldout_side,
        wanted=SOURCE_LEFT,
    )
    assert left.tolist() == [8]


def test_equal_train_quantiles_are_not_estimable() -> None:
    groups, coords = _data()
    groups[:8, :] = -1
    groups[:8, 4] = 0
    partition = partition_source_sides(
        groups=groups,
        coords=coords,
        axis=np.array([1.0, 0.0, 0.0]),
        train_indices=np.arange(8),
        heldout_indices=np.arange(8, 10),
    )
    assert partition.status == "not_estimable_equal_quantiles"
    assert np.all(partition.train_side == SOURCE_MIDDLE)
    assert not meets_claim4_event_thresholds(
        partition, min_train_per_side=1, min_heldout_per_side=1
    )


def test_train_and_heldout_overlap_is_rejected() -> None:
    groups, coords = _data()
    with pytest.raises(ValueError, match="overlap"):
        partition_source_sides(
            groups=groups,
            coords=coords,
            axis=np.array([1.0, 0.0, 0.0]),
            train_indices=np.arange(8),
            heldout_indices=np.array([7, 8]),
        )
