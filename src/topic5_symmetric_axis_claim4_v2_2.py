"""Leakage-safe source-side utilities for formal v2.2 Claim 4."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.topic5_symmetric_axis_propagation_state_v2_2 import (
    canonicalize_axis,
    train_only_source_side_thresholds,
)


SOURCE_LEFT = -1
SOURCE_MIDDLE = 0
SOURCE_RIGHT = 1


@dataclass(frozen=True)
class SourceSidePartition:
    train_projection: np.ndarray
    heldout_projection: np.ndarray
    train_side: np.ndarray
    heldout_side: np.ndarray
    q25: float
    q75: float
    status: str

    def counts(self) -> dict[str, int]:
        return {
            "train_left": int(np.sum(self.train_side == SOURCE_LEFT)),
            "train_right": int(np.sum(self.train_side == SOURCE_RIGHT)),
            "heldout_left": int(np.sum(self.heldout_side == SOURCE_LEFT)),
            "heldout_right": int(np.sum(self.heldout_side == SOURCE_RIGHT)),
        }


def event_source_projection(
    *,
    groups: np.ndarray,
    coords: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    """Project each observed first rank-set centroid onto one physical axis."""
    groups = np.asarray(groups, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.float64)
    if groups.ndim != 2:
        raise ValueError("groups must have shape [event, contact]")
    if coords.shape != (groups.shape[1], 3) or not np.all(np.isfinite(coords)):
        raise ValueError("coords must be finite and align with contacts")
    axis_unit = canonicalize_axis(axis)
    centered = coords - np.mean(coords, axis=0, keepdims=True)
    first = groups == 0
    first_count = first.sum(axis=1)
    if np.any(first_count == 0):
        raise ValueError("every event must contain an observed first rank set")
    centroid = (first[:, :, None] * centered[None, :, :]).sum(axis=1)
    centroid /= first_count[:, None]
    return centroid @ axis_unit


def _apply_thresholds(values: np.ndarray, q25: float, q75: float) -> np.ndarray:
    side = np.full(len(values), SOURCE_MIDDLE, dtype=np.int8)
    side[values <= q25] = SOURCE_LEFT
    side[values >= q75] = SOURCE_RIGHT
    return side


def partition_source_sides(
    *,
    groups: np.ndarray,
    coords: np.ndarray,
    axis: np.ndarray,
    train_indices: np.ndarray,
    heldout_indices: np.ndarray,
) -> SourceSidePartition:
    """Freeze source-side thresholds on train80 and apply them to heldout20."""
    train_indices = np.asarray(train_indices, dtype=np.int64)
    heldout_indices = np.asarray(heldout_indices, dtype=np.int64)
    if train_indices.ndim != 1 or heldout_indices.ndim != 1:
        raise ValueError("partition indices must be one-dimensional")
    if np.intersect1d(train_indices, heldout_indices).size:
        raise ValueError("train and heldout event partitions overlap")
    projection = event_source_projection(
        groups=groups,
        coords=coords,
        axis=axis,
    )
    train_projection = projection[train_indices]
    heldout_projection = projection[heldout_indices]
    thresholds = train_only_source_side_thresholds(train_projection)
    q25 = float(thresholds["left_max"])
    q75 = float(thresholds["right_min"])
    if q25 == q75:
        return SourceSidePartition(
            train_projection=train_projection,
            heldout_projection=heldout_projection,
            train_side=np.full(
                len(train_projection), SOURCE_MIDDLE, dtype=np.int8
            ),
            heldout_side=np.full(
                len(heldout_projection), SOURCE_MIDDLE, dtype=np.int8
            ),
            q25=q25,
            q75=q75,
            status="not_estimable_equal_quantiles",
        )
    return SourceSidePartition(
        train_projection=train_projection,
        heldout_projection=heldout_projection,
        train_side=_apply_thresholds(train_projection, q25, q75),
        heldout_side=_apply_thresholds(heldout_projection, q25, q75),
        q25=q25,
        q75=q75,
        status="ok",
    )


def side_event_indices(
    *,
    partition_indices: np.ndarray,
    side: np.ndarray,
    wanted: int,
) -> np.ndarray:
    """Return global event indices for one already-frozen source side."""
    partition_indices = np.asarray(partition_indices, dtype=np.int64)
    side = np.asarray(side, dtype=np.int8)
    if partition_indices.shape != side.shape:
        raise ValueError("partition indices and side labels must align")
    if wanted not in (SOURCE_LEFT, SOURCE_RIGHT):
        raise ValueError("wanted side must be SOURCE_LEFT or SOURCE_RIGHT")
    return partition_indices[side == wanted]


def meets_claim4_event_thresholds(
    partition: SourceSidePartition,
    *,
    min_train_per_side: int = 100,
    min_heldout_per_side: int = 25,
) -> bool:
    if partition.status != "ok":
        return False
    counts = partition.counts()
    return bool(
        counts["train_left"] >= min_train_per_side
        and counts["train_right"] >= min_train_per_side
        and counts["heldout_left"] >= min_heldout_per_side
        and counts["heldout_right"] >= min_heldout_per_side
    )
