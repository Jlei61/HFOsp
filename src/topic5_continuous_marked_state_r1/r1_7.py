"""Frozen support utilities for prospective R1.7A development replication."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coverage import CoverageTable


R1_7A_REVISION = "r1_7a_prospective_state_replication_v1"


@dataclass(frozen=True)
class RecordedValidationSplit:
    state_start: float
    state_stop: float
    mechanism_start: float
    mechanism_stop: float
    state_recorded_seconds: float
    mechanism_recorded_seconds: float
    total_recorded_seconds: float


def split_validation_by_recorded_time(
    coverage: CoverageTable,
    *,
    validation_start: float,
    validation_stop: float,
    state_fraction: float = 0.60,
) -> RecordedValidationSplit:
    """Split validation at 60% of recorded support, never wall-clock gaps."""
    if not 0.0 < float(state_fraction) < 1.0:
        raise ValueError("state_fraction must lie strictly between zero and one")
    start = np.maximum(np.asarray(coverage.start, dtype=np.float64), validation_start)
    stop = np.minimum(np.asarray(coverage.stop, dtype=np.float64), validation_stop)
    keep = stop > start
    start, stop = start[keep], stop[keep]
    if not len(start):
        raise ValueError("validation has no recorded support")
    duration = stop - start
    total = float(duration.sum())
    target = float(state_fraction) * total
    cumulative = np.cumsum(duration)
    index = int(np.searchsorted(cumulative, target, side="left"))
    before = float(cumulative[index - 1]) if index else 0.0
    boundary = float(start[index] + (target - before))
    if not validation_start < boundary < validation_stop:
        raise ValueError("recorded-time split did not create two nonempty layers")
    return RecordedValidationSplit(
        state_start=float(validation_start), state_stop=boundary,
        mechanism_start=boundary, mechanism_stop=float(validation_stop),
        state_recorded_seconds=target,
        mechanism_recorded_seconds=total - target,
        total_recorded_seconds=total,
    )


def block_bootstrap_length_seconds(
    train_event_time: np.ndarray,
    train_session: np.ndarray,
    *,
    minimum_seconds: float = 1800.0,
    maximum_seconds: float = 21600.0,
) -> float:
    """Freeze a conservative time-block length from TRAIN event intervals only."""
    time = np.asarray(train_event_time, dtype=np.float64)
    session = np.asarray(train_session)
    intervals = []
    for label in np.unique(session):
        value = np.sort(time[session == label])
        if len(value) >= 3:
            delta = np.diff(value)
            delta = delta[np.isfinite(delta) & (delta > 0)]
            intervals.extend(delta.tolist())
    if not intervals:
        return float(minimum_seconds)
    # About 100 median event gaps, bounded to a scientifically readable scale.
    candidate = 100.0 * float(np.median(intervals))
    return float(np.clip(candidate, minimum_seconds, maximum_seconds))
