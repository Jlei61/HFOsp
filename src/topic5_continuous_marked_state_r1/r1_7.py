"""Frozen support utilities for prospective R1.7A development replication."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coverage import CoverageTable


R1_7A_REVISION = "r1_7a_prospective_state_replication_v1"

# The frozen optimiser refuses to step on a non-finite gradient.  Such a cell is
# an instrument failure for that seed, not a scientific negative and not a
# silently dropped patient: it is recorded, kept in the five-seed denominator,
# and can never be counted as a stable checkpoint.  Every other failure -- shape,
# alignment, checkpoint, memory -- must still abort the cell.
NONFINITE_GRADIENT_STATUS = "NONFINITE_GRADIENT"
NONFINITE_GRADIENT_MESSAGES = (
    "R1.3 encountered a non-finite gradient norm",
    "R1.2 prefix encountered a non-finite gradient",
)


def is_nonfinite_gradient_failure(error: BaseException) -> bool:
    """Return true only for the frozen optimiser's own non-finite gradient guard."""
    return (
        isinstance(error, RuntimeError)
        and str(error) in NONFINITE_GRADIENT_MESSAGES
    )


def split_scored_payloads(
    payloads: list[dict],
) -> tuple[list[dict], int]:
    """Separate scorable seeds from recorded non-finite optimiser failures."""
    scored = [
        value for value in payloads
        if value.get("analysis_status") != NONFINITE_GRADIENT_STATUS
    ]
    return scored, len(payloads) - len(scored)


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


def coverage_segment_for_times(
    coverage: CoverageTable, event_time: np.ndarray
) -> np.ndarray:
    """Assign events to recorded coverage intervals, failing on gaps."""
    time = np.asarray(event_time, dtype=np.float64)
    segment = np.searchsorted(coverage.stop, time, side="right")
    safe = np.minimum(segment, len(coverage.start) - 1)
    valid = (
        (segment < len(coverage.start))
        & (time >= np.asarray(coverage.start)[safe])
        & (time < np.asarray(coverage.stop)[safe])
    )
    if not bool(np.all(valid)):
        raise ValueError("event time falls outside recorded coverage")
    return segment.astype(np.int64)


def complete_event_blocks_by_segment(
    event_segment: np.ndarray,
    keep: np.ndarray,
    *,
    block_events: int = 100,
) -> tuple[int, list[dict[str, int]]]:
    """Count complete event blocks without ever pooling across recording gaps."""
    segment = np.asarray(event_segment, dtype=np.int64)
    keep = np.asarray(keep, dtype=bool)
    if segment.shape != keep.shape or int(block_events) < 1:
        raise ValueError("event-block arrays disagree")
    rows = []
    total = 0
    for label in np.unique(segment[keep]):
        events = int(np.sum(keep & (segment == label)))
        blocks = events // int(block_events)
        rows.append({"segment": int(label), "events": events, "complete_blocks": blocks})
        total += blocks
    return int(total), rows
