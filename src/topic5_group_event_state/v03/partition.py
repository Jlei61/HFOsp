"""Nested physical-time partition for the v0.3 pilot.

The old three-way ``TRAIN/VAL/TEST`` split is not sufficient when a patient
specific contact grammar is fitted before the cross-event state model.  This
module makes the four causal stages explicit and locates every boundary on the
cumulative *recorded-time* axis, so gaps do not consume a split fraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.topic5_group_event_state.v02.timeline import CoverageSegment


PHASE_NAMES = ("calibration", "state_train", "dev_val", "dev_test")
NESTED_FRACTIONS = (0.20, 0.50, 0.10, 0.20)
GRAMMAR_FIT_FRACTION = 0.16


def recorded_epoch_at_fraction(
    segments: Sequence[CoverageSegment], fraction: float
) -> float:
    """Map a cumulative recorded-time fraction back to an absolute epoch."""

    if not segments:
        raise ValueError("cannot partition empty coverage")
    value = float(fraction)
    if not 0.0 <= value <= 1.0:
        raise ValueError("recorded-time fraction must lie in [0,1]")
    starts = np.asarray([s.start_epoch for s in segments], dtype=np.float64)
    durations = np.asarray([s.duration_seconds for s in segments], dtype=np.float64)
    cumulative = np.cumsum(durations)
    before = cumulative - durations
    target = float(cumulative[-1]) * value
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(index, len(segments) - 1)
    return float(starts[index] + target - before[index])


@dataclass(frozen=True)
class NestedTimePartition:
    """Calibration -> state fit -> development validation -> development test."""

    boundary_epochs: np.ndarray  # shape (3,)
    grammar_fit_stop_epoch: float
    recorded_seconds: dict[str, float]
    total_recorded_seconds: float
    fractions: tuple[float, float, float, float]

    def labels_of(self, epochs: np.ndarray) -> np.ndarray:
        values = np.asarray(epochs, dtype=np.float64)
        return (
            (values >= self.boundary_epochs[0]).astype(np.int64)
            + (values >= self.boundary_epochs[1]).astype(np.int64)
            + (values >= self.boundary_epochs[2]).astype(np.int64)
        )

    def bounds(self, phase: str) -> tuple[float, float]:
        if phase not in PHASE_NAMES:
            raise ValueError(f"unknown nested phase {phase!r}")
        index = PHASE_NAMES.index(phase)
        lower = -np.inf if index == 0 else float(self.boundary_epochs[index - 1])
        upper = np.inf if index == 3 else float(self.boundary_epochs[index])
        return lower, upper


def nested_time_partition(
    segments: Sequence[CoverageSegment],
    fractions: Sequence[float] = NESTED_FRACTIONS,
) -> NestedTimePartition:
    """Build the pre-registered 20/50/10/20 recorded-time partition."""

    if len(fractions) != 4 or abs(sum(float(v) for v in fractions) - 1.0) > 1e-9:
        raise ValueError("need four nested fractions summing to one")
    values = tuple(float(v) for v in fractions)
    total = float(sum(s.duration_seconds for s in segments))
    cumulative = np.cumsum(values)[:-1]
    boundaries = np.asarray(
        [recorded_epoch_at_fraction(segments, value) for value in cumulative],
        dtype=np.float64,
    )
    return NestedTimePartition(
        boundary_epochs=boundaries,
        grammar_fit_stop_epoch=recorded_epoch_at_fraction(
            segments, GRAMMAR_FIT_FRACTION
        ),
        recorded_seconds={
            name: total * fraction for name, fraction in zip(PHASE_NAMES, values)
        },
        total_recorded_seconds=total,
        fractions=values,
    )


def positions_for_phase(
    event_times: np.ndarray, stream_positions: np.ndarray,
    partition: NestedTimePartition, phase: str,
) -> np.ndarray:
    """Positions into the source sequence for one nested phase."""

    phase_index = PHASE_NAMES.index(phase)
    mask = partition.labels_of(event_times) == phase_index
    return np.asarray(stream_positions, dtype=np.int64)[np.flatnonzero(mask)]

