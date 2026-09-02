"""v0.3.2 nested partition: base_fit / inner_val / dev_val / dev_test.

Boundaries live on the cumulative *recorded-time* axis (gaps cost nothing), at
0.60 / 0.70 / 0.80.  ``base_refit`` is the union of ``base_fit`` and
``inner_val`` and is only ever used to refit an already-selected configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.topic5_group_event_state.v02.timeline import CoverageSegment
from src.topic5_group_event_state.v03.partition import recorded_epoch_at_fraction

EVAL_PHASES: tuple[str, ...] = ("base_fit", "inner_val", "dev_val", "dev_test")
REFIT_PHASE = "base_refit"
BOUNDARY_FRACTIONS: tuple[float, float, float] = (0.60, 0.70, 0.80)


@dataclass(frozen=True)
class EvalPartition:
    boundary_epochs: np.ndarray            # shape (3,)
    recorded_seconds: dict[str, float]
    total_recorded_seconds: float
    fractions: tuple[float, float, float]

    def labels_of(self, epochs: np.ndarray) -> np.ndarray:
        values = np.asarray(epochs, dtype=np.float64)
        return (
            (values >= self.boundary_epochs[0]).astype(np.int64)
            + (values >= self.boundary_epochs[1]).astype(np.int64)
            + (values >= self.boundary_epochs[2]).astype(np.int64)
        )

    def phase_of(self, epoch: float) -> str:
        return EVAL_PHASES[int(self.labels_of(np.asarray([epoch]))[0])]

    @staticmethod
    def phase_index(phase: str) -> int:
        if phase not in EVAL_PHASES:
            raise ValueError(f"unknown v0.3.2 phase {phase!r}")
        return EVAL_PHASES.index(phase)

    def mask_for_phase(self, epochs: np.ndarray, phase: str) -> np.ndarray:
        labels = self.labels_of(epochs)
        if phase == REFIT_PHASE:
            return labels <= 1
        return labels == self.phase_index(phase)

    def bounds(self, phase: str) -> tuple[float, float]:
        if phase == REFIT_PHASE:
            return -np.inf, float(self.boundary_epochs[1])
        index = self.phase_index(phase)
        lower = -np.inf if index == 0 else float(self.boundary_epochs[index - 1])
        upper = np.inf if index == 3 else float(self.boundary_epochs[index])
        return lower, upper

    def window_within_phase(self, start_epochs: np.ndarray, horizon: float) -> np.ndarray:
        """True when ``[t, t+h)`` starts and ends inside the same phase."""

        start = np.asarray(start_epochs, dtype=np.float64)
        stop = np.nextafter(start + float(horizon), -np.inf)
        return self.labels_of(start) == self.labels_of(stop)

    def as_dict(self) -> dict:
        return {
            "phase_names": list(EVAL_PHASES),
            "refit_phase": REFIT_PHASE,
            "boundary_fractions": list(self.fractions),
            "boundary_epochs": [float(v) for v in self.boundary_epochs],
            "recorded_seconds": dict(self.recorded_seconds),
            "total_recorded_seconds": float(self.total_recorded_seconds),
        }


def eval_partition(
    segments: Sequence[CoverageSegment],
    fractions: Sequence[float] = BOUNDARY_FRACTIONS,
) -> EvalPartition:
    """Build the frozen 60/70/80 recorded-time partition."""

    values = tuple(float(v) for v in fractions)
    if len(values) != 3 or not (0.0 < values[0] < values[1] < values[2] < 1.0):
        raise ValueError("need three increasing boundary fractions inside (0,1)")
    total = float(sum(s.duration_seconds for s in segments))
    boundaries = np.asarray(
        [recorded_epoch_at_fraction(segments, v) for v in values], dtype=np.float64
    )
    edges = (0.0,) + values + (1.0,)
    recorded = {
        name: total * (edges[i + 1] - edges[i]) for i, name in enumerate(EVAL_PHASES)
    }
    recorded[REFIT_PHASE] = total * values[1]
    return EvalPartition(
        boundary_epochs=boundaries,
        recorded_seconds=recorded,
        total_recorded_seconds=total,
        fractions=values,
    )
