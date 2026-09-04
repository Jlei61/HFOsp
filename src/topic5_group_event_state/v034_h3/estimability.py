"""Coverage-segment estimability for H3 exposure -> future-block tests.

Anchor rows are not sample size.  A block is complete only when its entire
exposure and future endpoint fit in one real coverage segment and one phase.
The effective denominator is obtained by interval scheduling of the *combined*
exposure+future support, not by dividing a session duration or counting sliding
anchors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CoveragePiece:
    segment_id: int
    phase: str
    start: float
    stop: float
    coverage_start: float | None = None
    coverage_stop: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.start) or not np.isfinite(self.stop) or self.stop <= self.start:
            raise ValueError("coverage piece must have finite start < stop")
        cstart = self.start if self.coverage_start is None else self.coverage_start
        cstop = self.stop if self.coverage_stop is None else self.coverage_stop
        if cstart > self.start or cstop < self.stop:
            raise ValueError("phase piece must lie inside its real coverage segment")

    @property
    def full_start(self) -> float:
        return float(self.start if self.coverage_start is None else self.coverage_start)

    @property
    def full_stop(self) -> float:
        return float(self.stop if self.coverage_stop is None else self.coverage_stop)


@dataclass(frozen=True)
class CompleteBlock:
    segment_id: int
    phase: str
    exposure_start: float
    boundary: float
    future_stop: float
    exposure_lo_event: int | None = None
    exposure_hi_event: int | None = None

    @property
    def support_start(self) -> float:
        return self.exposure_start

    @property
    def support_stop(self) -> float:
        return self.future_stop


@dataclass(frozen=True)
class BlockSupport:
    exposure_kind: str
    exposure_value: float
    future_seconds: float
    phase: str
    n_complete_candidates: int
    n_nonoverlap_blocks: int
    boundary_span_seconds: float
    support_span_seconds: float
    min_required_blocks: int
    estimable: bool
    core_eligible: bool
    tier: str
    reasons: tuple[str, ...]
    nonoverlap_blocks: tuple[CompleteBlock, ...]

    def as_dict(self, *, include_blocks: bool = False) -> dict:
        out = asdict(self)
        if not include_blocks:
            out.pop("nonoverlap_blocks", None)
        return out


def _validate_events(event_times: np.ndarray, event_segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(event_times, dtype=np.float64)
    segments = np.asarray(event_segments, dtype=np.int64)
    if times.ndim != 1 or segments.shape != times.shape:
        raise ValueError("event_times and event_segments must be aligned 1-D arrays")
    if times.size and np.any(np.diff(times) < 0):
        raise ValueError("event_times must be sorted")
    return times, segments


def _greedy_nonoverlap(blocks: Iterable[CompleteBlock]) -> tuple[CompleteBlock, ...]:
    """Maximum-cardinality non-overlap set for intervals, by earliest finish."""

    chosen: list[CompleteBlock] = []
    last_stop = -np.inf
    for block in sorted(blocks, key=lambda b: (b.support_stop, b.support_start)):
        if block.support_start >= last_stop:
            chosen.append(block)
            last_stop = block.support_stop
    return tuple(chosen)


def _support_summary(
    blocks: Sequence[CompleteBlock], *, exposure_kind: str, exposure_value: float,
    future_seconds: float, phase: str, min_blocks: int, core_horizon_limit_seconds: float,
) -> BlockSupport:
    nonoverlap = _greedy_nonoverlap(blocks)
    boundaries = np.asarray([b.boundary for b in blocks], dtype=np.float64)
    support_start = min((b.support_start for b in blocks), default=np.nan)
    support_stop = max((b.support_stop for b in blocks), default=np.nan)
    reasons: list[str] = []
    if len(nonoverlap) < int(min_blocks):
        reasons.append(f"independent_blocks={len(nonoverlap)}<{int(min_blocks)}")
    tier = "core" if float(future_seconds) <= float(core_horizon_limit_seconds) else "exploratory_long_horizon"
    if tier != "core":
        reasons.append(
            f"future_horizon={float(future_seconds):g}s>{float(core_horizon_limit_seconds):g}s_core_limit"
        )
    estimable = len(nonoverlap) >= int(min_blocks)
    return BlockSupport(
        exposure_kind=exposure_kind,
        exposure_value=float(exposure_value),
        future_seconds=float(future_seconds),
        phase=str(phase),
        n_complete_candidates=len(blocks),
        n_nonoverlap_blocks=len(nonoverlap),
        boundary_span_seconds=(float(np.ptp(boundaries)) if boundaries.size > 1 else 0.0),
        support_span_seconds=(float(support_stop - support_start) if blocks else 0.0),
        min_required_blocks=int(min_blocks),
        estimable=bool(estimable),
        core_eligible=bool(estimable and tier == "core"),
        tier=tier,
        reasons=tuple(reasons),
        nonoverlap_blocks=nonoverlap,
    )


def audit_event_count_design(
    event_times: np.ndarray,
    event_segments: np.ndarray,
    pieces: Sequence[CoveragePiece],
    *,
    n_events: int,
    future_seconds: float,
    min_blocks_by_phase: dict[str, int] | None = None,
    core_horizon_limit_seconds: float = 1800.0,
) -> dict[str, BlockSupport]:
    """Audit an N-event exposure followed by a physical-time future block.

    The event at ``boundary`` belongs to the future, not the exposure.  All N
    earlier events, the boundary and the endpoint must remain in one coverage
    piece.  This exactly prevents an event-session label from bridging a real
    recording gap.
    """

    times, segments = _validate_events(event_times, event_segments)
    n_events = int(n_events)
    if n_events <= 0 or future_seconds <= 0:
        raise ValueError("n_events and future_seconds must be positive")
    rules = min_blocks_by_phase or {p.phase: 1 for p in pieces}
    by_phase: dict[str, list[CompleteBlock]] = {phase: [] for phase in rules}
    for piece in pieces:
        idx = np.flatnonzero(
            (segments == int(piece.segment_id))
            & (times >= piece.full_start)
            & (times < piece.full_stop)
        )
        if idx.size <= n_events:
            continue
        # j is a position within this segment/phase event list.
        for j in range(n_events, idx.size):
            hi = int(idx[j])
            lo = int(idx[j - n_events])
            boundary = float(times[hi])
            if boundary < float(piece.start):
                continue
            future_stop = boundary + float(future_seconds)
            if future_stop > float(piece.stop) + 1e-9:
                break
            by_phase.setdefault(piece.phase, []).append(CompleteBlock(
                segment_id=int(piece.segment_id), phase=piece.phase,
                exposure_start=float(times[lo]), boundary=boundary, future_stop=future_stop,
                exposure_lo_event=lo, exposure_hi_event=hi,
            ))
    return {
        phase: _support_summary(
            by_phase.get(phase, ()), exposure_kind="event_count", exposure_value=n_events,
            future_seconds=future_seconds, phase=phase, min_blocks=int(min_blocks),
            core_horizon_limit_seconds=core_horizon_limit_seconds,
        )
        for phase, min_blocks in rules.items()
    }


def audit_physical_window_design(
    pieces: Sequence[CoveragePiece],
    *,
    exposure_seconds: float,
    future_seconds: float,
    anchor_step_seconds: float = 300.0,
    min_blocks_by_phase: dict[str, int] | None = None,
    core_horizon_limit_seconds: float = 1800.0,
) -> dict[str, BlockSupport]:
    """Audit physical boxcar exposure + future blocks inside true coverage."""

    if exposure_seconds <= 0 or future_seconds <= 0 or anchor_step_seconds <= 0:
        raise ValueError("window lengths and anchor step must be positive")
    rules = min_blocks_by_phase or {p.phase: 1 for p in pieces}
    by_phase: dict[str, list[CompleteBlock]] = {phase: [] for phase in rules}
    for piece in pieces:
        first = max(float(piece.start), piece.full_start + float(exposure_seconds))
        last = float(piece.stop) - float(future_seconds)
        if last < first:
            continue
        n = int(np.floor((last - first) / float(anchor_step_seconds))) + 1
        for k in range(n):
            boundary = first + k * float(anchor_step_seconds)
            by_phase.setdefault(piece.phase, []).append(CompleteBlock(
                segment_id=int(piece.segment_id), phase=piece.phase,
                exposure_start=boundary - float(exposure_seconds), boundary=boundary,
                future_stop=boundary + float(future_seconds),
            ))
    return {
        phase: _support_summary(
            by_phase.get(phase, ()), exposure_kind="physical_seconds",
            exposure_value=exposure_seconds, future_seconds=future_seconds,
            phase=phase, min_blocks=int(min_blocks),
            core_horizon_limit_seconds=core_horizon_limit_seconds,
        )
        for phase, min_blocks in rules.items()
    }
