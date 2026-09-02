"""Merging events, background cells and anchors into one causal timeline.

One rollout unit = one coverage segment (after the gap, seizure-onset and
postictal cuts).  The train/validation/test boundary is *not* a rollout boundary:
the recording is physically continuous across it and a state that reset there
would be judged on a warm-up it never had.  The boundary is enforced where it
belongs -- on which blocks may exist -- not on the state chain.

At an instant shared by several step kinds the order is cell, then anchor, then
event.  That ordering is the causal contract in one line: the background cell has
to be in force before anything reads it, and an anchor must be read before an
event standing on the same instant, because such an event belongs to the block
being predicted rather than to the history predicting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .background import CELL_SECONDS, cell_background, clock_features
from .models import KIND_ANCHOR, KIND_CELL, KIND_EVENT
from .support import ANCHOR_GRID_MINUTES, Interval, segment_anchor_grid, segment_bounds


@dataclass
class SegmentTimeline:
    """One coverage segment, flattened into ordered steps."""

    segment_id: int
    start: float
    stop: float
    step_time: np.ndarray        # (S,) float64
    step_kind: np.ndarray        # (S,) int8
    step_cell: np.ndarray        # (S,) int64  -- which background cell is in force
    event_row: np.ndarray        # (S,) int64  -- index into the event stream, -1 otherwise
    anchor_time: np.ndarray      # (A,) float64
    anchor_step: np.ndarray      # (A,) int64  -- position in step_time
    cell_features: np.ndarray    # (K, D) float32 -- background+clock per cell
    cell_valid: np.ndarray       # (K,) bool

    @property
    def n_steps(self) -> int:
        return int(self.step_time.size)


def build_segment_timeline(
    segment_id: int,
    start: float,
    stop: float,
    event_times: np.ndarray,
    anchor_background: tuple[np.ndarray, np.ndarray],
    recording_start: float,
    recording_stop: float,
    *,
    cell_seconds: float = CELL_SECONDS,
    anchor_minutes: int = ANCHOR_GRID_MINUTES,
) -> SegmentTimeline:
    """Flatten one segment into (cell | anchor | event) steps in causal order."""

    grid = segment_anchor_grid(start, stop, minutes=anchor_minutes)
    n_cells = int(np.floor((stop - start) / cell_seconds)) + 1
    cell_starts = start + cell_seconds * np.arange(n_cells, dtype=np.float64)

    bg_time, bg_features = anchor_background
    bg, valid = cell_background(cell_starts, bg_time, bg_features, cell_seconds=cell_seconds)
    clock = clock_features(cell_starts, recording_start, recording_stop)
    cell_features = np.concatenate([bg, clock], axis=1).astype(np.float32)

    lo = np.searchsorted(event_times, start, side="left")
    hi = np.searchsorted(event_times, stop, side="left")
    ev_rows = np.arange(lo, hi, dtype=np.int64)
    ev_times = event_times[lo:hi]

    times = np.concatenate([cell_starts, grid, ev_times])
    kinds = np.concatenate(
        [
            np.full(cell_starts.size, KIND_CELL, dtype=np.int8),
            np.full(grid.size, KIND_ANCHOR, dtype=np.int8),
            np.full(ev_times.size, KIND_EVENT, dtype=np.int8),
        ]
    )
    rows = np.concatenate(
        [
            np.full(cell_starts.size, -1, dtype=np.int64),
            np.full(grid.size, -1, dtype=np.int64),
            ev_rows,
        ]
    )
    # Stable lexsort: primary key time, secondary key kind.  ``np.lexsort`` reads
    # its keys last-to-first, so ``times`` must come last.
    order = np.lexsort((kinds, times))
    times, kinds, rows = times[order], kinds[order], rows[order]

    step_cell = np.clip(
        np.floor((times - start) / cell_seconds).astype(np.int64), 0, n_cells - 1
    )
    anchor_step = np.flatnonzero(kinds == KIND_ANCHOR).astype(np.int64)
    return SegmentTimeline(
        segment_id=int(segment_id),
        start=float(start),
        stop=float(stop),
        step_time=times.astype(np.float64),
        step_kind=kinds,
        step_cell=step_cell,
        event_row=rows,
        anchor_time=times[anchor_step].astype(np.float64),
        anchor_step=anchor_step,
        cell_features=cell_features,
        cell_valid=valid,
    )


def build_timelines(
    intervals: Sequence[Interval],
    event_times: np.ndarray,
    anchor_background: tuple[np.ndarray, np.ndarray],
    *,
    cell_seconds: float = CELL_SECONDS,
    anchor_minutes: int = ANCHOR_GRID_MINUTES,
) -> list[SegmentTimeline]:
    bounds = segment_bounds(intervals)
    if not bounds:
        return []
    recording_start = min(lo for lo, _ in bounds.values())
    recording_stop = max(hi for _, hi in bounds.values())
    return [
        build_segment_timeline(
            segment_id,
            lo,
            hi,
            event_times,
            anchor_background,
            recording_start,
            recording_stop,
            cell_seconds=cell_seconds,
            anchor_minutes=anchor_minutes,
        )
        for segment_id, (lo, hi) in sorted(bounds.items())
    ]


def label_anchors(
    timeline: SegmentTimeline,
    intervals: Sequence[Interval],
    horizons_minutes: Sequence[int],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Split label per anchor, and per horizon whether the block stays inside it.

    A block is valid only if its whole span lies in one split piece, which is how
    "targets do not cross a split, a gap or a seizure" becomes a boolean the
    trainer can honour rather than a sentence in a plan.
    """

    members = [i for i in intervals if i.segment_id == timeline.segment_id]
    anchors = timeline.anchor_time
    split = np.full(anchors.size, "", dtype=object)
    piece_stop = np.full(anchors.size, -np.inf)
    for interval in members:
        inside = (anchors >= interval.start - 1e-6) & (anchors < interval.stop - 1e-6)
        split[inside] = interval.split
        piece_stop[inside] = interval.stop

    valid: dict[int, np.ndarray] = {}
    for horizon in horizons_minutes:
        span = float(horizon) * 60.0
        valid[int(horizon)] = (anchors + span) <= (piece_stop + 1e-6)
    return split, valid
