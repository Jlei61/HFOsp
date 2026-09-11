"""Physical-time skeleton for Group-Event State v0.2: segments, split, anchors.

Three things v0.1 got structurally wrong for a slow-state question, all fixed
here and nowhere else:

1. v0.1 split TRAIN/VAL/TEST by **event count**.  A patient whose first hours
   fire ten times faster than the rest then gets a TRAIN split that is a few
   hours long and a TEST split that is days long -- or the reverse -- and the
   horizon axis stops meaning the same thing across patients.  v0.2 splits by
   cumulative *recorded* time (CC 7.1).

2. v0.1 carried state across seizures.  ``SubjectSequence.new_session`` only
   flips at recorded-session changes, so a seizure and its postictal hour were
   silently bridged into one state chain (CC 7.4-7.6, EI 3).

3. v0.1 had no fixed-time anchor at all; every quantity was per event, so a
   noisy hour contributed ten times the weight of a quiet hour to anything that
   was supposed to be a statement about *time* (CC 5.2).

Nothing in this module reads model output, and nothing reads
``index.json::split_bounds_on_interictal_index`` -- that field is the v0.1
event-count split and is forbidden on the v0.2 path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# One anchor every 5 minutes of recorded time (CC 5.2).
ANCHOR_GRID_SECONDS = 300.0

# Future-block horizons of the first round.  6 h is explored only where a single
# segment is actually that long, and is not part of this tuple by default.
HORIZONS_SECONDS: tuple[float, ...] = (300.0, 1800.0, 7200.0)

# Chronological development split on cumulative recorded time.
SPLIT_FRACTIONS = (0.70, 0.10, 0.20)
SPLIT_NAMES = ("train", "val", "test")

# CC 7.5: "the new segment starts 60 min after seizure offset".  Primary value
# for the first round; other lengths are sensitivity only.
POSTICTAL_EXCLUSION_SECONDS = 3600.0

# A stretch of recording shorter than one grid step can never carry an anchor.
MIN_SEGMENT_SECONDS = ANCHOR_GRID_SECONDS

# Every anchor gets at least this much in-segment history before it is scored,
# so no arm is asked to predict from a state that has just been initialised.
MIN_WARMUP_SECONDS = ANCHOR_GRID_SECONDS


@dataclass(frozen=True)
class RecordedSession:
    """One contiguously recorded stretch, as frozen by the v0.1 source audit."""

    session_id: int
    start_epoch: float
    stop_epoch: float

    @property
    def duration_seconds(self) -> float:
        return float(self.stop_epoch - self.start_epoch)


@dataclass(frozen=True)
class CoverageSegment:
    """A stretch inside which state may be carried and a window may be scored.

    Bounded by a recorded-session edge on one side or the other, or by a seizure
    (which ends a segment at its onset) and its postictal exclusion (after which
    the next segment starts).
    """

    segment_id: int
    session_id: int
    start_epoch: float
    stop_epoch: float

    @property
    def duration_seconds(self) -> float:
        return float(self.stop_epoch - self.start_epoch)


@dataclass(frozen=True)
class PhysicalTimeSplit:
    """Chronological split on the cumulative-recorded-time coordinate."""

    boundary_epochs: np.ndarray          # float64, shape (2,)
    recorded_seconds: dict[str, float]
    total_recorded_seconds: float
    fractions: tuple[float, float, float]

    def label_of(self, epoch: float) -> str:
        b0, b1 = float(self.boundary_epochs[0]), float(self.boundary_epochs[1])
        t = float(epoch)
        if t < b0:
            return "train"
        if t < b1:
            return "val"
        return "test"

    def labels_of(self, epochs: np.ndarray) -> np.ndarray:
        t = np.asarray(epochs, dtype=np.float64)
        idx = (t >= self.boundary_epochs[0]).astype(np.int64) + (
            t >= self.boundary_epochs[1]
        ).astype(np.int64)
        return idx


@dataclass(frozen=True)
class AnchorGrid:
    """Fixed physical-time anchors and their per-horizon target windows."""

    t_anchor: np.ndarray            # float64 (A,)
    segment_index: np.ndarray       # int64  (A,) index into the segment list
    session_id: np.ndarray          # int64  (A,)
    split_index: np.ndarray         # int64  (A,) 0=train 1=val 2=test
    last_event_pos: np.ndarray      # int64  (A,) index into event_times, -1 if none
    n_events_before: np.ndarray     # int64  (A,) events since this segment started
    seconds_since_last_event: np.ndarray  # float64 (A,), inf if none
    horizons_seconds: tuple[float, ...]
    eligible: np.ndarray            # bool   (A, H)
    window_lo: np.ndarray           # int64  (A, H) first event index in [t, t+h)
    window_hi: np.ndarray           # int64  (A, H) one past the last

    @property
    def n_anchors(self) -> int:
        return int(self.t_anchor.size)

    def split_mask(self, name: str) -> np.ndarray:
        return self.split_index == SPLIT_NAMES.index(name)


def sessions_from_inventory(rows: Iterable[Mapping[str, Any]]) -> list[RecordedSession]:
    """Read the frozen ``contiguous_session_inventory.csv`` rows for one patient.

    That file is the same coverage table the v0.1 consolidation used to assign
    ``session_of_event``, so segment ids stay comparable with the cached stream.
    """

    out = [
        RecordedSession(
            session_id=int(row["session_index"]),
            start_epoch=float(row["start_epoch"]),
            stop_epoch=float(row["end_epoch"]),
        )
        for row in rows
    ]
    out.sort(key=lambda s: (s.start_epoch, s.session_id))
    return out


def build_carry_segments(
    sessions: Sequence[RecordedSession],
    seizures: Sequence[Mapping[str, Any]] = (),
    *,
    postictal_exclusion_seconds: float = POSTICTAL_EXCLUSION_SECONDS,
    min_segment_seconds: float = MIN_SEGMENT_SECONDS,
) -> list[CoverageSegment]:
    """Cut recorded sessions at seizures and drop the postictal exclusion window.

    Clause C2 -- a recording gap is never bridged: sessions are already the
    maximal contiguous stretches, so each one is cut independently.

    Clause C3 -- a seizure ends the segment at ``onset`` and the next segment
    starts at ``offset + postictal_exclusion_seconds``.  No instant of
    ``[onset, offset + exclusion)`` belongs to any segment, so neither an anchor,
    nor a target window, nor a carried state can span it.
    """

    if postictal_exclusion_seconds < 0:
        raise ValueError("postictal_exclusion_seconds must be >= 0")
    blocked: list[tuple[float, float]] = []
    for sz in seizures:
        onset = float(sz["onset_epoch"])
        offset = float(sz["offset_epoch"])
        if not (math.isfinite(onset) and math.isfinite(offset)):
            raise ValueError(f"non-finite seizure bounds: {sz!r}")
        blocked.append((onset, max(offset, onset) + float(postictal_exclusion_seconds)))
    blocked.sort()

    segments: list[CoverageSegment] = []
    for session in sorted(sessions, key=lambda s: (s.start_epoch, s.session_id)):
        pieces = [(float(session.start_epoch), float(session.stop_epoch))]
        for lo, hi in blocked:
            nxt: list[tuple[float, float]] = []
            for a, b in pieces:
                if hi <= a or lo >= b:
                    nxt.append((a, b))
                    continue
                if a < lo:
                    nxt.append((a, min(lo, b)))
                if b > hi:
                    nxt.append((max(hi, a), b))
            pieces = nxt
        for a, b in pieces:
            if b - a >= float(min_segment_seconds) and b > a:
                segments.append(
                    CoverageSegment(
                        segment_id=len(segments),
                        session_id=int(session.session_id),
                        start_epoch=float(a),
                        stop_epoch=float(b),
                    )
                )
    return segments


def physical_time_split(
    segments: Sequence[CoverageSegment],
    fractions: Sequence[float] = SPLIT_FRACTIONS,
) -> PhysicalTimeSplit:
    """Split on cumulative recorded seconds, not on event index (clause C1).

    The boundary is located in the recorded-time coordinate and then mapped back
    to an absolute epoch, so a long unrecorded gap costs neither split anything.
    """

    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"need three fractions summing to 1, got {tuple(fractions)}")
    if not segments:
        raise ValueError("cannot split an empty coverage")

    starts = np.array([s.start_epoch for s in segments], dtype=np.float64)
    durations = np.array([s.duration_seconds for s in segments], dtype=np.float64)
    cum_after = np.cumsum(durations)
    cum_before = cum_after - durations
    total = float(cum_after[-1])

    def _epoch_at(recorded: float) -> float:
        r = float(min(max(recorded, 0.0), total))
        k = int(np.searchsorted(cum_after, r, side="left"))
        k = min(k, len(segments) - 1)
        return float(starts[k] + (r - cum_before[k]))

    r0 = total * float(fractions[0])
    r1 = total * float(fractions[0] + fractions[1])
    boundaries = np.array([_epoch_at(r0), _epoch_at(r1)], dtype=np.float64)
    return PhysicalTimeSplit(
        boundary_epochs=boundaries,
        recorded_seconds={
            "train": r0,
            "val": r1 - r0,
            "test": total - r1,
        },
        total_recorded_seconds=total,
        fractions=(float(fractions[0]), float(fractions[1]), float(fractions[2])),
    )


def assign_events_to_segments(
    event_times: np.ndarray, segments: Sequence[CoverageSegment]
) -> np.ndarray:
    """Segment index of every event, or ``-1`` when it belongs to none.

    Clause C3: an event inside a seizure's postictal exclusion is *dropped*, not
    attached to the nearest neighbouring segment.
    """

    t = np.asarray(event_times, dtype=np.float64)
    out = np.full(t.size, -1, dtype=np.int64)
    if not segments:
        return out
    starts = np.array([s.start_epoch for s in segments], dtype=np.float64)
    stops = np.array([s.stop_epoch for s in segments], dtype=np.float64)
    pos = np.searchsorted(starts, t, side="right") - 1
    ok = (pos >= 0) & (t < stops[np.clip(pos, 0, len(segments) - 1)])
    out[ok] = pos[ok]
    return out


def build_anchor_grid(
    segments: Sequence[CoverageSegment],
    split: PhysicalTimeSplit,
    event_times: np.ndarray,
    *,
    horizons_seconds: Sequence[float] = HORIZONS_SECONDS,
    grid_seconds: float = ANCHOR_GRID_SECONDS,
    min_warmup_seconds: float = MIN_WARMUP_SECONDS,
) -> AnchorGrid:
    """Anchors every ``grid_seconds`` of recorded time, with their target windows.

    Clause C4 -- the grid is uniform in *time*.  A busy hour and a quiet hour of
    equal length contribute the same number of anchors, so nothing downstream is
    re-weighted by IED rate.

    Clause C8 -- ``last_event_pos`` is the last event strictly before the anchor
    *and inside the same segment*; state is propagated forward from there by the
    real elapsed ``dt``.

    Clause C1 + C3 -- a horizon is eligible only when the whole window
    ``[t, t + h)`` lies inside this one segment and inside one split.
    """

    times = np.asarray(event_times, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError("event_times must be one-dimensional")
    if times.size and np.any(np.diff(times) < 0):
        raise ValueError("event_times must be sorted")
    horizons = tuple(float(h) for h in horizons_seconds)
    if not horizons:
        raise ValueError("at least one horizon is required")

    t_list: list[float] = []
    seg_list: list[int] = []
    for seg in segments:
        first = math.ceil(
            (seg.start_epoch + float(min_warmup_seconds)) / float(grid_seconds)
        ) * float(grid_seconds)
        n = int(math.floor((seg.stop_epoch - first) / float(grid_seconds))) + 1
        for k in range(max(n, 0)):
            t = first + k * float(grid_seconds)
            if t >= seg.stop_epoch:
                break
            t_list.append(t)
            seg_list.append(seg.segment_id)

    t_anchor = np.asarray(t_list, dtype=np.float64)
    seg_index = np.asarray(seg_list, dtype=np.int64)
    n = t_anchor.size
    session_id = np.array(
        [segments[i].session_id for i in seg_index], dtype=np.int64
    ) if n else np.zeros(0, dtype=np.int64)
    seg_start = np.array(
        [segments[i].start_epoch for i in seg_index], dtype=np.float64
    ) if n else np.zeros(0, dtype=np.float64)
    seg_stop = np.array(
        [segments[i].stop_epoch for i in seg_index], dtype=np.float64
    ) if n else np.zeros(0, dtype=np.float64)

    # C8: last event strictly before the anchor, and never from a earlier segment.
    pos = np.searchsorted(times, t_anchor, side="left") - 1
    in_segment = (pos >= 0) & (times[np.clip(pos, 0, max(times.size - 1, 0))] >= seg_start)
    last_pos = np.where(in_segment, pos, -1).astype(np.int64)
    since_last = np.where(
        last_pos >= 0,
        t_anchor - times[np.clip(last_pos, 0, max(times.size - 1, 0))],
        np.inf,
    ).astype(np.float64)
    first_in_seg = np.searchsorted(times, seg_start, side="left")
    n_before = np.where(last_pos >= 0, last_pos + 1 - first_in_seg, 0).astype(np.int64)

    eligible = np.zeros((n, len(horizons)), dtype=bool)
    window_lo = np.zeros((n, len(horizons)), dtype=np.int64)
    window_hi = np.zeros((n, len(horizons)), dtype=np.int64)
    label_start = split.labels_of(t_anchor)
    for h_i, horizon in enumerate(horizons):
        stop = t_anchor + horizon
        # half-open window: the last instant scored is stop - eps
        label_stop = split.labels_of(np.nextafter(stop, -np.inf))
        eligible[:, h_i] = (stop <= seg_stop) & (label_start == label_stop)
        window_lo[:, h_i] = np.searchsorted(times, t_anchor, side="left")
        window_hi[:, h_i] = np.searchsorted(times, stop, side="left")

    return AnchorGrid(
        t_anchor=t_anchor,
        segment_index=seg_index,
        session_id=session_id,
        split_index=label_start.astype(np.int64),
        last_event_pos=last_pos,
        n_events_before=n_before,
        seconds_since_last_event=since_last,
        horizons_seconds=horizons,
        eligible=eligible,
        window_lo=window_lo,
        window_hi=window_hi,
    )


def split_covered_intervals(
    segments: Sequence[CoverageSegment], split: PhysicalTimeSplit, name: str
) -> list[tuple[float, float]]:
    """Covered intervals of one split, cut by both segment and split edges."""

    lo_edge = -np.inf if name == "train" else float(split.boundary_epochs[
        0 if name == "val" else 1
    ])
    hi_edge = np.inf if name == "test" else float(split.boundary_epochs[
        0 if name == "train" else 1
    ])
    out: list[tuple[float, float]] = []
    for seg in segments:
        a = max(seg.start_epoch, lo_edge)
        b = min(seg.stop_epoch, hi_edge)
        if b > a:
            out.append((float(a), float(b)))
    return out


def effective_independent_windows(
    segments: Sequence[CoverageSegment],
    split: PhysicalTimeSplit,
    name: str,
    horizon_seconds: float,
) -> int:
    """How many *non-overlapping* horizon windows this split could hold (C11).

    The anchor grid steps 5 min, so at a 2 h horizon consecutive windows share
    96% of their content.  Reporting the anchor count as a sample size is the
    failure this repository has already been burned by; the honest denominator is
    covered time divided by the horizon, computed with the same segment logic the
    estimator uses.
    """

    total = 0
    for a, b in split_covered_intervals(segments, split, name):
        total += int(math.floor((b - a) / float(horizon_seconds)))
    return int(total)
