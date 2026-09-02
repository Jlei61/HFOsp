"""Coverage support for H3: what physical time is actually usable, and how it tiles.

Everything downstream of this file is measured in *recorded seconds*, never in
events.  An hour with no IED is an hour of evidence that the rate was low; an
hour that was never recorded is not evidence of anything, and the two must not
share a denominator.

Four cuts are applied, in this order, and none of them may be skipped:

``A1`` recording gaps    two packed blocks that do not abut within
                         ``SEAM_TOLERANCE_SECONDS`` bound different segments,
                         and so does a recorded block that produced no group
                         events (that hour is unobserved, not event-free).
``A2`` seizure onset     a segment ends at seizure onset.  No exposure window and
                         no target block may contain one.
``A3`` postictal         the next segment starts ``postictal_exclusion_s`` after
                         seizure offset (60 min primary, 0 min sensitivity).
``A4`` split             TRAIN / inner-validation / development-test boundaries
                         are placed on cumulative *recorded* time and then cut
                         the segments like any other boundary.

Only after all four does ``tile_blocks`` lay down disjoint fixed-time blocks.
A block that would run past the end of its interval is dropped, never shortened:
a 41-minute "120 min block" is a different estimand.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


# A1: same constant the v0.1 data contract froze; two consecutive blocks written
# by one acquisition run abut to within a sample or two.
SEAM_TOLERANCE_SECONDS = 2.0

# A3: primary postictal exclusion.  Other lengths are sensitivity only.
POSTICTAL_EXCLUSION_SECONDS = 3600.0

# A4: chronological split on cumulative recorded time (common contract §7.1).
# 60/20/20 rather than the v0.1 70/10/20.  A 10% inner-validation slice of a
# 24-hour patient is 2.4 hours, which carries five 120-minute blocks -- a
# checkpoint criterion estimated from five numbers selects noise.  Fixed here,
# before any development-test score existed, and identical for all three arms.
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)
SPLIT_NAMES = ("train", "inner_validation", "development_test")

# H3 §5: fixed-time main horizons.  6 h is explored only where support allows.
MAIN_HORIZONS_MINUTES = (5, 30, 120)
EXPLORATORY_HORIZON_MINUTES = 360

# CC §5.2: the fixed physical-time anchor grid.
ANCHOR_GRID_MINUTES = 5


@dataclass(frozen=True)
class Interval:
    """A half-open span of recorded, usable, single-split wall-clock time."""

    start: float
    stop: float
    segment_id: int
    split: str

    @property
    def duration(self) -> float:
        return float(self.stop - self.start)

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": float(self.start),
            "stop": float(self.stop),
            "segment_id": int(self.segment_id),
            "split": str(self.split),
            "duration_seconds": self.duration,
        }


@dataclass(frozen=True)
class Block:
    """One disjoint fixed-time future-target block and its exposure window."""

    subject: str
    split: str
    horizon_minutes: int
    block_index: int
    segment_id: int
    exposure_start: float
    anchor: float
    target_stop: float

    @property
    def key(self) -> tuple[str, str, int, int]:
        # B2: every paired comparison aligns on this key, never on array position.
        return (self.subject, self.split, int(self.horizon_minutes), int(self.block_index))

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["exposure_seconds"] = float(self.anchor - self.exposure_start)
        d["target_seconds"] = float(self.target_stop - self.anchor)
        return d


# --------------------------------------------------------------------------- A1


def build_coverage_segments(
    block_time_ranges: Sequence[tuple[float, float]],
    *,
    seam_tolerance_s: float = SEAM_TOLERANCE_SECONDS,
) -> list[tuple[float, float]]:
    """Merge recorded blocks into contiguous coverage segments.

    ``block_time_ranges`` is required and carries only blocks that actually
    produced group events.  There is deliberately no default: a ``None`` that
    silently means "the whole recording is one segment" is the exact failure this
    project has already paid for, and it would let a state chain walk across
    hours that were never recorded.
    """

    ranges = [(float(a), float(b)) for a, b in block_time_ranges]
    if not ranges:
        raise ValueError("block_time_ranges is empty; refusing to invent coverage")
    if any(b <= a for a, b in ranges):
        raise ValueError("block_time_ranges contains a non-positive duration")
    ranges.sort()
    segments: list[list[float]] = [list(ranges[0])]
    for start, stop in ranges[1:]:
        if start - segments[-1][1] <= float(seam_tolerance_s):
            segments[-1][1] = max(segments[-1][1], stop)
        else:
            segments.append([start, stop])
    return [(float(a), float(b)) for a, b in segments]


# --------------------------------------------------------------------------- A2/A3


def cut_intervals_at_seizures(
    segments: Sequence[tuple[float, float]],
    seizures: Sequence[tuple[float, float]],
    *,
    postictal_exclusion_s: float = POSTICTAL_EXCLUSION_SECONDS,
) -> list[tuple[float, float]]:
    """Remove ``[onset, offset + postictal)`` from every coverage segment.

    A2 and A3 are one operation because they are one physical statement: the
    seizure ends the segment it lands in, and what follows it is a *new* segment
    that only starts once the postictal exclusion has elapsed.  Bridging the two
    would let a state chain carry an ictal transition it never modelled.
    """

    if postictal_exclusion_s < 0:
        raise ValueError("postictal_exclusion_s must be non-negative")
    forbidden = sorted(
        (float(onset), float(offset) + float(postictal_exclusion_s))
        for onset, offset in seizures
    )
    out: list[tuple[float, float]] = []
    for seg_start, seg_stop in segments:
        pieces = [(float(seg_start), float(seg_stop))]
        for bad_start, bad_stop in forbidden:
            nxt: list[tuple[float, float]] = []
            for a, b in pieces:
                if bad_stop <= a or bad_start >= b:
                    nxt.append((a, b))
                    continue
                if a < bad_start:
                    nxt.append((a, bad_start))
                if bad_stop < b:
                    nxt.append((bad_stop, b))
            pieces = nxt
        out.extend(p for p in pieces if p[1] > p[0])
    return sorted(out)


# --------------------------------------------------------------------------- A4


def split_by_physical_time(
    intervals: Sequence[tuple[float, float]],
    *,
    fractions: Sequence[float] = SPLIT_FRACTIONS,
    names: Sequence[str] = SPLIT_NAMES,
) -> list[Interval]:
    """Chronological split on cumulative recorded seconds, cutting intervals.

    The v0.1 dataset ships a split on *event count*; the v0.2 contract requires
    recorded time.  A patient whose events cluster in one noisy night would
    otherwise have a development-test split that is minutes long in wall-clock
    terms while claiming 20% of the recording.

    The boundary is a real cut, not a label: an interval straddling it becomes two
    intervals, so no block can later span two splits.
    """

    if len(fractions) != len(names):
        raise ValueError("fractions and names must have the same length")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    ordered = sorted((float(a), float(b)) for a, b in intervals)
    total = sum(b - a for a, b in ordered)
    if total <= 0:
        return []
    cuts = np.cumsum(np.asarray(fractions, dtype=np.float64)) * total

    out: list[Interval] = []
    elapsed = 0.0
    for segment_id, (a, b) in enumerate(ordered):
        cursor = a
        while cursor < b - 1e-6:
            k = min(int(np.searchsorted(cuts, elapsed, side="right")), len(names) - 1)
            room = float(cuts[k] - elapsed) if k < len(names) - 1 else float("inf")
            stop = min(b, cursor + max(room, 1e-6))
            out.append(Interval(cursor, stop, segment_id, str(names[k])))
            elapsed += stop - cursor
            cursor = stop
    return out


# --------------------------------------------------------------------------- A7


def segment_bounds(intervals: Sequence[Interval]) -> dict[int, tuple[float, float]]:
    """Re-assemble the pre-split coverage segments the split cut into pieces.

    The split is a boundary for *targets*, not for the state chain: the recording
    really is continuous across it, and a state that reset there would be judged
    on a warm-up it never had.  Rollouts therefore run per segment, while blocks
    are confined to one split each.
    """

    out: dict[int, tuple[float, float]] = {}
    for interval in intervals:
        lo, hi = out.get(interval.segment_id, (interval.start, interval.stop))
        out[interval.segment_id] = (min(lo, interval.start), max(hi, interval.stop))
    return out


def segment_anchor_grid(
    segment_start: float, segment_stop: float, *, minutes: int = ANCHOR_GRID_MINUTES
) -> np.ndarray:
    """The fixed physical anchor grid of one coverage segment.

    Laid from the segment's own start so that background cells, training anchors
    and evaluation blocks all sit on *one* grid.  Two grids -- one for cells and
    one for blocks -- would let an anchor read a background cell that had already
    absorbed part of its own future block.
    """

    step = float(minutes) * 60.0
    n = int(np.floor((float(segment_stop) - float(segment_start)) / step)) + 1
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    return float(segment_start) + step * np.arange(n, dtype=np.float64)


def select_disjoint_anchors(
    grid: np.ndarray,
    intervals: Sequence[Interval],
    horizon_minutes: int,
    *,
    exposure_minutes: int | None = None,
    disjoint_exposure: bool = False,
) -> list[tuple[float, str, int]]:
    """Grid anchors whose whole (exposure, target) pair fits in one split piece.

    Greedy from each piece's own start, stepping by the tiling period, so the
    returned anchors are the *independent* ones.  Anchors are never shortened to
    fit; a block that would overrun the piece is dropped, because a 41-minute
    "120 min block" is a different estimand and would quietly pad the denominator.
    """

    horizon_s = float(horizon_minutes) * 60.0
    exposure_s = float(horizon_minutes if exposure_minutes is None else exposure_minutes) * 60.0
    period = exposure_s + horizon_s if disjoint_exposure else horizon_s
    grid = np.asarray(grid, dtype=np.float64)

    out: list[tuple[float, str, int]] = []
    for interval in intervals:
        eligible = grid[
            (grid >= interval.start + exposure_s - 1e-6)
            & (grid + horizon_s <= interval.stop + 1e-6)
        ]
        if eligible.size == 0:
            continue
        taken = eligible[0]
        out.append((float(taken), interval.split, int(interval.segment_id)))
        for candidate in eligible[1:]:
            if candidate >= taken + period - 1e-6:
                out.append((float(candidate), interval.split, int(interval.segment_id)))
                taken = candidate
    return out


def tile_blocks(
    subject: str,
    intervals: Sequence[Interval],
    horizon_minutes: int,
    *,
    exposure_minutes: int | None = None,
    disjoint_exposure: bool = True,
) -> list[Block]:
    """Grid-snapped disjoint (exposure, target) pairs inside each usable interval.

    ``disjoint_exposure=True`` (the perturbation setting) tiles with period
    ``exposure + target`` so that *both* halves of every pair are disjoint from
    every other pair.  ``False`` (the model-comparison setting) tiles targets at
    period ``target`` and lets each target's exposure be the span that precedes
    it, so the targets -- which are the statistical denominator -- stay disjoint
    while exposures may abut a neighbouring target.

    Either way the count returned is a count of **non-overlapping target blocks**.
    A sliding-window count is never produced by this function, so it cannot be
    mistaken for one downstream.
    """

    horizon_s = float(horizon_minutes) * 60.0
    exposure_s = float(horizon_minutes if exposure_minutes is None else exposure_minutes) * 60.0
    if horizon_s <= 0 or exposure_s <= 0:
        raise ValueError("horizon and exposure must be positive")

    bounds = segment_bounds(intervals)
    out: list[Block] = []
    index = 0
    for segment_id in sorted(bounds):
        lo, hi = bounds[segment_id]
        grid = segment_anchor_grid(lo, hi)
        members = [i for i in intervals if i.segment_id == segment_id]
        for anchor, split, seg in select_disjoint_anchors(
            grid,
            members,
            horizon_minutes,
            exposure_minutes=exposure_minutes,
            disjoint_exposure=disjoint_exposure,
        ):
            out.append(
                Block(
                    subject=str(subject),
                    split=split,
                    horizon_minutes=int(horizon_minutes),
                    block_index=index,
                    segment_id=seg,
                    exposure_start=float(anchor - exposure_s),
                    anchor=float(anchor),
                    target_stop=float(anchor + horizon_s),
                )
            )
            index += 1
    return out


# --------------------------------------------------------------------------- IO


def load_block_time_ranges(inventory_csv: Path, subject: str) -> list[tuple[float, float]]:
    """Recorded spans of the blocks that produced group events for one patient.

    A1's second clause lives here: a row that did not pass the source audit, or
    that produced zero events, is *excluded*, which makes that hour a segment
    boundary rather than a silently event-free hour.
    """

    ranges: list[tuple[float, float]] = []
    with Path(inventory_csv).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["subject"] != subject or row["status"] != "PASS":
                continue
            if int(float(row["n_events"] or 0)) <= 0:
                continue
            ranges.append((float(row["block_start_epoch"]), float(row["block_end_epoch"])))
    if not ranges:
        raise ValueError(f"{subject}: no PASS blocks with events in {inventory_csv}")
    return ranges


def load_seizures(dataset_index: Path) -> list[tuple[float, float]]:
    """Seizure ``[onset, offset)`` epochs from the consolidated dataset index.

    Provenance note kept deliberately loud: Yuquan seizures are pr1 spatial-extent
    *detections*, not clinical marks, and 10/21 Yuquan patients have zero of them.
    Zero detections means "none detected", never "no seizures", so a patient with
    no rows here still gets every other boundary applied unchanged.
    """

    index = json.loads(Path(dataset_index).read_text())
    out: list[tuple[float, float]] = []
    for s in index.get("seizures", []):
        onset, offset = float(s["onset_epoch"]), float(s["offset_epoch"])
        if offset < onset:
            raise ValueError(f"seizure offset precedes onset in {dataset_index}")
        out.append((onset, offset))
    return sorted(out)


def subject_support(
    subject: str,
    inventory_csv: Path,
    dataset_index: Path,
    *,
    postictal_exclusion_s: float = POSTICTAL_EXCLUSION_SECONDS,
    horizons: Iterable[int] = MAIN_HORIZONS_MINUTES,
) -> dict[str, Any]:
    """Full A1->A4 cut chain plus the block counts each horizon actually supports."""

    blocks_ranges = load_block_time_ranges(inventory_csv, subject)
    seizures = load_seizures(dataset_index)
    segments = build_coverage_segments(blocks_ranges)
    cut = cut_intervals_at_seizures(
        segments, seizures, postictal_exclusion_s=postictal_exclusion_s
    )
    intervals = split_by_physical_time(cut)

    per_split_hours = {name: 0.0 for name in SPLIT_NAMES}
    for interval in intervals:
        per_split_hours[interval.split] += interval.duration / 3600.0

    horizon_support: dict[str, Any] = {}
    for horizon in horizons:
        model_blocks = tile_blocks(subject, intervals, horizon, disjoint_exposure=False)
        pert_blocks = tile_blocks(subject, intervals, horizon, disjoint_exposure=True)
        horizon_support[str(horizon)] = {
            "n_independent_target_blocks": {
                name: sum(1 for b in model_blocks if b.split == name) for name in SPLIT_NAMES
            },
            "n_disjoint_exposure_target_pairs": {
                name: sum(1 for b in pert_blocks if b.split == name) for name in SPLIT_NAMES
            },
        }

    return {
        "subject": subject,
        "postictal_exclusion_s": float(postictal_exclusion_s),
        "n_recorded_blocks": len(blocks_ranges),
        "n_coverage_segments": len(segments),
        "recorded_hours": sum(b - a for a, b in blocks_ranges) / 3600.0,
        "coverage_hours": sum(b - a for a, b in segments) / 3600.0,
        "usable_hours_after_seizure_cuts": sum(b - a for a, b in cut) / 3600.0,
        "n_seizures": len(seizures),
        "n_usable_intervals": len(intervals),
        "split_hours": per_split_hours,
        "horizon_support": horizon_support,
        "intervals": [i.as_dict() for i in intervals],
    }
