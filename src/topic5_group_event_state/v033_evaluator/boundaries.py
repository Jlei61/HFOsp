"""Frozen data-boundary contract for v0.3.3 (plan Task 3, spec §4.1-§4.3).

Two different partitions of time exist and must not be confused:

* **target segments** -- recorded sessions cut at seizure onset, with the
  postictal exclusion removed (``v02.timeline.build_carry_segments``).  A
  target window ``[t, t+h)`` must lie inside one target segment *and* inside
  one partition phase (B1).
* **state carry units** -- recorded sessions.  The state is hard-reset only
  at a recording gap / session edge (B4); across a seizure inside one session
  it keeps its autonomous flow while the seizure and immediate-postictal
  events are simply not written (B2, B3).

A hard reset at seizure onset (the v0.3.2 behaviour, where carry = target
segment) is kept as the named sensitivity variant
``sensitivity_hard_seizure_reset`` and is never the default (B5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_VARIANT = "mainline"
VARIANTS: dict[str, dict[str, str]] = {
    "mainline": {
        "state_reset_at": "recorded_gap_or_session_edge_only",
        "seizure_and_immediate_postictal_events": "excluded_from_state_update_autonomous_flow_continues",
        "target_windows": "inside_one_target_segment_and_one_partition_phase",
        "role": "default",
    },
    "sensitivity_hard_seizure_reset": {
        "state_reset_at": "recorded_gap_or_session_edge_or_seizure_onset",
        "seizure_and_immediate_postictal_events": "excluded_from_state_update",
        "target_windows": "inside_one_target_segment_and_one_partition_phase",
        "role": "sensitivity_only",
    },
}


@dataclass(frozen=True)
class CarryUnit:
    unit_id: int
    session_id: int
    start_epoch: float
    stop_epoch: float


def boundary_variant(name: str) -> dict[str, str]:
    return dict(VARIANTS[name])


def state_carry_units(sessions: Sequence[Any]) -> list[CarryUnit]:
    """One carry unit per recorded session; sessions must be separated by a gap."""

    ordered = sorted(sessions, key=lambda s: (float(s.start_epoch), int(s.session_id)))
    units: list[CarryUnit] = []
    for i, s in enumerate(ordered):
        if i and float(s.start_epoch) < units[-1].stop_epoch:
            raise ValueError("recorded sessions overlap; a carry unit must be one contiguous recording")
        units.append(CarryUnit(unit_id=i, session_id=int(s.session_id),
                               start_epoch=float(s.start_epoch), stop_epoch=float(s.stop_epoch)))
    return units


def _starts_stops(intervals: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort([float(x.start_epoch) for x in intervals], kind="stable")
    starts = np.asarray([float(intervals[i].start_epoch) for i in order], dtype=np.float64)
    stops = np.asarray([float(intervals[i].stop_epoch) for i in order], dtype=np.float64)
    if starts.size > 1 and (starts[1:] < stops[:-1]).any():
        raise ValueError("intervals overlap")
    return starts, stops, order


def anchor_carry_index(times: np.ndarray, intervals: Sequence[Any]) -> np.ndarray:
    """Index (into ``intervals`` as given) of the interval holding each time, else -1."""

    t = np.asarray(times, dtype=np.float64).reshape(-1)
    out = np.full(t.size, -1, dtype=np.int64)
    if not intervals:
        return out
    starts, stops, order = _starts_stops(intervals)
    pos = np.searchsorted(starts, t, side="right") - 1
    ok = (pos >= 0) & (t < stops[np.clip(pos, 0, starts.size - 1)])
    out[ok] = np.asarray(order, dtype=np.int64)[pos[ok]]
    return out


def event_update_mask(event_times: np.ndarray, seizures: Sequence[Mapping[str, Any]],
                      *, postictal_seconds: float) -> np.ndarray:
    """False for events inside ``[onset, max(offset, onset) + postictal_seconds)`` of any seizure (B2)."""

    t = np.asarray(event_times, dtype=np.float64).reshape(-1)
    mask = np.ones(t.size, dtype=bool)
    for sz in seizures:
        onset = float(sz["onset_epoch"])
        stop = max(float(sz["offset_epoch"]), onset) + float(postictal_seconds)
        mask &= ~((t >= onset) & (t < stop))
    return mask


def target_window_valid(t_anchor: np.ndarray, horizon: float, segments: Sequence[Any],
                        partition: Any) -> np.ndarray:
    """Whole ``[t, t+h)`` inside one target segment and one partition phase (B1)."""

    t = np.asarray(t_anchor, dtype=np.float64).reshape(-1)
    seg = anchor_carry_index(t, segments)
    ok = seg >= 0
    stops = np.asarray([float(s.stop_epoch) for s in segments], dtype=np.float64) if segments else np.zeros(0)
    ok[ok] &= (t[ok] + float(horizon)) <= stops[seg[ok]]
    ok &= np.asarray(partition.window_within_phase(t, float(horizon)), dtype=bool)
    return ok


def carry_last_event(event_times: np.ndarray, event_unit: np.ndarray, update_mask: np.ndarray,
                     t_anchor: np.ndarray, anchor_unit: np.ndarray) -> np.ndarray:
    """Last *updating* event strictly before each anchor inside the anchor's carry unit; -1 if none.

    Events must be time-sorted.  Because carry units are disjoint, time-ordered
    intervals, the latest kept event before ``t`` either belongs to the anchor's
    unit or to an earlier unit -- in the latter case the anchor has no history
    inside its unit and starts from the reset state (B3/B4).
    """

    t_ev = np.asarray(event_times, dtype=np.float64).reshape(-1)
    if t_ev.size > 1 and (np.diff(t_ev) < 0).any():
        raise ValueError("event_times must be sorted")
    ev_unit = np.asarray(event_unit, dtype=np.int64).reshape(-1)
    keep = np.asarray(update_mask, dtype=bool).reshape(-1) & (ev_unit >= 0)
    kept = np.flatnonzero(keep)
    t_a = np.asarray(t_anchor, dtype=np.float64).reshape(-1)
    a_unit = np.asarray(anchor_unit, dtype=np.int64).reshape(-1)
    out = np.full(t_a.size, -1, dtype=np.int64)
    if kept.size == 0:
        return out
    pos = np.searchsorted(t_ev[kept], t_a, side="left") - 1
    cand = kept[np.clip(pos, 0, kept.size - 1)]
    ok = (pos >= 0) & (a_unit >= 0) & (ev_unit[cand] == a_unit)
    out[ok] = cand[ok]
    return out
