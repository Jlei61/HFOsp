"""Sessions and blocks for the slow-state contract.

Session bounds come from metadata source intervals, never from the first and last
detected event: a normally recorded stretch in which no HFO was detected is recorded
time with no events, not missing data.

Two gap quantities are kept apart.  `metadata_gap_seconds` is unrecorded wall time.
`event_silence_seconds` is the span between the last event of one session and the first
of the next, and is always at least as large when both sessions have events.  A session
with zero observed events has no defined event silence, so `event_silence_seconds` is None.
Neither ever means "no events occurred".
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _field(segment: Any, name: str) -> Any:
    return segment[name] if isinstance(segment, Mapping) else getattr(segment, name)


def build_sessions(
    segments: Sequence[Any], *, join_seconds: float
) -> list[dict[str, Any]]:
    ordered = sorted(
        segments, key=lambda s: (float(_field(s, "start_time")), str(_field(s, "source_id")))
    )
    sessions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for segment in ordered:
        start = float(_field(segment, "start_time"))
        stop = float(_field(segment, "stop_time"))
        group = str(_field(segment, "continuity_group"))
        montage = str(_field(segment, "montage_hash"))
        joinable = (
            current is not None
            and start - current["t_end"] <= float(join_seconds)
            and current["continuity_group"] == group
            and current["montage_hash"] == montage
        )
        if joinable:
            current["t_end"] = max(current["t_end"], stop)
            current["segment_ids"].append(str(_field(segment, "source_id")))
        else:
            if current is not None:
                sessions.append(current)
            current = {
                "t_start": start,
                "t_end": stop,
                "segment_ids": [str(_field(segment, "source_id"))],
                "continuity_group": group,
                "montage_hash": montage,
            }
    if current is not None:
        sessions.append(current)
    for index, session in enumerate(sessions):
        session["session_index"] = index
        session["segment_ids"] = tuple(session["segment_ids"])
    return sessions


def assign_events(
    sessions: Sequence[Mapping[str, Any]],
    event_times: Sequence[float],
    event_record_names: Sequence[str],
) -> list[dict[str, Any]]:
    times = np.asarray(event_times, dtype=float)
    names = np.asarray(event_record_names).astype(str)
    output = []
    for session in sessions:
        member = np.flatnonzero(np.isin(names, np.asarray(session["segment_ids"])))
        member = member[np.argsort(times[member], kind="stable")]
        row = dict(session)
        row["event_indices"] = member
        row["n_events"] = int(member.size)
        row["first_event_time"] = float(times[member].min()) if member.size else None
        row["last_event_time"] = float(times[member].max()) if member.size else None
        output.append(row)
    return output


def session_gaps(sessions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Report gaps between consecutive sessions.

    Returns a list of gap records with keys:
    - left_session, right_session (int): session indices
    - metadata_gap_seconds (float): unrecorded wall time between sessions
    - event_silence_seconds (float | None): time span between last event of left session
      and first event of right session. None if either session has zero observed events.
    - observed_events_during_gap (bool): always False
    """
    rows = []
    for left, right in zip(sessions, sessions[1:]):
        last = left.get("last_event_time")
        first = right.get("first_event_time")
        rows.append(
            {
                "left_session": int(left["session_index"]),
                "right_session": int(right["session_index"]),
                "metadata_gap_seconds": float(right["t_start"] - left["t_end"]),
                "event_silence_seconds": (
                    float(first - last) if last is not None and first is not None else None
                ),
                "observed_events_during_gap": False,
            }
        )
    return rows


def build_blocks(
    sessions: Sequence[Mapping[str, Any]],
    *,
    block_events: int,
    event_times: Sequence[float],
) -> list[dict[str, Any]]:
    size = int(block_events)
    if size < 2:
        raise ValueError("block_events must be at least 2")
    times = np.asarray(event_times, dtype=float)
    blocks: list[dict[str, Any]] = []
    previous_session: int | None = None
    previous_end: float | None = None
    for session in sessions:
        indices = np.asarray(session["event_indices"])
        for start in range(0, indices.size - size + 1, size):
            member = indices[start : start + size]
            t_start = float(times[member].min())
            t_end = float(times[member].max())
            if previous_session is None:
                stratum, delta = None, None
            else:
                stratum = (
                    "within_session"
                    if int(session["session_index"]) == previous_session
                    else "cross_gap"
                )
                delta = float(t_start - previous_end)
            blocks.append(
                {
                    "block_index": len(blocks),
                    "session_index": int(session["session_index"]),
                    "event_indices": member,
                    "t_start": t_start,
                    "t_end": t_end,
                    "delta_t_from_previous": delta,
                    "transition_stratum": stratum,
                }
            )
            previous_session = int(session["session_index"])
            previous_end = t_end
    return blocks


def dropped_remainders(
    sessions: Sequence[Mapping[str, Any]], *, block_events: int
) -> list[dict[str, Any]]:
    size = int(block_events)
    rows = []
    for session in sessions:
        total = int(np.asarray(session["event_indices"]).size)
        dropped = total if total < size else total % size
        if dropped:
            rows.append(
                {"session_index": int(session["session_index"]), "n_dropped": dropped}
            )
    return rows
