"""Independent windows, and what does not count as one.

Two hundred random splits estimate one window's uncertainty.  They are not two hundred
replicates, and no cohort or patient-level count may be built from them.  Primary windows
are non-overlapping tiles inside a session; sliding offsets exist only as sensitivity and
carry a non-zero `offset_fraction` so they can never be pooled with the primary by
accident.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def tile_event_windows(
    sessions: Sequence[Mapping[str, Any]], *, window_events: int
) -> list[dict[str, Any]]:
    size = int(window_events)
    if size < 2:
        raise ValueError("window_events must be at least 2")
    rows: list[dict[str, Any]] = []
    for session in sessions:
        indices = np.asarray(session["event_indices"])
        for start in range(0, indices.size - size + 1, size):
            rows.append(
                {
                    "window_index": len(rows),
                    "session_index": int(session["session_index"]),
                    "event_indices": indices[start : start + size],
                    "offset_fraction": 0.0,
                }
            )
    return rows


def sliding_event_windows(
    sessions: Sequence[Mapping[str, Any]],
    *,
    window_events: int,
    offsets: Sequence[float],
) -> list[dict[str, Any]]:
    size = int(window_events)
    rows: list[dict[str, Any]] = []
    for offset in offsets:
        if float(offset) == 0.0:
            raise ValueError("offset 0.0 is the primary tiling, not a sensitivity offset")
        shift = int(round(float(offset) * size))
        for session in sessions:
            indices = np.asarray(session["event_indices"])
            for start in range(shift, indices.size - size + 1, size):
                rows.append(
                    {
                        "window_index": len(rows),
                        "session_index": int(session["session_index"]),
                        "event_indices": indices[start : start + size],
                        "offset_fraction": float(offset),
                    }
                )
    return rows


def tile_clock_windows(
    sessions: Sequence[Mapping[str, Any]],
    event_times: Sequence[float],
    *,
    window_seconds: float,
    min_events: int,
) -> list[dict[str, Any]]:
    times = np.asarray(event_times, dtype=float)
    span = float(window_seconds)
    rows: list[dict[str, Any]] = []
    for session in sessions:
        indices = np.asarray(session["event_indices"])
        start = float(session["t_start"])
        end = float(session["t_end"])
        edge = start
        while edge + span <= end:
            member = indices[
                (times[indices] >= edge) & (times[indices] < edge + span)
            ]
            if member.size >= int(min_events):
                rows.append(
                    {
                        "window_index": len(rows),
                        "session_index": int(session["session_index"]),
                        "event_indices": member,
                        "offset_fraction": 0.0,
                        "t_start": edge,
                        "t_end": edge + span,
                    }
                )
            edge += span
    return rows


def scale_is_evaluable(windows: Sequence[Any], *, minimum: int) -> bool:
    return len(windows) >= int(minimum)
