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
    """Non-overlapping primary event windows tiled from session start.

    Returns windows as a list of dicts. The window_index is assigned by position
    within this returned list and is unique only within this call.
    """
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
    """Sensitivity variants with non-zero offsets; reject offset 0.0 upfront.

    Returns windows as a list of dicts. The window_index is assigned by position
    within this returned list and is unique only within this call.
    """
    # Validate all offsets upfront before building any windows
    for offset in offsets:
        if float(offset) == 0.0:
            raise ValueError("offset 0.0 is the primary tiling, not a sensitivity offset")

    size = int(window_events)
    rows: list[dict[str, Any]] = []
    for offset in offsets:
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
    """Non-overlapping wall-clock tiles from session metadata bounds.

    Returns windows as a list of dicts. The window_index is assigned by position
    within this returned list and is unique only within this call. Tiles that fall
    below min_events are dropped, not padded.
    """
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
    """Check whether a set of independent windows meets minimum sample count.

    The unit of measurement is the independent window (tiles or sliding offsets).
    Random splits drawn per window are not replicates and must never be counted
    as additional independent samples. This guard requires each item to carry a
    window_index key to distinguish windows from other collections (e.g., 200
    random splits).

    Raises ValueError if any item in windows is not a mapping or lacks window_index.
    """
    for item in windows:
        if not isinstance(item, Mapping):
            raise ValueError(
                f"scale_is_evaluable expects a list of windows (dicts with window_index), "
                f"not {type(item).__name__}"
            )
        if "window_index" not in item:
            raise ValueError(
                "scale_is_evaluable expects each item to have a window_index key; "
                "ensure you are passing windows, not random splits or other collections"
            )
    return len(windows) >= int(minimum)
