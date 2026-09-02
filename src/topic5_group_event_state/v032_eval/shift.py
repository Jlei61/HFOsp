"""Predefined circular shifts of a state trajectory inside one recording session.

The shift set is a fixed function of the session length (fractions j/denominator
of its anchor count), so no shift can be chosen because it flatters an arm.  A
donor is kept only when it is further than the target horizon (+ margin) from
the anchor it feeds, so the shifted state cannot leak the same future block.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def predefined_session_shifts(
    times: np.ndarray,
    session_of: np.ndarray,
    *,
    n_shifts: int = 5,
    denominator: int = 6,
    min_distance_seconds: float,
) -> list[dict[str, Any]]:
    t = np.asarray(times, dtype=np.float64)
    session = np.asarray(session_of, dtype=np.int64)
    if t.shape != session.shape:
        raise ValueError("times and session_of must align")
    if not (1 <= n_shifts < denominator):
        raise ValueError("need 1 <= n_shifts < denominator")
    specs: list[dict[str, Any]] = []
    sessions = sorted(int(s) for s in np.unique(session))
    for j in range(1, n_shifts + 1):
        donor = np.full(t.size, -1, dtype=np.int64)
        by_session: dict[int, int] = {}
        for s in sessions:
            idx = np.flatnonzero(session == s)
            idx = idx[np.argsort(t[idx], kind="stable")]
            n = idx.size
            k = int(round(n * j / denominator))
            by_session[s] = k
            if n < 3 or k <= 0 or k >= n:
                continue
            rolled = idx[(np.arange(n) + k) % n]
            distance = np.abs(t[idx] - t[rolled])
            ok = distance > float(min_distance_seconds)
            donor[idx[ok]] = rolled[ok]
        specs.append({
            "shift_id": j,
            "fraction": j / denominator,
            "shift_anchors_by_session": by_session,
            "donor_index": donor,
            "n_valid": int((donor >= 0).sum()),
            "min_distance_seconds": float(min_distance_seconds),
        })
    return specs


def apply_donor(values: np.ndarray, donor_index: np.ndarray) -> np.ndarray:
    """Return ``values`` re-indexed by donor; rows without a donor become NaN."""

    v = np.asarray(values, dtype=np.float64)
    donor = np.asarray(donor_index, dtype=np.int64)
    out = np.full(v.shape, np.nan, dtype=np.float64)
    ok = donor >= 0
    out[ok] = v[donor[ok]]
    return out
