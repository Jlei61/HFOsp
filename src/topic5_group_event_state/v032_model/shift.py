"""Session-preserving wrong-time donors for the shifted-state control."""

from __future__ import annotations

import numpy as np


def block_circular_donor(
    t_anchor: np.ndarray,
    segment: np.ndarray,
    indices: np.ndarray,
    *,
    horizon: float,
    fraction: float = 0.5,
) -> np.ndarray:
    """Circular shift of the anchor list *within each segment* by ``fraction``.

    Returns, for every entry of ``indices``, the local position (into
    ``indices``) of its donor, or ``-1`` when no donor is admissible: the donor
    must lie in the same segment (hence the same recording session) and at
    least one horizon away, and a segment with fewer than three anchors is
    skipped because a shift there is either the identity or its neighbour.
    """

    indices = np.asarray(indices, dtype=np.int64)
    donor = np.full(indices.size, -1, dtype=np.int64)
    seg_of = np.asarray(segment)[indices]
    times = np.asarray(t_anchor, dtype=np.float64)[indices]
    for seg in np.unique(seg_of):
        local = np.flatnonzero(seg_of == seg)
        if local.size < 3:
            continue
        order = local[np.argsort(times[local], kind="stable")]
        shift = int(round(order.size * float(fraction))) % order.size
        shift = max(shift, 1)
        candidate = np.roll(order, shift)
        far = np.abs(times[order] - times[candidate]) >= float(horizon)
        donor[order[far]] = candidate[far]
    return donor
