"""Budget-matched hand-placed dual-core fields for Topic 4 controls."""
from __future__ import annotations

import hashlib

import numpy as np


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def distances_to_dual_core(
    positions: np.ndarray, centers_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return distance and nearest-center index for every sheet position."""
    positions = np.asarray(positions, dtype=float)
    centers = np.asarray(centers_mm, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (n, 2)")
    if centers.shape != (2, 2) or not np.isfinite(centers).all():
        raise ValueError("dual-core centers must be finite with shape (2, 2)")
    distance = np.linalg.norm(
        positions[:, None, :] - centers[None, :, :], axis=2,
    )
    nearest = np.argmin(distance, axis=1)
    return distance[np.arange(len(positions)), nearest], nearest


def budget_matched_dual_core_h(
    positions: np.ndarray,
    centers_mm: np.ndarray,
    *,
    target_count: int,
) -> tuple[np.ndarray, dict]:
    """Build two hand-placed binary cores with an exact total E-node budget.

    Nodes are selected by distance to the nearer frozen center. Ties are broken
    by neuron index, making the field deterministic for a frozen network.
    """
    positions = np.asarray(positions, dtype=float)
    distance, nearest = distances_to_dual_core(positions, centers_mm)
    target_count = int(target_count)
    if target_count <= 0 or target_count > len(positions):
        raise ValueError("target_count must lie in [1, n_positions]")
    order = np.lexsort((np.arange(len(distance), dtype=np.int64), distance))
    selected = order[:target_count]
    h = np.zeros(len(positions), dtype=float)
    h[selected] = 1.0
    selected_distance = distance[selected]
    selected_nearest = nearest[selected]
    cutoff = float(np.max(selected_distance))
    strict_count = int(np.sum(distance < cutoff))
    audit = {
        "field_type": "manual_dual_core_budget_matched",
        "target_count": target_count,
        "selected_count": int(np.sum(h)),
        "distance_cutoff_mm": cutoff,
        "strictly_inside_cutoff_count": strict_count,
        "boundary_tie_count": int(np.sum(np.isclose(
            distance, cutoff, rtol=0.0, atol=1e-12,
        ))),
        "selected_per_core": [
            int(np.sum(selected_nearest == core)) for core in range(2)
        ],
        "maximum_selected_distance_per_core_mm": [
            float(np.max(selected_distance[selected_nearest == core]))
            if np.any(selected_nearest == core) else None
            for core in range(2)
        ],
        "h_sha256": _array_sha256(h),
    }
    return h, audit


def dual_core_query_h(
    positions: np.ndarray,
    centers_mm: np.ndarray,
    *,
    distance_cutoff_mm: float,
) -> np.ndarray:
    """Evaluate the frozen binary geometry on non-E query positions."""
    distance, _ = distances_to_dual_core(positions, centers_mm)
    cutoff = float(distance_cutoff_mm)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("distance cutoff must be finite and positive")
    return (distance <= cutoff).astype(float)
