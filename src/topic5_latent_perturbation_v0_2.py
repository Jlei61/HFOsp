"""Frozen numerical helpers for Topic 5.2 state perturbations.

The module is interictal-only.  It contains no early-ictal or SNN reader and
does not mutate model parameters.
"""
from __future__ import annotations

import hashlib

import numpy as np


DOSES = np.asarray([0.25, 0.50, 1.00], dtype=np.float64)
PRIMARY_DOSE = 0.50
PHASE_TARGETS = np.asarray([0.25, 0.50, 0.75], dtype=np.float64)
SUPPORT_K = 5
NODE_RANGE_TOLERANCE_FRACTION = 0.05
NODE_RANGE_TOLERANCE_FLOOR = 1e-4


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def unit_vector(value: np.ndarray) -> tuple[np.ndarray, bool]:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-10:
        return np.full_like(vector, np.nan), False
    return vector / norm, True


def residual_covariance_direction_sd(
    direction: np.ndarray,
    eigenvalues: np.ndarray,
    components: np.ndarray,
    diagonal: np.ndarray,
) -> float:
    """Directional SD from low-rank covariance plus diagonal correction."""
    direction, valid = unit_vector(direction)
    if not valid:
        return float("nan")
    values = np.asarray(eigenvalues, dtype=np.float64)
    vectors = np.asarray(components, dtype=np.float64)
    diag = np.asarray(diagonal, dtype=np.float64)
    keep = np.isfinite(values) & np.isfinite(vectors).all(axis=0) & (values >= 0)
    values = values[keep]
    vectors = vectors[:, keep]
    low_rank_diag = np.sum(vectors**2 * values[None, :], axis=1) if len(values) else 0.0
    correction = np.maximum(np.where(np.isfinite(diag), diag, 0.0) - low_rank_diag, 0.0)
    variance = float(np.sum(values * (direction @ vectors) ** 2)) if len(values) else 0.0
    variance += float(np.sum(correction * direction**2))
    return float(np.sqrt(max(variance, 0.0)))


def local_residual_normal_directions(
    components: np.ndarray,
    progress: np.ndarray,
    field: np.ndarray,
    count: int = 8,
) -> np.ndarray:
    """Take deterministic leading local-residual PCs normal to both axes."""
    basis: list[np.ndarray] = []
    for candidate in (progress, field):
        vector, valid = unit_vector(candidate)
        if valid:
            for previous in basis:
                vector -= previous * float(np.dot(previous, vector))
            vector, valid = unit_vector(vector)
            if valid:
                basis.append(vector)
    normals: list[np.ndarray] = []
    for candidate in np.asarray(components, dtype=np.float64).T:
        if not np.isfinite(candidate).all():
            continue
        vector = candidate.copy()
        for previous in (*basis, *normals):
            vector -= previous * float(np.dot(previous, vector))
        vector, valid = unit_vector(vector)
        if valid:
            normals.append(vector)
        if len(normals) == int(count):
            break
    if len(normals) != int(count):
        return np.full((int(count), len(progress)), np.nan, dtype=np.float64)
    return np.stack(normals)


def centered_unit_field(value: np.ndarray) -> tuple[np.ndarray, bool]:
    field = np.asarray(value, dtype=np.float64).copy()
    finite = np.isfinite(field)
    if int(finite.sum()) < 2:
        return np.full_like(field, np.nan), False
    field[~finite] = 0.0
    field[finite] -= float(np.mean(field[finite]))
    return unit_vector(field)


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def support_flags(
    hidden: np.ndarray,
    *,
    node_lower: np.ndarray,
    node_upper: np.ndarray,
    feature: np.ndarray,
    neighbor_model: object,
    knn_q95: float,
    residual_norm: float,
    residual_q95: float,
) -> tuple[bool, bool, bool, float]:
    node_ok = bool(np.all(hidden >= node_lower) and np.all(hidden <= node_upper))
    distance = float(neighbor_model.kneighbors(
        np.asarray(feature, dtype=np.float64)[None, :], return_distance=True
    )[0][0, -1])
    knn_ok = bool(np.isfinite(distance) and distance <= float(knn_q95))
    residual_ok = bool(np.isfinite(residual_norm) and residual_norm <= float(residual_q95))
    return node_ok, knn_ok, residual_ok, distance
