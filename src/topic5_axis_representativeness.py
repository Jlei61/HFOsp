"""Subject-first metrics for template-axis directional representativeness."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from src.topic5_interictal_direction_rose import (
    fit_endpoint_direction_3d,
    fit_event_directions_3d,
)


def summarize_direction_representativeness(
    event_directions: np.ndarray,
    axis: Sequence[float],
) -> Dict[str, float | int | np.ndarray]:
    """Summarize how a signed 3D axis represents signed event directions.

    ``mean_signed_cosine`` is the primary statistic.  It decomposes as
    ``resultant_length_3d * cos(axis_to_main_direction_deg)`` and therefore
    penalizes both a broad event distribution and a concentrated distribution
    pointing away from the template axis.
    """
    vectors = np.asarray(event_directions, float)
    u = np.asarray(axis, float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("event_directions must have shape (n_events, 3)")
    if u.shape != (3,) or not np.isfinite(u).all() or np.linalg.norm(u) < 1e-12:
        raise ValueError("axis must be a finite non-zero 3-vector")
    u = u / np.linalg.norm(u)
    valid = np.isfinite(vectors).all(axis=1)
    vectors = vectors[valid]
    if vectors.size:
        norm = np.linalg.norm(vectors, axis=1)
        vectors = vectors[np.isfinite(norm) & (norm > 1e-12)]
        if vectors.size:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    if not vectors.size:
        return {
            "n_events": 0,
            "mean_signed_cosine": float("nan"),
            "mean_angle_deg": float("nan"),
            "median_angle_deg": float("nan"),
            "fraction_within_30deg": float("nan"),
            "fraction_within_45deg": float("nan"),
            "resultant_length_3d": float("nan"),
            "axis_to_main_direction_deg": float("nan"),
            "mean_direction": np.full(3, np.nan),
        }

    cosines = np.clip(vectors @ u, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosines))
    mean_vector = vectors.mean(axis=0)
    resultant = float(np.linalg.norm(mean_vector))
    if resultant > 1e-12:
        mean_direction = mean_vector / resultant
        main_gap = float(np.degrees(np.arccos(np.clip(mean_direction @ u, -1.0, 1.0))))
    else:
        mean_direction = np.full(3, np.nan)
        main_gap = float("nan")
    return {
        "n_events": int(vectors.shape[0]),
        "mean_signed_cosine": float(cosines.mean()),
        "mean_angle_deg": float(angles.mean()),
        "median_angle_deg": float(np.median(angles)),
        "fraction_within_30deg": float(np.mean(angles <= 30.0)),
        "fraction_within_45deg": float(np.mean(angles <= 45.0)),
        "resultant_length_3d": resultant,
        "axis_to_main_direction_deg": main_gap,
        "mean_direction": mean_direction,
    }


def rank_shuffle_axis_null(
    template_rank: Sequence[float],
    coords: np.ndarray,
    event_directions: np.ndarray,
    *,
    method: str,
    n_perm: int = 1000,
    seed: int = 0,
    k_primary: int = 3,
) -> Dict[str, np.ndarray | int]:
    """Montage-controlled null by shuffling template ranks over contacts.

    Contact coordinates and the observed event-direction distribution remain
    fixed.  Only the contact-to-template-rank mapping is shuffled before the
    chosen axis estimator is rebuilt.
    """
    rank = np.asarray(template_rank, float)
    xyz = np.asarray(coords, float)
    vectors = np.asarray(event_directions, float)
    if rank.ndim != 1 or xyz.shape != (rank.size, 3):
        raise ValueError("template rank/coordinate shape mismatch")
    if method not in {"gradient", "endpoint"}:
        raise ValueError("method must be 'gradient' or 'endpoint'")
    if n_perm < 1:
        raise ValueError("n_perm must be positive")
    valid_rank = np.isfinite(rank) & np.isfinite(xyz).all(axis=1)
    indices = np.flatnonzero(valid_rank)
    if indices.size < 3:
        raise ValueError("fewer than three mapped template ranks")
    summary = summarize_direction_representativeness(vectors, [1.0, 0.0, 0.0])
    mean_direction = np.asarray(summary["mean_direction"], float)
    clean = vectors[np.isfinite(vectors).all(axis=1)]
    if not clean.size:
        raise ValueError("no finite event directions")
    clean = clean / np.linalg.norm(clean, axis=1, keepdims=True)

    rng = np.random.default_rng(seed)
    null_cosine = []
    null_main_gap = []
    max_attempts = max(20 * n_perm, n_perm + 100)
    attempts = 0
    while len(null_cosine) < n_perm and attempts < max_attempts:
        attempts += 1
        shuffled = rank.copy()
        shuffled[indices] = rank[rng.permutation(indices)]
        if method == "gradient":
            fit = fit_event_directions_3d(shuffled[:, None], xyz, min_contacts=3)
            axis = np.asarray(fit["directions"][0], float)
        else:
            fit = fit_endpoint_direction_3d(shuffled, xyz, k_primary=k_primary)
            axis = np.asarray(fit["direction"], float)
        if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
            continue
        axis = axis / np.linalg.norm(axis)
        null_cosine.append(float(np.mean(clean @ axis)))
        if np.isfinite(mean_direction).all():
            null_main_gap.append(
                float(np.degrees(np.arccos(np.clip(mean_direction @ axis, -1.0, 1.0))))
            )
        else:
            null_main_gap.append(float("nan"))
    if len(null_cosine) < n_perm:
        raise RuntimeError(
            f"only {len(null_cosine)}/{n_perm} valid shuffled axes after {attempts} attempts"
        )
    return {
        "mean_signed_cosine": np.asarray(null_cosine, float),
        "axis_to_main_direction_deg": np.asarray(null_main_gap, float),
        "attempts": int(attempts),
    }
