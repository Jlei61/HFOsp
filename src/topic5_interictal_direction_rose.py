"""Pure geometry helpers for interictal-only TA/TB direction roses.

The frozen Topic 5 template axes are three-dimensional least-squares gradients
whose positive direction is early-to-late.  These helpers estimate the same
quantity for individual masked-rank events, then express those 3D directions in
the plane spanned by the frozen TA and TB axes.  No ictal data enter this module.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from src.dab_gradient_axis import RCOND
from src.propagation_skeleton_geometry import build_endpoint_cores, compute_axis_frame

TWO_PI = 2.0 * np.pi


def _unit(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, float)
    norm = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("direction must be a finite non-zero 3-vector")
    return value / norm


def _fit_single_direction(
    rank: np.ndarray,
    xyz: np.ndarray,
    use: np.ndarray,
    *,
    rcond: float,
) -> tuple[np.ndarray, int]:
    """Fit one signed rank gradient and return its geometric matrix rank."""
    direction = np.full(3, np.nan, float)
    if int(use.sum()) < 3 or float(np.std(rank[use])) < 1e-12:
        return direction, 0
    x = xyz[use]
    x_centered = x - x.mean(axis=0)
    y = rank[use] - rank[use].mean()
    singular = np.linalg.svd(x_centered, compute_uv=False)
    effective_rank = 0
    if singular.size and singular.max() > 0:
        effective_rank = int((singular >= rcond * singular.max()).sum())
    beta, *_ = np.linalg.lstsq(x_centered, y, rcond=rcond)
    norm = float(np.linalg.norm(beta))
    if np.isfinite(norm) and norm >= 1e-12:
        direction = beta / norm
    return direction, effective_rank


def fit_event_directions_3d(
    event_ranks: np.ndarray,
    coords: np.ndarray,
    *,
    min_contacts: int = 3,
    rcond: float = RCOND,
) -> Dict[str, np.ndarray]:
    """Fit early-to-late 3D directions for masked-rank events.

    ``event_ranks`` is ``(n_contacts, n_events)`` with NaN for contacts that
    did not participate.  Rank increases from early to late, so the normalized
    least-squares gradient of rank is already the early-to-late propagation
    direction used by ``template_propagation_axis_v2``.  Events with too few
    mapped contacts, constant rank, or a degenerate gradient remain NaN.
    """
    values = np.asarray(event_ranks, float)
    xyz = np.asarray(coords, float)
    if values.ndim != 2:
        raise ValueError("event_ranks must be 2D (n_contacts, n_events)")
    if xyz.shape != (values.shape[0], 3):
        raise ValueError("coords must have shape (n_contacts, 3)")
    if min_contacts < 3:
        raise ValueError("min_contacts must be at least 3")

    n_events = values.shape[1]
    directions = np.full((n_events, 3), np.nan, float)
    n_valid = np.zeros(n_events, int)
    effective_rank = np.zeros(n_events, int)
    coord_ok = np.isfinite(xyz).all(axis=1)

    for event_index in range(n_events):
        rank = values[:, event_index]
        use = coord_ok & np.isfinite(rank)
        n_valid[event_index] = int(use.sum())
        if n_valid[event_index] < min_contacts:
            continue
        directions[event_index], effective_rank[event_index] = _fit_single_direction(
            rank, xyz, use, rcond=rcond
        )

    return {
        "directions": directions,
        "n_valid_contacts": n_valid,
        "effective_rank": effective_rank,
    }


def assess_event_direction_qc(
    event_ranks: np.ndarray,
    coords: np.ndarray,
    shafts: Sequence[str],
    *,
    directions: Optional[np.ndarray] = None,
    n_valid_contacts: Optional[np.ndarray] = None,
    effective_rank: Optional[np.ndarray] = None,
    min_contacts: int = 6,
    min_shafts: int = 2,
    min_effective_rank: int = 2,
    min_loco_valid_fraction: float = 0.8,
    min_loco_median_signed_cosine: float = 0.8,
    rcond: float = RCOND,
) -> Dict[str, np.ndarray]:
    """Apply geometry and leave-one-contact-out stability QC to event axes.

    The gate is deliberately independent of either frozen template direction:
    an event is retained only when its own contact geometry can estimate a
    stable signed gradient.  A leave-one-contact-out (LOCO) refit counts as
    valid only when it remains at least two-dimensional.  Its signed cosine is
    measured against the full-event direction, so polarity flips are treated
    as instability rather than hidden with an absolute cosine.
    """
    values = np.asarray(event_ranks, float)
    xyz = np.asarray(coords, float)
    shaft_values = np.asarray(shafts, object)
    if values.ndim != 2:
        raise ValueError("event_ranks must be 2D (n_contacts, n_events)")
    if xyz.shape != (values.shape[0], 3):
        raise ValueError("coords must have shape (n_contacts, 3)")
    if shaft_values.shape != (values.shape[0],):
        raise ValueError("shafts must have shape (n_contacts,)")
    if min_contacts < 3 or min_shafts < 1 or min_effective_rank < 1:
        raise ValueError("invalid event-direction QC thresholds")
    if not 0.0 <= min_loco_valid_fraction <= 1.0:
        raise ValueError("min_loco_valid_fraction must be in [0, 1]")
    if not -1.0 <= min_loco_median_signed_cosine <= 1.0:
        raise ValueError("min_loco_median_signed_cosine must be in [-1, 1]")

    n_events = values.shape[1]
    if directions is None or n_valid_contacts is None or effective_rank is None:
        fitted = fit_event_directions_3d(values, xyz, min_contacts=3, rcond=rcond)
        directions = fitted["directions"]
        n_valid_contacts = fitted["n_valid_contacts"]
        effective_rank = fitted["effective_rank"]
    direction_values = np.asarray(directions, float)
    n_valid = np.asarray(n_valid_contacts, int)
    ranks = np.asarray(effective_rank, int)
    if direction_values.shape != (n_events, 3):
        raise ValueError("directions must have shape (n_events, 3)")
    if n_valid.shape != (n_events,) or ranks.shape != (n_events,):
        raise ValueError("event QC arrays must have shape (n_events,)")

    n_shafts = np.zeros(n_events, int)
    loco_attempted = np.zeros(n_events, int)
    loco_valid = np.zeros(n_events, int)
    loco_valid_fraction = np.full(n_events, np.nan, float)
    loco_median_signed_cosine = np.full(n_events, np.nan, float)
    passes = np.zeros(n_events, bool)
    coord_ok = np.isfinite(xyz).all(axis=1)

    for event_index in range(n_events):
        rank = values[:, event_index]
        use = coord_ok & np.isfinite(rank)
        indices = np.flatnonzero(use)
        n_shafts[event_index] = int(np.unique(shaft_values[use]).size)
        if (
            n_valid[event_index] < min_contacts
            or n_shafts[event_index] < min_shafts
            or ranks[event_index] < min_effective_rank
            or not np.isfinite(direction_values[event_index]).all()
        ):
            continue

        signed_cosines = []
        loco_attempted[event_index] = int(indices.size)
        for omitted in indices:
            keep = use.copy()
            keep[omitted] = False
            loco_direction, loco_rank = _fit_single_direction(rank, xyz, keep, rcond=rcond)
            if loco_rank < min_effective_rank or not np.isfinite(loco_direction).all():
                continue
            signed_cosines.append(
                float(np.clip(loco_direction @ direction_values[event_index], -1.0, 1.0))
            )

        loco_valid[event_index] = len(signed_cosines)
        loco_valid_fraction[event_index] = (
            float(len(signed_cosines) / indices.size) if indices.size else float("nan")
        )
        if signed_cosines:
            loco_median_signed_cosine[event_index] = float(np.median(signed_cosines))
        passes[event_index] = bool(
            loco_valid_fraction[event_index] >= min_loco_valid_fraction
            and loco_median_signed_cosine[event_index] >= min_loco_median_signed_cosine
        )

    return {
        "passes": passes,
        "n_valid_contacts": n_valid,
        "n_shafts": n_shafts,
        "effective_rank": ranks,
        "loco_n_attempted": loco_attempted,
        "loco_n_valid": loco_valid,
        "loco_valid_fraction": loco_valid_fraction,
        "loco_median_signed_cosine": loco_median_signed_cosine,
    }


def fit_endpoint_direction_3d(
    rank: Sequence[float],
    coords: np.ndarray,
    *,
    k_primary: int = 3,
) -> Dict[str, object]:
    """Fit the legacy source-to-sink endpoint-centroid direction.

    This reuses the frozen endpoint-core contract: top/bottom ``k=3`` when
    at least seven mapped participants are available, ``k=2`` fallback for
    five or six, and no direction otherwise.  Smaller rank means earlier, so
    the returned vector points from the early/source centroid to the
    late/sink centroid.
    """
    rank_values = np.asarray(rank, float)
    xyz = np.asarray(coords, float)
    if rank_values.ndim != 1:
        raise ValueError("rank must be one-dimensional")
    if xyz.shape != (rank_values.size, 3):
        raise ValueError("coords must have shape (n_contacts, 3)")
    eligible = np.isfinite(rank_values) & np.isfinite(xyz).all(axis=1)
    base: Dict[str, object] = {
        "direction": np.full(3, np.nan, float),
        "n_valid_contacts": int(eligible.sum()),
        "k_used": 0,
        "tier": "descriptive_only",
        "axis_length": float("nan"),
        "source_idx": [],
        "sink_idx": [],
    }
    if int(eligible.sum()) == 0 or float(np.ptp(rank_values[eligible])) < 1e-12:
        return base
    cores = build_endpoint_cores(rank_values, eligible, k_primary=k_primary)
    base.update(
        k_used=int(cores["k_used"]),
        tier=str(cores["tier"]),
        source_idx=list(cores["source_idx"]),
        sink_idx=list(cores["sink_idx"]),
    )
    if cores["tier"] == "descriptive_only":
        return base
    frame = compute_axis_frame(xyz, cores["source_idx"], cores["sink_idx"])
    base["axis_length"] = float(frame["axis_length"])
    if bool(frame["degenerate_axis"]) or float(frame["axis_length"]) < 1e-12:
        return base
    source = np.asarray(frame["source_centroid"], float)
    sink = np.asarray(frame["sink_centroid"], float)
    base["direction"] = (sink - source) / float(frame["axis_length"])
    return base


def fit_event_endpoint_directions_3d(
    event_ranks: np.ndarray,
    coords: np.ndarray,
    *,
    k_primary: int = 3,
) -> Dict[str, np.ndarray]:
    """Apply the legacy endpoint-centroid estimator to masked-rank events."""
    values = np.asarray(event_ranks, float)
    xyz = np.asarray(coords, float)
    if values.ndim != 2:
        raise ValueError("event_ranks must be 2D (n_contacts, n_events)")
    if xyz.shape != (values.shape[0], 3):
        raise ValueError("coords must have shape (n_contacts, 3)")

    n_events = values.shape[1]
    directions = np.full((n_events, 3), np.nan, float)
    n_valid = np.zeros(n_events, int)
    k_used = np.zeros(n_events, int)
    axis_length = np.full(n_events, np.nan, float)
    tiers = np.full(n_events, "descriptive_only", dtype=object)
    for event_index in range(n_events):
        fitted = fit_endpoint_direction_3d(
            values[:, event_index], xyz, k_primary=k_primary
        )
        directions[event_index] = fitted["direction"]
        n_valid[event_index] = fitted["n_valid_contacts"]
        k_used[event_index] = fitted["k_used"]
        axis_length[event_index] = fitted["axis_length"]
        tiers[event_index] = fitted["tier"]
    return {
        "directions": directions,
        "n_valid_contacts": n_valid,
        "k_used": k_used,
        "axis_length": axis_length,
        "tier": tiers,
    }


def axis_pair_display_basis(
    axis_a: Sequence[float],
    axis_b: Sequence[float],
    *,
    fallback_transverse: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Return an orthonormal display basis with frozen TA at zero degrees.

    When TA and TB are not collinear, the display plane is their exact span and
    TB is placed in the positive half-plane.  For a nearly collinear pair, the
    frozen TA transverse direction is used as the otherwise undefined second
    basis vector.  This choice affects only the display handedness, never either
    frozen 3D template axis.
    """
    u_a = _unit(axis_a)
    u_b = _unit(axis_b)
    cosine = float(np.clip(u_a @ u_b, -1.0, 1.0))
    residual = u_b - cosine * u_a
    residual_norm = float(np.linalg.norm(residual))
    source = "ta_tb_axis_span"

    if residual_norm >= 1e-8:
        transverse = residual / residual_norm
    else:
        source = "frozen_ta_transverse_fallback"
        if fallback_transverse is None:
            base = np.array([1.0, 0.0, 0.0])
            if abs(float(base @ u_a)) > 0.8:
                base = np.array([0.0, 1.0, 0.0])
        else:
            base = _unit(fallback_transverse)
        transverse = base - float(base @ u_a) * u_a
        norm = float(np.linalg.norm(transverse))
        if norm < 1e-12:
            raise ValueError("fallback transverse direction is collinear with axis_a")
        transverse /= norm
        if float(u_b @ transverse) < 0:
            transverse = -transverse

    theta_b = float(np.mod(np.arctan2(u_b @ transverse, u_b @ u_a), TWO_PI))
    return {
        "axis_a": u_a,
        "transverse": transverse,
        "theta_b_rad": theta_b,
        "theta_b_deg": float(np.degrees(theta_b)),
        "cosine": cosine,
        "basis_source": source,
    }


def project_directions_to_angles(
    directions: np.ndarray,
    axis_a: Sequence[float],
    transverse: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Project 3D direction vectors into a frozen two-axis display plane."""
    vectors = np.asarray(directions, float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("directions must have shape (n_events, 3)")
    x_axis = _unit(axis_a)
    y_axis = _unit(transverse)
    x_comp = vectors @ x_axis
    y_comp = vectors @ y_axis
    projection_norm = np.hypot(x_comp, y_comp)
    angles = np.full(vectors.shape[0], np.nan, float)
    valid = np.isfinite(vectors).all(axis=1) & np.isfinite(projection_norm) & (projection_norm > 1e-12)
    angles[valid] = np.mod(np.arctan2(y_comp[valid], x_comp[valid]), TWO_PI)
    return {"angles": angles, "projection_norm": projection_norm}


def resultant_length(angles: Sequence[float]) -> float:
    """Circular resultant length in [0, 1], or NaN for an empty input."""
    values = np.asarray(angles, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.hypot(np.cos(values).mean(), np.sin(values).mean()))
