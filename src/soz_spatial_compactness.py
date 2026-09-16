"""Pure statistics for clinical-SOZ contact spatial compactness.

The primary question is subject-level: are mapped clinical SOZ contacts more
spatially compact than equally sized subsets drawn from all mapped invasive
contacts in the same patient?  A shaft-stratified null is a prespecified
sensitivity analysis that preserves the observed number of SOZ contacts on
each electrode lead/array (using the channel-name prefix as the grouping unit).
"""
from __future__ import annotations

from math import comb
from typing import Sequence

import numpy as np


def rms_radius(coords: np.ndarray) -> float:
    """Root-mean-square distance (mm) to a point set's own centroid."""
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("coords must have shape (n_contacts, 3)")
    if pts.shape[0] < 2 or not np.isfinite(pts).all():
        return float("nan")
    centroid = pts.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((pts - centroid) ** 2, axis=1))))


def median_pairwise_distance(coords: np.ndarray) -> float:
    """Median unique-pair Euclidean distance (mm) within a point set."""
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("coords must have shape (n_contacts, 3)")
    if pts.shape[0] < 2 or not np.isfinite(pts).all():
        return float("nan")
    delta = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)
    tri = np.triu_indices(pts.shape[0], k=1)
    return float(np.median(dist[tri]))


def spatial_metrics(coords: np.ndarray) -> dict[str, float]:
    """Primary RMS radius plus robust pairwise-distance sensitivity."""
    return {
        "rms_radius_mm": rms_radius(coords),
        "median_pairwise_mm": median_pairwise_distance(coords),
    }


def _summarize_null(values: np.ndarray, observed: float) -> dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    base = {
        "observed": float(observed),
        "null_median": float("nan"),
        "null_q025": float("nan"),
        "null_q975": float("nan"),
        "observed_to_null_median_ratio": float("nan"),
        "p_left": float("nan"),
        "n_null": int(vals.size),
    }
    if vals.size == 0 or not np.isfinite(observed):
        return base
    med = float(np.median(vals))
    # Add-one correction prevents a Monte-Carlo p-value of exactly zero.
    p_left = float((1 + np.count_nonzero(vals <= observed)) / (vals.size + 1))
    base.update(
        null_median=med,
        null_q025=float(np.percentile(vals, 2.5)),
        null_q975=float(np.percentile(vals, 97.5)),
        observed_to_null_median_ratio=(float(observed / med) if med > 0 else float("nan")),
        p_left=p_left,
    )
    return base


def _draw_metrics(coords: np.ndarray, draws: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rms = np.empty(len(draws), dtype=float)
    pairwise = np.empty(len(draws), dtype=float)
    for i, idx in enumerate(draws):
        metrics = spatial_metrics(coords[np.asarray(idx, dtype=int)])
        rms[i] = metrics["rms_radius_mm"]
        pairwise[i] = metrics["median_pairwise_mm"]
    return rms, pairwise


def all_contact_null(
    coords: np.ndarray,
    *,
    n_selected: int,
    observed_metrics: dict[str, float],
    n_null: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Equally sized subsets drawn from all mapped contacts in a subject."""
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise ValueError("coords must be a finite (n_contacts, 3) array")
    if n_selected < 2 or n_selected >= pts.shape[0]:
        raise ValueError("n_selected must be >=2 and smaller than n_contacts")
    if n_null < 1:
        raise ValueError("n_null must be positive")
    draws = [rng.choice(pts.shape[0], size=n_selected, replace=False) for _ in range(n_null)]
    rms, pairwise = _draw_metrics(pts, draws)
    return {
        "available": True,
        "definition": "equal-size subsets from all mapped invasive contacts",
        "rms_radius": _summarize_null(rms, observed_metrics["rms_radius_mm"]),
        "median_pairwise": _summarize_null(
            pairwise, observed_metrics["median_pairwise_mm"]
        ),
    }


def shaft_stratified_null(
    coords: np.ndarray,
    shaft_ids: Sequence[str],
    selected_mask: Sequence[bool],
    *,
    observed_metrics: dict[str, float],
    n_null: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Null preserving the exact selected-contact count on every lead/array.

    This conditions on coarse lead/array allocation and tests whether SOZ
    contacts are additionally compact because of their within-lead positions.
    When the grouping profile uniquely determines the selected set, the sensitivity
    is marked unavailable rather than returning a meaningless p-value.
    """
    pts = np.asarray(coords, dtype=float)
    shafts = np.asarray(list(shaft_ids), dtype=object)
    selected = np.asarray(selected_mask, dtype=bool)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise ValueError("coords must be a finite (n_contacts, 3) array")
    if shafts.shape != (pts.shape[0],) or selected.shape != (pts.shape[0],):
        raise ValueError("shaft_ids and selected_mask must align to coords")
    if any(s is None or str(s) == "" for s in shafts):
        return {"available": False, "reason": "unparseable_shaft", "n_unique_profiles": 0}
    if selected.sum() < 2 or selected.sum() >= pts.shape[0]:
        return {"available": False, "reason": "invalid_selected_count", "n_unique_profiles": 0}

    groups: list[tuple[np.ndarray, int]] = []
    n_unique = 1
    for shaft in sorted(set(str(s) for s in shafts)):
        idx = np.where(shafts == shaft)[0]
        k = int(selected[idx].sum())
        if k > 0:
            groups.append((idx, k))
            n_unique *= comb(int(idx.size), k)
    if n_unique <= 1:
        return {
            "available": False,
            "reason": "shaft_profile_has_no_permutation_freedom",
            "n_unique_profiles": int(n_unique),
        }

    draws: list[np.ndarray] = []
    for _ in range(n_null):
        parts = [rng.choice(idx, size=k, replace=False) for idx, k in groups]
        draws.append(np.concatenate(parts))
    rms, pairwise = _draw_metrics(pts, draws)
    return {
        "available": True,
        "definition": "exact SOZ-contact count preserved within every electrode lead/array",
        "n_unique_profiles": int(n_unique),
        "rms_radius": _summarize_null(rms, observed_metrics["rms_radius_mm"]),
        "median_pairwise": _summarize_null(
            pairwise, observed_metrics["median_pairwise_mm"]
        ),
    }


def analyze_subject_compactness(
    contact_names: Sequence[str],
    coords: np.ndarray,
    soz_names: Sequence[str],
    shaft_ids: Sequence[str],
    *,
    n_null: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Run primary and shaft-stratified compactness tests for one subject."""
    names = list(contact_names)
    pts = np.asarray(coords, dtype=float)
    if len(names) != pts.shape[0] or len(set(names)) != len(names):
        raise ValueError("contact_names must be unique and align to coords")
    soz_set = set(soz_names)
    selected = np.asarray([name in soz_set for name in names], dtype=bool)
    n_soz = int(selected.sum())
    if n_soz < 2 or n_soz >= len(names):
        raise ValueError("mapped SOZ set must contain >=2 contacts and leave a control contact")
    observed = spatial_metrics(pts[selected])
    all_metrics = spatial_metrics(pts)
    return {
        "n_contacts": int(len(names)),
        "n_soz_contacts": n_soz,
        "n_nonsoz_contacts": int(len(names) - n_soz),
        "observed_soz": observed,
        "all_contacts": all_metrics,
        "soz_to_all_rms_ratio": float(observed["rms_radius_mm"] / all_metrics["rms_radius_mm"]),
        "all_contact_null": all_contact_null(
            pts,
            n_selected=n_soz,
            observed_metrics=observed,
            n_null=n_null,
            rng=rng,
        ),
        "shaft_stratified_null": shaft_stratified_null(
            pts,
            shaft_ids,
            selected,
            observed_metrics=observed,
            n_null=n_null,
            rng=rng,
        ),
    }
