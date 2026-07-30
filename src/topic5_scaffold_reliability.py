"""Target-blind reliability metrics for interictal contact fields."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import rankdata


def participation_field(group_ids: np.ndarray) -> np.ndarray:
    """Event-first contact participation probability."""
    groups = np.asarray(group_ids)
    if groups.ndim != 2 or groups.shape[0] == 0:
        raise ValueError("group_ids must be a nonempty [event, contact] array")
    return np.mean(groups >= 0, axis=0, dtype=np.float64)


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation of midranks with explicit constant handling."""
    left = rankdata(np.asarray(left, dtype=np.float64))
    right = rankdata(np.asarray(right, dtype=np.float64))
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return float("nan")
    return float(left @ right / denominator)


def top_fraction_jaccard(
    left: np.ndarray, right: np.ndarray, *, fraction: float = 0.25
) -> float:
    """Jaccard of deterministic top-field contacts."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("fields must be aligned one-dimensional arrays")
    n_top = max(1, int(np.ceil(len(left) * float(fraction))))
    left_top = set(np.argsort(-left, kind="stable")[:n_top].tolist())
    right_top = set(np.argsort(-right, kind="stable")[:n_top].tolist())
    return float(len(left_top & right_top) / len(left_top | right_top))


def field_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Return the frozen contact-field reliability metrics."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("fields must be aligned one-dimensional arrays")
    return {
        "spearman_rho": rank_correlation(left, right),
        "top_quartile_jaccard": top_fraction_jaccard(left, right),
        "mean_absolute_error": float(np.mean(np.abs(left - right))),
    }


def event_count_saturation(
    train_group_ids: np.ndarray,
    reference_field: np.ndarray,
    *,
    event_counts: Iterable[int],
    n_subsamples: int,
    seed: int,
) -> list[dict[str, float | int]]:
    """Estimate field reliability from deterministic train-only subsamples."""
    groups = np.asarray(train_group_ids)
    if groups.ndim != 2 or groups.shape[0] == 0:
        raise ValueError("train_group_ids must be nonempty [event, contact]")
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, float | int]] = []
    for count in event_counts:
        count = int(count)
        if count < 1 or count > len(groups):
            continue
        for draw in range(int(n_subsamples)):
            indices = rng.choice(len(groups), size=count, replace=False)
            metric = field_comparison(
                participation_field(groups[indices]), reference_field
            )
            rows.append(
                {
                    "event_count": count,
                    "draw": draw,
                    **metric,
                }
            )
    return rows
