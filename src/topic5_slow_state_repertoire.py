"""The three primary repertoire descriptors of one event window.

Ties are read from `event_group_ids`, never from equal `event_local_rank` values: the
stored rank is a normalised rank among participating contacts, so equal values do not
recover the recruitment-group structure.  Contacts sharing a group are simultaneous and
contribute 0.5 to precedence.

Each family is returned separately and there is no combined score, so no caller can let
one surviving correlation coefficient stand in for the whole repertoire.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.stats import spearmanr

FAMILIES = ("participation", "mean_rank", "precedence")


def local_repertoire(
    rank_field: np.ndarray,
    participation: np.ndarray,
    group_ids: np.ndarray,
    *,
    min_participation_count: int,
    min_pair_count: int,
) -> dict[str, Any]:
    rank = np.asarray(rank_field, dtype=float)
    part = np.asarray(participation).astype(bool)
    groups = np.asarray(group_ids)
    if not (rank.shape == part.shape == groups.shape):
        raise ValueError("rank, participation and group_ids must share a shape")
    n_events, n_contacts = rank.shape

    counts = part.sum(axis=0).astype(float)
    rate = counts / float(n_events) if n_events else np.zeros(n_contacts)
    with np.errstate(invalid="ignore"):
        summed = np.where(part, rank, 0.0).sum(axis=0)
        mean_rank = np.where(counts > 0, summed / np.maximum(counts, 1.0), np.nan)
    mean_rank = np.where(counts >= int(min_participation_count), mean_rank, np.nan)

    pair_index: list[tuple[int, int]] = []
    precedence: list[float] = []
    pair_support: list[int] = []
    for i in range(n_contacts):
        for j in range(i + 1, n_contacts):
            both = part[:, i] & part[:, j]
            support = int(both.sum())
            pair_index.append((i, j))
            pair_support.append(support)
            if support < int(min_pair_count):
                precedence.append(np.nan)
                continue
            left, right = groups[both, i], groups[both, j]
            earlier = float(np.sum(left < right))
            tied = float(np.sum(left == right))
            precedence.append((earlier + 0.5 * tied) / support)

    supported_contacts = int(np.sum(counts >= int(min_participation_count)))
    supported_pairs = int(np.sum(np.asarray(pair_support) >= int(min_pair_count)))
    status = "RESOLVED"
    if supported_contacts < 3:
        status = "UNRESOLVED_TOO_FEW_CONTACTS"
    elif supported_pairs < 3:
        status = "UNRESOLVED_TOO_FEW_PAIRS"
    return {
        "participation_rate": rate,
        "masked_mean_rank": mean_rank,
        "precedence": np.asarray(precedence, dtype=float),
        "pair_index": pair_index,
        "contact_support": counts,
        "pair_support": np.asarray(pair_support, dtype=float),
        "n_supported_contacts": supported_contacts,
        "n_supported_pairs": supported_pairs,
        "n_events": int(n_events),
        "status": status,
    }


def _agree(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if int(keep.sum()) < 3:
        return None
    x, y = a[keep], b[keep]
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    value = spearmanr(x, y).statistic
    return None if not np.isfinite(value) else float(value)


def family_agreement(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, float | None]:
    return {
        "participation": _agree(left["participation_rate"], right["participation_rate"]),
        "mean_rank": _agree(left["masked_mean_rank"], right["masked_mean_rank"]),
        "precedence": _agree(left["precedence"], right["precedence"]),
    }


def resolved_families(agreement: Mapping[str, float | None]) -> int:
    return int(sum(agreement.get(name) is not None for name in FAMILIES))
