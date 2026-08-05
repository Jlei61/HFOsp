"""The three primary repertoire descriptors of one event window.

Ties are read from `event_group_ids`, never from equal `event_local_rank` values: the
stored rank is a normalised rank among participating contacts, so equal values do not
recover the recruitment-group structure.  Contacts sharing a group are simultaneous and
contribute 0.5 to precedence.

Each family is returned separately and there is no combined score, so no caller can let
one surviving correlation coefficient stand in for the whole repertoire.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

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
            if support < int(min_pair_count) or support == 0:
                precedence.append(np.nan)
                continue
            left, right = groups[both, i], groups[both, j]
            earlier = float(np.sum(left < right))
            tied = float(np.sum(left == right))
            precedence.append((earlier + 0.5 * tied) / support)

    supported_contacts = int(np.sum(counts >= int(min_participation_count)))
    supported_pairs = int(np.sum(np.asarray(pair_support) >= int(min_pair_count)))
    # 3 is the minimum set size on which a rank correlation is meaningful (matching _agree requirement)
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


def _column_mean(stack: np.ndarray) -> np.ndarray:
    """Mean down axis 0 over the finite entries only; all-nan columns stay nan.

    Same idiom as `local_repertoire`'s masked mean, so a descriptor a window could not
    estimate (nan) contributes nothing instead of poisoning the whole column.
    """
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    summed = np.where(finite, stack, 0.0).sum(axis=0)
    return np.where(counts > 0, summed / np.maximum(counts, 1.0), np.nan)


def estimate_backbone(train_repertoires: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The patient's global per-contact and per-pair main effects (§6.6).

    `raw descriptor = stable backbone + slow-state deviation`. This estimates the first
    term — the per-contact participation and mean-rank main effects and the per-pair
    precedence main effect — as the mean of each descriptor over the given windows.

    **The caller passes TRAIN windows and nothing else.** This function cannot check that,
    and a caller that hands it every window fits the backbone on the same data the scale
    curve is then evaluated against, which is exactly the leak §6.6 exists to prevent.

    Every window counts once. At one equal-event grid scale all windows hold the same
    number of events, so an unweighted mean over windows is also the pooled estimate;
    for a clock grid whose windows hold different event counts it is not, and it is still
    the main effect this contract removes.

    An entry no train window could estimate (a contact under the participation floor in
    every train window, a pair under the co-participation floor in every train window) is
    `nan` here. `_residualise_descriptors` in `topic5_slow_state_scale` reads that as
    "there is no backbone to remove" rather than propagating the nan into a window that
    did estimate that descriptor.

    `pair_index` is returned so a consumer can check that a window's pair layout is the
    one the pair main effects were estimated on; the layout is `i < j` in contact order,
    identical for every window with the same contact count.
    """
    windows = list(train_repertoires)
    if not windows:
        raise ValueError("estimate_backbone needs at least one train window repertoire")

    pair_index = [tuple(pair) for pair in windows[0]["pair_index"]]
    n_contacts = len(np.asarray(windows[0]["participation_rate"]))
    for position, window in enumerate(windows[1:], start=1):
        if len(np.asarray(window["participation_rate"])) != n_contacts:
            raise ValueError(
                f"train window {position} has a different contact count than window 0 — "
                "the backbone would average unrelated contacts by array position"
            )
        if [tuple(pair) for pair in window["pair_index"]] != pair_index:
            raise ValueError(
                f"train window {position} has a different pair layout than window 0 — "
                "the backbone would average unrelated pairs by array position"
            )

    def _stack(name: str) -> np.ndarray:
        return np.asarray([np.asarray(window[name], dtype=float) for window in windows])

    return {
        "participation_rate": _column_mean(_stack("participation_rate")),
        "masked_mean_rank": _column_mean(_stack("masked_mean_rank")),
        "precedence": _column_mean(_stack("precedence")),
        "pair_index": pair_index,
        "n_train_windows": len(windows),
    }
