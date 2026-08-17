"""Dense-anchor moving-block bootstrap helpers for Topic 5 V3.0."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from src.topic5_event_innovation_v3_0 import RankStateBasis, precedence_probability


def observable_gain_sufficient_statistics(
    basis: RankStateBasis,
    observed_fields: np.ndarray,
    supports: np.ndarray,
    future_windows: list[np.ndarray],
    ranks: np.ndarray,
    participation: np.ndarray,
    tie_groups: np.ndarray,
    autonomous_state: np.ndarray,
    event_state: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return exact row-level numerators and denominators for both scores."""

    observed = np.asarray(observed_fields, dtype=float)
    weight = np.asarray(supports, dtype=float)
    rank = np.asarray(ranks, dtype=float)
    mask = np.asarray(participation, dtype=bool)
    ties = np.asarray(tie_groups)
    automatic = basis.inverse(np.asarray(autonomous_state, dtype=float))
    driven = basis.inverse(np.asarray(event_state, dtype=float))
    n_rows = len(observed)
    if any(len(value) != n_rows for value in (weight, automatic, driven, future_windows)):
        raise ValueError("observable arrays are not row aligned")
    if rank.shape != mask.shape or rank.shape != ties.shape:
        raise ValueError("event arrays are not aligned")

    rank_numerator = np.zeros(n_rows, dtype=float)
    rank_denominator = np.zeros(n_rows, dtype=float)
    pair_numerator = np.zeros(n_rows, dtype=float)
    pair_denominator = np.zeros(n_rows, dtype=float)
    upper = np.triu(np.ones((rank.shape[1], rank.shape[1]), dtype=bool), k=1)
    for row in range(n_rows):
        valid = (
            np.isfinite(observed[row])
            & np.isfinite(automatic[row])
            & np.isfinite(driven[row])
            & (weight[row] > 0)
        )
        if np.any(valid):
            auto_error = (automatic[row, valid] - observed[row, valid]) ** 2
            driven_error = (driven[row, valid] - observed[row, valid]) ** 2
            rank_numerator[row] = np.sum(
                weight[row, valid] * (auto_error - driven_error)
            )
            rank_denominator[row] = np.sum(weight[row, valid])

        auto_probability = precedence_probability(automatic[row])
        driven_probability = precedence_probability(driven[row])
        for event in np.asarray(future_windows[row], dtype=np.int64):
            valid_contacts = mask[event] & np.isfinite(rank[event])
            pair_valid = (
                upper
                & valid_contacts[:, None]
                & valid_contacts[None, :]
                & (ties[event, :, None] != ties[event, None, :])
            )
            if not np.any(pair_valid):
                continue
            outcome = rank[event, :, None] < rank[event, None, :]
            auto_error = (outcome.astype(float) - auto_probability) ** 2
            driven_error = (outcome.astype(float) - driven_probability) ** 2
            pair_numerator[row] += np.sum(
                (auto_error - driven_error)[pair_valid]
            )
            pair_denominator[row] += np.sum(pair_valid)
    return {
        "rank_numerator": rank_numerator,
        "rank_denominator": rank_denominator,
        "pair_numerator": pair_numerator,
        "pair_denominator": pair_denominator,
    }


def standardized_propagation_gain(
    statistics: dict[str, np.ndarray],
    rows: np.ndarray,
    *,
    rank_scale: float,
    pair_scale: float,
) -> float:
    selected = np.asarray(rows, dtype=np.int64)
    rank_denominator = float(np.sum(statistics["rank_denominator"][selected]))
    pair_denominator = float(np.sum(statistics["pair_denominator"][selected]))
    if rank_denominator <= 0 or pair_denominator <= 0:
        return float("nan")
    rank_gain = float(np.sum(statistics["rank_numerator"][selected])) / rank_denominator
    pair_gain = float(np.sum(statistics["pair_numerator"][selected])) / pair_denominator
    return 0.5 * (
        rank_gain / max(float(rank_scale), 1e-12)
        + pair_gain / max(float(pair_scale), 1e-12)
    )


def moving_block_resamples(
    group: np.ndarray,
    event_index: np.ndarray,
    *,
    block_length: int,
    draws: int,
    seed: int,
) -> Iterable[np.ndarray]:
    """Sample overlapping row blocks within each continuity unit."""

    groups = np.asarray(group)
    events = np.asarray(event_index, dtype=np.int64)
    length = int(block_length)
    if groups.shape != events.shape or length < 1 or int(draws) < 1:
        raise ValueError("invalid moving-block bootstrap inputs")
    ordered_groups = []
    for value in np.unique(groups):
        rows = np.flatnonzero(groups == value)
        rows = rows[np.argsort(events[rows], kind="stable")]
        if not len(rows):
            continue
        boundaries = np.flatnonzero(np.diff(events[rows]) != 1) + 1
        ordered_groups.extend(
            segment for segment in np.split(rows, boundaries) if len(segment)
        )
    rng = np.random.default_rng(int(seed))
    for _ in range(int(draws)):
        selected = []
        for rows in ordered_groups:
            local_length = min(length, len(rows))
            n_blocks = int(np.ceil(len(rows) / local_length))
            maximum_start = len(rows) - local_length
            starts = rng.integers(0, maximum_start + 1, size=n_blocks)
            sampled = np.concatenate([
                rows[start : start + local_length] for start in starts
            ])[: len(rows)]
            selected.append(sampled)
        yield np.concatenate(selected).astype(np.int64)


__all__ = [
    "moving_block_resamples",
    "observable_gain_sufficient_statistics",
    "standardized_propagation_gain",
]
