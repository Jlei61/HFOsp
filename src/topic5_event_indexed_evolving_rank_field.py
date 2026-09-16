"""Non-parametric observability tools for event-indexed evolving rank fields.

This module deliberately contains no recurrent neural network.  Its job is to
decide whether a time-varying low-dimensional object is observable before a
state-space model is authorized.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr


EPS = 1e-12


@dataclass(frozen=True)
class EventBlock:
    """One full equal-event-count window inside one source recording."""

    indices: np.ndarray
    source_block_id: int
    within_source_order: int


@dataclass(frozen=True)
class BlockDescriptor:
    """Directly observed propagation summaries for one event block."""

    rank_field: np.ndarray
    participation: np.ndarray
    precedence: np.ndarray
    precedence_support: np.ndarray
    source_probability: np.ndarray
    sink_probability: np.ndarray

    @property
    def field(self) -> np.ndarray:
        return np.concatenate([self.rank_field, self.participation])


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Return a finite Spearman correlation when at least three values vary."""
    a = np.asarray(left, float).ravel()
    b = np.asarray(right, float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 3:
        return float("nan")
    if np.std(a[valid]) <= EPS or np.std(b[valid]) <= EPS:
        return float("nan")
    return float(spearmanr(a[valid], b[valid]).statistic)


def make_equal_event_blocks(
    ordered_indices: Sequence[int],
    source_block_ids: np.ndarray,
    block_size: int,
) -> list[EventBlock]:
    """Split events into full blocks without crossing a source recording."""
    indices = np.asarray(ordered_indices, dtype=int)
    source = np.asarray(source_block_ids, dtype=int)
    size = int(block_size)
    if indices.ndim != 1 or size < 2:
        raise ValueError("ordered_indices must be 1D and block_size >= 2")
    if np.any(indices < 0) or np.any(indices >= len(source)):
        raise ValueError("ordered_indices exceed source_block_ids")
    blocks: list[EventBlock] = []
    seen: list[int] = []
    for value in source[indices]:
        item = int(value)
        if item not in seen:
            seen.append(item)
    for source_id in seen:
        selected = indices[source[indices] == source_id]
        n_full = len(selected) // size
        for order in range(n_full):
            chunk = selected[order * size : (order + 1) * size]
            blocks.append(
                EventBlock(
                    indices=np.asarray(chunk, dtype=int),
                    source_block_id=source_id,
                    within_source_order=order,
                )
            )
    return blocks


def global_rank_prior(
    local_rank: np.ndarray,
    participation: np.ndarray,
    indices: Sequence[int],
) -> np.ndarray:
    """Estimate the train-only contact rank prior, falling back to 0.5."""
    rank = np.asarray(local_rank, float)[np.asarray(indices, dtype=int)]
    mask = np.asarray(participation, bool)[np.asarray(indices, dtype=int)]
    valid = mask & np.isfinite(rank)
    count = np.sum(valid, axis=0)
    total = np.sum(np.where(valid, rank, 0.0), axis=0)
    return np.divide(
        total,
        count,
        out=np.full(rank.shape[1], 0.5, dtype=float),
        where=count > 0,
    )


def estimate_block_descriptor(
    local_rank: np.ndarray,
    participation: np.ndarray,
    group_ids: np.ndarray,
    indices: Sequence[int],
    *,
    rank_prior: np.ndarray,
    shrinkage_prior_events: float,
    beta_prior: float,
) -> BlockDescriptor:
    """Estimate a masked rank field and precedence distribution for a block."""
    selected = np.asarray(indices, dtype=int)
    rank = np.asarray(local_rank, float)[selected]
    part = np.asarray(participation, bool)[selected]
    groups = np.asarray(group_ids, int)[selected]
    if rank.ndim != 2 or part.shape != rank.shape or groups.shape != rank.shape:
        raise ValueError("rank, participation, and group_ids must align")
    prior_rank = np.asarray(rank_prior, float)
    if prior_rank.shape != (rank.shape[1],):
        raise ValueError("rank_prior does not match contacts")
    alpha = float(shrinkage_prior_events)
    beta = float(beta_prior)
    if alpha < 0 or beta <= 0:
        raise ValueError("invalid shrinkage prior")
    valid = part & np.isfinite(rank)
    count = np.sum(valid, axis=0, dtype=float)
    rank_sum = np.sum(np.where(valid, rank, 0.0), axis=0)
    rank_field = (rank_sum + alpha * prior_rank) / (count + alpha)
    participation_probability = (np.sum(part, axis=0) + beta) / (
        len(selected) + 2.0 * beta
    )
    source_probability = (np.sum(groups == 0, axis=0) + beta) / (
        len(selected) + 2.0 * beta
    )
    event_maximum = np.max(groups, axis=1, initial=-1)
    sink = (groups >= 0) & (groups == event_maximum[:, None])
    sink_probability = (np.sum(sink, axis=0) + beta) / (
        len(selected) + 2.0 * beta
    )
    left_index, right_index = np.triu_indices(rank.shape[1], 1)
    left_group = groups[:, left_index]
    right_group = groups[:, right_index]
    valid_pair = (left_group >= 0) & (right_group >= 0)
    delta = left_group - right_group
    support = np.sum(valid_pair, axis=0, dtype=int)
    score = np.sum(valid_pair & (delta < 0), axis=0, dtype=float)
    score += 0.5 * np.sum(valid_pair & (delta == 0), axis=0, dtype=float)
    precedence = (score + beta) / (support + 2.0 * beta)
    return BlockDescriptor(
        rank_field=np.asarray(rank_field, float),
        participation=np.asarray(participation_probability, float),
        precedence=np.asarray(precedence, float),
        precedence_support=np.asarray(support, int),
        source_probability=np.asarray(source_probability, float),
        sink_probability=np.asarray(sink_probability, float),
    )


def descriptor_distance(
    left: BlockDescriptor,
    right: BlockDescriptor,
    *,
    contact_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Return scale-matched RMSE distances between observable summaries."""
    if contact_mask is None:
        mask = np.ones(len(left.rank_field), dtype=bool)
    else:
        mask = np.asarray(contact_mask, bool)
    if mask.shape != left.rank_field.shape or not np.any(mask):
        return {"rank": float("nan"), "participation": float("nan"), "field": float("nan"), "precedence": float("nan")}
    rank = float(np.sqrt(np.mean((left.rank_field[mask] - right.rank_field[mask]) ** 2)))
    participation = float(
        np.sqrt(np.mean((left.participation[mask] - right.participation[mask]) ** 2))
    )
    field = float(np.sqrt((rank * rank + participation * participation) / 2.0))
    pair_mask = np.asarray([], dtype=bool)
    if len(left.precedence):
        left_index, right_index = np.triu_indices(len(mask), 1)
        pair_mask = mask[left_index] & mask[right_index]
    precedence = (
        float(np.sqrt(np.mean((left.precedence[pair_mask] - right.precedence[pair_mask]) ** 2)))
        if np.any(pair_mask)
        else float("nan")
    )
    return {
        "rank": rank,
        "participation": participation,
        "field": field,
        "precedence": precedence,
    }


def block_reliability(
    local_rank: np.ndarray,
    participation: np.ndarray,
    group_ids: np.ndarray,
    blocks: Sequence[EventBlock],
    *,
    rank_prior: np.ndarray,
    shrinkage_prior_events: float,
    beta_prior: float,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    """Measure estimator repeatability without using between-block signal."""
    rng = np.random.default_rng(int(seed))
    rank_values: list[float] = []
    participation_values: list[float] = []
    precedence_values: list[float] = []
    for block in blocks:
        if len(block.indices) < 4:
            continue
        half = len(block.indices) // 2
        for _ in range(int(repeats)):
            order = rng.permutation(block.indices)
            left = estimate_block_descriptor(
                local_rank,
                participation,
                group_ids,
                order[:half],
                rank_prior=rank_prior,
                shrinkage_prior_events=shrinkage_prior_events,
                beta_prior=beta_prior,
            )
            right = estimate_block_descriptor(
                local_rank,
                participation,
                group_ids,
                order[half : 2 * half],
                rank_prior=rank_prior,
                shrinkage_prior_events=shrinkage_prior_events,
                beta_prior=beta_prior,
            )
            rank_values.append(safe_spearman(left.rank_field, right.rank_field))
            participation_values.append(
                safe_spearman(left.participation, right.participation)
            )
            precedence_values.append(safe_spearman(left.precedence, right.precedence))
    def median(values: Iterable[float]) -> float:
        array = np.asarray(list(values), float)
        return float(np.nanmedian(array)) if np.any(np.isfinite(array)) else float("nan")
    return {
        "rank_spearman_median": median(rank_values),
        "participation_spearman_median": median(participation_values),
        "precedence_spearman_median": median(precedence_values),
        "n_split_comparisons": len(rank_values),
    }


def stratified_uniform_subsample_blocks(
    blocks: Sequence[EventBlock],
    maximum: int,
) -> list[EventBlock]:
    """Deterministically retain source-stratified blocks across chronology."""
    limit = int(maximum)
    if limit < 1:
        raise ValueError("maximum must be positive")
    if len(blocks) <= limit:
        return list(blocks)
    by_source: dict[int, list[EventBlock]] = {}
    for block in blocks:
        by_source.setdefault(int(block.source_block_id), []).append(block)
    source_ids = sorted(by_source)
    counts = np.asarray([len(by_source[source]) for source in source_ids], float)
    quota = np.floor(limit * counts / np.sum(counts)).astype(int)
    quota[(quota == 0) & (counts > 0)] = 1
    while np.sum(quota) > limit:
        candidates = np.flatnonzero(quota > 1)
        if not len(candidates):
            break
        index = int(candidates[np.argmax(quota[candidates])])
        quota[index] -= 1
    while np.sum(quota) < limit:
        remaining = counts - quota
        index = int(np.argmax(remaining))
        if remaining[index] <= 0:
            break
        quota[index] += 1
    selected: list[EventBlock] = []
    for source, n_keep in zip(source_ids, quota):
        members = sorted(
            by_source[source], key=lambda block: block.within_source_order
        )
        positions = np.linspace(0, len(members) - 1, int(n_keep)).round().astype(int)
        selected.extend(members[int(position)] for position in np.unique(positions))
    return sorted(selected, key=lambda block: int(block.indices[0]))


def stratified_uniform_subsample_blocks_with_adjacency(
    blocks: Sequence[EventBlock],
    maximum: int,
) -> list[EventBlock]:
    """Subsample across chronology while retaining true adjacent block pairs."""
    limit = int(maximum)
    if limit < 2:
        raise ValueError("maximum must be at least two")
    if len(blocks) <= limit:
        return list(blocks)
    by_source: dict[int, list[EventBlock]] = {}
    for block in blocks:
        by_source.setdefault(int(block.source_block_id), []).append(block)
    source_ids = sorted(by_source)
    counts = np.asarray([len(by_source[source]) for source in source_ids], float)
    quota = np.floor(limit * counts / np.sum(counts)).astype(int)
    quota[(quota == 0) & (counts > 0)] = 1
    while np.sum(quota) > limit:
        candidates = np.flatnonzero(quota > 1)
        if not len(candidates):
            break
        index = int(candidates[np.argmax(quota[candidates])])
        quota[index] -= 1
    while np.sum(quota) < limit:
        remaining = counts - quota
        index = int(np.argmax(remaining))
        if remaining[index] <= 0:
            break
        quota[index] += 1
    selected: list[EventBlock] = []
    for source, n_keep in zip(source_ids, quota):
        members = sorted(
            by_source[source], key=lambda block: block.within_source_order
        )
        keep = min(int(n_keep), len(members))
        positions: set[int] = set()
        n_pairs = keep // 2
        if n_pairs and len(members) >= 2:
            starts = np.linspace(0, len(members) - 2, n_pairs).round().astype(int)
            for start in starts:
                positions.add(int(start))
                positions.add(int(start) + 1)
        if len(positions) < keep:
            fillers = np.linspace(0, len(members) - 1, len(members)).round().astype(int)
            for position in fillers:
                positions.add(int(position))
                if len(positions) == keep:
                    break
        selected.extend(members[position] for position in sorted(positions)[:keep])
    return sorted(selected, key=lambda block: int(block.indices[0]))


def within_source_pairs(
    blocks: Sequence[EventBlock],
    *,
    adjacent_only: bool = False,
) -> list[tuple[int, int, int]]:
    """List block pairs as (left index, right index, within-record lag)."""
    pairs: list[tuple[int, int, int]] = []
    by_source: dict[int, list[tuple[int, EventBlock]]] = {}
    for index, block in enumerate(blocks):
        by_source.setdefault(int(block.source_block_id), []).append((index, block))
    for members in by_source.values():
        members = sorted(members, key=lambda item: item[1].within_source_order)
        for left_position in range(len(members)):
            start = left_position + 1
            stop = min(start + 1, len(members)) if adjacent_only else len(members)
            for right_position in range(start, stop):
                left_index, left = members[left_position]
                right_index, right = members[right_position]
                lag = int(right.within_source_order - left.within_source_order)
                if adjacent_only and lag != 1:
                    continue
                pairs.append(
                    (
                        left_index,
                        right_index,
                        lag,
                    )
                )
    return pairs


def _sample_pairs(
    pairs: Sequence[tuple[int, int, int]],
    maximum: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    if len(pairs) <= int(maximum):
        return list(pairs)
    selected = np.sort(rng.choice(len(pairs), size=int(maximum), replace=False))
    return [pairs[int(index)] for index in selected]


def _one_distance_draw(
    local_rank: np.ndarray,
    participation: np.ndarray,
    group_ids: np.ndarray,
    blocks: Sequence[EventBlock],
    *,
    rank_prior: np.ndarray,
    shrinkage_prior_events: float,
    beta_prior: float,
    max_between_pairs: int,
    rng: np.random.Generator,
    contact_mask: np.ndarray | None = None,
) -> dict[str, float]:
    left_descriptors: list[BlockDescriptor] = []
    right_descriptors: list[BlockDescriptor] = []
    within_field = []
    within_precedence = []
    for block in blocks:
        order = rng.permutation(block.indices)
        half = len(order) // 2
        left = estimate_block_descriptor(
            local_rank,
            participation,
            group_ids,
            order[:half],
            rank_prior=rank_prior,
            shrinkage_prior_events=shrinkage_prior_events,
            beta_prior=beta_prior,
        )
        right = estimate_block_descriptor(
            local_rank,
            participation,
            group_ids,
            order[half : 2 * half],
            rank_prior=rank_prior,
            shrinkage_prior_events=shrinkage_prior_events,
            beta_prior=beta_prior,
        )
        left_descriptors.append(left)
        right_descriptors.append(right)
        distance = descriptor_distance(left, right, contact_mask=contact_mask)
        within_field.append(distance["field"])
        within_precedence.append(distance["precedence"])
    pairs = _sample_pairs(
        within_source_pairs(blocks), int(max_between_pairs), rng
    )
    between_field = []
    between_precedence = []
    lags = []
    for left_index, right_index, lag in pairs:
        distance = descriptor_distance(
            left_descriptors[left_index],
            left_descriptors[right_index],
            contact_mask=contact_mask,
        )
        between_field.append(distance["field"])
        between_precedence.append(distance["precedence"])
        lags.append(lag)
    within_field_median = float(np.nanmedian(within_field))
    within_precedence_median = float(np.nanmedian(within_precedence))
    between_field_median = float(np.nanmedian(between_field))
    between_precedence_median = float(np.nanmedian(between_precedence))
    adjacent_pairs = within_source_pairs(blocks, adjacent_only=True)
    adjacent_distance = [
        descriptor_distance(
            left_descriptors[left], left_descriptors[right], contact_mask=contact_mask
        )["field"]
        for left, right, _ in adjacent_pairs
    ]
    nonadjacent_distance = [
        value
        for value, lag in zip(between_field, lags)
        if lag >= 3 and np.isfinite(value)
    ]
    return {
        "field_ratio": between_field_median / max(within_field_median, EPS),
        "precedence_ratio": between_precedence_median
        / max(within_precedence_median, EPS),
        "field_distance_lag_spearman": safe_spearman(
            np.asarray(lags, float), np.asarray(between_field, float)
        ),
        "neighbor_gain": (
            float(np.nanmedian(nonadjacent_distance) - np.nanmedian(adjacent_distance))
            if nonadjacent_distance and adjacent_distance
            else float("nan")
        ),
        "within_field_distance": within_field_median,
        "between_field_distance": between_field_median,
        "within_precedence_distance": within_precedence_median,
        "between_precedence_distance": between_precedence_median,
    }


def permute_events_within_source(
    ordered_indices: Sequence[int],
    source_block_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Destroy chronology while preserving every complete event and record."""
    indices = np.asarray(ordered_indices, dtype=int)
    source = np.asarray(source_block_ids, dtype=int)
    output = indices.copy()
    for value in np.unique(source[indices]):
        positions = np.flatnonzero(source[indices] == value)
        output[positions] = rng.permutation(indices[positions])
    return output


def permute_block_events_within_source(
    blocks: Sequence[EventBlock],
    rng: np.random.Generator,
) -> list[EventBlock]:
    """Shuffle complete events while preserving sampled block positions/lags."""
    output: list[EventBlock | None] = [None] * len(blocks)
    by_source: dict[int, list[int]] = {}
    for block_index, block in enumerate(blocks):
        by_source.setdefault(int(block.source_block_id), []).append(block_index)
    for member_indices in by_source.values():
        pooled = np.concatenate([blocks[index].indices for index in member_indices])
        shuffled = rng.permutation(pooled)
        cursor = 0
        for block_index in member_indices:
            block = blocks[block_index]
            stop = cursor + len(block.indices)
            output[block_index] = EventBlock(
                indices=np.asarray(shuffled[cursor:stop], dtype=int),
                source_block_id=int(block.source_block_id),
                within_source_order=int(block.within_source_order),
            )
            cursor = stop
    if any(block is None for block in output):
        raise RuntimeError("failed to preserve all sampled blocks during permutation")
    return [block for block in output if block is not None]


def dynamic_observability_audit(
    local_rank: np.ndarray,
    participation: np.ndarray,
    group_ids: np.ndarray,
    confirmation_indices: Sequence[int],
    source_block_ids: np.ndarray,
    block_size: int,
    *,
    rank_prior: np.ndarray,
    shrinkage_prior_events: float,
    beta_prior: float,
    split_repeats: int,
    permutation_draws: int,
    max_between_pairs: int,
    distance_ratio_threshold: float,
    p_threshold: float,
    seed: int,
    contact_mask: np.ndarray | None = None,
    prebuilt_blocks: Sequence[EventBlock] | None = None,
    min_adjacent_pairs: int = 0,
) -> dict[str, object]:
    """Compare between-block variation with matched within-block noise/null."""
    indices = np.asarray(confirmation_indices, dtype=int)
    rng = np.random.default_rng(int(seed))
    blocks = (
        make_equal_event_blocks(indices, source_block_ids, int(block_size))
        if prebuilt_blocks is None
        else list(prebuilt_blocks)
    )
    if len(blocks) < 2:
        raise ValueError("dynamic audit requires at least two blocks")
    if prebuilt_blocks is not None:
        block_indices = np.concatenate([block.indices for block in blocks])
        if len(block_indices) != len(indices) or set(map(int, block_indices)) != set(map(int, indices)):
            raise ValueError("prebuilt_blocks do not match confirmation_indices")
        if any(len(block.indices) != int(block_size) for block in blocks):
            raise ValueError("prebuilt block size does not match block_size")
    observed_draws = [
        _one_distance_draw(
            local_rank,
            participation,
            group_ids,
            blocks,
            rank_prior=rank_prior,
            shrinkage_prior_events=shrinkage_prior_events,
            beta_prior=beta_prior,
            max_between_pairs=max_between_pairs,
            rng=rng,
            contact_mask=contact_mask,
        )
        for _ in range(int(split_repeats))
    ]
    keys = observed_draws[0]
    observed = {
        key: float(np.nanmedian([draw[key] for draw in observed_draws]))
        for key in keys
    }
    null = {key: [] for key in ("field_ratio", "precedence_ratio", "field_distance_lag_spearman", "neighbor_gain")}
    for _ in range(int(permutation_draws)):
        shuffled_blocks = (
            make_equal_event_blocks(
                permute_events_within_source(indices, source_block_ids, rng),
                source_block_ids,
                int(block_size),
            )
            if prebuilt_blocks is None
            else permute_block_events_within_source(blocks, rng)
        )
        draw = _one_distance_draw(
            local_rank,
            participation,
            group_ids,
            shuffled_blocks,
            rank_prior=rank_prior,
            shrinkage_prior_events=shrinkage_prior_events,
            beta_prior=beta_prior,
            max_between_pairs=max_between_pairs,
            rng=rng,
            contact_mask=contact_mask,
        )
        for key in null:
            null[key].append(float(draw[key]))

    def upper_p(key: str) -> float:
        values = np.asarray(null[key], float)
        values = values[np.isfinite(values)]
        if not len(values) or not np.isfinite(observed[key]):
            return float("nan")
        return float((1 + np.sum(values >= observed[key])) / (1 + len(values)))

    p_values = {key: upper_p(key) for key in null}
    field_pass = bool(
        observed["field_ratio"] >= float(distance_ratio_threshold)
        and p_values["field_ratio"] <= float(p_threshold)
    )
    precedence_supportive = bool(
        observed["precedence_ratio"] > 1.0
    )
    temporal_structure = bool(
        (
            observed["field_distance_lag_spearman"] > 0
            and p_values["field_distance_lag_spearman"] <= float(p_threshold)
        )
        or (
            observed["neighbor_gain"] > 0
            and p_values["neighbor_gain"] <= float(p_threshold)
        )
    )
    n_adjacent_pairs = len(within_source_pairs(blocks, adjacent_only=True))
    adjacency_eligible = bool(n_adjacent_pairs >= int(min_adjacent_pairs))
    return {
        "n_blocks": len(blocks),
        "n_adjacent_pairs": n_adjacent_pairs,
        "observed": observed,
        "permutation_p": p_values,
        "null_median": {
            key: float(np.nanmedian(values)) for key, values in null.items()
        },
        "field_variation_pass": field_pass,
        "precedence_supportive": precedence_supportive,
        "temporal_structure_supportive": temporal_structure,
        "adjacency_eligible": adjacency_eligible,
        "g0_pass": bool(field_pass and precedence_supportive and adjacency_eligible),
    }


def block_matrix(
    local_rank: np.ndarray,
    participation: np.ndarray,
    group_ids: np.ndarray,
    blocks: Sequence[EventBlock],
    *,
    rank_prior: np.ndarray,
    shrinkage_prior_events: float,
    beta_prior: float,
    contact_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return field rows and source-record IDs for low-rank diagnostics."""
    rows = []
    sources = []
    for block in blocks:
        descriptor = estimate_block_descriptor(
            local_rank,
            participation,
            group_ids,
            block.indices,
            rank_prior=rank_prior,
            shrinkage_prior_events=shrinkage_prior_events,
            beta_prior=beta_prior,
        )
        if contact_mask is None:
            rows.append(descriptor.field)
        else:
            mask = np.asarray(contact_mask, bool)
            rows.append(
                np.concatenate(
                    [descriptor.rank_field[mask], descriptor.participation[mask]]
                )
            )
        sources.append(block.source_block_id)
    return np.asarray(rows, float), np.asarray(sources, int)


def center_within_source(matrix: np.ndarray, source_ids: np.ndarray) -> np.ndarray:
    """Remove source-record intercepts while retaining within-record dynamics."""
    values = np.asarray(matrix, float)
    source = np.asarray(source_ids, int)
    output = values.copy()
    for source_id in np.unique(source):
        mask = source == source_id
        output[mask] -= np.mean(output[mask], axis=0, keepdims=True)
    return output


def _fit_pca_basis(matrix: np.ndarray, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, float)
    mean = np.mean(values, axis=0)
    centered = values - mean
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    k = min(int(dimension), len(right))
    return mean, right[:k].T


def pca_reconstruction_gain(
    train: np.ndarray,
    test: np.ndarray,
    dimension: int,
) -> float:
    """Held-out gain relative to a fixed train mean (K=0)."""
    train_values = np.asarray(train, float)
    test_values = np.asarray(test, float)
    mean, basis = _fit_pca_basis(train_values, int(dimension))
    centered = test_values - mean
    baseline = float(np.mean(centered**2))
    reconstruction = centered @ basis @ basis.T
    error = float(np.mean((centered - reconstruction) ** 2))
    return float(1.0 - error / max(baseline, EPS))


def subspace_similarity(left: np.ndarray, right: np.ndarray, dimension: int) -> float:
    """Mean cosine of principal angles between two fitted PCA subspaces."""
    _, basis_left = _fit_pca_basis(np.asarray(left, float), int(dimension))
    _, basis_right = _fit_pca_basis(np.asarray(right, float), int(dimension))
    singular = np.linalg.svd(basis_left.T @ basis_right, compute_uv=False)
    return float(np.mean(np.clip(singular, 0.0, 1.0)))
