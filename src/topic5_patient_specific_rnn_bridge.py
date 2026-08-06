"""Patient-specific, target-free interictal RNN bridge helpers.

The functions in this module never load ictal arrays.  They prepare a single
patient's chronological interictal splits and convert generated complete
events into contact-wise fields for a later, isolated target readout.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np

from src.topic5_rank_distribution import contact_rank_distribution


def chronological_60_20_20(record) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Respect the frozen outer 80/20 split and divide train80 into 60/20."""

    train80 = np.asarray(record.train_indices, dtype=np.int64)
    test20 = np.asarray(record.eval_indices, dtype=np.int64)
    if train80.size < 4 or test20.size < 1:
        raise ValueError(f"{record.subject}: insufficient chronological events")
    fit_n = int(np.floor(0.75 * train80.size))
    fit60 = train80[:fit_n]
    validation20 = train80[fit_n:]
    if not (fit60.size and validation20.size):
        raise ValueError(f"{record.subject}: empty fit or validation split")
    if np.intersect1d(fit60, validation20).size:
        raise RuntimeError("fit and validation overlap")
    if np.intersect1d(np.concatenate([fit60, validation20]), test20).size:
        raise RuntimeError("development events overlap frozen test20")
    return fit60, validation20, test20


def train_only_contact_features(
    group_ids: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """Three target-free node features estimated only from fit60 rank events."""

    groups = np.asarray(group_ids, dtype=np.int64)[np.asarray(indices, dtype=np.int64)]
    participation = np.mean(groups >= 0, axis=0)
    counts = np.maximum(np.max(groups, axis=1) + 1, 1)
    normalized = np.where(groups >= 0, groups / np.maximum(counts[:, None] - 1, 1), np.nan)
    finite = np.isfinite(normalized)
    denominator = np.sum(finite, axis=0)
    mean_rank = np.divide(
        np.nansum(normalized, axis=0),
        denominator,
        out=np.full(normalized.shape[1], 0.5, dtype=float),
        where=denominator > 0,
    )
    return np.column_stack(
        [participation, mean_rank, np.ones_like(participation)]
    ).astype(np.float32)


def record_with_split(record, fit: np.ndarray, evaluate: np.ndarray, features: np.ndarray):
    """Return a dataclass copy whose train/eval properties expose one split."""

    split = np.full(len(record.event_split), 2, dtype=np.uint8)
    split[np.asarray(fit, dtype=np.int64)] = 0
    split[np.asarray(evaluate, dtype=np.int64)] = 1
    return replace(
        record,
        contact_features=np.asarray(features, dtype=np.float32),
        event_split=split,
    )


def distribution_fields(
    group_ids: np.ndarray,
    group_count: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, np.ndarray]:
    """Five contact fields derived from one complete-event distribution."""

    summary = contact_rank_distribution(group_ids, group_count, bins=bins)
    participation = np.asarray(summary["participation_probability"], dtype=float)
    histogram = np.asarray(summary["rank_histogram"], dtype=float)
    mean_rank = np.asarray(summary["mean_rank"], dtype=float)
    early = participation * np.sum(histogram[:, :3], axis=1)
    late = participation * np.sum(histogram[:, -3:], axis=1)
    weighted = participation * (1.0 - np.where(np.isfinite(mean_rank), mean_rank, 0.5))
    return {
        "participation": participation,
        "early_joint_mass": early,
        "late_joint_mass": late,
        "endpoint_joint_mass": early + late,
        "weighted_earliness": weighted,
    }


def field_matrix(
    fields: Mapping[str, np.ndarray], order: Sequence[str]
) -> np.ndarray:
    matrix = np.column_stack([np.asarray(fields[name], dtype=float) for name in order])
    if matrix.ndim != 2 or matrix.shape[0] < 3:
        raise ValueError("contact field matrix is too small")
    return matrix


def shaft_labels(names: Sequence[str]) -> np.ndarray:
    """Canonical electrode-shaft labels used only by the sensitivity null."""

    from src.propagation_skeleton_geometry import parse_shaft

    labels = []
    for name in map(str, names):
        shaft, _ = parse_shaft(name)
        labels.append(str(shaft) if shaft is not None else name)
    return np.asarray(labels)


def permutation_indices(
    names: Sequence[str], *, n_draws: int, seed: int, within_shaft: bool
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n = len(names)
    if not within_shaft:
        return np.stack([rng.permutation(n) for _ in range(int(n_draws))])
    labels = shaft_labels(names)
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    draws = []
    for _ in range(int(n_draws)):
        index = np.arange(n)
        for group in groups:
            index[group] = rng.permutation(group)
        draws.append(index)
    return np.stack(draws)
