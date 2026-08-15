"""Cohort contracts for patient-conditioned data-driven Topic 4 SNNs.

This module contains only target/geometry transformations.  It does not build
fields or run the SNN, which keeps patient information out of candidate-library
generation by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

from src.lagpat_rank_audit import (
    build_masked_kmeans_features,
    mask_phantom_ranks,
)
from src.topic4_core_field_profile import split_by_block


MODE_NAMES = ("TA", "TB")


@dataclass(frozen=True)
class TargetConfig:
    minimum_participating_contacts: int = 3
    heldout_block_fraction: float = 0.3
    split_seed: int = 20260815
    kmeans_fit_max_events: int = 10_000
    kmeans_n_init: int = 20
    kmeans_seed: int = 20260815
    stability_seeds: tuple[int, ...] = (
        20260815, 20260816, 20260817, 20260818, 20260819,
    )
    stored_events_per_mode_per_split: int = 4096


def subject_raw_root(subject_id: str, *, epilepsiae_root: str | Path,
                     yuquan_root: str | Path) -> Path:
    dataset, subject = str(subject_id).split("_", 1)
    if dataset == "epilepsiae":
        legacy = Path(epilepsiae_root) / subject / "all_recs"
        return legacy if legacy.exists() else Path(epilepsiae_root) / subject
    if dataset == "yuquan":
        return Path(yuquan_root) / subject
    raise ValueError(f"unsupported dataset in subject id: {subject_id}")


def canonical_pair_contract(rank_displacement: dict) -> dict:
    """Return the first frozen stable-K=2 pair restricted to joint-valid nodes."""
    if int(rank_displacement.get("stable_k", -1)) != 2:
        raise ValueError("rank-displacement subject is not stable_k=2")
    pairs = rank_displacement.get("pairs") or []
    if not pairs:
        raise ValueError("rank-displacement subject has no template pair")
    pair = pairs[0]
    names = np.asarray([str(value) for value in pair["channel_names"]])
    valid = np.asarray(pair.get("joint_valid", np.ones(len(names))), dtype=bool)
    rank_a = np.asarray(pair["rank_a_dense_full"], dtype=float)
    rank_b = np.asarray(pair["rank_b_dense_full"], dtype=float)
    if not (len(names) == len(valid) == len(rank_a) == len(rank_b)):
        raise ValueError("rank-displacement pair arrays do not align")
    keep = valid & np.isfinite(rank_a) & np.isfinite(rank_b)
    if int(keep.sum()) < 3:
        raise ValueError("fewer than three joint-valid template contacts")
    return {
        "contact_order": names[keep].tolist(),
        "rank_a": rank_a[keep],
        "rank_b": rank_b[keep],
        "cluster_id_a": int(pair.get("cluster_id_a", 0)),
        "cluster_id_b": int(pair.get("cluster_id_b", 1)),
    }


def subset_pair_contract(pair: dict, contact_order: list[str]) -> dict:
    """Restrict a canonical pair to a geometry-supported contact order."""
    lookup = {name: index for index, name in enumerate(pair["contact_order"])}
    missing = [name for name in contact_order if name not in lookup]
    if missing:
        raise ValueError(f"geometry contacts absent from template pair: {missing}")
    indices = np.asarray([lookup[name] for name in contact_order], dtype=int)
    return {
        **pair,
        "contact_order": list(contact_order),
        "rank_a": np.asarray(pair["rank_a"], float)[indices],
        "rank_b": np.asarray(pair["rank_b"], float)[indices],
    }


def geometry_only_sheet_projection(coords: np.ndarray, *, sheet_size_mm: float,
                                   margin_mm: float) -> dict:
    """Deterministic geometry-only 3-D to 2-D PCA and isotropic sheet fit."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 3:
        raise ValueError("coords must have shape (n_contact>=3, 3)")
    if not np.isfinite(coords).all():
        raise ValueError("coords contain non-finite values")
    centered = coords - coords.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape) * np.finfo(float).eps * singular[0]
    matrix_rank = int(np.sum(singular > tolerance))
    if matrix_rank < 2:
        raise ValueError("contact geometry has rank below two")
    basis = vh[:2].copy()
    for row in basis:
        anchor = int(np.argmax(np.abs(row)))
        if row[anchor] < 0.0:
            row *= -1.0
    projected = centered @ basis.T
    lo = projected.min(axis=0)
    span = projected.max(axis=0) - lo
    usable = float(sheet_size_mm) - 2.0 * float(margin_mm)
    if usable <= 0.0 or float(span.max()) <= 0.0:
        raise ValueError("invalid sheet or degenerate projected geometry")
    scale = usable / float(span.max())
    extent = span * scale
    offset = float(margin_mm) + 0.5 * (usable - extent) - lo * scale
    sheet = projected * scale + offset
    if np.any(sheet < -1e-9) or np.any(sheet > float(sheet_size_mm) + 1e-9):
        raise RuntimeError("geometry projection escaped the sheet")
    return {
        "coords_sheet": sheet,
        "coords_projected": projected,
        "basis": basis,
        "center_3d": coords.mean(axis=0),
        "singular_values": singular,
        "matrix_rank": matrix_rank,
        "scale": float(scale),
        "offset": offset,
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    if float(np.std(left[valid])) < 1e-12 or float(np.std(right[valid])) < 1e-12:
        return float("nan")
    return float(spearmanr(left[valid], right[valid]).statistic)


def _normalized_template(rank: np.ndarray) -> np.ndarray:
    rank = np.asarray(rank, float)
    valid = np.isfinite(rank)
    output = np.full(rank.shape, np.nan, dtype=float)
    if int(valid.sum()) == 1:
        output[valid] = 0.5
    elif int(valid.sum()) > 1:
        order = np.argsort(np.argsort(rank[valid], kind="stable"), kind="stable")
        output[valid] = order / float(valid.sum() - 1)
    return output


def _profiles(masked_ranks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rows = []
    for mode in (0, 1):
        selected = masked_ranks[labels == mode]
        if not len(selected):
            rows.append(np.full(masked_ranks.shape[1], np.nan))
        else:
            with np.errstate(invalid="ignore"):
                rows.append(np.nanmean(selected, axis=0))
    return np.asarray(rows, dtype=float)


def _correlation_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray([
        [_spearman(left[i], right[j]) for j in range(len(right))]
        for i in range(len(left))
    ], dtype=float)


def _semantic_mapping(cluster_profiles: np.ndarray, rank_a: np.ndarray,
                      rank_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    canonical = np.asarray([
        _normalized_template(rank_a), _normalized_template(rank_b)
    ])
    matrix = _correlation_matrix(cluster_profiles, canonical)
    if not np.isfinite(matrix).all():
        raise ValueError("cluster-to-template correlation is not evaluable")
    rows, columns = linear_sum_assignment(-matrix)
    mapping = np.full(2, -1, dtype=int)
    mapping[rows] = columns
    if set(mapping.tolist()) != {0, 1}:
        raise RuntimeError("cluster-to-template mapping is not one-to-one")
    return mapping, matrix


def _precedence_states(masked_ranks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per unordered pair: i<j, j<i, or not jointly recruited."""
    masked_ranks = np.asarray(masked_ranks, float)
    pairs = np.asarray(list(combinations(range(masked_ranks.shape[1]), 2)), dtype=int)
    output = np.zeros((len(pairs), 3), dtype=float)
    for index, (left, right) in enumerate(pairs):
        lv = masked_ranks[:, left]
        rv = masked_ranks[:, right]
        joint = np.isfinite(lv) & np.isfinite(rv)
        output[index, 0] = np.mean(joint & (lv < rv))
        output[index, 1] = np.mean(joint & (rv < lv))
        output[index, 2] = 1.0 - np.mean(joint)
    return pairs, output


def _mode_descriptors(masked_ranks: np.ndarray, labels: np.ndarray) -> dict:
    result = {}
    pair_indices = None
    for mode, name in enumerate(MODE_NAMES):
        selected = masked_ranks[labels == mode]
        pairs, precedence = _precedence_states(selected)
        if pair_indices is None:
            pair_indices = pairs
        finite = np.isfinite(selected)
        counts = finite.sum(axis=0)
        profile = np.divide(
            np.nansum(selected, axis=0), counts,
            out=np.full(selected.shape[1], np.nan), where=counts > 0,
        )
        result[name] = {
            "n_events": int(len(selected)),
            "profile": profile,
            "recruitment": np.isfinite(selected).mean(axis=0),
            "precedence": precedence,
        }
    result["pair_indices"] = pair_indices
    return result


def _balanced_sample(masked_ranks: np.ndarray, labels: np.ndarray, *, limit: int,
                     seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    output = {}
    for mode, name in enumerate(MODE_NAMES):
        indices = np.flatnonzero(labels == mode)
        if len(indices) > int(limit):
            indices = np.sort(rng.choice(indices, size=int(limit), replace=False))
        output[name] = masked_ranks[indices]
    return output


def _stability(features: np.ndarray, fit_indices: np.ndarray, *, seeds: tuple[int, ...],
               n_init: int) -> dict:
    labels = [
        KMeans(n_clusters=2, n_init=int(n_init), random_state=int(seed)).fit_predict(
            features[fit_indices]
        )
        for seed in seeds
    ]
    pairwise = [
        adjusted_mutual_info_score(labels[i], labels[j])
        for i in range(len(labels)) for j in range(i + 1, len(labels))
    ]
    return {
        "pairwise_ami_median": float(np.median(pairwise)) if pairwise else 1.0,
        "pairwise_ami_min": float(np.min(pairwise)) if pairwise else 1.0,
        "n_seeds": int(len(seeds)),
    }


def build_crossfit_patient_target(data: dict, pair: dict, *, config: TargetConfig) -> dict:
    """Fit train-only masked KMeans and transform disjoint recording blocks."""
    patient_names = [str(value) for value in data["channel_names"]]
    contact_order = [str(value) for value in pair["contact_order"]]
    lookup = {name: index for index, name in enumerate(patient_names)}
    missing = [name for name in contact_order if name not in lookup]
    if missing:
        raise ValueError(f"raw patient data miss target contacts: {missing}")
    rows = np.asarray([lookup[name] for name in contact_order], dtype=int)
    ranks = np.asarray(data["ranks"], float)[rows]
    bools = np.asarray(data["bools"], bool)[rows]
    block_ids = np.asarray(data["block_ids"])
    if ranks.shape != bools.shape or ranks.shape[1] != len(block_ids):
        raise ValueError("patient ranks, masks, and block ids do not align")

    valid = np.flatnonzero(
        bools.sum(axis=0) >= int(config.minimum_participating_contacts)
    )
    if len(valid) < 4:
        raise ValueError("fewer than four readable patient events")
    train_local, heldout_local = split_by_block(
        block_ids[valid], config.heldout_block_fraction, config.split_seed,
    )
    train = valid[train_local]
    heldout = valid[heldout_local]
    if set(block_ids[train].tolist()) & set(block_ids[heldout].tolist()):
        raise RuntimeError("recording-block leakage in patient target")

    features = build_masked_kmeans_features(ranks, bools, impute="event_median")
    rng = np.random.default_rng(int(config.kmeans_seed))
    fit_indices = train
    if len(fit_indices) > int(config.kmeans_fit_max_events):
        fit_indices = np.sort(rng.choice(
            fit_indices, size=int(config.kmeans_fit_max_events), replace=False,
        ))
    kmeans = KMeans(
        n_clusters=2,
        n_init=int(config.kmeans_n_init),
        random_state=int(config.kmeans_seed),
    ).fit(features[fit_indices])
    train_raw = kmeans.predict(features[train])
    heldout_raw = kmeans.predict(features[heldout])
    masked = mask_phantom_ranks(ranks, bools, normalize=True).T
    raw_profiles = _profiles(masked[train], train_raw)
    mapping, template_matrix = _semantic_mapping(
        raw_profiles, np.asarray(pair["rank_a"]), np.asarray(pair["rank_b"]),
    )
    train_labels = mapping[train_raw]
    heldout_labels = mapping[heldout_raw]
    if np.any(np.bincount(train_labels, minlength=2) == 0):
        raise ValueError("train split does not contain both semantic modes")
    if np.any(np.bincount(heldout_labels, minlength=2) == 0):
        raise ValueError("held-out split does not contain both semantic modes")
    centers = np.empty_like(kmeans.cluster_centers_)
    for raw_cluster, semantic_mode in enumerate(mapping):
        centers[semantic_mode] = kmeans.cluster_centers_[raw_cluster]

    train_profiles = _profiles(masked[train], train_labels)
    heldout_profiles = _profiles(masked[heldout], heldout_labels)
    crossfit_matrix = _correlation_matrix(train_profiles, heldout_profiles)
    diagonal = float(np.mean(np.diag(crossfit_matrix)))
    crossed = float(np.mean(crossfit_matrix[[0, 1], [1, 0]]))

    train_distance = np.linalg.norm(features[train] - centers[train_labels], axis=1)
    heldout_distance = np.linalg.norm(
        features[heldout] - centers[heldout_labels], axis=1,
    )
    ood_threshold = float(np.quantile(train_distance, 0.95))
    stability = _stability(
        features, fit_indices,
        seeds=tuple(config.stability_seeds), n_init=config.kmeans_n_init,
    )
    train_descriptors = _mode_descriptors(masked[train], train_labels)
    heldout_descriptors = _mode_descriptors(masked[heldout], heldout_labels)
    return {
        "contact_order": contact_order,
        "train_event_indices": train,
        "heldout_event_indices": heldout,
        "train_block_ids": np.unique(block_ids[train]),
        "heldout_block_ids": np.unique(block_ids[heldout]),
        "train_labels": train_labels,
        "heldout_labels": heldout_labels,
        "kmeans_centers": centers,
        "cluster_to_semantic_mode": mapping,
        "cluster_to_frozen_template_matrix": template_matrix,
        "train_profiles": train_profiles,
        "heldout_profiles": heldout_profiles,
        "train_to_heldout_matrix": crossfit_matrix,
        "train_to_heldout_diagonal": diagonal,
        "train_to_heldout_crossed": crossed,
        "train_to_heldout_margin": diagonal - crossed,
        "train_mode_counts": np.bincount(train_labels, minlength=2),
        "heldout_mode_counts": np.bincount(heldout_labels, minlength=2),
        "train_distance_q95": ood_threshold,
        "heldout_ood_fraction": float(np.mean(heldout_distance > ood_threshold)),
        "kmeans_stability": stability,
        "train_descriptors": train_descriptors,
        "heldout_descriptors": heldout_descriptors,
        "train_samples": _balanced_sample(
            masked[train], train_labels,
            limit=config.stored_events_per_mode_per_split,
            seed=config.kmeans_seed + 100,
        ),
        "heldout_samples": _balanced_sample(
            masked[heldout], heldout_labels,
            limit=config.stored_events_per_mode_per_split,
            seed=config.kmeans_seed + 200,
        ),
    }
