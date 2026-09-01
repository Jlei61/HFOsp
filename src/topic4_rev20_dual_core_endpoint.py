"""Separated complete-distribution and validation readouts for rev20-DC."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.stats import spearmanr

from src.topic4_d6_natural_kmeans import (
    best_binary_alignment, natural_kmeans, normalize_event_ranks,
    patient_profiles,
)
from src.topic4_shaft_aware import (
    build_event_features, contract_groups, sliced_event_cloud_distance,
)
from src.topic4_shaft_aware_direction import assign_direction_modes


def embedding_from_training_arrays(arrays: Mapping) -> dict:
    return {
        "center": np.asarray(arrays["feature_center"], float),
        "scale": np.asarray(arrays["feature_scale"], float),
        "components": np.asarray(arrays["pca_components"], float),
        "directions": np.asarray(arrays["sw_directions"], float),
        "reference_z": np.asarray(arrays["global_reference_z"], float),
    }


def complete_distribution_distance(onsets, *, groups, embedding) -> float:
    """Unconditional all-family distance; this function has no label input."""
    features = build_event_features(np.asarray(onsets, float), groups)["features"]
    return sliced_event_cloud_distance(features, embedding)


def matched_patient_floor(patient_onsets, sample_size: int, *, groups,
                          embedding, draws: int, seed: int) -> dict:
    values = np.asarray(patient_onsets, float)
    sample_size = min(int(sample_size), len(values))
    if sample_size < 2:
        return {"draws": 0, "q05": None, "q50": None, "q95": None}
    rng = np.random.default_rng(int(seed))
    distances = np.asarray([
        complete_distribution_distance(
            values[rng.choice(len(values), sample_size, replace=False)],
            groups=groups, embedding=embedding,
        )
        for _ in range(int(draws))
    ])
    return {
        "draws": int(len(distances)),
        "sample_size": sample_size,
        "q05": float(np.quantile(distances, 0.05)),
        "q50": float(np.quantile(distances, 0.50)),
        "q95": float(np.quantile(distances, 0.95)),
    }


def _prototype_matrix(model_ranks, mapped_labels, patient_ranks,
                      patient_labels) -> np.ndarray:
    model = normalize_event_ranks(np.asarray(model_ranks, float))
    patient = patient_profiles(patient_ranks, patient_labels)
    prototypes = np.full_like(patient, np.nan)
    for mode in (0, 1):
        selected = np.asarray(mapped_labels, int) == mode
        if np.any(selected):
            count = np.sum(np.isfinite(model[selected]), axis=0)
            prototypes[mode] = np.divide(
                np.nansum(model[selected], axis=0), count,
                out=np.full(model.shape[1], np.nan), where=count > 0,
            )
    matrix = np.full((2, 2), np.nan)
    for model_mode in (0, 1):
        for patient_mode in (0, 1):
            finite = (
                np.isfinite(prototypes[model_mode])
                & np.isfinite(patient[patient_mode])
            )
            if np.sum(finite) >= 3:
                matrix[model_mode, patient_mode] = float(spearmanr(
                    prototypes[model_mode, finite], patient[patient_mode, finite],
                ).statistic)
    return matrix


def score_returned_families(onsets, ranks, returned, *, contract: Mapping,
                            training_arrays: Mapping,
                            classifier: Mapping, kmeans_seed: int) -> dict:
    """Score one network without allowing validation metrics into selection."""
    onsets = np.asarray(onsets, float)
    ranks = np.asarray(ranks, float)
    returned = np.asarray(returned, bool)
    if onsets.ndim != 2 or ranks.shape != onsets.shape:
        raise ValueError("onsets and ranks must be aligned event x contact tables")
    if returned.shape != (len(onsets),):
        raise ValueError("returned mask must align with events")
    values = onsets[returned]
    rank_values = ranks[returned]
    groups = contract_groups(contract)
    embedding = embedding_from_training_arrays(training_arrays)
    distribution = complete_distribution_distance(
        values, groups=groups, embedding=embedding,
    ) if len(values) >= 2 else float("nan")

    assigned = assign_direction_modes(
        values, groups=groups, embedding=embedding, classifier=classifier,
    )
    readable = np.sum(np.isfinite(values), axis=1) >= 3
    support_ood = np.asarray(assigned["ood"], bool)
    all_ood = ~readable | support_ood
    natural = natural_kmeans(
        rank_values, np.asarray(assigned["labels"], int),
        random_state=int(kmeans_seed),
    )
    validation = {
        "n_returned_families": int(len(values)),
        "n_readable_families": int(np.sum(readable)),
        "unreadable_fraction": float(np.mean(~readable)) if len(values) else 1.0,
        "ood_all_returned": float(np.mean(all_ood)) if len(values) else 1.0,
        "ood_readable_only": (
            float(np.mean(support_ood[readable])) if np.any(readable) else 1.0
        ),
        "natural_kmeans_status": natural["status"],
    }
    if natural["status"] == "OK":
        valid = np.asarray(natural["valid_event_mask"], bool)
        clusters = np.asarray(natural["cluster_labels"], int)
        alignment = best_binary_alignment(
            clusters, np.asarray(assigned["labels"], int)[valid],
        )
        mapped = np.asarray(alignment["mapped_labels"], int)
        matrix = _prototype_matrix(
            rank_values[valid], mapped,
            np.asarray(training_arrays["patient_train_ranks"], float),
            np.asarray(training_arrays["patient_train_old_labels"], int),
        )
        validation.update({
            "cluster_counts": np.bincount(mapped, minlength=2).tolist(),
            "two_clusters_present": bool(np.all(
                np.bincount(mapped, minlength=2) > 0
            )),
            "direction_balanced_alignment": natural[
                "direction_balanced_alignment"
            ],
            "direction_purity": natural["direction_purity"],
            "kmeans_seed_ami_median": natural["kmeans_seed_ami_median"],
            "silhouette": natural["silhouette"],
            "prototype_spearman_matrix": matrix.tolist(),
        })
    return {
        "selection": {
            "complete_distribution_distance_training": (
                None if not np.isfinite(distribution) else float(distribution)
            ),
            "selection_used_labels": False,
            "selection_used_ood": False,
            "selection_used_heldout": False,
        },
        "validation_diagnostic": validation,
    }
