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
    transform_patient_embedding,
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


def reference_embedding(patient_onsets, *, groups, embedding) -> dict:
    """Reuse the frozen training transform with a different event-cloud reference."""
    features = build_event_features(np.asarray(patient_onsets, float), groups)[
        "features"
    ]
    output = dict(embedding)
    output["reference_z"] = transform_patient_embedding(features, embedding)
    return output


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


def score_complete_distribution(onsets, returned, *, contract: Mapping,
                                training_arrays: Mapping) -> dict:
    """Selection-only score with no classifier, KMeans, OOD or held-out input."""
    onsets = np.asarray(onsets, float)
    returned = np.asarray(returned, bool)
    if onsets.ndim != 2:
        raise ValueError("onsets must be an event x contact table")
    if returned.shape != (len(onsets),):
        raise ValueError("returned mask must align with events")
    values = onsets[returned]
    groups = contract_groups(contract)
    embedding = embedding_from_training_arrays(training_arrays)
    distribution = complete_distribution_distance(
        values, groups=groups, embedding=embedding,
    ) if len(values) >= 2 else float("nan")
    return {
        "complete_distribution_distance_training": (
            None if not np.isfinite(distribution) else float(distribution)
        ),
        "n_returned_families": int(len(values)),
        "selection_used_labels": False,
        "selection_used_ood": False,
        "selection_used_heldout": False,
    }


def score_validation_endpoints(
        onsets, ranks, returned, *, contract: Mapping,
        training_arrays: Mapping, classifier: Mapping, kmeans_seed: int,
        patient_reference_onsets=None, patient_reference_ranks=None,
        patient_reference_labels=None) -> dict:
    """Validation-only KMeans/OOD and optional frozen-reference readout."""
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
        reference_ranks = (
            np.asarray(training_arrays["patient_train_ranks"], float)
            if patient_reference_ranks is None
            else np.asarray(patient_reference_ranks, float)
        )
        reference_labels = (
            np.asarray(training_arrays["patient_train_old_labels"], int)
            if patient_reference_labels is None
            else np.asarray(patient_reference_labels, int)
        )
        matrix = _prototype_matrix(
            rank_values[valid], mapped,
            reference_ranks, reference_labels,
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
    if patient_reference_onsets is not None:
        reference = reference_embedding(
            patient_reference_onsets, groups=groups, embedding=embedding,
        )
        distance = complete_distribution_distance(
            values, groups=groups, embedding=reference,
        ) if len(values) >= 2 else float("nan")
        validation["complete_distribution_distance_reference"] = (
            None if not np.isfinite(distance) else float(distance)
        )
    return validation


def score_returned_families(onsets, ranks, returned, *, contract: Mapping,
                            training_arrays: Mapping,
                            classifier: Mapping, kmeans_seed: int) -> dict:
    """Backward-compatible combined canary diagnostic."""
    return {
        "selection": score_complete_distribution(
            onsets, returned, contract=contract, training_arrays=training_arrays,
        ),
        "validation_diagnostic": score_validation_endpoints(
            onsets, ranks, returned, contract=contract,
            training_arrays=training_arrays, classifier=classifier,
            kmeans_seed=kmeans_seed,
        ),
    }
