"""Natural-proportion and cross-fitted KMeans metrics for Topic 4 D6."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

from src.lagpat_rank_audit import build_masked_kmeans_features


def normalize_event_ranks(ranks):
    ranks = np.asarray(ranks, float)
    output = np.full_like(ranks, np.nan)
    for index, row in enumerate(ranks):
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        values = row[finite]
        span = float(np.max(values) - np.min(values))
        output[index, finite] = (
            (values - np.min(values)) / span if span > 0 else 0.0
        )
    return output


def contact_split_folds(contract):
    """Alternate contacts within each shaft, preserving shaft identity."""
    folds = [[], []]
    for row in contract["contacts"]:
        fold = int(row["within_shaft_order_by_shared_axis"]) % 2
        folds[fold].append(int(row["contact_index"]))
    output = tuple(np.asarray(values, int) for values in folds)
    if set(output[0]).intersection(set(output[1])):
        raise ValueError("contact folds overlap")
    if len(output[0]) + len(output[1]) != len(contract["contacts"]):
        raise ValueError("contact folds do not cover the contract")
    return output


def patient_profiles(patient_ranks, patient_labels):
    ranks = normalize_event_ranks(patient_ranks)
    labels = np.asarray(patient_labels, int)
    return np.asarray([
        np.nanmean(ranks[labels == mode], axis=0) for mode in (0, 1)
    ])


def assign_to_profiles(ranks, profiles, contacts, *, minimum_contacts=3):
    ranks = normalize_event_ranks(ranks)
    contacts = np.asarray(contacts, int)
    labels = np.full(len(ranks), -1, int)
    distances = np.full((len(ranks), 2), np.nan)
    for event, row in enumerate(ranks):
        for mode in (0, 1):
            finite = (
                np.isfinite(row[contacts])
                & np.isfinite(profiles[mode, contacts])
            )
            if np.sum(finite) >= int(minimum_contacts):
                difference = row[contacts][finite] - profiles[mode, contacts][finite]
                distances[event, mode] = float(np.sqrt(np.mean(difference ** 2)))
        if np.all(np.isfinite(distances[event])):
            labels[event] = int(np.argmin(distances[event]))
    return labels, distances


def _spearman_matrix(model, patient, contacts):
    matrix = np.full((2, 2), np.nan)
    contacts = np.asarray(contacts, int)
    for row in (0, 1):
        for column in (0, 1):
            finite = (
                np.isfinite(model[row, contacts])
                & np.isfinite(patient[column, contacts])
            )
            if np.sum(finite) >= 3:
                matrix[row, column] = float(spearmanr(
                    model[row, contacts][finite],
                    patient[column, contacts][finite],
                ).statistic)
    return matrix


def signed_matrix_margin(matrix):
    matrix = np.asarray(matrix, float)
    signed = np.asarray([
        matrix[0, 0], matrix[1, 1], -matrix[0, 1], -matrix[1, 0],
    ])
    return float(np.nanmin(signed)) if np.any(np.isfinite(signed)) else None


def crossfit_patient_readout(ranks, patient_ranks, patient_labels, folds):
    """Assign on one contact fold and evaluate patient geometry on the other."""
    ranks = normalize_event_ranks(ranks)
    profiles = patient_profiles(patient_ranks, patient_labels)
    fold_rows, assignments = [], []
    for assignment_contacts, evaluation_contacts in (
            (folds[0], folds[1]), (folds[1], folds[0])):
        labels, distances = assign_to_profiles(
            ranks, profiles, assignment_contacts,
        )
        model = np.full_like(profiles, np.nan)
        counts = []
        for mode in (0, 1):
            selected = labels == mode
            counts.append(int(np.sum(selected)))
            if np.any(selected):
                values = ranks[selected]
                count = np.sum(np.isfinite(values), axis=0)
                model[mode] = np.divide(
                    np.nansum(values, axis=0), count,
                    out=np.full(values.shape[1], np.nan), where=count > 0,
                )
        matrix = _spearman_matrix(model, profiles, evaluation_contacts)
        fold_rows.append({
            "assignment_contacts": assignment_contacts.tolist(),
            "evaluation_contacts": evaluation_contacts.tolist(),
            "mode_counts": counts,
            "matrix": matrix,
            "signed_margin": signed_matrix_margin(matrix),
            "n_assigned": int(np.sum(labels >= 0)),
            "mean_assignment_margin": float(np.nanmean(
                np.abs(distances[:, 0] - distances[:, 1])
            )) if np.any(np.all(np.isfinite(distances), axis=1)) else None,
        })
        assignments.append(labels)
    stack = np.asarray([row["matrix"] for row in fold_rows], float)
    matrix = np.nanmean(stack, axis=0)
    consensus = np.where(
        (assignments[0] >= 0) & (assignments[0] == assignments[1]),
        assignments[0], -1,
    )
    return {
        "folds": fold_rows,
        "matrix": matrix,
        "signed_margin": signed_matrix_margin(matrix),
        "consensus_labels": consensus,
        "consensus_count": int(np.sum(consensus >= 0)),
        "consensus_fraction": float(np.mean(consensus >= 0)) if len(ranks) else 0.0,
    }


def best_binary_alignment(cluster_labels, direction_labels):
    cluster_labels = np.asarray(cluster_labels, int)
    direction_labels = np.asarray(direction_labels, int)
    valid = direction_labels >= 0
    contingency = np.zeros((2, 2), int)
    for cluster, mode in zip(cluster_labels[valid], direction_labels[valid]):
        contingency[int(cluster), int(mode)] += 1
    identity = contingency[0, 0] + contingency[1, 1]
    swapped = contingency[0, 1] + contingency[1, 0]
    swap = bool(swapped > identity)
    mapped = 1 - cluster_labels if swap else cluster_labels.copy()
    purity = float(max(identity, swapped) / max(1, contingency.sum()))
    recalls = []
    for mode in (0, 1):
        selected = valid & (direction_labels == mode)
        recalls.append(
            float(np.mean(mapped[selected] == mode)) if np.any(selected) else np.nan
        )
    balanced = float(np.nanmean(recalls)) if np.any(np.isfinite(recalls)) else None
    return {
        "purity": purity,
        "balanced_alignment": balanced,
        "contingency": contingency,
        "mapped_labels": mapped,
        "n_direction_labeled": int(np.sum(valid)),
    }


def _gmm_heldout_delta(features, random_state):
    if len(features) < 12:
        return None
    folds = min(3, max(2, len(features) // 8))
    splitter = KFold(n_splits=folds, shuffle=True, random_state=int(random_state))
    deltas = []
    for train, test in splitter.split(features):
        scores = []
        for components in (1, 2):
            model = GaussianMixture(
                n_components=components, covariance_type="diag",
                reg_covar=1e-5, n_init=3, random_state=int(random_state),
            ).fit(features[train])
            scores.append(float(model.score(features[test])))
        deltas.append(scores[1] - scores[0])
    return float(np.mean(deltas))


def natural_kmeans(ranks, direction_labels, *, random_state=0):
    ranks = np.asarray(ranks, float)
    valid = np.sum(np.isfinite(ranks), axis=1) >= 3
    ranks = ranks[valid]
    direction_labels = np.asarray(direction_labels, int)[valid]
    if len(ranks) < 8:
        return {"status": "INSUFFICIENT_EVENTS", "n_events": int(len(ranks))}
    rank_matrix = ranks.T
    participation = np.isfinite(rank_matrix)
    features = build_masked_kmeans_features(
        rank_matrix, participation, impute="event_median",
    )
    label_sets = [
        KMeans(n_clusters=2, n_init=50, random_state=seed).fit_predict(features)
        for seed in range(int(random_state), int(random_state) + 8)
    ]
    labels = label_sets[0]
    alignment = best_binary_alignment(labels, direction_labels)
    stability = [
        adjusted_mutual_info_score(label_sets[0], other)
        for other in label_sets[1:]
    ]
    recruited = np.sum(np.isfinite(ranks), axis=1).astype(float)
    extent_r = (
        float(abs(np.corrcoef(labels, recruited)[0, 1]))
        if np.std(recruited) > 0 and np.std(labels) > 0 else 0.0
    )
    projected = features @ (
        np.mean(features[labels == 1], axis=0)
        - np.mean(features[labels == 0], axis=0)
    )
    if np.mean(projected[labels == 1]) < np.mean(projected[labels == 0]):
        projected = -projected
    scale = max(float(np.std(projected)), 1e-12)
    valley_gap = float(
        (np.quantile(projected[labels == 1], 0.10)
         - np.quantile(projected[labels == 0], 0.90)) / scale
    )
    return {
        "status": "OK",
        "n_events": int(len(labels)),
        "cluster_counts": np.bincount(labels, minlength=2).tolist(),
        "direction_purity": alignment["purity"],
        "direction_balanced_alignment": alignment["balanced_alignment"],
        "direction_contingency": alignment["contingency"].tolist(),
        "n_crossfit_direction_labeled": alignment["n_direction_labeled"],
        "crossfit_direction_fraction": float(
            alignment["n_direction_labeled"] / max(1, len(labels))
        ),
        "kmeans_seed_ami_median": float(np.median(stability)),
        "silhouette": float(silhouette_score(features, labels)),
        "heldout_gmm_k2_minus_k1_loglik_per_event": _gmm_heldout_delta(
            features, random_state,
        ),
        "centroid_axis_valley_gap_sd": valley_gap,
        "absolute_cluster_extent_correlation": extent_r,
        "valid_event_mask": valid,
        "cluster_labels": labels,
    }


def network_bootstrap(values, *, draws=2000, seed=20260814):
    values = np.asarray([value for value in values if value is not None], float)
    if not len(values):
        return None
    rng = np.random.default_rng(int(seed))
    sampled = rng.choice(values, size=(int(draws), len(values)), replace=True)
    means = np.mean(sampled, axis=1)
    return {
        "n_networks": int(len(values)),
        "equal_network_mean": float(np.mean(values)),
        "network_bootstrap_q05": float(np.quantile(means, 0.05)),
        "network_bootstrap_q95": float(np.quantile(means, 0.95)),
    }
