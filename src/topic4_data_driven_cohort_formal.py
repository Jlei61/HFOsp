"""Shaft-balanced scoring for the formal data-driven SNN cohort."""
from __future__ import annotations

from itertools import combinations, permutations, product
from math import factorial

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

from src.topic4_canonical_shaft_layout import (
    balanced_precedence_error,
    balanced_recruitment_error,
    contact_shaft_contract,
)
from src.topic4_data_driven_cohort import (
    MODE_NAMES,
    _correlation_matrix,
    _mode_descriptors,
    _model_features,
    _profiles,
)


def _shaft_balanced_profile_error(model: np.ndarray, patient: np.ndarray,
                                  shaft_ids: list[str]) -> float:
    model = np.asarray(model, float)
    patient = np.asarray(patient, float)
    shafts = np.asarray(shaft_ids, object)
    if model.shape != patient.shape or model.shape != shafts.shape:
        raise ValueError("profiles and shaft ids must align")
    errors = []
    for shaft in sorted(set(shafts.tolist())):
        selected = shafts == shaft
        joint = selected & np.isfinite(model) & np.isfinite(patient)
        errors.append(
            float(np.mean(np.abs(model[joint] - patient[joint])))
            if np.any(joint) else 1.0
        )
    return float(np.mean(errors))


def _target_arrays(target: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    profiles = np.asarray(target["profiles"], float)
    recruitment = np.asarray(target["recruitment"], float)
    precedence = np.asarray(target["precedence"], float)
    if profiles.shape[0] != 2 or recruitment.shape != profiles.shape:
        raise ValueError("patient mode profile/recruitment target must have two modes")
    if precedence.ndim != 3 or precedence.shape[0] != 2 or precedence.shape[2] != 3:
        raise ValueError("patient precedence target must be (mode, pair, three states)")
    return profiles, recruitment, precedence


def _assert_target_contact_contract(target: dict, contact_names: list[str]) -> None:
    """Fail closed when the model montage and the patient target disagree.

    Shapes alone cannot catch a reordered montage: every array stays
    broadcast-compatible while contact ``i`` of the model is compared with a
    different patient contact.  The target therefore has to carry its own
    contact order and it has to match element by element.
    """
    if "contact_order" not in target:
        raise ValueError("patient target must carry contact_order for alignment")
    order = [str(value) for value in target["contact_order"]]
    if order != [str(value) for value in contact_names]:
        raise ValueError(
            "model contact names do not match the patient target contact order"
        )
    profiles, _, precedence = _target_arrays(target)
    if profiles.shape[1] != len(order):
        raise ValueError("patient profile target does not span the contact order")
    expected_pairs = len(order) * (len(order) - 1) // 2
    if precedence.shape[1] != expected_pairs:
        raise ValueError("patient precedence target does not span the contact pairs")


def _mode_loss(model_descriptor: dict, target: dict, mode: int, *,
               contact_names: list[str], shaft_ids: list[str],
               pair_indices: np.ndarray) -> dict:
    profiles, recruitment, precedence = _target_arrays(target)
    name = MODE_NAMES[int(mode)]
    profile_error = _shaft_balanced_profile_error(
        model_descriptor[name]["profile"], profiles[mode], shaft_ids,
    )
    recruitment_error = balanced_recruitment_error(
        model_descriptor[name]["recruitment"], recruitment[mode], shaft_ids,
    )
    precedence_error = balanced_precedence_error(
        model_descriptor[name]["precedence"], precedence[mode],
        contact_names, pair_indices,
    )
    return {
        "profile_error": profile_error,
        "recruitment_error": recruitment_error,
        "precedence_error": precedence_error,
        "loss": float(np.mean([
            profile_error, recruitment_error, precedence_error,
        ])),
    }


def _natural_clustering(features: np.ndarray, masked: np.ndarray, target: dict, *,
                        contact_names: list[str], shaft_ids: list[str],
                        pair_indices: np.ndarray, kmeans_seed: int,
                        kmeans_n_init: int) -> dict:
    natural = KMeans(
        n_clusters=2, n_init=int(kmeans_n_init), random_state=int(kmeans_seed),
    ).fit(features)
    descriptors = _mode_descriptors(masked, natural.labels_)
    costs = np.empty((2, 2), float)
    details = {}
    for raw_mode in (0, 1):
        for patient_mode in (0, 1):
            renamed = {
                MODE_NAMES[patient_mode]: descriptors[MODE_NAMES[raw_mode]],
            }
            profile, recruitment, precedence = _target_arrays(target)
            name = MODE_NAMES[patient_mode]
            current = {
                name: renamed[name],
                "pair_indices": pair_indices,
            }
            loss = _mode_loss(
                current, {
                    "profiles": profile,
                    "recruitment": recruitment,
                    "precedence": precedence,
                }, patient_mode, contact_names=contact_names,
                shaft_ids=shaft_ids, pair_indices=pair_indices,
            )
            costs[raw_mode, patient_mode] = loss["loss"]
            details[(raw_mode, patient_mode)] = loss
    rows, columns = linear_sum_assignment(costs)
    mapping = np.full(2, -1, int)
    mapping[rows] = columns
    aligned_descriptors = {
        MODE_NAMES[patient_mode]: descriptors[MODE_NAMES[raw_mode]]
        for raw_mode, patient_mode in zip(rows, columns)
    }
    aligned_profiles = np.asarray([
        aligned_descriptors[name]["profile"] for name in MODE_NAMES
    ])
    profile_matrix = _correlation_matrix(
        aligned_profiles, np.asarray(target["profiles"], float),
    )
    stability_labels = [
        KMeans(
            n_clusters=2, n_init=int(kmeans_n_init), random_state=seed,
        ).fit_predict(features)
        for seed in range(int(kmeans_seed), int(kmeans_seed) + 5)
    ]
    stability = [
        adjusted_mutual_info_score(stability_labels[left], stability_labels[right])
        for left in range(len(stability_labels))
        for right in range(left + 1, len(stability_labels))
    ]
    aligned_losses = np.asarray([
        details[(raw_mode, patient_mode)]["loss"]
        for raw_mode, patient_mode in sorted(zip(rows, columns), key=lambda row: row[1])
    ])
    return {
        "cluster_counts": np.bincount(natural.labels_, minlength=2),
        "cluster_to_patient_mode": mapping,
        "raw_labels": np.asarray(natural.labels_, int),
        # 0 is the cluster matched to the patient's TA, 1 the one matched to TB;
        # the figure must never show a raw KMeans id as a mode name.
        "aligned_labels": mapping[np.asarray(natural.labels_, int)],
        "mode_losses": aligned_losses,
        "weakest_mode_loss": float(np.max(aligned_losses)),
        "patient_profile_matrix": profile_matrix,
        "seed_ami_median": float(np.median(stability)),
        "silhouette": (
            float(silhouette_score(features, natural.labels_))
            if len(features) > 2 and len(np.unique(natural.labels_)) == 2 else None
        ),
    }


def score_model_ranks_shaft_balanced(
        model_ranks: np.ndarray, *, patient_centers: np.ndarray,
        target: dict, contact_names: list[str], patient_ood_threshold: float,
        minimum_contacts: int = 3, minimum_events_per_mode: int = 3,
        kmeans_seed: int = 20260815, kmeans_n_init: int = 20,
        include_natural_kmeans: bool = True) -> dict:
    """Score one network without letting contact or shaft density set the loss.

    ``include_natural_kmeans`` exists for the permutation null, which only needs
    the supervised statistic; running the unsupervised clustering for every one
    of the 64 draws would cost far more than the simulation being scored.
    """
    model_ranks = np.asarray(model_ranks, float)
    _assert_target_contact_contract(target, contact_names)
    contract = contact_shaft_contract(contact_names)
    if model_ranks.ndim != 2 or model_ranks.shape[1] != len(contact_names):
        raise ValueError("model ranks and contact names do not align")
    readable = np.isfinite(model_ranks).sum(axis=1) >= int(minimum_contacts)
    model_ranks = model_ranks[readable]
    minimum_total = 2 * int(minimum_events_per_mode)
    if len(model_ranks) < minimum_total:
        return {
            "status": "INSUFFICIENT_EVENTS",
            "n_readable_events": int(len(model_ranks)),
            "selection_score": 2.0,
        }
    features, masked = _model_features(model_ranks)
    centers = np.asarray(patient_centers, float)
    if centers.shape != (2, features.shape[1]):
        raise ValueError("patient centres and model features do not align")
    distance = np.linalg.norm(
        features[:, None, :] - centers[None, :, :], axis=2,
    )
    labels = np.argmin(distance, axis=1)
    assigned = distance[np.arange(len(distance)), labels]
    in_distribution = assigned <= float(patient_ood_threshold)
    counts = np.bincount(labels[in_distribution], minlength=2)
    ood_fraction = float(1.0 - in_distribution.mean())
    if np.any(counts < int(minimum_events_per_mode)):
        return {
            "status": "INSUFFICIENT_IN_DISTRIBUTION_MODE_SUPPORT",
            "n_readable_events": int(len(model_ranks)),
            "n_in_distribution_events": int(in_distribution.sum()),
            "supervised_mode_counts": counts,
            "ood_fraction": ood_fraction,
            "selection_score": float(1.5 + 0.5 * ood_fraction),
        }
    use_ranks = masked[in_distribution]
    use_labels = labels[in_distribution]
    descriptors = _mode_descriptors(use_ranks, use_labels)
    canonical_pairs = np.asarray(
        list(combinations(range(len(contact_names)), 2)), int,
    )
    if not np.array_equal(descriptors["pair_indices"], canonical_pairs):
        raise RuntimeError("model precedence pair contract changed")
    losses = [
        _mode_loss(
            descriptors, target, mode, contact_names=contact_names,
            shaft_ids=contract["shaft_ids"], pair_indices=canonical_pairs,
        )
        for mode in (0, 1)
    ]
    profiles = _profiles(use_ranks, use_labels)
    profile_matrix = _correlation_matrix(
        profiles, np.asarray(target["profiles"], float),
    )
    natural = _natural_clustering(
        features, masked, target, contact_names=contact_names,
        shaft_ids=contract["shaft_ids"], pair_indices=canonical_pairs,
        kmeans_seed=kmeans_seed, kmeans_n_init=kmeans_n_init,
    ) if include_natural_kmeans else None
    mode_losses = np.asarray([row["loss"] for row in losses])
    return {
        "status": "EVALUABLE",
        "n_readable_events": int(len(model_ranks)),
        "n_in_distribution_events": int(in_distribution.sum()),
        "supervised_mode_counts": counts,
        "ood_fraction": ood_fraction,
        "supervised_profile_matrix": profile_matrix,
        "mode_details": losses,
        "mode_losses": mode_losses,
        "weakest_mode_loss": float(np.max(mode_losses)),
        "selection_score": float(np.max(mode_losses) + 0.5 * ood_fraction),
        "natural_kmeans": natural,
    }


def within_shaft_null_contract(contact_names: list[str], *, n_permutations: int,
                               seed: int) -> dict:
    """Freeze distinct non-identity contact permutations that preserve shafts.

    Small montages do not have 64 distinct within-shaft permutations.  Filling
    the requested count by drawing with replacement would hide that: the null
    would look like 64 draws while carrying only a handful of distinct
    alternatives, so the smallest reachable permutation p-value is far above
    what the row count suggests.  When the non-identity group is no larger than
    the request the whole group is enumerated (an exact null); otherwise
    distinct permutations are drawn.  Either way the effective null size is
    reported and downstream code must use it instead of ``n_permutations``.
    """
    contract = contact_shaft_contract(contact_names)
    shafts = np.asarray(contract["shaft_ids"], object)
    groups = [np.flatnonzero(shafts == shaft) for shaft in contract["shaft_order"]]
    if not any(len(group) > 1 for group in groups):
        raise ValueError("within-shaft null needs a multi-contact shaft")
    n_permutations = int(n_permutations)
    if n_permutations < 1:
        raise ValueError("within-shaft null needs at least one permutation")
    identity = np.arange(len(contact_names), dtype=int)
    group_size = 1
    for group in groups:
        group_size *= factorial(len(group))
    n_non_identity = group_size - 1

    if n_non_identity <= n_permutations:
        rows = []
        for assignment in product(*[permutations(group.tolist()) for group in groups]):
            permutation = identity.copy()
            for group, ordering in zip(groups, assignment):
                permutation[group] = np.asarray(ordering, int)
            if not np.array_equal(permutation, identity):
                rows.append(permutation)
        exhaustive = True
    else:
        rng = np.random.default_rng(int(seed))
        seen, rows = set(), []
        maximum_attempts = max(1000, 200 * n_permutations)
        for _ in range(maximum_attempts):
            permutation = identity.copy()
            for group in groups:
                permutation[group] = rng.permutation(group)
            key = permutation.tobytes()
            if np.array_equal(permutation, identity) or key in seen:
                continue
            seen.add(key)
            rows.append(permutation)
            if len(rows) == n_permutations:
                break
        if len(rows) != n_permutations:
            raise RuntimeError("could not generate requested within-shaft nulls")
        exhaustive = False

    return {
        "permutations": np.asarray(rows, int),
        "n_requested": n_permutations,
        "effective_null_size": len(rows),
        "within_shaft_group_size": int(group_size),
        "n_non_identity_permutations": int(n_non_identity),
        "exhaustive": exhaustive,
        "minimum_reachable_p": 1.0 / float(len(rows) + 1),
        "seed": int(seed),
    }


def within_shaft_permutations(contact_names: list[str], *, n_permutations: int,
                              seed: int) -> np.ndarray:
    """Return only the frozen permutation rows of the within-shaft null."""
    return within_shaft_null_contract(
        contact_names, n_permutations=n_permutations, seed=seed,
    )["permutations"]
