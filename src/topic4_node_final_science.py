"""Final model-internal science audits for the rev15 Node-only field."""
from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from src.topic4_node_dualmode import cosine_similarity, source_topology_features
from src.topic4_node_dualmode import dual_mode_objective, lse_max


def _topology_summary_from_features(
    features_by_network: Sequence[np.ndarray],
    labels_by_network: Sequence[np.ndarray],
) -> dict:
    mode_templates = {0: [], 1: []}
    within_network = []
    for features, labels in zip(features_by_network, labels_by_network):
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels, dtype=int)
        network_reliability = []
        for mode in (0, 1):
            selected = features[labels == mode]
            mode_templates[mode].append(np.mean(selected, axis=0))
            if len(selected) >= 2 and len(selected[1::2]):
                reliability = cosine_similarity(
                    np.mean(selected[::2], axis=0),
                    np.mean(selected[1::2], axis=0),
                )
                if np.isfinite(reliability):
                    network_reliability.append(reliability)
        if network_reliability:
            within_network.append(float(np.mean(network_reliability)))

    modes = {}
    equal_network_templates = []
    for mode in (0, 1):
        templates = mode_templates[mode]
        similarities = np.asarray([
            cosine_similarity(templates[left], templates[right])
            for left, right in combinations(range(len(templates)), 2)
        ], dtype=float)
        similarities = similarities[np.isfinite(similarities)]
        modes[str(mode)] = {
            "n_networks": int(len(templates)),
            "pairwise_network_cosine_mean": (
                float(np.mean(similarities)) if len(similarities) else float("nan")
            ),
        }
        equal_network_templates.append(np.mean(templates, axis=0))
    separation = 1.0 - cosine_similarity(
        equal_network_templates[0], equal_network_templates[1],
    )
    across = np.asarray([
        modes[str(mode)]["pairwise_network_cosine_mean"] for mode in (0, 1)
    ], dtype=float)
    return {
        "modes": modes,
        "mean_within_network_split_half_cosine": (
            float(np.mean(within_network)) if within_network else float("nan")
        ),
        "mean_across_network_template_cosine": (
            float(np.mean(across)) if np.isfinite(across).all() else float("nan")
        ),
        "equal_network_between_mode_distance": float(separation),
    }


def weakest_mode_topology_quality(summary: dict) -> float:
    """Protect the weaker mode when combining reproducibility and separation."""
    reproducibility = np.asarray([
        summary["modes"][str(mode)]["pairwise_network_cosine_mean"]
        for mode in (0, 1)
    ], dtype=float)
    separation = float(summary["equal_network_between_mode_distance"])
    if not np.isfinite(reproducibility).all() or not np.isfinite(separation):
        return float("nan")
    return float(np.min(reproducibility) * separation)


def score_workers_against_patient_endpoint(
    workers: Sequence[dict],
    *,
    patient_ranks: np.ndarray,
    patient_labels: np.ndarray,
    contact_names: np.ndarray,
    projections: np.ndarray,
    calibration: dict,
) -> dict:
    """Score every independent network against one frozen patient endpoint."""
    rows = []
    for worker in workers:
        score = dual_mode_objective(
            np.asarray(worker["ranks"], dtype=float),
            np.asarray(worker["labels"], dtype=int),
            np.asarray(patient_ranks, dtype=float),
            np.asarray(patient_labels, dtype=int),
            np.asarray(contact_names).astype(str),
            missing_mode_penalty=1.0,
            projections=np.asarray(projections, dtype=float),
            calibration=calibration,
        )
        score.update({
            "seed": int(worker["seed"]),
            "n_events": int(len(worker["labels"])),
            "mode_counts": np.bincount(
                np.asarray(worker["labels"], dtype=int), minlength=2,
            ),
            "weakest_mode_cloud_lse": lse_max(np.asarray([
                score["modes"][str(mode)]["cloud"] for mode in (0, 1)
            ], dtype=float), tau=0.25),
        })
        rows.append(score)
    if not rows:
        raise ValueError("patient endpoint scoring requires at least one network")

    def mean(path: tuple[str, ...]) -> float:
        values = []
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            values.append(float(value))
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("patient endpoint scoring produced a non-finite loss")
        return float(np.mean(values))

    return {
        "n_networks": int(len(rows)),
        "patient_event_count": int(len(patient_labels)),
        "mean_weakest_mode_lse": mean(("weakest_mode_lse",)),
        "mean_mode_0_loss": mean(("modes", "0", "mean")),
        "mean_mode_1_loss": mean(("modes", "1", "mean")),
        "mean_weakest_mode_cloud_lse": mean(("weakest_mode_cloud_lse",)),
        "mean_mode_0_cloud_loss": mean(("modes", "0", "cloud")),
        "mean_mode_1_cloud_loss": mean(("modes", "1", "cloud")),
        "network_scores": rows,
    }


def topology_label_permutation_test(
    onset_maps_by_network: Sequence[np.ndarray],
    labels_by_network: Sequence[np.ndarray],
    *,
    draws: int = 4096,
    seed: int = 20260830,
) -> dict:
    """Compare mode-specific source topology with a within-network label null.

    Each permutation preserves every network's event count and A/B occupancy.
    Network identity, source maps and the number of events assigned to each mode
    therefore remain fixed.
    """
    if len(onset_maps_by_network) != len(labels_by_network):
        raise ValueError("network source maps and labels do not align")
    if len(onset_maps_by_network) < 2:
        raise ValueError("topology permutation requires at least two networks")
    if int(draws) < 1:
        raise ValueError("topology permutation draws must be positive")

    maps = [np.asarray(value, dtype=float) for value in onset_maps_by_network]
    labels = [np.asarray(value, dtype=int) for value in labels_by_network]
    for network_maps, network_labels in zip(maps, labels):
        if network_maps.ndim != 3 or network_labels.shape != (len(network_maps),):
            raise ValueError("one network source-map bundle does not align")
        if set(np.unique(network_labels).tolist()) != {0, 1}:
            raise ValueError("each network must contain both patient modes")

    features = [source_topology_features(value) for value in maps]
    observed_summary = _topology_summary_from_features(features, labels)
    observed = weakest_mode_topology_quality(observed_summary)
    if not np.isfinite(observed):
        raise ValueError("observed mode-specific topology is not evaluable")

    rng = np.random.default_rng(int(seed))
    null = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        permuted = [rng.permutation(value) for value in labels]
        null[draw] = weakest_mode_topology_quality(
            _topology_summary_from_features(features, permuted)
        )
    if not np.isfinite(null).all():
        raise RuntimeError("topology permutation produced a non-finite null")

    return {
        "observed_summary": observed_summary,
        "observed_weakest_mode_quality": observed,
        "null_q05": float(np.quantile(null, 0.05)),
        "null_q50": float(np.quantile(null, 0.50)),
        "null_q95": float(np.quantile(null, 0.95)),
        "upper_tail_p": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
        "above_null_q95": bool(observed > float(np.quantile(null, 0.95))),
        "draws": int(draws),
        "seed": int(seed),
        "null_contract": (
            "within-network label permutation preserving network identity, "
            "source maps and per-network mode occupancy"
        ),
        "quality_contract": (
            "minimum mode-specific cross-network template cosine multiplied "
            "by equal-network between-mode topology distance"
        ),
    }


def final_zero_simulation_decision(
    candidate_score: dict,
    reference_score: dict,
    topology_test: dict,
    reference_topology_test: dict | None = None,
) -> dict:
    """Apply the pre-intervention Node scientific clauses."""
    metrics = {
        "heldout_eventwise_prototype_r2": (
            float(candidate_score["heldout_eventwise_prototype_r2"]),
            float(reference_score["heldout_eventwise_prototype_r2"]),
        ),
        "weakest_mode_cloud_loss": (
            float(candidate_score["mean_weakest_mode_cloud_lse"]),
            float(reference_score["mean_weakest_mode_cloud_lse"]),
        ),
        "mode_0_cloud_loss": (
            float(candidate_score["mean_mode_0_cloud_loss"]),
            float(reference_score["mean_mode_0_cloud_loss"]),
        ),
        "mode_1_cloud_loss": (
            float(candidate_score["mean_mode_1_cloud_loss"]),
            float(reference_score["mean_mode_1_cloud_loss"]),
        ),
        "weakest_mode_loss": (
            float(candidate_score["mean_weakest_mode_lse"]),
            float(reference_score["mean_weakest_mode_lse"]),
        ),
        "mode_0_loss": (
            float(candidate_score["mean_mode_0_loss"]),
            float(reference_score["mean_mode_0_loss"]),
        ),
        "mode_1_loss": (
            float(candidate_score["mean_mode_1_loss"]),
            float(reference_score["mean_mode_1_loss"]),
        ),
    }
    if not np.isfinite(np.asarray(list(metrics.values()), dtype=float)).all():
        raise ValueError("held-out final-science metrics must be finite")
    r2, reference_r2 = metrics["heldout_eventwise_prototype_r2"]
    clauses = {
        "positive_heldout_eventwise_prototype_r2_and_improves_reference": {
            "candidate": r2,
            "reference": reference_r2,
            "delta_candidate_minus_reference": r2 - reference_r2,
            "pass": bool(r2 > 0.0 and r2 > reference_r2),
        },
        "complete_heldout_event_distribution_improves_reference": {
            "weakest_mode_candidate": metrics["weakest_mode_cloud_loss"][0],
            "weakest_mode_reference": metrics["weakest_mode_cloud_loss"][1],
            "mode_0_delta_candidate_minus_reference": (
                metrics["mode_0_cloud_loss"][0]
                - metrics["mode_0_cloud_loss"][1]
            ),
            "mode_1_delta_candidate_minus_reference": (
                metrics["mode_1_cloud_loss"][0]
                - metrics["mode_1_cloud_loss"][1]
            ),
            "pass": bool(
                metrics["weakest_mode_cloud_loss"][0]
                < metrics["weakest_mode_cloud_loss"][1]
                and metrics["mode_0_cloud_loss"][0]
                < metrics["mode_0_cloud_loss"][1]
                and metrics["mode_1_cloud_loss"][0]
                < metrics["mode_1_cloud_loss"][1]
            ),
            "metric_contract": (
                "mode-conditioned shaft-balanced sliced-Wasserstein distance "
                "over every held-out recruitment/rank event vector"
            ),
        },
        "weakest_mode_loss_improves_reference": {
            "candidate": metrics["weakest_mode_loss"][0],
            "reference": metrics["weakest_mode_loss"][1],
            "delta_candidate_minus_reference": (
                metrics["weakest_mode_loss"][0]
                - metrics["weakest_mode_loss"][1]
            ),
            "pass": bool(
                metrics["weakest_mode_loss"][0]
                < metrics["weakest_mode_loss"][1]
            ),
        },
        "both_patient_mode_losses_improve_reference": {
            "mode_0_delta_candidate_minus_reference": (
                metrics["mode_0_loss"][0] - metrics["mode_0_loss"][1]
            ),
            "mode_1_delta_candidate_minus_reference": (
                metrics["mode_1_loss"][0] - metrics["mode_1_loss"][1]
            ),
            "pass": bool(
                metrics["mode_0_loss"][0] < metrics["mode_0_loss"][1]
                and metrics["mode_1_loss"][0] < metrics["mode_1_loss"][1]
            ),
        },
        "mode_specific_source_topology_above_matched_null": {
            "observed": float(topology_test["observed_weakest_mode_quality"]),
            "null_q95": float(topology_test["null_q95"]),
            "upper_tail_p": float(topology_test["upper_tail_p"]),
            "pass": bool(topology_test["above_null_q95"]),
        },
    }
    if reference_topology_test is not None:
        candidate_topology = float(
            topology_test["observed_weakest_mode_quality"]
        )
        reference_topology = float(
            reference_topology_test["observed_weakest_mode_quality"]
        )
        if not np.isfinite(candidate_topology) or not np.isfinite(
            reference_topology
        ):
            raise ValueError("source-topology comparison must be finite")
        clauses["mode_specific_source_topology_improves_reference"] = {
            "candidate": candidate_topology,
            "reference": reference_topology,
            "delta_candidate_minus_reference": (
                candidate_topology - reference_topology
            ),
            "pass": bool(candidate_topology > reference_topology),
        }
    return {
        "accepted_for_same_checkpoint_intervention": bool(
            all(row["pass"] for row in clauses.values())
        ),
        "clauses": clauses,
        "intervention_completed": False,
        "node_freeze_permitted": False,
    }
