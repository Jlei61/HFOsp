"""Search and scoring helpers for the rev9-L component-pair oracle."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import qmc

from src.topic4_core_field_profile import (
    sliced_embedding_distance,
    transform_rank_curves,
)
from src.topic4_rev9_factorial import normalized_event_ranks, pairwise_precedence


DESCRIPTOR_NAMES = (
    "recruitment_mean_absolute_error",
    "precedence_mean_absolute_error",
    "mean_rank_profile_absolute_error",
    "event_distribution_sliced_wasserstein",
)


def sobol_candidates(search, bounds):
    """Generate the frozen 64-point search, reserving candidate 0 for zero."""
    n = int(search["n_candidates"])
    dimension = int(search["dimension"])
    if n < 2 or n & (n - 1):
        raise ValueError("Sobol candidate count must be a power of two")
    if dimension != 6:
        raise ValueError("rev9-L component-pair search is six-dimensional")
    lower, upper = map(float, bounds)
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("invalid gamma bounds")
    engine = qmc.Sobol(
        d=dimension, scramble=bool(search["scramble"]),
        seed=int(search["seed"]))
    points = qmc.scale(engine.random_base2(int(math.log2(n))), lower, upper)
    if search.get("candidate_zero_reserved_for_gamma0") is not True:
        raise ValueError("candidate zero must be reserved for the scalar baseline")
    points[0] = 0.0
    return [
        {"candidate_id": f"sobol_{index:03d}", "gamma": row.tolist()}
        for index, row in enumerate(points)
    ]


def selection_candidates_with_baseline(top_candidates, *, dimension=6):
    """Return the frozen fit selections plus the zero-residual comparator."""
    candidates = [
        {
            "candidate_id": str(row["candidate_id"]),
            "gamma": np.asarray(row["gamma"], float).tolist(),
        }
        for row in top_candidates
    ]
    if any(len(row["gamma"]) != dimension for row in candidates):
        raise ValueError("selection candidates do not match the gamma dimension")
    identifiers = [row["candidate_id"] for row in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("selection candidate identifiers must be unique")
    if "sobol_000" not in identifiers:
        candidates.append({
            "candidate_id": "sobol_000",
            "gamma": np.zeros(dimension, float).tolist(),
        })
    return candidates


def patient_descriptor_floor(
        patient_curves, patient_ranks, patient_labels, patient_blocks,
        reference, *, n_per_mode, repeats, seed, scale_minimum=1e-6):
    """Estimate matched-count training floors using independent block draws."""
    curves = np.asarray(patient_curves, float)
    ranks = np.asarray(patient_ranks, float)
    labels = np.asarray(patient_labels, int)
    blocks = np.asarray(patient_blocks)
    if not (len(curves) == len(ranks) == len(labels) == len(blocks)):
        raise ValueError("patient floor arrays must align")
    n_per_mode, repeats = int(n_per_mode), int(repeats)
    if n_per_mode < 2 or repeats < 10:
        raise ValueError("patient floor requires at least two events and ten repeats")
    rng = np.random.default_rng(int(seed))
    eligible = {}
    for mode in (0, 1):
        mode_blocks = np.unique(blocks[labels == mode])
        if len(mode_blocks) < n_per_mode:
            raise ValueError(f"mode {mode} has fewer than {n_per_mode} blocks")
        eligible[mode] = mode_blocks

    patient_z = transform_rank_curves(curves, reference)

    def mean_rank(values):
        normalized = normalized_event_ranks(values)
        count = np.isfinite(normalized).sum(axis=0)
        return np.divide(
            np.nansum(normalized, axis=0), count,
            out=np.full(normalized.shape[1], np.nan), where=count > 0)

    def mean_absolute(left, right, *, off_diagonal=False):
        valid = np.isfinite(left) & np.isfinite(right)
        if off_diagonal:
            valid &= ~np.eye(left.shape[0], dtype=bool)
        return float(np.mean(np.abs(left[valid] - right[valid])))

    patient_reference = {}
    for mode in (0, 1):
        use = labels == mode
        precedence, _ = pairwise_precedence(ranks[use])
        patient_reference[mode] = {
            "recruitment": np.isfinite(ranks[use]).mean(axis=0),
            "precedence": precedence,
            "profile": mean_rank(ranks[use]),
            "z": patient_z[use],
        }

    samples = {
        mode: {name: np.empty(repeats, float) for name in DESCRIPTOR_NAMES}
        for mode in ("A", "B")
    }
    sampled_blocks = np.empty((repeats, 2, n_per_mode), blocks.dtype)
    for repeat in range(repeats):
        selected_indices, selected_labels = [], []
        for mode in (0, 1):
            chosen_blocks = rng.choice(eligible[mode], n_per_mode, replace=False)
            sampled_blocks[repeat, mode] = chosen_blocks
            for block in chosen_blocks:
                options = np.flatnonzero((labels == mode) & (blocks == block))
                selected_indices.append(int(rng.choice(options)))
                selected_labels.append(mode)
        selected_indices = np.asarray(selected_indices, int)
        selected_labels = np.asarray(selected_labels, int)
        selected_z = transform_rank_curves(curves[selected_indices], reference)
        for mode, name in enumerate(("A", "B")):
            use = selected_labels == mode
            selected_ranks = ranks[selected_indices][use]
            recruitment = np.isfinite(selected_ranks).mean(axis=0)
            precedence, _ = pairwise_precedence(selected_ranks)
            profile = mean_rank(selected_ranks)
            values = {
                "recruitment_mean_absolute_error": mean_absolute(
                    recruitment, patient_reference[mode]["recruitment"]),
                "precedence_mean_absolute_error": mean_absolute(
                    precedence, patient_reference[mode]["precedence"],
                    off_diagonal=True),
                "mean_rank_profile_absolute_error": mean_absolute(
                    profile, patient_reference[mode]["profile"]),
                "event_distribution_sliced_wasserstein": sliced_embedding_distance(
                    selected_z[use], patient_reference[mode]["z"],
                    reference["directions"]),
            }
            for metric, value in values.items():
                if not np.isfinite(value):
                    raise RuntimeError(f"non-finite patient floor: {name}/{metric}")
                samples[name][metric][repeat] = float(value)

    summary = {"modes": {}}
    for mode in ("A", "B"):
        summary["modes"][mode] = {}
        for name in DESCRIPTOR_NAMES:
            values = samples[mode][name]
            q05, q25, median, q75, q95 = np.quantile(
                values, [0.05, 0.25, 0.5, 0.75, 0.95])
            summary["modes"][mode][name] = {
                "q05": float(q05),
                "q25": float(q25),
                "median": float(median),
                "q75": float(q75),
                "q95": float(q95),
                "scale_iqr": float(max(q75 - q25, float(scale_minimum))),
                "n": repeats,
            }
    return summary, samples, sampled_blocks


def score_candidate(descriptors, floor, readable_fraction, ood_fraction, *,
                    readable_weight, tau, ood_weight):
    """Compute the floor-normalized weakest-mode training objective."""
    mode_scores, standardized = {}, {}
    for mode in ("A", "B"):
        values, standardized[mode] = [], {}
        row = descriptors["modes"][mode]
        for name in DESCRIPTOR_NAMES:
            value = row[name]
            calibration = floor["modes"][mode][name]
            if value is None or not np.isfinite(value):
                z = float("inf")
                penalty = float("inf")
            else:
                z = (float(value) - calibration["median"]) / calibration["scale_iqr"]
                penalty = float(np.logaddexp(0.0, z))
            standardized[mode][name] = {
                "raw": None if value is None else float(value),
                "z": z,
                "softplus": penalty,
            }
            values.append(penalty)
        readable = float(readable_fraction[mode])
        mode_scores[mode] = float(
            np.mean(values) + float(readable_weight) * (1.0 - readable))
    tau = float(tau)
    maximum = max(mode_scores.values())
    shape = float(maximum + tau * math.log(np.mean([
        math.exp((value - maximum) / tau) for value in mode_scores.values()
    ])))
    ood = float(np.mean([ood_fraction["A"], ood_fraction["B"]]))
    return {
        "mode_scores": mode_scores,
        "standardized_descriptors": standardized,
        "weakest_mode_shape": shape,
        "mean_ood_fraction": ood,
        "objective": float(shape + float(ood_weight) * ood),
        "weak_mode": max(mode_scores, key=mode_scores.get),
    }


__all__ = [
    "DESCRIPTOR_NAMES",
    "patient_descriptor_floor",
    "score_candidate",
    "selection_candidates_with_baseline",
    "sobol_candidates",
]
