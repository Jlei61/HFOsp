"""Pure analysis helpers for the Topic 4 rev9-L learnability audit."""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.stats import spearmanr

from src.topic4_core_field_profile import (
    sliced_embedding_distance,
    transform_rank_curves,
)
from src.topic4_rev9_factorial import normalized_event_ranks, pairwise_precedence


MODE_NAMES = ("A", "B")


def theta_sha256(theta):
    """Hash a parameter vector with the frozen rev8 little-endian contract."""
    return hashlib.sha256(np.asarray(theta, dtype="<f8").tobytes()).hexdigest()


def correlation_loss(correlation):
    """Map a Spearman correlation in [-1, 1] to a loss in [0, 1]."""
    value = float(correlation)
    if not np.isfinite(value) or value < -1.0 - 1e-12 or value > 1.0 + 1e-12:
        raise ValueError("correlation must be finite and in [-1, 1]")
    return 0.5 * (1.0 - float(np.clip(value, -1.0, 1.0)))


def centered_smooth_worst(losses, *, tau):
    """Centered log-sum-exp approximation to the worst mode loss.

    Centering by log(n) makes the value equal to the common loss when every
    mode is equally difficult. It is a diagnostic proxy, not the rev9-L full
    recruitment/precedence/profile/distribution objective.
    """
    values = np.asarray(losses, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("losses must be a non-empty finite vector")
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    maximum = float(values.max())
    return float(maximum + tau * math.log(np.exp((values - maximum) / tau).mean()))


def candidate_replay_rows(checkpoint, *, tau):
    """Extract only metrics genuinely retained in a rev8 optimization file."""
    rows = []
    for history_index, record in enumerate(checkpoint.get("history", [])):
        mode = record.get("mode") or {}
        matched = mode.get("matched_correlations")
        losses = None
        if (isinstance(matched, (list, tuple)) and len(matched) == 2
                and all(value is not None and np.isfinite(value) for value in matched)):
            losses = [correlation_loss(value) for value in matched]
        rows.append({
            "history_index": int(history_index),
            "generation": record.get("generation"),
            "theta_sha256": theta_sha256(record.get("theta", [])),
            "old_joint_loss": record.get("joint_loss"),
            "global_distance": record.get("distance"),
            "mode_a_correlation": None if losses is None else float(matched[0]),
            "mode_b_correlation": None if losses is None else float(matched[1]),
            "mode_a_loss": None if losses is None else float(losses[0]),
            "mode_b_loss": None if losses is None else float(losses[1]),
            "weak_mode_loss": None if losses is None else centered_smooth_worst(
                losses, tau=tau),
            "support_eligible": bool(mode.get("support_eligible", False)),
            "cluster_counts": mode.get("cluster_counts"),
            "n_usable": record.get("n_usable"),
            "n_detected": record.get("n_detected"),
            "metric_availability": {
                "global_distance": "retained",
                "prototype_similarity": "retained" if losses is not None else "missing",
                "recruitment": "not_retained",
                "precedence": "not_retained",
                "event_distribution": "not_retained",
            },
        })
    return rows


def dominates(left, right, *, keys=("mode_a_loss", "mode_b_loss")):
    """Return true when left is no worse everywhere and better somewhere."""
    a = np.asarray([left.get(key) for key in keys], float)
    b = np.asarray([right.get(key) for key in keys], float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return False
    return bool(np.all(a <= b) and np.any(a < b))


def pareto_front_indices(rows, *, require_support=True,
                         keys=("mode_a_loss", "mode_b_loss")):
    """Indices on a two-loss Pareto front, optionally after support filtering."""
    eligible = [
        index for index, row in enumerate(rows)
        if (not require_support or bool(row.get("support_eligible", False)))
        and all(row.get(key) is not None and np.isfinite(row.get(key)) for key in keys)
    ]
    return [
        index for index in eligible
        if not any(dominates(rows[other], rows[index], keys=keys)
                   for other in eligible if other != index)
    ]


def spearman_association(left, right):
    """JSON-safe Spearman result after pairwise finite filtering."""
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or left[valid].std() < 1e-12 or right[valid].std() < 1e-12:
        return {"rho": None, "pvalue": None, "n": int(valid.sum())}
    result = spearmanr(left[valid], right[valid])
    return {
        "rho": float(result.statistic),
        "pvalue": float(result.pvalue),
        "n": int(valid.sum()),
    }


def _summary(values, *, seed, repeats):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"median": None, "q05": None, "q95": None,
                "bootstrap_median_interval_95": [None, None], "n": 0}
    median = float(np.median(values))
    if len(values) == 1:
        interval = [median, median]
    else:
        rng = np.random.default_rng(int(seed))
        samples = rng.choice(values, size=(int(repeats), len(values)), replace=True)
        interval = np.quantile(np.median(samples, axis=1), [0.025, 0.975]).tolist()
    return {
        "median": median,
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "bootstrap_median_interval_95": interval,
        "n": int(len(values)),
    }


def _safe_spearman(left, right):
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or left[valid].std() < 1e-12 or right[valid].std() < 1e-12:
        return float("nan")
    return float(spearmanr(left[valid], right[valid]).statistic)


def block_mode_reliability(
        curves, block_ids, labels, *, embedded=None,
        min_events_per_block_mode=5, bootstrap_seed=0, bootstrap_repeats=2000):
    """Training-block stability of frozen modes without reading held-out scores.

    Each block prototype is compared with the same mode's complement prototype,
    so the block being assessed does not contribute to its reference. KMeans
    labels must already be frozen by the caller.
    """
    curves = np.asarray(curves, float)
    block_ids = np.asarray(block_ids)
    labels = np.asarray(labels, int)
    embedded = curves if embedded is None else np.asarray(embedded, float)
    if curves.ndim != 2 or embedded.ndim != 2:
        raise ValueError("curves and embedded must be two-dimensional")
    if not (len(curves) == len(block_ids) == len(labels) == len(embedded)):
        raise ValueError("curves, blocks, labels and embedded must align")
    if not np.isfinite(curves).all() or not np.isfinite(embedded).all():
        raise ValueError("curves and embedded must be finite")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must use frozen mode ids 0 and 1")

    blocks = np.unique(block_ids)
    per_mode = {mode: [] for mode in (0, 1)}
    proportions = []
    for block in blocks:
        in_block = block_ids == block
        counts = np.bincount(labels[in_block], minlength=2)
        proportions.append({
            "block_id": str(block),
            "n_events": int(in_block.sum()),
            "mode_a_count": int(counts[0]),
            "mode_b_count": int(counts[1]),
            "mode_a_fraction": float(counts[0] / counts.sum()),
        })
        for mode in (0, 1):
            current = in_block & (labels == mode)
            complement = (~in_block) & (labels == mode)
            if (current.sum() < int(min_events_per_block_mode)
                    or complement.sum() < int(min_events_per_block_mode)):
                continue
            block_curve = curves[current].mean(axis=0)
            complement_curve = curves[complement].mean(axis=0)
            block_center = embedded[current].mean(axis=0)
            complement_center = embedded[complement].mean(axis=0)
            within = np.linalg.norm(embedded[current] - block_center, axis=1)
            per_mode[mode].append({
                "block_id": str(block),
                "n_events": int(current.sum()),
                "block_to_complement_spearman": _safe_spearman(
                    block_curve, complement_curve),
                "within_block_dispersion": float(within.mean()),
                "between_block_dispersion": float(np.linalg.norm(
                    block_center - complement_center)),
            })

    result = {
        "scientific_role": "patient-training target stability; not held-out ceiling",
        "n_events": int(len(curves)),
        "n_blocks": int(len(blocks)),
        "min_events_per_block_mode": int(min_events_per_block_mode),
        "mode_proportion_by_block": proportions,
        "modes": {},
    }
    for mode, name in enumerate(MODE_NAMES):
        rows = per_mode[mode]
        seed = int(bootstrap_seed) + mode * 1000
        result["modes"][name] = {
            "n_events": int((labels == mode).sum()),
            "n_eligible_blocks": int(len(rows)),
            "block_rows": rows,
            "block_to_complement_spearman": _summary(
                [row["block_to_complement_spearman"] for row in rows],
                seed=seed, repeats=bootstrap_repeats),
            "within_block_dispersion": _summary(
                [row["within_block_dispersion"] for row in rows],
                seed=seed + 1, repeats=bootstrap_repeats),
            "between_block_dispersion": _summary(
                [row["between_block_dispersion"] for row in rows],
                seed=seed + 2, repeats=bootstrap_repeats),
        }
    return result


def binary_js_divergence(left_probability, right_probability):
    """Jensen-Shannon divergence between two binary mode proportions."""
    left = np.asarray([left_probability, 1.0 - left_probability], float)
    right = np.asarray([right_probability, 1.0 - right_probability], float)
    if (not np.isfinite(left).all() or not np.isfinite(right).all()
            or np.any(left < 0.0) or np.any(right < 0.0)):
        raise ValueError("mode probabilities must be finite and in [0, 1]")
    middle = 0.5 * (left + right)

    def kl(value):
        valid = value > 0.0
        return float(np.sum(value[valid] * np.log(value[valid] / middle[valid])))

    return 0.5 * (kl(left) + kl(right))


def _mean_normalized_rank(ranks):
    normalized = normalized_event_ranks(ranks)
    finite = np.isfinite(normalized)
    count = finite.sum(axis=0)
    total = np.nansum(normalized, axis=0)
    return np.divide(total, count, out=np.full(normalized.shape[1], np.nan),
                     where=count > 0)


def _mean_absolute(left, right, *, exclude_diagonal=False):
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    valid = np.isfinite(left) & np.isfinite(right)
    if exclude_diagonal and left.ndim == 2:
        valid &= ~np.eye(left.shape[0], dtype=bool)
    return None if not valid.any() else float(np.mean(np.abs(left[valid] - right[valid])))


def mode_conditioned_descriptor_replay(
        model_curves, model_ranks, model_labels,
        patient_curves, patient_ranks, patient_labels, reference):
    """Recompute four mode descriptors where per-event arrays are retained.

    Distances remain separate because their finite-sample scales have not yet
    been calibrated to a common patient-training floor. Combining them here
    would manufacture an arbitrary objective after seeing the data.
    """
    model_curves = np.asarray(model_curves, float)
    model_ranks = np.asarray(model_ranks, float)
    model_labels = np.asarray(model_labels, int)
    patient_curves = np.asarray(patient_curves, float)
    patient_ranks = np.asarray(patient_ranks, float)
    patient_labels = np.asarray(patient_labels, int)
    if (len(model_curves) != len(model_ranks)
            or len(model_curves) != len(model_labels)):
        raise ValueError("model curves, ranks and labels must align")
    if (len(patient_curves) != len(patient_ranks)
            or len(patient_curves) != len(patient_labels)):
        raise ValueError("patient curves, ranks and labels must align")
    if model_ranks.shape[1] != patient_ranks.shape[1]:
        raise ValueError("model and patient ranks must share contact order")

    patient_z = transform_rank_curves(patient_curves, reference)
    model_z = transform_rank_curves(model_curves, reference)
    modes = {}
    for mode, name in enumerate(MODE_NAMES):
        model_use = model_labels == mode
        patient_use = patient_labels == mode
        model_recruitment = np.isfinite(model_ranks[model_use]).mean(axis=0)
        patient_recruitment = np.isfinite(patient_ranks[patient_use]).mean(axis=0)
        model_precedence, model_support = pairwise_precedence(model_ranks[model_use])
        patient_precedence, patient_support = pairwise_precedence(patient_ranks[patient_use])
        model_profile = _mean_normalized_rank(model_ranks[model_use])
        patient_profile = _mean_normalized_rank(patient_ranks[patient_use])
        curve_correlation = _safe_spearman(
            model_curves[model_use].mean(axis=0),
            patient_curves[patient_use].mean(axis=0))
        modes[name] = {
            "n_model_events": int(model_use.sum()),
            "n_patient_train_events": int(patient_use.sum()),
            "recruitment_mean_absolute_error": _mean_absolute(
                model_recruitment, patient_recruitment),
            "precedence_mean_absolute_error": _mean_absolute(
                model_precedence, patient_precedence, exclude_diagonal=True),
            "mean_rank_profile_absolute_error": _mean_absolute(
                model_profile, patient_profile),
            "event_distribution_sliced_wasserstein": sliced_embedding_distance(
                model_z[model_use], patient_z[patient_use], reference["directions"]),
            "curve_prototype_spearman": (
                None if not np.isfinite(curve_correlation) else curve_correlation),
            "curve_prototype_loss": (
                None if not np.isfinite(curve_correlation)
                else correlation_loss(curve_correlation)),
            "model_recruitment_probability": model_recruitment.tolist(),
            "patient_recruitment_probability": patient_recruitment.tolist(),
            "model_mean_normalized_rank": model_profile.tolist(),
            "patient_mean_normalized_rank": patient_profile.tolist(),
            "model_precedence_pairs_with_support": int(
                np.sum((model_support > 0) & ~np.eye(model_support.shape[0], dtype=bool))),
            "patient_precedence_pairs_with_support": int(
                np.sum((patient_support > 0) & ~np.eye(patient_support.shape[0], dtype=bool))),
        }
    model_a_fraction = float(np.mean(model_labels == 0))
    patient_a_fraction = float(np.mean(patient_labels == 0))
    return {
        "metric_contract": (
            "four descriptors reported separately; no uncalibrated aggregate score"
        ),
        "model_mode_a_fraction": model_a_fraction,
        "patient_train_mode_a_fraction": patient_a_fraction,
        "mode_proportion_js": binary_js_divergence(
            model_a_fraction, patient_a_fraction),
        "modes": modes,
    }
