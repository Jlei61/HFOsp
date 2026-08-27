"""Training-only rev14 objective for static Node free-field fitting.

The module has no patient or simulator I/O. Natural KMeans and patient held-out
data are deliberately absent: model candidates are ranked only by the frozen
patient-training representation and model-internal support diagnostics.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from src.topic4_node_dualmode import (
    event_features,
    js_divergence,
    lse_max,
    mode_distribution_components,
    normalize_event_ranks,
    shaft_balanced_feature_weights,
)


COMPONENTS = ("recruitment", "precedence", "profile", "cloud")


def _effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    denominator = float(np.sum(values ** 2))
    return 0.0 if denominator <= 0.0 else float(np.sum(values) ** 2 / denominator)


def _confidence_adjusted_support(weights: np.ndarray,
                                 confidence: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if weights.shape != confidence.shape or np.any(confidence < 0.0):
        raise ValueError("mode weights and confidence do not align")
    mass = float(np.sum(weights))
    if mass <= 0.0:
        return 0.0
    mean_confidence = float(np.sum(weights * confidence) / mass)
    return _effective_sample_size(weights) * mean_confidence


def _weighted_unique_sample_with_missing(
        ranks: np.ndarray, weights: np.ndarray, *, sample_size: int,
        rng: np.random.Generator) -> np.ndarray:
    """Sample unique events and represent missing support as un-recruited rows."""
    ranks = np.asarray(ranks, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    sample_size = int(sample_size)
    if ranks.ndim != 2 or weights.shape != (len(ranks),):
        raise ValueError("ranks and weights do not align")
    if sample_size <= 0 or np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("sample size and weights must be valid")
    output = np.full((sample_size, ranks.shape[1]), np.nan, dtype=np.float64)
    positive = np.flatnonzero(weights > 0.0)
    n_take = min(sample_size, len(positive))
    if n_take:
        probability = weights[positive] / float(np.sum(weights[positive]))
        selected = rng.choice(positive, size=n_take, replace=False, p=probability)
        output[:n_take] = ranks[selected]
    return output


def _eligible_patient_blocks(labels: np.ndarray, blocks: np.ndarray, *,
                             mode: int, sample_size: int) -> np.ndarray:
    return np.asarray([
        block for block in np.unique(blocks)
        if np.sum((labels == int(mode)) & (blocks == block)) >= int(sample_size)
    ])


def _floor_q95(calibration: Mapping, mode: int, component: str) -> float:
    value = float(calibration["modes"][str(int(mode))][component]["floor_q95"])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"nonpositive patient floor q95 for mode {mode} {component}")
    return value


def matched_mode_distances(
        model_ranks: np.ndarray, probability_b: np.ndarray,
        patient_ranks: np.ndarray, patient_labels: np.ndarray,
        patient_blocks: np.ndarray, contact_names: np.ndarray, *,
        projections: np.ndarray, calibration: Mapping,
        sample_size: int = 6, draws: int = 64,
        seed: int = 20260827) -> dict:
    """Estimate four-layer mode distances using matched unique 6-vs-6 draws."""
    model_ranks = np.asarray(model_ranks, dtype=np.float64)
    probability_b = np.asarray(probability_b, dtype=np.float64)
    patient_ranks = np.asarray(patient_ranks, dtype=np.float64)
    patient_labels = np.asarray(patient_labels, dtype=np.int8)
    patient_blocks = np.asarray(patient_blocks)
    if (model_ranks.ndim != 2 or probability_b.shape != (len(model_ranks),)
            or patient_ranks.ndim != 2
            or patient_ranks.shape[1] != model_ranks.shape[1]
            or patient_labels.shape != (len(patient_ranks),)
            or patient_blocks.shape != (len(patient_ranks),)
            or not np.all(np.isfinite(probability_b))
            or np.any((probability_b < 0.0) | (probability_b > 1.0))):
        raise ValueError("rev14 objective arrays do not align")
    if int(sample_size) <= 0 or int(draws) <= 0:
        raise ValueError("sample_size and draws must be positive")

    rng = np.random.default_rng(int(seed))
    output = {}
    confidence = np.abs(2.0 * probability_b - 1.0)
    for mode, weights in enumerate((1.0 - probability_b, probability_b)):
        effective = _confidence_adjusted_support(weights, confidence)
        sampling_weights = weights * confidence
        blocks = _eligible_patient_blocks(
            patient_labels, patient_blocks, mode=mode, sample_size=sample_size,
        )
        if len(blocks) == 0:
            raise ValueError(f"patient mode {mode} has no eligible recording block")
        if float(np.sum(sampling_weights)) <= 0.0:
            normalized = {key: 2.0 for key in COMPONENTS}
            raw_mean = None
        else:
            raw_draws = {key: [] for key in COMPONENTS}
            for _ in range(int(draws)):
                model_sample = _weighted_unique_sample_with_missing(
                    model_ranks, sampling_weights,
                    sample_size=sample_size, rng=rng,
                )
                block = rng.choice(blocks)
                available = np.flatnonzero(
                    (patient_labels == mode) & (patient_blocks == block)
                )
                patient_sample = patient_ranks[rng.choice(
                    available, size=int(sample_size), replace=False,
                )]
                raw = mode_distribution_components(
                    model_sample, patient_sample, contact_names, projections,
                )
                for key in COMPONENTS:
                    raw_draws[key].append(float(raw[key]))
            raw_mean = {key: float(np.mean(raw_draws[key])) for key in COMPONENTS}
            normalized = {
                key: raw_mean[key] / _floor_q95(calibration, mode, key)
                for key in COMPONENTS
            }
            support_fraction = min(1.0, effective / float(sample_size))
            normalized = {
                key: support_fraction * value + (1.0 - support_fraction) * 2.0
                for key, value in normalized.items()
            }
        output[str(mode)] = {
            **normalized,
            "mean": float(np.mean([normalized[key] for key in COMPONENTS])),
            "raw_draw_mean": raw_mean,
            "effective_events": effective,
            "confidence_adjusted": True,
            "soft_occupancy": float(np.mean(weights)) if len(weights) else 0.0,
        }
    return output


def _contrast(model_ranks: np.ndarray, probability_b: np.ndarray,
              patient_ranks: np.ndarray, patient_labels: np.ndarray,
              contact_names: np.ndarray) -> dict:
    model = event_features(normalize_event_ranks(model_ranks))
    patient = event_features(normalize_event_ranks(patient_ranks))
    probability_b = np.asarray(probability_b, dtype=np.float64)
    mode_weights = (1.0 - probability_b, probability_b)
    if not len(model) or any(float(np.sum(weights)) <= 0.0 for weights in mode_weights):
        return {"loss": 1.0, "alignment": 0.0}
    model_prototypes = np.asarray([
        np.sum((weights / np.sum(weights))[:, None] * model, axis=0)
        for weights in mode_weights
    ])
    patient_prototypes = np.asarray([
        np.mean(patient[np.asarray(patient_labels) == mode], axis=0)
        for mode in (0, 1)
    ])
    weights = shaft_balanced_feature_weights(contact_names)
    model_delta = model_prototypes[0] - model_prototypes[1]
    patient_delta = patient_prototypes[0] - patient_prototypes[1]
    model_norm = float(np.sqrt(np.sum(weights * model_delta ** 2)))
    patient_norm = float(np.sqrt(np.sum(weights * patient_delta ** 2)))
    denominator = model_norm * patient_norm
    cosine = 0.0 if denominator <= 0.0 else float(
        np.sum(weights * model_delta * patient_delta) / denominator
    )
    ratio = 0.0 if patient_norm <= 0.0 else model_norm / patient_norm
    amplitude = 0.0 if ratio <= 0.0 else float(np.exp(-abs(np.log(ratio))))
    alignment = float(max(0.0, cosine) * amplitude)
    return {"loss": 1.0 - alignment, "alignment": alignment}


def rev14_objective(
        model_ranks: np.ndarray, probability_b: np.ndarray,
        patient_ranks: np.ndarray, patient_labels: np.ndarray,
        patient_blocks: np.ndarray, contact_names: np.ndarray, *,
        projections: np.ndarray, calibration: Mapping,
        returned_families: int, contact_evaluable_families: int,
        overlap_excluded_families: int, less_than_three_contact_families: int,
        sample_size: int = 6, draws: int = 64, seed: int = 20260827,
        tau: float = 0.25) -> dict:
    """Compute finite J14 with explicit weak-mode and event-support costs."""
    model_ranks = np.asarray(model_ranks, dtype=np.float64)
    probability_b = np.asarray(probability_b, dtype=np.float64)
    returned = int(returned_families)
    contact_evaluable = int(contact_evaluable_families)
    overlap = int(overlap_excluded_families)
    under_recruited = int(less_than_three_contact_families)
    if not (0 <= contact_evaluable <= returned
            and 0 <= overlap <= contact_evaluable
            and 0 <= under_recruited <= returned):
        raise ValueError("event support counts violate the rev14 contract")
    if model_ranks.ndim != 2 or probability_b.shape != (len(model_ranks),):
        raise ValueError("model ranks and probabilities do not align")
    if len(model_ranks) > contact_evaluable - overlap:
        raise ValueError("scored events exceed isolated contact-evaluable support")

    if len(model_ranks):
        modes = matched_mode_distances(
            model_ranks, probability_b, patient_ranks, patient_labels,
            patient_blocks, contact_names, projections=projections,
            calibration=calibration, sample_size=sample_size, draws=draws,
            seed=seed,
        )
        model_occupancy = np.asarray([
            np.mean(1.0 - probability_b), np.mean(probability_b),
        ])
        patient_occupancy = np.bincount(
            np.asarray(patient_labels, dtype=np.int8), minlength=2,
        )
        occupancy = js_divergence(model_occupancy, patient_occupancy)
        ambiguity = float(np.mean(4.0 * probability_b * (1.0 - probability_b)))
        contrast = _contrast(
            model_ranks, probability_b, patient_ranks, patient_labels,
            contact_names,
        )
    else:
        modes = {
            str(mode): {
                **{key: 2.0 for key in COMPONENTS},
                "mean": 2.0,
                "raw_draw_mean": None,
                "effective_events": 0.0,
                "soft_occupancy": 0.0,
            }
            for mode in (0, 1)
        }
        occupancy = float(np.log(2.0))
        ambiguity = 0.0
        contrast = {"loss": 1.0, "alignment": 0.0}

    effective = np.asarray([
        float(modes[str(mode)]["effective_events"]) for mode in (0, 1)
    ])
    support = float(0.5 * np.sum(
        float(sample_size) / (effective + float(sample_size))
    ))
    contact_non_evaluable_fraction = (
        0.0 if returned == 0 else (returned - contact_evaluable) / returned
    )
    under_recruited_fraction = 0.0 if returned == 0 else under_recruited / returned
    support += contact_non_evaluable_fraction + under_recruited_fraction
    overlap_fraction = (
        0.0 if contact_evaluable == 0 else overlap / contact_evaluable
    )
    weakest = lse_max(
        np.asarray([modes["0"]["mean"], modes["1"]["mean"]]), tau=tau,
    )
    objective = (
        weakest + 0.5 * occupancy + 0.25 * ambiguity
        + 0.5 * float(contrast["loss"]) + 0.5 * overlap_fraction
        + 0.25 * support
    )
    return {
        "objective": float(objective),
        "modes": modes,
        "weakest_mode_lse": weakest,
        "occupancy_js": occupancy,
        "ambiguity": ambiguity,
        "contrast": contrast,
        "overlap_fraction": overlap_fraction,
        "support_loss": support,
        "support": {
            "effective_events": effective,
            "contact_non_evaluable_fraction": contact_non_evaluable_fraction,
            "less_than_three_contact_fraction": under_recruited_fraction,
        },
        "sampling": {
            "sample_size_per_side": int(sample_size),
            "draws": int(draws),
            "seed": int(seed),
            "model_sampling": "weighted_without_replacement_then_missing_rows",
            "mode_support": "soft_membership_times_absolute_classifier_margin",
            "patient_sampling": "within_one_training_recording_block_without_replacement",
            "normalization": "raw_distance_divided_by_patient_cross_block_floor_q95",
        },
    }
