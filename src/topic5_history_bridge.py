"""Low-capacity, patient-LOSO readout utilities for the history-to-ictal bridge."""
from __future__ import annotations

import numpy as np


def causal_ewma_contact_fields(
    participation: np.ndarray,
    relative_rank: np.ndarray,
    event_time: np.ndarray,
    *,
    cutoff_epoch: float,
    half_life_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal, time-weighted contact participation and rank fields.

    This is the minimal activity-load slow-state control.  It is commutative
    in event content once event-to-time assignment is fixed and therefore does
    not assume a nonlinear sequence grammar.
    """

    part = np.asarray(participation, dtype=np.float64)
    rank = np.asarray(relative_rank, dtype=np.float64)
    time = np.asarray(event_time, dtype=np.float64)
    if part.ndim != 2 or rank.shape != part.shape:
        raise ValueError("participation and relative_rank must be [event,contact]")
    if time.shape != (len(part),) or not len(time):
        raise ValueError("event_time must align with a nonempty causal prefix")
    half_life = float(half_life_hours)
    if half_life <= 0:
        raise ValueError("half_life_hours must be positive")
    age = float(cutoff_epoch) - time
    if np.any(~np.isfinite(age)) or np.any(age < 0):
        raise ValueError("all events must precede the causal cutoff")
    weight = np.exp(-np.log(2.0) * age / (half_life * 3600.0))
    denominator = max(float(weight.sum()), np.finfo(float).tiny)
    participation_field = (weight[:, None] * part).sum(0) / denominator
    valid_rank = np.isfinite(rank) & (part > 0)
    rank_weight = weight[:, None] * valid_rank
    rank_denominator = rank_weight.sum(0)
    rank_field = np.divide(
        (rank_weight * np.nan_to_num(rank, nan=0.0)).sum(0),
        rank_denominator,
        out=np.zeros(part.shape[1], dtype=np.float64),
        where=rank_denominator > 0,
    )
    return participation_field.astype(np.float32), rank_field.astype(np.float32)


def leave_one_seizure_out_residual(
    fields: np.ndarray,
) -> np.ndarray:
    """Residualize each seizure field by the mean of the other seizures."""

    value = np.asarray(fields, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 2:
        raise ValueError("fields must contain at least two seizures")
    total = value.sum(0, keepdims=True)
    other_mean = (total - value) / float(value.shape[0] - 1)
    return value - other_mean


def causal_contact_features(
    base_features: np.ndarray,
    prefix_participation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace train80 participation features by seizure-causal estimates."""

    features = np.asarray(base_features, dtype=np.float32).copy()
    participation = np.asarray(prefix_participation, dtype=np.float64)
    if participation.ndim != 2 or participation.shape[1] != features.shape[0]:
        raise ValueError("prefix participation must be [event,contact]")
    if participation.shape[0] == 0:
        raise ValueError("causal prefix must contain at least one event")
    support = (participation.sum(0) + 0.5) / (len(participation) + 1.0)
    features[:, 0] = support
    features[:, 1] = support - support.mean()
    return features, support.astype(np.float32)


def centered_field(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value - np.mean(value, axis=-1, keepdims=True)


def robust_z_field(value: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Within-field robust z-score used only for centered-energy evaluation."""

    value = np.asarray(value, dtype=np.float64)
    median = np.median(value, axis=-1, keepdims=True)
    mad = np.median(np.abs(value - median), axis=-1, keepdims=True)
    scale = np.maximum(1.4826 * mad, float(epsilon))
    return (value - median) / scale


def weighted_ridge_fit(
    features: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    *,
    alpha: float,
) -> dict[str, np.ndarray | float]:
    """Fit a shared low-parameter ridge after weighted train-only scaling."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    weight = np.asarray(sample_weight, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or weight.shape != (len(x),):
        raise ValueError("ridge arrays do not align")
    if np.any(weight < 0) or not np.any(weight > 0):
        raise ValueError("sample weights must be nonnegative and nonzero")
    weight = weight / weight.sum()
    mean = np.sum(x * weight[:, None], axis=0)
    centered = x - mean
    scale = np.sqrt(np.sum(np.square(centered) * weight[:, None], axis=0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = centered / scale
    y_mean = float(np.sum(y * weight))
    y_centered = y - y_mean
    root_weight = np.sqrt(weight)
    xw = standardized * root_weight[:, None]
    yw = y_centered * root_weight
    penalty = float(alpha) * np.eye(x.shape[1])
    system = xw.T @ xw + penalty
    right = xw.T @ yw
    try:
        coefficient = np.linalg.solve(system, right)
    except np.linalg.LinAlgError:
        coefficient = np.linalg.pinv(system) @ right
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "target_mean": y_mean,
        "coefficient": coefficient,
        "alpha": float(alpha),
    }


def weighted_ridge_predict(model: dict, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    return (
        (x - np.asarray(model["feature_mean"]))
        / np.asarray(model["feature_scale"])
    ) @ np.asarray(model["coefficient"]) + float(model["target_mean"])


def patient_balanced_contact_weights(
    patient_ids: np.ndarray,
    seizure_ids: np.ndarray,
    contact_counts: np.ndarray,
) -> np.ndarray:
    """Assign equal total weight to patients, seizures within patient, contacts."""

    patient = np.asarray(patient_ids).astype(str)
    seizure = np.asarray(seizure_ids).astype(str)
    counts = np.asarray(contact_counts, dtype=int)
    if not (len(patient) == len(seizure) == len(counts)):
        raise ValueError("weight arrays do not align")
    unique_patient = np.unique(patient)
    out = np.zeros(len(patient), dtype=np.float64)
    for current in unique_patient:
        patient_mask = patient == current
        patient_seizures = np.unique(seizure[patient_mask])
        for current_seizure in patient_seizures:
            mask = patient_mask & (seizure == current_seizure)
            if not np.any(mask):
                continue
            expected = np.unique(counts[mask])
            if len(expected) != 1 or int(expected[0]) != int(np.sum(mask)):
                raise ValueError("contact count does not match seizure rows")
            out[mask] = 1.0 / (
                len(unique_patient) * len(patient_seizures) * int(expected[0])
            )
    return out
