"""Masked linear observer and innovation-validity helpers for Topic 5 v3.0."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.topic5_event_innovation_v3_0 import MaskedRidgeFit, fit_masked_contact_ridge


EPS = 1e-12


@dataclass(frozen=True)
class StandardizedObserver:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    ridge: MaskedRidgeFit
    feature_name: str

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean):
            raise ValueError("observer feature matrix has the wrong shape")
        normalized = (values - self.feature_mean[None, :]) / self.feature_scale[None, :]
        return self.ridge.predict(normalized)


def fit_standardized_masked_observer(
    features: np.ndarray,
    target_rank: np.ndarray,
    participation: np.ndarray,
    *,
    alpha: float,
    feature_name: str,
    minimum_observations: int = 10,
    sample_weight: np.ndarray | None = None,
) -> StandardizedObserver:
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or np.any(~np.isfinite(values)):
        raise ValueError("observer features must be one finite matrix")
    weight = (
        np.ones(len(values), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if weight.shape != (len(values),) or np.any(~np.isfinite(weight)) or np.any(weight < 0) or np.sum(weight) <= 0:
        raise ValueError("sample_weight must contain finite non-negative mass")
    mean = np.average(values, axis=0, weights=weight)
    scale = np.sqrt(np.average((values - mean[None, :]) ** 2, axis=0, weights=weight))
    scale = np.where(scale > EPS, scale, 1.0)
    normalized = (values - mean[None, :]) / scale[None, :]
    ridge = fit_masked_contact_ridge(
        normalized,
        target_rank,
        participation,
        alpha=float(alpha),
        minimum_observations=int(minimum_observations),
        sample_weight=weight,
    )
    return StandardizedObserver(mean, scale, ridge, str(feature_name))


def masked_rank_mse(
    prediction: np.ndarray,
    target_rank: np.ndarray,
    participation: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    predicted = np.asarray(prediction, dtype=float)
    target = np.asarray(target_rank, dtype=float)
    valid = np.asarray(participation, dtype=bool)
    if predicted.shape != target.shape or valid.shape != target.shape:
        raise ValueError("masked score arrays must share one shape")
    valid = valid & np.isfinite(predicted) & np.isfinite(target)
    if not np.any(valid):
        raise ValueError("masked score has no valid targets")
    if sample_weight is None:
        return float(np.mean((predicted[valid] - target[valid]) ** 2))
    row_weight = np.asarray(sample_weight, dtype=float)
    if row_weight.shape != (len(target),) or np.any(~np.isfinite(row_weight)) or np.any(row_weight < 0):
        raise ValueError("sample_weight must be one finite non-negative row vector")
    expanded = np.broadcast_to(row_weight[:, None], target.shape)
    return float(np.average((predicted[valid] - target[valid]) ** 2, weights=expanded[valid]))


def concatenate_feature_ladder(
    state_features: Mapping[str, np.ndarray],
    nuisance: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the frozen pre20 -> multiscale -> lag-bin -> nuisance ladder."""

    required = ("pre20", "pre40", "pre80", "lag0_20", "lag20_40", "lag40_60", "lag60_80")
    missing = sorted(set(required) - set(state_features))
    if missing:
        raise ValueError(f"observer state features missing: {missing}")
    arrays = {name: np.asarray(state_features[name], dtype=float) for name in required}
    rows = {len(value) for value in arrays.values()}
    dimensions = {value.shape[1] for value in arrays.values() if value.ndim == 2}
    if len(rows) != 1 or len(dimensions) != 1 or any(value.ndim != 2 for value in arrays.values()):
        raise ValueError("observer state feature arrays are not aligned")
    nuisance_values = np.asarray(nuisance, dtype=float)
    if nuisance_values.ndim != 2 or len(nuisance_values) != next(iter(rows)):
        raise ValueError("observer nuisance features are not aligned")
    pre20 = arrays["pre20"]
    multiscale = np.hstack([pre20, arrays["pre40"], arrays["pre80"]])
    lag_bins = np.hstack([arrays[name] for name in required[3:]])
    return {
        "pre20": pre20,
        "pre20_40_80": multiscale,
        "four_lag_bins": lag_bins,
        "four_lag_bins_plus_time": np.hstack([lag_bins, nuisance_values]),
    }


def masked_max_abs_feature_residual_correlation(
    features: np.ndarray,
    residual: np.ndarray,
    valid: np.ndarray,
) -> float:
    """Maximum linear leftover dependence across feature/contact pairs."""

    design = np.asarray(features, dtype=float)
    errors = np.asarray(residual, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if design.ndim != 2 or errors.ndim != 2 or mask.shape != errors.shape or len(design) != len(errors):
        raise ValueError("residual-predictability arrays are not aligned")
    if np.any(~np.isfinite(design)):
        raise ValueError("residual-predictability features must be finite")
    maximum = 0.0
    found = False
    for contact in range(errors.shape[1]):
        selected = mask[:, contact] & np.isfinite(errors[:, contact])
        if np.sum(selected) < 4 or np.std(errors[selected, contact]) <= EPS:
            continue
        x = design[selected]
        y = errors[selected, contact]
        x_centered = x - x.mean(axis=0, keepdims=True)
        y_centered = y - y.mean()
        denominator = np.sqrt(
            np.sum(x_centered**2, axis=0) * np.sum(y_centered**2)
        )
        correlation = np.divide(
            x_centered.T @ y_centered,
            denominator,
            out=np.zeros(x.shape[1], dtype=float),
            where=denominator > EPS,
        )
        maximum = max(maximum, float(np.max(np.abs(correlation), initial=0.0)))
        found = True
    return maximum if found else float("nan")


def coherent_block_permutation(
    residual: np.ndarray,
    valid: np.ndarray,
    sequence_ids: np.ndarray,
    *,
    block_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute residual blocks once per sequence, preserving row coherence."""

    errors = np.asarray(residual)
    mask = np.asarray(valid)
    groups = np.asarray(sequence_ids)
    if errors.ndim != 2 or mask.shape != errors.shape or groups.shape != (len(errors),):
        raise ValueError("block permutation arrays are not aligned")
    block = int(block_size)
    if block < 1:
        raise ValueError("block_size must be positive")
    output = errors.copy()
    output_mask = mask.copy()
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        blocks = [rows[start : start + block] for start in range(0, len(rows), block)]
        order = rng.permutation(len(blocks))
        source = np.concatenate([blocks[index] for index in order]) if blocks else rows
        output[rows] = errors[source]
        output_mask[rows] = mask[source]
    return output, output_mask


def blocked_innovation_validity(
    features: np.ndarray,
    residual: np.ndarray,
    valid: np.ndarray,
    sequence_ids: np.ndarray,
    *,
    block_size: int,
    n_null: int,
    seed: int,
) -> dict[str, float | int | bool]:
    groups = np.asarray(sequence_ids)
    unique_groups = np.unique(groups)
    eligible_groups = [
        group
        for group in unique_groups
        if int(np.sum(groups == group)) >= 2 * int(block_size)
    ]
    if not eligible_groups:
        return {
            "valid": False,
            "observed_max_abs_correlation": float("nan"),
            "null_q95": float("nan"),
            "n_null_finite": 0,
            "n_eligible_groups": 0,
            "reason": "no continuity sequence contains at least two null blocks",
        }
    eligible = np.isin(groups, eligible_groups)
    groups = groups[eligible]
    features = np.asarray(features)[eligible]
    residual = np.asarray(residual)[eligible]
    valid = np.asarray(valid)[eligible]

    def statistic(errors: np.ndarray, mask: np.ndarray) -> float:
        values = [
            masked_max_abs_feature_residual_correlation(
                np.asarray(features)[groups == group],
                errors[groups == group],
                mask[groups == group],
            )
            for group in np.unique(groups)
        ]
        finite_values = [value for value in values if np.isfinite(value)]
        return max(finite_values) if finite_values else float("nan")

    observed = statistic(np.asarray(residual), np.asarray(valid))
    rng = np.random.default_rng(int(seed))
    null = np.empty(int(n_null), dtype=float)
    if len(null) < 20:
        raise ValueError("at least 20 blocked null draws are required")
    for index in range(len(null)):
        permuted, permuted_valid = coherent_block_permutation(
            residual, valid, groups, block_size=block_size, rng=rng
        )
        null[index] = statistic(permuted, permuted_valid)
    finite = null[np.isfinite(null)]
    if not np.isfinite(observed) or len(finite) < max(20, int(0.9 * len(null))):
        return {
            "valid": False,
            "observed_max_abs_correlation": float(observed),
            "null_q95": float("nan"),
            "n_null_finite": int(len(finite)),
            "n_eligible_groups": int(len(eligible_groups)),
        }
    threshold = float(np.quantile(finite, 0.95))
    return {
        "valid": bool(observed <= threshold),
        "observed_max_abs_correlation": float(observed),
        "null_q95": threshold,
        "n_null_finite": int(len(finite)),
        "n_eligible_groups": int(len(eligible_groups)),
    }


__all__ = [
    "StandardizedObserver",
    "blocked_innovation_validity",
    "coherent_block_permutation",
    "concatenate_feature_ladder",
    "fit_standardized_masked_observer",
    "masked_max_abs_feature_residual_correlation",
    "masked_rank_mse",
]
