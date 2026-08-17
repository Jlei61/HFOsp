"""Observable response scoring helpers for Topic 5 v3.0.

All projections remain in the patient contact coordinate system.  Missing
contacts are solved by masked least squares rather than being silently treated
as zero innovation.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from src.topic5_event_innovation_v3_0 import (
    LocalProjectionFit,
    RankStateBasis,
    precedence_probability,
)


EPS = 1e-10


def fit_weighted_local_projection(
    pre_state: np.ndarray,
    future_state: np.ndarray,
    innovation: np.ndarray,
    *,
    nuisance: np.ndarray | None = None,
    alpha: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> LocalProjectionFit:
    """Fit the v3.0 local projection with continuity-unit row weights."""

    pre = np.asarray(pre_state, dtype=float)
    future = np.asarray(future_state, dtype=float)
    event = np.asarray(innovation, dtype=float)
    if pre.ndim != 2 or future.ndim != 2 or event.ndim != 2:
        raise ValueError("local-projection arrays must be 2D")
    if not (len(pre) == len(future) == len(event)):
        raise ValueError("local-projection rows are not aligned")
    covariate = (
        np.empty((len(pre), 0), dtype=float)
        if nuisance is None
        else np.asarray(nuisance, dtype=float)
    )
    if covariate.ndim != 2 or len(covariate) != len(pre):
        raise ValueError("nuisance rows are not aligned")
    weight = (
        np.ones(len(pre), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if weight.shape != (len(pre),) or np.any(~np.isfinite(weight)) or np.any(weight < 0) or np.sum(weight) <= 0:
        raise ValueError("sample_weight must contain finite non-negative mass")
    design = np.hstack([pre, event, covariate])
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(design, future, sample_weight=weight)
    coefficient = np.atleast_2d(np.asarray(model.coef_, dtype=float))
    pre_stop = pre.shape[1]
    event_stop = pre_stop + event.shape[1]
    return LocalProjectionFit(
        intercept=np.atleast_1d(np.asarray(model.intercept_, dtype=float)),
        autonomous=coefficient[:, :pre_stop],
        impulse=coefficient[:, pre_stop:event_stop],
        nuisance=coefficient[:, event_stop:],
        alpha=float(alpha),
    )


def masked_state_projection(
    fields: np.ndarray,
    valid: np.ndarray,
    basis: RankStateBasis,
    *,
    alpha: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Project masked contact fields onto a frozen rank basis."""

    values = np.asarray(fields, dtype=float)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("masked state arrays must share one 2D shape")
    if values.shape[1] != len(basis.backbone):
        raise ValueError("field/basis contact mismatch")
    ridge = float(alpha)
    if ridge < 0:
        raise ValueError("alpha must be non-negative")
    states = np.zeros((len(values), basis.dimension), dtype=float)
    estimable = np.zeros(len(values), dtype=bool)
    identity = np.eye(basis.dimension)
    for row in range(len(values)):
        selected = mask[row]
        if np.sum(selected) < max(2, basis.dimension):
            continue
        loading = basis.loadings[selected]
        target = values[row, selected] - basis.backbone[selected]
        states[row] = np.linalg.solve(
            loading.T @ loading + ridge * identity,
            loading.T @ target,
        )
        estimable[row] = True
    return states, estimable


def masked_innovation_projection(
    residual: np.ndarray,
    valid: np.ndarray,
    basis: RankStateBasis,
    *,
    alpha: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Project contact-rank residuals without subtracting the backbone."""

    values = np.asarray(residual, dtype=float)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("innovation arrays must share one 2D shape")
    ridge = float(alpha)
    if ridge <= 0:
        raise ValueError("masked innovation projection requires positive ridge")
    states = np.zeros((len(values), basis.dimension), dtype=float)
    estimable = np.zeros(len(values), dtype=bool)
    identity = np.eye(basis.dimension)
    for row in range(len(values)):
        selected = mask[row]
        # Two recruited contacts define at least one relative-rank direction.
        # The positive ridge supplies a minimum-norm estimate when C_valid < K.
        if np.sum(selected) < 2:
            continue
        loading = basis.loadings[selected]
        states[row] = np.linalg.solve(
            loading.T @ loading + ridge * identity,
            loading.T @ values[row, selected],
        )
        estimable[row] = True
    return states, estimable


def masked_rank_field_mse(
    predicted: np.ndarray,
    observed: np.ndarray,
    support: np.ndarray,
) -> float:
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    weight = np.asarray(support, dtype=float)
    if predicted.shape != observed.shape or weight.shape != observed.shape:
        raise ValueError("rank-field score arrays are not aligned")
    valid = np.isfinite(predicted) & np.isfinite(observed) & (weight > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.average((predicted[valid] - observed[valid]) ** 2, weights=weight[valid]))


def future_precedence_brier(
    predicted_fields: np.ndarray,
    future_windows: list[np.ndarray],
    ranks: np.ndarray,
    participation: np.ndarray,
    tie_groups: np.ndarray,
) -> float:
    """Score predicted rank fields on event-level co-participating precedence."""

    predicted = np.asarray(predicted_fields, dtype=float)
    rank = np.asarray(ranks, dtype=float)
    mask = np.asarray(participation, dtype=bool)
    groups = np.asarray(tie_groups)
    if rank.shape != mask.shape or groups.shape != rank.shape:
        raise ValueError("event precedence arrays are not aligned")
    if predicted.ndim != 2 or len(predicted) != len(future_windows) or predicted.shape[1] != rank.shape[1]:
        raise ValueError("predicted fields do not match future windows")
    squared = 0.0
    count = 0
    upper = np.triu(np.ones((rank.shape[1], rank.shape[1]), dtype=bool), k=1)
    for field, window in zip(predicted, future_windows):
        probability = precedence_probability(field)
        for event in np.asarray(window, dtype=np.int64):
            valid_contacts = mask[event] & np.isfinite(rank[event])
            pair_valid = (
                upper
                & valid_contacts[:, None]
                & valid_contacts[None, :]
                & (groups[event, :, None] != groups[event, None, :])
            )
            if not np.any(pair_valid):
                continue
            outcome = rank[event, :, None] < rank[event, None, :]
            difference = outcome.astype(float) - probability
            squared += float(np.sum(difference[pair_valid] ** 2))
            count += int(np.sum(pair_valid))
    return squared / count if count else float("nan")


def observable_propagation_gain(
    basis: RankStateBasis,
    observed_fields: np.ndarray,
    supports: np.ndarray,
    future_windows: list[np.ndarray],
    ranks: np.ndarray,
    participation: np.ndarray,
    tie_groups: np.ndarray,
    autonomous_state: np.ndarray,
    event_state: np.ndarray,
) -> dict[str, float]:
    autonomous_field = basis.inverse(autonomous_state)
    event_field = basis.inverse(event_state)
    auto_rank = masked_rank_field_mse(autonomous_field, observed_fields, supports)
    event_rank = masked_rank_field_mse(event_field, observed_fields, supports)
    auto_pair = future_precedence_brier(
        autonomous_field, future_windows, ranks, participation, tie_groups
    )
    event_pair = future_precedence_brier(
        event_field, future_windows, ranks, participation, tie_groups
    )
    return {
        "rank_autonomous_mse": auto_rank,
        "rank_event_mse": event_rank,
        "rank_gain": auto_rank - event_rank,
        "precedence_autonomous_brier": auto_pair,
        "precedence_event_brier": event_pair,
        "precedence_gain": auto_pair - event_pair,
        "propagation_gain": 0.5 * ((auto_rank - event_rank) + (auto_pair - event_pair)),
    }


__all__ = [
    "fit_weighted_local_projection",
    "future_precedence_brier",
    "masked_innovation_projection",
    "masked_rank_field_mse",
    "masked_state_projection",
    "observable_propagation_gain",
]
