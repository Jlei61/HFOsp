"""Core mathematics for Topic 5 event-innovation analysis v3.0.

One row is one complete interictal event or one disjoint event window.  This
module contains no within-event recurrence and intentionally keeps pairwise
objects event-local to avoid materialising ``anchor x contact x contact``
tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge


EPS = 1e-8


def _as_float_2d(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return array


def masked_window_rank_field(
    ranks: np.ndarray,
    participation: np.ndarray,
    event_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-contact mean rank and support for one event window."""

    rank = _as_float_2d(ranks, "ranks")
    mask = np.asarray(participation, dtype=bool)
    if mask.shape != rank.shape:
        raise ValueError("participation must match ranks")
    indices = np.asarray(event_indices, dtype=int)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(rank)):
        raise ValueError("event_indices are out of range")
    selected_rank = rank[indices]
    selected_mask = mask[indices] & np.isfinite(selected_rank)
    support = selected_mask.sum(axis=0).astype(int)
    total = np.where(selected_mask, selected_rank, 0.0).sum(axis=0)
    mean = np.full(rank.shape[1], np.nan, dtype=float)
    valid = support > 0
    mean[valid] = total[valid] / support[valid]
    return mean, support


def rank_field_windows(
    ranks: np.ndarray,
    participation: np.ndarray,
    windows: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Build compact ``window x contact`` fields without history tensors."""

    fields = []
    supports = []
    for indices in windows:
        field, support = masked_window_rank_field(ranks, participation, indices)
        fields.append(field)
        supports.append(support)
    if not fields:
        n_contacts = _as_float_2d(ranks, "ranks").shape[1]
        return (
            np.empty((0, n_contacts), dtype=float),
            np.empty((0, n_contacts), dtype=int),
        )
    return np.vstack(fields), np.vstack(supports)


def rolling_past_rank_fields(
    ranks: np.ndarray,
    participation: np.ndarray,
    sequences: Sequence[np.ndarray],
    *,
    start_offset: int,
    stop_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return past-only rank fields for every event without crossing a reset.

    For event position ``t``, the interval is ``[t-stop_offset, t-start_offset)``.
    Hence ``start_offset=0, stop_offset=20`` summarizes the immediately
    preceding 20 complete events and never includes the current event.
    """

    rank = _as_float_2d(ranks, "ranks")
    mask = np.asarray(participation, dtype=bool)
    if mask.shape != rank.shape:
        raise ValueError("participation must match ranks")
    near, far = int(start_offset), int(stop_offset)
    if near < 0 or far <= near:
        raise ValueError("past offsets must satisfy 0 <= start < stop")
    fields = np.full(rank.shape, np.nan, dtype=np.float32)
    supports = np.zeros(rank.shape, dtype=np.int32)
    assigned = np.zeros(len(rank), dtype=bool)
    for values in sequences:
        indices = np.asarray(values, dtype=np.int64)
        if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
            raise ValueError("sequence indices must be unique one-dimensional arrays")
        if np.any(indices < 0) or np.any(indices >= len(rank)):
            raise ValueError("sequence indices exceed the event arrays")
        if np.any(assigned[indices]):
            raise ValueError("continuity sequences overlap")
        assigned[indices] = True
        selected = rank[indices]
        valid = mask[indices] & np.isfinite(selected)
        count_prefix = np.vstack(
            [np.zeros((1, rank.shape[1]), dtype=np.int64), np.cumsum(valid, axis=0)]
        )
        total_prefix = np.vstack(
            [
                np.zeros((1, rank.shape[1]), dtype=np.float64),
                np.cumsum(np.where(valid, selected, 0.0), axis=0),
            ]
        )
        positions = np.arange(far, len(indices), dtype=np.int64)
        if not len(positions):
            continue
        left = positions - far
        right = positions - near
        count = count_prefix[right] - count_prefix[left]
        total = total_prefix[right] - total_prefix[left]
        mean = np.full(total.shape, np.nan, dtype=np.float32)
        np.divide(total, count, out=mean, where=count > 0)
        fields[indices[positions]] = mean
        supports[indices[positions]] = count.astype(np.int32)
    return fields, supports


@dataclass(frozen=True)
class RankStateBasis:
    backbone: np.ndarray
    loadings: np.ndarray
    singular_values: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.loadings.shape[1])

    def transform(self, fields: np.ndarray) -> np.ndarray:
        values = _as_float_2d(fields, "fields")
        if values.shape[1] != len(self.backbone):
            raise ValueError("field contact dimension mismatch")
        filled = np.where(np.isfinite(values), values, self.backbone[None, :])
        return (filled - self.backbone[None, :]) @ self.loadings

    def inverse(self, states: np.ndarray) -> np.ndarray:
        values = _as_float_2d(states, "states")
        if values.shape[1] != self.dimension:
            raise ValueError("state dimension mismatch")
        return self.backbone[None, :] + values @ self.loadings.T


def fit_rank_state_basis(
    fields: np.ndarray,
    dimension: int,
    *,
    sample_weight: np.ndarray | None = None,
) -> RankStateBasis:
    """Fit a train-only low-rank basis around a stable contact backbone.

    ``sample_weight`` permits continuity-unit-balanced dense training without
    throwing away events from long recordings.
    """

    values = _as_float_2d(fields, "fields")
    if len(values) < 2:
        raise ValueError("at least two fields are required")
    if not 1 <= int(dimension) <= min(values.shape):
        raise ValueError("invalid state dimension")
    weight = (
        np.ones(len(values), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if weight.shape != (len(values),) or np.any(~np.isfinite(weight)) or np.any(weight < 0):
        raise ValueError("sample_weight must be one finite non-negative row vector")
    if np.sum(weight) <= 0:
        raise ValueError("sample_weight must contain positive mass")
    valid = np.isfinite(values)
    denominator = (valid * weight[:, None]).sum(axis=0)
    numerator = np.where(valid, values, 0.0) * weight[:, None]
    backbone = numerator.sum(axis=0) / np.where(denominator > 0, denominator, np.nan)
    if np.any(~np.isfinite(backbone)):
        raise ValueError("every contact needs train-window support")
    filled = np.where(np.isfinite(values), values, backbone[None, :])
    normalized_weight = weight / np.mean(weight[weight > 0])
    weighted_centered = (filled - backbone[None, :]) * np.sqrt(normalized_weight[:, None])
    _, singular, right = np.linalg.svd(weighted_centered, full_matrices=False)
    loadings = right[: int(dimension)].T
    return RankStateBasis(
        backbone=backbone,
        loadings=loadings,
        singular_values=singular,
    )


def masked_rank_reconstruction_error(
    observed_fields: np.ndarray,
    reconstructed_fields: np.ndarray,
    support: np.ndarray,
) -> float:
    """Support-weighted error on observed contact-window entries only."""

    observed = _as_float_2d(observed_fields, "observed_fields")
    reconstructed = _as_float_2d(reconstructed_fields, "reconstructed_fields")
    weight = np.asarray(support, dtype=float)
    if reconstructed.shape != observed.shape or weight.shape != observed.shape:
        raise ValueError("reconstruction arrays must share one shape")
    valid = np.isfinite(observed) & np.isfinite(reconstructed) & (weight > 0)
    if not np.any(valid):
        raise ValueError("no supported reconstruction entries")
    squared = (observed[valid] - reconstructed[valid]) ** 2
    return float(np.average(squared, weights=weight[valid]))


def _finite_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float).ravel()
    b = np.asarray(right, dtype=float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 3 or np.std(a[valid]) <= EPS or np.std(b[valid]) <= EPS:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


@dataclass(frozen=True)
class RankFieldReliability:
    raw: float
    contact_residualized: float
    n_windows: int
    n_paired_entries: int


def split_window_rank_reliability(
    ranks: np.ndarray,
    participation: np.ndarray,
    windows: Sequence[np.ndarray],
    *,
    contact_backbone: np.ndarray,
) -> RankFieldReliability:
    """Odd/even event split reliability for raw and dynamic rank fields."""

    backbone = np.asarray(contact_backbone, dtype=float)
    if backbone.ndim != 1 or len(backbone) != _as_float_2d(ranks, "ranks").shape[1]:
        raise ValueError("contact_backbone has the wrong shape")
    left_fields = []
    right_fields = []
    for values in windows:
        indices = np.asarray(values, dtype=np.int64)
        if len(indices) < 2:
            continue
        left, _ = masked_window_rank_field(ranks, participation, indices[::2])
        right, _ = masked_window_rank_field(ranks, participation, indices[1::2])
        left_fields.append(left)
        right_fields.append(right)
    if not left_fields:
        return RankFieldReliability(float("nan"), float("nan"), 0, 0)
    left = np.vstack(left_fields)
    right = np.vstack(right_fields)
    paired = np.isfinite(left) & np.isfinite(right)
    raw = _finite_correlation(np.where(paired, left, np.nan), np.where(paired, right, np.nan))
    residual_left = left - backbone[None, :]
    residual_right = right - backbone[None, :]
    residual = _finite_correlation(
        np.where(paired, residual_left, np.nan),
        np.where(paired, residual_right, np.nan),
    )
    return RankFieldReliability(
        raw=raw,
        contact_residualized=residual,
        n_windows=len(left),
        n_paired_entries=int(np.sum(paired)),
    )


def split_window_precedence_reliability(
    ranks: np.ndarray,
    participation: np.ndarray,
    windows: Sequence[np.ndarray],
    *,
    contact_backbone: np.ndarray,
) -> RankFieldReliability:
    """Odd/even reliability of pairwise precedence with ties left unordered."""

    backbone = np.asarray(contact_backbone, dtype=float)
    n_contacts = _as_float_2d(ranks, "ranks").shape[1]
    if backbone.shape != (n_contacts,) or np.any(~np.isfinite(backbone)):
        raise ValueError("contact_backbone must be one finite contact vector")
    rank = _as_float_2d(ranks, "ranks")
    participation_mask = np.asarray(participation, dtype=bool)
    if participation_mask.shape != rank.shape:
        raise ValueError("participation must match ranks")
    upper = np.triu_indices(n_contacts, k=1)
    backbone_pair = precedence_probability(backbone)[upper]
    left_rows = []
    right_rows = []
    valid_rows = []
    for values in windows:
        indices = np.asarray(values, dtype=np.int64)
        if len(indices) < 2:
            continue
        pair_fields = []
        pair_supports = []
        for half in (indices[::2], indices[1::2]):
            selected = rank[half]
            selected_valid = participation_mask[half] & np.isfinite(selected)
            both = selected_valid[:, upper[0]] & selected_valid[:, upper[1]]
            left_rank = selected[:, upper[0]]
            right_rank = selected[:, upper[1]]
            outcome = np.where(left_rank < right_rank, 1.0, np.where(left_rank > right_rank, 0.0, 0.5))
            support = both.sum(axis=0)
            total = np.where(both, outcome, 0.0).sum(axis=0)
            field = np.full(len(upper[0]), np.nan, dtype=float)
            np.divide(total, support, out=field, where=support > 0)
            pair_fields.append(field)
            pair_supports.append(support)
        left_pair, right_pair = pair_fields
        pair_valid = (pair_supports[0] > 0) & (pair_supports[1] > 0)
        left_rows.append(left_pair)
        right_rows.append(right_pair)
        valid_rows.append(pair_valid)
    if not left_rows:
        return RankFieldReliability(float("nan"), float("nan"), 0, 0)
    left_pair = np.vstack(left_rows)
    right_pair = np.vstack(right_rows)
    paired = np.vstack(valid_rows)
    raw = _finite_correlation(
        np.where(paired, left_pair, np.nan), np.where(paired, right_pair, np.nan)
    )
    residual = _finite_correlation(
        np.where(paired, left_pair - backbone_pair[None, :], np.nan),
        np.where(paired, right_pair - backbone_pair[None, :], np.nan),
    )
    return RankFieldReliability(
        raw=raw,
        contact_residualized=residual,
        n_windows=len(left_pair),
        n_paired_entries=int(np.sum(paired)),
    )


@dataclass(frozen=True)
class MaskedRidgeFit:
    intercept: np.ndarray
    coefficient: np.ndarray
    fitted_contacts: np.ndarray
    alpha: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = _as_float_2d(features, "features")
        if values.shape[1] != self.coefficient.shape[1]:
            raise ValueError("observer feature dimension mismatch")
        return self.intercept[None, :] + values @ self.coefficient.T


def fit_masked_contact_ridge(
    features: np.ndarray,
    target_rank: np.ndarray,
    target_participation: np.ndarray,
    *,
    alpha: float,
    minimum_observations: int = 10,
    sample_weight: np.ndarray | None = None,
) -> MaskedRidgeFit:
    """Fit one shared-feature ridge per contact using participating events only."""

    design = _as_float_2d(features, "features")
    target = _as_float_2d(target_rank, "target_rank")
    participation = np.asarray(target_participation, dtype=bool)
    if len(design) != len(target) or participation.shape != target.shape:
        raise ValueError("masked observer arrays are not aligned")
    if np.any(~np.isfinite(design)):
        raise ValueError("observer features must be finite")
    intercept = np.zeros(target.shape[1], dtype=float)
    coefficient = np.zeros((target.shape[1], design.shape[1]), dtype=float)
    fitted = np.zeros(target.shape[1], dtype=bool)
    minimum = max(2, int(minimum_observations))
    row_weight = (
        np.ones(len(design), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if row_weight.shape != (len(design),) or np.any(~np.isfinite(row_weight)) or np.any(row_weight < 0):
        raise ValueError("sample_weight must be one finite non-negative row vector")
    for contact in range(target.shape[1]):
        valid = participation[:, contact] & np.isfinite(target[:, contact])
        if np.sum(valid) < minimum:
            value = float(np.mean(target[valid, contact])) if np.any(valid) else float("nan")
            intercept[contact] = 0.5 if not np.isfinite(value) else float(value)
            continue
        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit(
            design[valid],
            target[valid, contact],
            sample_weight=row_weight[valid],
        )
        intercept[contact] = float(model.intercept_)
        coefficient[contact] = np.asarray(model.coef_, dtype=float)
        fitted[contact] = True
    return MaskedRidgeFit(intercept, coefficient, fitted, float(alpha))


def precedence_probability(rank_field: np.ndarray) -> np.ndarray:
    """Map one contact rank field to an observable pairwise precedence field."""

    field = np.asarray(rank_field, dtype=float)
    if field.ndim != 1 or np.any(~np.isfinite(field)):
        raise ValueError("rank_field must be one finite vector")
    difference = field[None, :] - field[:, None]
    difference = np.clip(difference, -40.0, 40.0)
    probability = 1.0 / (1.0 + np.exp(-difference))
    np.fill_diagonal(probability, 0.5)
    return probability


@dataclass(frozen=True)
class RankInnovation:
    residual: np.ndarray
    valid: np.ndarray


def rank_innovation(
    observed_rank: np.ndarray,
    participation: np.ndarray,
    predicted_rank: np.ndarray,
    reliability: np.ndarray | None = None,
) -> RankInnovation:
    """Masked participating-contact rank innovation for one complete event."""

    observed = np.asarray(observed_rank, dtype=float)
    predicted = np.asarray(predicted_rank, dtype=float)
    valid = np.asarray(participation, dtype=bool)
    if observed.ndim != 1 or predicted.shape != observed.shape or valid.shape != observed.shape:
        raise ValueError("rank innovation inputs must share one contact vector")
    valid = valid & np.isfinite(observed) & np.isfinite(predicted)
    residual = np.zeros_like(observed, dtype=float)
    residual[valid] = observed[valid] - predicted[valid]
    if reliability is not None:
        weight = np.asarray(reliability, dtype=float)
        if weight.shape != observed.shape or np.any(weight < 0):
            raise ValueError("invalid reliability")
        residual[valid] *= np.sqrt(weight[valid])
    return RankInnovation(residual=residual, valid=valid)


@dataclass(frozen=True)
class PairwiseInnovation:
    source: np.ndarray
    target: np.ndarray
    residual: np.ndarray


def pairwise_precedence_innovation(
    observed_rank: np.ndarray,
    group_ids: np.ndarray,
    participation: np.ndarray,
    predicted_rank: np.ndarray,
) -> PairwiseInnovation:
    """Return valid non-tied pair residuals without a dense pair tensor."""

    observed = np.asarray(observed_rank, dtype=float)
    groups = np.asarray(group_ids)
    valid = np.asarray(participation, dtype=bool)
    predicted = np.asarray(predicted_rank, dtype=float)
    if not (
        observed.ndim == 1
        and groups.shape == observed.shape
        and valid.shape == observed.shape
        and predicted.shape == observed.shape
    ):
        raise ValueError("pairwise inputs must share one contact vector")
    valid = valid & np.isfinite(observed) & np.isfinite(predicted)
    contacts = np.flatnonzero(valid)
    source = []
    target = []
    residual = []
    for left_index, left in enumerate(contacts):
        for right in contacts[left_index + 1 :]:
            if groups[left] == groups[right]:
                continue
            probability = 1.0 / (
                1.0 + np.exp(-np.clip(predicted[right] - predicted[left], -40, 40))
            )
            outcome = float(observed[left] < observed[right])
            source.append(int(left))
            target.append(int(right))
            residual.append(outcome - probability)
    return PairwiseInnovation(
        source=np.asarray(source, dtype=int),
        target=np.asarray(target, dtype=int),
        residual=np.asarray(residual, dtype=float),
    )


@dataclass(frozen=True)
class LocalProjectionFit:
    intercept: np.ndarray
    autonomous: np.ndarray
    impulse: np.ndarray
    nuisance: np.ndarray
    alpha: float

    def predict(
        self,
        pre_state: np.ndarray,
        innovation: np.ndarray,
        nuisance: np.ndarray | None = None,
    ) -> np.ndarray:
        pre = _as_float_2d(pre_state, "pre_state")
        event = _as_float_2d(innovation, "innovation")
        if len(pre) != len(event):
            raise ValueError("pre_state and innovation row mismatch")
        estimate = self.intercept[None, :] + pre @ self.autonomous.T
        estimate += event @ self.impulse.T
        if self.nuisance.shape[1]:
            if nuisance is None:
                raise ValueError("nuisance is required")
            covariate = _as_float_2d(nuisance, "nuisance")
            estimate += covariate @ self.nuisance.T
        return estimate


def fit_local_projection(
    pre_state: np.ndarray,
    future_state: np.ndarray,
    innovation: np.ndarray,
    *,
    nuisance: np.ndarray | None = None,
    alpha: float = 1.0,
) -> LocalProjectionFit:
    """Fit a regularized local projection in low-dimensional state space."""

    pre = _as_float_2d(pre_state, "pre_state")
    future = _as_float_2d(future_state, "future_state")
    event = _as_float_2d(innovation, "innovation")
    if not (len(pre) == len(future) == len(event)):
        raise ValueError("local projection row mismatch")
    covariate = (
        np.empty((len(pre), 0), dtype=float)
        if nuisance is None
        else _as_float_2d(nuisance, "nuisance")
    )
    if len(covariate) != len(pre):
        raise ValueError("nuisance row mismatch")
    design = np.hstack([pre, event, covariate])
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(design, future)
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


def uniform_cumulative_innovation(
    innovations: np.ndarray,
    window: int,
) -> np.ndarray:
    """Causal rolling innovation sum using O(NK) prefix storage."""

    values = _as_float_2d(innovations, "innovations")
    width = int(window)
    if width < 1:
        raise ValueError("window must be positive")
    prefix = np.vstack(
        [np.zeros((1, values.shape[1]), dtype=float), np.cumsum(values, axis=0)]
    )
    output = np.full(values.shape, np.nan, dtype=float)
    if len(values) >= width:
        output[width - 1 :] = prefix[width:] - prefix[:-width]
    return output


def innovation_alignment(innovations: np.ndarray) -> float:
    """Return 0–1 alignment; opposing innovations cancel toward zero."""

    values = _as_float_2d(innovations, "innovations")
    denominator = float(np.linalg.norm(values, axis=1).sum())
    if denominator <= EPS:
        return 0.0
    return float(np.linalg.norm(values.sum(axis=0)) / denominator)


def observable_impulse(
    basis: RankStateBasis,
    impulse: np.ndarray,
) -> np.ndarray:
    """Map a state-space impulse to contact-rank coordinates."""

    matrix = _as_float_2d(impulse, "impulse")
    if matrix.shape[0] != basis.dimension:
        raise ValueError("impulse target dimension mismatch")
    return basis.loadings @ matrix


__all__ = [
    "LocalProjectionFit",
    "MaskedRidgeFit",
    "PairwiseInnovation",
    "RankInnovation",
    "RankFieldReliability",
    "RankStateBasis",
    "fit_masked_contact_ridge",
    "fit_local_projection",
    "fit_rank_state_basis",
    "innovation_alignment",
    "masked_rank_reconstruction_error",
    "masked_window_rank_field",
    "observable_impulse",
    "pairwise_precedence_innovation",
    "precedence_probability",
    "rank_field_windows",
    "rank_innovation",
    "rolling_past_rank_fields",
    "split_window_precedence_reliability",
    "split_window_rank_reliability",
    "uniform_cumulative_innovation",
]
