"""Minimal event-history increment models for EERF v2.2 Phase 1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans

from src.topic5_event_indexed_evolving_rank_field import EventBlock, within_source_pairs


EPS = 1e-12


@dataclass(frozen=True)
class BlockObservation:
    """Observed block field plus its within-block chronological innovation."""

    block: EventBlock
    field: np.ndarray
    delta: np.ndarray
    covariates: np.ndarray


@dataclass(frozen=True)
class PairDataset:
    """Current block, chronological innovation, and next-block target."""

    current: np.ndarray
    delta: np.ndarray
    target: np.ndarray
    covariates: np.ndarray
    source: np.ndarray
    current_block_index: np.ndarray


@dataclass(frozen=True)
class RidgeModel:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    coefficient: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = (np.asarray(x, float) - self.x_mean) / self.x_scale
        return self.y_mean + values @ self.coefficient


@dataclass(frozen=True)
class PCATransform:
    mean: np.ndarray
    basis: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, float) - self.mean) @ self.basis

    def inverse(self, scores: np.ndarray) -> np.ndarray:
        return self.mean + np.asarray(scores, float) @ self.basis.T


def _estimate_field(
    local_rank: np.ndarray,
    participation: np.ndarray,
    indices: np.ndarray,
    *,
    rank_prior: np.ndarray,
    shrinkage_prior_events: float,
    beta_prior: float,
    contact_mask: np.ndarray | None,
) -> np.ndarray:
    rank = np.asarray(local_rank, float)[indices]
    part = np.asarray(participation, bool)[indices]
    valid = part & np.isfinite(rank)
    count = np.sum(valid, axis=0, dtype=float)
    total = np.sum(np.where(valid, rank, 0.0), axis=0)
    alpha = float(shrinkage_prior_events)
    field = (total + alpha * np.asarray(rank_prior, float)) / (count + alpha)
    beta = float(beta_prior)
    probability = (np.sum(part, axis=0) + beta) / (len(indices) + 2.0 * beta)
    if contact_mask is None:
        return np.concatenate([field, probability])
    mask = np.asarray(contact_mask, bool)
    return np.concatenate([field[mask], probability[mask]])


def estimate_block_observation(
    local_rank: np.ndarray,
    participation: np.ndarray,
    group_ids: np.ndarray,
    event_time: np.ndarray,
    block: EventBlock,
    *,
    rank_prior: np.ndarray,
    shrinkage_prior_events: float,
    beta_prior: float,
    maximum_within_source_order: int,
    contact_mask: np.ndarray | None = None,
    delta_order: Sequence[int] | None = None,
) -> BlockObservation:
    """Estimate a block field and late-minus-early event-history direction."""
    indices = np.asarray(block.indices, dtype=int)
    order = indices if delta_order is None else np.asarray(delta_order, dtype=int)
    if len(order) != len(indices) or set(map(int, order)) != set(map(int, indices)):
        raise ValueError("delta_order must permute exactly the block events")
    half = len(order) // 2
    if half < 2:
        raise ValueError("each block half must contain at least two events")
    if np.asarray(group_ids).shape != np.asarray(local_rank).shape:
        raise ValueError("group_ids and local_rank must align")
    kwargs = {
        "rank_prior": rank_prior,
        "shrinkage_prior_events": shrinkage_prior_events,
        "beta_prior": beta_prior,
        "contact_mask": contact_mask,
    }
    full = _estimate_field(local_rank, participation, indices, **kwargs)
    early = _estimate_field(local_rank, participation, order[:half], **kwargs)
    late = _estimate_field(local_rank, participation, order[-half:], **kwargs)
    times = np.sort(np.asarray(event_time, float)[indices])
    intervals = np.diff(times)
    duration = float(max(times[-1] - times[0], 0.0))
    median_iei = float(np.median(intervals)) if len(intervals) else 0.0
    chronology = float(block.within_source_order) / max(
        int(maximum_within_source_order), 1
    )
    return BlockObservation(
        block=block,
        field=np.asarray(full, float),
        delta=np.asarray(late - early, float),
        covariates=np.asarray(
            [chronology, np.log1p(duration), np.log1p(max(median_iei, 0.0))],
            float,
        ),
    )


def build_block_observations(
    local_rank: np.ndarray,
    participation: np.ndarray,
    group_ids: np.ndarray,
    event_time: np.ndarray,
    blocks: Sequence[EventBlock],
    *,
    rank_prior: np.ndarray,
    shrinkage_prior_events: float,
    beta_prior: float,
    contact_mask: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> list[BlockObservation]:
    """Build ordered observations, optionally shuffling order inside each block."""
    maxima: dict[int, int] = {}
    for block in blocks:
        source = int(block.source_block_id)
        maxima[source] = max(maxima.get(source, 0), int(block.within_source_order))
    output = []
    for block in blocks:
        delta_order = None if rng is None else rng.permutation(block.indices)
        output.append(
            estimate_block_observation(
                local_rank,
                participation,
                group_ids,
                event_time,
                block,
                rank_prior=rank_prior,
                shrinkage_prior_events=shrinkage_prior_events,
                beta_prior=beta_prior,
                maximum_within_source_order=maxima[int(block.source_block_id)],
                contact_mask=contact_mask,
                delta_order=delta_order,
            )
        )
    return output


def make_pair_dataset(observations: Sequence[BlockObservation]) -> PairDataset:
    """Create only true adjacent within-record current-to-future pairs."""
    blocks = [observation.block for observation in observations]
    pairs = within_source_pairs(blocks, adjacent_only=True)
    if not pairs:
        raise ValueError("no true adjacent block pairs")
    current = []
    delta = []
    target = []
    covariates = []
    source = []
    current_block_index = []
    for left, right, lag in pairs:
        if lag != 1:
            raise RuntimeError("non-adjacent pair entered Phase 1")
        current.append(observations[left].field)
        delta.append(observations[left].delta)
        target.append(observations[right].field)
        covariates.append(observations[left].covariates)
        source.append(observations[left].block.source_block_id)
        current_block_index.append(left)
    return PairDataset(
        current=np.asarray(current, float),
        delta=np.asarray(delta, float),
        target=np.asarray(target, float),
        covariates=np.asarray(covariates, float),
        source=np.asarray(source, int),
        current_block_index=np.asarray(current_block_index, int),
    )


def fit_pca_transform(fields: np.ndarray, dimension: int) -> PCATransform:
    values = np.asarray(fields, float)
    mean = np.mean(values, axis=0)
    _, _, right = np.linalg.svd(values - mean, full_matrices=False)
    k = min(int(dimension), len(right), values.shape[1])
    if k < 1:
        raise ValueError("PCA dimension must be positive")
    return PCATransform(mean=mean, basis=right[:k].T)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> RidgeModel:
    features = np.asarray(x, float)
    targets = np.asarray(y, float)
    x_mean = np.mean(features, axis=0)
    x_scale = np.std(features, axis=0)
    x_scale = np.where(x_scale > EPS, x_scale, 1.0)
    y_mean = np.mean(targets, axis=0)
    standardized = (features - x_mean) / x_scale
    gram = standardized.T @ standardized
    coefficient = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0]),
        standardized.T @ (targets - y_mean),
    )
    return RidgeModel(x_mean, x_scale, y_mean, coefficient)


def mean_squared_error(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((np.asarray(target, float) - np.asarray(prediction, float)) ** 2))


def select_ridge_alpha(
    x: np.ndarray,
    y: np.ndarray,
    alpha_grid: Sequence[float],
    validation_fraction: float,
) -> float:
    values = np.asarray(x, float)
    targets = np.asarray(y, float)
    cut = int(np.floor((1.0 - float(validation_fraction)) * len(values)))
    cut = min(max(cut, 3), len(values) - 2)
    scores = {}
    for alpha in map(float, alpha_grid):
        model = fit_ridge(values[:cut], targets[:cut], alpha)
        scores[alpha] = mean_squared_error(targets[cut:], model.predict(values[cut:]))
    return min(scores, key=lambda alpha: (scores[alpha], alpha))


def _fit_switching(
    current: np.ndarray,
    target: np.ndarray,
    n_states: int,
    seed: int,
) -> tuple[KMeans, np.ndarray, np.ndarray]:
    model = KMeans(n_clusters=int(n_states), random_state=int(seed), n_init=20)
    current_label = model.fit_predict(current)
    target_label = model.predict(target)
    counts = np.ones((int(n_states), int(n_states)), dtype=float)
    for left, right in zip(current_label, target_label):
        counts[int(left), int(right)] += 1.0
    transition = counts / np.sum(counts, axis=1, keepdims=True)
    return model, transition, model.cluster_centers_


def _predict_switching(
    model: KMeans,
    transition: np.ndarray,
    centers: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    labels = model.predict(current)
    return np.asarray([transition[int(label)] @ centers for label in labels], float)


def select_switching_states(
    current: np.ndarray,
    target: np.ndarray,
    state_grid: Sequence[int],
    validation_fraction: float,
    seed: int,
) -> int:
    cut = int(np.floor((1.0 - float(validation_fraction)) * len(current)))
    cut = min(max(cut, 5), len(current) - 2)
    candidates = [int(k) for k in state_grid if int(k) < cut]
    if not candidates:
        return 2
    scores = {}
    for k in candidates:
        model, transition, centers = _fit_switching(
            current[:cut], target[:cut], k, seed + k
        )
        prediction = _predict_switching(model, transition, centers, current[cut:])
        scores[k] = mean_squared_error(target[cut:], prediction)
    return min(scores, key=lambda k: (scores[k], k))


def _model_prediction(
    train_x: np.ndarray,
    train_target: np.ndarray,
    test_x: np.ndarray,
    alpha_grid: Sequence[float],
    validation_fraction: float,
) -> tuple[np.ndarray, float]:
    alpha = select_ridge_alpha(
        train_x, train_target, alpha_grid, validation_fraction
    )
    model = fit_ridge(train_x, train_target, alpha)
    return model.predict(test_x), alpha


def evaluate_model_ladder(
    train: PairDataset,
    test: PairDataset,
    *,
    dimension: int,
    alpha_grid: Sequence[float],
    switching_state_grid: Sequence[int],
    validation_fraction: float,
    seed: int,
) -> dict[str, object]:
    """Fit all matched models and score the same future block fields."""
    transform = fit_pca_transform(
        np.concatenate([train.current, train.target], axis=0), dimension
    )
    train_current = transform.transform(train.current)
    train_delta = np.asarray(train.delta, float) @ transform.basis
    train_target = transform.transform(train.target)
    test_current = transform.transform(test.current)
    test_delta = np.asarray(test.delta, float) @ transform.basis
    test_target = transform.transform(test.target)

    predictions: dict[str, np.ndarray] = {
        "fixed": np.repeat(
            np.mean(train.target, axis=0, keepdims=True), len(test.target), axis=0
        ),
        "persistence": np.asarray(test.current, float),
    }
    selected: dict[str, object] = {"dimension": transform.basis.shape[1]}
    autonomous_score, selected["autonomous_alpha"] = _model_prediction(
        train_current,
        train_target,
        test_current,
        alpha_grid,
        validation_fraction,
    )
    predictions["autonomous"] = transform.inverse(autonomous_score)
    time_score, selected["time_iei_alpha"] = _model_prediction(
        np.concatenate([train_current, train.covariates], axis=1),
        train_target,
        np.concatenate([test_current, test.covariates], axis=1),
        alpha_grid,
        validation_fraction,
    )
    predictions["time_iei"] = transform.inverse(time_score)
    event_score, selected["event_history_alpha"] = _model_prediction(
        np.concatenate([train_current, train_delta], axis=1),
        train_target,
        np.concatenate([test_current, test_delta], axis=1),
        alpha_grid,
        validation_fraction,
    )
    predictions["event_history"] = transform.inverse(event_score)
    event_time_score, selected["event_history_iei_alpha"] = _model_prediction(
        np.concatenate([train_current, train_delta, train.covariates], axis=1),
        train_target,
        np.concatenate([test_current, test_delta, test.covariates], axis=1),
        alpha_grid,
        validation_fraction,
    )
    predictions["event_history_iei"] = transform.inverse(event_time_score)

    n_states = select_switching_states(
        train_current,
        train_target,
        switching_state_grid,
        validation_fraction,
        seed,
    )
    selected["switching_states"] = n_states
    switching, transition, centers = _fit_switching(
        train_current, train_target, n_states, seed + 100
    )
    predictions["switching"] = transform.inverse(
        _predict_switching(switching, transition, centers, test_current)
    )
    mse = {
        name: mean_squared_error(test.target, prediction)
        for name, prediction in predictions.items()
    }
    baseline_names = ["persistence", "autonomous", "switching", "time_iei"]
    best_baseline = min(baseline_names, key=lambda name: mse[name])
    increment = float(mse[best_baseline] - mse["event_history"])
    return {
        "mse": mse,
        "selected": selected,
        "best_baseline": best_baseline,
        "event_increment_over_best": increment,
        "event_relative_gain_over_best": increment / max(mse[best_baseline], EPS),
        "n_train_pairs": len(train.target),
        "n_test_pairs": len(test.target),
    }


def evaluate_event_history_mse(
    train: PairDataset,
    test: PairDataset,
    *,
    dimension: int,
    alpha_grid: Sequence[float],
    validation_fraction: float,
) -> float:
    """Score only the event-history model for matched permutation nulls."""
    transform = fit_pca_transform(
        np.concatenate([train.current, train.target], axis=0), dimension
    )
    train_current = transform.transform(train.current)
    train_delta = np.asarray(train.delta, float) @ transform.basis
    train_target = transform.transform(train.target)
    test_current = transform.transform(test.current)
    test_delta = np.asarray(test.delta, float) @ transform.basis
    score, _ = _model_prediction(
        np.concatenate([train_current, train_delta], axis=1),
        train_target,
        np.concatenate([test_current, test_delta], axis=1),
        alpha_grid,
        validation_fraction,
    )
    return mean_squared_error(test.target, transform.inverse(score))


def permute_delta_within_source(
    values: np.ndarray,
    source: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.asarray(values, float).copy()
    for source_id in np.unique(source):
        indices = np.flatnonzero(source == source_id)
        output[indices] = values[rng.permutation(indices)]
    return output


def circular_shift_delta_within_source(
    values: np.ndarray,
    source: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.asarray(values, float).copy()
    for source_id in np.unique(source):
        indices = np.flatnonzero(source == source_id)
        if len(indices) < 2:
            continue
        shift = int(rng.integers(1, len(indices)))
        output[indices] = np.roll(values[indices], shift=shift, axis=0)
    return output


def replace_delta(dataset: PairDataset, delta: np.ndarray) -> PairDataset:
    return PairDataset(
        current=dataset.current,
        delta=np.asarray(delta, float),
        target=dataset.target,
        covariates=dataset.covariates,
        source=dataset.source,
        current_block_index=dataset.current_block_index,
    )
