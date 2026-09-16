"""V2.4 event-history forecasting with coherent chronology controls.

The stable repertoire is fitted elsewhere on training events only.  This module
keeps one complete event as one sequence step and explicitly separates simple
recency filters from a low-dimensional leaky history state.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from src.topic5_stable_repertoire_event_rnn import (
    EPS,
    StableTemplateEncoder,
    project_descriptor,
    repertoire_descriptor,
)


@dataclass(frozen=True)
class EventHistoryDataset:
    histories: np.ndarray
    recent_descriptors: np.ndarray
    history_descriptors: np.ndarray
    targets: np.ndarray
    last_mode: np.ndarray
    source_ids: np.ndarray
    history_start: np.ndarray
    history_stop: np.ndarray
    target_start: np.ndarray
    target_stop: np.ndarray
    history_event_indices: np.ndarray
    target_event_indices: np.ndarray
    history_positions: np.ndarray
    target_positions: np.ndarray
    history_event_times: np.ndarray
    target_event_times: np.ndarray
    source_lengths: np.ndarray
    origin_rows: np.ndarray
    donor_rows: np.ndarray
    time_features: np.ndarray
    surrogate_kind: str = "true_chronology"

    def __len__(self) -> int:
        return int(len(self.targets))

    def take(self, rows: Sequence[int]) -> "EventHistoryDataset":
        selected = np.asarray(rows, int)
        fields = {
            name: getattr(self, name)[selected]
            for name in (
                "histories",
                "recent_descriptors",
                "history_descriptors",
                "targets",
                "last_mode",
                "source_ids",
                "history_start",
                "history_stop",
                "target_start",
                "target_stop",
                "history_event_indices",
                "target_event_indices",
                "history_positions",
                "target_positions",
                "history_event_times",
                "target_event_times",
                "source_lengths",
                "origin_rows",
                "donor_rows",
                "time_features",
            )
        }
        return EventHistoryDataset(**fields, surrogate_kind=self.surrogate_kind)


@dataclass(frozen=True)
class FamilyScales:
    occupancy: float
    rank: float
    participation: float


@dataclass(frozen=True)
class V24Score:
    propagation: float
    recruitment: float
    repertoire: float
    occupancy: float
    rank: float
    participation: float
    raw_occupancy: float
    raw_rank: float
    raw_participation: float


@dataclass(frozen=True)
class FeatureRidgeModel:
    feature_name: str
    decay: float | None
    alpha: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    ridge: Ridge
    n_modes: int
    rank_prior: np.ndarray
    validation_score: V24Score

    def predict(self, dataset: EventHistoryDataset) -> np.ndarray:
        values = feature_matrix(
            dataset,
            self.feature_name,
            self.rank_prior,
            self.n_modes,
            decay=self.decay,
        )
        standardized = (values - self.feature_mean) / self.feature_scale
        return project_descriptor(self.ridge.predict(standardized), self.n_modes)


@dataclass(frozen=True)
class LowDimensionalStateModel:
    base_model: FeatureRidgeModel
    pca: PCA
    decay: float
    dimension: int
    alpha: float
    state_mean: np.ndarray
    state_scale: np.ndarray
    correction: Ridge
    n_modes: int
    validation_score: V24Score

    def predict(self, dataset: EventHistoryDataset) -> np.ndarray:
        base = self.base_model.predict(dataset)
        state = pca_leaky_state(dataset.histories, self.pca, self.decay)
        state = (state - self.state_mean) / self.state_scale
        return project_descriptor(base + self.correction.predict(state), self.n_modes)


def chronological_sequences(
    source_ids: np.ndarray,
    event_time: np.ndarray,
    eligible_indices: Sequence[int],
) -> dict[object, np.ndarray]:
    source = np.asarray(source_ids)
    times = np.asarray(event_time, float)
    eligible = np.asarray(eligible_indices, int)
    output: dict[object, np.ndarray] = {}
    for source_id in np.unique(source[eligible]):
        sequence = eligible[source[eligible] == source_id]
        output[source_id.item() if hasattr(source_id, "item") else source_id] = sequence[
            np.argsort(times[sequence], kind="mergesort")
        ]
    return output


def source_coherent_block_shuffle(
    sequences: Mapping[object, np.ndarray],
    *,
    block_size: int,
    seed: int,
) -> tuple[dict[object, np.ndarray], dict[str, list[int]]]:
    """Permute contiguous blocks once per source, preserving within-block order."""
    if int(block_size) < 2:
        raise ValueError("block_size must be at least two")
    rng = np.random.default_rng(int(seed))
    output: dict[object, np.ndarray] = {}
    metadata: dict[str, list[int]] = {}
    for source_id in sorted(sequences, key=str):
        sequence = np.asarray(sequences[source_id], int)
        blocks = [sequence[start : start + int(block_size)] for start in range(0, len(sequence), int(block_size))]
        if len(blocks) < 2:
            output[source_id] = sequence.copy()
            metadata[str(source_id)] = [0]
            continue
        permutation = rng.permutation(len(blocks))
        if np.array_equal(permutation, np.arange(len(blocks))):
            permutation = np.roll(permutation, 1)
        output[source_id] = np.concatenate([blocks[index] for index in permutation])
        metadata[str(source_id)] = permutation.astype(int).tolist()
    return output, metadata


def _time_features(history_times: np.ndarray, progress: float) -> np.ndarray:
    ordered = np.sort(np.asarray(history_times, float))
    gaps = np.diff(ordered)
    positive = gaps[gaps > 0]
    duration = float(ordered[-1] - ordered[0]) if len(ordered) > 1 else 0.0
    median_iei = float(np.median(positive)) if len(positive) else 0.0
    event_rate = float((len(ordered) - 1) / duration) if duration > 0 else 0.0
    return np.asarray(
        [np.log1p(duration), np.log1p(median_iei), np.log1p(event_rate), progress],
        float,
    )


def build_event_history_dataset(
    tokens: np.ndarray,
    modes: np.ndarray,
    local_rank: np.ndarray,
    participation: np.ndarray,
    event_time: np.ndarray,
    encoder: StableTemplateEncoder,
    sequences: Mapping[object, np.ndarray],
    *,
    history_length: int,
    horizon: int,
    stride: int | None = None,
    surrogate_kind: str = "true_chronology",
) -> EventHistoryDataset:
    """Build all overlapping histories from one coherent sequence per source."""
    values = np.asarray(tokens, float)
    times = np.asarray(event_time, float)
    length = int(history_length)
    future = int(horizon)
    step = future if stride is None else int(stride)
    if length < future or future < 2:
        raise ValueError("history_length >= horizon >= 2 is required")
    records = []
    row = 0
    for source_id in sorted(sequences, key=str):
        sequence = np.asarray(sequences[source_id], int)
        for anchor in range(length, len(sequence) - future + 1, step):
            history_positions = np.arange(anchor - length, anchor, dtype=int)
            target_positions = np.arange(anchor, anchor + future, dtype=int)
            history = sequence[history_positions]
            target = sequence[target_positions]
            recent = history[-future:]
            records.append(
                {
                    "histories": values[history],
                    "recent_descriptors": repertoire_descriptor(
                        recent, modes, local_rank, participation, encoder
                    ),
                    "history_descriptors": repertoire_descriptor(
                        history, modes, local_rank, participation, encoder
                    ),
                    "targets": repertoire_descriptor(
                        target, modes, local_rank, participation, encoder
                    ),
                    "last_mode": int(modes[history[-1]]),
                    "source_ids": source_id,
                    "history_start": int(history_positions[0]),
                    "history_stop": int(history_positions[-1]) + 1,
                    "target_start": int(target_positions[0]),
                    "target_stop": int(target_positions[-1]) + 1,
                    "history_event_indices": history,
                    "target_event_indices": target,
                    "history_positions": history_positions,
                    "target_positions": target_positions,
                    "history_event_times": times[history],
                    "target_event_times": times[target],
                    "source_lengths": len(sequence),
                    "origin_rows": row,
                    "donor_rows": row,
                    "time_features": _time_features(
                        times[history], float(anchor / max(len(sequence), 1))
                    ),
                }
            )
            row += 1
    if not records:
        raise ValueError("no within-source future-window samples")
    keys = [key for key in records[0] if key not in {"source_ids"}]
    fields = {key: np.stack([record[key] for record in records]) for key in keys}
    fields["source_ids"] = np.asarray([record["source_ids"] for record in records])
    return EventHistoryDataset(**fields, surrogate_kind=str(surrogate_kind))


def safe_circular_target_pairing(
    dataset: EventHistoryDataset,
    *,
    shift_fraction: float,
    horizon: int,
) -> tuple[EventHistoryDataset, dict[str, int]]:
    """Circularly pair complete target windows and synchronize all provenance."""
    if not 0.0 < float(shift_fraction) < 1.0:
        raise ValueError("shift_fraction must be between zero and one")
    kept: list[int] = []
    donors: list[int] = []
    shifts: dict[str, int] = {}
    for source_id in np.unique(dataset.source_ids):
        rows = np.flatnonzero(dataset.source_ids == source_id)
        rows = rows[np.argsort(dataset.target_start[rows], kind="mergesort")]
        if len(rows) < 2:
            continue
        shift = max(1, int(round(float(shift_fraction) * len(rows)))) % len(rows)
        if shift == 0:
            shift = 1
        shifts[str(source_id)] = int(shift)
        donor_rows = np.roll(rows, -shift)
        for recipient, donor in zip(rows, donor_rows):
            history = dataset.history_event_indices[recipient]
            target = dataset.target_event_indices[donor]
            if np.intersect1d(history, target).size:
                continue
            h0, h1 = int(dataset.history_start[recipient]), int(dataset.history_stop[recipient])
            t0, t1 = int(dataset.target_start[donor]), int(dataset.target_stop[donor])
            if t1 <= h0:
                gap = h0 - t1
            elif t0 >= h1:
                gap = t0 - h1
            else:
                gap = -1
            if gap < int(horizon):
                continue
            kept.append(int(recipient))
            donors.append(int(donor))
    if not kept:
        raise ValueError("no safe circular pair remains")
    recipient = np.asarray(kept, int)
    donor = np.asarray(donors, int)
    paired = dataset.take(recipient)
    paired = replace(
        paired,
        targets=dataset.targets[donor].copy(),
        target_start=dataset.target_start[donor].copy(),
        target_stop=dataset.target_stop[donor].copy(),
        target_event_indices=dataset.target_event_indices[donor].copy(),
        target_positions=dataset.target_positions[donor].copy(),
        target_event_times=dataset.target_event_times[donor].copy(),
        donor_rows=dataset.origin_rows[donor].copy(),
        time_features=np.stack(
            [
                _time_features(
                    paired.history_event_times[index],
                    float(dataset.target_start[donor[index]] / max(dataset.source_lengths[donor[index]], 1)),
                )
                for index in range(len(paired))
            ]
        ),
        surrogate_kind=f"safe_circular_{float(shift_fraction):.3f}",
    )
    return paired, shifts


def verify_event_history_contract(
    dataset: EventHistoryDataset,
    *,
    raw_source_ids: np.ndarray,
    raw_event_time: np.ndarray,
    eligible_indices: Sequence[int],
    horizon: int,
    require_future: bool,
) -> dict[str, bool]:
    source = np.asarray(raw_source_ids)
    times = np.asarray(raw_event_time, float)
    eligible = set(np.asarray(eligible_indices, int).tolist())
    all_indices = np.concatenate(
        [dataset.history_event_indices.ravel(), dataset.target_event_indices.ravel()]
    )
    disjoint = all(
        np.intersect1d(history, target).size == 0
        for history, target in zip(dataset.history_event_indices, dataset.target_event_indices)
    )
    separated = []
    for h0, h1, t0, t1 in zip(
        dataset.history_start, dataset.history_stop, dataset.target_start, dataset.target_stop
    ):
        if t1 <= h0:
            separated.append(h0 - t1 >= int(horizon))
        elif t0 >= h1:
            separated.append(t0 - h1 >= (0 if require_future else int(horizon)))
        else:
            separated.append(False)
    position_map_consistent = True
    for source_id in np.unique(dataset.source_ids):
        mapping: dict[int, int] = {}
        rows = np.flatnonzero(dataset.source_ids == source_id)
        for row in rows:
            for event, position in zip(
                np.concatenate([dataset.history_event_indices[row], dataset.target_event_indices[row]]),
                np.concatenate([dataset.history_positions[row], dataset.target_positions[row]]),
            ):
                if int(event) in mapping and mapping[int(event)] != int(position):
                    position_map_consistent = False
                mapping[int(event)] = int(position)
    nonoverlap = True
    for source_id in np.unique(dataset.source_ids):
        rows = np.flatnonzero(dataset.source_ids == source_id)
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                if np.intersect1d(
                    dataset.target_event_indices[left], dataset.target_event_indices[right]
                ).size:
                    nonoverlap = False
    return {
        "all_indices_eligible_train80": bool(all(int(value) in eligible for value in all_indices)),
        "history_target_disjoint": bool(disjoint),
        "history_target_separation_valid": bool(all(separated)),
        "history_strictly_precedes_target_when_required": bool(
            (not require_future)
            or np.all(dataset.history_stop <= dataset.target_start)
        ),
        "history_target_same_source": bool(
            np.all(source[dataset.history_event_indices] == dataset.source_ids[:, None])
            and np.all(source[dataset.target_event_indices] == dataset.source_ids[:, None])
        ),
        "target_windows_expected_size": bool(
            dataset.target_event_indices.shape[1] == int(horizon)
        ),
        "formal_targets_nonoverlapping": bool(nonoverlap),
        "raw_times_match_indices": bool(
            np.allclose(times[dataset.history_event_indices], dataset.history_event_times)
            and np.allclose(times[dataset.target_event_indices], dataset.target_event_times)
        ),
        "position_metadata_matches_bounds": bool(
            np.all(dataset.history_positions[:, 0] == dataset.history_start)
            and np.all(dataset.history_positions[:, -1] + 1 == dataset.history_stop)
            and np.all(dataset.target_positions[:, 0] == dataset.target_start)
            and np.all(dataset.target_positions[:, -1] + 1 == dataset.target_stop)
        ),
        "source_sequence_position_is_coherent": bool(position_map_consistent),
        "circular_donor_differs_when_applicable": bool(
            (not dataset.surrogate_kind.startswith("safe_circular"))
            or np.all(dataset.origin_rows != dataset.donor_rows)
        ),
    }


def verify_target_values(
    dataset: EventHistoryDataset,
    modes: np.ndarray,
    local_rank: np.ndarray,
    participation: np.ndarray,
    encoder: StableTemplateEncoder,
) -> bool:
    rebuilt = np.stack(
        [
            repertoire_descriptor(indices, modes, local_rank, participation, encoder)
            for indices in dataset.target_event_indices
        ]
    )
    return bool(np.allclose(rebuilt, dataset.targets, atol=1e-12, rtol=1e-12))


def descriptor_from_token_groups(
    groups: np.ndarray,
    rank_prior: np.ndarray,
    n_modes: int,
    *,
    rank_prior_events: float = 2.0,
    beta_prior: float = 1.0,
) -> np.ndarray:
    values = np.asarray(groups, float)
    if values.ndim != 3:
        raise ValueError("groups must be sample x event x token")
    n_contacts = len(np.asarray(rank_prior))
    ranks = values[:, :, :n_contacts]
    part = values[:, :, n_contacts : 2 * n_contacts]
    modes = values[:, :, -int(n_modes) :]
    occupancy = np.mean(modes, axis=1)
    count = np.sum(part, axis=1)
    total = np.sum(ranks * part, axis=1)
    alpha = float(rank_prior_events)
    rank = (total + alpha * np.asarray(rank_prior)[None, :]) / (count + alpha)
    beta = float(beta_prior)
    participation = (count + beta) / (values.shape[1] + 2.0 * beta)
    return np.concatenate([occupancy, rank, participation], axis=1)


def _weighted_mean(values: np.ndarray, decay: float) -> np.ndarray:
    length = values.shape[1]
    weights = float(decay) ** np.arange(length - 1, -1, -1, dtype=float)
    weights /= max(float(np.sum(weights)), EPS)
    return np.sum(values * weights[None, :, None], axis=1)


def random_equal_count_features(
    dataset: EventHistoryDataset,
    rank_prior: np.ndarray,
    n_modes: int,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    groups = np.stack(
        [history[np.sort(rng.choice(len(history), int(count), replace=False))] for history in dataset.histories]
    )
    return descriptor_from_token_groups(groups, rank_prior, n_modes)


def feature_matrix(
    dataset: EventHistoryDataset,
    feature_name: str,
    rank_prior: np.ndarray,
    n_modes: int,
    *,
    decay: float | None = None,
) -> np.ndarray:
    horizon = dataset.target_event_indices.shape[1]
    if feature_name == "recent_h":
        return np.asarray(dataset.recent_descriptors, float)
    if feature_name == "unordered_l":
        return np.asarray(dataset.history_descriptors, float)
    if feature_name == "first_h":
        return descriptor_from_token_groups(dataset.histories[:, :horizon], rank_prior, n_modes)
    if feature_name == "full_token_ewma":
        if decay is None:
            raise ValueError("full_token_ewma requires decay")
        return _weighted_mean(dataset.histories, decay)
    if feature_name == "descriptor_ewma":
        if decay is None:
            raise ValueError("descriptor_ewma requires decay")
        history = dataset.histories
        single = descriptor_from_token_groups(
            history.reshape(-1, 1, history.shape[-1]), rank_prior, n_modes
        ).reshape(len(history), history.shape[1], -1)
        return _weighted_mean(single, decay)
    if feature_name == "binned_lag":
        pieces = np.array_split(np.arange(dataset.histories.shape[1]), 4)
        return np.concatenate(
            [
                descriptor_from_token_groups(dataset.histories[:, indices], rank_prior, n_modes)
                for indices in pieces
            ],
            axis=1,
        )
    if feature_name == "time_nuisance":
        return np.asarray(dataset.time_features, float)
    raise ValueError(f"unknown feature_name: {feature_name}")


def family_scales_from_train(
    targets: np.ndarray, *, n_modes: int, n_contacts: int
) -> FamilyScales:
    values = np.asarray(targets, float)
    sections = (
        values[:, :n_modes],
        values[:, n_modes : n_modes + n_contacts],
        values[:, n_modes + n_contacts :],
    )
    scales = [float(np.mean(np.var(section, axis=0))) for section in sections]
    scales = [max(value, EPS) for value in scales]
    return FamilyScales(*scales)


def score_v24(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    n_modes: int,
    n_contacts: int,
    scales: FamilyScales,
) -> V24Score:
    truth = np.asarray(target, float)
    estimate = np.asarray(prediction, float)
    if truth.shape != estimate.shape:
        raise ValueError("target and prediction shapes differ")
    raw_occ = float(np.mean((truth[:, :n_modes] - estimate[:, :n_modes]) ** 2))
    raw_rank = float(
        np.mean(
            (
                truth[:, n_modes : n_modes + n_contacts]
                - estimate[:, n_modes : n_modes + n_contacts]
            )
            ** 2
        )
    )
    raw_part = float(
        np.mean((truth[:, n_modes + n_contacts :] - estimate[:, n_modes + n_contacts :]) ** 2)
    )
    occ = raw_occ / scales.occupancy
    rank = raw_rank / scales.rank
    part = raw_part / scales.participation
    return V24Score(
        propagation=float((occ + rank) / 2.0),
        recruitment=float(part),
        repertoire=float((occ + rank + part) / 3.0),
        occupancy=float(occ),
        rank=float(rank),
        participation=float(part),
        raw_occupancy=raw_occ,
        raw_rank=raw_rank,
        raw_participation=raw_part,
    )


def _standardize(train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(scale > EPS, scale, 1.0)
    return (train - mean) / scale, mean, scale


def fit_feature_ridge(
    train: EventHistoryDataset,
    validation: EventHistoryDataset,
    *,
    feature_name: str,
    rank_prior: np.ndarray,
    decay: float | None,
    alpha_grid: Iterable[float],
    n_modes: int,
    n_contacts: int,
    scales: FamilyScales,
) -> FeatureRidgeModel:
    train_x = feature_matrix(train, feature_name, rank_prior, n_modes, decay=decay)
    validation_x = feature_matrix(validation, feature_name, rank_prior, n_modes, decay=decay)
    standardized, mean, scale = _standardize(train_x)
    best = None
    for alpha in map(float, alpha_grid):
        ridge = Ridge(alpha=alpha).fit(standardized, train.targets)
        prediction = project_descriptor(ridge.predict((validation_x - mean) / scale), n_modes)
        score = score_v24(
            validation.targets,
            prediction,
            n_modes=n_modes,
            n_contacts=n_contacts,
            scales=scales,
        )
        candidate = (score.propagation, alpha, ridge, score)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return FeatureRidgeModel(
        feature_name=feature_name,
        decay=decay,
        alpha=best[1],
        feature_mean=mean,
        feature_scale=scale,
        ridge=best[2],
        n_modes=int(n_modes),
        rank_prior=np.asarray(rank_prior, float),
        validation_score=best[3],
    )


def refit_feature_ridge(
    train: EventHistoryDataset,
    validation: EventHistoryDataset,
    selected: FeatureRidgeModel,
) -> FeatureRidgeModel:
    train_x = feature_matrix(
        train, selected.feature_name, selected.rank_prior, selected.n_modes, decay=selected.decay
    )
    validation_x = feature_matrix(
        validation,
        selected.feature_name,
        selected.rank_prior,
        selected.n_modes,
        decay=selected.decay,
    )
    values = np.concatenate([train_x, validation_x])
    targets = np.concatenate([train.targets, validation.targets])
    standardized, mean, scale = _standardize(values)
    ridge = Ridge(alpha=float(selected.alpha)).fit(standardized, targets)
    return replace(selected, feature_mean=mean, feature_scale=scale, ridge=ridge)


def fit_array_ridge_predict(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    *,
    alpha_grid: Iterable[float],
    n_modes: int,
    n_contacts: int,
    scales: FamilyScales,
) -> tuple[np.ndarray, float, V24Score]:
    """Select ridge strength on validation, refit train+validation, predict test."""
    standardized, mean, scale = _standardize(np.asarray(train_x, float))
    validation_values = (np.asarray(validation_x, float) - mean) / scale
    best = None
    for alpha in map(float, alpha_grid):
        ridge = Ridge(alpha=alpha).fit(standardized, train_y)
        prediction = project_descriptor(ridge.predict(validation_values), n_modes)
        score = score_v24(
            validation_y,
            prediction,
            n_modes=n_modes,
            n_contacts=n_contacts,
            scales=scales,
        )
        candidate = (score.propagation, alpha, score)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    combined_x = np.concatenate([train_x, validation_x])
    combined_y = np.concatenate([train_y, validation_y])
    standardized, mean, scale = _standardize(combined_x)
    ridge = Ridge(alpha=float(best[1])).fit(standardized, combined_y)
    prediction = project_descriptor(
        ridge.predict((np.asarray(test_x, float) - mean) / scale), n_modes
    )
    return prediction, float(best[1]), best[2]


def fit_matched_recency_baselines(
    train: EventHistoryDataset,
    validation: EventHistoryDataset,
    *,
    rank_prior: np.ndarray,
    decay_grid: Iterable[float],
    alpha_grid: Iterable[float],
    n_modes: int,
    n_contacts: int,
    scales: FamilyScales,
) -> tuple[dict[str, FeatureRidgeModel], FeatureRidgeModel]:
    selected: dict[str, FeatureRidgeModel] = {}
    for name in ("full_token_ewma", "descriptor_ewma"):
        candidates = [
            fit_feature_ridge(
                train,
                validation,
                feature_name=name,
                rank_prior=rank_prior,
                decay=float(decay),
                alpha_grid=alpha_grid,
                n_modes=n_modes,
                n_contacts=n_contacts,
                scales=scales,
            )
            for decay in decay_grid
        ]
        selected[name] = min(candidates, key=lambda model: model.validation_score.propagation)
    selected["binned_lag"] = fit_feature_ridge(
        train,
        validation,
        feature_name="binned_lag",
        rank_prior=rank_prior,
        decay=None,
        alpha_grid=alpha_grid,
        n_modes=n_modes,
        n_contacts=n_contacts,
        scales=scales,
    )
    best = min(selected.values(), key=lambda model: model.validation_score.propagation)
    return selected, best


def pca_leaky_state(histories: np.ndarray, pca: PCA, decay: float) -> np.ndarray:
    values = np.asarray(histories, float)
    projected = pca.transform(values.reshape(-1, values.shape[-1])).reshape(
        values.shape[0], values.shape[1], -1
    )
    return _weighted_mean(projected, float(decay))


def fit_low_dimensional_state(
    train: EventHistoryDataset,
    validation: EventHistoryDataset,
    *,
    base_model: FeatureRidgeModel,
    dimension_grid: Iterable[int],
    decay_grid: Iterable[float],
    alpha_grid: Iterable[float],
    n_modes: int,
    n_contacts: int,
    scales: FamilyScales,
    seed: int,
) -> LowDimensionalStateModel:
    train_base = base_model.predict(train)
    validation_base = base_model.predict(validation)
    residual = train.targets - train_base
    events = train.histories.reshape(-1, train.histories.shape[-1])
    maximum = min(len(events), 100_000)
    if len(events) > maximum:
        rng = np.random.default_rng(int(seed))
        events = events[rng.choice(len(events), maximum, replace=False)]
    best = None
    for dimension in map(int, dimension_grid):
        pca = PCA(
            n_components=min(dimension, events.shape[1]), random_state=int(seed)
        ).fit(events)
        for decay in map(float, decay_grid):
            train_state = pca_leaky_state(train.histories, pca, decay)
            validation_state = pca_leaky_state(validation.histories, pca, decay)
            standardized, mean, scale = _standardize(train_state)
            for alpha in map(float, alpha_grid):
                correction = Ridge(alpha=alpha).fit(standardized, residual)
                prediction = project_descriptor(
                    validation_base + correction.predict((validation_state - mean) / scale),
                    n_modes,
                )
                score = score_v24(
                    validation.targets,
                    prediction,
                    n_modes=n_modes,
                    n_contacts=n_contacts,
                    scales=scales,
                )
                candidate = (
                    score.propagation,
                    dimension,
                    decay,
                    alpha,
                    pca,
                    mean,
                    scale,
                    correction,
                    score,
                )
                if best is None or candidate[:4] < best[:4]:
                    best = candidate
    assert best is not None
    final_base = refit_feature_ridge(train, validation, base_model)
    histories = np.concatenate([train.histories, validation.histories])
    targets = np.concatenate([train.targets, validation.targets])
    combined_events = histories.reshape(-1, histories.shape[-1])
    if len(combined_events) > maximum:
        rng = np.random.default_rng(int(seed))
        combined_events = combined_events[rng.choice(len(combined_events), maximum, replace=False)]
    pca = PCA(
        n_components=min(int(best[1]), combined_events.shape[1]), random_state=int(seed)
    ).fit(combined_events)
    combined_dataset = _concatenate_for_prediction(train, validation)
    base = final_base.predict(combined_dataset)
    state = pca_leaky_state(histories, pca, float(best[2]))
    standardized, mean, scale = _standardize(state)
    correction = Ridge(alpha=float(best[3])).fit(standardized, targets - base)
    return LowDimensionalStateModel(
        base_model=final_base,
        pca=pca,
        decay=float(best[2]),
        dimension=int(best[1]),
        alpha=float(best[3]),
        state_mean=mean,
        state_scale=scale,
        correction=correction,
        n_modes=int(n_modes),
        validation_score=best[8],
    )


def _concatenate_for_prediction(
    left: EventHistoryDataset, right: EventHistoryDataset
) -> EventHistoryDataset:
    fields = {}
    for name in EventHistoryDataset.__dataclass_fields__:
        if name == "surrogate_kind":
            continue
        fields[name] = np.concatenate([getattr(left, name), getattr(right, name)])
    return EventHistoryDataset(**fields, surrogate_kind=left.surrogate_kind)


def split_half_reliability_v24(
    dataset: EventHistoryDataset,
    modes: np.ndarray,
    local_rank: np.ndarray,
    participation: np.ndarray,
    encoder: StableTemplateEncoder,
    *,
    train_target_mean: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    if dataset.target_event_indices.shape[1] < 4:
        raise ValueError("future windows need at least four events")
    rng = np.random.default_rng(int(seed))
    n_modes = encoder.n_modes
    n_contacts = np.asarray(local_rank).shape[1]
    slices = {
        "occupancy": slice(0, n_modes),
        "rank": slice(n_modes, n_modes + n_contacts),
        "participation": slice(n_modes + n_contacts, n_modes + 2 * n_contacts),
    }
    store = {
        family: {
            mode: {"correlation": [], "reliability": []}
            for mode in ("raw", "train_mean_residualized")
        }
        for family in slices
    }
    for _ in range(int(repeats)):
        left = []
        right = []
        for target in dataset.target_event_indices:
            order = rng.permutation(target)
            half = len(order) // 2
            left.append(repertoire_descriptor(order[:half], modes, local_rank, participation, encoder))
            right.append(repertoire_descriptor(order[half : 2 * half], modes, local_rank, participation, encoder))
        left = np.stack(left)
        right = np.stack(right)
        for family, section in slices.items():
            for mode in ("raw", "train_mean_residualized"):
                a = left[:, section]
                b = right[:, section]
                full = dataset.targets[:, section]
                if mode == "train_mean_residualized":
                    center = np.asarray(train_target_mean)[section]
                    a = a - center
                    b = b - center
                    full = full - center
                flat_a = a.ravel()
                flat_b = b.ravel()
                correlation = (
                    float(spearmanr(flat_a, flat_b).statistic)
                    if np.std(flat_a) > EPS and np.std(flat_b) > EPS
                    else float("nan")
                )
                variance = float(np.var(full))
                noise = float(np.mean((flat_a - flat_b) ** 2) / 4.0)
                reliability = float(1.0 - noise / variance) if variance > EPS else float("nan")
                store[family][mode]["correlation"].append(correlation)
                store[family][mode]["reliability"].append(reliability)
    return {
        family: {
            mode: {
                "split_half_spearman_median": float(
                    np.nanmedian(values["correlation"])
                ),
                "variance_reliability_median": float(
                    np.nanmedian(values["reliability"])
                ),
            }
            for mode, values in modes_store.items()
        }
        for family, modes_store in store.items()
    }


def dataset_time_audit(dataset: EventHistoryDataset) -> dict[str, object]:
    history_duration = np.ptp(dataset.history_event_times, axis=1)
    target_duration = np.ptp(dataset.target_event_times, axis=1)
    history_iei = np.asarray(
        [np.median(np.diff(np.sort(row))) for row in dataset.history_event_times], float
    )
    event_rate = np.divide(
        dataset.history_event_indices.shape[1] - 1,
        history_duration,
        out=np.zeros_like(history_duration),
        where=history_duration > 0,
    )

    def quantiles(values: np.ndarray) -> dict[str, float]:
        return {
            "q10": float(np.quantile(values, 0.10)),
            "q50": float(np.quantile(values, 0.50)),
            "q90": float(np.quantile(values, 0.90)),
        }

    return {
        "n_windows": int(len(dataset)),
        "n_source_records": int(len(np.unique(dataset.source_ids))),
        "history_duration_seconds": quantiles(history_duration),
        "target_duration_seconds": quantiles(target_duration),
        "history_median_iei_seconds": quantiles(history_iei),
        "history_event_rate_hz": quantiles(event_rate),
        "source_progress": quantiles(dataset.time_features[:, -1]),
    }
