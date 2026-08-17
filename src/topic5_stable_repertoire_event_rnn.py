"""Event-indexed prediction of a stable interictal propagation repertoire.

One sequence step is one complete event.  Stable event modes are fitted on the
training sources only; models predict a non-overlapping future-event window.
The module intentionally contains no within-event next-rank recurrence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - numpy baselines remain importable
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


EPS = 1e-8


@dataclass(frozen=True)
class SourcePartition:
    train_sources: np.ndarray
    validation_sources: np.ndarray
    test_sources: np.ndarray

    def indices(
        self,
        source_ids: np.ndarray,
        name: str,
        eligible_indices: Sequence[int] | None = None,
    ) -> np.ndarray:
        sources = {
            "train": self.train_sources,
            "validation": self.validation_sources,
            "test": self.test_sources,
        }[name]
        source = np.asarray(source_ids)
        if eligible_indices is None:
            eligible = np.arange(len(source), dtype=int)
        else:
            eligible = np.asarray(eligible_indices, int)
        return eligible[np.isin(source[eligible], sources)]


@dataclass(frozen=True)
class StableTemplateEncoder:
    centers: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    rank_prior: np.ndarray
    n_modes: int

    def transform_rank(self, local_rank: np.ndarray, participation: np.ndarray) -> np.ndarray:
        features = masked_rank_features(local_rank, participation)
        standardized = (features - self.feature_mean) / self.feature_scale
        distances = np.sum(
            (standardized[:, None, :] - self.centers[None, :, :]) ** 2,
            axis=2,
        )
        return np.argmin(distances, axis=1).astype(np.int64)

    def event_tokens(
        self,
        local_rank: np.ndarray,
        participation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rank = np.asarray(local_rank, float)
        part = np.asarray(participation, bool)
        modes = self.transform_rank(rank, part)
        imputed = np.where(part & np.isfinite(rank), rank, self.rank_prior[None, :])
        one_hot = np.eye(self.n_modes, dtype=float)[modes]
        return np.concatenate([imputed, part.astype(float), one_hot], axis=1), modes


@dataclass(frozen=True)
class FutureWindowDataset:
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

    def __len__(self) -> int:
        return int(len(self.targets))


@dataclass(frozen=True)
class ScoreBreakdown:
    composite: float
    occupancy: float
    rank: float
    participation: float


@dataclass(frozen=True)
class RidgeResult:
    model: Ridge
    alpha: float
    validation_score: ScoreBreakdown


@dataclass(frozen=True)
class LinearStateResult:
    pca: PCA
    decay: float
    ridge: Ridge
    alpha: float
    dimension: int
    n_modes: int
    validation_score: ScoreBreakdown

    def predict(self, histories: np.ndarray) -> np.ndarray:
        state = leaky_linear_state(histories, self.pca, self.decay)
        return project_descriptor(self.ridge.predict(state), self.n_modes)


@dataclass(frozen=True)
class ResidualLinearStateResult:
    base_model: Ridge
    pca: PCA
    decay: float
    correction_model: Ridge
    base_alpha: float
    correction_alpha: float
    dimension: int
    n_modes: int
    validation_score: ScoreBreakdown

    def predict(self, dataset: FutureWindowDataset) -> np.ndarray:
        base = project_descriptor(
            self.base_model.predict(dataset.history_descriptors), self.n_modes
        )
        state = leaky_linear_state(dataset.histories, self.pca, self.decay)
        return project_descriptor(base + self.correction_model.predict(state), self.n_modes)


@dataclass
class GRUStateResult:
    model: object
    token_mean: np.ndarray
    token_scale: np.ndarray
    hidden_size: int
    weight_decay: float
    best_epoch: int
    best_validation_score: ScoreBreakdown
    n_parameters: int
    n_modes: int

    def predict(self, histories: np.ndarray, batch_size: int = 512) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is required for GRU prediction")
        values = (np.asarray(histories, float) - self.token_mean) / self.token_scale
        tensor = torch.as_tensor(values, dtype=torch.float32)
        output = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(tensor), int(batch_size)):
                output.append(self.model(tensor[start : start + int(batch_size)]).cpu().numpy())
        return np.concatenate(output)


@dataclass
class ResidualGRUStateResult:
    base_model: Ridge
    model: object
    token_mean: np.ndarray
    token_scale: np.ndarray
    hidden_size: int
    weight_decay: float
    base_alpha: float
    best_epoch: int
    best_validation_score: ScoreBreakdown
    n_parameters: int
    n_modes: int

    def predict(self, dataset: FutureWindowDataset, batch_size: int = 512) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is required for GRU prediction")
        base = project_descriptor(
            self.base_model.predict(dataset.history_descriptors), self.n_modes
        )
        values = (np.asarray(dataset.histories, float) - self.token_mean) / self.token_scale
        tensor = torch.as_tensor(values, dtype=torch.float32)
        correction = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(tensor), int(batch_size)):
                correction.append(
                    self.model(tensor[start : start + int(batch_size)]).cpu().numpy()
                )
        return project_descriptor(base + np.concatenate(correction), self.n_modes)


if nn is not None:
    class GRUEventState(nn.Module):
        """Future-window decoder whose recurrent step is one complete event."""

        def __init__(self, input_dim: int, hidden_size: int, target_dim: int, n_modes: int):
            super().__init__()
            self.gru = nn.GRU(int(input_dim), int(hidden_size), batch_first=True)
            self.readout = nn.Linear(int(hidden_size), int(target_dim))
            self.n_modes = int(n_modes)

        def forward(self, histories):
            _, hidden = self.gru(histories)
            values = torch.sigmoid(self.readout(hidden[-1]))
            occupancy = values[:, : self.n_modes]
            occupancy = occupancy / torch.clamp(occupancy.sum(dim=1, keepdim=True), min=EPS)
            return torch.cat([occupancy, values[:, self.n_modes :]], dim=1)


    class GRUResidualEventState(nn.Module):
        """Ordered correction on top of the unordered long-history baseline."""

        def __init__(self, input_dim: int, hidden_size: int, target_dim: int):
            super().__init__()
            self.gru = nn.GRU(int(input_dim), int(hidden_size), batch_first=True)
            self.readout = nn.Linear(int(hidden_size), int(target_dim))

        def forward(self, histories):
            _, hidden = self.gru(histories)
            return self.readout(hidden[-1])
else:  # pragma: no cover
    class GRUEventState:  # type: ignore[no-redef]
        pass

    class GRUResidualEventState:  # type: ignore[no-redef]
        pass


def concatenate_window_datasets(*datasets: FutureWindowDataset) -> FutureWindowDataset:
    if not datasets:
        raise ValueError("at least one dataset is required")
    return FutureWindowDataset(
        histories=np.concatenate([item.histories for item in datasets]),
        recent_descriptors=np.concatenate([item.recent_descriptors for item in datasets]),
        history_descriptors=np.concatenate([item.history_descriptors for item in datasets]),
        targets=np.concatenate([item.targets for item in datasets]),
        last_mode=np.concatenate([item.last_mode for item in datasets]),
        source_ids=np.concatenate([item.source_ids for item in datasets]),
        history_start=np.concatenate([item.history_start for item in datasets]),
        history_stop=np.concatenate([item.history_stop for item in datasets]),
        target_start=np.concatenate([item.target_start for item in datasets]),
        target_stop=np.concatenate([item.target_stop for item in datasets]),
        history_event_indices=np.concatenate([item.history_event_indices for item in datasets]),
        target_event_indices=np.concatenate([item.target_event_indices for item in datasets]),
    )


def chronological_source_partition(
    source_ids: np.ndarray,
    event_time: np.ndarray,
    eligible_indices: Sequence[int],
    fractions: Sequence[float] = (0.6, 0.2, 0.2),
) -> SourcePartition:
    """Split whole source recordings in chronology; no source is shared."""
    source = np.asarray(source_ids)
    times = np.asarray(event_time, float)
    eligible = np.asarray(eligible_indices, int)
    if eligible.ndim != 1 or len(eligible) == 0:
        raise ValueError("eligible_indices must be non-empty and one-dimensional")
    if not np.isclose(np.sum(fractions), 1.0):
        raise ValueError("fractions must sum to one")
    unique = np.unique(source[eligible])
    ordered = sorted(
        unique.tolist(),
        key=lambda value: float(np.min(times[eligible[source[eligible] == value]])),
    )
    n_sources = len(ordered)
    if n_sources < 5:
        raise ValueError("at least five source recordings are required")
    first_cut = max(1, int(np.floor(float(fractions[0]) * n_sources)))
    second_cut = max(first_cut + 1, int(np.floor(float(sum(fractions[:2])) * n_sources)))
    second_cut = min(second_cut, n_sources - 1)
    return SourcePartition(
        train_sources=np.asarray(ordered[:first_cut], dtype=source.dtype),
        validation_sources=np.asarray(ordered[first_cut:second_cut], dtype=source.dtype),
        test_sources=np.asarray(ordered[second_cut:], dtype=source.dtype),
    )


def masked_rank_features(local_rank: np.ndarray, participation: np.ndarray) -> np.ndarray:
    """Canonical event-median imputation for masked rank clustering."""
    rank = np.asarray(local_rank, float)
    part = np.asarray(participation, bool)
    if rank.ndim != 2 or part.shape != rank.shape:
        raise ValueError("local_rank and participation must be aligned event x contact arrays")
    valid = part & np.isfinite(rank)
    count = np.sum(valid, axis=1)
    if np.any(count == 0):
        raise ValueError("every event must contain at least one participating contact")
    masked = np.where(valid, rank, np.nan)
    medians = np.nanmedian(masked, axis=1)
    return np.where(valid, rank, medians[:, None])


def fit_stable_templates(
    local_rank: np.ndarray,
    participation: np.ndarray,
    train_indices: Sequence[int],
    *,
    n_modes: int = 2,
    seed: int = 0,
) -> StableTemplateEncoder:
    """Fit a patient-specific stable mode encoder on train events only."""
    rank = np.asarray(local_rank, float)
    part = np.asarray(participation, bool)
    selected = np.asarray(train_indices, int)
    if len(selected) < 10 * int(n_modes):
        raise ValueError("too few training events for stable templates")
    valid = part[selected] & np.isfinite(rank[selected])
    count = np.sum(valid, axis=0)
    total = np.sum(np.where(valid, rank[selected], 0.0), axis=0)
    prior = np.divide(total, count, out=np.full(rank.shape[1], 0.5), where=count > 0)
    features = masked_rank_features(rank[selected], part[selected])
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > EPS, scale, 1.0)
    standardized = (features - mean) / scale
    km = KMeans(n_clusters=int(n_modes), n_init=20, random_state=int(seed))
    km.fit(standardized)
    return StableTemplateEncoder(
        centers=np.asarray(km.cluster_centers_, float),
        feature_mean=np.asarray(mean, float),
        feature_scale=np.asarray(scale, float),
        rank_prior=np.asarray(prior, float),
        n_modes=int(n_modes),
    )


def repertoire_descriptor(
    indices: Sequence[int],
    modes: np.ndarray,
    local_rank: np.ndarray,
    participation: np.ndarray,
    encoder: StableTemplateEncoder,
    *,
    rank_prior_events: float = 2.0,
    beta_prior: float = 1.0,
) -> np.ndarray:
    selected = np.asarray(indices, int)
    labels = np.asarray(modes, int)[selected]
    rank = np.asarray(local_rank, float)[selected]
    part = np.asarray(participation, bool)[selected]
    occupancy = np.bincount(labels, minlength=encoder.n_modes).astype(float) / len(selected)
    valid = part & np.isfinite(rank)
    count = np.sum(valid, axis=0, dtype=float)
    total = np.sum(np.where(valid, rank, 0.0), axis=0)
    alpha = float(rank_prior_events)
    rank_mean = (total + alpha * encoder.rank_prior) / (count + alpha)
    beta = float(beta_prior)
    participation_mean = (np.sum(part, axis=0) + beta) / (len(selected) + 2.0 * beta)
    return np.concatenate([occupancy, rank_mean, participation_mean])


def build_future_window_dataset(
    tokens: np.ndarray,
    modes: np.ndarray,
    local_rank: np.ndarray,
    participation: np.ndarray,
    source_ids: np.ndarray,
    event_time: np.ndarray,
    eligible_indices: Sequence[int],
    encoder: StableTemplateEncoder,
    *,
    history_length: int,
    horizon: int,
    stride: int | None = None,
) -> FutureWindowDataset:
    """Build fixed-history, strictly future, within-source prediction samples."""
    values = np.asarray(tokens, float)
    source = np.asarray(source_ids)
    times = np.asarray(event_time, float)
    eligible = np.asarray(eligible_indices, int)
    length = int(history_length)
    future = int(horizon)
    step = future if stride is None else int(stride)
    if length < future or future < 2 or step < 1:
        raise ValueError("history_length >= horizon >= 2 and stride >= 1 are required")
    records: list[
        tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, object,
            int, int, int, int, np.ndarray, np.ndarray,
        ]
    ] = []
    for source_id in np.unique(source[eligible]):
        seq = eligible[source[eligible] == source_id]
        seq = seq[np.argsort(times[seq], kind="mergesort")]
        for anchor in range(length, len(seq) - future + 1, step):
            history = seq[anchor - length : anchor]
            target = seq[anchor : anchor + future]
            recent = history[-future:]
            records.append(
                (
                    values[history],
                    repertoire_descriptor(recent, modes, local_rank, participation, encoder),
                    repertoire_descriptor(history, modes, local_rank, participation, encoder),
                    repertoire_descriptor(target, modes, local_rank, participation, encoder),
                    int(modes[history[-1]]),
                    source_id,
                    int(history[0]),
                    int(history[-1]) + 1,
                    int(target[0]),
                    int(target[-1]) + 1,
                    history.copy(),
                    target.copy(),
                )
            )
    if not records:
        raise ValueError("no within-source future-window samples")
    return FutureWindowDataset(
        histories=np.stack([item[0] for item in records]),
        recent_descriptors=np.stack([item[1] for item in records]),
        history_descriptors=np.stack([item[2] for item in records]),
        targets=np.stack([item[3] for item in records]),
        last_mode=np.asarray([item[4] for item in records], int),
        source_ids=np.asarray([item[5] for item in records]),
        history_start=np.asarray([item[6] for item in records], int),
        history_stop=np.asarray([item[7] for item in records], int),
        target_start=np.asarray([item[8] for item in records], int),
        target_stop=np.asarray([item[9] for item in records], int),
        history_event_indices=np.stack([item[10] for item in records]),
        target_event_indices=np.stack([item[11] for item in records]),
    )


def score_predictions(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    n_modes: int,
    n_contacts: int,
) -> ScoreBreakdown:
    truth = np.asarray(target, float)
    estimate = np.asarray(prediction, float)
    if truth.shape != estimate.shape or truth.shape[1] != n_modes + 2 * n_contacts:
        raise ValueError("target and prediction do not match the repertoire descriptor")
    occupation = float(np.mean((truth[:, :n_modes] - estimate[:, :n_modes]) ** 2))
    rank_slice = slice(n_modes, n_modes + n_contacts)
    rank = float(np.mean((truth[:, rank_slice] - estimate[:, rank_slice]) ** 2))
    participation = float(np.mean((truth[:, n_modes + n_contacts :] - estimate[:, n_modes + n_contacts :]) ** 2))
    return ScoreBreakdown(
        composite=float((occupation + rank + participation) / 3.0),
        occupancy=occupation,
        rank=rank,
        participation=participation,
    )


def _torch_family_loss(prediction, target, n_modes: int, n_contacts: int):
    occupancy = torch.mean((prediction[:, :n_modes] - target[:, :n_modes]) ** 2)
    rank = torch.mean(
        (prediction[:, n_modes : n_modes + n_contacts] - target[:, n_modes : n_modes + n_contacts]) ** 2
    )
    participation = torch.mean(
        (prediction[:, n_modes + n_contacts :] - target[:, n_modes + n_contacts :]) ** 2
    )
    return (occupancy + rank + participation) / 3.0


def _set_torch_seed(seed: int) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for GRU training")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.use_deterministic_algorithms(True)


def _fit_one_gru(
    train_histories: np.ndarray,
    train_targets: np.ndarray,
    *,
    validation_histories: np.ndarray | None,
    validation_targets: np.ndarray | None,
    hidden_size: int,
    weight_decay: float,
    learning_rate: float,
    batch_size: int,
    maximum_epochs: int,
    patience: int,
    n_modes: int,
    n_contacts: int,
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[object, int, ScoreBreakdown]:
    _set_torch_seed(seed)
    model = GRUEventState(
        train_histories.shape[-1], hidden_size, train_targets.shape[-1], n_modes
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    train_set = TensorDataset(
        torch.as_tensor(train_histories, dtype=torch.float32),
        torch.as_tensor(train_targets, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        train_set,
        batch_size=min(int(batch_size), len(train_set)),
        shuffle=True,
        generator=generator,
    )
    best_state = None
    best_score = float("inf")
    best_breakdown = ScoreBreakdown(float("nan"), float("nan"), float("nan"), float("nan"))
    best_epoch = 0
    stale = 0
    epochs = int(fixed_epochs if fixed_epochs is not None else maximum_epochs)
    for epoch in range(epochs):
        model.train()
        for histories, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(histories)
            loss = _torch_family_loss(prediction, targets, n_modes, n_contacts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if fixed_epochs is not None:
            continue
        model.eval()
        with torch.no_grad():
            validation_prediction = model(
                torch.as_tensor(validation_histories, dtype=torch.float32)
            ).cpu().numpy()
        breakdown = score_predictions(
            validation_targets,
            validation_prediction,
            n_modes=n_modes,
            n_contacts=n_contacts,
        )
        if breakdown.composite < best_score - 1e-7:
            best_score = breakdown.composite
            best_breakdown = breakdown
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= 20 and stale >= int(patience):
            break
    if fixed_epochs is None:
        if best_state is None:
            raise RuntimeError("GRU training produced no finite validation checkpoint")
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs - 1
    return model, int(best_epoch), best_breakdown


def fit_gru_event_state(
    train: FutureWindowDataset,
    validation: FutureWindowDataset,
    *,
    hidden_size_grid: Iterable[int],
    weight_decay_grid: Iterable[float],
    learning_rate: float,
    batch_size: int,
    maximum_epochs: int,
    patience: int,
    n_modes: int,
    n_contacts: int,
    seed: int,
) -> GRUStateResult:
    """Select GRU capacity on validation, then refit on train+validation."""
    token_mean = np.mean(train.histories, axis=(0, 1), keepdims=True)
    token_scale = np.std(train.histories, axis=(0, 1), keepdims=True)
    token_scale = np.where(token_scale > EPS, token_scale, 1.0)
    train_x = (train.histories - token_mean) / token_scale
    validation_x = (validation.histories - token_mean) / token_scale
    best = None
    for hidden_size in map(int, hidden_size_grid):
        for weight_decay in map(float, weight_decay_grid):
            model, epoch, breakdown = _fit_one_gru(
                train_x,
                train.targets,
                validation_histories=validation_x,
                validation_targets=validation.targets,
                hidden_size=hidden_size,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                batch_size=batch_size,
                maximum_epochs=maximum_epochs,
                patience=patience,
                n_modes=n_modes,
                n_contacts=n_contacts,
                seed=seed,
            )
            candidate = (breakdown.composite, hidden_size, weight_decay, model, epoch, breakdown)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    assert best is not None
    combined = concatenate_window_datasets(train, validation)
    final_mean = np.mean(combined.histories, axis=(0, 1), keepdims=True)
    final_scale = np.std(combined.histories, axis=(0, 1), keepdims=True)
    final_scale = np.where(final_scale > EPS, final_scale, 1.0)
    combined_x = (combined.histories - final_mean) / final_scale
    final_model, _, _ = _fit_one_gru(
        combined_x,
        combined.targets,
        validation_histories=None,
        validation_targets=None,
        hidden_size=best[1],
        weight_decay=best[2],
        learning_rate=learning_rate,
        batch_size=batch_size,
        maximum_epochs=maximum_epochs,
        patience=patience,
        n_modes=n_modes,
        n_contacts=n_contacts,
        seed=seed,
        fixed_epochs=max(1, int(best[4]) + 1),
    )
    return GRUStateResult(
        model=final_model,
        token_mean=final_mean,
        token_scale=final_scale,
        hidden_size=int(best[1]),
        weight_decay=float(best[2]),
        best_epoch=int(best[4]),
        best_validation_score=best[5],
        n_parameters=int(sum(parameter.numel() for parameter in final_model.parameters())),
        n_modes=int(n_modes),
    )


def _fit_one_residual_gru(
    train_histories: np.ndarray,
    train_residual: np.ndarray,
    *,
    validation_histories: np.ndarray | None,
    validation_base: np.ndarray | None,
    validation_targets: np.ndarray | None,
    hidden_size: int,
    weight_decay: float,
    learning_rate: float,
    batch_size: int,
    maximum_epochs: int,
    patience: int,
    n_modes: int,
    n_contacts: int,
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[object, int, ScoreBreakdown]:
    _set_torch_seed(seed)
    model = GRUResidualEventState(
        train_histories.shape[-1], hidden_size, train_residual.shape[-1]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    train_set = TensorDataset(
        torch.as_tensor(train_histories, dtype=torch.float32),
        torch.as_tensor(train_residual, dtype=torch.float32),
    )
    loader = DataLoader(
        train_set,
        batch_size=min(int(batch_size), len(train_set)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    best_state = None
    best_score = float("inf")
    best_breakdown = ScoreBreakdown(float("nan"), float("nan"), float("nan"), float("nan"))
    best_epoch = 0
    stale = 0
    epochs = int(fixed_epochs if fixed_epochs is not None else maximum_epochs)
    for epoch in range(epochs):
        model.train()
        for histories, residual in loader:
            optimizer.zero_grad(set_to_none=True)
            correction = model(histories)
            loss = _torch_family_loss(correction, residual, n_modes, n_contacts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if fixed_epochs is not None:
            continue
        model.eval()
        with torch.no_grad():
            correction = model(
                torch.as_tensor(validation_histories, dtype=torch.float32)
            ).cpu().numpy()
        prediction = project_descriptor(validation_base + correction, n_modes)
        breakdown = score_predictions(
            validation_targets,
            prediction,
            n_modes=n_modes,
            n_contacts=n_contacts,
        )
        if breakdown.composite < best_score - 1e-7:
            best_score = breakdown.composite
            best_breakdown = breakdown
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= 20 and stale >= int(patience):
            break
    if fixed_epochs is None:
        if best_state is None:
            raise RuntimeError("residual GRU produced no finite validation checkpoint")
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs - 1
    return model, int(best_epoch), best_breakdown


def fit_residual_gru_event_state(
    train: FutureWindowDataset,
    validation: FutureWindowDataset,
    *,
    hidden_size_grid: Iterable[int],
    weight_decay_grid: Iterable[float],
    alpha_grid: Iterable[float],
    learning_rate: float,
    batch_size: int,
    maximum_epochs: int,
    patience: int,
    n_modes: int,
    n_contacts: int,
    seed: int,
) -> ResidualGRUStateResult:
    """Nest an ordered GRU correction on the matched unordered L-event baseline."""
    base_selected = fit_history_summary_ridge(
        train,
        validation,
        alpha_grid=alpha_grid,
        n_modes=n_modes,
        n_contacts=n_contacts,
    )
    train_base = project_descriptor(
        base_selected.model.predict(train.history_descriptors), n_modes
    )
    validation_base = project_descriptor(
        base_selected.model.predict(validation.history_descriptors), n_modes
    )
    token_mean = np.mean(train.histories, axis=(0, 1), keepdims=True)
    token_scale = np.std(train.histories, axis=(0, 1), keepdims=True)
    token_scale = np.where(token_scale > EPS, token_scale, 1.0)
    train_x = (train.histories - token_mean) / token_scale
    validation_x = (validation.histories - token_mean) / token_scale
    best = None
    for hidden_size in map(int, hidden_size_grid):
        for weight_decay in map(float, weight_decay_grid):
            model, epoch, breakdown = _fit_one_residual_gru(
                train_x,
                train.targets - train_base,
                validation_histories=validation_x,
                validation_base=validation_base,
                validation_targets=validation.targets,
                hidden_size=hidden_size,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                batch_size=batch_size,
                maximum_epochs=maximum_epochs,
                patience=patience,
                n_modes=n_modes,
                n_contacts=n_contacts,
                seed=seed,
            )
            candidate = (breakdown.composite, hidden_size, weight_decay, model, epoch, breakdown)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    assert best is not None
    combined = concatenate_window_datasets(train, validation)
    final_base = Ridge(alpha=float(base_selected.alpha)).fit(
        combined.history_descriptors, combined.targets
    )
    combined_base = project_descriptor(
        final_base.predict(combined.history_descriptors), n_modes
    )
    final_mean = np.mean(combined.histories, axis=(0, 1), keepdims=True)
    final_scale = np.std(combined.histories, axis=(0, 1), keepdims=True)
    final_scale = np.where(final_scale > EPS, final_scale, 1.0)
    combined_x = (combined.histories - final_mean) / final_scale
    final_model, _, _ = _fit_one_residual_gru(
        combined_x,
        combined.targets - combined_base,
        validation_histories=None,
        validation_base=None,
        validation_targets=None,
        hidden_size=best[1],
        weight_decay=best[2],
        learning_rate=learning_rate,
        batch_size=batch_size,
        maximum_epochs=maximum_epochs,
        patience=patience,
        n_modes=n_modes,
        n_contacts=n_contacts,
        seed=seed,
        fixed_epochs=max(1, int(best[4]) + 1),
    )
    return ResidualGRUStateResult(
        base_model=final_base,
        model=final_model,
        token_mean=final_mean,
        token_scale=final_scale,
        hidden_size=int(best[1]),
        weight_decay=float(best[2]),
        base_alpha=float(base_selected.alpha),
        best_epoch=int(best[4]),
        best_validation_score=best[5],
        n_parameters=int(sum(parameter.numel() for parameter in final_model.parameters())),
        n_modes=int(n_modes),
    )


def project_descriptor(prediction: np.ndarray, n_modes: int) -> np.ndarray:
    """Project unconstrained regression output to a valid repertoire descriptor."""
    values = np.clip(np.asarray(prediction, float), 0.0, 1.0)
    occupancy = values[:, : int(n_modes)]
    total = np.sum(occupancy, axis=1, keepdims=True)
    occupancy = np.divide(
        occupancy,
        total,
        out=np.full_like(occupancy, 1.0 / int(n_modes)),
        where=total > EPS,
    )
    values[:, : int(n_modes)] = occupancy
    return values


def fit_recent_ridge(
    train: FutureWindowDataset,
    validation: FutureWindowDataset,
    *,
    alpha_grid: Iterable[float],
    n_modes: int,
    n_contacts: int,
) -> RidgeResult:
    best: tuple[float, float, Ridge, ScoreBreakdown] | None = None
    for alpha in map(float, alpha_grid):
        model = Ridge(alpha=alpha).fit(train.recent_descriptors, train.targets)
        prediction = project_descriptor(model.predict(validation.recent_descriptors), n_modes)
        score = score_predictions(validation.targets, prediction, n_modes=n_modes, n_contacts=n_contacts)
        candidate = (score.composite, alpha, model, score)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return RidgeResult(model=best[2], alpha=best[1], validation_score=best[3])


def fit_history_summary_ridge(
    train: FutureWindowDataset,
    validation: FutureWindowDataset,
    *,
    alpha_grid: Iterable[float],
    n_modes: int,
    n_contacts: int,
) -> RidgeResult:
    best = None
    for alpha in map(float, alpha_grid):
        model = Ridge(alpha=alpha).fit(train.history_descriptors, train.targets)
        prediction = project_descriptor(model.predict(validation.history_descriptors), n_modes)
        score = score_predictions(
            validation.targets, prediction, n_modes=n_modes, n_contacts=n_contacts
        )
        candidate = (score.composite, alpha, model, score)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return RidgeResult(model=best[2], alpha=best[1], validation_score=best[3])


def estimate_transition_matrix(
    modes: np.ndarray,
    source_ids: np.ndarray,
    train_indices: Sequence[int],
    *,
    n_modes: int,
    pseudocount: float = 1.0,
) -> np.ndarray:
    counts = np.full((n_modes, n_modes), float(pseudocount))
    labels = np.asarray(modes, int)
    source = np.asarray(source_ids)
    selected = np.asarray(train_indices, int)
    for source_id in np.unique(source[selected]):
        seq = selected[source[selected] == source_id]
        for left, right in zip(seq[:-1], seq[1:]):
            counts[labels[left], labels[right]] += 1.0
    return counts / np.sum(counts, axis=1, keepdims=True)


def mode_conditioned_descriptors(
    train_indices: Sequence[int],
    modes: np.ndarray,
    local_rank: np.ndarray,
    participation: np.ndarray,
    encoder: StableTemplateEncoder,
) -> np.ndarray:
    selected = np.asarray(train_indices, int)
    output = []
    for mode in range(encoder.n_modes):
        idx = selected[np.asarray(modes, int)[selected] == mode]
        if len(idx) == 0:
            raise ValueError("empty train mode")
        descriptor = repertoire_descriptor(idx, modes, local_rank, participation, encoder)
        descriptor[: encoder.n_modes] = 0.0
        descriptor[mode] = 1.0
        output.append(descriptor)
    return np.stack(output)


def train_to_partition_template_stability(
    local_rank: np.ndarray,
    participation: np.ndarray,
    train_indices: Sequence[int],
    comparison_indices: Sequence[int],
    train_encoder: StableTemplateEncoder,
    *,
    seed: int = 0,
) -> dict[str, object]:
    """Independently re-fit a partition and match its templates to train."""
    train_idx = np.asarray(train_indices, int)
    comparison_idx = np.asarray(comparison_indices, int)
    comparison_encoder = fit_stable_templates(
        local_rank,
        participation,
        comparison_idx,
        n_modes=train_encoder.n_modes,
        seed=int(seed),
    )
    train_modes = train_encoder.transform_rank(local_rank, participation)
    comparison_modes = comparison_encoder.transform_rank(local_rank, participation)
    train_descriptors = mode_conditioned_descriptors(
        train_idx, train_modes, local_rank, participation, train_encoder
    )
    comparison_descriptors = mode_conditioned_descriptors(
        comparison_idx,
        comparison_modes,
        local_rank,
        participation,
        comparison_encoder,
    )
    k = train_encoder.n_modes
    n_contacts = np.asarray(local_rank).shape[1]
    train_rank = train_descriptors[:, k : k + n_contacts]
    comparison_rank = comparison_descriptors[:, k : k + n_contacts]
    correlation = np.full((k, k), -1.0, float)
    for left in range(k):
        for right in range(k):
            value = spearmanr(train_rank[left], comparison_rank[right]).statistic
            correlation[left, right] = float(value) if np.isfinite(value) else -1.0
    rows, columns = linear_sum_assignment(-correlation)
    mapping = {int(left): int(right) for left, right in zip(rows, columns)}
    matched = np.asarray([correlation[left, right] for left, right in zip(rows, columns)])
    assigned_by_train = train_modes[comparison_idx]
    independent = comparison_modes[comparison_idx]
    mapped = np.asarray([mapping[int(label)] for label in assigned_by_train], int)
    agreement = float(np.mean(mapped == independent))
    mean_match = float(np.mean(matched))
    if mean_match >= 0.8 and agreement >= 0.70:
        grade = "strong"
    elif mean_match >= 0.5 and agreement >= 0.50:
        grade = "moderate"
    else:
        grade = "weak"
    return {
        "grade": grade,
        "mean_match_spearman": mean_match,
        "minimum_match_spearman": float(np.min(matched)),
        "assignment_agreement": agreement,
        "correlation_matrix": correlation.tolist(),
        "mapping_train_to_partition": {str(key): value for key, value in mapping.items()},
    }


def transition_predictions(
    last_mode: np.ndarray,
    transition: np.ndarray,
    mode_descriptors: np.ndarray,
    horizon: int,
) -> np.ndarray:
    matrix = np.asarray(transition, float)
    k = matrix.shape[0]
    future_by_start = []
    for start in range(k):
        distribution = np.eye(k)[start]
        total = np.zeros(k, float)
        for _ in range(int(horizon)):
            distribution = distribution @ matrix
            total += distribution
        occupancy = total / float(horizon)
        prediction = occupancy @ mode_descriptors
        prediction[:k] = occupancy
        future_by_start.append(prediction)
    return np.stack(future_by_start)[np.asarray(last_mode, int)]


def leaky_linear_state(histories: np.ndarray, pca: PCA, decay: float) -> np.ndarray:
    values = np.asarray(histories, float)
    flat = values.reshape(-1, values.shape[-1])
    projected = pca.transform(flat).reshape(values.shape[0], values.shape[1], -1)
    state = np.zeros((len(values), projected.shape[-1]), float)
    for step in range(projected.shape[1]):
        state = float(decay) * state + projected[:, step, :]
    normalization = sum(float(decay) ** lag for lag in range(projected.shape[1]))
    return state / max(normalization, EPS)


def fit_linear_event_state(
    train: FutureWindowDataset,
    validation: FutureWindowDataset,
    *,
    dimension_grid: Iterable[int],
    decay_grid: Iterable[float],
    alpha_grid: Iterable[float],
    n_modes: int,
    n_contacts: int,
    seed: int = 0,
) -> LinearStateResult:
    train_events = train.histories.reshape(-1, train.histories.shape[-1])
    maximum = min(len(train_events), 100_000)
    if len(train_events) > maximum:
        rng = np.random.default_rng(int(seed))
        train_events = train_events[rng.choice(len(train_events), maximum, replace=False)]
    best: tuple[float, int, float, float, PCA, Ridge, ScoreBreakdown] | None = None
    for dimension in map(int, dimension_grid):
        pca = PCA(n_components=min(dimension, train_events.shape[1]), random_state=int(seed)).fit(train_events)
        for decay in map(float, decay_grid):
            train_state = leaky_linear_state(train.histories, pca, decay)
            validation_state = leaky_linear_state(validation.histories, pca, decay)
            for alpha in map(float, alpha_grid):
                ridge = Ridge(alpha=alpha).fit(train_state, train.targets)
                prediction = project_descriptor(ridge.predict(validation_state), n_modes)
                score = score_predictions(validation.targets, prediction, n_modes=n_modes, n_contacts=n_contacts)
                candidate = (score.composite, dimension, decay, alpha, pca, ridge, score)
                if best is None or candidate[:4] < best[:4]:
                    best = candidate
    assert best is not None
    return LinearStateResult(
        pca=best[4], decay=best[2], ridge=best[5], alpha=best[3],
        dimension=best[1], n_modes=int(n_modes), validation_score=best[6]
    )


def refit_linear_event_state(
    dataset: FutureWindowDataset,
    *,
    dimension: int,
    decay: float,
    alpha: float,
    n_modes: int,
    seed: int = 0,
) -> LinearStateResult:
    events = dataset.histories.reshape(-1, dataset.histories.shape[-1])
    maximum = min(len(events), 100_000)
    if len(events) > maximum:
        rng = np.random.default_rng(int(seed))
        events = events[rng.choice(len(events), maximum, replace=False)]
    pca = PCA(n_components=min(int(dimension), events.shape[1]), random_state=int(seed)).fit(events)
    state = leaky_linear_state(dataset.histories, pca, float(decay))
    ridge = Ridge(alpha=float(alpha)).fit(state, dataset.targets)
    fitted = LinearStateResult(
        pca=pca,
        decay=float(decay),
        ridge=ridge,
        alpha=float(alpha),
        dimension=int(dimension),
        n_modes=int(n_modes),
        validation_score=ScoreBreakdown(float("nan"), float("nan"), float("nan"), float("nan")),
    )
    return fitted


def fit_residual_linear_event_state(
    train: FutureWindowDataset,
    validation: FutureWindowDataset,
    *,
    dimension_grid: Iterable[int],
    decay_grid: Iterable[float],
    alpha_grid: Iterable[float],
    n_modes: int,
    n_contacts: int,
    seed: int = 0,
) -> ResidualLinearStateResult:
    """Nest an ordered low-dimensional history correction on the R1 baseline."""
    base_selected = fit_history_summary_ridge(
        train,
        validation,
        alpha_grid=alpha_grid,
        n_modes=n_modes,
        n_contacts=n_contacts,
    )
    base_model = base_selected.model
    train_base = project_descriptor(base_model.predict(train.history_descriptors), n_modes)
    validation_base = project_descriptor(
        base_model.predict(validation.history_descriptors), n_modes
    )
    residual = train.targets - train_base
    train_events = train.histories.reshape(-1, train.histories.shape[-1])
    maximum = min(len(train_events), 100_000)
    if len(train_events) > maximum:
        rng = np.random.default_rng(int(seed))
        train_events = train_events[rng.choice(len(train_events), maximum, replace=False)]
    best = None
    for dimension in map(int, dimension_grid):
        pca = PCA(n_components=min(dimension, train_events.shape[1]), random_state=int(seed)).fit(train_events)
        for decay in map(float, decay_grid):
            train_state = leaky_linear_state(train.histories, pca, decay)
            validation_state = leaky_linear_state(validation.histories, pca, decay)
            for alpha in map(float, alpha_grid):
                correction = Ridge(alpha=alpha).fit(train_state, residual)
                prediction = project_descriptor(
                    validation_base + correction.predict(validation_state), n_modes
                )
                score = score_predictions(
                    validation.targets,
                    prediction,
                    n_modes=n_modes,
                    n_contacts=n_contacts,
                )
                candidate = (score.composite, dimension, decay, alpha, pca, correction, score)
                if best is None or candidate[:4] < best[:4]:
                    best = candidate
    assert best is not None
    combined = concatenate_window_datasets(train, validation)
    final_base = Ridge(alpha=float(base_selected.alpha)).fit(
        combined.history_descriptors, combined.targets
    )
    combined_base = project_descriptor(
        final_base.predict(combined.history_descriptors), n_modes
    )
    events = combined.histories.reshape(-1, combined.histories.shape[-1])
    if len(events) > maximum:
        rng = np.random.default_rng(int(seed))
        events = events[rng.choice(len(events), maximum, replace=False)]
    final_pca = PCA(
        n_components=min(int(best[1]), events.shape[1]), random_state=int(seed)
    ).fit(events)
    state = leaky_linear_state(combined.histories, final_pca, float(best[2]))
    final_correction = Ridge(alpha=float(best[3])).fit(
        state, combined.targets - combined_base
    )
    return ResidualLinearStateResult(
        base_model=final_base,
        pca=final_pca,
        decay=float(best[2]),
        correction_model=final_correction,
        base_alpha=float(base_selected.alpha),
        correction_alpha=float(best[3]),
        dimension=int(best[1]),
        n_modes=int(n_modes),
        validation_score=best[6],
    )


def shuffled_histories(dataset: FutureWindowDataset, seed: int) -> FutureWindowDataset:
    rng = np.random.default_rng(int(seed))
    histories = np.asarray(dataset.histories).copy()
    for row in range(len(histories)):
        histories[row] = histories[row, rng.permutation(histories.shape[1])]
    return FutureWindowDataset(
        histories=histories,
        recent_descriptors=dataset.recent_descriptors,
        history_descriptors=dataset.history_descriptors,
        targets=dataset.targets,
        last_mode=dataset.last_mode,
        source_ids=dataset.source_ids,
        history_start=dataset.history_start,
        history_stop=dataset.history_stop,
        target_start=dataset.target_start,
        target_stop=dataset.target_stop,
        history_event_indices=dataset.history_event_indices,
        target_event_indices=dataset.target_event_indices,
    )


def circularly_shift_targets(dataset: FutureWindowDataset, shift: int | None = None) -> FutureWindowDataset:
    """Shift targets only within each source, preserving source-level marginals."""
    target = np.asarray(dataset.targets).copy()
    for source_id in np.unique(dataset.source_ids):
        idx = np.flatnonzero(dataset.source_ids == source_id)
        if len(idx) > 1:
            amount = max(1, len(idx) // 2) if shift is None else int(shift)
            target[idx] = target[np.roll(idx, amount)]
    return FutureWindowDataset(
        histories=dataset.histories,
        recent_descriptors=dataset.recent_descriptors,
        history_descriptors=dataset.history_descriptors,
        targets=target,
        last_mode=dataset.last_mode,
        source_ids=dataset.source_ids,
        history_start=dataset.history_start,
        history_stop=dataset.history_stop,
        target_start=dataset.target_start,
        target_stop=dataset.target_stop,
        history_event_indices=dataset.history_event_indices,
        target_event_indices=dataset.target_event_indices,
    )


def future_window_split_half_reliability(
    dataset: FutureWindowDataset,
    modes: np.ndarray,
    local_rank: np.ndarray,
    participation: np.ndarray,
    encoder: StableTemplateEncoder,
    *,
    repeats: int = 10,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Estimate how much future-window variation exceeds finite-event noise."""
    if dataset.target_event_indices.shape[1] < 4:
        raise ValueError("future windows need at least four events for reliability")
    rng = np.random.default_rng(int(seed))
    n_modes = encoder.n_modes
    n_contacts = np.asarray(local_rank).shape[1]
    slices = {
        "occupancy": slice(0, n_modes),
        "rank": slice(n_modes, n_modes + n_contacts),
        "participation": slice(n_modes + n_contacts, n_modes + 2 * n_contacts),
    }
    correlations = {name: [] for name in slices}
    reliabilities = {name: [] for name in slices}
    for _ in range(int(repeats)):
        left = []
        right = []
        for target in dataset.target_event_indices:
            order = rng.permutation(target)
            half = len(order) // 2
            left.append(
                repertoire_descriptor(order[:half], modes, local_rank, participation, encoder)
            )
            right.append(
                repertoire_descriptor(order[half : 2 * half], modes, local_rank, participation, encoder)
            )
        left = np.stack(left)
        right = np.stack(right)
        for name, section in slices.items():
            a = left[:, section].ravel()
            b = right[:, section].ravel()
            corr = float(spearmanr(a, b).statistic) if np.std(a) > EPS and np.std(b) > EPS else float("nan")
            full_variance = float(np.var(dataset.targets[:, section]))
            full_sample_noise = float(np.mean((a - b) ** 2) / 4.0)
            reliability = (
                float(1.0 - full_sample_noise / full_variance)
                if full_variance > EPS else float("nan")
            )
            correlations[name].append(corr)
            reliabilities[name].append(reliability)
    return {
        name: {
            "split_half_spearman_median": float(np.nanmedian(correlations[name])),
            "variance_reliability_median": float(np.nanmedian(reliabilities[name])),
        }
        for name in slices
    }


def verify_dataset_contract(
    dataset: FutureWindowDataset,
    horizon: int,
    source_ids: np.ndarray | None = None,
) -> Mapping[str, bool]:
    same_source = len(dataset) == len(dataset.source_ids)
    if source_ids is not None:
        source = np.asarray(source_ids)
        same_source = bool(
            np.all(source[dataset.history_event_indices] == dataset.source_ids[:, None])
            and np.all(source[dataset.target_event_indices] == dataset.source_ids[:, None])
        )
    nonoverlap = True
    for source_id in np.unique(dataset.source_ids):
        idx = np.flatnonzero(dataset.source_ids == source_id)
        order = idx[np.argsort(dataset.target_start[idx])]
        if len(order) > 1:
            for left, right in zip(order[:-1], order[1:]):
                if np.intersect1d(
                    dataset.target_event_indices[left], dataset.target_event_indices[right]
                ).size:
                    nonoverlap = False
    return {
        "history_strictly_precedes_target": bool(
            np.all(dataset.history_event_indices[:, -1] < dataset.target_event_indices[:, 0])
        ),
        "history_and_target_same_source": same_source,
        "target_windows_expected_size": bool(dataset.target_event_indices.shape[1] == int(horizon)),
        "formal_targets_nonoverlapping": bool(nonoverlap),
    }
