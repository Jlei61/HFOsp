"""Trainable event-level recurrent models for Topic 5 v2.5.

One recurrent step is one complete interictal event.  The models predict a
residual correction to a matched non-recurrent future-repertoire forecast.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge

from src.topic5_stable_repertoire_event_history_v2_4 import (
    EPS,
    EventHistoryDataset,
    FamilyScales,
    FeatureRidgeModel,
    V24Score,
    build_event_history_dataset,
    feature_matrix,
    score_v24,
)
from src.topic5_stable_repertoire_event_rnn import project_descriptor

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


@dataclass(frozen=True)
class WindowBalancedPartition:
    train_sources: np.ndarray
    validation_sources: np.ndarray
    test_sources: np.ndarray
    strategy: str
    formal_window_counts: dict[str, int]
    source_window_counts: dict[str, int]


@dataclass(frozen=True)
class RecurrentProfile:
    cell: str
    hidden_size: int
    num_layers: int
    block_size: int
    normalization: str
    input_layer_norm: bool
    hidden_layer_norm: bool
    optimizer: str
    learning_rate: float
    batch_size: int
    weight_decay: float
    gradient_clip: float
    dropout: float = 0.0
    participation_weight: float = 0.25


@dataclass
class TrainingTrace:
    train_loss: list[float]
    validation_propagation: list[float]
    validation_recruitment: list[float]
    gradient_norm_mean: list[float]
    gradient_norm_max: list[float]
    clipped_fraction: list[float]
    best_epoch: int
    stopped_epoch: int
    finite: bool
    validation_baseline_propagation: float | None
    best_is_untrained_baseline: bool


@dataclass
class TrainableRNNResult:
    model: object
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    baseline: FeatureRidgeModel
    profile: RecurrentProfile
    trace: TrainingTrace
    validation_score: V24Score | None
    n_modes: int
    n_contacts: int
    n_parameters: int

    def predict(self, dataset: EventHistoryDataset, batch_size: int = 512) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is required for recurrent prediction")
        base = self.baseline.predict(dataset)
        values = prepare_recurrent_inputs(
            dataset.histories,
            block_size=self.profile.block_size,
            mean=self.feature_mean,
            scale=self.feature_scale,
        )
        tensor = torch.as_tensor(values, dtype=torch.float32)
        correction = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(tensor), int(batch_size)):
                correction.append(
                    self.model(tensor[start : start + int(batch_size)]).cpu().numpy()
                )
        return project_descriptor(base + np.concatenate(correction), self.n_modes)


def formal_window_count(n_events: int, history_length: int, horizon: int) -> int:
    available = int(n_events) - int(history_length) - int(horizon)
    return max(0, 1 + available // int(horizon))


def _ordered_sources(
    source_ids: np.ndarray,
    event_time: np.ndarray,
    eligible_indices: Sequence[int],
) -> list[object]:
    source = np.asarray(source_ids)
    times = np.asarray(event_time, float)
    eligible = np.asarray(eligible_indices, int)
    unique = np.unique(source[eligible])
    return sorted(
        [value.item() if hasattr(value, "item") else value for value in unique],
        key=lambda value: float(np.min(times[eligible[source[eligible] == value]])),
    )


def window_balanced_source_partition(
    source_ids: np.ndarray,
    event_time: np.ndarray,
    eligible_indices: Sequence[int],
    *,
    history_length: int,
    horizon: int,
) -> WindowBalancedPartition:
    """Chronological source-disjoint split balanced by usable formal windows.

    Two-source patients use the earlier source for training and the later source
    for test.  Their hyperparameters must be frozen globally because no local
    validation source exists.
    """
    source = np.asarray(source_ids)
    eligible = np.asarray(eligible_indices, int)
    ordered = _ordered_sources(source, event_time, eligible)
    counts = {
        str(value): formal_window_count(
            int(np.sum(source[eligible] == value)), history_length, horizon
        )
        for value in ordered
    }
    usable = sum(value > 0 for value in counts.values())
    if usable < 2:
        raise ValueError("fewer than two chronological sources contain prediction windows")
    if len(ordered) == 2:
        train, validation, test = ordered[:1], [], ordered[1:]
        split_counts = {
            "train": counts[str(train[0])],
            "validation": 0,
            "test": counts[str(test[0])],
        }
        return WindowBalancedPartition(
            np.asarray(train, dtype=source.dtype),
            np.asarray(validation, dtype=source.dtype),
            np.asarray(test, dtype=source.dtype),
            "two_source_global_hyperparameters",
            split_counts,
            counts,
        )

    best = None
    total = float(sum(counts.values()))
    for first in range(1, len(ordered) - 1):
        for second in range(first + 1, len(ordered)):
            groups = (ordered[:first], ordered[first:second], ordered[second:])
            values = tuple(sum(counts[str(item)] for item in group) for group in groups)
            if min(values) < 1:
                continue
            fractions = np.asarray(values, float) / max(total, 1.0)
            error = float(np.sum((fractions - np.asarray([0.6, 0.2, 0.2])) ** 2))
            candidate = (error, -min(values), first, second, values)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("no chronological source cut gives train/validation/test windows")
    first, second, values = int(best[2]), int(best[3]), best[4]
    return WindowBalancedPartition(
        np.asarray(ordered[:first], dtype=source.dtype),
        np.asarray(ordered[first:second], dtype=source.dtype),
        np.asarray(ordered[second:], dtype=source.dtype),
        "window_balanced_source_disjoint",
        {"train": int(values[0]), "validation": int(values[1]), "test": int(values[2])},
        counts,
    )


def partition_indices(
    source_ids: np.ndarray,
    eligible_indices: Sequence[int],
    partition: WindowBalancedPartition,
    split: str,
) -> np.ndarray:
    selected = {
        "train": partition.train_sources,
        "validation": partition.validation_sources,
        "test": partition.test_sources,
    }[split]
    eligible = np.asarray(eligible_indices, int)
    source = np.asarray(source_ids)
    return eligible[np.isin(source[eligible], selected)]


def aggregate_event_steps(histories: np.ndarray, block_size: int) -> np.ndarray:
    values = np.asarray(histories, float)
    block = int(block_size)
    if values.ndim != 3:
        raise ValueError("histories must be sample x event x token")
    if block < 1 or values.shape[1] % block:
        raise ValueError("block_size must be a positive divisor of history length")
    if block == 1:
        return values.copy()
    return values.reshape(values.shape[0], values.shape[1] // block, block, values.shape[2]).mean(axis=2)


def fit_input_normalization(values: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    flattened = np.asarray(values, float).reshape(-1, values.shape[-1])
    if mode == "none":
        center = np.zeros(values.shape[-1], float)
        scale = np.ones(values.shape[-1], float)
    elif mode == "zscore":
        center = np.mean(flattened, axis=0)
        scale = np.std(flattened, axis=0)
    elif mode == "robust":
        center = np.median(flattened, axis=0)
        q25, q75 = np.quantile(flattened, [0.25, 0.75], axis=0)
        scale = (q75 - q25) / 1.349
    else:
        raise ValueError(f"unknown normalization: {mode}")
    scale = np.where(np.asarray(scale) > EPS, scale, 1.0)
    return np.asarray(center, float), np.asarray(scale, float)


def prepare_recurrent_inputs(
    histories: np.ndarray,
    *,
    block_size: int,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = aggregate_event_steps(histories, block_size)
    return (values - np.asarray(mean)[None, None, :]) / np.asarray(scale)[None, None, :]


def fit_fixed_feature_baseline(
    dataset: EventHistoryDataset,
    *,
    feature_name: str,
    decay: float | None,
    alpha: float,
    rank_prior: np.ndarray,
    n_modes: int,
) -> FeatureRidgeModel:
    values = feature_matrix(dataset, feature_name, rank_prior, n_modes, decay=decay)
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale > EPS, scale, 1.0)
    ridge = Ridge(alpha=float(alpha)).fit((values - mean) / scale, dataset.targets)
    placeholder = V24Score(*([float("nan")] * 9))
    return FeatureRidgeModel(
        feature_name=feature_name,
        decay=decay,
        alpha=float(alpha),
        feature_mean=mean,
        feature_scale=scale,
        ridge=ridge,
        n_modes=int(n_modes),
        rank_prior=np.asarray(rank_prior, float),
        validation_score=placeholder,
    )


if nn is not None:
    class ResidualEventRNN(nn.Module):
        """Causal trainable recurrent correction to a matched baseline."""

        def __init__(self, input_dim: int, target_dim: int, profile: RecurrentProfile):
            super().__init__()
            self.input_norm = nn.LayerNorm(input_dim) if profile.input_layer_norm else nn.Identity()
            cell = profile.cell.lower()
            classes = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}
            if cell not in classes:
                raise ValueError(f"unknown recurrent cell: {profile.cell}")
            kwargs = {
                "input_size": int(input_dim),
                "hidden_size": int(profile.hidden_size),
                "num_layers": int(profile.num_layers),
                "batch_first": True,
                "dropout": float(profile.dropout) if int(profile.num_layers) > 1 else 0.0,
                "bidirectional": False,
            }
            if cell == "rnn":
                kwargs["nonlinearity"] = "tanh"
            self.recurrent = classes[cell](**kwargs)
            self.hidden_norm = (
                nn.LayerNorm(int(profile.hidden_size)) if profile.hidden_layer_norm else nn.Identity()
            )
            self.readout = nn.Linear(int(profile.hidden_size), int(target_dim))
            nn.init.zeros_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)

        def forward(self, histories):
            output, _ = self.recurrent(self.input_norm(histories))
            return self.readout(self.hidden_norm(output[:, -1]))
else:  # pragma: no cover
    class ResidualEventRNN:  # type: ignore[no-redef]
        pass


def set_recurrent_seed(seed: int) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for recurrent training")
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)


def make_optimizer(model: object, profile: RecurrentProfile):
    name = profile.optimizer.lower()
    kwargs = {
        "lr": float(profile.learning_rate),
        "weight_decay": float(profile.weight_decay),
    }
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), **kwargs)
    raise ValueError(f"unknown optimizer: {profile.optimizer}")


def recurrent_family_loss(
    prediction,
    target,
    *,
    n_modes: int,
    n_contacts: int,
    scales: FamilyScales,
    participation_weight: float,
):
    occ = torch.mean((prediction[:, :n_modes] - target[:, :n_modes]) ** 2) / float(scales.occupancy)
    rank = torch.mean(
        (prediction[:, n_modes : n_modes + n_contacts] - target[:, n_modes : n_modes + n_contacts]) ** 2
    ) / float(scales.rank)
    part = torch.mean(
        (prediction[:, n_modes + n_contacts :] - target[:, n_modes + n_contacts :]) ** 2
    ) / float(scales.participation)
    return (occ + rank) / 2.0 + float(participation_weight) * part


def _global_gradient_norm(parameters: Iterable[object]) -> float:
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            squares.append(torch.sum(parameter.grad.detach() ** 2))
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.sum(torch.stack(squares))).cpu())


def fit_trainable_residual_rnn(
    train: EventHistoryDataset,
    *,
    baseline: FeatureRidgeModel,
    profile: RecurrentProfile,
    scales: FamilyScales,
    n_modes: int,
    n_contacts: int,
    seed: int,
    maximum_epochs: int,
    patience: int,
    minimum_epochs: int,
    validation: EventHistoryDataset | None = None,
    fixed_epochs: int | None = None,
) -> TrainableRNNResult:
    if torch is None:
        raise RuntimeError("PyTorch is required for recurrent training")
    set_recurrent_seed(seed)
    blocked = aggregate_event_steps(train.histories, profile.block_size)
    mean, scale = fit_input_normalization(blocked, profile.normalization)
    train_x = (blocked - mean[None, None, :]) / scale[None, None, :]
    train_base = baseline.predict(train)
    model = ResidualEventRNN(train_x.shape[-1], train.targets.shape[-1], profile)
    optimizer = make_optimizer(model, profile)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(train_x, dtype=torch.float32),
            torch.as_tensor(train_base, dtype=torch.float32),
            torch.as_tensor(train.targets, dtype=torch.float32),
        ),
        batch_size=min(int(profile.batch_size), len(train)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    validation_x = validation_base = None
    if validation is not None:
        validation_x = prepare_recurrent_inputs(
            validation.histories,
            block_size=profile.block_size,
            mean=mean,
            scale=scale,
        )
        validation_base = baseline.predict(validation)

    epochs = int(fixed_epochs if fixed_epochs is not None else maximum_epochs)
    best_state = None
    best_score = float("inf")
    best_breakdown = None
    best_epoch = -1
    stale = 0
    trace = TrainingTrace(
        train_loss=[],
        validation_propagation=[],
        validation_recruitment=[],
        gradient_norm_mean=[],
        gradient_norm_max=[],
        clipped_fraction=[],
        best_epoch=-1,
        stopped_epoch=-1,
        finite=True,
        validation_baseline_propagation=None,
        best_is_untrained_baseline=False,
    )
    if validation is not None:
        initial_score = score_v24(
            validation.targets,
            project_descriptor(validation_base.copy(), n_modes),
            n_modes=n_modes,
            n_contacts=n_contacts,
            scales=scales,
        )
        best_score = float(initial_score.propagation)
        best_breakdown = initial_score
        best_state = deepcopy(model.state_dict())
        trace.validation_baseline_propagation = float(initial_score.propagation)
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        norms = []
        clipped = []
        for histories, base, target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = base + model(histories)
            loss = recurrent_family_loss(
                prediction,
                target,
                n_modes=n_modes,
                n_contacts=n_contacts,
                scales=scales,
                participation_weight=profile.participation_weight,
            )
            if not torch.isfinite(loss):
                trace.finite = False
                raise RuntimeError("non-finite recurrent training loss")
            loss.backward()
            norm = _global_gradient_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(profile.gradient_clip))
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
            norms.append(norm)
            clipped.append(float(norm > float(profile.gradient_clip)))
        trace.train_loss.append(float(np.mean(batch_losses)))
        trace.gradient_norm_mean.append(float(np.mean(norms)))
        trace.gradient_norm_max.append(float(np.max(norms)))
        trace.clipped_fraction.append(float(np.mean(clipped)))

        if validation is not None:
            model.eval()
            with torch.no_grad():
                correction = model(torch.as_tensor(validation_x, dtype=torch.float32)).cpu().numpy()
            prediction = project_descriptor(validation_base + correction, n_modes)
            score = score_v24(
                validation.targets,
                prediction,
                n_modes=n_modes,
                n_contacts=n_contacts,
                scales=scales,
            )
            trace.validation_propagation.append(float(score.propagation))
            trace.validation_recruitment.append(float(score.recruitment))
            if np.isfinite(score.propagation) and score.propagation < best_score - 1e-7:
                best_score = float(score.propagation)
                best_state = deepcopy(model.state_dict())
                best_breakdown = score
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if epoch + 1 >= int(minimum_epochs) and stale >= int(patience):
                break
    trace.stopped_epoch = int(epoch)
    if validation is not None:
        if best_state is None:
            raise RuntimeError("no finite recurrent validation checkpoint")
        model.load_state_dict(best_state)
        trace.best_epoch = int(best_epoch)
        trace.best_is_untrained_baseline = bool(best_epoch == -1)
    else:
        trace.best_epoch = int(epoch)
    finite_parameters = all(torch.isfinite(parameter).all() for parameter in model.parameters())
    trace.finite = bool(trace.finite and finite_parameters)
    return TrainableRNNResult(
        model=model,
        feature_mean=mean,
        feature_scale=scale,
        baseline=baseline,
        profile=profile,
        trace=trace,
        validation_score=best_breakdown,
        n_modes=int(n_modes),
        n_contacts=int(n_contacts),
        n_parameters=int(sum(parameter.numel() for parameter in model.parameters())),
    )


def profile_from_mapping(values: Mapping[str, object]) -> RecurrentProfile:
    allowed = set(RecurrentProfile.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown recurrent profile fields: {sorted(unknown)}")
    return RecurrentProfile(**{key: values[key] for key in allowed if key in values})


def trace_to_dict(trace: TrainingTrace) -> dict[str, object]:
    return asdict(trace)
