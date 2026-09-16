"""Continuous stateful event-sequence RNN for Topic 5 v2.6.

One forward step is one complete interictal event. Hidden state is carried
between TBPTT chunks and reset only at source/session boundaries.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge
import torch
from torch import nn

from src.topic5_stable_repertoire_event_history_v2_4 import (
    EPS,
    FamilyScales,
    V24Score,
    score_v24,
)
from src.topic5_stable_repertoire_event_rnn import (
    StableTemplateEncoder,
    project_descriptor,
)


@dataclass(frozen=True)
class StatefulEventSequence:
    source_id: object
    tokens: np.ndarray
    targets: np.ndarray
    valid_mask: np.ndarray
    formal_mask: np.ndarray
    event_indices: np.ndarray
    target_event_indices: np.ndarray


@dataclass(frozen=True)
class StatefulProfile:
    cell: str
    hidden_size: int
    num_layers: int
    normalization: str
    input_layer_norm: bool
    hidden_layer_norm: bool
    optimizer: str
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    tbptt_length: int
    update_chunks: int
    dropout: float = 0.0
    participation_weight: float = 0.25


@dataclass
class StatefulTrainingTrace:
    train_loss: list[float]
    validation_trained_propagation: list[float]
    validation_nested_propagation: list[float]
    gradient_norm_mean: list[float]
    gradient_norm_max: list[float]
    clipped_fraction: list[float]
    state_norm_mean: list[float]
    best_trained_epoch: int
    best_nested_epoch: int
    stopped_epoch: int
    finite: bool


@dataclass
class StatefulFitResult:
    trained_model: nn.Module
    nested_model: nn.Module
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    profile: StatefulProfile
    trace: StatefulTrainingTrace
    trained_validation_score: V24Score
    nested_validation_score: V24Score
    n_modes: int
    n_contacts: int
    n_parameters: int
    carry_state: bool

    def predict(
        self,
        sequences: Sequence[StatefulEventSequence],
        *,
        checkpoint: str = "trained",
        formal: bool = True,
        return_states: bool = False,
    ):
        model = {
            "trained": self.trained_model,
            "nested": self.nested_model,
        }.get(checkpoint)
        if model is None:
            raise ValueError("checkpoint must be trained or nested")
        return rollout_sequences(
            model,
            sequences,
            mean=self.feature_mean,
            scale=self.feature_scale,
            chunk_length=self.profile.tbptt_length,
            carry_state=self.carry_state,
            formal=formal,
            return_states=return_states,
        )


@dataclass(frozen=True)
class ContinuousEWMARidge:
    decay: float
    alpha: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    ridge: Ridge
    n_modes: int

    def predict(self, sequences, *, formal: bool = True):
        features, targets, metadata = collect_ewma_samples(
            sequences, decay=self.decay, formal=formal
        )
        estimate = self.ridge.predict(
            (features - self.feature_mean[None, :]) / self.feature_scale[None, :]
        )
        return project_descriptor(estimate, self.n_modes), targets, metadata


def _future_descriptors(
    modes: np.ndarray,
    rank: np.ndarray,
    participation: np.ndarray,
    encoder: StableTemplateEncoder,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(modes, int)
    values = np.asarray(rank, float)
    part = np.asarray(participation, bool)
    n_events, n_contacts = part.shape
    future = int(horizon)
    targets = np.full((n_events, encoder.n_modes + 2 * n_contacts), np.nan, float)
    target_indices = np.full((n_events, future), -1, int)
    if n_events <= future:
        return targets, target_indices

    mode_values = np.eye(encoder.n_modes, dtype=float)[labels]
    valid = part & np.isfinite(values)
    rank_total = np.where(valid, values, 0.0)
    part_float = part.astype(float)

    def window_sum(array):
        prefix = np.concatenate(
            [np.zeros((1, array.shape[1]), float), np.cumsum(array, axis=0)],
            axis=0,
        )
        anchors = np.arange(n_events - future, dtype=int)
        start = anchors + 1
        stop = start + future
        return prefix[stop] - prefix[start]

    mode_sum = window_sum(mode_values)
    count = window_sum(valid.astype(float))
    total = window_sum(rank_total)
    participation_sum = window_sum(part_float)
    occupancy = mode_sum / future
    rank_mean = (
        total + 2.0 * encoder.rank_prior[None, :]
    ) / (count + 2.0)
    participation_mean = (participation_sum + 1.0) / (future + 2.0)
    available = n_events - future
    targets[:available] = np.concatenate(
        [occupancy, rank_mean, participation_mean], axis=1
    )
    target_indices[:available] = (
        np.arange(available)[:, None] + 1 + np.arange(future)[None, :]
    )
    return targets, target_indices


def build_stateful_sequences(
    tokens: np.ndarray,
    modes: np.ndarray,
    rank: np.ndarray,
    participation: np.ndarray,
    encoder: StableTemplateEncoder,
    sequences: Mapping[object, np.ndarray],
    *,
    horizon: int,
    warmup_events: int,
) -> list[StatefulEventSequence]:
    output = []
    for source_id in sorted(sequences, key=str):
        event_indices = np.asarray(sequences[source_id], int)
        local_targets, local_target_positions = _future_descriptors(
            np.asarray(modes)[event_indices],
            np.asarray(rank)[event_indices],
            np.asarray(participation)[event_indices],
            encoder,
            int(horizon),
        )
        valid = np.all(np.isfinite(local_targets), axis=1)
        valid[: max(0, int(warmup_events) - 1)] = False
        formal = np.zeros(len(event_indices), bool)
        anchors = np.arange(
            max(0, int(warmup_events) - 1),
            max(0, len(event_indices) - int(horizon)),
            int(horizon),
            dtype=int,
        )
        formal[anchors] = valid[anchors]
        mapped_targets = np.full_like(local_target_positions, -1)
        usable = local_target_positions >= 0
        mapped_targets[usable] = event_indices[local_target_positions[usable]]
        output.append(
            StatefulEventSequence(
                source_id=source_id,
                tokens=np.asarray(tokens)[event_indices].astype(np.float32),
                targets=local_targets.astype(np.float32),
                valid_mask=valid,
                formal_mask=formal,
                event_indices=event_indices,
                target_event_indices=mapped_targets,
            )
        )
    if not output:
        raise ValueError("no stateful source sequences")
    if not any(np.any(item.valid_mask) for item in output):
        raise ValueError("no causal future-repertoire targets")
    return output


def fit_normalization(sequences, mode: str):
    values = np.concatenate([item.tokens for item in sequences], axis=0).astype(float)
    if mode == "none":
        center = np.zeros(values.shape[1], float)
        scale = np.ones(values.shape[1], float)
    elif mode == "zscore":
        center = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
    elif mode == "robust":
        center = np.median(values, axis=0)
        q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
        scale = (q75 - q25) / 1.349
    else:
        raise ValueError(f"unknown normalization mode: {mode}")
    scale = np.where(scale > EPS, scale, 1.0)
    return np.asarray(center, float), np.asarray(scale, float)


def family_scales_from_sequences(sequences, *, n_modes: int, n_contacts: int):
    targets = np.concatenate(
        [item.targets[item.valid_mask] for item in sequences], axis=0
    )
    sections = (
        targets[:, :n_modes],
        targets[:, n_modes : n_modes + n_contacts],
        targets[:, n_modes + n_contacts :],
    )
    values = [max(float(np.mean(np.var(section, axis=0))), EPS) for section in sections]
    return FamilyScales(*values)


def mean_future_descriptor(sequences):
    targets = np.concatenate(
        [item.targets[item.valid_mask] for item in sequences], axis=0
    )
    return np.mean(targets, axis=0)


def _inverse_output_bias(mean: np.ndarray, n_modes: int) -> np.ndarray:
    values = np.clip(np.asarray(mean, float), 1e-5, 1.0 - 1e-5)
    occupancy = values[:n_modes]
    occupancy = occupancy / np.sum(occupancy)
    bias = np.empty_like(values)
    bias[:n_modes] = np.log(occupancy)
    bias[n_modes:] = np.log(values[n_modes:] / (1.0 - values[n_modes:]))
    return bias


class StatefulEventRNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        n_modes: int,
        profile: StatefulProfile,
        output_mean: np.ndarray,
    ):
        super().__init__()
        self.n_modes = int(n_modes)
        self.profile = profile
        self.input_norm = (
            nn.LayerNorm(int(input_dim)) if profile.input_layer_norm else nn.Identity()
        )
        classes = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}
        cell = profile.cell.lower()
        if cell not in classes:
            raise ValueError(f"unknown recurrent cell: {profile.cell}")
        kwargs = dict(
            input_size=int(input_dim),
            hidden_size=int(profile.hidden_size),
            num_layers=int(profile.num_layers),
            batch_first=True,
            dropout=(float(profile.dropout) if int(profile.num_layers) > 1 else 0.0),
            bidirectional=False,
        )
        if cell == "rnn":
            kwargs["nonlinearity"] = "tanh"
        self.recurrent = classes[cell](**kwargs)
        self.hidden_norm = (
            nn.LayerNorm(int(profile.hidden_size))
            if profile.hidden_layer_norm
            else nn.Identity()
        )
        self.readout = nn.Linear(int(profile.hidden_size), int(target_dim))
        nn.init.zeros_(self.readout.weight)
        with torch.no_grad():
            self.readout.bias.copy_(
                torch.as_tensor(
                    _inverse_output_bias(output_mean, self.n_modes), dtype=torch.float32
                )
            )

    def forward(self, events, hidden=None):
        recurrent, hidden = self.recurrent(self.input_norm(events), hidden)
        states = self.hidden_norm(recurrent)
        logits = self.readout(states)
        occupancy = torch.softmax(logits[..., : self.n_modes], dim=-1)
        remainder = torch.sigmoid(logits[..., self.n_modes :])
        return torch.cat([occupancy, remainder], dim=-1), hidden, states


def detach_hidden(hidden):
    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        return tuple(value.detach() for value in hidden)
    return hidden.detach()


def make_optimizer(model, profile: StatefulProfile):
    kwargs = dict(
        lr=float(profile.learning_rate), weight_decay=float(profile.weight_decay)
    )
    name = profile.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), **kwargs)
    raise ValueError(f"unknown optimizer: {profile.optimizer}")


def stateful_family_loss(
    prediction,
    target,
    *,
    n_modes: int,
    n_contacts: int,
    scales: FamilyScales,
    participation_weight: float,
):
    occupancy = torch.mean((prediction[:, :n_modes] - target[:, :n_modes]) ** 2)
    rank = torch.mean(
        (
            prediction[:, n_modes : n_modes + n_contacts]
            - target[:, n_modes : n_modes + n_contacts]
        )
        ** 2
    )
    participation = torch.mean(
        (
            prediction[:, n_modes + n_contacts :]
            - target[:, n_modes + n_contacts :]
        )
        ** 2
    )
    return (
        0.5 * (occupancy / scales.occupancy + rank / scales.rank)
        + float(participation_weight) * participation / scales.participation
    )


def _gradient_norm(parameters: Iterable[torch.Tensor]) -> float:
    squares = [
        torch.sum(parameter.grad.detach() ** 2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.sqrt(torch.sum(torch.stack(squares)))) if squares else 0.0


def rollout_sequences(
    model,
    sequences,
    *,
    mean,
    scale,
    chunk_length: int,
    carry_state: bool,
    formal: bool,
    return_states: bool,
):
    predictions = []
    targets = []
    metadata = []
    state_values = []
    model.eval()
    with torch.no_grad():
        for sequence in sequences:
            hidden = None
            source_predictions = []
            source_states = []
            for start in range(0, len(sequence.tokens), int(chunk_length)):
                stop = min(start + int(chunk_length), len(sequence.tokens))
                values = (
                    sequence.tokens[start:stop] - np.asarray(mean)[None, :]
                ) / np.asarray(scale)[None, :]
                output, next_hidden, states = model(
                    torch.as_tensor(values[None, :, :], dtype=torch.float32),
                    hidden if carry_state else None,
                )
                hidden = detach_hidden(next_hidden) if carry_state else None
                source_predictions.append(output[0].cpu().numpy())
                source_states.append(states[0].cpu().numpy())
            source_predictions = np.concatenate(source_predictions, axis=0)
            source_states = np.concatenate(source_states, axis=0)
            mask = sequence.formal_mask if formal else sequence.valid_mask
            selected = np.flatnonzero(mask)
            predictions.append(source_predictions[selected])
            targets.append(sequence.targets[selected])
            state_values.append(source_states[selected])
            metadata.extend(
                {
                    "source_id": str(sequence.source_id),
                    "event_index": int(sequence.event_indices[row]),
                    "target_event_indices": sequence.target_event_indices[row].tolist(),
                }
                for row in selected
            )
    result = (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        metadata,
    )
    if return_states:
        return (*result, np.concatenate(state_values, axis=0))
    return result


def score_model(
    model,
    sequences,
    *,
    mean,
    scale,
    profile,
    carry_state,
    n_modes,
    n_contacts,
    scales,
    formal,
):
    prediction, target, _ = rollout_sequences(
        model,
        sequences,
        mean=mean,
        scale=scale,
        chunk_length=profile.tbptt_length,
        carry_state=carry_state,
        formal=formal,
        return_states=False,
    )
    return score_v24(
        target,
        prediction,
        n_modes=n_modes,
        n_contacts=n_contacts,
        scales=scales,
    )


def set_seed(seed: int):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)


def fit_stateful_event_rnn(
    train_sequences,
    validation_sequences,
    *,
    profile: StatefulProfile,
    scales: FamilyScales,
    n_modes: int,
    n_contacts: int,
    seed: int,
    maximum_epochs: int,
    minimum_epochs: int,
    patience: int,
    carry_state: bool = True,
) -> StatefulFitResult:
    set_seed(seed)
    mean, scale = fit_normalization(train_sequences, profile.normalization)
    output_mean = mean_future_descriptor(train_sequences)
    input_dim = train_sequences[0].tokens.shape[1]
    target_dim = train_sequences[0].targets.shape[1]
    model = StatefulEventRNN(
        input_dim, target_dim, n_modes, profile, output_mean
    )
    optimizer = make_optimizer(model, profile)
    initial_state = deepcopy(model.state_dict())
    initial_score = score_model(
        model,
        validation_sequences,
        mean=mean,
        scale=scale,
        profile=profile,
        carry_state=carry_state,
        n_modes=n_modes,
        n_contacts=n_contacts,
        scales=scales,
        formal=False,
    )
    best_trained_score = None
    best_trained_state = None
    best_trained_epoch = -1
    best_nested_score = initial_score
    best_nested_state = deepcopy(initial_state)
    best_nested_epoch = -1
    stale = 0
    trace = StatefulTrainingTrace(
        train_loss=[],
        validation_trained_propagation=[],
        validation_nested_propagation=[],
        gradient_norm_mean=[],
        gradient_norm_max=[],
        clipped_fraction=[],
        state_norm_mean=[],
        best_trained_epoch=-1,
        best_nested_epoch=-1,
        stopped_epoch=-1,
        finite=True,
    )
    rng = np.random.default_rng(int(seed))
    for epoch in range(int(maximum_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        losses = []
        state_norms = []
        gradient_norms = []
        clipped = []
        for source_index in rng.permutation(len(train_sequences)):
            sequence = train_sequences[int(source_index)]
            hidden = None
            for start in range(0, len(sequence.tokens), int(profile.tbptt_length)):
                stop = min(start + int(profile.tbptt_length), len(sequence.tokens))
                values = (
                    sequence.tokens[start:stop] - mean[None, :]
                ) / scale[None, :]
                prediction, next_hidden, states = model(
                    torch.as_tensor(values[None, :, :], dtype=torch.float32),
                    hidden if carry_state else None,
                )
                hidden = detach_hidden(next_hidden) if carry_state else None
                state_norms.append(
                    float(torch.linalg.vector_norm(states.detach(), dim=-1).mean())
                )
                mask = sequence.valid_mask[start:stop]
                if np.any(mask):
                    selected = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long)
                    target = torch.as_tensor(
                        sequence.targets[start:stop][mask], dtype=torch.float32
                    )
                    loss = stateful_family_loss(
                        prediction[0].index_select(0, selected),
                        target,
                        n_modes=n_modes,
                        n_contacts=n_contacts,
                        scales=scales,
                        participation_weight=profile.participation_weight,
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite stateful RNN loss")
                    (loss / float(profile.update_chunks)).backward()
                    losses.append(float(loss.detach()))
                    pending += 1
                if pending >= int(profile.update_chunks):
                    norm = _gradient_norm(model.parameters())
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(profile.gradient_clip)
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    gradient_norms.append(norm)
                    clipped.append(float(norm > float(profile.gradient_clip)))
                    pending = 0
            hidden = None
        if pending:
            norm = _gradient_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(profile.gradient_clip)
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            gradient_norms.append(norm)
            clipped.append(float(norm > float(profile.gradient_clip)))

        validation_score = score_model(
            model,
            validation_sequences,
            mean=mean,
            scale=scale,
            profile=profile,
            carry_state=carry_state,
            n_modes=n_modes,
            n_contacts=n_contacts,
            scales=scales,
            formal=False,
        )
        trace.train_loss.append(float(np.mean(losses)))
        trace.validation_trained_propagation.append(validation_score.propagation)
        trace.gradient_norm_mean.append(float(np.mean(gradient_norms)))
        trace.gradient_norm_max.append(float(np.max(gradient_norms)))
        trace.clipped_fraction.append(float(np.mean(clipped)))
        trace.state_norm_mean.append(float(np.mean(state_norms)))
        if (
            best_trained_score is None
            or validation_score.propagation < best_trained_score.propagation - 1e-7
        ):
            best_trained_score = validation_score
            best_trained_state = deepcopy(model.state_dict())
            best_trained_epoch = epoch
        if validation_score.propagation < best_nested_score.propagation - 1e-7:
            best_nested_score = validation_score
            best_nested_state = deepcopy(model.state_dict())
            best_nested_epoch = epoch
            stale = 0
        else:
            stale += 1
        trace.validation_nested_propagation.append(best_nested_score.propagation)
        if epoch + 1 >= int(minimum_epochs) and stale >= int(patience):
            break
    if best_trained_state is None or best_trained_score is None:
        raise RuntimeError("no trained recurrent checkpoint")
    trace.best_trained_epoch = int(best_trained_epoch)
    trace.best_nested_epoch = int(best_nested_epoch)
    trace.stopped_epoch = int(epoch)
    finite = all(torch.isfinite(parameter).all() for parameter in model.parameters())
    trace.finite = bool(finite)
    trained_model = deepcopy(model)
    trained_model.load_state_dict(best_trained_state)
    nested_model = deepcopy(model)
    nested_model.load_state_dict(best_nested_state)
    return StatefulFitResult(
        trained_model=trained_model,
        nested_model=nested_model,
        feature_mean=mean,
        feature_scale=scale,
        profile=profile,
        trace=trace,
        trained_validation_score=best_trained_score,
        nested_validation_score=best_nested_score,
        n_modes=int(n_modes),
        n_contacts=int(n_contacts),
        n_parameters=int(sum(parameter.numel() for parameter in model.parameters())),
        carry_state=bool(carry_state),
    )


def continuous_ewma_features(sequence, decay: float):
    state = np.zeros(sequence.tokens.shape[1], float)
    mass = 0.0
    output = np.empty_like(sequence.tokens, dtype=float)
    for row, token in enumerate(sequence.tokens):
        state = float(decay) * state + np.asarray(token, float)
        mass = float(decay) * mass + 1.0
        output[row] = state / max(mass, EPS)
    return output


def collect_ewma_samples(sequences, *, decay: float, formal: bool):
    features = []
    targets = []
    metadata = []
    for sequence in sequences:
        values = continuous_ewma_features(sequence, decay)
        mask = sequence.formal_mask if formal else sequence.valid_mask
        selected = np.flatnonzero(mask)
        features.append(values[selected])
        targets.append(sequence.targets[selected])
        metadata.extend(
            {
                "source_id": str(sequence.source_id),
                "event_index": int(sequence.event_indices[row]),
                "target_event_indices": sequence.target_event_indices[row].tolist(),
            }
            for row in selected
        )
    return np.concatenate(features), np.concatenate(targets), metadata


def fit_continuous_ewma_ridge(
    train_sequences,
    *,
    decay: float,
    alpha: float,
    n_modes: int,
):
    features, targets, _ = collect_ewma_samples(
        train_sequences, decay=decay, formal=False
    )
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > EPS, scale, 1.0)
    ridge = Ridge(alpha=float(alpha)).fit((features - mean) / scale, targets)
    return ContinuousEWMARidge(
        decay=float(decay),
        alpha=float(alpha),
        feature_mean=mean,
        feature_scale=scale,
        ridge=ridge,
        n_modes=int(n_modes),
    )


def profile_from_mapping(values: Mapping[str, object]) -> StatefulProfile:
    allowed = set(StatefulProfile.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown stateful profile fields: {sorted(unknown)}")
    return StatefulProfile(**{key: values[key] for key in allowed})


def trace_to_dict(trace: StatefulTrainingTrace):
    return asdict(trace)
