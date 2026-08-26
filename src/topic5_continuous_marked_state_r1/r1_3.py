"""R1.3 fully target-trained raw observer on the exact event likelihood."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from . import contract
from .r1_2 import (
    FullAnchorDesign,
    FrozenEmbeddingStateModel,
    _flow,
    evaluate_full_t1,
)
from .r1_2b import _grouped_rows
from .raw_observation import RawAnchorReader


R1_3_REVISION = "r1_3_full_raw_temporal_exact_target_isolated_increment_v2"


class FullTargetObserverStateModel(FrozenEmbeddingStateModel):
    """Persistent state with a complete trainable observation Transformer."""

    def __init__(self, baseline_checkpoint: dict, history_dim: int,
                 n_contacts: int, adjacency: np.ndarray,
                 source_observer: nn.Module, *, use_raw: bool,
                 state_dim: int = 8):
        super().__init__(
            baseline_checkpoint, history_dim, n_contacts, adjacency,
            observation_dim=64, state_dim=state_dim,
        )
        if source_observer.raw is None:
            raise ValueError("R1.3 source observer must contain the raw stack")
        self.observer = copy.deepcopy(source_observer)
        self.use_raw = bool(use_raw)

    def observation_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.observer(**batch, use_raw=self.use_raw)


class FullAnchorObservationLoader:
    """Stream frozen full-design anchors from minute-chunked raw Zarr."""

    def __init__(self, subject: str, design: FullAnchorDesign,
                 event_times: np.ndarray, explicit_mean: np.ndarray,
                 explicit_scale: np.ndarray,
                 *, cached_explicit: np.ndarray | None = None,
                 cached_contact_mask: np.ndarray | None = None):
        self.subject = subject
        self.design = design
        self.reader = RawAnchorReader(subject, event_times)
        self.explicit_mean = np.asarray(explicit_mean, dtype=np.float32)
        self.explicit_scale = np.asarray(explicit_scale, dtype=np.float32)
        if self.explicit_mean.shape != self.explicit_scale.shape:
            raise ValueError("explicit scaler shapes disagree")
        if np.any(self.explicit_scale <= 0):
            raise ValueError("explicit scaler is invalid")
        self.cached_explicit = (
            None if cached_explicit is None
            else np.asarray(cached_explicit, dtype=np.float32)
        )
        self.cached_contact_mask = (
            None if cached_contact_mask is None
            else np.asarray(cached_contact_mask, dtype=bool)
        )
        if self.cached_explicit is not None:
            expected = (len(design.anchor_time), self.reader.raw.shape[1],
                        len(self.explicit_mean))
            if self.cached_explicit.shape != expected:
                raise ValueError("R1.3 cached explicit shape mismatch")
        if self.cached_contact_mask is not None:
            expected = (len(design.anchor_time), self.reader.raw.shape[1])
            if self.cached_contact_mask.shape != expected:
                raise ValueError("R1.3 cached contact-mask shape mismatch")

    def batch(self, anchor_ids: np.ndarray, *, device: torch.device | str,
              read_raw: bool) -> dict[str, torch.Tensor]:
        anchor_ids = np.asarray(anchor_ids, dtype=np.int64)
        if read_raw:
            observations = [
                self.reader.read(
                    float(self.design.anchor_time[index]),
                    compute_explicit=self.cached_explicit is None,
                )
                for index in anchor_ids
            ]
            if any(value is None for value in observations):
                raise RuntimeError("a frozen R1.3 full-design anchor became unreadable")
            values = [value for value in observations if value is not None]
            waveform = np.stack([value.waveform for value in values])
            sample_valid = np.stack([value.sample_valid for value in values])
        else:
            values = []
            waveform = np.zeros(
                (len(anchor_ids), self.reader.raw.shape[1], 1), dtype=np.float32
            )
            sample_valid = np.ones_like(waveform, dtype=bool)
        if self.cached_explicit is not None:
            explicit = np.asarray(self.cached_explicit[anchor_ids], dtype=np.float32)
        else:
            if not values:
                observations = [
                    self.reader.read(float(self.design.anchor_time[index]))
                    for index in anchor_ids
                ]
                if any(value is None for value in observations):
                    raise RuntimeError("a frozen R1.3 anchor became unreadable")
                values = [value for value in observations if value is not None]
            explicit = np.stack([value.explicit for value in values])
            explicit = (
                (explicit - self.explicit_mean) / self.explicit_scale
            ).astype(np.float32)
        if self.cached_contact_mask is not None:
            contact_mask = np.asarray(self.cached_contact_mask[anchor_ids], dtype=bool)
        else:
            contact_mask = np.stack([value.contact_mask for value in values])
        batch = len(anchor_ids)
        coordinates = self.reader.coordinates
        coordinate_valid = self.reader.coordinate_valid
        shaft_index = self.reader.shaft_index
        return {
            "explicit": torch.as_tensor(explicit, device=device),
            "waveform": torch.as_tensor(waveform, device=device),
            "sample_valid": torch.as_tensor(sample_valid, device=device),
            "contact_mask": torch.as_tensor(
                contact_mask, device=device
            ),
            "coordinates": torch.as_tensor(
                np.broadcast_to(coordinates, (batch, *coordinates.shape)).copy(),
                device=device,
            ),
            "coordinate_valid": torch.as_tensor(
                np.broadcast_to(
                    coordinate_valid, (batch, len(coordinate_valid))
                ).copy(), device=device,
            ),
            "shaft_index": torch.as_tensor(
                np.broadcast_to(shaft_index, (batch, len(shaft_index))).copy(),
                dtype=torch.long, device=device,
            ),
        }


def transfer_r1_2b_initialisation(model: FullTargetObserverStateModel,
                                  r1_2b_model: nn.Module) -> None:
    """Copy the fitted persistent core and spatial tail into the full observer."""
    model.state.load_state_dict(r1_2b_model.state.state_dict())
    model.state_timing.load_state_dict(r1_2b_model.state_timing.state_dict())
    model.state_contact.load_state_dict(r1_2b_model.state_contact.state_dict())
    model.state_size.load_state_dict(r1_2b_model.state_size.state_dict())
    tail = r1_2b_model.last_observer
    with torch.no_grad():
        model.observer.pool_token.copy_(tail.pool_token)
    model.observer.spatial.load_state_dict(tail.spatial.state_dict())
    model.observer.output_norm.load_state_dict(tail.output_norm.state_dict())


def initialise_raw_from_explicit(model: FullTargetObserverStateModel,
                                 explicit_checkpoint: dict,
                                 *, raw_gain: float = 0.02) -> None:
    """Start the raw arm from its paired fitted explicit R1.3 model."""
    model.load_state_dict(explicit_checkpoint, strict=True)
    model.use_raw = True
    with torch.no_grad():
        model.observer.raw_gain.fill_(float(raw_gain))


def _set_trainable(model: FullTargetObserverStateModel, *, stage: str) -> list[str]:
    """Align explicit T1, or isolate the paired raw residual increment.

    The raw arm starts from the completed paired explicit checkpoint.  Its
    common spatial/state/readout parameters must remain frozen; otherwise an
    apparent raw gain could be caused by four additional epochs of ordinary
    readout training.  All raw temporal layers still receive the exact event
    likelihood gradient through that frozen downstream map.
    """
    if stage not in {"observer_alignment", "joint_alignment"}:
        raise ValueError(f"unknown R1.3 stage {stage!r}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.use_raw:
        modules: list[nn.Module] = [model.observer.raw]
        model.observer.raw_gain.requires_grad_(True)
    else:
        modules = [
            model.state_timing, model.state_contact, model.state_size,
            model.observer.spatial, model.observer.output_norm,
            model.observer.explicit, model.observer.coordinate, model.observer.shaft,
        ]
        model.observer.pool_token.requires_grad_(True)
    if stage == "joint_alignment" and not model.use_raw:
        modules.append(model.state.correction)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    # The stable generator and deterministic baselines remain frozen in both
    # stages, so changing observer coordinates cannot be hidden by a new K.
    return [name for name, value in model.named_parameters() if value.requires_grad]


def _optimizer(model: FullTargetObserverStateModel, *, state_lr: float,
               observer_lr: float, raw_lr: float) -> torch.optim.Optimizer:
    state_modules = [model.state_timing, model.state_contact, model.state_size]
    if any(value.requires_grad for value in model.state.correction.parameters()):
        state_modules.append(model.state.correction)
    state_ids = {
        id(value) for module in state_modules for value in module.parameters()
        if value.requires_grad
    }
    raw_ids = {
        id(value) for value in model.observer.raw.parameters()
        if value.requires_grad
    }
    if model.observer.raw_gain.requires_grad:
        raw_ids.add(id(model.observer.raw_gain))
    state = [
        value for value in model.parameters()
        if value.requires_grad and id(value) in state_ids
    ]
    raw = [
        value for value in model.parameters()
        if value.requires_grad and id(value) in raw_ids
    ]
    observer = [
        value for value in model.parameters()
        if value.requires_grad and id(value) not in state_ids
        and id(value) not in raw_ids
    ]
    groups = []
    if state:
        groups.append({"params": state, "lr": float(state_lr)})
    if observer:
        groups.append({"params": observer, "lr": float(observer_lr)})
    if raw:
        groups.append({"params": raw, "lr": float(raw_lr)})
    if not groups:
        raise ValueError("R1.3 optimizer has no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=1e-3)


def _rows_for(chunk: np.ndarray, sorted_rows: np.ndarray,
              boundary: np.ndarray) -> np.ndarray:
    pieces = [
        sorted_rows[boundary[int(anchor)]:boundary[int(anchor) + 1]]
        for anchor in chunk
        if boundary[int(anchor) + 1] > boundary[int(anchor)]
    ]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def _gradient_groups(model: FullTargetObserverStateModel) -> dict[str, float]:
    groups = {
        "raw_tokenizer": "observer.raw.tokenizer",
        "raw_temporal_layer_0": "observer.raw.transformer.layers.0",
        "raw_temporal_layer_1": "observer.raw.transformer.layers.1",
        "spatial_fusion": "observer.spatial",
        "state_readout": "state_",
        "observation_correction": "state.correction",
    }
    result = {}
    named = list(model.named_parameters())
    for label, prefix in groups.items():
        square = 0.0
        for name, value in named:
            if name.startswith(prefix) and value.grad is not None:
                square += float(value.grad.detach().float().square().sum())
        result[label] = math.sqrt(square)
    return result


def train_epoch(model: FullTargetObserverStateModel,
                design: FullAnchorDesign,
                loader: FullAnchorObservationLoader,
                optimizer: torch.optim.Optimizer, *,
                device: torch.device | str,
                anchor_ids: np.ndarray,
                query_time_upper: float,
                chunk_anchors: int = 8,
                use_amp: bool = True) -> dict[str, float]:
    """One chronological truncated-BPTT pass through full TRAIN support."""
    selected = np.zeros(len(design.anchor_time), dtype=bool)
    selected[np.asarray(anchor_ids, dtype=np.int64)] = True
    event_allowed = np.flatnonzero(
        (design.event_split == 0) & (design.event_time < float(query_time_upper))
        & (design.event_source_anchor >= 0)
        & selected[np.maximum(design.event_source_anchor, 0)]
    )
    q_allowed = np.flatnonzero(
        (design.quadrature_split == 0)
        & (design.quadrature_time < float(query_time_upper))
        & (design.quadrature_source_anchor >= 0)
        & selected[np.maximum(design.quadrature_source_anchor, 0)]
    )
    event_sorted, event_boundary = _grouped_rows(
        event_allowed, design.event_source_anchor, len(design.anchor_time)
    )
    q_sorted, q_boundary = _grouped_rows(
        q_allowed, design.quadrature_source_anchor, len(design.anchor_time)
    )
    event_total = int(len(event_allowed))
    chunks = max(int(math.ceil(len(anchor_ids) / max(int(chunk_anchors), 1))), 1)
    scale = float(chunks) / max(event_total, 1)
    gradient_max: dict[str, float] = {}
    amp_enabled = bool(use_amp and str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    for label in design.session_label:
        anchors = np.flatnonzero(selected & (design.anchor_session == label))
        if not len(anchors):
            continue
        anchors = anchors[np.argsort(design.anchor_time[anchors], kind="stable")]
        state = torch.zeros(model.state.dim, device=device)
        cursor = float(design.session_start_for(np.asarray([label]))[0])
        for lo in range(0, len(anchors), int(chunk_anchors)):
            chunk = anchors[lo:lo + int(chunk_anchors)]
            batch = loader.batch(chunk, device=device, read_raw=model.use_raw)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp_enabled
            ):
                observation = model.observation_embedding(batch)
            observation = observation.float()
            transition_cache: dict[float, torch.Tensor] = {}
            chunk_states = []
            for local, anchor in enumerate(chunk):
                time = float(design.anchor_time[anchor])
                state_minus = _flow(
                    model, state, (time - cursor) / 60.0, transition_cache
                )
                state = model.state.correction(
                    state_minus, observation[local], enabled=True
                )
                chunk_states.append(state)
                cursor = time
            chunk_state = torch.stack(chunk_states)
            local_lookup = np.full(len(design.anchor_time), -1, dtype=np.int64)
            local_lookup[chunk] = np.arange(len(chunk), dtype=np.int64)
            event_rows = _rows_for(chunk, event_sorted, event_boundary)
            q_rows = _rows_for(chunk, q_sorted, q_boundary)
            if not len(event_rows) and not len(q_rows):
                state = state.detach()
                continue
            event_source = local_lookup[design.event_source_anchor[event_rows]]
            q_source = local_lookup[design.quadrature_source_anchor[q_rows]]
            event_source_state = chunk_state[
                torch.as_tensor(event_source, dtype=torch.long, device=device)
            ]
            q_source_state = chunk_state[
                torch.as_tensor(q_source, dtype=torch.long, device=device)
            ]
            event_delta = torch.as_tensor(
                (design.event_time[event_rows]
                 - design.anchor_time[design.event_source_anchor[event_rows]]) / 60.0,
                dtype=torch.float32, device=device,
            ).clamp(min=0.0)
            q_delta = torch.as_tensor(
                (design.quadrature_time[q_rows]
                 - design.anchor_time[design.quadrature_source_anchor[q_rows]]) / 60.0,
                dtype=torch.float32, device=device,
            ).clamp(min=0.0)
            matrix = model.state.generator.matrix().float()
            if len(event_rows):
                transition = torch.matrix_exp(
                    matrix.unsqueeze(0) * event_delta[:, None, None]
                )
                mu = model.state.generator.mu
                event_state = mu.unsqueeze(0) + torch.matmul(
                    transition, (event_source_state - mu).unsqueeze(-1)
                ).squeeze(-1)
                history = torch.as_tensor(
                    design.event_history[event_rows], device=device
                )
                event_log = model.timing_log_rate(history, event_state).sum()
                mark = model.mark_terms(
                    history, event_state,
                    torch.as_tensor(
                        design.event_group_ids[event_rows], dtype=torch.long,
                        device=device,
                    ),
                    torch.as_tensor(
                        design.event_group_count[event_rows], dtype=torch.long,
                        device=device,
                    ),
                )
                mark_log = mark.event_log_prob.sum()
            else:
                event_log = state.new_zeros(())
                mark_log = state.new_zeros(())
            if len(q_rows):
                transition = torch.matrix_exp(
                    matrix.unsqueeze(0) * q_delta[:, None, None]
                )
                mu = model.state.generator.mu
                q_state = mu.unsqueeze(0) + torch.matmul(
                    transition, (q_source_state - mu).unsqueeze(-1)
                ).squeeze(-1)
                history = torch.as_tensor(
                    design.quadrature_history[q_rows], device=device
                )
                log_rate = model.timing_log_rate(history, q_state)
                weight = torch.as_tensor(
                    design.quadrature_weight_seconds[q_rows],
                    dtype=log_rate.dtype, device=device,
                )
                survival = torch.sum(
                    weight * torch.exp(torch.clamp(log_rate, max=20.0))
                )
            else:
                survival = state.new_zeros(())
            loss = (survival - event_log - mark_log) * scale
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            current = _gradient_groups(model)
            for key, value in current.items():
                gradient_max[key] = max(gradient_max.get(key, 0.0), value)
            torch.nn.utils.clip_grad_norm_(
                [value for value in model.parameters() if value.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            state = state.detach()
    return gradient_max


def materialize_embedding(model: FullTargetObserverStateModel,
                          design: FullAnchorDesign,
                          loader: FullAnchorObservationLoader, *,
                          device: torch.device | str,
                          batch_size: int = 8,
                          use_amp: bool = True,
                          anchor_limit: int | None = None) -> np.ndarray:
    model.eval()
    rows = []
    amp_enabled = bool(use_amp and str(device).startswith("cuda"))
    limit = len(design.anchor_time) if anchor_limit is None else min(
        int(anchor_limit), len(design.anchor_time)
    )
    with torch.no_grad():
        for lo in range(0, limit, int(batch_size)):
            take = np.arange(lo, min(lo + int(batch_size), limit))
            batch = loader.batch(take, device=device, read_raw=model.use_raw)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp_enabled
            ):
                value = model.observation_embedding(batch)
            rows.append(value.float().cpu().numpy())
    observed = np.concatenate(rows).astype(np.float32)
    result = np.zeros((len(design.anchor_time), observed.shape[1]), dtype=np.float32)
    result[:limit] = observed
    if not np.isfinite(result).all():
        raise ValueError("R1.3 materialised embedding is non-finite")
    return result


@dataclass(frozen=True)
class FitTrace:
    selected_stage: str
    selected_stage_epoch: int
    selected_total_epoch: int
    inner_validation_joint_nll: float
    trajectory: list[dict]
    selection_gradient_max: dict[str, float]
    trainable_by_stage: dict[str, list[str]]


def fit_target_observer(model: FullTargetObserverStateModel,
                        design: FullAnchorDesign,
                        loader: FullAnchorObservationLoader, *,
                        device: torch.device | str,
                        observer_epochs: int = 2,
                        joint_epochs: int = 2,
                        state_lr: float = 3e-4,
                        observer_lr: float = 3e-5,
                        raw_lr: float = 1e-5,
                        chunk_anchors: int = 8) -> FitTrace:
    """TRAIN-inner selection followed by exact full-TRAIN refit."""
    train = design.anchor_ids("train")
    if len(train) < 10:
        raise ValueError("R1.3 needs at least ten TRAIN anchors")
    cut = int(np.clip(math.floor(0.8 * len(train)), 1, len(train) - 1))
    inner = train[:cut]
    boundary = float(design.anchor_time[train[cut]])
    train_end, _ = contract.load_split(design.subject)
    initial = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }

    def inner_value() -> float:
        embedding = materialize_embedding(
            model, design, loader, device=device, batch_size=chunk_anchors,
            anchor_limit=int(train[-1]) + 1,
        )
        return evaluate_full_t1(
            model, design, embedding, "train", device=device,
            time_lower=boundary, time_upper=train_end,
        ).joint_nll_per_event

    if model.use_raw:
        # Epoch zero is the paired explicit checkpoint, not an untrained random
        # raw residual.  The small non-zero gate is restored before optimisation
        # solely to let the exact event likelihood reach the full raw stack.
        raw_gain_initial = float(model.observer.raw_gain.detach())
        with torch.no_grad():
            model.observer.raw_gain.zero_()
        best_value = inner_value()
        with torch.no_grad():
            model.observer.raw_gain.fill_(raw_gain_initial)
    else:
        best_value = inner_value()
    best_stage = "epoch_zero"
    best_stage_epoch = 0
    best_total_epoch = 0
    trajectory = [{"stage": best_stage, "stage_epoch": 0,
                   "total_epoch": 0, "joint_nll": best_value}]
    total_epoch = 0
    gradient_max: dict[str, float] = {}
    trainable_by_stage = {}
    schedule = [
        ("observer_alignment", int(observer_epochs)),
        ("joint_alignment", int(joint_epochs)),
    ]
    for stage, epochs in schedule:
        trainable_by_stage[stage] = _set_trainable(model, stage=stage)
        optimizer = _optimizer(
            model, state_lr=state_lr, observer_lr=observer_lr, raw_lr=raw_lr
        )
        for stage_epoch in range(1, epochs + 1):
            total_epoch += 1
            model.train()
            current_gradient = train_epoch(
                model, design, loader, optimizer, device=device,
                anchor_ids=inner, query_time_upper=boundary,
                chunk_anchors=chunk_anchors,
            )
            for key, value in current_gradient.items():
                gradient_max[key] = max(gradient_max.get(key, 0.0), value)
            value = inner_value()
            trajectory.append({
                "stage": stage, "stage_epoch": stage_epoch,
                "total_epoch": total_epoch, "joint_nll": value,
            })
            if value < best_value:
                best_value = value
                best_stage = stage
                best_stage_epoch = stage_epoch
                best_total_epoch = total_epoch

    model.load_state_dict(initial)
    if best_total_epoch:
        for stage, epochs in schedule:
            stage_limit = 0
            if best_stage == stage:
                stage_limit = best_stage_epoch
            elif best_stage == "joint_alignment" and stage == "observer_alignment":
                stage_limit = epochs
            if stage_limit <= 0:
                continue
            _set_trainable(model, stage=stage)
            optimizer = _optimizer(
                model, state_lr=state_lr, observer_lr=observer_lr, raw_lr=raw_lr
            )
            for _ in range(stage_limit):
                model.train()
                train_epoch(
                    model, design, loader, optimizer, device=device,
                    anchor_ids=train, query_time_upper=train_end,
                    chunk_anchors=chunk_anchors,
                )
    elif model.use_raw:
        # A raw no-update result must be exactly the paired explicit model.
        with torch.no_grad():
            model.observer.raw_gain.zero_()
    model.eval()
    return FitTrace(
        selected_stage=best_stage,
        selected_stage_epoch=int(best_stage_epoch),
        selected_total_epoch=int(best_total_epoch),
        inner_validation_joint_nll=float(best_value),
        trajectory=trajectory,
        selection_gradient_max=gradient_max,
        trainable_by_stage=trainable_by_stage,
    )
