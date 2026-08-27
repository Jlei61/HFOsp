"""Selection-safe optimisation utilities for Continuous Marked State R1.6."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np
import torch

from . import contract
from .r1_2 import (
    FullAnchorDesign,
    FrozenEmbeddingStateModel,
    _train_epoch,
    evaluate_full_t1,
)
from .r1_3 import (
    FullAnchorObservationLoader,
    FullTargetObserverStateModel,
    _optimizer,
    _set_trainable,
    materialize_embedding,
    train_epoch,
)


R1_6_REVISION = "r1_6_optimizer_identifiability_nested_selection_v1"


@dataclass(frozen=True)
class NestedTimeSplit:
    base_train_ids: np.ndarray
    prefix_refit_ids: np.ndarray
    base_select_lower: float
    base_select_upper: float
    alignment_select_lower: float
    alignment_select_upper: float
    n_train_anchors: int
    base_train_cut: int
    prefix_refit_cut: int

    def summary(self) -> dict:
        value = asdict(self)
        value["base_train_ids"] = [
            int(self.base_train_ids[0]), int(self.base_train_ids[-1])
        ]
        value["prefix_refit_ids"] = [
            int(self.prefix_refit_ids[0]), int(self.prefix_refit_ids[-1])
        ]
        return value


def nested_time_split(design: FullAnchorDesign, *,
                      base_fraction: float = 0.6,
                      alignment_fraction: float = 0.8) -> NestedTimeSplit:
    """Make the 0--60/60--80/80--100 split entirely inside TRAIN."""
    if not 0.0 < float(base_fraction) < float(alignment_fraction) < 1.0:
        raise ValueError("nested fractions must satisfy 0 < base < alignment < 1")
    train = np.asarray(design.anchor_ids("train"), dtype=np.int64)
    if len(train) < 20:
        raise ValueError("R1.6 nested selection needs at least 20 TRAIN anchors")
    first = int(np.clip(math.floor(base_fraction * len(train)), 1, len(train) - 2))
    second = int(np.clip(
        math.floor(alignment_fraction * len(train)), first + 1, len(train) - 1
    ))
    base_lower = float(design.anchor_time[train[first]])
    alignment_lower = float(design.anchor_time[train[second]])
    train_end, _ = contract.load_split(design.subject)
    if not base_lower < alignment_lower < float(train_end):
        raise ValueError("R1.6 nested time boundaries are not strictly ordered")
    return NestedTimeSplit(
        base_train_ids=train[:first],
        prefix_refit_ids=train[:second],
        base_select_lower=base_lower,
        base_select_upper=alignment_lower,
        alignment_select_lower=alignment_lower,
        alignment_select_upper=float(train_end),
        n_train_anchors=int(len(train)),
        base_train_cut=int(first),
        prefix_refit_cut=int(second),
    )


def _state_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _parameter_norm(model: torch.nn.Module) -> float:
    return math.sqrt(sum(
        float(value.detach().float().square().sum())
        for value in model.parameters() if value.requires_grad
    ))


def _update_norm(before: dict[str, torch.Tensor],
                 model: torch.nn.Module) -> float:
    current = model.state_dict()
    return math.sqrt(sum(
        float((current[key].detach().cpu().float() - old.float()).square().sum())
        for key, old in before.items() if key in current
    ))


def fit_prefix_safe_core(model: FrozenEmbeddingStateModel,
                         design: FullAnchorDesign,
                         embedding: np.ndarray, *,
                         device: torch.device | str,
                         epochs: int = 4,
                         learning_rate: float = 3e-4,
                         weight_decay: float = 1e-3,
                         chunk_anchors: int = 256,
                         optimizer_name: str = "adamw",
                         grad_clip_norm: float | None = 1.0,
                         warmup_fraction: float = 0.0,
                         selection_min_delta: float = 0.0,
                         early_stopping_patience: int | None = None,
                         split: NestedTimeSplit | None = None
                         ) -> tuple[FrozenEmbeddingStateModel, dict]:
    """Fit the core without exposing epoch zero to alignment selection data."""
    split = split or nested_time_split(design)
    if float(selection_min_delta) < 0.0:
        raise ValueError("prefix selection_min_delta must be non-negative")
    if (early_stopping_patience is not None
            and int(early_stopping_patience) < 1):
        raise ValueError("prefix early_stopping_patience must be positive")
    initial = _state_snapshot(model)

    def value(lower: float, upper: float) -> dict:
        return asdict(evaluate_full_t1(
            model, design, embedding, "train", device=device,
            time_lower=float(lower), time_upper=float(upper),
        ))

    best_metrics = value(split.base_select_lower, split.base_select_upper)
    best_value = float(best_metrics["joint_nll_per_event"])
    train_metrics = value(
        float(design.anchor_time[split.base_train_ids[0]]),
        split.base_select_lower,
    )
    best_epoch = 0
    trajectory = [{
        "epoch": 0,
        "base_select_joint_nll": best_value,
        "base_select_metrics": best_metrics,
        "evaluated_train_metrics": train_metrics,
        "parameter_norm": _parameter_norm(model),
        "update_norm": 0.0,
        "optimizer_steps": 0, "clip_fraction": None,
        "preclip_norm_max": None, "postclip_norm_max": None,
    }]
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer_class = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
    }.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError(f"unsupported prefix optimizer {optimizer_name!r}")
    optimizer = optimizer_class([{
        "params": parameters, "lr": float(learning_rate),
        "base_lr": float(learning_rate), "group_name": "prefix",
    }], weight_decay=float(weight_decay))
    approximate_steps = max(
        int(math.ceil(len(split.base_train_ids) / max(int(chunk_anchors), 1))), 1
    ) * max(int(epochs), 1)
    step_state = {
        "step": 0,
        "warmup_steps": int(math.ceil(
            float(warmup_fraction) * approximate_steps
        )),
    }
    without_improvement = 0
    executed_epochs = 0
    for epoch in range(1, int(epochs) + 1):
        executed_epochs = int(epoch)
        before = _state_snapshot(model)
        model.train()
        diagnostics: dict = {}
        _train_epoch(
            model, design, embedding, optimizer, device=device,
            anchor_ids=split.base_train_ids,
            query_time_upper=split.base_select_lower,
            chunk_anchors=int(chunk_anchors),
            grad_clip_norm=grad_clip_norm, step_state=step_state,
            diagnostics=diagnostics,
        )
        current_metrics = value(
            split.base_select_lower, split.base_select_upper
        )
        current = float(current_metrics["joint_nll_per_event"])
        train_metrics = value(
            float(design.anchor_time[split.base_train_ids[0]]),
            split.base_select_lower,
        )
        trajectory.append({
            "epoch": int(epoch),
            "base_select_joint_nll": current,
            "base_select_metrics": current_metrics,
            "evaluated_train_metrics": train_metrics,
            "parameter_norm": _parameter_norm(model),
            "update_norm": _update_norm(before, model),
            **diagnostics,
        })
        if current < best_value - float(selection_min_delta):
            best_value = current
            best_epoch = int(epoch)
            without_improvement = 0
        else:
            without_improvement += 1
        if (early_stopping_patience is not None
                and without_improvement >= int(early_stopping_patience)):
            break

    model.load_state_dict(initial)
    if best_epoch:
        optimizer = optimizer_class([{
            "params": parameters, "lr": float(learning_rate),
            "base_lr": float(learning_rate), "group_name": "prefix",
        }], weight_decay=float(weight_decay))
        approximate_steps = max(
            int(math.ceil(
                len(split.prefix_refit_ids) / max(int(chunk_anchors), 1)
            )), 1
        ) * max(int(best_epoch), 1)
        step_state = {
            "step": 0,
            "warmup_steps": int(math.ceil(
                float(warmup_fraction) * approximate_steps
            )),
        }
        for _ in range(best_epoch):
            model.train()
            _train_epoch(
                model, design, embedding, optimizer, device=device,
                anchor_ids=split.prefix_refit_ids,
                query_time_upper=split.alignment_select_lower,
                chunk_anchors=int(chunk_anchors),
                grad_clip_norm=grad_clip_norm, step_state=step_state,
            )
    model.eval()
    trace = {
        "revision": R1_6_REVISION,
        "selected_epoch": int(best_epoch),
        "base_select_joint_nll": float(best_value),
        "trajectory": trajectory,
        "optimizer": optimizer_name,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "grad_clip_norm": (
            None if grad_clip_norm is None else float(grad_clip_norm)
        ),
        "warmup_fraction": float(warmup_fraction),
        "selection_min_delta": float(selection_min_delta),
        "early_stopping_patience": (
            None if early_stopping_patience is None
            else int(early_stopping_patience)
        ),
        "executed_epochs": int(executed_epochs),
        "chunk_anchors": int(chunk_anchors),
        "epochs_budget": int(epochs),
        "epoch_zero_seen_base_select": False,
        "epoch_zero_seen_alignment_select": False,
        "split": split.summary(),
    }
    return model, trace


def transfer_prefix_core(target: FullTargetObserverStateModel,
                         core: FrozenEmbeddingStateModel) -> None:
    """Copy the selection-safe state core without changing the observer."""
    target.state.load_state_dict(core.state.state_dict())
    target.state_timing.load_state_dict(core.state_timing.state_dict())
    target.state_contact.load_state_dict(core.state_contact.state_dict())
    target.state_size.load_state_dict(core.state_size.state_dict())


def parameter_group_update_norms(initial: dict[str, torch.Tensor],
                                 model: torch.nn.Module) -> dict[str, float]:
    """Report update norms in the scientific parameter groups."""
    groups = {
        "spatial_fusion": "observer.spatial",
        "explicit_projection": "observer.explicit",
        "observation_correction": "state.correction",
        "state_readout_timing": "state_timing",
        "state_readout_contact": "state_contact",
        "state_readout_size": "state_size",
        "stable_generator": "state.generator",
    }
    current = model.state_dict()
    result = {}
    for label, prefix in groups.items():
        square = 0.0
        for name, value in current.items():
            if name.startswith(prefix) and name in initial:
                delta = value.detach().cpu().float() - initial[name].float()
                square += float(delta.square().sum())
        result[label] = math.sqrt(square)
    return result


def fixed_overfit_segment(design: FullAnchorDesign,
                          split: NestedTimeSplit, *,
                          maximum_anchors: int = 64) -> tuple[np.ndarray, float, float]:
    """Choose the earliest continuous part of the best-supported TRAIN session."""
    eligible = np.asarray(split.base_train_ids, dtype=np.int64)
    sessions = []
    for label in design.session_label:
        local = eligible[design.anchor_session[eligible] == label]
        if len(local):
            local = local[np.argsort(design.anchor_time[local], kind="stable")]
            sessions.append(local)
    if not sessions:
        raise ValueError("R1.6 overfit check found no base-TRAIN session")
    sessions.sort(key=lambda row: (-len(row), float(design.anchor_time[row[0]])))
    selected = sessions[0][:min(int(maximum_anchors), len(sessions[0]))]
    if len(selected) < 8:
        raise ValueError("R1.6 overfit check needs at least eight continuous anchors")
    label = design.anchor_session[selected[0]]
    same_session = np.flatnonzero(design.anchor_session == label)
    same_session = same_session[
        np.argsort(design.anchor_time[same_session], kind="stable")
    ]
    following = same_session[design.anchor_time[same_session]
                             > design.anchor_time[selected[-1]]]
    upper = (
        float(design.anchor_time[following[0]]) if len(following)
        else float(split.base_select_lower)
    )
    upper = min(upper, float(split.base_select_lower))
    lower = float(design.anchor_time[selected[0]])
    if not lower < upper:
        raise ValueError("R1.6 overfit segment has no positive time support")
    return selected, lower, upper


def overfit_target_segment(model: FullTargetObserverStateModel,
                           design: FullAnchorDesign,
                           loader: FullAnchorObservationLoader, *,
                           device: torch.device | str,
                           split: NestedTimeSplit,
                           epochs: int = 20,
                           maximum_anchors: int = 64,
                           state_lr: float = 1e-3,
                           observer_lr: float = 1e-4,
                           weight_decay: float = 0.0,
                           grad_clip_norm: float | None = 5.0,
                           warmup_fraction: float = 0.1,
                           chunk_anchors: int = 8,
                           optimizer_name: str = "adamw") -> dict:
    """Ask only whether the exact target model can fit one short segment."""
    anchors, lower, upper = fixed_overfit_segment(
        design, split, maximum_anchors=maximum_anchors
    )
    _set_trainable(model, stage="joint_alignment")
    optimizer = _optimizer(
        model, state_lr=state_lr, observer_lr=observer_lr, raw_lr=1e-5,
        optimizer_name=optimizer_name, weight_decay=weight_decay,
    )
    total_steps = max(
        int(math.ceil(len(anchors) / max(int(chunk_anchors), 1))), 1
    ) * max(int(epochs), 1)
    step_state = {
        "step": 0,
        "warmup_steps": int(math.ceil(float(warmup_fraction) * total_steps)),
    }

    def metrics() -> dict:
        embedding = materialize_embedding(
            model, design, loader, device=device, batch_size=chunk_anchors,
            anchor_limit=int(anchors[-1]) + 1,
        )
        return asdict(evaluate_full_t1(
            model, design, embedding, "train", device=device,
            time_lower=lower, time_upper=upper,
        ))

    initial = metrics()
    trajectory = [{
        "epoch": 0, "metrics": initial, "optimizer_steps": 0,
        "update_norm": 0.0, "clip_fraction": None,
    }]
    for epoch in range(1, int(epochs) + 1):
        before = _state_snapshot(model)
        diagnostics: dict = {}
        train_epoch(
            model, design, loader, optimizer, device=device,
            anchor_ids=anchors, query_time_upper=upper,
            chunk_anchors=chunk_anchors, grad_clip_norm=grad_clip_norm,
            step_state=step_state, diagnostics=diagnostics,
        )
        trajectory.append({
            "epoch": int(epoch), "metrics": metrics(), **diagnostics,
            "update_norm": _update_norm(before, model),
        })
    final = trajectory[-1]["metrics"]
    return {
        "status": "COMPLETE", "revision": R1_6_REVISION,
        "segment_anchor_ids": [int(anchors[0]), int(anchors[-1])],
        "n_anchors": int(len(anchors)), "time_lower": lower,
        "time_upper": upper, "epochs": int(epochs),
        "optimizer": optimizer_name, "state_lr": float(state_lr),
        "observer_lr": float(observer_lr), "weight_decay": float(weight_decay),
        "grad_clip_norm": grad_clip_norm,
        "warmup_fraction": float(warmup_fraction),
        "initial_joint_nll": float(initial["joint_nll_per_event"]),
        "final_joint_nll": float(final["joint_nll_per_event"]),
        "joint_nll_improvement": float(
            initial["joint_nll_per_event"] - final["joint_nll_per_event"]
        ),
        "trajectory": trajectory,
    }
