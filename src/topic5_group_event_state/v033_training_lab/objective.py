"""Trainable protocol and the reference count-profile objective (design §4).

A ``Trainable`` couples a model family to a scientific objective.  The training
laboratory only ever calls this protocol, so Agent C can register the grammar
objective (``S_G``) without the harness changing.  The reference implementation
is the ``S_N`` future-burden profile: NB residuals per bin on top of the
explicit history baseline supplied by the data view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_group_event_state.v032_model.readout import nb_log_prob  # re-use, do not re-invent

from .data import DataView
from .models import ArchConfig, FlexibleResidualStateModel, build_flexible_model


@dataclass
class LossTerms:
    idx: np.ndarray
    nll: Tensor            # (A,) sum over bins, FP32
    per_bin_nll: Tensor    # (A, B)
    weights: Tensor        # (A,) sampling weights (mean 1 on TRAIN)
    modulation: Tensor     # (A, B) log mu - log mu_H
    state_raw: Tensor      # (A, D)
    state_std: Tensor      # (A, D)
    log_mu: Tensor         # (A, B)
    log_mu_h: Tensor       # (A, B)
    y: Tensor              # (A, B)


class Trainable(Protocol):
    name: str

    def build(self, arch: ArchConfig, view: DataView, seed: int) -> nn.Module: ...

    def param_groups(self, model: nn.Module, lrs: Mapping[str, float], weight_decay: float) -> list[dict[str, Any]]: ...

    def tensors(self, view: DataView, device: torch.device) -> dict[str, Tensor]: ...

    def refresh_statistics(self, model: nn.Module, view: DataView, device: torch.device,
                           tensors: dict[str, Tensor] | None = None) -> None: ...

    def loss_terms(self, model: nn.Module, view: DataView, phase: str, *, device: torch.device,
                   differentiable_statistics: bool, sampling: str, lookback_seconds: float,
                   tensors: dict[str, Tensor] | None = None, state_override: Tensor | None = None) -> LossTerms: ...

    def h_only_nll(self, view: DataView, phase: str) -> np.ndarray: ...


class ResidualCountTrainable:
    """``S_N`` family: future-burden count profile as NB residuals on ``log mu_H`` per bin."""

    name = "count_profile"

    def __init__(self) -> None:
        self._cache: dict[tuple[int, str], dict[str, Tensor]] = {}

    def build(self, arch: ArchConfig, view: DataView, seed: int) -> FlexibleResidualStateModel:
        if view.missing_h_bins:
            raise ValueError(f"data view lacks log_mu_H for bins {view.missing_h_bins}; request must be HELD")
        return build_flexible_model(arch, in_dim=view.n_features, n_bins=view.n_bins,
                                    log_r_init=view.log_r_h, seed=seed)

    def param_groups(self, model: FlexibleResidualStateModel, lrs: Mapping[str, float], weight_decay: float) -> list[dict[str, Any]]:
        return model.param_groups(lrs, weight_decay)

    def tensors(self, view: DataView, device: torch.device) -> dict[str, Tensor]:
        key = (id(view), str(device))
        if key not in self._cache:
            train = view.phase_index["train"]
            self._cache[key] = {
                "x": torch.from_numpy(np.ascontiguousarray(view.x_scaled)).to(device, torch.float32),
                "times": torch.from_numpy(view.event_times).to(device, torch.float64),
                "segment": torch.from_numpy(view.event_segment).to(device, torch.long),
                "train_event_mask": torch.from_numpy(view.train_event_mask).to(device),
                "t_anchor": torch.from_numpy(view.t_anchor).to(device, torch.float64),
                "last_event_pos": torch.from_numpy(view.last_event_pos).to(device, torch.long),
                "train_anchor_time": torch.from_numpy(view.t_anchor[train]).to(device, torch.float64),
                "train_last_event_pos": torch.from_numpy(view.last_event_pos[train]).to(device, torch.long),
                "log_mu_h": torch.from_numpy(np.nan_to_num(view.log_mu_h, nan=0.0)).to(device, torch.float32),
                "counts": torch.from_numpy(view.counts.astype(np.float32)).to(device, torch.float32),
            }
        return self._cache[key]

    def refresh_statistics(self, model: FlexibleResidualStateModel, view: DataView, device: torch.device,
                           tensors: dict[str, Tensor] | None = None) -> None:
        t = tensors or self.tensors(view, device)
        model.refresh_train_statistics(t["x"], t["train_event_mask"], t["times"], t["segment"],
                                       t["train_anchor_time"], t["train_last_event_pos"])

    def loss_terms(self, model: FlexibleResidualStateModel, view: DataView, phase: str, *, device: torch.device,
                   differentiable_statistics: bool, sampling: str, lookback_seconds: float,
                   tensors: dict[str, Tensor] | None = None, state_override: Tensor | None = None) -> LossTerms:
        if differentiable_statistics and phase != "train":
            raise ValueError("differentiable TRAIN statistics are only defined on the train pass")
        idx = view.phase_index[phase]                       # KeyError for anything but train / inner_val
        view.assert_no_dev_test(idx)
        t = tensors or self.tensors(view, device)
        train_mask = t["train_event_mask"] if differentiable_statistics else None
        _pre, post = model.trajectory(t["x"], t["times"], t["segment"], train_mask)
        idx_t = torch.from_numpy(idx).to(device)
        state = model.anchor_states(post, t["times"], t["t_anchor"][idx_t], t["last_event_pos"][idx_t])
        train_state = state if differentiable_statistics else None
        if state_override is not None:
            state = state_override.to(state.dtype)
            train_state = None
        state_std = model.standardize_state(state, train_state)
        log_mu_h = t["log_mu_h"][idx_t]
        y = t["counts"][idx_t]
        log_mu = model.adapter(log_mu_h, state_std)
        per_bin = torch.stack([
            -nb_log_prob(y[:, b], torch.exp(log_mu[:, b]), model.adapter.log_r[b]) for b in range(model.n_bins)
        ], dim=1)
        weights = torch.from_numpy(view.sample_weights(phase, sampling, lookback_seconds=lookback_seconds)).to(device, torch.float32)
        return LossTerms(idx=idx, nll=per_bin.sum(dim=1), per_bin_nll=per_bin, weights=weights,
                         modulation=log_mu - log_mu_h, state_raw=state.to(torch.float32), state_std=state_std,
                         log_mu=log_mu, log_mu_h=log_mu_h, y=y)

    def h_only_nll(self, view: DataView, phase: str) -> np.ndarray:
        idx = view.phase_index[phase]
        view.assert_no_dev_test(idx)
        y = torch.from_numpy(view.counts[idx].astype(np.float32))
        log_mu_h = torch.from_numpy(view.log_mu_h[idx].astype(np.float32))
        total = torch.zeros(idx.size, dtype=torch.float32)
        for b in range(view.n_bins):
            total = total - nb_log_prob(y[:, b], torch.exp(log_mu_h[:, b]), torch.tensor(float(view.log_r_h[b])))
        return total.numpy().astype(np.float64)


TRAINABLE_REGISTRY: dict[str, type] = {ResidualCountTrainable.name: ResidualCountTrainable}
