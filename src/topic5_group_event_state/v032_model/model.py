"""Assembly of encoder, state backbone and residual adapter (design §3-§4)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .config import ModelConfig
from .encoder import EventProjection
from .readout import ResidualCountAdapter
from .state import MarkedLeakyBank, RepairedRecurrentState


class ResidualStateModel(nn.Module):
    def __init__(self, cfg: ModelConfig, in_dim: int, log_r_init: float) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        if cfg.architecture == "leaky_bank":
            self.encoder = EventProjection(in_dim, cfg.encoder_hidden, cfg.phi_dim)
            self.state = MarkedLeakyBank(
                cfg.taus_seconds, cfg.channels_per_tau,
                chunk_seconds=cfg.chunk_seconds, detach_chunks=cfg.detach_chunks,
            )
            write_dim = cfg.phi_dim
        else:
            self.encoder = EventProjection(in_dim, cfg.encoder_hidden, cfg.rnn_event_dim)
            self.state = RepairedRecurrentState(
                cfg.taus_seconds, cfg.channels_per_tau,
                event_dim=cfg.rnn_event_dim, hidden=cfg.rnn_hidden,
            )
            write_dim = cfg.rnn_event_dim
        self.adapter = ResidualCountAdapter(self.state.state_dim, cfg.alpha_init, log_r_init)
        # TRAIN-only centring of phi (bank) / embedding (rnn); frozen into checkpoints.
        self.register_buffer("phi_mean", torch.zeros(write_dim))
        # TRAIN-only fixed mean/scale of the anchor state (design §3.1: no per-time
        # LayerNorm).  A tau=2 h integrator holds rate*tau ~ 10^2-10^3 event writes,
        # so an unscaled readout is badly conditioned; the statistics are recomputed
        # (detached) every optimizer step and frozen into the checkpoint.
        self.register_buffer("train_mean_state", torch.zeros(self.state.state_dim))
        self.register_buffer("train_state_scale", torch.ones(self.state.state_dim))

    # ----------------------------------------------------------------- forward
    @property
    def state_dim(self) -> int:
        return int(self.state.state_dim)

    def project(self, x_std: Tensor) -> Tensor:
        if self.cfg.amp_encoder and x_std.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.encoder(x_std)
            return out.float()
        return self.encoder(x_std)

    def writes(self, x_std: Tensor, train_event_mask: Tensor | None = None) -> Tensor:
        """Bank: ``tanh(phi - mean_train(phi))``; RNN: centred embedding.

        With ``train_event_mask`` (training forward) the TRAIN mean is a
        differentiable function of the current batch -- the whole state-train
        event set -- and its detached value is stored for evaluation/replay.
        Without it (evaluation, replay, export) the frozen buffer is used.
        """

        phi = self.project(x_std)
        if train_event_mask is not None:
            mask = train_event_mask.to(torch.bool)
            mean = phi[mask].mean(dim=0)
            self.phi_mean.copy_(mean.detach())
        else:
            mean = self.phi_mean
        if self.cfg.architecture == "leaky_bank":
            return torch.tanh(phi - mean)
        return phi - mean

    @torch.no_grad()
    def refresh_train_mean(self, x_std: Tensor, train_event_mask: Tensor) -> None:
        mask = train_event_mask.to(torch.bool)
        if int(mask.sum()) == 0:
            raise ValueError("train mean needs at least one state-train event")
        self.phi_mean.copy_(self.project(x_std[mask]).mean(dim=0).detach())

    def trajectory(
        self, x_std: Tensor, times: Tensor, segment_ids: Tensor, train_event_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        return self.state(self.writes(x_std, train_event_mask), times, segment_ids)

    def anchor_states(
        self, state_post: Tensor, event_times: Tensor, t_anchor: Tensor, last_event_pos: Tensor
    ) -> Tensor:
        return self.state.anchor(state_post, event_times, t_anchor, last_event_pos)

    def standardize_state(self, anchor_state: Tensor, train_state: Tensor | None = None) -> Tensor:
        """``(S - mean_train) / scale_train``.

        ``train_state`` (training forward only) supplies the TRAIN anchor states of
        the current batch so the statistics stay differentiable; their detached
        values are stored in the buffers used at evaluation and replay.
        """

        s = anchor_state.to(torch.float32)
        if train_state is not None:
            ref = train_state.to(torch.float32)
            mean = ref.mean(dim=0)
            scale = ref.std(dim=0, unbiased=False)
            scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
            self.train_mean_state.copy_(mean.detach())
            self.train_state_scale.copy_(scale.detach())
            return (s - mean) / scale
        return (s - self.train_mean_state) / self.train_state_scale

    def log_mu(self, log_mu_h: Tensor, anchor_state: Tensor, train_state: Tensor | None = None) -> Tensor:
        return self.adapter(log_mu_h, self.standardize_state(anchor_state, train_state))

    def modulation_jacobian(self) -> Tensor:
        """d(log mu)/d(raw anchor state): alpha * w / scale, per state dimension."""

        return (self.adapter.alpha * self.adapter.w.weight.squeeze(0) / self.train_state_scale).detach()

    @torch.no_grad()
    def refresh_train_statistics(
        self,
        x_std: Tensor,
        train_event_mask: Tensor,
        times: Tensor,
        segment_ids: Tensor,
        train_anchor_time: Tensor,
        train_last_event_pos: Tensor,
    ) -> None:
        """Recompute phi_mean and the TRAIN anchor-state mean/scale with current theta."""

        self.refresh_train_mean(x_std, train_event_mask)
        _pre, post = self.trajectory(x_std, times, segment_ids)
        state = self.anchor_states(post, times, train_anchor_time, train_last_event_pos).to(torch.float32)
        if state.shape[0] == 0:
            raise ValueError("train statistics need at least one state-train anchor")
        scale = state.std(dim=0, unbiased=False)
        self.train_mean_state.copy_(state.mean(dim=0))
        self.train_state_scale.copy_(torch.where(scale > 1e-6, scale, torch.ones_like(scale)))

    # ------------------------------------------------------------- optimiser
    def param_groups(self, cfg: ModelConfig) -> list[dict[str, Any]]:
        """Named groups; biases / gate / dispersion never receive weight decay."""

        state_weights = [p for n, p in self.state.named_parameters() if p.ndim > 1]
        state_bias = [p for n, p in self.state.named_parameters() if p.ndim <= 1]
        groups = [
            {"name": "encoder_weights", "params": self.encoder.weight_parameters(),
             "lr": cfg.lr_encoder, "weight_decay": cfg.weight_decay},
            {"name": "encoder_bias", "params": self.encoder.bias_parameters(),
             "lr": cfg.lr_encoder, "weight_decay": 0.0},
            {"name": "state_weights", "params": state_weights,
             "lr": cfg.lr_state, "weight_decay": cfg.weight_decay},
            {"name": "state_bias", "params": state_bias, "lr": cfg.lr_state, "weight_decay": 0.0},
            {"name": "adapter_w", "params": [self.adapter.w.weight],
             "lr": cfg.lr_adapter, "weight_decay": cfg.weight_decay},
            {"name": "adapter_gate_alpha", "params": [self.adapter.alpha],
             "lr": cfg.lr_adapter, "weight_decay": 0.0},
            {"name": "adapter_dispersion", "params": [self.adapter.log_r],
             "lr": cfg.lr_adapter, "weight_decay": 0.0},
        ]
        return groups


def build_model(cfg: ModelConfig, *, in_dim: int, log_r_init: float, seed: int) -> ResidualStateModel:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    return ResidualStateModel(cfg, in_dim, log_r_init)
