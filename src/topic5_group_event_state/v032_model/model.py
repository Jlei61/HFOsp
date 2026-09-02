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
        self.register_buffer("train_mean_state", torch.zeros(self.state.state_dim))

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

    def writes(self, x_std: Tensor) -> Tensor:
        """Bank: ``tanh(phi - mean_train(phi))``; RNN: centred embedding."""

        phi = self.project(x_std)
        if self.cfg.architecture == "leaky_bank":
            return torch.tanh(phi - self.phi_mean)
        return phi - self.phi_mean

    @torch.no_grad()
    def refresh_train_mean(self, x_std: Tensor, train_event_mask: Tensor) -> None:
        mask = train_event_mask.to(torch.bool)
        if int(mask.sum()) == 0:
            raise ValueError("train mean needs at least one state-train event")
        self.phi_mean.copy_(self.project(x_std[mask]).mean(dim=0).detach())

    def trajectory(self, x_std: Tensor, times: Tensor, segment_ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.state(self.writes(x_std), times, segment_ids)

    def anchor_states(
        self, state_post: Tensor, event_times: Tensor, t_anchor: Tensor, last_event_pos: Tensor
    ) -> Tensor:
        return self.state.anchor(state_post, event_times, t_anchor, last_event_pos)

    def log_mu(self, log_mu_h: Tensor, anchor_state: Tensor) -> Tensor:
        return self.adapter(log_mu_h, anchor_state)

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
