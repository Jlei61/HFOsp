"""Spatial-view event encoder, persistent state and frozen-decoder adapters."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn

from src.topic5_group_event_state.v032_model.state import MarkedLeakyBank
from src.topic5_group_event_state.v033_evaluator.oracle import (
    conditional_bernoulli_logpmf_torch,
)
from src.topic5_group_event_state.v033_training_lab.contact_grammar import (
    LegacyContactGrammar,
)
from src.topic5_group_event_state.v033_training_lab.sg_o2 import (
    FrozenLegacyStateScorer,
)
from src.topic5_rank_distribution import next_set_stop_loss

from .contracts import ArchConfig, OptimizerConfig, optimizer_contract


class ResidualBlock(nn.Module):
    def __init__(self, width: int, residual: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear = nn.Linear(width, width)
        self.residual = bool(residual)

    def forward(self, x: Tensor) -> Tensor:
        update = torch.nn.functional.gelu(self.linear(self.norm(x)))
        return (x + update) / math.sqrt(2.0) if self.residual else update


class SpatialEventEncoder(nn.Module):
    """Encode one complete group event, not a single contact or fixed window."""

    def __init__(self, input_dim: int, config: ArchConfig) -> None:
        super().__init__()
        self.config = config.validate()
        self.input = nn.Linear(int(input_dim), config.width)
        self.blocks = nn.ModuleList([
            ResidualBlock(config.width, config.residual) for _ in range(config.depth)
        ])
        self.write = nn.Linear(config.width, config.write_width)
        self.gate = nn.Linear(config.width, config.write_width)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # A small but nonzero initial write avoids both a wide-open updater and
        # the exact-zero-gradient failure seen in earlier state experiments.
        nn.init.constant_(self.gate.bias, -1.5)

    def forward(self, event_token: Tensor) -> Tensor:
        hidden = torch.nn.functional.gelu(self.input(event_token.to(torch.float32)))
        for block in self.blocks:
            hidden = block(hidden)
        return torch.sigmoid(self.gate(hidden)) * torch.tanh(self.write(hidden))


class SpatialAuxiliaryHeads(nn.Module):
    """Functional S_P readout: subset, continue, positive extent and lag."""

    def __init__(self, state_dim: int, n_contacts: int, rank: int) -> None:
        super().__init__()
        rank = min(int(rank), int(state_dim), int(n_contacts))
        self.subset = nn.Sequential(
            nn.Linear(state_dim, rank, bias=False), nn.Linear(rank, n_contacts, bias=False)
        )
        self.continue_head = nn.Linear(state_dim, 1)
        self.extent_head = nn.Linear(state_dim, 1)
        self.lag_head = nn.Sequential(
            nn.Linear(state_dim, rank, bias=False), nn.Linear(rank, n_contacts, bias=False)
        )
        self.register_buffer("base_contact_logits", torch.zeros(n_contacts))
        self.register_buffer("base_log_extent", torch.tensor(0.0))
        self.register_buffer("base_continue_logit", torch.tensor(0.0))
        self.register_buffer("base_lag", torch.zeros(n_contacts))
        self.register_buffer("lag_scale", torch.ones(n_contacts))
        self._zero_residuals()

    def _zero_residuals(self) -> None:
        # Zero functional residual at construction is an exact baseline parity
        # point; upstream state gradients become nonzero after the heads move.
        for seq in (self.subset, self.lag_head):
            nn.init.normal_(seq[0].weight, std=0.02)
            nn.init.normal_(seq[1].weight, std=1e-3)
        nn.init.normal_(self.continue_head.weight, std=1e-3)
        nn.init.normal_(self.extent_head.weight, std=1e-3)
        nn.init.zeros_(self.continue_head.bias)
        nn.init.zeros_(self.extent_head.bias)

    @torch.no_grad()
    def set_training_baselines(
        self,
        *,
        contact_logits: Tensor,
        log_extent: float,
        continue_logit: float,
        lag_mean: Tensor,
        lag_scale: Tensor,
    ) -> None:
        self.base_contact_logits.copy_(contact_logits)
        self.base_log_extent.fill_(float(log_extent))
        self.base_continue_logit.fill_(float(continue_logit))
        self.base_lag.copy_(lag_mean)
        self.lag_scale.copy_(lag_scale.clamp_min(1e-4))

    def forward(self, state: Tensor) -> Mapping[str, Tensor]:
        return {
            "subset_logits": self.base_contact_logits + self.subset(state),
            "continue_logits": self.base_continue_logit + self.continue_head(state).squeeze(-1),
            "log_extent": self.base_log_extent + self.extent_head(state).squeeze(-1),
            "lag_mean": self.base_lag + self.lag_head(state),
        }

    def losses(
        self,
        state: Tensor,
        *,
        participation: Tensor,
        positive_extent: Tensor,
        group_count: Tensor,
        relative_lag: Tensor,
        lag_valid: Tensor,
    ) -> Mapping[str, Tensor]:
        out = self(state)
        subset = -conditional_bernoulli_logpmf_torch(
            out["subset_logits"], participation.bool()
        )
        continue_target = (group_count > 1).to(out["continue_logits"].dtype)
        cont = torch.nn.functional.binary_cross_entropy_with_logits(
            out["continue_logits"], continue_target, reduction="none"
        )
        extent = positive_extent.to(out["log_extent"].dtype).clamp_min(1.0)
        extent_nll = torch.exp(out["log_extent"]) - extent * out["log_extent"] \
            + torch.lgamma(extent + 1.0)
        z = (relative_lag - out["lag_mean"]) / self.lag_scale
        lag_terms = 0.5 * z.square() + torch.log(self.lag_scale)
        lag_mask = lag_valid.bool() & participation.bool()
        lag_nll = (lag_terms * lag_mask).sum(1) / lag_mask.sum(1).clamp_min(1)
        return {
            "subset_nll": subset,
            "continue_nll": cont,
            "extent_nll": extent_nll,
            "lag_nll": lag_nll,
        }


class LegacyTrainMeanAdapter(nn.Module):
    """No-state recalibration fitted on the early STATE_TRAIN subperiod only.

    The older contact decoder has already seen the calibration prefix, but its
    average contact/STOP level can drift before state training starts.  These
    output-level offsets give the no-state baseline that same opportunity to
    update.  They are frozen before the recurrent state is trained.
    """

    def __init__(self, n_contacts: int) -> None:
        super().__init__()
        self.contact_bias = nn.Parameter(torch.zeros(int(n_contacts)), requires_grad=False)
        self.stop_bias = nn.Parameter(torch.zeros(()), requires_grad=False)

    def set_trainable(self, value: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(bool(value))

    def apply(self, output: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return {
            **output,
            "contact_logits": output["contact_logits"] + self.contact_bias[None, None, :],
            "stop_logits": output["stop_logits"] + self.stop_bias,
        }

class SpatialStateModel(nn.Module):
    """S_P producer; a legacy decoder is optional for synthetic diagnostics."""

    def __init__(
        self,
        *,
        input_dim: int,
        n_contacts: int,
        config: ArchConfig,
        legacy_decoder: LegacyContactGrammar | None = None,
    ) -> None:
        super().__init__()
        self.config = config.validate()
        self.encoder = SpatialEventEncoder(input_dim, config)
        self.state_bank = MarkedLeakyBank(
            config.taus_seconds,
            config.write_width,
            chunk_seconds=3600.0,
            detach_chunks=False,
        )
        self.functional = SpatialAuxiliaryHeads(
            config.state_dim, n_contacts, config.adapter_rank
        )
        self.legacy = None if legacy_decoder is None else FrozenLegacyStateScorer(
            legacy_decoder, state_dim=config.state_dim, rank=config.adapter_rank
        )
        self.train_mean_adapter = (
            None if legacy_decoder is None else LegacyTrainMeanAdapter(n_contacts)
        )

    def trajectory(
        self,
        event_token: Tensor,
        event_time: Tensor,
        event_segment: Tensor,
        anchor_time: Tensor,
        last_event_pos: Tensor,
        train_anchor_rows: Tensor,
    ) -> Tensor:
        write = self.encoder(event_token)
        _pre, post = self.state_bank(write, event_time, event_segment)
        anchor = self.state_bank.anchor(post, event_time, anchor_time, last_event_pos)
        reference = anchor[train_anchor_rows]
        centre = reference.mean(0)
        scale = reference.std(0, unbiased=False).clamp_min(1e-4)
        return (anchor - centre) / scale

    def legacy_event_nll(
        self, group_ids: Tensor, group_count: Tensor, state: Tensor
    ) -> Tensor:
        if self.legacy is None:
            raise RuntimeError("legacy decoder was not attached")
        output = self.legacy(group_ids, group_count, state)
        if self.train_mean_adapter is not None:
            output = self.train_mean_adapter.apply(output)
        return next_set_stop_loss(output, group_ids, group_count)["event_nll"]


def build_optimizer(
    model: SpatialStateModel, config: OptimizerConfig
) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    config.validate()
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    adapter = [
        p for p in model.legacy.residual.parameters() if p.requires_grad
    ] if model.legacy is not None else []
    auxiliary = [p for p in model.functional.parameters() if p.requires_grad]
    groups = [
        {"name": "encoder", "params": encoder, "lr": config.lr_encoder},
        {"name": "state_adapter", "params": adapter, "lr": config.lr_state_adapter},
        {"name": "auxiliary", "params": auxiliary, "lr": config.lr_auxiliary},
    ]
    groups = [group for group in groups if group["params"]]
    assigned = [id(p) for group in groups for p in group["params"]]
    expected = [id(p) for p in model.parameters() if p.requires_grad]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(expected):
        raise RuntimeError("spatial-state optimizer groups are incomplete or duplicated")
    optimizer = torch.optim.AdamW(
        groups,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    return optimizer, optimizer_contract(config)
