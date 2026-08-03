"""Minimal static-anchored residual models for Topic 5 v0.4.

The module contains only the differentiable field composition, the sign-free
dual-candidate loss, and the two allowed history representations.  It does not
load targets, choose folds, run statistics, or define scientific gates.
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from src.topic5_history_rnn import TimeDecayHistoryGRU


def center(value: Tensor) -> Tensor:
    return value - value.mean(dim=-1, keepdim=True)


def unit_eps(value: Tensor, epsilon: float = 1e-8) -> Tensor:
    centered = center(value)
    norm = torch.sqrt(centered.square().sum(dim=-1, keepdim=True) + float(epsilon))
    return centered / norm


def safe_unit_residual(
    value: Tensor,
    *,
    epsilon: float = 1e-8,
    norm_threshold: float = 1e-6,
) -> Tensor:
    centered = center(value)
    norm = torch.sqrt(centered.square().sum(dim=-1, keepdim=True))
    normalized = centered / (norm + float(epsilon))
    return torch.where(norm >= float(norm_threshold), normalized, torch.zeros_like(value))


def compose_static_residual(
    static: Tensor,
    residual: Tensor,
    gain: Tensor | float,
    *,
    epsilon: float = 1e-8,
    norm_threshold: float = 1e-6,
) -> Tensor:
    base = unit_eps(static, epsilon)
    correction = safe_unit_residual(
        residual, epsilon=epsilon, norm_threshold=norm_threshold
    )
    gain_tensor = torch.as_tensor(gain, dtype=base.dtype, device=base.device)
    composed = unit_eps(base + gain_tensor * correction, epsilon)
    return torch.where(gain_tensor == 0, base, composed)


def soft_rank(value: Tensor, temperature: float = 0.1) -> Tensor:
    """Differentiable rank with ascending-rank orientation."""

    if value.ndim != 1:
        raise ValueError("soft_rank expects a one-dimensional contact field")
    difference = value.unsqueeze(0) - value.unsqueeze(1)  # [i,j] = v_j-v_i
    probability = torch.sigmoid(difference / float(temperature))
    probability = probability - torch.diag_embed(torch.diagonal(probability))
    return 1.0 + probability.sum(dim=-1)


def _correlation(a: Tensor, b: Tensor, epsilon: float = 1e-8) -> Tensor:
    left = center(a)
    right = center(b)
    denominator = torch.sqrt(left.square().sum() * right.square().sum() + float(epsilon))
    return (left * right).sum() / denominator


def soft_maxab_score(
    prediction_a: Tensor,
    prediction_b: Tensor,
    target_midrank: Tensor,
    *,
    rank_temperature: float = 0.1,
    max_temperature: float = 0.05,
) -> Tensor:
    rank_a = soft_rank(prediction_a, rank_temperature)
    rank_b = soft_rank(prediction_b, rank_temperature)
    correlations = torch.stack(
        [
            _correlation(rank_a, target_midrank).abs(),
            _correlation(rank_b, target_midrank).abs(),
        ]
    )
    weight = torch.softmax(correlations / float(max_temperature), dim=0)
    return torch.sum(weight * correlations)


def patient_balanced_soft_maxab(
    predictions: Iterable[tuple[Tensor, Tensor, Tensor]],
    *,
    rank_temperature: float = 0.1,
    max_temperature: float = 0.05,
) -> Tensor:
    scores = [
        soft_maxab_score(
            field_a,
            field_b,
            target_rank,
            rank_temperature=rank_temperature,
            max_temperature=max_temperature,
        )
        for field_a, field_b, target_rank in predictions
    ]
    if not scores:
        raise ValueError("a patient batch must contain at least one seizure")
    return torch.stack(scores).mean()


class DualCandidateResidualHead(nn.Module):
    """Two bias-free contact-query residual heads plus shared-cohort gains."""

    def __init__(
        self,
        state_dim: int,
        contact_dim: int,
        *,
        initial_gain: float = 1e-3,
        epsilon: float = 1e-8,
        norm_threshold: float = 1e-6,
    ):
        super().__init__()
        if not 0 < float(initial_gain) < 1:
            raise ValueError("initial_gain must be between zero and one")
        self.query_a = nn.Linear(int(state_dim), int(contact_dim), bias=False)
        self.query_b = nn.Linear(int(state_dim), int(contact_dim), bias=False)
        nn.init.normal_(self.query_a.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.query_b.weight, mean=0.0, std=1e-3)
        # Squared coordinates retain a usable gradient near zero.  A sigmoid
        # initialized at 1e-3 is numerically saturated and made the smoke model
        # indistinguishable from static even when the residual loss was finite.
        self.raw_gain = nn.Parameter(
            torch.full((2,), float(initial_gain) ** 0.5)
        )
        self.epsilon = float(epsilon)
        self.norm_threshold = float(norm_threshold)

    @property
    def gains(self) -> Tensor:
        return self.raw_gain.square().clamp(max=1.0)

    def residuals(self, state: Tensor, contact_embedding: Tensor) -> tuple[Tensor, Tensor]:
        if state.ndim != 1 or contact_embedding.ndim != 2:
            raise ValueError("state/contact_embedding shapes must be [state] and [contact,feature]")
        residual_a = contact_embedding @ self.query_a(state)
        residual_b = contact_embedding @ self.query_b(state)
        return residual_a, residual_b

    def forward(
        self,
        state: Tensor,
        contact_embedding: Tensor,
        static_a: Tensor,
        static_b: Tensor,
    ) -> dict[str, Tensor]:
        residual_a, residual_b = self.residuals(state, contact_embedding)
        return {
            "candidate_a": compose_static_residual(
                static_a,
                residual_a,
                self.gains[0],
                epsilon=self.epsilon,
                norm_threshold=self.norm_threshold,
            ),
            "candidate_b": compose_static_residual(
                static_b,
                residual_b,
                self.gains[1],
                epsilon=self.epsilon,
                norm_threshold=self.norm_threshold,
            ),
            "residual_a": residual_a,
            "residual_b": residual_b,
            "gains": self.gains,
        }


class TimeAwareNonrecurrentResidual(nn.Module):
    """A single linear 130->16 projection followed by the common field head."""

    def __init__(
        self,
        summary_dim: int,
        state_dim: int,
        contact_dim: int,
        **head_kwargs,
    ):
        super().__init__()
        self.summary_projection = nn.Linear(int(summary_dim), int(state_dim))
        nn.init.xavier_uniform_(self.summary_projection.weight, gain=0.1)
        nn.init.zeros_(self.summary_projection.bias)
        self.head = DualCandidateResidualHead(state_dim, contact_dim, **head_kwargs)

    def forward(
        self,
        summary: Tensor,
        contact_embedding: Tensor,
        static_a: Tensor,
        static_b: Tensor,
    ) -> dict[str, Tensor]:
        state = self.summary_projection(summary)
        output = self.head(state, contact_embedding, static_a, static_b)
        output["state"] = state
        return output


def fixed_time_aware_summary(
    event_embedding: Tensor,
    event_time: Tensor,
    cutoff_time: Tensor | float,
    *,
    tau_hours: float = 2.0,
) -> Tensor:
    """Return EWMA, mean, max, last, log-count and log-span."""

    if event_embedding.ndim != 2 or event_time.ndim != 1:
        raise ValueError("history inputs must be [event,feature] and [event]")
    if len(event_embedding) != len(event_time) or len(event_time) == 0:
        raise ValueError("history inputs are empty or misaligned")
    cutoff = torch.as_tensor(cutoff_time, dtype=event_time.dtype, device=event_time.device)
    age = (cutoff - event_time).clamp_min(0.0)
    tau_seconds = float(tau_hours) * 3600.0
    weight = torch.exp(-age / tau_seconds)
    ewma = (event_embedding * weight[:, None]).sum(0) / weight.sum().clamp_min(1e-12)
    mean = event_embedding.mean(0)
    maximum = event_embedding.max(0).values
    last = event_embedding[-1]
    count = torch.log1p(event_embedding.new_tensor(float(len(event_embedding))))
    span = torch.log1p((event_time[-1] - event_time[0]).clamp_min(0.0))
    return torch.cat([ewma, mean, maximum, last, count[None], span[None]])


def run_history_to_cutoff(
    history: TimeDecayHistoryGRU,
    event_embedding: Tensor,
    event_time: Tensor,
    cutoff_time: Tensor | float,
    *,
    chunk_events: int = 256,
) -> Tensor:
    """Run the complete history without detaching and decay to the cutoff."""

    if event_embedding.ndim != 2 or event_time.ndim != 1:
        raise ValueError("history inputs must be [event,feature] and [event]")
    if len(event_embedding) != len(event_time) or len(event_time) == 0:
        raise ValueError("history inputs are empty or misaligned")
    state = None
    for start in range(0, len(event_time), int(chunk_events)):
        stop = min(start + int(chunk_events), len(event_time))
        local_time = event_time[start:stop]
        delta = torch.zeros_like(local_time)
        if start > 0:
            delta[0] = local_time[0] - event_time[start - 1]
        if len(local_time) > 1:
            delta[1:] = local_time[1:] - local_time[:-1]
        reset = torch.zeros((1, stop - start), dtype=torch.bool, device=event_time.device)
        if start == 0:
            reset[:, 0] = True
        mask = torch.ones_like(reset)
        _, state = history.forward_masked(
            event_embedding[start:stop].unsqueeze(0),
            delta.unsqueeze(0),
            reset,
            mask,
            initial_state=state,
        )
    cutoff = torch.as_tensor(cutoff_time, dtype=event_time.dtype, device=event_time.device)
    return history.decay(state, (cutoff - event_time[-1]).clamp_min(0.0))[0]
