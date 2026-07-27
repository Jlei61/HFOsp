"""Minimal categorical competitive-propagation RNN for Topic-5 v2.3."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


EPS = 1.0e-8


@dataclass(frozen=True)
class EventForward:
    losses: torch.Tensor
    probabilities: tuple[torch.Tensor, ...]
    eligible_indices: tuple[torch.Tensor, ...]
    targets: tuple[int, ...]
    propagation_states: tuple[torch.Tensor, ...]
    competition_states: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class BatchForward:
    event_losses: torch.Tensor
    decision_loss_sum: torch.Tensor
    decision_count: torch.Tensor


def nearest_neighbour_scale(coords: torch.Tensor) -> torch.Tensor:
    distance = torch.cdist(coords, coords)
    eye = torch.eye(len(coords), dtype=torch.bool, device=coords.device)
    distance = distance.masked_fill(eye | (distance <= EPS), torch.inf)
    nearest = distance.min(dim=1).values
    finite = nearest[torch.isfinite(nearest)]
    if finite.numel() == 0:
        raise ValueError("contact geometry has no non-zero neighbour distance")
    return finite.median()


def kernel_bases(
    coords: torch.Tensor,
    axis: torch.Tensor,
    *,
    anisotropy_ratio: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Frobenius-normalized local/axis bases and axis projection."""
    coords = coords.to(dtype=torch.float64)
    axis = axis.to(dtype=torch.float64)
    axis = axis / torch.linalg.vector_norm(axis).clamp_min(EPS)
    centered = coords - coords.mean(dim=0, keepdim=True)
    delta = centered[:, None, :] - centered[None, :, :]
    distance_sq = torch.sum(delta * delta, dim=-1)
    parallel = torch.einsum("ijd,d->ij", delta, axis)
    perpendicular_sq = torch.clamp(distance_sq - parallel * parallel, min=0.0)
    scale = nearest_neighbour_scale(centered)
    local = torch.exp(-distance_sq / (2.0 * scale * scale))
    axial = torch.exp(
        -parallel * parallel
        / (2.0 * (anisotropy_ratio * scale) ** 2)
        - perpendicular_sq / (2.0 * scale * scale)
    )
    eye = torch.eye(len(coords), dtype=torch.bool, device=coords.device)
    local = local.masked_fill(eye, 0.0)
    axial = axial.masked_fill(eye, 0.0)
    local = local / torch.linalg.matrix_norm(local).clamp_min(EPS)
    axial = axial / torch.linalg.matrix_norm(axial).clamp_min(EPS)
    projection = centered @ axis
    return local, axial, projection


def symmetric_normalize(adjacency: torch.Tensor) -> torch.Tensor:
    adjacency = (adjacency + adjacency.T) / 2.0
    degree = adjacency.sum(dim=1).clamp_min(EPS)
    inverse = torch.rsqrt(degree)
    return inverse[:, None] * adjacency * inverse[None, :]


def directional_basis(
    symmetric: torch.Tensor, projection: torch.Tensor
) -> torch.Tensor:
    difference = projection[None, :] - projection[:, None]
    nonzero = torch.abs(difference[difference != 0])
    scale = (
        nonzero.median()
        if nonzero.numel() > 0
        else torch.ones((), dtype=symmetric.dtype, device=symmetric.device)
    )
    directed = symmetric * torch.tanh(difference / scale.clamp_min(EPS))
    return (directed - directed.T) / 2.0


def source_direction_score(
    event: torch.Tensor, projection: torch.Tensor
) -> torch.Tensor:
    source = event == 0
    scale = projection.std(unbiased=False).clamp_min(EPS)
    return torch.tanh(projection[source].mean() / scale)


def has_non_source_tie(event: np.ndarray | torch.Tensor) -> bool:
    values = np.asarray(event, dtype=np.int64)
    ranks = values[values > 0]
    if ranks.size == 0:
        return False
    _, counts = np.unique(ranks, return_counts=True)
    return bool(np.any(counts > 1))


class CompetitivePropagationRNN(nn.Module):
    """Four-scalar patient model with no dense contact-mixing bypass."""

    def __init__(
        self,
        *,
        coords: np.ndarray | torch.Tensor,
        axis: np.ndarray | torch.Tensor,
        node_logit: np.ndarray | torch.Tensor,
        rho_propagation: float,
        rho_competition: float,
        anisotropy_ratio: float = 2.0,
        local_only: bool = False,
        no_competition: bool = False,
        no_source: bool = False,
        no_history: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= rho_propagation < rho_competition < 1.0:
            raise ValueError("require 0 <= rho_propagation < rho_competition < 1")
        coords_tensor = torch.as_tensor(coords, dtype=torch.float64)
        axis_tensor = torch.as_tensor(axis, dtype=torch.float64)
        node_tensor = torch.as_tensor(node_logit, dtype=torch.float64)
        if coords_tensor.ndim != 2 or coords_tensor.shape[1] != 3:
            raise ValueError("coords must have shape [contacts, 3]")
        if node_tensor.shape != (len(coords_tensor),):
            raise ValueError("node_logit must match contact count")
        local, axial, projection = kernel_bases(
            coords_tensor, axis_tensor, anisotropy_ratio=anisotropy_ratio
        )
        self.register_buffer("local_basis", local)
        self.register_buffer("axis_basis", axial)
        self.register_buffer("projection", projection)
        self.register_buffer("node_logit", node_tensor)
        self.register_buffer(
            "rho_propagation",
            torch.tensor(0.0 if no_history else rho_propagation, dtype=torch.float64),
        )
        self.register_buffer(
            "rho_competition",
            torch.tensor(0.0 if no_history else rho_competition, dtype=torch.float64),
        )
        self.local_only = bool(local_only)
        self.no_competition = bool(no_competition)
        self.no_source = bool(no_source)
        self.no_history = bool(no_history)

        self.raw_gamma = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.raw_gain_propagation = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float64)
        )
        self.raw_gain_competition = nn.Parameter(
            torch.tensor(-1.0, dtype=torch.float64)
        )
        self.raw_source_beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    @property
    def gamma(self) -> torch.Tensor:
        if self.local_only:
            return torch.zeros((), dtype=torch.float64, device=self.raw_gamma.device)
        return torch.sigmoid(self.raw_gamma)

    @property
    def gain_propagation(self) -> torch.Tensor:
        return F.softplus(self.raw_gain_propagation)

    @property
    def gain_competition(self) -> torch.Tensor:
        if self.no_competition:
            return torch.zeros(
                (), dtype=torch.float64, device=self.raw_gain_competition.device
            )
        return F.softplus(self.raw_gain_competition)

    @property
    def source_beta(self) -> torch.Tensor:
        if self.no_source:
            return torch.zeros(
                (), dtype=torch.float64, device=self.raw_source_beta.device
            )
        return 4.0 * torch.tanh(self.raw_source_beta)

    def operators(self) -> tuple[torch.Tensor, torch.Tensor]:
        adjacency = (
            (1.0 - self.gamma) * self.local_basis
            + self.gamma * self.axis_basis
        )
        symmetric = symmetric_normalize(adjacency)
        directed = directional_basis(symmetric, self.projection)
        return symmetric, directed

    def forward_event(
        self, event: np.ndarray | torch.Tensor
    ) -> EventForward:
        event = torch.as_tensor(event, dtype=torch.long, device=self.node_logit.device)
        if has_non_source_tie(event.detach().cpu().numpy()):
            raise ValueError("primary categorical model excludes non-source tied ranks")
        valid = event >= 0
        if not torch.any(valid):
            raise ValueError("event has no participating contacts")
        n_steps = int(event[valid].max().item()) + 1
        if n_steps < 2:
            empty = torch.empty(0, dtype=torch.float64, device=event.device)
            return EventForward(empty, (), (), (), (), ())

        symmetric, directed = self.operators()
        source_score = source_direction_score(event, self.projection)
        propagation = torch.zeros_like(self.node_logit)
        competition = torch.zeros_like(self.node_logit)
        losses: list[torch.Tensor] = []
        probabilities: list[torch.Tensor] = []
        eligible_rows: list[torch.Tensor] = []
        targets: list[int] = []
        propagation_rows: list[torch.Tensor] = []
        competition_rows: list[torch.Tensor] = []
        for step in range(n_steps - 1):
            current = event == step
            x = current.to(torch.float64)
            x = x / x.sum().clamp_min(1.0)
            propagation = (
                self.rho_propagation * propagation + x @ symmetric
            )
            competition = (
                self.rho_competition * competition + x @ symmetric
            )
            directional_drive = x @ directed
            score = (
                self.node_logit
                + self.gain_propagation * propagation
                - self.gain_competition * competition
                + self.source_beta * source_score * directional_drive
            )
            seen = (event >= 0) & (event <= step)
            eligible = torch.nonzero(~seen, as_tuple=False).flatten()
            target_contact = torch.nonzero(
                event == (step + 1), as_tuple=False
            ).flatten()
            if target_contact.numel() != 1:
                raise ValueError("primary target rank must contain one contact")
            target_position = torch.nonzero(
                eligible == target_contact.item(), as_tuple=False
            ).flatten()
            if target_position.numel() != 1:
                raise RuntimeError("next contact is not eligible")
            logits = score[eligible]
            target_index = int(target_position.item())
            losses.append(
                F.cross_entropy(
                    logits.unsqueeze(0),
                    torch.tensor([target_index], device=event.device),
                )
            )
            probabilities.append(torch.softmax(logits, dim=0))
            eligible_rows.append(eligible)
            targets.append(target_index)
            propagation_rows.append(propagation)
            competition_rows.append(competition)
        return EventForward(
            losses=torch.stack(losses),
            probabilities=tuple(probabilities),
            eligible_indices=tuple(eligible_rows),
            targets=tuple(targets),
            propagation_states=tuple(propagation_rows),
            competition_states=tuple(competition_rows),
        )

    def mean_event_nll(self, event: np.ndarray | torch.Tensor) -> torch.Tensor:
        result = self.forward_event(event)
        if result.losses.numel() == 0:
            return torch.zeros((), dtype=torch.float64, device=self.node_logit.device)
        return result.losses.mean()

    def forward_batch(
        self,
        groups: np.ndarray | torch.Tensor,
        group_count: np.ndarray | torch.Tensor,
    ) -> BatchForward:
        """Vectorized event-first categorical NLL for untied rank sequences."""
        groups = torch.as_tensor(
            groups, dtype=torch.long, device=self.node_logit.device
        )
        group_count = torch.as_tensor(
            group_count, dtype=torch.long, device=self.node_logit.device
        )
        if groups.ndim != 2 or groups.shape[1] != self.node_logit.numel():
            raise ValueError("groups must have shape [events, contacts]")
        if group_count.shape != (len(groups),):
            raise ValueError("group_count must have shape [events]")
        if torch.any(group_count < 1):
            raise ValueError("each event must contain at least one rank")
        has_tie = any(
            bool(torch.any((groups == rank).sum(dim=1) > 1))
            for rank in range(1, groups.shape[1])
        )
        if has_tie:
            raise ValueError("primary categorical model excludes non-source tied ranks")

        symmetric, directed = self.operators()
        batch_size, n_contacts = groups.shape
        dtype = self.node_logit.dtype
        propagation = torch.zeros(
            (batch_size, n_contacts), dtype=dtype, device=groups.device
        )
        competition = torch.zeros_like(propagation)
        source = groups == 0
        source_count = source.sum(dim=1).clamp_min(1).to(dtype)
        source_projection = (
            source.to(dtype) * self.projection[None, :]
        ).sum(dim=1) / source_count
        source_scale = self.projection.std(unbiased=False).clamp_min(EPS)
        source_score = torch.tanh(source_projection / source_scale)
        loss_sum = torch.zeros(batch_size, dtype=dtype, device=groups.device)
        decision_count = torch.zeros_like(loss_sum)

        max_count = int(group_count.max().item())
        for step in range(max_count - 1):
            active = group_count > (step + 1)
            current = groups == step
            current_count = current.sum(dim=1).clamp_min(1).to(dtype)
            x = current.to(dtype) / current_count[:, None]
            propagation = self.rho_propagation * propagation + x @ symmetric
            competition = self.rho_competition * competition + x @ symmetric
            directional_drive = x @ directed
            score = (
                self.node_logit[None, :]
                + self.gain_propagation * propagation
                - self.gain_competition * competition
                + self.source_beta
                * source_score[:, None]
                * directional_drive
            )
            seen = (groups >= 0) & (groups <= step)
            eligible = ~seen
            masked_score = score.masked_fill(~eligible, -torch.inf)
            target = groups == (step + 1)
            target_count = target.sum(dim=1)
            if torch.any(active & (target_count != 1)):
                raise ValueError("primary target rank must contain one contact")
            target_score = torch.where(
                target, score, torch.zeros_like(score)
            ).sum(dim=1)
            decision_nll = torch.logsumexp(masked_score, dim=1) - target_score
            decision_nll = torch.where(
                active, decision_nll, torch.zeros_like(decision_nll)
            )
            loss_sum = loss_sum + decision_nll
            decision_count = decision_count + active.to(dtype)
        event_losses = torch.where(
            decision_count > 0,
            loss_sum / decision_count.clamp_min(1.0),
            torch.zeros_like(loss_sum),
        )
        return BatchForward(
            event_losses=event_losses,
            decision_loss_sum=loss_sum,
            decision_count=decision_count,
        )

    def parameter_summary(self) -> dict[str, float]:
        return {
            "gamma": float(self.gamma.detach().cpu()),
            "gain_propagation": float(
                self.gain_propagation.detach().cpu()
            ),
            "gain_competition": float(
                self.gain_competition.detach().cpu()
            ),
            "source_beta": float(self.source_beta.detach().cpu()),
            "rho_propagation": float(self.rho_propagation.detach().cpu()),
            "rho_competition": float(self.rho_competition.detach().cpu()),
        }
