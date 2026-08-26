"""State-conditioned event readout.

``p_psi(c_{e,k+1}, STOP | c_{e,1:k}, G_p, mu_p, H_e^-, r_e^-)``.

The readout is factorised so that four endpoints can be reported separately and
one of them is not a synonym for how many contacts participated:

``order_nll``          Plackett-Luce over the *true participating set* only.  It is
                       a pure ordering endpoint: it is invariant to how many
                       contacts took part.
``selection_nll``      next contact among all not-yet-recruited contacts.  Joint
                       set-and-order, and therefore load-coupled.
``stop_nll``           continue-versus-stop at each recruitment step.
``participation_nll``  per-contact Bernoulli participation.

Ties are exchangeable multi-selections at one step, taken from the explicit group
identity; no within-tie order is ever invented.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

NEG = -1e9


@dataclass
class EventSteps:
    """Padded per-step tensors for a chunk of events."""

    recruited: torch.Tensor      # (T, S, N) float, contacts already recruited
    target: torch.Tensor         # (T, S, N) float, the tied group selected at this step
    pl_candidate: torch.Tensor   # (T, S, N) bool, remaining participants
    seq_candidate: torch.Tensor  # (T, S, N) bool, all contacts not yet recruited
    select_step: torch.Tensor    # (T, S) bool, a group is selected at this step
    active_step: torch.Tensor    # (T, S) bool, select steps plus the terminal stop step
    multiplicity: torch.Tensor   # (T, S) float, |S_k|
    step_norm: torch.Tensor      # (T, S) float, k / n_groups
    recruited_frac: torch.Tensor # (T, S) float
    n_participants: torch.Tensor # (T,) float
    n_groups: torch.Tensor       # (T,) float


def build_event_steps(participation: torch.Tensor, group_ids: torch.Tensor,
                      n_groups: torch.Tensor) -> EventSteps:
    """Expand a chunk of events into padded recruitment steps."""
    T, N = participation.shape
    S = int(n_groups.max().item()) + 1              # + terminal stop step
    device = participation.device
    k = torch.arange(S, device=device).view(1, S, 1)
    grp = torch.where(participation, group_ids, torch.full_like(group_ids, 1 << 20)).unsqueeze(1)
    part = participation.unsqueeze(1)

    recruited = (part & (grp < k)).float()
    target = (part & (grp == k)).float()
    pl_candidate = part & (grp >= k)
    seq_candidate = ~(part & (grp < k))
    select_step = k.squeeze(-1) < n_groups.view(T, 1)
    active_step = k.squeeze(-1) <= n_groups.view(T, 1)
    multiplicity = target.sum(-1)
    denom = n_groups.clamp(min=1).float().view(T, 1)
    step_norm = torch.arange(S, device=device).view(1, S).float() / denom
    n_part = participation.sum(-1).float()
    recruited_frac = recruited.sum(-1) / n_part.clamp(min=1).view(T, 1)
    return EventSteps(recruited, target, pl_candidate, seq_candidate, select_step,
                      active_step, multiplicity, step_norm, recruited_frac,
                      n_part, n_groups.float())


class EventDecoder(nn.Module):
    """Shared decoder; the fast event state is the recruited-set graph diffusion."""

    def __init__(self, state_dim: int, feature_dim: int, n_relations: int = 3,
                 hidden: int = 16):
        super().__init__()
        self.n_relations = n_relations
        self.static_node = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.static_node.weight)
        nn.init.zeros_(self.static_node.bias)
        self.prefix_weight = nn.Parameter(torch.zeros(n_relations))
        self.step_weight = nn.Parameter(torch.zeros(2))
        self.stop_head = nn.Sequential(
            nn.Linear(state_dim + 4, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.stop_head[-1].weight)
        nn.init.zeros_(self.stop_head[-1].bias)
        self.participation_head = nn.Linear(state_dim + feature_dim, 1)
        nn.init.zeros_(self.participation_head.weight)
        nn.init.zeros_(self.participation_head.bias)

    def forward(self, steps: EventSteps, *, adjacency: torch.Tensor,
                node_features: torch.Tensor, baseline_order: torch.Tensor,
                baseline_participation: torch.Tensor, baseline_stop: torch.Tensor,
                state: torch.Tensor, resource: torch.Tensor,
                adapter: dict, return_steps: bool = False) -> dict[str, torch.Tensor]:
        T, S, N = steps.recruited.shape
        # a no-state adapter must be state-free everywhere, including the heads
        # that do not go through the adapter at all
        if not adapter.get("state_visible", True):
            state = torch.zeros_like(state)
            resource = torch.ones_like(resource)

        base = baseline_order.view(1, N) + self.static_node(node_features).view(1, N)
        node_shift = adapter.get("node_shift")
        node_scale = adapter.get("node_scale")
        if node_scale is not None:
            base = base * node_scale
        if node_shift is not None:
            base = base + node_shift
        base = base.view(T, 1, N) if base.dim() == 2 and base.shape[0] == T else base.view(1, 1, N).expand(T, 1, N)

        adj = adjacency
        edge_gate = adapter.get("edge_gate")
        if edge_gate is not None:
            adj = adj.unsqueeze(0) * edge_gate           # (T, R, N, N)
            neighbour = torch.einsum("tkj,trji->tkri", steps.recruited, adj)
        else:
            neighbour = torch.einsum("tkj,rji->tkri", steps.recruited, adj)
        prefix = torch.einsum("tkri,r->tki", neighbour, self.prefix_weight)

        global_terms = adapter["global"]
        step_bias = global_terms[:, 0].view(T, 1, 1)
        stop_bias = global_terms[:, 1].view(T, 1)
        part_bias = global_terms[:, 2].view(T, 1)

        step_feat = (self.step_weight[0] * steps.step_norm + self.step_weight[1] * steps.recruited_frac)
        logits = base + prefix + step_feat.unsqueeze(-1) + step_bias

        order_per_step = self._selection_terms(logits, steps, steps.pl_candidate)
        selection_per_step = self._selection_terms(logits, steps, steps.seq_candidate)
        order_nll = -(order_per_step * steps.select_step.float()).sum(-1) / steps.n_participants.clamp(min=1)
        selection_nll = -(selection_per_step * steps.select_step.float()).sum(-1) / steps.n_participants.clamp(min=1)

        pooled_state = state.mean(dim=-2)                                  # (T, D)
        stop_in = torch.cat([
            pooled_state.unsqueeze(1).expand(T, S, pooled_state.shape[-1]),
            steps.step_norm.unsqueeze(-1),
            steps.recruited_frac.unsqueeze(-1),
            resource.view(T, 1, 1).expand(T, S, 1),
            steps.n_participants.view(T, 1, 1).expand(T, S, 1) / float(N),
        ], dim=-1)
        stop_logit = self.stop_head(stop_in).squeeze(-1) + baseline_stop + stop_bias
        continue_target = steps.select_step.float()
        stop_terms = torch.nn.functional.binary_cross_entropy_with_logits(
            -stop_logit, continue_target, reduction="none")
        stop_nll = (stop_terms * steps.active_step.float()).sum(-1) / steps.active_step.float().sum(-1).clamp(min=1)

        part_in = torch.cat([state, node_features.unsqueeze(0).expand(T, N, node_features.shape[-1])], dim=-1)
        part_logit = self.participation_head(part_in).squeeze(-1) + baseline_participation.view(1, N) + part_bias
        participation_target = (steps.target.sum(1) > 0).float()
        participation_nll = torch.nn.functional.binary_cross_entropy_with_logits(
            part_logit, participation_target, reduction="none").mean(-1)

        out = {
            "order_nll": order_nll,
            "selection_nll": selection_nll,
            "stop_nll": stop_nll,
            "participation_nll": participation_nll,
            "event_nll": selection_nll + stop_nll,
        }
        if return_steps:
            # per-recruitment-step log-probabilities, so a targeted analysis can
            # read the branch at one prefix depth instead of the whole event
            out["order_step_logprob"] = order_per_step
            out["selection_step_logprob"] = selection_per_step
            out["select_step"] = steps.select_step
            out["n_groups"] = steps.n_groups
        return out

    @staticmethod
    def _selection_terms(logits: torch.Tensor, steps: EventSteps,
                         candidate: torch.Tensor) -> torch.Tensor:
        """Per-step log-probability of the tie group selected at that step.

        A tie of size m is an exchangeable multi-selection: all m members are
        drawn from the same step-k distribution and removed together, so no
        within-tie order is ever invented.
        """
        masked = torch.where(candidate, logits, torch.full_like(logits, NEG))
        normaliser = torch.logsumexp(masked, dim=-1)
        chosen = (logits * steps.target).sum(-1)
        return chosen - steps.multiplicity * normaliser


def chunk_step_cost(n_groups: np.ndarray, n_contacts: int) -> int:
    """Padded step-tensor element count for a chunk; used to size chunks."""
    return int((n_groups.max() + 1) * n_contacts)
