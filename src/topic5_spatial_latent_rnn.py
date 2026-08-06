"""Model family for the Topic 5 spatial latent propagation RNN v0.1.

Five arms share the same prediction task and the same loss.  They do NOT
share an output head: the recurrent arm reads out densely from a hidden
vector, the contact-graph arm reads one scalar per contact node, and the
latent arms emit per tissue unit and project through the fixed operator H.
So this is a comparison of whole parameterisations, not a factorial
decomposition of one factor at a time:

``STATIC_CONTACT``            per-contact frequency floor, no recurrence
``ORDINARY_GRU``              one unconstrained recurrent state, free readout
``CONTACT_GRAPH_RNN``         one node per contact, learned directed adjacency
``LATENT_FIXED_LOCAL_RNN``    latent nodes, fixed kNN graph
``LATENT_LEARNED_SPATIAL_RNN`` latent nodes, graph learned under wiring economy

In the latent arms a contact is an observation port, never a node.  The observed
rank set enters through ``H.T`` and the prediction leaves through the same ``H``;
there is no dense contact-to-contact path anywhere, so every cross-space
prediction has to travel over the learned graph.

Geometry enters the wiring loss only.  It never reaches a node feature or the
output head.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

ARMS = (
    "STATIC_CONTACT",
    "ORDINARY_GRU",
    "CONTACT_GRAPH_RNN",
    "LATENT_FIXED_LOCAL_RNN",
    "LATENT_LEARNED_SPATIAL_RNN",
    "LATENT_DENSE_RNN",
)
LATENT_ARMS = ("LATENT_FIXED_LOCAL_RNN", "LATENT_LEARNED_SPATIAL_RNN", "LATENT_DENSE_RNN")

# Hard-Concrete stretch interval (Louizos, Welling & Kingma 2018).
HC_GAMMA, HC_ZETA = -0.1, 1.1
NEG_INF = -1e9


@dataclass
class ModelConfig:
    arm: str
    n_contacts: int
    n_nodes: int = 0
    hidden: int = 4
    microsteps: int = 3
    ordinary_hidden: int = 64
    edge_budget: float = 6.0
    knn_k: int = 6
    use_contact_bias: bool = True
    logit_scale_init: float = 4.0
    gate_init_log_alpha: float = 2.0
    seed: int = 0
    # geometry, used by the wiring loss only
    normalised_distance: np.ndarray | None = field(default=None, repr=False)
    fixed_edge_mask: np.ndarray | None = field(default=None, repr=False)
    observation_operator: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; use one of {ARMS}")
        if self.arm in LATENT_ARMS and self.observation_operator is None:
            raise ValueError(f"{self.arm} needs an observation operator")


@dataclass(frozen=True)
class EventTensors:
    """Padded teacher-forcing view of a set of events."""

    x: Tensor            # (B, T, C) rank set observed at step t
    recruited: Tensor    # (B, T, C) everything recruited up to and including t
    available: Tensor    # (B, T, C) candidates for step t+1
    target: Tensor       # (B, T, C) rank set at step t+1
    valid: Tensor        # (B, T)    step exists
    is_last: Tensor      # (B, T)    t is the final rank of the event

    def to(self, device: torch.device) -> "EventTensors":
        return EventTensors(*[v.to(device) for v in
                              (self.x, self.recruited, self.available,
                               self.target, self.valid, self.is_last)])


def build_event_tensors(group_ids: np.ndarray) -> EventTensors:
    """Turn dense per-event contact ranks into padded step tensors.

    ``group_ids[e, c]`` is the rank at which contact ``c`` joins event ``e``, or
    -1.  Ranks are already densified by the cache builder.
    """
    ranks = np.asarray(group_ids)
    n_events, n_contacts = ranks.shape
    lengths = np.array([int(r[r >= 0].max()) + 1 if np.any(r >= 0) else 0 for r in ranks])
    steps = int(lengths.max())

    x = np.zeros((n_events, steps, n_contacts), np.float32)
    target = np.zeros_like(x)
    recruited = np.zeros_like(x)
    valid = np.zeros((n_events, steps), bool)
    is_last = np.zeros((n_events, steps), bool)

    for e, row in enumerate(ranks):
        length = lengths[e]
        for t in range(length):
            member = row == t
            x[e, t, member] = 1.0
            recruited[e, t] = (row >= 0) & (row <= t)
            if t + 1 < length:
                target[e, t, row == t + 1] = 1.0
        valid[e, :length] = True
        if length:
            is_last[e, length - 1] = True

    available = (recruited == 0) & valid[:, :, None]
    return EventTensors(
        x=torch.from_numpy(x),
        recruited=torch.from_numpy(recruited),
        available=torch.from_numpy(available),
        target=torch.from_numpy(target),
        valid=torch.from_numpy(valid),
        is_last=torch.from_numpy(is_last),
    )


class HardConcreteGate(nn.Module):
    """Stochastic gate whose expected L0 is differentiable.

    ``log_alpha`` starts positive so the warm-up phase begins with an effectively
    dense graph; the wiring penalty then has to buy each edge back.
    """

    # Once a gate's value hits the hard 0 of the stretch clamp its gradient dies
    # and the edge can never be revived, so log_alpha is kept inside a range
    # where the sigmoid still has slope.  Without this an over-strong penalty
    # annihilates the graph in a couple of epochs and nothing can pull it back.
    LOG_ALPHA_RANGE = (-4.0, 4.0)

    def __init__(self, shape: Sequence[int], init_log_alpha: float = 2.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full(tuple(shape), float(init_log_alpha)))

    def _bounded(self) -> Tensor:
        low, high = self.LOG_ALPHA_RANGE
        return low + (high - low) * torch.sigmoid(self.log_alpha)

    def forward(self, temperature: float) -> Tensor:
        log_alpha = self._bounded()
        if self.training:
            u = torch.rand_like(log_alpha).clamp_(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + log_alpha) / temperature)
        else:
            s = torch.sigmoid(log_alpha / temperature)
        return (s * (HC_ZETA - HC_GAMMA) + HC_GAMMA).clamp(0.0, 1.0)

    def open_probability(self, temperature: float) -> Tensor:
        """``P(gate > 0)`` — the quantity the edge budget is written against."""
        return torch.sigmoid(
            self._bounded() - temperature * math.log(-HC_GAMMA / HC_ZETA)
        )


class NodeCell(nn.Module):
    """Shared local update: one GRU cell reused by every node of a patient."""

    def __init__(self, hidden: int):
        super().__init__()
        self.cell = nn.GRUCell(1 + hidden, hidden)
        self.hidden = hidden

    def forward(self, h: Tensor, injection: Tensor, message: Tensor) -> Tensor:
        b, n, d = h.shape
        inputs = torch.cat([injection.unsqueeze(-1), message], dim=-1)
        return self.cell(inputs.reshape(b * n, -1), h.reshape(b * n, d)).view(b, n, d)


class GraphRecurrence(nn.Module):
    """Directed gated adjacency plus degree-normalised message passing."""

    def __init__(self, n_nodes: int, hidden: int, learnable: bool,
                 fixed_mask: np.ndarray | None, dense: bool, seed: int,
                 gate_init_log_alpha: float = 2.0):
        super().__init__()
        generator = torch.Generator().manual_seed(int(seed))
        self.n_nodes = n_nodes
        self.learnable = learnable
        self.weight = nn.Parameter(
            0.1 * torch.randn(n_nodes, n_nodes, generator=generator)
        )
        self.register_buffer("no_self", 1.0 - torch.eye(n_nodes))
        if learnable:
            self.gate = HardConcreteGate((n_nodes, n_nodes), gate_init_log_alpha)
            self.register_buffer("mask", torch.ones(n_nodes, n_nodes))
        else:
            self.gate = None
            base = torch.ones(n_nodes, n_nodes) if dense else torch.from_numpy(
                np.asarray(fixed_mask, np.float32)
            )
            self.register_buffer("mask", base.float())
        self.register_buffer("frozen_mask", torch.ones(n_nodes, n_nodes))
        self.topology_frozen = False

    def adjacency(self, temperature: float) -> Tensor:
        weights = torch.tanh(self.weight) * self.no_self
        if self.learnable:
            if self.topology_frozen:
                # Freezing means the retained edges are simply on and the rest
                # are gone.  Keeping the gate factor here would multiply the
                # survivors by their own near-zero opening probability and hand
                # back an empty graph, which is what "fine-tune the retained
                # weights" is meant to prevent.
                return weights * self.frozen_mask
            return weights * self.gate(temperature) * self.mask
        return weights * self.mask

    def message(self, h: Tensor, adjacency: Tensor) -> Tensor:
        # msg_i = sum_j A[j, i] phi(h_j) / (sum_j |A[j, i]| + eps)
        phi = torch.tanh(h)
        numerator = torch.einsum("bjd,ji->bid", phi, adjacency)
        denominator = adjacency.abs().sum(dim=0).clamp_min(1e-6)
        return numerator / denominator.view(1, -1, 1)

    def freeze_topology(self, temperature: float, edge_budget: float) -> int:
        """Retain the budgeted number of edges, ranked by opening probability.

        Thresholding at P > 0.5 looks natural and is wrong here.  The budget
        constrains the SUM of opening probabilities, which many barely-open edges
        satisfy just as well as a few reliably-open ones -- and the wiring term
        prefers the former, because each weak edge carries less magnitude.  Under
        a 0.5 threshold that entire graph then freezes to nothing.  Taking the
        top-k makes the frozen degree equal the budget by construction and leaves
        the wiring economy to decide which edges rank highest.
        """
        if not self.learnable:
            return int(self.mask.sum().item())
        with torch.no_grad():
            score = self.gate.open_probability(temperature) * self.no_self
            k = max(1, int(round(float(edge_budget) * self.n_nodes)))
            flat = score.flatten()
            keep_idx = torch.topk(flat, min(k, flat.numel())).indices
            keep = torch.zeros_like(flat)
            keep[keep_idx] = 1.0
            self.frozen_mask.copy_(keep.view_as(score) * self.no_self)
        self.topology_frozen = True
        return int(self.frozen_mask.sum().item())

    def edge_statistics(self, temperature: float) -> dict[str, float]:
        with torch.no_grad():
            adjacency = self.adjacency(temperature)
            live = (adjacency.abs() > 0).float()
            soft = (
                float((self.gate.open_probability(temperature) * self.no_self).sum().item()
                      / self.n_nodes)
                if self.learnable else float("nan")
            )
            return {
                "n_edges": float(live.sum().item()),
                "mean_degree": float(live.sum().item() / self.n_nodes),
                # the quantity the budget term actually controls; it can differ
                # from mean_degree, and only logging the latter hides the drift
                "budget_degree": soft,
                "mean_abs_weight": float(
                    (adjacency.abs().sum() / live.sum().clamp_min(1)).item()
                ),
            }


class SLPModel(nn.Module):
    """One patient, one arm.  All arms expose the same next-set / STOP interface,
    but the mapping onto that interface differs by arm; see the module docstring.

    Every parameter here is fitted per patient -- the shared NodeCell is shared
    across the nodes OF ONE PATIENT, not across patients -- so patients differ in
    the cell, the bias, the emission, the scale and the STOP head as well as in
    the graph.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        torch.manual_seed(config.seed)
        c = config.n_contacts
        self.contact_bias = nn.Parameter(torch.zeros(c))
        self.use_contact_bias = bool(config.use_contact_bias)

        arm = config.arm
        if arm == "STATIC_CONTACT":
            self.stop_head = nn.Linear(2, 1)
        elif arm == "ORDINARY_GRU":
            self.gru = nn.GRUCell(2 * c, config.ordinary_hidden)
            self.readout = nn.Linear(config.ordinary_hidden, c)
            self.stop_head = nn.Linear(config.ordinary_hidden + 2, 1)
        else:
            n_nodes = c if arm == "CONTACT_GRAPH_RNN" else config.n_nodes
            self.n_nodes = n_nodes
            self.cell = NodeCell(config.hidden)
            self.graph = GraphRecurrence(
                n_nodes=n_nodes,
                hidden=config.hidden,
                learnable=arm in ("CONTACT_GRAPH_RNN", "LATENT_LEARNED_SPATIAL_RNN"),
                fixed_mask=config.fixed_edge_mask,
                dense=arm == "LATENT_DENSE_RNN",
                seed=config.seed,
                gate_init_log_alpha=config.gate_init_log_alpha,
            )
            self.stop_head = nn.Linear(config.hidden + 2, 1)
            if arm == "CONTACT_GRAPH_RNN":
                self.node_readout = nn.Linear(config.hidden, 1)
            else:
                self.emission = nn.Linear(config.hidden, 1)
                self.logit_scale = nn.Parameter(torch.tensor(float(config.logit_scale_init)))
                self.register_buffer(
                    "H", torch.from_numpy(np.asarray(config.observation_operator, np.float32))
                )
        if config.normalised_distance is not None:
            self.register_buffer(
                "edge_distance",
                torch.from_numpy(np.asarray(config.normalised_distance, np.float32)),
            )
        else:
            self.edge_distance = None

    # -- recurrence ------------------------------------------------------
    def initial_state(self, batch: int, device: torch.device) -> Tensor | None:
        arm = self.config.arm
        if arm == "STATIC_CONTACT":
            return None
        if arm == "ORDINARY_GRU":
            return torch.zeros(batch, self.config.ordinary_hidden, device=device)
        return torch.zeros(batch, self.n_nodes, self.config.hidden, device=device)

    def step(self, state, x_t: Tensor, recruited: Tensor, t_norm: Tensor,
             temperature: float):
        """One observed rank transition.  Returns (state, contact_logits, stop_logit)."""
        arm = self.config.arm
        bias = self.contact_bias if self.use_contact_bias else torch.zeros_like(self.contact_bias)
        frac = recruited.mean(dim=1, keepdim=True)

        if arm == "STATIC_CONTACT":
            logits = bias.unsqueeze(0).expand(x_t.shape[0], -1)
            stop = self.stop_head(torch.cat([t_norm, frac], dim=1)).squeeze(-1)
            return state, logits, stop

        if arm == "ORDINARY_GRU":
            state = self.gru(torch.cat([x_t, recruited], dim=1), state)
            logits = self.readout(state) + bias
            stop = self.stop_head(torch.cat([state, t_norm, frac], dim=1)).squeeze(-1)
            return state, logits, stop

        adjacency = self.graph.adjacency(temperature)
        if arm == "CONTACT_GRAPH_RNN":
            injection = x_t
        else:
            injection = torch.einsum("bc,cm->bm", x_t, self.H)

        for k in range(self.config.microsteps):
            message = self.graph.message(state, adjacency)
            drive = injection if k == 0 else torch.zeros_like(injection)
            state = self.cell(state, drive, message)

        if arm == "CONTACT_GRAPH_RNN":
            logits = self.node_readout(state).squeeze(-1) + bias
        else:
            emission = F.softplus(self.emission(state).squeeze(-1))
            logits = bias + self.logit_scale * torch.einsum("bm,cm->bc", emission, self.H)
        pooled = state.mean(dim=1)
        stop = self.stop_head(torch.cat([pooled, t_norm, frac], dim=1)).squeeze(-1)
        return state, logits, stop

    def forward(self, x: Tensor, recruited: Tensor, valid: Tensor,
                temperature: float = 1.0) -> tuple[Tensor, Tensor]:
        """Teacher-forced pass. x/recruited: (B, T, C); valid: (B, T)."""
        batch, steps, _ = x.shape
        state = self.initial_state(batch, x.device)
        contact_logits, stop_logits = [], []
        for t in range(steps):
            t_norm = torch.full((batch, 1), t / max(steps - 1, 1), device=x.device)
            state, logits, stop = self.step(
                state, x[:, t], recruited[:, t], t_norm, temperature
            )
            contact_logits.append(logits)
            stop_logits.append(stop)
        return torch.stack(contact_logits, 1), torch.stack(stop_logits, 1)

    # -- regularisers ----------------------------------------------------
    def wiring_loss(self, temperature: float) -> Tensor:
        if self.config.arm not in ("CONTACT_GRAPH_RNN", "LATENT_LEARNED_SPATIAL_RNN"):
            return torch.zeros((), device=self.contact_bias.device)
        gate = self.graph.gate.open_probability(temperature) * self.graph.no_self
        magnitude = torch.tanh(self.graph.weight).abs()
        return (gate * magnitude * self.edge_distance).sum() / self.graph.n_nodes

    def edge_budget_loss(self, temperature: float) -> Tensor:
        if self.config.arm not in ("CONTACT_GRAPH_RNN", "LATENT_LEARNED_SPATIAL_RNN"):
            return torch.zeros((), device=self.contact_bias.device)
        gate = self.graph.gate.open_probability(temperature) * self.graph.no_self
        degree = gate.sum() / self.graph.n_nodes
        # Quadratic near the target, linear far from it.  A plain square starts
        # at (59 - 6)^2 on a dense graph, which swamps the task loss by three
        # orders of magnitude and drives every gate straight through the floor.
        return F.smooth_l1_loss(degree, torch.full_like(degree, self.config.edge_budget))


def next_set_stop_loss(
    contact_logits: Tensor,
    stop_logits: Tensor,
    target: Tensor,
    available: Tensor,
    valid: Tensor,
    is_last: Tensor,
    stop_weight: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Multi-label next-rank BCE plus STOP BCE.

    A contact recruited earlier in the event cannot be recruited again, so it is
    removed from both the likelihood and the normalisation -- that support mask
    is a deterministic property of the event, not something the model predicts.
    """
    # A contact excluded from `available` is outside this model's scope -- it is
    # already recruited, or withheld by leave-contact-out.  It must leave both
    # the prediction and the target, otherwise its -inf logit is scored against a
    # positive label and the loss diverges.
    target = target * available.float()
    predict = valid & ~is_last
    masked = contact_logits.masked_fill(~available, NEG_INF)
    per_contact = F.binary_cross_entropy_with_logits(
        masked, target, reduction="none"
    ) * available.float()
    denom = available.float().sum(-1).clamp_min(1.0)
    per_step = per_contact.sum(-1) / denom
    n_predict = predict.float().sum().clamp_min(1.0)
    next_loss = (per_step * predict.float()).sum() / n_predict

    stop_loss = (
        F.binary_cross_entropy_with_logits(stop_logits, is_last.float(), reduction="none")
        * valid.float()
    ).sum() / valid.float().sum().clamp_min(1.0)
    return next_loss + stop_weight * stop_loss, next_loss.detach(), stop_loss.detach()


def cardinality_conditioned_nll(
    contact_logits: Tensor,
    target: Tensor,
    available: Tensor,
    predict: Tensor,
) -> Tensor:
    """Mean per-contact NLL of the true next set under the observed cardinality.

    Reported alongside the primary BCE so this run is comparable with the
    existing Topic 5 contact-NLL line.
    """
    target = target * available.float()
    masked = contact_logits.masked_fill(~available, NEG_INF)
    log_prob = torch.log_softmax(masked, dim=-1)
    chosen = (log_prob * target).sum(-1)
    count = target.sum(-1).clamp_min(1.0)
    per_step = -chosen / count
    return (per_step * predict.float()).sum() / predict.float().sum().clamp_min(1.0)
