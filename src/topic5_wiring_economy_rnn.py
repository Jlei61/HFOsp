"""WE-SLP-RNN v0.3: a masked recurrent network on a patient's tissue plane.

State lives on tissue units placed in the patient's own 2D propagation plane.
Contacts are read-out ports only: input enters through the transpose of the same
local Gaussian operator the output leaves through, so nothing travels from one
contact to another except through the recurrent mask.

The mask is a fixed sparse resource.  It never grows or shrinks; each epoch the
weakest active edges are deleted and the same number of inactive edges are grown
back, preferentially between units that sit close together on the plane.  The
task decides which of the proposed edges survive.

Primary cell is a plain leaky RNN, on purpose: it has one recurrent matrix, so
``mask * W`` *is* the graph and every topology statement points at one object.
The gated cell is kept as a confirmatory arm and has to synthesise an edge
strength out of three matrices.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

CELLS = ("rnn", "gru")
ARMS = ("STATIC_CONTACT", "DENSE_TISSUE", "RANDOM_SET", "SPATIAL_SET")
SPARSE_ARMS = ("RANDOM_SET", "SPATIAL_SET")
NEG_INF = -1e9
GROW_EPS_MM = 0.1


@dataclass
class WEConfig:
    arm: str
    cell: str = "rnn"
    n_contacts: int = 0
    n_nodes: int = 0
    state_dim: int = 1
    density: float = 0.10
    eta: float = 0.03
    d0_mm: float = 10.0
    stop_hidden: int = 16
    seed: int = 0
    observation_operator: np.ndarray | None = field(default=None, repr=False)
    node_distance_mm: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; use one of {ARMS}")
        if self.cell not in CELLS:
            raise ValueError(f"unknown cell {self.cell!r}; use one of {CELLS}")
        if self.arm != "STATIC_CONTACT":
            if self.observation_operator is None:
                raise ValueError(f"{self.arm} needs an observation operator")
            if self.node_distance_mm is None:
                raise ValueError(f"{self.arm} needs node distances in mm")


def active_edge_count(n_nodes: int, density: float) -> int:
    """How many directed edges the fixed sparse resource pays for."""
    return int(round(float(density) * n_nodes * (n_nodes - 1)))


def initial_mask(n_nodes: int, density: float, distance_mm: np.ndarray | None,
                 spatial: bool, seed: int) -> np.ndarray:
    """Sample the starting edge set, distance-biased or uniform.

    Both arms pay for the same number of edges; only the proposal distribution
    differs.  That is the whole of the manipulation, so it is the only thing
    allowed to differ here.
    """
    rng = np.random.default_rng(seed)
    off = ~np.eye(n_nodes, dtype=bool)
    candidates = np.flatnonzero(off.reshape(-1))
    k = min(active_edge_count(n_nodes, density), candidates.size)
    if spatial:
        weights = 1.0 / (np.asarray(distance_mm, float).reshape(-1)[candidates] + GROW_EPS_MM)
        probability = weights / weights.sum()
    else:
        probability = None
    chosen = rng.choice(candidates, size=k, replace=False, p=probability)
    mask = np.zeros(n_nodes * n_nodes, np.float32)
    mask[chosen] = 1.0
    return mask.reshape(n_nodes, n_nodes)


class WEModel(nn.Module):
    def __init__(self, config: WEConfig):
        super().__init__()
        self.config = config
        self.arm = config.arm
        self.cell = config.cell
        self.n_contacts = int(config.n_contacts)
        self.n_nodes = int(config.n_nodes)
        self.state_dim = int(config.state_dim)
        torch.manual_seed(int(config.seed))

        self.contact_bias = nn.Parameter(torch.zeros(self.n_contacts))
        self.stop_head = nn.Sequential(
            nn.Linear(4, int(config.stop_hidden)), nn.Tanh(),
            nn.Linear(int(config.stop_hidden), 1),
        )
        if self.arm == "STATIC_CONTACT":
            return

        n, d = self.n_nodes, self.state_dim
        width = n * d
        self.register_buffer("H", torch.as_tensor(config.observation_operator, dtype=torch.float32))
        self.register_buffer("D_mm", torch.as_tensor(config.node_distance_mm, dtype=torch.float32))

        if self.arm == "DENSE_TISSUE":
            node_mask = 1.0 - np.eye(n, dtype=np.float32)
        else:
            node_mask = initial_mask(
                n, config.density, config.node_distance_mm,
                spatial=(self.arm == "SPATIAL_SET"), seed=int(config.seed),
            )
        self.register_buffer("node_mask", torch.as_tensor(node_mask))
        self.register_buffer("initial_node_mask", torch.as_tensor(node_mask).clone())
        self.mask_frozen = False

        scale = 1.0 / math.sqrt(max(1.0, float(config.density) * n * d))
        gates = 3 if self.cell == "gru" else 1
        self.recurrent = nn.Parameter(scale * torch.randn(gates, width, width))
        self.input_gain = nn.Parameter(torch.ones(gates, n, d))
        self.bias = nn.Parameter(torch.zeros(gates, width))
        self.kappa_logit = nn.Parameter(torch.zeros(1))
        self.readout_gain = nn.Parameter(torch.tensor(4.0))

    # ---- graph ------------------------------------------------------------
    def _expanded_mask(self) -> Tensor:
        """Node mask lifted to state channels: ``M ⊗ 1_{d×d}``.

        The graph constrains which tissue units may talk to each other; it says
        nothing about a unit's internal channels, so the block is filled.
        """
        if self.state_dim == 1:
            return self.node_mask
        return torch.kron(self.node_mask, torch.ones(self.state_dim, self.state_dim,
                                                     device=self.node_mask.device))

    def edge_strength(self) -> Tensor:
        """Per node-pair edge magnitude used for pruning and weighted topology."""
        n, d = self.n_nodes, self.state_dim
        blocks = self.recurrent.detach().reshape(-1, n, d, n, d)
        return blocks.pow(2).sum(dim=(0, 2, 4)).sqrt()

    def masked_recurrent(self) -> Tensor:
        return self.recurrent * self._expanded_mask().unsqueeze(0)

    def wiring_cost(self) -> Tensor:
        """Mean edge length in units of ``d0``, weighted by edge strength."""
        n, d = self.n_nodes, self.state_dim
        blocks = self.recurrent.reshape(-1, n, d, n, d)
        strength = blocks.pow(2).sum(dim=(0, 2, 4)).clamp_min(1e-12).sqrt()
        weighted = self.node_mask * strength * (self.D_mm / float(self.config.d0_mm))
        return weighted.sum() / self.node_mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def rewire(self, zeta: float) -> int:
        """Delete the weakest ``zeta`` of active edges, grow the same number back.

        New edges start at exactly zero weight.  Their gradient is not zero, so
        the task can still pull them up; seeding them with noise would inject
        structure the task did not ask for right at the moment we are asking
        what the task built.
        """
        if self.arm not in SPARSE_ARMS or self.mask_frozen or zeta <= 0.0:
            return 0
        n = self.n_nodes
        mask = self.node_mask
        active = torch.nonzero(mask.reshape(-1) > 0).flatten()
        n_drop = int(round(float(zeta) * active.numel()))
        if n_drop < 1:
            return 0
        strength = self.edge_strength().reshape(-1)[active]
        drop = active[torch.argsort(strength)[:n_drop]]

        flat = mask.reshape(-1).clone()
        flat[drop] = 0.0
        off_diagonal = (1.0 - torch.eye(n, device=mask.device)).reshape(-1)
        inactive = torch.nonzero((flat == 0) & (off_diagonal > 0)).flatten()
        if inactive.numel() < n_drop:
            return 0
        if self.arm == "SPATIAL_SET":
            weights = 1.0 / (self.D_mm.reshape(-1)[inactive] + GROW_EPS_MM)
            probability = (weights / weights.sum()).double()
        else:
            probability = torch.full((inactive.numel(),), 1.0 / inactive.numel(),
                                     dtype=torch.float64, device=mask.device)
        grow = inactive[torch.multinomial(probability, n_drop, replacement=False)]
        flat[grow] = 1.0
        self.node_mask.copy_(flat.reshape(n, n))

        # Reset the grown edges and clear whatever the dropped ones carried, so
        # a resurrected pair does not inherit a weight from an earlier life.
        d = self.state_dim
        block = torch.zeros(n * n, dtype=torch.bool, device=mask.device)
        block[drop] = True
        block[grow] = True
        touched = block.reshape(n, n)
        if d > 1:
            touched = torch.kron(touched.float(), torch.ones(d, d, device=mask.device)).bool()
        self.recurrent[:, touched] = 0.0
        return n_drop

    def freeze_mask(self) -> None:
        self.mask_frozen = True

    def graph_snapshot(self) -> dict:
        if self.arm == "STATIC_CONTACT":
            return {}
        return {
            "mask": self.node_mask.detach().cpu().numpy().astype(np.uint8),
            "initial_mask": self.initial_node_mask.detach().cpu().numpy().astype(np.uint8),
            "strength": self.edge_strength().cpu().numpy().astype(np.float32),
            "D_mm": self.D_mm.detach().cpu().numpy().astype(np.float32),
        }

    # ---- dynamics ---------------------------------------------------------
    def _inject(self, x_t: Tensor) -> Tensor:
        """``u_t = Hᵀ x_t`` lifted to state channels by a per-position gain."""
        u = x_t @ self.H                                     # (B, M)
        return u.unsqueeze(1).unsqueeze(-1) * self.input_gain.unsqueeze(0)

    def _step(self, h: Tensor, x_t: Tensor) -> Tensor:
        b = h.shape[0]
        n, d = self.n_nodes, self.state_dim
        u = self._inject(x_t).reshape(b, -1, n * d)          # (B, G, M*d)
        w = self.masked_recurrent()
        if self.cell == "rnn":
            pre = u[:, 0] + h @ w[0].T + self.bias[0]
            kappa = torch.sigmoid(self.kappa_logit)
            return (1.0 - kappa) * h + kappa * torch.tanh(pre)
        r = torch.sigmoid(u[:, 0] + h @ w[0].T + self.bias[0])
        z = torch.sigmoid(u[:, 1] + h @ w[1].T + self.bias[1])
        cand = torch.tanh(u[:, 2] + (r * h) @ w[2].T + self.bias[2])
        return (1.0 - z) * h + z * cand

    def _readout(self, h: Tensor) -> Tensor:
        unit = h.reshape(h.shape[0], self.n_nodes, self.state_dim).mean(-1)
        return self.contact_bias + self.readout_gain * (unit @ self.H.T)

    def _stop(self, h: Tensor | None, t_norm: Tensor, recruited_fraction: Tensor) -> Tensor:
        if h is None:
            zeros = torch.zeros_like(t_norm)
            features = torch.stack([zeros, zeros, t_norm, recruited_fraction], dim=-1)
        else:
            unit = h.reshape(h.shape[0], self.n_nodes, self.state_dim).mean(-1)
            features = torch.stack([unit.mean(-1), unit.max(-1).values,
                                    t_norm, recruited_fraction], dim=-1)
        return self.stop_head(features).squeeze(-1)

    def forward(self, x: Tensor, recruited: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        b, steps, _ = x.shape
        device = x.device
        h = None
        if self.arm != "STATIC_CONTACT":
            h = torch.zeros(b, self.n_nodes * self.state_dim, device=device)
        logits, stops = [], []
        denom = max(1, steps - 1)
        for t in range(steps):
            if h is not None:
                h = self._step(h, x[:, t])
                logits.append(self._readout(h))
            else:
                logits.append(self.contact_bias.expand(b, -1))
            t_norm = torch.full((b,), t / denom, device=device)
            stops.append(self._stop(h, t_norm, recruited[:, t].mean(-1)))
        return torch.stack(logits, 1), torch.stack(stops, 1)


def next_rank_stop_loss(
    contact_logits: Tensor,
    stop_logits: Tensor,
    target: Tensor,
    available: Tensor,
    valid: Tensor,
    is_last: Tensor,
    stop_weight: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Multi-label next-rank BCE plus STOP BCE.

    Identical in form to the SLP-RNN v0.1 objective so the two versions'
    numbers stay on the same scale.  A contact recruited earlier cannot be
    recruited again, so it leaves both the likelihood and the normalisation.
    """
    target = target * available.float()
    predict = valid & ~is_last
    masked = contact_logits.masked_fill(~available, NEG_INF)
    per_contact = F.binary_cross_entropy_with_logits(masked, target, reduction="none")
    per_contact = per_contact * available.float()
    per_step = per_contact.sum(-1) / available.float().sum(-1).clamp_min(1.0)
    next_loss = (per_step * predict.float()).sum() / predict.float().sum().clamp_min(1.0)
    stop_loss = (
        F.binary_cross_entropy_with_logits(stop_logits, is_last.float(), reduction="none")
        * valid.float()
    ).sum() / valid.float().sum().clamp_min(1.0)
    return next_loss + stop_weight * stop_loss, next_loss.detach(), stop_loss.detach()


def cardinality_conditioned_nll(contact_logits: Tensor, target: Tensor,
                                available: Tensor, predict: Tensor) -> Tensor:
    """Mean per-contact NLL of the true next set under the observed cardinality."""
    target = target * available.float()
    masked = contact_logits.masked_fill(~available, NEG_INF)
    log_prob = torch.log_softmax(masked, dim=-1)
    chosen = (log_prob * target).sum(-1)
    per_step = -chosen / target.sum(-1).clamp_min(1.0)
    return (per_step * predict.float()).sum() / predict.float().sum().clamp_min(1.0)


def build_event_tensors(ranks: np.ndarray) -> dict[str, Tensor]:
    """Turn dense per-event contact ranks into padded teacher-forcing tensors."""
    ranks = np.asarray(ranks)
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
            x[e, t, row == t] = 1.0
            recruited[e, t] = (row >= 0) & (row <= t)
            if t + 1 < length:
                target[e, t, row == t + 1] = 1.0
        valid[e, :length] = True
        if length:
            is_last[e, length - 1] = True
    available = (recruited == 0) & valid[:, :, None]
    return {
        "x": torch.from_numpy(x),
        "recruited": torch.from_numpy(recruited),
        "available": torch.from_numpy(available),
        "target": torch.from_numpy(target),
        "valid": torch.from_numpy(valid),
        "is_last": torch.from_numpy(is_last),
    }


def zeta_schedule(epoch: int, warmup: int, rewire_epochs: int, zeta0: float) -> float:
    """Cosine anneal the rewiring rate to exactly zero before the mask freezes."""
    if epoch < warmup or epoch >= warmup + rewire_epochs:
        return 0.0
    progress = (epoch - warmup) / max(1, rewire_epochs)
    return float(zeta0 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def rollout(model: WEModel, starts: Sequence[np.ndarray], n_contacts: int,
            max_steps: int, device: torch.device) -> list[list[list[int]]]:
    """Same-start free generation, one rank set per step, argmax not threshold.

    A fixed 0.5 cut against calibrated probabilities near 0.07 makes every event
    stop after one step; that is the threshold's behaviour being reported as the
    model's.  One contact per step by argmax, and STOP decided by its own head.
    """
    model.eval()
    out: list[list[list[int]]] = []
    for start in starts:
        h = (torch.zeros(1, model.n_nodes * model.state_dim, device=device)
             if model.arm != "STATIC_CONTACT" else None)
        recruited = torch.zeros(1, n_contacts, device=device)
        x = torch.zeros(1, n_contacts, device=device)
        x[0, list(start)] = 1.0
        recruited[0, list(start)] = 1.0
        sequence = [list(map(int, start))]
        denom = max(1, max_steps - 1)
        for t in range(max_steps):
            if h is not None:
                h = model._step(h, x)
                logits = model._readout(h)
            else:
                logits = model.contact_bias.expand(1, -1)
            t_norm = torch.full((1,), t / denom, device=device)
            if torch.sigmoid(model._stop(h, t_norm, recruited.mean(-1))).item() > 0.5:
                break
            logits = logits.masked_fill(recruited > 0, NEG_INF)
            if bool((recruited > 0).all()):
                break
            pick = int(logits.argmax(-1).item())
            sequence.append([pick])
            x = torch.zeros(1, n_contacts, device=device)
            x[0, pick] = 1.0
            recruited[0, pick] = 1.0
        out.append(sequence)
    return out
