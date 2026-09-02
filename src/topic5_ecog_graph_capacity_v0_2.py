"""Topic 5.2D v0.2 — few-parameter graph recurrence for the high-density ECoG grids.

This is the construct-validity case series, not a replication of the SEEG cohort.
Its state is the full contact field rather than an ``r``-dimensional coordinate,
and the capacity being varied is the number of free coefficients in a polynomial
of the *frozen* grid adjacency:

    G1 = a0 I + a1 A
    G2 = a0 I + a1 A + a2 A^2
    G3 = a0 I + sum_b a_b A_b     (one-/two-hop x frozen grid direction classes)

``FREE_SAME_STATE_UPPER_BOUND`` replaces the polynomial with an unconstrained
``C x C`` transition.  It is declared here as a capacity ceiling for the ECoG
case series only — the SEEG design forbids a free contact transition because
there it would be a bypass, whereas here it is the reference the graph
recurrence is measured against.

Because contact identity, the inputs and the outputs stay fixed while only the
graph changes, the ECoG swap may be called ``RUNTIME_GRAPH_SWAP``.  The SEEG
basis swap may not; the two are never pooled.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

CAPACITIES = ("G1", "G2", "G3", "FREE_SAME_STATE_UPPER_BOUND")
GRAPH_FAMILIES = {
    "OBSERVED_GRID": "TRUE_GRID",
    "IDENTITY_PERMUTED_GRID": "WRONG_GRID",
    "DEGREE_AND_DISTANCE_REWIRED_GRID": "DEGREE_RANDOM",
}
# Eight pre-frozen null graphs out of the 31 available, chosen by index before
# any result exists; the full 31 stay available for low-cost scoring sensitivity.
NULL_GRAPH_INDICES: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 28)
PRIMARY_MICROSTEPS = 2
SENSITIVITY_MICROSTEPS = 1
READOUT_ORDER = 3


def symmetric_normalise(matrix: np.ndarray) -> np.ndarray:
    degree = matrix.sum(axis=1)
    scale = np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1e-9)), 0.0)
    return matrix * scale[:, None] * scale[None, :]


def load_graph(root: Path, subject: str, family: str, index: int | None) -> tuple[np.ndarray, np.ndarray, dict]:
    directory = Path(root) / subject / "four_neighbour"
    name = family if index is None else f"{family}_{index:02d}"
    payload = np.load(directory / f"{name}.npz", allow_pickle=False)
    audit_path = directory / f"{name}.audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    mask = np.asarray(payload["mask"], dtype=bool)
    np.fill_diagonal(mask, False)
    return mask, np.asarray(payload["coordinates"], dtype=np.float64), audit


def graph_blocks(mask: np.ndarray, coords: np.ndarray, capacity: str) -> list[np.ndarray]:
    """Frozen operators whose weighted sum is ``G - a0 I``."""
    adjacency = np.asarray(mask, dtype=float)
    normalised = symmetric_normalise(adjacency)
    if capacity == "G1":
        return [normalised]
    if capacity in ("G2", "FREE_SAME_STATE_UPPER_BOUND"):
        return [normalised, normalised @ normalised]
    if capacity == "G3":
        delta = coords[None, :, :] - coords[:, None, :]
        horizontal = np.abs(delta[:, :, 0]) >= np.abs(delta[:, :, 1])
        two_hop = ((adjacency @ adjacency) > 0) & ~mask
        np.fill_diagonal(two_hop, False)
        return [
            symmetric_normalise((mask & horizontal).astype(float)),
            symmetric_normalise((mask & ~horizontal).astype(float)),
            symmetric_normalise((two_hop & horizontal).astype(float)),
            symmetric_normalise((two_hop & ~horizontal).astype(float)),
        ]
    raise ValueError(f"unknown capacity {capacity!r}")


def readout_blocks(mask: np.ndarray) -> list[np.ndarray]:
    """``[I, A, A^2]`` — the same for every structure, so only the graph differs."""
    normalised = symmetric_normalise(np.asarray(mask, dtype=float))
    return [np.eye(mask.shape[0]), normalised, normalised @ normalised]


@dataclass
class EcogConfig:
    structure: str
    family: str
    capacity: str
    n_contacts: int
    n_horizons: int
    max_cardinality: int
    microsteps: int = PRIMARY_MICROSTEPS


class EcogGraphMotif(nn.Module):
    """Graph recurrence over the observed grid, deliberately parameter-poor.

    Exposes the same ``forward(batch, ordered_path=...)`` contract as the SEEG
    operator so the shared evaluator, the exact subset law and the two unordered
    baselines are literally the same code on both datasets.
    """

    def __init__(self, config: EcogConfig, transition_blocks: list[np.ndarray],
                 output_blocks: list[np.ndarray]) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("transition_blocks",
                             torch.as_tensor(np.stack(transition_blocks), dtype=torch.float32))
        self.register_buffer("output_blocks",
                             torch.as_tensor(np.stack(output_blocks), dtype=torch.float32))
        self.alpha_identity = nn.Parameter(torch.ones(1))
        self.free = config.capacity == "FREE_SAME_STATE_UPPER_BOUND"
        if self.free:
            self.free_transition = nn.Parameter(
                torch.zeros(config.n_contacts, config.n_contacts))
        else:
            self.alpha = nn.Parameter(torch.zeros(len(transition_blocks)))
        shape = (config.n_horizons, READOUT_ORDER) \
            if config.family != "AUTONOMOUS_SHARED_OPERATOR" else (READOUT_ORDER,)
        self.gamma = nn.Parameter(torch.zeros(*shape))
        card_shape = (config.n_horizons, config.max_cardinality) \
            if config.family != "AUTONOMOUS_SHARED_OPERATOR" else (config.max_cardinality,)
        self.card_scale = nn.Parameter(torch.zeros(*card_shape))
        if config.family != "AUTONOMOUS_SHARED_OPERATOR":
            self.delta = nn.Parameter(torch.zeros(READOUT_ORDER))

    def transition(self) -> torch.Tensor:
        identity = torch.eye(self.config.n_contacts, device=self.alpha_identity.device)
        if self.free:
            return self.alpha_identity * identity + self.free_transition
        return self.alpha_identity * identity + torch.einsum(
            "b,bij->ij", self.alpha, self.transition_blocks)

    def _advance(self, state: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
        for _ in range(self.config.microsteps):
            state = state @ operator.T
        return state

    def _decode(self, state: torch.Tensor, horizon: int | None) -> torch.Tensor:
        weights = self.gamma if horizon is None else self.gamma[horizon]
        blocks = torch.einsum("b,bij->ij", weights, self.output_blocks)
        return state @ blocks.T

    def prefix_state(self, batch) -> torch.Tensor:
        operator = self.transition()
        state = batch.prefix_sets[:, 0]
        for step in range(1, batch.prefix_len):
            state = self._advance(state, operator) + batch.prefix_sets[:, step]
        return state

    def forward(self, batch, ordered_path: bool = True) -> dict[str, torch.Tensor]:
        state = self.prefix_state(batch)
        if not ordered_path:
            state = torch.zeros_like(state)
        operator = self.transition()
        contact, cardinality = [], []
        if self.config.family == "AUTONOMOUS_SHARED_OPERATOR":
            rolled = state
            for _ in range(self.config.n_horizons):
                rolled = self._advance(rolled, operator)
                contact.append(self._decode(rolled, None))
                cardinality.append(rolled.mean(dim=1, keepdim=True) * self.card_scale.unsqueeze(0))
            suffix = None
        else:
            for horizon in range(self.config.n_horizons):
                contact.append(self._decode(state, horizon))
                cardinality.append(
                    state.mean(dim=1, keepdim=True) * self.card_scale[horizon].unsqueeze(0))
            suffix = state @ torch.einsum("b,bij->ij", self.delta, self.output_blocks).T
        return {"contact": torch.stack(contact, dim=1),
                "cardinality": torch.stack(cardinality, dim=1),
                "suffix": suffix, "state": state}


def swap_graph(model: EcogGraphMotif, transition_blocks: list[np.ndarray],
               output_blocks: list[np.ndarray]) -> None:
    """Replace the frozen grid in place — no parameter update, no recalibration."""
    with torch.no_grad():
        model.transition_blocks.copy_(
            torch.as_tensor(np.stack(transition_blocks), dtype=torch.float32))
        model.output_blocks.copy_(
            torch.as_tensor(np.stack(output_blocks), dtype=torch.float32))
