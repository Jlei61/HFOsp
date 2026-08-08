"""Frozen v0.4 contracts for the Topic 5 recurrent-motif benchmark.

This module contains only objects shared by every model family: the factorial
matrix, rank-set shuffles, the validation-calibrated rollout size decoder and
the no-future-information free rollout.  Keeping them here prevents individual
arms from quietly using different generation rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.topic5_wiring_economy_rnn import NEG_INF, WEModel


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    arm: str
    eta: float
    seeds: tuple[int, ...] = (0, 1, 2)


MODEL_SPECS: dict[str, ModelSpec] = {
    "M0_NO_REC": ModelSpec("M0_NO_REC", "STATIC_CONTACT", 0.0),
    "M1_DENSE": ModelSpec("M1_DENSE", "DENSE_TISSUE", 0.0),
    "M2_UNIFORM_SET": ModelSpec("M2_UNIFORM_SET", "RANDOM_SET", 0.0),
    "M3_FIXED_LOCAL": ModelSpec("M3_FIXED_LOCAL", "FIXED_LOCAL", 0.0),
    "M4_SPATIAL_GROWTH": ModelSpec("M4_SPATIAL_GROWTH", "SPATIAL_SET_NOCOST", 0.0),
    "M5_SPATIAL_LOW": ModelSpec("M5_SPATIAL_LOW", "SPATIAL_SET", 0.01),
    "M6_SPATIAL_MID": ModelSpec("M6_SPATIAL_MID", "SPATIAL_SET", 0.03),
    "M7_SPATIAL_HIGH": ModelSpec("M7_SPATIAL_HIGH", "SPATIAL_SET", 0.10),
    "M8_UNIFORM_COST_MID": ModelSpec("M8_UNIFORM_COST_MID", "RANDOM_SET_COST", 0.03),
    "C_ORDER_SHUFFLED": ModelSpec("C_ORDER_SHUFFLED", "SPATIAL_SET", 0.03),
    "C_FULL_RANK_SHUFFLED": ModelSpec(
        "C_FULL_RANK_SHUFFLED", "SPATIAL_SET", 0.03, seeds=(0,)
    ),
}

CORE_IDS = (
    "M0_NO_REC", "M1_DENSE", "M2_UNIFORM_SET", "M3_FIXED_LOCAL",
    "M4_SPATIAL_GROWTH", "M6_SPATIAL_MID", "M8_UNIFORM_COST_MID",
    "C_ORDER_SHUFFLED",
)
DOSE_IDS = ("M5_SPATIAL_LOW", "M7_SPATIAL_HIGH", "C_FULL_RANK_SHUFFLED")
GRU_IDS = ("M0_NO_REC", "M1_DENSE", "M2_UNIFORM_SET", "M3_FIXED_LOCAL", "M6_SPATIAL_MID")


def shuffle_rank_sets(ranks: np.ndarray, seed: int, keep_first: bool = True) -> np.ndarray:
    """Shuffle whole rank sets within each event, optionally preserving rank 1.

    Contacts tied in one observed rank remain tied.  Thus the control changes
    only temporal order, not participation, event length or within-rank sets.
    """
    rng = np.random.default_rng(seed)
    out = np.asarray(ranks).copy()
    for event, row in enumerate(np.asarray(ranks)):
        labels = np.unique(row[row >= 0])
        movable = labels[labels > 0] if keep_first else labels
        if movable.size < 2:
            continue
        permuted = rng.permutation(movable)
        mapping = {int(old): int(new) for old, new in zip(movable, permuted)}
        for old, new in mapping.items():
            out[event, row == old] = new
    return out


class RolloutSizeHead(nn.Module):
    """Shared low-capacity next-rank cardinality decoder (4 -> 16 -> C)."""

    def __init__(self, n_contacts: int):
        super().__init__()
        self.n_contacts = int(n_contacts)
        self.network = nn.Sequential(
            nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, self.n_contacts)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def state_features(
    model: WEModel,
    h: torch.Tensor | None,
    t: int,
    recruited_fraction: torch.Tensor,
) -> torch.Tensor:
    """Four causal features used by STOP and the common size head."""
    t_norm = torch.full_like(recruited_fraction, t / max(1, model.n_contacts - 1))
    if h is None:
        zero = torch.zeros_like(t_norm)
        return torch.stack((zero, zero, t_norm, recruited_fraction), dim=-1)
    unit = h.reshape(h.shape[0], model.n_nodes, model.state_dim).mean(-1)
    return torch.stack((unit.mean(-1), unit.max(-1).values, t_norm, recruited_fraction), dim=-1)


@torch.no_grad()
def teacher_forced_size_examples(
    model: WEModel,
    tensors: dict[str, torch.Tensor],
    event_indices: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect causal features and next-set sizes from selected events."""
    model.eval()
    feature_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    indices = np.asarray(event_indices, int)
    for begin in range(0, len(indices), int(batch_size)):
        selected = torch.as_tensor(indices[begin:begin + int(batch_size)])
        x = tensors["x"][selected].to(device)
        recruited = tensors["recruited"][selected].to(device)
        valid = tensors["valid"][selected].to(device)
        is_last = tensors["is_last"][selected].to(device)
        target = tensors["target"][selected].to(device)
        h = (torch.zeros(len(selected), model.n_nodes * model.state_dim, device=device)
             if model.arm != "STATIC_CONTACT" else None)
        feature_grid: list[torch.Tensor] = []
        for t in range(x.shape[1]):
            if h is not None:
                h = model._step(h, x[:, t])
            feature_grid.append(state_features(model, h, t, recruited[:, t].mean(-1)))
        features = torch.stack(feature_grid, dim=1)
        continuing = valid & ~is_last
        feature_rows.append(features[continuing].cpu())
        target_rows.append((target.sum(-1).long()[continuing] - 1).cpu())
    if not feature_rows:
        return torch.empty(0, 4), torch.empty(0, dtype=torch.long)
    return torch.cat(feature_rows), torch.cat(target_rows)


def fit_rollout_size_head(
    model: WEModel,
    tensors: dict[str, torch.Tensor],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
    seed: int,
    max_epochs: int = 200,
    patience: int = 20,
) -> tuple[RolloutSizeHead, dict[str, Any]]:
    """Fit the decoder after freezing recurrence; validation selects the epoch."""
    train_x, train_y = teacher_forced_size_examples(model, tensors, train_indices, device)
    val_x, val_y = teacher_forced_size_examples(model, tensors, validation_indices, device)
    if train_y.numel() == 0 or val_y.numel() == 0:
        raise RuntimeError("rollout size decoder requires train and validation continue steps")
    torch.manual_seed(int(seed) + 4242)
    head = RolloutSizeHead(model.n_contacts).to(device)
    optimiser = torch.optim.Adam(head.parameters(), lr=1e-2)
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    curve: list[dict[str, float]] = []
    for epoch in range(int(max_epochs)):
        head.train()
        loss = nn.functional.cross_entropy(head(train_x), train_y)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        head.eval()
        with torch.no_grad():
            val = nn.functional.cross_entropy(head(val_x), val_y)
        curve.append({"epoch": epoch, "train_nll": float(loss), "validation_nll": float(val)})
        if float(val) < best - 1e-6:
            best = float(val)
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is None:
        raise RuntimeError("rollout size decoder did not produce a finite checkpoint")
    head.load_state_dict(best_state)
    return head, {
        "n_train_decisions": int(train_y.numel()),
        "n_validation_decisions": int(val_y.numel()),
        "best_validation_nll": best,
        "n_epochs": len(curve),
        "curve": curve,
    }


@torch.no_grad()
def rollout_with_size_head(
    model: WEModel,
    size_head: RolloutSizeHead,
    starts: Sequence[np.ndarray],
    device: torch.device,
) -> list[list[list[int]]]:
    """Generate complete rank sets without reading observed future cardinality."""
    model.eval()
    size_head.eval()
    generated: list[list[list[int]]] = []
    for start in starts:
        start = np.asarray(start, int)
        if start.size == 0:
            raise ValueError("a free rollout needs an observed first-rank seed")
        h = (torch.zeros(1, model.n_nodes * model.state_dim, device=device)
             if model.arm != "STATIC_CONTACT" else None)
        recruited = torch.zeros(1, model.n_contacts, device=device)
        x = torch.zeros_like(recruited)
        x[0, start.tolist()] = 1.0
        recruited[0, start.tolist()] = 1.0
        sequence = [start.tolist()]
        for t in range(model.n_contacts):
            if h is not None:
                h = model._step(h, x)
                logits = model._readout(h)
            else:
                logits = model.contact_bias.expand(1, -1)
            fraction = recruited.mean(-1)
            features = state_features(model, h, t, fraction)
            stop_probability = torch.sigmoid(model._stop(
                h, features[:, 2], fraction
            )).item()
            if stop_probability >= 0.5 or bool((recruited > 0).all()):
                break
            k = int(size_head(features).argmax(-1).item()) + 1
            eligible = np.flatnonzero((recruited[0] == 0).cpu().numpy())
            k = min(k, int(eligible.size))
            score = logits[0].detach().cpu().numpy()
            # Lexicographic order makes exact ties select the lower contact index.
            ordered = np.lexsort((eligible, -score[eligible]))
            picked = eligible[ordered[:k]].astype(int)
            sequence.append(picked.tolist())
            x = torch.zeros_like(recruited)
            x[0, picked.tolist()] = 1.0
            recruited[0, picked.tolist()] = 1.0
        generated.append(sequence)
    return generated


ROLLOUT_DECODER_CONTRACT = {
    "version": "topic5_rnn_motif_v0_4",
    "input_features": ["mean_state", "max_state", "rank_index_over_contact_count", "recruited_fraction"],
    "network": "Linear(4,16)-Tanh-Linear(16,n_contacts)",
    "fit_support": "train teacher-forced continue decisions only",
    "selection_support": "interictal validation only",
    "stop_precedence": "stop_probability>=0.5 before contact selection",
    "cardinality": "argmax(size_head)+1; never observed future set size",
    "repeat_mask": True,
    "tie_break": "lowest contact index",
    "maximum_ranks": "n_contacts",
}
