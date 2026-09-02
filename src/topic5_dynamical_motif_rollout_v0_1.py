"""Stochastic decoder and batched closed-loop rollout for the motif RNN.

The decoder is frozen on the calibration split and shared by every model: STOP
is sampled first, then the next rank-set size, then an exact fixed-cardinality
subset of the contacts that have not been recruited yet.  The observed future
cardinality is never read.  All models consume the same uniform stream so that
paired comparisons are common-random-number comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_dynamical_motif_rnn_v0_1 import MotifRNN, rollout_displacement_update
from src.topic5_shared_propagation_field import (
    conditional_k_subset_log_prob,
    sample_conditional_k_subset,
)

NEG_INF = -1e9


def stable_seed(label: str, salt: int = 0) -> int:
    digest = hashlib.sha256(f"{label}|{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 63 - 1)


class SizeHead(nn.Module):
    """Shared low-capacity next-rank cardinality decoder (4 -> 16 -> C)."""

    def __init__(self, n_contacts: int):
        super().__init__()
        self.n_contacts = int(n_contacts)
        self.network = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, self.n_contacts))

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


@dataclass
class DecoderContract:
    contact_temperature: float
    cardinality_temperature: float
    stop_temperature: float
    n_calibration_decisions: int
    n_calibration_events: int
    contact_nll_before: float
    contact_nll_after: float
    cardinality_nll_before: float
    cardinality_nll_after: float
    stop_bce_before: float
    stop_bce_after: float
    size_head_train_decisions: int
    size_head_validation_nll: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@torch.no_grad()
def teacher_forced_traces(
    model: MotifRNN,
    tensors: dict[str, Tensor],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, Tensor]:
    """Causal features, contact logits and STOP logits on selected events."""
    model.eval()
    indices = np.asarray(indices, dtype=int)
    features, contact_logits, stop_logits = [], [], []
    targets, availables, predicts, is_last, valid = [], [], [], [], []
    for begin in range(0, indices.size, int(batch_size)):
        chosen = torch.as_tensor(indices[begin:begin + int(batch_size)])
        batch = {key: tensors[key][chosen].to(device)
                 for key in ("x", "recruited", "displacement", "target", "available",
                             "valid", "is_last")}
        logits, stops, _ = model(batch["x"], batch["recruited"], batch["displacement"])
        steps = batch["x"].shape[1]
        denom = max(1, model.n_contacts - 1)
        # Rebuild the same four causal features the STOP head consumed.
        h = torch.zeros(len(chosen), model.n_nodes, device=device)
        terms = model.recurrent_terms()
        u, _ = model.axis_unit()
        gate = model.direction_gate(batch["displacement"], u)
        rows = []
        for t in range(steps):
            h = model.step(h, batch["x"][:, t], gate[:, t], terms)
            t_norm = torch.full((len(chosen),), t / denom, device=device)
            rows.append(model.state_features(h, t_norm, batch["recruited"][:, t].mean(-1)))
        features.append(torch.stack(rows, 1).cpu())
        contact_logits.append(logits.cpu())
        stop_logits.append(stops.cpu())
        targets.append(batch["target"].cpu())
        availables.append(batch["available"].cpu())
        predicts.append((batch["valid"] & ~batch["is_last"]).cpu())
        is_last.append(batch["is_last"].cpu())
        valid.append(batch["valid"].cpu())
    return {
        "features": torch.cat(features),
        "contact_logits": torch.cat(contact_logits),
        "stop_logits": torch.cat(stop_logits),
        "target": torch.cat(targets),
        "available": torch.cat(availables),
        "predict": torch.cat(predicts),
        "is_last": torch.cat(is_last),
        "valid": torch.cat(valid),
    }


def fit_size_head(
    train: dict[str, Tensor],
    calibration: dict[str, Tensor],
    n_contacts: int,
    seed: int,
    device: torch.device,
    max_epochs: int = 400,
    patience: int = 30,
) -> tuple[SizeHead, dict]:
    """Fit on train continue decisions; the calibration split selects the epoch."""
    def flatten(trace: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        predict = trace["predict"]
        x = trace["features"][predict]
        y = (trace["target"].sum(-1).long()[predict] - 1).clamp_min(0)
        return x, y

    train_x, train_y = flatten(train)
    val_x, val_y = flatten(calibration)
    if train_y.numel() == 0 or val_y.numel() == 0:
        raise RuntimeError("size head needs train and calibration continue decisions")
    torch.manual_seed(int(seed) + 4242)
    head = SizeHead(n_contacts).to(device)
    optimiser = torch.optim.Adam(head.parameters(), lr=1e-2)
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)
    best, best_state, stale = float("inf"), None, 0
    for _ in range(int(max_epochs)):
        head.train()
        loss = nn.functional.cross_entropy(head(train_x), train_y)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        head.eval()
        with torch.no_grad():
            value = float(nn.functional.cross_entropy(head(val_x), val_y))
        if value < best - 1e-6:
            best, stale = value, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is None:
        raise RuntimeError("size head produced no finite checkpoint")
    head.load_state_dict(best_state)
    return head, {
        "size_head_train_decisions": int(train_y.numel()),
        "size_head_validation_nll": float(best),
    }


def _subset_nll(logits: Tensor, target: Tensor, available: Tensor, temperature: float) -> float:
    scaled = logits / float(temperature)
    log_prob = conditional_k_subset_log_prob(scaled, target > 0.5, available)
    return float(-log_prob.mean())


def calibrate_temperatures(
    trace: dict[str, Tensor],
    size_head: SizeHead,
    device: torch.device,
    grid: Sequence[float] | None = None,
) -> dict[str, float]:
    """One scalar temperature each for contacts, cardinality and STOP."""
    grid = list(grid) if grid is not None else list(np.exp(np.linspace(np.log(0.25), np.log(4.0), 33)))
    predict = trace["predict"]
    logits = trace["contact_logits"][predict].to(device)
    target = trace["target"][predict].to(device)
    available = trace["available"][predict].to(device)
    contact_scores = [(_subset_nll(logits, target, available, t), t) for t in grid]
    contact_before = _subset_nll(logits, target, available, 1.0)
    contact_nll, contact_temperature = min(contact_scores)

    features = trace["features"][predict].to(device)
    sizes = (target.sum(-1).long() - 1).clamp_min(0)
    with torch.no_grad():
        size_logits = size_head(features)
    card_scores = []
    for t in grid:
        value = float(nn.functional.cross_entropy(size_logits / t, sizes))
        card_scores.append((value, t))
    card_before = float(nn.functional.cross_entropy(size_logits, sizes))
    card_nll, card_temperature = min(card_scores)

    valid = trace["valid"]
    stop_logits = trace["stop_logits"][valid].to(device)
    stop_target = trace["is_last"][valid].float().to(device)
    stop_scores = []
    for t in grid:
        value = float(nn.functional.binary_cross_entropy_with_logits(stop_logits / t, stop_target))
        stop_scores.append((value, t))
    stop_before = float(nn.functional.binary_cross_entropy_with_logits(stop_logits, stop_target))
    stop_bce, stop_temperature = min(stop_scores)

    return {
        "contact_temperature": float(contact_temperature),
        "cardinality_temperature": float(card_temperature),
        "stop_temperature": float(stop_temperature),
        "contact_nll_before": contact_before,
        "contact_nll_after": float(contact_nll),
        "cardinality_nll_before": card_before,
        "cardinality_nll_after": float(card_nll),
        "stop_bce_before": stop_before,
        "stop_bce_after": float(stop_bce),
        "n_calibration_decisions": int(predict.sum()),
        "n_calibration_events": int(trace["valid"].shape[0]),
    }


def _direction_weight(model: MotifRNN, displacement: Tensor, s: Tensor, u: Tensor):
    """Per-event direction weight, or ``None`` when the global axis is used."""
    if model.config.direction_mode == "GLOBAL_AXIS":
        return None
    norm = displacement.norm(dim=-1, keepdim=True)
    return displacement / (norm + 1e-9)


@torch.no_grad()
def sample_next_set(
    logits: Tensor,
    available: Tensor,
    cardinality: Tensor,
    generator: torch.Generator,
    uniforms: Tensor | None = None,
) -> Tensor:
    """Exact fixed-cardinality subset draw with a singleton fast path.

    The frozen cohort has a tied-rank fraction below 1.4e-4, so almost every
    row asks for one contact.  For those rows the fixed-cardinality law *is* the
    masked softmax, and Gumbel-max over a pre-drawn uniform block reproduces it
    while removing roughly two hundred small kernel launches per step.  The
    uniform block has a fixed shape, so the stream is identical across models
    even when they disagree about the cardinality.
    """
    if uniforms is None:
        uniforms = torch.rand(logits.shape, device=logits.device, dtype=logits.dtype,
                              generator=generator)
    picked = torch.zeros_like(available)
    singles = cardinality == 1
    if bool(singles.any()):
        masked = logits[singles].masked_fill(~available[singles], NEG_INF)
        u = uniforms[singles].clamp(1e-20, 1.0 - 1e-7)
        chosen = (masked - torch.log(-torch.log(u))).argmax(dim=-1)
        rows = torch.zeros_like(masked, dtype=torch.bool)
        rows.scatter_(1, chosen[:, None], True)
        picked[singles] = rows
    multiples = cardinality > 1
    if bool(multiples.any()):
        picked[multiples] = sample_conditional_k_subset(
            logits[multiples], available[multiples], cardinality[multiples],
            generator=generator,
        )
    return picked


@torch.no_grad()
def stochastic_rollout(
    model: MotifRNN,
    size_head: SizeHead,
    contract: DecoderContract,
    starts: Tensor,
    contacts_xy_mm: np.ndarray,
    device: torch.device,
    *,
    mode: str = "FULL_STOP",
    horizon: int | None = None,
    rng_label: str = "rollout",
    gate_rule: str = "M2-2RANK",
) -> dict[str, np.ndarray]:
    """Batched closed-loop generation from an observed first rank set.

    ``mode='FULL_STOP'`` keeps STOP, cardinality, the repeat mask and the
    maximum-rank rule.  ``mode='FIXED_H'`` ignores STOP and always emits
    ``horizon`` further rank sets, so direction and extent are scored without
    the termination head.
    """
    if mode not in ("FULL_STOP", "FIXED_H"):
        raise ValueError(f"unknown rollout mode {mode!r}")
    if mode == "FIXED_H" and not horizon:
        raise ValueError("FIXED_H needs a positive horizon")
    model.eval()
    size_head.eval()
    prefix = starts.to(device).float()
    if prefix.dim() == 2:
        prefix = prefix[:, None, :]
    batch, n_prefix, _ = prefix.shape
    n_contacts = model.n_contacts
    xy = torch.as_tensor(np.asarray(contacts_xy_mm, dtype=np.float32), device=device)

    max_steps = int(horizon) if mode == "FIXED_H" else int(n_contacts)
    h = torch.zeros(batch, model.n_nodes, device=device)
    terms = model.recurrent_terms()
    u, _ = model.axis_unit()

    recruited = (prefix.sum(1) > 0).float()
    x = prefix[:, 0].clone()
    counts = prefix.sum(-1, keepdim=True).clamp_min(1.0)
    centroid_start = prefix[:, 0] @ xy / counts[:, 0]
    displacement = torch.zeros(batch, 2, device=device)
    alive = torch.ones(batch, dtype=torch.bool, device=device)
    denom = max(1, n_contacts - 1)

    sequence = torch.zeros(batch, n_prefix + max_steps, n_contacts, device=device)
    sequence[:, :n_prefix] = prefix
    emitted = torch.zeros(batch, dtype=torch.long, device=device)
    stop_step = torch.full((batch,), max_steps + 1, dtype=torch.long, device=device)

    stop_generator = torch.Generator(device="cpu").manual_seed(stable_seed(rng_label, 11))
    card_generator = torch.Generator(device="cpu").manual_seed(stable_seed(rng_label, 22))

    # Drive the observed prefix through the model before free generation so the
    # counterfactual sees the same closed-loop state the real event produced.
    for t in range(1, n_prefix):
        s = model.direction_gate(displacement, u)
        h = model.step(h, x, s, terms, _direction_weight(model, displacement, s, u))
        x = prefix[:, t]
        centroid_now = (prefix[:, t] @ xy) / counts[:, t]
        displacement = rollout_displacement_update(
            displacement, centroid_start, centroid_now, t, gate_rule)

    for t in range(n_prefix - 1, n_prefix - 1 + max_steps):
        s = model.direction_gate(displacement, u)
        h = model.step(h, x, s, terms, _direction_weight(model, displacement, s, u))
        logits = model.readout(h)
        t_norm = torch.full((batch,), t / denom, device=device)
        features = model.state_features(h, t_norm, recruited.mean(-1))

        available = recruited < 0.5
        exhausted = ~available.any(-1)
        if mode == "FULL_STOP":
            stop_probability = torch.sigmoid(
                model.stop_logit(features) / contract.stop_temperature
            )
            draw = torch.rand(batch, generator=stop_generator).to(device)
            stops = (draw < stop_probability) | exhausted
        else:
            stops = exhausted
        newly_stopped = alive & stops
        stop_step = torch.where(newly_stopped, torch.full_like(stop_step, t), stop_step)
        alive = alive & ~stops
        if not bool(alive.any()):
            break

        size_logits = size_head(features) / contract.cardinality_temperature
        candidate_count = available.sum(-1)
        size_mask = torch.arange(n_contacts, device=device)[None, :] < candidate_count[:, None]
        size_logits = size_logits.masked_fill(~size_mask, NEG_INF)
        probability = torch.softmax(size_logits, dim=-1)
        draw = torch.rand(batch, generator=card_generator).to(device)
        cumulative = probability.cumsum(-1)
        k = (cumulative < draw[:, None]).sum(-1).clamp(0, n_contacts - 1) + 1
        k = torch.minimum(k, candidate_count.clamp_min(1))
        k = torch.where(alive, k, torch.zeros_like(k))

        subset_generator = torch.Generator(device=logits.device).manual_seed(
            stable_seed(rng_label, 1000 + t)
        )
        uniforms = torch.rand(logits.shape, device=logits.device, dtype=logits.dtype,
                              generator=subset_generator)
        picked = sample_next_set(
            logits / contract.contact_temperature,
            available,
            k,
            generator=subset_generator,
            uniforms=uniforms,
        ).float()
        picked = picked * alive[:, None].float()

        sequence[:, t + 1] = picked
        emitted = emitted + alive.long()
        recruited = torch.clamp(recruited + picked, max=1.0)
        x = picked
        count = picked.sum(-1, keepdim=True)
        centroid_now = torch.where(
            count > 0, (picked @ xy) / count.clamp_min(1.0), centroid_start
        )
        displacement = torch.where(
            alive[:, None],
            rollout_displacement_update(
                displacement, centroid_start, centroid_now, t + 1, gate_rule
            ),
            displacement,
        )

    return {
        "sequence": sequence.cpu().numpy().astype(np.uint8),
        "n_emitted": (emitted + (n_prefix - 1)).cpu().numpy().astype(np.int64),
        "n_generated": emitted.cpu().numpy().astype(np.int64),
        "n_prefix": int(n_prefix),
        "stop_step": stop_step.cpu().numpy().astype(np.int64),
    }


def summarise_sequences(
    sequence: np.ndarray,
    n_emitted: np.ndarray,
    contacts_xy_mm: np.ndarray,
    axis_u: np.ndarray,
    late_fraction: float = 0.2,
) -> dict[str, np.ndarray]:
    """Endpoint / extent / length summary ``S`` for real and generated events."""
    xy = np.asarray(contacts_xy_mm, dtype=float)
    u = np.asarray(axis_u, dtype=float)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    u_perp = np.array([-u[1], u[0]])
    sequence = np.asarray(sequence)
    batch, steps, n_contacts = sequence.shape
    dense = sequence.astype(np.float32)
    counts = dense.sum(-1)                                        # (B, S)
    within = np.arange(steps)[None, :] <= np.asarray(n_emitted)[:, None]
    used = (counts > 0) & within
    centroids = np.zeros((batch, steps, 2))
    np.divide(dense @ xy, counts[..., None], out=centroids, where=counts[..., None] > 0)

    n_rank = used.sum(1).astype(int)
    last_index = np.where(n_rank > 0, used.shape[1] - 1 - used[:, ::-1].argmax(1), 0)
    rows = np.arange(batch)
    r_last = centroids[rows, last_index]

    n_late = np.maximum(1, np.ceil(late_fraction * np.maximum(n_rank, 1)).astype(int))
    order = np.cumsum(used[:, ::-1], axis=1)[:, ::-1]             # rank from the end, 1-based
    late = used & (order <= n_late[:, None])
    late_weight = late.astype(float)
    r_late = (centroids * late_weight[..., None]).sum(1) / np.maximum(late_weight.sum(1), 1)[:, None]

    field = ((dense * within[..., None]).sum(1) > 0).astype(np.float32)
    projected = xy @ u
    transverse = xy @ u_perp
    big = np.where(field > 0, projected[None, :], np.inf)
    small = np.where(field > 0, projected[None, :], -np.inf)
    l_axis = np.where(field.sum(1) > 0, small.max(1) - big.min(1), 0.0)
    big_t = np.where(field > 0, transverse[None, :], np.inf)
    small_t = np.where(field > 0, transverse[None, :], -np.inf)
    l_orth = np.where(field.sum(1) > 0, small_t.max(1) - big_t.min(1), 0.0)
    n_contact = field.sum(1).astype(int)
    empty = n_rank == 0
    r_last[empty] = 0.0
    r_late[empty] = 0.0
    return {
        "r_last": r_last,
        "r_late": r_late,
        "l_axis": l_axis,
        "l_orth": l_orth,
        "n_rank": n_rank,
        "n_contact": n_contact,
        "contact_field": field,
    }


def energy_score(samples: np.ndarray, observation: np.ndarray) -> float:
    """Proper multivariate energy score ``E||S-y|| - 0.5 E||S-S'||``."""
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    observation = np.asarray(observation, dtype=float).reshape(1, -1)
    if samples.shape[0] < 2:
        return float(np.linalg.norm(samples - observation, axis=1).mean())
    first = float(np.linalg.norm(samples - observation, axis=1).mean())
    difference = samples[:, None, :] - samples[None, :, :]
    second = float(np.linalg.norm(difference, axis=-1).mean())
    return first - 0.5 * second


def energy_score_error(samples: np.ndarray, observation: np.ndarray, n_batches: int = 4) -> float:
    """Monte-Carlo standard error of the energy score from disjoint sub-batches."""
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    if samples.shape[0] < 2 * n_batches:
        return float("nan")
    blocks = np.array_split(np.arange(samples.shape[0]), n_batches)
    values = [energy_score(samples[block], observation) for block in blocks]
    return float(np.std(values, ddof=1) / np.sqrt(n_batches))
