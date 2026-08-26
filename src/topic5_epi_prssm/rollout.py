"""Cohort scan, observer-off open-loop rollout, state reset and delta-t shuffle.

The open-loop contract is structural, not conventional: with ``correction_on``
false the scan never touches ``marks`` again and the exposure arm consumes an
expected load, so a future mark or a future load cannot leak in by accident.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import CohortBatch, EpiPRSSM, PatientTensors, SlowState

ENDPOINTS = ("event_nll", "order_nll", "selection_nll", "stop_nll", "participation_nll")


@dataclass
class ScanResult:
    state_minus: torch.Tensor       # (T, P, N_pad, D)
    resource_minus: torch.Tensor    # (T, P)
    exposure_minus: torch.Tensor    # (T, P)
    active: torch.Tensor            # (P, T)
    final: SlowState
    correction_energy: torch.Tensor
    flexible_penalty: torch.Tensor
    resource_floor_fraction: float


def cohort_scan(model: EpiPRSSM, batch: CohortBatch, t0: int, t1: int, z0: SlowState,
                *, correction_on: bool = True, expected_load: torch.Tensor | None = None,
                collect: bool = True) -> ScanResult:
    step = batch.gather(t0, t1)
    z = z0
    states: list[torch.Tensor] = []
    resources: list[torch.Tensor] = []
    exposures: list[torch.Tensor] = []
    device = batch.device
    energy = torch.zeros((), device=device)
    penalty = torch.zeros((), device=device)
    floor = torch.zeros((), device=device)
    n_steps = t1 - t0
    # a chunk that ends before the shortest patient runs out needs no blending;
    # decided from the frozen lengths, never by synchronising a device tensor
    all_active = bool(t1 <= int(batch.lengths.min())) if len(batch.lengths) else False
    for t in range(n_steps):
        moved = model.propagate(z, batch, step, t)
        if all_active:
            z_new = moved
        else:
            z_new = z.blend(moved, step["active"][:, t])
        if collect:
            states.append(z_new.state)
            resources.append(z_new.resource)
            exposures.append(z_new.exposure)
        floor = floor + (z_new.resource <= 1.01e-3).float().mean()
        absorbed = model.absorb(z_new, step, t, load=None if correction_on else expected_load)
        z = absorbed if all_active else z_new.blend(absorbed, step["active"][:, t])
        if correction_on:
            observed, step_energy, step_penalty = model.observe(z, batch, step, t)
            z = observed if all_active else z.blend(observed, step["active"][:, t])
            energy = energy + step_energy
            penalty = penalty + step_penalty
    return ScanResult(
        state_minus=torch.stack(states) if collect and states else torch.empty(0),
        resource_minus=torch.stack(resources) if collect and resources else torch.empty(0),
        exposure_minus=torch.stack(exposures) if collect and exposures else torch.empty(0),
        active=step["active"],
        final=z,
        correction_energy=energy / max(n_steps, 1),
        flexible_penalty=penalty / max(n_steps, 1),
        resource_floor_fraction=float(floor.item()) / max(n_steps, 1),
    )


def score_scan(model: EpiPRSSM, batch: CohortBatch, result: ScanResult, t0: int,
               ) -> dict[str, dict[str, np.ndarray]]:
    """Per-patient endpoint arrays for the events covered by ``result``."""
    out: dict[str, dict[str, np.ndarray]] = {}
    span = result.state_minus.shape[0] if result.state_minus.numel() else 0
    for p, patient in enumerate(batch.patients):
        take = int(min(span, max(int(batch.lengths[p]) - t0, 0)))
        if take == 0:
            continue
        lo = batch.starts[p] + t0
        index = torch.arange(lo, lo + take, device=batch.device)
        state = result.state_minus[:take, p, : patient.n_contacts, :]
        resource = result.resource_minus[:take, p]
        scores = model.score_events(patient, index, state, resource)
        out[patient.subject] = {k: scores[k].detach().cpu().numpy() for k in ENDPOINTS}
        out[patient.subject]["event_index"] = index.cpu().numpy()
        out[patient.subject]["exposure"] = result.exposure_minus[:take, p].detach().cpu().numpy()
        out[patient.subject]["resource"] = resource.detach().cpu().numpy()
    return out


def scan_loss(model: EpiPRSSM, batch: CohortBatch, result: ScanResult, t0: int,
              order_weight: float = 0.0) -> torch.Tensor:
    """Mean training loss over the events covered by ``result`` (graph retained).

    ``order_weight`` adds the masked recruitment-order likelihood to the objective.
    It defaults to zero, which is the joint event likelihood the ladder was trained
    on; an arm that sets it asks whether the state can explain the ordering when the
    ordering is actually a training target rather than an unoptimised read-out.
    """
    total = None
    count = 0
    span = result.state_minus.shape[0] if result.state_minus.numel() else 0
    for p, patient in enumerate(batch.patients):
        take = int(min(span, max(int(batch.lengths[p]) - t0, 0)))
        if take == 0:
            continue
        lo = batch.starts[p] + t0
        index = torch.arange(lo, lo + take, device=batch.device)
        state = result.state_minus[:take, p, : patient.n_contacts, :]
        resource = result.resource_minus[:take, p]
        scores = model.score_events(patient, index, state, resource)
        term = (scores["event_nll"] + scores["participation_nll"]).sum()
        if order_weight:
            term = term + order_weight * scores["order_nll"].sum()
        total = term if total is None else total + term
        count += take
    if total is None:
        return torch.zeros((), device=batch.device, requires_grad=True)
    return total / max(count, 1)


@torch.no_grad()
def score_window(model: EpiPRSSM, batch: CohortBatch, z0: SlowState, *, chunk: int = 256,
                 correction_on: bool = True, expected_load: torch.Tensor | None = None
                 ) -> tuple[dict[str, dict[str, np.ndarray]], SlowState]:
    """Score a whole aligned window, carrying the state across chunks."""
    collected: dict[str, dict[str, list[np.ndarray]]] = {}
    z = z0
    total = batch.max_length
    position = 0
    while position < total:
        end = min(position + chunk, total)
        result = cohort_scan(model, batch, position, end, z, correction_on=correction_on,
                             expected_load=expected_load)
        piece = score_scan(model, batch, result, position)
        for subject, values in piece.items():
            store = collected.setdefault(subject, {k: [] for k in values})
            for key, array in values.items():
                store[key].append(array)
        z = result.final
        position = end
    merged = {s: {k: np.concatenate(v) for k, v in d.items()} for s, d in collected.items()}
    return merged, z


@torch.no_grad()
def carry_state(model: EpiPRSSM, batch: CohortBatch, z0: SlowState, *, chunk: int = 512
                ) -> SlowState:
    """Warm the state causally through a window without collecting or scoring."""
    z = z0
    total = batch.max_length
    position = 0
    while position < total:
        end = min(position + chunk, total)
        z = cohort_scan(model, batch, position, end, z, correction_on=True, collect=False).final
        position = end
    return z


def shuffled_delta_t_batch(batch: CohortBatch, rng: np.random.Generator) -> CohortBatch:
    """Permute inter-event intervals inside each patient's window, keeping order.

    Separates 'the state uses real elapsed time' from 'the state uses event
    order'.  Marks, participation and splits are untouched.
    """
    import copy
    patients = []
    for p, patient in enumerate(batch.patients):
        clone = copy.copy(patient)
        delta = patient.delta_t.clone()
        lo, hi = int(batch.starts[p]), int(batch.starts[p] + batch.lengths[p])
        window = delta[lo:hi].cpu().numpy()
        delta[lo:hi] = torch.as_tensor(window[rng.permutation(len(window))],
                                       dtype=delta.dtype, device=delta.device)
        clone.delta_t = delta
        clone.log_delta_t = torch.log1p(delta)
        patients.append(clone)
    return CohortBatch(tuple(patients), batch.starts, batch.lengths, batch.n_pad,
                       batch.node_mask, batch.adjacency, batch.device)
