"""Open- and closed-loop branch execution for Topic 5.2 perturbations."""
from __future__ import annotations

import numpy as np
import torch

from src.topic5_latent_landscape_v0_2 import rank_matrix_to_event_fields


@torch.no_grad()
def raw_logits_stop(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    rank_index: torch.Tensor,
    recruited: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = model._readout(hidden)
    fraction = recruited.float().mean(-1)
    t_norm = rank_index.float() / max(1, int(model.n_contacts) - 1)
    stop = torch.sigmoid(model._stop(hidden, t_norm, fraction))
    unit = hidden.reshape(hidden.shape[0], model.n_nodes, model.state_dim).mean(-1)
    features = torch.stack(
        (unit.mean(-1), unit.max(-1).values, t_norm, fraction), dim=-1
    )
    return logits, stop, features


def deterministic_sets(
    logits: np.ndarray,
    recruited: np.ndarray,
    sizes: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Frozen decoder tie-break: highest raw logit, then lowest contact index."""
    output = np.zeros_like(recruited, dtype=np.uint8)
    for row in np.flatnonzero(active):
        eligible = np.flatnonzero(~recruited[row].astype(bool))
        if not len(eligible):
            continue
        count = min(int(sizes[row]), len(eligible))
        order = np.lexsort((eligible, -logits[row, eligible]))
        output[row, eligible[order[:count]]] = 1
    return output


@torch.no_grad()
def rollout_hidden_branches(
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    hidden: np.ndarray,
    recruited: np.ndarray,
    rank_index: np.ndarray,
    prefix_ranks: np.ndarray,
    device: torch.device,
    response_horizon: int = 3,
) -> dict[str, np.ndarray]:
    """Roll out arbitrary hidden branches while preserving each branch's q bookkeeping."""
    h = torch.as_tensor(hidden, dtype=torch.float32, device=device)
    r = torch.as_tensor(recruited, dtype=torch.bool, device=device)
    k = torch.as_tensor(rank_index, dtype=torch.long, device=device)
    active = torch.ones(len(hidden), dtype=torch.bool, device=device)
    terminal = np.asarray(prefix_ranks, dtype=np.int16).copy()
    raw = np.full(
        (len(hidden), int(response_horizon) + 1, int(model.n_contacts)),
        np.nan, dtype=np.float32,
    )
    stop_trajectory = np.full(
        (len(hidden), int(model.n_contacts) + 1), np.nan, dtype=np.float32
    )
    active_trajectory = np.zeros(
        (len(hidden), int(model.n_contacts) + 1), dtype=np.uint8
    )
    generated_steps = np.zeros(len(hidden), dtype=np.int16)
    for rollout_step in range(int(model.n_contacts) + 1):
        logits, stop, features = raw_logits_stop(model, h, k, r)
        finite = bool(torch.isfinite(logits).all() and torch.isfinite(stop).all())
        if not finite:
            raise FloatingPointError("closed-loop branch produced nonfinite logits/STOP")
        active_np = active.detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()
        stop_np = stop.detach().cpu().numpy()
        if rollout_step <= int(response_horizon):
            raw[active_np, rollout_step] = logits_np[active_np]
        stop_trajectory[active_np, rollout_step] = stop_np[active_np]
        active_trajectory[:, rollout_step] = active_np.astype(np.uint8)
        exhausted = r.all(-1)
        continuing = active & (stop < 0.5) & ~exhausted
        if not bool(continuing.any()) or rollout_step == int(model.n_contacts):
            active = continuing
            break
        sizes = decoder(features).argmax(-1) + 1
        next_set = deterministic_sets(
            logits_np,
            r.detach().cpu().numpy(),
            sizes.detach().cpu().numpy(),
            continuing.detach().cpu().numpy(),
        )
        next_set_tensor = torch.as_tensor(next_set, dtype=torch.float32, device=device)
        next_h = model._step(h, next_set_tensor)
        h = torch.where(continuing[:, None], next_h, h)
        r = r | next_set_tensor.bool()
        continuing_np = continuing.detach().cpu().numpy()
        k = torch.where(continuing, k + 1, k)
        generated_steps[continuing_np] += 1
        terminal_rows = np.flatnonzero(continuing_np)
        for row in terminal_rows:
            terminal[row, next_set[row].astype(bool)] = int(
                np.asarray(rank_index, dtype=int)[row] + generated_steps[row]
            )
        active = continuing
    full_field, recurrence_field = rank_matrix_to_event_fields(terminal)
    return {
        "raw_logits": raw,
        "stop_probability": stop_trajectory,
        "active": active_trajectory,
        "terminal_ranks": terminal,
        "terminal_full_field": full_field.astype(np.float32),
        "terminal_start_removed_field": recurrence_field.astype(np.float32),
        "generated_steps": generated_steps,
    }


def project_centered_contact_response(
    response: np.ndarray, progress_axis: np.ndarray, field_axis: np.ndarray
) -> np.ndarray:
    values = np.asarray(response, dtype=np.float64)
    finite = np.isfinite(values)
    count = finite.sum(axis=-1, keepdims=True)
    mean = np.divide(
        np.where(finite, values, 0.0).sum(axis=-1, keepdims=True),
        np.maximum(count, 1),
    )
    centered = np.where(finite, values - mean, 0.0)
    return np.stack([
        np.einsum("...c,c->...", centered, np.asarray(progress_axis, dtype=float)),
        np.einsum("...c,c->...", centered, np.asarray(field_axis, dtype=float)),
    ], axis=-1)


def prefix_ranks_for_references(
    ranks: np.ndarray, event_index: np.ndarray, step: np.ndarray
) -> np.ndarray:
    source = np.asarray(ranks)[np.asarray(event_index, dtype=int)]
    limit = np.asarray(step, dtype=int)[:, None]
    return np.where((source >= 0) & (source <= limit), source, -1).astype(np.int16)
