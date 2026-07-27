"""Coordinate-free node-bias and first-order Markov controls for v2.2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize


EPS = 1.0e-8


@dataclass(frozen=True)
class SharedStop:
    c0: float
    c_n: float
    n_decisions: int
    n_terminal: int
    optimizer_success: bool


def decision_rows(
    groups: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return train-only seen fractions and terminal labels."""
    groups = np.asarray(groups, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    seen_fraction = []
    terminal = []
    for event_index in indices:
        event = groups[event_index]
        present = event >= 0
        if not np.any(present):
            raise ValueError("event has no participating contacts")
        n_steps = int(np.max(event[present])) + 1
        for step in range(n_steps):
            seen_fraction.append(float(np.mean((event >= 0) & (event <= step))))
            terminal.append(float(step + 1 == n_steps))
    return (
        np.asarray(seen_fraction, dtype=np.float64),
        np.asarray(terminal, dtype=np.float64),
    )


def fit_shared_stop(
    patient_decisions: Iterable[tuple[np.ndarray, np.ndarray]],
) -> SharedStop:
    """Fit one c0 + c_n*f_seen STOP from pooled train80 decisions."""
    rows = list(patient_decisions)
    if not rows:
        raise ValueError("at least one patient decision table is required")
    seen = np.concatenate([row[0] for row in rows])
    terminal = np.concatenate([row[1] for row in rows])
    if seen.size == 0 or terminal.shape != seen.shape:
        raise ValueError("invalid STOP decision inventory")
    terminal_rate = np.clip(np.mean(terminal), EPS, 1.0 - EPS)
    initial = np.array(
        [np.log(terminal_rate) - np.log1p(-terminal_rate), 0.0],
        dtype=np.float64,
    )

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        c0, c_n = theta
        logits = c0 + c_n * seen
        loss = np.logaddexp(0.0, logits) - terminal * logits
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        residual = probability - terminal
        gradient = np.array(
            [np.sum(residual), np.sum(residual * seen)],
            dtype=np.float64,
        )
        return float(np.sum(loss)), gradient

    result = minimize(
        lambda theta: objective(theta)[0],
        initial,
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        bounds=[(None, None), (0.0, None)],
        options={"maxiter": 10_000, "ftol": 1.0e-12, "gtol": 1.0e-9},
    )
    if not np.all(np.isfinite(result.x)):
        raise FloatingPointError("non-finite shared STOP fit")
    return SharedStop(
        c0=float(result.x[0]),
        c_n=float(result.x[1]),
        n_decisions=int(len(seen)),
        n_terminal=int(np.sum(terminal)),
        optimizer_success=bool(result.success),
    )


def estimate_hazard(groups: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    """Train-only discrete next-rank hazard with Beta(1,1) smoothing."""
    groups = np.asarray(groups, dtype=np.int64)
    n_contacts = groups.shape[1]
    n_next = np.zeros(n_contacts, dtype=np.float64)
    n_eligible = np.zeros(n_contacts, dtype=np.float64)
    for event_index in np.asarray(train_indices, dtype=np.int64):
        event = groups[event_index]
        present = event >= 0
        n_steps = int(np.max(event[present])) + 1
        for step in range(n_steps):
            seen = (event >= 0) & (event <= step)
            eligible = ~seen
            n_eligible += eligible
            if step + 1 < n_steps:
                n_next += event == (step + 1)
    return np.clip(
        (n_next + 1.0) / (n_eligible + 2.0),
        EPS,
        1.0 - EPS,
    )


def estimate_markov(
    groups: np.ndarray,
    train_indices: np.ndarray,
    node_hazard: np.ndarray,
    *,
    concentration: float = 10.0,
) -> np.ndarray:
    """Tie-weighted first-order transition hazard with frozen smoothing."""
    groups = np.asarray(groups, dtype=np.int64)
    node_hazard = np.asarray(node_hazard, dtype=np.float64)
    n_contacts = groups.shape[1]
    if node_hazard.shape != (n_contacts,):
        raise ValueError("node hazard does not align with contacts")
    count = np.zeros((n_contacts, n_contacts), dtype=np.float64)
    exposure = np.zeros_like(count)
    for event_index in np.asarray(train_indices, dtype=np.int64):
        event = groups[event_index]
        present = event >= 0
        n_steps = int(np.max(event[present])) + 1
        for step in range(n_steps):
            current = np.flatnonzero(event == step)
            if current.size == 0:
                raise ValueError("rank sets must be contiguous and non-empty")
            weight = 1.0 / current.size
            seen = (event >= 0) & (event <= step)
            eligible = np.flatnonzero(~seen)
            exposure[np.ix_(current, eligible)] += weight
            if step + 1 < n_steps:
                following = np.flatnonzero(event == (step + 1))
                count[np.ix_(current, following)] += weight
    transition = (
        count + concentration * node_hazard[None, :]
    ) / (exposure + concentration)
    return np.clip(transition, EPS, 1.0 - EPS)


def _log_sigmoid(value: float) -> float:
    return float(-np.logaddexp(0.0, -value))


def _event_nll(
    *,
    event: np.ndarray,
    node_hazard: np.ndarray,
    stop: SharedStop,
    transition: np.ndarray | None,
) -> float:
    present = event >= 0
    n_steps = int(np.max(event[present])) + 1
    terms = []
    for step in range(n_steps):
        current = event == step
        seen = (event >= 0) & (event <= step)
        eligible = ~seen
        eligible_count = max(1, int(np.sum(eligible)))
        stop_logit = stop.c0 + stop.c_n * float(np.mean(seen))
        terminal = step + 1 == n_steps
        if terminal:
            log_probability = _log_sigmoid(stop_logit)
        else:
            if transition is None:
                hazard = node_hazard
            else:
                hazard = np.mean(transition[current], axis=0)
            hazard = np.clip(hazard, EPS, 1.0 - EPS)
            target = event == (step + 1)
            bernoulli = np.sum(
                target[eligible] * np.log(hazard[eligible])
                + (~target[eligible]) * np.log1p(-hazard[eligible])
            )
            log_empty = float(np.sum(np.log1p(-hazard[eligible])))
            log_z = float(np.log(-np.expm1(min(log_empty, -np.finfo(float).eps))))
            log_probability = (
                _log_sigmoid(-stop_logit) + float(bernoulli) - log_z
            )
        terms.append(-log_probability / eligible_count)
    return float(np.mean(terms))


def evaluate_models(
    *,
    groups: np.ndarray,
    heldout_indices: np.ndarray,
    node_hazard: np.ndarray,
    transition: np.ndarray,
    stop: SharedStop,
) -> dict[str, np.ndarray | float]:
    """Return event-first heldout NLL for both matched controls."""
    node = []
    markov = []
    for event_index in np.asarray(heldout_indices, dtype=np.int64):
        event = np.asarray(groups[event_index], dtype=np.int64)
        node.append(
            _event_nll(
                event=event,
                node_hazard=node_hazard,
                stop=stop,
                transition=None,
            )
        )
        markov.append(
            _event_nll(
                event=event,
                node_hazard=node_hazard,
                stop=stop,
                transition=transition,
            )
        )
    node_array = np.asarray(node, dtype=np.float64)
    markov_array = np.asarray(markov, dtype=np.float64)
    if not np.all(np.isfinite(node_array)) or not np.all(np.isfinite(markov_array)):
        raise FloatingPointError("non-finite sequence sensitivity NLL")
    return {
        "node_event_nll": node_array,
        "markov_event_nll": markov_array,
        "node_patient_nll": float(np.mean(node_array)),
        "markov_patient_nll": float(np.mean(markov_array)),
        "markov_benefit": float(np.mean(node_array) - np.mean(markov_array)),
    }


def contact_descriptives(
    *,
    groups: np.ndarray,
    train_indices: np.ndarray,
    heldout_indices: np.ndarray,
    node_hazard: np.ndarray,
    transition: np.ndarray,
) -> list[dict[str, float | int]]:
    """Observed participation/rank summaries plus heldout prefix hazards."""
    groups = np.asarray(groups, dtype=np.int64)
    n_contacts = groups.shape[1]
    eligible_count = np.zeros(n_contacts, dtype=np.float64)
    observed_next = np.zeros(n_contacts, dtype=np.float64)
    markov_hazard_sum = np.zeros(n_contacts, dtype=np.float64)
    for event_index in np.asarray(heldout_indices, dtype=np.int64):
        event = groups[event_index]
        n_steps = int(np.max(event[event >= 0])) + 1
        for step in range(n_steps):
            current = event == step
            seen = (event >= 0) & (event <= step)
            eligible = ~seen
            eligible_count += eligible
            if step + 1 < n_steps:
                observed_next += event == (step + 1)
            hazard = np.mean(transition[current], axis=0)
            markov_hazard_sum += eligible * hazard

    rows = []
    train = groups[np.asarray(train_indices, dtype=np.int64)]
    heldout = groups[np.asarray(heldout_indices, dtype=np.int64)]
    for contact in range(n_contacts):
        train_rank = train[:, contact]
        heldout_rank = heldout[:, contact]
        train_valid = train_rank >= 0
        heldout_valid = heldout_rank >= 0
        heldout_normalized = []
        for event, rank in zip(heldout, heldout_rank):
            if rank < 0:
                continue
            n_steps = int(np.max(event[event >= 0])) + 1
            heldout_normalized.append(
                float(rank / max(1, n_steps - 1))
            )
        rows.append(
            {
                "contact_index": contact,
                "train_participation_probability": float(np.mean(train_valid)),
                "heldout_participation_probability": float(
                    np.mean(heldout_valid)
                ),
                "heldout_normalized_rank_median": (
                    float(np.median(heldout_normalized))
                    if heldout_normalized
                    else float("nan")
                ),
                "heldout_next_hazard_observed": float(
                    observed_next[contact] / max(1.0, eligible_count[contact])
                ),
                "node_bias_next_hazard": float(node_hazard[contact]),
                "markov_next_hazard_mean": float(
                    markov_hazard_sum[contact]
                    / max(1.0, eligible_count[contact])
                ),
                "heldout_eligible_decisions": int(eligible_count[contact]),
            }
        )
    return rows
