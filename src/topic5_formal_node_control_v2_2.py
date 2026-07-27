"""Exact coordinate-free formal node-bias control for Topic-5 v2.2."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


EPS = 1.0e-8


@dataclass(frozen=True)
class StopHistogram:
    seen_fraction: np.ndarray
    decision_weight: np.ndarray
    terminal_weight: np.ndarray
    raw_decisions: np.ndarray
    raw_terminal: np.ndarray


@dataclass(frozen=True)
class NodeControlStop:
    c0: float
    c_n: float
    n_decisions: int
    n_terminal: int
    optimizer_success: bool


def stop_histogram(groups: np.ndarray, indices: np.ndarray) -> StopHistogram:
    """Compress the patient-balanced event-first STOP objective exactly.

    Each patient's train events have total event weight one.  Within an event,
    decisions are averaged, and each decision log likelihood is normalized by
    its number of eligible contacts, exactly as in the formal RNN loss.
    """
    values = np.asarray(groups, dtype=np.int64)
    n_contacts = values.shape[1]
    event_indices = np.asarray(indices, dtype=np.int64)
    if event_indices.size == 0:
        raise ValueError("patient has no train events")
    decision_weight = np.zeros(n_contacts + 1, dtype=np.float64)
    terminal_weight = np.zeros(n_contacts + 1, dtype=np.float64)
    decisions = np.zeros(n_contacts + 1, dtype=np.int64)
    terminals = np.zeros(n_contacts + 1, dtype=np.int64)
    for event_index in event_indices:
        event = values[event_index]
        present = event >= 0
        if not np.any(present):
            raise ValueError("event has no participating contacts")
        n_steps = int(np.max(event[present])) + 1
        for step in range(n_steps):
            seen = (event >= 0) & (event <= step)
            n_seen = int(np.sum(seen))
            n_eligible = max(1, int(np.sum(~seen)))
            weight = (
                1.0
                / float(event_indices.size)
                / float(n_steps)
                / float(n_eligible)
            )
            decision_weight[n_seen] += weight
            decisions[n_seen] += 1
            if step + 1 == n_steps:
                terminal_weight[n_seen] += weight
                terminals[n_seen] += 1
    keep = decisions > 0
    return StopHistogram(
        seen_fraction=np.flatnonzero(keep).astype(np.float64) / n_contacts,
        decision_weight=decision_weight[keep],
        terminal_weight=terminal_weight[keep],
        raw_decisions=decisions[keep],
        raw_terminal=terminals[keep],
    )


def fit_loso_stop(histograms: Iterable[StopHistogram]) -> NodeControlStop:
    rows = list(histograms)
    if not rows:
        raise ValueError("at least one training histogram is required")
    seen = np.concatenate([row.seen_fraction for row in rows])
    decisions = np.concatenate([row.decision_weight for row in rows])
    terminal = np.concatenate([row.terminal_weight for row in rows])
    raw_decisions = np.concatenate([row.raw_decisions for row in rows])
    raw_terminal = np.concatenate([row.raw_terminal for row in rows])
    terminal_rate = np.clip(
        terminal.sum() / decisions.sum(), EPS, 1.0 - EPS
    )
    initial = np.asarray(
        [np.log(terminal_rate) - np.log1p(-terminal_rate), 0.0],
        dtype=np.float64,
    )

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = theta[0] + theta[1] * seen
        loss = decisions * np.logaddexp(0.0, logits) - terminal * logits
        probability = 1.0 / (
            1.0 + np.exp(-np.clip(logits, -50.0, 50.0))
        )
        residual = decisions * probability - terminal
        gradient = np.asarray(
            [residual.sum(), np.sum(residual * seen)], dtype=np.float64
        )
        return float(loss.sum()), gradient

    result = minimize(
        lambda theta: objective(theta)[0],
        initial,
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        bounds=[(None, None), (0.0, None)],
        options={"maxiter": 10_000, "ftol": 1.0e-12, "gtol": 1.0e-9},
    )
    if not np.all(np.isfinite(result.x)):
        raise FloatingPointError("non-finite formal STOP fit")
    return NodeControlStop(
        c0=float(result.x[0]),
        c_n=float(result.x[1]),
        n_decisions=int(raw_decisions.sum()),
        n_terminal=int(raw_terminal.sum()),
        optimizer_success=bool(result.success),
    )


def estimate_node_hazard(
    groups: np.ndarray, train_indices: np.ndarray
) -> np.ndarray:
    """Eligible-prefix next-rank hazard with Beta(1,1) smoothing."""
    values = np.asarray(groups, dtype=np.int64)
    n_next = np.zeros(values.shape[1], dtype=np.float64)
    n_eligible = np.zeros(values.shape[1], dtype=np.float64)
    for event_index in np.asarray(train_indices, dtype=np.int64):
        event = values[event_index]
        present = event >= 0
        n_steps = int(np.max(event[present])) + 1
        for step in range(n_steps):
            seen = (event >= 0) & (event <= step)
            n_eligible += ~seen
            if step + 1 < n_steps:
                n_next += event == (step + 1)
    return np.clip(
        (n_next + 1.0) / (n_eligible + 2.0),
        EPS,
        1.0 - EPS,
    )


def _log_sigmoid(value: float) -> float:
    return float(-np.logaddexp(0.0, -value))


def node_control_event_nll(
    event: np.ndarray,
    node_hazard: np.ndarray,
    stop: NodeControlStop,
) -> float:
    values = np.asarray(event, dtype=np.int64)
    hazard = np.asarray(node_hazard, dtype=np.float64)
    present = values >= 0
    n_steps = int(np.max(values[present])) + 1
    terms = []
    for step in range(n_steps):
        seen = (values >= 0) & (values <= step)
        eligible = ~seen
        n_eligible = max(1, int(eligible.sum()))
        stop_logit = stop.c0 + stop.c_n * float(seen.mean())
        if step + 1 == n_steps:
            log_probability = _log_sigmoid(stop_logit)
        else:
            target = values == (step + 1)
            bernoulli = np.sum(
                target[eligible] * np.log(hazard[eligible])
                + (~target[eligible]) * np.log1p(-hazard[eligible])
            )
            log_empty = float(np.sum(np.log1p(-hazard[eligible])))
            log_z = float(
                np.log(-np.expm1(min(log_empty, -np.finfo(float).eps)))
            )
            log_probability = (
                _log_sigmoid(-stop_logit) + float(bernoulli) - log_z
            )
        terms.append(-log_probability / n_eligible)
    return float(np.mean(terms))


def evaluate_node_control(
    *,
    groups: np.ndarray,
    heldout_indices: np.ndarray,
    node_hazard: np.ndarray,
    stop: NodeControlStop,
) -> np.ndarray:
    values = np.asarray(
        [
            node_control_event_nll(groups[index], node_hazard, stop)
            for index in np.asarray(heldout_indices, dtype=np.int64)
        ],
        dtype=np.float64,
    )
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise FloatingPointError("formal node-control NLL is empty or non-finite")
    return values
