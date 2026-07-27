"""Train-only transition-signal decomposition for Topic-5 v0.1.

The module contains no neural network.  It decomposes a regularized
first-order conditional log-hazard residual into local geometry, symmetric,
skew/directed, physical-axis, and history components under one exact
conditional-nonempty set likelihood.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit


EPS = 1.0e-8


@dataclass(frozen=True)
class PairResidual:
    node_hazard: np.ndarray
    node_logit: np.ndarray
    transition_probability: np.ndarray
    residual: np.ndarray
    count: np.ndarray
    exposure: np.ndarray


@dataclass(frozen=True)
class StopParameters:
    c0: float
    c_n: float


def contact_shaft(contact: str) -> str:
    """Return the contact-name prefix before the final integer."""
    match = re.match(r"^(.*?)(\d+)$", str(contact).strip())
    return match.group(1) if match else str(contact).strip()


def logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(value) - np.log1p(-value)


def estimate_node_hazard(
    groups: np.ndarray,
    indices: np.ndarray,
    *,
    pseudocount: float = 1.0,
) -> np.ndarray:
    values = np.asarray(groups, dtype=np.int64)
    n_next = np.zeros(values.shape[1], dtype=np.float64)
    n_eligible = np.zeros(values.shape[1], dtype=np.float64)
    for event in values[np.asarray(indices, dtype=np.int64)]:
        n_steps = int(np.max(event[event >= 0])) + 1
        for step in range(n_steps):
            seen = (event >= 0) & (event <= step)
            n_eligible += ~seen
            if step + 1 < n_steps:
                n_next += event == (step + 1)
    return np.clip(
        (n_next + pseudocount) / (n_eligible + 2.0 * pseudocount),
        EPS,
        1.0 - EPS,
    )


def estimate_pair_residual(
    groups: np.ndarray,
    indices: np.ndarray,
    *,
    concentration: float = 10.0,
) -> PairResidual:
    """Estimate tie-weighted train-only pair transition residual."""
    values = np.asarray(groups, dtype=np.int64)
    n_contacts = values.shape[1]
    node_hazard = estimate_node_hazard(values, indices)
    count = np.zeros((n_contacts, n_contacts), dtype=np.float64)
    exposure = np.zeros_like(count)
    for event in values[np.asarray(indices, dtype=np.int64)]:
        n_steps = int(np.max(event[event >= 0])) + 1
        for step in range(n_steps):
            current = np.flatnonzero(event == step)
            weight = 1.0 / len(current)
            seen = (event >= 0) & (event <= step)
            eligible = np.flatnonzero(~seen)
            exposure[np.ix_(current, eligible)] += weight
            if step + 1 < n_steps:
                following = np.flatnonzero(event == (step + 1))
                count[np.ix_(current, following)] += weight
    transition = (
        count + concentration * node_hazard[None, :]
    ) / (exposure + concentration)
    transition = np.clip(transition, EPS, 1.0 - EPS)
    node_logit = logit(node_hazard)
    residual = logit(transition) - node_logit[None, :]
    np.fill_diagonal(residual, 0.0)
    return PairResidual(
        node_hazard=node_hazard,
        node_logit=node_logit,
        transition_probability=transition,
        residual=residual,
        count=count,
        exposure=exposure,
    )


def symmetric_skew(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(residual, dtype=np.float64)
    return (values + values.T) / 2.0, (values - values.T) / 2.0


def nearest_neighbour_scale(coords: np.ndarray) -> float:
    values = np.asarray(coords, dtype=np.float64)
    distance = np.linalg.norm(values[:, None] - values[None, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    nearest = np.min(np.where(distance > EPS, distance, np.inf), axis=1)
    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        raise ValueError("geometry has no non-zero nearest-neighbour distance")
    return float(np.median(finite))


def geometry_features(
    names: Iterable[str], coords: np.ndarray
) -> dict[str, np.ndarray]:
    names = list(map(str, names))
    values = np.asarray(coords, dtype=np.float64)
    scale = nearest_neighbour_scale(values)
    distance = np.linalg.norm(values[:, None] - values[None, :], axis=-1)
    shafts = np.asarray([contact_shaft(name) for name in names])
    same = (shafts[:, None] == shafts[None, :]).astype(np.float64)
    np.fill_diagonal(same, 0.0)
    local = np.exp(-distance**2 / (2.0 * scale**2))
    np.fill_diagonal(local, 0.0)
    return {
        "same_shaft": same,
        "local_distance": local,
        "distance": distance,
        "scale": np.asarray(scale),
    }


def weighted_ridge_residual(
    target: np.ndarray,
    features: list[np.ndarray],
    exposure: np.ndarray,
    *,
    ridge: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a small weighted ridge projection of the pair residual."""
    target = np.asarray(target, dtype=np.float64)
    n = target.shape[0]
    mask = ~np.eye(n, dtype=bool)
    design = np.column_stack([np.asarray(item)[mask] for item in features])
    response = target[mask]
    weight = np.asarray(exposure, dtype=np.float64)[mask] + 1.0
    root = np.sqrt(weight)
    lhs = (design * root[:, None]).T @ (design * root[:, None])
    lhs += ridge * np.eye(design.shape[1])
    rhs = (design * root[:, None]).T @ (response * root)
    coefficient = np.linalg.solve(lhs, rhs)
    fitted = sum(
        value * feature for value, feature in zip(coefficient, features)
    )
    np.fill_diagonal(fitted, 0.0)
    mse = float(np.average((target[mask] - fitted[mask]) ** 2, weights=weight))
    return fitted, coefficient, mse


def fibonacci_axes(n_directions: int = 32) -> np.ndarray:
    """Deterministic sign-invariant hemisphere directions."""
    if n_directions < 4:
        raise ValueError("at least four directions are required")
    golden = np.pi * (3.0 - np.sqrt(5.0))
    rows = []
    for index in range(n_directions * 2):
        z = 1.0 - (2.0 * index + 1.0) / (n_directions * 2)
        radius = np.sqrt(max(0.0, 1.0 - z * z))
        axis = np.asarray(
            [radius * np.cos(golden * index), radius * np.sin(golden * index), z]
        )
        anchor = int(np.argmax(np.abs(axis)))
        if axis[anchor] < 0:
            axis = -axis
        if axis[2] >= -EPS:
            rows.append(axis / np.linalg.norm(axis))
        if len(rows) == n_directions:
            break
    return np.asarray(rows, dtype=np.float64)


def axis_kernel(
    coords: np.ndarray, axis: np.ndarray, *, ratio: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(coords, dtype=np.float64)
    centered = values - values.mean(axis=0)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    delta = centered[:, None] - centered[None, :]
    distance_sq = np.sum(delta**2, axis=-1)
    scale = nearest_neighbour_scale(centered)
    parallel = np.einsum("ijd,d->ij", delta, axis)
    perpendicular_sq = np.maximum(0.0, distance_sq - parallel**2)
    local = np.exp(-distance_sq / (2.0 * scale**2))
    axial = np.exp(
        -parallel**2 / (2.0 * (ratio * scale) ** 2)
        - perpendicular_sq / (2.0 * scale**2)
    )
    np.fill_diagonal(local, 0.0)
    np.fill_diagonal(axial, 0.0)
    local /= max(EPS, np.linalg.norm(local))
    axial /= max(EPS, np.linalg.norm(axial))
    return local, axial, centered @ axis


def select_axis_residual(
    pair: PairResidual,
    coords: np.ndarray,
    base_features: list[np.ndarray],
    *,
    n_directions: int = 32,
    ratio: float = 2.0,
    ridge: float = 1.0,
) -> dict[str, np.ndarray | float | int]:
    best: dict[str, np.ndarray | float | int] | None = None
    for index, axis in enumerate(fibonacci_axes(n_directions)):
        local, axial, projection = axis_kernel(coords, axis, ratio=ratio)
        excess = axial - local
        fitted, coefficient, mse = weighted_ridge_residual(
            pair.residual,
            [*base_features, excess],
            pair.exposure,
            ridge=ridge,
        )
        if best is None or mse < float(best["train_pair_mse"]):
            best = {
                "axis_index": index,
                "axis": axis,
                "projection": projection,
                "residual": fitted,
                "coefficients": coefficient,
                "train_pair_mse": mse,
                "axis_excess": excess,
                "local_axis_frobenius_cosine": float(
                    np.sum(local * axial)
                    / max(EPS, np.linalg.norm(local) * np.linalg.norm(axial))
                ),
            }
    if best is None:
        raise RuntimeError("axis candidate selection failed")
    return best


def _log_sigmoid(value: float) -> float:
    return float(-np.logaddexp(0.0, -value))


def conditional_nonempty_nll(
    logits: np.ndarray, target: np.ndarray
) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target, dtype=bool)
    log_hazard = -np.logaddexp(0.0, -logits)
    log_one_minus = -np.logaddexp(0.0, logits)
    bernoulli = float(
        np.sum(target * log_hazard + (~target) * log_one_minus)
    )
    log_empty = float(log_one_minus.sum())
    log_z = float(np.log(-np.expm1(min(log_empty, -np.finfo(float).eps))))
    return float(-bernoulli + log_z)


def source_direction_scores(
    groups: np.ndarray,
    indices: np.ndarray,
    projection: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    scores = np.zeros(len(groups), dtype=np.float64)
    for event_index, event in enumerate(groups):
        source = event == 0
        scores[event_index] = float(np.mean(projection[source]))
    train_values = scores[np.asarray(indices, dtype=np.int64)]
    center = float(np.median(train_values))
    q25, q75 = np.quantile(train_values, [0.25, 0.75])
    scale = float(max(EPS, q75 - q25))
    return np.tanh((scores - center) / scale), center, scale


def directional_axis_matrix(
    coords: np.ndarray,
    axis: np.ndarray,
    *,
    ratio: float = 2.0,
) -> np.ndarray:
    _, symmetric, projection = axis_kernel(coords, axis, ratio=ratio)
    scale = max(EPS, float(np.median(np.abs(np.subtract.outer(projection, projection)))))
    direction = np.tanh(
        (projection[None, :] - projection[:, None]) / scale
    )
    matrix = symmetric * direction
    np.fill_diagonal(matrix, 0.0)
    return (matrix - matrix.T) / 2.0


def history_contacts(
    event: np.ndarray,
    step: int,
    mode: str,
    *,
    decay: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "source_only":
        contacts = np.flatnonzero(event == 0)
        return contacts, np.ones(len(contacts))
    if mode == "last_rank":
        contacts = np.flatnonzero(event == step)
        return contacts, np.ones(len(contacts))
    if mode.startswith("last_"):
        depth = int(mode.split("_")[1])
        first = max(0, step - depth + 1)
        contacts = np.flatnonzero((event >= first) & (event <= step))
        return contacts, np.ones(len(contacts))
    contacts = np.flatnonzero((event >= 0) & (event <= step))
    if mode == "unordered_full_prefix":
        return contacts, np.ones(len(contacts))
    if mode == "ordered_full_prefix":
        age = step - event[contacts]
        return contacts, np.power(decay, age.astype(np.float64))
    raise ValueError(f"unknown history mode: {mode}")


def event_nll(
    event: np.ndarray,
    *,
    node_logit: np.ndarray,
    residual: np.ndarray,
    stop: StopParameters,
    history_mode: str = "last_rank",
    history_decay: float = 0.5,
    probability_transition: np.ndarray | None = None,
    source_score: float = 0.0,
    directional_matrix: np.ndarray | None = None,
    directional_beta: float = 0.0,
) -> float:
    n_steps = int(np.max(event[event >= 0])) + 1
    terms = []
    for step in range(n_steps):
        seen = (event >= 0) & (event <= step)
        eligible = ~seen
        n_eligible = max(1, int(eligible.sum()))
        stop_logit = stop.c0 + stop.c_n * float(seen.mean())
        if step + 1 == n_steps:
            total = -_log_sigmoid(stop_logit)
        else:
            contacts, weights = history_contacts(
                event, step, history_mode, decay=history_decay
            )
            weights = weights / weights.sum()
            if probability_transition is not None:
                hazard = np.average(
                    probability_transition[contacts], axis=0, weights=weights
                )
                logits = logit(hazard[eligible])
            else:
                transition_drive = np.average(
                    residual[contacts], axis=0, weights=weights
                )
                logits = node_logit[eligible] + transition_drive[eligible]
            if directional_matrix is not None and directional_beta != 0.0:
                current = np.flatnonzero(event == step)
                directional = np.mean(directional_matrix[current], axis=0)
                logits = (
                    logits
                    + directional_beta
                    * source_score
                    * directional[eligible]
                )
            target = (event == (step + 1))[eligible]
            total = (
                -_log_sigmoid(-stop_logit)
                + conditional_nonempty_nll(logits, target)
            )
        terms.append(total / n_eligible)
    return float(np.mean(terms))


def evaluate_model(
    groups: np.ndarray,
    indices: np.ndarray,
    *,
    node_logit: np.ndarray,
    residual: np.ndarray,
    stop: StopParameters,
    history_mode: str = "last_rank",
    history_decay: float = 0.5,
    probability_transition: np.ndarray | None = None,
    source_scores: np.ndarray | None = None,
    directional_matrix: np.ndarray | None = None,
    directional_beta: float = 0.0,
) -> np.ndarray:
    scores = np.zeros(len(indices), dtype=np.float64)
    for row, event_index in enumerate(np.asarray(indices, dtype=np.int64)):
        scores[row] = event_nll(
            groups[event_index],
            node_logit=node_logit,
            residual=residual,
            stop=stop,
            history_mode=history_mode,
            history_decay=history_decay,
            probability_transition=probability_transition,
            source_score=(
                0.0 if source_scores is None else float(source_scores[event_index])
            ),
            directional_matrix=directional_matrix,
            directional_beta=directional_beta,
        )
    return scores


def fit_directional_beta(
    groups: np.ndarray,
    indices: np.ndarray,
    *,
    node_logit: np.ndarray,
    residual: np.ndarray,
    stop: StopParameters,
    source_scores: np.ndarray,
    directional_matrix: np.ndarray,
    max_events: int = 20_000,
) -> float:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) > max_events:
        positions = np.linspace(0, len(indices) - 1, max_events).astype(int)
        indices = indices[positions]

    def objective(beta: float) -> float:
        return float(
            np.mean(
                evaluate_model(
                    groups,
                    indices,
                    node_logit=node_logit,
                    residual=residual,
                    stop=stop,
                    source_scores=source_scores,
                    directional_matrix=directional_matrix,
                    directional_beta=beta,
                )
            )
        )

    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=(-4.0, 4.0),
        options={"xatol": 1.0e-3, "maxiter": 40},
    )
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError("source-conditioned directional beta fit failed")
    return float(result.x)


def positive_contact_nll_by_shaft(
    groups: np.ndarray,
    indices: np.ndarray,
    *,
    names: Iterable[str],
    node_logit: np.ndarray,
    residual: np.ndarray,
    source_scores: np.ndarray | None = None,
    directional_matrix: np.ndarray | None = None,
    directional_beta: float = 0.0,
) -> dict[str, float | int]:
    """Descriptive target-contact log loss split by current-set shaft relation."""
    shafts = np.asarray([contact_shaft(name) for name in names])
    same_values: list[float] = []
    cross_values: list[float] = []
    for event_index in np.asarray(indices, dtype=np.int64):
        event = groups[event_index]
        n_steps = int(np.max(event[event >= 0])) + 1
        for step in range(n_steps - 1):
            current = np.flatnonzero(event == step)
            target = np.flatnonzero(event == (step + 1))
            drive = np.mean(residual[current], axis=0)
            logits = node_logit + drive
            if directional_matrix is not None and directional_beta != 0.0:
                directional = np.mean(directional_matrix[current], axis=0)
                logits = (
                    logits
                    + directional_beta
                    * float(source_scores[event_index])
                    * directional
                )
            for contact in target:
                value = float(np.logaddexp(0.0, -logits[contact]))
                if np.any(shafts[current] == shafts[contact]):
                    same_values.append(value)
                else:
                    cross_values.append(value)
    return {
        "same_shaft_positive_nll": (
            float(np.mean(same_values)) if same_values else float("nan")
        ),
        "cross_shaft_positive_nll": (
            float(np.mean(cross_values)) if cross_values else float("nan")
        ),
        "n_same_shaft_positive_contacts": len(same_values),
        "n_cross_shaft_positive_contacts": len(cross_values),
    }


def cross_shaft_conditional_nll(
    groups: np.ndarray,
    indices: np.ndarray,
    *,
    names: Iterable[str],
    node_logit: np.ndarray,
    residual: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Event-first NLL for prefixes with an observed cross-shaft next contact.

    The eligible set contains unseen contacts whose shaft is absent from the
    current rank set.  Positive and negative eligible contacts both enter the
    conditional-nonempty likelihood.  Prefixes without a cross-shaft target
    are outside this explicitly conditional endpoint.
    """
    shafts = np.asarray([contact_shaft(name) for name in names])
    event_scores: list[float] = []
    n_prefixes = 0
    for event_index in np.asarray(indices, dtype=np.int64):
        event = groups[event_index]
        n_steps = int(np.max(event[event >= 0])) + 1
        prefix_scores: list[float] = []
        for step in range(n_steps - 1):
            current = np.flatnonzero(event == step)
            seen = (event >= 0) & (event <= step)
            current_shafts = np.unique(shafts[current])
            eligible = (~seen) & (~np.isin(shafts, current_shafts))
            target = event == (step + 1)
            if not np.any(target & eligible):
                continue
            drive = np.mean(residual[current], axis=0)
            logits = node_logit[eligible] + drive[eligible]
            prefix_scores.append(
                conditional_nonempty_nll(logits, target[eligible])
                / max(1, int(eligible.sum()))
            )
            n_prefixes += 1
        if prefix_scores:
            event_scores.append(float(np.mean(prefix_scores)))
    return np.asarray(event_scores, dtype=np.float64), n_prefixes


def choose_history_decay(
    groups: np.ndarray,
    train_indices: np.ndarray,
    stop: StopParameters,
    *,
    candidates: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> tuple[float, dict[float, float]]:
    train = np.asarray(train_indices, dtype=np.int64)
    boundary = int(np.floor(0.75 * len(train)))
    fit = train[:boundary]
    validation = train[boundary:]
    pair = estimate_pair_residual(groups, fit)
    scores = {}
    for decay in candidates:
        scores[decay] = float(
            np.mean(
                evaluate_model(
                    groups,
                    validation,
                    node_logit=pair.node_logit,
                    residual=pair.residual,
                    stop=stop,
                    history_mode="ordered_full_prefix",
                    history_decay=decay,
                )
            )
        )
    selected = min(candidates, key=lambda value: (scores[value], value))
    return float(selected), scores
