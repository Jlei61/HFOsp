"""Numerical helpers for Topic 5.2 Pass 1 system identification.

This module is target-free.  It operates on frozen interictal event ranks and
teacher-forced hidden states only; it has no SNN or early-ictal reader.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping

import numpy as np
import torch

from src.topic5_latent_landscape_v0_2 import rank_matrix_to_event_fields


PHASE_BINS = 5
SPLINE_KNOT_SETS = ((), (0.5,), (1.0 / 3.0, 2.0 / 3.0))
RIDGE_GRID = (1e-6, 1e-4, 1e-2, 1.0)
EMERGENCE_RIDGE_GRID = (1e-4, 1e-2, 1.0)


@dataclass(frozen=True)
class FutureFieldData:
    axis: np.ndarray
    train_mean_field: np.ndarray
    event_coordinate: np.ndarray
    event_coordinate_z: np.ndarray
    event_coordinate_shuffled_z: np.ndarray
    train_coordinate_mean: float
    train_coordinate_scale: float
    positive_mode: int
    negative_mode: int
    tier: str
    n_common_contacts: int
    contrast_norm: float


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def build_future_field_data(
    ranks: np.ndarray,
    split: np.ndarray,
    full_train_mode: np.ndarray,
    *,
    positive_mode: int,
    negative_mode: int,
    tier: str,
    shuffle_key: str,
) -> FutureFieldData:
    """Freeze a start-removed train field axis and event coordinates."""
    _, recurrence = rank_matrix_to_event_fields(ranks)
    split = np.asarray(split)
    labels = np.asarray(full_train_mode)
    train = split == 0
    means: dict[int, np.ndarray] = {}
    for mode in (int(positive_mode), int(negative_mode)):
        use = train & (labels == mode)
        if not np.any(use):
            raise ValueError(f"axis-train mode {mode} is missing")
        with np.errstate(invalid="ignore"):
            means[mode] = np.nanmean(recurrence[use], axis=0)
    contrast = means[int(positive_mode)] - means[int(negative_mode)]
    common = np.isfinite(contrast)
    if int(common.sum()) < 2:
        raise ValueError("future-field contrast has fewer than two common contacts")
    centered = contrast[common] - float(np.mean(contrast[common]))
    norm = float(np.linalg.norm(centered))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps * int(common.sum()):
        raise ValueError("future-field contrast is numerically degenerate")
    axis = np.zeros(recurrence.shape[1], dtype=np.float64)
    axis[common] = centered / norm
    with np.errstate(invalid="ignore"):
        train_mean = np.nanmean(recurrence[train], axis=0)
    train_mean = np.where(np.isfinite(train_mean), train_mean, 0.0)
    filled = np.where(np.isfinite(recurrence), recurrence, train_mean[None, :])
    coordinate = (filled - train_mean[None, :]) @ axis
    mean = float(np.mean(coordinate[train]))
    scale = float(np.std(coordinate[train], ddof=0))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("future-field coordinate has degenerate train variance")
    coordinate_z = (coordinate - mean) / scale
    shuffled = coordinate_z.copy()
    for split_id in (0, 1, 2):
        indices = np.flatnonzero(split == split_id)
        rng = np.random.default_rng(_stable_seed(shuffle_key, split_id))
        shuffled[indices] = coordinate_z[rng.permutation(indices)]
    return FutureFieldData(
        axis=axis,
        train_mean_field=train_mean,
        event_coordinate=coordinate,
        event_coordinate_z=coordinate_z,
        event_coordinate_shuffled_z=shuffled,
        train_coordinate_mean=mean,
        train_coordinate_scale=scale,
        positive_mode=int(positive_mode),
        negative_mode=int(negative_mode),
        tier=str(tier),
        n_common_contacts=int(common.sum()),
        contrast_norm=norm,
    )


@torch.no_grad()
def teacher_forced_hidden(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return hidden states after each teacher-forced input rank set."""
    if x.ndim != 3:
        raise ValueError("x must be batch x step x contact")
    h = torch.zeros(x.shape[0], model.n_nodes * model.state_dim, device=x.device)
    states: list[torch.Tensor] = []
    for step in range(x.shape[1]):
        h = model._step(h, x[:, step])
        states.append(h)
    return torch.stack(states, dim=1)


def phase_bin(phase: np.ndarray, n_bins: int = PHASE_BINS) -> np.ndarray:
    phase = np.asarray(phase, dtype=float)
    return np.minimum((phase * n_bins).astype(int), n_bins - 1)


def event_first_phase_balanced_weights(
    event_index: np.ndarray,
    split: np.ndarray,
    phase_bins: np.ndarray,
    n_bins: int = PHASE_BINS,
) -> np.ndarray:
    """Give equal mass to phase bins and equal event mass within each bin."""
    event_index = np.asarray(event_index)
    split = np.asarray(split)
    phase_bins = np.asarray(phase_bins)
    if not (event_index.shape == split.shape == phase_bins.shape):
        raise ValueError("state metadata arrays must align")
    weights = np.zeros(len(event_index), dtype=np.float64)
    for split_id in np.unique(split):
        split_mask = split == split_id
        nonempty = [
            b for b in range(n_bins) if np.any(split_mask & (phase_bins == b))
        ]
        for b in nonempty:
            use = np.flatnonzero(split_mask & (phase_bins == b))
            events = np.unique(event_index[use])
            for event in events:
                positions = use[event_index[use] == event]
                weights[positions] = 1.0 / (
                    len(nonempty) * len(events) * len(positions)
                )
        total = float(weights[split_mask].sum())
        if not np.isclose(total, 1.0, atol=1e-10):
            raise RuntimeError(f"phase-balanced split weights sum to {total}")
    return weights


def robust_center_scale(train_hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(train_hidden, dtype=np.float64)
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center[None, :]), axis=0)
    scale = 1.4826 * mad
    standard = np.std(values, axis=0, ddof=0)
    fallback = (~np.isfinite(scale)) | (scale <= 1e-6)
    scale[fallback] = standard[fallback]
    constant = (~np.isfinite(scale)) | (scale <= 1e-6)
    scale[constant] = 1.0
    return center, scale, constant


def spline_basis(phase: np.ndarray, knots: Iterable[float] = ()) -> np.ndarray:
    s = np.asarray(phase, dtype=np.float64).reshape(-1)
    columns = [np.ones_like(s), s, s**2, s**3]
    columns.extend(np.maximum(s - float(knot), 0.0) ** 3 for knot in knots)
    return np.column_stack(columns)


def spline_derivative(phase: np.ndarray, knots: Iterable[float] = ()) -> np.ndarray:
    s = np.asarray(phase, dtype=np.float64).reshape(-1)
    columns = [np.zeros_like(s), np.ones_like(s), 2.0 * s, 3.0 * s**2]
    columns.extend(3.0 * np.maximum(s - float(knot), 0.0) ** 2 for knot in knots)
    return np.column_stack(columns)


def weighted_ridge(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    *,
    penalize_intercept: bool = False,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    gram = x.T @ (w[:, None] * x)
    cross = x.T @ (w[:, None] * y)
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    if not penalize_intercept and penalty.size:
        penalty[0, 0] = 0.0
    try:
        return np.linalg.solve(gram + penalty, cross)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram + penalty, rcond=1e-10) @ cross


def weighted_r2(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mean = np.sum(w[:, None] * y, axis=0) / max(float(w.sum()), 1e-12)
    residual = float(np.sum(w[:, None] * (y - prediction) ** 2))
    total = float(np.sum(w[:, None] * (y - mean[None, :]) ** 2))
    return float(1.0 - residual / total) if total > 1e-12 else float("nan")


def weighted_r2_scalar(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    mean = float(np.sum(w * y) / max(float(w.sum()), 1e-12))
    residual = float(np.sum(w * (y - prediction) ** 2))
    total = float(np.sum(w * (y - mean) ** 2))
    return float(1.0 - residual / total) if total > 1e-12 else float("nan")


def weighted_pca(
    y: np.ndarray, weights: np.ndarray, max_components: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mean = np.sum(w[:, None] * y, axis=0) / max(float(w.sum()), 1e-12)
    centered = y - mean[None, :]
    covariance = centered.T @ (w[:, None] * centered) / max(float(w.sum()), 1e-12)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    take = order[: min(int(max_components), y.shape[1])]
    return mean, np.maximum(eigenvalues[take], 0.0), eigenvectors[:, take]


def observable_design(
    step: np.ndarray,
    n_contacts: int,
    current_x: np.ndarray,
    recruited: np.ndarray,
) -> np.ndarray:
    step = np.asarray(step, dtype=float)
    x = np.asarray(current_x, dtype=float)
    r = np.asarray(recruited, dtype=float)
    return np.column_stack([
        np.ones(len(step)),
        step / max(int(n_contacts) - 1, 1),
        r.mean(axis=1),
        x.sum(axis=1),
        x,
        r,
    ])


def orthogonalize_field_axis(progress: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, bool]:
    progress = np.asarray(progress, dtype=float)
    field = np.asarray(field, dtype=float)
    pnorm = float(np.linalg.norm(progress))
    if pnorm <= 1e-12:
        return np.full_like(field, np.nan), True
    p = progress / pnorm
    residual = field - p * float(np.dot(p, field))
    norm = float(np.linalg.norm(residual))
    if norm <= 1e-8 * max(float(np.linalg.norm(field)), 1.0):
        return np.full_like(field, np.nan), True
    return residual / norm, False


def leaky_rnn_jvp(
    model: torch.nn.Module,
    h: torch.Tensor,
    x_next: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    """Exact Jacobian-vector products for the frozen leaky tanh RNN cell.

    ``vectors`` may be ``(batch, hidden)`` or ``(batch, direction, hidden)``.
    This avoids materialising an N-by-N Jacobian and is mathematically identical
    to differentiating ``model._step(h, x_next)`` with respect to ``h``.
    """
    if getattr(model, "cell", None) != "rnn" or int(model.state_dim) != 1:
        raise ValueError("analytic JVP is restricted to the frozen state_dim=1 RNN contract")
    if h.ndim != 2 or x_next.ndim != 2 or h.shape[0] != x_next.shape[0]:
        raise ValueError("h and x_next must be aligned batch matrices")
    squeeze = vectors.ndim == 2
    v = vectors[:, None, :] if squeeze else vectors
    if v.ndim != 3 or v.shape[0] != h.shape[0] or v.shape[2] != h.shape[1]:
        raise ValueError("vectors must align to batch and hidden dimensions")
    recurrent = model.masked_recurrent()[0]
    injected = model._inject(x_next).reshape(h.shape[0], -1, h.shape[1])[:, 0]
    pre = injected + h @ recurrent.T + model.bias[0]
    derivative = 1.0 - torch.tanh(pre) ** 2
    kappa = torch.sigmoid(model.kappa_logit)
    transported = torch.matmul(v, recurrent.T)
    result = (1.0 - kappa) * v + kappa * derivative[:, None, :] * transported
    return result[:, 0] if squeeze else result


def interpolate_phase_vectors(
    phase_grid: np.ndarray, vectors: np.ndarray, phases: np.ndarray
) -> np.ndarray:
    grid = np.asarray(phase_grid, dtype=float)
    values = np.asarray(vectors, dtype=float)
    query = np.asarray(phases, dtype=float).reshape(-1)
    if values.shape[0] != len(grid):
        raise ValueError("phase grid and vector rows must align")
    output = np.empty((len(query), values.shape[1]), dtype=float)
    for column in range(values.shape[1]):
        output[:, column] = np.interp(query, grid, values[:, column])
    return output
