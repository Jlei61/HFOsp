"""Conditional rich-mark readout for v0.3.7 H2a.

The frozen contact decoder supplies a representation after the first two tied
groups.  A capacity-matched ridge readout then asks whether the pre-event
observer context improves prediction of the remaining whole-event rich mark.
All transforms and ridge choices are fitted on FIT and selected on INNER;
SELECTION is report-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from src.topic5_group_event_state.v035.stepwise_decoder import StepwiseConditionedDecoder
from src.topic5_group_event_state.v034_spatial_state.we_decoder import event_batch


RIDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def leaked_event_context(ranks: np.ndarray) -> np.ndarray:
    """Explicitly non-causal context used only as an assay-sensitivity control."""

    value = np.asarray(ranks, dtype=np.int16)
    participating = value >= 0
    n_groups = np.maximum(1, value.max(axis=1) + 1)
    rank_scaled = np.where(participating, (value + 1) / n_groups[:, None], 0.0)
    size = participating.mean(axis=1, keepdims=True)
    extent = n_groups[:, None] / max(1, value.shape[1])
    return np.concatenate((participating.astype(np.float32), rank_scaled.astype(np.float32), size, extent), axis=1)


def prefix_hidden(
    model: StepwiseConditionedDecoder,
    tensors: dict[str, Tensor],
    context: Tensor | None,
    rows: np.ndarray,
    *,
    use_dynamic: bool,
    batch_size: int,
    prefix_groups: int = 2,
) -> np.ndarray:
    """Frozen decoder representation after a causal observed event prefix."""

    parts: list[np.ndarray] = []
    step = int(prefix_groups) - 1
    if step < 0:
        raise ValueError("prefix_groups must be positive")
    with torch.no_grad():
        for start in range(0, int(rows.size), int(batch_size)):
            index = torch.as_tensor(rows[start:start + batch_size], dtype=torch.long, device=tensors["x"].device)
            batch = event_batch(tensors, index)
            state = None if context is None else context[index]
            hidden = model.hidden_sequence(
                batch["x"], batch["recruited"], batch["valid"], state,
                use_static=True, use_dynamic=use_dynamic,
            )
            if hidden.shape[1] <= step:
                raise ValueError("decoder sequence shorter than requested prefix")
            parts.append(hidden[:, step].detach().cpu().numpy().astype(np.float64))
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 0), dtype=np.float64)


@dataclass(frozen=True)
class RidgeMarkModel:
    centre: np.ndarray
    scale: np.ndarray
    beta: np.ndarray
    ridge: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = np.clip((np.asarray(x, dtype=np.float64) - self.centre) / self.scale, -12.0, 12.0)
        design = np.concatenate((np.ones((xs.shape[0], 1)), xs), axis=1)
        return design @ self.beta


def _fit_at_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> RidgeMarkModel:
    centre = np.median(x, axis=0)
    scale = 1.4826 * np.median(np.abs(x - centre), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    xs = np.clip((x - centre) / scale, -12.0, 12.0)
    design = np.concatenate((np.ones((xs.shape[0], 1)), xs), axis=1)
    gram = design.T @ design
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge) * max(1.0, float(xs.shape[0]))
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(gram + penalty, design.T @ y)
    return RidgeMarkModel(centre=centre, scale=scale, beta=beta, ridge=float(ridge))


def mean_squared_mark(model: RidgeMarkModel, x: np.ndarray, y: np.ndarray) -> float:
    residual = model.predict(x) - np.asarray(y, dtype=np.float64)
    return float(np.mean(residual * residual))


def fit_rich_mark_readout(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_inner: np.ndarray,
    y_inner: np.ndarray,
) -> tuple[RidgeMarkModel, dict[str, object]]:
    """Choose regularisation on INNER without ever refitting on INNER."""

    candidates = []
    best: RidgeMarkModel | None = None
    best_loss = float("inf")
    for ridge in RIDGES:
        model = _fit_at_ridge(np.asarray(x_fit, dtype=np.float64), np.asarray(y_fit, dtype=np.float64), ridge)
        loss = mean_squared_mark(model, x_inner, y_inner)
        candidates.append({"ridge": float(ridge), "inner_mse": float(loss)})
        if np.isfinite(loss) and loss < best_loss:
            best, best_loss = model, loss
    if best is None:
        raise RuntimeError("no finite rich-mark ridge candidate")
    return best, {"selected_ridge": best.ridge, "best_inner_mse": best_loss, "candidates": candidates}

