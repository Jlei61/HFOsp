"""Exact-window long event innovation and affine frozen-state edge tools."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .t2_r2 import (
    evaluate_r2_edge,
)
from .t2_s1 import OneStepDesign, _evaluate_rows, _score


H3_LONG_REVISION = "h3_long_exact_boxcar_affine_edge_v2"
H3_LONG_SUPPORT_REVISION = "r1_5_h3_long_exact_recorded_segment_support_v2"
SCALES = (1000, 3000, 10000)
SOURCES = ("load", "participation")
SYNTHETIC_TRUTHS = (
    "positive", "zero", "reversed", "constant", "observed_drift",
    "unobserved_drift",
)


def independent_endpoint_rows(
    current_index: np.ndarray,
    event_segment: np.ndarray,
    *,
    width_events: int,
) -> np.ndarray:
    """Greedily retain endpoints whose complete exposure units do not overlap."""
    current = np.asarray(current_index, dtype=np.int64)
    segment = np.asarray(event_segment, dtype=np.int64)
    if current.ndim != 1 or np.any(current < 0) or np.any(current >= len(segment)):
        raise ValueError("invalid independent endpoint input")
    keep: list[int] = []
    width = int(width_events)
    for label in np.unique(segment[current]):
        rows = np.flatnonzero(segment[current] == label)
        order = rows[np.argsort(current[rows])]
        last = -10**18
        for row in order:
            endpoint = int(current[row])
            if endpoint - last >= width:
                keep.append(int(row))
                last = endpoint
    return np.asarray(sorted(keep), dtype=np.int64)


def classify_affine_estimability(audit: dict, fit: dict) -> str:
    """Keep numerical failures separate from a genuine zero-selected edge."""
    if not audit.get("gradient_finite", False):
        return "NONFINITE_GRADIENT"
    if audit.get("affine_design_rank") != audit.get("expected_affine_rank"):
        return "RANK_DEGENERATE"
    if min(audit.get("exposure_sd", [0.0])) <= 1e-8:
        return "EXPOSURE_DEGENERATE"
    if audit.get("matrix_gradient_at_zero_norm", 0.0) <= 1e-8:
        return "ZERO_GRADIENT"
    if not fit.get("edge_left_zero_initialisation", False):
        return "ZERO_SELECTED"
    return "ESTIMABLE"


def state_matching_estimable(audit: dict) -> bool:
    """Reject only collapsed placebo matching, not ordinary imperfect matches."""
    distance = audit.get("match_distance_q95")
    return bool(
        audit.get("matched", 0) > 0
        and audit.get("unique_donors", 0) >= 10
        and audit.get("effective_donors", 0.0) >= 8.0
        and audit.get("maximum_donor_reuse_fraction", 1.0) <= 0.25
        and distance is not None
        and np.isfinite(float(distance))
    )


def exact_boxcar_event_exposure(
    innovation: np.ndarray,
    segment: np.ndarray,
    *,
    scale_events: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sum exactly the current and previous N-1 innovations within a segment."""
    value = np.asarray(innovation, dtype=np.float64)
    scalar = value.ndim == 1
    if scalar:
        value = value[:, None]
    segment = np.asarray(segment, dtype=np.int64)
    n = int(scale_events)
    if value.ndim != 2 or len(value) != len(segment) or n < 1:
        raise ValueError("invalid exact-window exposure input")
    output = np.zeros_like(value, dtype=np.float64)
    eligible = np.zeros(len(value), dtype=bool)
    rows = []
    for label in np.unique(segment):
        index = np.flatnonzero(segment == label)
        local = value[index]
        cumulative = np.vstack([
            np.zeros((1, local.shape[1]), dtype=np.float64),
            np.cumsum(local, axis=0),
        ])
        if len(index) >= n:
            position = np.arange(n - 1, len(index), dtype=np.int64)
            output[index[position]] = (
                cumulative[position + 1] - cumulative[position + 1 - n]
            )
            eligible[index[position]] = True
        rows.append({
            "segment": int(label), "events": int(len(index)),
            "eligible": int(max(0, len(index) - n + 1)),
        })
    result = output[:, 0] if scalar else output
    return result.astype(np.float32), eligible, {
        "revision": H3_LONG_REVISION,
        "scale_events": n,
        "window_kind": "exact_last_n_events_including_current",
        "resets_at_recorded_segment": True,
        "exponential_tail": False,
        "segment_rows": rows,
    }


def exact_previous_block_placebo(
    exposure: np.ndarray,
    segment: np.ndarray,
    *,
    scale_events: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Use the immediately preceding disjoint N-event block as a causal placebo."""
    exposure = np.asarray(exposure, dtype=np.float64)
    segment = np.asarray(segment, dtype=np.int64)
    n = int(scale_events)
    output = np.zeros_like(exposure, dtype=np.float64)
    eligible = np.zeros(len(exposure), dtype=bool)
    rows = []
    for label in np.unique(segment):
        index = np.flatnonzero(segment == label)
        # Real at position p covers [p-N+1,p].  Exposure at p-N covers
        # [p-2N+1,p-N], so the two windows are exactly disjoint.
        if len(index) >= 2 * n:
            position = np.arange(2 * n - 1, len(index), dtype=np.int64)
            output[index[position]] = exposure[index[position - n]]
            eligible[index[position]] = True
        rows.append({
            "segment": int(label), "events": int(len(index)),
            "eligible": int(max(0, len(index) - 2 * n + 1)),
        })
    return output.astype(np.float32), eligible, {
        "scale_events": n,
        "lag_events": n,
        "real_and_placebo_windows_exactly_disjoint": True,
        "strictly_past": True,
        "resets_at_recorded_segment": True,
        "segment_rows": rows,
    }


def standardise_exposure_on_train(
    exposure: np.ndarray,
    train_mask: np.ndarray,
    eligible: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Centre and scale one arm using only its eligible TRAIN values."""
    value = np.asarray(exposure, dtype=np.float64)
    scalar = value.ndim == 1
    if scalar:
        value = value[:, None]
    train = np.asarray(train_mask, dtype=bool) & np.asarray(eligible, dtype=bool)
    if not np.any(train):
        raise ValueError("long exposure has no eligible TRAIN values")
    centre = value[train].mean(0)
    scale = value[train].std(0)
    if not np.isfinite(scale).all() or np.any(scale <= 1e-8):
        raise ValueError("long exposure is degenerate on TRAIN")
    output = (value - centre) / scale
    output[~np.asarray(eligible, dtype=bool)] = 0.0
    result = output[:, 0] if scalar else output
    return result.astype(np.float32), {
        "train_centre": centre.tolist(), "train_scale": scale.tolist(),
        "validation_statistics_used": False,
    }


def chronological_trend_exposure(
    event_time: np.ndarray,
    segment: np.ndarray,
    exposure_dim: int,
) -> np.ndarray:
    """Equal-dimensional clock control for omitted monotone slow drift."""
    time = np.asarray(event_time, dtype=np.float64)
    segment = np.asarray(segment, dtype=np.int64)
    dim = int(exposure_dim)
    if time.ndim != 1 or len(time) != len(segment) or dim < 1:
        raise ValueError("invalid chronological trend control")
    elapsed_days = (time - float(np.min(time))) / 86400.0
    columns = [np.power(elapsed_days, order) for order in range(1, dim + 1)]
    return np.column_stack(columns[:dim]).astype(np.float32)


class AffineExposureEdge(nn.Module):
    """Equal-budget fitted intercept plus signed exposure-to-state mapping."""

    def __init__(self, state_dim: int, exposure_dim: int = 1):
        super().__init__()
        self.intercept = nn.Parameter(torch.zeros(int(state_dim)))
        self.matrix = nn.Parameter(torch.zeros(int(exposure_dim), int(state_dim)))

    def forward(self, state: torch.Tensor, exposure: torch.Tensor) -> torch.Tensor:
        if exposure.ndim == 1:
            exposure = exposure.unsqueeze(-1)
        return state + self.intercept.unsqueeze(0) + exposure @ self.matrix


def affine_estimability_audit(
    model,
    design: OneStepDesign,
    *,
    device: torch.device | str,
    batch_size: int = 4096,
) -> dict:
    train = np.flatnonzero(design.split == 0)
    if not len(train):
        raise ValueError("long edge audit has no TRAIN pairs")
    exposure = np.asarray(design.exposure[train], dtype=np.float64)
    matrix = exposure[:, None] if exposure.ndim == 1 else exposure
    edge = AffineExposureEdge(
        design.current_state.shape[1], matrix.shape[1]
    ).to(device)
    total = len(train)
    for lo in range(0, total, int(batch_size)):
        rows = train[lo:lo + int(batch_size)]
        loss, _ = _score(
            model, edge, design, rows, device=device, require_grad=True
        )
        (loss * (len(rows) / total)).backward()
    matrix_gradient = edge.matrix.grad.detach().cpu().numpy()
    intercept_gradient = edge.intercept.grad.detach().cpu().numpy()
    affine_design = np.column_stack([np.ones(len(matrix)), matrix])
    return {
        "train_pairs": int(total),
        "exposure_dim": int(matrix.shape[1]),
        "exposure_rank": int(np.linalg.matrix_rank(matrix)),
        "affine_design_rank": int(np.linalg.matrix_rank(affine_design)),
        "expected_affine_rank": int(matrix.shape[1] + 1),
        "exposure_mean": matrix.mean(0).tolist(),
        "exposure_sd": matrix.std(0).tolist(),
        "matrix_gradient_at_zero_norm": float(np.linalg.norm(matrix_gradient)),
        "intercept_gradient_at_zero_norm": float(
            np.linalg.norm(intercept_gradient)
        ),
        "gradient_finite": bool(
            np.isfinite(matrix_gradient).all()
            and np.isfinite(intercept_gradient).all()
        ),
    }


def fit_affine_edge(
    model,
    design: OneStepDesign,
    *,
    device: torch.device | str,
    seed: int = 0,
    epochs: int = 30,
    learning_rate: float = 2e-2,
    batch_size: int = 4096,
) -> tuple[AffineExposureEdge, dict]:
    """Chronologically select and all-TRAIN refit an equal-budget affine edge."""
    torch.manual_seed(int(seed))
    train = np.flatnonzero(design.split == 0)
    if len(train) < 100:
        raise ValueError("H3-long needs at least 100 TRAIN pairs")
    cut = int(np.clip(math.floor(0.8 * len(train)), 1, len(train) - 1))
    inner_train, inner_validation = train[:cut], train[cut:]
    exposure_dim = 1 if design.exposure.ndim == 1 else design.exposure.shape[1]

    def fresh() -> AffineExposureEdge:
        return AffineExposureEdge(
            design.current_state.shape[1], exposure_dim
        ).to(device)

    base = _evaluate_rows(
        model, fresh(), design, inner_validation,
        device=device, batch_size=batch_size,
    )
    best_epoch = 0
    best_value = float(base.joint_nll_per_event)
    edge = fresh()
    optimizer = torch.optim.AdamW(
        edge.parameters(), lr=float(learning_rate), weight_decay=1e-3
    )
    trajectory = [{"epoch": 0, "joint_nll": best_value}]
    rng = np.random.default_rng(int(seed))
    for epoch in range(1, int(epochs) + 1):
        order = rng.permutation(inner_train)
        for lo in range(0, len(order), int(batch_size)):
            rows = order[lo:lo + int(batch_size)]
            loss, _ = _score(
                model, edge, design, rows, device=device, require_grad=True
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(edge.parameters(), 1.0)
            optimizer.step()
        if epoch == 1 or epoch % 2 == 0:
            metric = _evaluate_rows(
                model, edge, design, inner_validation,
                device=device, batch_size=batch_size,
            ).joint_nll_per_event
            trajectory.append({"epoch": int(epoch), "joint_nll": float(metric)})
            if metric < best_value:
                best_value = float(metric)
                best_epoch = int(epoch)
    edge = fresh()
    if best_epoch:
        optimizer = torch.optim.AdamW(
            edge.parameters(), lr=float(learning_rate), weight_decay=1e-3
        )
        rng = np.random.default_rng(int(seed))
        for _ in range(best_epoch):
            order = rng.permutation(train)
            for lo in range(0, len(order), int(batch_size)):
                rows = order[lo:lo + int(batch_size)]
                loss, _ = _score(
                    model, edge, design, rows, device=device, require_grad=True
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(edge.parameters(), 1.0)
                optimizer.step()
    matrix_norm = float(torch.linalg.vector_norm(edge.matrix).detach().cpu())
    intercept_norm = float(
        torch.linalg.vector_norm(edge.intercept).detach().cpu()
    )
    return edge.eval(), {
        "selected_epoch": int(best_epoch),
        "inner_validation_joint_nll": best_value,
        "trajectory": trajectory,
        "matrix_norm": matrix_norm,
        "intercept_norm": intercept_norm,
        "edge_left_zero_initialisation": bool(matrix_norm > 1e-8),
        "intercept_left_zero_initialisation": bool(intercept_norm > 1e-8),
    }


def evaluate_affine_edge(model, edge: AffineExposureEdge,
                         design: OneStepDesign, *, split: str,
                         device: torch.device | str,
                         batch_size: int = 4096):
    return evaluate_r2_edge(
        model, edge, design, split=split, device=device,
        batch_size=batch_size,
    )
