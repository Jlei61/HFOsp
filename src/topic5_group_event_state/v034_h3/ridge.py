"""Scale-stable ridge fit and non-estimability diagnostics for H3 arms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RidgeFit:
    selected_lambda: float
    selected_at_grid_edge: bool
    train_mse: float
    validation_mse: float
    intercept_validation_mse: float
    validation_to_intercept_ratio: float
    raw_gram_diagonal_mean: float
    standardised_condition_number: float
    finite: bool
    estimable: bool
    divergence_reasons: tuple[str, ...]
    intercept: tuple[float, ...]
    coefficient: tuple[tuple[float, ...], ...]

    def as_dict(self) -> dict:
        return asdict(self)


def _as_2d_y(y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=np.float64)
    return out[:, None] if out.ndim == 1 else out


def fit_scale_stable_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    lambdas: Sequence[float] = (1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4),
    divergence_factor: float = 4.0,
) -> tuple[np.ndarray, RidgeFit]:
    """TRAIN-standardised ridge with an unpenalised, explicitly fitted intercept.

    Standardising every design column before applying lambda makes selection
    invariant to feature/operator unit changes.  A fitted arm performing worse
    than ``divergence_factor`` times the intercept-only comparator is labelled
    non-estimable rather than a biological negative.
    """

    xt = np.asarray(x_train, dtype=np.float64)
    xv = np.asarray(x_validation, dtype=np.float64)
    yt = _as_2d_y(y_train)
    yv = _as_2d_y(y_validation)
    if xt.ndim != 2 or xv.ndim != 2 or xt.shape[1] != xv.shape[1]:
        raise ValueError("train/validation X must be 2-D with the same columns")
    if yt.shape[0] != xt.shape[0] or yv.shape[0] != xv.shape[0] or yt.shape[1] != yv.shape[1]:
        raise ValueError("X/Y rows and Y output dimensions must align")
    lam = np.asarray(tuple(float(v) for v in lambdas), dtype=np.float64)
    if lam.size == 0 or np.any(lam < 0) or not np.all(np.isfinite(lam)):
        raise ValueError("lambdas must be finite and non-negative")
    x_mean = xt.mean(axis=0)
    x_scale = xt.std(axis=0)
    x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
    zt = (xt - x_mean) / x_scale
    zv = (xv - x_mean) / x_scale
    y_mean = yt.mean(axis=0)
    yc = yt - y_mean
    gram = zt.T @ zt
    rhs = zt.T @ yc
    raw_gram_diag = float(np.mean(np.diag(xt.T @ xt))) if xt.shape[1] else 0.0
    candidates: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    eye = np.eye(xt.shape[1], dtype=np.float64)
    for value in lam:
        try:
            beta_z = np.linalg.solve(gram + float(value) * eye, rhs)
        except np.linalg.LinAlgError:
            beta_z = np.linalg.pinv(gram + float(value) * eye) @ rhs
        pred = (zv @ beta_z) + y_mean
        loss = float(np.mean((yv - pred) ** 2))
        candidates.append((float(value), beta_z, pred, loss))
    losses = np.asarray([v[3] for v in candidates], dtype=np.float64)
    best = int(np.nanargmin(losses)) if np.isfinite(losses).any() else 0
    selected, beta_z, pred, val_mse = candidates[best]
    beta_raw = beta_z / x_scale[:, None]
    intercept = y_mean - x_mean @ beta_raw
    train_pred = xt @ beta_raw + intercept
    train_mse = float(np.mean((yt - train_pred) ** 2))
    intercept_pred = np.repeat(y_mean[None, :], yv.shape[0], axis=0)
    intercept_mse = float(np.mean((yv - intercept_pred) ** 2))
    ratio = float(val_mse / max(intercept_mse, 1e-12))
    finite = bool(np.isfinite(val_mse) and np.all(np.isfinite(beta_raw)) and np.all(np.isfinite(pred)))
    reasons: list[str] = []
    if not finite:
        reasons.append("non_finite_fit")
    if ratio > float(divergence_factor):
        reasons.append(f"validation_mse_over_intercept={ratio:.6g}>{float(divergence_factor):g}")
    if best in (0, len(candidates) - 1):
        reasons.append("ridge_selected_at_grid_edge")
    estimable = finite and ratio <= float(divergence_factor)
    fit = RidgeFit(
        selected_lambda=selected,
        selected_at_grid_edge=best in (0, len(candidates) - 1),
        train_mse=train_mse,
        validation_mse=val_mse,
        intercept_validation_mse=intercept_mse,
        validation_to_intercept_ratio=ratio,
        raw_gram_diagonal_mean=raw_gram_diag,
        standardised_condition_number=float(np.linalg.cond(gram + selected * eye)),
        finite=finite,
        estimable=estimable,
        divergence_reasons=tuple(reasons),
        intercept=tuple(float(v) for v in np.ravel(intercept)),
        coefficient=tuple(tuple(float(v) for v in row) for row in beta_raw),
    )
    if np.asarray(y_validation).ndim == 1:
        pred = pred[:, 0]
    return pred, fit
