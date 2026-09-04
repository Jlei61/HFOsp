"""Optimizer-trace audit shared by future v0.3.4 H3 human runners.

The audit does not tune an optimizer.  It makes no-learning, budget-edge and
explosive fits machine-visible so none can be counted as a scientific zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class OptimizerTraceAudit:
    selected_step: int
    budget_steps: int
    n_effective_updates: int
    selected_at_initialisation: bool
    selected_at_budget_edge: bool
    finite: bool
    validation_to_intercept_ratio: float
    max_relative_update: float
    no_learning: bool
    divergent: bool
    estimable: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def audit_optimizer_trace(
    *,
    steps: np.ndarray,
    inner_validation_loss: np.ndarray,
    intercept_inner_validation_loss: float,
    update_norm: np.ndarray,
    parameter_norm: np.ndarray,
    selected_step: int,
    budget_steps: int,
    divergence_factor: float = 4.0,
    update_epsilon: float = 1e-12,
) -> OptimizerTraceAudit:
    step = np.asarray(steps, dtype=np.int64)
    loss = np.asarray(inner_validation_loss, dtype=np.float64)
    update = np.asarray(update_norm, dtype=np.float64)
    param = np.asarray(parameter_norm, dtype=np.float64)
    if not (step.ndim == loss.ndim == update.ndim == param.ndim == 1):
        raise ValueError("optimizer traces must be one-dimensional")
    if not (step.size == loss.size == update.size == param.size) or step.size == 0:
        raise ValueError("optimizer traces must be non-empty and aligned")
    match = np.flatnonzero(step == int(selected_step))
    if match.size != 1:
        raise ValueError("selected_step must occur exactly once in trace")
    selected_loss = float(loss[int(match[0])])
    ratio = selected_loss / max(float(intercept_inner_validation_loss), 1e-12)
    relative = update / np.maximum(param, 1e-12)
    finite = bool(
        np.all(np.isfinite(loss)) and np.all(np.isfinite(update))
        and np.all(np.isfinite(param)) and np.isfinite(ratio)
    )
    n_updates = int(np.sum(update > float(update_epsilon)))
    no_learning = n_updates == 0 or int(selected_step) == 0
    divergent = (not finite) or ratio > float(divergence_factor)
    reasons: list[str] = []
    if no_learning:
        reasons.append("no_effective_learning_or_initialisation_selected")
    if not finite:
        reasons.append("non_finite_optimizer_trace")
    if ratio > float(divergence_factor):
        reasons.append(f"selected_loss_over_intercept={ratio:.6g}>{float(divergence_factor):g}")
    if int(selected_step) >= int(budget_steps):
        reasons.append("selected_at_budget_edge")
    return OptimizerTraceAudit(
        selected_step=int(selected_step), budget_steps=int(budget_steps),
        n_effective_updates=n_updates, selected_at_initialisation=int(selected_step) == 0,
        selected_at_budget_edge=int(selected_step) >= int(budget_steps), finite=finite,
        validation_to_intercept_ratio=float(ratio),
        max_relative_update=float(np.nanmax(relative)) if relative.size else 0.0,
        no_learning=no_learning, divergent=divergent,
        estimable=bool(not no_learning and not divergent), reasons=tuple(reasons),
    )


def optimizer_scale_equivalent(
    reference_loss: np.ndarray,
    rescaled_loss: np.ndarray,
    reference_relative_update: np.ndarray,
    rescaled_relative_update: np.ndarray,
    *,
    rtol: float = 0.05,
    atol: float = 1e-8,
) -> dict:
    """Compare unit-rescaled canary traces after TRAIN standardisation."""

    a = np.asarray(reference_loss, dtype=np.float64)
    b = np.asarray(rescaled_loss, dtype=np.float64)
    u = np.asarray(reference_relative_update, dtype=np.float64)
    v = np.asarray(rescaled_relative_update, dtype=np.float64)
    if a.shape != b.shape or u.shape != v.shape:
        raise ValueError("reference and rescaled traces must align")
    loss_ok = bool(np.allclose(a, b, rtol=rtol, atol=atol))
    update_ok = bool(np.allclose(u, v, rtol=rtol, atol=atol))
    return {
        "passed": loss_ok and update_ok,
        "loss_trace_equivalent": loss_ok,
        "relative_update_trace_equivalent": update_ok,
        "rtol": float(rtol),
        "atol": float(atol),
    }
