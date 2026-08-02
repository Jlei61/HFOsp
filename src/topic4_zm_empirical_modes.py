"""Empirical axial-mode diagnostics for the Z/M lifecycle sprint.

This module fits a one-step, ridge-regularised DMD operator to a demeaned
24-bin axial activity field.  It is an empirical local propagator, not the
Jacobian of the full SNN.  Its purpose is to distinguish a fixed common-mode
amplitude train from a genuinely phase-staggered axial mode before any new
mechanism is migrated into the simulator.
"""
from __future__ import annotations

import numpy as np


VERSION = "zm_empirical_axial_modes_v1_2026-08-02"


def _participation_ratio(v: np.ndarray) -> float:
    power = np.abs(v) ** 2
    denom = float(np.sum(power ** 2))
    return float(np.sum(power) ** 2 / denom) if denom > 0.0 else 0.0


def _phase_gradient(v: np.ndarray) -> tuple[float | None, float | None]:
    # A real mode only has arbitrary 0/pi sign flips; treating those signs as a
    # propagating phase ramp manufactures a traveling-wave result.
    if np.linalg.norm(np.imag(v)) <= 1e-8 * max(np.linalg.norm(v), np.finfo(float).eps):
        return None, None
    amp = np.abs(v)
    keep = amp >= 0.2 * float(amp.max())
    if np.count_nonzero(keep) < 4:
        return None, None
    x = np.arange(v.size, dtype=float)[keep]
    phase = np.unwrap(np.angle(v[keep]))
    weight = amp[keep] ** 2
    design = np.column_stack((x, np.ones_like(x)))
    wd = design * np.sqrt(weight)[:, None]
    wy = phase * np.sqrt(weight)
    slope, intercept = np.linalg.lstsq(wd, wy, rcond=None)[0]
    fitted = slope * x + intercept
    ss_res = float(np.sum(weight * (phase - fitted) ** 2))
    centre = float(np.sum(weight * phase) / np.sum(weight))
    ss_tot = float(np.sum(weight * (phase - centre) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > np.finfo(float).eps else 0.0
    return float(slope), float(r2)


def _mode_rows(operator: np.ndarray, dt_ms: float) -> list[dict]:
    eigvals, right = np.linalg.eig(operator)
    left_vals, left = np.linalg.eig(operator.T)
    uniform = np.ones(operator.shape[0], dtype=complex)
    uniform /= np.linalg.norm(uniform)
    rows = []
    for index, mu in enumerate(eigvals):
        v = right[:, index].astype(complex)
        v /= np.linalg.norm(v)
        li = int(np.argmin(np.abs(left_vals - mu)))
        w = left[:, li].astype(complex)
        pairing = np.dot(w, v)
        if abs(pairing) > np.finfo(float).eps:
            w /= pairing
        slope, phase_r2 = _phase_gradient(v)
        magnitude = float(abs(mu))
        rows.append({
            "index": int(index),
            "mu_real": float(np.real(mu)),
            "mu_imag": float(np.imag(mu)),
            "magnitude": magnitude,
            "growth_per_s": float(np.log(max(magnitude, np.finfo(float).tiny)) * 1000.0 / dt_ms),
            "frequency_hz": float(abs(np.angle(mu)) * 1000.0 / (2.0 * np.pi * dt_ms)),
            "uniform_overlap": float(abs(np.vdot(uniform, v))),
            "participation_ratio_bins": _participation_ratio(v),
            "phase_gradient_rad_per_bin": slope,
            "phase_gradient_r2": phase_r2,
            "right_real": np.real(v).tolist(),
            "right_imag": np.imag(v).tolist(),
            "left_real": np.real(w).tolist(),
            "left_imag": np.imag(w).tolist(),
        })
    return sorted(rows, key=lambda row: row["growth_per_s"], reverse=True)


def fit_axial_dmd(
    kymo_axial: np.ndarray,
    *,
    dt_ms: float,
    start_ms: float = 0.0,
    end_ms: float | None = None,
    ridge_fraction: float = 1e-4,
    finite_horizon_ms: float = 100.0,
) -> dict:
    """Fit and summarize a demeaned one-step axial DMD operator.

    ``kymo_axial`` is ``[axial_bin, time]``.  The returned path-mode row is the
    least common-mode-dominated oscillatory mode among the five slowest-decay
    modes.  This is an exploratory routing label, not a Hopf claim.
    """
    field = np.asarray(kymo_axial, dtype=float)
    if field.ndim != 2 or field.shape[0] < 4 or not np.isfinite(field).all():
        raise ValueError("kymo_axial must be a finite [bin,time] array")
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        raise ValueError("dt_ms must be positive")
    i0 = max(0, int(np.floor(float(start_ms) / dt_ms)))
    i1 = field.shape[1] if end_ms is None else min(field.shape[1], int(np.ceil(float(end_ms) / dt_ms)))
    segment = field[:, i0:i1].T
    if segment.shape[0] < max(40, 2 * field.shape[0] + 1):
        raise ValueError("DMD window is too short for the axial state dimension")
    mean = segment.mean(axis=0)
    centred = segment - mean
    x, y = centred[:-1], centred[1:]
    split = max(field.shape[0] + 1, int(round(0.7 * x.shape[0])))
    xt, yt = x[:split], y[:split]
    gram = xt.T @ xt
    ridge = float(ridge_fraction) * float(np.trace(gram)) / gram.shape[0]
    operator_train = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), xt.T @ yt).T
    prediction = x[split:] @ operator_train.T
    denom = float(np.linalg.norm(y[split:]))
    heldout_error = float(np.linalg.norm(prediction - y[split:]) / denom) if denom > 0.0 else None
    gram_all = x.T @ x
    ridge_all = float(ridge_fraction) * float(np.trace(gram_all)) / gram_all.shape[0]
    operator = np.linalg.solve(gram_all + ridge_all * np.eye(gram_all.shape[0]), x.T @ y).T
    modes = _mode_rows(operator, float(dt_ms))
    leading = modes[0]
    # Ignore tiny, rapidly decaying numerical directions: their angle can look
    # like an arbitrary high frequency even though they carry no persistent
    # dynamics.  A path candidate must retain at least half its amplitude per
    # sample and live near the slow edge of the fitted spectrum.
    slow_edge = leading["growth_per_s"] - 50.0
    pool = [
        row for row in modes
        if row["frequency_hz"] > 0.5
        and row["magnitude"] >= 0.5
        and row["growth_per_s"] >= slow_edge
    ]
    path = min(pool, key=lambda row: row["uniform_overlap"]) if pool else min(
        modes[: min(8, len(modes))], key=lambda row: row["uniform_overlap"]
    )
    singular = np.linalg.svd(centred, compute_uv=False)
    pc1 = float(singular[0] ** 2 / np.sum(singular ** 2)) if np.any(singular) else 0.0
    steps = max(1, int(round(float(finite_horizon_ms) / float(dt_ms))))
    gain = float(np.linalg.svd(np.linalg.matrix_power(operator, steps), compute_uv=False)[0])
    return {
        "version": VERSION,
        "n_bins": int(field.shape[0]),
        "n_timepoints": int(segment.shape[0]),
        "dt_ms": float(dt_ms),
        "start_ms": float(i0 * dt_ms),
        "end_ms": float(i1 * dt_ms),
        "ridge_fraction": float(ridge_fraction),
        "heldout_relative_error": heldout_error,
        "pc1_fraction": pc1,
        "finite_time_gain_100ms": gain,
        "leading_mode": leading,
        "pathological_mode_candidate": path,
        "modes": modes,
        "operator": operator.tolist(),
        "mean_field": mean.tolist(),
        "claim_boundary": "empirical axial DMD propagator; not a full-SNN Jacobian or Hopf proof",
    }


def project_on_left_mode(kymo_axial: np.ndarray, mode_row: dict, mean_field: np.ndarray) -> np.ndarray:
    field = np.asarray(kymo_axial, dtype=float).T
    mean = np.asarray(mean_field, dtype=float)
    left = np.asarray(mode_row["left_real"], float) + 1j * np.asarray(mode_row["left_imag"], float)
    if field.ndim != 2 or mean.shape != (field.shape[1],) or left.shape != mean.shape:
        raise ValueError("field, mean and left mode do not align")
    return (field - mean) @ left
