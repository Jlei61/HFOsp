"""Cheap entry/exit geometry for the current-based MZ lifecycle line.

This module does not introduce a new rate model.  It reuses the locked Stage-0C
nine-state E/I + synapse + delayed-pool system and adds one *frozen* additive
E-cell recovery current ``A`` (mV) for counterfactual analysis::

    mu_E -> mu_E - A

The frozen current is an oracle for the existing ``eta_m * m`` mechanism.  It
lets us ask whether additive recovery moves a fixed-point fold and whether a
state on the established fast cycle returns to the low basin.  It is not itself
a slow-variable implementation and a state fork is not a periodic-orbit
continuation certificate.

Rates are kHz, time is ms, and all membrane moments are mV, matching Stage 0C.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from src.sef_hfo_lif import TAU_AMPA, TAU_GABA, TAU_ME, TAU_MI
from src.topic4_spatial_slowfast_stage0c import (
    S_MAX,
    TAU_FAST_MS,
    TAU_MU_MS,
    TAU_S_MS,
    PoolParameters,
    equilibrium_state,
    recruitment_sensor,
)
from src.topic4_spatial_slowfast_stage0c_transfer import (
    PreparedPoolParameters,
    moments_from_prepared,
    prepare_pool_parameters,
)


@dataclass(frozen=True)
class FoldPoint:
    """One fixed-point saddle-node candidate on the ``(z, A)`` surface."""

    additive_mv: float
    z: float
    r_e_khz: float
    r_i_khz: float
    residual_inf: float
    residual_rate_det: float
    leading_real_per_ms: float

    def as_dict(self) -> dict[str, float]:
        return {
            "additive_mv": float(self.additive_mv),
            "z": float(self.z),
            "rE_hz": float(1000.0 * self.r_e_khz),
            "rI_hz": float(1000.0 * self.r_i_khz),
            "residual_inf": float(self.residual_inf),
            "residual_rate_det": float(self.residual_rate_det),
            "leading_real_per_ms": float(self.leading_real_per_ms),
        }


def _broadcast_additive(additive_mv: float | np.ndarray, n: int) -> np.ndarray:
    value = np.asarray(additive_mv, dtype=float)
    if value.ndim == 0:
        value = np.full(n, float(value), dtype=float)
    if value.shape != (n,) or not np.all(np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("additive_mv must be finite, non-negative, and align with states")
    return value


def additive_rhs_prepared(
    state: np.ndarray,
    prepared: PreparedPoolParameters,
    transfer: Any,
    additive_mv: float | np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Evaluate the unchanged Stage-0C RHS after subtracting frozen E current ``A``."""

    state = np.asarray(state, dtype=float)
    if state.ndim != 2 or state.shape != (prepared.size, 9):
        raise ValueError("state and prepared parameters must align as (n,9)")
    additive = _broadcast_additive(additive_mv, prepared.size)
    mu_e, sigma_e, mu_i, sigma_i, s_eff = moments_from_prepared(state, prepared)
    mu_e_effective = mu_e - additive
    target_e = transfer.rate(mu_e_effective, sigma_e, "E")
    target_i = transfer.rate(mu_i, sigma_i, "I")
    out = np.empty_like(state)
    out[:, 0] = (-state[:, 0] + target_e) / TAU_ME
    out[:, 1] = (-state[:, 1] + target_i) / TAU_MI
    out[:, 2] = (state[:, 0] - state[:, 2]) / TAU_AMPA
    out[:, 3] = (state[:, 1] - state[:, 3]) / TAU_GABA
    out[:, 4] = (state[:, 0] - state[:, 4]) / TAU_AMPA
    out[:, 5] = (state[:, 1] - state[:, 5]) / TAU_GABA
    out[:, 6] = (state[:, 0] - state[:, 6]) / TAU_FAST_MS
    out[:, 7] = (-state[:, 7] + recruitment_sensor(state[:, 6])) / TAU_MU_MS
    out[:, 8] = (-state[:, 8] + S_MAX * state[:, 7]) / TAU_S_MS
    return out, (mu_e_effective, sigma_e, mu_i, sigma_i, s_eff)


def additive_rhs(
    state: np.ndarray,
    params: PoolParameters | Sequence[PoolParameters],
    transfer: Any,
    additive_mv: float | np.ndarray,
) -> np.ndarray:
    """Convenience wrapper for one state or an aligned state batch."""

    state = np.asarray(state, dtype=float)
    one = state.ndim == 1
    batch = state[None, :] if one else state
    if batch.ndim != 2 or batch.shape[1] != 9:
        raise ValueError("state must have shape (9,) or (n,9)")
    if isinstance(params, PoolParameters):
        points = [params] * batch.shape[0]
    else:
        points = list(params)
        if len(points) != batch.shape[0]:
            raise ValueError("one PoolParameters object is required per state")
    rhs, _ = additive_rhs_prepared(
        batch, prepare_pool_parameters(points), transfer, additive_mv
    )
    return rhs[0] if one else rhs


def equilibrium_residual(
    rates_khz: Sequence[float],
    params: PoolParameters,
    transfer: Any,
    additive_mv: float = 0.0,
) -> np.ndarray:
    """Two-dimensional E/I nullcline residual on the equilibrium manifold."""

    rates = np.asarray(rates_khz, dtype=float)
    if rates.shape != (2,) or np.any(rates <= 0.0) or not np.all(np.isfinite(rates)):
        return np.full(2, np.nan)
    rhs = additive_rhs(equilibrium_state(rates), params, transfer, additive_mv)
    # ``-tau*rhs`` is rate minus transfer target, still in kHz.
    return np.asarray([-TAU_ME * rhs[0], -TAU_MI * rhs[1]], dtype=float)


def residual_rate_jacobian(
    rates_khz: Sequence[float],
    params: PoolParameters,
    transfer: Any,
    additive_mv: float,
    *,
    relative_step: float = 2e-4,
    absolute_step_khz: float = 2e-7,
) -> np.ndarray:
    """Centered 2x2 Jacobian of the equilibrium nullcline residual."""

    rates = np.asarray(rates_khz, dtype=float)
    if rates.shape != (2,) or np.any(rates <= 0.0):
        raise ValueError("rates must be positive with shape (2,)")
    jac = np.empty((2, 2), dtype=float)
    for column in range(2):
        step = max(absolute_step_khz, relative_step * max(float(rates[column]), 1e-3))
        plus = rates.copy()
        minus = rates.copy()
        plus[column] += step
        minus[column] -= step
        if minus[column] <= 0.0:
            raise ValueError("rate-Jacobian probe left positive rate domain")
        jac[:, column] = (
            equilibrium_residual(plus, params, transfer, additive_mv)
            - equilibrium_residual(minus, params, transfer, additive_mv)
        ) / (2.0 * step)
    return jac


def full_rhs_jacobian(
    state: Sequence[float],
    params: PoolParameters,
    transfer: Any,
    additive_mv: float,
    *,
    relative_step: float = 2e-4,
    absolute_step: float = 2e-7,
) -> np.ndarray:
    """Centered full 9D Jacobian of the same additive-current vector field."""

    state = np.asarray(state, dtype=float)
    if state.shape != (9,) or not np.all(np.isfinite(state)):
        raise ValueError("state must be finite with shape (9,)")
    jac = np.empty((9, 9), dtype=float)
    for column in range(9):
        step = max(absolute_step, relative_step * max(abs(float(state[column])), 1e-3))
        plus = state.copy()
        minus = state.copy()
        plus[column] += step
        minus[column] -= step
        jac[:, column] = (
            additive_rhs(plus, params, transfer, additive_mv)
            - additive_rhs(minus, params, transfer, additive_mv)
        ) / (2.0 * step)
    return jac


def _fold_fit_residual(
    variables: np.ndarray,
    *,
    alpha_g: float,
    additive_mv: float,
    transfer: Any,
    w_ee_mult: float,
    ratio: float,
) -> np.ndarray:
    r_e, r_i, z = map(float, variables)
    if r_e <= 0.0 or r_i <= 0.0 or not 0.6 < z < 1.0:
        return np.asarray([1e3, 1e3, 1e3], dtype=float)
    params = PoolParameters(z, alpha_g, w_ee_mult, ratio)
    residual = equilibrium_residual((r_e, r_i), params, transfer, additive_mv)
    jac = residual_rate_jacobian((r_e, r_i), params, transfer, additive_mv)
    # Hz scaling makes the equilibrium residual commensurate with det(J_R).
    return np.asarray([1000.0 * residual[0], 1000.0 * residual[1], np.linalg.det(jac)])


def solve_fold(
    additive_mv: float,
    transfer: Any,
    *,
    initial: Sequence[float] = (0.00203, 0.00706, 0.87447),
    alpha_g: float = 15.0,
    w_ee_mult: float = 1.1,
    ratio: float = 1.0,
) -> FoldPoint:
    """Solve ``R_E=R_I=det(dR/dr)=0`` for one frozen additive current."""

    fit = least_squares(
        lambda value: _fold_fit_residual(
            value,
            alpha_g=float(alpha_g),
            additive_mv=float(additive_mv),
            transfer=transfer,
            w_ee_mult=float(w_ee_mult),
            ratio=float(ratio),
        ),
        np.asarray(initial, dtype=float),
        bounds=([1e-5, 1e-5, 0.60], [0.020, 0.050, 0.999]),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=1200,
        x_scale=np.asarray([0.002, 0.007, 0.05]),
    )
    residual = _fold_fit_residual(
        fit.x,
        alpha_g=float(alpha_g),
        additive_mv=float(additive_mv),
        transfer=transfer,
        w_ee_mult=float(w_ee_mult),
        ratio=float(ratio),
    )
    if not fit.success or not np.all(np.isfinite(residual)) or np.max(np.abs(residual)) > 2e-5:
        raise RuntimeError(
            f"fold solve failed for A={additive_mv}: success={fit.success}, residual={residual}"
        )
    r_e, r_i, z = map(float, fit.x)
    params = PoolParameters(z, alpha_g, w_ee_mult, ratio)
    state = equilibrium_state((r_e, r_i))
    leading = np.linalg.eigvals(full_rhs_jacobian(state, params, transfer, additive_mv))
    leading_real = float(np.max(leading.real))
    rate_det = float(
        np.linalg.det(residual_rate_jacobian((r_e, r_i), params, transfer, additive_mv))
    )
    return FoldPoint(
        additive_mv=float(additive_mv),
        z=z,
        r_e_khz=r_e,
        r_i_khz=r_i,
        residual_inf=float(np.max(np.abs(residual))),
        residual_rate_det=rate_det,
        leading_real_per_ms=leading_real,
    )


def find_equilibria(
    params: PoolParameters,
    transfer: Any,
    additive_mv: float,
    *,
    seeds_khz: Sequence[Sequence[float]] | None = None,
    residual_tolerance_khz: float = 2e-8,
    cluster_tolerance_khz: float = 2e-5,
) -> list[dict[str, Any]]:
    """Find distinct equilibria and classify them with the full 9D Jacobian."""

    if seeds_khz is None:
        e_seeds = (1e-4, 4e-4, 1e-3, 2e-3, 4e-3, 8e-3, 0.02, 0.06)
        i_seeds = (1e-3, 4e-3, 7e-3, 0.012, 0.03, 0.08, 0.18)
        seeds_khz = [(e, i) for e in e_seeds for i in i_seeds]
    roots: list[np.ndarray] = []
    for seed in seeds_khz:
        try:
            fit = least_squares(
                lambda rates: equilibrium_residual(rates, params, transfer, additive_mv),
                np.asarray(seed, dtype=float),
                bounds=([1e-8, 1e-8], [0.25, 0.60]),
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=500,
                x_scale="jac",
            )
        except ValueError:
            continue
        residual = equilibrium_residual(fit.x, params, transfer, additive_mv)
        if (
            fit.success
            and np.all(np.isfinite(residual))
            and float(np.max(np.abs(residual))) <= float(residual_tolerance_khz)
            and not any(
                float(np.max(np.abs(fit.x - previous))) <= float(cluster_tolerance_khz)
                for previous in roots
            )
        ):
            roots.append(np.asarray(fit.x, dtype=float))
    output: list[dict[str, Any]] = []
    for rates in sorted(roots, key=lambda value: (value[0], value[1])):
        state = equilibrium_state(rates)
        eigenvalues = np.linalg.eigvals(
            full_rhs_jacobian(state, params, transfer, additive_mv)
        )
        leading = eigenvalues[int(np.argmax(eigenvalues.real))]
        output.append(
            {
                "rE_hz": float(1000.0 * rates[0]),
                "rI_hz": float(1000.0 * rates[1]),
                "stability": "stable" if leading.real < -1e-5 else (
                    "unstable" if leading.real > 1e-5 else "marginal"
                ),
                "leading_real_per_ms": float(leading.real),
                "leading_imag_per_ms": float(abs(leading.imag)),
                "residual_inf_khz": float(
                    np.max(np.abs(equilibrium_residual(rates, params, transfer, additive_mv)))
                ),
            }
        )
    return output


def nullcline_grid(
    e_hz: Sequence[float],
    i_hz: Sequence[float],
    params: PoolParameters,
    transfer: Any,
    additive_mv: float,
    *,
    chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Return E and I equilibrium residual fields on an ``I x E`` grid."""

    e = np.asarray(e_hz, dtype=float) * 1e-3
    i = np.asarray(i_hz, dtype=float) * 1e-3
    if e.ndim != 1 or i.ndim != 1 or np.any(e <= 0.0) or np.any(i <= 0.0):
        raise ValueError("nullcline axes must be positive 1D rate arrays")
    ee, ii = np.meshgrid(e, i)
    flat = np.column_stack((ee.ravel(), ii.ravel()))
    output = np.full((flat.shape[0], 2), np.nan, dtype=float)
    for start in range(0, flat.shape[0], int(chunk_size)):
        stop = min(flat.shape[0], start + int(chunk_size))
        states = np.vstack([equilibrium_state(row) for row in flat[start:stop]])
        rhs = additive_rhs(states, [params] * states.shape[0], transfer, additive_mv)
        output[start:stop, 0] = -TAU_ME * rhs[:, 0]
        output[start:stop, 1] = -TAU_MI * rhs[:, 1]
    shape = (i.size, e.size)
    return output[:, 0].reshape(shape), output[:, 1].reshape(shape)


def integrate_frozen_forks(
    initial_states: np.ndarray,
    params: Sequence[PoolParameters],
    transfer: Any,
    additive_mv: Sequence[float] | np.ndarray,
    *,
    dt_ms: float,
    duration_ms: float,
    save_dt_ms: float = 1.0,
) -> dict[str, np.ndarray]:
    """Vectorized Euler state forks with no clipping and a compact E-rate trace."""

    state = np.asarray(initial_states, dtype=float).copy()
    points = list(params)
    additive = np.asarray(additive_mv, dtype=float)
    if state.ndim != 2 or state.shape != (len(points), 9) or additive.shape != (len(points),):
        raise ValueError("initial states, parameters, and additive current must align")
    if dt_ms <= 0.0 or duration_ms <= dt_ms or save_dt_ms < dt_ms:
        raise ValueError("invalid integration time contract")
    n_steps = int(round(float(duration_ms) / float(dt_ms)))
    save_stride = int(round(float(save_dt_ms) / float(dt_ms)))
    if not np.isclose(n_steps * dt_ms, duration_ms) or not np.isclose(save_stride * dt_ms, save_dt_ms):
        raise ValueError("duration and save_dt must be integer multiples of dt")
    prepared = prepare_pool_parameters(points)
    sample_steps = np.arange(0, n_steps + 1, save_stride, dtype=int)
    if sample_steps[-1] != n_steps:
        sample_steps = np.r_[sample_steps, n_steps]
    trace_e = np.full((sample_steps.size, len(points)), np.nan, dtype=np.float32)
    trace_i = np.full_like(trace_e, np.nan)
    support_violation = np.zeros(len(points), dtype=np.int64)
    finite = np.ones(len(points), dtype=bool)
    sample_index = 0
    for step in range(n_steps + 1):
        rhs, moments = additive_rhs_prepared(state, prepared, transfer, additive)
        mu_e, sigma_e, mu_i, sigma_i, _ = moments
        supported = transfer.support_mask(mu_e, sigma_e) & transfer.support_mask(mu_i, sigma_i)
        support_violation += ~supported
        finite &= np.all(np.isfinite(state), axis=1) & np.all(np.isfinite(rhs), axis=1)
        if sample_index < sample_steps.size and step == sample_steps[sample_index]:
            trace_e[sample_index] = state[:, 0]
            trace_i[sample_index] = state[:, 1]
            sample_index += 1
        if step == n_steps:
            break
        state += float(dt_ms) * rhs
    return {
        "time_ms": sample_steps.astype(float) * float(dt_ms),
        "rE_khz": trace_e,
        "rI_khz": trace_i,
        "final_state": state,
        "finite": finite,
        "support_violation_count": support_violation,
    }


def summarize_fork_trace(
    time_ms: Sequence[float],
    rate_e_khz: Sequence[float],
    *,
    audit_start_ms: float = 10_000.0,
    tail_start_ms: float = 15_000.0,
    peak_height_hz: float = 20.0,
    minimum_peak_distance_ms: float = 300.0,
) -> dict[str, Any]:
    """Classify an operational frozen-state fork without claiming cycle continuation."""

    time = np.asarray(time_ms, dtype=float)
    rate = 1000.0 * np.asarray(rate_e_khz, dtype=float)
    if time.ndim != 1 or rate.shape != time.shape or time.size < 3:
        raise ValueError("time and rate trace must be aligned 1D arrays")
    if not np.all(np.isfinite(rate)):
        return {"status": "numerical_unresolved", "finite": False}
    dt = float(np.median(np.diff(time)))
    audit = time >= float(audit_start_ms)
    tail = time >= float(tail_start_ms)
    peaks, _ = find_peaks(
        rate[audit],
        height=float(peak_height_hz),
        distance=max(1, int(round(float(minimum_peak_distance_ms) / dt))),
    )
    peak_times = time[audit][peaks]
    period = float(np.median(np.diff(peak_times))) if peak_times.size >= 2 else None
    tail_min = float(np.min(rate[tail]))
    tail_max = float(np.max(rate[tail]))
    tail_mean = float(np.mean(rate[tail]))
    if tail_max - tail_min <= 0.5 and tail_mean < 5.0:
        status = "settled_low"
    elif peak_times.size >= 2 and tail_max >= peak_height_hz:
        status = "oscillatory"
    else:
        status = "long_transient_or_unresolved"
    return {
        "status": status,
        "finite": True,
        "n_peaks_after_audit_start": int(peak_times.size),
        "period_ms": period,
        "tail_mean_hz": tail_mean,
        "tail_min_hz": tail_min,
        "tail_max_hz": tail_max,
    }


def fit_inverse_sqrt_period(
    z: Sequence[float], period_ms: Sequence[float], z_fold: float
) -> dict[str, float]:
    """Fit the SNIC diagnostic ``T=c0+c1/sqrt(z_fold-z)`` and report R2."""

    z = np.asarray(z, dtype=float)
    period = np.asarray(period_ms, dtype=float)
    valid = np.isfinite(z) & np.isfinite(period) & (z < float(z_fold))
    if int(valid.sum()) < 3:
        raise ValueError("at least three finite below-fold periods are required")
    x = 1.0 / np.sqrt(float(z_fold) - z[valid])
    design = np.column_stack((np.ones(x.size), x))
    coef, *_ = np.linalg.lstsq(design, period[valid], rcond=None)
    predicted = design @ coef
    residual = float(np.sum((period[valid] - predicted) ** 2))
    total = float(np.sum((period[valid] - np.mean(period[valid])) ** 2))
    r2 = 1.0 - residual / total if total > 0.0 else np.nan
    return {
        "intercept_ms": float(coef[0]),
        "coefficient_ms_sqrt_z": float(coef[1]),
        "r_squared": float(r2),
        "n_points": int(valid.sum()),
    }


def macro_recovery_flow(
    m: np.ndarray | float,
    drive: np.ndarray | float,
    *,
    k_up_per_s: float,
    k_down_per_s: float,
    decay_guard: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Bounded build/decay law proposed for the next M implementation."""

    m, drive, guard = np.broadcast_arrays(
        np.asarray(m, dtype=float),
        np.asarray(drive, dtype=float),
        np.asarray(decay_guard, dtype=float),
    )
    if np.any((m < 0.0) | (m > 1.0) | (drive < 0.0) | (drive > 1.0)):
        raise ValueError("m and drive must lie in [0,1]")
    if k_up_per_s <= 0.0 or k_down_per_s <= 0.0 or np.any((guard < 0.0) | (guard > 1.0)):
        raise ValueError("rates must be positive and decay_guard must lie in [0,1]")
    return (
        float(k_up_per_s) * drive * (1.0 - m)
        - float(k_down_per_s) * (1.0 - drive) * guard * m
    )
