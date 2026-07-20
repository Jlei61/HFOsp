"""Event-located periodic-orbit tools for the additive-current MZ line.

The frozen Stage-0E/0F implementations are intentionally left untouched: their
RHS is hard-wired to ``A=0``.  This module applies the same directed Poincare
semantics to the locked nine-state Stage-0C fast system with

    mu_E -> mu_E - A.

It follows stable cycles by event-restarted return-map iteration.  Failure of
that iteration is *not* by itself a certificate that no unstable cycle exists;
the caller must report that distinction explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.topic4_mz_entry_exit_nullclines import additive_rhs_prepared
from src.topic4_spatial_slowfast_stage0c import (
    E_CEILING_KHZ,
    FINITE_HIGH_MAX_KHZ,
    I_CEILING_KHZ,
    S_MAX,
    PoolParameters,
)
from src.topic4_spatial_slowfast_stage0c_transfer import (
    PreparedPoolParameters,
    prepare_pool_parameters,
)
from src.topic4_spatial_slowfast_stage0e import (
    SECTION_COORDINATES,
    SectionDefinition,
    interpolate_upward_crossing,
    scaled_inf_distance,
)


@dataclass(frozen=True)
class AdditiveReturnAudit:
    """Compact physical and numerical audit for one directed return."""

    status: str
    return_time_ms: float | None
    crossing_state: np.ndarray | None
    transversality_per_ms: float | None
    peak_r_e_hz: float
    over_100hz_count: int
    support_violation_count: int
    state_bound_violation_count: int
    finite: bool
    trace: dict[str, np.ndarray] | None = None
    minimum_target_distance: float | None = None

    @property
    def valid(self) -> bool:
        return self.status == "clean_return"


def _state_bad(state: np.ndarray) -> bool:
    state = np.asarray(state, dtype=float)
    if state.shape != (9,) or not np.all(np.isfinite(state)):
        return True
    if np.any(state < -1e-9):
        return True
    if state[0] > E_CEILING_KHZ + 1e-9 or state[1] > I_CEILING_KHZ + 1e-9:
        return True
    if np.any(state[[2, 4, 6]] > E_CEILING_KHZ + 1e-9):
        return True
    if np.any(state[[3, 5]] > I_CEILING_KHZ + 1e-9):
        return True
    return bool(state[7] > 1.0 + 1e-9 or state[8] > S_MAX + 1e-9)


def _evaluate(
    state: np.ndarray,
    prepared: PreparedPoolParameters,
    transfer: Any,
    additive_mv: float,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], bool, bool]:
    rhs, moments = additive_rhs_prepared(
        np.asarray(state, dtype=float)[None, :], prepared, transfer, float(additive_mv)
    )
    mu_e, sigma_e, mu_i, sigma_i, _ = moments
    finite = bool(
        np.all(np.isfinite(state))
        and np.all(np.isfinite(rhs[0]))
        and all(np.all(np.isfinite(value)) for value in moments)
    )
    support = bool(
        finite
        and transfer.support_mask(mu_e, sigma_e)[0]
        and transfer.support_mask(mu_i, sigma_i)[0]
    )
    return rhs[0], moments, finite, support


def integrate_additive_return(
    initial_state: Sequence[float],
    params: PoolParameters,
    transfer: Any,
    additive_mv: float,
    *,
    dt_ms: float,
    section: SectionDefinition,
    record_trace: bool = False,
    distance_target: Sequence[float] | None = None,
    distance_scales: Sequence[float] | None = None,
) -> AdditiveReturnAudit:
    """Apply one event-restarted directed return of the additive fast system."""

    section.validate()
    state = np.asarray(initial_state, dtype=float).copy()
    if _state_bad(state) or not np.isclose(state[section.index], section.level, atol=1e-10):
        raise ValueError("initial state must be a natural state on the locked section")
    if dt_ms <= 0.0:
        raise ValueError("dt_ms must be positive")
    if additive_mv < 0.0 or not np.isfinite(additive_mv):
        raise ValueError("additive_mv must be finite and non-negative")
    n_steps = int(np.ceil(float(section.max_return_ms) / float(dt_ms))) + 1
    prepared = prepare_pool_parameters([params.validate()])
    below_seen = False
    peak = 1000.0 * float(state[0])
    over_100 = 0
    support_violations = 0
    bound_violations = 0
    finite = True
    trace_time: list[float] = []
    trace_state: list[np.ndarray] = []
    target = None if distance_target is None else np.asarray(distance_target, dtype=float)
    scales = None if distance_scales is None else np.asarray(distance_scales, dtype=float)
    if (target is None) != (scales is None):
        raise ValueError("distance target and scales must be supplied together")
    if target is not None and (target.shape != (9,) or scales.shape != (9,)):
        raise ValueError("distance target and scales must have shape (9,)")
    minimum_distance = np.inf

    for step in range(n_steps):
        time_ms = float(step) * float(dt_ms)
        rhs, _, this_finite, supported = _evaluate(
            state, prepared, transfer, float(additive_mv)
        )
        finite &= this_finite
        support_violations += int(not supported)
        bad = _state_bad(state)
        bound_violations += int(bad)
        peak = max(peak, 1000.0 * float(state[0]))
        over_100 += int(state[0] >= FINITE_HIGH_MAX_KHZ)
        if target is not None:
            minimum_distance = min(
                minimum_distance,
                float(scaled_inf_distance(state, target, scales)),
            )
        if record_trace:
            trace_time.append(time_ms)
            trace_state.append(state.copy())
        if not this_finite or not supported or bad:
            return AdditiveReturnAudit(
                status="physical_or_numerical_failure",
                return_time_ms=None,
                crossing_state=None,
                transversality_per_ms=None,
                peak_r_e_hz=peak,
                over_100hz_count=over_100,
                support_violation_count=support_violations,
                state_bound_violation_count=bound_violations,
                finite=finite,
                trace=_trace_payload(trace_time, trace_state) if record_trace else None,
                minimum_target_distance=(
                    float(minimum_distance) if np.isfinite(minimum_distance) else None
                ),
            )
        next_state = state + float(dt_ms) * rhs
        h0 = float(state[section.index] - section.level)
        h1 = float(next_state[section.index] - section.level)
        if h0 < 0.0:
            below_seen = True
        if below_seen and h0 < 0.0 <= h1:
            crossing_time, crossing = interpolate_upward_crossing(
                state, next_state, time_ms, float(dt_ms), section
            )
            crossing_rhs, _, crossing_finite, crossing_supported = _evaluate(
                crossing, prepared, transfer, float(additive_mv)
            )
            transversality = float(crossing_rhs[section.index])
            clean = bool(
                crossing_finite
                and crossing_supported
                and not _state_bad(crossing)
                and crossing[0] < FINITE_HIGH_MAX_KHZ
                and transversality > 0.0
                and section.min_return_ms <= crossing_time <= section.max_return_ms
            )
            if record_trace:
                trace_time.append(crossing_time)
                trace_state.append(crossing.copy())
            if target is not None:
                minimum_distance = min(
                    minimum_distance,
                    float(scaled_inf_distance(crossing, target, scales)),
                )
            return AdditiveReturnAudit(
                status="clean_return" if clean else "unclean_crossing",
                return_time_ms=crossing_time,
                crossing_state=crossing,
                transversality_per_ms=transversality,
                peak_r_e_hz=peak,
                over_100hz_count=over_100,
                support_violation_count=support_violations + int(not crossing_supported),
                state_bound_violation_count=bound_violations + int(_state_bad(crossing)),
                finite=bool(finite and crossing_finite),
                trace=_trace_payload(trace_time, trace_state) if record_trace else None,
                minimum_target_distance=(
                    float(minimum_distance) if np.isfinite(minimum_distance) else None
                ),
            )
        state = next_state

    return AdditiveReturnAudit(
        status="no_return_within_locked_window",
        return_time_ms=None,
        crossing_state=None,
        transversality_per_ms=None,
        peak_r_e_hz=peak,
        over_100hz_count=over_100,
        support_violation_count=support_violations,
        state_bound_violation_count=bound_violations,
        finite=finite,
        trace=_trace_payload(trace_time, trace_state) if record_trace else None,
        minimum_target_distance=(
            float(minimum_distance) if np.isfinite(minimum_distance) else None
        ),
    )


def _trace_payload(time: list[float], state: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "time_ms": np.asarray(time, dtype=float),
        "state": np.asarray(state, dtype=float).reshape((-1, 9)),
    }


def shoot_additive_cycle(
    seed_state: Sequence[float],
    params: PoolParameters,
    transfer: Any,
    additive_mv: float,
    *,
    dt_ms: float,
    section: SectionDefinition,
    scales: Sequence[float],
    max_iterations: int = 20,
    minimum_iterations: int = 4,
    residual_tolerance: float = 1e-7,
    period_cv_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Follow an attracting additive cycle by repeated directed Poincare maps."""

    current = np.asarray(seed_state, dtype=float).copy()
    scale = np.asarray(scales, dtype=float)
    if current.shape != (9,) or scale.shape != (9,) or np.any(scale <= 0.0):
        raise ValueError("shooting requires a nine-state seed and positive scales")
    current[section.index] = section.level
    residuals: list[float] = []
    periods: list[float] = []
    transversality: list[float] = []
    states = [current.copy()]
    return_status: list[str] = []

    for _ in range(int(max_iterations)):
        mapped = integrate_additive_return(
            current,
            params,
            transfer,
            additive_mv,
            dt_ms=dt_ms,
            section=section,
        )
        return_status.append(mapped.status)
        if not mapped.valid:
            return {
                "accepted": False,
                "reason": mapped.status,
                "fixed_state": current,
                "iterate_state": np.asarray(states),
                "residual": np.asarray(residuals),
                "period_ms": np.asarray(periods),
                "transversality_per_ms": np.asarray(transversality),
                "return_status": return_status,
                "last_return_audit": mapped,
            }
        returned = np.asarray(mapped.crossing_state, dtype=float)
        residuals.append(float(scaled_inf_distance(returned, current, scale)))
        periods.append(float(mapped.return_time_ms))
        transversality.append(float(mapped.transversality_per_ms))
        current = returned
        states.append(current.copy())
        if len(residuals) >= int(minimum_iterations):
            recent_period = np.asarray(periods[-3:], dtype=float)
            period_cv = float(np.std(recent_period) / np.mean(recent_period))
            if residuals[-1] <= float(residual_tolerance) and period_cv <= float(period_cv_tolerance):
                break

    residual_array = np.asarray(residuals, dtype=float)
    period_array = np.asarray(periods, dtype=float)
    recent_period = period_array[-3:]
    period_cv = (
        float(np.std(recent_period) / np.mean(recent_period))
        if recent_period.size >= 2 and np.mean(recent_period) > 0.0
        else np.inf
    )
    converged = bool(
        residual_array.size >= int(minimum_iterations)
        and residual_array[-1] <= float(residual_tolerance)
        and period_cv <= float(period_cv_tolerance)
    )
    if not converged:
        return {
            "accepted": False,
            "reason": "shooting_not_converged",
            "fixed_state": current,
            "iterate_state": np.asarray(states),
            "residual": residual_array,
            "period_ms": period_array,
            "period_cv": period_cv,
            "transversality_per_ms": np.asarray(transversality),
            "return_status": return_status,
        }

    first = integrate_additive_return(
        current, params, transfer, additive_mv, dt_ms=dt_ms, section=section
    )
    second = (
        integrate_additive_return(
            first.crossing_state,
            params,
            transfer,
            additive_mv,
            dt_ms=dt_ms,
            section=section,
        )
        if first.valid
        else None
    )
    p_closure = (
        float(scaled_inf_distance(first.crossing_state, current, scale))
        if first.valid
        else np.inf
    )
    p2_closure = (
        float(scaled_inf_distance(second.crossing_state, first.crossing_state, scale))
        if second is not None and second.valid
        else np.inf
    )
    validation_pass = bool(
        first.valid
        and second is not None
        and second.valid
        and p_closure <= 10.0 * float(residual_tolerance)
        and p2_closure <= 10.0 * float(residual_tolerance)
    )
    return {
        "accepted": validation_pass,
        "reason": "accepted_stable_cycle" if validation_pass else "p_or_p2_validation_failed",
        "fixed_state": current,
        "iterate_state": np.asarray(states),
        "residual": residual_array,
        "period_ms": period_array,
        "period_cv": period_cv,
        "transversality_per_ms": np.asarray(transversality),
        "return_status": return_status,
        "p_closure": p_closure,
        "p2_closure": p2_closure,
        "validated_period_ms": float(first.return_time_ms) if first.valid else None,
        "peak_r_e_hz": float(first.peak_r_e_hz) if first.valid else None,
        "over_100hz_count": int(first.over_100hz_count) if first.valid else None,
    }


def finite_difference_poincare(
    fixed_state: Sequence[float],
    params: PoolParameters,
    transfer: Any,
    additive_mv: float,
    *,
    dt_ms: float,
    section: SectionDefinition,
    scales: Sequence[float],
    epsilon_relative: float,
) -> dict[str, Any]:
    """Central finite-difference transverse Poincare Jacobian at one cycle."""

    fixed = np.asarray(fixed_state, dtype=float)
    scale = np.asarray(scales, dtype=float)
    if fixed.shape != (9,) or scale.shape != (9,) or epsilon_relative <= 0.0:
        raise ValueError("invalid Poincare finite-difference inputs")
    returned: list[np.ndarray] = []
    audits: list[AdditiveReturnAudit] = []
    for coordinate in SECTION_COORDINATES:
        for sign in (1.0, -1.0):
            probe = fixed.copy()
            probe[coordinate] += sign * float(epsilon_relative) * scale[coordinate]
            probe[section.index] = section.level
            if _state_bad(probe):
                return {"valid": False, "reason": "probe_outside_natural_bounds"}
            mapped = integrate_additive_return(
                probe,
                params,
                transfer,
                additive_mv,
                dt_ms=dt_ms,
                section=section,
            )
            audits.append(mapped)
            if not mapped.valid:
                return {
                    "valid": False,
                    "reason": "one_or_more_probe_returns_failed",
                    "probe_status": [audit.status for audit in audits],
                }
            returned.append(np.asarray(mapped.crossing_state, dtype=float))
    returned_array = np.asarray(returned, dtype=float)
    matrix = np.empty((SECTION_COORDINATES.size, SECTION_COORDINATES.size), dtype=float)
    for column, _ in enumerate(SECTION_COORDINATES):
        plus = returned_array[2 * column]
        minus = returned_array[2 * column + 1]
        normalized = (plus - minus) / (2.0 * float(epsilon_relative) * scale)
        matrix[:, column] = normalized[SECTION_COORDINATES]
    multipliers = np.linalg.eigvals(matrix)
    return {
        "valid": bool(np.all(np.isfinite(matrix)) and np.all(np.isfinite(multipliers))),
        "epsilon_relative": float(epsilon_relative),
        "matrix": matrix,
        "multipliers": multipliers,
        "spectral_radius": float(np.max(np.abs(multipliers))),
        "nearest_plus_one_distance": float(np.min(np.abs(multipliers - 1.0))),
        "minimum_probe_transversality_per_ms": float(
            min(audit.transversality_per_ms for audit in audits)
        ),
    }


def predict_section_state(
    previous: Sequence[float],
    previous_previous: Sequence[float] | None,
    parameter: float,
    previous_parameter: float,
    previous_previous_parameter: float | None,
    section: SectionDefinition,
) -> np.ndarray:
    """Secant predictor used only to seed the next stable-cycle corrector."""

    current = np.asarray(previous, dtype=float)
    if previous_previous is None or previous_previous_parameter is None:
        predicted = current.copy()
    else:
        older = np.asarray(previous_previous, dtype=float)
        denominator = float(previous_parameter) - float(previous_previous_parameter)
        if np.isclose(denominator, 0.0):
            predicted = current.copy()
        else:
            fraction = (float(parameter) - float(previous_parameter)) / denominator
            predicted = current + fraction * (current - older)
    predicted[section.index] = section.level
    return predicted

