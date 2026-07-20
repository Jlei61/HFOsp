"""Stage-0F v1.1 event-map and boundary-stencil engineering repair."""

from __future__ import annotations

from math import log
from typing import Any, Mapping, Sequence

import numpy as np

from src.sef_hfo_lif import TREF_E, TREF_I, TAU_ME, TAU_MI
from src.topic4_spatial_slowfast_stage0c import (
    E_CEILING_KHZ,
    FINITE_HIGH_MAX_KHZ,
    I_CEILING_KHZ,
    S_MAX,
    PoolParameters,
)
from src.topic4_spatial_slowfast_stage0c_transfer import (
    _rhs_and_moments,
    moments_from_prepared,
    prepare_pool_parameters,
)
from src.topic4_spatial_slowfast_stage0e import (
    SECTION_COORDINATES,
    SECTION_INDEX,
    SectionDefinition,
    _natural_state_bad,
    audit_row,
    central_jacobian_from_returns,
    integrate_to_returns_batch,
    poincare_fixed_point_shooting,
    scaled_inf_distance,
    shooting_cycle_validation,
)
from src.topic4_spatial_slowfast_stage0f import (
    LOCKED_POINTS,
    LOG_SQRT_PI,
    SmoothDomain,
    SmoothSiegertTransfer,
    analytic_rhs_jacobian,
    exact_siegert_rate_derivatives,
    lut_orbit_parity_summary,
    match_multipliers,
    normalized_frobenius_difference,
    smooth_dt_parity_summary,
    transfer_parity_summary,
)


DERIVATIVE_LABELS_V11 = (
    "chain_rule",
    "finite_difference_1e-5",
    "finite_difference_3e-6",
)


def _strict_probe_state_valid(state: np.ndarray) -> bool:
    """Exact natural-domain membership for ambient derivative probes."""

    state = np.asarray(state, dtype=float)
    if state.shape != (9,) or not np.all(np.isfinite(state)) or np.any(state < 0.0):
        return False
    upper = np.asarray(
        [
            E_CEILING_KHZ,
            I_CEILING_KHZ,
            E_CEILING_KHZ,
            I_CEILING_KHZ,
            E_CEILING_KHZ,
            I_CEILING_KHZ,
            E_CEILING_KHZ,
            1.0,
            S_MAX,
        ],
        dtype=float,
    )
    return bool(np.all(state <= upper))


class SmoothSiegertTransferV11(SmoothSiegertTransfer):
    """Same cubic transfer with a value-only hot path for nominal integration."""

    def log_integral_value(self, mu_mv: np.ndarray, sigma_mv: np.ndarray) -> np.ndarray:
        mu, sigma = np.broadcast_arrays(
            np.asarray(mu_mv, dtype=float), np.asarray(sigma_mv, dtype=float)
        )
        output = np.full(mu.shape, np.nan, dtype=float)
        valid = self.support_mask(mu, sigma)
        if np.any(valid):
            output[valid] = self._spline.ev(mu[valid], sigma[valid])
        return output

    def rate(self, mu_mv: np.ndarray, sigma_mv: np.ndarray, pop: str) -> np.ndarray:
        log_integral = self.log_integral_value(mu_mv, sigma_mv)
        if pop == "E":
            tau_m, tau_ref = float(TAU_ME), float(TREF_E)
        elif pop == "I":
            tau_m, tau_ref = float(TAU_MI), float(TREF_I)
        else:
            raise ValueError("pop must be E or I")
        log_q = log(tau_m) + LOG_SQRT_PI + log_integral
        return np.exp(-np.logaddexp(log(tau_ref), log_q))


def _probe_rhs(
    states: np.ndarray,
    params: PoolParameters,
    transfer: SmoothSiegertTransferV11,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Evaluate derivative probes with explicit finite/divisor/support audits."""

    states = np.asarray(states, dtype=float)
    if (
        states.ndim != 2
        or states.shape[1] != 9
        or not all(_strict_probe_state_valid(row) for row in states)
    ):
        raise FloatingPointError("finite-difference probe left natural state domain")
    divisor = 1.0 + float(params.alpha_g) * states[:, 8]
    if np.any(~np.isfinite(divisor)) or np.any(divisor <= 0.0):
        raise FloatingPointError("finite-difference probe has invalid divisor")
    prepared = prepare_pool_parameters([params] * states.shape[0])
    rhs, moments = _rhs_and_moments(
        states,
        prepared,
        transfer,
        mechanism="dynamic",
        clamp_s=None,
        subtractive_beta_mv=None,
    )
    mu_e, sigma_e, mu_i, sigma_i, _ = moments
    support = transfer.support_mask(mu_e, sigma_e) & transfer.support_mask(mu_i, sigma_i)
    moment_matrix = np.column_stack((mu_e, sigma_e, mu_i, sigma_i))
    if (
        not np.all(support)
        or not np.all(np.isfinite(states))
        or not np.all(np.isfinite(rhs))
        or not np.all(np.isfinite(moment_matrix))
    ):
        raise FloatingPointError("finite-difference probe failed RHS or transfer-support audit")
    return rhs, moments


def boundary_aware_rhs_jacobian(
    state: np.ndarray,
    params: PoolParameters,
    transfer: SmoothSiegertTransferV11,
    *,
    scales: np.ndarray,
    relative_step: float,
    absolute_floor: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Second-order central/forward/backward RHS Jacobian without step adaptation."""

    state = np.asarray(state, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if state.shape != (9,) or scales.shape != (9,) or np.any(scales <= 0.0):
        raise ValueError("invalid boundary-aware Jacobian input")
    if relative_step <= 0.0 or absolute_floor <= 0.0 or bool(_natural_state_bad(state)):
        raise ValueError("invalid boundary-aware Jacobian schedule or nominal state")
    steps = np.maximum(float(absolute_floor), float(relative_step) * scales)
    probe_states: list[np.ndarray] = []
    recipes: list[tuple[str, int, int]] = []
    for coordinate in range(9):
        h = float(steps[coordinate])
        minus_one = state.copy()
        plus_one = state.copy()
        minus_two = state.copy()
        plus_two = state.copy()
        minus_one[coordinate] -= h
        plus_one[coordinate] += h
        minus_two[coordinate] -= 2.0 * h
        plus_two[coordinate] += 2.0 * h
        central_ok = _strict_probe_state_valid(minus_one) and _strict_probe_state_valid(plus_one)
        forward_ok = _strict_probe_state_valid(plus_one) and _strict_probe_state_valid(plus_two)
        backward_ok = _strict_probe_state_valid(minus_one) and _strict_probe_state_valid(minus_two)
        start = len(probe_states)
        if central_ok:
            probe_states.extend((plus_one, minus_one))
            recipes.append(("central", start, start + 1))
        elif forward_ok:
            probe_states.extend((plus_one, plus_two))
            recipes.append(("forward", start, start + 1))
        elif backward_ok:
            probe_states.extend((minus_one, minus_two))
            recipes.append(("backward", start, start + 1))
        else:
            raise FloatingPointError(
                f"no locked second-order stencil exists for coordinate {coordinate}"
            )
    baseline_rhs, _ = _probe_rhs(state[None, :], params, transfer)
    probe_rhs, _ = _probe_rhs(np.asarray(probe_states, dtype=float), params, transfer)
    jacobian = np.empty((9, 9), dtype=float)
    counts = {"central": 0, "forward": 0, "backward": 0}
    coordinate_stencils: list[str] = []
    for coordinate, (stencil, first, second) in enumerate(recipes):
        h = float(steps[coordinate])
        counts[stencil] += 1
        coordinate_stencils.append(stencil)
        if stencil == "central":
            jacobian[:, coordinate] = (probe_rhs[first] - probe_rhs[second]) / (2.0 * h)
        elif stencil == "forward":
            jacobian[:, coordinate] = (
                -3.0 * baseline_rhs[0] + 4.0 * probe_rhs[first] - probe_rhs[second]
            ) / (2.0 * h)
        else:
            jacobian[:, coordinate] = (
                3.0 * baseline_rhs[0] - 4.0 * probe_rhs[first] + probe_rhs[second]
            ) / (2.0 * h)
    if not np.all(np.isfinite(jacobian)):
        raise FloatingPointError("boundary-aware RHS Jacobian is nonfinite")
    return jacobian, {
        "relative_step": float(relative_step),
        "absolute_floor": float(absolute_floor),
        "steps": steps,
        "coordinate_stencils": coordinate_stencils,
        "counts": counts,
        "all_probe_rhs_finite_supported": True,
        "first_order_fallback_used": False,
        "adaptive_step_used": False,
    }


def event_restarted_closure(
    fixed_state: np.ndarray,
    params: PoolParameters,
    transfer: SmoothSiegertTransferV11,
    *,
    dt_ms: float,
    section: SectionDefinition,
    scales: np.ndarray,
) -> dict[str, Any]:
    """Measure P and P2 on exactly the event-restarted shooting map."""

    fixed = np.asarray(fixed_state, dtype=float)
    batch = integrate_to_returns_batch(
        fixed[None, :],
        [params],
        transfer,
        dt_ms=dt_ms,
        n_returns=2,
        section=section,
    )
    valid = bool(batch["valid"][0])
    returns = np.asarray(batch["return_state"][:, 0], dtype=float)
    times = np.asarray(batch["return_time_ms"][:, 0], dtype=float)
    if valid:
        p_closure = float(scaled_inf_distance(returns[0], fixed, scales))
        p2_closure = float(scaled_inf_distance(returns[1], fixed, scales))
        p2_vs_p = float(scaled_inf_distance(returns[1], returns[0], scales))
    else:
        p_closure = p2_closure = p2_vs_p = np.inf
    return {
        "valid": valid,
        "p_closure": p_closure,
        "p2_closure": p2_closure,
        "p2_vs_p_closure": p2_vs_p,
        "return_state": returns,
        "return_time_ms": times,
        "period_ms": np.asarray([times[0], times[1] - times[0]], dtype=float),
        "transversality_per_ms": np.asarray(batch["transversality_per_ms"][:, 0], dtype=float),
        "physical_support_audit": audit_row(batch["audit"], 0),
        "crossing_audit": [batch["crossing_audit"][index][0] for index in range(2)],
    }


def shooting_summary_v1_1(
    shooting: Mapping[str, Any],
    cycle: Mapping[str, Any] | None,
    restarted: Mapping[str, Any] | None,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply only registered same-map closure and aligned-cycle gates."""

    residual = np.asarray(shooting.get("residual", []), dtype=float)
    periods = np.asarray(shooting.get("period_ms", []), dtype=float)
    tail = periods[-4:]
    cv = float(np.std(tail) / np.mean(tail)) if tail.size == 4 and np.mean(tail) > 0.0 else np.inf
    cycle_valid = bool(cycle is not None and cycle.get("valid", False))
    aligned = float(cycle.get("aligned_cycle_residual", np.inf)) if cycle is not None else np.inf
    nonrestarted_second = (
        float(cycle.get("second_closure_residual", np.nan)) if cycle is not None else np.nan
    )
    p_closure = float(restarted.get("p_closure", np.inf)) if restarted is not None else np.inf
    p2_closure = float(restarted.get("p2_closure", np.inf)) if restarted is not None else np.inf
    checks = {
        "shooting_converged": bool(shooting.get("converged", False)),
        "fixed_point_residual": bool(
            residual.size and residual[-1] <= float(cfg["residual_tolerance"])
        ),
        "period_cv": bool(cv <= float(cfg["period_cv_tolerance"])),
        "cycle_valid": cycle_valid,
        "event_restarted_map_valid": bool(restarted is not None and restarted.get("valid", False)),
        "event_restarted_p_closure": bool(
            p_closure <= float(cfg["event_restarted_p_closure_tolerance"])
        ),
        "event_restarted_p2_closure": bool(
            p2_closure <= float(cfg["event_restarted_p2_closure_tolerance"])
        ),
        "two_cycle_aligned": bool(
            aligned <= float(cfg["aligned_cycle_residual_tolerance"])
        ),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "residual_series": residual,
        "period_series_ms": periods,
        "last_four_period_cv": cv if np.isfinite(cv) else None,
        "event_restarted_p_closure": p_closure if np.isfinite(p_closure) else None,
        "event_restarted_p2_closure": p2_closure if np.isfinite(p2_closure) else None,
        "aligned_cycle_residual": aligned if np.isfinite(aligned) else None,
        "non_event_restarted_second_closure_diagnostic_only": (
            nonrestarted_second if np.isfinite(nonrestarted_second) else None
        ),
    }


def _event_located_tangent_v1_1(
    before: np.ndarray,
    after: np.ndarray,
    tangent_before: np.ndarray,
    tangent_after: np.ndarray,
    section: SectionDefinition,
) -> tuple[float, np.ndarray, np.ndarray]:
    h0 = float(before[section.index] - section.level)
    h1 = float(after[section.index] - section.level)
    if not h0 < 0.0 <= h1:
        raise ValueError("event tangent requires an upward bracket")
    delta = after - before
    denominator = float(delta[section.index])
    fraction = float(-h0 / denominator)
    delta_tangent = tangent_after - tangent_before
    d_fraction = (
        -tangent_before[section.index] * denominator
        - (section.level - before[section.index]) * delta_tangent[section.index]
    ) / denominator**2
    crossing = before + fraction * delta
    tangent = (
        tangent_before
        + fraction * delta_tangent
        + delta[:, None] * d_fraction[None, :]
    )
    crossing[section.index] = section.level
    return fraction, crossing, tangent


def discrete_variational_poincare_v1_1(
    fixed_state: np.ndarray,
    params: PoolParameters,
    transfer: SmoothSiegertTransferV11,
    *,
    dt_ms: float,
    section: SectionDefinition,
    scales: np.ndarray,
    finite_difference_relative_steps: Sequence[float],
    finite_difference_absolute_floor: float,
) -> dict[str, Any]:
    """Propagate chain and two boundary-aware tangent maps on one nominal return."""

    fixed = np.asarray(fixed_state, dtype=float).copy()
    scales = np.asarray(scales, dtype=float)
    relative_steps = tuple(float(value) for value in finite_difference_relative_steps)
    if relative_steps != (1.0e-5, 3.0e-6):
        raise ValueError("Stage0F v1.1 finite-difference steps drifted")
    fixed[section.index] = section.level
    basis = np.zeros((9, 8), dtype=float)
    for column, coordinate in enumerate(SECTION_COORDINATES):
        basis[coordinate, column] = scales[coordinate]
    tangents = {label: basis.copy() for label in DERIVATIVE_LABELS_V11}
    state = fixed.copy()
    prepared = prepare_pool_parameters([params])
    identity = np.eye(9)
    below_seen = False
    max_steps = int(np.ceil(section.max_return_ms / dt_ms)) + 2
    state_trace: list[np.ndarray] = []
    moment_trace: list[np.ndarray] = []
    time_trace: list[float] = []
    state_min = np.full(9, np.inf)
    state_max = np.full(9, -np.inf)
    moment_min = np.full(4, np.inf)
    moment_max = np.full(4, -np.inf)
    peak_e_hz = -np.inf
    stencil_totals = {
        "finite_difference_1e-5": {"central": 0, "forward": 0, "backward": 0},
        "finite_difference_3e-6": {"central": 0, "forward": 0, "backward": 0},
    }
    local_differences: list[list[float]] = []

    for step in range(max_steps):
        rhs, moments = _rhs_and_moments(
            state[None, :],
            prepared,
            transfer,
            mechanism="dynamic",
            clamp_s=None,
            subtractive_beta_mv=None,
        )
        mu_e, sigma_e, mu_i, sigma_i, _ = moments
        moment = np.asarray([mu_e[0], sigma_e[0], mu_i[0], sigma_i[0]], dtype=float)
        support = bool(
            transfer.support_mask(mu_e, sigma_e)[0]
            and transfer.support_mask(mu_i, sigma_i)[0]
        )
        physical = bool(
            support
            and not bool(_natural_state_bad(state))
            and state[0] < FINITE_HIGH_MAX_KHZ
            and np.all(np.isfinite(state))
            and np.all(np.isfinite(rhs[0]))
            and np.all(np.isfinite(moment))
        )
        if not physical:
            return {"valid": False, "reason": "nominal_orbit_physical_or_support_failure"}
        state_trace.append(state.copy())
        moment_trace.append(moment)
        time_trace.append(step * dt_ms)
        state_min = np.minimum(state_min, state)
        state_max = np.maximum(state_max, state)
        moment_min = np.minimum(moment_min, moment)
        moment_max = np.maximum(moment_max, moment)
        peak_e_hz = max(peak_e_hz, 1000.0 * float(state[0]))

        chain_jacobian = analytic_rhs_jacobian(state, params, transfer)
        finite_jacobians: list[np.ndarray] = []
        finite_metadata: list[dict[str, Any]] = []
        for relative in relative_steps:
            jacobian, metadata = boundary_aware_rhs_jacobian(
                state,
                params,
                transfer,
                scales=scales,
                relative_step=relative,
                absolute_floor=finite_difference_absolute_floor,
            )
            finite_jacobians.append(jacobian)
            finite_metadata.append(metadata)
        for label, metadata in zip(DERIVATIVE_LABELS_V11[1:], finite_metadata):
            for stencil, count in metadata["counts"].items():
                stencil_totals[label][stencil] += int(count)
        local_differences.append(
            [
                normalized_frobenius_difference(
                    chain_jacobian, finite, norm_floor=1e-12
                )
                for finite in finite_jacobians
            ]
        )
        jacobians = {
            "chain_rule": chain_jacobian,
            "finite_difference_1e-5": finite_jacobians[0],
            "finite_difference_3e-6": finite_jacobians[1],
        }
        next_state = state + dt_ms * rhs[0]
        next_tangents = {
            label: (identity + dt_ms * jacobians[label]) @ tangents[label]
            for label in DERIVATIVE_LABELS_V11
        }
        h0 = float(state[section.index] - section.level)
        h1 = float(next_state[section.index] - section.level)
        if h0 < 0.0:
            below_seen = True
        if below_seen and h0 < 0.0 <= h1:
            crossing = None
            fraction = None
            crossing_tangents: dict[str, np.ndarray] = {}
            for label in DERIVATIVE_LABELS_V11:
                this_fraction, this_crossing, this_tangent = _event_located_tangent_v1_1(
                    state, next_state, tangents[label], next_tangents[label], section
                )
                if crossing is None:
                    crossing = this_crossing
                    fraction = this_fraction
                elif not np.allclose(crossing, this_crossing, atol=1e-14, rtol=0.0):
                    raise RuntimeError("nominal crossing differs across derivative constructions")
                crossing_tangents[label] = this_tangent
            assert crossing is not None and fraction is not None
            crossing_rhs, crossing_moments = _rhs_and_moments(
                crossing[None, :],
                prepared,
                transfer,
                mechanism="dynamic",
                clamp_s=None,
                subtractive_beta_mv=None,
            )
            crossing_moment = np.asarray([value[0] for value in crossing_moments[:4]], dtype=float)
            continuous_transversality = float(crossing_rhs[0, section.index])
            discrete_transversality = float(
                (next_state[section.index] - state[section.index]) / dt_ms
            )
            state_min = np.minimum(state_min, crossing)
            state_max = np.maximum(state_max, crossing)
            moment_min = np.minimum(moment_min, crossing_moment)
            moment_max = np.maximum(moment_max, crossing_moment)
            matrices: dict[str, np.ndarray] = {}
            multipliers: dict[str, np.ndarray] = {}
            radii: dict[str, float] = {}
            section_rows: dict[str, float] = {}
            full_tangents: dict[str, np.ndarray] = {}
            transverse_scales = scales[SECTION_COORDINATES]
            for label, tangent in crossing_tangents.items():
                full = tangent / scales[:, None]
                matrix = tangent[SECTION_COORDINATES] / transverse_scales[:, None]
                values = np.linalg.eigvals(matrix)
                matrices[label] = matrix
                multipliers[label] = values
                radii[label] = float(np.max(np.abs(values)))
                section_rows[label] = float(np.max(np.abs(full[SECTION_INDEX])))
                full_tangents[label] = full
            local = np.asarray(local_differences, dtype=float)
            return {
                "valid": bool(
                    np.all(np.isfinite(crossing))
                    and np.isfinite(continuous_transversality)
                    and np.isfinite(discrete_transversality)
                    and all(np.all(np.isfinite(value)) for value in matrices.values())
                ),
                "period_ms": float((step + fraction) * dt_ms),
                "crossing_state": crossing,
                "crossing_fraction": float(fraction),
                "continuous_rhs_transversality_per_ms": continuous_transversality,
                "discrete_bracket_transversality_per_ms": discrete_transversality,
                "bracket_state_before": state,
                "bracket_state_after": next_state,
                "n_euler_steps": int(step + 1),
                "poincare_matrices": matrices,
                "full_event_tangents_normalized": full_tangents,
                "multipliers": multipliers,
                "spectral_radii": radii,
                "section_row_max_abs": section_rows,
                "stencil_totals": stencil_totals,
                "local_rhs_jacobian_chain_fd_max_relative": np.max(local, axis=0),
                "local_rhs_jacobian_chain_fd_median_relative": np.median(local, axis=0),
                "nominal_time_ms": np.asarray(time_trace, dtype=float),
                "nominal_state_trace": np.asarray(state_trace, dtype=float),
                "nominal_moment_trace": np.asarray(moment_trace, dtype=float),
                "event_crossing_moment": crossing_moment,
                "nominal_audit": {
                    "clean": True,
                    "n_every_euler_states": len(state_trace),
                    "event_located_crossing_audited_separately": True,
                    "state_minimum": state_min,
                    "state_maximum": state_max,
                    "moment_minimum": moment_min,
                    "moment_maximum": moment_max,
                    "peak_rE_hz": peak_e_hz,
                    "support_violation_count": 0,
                    "state_bound_violation_count": 0,
                    "over_100hz_count": 0,
                },
            }
        state = next_state
        tangents = next_tangents
    return {"valid": False, "reason": "no_upward_return_within_locked_window"}


def nominal_map_identity_summary(
    restarted: Mapping[str, Any],
    variational: Mapping[str, Any],
    scales: np.ndarray,
    cfg: Mapping[str, float],
) -> dict[str, Any]:
    """Require shooting and variational integrators to represent the same map."""

    if not bool(restarted.get("valid", False)) or not bool(variational.get("valid", False)):
        return {"pass": False, "reason": "invalid_nominal_map"}
    shooting_period = float(np.asarray(restarted["return_time_ms"])[0])
    variational_period = float(variational["period_ms"])
    period_difference = abs(shooting_period - variational_period)
    shooting_crossing = np.asarray(restarted["return_state"])[0]
    variational_crossing = np.asarray(variational["crossing_state"])
    crossing_difference = float(
        scaled_inf_distance(shooting_crossing, variational_crossing, scales)
    )
    continuous = float(variational["continuous_rhs_transversality_per_ms"])
    discrete = float(variational["discrete_bracket_transversality_per_ms"])
    checks = {
        "period_identity": bool(
            period_difference <= float(cfg["nominal_period_identity_abs_ms"])
        ),
        "full_crossing_identity": bool(
            crossing_difference <= float(cfg["nominal_crossing_identity_scaled"])
        ),
        "continuous_transversality": bool(
            continuous >= float(cfg["minimum_continuous_transversality_per_ms"])
        ),
        "discrete_transversality": bool(
            discrete >= float(cfg["minimum_discrete_transversality_per_ms"])
        ),
        "nominal_physical_support": bool(
            variational.get("nominal_audit", {}).get("clean", False)
            and restarted.get("physical_support_audit", {}).get("clean", False)
        ),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "shooting_period_ms": shooting_period,
        "variational_period_ms": variational_period,
        "period_difference_ms": period_difference,
        "crossing_difference_scaled": crossing_difference,
        "continuous_rhs_transversality_per_ms": continuous,
        "discrete_bracket_transversality_per_ms": discrete,
        "nominal_audit": variational.get("nominal_audit"),
    }


def variational_consistency_summary_v1_1(
    result: Mapping[str, Any], cfg: Mapping[str, float]
) -> dict[str, Any]:
    if not bool(result.get("valid", False)):
        return {"pass": False, "reason": result.get("reason", "invalid_variational_result")}
    matrices = result["poincare_matrices"]
    floor = float(cfg["matrix_norm_floor"])
    comparisons = {
        "fd_ladder": normalized_frobenius_difference(
            matrices[DERIVATIVE_LABELS_V11[1]],
            matrices[DERIVATIVE_LABELS_V11[2]],
            norm_floor=floor,
        ),
        "chain_vs_fd_1e-5": normalized_frobenius_difference(
            matrices["chain_rule"], matrices[DERIVATIVE_LABELS_V11[1]], norm_floor=floor
        ),
        "chain_vs_fd_3e-6": normalized_frobenius_difference(
            matrices["chain_rule"], matrices[DERIVATIVE_LABELS_V11[2]], norm_floor=floor
        ),
    }
    radii = np.asarray([result["spectral_radii"][label] for label in DERIVATIVE_LABELS_V11])
    section_row = max(float(result["section_row_max_abs"][label]) for label in DERIVATIVE_LABELS_V11)
    continuous = float(result["continuous_rhs_transversality_per_ms"])
    discrete = float(result["discrete_bracket_transversality_per_ms"])
    checks = {
        "matrix_agreement": bool(
            max(comparisons.values()) <= float(cfg["matrix_relative_difference_max"])
        ),
        "spectral_radius_agreement": bool(
            float(np.ptp(radii)) <= float(cfg["spectral_radius_range_max"])
        ),
        "event_section_row": bool(section_row <= float(cfg["section_row_abs_max"])),
        "continuous_transversality": bool(
            continuous >= float(cfg["minimum_continuous_transversality_per_ms"])
        ),
        "discrete_transversality": bool(
            discrete >= float(cfg["minimum_discrete_transversality_per_ms"])
        ),
        "physical_support": bool(result.get("nominal_audit", {}).get("clean", False)),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "matrix_relative_differences": comparisons,
        "spectral_radii": {
            label: float(result["spectral_radii"][label]) for label in DERIVATIVE_LABELS_V11
        },
        "spectral_radius_range": float(np.ptp(radii)),
        "maximum_normalized_section_row": section_row,
        "continuous_rhs_transversality_per_ms": continuous,
        "discrete_bracket_transversality_per_ms": discrete,
        "stencil_totals": result["stencil_totals"],
    }


def whole_return_poincare_jv(
    fixed_state: np.ndarray,
    params: PoolParameters,
    transfer: SmoothSiegertTransferV11,
    *,
    dt_ms: float,
    section: SectionDefinition,
    scales: np.ndarray,
    epsilon_relative: float,
) -> dict[str, Any]:
    """End-to-end central Jv columns of the complete smooth return map."""

    fixed = np.asarray(fixed_state, dtype=float)
    states: list[np.ndarray] = []
    for coordinate in SECTION_COORDINATES:
        for sign in (1.0, -1.0):
            changed = fixed.copy()
            changed[coordinate] += sign * float(epsilon_relative) * scales[coordinate]
            changed[section.index] = section.level
            states.append(changed)
    initial = np.asarray(states, dtype=float)
    if not all(_strict_probe_state_valid(row) for row in initial):
        return {
            "valid": False,
            "reason": "whole_return_input_left_natural_domain",
            "initial_probe_audit": {
                "clean": False,
                "n_probes": 16,
                "strict_natural_domain_count": int(
                    sum(_strict_probe_state_valid(row) for row in initial)
                ),
            },
            "crossing_audit": [],
            "crossing_audits_clean": False,
        }
    try:
        _probe_rhs(initial, params, transfer)
    except (FloatingPointError, ValueError) as error:
        return {
            "valid": False,
            "reason": "whole_return_initial_probe_rhs_or_support_failure",
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "initial_probe_audit": {
                "clean": False,
                "n_probes": 16,
                "strict_natural_domain_count": 16,
                "rhs_or_support_exception": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            },
            "crossing_audit": [],
            "crossing_audits_clean": False,
        }
    initial_probe_audit = {
        "clean": True,
        "n_probes": 16,
        "strict_natural_domain_count": 16,
        "positive_divisor_count": 16,
        "finite_rhs_count": 16,
        "transfer_support_count": 16,
    }
    try:
        batch = integrate_to_returns_batch(
            initial,
            [params] * 16,
            transfer,
            dt_ms=dt_ms,
            n_returns=1,
            section=section,
        )
    except (FloatingPointError, ValueError, RuntimeError) as error:
        return {
            "valid": False,
            "reason": "whole_return_integration_exception",
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "initial_probe_audit": initial_probe_audit,
            "crossing_audit": [],
            "crossing_audits_clean": False,
        }
    partial_crossing_audits = [
        batch["crossing_audit"][0][index]
        for index in range(16)
        if batch["crossing_audit"][0][index] is not None
    ]
    if not np.all(batch["valid"]):
        return {
            "valid": False,
            "reason": "whole_return_probe_failed",
            "valid_count": int(np.sum(batch["valid"])),
            "initial_probe_audit": initial_probe_audit,
            "per_probe_audit": [audit_row(batch["audit"], index) for index in range(16)],
            "crossing_audit": partial_crossing_audits,
            "crossing_audit_count": len(partial_crossing_audits),
            "crossing_audits_clean": False,
        }
    returned = np.asarray(batch["return_state"][0], dtype=float)
    matrix = central_jacobian_from_returns(returned, scales, float(epsilon_relative))
    periods = np.asarray(batch["return_time_ms"][0], dtype=float)
    transversalities = np.asarray(batch["transversality_per_ms"][0], dtype=float)
    crossing_audits = [batch["crossing_audit"][0][index] for index in range(16)]
    crossing_clean = bool(
        len(crossing_audits) == 16
        and all(audit is not None and bool(audit.get("clean", False)) for audit in crossing_audits)
    )
    return {
        "valid": bool(
            np.all(np.isfinite(matrix))
            and np.all(np.isfinite(periods))
            and np.all(np.isfinite(transversalities))
            and crossing_clean
        ),
        "epsilon_relative": float(epsilon_relative),
        "jv_matrix": matrix,
        "spectral_radius_diagnostic": float(np.max(np.abs(np.linalg.eigvals(matrix)))),
        "return_period_ms": periods,
        "return_period_minimum_ms": float(np.min(periods)),
        "return_period_maximum_ms": float(np.max(periods)),
        "period_band_pass": bool(
            np.all(periods >= section.min_return_ms)
            and np.all(periods <= section.max_return_ms)
        ),
        "minimum_transversality_per_ms": float(np.min(transversalities)),
        "maximum_transversality_per_ms": float(np.max(transversalities)),
        "crossing_audits_clean": crossing_clean,
        "crossing_audit": crossing_audits,
        "initial_probe_audit": initial_probe_audit,
        "per_probe_audit": [audit_row(batch["audit"], index) for index in range(16)],
    }


def whole_return_jv_summary(
    rows: Sequence[Mapping[str, Any]],
    chain_matrix: np.ndarray,
    cfg: Mapping[str, float],
    *,
    norm_floor: float,
) -> dict[str, Any]:
    rows = list(rows)
    expected = [1.0e-3, 3.0e-4]
    if (
        len(rows) != 2
        or [float(row.get("epsilon_relative", np.nan)) for row in rows] != expected
        or not all(bool(row.get("valid", False)) for row in rows)
    ):
        return {"pass": False, "reason": "invalid_whole_return_jv_level"}
    matrices = [np.asarray(row["jv_matrix"], dtype=float) for row in rows]
    comparisons = {
        "epsilon_ladder": normalized_frobenius_difference(
            matrices[0], matrices[1], norm_floor=norm_floor
        ),
        "chain_vs_epsilon_1e-3": normalized_frobenius_difference(
            chain_matrix, matrices[0], norm_floor=norm_floor
        ),
        "chain_vs_epsilon_3e-4": normalized_frobenius_difference(
            chain_matrix, matrices[1], norm_floor=norm_floor
        ),
    }
    radii = np.asarray(
        [
            float(np.max(np.abs(np.linalg.eigvals(chain_matrix)))),
            float(rows[0]["spectral_radius_diagnostic"]),
            float(rows[1]["spectral_radius_diagnostic"]),
        ]
    )
    checks = {
        "matrix_agreement": bool(
            max(comparisons.values()) <= float(cfg["matrix_relative_difference_max"])
        ),
        "spectral_radius_agreement": bool(
            float(np.ptp(radii)) <= float(cfg["spectral_radius_range_max"])
        ),
        "all_probe_audits_clean": bool(
            all(all(probe["clean"] for probe in row["per_probe_audit"]) for row in rows)
        ),
        "all_initial_probe_audits_clean": bool(
            all(bool(row.get("initial_probe_audit", {}).get("clean", False)) for row in rows)
        ),
        "all_crossing_audits_clean": bool(
            all(
                bool(row.get("crossing_audits_clean", False))
                and len(row.get("crossing_audit", [])) == 16
                and all(bool(audit.get("clean", False)) for audit in row["crossing_audit"])
                for row in rows
            )
        ),
        "all_period_bands_pass": bool(
            all(bool(row.get("period_band_pass", False)) for row in rows)
        ),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "matrix_relative_differences": comparisons,
        "spectral_radii_chain_and_jv": radii,
        "spectral_radius_range": float(np.ptp(radii)),
    }


def variational_transfer_parity_rows(
    variational: Mapping[str, Any],
    params: PoolParameters,
    transfer: SmoothSiegertTransferV11,
    *,
    dt_ms: float,
    n_samples: int,
) -> list[dict[str, Any]]:
    """Direct exact parity on sampled actual Euler states of the tangent return."""

    states = np.asarray(variational["nominal_state_trace"], dtype=float)
    times = np.asarray(variational["nominal_time_ms"], dtype=float)
    if states.ndim != 2 or states.shape[1] != 9 or states.shape[0] < n_samples:
        raise ValueError("variational nominal trace is too short for locked parity audit")
    indices = np.rint(np.linspace(0, states.shape[0] - 1, int(n_samples))).astype(int)
    if np.unique(indices).size != int(n_samples):
        raise RuntimeError("locked variational parity indices are not unique")
    sampled = states[indices]
    moments = moments_from_prepared(
        sampled, prepare_pool_parameters([params] * int(n_samples))
    )
    mu_e, sigma_e, mu_i, sigma_i, _ = moments
    rows: list[dict[str, Any]] = []
    for pop, mu, sigma in (("E", mu_e, sigma_e), ("I", mu_i, sigma_i)):
        smooth = transfer.rate_with_derivatives(mu, sigma, pop)
        exact = exact_siegert_rate_derivatives(mu, sigma, pop)
        for sample_index, source_index in enumerate(indices):
            row: dict[str, Any] = {
                "z": float(params.z),
                "alpha_G": float(params.alpha_g),
                "dt_ms": float(dt_ms),
                "sample_index": int(sample_index),
                "source_state_index": int(source_index),
                "source_time_ms": float(times[source_index]),
                "source": "variational_nominal_euler_state",
                "population": pop,
                "mu_mv": float(mu[sample_index]),
                "sigma_mv": float(sigma[sample_index]),
            }
            for label, observed_values, target_values in zip(
                ("rate_khz", "d_rate_d_mu_khz_per_mv", "d_rate_d_sigma_khz_per_mv"),
                smooth,
                exact,
            ):
                observed = float(observed_values[sample_index])
                target = float(target_values[sample_index])
                row[f"smooth_{label}"] = observed
                row[f"exact_{label}"] = target
                row[f"absolute_error_{label}"] = abs(observed - target)
            rows.append(row)
    return rows


def stability_certificate_summary_v1_1(
    base: Mapping[str, Any], half: Mapping[str, Any], cfg: Mapping[str, float]
) -> dict[str, Any]:
    if not bool(base.get("valid", False)) or not bool(half.get("valid", False)):
        return {"pass": False, "reason": "invalid_variational_result"}
    base_rho = {label: float(base["spectral_radii"][label]) for label in DERIVATIVE_LABELS_V11}
    half_rho = {label: float(half["spectral_radii"][label]) for label in DERIVATIVE_LABELS_V11}
    method_spread = max(float(np.ptp(list(base_rho.values()))), float(np.ptp(list(half_rho.values()))))
    dt_differences = {label: abs(base_rho[label] - half_rho[label]) for label in DERIVATIVE_LABELS_V11}
    dt_spread = max(dt_differences.values())
    rho_max = max(*base_rho.values(), *half_rho.values())
    margin = 1.0 - rho_max
    required = max(
        float(cfg["minimum_unit_circle_margin"]),
        float(cfg["uncertainty_multiplier"]) * method_spread,
        float(cfg["uncertainty_multiplier"]) * dt_spread,
    )
    return {
        "pass": bool(rho_max < 1.0 and margin >= required),
        "base_spectral_radii": base_rho,
        "half_spectral_radii": half_rho,
        "method_spectral_radius_spread": method_spread,
        "same_method_dt_differences": dt_differences,
        "dt_spectral_radius_spread": dt_spread,
        "rho_max": rho_max,
        "unit_circle_margin": margin,
        "required_margin": required,
        "all_nontrivial_multipliers_inside_unit_circle": bool(rho_max < 1.0),
        "multiplier_matching_base_vs_half": {
            label: match_multipliers(base["multipliers"][label], half["multipliers"][label])
            for label in DERIVATIVE_LABELS_V11
        },
    }


def run_point_certificate_v1_1(
    params: PoolParameters,
    transfer: SmoothSiegertTransferV11,
    stage0e_inputs: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    params.validate()
    if (float(params.z), float(params.alpha_g)) not in LOCKED_POINTS:
        raise ValueError("Stage0F v1.1 parameter point drifted")
    scales = np.asarray(stage0e_inputs["scales"], dtype=float)
    section_cfg = cfg["section"]
    section = SectionDefinition(
        index=int(section_cfg["state_index"]),
        level=float(section_cfg["level"]),
        direction=str(section_cfg["direction"]),
        min_return_ms=float(section_cfg["min_return_ms"]),
        max_return_ms=float(section_cfg["max_return_ms"]),
    ).validate()
    result: dict[str, Any] = {
        "z": float(params.z),
        "alpha_G": float(params.alpha_g),
        "outcome": "periodic_orbit_derivative_unresolved",
        "derivative_certified": False,
        "failed_gates": [],
        "stage1_open": False,
        "space_open": False,
        "state_scales": scales,
    }
    artifacts: dict[str, Any] = {
        "shootings": {},
        "cycles": {},
        "event_restarted": {},
        "variational": {},
        "whole_return_jv": {},
        "transfer_parity_rows": [],
    }
    for label, dt_ms in zip(("base", "half"), (0.125, 0.0625)):
        shooting = poincare_fixed_point_shooting(
            np.asarray(stage0e_inputs[f"{label}_shooting_seed"], dtype=float),
            params,
            transfer,
            dt_ms=dt_ms,
            section=section,
            scales=scales,
            max_iterations=int(cfg["shooting"]["max_iterations"]),
            residual_tolerance=float(cfg["shooting"]["residual_tolerance"]),
            minimum_iterations=int(cfg["shooting"]["minimum_iterations"]),
        )
        artifacts["shootings"][label] = shooting
        cycle = None
        restarted = None
        if bool(shooting.get("converged", False)):
            cycle = shooting_cycle_validation(
                shooting["fixed_state"],
                params,
                transfer,
                dt_ms=dt_ms,
                section=section,
                scales=scales,
                n_phase=int(cfg["phase_bins"]),
            )
            restarted = event_restarted_closure(
                shooting["fixed_state"],
                params,
                transfer,
                dt_ms=dt_ms,
                section=section,
                scales=scales,
            )
        gate = shooting_summary_v1_1(shooting, cycle, restarted, cfg["shooting"])
        result[f"{label}_smooth_shooting"] = gate
        if not gate["pass"]:
            result["failed_gates"].append(f"{label}_smooth_shooting")
            return result, artifacts
        assert cycle is not None and restarted is not None
        artifacts["cycles"][label] = cycle
        artifacts["event_restarted"][label] = restarted
        lut_parity = lut_orbit_parity_summary(
            cycle,
            stage0e_inputs[f"{label}_lut_trace"],
            scales,
            cfg["orbit_parity"],
            n_phase=int(cfg["phase_bins"]),
        )
        result[f"{label}_lut_orbit_parity"] = lut_parity
        if not lut_parity["pass"]:
            result["outcome"] = "smooth_orbit_not_lut_equivalent"
            result["failed_gates"].append(f"{label}_lut_orbit_parity")
            return result, artifacts

        try:
            variation = discrete_variational_poincare_v1_1(
                shooting["fixed_state"],
                params,
                transfer,
                dt_ms=dt_ms,
                section=section,
                scales=scales,
                finite_difference_relative_steps=cfg["variational"]["finite_difference_relative_steps"],
                finite_difference_absolute_floor=float(
                    cfg["variational"]["finite_difference_absolute_floor"]
                ),
            )
        except (FloatingPointError, ValueError, RuntimeError) as error:
            variation = {
                "valid": False,
                "reason": "variational_derivative_exception",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            }
        artifacts["variational"][label] = variation
        identity = nominal_map_identity_summary(
            restarted, variation, scales, cfg["variational"]
        )
        consistency = variational_consistency_summary_v1_1(
            variation, cfg["variational"]
        )
        result[f"{label}_nominal_map_identity"] = identity
        result[f"{label}_variational_consistency"] = consistency
        if not identity["pass"]:
            result["failed_gates"].append(f"{label}_nominal_map_identity")
        if not consistency["pass"]:
            result["failed_gates"].append(f"{label}_variational_consistency")
        if result["failed_gates"]:
            return result, artifacts

        jv_rows = []
        for epsilon in cfg["whole_return_jv"]["epsilon_relative"]:
            try:
                row = whole_return_poincare_jv(
                    shooting["fixed_state"],
                    params,
                    transfer,
                    dt_ms=dt_ms,
                    section=section,
                    scales=scales,
                    epsilon_relative=float(epsilon),
                )
            except (FloatingPointError, ValueError, RuntimeError) as error:
                row = {
                    "valid": False,
                    "epsilon_relative": float(epsilon),
                    "reason": "whole_return_jv_exception",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }
            jv_rows.append(row)
        artifacts["whole_return_jv"][label] = jv_rows
        jv_summary = whole_return_jv_summary(
            jv_rows,
            variation["poincare_matrices"]["chain_rule"],
            cfg["whole_return_jv"],
            norm_floor=float(cfg["variational"]["matrix_norm_floor"]),
        )
        result[f"{label}_whole_return_jv"] = jv_summary
        if not jv_summary["pass"]:
            result["failed_gates"].append(f"{label}_whole_return_jv")
            return result, artifacts

        try:
            parity_rows = variational_transfer_parity_rows(
                variation,
                params,
                transfer,
                dt_ms=dt_ms,
                n_samples=int(cfg["transfer_parity_variational_samples"]),
            )
        except (FloatingPointError, ValueError, RuntimeError) as error:
            result["failed_gates"].append(f"{label}_exact_transfer_parity_exception")
            result[f"{label}_exact_transfer_parity_exception"] = {
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            }
            return result, artifacts
        artifacts["transfer_parity_rows"].extend(parity_rows)

    result["smooth_dt_parity"] = smooth_dt_parity_summary(
        artifacts["cycles"]["base"],
        artifacts["cycles"]["half"],
        scales,
        cfg["orbit_parity"],
    )
    result["transfer_parity"] = transfer_parity_summary(
        artifacts["transfer_parity_rows"], cfg["transfer_parity"]
    )
    result["stability_certificate"] = stability_certificate_summary_v1_1(
        artifacts["variational"]["base"],
        artifacts["variational"]["half"],
        cfg["stability"],
    )
    final_gates = {
        "smooth_dt_parity": result["smooth_dt_parity"]["pass"],
        "exact_transfer_value_derivative_parity": result["transfer_parity"]["pass"],
        "floquet_stability_margin": result["stability_certificate"]["pass"],
    }
    result["failed_gates"].extend([name for name, passed in final_gates.items() if not passed])
    if not result["failed_gates"]:
        result["outcome"] = "stable_periodic_orbit_derivative_certified"
        result["derivative_certified"] = True
    return result, artifacts
