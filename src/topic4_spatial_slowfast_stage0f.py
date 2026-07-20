"""Stage-0F smooth-transfer discrete-variational Floquet certificate.

This module does not change the Stage-0C equations.  It constructs an
interpolating C2 representation of the unchanged exact-Siegert log-integral
table, audits values and derivatives against direct quadrature, and
differentiates the forward-Euler, event-located Poincare map in two ways.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import linear_sum_assignment

from src.sef_hfo_lif import (
    C_EE,
    C_EI,
    C_IE,
    C_II,
    JX_E,
    JX_I,
    TAU_AMPA,
    TAU_GABA,
    TAU_ME,
    TAU_MI,
    TREF_E,
    TREF_I,
    V_RESET,
    V_TH,
    W_EI,
    W_IE,
    W_II,
)
from src.topic4_spatial_slowfast_stage0c import (
    FINITE_HIGH_MAX_KHZ,
    S_MAX,
    TAU_FAST_MS,
    TAU_MU_MS,
    TAU_S_MS,
    E0_KHZ,
    E50_KHZ,
    PoolParameters,
)
from src.topic4_spatial_slowfast_stage0c_transfer import (
    ExtendedSiegertTransfer,
    PreparedPoolParameters,
    _LOG_SQRT_PI,
    _log_erfcx_negative_argument,
    _rhs_and_moments,
    moments_from_prepared,
    prepare_pool_parameters,
    stable_siegert_log_integral,
)
from src.topic4_spatial_slowfast_stage0e import (
    SECTION_COORDINATES,
    SECTION_INDEX,
    SectionDefinition,
    _natural_state_bad,
    aligned_waveform_residual,
    interpolate_upward_crossing,
    phase_resample,
    poincare_fixed_point_shooting,
    shooting_cycle_validation,
)


LOG_SQRT_PI = 0.5 * log(np.pi)
LOCKED_POINTS = ((0.85, 15.0), (0.85, 16.0))
DERIVATIVE_LABELS = ("chain_rule", "centered_1e-5", "centered_3e-6")


@dataclass(frozen=True)
class SmoothDomain:
    """Locked orbit-local support of the C2 interpolation."""

    mu_min_mv: float = -160.0
    mu_max_mv: float = 80.0
    sigma_min_mv: float = 3.0
    sigma_max_mv: float = 20.0

    def validate(self) -> "SmoothDomain":
        if not self.mu_min_mv < self.mu_max_mv:
            raise ValueError("invalid smooth mu domain")
        if not 0.0 < self.sigma_min_mv < self.sigma_max_mv:
            raise ValueError("invalid smooth sigma domain")
        return self


class SmoothSiegertTransfer:
    """Cubic interpolation of the unchanged exact-Siegert log-integral table."""

    def __init__(
        self,
        mu_axis: np.ndarray,
        sigma_axis: np.ndarray,
        log_integral_table: np.ndarray,
        *,
        domain: SmoothDomain,
        kx: int = 3,
        ky: int = 3,
        smoothing: float = 0.0,
    ) -> None:
        domain.validate()
        mu_axis = np.asarray(mu_axis, dtype=float)
        sigma_axis = np.asarray(sigma_axis, dtype=float)
        table = np.asarray(log_integral_table, dtype=float)
        if table.shape != (mu_axis.size, sigma_axis.size):
            raise ValueError("transfer table shape mismatch")
        mu_mask = (mu_axis >= domain.mu_min_mv) & (mu_axis <= domain.mu_max_mv)
        sigma_mask = (sigma_axis >= domain.sigma_min_mv) & (sigma_axis <= domain.sigma_max_mv)
        self.mu_axis = mu_axis[mu_mask]
        self.sigma_axis = sigma_axis[sigma_mask]
        local = table[np.ix_(mu_mask, sigma_mask)]
        if self.mu_axis.size < kx + 1 or self.sigma_axis.size < ky + 1:
            raise ValueError("smooth domain contains too few transfer nodes")
        observed_bounds = (
            self.mu_axis[0],
            self.mu_axis[-1],
            self.sigma_axis[0],
            self.sigma_axis[-1],
        )
        expected_bounds = (
            domain.mu_min_mv,
            domain.mu_max_mv,
            domain.sigma_min_mv,
            domain.sigma_max_mv,
        )
        if not np.allclose(observed_bounds, expected_bounds, atol=1e-12, rtol=0.0):
            raise ValueError("locked smooth-domain boundaries are absent from table axes")
        if not np.all(np.isfinite(local)):
            raise ValueError("smooth transfer table contains nonfinite values")
        self.domain = domain
        self.name = "orbit_local_cubic_exact_table"
        self._spline = RectBivariateSpline(
            self.mu_axis,
            self.sigma_axis,
            local,
            kx=int(kx),
            ky=int(ky),
            s=float(smoothing),
        )

    @classmethod
    def from_extended(
        cls,
        transfer: ExtendedSiegertTransfer,
        *,
        domain: SmoothDomain,
        kx: int = 3,
        ky: int = 3,
        smoothing: float = 0.0,
    ) -> "SmoothSiegertTransfer":
        return cls(
            transfer.mu_axis,
            transfer.sigma_axis,
            transfer.log_integral_table,
            domain=domain,
            kx=kx,
            ky=ky,
            smoothing=smoothing,
        )

    def support_mask(self, mu_mv: np.ndarray, sigma_mv: np.ndarray) -> np.ndarray:
        mu, sigma = np.broadcast_arrays(
            np.asarray(mu_mv, dtype=float), np.asarray(sigma_mv, dtype=float)
        )
        return (
            np.isfinite(mu)
            & np.isfinite(sigma)
            & (mu >= self.domain.mu_min_mv)
            & (mu <= self.domain.mu_max_mv)
            & (sigma >= self.domain.sigma_min_mv)
            & (sigma <= self.domain.sigma_max_mv)
        )

    def log_integral_and_derivatives(
        self, mu_mv: np.ndarray, sigma_mv: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu, sigma = np.broadcast_arrays(
            np.asarray(mu_mv, dtype=float), np.asarray(sigma_mv, dtype=float)
        )
        shape = mu.shape
        valid = self.support_mask(mu, sigma)
        log_i = np.full(shape, np.nan, dtype=float)
        d_mu = np.full(shape, np.nan, dtype=float)
        d_sigma = np.full(shape, np.nan, dtype=float)
        if np.any(valid):
            x = mu[valid]
            y = sigma[valid]
            log_i[valid] = self._spline.ev(x, y)
            d_mu[valid] = self._spline.ev(x, y, dx=1, dy=0)
            d_sigma[valid] = self._spline.ev(x, y, dx=0, dy=1)
        return log_i, d_mu, d_sigma

    def rate_with_derivatives(
        self, mu_mv: np.ndarray, sigma_mv: np.ndarray, pop: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_i, dlog_dmu, dlog_dsigma = self.log_integral_and_derivatives(mu_mv, sigma_mv)
        tau_m, tau_ref = _population_constants(pop)
        log_q = log(tau_m) + LOG_SQRT_PI + log_i
        log_denominator = np.logaddexp(log(tau_ref), log_q)
        rate = np.exp(-log_denominator)
        integral_weight = np.exp(log_q - log_denominator)
        common = -rate * integral_weight
        return rate, common * dlog_dmu, common * dlog_dsigma

    def rate(self, mu_mv: np.ndarray, sigma_mv: np.ndarray, pop: str) -> np.ndarray:
        return self.rate_with_derivatives(mu_mv, sigma_mv, pop)[0]


def _population_constants(pop: str) -> tuple[float, float]:
    if pop == "E":
        return float(TAU_ME), float(TREF_E)
    if pop == "I":
        return float(TAU_MI), float(TREF_I)
    raise ValueError("pop must be E or I")


def _signed_scaled_pair(first_weight: float, first_log: float, second_weight: float, second_log: float) -> float:
    """Evaluate ``w1*exp(l1)+w2*exp(l2)`` without exponential overflow."""

    scale = max(first_log, second_log)
    return float(
        np.exp(scale)
        * (
            first_weight * np.exp(first_log - scale)
            + second_weight * np.exp(second_log - scale)
        )
    )


def exact_siegert_rate_derivatives_scalar(
    mu_mv: float, sigma_mv: float, pop: str
) -> tuple[float, float, float]:
    """Direct log-quadrature Siegert rate and analytic moving-boundary slopes."""

    if not np.isfinite(mu_mv) or not np.isfinite(sigma_mv) or sigma_mv <= 0.0:
        raise ValueError("exact transfer input must be finite with sigma>0")
    tau_m, tau_ref = _population_constants(pop)
    log_i = stable_siegert_log_integral(float(mu_mv), float(sigma_mv))
    a = (V_RESET - float(mu_mv)) / float(sigma_mv)
    b = (V_TH - float(mu_mv)) / float(sigma_mv)
    log_fa = _log_erfcx_negative_argument(a)
    log_fb = _log_erfcx_negative_argument(b)
    if not log_fb >= log_fa:
        raise FloatingPointError("Siegert integrand monotonicity failed")
    delta = log_fa - log_fb
    if delta < -36.0:
        log_fb_minus_fa = log_fb
    else:
        log_fb_minus_fa = log_fb + float(np.log(-np.expm1(delta)))
    dlog_i_dmu = -float(np.exp(log_fb_minus_fa - log_i) / sigma_mv)
    scale = max(log_fa, log_fb)
    scaled_sigma_numerator = (
        a * np.exp(log_fa - scale) - b * np.exp(log_fb - scale)
    )
    dlog_i_dsigma = float(
        np.exp(scale - log_i) * scaled_sigma_numerator / sigma_mv
    )
    log_q = log(tau_m) + LOG_SQRT_PI + log_i
    log_denominator = float(np.logaddexp(log(tau_ref), log_q))
    rate = float(np.exp(-log_denominator))
    integral_weight = float(np.exp(log_q - log_denominator))
    common = -rate * integral_weight
    return rate, common * dlog_i_dmu, common * dlog_i_dsigma


def exact_siegert_rate_derivatives(
    mu_mv: np.ndarray, sigma_mv: np.ndarray, pop: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vector wrapper around the direct scalar exact-transfer audit."""

    mu, sigma = np.broadcast_arrays(
        np.asarray(mu_mv, dtype=float), np.asarray(sigma_mv, dtype=float)
    )
    rate = np.empty(mu.shape, dtype=float)
    d_mu = np.empty(mu.shape, dtype=float)
    d_sigma = np.empty(mu.shape, dtype=float)
    for index in np.ndindex(mu.shape):
        rate[index], d_mu[index], d_sigma[index] = exact_siegert_rate_derivatives_scalar(
            float(mu[index]), float(sigma[index]), pop
        )
    return rate, d_mu, d_sigma


def recruitment_sensor_derivative(rate_khz: float) -> float:
    """Derivative of the locked quadratic Stage-0C recruitment sensor."""

    excess = float(rate_khz) - E0_KHZ
    if excess <= 0.0:
        return 0.0
    denominator = E50_KHZ**2 + excess**2
    return float(2.0 * excess * E50_KHZ**2 / denominator**2)


def smooth_rhs(
    state: np.ndarray,
    prepared: PreparedPoolParameters,
    transfer: SmoothSiegertTransfer,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Evaluate the unchanged Stage-0C dynamic-pool RHS."""

    return _rhs_and_moments(
        np.asarray(state, dtype=float),
        prepared,
        transfer,  # type: ignore[arg-type]
        mechanism="dynamic",
        clamp_s=None,
        subtractive_beta_mv=None,
    )


def analytic_rhs_jacobian(
    state: np.ndarray,
    params: PoolParameters,
    transfer: SmoothSiegertTransfer,
) -> np.ndarray:
    """Chain-rule Jacobian of the smooth Stage-0C RHS at one state."""

    state = np.asarray(state, dtype=float)
    if state.shape != (9,) or bool(_natural_state_bad(state)):
        raise ValueError("analytic Jacobian requires one natural nine-state vector")
    params.validate()
    prepared = prepare_pool_parameters([params])
    mu_e, sigma_e, mu_i, sigma_i, _ = moments_from_prepared(state[None, :], prepared)
    _, d_e_mu, d_e_sigma = transfer.rate_with_derivatives(mu_e, sigma_e, "E")
    _, d_i_mu, d_i_sigma = transfer.rate_with_derivatives(mu_i, sigma_i, "I")
    if not all(np.isfinite(value[0]) for value in (d_e_mu, d_e_sigma, d_i_mu, d_i_sigma)):
        raise FloatingPointError("smooth transfer derivative left locked support")

    alpha = float(params.alpha_g)
    divisor = 1.0 + alpha * float(state[8])
    w_ee = float(prepared.w_ee[0])
    w_ei = float(params.z * W_EI)

    mu_e_gradient = np.zeros(9, dtype=float)
    var_e_gradient = np.zeros(9, dtype=float)
    recurrent_mean_coefficient = TAU_ME * C_EE * w_ee
    recurrent_variance_coefficient = TAU_ME * C_EE * w_ee**2
    mu_e_gradient[2] = recurrent_mean_coefficient / divisor
    mu_e_gradient[3] = -TAU_ME * C_EI * w_ei
    mu_e_gradient[8] = (
        -recurrent_mean_coefficient * float(state[2]) * alpha / divisor**2
    )
    var_e_gradient[2] = recurrent_variance_coefficient / divisor**2
    var_e_gradient[3] = TAU_ME * C_EI * w_ei**2
    var_e_gradient[8] = (
        -2.0 * recurrent_variance_coefficient * float(state[2]) * alpha / divisor**3
    )
    sigma_e_gradient = var_e_gradient / (2.0 * float(sigma_e[0]))

    mu_i_gradient = np.zeros(9, dtype=float)
    var_i_gradient = np.zeros(9, dtype=float)
    mu_i_gradient[4] = TAU_MI * C_IE * W_IE
    mu_i_gradient[5] = -TAU_MI * C_II * W_II
    var_i_gradient[4] = TAU_MI * C_IE * W_IE**2
    var_i_gradient[5] = TAU_MI * C_II * W_II**2
    sigma_i_gradient = var_i_gradient / (2.0 * float(sigma_i[0]))

    jacobian = np.zeros((9, 9), dtype=float)
    jacobian[0] = (
        float(d_e_mu[0]) * mu_e_gradient
        + float(d_e_sigma[0]) * sigma_e_gradient
    ) / TAU_ME
    jacobian[0, 0] -= 1.0 / TAU_ME
    jacobian[1] = (
        float(d_i_mu[0]) * mu_i_gradient
        + float(d_i_sigma[0]) * sigma_i_gradient
    ) / TAU_MI
    jacobian[1, 1] -= 1.0 / TAU_MI

    jacobian[2, 0] = 1.0 / TAU_AMPA
    jacobian[2, 2] = -1.0 / TAU_AMPA
    jacobian[3, 1] = 1.0 / TAU_GABA
    jacobian[3, 3] = -1.0 / TAU_GABA
    jacobian[4, 0] = 1.0 / TAU_AMPA
    jacobian[4, 4] = -1.0 / TAU_AMPA
    jacobian[5, 1] = 1.0 / TAU_GABA
    jacobian[5, 5] = -1.0 / TAU_GABA
    jacobian[6, 0] = 1.0 / TAU_FAST_MS
    jacobian[6, 6] = -1.0 / TAU_FAST_MS
    jacobian[7, 6] = recruitment_sensor_derivative(float(state[6])) / TAU_MU_MS
    jacobian[7, 7] = -1.0 / TAU_MU_MS
    jacobian[8, 7] = S_MAX / TAU_S_MS
    jacobian[8, 8] = -1.0 / TAU_S_MS
    if not np.all(np.isfinite(jacobian)):
        raise FloatingPointError("analytic RHS Jacobian is nonfinite")
    return jacobian


def centered_rhs_jacobian(
    state: np.ndarray,
    params: PoolParameters,
    transfer: SmoothSiegertTransfer,
    *,
    scales: np.ndarray,
    relative_step: float,
    absolute_floor: float,
) -> np.ndarray:
    """Central-state-difference RHS Jacobian of the same smooth vector field."""

    state = np.asarray(state, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if state.shape != (9,) or scales.shape != (9,) or np.any(scales <= 0.0):
        raise ValueError("invalid centered Jacobian state or scales")
    if relative_step <= 0.0 or absolute_floor <= 0.0:
        raise ValueError("centered Jacobian steps must be positive")
    steps = np.maximum(float(absolute_floor), float(relative_step) * scales)
    perturbed = np.repeat(state[None, :], 18, axis=0)
    for coordinate in range(9):
        perturbed[2 * coordinate, coordinate] += steps[coordinate]
        perturbed[2 * coordinate + 1, coordinate] -= steps[coordinate]
    if np.any(_natural_state_bad(perturbed)):
        raise FloatingPointError("centered RHS perturbation left natural bounds")
    rhs, _ = smooth_rhs(
        perturbed,
        prepare_pool_parameters([params] * 18),
        transfer,
    )
    jacobian = np.empty((9, 9), dtype=float)
    for coordinate in range(9):
        jacobian[:, coordinate] = (
            rhs[2 * coordinate] - rhs[2 * coordinate + 1]
        ) / (2.0 * steps[coordinate])
    if not np.all(np.isfinite(jacobian)):
        raise FloatingPointError("centered RHS Jacobian is nonfinite")
    return jacobian


def normalized_frobenius_difference(
    left: np.ndarray, right: np.ndarray, *, norm_floor: float
) -> float:
    """Symmetric matrix difference with an explicit near-zero denominator floor."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 2 or norm_floor <= 0.0:
        raise ValueError("invalid matrix-difference input")
    denominator = max(
        float(np.linalg.norm(left, ord="fro")),
        float(np.linalg.norm(right, ord="fro")),
        float(norm_floor),
    )
    return float(np.linalg.norm(left - right, ord="fro") / denominator)


def _event_located_tangent(
    state_before: np.ndarray,
    state_after: np.ndarray,
    tangent_before: np.ndarray,
    tangent_after: np.ndarray,
    *,
    section: SectionDefinition,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Differentiate the exact linear interpolation used to locate a crossing."""

    before = np.asarray(state_before, dtype=float)
    after = np.asarray(state_after, dtype=float)
    a_before = np.asarray(tangent_before, dtype=float)
    a_after = np.asarray(tangent_after, dtype=float)
    if before.shape != (9,) or after.shape != (9,) or a_before.shape != (9, 8) or a_after.shape != (9, 8):
        raise ValueError("event tangent shapes are invalid")
    h0 = float(before[section.index] - section.level)
    h1 = float(after[section.index] - section.level)
    if not h0 < 0.0 <= h1:
        raise ValueError("event tangent requires an upward bracket")
    delta = after - before
    denominator = float(delta[section.index])
    fraction = float(-h0 / denominator)
    delta_a = a_after - a_before
    d_fraction = (
        -a_before[section.index] * denominator
        - (section.level - before[section.index]) * delta_a[section.index]
    ) / denominator**2
    crossing = before + fraction * delta
    tangent = a_before + fraction * delta_a + delta[:, None] * d_fraction[None, :]
    crossing[section.index] = section.level
    return fraction, crossing, tangent


def discrete_variational_poincare(
    fixed_state: np.ndarray,
    params: PoolParameters,
    transfer: SmoothSiegertTransfer,
    *,
    dt_ms: float,
    section: SectionDefinition,
    scales: np.ndarray,
    centered_relative_steps: Sequence[float],
    centered_absolute_floor: float,
) -> dict[str, Any]:
    """Propagate three tangent maps along one nominal event-located return."""

    fixed = np.asarray(fixed_state, dtype=float).copy()
    scales = np.asarray(scales, dtype=float)
    relative_steps = tuple(float(value) for value in centered_relative_steps)
    if relative_steps != (1.0e-5, 3.0e-6):
        raise ValueError("Stage0F centered derivative steps drifted")
    if fixed.shape != (9,) or scales.shape != (9,) or dt_ms <= 0.0:
        raise ValueError("invalid variational Poincare input")
    fixed[section.index] = section.level
    initial_tangent = np.zeros((9, 8), dtype=float)
    for column, coordinate in enumerate(SECTION_COORDINATES):
        initial_tangent[coordinate, column] = scales[coordinate]
    tangents = {
        "chain_rule": initial_tangent.copy(),
        "centered_1e-5": initial_tangent.copy(),
        "centered_3e-6": initial_tangent.copy(),
    }
    state = fixed
    prepared = prepare_pool_parameters([params])
    below_seen = False
    local_differences: list[list[float]] = []
    max_steps = int(np.ceil(section.max_return_ms / dt_ms)) + 2
    identity = np.eye(9)

    for step in range(max_steps):
        rhs_batch, moments = smooth_rhs(state[None, :], prepared, transfer)
        rhs = rhs_batch[0]
        mu_e, sigma_e, mu_i, sigma_i, _ = moments
        support = bool(
            transfer.support_mask(mu_e, sigma_e)[0]
            and transfer.support_mask(mu_i, sigma_i)[0]
        )
        if (
            not support
            or bool(_natural_state_bad(state))
            or state[0] >= FINITE_HIGH_MAX_KHZ
            or not np.all(np.isfinite(rhs))
        ):
            return {"valid": False, "reason": "nominal_orbit_physical_or_support_failure"}
        chain_jac = analytic_rhs_jacobian(state, params, transfer)
        centered_jacs = [
            centered_rhs_jacobian(
                state,
                params,
                transfer,
                scales=scales,
                relative_step=relative,
                absolute_floor=centered_absolute_floor,
            )
            for relative in relative_steps
        ]
        local_differences.append(
            [
                normalized_frobenius_difference(chain_jac, centered, norm_floor=1e-12)
                for centered in centered_jacs
            ]
        )
        jacobians = {
            "chain_rule": chain_jac,
            "centered_1e-5": centered_jacs[0],
            "centered_3e-6": centered_jacs[1],
        }
        next_state = state + dt_ms * rhs
        next_tangents = {
            label: (identity + dt_ms * jacobians[label]) @ tangent
            for label, tangent in tangents.items()
        }
        h0 = float(state[section.index] - section.level)
        h1 = float(next_state[section.index] - section.level)
        if h0 < 0.0:
            below_seen = True
        if below_seen and h0 < 0.0 <= h1:
            crossing_tangents: dict[str, np.ndarray] = {}
            crossing = None
            fraction = None
            for label in DERIVATIVE_LABELS:
                current_fraction, current_crossing, tangent = _event_located_tangent(
                    state,
                    next_state,
                    tangents[label],
                    next_tangents[label],
                    section=section,
                )
                if crossing is None:
                    crossing = current_crossing
                    fraction = current_fraction
                elif not np.allclose(crossing, current_crossing, atol=1e-14, rtol=0.0):
                    raise RuntimeError("derivative constructions disagree on nominal crossing")
                crossing_tangents[label] = tangent
            assert crossing is not None and fraction is not None
            crossing_rhs, _ = smooth_rhs(crossing[None, :], prepared, transfer)
            transversality = float(crossing_rhs[0, section.index])
            matrices: dict[str, np.ndarray] = {}
            full_tangents: dict[str, np.ndarray] = {}
            section_row_max: dict[str, float] = {}
            multipliers: dict[str, np.ndarray] = {}
            radii: dict[str, float] = {}
            transverse_scales = scales[SECTION_COORDINATES]
            for label, tangent in crossing_tangents.items():
                full_normalized = tangent / scales[:, None]
                matrix = tangent[SECTION_COORDINATES] / transverse_scales[:, None]
                eigenvalues = np.linalg.eigvals(matrix)
                matrices[label] = matrix
                full_tangents[label] = full_normalized
                section_row_max[label] = float(np.max(np.abs(full_normalized[SECTION_INDEX])))
                multipliers[label] = eigenvalues
                radii[label] = float(np.max(np.abs(eigenvalues)))
            local = np.asarray(local_differences, dtype=float)
            return {
                "valid": bool(
                    np.all(np.isfinite(crossing))
                    and np.isfinite(transversality)
                    and all(np.all(np.isfinite(value)) for value in matrices.values())
                ),
                "period_ms": float((step + fraction) * dt_ms),
                "crossing_state": crossing,
                "crossing_fraction": fraction,
                "transversality_per_ms": transversality,
                "n_euler_steps": int(step + 1),
                "poincare_matrices": matrices,
                "full_event_tangents_normalized": full_tangents,
                "multipliers": multipliers,
                "spectral_radii": radii,
                "section_row_max_abs": section_row_max,
                "local_rhs_jacobian_chain_centered_max_relative": np.max(local, axis=0),
                "local_rhs_jacobian_chain_centered_median_relative": np.median(local, axis=0),
            }
        state = next_state
        tangents = next_tangents
    return {"valid": False, "reason": "no_upward_return_within_locked_window"}


def shooting_summary(
    shooting: Mapping[str, Any],
    cycle: Mapping[str, Any] | None,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the locked smooth-orbit shooting gates."""

    residual = np.asarray(shooting.get("residual", []), dtype=float)
    periods = np.asarray(shooting.get("period_ms", []), dtype=float)
    tail = periods[-4:]
    cv = float(np.std(tail) / np.mean(tail)) if tail.size == 4 and np.mean(tail) > 0 else np.inf
    cycle_valid = bool(cycle is not None and cycle.get("valid", False))
    closure = float(cycle.get("closure_residual", np.inf)) if cycle is not None else np.inf
    second = float(cycle.get("second_closure_residual", np.inf)) if cycle is not None else np.inf
    aligned = float(cycle.get("aligned_cycle_residual", np.inf)) if cycle is not None else np.inf
    tolerance = float(cfg["residual_tolerance"])
    checks = {
        "shooting_converged": bool(shooting.get("converged", False)),
        "fixed_point_residual": bool(residual.size and residual[-1] <= tolerance),
        "period_cv": bool(cv <= float(cfg["period_cv_tolerance"])),
        "cycle_valid": cycle_valid,
        "first_closure": bool(closure <= tolerance),
        "second_closure": bool(second <= tolerance),
        "two_cycle_aligned": bool(aligned <= float(cfg["aligned_cycle_residual_tolerance"])),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "residual_series": residual,
        "period_series_ms": periods,
        "last_four_period_cv": cv if np.isfinite(cv) else None,
        "closure_residual": closure if np.isfinite(closure) else None,
        "second_closure_residual": second if np.isfinite(second) else None,
        "aligned_cycle_residual": aligned if np.isfinite(aligned) else None,
    }


def lut_orbit_parity_summary(
    smooth_cycle: Mapping[str, Any],
    lut_trace: Mapping[str, np.ndarray],
    scales: np.ndarray,
    cfg: Mapping[str, float],
    *,
    n_phase: int,
) -> dict[str, Any]:
    """Compare a smooth orbit to the corresponding immutable Stage-0E orbit."""

    crossing_times = np.asarray(lut_trace["crossing_time_ms"], dtype=float)
    if crossing_times.size < 1:
        return {"pass": False, "reason": "Stage0E trace has no complete cycle"}
    lut_period = float(crossing_times[0])
    lut_waveform = phase_resample(
        lut_trace["time_ms"], lut_trace["state"], 0.0, lut_period, n_phase
    )
    smooth_period = float(np.asarray(smooth_cycle["period_ms"])[0])
    smooth_waveform = np.asarray(smooth_cycle["waveform_first"], dtype=float)
    residual = aligned_waveform_residual(smooth_waveform, lut_waveform, scales)
    difference = abs(smooth_period - lut_period)
    return {
        "pass": bool(
            difference <= float(cfg["period_abs_ms"])
            and residual <= float(cfg["aligned_waveform_residual"])
        ),
        "smooth_period_ms": smooth_period,
        "lut_period_ms": lut_period,
        "period_difference_ms": difference,
        "aligned_waveform_residual": residual,
    }


def smooth_dt_parity_summary(
    base_cycle: Mapping[str, Any],
    half_cycle: Mapping[str, Any],
    scales: np.ndarray,
    cfg: Mapping[str, float],
) -> dict[str, Any]:
    """Compare independently shot smooth base/half discrete orbits."""

    base_period = float(np.mean(np.asarray(base_cycle["period_ms"], dtype=float)))
    half_period = float(np.mean(np.asarray(half_cycle["period_ms"], dtype=float)))
    difference = abs(base_period - half_period)
    tolerance = max(
        float(cfg["dt_period_abs_ms"]),
        float(cfg["dt_period_relative"]) * abs(base_period),
    )
    waveform = aligned_waveform_residual(
        base_cycle["waveform_first"], half_cycle["waveform_first"], scales
    )
    return {
        "pass": bool(
            difference <= tolerance
            and waveform <= float(cfg["dt_aligned_waveform_residual"])
        ),
        "base_period_ms": base_period,
        "half_period_ms": half_period,
        "period_difference_ms": difference,
        "period_tolerance_ms": tolerance,
        "aligned_waveform_residual": waveform,
    }


def transfer_parity_rows(
    cycle: Mapping[str, Any],
    params: PoolParameters,
    transfer: SmoothSiegertTransfer,
    *,
    dt_ms: float,
    n_phase: int,
) -> list[dict[str, Any]]:
    """Audit smooth values/slopes against direct exact Siegert on one orbit."""

    trace = cycle["trace"]
    period = float(np.asarray(cycle["period_ms"])[0])
    states = phase_resample(trace["time_ms"], trace["state"], 0.0, period, n_phase)
    moments = moments_from_prepared(states, prepare_pool_parameters([params] * n_phase))
    mu_e, sigma_e, mu_i, sigma_i, _ = moments
    rows: list[dict[str, Any]] = []
    for pop, mu, sigma in (("E", mu_e, sigma_e), ("I", mu_i, sigma_i)):
        smooth = transfer.rate_with_derivatives(mu, sigma, pop)
        exact = exact_siegert_rate_derivatives(mu, sigma, pop)
        for phase_index in range(n_phase):
            row: dict[str, Any] = {
                "z": float(params.z),
                "alpha_G": float(params.alpha_g),
                "dt_ms": float(dt_ms),
                "phase_index": int(phase_index),
                "phase_fraction": float(phase_index / (n_phase - 1)),
                "population": pop,
                "mu_mv": float(mu[phase_index]),
                "sigma_mv": float(sigma[phase_index]),
            }
            for label, smooth_values, exact_values in zip(
                ("rate_khz", "d_rate_d_mu_khz_per_mv", "d_rate_d_sigma_khz_per_mv"),
                smooth,
                exact,
            ):
                observed = float(smooth_values[phase_index])
                target = float(exact_values[phase_index])
                row[f"smooth_{label}"] = observed
                row[f"exact_{label}"] = target
                row[f"absolute_error_{label}"] = abs(observed - target)
            rows.append(row)
    return rows


def transfer_parity_summary(
    rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, float]
) -> dict[str, Any]:
    """Apply locked exact-value and exact-derivative transfer gates."""

    rows = list(rows)
    if not rows:
        return {"pass": False, "reason": "no_transfer_parity_rows"}
    by_population: dict[str, Any] = {}
    all_pass = True
    definitions = (
        (
            "rate_khz",
            float(cfg["rate_absolute_khz"]),
            float(cfg["rate_relative"]),
            float(cfg["rate_relative_floor_khz"]),
        ),
        (
            "d_rate_d_mu_khz_per_mv",
            float(cfg["derivative_absolute_khz_per_mv"]),
            float(cfg["derivative_relative"]),
            float(cfg["derivative_relative_floor_khz_per_mv"]),
        ),
        (
            "d_rate_d_sigma_khz_per_mv",
            float(cfg["derivative_absolute_khz_per_mv"]),
            float(cfg["derivative_relative"]),
            float(cfg["derivative_relative_floor_khz_per_mv"]),
        ),
    )
    for pop in ("E", "I"):
        members = [row for row in rows if row["population"] == pop]
        pop_metrics: dict[str, Any] = {}
        pop_pass = bool(members)
        for label, abs_tolerance, rel_tolerance, rel_floor in definitions:
            observed = np.asarray([float(row[f"smooth_{label}"]) for row in members])
            exact = np.asarray([float(row[f"exact_{label}"]) for row in members])
            absolute = np.abs(observed - exact)
            relative = absolute / np.maximum(np.abs(exact), rel_floor)
            metric_pass = bool(
                np.all(np.isfinite(observed))
                and np.all(np.isfinite(exact))
                and float(np.max(absolute)) <= abs_tolerance
                and float(np.max(relative)) <= rel_tolerance
            )
            pop_pass &= metric_pass
            pop_metrics[label] = {
                "pass": metric_pass,
                "maximum_absolute_error": float(np.max(absolute)),
                "maximum_relative_error_with_floor": float(np.max(relative)),
                "absolute_tolerance": abs_tolerance,
                "relative_tolerance": rel_tolerance,
                "relative_floor": rel_floor,
            }
        by_population[pop] = {"pass": pop_pass, "metrics": pop_metrics}
        all_pass &= pop_pass
    return {"pass": bool(all_pass), "populations": by_population, "n_rows": len(rows)}


def variational_consistency_summary(
    result: Mapping[str, Any], cfg: Mapping[str, float]
) -> dict[str, Any]:
    """Apply same-dt agreement and event-sensitivity gates."""

    if not bool(result.get("valid", False)):
        return {"pass": False, "reason": result.get("reason", "invalid_variational_result")}
    matrices = result["poincare_matrices"]
    floor = float(cfg["matrix_norm_floor"])
    comparisons = {
        "centered_ladder": normalized_frobenius_difference(
            matrices["centered_1e-5"], matrices["centered_3e-6"], norm_floor=floor
        ),
        "chain_vs_centered_1e-5": normalized_frobenius_difference(
            matrices["chain_rule"], matrices["centered_1e-5"], norm_floor=floor
        ),
        "chain_vs_centered_3e-6": normalized_frobenius_difference(
            matrices["chain_rule"], matrices["centered_3e-6"], norm_floor=floor
        ),
    }
    radii = np.asarray([result["spectral_radii"][label] for label in DERIVATIVE_LABELS])
    section_row = max(float(result["section_row_max_abs"][label]) for label in DERIVATIVE_LABELS)
    checks = {
        "matrix_agreement": bool(
            max(comparisons.values()) <= float(cfg["matrix_relative_difference_max"])
        ),
        "spectral_radius_agreement": bool(
            float(np.ptp(radii)) <= float(cfg["spectral_radius_range_max"])
        ),
        "event_section_row": bool(section_row <= float(cfg["section_row_abs_max"])),
        "event_transversality": bool(
            float(result["transversality_per_ms"])
            >= float(cfg["minimum_transversality_per_ms"])
        ),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "matrix_relative_differences": comparisons,
        "spectral_radii": {label: float(result["spectral_radii"][label]) for label in DERIVATIVE_LABELS},
        "spectral_radius_range": float(np.ptp(radii)),
        "maximum_normalized_section_row": section_row,
        "transversality_per_ms": float(result["transversality_per_ms"]),
        "matrix_norm_floor": floor,
    }


def match_multipliers(left: np.ndarray, right: np.ndarray) -> list[dict[str, Any]]:
    """Diagnostic minimum-cost matching of two complex multiplier sets."""

    left = np.asarray(left, dtype=complex)
    right = np.asarray(right, dtype=complex)
    if left.shape != (8,) or right.shape != (8,):
        raise ValueError("multiplier matching requires two length-eight vectors")
    rows, columns = linear_sum_assignment(np.abs(left[:, None] - right[None, :]))
    return [
        {
            "left_index": int(i),
            "right_index": int(j),
            "left_real": float(left[i].real),
            "left_imag": float(left[i].imag),
            "right_real": float(right[j].real),
            "right_imag": float(right[j].imag),
            "complex_distance": float(abs(left[i] - right[j])),
        }
        for i, j in zip(rows, columns)
    ]


def stability_certificate_summary(
    base: Mapping[str, Any],
    half: Mapping[str, Any],
    cfg: Mapping[str, float],
) -> dict[str, Any]:
    """Apply the locked cross-method and cross-dt unit-circle margin."""

    if not bool(base.get("valid", False)) or not bool(half.get("valid", False)):
        return {"pass": False, "reason": "invalid_variational_result"}
    base_rho = {label: float(base["spectral_radii"][label]) for label in DERIVATIVE_LABELS}
    half_rho = {label: float(half["spectral_radii"][label]) for label in DERIVATIVE_LABELS}
    method_spread = max(
        float(np.ptp(list(base_rho.values()))),
        float(np.ptp(list(half_rho.values()))),
    )
    dt_differences = {label: abs(base_rho[label] - half_rho[label]) for label in DERIVATIVE_LABELS}
    dt_spread = max(dt_differences.values())
    rho_max = max(*base_rho.values(), *half_rho.values())
    margin = 1.0 - rho_max
    required = max(
        float(cfg["minimum_unit_circle_margin"]),
        float(cfg["uncertainty_multiplier"]) * method_spread,
        float(cfg["uncertainty_multiplier"]) * dt_spread,
    )
    matches = {
        label: match_multipliers(base["multipliers"][label], half["multipliers"][label])
        for label in DERIVATIVE_LABELS
    }
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
        "multiplier_matching_base_vs_half": matches,
    }


def run_point_certificate(
    params: PoolParameters,
    transfer: SmoothSiegertTransfer,
    stage0e_inputs: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the complete locked Stage-0F audit for one parameter point."""

    params.validate()
    if (float(params.z), float(params.alpha_g)) not in LOCKED_POINTS:
        raise ValueError("Stage0F parameter point drifted")
    scales = np.asarray(stage0e_inputs["scales"], dtype=float)
    if scales.shape != (9,) or np.any(scales <= 0.0):
        raise ValueError("Stage0E state scales are invalid")
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
    artifacts: dict[str, Any] = {}
    cycles: dict[str, Any] = {}
    shootings: dict[str, Any] = {}
    parity_rows: list[dict[str, Any]] = []
    variational: dict[str, Any] = {}
    time_steps = tuple(float(value) for value in cfg["time_steps_ms"])
    if time_steps != (0.125, 0.0625):
        raise ValueError("Stage0F time steps drifted")

    for label, dt_ms in zip(("base", "half"), time_steps):
        seed = np.asarray(stage0e_inputs[f"{label}_shooting_seed"], dtype=float)
        shooting = poincare_fixed_point_shooting(
            seed,
            params,
            transfer,  # type: ignore[arg-type]
            dt_ms=dt_ms,
            section=section,
            scales=scales,
            max_iterations=int(cfg["shooting"]["max_iterations"]),
            residual_tolerance=float(cfg["shooting"]["residual_tolerance"]),
            minimum_iterations=int(cfg["shooting"]["minimum_iterations"]),
        )
        shootings[label] = shooting
        cycle = None
        if bool(shooting.get("converged", False)):
            cycle = shooting_cycle_validation(
                shooting["fixed_state"],
                params,
                transfer,  # type: ignore[arg-type]
                dt_ms=dt_ms,
                section=section,
                scales=scales,
                n_phase=int(cfg["phase_bins"]),
            )
        gate = shooting_summary(shooting, cycle, cfg["shooting"])
        result[f"{label}_smooth_shooting"] = gate
        if not gate["pass"]:
            result["failed_gates"].append(f"{label}_smooth_shooting")
            artifacts["shootings"] = shootings
            return result, artifacts
        assert cycle is not None
        cycles[label] = cycle
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
            artifacts.update({"shootings": shootings, "cycles": cycles})
            return result, artifacts
        parity_rows.extend(
            transfer_parity_rows(
                cycle,
                params,
                transfer,
                dt_ms=dt_ms,
                n_phase=int(cfg["transfer_parity_phase_samples"]),
            )
        )
        variational[label] = discrete_variational_poincare(
            shooting["fixed_state"],
            params,
            transfer,
            dt_ms=dt_ms,
            section=section,
            scales=scales,
            centered_relative_steps=cfg["variational"]["centered_relative_steps"],
            centered_absolute_floor=float(cfg["variational"]["centered_absolute_floor"]),
        )
        result[f"{label}_variational_consistency"] = variational_consistency_summary(
            variational[label], cfg["variational"]
        )

    result["smooth_dt_parity"] = smooth_dt_parity_summary(
        cycles["base"], cycles["half"], scales, cfg["orbit_parity"]
    )
    result["transfer_parity"] = transfer_parity_summary(parity_rows, cfg["transfer_parity"])
    result["stability_certificate"] = stability_certificate_summary(
        variational["base"], variational["half"], cfg["stability"]
    )
    gate_map = {
        "smooth_dt_parity": bool(result["smooth_dt_parity"]["pass"]),
        "exact_transfer_value_derivative_parity": bool(result["transfer_parity"]["pass"]),
        "base_variational_consistency": bool(result["base_variational_consistency"]["pass"]),
        "half_variational_consistency": bool(result["half_variational_consistency"]["pass"]),
        "floquet_stability_margin": bool(result["stability_certificate"]["pass"]),
    }
    result["failed_gates"].extend([name for name, passed in gate_map.items() if not passed])
    if not result["failed_gates"]:
        result["outcome"] = "stable_periodic_orbit_derivative_certified"
        result["derivative_certified"] = True
    artifacts.update(
        {
            "shootings": shootings,
            "cycles": cycles,
            "transfer_parity_rows": parity_rows,
            "variational": variational,
        }
    )
    return result, artifacts
