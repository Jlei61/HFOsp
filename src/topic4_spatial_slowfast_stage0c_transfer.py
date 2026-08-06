"""Numerically independent transfer-support audit for Stage 0C.

This module deliberately does not modify the primary Stage-0C implementation.
It evaluates the same nine-dimensional frozen fast system with an extended,
log-domain Siegert transfer whose support is explicit and never clipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erfcx, log_ndtr

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
    W_EE,
    W_EI,
    W_IE,
    W_II,
    nu_theta_pop,
)
from src.topic4_spatial_slowfast_stage0b import ForkClassifierThresholds, classify_rate_trace
from src.topic4_spatial_slowfast_stage0c import (
    E_CEILING_KHZ,
    FINITE_HIGH_MAX_KHZ,
    I_CEILING_KHZ,
    S_MAX,
    TAU_FAST_MS,
    TAU_MU_MS,
    TAU_S_MS,
    PoolParameters,
    moments_from_state,
    recruitment_sensor,
)


_LOG_TWO = log(2.0)
_LOG_SQRT_PI = 0.5 * log(np.pi)
_ENDPOINT_SWITCH = 6.0
_LOG_TAIL_CUTOFF = 80.0
_CANDIDATE_CLASSES = {"bounded_tonic_candidate", "bounded_oscillatory_candidate"}


@dataclass(frozen=True)
class TransferSupport:
    """Locked extended domain; values outside it are unresolved, never clipped."""

    mu_min_mv: float = -2500.0
    mu_core_min_mv: float = -250.0
    mu_max_mv: float = 120.0
    sigma_min_mv: float = 0.5
    sigma_max_mv: float = 50.0

    def validate(self) -> "TransferSupport":
        values = (
            self.mu_min_mv,
            self.mu_core_min_mv,
            self.mu_max_mv,
            self.sigma_min_mv,
            self.sigma_max_mv,
        )
        if not all(np.isfinite(values)):
            raise ValueError("transfer support values must be finite")
        if not self.mu_min_mv < self.mu_core_min_mv < self.mu_max_mv:
            raise ValueError("require mu_min < mu_core_min < mu_max")
        if not 0.0 < self.sigma_min_mv < self.sigma_max_mv:
            raise ValueError("require 0 < sigma_min < sigma_max")
        return self


@dataclass(frozen=True)
class TransferResolution:
    """Resolution of the dense core and geometric low-mu tail."""

    name: str
    mu_core_step_mv: float
    sigma_step_mv: float
    n_tail: int

    def validate(self) -> "TransferResolution":
        if self.name not in {"coarse", "fine", "extra_fine", "test"}:
            raise ValueError("resolution name must be coarse, fine, extra_fine, or test")
        if self.mu_core_step_mv <= 0.0 or self.sigma_step_mv <= 0.0 or self.n_tail < 2:
            raise ValueError("invalid transfer resolution")
        return self


COARSE_RESOLUTION = TransferResolution("coarse", 0.5, 0.25, 64)
FINE_RESOLUTION = TransferResolution("fine", 0.25, 0.125, 128)


def _log_erfcx_negative_argument(x: float) -> float:
    """Return log(erfcx(-x)) without overflow or cancellation."""

    if x <= 0.0:
        return float(np.log(erfcx(-x)))
    return float(x * x + _LOG_TWO + log_ndtr(sqrt(2.0) * x))


def stable_siegert_log_integral(
    mu_mv: float,
    sigma_mv: float,
    *,
    v_th_mv: float = V_TH,
    v_reset_mv: float = V_RESET,
) -> float:
    """Log of the Siegert integral using scaled quadrature.

    For large positive upper limits the integrand is exponentially concentrated
    at that endpoint.  Reparameterising distance from the endpoint avoids both
    overflow and the missed-boundary-layer failure of direct quadrature.  The
    optional truncation occurs only after a relative log drop of 80.
    """

    values = (mu_mv, sigma_mv, v_th_mv, v_reset_mv)
    if not all(np.isfinite(values)):
        raise ValueError("Siegert arguments must be finite")
    if sigma_mv <= 0.0 or v_th_mv < v_reset_mv:
        raise ValueError("require sigma>0 and threshold>=reset")
    if v_th_mv == v_reset_mv:
        return -np.inf
    lower = (v_reset_mv - mu_mv) / sigma_mv
    upper = (v_th_mv - mu_mv) / sigma_mv
    log_upper = _log_erfcx_negative_argument(upper)

    if upper > _ENDPOINT_SWITCH:
        scale = 2.0 * upper
        transformed_width = scale * (upper - lower)
        integration_limit = min(transformed_width, _LOG_TAIL_CUTOFF)

        def scaled_endpoint_integrand(distance: float) -> float:
            x = upper - distance / scale
            return np.exp(_log_erfcx_negative_argument(x) - log_upper) / scale

        scaled_integral, _ = quad(
            scaled_endpoint_integrand,
            0.0,
            integration_limit,
            epsabs=1e-13,
            epsrel=1e-11,
            limit=160,
        )
        scale_log = log_upper
    else:
        log_lower = _log_erfcx_negative_argument(lower)
        scale_log = max(log_lower, log_upper)
        scaled_integral, _ = quad(
            lambda x: np.exp(_log_erfcx_negative_argument(x) - scale_log),
            lower,
            upper,
            epsabs=1e-13,
            epsrel=1e-11,
            limit=160,
        )
    if not np.isfinite(scaled_integral) or scaled_integral <= 0.0:
        raise FloatingPointError("scaled Siegert integral is nonpositive or nonfinite")
    return float(scale_log + log(scaled_integral))


def stable_siegert_log_rate(
    mu_mv: float,
    sigma_mv: float,
    tau_m_ms: float,
    tau_ref_ms: float,
    *,
    v_th_mv: float = V_TH,
) -> float:
    """Numerically stable log firing rate in log(kHz)."""

    if tau_m_ms <= 0.0 or tau_ref_ms <= 0.0:
        raise ValueError("time constants must be positive")
    if v_th_mv == V_RESET:
        return float(-log(tau_ref_ms))
    log_integral = stable_siegert_log_integral(mu_mv, sigma_mv, v_th_mv=v_th_mv)
    log_integral_term = log(tau_m_ms) + _LOG_SQRT_PI + log_integral
    log_denominator = float(np.logaddexp(log(tau_ref_ms), log_integral_term))
    return -log_denominator


def stable_siegert_rate(
    mu_mv: float,
    sigma_mv: float,
    tau_m_ms: float,
    tau_ref_ms: float,
    *,
    v_th_mv: float = V_TH,
) -> float:
    """Stable exact-Siegert rate; zero is only IEEE exponential underflow."""

    return float(np.exp(stable_siegert_log_rate(mu_mv, sigma_mv, tau_m_ms, tau_ref_ms, v_th_mv=v_th_mv)))


def transfer_axes(
    support: TransferSupport,
    resolution: TransferResolution,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the locked irregular-mu and regular-sigma axes."""

    support.validate()
    resolution.validate()
    dense_mu = np.arange(
        support.mu_core_min_mv,
        support.mu_max_mv + 0.5 * resolution.mu_core_step_mv,
        resolution.mu_core_step_mv,
        dtype=float,
    )
    if dense_mu[-1] < support.mu_max_mv:
        dense_mu = np.r_[dense_mu, support.mu_max_mv]
    else:
        dense_mu[-1] = support.mu_max_mv
    tail_span = support.mu_core_min_mv - support.mu_min_mv
    tail_distance = np.geomspace(
        resolution.mu_core_step_mv,
        tail_span,
        resolution.n_tail,
        dtype=float,
    )
    tail_mu = support.mu_core_min_mv - tail_distance
    mu_axis = np.unique(np.r_[tail_mu, dense_mu])
    sigma_axis = np.arange(
        support.sigma_min_mv,
        support.sigma_max_mv + 0.5 * resolution.sigma_step_mv,
        resolution.sigma_step_mv,
        dtype=float,
    )
    if sigma_axis[-1] < support.sigma_max_mv:
        sigma_axis = np.r_[sigma_axis, support.sigma_max_mv]
    else:
        sigma_axis[-1] = support.sigma_max_mv
    if not np.isclose(mu_axis[0], support.mu_min_mv) or not np.isclose(mu_axis[-1], support.mu_max_mv):
        raise RuntimeError("mu axis failed to cover the locked support")
    if not np.isclose(sigma_axis[0], support.sigma_min_mv) or not np.isclose(sigma_axis[-1], support.sigma_max_mv):
        raise RuntimeError("sigma axis failed to cover the locked support")
    return mu_axis, sigma_axis


def build_log_integral_table(mu_axis: np.ndarray, sigma_axis: np.ndarray) -> np.ndarray:
    """Precompute the population-independent log Siegert integral."""

    mu_axis = np.asarray(mu_axis, dtype=float)
    sigma_axis = np.asarray(sigma_axis, dtype=float)
    table = np.empty((mu_axis.size, sigma_axis.size), dtype=np.float64)
    for mu_index, mu_mv in enumerate(mu_axis):
        for sigma_index, sigma_mv in enumerate(sigma_axis):
            table[mu_index, sigma_index] = stable_siegert_log_integral(float(mu_mv), float(sigma_mv))
    if not np.all(np.isfinite(table)):
        raise FloatingPointError("extended log-integral table contains nonfinite entries")
    return table


class ExtendedSiegertTransfer:
    """Bilinear log-integral transfer with explicit no-extrapolation support."""

    def __init__(
        self,
        mu_axis: np.ndarray,
        sigma_axis: np.ndarray,
        log_integral_table: np.ndarray,
        *,
        name: str,
    ) -> None:
        self.mu_axis = np.asarray(mu_axis, dtype=float)
        self.sigma_axis = np.asarray(sigma_axis, dtype=float)
        self.log_integral_table = np.asarray(log_integral_table, dtype=float)
        expected = (self.mu_axis.size, self.sigma_axis.size)
        if self.log_integral_table.shape != expected:
            raise ValueError(f"log integral table has shape {self.log_integral_table.shape}, expected {expected}")
        if np.any(np.diff(self.mu_axis) <= 0.0) or np.any(np.diff(self.sigma_axis) <= 0.0):
            raise ValueError("transfer axes must be strictly increasing")
        if not np.all(np.isfinite(self.log_integral_table)):
            raise ValueError("log integral table must be finite")
        self.name = str(name)
        self._interpolator = RegularGridInterpolator(
            (self.mu_axis, self.sigma_axis),
            self.log_integral_table,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    @classmethod
    def build(
        cls,
        support: TransferSupport,
        resolution: TransferResolution,
    ) -> "ExtendedSiegertTransfer":
        mu_axis, sigma_axis = transfer_axes(support, resolution)
        return cls(mu_axis, sigma_axis, build_log_integral_table(mu_axis, sigma_axis), name=resolution.name)

    def support_mask(self, mu_mv: np.ndarray, sigma_mv: np.ndarray) -> np.ndarray:
        mu_mv, sigma_mv = np.broadcast_arrays(np.asarray(mu_mv, dtype=float), np.asarray(sigma_mv, dtype=float))
        return (
            np.isfinite(mu_mv)
            & np.isfinite(sigma_mv)
            & (mu_mv >= self.mu_axis[0])
            & (mu_mv <= self.mu_axis[-1])
            & (sigma_mv >= self.sigma_axis[0])
            & (sigma_mv <= self.sigma_axis[-1])
        )

    def log_integral(self, mu_mv: np.ndarray, sigma_mv: np.ndarray) -> np.ndarray:
        mu_mv, sigma_mv = np.broadcast_arrays(np.asarray(mu_mv, dtype=float), np.asarray(sigma_mv, dtype=float))
        points = np.column_stack((mu_mv.ravel(), sigma_mv.ravel()))
        return np.asarray(self._interpolator(points), dtype=float).reshape(mu_mv.shape)

    def rate(self, mu_mv: np.ndarray, sigma_mv: np.ndarray, pop: str) -> np.ndarray:
        log_integral = self.log_integral(mu_mv, sigma_mv)
        if pop == "E":
            tau_m, tau_ref = TAU_ME, TREF_E
        elif pop == "I":
            tau_m, tau_ref = TAU_MI, TREF_I
        else:
            raise ValueError("pop must be E or I")
        log_integral_term = log(tau_m) + _LOG_SQRT_PI + log_integral
        with np.errstate(invalid="ignore"):
            log_denominator = np.logaddexp(log(tau_ref), log_integral_term)
        return np.exp(-log_denominator)


@dataclass(frozen=True)
class PreparedPoolParameters:
    """Validated, vectorized coefficients reused at every Euler step."""

    z: np.ndarray
    alpha_g: np.ndarray
    w_ee: np.ndarray
    nu_ext: np.ndarray

    @property
    def size(self) -> int:
        return int(self.z.size)


def prepare_pool_parameters(params: Sequence[PoolParameters]) -> PreparedPoolParameters:
    """Validate once, then materialize immutable arrays for the hot loop."""

    checked = [point.validate() for point in params]
    if not checked:
        raise ValueError("at least one PoolParameters object is required")
    arrays = PreparedPoolParameters(
        z=np.asarray([point.z for point in checked], dtype=float),
        alpha_g=np.asarray([point.alpha_g for point in checked], dtype=float),
        w_ee=np.asarray([point.w_ee_mult * W_EE for point in checked], dtype=float),
        nu_ext=np.asarray([point.ratio * nu_theta_pop() for point in checked], dtype=float),
    )
    for value in (arrays.z, arrays.alpha_g, arrays.w_ee, arrays.nu_ext):
        value.setflags(write=False)
    return arrays


def moments_from_prepared(
    state: np.ndarray,
    prepared: PreparedPoolParameters,
    *,
    mechanism: str = "dynamic",
    clamp_s: float | None = None,
    subtractive_beta_mv: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stage-0C moments with prevalidated arrays; algebra matches the primary code."""

    state = np.asarray(state, dtype=float)
    if state.ndim != 2 or state.shape != (prepared.size, 9):
        raise ValueError("state and prepared parameters must align")
    if mechanism not in {"dynamic", "instantaneous", "clamped", "matched_subtractive", "mean_only"}:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    if mechanism == "clamped":
        if clamp_s is None or not np.isfinite(clamp_s) or clamp_s < 0.0:
            raise ValueError("clamped mechanism requires finite clamp_s>=0")
        s_eff = np.full(prepared.size, float(clamp_s))
    elif mechanism == "instantaneous":
        s_eff = recruitment_sensor(state[:, 0])
    else:
        s_eff = state[:, 8]
    divisor = 1.0 + prepared.alpha_g * s_eff
    # A divergent ablation must fail closed for that fork, not abort the whole
    # audit.  NaN is an explicit invalid-state sentinel here; it is not a clip
    # or an extrapolated dynamical value.
    divisor = np.where(np.isfinite(divisor) & (divisor > 0.0), divisor, np.nan)

    s_ee, s_ei, s_ie, s_ii = (state[:, index] for index in range(2, 6))
    w_ei = prepared.z * W_EI
    recurrent_mean_e = TAU_ME * C_EE * prepared.w_ee * s_ee
    recurrent_var_e = TAU_ME * C_EE * prepared.w_ee**2 * s_ee
    if mechanism == "matched_subtractive":
        if subtractive_beta_mv is None or not np.isfinite(subtractive_beta_mv) or subtractive_beta_mv < 0.0:
            raise ValueError("matched_subtractive requires finite beta>=0")
        recurrent_mean_e = recurrent_mean_e - float(subtractive_beta_mv) * s_eff
        recurrent_var_effective = recurrent_var_e
    else:
        recurrent_mean_e = recurrent_mean_e / divisor
        recurrent_var_effective = recurrent_var_e if mechanism == "mean_only" else recurrent_var_e / divisor**2
    mu_e = recurrent_mean_e - TAU_ME * C_EI * w_ei * s_ei + TAU_ME * JX_E * prepared.nu_ext
    var_e = recurrent_var_effective + TAU_ME * C_EI * w_ei**2 * s_ei + TAU_ME * JX_E**2 * prepared.nu_ext
    mu_i = TAU_MI * (C_IE * W_IE * s_ie - C_II * W_II * s_ii) + TAU_MI * JX_I * prepared.nu_ext
    var_i = TAU_MI * (C_IE * W_IE**2 * s_ie + C_II * W_II**2 * s_ii) + TAU_MI * JX_I**2 * prepared.nu_ext
    return (
        mu_e,
        np.sqrt(np.maximum(var_e, 1e-9)),
        mu_i,
        np.sqrt(np.maximum(var_i, 1e-9)),
        s_eff,
    )


def _rhs_and_moments(
    state: np.ndarray,
    prepared: PreparedPoolParameters,
    transfer: ExtendedSiegertTransfer,
    *,
    mechanism: str,
    clamp_s: float | None,
    subtractive_beta_mv: float | None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    moments = moments_from_prepared(
        state,
        prepared,
        mechanism=mechanism,
        clamp_s=clamp_s,
        subtractive_beta_mv=subtractive_beta_mv,
    )
    mu_e, sigma_e, mu_i, sigma_i, _ = moments
    target_e = transfer.rate(mu_e, sigma_e, "E")
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
    return out, moments


def simulate_extended_forks(
    initial_states: np.ndarray,
    params: Sequence[PoolParameters],
    transfer: ExtendedSiegertTransfer,
    *,
    dt_ms: float,
    duration_ms: float,
    save_stride: int,
    audit_tail_fraction: float,
    mechanism: str = "dynamic",
    clamp_s: float | None = None,
    subtractive_beta_mv: float | None = None,
) -> dict[str, np.ndarray]:
    """Vectorized Euler integration with an audit at every state, including t=0."""

    state = np.asarray(initial_states, dtype=float).copy()
    if state.ndim != 2 or state.shape[1] != 9 or state.shape[0] != len(params):
        raise ValueError("initial_states and params must align as (n,9) and length n")
    if dt_ms <= 0.0 or duration_ms <= dt_ms or save_stride < 1:
        raise ValueError("invalid integration contract")
    prepared = prepare_pool_parameters(params)
    n_steps = int(round(duration_ms / dt_ms))
    if not np.isclose(n_steps * dt_ms, duration_ms):
        raise ValueError("duration must be an integer multiple of dt")
    sample_steps = np.arange(0, n_steps + 1, save_stride, dtype=int)
    if sample_steps[-1] != n_steps:
        sample_steps = np.r_[sample_steps, n_steps]
    n_forks = state.shape[0]
    trace_names = (
        "rE_khz",
        "rI_khz",
        "rE_fast_khz",
        "mu_G",
        "S_G",
        "muE_mV",
        "sigmaE_mV",
        "muI_mV",
        "sigmaI_mV",
        "divisor",
    )
    traces = {name: np.full((sample_steps.size, n_forks), np.nan, dtype=np.float32) for name in trace_names}
    count_names = (
        "support_violation",
        "pool_bound",
        "rate_bound",
        "synapse_bound",
        "negative_rate",
        "e_refractory",
        "i_refractory",
        "over_100hz",
    )
    counts = {name: np.zeros(n_forks, dtype=np.int64) for name in count_names}
    tail_counts = {name: np.zeros(n_forks, dtype=np.int64) for name in count_names}
    finite = np.ones(n_forks, dtype=bool)
    tail_start_step = int(np.floor((1.0 - audit_tail_fraction) * n_steps))
    n_tail_states = n_steps - tail_start_step + 1
    min_moments = np.full((n_forks, 4), np.inf)
    max_moments = np.full((n_forks, 4), -np.inf)
    stepwise_peak = state[:, 0].copy()
    tail_peak = np.full(n_forks, -np.inf)
    sample_index = 0

    for step in range(n_steps + 1):
        rhs, moments = _rhs_and_moments(
            state,
            prepared,
            transfer,
            mechanism=mechanism,
            clamp_s=clamp_s,
            subtractive_beta_mv=subtractive_beta_mv,
        )
        mu_e, sigma_e, mu_i, sigma_i, s_eff = moments
        moment_matrix = np.column_stack((mu_e, sigma_e, mu_i, sigma_i))
        min_moments = np.fmin(min_moments, moment_matrix)
        max_moments = np.fmax(max_moments, moment_matrix)
        support_bad = ~(transfer.support_mask(mu_e, sigma_e) & transfer.support_mask(mu_i, sigma_i))
        pool_bad = (
            (state[:, 6] < -1e-7)
            | (state[:, 6] > E_CEILING_KHZ + 1e-7)
            | (state[:, 7] < -1e-7)
            | (state[:, 7] > 1.0 + 1e-5)
            | (state[:, 8] < -1e-7)
            | (state[:, 8] > S_MAX + 1e-5)
        )
        rate_bad = (
            (state[:, 0] < -1e-7)
            | (state[:, 0] > E_CEILING_KHZ + 1e-7)
            | (state[:, 1] < -1e-7)
            | (state[:, 1] > I_CEILING_KHZ + 1e-7)
        )
        synapse_bad = (
            (state[:, 2] < -1e-7)
            | (state[:, 2] > E_CEILING_KHZ + 1e-7)
            | (state[:, 4] < -1e-7)
            | (state[:, 4] > E_CEILING_KHZ + 1e-7)
            | (state[:, 3] < -1e-7)
            | (state[:, 3] > I_CEILING_KHZ + 1e-7)
            | (state[:, 5] < -1e-7)
            | (state[:, 5] > I_CEILING_KHZ + 1e-7)
        )
        flags = {
            "support_violation": support_bad,
            "pool_bound": pool_bad,
            "rate_bound": rate_bad,
            "synapse_bound": synapse_bad,
            "negative_rate": (state[:, 0] < -1e-7) | (state[:, 1] < -1e-7),
            "e_refractory": state[:, 0] >= 0.95 * E_CEILING_KHZ,
            "i_refractory": state[:, 1] >= 0.95 * I_CEILING_KHZ,
            "over_100hz": state[:, 0] >= FINITE_HIGH_MAX_KHZ,
        }
        for name, flag in flags.items():
            counts[name] += flag
            if step >= tail_start_step:
                tail_counts[name] += flag
        now_finite = np.all(np.isfinite(state), axis=1) & np.all(np.isfinite(rhs), axis=1)
        finite &= now_finite
        stepwise_peak = np.fmax(stepwise_peak, state[:, 0])
        if step >= tail_start_step:
            tail_peak = np.fmax(tail_peak, state[:, 0])

        if sample_index < sample_steps.size and step == sample_steps[sample_index]:
            traces["rE_khz"][sample_index] = state[:, 0]
            traces["rI_khz"][sample_index] = state[:, 1]
            traces["rE_fast_khz"][sample_index] = state[:, 6]
            traces["mu_G"][sample_index] = state[:, 7]
            traces["S_G"][sample_index] = state[:, 8]
            traces["muE_mV"][sample_index] = mu_e
            traces["sigmaE_mV"][sample_index] = sigma_e
            traces["muI_mV"][sample_index] = mu_i
            traces["sigmaI_mV"][sample_index] = sigma_i
            traces["divisor"][sample_index] = 1.0 + prepared.alpha_g * s_eff
            sample_index += 1
        if step == n_steps:
            break
        state += dt_ms * rhs
        bad_after_update = ~np.all(np.isfinite(state), axis=1)
        state[bad_after_update] = np.nan

    output: dict[str, np.ndarray] = {
        **traces,
        "time_ms": sample_steps.astype(float) * dt_ms,
        "final_state": state,
        "finite": finite,
        "audit_n_euler_states": np.asarray(n_steps + 1, dtype=np.int64),
        "audit_n_tail_euler_states": np.asarray(n_tail_states, dtype=np.int64),
        "audit_tail_start_ms": np.asarray(tail_start_step * dt_ms, dtype=float),
        "stepwise_peak_rE_hz": 1000.0 * stepwise_peak,
        "stepwise_tail_peak_rE_hz": 1000.0 * tail_peak,
        "moment_min_stepwise": min_moments,
        "moment_max_stepwise": max_moments,
    }
    for name in count_names:
        output[f"{name}_step_count"] = counts[name]
        output[f"{name}_tail_step_count"] = tail_counts[name]
        output[f"{name}_occupancy_stepwise"] = counts[name] / float(n_steps + 1)
        output[f"{name}_tail_occupancy_stepwise"] = tail_counts[name] / float(n_tail_states)
    return output


def classify_extended_batch(
    metadata: Sequence[Mapping[str, Any]],
    simulation: Mapping[str, np.ndarray],
    thresholds: ForkClassifierThresholds,
    *,
    transfer_name: str,
    phase: str,
) -> list[dict[str, Any]]:
    """Classify trajectories while preserving every-Euler audit fields."""

    rows: list[dict[str, Any]] = []
    moment_names = ("muE_mV", "sigmaE_mV", "muI_mV", "sigmaI_mV")
    for index, meta in enumerate(metadata):
        metrics = classify_rate_trace(simulation["time_ms"], simulation["rE_khz"][:, index], thresholds)
        row: dict[str, Any] = {
            **dict(meta),
            **metrics,
            "transfer_resolution": transfer_name,
            "phase": phase,
            "support_clip_or_extrapolation": False,
            "pool_numerical_clip": False,
            "stepwise_peak_rE_hz": float(np.asarray(simulation["stepwise_peak_rE_hz"])[index]),
            "stepwise_tail_peak_rE_hz": float(np.asarray(simulation["stepwise_tail_peak_rE_hz"])[index]),
        }
        for audit_name in (
            "support_violation",
            "pool_bound",
            "rate_bound",
            "synapse_bound",
            "negative_rate",
            "e_refractory",
            "i_refractory",
            "over_100hz",
        ):
            count = int(np.asarray(simulation[f"{audit_name}_step_count"])[index])
            tail_count = int(np.asarray(simulation[f"{audit_name}_tail_step_count"])[index])
            row[f"{audit_name}_step_count"] = count
            row[f"{audit_name}_tail_step_count"] = tail_count
            row[f"{audit_name}_occupancy_stepwise"] = float(
                np.asarray(simulation[f"{audit_name}_occupancy_stepwise"])[index]
            )
            row[f"{audit_name}_tail_occupancy_stepwise"] = float(
                np.asarray(simulation[f"{audit_name}_tail_occupancy_stepwise"])[index]
            )
        for moment_index, moment_name in enumerate(moment_names):
            row[f"{moment_name}_min_stepwise"] = float(np.asarray(simulation["moment_min_stepwise"])[index, moment_index])
            row[f"{moment_name}_max_stepwise"] = float(np.asarray(simulation["moment_max_stepwise"])[index, moment_index])
        rows.append(row)
    return rows


def direct_exact_error_audit(
    simulation: Mapping[str, np.ndarray],
    transfer: ExtendedSiegertTransfer,
    *,
    max_points_per_fork: int = 16,
) -> dict[str, Any]:
    """Compare deterministic saved-state samples with direct stable quadrature."""

    n_time, n_forks = np.asarray(simulation["rE_khz"]).shape
    sample_records: list[tuple[int, str, float, float, float, float]] = []
    for fork in range(n_forks):
        base = np.linspace(max(0, int(0.6 * n_time)), n_time - 1, max_points_per_fork, dtype=int)
        extra: list[int] = []
        for name in ("muE_mV", "sigmaE_mV", "muI_mV", "sigmaI_mV"):
            values = np.asarray(simulation[name])[:, fork]
            if np.any(np.isfinite(values)):
                for center in (int(np.nanargmin(values)), int(np.nanargmax(values))):
                    extra.extend(range(max(0, center - 2), min(n_time, center + 3)))
        for time_index in np.unique(np.r_[base, extra]):
            for pop, mu_name, sigma_name, tau_m, tau_ref in (
                ("E", "muE_mV", "sigmaE_mV", TAU_ME, TREF_E),
                ("I", "muI_mV", "sigmaI_mV", TAU_MI, TREF_I),
            ):
                mu = float(np.asarray(simulation[mu_name])[time_index, fork])
                sigma = float(np.asarray(simulation[sigma_name])[time_index, fork])
                if not np.isfinite(mu) or not np.isfinite(sigma):
                    continue
                approx = float(transfer.rate(np.asarray([mu]), np.asarray([sigma]), pop)[0])
                exact = stable_siegert_rate(mu, sigma, tau_m, tau_ref)
                sample_records.append((fork, pop, mu, sigma, exact, approx))

    def summarize(records: list[tuple[int, str, float, float, float, float]]) -> dict[str, Any]:
        exact_values = np.asarray([row[4] for row in records], dtype=float)
        approx_values = np.asarray([row[5] for row in records], dtype=float)
        absolute_hz = 1000.0 * np.abs(approx_values - exact_values)
        meaningful = exact_values >= 1e-9
        relative = np.abs(approx_values[meaningful] - exact_values[meaningful]) / exact_values[meaningful]
        return {
            "n_samples": len(records),
            "n_meaningful_rate_samples": int(np.sum(meaningful)),
            "max_abs_error_hz": float(np.max(absolute_hz)) if absolute_hz.size else np.nan,
            "p99_abs_error_hz": float(np.percentile(absolute_hz, 99.0)) if absolute_hz.size else np.nan,
            "max_relative_error_meaningful": float(np.max(relative)) if relative.size else np.nan,
            "p99_relative_error_meaningful": float(np.percentile(relative, 99.0)) if relative.size else np.nan,
            "meaningful_rate_floor_khz": 1e-9,
            "pass": bool(
                absolute_hz.size
                and np.all(np.isfinite(absolute_hz))
                and np.max(absolute_hz) <= 0.25
                and (not relative.size or np.percentile(relative, 99.0) <= 0.02)
            ),
        }

    overall = summarize(sample_records)
    overall["per_fork"] = [
        {"fork_index": fork, **summarize([row for row in sample_records if row[0] == fork])}
        for fork in range(n_forks)
    ]
    overall["all_forks_pass"] = bool(all(row["pass"] for row in overall["per_fork"]))
    overall["pass"] = bool(overall["pass"] and overall["all_forks_pass"])
    return overall


def temporal_refinement_status(
    confirm_row: Mapping[str, Any],
    refined_row: Mapping[str, Any],
    *,
    exact_error_pass: bool,
) -> str:
    """Require dt/2 to preserve the confirmed object's class, rate, and frequency."""

    status = resolution_pair_status(refined_row, refined_row, exact_error_pass=exact_error_pass)
    if status != "candidate_survives":
        return status
    confirm_class = str(confirm_row.get("classification"))
    refined_class = str(refined_row.get("classification"))
    if confirm_class != refined_class or confirm_class not in _CANDIDATE_CLASSES:
        return "numerical_unresolved"
    rates = np.asarray([confirm_row.get("tail_mean_hz"), refined_row.get("tail_mean_hz")], dtype=float)
    if not np.all(np.isfinite(rates)) or float(np.ptp(rates)) > max(1.0, 0.10 * float(np.mean(rates))):
        return "numerical_unresolved"
    if confirm_class == "bounded_oscillatory_candidate":
        frequencies = np.asarray(
            [confirm_row.get("dominant_frequency_hz"), refined_row.get("dominant_frequency_hz")], dtype=float
        )
        if (
            not np.all(np.isfinite(frequencies))
            or float(np.ptp(frequencies)) > max(0.5, 0.15 * float(np.mean(frequencies)))
        ):
            return "numerical_unresolved"
    return "candidate_survives"


def resolution_pair_status(
    coarse_row: Mapping[str, Any],
    fine_row: Mapping[str, Any],
    *,
    exact_error_pass: bool,
) -> str:
    """Map a coarse/fine pair to the four locked transfer-audit outcomes."""

    fatal_fields = (
        "support_violation_step_count",
        "pool_bound_step_count",
        "rate_bound_step_count",
        "synapse_bound_step_count",
        "negative_rate_step_count",
    )
    if (
        not bool(coarse_row.get("finite", False))
        or not bool(fine_row.get("finite", False))
        or any(int(coarse_row.get(field, 0)) > 0 or int(fine_row.get(field, 0)) > 0 for field in fatal_fields)
        or float(coarse_row.get("e_refractory_tail_occupancy_stepwise", 0.0)) > 0.05
        or float(fine_row.get("e_refractory_tail_occupancy_stepwise", 0.0)) > 0.05
        or float(coarse_row.get("i_refractory_tail_occupancy_stepwise", 0.0)) > 0.05
        or float(fine_row.get("i_refractory_tail_occupancy_stepwise", 0.0)) > 0.05
        or not exact_error_pass
    ):
        return "numerical_unresolved"
    if int(coarse_row.get("over_100hz_tail_step_count", 0)) > 0 or int(fine_row.get("over_100hz_tail_step_count", 0)) > 0:
        return "becomes_over_100"
    coarse_class = str(coarse_row.get("classification"))
    fine_class = str(fine_row.get("classification"))
    if coarse_class == fine_class == "low_fixed_point":
        return "collapses_low"
    if coarse_class in _CANDIDATE_CLASSES and fine_class == coarse_class:
        rates = np.asarray([coarse_row["tail_mean_hz"], fine_row["tail_mean_hz"]], dtype=float)
        if float(np.ptp(rates)) <= max(1.0, 0.1 * float(np.mean(rates))):
            if coarse_class != "bounded_oscillatory_candidate":
                return "candidate_survives"
            frequencies = np.asarray(
                [coarse_row["dominant_frequency_hz"], fine_row["dominant_frequency_hz"]], dtype=float
            )
            if float(np.ptp(frequencies)) <= max(0.5, 0.15 * float(np.mean(frequencies))):
                return "candidate_survives"
    return "numerical_unresolved"
