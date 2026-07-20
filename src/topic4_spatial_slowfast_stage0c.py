"""Stage 0C: M4 dynamic divisive pool on the homogeneous E/I fast system.

The frozen state is ``[rE, rI, sEE, sEI, sIE, sII, rE_fast, mu_G, S_G]``.
Rates are kHz and time is ms.  ``z`` is a frozen postsynaptic inhibitory-efficacy
coordinate (the Stage-0B ``q`` axis); the local recovery variable is fixed at zero.

Only recurrent E input to E cells is normalized::

    D = 1 + alpha_G * S_G
    recurrent E mean     -> recurrent E mean / D
    recurrent E variance -> recurrent E variance / D**2

External excitation, I->E inhibition, and all inputs to I cells are unchanged.
The pool ODE is intentionally never numerically clipped.  Candidate acceptance
instead audits the natural state bounds and the transfer-table support explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

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
    W_EE,
    W_EI,
    W_IE,
    W_II,
    nu_theta_pop,
)
from src.topic4_m3b_spectral_phase import _LUT_MU, _LUT_SIG, _phi_field
from src.topic4_spatial_slowfast_stage0b import (
    FastParameters,
    ForkClassifierThresholds,
    classify_rate_trace,
    fast_rhs as stage0b_fast_rhs,
    moments_from_state as stage0b_moments_from_state,
)


STATE_FIELDS: tuple[str, ...] = (
    "rE",
    "rI",
    "sEE",
    "sEI",
    "sIE",
    "sII",
    "rE_fast",
    "mu_G",
    "S_G",
)
E_CEILING_KHZ: float = 1.0 / TREF_E
I_CEILING_KHZ: float = 1.0 / TREF_I
FINITE_HIGH_MAX_KHZ: float = 0.100
TAU_FAST_MS: float = 15.0
TAU_MU_MS: float = 30.0
TAU_S_MS: float = 80.0
S_MAX: float = 1.0
E0_KHZ: float = 0.005
E50_KHZ: float = 0.015
N_PSI: float = 2.0


@dataclass(frozen=True)
class PoolParameters:
    """One locked Stage-0C frozen parameter point."""

    z: float
    alpha_g: float
    w_ee_mult: float = 1.1
    ratio: float = 1.0

    def validate(self) -> "PoolParameters":
        values = (self.z, self.alpha_g, self.w_ee_mult, self.ratio)
        if not all(np.isfinite(values)):
            raise ValueError("pool parameters must be finite")
        if not 0.0 < self.z <= 1.0:
            raise ValueError("z must lie in (0,1]")
        if self.alpha_g < 0.0 or self.w_ee_mult <= 0.0 or self.ratio <= 0.0:
            raise ValueError("require alpha_g>=0, w_ee_mult>0, ratio>0")
        return self


def recruitment_sensor(rate_khz: np.ndarray | float) -> np.ndarray:
    """Locked M4 recruitment nonlinearity, evaluated before homogeneous pooling."""

    excess = np.maximum(np.asarray(rate_khz, dtype=float) - E0_KHZ, 0.0)
    numerator = excess**N_PSI
    return numerator / (E50_KHZ**N_PSI + numerator)


def equilibrium_state(rates: Sequence[float]) -> np.ndarray:
    """Lift ``(rE,rI)`` onto synaptic and pool equilibrium manifolds."""

    r_e, r_i = map(float, rates)
    sensor = float(recruitment_sensor(r_e))
    return np.asarray([r_e, r_i, r_e, r_i, r_e, r_i, r_e, sensor, sensor], dtype=float)


def _parameter_arrays(
    params: PoolParameters | Sequence[PoolParameters], n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[PoolParameters]]:
    if isinstance(params, PoolParameters):
        checked = params.validate()
        checked_list = [checked] * n
    else:
        checked_list = [point.validate() for point in params]
        if len(checked_list) != n:
            raise ValueError("one PoolParameters object is required per state")
    return (
        np.asarray([point.z for point in checked_list]),
        np.asarray([point.alpha_g for point in checked_list]),
        np.asarray([point.w_ee_mult for point in checked_list]),
        np.asarray([point.ratio for point in checked_list]),
        checked_list,
    )


def moments_from_state(
    state: np.ndarray,
    params: PoolParameters | Sequence[PoolParameters],
    *,
    mechanism: str = "dynamic",
    clamp_s: float | None = None,
    subtractive_beta_mv: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return instantaneous moments and effective pool output.

    ``mechanism`` is normally ``dynamic``.  The other modes are only used after a
    confirmed candidate in the pre-registered minimal ablation:

    - ``instantaneous``: ``S_eff=Psi(rE)`` (both pool lags bypassed);
    - ``clamped``: constant ``S_eff=clamp_s``;
    - ``matched_subtractive``: subtract ``beta*S_G`` from recurrent mean and leave
      recurrent variance unchanged;
    - ``mean_only``: divide recurrent mean but leave recurrent variance unchanged.
    """

    state = np.asarray(state, dtype=float)
    one = state.ndim == 1
    batch = state[None, :] if one else state
    if batch.ndim != 2 or batch.shape[1] != 9:
        raise ValueError("state must have shape (9,) or (n,9)")
    z, alpha, w_mult, ratio, checked = _parameter_arrays(params, batch.shape[0])
    if mechanism not in {"dynamic", "instantaneous", "clamped", "matched_subtractive", "mean_only"}:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    if mechanism == "clamped":
        if clamp_s is None or not np.isfinite(clamp_s) or clamp_s < 0.0:
            raise ValueError("clamped mechanism requires finite clamp_s>=0")
        s_eff = np.full(batch.shape[0], float(clamp_s))
    elif mechanism == "instantaneous":
        s_eff = recruitment_sensor(batch[:, 0])
    else:
        s_eff = batch[:, 8]
    divisor = 1.0 + alpha * s_eff
    if np.any(divisor <= 0.0) or not np.all(np.isfinite(divisor)):
        raise FloatingPointError("nonpositive or nonfinite divisive factor")

    s_ee, s_ei, s_ie, s_ii = (batch[:, index] for index in range(2, 6))
    wee = w_mult * W_EE
    w_ei = z * W_EI
    nuext = ratio * nu_theta_pop()
    recurrent_mean_e = TAU_ME * C_EE * wee * s_ee
    recurrent_var_e = TAU_ME * C_EE * wee**2 * s_ee
    if mechanism == "matched_subtractive":
        if subtractive_beta_mv is None or not np.isfinite(subtractive_beta_mv) or subtractive_beta_mv < 0.0:
            raise ValueError("matched_subtractive requires finite subtractive_beta_mv>=0")
        recurrent_mean_e = recurrent_mean_e - float(subtractive_beta_mv) * s_eff
        recurrent_var_effective = recurrent_var_e
    else:
        recurrent_mean_e = recurrent_mean_e / divisor
        recurrent_var_effective = recurrent_var_e if mechanism == "mean_only" else recurrent_var_e / divisor**2

    mu_e = recurrent_mean_e - TAU_ME * C_EI * w_ei * s_ei + TAU_ME * JX_E * nuext
    var_e = recurrent_var_effective + TAU_ME * C_EI * w_ei**2 * s_ei + TAU_ME * JX_E**2 * nuext
    mu_i = TAU_MI * (C_IE * W_IE * s_ie - C_II * W_II * s_ii) + TAU_MI * JX_I * nuext
    var_i = TAU_MI * (C_IE * W_IE**2 * s_ie + C_II * W_II**2 * s_ii) + TAU_MI * JX_I**2 * nuext

    # Load-bearing alpha=0 parity: route the first four moments through the exact
    # Stage-0B implementation.  This also prevents harmless expression-order drift
    # from obscuring the scientific regression check.
    parity = alpha == 0.0
    if np.any(parity) and mechanism in {"dynamic", "instantaneous", "clamped", "mean_only"}:
        base_params = [FastParameters(point.w_ee_mult, point.z, point.ratio) for point in np.asarray(checked, dtype=object)[parity]]
        base = stage0b_moments_from_state(batch[parity, :6], base_params)
        mu_e[parity], sigma_e_parity, mu_i[parity], sigma_i_parity = base
    else:
        sigma_e_parity = sigma_i_parity = None
    sigma_e = np.sqrt(np.maximum(var_e, 1e-9))
    sigma_i = np.sqrt(np.maximum(var_i, 1e-9))
    if np.any(parity) and mechanism in {"dynamic", "instantaneous", "clamped", "mean_only"}:
        sigma_e[parity] = sigma_e_parity
        sigma_i[parity] = sigma_i_parity
    out = (mu_e, sigma_e, mu_i, sigma_i, s_eff)
    if one:
        return tuple(value[0] for value in out)  # type: ignore[return-value]
    return out


def pool_rhs(
    state: np.ndarray,
    params: PoolParameters | Sequence[PoolParameters],
    *,
    mechanism: str = "dynamic",
    clamp_s: float | None = None,
    subtractive_beta_mv: float | None = None,
) -> np.ndarray:
    """Nine-dimensional self-consistent RHS; the pool ODE has no clipping."""

    state = np.asarray(state, dtype=float)
    one = state.ndim == 1
    batch = state[None, :] if one else state
    if batch.ndim != 2 or batch.shape[1] != 9:
        raise ValueError("state must have shape (9,) or (n,9)")
    _, alpha, _, _, checked = _parameter_arrays(params, batch.shape[0])
    mu_e, sigma_e, mu_i, sigma_i, _ = moments_from_state(
        batch,
        checked,
        mechanism=mechanism,
        clamp_s=clamp_s,
        subtractive_beta_mv=subtractive_beta_mv,
    )
    target_e = _phi_field(mu_e, sigma_e, "E")
    target_i = _phi_field(mu_i, sigma_i, "I")
    out = np.empty_like(batch)
    out[:, 0] = (-batch[:, 0] + target_e) / TAU_ME
    out[:, 1] = (-batch[:, 1] + target_i) / TAU_MI
    out[:, 2] = (batch[:, 0] - batch[:, 2]) / TAU_AMPA
    out[:, 3] = (batch[:, 1] - batch[:, 3]) / TAU_GABA
    out[:, 4] = (batch[:, 0] - batch[:, 4]) / TAU_AMPA
    out[:, 5] = (batch[:, 1] - batch[:, 5]) / TAU_GABA
    out[:, 6] = (batch[:, 0] - batch[:, 6]) / TAU_FAST_MS
    drive = recruitment_sensor(batch[:, 6])
    out[:, 7] = (-batch[:, 7] + drive) / TAU_MU_MS
    out[:, 8] = (-batch[:, 8] + S_MAX * batch[:, 7]) / TAU_S_MS

    parity = alpha == 0.0
    if np.any(parity) and mechanism in {"dynamic", "instantaneous", "clamped", "mean_only"}:
        base_params = [FastParameters(point.w_ee_mult, point.z, point.ratio) for point in np.asarray(checked, dtype=object)[parity]]
        out[parity, :6] = stage0b_fast_rhs(batch[parity, :6], base_params)
    return out[0] if one else out


def fixed_point_residual(rates: Sequence[float], params: PoolParameters) -> np.ndarray:
    rhs = pool_rhs(equilibrium_state(rates), params)
    return np.asarray([-TAU_ME * rhs[0], -TAU_MI * rhs[1]], dtype=float)


def numerical_jacobian(
    state: Sequence[float] | np.ndarray,
    params: PoolParameters,
    *,
    mechanism: str = "dynamic",
    clamp_s: float | None = None,
    subtractive_beta_mv: float | None = None,
    relative_step: float = 2e-4,
    absolute_step: float = 2e-7,
) -> np.ndarray:
    """Centered 9x9 Jacobian of the same nonlinear RHS."""

    state = np.asarray(state, dtype=float)
    if state.shape != (9,) or not np.all(np.isfinite(state)):
        raise ValueError("state must be finite with shape (9,)")
    jac = np.empty((9, 9), dtype=float)
    for column in range(9):
        h = max(absolute_step, relative_step * max(abs(float(state[column])), 1e-3))
        plus = state.copy()
        minus = state.copy()
        plus[column] += h
        minus[column] -= h
        jac[:, column] = (
            pool_rhs(plus, params, mechanism=mechanism, clamp_s=clamp_s, subtractive_beta_mv=subtractive_beta_mv)
            - pool_rhs(minus, params, mechanism=mechanism, clamp_s=clamp_s, subtractive_beta_mv=subtractive_beta_mv)
        ) / (2.0 * h)
    return jac


def _default_root_seeds() -> list[tuple[float, float]]:
    e = (1e-6, 2.5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 4e-2, 8e-2, 0.20, 0.45)
    i = (1e-6, 1e-3, 5e-3, 2e-2, 8e-2, 0.20, 0.50, 0.90)
    return [(a, b) for a in e for b in i]


def _root_lut_clipped(state: np.ndarray, params: PoolParameters) -> bool:
    mu_e, sigma_e, mu_i, sigma_i, _ = moments_from_state(state, params)
    return bool(
        mu_e < _LUT_MU[0]
        or mu_e > _LUT_MU[1]
        or mu_i < _LUT_MU[0]
        or mu_i > _LUT_MU[1]
        or sigma_e < _LUT_SIG[0]
        or sigma_e > _LUT_SIG[1]
        or sigma_i < _LUT_SIG[0]
        or sigma_i > _LUT_SIG[1]
    )


def find_fixed_points(
    params: PoolParameters,
    *,
    warm_roots: Iterable[Sequence[float]] = (),
    residual_tolerance: float = 2e-7,
    cluster_tolerance_khz: float = 2e-5,
) -> list[dict[str, Any]]:
    """Find roots in 2D and classify stability from the full 9D Jacobian."""

    params.validate()
    roots: list[np.ndarray] = []
    for seed in list(warm_roots) + _default_root_seeds():
        initial = np.clip(np.asarray(seed, dtype=float), [1e-9, 1e-9], [E_CEILING_KHZ, I_CEILING_KHZ])
        fit = least_squares(
            lambda rates: fixed_point_residual(rates, params),
            initial,
            bounds=([1e-9, 1e-9], [E_CEILING_KHZ, I_CEILING_KHZ]),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=350,
        )
        root = np.asarray(fit.x, dtype=float)
        residual = float(np.linalg.norm(fixed_point_residual(root, params), ord=np.inf))
        if not fit.success or residual > residual_tolerance or not np.all(np.isfinite(root)):
            continue
        if not any(np.linalg.norm(root - old, ord=np.inf) <= cluster_tolerance_khz for old in roots):
            roots.append(root)

    output: list[dict[str, Any]] = []
    for root in sorted(roots, key=lambda value: (value[0], value[1])):
        state = equilibrium_state(root)
        jac = numerical_jacobian(state, params)
        eigenvalues = np.linalg.eigvals(jac)
        leading = eigenvalues[int(np.argmax(eigenvalues.real))]
        rate_hz = float(1000.0 * root[0])
        if rate_hz <= 5.0:
            branch_class = "low_root"
        elif rate_hz < 100.0:
            branch_class = "finite_high_root"
        else:
            branch_class = "saturation_cliff_root"
        stability = "stable" if leading.real < -1e-4 else ("unstable" if leading.real > 1e-4 else "marginal")
        output.append(
            {
                "rE_khz": float(root[0]),
                "rI_khz": float(root[1]),
                "rE_hz": rate_hz,
                "rI_hz": float(1000.0 * root[1]),
                "residual_inf": float(np.linalg.norm(fixed_point_residual(root, params), ord=np.inf)),
                "stability": stability,
                "branch_class": branch_class,
                "leading_real_per_ms": float(leading.real),
                "leading_imag_per_ms": float(abs(leading.imag)),
                "leading_frequency_hz": float(1000.0 * abs(leading.imag) / (2.0 * np.pi)),
                "S_G": float(state[8]),
                "divisor": float(1.0 + params.alpha_g * state[8]),
                "lut_clip_at_root": _root_lut_clipped(state, params),
                "jacobian_dimension": 9,
            }
        )
    return output


def continuation_root_scan(
    z_values: Sequence[float], alpha_values: Sequence[float], *, w_ee_mult: float = 1.1, ratio: float = 1.0
) -> list[dict[str, Any]]:
    """Warm-continuation scan along z and adjacent alpha rows."""

    rows: list[dict[str, Any]] = []
    previous_alpha: dict[float, list[Sequence[float]]] = {}
    for alpha in alpha_values:
        previous_z: list[Sequence[float]] = []
        current_alpha: dict[float, list[Sequence[float]]] = {}
        for z in z_values:
            params = PoolParameters(float(z), float(alpha), float(w_ee_mult), float(ratio))
            roots = find_fixed_points(params, warm_roots=list(previous_z) + list(previous_alpha.get(float(z), [])))
            rows.append(
                {
                    "z": float(z),
                    "alpha_G": float(alpha),
                    "w_ee_mult": float(w_ee_mult),
                    "ratio": float(ratio),
                    "n_roots": len(roots),
                    "roots": roots,
                }
            )
            previous_z = [[root["rE_khz"], root["rI_khz"]] for root in roots]
            current_alpha[float(z)] = previous_z
        previous_alpha = current_alpha
    return rows


_BASE_PROBES: tuple[tuple[str, float, float], ...] = (
    ("probe_low", 1e-5, 1e-5),
    ("probe_rest", 5e-4, 2e-3),
    ("probe_5hz", 5e-3, 1e-2),
    ("probe_20hz", 2e-2, 4e-2),
    ("probe_boundary", 8e-2, 1.5e-1),
    ("probe_high", 2e-1, 4e-1),
    ("probe_ceiling", 4.5e-1, 8e-1),
)
_STAGE0B_OFF_MANIFOLD: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("e_synapse_loaded_i_low", (0.001, 0.003, 0.020, 0.005, 0.020, 0.005)),
    ("i_synapse_loaded_e_low", (0.001, 0.003, 0.005, 0.020, 0.005, 0.020)),
    ("rate_high_synapse_low", (0.080, 0.150, 0.003, 0.008, 0.003, 0.008)),
    ("rate_low_synapse_high", (0.001, 0.003, 0.020, 0.020, 0.020, 0.020)),
)


def build_state_forks(
    root_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, list[PoolParameters]]:
    """Build on-manifold and history-bearing probes spanning all nine dimensions."""

    metadata: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    parameters: list[PoolParameters] = []
    for point in root_rows:
        params = PoolParameters(float(point["z"]), float(point["alpha_G"]), float(point["w_ee_mult"]), float(point["ratio"]))

        def add(kind: str, label: str, state: Sequence[float], root_index: int | None = None) -> None:
            array = np.asarray(state, dtype=float)
            if array.shape != (9,) or not np.all(np.isfinite(array)) or np.any(array < 0.0):
                raise ValueError("fork state must be finite, nonnegative, shape (9,)")
            if array[0] > E_CEILING_KHZ or array[1] > I_CEILING_KHZ:
                raise ValueError("fork rate exceeds refractory ceiling")
            metadata.append(
                {
                    "z": params.z,
                    "alpha_G": params.alpha_g,
                    "w_ee_mult": params.w_ee_mult,
                    "ratio": params.ratio,
                    "initial_kind": kind,
                    "initial_label": label,
                    "root_index": root_index,
                    "initial_rE_hz": float(1000.0 * array[0]),
                    "initial_rI_hz": float(1000.0 * array[1]),
                    "initial_rE_fast_hz": float(1000.0 * array[6]),
                    "initial_mu_G": float(array[7]),
                    "initial_S_G": float(array[8]),
                }
            )
            states.append(array)
            parameters.append(params)

        for label, r_e, r_i in _BASE_PROBES:
            add("on_manifold_probe", label, equilibrium_state((r_e, r_i)))
        for label, state6 in _STAGE0B_OFF_MANIFOLD:
            sensor = float(recruitment_sensor(state6[0]))
            add("stage0b_off_manifold_probe", label, (*state6, state6[0], sensor, sensor))
        pool_probes = (
            ("pool_unloaded_at_high_rate", (0.080, 0.150, 0.080, 0.150, 0.080, 0.150, 0.005, 0.0, 0.0)),
            ("sensor_loaded_pool_empty", (0.001, 0.003, 0.001, 0.003, 0.001, 0.003, 0.080, 0.0, 0.0)),
            ("mu_loaded_s_empty", (0.001, 0.003, 0.001, 0.003, 0.001, 0.003, 0.001, 0.8, 0.0)),
            ("pool_loaded_at_low_rate", (0.001, 0.003, 0.001, 0.003, 0.001, 0.003, 0.001, 0.8, 0.8)),
        )
        for label, state in pool_probes:
            add("pool_off_manifold_probe", label, state)
        for root_index, root in enumerate(point.get("roots", [])):
            r_e, r_i = float(root["rE_khz"]), float(root["rI_khz"])
            add("exact_root", f"root_{root_index}", equilibrium_state((r_e, r_i)), root_index)
            for suffix, scale in (("minus", 0.90), ("plus", 1.10)):
                add(
                    "root_perturbation",
                    f"root_{root_index}_{suffix}",
                    equilibrium_state((float(np.clip(r_e * scale, 1e-9, E_CEILING_KHZ)), r_i)),
                    root_index,
                )
    return metadata, np.asarray(states, dtype=float), parameters


def simulate_forks(
    initial_states: np.ndarray,
    params: Sequence[PoolParameters],
    *,
    dt_ms: float,
    duration_ms: float,
    save_stride: int,
    mechanism: str = "dynamic",
    clamp_s: float | None = None,
    subtractive_beta_mv: float | None = None,
    audit_tail_fraction: float = 0.40,
) -> dict[str, np.ndarray]:
    """Vectorized forward-Euler fork integration with explicit pool-bound traces."""

    state = np.asarray(initial_states, dtype=float).copy()
    if state.ndim != 2 or state.shape[1] != 9 or state.shape[0] != len(params):
        raise ValueError("initial_states and params must align as (n,9) and length n")
    if dt_ms <= 0.0 or duration_ms <= dt_ms or save_stride < 1:
        raise ValueError("invalid integration contract")
    if not 0.1 <= audit_tail_fraction <= 0.9:
        raise ValueError("audit_tail_fraction must lie in [0.1,0.9]")
    n_steps = int(round(duration_ms / dt_ms))
    if not np.isclose(n_steps * dt_ms, duration_ms, atol=1e-9):
        raise ValueError("duration_ms must be an integer multiple of dt_ms")
    sample_steps = np.arange(0, n_steps + 1, int(save_stride), dtype=int)
    if sample_steps[-1] != n_steps:
        sample_steps = np.r_[sample_steps, n_steps]
    shape = (sample_steps.size, state.shape[0])
    traces = {name: np.empty(shape, dtype=np.float32) for name in ("rE_khz", "rI_khz", "rE_fast_khz", "mu_G", "S_G", "muE_mV", "sigmaE_mV", "muI_mV", "sigmaI_mV", "divisor")}

    def save(index: int) -> None:
        mu_e, sigma_e, mu_i, sigma_i, s_eff = moments_from_state(
            state,
            params,
            mechanism=mechanism,
            clamp_s=clamp_s,
            subtractive_beta_mv=subtractive_beta_mv,
        )
        traces["rE_khz"][index] = state[:, 0]
        traces["rI_khz"][index] = state[:, 1]
        traces["rE_fast_khz"][index] = state[:, 6]
        traces["mu_G"][index] = state[:, 7]
        traces["S_G"][index] = state[:, 8]
        traces["muE_mV"][index] = mu_e
        traces["sigmaE_mV"][index] = sigma_e
        traces["muI_mV"][index] = mu_i
        traces["sigmaI_mV"][index] = sigma_i
        alpha = np.asarray([point.alpha_g for point in params])
        traces["divisor"][index] = 1.0 + alpha * s_eff

    n_forks = state.shape[0]
    lut_clip_step_count = np.zeros(n_forks, dtype=np.int64)
    lut_clip_tail_step_count = np.zeros(n_forks, dtype=np.int64)
    pool_bound_step_count = np.zeros(n_forks, dtype=np.int64)
    pool_bound_tail_step_count = np.zeros(n_forks, dtype=np.int64)
    rate_bound_step_count = np.zeros(n_forks, dtype=np.int64)
    rate_bound_tail_step_count = np.zeros(n_forks, dtype=np.int64)
    synapse_bound_step_count = np.zeros(n_forks, dtype=np.int64)
    synapse_bound_tail_step_count = np.zeros(n_forks, dtype=np.int64)
    negative_rate_step_count = np.zeros(n_forks, dtype=np.int64)
    negative_rate_tail_step_count = np.zeros(n_forks, dtype=np.int64)
    e_refractory_step_count = np.zeros(n_forks, dtype=np.int64)
    e_refractory_tail_step_count = np.zeros(n_forks, dtype=np.int64)
    i_refractory_step_count = np.zeros(n_forks, dtype=np.int64)
    i_refractory_tail_step_count = np.zeros(n_forks, dtype=np.int64)
    over_100hz_step_count = np.zeros(n_forks, dtype=np.int64)
    over_100hz_tail_step_count = np.zeros(n_forks, dtype=np.int64)
    tail_start_step = int(np.floor((1.0 - audit_tail_fraction) * n_steps))
    n_tail_euler_states = n_steps - tail_start_step + 1
    stepwise_peak_rE_khz = state[:, 0].copy()
    stepwise_tail_peak_rE_khz = np.full(n_forks, -np.inf, dtype=float)
    stepwise_min_mu_G = state[:, 7].copy()
    stepwise_max_mu_G = state[:, 7].copy()
    stepwise_min_S_G = state[:, 8].copy()
    stepwise_max_S_G = state[:, 8].copy()

    def audit_step(step_index: int) -> None:
        """Audit every Euler state, not only the downsampled saved frames."""

        nonlocal stepwise_peak_rE_khz, stepwise_min_mu_G, stepwise_max_mu_G
        nonlocal stepwise_tail_peak_rE_khz, stepwise_min_S_G, stepwise_max_S_G
        mu_e, sigma_e, mu_i, sigma_i, _ = moments_from_state(
            state,
            params,
            mechanism=mechanism,
            clamp_s=clamp_s,
            subtractive_beta_mv=subtractive_beta_mv,
        )
        clipped = (
            (mu_e < _LUT_MU[0]) | (mu_e > _LUT_MU[1])
            | (mu_i < _LUT_MU[0]) | (mu_i > _LUT_MU[1])
            | (sigma_e < _LUT_SIG[0]) | (sigma_e > _LUT_SIG[1])
            | (sigma_i < _LUT_SIG[0]) | (sigma_i > _LUT_SIG[1])
        )
        pool_bad = (
            (state[:, 6] < -1e-7) | (state[:, 6] > E_CEILING_KHZ + 1e-7)
            | (state[:, 7] < -1e-7) | (state[:, 7] > 1.0 + 1e-5)
            | (state[:, 8] < -1e-7) | (state[:, 8] > S_MAX + 1e-5)
        )
        rate_bad = (
            (state[:, 0] < -1e-7) | (state[:, 0] > E_CEILING_KHZ + 1e-7)
            | (state[:, 1] < -1e-7) | (state[:, 1] > I_CEILING_KHZ + 1e-7)
        )
        synapse_bad = (
            (state[:, 2] < -1e-7) | (state[:, 2] > E_CEILING_KHZ + 1e-7)
            | (state[:, 4] < -1e-7) | (state[:, 4] > E_CEILING_KHZ + 1e-7)
            | (state[:, 3] < -1e-7) | (state[:, 3] > I_CEILING_KHZ + 1e-7)
            | (state[:, 5] < -1e-7) | (state[:, 5] > I_CEILING_KHZ + 1e-7)
        )
        negative_rate = (state[:, 0] < -1e-7) | (state[:, 1] < -1e-7)
        e_refractory = state[:, 0] >= 0.95 * E_CEILING_KHZ
        i_refractory = state[:, 1] >= 0.95 * I_CEILING_KHZ
        over_100 = state[:, 0] >= FINITE_HIGH_MAX_KHZ
        lut_clip_step_count[:] += clipped
        pool_bound_step_count[:] += pool_bad
        rate_bound_step_count[:] += rate_bad
        synapse_bound_step_count[:] += synapse_bad
        negative_rate_step_count[:] += negative_rate
        e_refractory_step_count[:] += e_refractory
        i_refractory_step_count[:] += i_refractory
        over_100hz_step_count[:] += over_100
        if step_index >= tail_start_step:
            lut_clip_tail_step_count[:] += clipped
            pool_bound_tail_step_count[:] += pool_bad
            rate_bound_tail_step_count[:] += rate_bad
            synapse_bound_tail_step_count[:] += synapse_bad
            negative_rate_tail_step_count[:] += negative_rate
            e_refractory_tail_step_count[:] += e_refractory
            i_refractory_tail_step_count[:] += i_refractory
            over_100hz_tail_step_count[:] += over_100
            stepwise_tail_peak_rE_khz = np.fmax(stepwise_tail_peak_rE_khz, state[:, 0])
        stepwise_peak_rE_khz = np.fmax(stepwise_peak_rE_khz, state[:, 0])
        stepwise_min_mu_G = np.fmin(stepwise_min_mu_G, state[:, 7])
        stepwise_max_mu_G = np.fmax(stepwise_max_mu_G, state[:, 7])
        stepwise_min_S_G = np.fmin(stepwise_min_S_G, state[:, 8])
        stepwise_max_S_G = np.fmax(stepwise_max_S_G, state[:, 8])

    save(0)
    audit_step(0)
    finite = np.ones(state.shape[0], dtype=bool)
    sample_index = 1
    for step in range(1, n_steps + 1):
        state += dt_ms * pool_rhs(
            state,
            params,
            mechanism=mechanism,
            clamp_s=clamp_s,
            subtractive_beta_mv=subtractive_beta_mv,
        )
        now_finite = np.all(np.isfinite(state), axis=1)
        finite &= now_finite
        if not np.all(now_finite):
            state[~now_finite] = np.nan
        audit_step(step)
        if sample_index < sample_steps.size and step == sample_steps[sample_index]:
            save(sample_index)
            sample_index += 1
    traces.update(
        {
            "time_ms": sample_steps.astype(float) * dt_ms,
            "final_state": state,
            "finite": finite,
            "audit_n_euler_states": np.asarray(n_steps + 1, dtype=np.int64),
            "audit_n_tail_euler_states": np.asarray(n_tail_euler_states, dtype=np.int64),
            "audit_tail_start_ms": np.asarray(tail_start_step * dt_ms, dtype=float),
            "lut_clip_step_count": lut_clip_step_count,
            "lut_clip_occupancy_stepwise": lut_clip_step_count / float(n_steps + 1),
            "lut_clip_tail_step_count": lut_clip_tail_step_count,
            "lut_clip_tail_occupancy_stepwise": lut_clip_tail_step_count / float(n_tail_euler_states),
            "pool_bound_step_count": pool_bound_step_count,
            "pool_bound_occupancy_stepwise": pool_bound_step_count / float(n_steps + 1),
            "pool_bound_tail_step_count": pool_bound_tail_step_count,
            "pool_bound_tail_occupancy_stepwise": pool_bound_tail_step_count / float(n_tail_euler_states),
            "rate_bound_step_count": rate_bound_step_count,
            "rate_bound_occupancy_stepwise": rate_bound_step_count / float(n_steps + 1),
            "rate_bound_tail_step_count": rate_bound_tail_step_count,
            "rate_bound_tail_occupancy_stepwise": rate_bound_tail_step_count / float(n_tail_euler_states),
            "synapse_bound_step_count": synapse_bound_step_count,
            "synapse_bound_occupancy_stepwise": synapse_bound_step_count / float(n_steps + 1),
            "synapse_bound_tail_step_count": synapse_bound_tail_step_count,
            "synapse_bound_tail_occupancy_stepwise": synapse_bound_tail_step_count / float(n_tail_euler_states),
            "negative_rate_step_count": negative_rate_step_count,
            "negative_rate_occupancy_stepwise": negative_rate_step_count / float(n_steps + 1),
            "negative_rate_tail_step_count": negative_rate_tail_step_count,
            "negative_rate_tail_occupancy_stepwise": negative_rate_tail_step_count / float(n_tail_euler_states),
            "e_refractory_step_count": e_refractory_step_count,
            "e_refractory_occupancy_stepwise": e_refractory_step_count / float(n_steps + 1),
            "e_refractory_tail_step_count": e_refractory_tail_step_count,
            "e_refractory_tail_occupancy_stepwise": e_refractory_tail_step_count / float(n_tail_euler_states),
            "i_refractory_step_count": i_refractory_step_count,
            "i_refractory_occupancy_stepwise": i_refractory_step_count / float(n_steps + 1),
            "i_refractory_tail_step_count": i_refractory_tail_step_count,
            "i_refractory_tail_occupancy_stepwise": i_refractory_tail_step_count / float(n_tail_euler_states),
            "over_100hz_step_count": over_100hz_step_count,
            "over_100hz_occupancy_stepwise": over_100hz_step_count / float(n_steps + 1),
            "over_100hz_tail_step_count": over_100hz_tail_step_count,
            "over_100hz_tail_occupancy_stepwise": over_100hz_tail_step_count / float(n_tail_euler_states),
            "stepwise_peak_rE_hz": 1000.0 * stepwise_peak_rE_khz,
            "stepwise_tail_peak_rE_hz": 1000.0 * stepwise_tail_peak_rE_khz,
            "stepwise_min_mu_G": stepwise_min_mu_G,
            "stepwise_max_mu_G": stepwise_max_mu_G,
            "stepwise_min_S_G": stepwise_min_S_G,
            "stepwise_max_S_G": stepwise_max_S_G,
        }
    )
    return traces


_CANDIDATE_LABELS = {"bounded_tonic_candidate", "bounded_oscillatory_candidate"}


def _lut_audit(simulation: Mapping[str, np.ndarray], index: int, tail_fraction: float) -> dict[str, Any]:
    n_time = int(simulation["time_ms"].size)
    start = max(1, int(np.floor((1.0 - tail_fraction) * n_time)))
    mu_e = np.asarray(simulation["muE_mV"])[:, index]
    sigma_e = np.asarray(simulation["sigmaE_mV"])[:, index]
    mu_i = np.asarray(simulation["muI_mV"])[:, index]
    sigma_i = np.asarray(simulation["sigmaI_mV"])[:, index]
    clipped = (
        (mu_e < _LUT_MU[0]) | (mu_e > _LUT_MU[1])
        | (mu_i < _LUT_MU[0]) | (mu_i > _LUT_MU[1])
        | (sigma_e < _LUT_SIG[0]) | (sigma_e > _LUT_SIG[1])
        | (sigma_i < _LUT_SIG[0]) | (sigma_i > _LUT_SIG[1])
    )
    return {
        "lut_clip_any_saved": bool(np.any(clipped)),
        "lut_clip_any_tail": bool(np.any(clipped[start:])),
        "lut_clip_occupancy_saved": float(np.mean(clipped)),
        "lut_clip_occupancy_tail": float(np.mean(clipped[start:])),
    }


def classify_fork_batch(
    metadata: Sequence[Mapping[str, Any]],
    simulation: Mapping[str, np.ndarray],
    thresholds: ForkClassifierThresholds | None = None,
) -> list[dict[str, Any]]:
    """Apply Stage-0B rate criteria plus LUT and unclipped-pool audits."""

    thresholds = (thresholds or ForkClassifierThresholds()).validate()
    rows: list[dict[str, Any]] = []
    for index, meta in enumerate(metadata):
        metrics = classify_rate_trace(simulation["time_ms"], simulation["rE_khz"][:, index], thresholds)
        lut = _lut_audit(simulation, index, thresholds.tail_fraction)
        pool_values = np.column_stack(
            [simulation["rE_fast_khz"][:, index], simulation["mu_G"][:, index], simulation["S_G"][:, index]]
        )
        pool_violation_saved = bool(
            np.any(pool_values < -1e-7)
            or np.any(simulation["mu_G"][:, index] > 1.0 + 1e-5)
            or np.any(simulation["S_G"][:, index] > S_MAX + 1e-5)
        )
        lut_clip_step_count = int(np.asarray(simulation["lut_clip_step_count"])[index])
        lut_clip_tail_step_count = int(np.asarray(simulation["lut_clip_tail_step_count"])[index])
        pool_bound_step_count = int(np.asarray(simulation["pool_bound_step_count"])[index])
        pool_bound_tail_step_count = int(np.asarray(simulation["pool_bound_tail_step_count"])[index])
        rate_bound_step_count = int(np.asarray(simulation["rate_bound_step_count"])[index])
        rate_bound_tail_step_count = int(np.asarray(simulation["rate_bound_tail_step_count"])[index])
        synapse_bound_step_count = int(np.asarray(simulation["synapse_bound_step_count"])[index])
        synapse_bound_tail_step_count = int(np.asarray(simulation["synapse_bound_tail_step_count"])[index])
        over_100hz_tail_step_count = int(np.asarray(simulation["over_100hz_tail_step_count"])[index])
        invalid = (
            lut_clip_step_count > 0
            or pool_bound_step_count > 0
            or rate_bound_step_count > 0
            or synapse_bound_step_count > 0
            or over_100hz_tail_step_count > 0
        )
        if metrics["classification"] in _CANDIDATE_LABELS and invalid:
            metrics["pre_audit_classification"] = metrics["classification"]
            metrics["classification"] = "audit_invalid_candidate"
        rows.append(
            {
                **dict(meta),
                **metrics,
                **lut,
                "pool_bound_violation_saved": pool_violation_saved,
                "lut_clip_any_step": lut_clip_step_count > 0,
                "lut_clip_step_count": lut_clip_step_count,
                "lut_clip_occupancy_stepwise": float(np.asarray(simulation["lut_clip_occupancy_stepwise"])[index]),
                "lut_clip_tail_step_count": lut_clip_tail_step_count,
                "lut_clip_tail_occupancy_stepwise": float(np.asarray(simulation["lut_clip_tail_occupancy_stepwise"])[index]),
                "pool_bound_violation_any_step": pool_bound_step_count > 0,
                "pool_bound_step_count": pool_bound_step_count,
                "pool_bound_occupancy_stepwise": float(np.asarray(simulation["pool_bound_occupancy_stepwise"])[index]),
                "pool_bound_tail_step_count": pool_bound_tail_step_count,
                "pool_bound_tail_occupancy_stepwise": float(np.asarray(simulation["pool_bound_tail_occupancy_stepwise"])[index]),
                "rate_bound_violation_any_step": rate_bound_step_count > 0,
                "rate_bound_step_count": rate_bound_step_count,
                "rate_bound_occupancy_stepwise": float(np.asarray(simulation["rate_bound_occupancy_stepwise"])[index]),
                "rate_bound_tail_step_count": rate_bound_tail_step_count,
                "rate_bound_tail_occupancy_stepwise": float(np.asarray(simulation["rate_bound_tail_occupancy_stepwise"])[index]),
                "synapse_bound_violation_any_step": synapse_bound_step_count > 0,
                "synapse_bound_step_count": synapse_bound_step_count,
                "synapse_bound_occupancy_stepwise": float(np.asarray(simulation["synapse_bound_occupancy_stepwise"])[index]),
                "synapse_bound_tail_step_count": synapse_bound_tail_step_count,
                "synapse_bound_tail_occupancy_stepwise": float(np.asarray(simulation["synapse_bound_tail_occupancy_stepwise"])[index]),
                "negative_rate_step_count": int(np.asarray(simulation["negative_rate_step_count"])[index]),
                "negative_rate_occupancy_stepwise": float(np.asarray(simulation["negative_rate_occupancy_stepwise"])[index]),
                "negative_rate_tail_step_count": int(np.asarray(simulation["negative_rate_tail_step_count"])[index]),
                "negative_rate_tail_occupancy_stepwise": float(np.asarray(simulation["negative_rate_tail_occupancy_stepwise"])[index]),
                "e_refractory_step_count": int(np.asarray(simulation["e_refractory_step_count"])[index]),
                "e_refractory_occupancy_stepwise": float(np.asarray(simulation["e_refractory_occupancy_stepwise"])[index]),
                "e_refractory_tail_step_count": int(np.asarray(simulation["e_refractory_tail_step_count"])[index]),
                "e_refractory_tail_occupancy_stepwise": float(np.asarray(simulation["e_refractory_tail_occupancy_stepwise"])[index]),
                "i_refractory_step_count": int(np.asarray(simulation["i_refractory_step_count"])[index]),
                "i_refractory_occupancy_stepwise": float(np.asarray(simulation["i_refractory_occupancy_stepwise"])[index]),
                "i_refractory_tail_step_count": int(np.asarray(simulation["i_refractory_tail_step_count"])[index]),
                "i_refractory_tail_occupancy_stepwise": float(np.asarray(simulation["i_refractory_tail_occupancy_stepwise"])[index]),
                "over_100hz_step_count": int(np.asarray(simulation["over_100hz_step_count"])[index]),
                "over_100hz_occupancy_stepwise": float(np.asarray(simulation["over_100hz_occupancy_stepwise"])[index]),
                "over_100hz_tail_step_count": over_100hz_tail_step_count,
                "over_100hz_tail_occupancy_stepwise": float(np.asarray(simulation["over_100hz_tail_occupancy_stepwise"])[index]),
                "stepwise_peak_rE_hz": float(np.asarray(simulation["stepwise_peak_rE_hz"])[index]),
                "stepwise_tail_peak_rE_hz": float(np.asarray(simulation["stepwise_tail_peak_rE_hz"])[index]),
                "mu_G_min_saved": float(np.nanmin(simulation["mu_G"][:, index])),
                "mu_G_max_saved": float(np.nanmax(simulation["mu_G"][:, index])),
                "S_G_min_saved": float(np.nanmin(simulation["S_G"][:, index])),
                "S_G_max_saved": float(np.nanmax(simulation["S_G"][:, index])),
                "mu_G_min_stepwise": float(np.asarray(simulation["stepwise_min_mu_G"])[index]),
                "mu_G_max_stepwise": float(np.asarray(simulation["stepwise_max_mu_G"])[index]),
                "S_G_min_stepwise": float(np.asarray(simulation["stepwise_min_S_G"])[index]),
                "S_G_max_stepwise": float(np.asarray(simulation["stepwise_max_S_G"])[index]),
                "divisor_min_saved": float(np.nanmin(simulation["divisor"][:, index])),
                "divisor_max_saved": float(np.nanmax(simulation["divisor"][:, index])),
                "pool_numerical_clip": False,
            }
        )
    return rows


def select_confirm_candidates(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return [
        index
        for index, row in enumerate(rows)
        if row["classification"] in _CANDIDATE_LABELS
        and row["initial_kind"] != "exact_root"
        and float(row["tail_peak_hz"]) < 100.0
        and not bool(row["lut_clip_any_step"])
        and not bool(row["pool_bound_violation_any_step"])
        and not bool(row["rate_bound_violation_any_step"])
        and not bool(row["synapse_bound_violation_any_step"])
        and int(row["over_100hz_tail_step_count"]) == 0
    ]


def summarize_stage0c(
    root_rows: Sequence[Mapping[str, Any]],
    screen_rows: Sequence[Mapping[str, Any]],
    confirm_rows: Sequence[Mapping[str, Any]],
    *,
    alpha0_parity: Mapping[str, Any],
    ablation_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Fail-closed verdict for the independent dynamic-pool screen."""

    dynamical = [row for row in screen_rows if row.get("initial_kind") != "exact_root"]
    screen_counts = {
        label: int(sum(row["classification"] == label for row in dynamical))
        for label in sorted({str(row["classification"]) for row in dynamical})
    }
    confirmed = [
        dict(row)
        for row in confirm_rows
        if row["classification"] in _CANDIDATE_LABELS
        and not bool(row.get("lut_clip_any_step", False))
        and not bool(row.get("pool_bound_violation_any_step", False))
        and not bool(row.get("rate_bound_violation_any_step", False))
        and not bool(row.get("synapse_bound_violation_any_step", False))
        and int(row.get("over_100hz_tail_step_count", 0)) == 0
        and float(row.get("stepwise_tail_peak_rE_hz", row.get("tail_peak_hz", 0.0))) < 100.0
    ]
    roots = [root for point in root_rows for root in point.get("roots", [])]
    stable_finite = [
        root for root in roots
        if root["branch_class"] == "finite_high_root"
        and root["stability"] == "stable"
        and not bool(root["lut_clip_at_root"])
    ]
    uncertain_labels = {
        "numerical_divergence",
        "indeterminate_long_transient",
        "bounded_indeterminate",
        "audit_invalid_candidate",
    }
    uncertain = bool(set(screen_counts) & uncertain_labels)
    only_low_or_saturation = bool(screen_counts) and set(screen_counts).issubset(
        {"low_fixed_point", "saturation_or_over_100hz"}
    )
    root_lookup = {
        (round(float(point["z"]), 8), round(float(point["alpha_G"]), 8)): point.get("roots", [])
        for point in root_rows
    }

    # A single fork is not an attractor/basin.  A supported point requires two
    # different non-exact histories converging to the same finite object.  Tonic
    # support must additionally coincide with a stable, unclipped 9D root.
    point_support: list[dict[str, Any]] = []
    keys = sorted({(float(row["z"]), float(row["alpha_G"])) for row in confirmed})
    for z_value, alpha_value in keys:
        point_rows = [
            row for row in confirmed
            if np.isclose(float(row["z"]), z_value)
            and np.isclose(float(row["alpha_G"]), alpha_value)
            and row.get("initial_kind") != "exact_root"
        ]
        for object_class in sorted({str(row["classification"]) for row in point_rows}):
            members = [row for row in point_rows if row["classification"] == object_class]
            labels = sorted({str(row["initial_label"]) for row in members})
            rates = np.asarray([float(row["tail_mean_hz"]) for row in members], dtype=float)
            rate_mean = float(np.mean(rates)) if rates.size else np.nan
            rate_spread = float(np.ptp(rates)) if rates.size else np.inf
            rate_match = bool(rates.size >= 2 and rate_spread <= max(5.0, 0.20 * rate_mean))
            frequency_match = True
            frequency_mean: float | None = None
            if object_class == "bounded_oscillatory_candidate":
                frequencies = np.asarray([float(row["dominant_frequency_hz"]) for row in members])
                frequency_mean = float(np.mean(frequencies))
                frequency_match = bool(
                    frequencies.size >= 2
                    and float(np.ptp(frequencies)) <= max(1.0, 0.25 * frequency_mean)
                )
            point_roots = root_lookup.get((round(z_value, 8), round(alpha_value, 8)), [])
            matching_roots = [
                root for root in point_roots
                if root["branch_class"] == "finite_high_root"
                and root["stability"] == "stable"
                and not bool(root["lut_clip_at_root"])
                and abs(float(root["rE_hz"]) - rate_mean) <= max(5.0, 0.20 * rate_mean)
            ]
            root_requirement = object_class == "bounded_oscillatory_candidate" or bool(matching_roots)
            supported = bool(len(labels) >= 2 and rate_match and frequency_match and root_requirement)
            point_support.append(
                {
                    "z": z_value,
                    "alpha_G": alpha_value,
                    "object_class": object_class,
                    "supported": supported,
                    "n_confirmed_nonexact_forks": len(members),
                    "n_distinct_initial_conditions": len(labels),
                    "initial_labels": labels,
                    "tail_mean_hz": rate_mean,
                    "tail_rate_spread_hz": rate_spread,
                    "dominant_frequency_hz": frequency_mean,
                    "rate_match": rate_match,
                    "frequency_match": frequency_match,
                    "n_matching_stable_unclipped_roots": len(matching_roots),
                    "tonic_root_requirement_pass": root_requirement,
                }
            )

    supported_points = [point for point in point_support if point["supported"]]
    z_axis = sorted({float(point["z"]) for point in root_rows})
    alpha_axis = sorted({float(point["alpha_G"]) for point in root_rows})
    z1_low_baseline_by_alpha = {
        float(alpha): bool(
            any(
                root["branch_class"] == "low_root"
                and root["stability"] == "stable"
                and float(root["rE_hz"]) < 5.0
                and not bool(root["lut_clip_at_root"])
                for root in root_lookup.get((1.0, round(float(alpha), 8)), [])
            )
        )
        for alpha in alpha_axis
    }

    def adjacent(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        same_z = np.isclose(float(left["z"]), float(right["z"]))
        same_alpha = np.isclose(float(left["alpha_G"]), float(right["alpha_G"]))
        if same_z and not same_alpha:
            return abs(alpha_axis.index(float(left["alpha_G"])) - alpha_axis.index(float(right["alpha_G"]))) == 1
        if same_alpha and not same_z:
            return abs(z_axis.index(float(left["z"])) - z_axis.index(float(right["z"]))) == 1
        return False

    def same_object(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        if left["object_class"] != right["object_class"]:
            return False
        mean_rate = 0.5 * (float(left["tail_mean_hz"]) + float(right["tail_mean_hz"]))
        if abs(float(left["tail_mean_hz"]) - float(right["tail_mean_hz"])) > max(5.0, 0.20 * mean_rate):
            return False
        if left["object_class"] == "bounded_oscillatory_candidate":
            f_left = float(left["dominant_frequency_hz"])
            f_right = float(right["dominant_frequency_hz"])
            if abs(f_left - f_right) > max(1.0, 0.25 * 0.5 * (f_left + f_right)):
                return False
        return True

    # The alpha=0 same-z counterfactual must not already contain the object.
    def alpha0_absent(point: Mapping[str, Any]) -> bool:
        for control in confirmed:
            if not (
                np.isclose(float(control["z"]), float(point["z"]))
                and np.isclose(float(control["alpha_G"]), 0.0)
                and control["classification"] == point["object_class"]
            ):
                continue
            control_rate = float(control["tail_mean_hz"])
            point_rate = float(point["tail_mean_hz"])
            if abs(control_rate - point_rate) <= max(5.0, 0.20 * 0.5 * (control_rate + point_rate)):
                if point["object_class"] != "bounded_oscillatory_candidate":
                    return False
                control_frequency = float(control["dominant_frequency_hz"])
                point_frequency = float(point["dominant_frequency_hz"])
                if abs(control_frequency - point_frequency) <= max(
                    1.0, 0.25 * 0.5 * (control_frequency + point_frequency)
                ):
                    return False
        for control in supported_points:
            if (
                np.isclose(float(control["z"]), float(point["z"]))
                and np.isclose(float(control["alpha_G"]), 0.0)
                and same_object(control, point)
            ):
                return False
        for root in root_lookup.get((round(float(point["z"]), 8), 0.0), []):
            if (
                root["branch_class"] == "finite_high_root"
                and root["stability"] == "stable"
                and not bool(root["lut_clip_at_root"])
                and abs(float(root["rE_hz"]) - float(point["tail_mean_hz"]))
                <= max(5.0, 0.20 * float(point["tail_mean_hz"]))
            ):
                return False
        return True

    adjacent_support_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(supported_points):
        for right in supported_points[left_index + 1 :]:
            if (
                float(left["alpha_G"]) > 0.0
                and float(right["alpha_G"]) > 0.0
                and adjacent(left, right)
                and same_object(left, right)
                and alpha0_absent(left)
                and alpha0_absent(right)
                and z1_low_baseline_by_alpha.get(float(left["alpha_G"]), False)
                and z1_low_baseline_by_alpha.get(float(right["alpha_G"]), False)
            ):
                adjacent_support_pairs.append(
                    {
                        "left": {key: left[key] for key in ("z", "alpha_G", "object_class", "tail_mean_hz")},
                        "right": {key: right[key] for key in ("z", "alpha_G", "object_class", "tail_mean_hz")},
                        "alpha0_counterfactual_absent": True,
                        "z1_stable_low_preserved": True,
                    }
                )

    robust_object = bool(adjacent_support_pairs)
    parity_pass = bool(alpha0_parity.get("pass", False))
    if parity_pass and robust_object:
        verdict = "GO_DYNAMIC_POOL_FINITE_FAST_OBJECT"
        passed = True
        reason = "动态除法池产生了跨相邻参数格、每格至少两个不同初态确认且alpha=0无对应物的有限快态对象。"
    elif parity_pass and only_low_or_saturation and not uncertain and not stable_finite:
        verdict = "CLEAN_NO_GO_DYNAMIC_POOL_LOW_OR_SATURATION_ONLY"
        passed = False
        reason = "锁定网格只有低态或超过100 Hz饱和态；不开phi、慢变量或空间层。"
    elif not parity_pass:
        verdict = "ENGINEERING_FAIL_ALPHA0_PARITY"
        passed = False
        reason = "alpha_G=0未复刻Stage0B，Stage0C结果不可解释。"
    else:
        verdict = "INCONCLUSIVE_NO_CONFIRMED_FINITE_FAST_OBJECT"
        passed = False
        reason = "未确认有限快态，且仍有长瞬态、clip依赖或未决轨迹。"

    arm_counts: dict[str, dict[str, int]] = {}
    for mechanism in sorted({str(row.get("mechanism")) for row in ablation_rows}):
        subset = [row for row in ablation_rows if str(row.get("mechanism")) == mechanism]
        arm_counts[mechanism] = {
            label: int(sum(row["classification"] == label for row in subset))
            for label in sorted({str(row["classification"]) for row in subset})
        }
    dynamic_osc = any(
        row.get("mechanism") == "dynamic"
        and row.get("classification") == "bounded_oscillatory_candidate"
        for row in ablation_rows
    )
    non_dynamic_osc = any(
        row.get("mechanism") != "dynamic"
        and row.get("classification") == "bounded_oscillatory_candidate"
        for row in ablation_rows
    )
    dynamic_specific_oscillation = bool(dynamic_osc and not non_dynamic_osc)
    if not ablation_rows:
        ablation_interpretation = "not_run_no_confirmed_candidate"
    elif dynamic_specific_oscillation:
        ablation_interpretation = "supports_delay_specific_oscillatory_object"
    else:
        ablation_interpretation = (
            "does_not_isolate_delay; any finite-state result is attributable only to nonlinear_gain_compression"
        )
    return {
        "verdict": verdict,
        "stage0c_pass": passed,
        "open_phi_or_spatial": passed,
        "stop_rule_triggered": verdict == "CLEAN_NO_GO_DYNAMIC_POOL_LOW_OR_SATURATION_ONLY",
        "reason_cn": reason,
        "screen_classification_counts": screen_counts,
        "all_forks_classification_counts": {
            label: int(sum(row["classification"] == label for row in screen_rows))
            for label in sorted({str(row["classification"]) for row in screen_rows})
        },
        "initial_kind_counts": {
            label: int(sum(row["initial_kind"] == label for row in screen_rows))
            for label in sorted({str(row["initial_kind"]) for row in screen_rows})
        },
        "n_parameter_points": len(root_rows),
        "n_roots": len(roots),
        "n_stable_finite_unclipped_roots": len(stable_finite),
        "n_confirmed_candidates": len(confirmed),
        "confirmed_candidates": confirmed,
        "candidate_point_support": point_support,
        "n_supported_parameter_points": len(supported_points),
        "adjacent_support_pairs": adjacent_support_pairs,
        "n_adjacent_support_pairs": len(adjacent_support_pairs),
        "go_requires_two_initial_conditions_per_point": True,
        "go_requires_adjacent_parameter_points": True,
        "go_requires_alpha0_counterfactual_absence": True,
        "go_requires_z1_stable_low_unclipped": True,
        "z1_stable_low_unclipped_by_alpha": {
            f"{alpha:g}": passed for alpha, passed in z1_low_baseline_by_alpha.items()
        },
        "alpha0_parity": dict(alpha0_parity),
        "ablation_run": bool(ablation_rows),
        "n_ablation_rows": len(ablation_rows),
        "ablation_contract": ["dynamic", "instantaneous", "clamped", "matched_subtractive", "mean_only"],
        "ablation_classification_counts_by_arm": arm_counts,
        "dynamic_specific_oscillation": dynamic_specific_oscillation,
        "ablation_interpretation": ablation_interpretation,
        "contract": {
            "state_dimension": 9,
            "frozen_z": True,
            "frozen_local_recovery": 0.0,
            "recurrent_e_mean": "divide_by_D",
            "recurrent_e_variance": "divide_by_D_squared",
            "other_moments": "unchanged",
            "pool_ode_numerical_clip": False,
            "candidate_ceiling_hz": 100.0,
            "downstream_stop": "no candidate => no phi, slow, or spatial expansion",
        },
    }
