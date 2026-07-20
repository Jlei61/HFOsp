"""Stage 0B: homogeneous frozen E/I fast-topology screen.

This module is deliberately independent of the Stage-0A normal-form oracle and of
the full SNN/conductance implementation.  It reuses the audited Brunel/LIF constants
and the six-state M3B layout, but its nonlinear RHS recomputes *both* ``mu`` and
``sigma`` from the instantaneous synaptic states.  The existing M3B ``field_rhs``
freezes sigma at an operating point by design and is therefore unsuitable for the
large-amplitude state forks used here.

Rates are in kHz and time is in ms.  The frozen state is
``[rE, rI, sEE, sEI, sIE, sII]``.  In a homogeneous field all normalized spatial
kernels reduce to the identity on a constant, so the four synaptic filters target
``rE, rI, rE, rI`` respectively.
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
    lif_rate,
    nu_theta_pop,
)
from src.topic4_m3b_spectral_phase import _LUT_MU, _LUT_SIG, _phi_field


STATE_FIELDS: tuple[str, ...] = ("rE", "rI", "sEE", "sEI", "sIE", "sII")
E_CEILING_KHZ: float = 1.0 / TREF_E
I_CEILING_KHZ: float = 1.0 / TREF_I
FINITE_HIGH_MAX_KHZ: float = 0.100


@dataclass(frozen=True)
class FastParameters:
    """One frozen homogeneous E/I parameter point."""

    w_ee_mult: float
    q: float
    ratio: float = 1.0

    def validate(self) -> "FastParameters":
        values = (self.w_ee_mult, self.q, self.ratio)
        if not all(np.isfinite(values)):
            raise ValueError("fast parameters must be finite")
        if self.w_ee_mult <= 0 or not 0 < self.q <= 1 or self.ratio <= 0:
            raise ValueError("require w_ee_mult>0, 0<q<=1, ratio>0")
        return self


@dataclass(frozen=True)
class ForkClassifierThresholds:
    """Fail-closed thresholds for the finite-amplitude state-fork classifier."""

    tail_fraction: float = 0.40
    low_mean_hz: float = 5.0
    finite_high_max_hz: float = 100.0
    fixed_sd_floor_hz: float = 0.5
    fixed_cv: float = 0.03
    max_relative_drift: float = 0.08
    max_abs_slope_hz_s: float = 1.0
    min_oscillation_p2p_hz: float = 2.0
    min_oscillation_cycles: float = 3.0
    min_spectral_power_ratio: float = 0.20
    refractory_margin_fraction: float = 0.05
    max_refractory_occupancy: float = 0.05

    def validate(self) -> "ForkClassifierThresholds":
        if not 0.1 <= self.tail_fraction <= 0.9:
            raise ValueError("tail_fraction must be in [0.1, 0.9]")
        if not 0 < self.low_mean_hz < self.finite_high_max_hz:
            raise ValueError("require 0 < low_mean_hz < finite_high_max_hz")
        if not 0 < self.fixed_cv < 1 or not 0 < self.max_relative_drift < 1:
            raise ValueError("relative classifier thresholds must lie in (0, 1)")
        if self.min_oscillation_cycles < 2:
            raise ValueError("min_oscillation_cycles must be >=2")
        return self


def _transfer(mu: np.ndarray, sigma: np.ndarray, pop: str) -> np.ndarray:
    """Reuse M3B's audited vectorized LIF table without freezing sigma."""

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    shape = np.broadcast_shapes(mu.shape, sigma.shape)
    return _phi_field(np.broadcast_to(mu, shape), np.broadcast_to(sigma, shape), pop)


def moments_from_state(
    state: np.ndarray, params: FastParameters | Sequence[FastParameters]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return instantaneous ``muE, sigmaE, muI, sigmaI``.

    Unlike ``topic4_m3b_spectral_phase.field_rhs``, sigma is computed from the
    current synaptic states on every call.  ``state`` can be ``(6,)`` or ``(n,6)``;
    a matching sequence of parameters enables a vectorized parameter/fork batch.
    """

    state = np.asarray(state, dtype=float)
    one = state.ndim == 1
    if one:
        state = state[None, :]
    if state.ndim != 2 or state.shape[1] != 6:
        raise ValueError("state must have shape (6,) or (n,6)")
    if isinstance(params, FastParameters):
        checked = params.validate()
        w_mult = np.full(state.shape[0], checked.w_ee_mult)
        q = np.full(state.shape[0], checked.q)
        ratio = np.full(state.shape[0], checked.ratio)
    else:
        checked_list = [p.validate() for p in params]
        if len(checked_list) != state.shape[0]:
            raise ValueError("one FastParameters object is required per state")
        w_mult = np.asarray([p.w_ee_mult for p in checked_list])
        q = np.asarray([p.q for p in checked_list])
        ratio = np.asarray([p.ratio for p in checked_list])

    sEE, sEI, sIE, sII = (state[:, i] for i in range(2, 6))
    wee = w_mult * W_EE
    w_ei = q * W_EI
    nuext = ratio * nu_theta_pop()
    mu_e = TAU_ME * (C_EE * wee * sEE - C_EI * w_ei * sEI) + TAU_ME * JX_E * nuext
    mu_i = TAU_MI * (C_IE * W_IE * sIE - C_II * W_II * sII) + TAU_MI * JX_I * nuext
    var_e = TAU_ME * (C_EE * wee**2 * sEE + C_EI * w_ei**2 * sEI) + TAU_ME * JX_E**2 * nuext
    var_i = TAU_MI * (C_IE * W_IE**2 * sIE + C_II * W_II**2 * sII) + TAU_MI * JX_I**2 * nuext
    out = (
        mu_e,
        np.sqrt(np.maximum(var_e, 1e-9)),
        mu_i,
        np.sqrt(np.maximum(var_i, 1e-9)),
    )
    if one:
        return tuple(x[0] for x in out)  # type: ignore[return-value]
    return out


def fast_rhs(
    state: np.ndarray, params: FastParameters | Sequence[FastParameters]
) -> np.ndarray:
    """Self-consistent nonlinear six-state RHS in kHz/ms."""

    state = np.asarray(state, dtype=float)
    one = state.ndim == 1
    batch = state[None, :] if one else state
    if batch.ndim != 2 or batch.shape[1] != 6:
        raise ValueError("state must have shape (6,) or (n,6)")
    mu_e, sigma_e, mu_i, sigma_i = moments_from_state(batch, params)
    target_e = _transfer(mu_e, sigma_e, "E")
    target_i = _transfer(mu_i, sigma_i, "I")
    out = np.empty_like(batch)
    out[:, 0] = (-batch[:, 0] + target_e) / TAU_ME
    out[:, 1] = (-batch[:, 1] + target_i) / TAU_MI
    out[:, 2] = (batch[:, 0] - batch[:, 2]) / TAU_AMPA
    out[:, 3] = (batch[:, 1] - batch[:, 3]) / TAU_GABA
    out[:, 4] = (batch[:, 0] - batch[:, 4]) / TAU_AMPA
    out[:, 5] = (batch[:, 1] - batch[:, 5]) / TAU_GABA
    return out[0] if one else out


def lut_clip_audit(
    mu_e: np.ndarray,
    sigma_e: np.ndarray,
    mu_i: np.ndarray,
    sigma_i: np.ndarray,
    *,
    tail_fraction: float = 0.40,
) -> list[dict[str, Any]]:
    """Audit M3B transfer-LUT clipping over saved frames, per fork.

    ``_phi_field`` intentionally clips to its tabulated support.  That is acceptable
    for a cheap scan only if it is visible: a putative finite-high candidate with
    clipping at any audited saved frame cannot pass Stage0B.  Arrays are ``(time,fork)``.
    """

    arrays = [np.asarray(x, dtype=float) for x in (mu_e, sigma_e, mu_i, sigma_i)]
    if not arrays or arrays[0].ndim != 2 or any(x.shape != arrays[0].shape for x in arrays):
        raise ValueError("moment traces must share shape (time,fork)")
    start = max(1, int(np.floor((1.0 - tail_fraction) * arrays[0].shape[0])))
    clip = (
        (arrays[0] < _LUT_MU[0])
        | (arrays[0] > _LUT_MU[1])
        | (arrays[2] < _LUT_MU[0])
        | (arrays[2] > _LUT_MU[1])
        | (arrays[1] < _LUT_SIG[0])
        | (arrays[1] > _LUT_SIG[1])
        | (arrays[3] < _LUT_SIG[0])
        | (arrays[3] > _LUT_SIG[1])
    )
    out: list[dict[str, Any]] = []
    for index in range(arrays[0].shape[1]):
        out.append(
            {
                "lut_clip_occupancy_saved": float(np.mean(clip[:, index])),
                "lut_clip_occupancy_tail": float(np.mean(clip[start:, index])),
                "lut_clip_any_saved": bool(np.any(clip[:, index])),
                "lut_clip_any_tail": bool(np.any(clip[start:, index])),
                "lut_muE_min_mV": float(np.min(arrays[0][:, index])),
                "lut_muE_max_mV": float(np.max(arrays[0][:, index])),
                "lut_sigmaE_min_mV": float(np.min(arrays[1][:, index])),
                "lut_sigmaE_max_mV": float(np.max(arrays[1][:, index])),
                "lut_muI_min_mV": float(np.min(arrays[2][:, index])),
                "lut_muI_max_mV": float(np.max(arrays[2][:, index])),
                "lut_sigmaI_min_mV": float(np.min(arrays[3][:, index])),
                "lut_sigmaI_max_mV": float(np.max(arrays[3][:, index])),
                "lut_support_mu_mV": [float(_LUT_MU[0]), float(_LUT_MU[1])],
                "lut_support_sigma_mV": [float(_LUT_SIG[0]), float(_LUT_SIG[1])],
                "lut_audit_sampling": "saved_frames",
            }
        )
    return out


def equilibrium_state(rates: Sequence[float]) -> np.ndarray:
    """Lift homogeneous fixed-point rates ``(rE,rI)`` into the six-state system."""

    r_e, r_i = map(float, rates)
    return np.asarray([r_e, r_i, r_e, r_i, r_e, r_i], dtype=float)


def fixed_point_residual(rates: Sequence[float], params: FastParameters) -> np.ndarray:
    """Two-dimensional self-consistency residual shared with ``fast_rhs``."""

    state = equilibrium_state(rates)
    rhs = fast_rhs(state, params)
    return np.asarray([-TAU_ME * rhs[0], -TAU_MI * rhs[1]], dtype=float)


def numerical_jacobian(
    state: Sequence[float] | np.ndarray,
    params: FastParameters,
    *,
    relative_step: float = 2e-4,
    absolute_step: float = 2e-7,
) -> np.ndarray:
    """Centered finite-difference Jacobian of the same self-consistent RHS."""

    state = np.asarray(state, dtype=float)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("state must be finite with shape (6,)")
    jac = np.empty((6, 6), dtype=float)
    for column in range(6):
        h = max(absolute_step, relative_step * max(abs(state[column]), 1e-3))
        plus = state.copy()
        minus = state.copy()
        plus[column] += h
        minus[column] -= h
        jac[:, column] = (fast_rhs(plus, params) - fast_rhs(minus, params)) / (2.0 * h)
    return jac


def fast_rhs_exact_siegert(state: Sequence[float] | np.ndarray, params: FastParameters) -> np.ndarray:
    """Scalar exact-Siegert counterpart used only for root sensitivity audits."""

    state = np.asarray(state, dtype=float)
    if state.shape != (6,):
        raise ValueError("exact Siegert RHS accepts one state with shape (6,)")
    mu_e, sigma_e, mu_i, sigma_i = moments_from_state(state, params)
    out = np.empty(6, dtype=float)
    out[0] = (-state[0] + lif_rate(float(mu_e), float(sigma_e), TAU_ME, TREF_E)) / TAU_ME
    out[1] = (-state[1] + lif_rate(float(mu_i), float(sigma_i), TAU_MI, TREF_I)) / TAU_MI
    out[2] = (state[0] - state[2]) / TAU_AMPA
    out[3] = (state[1] - state[3]) / TAU_GABA
    out[4] = (state[0] - state[4]) / TAU_AMPA
    out[5] = (state[1] - state[5]) / TAU_GABA
    return out


def _exact_fixed_point_residual(rates: Sequence[float], params: FastParameters) -> np.ndarray:
    rhs = fast_rhs_exact_siegert(equilibrium_state(rates), params)
    return np.asarray([-TAU_ME * rhs[0], -TAU_MI * rhs[1]], dtype=float)


def _exact_siegert_jacobian(state: np.ndarray, params: FastParameters) -> np.ndarray:
    jac = np.empty((6, 6), dtype=float)
    for column in range(6):
        h = max(2e-7, 2e-4 * max(abs(float(state[column])), 1e-3))
        plus = state.copy()
        minus = state.copy()
        plus[column] += h
        minus[column] -= h
        jac[:, column] = (
            fast_rhs_exact_siegert(plus, params) - fast_rhs_exact_siegert(minus, params)
        ) / (2.0 * h)
    return jac


def exact_siegert_root_audit(root_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Locally refine every LUT root with the unclipped exact Siegert transfer.

    This is a sensitivity audit, not a replacement dense continuation: it answers
    whether the discovered sub-100-Hz separators remain unstable and whether the
    discovered high attractors remain above 100 Hz when LUT clipping is removed.
    """

    out: list[dict[str, Any]] = []
    for point in root_rows:
        params = FastParameters(float(point["w_ee_mult"]), float(point["q"]), float(point["ratio"]))
        for root_index, root in enumerate(point.get("roots", [])):
            initial = np.asarray([root["rE_khz"], root["rI_khz"]], dtype=float)
            fit = least_squares(
                lambda rates: _exact_fixed_point_residual(rates, params),
                np.clip(initial, [1e-9, 1e-9], [E_CEILING_KHZ, I_CEILING_KHZ]),
                bounds=([1e-9, 1e-9], [E_CEILING_KHZ, I_CEILING_KHZ]),
                xtol=2e-11,
                ftol=2e-11,
                gtol=2e-11,
                max_nfev=300,
            )
            refined = np.asarray(fit.x, dtype=float)
            residual = float(np.linalg.norm(_exact_fixed_point_residual(refined, params), ord=np.inf))
            converged = bool(fit.success and residual <= 2e-7)
            if converged:
                eigenvalues = np.linalg.eigvals(_exact_siegert_jacobian(equilibrium_state(refined), params))
                leading = eigenvalues[int(np.argmax(eigenvalues.real))]
                stability = (
                    "stable"
                    if leading.real < -1e-4
                    else ("unstable" if leading.real > 1e-4 else "marginal")
                )
            else:
                leading = complex(np.nan, np.nan)
                stability = "unresolved"
            exact_e_hz = float(refined[0] * 1000.0)
            if exact_e_hz < 5.0:
                exact_class = "low"
            elif exact_e_hz < 100.0:
                exact_class = "finite_high"
            else:
                exact_class = "over_100hz"
            out.append(
                {
                    "w_ee_mult": params.w_ee_mult,
                    "q": params.q,
                    "source_root_index": root_index,
                    "source_rE_hz": float(root["rE_hz"]),
                    "source_stability": root["stability"],
                    "source_branch_class": root["branch_class"],
                    "source_lut_clipped": bool(root["lut_clip_at_root"]),
                    "exact_converged": converged,
                    "exact_residual_inf": residual,
                    "exact_rE_hz": exact_e_hz,
                    "exact_rI_hz": float(refined[1] * 1000.0),
                    "exact_stability": stability,
                    "exact_rate_class": exact_class,
                    "exact_leading_real_per_ms": float(leading.real),
                    "exact_leading_imag_per_ms": float(abs(leading.imag)),
                    "exact_leading_frequency_hz": float(
                        1000.0 * abs(leading.imag) / (2.0 * np.pi)
                    ),
                    "audit_scope": "local_refinement_of_lut_discovered_root_not_dense_exact_continuation",
                }
            )
    return out


def summarize_exact_siegert_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fail-closed verdict for local exact-Siegert refinement of LUT roots."""

    rows = list(rows)
    converged = [row for row in rows if bool(row["exact_converged"])]
    source_sub100_unstable = [
        row
        for row in rows
        if row["source_stability"] == "unstable" and float(row["source_rE_hz"]) < 100.0
    ]
    source_stable_over100 = [
        row
        for row in rows
        if row["source_stability"] == "stable" and float(row["source_rE_hz"]) >= 100.0
    ]
    exact_stable_finite = [
        row
        for row in converged
        if row["exact_stability"] == "stable" and row["exact_rate_class"] == "finite_high"
    ]
    failure_reasons: list[str] = []
    if len(converged) != len(rows):
        failure_reasons.append("not_all_source_roots_converged")
    if any(row["exact_stability"] != "unstable" for row in source_sub100_unstable):
        failure_reasons.append("source_sub100_unstable_not_preserved")
    if any(
        row["exact_stability"] != "stable" or row["exact_rate_class"] != "over_100hz"
        for row in source_stable_over100
    ):
        failure_reasons.append("source_stable_over100_not_preserved")
    if exact_stable_finite:
        failure_reasons.append("exact_stable_finite_high_found")
    return {
        "scope": "local_refinement_of_lut_discovered_roots_not_dense_exact_continuation",
        "n_source_roots": len(rows),
        "n_converged": len(converged),
        "n_stable_finite_high": len(exact_stable_finite),
        "n_source_sub100_unstable": len(source_sub100_unstable),
        "n_source_sub100_unstable_remaining_unstable": int(
            sum(row["exact_stability"] == "unstable" for row in source_sub100_unstable)
        ),
        "n_source_stable_over100": len(source_stable_over100),
        "n_source_stable_over100_remaining_stable_over100": int(
            sum(
                row["exact_stability"] == "stable" and row["exact_rate_class"] == "over_100hz"
                for row in source_stable_over100
            )
        ),
        "supports_lut_no_go": not failure_reasons,
        "failure_reasons": failure_reasons,
    }


def _default_root_seeds() -> list[tuple[float, float]]:
    e = (1e-6, 2.5e-4, 1e-3, 5e-3, 2e-2, 8e-2, 0.20, 0.45)
    i = (1e-6, 1e-3, 5e-3, 2e-2, 8e-2, 0.20, 0.50, 0.90)
    return [(a, b) for a in e for b in i]


def find_fixed_points(
    params: FastParameters,
    *,
    warm_roots: Iterable[Sequence[float]] = (),
    residual_tolerance: float = 2e-7,
    cluster_tolerance_khz: float = 2e-5,
) -> list[dict[str, Any]]:
    """Find and classify homogeneous roots from dense and continuation seeds."""

    params.validate()
    seeds = list(warm_roots) + _default_root_seeds()
    roots: list[np.ndarray] = []
    for seed in seeds:
        seed_arr = np.clip(np.asarray(seed, dtype=float), [1e-9, 1e-9], [E_CEILING_KHZ, I_CEILING_KHZ])
        fit = least_squares(
            lambda rates: fixed_point_residual(rates, params),
            seed_arr,
            bounds=([1e-9, 1e-9], [E_CEILING_KHZ, I_CEILING_KHZ]),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=300,
        )
        root = np.asarray(fit.x, dtype=float)
        resid = float(np.linalg.norm(fixed_point_residual(root, params), ord=np.inf))
        if not fit.success or resid > residual_tolerance or not np.all(np.isfinite(root)):
            continue
        if not any(float(np.linalg.norm(root - old, ord=np.inf)) <= cluster_tolerance_khz for old in roots):
            roots.append(root)

    out: list[dict[str, Any]] = []
    for root in sorted(roots, key=lambda x: (x[0], x[1])):
        state = equilibrium_state(root)
        mu_e, sigma_e, mu_i, sigma_i = moments_from_state(state, params)
        lut_clip = bool(
            mu_e < _LUT_MU[0]
            or mu_e > _LUT_MU[1]
            or mu_i < _LUT_MU[0]
            or mu_i > _LUT_MU[1]
            or sigma_e < _LUT_SIG[0]
            or sigma_e > _LUT_SIG[1]
            or sigma_i < _LUT_SIG[0]
            or sigma_i > _LUT_SIG[1]
        )
        jac = numerical_jacobian(state, params)
        eigenvalues = np.linalg.eigvals(jac)
        leading = eigenvalues[int(np.argmax(eigenvalues.real))]
        rate_hz = float(root[0] * 1000.0)
        if rate_hz >= 100.0:
            branch_class = "saturation_cliff_root"
        elif rate_hz <= 5.0:
            branch_class = "low_root"
        else:
            branch_class = "finite_high_root"
        stability = "stable" if leading.real < -1e-4 else ("unstable" if leading.real > 1e-4 else "marginal")
        out.append(
            {
                "rE_khz": float(root[0]),
                "rI_khz": float(root[1]),
                "rE_hz": rate_hz,
                "rI_hz": float(root[1] * 1000.0),
                "residual_inf": float(np.linalg.norm(fixed_point_residual(root, params), ord=np.inf)),
                "leading_real_per_ms": float(leading.real),
                "leading_imag_per_ms": float(abs(leading.imag)),
                "leading_frequency_hz": float(1000.0 * abs(leading.imag) / (2.0 * np.pi)),
                "stability": stability,
                "branch_class": branch_class,
                "muE_mV": float(mu_e),
                "sigmaE_mV": float(sigma_e),
                "muI_mV": float(mu_i),
                "sigmaI_mV": float(sigma_i),
                "lut_clip_at_root": lut_clip,
            }
        )
    return out


def continuation_root_scan(
    w_ee_values: Sequence[float], q_values: Sequence[float], *, ratio: float = 1.0
) -> list[dict[str, Any]]:
    """Dense-root scan with warm continuation along q and across adjacent wEE rows."""

    rows: list[dict[str, Any]] = []
    previous_w: dict[float, list[Sequence[float]]] = {}
    for w_ee in w_ee_values:
        previous_q: list[Sequence[float]] = []
        current_w: dict[float, list[Sequence[float]]] = {}
        for q in q_values:
            warm = list(previous_q) + list(previous_w.get(float(q), []))
            params = FastParameters(float(w_ee), float(q), float(ratio))
            roots = find_fixed_points(params, warm_roots=warm)
            rows.append(
                {
                    "w_ee_mult": float(w_ee),
                    "q": float(q),
                    "ratio": float(ratio),
                    "n_roots": len(roots),
                    "roots": roots,
                }
            )
            previous_q = [[r["rE_khz"], r["rI_khz"]] for r in roots]
            current_w[float(q)] = previous_q
        previous_w = current_w
    return rows


def build_state_forks(root_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], np.ndarray, list[FastParameters]]:
    """Build boundary-spanning probes plus exact/perturbed roots for every grid point."""

    base_probes = (
        ("probe_low", 1e-5, 1e-5),
        ("probe_rest", 5e-4, 2e-3),
        ("probe_5hz", 5e-3, 1e-2),
        ("probe_20hz", 2e-2, 4e-2),
        ("probe_boundary", 8e-2, 1.5e-1),
        ("probe_high", 2e-1, 4e-1),
        ("probe_ceiling", 4.5e-1, 8e-1),
    )
    metadata: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    params_list: list[FastParameters] = []
    for point in root_rows:
        params = FastParameters(float(point["w_ee_mult"]), float(point["q"]), float(point["ratio"]))

        def add(
            kind: str,
            label: str,
            r_e: float,
            r_i: float,
            root_index: int | None = None,
            state_override: Sequence[float] | None = None,
        ) -> None:
            metadata.append(
                {
                    "w_ee_mult": params.w_ee_mult,
                    "q": params.q,
                    "ratio": params.ratio,
                    "initial_kind": kind,
                    "initial_label": label,
                    "root_index": root_index,
                    "initial_rE_hz": float(r_e * 1000.0),
                    "initial_rI_hz": float(r_i * 1000.0),
                }
            )
            state = equilibrium_state((r_e, r_i)) if state_override is None else np.asarray(state_override, dtype=float)
            if state.shape != (6,) or not np.all(np.isfinite(state)) or np.any(state < 0):
                raise ValueError("state-fork initial states must be finite, nonnegative, shape (6,)")
            if state[0] > E_CEILING_KHZ or state[1] > I_CEILING_KHZ:
                raise ValueError("rate initial condition exceeds refractory ceiling")
            states.append(state)
            params_list.append(params)

        for label, r_e, r_i in base_probes:
            add("probe", label, r_e, r_i)
        # Four history-bearing probes leave the steady-synapse manifold while staying
        # physical/nonnegative.  In homogeneous 0D, sEE==sIE and sEI==sII reflect a
        # realizable common presynaptic history; AMPA/GABA loads may nevertheless lag
        # the instantaneous rates because their time constants differ.
        off_manifold = (
            ("e_synapse_loaded_i_low", [0.001, 0.003, 0.020, 0.005, 0.020, 0.005]),
            ("i_synapse_loaded_e_low", [0.001, 0.003, 0.005, 0.020, 0.005, 0.020]),
            ("rate_high_synapse_low", [0.080, 0.150, 0.003, 0.008, 0.003, 0.008]),
            ("rate_low_synapse_high", [0.001, 0.003, 0.020, 0.020, 0.020, 0.020]),
        )
        for label, state in off_manifold:
            add("off_manifold_probe", label, state[0], state[1], state_override=state)
        for root_index, root in enumerate(point["roots"]):
            r_e = float(root["rE_khz"])
            r_i = float(root["rI_khz"])
            add("exact_root", f"root_{root_index}", r_e, r_i, root_index)
            for sign, scale in (("minus", 0.90), ("plus", 1.10)):
                add(
                    "root_perturbation",
                    f"root_{root_index}_{sign}",
                    np.clip(r_e * scale, 1e-9, E_CEILING_KHZ),
                    r_i,
                    root_index,
                )
    return metadata, np.asarray(states, dtype=float), params_list


def simulate_forks(
    initial_states: np.ndarray,
    params: Sequence[FastParameters],
    *,
    dt_ms: float,
    duration_ms: float,
    save_stride: int,
) -> dict[str, np.ndarray]:
    """Vectorized deterministic forward-Euler state forks."""

    state = np.asarray(initial_states, dtype=float).copy()
    if state.ndim != 2 or state.shape[1] != 6 or state.shape[0] != len(params):
        raise ValueError("initial_states and params must align as (n,6) and length n")
    if dt_ms <= 0 or duration_ms <= dt_ms or save_stride < 1:
        raise ValueError("invalid integration contract")
    n_steps = int(round(duration_ms / dt_ms))
    if not np.isclose(n_steps * dt_ms, duration_ms, atol=1e-9):
        raise ValueError("duration_ms must be an integer multiple of dt_ms")
    sample_steps = np.arange(0, n_steps + 1, int(save_stride), dtype=int)
    if sample_steps[-1] != n_steps:
        sample_steps = np.r_[sample_steps, n_steps]
    r_e = np.empty((sample_steps.size, state.shape[0]), dtype=np.float32)
    r_i = np.empty_like(r_e)
    mu_e = np.empty_like(r_e)
    sigma_e = np.empty_like(r_e)
    mu_i = np.empty_like(r_e)
    sigma_i = np.empty_like(r_e)
    r_e[0] = state[:, 0]
    r_i[0] = state[:, 1]
    mu_e[0], sigma_e[0], mu_i[0], sigma_i[0] = moments_from_state(state, params)
    sample_index = 1
    finite = np.ones(state.shape[0], dtype=bool)
    for step in range(1, n_steps + 1):
        state += dt_ms * fast_rhs(state, params)
        now_finite = np.all(np.isfinite(state), axis=1)
        finite &= now_finite
        if not np.all(now_finite):
            state[~now_finite] = np.nan
        if sample_index < sample_steps.size and step == sample_steps[sample_index]:
            r_e[sample_index] = state[:, 0]
            r_i[sample_index] = state[:, 1]
            mu_e[sample_index], sigma_e[sample_index], mu_i[sample_index], sigma_i[sample_index] = (
                moments_from_state(state, params)
            )
            sample_index += 1
    return {
        "time_ms": sample_steps.astype(float) * dt_ms,
        "rE_khz": r_e,
        "rI_khz": r_i,
        "muE_mV": mu_e,
        "sigmaE_mV": sigma_e,
        "muI_mV": mu_i,
        "sigmaI_mV": sigma_i,
        "final_state": state,
        "finite": finite,
    }


def classify_rate_trace(
    time_ms: Sequence[float] | np.ndarray,
    rate_e_khz: Sequence[float] | np.ndarray,
    thresholds: ForkClassifierThresholds | None = None,
) -> dict[str, Any]:
    """Reject ceilings and drifts before accepting a finite high state/orbit."""

    thresholds = (thresholds or ForkClassifierThresholds()).validate()
    time_ms = np.asarray(time_ms, dtype=float)
    rate_hz = 1000.0 * np.asarray(rate_e_khz, dtype=float)
    if time_ms.ndim != 1 or rate_hz.shape != time_ms.shape or time_ms.size < 40:
        raise ValueError("time and rate must be aligned 1D arrays with >=40 samples")
    if not np.all(np.isfinite(time_ms)) or np.any(np.diff(time_ms) <= 0):
        raise ValueError("time_ms must be finite and strictly increasing")
    if not np.all(np.isfinite(rate_hz)):
        return {"classification": "numerical_divergence", "finite": False}

    start = max(1, int(np.floor((1.0 - thresholds.tail_fraction) * rate_hz.size)))
    tail = rate_hz[start:]
    tail_time_s = time_ms[start:] / 1000.0
    mean_hz = float(np.mean(tail))
    sd_hz = float(np.std(tail))
    peak_hz = float(np.max(tail))
    trough_hz = float(np.min(tail))
    split = max(2, tail.size // 2)
    first_mean = float(np.mean(tail[:split]))
    second_mean = float(np.mean(tail[split:]))
    relative_drift = float(abs(second_mean - first_mean) / max(mean_hz, thresholds.low_mean_hz))
    slope_hz_s = float(np.polyfit(tail_time_s, tail, 1)[0])
    refractory_cut_hz = 1000.0 * E_CEILING_KHZ * (1.0 - thresholds.refractory_margin_fraction)
    refractory_occupancy = float(np.mean(tail >= refractory_cut_hz))
    over_100_occupancy = float(np.mean(tail >= thresholds.finite_high_max_hz))

    centered = tail - mean_hz
    dt_s = float(np.median(np.diff(tail_time_s)))
    power = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(centered.size, d=dt_s)
    if power.size > 1 and float(np.sum(power[1:])) > 0:
        peak_index = int(np.argmax(power[1:]) + 1)
        dominant_frequency_hz = float(frequencies[peak_index])
        spectral_power_ratio = float(power[peak_index] / np.sum(power[1:]))
    else:
        dominant_frequency_hz = 0.0
        spectral_power_ratio = 0.0
    cycles = float(dominant_frequency_hz * (tail_time_s[-1] - tail_time_s[0]))

    drifting = bool(
        relative_drift > thresholds.max_relative_drift
        or abs(slope_hz_s) > thresholds.max_abs_slope_hz_s
    )
    fixed_sd_limit = max(thresholds.fixed_sd_floor_hz, thresholds.fixed_cv * max(mean_hz, 1.0))
    if refractory_occupancy > thresholds.max_refractory_occupancy or peak_hz >= (
        thresholds.finite_high_max_hz
    ):
        label = "saturation_or_over_100hz"
    elif drifting:
        label = "indeterminate_long_transient"
    elif mean_hz <= thresholds.low_mean_hz and sd_hz <= fixed_sd_limit:
        label = "low_fixed_point"
    elif mean_hz > thresholds.low_mean_hz and sd_hz <= fixed_sd_limit:
        label = "bounded_tonic_candidate"
    elif (
        mean_hz > thresholds.low_mean_hz
        and peak_hz < thresholds.finite_high_max_hz
        and peak_hz - trough_hz >= thresholds.min_oscillation_p2p_hz
        and cycles >= thresholds.min_oscillation_cycles
        and spectral_power_ratio >= thresholds.min_spectral_power_ratio
    ):
        label = "bounded_oscillatory_candidate"
    else:
        label = "bounded_indeterminate"
    return {
        "classification": label,
        "finite": True,
        "tail_mean_hz": mean_hz,
        "tail_sd_hz": sd_hz,
        "tail_peak_hz": peak_hz,
        "tail_trough_hz": trough_hz,
        "tail_relative_drift": relative_drift,
        "tail_slope_hz_s": slope_hz_s,
        "dominant_frequency_hz": dominant_frequency_hz,
        "spectral_power_ratio": spectral_power_ratio,
        "tail_cycles": cycles,
        "refractory_ceiling_occupancy": refractory_occupancy,
        "over_100hz_occupancy": over_100_occupancy,
        "tail_start_ms": float(time_ms[start]),
    }


_CANDIDATE_LABELS = {"bounded_tonic_candidate", "bounded_oscillatory_candidate"}


def classify_fork_batch(
    metadata: Sequence[Mapping[str, Any]],
    simulation: Mapping[str, np.ndarray],
    thresholds: ForkClassifierThresholds | None = None,
) -> list[dict[str, Any]]:
    """Attach trace classification and LUT-clipping audit to every state fork."""

    thresholds = (thresholds or ForkClassifierThresholds()).validate()
    audits = lut_clip_audit(
        simulation["muE_mV"],
        simulation["sigmaE_mV"],
        simulation["muI_mV"],
        simulation["sigmaI_mV"],
        tail_fraction=thresholds.tail_fraction,
    )
    rows: list[dict[str, Any]] = []
    for index, meta in enumerate(metadata):
        metrics = classify_rate_trace(
            simulation["time_ms"], simulation["rE_khz"][:, index], thresholds
        )
        audit = audits[index]
        if metrics["classification"] in _CANDIDATE_LABELS and audit["lut_clip_any_saved"]:
            metrics["pre_lut_audit_classification"] = metrics["classification"]
            metrics["classification"] = "lut_clipped_candidate_invalid"
        rows.append({**dict(meta), **metrics, **audit})
    return rows


def select_confirm_candidates(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    """Only non-exact, sub-100-Hz bounded screen forks may enter confirmation."""

    return [
        index
        for index, row in enumerate(rows)
        if row["classification"] in _CANDIDATE_LABELS
        and row["initial_kind"] != "exact_root"
        and float(row["tail_peak_hz"]) < 100.0
    ]


def summarize_stage0b(
    root_rows: Sequence[Mapping[str, Any]],
    screen_rows: Sequence[Mapping[str, Any]],
    confirm_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the fail-closed Stage-0B verdict and downstream stop rule."""

    # Exact initialization on an unstable separator root stays there by arithmetic and
    # can look like a tonic state.  It is equation/Jacobian evidence only, never basin
    # evidence.  Verdict counts therefore use only dynamical probes/perturbations.
    dynamical_rows = [row for row in screen_rows if row.get("initial_kind") != "exact_root"]
    labels, counts = np.unique(
        [str(row["classification"]) for row in dynamical_rows], return_counts=True
    )
    screen_counts = {str(label): int(count) for label, count in zip(labels, counts)}
    confirm_labels, confirm_counts_array = np.unique(
        [str(row["classification"]) for row in confirm_rows], return_counts=True
    ) if confirm_rows else (np.asarray([], dtype=str), np.asarray([], dtype=int))
    confirm_counts = {
        str(label): int(count) for label, count in zip(confirm_labels, confirm_counts_array)
    }
    confirmed_candidates = [
        dict(row) for row in confirm_rows if row["classification"] in _CANDIDATE_LABELS
    ]
    has_confirmed = bool(confirmed_candidates)
    uncertain = any(
        label in screen_counts
        for label in ("numerical_divergence", "indeterminate_long_transient", "bounded_indeterminate")
    )
    only_low_or_saturation = bool(screen_counts) and set(screen_counts).issubset(
        {"low_fixed_point", "saturation_or_over_100hz"}
    )
    root_classes = [
        str(root["branch_class"]) for point in root_rows for root in point.get("roots", [])
    ]
    finite_stable_root = any(
        root["branch_class"] == "finite_high_root" and root["stability"] == "stable"
        for point in root_rows
        for root in point.get("roots", [])
    )
    all_roots = [root for point in root_rows for root in point.get("roots", [])]
    nonexact_clipped = [
        row for row in dynamical_rows if bool(row.get("lut_clip_any_saved", False))
    ]

    if has_confirmed and (finite_stable_root or any(
        row["classification"] == "bounded_oscillatory_candidate" for row in confirmed_candidates
    )):
        verdict = "GO_FINITE_HIGH_OBJECT_CONFIRMED"
        stage0b_pass = True
        reason = "存在经细步长长时间确认的低于100 Hz有限高态对象。"
    elif only_low_or_saturation and not uncertain and not finite_stable_root:
        verdict = "CLEAN_NO_GO_LOW_OR_SATURATION_CLIFF_ONLY"
        stage0b_pass = False
        reason = "全网格只有低固定点与超过100 Hz的饱和悬崖；按停止规则关闭Stage 1--3。"
    else:
        verdict = "INCONCLUSIVE_NO_CONFIRMED_FINITE_HIGH_OBJECT"
        stage0b_pass = False
        reason = "未确认有限高态，且仍有长瞬态/未决轨迹；下游保持关闭。"

    return {
        "verdict": verdict,
        "stage0b_pass": stage0b_pass,
        "stage1_to_3_open": stage0b_pass,
        "stop_rule_triggered": verdict == "CLEAN_NO_GO_LOW_OR_SATURATION_CLIFF_ONLY",
        "reason_cn": reason,
        "screen_classification_counts": screen_counts,
        "screen_initial_kind_counts": {
            label: int(sum(row.get("initial_kind") == label for row in screen_rows))
            for label in sorted({str(row.get("initial_kind")) for row in screen_rows})
        },
        "off_manifold_classification_counts": {
            label: int(
                sum(
                    row.get("initial_kind") == "off_manifold_probe"
                    and row["classification"] == label
                    for row in screen_rows
                )
            )
            for label in sorted(
                {
                    str(row["classification"])
                    for row in screen_rows
                    if row.get("initial_kind") == "off_manifold_probe"
                }
            )
        },
        "n_off_manifold_forks": int(
            sum(row.get("initial_kind") == "off_manifold_probe" for row in screen_rows)
        ),
        "all_forks_classification_counts": {
            label: int(sum(str(row["classification"]) == label for row in screen_rows))
            for label in sorted({str(row["classification"]) for row in screen_rows})
        },
        "verdict_fork_contract": (
            "exclude_exact_root; use equilibrium-manifold probes, off-manifold history probes, "
            "root perturbations, and root stability"
        ),
        "confirm_classification_counts": confirm_counts,
        "n_parameter_points": int(len(root_rows)),
        "n_roots": int(len(root_classes)),
        "root_classification_counts": {
            label: int(root_classes.count(label)) for label in sorted(set(root_classes))
        },
        "n_confirmed_candidates": int(len(confirmed_candidates)),
        "confirmed_candidates": confirmed_candidates,
        "lut_audit": {
            "n_nonexact_forks_clipped_saved": int(len(nonexact_clipped)),
            "n_nonexact_forks_clipped_tail": int(
                sum(bool(row.get("lut_clip_any_tail", False)) for row in dynamical_rows)
            ),
            "n_invalidated_finite_candidates": int(
                sum(row["classification"] == "lut_clipped_candidate_invalid" for row in dynamical_rows)
            ),
            "n_roots_clipped": int(sum(bool(root.get("lut_clip_at_root", False)) for root in all_roots)),
            "n_sub100_unstable_roots_clipped": int(
                sum(
                    root["stability"] == "unstable"
                    and root.get("branch_class") != "saturation_cliff_root"
                    and float(root.get("rE_hz", 0.0)) < 100.0
                    and bool(root.get("lut_clip_at_root", False))
                    for root in all_roots
                )
            ),
            "candidate_pass_requires_no_saved_frame_clipping": True,
        },
        "phi_arm_allowed": bool(
            any(
                row["classification"] == "bounded_tonic_candidate"
                and float(row["tail_peak_hz"]) < 100.0
                for row in confirm_rows
            )
        ),
        "phi_arm_run": False,
        "contract": {
            "ratio": 1.0,
            "noise": False,
            "slow_variables": False,
            "spatial_coupling": False,
            "sigma_update": "self_consistent_each_rhs_call",
            "finite_high_ceiling_hz": 100.0,
            "confirm_rule": "only_nonexact_bounded_screen_candidate_below_100hz",
            "dynamic_phi_rule": "allowed_only_after_confirmed_bounded_tonic_below_100hz",
        },
    }


def root_boundary_summary(root_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compact per-wEE q intervals for low, finite-high, and saturation roots."""

    out: list[dict[str, Any]] = []
    for w_ee in sorted({float(row["w_ee_mult"]) for row in root_rows}):
        subset = [row for row in root_rows if float(row["w_ee_mult"]) == w_ee]
        record: dict[str, Any] = {"w_ee_mult": w_ee}
        for branch_class in ("low_root", "finite_high_root", "saturation_cliff_root"):
            qs = [
                float(row["q"])
                for row in subset
                if any(root["branch_class"] == branch_class for root in row.get("roots", []))
            ]
            record[branch_class] = {
                "present": bool(qs),
                "q_min": min(qs) if qs else None,
                "q_max": max(qs) if qs else None,
                "n_q": len(qs),
            }
        stability_groups = {
            "stable_low": lambda root: root["stability"] == "stable" and root["rE_hz"] < 5.0,
            "unstable_separator_below_100hz": lambda root: root["stability"] == "unstable"
            and root["rE_hz"] < 100.0,
            "stable_over_100hz": lambda root: root["stability"] == "stable"
            and root["rE_hz"] >= 100.0,
        }
        for label, predicate in stability_groups.items():
            qs = [
                float(row["q"])
                for row in subset
                if any(predicate(root) for root in row.get("roots", []))
            ]
            record[label] = {
                "present": bool(qs),
                "q_min": min(qs) if qs else None,
                "q_max": max(qs) if qs else None,
                "n_q": len(qs),
            }
        out.append(record)
    return out
