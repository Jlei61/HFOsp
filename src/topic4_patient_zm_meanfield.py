"""Patient-matched coarse mean-field bridge for the spatial Z/M SNN.

The frozen SNN connectivity matrices store synaptic-rise jumps rather than
physical per-spike voltages.  This module converts them back to physical edge
weights, bins the *realized* patient-specific graph, and constructs a
diffusion-approximation LIF fixed-point system.  It is deliberately separate
from the generic Gaussian-kernel M3B atlas: a bifurcation claim is only
meaningful when the deterministic reduction and the SNN share the same graph,
threshold field, q target, external drive and M convention.

Rates use spikes/ms (numerically kHz), voltages use mV and time uses ms.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.optimize import root
from scipy.special import erfcx


@dataclass(frozen=True)
class PatientCoarseZMModel:
    """Coarse deterministic reduction of one realized patient SNN graph."""

    n_grid: int
    sheet_l_mm: float
    count_e: np.ndarray
    count_i: np.ndarray
    threshold_nodes_e: np.ndarray
    threshold_weights_e: np.ndarray
    w_ee: np.ndarray
    w_ei: np.ndarray
    w_ie: np.ndarray
    w_ii: np.ndarray
    v_ee: np.ndarray
    v_ei: np.ndarray
    v_ie: np.ndarray
    v_ii: np.ndarray
    tau_mem_e_ms: float
    tau_mem_i_ms: float
    tau_ref_e_ms: float
    tau_ref_i_ms: float
    tau_ampa_ms: float
    tau_gaba_ms: float
    v_reset_mv: float
    v_threshold_i_mv: float
    j_ext_e_mv: float
    j_ext_i_mv: float
    nu_ext_per_ms: float

    @property
    def n_cells(self) -> int:
        return int(self.n_grid) ** 2

    def validate(self) -> None:
        n = self.n_cells
        vectors = {
            "count_e": self.count_e,
            "count_i": self.count_i,
        }
        for name, value in vectors.items():
            if np.asarray(value).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
            if np.any(np.asarray(value) <= 0):
                raise ValueError(
                    f"every coarse cell must contain both populations; {name} has empty cells")
        if self.threshold_nodes_e.shape != self.threshold_weights_e.shape:
            raise ValueError("threshold nodes and weights must share shape")
        if self.threshold_nodes_e.shape[0] != n:
            raise ValueError("threshold support must have one row per coarse cell")
        if not np.allclose(self.threshold_weights_e.sum(axis=1), 1.0):
            raise ValueError("threshold weights must sum to one in every cell")
        for name in ("w_ee", "w_ei", "w_ie", "w_ii",
                     "v_ee", "v_ei", "v_ie", "v_ii"):
            value = np.asarray(getattr(self, name))
            if value.shape != (n, n):
                raise ValueError(f"{name} must have shape ({n}, {n})")
            if not np.all(np.isfinite(value)) or np.any(value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if np.any(self.threshold_nodes_e < self.v_reset_mv):
            raise ValueError("E threshold support falls below reset")


@dataclass(frozen=True)
class FixedPointSolution:
    """One audited fixed-point solve; rates remain in spikes/ms."""

    rates: np.ndarray
    q: float
    eta_m: float
    tau_m_slow_ms: float
    converged: bool
    physical: bool
    residual_inf: float
    nfev: int
    njev: int
    message: str

    @property
    def rate_e(self) -> np.ndarray:
        return self.rates[:self.rates.size // 2]

    @property
    def rate_i(self) -> np.ndarray:
        return self.rates[self.rates.size // 2:]

    @property
    def mean_rate_e_hz(self) -> float:
        return 1000.0 * float(np.mean(self.rate_e))

    @property
    def mean_rate_i_hz(self) -> float:
        return 1000.0 * float(np.mean(self.rate_i))


@dataclass(frozen=True)
class PseudoArclengthPoint:
    """One corrected point and tangent on an arclength branch."""

    solution: FixedPointSolution
    tangent_rates: np.ndarray
    tangent_q: float
    corrector_iterations: int
    step_size: float


def spatial_cell_index(positions_xy, *, n_grid: int, sheet_l_mm: float) -> np.ndarray:
    """Map [0,L]^2 positions to row-major ``y*n_grid+x`` cell indices."""
    positions = np.asarray(positions_xy, float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (n, 2)")
    if n_grid < 1 or sheet_l_mm <= 0.0:
        raise ValueError("n_grid and sheet_l_mm must be positive")
    scaled = np.floor(positions / float(sheet_l_mm) * int(n_grid)).astype(int)
    np.clip(scaled, 0, int(n_grid) - 1, out=scaled)
    return scaled[:, 1] * int(n_grid) + scaled[:, 0]


def grouped_threshold_support(values, groups, *, n_groups: int,
                              n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Represent each cell's empirical thresholds by weighted sorted chunks.

    Chunk means preserve the cell mean exactly and approximate the nonlinear
    threshold integral without assuming a Gaussian distribution.
    """
    values = np.asarray(values, float)
    groups = np.asarray(groups, int)
    if values.shape != groups.shape:
        raise ValueError("values and groups must align")
    if n_groups < 1:
        raise ValueError("n_groups must be positive")
    nodes = np.zeros((int(n_cells), int(n_groups)), float)
    weights = np.zeros_like(nodes)
    for cell in range(int(n_cells)):
        selected = np.sort(values[groups == cell])
        if selected.size == 0:
            raise ValueError(f"threshold cell {cell} is empty")
        chunks = np.array_split(selected, min(int(n_groups), selected.size))
        for index, chunk in enumerate(chunks):
            nodes[cell, index] = float(np.mean(chunk))
            weights[cell, index] = float(chunk.size / selected.size)
        # Zero-weight padding gets a valid threshold so vectorized evaluation
        # never enters an unphysical v_th < reset branch.
        nodes[cell, len(chunks):] = nodes[cell, len(chunks) - 1]
    return nodes, weights


def lif_rate_gauss_legendre(mu_mv, sigma_mv, *, tau_mem_ms: float,
                            tau_ref_ms: float, v_threshold_mv,
                            v_reset_mv: float = 11.0,
                            quadrature_order: int = 16) -> np.ndarray:
    """Vectorized Siegert LIF transfer evaluated by Gauss-Legendre quadrature."""
    mu, sigma, threshold = np.broadcast_arrays(
        np.asarray(mu_mv, float), np.asarray(sigma_mv, float),
        np.asarray(v_threshold_mv, float))
    if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(sigma)):
        raise ValueError("mu and sigma must be finite")
    if np.any(sigma <= 0.0):
        raise ValueError("sigma must be positive")
    if np.any(threshold < float(v_reset_mv)):
        raise ValueError("threshold below reset is unphysical")
    x, weight = leggauss(int(quadrature_order))
    lower = (float(v_reset_mv) - mu) / sigma
    upper = (threshold - mu) / sigma
    sample = (0.5 * (lower + upper))[..., None] + (
        0.5 * (upper - lower))[..., None] * x
    integral = 0.5 * (upper - lower) * np.sum(weight * erfcx(-sample), axis=-1)
    denominator = float(tau_ref_ms) + float(tau_mem_ms) * np.sqrt(np.pi) * integral
    rate = np.divide(1.0, denominator,
                     out=np.zeros_like(denominator), where=np.isfinite(denominator))
    return np.clip(rate, 0.0, 1.0 / float(tau_ref_ms))


def _threshold_averaged_e_transfer(model: PatientCoarseZMModel, mu, sigma):
    rates = lif_rate_gauss_legendre(
        np.asarray(mu)[:, None], np.asarray(sigma)[:, None],
        tau_mem_ms=model.tau_mem_e_ms,
        tau_ref_ms=model.tau_ref_e_ms,
        v_threshold_mv=model.threshold_nodes_e,
        v_reset_mv=model.v_reset_mv,
    )
    return np.sum(model.threshold_weights_e * rates, axis=1)


def transfer_rates(model: PatientCoarseZMModel, mu_e, sigma_e, mu_i, sigma_i):
    """Threshold-averaged E and homogeneous-threshold I transfer."""
    phi_e = _threshold_averaged_e_transfer(model, mu_e, sigma_e)
    phi_i = lif_rate_gauss_legendre(
        mu_i, sigma_i, tau_mem_ms=model.tau_mem_i_ms,
        tau_ref_ms=model.tau_ref_i_ms,
        v_threshold_mv=model.v_threshold_i_mv,
        v_reset_mv=model.v_reset_mv,
    )
    return phi_e, phi_i


def moments(model: PatientCoarseZMModel, rate_e, rate_i, *, q: float,
            eta_m: float = 0.0, tau_m_slow_ms: float = 12.5):
    """Diffusion-approximation input moments for the matched coarse graph."""
    rate_e = np.asarray(rate_e, float)
    rate_i = np.asarray(rate_i, float)
    if rate_e.shape != (model.n_cells,) or rate_i.shape != (model.n_cells,):
        raise ValueError("rate vectors must have one value per coarse cell")
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    if eta_m < 0.0 or tau_m_slow_ms <= 0.0:
        raise ValueError("M parameters must be non-negative/positive")
    te, ti = model.tau_mem_e_ms, model.tau_mem_i_ms
    mu_e = te * (model.w_ee @ rate_e - float(q) * (model.w_ei @ rate_i)
                 + model.j_ext_e_mv * model.nu_ext_per_ms)
    mu_e -= float(eta_m) * float(tau_m_slow_ms) * rate_e
    mu_i = ti * (model.w_ie @ rate_e - model.w_ii @ rate_i
                 + model.j_ext_i_mv * model.nu_ext_per_ms)
    variance_e = te * (
        model.v_ee @ rate_e + float(q) ** 2 * (model.v_ei @ rate_i)
        + model.j_ext_e_mv ** 2 * model.nu_ext_per_ms)
    variance_i = ti * (
        model.v_ie @ rate_e + model.v_ii @ rate_i
        + model.j_ext_i_mv ** 2 * model.nu_ext_per_ms)
    sigma_e = np.sqrt(np.maximum(variance_e, 1e-12))
    sigma_i = np.sqrt(np.maximum(variance_i, 1e-12))
    return mu_e, sigma_e, mu_i, sigma_i


def fixed_point_residual(model: PatientCoarseZMModel, rates, *, q: float,
                         eta_m: float = 0.0,
                         tau_m_slow_ms: float = 12.5) -> np.ndarray:
    """Residual ``[rE-PhiE, rI-PhiI]`` after eliminating steady synapses/M."""
    rates = np.asarray(rates, float)
    if rates.shape != (2 * model.n_cells,):
        raise ValueError("rates must concatenate E then I coarse-cell rates")
    rate_e, rate_i = np.split(rates, 2)
    mu_e, sigma_e, mu_i, sigma_i = moments(
        model, rate_e, rate_i, q=q, eta_m=eta_m,
        tau_m_slow_ms=tau_m_slow_ms)
    phi_e, phi_i = transfer_rates(model, mu_e, sigma_e, mu_i, sigma_i)
    return np.r_[rate_e - phi_e, rate_i - phi_i]


def _transfer_derivatives(model, mu_e, sigma_e, mu_i, sigma_i, *, h=1e-3):
    e_plus, i_plus = transfer_rates(
        model, mu_e + h, sigma_e, mu_i + h, sigma_i)
    e_minus, i_minus = transfer_rates(
        model, mu_e - h, sigma_e, mu_i - h, sigma_i)
    dmu_e = (e_plus - e_minus) / (2.0 * h)
    dmu_i = (i_plus - i_minus) / (2.0 * h)
    e_plus_s, i_plus_s = transfer_rates(
        model, mu_e, sigma_e + h, mu_i, sigma_i + h)
    e_minus_s, i_minus_s = transfer_rates(
        model, mu_e, np.maximum(sigma_e - h, 1e-9),
        mu_i, np.maximum(sigma_i - h, 1e-9))
    dsig_e = (e_plus_s - e_minus_s) / (
        sigma_e + h - np.maximum(sigma_e - h, 1e-9))
    dsig_i = (i_plus_s - i_minus_s) / (
        sigma_i + h - np.maximum(sigma_i - h, 1e-9))
    return dmu_e, dsig_e, dmu_i, dsig_i


def fixed_point_jacobian(model: PatientCoarseZMModel, rates, *, q: float,
                         eta_m: float = 0.0,
                         tau_m_slow_ms: float = 12.5) -> np.ndarray:
    """Dense Jacobian of the fixed-point residual, including variance gain."""
    rates = np.asarray(rates, float)
    rate_e, rate_i = np.split(rates, 2)
    mu_e, sigma_e, mu_i, sigma_i = moments(
        model, rate_e, rate_i, q=q, eta_m=eta_m,
        tau_m_slow_ms=tau_m_slow_ms)
    pme, pse, pmi, psi = _transfer_derivatives(
        model, mu_e, sigma_e, mu_i, sigma_i)
    te, ti = model.tau_mem_e_ms, model.tau_mem_i_ms
    n = model.n_cells
    identity = np.eye(n)
    dmu_ee = te * model.w_ee - float(eta_m) * float(tau_m_slow_ms) * identity
    dmu_ei = -te * float(q) * model.w_ei
    dmu_ie = ti * model.w_ie
    dmu_ii = -ti * model.w_ii
    dsig_ee = te * model.v_ee / (2.0 * sigma_e[:, None])
    dsig_ei = te * float(q) ** 2 * model.v_ei / (2.0 * sigma_e[:, None])
    dsig_ie = ti * model.v_ie / (2.0 * sigma_i[:, None])
    dsig_ii = ti * model.v_ii / (2.0 * sigma_i[:, None])
    j_ee = identity - pme[:, None] * dmu_ee - pse[:, None] * dsig_ee
    j_ei = -pme[:, None] * dmu_ei - pse[:, None] * dsig_ei
    j_ie = -pmi[:, None] * dmu_ie - psi[:, None] * dsig_ie
    j_ii = identity - pmi[:, None] * dmu_ii - psi[:, None] * dsig_ii
    return np.block([[j_ee, j_ei], [j_ie, j_ii]])


def fixed_point_q_derivative(model: PatientCoarseZMModel, rates, *, q: float,
                             eta_m: float = 0.0,
                             tau_m_slow_ms: float = 12.5) -> np.ndarray:
    """Exact partial derivative ``dF/dq`` of the fixed-point residual."""
    rates = np.asarray(rates, float)
    rate_e, rate_i = np.split(rates, 2)
    mu_e, sigma_e, mu_i, sigma_i = moments(
        model, rate_e, rate_i, q=q, eta_m=eta_m,
        tau_m_slow_ms=tau_m_slow_ms)
    pme, pse, _pmi, _psi = _transfer_derivatives(
        model, mu_e, sigma_e, mu_i, sigma_i)
    dmu_e = -model.tau_mem_e_ms * (model.w_ei @ rate_i)
    dsigma_e = (
        model.tau_mem_e_ms * float(q) * (model.v_ei @ rate_i)
        / sigma_e)
    return np.r_[-pme * dmu_e - pse * dsigma_e,
                 np.zeros(model.n_cells, float)]


def solve_fixed_point(model: PatientCoarseZMModel, *, q: float,
                      initial_rates, eta_m: float = 0.0,
                      tau_m_slow_ms: float = 12.5,
                      residual_tolerance: float = 1e-9,
                      maxfev: int = 2000) -> FixedPointSolution:
    """Solve one fixed point without silently accepting a nonphysical root.

    The nonlinear solver is intentionally unrestricted because a bounded
    least-squares minimum is not necessarily a root.  Acceptance is separate:
    every rate must lie between zero and its refractory ceiling and the
    infinity-norm residual must pass the declared tolerance.
    """
    initial = np.asarray(initial_rates, float)
    if initial.shape != (2 * model.n_cells,):
        raise ValueError("initial_rates must concatenate E then I rates")
    arguments = {
        "model": model, "q": float(q), "eta_m": float(eta_m),
        "tau_m_slow_ms": float(tau_m_slow_ms),
    }
    answer = root(
        lambda values: fixed_point_residual(rates=values, **arguments),
        initial,
        jac=lambda values: fixed_point_jacobian(rates=values, **arguments),
        method="hybr", options={"xtol": 1e-10, "maxfev": int(maxfev)},
    )
    rates = np.asarray(answer.x, float)
    residual_inf = float(np.linalg.norm(
        fixed_point_residual(rates=rates, **arguments), ord=np.inf))
    ceiling_e = 1.0 / model.tau_ref_e_ms
    ceiling_i = 1.0 / model.tau_ref_i_ms
    physical = bool(
        np.all(np.isfinite(rates))
        and np.all(rates[:model.n_cells] >= -1e-10)
        and np.all(rates[model.n_cells:] >= -1e-10)
        and np.all(rates[:model.n_cells] <= ceiling_e + 1e-10)
        and np.all(rates[model.n_cells:] <= ceiling_i + 1e-10))
    converged = bool(answer.success and physical
                     and residual_inf <= float(residual_tolerance))
    return FixedPointSolution(
        rates=rates, q=float(q), eta_m=float(eta_m),
        tau_m_slow_ms=float(tau_m_slow_ms), converged=converged,
        physical=physical, residual_inf=residual_inf,
        nfev=int(getattr(answer, "nfev", -1)),
        njev=int(getattr(answer, "njev", -1)), message=str(answer.message),
    )


def _arc_inner(rate_a, q_a, rate_b, q_b) -> float:
    """Grid-size-invariant inner product: rate RMS plus q coordinate."""
    rate_a = np.asarray(rate_a, float)
    rate_b = np.asarray(rate_b, float)
    return float(np.dot(rate_a, rate_b) / rate_a.size + float(q_a) * float(q_b))


def _normalise_arc_tangent(rate_part, q_part):
    norm = np.sqrt(_arc_inner(rate_part, q_part, rate_part, q_part))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("zero or non-finite arclength tangent")
    return np.asarray(rate_part, float) / norm, float(q_part) / norm


def pseudo_arclength_continue(
        model: PatientCoarseZMModel, first: FixedPointSolution,
        second: FixedPointSolution, *, eta_m: float = 0.0,
        tau_m_slow_ms: float = 12.5, step_size: float = 0.0025,
        n_steps: int = 20, residual_tolerance: float = 1e-9,
        max_corrector_iterations: int = 16) -> list[PseudoArclengthPoint]:
    """Predictor-corrector continuation that can pass a simple fold.

    ``first`` and ``second`` must be distinct, already converged roots on the
    same branch.  The tangent norm uses rate RMS, rather than the raw norm of
    all coarse cells, so changing grid resolution does not rescale ``q``.
    """
    if not first.converged or not second.converged:
        raise ValueError("arclength seeds must be converged physical roots")
    if first.rates.shape != second.rates.shape:
        raise ValueError("arclength seed shapes differ")
    if step_size <= 0.0 or n_steps < 0:
        raise ValueError("invalid arclength step count/size")
    tangent_x, tangent_q = _normalise_arc_tangent(
        second.rates - first.rates, second.q - first.q)
    points = [PseudoArclengthPoint(
        solution=second, tangent_rates=tangent_x,
        tangent_q=tangent_q, corrector_iterations=0, step_size=0.0)]
    current_x = np.array(second.rates, copy=True)
    current_q = float(second.q)
    n_variables = current_x.size
    for _index in range(int(n_steps)):
        predictor_x = current_x + float(step_size) * tangent_x
        predictor_q = current_q + float(step_size) * tangent_q
        candidate_x = np.array(predictor_x, copy=True)
        candidate_q = float(predictor_q)
        converged = False
        correction_count = 0
        for correction_count in range(1, int(max_corrector_iterations) + 1):
            residual = fixed_point_residual(
                model, candidate_x, q=candidate_q, eta_m=eta_m,
                tau_m_slow_ms=tau_m_slow_ms)
            arc_residual = _arc_inner(
                tangent_x, tangent_q,
                candidate_x - predictor_x, candidate_q - predictor_q)
            error = max(float(np.linalg.norm(residual, ord=np.inf)),
                        abs(float(arc_residual)))
            if error <= float(residual_tolerance):
                converged = True
                break
            jacobian = fixed_point_jacobian(
                model, candidate_x, q=candidate_q, eta_m=eta_m,
                tau_m_slow_ms=tau_m_slow_ms)
            q_column = fixed_point_q_derivative(
                model, candidate_x, q=candidate_q, eta_m=eta_m,
                tau_m_slow_ms=tau_m_slow_ms)
            augmented = np.empty((n_variables + 1, n_variables + 1), float)
            augmented[:-1, :-1] = jacobian
            augmented[:-1, -1] = q_column
            augmented[-1, :-1] = tangent_x / n_variables
            augmented[-1, -1] = tangent_q
            update = np.linalg.solve(
                augmented, -np.r_[residual, arc_residual])
            # A short backtracking line search prevents an otherwise valid
            # corrector from entering negative-rate overflow territory.
            accepted = False
            for exponent in range(9):
                fraction = 0.5 ** exponent
                trial_x = candidate_x + fraction * update[:-1]
                trial_q = candidate_q + fraction * update[-1]
                if (np.min(trial_x) < -0.05 or np.max(trial_x) > 1.1
                        or not -0.25 <= trial_q <= 1.25):
                    continue
                trial_residual = fixed_point_residual(
                    model, trial_x, q=trial_q, eta_m=eta_m,
                    tau_m_slow_ms=tau_m_slow_ms)
                trial_arc = _arc_inner(
                    tangent_x, tangent_q,
                    trial_x - predictor_x, trial_q - predictor_q)
                trial_error = max(
                    float(np.linalg.norm(trial_residual, ord=np.inf)),
                    abs(float(trial_arc)))
                if trial_error < error:
                    candidate_x, candidate_q = trial_x, float(trial_q)
                    accepted = True
                    break
            if not accepted:
                break
        residual_inf = float(np.linalg.norm(fixed_point_residual(
            model, candidate_x, q=candidate_q, eta_m=eta_m,
            tau_m_slow_ms=tau_m_slow_ms), ord=np.inf))
        ceiling_e = 1.0 / model.tau_ref_e_ms
        ceiling_i = 1.0 / model.tau_ref_i_ms
        physical = bool(
            np.all(candidate_x[:model.n_cells] >= -1e-10)
            and np.all(candidate_x[model.n_cells:] >= -1e-10)
            and np.all(candidate_x[:model.n_cells] <= ceiling_e + 1e-10)
            and np.all(candidate_x[model.n_cells:] <= ceiling_i + 1e-10))
        solution = FixedPointSolution(
            rates=np.asarray(candidate_x), q=float(candidate_q),
            eta_m=float(eta_m), tau_m_slow_ms=float(tau_m_slow_ms),
            converged=bool(converged and physical), physical=physical,
            residual_inf=residual_inf, nfev=int(correction_count), njev=-1,
            message=("pseudo-arclength corrector converged" if converged
                     else "pseudo-arclength corrector failed"),
        )
        if not solution.converged:
            points.append(PseudoArclengthPoint(
                solution=solution, tangent_rates=tangent_x,
                tangent_q=tangent_q, corrector_iterations=correction_count,
                step_size=float(step_size)))
            break
        jacobian = fixed_point_jacobian(
            model, candidate_x, q=candidate_q, eta_m=eta_m,
            tau_m_slow_ms=tau_m_slow_ms)
        q_column = fixed_point_q_derivative(
            model, candidate_x, q=candidate_q, eta_m=eta_m,
            tau_m_slow_ms=tau_m_slow_ms)
        augmented = np.empty((n_variables + 1, n_variables + 1), float)
        augmented[:-1, :-1] = jacobian
        augmented[:-1, -1] = q_column
        augmented[-1, :-1] = tangent_x / n_variables
        augmented[-1, -1] = tangent_q
        rhs = np.zeros(n_variables + 1, float)
        rhs[-1] = 1.0
        new_tangent = np.linalg.solve(augmented, rhs)
        next_x, next_q = _normalise_arc_tangent(
            new_tangent[:-1], new_tangent[-1])
        if _arc_inner(next_x, next_q, tangent_x, tangent_q) < 0.0:
            next_x, next_q = -next_x, -next_q
        tangent_x, tangent_q = next_x, next_q
        current_x, current_q = np.asarray(candidate_x), float(candidate_q)
        points.append(PseudoArclengthPoint(
            solution=solution, tangent_rates=tangent_x,
            tangent_q=tangent_q, corrector_iterations=correction_count,
            step_size=float(step_size)))
    return points


def dynamic_jacobian(model: PatientCoarseZMModel, rates, *, q: float,
                     eta_m: float = 0.0,
                     tau_m_slow_ms: float = 12.5):
    """Sparse zero-delay rate/synapse/M Jacobian with op variance frozen.

    This matches the established M3B stability convention.  It is informative
    for a saddle-node zero mode; delay-aware Hopf classification remains a
    separate requirement.
    """
    rates = np.asarray(rates, float)
    rate_e, rate_i = np.split(rates, 2)
    mu_e, sigma_e, mu_i, sigma_i = moments(
        model, rate_e, rate_i, q=q, eta_m=eta_m,
        tau_m_slow_ms=tau_m_slow_ms)
    pme, _pse, pmi, _psi = _transfer_derivatives(
        model, mu_e, sigma_e, mu_i, sigma_i)
    n = model.n_cells
    eye = sparse.eye(n, format="csr")
    zero = sparse.csr_matrix((n, n))
    de = sparse.diags(pme)
    di = sparse.diags(pmi)
    te, ti = model.tau_mem_e_ms, model.tau_mem_i_ms
    ta, tg = model.tau_ampa_ms, model.tau_gaba_ms
    include_m = float(eta_m) != 0.0
    n_blocks = 7 if include_m else 6
    blocks = [[zero for _ in range(n_blocks)] for _ in range(n_blocks)]
    blocks[0][0] = -eye / te
    blocks[0][2] = de / te
    blocks[0][3] = -float(q) * de / te
    blocks[1][1] = -eye / ti
    blocks[1][4] = di / ti
    blocks[1][5] = -di / ti
    blocks[2][0] = sparse.csr_matrix(te * model.w_ee) / ta
    blocks[2][2] = -eye / ta
    blocks[3][1] = sparse.csr_matrix(te * model.w_ei) / tg
    blocks[3][3] = -eye / tg
    blocks[4][0] = sparse.csr_matrix(ti * model.w_ie) / ta
    blocks[4][4] = -eye / ta
    blocks[5][1] = sparse.csr_matrix(ti * model.w_ii) / tg
    blocks[5][5] = -eye / tg
    if include_m:
        blocks[0][6] = -float(eta_m) * de / te
        blocks[6][0] = eye
        blocks[6][6] = -eye / float(tau_m_slow_ms)
    return sparse.bmat(blocks, format="csr")


def _aggregate_pathway(matrices, *, target_cells, source_cells,
                       target_mask, n_cells, target_counts,
                       physical_factor):
    total = np.zeros((n_cells, n_cells), float)
    squared = np.zeros_like(total)
    for matrix in matrices:
        if matrix.nnz == 0:
            continue
        coo = matrix.tocoo(copy=False)
        row = np.asarray(coo.row, np.int64)
        selected = np.asarray(target_mask(row), bool)
        if not np.any(selected):
            continue
        target = target_cells[row[selected]]
        source = source_cells[np.asarray(coo.col, np.int64)[selected]]
        weight = np.asarray(coo.data[selected], float) * float(physical_factor)
        flat = target * int(n_cells) + source
        total += np.bincount(flat, weights=weight,
                             minlength=n_cells * n_cells).reshape(n_cells, n_cells)
        squared += np.bincount(flat, weights=weight ** 2,
                               minlength=n_cells * n_cells).reshape(n_cells, n_cells)
    denominator = np.asarray(target_counts, float)[:, None]
    return total / denominator, squared / denominator


def build_patient_coarse_model(substrate, *, n_grid: int = 20,
                               threshold_groups: int = 8):
    """Coarse-grain a frozen ``topic4_zm_ictal_transition.Substrate``."""
    params = substrate.params
    n_e, n_i = int(substrate.n_e), int(substrate.n_i)
    n_cells = int(n_grid) ** 2
    cell_e = spatial_cell_index(
        substrate.positions_e, n_grid=n_grid, sheet_l_mm=params.L)
    cell_i = spatial_cell_index(
        substrate.positions_i, n_grid=n_grid, sheet_l_mm=params.L)
    all_cells = np.r_[cell_e, cell_i]
    count_e = np.bincount(cell_e, minlength=n_cells)
    count_i = np.bincount(cell_i, minlength=n_cells)
    if np.any(count_e == 0) or np.any(count_i == 0):
        raise ValueError(
            "coarse grid contains empty population cells; reduce n_grid")
    threshold_nodes, threshold_weights = grouped_threshold_support(
        np.asarray(substrate.vtheta[:n_e], float), cell_e,
        n_groups=threshold_groups, n_cells=n_cells)
    ampa = substrate.net["ampa_by_delay"]
    gaba = substrate.net["gaba_by_delay"]
    w_ee, v_ee = _aggregate_pathway(
        ampa, target_cells=all_cells, source_cells=cell_e,
        target_mask=lambda rows: rows < n_e, n_cells=n_cells,
        target_counts=count_e,
        physical_factor=params.tau_r_AMPA / params.tau_m_E)
    w_ie, v_ie = _aggregate_pathway(
        ampa, target_cells=all_cells, source_cells=cell_e,
        target_mask=lambda rows: rows >= n_e, n_cells=n_cells,
        target_counts=count_i,
        physical_factor=params.tau_r_AMPA / params.tau_m_I)
    w_ei, v_ei = _aggregate_pathway(
        gaba, target_cells=all_cells, source_cells=cell_i,
        target_mask=lambda rows: rows < n_e, n_cells=n_cells,
        target_counts=count_e,
        physical_factor=params.tau_r_GABA / params.tau_m_E)
    w_ii, v_ii = _aggregate_pathway(
        gaba, target_cells=all_cells, source_cells=cell_i,
        target_mask=lambda rows: rows >= n_e, n_cells=n_cells,
        target_counts=count_i,
        physical_factor=params.tau_r_GABA / params.tau_m_I)
    from src.snn_engine.params import compute_nu_theta
    nu_ext = float(params.nu_ext_ratio * compute_nu_theta(params)[0])
    model = PatientCoarseZMModel(
        n_grid=int(n_grid), sheet_l_mm=float(params.L),
        count_e=count_e, count_i=count_i,
        threshold_nodes_e=threshold_nodes,
        threshold_weights_e=threshold_weights,
        w_ee=w_ee, w_ei=w_ei, w_ie=w_ie, w_ii=w_ii,
        v_ee=v_ee, v_ei=v_ei, v_ie=v_ie, v_ii=v_ii,
        tau_mem_e_ms=float(params.tau_m_E),
        tau_mem_i_ms=float(params.tau_m_I),
        tau_ref_e_ms=float(params.tau_ref_E),
        tau_ref_i_ms=float(params.tau_ref_I),
        tau_ampa_ms=float(params.tau_d_AMPA),
        tau_gaba_ms=float(params.tau_d_GABA),
        v_reset_mv=float(params.V_reset),
        v_threshold_i_mv=float(params.V_th),
        j_ext_e_mv=float(params.J_ext_E),
        j_ext_i_mv=float(params.J_ext_I),
        nu_ext_per_ms=nu_ext,
    )
    model.validate()
    return model


def save_patient_coarse_model(path, model: PatientCoarseZMModel) -> dict:
    """Atomically save a numeric-only NPZ and return its identity record."""
    model.validate()
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for item in fields(PatientCoarseZMModel):
        value = getattr(model, item.name)
        payload[item.name] = np.asarray(value)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem + ".", suffix=".npz")
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def load_patient_coarse_model(path) -> PatientCoarseZMModel:
    """Load the numeric-only deterministic bridge without pickle."""
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as archive:
        missing = [item.name for item in fields(PatientCoarseZMModel)
                   if item.name not in archive.files]
        if missing:
            raise ValueError("coarse model archive is incomplete: " + ", ".join(missing))
        values = {}
        array_names = {
            "count_e", "count_i", "threshold_nodes_e", "threshold_weights_e",
            "w_ee", "w_ei", "w_ie", "w_ii", "v_ee", "v_ei", "v_ie", "v_ii",
        }
        integer_names = {"n_grid"}
        for item in fields(PatientCoarseZMModel):
            raw = np.asarray(archive[item.name])
            if item.name in array_names:
                values[item.name] = raw
            elif item.name in integer_names:
                values[item.name] = int(raw)
            else:
                values[item.name] = float(raw)
    model = PatientCoarseZMModel(**values)
    model.validate()
    return model


def homogeneous_one_cell_model(*, ratio: float = 1.0):
    """One-cell contract fixture matching ``sef_hfo_lif._ms`` exactly."""
    from src.sef_hfo_lif import (
        C_EE, C_EI, C_IE, C_II, JX_E, JX_I, TAU_AMPA, TAU_GABA,
        TAU_ME, TAU_MI, TREF_E, TREF_I, V_RESET, V_TH,
        W_EE, W_EI, W_IE, W_II, nu_theta_pop,
    )
    scalar = lambda value: np.asarray([[float(value)]])
    return PatientCoarseZMModel(
        n_grid=1, sheet_l_mm=1.0,
        count_e=np.asarray([1]), count_i=np.asarray([1]),
        threshold_nodes_e=np.asarray([[V_TH]], float),
        threshold_weights_e=np.asarray([[1.0]], float),
        w_ee=scalar(C_EE * W_EE), w_ei=scalar(C_EI * W_EI),
        w_ie=scalar(C_IE * W_IE), w_ii=scalar(C_II * W_II),
        v_ee=scalar(C_EE * W_EE ** 2), v_ei=scalar(C_EI * W_EI ** 2),
        v_ie=scalar(C_IE * W_IE ** 2), v_ii=scalar(C_II * W_II ** 2),
        tau_mem_e_ms=TAU_ME, tau_mem_i_ms=TAU_MI,
        tau_ref_e_ms=TREF_E, tau_ref_i_ms=TREF_I,
        tau_ampa_ms=TAU_AMPA, tau_gaba_ms=TAU_GABA,
        v_reset_mv=V_RESET, v_threshold_i_mv=V_TH,
        j_ext_e_mv=JX_E, j_ext_i_mv=JX_I,
        nu_ext_per_ms=float(ratio * nu_theta_pop()),
    )
