"""Frozen spatial-Z reduction for the rev21 data-driven dual-core substrate.

The rev21 SNN already has one ``z_i`` per excitatory neuron.  Earlier
continuation code replaced that field by one homogeneous scalar ``q``.  This
module keeps the same coarse-grained realized graph but projects inhibitory
efficacy onto three spatial regions: core A, core B and surround.

Rates are spikes/ms, time is ms and voltages are mV, matching
``topic4_patient_zm_meanfield``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import root

from src.topic4_patient_zm_meanfield import (
    PatientCoarseZMModel,
    _transfer_derivatives,
    spatial_cell_index,
    transfer_rates,
)


@dataclass(frozen=True)
class DualCoreSpatialZMap:
    """Fraction of E cells assigned to each frozen core in every grid cell."""

    core_a_fraction_e: np.ndarray
    core_b_fraction_e: np.ndarray
    centers_mm: np.ndarray
    selected_count_per_core: np.ndarray

    def validate(self, model: PatientCoarseZMModel) -> None:
        n = model.n_cells
        for name in ("core_a_fraction_e", "core_b_fraction_e"):
            value = np.asarray(getattr(self, name), float)
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
            if np.any(~np.isfinite(value)) or np.any((value < 0) | (value > 1)):
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        if np.any(self.core_a_fraction_e + self.core_b_fraction_e > 1 + 1e-12):
            raise ValueError("dual-core fractions cannot exceed one")
        if np.asarray(self.centers_mm).shape != (2, 2):
            raise ValueError("centers_mm must have shape (2, 2)")
        if np.asarray(self.selected_count_per_core).shape != (2,):
            raise ValueError("selected_count_per_core must have shape (2,)")

    @property
    def surround_fraction_e(self) -> np.ndarray:
        return 1.0 - self.core_a_fraction_e - self.core_b_fraction_e

    def z_field(self, *, z_a: float, z_b: float, z_surround: float) -> np.ndarray:
        values = np.asarray([z_a, z_b, z_surround], float)
        if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("z_a, z_b and z_surround must lie in [0, 1]")
        return (
            self.core_a_fraction_e * float(z_a)
            + self.core_b_fraction_e * float(z_b)
            + self.surround_fraction_e * float(z_surround)
        )

    def z_second_moment_field(self, *, z_a: float, z_b: float,
                              z_surround: float) -> np.ndarray:
        """Return within-cell E[Z^2] for mixed core/surround coarse cells."""
        values = np.asarray([z_a, z_b, z_surround], float)
        if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("z_a, z_b and z_surround must lie in [0, 1]")
        return (
            self.core_a_fraction_e * float(z_a) ** 2
            + self.core_b_fraction_e * float(z_b) ** 2
            + self.surround_fraction_e * float(z_surround) ** 2
        )

    def depletion_profile(self, *, core_a_weight: float = 1.0,
                          core_b_weight: float = 1.0,
                          surround_weight: float = 0.7) -> np.ndarray:
        weights = np.asarray(
            [core_a_weight, core_b_weight, surround_weight], float,
        )
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("depletion weights must be finite and non-negative")
        return (
            self.core_a_fraction_e * weights[0]
            + self.core_b_fraction_e * weights[1]
            + self.surround_fraction_e * weights[2]
        )


@dataclass(frozen=True)
class SpatialZFixedPoint:
    rates: np.ndarray
    parameter: float
    z_a: float
    z_b: float
    z_surround: float
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
        return self.rates[: self.rates.size // 2]

    @property
    def rate_i(self) -> np.ndarray:
        return self.rates[self.rates.size // 2 :]

    @property
    def mean_rate_e_hz(self) -> float:
        return 1000.0 * float(np.mean(self.rate_e))


@dataclass(frozen=True)
class SpatialZArcPoint:
    solution: SpatialZFixedPoint
    tangent_rates: np.ndarray
    tangent_parameter: float
    corrector_iterations: int
    step_size: float


def build_dual_core_z_map(
    model: PatientCoarseZMModel,
    positions_e,
    h_e,
    centers_mm,
) -> DualCoreSpatialZMap:
    """Project the exact binary dual-core membership onto the coarse grid."""
    positions = np.asarray(positions_e, float)
    h = np.asarray(h_e, float)
    centers = np.asarray(centers_mm, float)
    if positions.shape != (h.size, 2):
        raise ValueError("positions_e and h_e must align")
    if centers.shape != (2, 2):
        raise ValueError("centers_mm must have shape (2, 2)")
    selected = h >= 0.5
    distance = np.linalg.norm(
        positions[:, None, :] - centers[None, :, :], axis=2,
    )
    nearest = np.argmin(distance, axis=1)
    cell = spatial_cell_index(
        positions, n_grid=model.n_grid, sheet_l_mm=model.sheet_l_mm,
    )
    count = np.asarray(model.count_e, float)
    fractions = []
    selected_counts = []
    for core in (0, 1):
        mask = selected & (nearest == core)
        selected_counts.append(int(np.sum(mask)))
        fractions.append(
            np.bincount(cell[mask], minlength=model.n_cells) / count
        )
    result = DualCoreSpatialZMap(
        core_a_fraction_e=np.asarray(fractions[0], float),
        core_b_fraction_e=np.asarray(fractions[1], float),
        centers_mm=centers,
        selected_count_per_core=np.asarray(selected_counts, int),
    )
    result.validate(model)
    if int(np.sum(result.selected_count_per_core)) != int(np.sum(selected)):
        raise RuntimeError("coarse dual-core projection lost selected neurons")
    return result


def spatial_z_moments(model: PatientCoarseZMModel, rate_e, rate_i, *, z_field,
                      z_second_moment=None,
                      eta_m: float = 0.0, tau_m_slow_ms: float = 500.0):
    rate_e = np.asarray(rate_e, float)
    rate_i = np.asarray(rate_i, float)
    z = np.asarray(z_field, float)
    z2 = z ** 2 if z_second_moment is None else np.asarray(
        z_second_moment, float)
    n = model.n_cells
    if (rate_e.shape != (n,) or rate_i.shape != (n,)
            or z.shape != (n,) or z2.shape != (n,)):
        raise ValueError("rates and Z moments must have one value per coarse cell")
    if np.any(~np.isfinite(z)) or np.any((z < 0) | (z > 1)):
        raise ValueError("z_field must be finite and lie in [0, 1]")
    if (np.any(~np.isfinite(z2)) or np.any(z2 < z ** 2 - 1e-12)
            or np.any(z2 > 1 + 1e-12)):
        raise ValueError("z_second_moment must be physical and >= z_field^2")
    if eta_m < 0 or tau_m_slow_ms <= 0:
        raise ValueError("M parameters must be non-negative/positive")
    te, ti = model.tau_mem_e_ms, model.tau_mem_i_ms
    inhibitory_mean = model.w_ei @ rate_i
    inhibitory_variance = model.v_ei @ rate_i
    mu_e = te * (
        model.w_ee @ rate_e - z * inhibitory_mean
        + model.j_ext_e_mv * model.nu_ext_per_ms
    )
    mu_e -= float(eta_m) * float(tau_m_slow_ms) * rate_e
    mu_i = ti * (
        model.w_ie @ rate_e - model.w_ii @ rate_i
        + model.j_ext_i_mv * model.nu_ext_per_ms
    )
    variance_e = te * (
        model.v_ee @ rate_e + z2 * inhibitory_variance
        + model.j_ext_e_mv ** 2 * model.nu_ext_per_ms
    )
    variance_i = ti * (
        model.v_ie @ rate_e + model.v_ii @ rate_i
        + model.j_ext_i_mv ** 2 * model.nu_ext_per_ms
    )
    return (
        mu_e,
        np.sqrt(np.maximum(variance_e, 1e-12)),
        mu_i,
        np.sqrt(np.maximum(variance_i, 1e-12)),
    )


def spatial_z_residual(model: PatientCoarseZMModel, rates, *, z_field,
                       z_second_moment=None,
                       eta_m: float = 0.0,
                       tau_m_slow_ms: float = 500.0) -> np.ndarray:
    rates = np.asarray(rates, float)
    if rates.shape != (2 * model.n_cells,):
        raise ValueError("rates must concatenate E then I coarse-cell rates")
    rate_e, rate_i = np.split(rates, 2)
    mu_e, sigma_e, mu_i, sigma_i = spatial_z_moments(
        model, rate_e, rate_i, z_field=z_field,
        z_second_moment=z_second_moment, eta_m=eta_m,
        tau_m_slow_ms=tau_m_slow_ms,
    )
    phi_e, phi_i = transfer_rates(model, mu_e, sigma_e, mu_i, sigma_i)
    return np.r_[rate_e - phi_e, rate_i - phi_i]


def spatial_z_jacobian(model: PatientCoarseZMModel, rates, *, z_field,
                       z_second_moment=None,
                       eta_m: float = 0.0,
                       tau_m_slow_ms: float = 500.0) -> np.ndarray:
    rates = np.asarray(rates, float)
    rate_e, rate_i = np.split(rates, 2)
    z = np.asarray(z_field, float)
    z2 = z ** 2 if z_second_moment is None else np.asarray(
        z_second_moment, float)
    mu_e, sigma_e, mu_i, sigma_i = spatial_z_moments(
        model, rate_e, rate_i, z_field=z, z_second_moment=z2, eta_m=eta_m,
        tau_m_slow_ms=tau_m_slow_ms,
    )
    pme, pse, pmi, psi = _transfer_derivatives(
        model, mu_e, sigma_e, mu_i, sigma_i,
    )
    te, ti = model.tau_mem_e_ms, model.tau_mem_i_ms
    n = model.n_cells
    identity = np.eye(n)
    dmu_ee = te * model.w_ee - float(eta_m) * float(tau_m_slow_ms) * identity
    dmu_ei = -te * z[:, None] * model.w_ei
    dmu_ie = ti * model.w_ie
    dmu_ii = -ti * model.w_ii
    dsig_ee = te * model.v_ee / (2.0 * sigma_e[:, None])
    dsig_ei = te * z2[:, None] * model.v_ei / (2.0 * sigma_e[:, None])
    dsig_ie = ti * model.v_ie / (2.0 * sigma_i[:, None])
    dsig_ii = ti * model.v_ii / (2.0 * sigma_i[:, None])
    return np.block([
        [identity - pme[:, None] * dmu_ee - pse[:, None] * dsig_ee,
         -pme[:, None] * dmu_ei - pse[:, None] * dsig_ei],
        [-pmi[:, None] * dmu_ie - psi[:, None] * dsig_ie,
         identity - pmi[:, None] * dmu_ii - psi[:, None] * dsig_ii],
    ])


def path_state(z_map: DualCoreSpatialZMap, parameter: float, *,
               core_a_weight: float = 1.0, core_b_weight: float = 1.0,
               surround_weight: float = 0.7):
    """Return regional Z values and the cell-wise field for depletion ``s``."""
    s = float(parameter)
    weights = np.asarray(
        [core_a_weight, core_b_weight, surround_weight], float,
    )
    regional = 1.0 - s * weights
    if np.any((regional < 0) | (regional > 1)):
        raise ValueError("path parameter puts a regional Z outside [0, 1]")
    z = 1.0 - s * z_map.depletion_profile(
        core_a_weight=weights[0], core_b_weight=weights[1],
        surround_weight=weights[2],
    )
    return float(regional[0]), float(regional[1]), float(regional[2]), z


def path_parameter_derivative(model: PatientCoarseZMModel, rates, *, z_field,
                              depletion_profile, z_second_moment=None,
                              z_second_moment_derivative=None,
                              eta_m: float = 0.0,
                              tau_m_slow_ms: float = 500.0) -> np.ndarray:
    """Exact partial derivative dF/ds for ``Z(x)=1-s*profile(x)``."""
    rates = np.asarray(rates, float)
    rate_e, rate_i = np.split(rates, 2)
    z = np.asarray(z_field, float)
    profile = np.asarray(depletion_profile, float)
    z2 = z ** 2 if z_second_moment is None else np.asarray(
        z_second_moment, float)
    mu_e, sigma_e, mu_i, sigma_i = spatial_z_moments(
        model, rate_e, rate_i, z_field=z, z_second_moment=z2,
        eta_m=eta_m,
        tau_m_slow_ms=tau_m_slow_ms,
    )
    pme, pse, _pmi, _psi = _transfer_derivatives(
        model, mu_e, sigma_e, mu_i, sigma_i,
    )
    inhibitory_mean = model.w_ei @ rate_i
    inhibitory_variance = model.v_ei @ rate_i
    dmu_ds = model.tau_mem_e_ms * profile * inhibitory_mean
    if z_second_moment_derivative is None:
        dz2_ds = -2.0 * z * profile
    else:
        dz2_ds = np.asarray(z_second_moment_derivative, float)
    dvar_ds = model.tau_mem_e_ms * dz2_ds * inhibitory_variance
    dsigma_ds = dvar_ds / (2.0 * sigma_e)
    return np.r_[-pme * dmu_ds - pse * dsigma_ds,
                 np.zeros(model.n_cells, float)]


def solve_spatial_z_fixed_point(
    model: PatientCoarseZMModel,
    z_map: DualCoreSpatialZMap,
    *,
    parameter: float,
    initial_rates,
    core_a_weight: float = 1.0,
    core_b_weight: float = 1.0,
    surround_weight: float = 0.7,
    eta_m: float = 0.0,
    tau_m_slow_ms: float = 500.0,
    maxfev: int = 5000,
) -> SpatialZFixedPoint:
    z_a, z_b, z_surround, z = path_state(
        z_map, parameter, core_a_weight=core_a_weight,
        core_b_weight=core_b_weight, surround_weight=surround_weight,
    )
    return solve_regional_z_fixed_point(
        model, z_map, z_a=z_a, z_b=z_b, z_surround=z_surround,
        initial_rates=initial_rates, parameter=float(parameter),
        eta_m=eta_m, tau_m_slow_ms=tau_m_slow_ms, maxfev=maxfev,
    )


def solve_regional_z_fixed_point(
    model: PatientCoarseZMModel,
    z_map: DualCoreSpatialZMap,
    *,
    z_a: float,
    z_b: float,
    z_surround: float,
    initial_rates,
    parameter: float = np.nan,
    eta_m: float = 0.0,
    tau_m_slow_ms: float = 500.0,
    maxfev: int = 5000,
) -> SpatialZFixedPoint:
    """Solve one fixed point for independently specified regional Z values."""
    z_map.validate(model)
    z = z_map.z_field(z_a=z_a, z_b=z_b, z_surround=z_surround)
    z2 = z_map.z_second_moment_field(
        z_a=z_a, z_b=z_b, z_surround=z_surround)
    initial = np.asarray(initial_rates, float)
    if initial.shape != (2 * model.n_cells,):
        raise ValueError("initial_rates must concatenate E then I rates")
    solved = root(
        lambda value: spatial_z_residual(
            model, value, z_field=z, z_second_moment=z2, eta_m=eta_m,
            tau_m_slow_ms=tau_m_slow_ms,
        ),
        initial,
        jac=lambda value: spatial_z_jacobian(
            model, value, z_field=z, z_second_moment=z2, eta_m=eta_m,
            tau_m_slow_ms=tau_m_slow_ms,
        ),
        method="hybr",
        options={"maxfev": int(maxfev), "xtol": 1e-10},
    )
    residual = spatial_z_residual(
        model, solved.x, z_field=z, z_second_moment=z2, eta_m=eta_m,
        tau_m_slow_ms=tau_m_slow_ms,
    )
    upper = np.r_[
        np.full(model.n_cells, 1.0 / model.tau_ref_e_ms),
        np.full(model.n_cells, 1.0 / model.tau_ref_i_ms),
    ]
    physical = bool(
        np.all(np.isfinite(solved.x))
        and np.all(solved.x >= -1e-10)
        and np.all(solved.x <= upper + 1e-10)
    )
    return SpatialZFixedPoint(
        rates=np.asarray(solved.x, float), parameter=float(parameter),
        z_a=float(z_a), z_b=float(z_b), z_surround=float(z_surround),
        eta_m=float(eta_m), tau_m_slow_ms=float(tau_m_slow_ms),
        converged=bool(solved.success and np.max(np.abs(residual)) < 1e-8),
        physical=physical, residual_inf=float(np.max(np.abs(residual))),
        nfev=int(getattr(solved, "nfev", -1)),
        njev=int(getattr(solved, "njev", -1)), message=str(solved.message),
    )


def _arc_inner(x1, s1, x2, s2):
    return float(np.dot(x1, x2) / x1.size + s1 * s2)


def _normalise_tangent(x, s):
    norm = np.sqrt(float(np.dot(x, x) / x.size + s * s))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("invalid arclength tangent")
    return np.asarray(x, float) / norm, float(s) / norm


def pseudo_arclength_spatial_z(
    model: PatientCoarseZMModel,
    z_map: DualCoreSpatialZMap,
    first: SpatialZFixedPoint,
    second: SpatialZFixedPoint,
    *,
    core_a_weight: float = 1.0,
    core_b_weight: float = 1.0,
    surround_weight: float = 0.7,
    eta_m: float = 0.0,
    tau_m_slow_ms: float = 500.0,
    step_size: float = 0.005,
    n_steps: int = 120,
    max_corrector_iterations: int = 20,
) -> list[SpatialZArcPoint]:
    """Predictor-corrector continuation along a frozen spatial-Z path."""
    if not first.converged or not second.converged:
        raise ValueError("continuation seeds must be converged")
    tangent_x, tangent_s = _normalise_tangent(
        second.rates - first.rates, second.parameter - first.parameter,
    )
    points = [
        SpatialZArcPoint(first, tangent_x, tangent_s, 0, step_size),
        SpatialZArcPoint(second, tangent_x, tangent_s, 0, step_size),
    ]
    current_x = np.asarray(second.rates, float)
    current_s = float(second.parameter)
    nvar = current_x.size
    profile = z_map.depletion_profile(
        core_a_weight=core_a_weight, core_b_weight=core_b_weight,
        surround_weight=surround_weight,
    )
    fractions = (
        z_map.core_a_fraction_e,
        z_map.core_b_fraction_e,
        z_map.surround_fraction_e,
    )
    weights = (float(core_a_weight), float(core_b_weight),
               float(surround_weight))
    for _ in range(int(n_steps)):
        predicted_x = current_x + float(step_size) * tangent_x
        predicted_s = current_s + float(step_size) * tangent_s
        candidate_x = predicted_x.copy()
        candidate_s = float(predicted_s)
        converged = False
        correction_count = 0
        for correction_count in range(1, int(max_corrector_iterations) + 1):
            try:
                z_a, z_b, z_surround, z = path_state(
                    z_map, candidate_s, core_a_weight=core_a_weight,
                    core_b_weight=core_b_weight,
                    surround_weight=surround_weight,
                )
                z2 = z_map.z_second_moment_field(
                    z_a=z_a, z_b=z_b, z_surround=z_surround)
                dz2_ds = -2.0 * sum(
                    fraction * weight * regional
                    for fraction, weight, regional in zip(
                        fractions, weights, (z_a, z_b, z_surround)))
            except ValueError:
                break
            residual = spatial_z_residual(
                model, candidate_x, z_field=z, z_second_moment=z2,
                eta_m=eta_m,
                tau_m_slow_ms=tau_m_slow_ms,
            )
            arc_residual = _arc_inner(
                candidate_x - predicted_x, candidate_s - predicted_s,
                tangent_x, tangent_s,
            )
            total = np.r_[residual, arc_residual]
            if np.max(np.abs(total)) < 1e-9:
                converged = True
                break
            jac = spatial_z_jacobian(
                model, candidate_x, z_field=z, z_second_moment=z2,
                eta_m=eta_m,
                tau_m_slow_ms=tau_m_slow_ms,
            )
            column = path_parameter_derivative(
                model, candidate_x, z_field=z, depletion_profile=profile,
                z_second_moment=z2,
                z_second_moment_derivative=dz2_ds,
                eta_m=eta_m, tau_m_slow_ms=tau_m_slow_ms,
            )
            augmented = np.empty((nvar + 1, nvar + 1), float)
            augmented[:-1, :-1] = jac
            augmented[:-1, -1] = column
            augmented[-1, :-1] = tangent_x / nvar
            augmented[-1, -1] = tangent_s
            try:
                delta = np.linalg.solve(augmented, -total)
            except np.linalg.LinAlgError:
                break
            candidate_x += delta[:-1]
            candidate_s += float(delta[-1])
        if not converged:
            break
        upper = np.r_[
            np.full(model.n_cells, 1.0 / model.tau_ref_e_ms),
            np.full(model.n_cells, 1.0 / model.tau_ref_i_ms),
        ]
        solution = SpatialZFixedPoint(
            rates=np.asarray(candidate_x, float), parameter=float(candidate_s),
            z_a=float(z_a), z_b=float(z_b), z_surround=float(z_surround),
            eta_m=float(eta_m), tau_m_slow_ms=float(tau_m_slow_ms),
            converged=True,
            physical=bool(
                np.all(np.isfinite(candidate_x))
                and np.all(candidate_x >= -1e-10)
                and np.all(candidate_x <= upper + 1e-10)
            ),
            residual_inf=float(np.max(np.abs(residual))),
            nfev=correction_count, njev=correction_count,
            message="pseudo-arclength corrector converged",
        )
        jac = spatial_z_jacobian(
            model, candidate_x, z_field=z, z_second_moment=z2,
            eta_m=eta_m,
            tau_m_slow_ms=tau_m_slow_ms,
        )
        column = path_parameter_derivative(
            model, candidate_x, z_field=z, depletion_profile=profile,
            z_second_moment=z2,
            z_second_moment_derivative=dz2_ds,
            eta_m=eta_m, tau_m_slow_ms=tau_m_slow_ms,
        )
        augmented = np.empty((nvar + 1, nvar + 1), float)
        augmented[:-1, :-1] = jac
        augmented[:-1, -1] = column
        augmented[-1, :-1] = tangent_x / nvar
        augmented[-1, -1] = tangent_s
        rhs = np.zeros(nvar + 1, float)
        rhs[-1] = 1.0
        new_tangent = np.linalg.solve(augmented, rhs)
        next_x, next_s = _normalise_tangent(new_tangent[:-1], new_tangent[-1])
        if _arc_inner(next_x, next_s, tangent_x, tangent_s) < 0:
            next_x, next_s = -next_x, -next_s
        tangent_x, tangent_s = next_x, next_s
        current_x, current_s = candidate_x, float(candidate_s)
        points.append(SpatialZArcPoint(
            solution, tangent_x, tangent_s, correction_count, step_size,
        ))
    return points


def closest_zero_eigenvalue(model: PatientCoarseZMModel,
                            solution: SpatialZFixedPoint,
                            z_map: DualCoreSpatialZMap,
                            *, core_a_weight: float = 1.0,
                            core_b_weight: float = 1.0,
                            surround_weight: float = 0.7):
    _za, _zb, _zs, z = path_state(
        z_map, solution.parameter, core_a_weight=core_a_weight,
        core_b_weight=core_b_weight, surround_weight=surround_weight,
    )
    z2 = z_map.z_second_moment_field(
        z_a=_za, z_b=_zb, z_surround=_zs)
    jac = spatial_z_jacobian(
        model, solution.rates, z_field=z, z_second_moment=z2,
        eta_m=solution.eta_m,
        tau_m_slow_ms=solution.tau_m_slow_ms,
    )
    values = np.linalg.eigvals(jac)
    return values[int(np.argmin(np.abs(values)))]


def spatial_z_dynamic_jacobian(model: PatientCoarseZMModel, rates, *, z_field,
                               z_second_moment=None, eta_m: float = 0.0,
                               tau_m_slow_ms: float = 500.0):
    """Zero-delay rate/synapse/M Jacobian with operating variance frozen.

    This is a labelled stability sensitivity, matching the established
    mean-field convention.  It is not a delay-aware stability theorem.
    """
    rates = np.asarray(rates, float)
    rate_e, rate_i = np.split(rates, 2)
    z = np.asarray(z_field, float)
    z2 = z ** 2 if z_second_moment is None else np.asarray(
        z_second_moment, float)
    mu_e, sigma_e, mu_i, sigma_i = spatial_z_moments(
        model, rate_e, rate_i, z_field=z, z_second_moment=z2,
        eta_m=eta_m, tau_m_slow_ms=tau_m_slow_ms)
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
    blocks[0][3] = -sparse.diags(z) @ de / te
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


def regional_rates_hz(model: PatientCoarseZMModel, z_map: DualCoreSpatialZMap,
                      rate_e) -> dict[str, float]:
    rates = np.asarray(rate_e, float)
    counts = np.asarray(model.count_e, float)
    weights = {
        "core_a": counts * z_map.core_a_fraction_e,
        "core_b": counts * z_map.core_b_fraction_e,
        "surround": counts * z_map.surround_fraction_e,
    }
    return {
        name: 1000.0 * float(np.average(rates, weights=value))
        for name, value in weights.items() if np.sum(value) > 0
    }
