"""Regional entry-fold and frozen-current exit tools for the P3 MZ scaffold.

This module keeps the locked M3B spatial operators and the current-based
Stage-0C fast equations.  It changes neither E-to-E weights nor kernels.  The
only control coordinates are a shared core/annulus inhibitory resource
``z_regional`` and a shared core/annulus additive E-cell current.  The far bath
stays at its registered interictal resource and receives no additive current.

The equilibrium manifold has ``2*P`` rate coordinates.  Synapses, the fast
sensor, and the single area-weighted global pool are reconstructed exactly from
those rates.  Frozen ``z/p/m`` coordinates are excluded from the fast Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares

from src.sef_hfo_lif import TAU_ME, TAU_MI
from src.topic4_mz_spatial_patch import (
    PatchParameters,
    PreparedPatchRHS,
    pack_patch_state,
    patch_rhs_fast,
    patch_rhs_fast_and_moments,
    state_size,
)
from src.topic4_spatial_slowfast_stage0c import S_MAX, recruitment_sensor


@dataclass(frozen=True)
class RegionalFoldPoint:
    """One augmented saddle-node solution on the regional-Z coordinate."""

    z_regional: float
    additive_mv: float
    rates_khz: np.ndarray
    null_vector: np.ndarray
    residual_inf: float
    rate_sigma_min: float
    fast_leading_real_per_ms: float
    fast_leading_imag_per_ms: float
    left_fz: float
    left_d2f_vv: float
    support_all: bool
    critical_fast_mode: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        n_patches = int(self.rates_khz.size // 2)
        return {
            "z_regional": float(self.z_regional),
            "additive_mv": float(self.additive_mv),
            "rE_hz": (1000.0 * self.rates_khz[:n_patches]).tolist(),
            "rI_hz": (1000.0 * self.rates_khz[n_patches:]).tolist(),
            "augmented_residual_inf": float(self.residual_inf),
            "rate_jacobian_sigma_min": float(self.rate_sigma_min),
            "fast_leading_real_per_ms": float(self.fast_leading_real_per_ms),
            "fast_leading_imag_per_ms": float(self.fast_leading_imag_per_ms),
            "left_Fz": float(self.left_fz),
            "left_D2F_vv": float(self.left_d2f_vv),
            "support_all": bool(self.support_all),
            "rate_null_vector": self.null_vector.tolist(),
            "critical_fast_mode": self.critical_fast_mode.tolist(),
        }


def regional_equilibrium_state(
    rates_khz: Sequence[float],
    prepared: PreparedPatchRHS,
    parameters: PatchParameters,
    *,
    z_regional: float,
    z_bath: float = 0.90,
    additive_mv: float = 0.0,
) -> np.ndarray:
    """Lift regional E/I rates to the exact frozen equilibrium manifold."""

    rates = np.asarray(rates_khz, dtype=float)
    n_patches = prepared.n_patches
    if n_patches != 3:
        raise ValueError("regional entry/exit oracle is registered for core/annulus/bath P=3")
    if rates.shape != (2 * n_patches,) or np.any(rates <= 0.0) or not np.all(np.isfinite(rates)):
        raise ValueError("rates must be finite positive [E(P), I(P)] coordinates")
    params = parameters.validate()
    if not 0.0 < z_regional <= 1.0 or not 0.0 < z_bath <= 1.0:
        raise ValueError("regional and bath z must lie in (0,1]")
    if not 0.0 <= additive_mv <= params.additive_max_mv:
        raise ValueError("regional additive current must lie in [0, additive_max_mv]")
    e = rates[:n_patches]
    i = rates[n_patches:]
    z = np.asarray([z_regional, z_regional, z_bath], dtype=float)
    m = np.asarray(
        [additive_mv / params.additive_max_mv, additive_mv / params.additive_max_mv, 0.0],
        dtype=float,
    ) if params.additive_max_mv > 0.0 else np.zeros(n_patches, dtype=float)
    mu_g = float(prepared.patch_weights @ recruitment_sensor(e))
    local = {
        "rE": e,
        "rI": i,
        "sEE": prepared.K_EE @ e,
        "sEI": prepared.K_I @ i,
        "sIE": prepared.K_I @ e,
        "sII": prepared.K_I @ i,
        "rE_fast": e,
        "z": z,
        "p": np.zeros(n_patches, dtype=float),
        "m": m,
    }
    return pack_patch_state(local, mu_g=mu_g, s_g=S_MAX * mu_g)


def regional_equilibrium_residual(
    rates_khz: Sequence[float],
    prepared: PreparedPatchRHS,
    parameters: PatchParameters,
    transfer: Any,
    *,
    z_regional: float,
    z_bath: float = 0.90,
    additive_mv: float = 0.0,
) -> np.ndarray:
    """Return ``[rE-PhiE, rI-PhiI]`` in kHz on the equilibrium manifold."""

    try:
        state = regional_equilibrium_state(
            rates_khz, prepared, parameters,
            z_regional=z_regional, z_bath=z_bath, additive_mv=additive_mv,
        )
    except ValueError:
        return np.full(2 * prepared.n_patches, np.nan)
    rhs = patch_rhs_fast(state, prepared, transfer)
    p = prepared.n_patches
    return np.r_[-TAU_ME * rhs[:p], -TAU_MI * rhs[p:2 * p]]


def regional_rate_jacobian(
    rates_khz: Sequence[float],
    prepared: PreparedPatchRHS,
    parameters: PatchParameters,
    transfer: Any,
    *,
    z_regional: float,
    z_bath: float = 0.90,
    additive_mv: float = 0.0,
    relative_step: float = 1.0e-4,
    absolute_step_khz: float = 1.0e-7,
) -> np.ndarray:
    """Centered Jacobian of the regional rate nullcline residual."""

    rates = np.asarray(rates_khz, dtype=float)
    if rates.shape != (2 * prepared.n_patches,) or np.any(rates <= 0.0):
        raise ValueError("rates must be positive and aligned")
    jac = np.empty((rates.size, rates.size), dtype=float)
    for column in range(rates.size):
        step = max(absolute_step_khz, relative_step * max(abs(float(rates[column])), 1.0e-3))
        plus = rates.copy()
        minus = rates.copy()
        plus[column] += step
        minus[column] -= step
        jac[:, column] = (
            regional_equilibrium_residual(
                plus, prepared, parameters, transfer,
                z_regional=z_regional, z_bath=z_bath, additive_mv=additive_mv,
            )
            - regional_equilibrium_residual(
                minus, prepared, parameters, transfer,
                z_regional=z_regional, z_bath=z_bath, additive_mv=additive_mv,
            )
        ) / (2.0 * step)
    return jac


def regional_fast_jacobian(
    state: Sequence[float],
    prepared: PreparedPatchRHS,
    transfer: Any,
    *,
    relative_step: float = 2.0e-4,
    absolute_step: float = 2.0e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast Jacobian after removing the nine frozen ``z/p/m`` zero modes."""

    vector = np.asarray(state, dtype=float)
    p = prepared.n_patches
    if vector.shape != (state_size(p),) or not np.all(np.isfinite(vector)):
        raise ValueError("state must be one finite packed patch state")
    fast_indices = np.r_[np.arange(7 * p), [10 * p, 10 * p + 1]].astype(int)
    jac = np.empty((fast_indices.size, fast_indices.size), dtype=float)
    for output_column, state_index in enumerate(fast_indices):
        step = max(
            absolute_step,
            relative_step * max(abs(float(vector[state_index])), 1.0e-3),
        )
        plus = vector.copy()
        minus = vector.copy()
        plus[state_index] += step
        minus[state_index] -= step
        jac[:, output_column] = (
            patch_rhs_fast(plus, prepared, transfer)[fast_indices]
            - patch_rhs_fast(minus, prepared, transfer)[fast_indices]
        ) / (2.0 * step)
    return jac, fast_indices


def find_regional_equilibria(
    z_regional: float,
    prepared: PreparedPatchRHS,
    parameters: PatchParameters,
    transfer: Any,
    *,
    z_bath: float = 0.90,
    additive_mv: float = 0.0,
    seeds_khz: Sequence[Sequence[float]] | None = None,
    residual_tolerance_khz: float = 2.0e-8,
    cluster_tolerance_khz: float = 2.0e-5,
) -> list[dict[str, Any]]:
    """Find distinct regional fixed points and classify their 23D fast spectrum."""

    p = prepared.n_patches
    if seeds_khz is None:
        e_levels = (0.0005, 0.002, 0.004, 0.006, 0.012)
        i_levels = (0.003, 0.007, 0.010, 0.020)
        seeds_khz = [
            np.r_[np.full(p, e), np.full(p, i)]
            for e in e_levels for i in i_levels
        ]
    roots: list[np.ndarray] = []
    for seed in seeds_khz:
        candidate = np.asarray(seed, dtype=float)
        if candidate.shape != (2 * p,):
            raise ValueError("every regional equilibrium seed must have 2P coordinates")
        try:
            fit = least_squares(
                lambda rates: regional_equilibrium_residual(
                    rates, prepared, parameters, transfer,
                    z_regional=z_regional, z_bath=z_bath, additive_mv=additive_mv,
                ),
                candidate,
                bounds=(np.full(2 * p, 1.0e-8), np.r_[np.full(p, 0.25), np.full(p, 0.60)]),
                xtol=1.0e-12, ftol=1.0e-12, gtol=1.0e-12,
                max_nfev=1000, x_scale="jac",
            )
        except ValueError:
            continue
        residual = regional_equilibrium_residual(
            fit.x, prepared, parameters, transfer,
            z_regional=z_regional, z_bath=z_bath, additive_mv=additive_mv,
        )
        if (
            fit.success
            and np.all(np.isfinite(residual))
            and float(np.max(np.abs(residual))) <= residual_tolerance_khz
            and not any(float(np.max(np.abs(fit.x - old))) <= cluster_tolerance_khz for old in roots)
        ):
            roots.append(np.asarray(fit.x, dtype=float))
    output = []
    for root in sorted(roots, key=lambda value: tuple(value[:p])):
        state = regional_equilibrium_state(
            root, prepared, parameters,
            z_regional=z_regional, z_bath=z_bath, additive_mv=additive_mv,
        )
        jac, _ = regional_fast_jacobian(state, prepared, transfer)
        eigenvalues = np.linalg.eigvals(jac)
        leading = eigenvalues[int(np.argmax(eigenvalues.real))]
        output.append({
            "z_regional": float(z_regional),
            "additive_mv": float(additive_mv),
            "rE_hz": (1000.0 * root[:p]).tolist(),
            "rI_hz": (1000.0 * root[p:]).tolist(),
            "stability": "stable" if leading.real < -1.0e-5 else (
                "unstable" if leading.real > 1.0e-5 else "marginal"
            ),
            "leading_real_per_ms": float(leading.real),
            "leading_imag_per_ms": float(abs(leading.imag)),
            "residual_inf_khz": float(np.max(np.abs(regional_equilibrium_residual(
                root, prepared, parameters, transfer,
                z_regional=z_regional, z_bath=z_bath, additive_mv=additive_mv,
            )))),
            "rates_khz": root.tolist(),
        })
    return output


def solve_regional_fold(
    prepared: PreparedPatchRHS,
    parameters: PatchParameters,
    transfer: Any,
    *,
    additive_mv: float = 0.0,
    z_bath: float = 0.90,
    initial_rates_khz: Sequence[float] = (
        0.00337716, 0.00227267, 0.00086962,
        0.00856125, 0.00731114, 0.00574814,
    ),
    initial_z: float = 0.8575,
) -> RegionalFoldPoint:
    """Solve ``F=0, D_rF v=0, ||v||=1`` for the regional-Z fold."""

    p = prepared.n_patches
    x0 = np.asarray(initial_rates_khz, dtype=float)
    if x0.shape != (2 * p,):
        raise ValueError("fold initial rates must have 2P coordinates")
    jac0 = regional_rate_jacobian(
        x0, prepared, parameters, transfer,
        z_regional=initial_z, z_bath=z_bath, additive_mv=additive_mv,
    )
    v0 = np.linalg.svd(jac0)[2][-1]
    v0 /= np.linalg.norm(v0)

    def augmented(value: np.ndarray) -> np.ndarray:
        rates = value[:2 * p]
        z_value = float(value[2 * p])
        null = value[2 * p + 1:]
        residual = regional_equilibrium_residual(
            rates, prepared, parameters, transfer,
            z_regional=z_value, z_bath=z_bath, additive_mv=additive_mv,
        )
        jac = regional_rate_jacobian(
            rates, prepared, parameters, transfer,
            z_regional=z_value, z_bath=z_bath, additive_mv=additive_mv,
        )
        return np.r_[1000.0 * residual, jac @ null, null @ null - 1.0]

    y0 = np.r_[x0, float(initial_z), v0]
    fit = least_squares(
        augmented,
        y0,
        bounds=(
            np.r_[np.full(2 * p, 1.0e-8), 0.84, np.full(2 * p, -1.5)],
            np.r_[np.full(p, 0.03), np.full(p, 0.08), 0.87, np.full(2 * p, 1.5)],
        ),
        xtol=1.0e-12, ftol=1.0e-12, gtol=1.0e-12,
        max_nfev=1500,
        x_scale=np.r_[np.full(p, 0.003), np.full(p, 0.008), 0.01, np.ones(2 * p)],
    )
    residual = augmented(fit.x)
    if not fit.success or not np.all(np.isfinite(residual)) or np.max(np.abs(residual)) > 2.0e-8:
        raise RuntimeError(f"regional fold solve failed: {fit.message}; residual={residual}")
    rates = fit.x[:2 * p]
    z_fold = float(fit.x[2 * p])
    null = fit.x[2 * p + 1:]
    null /= np.linalg.norm(null)
    rate_jac = regional_rate_jacobian(
        rates, prepared, parameters, transfer,
        z_regional=z_fold, z_bath=z_bath, additive_mv=additive_mv,
    )
    left, singular, right = np.linalg.svd(rate_jac)
    u = left[:, -1]
    v = right[-1]
    z_step = 1.0e-5
    fz = (
        regional_equilibrium_residual(
            rates, prepared, parameters, transfer,
            z_regional=z_fold + z_step, z_bath=z_bath, additive_mv=additive_mv,
        )
        - regional_equilibrium_residual(
            rates, prepared, parameters, transfer,
            z_regional=z_fold - z_step, z_bath=z_bath, additive_mv=additive_mv,
        )
    ) / (2.0 * z_step)
    v_step = 1.0e-5
    d2f_vv = (
        regional_equilibrium_residual(
            rates + v_step * v, prepared, parameters, transfer,
            z_regional=z_fold, z_bath=z_bath, additive_mv=additive_mv,
        )
        - 2.0 * regional_equilibrium_residual(
            rates, prepared, parameters, transfer,
            z_regional=z_fold, z_bath=z_bath, additive_mv=additive_mv,
        )
        + regional_equilibrium_residual(
            rates - v_step * v, prepared, parameters, transfer,
            z_regional=z_fold, z_bath=z_bath, additive_mv=additive_mv,
        )
    ) / v_step**2
    state = regional_equilibrium_state(
        rates, prepared, parameters,
        z_regional=z_fold, z_bath=z_bath, additive_mv=additive_mv,
    )
    fast_jac, _ = regional_fast_jacobian(state, prepared, transfer)
    eigenvalues, eigenvectors = np.linalg.eig(fast_jac)
    lead_index = int(np.argmax(eigenvalues.real))
    leading = eigenvalues[lead_index]
    critical = np.real(eigenvectors[:, lead_index])
    critical /= np.max(np.abs(critical))
    _, moments = patch_rhs_fast_and_moments(state, prepared, transfer)
    mu_e, sigma_e, mu_i, sigma_i, _ = moments
    support = transfer.support_mask(mu_e, sigma_e) & transfer.support_mask(mu_i, sigma_i)
    return RegionalFoldPoint(
        z_regional=z_fold,
        additive_mv=float(additive_mv),
        rates_khz=rates,
        null_vector=null,
        residual_inf=float(np.max(np.abs(residual))),
        rate_sigma_min=float(singular[-1]),
        fast_leading_real_per_ms=float(leading.real),
        fast_leading_imag_per_ms=float(abs(leading.imag)),
        left_fz=float(u @ fz),
        left_d2f_vv=float(u @ d2f_vv),
        support_all=bool(np.all(support)),
        critical_fast_mode=critical,
    )


def solve_regional_additive_fold(
    z_regional: float,
    prepared: PreparedPatchRHS,
    parameters: PatchParameters,
    transfer: Any,
    *,
    z_bath: float = 0.90,
    initial_rates_khz: Sequence[float] = (
        0.0034, 0.0023, 0.00087, 0.0086, 0.0073, 0.00575,
    ),
    initial_additive_mv: float = 0.02,
) -> RegionalFoldPoint:
    """Solve the low-state fold in additive current at one fixed regional Z."""

    p = prepared.n_patches
    x0 = np.asarray(initial_rates_khz, dtype=float)
    if x0.shape != (2 * p,):
        raise ValueError("fold initial rates must have 2P coordinates")
    jac0 = regional_rate_jacobian(
        x0, prepared, parameters, transfer,
        z_regional=z_regional, z_bath=z_bath, additive_mv=initial_additive_mv,
    )
    v0 = np.linalg.svd(jac0)[2][-1]
    v0 /= np.linalg.norm(v0)

    def augmented(value: np.ndarray) -> np.ndarray:
        rates = value[:2 * p]
        additive = float(value[2 * p])
        null = value[2 * p + 1:]
        residual = regional_equilibrium_residual(
            rates, prepared, parameters, transfer,
            z_regional=z_regional, z_bath=z_bath, additive_mv=additive,
        )
        jac = regional_rate_jacobian(
            rates, prepared, parameters, transfer,
            z_regional=z_regional, z_bath=z_bath, additive_mv=additive,
        )
        return np.r_[1000.0 * residual, jac @ null, null @ null - 1.0]

    fit = least_squares(
        augmented,
        np.r_[x0, float(initial_additive_mv), v0],
        bounds=(
            np.r_[np.full(2 * p, 1.0e-8), 0.0, np.full(2 * p, -1.5)],
            np.r_[np.full(p, 0.03), np.full(p, 0.08), parameters.additive_max_mv,
                  np.full(2 * p, 1.5)],
        ),
        xtol=1.0e-12, ftol=1.0e-12, gtol=1.0e-12,
        max_nfev=1500,
        x_scale=np.r_[np.full(p, 0.003), np.full(p, 0.008), 0.05, np.ones(2 * p)],
    )
    residual = augmented(fit.x)
    if not fit.success or not np.all(np.isfinite(residual)) or np.max(np.abs(residual)) > 2.0e-8:
        raise RuntimeError(f"regional additive fold solve failed: {fit.message}; residual={residual}")
    rates = fit.x[:2 * p]
    additive = float(fit.x[2 * p])
    rate_jac = regional_rate_jacobian(
        rates, prepared, parameters, transfer,
        z_regional=z_regional, z_bath=z_bath, additive_mv=additive,
    )
    left, singular, right = np.linalg.svd(rate_jac)
    u = left[:, -1]
    v = right[-1]
    a_step = 1.0e-5
    fa = (
        regional_equilibrium_residual(
            rates, prepared, parameters, transfer,
            z_regional=z_regional, z_bath=z_bath, additive_mv=additive + a_step,
        )
        - regional_equilibrium_residual(
            rates, prepared, parameters, transfer,
            z_regional=z_regional, z_bath=z_bath, additive_mv=max(0.0, additive - a_step),
        )
    ) / (a_step + min(a_step, additive))
    v_step = 1.0e-5
    d2f_vv = (
        regional_equilibrium_residual(
            rates + v_step * v, prepared, parameters, transfer,
            z_regional=z_regional, z_bath=z_bath, additive_mv=additive,
        )
        - 2.0 * regional_equilibrium_residual(
            rates, prepared, parameters, transfer,
            z_regional=z_regional, z_bath=z_bath, additive_mv=additive,
        )
        + regional_equilibrium_residual(
            rates - v_step * v, prepared, parameters, transfer,
            z_regional=z_regional, z_bath=z_bath, additive_mv=additive,
        )
    ) / v_step**2
    state = regional_equilibrium_state(
        rates, prepared, parameters,
        z_regional=z_regional, z_bath=z_bath, additive_mv=additive,
    )
    fast_jac, _ = regional_fast_jacobian(state, prepared, transfer)
    eigenvalues, eigenvectors = np.linalg.eig(fast_jac)
    lead_index = int(np.argmax(eigenvalues.real))
    leading = eigenvalues[lead_index]
    critical = np.real(eigenvectors[:, lead_index])
    critical /= np.max(np.abs(critical))
    _, moments = patch_rhs_fast_and_moments(state, prepared, transfer)
    mu_e, sigma_e, mu_i, sigma_i, _ = moments
    support = transfer.support_mask(mu_e, sigma_e) & transfer.support_mask(mu_i, sigma_i)
    return RegionalFoldPoint(
        z_regional=float(z_regional),
        additive_mv=additive,
        rates_khz=rates,
        null_vector=fit.x[2 * p + 1:] / np.linalg.norm(fit.x[2 * p + 1:]),
        residual_inf=float(np.max(np.abs(residual))),
        rate_sigma_min=float(singular[-1]),
        fast_leading_real_per_ms=float(leading.real),
        fast_leading_imag_per_ms=float(abs(leading.imag)),
        left_fz=float(u @ fa),
        left_d2f_vv=float(u @ d2f_vv),
        support_all=bool(np.all(support)),
        critical_fast_mode=critical,
    )
