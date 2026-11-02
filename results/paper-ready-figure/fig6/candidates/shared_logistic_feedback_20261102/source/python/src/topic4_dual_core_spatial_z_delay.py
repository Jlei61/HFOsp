"""Delay-aware linear assay for the data-driven dual-core spatial-Z reduction.

The fixed points do not depend on axonal delay, but their stability does.  This
module retains every quantized delay bin from the realized SNN graph, projects
each bin to the same coarse cells as the fixed-point model, and estimates the
dominant growth rate of the native-dt delayed tangent map by power iteration.

The tangent map includes both transfer-function mean gain and the instantaneous
diffusion-variance gain used by the nonlinear coarse model.  OU forcing is
stochastic and is therefore handled by a separate residence/transition assay
rather than folded into a linear eigenvalue label.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

from src.topic4_patient_zm_meanfield import (
    PatientCoarseZMModel,
    _aggregate_pathway,
    _transfer_derivatives,
    spatial_cell_index,
    transfer_rates,
)
from src.topic4_dual_core_spatial_z import spatial_z_moments


@dataclass(frozen=True)
class CoarseDelayOperators:
    """Coarse pathway matrices concatenated over delay steps 1..D."""

    dt_ms: float
    max_delay_steps: int
    w_ee_history: sparse.csr_matrix
    w_ei_history: sparse.csr_matrix
    w_ie_history: sparse.csr_matrix
    w_ii_history: sparse.csr_matrix

    def validate(self, model: PatientCoarseZMModel) -> None:
        n = model.n_cells
        expected = (n, self.max_delay_steps * n)
        if self.dt_ms <= 0.0 or self.max_delay_steps < 1:
            raise ValueError("delay dt and maximum delay must be positive")
        for name in (
                "w_ee_history", "w_ei_history",
                "w_ie_history", "w_ii_history"):
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if value.nnz and (
                    not np.all(np.isfinite(value.data))
                    or np.any(value.data < 0.0)):
                raise ValueError(f"{name} must be finite and non-negative")


def _history_matrix(matrices) -> sparse.csr_matrix:
    return sparse.hstack(
        [sparse.csr_matrix(matrix) for matrix in matrices], format="csr")


def _sum_history_blocks(matrix: sparse.csr_matrix, n: int) -> np.ndarray:
    total = np.zeros((n, n), float)
    for index in range(matrix.shape[1] // n):
        total += matrix[:, index * n:(index + 1) * n].toarray()
    return total


def build_coarse_delay_operators(substrate, model: PatientCoarseZMModel
                                 ) -> CoarseDelayOperators:
    """Project the realized SNN's exact delay bins onto the audited coarse grid."""
    n_e, n_i = int(substrate.n_e), int(substrate.n_i)
    cell_e = spatial_cell_index(
        substrate.positions_e, n_grid=model.n_grid,
        sheet_l_mm=model.sheet_l_mm)
    cell_i = spatial_cell_index(
        substrate.positions_i, n_grid=model.n_grid,
        sheet_l_mm=model.sheet_l_mm)
    all_cells = np.r_[cell_e, cell_i]
    count_e = np.asarray(model.count_e, float)
    count_i = np.asarray(model.count_i, float)
    maximum = int(substrate.net["max_delay_steps"])
    ampa = substrate.net["ampa_by_delay"]
    gaba = substrate.net["gaba_by_delay"]
    if len(ampa) != maximum + 1 or len(gaba) != maximum + 1:
        raise ValueError("SNN delay-bin lists do not match max_delay_steps")

    pathways = {name: [] for name in ("ee", "ei", "ie", "ii")}
    for step in range(1, maximum + 1):
        ee, _ = _aggregate_pathway(
            (ampa[step],), target_cells=all_cells, source_cells=cell_e,
            target_mask=lambda rows: rows < n_e, n_cells=model.n_cells,
            target_counts=count_e,
            physical_factor=substrate.params.tau_r_AMPA
            / substrate.params.tau_m_E)
        ie, _ = _aggregate_pathway(
            (ampa[step],), target_cells=all_cells, source_cells=cell_e,
            target_mask=lambda rows: rows >= n_e, n_cells=model.n_cells,
            target_counts=count_i,
            physical_factor=substrate.params.tau_r_AMPA
            / substrate.params.tau_m_I)
        ei, _ = _aggregate_pathway(
            (gaba[step],), target_cells=all_cells, source_cells=cell_i,
            target_mask=lambda rows: rows < n_e, n_cells=model.n_cells,
            target_counts=count_e,
            physical_factor=substrate.params.tau_r_GABA
            / substrate.params.tau_m_E)
        ii, _ = _aggregate_pathway(
            (gaba[step],), target_cells=all_cells, source_cells=cell_i,
            target_mask=lambda rows: rows >= n_e, n_cells=model.n_cells,
            target_counts=count_i,
            physical_factor=substrate.params.tau_r_GABA
            / substrate.params.tau_m_I)
        for name, value in (("ee", ee), ("ei", ei), ("ie", ie), ("ii", ii)):
            pathways[name].append(value)

    result = CoarseDelayOperators(
        dt_ms=float(substrate.params.dt),
        max_delay_steps=maximum,
        w_ee_history=_history_matrix(pathways["ee"]),
        w_ei_history=_history_matrix(pathways["ei"]),
        w_ie_history=_history_matrix(pathways["ie"]),
        w_ii_history=_history_matrix(pathways["ii"]),
    )
    result.validate(model)
    for name in ("ee", "ei", "ie", "ii"):
        observed = _sum_history_blocks(
            getattr(result, f"w_{name}_history"), model.n_cells)
        expected = np.asarray(getattr(model, f"w_{name}"), float)
        if not np.allclose(observed, expected, rtol=1e-11, atol=1e-13):
            raise RuntimeError(f"delay-bin projection does not sum to model.w_{name}")
    return result


def _state_norm(rate_e, rate_i, synapses, history_e, history_i) -> float:
    squared = (
        np.dot(rate_e, rate_e) + np.dot(rate_i, rate_i)
        + np.dot(synapses.ravel(), synapses.ravel())
        + np.dot(history_e.ravel(), history_e.ravel())
        + np.dot(history_i.ravel(), history_i.ravel())
    )
    return float(np.sqrt(squared))


def delayed_step_matrix(
    model: PatientCoarseZMModel,
    operators: CoarseDelayOperators,
    rates,
    *,
    z_field,
    z_second_moment=None,
    eta_m: float = 0.0,
    tau_m_slow_ms: float = 500.0,
) -> sparse.csr_matrix:
    """Build the exact native-``dt`` tangent map including every delay bin.

    The state order is ``rE, rI, sEE, sEI, sIE, sII, historyE, historyI``
    and, when ``eta_m > 0``, a final local ``M`` block.
    ``history[0]`` is the immediately preceding rate state, matching
    :func:`simulate_delayed_ou_trajectory` and the SNN delay convention.
    """
    operators.validate(model)
    rates = np.asarray(rates, float)
    if rates.shape != (2 * model.n_cells,):
        raise ValueError("rates must concatenate E then I cells")
    rate_e, rate_i = np.split(rates, 2)
    z = np.asarray(z_field, float)
    z2 = z ** 2 if z_second_moment is None else np.asarray(
        z_second_moment, float)
    mu_e, sigma_e, mu_i, sigma_i = spatial_z_moments(
        model, rate_e, rate_i, z_field=z, z_second_moment=z2)
    pme, pse, pmi, psi = _transfer_derivatives(
        model, mu_e, sigma_e, mu_i, sigma_i)

    eta_m = float(eta_m)
    tau_m_slow_ms = float(tau_m_slow_ms)
    if not np.isfinite(eta_m) or eta_m < 0.0:
        raise ValueError("eta_m must be finite and non-negative")
    if not np.isfinite(tau_m_slow_ms) or tau_m_slow_ms <= 0.0:
        raise ValueError("tau_m_slow_ms must be finite and positive")
    include_adaptation = eta_m > 0.0
    n = model.n_cells
    delay = operators.max_delay_steps
    dt = operators.dt_ms
    te, ti = model.tau_mem_e_ms, model.tau_mem_i_ms
    ta, tg = model.tau_ampa_ms, model.tau_gaba_ms
    eye = sparse.eye(n, format="csr")
    de = sparse.diags(pme, format="csr")
    di = sparse.diags(pmi, format="csr")
    ze = sparse.diags(z, format="csr")
    variance_ee = sparse.diags(
        pse / (2.0 * sigma_e), format="csr") @ sparse.csr_matrix(model.v_ee)
    variance_ei = sparse.diags(
        pse * z2 / (2.0 * sigma_e), format="csr") @ sparse.csr_matrix(
            model.v_ei)
    variance_ie = sparse.diags(
        psi / (2.0 * sigma_i), format="csr") @ sparse.csr_matrix(model.v_ie)
    variance_ii = sparse.diags(
        psi / (2.0 * sigma_i), format="csr") @ sparse.csr_matrix(model.v_ii)

    # Eight macro blocks without M; a ninth local adaptation block is appended
    # only when eta_m is active so the eta_m=0 spectrum remains unchanged.
    n_blocks = 9 if include_adaptation else 8
    blocks = [[None for _ in range(n_blocks)] for _ in range(n_blocks)]
    sizes = [n, n, n, n, n, n, delay * n, delay * n]
    if include_adaptation:
        sizes.append(n)
    for row in range(n_blocks):
        for column in range(n_blocks):
            blocks[row][column] = sparse.csr_matrix((sizes[row], sizes[column]))

    blocks[0][0] = (1.0 - dt / te) * eye + dt * variance_ee
    blocks[0][1] = dt * variance_ei
    blocks[0][2] = (dt / te) * de
    blocks[0][3] = -(dt / te) * ze @ de
    if include_adaptation:
        blocks[0][8] = -(dt / te) * eta_m * de
    blocks[1][0] = dt * variance_ie
    blocks[1][1] = (1.0 - dt / ti) * eye + dt * variance_ii
    blocks[1][4] = (dt / ti) * di
    blocks[1][5] = -(dt / ti) * di

    blocks[2][2] = (1.0 - dt / ta) * eye
    blocks[2][6] = (dt * te / ta) * operators.w_ee_history
    blocks[3][3] = (1.0 - dt / tg) * eye
    blocks[3][7] = (dt * te / tg) * operators.w_ei_history
    blocks[4][4] = (1.0 - dt / ta) * eye
    blocks[4][6] = (dt * ti / ta) * operators.w_ie_history
    blocks[5][5] = (1.0 - dt / tg) * eye
    blocks[5][7] = (dt * ti / tg) * operators.w_ii_history

    history_shift = sparse.diags(
        np.ones(max((delay - 1) * n, 0), float), offsets=-n,
        shape=(delay * n, delay * n), format="csr")
    history_insert = sparse.vstack([
        eye,
        sparse.csr_matrix(((delay - 1) * n, n)),
    ], format="csr")
    blocks[6][0] = history_insert
    blocks[6][6] = history_shift
    blocks[7][1] = history_insert
    blocks[7][7] = history_shift
    if include_adaptation:
        blocks[8][0] = dt * eye
        blocks[8][8] = (1.0 - dt / tau_m_slow_ms) * eye
    return sparse.bmat(blocks, format="csr")


def delayed_leading_eigenvalues(
    model: PatientCoarseZMModel,
    operators: CoarseDelayOperators,
    rates,
    *,
    z_field,
    z_second_moment=None,
    eta_m: float = 0.0,
    tau_m_slow_ms: float = 500.0,
    k: int = 8,
    tolerance: float = 1e-8,
    maxiter: int = 100000,
) -> list[dict]:
    """Return leading discrete-delay modes ranked by growth exponent.

    Each multiplier ``mu`` of the native-step map is reported as the
    continuous growth exponent ``log(abs(mu))/dt`` and oscillation frequency
    ``abs(angle(mu))/(2*pi*dt)``.  This directly resolves whether an apparent
    high-state instability is a real oscillatory delay mode.
    """
    matrix = delayed_step_matrix(
        model, operators, rates, z_field=z_field,
        z_second_moment=z_second_moment, eta_m=eta_m,
        tau_m_slow_ms=tau_m_slow_ms)
    values = eigs(
        matrix, k=int(k), which="LM", return_eigenvectors=False,
        tol=float(tolerance), maxiter=int(maxiter))
    dt = float(operators.dt_ms)
    records = []
    for value in values:
        magnitude = float(abs(value))
        records.append({
            "multiplier_real": float(value.real),
            "multiplier_imag": float(value.imag),
            "multiplier_abs": magnitude,
            "growth_rate_per_ms": float(np.log(magnitude) / dt),
            "frequency_hz": float(abs(np.angle(value)) * 1000.0
                                  / (2.0 * np.pi * dt)),
        })
    return sorted(
        records, key=lambda record: record["growth_rate_per_ms"],
        reverse=True)


def delayed_growth_rate(
    model: PatientCoarseZMModel,
    operators: CoarseDelayOperators,
    rates,
    *,
    z_field,
    z_second_moment=None,
    n_steps: int = 5000,
    burn_in_steps: int = 2500,
    seeds=(101, 102, 103),
) -> dict:
    """Estimate the dominant native-dt delayed growth exponent in ``ms^-1``."""
    operators.validate(model)
    if not 0 <= int(burn_in_steps) < int(n_steps):
        raise ValueError("burn_in_steps must lie in [0, n_steps)")
    rates = np.asarray(rates, float)
    rate_e, rate_i = np.split(rates, 2)
    z = np.asarray(z_field, float)
    z2 = z ** 2 if z_second_moment is None else np.asarray(
        z_second_moment, float)
    mu_e, sigma_e, mu_i, sigma_i = spatial_z_moments(
        model, rate_e, rate_i, z_field=z, z_second_moment=z2)
    pme, pse, pmi, psi = _transfer_derivatives(
        model, mu_e, sigma_e, mu_i, sigma_i)
    n = model.n_cells
    delay = operators.max_delay_steps
    dt = operators.dt_ms
    te, ti = model.tau_mem_e_ms, model.tau_mem_i_ms
    ta, tg = model.tau_ampa_ms, model.tau_gaba_ms
    estimates = []
    block_estimates = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        de = rng.standard_normal(n)
        di = rng.standard_normal(n)
        syn = rng.standard_normal((4, n))
        he = rng.standard_normal((delay, n))
        hi = rng.standard_normal((delay, n))
        scale = _state_norm(de, di, syn, he, hi)
        de, di, syn, he, hi = (
            de / scale, di / scale, syn / scale, he / scale, hi / scale)
        log_scales = []
        for step in range(int(n_steps)):
            flat_e = he.reshape(-1)
            flat_i = hi.reshape(-1)
            drive = np.stack([
                te * (operators.w_ee_history @ flat_e) / ta,
                te * (operators.w_ei_history @ flat_i) / tg,
                ti * (operators.w_ie_history @ flat_e) / ta,
                ti * (operators.w_ii_history @ flat_i) / tg,
            ])
            dsigma_e = te * (
                model.v_ee @ de + z2 * (model.v_ei @ di)
            ) / (2.0 * sigma_e)
            dsigma_i = ti * (
                model.v_ie @ de + model.v_ii @ di
            ) / (2.0 * sigma_i)
            next_de = de + dt * (
                -de + pme * (syn[0] - z * syn[1])
                + pse * dsigma_e) / te
            next_di = di + dt * (
                -di + pmi * (syn[2] - syn[3])
                + psi * dsigma_i) / ti
            next_syn = syn + dt * (
                drive - syn / np.asarray([ta, tg, ta, tg])[:, None])
            next_he = np.empty_like(he)
            next_hi = np.empty_like(hi)
            next_he[0], next_hi[0] = de, di
            next_he[1:], next_hi[1:] = he[:-1], hi[:-1]
            scale = _state_norm(
                next_de, next_di, next_syn, next_he, next_hi)
            if not np.isfinite(scale) or scale <= 0.0:
                raise RuntimeError("delayed tangent iteration became non-finite")
            de, di, syn, he, hi = (
                next_de / scale, next_di / scale, next_syn / scale,
                next_he / scale, next_hi / scale)
            if step >= int(burn_in_steps):
                log_scales.append(np.log(scale))
        estimates.append(float(np.mean(log_scales) / dt))
        blocks = np.array_split(np.asarray(log_scales, float), 4)
        block_estimates.append([
            float(np.mean(block) / dt) for block in blocks if block.size
        ])
    estimates = np.asarray(estimates, float)
    recent_blocks = np.asarray([row[-2:] for row in block_estimates], float)
    threshold = 1e-4
    classification = (
        "unstable" if np.all(recent_blocks > threshold)
        else "stable" if np.all(recent_blocks < -threshold)
        else "unresolved"
    )
    return {
        "growth_rate_per_ms": float(np.median(estimates)),
        "per_seed_growth_rate_per_ms": estimates.tolist(),
        "per_seed_quarter_growth_rate_per_ms": block_estimates,
        "spread_per_ms": float(np.max(estimates) - np.min(estimates)),
        "n_steps": int(n_steps),
        "burn_in_steps": int(burn_in_steps),
        "dt_ms": float(dt),
        "maximum_delay_ms": float(delay * dt),
        "classification": classification,
        "classification_rule": (
            "all seeds and both final post-burn-in quarters exceed +1e-4 "
            "per ms (unstable) or fall below -1e-4 per ms (stable)"
        ),
    }


def coarse_spatial_ou_trace(substrate, model: PatientCoarseZMModel, config,
                            *, seed: int, n_steps: int) -> np.ndarray:
    """Generate the exact SNN OU field and average it onto the coarse E cells."""
    from src.topic4_spatial_ou_drive import SpatialOUConfig, SpatialOUDrive

    cfg = SpatialOUConfig(
        mode=str(config["mode"]),
        sigma_rate_per_ms=float(config["sigma_rate_per_ms"]),
        tau_ms=float(config["tau_ms"]),
        ell_mm=float(config["ell_mm"]),
        update_interval_ms=float(config["update_interval_ms"]),
        grid_spacing_mm=float(config["grid_spacing_mm"]),
        seed=int(seed) + int(config.get("seed_offset", 0)),
    )
    drive = SpatialOUDrive(
        substrate.positions_e, model.sheet_l_mm,
        float(substrate.params.dt), cfg)
    cell_e = spatial_cell_index(
        substrate.positions_e, n_grid=model.n_grid,
        sheet_l_mm=model.sheet_l_mm)
    count = np.asarray(model.count_e, float)
    result = np.empty((int(n_steps), model.n_cells), np.float32)
    cached = np.bincount(
        cell_e, weights=drive.step(0.0), minlength=model.n_cells) / count
    update_steps = int(round(cfg.update_interval_ms / substrate.params.dt))
    for step in range(int(n_steps)):
        if step and step % update_steps == 0:
            values = drive.step(step * substrate.params.dt)
            cached = np.bincount(
                cell_e, weights=values, minlength=model.n_cells) / count
        result[step] = cached
    return result


def simulate_delayed_ou_trajectory(
    model: PatientCoarseZMModel,
    operators: CoarseDelayOperators,
    initial_rates,
    *,
    z_field,
    z_second_moment=None,
    ou_rate_e=None,
    tail_steps: int = 500,
    reference_rates=None,
    eta_m: float = 0.0,
    tau_m_slow_ms: float = 500.0,
    initial_m=None,
    initial_synapses=None,
    initial_history_e=None,
    initial_history_i=None,
) -> dict:
    """Integrate the nonlinear coarse rate model with delay bins and OU drive.

    Recurrent means use the exact coarse delay operators.  Recurrent variance
    follows the current coarse rates, and the external mean/variance use the
    clipped OU-modulated afferent rate, so this assay does not freeze the
    operating variance.  When ``eta_m > 0``, the coarse adaptation state obeys

    ``dm/dt = -m/tau_m + r_E`` and contributes ``-eta_m*m`` to the E mean
    input, exactly matching the rate-level steady-state convention used by
    :func:`src.topic4_dual_core_spatial_z.solve_spatial_z_fixed_point`.

    Z remains frozen by design: this is a fast-subsystem/regime assay at a
    specified spatial inhibitory-efficacy field, not a replacement for the
    full SNN's dynamic per-neuron Z equation.
    """
    operators.validate(model)
    rates = np.asarray(initial_rates, float).copy()
    if rates.shape != (2 * model.n_cells,):
        raise ValueError("initial_rates must concatenate E then I cells")
    z = np.asarray(z_field, float)
    z2 = z ** 2 if z_second_moment is None else np.asarray(
        z_second_moment, float)
    if ou_rate_e is None:
        ou = np.zeros((1, model.n_cells), float)
    else:
        ou = np.asarray(ou_rate_e, float)
    if ou.ndim != 2 or ou.shape[1] != model.n_cells:
        raise ValueError("ou_rate_e must have shape (steps, n_cells)")
    eta_m = float(eta_m)
    tau_m_slow_ms = float(tau_m_slow_ms)
    if not np.isfinite(eta_m) or eta_m < 0.0:
        raise ValueError("eta_m must be finite and non-negative")
    if not np.isfinite(tau_m_slow_ms) or tau_m_slow_ms <= 0.0:
        raise ValueError("tau_m_slow_ms must be finite and positive")
    steps = int(ou.shape[0])
    tail_steps = min(int(tail_steps), steps)
    if tail_steps < 1:
        raise ValueError("tail_steps must be positive")
    n = model.n_cells
    dt = operators.dt_ms
    delay = operators.max_delay_steps
    rate_e, rate_i = np.split(rates, 2)
    if initial_m is None:
        adaptation = (
            tau_m_slow_ms * rate_e if eta_m > 0.0
            else np.zeros_like(rate_e)
        )
    else:
        adaptation = np.asarray(initial_m, float).copy()
        if adaptation.shape != (n,):
            raise ValueError("initial_m must have one value per coarse E cell")
        if np.any(~np.isfinite(adaptation)) or np.any(adaptation < 0.0):
            raise ValueError("initial_m must be finite and non-negative")
    if initial_synapses is None:
        syn = np.stack([
            model.tau_mem_e_ms * (model.w_ee @ rate_e),
            model.tau_mem_e_ms * (model.w_ei @ rate_i),
            model.tau_mem_i_ms * (model.w_ie @ rate_e),
            model.tau_mem_i_ms * (model.w_ii @ rate_i),
        ])
    else:
        syn = np.asarray(initial_synapses, float).copy()
        if syn.shape != (4, n) or np.any(~np.isfinite(syn)):
            raise ValueError("initial_synapses must be finite with shape (4, n_cells)")

    def initial_history(value, rate, label):
        if value is None:
            return np.repeat(rate[None, :], delay, axis=0)
        result = np.asarray(value, float).copy()
        if result.shape != (delay, n) or np.any(~np.isfinite(result)):
            raise ValueError(
                f"{label} must be finite with shape (max_delay_steps, n_cells)")
        return result

    history_e = initial_history(initial_history_e, rate_e, "initial_history_e")
    history_i = initial_history(initial_history_i, rate_i, "initial_history_i")
    trace_e = np.empty(steps, np.float32)
    trace_i = np.empty(steps, np.float32)
    trace_m = np.empty(steps, np.float32)
    trace_rms = np.empty(steps, np.float32)
    reference_rates = (rates.copy() if reference_rates is None
                       else np.asarray(reference_rates, float))
    if reference_rates.shape != rates.shape:
        raise ValueError("reference_rates must match initial_rates")
    tail_sum = np.zeros(2 * n, float)
    te, ti = model.tau_mem_e_ms, model.tau_mem_i_ms
    ta, tg = model.tau_ampa_ms, model.tau_gaba_ms
    for step in range(steps):
        flat_e = history_e.reshape(-1)
        flat_i = history_i.reshape(-1)
        drive = np.stack([
            te * (operators.w_ee_history @ flat_e) / ta,
            te * (operators.w_ei_history @ flat_i) / tg,
            ti * (operators.w_ie_history @ flat_e) / ta,
            ti * (operators.w_ii_history @ flat_i) / tg,
        ])
        next_syn = syn + dt * (
            drive - syn / np.asarray([ta, tg, ta, tg])[:, None])
        nu_e = np.maximum(model.nu_ext_per_ms + ou[step], 0.0)
        mu_e = (
            syn[0] - z * syn[1] - eta_m * adaptation
            + te * model.j_ext_e_mv * nu_e
        )
        mu_i = (
            syn[2] - syn[3]
            + ti * model.j_ext_i_mv * model.nu_ext_per_ms)
        variance_e = te * (
            model.v_ee @ rate_e + z2 * (model.v_ei @ rate_i)
            + model.j_ext_e_mv ** 2 * nu_e)
        variance_i = ti * (
            model.v_ie @ rate_e + model.v_ii @ rate_i
            + model.j_ext_i_mv ** 2 * model.nu_ext_per_ms)
        phi_e, phi_i = transfer_rates(
            model, mu_e, np.sqrt(np.maximum(variance_e, 1e-12)),
            mu_i, np.sqrt(np.maximum(variance_i, 1e-12)))
        next_e = np.clip(
            rate_e + dt * (-rate_e + phi_e) / te,
            0.0, 1.0 / model.tau_ref_e_ms)
        next_i = np.clip(
            rate_i + dt * (-rate_i + phi_i) / ti,
            0.0, 1.0 / model.tau_ref_i_ms)
        next_adaptation = np.maximum(
            adaptation + dt * (-adaptation / tau_m_slow_ms + rate_e),
            0.0,
        )
        next_history_e = np.empty_like(history_e)
        next_history_i = np.empty_like(history_i)
        next_history_e[0], next_history_i[0] = rate_e, rate_i
        next_history_e[1:], next_history_i[1:] = (
            history_e[:-1], history_i[:-1])
        rate_e, rate_i = next_e, next_i
        adaptation = next_adaptation
        syn = next_syn
        history_e, history_i = next_history_e, next_history_i
        trace_e[step] = 1000.0 * np.mean(rate_e)
        trace_i[step] = 1000.0 * np.mean(rate_i)
        trace_m[step] = np.mean(adaptation)
        trace_rms[step] = 1000.0 * np.sqrt(np.mean(
            (np.r_[rate_e, rate_i] - reference_rates) ** 2))
        if step >= steps - tail_steps:
            tail_sum += np.r_[rate_e, rate_i]
    return {
        "final_rates": np.r_[rate_e, rate_i],
        "tail_mean_rates": tail_sum / tail_steps,
        "tail_steps": tail_steps,
        "mean_e_rate_hz": trace_e,
        "mean_i_rate_hz": trace_i,
        "mean_adaptation_state": trace_m,
        "final_adaptation_state": adaptation,
        "final_synapses": syn,
        "final_history_e": history_e,
        "final_history_i": history_i,
        "eta_m": eta_m,
        "tau_m_slow_ms": tau_m_slow_ms,
        "rms_rate_deviation_from_initial_hz": trace_rms,
        "dt_ms": float(dt),
    }
