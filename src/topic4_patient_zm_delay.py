"""Delay-aware linear stability for the patient-matched coarse Z/M model.

The fixed-point reduction stores pathway sums but not the realized conduction
delays.  This module conservatively aggregates the original delay-binned SNN
edges onto the same coarse cells, then constructs a discrete-time linear map
with an explicit rate-history shift register.  Every recurrent first and
second weight moment is preserved exactly under temporal rebinning; only the
requested history time step changes the delay resolution.

Two explicitly labelled closures are supported.  ``self_consistent_variance``
lets delayed rates perturb the stationary diffusion variance, so its zero-
frequency gain is exactly the one used by the fixed-point equations.
``frozen_variance`` retains the older M3B diagnostic convention as a
sensitivity analysis.  Neither closure models colored OU fluctuations.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

from src.topic4_patient_zm_meanfield import (
    PatientCoarseZMModel,
    _transfer_derivatives,
    moments,
    spatial_cell_index,
)


PATHWAYS = ("ee", "ei", "ie", "ii")


@dataclass(frozen=True)
class PatientCoarseDelayOperator:
    """Sparse per-delay pathway weights aligned to a coarse Z/M model."""

    n_grid: int
    sheet_l_mm: float
    source_delay_dt_ms: float
    max_delay_steps: int
    delay_step_ee: np.ndarray
    target_ee: np.ndarray
    source_ee: np.ndarray
    weight_ee: np.ndarray
    variance_weight_ee: np.ndarray
    delay_step_ei: np.ndarray
    target_ei: np.ndarray
    source_ei: np.ndarray
    weight_ei: np.ndarray
    variance_weight_ei: np.ndarray
    delay_step_ie: np.ndarray
    target_ie: np.ndarray
    source_ie: np.ndarray
    weight_ie: np.ndarray
    variance_weight_ie: np.ndarray
    delay_step_ii: np.ndarray
    target_ii: np.ndarray
    source_ii: np.ndarray
    weight_ii: np.ndarray
    variance_weight_ii: np.ndarray

    @property
    def n_cells(self):
        return int(self.n_grid) ** 2

    def pathway(self, name):
        if name not in PATHWAYS:
            raise KeyError(name)
        return tuple(np.asarray(getattr(self, f"{field}_{name}"))
                     for field in ("delay_step", "target", "source", "weight",
                                   "variance_weight"))

    def validate(self):
        if self.n_grid < 1 or self.source_delay_dt_ms <= 0.0:
            raise ValueError("delay operator grid and time step must be positive")
        n = self.n_cells
        for name in PATHWAYS:
            delay, target, source, weight, variance_weight = self.pathway(name)
            if not (delay.shape == target.shape == source.shape == weight.shape
                    == variance_weight.shape):
                raise ValueError(f"{name} delay arrays must align")
            if delay.ndim != 1 or np.any(delay < 1):
                raise ValueError(f"{name} delay steps must be positive vectors")
            if np.any(target < 0) or np.any(target >= n):
                raise ValueError(f"{name} target indices out of range")
            if np.any(source < 0) or np.any(source >= n):
                raise ValueError(f"{name} source indices out of range")
            if np.any(~np.isfinite(weight)) or np.any(weight <= 0.0):
                raise ValueError(f"{name} weights must be positive and finite")
            if (np.any(~np.isfinite(variance_weight))
                    or np.any(variance_weight <= 0.0)):
                raise ValueError(
                    f"{name} variance weights must be positive and finite")
            if delay.size and int(np.max(delay)) > int(self.max_delay_steps):
                raise ValueError(f"{name} delay exceeds declared maximum")


def _aggregate_delayed_pathway(
        matrices, *, target_cells, source_cells, target_mask,
        n_cells, target_counts, physical_factor):
    delay_out, target_out, source_out, weight_out, variance_out = [], [], [], [], []
    denominator = np.asarray(target_counts, float)
    for delay_step, matrix in enumerate(matrices):
        if delay_step == 0 or matrix.nnz == 0:
            continue
        coo = matrix.tocoo(copy=False)
        rows = np.asarray(coo.row, np.int64)
        selected = np.asarray(target_mask(rows), bool)
        if not np.any(selected):
            continue
        target = target_cells[rows[selected]]
        source = source_cells[np.asarray(coo.col, np.int64)[selected]]
        physical_weight = (np.asarray(coo.data[selected], float)
                           * float(physical_factor))
        flat = target * int(n_cells) + source
        total = np.bincount(
            flat, weights=physical_weight, minlength=int(n_cells) ** 2)
        variance_total = np.bincount(
            flat, weights=physical_weight ** 2,
            minlength=int(n_cells) ** 2)
        occupied = np.flatnonzero(total > 0.0)
        occupied_target = (occupied // int(n_cells)).astype(np.int64)
        delay_out.append(np.full(occupied.size, delay_step, np.int32))
        target_out.append(occupied_target.astype(np.int32))
        source_out.append((occupied % int(n_cells)).astype(np.int32))
        weight_out.append(total[occupied] / denominator[occupied_target])
        variance_out.append(
            variance_total[occupied] / denominator[occupied_target])
    if not delay_out:
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.int32), np.empty(0, float),
                np.empty(0, float))
    return (np.concatenate(delay_out), np.concatenate(target_out),
            np.concatenate(source_out), np.concatenate(weight_out),
            np.concatenate(variance_out))


def build_patient_coarse_delay_operator(substrate, *, n_grid: int = 20):
    """Aggregate all realized recurrent SNN edges without dropping delays."""
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
        raise ValueError("coarse delay grid contains empty population cells")
    ampa = substrate.net["ampa_by_delay"]
    gaba = substrate.net["gaba_by_delay"]
    path = {}
    path["ee"] = _aggregate_delayed_pathway(
        ampa, target_cells=all_cells, source_cells=cell_e,
        target_mask=lambda rows: rows < n_e, n_cells=n_cells,
        target_counts=count_e,
        physical_factor=params.tau_r_AMPA / params.tau_m_E)
    path["ie"] = _aggregate_delayed_pathway(
        ampa, target_cells=all_cells, source_cells=cell_e,
        target_mask=lambda rows: rows >= n_e, n_cells=n_cells,
        target_counts=count_i,
        physical_factor=params.tau_r_AMPA / params.tau_m_I)
    path["ei"] = _aggregate_delayed_pathway(
        gaba, target_cells=all_cells, source_cells=cell_i,
        target_mask=lambda rows: rows < n_e, n_cells=n_cells,
        target_counts=count_e,
        physical_factor=params.tau_r_GABA / params.tau_m_E)
    path["ii"] = _aggregate_delayed_pathway(
        gaba, target_cells=all_cells, source_cells=cell_i,
        target_mask=lambda rows: rows >= n_e, n_cells=n_cells,
        target_counts=count_i,
        physical_factor=params.tau_r_GABA / params.tau_m_I)
    values = {
        "n_grid": int(n_grid), "sheet_l_mm": float(params.L),
        "source_delay_dt_ms": float(params.delay_dt),
        "max_delay_steps": int(substrate.net["max_delay_steps"]),
    }
    for name in PATHWAYS:
        delay, target, source, weight, variance_weight = path[name]
        values.update({
            f"delay_step_{name}": delay,
            f"target_{name}": target,
            f"source_{name}": source,
            f"weight_{name}": weight,
            f"variance_weight_{name}": variance_weight,
        })
    operator = PatientCoarseDelayOperator(**values)
    operator.validate()
    return operator


def save_patient_coarse_delay_operator(path, operator):
    operator.validate()
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_grid": np.asarray(operator.n_grid),
        "sheet_l_mm": np.asarray(operator.sheet_l_mm),
        "source_delay_dt_ms": np.asarray(operator.source_delay_dt_ms),
        "max_delay_steps": np.asarray(operator.max_delay_steps),
    }
    for name in PATHWAYS:
        for field in ("delay_step", "target", "source", "weight",
                      "variance_weight"):
            payload[f"{field}_{name}"] = np.asarray(
                getattr(operator, f"{field}_{name}"))
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem + ".", suffix=".npz")
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": int(path.stat().st_size),
    }


def load_patient_coarse_delay_operator(path):
    with np.load(Path(path).resolve(), allow_pickle=False) as archive:
        scalar_int = {"n_grid", "max_delay_steps"}
        scalar_float = {"sheet_l_mm", "source_delay_dt_ms"}
        values = {}
        for name in scalar_int:
            values[name] = int(np.asarray(archive[name]))
        for name in scalar_float:
            values[name] = float(np.asarray(archive[name]))
        for pathway in PATHWAYS:
            for field in ("delay_step", "target", "source", "weight",
                          "variance_weight"):
                values[f"{field}_{pathway}"] = np.asarray(
                    archive[f"{field}_{pathway}"])
    operator = PatientCoarseDelayOperator(**values)
    operator.validate()
    return operator


def pathway_weight_matrix(operator, name):
    delay, target, source, weight, _variance_weight = operator.pathway(name)
    del delay
    n = operator.n_cells
    return sparse.coo_matrix(
        (weight, (target, source)), shape=(n, n)).tocsr()


def pathway_variance_matrix(operator, name):
    delay, target, source, _weight, variance_weight = operator.pathway(name)
    del delay
    n = operator.n_cells
    return sparse.coo_matrix(
        (variance_weight, (target, source)), shape=(n, n)).tocsr()


def delay_summary(operator, name):
    delay, _target, _source, weight, variance_weight = operator.pathway(name)
    delay_ms = delay.astype(float) * float(operator.source_delay_dt_ms)
    order = np.argsort(delay_ms)
    delay_ms = delay_ms[order]
    weight = weight[order]
    cumulative = np.cumsum(weight) / np.sum(weight)
    quantiles = {}
    for value in (0.05, 0.5, 0.95, 0.99):
        quantiles[str(value)] = float(
            delay_ms[min(np.searchsorted(cumulative, value), delay_ms.size - 1)])
    return {
        "aggregated_entries": int(delay.size),
        "total_weight": float(np.sum(weight)),
        "total_variance_weight": float(np.sum(variance_weight)),
        "minimum_delay_ms": float(np.min(delay_ms)),
        "maximum_delay_ms": float(np.max(delay_ms)),
        "weighted_mean_delay_ms": float(np.average(delay_ms, weights=weight)),
        "weighted_quantiles_ms": quantiles,
    }


def _append(rows, cols, values, row, col, value):
    rows.append(np.asarray(row, np.int64))
    cols.append(np.asarray(col, np.int64))
    values.append(np.asarray(value, float))


def delayed_discrete_linear_map(
        model: PatientCoarseZMModel, operator: PatientCoarseDelayOperator,
        rates, *, q: float, eta_m: float = 0.0,
        tau_m_slow_ms: float = 12.5, history_dt_ms: float = 0.5,
        variance_closure: str = "self_consistent_variance"):
    """Build the explicit-history one-step map at one fixed point.

    The linear decay and piecewise-constant coupling in each rate/synapse/M
    equation are integrated exactly over ``history_dt_ms``.  Realized edge
    delays are rounded only when rebinned onto this history grid.
    """
    operator.validate()
    if model.n_grid != operator.n_grid:
        raise ValueError("mean-field and delay grids differ")
    if history_dt_ms <= 0.0:
        raise ValueError("history_dt_ms must be positive")
    if variance_closure not in {"self_consistent_variance", "frozen_variance"}:
        raise ValueError("unknown variance closure")
    rates = np.asarray(rates, float)
    rate_e, rate_i = np.split(rates, 2)
    mu_e, sigma_e, mu_i, sigma_i = moments(
        model, rate_e, rate_i, q=q, eta_m=eta_m,
        tau_m_slow_ms=tau_m_slow_ms)
    pme, pse, pmi, psi = _transfer_derivatives(
        model, mu_e, sigma_e, mu_i, sigma_i)
    dt = float(history_dt_ms)
    n = model.n_cells
    rebinned = {}
    maximum_lag = 1
    for name in PATHWAYS:
        delay, target, source, weight, variance_weight = operator.pathway(name)
        delay_ms = delay.astype(float) * float(operator.source_delay_dt_ms)
        lag = np.maximum(1, np.rint(delay_ms / dt).astype(np.int32))
        maximum_lag = max(maximum_lag, int(np.max(lag)))
        rebinned[name] = (lag, target, source, weight, variance_weight)
    base_blocks = 7
    base_size = base_blocks * n
    dimension = base_size + 2 * maximum_lag * n
    rows, cols, values = [], [], []
    index = np.arange(n, dtype=np.int64)

    ae = np.exp(-dt / model.tau_mem_e_ms)
    ai = np.exp(-dt / model.tau_mem_i_ms)
    aa = np.exp(-dt / model.tau_ampa_ms)
    ag = np.exp(-dt / model.tau_gaba_ms)
    am = np.exp(-dt / float(tau_m_slow_ms))
    _append(rows, cols, values, index, index, np.full(n, ae))
    _append(rows, cols, values, n + index, n + index, np.full(n, ai))
    _append(rows, cols, values, index, 2 * n + index, (1.0 - ae) * pme)
    _append(rows, cols, values, index, 3 * n + index,
            -(1.0 - ae) * float(q) * pme)
    _append(rows, cols, values, n + index, 4 * n + index, (1.0 - ai) * pmi)
    _append(rows, cols, values, n + index, 5 * n + index,
            -(1.0 - ai) * pmi)
    _append(rows, cols, values, index, 6 * n + index,
            -(1.0 - ae) * float(eta_m) * pme)
    for block, decay in ((2, aa), (3, ag), (4, aa), (5, ag)):
        _append(rows, cols, values, block * n + index, block * n + index,
                np.full(n, decay))
    _append(rows, cols, values, 6 * n + index, 6 * n + index,
            np.full(n, am))
    _append(rows, cols, values, 6 * n + index, index,
            np.full(n, float(tau_m_slow_ms) * (1.0 - am)))

    history_e = base_size
    history_i = base_size + maximum_lag * n
    pathway_spec = {
        "ee": (2 * n, history_e, (1.0 - aa) * model.tau_mem_e_ms),
        "ei": (3 * n, history_i, (1.0 - ag) * model.tau_mem_e_ms),
        "ie": (4 * n, history_e, (1.0 - aa) * model.tau_mem_i_ms),
        "ii": (5 * n, history_i, (1.0 - ag) * model.tau_mem_i_ms),
    }
    for name, (row_offset, history_offset, coefficient) in pathway_spec.items():
        lag, target, source, weight, _variance_weight = rebinned[name]
        _append(
            rows, cols, values,
            row_offset + target,
            history_offset + (lag.astype(np.int64) - 1) * n + source,
            float(coefficient) * weight)

    # In the self-consistent stationary-diffusion closure, delayed source-rate
    # perturbations change sigma through the same squared edge weights used in
    # ``moments``.  This direct history coupling is deliberately unfiltered:
    # it is the minimal closure whose lambda=0 gain equals the fixed-point
    # Jacobian.  The frozen-variance convention remains available as a labelled
    # sensitivity analysis.
    if variance_closure == "self_consistent_variance":
        variance_spec = {
            "ee": (0, history_e, (1.0 - ae) * pse
                   * model.tau_mem_e_ms / (2.0 * sigma_e)),
            "ei": (0, history_i, (1.0 - ae) * pse
                   * model.tau_mem_e_ms * float(q) ** 2 / (2.0 * sigma_e)),
            "ie": (n, history_e, (1.0 - ai) * psi
                   * model.tau_mem_i_ms / (2.0 * sigma_i)),
            "ii": (n, history_i, (1.0 - ai) * psi
                   * model.tau_mem_i_ms / (2.0 * sigma_i)),
        }
        for name, (row_offset, history_offset, coefficient) in variance_spec.items():
            lag, target, source, _weight, variance_weight = rebinned[name]
            _append(
                rows, cols, values,
                row_offset + target,
                history_offset + (lag.astype(np.int64) - 1) * n + source,
                coefficient[target] * variance_weight)

    # Exact discrete shift register: h_1(t+dt)=r(t), h_k(t+dt)=h_{k-1}(t).
    _append(rows, cols, values, history_e + index, index, np.ones(n))
    _append(rows, cols, values, history_i + index, n + index, np.ones(n))
    if maximum_lag > 1:
        shift = np.arange((maximum_lag - 1) * n, dtype=np.int64)
        _append(rows, cols, values, history_e + n + shift,
                history_e + shift, np.ones(shift.size))
        _append(rows, cols, values, history_i + n + shift,
                history_i + shift, np.ones(shift.size))
    matrix = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dimension, dimension)).tocsr()
    matrix.sum_duplicates()
    metadata = {
        "history_dt_ms": dt,
        "maximum_lag_steps": int(maximum_lag),
        "maximum_rebinned_delay_ms": float(maximum_lag * dt),
        "dimension": int(dimension),
        "nonzero": int(matrix.nnz),
        "variance_closure": variance_closure,
        "convention": (
            "explicit_rate_history_shift_register_with_exact_linear_decay_"
            + variance_closure),
    }
    return matrix, metadata


def stationary_delay_mode_vector(
        model: PatientCoarseZMModel, operator: PatientCoarseDelayOperator,
        rate_direction, *, tau_m_slow_ms: float, history_dt_ms: float):
    """Lift a rate direction into the lambda=0 delay-state subspace.

    Every history slot contains the same perturbation, synaptic blocks take
    their stationary gains and M follows ``delta M=tau_M delta r_E``.  Thus a
    fixed-point null vector must become a unit-multiplier vector for the
    self-consistent delay map, independent of the realized delay distribution.
    """
    rate_direction = np.asarray(rate_direction, float)
    n = model.n_cells
    if rate_direction.shape != (2 * n,):
        raise ValueError("rate direction must concatenate E then I cells")
    if history_dt_ms <= 0.0 or tau_m_slow_ms <= 0.0:
        raise ValueError("time constants must be positive")
    maximum_lag = 1
    for name in PATHWAYS:
        delay, _target, _source, _weight, _variance_weight = operator.pathway(name)
        delay_ms = delay.astype(float) * float(operator.source_delay_dt_ms)
        maximum_lag = max(
            maximum_lag,
            int(np.max(np.maximum(
                1, np.rint(delay_ms / float(history_dt_ms)).astype(np.int32)))))
    rate_e, rate_i = np.split(rate_direction, 2)
    base_size = 7 * n
    vector = np.zeros(base_size + 2 * maximum_lag * n, float)
    vector[:n] = rate_e
    vector[n:2 * n] = rate_i
    vector[2 * n:3 * n] = model.tau_mem_e_ms * (model.w_ee @ rate_e)
    vector[3 * n:4 * n] = model.tau_mem_e_ms * (model.w_ei @ rate_i)
    vector[4 * n:5 * n] = model.tau_mem_i_ms * (model.w_ie @ rate_e)
    vector[5 * n:6 * n] = model.tau_mem_i_ms * (model.w_ii @ rate_i)
    vector[6 * n:7 * n] = float(tau_m_slow_ms) * rate_e
    vector[base_size:base_size + maximum_lag * n] = np.tile(
        rate_e, maximum_lag)
    vector[base_size + maximum_lag * n:] = np.tile(rate_i, maximum_lag)
    return vector


def dominant_delay_modes(matrix, *, history_dt_ms: float, k: int = 12,
                         tolerance: float = 1e-7, max_iterations: int = 20000):
    """Return dominant discrete multipliers and continuous-time exponents."""
    if k < 1 or k >= matrix.shape[0] - 1:
        raise ValueError("invalid eigenvalue count")
    multipliers = eigs(
        matrix, k=int(k), which="LM", return_eigenvectors=False,
        tol=float(tolerance), maxiter=int(max_iterations),
        ncv=min(matrix.shape[0] - 1, max(2 * int(k) + 8, 32)))
    exponents = np.log(multipliers.astype(complex)) / float(history_dt_ms)
    order = np.argsort(exponents.real)[::-1]
    return multipliers[order], exponents[order]
