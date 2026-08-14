"""Coarse weighted E->I->E geometry for the FCXR-LC6A graph family."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from src.topic4_fcxr_lc6_surround import EToIGraph, axis_unit, projected_offsets


@dataclass(frozen=True)
class SpatialBins:
    cell_bin: np.ndarray
    centers: np.ndarray
    n_bins_axis: int
    sheet_size_mm: float


def spatial_bins(positions, *, sheet_size_mm: float, n_bins_axis: int) -> SpatialBins:
    if n_bins_axis < 2 or sheet_size_mm <= 0.0:
        raise ValueError("spatial binning needs positive sheet size and at least two bins")
    positions = np.asarray(positions, float)
    scaled = np.floor(positions / float(sheet_size_mm) * int(n_bins_axis)).astype(int)
    scaled = np.clip(scaled, 0, int(n_bins_axis) - 1)
    cell_bin = scaled[:, 0] * int(n_bins_axis) + scaled[:, 1]
    width = float(sheet_size_mm) / int(n_bins_axis)
    coordinate = (np.arange(int(n_bins_axis)) + .5) * width
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="ij")
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    return SpatialBins(cell_bin.astype(np.int32), centers, int(n_bins_axis), float(sheet_size_mm))


def coarse_two_hop_operator(
    e_to_i: EToIGraph,
    i_to_e: EToIGraph,
    e_bins: SpatialBins,
    *,
    n_e: int,
    n_i: int,
) -> sparse.csr_matrix:
    """Return target-E-bin x source-E-bin inhibitory-magnitude operator."""

    if e_to_i.n_targets != n_i or i_to_e.n_targets != n_e:
        raise ValueError("population sizes do not align to E-to-I and I-to-E tables")
    n_bins = e_bins.n_bins_axis ** 2
    # A: inhibitory target x excitatory source bin.
    a_row = np.repeat(np.arange(n_i, dtype=np.int32), e_to_i.in_degree)
    a_col = e_bins.cell_bin[e_to_i.sources.ravel()]
    a_data = np.abs(e_to_i.weights.ravel()).astype(float)
    a = sparse.csr_matrix((a_data, (a_row, a_col)), shape=(n_i, n_bins))
    # B: excitatory target bin x inhibitory source.
    e_target = np.repeat(np.arange(n_e, dtype=np.int32), i_to_e.in_degree)
    b_row = e_bins.cell_bin[e_target]
    b_col = i_to_e.sources.ravel()
    b_data = np.abs(i_to_e.weights.ravel()).astype(float)
    b = sparse.csr_matrix((b_data, (b_row, b_col)), shape=(n_bins, n_i))
    return (b @ a).tocsr()


def _weighted_moments(values, weights):
    weights = np.asarray(weights, float)
    values = np.asarray(values, float)
    total = float(weights.sum())
    if total <= 0.0:
        return float("nan"), float("nan")
    mean = float(np.dot(weights, values) / total)
    variance = float(np.dot(weights, np.square(values - mean)) / total)
    return mean, float(np.sqrt(max(0.0, variance)))


def summarize_two_hop_operator(
    operator,
    e_bins: SpatialBins,
    axis,
    *,
    ee_sigma_parallel_mm: float,
    ee_sigma_perpendicular_mm: float,
    edge_margin_mm: float,
) -> dict:
    coo = sparse.coo_matrix(operator)
    if coo.nnz == 0:
        raise ValueError("two-hop operator is empty")
    source = e_bins.centers[coo.col]
    target = e_bins.centers[coo.row]
    parallel, perpendicular = projected_offsets(source, target, axis)
    mass = np.abs(coo.data).astype(float)
    mean_parallel, sigma_parallel = _weighted_moments(parallel, mass)
    mean_perpendicular, sigma_perpendicular = _weighted_moments(perpendicular, mass)
    q_parallel = sigma_parallel / float(ee_sigma_parallel_mm)
    center = (
        np.abs(parallel) <= float(ee_sigma_parallel_mm)
    ) & (
        np.abs(perpendicular) <= float(ee_sigma_perpendicular_mm)
    )
    surround = (
        (np.abs(parallel) > float(ee_sigma_parallel_mm))
        & (np.abs(parallel) <= 3.0 * float(ee_sigma_parallel_mm))
        & (np.abs(perpendicular) <= 1.5 * float(ee_sigma_perpendicular_mm))
    )
    forward = mass[parallel > 0].sum()
    backward = mass[parallel < 0].sum()
    source_edge_distance = np.min(
        np.column_stack((source, e_bins.sheet_size_mm - source)), axis=1,
    )
    interior = source_edge_distance >= float(edge_margin_mm)
    _, sigma_parallel_interior = _weighted_moments(parallel[interior], mass[interior])
    _, sigma_parallel_edge = _weighted_moments(parallel[~interior], mass[~interior])
    center_mass = float(mass[center].sum())
    surround_mass = float(mass[surround].sum())
    return {
        "n_nonzero_bin_pairs": int(coo.nnz),
        "total_inhibitory_magnitude": float(mass.sum()),
        "mean_parallel_mm": mean_parallel,
        "mean_perpendicular_mm": mean_perpendicular,
        "sigma_parallel_mm": sigma_parallel,
        "sigma_perpendicular_mm": sigma_perpendicular,
        "q_parallel_two_hop": float(q_parallel),
        "center_mass": center_mass,
        "surround_mass": surround_mass,
        "surround_center_ratio": float(surround_mass / max(center_mass, 1e-30)),
        "forward_backward_mass_ratio": float(forward / max(backward, 1e-30)),
        "interior_sigma_parallel_mm": sigma_parallel_interior,
        "edge_sigma_parallel_mm": sigma_parallel_edge,
        "center_definition": "abs(d_parallel)<=sigma_EE_parallel and abs(d_perp)<=sigma_EE_perp",
        "surround_definition": "sigma_EE_parallel<abs(d_parallel)<=3sigma and abs(d_perp)<=1.5sigma_perp",
    }


def sample_two_hop_latencies(
    e_to_i: EToIGraph,
    i_to_e: EToIGraph,
    *,
    n_e: int,
    n_i: int,
    engine_dt_ms: float,
    n_paths: int,
    audit_seed: int,
) -> dict:
    """Importance-sample actual E->I->E paths for delay quantiles."""

    if n_paths <= 0 or engine_dt_ms <= 0.0:
        raise ValueError("latency audit needs positive path count and engine dt")
    flat_i = i_to_e.sources.ravel().astype(np.int32)
    flat_target_e = np.repeat(np.arange(n_e, dtype=np.int32), i_to_e.in_degree)
    flat_weight = np.abs(i_to_e.weights.ravel()).astype(float)
    flat_delay = i_to_e.delay_steps.ravel().astype(np.int32)
    order = np.argsort(flat_i, kind="stable")
    flat_i = flat_i[order]
    flat_target_e = flat_target_e[order]
    flat_weight = flat_weight[order]
    flat_delay = flat_delay[order]
    counts = np.bincount(flat_i, minlength=n_i)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    incoming_mass = np.abs(e_to_i.weights).sum(axis=1)
    outgoing_mass = np.bincount(flat_i, weights=flat_weight, minlength=n_i)
    i_mass = incoming_mass * outgoing_mass
    if i_mass.sum() <= 0.0:
        raise ValueError("two-hop path mass is zero")
    rng = np.random.default_rng(int(audit_seed))
    sampled_i = rng.choice(n_i, size=int(n_paths), p=i_mass / i_mass.sum())
    latency = np.empty(int(n_paths), dtype=float)
    for index, inhibitory in enumerate(sampled_i):
        e_in_weight = np.abs(e_to_i.weights[inhibitory]).astype(float)
        in_slot = int(rng.choice(e_to_i.in_degree, p=e_in_weight / e_in_weight.sum()))
        start, stop = int(offsets[inhibitory]), int(offsets[inhibitory + 1])
        out_weight = flat_weight[start:stop]
        out_slot = int(rng.choice(stop - start, p=out_weight / out_weight.sum()))
        latency[index] = (
            int(e_to_i.delay_steps[inhibitory, in_slot])
            + int(flat_delay[start + out_slot])
        ) * float(engine_dt_ms)
    return {
        "n_paths": int(n_paths),
        "audit_seed": int(audit_seed),
        "median_ms": float(np.median(latency)),
        "q95_ms": float(np.quantile(latency, .95)),
        "q99_ms": float(np.quantile(latency, .99)),
    }
