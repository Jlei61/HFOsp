"""Directed graph-spectral E-to-E redistribution for Topic 4 rev10-R."""
from __future__ import annotations

import copy
import hashlib

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy import sparse
from scipy.sparse.linalg import svds

from src.topic4_core_connectivity import (
    _hash_sparse_bins,
    _invalidate_ampa_caches,
    incoming_ee_weight,
)


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def summed_ee_operator(ampa_by_delay, n_e):
    """Sum fixed-delay E-to-E bins without densifying the neuron graph."""
    n_e = int(n_e)
    counts = [int(matrix[:n_e, :].nnz) for matrix in ampa_by_delay]
    total = int(sum(counts))
    rows = np.empty(total, np.int32)
    columns = np.empty(total, np.int32)
    data = np.empty(total, np.float64)
    offset = 0
    for matrix, count in zip(ampa_by_delay, counts):
        coo = matrix[:n_e, :].tocoo(copy=False)
        stop = offset + count
        rows[offset:stop] = coo.row
        columns[offset:stop] = coo.col
        data[offset:stop] = coo.data
        offset = stop
    result = sparse.coo_matrix(
        (data, (rows, columns)), shape=(n_e, n_e),
    ).tocsr()
    result.sum_duplicates()
    result.sort_indices()
    if np.any(~np.isfinite(result.data)) or np.any(result.data <= 0.0):
        raise ValueError("E-to-E operator must contain finite positive weights")
    return result


def two_sided_normalized_operator(weight):
    """Return D_in^-1/2 W D_out^-1/2 for row-target W."""
    weight = sparse.csr_matrix(weight, dtype=np.float64)
    incoming = np.asarray(weight.sum(axis=1)).ravel()
    outgoing = np.asarray(weight.sum(axis=0)).ravel()
    inverse_in = np.divide(
        1.0, np.sqrt(incoming), out=np.zeros_like(incoming), where=incoming > 0.0,
    )
    inverse_out = np.divide(
        1.0, np.sqrt(outgoing), out=np.zeros_like(outgoing), where=outgoing > 0.0,
    )
    normalized = weight.multiply(inverse_in[:, None])
    normalized = normalized.multiply(inverse_out[None, :]).tocsr()
    normalized.sort_indices()
    return normalized, incoming, outgoing


def _canonicalize_svd_pairs(left, right_t):
    left = np.asarray(left, float).copy()
    right_t = np.asarray(right_t, float).copy()
    for mode in range(left.shape[1]):
        pivot = int(np.argmax(np.abs(left[:, mode])))
        sign = 1.0 if left[pivot, mode] >= 0.0 else -1.0
        left[:, mode] *= sign
        right_t[mode] *= sign
    return left, right_t


def build_directed_spectral_basis(ampa_by_delay, n_e, *, rank=4,
                                  extra_modes=1, random_state=20260812,
                                  tolerance=1e-8, maxiter=None):
    """Build paired graph coordinates after dropping the degree mode.

    One extra excluded mode is retained in diagnostics so the truncation-boundary
    singular gap is visible.  Coordinates are unit RMS; their paired products
    are unchanged by the joint sign convention.
    """
    rank, extra_modes = int(rank), int(extra_modes)
    if rank < 1 or extra_modes < 1:
        raise ValueError("rank and extra_modes must be positive")
    weight = summed_ee_operator(ampa_by_delay, n_e)
    normalized, incoming, outgoing = two_sided_normalized_operator(weight)
    requested = rank + 1 + extra_modes
    if requested >= min(normalized.shape):
        raise ValueError("graph is too small for the requested spectral rank")
    left, singular, right_t = svds(
        normalized, k=requested, which="LM", tol=float(tolerance),
        maxiter=maxiter, return_singular_vectors=True,
        random_state=int(random_state),
    )
    order = np.argsort(singular)[::-1]
    singular = np.asarray(singular[order], float)
    left, right_t = left[:, order], right_t[order]
    left, right_t = _canonicalize_svd_pairs(left, right_t)
    retained = slice(1, 1 + rank)
    u = np.asarray(left[:, retained], float)
    v = np.asarray(right_t[retained].T, float)
    u *= np.sqrt(len(u))
    v *= np.sqrt(len(v))
    retained_singular = singular[retained]
    relative_gap = np.divide(
        singular[:-1] - singular[1:], singular[:-1],
        out=np.full(len(singular) - 1, np.nan), where=singular[:-1] > 0.0,
    )
    graph_hash = _hash_sparse_bins(ampa_by_delay, rows=slice(0, int(n_e)))
    return {
        "u": u,
        "v": v,
        "singular_values": retained_singular,
        "all_computed_singular_values": singular,
        "leading_degree_singular_value": float(singular[0]),
        "retained_relative_gaps": relative_gap[1:rank].copy(),
        "truncation_boundary_relative_gap": float(relative_gap[rank]),
        "rank": rank,
        "n_e": int(n_e),
        "n_edges_summed": int(weight.nnz),
        "n_delay_entries": int(sum(matrix[:int(n_e), :].nnz
                                   for matrix in ampa_by_delay)),
        "n_zero_incoming": int(np.sum(incoming <= 0.0)),
        "n_zero_outgoing": int(np.sum(outgoing <= 0.0)),
        "graph_weight_sha256": graph_hash,
        "u_sha256": array_sha256(u),
        "v_sha256": array_sha256(v),
        "singular_values_sha256": array_sha256(retained_singular),
    }


def spectral_response_design(singular_values, n_coefficients):
    """Build the shared Chebyshev response design on normalized spectrum."""
    singular = np.asarray(singular_values, float)
    n_coefficients = int(n_coefficients)
    if singular.ndim != 1 or n_coefficients != len(singular):
        raise ValueError("one spectral coefficient is required per retained mode")
    if np.any(~np.isfinite(singular)) or np.any(singular <= 0.0):
        raise ValueError("singular values must be finite and positive")
    coordinate = 2.0 * singular / float(singular[0]) - 1.0
    return chebvander(coordinate, n_coefficients - 1)


def spectral_response_weights(singular_values, coefficients):
    """Evaluate a shared Chebyshev response on graph singular values."""
    coefficients = np.asarray(coefficients, float)
    if np.any(~np.isfinite(coefficients)):
        raise ValueError("spectral coefficients must be finite")
    design = spectral_response_design(singular_values, len(coefficients))
    return design @ coefficients


def spectral_edge_logits(rows, columns, basis, coefficients):
    rows = np.asarray(rows, np.int64)
    columns = np.asarray(columns, np.int64)
    u, v = np.asarray(basis["u"], float), np.asarray(basis["v"], float)
    if rows.shape != columns.shape:
        raise ValueError("edge rows and columns must align")
    if u.shape != v.shape or u.shape[0] != int(basis["n_e"]):
        raise ValueError("left and right graph coordinates do not align")
    response = spectral_response_weights(
        basis["singular_values"], coefficients,
    )
    return np.einsum("ij,j,ij->i", u[rows], response, v[columns])


def sample_spectral_edge_features(ampa_by_delay, basis, *,
                                  sample_limit=250_000):
    """Return a deterministic edge sample in shared Chebyshev coordinates."""
    n_e, rank = int(basis["n_e"]), int(basis["rank"])
    u, v = np.asarray(basis["u"], float), np.asarray(basis["v"], float)
    design = spectral_response_design(basis["singular_values"], rank)
    total = int(sum(matrix[:n_e, :].nnz for matrix in ampa_by_delay))
    stride = max(1, int(np.ceil(total / max(1, int(sample_limit)))))
    sampled, feature_max = [], np.zeros(rank, float)
    offset = 0
    for matrix in ampa_by_delay:
        coo = matrix.tocoo(copy=False)
        ee = coo.row < n_e
        rows = np.asarray(coo.row[ee], np.int64)
        columns = np.asarray(coo.col[ee], np.int64)
        local = np.flatnonzero(
            (offset + np.arange(len(rows), dtype=np.int64)) % stride == 0
        )
        if len(local):
            pair = u[rows[local], :, None] * v[columns[local], :, None]
            sampled.append(np.sum(pair * design[None, :, :], axis=1))
        offset += len(rows)
        pair = u[rows, :, None] * v[columns, :, None]
        features = np.sum(pair * design[None, :, :], axis=1)
        feature_max = np.maximum(feature_max, np.max(np.abs(features), axis=0))
    return {
        "features": (
            np.concatenate(sampled, axis=0)
            if sampled else np.empty((0, rank), float)
        ),
        "feature_abs_max": feature_max,
        "n_ee_delay_entries": total,
        "sample_stride": stride,
    }


def reconstructed_spectral_field(u, v, singular_values, coefficients):
    """Dense helper for small rotation-invariance tests only."""
    response = spectral_response_weights(singular_values, coefficients)
    return (np.asarray(u, float) * response[None, :]) @ np.asarray(v, float).T


def _target_distribution_summary(kl, ess_ratio, positive_targets):
    selected = positive_targets & np.isfinite(kl) & np.isfinite(ess_ratio)
    if not np.any(selected):
        return {
            "n_targets": 0, "kl_median": None, "kl_p99": None,
            "ess_ratio_median": None, "ess_ratio_p01": None,
        }
    return {
        "n_targets": int(np.sum(selected)),
        "kl_median": float(np.median(kl[selected])),
        "kl_p99": float(np.percentile(kl[selected], 99)),
        "ess_ratio_median": float(np.median(ess_ratio[selected])),
        "ess_ratio_p01": float(np.percentile(ess_ratio[selected], 1)),
    }


def graph_spectral_ee_flow(net, basis, coefficients, *,
                           ratio_sample_limit=1_000_000):
    """Redistribute fixed E-to-E edges under exact target incoming budgets."""
    n_e = int(net["NE"])
    coefficients = np.asarray(coefficients, float)
    if int(basis["n_e"]) != n_e:
        raise ValueError("graph basis and network size differ")
    if coefficients.shape != (int(basis["rank"]),):
        raise ValueError("spectral coefficient count differs from graph rank")
    if basis.get("graph_weight_sha256") != _hash_sparse_bins(
            net["ampa_by_delay"], rows=slice(0, n_e)):
        raise ValueError("graph basis was built from another E-to-E graph")

    old_bins = net["ampa_by_delay"]
    old_topology = _hash_sparse_bins(old_bins, include_data=False)
    old_ei = _hash_sparse_bins(old_bins, rows=slice(n_e, None))
    old_gaba = _hash_sparse_bins(net["gaba_by_delay"])
    old_data_hash = _hash_sparse_bins(old_bins)
    old_incoming = incoming_ee_weight(old_bins, n_e)
    positive_targets = old_incoming > 0.0
    response = spectral_response_weights(basis["singular_values"], coefficients)

    if np.all(coefficients == 0.0):
        new_net = copy.copy(net)
        new_net["ampa_by_delay"] = [matrix.copy() for matrix in old_bins]
        removed = _invalidate_ampa_caches(new_net)
        return new_net, {
            "mechanism": "directed_graph_spectral_chebyshev_v1",
            "exact_noop": True, "coefficients": coefficients.tolist(),
            "spectral_response": response.tolist(),
            "max_abs_incoming_E_error": 0.0,
            "topology_unchanged": True, "delay_assignment_unchanged": True,
            "e_to_i_unchanged": True, "gaba_unchanged": True,
            "ampa_data_unchanged": True,
            "invalidated_ampa_cache_keys": removed,
            "edge_ratio": {"min": 1.0, "p01": 1.0, "median": 1.0,
                           "p99": 1.0, "max": 1.0,
                           "quantile_sample_size": 0, "sample_stride": 1},
            "target_distribution": _target_distribution_summary(
                np.zeros(n_e), np.ones(n_e), positive_targets,
            ),
        }

    target_max = np.full(n_e, -np.inf, float)
    total_edges = 0
    for matrix in old_bins:
        coo = matrix.tocoo(copy=False)
        ee = coo.row < n_e
        rows = np.asarray(coo.row[ee], np.int64)
        columns = np.asarray(coo.col[ee], np.int64)
        data = np.asarray(coo.data[ee], np.float64)
        if np.any(~np.isfinite(data)) or np.any(data <= 0.0):
            raise ValueError("stored E-to-E weights must be finite and positive")
        logits = np.log(data) + spectral_edge_logits(
            rows, columns, basis, coefficients,
        )
        np.maximum.at(target_max, rows, logits)
        total_edges += len(data)
    target_sum = np.zeros(n_e, float)
    for matrix in old_bins:
        coo = matrix.tocoo(copy=False)
        ee = coo.row < n_e
        rows = np.asarray(coo.row[ee], np.int64)
        columns = np.asarray(coo.col[ee], np.int64)
        data = np.asarray(coo.data[ee], np.float64)
        logits = np.log(data) + spectral_edge_logits(
            rows, columns, basis, coefficients,
        )
        target_sum += np.bincount(
            rows, weights=np.exp(logits - target_max[rows]), minlength=n_e,
        )

    stride = max(1, int(np.ceil(total_edges / max(1, int(ratio_sample_limit)))))
    ratio_samples = []
    ratio_min, ratio_max = np.inf, 0.0
    kl = np.zeros(n_e, float)
    old_concentration = np.zeros(n_e, float)
    new_concentration = np.zeros(n_e, float)
    old_outgoing = np.zeros(n_e, float)
    new_outgoing = np.zeros(n_e, float)
    old_delay_mass = np.zeros(n_e, float)
    new_delay_mass = np.zeros(n_e, float)
    new_bins = []
    global_offset = 0
    for delay, matrix in enumerate(old_bins):
        coo = matrix.tocoo(copy=True)
        ee = coo.row < n_e
        rows = np.asarray(coo.row[ee], np.int64)
        columns = np.asarray(coo.col[ee], np.int64)
        data = np.asarray(coo.data[ee], np.float64)
        logits = np.log(data) + spectral_edge_logits(
            rows, columns, basis, coefficients,
        )
        new_data = old_incoming[rows] * np.exp(
            logits - target_max[rows],
        ) / target_sum[rows]
        if np.any(~np.isfinite(new_data)) or np.any(new_data <= 0.0):
            raise RuntimeError("edge flow produced invalid E-to-E weights")
        ratio = new_data / data
        ratio_min = min(ratio_min, float(np.min(ratio, initial=np.inf)))
        ratio_max = max(ratio_max, float(np.max(ratio, initial=0.0)))
        local = np.flatnonzero(
            (global_offset + np.arange(len(ratio), dtype=np.int64)) % stride == 0
        )
        if len(local):
            ratio_samples.append(ratio[local])
        global_offset += len(ratio)
        p_old = data / old_incoming[rows]
        p_new = new_data / old_incoming[rows]
        kl += np.bincount(
            rows, weights=p_new * np.log(p_new / p_old), minlength=n_e,
        )
        old_concentration += np.bincount(
            rows, weights=p_old ** 2, minlength=n_e,
        )
        new_concentration += np.bincount(
            rows, weights=p_new ** 2, minlength=n_e,
        )
        old_outgoing += np.bincount(columns, weights=data, minlength=n_e)
        new_outgoing += np.bincount(columns, weights=new_data, minlength=n_e)
        old_delay_mass += float(delay) * np.bincount(
            rows, weights=data, minlength=n_e,
        )
        new_delay_mass += float(delay) * np.bincount(
            rows, weights=new_data, minlength=n_e,
        )
        coo.data[ee] = new_data
        new_bins.append(coo.tocsc())

    new_net = copy.copy(net)
    new_net["ampa_by_delay"] = new_bins
    removed = _invalidate_ampa_caches(new_net)
    new_incoming = incoming_ee_weight(new_bins, n_e)
    incoming_error = np.abs(new_incoming - old_incoming)
    topology_unchanged = _hash_sparse_bins(new_bins, include_data=False) == old_topology
    e_to_i_unchanged = _hash_sparse_bins(
        new_bins, rows=slice(n_e, None),
    ) == old_ei
    gaba_unchanged = _hash_sparse_bins(new_net["gaba_by_delay"]) == old_gaba
    if (float(np.max(incoming_error, initial=0.0)) > 1e-9
            or not topology_unchanged or not e_to_i_unchanged or not gaba_unchanged):
        raise RuntimeError("edge flow violated the structural conservation contract")
    ess_old = np.divide(
        1.0, old_concentration, out=np.full(n_e, np.nan),
        where=old_concentration > 0.0,
    )
    ess_new = np.divide(
        1.0, new_concentration, out=np.full(n_e, np.nan),
        where=new_concentration > 0.0,
    )
    ess_ratio = np.divide(
        ess_new, ess_old, out=np.full(n_e, np.nan),
        where=np.isfinite(ess_old) & (ess_old > 0.0),
    )
    outgoing_ratio = np.divide(
        new_outgoing, old_outgoing, out=np.full(n_e, np.nan),
        where=old_outgoing > 0.0,
    )
    old_delay = np.divide(
        old_delay_mass, old_incoming, out=np.full(n_e, np.nan),
        where=old_incoming > 0.0,
    )
    new_delay = np.divide(
        new_delay_mass, new_incoming, out=np.full(n_e, np.nan),
        where=new_incoming > 0.0,
    )
    sampled = np.concatenate(ratio_samples) if ratio_samples else np.ones(1)
    finite_outgoing = outgoing_ratio[np.isfinite(outgoing_ratio)]
    finite_delay = (new_delay - old_delay)[np.isfinite(new_delay - old_delay)]
    return new_net, {
        "mechanism": "directed_graph_spectral_chebyshev_v1",
        "exact_noop": False, "coefficients": coefficients.tolist(),
        "spectral_response": response.tolist(),
        "n_ee_delay_entries": int(total_edges),
        "max_abs_incoming_E_error": float(np.max(incoming_error, initial=0.0)),
        "mean_abs_incoming_E_error": float(np.mean(incoming_error)),
        "topology_unchanged": bool(topology_unchanged),
        "delay_assignment_unchanged": bool(topology_unchanged),
        "e_to_i_unchanged": bool(e_to_i_unchanged),
        "gaba_unchanged": bool(gaba_unchanged),
        "ampa_data_unchanged": _hash_sparse_bins(new_bins) == old_data_hash,
        "invalidated_ampa_cache_keys": removed,
        "edge_ratio": {
            "min": ratio_min, "p01": float(np.percentile(sampled, 1)),
            "median": float(np.median(sampled)),
            "p99": float(np.percentile(sampled, 99)), "max": ratio_max,
            "quantile_sample_size": int(len(sampled)), "sample_stride": stride,
        },
        "target_distribution": _target_distribution_summary(
            kl, ess_ratio, positive_targets,
        ),
        "source_outgoing_ratio": {
            "p01": float(np.percentile(finite_outgoing, 1)),
            "median": float(np.median(finite_outgoing)),
            "p99": float(np.percentile(finite_outgoing, 99)),
        },
        "effective_delay_bin_delta": {
            "p01": float(np.percentile(finite_delay, 1)),
            "median": float(np.median(finite_delay)),
            "p99": float(np.percentile(finite_delay, 99)),
            "max_abs": float(np.max(np.abs(finite_delay), initial=0.0)),
        },
    }


__all__ = [
    "array_sha256", "build_directed_spectral_basis", "graph_spectral_ee_flow",
    "reconstructed_spectral_field", "spectral_edge_logits",
    "spectral_response_weights", "sample_spectral_edge_features",
    "summed_ee_operator",
    "two_sided_normalized_operator",
]
