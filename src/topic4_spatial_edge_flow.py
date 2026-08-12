"""Observation-invariant continuous spatial edge flow for Topic 4 rev10-R2."""
from __future__ import annotations

import copy
import hashlib

import numpy as np

from src.topic4_core_connectivity import (
    _hash_sparse_bins,
    _invalidate_ampa_caches,
    incoming_ee_weight,
)


SCALAR_BASIS_NAMES = (
    "constant", "mid_x", "mid_y", "mid_x_mid_y", "P2_mid_x", "P2_mid_y",
)
FEATURE_NAMES = tuple(
    f"flow_{axis}_{basis}"
    for axis in ("x", "y") for basis in SCALAR_BASIS_NAMES
)


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def quadratic_sheet_basis(xy, *, L=20.0):
    """Complete degree-two polynomial basis on fixed physical sheet coordinates."""
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("sheet coordinates must have shape (n, 2)")
    if not np.isfinite(xy).all() or float(L) <= 0.0:
        raise ValueError("sheet coordinates and L must be finite")
    normalized = 2.0 * xy / float(L) - 1.0
    x, y = normalized[:, 0], normalized[:, 1]
    return np.column_stack((
        np.ones(len(x)), x, y, x * y,
        0.5 * (3.0 * x ** 2 - 1.0),
        0.5 * (3.0 * y ** 2 - 1.0),
    ))


def spatial_vector_edge_features(target_xy, source_xy, *, L=20.0,
                                 length_scale=1.0):
    """Evaluate one smooth directed-vector feature row per existing edge."""
    target = np.asarray(target_xy, float)
    source = np.asarray(source_xy, float)
    if target.shape != source.shape or target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("target/source coordinates must align as (edges, 2)")
    length_scale = float(length_scale)
    if length_scale <= 0.0 or not np.isfinite(length_scale):
        raise ValueError("length_scale must be finite and positive")
    midpoint_basis = quadratic_sheet_basis(
        0.5 * (target + source), L=L,
    )
    displacement = (source - target) / length_scale
    return np.concatenate((
        midpoint_basis * displacement[:, 0, None],
        midpoint_basis * displacement[:, 1, None],
    ), axis=1)


def spatial_vector_field(xy, coefficients, *, L=20.0):
    """Evaluate the two-dimensional flow vector represented by coefficients."""
    coefficients = np.asarray(coefficients, float)
    if coefficients.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"coefficients must have shape ({len(FEATURE_NAMES)},)")
    basis = quadratic_sheet_basis(xy, L=L)
    shaped = coefficients.reshape(2, len(SCALAR_BASIS_NAMES))
    return np.column_stack((basis @ shaped[0], basis @ shaped[1]))


def spatial_vector_edge_logits(target_xy, source_xy, coefficients, *,
                               L=20.0, length_scale=1.0):
    coefficients = np.asarray(coefficients, float)
    if coefficients.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"coefficients must have shape ({len(FEATURE_NAMES)},)")
    if not np.isfinite(coefficients).all():
        raise ValueError("coefficients must be finite")
    return spatial_vector_edge_features(
        target_xy, source_xy, L=L, length_scale=length_scale,
    ) @ coefficients


def sample_spatial_edge_features(ampa_by_delay, positions_e, *, L=20.0,
                                 length_scale=1.0, sample_limit=250_000):
    """Stream exact feature moments/maxima and a deterministic edge sample."""
    positions = np.asarray(positions_e, float)
    n_e, n_features = len(positions), len(FEATURE_NAMES)
    total = int(sum(matrix[:n_e, :].nnz for matrix in ampa_by_delay))
    if total <= 0:
        raise ValueError("network contains no E-to-E edges")
    stride = max(1, int(np.ceil(total / max(1, int(sample_limit)))))
    sampled, feature_max = [], np.zeros(n_features, float)
    feature_sum, feature_gram = np.zeros(n_features), np.zeros((n_features, n_features))
    offset = 0
    for matrix in ampa_by_delay:
        coo = matrix.tocoo(copy=False)
        ee = coo.row < n_e
        rows = np.asarray(coo.row[ee], np.int64)
        columns = np.asarray(coo.col[ee], np.int64)
        if not len(rows):
            continue
        features = spatial_vector_edge_features(
            positions[rows], positions[columns], L=L,
            length_scale=length_scale,
        )
        local = np.flatnonzero(
            (offset + np.arange(len(rows), dtype=np.int64)) % stride == 0
        )
        if len(local):
            sampled.append(features[local])
        offset += len(rows)
        feature_max = np.maximum(feature_max, np.max(np.abs(features), axis=0))
        feature_sum += features.sum(axis=0)
        feature_gram += features.T @ features
    return {
        "features": np.concatenate(sampled, axis=0),
        "feature_abs_max": feature_max,
        "feature_sum": feature_sum,
        "feature_gram": feature_gram,
        "n_ee_delay_entries": total,
        "sample_stride": stride,
    }


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


def spatial_vector_ee_flow(net, positions_e, coefficients, *, L=20.0,
                           length_scale=1.0, ratio_sample_limit=1_000_000):
    """Redistribute fixed E-to-E edges with one continuous directed flow field."""
    n_e = int(net["NE"])
    positions = np.asarray(positions_e, float)
    coefficients = np.asarray(coefficients, float)
    if positions.shape != (n_e, 2):
        raise ValueError(f"positions_e must have shape ({n_e}, 2)")
    if coefficients.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"coefficients must have shape ({len(FEATURE_NAMES)},)")
    old_bins = net["ampa_by_delay"]
    old_topology = _hash_sparse_bins(old_bins, include_data=False)
    old_ei = _hash_sparse_bins(old_bins, rows=slice(n_e, None))
    old_gaba = _hash_sparse_bins(net["gaba_by_delay"])
    old_data_hash = _hash_sparse_bins(old_bins)
    old_incoming = incoming_ee_weight(old_bins, n_e)
    positive_targets = old_incoming > 0.0
    audit_base = {
        "mechanism": "continuous_quadratic_midpoint_vector_flow_v1",
        "coefficients": coefficients.tolist(),
        "coefficients_sha256": array_sha256(coefficients),
        "feature_names": FEATURE_NAMES,
        "sheet_L_mm": float(L),
        "displacement_length_scale_mm": float(length_scale),
    }
    if np.all(coefficients == 0.0):
        new_net = copy.copy(net)
        new_net["ampa_by_delay"] = [matrix.copy() for matrix in old_bins]
        removed = _invalidate_ampa_caches(new_net)
        return new_net, {
            **audit_base, "exact_noop": True,
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

    def edge_logits(rows, columns):
        return spatial_vector_edge_logits(
            positions[rows], positions[columns], coefficients,
            L=L, length_scale=length_scale,
        )

    target_max = np.full(n_e, -np.inf, float)
    total_edges = 0
    for matrix in old_bins:
        coo = matrix.tocoo(copy=False)
        ee = coo.row < n_e
        rows = np.asarray(coo.row[ee], np.int64)
        columns = np.asarray(coo.col[ee], np.int64)
        data = np.asarray(coo.data[ee], np.float64)
        if not len(data):
            continue
        if np.any(~np.isfinite(data)) or np.any(data <= 0.0):
            raise ValueError("stored E-to-E weights must be finite and positive")
        np.maximum.at(target_max, rows, np.log(data) + edge_logits(rows, columns))
        total_edges += len(data)
    target_sum = np.zeros(n_e, float)
    for matrix in old_bins:
        coo = matrix.tocoo(copy=False)
        ee = coo.row < n_e
        rows = np.asarray(coo.row[ee], np.int64)
        columns = np.asarray(coo.col[ee], np.int64)
        data = np.asarray(coo.data[ee], np.float64)
        if not len(data):
            continue
        logits = np.log(data) + edge_logits(rows, columns)
        target_sum += np.bincount(
            rows, weights=np.exp(logits - target_max[rows]), minlength=n_e,
        )

    stride = max(1, int(np.ceil(total_edges / max(1, int(ratio_sample_limit)))))
    ratio_samples = []
    ratio_min, ratio_max = np.inf, 0.0
    kl = np.zeros(n_e, float)
    old_concentration, new_concentration = np.zeros(n_e), np.zeros(n_e)
    old_outgoing, new_outgoing = np.zeros(n_e), np.zeros(n_e)
    old_delay_mass, new_delay_mass = np.zeros(n_e), np.zeros(n_e)
    new_bins, global_offset = [], 0
    for delay, matrix in enumerate(old_bins):
        coo = matrix.tocoo(copy=True)
        ee = coo.row < n_e
        rows = np.asarray(coo.row[ee], np.int64)
        columns = np.asarray(coo.col[ee], np.int64)
        data = np.asarray(coo.data[ee], np.float64)
        if not len(data):
            new_bins.append(coo.tocsc())
            continue
        logits = np.log(data) + edge_logits(rows, columns)
        new_data = old_incoming[rows] * np.exp(
            logits - target_max[rows],
        ) / target_sum[rows]
        if np.any(~np.isfinite(new_data)) or np.any(new_data <= 0.0):
            raise RuntimeError("spatial edge flow produced invalid E-to-E weights")
        ratio = new_data / data
        ratio_min = min(ratio_min, float(np.min(ratio)))
        ratio_max = max(ratio_max, float(np.max(ratio)))
        local = np.flatnonzero(
            (global_offset + np.arange(len(ratio), dtype=np.int64)) % stride == 0
        )
        if len(local):
            ratio_samples.append(ratio[local])
        global_offset += len(ratio)
        p_old, p_new = data / old_incoming[rows], new_data / old_incoming[rows]
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
        raise RuntimeError("spatial edge flow violated structural conservation")
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
    delay_delta = new_delay - old_delay
    finite_delay = delay_delta[np.isfinite(delay_delta)]
    return new_net, {
        **audit_base, "exact_noop": False,
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
        },
    }
