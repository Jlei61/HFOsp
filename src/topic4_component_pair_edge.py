"""Target-normalized four-parameter component-pair E-to-E mapper."""
from __future__ import annotations

import copy

import numpy as np

from src.topic4_core_connectivity import (
    _group_summary,
    _hash_sparse_bins,
    _invalidate_ampa_caches,
    incoming_ee_weight,
)


ETA_NAMES = ("eta11", "eta22", "eta1_from_2", "eta2_from_1")


def normalized_component_responsibilities(contributions):
    """Normalize raw Gaussian contributions without inflating zero-mass tails."""
    values = np.asarray(contributions, float)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("contributions must have shape (E neuron, at least 3)")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("component contributions must be finite and non-negative")
    total = values.sum(axis=1, keepdims=True)
    return np.divide(values, total, out=np.zeros_like(values), where=total > 0.0)


def eta_matrix(eta):
    """Rows are target components; columns are source components."""
    values = np.asarray(eta, float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("eta must contain four finite values")
    eta11, eta22, eta1_from_2, eta2_from_1 = values
    return np.asarray([
        [eta11, eta1_from_2],
        [eta2_from_1, eta22],
    ], float)


def component_pair_normalized_ee(net, responsibilities, eta):
    """Redistribute E-to-E weights by component-to-component pair identity.

    The transform preserves topology, delays, E-to-I/GABA weights, and every
    E target's total incoming recurrent-E budget. Component 3 has no fitted
    coefficient and therefore remains the frozen background/negative control.
    """
    n_e = int(net["NE"])
    responsibilities = np.asarray(responsibilities, float)
    if responsibilities.ndim != 2 or responsibilities.shape[0] != n_e \
            or responsibilities.shape[1] < 3:
        raise ValueError(f"responsibilities must have shape ({n_e}, at least 3)")
    if (not np.isfinite(responsibilities).all()
            or np.any((responsibilities < 0.0) | (responsibilities > 1.0))):
        raise ValueError("responsibilities must be finite and lie in [0, 1]")
    matrix = eta_matrix(eta)

    old_bins = net["ampa_by_delay"]
    old_topology_hash = _hash_sparse_bins(old_bins, include_data=False)
    old_e_to_i_hash = _hash_sparse_bins(old_bins, rows=slice(n_e, None))
    old_gaba_hash = _hash_sparse_bins(net["gaba_by_delay"])
    old_incoming = incoming_ee_weight(old_bins, n_e)
    positive_targets = old_incoming > 0.0

    if np.all(matrix == 0.0):
        new_net = copy.copy(net)
        new_net["ampa_by_delay"] = [value.copy() for value in old_bins]
        removed = _invalidate_ampa_caches(new_net)
        return new_net, {
            "mechanism": "component_pair_target_normalized_EE_exp_v1",
            "eta_names": list(ETA_NAMES), "eta": np.asarray(eta, float).tolist(),
            "eta_matrix_target_by_source": matrix.tolist(),
            "exact_noop": True, "n_E": n_e,
            "max_abs_incoming_E_error": 0.0,
            "topology_unchanged": True, "e_to_i_unchanged": True,
            "gaba_unchanged": True,
            "invalidated_ampa_cache_keys": removed,
            "edge_ratio": {"min": 1.0, "p01": 1.0, "median": 1.0,
                           "p99": 1.0, "max": 1.0},
        }

    records, all_rows, all_cols, all_data = [], [], [], []
    for old in old_bins:
        coo = old.tocoo(copy=True)
        is_ee = coo.row < n_e
        data = np.asarray(coo.data[is_ee], np.float64)
        if np.any(~np.isfinite(data)) or np.any(data <= 0.0):
            raise ValueError("stored E-to-E weights must be finite and positive")
        records.append((coo, is_ee, len(data)))
        all_rows.append(np.asarray(coo.row[is_ee], np.int64))
        all_cols.append(np.asarray(coo.col[is_ee], np.int64))
        all_data.append(data)
    rows = np.concatenate(all_rows) if all_rows else np.empty(0, np.int64)
    cols = np.concatenate(all_cols) if all_cols else np.empty(0, np.int64)
    data = np.concatenate(all_data) if all_data else np.empty(0, np.float64)
    if not len(data):
        raise ValueError("network contains no E-to-E edges")

    incoming = np.bincount(rows, weights=data, minlength=n_e)
    target_r = responsibilities[rows, :2]
    source_r = responsibilities[cols, :2]
    log_multiplier = np.einsum(
        "ni,ij,nj->n", target_r, matrix, source_r, optimize=True)
    log_unnormalized = np.log(data) + log_multiplier
    target_max = np.full(n_e, -np.inf)
    np.maximum.at(target_max, rows, log_unnormalized)
    exp_shifted = np.exp(log_unnormalized - target_max[rows])
    target_sum = np.bincount(rows, weights=exp_shifted, minlength=n_e)
    new_data = incoming[rows] * exp_shifted / target_sum[rows]
    if np.any(~np.isfinite(new_data)) or np.any(new_data <= 0.0):
        raise RuntimeError("component-pair transform produced invalid E-to-E weights")

    new_bins, offset = [], 0
    for coo, is_ee, count in records:
        coo.data[is_ee] = new_data[offset:offset + count]
        offset += count
        new_bins.append(coo.tocsc())
    new_net = copy.copy(net)
    new_net["ampa_by_delay"] = new_bins
    removed = _invalidate_ampa_caches(new_net)
    new_incoming = incoming_ee_weight(new_bins, n_e)
    abs_error = np.abs(new_incoming - old_incoming)
    topology = _hash_sparse_bins(new_bins, include_data=False) == old_topology_hash
    e_to_i = (_hash_sparse_bins(new_bins, rows=slice(n_e, None)) == old_e_to_i_hash)
    gaba = _hash_sparse_bins(new_net["gaba_by_delay"]) == old_gaba_hash
    if (abs_error.max(initial=0.0) > 1e-9 or not topology
            or not e_to_i or not gaba):
        raise RuntimeError("component-pair transform violated structural conservation")

    ratio = new_data / data
    p_old = data / incoming[rows]
    p_new = new_data / incoming[rows]
    kl = np.bincount(rows, weights=p_new * np.log(p_new / p_old), minlength=n_e)
    old_concentration = np.bincount(rows, weights=p_old ** 2, minlength=n_e)
    new_concentration = np.bincount(rows, weights=p_new ** 2, minlength=n_e)
    ess_ratio = np.divide(
        old_concentration, new_concentration,
        out=np.full(n_e, np.nan), where=new_concentration > 0.0)
    dominant = np.argmax(responsibilities[:, :3], axis=1)
    groups = {
        "all_nonzero_incoming": _group_summary(
            kl, ess_ratio, positive_targets),
        "component_1_dominant": _group_summary(
            kl, ess_ratio, positive_targets & (dominant == 0)),
        "component_2_dominant": _group_summary(
            kl, ess_ratio, positive_targets & (dominant == 1)),
        "component_3_dominant": _group_summary(
            kl, ess_ratio, positive_targets & (dominant == 2)),
    }
    return new_net, {
        "mechanism": "component_pair_target_normalized_EE_exp_v1",
        "eta_names": list(ETA_NAMES), "eta": np.asarray(eta, float).tolist(),
        "eta_matrix_target_by_source": matrix.tolist(),
        "component_3_parameterized": False,
        "exact_noop": False, "n_E": n_e, "n_ee_edges": int(len(data)),
        "max_abs_incoming_E_error": float(abs_error.max(initial=0.0)),
        "mean_abs_incoming_E_error": float(abs_error.mean()),
        "topology_unchanged": bool(topology),
        "e_to_i_unchanged": bool(e_to_i), "gaba_unchanged": bool(gaba),
        "invalidated_ampa_cache_keys": removed,
        "edge_ratio": {
            "min": float(ratio.min()), "p01": float(np.percentile(ratio, 1)),
            "median": float(np.median(ratio)), "p99": float(np.percentile(ratio, 99)),
            "max": float(ratio.max()),
        },
        "target_groups": groups,
    }
