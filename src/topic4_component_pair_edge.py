"""Conserved six-parameter residual component-pair E-to-E mapper."""
from __future__ import annotations

import copy

import numpy as np

from src.topic4_core_connectivity import (
    _group_summary,
    _hash_sparse_bins,
    _invalidate_ampa_caches,
    field_normalized_ee_pair,
    incoming_ee_weight,
)
from src.topic4_rev9_edge_structure import field_background_membership


GAMMA_NAMES = (
    "gamma_c1_from_c1",
    "gamma_c1_from_c2",
    "gamma_c2_from_c1",
    "gamma_c2_from_c2",
    "gamma_bg_from_c1",
    "gamma_bg_from_c2",
)
TARGET_GROUPS = ("component_1", "component_2", "background")
SOURCE_GROUPS = ("component_1", "component_2")


def component_background_membership(h, contributions):
    """Return C1/C2/C3/background memberships without inflating field tails."""
    result = field_background_membership(h, contributions)
    membership = np.asarray(result["membership"], float)
    if membership.shape[1] != 4:
        raise ValueError("the rev9-L oracle requires exactly three field components")
    return membership


def gamma_matrix(gamma):
    """Rows are C1/C2/background targets; columns are C1/C2 sources."""
    values = np.asarray(gamma, float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("gamma must contain six finite values")
    return values.reshape(3, 2)


def _ee_records(bins, n_e):
    records, all_rows, all_cols, all_data = [], [], [], []
    for matrix in bins:
        coo = matrix.tocoo(copy=True)
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
    return records, rows, cols, data


def _ratio_summary(values):
    values = np.asarray(values, float)
    return {
        "min": float(values.min()),
        "p01": float(np.percentile(values, 1)),
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _residual_transform(net, membership, gamma):
    """Apply only the six-parameter residual to an already scalar-mapped net."""
    n_e = int(net["NE"])
    membership = np.asarray(membership, float)
    if membership.shape != (n_e, 4):
        raise ValueError(f"membership must have shape ({n_e}, 4)")
    if (not np.isfinite(membership).all() or np.any(membership < 0.0)
            or not np.allclose(membership.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)):
        raise ValueError("membership must be finite, non-negative and sum to one")
    matrix = gamma_matrix(gamma)

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
        zero, one = np.zeros(n_e), np.ones(n_e)
        dominant = np.argmax(membership, axis=1)
        groups = {
            "all_nonzero_incoming": _group_summary(zero, one, positive_targets),
            "component_1_dominant": _group_summary(
                zero, one, positive_targets & (dominant == 0)),
            "component_2_dominant": _group_summary(
                zero, one, positive_targets & (dominant == 1)),
            "component_3_dominant": _group_summary(
                zero, one, positive_targets & (dominant == 2)),
            "background_dominant": _group_summary(
                zero, one, positive_targets & (dominant == 3)),
        }
        return new_net, {
            "exact_noop": True,
            "max_abs_incoming_E_error": 0.0,
            "topology_unchanged": True,
            "e_to_i_unchanged": True,
            "gaba_unchanged": True,
            "invalidated_ampa_cache_keys": removed,
            "edge_ratio": _ratio_summary(np.ones(1)),
            "target_groups": groups,
        }

    records, rows, cols, data = _ee_records(old_bins, n_e)
    incoming = np.bincount(rows, weights=data, minlength=n_e)
    # C3 is intentionally absent. Background/C3 sources are the zero-residual
    # reference; only C1 and C2 source memberships enter the interaction.
    target_r = membership[rows][:, (0, 1, 3)]
    source_r = membership[cols, :2]
    log_multiplier = np.einsum(
        "ni,ij,nj->n", target_r, matrix, source_r, optimize=True)
    log_unnormalized = np.log(data) + log_multiplier
    target_max = np.full(n_e, -np.inf)
    np.maximum.at(target_max, rows, log_unnormalized)
    exp_shifted = np.exp(log_unnormalized - target_max[rows])
    target_sum = np.bincount(rows, weights=exp_shifted, minlength=n_e)
    new_data = incoming[rows] * exp_shifted / target_sum[rows]
    if np.any(~np.isfinite(new_data)) or np.any(new_data <= 0.0):
        raise RuntimeError("component-pair residual produced invalid E-to-E weights")

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
    e_to_i = _hash_sparse_bins(new_bins, rows=slice(n_e, None)) == old_e_to_i_hash
    gaba = _hash_sparse_bins(new_net["gaba_by_delay"]) == old_gaba_hash
    if (abs_error.max(initial=0.0) > 1e-9 or not topology or not e_to_i or not gaba):
        raise RuntimeError("component-pair residual violated structural conservation")

    ratio = new_data / data
    p_old = data / incoming[rows]
    p_new = new_data / incoming[rows]
    kl = np.bincount(rows, weights=p_new * np.log(p_new / p_old), minlength=n_e)
    old_concentration = np.bincount(rows, weights=p_old ** 2, minlength=n_e)
    new_concentration = np.bincount(rows, weights=p_new ** 2, minlength=n_e)
    ess_ratio = np.divide(
        old_concentration, new_concentration,
        out=np.full(n_e, np.nan), where=new_concentration > 0.0)
    dominant = np.argmax(membership, axis=1)
    groups = {
        "all_nonzero_incoming": _group_summary(kl, ess_ratio, positive_targets),
        "component_1_dominant": _group_summary(
            kl, ess_ratio, positive_targets & (dominant == 0)),
        "component_2_dominant": _group_summary(
            kl, ess_ratio, positive_targets & (dominant == 1)),
        "component_3_dominant": _group_summary(
            kl, ess_ratio, positive_targets & (dominant == 2)),
        "background_dominant": _group_summary(
            kl, ess_ratio, positive_targets & (dominant == 3)),
    }
    return new_net, {
        "exact_noop": False,
        "max_abs_incoming_E_error": float(abs_error.max(initial=0.0)),
        "mean_abs_incoming_E_error": float(abs_error.mean()),
        "topology_unchanged": bool(topology),
        "e_to_i_unchanged": bool(e_to_i),
        "gaba_unchanged": bool(gaba),
        "invalidated_ampa_cache_keys": removed,
        "edge_ratio": _ratio_summary(ratio),
        "target_groups": groups,
    }


def component_pair_normalized_ee(net, h_e, membership, gamma, *, alpha=0.75):
    """Add a six-parameter component-pair residual to the frozen scalar edge.

    The final edge log multiplier is ``alpha*h_t*h_s + r_t G r_s``. The
    residual target rows are C1, C2 and background; source columns are C1 and
    C2. C3 remains an unparameterized negative control. Both stages conserve
    every postsynaptic target's total incoming recurrent-E budget.
    """
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    matrix = gamma_matrix(gamma)
    original_records, original_rows, original_cols, original_data = _ee_records(
        net["ampa_by_delay"], int(net["NE"]))
    del original_records
    scalar_net, scalar_diagnostics = field_normalized_ee_pair(
        net, h_e, alpha, beta=0.0)
    final_net, residual_diagnostics = _residual_transform(
        scalar_net, membership, matrix.ravel())
    _, final_rows, final_cols, final_data = _ee_records(
        final_net["ampa_by_delay"], int(net["NE"]))
    if (not np.array_equal(original_rows, final_rows)
            or not np.array_equal(original_cols, final_cols)):
        raise RuntimeError("component-pair mapper changed edge identity")
    edge_ratio = final_data / original_data
    return final_net, {
        "mechanism": "component_pair_residual_target_normalized_EE_exp_v2",
        "alpha": alpha,
        "gamma_names": list(GAMMA_NAMES),
        "gamma": np.asarray(gamma, float).tolist(),
        "gamma_matrix_target_by_source": matrix.tolist(),
        "target_groups": list(TARGET_GROUPS),
        "source_groups": list(SOURCE_GROUPS),
        "component_3_parameterized": False,
        "residual_exact_noop": bool(residual_diagnostics["exact_noop"]),
        "n_E": int(net["NE"]),
        "n_ee_edges": int(len(final_data)),
        "max_abs_incoming_E_error": residual_diagnostics[
            "max_abs_incoming_E_error"],
        "topology_unchanged": bool(
            scalar_diagnostics["topology_unchanged"]
            and residual_diagnostics["topology_unchanged"]),
        "e_to_i_unchanged": bool(
            scalar_diagnostics["e_to_i_unchanged"]
            and residual_diagnostics["e_to_i_unchanged"]),
        "gaba_unchanged": bool(
            scalar_diagnostics["gaba_unchanged"]
            and residual_diagnostics["gaba_unchanged"]),
        "edge_ratio": _ratio_summary(edge_ratio),
        "residual_edge_ratio": residual_diagnostics["edge_ratio"],
        "residual_target_groups": residual_diagnostics["target_groups"],
        "scalar_baseline_diagnostics": scalar_diagnostics,
        "exploratory_warnings": {
            "ratio_outside_0p25_4": bool(
                np.any((edge_ratio < 0.25) | (edge_ratio > 4.0))),
        },
    }


__all__ = [
    "GAMMA_NAMES",
    "component_background_membership",
    "component_pair_normalized_ee",
    "gamma_matrix",
]
