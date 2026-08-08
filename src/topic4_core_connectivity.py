"""Map a learned continuous core field onto recurrent E-to-E connectivity."""
from __future__ import annotations

import copy

import numpy as np
from scipy import sparse


def incoming_ee_weight(ampa_by_delay, n_e):
    """Total recurrent E input to every E target, summed over delay bins."""
    total = np.zeros(int(n_e), dtype=float)
    for matrix in ampa_by_delay:
        total += np.asarray(matrix[:n_e, :].sum(axis=1)).ravel()
    return total


def ee_field_partition(ampa_by_delay, h_e, core_quantile=0.95):
    """Weight accounting for a diagnostic high-field subset.

    The default top 5% is close to the effective support of the current fixed
    field budget. This subset is only a readable summary; the mechanism remains
    continuous and never thresholds ``h_e`` when changing an edge.
    """
    h_e = np.asarray(h_e, dtype=float)
    cut = float(np.quantile(h_e, float(core_quantile)))
    core = h_e >= cut
    weight = dict(within_core=0.0, core_target_other_source=0.0,
                  other_target_core_source=0.0, outside_core=0.0)
    edge_count = {key: 0 for key in weight}
    for matrix in ampa_by_delay:
        coo = matrix.tocoo()
        is_ee = coo.row < len(h_e)
        rows, cols, data = coo.row[is_ee], coo.col[is_ee], coo.data[is_ee]
        masks = dict(
            within_core=core[rows] & core[cols],
            core_target_other_source=core[rows] & ~core[cols],
            other_target_core_source=~core[rows] & core[cols],
            outside_core=~core[rows] & ~core[cols],
        )
        for key, mask in masks.items():
            weight[key] += float(data[mask].sum())
            edge_count[key] += int(mask.sum())
    return dict(core_quantile=float(core_quantile), h_cut=cut,
                n_core=int(core.sum()), weight=weight, edge_count=edge_count)


def field_normalized_ee_core(net, h_e, alpha):
    """Strengthen field-internal E-to-E edges without changing E input totals.

    The unnormalized edge multiplier is ``1 + alpha*h_target*h_source``.
    A target-specific factor then restores each E target's original summed
    incoming E weight across all delay bins. E-to-I edges, topology and delays
    are untouched. The returned network is independent of ``net``.
    """
    n_e = int(net["NE"])
    h_e = np.asarray(h_e, dtype=float)
    alpha = float(alpha)
    if h_e.shape != (n_e,):
        raise ValueError(f"h_e must have shape ({n_e},), got {h_e.shape}")
    if not np.all(np.isfinite(h_e)) or np.any((h_e < 0.0) | (h_e > 1.0)):
        raise ValueError("h_e must be finite and lie in [0, 1]")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")

    old_bins = net["ampa_by_delay"]
    old_incoming = incoming_ee_weight(old_bins, n_e)
    weighted_bins = []
    for matrix in old_bins:
        coo = matrix.tocoo(copy=True)
        is_ee = coo.row < n_e
        coo.data[is_ee] *= 1.0 + alpha * h_e[coo.row[is_ee]] * h_e[coo.col[is_ee]]
        weighted_bins.append(coo.tocsc())

    weighted_incoming = incoming_ee_weight(weighted_bins, n_e)
    target_norm = np.ones(n_e, dtype=float)
    valid = weighted_incoming > 0.0
    target_norm[valid] = old_incoming[valid] / weighted_incoming[valid]

    new_bins = []
    for matrix in weighted_bins:
        coo = matrix.tocoo(copy=True)
        is_ee = coo.row < n_e
        coo.data[is_ee] *= target_norm[coo.row[is_ee]]
        new_bins.append(coo.tocsc())

    new_net = copy.copy(net)
    new_net["ampa_by_delay"] = new_bins
    # The flattened cache, if present, encodes the old weights.
    new_net.pop("ampa_flat", None)
    new_incoming = incoming_ee_weight(new_bins, n_e)
    abs_error = np.abs(new_incoming - old_incoming)
    diagnostics = dict(
        mechanism="field_normalized_EE_core",
        alpha=alpha,
        max_abs_incoming_E_error=float(abs_error.max(initial=0.0)),
        mean_abs_incoming_E_error=float(abs_error.mean() if len(abs_error) else 0.0),
        n_E=n_e,
        n_ampa_edges=int(sum(matrix.nnz for matrix in old_bins)),
    )
    return new_net, diagnostics
