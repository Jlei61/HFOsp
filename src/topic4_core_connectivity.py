"""Map a learned continuous core field onto recurrent E-to-E connectivity."""
from __future__ import annotations

import copy
import hashlib

import numpy as np
from scipy import sparse


def incoming_ee_weight(ampa_by_delay, n_e):
    """Total recurrent E input to every E target, summed over delay bins."""
    total = np.zeros(int(n_e), dtype=float)
    for matrix in ampa_by_delay:
        total += np.asarray(matrix[:n_e, :].sum(axis=1)).ravel()
    return total


def _hash_sparse_bins(matrices, *, rows=None, include_data=True):
    """Hash sparse matrices after canonical CSR conversion."""
    digest = hashlib.sha256()
    for delay, matrix in enumerate(matrices):
        view = matrix if rows is None else matrix[rows, :]
        csr = view.tocsr(copy=True)
        csr.sort_indices()
        digest.update(np.asarray([delay, *csr.shape], np.int64).tobytes())
        digest.update(np.asarray(csr.indptr, np.int64).tobytes())
        digest.update(np.asarray(csr.indices, np.int64).tobytes())
        if include_data:
            digest.update(np.asarray(csr.data, np.float64).tobytes())
    return digest.hexdigest()


def _invalidate_ampa_caches(net):
    """Drop every known or explicitly registered AMPA-derived cache."""
    removed = []
    registered = set(net.get("_ampa_derived_cache_keys", ()))
    for key in list(net):
        derived = (key in registered or
                   (key != "ampa_by_delay" and
                    (key.startswith("ampa_") or key.startswith("_ampa_"))))
        if derived:
            removed.append(key)
            net.pop(key, None)
    return sorted(removed)


def _group_summary(kl, ess_ratio, mask):
    values = np.asarray(kl, float)[mask]
    ratios = np.asarray(ess_ratio, float)[mask]
    finite = np.isfinite(values) & np.isfinite(ratios)
    if not finite.any():
        return dict(n_targets=0, kl_median=None, kl_p99=None,
                    ess_ratio_median=None, ess_ratio_p01=None)
    values, ratios = values[finite], ratios[finite]
    return dict(
        n_targets=int(len(values)),
        kl_median=float(np.median(values)),
        kl_p99=float(np.percentile(values, 99)),
        ess_ratio_median=float(np.median(ratios)),
        ess_ratio_p01=float(np.percentile(ratios, 1)),
    )


def field_normalized_ee_pair(net, h_e, alpha, *, beta=0.0, pos_e=None,
                             l_ee=None, active_vth_shift=None):
    """Redistribute E-to-E weights within each target's fixed incoming budget.

    Rows are targets and columns are E sources. The primary log multiplier is
    ``alpha*h_target*h_source``. An optional radial term adds
    ``beta*h_target*h_source*kappa_tilde``. Normalization is performed jointly
    across all delay bins with a target-wise log-sum-exp.

    Ratio/KL/ESS limits are diagnostics for the exploratory rev9 round. The
    function only rejects malformed inputs or a transform that violates the
    exact structural/incoming-weight contract.
    """
    n_e = int(net["NE"])
    h_e = np.asarray(h_e, dtype=float)
    alpha, beta = float(alpha), float(beta)
    if h_e.shape != (n_e,):
        raise ValueError(f"h_e must have shape ({n_e},), got {h_e.shape}")
    if not np.all(np.isfinite(h_e)) or np.any((h_e < 0.0) | (h_e > 1.0)):
        raise ValueError("h_e must be finite and lie in [0, 1]")
    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha < 0.0 or beta < 0.0:
        raise ValueError("alpha and beta must be finite and non-negative")
    if beta > 0.0:
        pos_e = np.asarray(pos_e, dtype=float) if pos_e is not None else None
        if pos_e is None or pos_e.shape != (n_e, 2):
            raise ValueError(f"beta>0 requires pos_e with shape ({n_e}, 2)")
        if l_ee is None or not np.isfinite(l_ee) or float(l_ee) <= 0.0:
            raise ValueError("beta>0 requires a finite positive l_ee")

    old_bins = net["ampa_by_delay"]
    old_topology_hash = _hash_sparse_bins(old_bins, include_data=False)
    old_e_to_i_hash = _hash_sparse_bins(old_bins, rows=slice(n_e, None))
    old_gaba_hash = _hash_sparse_bins(net["gaba_by_delay"])
    old_incoming = incoming_ee_weight(old_bins, n_e)
    positive_targets = old_incoming > 0.0

    # Preserve exact values at the origin. Copies keep the returned network
    # independent while avoiding normalization roundoff.
    if alpha == 0.0 and beta == 0.0:
        new_net = copy.copy(net)
        new_net["ampa_by_delay"] = [matrix.copy() for matrix in old_bins]
        removed = _invalidate_ampa_caches(new_net)
        one = np.ones(n_e, dtype=float)
        zero = np.zeros(n_e, dtype=float)
        groups = {
            "all_nonzero_incoming": _group_summary(zero, one, positive_targets),
            "h_positive": _group_summary(zero, one, positive_targets & (h_e > 0.0)),
            "h_top10_percent": _group_summary(
                zero, one, positive_targets & (h_e >= np.quantile(h_e, 0.9))),
        }
        return new_net, dict(
            mechanism="field_normalized_EE_pair_exp_v2",
            alpha=alpha, beta=beta, exact_noop=True, n_E=n_e,
            n_zero_incoming_e_targets=int((~positive_targets).sum()),
            max_abs_incoming_E_error=0.0,
            topology_unchanged=True, e_to_i_unchanged=True, gaba_unchanged=True,
            invalidated_ampa_cache_keys=removed,
            edge_ratio=dict(min=1.0, p01=1.0, median=1.0, p99=1.0, max=1.0),
            target_groups=groups,
        )

    records = []
    all_rows, all_cols, all_data = [], [], []
    for matrix in old_bins:
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
    if len(data) == 0:
        raise ValueError("network contains no E-to-E edges")

    incoming = np.bincount(rows, weights=data, minlength=n_e)
    feature = h_e[rows] * h_e[cols]
    log_multiplier = alpha * feature
    if beta > 0.0:
        dist2 = ((pos_e[rows] - pos_e[cols]) ** 2).sum(axis=1)
        kappa = np.exp(-dist2 / (2.0 * float(l_ee) ** 2))
        weighted_kappa = np.bincount(rows, weights=data * kappa, minlength=n_e)
        mean = np.divide(weighted_kappa, incoming, out=np.zeros(n_e), where=incoming > 0.0)
        weighted_var = np.bincount(
            rows, weights=data * (kappa - mean[rows]) ** 2, minlength=n_e)
        sd = np.sqrt(np.divide(weighted_var, incoming, out=np.zeros(n_e),
                               where=incoming > 0.0))
        kappa_tilde = (kappa - mean[rows]) / np.maximum(sd[rows], 1e-12)
        log_multiplier = log_multiplier + beta * feature * kappa_tilde

    log_unnormalized = np.log(data) + log_multiplier
    target_max = np.full(n_e, -np.inf, dtype=float)
    np.maximum.at(target_max, rows, log_unnormalized)
    exp_shifted = np.exp(log_unnormalized - target_max[rows])
    target_sum = np.bincount(rows, weights=exp_shifted, minlength=n_e)
    new_data = incoming[rows] * exp_shifted / target_sum[rows]
    if np.any(~np.isfinite(new_data)) or np.any(new_data <= 0.0):
        raise RuntimeError("edge transform produced invalid E-to-E weights")

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
    topology_unchanged = _hash_sparse_bins(new_bins, include_data=False) == old_topology_hash
    e_to_i_unchanged = (_hash_sparse_bins(new_bins, rows=slice(n_e, None)) ==
                        old_e_to_i_hash)
    gaba_unchanged = _hash_sparse_bins(new_net["gaba_by_delay"]) == old_gaba_hash
    if (abs_error.max(initial=0.0) > 1e-9 or not topology_unchanged or
            not e_to_i_unchanged or not gaba_unchanged):
        raise RuntimeError("edge transform violated the structural conservation contract")

    edge_ratio = new_data / data
    p_old = data / incoming[rows]
    p_new = new_data / incoming[rows]
    kl = np.bincount(rows, weights=p_new * np.log(p_new / p_old), minlength=n_e)
    concentration_old = np.bincount(rows, weights=p_old ** 2, minlength=n_e)
    concentration_new = np.bincount(rows, weights=p_new ** 2, minlength=n_e)
    ess_old = np.divide(
        1.0, concentration_old, out=np.full(n_e, np.nan),
        where=concentration_old > 0.0)
    ess_new = np.divide(
        1.0, concentration_new, out=np.full(n_e, np.nan),
        where=concentration_new > 0.0)
    ess_ratio = np.divide(
        ess_new, ess_old, out=np.full(n_e, np.nan),
        where=np.isfinite(ess_old) & (ess_old > 0.0))
    top10 = h_e >= np.quantile(h_e, 0.9)
    groups = {
        "all_nonzero_incoming": _group_summary(kl, ess_ratio, positive_targets),
        "h_positive": _group_summary(kl, ess_ratio, positive_targets & (h_e > 0.0)),
        "h_top10_percent": _group_summary(kl, ess_ratio, positive_targets & top10),
    }
    if active_vth_shift is not None:
        shift = np.asarray(active_vth_shift, float)
        if shift.shape != (n_e,):
            raise ValueError(f"active_vth_shift must have shape ({n_e},)")
        groups["abs_vth_shift_ge_0p1_mV"] = _group_summary(
            kl, ess_ratio, positive_targets & (np.abs(shift) >= 0.1))

    return new_net, dict(
        mechanism="field_normalized_EE_pair_exp_v2",
        alpha=alpha, beta=beta, exact_noop=False, n_E=n_e,
        n_ee_edges=int(len(data)),
        n_zero_incoming_e_targets=int((~positive_targets).sum()),
        max_abs_incoming_E_error=float(abs_error.max(initial=0.0)),
        mean_abs_incoming_E_error=float(abs_error.mean()),
        topology_unchanged=bool(topology_unchanged),
        e_to_i_unchanged=bool(e_to_i_unchanged),
        gaba_unchanged=bool(gaba_unchanged),
        invalidated_ampa_cache_keys=removed,
        edge_ratio=dict(
            min=float(edge_ratio.min()), p01=float(np.percentile(edge_ratio, 1)),
            median=float(np.median(edge_ratio)), p99=float(np.percentile(edge_ratio, 99)),
            max=float(edge_ratio.max())),
        target_groups=groups,
        exploratory_warnings=dict(
            ratio_outside_0p25_4=bool(np.any((edge_ratio < 0.25) | (edge_ratio > 4.0))),
            all_target_kl_above_reference=bool(
                groups["all_nonzero_incoming"]["kl_p99"] > 0.20),
        ),
    )


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
