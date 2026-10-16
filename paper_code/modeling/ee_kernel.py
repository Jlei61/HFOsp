"""E→E resampling copied from topic4_core_connectivity_v2 (paper Fig4 C/D).

Preserve E→I and all GABA edges; the original kernel RNG is separate from
placement and from dynamics. Only bookkeeping hashes have been omitted.
"""
import numpy as np
from scipy import sparse
from modeling.src.snn_engine.connectivity import _group_by_delay, _positive_weight_keys
from modeling.src.snn_engine.connectivity_rot import _kernel_logweights_rot
_KEY_KERNEL=1

def _edges(bins):
    rows, cols, data, steps = [], [], [], []
    for d, m in enumerate(bins):
        if m.nnz == 0:
            continue
        c = m.tocoo(copy=False)
        rows.append(np.asarray(c.row, np.int64)); cols.append(np.asarray(c.col, np.int64))
        data.append(np.asarray(c.data, np.float64)); steps.append(np.full(c.nnz, d, np.int64))
    cat = lambda xs, dtype: np.concatenate(xs) if xs else np.empty(0, dtype)
    return cat(rows, np.int64), cat(cols, np.int64), cat(data, np.float64), cat(steps, np.int64)

def _regroup(rows, cols, data, steps, n_rows, n_src, max_steps=None):
    max_steps = int(steps.max()) if max_steps is None else int(max_steps)
    return _group_by_delay(rows, cols, data, steps, max_steps, n_rows, n_src)

def _pad(bins, length, shape):
    empty = sparse.csc_matrix(shape)
    return list(bins) + [empty] * (length - len(bins))

def _delay_steps(distance, p):
    step = max(1, int(round(p.delay_dt / p.dt)))
    return np.maximum(1, np.round((p.tau0 + distance / p.v_axon) / p.delay_dt).astype(np.int64)) * step

def ee_kernel(p, *, theta_deg, ar, perp_scale=1., parallel_scale=1., angle_offset_deg=0.):
    """Rotated elliptical-exponential E->E kernel; scales act on the two axes independently (design §3)."""
    l_par = float(p.l_EE) * np.sqrt(float(ar)) * float(parallel_scale)
    l_perp = float(p.l_EE) / np.sqrt(float(ar)) * float(perp_scale)
    theta = float(theta_deg) + float(angle_offset_deg)
    return dict(l_par=l_par, l_perp=l_perp, theta_deg=theta, theta_rad=float(np.deg2rad(theta)),
                reference_theta_deg=float(theta_deg), reference_ar=float(ar),
                perp_scale=float(perp_scale), parallel_scale=float(parallel_scale), angle_offset_deg=float(angle_offset_deg),
                is_reference=bool(perp_scale == 1. and parallel_scale == 1. and angle_offset_deg == 0.))

def _kernel_weights(target_xy, source_xy, kernel):
    lw = _kernel_logweights_rot(source_xy - target_xy, kernel['l_par'], kernel['l_perp'], kernel['theta_rad'])
    return np.exp(lw - lw.max())

def _topology_rng(topology_seed, label):
    return np.random.default_rng(np.random.SeedSequence([int(topology_seed), 20260910, int(label)]))

def resample_ee_kernel(ampa_bins, positions, n_e, *, topology_seed, kernel, p):
    """C6: resample every E target's E->E in-edges under `kernel`, keeping the baseline
    in-degree and per-target mean weight; delays follow tau0 + d/v; E->I edges preserved."""
    pos = np.asarray(positions, float); pos_e = pos[:n_e]
    rows, cols, data, steps = _edges(ampa_bins)
    ee = rows < n_e
    n_rows = ampa_bins[0].shape[0]
    indeg = np.bincount(rows[ee], minlength=n_e)
    wsum = np.bincount(rows[ee], weights=data[ee], minlength=n_e)
    wmean = np.divide(wsum, indeg, out=np.zeros(n_e), where=indeg > 0)
    rng = _topology_rng(topology_seed, _KEY_KERNEL)
    new_rows, new_cols = [], []
    for i in range(n_e):
        c = int(indeg[i])
        if c == 0:
            continue
        w = _kernel_weights(pos_e[i], pos_e, kernel); w[i] = 0.
        keys = _positive_weight_keys(w, rng)
        chosen = np.argpartition(keys, c - 1)[:c]
        new_rows.append(np.full(c, i, np.int64)); new_cols.append(chosen.astype(np.int64))
    new_rows = np.concatenate(new_rows); new_cols = np.concatenate(new_cols)
    dist = np.linalg.norm(pos_e[new_cols] - pos_e[new_rows], axis=1)
    new_steps = _delay_steps(dist, p); new_data = wmean[new_rows]
    all_rows = np.concatenate([new_rows, rows[~ee]]); all_cols = np.concatenate([new_cols, cols[~ee]])
    all_data = np.concatenate([new_data, data[~ee]]); all_steps = np.concatenate([new_steps, steps[~ee]])
    bins = _regroup(all_rows, all_cols, all_data, all_steps, n_rows, n_e)
    dz = pos_e[new_cols] - pos_e[new_rows]
    u = np.array([np.cos(kernel['theta_rad']), np.sin(kernel['theta_rad'])]); v = np.array([-u[1], u[0]])
    audit = dict(stage='kernel', kernel=kernel, topology_seed=int(topology_seed), rng_label=_KEY_KERNEL,
                 n_ee_edges=int(len(new_rows)), n_e_to_i_edges=int((~ee).sum()),
                 indegree_preserved=bool(np.array_equal(np.bincount(new_rows, minlength=n_e), indeg)),
                 partner_spread_along_axis_mm=float(np.std(dz @ u)), partner_spread_across_axis_mm=float(np.std(dz @ v)),
                 mean_partner_distance_mm=float(dist.mean()), max_partner_distance_mm=float(dist.max()),
                 mean_delay_ms=float((p.tau0 + dist / p.v_axon).mean()), max_delay_steps=int(new_steps.max()),
                 total_ee_weight=float(new_data.sum()), baseline_total_ee_weight=float(data[ee].sum()))
    return bins, audit
