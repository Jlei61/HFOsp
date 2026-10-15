"""core_connectivity_v2 physics (design 2026-09-10; checklist C1, C4-C7).

Lowering-only two-core threshold field, block-wise pathway weight scaling by
explicit pre/post type and A/B/O membership, real core-to-out in-degree changes
and whole-medium E->E kernel resampling with distance delays. Every stage is
computed from the immutable baseline graph handed in; no cached graph, historical
module or ``Params`` weight is modified.
"""
from __future__ import annotations
import copy
import hashlib
import numpy as np
from scipy import sparse
from connectivity import _group_by_delay, _positive_weight_keys
from connectivity_rot import _kernel_logweights_rot
from src.topic4_core_field import sample_core_quantiles, core_thresholds

PHYSICS_VERSION = 'core_connectivity_v2'
WEIGHT_FACTORS = ('EE_same_core_scale', 'EE_core_to_out_scale', 'EE_out_to_out_scale',
                  'EI_same_core_scale', 'IE_same_core_scale', 'II_same_core_scale')
DEGREE_KEY = 'EE_core_to_out_degree_scale'
KERNEL_KEYS = ('EE_kernel_perp_scale', 'EE_kernel_parallel_scale', 'EE_angle_offset_deg')
GEOMETRY_KEYS = ('depth_A_scale', 'depth_B_scale', 'radius_A_mm', 'radius_B_mm')
_KEY_KERNEL, _KEY_DEGREE = 1, 2


def array_sha256(values):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values)).view(np.uint8)).hexdigest()


# ---------------------------------------------------------------- C1 geometry / thresholds
def core_index_for(positions, centers, radii):
    """-1 outside every disk; otherwise the nearest containing core (0=A, 1=B)."""
    pos = np.asarray(positions, float); centers = np.asarray(centers, float); radii = np.asarray(radii, float)
    d = np.linalg.norm(pos[:, None] - centers[None], axis=2)
    inside = d <= radii[None]
    nearest = np.argmin(np.where(inside, d, np.inf), axis=1)
    return np.where(inside.any(1), nearest, -1)


def threshold_field(positions_e, centers, radii, depth_scales, *, n_total, quantile_seed,
                    core_mean, core_std, v_base, floor_mv=11., sheet_mm=20.):
    """C1: Vth = v_base - a_k * d_i inside core k, d_i = max(v_base - Vraw_i, 0), floor at 11 mV.

    Latent Vraw_i is drawn once per neuron identity (quantile_seed) and never
    depends on centers, radii, scales or core order. Overlap takes the stronger
    lowering (minimum threshold), never the sum. Outside and I cells keep v_base.
    """
    pos = np.asarray(positions_e, float); n_e = len(pos)
    centers = np.asarray(centers, float); radii = np.asarray(radii, float); scales = np.asarray(depth_scales, float)
    if centers.shape != (2, 2) or radii.shape != (2,) or scales.shape != (2,):
        raise ValueError('two cores with centers (2,2), radii (2,) and depth scales (2,) required')
    if np.any(radii < 0) or np.any(scales < 0) or not np.isfinite(centers).all():
        raise ValueError('radii and depth scales must be finite and non-negative')
    latent_raw = core_thresholds(sample_core_quantiles(n_e, int(quantile_seed)), float(core_mean), float(core_std))
    d_latent = np.maximum(float(v_base) - latent_raw, 0.)
    dist = np.linalg.norm(pos[:, None] - centers[None], axis=2)
    inside = dist <= radii[None]                                   # (n_e, 2)
    candidate = np.where(inside, float(v_base) - scales[None] * d_latent[:, None], np.inf)
    raw = np.min(candidate, axis=1)
    member = inside.any(1)
    vth_e = np.where(member, np.maximum(raw, float(floor_mv)), float(v_base))
    clipped = member & (raw < float(floor_mv))
    core_index = core_index_for(pos, centers, radii)
    vtheta = np.full(int(n_total), float(v_base)); vtheta[:n_e] = vth_e
    lowering = float(v_base) - vth_e
    audit = dict(
        physics=PHYSICS_VERSION, centers_mm=centers.tolist(), radii_mm=radii.tolist(),
        depth_scales=scales.tolist(), floor_mV=float(floor_mv), v_base_mV=float(v_base),
        members=[int((core_index == k).sum()) for k in range(2)], n_members=int(member.sum()),
        overlap_members=int(inside.all(1).sum()),
        total_lowering_mV=float(lowering.sum()),
        total_lowering_per_core_mV=[float(lowering[core_index == k].sum()) for k in range(2)],
        mean_lowering_per_member_mV=[float(lowering[core_index == k].mean()) if (core_index == k).any() else None for k in range(2)],
        floor_clipped_count=int(clipped.sum()), floor_clipped_total_mV=float(np.sum(float(floor_mv) - raw[clipped])),
        n_raised=int(np.sum(vth_e > float(v_base))), n_lowered=int(np.sum(vth_e < float(v_base))),
        latent_identity_sha256=array_sha256(d_latent), quantile_seed=int(quantile_seed),
        boundary_clipped=[bool(min(c[0], c[1], sheet_mm - c[0], sheet_mm - c[1]) < r) for c, r in zip(centers, radii)],
        disks_overlap=bool(np.linalg.norm(centers[1] - centers[0]) < radii.sum()),
        min_vtheta_mV=float(vth_e.min()), max_vtheta_mV=float(vth_e.max()),
    )
    return dict(vtheta=vtheta, delta_vtheta=-lowering, core_index=core_index, d_latent=d_latent, audit=audit)


# ---------------------------------------------------------------- sparse helpers
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


def _adjacency_sha256(rows, cols):
    order = np.lexsort((cols, rows))
    return array_sha256(np.column_stack([rows[order], cols[order]]))


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


# ---------------------------------------------------------------- C4 block weights
def _block_masks(rows, cols, n_e, e_core, i_core, pathway):
    target_e = rows < n_e
    tc = np.where(target_e, e_core[np.minimum(rows, n_e - 1)], i_core[np.maximum(rows - n_e, 0)])
    sc = e_core[cols] if pathway == 'ampa' else i_core[cols]
    same = (tc >= 0) & (sc == tc)
    if pathway == 'ampa':
        return {'EE_same_core_scale': target_e & same,
                'EE_core_to_out_scale': target_e & (tc < 0) & (sc >= 0),
                'EE_out_to_out_scale': target_e & (tc < 0) & (sc < 0),
                'EI_same_core_scale': (~target_e) & same}
    return {'IE_same_core_scale': target_e & same, 'II_same_core_scale': (~target_e) & same}


def scale_pathway_blocks(ampa_bins, gaba_bins, n_e, e_core, i_core, factors):
    """C4: multiply weights of the six named blocks; edges, delays and every other block untouched."""
    factors = {k: float(factors.get(k, 1.)) for k in WEIGHT_FACTORS}
    if any(not np.isfinite(v) or v <= 0 for v in factors.values()):
        raise ValueError('weight factors must be finite and positive')
    e_core = np.asarray(e_core, int); i_core = np.asarray(i_core, int)
    audit = dict(factors=factors, blocks={}, exact_noop=all(v == 1. for v in factors.values()))
    outputs = []
    for pathway, bins in (('ampa', ampa_bins), ('gaba', gaba_bins)):
        names = [k for k in WEIGHT_FACTORS if k in _block_masks(np.zeros(0, int), np.zeros(0, int), n_e, e_core, i_core, pathway)]
        counts = {k: 0 for k in names}; before = {k: 0. for k in names}; after = {k: 0. for k in names}
        touched = any(factors[k] != 1. for k in names)
        new_bins = []
        for m in bins:
            if m.format != 'csc':
                raise ValueError('engine contract requires CSC delay bins')
            if m.nnz == 0:
                new_bins.append(m); continue
            coo = m.tocoo(copy=False)
            rows = np.asarray(coo.row, np.int64); cols = np.asarray(coo.col, np.int64)
            masks = _block_masks(rows, cols, n_e, e_core, i_core, pathway)
            new = m.copy() if touched else m
            data = new.data if touched else m.data
            # CSC .data order == COO(copy=False) order for the same matrix (column-major, sorted indices).
            for k in names:
                sel = masks[k]; counts[k] += int(sel.sum()); before[k] += float(m.data[sel].sum())
                if touched and factors[k] != 1.:
                    data[sel] *= factors[k]
                after[k] += float(data[sel].sum())
            new_bins.append(new)
        for k in names:
            audit['blocks'][k] = dict(n_edges=counts[k], weight_before=before[k], weight_after=after[k], factor=factors[k])
            if counts[k] and not np.isclose(after[k], before[k] * factors[k], rtol=1e-9, atol=1e-9):
                raise RuntimeError(f'block dose audit failed for {k}')
        outputs.append(new_bins)
    return outputs[0], outputs[1], audit


# ---------------------------------------------------------------- C6 kernel resampling
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
                 adjacency_identical_to_baseline=bool(_adjacency_sha256(new_rows, new_cols) == _adjacency_sha256(rows[ee], cols[ee])),
                 partner_spread_along_axis_mm=float(np.std(dz @ u)), partner_spread_across_axis_mm=float(np.std(dz @ v)),
                 mean_partner_distance_mm=float(dist.mean()), max_partner_distance_mm=float(dist.max()),
                 mean_delay_ms=float((p.tau0 + dist / p.v_axon).mean()), max_delay_steps=int(new_steps.max()),
                 total_ee_weight=float(new_data.sum()), baseline_total_ee_weight=float(data[ee].sum()))
    return bins, audit


# ---------------------------------------------------------------- C5 core-to-out in-degree
def apply_core_to_out_degree(ampa_bins, positions, n_e, e_core, factor, *, topology_seed, kernel, p):
    """C5: per outside E target, set the number of in-edges from core E sources to
    round(baseline * factor) by deterministic removal / kernel-weighted addition."""
    factor = float(factor)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('degree factor must be finite and positive')
    pos = np.asarray(positions, float); pos_e = pos[:n_e]; e_core = np.asarray(e_core, int)
    rows, cols, data, steps = _edges(ampa_bins)
    n_rows = ampa_bins[0].shape[0]
    ee = rows < n_e
    block = ee & (e_core[np.minimum(rows, n_e - 1)] < 0) & (e_core[cols] >= 0)
    outside = np.flatnonzero(e_core < 0); core_sources = np.flatnonzero(e_core >= 0)
    base_count = np.bincount(rows[block], minlength=n_e)
    new_count = np.where(base_count > 0, np.floor(base_count * factor + .5).astype(np.int64), 0)
    indeg = np.bincount(rows[ee], minlength=n_e)
    wmean = np.divide(np.bincount(rows[ee], weights=data[ee], minlength=n_e), indeg, out=np.zeros(n_e), where=indeg > 0)
    rng = _topology_rng(topology_seed, _KEY_DEGREE)
    keep = np.ones(len(rows), bool)
    add_rows, add_cols = [], []
    block_idx = np.flatnonzero(block)
    order = np.lexsort((cols[block_idx], rows[block_idx])); block_idx = block_idx[order]
    key_of = np.full(len(rows), np.inf)
    key_of[block_idx] = rng.random(len(block_idx))       # one fixed key per block edge, canonical (row, col) order
    if factor < 1.:
        for t in np.flatnonzero(new_count < base_count):
            sel = block_idx[rows[block_idx] == t]
            drop = sel[np.argsort(key_of[sel], kind='stable')[new_count[t]:]]
            keep[drop] = False
    elif factor > 1.:
        for t in np.flatnonzero(new_count > base_count):
            need = int(new_count[t] - base_count[t])
            existing = cols[block_idx[rows[block_idx] == t]]
            w = _kernel_weights(pos_e[t], pos_e[core_sources], kernel)
            w[np.isin(core_sources, existing)] = 0.
            w[core_sources == t] = 0.
            nz = int(np.count_nonzero(w))
            if nz == 0:
                continue
            take = min(need, nz)
            keys = _positive_weight_keys(w, rng)
            chosen = core_sources[np.argpartition(keys, take - 1)[:take]]
            add_rows.append(np.full(take, t, np.int64)); add_cols.append(chosen.astype(np.int64))
    if add_rows:
        add_rows = np.concatenate(add_rows); add_cols = np.concatenate(add_cols)
        dist = np.linalg.norm(pos_e[add_cols] - pos_e[add_rows], axis=1)
        add_steps = _delay_steps(dist, p); add_data = wmean[add_rows]
    else:
        add_rows = add_cols = add_steps = np.empty(0, np.int64); add_data = np.empty(0, float)
    all_rows = np.concatenate([rows[keep], add_rows]); all_cols = np.concatenate([cols[keep], add_cols])
    all_data = np.concatenate([data[keep], add_data]); all_steps = np.concatenate([steps[keep], add_steps])
    if len(set(zip(all_rows.tolist(), all_cols.tolist()))) != len(all_rows) if len(all_rows) < 5_000_000 else False:
        raise RuntimeError('duplicate edge after degree change')
    bins = _regroup(all_rows, all_cols, all_data, all_steps, n_rows, n_e)
    final_block = (all_rows < n_e) & (e_core[np.minimum(all_rows, n_e - 1)] < 0) & (e_core[all_cols] >= 0)
    final_count = np.bincount(all_rows[final_block], minlength=n_e)
    audit = dict(stage='degree', factor=factor, topology_seed=int(topology_seed), rng_label=_KEY_DEGREE,
                 n_outside_targets=int(len(outside)), n_targets_with_core_input=int(np.sum(base_count[outside] > 0)),
                 n_zero_baseline_targets=int(np.sum(base_count[outside] == 0)),
                 n_edges_removed=int((~keep).sum()), n_edges_added=int(len(add_rows)),
                 baseline_block_edges=int(block.sum()), final_block_edges=int(final_block.sum()),
                 requested_block_edges=int(new_count[outside].sum()),
                 shortfall_targets=int(np.sum(final_count[outside] < new_count[outside])),
                 mean_block_indegree_before=float(base_count[outside][base_count[outside] > 0].mean()) if np.any(base_count[outside] > 0) else None,
                 mean_block_indegree_after=float(final_count[outside][base_count[outside] > 0].mean()) if np.any(base_count[outside] > 0) else None,
                 block_weight_before=float(data[block].sum()), block_weight_after=float(all_data[final_block].sum()),
                 kernel=kernel)
    return bins, audit


# ---------------------------------------------------------------- summaries
def pathway_block_summary(ampa_bins, gaba_bins, n_e, e_core, i_core):
    e_core = np.asarray(e_core, int); i_core = np.asarray(i_core, int)
    out = {}
    for pathway, bins in (('ampa', ampa_bins), ('gaba', gaba_bins)):
        rows, cols, data, steps = _edges(bins)
        target_e = rows < n_e
        tc = np.where(target_e, e_core[np.minimum(rows, n_e - 1)], i_core[np.maximum(rows - n_e, 0)])
        sc = e_core[cols] if pathway == 'ampa' else i_core[cols]
        pre = 'E' if pathway == 'ampa' else 'I'
        for post_name, post_mask in (('E', target_e), ('I', ~target_e)):
            for tname, tsel in (('A', tc == 0), ('B', tc == 1), ('O', tc < 0)):
                for sname, ssel in (('A', sc == 0), ('B', sc == 1), ('O', sc < 0)):
                    sel = post_mask & tsel & ssel
                    if sel.any():
                        out[f'{pre}{sname}->{post_name}{tname}'] = dict(n_edges=int(sel.sum()), total_weight=float(data[sel].sum()),
                                                                       mean_delay_ms=float(steps[sel].mean() * 0.1))
    return out


def build_candidate_network(net, positions, n_e, e_core, i_core, params, *, topology_seed, p, reference_kernel):
    """C7: kernel baseline -> core-to-out degree -> block weights, each from the immutable input."""
    params = dict(params)
    kernel = ee_kernel(p, theta_deg=reference_kernel['reference_theta_deg'] if 'reference_theta_deg' in reference_kernel else reference_kernel['theta_deg'],
                       ar=reference_kernel['reference_ar'] if 'reference_ar' in reference_kernel else reference_kernel['ar'],
                       perp_scale=params.get('EE_kernel_perp_scale', 1.), parallel_scale=params.get('EE_kernel_parallel_scale', 1.),
                       angle_offset_deg=params.get('EE_angle_offset_deg', 0.))
    stages, audits = [], {}
    ampa = net['ampa_by_delay']; gaba = net['gaba_by_delay']
    adjacency_changes = False
    if not kernel['is_reference']:
        ampa, audits['kernel'] = resample_ee_kernel(ampa, positions, n_e, topology_seed=topology_seed, kernel=kernel, p=p)
        stages.append('kernel'); adjacency_changes = True
    degree = float(params.get(DEGREE_KEY, 1.))
    if degree != 1.:
        ampa, audits['degree'] = apply_core_to_out_degree(ampa, positions, n_e, e_core, degree, topology_seed=topology_seed, kernel=kernel, p=p)
        stages.append('degree'); adjacency_changes = True
    ampa, gaba, audits['weights'] = scale_pathway_blocks(ampa, gaba, n_e, e_core, i_core, {k: params.get(k, 1.) for k in WEIGHT_FACTORS})
    stages.append('weights')
    length = max(len(ampa), len(gaba))
    ampa = _pad(ampa, length, ampa[0].shape); gaba = _pad(gaba, length, gaba[0].shape)
    built = {k: v for k, v in net.items() if not (k.startswith('ampa_') or k.startswith('_ampa') or k.startswith('gaba_') or k.startswith('_gaba')) or k in ('ampa_by_delay', 'gaba_by_delay')}
    built['ampa_by_delay'] = ampa; built['gaba_by_delay'] = gaba; built['max_delay_steps'] = length - 1
    audit = dict(physics=PHYSICS_VERSION, stages=stages, adjacency_changes=adjacency_changes, kernel=kernel,
                 parameters={k: float(v) for k, v in params.items()}, stage_audits=audits,
                 max_delay_steps=length - 1, block_summary=pathway_block_summary(ampa, gaba, n_e, e_core, i_core))
    return built, audit
