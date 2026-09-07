"""Explicit fixed-graph perturbations for the interictal development pilot.

Sparse rows are postsynaptic targets; AMPA columns are E sources and GABA
columns are I sources. Cached graphs are never written by this module.
"""
import hashlib
import numpy as np


DEFAULTS = dict(E_to_E_weight_scale=1., E_to_I_weight_scale=1.,
                I_to_E_weight_scale=1., I_to_I_weight_scale=1.,
                tau_d_GABA_ms=18.)


def sparse_digest(bins, *, topology=False):
    h = hashlib.sha256()
    for mat in bins:
        h.update(str((mat.shape, mat.format)).encode())
        for a in (mat.indptr, mat.indices) if topology else (mat.indptr, mat.indices, mat.data):
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def scale_target_pathways(bins, n_e, e_scale, i_scale):
    """Copy only changed matrices; preserve indices, delays and source identity."""
    if not all(np.isfinite(v) and v > 0 for v in (e_scale, i_scale)):
        raise ValueError('pathway scales must be finite and positive')
    before = np.zeros(2); after = np.zeros(2); output = []
    for mat in bins:
        if mat.format != 'csc':
            raise ValueError('engine contract requires CSC delay bins')
        is_e = mat.indices < n_e
        before += [mat.data[is_e].sum(dtype=float), mat.data[~is_e].sum(dtype=float)]
        new = mat if e_scale == i_scale == 1. else mat.copy()
        if new is not mat:
            new.data[is_e] *= e_scale
            new.data[~is_e] *= i_scale
        after += [new.data[is_e].sum(dtype=float), new.data[~is_e].sum(dtype=float)]
        output.append(new)
    if not np.allclose(after, before * [e_scale, i_scale], rtol=2e-6, atol=1e-9):
        raise RuntimeError('pathway dose audit failed')
    return output, {'before_sum': before.tolist(), 'after_sum': after.tolist()}


def apply_parameters(substrate, requested):
    unknown = set(requested) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'unregistered parameters: {sorted(unknown)}')
    p = {**DEFAULTS, **requested}
    if not all(np.isfinite(v) and v > 0 for v in p.values()):
        raise ValueError('positive finite parameter values required')
    if p['tau_d_GABA_ms'] <= substrate.params.tau_r_GABA:
        raise ValueError('GABA decay must exceed rise time')
    net = dict(substrate.net)
    audits = {}
    for key, a, b in [('ampa_by_delay', 'E_to_E_weight_scale', 'E_to_I_weight_scale'),
                      ('gaba_by_delay', 'I_to_E_weight_scale', 'I_to_I_weight_scale')]:
        old = net[key]
        topology = sparse_digest(old, topology=True)
        data_before = sparse_digest(old)
        new, dose = scale_target_pathways(old, substrate.n_e, p[a], p[b])
        if sparse_digest(new, topology=True) != topology:
            raise RuntimeError('fixed-graph perturbation changed topology')
        net[key] = new
        audits[key] = {**dose, 'topology_sha256': topology,
                       'before_sha256': data_before, 'after_sha256': sparse_digest(new)}
    # Cache arrays derived from the sparse weights must not survive a dose change.
    if any(p[k] != DEFAULTS[k] for k in DEFAULTS if k.endswith('_scale')):
        from src.topic4_rev20_dual_core_mechanism import _invalidate_ampa_caches
        _invalidate_ampa_caches(net)
        # The engine currently builds its GABA flattening afresh; fail on future caches.
        if any(k.startswith('_gaba') for k in net):
            raise RuntimeError('new GABA cache needs an explicit invalidation contract')
    old_tau = float(substrate.params.tau_d_GABA)
    substrate.params.tau_d_GABA = float(p['tau_d_GABA_ms'])
    substrate.net = net
    return {'requested': p, 'effective': p, 'sparse_pathways': audits,
            'baseline_tau_d_GABA_ms': old_tau,
            'kinetics_contract': 'Change GABA decay at fixed recurrent jump, rise, membrane and external drive parameters; integrated response is not dose-matched.',
            'topology_and_delay_assignments_preserved': True,
            'cached_graph_not_modified': True}
