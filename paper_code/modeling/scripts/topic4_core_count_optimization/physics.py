"""K=1/2/3/4 circular cores with the original E1146 graph, rates and threshold law.

The K=2 field is delegated to the accepted implementation. The new K=1/3
field uses its identical latent neuron thresholds and nearest-disk membership.
OU streams 0/1 retain the original two-core RNG; streams 2/3 are independent.
"""
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import fcntl
import pickle
import numpy as np
from modeling.scripts.topic4_core_count_optimization.common import OUT, native_imports
native_imports()
from modeling.scripts import run_topic4_shape_output_response as original
from modeling.src.topic4_core_field import sample_core_quantiles, core_thresholds
from modeling.src.topic4_core_ou_correlation import CoreOUMixture
base, rt, v2, shape = (original.base, original.rt, original.v2, original.shape)

def core_field(pos_e, pos_i, centers, radii, scales, **kw):
    k = len(centers)
    if k == 2:
        return shape.shape_field(pos_e, pos_i, centers, radii, scales, {}, **kw)
    if k not in (1, 3, 4):
        raise ValueError('Only K=1/2/3/4 is authorized')
    centers, radii, scales = (np.asarray(centers), np.asarray(radii), np.asarray(scales))
    if centers.shape != (k, 2) or radii.shape != (k,) or scales.shape != (k,):
        raise ValueError('Core shape')
    if not np.isfinite(centers).all() or np.any(radii <= 0) or np.any(scales != 1):
        raise ValueError('Invalid core physics')
    if np.any(centers - radii[:, None] < 0) or np.any(centers + radii[:, None] > kw['sheet_mm']):
        raise ValueError('Core outside sheet')
    ec = v2.core_index_for(pos_e, centers, radii)
    ic = v2.core_index_for(pos_i, centers, radii)
    raw = core_thresholds(sample_core_quantiles(len(pos_e), int(kw['quantile_seed'])), float(kw['core_mean']), float(kw['core_std']))
    base_v = float(kw['v_base'])
    cap = base_v - 11.0
    latent = np.maximum(base_v - raw, 0.0)
    lowering = np.where(ec >= 0, np.minimum(latent, cap), 0.0)
    vtheta = np.full(kw['n_total'], base_v)
    vtheta[:len(pos_e)] -= lowering
    counts = [int((ec == j).sum()) for j in range(k)]
    audit = dict(physics='core_count_circular_v1', centers_mm=centers, radii_mm=radii, depth_scales=scales, members=counts, n_members=int((ec >= 0).sum()), n_lowered=int((lowering > 0).sum()), n_raised=int((vtheta > base_v).sum()), total_lowering_mV=float(lowering.sum()), total_lowering_per_core_mV=[float(lowering[ec == j].sum()) for j in range(k)], mean_lowering_per_member_mV=[float(lowering[ec == j].mean()) if counts[j] else None for j in range(k)], floor_mV=11.0, floor_clipped_count=int(((latent > cap) & (ec >= 0)).sum()), v_base_mV=base_v, latent_identity_sha256=v2.array_sha256(latent), quantile_seed=int(kw['quantile_seed']), min_vtheta_mV=float(vtheta.min()), max_vtheta_mV=float(vtheta.max()), boundary_clipped=[False] * k)
    return dict(vtheta=vtheta, delta_vtheta=-lowering, core_index=ec, i_core_index=ic, d_latent=latent, audit=audit)

class CoreCountOU(CoreOUMixture):

    def __init__(self, core_index, signal_per_ms, mean_scale, rho, dt_ms, tau_ms, sigma_n, dynamics_seed):
        k = int(np.max(core_index)) + 1
        super().__init__(np.minimum(core_index, 1), signal_per_ms, mean_scale, rho, dt_ms, tau_ms, sigma_n, dynamics_seed)
        if k not in (1, 2, 3, 4) or rho != 0:
            raise ValueError('This comparison fixes independent core OU')
        self.k = k
        self.core_index = np.asarray(core_index, int)
        self.members = [np.flatnonzero(self.core_index == j) for j in range(k)]
        self.state = np.zeros(k)
        self.third_rng = np.random.default_rng(np.random.SeedSequence([int(dynamics_seed), 20260928, 203]))
        self.fourth_rng = np.random.default_rng(np.random.SeedSequence([int(dynamics_seed), 20260928, 204]))

    def step(self, time_ms):
        if not np.isclose(time_ms, self.n_steps * self.dt, rtol=0, atol=1e-07):
            raise ValueError('OU step order')
        draws = self.rng.standard_normal(2)
        if self.k >= 3:
            draws = np.r_[draws, self.third_rng.standard_normal()]
        if self.k == 4:
            draws = np.r_[draws, self.fourth_rng.standard_normal()]
        self.state = self.a * self.state + self.b * draws[:self.k]
        for j, index in enumerate(self.members):
            self.values[index] = (self.mean_scale - 1) * self.signal + self.independent_loading * self.state[j]
        self.n_steps += 1
        return self.values

    def metadata(self):
        d = super().metadata()
        d.update(core_count=self.k, initial_state=[0.0] * self.k, stream_contract='First two RNG streams exactly preserve historical K=2; independent third stream [seed,20260928,203]; fourth stream [seed,20260928,204].')
        return d

class InputAudit(base.RateAudit):

    def __init__(self, groups, nt, dt, drive):
        super().__init__(groups, nt, dt)
        self.drive = drive
        self.max_core_error = {f'core{chr(65 + j)}E': 0.0 for j in range(drive.k)}
        self.max_I_deviation = 0.0

    def observe(self, t, tm, signal, xi, rates):
        super().observe(t, tm, signal, xi, rates)
        for j, name in enumerate(self.max_core_error):
            expected = max(self.drive.mean_scale * signal + self.drive.state[j], 0.0)
            self.max_core_error[name] = max(self.max_core_error[name], float(np.max(abs(rates[self.groups[name]] - expected), initial=0.0)))
        self.max_I_deviation = max(self.max_I_deviation, float(np.max(abs(rates[self.groups['allI']] - max(signal, 0.0)), initial=0.0)))

    def summary(self):
        return dict(maximum_core_rate_error=self.max_core_error, maximum_I_rate_error=self.max_I_deviation, maximum_outside_rate_error=self.max_outside_deviation, calls=self.calls)

def pathway_summary(ampa, gaba, ne, ec, ic, dt):
    out = {}
    k = int(np.max(ec)) + 1
    labels = [(chr(65 + j), j) for j in range(k)] + [('O', -1)]
    for pre, bins in [('E', ampa), ('I', gaba)]:
        rows, cols, data, steps = v2._edges(bins)
        target_e = rows < ne
        tc = np.where(target_e, ec[np.minimum(rows, ne - 1)], ic[np.maximum(rows - ne, 0)])
        sc = ec[cols] if pre == 'E' else ic[cols]
        for post, mask in [('E', target_e), ('I', ~target_e)]:
            for tn, t in labels:
                for sn, s in labels:
                    sel = mask & (tc == t) & (sc == s)
                    if sel.any():
                        out[f'{pre}{sn}->{post}{tn}'] = dict(n_edges=int(sel.sum()), total_weight=float(data[sel].sum()), mean_delay_ms=float(steps[sel].mean() * dt))
    return out

def build_unit(candidate, topology_seed, dynamics_seed, network_record=None):
    parent = rt.read(base.PARENT)
    sub, _, _, _, audit, _ = rt.build_frozen_substrate(parent, topology_seed, dynamics_seed, frozen_manifest=network_record)
    baseline_identity = rt.static_identity(sub, audit)
    ne, ni = (sub.n_e, sub.n_i)
    prm = candidate['parameters']
    pos = np.asarray(sub.net['pos'], float)
    reg = sub.extras['placement']
    reference = v2.ee_kernel(sub.params, theta_deg=float(reg['theta_deg']), ar=float(sub.engine['AR']))
    field = core_field(sub.positions_e, sub.positions_i, candidate['centers_mm'], candidate['radii_mm'], candidate['depth_scales'], n_total=ne + ni, quantile_seed=sub.stage['quantile_seed'], core_mean=sub.engine['core_mean'], core_std=sub.engine['core_std'], v_base=sub.engine['v_base'], sheet_mm=float(sub.engine['L']))
    ec, ic = (field['core_index'], field['i_core_index'])
    kernel = v2.ee_kernel(sub.params, theta_deg=reference['theta_deg'], ar=float(sub.engine['AR']), perp_scale=prm['EE_kernel_perp_scale'], parallel_scale=prm['EE_kernel_parallel_scale'], angle_offset_deg=prm['EE_angle_offset_deg'])
    key = base.canonical_sha(dict(topology=topology_seed, baseline=baseline_identity, kernel=kernel, version=shape.VERSION))
    cache = OUT / 'global_graph_cache'
    cache.mkdir(exist_ok=True)
    with (cache / (key + '.lock')).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = cache / (key + '.pkl')
        if path.exists():
            with path.open('rb') as f:
                ampa, ka = pickle.load(f)
        else:
            ampa = sub.net['ampa_by_delay']
            ka = dict(exact_noop=True)
            if not kernel['is_reference']:
                ampa, ka = v2.resample_ee_kernel(ampa, pos, ne, topology_seed=topology_seed, kernel=kernel, p=sub.params)
            tmp = path.with_suffix('.tmp')
            with tmp.open('wb') as f:
                pickle.dump((ampa, ka), f, pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
    ampa, local = shape.local_outgoing(ampa, pos, ne, ec, kernel, candidate.get('outgoing', {}), sub.params, topology_seed)
    da = dict(exact_noop=True)
    if prm['EE_core_to_out_degree_scale'] != 1:
        ampa, da = v2.apply_core_to_out_degree(ampa, pos, ne, ec, prm['EE_core_to_out_degree_scale'], topology_seed=topology_seed, kernel=kernel, p=sub.params)
    ampa, gaba, wa = v2.scale_pathway_blocks(ampa, sub.net['gaba_by_delay'], ne, ec, ic, prm)
    length = max(len(ampa), len(gaba))
    ampa = v2._pad(ampa, length, ampa[0].shape)
    gaba = v2._pad(gaba, length, gaba[0].shape)
    net = {k: v for k, v in sub.net.items() if not (k.startswith('ampa_') or k.startswith('_ampa') or k.startswith('gaba_') or k.startswith('_gaba')) or k in ('ampa_by_delay', 'gaba_by_delay')}
    net.update(ampa_by_delay=ampa, gaba_by_delay=gaba, max_delay_steps=length - 1)
    sub.net = net
    sub.vtheta = field['vtheta']
    sub.delta_vtheta = field['delta_vtheta']
    sub.h_e = (ec >= 0).astype(float)
    sub.h_i = (ic >= 0).astype(float)
    groups = {f'core{chr(65 + k)}E': np.flatnonzero(ec == k) for k in range(candidate['core_count'])}
    groups.update({f'core{chr(65 + k)}I': ne + np.flatnonzero(ic == k) for k in range(candidate['core_count'])})
    groups.update(surroundE=np.flatnonzero(ec < 0), allI=np.arange(ne, ne + ni))
    loading = np.r_[(ec >= 0).astype(float), np.zeros(ni)]
    deterministic = ~(loading > 0)
    identity = dict(positions_E_sha256=base.array_sha256(np.asarray(sub.positions_e, np.float32)), vtheta_sha256=base.array_sha256(np.asarray(sub.vtheta, np.float32)), vtheta_float64_sha256=base.array_sha256(np.asarray(sub.vtheta, np.float64)), h_sha256=base.array_sha256(np.asarray(sub.h_e, np.float32)), core_index_sha256=base.array_sha256(np.asarray(ec, np.int64)), ampa_topology_sha256=base._hash_sparse_bins(ampa, include_data=False), ampa_values_sha256=base._hash_sparse_bins(ampa), gaba_topology_sha256=base._hash_sparse_bins(gaba, include_data=False), gaba_values_sha256=base._hash_sparse_bins(gaba))
    applied = dict(physics=shape.VERSION, candidate=candidate, topology_seed=topology_seed, dynamics_seed=dynamics_seed, reference_kernel=reference, threshold=field['audit'], graph=dict(kernel=kernel, global_cache=str(path), stage_audits=dict(kernel=ka, local_output=local, degree=da, weights=wa), block_summary=pathway_summary(ampa, gaba, ne, ec, ic, sub.params.dt), max_delay_steps=length - 1), identity=identity, baseline_identity=baseline_identity, network_cache=sub.network_cache, input=dict(mode='core_poisson_outside_expected', n_stochastic=int((loading > 0).sum()), n_deterministic=int(deterministic.sum()), outside='exact expected arrivals nu_signal*dt', spatial_ou='off', slow_I_state='off', ZM='off', kick='off'), group_counts={k: int(len(v)) for k, v in groups.items()}, I_core_members=[int((ic == k).sum()) for k in range(candidate['core_count'])])
    assert applied['threshold']['n_raised'] == 0
    return (sub, groups, loading, deterministic, applied, (ec.astype(np.int8), ic.astype(np.int8)))
