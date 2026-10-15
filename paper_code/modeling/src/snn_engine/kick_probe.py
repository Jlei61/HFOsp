"""
Excitability probe for the spatially-structured E-I LIF network.

Question: at a QUIET, sub-oscillation-onset external drive (nu_ext_ratio < 1),
is the network EXCITABLE? I.e. does a localized transient "kick" on the
external drive of a small disk of E neurons trigger an event that
  (a) recruits E neurons BEYOND the kicked patch (spread), and
  (b) returns to baseline (self-limited),
versus fizzling locally, or running into sustained (runaway) activity?

This file adds EXACTLY ONE new mechanism to the core integration loop of
`model.simulate`: a localized transient kick on the external Poisson rate.
`simulate_kick` is a verbatim copy of that loop except for the kick; the
`_verify_*` functions below prove the loop is otherwise identical (pre-kick
trajectories are bit-identical between kick-ON and kick-OFF).

Run:  python kick_probe.py
"""
from __future__ import annotations
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from modeling.src.snn_engine.params import Params, compute_nu_theta
from modeling.src.snn_engine.model import build_network
OUT = OUTPUT_ROOT / 'modeling' / 'kick_probe'
os.makedirs(OUT, exist_ok=True)
R_KICK = 0.15
T_KICK = 150.0
DUR_KICK = 18.0

def _flatten_by_source(by_delay, bins, Nsrc):
    """Flatten the per-delay CSC (N x Nsrc) connectivity into SOURCE-indexed edge
    arrays so a spike scatter is O(#firing-edges) via a SINGLE np.add.at, instead
    of a dense N-add per delay bin (the paper-scale integration-loop bottleneck).
    Returns (indptr[Nsrc+1], dst[nnz], dly[nnz], w[nnz]); source s's out-edges are
    dst/dly/w[indptr[s]:indptr[s+1]]. Results-preserving: same edges/weights."""
    src, dst, dly, w = ([], [], [], [])
    for d in bins:
        coo = by_delay[d].tocoo()
        src.append(coo.col)
        dst.append(coo.row)
        dly.append(np.full(coo.nnz, d, np.int32))
        w.append(coo.data)
    src = np.concatenate(src)
    dst = np.concatenate(dst).astype(np.int64)
    dly = np.concatenate(dly)
    w = np.concatenate(w)
    o = np.argsort(src, kind='stable')
    indptr = np.searchsorted(src[o], np.arange(Nsrc + 1)).astype(np.int64)
    return (indptr, dst[o], dly[o], w[o])

def ee_std_recover_factor(dt, tau_ms):
    """Per-step recovery factor f for x += (1-x)*f, the exact solution of dx/dt=(1-x)/tau over dt."""
    return float(1.0 - np.exp(-dt / tau_ms))

def ee_std_apply(a_w, a_dst, x_per_edge, NE):
    """E->E presynaptic depression: scale each AMPA edge weight by the presynaptic availability
    x_per_edge ONLY for E targets (a_dst < NE); E->I edges (a_dst >= NE) are returned unchanged."""
    return a_w * np.where(a_dst < NE, x_per_edge, 1.0)

def ee_std_source_availability(x_dep, spiking_sources, mode):
    """Availability applied to firing E sources for local or mean-matched STD.

    The global control retains the same latent per-source resource dynamics as
    the local arm, but applies their instantaneous mean to every outgoing E->E
    source.  Thus an identical spike history has an identical mean resource
    dose while only the local arm retains source identity in edge application.
    """
    x_dep = np.asarray(x_dep, dtype=float)
    spiking_sources = np.asarray(spiking_sources, dtype=np.int64)
    if mode == 'local':
        return x_dep[spiking_sources]
    if mode == 'global':
        return np.full(spiking_sources.shape, x_dep.mean(), dtype=float)
    raise ValueError('ee_std_mode must be local or global')

def membrane_step(V, I_E, I_I, decay_V, *, shunt_gaba=False, e_gaba=11.0, g_gaba_scale=0.0):
    """One LIF membrane update. Default (shunt_gaba=False) = current-based LIF, BIT-IDENTICAL
    to the pre-2026-06-19 engine: V_inf = I_E - I_I; V -> V_inf + (V - V_inf)*decay_V.

    shunt_gaba=True = conductance-based SHUNTING inhibition: GABA is a conductance
    g_I = g_gaba_scale*max(I_I,0) pulling V toward the reversal e_gaba, so it gates spike
    initiation regardless of excitatory drive magnitude:
        V_inf = (I_E + g_I*e_gaba) / (1 + g_I);  V -> V_inf + (V - V_inf)*decay_V**(1+g_I).
    (decay_V**(1+g_I) == exp(-dt*(1+g_I)/tau_m) since decay_V = exp(-dt/tau_m): shunting also
    shortens the effective membrane time constant.)"""
    if not shunt_gaba:
        I_net = I_E - I_I
        return I_net + (V - I_net) * decay_V
    g_I = g_gaba_scale * np.maximum(I_I, 0.0)
    V_inf = (I_E + g_I * e_gaba) / (1.0 + g_I)
    return V_inf + (V - V_inf) * decay_V ** (1.0 + g_I)

def simulate_kick(p: Params, net, KICK_BOOST, slow=None, nu_signal_fn=None, verbose=False, kick_center=None, lfp_recorder=None, r_kick=None, t_kick=None, V_th_per_neuron=None, perturb=None, forced_spike_mask=None, forced_spike_ms=None, early_stop_runaway=False, es_thresh_hz=120.0, es_dur_ms=100.0, post_runaway_record_ms=0.0, checkpoint_steps=None, checkpoint_sink=None, resume_state=None, time_offset_ms=0.0, ee_std_u=0.0, ee_std_tau_ms=0.0, dump_ee_std_trace=False, ee_std_trace_maskE=None, t_kick2=None, KICK_BOOST2=0.0, shunt_gaba=False, e_gaba=None, g_gaba_scale=0.0, dump_i_spikes=False, dump_drive=False, dump_pathway_trace=False, pathway_trace_dt_ms=1.0, feedback_gain=0.0, feedback_tau_ms=0.0, dump_fb=False, fb_override_trace=None, ee_std_mode='local', external_e_rate_drive=None, node_accessibility=None, initial_voltage=None, step_observer=None, external_i_state=None):
    """Verbatim copy of model.simulate's integration loop, with ONE addition:
    a localized transient kick on the external Poisson rate. The kick adds
    `KICK_BOOST` (extra external rate, 1/ms) to the E neurons in a disk of
    radius R_KICK about the sheet center, during [T_KICK, T_KICK+DUR_KICK).

    `forced_spike_mask` and `forced_spike_ms` are an off-by-default causal
    assay hook. Together they inject one exact E-neuron presynaptic spike packet
    on the simulation time grid; the packet then follows the ordinary reset,
    refractory, delay-ring and synaptic-weight path.

    `initial_voltage` (ISCP-v1, off by default) is an explicit t=0 membrane
    voltage for every neuron. None keeps the literal `V = V_reset` line. It is
    copied, validated (shape (N,), finite, strictly below the effective
    threshold) and assigned ONCE before the recorder RNG draws, so it consumes
    no simulation RNG and touches no other state; it is mutually exclusive
    with `resume_state`. `step_observer` (off by default) is a read-only
    per-step observer (`observe(t, tm, xi, nu_now, ext, delta_rate, V, I_E,
    I_I, spk)`) used for streaming external-input digests and state traces; it
    must not draw from the simulation RNG or write to the arrays it receives.

    Returns the standard recorders plus per-step inside/outside-disk E-spike
    counts. Use slow=None (epilepsy layer off). KICK_BOOST=0 -> pure control
    (the external block then reduces to model.simulate's, modulo scalar-vs-array
    poisson internals).
    """
    if external_i_state is not None:
        if resume_state is not None or checkpoint_steps is not None:
            raise ValueError('continuous I state checkpointing is not implemented in R1')
        if np.any(external_i_state.indices < net['NE']) or np.any(external_i_state.indices >= net['NE'] + net['NI']):
            raise ValueError('continuous I state may target only inhibitory neurons')
    e_gaba = p.E_gaba if e_gaba is None else e_gaba
    E_A = e_gaba
    rng = net['rng']
    NE, NI = (net['NE'], net['NI'])
    N = NE + NI
    labels = net['labels']
    pos = net['pos']
    ampa = net['ampa_by_delay']
    gaba = net['gaba_by_delay']
    M = net['max_delay_steps'] + 1
    dt = p.dt
    nsteps = int(round(p.T / dt))
    forced_on = forced_spike_mask is not None or forced_spike_ms is not None
    if forced_on:
        if forced_spike_mask is None or forced_spike_ms is None:
            raise ValueError('forced spike mask and time must be provided together')
        forced_spike_mask = np.asarray(forced_spike_mask, bool)
        if forced_spike_mask.shape != (N,):
            raise ValueError('forced spike mask must align to all neurons')
        if np.any(forced_spike_mask[NE:]):
            raise ValueError('forced spike packet must contain excitatory neurons only')
        if not np.any(forced_spike_mask[:NE]):
            raise ValueError('forced spike packet must contain at least one E neuron')
        forced_spike_step = int(round((float(forced_spike_ms) - time_offset_ms) / dt))
        if abs(time_offset_ms + forced_spike_step * dt - float(forced_spike_ms)) > 1e-09:
            raise ValueError('forced spike time must lie on the simulation time grid')
        if forced_spike_step < 0 or forced_spike_step >= nsteps:
            raise ValueError('forced spike time lies outside the simulation')
    else:
        forced_spike_step = None
    forced_spike_collision_count = 0
    external_rate_clip_count = 0
    external_rate_sample_count = 0
    decay_sE = np.exp(-dt / p.tau_r_AMPA)
    decay_IE = np.exp(-dt / p.tau_d_AMPA)
    decay_sI = np.exp(-dt / p.tau_r_GABA)
    decay_II = np.exp(-dt / p.tau_d_GABA)
    tau_m = np.where(labels == 0, p.tau_m_E, p.tau_m_I).astype(np.float64)
    decay_V = np.exp(-dt / tau_m)
    ref_steps = np.where(labels == 0, int(round(p.tau_ref_E / dt)), int(round(p.tau_ref_I / dt))).astype(np.int32)
    ext_incr = tau_m / p.tau_r_AMPA * np.where(labels == 0, p.J_ext_E, p.J_ext_I)
    ampa_bins = [d for d in range(M) if ampa[d].nnz > 0]
    gaba_bins = [d for d in range(M) if gaba[d].nnz > 0]
    if 'ampa_flat' not in net:
        net['ampa_flat'] = _flatten_by_source(ampa, ampa_bins, NE)
        net['gaba_flat'] = _flatten_by_source(gaba, gaba_bins, NI)
    a_indptr, a_dst, a_dly, a_w = net['ampa_flat']
    g_indptr, g_dst, g_dly, g_w = net['gaba_flat']
    nu_theta, _, _ = compute_nu_theta(p)
    if nu_signal_fn is None:
        nu_sig_const = p.nu_ext_ratio * nu_theta
        nu_signal_fn = lambda t_ms: nu_sig_const
    sigma_n_inv_ms = p.sigma_n * 0.001
    sigma_xi = sigma_n_inv_ms * np.sqrt(p.tau_n / 2.0)
    ou_a = np.exp(-dt / p.tau_n)
    ou_b = sigma_xi * np.sqrt(1.0 - ou_a * ou_a)
    xi = 0.0
    center = np.array([p.L / 2, p.L / 2]) if kick_center is None else np.asarray(kick_center, float)
    is_E = labels == 0
    dist_c = np.linalg.norm(pos - center, axis=1)
    rk = R_KICK if r_kick is None else float(r_kick)
    tk = T_KICK if t_kick is None else float(t_kick)
    kick_mask = is_E & (dist_c <= rk)
    outside_mask = is_E & ~kick_mask
    if initial_voltage is None:
        V = np.full(N, p.V_reset, dtype=np.float64)
    else:
        if resume_state is not None:
            raise ValueError('initial_voltage and resume_state are mutually exclusive')
        V = np.array(initial_voltage, dtype=np.float64, copy=True)
        if V.shape != (N,):
            raise ValueError('initial_voltage must have shape (N,)')
        if not np.isfinite(V).all():
            raise ValueError('initial_voltage must be finite')
        _vth_initial = p.V_th if V_th_per_neuron is None else np.asarray(V_th_per_neuron, dtype=np.float64)
        if np.any(V >= _vth_initial):
            raise ValueError('initial_voltage must lie strictly below the effective threshold')
    initial_V = V.copy()
    ref = np.zeros(N, dtype=np.int32)
    s_E = np.zeros(N)
    I_E = np.zeros(N)
    s_I = np.zeros(N)
    I_I = np.zeros(N)
    pathway_trace_on = bool(dump_pathway_trace)
    track_rec = bool(getattr(getattr(slow, 'cfg', None), 'use_SG', False) or pathway_trace_on)
    if track_rec:
        s_E_rec = np.zeros(N)
        I_E_rec = np.zeros(N)
    if pathway_trace_on:
        pathway_trace_stride = int(round(float(pathway_trace_dt_ms) / dt))
        if pathway_trace_stride < 1 or not np.isclose(pathway_trace_stride * dt, float(pathway_trace_dt_ms), atol=1e-12, rtol=0.0):
            raise ValueError('pathway_trace_dt_ms must lie on the simulation grid')
        pathway_trace_size = (nsteps + pathway_trace_stride - 1) // pathway_trace_stride
        pathway_time_ms = np.empty(pathway_trace_size, dtype=np.float32)
        pathway_recurrent_E_to_E_mean = np.empty(pathway_trace_size, dtype=np.float32)
        pathway_recurrent_E_to_I_mean = np.empty(pathway_trace_size, dtype=np.float32)
        pathway_GABA_to_E_mean = np.empty(pathway_trace_size, dtype=np.float32)
        pathway_trace_count = 0
    ring_sE = np.zeros((M, N))
    ring_sI = np.zeros((M, N))
    ee_std_on = ee_std_u > 0.0
    if ee_std_on:
        assert ee_std_tau_ms > 0.0, 'ee_std_u>0 requires ee_std_tau_ms>0'
        if ee_std_mode not in {'local', 'global'}:
            raise ValueError('ee_std_mode must be local or global')
        x_dep = np.ones(NE)
        x_rec_f = ee_std_recover_factor(dt, ee_std_tau_ms)
    fb_dyn = feedback_gain > 0.0 and fb_override_trace is None
    fb_static = fb_override_trace is not None
    fb_on = fb_dyn or fb_static
    if fb_dyn:
        assert feedback_tau_ms > 0.0, 'feedback_gain>0 requires feedback_tau_ms>0'
        assert slow is None, 'A1c rides the default current-based membrane_step; slow must be None'
        assert not shunt_gaba, 'A1c rides the default current-based membrane_step; shunt_gaba must be False'
        r_ema = 0.0
        alpha_fb = float(1.0 - np.exp(-dt / feedback_tau_ms))
        inv_dt_ms = 1.0 / (dt * 0.001)
        I_global_trace = np.zeros(nsteps) if dump_fb else None
    elif fb_static:
        assert slow is None and (not shunt_gaba), 'fb_override rides the default current-based membrane_step'
        fb_override_trace = np.asarray(fb_override_trace, float)
        assert fb_override_trace.shape[0] >= nsteps, 'fb_override_trace shorter than nsteps'
    rate_E = np.zeros(nsteps)
    rate_I = np.zeros(nsteps)
    _es_alpha = 1.0 - np.exp(-dt / 20.0)
    _es_ema = 0.0
    _es_dur = int(round(es_dur_ms / dt))
    _es_run = 0
    _stop_t = nsteps
    _post_steps = int(round(float(post_runaway_record_ms) / dt))
    if _post_steps < 0:
        raise ValueError('post_runaway_record_ms must be non-negative')
    _detect_t = None
    spk_t = []
    spk_i = []
    ras_keepE = rng.choice(NE, size=min(80, NE), replace=False)
    ras_keepI = NE + rng.choice(NI, size=min(20, NI), replace=False)
    ras_keep = np.concatenate([ras_keepE, ras_keepI])
    ras_mask = np.zeros(N, dtype=bool)
    ras_mask[ras_keep] = True
    _resume_step = 0
    if resume_state is not None:
        from modeling.src.snn_engine.checkpoint import restore_external_drive, restore_slow
        if not np.isclose(float(resume_state['absolute_time_ms']), float(time_offset_ms), atol=1e-09):
            raise ValueError('time_offset_ms does not continue the checkpoint clock')
        if resume_state['ring_sE'].shape != ring_sE.shape:
            raise ValueError('checkpoint delay-ring shape differs from this network')
        if bool(resume_state['track_rec']) != bool(track_rec):
            raise ValueError('checkpoint track_rec differs from this run')
        V[:] = resume_state['V']
        ref[:] = resume_state['ref']
        s_E[:] = resume_state['s_E']
        I_E = np.array(resume_state['I_E'], copy=True)
        s_I[:] = resume_state['s_I']
        I_I = np.array(resume_state['I_I'], copy=True)
        ring_sE[:] = resume_state['ring_sE']
        ring_sI[:] = resume_state['ring_sI']
        xi = float(resume_state['xi'])
        rng.bit_generator.state = resume_state['rng_state']
        ras_keep = np.array(resume_state['ras_keep'], copy=True)
        ras_mask = np.zeros(N, dtype=bool)
        ras_mask[ras_keep] = True
        _es_ema = float(resume_state['es_ema'])
        _es_run = int(resume_state['es_run'])
        if track_rec:
            s_E_rec[:] = resume_state['s_E_rec']
            I_E_rec = np.array(resume_state['I_E_rec'], copy=True)
        restore_slow(resume_state, slow)
        restore_external_drive(resume_state, external_e_rate_drive)
        if node_accessibility is not None or resume_state.get('node_accessibility') is not None:
            from modeling.src.snn_engine.checkpoint import restore_node_accessibility
            restore_node_accessibility(resume_state, node_accessibility)
        _resume_step = int(resume_state['step'])
    _ckpt_steps = set() if checkpoint_steps is None else {int(v) for v in checkpoint_steps}
    if _ckpt_steps and checkpoint_sink is None:
        raise ValueError('checkpoint_steps requires a checkpoint_sink')
    spk_inside = np.zeros(nsteps)
    spk_outside = np.zeros(nsteps)
    E_spk_bool = np.zeros((nsteps, NE), dtype=bool)
    I_spk_bool = np.zeros((nsteps, NI), dtype=bool) if dump_i_spikes else None
    _peak_act = -1
    I_E_peak = I_I_peak = None
    lfp_trace = np.zeros((nsteps, len(lfp_recorder.sites))) if lfp_recorder is not None else None
    if dump_ee_std_trace:
        xdep_mean = np.zeros(nsteps)
        xdep_min = np.zeros(nsteps)
        xdep_mask_mean = np.zeros(nsteps) if ee_std_trace_maskE is not None else None

        def _rec_xdep(tt):
            if ee_std_on:
                xdep_mean[tt] = x_dep.mean()
                xdep_min[tt] = x_dep.min()
                if xdep_mask_mean is not None:
                    xdep_mask_mean[tt] = x_dep[ee_std_trace_maskE].mean()
            else:
                xdep_mean[tt] = 1.0
                xdep_min[tt] = 1.0
                if xdep_mask_mean is not None:
                    xdep_mask_mean[tt] = 1.0
    t0 = time.time()
    for t in range(nsteps):
        _abs_step = _resume_step + t
        if _abs_step in _ckpt_steps:
            from modeling.src.snn_engine.checkpoint import capture
            _capture_kwargs = dict(step=_abs_step, absolute_time_ms=time_offset_ms + t * dt, V=V, ref=ref, s_E=s_E, I_E=I_E, s_I=s_I, I_I=I_I, ring_sE=ring_sE, ring_sI=ring_sI, xi=xi, rng=rng, ras_keep=ras_keep, es_ema=_es_ema, es_run=_es_run, track_rec=track_rec, s_E_rec=s_E_rec if track_rec else None, I_E_rec=I_E_rec if track_rec else None, slow=slow, external_drive=external_e_rate_drive)
            if node_accessibility is not None:
                _capture_kwargs['node_accessibility'] = node_accessibility
            checkpoint_sink(_abs_step, capture(**_capture_kwargs))
        tm = time_offset_ms + t * dt
        xi = ou_a * xi + ou_b * rng.standard_normal()
        nu_now = nu_signal_fn(tm) + xi
        if nu_now < 0.0:
            nu_now = 0.0
        s_E *= decay_sE
        s_I *= decay_sI
        slot = _abs_step % M
        if track_rec:
            s_E_rec *= decay_sE
            s_E_rec += ring_sE[slot]
        s_E += ring_sE[slot]
        ring_sE[slot] = 0.0
        s_I += ring_sI[slot]
        ring_sI[slot] = 0.0
        if ee_std_on:
            x_dep += (1.0 - x_dep) * x_rec_f
        nu_vec = np.full(N, max(nu_now, 0.0))
        if external_e_rate_drive is not None:
            delta_rate = np.asarray(external_e_rate_drive.step(tm), float)
            if delta_rate.shape != (NE,) or not np.isfinite(delta_rate).all():
                raise ValueError('external E rate drive must return finite shape (NE,)')
            raw_e_rate = nu_vec[:NE] + delta_rate
            external_rate_clip_count += int(np.count_nonzero(raw_e_rate < 0.0))
            external_rate_sample_count += int(NE)
            nu_vec[:NE] = np.maximum(raw_e_rate, 0.0)
        if tk <= tm < tk + DUR_KICK:
            nu_vec[kick_mask] += KICK_BOOST
        if t_kick2 is not None and t_kick2 <= tm < t_kick2 + DUR_KICK:
            nu_vec[kick_mask] += KICK_BOOST2
        ext = rng.poisson(nu_vec * dt, size=N).astype(np.float64)
        if external_i_state is not None:
            external_i_state.apply(ext, nu_vec, t, dt)
        s_E += ext * ext_incr
        I_E = s_E + (I_E - s_E) * decay_IE
        I_I = s_I + (I_I - s_I) * decay_II
        if track_rec:
            I_E_rec = s_E_rec + (I_E_rec - s_E_rec) * decay_IE
        if pathway_trace_on and t % pathway_trace_stride == 0:
            index = pathway_trace_count
            pathway_time_ms[index] = tm
            pathway_recurrent_E_to_E_mean[index] = np.mean(I_E_rec[:NE])
            pathway_recurrent_E_to_I_mean[index] = np.mean(I_E_rec[NE:])
            pathway_GABA_to_E_mean[index] = np.mean(I_I[:NE])
            pathway_trace_count += 1
        if lfp_trace is not None:
            lfp_trace[t] = lfp_recorder.sample(I_E, I_I)
        if slow is not None:
            if track_rec:
                I_net = slow.apply_currents(I_E, I_I, labels, I_E_rec)
            else:
                I_net = slow.apply_currents(I_E, I_I, labels)
            base_vth = p.V_th if V_th_per_neuron is None else V_th_per_neuron
            V_th_eff = slow.threshold(base_vth)
        else:
            V_th_eff = p.V_th if V_th_per_neuron is None else V_th_per_neuron
        if node_accessibility is not None:
            controller_vth = np.asarray(V_th_eff, dtype=float)
            if controller_vth.ndim == 0:
                controller_vth = np.full(N, float(controller_vth), dtype=float)
            if V_th_per_neuron is not None and np.shares_memory(controller_vth, np.asarray(V_th_per_neuron)):
                controller_vth = np.array(controller_vth, copy=True)
            V_th_eff = np.asarray(node_accessibility.threshold(controller_vth), dtype=float)
            if V_th_eff.shape != (N,) or not np.isfinite(V_th_eff).all():
                raise ValueError('node accessibility threshold must return finite shape (N,)')
            if float(np.min(V_th_eff)) <= float(p.V_reset) + 1.0:
                raise ValueError('node accessibility threshold fell below V_reset + 1 mV')
        if perturb is not None and perturb['kind'] == 'inhibitory_pulse' and (perturb['t0'] <= tm < perturb['t1']):
            _tgt = perturb.get('target_mask', is_E)
            V_th_eff = np.asarray(V_th_eff, float) + perturb['val'] * _tgt
        ref -= 1
        np.maximum(ref, 0, out=ref)
        free = ref == 0
        if slow is not None:
            if hasattr(slow, 'uses_shunt') and slow.uses_shunt():
                g = np.zeros_like(V)
                g[:slow.nE] = slow.shunt_g_at_E()
                V_inf = (I_net + g * E_A) / (1.0 + g)
                Vtmp = V_inf + (V - V_inf) * decay_V ** (1.0 + g)
            else:
                Vtmp = I_net + (V - I_net) * decay_V
        elif fb_on:
            ig_t = feedback_gain * r_ema if fb_dyn else float(fb_override_trace[t])
            I_fb = np.where(is_E, ig_t, 0.0)
            Vtmp = membrane_step(V, I_E, I_I + I_fb, decay_V, shunt_gaba=shunt_gaba, e_gaba=e_gaba, g_gaba_scale=g_gaba_scale)
        else:
            Vtmp = membrane_step(V, I_E, I_I, decay_V, shunt_gaba=shunt_gaba, e_gaba=e_gaba, g_gaba_scale=g_gaba_scale)
        V = np.where(free, Vtmp, p.V_reset)
        spk = free & (V >= (V_th_eff if np.isscalar(V_th_eff) else V_th_eff))
        if forced_spike_step is not None and t == forced_spike_step:
            forced_spike_collision_count = int(np.count_nonzero(spk & forced_spike_mask))
            spk[forced_spike_mask] = True
        V[spk] = p.V_reset
        ref[spk] = ref_steps[spk]
        if slow is not None:
            slow.step(spk, labels, dt)
            if perturb is not None and perturb['kind'] == 'qI_refill' and (perturb['t0'] <= tm < perturb['t1']):
                slow.q_I[:] = perturb['val']
        if node_accessibility is not None:
            node_accessibility.step(spk[:NE], dt)
        rate_E[t] = spk[:NE].sum()
        rate_I[t] = spk[NE:].sum()
        if fb_dyn:
            if dump_fb:
                I_global_trace[t] = feedback_gain * r_ema
            r_ema += alpha_fb * (rate_E[t] / NE * inv_dt_ms - r_ema)
        spk_inside[t] = spk[kick_mask].sum()
        spk_outside[t] = spk[outside_mask].sum()
        E_spk_bool[t] = spk[:NE]
        if step_observer is not None:
            step_observer.observe(t, tm, xi, nu_now, ext, delta_rate if external_e_rate_drive is not None else None, V, I_E, I_I, spk)
        if early_stop_runaway:
            _es_ema += _es_alpha * (rate_E[t] / NE / dt * 1000.0 - _es_ema)
            _es_run = _es_run + 1 if _es_ema >= es_thresh_hz else 0
            if _detect_t is None and _es_run >= _es_dur:
                _detect_t = t + 1
            if _detect_t is not None and t + 1 >= _detect_t + _post_steps:
                _stop_t = t + 1
                if dump_ee_std_trace:
                    _rec_xdep(t)
                break
        if dump_i_spikes:
            I_spk_bool[t] = spk[NE:]
        if dump_drive:
            _na = int(spk.sum())
            if _na > _peak_act:
                _peak_act = _na
                I_E_peak = I_E.copy()
                I_I_peak = I_I.copy()
        if spk.any():
            idx = np.where(spk & ras_mask)[0]
            if idx.size:
                spk_t.append(np.full(idx.size, tm))
                spk_i.append(idx)
            spE = np.where(spk[:NE])[0]
            spI = np.where(spk[NE:])[0]
            if spE.size:
                st = a_indptr[spE]
                cnt = a_indptr[spE + 1] - st
                tot = int(cnt.sum())
                if tot:
                    idx = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(st, cnt)
                    if ee_std_on:
                        x_source = ee_std_source_availability(x_dep, spE, ee_std_mode)
                        x_per_edge = np.repeat(x_source, cnt)
                        w_eff = ee_std_apply(a_w[idx], a_dst[idx], x_per_edge, NE)
                        np.add.at(ring_sE, ((_abs_step + a_dly[idx]) % M, a_dst[idx]), w_eff)
                        x_dep[spE] *= 1.0 - ee_std_u
                    else:
                        np.add.at(ring_sE, ((_abs_step + a_dly[idx]) % M, a_dst[idx]), a_w[idx])
            if spI.size:
                st = g_indptr[spI]
                cnt = g_indptr[spI + 1] - st
                tot = int(cnt.sum())
                if tot:
                    idx = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(st, cnt)
                    np.add.at(ring_sI, ((_abs_step + g_dly[idx]) % M, g_dst[idx]), g_w[idx])
        if verbose and t % max(1, nsteps // 5) == 0:
            print(f'  sim {t}/{nsteps}  ({tm:.0f} ms)  rate_E={rate_E[t] / NE / dt * 1000.0:.1f} Hz  elapsed {time.time() - t0:.1f}s', flush=True)
        if dump_ee_std_trace:
            _rec_xdep(t)
    if _stop_t < nsteps:
        nsteps = _stop_t
        rate_E, rate_I = (rate_E[:nsteps], rate_I[:nsteps])
        E_spk_bool = E_spk_bool[:nsteps]
        spk_inside, spk_outside = (spk_inside[:nsteps], spk_outside[:nsteps])
        if I_spk_bool is not None:
            I_spk_bool = I_spk_bool[:nsteps]
        if lfp_trace is not None:
            lfp_trace = lfp_trace[:nsteps]
        if dump_ee_std_trace:
            xdep_mean = xdep_mean[:nsteps]
            xdep_min = xdep_min[:nsteps]
            if xdep_mask_mean is not None:
                xdep_mask_mean = xdep_mask_mean[:nsteps]
    rate_E_hz = rate_E / NE / dt * 1000.0
    rate_I_hz = rate_I / NI / dt * 1000.0
    res = dict(times=time_offset_ms + np.arange(nsteps) * dt, runaway_early_stop_ms=None if _detect_t is None else round(time_offset_ms + _detect_t * dt, 1), rate_E=rate_E_hz, rate_I=rate_I_hz, spk_inside=spk_inside, spk_outside=spk_outside, E_spk_bool=E_spk_bool, n_inside=int(kick_mask.sum()), n_outside=int(outside_mask.sum()), NE=NE, nu_theta=nu_theta, wall_s=time.time() - t0, forced_spike_step=forced_spike_step, forced_spike_requested_count=0 if forced_spike_step is None else int(forced_spike_mask.sum()), forced_spike_collision_count=int(forced_spike_collision_count), lfp_trace=lfp_trace, lfp_sites=None if lfp_recorder is None else lfp_recorder.sites, external_e_rate_drive=None if external_e_rate_drive is None else {'negative_rate_clip_count': int(external_rate_clip_count), 'rate_sample_count': int(external_rate_sample_count), 'negative_rate_clip_fraction': float(external_rate_clip_count / max(external_rate_sample_count, 1))})
    res['post_runaway_recorded_ms'] = 0.0 if _detect_t is None else round((nsteps - _detect_t) * dt, 1)
    res['initial_V'] = initial_V
    if dump_i_spikes:
        res['I_spk_bool'] = I_spk_bool
    if dump_drive:
        res['I_E_peak'] = I_E_peak
        res['I_I_peak'] = I_I_peak
    if pathway_trace_on:
        res['pathway_trace'] = {'time_ms': pathway_time_ms[:pathway_trace_count], 'recurrent_E_to_E_mean': pathway_recurrent_E_to_E_mean[:pathway_trace_count], 'recurrent_E_to_I_mean': pathway_recurrent_E_to_I_mean[:pathway_trace_count], 'GABA_to_E_mean': pathway_GABA_to_E_mean[:pathway_trace_count]}
    if fb_dyn and dump_fb:
        res['I_global_trace'] = I_global_trace
    elif fb_static and dump_fb:
        res['I_global_trace'] = np.asarray(fb_override_trace[:nsteps], float)
    if dump_ee_std_trace:
        res['xdep_mean'] = xdep_mean
        res['xdep_min'] = xdep_min
        if xdep_mask_mean is not None:
            res['xdep_mask_mean'] = xdep_mask_mean
    return res

def peak_active_fraction(E_spk_bool, dt, t_lo, t_hi, bin_ms=5.0):
    """Max over `bin_ms`-ms bins of (distinct E neurons that spiked in the
    bin) / NE, within [t_lo, t_hi) ms. Uses DISTINCT neurons per bin (OR over
    the bin), not a rate sum, so a cell firing twice in a bin is counted once.
    """
    nsteps, NE = E_spk_bool.shape
    bin_steps = int(round(bin_ms / dt))
    i_lo = int(round(t_lo / dt))
    i_hi = int(round(t_hi / dt))
    best = 0.0
    for b0 in range(i_lo, i_hi, bin_steps):
        b1 = min(b0 + bin_steps, i_hi)
        if b1 <= b0:
            continue
        distinct = E_spk_bool[b0:b1].any(axis=0).sum()
        best = max(best, distinct / NE)
    return float(best)

def window_mean_rate(res, t_lo, t_hi, dt):
    i_lo = int(round(t_lo / dt))
    i_hi = int(round(t_hi / dt))
    return float(res['rate_E'][i_lo:i_hi].mean())

def window_peak_rate(res, t_lo, t_hi, dt):
    i_lo = int(round(t_lo / dt))
    i_hi = int(round(t_hi / dt))
    return float(res['rate_E'][i_lo:i_hi].max())

def window_spike_total(arr, t_lo, t_hi, dt):
    i_lo = int(round(t_lo / dt))
    i_hi = int(round(t_hi / dt))
    return float(arr[i_lo:i_hi].sum())

def compute_metrics(res, dt):
    baseline = window_mean_rate(res, 50.0, 150.0, dt)
    peak = window_peak_rate(res, 150.0, 300.0, dt)
    tail = window_mean_rate(res, 380.0, 450.0, dt)
    returned = tail <= 1.5 * baseline
    inside = window_spike_total(res['spk_inside'], 150.0, 300.0, dt)
    outside = window_spike_total(res['spk_outside'], 150.0, 300.0, dt)
    ratio = outside / inside if inside > 0 else float('nan')
    paf = peak_active_fraction(res['E_spk_bool'], dt, 150.0, 300.0)
    return dict(baseline=baseline, peak=peak, tail=tail, returned=bool(returned), inside=inside, outside=outside, ratio=ratio, peak_active_frac=paf)

def classify(m_on, m_off):
    """Verdict on the kick-ON run; the kick-OFF run is the spontaneous control.

    Spread is judged by ABSOLUTE outside-disk recruitment of the kick-ON run
    relative to its spontaneous control (the OUTSIDE/INSIDE *ratio* is biased by
    the ~13:1 outside:inside E-population imbalance, so ratio>1 alone is not
    spread). 'recruited beyond seed' = outside-disk spikes clearly exceed the
    control's, and the event lifts the global E-rate above baseline.
    """
    out_on, out_off = (m_on['outside'], m_off['outside'])
    peak_on, peak_off = (m_on['peak'], m_off['peak'])
    base_on = m_on['baseline']
    outside_recruit = out_on > 1.5 * max(out_off, 1.0)
    event = peak_on > 2.0 * max(base_on, 1e-06) and peak_on > 1.5 * peak_off
    if not m_on['returned']:
        return 'runaway_sustained'
    if event and (not outside_recruit):
        return 'fizzle'
    if not event:
        if outside_recruit:
            return 'self_limited_spread'
        return 'indistinguishable_from_spontaneous'
    if event and outside_recruit and (peak_on > 1.5 * peak_off):
        return 'self_limited_spread'
    return 'indistinguishable_from_spontaneous'

def verify_pre_kick_identical(res_on, res_off, dt):
    """Before T_KICK nothing differs between kick-ON and kick-OFF (same seed,
    same array-poisson path, kick inactive). Bit-identical E-rate proves the
    loop is otherwise identical / no extra RNG draws were introduced."""
    i_kick = int(round(T_KICK / dt))
    a = res_on['rate_E'][:i_kick]
    b = res_off['rate_E'][:i_kick]
    return (bool(np.array_equal(a, b)), i_kick)

def fresh_run(p, net, KICK_BOOST, kick_center=None):
    net['rng'] = np.random.default_rng(p.seed)
    return simulate_kick(p, net, KICK_BOOST=KICK_BOOST, kick_center=kick_center)

def main():
    dt = 0.1
    base = dict(g=3.6, L=1.0, density=4000.0, T=450.0, seed=1)
    p06 = Params(nu_ext_ratio=0.6, **base)
    nu_theta = compute_nu_theta(p06)[0]
    boosts = {'2x': 2 * nu_theta, '4x': 4 * nu_theta}
    print(f"nu_theta = {nu_theta * 1000.0:.1f} Hz ; nu_signal(0.6) = {0.6 * nu_theta * 1000.0:.1f} Hz ; KICK_BOOST 2x={boosts['2x'] * 1000.0:.0f} Hz 4x={boosts['4x'] * 1000.0:.0f} Hz", flush=True)
    net = build_network(p06, verbose=False)
    rows = []
    results = {}

    def do_pair(p, ratio, boost_label, boost_val):
        res_on = fresh_run(p, net, KICK_BOOST=boost_val)
        res_off = fresh_run(p, net, KICK_BOOST=0.0)
        ok, i_kick = verify_pre_kick_identical(res_on, res_off, dt)
        print(f'[verify] ratio={ratio} {boost_label}: pre-kick (<{T_KICK:.0f}ms, {i_kick} steps) E-rate bit-identical kick-ON vs OFF = {ok}', flush=True)
        m_on = compute_metrics(res_on, dt)
        m_off = compute_metrics(res_off, dt)
        cls_on = classify(m_on, m_off)
        rows.append((ratio, boost_label, boost_val, 'on', m_on, cls_on))
        rows.append((ratio, boost_label, boost_val, 'off', m_off, 'control'))
        results[ratio, boost_label, 'on'] = res_on
        results[ratio, boost_label, 'off'] = res_off
        return (m_on, m_off, cls_on)
    spread_seen = False
    for bl in ('2x', '4x'):
        m_on, m_off, cls = do_pair(p06, 0.6, bl, boosts[bl])
        if m_on['outside'] > 1.5 * max(m_off['outside'], 1.0):
            spread_seen = True
    if not spread_seen:
        print('[info] both 0.6 boosts show no outside-disk recruitment -> running fallback ratio=0.65 @ 4x', flush=True)
        p065 = Params(nu_ext_ratio=0.65, **base)
        do_pair(p065, 0.65, '4x', boosts['4x'])
    else:
        print('[info] outside-disk recruitment seen at 0.6 -> fallback not needed', flush=True)
    print('\n===== METRICS TABLE =====')
    hdr = ('ratio', 'boost', 'kick', 'base_Hz', 'peak_Hz', 'returned', 'in_spk', 'out_spk', 'out/in', 'pk_actfrac', 'class')
    print('{:>5} {:>5} {:>4} {:>8} {:>8} {:>8} {:>8} {:>9} {:>7} {:>10} {:>30}'.format(*hdr))
    for ratio, bl, bv, ko, m, cls in rows:
        print('{:>5} {:>5} {:>4} {:>8.2f} {:>8.2f} {:>8} {:>8.0f} {:>9.0f} {:>7.2f} {:>10.4f} {:>30}'.format(ratio, bl, ko, m['baseline'], m['peak'], str(m['returned']), m['inside'], m['outside'], m['ratio'], m['peak_active_frac'], cls))
    cand = [(k, results[k]) for k in results if k[2] == 'on' and k[0] == 0.6]
    best_key = max(cand, key=lambda kv: window_spike_total(kv[1]['spk_outside'], 150.0, 300.0, dt))[0]
    best_on = results[best_key]
    best_off = results[best_key[0], best_key[1], 'off']
    times = best_on['times']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.axvspan(150.0, 168.0, color='gold', alpha=0.3, label='kick window')
    ax1.plot(times, best_off['rate_E'], color='0.6', lw=1.0, label='kick OFF (control)')
    ax1.plot(times, best_on['rate_E'], color='C3', lw=1.2, label='kick ON')
    ax1.set_ylabel('E rate (Hz)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title(f"Excitability kick probe  (ratio={best_key[0]}, boost={best_key[1]} = {best_on['nu_theta'] * 1000.0 * int(best_key[1][0]):.0f} Hz)")
    ax2.axvspan(150.0, 168.0, color='gold', alpha=0.3)
    ax2.plot(times, best_on['spk_inside'], color='C0', lw=1.0, label=f"inside disk (n={best_on['n_inside']} E)")
    ax2.plot(times, best_on['spk_outside'], color='C1', lw=1.0, label=f"outside disk (n={best_on['n_outside']} E)")
    ax2.set_ylabel('E spikes / step')
    ax2.set_xlabel('time (ms)')
    ax2.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    figpath = os.path.join(OUT, 'kick_probe.png')
    fig.savefig(figpath, dpi=130)
    print(f'\n[figure] saved {figpath}')
if __name__ == '__main__':
    main()
