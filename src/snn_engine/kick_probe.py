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
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from params import Params, compute_nu_theta
from model import build_network

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

# ---- kick geometry / timing (fixed by the spec) ----
R_KICK = 0.15      # mm   radius of the kicked disk (E neurons only)
T_KICK = 150.0     # ms   kick onset
DUR_KICK = 18.0    # ms   kick duration


def _flatten_by_source(by_delay, bins, Nsrc):
    """Flatten the per-delay CSC (N x Nsrc) connectivity into SOURCE-indexed edge
    arrays so a spike scatter is O(#firing-edges) via a SINGLE np.add.at, instead
    of a dense N-add per delay bin (the paper-scale integration-loop bottleneck).
    Returns (indptr[Nsrc+1], dst[nnz], dly[nnz], w[nnz]); source s's out-edges are
    dst/dly/w[indptr[s]:indptr[s+1]]. Results-preserving: same edges/weights."""
    src, dst, dly, w = [], [], [], []
    for d in bins:
        coo = by_delay[d].tocoo()              # row = target (dst), col = source
        src.append(coo.col)
        dst.append(coo.row)
        dly.append(np.full(coo.nnz, d, np.int32))
        w.append(coo.data)
    src = np.concatenate(src)
    dst = np.concatenate(dst).astype(np.int64)
    dly = np.concatenate(dly)
    w = np.concatenate(w)
    o = np.argsort(src, kind="stable")
    indptr = np.searchsorted(src[o], np.arange(Nsrc + 1)).astype(np.int64)
    return indptr, dst[o], dly[o], w[o]


def ee_std_recover_factor(dt, tau_ms):
    """Per-step recovery factor f for x += (1-x)*f, the exact solution of dx/dt=(1-x)/tau over dt."""
    return float(1.0 - np.exp(-dt / tau_ms))


def ee_std_apply(a_w, a_dst, x_per_edge, NE):
    """E->E presynaptic depression: scale each AMPA edge weight by the presynaptic availability
    x_per_edge ONLY for E targets (a_dst < NE); E->I edges (a_dst >= NE) are returned unchanged."""
    return a_w * np.where(a_dst < NE, x_per_edge, 1.0)


def i2e_depression_apply(g_w, g_dst, d_per_edge, NE):
    """Scale inhibitory edges onto E by presynaptic resource; spare I->I."""
    return np.asarray(g_w) * np.where(
        np.asarray(g_dst) < int(NE), np.asarray(d_per_edge), 1.0
    )


def scatter_i2e_emissions_at_spike_time(
    ring_sI, arrival_slots, targets, weights, resources_at_emission, NE
):
    """Enqueue I events after fixing their amplitude at I-spike emission.

    The effective weight, not a reference to the live resource, enters the
    delay ring. Later resource changes cannot reweight an event in flight.
    """
    effective = i2e_depression_apply(
        weights, targets, resources_at_emission, NE
    )
    np.add.at(
        ring_sI,
        (np.asarray(arrival_slots, dtype=np.int64),
         np.asarray(targets, dtype=np.int64)),
        effective,
    )
    return effective


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


def som_shunt_membrane_step(
    V, I_net, I_slow, decay_V, is_E, *, g_scale, e_gaba, z_e=None
):
    """Apply the broad-slow SOM channel as an E-only GABA conductance."""
    if float(g_scale) == 0.0:
        return I_net + (V - I_net) * decay_V
    g = np.zeros_like(V, dtype=float)
    g[: int(np.count_nonzero(is_E))] = (
        float(g_scale) * np.maximum(np.asarray(I_slow)[: np.count_nonzero(is_E)], 0.0)
    )
    if z_e is not None:
        g[: np.count_nonzero(is_E)] *= np.asarray(z_e, dtype=float)
    V_inf = (np.asarray(I_net, dtype=float) + g * float(e_gaba)) / (1.0 + g)
    return V_inf + (np.asarray(V, dtype=float) - V_inf) * np.asarray(decay_V) ** (1.0 + g)


def simulate_kick(p: Params, net, KICK_BOOST, slow=None, nu_signal_fn=None,
                  verbose=False, kick_center=None, lfp_recorder=None, r_kick=None, t_kick=None,
                  V_th_per_neuron=None, perturb=None,
                  early_stop_runaway=False, es_thresh_hz=120.0, es_dur_ms=100.0,
                  ee_std_u=0.0, ee_std_tau_ms=0.0,
                  dump_ee_std_trace=False, ee_std_trace_maskE=None, t_kick2=None, KICK_BOOST2=0.0,
                  shunt_gaba=False, e_gaba=None, g_gaba_scale=0.0,
                  dump_i_spikes=False, dump_drive=False,
                  dump_lfp_components=False,
                  feedback_gain=0.0, feedback_tau_ms=0.0, dump_fb=False, fb_override_trace=None,
                  zm_ckpt=None):
    """Verbatim copy of model.simulate's integration loop, with ONE addition:
    a localized transient kick on the external Poisson rate. The kick adds
    `KICK_BOOST` (extra external rate, 1/ms) to the E neurons in a disk of
    radius R_KICK about the sheet center, during [T_KICK, T_KICK+DUR_KICK).

    Returns the standard recorders plus per-step inside/outside-disk E-spike
    counts. Use slow=None (epilepsy layer off). KICK_BOOST=0 -> pure control
    (the external block then reduces to model.simulate's, modulo scalar-vs-array
    poisson internals).
    """
    e_gaba = p.E_gaba if e_gaba is None else e_gaba   # M2 shunting GABA reversal (=V_reset); default path unused
    E_A = e_gaba                                      # M4-3A a-shunt reversal; reuses the resolved e_gaba default
    rng = net["rng"]
    NE, NI = net["NE"], net["NI"]
    N = NE + NI
    labels = net["labels"]
    pos = net["pos"]
    ampa = net["ampa_by_delay"]
    gaba = net["gaba_by_delay"]
    gaba_slow = net.get("gaba_slow_by_delay")
    dual_gaba_on = gaba_slow is not None
    slow_gaba_shunt_on = bool(
        dual_gaba_on and net.get("gaba_slow_membrane_mode", "current") == "shunt"
    )
    M = net["max_delay_steps"] + 1

    dt = p.dt
    nsteps = int(round(p.T / dt))

    # ---- precomputed decays ---- (identical to model.simulate)
    decay_sE = np.exp(-dt / p.tau_r_AMPA)
    decay_IE = np.exp(-dt / p.tau_d_AMPA)
    decay_sI = np.exp(-dt / p.tau_r_GABA)
    decay_II = np.exp(-dt / p.tau_d_GABA)
    if dual_gaba_on:
        if len(gaba_slow) != M:
            raise ValueError("dual GABA slow bins must match max_delay_steps")
        decay_sI_slow = np.exp(-dt / float(net["gaba_slow_tau_r_ms"]))
        decay_II_slow = np.exp(-dt / float(net["gaba_slow_tau_d_ms"]))
    tau_m = np.where(labels == 0, p.tau_m_E, p.tau_m_I).astype(np.float64)
    decay_V = np.exp(-dt / tau_m)

    ref_steps = np.where(labels == 0,
                         int(round(p.tau_ref_E / dt)),
                         int(round(p.tau_ref_I / dt))).astype(np.int32)

    ext_incr = (tau_m / p.tau_r_AMPA) * np.where(labels == 0, p.J_ext_E, p.J_ext_I)

    ampa_bins = [d for d in range(M) if ampa[d].nnz > 0]
    gaba_bins = [d for d in range(M) if gaba[d].nnz > 0]
    gaba_slow_bins = (
        [d for d in range(M) if gaba_slow[d].nnz > 0]
        if dual_gaba_on else []
    )
    # source-indexed flat edges for O(#firing-edges) scatter; cache on net (reused across runs)
    if "ampa_flat" not in net:
        net["ampa_flat"] = _flatten_by_source(ampa, ampa_bins, NE)
        net["gaba_flat"] = _flatten_by_source(gaba, gaba_bins, NI)
    if dual_gaba_on and "gaba_slow_flat" not in net:
        net["gaba_slow_flat"] = _flatten_by_source(
            gaba_slow, gaba_slow_bins, NI
        )
    a_indptr, a_dst, a_dly, a_w = net["ampa_flat"]
    g_indptr, g_dst, g_dly, g_w = net["gaba_flat"]
    if dual_gaba_on:
        gs_indptr, gs_dst, gs_dly, gs_w = net["gaba_slow_flat"]

    # ---- external drive scale ---- (identical to model.simulate)
    nu_theta, _, _ = compute_nu_theta(p)
    if nu_signal_fn is None:
        nu_sig_const = p.nu_ext_ratio * nu_theta
        nu_signal_fn = lambda t_ms: nu_sig_const
    sigma_n_inv_ms = p.sigma_n * 1e-3
    sigma_xi = sigma_n_inv_ms * np.sqrt(p.tau_n / 2.0)
    ou_a = np.exp(-dt / p.tau_n)
    ou_b = sigma_xi * np.sqrt(1.0 - ou_a * ou_a)
    xi = 0.0

    # ============== THE ONLY NEW MECHANISM: localized kick mask ==============
    center = (np.array([p.L / 2, p.L / 2]) if kick_center is None
              else np.asarray(kick_center, float))   # default = sheet center (back-compat)
    is_E = labels == 0
    dist_c = np.linalg.norm(pos - center, axis=1)
    rk = R_KICK if r_kick is None else float(r_kick)   # patched: kick radius override
    tk = T_KICK if t_kick is None else float(t_kick)    # patched: kick-onset override (early kick)
    kick_mask = is_E & (dist_c <= rk)              # E neurons inside the disk
    outside_mask = is_E & ~kick_mask               # all other E neurons
    # ========================================================================

    # ---- state ---- (identical)
    V = np.full(N, p.V_reset, dtype=np.float64)
    ref = np.zeros(N, dtype=np.int32)
    s_E = np.zeros(N); I_E = np.zeros(N)
    s_I = np.zeros(N); I_I = np.zeros(N)
    if dual_gaba_on:
        s_I_slow = np.zeros(N); I_I_slow = np.zeros(N)
    # ---- M4: recurrent-only AMPA accumulator (OFF by default -> no alloc/float touch on the default
    # path). Tracks the recurrent (delay-ring) component of I_E separately so the shared pool can DIVIDE
    # only recurrent E input; the combined I_E accumulation below is untouched (byte-parity). ----
    conductance_on = bool(
        slow is not None
        and hasattr(slow, "uses_zm_conductance")
        and slow.uses_zm_conductance()
    )
    conductance_homotopy_on = bool(
        slow is not None
        and hasattr(slow, "uses_zm_conductance_homotopy")
        and slow.uses_zm_conductance_homotopy()
    )
    if slow_gaba_shunt_on and (conductance_on or conductance_homotopy_on):
        raise ValueError("SOM shunt cannot be combined with the whole-membrane conductance arms")
    if conductance_on:
        cond_cfg = slow.zm_conductance_config()
        if float(cond_cfg.tau_m_E) != float(p.tau_m_E):
            raise ValueError(
                "conductance tau_m_E must match the simulator E-cell tau_m_E"
            )
    track_rec = bool(
        getattr(getattr(slow, "cfg", None), "use_SG", False)
        or conductance_on
        or conductance_homotopy_on
    )
    if track_rec:
        s_E_rec = np.zeros(N); I_E_rec = np.zeros(N)
    i2e_dep_on = bool(
        slow is not None
        and hasattr(slow, "uses_i2e_depression")
        and slow.uses_i2e_depression()
    )
    ring_sE = np.zeros((M, N))
    ring_sI = np.zeros((M, N))
    if dual_gaba_on:
        ring_sI_slow = np.zeros((M, N))

    # ---- M1: presynaptic E->E short-term depression (default OFF; gated on ee_std_u>0 so the
    # default path is bit-identical to M0 -- no allocation, no new RNG draws, no float touches). ----
    ee_std_on = ee_std_u > 0.0
    if ee_std_on:
        assert ee_std_tau_ms > 0.0, "ee_std_u>0 requires ee_std_tau_ms>0"
        x_dep = np.ones(NE)                                  # availability per E neuron, recovers to 1
        x_rec_f = ee_std_recover_factor(dt, ee_std_tau_ms)

    # ---- A1c: DYNAMIC GLOBAL FEEDBACK RESTRAINT (default OFF; gated on feedback_gain>0 -> bit-parity;
    # no alloc / no RNG / no float touch on the gain=0 path). I_global = feedback_gain * EMA_Hz(global E
    # rate), injected as extra inhibition on E cells only: I_net = I_E - (I_I + I_global). The EMA is an
    # intensive Hz proxy (NE-invariant). NAME: this is a global-feedback-RESTRAINT screen, NOT inhibitory
    # exhaustion (that is the Cl-/z dynamic of A2). ----
    # fb_dyn: dynamic closed-loop feedback (gain*EMA, the default A1c path). fb_static (P1-3 control): a
    # PRESCRIBED per-step I_global(t) injected instead of the EMA (matched-constant or time-shuffled brake),
    # to test whether a terminating run needs the feedback CAUSALLY locked to the rate or just enough DC.
    # fb_override_trace=None => fb_dyn == (gain>0): every pre-existing path is byte-identical (re-bless gate).
    fb_dyn = feedback_gain > 0.0 and fb_override_trace is None
    fb_static = fb_override_trace is not None
    fb_on = fb_dyn or fb_static
    if fb_dyn:
        assert feedback_tau_ms > 0.0, "feedback_gain>0 requires feedback_tau_ms>0"
        assert slow is None, "A1c rides the default current-based membrane_step; slow must be None"
        assert not shunt_gaba, "A1c rides the default current-based membrane_step; shunt_gaba must be False"
        r_ema = 0.0                                           # filtered global E rate proxy (Hz)
        alpha_fb = float(1.0 - np.exp(-dt / feedback_tau_ms)) # exact low-pass coeff (engine convention)
        inv_dt_ms = 1.0 / (dt * 1e-3)                         # spike count -> Hz (matches /NE/dt*1e3 readout)
        I_global_trace = np.zeros(nsteps) if dump_fb else None
    elif fb_static:
        assert slow is None and not shunt_gaba, "fb_override rides the default current-based membrane_step"
        fb_override_trace = np.asarray(fb_override_trace, float)
        assert fb_override_trace.shape[0] >= nsteps, "fb_override_trace shorter than nsteps"

    # ---- recorders ---- (model.simulate's, kept so the RNG stream matches) ----
    rate_E = np.zeros(nsteps); rate_I = np.zeros(nsteps)
    # runaway early-stop (perf): truncate once the 20ms-EMA per-neuron rate is sustained >= threshold -- a
    # runaway is a saturated plateau, so simulating it fully is wasted O(N) scatter. The 20ms EMA matches the
    # verdict's _smooth(20ms) + _first_sustained, so it fires on the SAME runaways (robust to burst oscillation).
    # OFF (early_stop_runaway=False) -> the block below never runs -> no behaviour change.
    _es_alpha = 1.0 - np.exp(-dt / 20.0)                 # 20ms EMA (matches runner _smooth win_ms=20)
    _es_ema = 0.0; _es_dur = int(round(es_dur_ms / dt)); _es_run = 0; _stop_t = nsteps
    spk_t = []; spk_i = []
    ras_keepE = rng.choice(NE, size=min(80, NE), replace=False)
    ras_keepI = NE + rng.choice(NI, size=min(20, NI), replace=False)
    ras_keep = np.concatenate([ras_keepE, ras_keepI])
    ras_mask = np.zeros(N, dtype=bool); ras_mask[ras_keep] = True
    # ---- NEW recorders: spread readout + distinct-neuron active fraction ----
    spk_inside = np.zeros(nsteps)
    spk_outside = np.zeros(nsteps)
    E_spk_bool = np.zeros((nsteps, NE), dtype=bool)   # for distinct-neuron bins
    I_spk_bool = np.zeros((nsteps, NI), dtype=bool) if dump_i_spikes else None  # M2 diag (readout-only)
    _peak_act = -1                                    # M2 diag: snapshot I_E/I_I at the peak-active frame
    I_E_peak = I_I_peak = None
    # optional current-based LFP (|I_E|+|I_I| forward model) at custom sites (Increment-2/3)
    lfp_trace = (np.zeros((nsteps, len(lfp_recorder.sites)))
                 if lfp_recorder is not None else None)
    lfp_current_proxy_trace = (
        np.zeros((nsteps, len(lfp_recorder.sites)))
        if lfp_recorder is not None and conductance_on
        else None
    )
    lfp_exc_trace = (
        np.zeros((nsteps, len(lfp_recorder.sites)))
        if lfp_recorder is not None and dump_lfp_components else None
    )
    lfp_inh_trace = (
        np.zeros((nsteps, len(lfp_recorder.sites)))
        if lfp_recorder is not None and dump_lfp_components else None
    )
    # ---- M4-2: x_dep depression trace (gated; OFF -> no alloc -> byte-parity). Arm 0 (STD off) emits 1.0. ----
    if dump_ee_std_trace:
        xdep_mean = np.zeros(nsteps); xdep_min = np.zeros(nsteps)
        xdep_mask_mean = np.zeros(nsteps) if ee_std_trace_maskE is not None else None

        def _rec_xdep(tt):                       # record x_dep for step tt (called at end-of-step AND at early-stop break)
            if ee_std_on:
                xdep_mean[tt] = x_dep.mean(); xdep_min[tt] = x_dep.min()
                if xdep_mask_mean is not None:
                    xdep_mask_mean[tt] = x_dep[ee_std_trace_maskE].mean()
            else:                                # Arm 0 (STD off): availability == 1.0 everywhere
                xdep_mean[tt] = 1.0; xdep_min[tt] = 1.0
                if xdep_mask_mean is not None:
                    xdep_mask_mean[tt] = 1.0

    # ---- Z/M branch-decision checkpoint controller (spec 2026-07-26 rev3.1 §3.1; OFF by default).
    # zm_ckpt=None -> `_ck_*` are all False/None -> every branch below is skipped -> no new RNG draw,
    # allocation or float op -> byte-parity with the pre-edit engine (tests/test_topic4_zm_checkpoint
    # _hook.py compares against tests/fixtures/topic4_zm_preedit_parity.npz). State packing lives in
    # src/topic4_zm_checkpoint.py; the timestep mathematics below stay single-source. ----
    t_start = 0
    _ck_mean = _ck_dump = False
    _ck_snap = None
    if zm_ckpt is not None:
        _st0 = zm_ckpt.begin(nsteps=nsteps, rng=rng, slow=slow)
        if _st0 is not None:
            V[:] = _st0["V"]; ref[:] = _st0["ref"]
            s_E[:] = _st0["s_E"]; I_E[:] = _st0["I_E"]
            s_I[:] = _st0["s_I"]; I_I[:] = _st0["I_I"]
            ring_sE[:] = _st0["ring_sE"]; ring_sI[:] = _st0["ring_sI"]
            if dual_gaba_on:
                # A legacy checkpoint has no broad-slow history.  Starting the
                # added channel at zero preserves the exact fork state; the
                # old current then washes out under the local-fast kinetics.
                if "s_I_slow" in _st0:
                    s_I_slow[:] = _st0["s_I_slow"]
                    I_I_slow[:] = _st0["I_I_slow"]
                    ring_sI_slow[:] = _st0["ring_sI_slow"]
            xi = float(_st0["xi"]); t_start = int(_st0["t"])
            _es_ema = float(_st0["_es_ema"]); _es_run = int(_st0["_es_run"])
            if track_rec:
                s_E_rec[:] = _st0["s_E_rec"]; I_E_rec[:] = _st0["I_E_rec"]
            if ee_std_on:
                x_dep[:] = _st0["x_dep"]
            if fb_dyn:
                r_ema = float(_st0["r_ema"])
        _ck_mean = zm_ckpt.ext_mean_only
        _ck_dump = zm_ckpt.dump_ext
        _ck_snap = zm_ckpt.snapshot_steps
        if _ck_mean:
            xi = 0.0                      # mean_input_only: drop the OU fluctuation, keep the mean

    t0 = time.time()
    for t in range(nsteps):
        tg = t + t_start                  # ABSOLUTE step (== t when not resuming -> parity)
        tm = tg * dt
        # ----- external homogeneous Poisson rate (Eq 6) -----
        if not _ck_mean:
            xi = ou_a * xi + ou_b * rng.standard_normal()
        nu_now = nu_signal_fn(tm) + xi
        if nu_now < 0.0:
            nu_now = 0.0

        # ----- synaptic gating s: decay, recurrent arrivals, external -----
        s_E *= decay_sE
        s_I *= decay_sI
        if dual_gaba_on:
            s_I_slow *= decay_sI_slow
        slot = tg % M
        if track_rec:
            # HARD CONSTRAINT: read ring_sE[slot] HERE, BEFORE the next line clears it (ring_sE[slot]=0.0).
            # Moving this read after the clear makes I_E_rec read 0 -> divisive term silently no-ops.
            s_E_rec *= decay_sE
            s_E_rec += ring_sE[slot]                     # recurrent (E->E / E->I) arrivals only, pre-zeroing
        s_E += ring_sE[slot]; ring_sE[slot] = 0.0
        s_I += ring_sI[slot]; ring_sI[slot] = 0.0
        if dual_gaba_on:
            s_I_slow += ring_sI_slow[slot]
            ring_sI_slow[slot] = 0.0
        if ee_std_on:
            x_dep += (1.0 - x_dep) * x_rec_f                 # M1: recover availability toward 1 each step
        # ===================== KICK: the only change vs model.simulate =====================
        nu_vec = np.full(N, max(nu_now, 0.0))
        if tk <= tm < tk + DUR_KICK:
            nu_vec[kick_mask] += KICK_BOOST          # extra external rate, units 1/ms
        if t_kick2 is not None and t_kick2 <= tm < t_kick2 + DUR_KICK:
            nu_vec[kick_mask] += KICK_BOOST2         # M4-2 post-offset retrigger probe (same source core; None -> parity)
        if _ck_mean:
            ext = nu_vec * dt                        # deterministic external MEAN (no Poisson draw)
        else:
            ext = rng.poisson(nu_vec * dt, size=N).astype(np.float64)
        if _ck_dump:                                 # paired-noise audit: the drive actually delivered
            zm_ckpt.ext_nu[t] = nu_now; zm_ckpt.ext_sum[t] = ext.sum()
        s_E += ext * ext_incr
        # ==================================================================================

        # ----- synaptic currents (low-pass of s) -----
        I_E = s_E + (I_E - s_E) * decay_IE
        I_I = s_I + (I_I - s_I) * decay_II
        if dual_gaba_on:
            I_I_slow = s_I_slow + (I_I_slow - s_I_slow) * decay_II_slow
            I_I_lfp = I_I + I_I_slow
            I_I_effective = I_I if slow_gaba_shunt_on else I_I_lfp
        else:
            I_I_lfp = I_I
            I_I_effective = I_I
        if track_rec:
            I_E_rec = s_E_rec + (I_E_rec - s_E_rec) * decay_IE
        if lfp_trace is not None and not conductance_on:  # current-based LFP at custom sites
            lfp_trace[t] = lfp_recorder.sample(I_E, I_I_lfp)
            if dump_lfp_components:
                lfp_exc_trace[t], lfp_inh_trace[t] = lfp_recorder.sample_components(
                    I_E, I_I_lfp
                )

        # slow layer off (slow=None)
        conductance_state = None
        conductance_homotopy_state = None
        if slow is not None:
            if conductance_on:
                # Phase-D: raw drives enter the unit-safe conductance membrane
                # directly.  Do NOT first apply z/m or the old S_G divisor.
                conductance_state = slow.zm_conductance_step(
                    V, I_E, I_I_effective, decay_V
                )
                if lfp_trace is not None:
                    lfp_trace[t] = lfp_recorder.sample(
                        conductance_state["I_exc"],
                        conductance_state["I_inh"],
                    )
                    lfp_current_proxy_trace[t] = lfp_recorder.sample(
                        I_E, I_I_effective
                    )
            elif track_rec:
                I_net = slow.apply_currents(I_E, I_I_effective, labels, I_E_rec)
            else:
                I_net = slow.apply_currents(I_E, I_I_effective, labels)
            if conductance_homotopy_on:
                I_E_homotopy = I_E
                if (
                    getattr(slow.cfg, "use_mode_H", False)
                    and slow.cfg.rho_mode_H > 0.0
                ):
                    # H acts only on recurrent E input.  The native branch has
                    # already received the same gain in apply_currents(); the
                    # conductance endpoint receives it before kappa_E mapping.
                    I_E_homotopy = I_E.copy()
                    I_E_homotopy[: slow.nE] += (
                        I_E_rec[: slow.nE] * slow.mode_H_gain_at_E()
                    )
                conductance_homotopy_state = slow.zm_conductance_homotopy_step(
                    V, I_E_homotopy, I_I_effective, I_net, decay_V
                )
            # off-by-default hook: under slow, use the per-neuron threshold substrate when provided
            # (lets z/g_K ride a heterogeneous core); V_th_per_neuron=None -> uniform p.V_th (unchanged).
            base_vth = p.V_th if V_th_per_neuron is None else V_th_per_neuron
            V_th_eff = slow.threshold(base_vth)
        else:
            V_th_eff = p.V_th if V_th_per_neuron is None else V_th_per_neuron

        # off-by-default reversibility/basin perturbation (perturb=None -> no float touched -> byte-parity):
        # inhibitory_pulse = transiently RAISE the E threshold (suppress E firing) WITHOUT touching q_I.
        if perturb is not None and perturb["kind"] == "inhibitory_pulse" and perturb["t0"] <= tm < perturb["t1"]:
            _tgt = perturb.get("target_mask", is_E)          # spatial stim locus (default all E -> byte-parity)
            V_th_eff = np.asarray(V_th_eff, float) + perturb["val"] * _tgt

        # ----- membrane (Eq 3) + refractory -----
        ref -= 1
        np.maximum(ref, 0, out=ref)
        free = ref == 0
        if slow is not None:
            # M4-3A conductance a-shunt (form A). uses_shunt() is SpatialSlowField-only (Task 4); the
            # OTHER "slow" implementers (FrozenSlowVars/SlowVars, RegionalResource) have no a-shunt
            # concept, so hasattr guards them onto the literal parity path below (plan-correction for
            # polymorphic slow=; mirrors the getattr(...,'use_SG',False) duck-typing a few lines up).
            if conductance_on:
                Vtmp = conductance_state["V_next"]
            elif conductance_homotopy_on:
                Vtmp = conductance_homotopy_state["V_next"]
            elif slow_gaba_shunt_on:
                z_e = (
                    slow.z[:NE]
                    if getattr(getattr(slow, "cfg", None), "use_z", False)
                    else None
                )
                Vtmp = som_shunt_membrane_step(
                    V,
                    I_net,
                    I_I_slow,
                    decay_V,
                    is_E,
                    g_scale=float(net["gaba_slow_shunt_scale"]),
                    e_gaba=float(net["gaba_slow_e_gaba_mv"]),
                    z_e=z_e,
                )
            elif hasattr(slow, "uses_shunt") and slow.uses_shunt():
                g = np.zeros_like(V)
                g[:slow.nE] = slow.shunt_g_at_E()                  # E-only; I cells g=0 -> parity
                V_inf = (I_net + g * E_A) / (1.0 + g)              # a NEVER divides signed net (reversal-clamped)
                Vtmp = V_inf + (V - V_inf) * decay_V ** (1.0 + g)
            else:
                Vtmp = I_net + (V - I_net) * decay_V               # literal pre-change path -> byte parity
        elif fb_on:
            # A1c: extra global inhibition on E cells -> effective inhibition (I_I + I_global) on E only.
            ig_t = feedback_gain * r_ema if fb_dyn else float(fb_override_trace[t])
            I_fb = np.where(is_E, ig_t, 0.0)
            Vtmp = membrane_step(V, I_E, I_I_effective + I_fb, decay_V,
                                 shunt_gaba=shunt_gaba, e_gaba=e_gaba, g_gaba_scale=g_gaba_scale)
        else:
            Vtmp = membrane_step(V, I_E, I_I_effective, decay_V,      # literal pre-edit call (gain=0 parity)
                                 shunt_gaba=shunt_gaba, e_gaba=e_gaba, g_gaba_scale=g_gaba_scale)
        V = np.where(free, Vtmp, p.V_reset)
        spk = free & (V >= (V_th_eff if np.isscalar(V_th_eff) else V_th_eff))
        V[spk] = p.V_reset
        ref[spk] = ref_steps[spk]

        if slow is not None:
            slow.step(spk, labels, dt)
            # qI_refill = transiently RESET the inhibitory resource to `val` (directly repair the slow var).
            if perturb is not None and perturb["kind"] == "qI_refill" and perturb["t0"] <= tm < perturb["t1"]:
                slow.q_I[:] = perturb["val"]

        # ----- record -----
        rate_E[t] = spk[:NE].sum()
        rate_I[t] = spk[NE:].sum()
        if fb_dyn:
            if dump_fb:
                I_global_trace[t] = feedback_gain * r_ema    # the I_global USED at step t (pre-update)
            r_ema += alpha_fb * (rate_E[t] / NE * inv_dt_ms - r_ema)   # EMA, consumed at TOP of t+1
        # NEW: spread + distinct-neuron readout
        spk_inside[t] = spk[kick_mask].sum()
        spk_outside[t] = spk[outside_mask].sum()
        E_spk_bool[t] = spk[:NE]
        if early_stop_runaway:                                       # runaway detected -> break before the O(N) scatter
            _es_ema += _es_alpha * (rate_E[t] / NE / dt * 1e3 - _es_ema)   # 20ms-EMA per-neuron rate (Hz)
            _es_run = _es_run + 1 if _es_ema >= es_thresh_hz else 0
            if _es_run >= _es_dur:
                _stop_t = t + 1
                if dump_ee_std_trace:            # break frame is KEPT (_stop_t=t+1) -> write its trace, else phantom 0
                    _rec_xdep(t)                 # (pre-depletion: this frame's O(N) scatter is skipped by design)
                break
        if dump_i_spikes:
            I_spk_bool[t] = spk[NE:]
        if dump_drive:
            _na = int(spk.sum())
            if _na > _peak_act:
                _peak_act = _na
                I_E_peak = I_E.copy()
                I_I_peak = I_I_lfp.copy()
        if spk.any():
            idx = np.where(spk & ras_mask)[0]
            if idx.size:
                spk_t.append(np.full(idx.size, tm))
                spk_i.append(idx)
            # ----- scatter spikes into delay ring -----
            # PERF (2026-06-15): the firers' synapses are SPARSE -- scatter only the
            # nonzero target rows (np.add.at on the column-gathered COO) instead of
            # building+adding a DENSE N-vector for every delay bin. At paper scale this
            # is the integration-loop bottleneck (a dense N add per ~206 bins per step).
            # Results-preserving: same column-gathered weights, only zero-adds skipped;
            # verified spike-identical against the pre-opt engine (tests/test_snn_engine_scatter.py).
            spE = np.where(spk[:NE])[0]
            spI = np.where(spk[NE:])[0]
            if spE.size:
                st = a_indptr[spE]; cnt = a_indptr[spE + 1] - st; tot = int(cnt.sum())
                if tot:
                    idx = (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                           + np.repeat(st, cnt))            # concat of each firer's edge range
                    if ee_std_on:
                        # M1: E->E edges scaled by the presynaptic availability at spike time (x_j(t-));
                        # E->I edges untouched. Then deplete the firers (vesicle use): x_j(t+)=x_j*(1-U).
                        x_per_edge = np.repeat(x_dep[spE], cnt)
                        w_eff = ee_std_apply(a_w[idx], a_dst[idx], x_per_edge, NE)
                        np.add.at(ring_sE, ((tg + a_dly[idx]) % M, a_dst[idx]), w_eff)
                        x_dep[spE] *= (1.0 - ee_std_u)
                    else:
                        np.add.at(ring_sE, ((tg + a_dly[idx]) % M, a_dst[idx]), a_w[idx])
            if spI.size:
                st = g_indptr[spI]; cnt = g_indptr[spI + 1] - st; tot = int(cnt.sum())
                if tot:
                    idx = (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                           + np.repeat(st, cnt))
                    if i2e_dep_on:
                        # Presynaptic resource scales only I->E edges.  I->I
                        # weights remain native, preserving inhibitory local
                        # containment while active I sources transiently lose
                        # efficacy onto E cells.
                        d_per_edge = np.repeat(
                            slow.i2e_resource_at_sources(spI), cnt
                        )
                        w_eff = scatter_i2e_emissions_at_spike_time(
                            ring_sI,
                            (tg + g_dly[idx]) % M,
                            g_dst[idx],
                            g_w[idx],
                            d_per_edge,
                            NE,
                        )
                    else:
                        np.add.at(ring_sI, ((tg + g_dly[idx]) % M, g_dst[idx]), g_w[idx])
                if dual_gaba_on:
                    st_s = gs_indptr[spI]
                    cnt_s = gs_indptr[spI + 1] - st_s
                    tot_s = int(cnt_s.sum())
                    if tot_s:
                        idx_s = (
                            np.arange(tot_s)
                            - np.repeat(np.cumsum(cnt_s) - cnt_s, cnt_s)
                            + np.repeat(st_s, cnt_s)
                        )
                        if i2e_dep_on:
                            d_per_edge_s = np.repeat(
                                slow.i2e_resource_at_sources(spI), cnt_s
                            )
                            scatter_i2e_emissions_at_spike_time(
                                ring_sI_slow,
                                (tg + gs_dly[idx_s]) % M,
                                gs_dst[idx_s],
                                gs_w[idx_s],
                                d_per_edge_s,
                                NE,
                            )
                        else:
                            np.add.at(
                                ring_sI_slow,
                                ((tg + gs_dly[idx_s]) % M, gs_dst[idx_s]),
                                gs_w[idx_s],
                            )
                if i2e_dep_on:
                    slow.consume_i2e_sources(spI)

        if verbose and (t % max(1, nsteps // 5) == 0):
            print(f"  sim {t}/{nsteps}  ({tm:.0f} ms)  "
                  f"rate_E={rate_E[t]/NE/dt*1e3:.1f} Hz  elapsed {time.time()-t0:.1f}s",
                  flush=True)

        if dump_ee_std_trace:                    # normal frame: record post-depletion (:378), not post-recovery (:259)
            _rec_xdep(t)

        if _ck_snap is not None and (tg + 1) in _ck_snap:   # END-of-step state = what step tg+1 needs
            zm_ckpt.take(tg + 1, store=True, rng=rng, slow=slow, V=V, ref=ref, s_E=s_E, I_E=I_E,
                         s_I=s_I, I_I=I_I, ring_sE=ring_sE, ring_sI=ring_sI, xi=xi,
                         _es_ema=_es_ema, _es_run=_es_run,
                         s_E_rec=(s_E_rec if track_rec else None),
                         I_E_rec=(I_E_rec if track_rec else None),
                         x_dep=(x_dep if ee_std_on else None),
                         s_I_slow=(s_I_slow if dual_gaba_on else None),
                         I_I_slow=(I_I_slow if dual_gaba_on else None),
                         ring_sI_slow=(ring_sI_slow if dual_gaba_on else None),
                         r_ema=(r_ema if fb_dyn else None))

    if zm_ckpt is not None and zm_ckpt.return_final_state:
        # a runaway early-stop breaks BEFORE the delay scatter of the break frame, so that state is
        # mid-step and must never be forked from -> fail closed rather than hand back a bad snapshot
        if _stop_t < nsteps:
            zm_ckpt.final_truncated = True
        else:
            zm_ckpt.take(t_start + nsteps, store=False, rng=rng, slow=slow, V=V, ref=ref, s_E=s_E,
                         I_E=I_E, s_I=s_I, I_I=I_I, ring_sE=ring_sE, ring_sI=ring_sI, xi=xi,
                         _es_ema=_es_ema, _es_run=_es_run,
                         s_E_rec=(s_E_rec if track_rec else None),
                         I_E_rec=(I_E_rec if track_rec else None),
                         x_dep=(x_dep if ee_std_on else None),
                         s_I_slow=(s_I_slow if dual_gaba_on else None),
                         I_I_slow=(I_I_slow if dual_gaba_on else None),
                         ring_sI_slow=(ring_sI_slow if dual_gaba_on else None),
                         r_ema=(r_ema if fb_dyn else None))

    if _stop_t < nsteps:                                             # runaway early-stop: truncate per-step arrays
        nsteps = _stop_t
        rate_E, rate_I = rate_E[:nsteps], rate_I[:nsteps]
        E_spk_bool = E_spk_bool[:nsteps]
        spk_inside, spk_outside = spk_inside[:nsteps], spk_outside[:nsteps]
        if I_spk_bool is not None:
            I_spk_bool = I_spk_bool[:nsteps]
        if lfp_trace is not None:
            lfp_trace = lfp_trace[:nsteps]
        if lfp_current_proxy_trace is not None:
            lfp_current_proxy_trace = lfp_current_proxy_trace[:nsteps]
        if lfp_exc_trace is not None:
            lfp_exc_trace = lfp_exc_trace[:nsteps]
            lfp_inh_trace = lfp_inh_trace[:nsteps]
        if dump_ee_std_trace:
            xdep_mean = xdep_mean[:nsteps]; xdep_min = xdep_min[:nsteps]
            if xdep_mask_mean is not None:
                xdep_mask_mean = xdep_mask_mean[:nsteps]
    rate_E_hz = rate_E / NE / dt * 1e3
    rate_I_hz = rate_I / NI / dt * 1e3
    res = dict(
        times=np.arange(nsteps) * dt,
        runaway_early_stop_ms=(None if _stop_t >= (int(round(p.T / dt))) else round(_stop_t * dt, 1)),
        rate_E=rate_E_hz, rate_I=rate_I_hz,
        spk_inside=spk_inside, spk_outside=spk_outside,
        E_spk_bool=E_spk_bool,
        n_inside=int(kick_mask.sum()), n_outside=int(outside_mask.sum()),
        NE=NE, nu_theta=nu_theta, wall_s=time.time() - t0,
        lfp_trace=lfp_trace,                                    # (nsteps, n_sites) or None
        lfp_current_proxy_trace=lfp_current_proxy_trace,
        lfp_exc_trace=lfp_exc_trace,
        lfp_inh_trace=lfp_inh_trace,
        lfp_sites=(None if lfp_recorder is None else lfp_recorder.sites),
    )
    if zm_ckpt is not None:
        res["t_start"] = t_start
        if zm_ckpt.dump_ext:
            res["zm_ext_nu"] = zm_ckpt.ext_nu[:nsteps]
            res["zm_ext_sum"] = zm_ckpt.ext_sum[:nsteps]
    if dump_i_spikes:
        res["I_spk_bool"] = I_spk_bool
    if dump_drive:
        res["I_E_peak"] = I_E_peak
        res["I_I_peak"] = I_I_peak
    if fb_dyn and dump_fb:
        res["I_global_trace"] = I_global_trace                  # (nsteps,) the per-step scalar I_global
    elif fb_static and dump_fb:
        res["I_global_trace"] = np.asarray(fb_override_trace[:nsteps], float)  # the prescribed brake (control)
    if dump_ee_std_trace:
        res["xdep_mean"] = xdep_mean
        res["xdep_min"] = xdep_min
        if xdep_mask_mean is not None:
            res["xdep_mask_mean"] = xdep_mask_mean
    return res


# ======================= metrics =======================
def peak_active_fraction(E_spk_bool, dt, t_lo, t_hi, bin_ms=5.0):
    """Max over `bin_ms`-ms bins of (distinct E neurons that spiked in the
    bin) / NE, within [t_lo, t_hi) ms. Uses DISTINCT neurons per bin (OR over
    the bin), not a rate sum, so a cell firing twice in a bin is counted once.
    """
    nsteps, NE = E_spk_bool.shape
    bin_steps = int(round(bin_ms / dt))
    i_lo = int(round(t_lo / dt)); i_hi = int(round(t_hi / dt))
    best = 0.0
    for b0 in range(i_lo, i_hi, bin_steps):
        b1 = min(b0 + bin_steps, i_hi)
        if b1 <= b0:
            continue
        distinct = E_spk_bool[b0:b1].any(axis=0).sum()
        best = max(best, distinct / NE)
    return float(best)


def window_mean_rate(res, t_lo, t_hi, dt):
    i_lo = int(round(t_lo / dt)); i_hi = int(round(t_hi / dt))
    return float(res["rate_E"][i_lo:i_hi].mean())


def window_peak_rate(res, t_lo, t_hi, dt):
    i_lo = int(round(t_lo / dt)); i_hi = int(round(t_hi / dt))
    return float(res["rate_E"][i_lo:i_hi].max())


def window_spike_total(arr, t_lo, t_hi, dt):
    i_lo = int(round(t_lo / dt)); i_hi = int(round(t_hi / dt))
    return float(arr[i_lo:i_hi].sum())


def compute_metrics(res, dt):
    baseline = window_mean_rate(res, 50.0, 150.0, dt)
    peak = window_peak_rate(res, 150.0, 300.0, dt)
    tail = window_mean_rate(res, 380.0, 450.0, dt)
    returned = tail <= 1.5 * baseline
    inside = window_spike_total(res["spk_inside"], 150.0, 300.0, dt)
    outside = window_spike_total(res["spk_outside"], 150.0, 300.0, dt)
    ratio = (outside / inside) if inside > 0 else float("nan")
    paf = peak_active_fraction(res["E_spk_bool"], dt, 150.0, 300.0)
    return dict(baseline=baseline, peak=peak, tail=tail, returned=bool(returned),
                inside=inside, outside=outside, ratio=ratio, peak_active_frac=paf)


def classify(m_on, m_off):
    """Verdict on the kick-ON run; the kick-OFF run is the spontaneous control.

    Spread is judged by ABSOLUTE outside-disk recruitment of the kick-ON run
    relative to its spontaneous control (the OUTSIDE/INSIDE *ratio* is biased by
    the ~13:1 outside:inside E-population imbalance, so ratio>1 alone is not
    spread). 'recruited beyond seed' = outside-disk spikes clearly exceed the
    control's, and the event lifts the global E-rate above baseline.
    """
    out_on, out_off = m_on["outside"], m_off["outside"]
    peak_on, peak_off = m_on["peak"], m_off["peak"]
    base_on = m_on["baseline"]

    # Did the kick raise outside-disk recruitment well beyond the control?
    outside_recruit = out_on > 1.5 * max(out_off, 1.0)
    # Did the kick produce a transient E-rate event clearly above its own
    # baseline AND clearly above the spontaneous control's peak?
    event = (peak_on > 2.0 * max(base_on, 1e-6)) and (peak_on > 1.5 * peak_off)

    if not m_on["returned"]:
        return "runaway_sustained"
    if event and not outside_recruit:
        # an event happened but did not escape the seed patch
        return "fizzle"
    if not event:
        # no clear transient relative to the control
        if outside_recruit:
            return "self_limited_spread"   # rare; recruitment without rate event
        return "indistinguishable_from_spontaneous"
    # event + outside recruitment + returned
    if event and outside_recruit and (peak_on > 1.5 * peak_off):
        return "self_limited_spread"
    return "indistinguishable_from_spontaneous"


# ======================= verification =======================
def verify_pre_kick_identical(res_on, res_off, dt):
    """Before T_KICK nothing differs between kick-ON and kick-OFF (same seed,
    same array-poisson path, kick inactive). Bit-identical E-rate proves the
    loop is otherwise identical / no extra RNG draws were introduced."""
    i_kick = int(round(T_KICK / dt))
    a = res_on["rate_E"][:i_kick]
    b = res_off["rate_E"][:i_kick]
    return bool(np.array_equal(a, b)), i_kick


# ======================= runs =======================
def fresh_run(p, net, KICK_BOOST, kick_center=None):
    net["rng"] = np.random.default_rng(p.seed)   # seed-match each run
    return simulate_kick(p, net, KICK_BOOST=KICK_BOOST, kick_center=kick_center)


def main():
    dt = 0.1
    base = dict(g=3.6, L=1.0, density=4000.0, T=450.0, seed=1)
    p06 = Params(nu_ext_ratio=0.6, **base)
    nu_theta = compute_nu_theta(p06)[0]
    boosts = {"2x": 2 * nu_theta, "4x": 4 * nu_theta}

    print(f"nu_theta = {nu_theta*1e3:.1f} Hz ; nu_signal(0.6) = "
          f"{0.6*nu_theta*1e3:.1f} Hz ; KICK_BOOST 2x={boosts['2x']*1e3:.0f} Hz "
          f"4x={boosts['4x']*1e3:.0f} Hz", flush=True)

    # Build the network ONCE; reuse across all runs.
    net = build_network(p06, verbose=False)

    rows = []          # (ratio, boost_label, boost_val, kick_on/off, metrics)
    results = {}       # (ratio, boost_label, on/off) -> res

    def do_pair(p, ratio, boost_label, boost_val):
        res_on = fresh_run(p, net, KICK_BOOST=boost_val)
        res_off = fresh_run(p, net, KICK_BOOST=0.0)
        ok, i_kick = verify_pre_kick_identical(res_on, res_off, dt)
        print(f"[verify] ratio={ratio} {boost_label}: pre-kick (<{T_KICK:.0f}ms, "
              f"{i_kick} steps) E-rate bit-identical kick-ON vs OFF = {ok}", flush=True)
        m_on = compute_metrics(res_on, dt)
        m_off = compute_metrics(res_off, dt)
        cls_on = classify(m_on, m_off)
        # OFF classification: is the control itself event-like? -> reference only
        rows.append((ratio, boost_label, boost_val, "on", m_on, cls_on))
        rows.append((ratio, boost_label, boost_val, "off", m_off, "control"))
        results[(ratio, boost_label, "on")] = res_on
        results[(ratio, boost_label, "off")] = res_off
        return m_on, m_off, cls_on

    # ---- ratio 0.6, both boosts ----
    spread_seen = False
    for bl in ("2x", "4x"):
        m_on, m_off, cls = do_pair(p06, 0.6, bl, boosts[bl])
        # outside recruitment beyond control?
        if m_on["outside"] > 1.5 * max(m_off["outside"], 1.0):
            spread_seen = True

    # ---- fallback: if BOTH 0.6 boosts fizzle (no outside recruitment), run 0.65 @ 4x ----
    if not spread_seen:
        print("[info] both 0.6 boosts show no outside-disk recruitment -> "
              "running fallback ratio=0.65 @ 4x", flush=True)
        p065 = Params(nu_ext_ratio=0.65, **base)
        # nu_theta and KICK_BOOST unchanged (compute_nu_theta independent of ratio);
        # only nu_sig_const changes inside simulate_kick.
        do_pair(p065, 0.65, "4x", boosts["4x"])
    else:
        print("[info] outside-disk recruitment seen at 0.6 -> fallback not needed",
              flush=True)

    # ======================= table =======================
    print("\n===== METRICS TABLE =====")
    hdr = ("ratio", "boost", "kick", "base_Hz", "peak_Hz", "returned",
           "in_spk", "out_spk", "out/in", "pk_actfrac", "class")
    print("{:>5} {:>5} {:>4} {:>8} {:>8} {:>8} {:>8} {:>9} {:>7} {:>10} {:>30}".format(*hdr))
    for ratio, bl, bv, ko, m, cls in rows:
        print("{:>5} {:>5} {:>4} {:>8.2f} {:>8.2f} {:>8} {:>8.0f} {:>9.0f} "
              "{:>7.2f} {:>10.4f} {:>30}".format(
                  ratio, bl, ko, m["baseline"], m["peak"], str(m["returned"]),
                  m["inside"], m["outside"], m["ratio"], m["peak_active_frac"], cls))

    # ======================= figure =======================
    # Pick the "best boost" kick-ON run at ratio 0.6: the one with the largest
    # outside-disk recruitment.
    cand = [(k, results[k]) for k in results if k[2] == "on" and k[0] == 0.6]
    best_key = max(cand, key=lambda kv: window_spike_total(
        kv[1]["spk_outside"], 150.0, 300.0, dt))[0]
    best_on = results[best_key]
    best_off = results[(best_key[0], best_key[1], "off")]
    times = best_on["times"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.axvspan(150.0, 168.0, color="gold", alpha=0.3, label="kick window")
    ax1.plot(times, best_off["rate_E"], color="0.6", lw=1.0, label="kick OFF (control)")
    ax1.plot(times, best_on["rate_E"], color="C3", lw=1.2, label="kick ON")
    ax1.set_ylabel("E rate (Hz)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(f"Excitability kick probe  (ratio={best_key[0]}, "
                  f"boost={best_key[1]} = {best_on['nu_theta']*1e3*int(best_key[1][0]):.0f} Hz)")

    # bottom: inside vs outside disk E-spike count per step (kick-ON best run)
    ax2.axvspan(150.0, 168.0, color="gold", alpha=0.3)
    ax2.plot(times, best_on["spk_inside"], color="C0", lw=1.0,
             label=f"inside disk (n={best_on['n_inside']} E)")
    ax2.plot(times, best_on["spk_outside"], color="C1", lw=1.0,
             label=f"outside disk (n={best_on['n_outside']} E)")
    ax2.set_ylabel("E spikes / step")
    ax2.set_xlabel("time (ms)")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    figpath = os.path.join(OUT, "kick_probe.png")
    fig.savefig(figpath, dpi=130)
    print(f"\n[figure] saved {figpath}")


if __name__ == "__main__":
    main()
