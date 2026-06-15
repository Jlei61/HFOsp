"""
Spatially-structured E-I LIF network -- the wave engine.

Equations implemented (current-based Brunel form, paper Methods 4.1):

  membrane (Eq 3):   tau_m dV/dt = -V + I_E - I_I     [V_L = 0]
  synapse  (Eq 4-5): delayed difference-of-exponentials via gating s and
                     current I, distinct AMPA / GABA time constants.
                     A presynaptic spike injects  Delta s = (tau_m / tau_r) * w
                     (the w*tau_m/tau_r jump is baked into the weight matrices).
  external (Eq 6):   spatially homogeneous Poisson at rate nu_ext(t),
                     nu_ext(t) = [nu_signal(t) + xi(t)]_+, xi an OU process.
  delays   (Eq 7):   distance-dependent, handled by a ring buffer.

Integration: exponential Euler for the linear membrane and synaptic ODEs
(exact for piecewise-constant input over dt).

The three epilepsy slow variables from page 4 of the deck (disinhibition z,
adaptive threshold phi, sAHP g_K) are NOT in this core loop; they attach via
slow_vars.py and are off by default. Hook points are marked `SLOWVAR HOOK`.
"""

from __future__ import annotations
import time
import numpy as np

from params import Params, compute_nu_theta
from connectivity import place_neurons, build_connectivity


def build_network(p: Params, verbose=True):
    rng = np.random.default_rng(p.seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    if verbose:
        print(f"network: N={NE+NI:,}  (E={NE:,}, I={NI:,})  "
              f"L={p.L} mm  density={p.density:g}/mm^2", flush=True)
    net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=verbose)
    net["rng"] = rng
    return net


def simulate(p: Params, net, slow=None, nu_signal_fn=None,
             record_lfp=None, lfp_every=2, verbose=True):
    """Run the network.

    slow          : optional SlowVars instance (page-4 epilepsy layer); None = off
    nu_signal_fn  : optional callable t_ms -> nu_signal (1/ms). Default: constant
                    p.nu_ext_ratio * nu_theta. Use for a movement/ictal ramp.
    record_lfp    : optional LFPRecorder (lfp.py). If given, LFP frames are saved.
    """
    rng = net["rng"]
    NE, NI = net["NE"], net["NI"]
    N = NE + NI
    labels = net["labels"]
    ampa = net["ampa_by_delay"]
    gaba = net["gaba_by_delay"]
    M = net["max_delay_steps"] + 1

    dt = p.dt
    nsteps = int(round(p.T / dt))

    # ---- precomputed decays ----
    decay_sE = np.exp(-dt / p.tau_r_AMPA)
    decay_IE = np.exp(-dt / p.tau_d_AMPA)
    decay_sI = np.exp(-dt / p.tau_r_GABA)
    decay_II = np.exp(-dt / p.tau_d_GABA)
    tau_m = np.where(labels == 0, p.tau_m_E, p.tau_m_I).astype(np.float64)
    decay_V = np.exp(-dt / tau_m)

    ref_steps = np.where(labels == 0,
                         int(round(p.tau_ref_E / dt)),
                         int(round(p.tau_ref_I / dt))).astype(np.int32)

    # external AMPA jump per neuron: (tau_m/tau_r_AMPA) * J_ext_target
    ext_incr = (tau_m / p.tau_r_AMPA) * np.where(labels == 0, p.J_ext_E, p.J_ext_I)

    # nonempty delay bins
    ampa_bins = [d for d in range(M) if ampa[d].nnz > 0]
    gaba_bins = [d for d in range(M) if gaba[d].nnz > 0]

    # ---- external drive scale ----
    nu_theta, _, _ = compute_nu_theta(p)           # 1/ms
    if nu_signal_fn is None:
        nu_sig_const = p.nu_ext_ratio * nu_theta
        nu_signal_fn = lambda t_ms: nu_sig_const
    # OU on xi (1/ms): sigma_xi = sigma_n * sqrt(tau_n/2), sigma_n in (1/ms)/sqrt(ms)
    sigma_n_inv_ms = p.sigma_n * 1e-3
    sigma_xi = sigma_n_inv_ms * np.sqrt(p.tau_n / 2.0)
    ou_a = np.exp(-dt / p.tau_n)
    ou_b = sigma_xi * np.sqrt(1.0 - ou_a * ou_a)
    xi = 0.0

    # ---- state ----
    V = np.full(N, p.V_reset, dtype=np.float64)
    ref = np.zeros(N, dtype=np.int32)
    s_E = np.zeros(N); I_E = np.zeros(N)
    s_I = np.zeros(N); I_I = np.zeros(N)
    ring_sE = np.zeros((M, N))
    ring_sI = np.zeros((M, N))

    # ---- recorders ----
    rate_E = np.zeros(nsteps); rate_I = np.zeros(nsteps)
    spk_t = []; spk_i = []
    # subsample raster: keep ~80 E + 20 I neurons
    ras_keepE = rng.choice(NE, size=min(80, NE), replace=False)
    ras_keepI = NE + rng.choice(NI, size=min(20, NI), replace=False)
    ras_keep = np.concatenate([ras_keepE, ras_keepI])
    ras_mask = np.zeros(N, dtype=bool); ras_mask[ras_keep] = True
    lfp_frames = []; lfp_times = []

    t0 = time.time()
    for t in range(nsteps):
        tm = t * dt
        # ----- external homogeneous Poisson rate (Eq 6) -----
        xi = ou_a * xi + ou_b * rng.standard_normal()
        nu_now = nu_signal_fn(tm) + xi
        if nu_now < 0.0:
            nu_now = 0.0

        # ----- synaptic gating s: decay, recurrent arrivals, external -----
        s_E *= decay_sE
        s_I *= decay_sI
        slot = t % M
        s_E += ring_sE[slot]; ring_sE[slot] = 0.0
        s_I += ring_sI[slot]; ring_sI[slot] = 0.0
        ext = rng.poisson(nu_now * dt, size=N).astype(np.float64)
        s_E += ext * ext_incr

        # ----- synaptic currents (low-pass of s) -----
        I_E = s_E + (I_E - s_E) * decay_IE
        I_I = s_I + (I_I - s_I) * decay_II

        # SLOWVAR HOOK: disinhibition z scales I_I; sAHP g_K adds an outward term
        if slow is not None:
            I_net = slow.apply_currents(I_E, I_I, labels)
            V_th_eff = slow.threshold(p.V_th)      # adaptive threshold phi
        else:
            I_net = I_E - I_I
            V_th_eff = p.V_th

        # ----- membrane (Eq 3) + refractory -----
        ref -= 1
        np.maximum(ref, 0, out=ref)
        free = ref == 0
        Vtmp = I_net + (V - I_net) * decay_V
        V = np.where(free, Vtmp, p.V_reset)
        spk = free & (V >= (V_th_eff if np.isscalar(V_th_eff) else V_th_eff))
        V[spk] = p.V_reset
        ref[spk] = ref_steps[spk]

        # SLOWVAR HOOK: let slow variables integrate on this step's spikes
        if slow is not None:
            slow.step(spk, labels, dt)

        # ----- record -----
        rate_E[t] = spk[:NE].sum()
        rate_I[t] = spk[NE:].sum()
        if spk.any():
            idx = np.where(spk & ras_mask)[0]
            if idx.size:
                spk_t.append(np.full(idx.size, tm))
                spk_i.append(idx)
            # ----- scatter spikes into delay ring -----
            spE = np.where(spk[:NE])[0]
            spI = np.where(spk[NE:])[0]
            if spE.size:
                for d in ampa_bins:
                    contrib = np.asarray(ampa[d][:, spE].sum(axis=1)).ravel()
                    ring_sE[(t + d) % M] += contrib
            if spI.size:
                for d in gaba_bins:
                    contrib = np.asarray(gaba[d][:, spI].sum(axis=1)).ravel()
                    ring_sI[(t + d) % M] += contrib

        if record_lfp is not None and (t % lfp_every == 0):
            lfp_frames.append(record_lfp.sample(I_E, I_I))
            lfp_times.append(tm)

        if verbose and (t % max(1, nsteps // 10) == 0):
            print(f"  sim {t}/{nsteps}  ({tm:.0f} ms)  "
                  f"rate_E={rate_E[t]/NE/dt*1e3:.1f} Hz  elapsed {time.time()-t0:.1f}s",
                  flush=True)

    # convert rates to Hz
    rate_E_hz = rate_E / NE / dt * 1e3
    rate_I_hz = rate_I / NI / dt * 1e3
    out = dict(
        times=np.arange(nsteps) * dt,
        rate_E=rate_E_hz, rate_I=rate_I_hz,
        spk_t=np.concatenate(spk_t) if spk_t else np.array([]),
        spk_i=np.concatenate(spk_i) if spk_i else np.array([]),
        ras_order=ras_keep,
        nu_theta=nu_theta,
        wall_s=time.time() - t0,
    )
    if record_lfp is not None:
        out["lfp"] = np.array(lfp_frames)        # (frames, n_sites)
        out["lfp_t"] = np.array(lfp_times)
        out["lfp_sites"] = record_lfp.sites
    if verbose:
        print(f"done in {out['wall_s']:.1f}s  "
              f"mean rate_E={rate_E_hz.mean():.2f} Hz  rate_I={rate_I_hz.mean():.2f} Hz",
              flush=True)
    return out
