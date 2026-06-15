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


def simulate_kick(p: Params, net, KICK_BOOST, slow=None, nu_signal_fn=None,
                  verbose=False, kick_center=None, lfp_recorder=None, r_kick=None, t_kick=None,
                  V_th_per_neuron=None):
    """Verbatim copy of model.simulate's integration loop, with ONE addition:
    a localized transient kick on the external Poisson rate. The kick adds
    `KICK_BOOST` (extra external rate, 1/ms) to the E neurons in a disk of
    radius R_KICK about the sheet center, during [T_KICK, T_KICK+DUR_KICK).

    Returns the standard recorders plus per-step inside/outside-disk E-spike
    counts. Use slow=None (epilepsy layer off). KICK_BOOST=0 -> pure control
    (the external block then reduces to model.simulate's, modulo scalar-vs-array
    poisson internals).
    """
    rng = net["rng"]
    NE, NI = net["NE"], net["NI"]
    N = NE + NI
    labels = net["labels"]
    pos = net["pos"]
    ampa = net["ampa_by_delay"]
    gaba = net["gaba_by_delay"]
    M = net["max_delay_steps"] + 1

    dt = p.dt
    nsteps = int(round(p.T / dt))

    # ---- precomputed decays ---- (identical to model.simulate)
    decay_sE = np.exp(-dt / p.tau_r_AMPA)
    decay_IE = np.exp(-dt / p.tau_d_AMPA)
    decay_sI = np.exp(-dt / p.tau_r_GABA)
    decay_II = np.exp(-dt / p.tau_d_GABA)
    tau_m = np.where(labels == 0, p.tau_m_E, p.tau_m_I).astype(np.float64)
    decay_V = np.exp(-dt / tau_m)

    ref_steps = np.where(labels == 0,
                         int(round(p.tau_ref_E / dt)),
                         int(round(p.tau_ref_I / dt))).astype(np.int32)

    ext_incr = (tau_m / p.tau_r_AMPA) * np.where(labels == 0, p.J_ext_E, p.J_ext_I)

    ampa_bins = [d for d in range(M) if ampa[d].nnz > 0]
    gaba_bins = [d for d in range(M) if gaba[d].nnz > 0]
    # source-indexed flat edges for O(#firing-edges) scatter; cache on net (reused across runs)
    if "ampa_flat" not in net:
        net["ampa_flat"] = _flatten_by_source(ampa, ampa_bins, NE)
        net["gaba_flat"] = _flatten_by_source(gaba, gaba_bins, NI)
    a_indptr, a_dst, a_dly, a_w = net["ampa_flat"]
    g_indptr, g_dst, g_dly, g_w = net["gaba_flat"]

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
    ring_sE = np.zeros((M, N))
    ring_sI = np.zeros((M, N))

    # ---- recorders ---- (model.simulate's, kept so the RNG stream matches) ----
    rate_E = np.zeros(nsteps); rate_I = np.zeros(nsteps)
    spk_t = []; spk_i = []
    ras_keepE = rng.choice(NE, size=min(80, NE), replace=False)
    ras_keepI = NE + rng.choice(NI, size=min(20, NI), replace=False)
    ras_keep = np.concatenate([ras_keepE, ras_keepI])
    ras_mask = np.zeros(N, dtype=bool); ras_mask[ras_keep] = True
    # ---- NEW recorders: spread readout + distinct-neuron active fraction ----
    spk_inside = np.zeros(nsteps)
    spk_outside = np.zeros(nsteps)
    E_spk_bool = np.zeros((nsteps, NE), dtype=bool)   # for distinct-neuron bins
    # optional current-based LFP (|I_E|+|I_I| forward model) at custom sites (Increment-2/3)
    lfp_trace = (np.zeros((nsteps, len(lfp_recorder.sites)))
                 if lfp_recorder is not None else None)

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
        # ===================== KICK: the only change vs model.simulate =====================
        nu_vec = np.full(N, max(nu_now, 0.0))
        if tk <= tm < tk + DUR_KICK:
            nu_vec[kick_mask] += KICK_BOOST          # extra external rate, units 1/ms
        ext = rng.poisson(nu_vec * dt, size=N).astype(np.float64)
        s_E += ext * ext_incr
        # ==================================================================================

        # ----- synaptic currents (low-pass of s) -----
        I_E = s_E + (I_E - s_E) * decay_IE
        I_I = s_I + (I_I - s_I) * decay_II
        if lfp_trace is not None:                       # current-based LFP at custom sites
            lfp_trace[t] = lfp_recorder.sample(I_E, I_I)

        # slow layer off (slow=None)
        if slow is not None:
            I_net = slow.apply_currents(I_E, I_I, labels)
            V_th_eff = slow.threshold(p.V_th)
        else:
            I_net = I_E - I_I
            V_th_eff = p.V_th if V_th_per_neuron is None else V_th_per_neuron

        # ----- membrane (Eq 3) + refractory -----
        ref -= 1
        np.maximum(ref, 0, out=ref)
        free = ref == 0
        Vtmp = I_net + (V - I_net) * decay_V
        V = np.where(free, Vtmp, p.V_reset)
        spk = free & (V >= (V_th_eff if np.isscalar(V_th_eff) else V_th_eff))
        V[spk] = p.V_reset
        ref[spk] = ref_steps[spk]

        if slow is not None:
            slow.step(spk, labels, dt)

        # ----- record -----
        rate_E[t] = spk[:NE].sum()
        rate_I[t] = spk[NE:].sum()
        # NEW: spread + distinct-neuron readout
        spk_inside[t] = spk[kick_mask].sum()
        spk_outside[t] = spk[outside_mask].sum()
        E_spk_bool[t] = spk[:NE]
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
                    np.add.at(ring_sE, ((t + a_dly[idx]) % M, a_dst[idx]), a_w[idx])
            if spI.size:
                st = g_indptr[spI]; cnt = g_indptr[spI + 1] - st; tot = int(cnt.sum())
                if tot:
                    idx = (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                           + np.repeat(st, cnt))
                    np.add.at(ring_sI, ((t + g_dly[idx]) % M, g_dst[idx]), g_w[idx])

        if verbose and (t % max(1, nsteps // 5) == 0):
            print(f"  sim {t}/{nsteps}  ({tm:.0f} ms)  "
                  f"rate_E={rate_E[t]/NE/dt*1e3:.1f} Hz  elapsed {time.time()-t0:.1f}s",
                  flush=True)

    rate_E_hz = rate_E / NE / dt * 1e3
    rate_I_hz = rate_I / NI / dt * 1e3
    return dict(
        times=np.arange(nsteps) * dt,
        rate_E=rate_E_hz, rate_I=rate_I_hz,
        spk_inside=spk_inside, spk_outside=spk_outside,
        E_spk_bool=E_spk_bool,
        n_inside=int(kick_mask.sum()), n_outside=int(outside_mask.sum()),
        NE=NE, nu_theta=nu_theta, wall_s=time.time() - t0,
        lfp_trace=lfp_trace,                                    # (nsteps, n_sites) or None
        lfp_sites=(None if lfp_recorder is None else lfp_recorder.sites),
    )


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
