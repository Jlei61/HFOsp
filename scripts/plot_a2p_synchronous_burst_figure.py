"""A2-P main explanatory figure: the slow variable enlarges event EXTENT but the large
state is a GLOBAL SYNCHRONOUS BURST, not a traveling seizure.

Styled after core_model_s3_brakeoff_stim_ab.png (source-space event maps on the left, a
virtual-SEEG contact read-out on the right). Re-simulates ONE best-point g_K run in-process
(l0_g1.0 / twoend_equal, k_use=0.2, gk_max=0.03, tau_k=2000) — the same operating point as
the committed explore3_gk best — keeps the spike matrix, and renders:

  LEFT  : two source-space onset maps (each E neuron coloured by WHEN it fired in the event).
          A traveling wave shows a one-sided onset gradient; a synchronous burst is ~uniform.
          - top: a small low-permissivity event (between bursts)
          - bottom: a large high-permissivity event (during a burst) -> wide BUT ~uniform onset
  RIGHT : virtual-SEEG read-out — per-contact traces over a window spanning both events, with
          permissivity rho(t) on top. During the burst every contact peaks at ~the same time
          (the onset-order line is near-vertical) = synchronous, no propagation direction.
  INSET : spatial extent rises with permissivity, but the fraction of events with a readable
          propagation direction stays flat and low -> extent up, propagation not.

Run:  python scripts/plot_a2p_synchronous_burst_figure.py [--T 8000] [--seed 1]
"""
from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.getcwd()); sys.path.insert(0, "src/snn_engine")
import scripts.run_sef_hfo_snn_cm_spontaneous_readout as C  # noqa: E402
from src.sef_hfo_a2 import build_regional_resource  # noqa: E402
from src.sef_hfo_a1b import local_global_ratio  # noqa: E402

OUT = Path("results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg/a2p_propagation_gate")
FIG = OUT / "figures"
DT = C.DT


# --------------------------------------------------------------------------- sim
def simulate_a2(a, frozen_q=None):
    """Faithful l0_g1.0 best-point build+run (a1b knobs off, M3 off), keeping the spk matrix.

    frozen_q=(q_core, q_global) freezes the regional resource at those values instead of
    depleting it dynamically -- used by the mapping calibration sign test."""
    theta_rad = np.deg2rad(a.theta)
    axis_unit = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    p = C.Params(g=3.6, L=a.L, density=a.density, T=a.T, dt=DT, nu_ext_ratio=a.drive, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    pos, labels, NE, NI = C.place_neurons(p, rng)
    center = np.array([a.L / 2, a.L / 2]); half = a.L / 2
    # M3 region/hub overlay computed exactly as the runner does (consumes RNG identically)
    regions = C.corridor_regions(pos[:NE], center, axis_unit, half,
                                 corridor_half_frac=0.75, hub_frac=0.03, global_gap_frac=0.0)
    hub_mask = C.hub_mask_E(NE, regions["hub_idx"])
    net = C.build_connectivity_rot(
        p, pos, labels, NE, NI, rng, theta_EE=theta_rad, AR=a.AR, verbose=False, prune_radius=None,
        gate_scale=0.0, l_gate=None, C_gate=None, ei_gate_scale=0.0, l_ei_gate=None, C_ei_gate=None,
        hub_mask_E=hub_mask, hub_long_range_C=12, l_hub_long=6.0, hub_gain=0.0)
    vth, core_mask, foci, core_masks = C.build_lesion_vth(
        net, NE, axis_unit, center, half, "twoend_equal",
        a.core_mean, a.core_std, a.core_r, a.dephase, a.seed, sep_frac=a.sep_frac)
    m = C.montage(center, a.theta, 0.0, a.nc, pitch=C.PITCH)
    valid = C.valid_mask(m, net["pos"][:NE], a.L, p.Rr)
    rec = C.LFPRecorder(p, net["pos"], net["labels"], sites=m.contacts)
    net["rng"] = np.random.default_rng(a.seed)
    _foci_masks = core_masks if a.a2_mode == "per_core" else None
    if frozen_q is not None:
        slow = build_regional_resource(NE + NI, p.V_th, core_mask, NE, mode=a.a2_mode,
                                       frozen=True, frozen_q_core=float(frozen_q[0]),
                                       frozen_q_global=float(frozen_q[1]),
                                       tau_rec=a.a2_tau_rec, tau_a=100.0, q_min=a.a2_q_min,
                                       gk_max=a.a2_gk_max, tau_k=a.a2_tau_k, foci_masks=_foci_masks)
    else:
        slow = build_regional_resource(NE + NI, p.V_th, core_mask, NE, mode=a.a2_mode,
                                       k_use=a.a2_k_use, tau_rec=a.a2_tau_rec, tau_a=100.0, q_min=a.a2_q_min,
                                       gk_max=a.a2_gk_max, tau_k=a.a2_tau_k, foci_masks=_foci_masks)
    res = C.simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(center), r_kick=a.core_r,
                          t_kick=1e9, V_th_per_neuron=vth, slow=slow, lfp_recorder=rec,
                          dump_drive=True)
    qc = np.asarray(slow.trace_core, float); qg = np.asarray(slow.trace_global, float)
    lgr = local_global_ratio(1.0, 1.0, 1.0)
    rho_t = lgr / (qc * qg)
    return dict(p=p, net=net, NE=NE, posE=net["pos"][:NE], axis_unit=axis_unit, foci=np.asarray(foci),
                m=m, valid=valid, vth=vth, spk=res["E_spk_bool"], lfp=res["lfp_trace"],
                times=np.asarray(res["times"]), rho_t=rho_t, core_mask=np.asarray(core_mask),
                trace_core=qc, trace_global=qg, trace_gk=np.asarray(slow.trace_gk, float),
                q_core_min=float(qc.min()), q_global_min=float(qg.min()))


def _onset_gradient(posE, onset, fired, axis_unit):
    """Regress first-spike time on (x,y) over fired neurons. R2 = how much the firing ORDER is
    explained by position (a clean traveling wave -> high R2; synchronous/disorganized -> low R2).
    Returns (R2, |alignment of gradient with the inter-core axis|, propagation speed proxy)."""
    if fired.sum() < 20:
        return 0.0, 0.0, 0.0
    X = np.c_[np.ones(fired.sum()), posE[fired]]
    y = onset[fired]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    g = coef[1:]                                  # ms per mm gradient
    gnorm = float(np.linalg.norm(g))
    align = float(abs(g @ axis_unit) / gnorm) if gnorm > 1e-9 else 0.0
    return float(r2), align, gnorm


def read_events(sim):
    spk = sim["spk"]; m = sim["m"]
    af, bin_w = C.active_fraction(spk, DT, C.BIN_MS)
    nb0, nb1 = int(C.BASELINE_MS[0] / bin_w), int(C.BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    bar = floor + C.CAL_FRAC * (af.max() - floor)              # record_peak bar (runner default)
    raw = C.detect_events(af, bin_w, event_on_frac=bar)
    env_f, fdt, _ = C.snn_event_envelope(spk, sim["posE"], m, DT)
    rho = sim["rho_t"]
    out = []
    for ev in raw:
        rdv = C.read_event(env_f, fdt, m, sim["valid"], (ev["t_on"], ev["t_off"]), sim["axis_unit"])
        i0 = int(ev["t_on"] / DT)
        rho_pre = float(rho[max(0, i0 - 2000):max(1, i0 - 500)].mean())
        onset = C.per_neuron_onset(spk, ev["t_on"], ev["t_off"], DT)
        fired = np.isfinite(onset)
        au = sim["axis_unit"]; perp = np.array([-au[1], au[0]])
        if fired.sum() >= 3:
            pf = sim["posE"][fired]; cen = pf.mean(0)
            r95 = float(np.percentile(np.linalg.norm(pf - cen, axis=1), 95))
            onset_spread = float(np.nanpercentile(onset[fired], 95) - np.nanpercentile(onset[fired], 5))
            av = (pf - cen) @ au; qv = (pf - cen) @ perp              # along-axis / perpendicular spread
            reach_along = float(np.percentile(av, 95) - np.percentile(av, 5))
            reach_perp = float(np.percentile(qv, 95) - np.percentile(qv, 5))
            isotropy = float(reach_perp / max(reach_along, 1e-9))      # ->1 = round/off-axis; <1 = axis-elongated
        else:
            r95, onset_spread, reach_along, reach_perp, isotropy = 0.0, 0.0, 0.0, 0.0, 0.0
        grad_r2, grad_align, grad_norm = _onset_gradient(sim["posE"], onset, fired, sim["axis_unit"])
        out.append(dict(t_on=round(ev["t_on"], 1), t_off=round(ev["t_off"], 1),
                        returned=bool(ev["returned"]), n_part=rdv["n_part"], sign=rdv["sign"],
                        ranks=rdv["ranks"], axis_err=rdv["axis_err"], rho_pre=rho_pre,
                        r95_src=r95, n_fired=int(fired.sum()), onset_spread=onset_spread,
                        reach_along=reach_along, reach_perp=reach_perp, isotropy=isotropy,
                        grad_r2=grad_r2, grad_align=grad_align, grad_norm=grad_norm))
    return out


NPZ = Path("/tmp/claude-1002/-home-honglab-leijiaxin-HFOsp/cad82772-4d6a-4fa1-88f6-4c3b08dc888b/scratchpad/a2p_figdata.npz")


def _onset_panel(ax, posE, onset, foci, L, axis_unit, title, vmax):
    fired = np.isfinite(onset)
    ax.scatter(posE[:, 0], posE[:, 1], c="0.90", s=0.5, rasterized=True)
    sc = None
    if fired.any():
        sc = ax.scatter(posE[fired, 0], posE[fired, 1], c=onset[fired], s=1.3, cmap="viridis",
                        vmin=0, vmax=vmax, rasterized=True)
    for f in foci:
        ax.add_patch(plt.Circle(f, 1.5, ec="crimson", fc="none", lw=1.3, ls="--", zorder=6))
    c = np.array([L / 2, L / 2])
    ax.annotate("", xy=c + 7 * axis_unit, xytext=c - 7 * axis_unit,
                arrowprops=dict(arrowstyle="-", color="crimson", lw=0.8, ls=":", alpha=0.7))
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9.4, fontweight="bold")
    return sc


def plot_from_npz(a):
    d = np.load(NPZ, allow_pickle=True)
    events = json.load(open(NPZ.with_suffix(".events.json")))
    posE = d["posE"]; onsets = d["onsets"]; ton = d["t_on"]; foci = d["foci"]; L = float(d["L"])
    au = d["axis_unit"]; names = [str(x) for x in d["names"]]; contacts = d["contacts"]
    rho_t = d["rho_t"]; times = d["times"]; lfp = d["lfp"]
    cand = [e for e in events if e["n_fired"] >= 3]
    # representative events: a small low-permissivity local event + the largest high-perm recruitment wave
    lo = min(cand, key=lambda e: (e["rho_pre"], e["r95_src"]))
    hi = max(cand, key=lambda e: e["r95_src"])
    ilo = int(np.argmin(np.abs(ton - lo["t_on"]))); ihi = int(np.argmin(np.abs(ton - hi["t_on"])))
    vlo = max(1.0, np.nanpercentile(onsets[ilo][np.isfinite(onsets[ilo])], 98))
    vhi = max(1.0, np.nanpercentile(onsets[ihi][np.isfinite(onsets[ihi])], 98))

    fig = plt.figure(figsize=(16.5, 7.0), facecolor="white")
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.05, 1.05, 2.5], height_ratios=[3.1, 1.0],
                           wspace=0.16, hspace=0.46, left=0.035, right=0.99, bottom=0.10, top=0.84)
    axA = fig.add_subplot(gs[0, 0])
    scA = _onset_panel(axA, posE, onsets[ilo], foci, L, au,
                       "low permissivity:\nsmall local event (one focus)\nspread r95=%.0f, %d cells, lasts %.0f ms" % (lo["r95_src"], lo["n_fired"], lo["onset_spread"]), vlo)
    cbA = fig.colorbar(scA, ax=axA, fraction=0.046, pad=0.02, location="bottom"); cbA.ax.tick_params(labelsize=7)
    axB = fig.add_subplot(gs[0, 1])
    scB = _onset_panel(axB, posE, onsets[ihi], foci, L, au,
                       "high permissivity:\nlarge whole-field recruitment WAVE\nspread r95=%.0f, %d cells, lasts %.0f ms" % (hi["r95_src"], hi["n_fired"], hi["onset_spread"]), vhi)
    cbB = fig.colorbar(scB, ax=axB, fraction=0.046, pad=0.02, location="bottom"); cbB.ax.tick_params(labelsize=7)
    cbB.set_label("time each cell first fires within the event (ms) — a smooth colour gradient across space = a travelling recruitment wave", fontsize=7.6)

    # readout around the big recruitment wave (show the directed peak order = diagonal line)
    gsr = gs[0, 2].subgridspec(2, 1, height_ratios=[1, 6], hspace=0.06)
    ax_rho = fig.add_subplot(gsr[0]); ax_ro = fig.add_subplot(gsr[1])
    win = (max(0, hi["t_on"] - 450), min(times[-1], hi["t_off"] + 250))
    # neighbouring events visible in window
    vis = [e for e in cand if win[0] <= e["t_on"] <= win[1]]
    sel = (times >= win[0]) & (times <= win[1]); ts = times[sel]
    pax = np.array([-au[1], au[0]])
    order = np.lexsort(([c @ pax for c in contacts], [c @ au for c in contacts]))
    sub = lfp.T[order][:, sel]
    base = np.median(sub, axis=1, keepdims=True); scale = np.maximum(sub.max(axis=1, keepdims=True) - base, 1e-9)
    zt = (sub - base) / scale; OFFV = 1.35; y = np.arange(len(order)) * OFFV
    ridx = (np.arange(len(rho_t)) * DT >= win[0]) & (np.arange(len(rho_t)) * DT <= win[1])
    ax_rho.plot(np.arange(len(rho_t))[ridx] * DT, rho_t[ridx], color="#6a1b9a", lw=1.4)
    ax_rho.set_ylabel("permissivity", fontsize=8); ax_rho.set_xlim(*win); ax_rho.set_xticks([]); ax_rho.tick_params(labelsize=7)
    ax_rho.set_title("virtual-SEEG read-out — the large wave sweeps the contacts in order (black line tilts = a travelling front, not synchronous)",
                     fontsize=9, fontweight="bold", loc="left")
    for i, ci in enumerate(order):
        col = "#1f9e9e" if names[ci].startswith("B") else "#e8743b"
        ax_ro.plot(ts, zt[i] + y[i], color=col, lw=0.8, alpha=0.95)
    for e in vis:
        big = e["t_on"] == hi["t_on"]
        ax_ro.axvspan(e["t_on"], e["t_off"], color="#ffd6cc" if big else "#e7e7e7", alpha=0.6 if big else 0.4, lw=0, zorder=0)
        pts = []
        ranks = e.get("ranks") or {}
        for i, ci in enumerate(order):
            if ranks.get(names[ci]) is None:
                continue
            mm = (ts >= e["t_on"]) & (ts <= e["t_off"])
            if mm.sum() < 2:
                continue
            pi = np.flatnonzero(mm)[int(np.argmax(zt[i][mm]))]
            pts.append((ts[pi], zt[i][pi] + y[i]))
            ax_ro.plot(ts[pi], zt[i][pi] + y[i], "o", ms=3, mfc="k", mec="white", mew=0.4, zorder=5)
        if len(pts) >= 2:
            px, py = zip(*sorted(pts, key=lambda z: z[1]))
            ax_ro.plot(px, py, "-", color="k", lw=1.1, alpha=0.75, zorder=4)
    ax_ro.text(hi["t_on"], y[-1] + 0.9, "large recruitment wave", color="#b23", fontsize=9, fontweight="bold", va="bottom")
    ax_ro.set_xlim(*win); ax_ro.set_yticks(y); ax_ro.set_yticklabels([names[i] for i in order])
    for tick, ci in zip(ax_ro.get_yticklabels(), order):
        tick.set_color("#1f9e9e" if names[ci].startswith("B") else "#e8743b"); tick.set_fontsize(7.2)
    ax_ro.tick_params(axis="y", length=0); ax_ro.tick_params(axis="x", labelsize=7.6)
    for sp in ax_ro.spines.values():
        sp.set_visible(False)
    ax_ro.set_xlabel("time (ms)", fontsize=8.5)

    # inset: extent rises with permissivity, propagation coherence stays high
    axIn = fig.add_subplot(gs[1, :2])
    rho = np.array([e["rho_pre"] for e in cand]); r95 = np.array([e["r95_src"] for e in cand])
    gr2 = np.array([e["grad_r2"] for e in cand])
    s = axIn.scatter(rho, r95, c=gr2, cmap="plasma", vmin=0, vmax=1, s=46, edgecolor="0.3", lw=0.4)
    axIn.scatter(hi["rho_pre"], hi["r95_src"], s=180, facecolor="none", edgecolor="#b23", lw=1.6)
    axIn.annotate("large wave", (hi["rho_pre"], hi["r95_src"]), (hi["rho_pre"] - 0.06, hi["r95_src"] - 2),
                  fontsize=8, color="#b23")
    cb2 = fig.colorbar(s, ax=axIn, fraction=0.06, pad=0.02); cb2.set_label("propagation coherence\n(position predicts firing order)", fontsize=7.5); cb2.ax.tick_params(labelsize=7)
    axIn.set_xlabel("permissivity before the event", fontsize=8.5)
    axIn.set_ylabel("event spread r95 (a.u.)", fontsize=8.5); axIn.tick_params(labelsize=7.5)
    axIn.set_title("extent rises with permissivity; events stay coherent (bright = position predicts firing order) — they propagate at every size",
                   fontsize=8.4, fontweight="bold")

    fig.text(0.012, 0.95, "A2-P", fontsize=15, fontweight="bold")
    fig.suptitle("A dynamic inhibitory-resource slow variable gates small local events  →  a large whole-field coherent recruitment WAVE "
                 "(directed spread along the inter-core axis — not a synchronous burst)", fontsize=11.8, fontweight="bold", y=0.975)
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / a.out_name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("wrote", out, "| low ev t=%.0f rho=%.2f r95=%.1f | big wave t=%.0f rho=%.2f r95=%.1f gradR2=%.2f"
          % (lo["t_on"], lo["rho_pre"], lo["r95_src"], hi["t_on"], hi["rho_pre"], hi["r95_src"], hi["grad_r2"]))


def _simulate_and_dump(a):
    C._engine_guard()
    sim = simulate_a2(a)
    events = read_events(sim)
    cand = [e for e in events if e["n_fired"] >= 3]
    # onset arrays for all events (NE per event) so the figure can be re-rendered without re-sim
    onsets = np.full((len(events), sim["NE"]), np.nan, np.float32)
    for j, e in enumerate(events):
        onsets[j] = C.per_neuron_onset(sim["spk"], e["t_on"], e["t_off"], DT).astype(np.float32)
    NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        NPZ, posE=sim["posE"].astype(np.float32), onsets=onsets,
        t_on=np.array([e["t_on"] for e in events]), t_off=np.array([e["t_off"] for e in events]),
        contacts=np.asarray(sim["m"].contacts, float), names=np.array([str(x) for x in sim["m"].names]),
        axis_unit=sim["axis_unit"], foci=sim["foci"], L=a.L,
        rho_t=sim["rho_t"].astype(np.float32), times=sim["times"].astype(np.float32),
        lfp=np.asarray(sim["lfp"], np.float32))
    json.dump(events, open(NPZ.with_suffix(".events.json"), "w"), default=float)
    # diagnostic table sorted by permissivity — decides framing BEFORE the figure
    print(f"\nn_events={len(events)}  first t_on={[e['t_on'] for e in events[:6]]}  (committed run: 106,138,556,588,1774,1808)")
    print("\n  t_on   rho_pre  r95_src  n_fired  onset_spread  grad_R2  grad_align  n_part  sign")
    for e in sorted(cand, key=lambda z: z["rho_pre"]):
        print("  %6.0f   %.3f   %6.2f   %6d   %8.1f ms   %.2f     %.2f      %4d   %s"
              % (e["t_on"], e["rho_pre"], e["r95_src"], e["n_fired"], e["onset_spread"],
                 e["grad_r2"], e["grad_align"], e["n_part"], e["sign"]))
    import numpy as _np
    rho = _np.array([e["rho_pre"] for e in cand])
    print("\nlow-rho third vs high-rho third (median):")
    lo3 = [e for e in cand if e["rho_pre"] <= _np.percentile(rho, 33)]
    hi3 = [e for e in cand if e["rho_pre"] >= _np.percentile(rho, 66)]
    for lab, grp in [("LOW-rho", lo3), ("HIGH-rho", hi3)]:
        print("  %-9s r95=%.2f  onset_spread=%.1fms  grad_R2=%.2f  grad_align=%.2f  readable_frac=%.2f" % (
            lab, _np.median([e["r95_src"] for e in grp]), _np.median([e["onset_spread"] for e in grp]),
            _np.median([e["grad_r2"] for e in grp]), _np.median([e["grad_align"] for e in grp]),
            _np.mean([e["sign"] is not None for e in grp])))
    print("\nwrote", NPZ)


def _add_a2_args(ap):
    """Slow-variable config — defaults reproduce the committed best point (core_only)."""
    ap.add_argument("--a2-mode", choices=["core_only", "two_tank", "per_core"], default="core_only")
    ap.add_argument("--a2-k-use", type=float, default=0.2)
    ap.add_argument("--a2-q-min", type=float, default=0.25)
    ap.add_argument("--a2-tau-rec", type=float, default=2000.0)
    ap.add_argument("--a2-gk-max", type=float, default=0.03)
    ap.add_argument("--a2-tau-k", type=float, default=2000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--density", type=float, default=100.0)
    ap.add_argument("--theta", type=float, default=45.0)
    ap.add_argument("--AR", type=float, default=2.0)
    ap.add_argument("--drive", type=float, default=0.6)
    ap.add_argument("--T", type=float, default=8000.0)
    ap.add_argument("--core-mean", type=float, default=17.5)
    ap.add_argument("--core-std", type=float, default=1.0)
    ap.add_argument("--core-r", type=float, default=1.5)
    ap.add_argument("--sep-frac", type=float, default=0.7)
    ap.add_argument("--dephase", type=float, default=0.3)
    ap.add_argument("--nc", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    _add_a2_args(ap)
    ap.add_argument("--plot", action="store_true", help="render from the dumped npz (no re-sim)")
    ap.add_argument("--out-name", default="a2p_synchronous_burst_main.png")
    a = ap.parse_args()
    if not a.plot:
        _simulate_and_dump(a)
        return
    plot_from_npz(a)


if __name__ == "__main__":
    main()
