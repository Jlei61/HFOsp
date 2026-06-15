"""
Traveling-wave vs synchronous-flash discriminator on a BIGGER sheet (L=2 mm).

A prior L=1 mm kick test recruited the whole sheet in ~2 ms (conduction time
across the tiny sheet), which cannot distinguish a traveling wave from a
synchronous burst, nor dynamical self-limiting from finite-size exhaustion.
Here we use L=2 mm (~5 E->E kernel widths) so an activation front has room
to resolve in space and time.

We REUSE kick_probe.py's kick mechanism verbatim (simulate_kick / fresh_run /
R_KICK / T_KICK / DUR_KICK / verify_pre_kick_identical). The kick adds
KICK_BOOST = 2*nu_theta extra external Poisson rate to E neurons in a central
disk of radius R_KICK=0.15 mm during [T_KICK, T_KICK+DUR_KICK)=[150,168) ms.
E_spk_bool (nsteps x NE), aligned to net["pos"][:NE], IS the full
(E-position, time) spike record; no existing file is modified.

Three factual questions (per the task):
  Q1 Traveling vs synchronous: does the active-region radius grow gradually
     over several 2-ms bins (-> front speed mm/ms) or jump ~0 -> half-width in
     one bin (-> synchronous)?
  Q2 Anisotropy: is the peak active region elongated along the (1,1)=45 deg
     E->E connectivity axis (rho_EE=0.6)? angle + elongation ratio.
  Q3 Self-limiting type: does active fraction peak BELOW 1.0 and return
     (dynamical) or reach ~1.0 / fill the sheet (finite-size, inconclusive)?

CONFOUND handled (advisor): at quiet drive (ratio 0.6) scattered spontaneous
spikes sit all over the sheet. The 90th-pct-radius over ALL spiking E in a bin
can be pinned to the corner by background from the first bin, masking gradual
growth, and "do corners ever activate" is trivially True over 300 ms. So we:
  * measure baseline per-bin active fraction (pre-kick ON bins + OFF run) first;
  * report BOTH the spec radius (all spiking E) AND a background-robust radius
    (kick-ON-minus-OFF excess spikes per bin) so a flat-high spec radius driven
    by spontaneous background is not mis-read as "synchronous";
  * tie the corner test to the EVENT: corner activation above the OFF corner
    rate during the peak window, and whether the 90th-pct radius ever reaches
    0.9*half-diagonal.

Run:  python kick_propagation.py
Durable outputs: outputs/kick_propagation.png , outputs/kick_propagation_numbers.json
"""

from __future__ import annotations
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from params import Params, compute_nu_theta
from model import build_network
from kick_probe import (
    simulate_kick, fresh_run, verify_pre_kick_identical,
    R_KICK, T_KICK, DUR_KICK,
)

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

BIN_MS = 2.0            # spatial-temporal time bin (spec)
N_POST_BINS = 12        # number of post-kick bins to report in the table
PEAK_WINDOW = (150.0, 300.0)   # ms; search for event peak in here


# ----------------------------- spatial metrics -----------------------------
def bin_active_mask(E_spk_bool, dt, b0_ms, bin_ms=BIN_MS):
    """Boolean (NE,) of E neurons that spiked at least once in [b0_ms, b0_ms+bin_ms)."""
    bin_steps = int(round(bin_ms / dt))
    i0 = int(round(b0_ms / dt))
    i1 = i0 + bin_steps
    return E_spk_bool[i0:i1].any(axis=0)


def active_radius_p90(active_mask, posE, center):
    """90th-pct of |pos - center| over active E neurons; nan if none active."""
    if active_mask.sum() == 0:
        return float("nan")
    d = np.linalg.norm(posE[active_mask] - center, axis=1)
    return float(np.percentile(d, 90.0))


def principal_axis(active_mask, posE):
    """Covariance principal axis of active-E positions.

    Returns (angle_deg_mod180, elongation_ratio=sqrt(lmax/lmin), n_active).
    angle is the major-axis direction in degrees, folded to [0,180).
    """
    n = int(active_mask.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    pts = posE[active_mask]                       # (n,2)
    C = np.cov(pts, rowvar=False)                 # 2x2
    evals, evecs = np.linalg.eigh(C)              # ascending eigenvalues
    lmin, lmax = evals[0], evals[1]
    vmaj = evecs[:, 1]                            # eigenvector of largest eigenvalue
    angle = np.degrees(np.arctan2(vmaj[1], vmaj[0])) % 180.0
    ratio = float(np.sqrt(lmax / lmin)) if lmin > 0 else float("inf")
    return float(angle), ratio, n


# ----------------------------- main analysis -----------------------------
def analyze(res_on, res_off, p, posE, center):
    dt = p.dt
    NE = res_on["NE"]
    half_diag = (p.L / np.sqrt(2.0))              # center-to-corner distance
    corner_thresh = 0.9 * half_diag               # "far corner" radius
    half_width = p.L / 2.0

    Eon = res_on["E_spk_bool"]
    Eoff = res_off["E_spk_bool"]

    # ---- baseline per-bin active fraction (background confound check) ----
    # pre-kick ON bins (0..T_KICK) and OFF run over the whole record.
    def per_bin_fracs(E, lo_ms, hi_ms):
        out = []
        for b0 in np.arange(lo_ms, hi_ms, BIN_MS):
            out.append(bin_active_mask(E, dt, b0).sum() / NE)
        return np.array(out)

    pre_kick_frac = per_bin_fracs(Eon, 0.0, T_KICK)        # ON, pre-kick
    off_post_frac = per_bin_fracs(Eoff, T_KICK, PEAK_WINDOW[1])  # OFF, same window
    baseline_bin_frac_mean = float(pre_kick_frac.mean())
    baseline_bin_frac_max = float(pre_kick_frac.max())
    off_bin_frac_mean = float(off_post_frac.mean())

    # ---- per-bin table over N_POST_BINS post-kick bins (kick-ON) ----
    rows = []
    for k in range(N_POST_BINS):
        b0 = T_KICK + k * BIN_MS
        m_on = bin_active_mask(Eon, dt, b0)
        m_off = bin_active_mask(Eoff, dt, b0)
        frac_on = m_on.sum() / NE
        frac_off = m_off.sum() / NE
        # spec radius: all spiking E in the bin (kick-ON)
        rad_spec = active_radius_p90(m_on, posE, center)
        # background-robust radius: E active in ON but NOT in OFF (excess/event)
        m_excess = m_on & ~m_off
        rad_excess = active_radius_p90(m_excess, posE, center)
        rows.append(dict(
            bin_idx=k,
            t_start=float(b0),
            t_end=float(b0 + BIN_MS),
            active_frac_on=float(frac_on),
            active_frac_off=float(frac_off),
            radius_spec_p90=rad_spec,
            radius_excess_p90=rad_excess,
            n_active_on=int(m_on.sum()),
            n_excess=int(m_excess.sum()),
        ))

    # ---- peak bin (max ON active fraction in PEAK_WINDOW) ----
    peak_frac = -1.0
    peak_b0 = None
    for b0 in np.arange(PEAK_WINDOW[0], PEAK_WINDOW[1], BIN_MS):
        f = bin_active_mask(Eon, dt, b0).sum() / NE
        if f > peak_frac:
            peak_frac = f
            peak_b0 = float(b0)
    peak_mask = bin_active_mask(Eon, dt, peak_b0)
    peak_angle, peak_ratio, peak_n = principal_axis(peak_mask, posE)
    # excess (event-only) axis at the peak bin
    peak_excess_mask = peak_mask & ~bin_active_mask(Eoff, dt, peak_b0)
    peak_ex_angle, peak_ex_ratio, peak_ex_n = principal_axis(peak_excess_mask, posE)

    # ---- early spreading-phase axis (first few post-kick bins, excess) ----
    spread_axes = []
    for k in range(6):
        b0 = T_KICK + k * BIN_MS
        m = bin_active_mask(Eon, dt, b0) & ~bin_active_mask(Eoff, dt, b0)
        a, r, n = principal_axis(m, posE)
        spread_axes.append(dict(t_start=float(b0), angle=a, ratio=r, n=n))

    # ---- Q1: front speed from the EXCESS radius (background-robust) ----
    # use radius_excess over the rising, pre-plateau bins.
    rad_ex = np.array([r["radius_excess_p90"] for r in rows], dtype=float)
    rad_spec = np.array([r["radius_spec_p90"] for r in rows], dtype=float)
    t_centers = np.array([r["t_start"] + BIN_MS / 2.0 for r in rows])
    # plateau value (robust): max finite excess radius
    finite_ex = rad_ex[np.isfinite(rad_ex)]
    plateau_ex = float(finite_ex.max()) if finite_ex.size else float("nan")
    # rising bins: from bin0 up to first bin >= 0.8*plateau (gradual discriminator)
    front_speed = None
    rise_bins = None
    if np.isfinite(plateau_ex) and plateau_ex > 0:
        reach = np.where(rad_ex >= 0.8 * plateau_ex)[0]
        if reach.size:
            i_plateau = int(reach[0])
            rise_bins = i_plateau  # number of bins to reach 80% plateau (0-indexed)
            if i_plateau >= 1:
                # slope over rising bins (excluding plateau bin endpoint)
                dr = rad_ex[i_plateau] - rad_ex[0]
                dts = t_centers[i_plateau] - t_centers[0]
                front_speed = float(dr / dts) if dts > 0 else None

    # ---- Q3: corner activation tied to the event ----
    # during the peak window, did any E beyond corner_thresh activate ABOVE the
    # OFF corner rate? also: did the spec/excess radius ever reach corner_thresh?
    far_mask = np.linalg.norm(posE - center, axis=1) > corner_thresh
    n_far = int(far_mask.sum())
    # excess corner activations across the whole peak window
    i_lo = int(round(PEAK_WINDOW[0] / dt))
    i_hi = int(round(PEAK_WINDOW[1] / dt))
    far_on = (Eon[i_lo:i_hi][:, far_mask].any(axis=0)).sum()
    far_off = (Eoff[i_lo:i_hi][:, far_mask].any(axis=0)).sum()
    corner_excess_event = int(far_on) - int(far_off)
    radius_reached_corner = bool(np.nanmax(rad_spec) >= corner_thresh)
    radius_excess_reached_corner = bool(
        (np.nanmax(rad_ex) >= corner_thresh) if finite_ex.size else False)

    # peak active fraction (whole event window, distinct neurons per 2-ms bin)
    peak_active_frac = float(peak_frac)
    # also a 5-ms-bin peak for comparison with kick_probe's metric
    self_limit = "dynamical" if peak_active_frac < 0.9 else "finite_size_or_saturated"

    return dict(
        config=dict(
            g=p.g, L=p.L, density=p.density, T=p.T,
            nu_ext_ratio=p.nu_ext_ratio, seed=p.seed,
            NE=NE,
            n_inside=int(res_on["n_inside"]), n_outside=int(res_on["n_outside"]),
            R_KICK=R_KICK, T_KICK=T_KICK, DUR_KICK=DUR_KICK,
            KICK_BOOST_Hz=float(2 * res_on["nu_theta"] * 1e3),
            nu_theta_Hz=float(res_on["nu_theta"] * 1e3),
            bin_ms=BIN_MS, half_width=half_width, half_diag=float(half_diag),
            corner_thresh=float(corner_thresh),
        ),
        background=dict(
            baseline_bin_frac_mean=baseline_bin_frac_mean,
            baseline_bin_frac_max=baseline_bin_frac_max,
            off_post_bin_frac_mean=off_bin_frac_mean,
        ),
        table=rows,
        Q1_traveling=dict(
            front_speed_mm_per_ms=front_speed,
            rise_bins_to_80pct_plateau=rise_bins,
            plateau_excess_radius_mm=plateau_ex,
            radius_excess_p90=[None if not np.isfinite(x) else float(x) for x in rad_ex],
            radius_spec_p90=[None if not np.isfinite(x) else float(x) for x in rad_spec],
            verdict=("synchronous" if (rise_bins is not None and rise_bins <= 1)
                     else ("traveling" if front_speed is not None else "fizzle_or_unresolved")),
        ),
        Q2_anisotropy=dict(
            peak_bin_t_start=peak_b0,
            peak_angle_deg=peak_angle, peak_ratio=peak_ratio, peak_n=peak_n,
            peak_excess_angle_deg=peak_ex_angle, peak_excess_ratio=peak_ex_ratio,
            peak_excess_n=peak_ex_n,
            spread_axes=spread_axes,
            aligned_45=bool(20.0 <= peak_angle <= 70.0) if np.isfinite(peak_angle) else None,
            ratio_gt_1p3=bool(peak_ratio > 1.3) if np.isfinite(peak_ratio) else None,
            verdict_anisotropic=bool(
                (20.0 <= peak_angle <= 70.0) and (peak_ratio > 1.3)
            ) if (np.isfinite(peak_angle) and np.isfinite(peak_ratio)) else None,
        ),
        Q3_self_limit=dict(
            peak_active_frac=peak_active_frac,
            self_limit_type=self_limit,
            corner_excess_event=corner_excess_event,
            n_far_neurons=n_far,
            far_corner_activated_above_off=bool(corner_excess_event > 0.1 * n_far),
            radius_spec_reached_corner=radius_reached_corner,
            radius_excess_reached_corner=radius_excess_reached_corner,
        ),
        peak_b0=peak_b0,
    )


# ----------------------------- figure -----------------------------
def make_figure(res_on, p, posE, center, analysis, figpath):
    dt = p.dt
    Eon = res_on["E_spk_bool"]
    L = p.L
    # 6 snapshot windows of ~3 ms spanning kick onset -> peak -> decay.
    peak_b0 = analysis["peak_b0"]
    # windows: start a touch before kick, march through peak, into decay.
    win_ms = 3.0
    starts = [T_KICK, T_KICK + 3, T_KICK + 6,
              peak_b0, peak_b0 + 6, peak_b0 + 18]
    fig, axes = plt.subplots(1, 6, figsize=(20, 3.6))
    # kick disk
    theta = np.linspace(0, 2 * np.pi, 100)
    disk_x = center[0] + R_KICK * np.cos(theta)
    disk_y = center[1] + R_KICK * np.sin(theta)
    for ax, t0 in zip(axes, starts):
        i0 = int(round(t0 / dt))
        i1 = int(round((t0 + win_ms) / dt))
        m = Eon[i0:i1].any(axis=0)
        pts = posE[m]
        ax.scatter(pts[:, 0], pts[:, 1], s=2, c="C3", alpha=0.6, linewidths=0)
        ax.plot(disk_x, disk_y, color="gold", lw=1.5)
        ax.set_xlim(0, L); ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.set_title(f"{t0:.0f}-{t0 + win_ms:.0f} ms\n(n={int(m.sum())})", fontsize=9)
        ax.set_xticks([0, L / 2, L]); ax.set_yticks([0, L / 2, L])
    axes[0].set_ylabel("y (mm)")
    for ax in axes:
        ax.set_xlabel("x (mm)")
    fig.suptitle(
        f"Localized kick on L={L} mm sheet (quiet drive ratio={p.nu_ext_ratio}); "
        f"E spikes per ~3-ms window. (1,1)=45 deg is the E->E long axis.",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(figpath, dpi=130)
    plt.close(fig)


def main():
    t_start = time.time()
    p = Params(g=3.6, L=2.0, density=4000.0, T=450.0, nu_ext_ratio=0.6, seed=1)
    dt = p.dt
    nu_theta = compute_nu_theta(p)[0]
    KICK_BOOST = 2 * nu_theta
    center = np.array([p.L / 2, p.L / 2])

    print(f"[cfg] L={p.L} density={p.density} -> N~{int(p.density*p.L*p.L)}", flush=True)
    print(f"[cfg] nu_theta={nu_theta*1e3:.1f} Hz  nu_signal(0.6)={0.6*nu_theta*1e3:.1f} Hz  "
          f"KICK_BOOST(2x)={KICK_BOOST*1e3:.0f} Hz", flush=True)

    print("[build] building network (a few minutes at N~16000) ...", flush=True)
    net = build_network(p, verbose=True)
    posE = net["pos"][:net["NE"]].copy()
    print(f"[build] done in {time.time()-t_start:.1f}s  NE={net['NE']} NI={net['NI']}", flush=True)

    print("[run] kick-ON ...", flush=True)
    res_on = fresh_run(p, net, KICK_BOOST=KICK_BOOST)
    print(f"[run] kick-ON done in {res_on['wall_s']:.1f}s", flush=True)
    print("[run] kick-OFF control ...", flush=True)
    res_off = fresh_run(p, net, KICK_BOOST=0.0)
    print(f"[run] kick-OFF done in {res_off['wall_s']:.1f}s", flush=True)

    ok, i_kick = verify_pre_kick_identical(res_on, res_off, dt)
    print(f"[verify] pre-kick (<{T_KICK:.0f}ms) E-rate bit-identical ON vs OFF = {ok}", flush=True)

    print("[analyze] computing spatial-temporal metrics ...", flush=True)
    analysis = analyze(res_on, res_off, p, posE, center)
    analysis["verify_pre_kick_identical"] = bool(ok)

    # peak global E-rate sanity (from rate trace)
    i_lo = int(round(PEAK_WINDOW[0] / dt)); i_hi = int(round(PEAK_WINDOW[1] / dt))
    analysis["config"]["peak_E_rate_Hz"] = float(res_on["rate_E"][i_lo:i_hi].max())
    analysis["config"]["baseline_E_rate_Hz"] = float(
        res_on["rate_E"][int(50/dt):int(150/dt)].mean())

    figpath = os.path.join(OUT, "kick_propagation.png")
    make_figure(res_on, p, posE, center, analysis, figpath)
    print(f"[figure] saved {figpath}", flush=True)

    jpath = os.path.join(OUT, "kick_propagation_numbers.json")
    with open(jpath, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"[json] saved {jpath}", flush=True)

    # ---- console summary ----
    print("\n===== RADIUS / ACTIVE-FRACTION TABLE (kick-ON, 2-ms bins) =====", flush=True)
    print("{:>4} {:>8} {:>12} {:>12} {:>13} {:>13} {:>8}".format(
        "bin", "t(ms)", "frac_on", "frac_off", "rad_spec_mm", "rad_excess_mm", "n_excess"))
    for r in analysis["table"]:
        rs = r["radius_spec_p90"]; re = r["radius_excess_p90"]
        print("{:>4} {:>8.0f} {:>12.4f} {:>12.4f} {:>13} {:>13} {:>8}".format(
            r["bin_idx"], r["t_start"], r["active_frac_on"], r["active_frac_off"],
            f"{rs:.3f}" if np.isfinite(rs) else "nan",
            f"{re:.3f}" if np.isfinite(re) else "nan",
            r["n_excess"]))

    bg = analysis["background"]
    print(f"\n[background] pre-kick per-bin frac mean={bg['baseline_bin_frac_mean']:.5f} "
          f"max={bg['baseline_bin_frac_max']:.5f}  OFF-post mean={bg['off_post_bin_frac_mean']:.5f}",
          flush=True)
    q1 = analysis["Q1_traveling"]; q2 = analysis["Q2_anisotropy"]; q3 = analysis["Q3_self_limit"]
    print(f"[Q1] front_speed={q1['front_speed_mm_per_ms']} mm/ms  "
          f"rise_bins={q1['rise_bins_to_80pct_plateau']}  plateau_excess_rad="
          f"{q1['plateau_excess_radius_mm']:.3f}  verdict={q1['verdict']}", flush=True)
    print(f"[Q2] peak angle={q2['peak_angle_deg']:.1f} deg ratio={q2['peak_ratio']:.2f} "
          f"(n={q2['peak_n']}) aligned45={q2['aligned_45']} ratio>1.3={q2['ratio_gt_1p3']} "
          f"verdict={q2['verdict_anisotropic']}", flush=True)
    print(f"     peak EXCESS angle={q2['peak_excess_angle_deg']:.1f} "
          f"ratio={q2['peak_excess_ratio']:.2f} (n={q2['peak_excess_n']})", flush=True)
    print(f"[Q3] peak_active_frac={q3['peak_active_frac']:.4f}  type={q3['self_limit_type']}  "
          f"corner_excess={q3['corner_excess_event']}/{q3['n_far_neurons']}  "
          f"far_activated={q3['far_corner_activated_above_off']}  "
          f"spec_radius_reached_corner={q3['radius_spec_reached_corner']}", flush=True)
    print(f"\n[done] total wall {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
