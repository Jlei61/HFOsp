"""
ONSET-FRONT axis-tracking discriminator for the spiking E-I network.

Question (the load-bearing 0d discriminator): when a localized kick triggers a
propagation event, does the LEADING EDGE of recruitment (the onset front, before
the event fills the sheet) elongate ALONG the E->E connectivity long axis, and
does that axis TRACK the imposed connectivity angle theta_EE as it rotates? An
isotropic E->E control must give NO preferred axis.

Why a prior probe was inconclusive: it used the PEAK-bin active region, which by
the peak has filled the sheet, so boundaries wash out the covariance. Fix: an
ONSET-FRONT metric over a short window right after kick offset, on a BIGGER
sheet, with multiple seeds, axis ROTATION (theta in {0,45,90}), and an isotropic
control (AR=1).

Background confound (inherited from kick_propagation.py): at quiet drive
(ratio 0.6) spontaneous spikes sit all over the sheet. A position covariance is
dominated by the most distant points, so sheet-wide background can blind the
front metric -> ratio~1, random angle, for EVERY condition (mimicking a true
isotropic negative). Guard: run a kick-OFF control per condition; define the
front set as ON-onset AND NOT OFF-onset (excess); report BOTH raw-spec and
excess axis/ratio/n; the OFF-front is the metric's noise floor. Trust the
verdict only after OFF-vs-ON separation confirms the metric is not blind.

Reuses `kick_probe.simulate_kick` VERBATIM (same kick mechanism, decays,
refractory, ring buffer, membrane, RNG path). Only the connectivity is built
via `build_connectivity_rot`. No existing file is modified.

Run:  python anisotropy_front.py
Durable outputs (written incrementally so nothing is lost if interrupted):
  ../../data/anisotropy_front_numbers.json
  ../../data/anisotropy_front.log
  ../../figures/anisotropy_front_test.png
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
from connectivity import place_neurons
from connectivity_rot import build_connectivity_rot
from kick_probe import simulate_kick, R_KICK, T_KICK, DUR_KICK

# ---- output roots ----
HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "..", "data"))
FIGS = os.path.abspath(os.path.join(HERE, "..", "..", "figures"))
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)
JSON_PATH = os.path.join(DATA, "anisotropy_front_numbers.json")
LOG_PATH = os.path.join(DATA, "anisotropy_front.log")
FIG_PATH = os.path.join(FIGS, "anisotropy_front_test.png")

# ---- front windows (ms after kick offset) ----
W_LIST = [6.0, 8.0, 10.0]          # report all three
W_PRIMARY = 8.0
FRONT_LO = T_KICK + DUR_KICK        # kick offset = 168 ms
# discriminator thresholds (spec)
AXIS_ERR_MAX = 25.0
RATIO_MIN = 1.3

_logf = None


def log(msg):
    print(msg, flush=True)
    if _logf is not None:
        _logf.write(msg + "\n")
        _logf.flush()


def build_network_rot(p, theta_EE, AR, verbose=False):
    """Mirror of model.build_network but with the rotated E->E kernel. Placement
    and kernel sampling both consume p.seed via a single fresh rng."""
    rng = np.random.default_rng(p.seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=theta_EE, AR=AR, verbose=verbose)
    net["rng"] = rng
    return net


def onset_times(E_spk_bool, dt, t_after_ms):
    """First spike time (ms) per E neuron AT OR AFTER t_after_ms; inf if never.

    Returns (NE,) float array. Vectorized: argmax over the post-threshold slice
    gives the first True index (0 if the row is all-False), masked by any()."""
    i0 = int(round(t_after_ms / dt))
    post = E_spk_bool[i0:]                       # (nsteps-i0, NE)
    ever = post.any(axis=0)
    first_idx = post.argmax(axis=0)              # first True, or 0 if none
    onset = np.full(E_spk_bool.shape[1], np.inf)
    onset[ever] = (i0 + first_idx[ever]) * dt
    return onset


def front_mask(onset, lo, hi):
    """E neurons whose onset falls in [lo, hi) ms."""
    return (onset >= lo) & (onset < hi)


def principal_axis(mask, posE, center):
    """Covariance principal axis of (front positions - center).

    Returns (angle_deg_mod180, ratio=sqrt(lmax/lmin), n). Uses positions
    relative to center per spec. nan/nan/n if fewer than 3 points."""
    n = int(mask.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    pts = posE[mask] - center
    C = np.cov(pts, rowvar=False)
    evals, evecs = np.linalg.eigh(C)
    lmin, lmax = evals[0], evals[1]
    vmaj = evecs[:, 1]
    angle = np.degrees(np.arctan2(vmaj[1], vmaj[0])) % 180.0
    ratio = float(np.sqrt(lmax / lmin)) if lmin > 0 else float("inf")
    return float(angle), ratio, n


def axis_error(angle_deg, theta_deg):
    """Circular axis error mod 180: min(|a-t|, 180-|a-t|)."""
    if not np.isfinite(angle_deg):
        return float("nan")
    d = abs(angle_deg - (theta_deg % 180.0))
    return float(min(d, 180.0 - d))


def max_active_frac(E_spk_bool, dt, t_lo, t_hi, bin_ms=2.0):
    """Max over bins of (distinct E spiking in bin)/NE within [t_lo,t_hi)."""
    NE = E_spk_bool.shape[1]
    bin_steps = int(round(bin_ms / dt))
    i_lo = int(round(t_lo / dt)); i_hi = int(round(t_hi / dt))
    best = 0.0
    for b0 in range(i_lo, i_hi, bin_steps):
        b1 = min(b0 + bin_steps, i_hi)
        if b1 <= b0:
            continue
        best = max(best, E_spk_bool[b0:b1].any(axis=0).sum() / NE)
    return float(best)


def analyze_run(res_on, res_off, p, posE, center, theta_deg):
    """Onset-front axis for every W, both raw (ON-only) and excess (ON & not OFF)."""
    dt = p.dt
    onset_on = onset_times(res_on["E_spk_bool"], dt, T_KICK)
    onset_off = onset_times(res_off["E_spk_bool"], dt, T_KICK)
    # OFF front membership over the same windows, for the excess set
    out = {"theta_deg": theta_deg, "windows": {}}
    for W in W_LIST:
        lo, hi = FRONT_LO, FRONT_LO + W
        m_on = front_mask(onset_on, lo, hi)
        m_off = front_mask(onset_off, lo, hi)
        m_excess = m_on & ~m_off
        a_raw, r_raw, n_raw = principal_axis(m_on, posE, center)
        a_exc, r_exc, n_exc = principal_axis(m_excess, posE, center)
        a_off, r_off, n_off = principal_axis(m_off, posE, center)
        out["windows"][f"W{int(W)}"] = dict(
            W_ms=W, lo_ms=lo, hi_ms=hi,
            raw=dict(angle_deg=a_raw, axis_error_deg=axis_error(a_raw, theta_deg),
                     ratio=r_raw, n=n_raw),
            excess=dict(angle_deg=a_exc, axis_error_deg=axis_error(a_exc, theta_deg),
                        ratio=r_exc, n=n_exc),
            off_front=dict(angle_deg=a_off, axis_error_deg=axis_error(a_off, theta_deg),
                           ratio=r_off, n=n_off),
        )
    # event-fill diagnostics: max active fraction over the whole event window
    out["max_active_frac_event"] = max_active_frac(
        res_on["E_spk_bool"], dt, T_KICK, p.T)
    # primary-W front-set size vs total recruited (front << total -> pre-saturation)
    total_recruited = int((onset_on < np.inf).sum())
    out["total_recruited_E"] = total_recruited
    out["NE"] = int(res_on["NE"])
    return out, onset_on


def fresh_run_rot(p, net, KICK_BOOST):
    """Reseed rng each run (mirror kick_probe.fresh_run) then simulate_kick."""
    net["rng"] = np.random.default_rng(p.seed)
    return simulate_kick(p, net, KICK_BOOST=KICK_BOOST)


# ----------------------------- conditions -----------------------------
def condition_list():
    conds = []
    for theta in (0, 45, 90):
        for seed in (1, 2, 3):
            conds.append(dict(label="aniso", theta_deg=theta, AR=2.0, seed=seed))
    for seed in (1, 2, 3):
        conds.append(dict(label="iso", theta_deg=0, AR=1.0, seed=seed))
    return conds


# ----------------------------- figure -----------------------------
def make_figure(panels, L, center, figpath):
    """panels: list of (title, theta_deg, AR, posE, onset_on). One panel each."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.6))
    if n == 1:
        axes = [axes]
    th = np.linspace(0, 2 * np.pi, 120)
    disk_x = center[0] + R_KICK * np.cos(th)
    disk_y = center[1] + R_KICK * np.sin(th)
    for ax, (title, theta_deg, AR, posE, onset) in zip(axes, panels):
        finite = np.isfinite(onset)
        # color by onset time relative to kick offset; early=dark
        t_rel = onset[finite] - FRONT_LO
        vmin, vmax = 0.0, 40.0
        sc = ax.scatter(posE[finite, 0], posE[finite, 1], c=t_rel,
                        cmap="viridis", vmin=vmin, vmax=vmax, s=4,
                        linewidths=0)
        ax.plot(disk_x, disk_y, color="red", lw=1.6, zorder=5)
        # dashed line through center showing imposed theta_EE axis
        if AR > 1.0:
            ang = np.radians(theta_deg)
            length = 0.42 * L
            dx, dy = length * np.cos(ang), length * np.sin(ang)
            ax.plot([center[0] - dx, center[0] + dx],
                    [center[1] - dy, center[1] + dy],
                    color="red", lw=1.8, ls="--", zorder=6,
                    label=f"theta_EE={theta_deg}deg")
            ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
        else:
            ax.set_title("", fontsize=1)  # iso: no axis line
        ax.set_xlim(0, L); ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (mm)")
        ax.set_xticks([0, L / 2, L]); ax.set_yticks([0, L / 2, L])
    axes[0].set_ylabel("y (mm)")
    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.01)
    cbar.set_label("onset time after kick offset (ms)\n(early = dark)", fontsize=9)
    fig.suptitle(
        "Kick onset-front vs rotated E->E connectivity axis (seed 1). "
        "Dashed red = imposed theta_EE long axis. "
        "Is the EARLY (dark) front elongated along the dashed line?",
        fontsize=12)
    fig.savefig(figpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"[figure] saved {figpath}")


# ----------------------------- aggregate / verdict -----------------------------
def aggregate_for_W(runs, W):
    """Per (label, theta_deg): mean+/-sd over seeds at window W, for raw, excess,
    off_front (no verdicts -- reported for the W=6/8/10 comparison)."""
    by_key = {}
    for r in runs:
        if "error" in r:
            continue
        key = (r["analysis"]["theta_deg"], r["cond"]["label"])
        by_key.setdefault(key, []).append(r["analysis"]["windows"][f"W{int(W)}"])

    def msd(vals):
        a = np.array([v for v in vals if v is not None and np.isfinite(v)], float)
        if a.size == 0:
            return None, None
        return float(a.mean()), float(a.std(ddof=0))

    agg = {}
    for (theta, label), wins in sorted(by_key.items()):
        entry = {"theta_deg": theta, "label": label, "n_seeds": len(wins), "W_ms": W}
        for which in ("raw", "excess", "off_front"):
            e_m, e_s = msd([w[which]["axis_error_deg"] for w in wins])
            r_m, r_s = msd([w[which]["ratio"] for w in wins])
            n_m, _ = msd([float(w[which]["n"]) for w in wins])
            entry[which] = dict(axis_error_mean=e_m, axis_error_sd=e_s,
                                ratio_mean=r_m, ratio_sd=r_s, n_mean=n_m)
        agg[f"{label}_theta{theta}"] = entry
    return agg


def aggregate(runs):
    """Per (label, theta_deg): mean+/-sd over seeds at W_primary, for raw,
    excess, off_front. Then the two verdicts."""
    by_key = {}
    for r in runs:
        if "error" in r:
            continue
        key = (r["analysis"]["theta_deg"], r["cond"]["label"])
        by_key.setdefault(key, []).append(r["analysis"]["windows"][f"W{int(W_PRIMARY)}"])

    def msd(vals):
        a = np.array([v for v in vals if v is not None and np.isfinite(v)], float)
        if a.size == 0:
            return None, None, 0
        return float(a.mean()), float(a.std(ddof=0)), int(a.size)

    agg = {}
    for (theta, label), wins in sorted(by_key.items()):
        entry = {"theta_deg": theta, "label": label, "n_seeds": len(wins)}
        for which in ("raw", "excess", "off_front"):
            ang = [w[which]["angle_deg"] for w in wins]
            err = [w[which]["axis_error_deg"] for w in wins]
            rat = [w[which]["ratio"] for w in wins]
            nn = [w[which]["n"] for w in wins]
            a_m, a_s, _ = msd(ang)
            e_m, e_s, _ = msd(err)
            r_m, r_s, _ = msd(rat)
            n_m, _, _ = msd([float(x) for x in nn])
            entry[which] = dict(angle_mean=a_m, angle_sd=a_s,
                                axis_error_mean=e_m, axis_error_sd=e_s,
                                ratio_mean=r_m, ratio_sd=r_s,
                                n_mean=n_m)
        agg[f"{label}_theta{theta}"] = entry

    # ---- verdicts (use the EXCESS metric: background-robust) ----
    # Discriminator PASS if EACH aniso theta in {0,45,90}: mean axis-error<25 AND mean ratio>1.3
    aniso_ok = {}
    for theta in (0, 45, 90):
        e = agg.get(f"aniso_theta{theta}")
        if e is None:
            aniso_ok[theta] = None
            continue
        em = e["excess"]["axis_error_mean"]
        rm = e["excess"]["ratio_mean"]
        ok = (em is not None and rm is not None
              and em < AXIS_ERR_MAX and rm > RATIO_MIN)
        aniso_ok[theta] = dict(axis_error_mean=em, ratio_mean=rm, pass_=bool(ok))
    discriminator_pass = all(
        (aniso_ok[t] is not None and aniso_ok[t]["pass_"]) for t in (0, 45, 90))

    # Isotropic control: mean ratio < 1.3 (no preferred axis)
    iso = agg.get("iso_theta0")
    iso_ratio = iso["excess"]["ratio_mean"] if iso else None
    iso_circular = (iso_ratio is not None and iso_ratio < RATIO_MIN)

    verdicts = dict(
        metric_for_verdict="excess (ON-onset AND NOT OFF-onset), W=8ms",
        thresholds=dict(axis_error_max_deg=AXIS_ERR_MAX, ratio_min=RATIO_MIN),
        per_theta=aniso_ok,
        discriminator_pass=bool(discriminator_pass),
        isotropic_ratio_mean=iso_ratio,
        isotropic_circular_lt_1p3=bool(iso_circular) if iso_ratio is not None else None,
    )
    return agg, verdicts


def main():
    global _logf
    _logf = open(LOG_PATH, "w")
    t_all = time.time()

    base = Params(g=3.6, L=3.0, density=1800.0, T=320.0, nu_ext_ratio=0.6)
    nu_theta = compute_nu_theta(base)[0]
    KICK_BOOST = 2 * nu_theta
    center = np.array([base.L / 2, base.L / 2])
    N = int(round(base.density * base.L * base.L))
    log(f"[cfg] L={base.L} density={base.density} -> N~{N}  "
        f"nu_theta={nu_theta*1e3:.1f} Hz  KICK_BOOST(2x)={KICK_BOOST*1e3:.0f} Hz")
    log(f"[cfg] front window starts at kick offset {FRONT_LO:.0f} ms; "
        f"W in {W_LIST} (primary {W_PRIMARY}); thresholds err<{AXIS_ERR_MAX} ratio>{RATIO_MIN}")

    conds = condition_list()
    runs = []
    fig_panels = {}     # (label,theta) for seed==1 -> panel data

    out_doc = dict(
        config=dict(g=base.g, L=base.L, density=base.density, T=base.T,
                    nu_ext_ratio=base.nu_ext_ratio, N_approx=N,
                    R_KICK=R_KICK, T_KICK=T_KICK, DUR_KICK=DUR_KICK,
                    KICK_BOOST_Hz=float(KICK_BOOST * 1e3),
                    nu_theta_Hz=float(nu_theta * 1e3),
                    front_lo_ms=FRONT_LO, W_list=W_LIST, W_primary=W_PRIMARY,
                    axis_error_max_deg=AXIS_ERR_MAX, ratio_min=RATIO_MIN,
                    AR_aniso=2.0, AR_iso=1.0, l_EE=base.l_EE),
        runs=[], aggregate=None, verdicts=None,
        sanity_check=None,
    )

    # sanity check (Step A) into the doc
    from connectivity_rot import _sanity_check
    log("\n--- connectivity_rot sanity check ---")
    sc = _sanity_check()
    out_doc["sanity_check"] = {k: dict(angle_deg=v[0], ratio=v[1])
                              for k, v in sc.items()}

    def save_partial():
        with open(JSON_PATH, "w") as f:
            json.dump(out_doc, f, indent=2)

    for ci, cond in enumerate(conds):
        theta_rad = np.radians(cond["theta_deg"])
        p = Params(g=3.6, L=3.0, density=1800.0, T=320.0,
                   nu_ext_ratio=0.6, seed=cond["seed"])
        tag = f"[{ci+1}/{len(conds)}] {cond['label']} theta={cond['theta_deg']} " \
              f"AR={cond['AR']} seed={cond['seed']}"
        log(f"\n{tag}")
        try:
            t0 = time.time()
            net = build_network_rot(p, theta_EE=theta_rad, AR=cond["AR"],
                                    verbose=False)
            posE = net["pos"][:net["NE"]].copy()
            log(f"  built net in {time.time()-t0:.1f}s  NE={net['NE']} NI={net['NI']}")
            res_on = fresh_run_rot(p, net, KICK_BOOST=KICK_BOOST)
            res_off = fresh_run_rot(p, net, KICK_BOOST=0.0)
            # pre-kick identity check (RNG path intact)
            i_kick = int(round(T_KICK / p.dt))
            pre_ok = bool(np.array_equal(res_on["rate_E"][:i_kick],
                                         res_off["rate_E"][:i_kick]))
            analysis, onset_on = analyze_run(res_on, res_off, p, posE, center,
                                             cond["theta_deg"])
            wp = analysis["windows"][f"W{int(W_PRIMARY)}"]
            log(f"  pre_kick_identical={pre_ok}  "
                f"max_active_frac={analysis['max_active_frac_event']:.3f}  "
                f"total_recruited={analysis['total_recruited_E']}/{analysis['NE']}")
            log(f"  W8 raw   : angle={wp['raw']['angle_deg']:.1f} "
                f"err={wp['raw']['axis_error_deg']:.1f} ratio={wp['raw']['ratio']:.2f} "
                f"n={wp['raw']['n']}")
            log(f"  W8 excess: angle={wp['excess']['angle_deg']:.1f} "
                f"err={wp['excess']['axis_error_deg']:.1f} ratio={wp['excess']['ratio']:.2f} "
                f"n={wp['excess']['n']}")
            log(f"  W8 OFF   : angle={wp['off_front']['angle_deg']:.1f} "
                f"ratio={wp['off_front']['ratio']:.2f} n={wp['off_front']['n']} "
                f"(noise floor)")
            runs.append(dict(cond=cond, analysis=analysis,
                             pre_kick_identical=pre_ok,
                             build_s=float(time.time() - t0)))
            out_doc["runs"] = [dict(cond=r["cond"], analysis=r["analysis"],
                                    pre_kick_identical=r["pre_kick_identical"])
                               for r in runs]
            # capture seed-1 panel data
            if cond["seed"] == 1:
                fig_panels[(cond["label"], cond["theta_deg"])] = (
                    posE.copy(), onset_on.copy(), cond["AR"])
            # free big arrays
            del res_on, res_off, net
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log(f"  ERROR: {e}\n{tb}")
            runs.append(dict(cond=cond, error=str(e)))
            out_doc["runs"].append(dict(cond=cond, error=str(e)))
        save_partial()

    # ---- aggregate + verdicts ----
    agg, verdicts = aggregate(runs)
    out_doc["aggregate"] = agg          # W=8 primary (with verdict)
    out_doc["aggregate_by_W"] = {f"W{int(W)}": aggregate_for_W(runs, W)
                                 for W in W_LIST}   # W=6/8/10 comparison
    out_doc["verdicts"] = verdicts
    save_partial()

    log("\n===== W=6/8/10 EXCESS-RATIO COMPARISON (front thins -> less boundary fill) =====")
    for theta in (0, 45, 90):
        line = f"  aniso theta={theta}: "
        for W in W_LIST:
            e = out_doc["aggregate_by_W"][f"W{int(W)}"].get(f"aniso_theta{theta}")
            if e:
                line += (f"W{int(W)}[ratio={e['excess']['ratio_mean']:.2f} "
                         f"err={e['excess']['axis_error_mean']:.1f} "
                         f"n~{e['excess']['n_mean']:.0f}]  ")
        log(line)
    for W in W_LIST:
        e = out_doc["aggregate_by_W"][f"W{int(W)}"].get("iso_theta0")
        if e:
            log(f"  iso W{int(W)}: ratio={e['excess']['ratio_mean']:.2f} "
                f"n~{e['excess']['n_mean']:.0f}")

    log("\n===== PER-THETA AGGREGATE (W=8ms, excess metric) =====")
    for key, e in agg.items():
        ex = e["excess"]
        log(f"  {key:>16}: err={ex['axis_error_mean']} +/- {ex['axis_error_sd']}  "
            f"ratio={ex['ratio_mean']} +/- {ex['ratio_sd']}  n~{ex['n_mean']}")
    log("\n===== VERDICTS =====")
    log(f"  discriminator_pass (all theta err<25 & ratio>1.3) = "
        f"{verdicts['discriminator_pass']}")
    for t, v in verdicts["per_theta"].items():
        log(f"    theta={t}: {v}")
    log(f"  isotropic ratio_mean = {verdicts['isotropic_ratio_mean']}  "
        f"circular(<1.3) = {verdicts['isotropic_circular_lt_1p3']}")

    # ---- figure: one panel per condition (theta 0,45,90, iso), seed 1 ----
    panel_order = [("aniso", 0), ("aniso", 45), ("aniso", 90), ("iso", 0)]
    panels = []
    for label, theta in panel_order:
        if (label, theta) in fig_panels:
            posE, onset_on, AR = fig_panels[(label, theta)]
            title = (f"theta_EE={theta}deg (AR=2)" if label == "aniso"
                     else "isotropic (AR=1)")
            panels.append((title, theta, AR, posE, onset_on))
    if panels:
        make_figure(panels, base.L, center, FIG_PATH)

    log(f"\n[json] saved {JSON_PATH}")
    log(f"[log]  saved {LOG_PATH}")
    log(f"[done] total wall {time.time()-t_all:.1f}s")
    _logf.close()


if __name__ == "__main__":
    main()
