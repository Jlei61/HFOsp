"""M3A core-model-style figure per slow-variable combo (reuses the canonical mechanism_4panel: a
heterogeneity/lesion map | b spontaneous-event onset/propagation order field | c ∥-shaft electrode
read-out | d ⊥-shaft read-out — the same visual as core_model_s3_brakeon). One figure per combo
from the cm-spontaneous readout written to an arbitrary out dir (so the M3A sweep dir is not the
hardcoded canonical dir). Adds a combo+phenotype subtitle. Pure plotting, NO re-sim.

Adapts run_sef_hfo_snn_cm_spontaneous_readout's rep_{tag}.npz schema + the canonical mechanism plot
(_figure in plot_sef_hfo_snn_cm_spontaneous_mechanism.py) — the heavy drawing helper mechanism_4panel
is reused unchanged.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.getcwd())
from src.sef_hfo_plot import mechanism_4panel           # noqa: E402
from src.sef_hfo_heterogeneity import local_vth_spread  # noqa: E402


def _contact_peak_times(lfp, t, win):
    m = (t >= win[0]) & (t <= win[1])
    idx = np.flatnonzero(m)
    return np.array([t[idx[int(np.argmax(lfp[ci, m]))]] for ci in range(lfp.shape[0])])


def _shaft(lfp, contacts, names, t, win, u_shaft, panel_title, contact_c):
    pre = t < win[0]
    base = lfp[:, pre].mean(axis=1); bstd = lfp[:, pre].std(axis=1) + 1e-9
    part = ((lfp[:, (t >= win[0]) & (t <= win[1])].max(axis=1) - base) / bstd) > 30.0
    c = np.asarray(contacts, float)
    return dict(contacts=c, part=part, names=list(names), s=(c - c.mean(0)) @ u_shaft,
                signal=lfp, panel_title=panel_title, contact_c=contact_c)


def _phenotype_subtitle(out_dir, tag):
    p = os.path.join(out_dir, f"readout_{tag}.json")
    if not os.path.exists(p):
        return ""
    s = json.load(open(p)); c = s["config"]
    sv = c.get("slow_var", "none"); lv = c.get("slow_level")
    knob = (f"slow={sv}={lv}" if sv not in (None, "none")
            else (f"e_GABA={c.get('e_gaba')} g={c.get('g_gaba_scale')}" if c.get("shunt_gaba")
                  else "OFF baseline"))
    return (f"{knob}  |  core m={c.get('core_mean')} std={c.get('core_std')} r={c.get('core_r')}  |  "
            f"events={s['n_events']} clean fwd/rev={s['n_clean_forward']}/{s['n_clean_reverse']} "
            f"bar={s['detector']['bar']}")


def core_model_figure(npz_path, fig_path, out_dir):
    from pathlib import Path
    z = np.load(npz_path, allow_pickle=True)
    tag = Path(npz_path).stem.replace("rep_", "")
    L = float(z["L"]); theta = float(z["theta"]); nc = int(z["nc"])
    t = np.asarray(z["times"]); lfp = np.asarray(z["lfp"]).T
    ev_t = float(z["event_peak_t"]); t_on = float(z["event_t_on"]); t_off = float(z["event_t_off"])
    win = (ev_t - 25.0, ev_t + 30.0)
    tm = (t >= t_on - 40.0) & (t <= t_off + 60.0)
    t = t[tm]; lfp = lfp[:, tm]
    u_par = np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))])
    u_perp = np.array([np.cos(np.deg2rad(theta + 90)), np.sin(np.deg2rad(theta + 90))])
    pk = _contact_peak_times(lfp, t, win) - float(z["event_t_on"])
    par = _shaft(lfp[:nc], z["contacts"][:nc], [str(s) for s in z["names"][:nc]], t, win,
                 u_par, "∥ axis — peaks sweep (reads direction)", pk[:nc])
    perp = _shaft(lfp[nc:2 * nc], z["contacts"][nc:2 * nc], [str(s) for s in z["names"][nc:2 * nc]],
                  t, win, u_perp, "⊥ axis — peaks aligned (no direction)", pk[nc:2 * nc])
    posE = z["posE"]; vth = z["vth"]; NE = len(posE)
    spread = local_vth_spread(posE, vth[:NE], np.ones(NE, bool), 0.3)
    onset_rel = np.asarray(z["onset_core"], float); fin = np.isfinite(onset_rel)
    onset_rel = onset_rel - (np.nanmin(onset_rel) if fin.any() else 0.0)
    vlim = ((float(np.nanpercentile(onset_rel[fin], 5)), float(np.nanpercentile(onset_rel[fin], 95)))
            if fin.sum() > 5 else None)
    foci = np.asarray(z["foci"]) if "foci" in z.files else np.asarray([z["patch"]])
    sign = float(z["sign"]) if "sign" in z.files else 0.0
    dirword = "forward" if sign > 0 else ("reverse" if sign < 0 else "no-direction")
    disk_xy = z["patch"]; star_xy = np.asarray(z["kick"]) if "kick" in z.files else np.asarray(disk_xy)
    pr = float(z["patch_r"]); nfoci = "2 foci" if foci.shape[0] > 1 else "1 focus"
    extra = [(float(f[0]), float(f[1]), pr) for f in foci if not np.allclose(f, disk_xy)]
    sub = _phenotype_subtitle(out_dir, tag)
    mechanism_4panel(
        fig_path, field_xy=posE, kick_xy=star_xy, axis_deg=theta, extent=(0, L, 0, L),
        map_a=dict(field_c=spread, clabel="local V_th spread (mV)", cmap="plasma",
                   vlim=None, color_contacts=False, title=f"lesion / heterogeneity map · {nfoci}"),
        map_b=dict(field_c=onset_rel, clabel="time after event onset (ms)", cmap=None,
                   vlim=vlim, color_contacts=True, title=f"spontaneous event — onset / propagation ({dirword})"),
        par=par, perp=perp, t=t, event_window=win,
        signal_ylabel="current-LFP (|I_E|+|I_I|)", name_fs=8, label_endpoints_only=True,
        contact_note=f"M3A Stage-3 core + frozen slow var — {sub}",
        patch_circle=(float(disk_xy[0]), float(disk_xy[1]), pr), extra_patch_circles=extra,
        title=f"M3A core-model — {tag}  ({nfoci}, representative {dirword} event)\n{sub}")
    print(f"  wrote {fig_path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/topic4_sef_hfo/m3a_slowvars/cm_sweep")
    ap.add_argument("--tag", default=None, help="single tag; default = all rep_*.npz in out-dir/per_event")
    a = ap.parse_args(argv)
    fig_dir = os.path.join(a.out_dir, "figures"); os.makedirs(fig_dir, exist_ok=True)
    if a.tag:
        reps = [os.path.join(a.out_dir, "per_event", f"rep_{a.tag}.npz")]
    else:
        reps = sorted(glob.glob(os.path.join(a.out_dir, "per_event", "rep_*.npz")))
        reps = [r for r in reps if not (r.endswith("_fwd.npz") or r.endswith("_rev.npz"))]
    if not reps:
        print(f"no rep_*.npz in {a.out_dir}/per_event"); return 1
    for r in reps:
        if not os.path.exists(r):
            print(f"  (skip, missing {r})"); continue
        tag = os.path.basename(r)[len("rep_"):-len(".npz")]
        out_png = os.path.join(fig_dir, f"core_model_{tag}.png")
        if os.path.exists(out_png):
            continue                     # idempotent: skip already-rendered combos
        core_model_figure(r, out_png, a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
