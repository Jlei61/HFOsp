"""4-panel mechanism figures for the SPONTANEOUS (noise-driven, no kick) cm-SNN read-out —
the kick-version `mechanism_<tag>.png` standard, reproduced for the self-igniting-lesion case
(user 2026-06-10). Same 4-panel layout (a heterogeneity/lesion map | b onset/propagation map |
c ∥-shaft read-out | d ⊥-shaft read-out, c/d shared), but the source is the LESION that
self-ignited the representative event — there is NO external kick marker.

Consumes per_event/rep_<tag>.npz written by run_sef_hfo_snn_cm_spontaneous_readout.py.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.getcwd())
from src.sef_hfo_plot import mechanism_4panel          # noqa: E402
from src.sef_hfo_heterogeneity import local_vth_spread  # noqa: E402

OUT = Path("results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous")
FIG = OUT / "figures"


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


def _figure(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    tag = str(z["lesion"]) if "lesion" in z.files else Path(npz_path).stem.replace("rep_", "")
    tag = Path(npz_path).stem.replace("rep_", "")
    L = float(z["L"]); theta = float(z["theta"]); nc = int(z["nc"])
    t = np.asarray(z["times"]); lfp = np.asarray(z["lfp"]).T               # (n_contact, nt)
    ev_t = float(z["event_peak_t"])
    t_on = float(z["event_t_on"]); t_off = float(z["event_t_off"])
    win = (ev_t - 25.0, ev_t + 30.0)
    # The spontaneous trace contains the WHOLE train (~7 events); _trace_panel takes each
    # contact's peak as argmax over the full trace, which would lock onto the global-max event.
    # Trim to JUST the representative event (+ pre-window for the baseline) so the peak locus
    # and slant reflect THIS event, matching the kick version's one-event-per-trace assumption.
    tm = (t >= t_on - 40.0) & (t <= t_off + 60.0)
    t = t[tm]; lfp = lfp[:, tm]
    u_par = np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))])
    u_perp = np.array([np.cos(np.deg2rad(theta + 90)), np.sin(np.deg2rad(theta + 90))])
    pk = _contact_peak_times(lfp, t, win) - float(z["event_t_on"])         # ms after event onset
    par = _shaft(lfp[:nc], z["contacts"][:nc], [str(s) for s in z["names"][:nc]], t, win,
                 u_par, "∥ axis — peaks sweep (reads direction)", pk[:nc])
    perp = _shaft(lfp[nc:2 * nc], z["contacts"][nc:2 * nc], [str(s) for s in z["names"][nc:2 * nc]],
                  t, win, u_perp, "⊥ axis — peaks aligned (no direction)", pk[nc:2 * nc])

    posE = z["posE"]; vth = z["vth"]; NE = len(posE)
    spread = local_vth_spread(posE, vth[:NE], np.ones(NE, bool), 0.3)
    onset_rel = np.asarray(z["onset_core"], float)
    fin = np.isfinite(onset_rel)
    onset_rel = onset_rel - (np.nanmin(onset_rel) if fin.any() else 0.0)
    vlim = ((float(np.nanpercentile(onset_rel[fin], 5)), float(np.nanpercentile(onset_rel[fin], 95)))
            if fin.sum() > 5 else None)
    foci = np.asarray(z["foci"]) if "foci" in z.files else np.asarray([z["patch"]])
    sign = float(z["sign"]) if "sign" in z.files else 0.0
    dirword = "forward" if sign > 0 else ("reverse" if sign < 0 else "no-direction")
    src_xy = z["patch"]            # the focus that self-ignited this representative event (star)
    pr = float(z["patch_r"])
    nfoci = "2 foci (dephased)" if foci.shape[0] > 1 else "1 focus"
    # draw EVERY lesion focus as a dashed core; the star (kick_xy) marks the one that ignited
    # THIS event. For one-focus configs extra is empty (single ring, as before).
    extra = [(float(f[0]), float(f[1]), pr) for f in foci
             if not np.allclose(f, src_xy)]

    mechanism_4panel(
        str(FIG / f"mechanism_spont_{tag}.png"),
        field_xy=posE, kick_xy=src_xy, axis_deg=theta, extent=(0, L, 0, L),
        map_a=dict(field_c=spread, clabel="local V_th spread (mV)", cmap="plasma",
                   vlim=None, color_contacts=False, title=f"lesion / heterogeneity map · {nfoci}"),
        map_b=dict(field_c=onset_rel, clabel="time after event onset (ms)", cmap=None,
                   vlim=vlim, color_contacts=True, title=f"spontaneous event — onset / propagation ({dirword})"),
        par=par, perp=perp, t=t, event_window=win,
        signal_ylabel="current-LFP (|I_E|+|I_I|)", name_fs=8, label_endpoints_only=True,
        contact_note="SPONTANEOUS (no kick) — star = lesion focus that self-ignited THIS event, "
                     "dashed rings = ALL lesion foci; contacts SCALED to model sheet, real 4mm "
                     "spacing; b-contacts coloured by arrival time; a boundary ring on mean-shifted "
                     "core = mean jump, see core interior",
        patch_circle=(float(src_xy[0]), float(src_xy[1]), pr), extra_patch_circles=extra,
        title=f"SPONTANEOUS lesion read-out — {tag}  ({nfoci}, representative {dirword} event) — "
              f"heterogeneity + propagation + electrode read-out")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(str(OUT / "per_event" / "rep_*.npz")))
    if not paths:
        print("no rep_*.npz found — run run_sef_hfo_snn_cm_spontaneous_readout.py first")
        return
    for pth in paths:
        _figure(pth)


if __name__ == "__main__":
    main()
