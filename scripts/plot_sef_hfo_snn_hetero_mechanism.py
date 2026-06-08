"""Figures from the heterogeneity grid (spec 2026-06-08; 2x2 + propagation view
2026-06-08c per review). Per representative we draw TWO figures:
  propagation_<tag>.png  — reference-style: panel-a coloured by ONSET time, electrode
                           CONTACTS coloured by arrival time, b/c peak-centroid locus
                           (so the peak-time propagation ORDER is visible)
  heterogeneity_<tag>.png — panel-a coloured by LOCAL V_th SPREAD (distinct 'plasma'
                           colorbar) + pathology-core outline (what/where/how-big)
Plus grid_overview.png (2x2: variance/mean/combined, ignition marked) and
baseline_compare.png (source-space rate: shows pre-kick self-ignition directly).

Representatives = the mid-patch 2x2 (matched/mean_only/unmatched vs baseline) +
the max-effect cell. d_core_paf is a within-evoked-event measure ONLY for
evoked_clean cells; igniting cells are flagged (review 1B).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.sef_hfo_plot import mechanism_4panel
from src.sef_hfo_heterogeneity import local_vth_spread

OUT = Path("results/topic4_sef_hfo/snn_heterogeneity")
FIG = OUT / "figures"
T_KICK = 150.0
COND_COLOR = {"matched": "steelblue", "mean_only": "darkorange", "unmatched": "firebrick"}


def _effect(c):
    return abs(c["d_core_paf"]) if c.get("d_core_paf") is not None else 0.0


def _contact_peak_times(lfp, t, win):
    """Per-contact peak time (ms) of the LFP within win=(t0,t1)."""
    m = (t >= win[0]) & (t <= win[1])
    idx = np.flatnonzero(m)
    return np.array([t[idx[int(np.argmax(lfp[ci, m]))]] for ci in range(lfp.shape[0])])


def _shaft(lfp, contacts, names, t, win, u_shaft, panel_title, contact_c):
    pre = t < win[0]; inwin = (t >= win[0]) & (t <= win[1])
    base = lfp[:, pre].mean(axis=1); bstd = lfp[:, pre].std(axis=1) + 1e-9
    part = ((lfp[:, inwin].max(axis=1) - base) / bstd) > 30.0
    c = np.asarray(contacts, float)
    return dict(contacts=c, part=part, names=list(names), s=(c - c.mean(0)) @ u_shaft,
                signal=lfp, panel_title=panel_title, contact_c=contact_c)


def _load(cell):
    z = np.load(OUT / "per_cell" / f"cell{cell['idx']:02d}.npz", allow_pickle=True)
    L = float(z["L"]); theta = float(z["theta"]); nc = int(z["nc"])
    t = np.asarray(z["times"]); lfp = np.asarray(z["lfp"]).T          # (n_contact, nt)
    ev_t = float(z["event_peak_t"]) if "event_peak_t" in z.files else 176.0
    win = (ev_t - 25.0, ev_t + 30.0)
    u_par = np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))])
    u_perp = np.array([np.cos(np.deg2rad(theta + 90)), np.sin(np.deg2rad(theta + 90))])
    pk = _contact_peak_times(lfp, t, win) - T_KICK                   # ms after kick
    par = _shaft(lfp[:nc], z["contacts"][:nc], [str(s) for s in z["names"][:nc]], t, win,
                 u_par, "∥ axis — peaks sweep (reads direction)", pk[:nc])
    perp = _shaft(lfp[nc:2 * nc], z["contacts"][nc:2 * nc],
                  [str(s) for s in z["names"][nc:2 * nc]], t, win, u_perp,
                  "⊥ axis — peaks aligned (no direction)", pk[nc:2 * nc])
    return z, L, theta, t, win, par, perp


def _mechanism_figure(cell, tag):
    """One 4-panel figure: a=heterogeneity map | b=onset/propagation map |
    c=∥ read-out | d=⊥ read-out (c/d shared, not duplicated)."""
    z, L, theta, t, win, par, perp = _load(cell)
    posE = z["posE"]; vth = z["vth"]; NE = len(posE)
    spread = local_vth_spread(posE, vth[:NE], np.ones(NE, bool), 0.3)
    onset_rel = np.asarray(z["onset_core"], float) - np.nanmin(z["onset_core"])
    fin = np.isfinite(onset_rel)
    # clip onset colour to 5–95th pct so late-firing outliers don't blow out the gradient
    vlim = ((float(np.nanpercentile(onset_rel[fin], 5)), float(np.nanpercentile(onset_rel[fin], 95)))
            if fin.sum() > 5 else None)
    mechanism_4panel(
        str(FIG / f"mechanism_{tag}.png"),
        field_xy=posE, kick_xy=z["kick"], axis_deg=theta, extent=(0, L, 0, L),
        map_a=dict(field_c=spread, clabel="local V_th spread (mV)", cmap="plasma",
                   vlim=None, color_contacts=False,
                   title=f"heterogeneity map · {cell['cond']} core"),
        map_b=dict(field_c=onset_rel, clabel="time after event onset (ms)", cmap=None,
                   vlim=vlim, color_contacts=True, title="onset / propagation map"),
        par=par, perp=perp, t=t, event_window=win, name_fs=8, label_endpoints_only=True,
        signal_ylabel="current-LFP (|I_E|+|I_I|)",
        contact_note="contacts SCALED to model sheet — NOT real SEEG spacing; firing-density read-out "
                     "(NOT LFP); dashed = pathology core; b-contacts coloured by arrival time; "
                     "a heterogeneity boundary ring on mean-shifted cores = mean jump, see core interior",
        patch_circle=(float(z["patch"][0]), float(z["patch"][1]), float(z["patch_r"])),
        title=f"Pathology core ({cell['cond']}) — heterogeneity + propagation + electrode read-out ({tag})")


def _overview(cells):
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    vals = [(c["d_core_paf"] if c["d_core_paf"] is not None else np.nan) for c in cells]
    cols = [COND_COLOR.get(c["cond"], "0.5") for c in cells]
    bars = ax.bar(range(len(cells)), vals, color=cols)
    for i, c in enumerate(cells):
        if c["core_prekick_ignited"]:
            bars[i].set_hatch("///")                                 # ignited = not within-event synchrony
        if c["kick_on_patch"]:
            bars[i].set_edgecolor("black"); bars[i].set_linewidth(1.4)
    labels = [f"s{c['sweep']}\n{c['kname']}/{c['pname']}\n{c['cond'][:4]}" for c in cells]
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(cells))); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("Δ core peak-active-fraction (core − baseline)")
    ax.set_title("Heterogeneity 2×2: blue=matched(variance), orange=mean_only, red=unmatched(combined)\n"
                 "hatch = core self-ignites pre-kick (state index, NOT within-event synchrony) · "
                 "black edge = kick-on-patch", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "grid_overview.png", dpi=130); plt.close(fig)


def _pick_representatives(cells):
    picked, used = {}, set()

    def take(tag, c):
        if c is not None and c["idx"] not in used:
            picked[tag] = c; used.add(c["idx"])

    def find(pname, cond):
        cand = [c for c in cells if c["pname"] == pname and c["cond"] == cond
                and c["sweep"] == 2 and not c["kick_on_patch"]]
        return cand[0] if cand else None

    take("mid_matched", find("mid", "matched"))         # variance axis (clean evoked)
    take("mid_mean_only", find("mid", "mean_only"))     # mean axis (self-ignites)
    take("mid_unmatched", find("mid", "unmatched"))     # combined
    take("maxeffect", max(cells, key=_effect))
    return picked


def _baseline_compare(reps):
    n = len(reps)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 3.2), squeeze=False)
    for ax, (tag, c) in zip(axes[0], reps.items()):
        z = np.load(OUT / "per_cell" / f"cell{c['idx']:02d}.npz", allow_pickle=True)
        t = np.asarray(z["times"])
        ax.axvline(T_KICK, color="0.7", ls=":", lw=1.0)
        ax.plot(t, z["base_rate"], color="0.6", lw=1.1, label="baseline (wide)")
        ax.plot(t, z["rate"], color=COND_COLOR.get(c["cond"], "C3"), lw=1.1, label=c["cond"])
        ig = c["core_prekick_ignited"]
        dc = c["d_core_paf"]
        ttl = f"{tag}\n{'IGNITES pre-kick' if ig else 'evoked'}  Δpaf={dc:+.3f}" if dc is not None else tag
        ax.set_title(ttl, fontsize=8)
        ax.set_xlabel("time (ms)"); ax.legend(fontsize=6)
    axes[0, 0].set_ylabel("E population rate (Hz)")
    fig.suptitle("Network state: baseline vs pathology core (source-space rate; "
                 "dotted = kick @150ms)")
    fig.tight_layout(); fig.savefig(FIG / "baseline_compare.png", dpi=130); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for pat in ("propagation_*", "heterogeneity_*", "mechanism_*"):   # drop stale split/old names
        for ext in ("png", "pdf"):
            for old in FIG.glob(f"{pat}.{ext}"):
                old.unlink()
    cells = json.loads((OUT / "grid_metrics.json").read_text())["cells"]
    _overview(cells)
    reps = _pick_representatives(cells)
    for tag, c in reps.items():
        _mechanism_figure(c, tag)
    _baseline_compare(reps)
    (OUT / "cohort_summary.json").write_text(json.dumps(
        {"representatives": {k: v["idx"] for k, v in reps.items()}, "n_cells": len(cells)},
        indent=2))
    print("representatives:", {k: v["idx"] for k, v in reps.items()})
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
