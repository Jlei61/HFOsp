"""Paper-ready subject-specific SNN panel (Topic 4), 1-row x 4-col, s3_brakeoff style.

Layout (matching scripts/paper_figures/plot_fig5_core_model_s3_brakeoff.py):
  mechanism | tempA source | tempB source | electrode readout

- mechanism: heterogeneity map (per-E-neuron 18-V_th, plasma), the two template-source cores
  (dashed circles), the patient electrodes (real contact layout), and the E->E long-axis corridor
  as a light band overlay. The core-member contacts are highlighted to show the cores OVERLAP the
  electrode interictal-event-onset (early) region.
- tempA source / tempB source: a representative propagation event when each template's source core
  is the ignition site (viridis relative onset), real contact layout.
- electrode readout: source-core run (tempA, forward shading) + sink-core run (tempB, reverse
  shading) concatenated on the patient montage; peak-locus per clean readable event.

Plotting-only: consumes figdata_<tag>.npz + readout_<tag>.json from run_sef_hfo_subject_snn.py
(template_source placement). No simulation rerun.
"""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/topic4_sef_hfo/field_swap_subject_snn"

FWD_SHADE, REV_SHADE = "#f4b266", "#78a6d8"
SHAFT_COLS = ["#e8743b", "#1f9e9e", "#7b5cb8", "#3b7a3b"]
AXIS_COL = "#a65f00"
OFF, SHADE_PAD_MS = 1.35, 22.0
DOT, ALPHA = 8.0, 0.90


def _axis(theta_deg):
    th = np.deg2rad(theta_deg); u = np.array([np.cos(th), np.sin(th)])
    return u, np.array([-u[1], u[0]])


def _shaft(name):
    m = re.match(r"[A-Za-z]+", str(name)); return m.group(0) if m else str(name)


def _shaft_color(name, shafts):
    return SHAFT_COLS[shafts.index(_shaft(name)) % len(SHAFT_COLS)]


def _load(tag):
    return (json.load(open(RUN / f"readout_{tag}.json")),
            np.load(RUN / f"figdata_{tag}.npz", allow_pickle=True))


def _style(ax, L):
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", fontsize=7.6); ax.set_ylabel("y (mm)", fontsize=7.6)
    ax.tick_params(axis="both", labelsize=7.0, length=2.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color("0.25")


def _ee_band(center, foci, u, p, core_r):
    half_w = max(1.5 * core_r, 1.6)
    proj = (np.asarray(foci) - center) @ u
    a = center + u * (float(proj.min()) - 2.0); b = center + u * (float(proj.max()) + 2.0)
    return np.vstack([a + half_w * p, b + half_w * p, b - half_w * p, a - half_w * p])


def _draw_contacts(ax, contacts, names, shafts, core_a, core_b):
    contacts = np.asarray(contacts, float)
    for sh in shafts:
        idx = [i for i, n in enumerate(names) if _shaft(n) == sh]
        c = contacts[idx]; col = _shaft_color(sh, shafts)
        ax.plot(c[:, 0], c[:, 1], color=col, lw=1.0, alpha=0.55, zorder=5)
        ax.scatter(c[:, 0], c[:, 1], s=34, marker="o", fc="white", ec=col, lw=1.0, zorder=6)
    # highlight the template-source core-member contacts (the early electrodes)
    for nm, hi in (("A", core_a), ("B", core_b)):
        ec = "crimson" if nm == "A" else "#1f4fd8"
        for n in hi:
            if n in names:
                xy = contacts[list(names).index(n)]
                ax.scatter([xy[0]], [xy[1]], s=70, marker="o", fc="none", ec=ec, lw=1.8, zorder=7)


def _plot_mechanism(ax, fd, core_a, core_b, shafts):
    pos = np.asarray(fd["posE"], float); v = np.asarray(fd["vth"], float)
    foci = np.asarray(fd["foci"], float); contacts = np.asarray(fd["contacts"], float)
    names = [str(x) for x in fd["names"]]; L = float(fd["L"]); core_r = float(fd["core_r"])
    u, p = _axis(float(fd["theta_deg"])); center = np.array([L / 2, L / 2])
    ax.scatter(pos[:, 0], pos[:, 1], c=np.clip(18.0 - v, 0.0, None), s=DOT, cmap="plasma",
               vmin=0.0, vmax=1.2, alpha=ALPHA, linewidths=0, rasterized=True, zorder=2)
    ax.add_patch(Polygon(_ee_band(center, foci, u, p, core_r), closed=True, fc=FWD_SHADE,
                         ec=AXIS_COL, lw=1.3, alpha=0.28, zorder=3))
    for i, f in enumerate(foci):
        ax.add_patch(plt.Circle(f, core_r, fill=False, ec="crimson", lw=1.25, ls="--", zorder=7))
        ax.text(f[0], f[1] + 1.0, "A" if i == 0 else "B", fontsize=9, color="crimson",
                fontweight="bold", ha="center", va="bottom",
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    ax.annotate("", xy=foci[1], xytext=foci[0],
                arrowprops=dict(arrowstyle="-|>", color=AXIS_COL, lw=1.7), zorder=8)
    _draw_contacts(ax, contacts, names, shafts, core_a, core_b)
    ax.set_title("mechanism", fontsize=9.5, fontweight="bold", pad=5); _style(ax, L)


def _plot_event(ax, fd, rep, title, shafts, core_a, core_b):
    pos = np.asarray(fd["posE"], float); foci = np.asarray(fd["foci"], float)
    contacts = np.asarray(fd["contacts"], float); names = [str(x) for x in fd["names"]]
    L = float(fd["L"]); core_r = float(fd["core_r"]); u, _ = _axis(float(fd["theta_deg"]))
    onset = np.asarray(rep["onset"], float) if rep else np.full(len(pos), np.nan)
    fin = np.isfinite(onset); bg = np.zeros(len(pos), bool); bg[::4] = True
    ax.scatter(pos[bg & ~fin, 0], pos[bg & ~fin, 1], s=1.2, c="0.86", alpha=0.35,
               linewidths=0, rasterized=True, zorder=1)
    if fin.any():
        rel = onset.copy(); rel[fin] -= np.nanmin(rel[fin])
        vmax = max(1.0, float(np.nanpercentile(rel[fin], 98)))
        ax.scatter(pos[fin, 0], pos[fin, 1], c=rel[fin], s=DOT, cmap="viridis", vmin=0.0,
                   vmax=vmax, alpha=ALPHA, linewidths=0, rasterized=True, zorder=2)
    # ignition core = A for tempA (foci[0]) / B for tempB (foci[1]); marked with star
    src_i = 0 if title.endswith("A source") else 1
    for i, f in enumerate(foci):
        ax.add_patch(plt.Circle(f, core_r, fill=False, ec="crimson", lw=1.2, ls="--", zorder=5))
        if i == src_i:
            ax.scatter([f[0]], [f[1]], marker="*", s=150, c="black", ec="white", lw=0.8, zorder=7)
    ax.plot([foci[0, 0], foci[1, 0]], [foci[0, 1], foci[1, 1]], color="0.20", lw=1.2, alpha=0.7, zorder=4)
    _draw_contacts(ax, contacts, names, shafts, core_a, core_b)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5); _style(ax, L)


def _active_order(names, contacts, u, p, ev_lists):
    active = set()
    for evs in ev_lists:
        for e in evs:
            active.update(n for n, val in (e.get("ranks") or {}).items() if val is not None)
    keep = [i for i, n in enumerate(names) if n in active] or list(range(len(names)))
    pp = np.array([contacts[i] @ u for i in keep]); qq = np.array([contacts[i] @ p for i in keep])
    order = np.lexsort((qq, pp)); return [keep[i] for i in order]


def _plot_readout(ax, fd, events, order, names, shafts, window_ms=5000.0):
    """Single twoend run: LFP train on the patient montage; forward events warm, reverse cool;
    peak-locus per clean readable event (matches the reference s3_brakeoff readout)."""
    lfp = np.abs(np.asarray(fd["lfp_trace"], float)); t = np.asarray(fd["times"], float)
    win_hi = min(float(window_ms), float(t[-1])); sel = (t >= 0.0) & (t <= win_hi); ts = t[sel]
    sub = lfp[sel][:, order].T
    base = np.median(sub, axis=1, keepdims=True)
    scale = np.maximum(sub.max(axis=1, keepdims=True) - base, 1e-9)
    zt = (sub - base) / scale
    y = np.arange(len(order)) * OFF
    for row, ci in enumerate(order):
        ax.plot(ts, zt[row] + y[row], color=_shaft_color(names[ci], shafts), lw=0.72, alpha=0.9, zorder=3)
    n_fwd = n_rev = 0
    for e in events:
        if e.get("sign") is None or e.get("n_part", 0) < 4 or e["t_on"] > win_hi:
            continue
        fwd = e["sign"] > 0; n_fwd += fwd; n_rev += not fwd
        shade = FWD_SHADE if fwd else REV_SHADE
        ax.axvspan(max(0.0, e["t_on"] - SHADE_PAD_MS), min(win_hi, e["t_off"] + SHADE_PAD_MS),
                   color=shade, alpha=0.26, lw=0, zorder=0)
        pts, ranks = [], (e.get("ranks") or {})
        for row, ci in enumerate(order):
            if ranks.get(names[ci]) is None:
                continue
            m = (ts >= e["t_on"]) & (ts <= e["t_off"])
            if m.sum() < 2:
                continue
            pi = np.flatnonzero(m)[int(np.argmax(zt[row][m]))]
            pts.append((ts[pi], zt[row][pi] + y[row]))
            ax.plot(ts[pi], zt[row][pi] + y[row], "o", ms=2.2, mfc="black", mec="white", mew=0.35, zorder=6)
        if len(pts) >= 2:
            px, py = zip(*sorted(pts)); ax.plot(px, py, "-", color="black", lw=0.72, alpha=0.7, zorder=5)
    ax.set_xlim(0, win_hi); ax.set_yticks(y); ax.set_yticklabels([names[i] for i in order], fontsize=7.0)
    for tick, ci in zip(ax.get_yticklabels(), order):
        tick.set_color(_shaft_color(names[ci], shafts))
    ax.tick_params(axis="y", length=2.5, labelsize=7.0, color="0.35")
    ax.tick_params(axis="x", length=3.0, labelsize=7.5, color="0.35")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.35"); ax.spines[side].set_linewidth(0.8)
    ax.set_xlabel("time (ms)", fontsize=8.2)
    ax.legend(handles=[Patch(facecolor=FWD_SHADE, alpha=0.4, label="tempA (forward) event"),
                       Patch(facecolor=REV_SHADE, alpha=0.4, label="tempB (reverse) event"),
                       Line2D([0], [0], color="black", marker="o", lw=0.8, ms=3, mfc="black", mec="white", label="peak order")],
              frameon=False, fontsize=7.4, loc="upper right", bbox_to_anchor=(1.0, 1.045),
              borderaxespad=0.0, ncol=3, handlelength=1.5, columnspacing=0.9)
    return {"forward_events": n_fwd, "reverse_events": n_rev}


def compose(twoend_tag, source_tag, sink_tag, fig_name, subject_label):
    to, tfd = _load(twoend_tag)            # twoend (spontaneous) -> mechanism + readout
    # single-run mode: if no dedicated source/sink runs, take the tempA/tempB event panels
    # from the twoend run's own best forward / best reverse spontaneous events (rep_fwd/rep_rev).
    single = not (source_tag and sink_tag)
    if single:
        so, sfd, ko, kfd = to, tfd, to, tfd
    else:
        so, sfd = _load(source_tag)        # source-core only -> clean tempA event panel
        ko, kfd = _load(sink_tag)          # sink-core only   -> clean tempB event panel
    reg = tfd["reg"].item(); core_a = list(reg["source_names"]); core_b = list(reg["sink_names"])
    names = [str(x) for x in tfd["names"]]; shafts = sorted(set(_shaft(n) for n in names))
    u, p = _axis(float(tfd["theta_deg"]))
    order = _active_order(names, np.asarray(tfd["contacts"], float), u, p, [to["events"]])
    rep_s = sfd["rep_fwd"].item() or sfd["rep_rev"].item()
    rep_k = kfd["rep_rev"].item() or kfd["rep_fwd"].item()

    fig = plt.figure(figsize=(18.0, 4.45), facecolor="white")
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.0, 1.0, 1.0, 2.75],
                           left=0.045, right=0.992, bottom=0.16, top=0.86, wspace=0.16)
    _plot_mechanism(fig.add_subplot(gs[0, 0]), tfd, core_a, core_b, shafts)
    _plot_event(fig.add_subplot(gs[0, 1]), sfd, rep_s, "tempA source", shafts, core_a, core_b)
    _plot_event(fig.add_subplot(gs[0, 2]), kfd, rep_k, "tempB source", shafts, core_a, core_b)
    # driven-pooled runs supply time-adjusted readout_window_events matching the concatenated trace;
    # spontaneous runs fall back to the run's own events.
    readout_evs = to.get("readout_window_events", to["events"])
    stats = _plot_readout(fig.add_subplot(gs[0, 3]), tfd, readout_evs, order, names, shafts)
    fig.text(0.012, 0.925, "A", fontsize=19, fontweight="bold")

    outdir = ROOT / f"results/paper-ready-figure/{fig_name}/figures"
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{fig_name}.png"; pdf = outdir / f"{fig_name}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white"); plt.close(fig)
    meta = dict(figure=fig_name, subject=subject_label,
                twoend_tag=twoend_tag, source_tag=source_tag, sink_tag=sink_tag,
                placement="template_source (earliest-3 of each template = the two template sources)",
                source_core=core_a, sink_core=core_b,
                twoend_spontaneous_dir=f"{to['dir_forward']}/{to['dir_reverse']}",
                source_only_dir=(None if single else f"{so['dir_forward']}/{so['dir_reverse']}"),
                sink_only_dir=(None if single else f"{ko['dir_forward']}/{ko['dir_reverse']}"),
                single_run_mode=single,
                readout_events=stats,
                notes=["Plotting-only; no SNN rerun.",
                       ("Single-run mode: mechanism + readout + tempA/tempB event panels ALL from the one "
                        "spontaneous twoend run (tempA=best forward event, tempB=best reverse event)." if single
                        else "Mechanism + readout = spontaneous twoend run; tempA/tempB panels = source-only/sink-only runs."),
                       "k_dir=2 sparse-electrode readout; real-geometry plane-fit (no core-anchoring)."])
    (outdir / f"{fig_name}_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {png}\nwrote {pdf}\nreadout {stats}")
    return outdir, meta


def main():
    os.chdir(ROOT)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--twoend-tag", default="epilepsiae_1146_twoend_equal_tsrc_s3")
    ap.add_argument("--source-tag", default=None, help="omit for single-run mode (tempA from twoend rep_fwd)")
    ap.add_argument("--sink-tag", default=None, help="omit for single-run mode (tempB from twoend rep_rev)")
    ap.add_argument("--fig-name", default="fig_subject_snn_epilepsiae_1146")
    ap.add_argument("--label", default="epilepsiae_1146 (ICL strip)")
    a = ap.parse_args()
    compose(a.twoend_tag, a.source_tag, a.sink_tag, a.fig_name, a.label)


if __name__ == "__main__":
    main()
