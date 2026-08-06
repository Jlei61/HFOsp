#!/usr/bin/env python
"""Fig — the lifecycle mechanism on the patient's own geometry, in the four-column standard.

Same single-row grammar as the core-model figure: what the mechanism is, one propagation event
from each direction, and what the same virtual montage reads out.  What differs is the run behind
it -- the slow variables are on, so the readout column carries a whole lifecycle rather than a
stretch of interictal train: the events that precede entry, the entry itself, and the discharge.

Two things about this figure are not the exemplar's and are stated here, in the metadata and in
the figures README rather than left for a reader to assume:

* the contact traces are **spike-weighted** through the montage's own distance weighting, not the
  current-based local field, because the slow-variable loop does not expose per-step currents;
* the two shafts are the subject's real ones (SCL, ICL), not a synthetic A/B pair, so the locked
  shaft colours are applied to those.

Scientific boundary, per the style guide: this is a model substrate + two directions + virtual
electrode readout illustration.  Forward and reverse events appearing here do not establish the
real patient's mechanism.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/subject_runaway"
PAPER = ROOT / "results/paper-ready-figure/fig_subject_snn_e1146_runaway/figures"

FWD_SHADE = "#f4b266"          # locked: forward is warm
REV_SHADE = "#78a6d8"          # locked: reverse is light blue
SHAFT_COLOR = {"SCL": "#e8743b", "ICL": "#1f9e9e"}   # locked: A shaft orange, B shaft cyan
AXIS_COL = "#a65f00"
SPATIAL_DOT = 8.0
SPATIAL_ALPHA = 0.90


def _shaft(name):
    return "SCL" if str(name).upper().startswith("SCL") else "ICL"


def _style_spatial(ax, L):
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", fontsize=8.5)
    ax.set_ylabel("y (mm)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)


def _endpoint_labels(contacts, names):
    """Only each shaft's two extremes get a label; fifteen of them collide into mush."""
    keep = set()
    for sh in set(_shaft(n) for n in names):
        idx = [i for i, n in enumerate(names) if _shaft(n) == sh]
        if not idx:
            continue
        xs = np.asarray(contacts)[idx]
        span = xs.max(axis=0) - xs.min(axis=0)
        along = int(np.argmax(np.abs(span)))
        keep.add(idx[int(np.argmin(xs[:, along]))])
        keep.add(idx[int(np.argmax(xs[:, along]))])
    return keep


def _draw_contacts(ax, contacts, names, label=True):
    keep = _endpoint_labels(contacts, names) if label else set()
    for i, (xy, nm) in enumerate(zip(contacts, names)):
        c = SHAFT_COLOR[_shaft(nm)]
        ax.scatter([xy[0]], [xy[1]], marker="s", s=22, facecolor="none",
                   edgecolor=c, lw=1.2, zorder=9)
        if i in keep:
            ax.text(xy[0], xy[1] + 0.55, str(nm), fontsize=6.4, color=c, ha="center",
                    va="bottom", zorder=10,
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])


def _axis_band(center, axis_unit, half_len, half_width):
    perp = np.array([-axis_unit[1], axis_unit[0]])
    a = center - axis_unit * half_len
    b = center + axis_unit * half_len
    return [a + perp * half_width, b + perp * half_width,
            b - perp * half_width, a - perp * half_width]


def _plot_mechanism(ax, z):
    posE = np.asarray(z["posE"], float)
    vth = np.asarray(z["vth"], float)[:len(posE)]
    contacts = np.asarray(z["contacts"], float)
    names = [str(n) for n in z["contact_names"]]
    L = float(z["L"][0])
    src, snk = np.asarray(z["src_xy"], float), np.asarray(z["snk_xy"], float)
    axis_unit = np.asarray(z["axis_unit"], float)
    core_r = float(z["core_r"][0])
    center = (src + snk) / 2.0

    ax.scatter(posE[:, 0], posE[:, 1], c=np.clip(18.0 - vth, 0.0, None),
               s=SPATIAL_DOT, cmap="plasma", vmin=0.0, vmax=1.2, alpha=SPATIAL_ALPHA,
               linewidths=0, rasterized=True, zorder=2)
    half = float(np.linalg.norm(snk - src)) / 2.0 + core_r
    ax.add_patch(Polygon(_axis_band(center, axis_unit, half, 0.55 * core_r), closed=True,
                         fc=FWD_SHADE, ec=AXIS_COL, lw=1.0, alpha=0.18, zorder=3))
    for f, mark in ((src, "-"), (snk, "+")):
        ax.add_patch(plt.Circle(f, core_r, fill=False, ec="crimson", lw=1.25, ls="--", zorder=7))
        ax.text(f[0], f[1] + core_r + 0.35, mark, fontsize=9, color="crimson",
                fontweight="bold", ha="center", va="bottom",
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    ax.annotate("", xy=center + axis_unit * half, xytext=center - axis_unit * half,
                arrowprops=dict(arrowstyle="-|>", color=AXIS_COL, lw=1.7), zorder=8)
    _draw_contacts(ax, contacts, names)
    ax.set_title("mechanism", fontsize=9.5, fontweight="bold", pad=5)
    _style_spatial(ax, L)


def _plot_event(ax, z, onset, title, src_side):
    posE = np.asarray(z["posE"], float)
    contacts = np.asarray(z["contacts"], float)
    names = [str(n) for n in z["contact_names"]]
    L = float(z["L"][0])
    src, snk = np.asarray(z["src_xy"], float), np.asarray(z["snk_xy"], float)
    core_r = float(z["core_r"][0])
    fin = np.isfinite(onset)

    bg = np.zeros(len(posE), bool)
    bg[::4] = True
    ax.scatter(posE[bg & ~fin, 0], posE[bg & ~fin, 1], s=1.2, c="0.86", alpha=0.35,
               linewidths=0, rasterized=True, zorder=1)
    if fin.any():
        rel = onset[fin] - np.nanmin(onset[fin])
        vmax = max(1.0, float(np.percentile(rel, 98)))
        ax.scatter(posE[fin, 0], posE[fin, 1], c=rel, s=SPATIAL_DOT, cmap="viridis",
                   vmin=0.0, vmax=vmax, alpha=SPATIAL_ALPHA, linewidths=0,
                   rasterized=True, zorder=2)
    for i, f in enumerate((src, snk)):
        ax.add_patch(plt.Circle(f, core_r, fill=False, ec="crimson", lw=1.2, ls="--", zorder=5))
        if (i == 0) == (src_side == "source"):
            ax.scatter([f[0]], [f[1]], marker="*", s=150, c="black", ec="white",
                       lw=0.8, zorder=7)
    ax.plot([src[0], snk[0]], [src[1], snk[1]], color="0.20", lw=1.2, alpha=0.75, zorder=4)
    _draw_contacts(ax, contacts, names, label=False)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)
    _style_spatial(ax, L)


def _plot_readout(ax, z, rec):
    trace = np.asarray(z["contact_trace"], float)
    names = [str(n) for n in z["contact_names"]]
    bin_ms = float(z["contact_bin_ms"][0])
    t = np.arange(trace.shape[0]) * bin_ms / 1000.0
    peak = trace.max(axis=0)
    active = np.where(peak > 0.05 * peak.max())[0]
    order = sorted(active, key=lambda k: (_shaft(names[k]), names[k]))

    onset_ms = rec.get("onset_ms")
    for e in rec["interictal_directions"]["events"]:
        if e["direction"] is None:
            continue
        c = FWD_SHADE if e["direction"] == "forward" else REV_SHADE
        t0 = e["t_on_ms"] / 1000.0
        ax.axvspan(t0 - 0.02, t0 + e["dur_ms"] / 1000.0 + 0.02, color=c, alpha=0.75,
                   lw=0, zorder=1)
    if onset_ms is not None:
        ax.axvline(onset_ms / 1000.0, color="#111111", lw=1.4, zorder=6)
        ax.text(onset_ms / 1000.0 + 0.4, len(order) - 0.9, "enters", fontsize=8,
                color="#111111", va="top", zorder=7,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

    span = 1.0
    for row, k in enumerate(order):
        v = trace[:, k]
        v = v / max(v.max(), 1e-12)
        ax.plot(t, row + v * 0.85 * span, lw=0.55, color=SHAFT_COLOR[_shaft(names[k])], zorder=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[k] for k in order], fontsize=6.5)
    for lab, k in zip(ax.get_yticklabels(), order):
        lab.set_color(SHAFT_COLOR[_shaft(names[k])])
    ax.set_xlabel("time (s)", fontsize=8.5)
    ax.set_xlim(0, t[-1] if t.size else 1.0)
    ax.set_ylim(-0.6, len(order))
    ax.tick_params(axis="x", labelsize=7.5)
    ax.legend(handles=[Patch(fc=FWD_SHADE, label="forward event"),
                       Patch(fc=REV_SHADE, label="reverse event"),
                       Line2D([], [], color="#111111", lw=1.4, label="entry")],
              frameon=False, fontsize=7.5, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.005))


def compose(conn_seed):
    rec = json.load(open(RUN / f"seed{conn_seed}.json"))
    z = np.load(RUN / f"seed{conn_seed}_traces.npz", allow_pickle=True)
    cap_path = RUN / f"seed{conn_seed}_events.json"
    onsets_path = RUN / f"seed{conn_seed}_events_onsets.npz"
    picks, maps = None, None
    if cap_path.is_file() and onsets_path.is_file():
        cap = json.load(open(cap_path))
        oz = np.load(onsets_path)
        maps = np.asarray(oz["onset_maps"], float)
        fwd = [(i, e) for i, e in enumerate(cap["events"]) if e["direction"] == "forward"]
        rev = [(i, e) for i, e in enumerate(cap["events"]) if e["direction"] == "reverse"]
        if fwd and rev:
            picks = (max(fwd, key=lambda p: abs(p[1]["axis_corr"])),
                     max(rev, key=lambda p: abs(p[1]["axis_corr"])))

    fig = plt.figure(figsize=(18.0, 4.45), facecolor="white")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 2.75], wspace=0.28,
                          left=0.035, right=0.985, top=0.88, bottom=0.14)
    _plot_mechanism(fig.add_subplot(gs[0, 0]), z)
    for col, side, title in ((1, "source", "tempA source"), (2, "sink", "tempB source")):
        ax = fig.add_subplot(gs[0, col])
        if picks is None:
            ax.text(0.5, 0.5, "no event of this direction\nbefore entry", ha="center",
                    va="center", fontsize=9, color="#7f8c8d", transform=ax.transAxes)
            ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)
            _style_spatial(ax, float(z["L"][0]))
        else:
            idx = picks[col - 1][0]
            _plot_event(ax, z, maps[idx], title, side)
    _plot_readout(fig.add_subplot(gs[0, 3]), z, rec)

    PAPER.mkdir(parents=True, exist_ok=True)
    stem = PAPER / f"fig_subject_snn_e1146_runaway_seed{conn_seed}"
    fig.savefig(f"{stem}.png", dpi=200)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)

    d = rec["interictal_directions"]
    meta = dict(
        figure=f"E1146 subject SNN — lifecycle on the patient's geometry (seed {conn_seed})",
        layout="single row: mechanism | tempA source | tempB source | electrode readout",
        subject=rec["subject"], montage=rec["montage"],
        connection_seed=conn_seed, noise_seed=rec["noise_seed"], run_ms=rec["run_ms"],
        source_contacts=rec["source_names"], sink_contacts=rec["sink_names"],
        n_contacts=len(rec["contact_names"]), n_contacts_offsheet=rec["n_contacts_offsheet"],
        ran_away=rec["ran_away"], onset_ms=rec["onset_ms"],
        interictal=dict(n_forward=d["n_forward"], n_reverse=d["n_reverse"],
                        n_undetermined=d["n_undetermined"], bidirectional=d["bidirectional"]),
        n_returning_before_onset=rec["n_returning_before_onset"],
        entry_class=rec["entry_class"],
        middle_columns=("representative forward and reverse pre-entry events, per-cell onset"
                        if picks is not None else
                        "not drawn: the run did not produce both directions before entry"),
        readout_kind=rec["readout_kind"],
        shaft_colours="the subject's real shafts SCL and ICL, not a synthetic A/B pair",
        boundary=("model substrate + two directions + virtual electrode readout; forward and "
                  "reverse events here do not establish the real patient's mechanism"),
        outputs=[str(Path(f"{stem}.png").relative_to(ROOT)),
                 str(Path(f"{stem}.pdf").relative_to(ROOT))])
    with open(f"{stem}_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    return stem, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="3,1")
    args = ap.parse_args()
    for s in [int(x) for x in args.seeds.split(",") if x.strip()]:
        if not (RUN / f"seed{s}.json").is_file():
            print(f"  seed {s}: no run on disk yet")
            continue
        stem, meta = compose(s)
        print(f"  seed {s}: {stem}.png")
        print(f"    ran_away={meta['ran_away']} onset={meta['onset_ms']} "
              f"interictal {meta['interictal']['n_forward']}fwd/"
              f"{meta['interictal']['n_reverse']}rev  middle columns: {meta['middle_columns']}")


if __name__ == "__main__":
    main()
