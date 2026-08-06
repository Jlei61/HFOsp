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

The two middle columns are keyed by **source identity**, which is what the four-column standard
asks of them: each shows an event that started at one of the two template source regions, with
the star on that event's own ignition site.  They are not keyed by the sign of an onset-vs-axis
correlation.  Most pre-entry events here are patches around one core rather than transits between
the two, and for a patch that sign reports which flank of the ignition is longer -- a different
question from where the event began.

Scientific boundary, per the style guide: this is a model substrate + two source regions + virtual
electrode readout illustration.  Events of either source identity appearing here do not establish
the real patient's mechanism.
"""
from __future__ import annotations

import argparse
import json
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_fcxr_lc3_ignition import (  # noqa: E402
    CLOSER_BY,
    EARLY_Q,
    classify_events,
    pick_representatives,
)

RUN = ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/subject_runaway"
PAPER = ROOT / "results/paper-ready-figure/fig_subject_snn_e1146_runaway/figures"

A_SHADE = "#f4b266"            # locked: warm for the first source region
B_SHADE = "#78a6d8"            # locked: light blue for the second
MID_SHADE = "#c9ccd1"          # ignition midway between them: attributable to neither
BAND_FILL = "#f4b266"          # the E->E long-axis band in the mechanism panel
READOUT_TAIL_S = 3.0           # seconds of discharge kept after entry
PEAK_FRAC = 0.30               # a contact took part in an event at this share of the event peak
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
                         fc=BAND_FILL, ec=AXIS_COL, lw=1.0, alpha=0.18, zorder=3))
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


def _plot_event(ax, z, onset, title, star_xy, empty_note=None):
    """``onset`` may be None: the region ignited nothing, and the panel says so on the geometry
    rather than as a blank box, so a reader can see *where* nothing happened."""
    posE = np.asarray(z["posE"], float)
    contacts = np.asarray(z["contacts"], float)
    names = [str(n) for n in z["contact_names"]]
    L = float(z["L"][0])
    src, snk = np.asarray(z["src_xy"], float), np.asarray(z["snk_xy"], float)
    core_r = float(z["core_r"][0])
    fin = np.isfinite(onset) if onset is not None else np.zeros(len(posE), bool)

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
    if empty_note is not None:
        ax.text(0.5, 0.955, empty_note, ha="center", va="top", fontsize=8.2, color="#7f8c8d",
                transform=ax.transAxes, zorder=11,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    for f in (src, snk):
        ax.add_patch(plt.Circle(f, core_r, fill=False, ec="crimson", lw=1.2, ls="--", zorder=5))
    ax.plot([src[0], snk[0]], [src[1], snk[1]], color="0.20", lw=1.2, alpha=0.75, zorder=4)
    if star_xy is not None:
        ax.scatter([star_xy[0]], [star_xy[1]], marker="*", s=150, c="black", ec="white",
                   lw=0.8, zorder=7)
    _draw_contacts(ax, contacts, names, label=False)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)
    _style_spatial(ax, L)


def _plot_readout(ax, z, rec, shaded):
    """``shaded`` maps each captured event to "a" / "b" / None by its ignition site."""
    trace = np.asarray(z["contact_trace"], float)
    names = [str(n) for n in z["contact_names"]]
    bin_ms = float(z["contact_bin_ms"][0])
    t = np.arange(trace.shape[0]) * bin_ms / 1000.0
    peak = trace.max(axis=0)
    active = np.where(peak > 0.05 * peak.max())[0]
    order = sorted(active, key=lambda k: (_shaft(names[k]), names[k]))

    onset_ms = rec.get("onset_ms")
    t_end = (onset_ms / 1000.0 + READOUT_TAIL_S) if onset_ms is not None else t[-1]
    t_end = min(t_end, t[-1] if t.size else t_end)

    for e, side in shaded:
        c = {"a": A_SHADE, "b": B_SHADE}.get(side, MID_SHADE)
        t0 = e["t_on_ms"] / 1000.0
        ax.axvspan(t0 - 0.02, t0 + e["dur_ms"] / 1000.0 + 0.02, color=c, alpha=0.75,
                   lw=0, zorder=1)
    if onset_ms is not None:
        ax.axvline(onset_ms / 1000.0, color="#111111", lw=1.4, zorder=6)
        ax.text(onset_ms / 1000.0 + 0.12, len(order) - 0.9, "enters", fontsize=8,
                color="#111111", va="top", zorder=7,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

    # The trace is already a fraction: each contact's weights sum to 1, so a value is the share
    # of that contact's neighbourhood recruited in the bin.  Plot it as-is -- rescaling each row
    # to its own maximum would make rows look alike whatever they actually recruited.
    for row, k in enumerate(order):
        ax.plot(t, row + trace[:, k] * 0.85, lw=0.55,
                color=SHAFT_COLOR[_shaft(names[k])], zorder=4)

    # peak order within each event: when each participating contact peaks
    for e, _side in shaded:
        i0 = int(round(e["t_on_ms"] / bin_ms))
        i1 = min(trace.shape[0], int(round(e["t_off_ms"] / bin_ms)) + 1)
        if i1 - i0 < 2:
            continue
        seg = trace[i0:i1, order]
        hi = seg.max(axis=0)
        took_part = hi >= PEAK_FRAC * hi.max() if hi.max() > 0 else np.zeros(len(order), bool)
        rows = np.where(took_part)[0]
        if rows.size:
            ax.scatter((i0 + seg[:, rows].argmax(axis=0)) * bin_ms / 1000.0, rows,
                       s=3.2, c="#111111", linewidths=0, zorder=8)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[k] for k in order], fontsize=6.5)
    for lab, k in zip(ax.get_yticklabels(), order):
        lab.set_color(SHAFT_COLOR[_shaft(names[k])])
    ax.set_xlabel("time (s)", fontsize=8.5)
    ax.set_xlim(0, t_end)
    ax.set_ylim(-0.6, len(order))
    ax.tick_params(axis="x", labelsize=7.5)
    if t.size and t_end < t[-1] - 1e-9:
        ax.text(0.995, 0.012, f"discharge continues to {t[-1]:.0f} s", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.0, color="#444444",
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")], zorder=9)
    handles = [Patch(fc=A_SHADE, label="tempA-source event"),
               Patch(fc=B_SHADE, label="tempB-source event")]
    if any(side is None for _e, side in shaded):
        handles.append(Patch(fc=MID_SHADE, label="mid-axis start (neither)"))
    ax.legend(handles=handles + [
               Line2D([], [], color="#111111", marker="o", ls="none", ms=2.6,
                      label="contact peak order"),
               Line2D([], [], color="#111111", lw=1.4, label="entry")],
              frameon=False, fontsize=7.5, ncol=5, loc="lower center",
              bbox_to_anchor=(0.5, 1.005))


def compose(conn_seed):
    rec = json.load(open(RUN / f"seed{conn_seed}.json"))
    z = np.load(RUN / f"seed{conn_seed}_traces.npz", allow_pickle=True)
    cap_path = RUN / f"seed{conn_seed}_events.json"
    onsets_path = RUN / f"seed{conn_seed}_events_onsets.npz"
    cls, picks, maps, shaded = [], (None, None), None, []
    if cap_path.is_file() and onsets_path.is_file():
        cap = json.load(open(cap_path))
        maps = np.asarray(np.load(onsets_path)["onset_maps"], float)
        cls = classify_events(maps, np.asarray(z["posE"], float),
                              np.asarray(z["src_xy"], float), np.asarray(z["snk_xy"], float))
        picks = pick_representatives(cls)
        shaded = list(zip(cap["events"], [c["source"] for c in cls]))

    fig = plt.figure(figsize=(18.0, 4.45), facecolor="white")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 2.75], wspace=0.28,
                          left=0.035, right=0.985, top=0.88, bottom=0.14)
    _plot_mechanism(fig.add_subplot(gs[0, 0]), z)
    for col, title in ((1, "tempA source"), (2, "tempB source")):
        ax = fig.add_subplot(gs[0, col])
        idx = picks[col - 1]
        if idx is None:
            note = ("no per-cell onset capture on disk" if not cls else
                    "no event started at this source region before entry")
            _plot_event(ax, z, None, title, None, empty_note=note)
        else:
            _plot_event(ax, z, maps[idx], title, cls[idx]["ignition_xy"])
    _plot_readout(fig.add_subplot(gs[0, 3]), z, rec, shaded)

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
        middle_columns=dict(
            keyed_by="source identity: which of the two template source regions ignited",
            rule=(f"ignition = centroid of the earliest {EARLY_Q:g}% of cells to fire; a region "
                  f"owns it when it is >={CLOSER_BY:g}x nearer than the other; the star marks "
                  f"the event's own ignition, not a core centre"),
            picked=dict(tempA=picks[0], tempB=picks[1]),
            per_event=[dict(t_on_ms=e["t_on_ms"], source=c["source"],
                            dist_a_mm=c["dist_a_mm"], dist_b_mm=c["dist_b_mm"],
                            axis_corr=e["axis_corr"])
                       for e, c in zip(cap["events"], cls)] if cls else [],
            n_from_a=sum(1 for c in cls if c["source"] == "a"),
            n_from_b=sum(1 for c in cls if c["source"] == "b"),
            n_unattributed=sum(1 for c in cls if c["source"] is None),
        ) if cls else "not drawn: no per-cell onset capture on disk",
        readout_kind=rec["readout_kind"],
        readout_window=(f"0 to entry + {READOUT_TAIL_S:g} s so the pre-entry events are legible; "
                        f"the discharge runs to the end of the "
                        f"{rec['run_ms'] / 1000:.0f} s record"),
        readout_peak_marks=(f"a contact is marked in an event when its within-event peak reaches "
                            f"{PEAK_FRAC:g} of the largest contact peak in that event"),
        readout_units=("each trace is the share of that contact's weighted neighbourhood "
                       "recruited per 2 ms bin (contact weights sum to 1), so rows are "
                       "comparable to each other and 1.0 means the whole neighbourhood fired"),
        shaft_colours="the subject's real shafts SCL and ICL, not a synthetic A/B pair",
        boundary=("model substrate + two source regions + virtual electrode readout; events of "
                  "either source identity here do not establish the real patient's mechanism. "
                  "Most pre-entry events are patches around one core, not transits between the "
                  "two -- see middle_columns.per_event for each event's ignition distances"),
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
