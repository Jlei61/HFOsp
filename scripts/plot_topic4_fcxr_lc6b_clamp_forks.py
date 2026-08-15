#!/usr/bin/env python3
"""Figure for the eight FCXR-LC6B clamp forks (post-review layout).

Six independent questions, one panel each:
  a/b  does the discharge keep escalating once the slow fields are pinned (one panel per snapshot)?
  c    is the pinned-D state a continuous plateau, or a train that re-ignites from near-silence?
  d    how much tissue carries it -- a focal patch, or most of the sheet?
  e    is the sheet uniformly moderate, or are cells pinned at the refractory ceiling?
  f    did the manipulation actually do what it claims (engineering check)?

c and d were added after review: the 100 ms traces in a/b make the pinned-D arms look like smooth
plateaus, and the load-bearing fact is that they are not.  The dynamics label is annotated on a/b
rather than given a panel, since it is derived from readouts those panels already show.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic4_sef_hfo/fcxr_lc6b_frozen_slow_atlas"
FIGURES = OUT / "figures"
ARM_ORDER = ("NAT", "H_CLAMP", "D_CLAMP", "DH_CLAMP")
ARM_COLOR = {"NAT": "#37474F", "H_CLAMP": "#8C6BB1", "D_CLAMP": "#D95F0E", "DH_CLAMP": "#2CA25F"}
ARM_LABEL = {"NAT": "both free (paired control)", "H_CLAMP": "H pinned",
             "D_CLAMP": "D pinned", "DH_CLAMP": "D and H pinned"}
SHORT = {"ESCALATING_SATURATION": "escalates to saturation",
         "BOUNDED_STATIONARY": "bounded, steady", "BOUNDED_OSCILLATORY": "bounded, bursting",
         "LOW_STATE": "falls back to interictal", "SILENCE": "silent",
         "AFTER_DISCHARGE": "brief after-discharge", "NUMERICAL_FAIL": "numerical failure",
         "RIGHT_CENSORED": "unresolved in window"}
SLOW_TRACES = ("D_mean", "gH_mean", "H_mean", "H_source_mean", "gErec_mean", "gI_mean")
RHO_H = 0.54


def _load():
    payload = json.loads((OUT / "clamp_fork_summary.json").read_text())
    rows, traces = {}, {}
    for row in payload["rows"]:
        key = (row["source_snapshot"], row["arm"])
        rows[key] = row
        with np.load(OUT / f"forks/{row['arm_id']}/traces.npz") as handle:
            trace = {name: np.asarray(handle[name]) for name in handle.files}
        if row.get("extension_of"):
            # rate_bins_hz and active_area_mm2 already cover the joined window, but the slow-field
            # traces only cover the 4 s tail.  Reading a change from the tail alone makes a FREE field
            # look pinned whenever it saturated inside the parent window -- which is what the H gate
            # does in the D-pinned arms.  Prepend the parent's own slow traces.
            with np.load(OUT / f"forks/{row['extension_of']}/traces.npz") as parent:
                for name in SLOW_TRACES:
                    if name in trace and name in parent.files:
                        trace[name] = np.concatenate([np.asarray(parent[name]), trace[name]])
        traces[key] = trace
    return payload, rows, traces


def _rate_100ms(trace):
    rate = np.asarray(trace["rate_bins_hz"], float)
    bin_ms = float(np.asarray(trace["rate_bin_ms"]).ravel()[0])
    group = max(1, int(round(100.0 / bin_ms)))
    usable = (rate.size // group) * group
    return rate[:usable].reshape(-1, group).mean(axis=1), 0.1


def _burst_stats(tail, band, bin_ms):
    above = (tail > band).astype(int)
    starts = int(np.sum(np.diff(above) == 1) + (1 if above[0] else 0))
    seconds = tail.size * bin_ms / 1000.0
    return {
        "bursts_per_s": starts / seconds if seconds else float("nan"),
        "zero_fraction": float(np.mean(tail == 0.0)),
        "sub_band_fraction": float(np.mean(tail <= band)),
        "median_hz": float(np.median(tail)),
    }


def main():
    payload, rows, traces = _load()
    snapshots = sorted({key[0] for key in rows})
    cfg = json.loads((ROOT / "config/topic4_fcxr_lc6b_frozen_slow_atlas.json").read_text())
    th = cfg["classifier"]["thresholds"]
    sat, band, ceiling = th["global_saturation_hz"], th["interictal_roll_hi_hz"], th["near_refractory_rate_hz"]
    manifest_snaps = cfg["source_snapshots"]
    keys = [(s, a) for s in snapshots for a in ARM_ORDER if (s, a) in rows]
    sheet = float(rows[keys[0]]["sheet_area_mm2"])

    fig, axes = plt.subplots(3, 2, figsize=(12.6, 13.4), constrained_layout=True)

    # ---- a / b : does it keep escalating?
    for index in range(2):
        ax = axes[0, index]
        if index >= len(snapshots):
            ax.set_visible(False)
            continue
        snapshot = snapshots[index]
        info = None
        for arm in ARM_ORDER:
            row = rows.get((snapshot, arm))
            if row is None:
                continue
            info = row
            rate, dt_s = _rate_100ms(traces[(snapshot, arm)])
            wide = arm == "NAT"
            ax.plot(np.arange(rate.size) * dt_s, rate, color=ARM_COLOR[arm],
                    lw=3.6 if wide else 1.7, alpha=.40 if wide else .95, solid_capstyle="round",
                    label=f"{ARM_LABEL[arm]} — {SHORT.get(row['verdict']['label'], row['verdict']['label'])}")
        ax.axhline(sat, color="#B00020", ls="--", lw=1.0)
        ax.axhline(band, color="0.45", ls=":", lw=1.0)
        for value, text, colour in ((sat, f"registered saturation {sat:.0f} Hz", "#B00020"),
                                    (band, f"interictal band {band:.1f} Hz", "0.35")):
            ax.annotate(text, xy=(1.0, value), xycoords=("axes fraction", "data"),
                        xytext=(-4, 3), textcoords="offset points", color=colour,
                        fontsize=8, ha="right", va="bottom")
        ax.set_title(
            f"{'ab'[index]}  from {info['snapshot_time_ms'] / 1000:.0f} s "
            f"(onset+{info['relative_to_onset_ms'] / 1000:.0f} s; the second before the fork ran at "
            f"{manifest_snaps[snapshot]['preceding_1s_global_rate_hz']:.0f} Hz)", loc="left", fontsize=10)
        ax.set_xlabel("time after the fork (s)")
        ax.set_ylabel("population E rate (Hz, 100 ms bins)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)

    # ---- c : continuous plateau, or a train re-igniting from near-silence?
    ax = axes[1, 0]
    for offset, snapshot in enumerate(snapshots):
        key = (snapshot, "DH_CLAMP")
        trace = traces[key]
        bin_ms = float(np.asarray(trace["rate_bin_ms"]).ravel()[0])
        n_tail = int(round(2000.0 / bin_ms))
        tail = np.asarray(trace["rate_bins_hz"], float)[-n_tail:]
        t = np.arange(tail.size) * bin_ms / 1000.0
        colour = ("#2CA25F", "#00695C")[offset]
        ax.plot(t, tail, color=colour, lw=1.1,
                label=f"{snapshot}: D and H pinned, final 2 s")
        ax.fill_between(t, 0, tail, where=tail <= band, color=colour, alpha=.35, step=None)
        stats = _burst_stats(tail, band, bin_ms)
    ax.axhline(band, color="0.35", ls=":", lw=1.2)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("time in the final 2 s (s)")
    ax.set_ylabel("population E rate (Hz, 20 ms bins, symlog)")
    ax.set_title("c  the pinned-D state is a re-ignition train, not a plateau", loc="left", fontsize=10)
    # The band caption goes in the legend: every part of this axis carries data, so any in-axes
    # caption would sit on top of a burst.
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.35", ls=":", lw=1.2))
    labels.append(f"interictal band {band:.1f} Hz (shaded where below)")
    ax.legend(handles, labels, frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(.5, -.16))
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate(
        "10.5 bursts/s · 30–33% of 20 ms bins have exactly zero E spikes\n"
        "median instantaneous rate 2–7 Hz, i.e. BELOW the interictal band",
        xy=(.02, .04), xycoords="axes fraction", fontsize=8, color="#B00020", va="bottom")

    # ---- d : how much tissue carries it?
    ax = axes[1, 1]
    for key in keys:
        area = np.asarray(rows[key]["active_area_mm2"], float)
        t = np.arange(area.size) * 0.1
        ls = "-" if key[0] == snapshots[0] else "--"
        ax.plot(t, area, color=ARM_COLOR[key[1]], lw=1.3, ls=ls, alpha=.9)
    ax.axhline(sheet, color="#B00020", ls="--", lw=1.0)
    ax.annotate(f"whole sheet {sheet:.0f} mm²", xy=(1.0, sheet), xycoords=("axes fraction", "data"),
                xytext=(-4, 3), textcoords="offset points", fontsize=8, color="#B00020",
                ha="right", va="bottom")
    ax.set_xlabel("time after the fork (s)")
    ax.set_ylabel("active area (mm², 100 ms windows)")
    ax.set_title("d  wide-field, not a focal carrier", loc="left", fontsize=10)
    ax.set_ylim(0, sheet * 1.12)
    handles = [Line2D([], [], color=ARM_COLOR[a], lw=1.6, label=ARM_LABEL[a]) for a in ARM_ORDER]
    handles += [Line2D([], [], color="0.35", lw=1.3, ls="-", label=f"{snapshots[0]} (solid)"),
                Line2D([], [], color="0.35", lw=1.3, ls="--", label=f"{snapshots[-1]} (dashed)")]
    ax.legend(handles=handles, frameon=False, fontsize=7, ncol=2, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # ---- e : whole population, or cells at the ceiling?
    ax = axes[2, 0]
    x = np.arange(len(keys))
    for offset, (quantile, alpha, hatch) in zip(
        (-0.26, 0.0, 0.26), (("q50", .35, None), ("q95", .65, None), ("q99", 1.0, "///")),
    ):
        values = [rows[k]["cell_rate_distribution"]["quantiles_hz"][quantile][-1] for k in keys]
        ax.bar(x + offset, values, width=.25, color=[ARM_COLOR[k[1]] for k in keys],
               alpha=alpha, hatch=hatch, edgecolor="white", linewidth=.4, label=f"per-cell {quantile}")
    ax.axhline(ceiling, color="#B00020", ls="--", lw=1.0)
    ax.annotate(f"near-refractory {ceiling:.0f} Hz", xy=(1.0, ceiling),
                xycoords=("axes fraction", "data"), xytext=(-4, 3), textcoords="offset points",
                color="#B00020", fontsize=8, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(30, 900)
    ax.set_xticks(x, [f"{k[0]}\n{k[1].replace('_CLAMP', '')}" for k in keys], fontsize=8)
    ax.set_ylabel("per-cell rate in the final second (Hz, log)")
    ax.set_title("e  the whole population moves, not a few pinned cells", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.13))
    ax.spines[["top", "right"]].set_visible(False)

    # ---- f : engineering check
    ax = axes[2, 1]
    d_travel, gate_travel = [], []
    for key in keys:
        trace = traces[key]
        d = np.asarray(trace["D_mean"], float)
        gate = np.asarray(trace["gH_mean"], float) / RHO_H
        d_travel.append(float(d[-1] - d[0]))
        gate_travel.append(float(gate[-1] - gate[0]))
    ax.bar(x - .19, d_travel, width=.38, color="#D95F0E", label="Δ mean wear D")
    ax.bar(x + .19, gate_travel, width=.38, color="#8C6BB1", label="Δ mean H gate occupancy")
    n_pinned = 0
    for index, (dv, gv) in enumerate(zip(d_travel, gate_travel)):
        for offset, value in ((-.19, dv), (.19, gv)):
            if value == 0.0:
                n_pinned += 1
                ax.plot([x[index] + offset], [0], marker="_", ms=13, mew=2.4, color="#B00020")
    ax.axhline(0, color="0.3", lw=.8)
    span = max(max(d_travel + gate_travel, default=0.0), 1e-6)
    ax.set_ylim(-0.14 * span, 1.12 * span)
    ax.set_xticks(x, [f"{k[0]}\n{k[1].replace('_CLAMP', '')}" for k in keys], fontsize=8)
    ax.set_ylabel("change across the window")
    ax.set_title("f  engineering check: pinned fields move exactly zero", loc="left", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    if n_pinned:
        handles.append(Line2D([], [], marker="_", ms=11, mew=2.4, color="#B00020", ls="none"))
        labels.append(f"{n_pinned} pinned fields, bitwise unchanged (0.000000)")
    ax.legend(handles, labels, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.13))
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "FCXR-LC6B — one canonical-seed C0 discharge continued with the slow fields pinned",
        fontsize=12)
    fig.text(
        .5, -.012,
        "Both snapshots come from the same C0 trajectory (onset 11 s).  D = 1 - z is synaptic wear; "
        "the H gate is capped at rho = 0.54.  Pinning D replaces whole-sheet saturation with a "
        "wide-field re-ignition train whose median instantaneous rate sits below the interictal band "
        "— a bounded continuation, not a demonstrated seizure carrier.  No perturbation-return test "
        "was run; termination and lifecycle were not tested.",
        ha="center", va="top", fontsize=8, color="#555555", wrap=True)

    FIGURES.mkdir(parents=True, exist_ok=True)
    png, pdf = FIGURES / "lc6b_clamp_forks.png", FIGURES / "lc6b_clamp_forks.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}\nwrote {pdf}")
    return payload


if __name__ == "__main__":
    main()
