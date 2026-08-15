#!/usr/bin/env python3
"""One figure for the eight FCXR-LC6B clamp forks.

Three independent questions, one axis family each:
  a/b  does the discharge keep escalating after the slow fields are pinned (one panel per snapshot)?
  c    is the sheet uniformly elevated, or is a patch pinned at the refractory ceiling?
  d    did the clamp actually hold, and how far did the free slow variable travel meanwhile?

The dynamics label is annotated on a/b rather than given a panel of its own: it is derived from the
same rate and per-cell readouts those panels already show, so a separate panel would be the same
information drawn twice.
"""
from __future__ import annotations

import json
import os
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
SHORT = {"ESCALATING_SATURATION": "escalates to saturation", "BOUNDED_STATIONARY": "bounded, steady",
         "BOUNDED_OSCILLATORY": "bounded, bursting", "LOW_STATE": "falls back to interictal",
         "SILENCE": "silent", "AFTER_DISCHARGE": "brief after-discharge",
         "NUMERICAL_FAIL": "numerical failure", "RIGHT_CENSORED": "unresolved in window"}


SLOW_TRACES = ("D_mean", "gH_mean", "H_mean", "H_source_mean", "gErec_mean", "gI_mean")


def _load():
    payload = json.loads((OUT / "clamp_fork_summary.json").read_text())
    rows, traces = {}, {}
    for row in payload["rows"]:
        key = (row["source_snapshot"], row["arm"])
        rows[key] = row
        with np.load(OUT / f"forks/{row['arm_id']}/traces.npz") as handle:
            trace = {name: np.asarray(handle[name]) for name in handle.files}
        if row.get("extension_of"):
            # The extension bundle's rate_bins_hz already covers the joined window, but the slow-field
            # traces only cover the 4 s tail.  Reading Δ from the tail alone makes a FREE field look
            # pinned whenever it happened to saturate inside the parent window -- which is exactly what
            # the H gate does in the D-pinned arms.  Prepend the parent's own slow traces.
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


def main():
    payload, rows, traces = _load()
    snapshots = sorted({key[0] for key in rows})
    thresholds = json.loads((ROOT / "config/topic4_fcxr_lc6b_frozen_slow_atlas.json").read_text())
    sat = thresholds["classifier"]["thresholds"]["global_saturation_hz"]
    band = thresholds["classifier"]["thresholds"]["interictal_roll_hi_hz"]
    ceiling = thresholds["classifier"]["thresholds"]["near_refractory_rate_hz"]
    rho = thresholds["prior_expectation"]["h_actuator_ceiling"]
    rho_value = 0.54 if "0.54" in rho else 0.54

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), constrained_layout=True)

    manifest_snaps = thresholds["source_snapshots"]

    # a / b -- one panel per source snapshot
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
            t = np.arange(rate.size) * dt_s
            # NAT is drawn as a wide translucent band so an arm that lands on top of it (H pinned at
            # onset+4 s is nearly identical to it) stays visible instead of erasing it.
            wide = arm == "NAT"
            ax.plot(t, rate, color=ARM_COLOR[arm], lw=3.6 if wide else 1.7,
                    alpha=.40 if wide else .95, solid_capstyle="round",
                    label=f"{ARM_LABEL[arm]} — {SHORT.get(row['verdict']['label'], row['verdict']['label'])}")
        ax.axhline(sat, color="#B00020", ls="--", lw=1.0)
        ax.axhline(band, color="0.45", ls=":", lw=1.0)
        # Threshold captions sit outside the data area on the right, so no curve is ever covered.
        for value, text, colour in ((sat, f"registered saturation {sat:.0f} Hz", "#B00020"),
                                    (band, f"interictal band {band:.1f} Hz", "0.35")):
            ax.annotate(text, xy=(1.0, value), xycoords=("axes fraction", "data"),
                        xytext=(-4, 3), textcoords="offset points",
                        color=colour, fontsize=8, ha="right", va="bottom")
        entering = manifest_snaps[snapshot]["preceding_1s_global_rate_hz"]
        ax.set_title(
            f"{'ab'[index]}  from {info['snapshot_time_ms'] / 1000:.0f} s "
            f"(onset+{info['relative_to_onset_ms'] / 1000:.0f} s; the second before the fork "
            f"ran at {entering:.0f} Hz)", loc="left", fontsize=10)
        ax.set_xlabel("time after the fork (s)")
        ax.set_ylabel("population E rate (Hz, 100 ms bins)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)

    # c -- per-cell rate distribution in the final second
    ax = axes[1, 0]
    keys = [(snapshot, arm) for snapshot in snapshots for arm in ARM_ORDER if (snapshot, arm) in rows]
    x = np.arange(len(keys))
    for offset, (quantile, alpha, hatch) in zip(
        (-0.26, 0.0, 0.26), (("q50", .35, None), ("q95", .65, None), ("q99", 1.0, "///")),
    ):
        values = []
        for key in keys:
            dist = rows[key]["cell_rate_distribution"]
            values.append(np.nan if dist is None else dist["quantiles_hz"][quantile][-1])
        ax.bar(x + offset, values, width=.25, color=[ARM_COLOR[k[1]] for k in keys],
               alpha=alpha, hatch=hatch, edgecolor="white", linewidth=.4,
               label=f"per-cell {quantile}")
    ax.axhline(ceiling, color="#B00020", ls="--", lw=1.0)
    ax.annotate(f"near-refractory {ceiling:.0f} Hz", xy=(1.0, ceiling),
                xycoords=("axes fraction", "data"), xytext=(-4, 3),
                textcoords="offset points", color="#B00020", fontsize=8,
                ha="right", va="bottom")
    # Log scale: the escalating arms sit at the refractory ceiling and the pinned ones an order of
    # magnitude below, so a linear axis flattens the pinned arms into an unreadable strip.
    ax.set_yscale("log")
    ax.set_ylim(30, 900)
    ax.set_xticks(x, [f"{k[0]}\n{k[1].replace('_CLAMP', '')}" for k in keys], fontsize=8)
    ax.set_ylabel("per-cell rate in the final second (Hz, log)")
    ax.set_title("c  a whole moderate sheet or cells at the refractory ceiling?",
                 loc="left", fontsize=10)
    # Outside the data area: log bars run down to the axis floor, so any in-axes legend covers them.
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.13))
    ax.spines[["top", "right"]].set_visible(False)

    # d -- manipulation check: how far did each slow field travel?
    ax = axes[1, 1]
    d_travel, gate_travel = [], []
    for key in keys:
        trace = traces[key]
        d = np.asarray(trace["D_mean"], float)
        gate = np.asarray(trace["gH_mean"], float) / rho_value
        d_travel.append(float(d[-1] - d[0]))
        gate_travel.append(float(gate[-1] - gate[0]))
    ax.bar(x - .19, d_travel, width=.38, color="#D95F0E", label="Δ mean wear D")
    ax.bar(x + .19, gate_travel, width=.38, color="#8C6BB1", label="Δ mean H gate occupancy")
    span = max(max(d_travel + gate_travel, default=0.0), 1e-6)
    n_pinned = 0
    for index, (dv, gv) in enumerate(zip(d_travel, gate_travel)):
        for offset, value in ((-.19, dv), (.19, gv)):
            if value == 0.0:
                n_pinned += 1
                ax.plot([x[index] + offset], [0], marker="_", ms=13, mew=2.4, color="#B00020")
    ax.axhline(0, color="0.3", lw=.8)
    ax.set_ylim(-0.14 * span, 1.12 * span)
    ax.set_xticks(x, [f"{k[0]}\n{k[1].replace('_CLAMP', '')}" for k in keys], fontsize=8)
    ax.set_ylabel("change across the window")
    ax.set_title("d  manipulation check: pinned fields move exactly zero", loc="left", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    if n_pinned:
        # The zero-height bars need a key, and a free-floating caption would land on the tall bars.
        handles.append(Line2D([], [], marker="_", ms=11, mew=2.4, color="#B00020", ls="none"))
        labels.append(f"{n_pinned} pinned fields, bitwise unchanged (0.000000)")
    ax.legend(handles, labels, frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(.5, -.13))
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "FCXR-LC6B — one canonical-seed C0 discharge continued with the slow fields pinned",
        fontsize=12)
    decision = payload["decision"]
    fig.text(
        .5, -.02,
        f"Both snapshots come from the same C0 trajectory (onset 11 s). "
        f"D = 1 - z is synaptic wear; the H gate is bounded above by rho = {rho_value}. "
        f"A bounded label means the branch persisted across this window; no perturbation-return "
        f"test was run. First-round branch: {decision['branch']}.",
        ha="center", va="top", fontsize=8, color="#555555")

    FIGURES.mkdir(parents=True, exist_ok=True)
    png, pdf = FIGURES / "lc6b_clamp_forks.png", FIGURES / "lc6b_clamp_forks.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}\nwrote {pdf}")
    return payload


if __name__ == "__main__":
    main()
