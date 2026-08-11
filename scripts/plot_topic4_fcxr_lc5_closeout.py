#!/usr/bin/env python3
"""FCXR-LC5 closeout evidence figure.

Three panels, three independent questions:

A  did the network enter on its own, and did it ever come back down?
B  is the high state a bounded carrier, or a population pinned at its own firing ceiling?
C  does the pre-registered load scaling have a solution on this source?

This is a diagnostic / stop-evidence figure, not a paper mechanism figure, so it deliberately does
not use the single-row mechanism layout reserved for those.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-fcxr-lc5")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_fcxr_lc5 import load_sparse_spike_stream  # noqa: E402

OUT = ROOT / "results/topic4_sef_hfo/fcxr_lc5_episode_pump"
BUNDLE = OUT / "u1_capture"
FIGDIR = OUT / "figures"
DT_MS = 0.05

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

WINDOW_LABEL = {
    "baseline": "7-11",
    "entry": "11-12",
    "high_reference": "12-15",
    "late_1": "15-18",
    "late_2": "18-21",
    "late_3": "21-22",
}


def _load():
    summary = json.loads((BUNDLE / "u1_capture_summary.json").read_text())
    adj = json.loads((OUT / "u1_carrier_adjudication.json").read_text())
    scale = json.loads((OUT / "u1b_scale_lock/u1b_scale_verdict.json").read_text())
    ledger = json.loads((BUNDLE / "u1_event_ledger.json").read_text())
    traces = np.load(BUNDLE / "u1_capture_traces.npz")
    stream = load_sparse_spike_stream(BUNDLE / "u1_sparse_spikes.npz")
    return summary, adj, scale, ledger, traces, stream


def _panel_a(ax_act, ax_slow, summary, adj, ledger, traces):
    onset_s = float(summary["onset_ms"]) / 1000.0
    total_s = float(summary["T_ms"]) / 1000.0
    af = np.asarray(traces["af"], float)
    af_t = np.arange(af.size) * float(traces["af_dt_ms"][0]) / 1000.0

    for name, colour in (("baseline", "0.88"), ("high_reference", "#ffd8a8")):
        win = next(w for w in adj["windows"] if w["window"] == name)
        for ax in (ax_act, ax_slow):
            ax.axvspan(win["window_ms"][0] / 1000.0, win["window_ms"][1] / 1000.0,
                       color=colour, zorder=0, lw=0)

    ax_act.plot(af_t, 100.0 * af, color="0.25", lw=0.6, zorder=3)
    ax_act.set_yscale("log")
    ax_act.set_ylabel("cells firing (%)")
    ax_act.set_ylim(2e-3, 900.0)
    ax_act.set_xlim(0.0, total_s)

    pre = [e for e in ledger["events"] if e["returned"] and e["t_off_ms"] < summary["onset_ms"]]
    ax_act.plot([e["t_on_ms"] / 1000.0 for e in pre], np.full(len(pre), 330.0),
                marker="v", ls="none", ms=2.8, color="#1f77b4", zorder=4)
    ax_act.text(0.15, 560.0, f"{len(pre)} brief events that each returned to rest",
                color="#1f77b4", fontsize=7, va="center")
    ax_act.axvline(onset_s, color="#d62728", lw=1.0, zorder=2)
    ax_act.text(9.0, 150.0, "rest reference", color="0.45", fontsize=7, ha="center")
    ax_act.text(13.5, 560.0, "load scale was fixed\nfrom this window", color="#a86b1d",
                fontsize=7, ha="center", va="center")
    ax_act.text(16.9, 2.2e-2,
                f"entry at {onset_s:.0f} s, then never\nback to rest for the rest\nof the recording",
                color="#d62728", fontsize=7, va="center", ha="left")

    slow_t = np.arange(traces["D_mean"].size) * float(traces["slow_trace_dt_ms"][0]) / 1000.0
    ax_slow.plot(slow_t, np.asarray(traces["D_mean"], float), color="#2ca02c", lw=1.1, zorder=3)
    ax_slow.set_ylabel("loss of inhibition", color="#2ca02c")
    ax_slow.tick_params(axis="y", labelcolor="#2ca02c")
    ax_slow.set_xlabel("time (s)")
    ax_slow.set_xlim(0.0, total_s)
    twin = ax_slow.twinx()
    twin.plot(slow_t, np.asarray(traces["H_mean"], float), color="#9467bd", lw=1.1, zorder=3)
    twin.set_ylabel("self-drive", color="#9467bd")
    twin.tick_params(axis="y", labelcolor="#9467bd")
    twin.spines["top"].set_visible(False)
    ax_act.set_title("A   entered on its own, then never came back down", loc="left",
                     fontweight="bold")


def _panel_b(ax, adj, stream):
    windows = adj["windows"]
    rates = []
    for w in windows:
        lo, hi = w["window_ms"]
        rates.append(stream.per_cell_rate_hz(
            lo_step=int(round(lo / DT_MS)), hi_step=int(round(hi / DT_MS)), dt_ms=DT_MS
        ))
    colours = plt.cm.viridis(np.linspace(0.08, 0.92, len(windows)))
    parts = ax.violinplot([np.clip(r, 0.5, None) for r in rates], positions=np.arange(len(windows)),
                          widths=0.82, showextrema=True, showmedians=True)
    for body, colour in zip(parts["bodies"], colours):
        body.set_facecolor(colour)
        body.set_alpha(0.85)
        body.set_edgecolor("none")
    for key, lw in (("cmedians", 1.1), ("cmins", 0.7), ("cmaxes", 0.7), ("cbars", 0.7)):
        parts[key].set_color("0.15")
        parts[key].set_linewidth(lw)

    hard = adj["registered_criteria"]["hard_single_cell_ceiling_hz"]
    reg = adj["registered_criteria"]["registered_sat_ceiling_hz"]
    ax.axhline(hard, color="#d62728", lw=1.0, ls="-", zorder=1)
    ax.axhline(reg, color="#d62728", lw=1.0, ls="--", zorder=1)
    ax.text(-0.44, hard * 0.86, "fastest a cell can possibly fire", color="#d62728",
            ha="left", va="top", fontsize=6.5)
    ax.text(-0.44, reg * 0.86, "saturation line this model already uses", color="#d62728",
            ha="left", va="top", fontsize=6.5)
    ax.text(4.5, 6.0, "share of cells past the\nsaturation line", color="#d62728", ha="center",
            fontsize=6.5)
    for i, w in enumerate(windows):
        if w["above_sat_ceiling_fraction"] > 0.0:
            ax.text(i, 1.15, f"{100 * w['above_sat_ceiling_fraction']:.0f}%", ha="center",
                    color="#d62728", fontsize=6.5)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(windows)))
    ax.set_xticklabels([WINDOW_LABEL[w["window"]] for w in windows])
    ax.set_xlabel("time window (s)")
    ax.set_ylabel("firing rate of each cell (Hz)")
    ax.set_ylim(0.5, hard * 3.4)
    ax.set_title("B   climbs into the cells' own ceiling", loc="left", fontweight="bold")


def _panel_c(ax, scale):
    sweep = scale["window_sweep"]
    x = np.arange(len(sweep))
    sup = np.array([w["admissible_target_activation_sup"] for w in sweep])
    locked = float(scale["target_activation"])
    ax.fill_between(x, sup, 3.0, color="#d62728", alpha=0.12, lw=0, zorder=0)
    ax.plot(x, sup, marker="o", ms=4.0, color="0.2", lw=1.2, zorder=3)
    ax.axhline(locked, color="#1f77b4", lw=1.2, zorder=2)
    ax.text(-0.42, locked * 1.13, "level fixed before the run", color="#1f77b4", ha="left",
            fontsize=6.5)
    ax.text(3.9, 1.55, "above this curve\nthe load never settles", color="#b03030", fontsize=6.5,
            ha="center")
    for i in (2, len(sweep) - 1):
        ax.annotate(f"{sup[i]:.2f}", (x[i], sup[i]), textcoords="offset points", xytext=(0, -11),
                    ha="center", fontsize=6.5, color="0.2")
    ax.set_yscale("log")
    ax.set_ylim(0.07, 3.0)
    ax.set_yticks([0.1, 0.2, 0.5, 1.0, 2.0])
    ax.set_yticklabels(["0.1", "0.2", "0.5", "1.0", "2.0"])
    ax.set_xticks(x)
    ax.set_xticklabels([WINDOW_LABEL[w["window"]] for w in sweep])
    ax.set_xlabel("time window (s)")
    ax.set_ylabel("highest load level that settles")
    ax.set_title("C   planned load level has no solution", loc="left", fontweight="bold")


def main():
    summary, adj, scale, ledger, traces, stream = _load()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.4, 7.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 0.8, 2.1], hspace=0.62, wspace=0.34)
    ax_act = fig.add_subplot(gs[0, :])
    ax_slow = fig.add_subplot(gs[1, :], sharex=ax_act)
    _panel_a(ax_act, ax_slow, summary, adj, ledger, traces)
    _panel_b(fig.add_subplot(gs[2, 0]), adj, stream)
    _panel_c(fig.add_subplot(gs[2, 1]), scale)
    plt.setp(ax_act.get_xticklabels(), visible=False)

    stem = FIGDIR / "lc5_source_capture_and_scale_stop"
    fig.savefig(f"{stem}.png")
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    meta = {
        "figure": stem.name,
        "panels": {
            "A": "did the network enter on its own, and did it ever come back down",
            "B": "is the high state a bounded carrier or a population at its firing ceiling",
            "C": "does the pre-registered load scaling have a solution on this source",
        },
        "sources": {
            "capture_summary": str(BUNDLE / "u1_capture_summary.json"),
            "carrier_adjudication": str(OUT / "u1_carrier_adjudication.json"),
            "scale_verdict": str(OUT / "u1b_scale_lock/u1b_scale_verdict.json"),
        },
        "capture_spike_stream_sha256": summary["spike_stream_sha256"],
        "capture_config_sha256": summary["config_sha256"],
        "connection_seed": summary["connection_seed"],
        "noise_seed": summary["noise_seed"],
        "source_type": adj["source_type"],
        "scale_verdict": scale["status"],
    }
    (FIGDIR / f"{stem.name}_metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"png": f"{stem}.png", "pdf": f"{stem}.pdf"}, indent=2))


if __name__ == "__main__":
    main()
