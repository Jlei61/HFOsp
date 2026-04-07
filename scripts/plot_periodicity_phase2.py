#!/usr/bin/env python3
"""
Visualize Phase 2 periodicity experiment results.

Produces:
  - Exp 1: W vs f_peak scatter with y=1/x reference line
  - Exp 2: Side-by-side PSD comparison (window_start vs centroid)
  - Exp 3: Hazard function H(t) for packed vs centroid events
  - Exp 4: IEI return map (Poincaré plot)
  - Exp 5: Propagation stereotypy bar chart (SOZ vs non-SOZ)

Usage:
    python scripts/plot_periodicity_phase2.py
    python scripts/plot_periodicity_phase2.py --exp 1,2

These figures are descriptive summaries of the Phase 2 outputs. In particular:
  - Exp 3 hazard curves are qualitative visualizations of dead-time structure.
  - Exp 4 panel titles report the correlation on log-IEI pairs; the naive
    Pearson p-value is intentionally omitted from the figure because it is not
    the final inferential statistic.
  - Exp 5 should be read as exploratory until multi-seed / mixed-model
    follow-up is done.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/event_periodicity/phase2")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"  {name} not found, skipping")
        return {}
    with open(path) as f:
        return json.load(f)


# ==========================================================================
# Exp 1: PackWinLen Sweep
# ==========================================================================

def plot_exp1():
    """W vs f_peak scatter + y=1/x reference."""
    data = _load("exp1_packing_sweep.json")
    if not data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    w_ref = np.linspace(0.08, 1.1, 100)
    ax.plot(w_ref * 1000, 1.0 / w_ref, "k--", lw=1, alpha=0.5, label="f = 1/W")

    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    for i, (sub, sweeps) in enumerate(data.items()):
        if isinstance(sweeps, dict) and "error" in sweeps:
            continue
        ws = [r["window_sec"] * 1000 for r in sweeps if r.get("peak_freq")]
        fs = [r["peak_freq"] for r in sweeps if r.get("peak_freq")]
        if ws:
            ax.scatter(ws, fs, color=colors[i], s=30, label=sub, zorder=3)
            ax.plot(ws, fs, color=colors[i], lw=0.8, alpha=0.5)

    ax.set_xlabel("Pack Window W (ms)", fontsize=11)
    ax.set_ylabel("Peak Frequency (Hz)", fontsize=11)
    ax.set_title("Exp 1: f_peak vs Pack Window Size", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(50, 1100)
    ax.set_ylim(0, 12)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for i, (sub, sweeps) in enumerate(data.items()):
        if isinstance(sweeps, dict) and "error" in sweeps:
            continue
        ws = [r["window_sec"] * 1000 for r in sweeps]
        ns = [r["n_events"] for r in sweeps]
        ax2.plot(ws, ns, "o-", color=colors[i], lw=1, ms=4, label=sub)

    ax2.set_xlabel("Pack Window W (ms)", fontsize=11)
    ax2.set_ylabel("Number of Events", fontsize=11)
    ax2.set_title("Event Count vs Window Size", fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp1_packing_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved exp1_packing_sweep.png")


# ==========================================================================
# Exp 2: Centroid Bypass
# ==========================================================================

def plot_exp2():
    """Compare peak frequencies: window_start vs centroid methods."""
    data = _load("exp2_centroid_bypass.json")
    if not data:
        return

    subjects = []
    ws_peaks = []
    mc_peaks = []
    ic_peaks = []

    for key, methods in data.items():
        if isinstance(methods, dict) and "error" in methods:
            continue
        sub = key.split("/")[-1]
        subjects.append(sub)
        ws_peaks.append(methods.get("window_start", {}).get("peak_freq"))
        mc_peaks.append(methods.get("mean_centroid", {}).get("peak_freq"))
        ic_peaks.append(methods.get("ignition_centroid", {}).get("peak_freq"))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = np.arange(len(subjects))
    width = 0.25

    def _vals(arr):
        return [v if v is not None else 0 for v in arr]

    bars1 = ax.bar(x - width, _vals(ws_peaks), width, label="Window Start", alpha=0.8)
    bars2 = ax.bar(x, _vals(mc_peaks), width, label="Mean Centroid", alpha=0.8)
    bars3 = ax.bar(x + width, _vals(ic_peaks), width, label="Ignition Centroid", alpha=0.8)

    for bars, vals in [(bars1, ws_peaks), (bars2, mc_peaks), (bars3, ic_peaks)]:
        for bar, v in zip(bars, vals):
            if v is None:
                bar.set_hatch("//")
                bar.set_alpha(0.2)

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Peak Frequency (Hz)", fontsize=11)
    ax.set_title("Exp 2: Centroid Bypass — Peak Freq by Event Definition", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp2_centroid_bypass.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved exp2_centroid_bypass.png")


# ==========================================================================
# Exp 3: Hazard Function
# ==========================================================================

def plot_exp3():
    """H(t) curves for packed vs centroid events, per subject."""
    data = _load("exp3_hazard.json")
    if not data:
        return

    valid = {k: v for k, v in data.items() if isinstance(v, dict) and "packed" in v}
    if not valid:
        print("  No valid hazard data")
        return

    n = len(valid)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for idx, (key, sub_data) in enumerate(sorted(valid.items())):
        ax = axes[idx // ncols][idx % ncols]
        sub = key.split("/")[-1]

        for method, style in [("packed", ("k-", 1.5)),
                              ("mean_centroid", ("b--", 1.0)),
                              ("ignition_centroid", ("r:", 1.0))]:
            if method not in sub_data:
                continue
            md = sub_data[method]
            t = np.array(md["t"])
            h = np.array(md["hazard"])
            ax.plot(t, h, style[0], lw=style[1], label=method, alpha=0.8)

        ax.set_title(f"{sub}", fontsize=9)
        ax.set_xlabel("t (s)", fontsize=8)
        ax.set_ylabel("H(t)", fontsize=8)
        ax.legend(fontsize=6)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.2)

    for idx in range(len(valid), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Exp 3: IEI Hazard Function", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp3_hazard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved exp3_hazard.png")


# ==========================================================================
# Exp 4: Return Map
# ==========================================================================

def plot_exp4():
    """IEI[n] vs IEI[n+1] Poincaré plots."""
    data = _load("exp4_return_map.json")
    if not data:
        return

    valid = {k: v for k, v in data.items()
             if isinstance(v, dict) and "packed" in v and "iei_n" in v.get("packed", {})}
    if not valid:
        print("  No valid return map data")
        return

    n = len(valid)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)

    for idx, (key, sub_data) in enumerate(sorted(valid.items())):
        ax = axes[idx // ncols][idx % ncols]
        sub = key.split("/")[-1]

        packed = sub_data["packed"]
        iei_n = np.array(packed["iei_n"])
        iei_n1 = np.array(packed["iei_n1"])

        ax.scatter(iei_n, iei_n1, s=2, alpha=0.3, c="steelblue", rasterized=True)
        lims = [np.percentile(iei_n, 1), np.percentile(iei_n, 99)]
        ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("IEI[n] (s)", fontsize=8)
        ax.set_ylabel("IEI[n+1] (s)", fontsize=8)
        r = packed.get("serial_corr", np.nan)
        ax.set_title(f"{sub}  r(log-IEI)={r:.3f}", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)

    for idx in range(len(valid), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Exp 4: IEI Return Map (Poincaré Plot)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp4_return_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved exp4_return_map.png")


# ==========================================================================
# Exp 5: Propagation Stereotypy
# ==========================================================================

def plot_exp5():
    """Bar chart of mean Kendall tau, SOZ vs non-SOZ."""
    data = _load("exp5_stereotypy.json")
    if not data:
        return

    valid = {k: v for k, v in data.items()
             if isinstance(v, dict) and "mean_tau" in v and not np.isnan(v["mean_tau"])}
    if not valid:
        print("  No valid stereotypy data")
        return

    subjects = sorted(valid.keys())
    x = np.arange(len(subjects))

    all_tau = [valid[s]["mean_tau"] for s in subjects]
    soz_tau = [valid[s].get("soz_mean_tau", np.nan) for s in subjects]
    nonsoz_tau = [valid[s].get("nonsoz_mean_tau", np.nan) for s in subjects]

    has_soz = any(np.isfinite(t) for t in soz_tau)

    if has_soz:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

    ax1.bar(x, all_tau, color="steelblue", alpha=0.8)
    ax1.set_ylabel("Mean Kendall τ", fontsize=11)
    ax1.set_title("Exp 5: Propagation Stereotypy (All Events)", fontsize=12)
    ax1.axhline(0, color="black", lw=0.5)
    ax1.grid(True, axis="y", alpha=0.3)

    if has_soz:
        width = 0.35
        soz_vals = [t if np.isfinite(t) else 0 for t in soz_tau]
        nonsoz_vals = [t if np.isfinite(t) else 0 for t in nonsoz_tau]
        ax2.bar(x - width / 2, soz_vals, width, label="SOZ events", color="tomato", alpha=0.8)
        ax2.bar(x + width / 2, nonsoz_vals, width, label="non-SOZ events", color="forestgreen", alpha=0.8)
        ax2.set_ylabel("Mean Kendall τ", fontsize=11)
        ax2.set_title("SOZ vs non-SOZ Stratification", fontsize=12)
        ax2.legend(fontsize=9)
        ax2.axhline(0, color="black", lw=0.5)
        ax2.grid(True, axis="y", alpha=0.3)

    bottom_ax = ax2 if has_soz else ax1
    bottom_ax.set_xticks(x)
    bottom_ax.set_xticklabels([s.split("/")[-1] for s in subjects],
                               rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp5_stereotypy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved exp5_stereotypy.png")


# ==========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all")
    args = parser.parse_args()

    if args.exp == "all":
        exps = {1, 2, 3, 4, 5}
    else:
        exps = {int(x) for x in args.exp.split(",")}

    print(f"Phase 2 visualization — experiments: {sorted(exps)}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Figures dir: {FIG_DIR}")

    if 1 in exps:
        plot_exp1()
    if 2 in exps:
        plot_exp2()
    if 3 in exps:
        plot_exp3()
    if 4 in exps:
        plot_exp4()
    if 5 in exps:
        plot_exp5()

    print(f"\nDone. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
