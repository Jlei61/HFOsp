#!/usr/bin/env python3
"""
Visualize Phase 2 periodicity experiment results.

Produces:
  - Exp 1: W vs f_peak scatter with y=1/x reference line
  - Exp 2: Side-by-side PSD comparison (window_start vs centroid)
  - Exp 3: Hazard function H(t) for packed vs centroid events
  - Exp 4: IEI return map (Poincaré plot)
  - Exp 5: Propagation stereotypy bar chart (SOZ vs non-SOZ)
  - Exp 6A: real PSD vs analytic renewal overlay
  - Exp 6B: SOZ vs non-SOZ dead-time paired scatter + Wilcoxon
  - Exp 7: serial-correlation deep-dive summary panels

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
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, wilcoxon

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
# Exp 6: Renewal analytic PSD + SOZ dead-time
# ==========================================================================

_EMP_COLOR = "#2166AC"
_ANA_COLOR = "#B2182B"
_AP_COLOR = "#888888"
_SOZ_COLOR = "#D6604D"
_NSOZ_COLOR = "#4393C3"
_MIN_EVENTS_DT = 50


def _minmax_band(arr, freqs, f_lo=0.5, f_hi=8.0):
    """Min-max normalize within [f_lo, f_hi]. Returns None if degenerate."""
    band = (freqs >= f_lo) & (freqs <= f_hi)
    a = arr[band]
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if hi - lo < 1e-30:
        return None
    return (arr - lo) / (hi - lo)


def _collect_exp6_records(psd_data):
    """Parse exp6 JSON into per-subject records with peak metrics."""
    records = []
    for key, rec in psd_data.items():
        if not isinstance(rec, dict) or "error" in rec or "warning" in rec:
            continue
        freqs = np.asarray(rec.get("delta_freqs", []), dtype=float)
        delta_psd = np.asarray(rec.get("delta_psd", []), dtype=float)
        analytic = np.asarray(rec.get("analytic_delta", []), dtype=float)
        if freqs.size == 0 or delta_psd.size == 0 or analytic.size == 0:
            continue

        smoothed = _smooth_psd(delta_psd)
        band = (freqs >= 0.5) & (freqs <= 5.0)
        fb = freqs[band]
        if smoothed[band].max() <= 0 or analytic[band].max() <= 0:
            continue

        emp_peak = float(fb[np.argmax(smoothed[band])])
        ana_peak = float(fb[np.argmax(analytic[band])])

        records.append({
            "key": key,
            "sub": key.split("/")[-1],
            "dataset": key.split("/")[0],
            "has_peak": rec.get("has_peak_0p5_10hz", False),
            "emp_peak": emp_peak,
            "ana_peak": ana_peak,
            "delta_f": abs(emp_peak - ana_peak),
            "freqs": freqs,
            "delta_psd": delta_psd,
            "analytic": analytic,
            "iei_min": rec["iei_min"],
            "n_events": rec["iei_n"],
        })
    return records


def _smooth_psd(psd, window=101, polyorder=3):
    """Savitzky-Golay smooth; window must be odd and < len."""
    n = len(psd)
    w = min(window, n - 1)
    if w % 2 == 0:
        w -= 1
    if w < polyorder + 2:
        return psd
    return savgol_filter(psd, w, polyorder)


def _draw_overlay_panel(ax, rec, tag=""):
    """Draw a single PSD overlay panel: smoothed empirical vs analytic renewal."""
    freqs = rec["freqs"]
    band = (freqs >= 0.5) & (freqs <= 8.0)

    smoothed_emp = _smooth_psd(rec["delta_psd"])
    emp_n = _minmax_band(smoothed_emp, freqs)
    ana_n = _minmax_band(rec["analytic"], freqs)
    if emp_n is None or ana_n is None:
        ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes, ha="center")
        return

    raw_n = _minmax_band(rec["delta_psd"], freqs)
    if raw_n is not None:
        ax.plot(freqs[band], raw_n[band], color=_EMP_COLOR, lw=0.3, alpha=0.25)

    ax.plot(freqs[band], emp_n[band], color=_EMP_COLOR, lw=2.0,
            label="empirical PSD (smoothed)")
    ax.plot(freqs[band], ana_n[band], color=_ANA_COLOR, lw=1.6, ls="--",
            label="analytic renewal")

    ax.axvline(rec["emp_peak"], color=_EMP_COLOR, lw=0.9, ls=":", alpha=0.7)
    ax.axvline(rec["ana_peak"], color=_ANA_COLOR, lw=0.9, ls=":", alpha=0.7)

    ax.annotate(f'{rec["emp_peak"]:.1f}', xy=(rec["emp_peak"], 1.02),
                fontsize=7, color=_EMP_COLOR, ha="center",
                xycoords=("data", "axes fraction"))
    offset = 0.08 if abs(rec["emp_peak"] - rec["ana_peak"]) < 0.4 else 0.0
    ax.annotate(f'{rec["ana_peak"]:.1f}', xy=(rec["ana_peak"], 0.94 - offset),
                fontsize=7, color=_ANA_COLOR, ha="center",
                xycoords=("data", "axes fraction"))

    ax.set_xlim(0.5, 8)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_ylabel("Normalized PSD", fontsize=8)
    ax.set_title(
        f"{tag}{rec['sub']}   |Δf|={rec['delta_f']:.2f} Hz\n"
        f"τᵣ={rec['iei_min']:.3f}s   n={rec['n_events']}",
        fontsize=8.5, pad=4)
    ax.grid(True, alpha=0.15)


def plot_exp6():
    """Fig 6A: delta-train PSD vs analytic renewal overlay + cohort scatter.
    Fig 6B: SOZ vs non-SOZ dead-time (filtered, log scale)."""
    psd_data = _load("exp6_renewal_psd.json")
    dt_data = _load("exp6_soz_deadtime.json")
    if not psd_data:
        return

    records = _collect_exp6_records(psd_data)
    if not records:
        print("  No valid exp6 records")
        return

    with_peak = sorted([r for r in records if r["has_peak"]], key=lambda x: x["delta_f"])
    n_peak = len(with_peak)

    best3 = with_peak[:3]
    worst2 = with_peak[-2:] if n_peak >= 5 else with_peak[max(0, n_peak - 2):]

    # ---- Figure 6A: 2×3 grid ----
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.30,
                          left=0.06, right=0.97, top=0.88, bottom=0.08)

    for i, rec in enumerate(best3):
        ax = fig.add_subplot(gs[0, i])
        _draw_overlay_panel(ax, rec, tag="")
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    for i, rec in enumerate(worst2):
        ax = fig.add_subplot(gs[1, i])
        _draw_overlay_panel(ax, rec, tag="")

    # Cohort scatter: analytic vs empirical peak freq
    ax_sc = fig.add_subplot(gs[1, 2])
    if with_peak:
        emp_fs = np.array([r["emp_peak"] for r in with_peak])
        ana_fs = np.array([r["ana_peak"] for r in with_peak])
        dsets = [r["dataset"] for r in with_peak]
        colors = [_EMP_COLOR if d == "yuquan" else "#E08214" for d in dsets]

        ax_sc.scatter(ana_fs, emp_fs, c=colors, s=40, edgecolors="white", linewidths=0.5,
                      zorder=3)
        lims = [0.5, 5.5]
        ax_sc.plot(lims, lims, color="gray", ls="--", lw=0.8, alpha=0.5, zorder=1)

        r_val, p_val = pearsonr(ana_fs, emp_fs)
        dfs = np.abs(emp_fs - ana_fs)
        n_close = int(np.sum(dfs < 0.5))
        n_mid = int(np.sum(dfs < 1.0))

        ax_sc.set_xlabel("Analytic peak (Hz)", fontsize=9)
        ax_sc.set_ylabel("Empirical peak (Hz)", fontsize=9)
        ax_sc.set_xlim(*lims)
        ax_sc.set_ylim(*lims)
        ax_sc.set_aspect("equal")
        ax_sc.grid(True, alpha=0.15)
        ax_sc.set_title(
            f"Cohort (n={n_peak})\n"
            f"|Δf| med={np.median(dfs):.2f} Hz, <0.5 Hz: {n_close}/{n_peak}, "
            f"<1 Hz: {n_mid}/{n_peak}\n"
            f"r={r_val:.2f}, p={p_val:.2g}",
            fontsize=8, pad=4)

        for d_label, d_color, d_name in [("yuquan", _EMP_COLOR, "Yuquan"),
                                          ("epilepsiae", "#E08214", "Epilepsiae")]:
            ax_sc.scatter([], [], c=d_color, s=30, label=d_name)
        ax_sc.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Exp 6A: Shifted-Gamma Renewal PSD vs Empirical Delta-Train PSD\n"
        "Top row: best peak-location matches (sorted by |Δf|) · "
        "Bottom left: worst matches · Bottom right: cohort summary",
        fontsize=11)
    fig.savefig(FIG_DIR / "exp6a_renewal_overlay.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved exp6a_renewal_overlay.png")

    # ---- Print cohort summary ----
    if with_peak:
        dfs_all = [r["delta_f"] for r in with_peak]
        print(f"  [cohort] {n_peak} subjects with specparam peak:")
        print(f"    |Δf| median={np.median(dfs_all):.2f}, "
              f"<0.5Hz: {sum(d<0.5 for d in dfs_all)}/{n_peak}, "
              f"<1Hz: {sum(d<1.0 for d in dfs_all)}/{n_peak}")

    # ---- Figure 6B: SOZ vs non-SOZ dead-time ----
    if not dt_data:
        return

    pairs = []
    for key, rec in sorted(dt_data.items()):
        if not isinstance(rec, dict) or "error" in rec or "warning" in rec:
            continue
        soz, nsoz = rec.get("soz", {}), rec.get("nonsoz", {})
        sn, nn = soz.get("n_events", 0), nsoz.get("n_events", 0)
        if sn < _MIN_EVENTS_DT or nn < _MIN_EVENTS_DT:
            continue
        if "iei_p02" not in soz or "iei_p02" not in nsoz:
            continue
        pairs.append({
            "key": key, "sub": key.split("/")[-1],
            "soz_p02": soz["iei_p02"], "nsoz_p02": nsoz["iei_p02"],
            "soz_med": soz["iei_median"], "nsoz_med": nsoz["iei_median"],
            "soz_n": sn, "nsoz_n": nn,
        })

    if len(pairs) < 3:
        print(f"  Only {len(pairs)} valid dead-time pairs (need ≥3), skipping Fig 6B")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    for ax, soz_key, nsoz_key, label in [
        (axes[0], "soz_p02", "nsoz_p02", "IEI 2nd percentile (dead-time proxy)"),
        (axes[1], "soz_med", "nsoz_med", "IEI median"),
    ]:
        svals = np.array([p[soz_key] for p in pairs])
        nvals = np.array([p[nsoz_key] for p in pairs])

        for i, p_rec in enumerate(pairs):
            ax.plot([0, 1], [svals[i], nvals[i]], color="#AAAAAA", lw=0.9, zorder=1)
        ax.scatter(np.zeros(len(svals)), svals, c=_SOZ_COLOR, s=50,
                   edgecolors="white", linewidths=0.5, zorder=3, label="SOZ")
        ax.scatter(np.ones(len(nvals)), nvals, c=_NSOZ_COLOR, s=50,
                   edgecolors="white", linewidths=0.5, zorder=3, label="non-SOZ")

        for i, p_rec in enumerate(pairs):
            ax.annotate(p_rec["sub"], (0, svals[i]), fontsize=5.5,
                        xytext=(-6, 0), textcoords="offset points", ha="right",
                        color=_SOZ_COLOR, alpha=0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["SOZ\nevents", "non-SOZ\nevents"], fontsize=9)
        ax.set_ylabel(f"{label} (s)", fontsize=9)
        ax.set_yscale("log")
        ax.set_xlim(-0.4, 1.4)
        ax.grid(True, axis="y", alpha=0.15, which="both")

        try:
            stat = wilcoxon(svals, nvals, alternative="two-sided")
            p_txt = f"Wilcoxon p={stat.pvalue:.3g}"
        except Exception:
            p_txt = "Wilcoxon p=N/A"

        direction = "SOZ < non-SOZ" if np.median(svals) < np.median(nvals) else "SOZ ≥ non-SOZ"
        ax.set_title(f"{label}\nn={len(pairs)} pairs (both groups ≥{_MIN_EVENTS_DT} events)\n"
                     f"{p_txt}, {direction}", fontsize=8.5, pad=4)

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Exp 6B: SOZ vs non-SOZ Dead-Time Comparison", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_DIR / "exp6b_soz_deadtime.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved exp6b_soz_deadtime.png ({len(pairs)} pairs)")


# ==========================================================================

def plot_exp7():
    """Fig 7A-D: serial-correlation decay, detrending, within-block, SOZ split."""
    data = _load("exp7_serial_corr_deep.json")
    if not data:
        return

    valid = [
        rec for rec in data.values()
        if isinstance(rec, dict) and "error" not in rec and "warning" not in rec
    ]
    if not valid:
        print("  No valid exp7 data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))

    # ---- 7A: lag-k decay ----
    ax = axes[0, 0]
    lag_map = {}
    half_lives = []
    half_life_secs = []
    for rec in valid:
        decay = rec.get("serial_decay", {})
        lags = np.asarray(decay.get("lags", []), dtype=float)
        rs = np.asarray(decay.get("rs", []), dtype=float)
        if lags.size == 0 or rs.size == 0:
            continue
        ax.plot(lags, rs, color="0.7", lw=0.8, alpha=0.55)
        for lag, r in zip(lags, rs):
            lag_map.setdefault(int(lag), []).append(float(r))
        half_lag = decay.get("half_life_lag")
        half_sec = decay.get("half_life_sec")
        if half_lag is not None and np.isfinite(half_lag):
            half_lives.append(float(half_lag))
        if half_sec is not None and np.isfinite(half_sec):
            half_life_secs.append(float(half_sec))

    if lag_map:
        common_lags = np.array(sorted(lag_map))
        median_rs = np.array([np.median(lag_map[int(lag)]) for lag in common_lags])
        ax.plot(common_lags, median_rs, color="black", lw=2.2, label="median across subjects")
    if half_lives:
        med_half_lag = float(np.median(half_lives))
        med_half_sec = float(np.median(half_life_secs)) if half_life_secs else np.nan
        ax.axvline(med_half_lag, color="#B2182B", ls="--", lw=1.1, alpha=0.8)
        ax.text(
            med_half_lag,
            0.02,
            f"median half-life\n{med_half_lag:.1f} lags\n{med_half_sec:.0f} s",
            color="#B2182B",
            fontsize=8,
            ha="left",
            va="bottom",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Lag k", fontsize=9)
    ax.set_ylabel("r(log IEI[n], log IEI[n+k])", fontsize=9)
    ax.set_title("Fig 7A: Lag-k serial-correlation decay", fontsize=10)
    ax.grid(True, alpha=0.18, which="both")
    if lag_map:
        ax.legend(fontsize=8, loc="upper right")

    # ---- 7B: detrended vs raw ----
    ax = axes[0, 1]
    raw_vals = []
    det_vals = []
    det_fracs = []
    for rec in valid:
        det = rec.get("detrended", {})
        raw_r = det.get("raw_r")
        detrended_r = det.get("detrended_r")
        if raw_r is None or detrended_r is None:
            continue
        if not (np.isfinite(raw_r) and np.isfinite(detrended_r)):
            continue
        raw_vals.append(float(raw_r))
        det_vals.append(float(detrended_r))
        frac = det.get("detrend_fraction")
        if frac is not None and np.isfinite(frac):
            det_fracs.append(float(frac))
    raw_vals = np.asarray(raw_vals, dtype=float)
    det_vals = np.asarray(det_vals, dtype=float)
    if raw_vals.size:
        lim_lo = min(np.min(raw_vals), np.min(det_vals)) - 0.02
        lim_hi = max(np.max(raw_vals), np.max(det_vals)) + 0.02
        ax.scatter(raw_vals, det_vals, s=38, c="#2166AC", alpha=0.8,
                   edgecolors="white", linewidths=0.5)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.8, alpha=0.5)
        try:
            stat = wilcoxon(raw_vals, det_vals, alternative="greater")
            p_txt = f"Wilcoxon p={stat.pvalue:.3g}"
        except Exception:
            p_txt = "Wilcoxon p=N/A"
        frac_txt = f"median detrend fraction={np.median(det_fracs):.2f}" if det_fracs else "median detrend fraction=N/A"
        ax.set_title(
            "Fig 7B: Raw vs detrended lag-1 r\n"
            f"n={raw_vals.size}, {p_txt}\n{frac_txt}",
            fontsize=10,
        )
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
    else:
        ax.text(0.5, 0.5, "no valid pairs", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Fig 7B: Raw vs detrended lag-1 r", fontsize=10)
    ax.set_xlabel("Raw lag-1 r", fontsize=9)
    ax.set_ylabel("Detrended lag-1 r", fontsize=9)
    ax.grid(True, alpha=0.18)

    # ---- 7C: within-block distribution ----
    ax = axes[1, 0]
    block_rs = []
    pooled_rs = []
    for rec in valid:
        within = rec.get("within_block", {})
        pooled = within.get("pooled_r")
        if pooled is not None and np.isfinite(pooled):
            pooled_rs.append(float(pooled))
        for blk in within.get("per_block", []):
            r = blk.get("serial_corr")
            if r is not None and np.isfinite(r):
                block_rs.append(float(r))
    if block_rs:
        x_block = np.random.default_rng(0).normal(loc=0.0, scale=0.03, size=len(block_rs))
        ax.scatter(x_block, block_rs, s=18, color="0.75", alpha=0.6, label="per-block r")
    if pooled_rs:
        x_pool = np.random.default_rng(1).normal(loc=1.0, scale=0.03, size=len(pooled_rs))
        ax.scatter(x_pool, pooled_rs, s=28, color="black", alpha=0.85, label="subject pooled within-block r")
        ax.hlines(np.median(pooled_rs), 0.82, 1.18, color="#B2182B", lw=2.0)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Per-block", "Pooled by subject"], fontsize=9)
    ax.set_ylabel("Lag-1 r", fontsize=9)
    ax.set_title(
        "Fig 7C: Within-block lag-1 serial correlation\n"
        f"blocks={len(block_rs)}, subjects={len(pooled_rs)}",
        fontsize=10,
    )
    ax.grid(True, axis="y", alpha=0.18)
    if block_rs or pooled_rs:
        ax.legend(fontsize=8, loc="upper right")

    # ---- 7D: SOZ vs non-SOZ ----
    ax = axes[1, 1]
    soz_vals = []
    nsoz_vals = []
    for rec in valid:
        strat = rec.get("soz_stratified", {})
        soz = strat.get("soz", {})
        nsoz = strat.get("nonsoz", {})
        if soz.get("n_events", 0) < 50 or nsoz.get("n_events", 0) < 50:
            continue
        r_soz = soz.get("lag1_r")
        r_nsoz = nsoz.get("lag1_r")
        if r_soz is None or r_nsoz is None:
            continue
        if not (np.isfinite(r_soz) and np.isfinite(r_nsoz)):
            continue
        soz_vals.append(float(r_soz))
        nsoz_vals.append(float(r_nsoz))
    soz_vals = np.asarray(soz_vals, dtype=float)
    nsoz_vals = np.asarray(nsoz_vals, dtype=float)
    if soz_vals.size:
        for i in range(soz_vals.size):
            ax.plot([0, 1], [soz_vals[i], nsoz_vals[i]], color="0.75", lw=0.9)
        ax.scatter(np.zeros(soz_vals.size), soz_vals, c=_SOZ_COLOR, s=48,
                   edgecolors="white", linewidths=0.5, label="SOZ")
        ax.scatter(np.ones(nsoz_vals.size), nsoz_vals, c=_NSOZ_COLOR, s=48,
                   edgecolors="white", linewidths=0.5, label="non-SOZ")
        try:
            stat = wilcoxon(soz_vals, nsoz_vals, alternative="two-sided")
            p_txt = f"Wilcoxon p={stat.pvalue:.3g}"
        except Exception:
            p_txt = "Wilcoxon p=N/A"
        direction = "SOZ > non-SOZ" if np.median(soz_vals) > np.median(nsoz_vals) else "SOZ <= non-SOZ"
        ax.set_title(
            "Fig 7D: SOZ vs non-SOZ lag-1 r\n"
            f"n={soz_vals.size} pairs (both groups >=50 events)\n{p_txt}, {direction}",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="upper left")
    else:
        ax.text(0.5, 0.5, "no valid SOZ/non-SOZ pairs", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Fig 7D: SOZ vs non-SOZ lag-1 r", fontsize=10)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SOZ", "non-SOZ"], fontsize=9)
    ax.set_ylabel("Lag-1 r", fontsize=9)
    ax.grid(True, axis="y", alpha=0.18)

    fig.suptitle("Exp 7: IEI Serial-Correlation Deep Dive", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "exp7_serial_corr_deep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved exp7_serial_corr_deep.png")


# ==========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all")
    args = parser.parse_args()

    if args.exp == "all":
        exps = {1, 2, 3, 4, 5, 6, 7}
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
    if 6 in exps:
        plot_exp6()
    if 7 in exps:
        plot_exp7()

    print(f"\nDone. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
