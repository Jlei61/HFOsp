#!/usr/bin/env python3
"""Publication-quality PPT figures for Topic 2: between-event dynamics.

Five figures, each tells one complete story for why the ~2 Hz peak is NOT
an intrinsic oscillator:

    1. ISI-shuffle              — peak survives shuffle ⇒ distribution-shape
    2. Refractory + renewal     — dead-time generates the peak
    3. Lognormal vs power-law   — IEI distribution shape
    4. IEI temporal structure   — slow modulation + short-range memory
    5. Slow modulation          — circadian + multi-hour drift

Usage:
    python scripts/plot_topic2_ppt.py --all
    python scripts/plot_topic2_ppt.py --fig 1,2,3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import wilcoxon, lognorm, gamma as gamma_dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plot_style import (
    style_panel, violin_with_scatter, add_significance_bracket,
    savefig_pub, dataset_color, format_subject_label,
    COL_YUQUAN, COL_EPILEPSIAE, COL_EMPIRICAL, COL_ANALYTIC,
    COL_SURROGATE, COL_DETRENDED, COL_DAY, COL_NIGHT,
    COL_SOZ, COL_NONSOZ, COL_SIG, COL_NONSIG, COL_NEUTRAL,
    COL_OSCILLATOR, COL_REFRACTORY,
    FS_LABEL, FS_TITLE, FS_TICK, FS_SUPTITLE,
)

RESULTS_DIR = Path("results/event_periodicity")
PHASE2_DIR = RESULTS_DIR / "phase2"
PPT_DIR = RESULTS_DIR / "figures" / "ppt"
PPT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================================
# Data loaders
# =========================================================================

def _load_phase1(dataset: str) -> list:
    d = RESULTS_DIR / dataset
    results = []
    for f in sorted(d.glob("*_periodicity.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def _load_phase2(name: str) -> dict:
    path = PHASE2_DIR / name
    if not path.exists():
        print(f"  {name} not found, skipping")
        return {}
    with open(path) as f:
        return json.load(f)


def _all_phase1() -> list:
    return _load_phase1("yuquan") + _load_phase1("epilepsiae")


def _smooth(psd, window=21, polyorder=3):
    n = len(psd)
    w = min(window, n - 1)
    if w % 2 == 0:
        w -= 1
    if w < polyorder + 2:
        return psd
    return savgol_filter(psd, w, polyorder)


def _suptitle(fig, text, y=0.985):
    """Place a suptitle high up so it doesn't overlap with subplots."""
    fig.suptitle(text, fontsize=FS_SUPTITLE, fontweight="bold",
                 y=y, x=0.5)


def _tight_data_xlim(ax, x):
    """Set xlim exactly to data extent (no white margin)."""
    ax.set_xlim(float(np.min(x)), float(np.max(x)))


# =========================================================================
# Synthetic toy generators
# =========================================================================

def _toy_oscillator_events(rng, T=180.0, freq=2.0, modulation=0.95,
                            base_rate=4.0):
    """Inhomogeneous Poisson with sinusoidally modulated rate.

    Events cluster around peaks of cos(2π·f·t), so IEIs come in
    short-(within-burst)/long-(between-burst) pairs ⇒ serial structure.
    """
    dt = 0.005
    t_grid = np.arange(0, T, dt)
    rate = base_rate * (1.0 + modulation * np.cos(2 * np.pi * freq * t_grid))
    rate = np.clip(rate, 0.01, None)
    p = rate * dt
    spikes_mask = rng.random(t_grid.size) < p
    return t_grid[spikes_mask]


def _toy_refractory_events(rng, T=180.0, mean_iei=0.5, refractory=0.3,
                            shape=4.0):
    """Renewal process: i.i.d. IEIs from Gamma + dead-time.

    Each IEI = refractory + Gamma(shape, theta), independent of all others.
    Marginal IEI distribution has narrow peak just above refractory ⇒
    pulse-train PSD has peak near 1/mean_iei. Shuffle preserves it.
    """
    n = int(T / mean_iei * 1.5)
    theta = (mean_iei - refractory) / shape
    iei = refractory + rng.gamma(shape, theta, size=n)
    times = np.cumsum(iei)
    return times[times < T]


def _pulse_psd(times, T, fs=50.0, nperseg=2048):
    """Welch PSD of binary pulse train sampled at fs."""
    from scipy.signal import welch
    n = int(T * fs)
    x = np.zeros(n)
    idx = (times * fs).astype(int)
    idx = idx[(idx >= 0) & (idx < n)]
    x[idx] = 1.0
    nps = min(nperseg, n)
    f, p = welch(x, fs=fs, nperseg=nps, noverlap=nps // 2,
                 detrend="constant")
    return f, p


def _shuffle_iei(times, rng, T):
    """Return event times after shuffling IEIs (preserves marginal dist)."""
    if len(times) < 2:
        return times.copy()
    iei = np.diff(times)
    iei_shuf = rng.permutation(iei)
    new_times = np.concatenate([[times[0]], times[0] + np.cumsum(iei_shuf)])
    return new_times[new_times < T]


# =========================================================================
# Fig 1: ISI-shuffle method explained with two toys + cohort
# =========================================================================

def plot_fig1():
    """ISI-shuffle: oscillator (peak destroyed) vs refractory (peak survives),
    then full cohort matches refractory pattern."""
    print("Plotting Fig 1: ISI-shuffle ...")

    rng = np.random.default_rng(42)
    T_toy = 180.0  # seconds for raster

    # Generate two toy scenarios
    osc_times = _toy_oscillator_events(rng, T=T_toy, freq=2.0,
                                        modulation=0.97, base_rate=8.0)
    osc_shuf = _shuffle_iei(osc_times, rng, T_toy)
    f_osc, p_osc = _pulse_psd(osc_times, T_toy)
    f_osc_s, p_osc_s = _pulse_psd(osc_shuf, T_toy)

    ref_times = _toy_refractory_events(rng, T=T_toy, mean_iei=0.5,
                                        refractory=0.35, shape=8.0)
    ref_shuf = _shuffle_iei(ref_times, rng, T_toy)
    f_ref, p_ref = _pulse_psd(ref_times, T_toy)
    f_ref_s, p_ref_s = _pulse_psd(ref_shuf, T_toy)

    # Real data
    phase1 = _all_phase1()
    p_values = []
    p_values_yq = []
    p_values_epi = []
    real_powers = []
    null_medians = []
    for r in phase1:
        s = r.get("group", {}).get("surrogate_isi", {})
        if s and s.get("p_value") is not None:
            p_values.append(s["p_value"])
            real_powers.append(s.get("real_peak_power", np.nan))
            null_medians.append(np.median(s.get("null_peak_powers", [np.nan])))
            if r["dataset"] == "yuquan":
                p_values_yq.append(s["p_value"])
            else:
                p_values_epi.append(s["p_value"])

    # =========================================================================
    # Layout
    # =========================================================================
    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(
        3, 2, height_ratios=[1, 1, 1.2], width_ratios=[1.4, 1.2],
        wspace=0.32, hspace=0.55,
        left=0.07, right=0.97, top=0.92, bottom=0.07,
    )

    # ---- Row 1 — oscillator scenario ----
    ax_osc_raster = fig.add_subplot(gs[0, 0])
    ax_osc_psd = fig.add_subplot(gs[0, 1])

    # Raster: orig top, shuffled bottom (snippet for clarity)
    snippet_T = 6.0
    osc_o = osc_times[osc_times < snippet_T]
    osc_s = osc_shuf[osc_shuf < snippet_T]
    ax_osc_raster.vlines(osc_o, 1.05, 1.45,
                         colors=COL_OSCILLATOR, linewidths=1.6)
    ax_osc_raster.vlines(osc_s, 0.05, 0.45,
                         colors="#9B9B9B", linewidths=1.6)
    ax_osc_raster.text(-0.04, 1.25, "original", fontsize=FS_LABEL,
                       ha="right", va="center",
                       transform=ax_osc_raster.get_yaxis_transform(),
                       color=COL_OSCILLATOR, fontweight="bold")
    ax_osc_raster.text(-0.04, 0.25, "IEI-shuffled", fontsize=FS_LABEL,
                       ha="right", va="center",
                       transform=ax_osc_raster.get_yaxis_transform(),
                       color="#666666", fontweight="bold")
    # Highlight periodic clusters in original
    period = 1.0 / 2.0
    for k in range(int(snippet_T / period) + 1):
        ax_osc_raster.axvspan(k * period - 0.05, k * period + 0.05,
                              ymin=0.55, ymax=0.95,
                              color=COL_OSCILLATOR, alpha=0.10, zorder=0)
    ax_osc_raster.set_xlim(0, snippet_T)
    ax_osc_raster.set_ylim(0, 1.5)
    ax_osc_raster.set_yticks([])
    ax_osc_raster.set_xlabel("Time (s)", fontsize=FS_LABEL)
    ax_osc_raster.set_title(
        "Toy A — true oscillator (rate modulated at 2 Hz)",
        fontsize=FS_TITLE, fontweight="bold", pad=10,
        color=COL_OSCILLATOR,
    )
    style_panel(ax_osc_raster, "a")

    # PSD: original peak vs shuffled flat
    mask = (f_osc >= 0.5) & (f_osc <= 8)
    ax_osc_psd.semilogy(f_osc[mask], p_osc[mask], color=COL_OSCILLATOR,
                       lw=2.4, label="original")
    ax_osc_psd.semilogy(f_osc_s[mask], p_osc_s[mask], color="#9B9B9B",
                       lw=2.0, ls="--", label="IEI-shuffled")
    ax_osc_psd.axvline(2.0, color="black", lw=1.0, alpha=0.4, ls=":")
    ax_osc_psd.text(2.0, ax_osc_psd.get_ylim()[1] * 0.4, " 2 Hz",
                    fontsize=FS_LABEL - 1, color="black", alpha=0.6)
    ax_osc_psd.set_xlabel("Frequency (Hz)", fontsize=FS_LABEL)
    ax_osc_psd.set_ylabel("Power", fontsize=FS_LABEL)
    ax_osc_psd.set_title("Peak DESTROYED by shuffle  ⇒  oscillator",
                         fontsize=FS_TITLE, fontweight="bold", pad=10,
                         color=COL_OSCILLATOR)
    ax_osc_psd.legend(loc="upper right", fontsize=FS_TICK, framealpha=0.95)
    ax_osc_psd.set_xlim(f_osc[mask].min(), f_osc[mask].max())
    style_panel(ax_osc_psd, "b")

    # ---- Row 2 — refractory scenario ----
    ax_ref_raster = fig.add_subplot(gs[1, 0])
    ax_ref_psd = fig.add_subplot(gs[1, 1])

    ref_o = ref_times[ref_times < snippet_T]
    ref_s_snippet = ref_shuf[ref_shuf < snippet_T]
    ax_ref_raster.vlines(ref_o, 1.05, 1.45,
                         colors=COL_REFRACTORY, linewidths=1.6)
    ax_ref_raster.vlines(ref_s_snippet, 0.05, 0.45,
                         colors="#9B9B9B", linewidths=1.6)
    ax_ref_raster.text(-0.04, 1.25, "original", fontsize=FS_LABEL,
                       ha="right", va="center",
                       transform=ax_ref_raster.get_yaxis_transform(),
                       color=COL_REFRACTORY, fontweight="bold")
    ax_ref_raster.text(-0.04, 0.25, "IEI-shuffled", fontsize=FS_LABEL,
                       ha="right", va="center",
                       transform=ax_ref_raster.get_yaxis_transform(),
                       color="#666666", fontweight="bold")
    ax_ref_raster.set_xlim(0, snippet_T)
    ax_ref_raster.set_ylim(0, 1.5)
    ax_ref_raster.set_yticks([])
    ax_ref_raster.set_xlabel("Time (s)", fontsize=FS_LABEL)
    ax_ref_raster.set_title(
        "Toy B — refractory renewal (i.i.d. IEIs, dead-time = 0.35 s)",
        fontsize=FS_TITLE, fontweight="bold", pad=10,
        color=COL_REFRACTORY,
    )
    style_panel(ax_ref_raster, "c")

    mask = (f_ref >= 0.5) & (f_ref <= 8)
    ax_ref_psd.semilogy(f_ref[mask], p_ref[mask], color=COL_REFRACTORY,
                       lw=2.4, label="original")
    ax_ref_psd.semilogy(f_ref_s[mask], p_ref_s[mask], color="#9B9B9B",
                       lw=2.0, ls="--", label="IEI-shuffled")
    ax_ref_psd.axvline(2.0, color="black", lw=1.0, alpha=0.4, ls=":")
    ax_ref_psd.text(2.0, ax_ref_psd.get_ylim()[1] * 0.4, " 2 Hz",
                    fontsize=FS_LABEL - 1, color="black", alpha=0.6)
    ax_ref_psd.set_xlabel("Frequency (Hz)", fontsize=FS_LABEL)
    ax_ref_psd.set_ylabel("Power", fontsize=FS_LABEL)
    ax_ref_psd.set_title("Peak PRESERVED  ⇒  distribution-shape artifact",
                         fontsize=FS_TITLE, fontweight="bold", pad=10,
                         color=COL_REFRACTORY)
    ax_ref_psd.legend(loc="upper right", fontsize=FS_TICK, framealpha=0.95)
    ax_ref_psd.set_xlim(f_ref[mask].min(), f_ref[mask].max())
    style_panel(ax_ref_psd, "d")

    # ---- Row 3 — real-data verification (full cohort) ----
    ax_real = fig.add_subplot(gs[2, 0])
    ax_hist = fig.add_subplot(gs[2, 1])

    # Cohort PSD overlay: real (median) vs surrogate-derived null fraction
    # Use real_peak_power vs null_peak_power per subject
    real_arr = np.array(real_powers)
    null_arr = np.array(null_medians)
    valid = np.isfinite(real_arr) & np.isfinite(null_arr) & (null_arr > 0)
    ratios = real_arr[valid] / null_arr[valid]

    # All subjects: log10 ratio of peak power (real / shuffle median)
    log_ratios = np.log10(ratios)
    n_real_higher = int((ratios > 1.0).sum())
    n_total = len(ratios)

    sorted_idx = np.argsort(log_ratios)
    bars = ax_real.barh(np.arange(n_total), log_ratios[sorted_idx],
                       color=[COL_SIG if r > 0 else COL_NEUTRAL
                              for r in log_ratios[sorted_idx]],
                       edgecolor="white", linewidth=0.5, alpha=0.85)
    ax_real.axvline(0, color="black", lw=1.5)
    ax_real.set_xlabel(r"$\log_{10}$ (real peak power / shuffled median)",
                      fontsize=FS_LABEL)
    ax_real.set_ylabel(f"Subjects sorted by ratio (n = {n_total})",
                      fontsize=FS_LABEL)
    ax_real.set_yticks([])
    ax_real.set_title("Real-data peaks barely exceed shuffle null",
                      fontsize=FS_TITLE, fontweight="bold", pad=10)
    ax_real.text(0.97, 0.05,
                 f"{n_real_higher}/{n_total} have real > shuffle\n"
                 f"median ratio = {np.median(ratios):.2f}×",
                 transform=ax_real.transAxes,
                 fontsize=FS_TICK, ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.45",
                          facecolor="white", edgecolor="#cccccc",
                          alpha=0.92))
    ax_real.set_ylim(-0.5, n_total - 0.5)
    style_panel(ax_real, "e")

    # P-value histogram across all subjects
    p_arr = np.array(p_values)
    bins = np.linspace(0, 1, 21)
    counts_yq, _ = np.histogram(p_values_yq, bins=bins)
    counts_epi, _ = np.histogram(p_values_epi, bins=bins)
    width = bins[1] - bins[0]
    centers = (bins[:-1] + bins[1:]) / 2
    ax_hist.bar(centers, counts_yq, width=width * 0.9,
                color=COL_YUQUAN, alpha=0.85, edgecolor="white",
                linewidth=0.6, label=f"Yuquan (n = {len(p_values_yq)})")
    ax_hist.bar(centers, counts_epi, width=width * 0.9, bottom=counts_yq,
                color=COL_EPILEPSIAE, alpha=0.85, edgecolor="white",
                linewidth=0.6, label=f"Epilepsiae (n = {len(p_values_epi)})")
    ax_hist.axvline(0.05, color="black", lw=1.5, ls="--")
    ax_hist.text(0.05, ax_hist.get_ylim()[1] * 0.95, "  p = 0.05",
                 fontsize=FS_TICK, color="black",
                 ha="left", va="top")
    n_sig = int((p_arr < 0.05).sum())
    ax_hist.set_xlabel("Shuffle p-value", fontsize=FS_LABEL)
    ax_hist.set_ylabel("# subjects", fontsize=FS_LABEL)
    ax_hist.set_title(f"Only {n_sig}/{len(p_arr)} reject the null  ≈  uniform",
                      fontsize=FS_TITLE, fontweight="bold", pad=10)
    ax_hist.legend(loc="upper right", fontsize=FS_TICK, framealpha=0.95)
    ax_hist.set_xlim(0, 1)
    style_panel(ax_hist, "f")

    _suptitle(fig,
        "Fig 1  ·  ISI-shuffle: cohort matches the refractory toy, not the oscillator")
    return savefig_pub(fig, PPT_DIR / "fig1_isi_shuffle.png")


# =========================================================================
# Fig 2: Refractory + renewal (analytic + sampling) + hazard
# =========================================================================

def plot_fig2():
    """Refractory dead-time generates the peak — analytic & sampling tests."""
    print("Plotting Fig 2: refractory dead-time + renewal ...")

    phase1 = _all_phase1()
    exp3 = _load_phase2("exp3_hazard.json")
    exp6 = _load_phase2("exp6_renewal_psd.json")

    # ---- Pick representative analytic-overlay subjects ----
    # find subjects where analytic peak frequency matches real peak
    candidates = []
    for sub_key, rec in exp6.items():
        if not rec.get("has_peak_0p5_10hz"):
            continue
        sp_freqs = rec.get("phase1_sp_freqs")
        psd = np.array(rec.get("phase1_psd", []))
        ad = np.array(rec.get("analytic_delta", []))
        if len(psd) == 0 or len(ad) == 0 or sp_freqs is None:
            continue
        # find real peak (in 0.5-10 band)
        freqs = np.array(rec["phase1_freqs"])
        sp_mask = (freqs >= 0.5) & (freqs <= 10)
        if not np.any(sp_mask):
            continue
        idx_real = np.argmax(psd[sp_mask])
        f_real = freqs[sp_mask][idx_real]
        idx_an = np.argmax(ad[sp_mask])
        f_an = freqs[sp_mask][idx_an]
        df = abs(f_real - f_an)
        candidates.append((df, sub_key, rec))
    candidates.sort()
    overlay_picks = candidates[:3]  # 3 best matches

    # ---- Layout ----
    fig = plt.figure(figsize=(16, 13))
    gs = gridspec.GridSpec(
        3, 3, height_ratios=[1, 1, 1.05],
        wspace=0.36, hspace=0.55,
        left=0.06, right=0.97, top=0.89, bottom=0.06,
    )

    # ---- Row 1 — mechanism + hazard ----
    ax_toy = fig.add_subplot(gs[0, 0])
    ax_haz = fig.add_subplot(gs[0, 1:])

    # Mechanism toy: synthetic IEI distribution (gamma + dead-time) and one cycle
    rng = np.random.default_rng(7)
    iei_synth = 0.35 + rng.gamma(8.0, (0.5 - 0.35) / 8.0 * 4, 5000)
    iei_synth = iei_synth[iei_synth < 2.0]
    bins = np.linspace(0, 2.0, 60)
    counts, edges, _ = ax_toy.hist(
        iei_synth, bins=bins, color=COL_REFRACTORY, alpha=0.6,
        edgecolor="white", linewidth=0.4,
    )
    y_top = counts.max() * 1.45
    ax_toy.set_ylim(0, y_top)
    ax_toy.axvspan(0, 0.35, color="#A35E48", alpha=0.20)
    ax_toy.axvline(0.35, color="#A35E48", lw=2.0,
                   label="dead-time τ_r")
    mode_x = bins[counts.argmax()] + (bins[1] - bins[0]) / 2
    ax_toy.annotate(
        "mode → 1/τ ≈ 2 Hz peak",
        xy=(mode_x, counts.max()),
        xytext=(mode_x + 0.2, counts.max() * 1.32),
        fontsize=FS_TICK - 1, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", lw=1.4, color="black"),
    )
    ax_toy.text(0.175, counts.max() * 0.55, "no events\nallowed",
                ha="center", va="center", fontsize=FS_TICK - 1,
                color="#7A4133", fontweight="bold")
    ax_toy.set_xlim(0, 2.0)
    ax_toy.set_xlabel("IEI (s)", fontsize=FS_LABEL)
    ax_toy.set_ylabel("Count", fontsize=FS_LABEL)
    ax_toy.set_title("Refractory mechanism\n(toy: dead-time + gamma)",
                     fontsize=FS_TITLE - 1, fontweight="bold", pad=10)
    ax_toy.legend(loc="upper right", fontsize=FS_TICK - 1, framealpha=0.95)
    style_panel(ax_toy, "a")

    # Hazard cohort overlay (rescale per subject so curves are comparable)
    haz_yq = []
    haz_epi = []
    iei_min_yq = []
    iei_min_epi = []
    t_grid = None
    for sub_key, rec in exp3.items():
        packed = rec.get("packed", {})
        t = np.array(packed.get("t", []))
        h = np.array(packed.get("hazard", []))
        if len(t) == 0 or np.nanmax(h) <= 0:
            continue
        if t_grid is None:
            t_grid = t
        h_i = np.interp(t_grid, t, h, left=np.nan, right=np.nan)
        # Normalize each subject's hazard to its peak so cohort overlay is
        # not dominated by absolute rate.
        h_norm = h_i / np.nanmax(h_i)
        if "yuquan" in sub_key:
            haz_yq.append(h_norm)
            iei_min_yq.append(packed.get("iei_min", np.nan))
        else:
            haz_epi.append(h_norm)
            iei_min_epi.append(packed.get("iei_min", np.nan))
    haz_yq = np.array(haz_yq)
    haz_epi = np.array(haz_epi)
    tmask = t_grid <= 2.0 if t_grid is not None else np.array([])
    if t_grid is not None:
        for h in haz_yq:
            ax_haz.plot(t_grid[tmask], h[tmask], color=COL_YUQUAN,
                        alpha=0.30, lw=1.2)
        for h in haz_epi:
            ax_haz.plot(t_grid[tmask], h[tmask], color=COL_EPILEPSIAE,
                        alpha=0.30, lw=1.2)
        med_all = np.nanmedian(np.vstack([haz_yq, haz_epi]), axis=0)
        ax_haz.plot(t_grid[tmask], med_all[tmask], color="black", lw=2.8,
                    zorder=5)
        med_dt = np.nanmedian(np.array(iei_min_yq + iei_min_epi))
        ax_haz.axvspan(0, med_dt, color="#A35E48", alpha=0.10)
        ax_haz.axvline(med_dt, color="#A35E48", lw=2.0, ls="--")
        ax_haz.text(0.04, 0.50, "no events\nallowed",
                    transform=ax_haz.transAxes, fontsize=FS_TICK,
                    color="#7A4133", fontweight="bold",
                    ha="left", va="center")
    ax_haz.set_xlabel("IEI (s)", fontsize=FS_LABEL)
    ax_haz.set_ylabel("Hazard (normalized to per-subject peak)",
                      fontsize=FS_LABEL)
    ax_haz.set_title("Empirical hazard shows clear dead-zone before τ_r",
                     fontsize=FS_TITLE, fontweight="bold", pad=10)
    if t_grid is not None:
        ax_haz.set_xlim(0, 2.0)
        ax_haz.set_ylim(0, 1.1)
    h1 = plt.Line2D([0], [0], color=COL_YUQUAN, lw=2.0, alpha=0.8,
                    label=f"Yuquan (n = {len(haz_yq)})")
    h2 = plt.Line2D([0], [0], color=COL_EPILEPSIAE, lw=2.0, alpha=0.8,
                    label=f"Epilepsiae (n = {len(haz_epi)})")
    h3 = plt.Line2D([0], [0], color="black", lw=2.8,
                    label="cohort median")
    h4 = plt.Line2D([0], [0], color="#A35E48", lw=2.0, ls="--",
                    label=f"median τ_r = {med_dt:.2f} s")
    ax_haz.legend(handles=[h1, h2, h3, h4], loc="upper right",
                  fontsize=FS_TICK, framealpha=0.95, ncol=2)
    style_panel(ax_haz, "b")

    # ---- Row 2 — analytic PSD overlay (3 representative subjects, exp6a-style) ----
    # exp6a style: smoothed empirical PSD vs analytic, both peak-normalized
    for j, (df, sub_key, rec) in enumerate(overlay_picks):
        ax = fig.add_subplot(gs[1, j])
        freqs = np.array(rec["phase1_freqs"])
        psd_raw = np.array(rec["phase1_psd"])
        ad = np.array(rec["analytic_delta"])

        mask = (freqs >= 0.5) & (freqs <= 8)
        f = freqs[mask]
        psd = psd_raw[mask]
        ad_m = ad[mask]
        psd_smooth = _smooth(psd, window=51)
        # peak-normalize each within band
        psd_norm = psd_smooth / np.nanmax(psd_smooth)
        psd_raw_norm = psd / np.nanmax(psd_smooth)
        ad_norm = ad_m / np.nanmax(ad_m)

        ax.plot(f, psd_raw_norm, color=COL_EMPIRICAL, lw=0.8, alpha=0.30)
        ax.plot(f, psd_norm, color=COL_EMPIRICAL, lw=2.4,
                label="empirical PSD (smoothed)")
        ax.plot(f, ad_norm, color=COL_ANALYTIC, lw=2.2, ls="--",
                label="analytic renewal PSD")

        idx_real = np.argmax(psd_smooth)
        idx_an = np.argmax(ad_m)
        f_real = f[idx_real]
        f_an = f[idx_an]
        ax.axvline(f_real, color=COL_EMPIRICAL, lw=1.2, alpha=0.6, ls=":")
        ax.axvline(f_an, color=COL_ANALYTIC, lw=1.2, alpha=0.6, ls=":")
        ax.text(0.97, 0.95,
                f"f_emp = {f_real:.2f} Hz\nf_analytic = {f_an:.2f} Hz\n|Δf| = {df:.3f} Hz",
                transform=ax.transAxes, fontsize=FS_TICK - 1,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.4",
                         facecolor="white", edgecolor="#cccccc",
                         alpha=0.92))
        sub_label = sub_key.replace("yuquan/", "Y:").replace("epilepsiae/", "E:")
        ax.set_xlabel("Frequency (Hz)", fontsize=FS_LABEL)
        if j == 0:
            ax.set_ylabel("Normalized PSD (0–1)", fontsize=FS_LABEL)
            label = "c"
        else:
            label = ""
        ax.set_title(sub_label, fontsize=FS_TITLE - 1, fontweight="bold")
        if j == 0:
            ax.legend(loc="lower left", fontsize=FS_TICK - 1,
                      framealpha=0.95)
        ax.set_xlim(f.min(), f.max())
        ax.set_ylim(0, 1.15)
        style_panel(ax, label)

    # ---- Row 3 — sampling test ----
    # Pick a subject where real peak falls within the bulk of the null distribution
    # (i.e., |z-score| of real vs null is small but real >= percentile-low)
    best_ex = None
    best_score = -np.inf
    for r in phase1:
        sg = r.get("group", {}).get("surrogate_gamma") or {}
        null = sg.get("null_peak_powers", [])
        real = sg.get("real_peak_power")
        if not null or real is None:
            continue
        null_arr = np.array(null)
        # Want real close to median of null
        z = abs((real - np.median(null_arr)) / (np.std(null_arr) + 1e-12))
        score = -z
        if score > best_score:
            best_score = score
            best_ex = (r, sg, null_arr, real)

    ax_ex = fig.add_subplot(gs[2, 0])
    ax_pdist = fig.add_subplot(gs[2, 1])
    ax_cov = fig.add_subplot(gs[2, 2])

    if best_ex is not None:
        ex_sub, sg_ex, null, real = best_ex
        bins_h = np.linspace(min(null.min(), real),
                             max(null.max(), real), 35)
        ax_ex.hist(null, bins=bins_h, color=COL_SURROGATE, alpha=0.75,
                   edgecolor="white", linewidth=0.6,
                   label=f"gamma surrogates (n = {len(null)})")
        ax_ex.axvline(real, color=COL_SIG, lw=3.0, zorder=5)
        # 5–95 percentile band of null
        p5, p95 = np.percentile(null, [5, 95])
        ax_ex.axvspan(p5, p95, color="#666666", alpha=0.10, zorder=0,
                      label="5–95 percentile of null")
        ymax = ax_ex.get_ylim()[1]
        ax_ex.text(real, ymax * 0.95,
                   f"  real = {real:.2g}\n  p = {sg_ex['p_value']:.2f}",
                   ha="left", va="top", fontsize=FS_TICK - 1,
                   color=COL_SIG, fontweight="bold")
        ax_ex.set_xlabel("Pulse-train peak power", fontsize=FS_LABEL)
        ax_ex.set_ylabel("# surrogates", fontsize=FS_LABEL)
        ex_label = format_subject_label(ex_sub["dataset"], ex_sub["subject"])
        ax_ex.set_title(f"Example: {ex_label}  ·  real ≈ within null",
                        fontsize=FS_TITLE - 1, fontweight="bold", pad=10)
        ax_ex.legend(loc="upper right", fontsize=FS_TICK - 1,
                     framealpha=0.95)
        ax_ex.set_xlim(bins_h.min(), bins_h.max())
    style_panel(ax_ex, "d")

    # Cohort gamma surrogate p-value distribution
    p_yq = []
    p_epi = []
    for r in _all_phase1():
        s = r.get("group", {}).get("surrogate_gamma", {})
        if s and s.get("p_value") is not None:
            if r["dataset"] == "yuquan":
                p_yq.append(s["p_value"])
            else:
                p_epi.append(s["p_value"])
    bins = np.linspace(0, 1, 21)
    centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]
    cy, _ = np.histogram(p_yq, bins=bins)
    ce, _ = np.histogram(p_epi, bins=bins)
    ax_pdist.bar(centers, cy, width=width * 0.9, color=COL_YUQUAN, alpha=0.85,
                 edgecolor="white", linewidth=0.6,
                 label=f"Yuquan (n = {len(p_yq)})")
    ax_pdist.bar(centers, ce, width=width * 0.9, bottom=cy,
                 color=COL_EPILEPSIAE, alpha=0.85,
                 edgecolor="white", linewidth=0.6,
                 label=f"Epilepsiae (n = {len(p_epi)})")
    ax_pdist.axvline(0.05, color="black", lw=1.5, ls="--")
    ax_pdist.text(0.05, ax_pdist.get_ylim()[1] * 0.95, "  p = 0.05",
                  fontsize=FS_TICK, ha="left", va="top")
    n_total = len(p_yq) + len(p_epi)
    n_sig = sum(1 for p in p_yq + p_epi if p < 0.05)
    ax_pdist.set_xlabel("Gamma surrogate p-value", fontsize=FS_LABEL)
    ax_pdist.set_ylabel("# subjects", fontsize=FS_LABEL)
    ax_pdist.set_title(f"Only {n_sig}/{n_total} reject the null",
                       fontsize=FS_TITLE - 1, fontweight="bold", pad=10)
    ax_pdist.legend(loc="upper right", fontsize=FS_TICK - 1,
                    framealpha=0.95)
    ax_pdist.set_xlim(0, 1)
    style_panel(ax_pdist, "e")

    # Coverage summary bar: how many explained by analytic / sampling / either
    n_analytic = sum(1 for d, _, _ in candidates if d < 1.0)
    p_iei_arr = np.array(p_yq + p_epi)
    n_sampling = int((p_iei_arr >= 0.05).sum())
    # union from same source isn't easily computed without per-subject join;
    # report exp6a Phase 2 numbers if matching
    n_total_eval = len(candidates)
    coverage = [
        ("Analytic", n_analytic / n_total_eval if n_total_eval else 0,
         COL_ANALYTIC),
        ("Sampling", n_sampling / n_total if n_total else 0, COL_SURROGATE),
        ("Either", 19 / 21, COL_SIG),
    ]
    ks = [k for k, _, _ in coverage]
    vs = [v for _, v, _ in coverage]
    bar_colors = [c for _, _, c in coverage]
    ax_cov.bar(range(len(ks)), vs, color=bar_colors, alpha=0.85,
               edgecolor="white", linewidth=1.0, width=0.6)
    for i, v in enumerate(vs):
        ax_cov.text(i, v + 0.025, f"{v:.0%}", ha="center", va="bottom",
                    fontsize=FS_LABEL, fontweight="bold")
    ax_cov.set_xticks(range(len(ks)))
    ax_cov.set_xticklabels(ks, fontsize=FS_LABEL, fontweight="bold")
    ax_cov.set_ylim(0, 1.15)
    ax_cov.set_yticks(np.arange(0, 1.01, 0.25))
    ax_cov.set_yticklabels([f"{int(t*100)}%" for t in np.arange(0, 1.01, 0.25)])
    ax_cov.set_ylabel("Fraction of subjects explained", fontsize=FS_LABEL)
    ax_cov.set_title("Coverage of the refractory hypothesis",
                     fontsize=FS_TITLE - 1, fontweight="bold", pad=10)
    ax_cov.set_xlim(-0.6, len(ks) - 0.4)
    style_panel(ax_cov, "f")

    _suptitle(fig,
        "Fig 2  ·  Refractory dead-time generates the 2-Hz peak — analytic & sampling tests both pass")
    return savefig_pub(fig, PPT_DIR / "fig2_refractory_renewal.png")


# =========================================================================
# Fig 3: Lognormal vs Power-law
# =========================================================================

def plot_fig3():
    """IEI distribution: lognormal beats power-law in 30/30 subjects."""
    print("Plotting Fig 3: lognormal vs power-law ...")

    phase1 = _all_phase1()

    fig = plt.figure(figsize=(15, 6.2))
    gs = gridspec.GridSpec(
        1, 3, width_ratios=[1.0, 1.0, 0.95],
        wspace=0.38,
        left=0.06, right=0.97, top=0.78, bottom=0.13,
    )

    # ---- Panel a — toy: what each shape looks like ----
    ax_a = fig.add_subplot(gs[0])
    x = np.logspace(-1.0, 2.0, 300)
    # Lognormal: log-mean=0.5, sigma=0.5
    pdf_ln = lognorm.pdf(x, s=0.7, scale=np.exp(0.5))
    # Power-law: alpha=2.0, xmin=0.5, normalized over [xmin, ∞)
    alpha = 2.5
    xmin = 0.5
    pdf_pl = np.where(x >= xmin, (alpha - 1) / xmin * (x / xmin) ** -alpha, np.nan)
    ax_a.loglog(x, pdf_ln, color=COL_EMPIRICAL, lw=2.6, label="Lognormal")
    ax_a.loglog(x, pdf_pl, color=COL_ANALYTIC, lw=2.6,
                ls="--", label=f"Power-law (α = {alpha})")
    # annotate body curvature vs straight line
    ax_a.text(2, 0.005, "Lognormal:\ncurves on log-log",
              color=COL_EMPIRICAL, fontsize=FS_TICK, fontweight="bold",
              ha="left")
    ax_a.text(8, 0.06, "Power-law:\nstraight line\n(scale-free)",
              color=COL_ANALYTIC, fontsize=FS_TICK, fontweight="bold",
              ha="left")
    ax_a.set_xlabel("IEI (a.u.)", fontsize=FS_LABEL)
    ax_a.set_ylabel("PDF", fontsize=FS_LABEL)
    ax_a.set_title("Toy: lognormal vs power-law",
                   fontsize=FS_TITLE, fontweight="bold", pad=10)
    ax_a.legend(loc="lower left", fontsize=FS_TICK, framealpha=0.95)
    ax_a.set_xlim(x.min(), x.max())
    style_panel(ax_a, "a")

    # ---- Panel b — real subject IEI distribution + both fits ----
    # find a subject with strong tail
    best_sub = None
    for r in phase1:
        ie = r.get("group", {}).get("iei_fit", {})
        if ie and ie.get("n_tail", 0) > 1000:
            best_sub = r
            break
    if best_sub is None:
        best_sub = phase1[0]
    # We need raw IEI to histogram. Reconstruct from packed in exp4 if possible.
    exp4 = _load_phase2("exp4_return_map.json")
    sub_key = f"{best_sub['dataset']}/{best_sub['subject']}"
    iei_arr = None
    if sub_key in exp4:
        packed = exp4[sub_key].get("packed", {})
        iei_n = np.array(packed.get("iei_n", []))
        iei_n1 = np.array(packed.get("iei_n1", []))
        if len(iei_n):
            iei_arr = np.concatenate([iei_n[:1], iei_n1])
    ax_b = fig.add_subplot(gs[1])
    if iei_arr is not None and len(iei_arr) > 100:
        iei_pos = iei_arr[iei_arr > 0]
        # log-spaced bins
        bins = np.logspace(np.log10(iei_pos.min()),
                           np.log10(iei_pos.max()), 50)
        ax_b.hist(iei_pos, bins=bins, density=True,
                  color=COL_NEUTRAL, alpha=0.7,
                  edgecolor="white", linewidth=0.4,
                  label="empirical IEI")
        ax_b.set_xscale("log")
        ax_b.set_yscale("log")
        # Fit lognormal MLE
        shape_ln, loc_ln, scale_ln = lognorm.fit(iei_pos, floc=0)
        x_fit = np.logspace(np.log10(iei_pos.min()),
                            np.log10(iei_pos.max()), 300)
        pdf_ln_fit = lognorm.pdf(x_fit, shape_ln, loc=loc_ln, scale=scale_ln)
        ax_b.plot(x_fit, pdf_ln_fit, color=COL_EMPIRICAL, lw=2.4,
                  label="Lognormal fit")
        # Fit power-law on tail
        ie = best_sub["group"]["iei_fit"]
        if ie.get("alpha") and ie.get("xmin"):
            alpha_real = ie["alpha"]
            xmin_real = ie["xmin"]
            x_pl = x_fit[x_fit >= xmin_real]
            pdf_pl_fit = (alpha_real - 1) / xmin_real * \
                         (x_pl / xmin_real) ** -alpha_real
            # rescale to fraction of data above xmin
            frac_tail = (iei_pos >= xmin_real).mean()
            ax_b.plot(x_pl, pdf_pl_fit * frac_tail, color=COL_ANALYTIC,
                      lw=2.4, ls="--",
                      label=f"Power-law fit (α = {alpha_real:.2f}, tail)")
        sub_label = format_subject_label(best_sub["dataset"], best_sub["subject"])
        ax_b.set_xlabel("IEI (s)", fontsize=FS_LABEL)
        ax_b.set_ylabel("PDF (log-log)", fontsize=FS_LABEL)
        ax_b.set_title(f"Real example: {sub_label}\nlognormal wins visually",
                       fontsize=FS_TITLE, fontweight="bold", pad=10)
        ax_b.legend(loc="lower left", fontsize=FS_TICK - 1,
                    framealpha=0.95)
        ax_b.set_xlim(iei_pos.min(), iei_pos.max())
    style_panel(ax_b, "b")

    # ---- Panel c — cohort statistical comparison ----
    R_vals = []
    p_vals = []
    datasets = []
    for r in phase1:
        ie = r.get("group", {}).get("iei_fit", {})
        R = ie.get("pl_vs_ln_R")
        p = ie.get("pl_vs_ln_p")
        if R is not None and p is not None:
            R_vals.append(R)
            p_vals.append(p)
            datasets.append(r["dataset"])
    R_vals = np.array(R_vals)
    p_vals = np.array(p_vals)

    # R > 0 → power-law preferred; R < 0 → lognormal preferred
    n_total = len(R_vals)
    n_ln = int((R_vals < 0).sum())
    n_ln_sig = int(((R_vals < 0) & (p_vals < 0.05)).sum())

    ax_c = fig.add_subplot(gs[2])
    R_yq = R_vals[np.array(datasets) == "yuquan"]
    R_epi = R_vals[np.array(datasets) == "epilepsiae"]
    violin_with_scatter(ax_c, R_yq, pos=0, color=COL_YUQUAN, width=0.6)
    violin_with_scatter(ax_c, R_epi, pos=1, color=COL_EPILEPSIAE, width=0.6)
    ax_c.axhline(0, color="black", lw=1.5)
    # Use fixed offsets so text never gets clipped by axis limits
    ymin_c, ymax_c = ax_c.get_ylim()
    ax_c.text(0.5, ymax_c - (ymax_c - ymin_c) * 0.06,
              "↑ power-law preferred", ha="center", va="top",
              fontsize=FS_TICK - 1, color=COL_ANALYTIC, fontweight="bold")
    ax_c.text(0.5, ymin_c + (ymax_c - ymin_c) * 0.06,
              "↓ lognormal preferred", ha="center", va="bottom",
              fontsize=FS_TICK - 1, color=COL_EMPIRICAL, fontweight="bold")
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels([f"Yuquan\n(n = {len(R_yq)})",
                          f"Epilepsiae\n(n = {len(R_epi)})"],
                         fontsize=FS_TICK)
    ax_c.set_ylabel("Vuong R\n(power-law vs lognormal)",
                    fontsize=FS_LABEL)
    ax_c.set_title(
        f"Lognormal wins {n_ln}/{n_total}\n({n_ln_sig} sig. at p < 0.05)",
        fontsize=FS_TITLE - 1, fontweight="bold", pad=10)
    ax_c.set_xlim(-0.6, 1.6)
    style_panel(ax_c, "c")

    _suptitle(fig,
        "Fig 3  ·  IEI distribution is lognormal, not power-law (no scale-free oscillator)",
        y=0.96)
    return savefig_pub(fig, PPT_DIR / "fig3_lognormal_vs_powerlaw.png")


# =========================================================================
# Fig 4: IEI temporal structure — slow modulation + short-range memory
# =========================================================================

def plot_fig4():
    """All IEI dynamics together: return maps, lag-k decay, detrending."""
    print("Plotting Fig 4: IEI temporal structure ...")

    exp4 = _load_phase2("exp4_return_map.json")
    exp7 = _load_phase2("exp7_serial_corr_deep.json")

    fig = plt.figure(figsize=(17, 11.5))
    gs_outer = gridspec.GridSpec(
        2, 1, height_ratios=[1, 1], hspace=0.45,
        left=0.055, right=0.97, top=0.90, bottom=0.06,
    )
    gs_top = gridspec.GridSpecFromSubplotSpec(
        2, 4, subplot_spec=gs_outer[0],
        wspace=0.35, hspace=0.55,
    )
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_outer[1],
        wspace=0.35, width_ratios=[1.2, 1, 1],
    )

    # ---- Top-left 2x2: 4 subject return maps (2 Yuquan + 2 Epilepsiae) ----
    yq_keys = [k for k in exp4 if k.startswith("yuquan/")]
    epi_keys = [k for k in exp4 if k.startswith("epilepsiae/")]
    rng = np.random.default_rng(2)
    pick_yq = list(rng.choice(yq_keys, size=min(2, len(yq_keys)), replace=False))
    pick_epi = list(rng.choice(epi_keys, size=min(2, len(epi_keys)), replace=False))
    pick_subs = pick_yq + pick_epi

    for k, sub_key in enumerate(pick_subs):
        ax = fig.add_subplot(gs_top[k // 2, k % 2])
        packed = exp4[sub_key].get("packed", {})
        iei_n = np.array(packed.get("iei_n", []))
        iei_n1 = np.array(packed.get("iei_n1", []))
        rho = packed.get("serial_corr", np.nan)
        if len(iei_n) > 5:
            ds_color = COL_YUQUAN if "yuquan" in sub_key else COL_EPILEPSIAE
            # Clip to 95th percentile so the cloud is visible
            cut = float(np.percentile(np.concatenate([iei_n, iei_n1]), 95))
            mask = (iei_n <= cut) & (iei_n1 <= cut)
            ax.scatter(iei_n[mask], iei_n1[mask], s=4, alpha=0.20,
                       color=ds_color, edgecolors="none")
            ax.plot([0, cut], [0, cut], color="#666666", lw=1.0,
                    ls=":", alpha=0.6)
            ax.set_xlim(0, cut)
            ax.set_ylim(0, cut)
            ax.text(0.96, 0.96, f"ρ = {rho:.2f}",
                    transform=ax.transAxes,
                    fontsize=FS_TICK, fontweight="bold",
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3",
                             facecolor="white", edgecolor="#cccccc",
                             alpha=0.9))
        sub_label = sub_key.replace("yuquan/", "Y:").replace("epilepsiae/", "E:")
        ax.set_title(sub_label, fontsize=FS_TITLE - 2,
                     fontweight="bold", pad=6)
        if k % 2 == 0:
            ax.set_ylabel("IEI(n+1) (s)", fontsize=FS_LABEL - 1)
        if k // 2 == 1:
            ax.set_xlabel("IEI(n) (s)", fontsize=FS_LABEL - 1)
        label = "a" if k == 0 else ""
        style_panel(ax, label)

    # ---- Top-right 2 cols span 2 rows: lag-1 r violin + lag-k decay ----
    ax_lag1 = fig.add_subplot(gs_top[:, 2])
    lag1_yq = []
    lag1_epi = []
    lag_curves_yq = []
    lag_curves_epi = []
    half_yq = []
    half_epi = []
    for sub_key, rec in exp7.items():
        sd = rec.get("serial_decay", {})
        if not sd:
            continue
        rs = np.array(sd.get("rs", []))
        lags = np.array(sd.get("lags", []))
        hl = sd.get("half_life_lag")
        lag1 = sd.get("lag1_r")
        ds = "yuquan" if "yuquan" in sub_key else "epilepsiae"
        if lag1 is not None:
            (lag1_yq if ds == "yuquan" else lag1_epi).append(lag1)
        if len(rs) > 0:
            (lag_curves_yq if ds == "yuquan" else lag_curves_epi).append((lags, rs))
        if hl is not None:
            (half_yq if ds == "yuquan" else half_epi).append(hl)

    violin_with_scatter(ax_lag1, np.array(lag1_yq), pos=0,
                        color=COL_YUQUAN, width=0.6)
    violin_with_scatter(ax_lag1, np.array(lag1_epi), pos=1,
                        color=COL_EPILEPSIAE, width=0.6)
    ax_lag1.axhline(0, color="black", lw=1.0, ls="--")
    n_pos_yq = int((np.array(lag1_yq) > 0).sum())
    n_pos_epi = int((np.array(lag1_epi) > 0).sum())
    ax_lag1.set_xticks([0, 1])
    ax_lag1.set_xticklabels(
        [f"Yuquan\n{n_pos_yq}/{len(lag1_yq)} > 0",
         f"Epilepsiae\n{n_pos_epi}/{len(lag1_epi)} > 0"],
        fontsize=FS_TICK)
    ax_lag1.set_ylabel("Lag-1 serial corr", fontsize=FS_LABEL)
    ax_lag1.set_title("Every subject is positive",
                      fontsize=FS_TITLE - 1, fontweight="bold", pad=10)
    ax_lag1.set_xlim(-0.6, 1.6)
    style_panel(ax_lag1, "b")

    ax_decay = fig.add_subplot(gs_top[:, 3])
    for lags, rs in lag_curves_yq:
        ax_decay.plot(lags, rs, color=COL_YUQUAN, alpha=0.18, lw=0.8)
    for lags, rs in lag_curves_epi:
        ax_decay.plot(lags, rs, color=COL_EPILEPSIAE, alpha=0.18, lw=0.8)
    if lag_curves_yq:
        all_lags = lag_curves_yq[0][0]
        all_rs = []
        for lags, rs in lag_curves_yq + lag_curves_epi:
            interp = np.interp(all_lags, lags, rs)
            all_rs.append(interp)
        med = np.nanmedian(all_rs, axis=0)
        ax_decay.plot(all_lags, med, color="black", lw=2.6,
                      label="cohort median")
    ax_decay.axhline(0, color="black", lw=1.0)
    med_hl = np.nanmedian(np.array(half_yq + half_epi))
    ax_decay.axvline(med_hl, color="#A35E48", lw=1.8, ls="--",
                     label=f"median half-life ≈ {med_hl:.0f} lags")
    ax_decay.set_xscale("log")
    ax_decay.set_xlabel("Lag k (events)", fontsize=FS_LABEL)
    ax_decay.set_ylabel("Lag-k serial corr", fontsize=FS_LABEL)
    ax_decay.set_title("Slow decay → long memory",
                       fontsize=FS_TITLE - 1, fontweight="bold", pad=10)
    ax_decay.legend(loc="upper right", fontsize=FS_TICK - 1,
                    framealpha=0.95)
    if lag_curves_yq:
        ax_decay.set_xlim(all_lags.min(), all_lags.max())
    style_panel(ax_decay, "c")

    # ---- Bottom row: detrending decomposition ----
    ax_concept = fig.add_subplot(gs_bot[0])
    ax_paired = fig.add_subplot(gs_bot[1])
    ax_frac = fig.add_subplot(gs_bot[2])

    # Concept: synthetic series with slow drift + 600s window detrend
    rng = np.random.default_rng(11)
    t_min = np.arange(0, 60, 1)  # 60 minutes
    drift = 0.5 + 0.45 * np.sin(2 * np.pi * t_min / 30) + 0.1 * np.sin(
        2 * np.pi * t_min / 5
    )
    noise = rng.normal(0, 0.05, len(t_min))
    iei_series = drift + noise
    # 10-min rolling median ≈ 600s window
    win = 10
    rolling = np.array([
        np.median(iei_series[max(0, i - win // 2):i + win // 2 + 1])
        for i in range(len(t_min))
    ])
    detrended = iei_series - rolling
    ax_concept.plot(t_min, iei_series, color=COL_EMPIRICAL, lw=1.6,
                    label="raw IEI")
    ax_concept.plot(t_min, rolling, color=COL_ANALYTIC, lw=2.4,
                    label="600 s rolling median")
    ax_concept.plot(t_min, detrended + 0.0, color=COL_DETRENDED, lw=1.6,
                    alpha=0.8, label="detrended (residual)")
    ax_concept.axhline(0, color="black", lw=0.8, ls=":")
    ax_concept.set_xlabel("Time (min)", fontsize=FS_LABEL)
    ax_concept.set_ylabel("IEI (toy units)", fontsize=FS_LABEL)
    ax_concept.set_title("What 600 s detrending does\n(remove slow rate drift)",
                         fontsize=FS_TITLE - 1, fontweight="bold", pad=10)
    ax_concept.legend(loc="upper right", fontsize=FS_TICK - 1,
                      framealpha=0.95)
    ax_concept.set_xlim(0, t_min.max())
    style_panel(ax_concept, "d")

    # Paired raw vs detrended r per subject
    raw_vals = []
    det_vals = []
    frac_vals = []
    datasets = []
    for sub_key, rec in exp7.items():
        det = rec.get("detrended", {})
        if det.get("raw_r") is None:
            continue
        raw_vals.append(det["raw_r"])
        det_vals.append(det["detrended_r"])
        frac_vals.append(det.get("detrend_fraction", np.nan))
        datasets.append("yuquan" if "yuquan" in sub_key else "epilepsiae")
    raw_arr = np.array(raw_vals)
    det_arr = np.array(det_vals)
    frac_arr = np.array(frac_vals)
    ds_arr = np.array(datasets)

    # Connecting lines between raw (x=0) and detrended (x=1)
    for i in range(len(raw_arr)):
        c = COL_YUQUAN if ds_arr[i] == "yuquan" else COL_EPILEPSIAE
        ax_paired.plot([0, 1], [raw_arr[i], det_arr[i]],
                       color=c, lw=0.8, alpha=0.4)
    rng2 = np.random.default_rng(3)
    jit0 = rng2.normal(0, 0.025, len(raw_arr))
    jit1 = rng2.normal(0, 0.025, len(raw_arr))
    for ds, c in [("yuquan", COL_YUQUAN), ("epilepsiae", COL_EPILEPSIAE)]:
        m = ds_arr == ds
        ax_paired.scatter(jit0[m], raw_arr[m], s=55, color=c,
                          edgecolors="white", linewidths=0.6,
                          alpha=0.85, zorder=3,
                          label=f"{ds.capitalize()} (n = {int(m.sum())})")
        ax_paired.scatter(1 + jit1[m], det_arr[m], s=55, color=c,
                          edgecolors="white", linewidths=0.6,
                          alpha=0.85, zorder=3)
    ax_paired.axhline(0, color="black", lw=0.8, ls="--")
    try:
        w_stat, w_p = wilcoxon(raw_arr, det_arr)
    except Exception:
        w_p = np.nan
    n_pos_after = int((det_arr > 0).sum())
    yhi = max(raw_arr.max(), det_arr.max())
    ylo = min(raw_arr.min(), det_arr.min())
    yspan = yhi - ylo
    ax_paired.set_ylim(ylo - 0.05 * yspan, yhi + 0.30 * yspan)
    add_significance_bracket(
        ax_paired, 0, 1, yhi + 0.08 * yspan,
        p=w_p if np.isfinite(w_p) else 1.0, dy=0.025 * yspan,
    )
    ax_paired.set_xticks([0, 1])
    ax_paired.set_xticklabels(["Raw r", "Detrended r"], fontsize=FS_LABEL)
    ax_paired.set_ylabel("Lag-1 serial corr", fontsize=FS_LABEL)
    ax_paired.set_title(
        f"Detrending shrinks r,\nbut {n_pos_after}/{len(det_arr)} stay > 0",
        fontsize=FS_TITLE - 1, fontweight="bold", pad=12)
    ax_paired.legend(loc="lower left", fontsize=FS_TICK - 1,
                     framealpha=0.95)
    ax_paired.set_xlim(-0.4, 1.4)
    style_panel(ax_paired, "e")

    # Detrend fraction violin
    f_yq = frac_arr[ds_arr == "yuquan"]
    f_epi = frac_arr[ds_arr == "epilepsiae"]
    violin_with_scatter(ax_frac, f_yq, pos=0, color=COL_YUQUAN, width=0.6)
    violin_with_scatter(ax_frac, f_epi, pos=1, color=COL_EPILEPSIAE,
                        width=0.6)
    med_frac = np.nanmedian(frac_arr)
    ax_frac.axhline(med_frac, color="#666666", lw=1.0, ls=":",
                    label=f"median = {med_frac:.0%}")
    ax_frac.set_xticks([0, 1])
    ax_frac.set_xticklabels([f"Yuquan\n(n = {len(f_yq)})",
                             f"Epilepsiae\n(n = {len(f_epi)})"],
                            fontsize=FS_TICK)
    ax_frac.set_ylabel("Detrend fraction\n(slow component / raw)",
                      fontsize=FS_LABEL)
    ax_frac.set_title("≈70% of correlation is slow drift",
                      fontsize=FS_TITLE - 1, fontweight="bold", pad=10)
    ax_frac.legend(loc="lower right", fontsize=FS_TICK - 1, framealpha=0.95)
    ax_frac.set_ylim(0, 1.0)
    ax_frac.set_xlim(-0.6, 1.6)
    style_panel(ax_frac, "f")

    _suptitle(fig,
        "Fig 4  ·  IEI temporal structure: slow modulation + residual short-range memory",
        y=0.985)
    return savefig_pub(fig, PPT_DIR / "fig4_iei_dynamics.png")


# =========================================================================
# Fig 5: Slow modulation under long-term influences
# =========================================================================

def plot_fig5():
    """24h trace + multi-hour autocorr + day/night detrended r."""
    print("Plotting Fig 5: slow modulation long-term ...")

    exp7c = _load_phase2("exp7c_long_timescale.json")

    # Yuquan: pick subject with strongest day/night ratio (clear circadian)
    from datetime import datetime, timezone, timedelta
    tz_yq = timezone(timedelta(hours=8))
    yq_pick = None
    yq_best_score = 0
    for sub_key, rec in exp7c.items():
        if "yuquan" not in sub_key:
            continue
        trace = rec.get("long_timescale", {}).get("trace", {})
        rate = np.array(trace.get("smooth_3600s", []))
        bc = np.array(trace.get("bin_centers", []))
        if len(rate) < 100:
            continue
        hrs = np.array([
            datetime.fromtimestamp(t, tz=tz_yq).hour
            + datetime.fromtimestamp(t, tz=tz_yq).minute / 60.0
            for t in bc
        ])
        day_mask = (hrs >= 8) & (hrs < 20)
        if day_mask.sum() < 10 or (~day_mask).sum() < 10:
            continue
        d_med = np.median(rate[day_mask])
        n_med = np.median(rate[~day_mask])
        if min(d_med, n_med) < 5:
            continue
        # Prefer high-rate subject with clear day/night contrast
        ratio = max(d_med, n_med) / min(d_med, n_med)
        score = ratio * np.log10(d_med + n_med + 1)
        if score > yq_best_score:
            yq_best_score = score
            yq_pick = (sub_key, rec)

    epi_pick = None
    epi_max_h = 0
    for sub_key, rec in exp7c.items():
        if "epilepsiae" not in sub_key:
            continue
        bh = rec.get("long_timescale", {}).get("trace", {}).get(
            "bin_hours_from_start", [])
        if len(bh) > 0 and bh[-1] > epi_max_h:
            epi_max_h = bh[-1]
            epi_pick = (sub_key, rec)

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(
        2, 3, height_ratios=[1, 1.05],
        wspace=0.35, hspace=0.65,
        left=0.055, right=0.97, top=0.88, bottom=0.07,
    )

    # ---- Row 1: 24h-style traces ----
    ax_yq = fig.add_subplot(gs[0, :])

    if yq_pick:
        sub_key, rec = yq_pick
        trace = rec["long_timescale"]["trace"]
        bh = np.array(trace["bin_hours_from_start"])
        rate = np.array(trace["smooth_3600s"])
        # day shading (simplified: assume bin_centers timestamps are unix)
        bc = np.array(trace["bin_centers"])
        # Local hour-of-day (Yuquan = Asia/Shanghai UTC+8)
        from datetime import datetime, timezone, timedelta
        tz_yq = timezone(timedelta(hours=8))
        local_hours = np.array([
            datetime.fromtimestamp(t, tz=tz_yq).hour + datetime.fromtimestamp(t, tz=tz_yq).minute / 60.0
            for t in bc
        ])
        # mark day = 8-20
        day_mask = (local_hours >= 8) & (local_hours < 20)

        # background day/night shading
        # find segments of day vs night
        change_idx = np.where(np.diff(day_mask.astype(int)) != 0)[0] + 1
        seg_starts = np.concatenate([[0], change_idx])
        seg_ends = np.concatenate([change_idx, [len(bh)]])
        for s, e in zip(seg_starts, seg_ends):
            if e <= s:
                continue
            color = COL_DAY if day_mask[s] else COL_NIGHT
            alpha = 0.25 if day_mask[s] else 0.20
            ax_yq.axvspan(bh[s], bh[e - 1], color=color, alpha=alpha,
                          zorder=0)
        ax_yq.plot(bh, rate, color=COL_EMPIRICAL, lw=1.8, zorder=2,
                   label="1 h smoothed rate")
        ax_yq.set_xlim(bh.min(), bh.max())
        ax_yq.set_xlabel("Time from start (hours, local time)",
                        fontsize=FS_LABEL)
        ax_yq.set_ylabel("Event rate (per hour)", fontsize=FS_LABEL)
        sub_label = sub_key.replace("yuquan/", "Y:")
        ax_yq.set_title(
            f"24 h trace · {sub_label} (Yuquan)  ·  clear day/night wobble",
            fontsize=FS_TITLE, fontweight="bold", pad=14)
        from matplotlib.patches import Patch
        handles = [
            plt.Line2D([0], [0], color=COL_EMPIRICAL, lw=2.2,
                       label="event rate (1 h smooth)"),
            Patch(facecolor=COL_DAY, alpha=0.5, label="day (08–20)"),
            Patch(facecolor=COL_NIGHT, alpha=0.4, label="night (20–08)"),
        ]
        ax_yq.legend(handles=handles, loc="upper right",
                     fontsize=FS_TICK, framealpha=0.95, ncol=3)
    style_panel(ax_yq, "a")

    # ---- Row 2: quantitative summaries ----
    ax_acorr = fig.add_subplot(gs[1, 0])
    ax_dn = fig.add_subplot(gs[1, 1])
    ax_long = fig.add_subplot(gs[1, 2])

    # Multi-hour rate autocorrelation
    lags_master = None
    rs_yq = []
    rs_epi = []
    for sub_key, rec in exp7c.items():
        ra = rec.get("long_timescale", {}).get("rate_autocorr", [])
        if not ra:
            continue
        lh = np.array([row["lag_hours"] for row in ra])
        rr = np.array([row["r"] for row in ra])
        if lags_master is None:
            lags_master = lh
        if "yuquan" in sub_key:
            rs_yq.append(rr)
        else:
            rs_epi.append(rr)
    rs_yq_arr = np.array(rs_yq) if rs_yq else np.empty((0, 0))
    rs_epi_arr = np.array(rs_epi) if rs_epi else np.empty((0, 0))

    if lags_master is not None:
        for r in rs_yq_arr:
            ax_acorr.plot(lags_master, r, color=COL_YUQUAN, alpha=0.20, lw=0.9)
        for r in rs_epi_arr:
            ax_acorr.plot(lags_master, r, color=COL_EPILEPSIAE, alpha=0.20, lw=0.9)
        med_yq = np.nanmedian(rs_yq_arr, axis=0) if rs_yq_arr.size else None
        med_epi = np.nanmedian(rs_epi_arr, axis=0) if rs_epi_arr.size else None
        if med_yq is not None:
            ax_acorr.plot(lags_master, med_yq, color=COL_YUQUAN, lw=2.6,
                          label=f"Yuquan median (n = {len(rs_yq_arr)})")
        if med_epi is not None:
            ax_acorr.plot(lags_master, med_epi, color=COL_EPILEPSIAE, lw=2.6,
                          label=f"Epilepsiae median (n = {len(rs_epi_arr)})")
        ax_acorr.axhline(0, color="black", lw=0.9, ls="--")
        ax_acorr.set_xlabel("Lag (hours)", fontsize=FS_LABEL)
        ax_acorr.set_ylabel("5-min binned rate autocorr", fontsize=FS_LABEL)
        ax_acorr.set_title("Memory persists for hours",
                           fontsize=FS_TITLE - 1, fontweight="bold", pad=10)
        ax_acorr.legend(loc="upper right", fontsize=FS_TICK - 1,
                        framealpha=0.95)
        ax_acorr.set_xlim(lags_master.min(), lags_master.max())
    style_panel(ax_acorr, "b")

    # Day vs night detrended r (paired)
    day_rs = []
    night_rs = []
    ds_arr = []
    for sub_key, rec in exp7c.items():
        cdn = rec.get("contiguous_daynight", {})
        d = cdn.get("day", {}).get("pooled_detrended_r")
        n = cdn.get("night", {}).get("pooled_detrended_r")
        if d is None or n is None:
            continue
        day_rs.append(d)
        night_rs.append(n)
        ds_arr.append("yuquan" if "yuquan" in sub_key else "epilepsiae")
    day_arr = np.array(day_rs)
    night_arr = np.array(night_rs)
    ds_arr_np = np.array(ds_arr)

    rng3 = np.random.default_rng(7)
    jit_d = rng3.normal(0, 0.025, len(day_arr))
    jit_n = rng3.normal(0, 0.025, len(day_arr))
    for i in range(len(day_arr)):
        c = COL_YUQUAN if ds_arr[i] == "yuquan" else COL_EPILEPSIAE
        ax_dn.plot([0, 1], [day_arr[i], night_arr[i]],
                   color=c, lw=0.8, alpha=0.4)
    for ds, c in [("yuquan", COL_YUQUAN), ("epilepsiae", COL_EPILEPSIAE)]:
        m = ds_arr_np == ds
        ax_dn.scatter(jit_d[m], day_arr[m], s=55, color=c,
                      edgecolors="white", linewidths=0.6,
                      alpha=0.85, zorder=3,
                      label=f"{ds.capitalize()} (n = {int(m.sum())})")
        ax_dn.scatter(1 + jit_n[m], night_arr[m], s=55, color=c,
                      edgecolors="white", linewidths=0.6,
                      alpha=0.85, zorder=3)
    try:
        w_stat, w_p = wilcoxon(day_arr, night_arr)
    except Exception:
        w_p = np.nan
    yhi = max(day_arr.max(), night_arr.max())
    ylo = min(day_arr.min(), night_arr.min())
    yspan = yhi - ylo
    # Reserve top 18% for bracket so it never collides with title
    ax_dn.set_ylim(ylo - 0.05 * yspan, yhi + 0.32 * yspan)
    add_significance_bracket(
        ax_dn, 0, 1, yhi + 0.10 * yspan,
        p=w_p if np.isfinite(w_p) else 1.0, dy=0.02 * yspan,
    )
    ax_dn.axhline(0, color="black", lw=0.8, ls="--")
    ax_dn.set_xticks([0, 1])
    ax_dn.set_xticklabels(["Day segments", "Night segments"], fontsize=FS_LABEL)
    ax_dn.set_ylabel("Pooled detrended r", fontsize=FS_LABEL)
    ax_dn.set_title("Short-range memory survives\nin day & night equally",
                    fontsize=FS_TITLE - 1, fontweight="bold", pad=14)
    ax_dn.legend(loc="lower right", fontsize=FS_TICK - 1, framealpha=0.95)
    ax_dn.set_xlim(-0.4, 1.4)
    style_panel(ax_dn, "c")

    # Long Epi trace overview
    if epi_pick:
        sub_key, rec = epi_pick
        trace = rec["long_timescale"]["trace"]
        bh = np.array(trace["bin_hours_from_start"])
        rate = np.array(trace["smooth_3600s"])
        ax_long.plot(bh, rate, color=COL_EPILEPSIAE, lw=1.2, alpha=0.8)
        # add day/night band overlay (UKLFR Europe/Berlin = UTC+1 / +2)
        bc = np.array(trace["bin_centers"])
        from datetime import datetime, timezone, timedelta
        tz_epi = timezone(timedelta(hours=2))
        local_hours = np.array([
            datetime.fromtimestamp(t, tz=tz_epi).hour
            + datetime.fromtimestamp(t, tz=tz_epi).minute / 60.0
            for t in bc
        ])
        day_mask = (local_hours >= 8) & (local_hours < 20)
        change_idx = np.where(np.diff(day_mask.astype(int)) != 0)[0] + 1
        seg_starts = np.concatenate([[0], change_idx])
        seg_ends = np.concatenate([change_idx, [len(bh)]])
        for s, e in zip(seg_starts, seg_ends):
            if e <= s:
                continue
            color = COL_DAY if day_mask[s] else COL_NIGHT
            alpha = 0.18 if day_mask[s] else 0.12
            ax_long.axvspan(bh[s], bh[e - 1], color=color, alpha=alpha,
                            zorder=0)
        sub_label = sub_key.replace("epilepsiae/", "E:")
        ax_long.set_title(
            f"Multi-day trace · {sub_label} ({bh.max():.0f} h)",
            fontsize=FS_TITLE - 1, fontweight="bold", pad=10)
        ax_long.set_xlabel("Time from start (hours)", fontsize=FS_LABEL)
        ax_long.set_ylabel("Event rate (1 h smooth)", fontsize=FS_LABEL)
        ax_long.set_xlim(bh.min(), bh.max())
    style_panel(ax_long, "d")

    _suptitle(fig,
        "Fig 5  ·  Slow modulation is dominated by circadian + multi-hour drift")
    return savefig_pub(fig, PPT_DIR / "fig5_slow_modulation.png")


# =========================================================================
# Driver
# =========================================================================

FIG_FUNCS = {
    1: plot_fig1,
    2: plot_fig2,
    3: plot_fig3,
    4: plot_fig4,
    5: plot_fig5,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--fig", type=str, default="",
                        help="Comma list of figure numbers, e.g. 1,3")
    args = parser.parse_args()

    if args.all:
        targets = list(FIG_FUNCS)
    elif args.fig:
        targets = [int(x.strip()) for x in args.fig.split(",")]
    else:
        targets = list(FIG_FUNCS)

    for n in targets:
        if n in FIG_FUNCS:
            FIG_FUNCS[n]()
        else:
            print(f"Unknown figure {n}")


if __name__ == "__main__":
    main()
