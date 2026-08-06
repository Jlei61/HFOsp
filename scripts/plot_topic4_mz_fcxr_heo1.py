"""FCXR-HEO1 diagnostic figures (parameter-scan diagnostics — NOT the paper Figure 5).

  1 high_energy_branch_map.png       — workpoint gate + screen grid labelled by verdict (not just rate).
  2 candidate_virtual_seeg_spectral  — one continuous run: contact stack, spectrogram, six-band energy,
                                        population rate, plateau window (shows it is not a discrete IED train).
  3 candidate_spatial_modes          — baseline-IED / early-high / plateau broadband energy on the real
                                        E1146 contact geometry + coverage + IPR.
Consumes the screen artifacts under high_energy_oscillatory_branch/. dpi=150 diagnostic.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from topic4_mz_fcxr_heo1 import (  # noqa: E402
    decimate_to_work, band_power_spectrogram, build_baseline_reference, classify_heo, BANDS,
    band_db_field, oscillation_probe, Z_GATE)

OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay",
                   "high_energy_oscillatory_branch")
FIG = os.path.join(OUT, "figures")
SHAFT_COLS = {"SCL": "#e8743b", "ICL": "#1f9e9e"}
CAT_COLS = dict(HEO="#2ca02c", plateau_no_osc="#dd8452", platform_transient="#e8c547",
                subthreshold="#9aa0a6", runaway="#c44e52", unsafe="#7a1f1f", nores="#d9dbe0")
FOOTER = "FCXR-HEO1 parameter-scan diagnostic — not the paper Figure 5"


def _shaft(n):
    import re
    m = re.match(r"[A-Za-z]+", str(n))
    return m.group(0) if m else str(n)


def _cat(r):
    if r.get("numerical_unsafe"):
        return "unsafe"
    if r.get("runaway_early_stop_ms") is not None:
        return "runaway"
    if r.get("HEO_BRANCH"):
        return "HEO"
    if r.get("gate_A_plateau") and r.get("gate_C_platform") and not r.get("gate_D_oscillation"):
        return "plateau_no_osc"
    if r.get("gate_C_platform"):
        return "platform_transient"
    return "subthreshold"


# ----------------------------------------------------------------- figure 1: branch map
def fig_branch_map():
    bm = json.load(open(os.path.join(OUT, "branch_map.json")))
    wp = bm["workpoint"]; cells = bm["cells"]
    arms = sorted({(r["gate_quantile"], r["A_c"]) for r in wp})
    fig, (axw, axs) = plt.subplots(1, 2, figsize=(13.0, 5.0), gridspec_kw=dict(width_ratios=[1, 2.4]))

    # workpoint panel
    for r in wp:
        i = arms.index((r["gate_quantile"], r["A_c"]))
        col = "#2ca02c" if r["preserved"] else "#c44e52"
        axw.scatter(0, i, s=340, c=col, edgecolors="k", lw=0.8, zorder=3)
        axw.annotate(r["workpoint_label"].replace("_WORKPOINT", "").replace("_", " ").lower(),
                     (0.16, i), fontsize=7.0, va="center")
    axw.set_yticks(range(len(arms)))
    axw.set_yticklabels([f"gate Q{q:g}\nA_c={A:g}" for q, A in arms], fontsize=8)
    axw.set_xticks([]); axw.set_xlim(-0.35, 1.1); axw.set_ylim(-0.6, len(arms) - 0.4)
    axw.set_title("workpoint gate (D=0, no-kick)\ngreen=INTERICTAL preserved / red=broken", fontsize=9.5)

    # screen grid panel
    Ds = sorted({r["D"] for r in cells}) or [0.13, 0.15]
    ics = ["nokick", "kick3", "kick12"]
    cols = [(D, ic) for D in Ds for ic in ics]
    by = {(r["gate_quantile"], r["A_c"], r["D"], r["ic"]): r for r in cells}
    for yi, (q, A) in enumerate(arms):
        for xi, (D, ic) in enumerate(cols):
            r = by.get((q, A, D, ic))
            cat = _cat(r) if r else "nores"
            axs.add_patch(plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1, facecolor=CAT_COLS[cat],
                          edgecolor="white", lw=1.2))
            if r and r.get("HEO_BRANCH"):
                axs.plot(xi, yi, marker="*", ms=15, c="k", zorder=4)
            if r:
                axs.annotate(f"{r['mean_rate_hz']:.0f}", (xi, yi + 0.30), ha="center", fontsize=6.2,
                             color="k", zorder=5)
    axs.set_xticks(range(len(cols)))
    axs.set_xticklabels([f"D={D:g}\n{ic}" for D, ic in cols], fontsize=7.5)
    axs.set_yticks(range(len(arms)))
    axs.set_yticklabels([f"Q{q:g} A_c={A:g}" for q, A in arms], fontsize=8)
    axs.set_xlim(-0.5, len(cols) - 0.5); axs.set_ylim(-0.5, len(arms) - 0.5)
    axs.set_title(f"cooperative screen  (n_HEO={bm.get('n_heo', 0)}/{bm.get('n_cells', 0)}; "
                  f"star=HEO, number=mean rate Hz)", fontsize=9.5)
    handles = [Patch(facecolor=CAT_COLS[k], edgecolor="white", label=k.replace("_", " "))
               for k in ("HEO", "plateau_no_osc", "platform_transient", "subthreshold", "runaway", "unsafe")]
    axs.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=7.6, frameon=False)
    fig.text(0.5, 0.005, FOOTER, ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(FIG, "high_energy_branch_map.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- helpers for candidate figures
def _load_cell_trace(label, subdir="screen_cells"):
    d = np.load(os.path.join(OUT, subdir, f"{label}_trace.npz"), allow_pickle=True)
    return np.asarray(d["lfp_trace"], float), np.asarray(d["rate_E"], float)


def _load_baseline_bits():
    d = np.load(os.path.join(OUT, "baseline_lfp_seed1.npz"), allow_pickle=True)
    return (np.asarray(d["lfp_trace"], float), np.asarray(d["rate_E"], float),
            np.asarray(d["contacts"], float), [str(x) for x in d["names"]], np.asarray(d["scl_mask"], bool))


# ----------------------------------------------------------------- figure 2: virtual-SEEG spectral
def fig_candidate_spectral(label, row, dt=0.05):
    lfp, rate = _load_cell_trace(label)
    base_lfp, base_rate, contacts, names, scl = _load_baseline_bits()
    ldec, fs = decimate_to_work(lfp, dt)
    rdec, _ = decimate_to_work(rate, dt)
    pr = oscillation_probe(lfp, rate, dt)
    t = np.arange(ldec.shape[0]) / fs
    plateau = (row.get("plateau") or {})
    fig, ax = plt.subplots(4, 1, figsize=(11.0, 11.0), gridspec_kw=dict(height_ratios=[2.4, 1.6, 1.6, 1.0]))

    # (a) contact stack colored by shaft
    step = np.nanmax(np.abs(ldec)) * 1.05 + 1e-9
    for c in range(ldec.shape[1]):
        ax[0].plot(t, ldec[:, c] + c * step, lw=0.4, color=SHAFT_COLS.get(_shaft(names[c]), "0.4"))
    ax[0].set_yticks([c * step for c in range(len(names))]); ax[0].set_yticklabels(names, fontsize=6.5)
    ax[0].set_title(f"virtual-SEEG contact stack — {label}", fontsize=10)
    ax[0].set_xlim(0, t[-1])

    # (b) spectrogram of the highest-energy contact
    cbest = int(np.argmax(np.var(ldec, axis=0)))
    f, tt, Sxx = _spec(ldec[:, cbest], fs)
    ax[1].pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-12), cmap="magma", shading="auto")
    ax[1].set_ylim(0, 160); ax[1].set_ylabel("Hz"); ax[1].set_title(f"spectrogram — {names[cbest]}", fontsize=9)

    # (c) six-band energy (median over contacts) over time
    bp, tcen = band_power_spectrogram(ldec, fs)
    for b, (lo, hi) in enumerate(BANDS):
        ax[2].plot(tcen, np.log10(np.median(bp[:, :, b], axis=1) + 1e-12), lw=1.1, label=f"{lo:g}-{hi:g}Hz")
    ax[2].legend(fontsize=6.5, ncol=6, loc="upper center"); ax[2].set_ylabel("log10 band power")
    ax[2].set_title("six-band energy (median across contacts)", fontsize=9)

    # (d) population rate (the coherent rhythm rides on this high-rate level)
    ax[3].plot(t, rdec, lw=0.7, color="#333"); ax[3].set_ylabel("rate_E (Hz)"); ax[3].set_xlabel("time (s)")
    ax[3].set_title(f"population rate — coherent {pr['rate_dominant_hz']:.1f} Hz on ~{pr['mean_rate_hz']:.0f} Hz mean "
                    f"(cross-contact coh {pr['coherence_med']:.2f}, phase span {pr['phase_span_deg']:.0f}°)", fontsize=8.5)
    for a in ax:
        if plateau:
            i, j = plateau["i"], plateau["j"]
            a.axvspan(tcen[i] - 0.5, tcen[j] + 0.5, color="crimson", alpha=0.10)
        a.set_xlim(0, t[-1])
    fig.text(0.5, 0.004, FOOTER + f"  |  HEO={row.get('HEO_BRANCH')}", ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(os.path.join(FIG, "candidate_virtual_seeg_spectral.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _spec(sig, fs):
    from scipy.signal import spectrogram
    nper = min(len(sig), int(round(fs)))
    return spectrogram(sig, fs=fs, window="hann", nperseg=nper, noverlap=nper - int(round(0.1 * fs)),
                       scaling="density", mode="psd")


# ----------------------------------------------------------------- figure 3: per-band ΔdB field
def fig_spatial_modes(label, row, dt=0.05):
    """Baseline-normalized ΔdB per band on the real E1146 contact geometry (the composite quantity the
    platform gate actually uses) + a ring where a contact EVER clears the strict per-band gate
    (robust-z>=Z_GATE AND power>=q99). Shows WHERE fast energy reaches vs where the strict gate is met —
    unlike an absolute-energy field, this is directly comparable to the verdict."""
    lfp, rate = _load_cell_trace(label)
    base_lfp, base_rate, contacts, names, scl = _load_baseline_bits()
    ref = build_baseline_reference(base_lfp, base_rate, dt)
    ddb = band_db_field(lfp, dt, ref)                              # (15,6) per-contact per-band ΔdB
    ldec, fs = decimate_to_work(lfp, dt)
    bp, _ = band_power_spectrogram(ldec, fs)
    logbp = np.log10(np.maximum(bp, 1e-300))
    denom = 1.4826 * ref["mad_log"]
    z = np.where(denom[None] > 0, (logbp - ref["med_log"][None]) / np.where(denom[None] > 0, denom[None], 1), -np.inf)
    ever_pass = ((z >= Z_GATE) & (bp >= ref["q99_power"][None])).any(axis=0)   # (15,6) strict per-band pass
    pr = oscillation_probe(lfp, rate, dt)
    show = [(3, "13-30 Hz"), (4, "30-80 Hz"), (5, "80-150 Hz")]
    vmax = max(6.0, float(np.nanmax(np.abs(ddb[:, [3, 4, 5]]))))
    fig, ax = plt.subplots(1, 3, figsize=(13.8, 4.8))
    sc = None
    for k, (bi, nm) in enumerate(show):
        lw = [2.4 if p else 0.7 for p in ever_pass[:, bi]]
        sc = ax[k].scatter(contacts[:, 0], contacts[:, 1], c=ddb[:, bi], s=250, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, edgecolors="k", linewidths=lw)
        for xy, n2 in zip(contacts, names):
            ax[k].annotate(n2, xy, fontsize=5.3, ha="center", va="center")
        ax[k].set_title(f"{nm} ΔdB   ({int(ever_pass[:, bi].sum())}/15 clear strict gate)", fontsize=9)
        ax[k].set_aspect("equal"); ax[k].set_xticks([]); ax[k].set_yticks([])
    fig.colorbar(sc, ax=ax, fraction=0.024, pad=0.01, label="ΔdB vs slow-off baseline (thick ring = clears strict per-band gate)")
    fig.suptitle(f"E1146 baseline-normalized band ΔdB — {label}   "
                 f"(coherent {pr['rate_dominant_hz']:.1f} Hz, coh {pr['coherence_med']:.2f}, "
                 f"phase span {pr['phase_span_deg']:.0f}°, platform {row.get('max_platform_contacts', 0)}/15)",
                 fontsize=10.5)
    fig.text(0.5, 0.01, FOOTER, ha="center", fontsize=7.5, color="0.4")
    fig.savefig(os.path.join(FIG, "candidate_spatial_modes.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_branch_map()
    print("[plot] branch map written")
    cv = json.load(open(os.path.join(OUT, "candidate_verdict.json")))
    bm = json.load(open(os.path.join(OUT, "branch_map.json")))
    cand = cv.get("candidate")
    if cand is None:
        # NO-GO: pick the most-elevated non-HEO cell (max platform coverage then mean rate) for diagnostics
        cells = bm["cells"]
        cand = max(cells, key=lambda r: (r.get("max_platform_contacts", 0), r.get("mean_rate_hz", 0))) if cells else None
    if cand is not None:
        fig_candidate_spectral(cand["label"], cand)
        fig_spatial_modes(cand["label"], cand)
        print(f"[plot] candidate figures for {cand['label']} (HEO={cand.get('HEO_BRANCH')})")
    else:
        print("[plot] no cells to render candidate figures")


if __name__ == "__main__":
    main()
