#!/usr/bin/env python3
"""
Generate periodicity summary figures comparable to paper Fig 3C, S7, S13.

Produces:
  1. Per-subject PSD with specparam decomposition (S7-equivalent)
  2. Cohort stacked PSD + peak frequency histogram (Fig 3C-equivalent)
  3. IEI distribution summary table (S13-equivalent)

Usage:
    python scripts/plot_event_periodicity.py

These plots summarize the reproduced PSD / IEI results, but they should be read
through the updated scientific interpretation:
  - a visible ~2 Hz group peak is not, by itself, evidence for an intrinsic
    oscillator
  - surrogate/null-model results and IEI model comparison determine whether a
    peak is biologically meaningful or a refractory / distribution artifact
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/event_periodicity")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_results(dataset: str) -> list:
    d = RESULTS_DIR / dataset
    results = []
    for f in sorted(d.glob("*_periodicity.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def plot_subject_psd(result: dict, outdir: Path) -> None:
    """Plot PSD + specparam decomposition for one subject (Fig S7 equivalent)."""
    sub = result["subject"]
    group = result.get("group")
    channels = result.get("channels", [])

    n_panels = 1 + len([c for c in channels if c.get("specparam")])
    if n_panels == 0:
        return
    if group and not group.get("specparam"):
        return

    fig, axes = plt.subplots(1, min(n_panels, 5), figsize=(4 * min(n_panels, 5), 3.5),
                             squeeze=False)
    axes = axes.flatten()

    panel_idx = 0

    if group and group.get("specparam"):
        sp = group["specparam"]
        freqs = np.array(sp["freqs"])
        power = np.array(sp["power_spectrum"])
        ap_fit = np.array(sp["ap_fit"])
        modeled = np.array(sp["fooofed_spectrum"])

        ax = axes[panel_idx]
        ax.semilogy(freqs, 10 ** power, "k-", lw=1, label="PSD")
        ax.semilogy(freqs, 10 ** ap_fit, "b--", lw=0.8, label="Aperiodic")
        ax.semilogy(freqs, 10 ** modeled, "r-", lw=0.8, alpha=0.7, label="Model")

        peaks = np.array(sp["peaks"])
        for pk in peaks:
            if 0.5 < pk[0] < 5.0:
                ax.axvline(pk[0], color="orange", ls=":", lw=0.7, alpha=0.8)
                ax.text(pk[0], ax.get_ylim()[1] * 0.8, f"{pk[0]:.1f}Hz",
                        fontsize=7, ha="center", color="orange")

        ax.set_title(f"Group (n={group['n_events']})", fontsize=8)
        ax.set_xlabel("Freq (Hz)", fontsize=8)
        ax.set_ylabel("Power", fontsize=8)
        ax.legend(fontsize=6)
        ax.set_xlim(0, 10)
        panel_idx += 1

    for ch in channels:
        if panel_idx >= len(axes):
            break
        if not ch.get("specparam"):
            continue
        sp = ch["specparam"]
        freqs = np.array(sp["freqs"])
        power = np.array(sp["power_spectrum"])
        ap_fit = np.array(sp["ap_fit"])

        ax = axes[panel_idx]
        ax.semilogy(freqs, 10 ** power, "k-", lw=1)
        ax.semilogy(freqs, 10 ** ap_fit, "b--", lw=0.8)

        peaks = np.array(sp["peaks"])
        for pk in peaks:
            if 0.5 < pk[0] < 5.0:
                ax.axvline(pk[0], color="orange", ls=":", lw=0.7)

        ax.set_title(f"{ch['channel']} (n={ch['n_events']})", fontsize=8)
        ax.set_xlabel("Freq (Hz)", fontsize=8)
        ax.set_xlim(0, 10)
        panel_idx += 1

    for i in range(panel_idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"{sub} — Event PSD + Specparam", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / f"{sub}_psd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cohort_psd_stack(results: list, dataset: str) -> None:
    """Stacked PSD + peak histogram (Fig 3C equivalent)."""
    fig, (ax_stack, ax_hist) = plt.subplots(1, 2, figsize=(10, 5))

    peak_freqs = []
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

    for i, res in enumerate(results):
        grp = res.get("group")
        if not grp or not grp.get("psd"):
            continue
        psd = grp["psd"]
        freqs = np.array(psd["freqs"])
        power = np.array(psd["power"])

        power_db = 10 * np.log10(power + 1e-30)
        power_norm = power_db - np.mean(power_db)

        ax_stack.plot(freqs, power_norm + i * 3, color=colors[i], lw=0.8,
                      label=res["subject"])

        sp = grp.get("specparam")
        if sp and sp.get("peaks"):
            peaks = np.array(sp["peaks"])
            for pk in peaks:
                if 0.5 < pk[0] < 5.0:
                    peak_freqs.append(pk[0])

    ax_stack.set_xlabel("Frequency (Hz)", fontsize=10)
    ax_stack.set_ylabel("Normalized Power (stacked)", fontsize=10)
    ax_stack.set_title(f"{dataset} — Stacked Group PSD", fontsize=11)
    ax_stack.set_xlim(0, 8)
    ax_stack.legend(fontsize=6, ncol=2, loc="upper right")

    if peak_freqs:
        ax_hist.hist(peak_freqs, bins=np.arange(0.5, 5.5, 0.25),
                     color="steelblue", edgecolor="white", alpha=0.8)
        ax_hist.axvline(np.median(peak_freqs), color="red", ls="--", lw=1,
                        label=f"median={np.median(peak_freqs):.2f}Hz")
        ax_hist.set_xlabel("Peak Frequency (Hz)", fontsize=10)
        ax_hist.set_ylabel("Count", fontsize=10)
        ax_hist.set_title(f"Peak Distribution (n={len(peak_freqs)})", fontsize=11)
        ax_hist.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{dataset}_cohort_psd_stack.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_iei_summary(results: list, dataset: str) -> None:
    """IEI distribution summary (S13 equivalent)."""
    rows = []
    for res in results:
        grp = res.get("group")
        if not grp or not grp.get("iei_fit"):
            continue
        iei = grp["iei_fit"]
        sp = grp.get("specparam", {})
        peaks = sp.get("peaks", [])
        pf = None
        for pk in peaks:
            if 0.5 < pk[0] < 5.0:
                pf = pk[0]
                break
        rows.append({
            "subject": res["subject"],
            "n_events": grp["n_events"],
            "peak_freq": pf,
            "alpha": iei["alpha"],
            "xmin": iei["xmin"],
            "pl_vs_ln_R": iei["pl_vs_ln_R"],
            "pl_vs_ln_p": iei["pl_vs_ln_p"],
            "iei_mean": iei["iei_mean"],
            "iei_median": iei["iei_median"],
        })

    if not rows:
        return

    fig, (ax_alpha, ax_llr) = plt.subplots(1, 2, figsize=(10, 5))

    subjects = [r["subject"] for r in rows]
    alphas = [r["alpha"] for r in rows]
    llr_R = [r["pl_vs_ln_R"] for r in rows]
    x = np.arange(len(subjects))

    ax_alpha.barh(x, alphas, color="steelblue", alpha=0.8)
    ax_alpha.set_yticks(x)
    ax_alpha.set_yticklabels(subjects, fontsize=7)
    ax_alpha.set_xlabel("Power-law exponent (α)", fontsize=10)
    ax_alpha.set_title(f"{dataset} — IEI Power-law α", fontsize=11)
    ax_alpha.axvline(2.0, color="red", ls="--", lw=0.8, alpha=0.5, label="α=2.0")
    ax_alpha.legend(fontsize=8)

    colors = ["green" if r > 0 else "red" for r in llr_R]
    ax_llr.barh(x, llr_R, color=colors, alpha=0.8)
    ax_llr.set_yticks(x)
    ax_llr.set_yticklabels(subjects, fontsize=7)
    ax_llr.set_xlabel("Log-likelihood ratio (PL vs LN)", fontsize=10)
    ax_llr.set_title("Power-law vs Lognormal\n(R>0 = PL preferred)", fontsize=10)
    ax_llr.axvline(0, color="black", ls="-", lw=0.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{dataset}_iei_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    for dataset in ["yuquan", "epilepsiae"]:
        results = load_results(dataset)
        if not results:
            print(f"No results for {dataset}")
            continue

        print(f"=== {dataset}: {len(results)} subjects ===")

        subdir = FIG_DIR / dataset
        subdir.mkdir(parents=True, exist_ok=True)
        for res in results:
            plot_subject_psd(res, subdir)

        plot_cohort_psd_stack(results, dataset)
        plot_iei_summary(results, dataset)

    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
