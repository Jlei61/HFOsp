"""Mean-amplitude scan figure (2026-06-09). One 3-panel figure, each panel a
distinct question (CLAUDE.md §7):
  a  ignition probability vs core mean (wide vs narrow) — IS there a boundary, and
     does the wide tail shift the whole curve to a HIGHER mean (horizontal shift,
     the detectable form of "tail helps near boundary")? mean=18 = sanity anchor.
  b  ignition latency vs core mean (igniting trials) — how violently past the boundary.
  c  evoked co-activation vs core mean (NON-igniting trials only) — does the variance-
     only effect stay null as mean approaches the boundary from above?

Read-only over mean_scan_metrics.json — no simulation re-run.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/topic4_sef_hfo/snn_heterogeneity")
FIG = OUT / "figures"
SCOLOR = {"wide": "darkorange", "narrow": "firebrick"}
SLABEL = {"wide": "wide spread (std 1.5)", "narrow": "narrow spread (std 0.5)"}


def _load(tag=""):
    return json.loads((OUT / f"mean_scan{tag}_metrics.json").read_text())


def _panel_ignition(ax, d):
    means = sorted(d["means"])
    for s in ("wide", "narrow"):
        p = np.array([d["aggregate"][s][f"{m:.1f}"]["ignition_rate"] for m in means])
        n = np.array([d["aggregate"][s][f"{m:.1f}"]["n"] for m in means])
        sem = np.sqrt(p * (1 - p) / n)
        ax.errorbar(means, p, yerr=sem, fmt="-o", color=SCOLOR[s], lw=2, ms=7,
                    capsize=4, label=SLABEL[s])
        xc = d["ignition_cross_mean"][s]
        if xc is not None and np.isfinite(xc):
            ax.axvline(xc, color=SCOLOR[s], ls=":", lw=1.3)
    ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
    ax.scatter([18.0], [d["aggregate"]["narrow"]["18.0"]["ignition_rate"]], s=140,
               facecolors="none", edgecolors="k", zorder=5, lw=1.4)
    ax.annotate("mean=18 sanity anchor\n(0.00 = reproduces sweep null)", xy=(18.0, 0.03),
                xytext=(17.85, 0.42), fontsize=7.5, ha="left",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    sh = d.get("boundary_shift_wide_minus_narrow")
    sht = (f"boundary shift (wide−narrow) = {sh:+.2f} mV  (wide tail ignites at higher mean)"
           if sh is not None else "boundary shift: n/a (a curve never crosses 0.5)")
    ax.set_xlabel("core mean threshold (mV)   — lower = more excitable →")
    ax.set_ylabel("P(core self-ignites before stimulus)")
    ax.set_title(f"a · ignition boundary\n{sht}", fontsize=10)
    ax.invert_xaxis()                                    # mean decreasing left→right
    ax.set_ylim(-0.05, 1.10); ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)


def _panel_latency(ax, d):
    means = sorted(d["means"])
    for s in ("wide", "narrow"):
        xs, ys, es = [], [], []
        for m in means:
            ci = d["aggregate"][s][f"{m:.1f}"]["latency_igniting"]
            if ci is not None:
                xs.append(m); ys.append(ci["mean"]); es.append(ci["sem"])
        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt="-o", color=SCOLOR[s], lw=2, ms=7,
                        capsize=4, label=SLABEL[s])
    ax.set_xlabel("core mean threshold (mV)")
    ax.set_ylabel("pre-kick ignition latency (ms)")
    ax.set_title("b · how early it self-ignites\n(igniting trials only)", fontsize=10)
    ax.invert_xaxis(); ax.legend(fontsize=8); ax.grid(alpha=0.3)


def _panel_evoked(ax, d):
    means = sorted(d["means"])
    ax.axhline(0, color="k", lw=0.8)
    for s in ("wide", "narrow"):
        xs, ys, es = [], [], []
        for m in means:
            ci = d["aggregate"][s][f"{m:.1f}"]["d_core_paf_evoked"]
            if ci is not None and ci["n"] >= 2:
                xs.append(m); ys.append(ci["mean"]); es.append(ci["sem"])
        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt="-o", color=SCOLOR[s], lw=2, ms=7,
                        capsize=4, label=SLABEL[s])
    ax.set_xlabel("core mean threshold (mV)")
    ax.set_ylabel("Δ core co-activation\n(pathology − healthy, non-igniting)")
    ax.set_title("c · evoked effect above the boundary\n(non-igniting trials only)", fontsize=10)
    ax.invert_xaxis(); ax.legend(fontsize=8); ax.grid(alpha=0.3)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="", help="metrics/figure suffix, e.g. _fine")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    d = _load(a.tag)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
    _panel_ignition(axes[0], d)
    _panel_latency(axes[1], d)
    _panel_evoked(axes[2], d)
    nseed = len(d['conn_seeds']) * len(d['fn_seeds'])
    pass_lbl = ("2nd pass / fresh networks" if a.tag else "")
    fig.suptitle(f"Pathology-core mean-amplitude scan{(' — ' + pass_lbl) if pass_lbl else ''}: "
                 f"ignition boundary as core mean walks down "
                 f"({len(d['conn_seeds'])} networks × {len(d['fn_seeds'])} field-noise = {nseed} "
                 "independent realizations/cell; fixed mid core + end kick)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / f"mean_scan{a.tag}.png", dpi=140); plt.close(fig)
    print(f"wrote mean_scan{a.tag}.png")


if __name__ == "__main__":
    main()
