"""Two key figures from the multi-seed + position sweep (review 2026-06-09):

  sweep_ignition.png       — ignition probability per core position for the three
                             pathology types. Shows the headline (mean-down ignites,
                             variance-only never) AND the small second-order effect
                             (narrow+mean-down ignites slightly LESS reliably than
                             wide+mean-down — the wide low-threshold tail helps seed).
  sweep_matched_evoked.png — variance-only (mean-held) core's effect on evoked-event
                             co-activation, per position, mean ± SEM. Title states the
                             honest read: NO consistent direction across positions
                             (grand mean ≈ 0 with position-dependent +/- swings), NOT a
                             clean zero.

Read-only over results/.../sweep_metrics.json — no simulation re-run.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/topic4_sef_hfo/snn_heterogeneity")
FIG = OUT / "figures"
COND_COLOR = {"matched": "steelblue", "mean_only": "darkorange", "unmatched": "firebrick"}
COND_LABEL = {"matched": "variance-only (mean held)",
              "mean_only": "mean lowered (spread wide)",
              "unmatched": "mean lowered + spread narrow"}
# along-axis (far→near kick) then off-axis; pretty x labels
XLABEL = {"axis-0.5": "−0.5", "axis-0.3": "−0.3", "axis-0.1": "−0.1",
          "axis+0.1": "+0.1", "axis+0.3": "+0.3", "offaxis+": "off+", "offaxis-": "off−"}


def _load():
    d = json.loads((OUT / "sweep_metrics.json").read_text())
    return d, d["aggregate"], list(d["aggregate"].keys())


def _counts(raw, cond):
    """(n_ignited, n_total) for a condition across all position-seed cells."""
    ig = [r[f"ignited_{cond}"] for r in raw]
    return int(sum(ig)), len(ig)


def fig_ignition(d, agg, positions):
    raw = d["raw"]
    x = np.arange(len(positions))
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for cond in ("matched", "mean_only", "unmatched"):
        y = [agg[p][cond]["ignition_rate"] for p in positions]
        n_ig, n_tot = _counts(raw, cond)
        ax.plot(x, y, "-o", color=COND_COLOR[cond], lw=2.0, ms=7,
                label=f"{COND_LABEL[cond]}  ({n_ig}/{n_tot} ignite)")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks(x); ax.set_xticklabels([XLABEL[p] for p in positions])
    ax.set_xlabel("pathology-core position  (along propagation axis, far → near the stimulus  |  off-axis)")
    ax.set_ylabel("fraction of seeds where the core\nself-ignites before the stimulus")
    ax.set_title("Self-ignition is gated by mean excitability; spread modulates its reliability\n"
                 "mean-lowered cores ignite; variance-only never does; narrowing the spread "
                 "lowers ignition slightly", fontsize=10)
    ax.legend(fontsize=8, loc="center left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "sweep_ignition.png", dpi=140); plt.close(fig)


def fig_matched_evoked(d, agg, positions):
    raw = d["raw"]
    pooled = np.array([r["d_core_matched"] for r in raw], float)   # all 84 (none ignite)
    gm = float(pooled.mean()); gsem = float(pooled.std(ddof=1) / np.sqrt(len(pooled)))
    x = np.arange(len(positions))
    means = np.array([agg[p]["matched"]["d_core_evoked"]["mean"] for p in positions])
    sems = np.array([agg[p]["matched"]["d_core_evoked"]["sem"] for p in positions])
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.axhline(0, color="k", lw=0.9)
    ax.axhspan(gm - gsem, gm + gsem, color="steelblue", alpha=0.13,
               label=f"pooled mean {gm:+.3f} ± {gsem:.3f} (n={len(pooled)})")
    ax.errorbar(x, means, yerr=sems, fmt="o-", color="steelblue", lw=1.6, ms=7,
                capsize=4, label="per-position mean ± SEM (n=12 each)")
    ax.set_xticks(x); ax.set_xticklabels([XLABEL[p] for p in positions])
    ax.set_xlabel("pathology-core position  (along propagation axis, far → near the stimulus  |  off-axis)")
    ax.set_ylabel("change in core co-activation\n(variance-only core − healthy core)")
    ax.set_title("Variance-only core (mean held): no consistent direction across positions\n"
                 "position-dependent ± swings, pooled mean ≈ 0 — not evidence of a stable effect",
                 fontsize=10)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "sweep_matched_evoked.png", dpi=140); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d, agg, positions = _load()
    fig_ignition(d, agg, positions)
    fig_matched_evoked(d, agg, positions)
    print("wrote:", "sweep_ignition.png", "sweep_matched_evoked.png")


if __name__ == "__main__":
    main()
