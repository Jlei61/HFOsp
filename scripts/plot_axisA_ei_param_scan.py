#!/usr/bin/env python3
"""Figure for the Axis-A E/I parameter scan (the broad-basis NULL).

One panel, one question: across the gentler E/I window, is there ANY operating point that
yields clean directional templates while the bare sheet stays quiet? Bars = clean
directional templates per run (all 0-1, none reach the gate of 6); line = inter-event
background. Reads ei_param_scan.json. Self-contained labels. Usage: python ...
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/ei_param_scan"
QUIET = 0.001

# gentle -> stronger, readable labels
ORDER = [
    ("inhib_ei0.7", "inhib\n0.7"), ("inhib_ei0.6", "inhib\n0.6"), ("inhib_ei0.5", "inhib\n0.5"),
    ("recur_ee1.3", "recur\n1.3"), ("recur_ee1.5", "recur\n1.5"),
    ("comb_ei0.7_ee1.3", "comb\n0.7+1.3"), ("comb_ei0.6_ee1.4", "comb\n0.6+1.4"),
    ("inhib_ei0.7_seed17.8", "inhib+seed\n0.7+17.8"), ("inhib_ei0.7_seed17.5", "inhib+seed\n0.7+17.5"),
    ("inhib_ei0.6_seed17.6", "inhib+seed\n0.6+17.6"),
]


def main():
    d = json.loads((SP / "ei_param_scan.json").read_text())["cells"]
    cells = [(lbl, d[k]) for k, lbl in ORDER if k in d and d[k]["status"] == "ok"]
    xs = list(range(len(cells)))
    clean = [c[1]["n_clean_directional"] for c in cells]
    bg = [c[1]["true_floor"] for c in cells]
    quietc = ["#2d6ac2" if c[1]["bare_sheet_quiet"] else "#c2452d" for c in cells]

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.bar(xs, clean, color=quietc, alpha=0.85)
    ax.axhline(6, ls="--", c="#2a8a2a", lw=1.5)
    ax.text(0.02, 6.15, "template gate (≥6)", c="#2a8a2a", fontsize=8,
            transform=ax.get_yaxis_transform())
    ax.set_ylabel("clean directional templates / run"); ax.set_ylim(0, 8)
    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in cells], fontsize=7.5)
    ax2 = ax.twinx()
    ax2.plot(xs, bg, "k-o", lw=1.6, ms=5)
    ax2.axhline(QUIET, ls=":", c="gray", lw=1)
    ax2.text(len(xs) - 0.5, QUIET * 1.3, "bare-sheet-quiet limit", c="gray", fontsize=8, ha="right")
    ax2.set_ylabel("inter-event background (active fraction)")
    ax2.set_ylim(0, max(bg) * 1.2)
    # legend: bar color = quiet/broke
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2d6ac2", label="bare sheet quiet"),
                       Patch(color="#c2452d", label="bare sheet broke")],
              fontsize=8, loc="upper right")
    ax.set_title("E/I parameter scan: no operating point gives quiet + clean templates\n"
                 "(gentle = quiet but 0-1 templates; stronger = breaks quiet; no window between)",
                 fontsize=10)
    fig.tight_layout()
    fd = SP / "figures"; fd.mkdir(parents=True, exist_ok=True)
    fig.savefig(fd / "ei_param_scan.png", dpi=140); plt.close(fig)
    print(f"wrote {fd/'ei_param_scan.png'}")


if __name__ == "__main__":
    main()
