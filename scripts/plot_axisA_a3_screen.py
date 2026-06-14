#!/usr/bin/env python3
"""Paper-grade figure for the Axis-A A3 E/I-lesion screen (the NULL).

Two independent panels (CLAUDE.md §7):
  A. Does a stronger local E/I lesion GROW the event, or just raise the background?
     -> max contacts recruited stays flat (~7-8) while inter-event background jumps.
  B. Is "events stay local / no clean directional template" ROBUST across seeds?
     -> clean directional templates per run ~0 across seeds, vs the V_th-down reference.

Reads a3_0a_scan.json (+ a3_seed_confirm.json if present). Self-contained labels (no
codenames). Usage: python scripts/plot_axisA_a3_screen.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
SCAN = SP / "a3_0a_scan/a3_0a_scan.json"
CONFIRM = SP / "a3_seed_confirm/a3_seed_confirm.json"
VTH_BASELINE_CLEAN = 9            # V_th-down clean directional events at T=2000 (a1_0a wide band)
PART_MIN = 7

LABEL = {"oneend_inhib": "inhibition collapse", "oneend_recur": "recurrent excitation"}
COLOR = {"oneend_inhib": "#c2452d", "oneend_recur": "#2d6ac2"}


def main():
    scan = json.loads(SCAN.read_text())["cells"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # --- Panel A: event size vs background, ordered weak->strong per lesion ---
    order = [c for c in scan if c["status"] == "ok"]
    xs = list(range(len(order)))
    xlab = []
    for c in order:
        strength = "weak" if (c["knob"] == "ei_scale" and c["value"] == 0.5) or \
                             (c["knob"] == "ee_gain" and c["value"] == 1.5) else \
                   ("strong" if (c["knob"] == "ei_scale" and c["value"] == 0.2) or
                                (c["knob"] == "ee_gain" and c["value"] == 2.5) else "mid")
        xlab.append(f"{LABEL[c['lesion']].split()[0]}\n{strength}")
    sizes = [c["n_part_max"] for c in order]
    bg = [c["true_floor"] for c in order]
    bars = ax[0].bar(xs, sizes, color=[COLOR[c["lesion"]] for c in order], alpha=0.85,
                     label="max contacts recruited")
    ax[0].axhline(PART_MIN, ls="--", c="gray", lw=1)
    ax[0].text(-0.45, PART_MIN + 0.2, "readable-template floor (≥7)", c="gray", fontsize=8, ha="left")
    ax[0].set_ylabel("max contacts recruited / event")
    ax[0].set_ylim(0, 12.5)
    ax2 = ax[0].twinx()
    ax2.plot(xs, bg, "k-o", lw=1.6, ms=5, label="inter-event background")
    ax2.set_ylabel("inter-event background (active fraction)")
    ax2.set_ylim(0, max(bg) * 1.25)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(xlab, fontsize=8)
    ax[0].set_title("Stronger lesion raises background, not event size")
    h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    # --- Panel B: clean directional templates per seed (robustness) ---
    if CONFIRM.exists():
        vd = json.loads(CONFIRM.read_text())["verdict"]
        for les, c in COLOR.items():
            seeds = list(range(1, len(vd[les]["clean_dir_per_seed"]) + 1))
            ax[1].plot(seeds, vd[les]["clean_dir_per_seed"], "-o", c=c, lw=1.6, ms=6,
                       label=LABEL[les])
        ax[1].set_xticks(seeds)
    else:
        ax[1].text(0.5, 0.5, "seed-confirm pending", ha="center", transform=ax[1].transAxes)
    ax[1].axhline(VTH_BASELINE_CLEAN, ls="--", c="#2a8a2a", lw=1.5)
    ax[1].text(0.02, VTH_BASELINE_CLEAN - 0.7, "V_th-down reference", c="#2a8a2a",
               fontsize=8, transform=ax[1].get_yaxis_transform())
    ax[1].set_xlabel("network seed"); ax[1].set_ylabel("clean directional templates / run")
    ax[1].set_ylim(-0.5, VTH_BASELINE_CLEAN + 2)
    ax[1].set_title("Local E/I events robustly fail to read as templates")
    ax[1].legend(fontsize=8, loc="center right")

    fig.suptitle("Local E/I lesion does not reproduce the threshold-lowering read-out signature "
                 "(screen, T=2000)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fd = SP / "a3_0a_scan/figures"; fd.mkdir(parents=True, exist_ok=True)
    fig.savefig(fd / "a3_ei_screen.png", dpi=140); plt.close(fig)
    print(f"wrote {fd/'a3_ei_screen.png'}")


if __name__ == "__main__":
    main()
