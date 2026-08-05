#!/usr/bin/env python3
"""Where the branch changes, and whether it changes smoothly.

Two panels, two questions.  How large is the sustained swing at each strength,
and how long the system keeps moving before it stops.  A continuous instability
would grow the swing from zero and stretch that time without bound; a jump in
one and a cliff in the other say the two branches are separate.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analyze_topic4_zm_subtractive_pool_carrier import CLEAN_FLOOR  # noqa: E402


OUT = ROOT / "results/topic4_sef_hfo/zm_mode_lifecycle"
RUN_MS = 12000.0
COLOURS = {
    "decays_to_tonic_fixed_point": "#4a6fb5",
    "persistent_deep_gap_burst_train": "#d95f45",
}
LABELS = {
    "decays_to_tonic_fixed_point": "stops moving, ends on the tonic point",
    "persistent_deep_gap_burst_train": "keeps moving, but with deep gaps",
}


def main():
    payload = json.loads(
        (OUT / "subtractive_pool_carrier_summary.json").read_text()
    )
    points = []
    for name, row in payload["verdict"]["durability"].items():
        if not name.endswith("_g0.32_s1"):
            continue                                  # one substrate, one wiring
        beta = float(name.split("_")[0][1:])
        profile = row["cv_block_profile"]
        block = RUN_MS / len(profile) / 1000.0
        flat = next(
            ((i + 1) * block for i, c in enumerate(profile) if c <= CLEAN_FLOOR),
            None,
        )
        points.append((beta, row["sustained_modulation_hz"], flat,
                       row["long_run_class"], row["post_onset_deep_gap_fraction"]))
    points.sort()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    betas = [p[0] for p in points]
    for cls, colour in COLOURS.items():
        sel = [p for p in points if p[3] == cls]
        if not sel:
            continue
        axes[0].scatter([p[0] for p in sel], [p[1] for p in sel], s=60,
                        color=colour, label=LABELS[cls], zorder=3)
    axes[0].plot(betas, [p[1] for p in points], "-", color="#999", lw=.9, zorder=2)
    axes[0].set_xscale("symlog", linthresh=1.0)
    axes[0].set(xlabel="subtractive pool strength",
                ylabel="sustained rate swing (Hz)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_title("the swing jumps rather than growing from zero", fontsize=10)

    ceiling = RUN_MS / 1000.0
    for cls, colour in COLOURS.items():
        sel = [p for p in points if p[3] == cls]
        if not sel:
            continue
        axes[1].scatter([p[0] for p in sel],
                        [p[2] if p[2] is not None else ceiling for p in sel],
                        s=60, color=colour,
                        marker="o" if cls.startswith("decays") else "^", zorder=3)
    axes[1].axhline(ceiling, color="#333", ls=":", lw=1)
    axes[1].annotate("still moving when the run ended", xy=(0.02, ceiling),
                     xycoords=("axes fraction", "data"), ha="left", va="top",
                     fontsize=8, color="#333")
    axes[1].set_xscale("symlog", linthresh=1.0)
    axes[1].set(ylim=(0, ceiling * 1.08),
                xlabel="subtractive pool strength",
                ylabel="time until the rate stops moving (s)")
    axes[1].set_title("and that time does not stretch without bound", fontsize=10)
    fig.suptitle(
        "Two separate branches, not one continuously destabilising state"
        "  —  frozen slow variables", fontsize=12,
    )
    fig_dir = OUT / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "subtractive_branch_diagram.png", dpi=170)
    plt.close(fig)
    print(fig_dir / "subtractive_branch_diagram.png")
    for beta, swing, flat, cls, gap in points:
        print(f"  beta={beta:<9g} swing={swing:6.1f} Hz  "
              f"t_flat={'never' if flat is None else f'{flat:.0f} s':>7}  "
              f"gap={gap:.3f}  {cls}")


if __name__ == "__main__":
    main()
