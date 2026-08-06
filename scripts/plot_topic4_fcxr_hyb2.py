#!/usr/bin/env python3
"""Plot the canonical FCXR-HYB2 gate summary from reduced JSON artifacts.

This script does not re-run a simulation or re-adjudicate a gate.  It only
renders the already locked Gate B0 and Gate A0 outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path(
    "results/topic4_sef_hfo/mz_full_conductance_spatial_relay/"
    "hyb2_event_limited_recruitment"
)
LEVELS = ("S25", "S50", "S75")


def _load(path: Path):
    with path.open() as f:
        return json.load(f)


def build_figure(root: Path, out: Path) -> None:
    b0 = [_load(root / f"gate_b0_seed{s}.json") for s in (1, 3)]
    a0 = {
        level: {
            arm: _load(root / f"gate_a0_arm_{arm}_{level}.json")
            for arm in ("off", "on")
        }
        for level in LEVELS
    }
    verdict = _load(root / "gate_a0.json")

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.25), constrained_layout=True)
    colors = {"off": "#777777", "on": "#d1495b"}
    x = np.arange(len(LEVELS))

    # A: the membrane-facing B0 gate, deliberately separated from hidden q_v.
    occ = [float(x_["checks"]["active_occupancy"]["value"]) for x_ in b0]
    axes[0].bar(["seed1", "seed3"], occ, color="#4c78a8", width=0.62)
    axes[0].axhline(0.01, color="black", linestyle="--", linewidth=1, label="Gate = 1%")
    axes[0].set_yscale("symlog", linthresh=1e-5)
    axes[0].set_ylim(-1e-6, 2e-2)
    axes[0].set_ylabel("Interictal $R_{evt}$ occupancy")
    axes[0].set_title("A  Baseline visibility")
    axes[0].legend(frameon=False, fontsize=8)
    for i, v in enumerate(occ):
        axes[0].text(i, max(v, 1.5e-6), f"{v:.2g}", ha="center", va="bottom", fontsize=8)

    # B: eligibility was decided from the off arm alone and is the blocking result.
    nvox = [float(a0[l]["off"]["n_occupied_voxels"]) for l in LEVELS]
    frac = [float(a0[l]["off"]["participant_voxels"]) / n for l, n in zip(LEVELS, nvox)]
    axes[1].bar(x, np.asarray(frac) * 100, color="#72b7b2", width=0.62)
    axes[1].axhline(90, color="black", linestyle="--", linewidth=1, label="Ceiling = 90%")
    axes[1].set_xticks(x, LEVELS)
    axes[1].set_ylim(84, 94)
    axes[1].set_ylabel("Off-arm occupied voxels (%)")
    axes[1].set_title("B  A0 eligibility")
    axes[1].legend(frameon=False, fontsize=8)
    for i, v in enumerate(frac):
        axes[1].text(i, v * 100 + 0.25, f"{100*v:.1f}%", ha="center", fontsize=8)

    # C: the actuator was exposed increasingly strongly along the locked Z ladder.
    for arm, marker in (("off", "o"), ("on", "s")):
        vals = [float(a0[l][arm]["max_R_evt"]) for l in LEVELS]
        axes[2].plot(x, vals, marker=marker, color=colors[arm], label=arm)
    axes[2].axhline(4.134151260609386, color="black", linestyle=":", linewidth=1,
                    label="$I_{R,max}$")
    axes[2].set_xticks(x, LEVELS)
    axes[2].set_ylabel("Maximum $R_{evt}$ drive")
    axes[2].set_title("C  Actuator exposure")
    axes[2].legend(frameon=False, fontsize=8)

    # D: show why equality cannot be read as inefficacy once B is ceiling-confounded.
    width = 0.36
    off_rate = [float(a0[l]["off"]["end_rate_hz"]) for l in LEVELS]
    on_rate = [float(a0[l]["on"]["end_rate_hz"]) for l in LEVELS]
    axes[3].bar(x - width / 2, off_rate, width, color=colors["off"], label="off")
    axes[3].bar(x + width / 2, on_rate, width, color=colors["on"], label="on")
    axes[3].set_xticks(x, LEVELS)
    axes[3].set_ylabel("Final 1-s E rate (Hz)")
    axes[3].set_title("D  Downstream activity")
    axes[3].legend(frameon=False, fontsize=8)

    fig.suptitle(
        "FCXR-HYB2: baseline-safe actuator, but all locked A0 inputs are spatially ceiling-confounded",
        fontsize=11,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    if verdict.get("status") != "A0_UNDECIDABLE_ALL_LEVELS":
        raise RuntimeError(f"unexpected A0 verdict after rendering: {verdict.get('status')}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or args.root / "figures" / "hyb2_gate_summary.png"
    build_figure(args.root, out)
    print(out)


if __name__ == "__main__":
    main()
