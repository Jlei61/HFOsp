#!/usr/bin/env python3
"""The branch trade-off, drawn on three twelve-second trajectories.

This is a mechanism-failure figure, not a lifecycle figure.  It shows that the
two properties a seizure carrier needs — staying continuous and staying
modulated — were reachable one at a time and never together.
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
from scripts.analyze_topic4_zm_subtractive_pool_carrier import (  # noqa: E402
    cv_block_profile, modulation_amplitude_hz,
)


LONG = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint/seed1"
OUT = ROOT / "results/topic4_sef_hfo/zm_mode_lifecycle"
STEM = ("i2e__tauD300.7__d0.7281__s1__modeH0t250__mc30pg0.32e60{beta}"
        "__freeze_bounded_late__peak__pvSOMq0.25f0.35sig1.5c64rd3tr4td60s1"
        "__T12s__gM1__tauM500")
COLUMNS = (
    (0.0, "no subtractive term", "continuous, but the rate stops moving by 1.5 s"),
    (3.9171, "subtractive term at 6% removal", "keeps moving for 4 s, then stops"),
    (7.8342, "subtractive term at 10% removal", "keeps moving, but the gaps return"),
)


def _load(beta):
    tag = "" if beta == 0.0 else f"__bSG{beta:g}"
    root = LONG / STEM.format(beta=tag)
    summary = json.loads((root / "summary.json").read_text())
    with np.load(root / "traces.npz", allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key], float) for key in data.files}
    return summary, arrays


def main():
    loaded = [(beta, title, sub, *_load(beta)) for beta, title, sub in COLUMNS]
    fig, axes = plt.subplots(
        4, 3, figsize=(15, 10), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.1, 1.3, 0.9, 0.9]},
    )
    rate_top = max(a["fine_core_rate_hz"].max() for *_, a in loaded)
    for col, (beta, title, sub, summary, a) in enumerate(loaded):
        t = a["fine_time_ms"] / 1000.0
        rate = a["fine_core_rate_hz"]
        axes[0, col].plot(t, rate, color="#d95f45", lw=.5)
        axes[0, col].set(xlabel="time (s)", ylabel="core rate (Hz)",
                         ylim=(0, rate_top * 1.05))
        axes[0, col].set_title(f"{title}\n{sub}", fontsize=11)

        kymo = a["coarse_kymo_axial"]
        axes[1, col].imshow(kymo, origin="lower", aspect="auto", cmap="magma",
                            extent=[0, .025 * kymo.shape[1], 0, kymo.shape[0]])
        axes[1, col].set(xlabel="time (s)", ylabel="axis bin")

        profile = cv_block_profile(rate, fs=500.0, block_ms=2000.0)
        centres = np.arange(len(profile)) * 2.0 + 1.0
        axes[2, col].bar(centres, profile, width=1.6, color="#4a6fb5")
        axes[2, col].axhline(0.25, color="#333", ls=":", lw=1)
        axes[2, col].set(xlabel="time (s)", ylabel="rate variability\nper 2 s",
                         ylim=(0, 2.0))
        axes[2, col].annotate(
            f"swing {modulation_amplitude_hz(rate[-4000:]):.0f} Hz",
            xy=(0.97, 0.9), xycoords="axes fraction", ha="right", fontsize=9)

        sg = a["trace_S_G"]
        ts = np.arange(sg.size) * 0.0001
        axes[3, col].plot(ts, sg, color="#2b7a5b", lw=.9, label="shared pool")
        axes[3, col].plot(ts, beta * sg, color="#8a4fa8", lw=.9,
                          label="subtracted current")
        axes[3, col].set(xlabel="time (s)", ylabel="pool state")
        if col == 0:
            axes[3, col].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Continuity and modulation were reachable one at a time, never together"
        "  —  frozen slow variables, no seizure lifecycle is claimed here",
        fontsize=13,
    )
    fig_dir = OUT / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "subtractive_branch_tradeoff.png", dpi=170)
    plt.close(fig)
    print(fig_dir / "subtractive_branch_tradeoff.png")


if __name__ == "__main__":
    main()
