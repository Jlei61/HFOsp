"""Step 1 (drive × σ) joint map figure (contract §10.4 / §11).

Two independent panels (CLAUDE.md §7):
  A — fraction of random-seed runs that produced discrete self-terminating events
      (the quantity the acceptance band thresholds): "how often discrete?"
  B — the dominant regime per cell (silent / discrete / continuous-too-frequent /
      non-excitable): "when it is NOT discrete, what happens instead?"

Panel B is not a re-draw of A: A hides whether a non-discrete cell is silent
(sub-threshold) or continuously active (events too frequent to separate) — that
silent↔too-frequent structure is the whole point of the result.

Reads results/topic4_sef_hfo/step1_noise/joint_A_tau100.json.
Run: python scripts/plot_sef_hfo_step1_joint.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plot_style import (  # noqa: E402
    new_figure, style_panel, FS_LABEL, FS_TITLE,
    COL_SURROGATE, COL_SIG, COL_DAY,
)

DATA = Path("results/topic4_sef_hfo/step1_noise/joint_A_tau100.json")
OUTDIR = Path("results/topic4_sef_hfo/step1_noise/figures")

# regime -> categorical code + display
REGIME_CODE = {"non_excitable": 0, "silent": 1, "discrete": 2, "continuous": 3}
REGIME_COL = {0: "#3A3A3A", 1: COL_DAY, 2: COL_SURROGATE, 3: COL_SIG}  # dark / cream / sage / rust
REGIME_LABEL = {0: "non-excitable\n(kick fizzles)", 1: "silent\n(sub-threshold)",
                2: "discrete events", 3: "continuous\n(too frequent)"}


def _dominant_regime(per_sigma_entry):
    """Map a cell's regime_counts to a display category (modal label)."""
    rc = per_sigma_entry["regime_counts"]
    # collapse the detector labels into the four display categories
    disc = rc.get("discrete_events", 0)
    silent = rc.get("extinction_only", 0)
    cont = rc.get("sustained", 0) + rc.get("runaway", 0) + rc.get("captured_high", 0)
    winner = max([("silent", silent), ("discrete", disc), ("continuous", cont)],
                 key=lambda kv: kv[1])
    return REGIME_CODE[winner[0]]


def main():
    d = json.loads(DATA.read_text())
    drives = d["params"]["drives"]
    sigmas = d["params"]["sigmas"]
    by_drive = {r["drive"]: r for r in d["per_drive"]}
    rs = d["region_summary"]

    nF = np.full((len(drives), len(sigmas)), np.nan)   # frac_discrete
    nR = np.full((len(drives), len(sigmas)), np.nan)   # regime code
    for i, dr in enumerate(drives):
        r = by_drive[dr]
        if r["undetectable"]:
            nR[i, :] = REGIME_CODE["non_excitable"]
            continue
        ps = {p["sigma"]: p for p in r["per_sigma"]}
        for j, sg in enumerate(sigmas):
            if sg in ps:
                nF[i, j] = ps[sg]["frac_discrete"]
                nR[i, j] = _dominant_regime(ps[sg])

    fig, (axA, axB) = new_figure(1, 2, figsize=(15, 6))

    # ---- Panel A: fraction of seeds discrete --------------------------------
    im = axA.imshow(nF, origin="lower", aspect="auto", cmap="viridis",
                    vmin=0, vmax=1, interpolation="nearest")
    axA.set_xticks(range(len(sigmas)))
    axA.set_xticklabels([f"{s:g}" for s in sigmas])
    axA.set_yticks(range(len(drives)))
    axA.set_yticklabels([f"{dr:g}" for dr in drives])
    axA.set_xlabel("noise amplitude σ (mV)", fontsize=FS_LABEL)
    axA.set_ylabel("external drive  ν_ext / ν_θ", fontsize=FS_LABEL)
    axA.set_title("Fraction of seeds with discrete events", fontsize=FS_TITLE)
    # annotate each cell; cross-hatch non-excitable row
    for i, dr in enumerate(drives):
        for j, sg in enumerate(sigmas):
            if np.isnan(nF[i, j]):
                axA.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                            hatch="xx", edgecolor="#3A3A3A", lw=0))
                continue
            v = nF[i, j]
            axA.text(j, i, f"{v:.1f}".lstrip("0") if v > 0 else "·",
                     ha="center", va="center", fontsize=11,
                     color="white" if v > 0.5 else "#222222",
                     fontweight="bold" if v > 0 else "normal")
    cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
    cb.set_label("fraction of 5 seeds", fontsize=FS_LABEL - 1)
    style_panel(axA, "a", label_x=-0.12, label_y=1.05)
    # acceptance criterion note (machined, §10.4)
    acc = "none" if rs["n_accepted_cells"] == 0 else str(rs["accepted_cells"])
    axA.text(0.5, -0.20, f"accepted cells (≥60% seeds & rate∈[0.01,1]/s): {acc}    "
             f"robust 2-D region: {'yes' if rs['robust_2d_block'] else 'no'}",
             transform=axA.transAxes, ha="center", fontsize=FS_LABEL - 2, color="#444444")

    # ---- Panel B: dominant regime -------------------------------------------
    cmap = ListedColormap([REGIME_COL[k] for k in sorted(REGIME_COL)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    axB.imshow(nR, origin="lower", aspect="auto", cmap=cmap, norm=norm,
               interpolation="nearest")
    axB.set_xticks(range(len(sigmas)))
    axB.set_xticklabels([f"{s:g}" for s in sigmas])
    axB.set_yticks(range(len(drives)))
    axB.set_yticklabels([f"{dr:g}" for dr in drives])
    axB.set_xlabel("noise amplitude σ (mV)", fontsize=FS_LABEL)
    axB.set_ylabel("external drive  ν_ext / ν_θ", fontsize=FS_LABEL)
    axB.set_title("Dominant regime", fontsize=FS_TITLE)
    style_panel(axB, "b", label_x=-0.12, label_y=1.05)
    handles = [Patch(facecolor=REGIME_COL[k], edgecolor="#444", label=REGIME_LABEL[k])
               for k in sorted(REGIME_COL)]
    axB.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
               fontsize=FS_LABEL - 2, frameon=False)

    fig.suptitle("Homogeneous rate field, recovery off — noise-driven events across (drive × σ)",
                 fontsize=FS_TITLE + 1, y=1.02)
    fig.tight_layout()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUTDIR / f"step1_joint_drive_sigma.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
