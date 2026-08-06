"""FCXR-HEO3 H3.1b figure — geometry correction + load-matched mean-field, three questions:
(1) did centring the stripes ON the cores change the dynamics (vs the boundary-aligned H3.1 run and vs
    the centred arm's own shuffle)?
(2) is "inter-cell differences carry the broadening" still true once the mean-field arm's potassium
    load is matched (H3.1 ran it at 2.3x load, confounded)?
(3) do the two cores take turns now that they really have different recovery times?
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MZ = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay")
OUT = os.path.join(MZ, "heo3")
CRIT = [("recruited", "#4c72b0"), ("broadband", "#e8a33d"), ("desynchronized", "#c44e52"), ("high_energy", "#55a868")]


def main():
    j1 = json.load(open(os.path.join(OUT, "stage1_joint.json")))
    j2 = json.load(open(os.path.join(OUT, "stage1b_joint.json")))
    J = {**j1, **j2}
    order = [("dyn_tau250_frac0.1", "uniform\nDYNAMIC"), ("patch_dyn_strong", "patch 4×\nBOUNDARY-aligned\n(H3.1 bug)"),
             ("patch_dyn_centered", "patch 4×\nCENTRED\n(corrected)"),
             ("patch_dyn_centered_shuffled", "centred\nSHUFFLED"),
             ("patch_dyn_centered_swapped", "centred\nSWAPPED")]
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 4.7), gridspec_kw=dict(width_ratios=[1.9, 1.15, 1.0], wspace=0.42))

    x = np.arange(len(order)); w = 0.2
    for i, (c, col) in enumerate(CRIT):
        ax[0].bar(x + (i - 1.5) * w, [100 * J[k]["frac_by_criterion"][c] for k, _ in order], w, color=col, label=c)
    ax[0].axvspan(1.5, 2.5, color="#e8f4e8", zorder=0)
    ax[0].set_xticks(x); ax[0].set_xticklabels([n for _, n in order], fontsize=6.6)
    ax[0].set_ylabel("% of 1 s windows"); ax[0].set_ylim(0, 115)
    ax[0].legend(fontsize=7, ncol=4, loc="upper center")
    ax[0].set_title("(1) centring the stripes ON the cores changed the dynamics\n"
                    "recruit 74→100%, desync 10→23%, energy 52→74% — and it beats its own SHUFFLE "
                    "(desync 23 vs 10%)\nBUT joint target windows are still 0 in every arm", fontsize=8.5)

    pair = [("dyn_tau250_frac0.1", "per-cell m\n(uniform DYN)\ngM 0.191"),
            ("meanfield_dyn", "mean-field\nunmatched\ngM 0.438"),
            ("meanfield_dyn_loadmatched", "mean-field\nLOAD-MATCHED\ngM 0.207")]
    xb = np.arange(3)
    ax[1].bar(xb - 0.18, [100 * J[k]["frac_by_criterion"]["broadband"] for k, _ in pair], 0.34,
              color="#e8a33d", label="broadband")
    ax[1].bar(xb + 0.18, [100 * J[k]["frac_by_criterion"]["desynchronized"] for k, _ in pair], 0.34,
              color="#c44e52", label="desynchronized")
    for i, (k, _) in enumerate(pair):
        v = 100 * J[k]["frac_by_criterion"]["broadband"]
        ax[1].text(i - 0.18, v + 0.8, f"{v:.0f}%", ha="center", fontsize=7,
                   color="#8a5a00" if v > 0 else "0.45")
    ax[1].set_xticks(xb); ax[1].set_xticklabels([n for _, n in pair], fontsize=6.6)
    ax[1].set_ylabel("% of windows"); ax[1].set_ylim(0, 40); ax[1].legend(fontsize=7, loc="upper right")
    ax[1].set_title("(2) population-mean m kills the broadening\nAT MATCHED LOAD → inter-cell\n"
                    "differences carry it", fontsize=8.5)

    alt_lab, alt = [], []
    for k, n in order[2:]:
        f = os.path.join(OUT, "arms", k + ".json")
        if os.path.exists(f):
            alt.append(json.load(open(f))["region_alternation"]); alt_lab.append(n.replace("\n", " "))
    f = os.path.join(OUT, "arms", "meanfield_dyn_loadmatched.json")
    alt.append(json.load(open(f))["region_alternation"]); alt_lab.append("mean-field matched")
    ax[2].barh(np.arange(len(alt)), alt, color=["#c44e52" if v > 0 else "#4c72b0" for v in alt])
    ax[2].axvline(0, c="0.3", lw=0.8); ax[2].axvline(0.9, ls=":", c="0.5", lw=1)
    ax[2].set_yticks(np.arange(len(alt))); ax[2].set_yticklabels(alt_lab, fontsize=6.8)
    ax[2].set_xlim(-1, 1.05); ax[2].set_xlabel("corr(core-source rate, core-sink rate)")
    ax[2].set_title("(3) do the cores take turns NOW?\nsource all-fast vs sink all-slow —\n"
                    "still +0.96, they rise and fall together", fontsize=8.5)

    fig.text(0.5, 0.005, "FCXR-HEO3 H3.1b — corrected stripe phase (source core 0% slow / sink core 100% slow) + load-matched mean-field control",
             ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "figures", "stage1b_geometry_fix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[h3.1b] wrote figures/stage1b_geometry_fix.png")


if __name__ == "__main__":
    main()
