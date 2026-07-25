"""FCXR-HEO3 H3.1 figure — the causal 2x2 + controls, three independent questions:
(1) did SPATIAL ORGANIZATION of recovery time move any joint criterion (2x2 + shuffle control)?
(2) are INTER-CELL adaptation differences load-bearing (mean-field control vs uniform dynamic)?
(3) do regions TAKE TURNS (corrected raw-rate alternation, not the shares tautology)?
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
ORDER = ["m_off", "uniform_static", "patch_static", "dyn_tau250_frac0.1", "patch_dyn_weak",
         "patch_dyn_strong", "patch_dyn_strong_shuffled", "meanfield_dyn"]
SHORT = {"m_off": "reference\n(no adapt)", "uniform_static": "uniform\nSTATIC", "patch_static": "patch\nSTATIC",
         "dyn_tau250_frac0.1": "uniform\nDYNAMIC", "patch_dyn_weak": "patch DYN\nweak 2×",
         "patch_dyn_strong": "patch DYN\nstrong 4×", "patch_dyn_strong_shuffled": "patch DYN 4×\nSHUFFLED",
         "meanfield_dyn": "mean-field\nDYNAMIC"}
CRIT = [("recruited", "#4c72b0"), ("broadband", "#e8a33d"), ("desynchronized", "#c44e52"), ("high_energy", "#55a868")]


def main():
    j = json.load(open(os.path.join(OUT, "stage1_joint.json")))
    fig, ax = plt.subplots(1, 3, figsize=(17.0, 4.6),
                           gridspec_kw=dict(width_ratios=[2.0, 1.0, 1.05], wspace=0.28))

    # (1) per-criterion occupancy across the 2x2 + controls
    x = np.arange(len(ORDER)); w = 0.2
    for i, (c, col) in enumerate(CRIT):
        ax[0].bar(x + (i - 1.5) * w, [100 * j[a]["frac_by_criterion"][c] for a in ORDER], w, color=col, label=c)
    ax[0].axvspan(2.5, 5.5, color="0.93", zorder=0)
    for a_i, a in enumerate(ORDER):
        if j[a]["frac_target"] > 0:
            ax[0].text(a_i, 103, "TARGET", ha="center", fontsize=7, color="#2ca02c")
    ax[0].set_xticks(x); ax[0].set_xticklabels([SHORT[a] for a in ORDER], fontsize=6.4, rotation=18, ha="right")
    ax[0].set_ylabel("% of 1 s windows meeting the criterion"); ax[0].set_ylim(0, 112)
    ax[0].legend(fontsize=7, ncol=4, loc="upper center")
    ax[0].set_title("(1) did spatial organization move anything?\n"
                    "patch DYN ≈ its SHUFFLE ≈ uniform DYN · joint target 0/8", fontsize=9)

    # (2) inter-cell differences are load-bearing: mean-field kills the broadening
    pair = ["dyn_tau250_frac0.1", "meanfield_dyn"]
    xb = np.arange(2)
    ax[1].bar(xb - 0.18, [100 * j[a]["frac_by_criterion"]["broadband"] for a in pair], 0.34,
              color="#e8a33d", label="broadband")
    ax[1].bar(xb + 0.18, [100 * j[a]["frac_by_criterion"]["desynchronized"] for a in pair], 0.34,
              color="#c44e52", label="desynchronized")
    ax[1].set_xticks(xb); ax[1].set_xticklabels(["per-cell m\n(uniform DYNAMIC)", "population-mean m\n(mean-field)"], fontsize=7.5)
    ax[1].set_ylabel("% of windows"); ax[1].legend(fontsize=7)
    ax[1].set_title("(2) are inter-cell differences\nload-bearing?", fontsize=9)

    # (3) do regions take turns? (corrected metric)
    import src.topic4_mz_fcxr_heo3 as H3
    src0 = json.load(open(os.path.join(OUT, "stage0_source_space.json")))["arms"]
    alts, labs = [], []
    for a in ORDER:
        f = os.path.join(OUT, "arms", a + ".json")
        if os.path.exists(f):
            alts.append(json.load(open(f))["region_alternation"])
        elif a in src0:                                        # reused arms audited in H3.0
            alts.append(H3.region_alternation(src0[a]["rows"]))
        else:
            continue
        labs.append(SHORT[a])
    ax[2].barh(np.arange(len(alts)), alts, color=["#c44e52" if v > 0 else "#4c72b0" for v in alts])
    ax[2].axvline(0, c="0.3", lw=0.8)
    ax[2].set_yticks(np.arange(len(alts))); ax[2].set_yticklabels(labs, fontsize=6.8)
    ax[2].set_xlim(-1, 1.05); ax[2].set_xlabel("corr(core-source rate, core-sink rate)")
    ax[2].set_title("(3) do the two cores TAKE TURNS?\nall positive → no alternation", fontsize=9)
    ax[2].text(0.55, len(alts) - 0.5, "common drive →", fontsize=6.5, color="0.35")
    ax[2].text(-0.95, len(alts) - 0.5, "← alternation", fontsize=6.5, color="0.35")

    fig.text(0.5, 0.005, "FCXR-HEO3 H3.1 — patchy recovery time (K load held fixed per cell) does not create alternation or a sustainable broadened state",
             ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "figures", "stage1_2x2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[h3.1] wrote figures/stage1_2x2.png")


if __name__ == "__main__":
    main()
