#!/usr/bin/env python3
"""§六/§四 figure — unified multiband estimand across the R2/R3 x sigma-policy 2x2.

Reads the 4-cell tensors + pFWER table written by
`scripts/run_topic5_multiband_4cell_estimand.py` and renders a 2x2 panel grid
(rows = readout R3/R2, cols = policy subject_fixed/frozen_per_model). Each panel
is the seven-band per-subject Eobs (D - median_k N) with the cohort bar = Tobs
and a star from the JOINT 28-family maxT pFWER. A star in one cell but not
another is NOT evidence that the methods differ (see summary claim boundary).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
DEF = _ROOT / "results/topic5_ictal_recruitment/field_concordance_grid_method_sensitivity"
STAGE = _ROOT / "results/paper-ready-figure/fig3_ictal_field_concordance_grid_method_sensitivity"
BAND_LABELS = {"delta_HYP_slow": "δ", "theta_preictal_PAC": "θ", "alpha_sharp_leq13": "α",
               "beta_LVFA_low": "β", "gamma_LVFA": "γ", "hg_low_ripple": "R", "ripple_high": "FR"}
SIG_C, NS_C = "#c44e52", "#cfcfcf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEF))
    ap.add_argument("--stage", default=str(STAGE))
    args = ap.parse_args()
    root = Path(args.root)
    npz = np.load(root / "multiband_4cell_tensors.npz", allow_pickle=True)
    Eobs = npz["Eobs"]                       # (s,b,m)
    Tobs = npz["Tobs"]                       # (b,m)
    cells = [str(c) for c in npz["cells"]]
    bands = [str(b) for b in npz["bands"]]
    est = pd.read_csv(root / "multiband_4cell_estimand.csv")
    summ = json.loads((root / "multiband_4cell_estimand_summary.json").read_text())

    order = [("R3", "subject_fixed"), ("R3", "frozen_per_model"),
             ("R2", "subject_fixed"), ("R2", "frozen_per_model")]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharey=True)
    for ax, (meth, pol) in zip(axes.ravel(), order):
        cell = f"{meth}::{pol}"
        mi = cells.index(cell)
        for bi, b in enumerate(bands):
            vals = Eobs[:, bi, mi]
            vals = vals[np.isfinite(vals)]
            p = float(est[(est.cell == cell) & (est.band == b)].joint_28_family_maxt_pfwer.iloc[0])
            marked = p < 0.05
            color = SIG_C if marked else NS_C
            if len(vals) >= 2 and np.ptp(vals) > 0:
                vp = ax.violinplot([vals], positions=[bi], widths=0.8,
                                   showmedians=False, showextrema=False)
                vp["bodies"][0].set_facecolor(color)
                vp["bodies"][0].set_edgecolor("gray")
                vp["bodies"][0].set_alpha(0.4)
            jit = np.random.default_rng(bi).uniform(-0.08, 0.08, len(vals))
            ax.scatter(bi + jit, vals, s=16, c="#333", alpha=0.7, zorder=4)
            ax.hlines(Tobs[bi, mi], bi - 0.32, bi + 0.32, color="k", lw=2.4, zorder=6)
            ax.annotate("*" if marked else "n.s.", (bi, ax.get_ylim()[1]),
                        ha="center", va="top", fontsize=15 if marked else 9,
                        color=SIG_C if marked else "gray", weight="bold" if marked else "normal")
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([BAND_LABELS[b] for b in bands])
        ax.set_title(f"{meth} · {pol}", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Eobs = D − median$_k$ N\n(subject-level Δ)", fontsize=10)
    fig.suptitle("Unified multiband estimand · R2/R3 × smoothing policy · "
                 f"stars = joint 28-family maxT pFWER<0.05 (n={summ['n_subjects']})", fontsize=12)
    # attribution comes from DIRECT paired contrasts, NOT "R3 star vs R2 no star"
    fc = summ.get("formal_contrasts", {})
    if fc:
        def _p(k):
            v = fc.get(k, {})
            return f"{v.get('median_effect', float('nan')):+.3f} (paired p={v.get('paired_wilcoxon_p', float('nan')):.3f})"
        fig.text(0.5, 0.005,
                 "attribution = direct paired contrasts (NOT star deltas): "
                 f"readout R3−R2 {_p('readout_R3_minus_R2')} · "
                 f"sigma frozen−subject_fixed {_p('sigma_frozen_minus_subjectfixed')} · "
                 f"interaction {_p('interaction_readout_x_sigma')}",
                 ha="center", va="bottom", fontsize=8.2, color="0.25")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    stage = Path(args.stage) / "figures"
    stage.mkdir(parents=True, exist_ok=True)
    fig.savefig(stage / "multiband_4cell_sigma_policy.png", dpi=200, bbox_inches="tight")
    fig.savefig(stage / "multiband_4cell_sigma_policy.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {stage/'multiband_4cell_sigma_policy.png'}")
    print(f"joint-28 significant: {summ['n_joint_significant_0p05']}; "
          f"per-cell significant: {summ['n_per_cell_significant_0p05']}")


if __name__ == "__main__":
    main()
