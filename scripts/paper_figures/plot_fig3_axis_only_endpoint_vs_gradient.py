#!/usr/bin/env python3
"""Axis-only figure: endpoint vs gradient-primary seven-band field concordance.

Panel A: seven bands, per-subject Δ (D - median_k N) for gradient (blue) and endpoint
(orange) side by side, cohort bar per axis, star = that axis' own seven-band maxT pFWER.
Panel B: the DIRECT per-band paired margin contrast (endpoint - gradient) per subject,
cohort bar, star = direct paired Wilcoxon p. Caption states the axis+routing confound
and gradient-primary / endpoint-sensitivity.
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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

_ROOT = Path(__file__).resolve().parents[2]
MS = _ROOT / "results/topic5_ictal_recruitment/field_concordance_grid_method_sensitivity/n161_subject_fixed"
EP = _ROOT / "results/topic5_ictal_recruitment/field_concordance_grid_endpoint_axis/n161_endpoint"
CMP = _ROOT / "results/topic5_ictal_recruitment/field_concordance_grid_endpoint_axis"
STAGE = _ROOT / "results/paper-ready-figure/fig3_ictal_field_concordance_grid_method_sensitivity/axis_only"
BANDS = ["delta_HYP_slow", "theta_preictal_PAC", "alpha_sharp_leq13", "beta_LVFA_low",
         "gamma_LVFA", "hg_low_ripple", "ripple_high"]
LAB = {"delta_HYP_slow": "δ", "theta_preictal_PAC": "θ", "alpha_sharp_leq13": "α",
       "beta_LVFA_low": "β", "gamma_LVFA": "γ", "hg_low_ripple": "R", "ripple_high": "FR"}
COL_G, COL_E = "#4c78a8", "#e08214"


def _star(p):
    return "*" if (np.isfinite(p) and p < 0.05) else "n.s."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradient-dir", default=str(MS))
    ap.add_argument("--endpoint-dir", default=str(EP))
    ap.add_argument("--compare-dir", default=str(CMP))
    ap.add_argument("--stage", default=str(STAGE))
    args = ap.parse_args()
    gs = pd.read_csv(Path(args.gradient_dir) / "multiband_subject.csv")
    es = pd.read_csv(Path(args.endpoint_dir) / "multiband_subject.csv")
    gc = pd.read_csv(Path(args.gradient_dir) / "multiband_cohort.csv").set_index("band")
    ec = pd.read_csv(Path(args.endpoint_dir) / "multiband_cohort.csv").set_index("band")
    pb = pd.read_csv(Path(args.compare_dir) / "endpoint_package_vs_gradient_primary_per_band.csv").set_index("band")
    summ = json.loads((Path(args.compare_dir) / "endpoint_package_vs_gradient_primary_summary.json").read_text())
    merged = gs.merge(es, on=["band", "subject"], suffixes=("_grad", "_ep"))
    merged["diff"] = merged["delta_ep"] - merged["delta_grad"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 6.0))

    # Panel A: gradient vs endpoint per-subject Δ, side by side
    for bi, b in enumerate(BANDS):
        for j, (df, col, cc) in enumerate([(gs, COL_G, gc), (es, COL_E, ec)]):
            vals = df[df.band == b].delta.dropna().values
            x = bi + (-0.19 if j == 0 else 0.19)
            if len(vals) >= 2 and np.ptp(vals) > 0:
                vp = axA.violinplot([vals], positions=[x], widths=0.33, showextrema=False)
                vp["bodies"][0].set_facecolor(col); vp["bodies"][0].set_alpha(0.35)
                vp["bodies"][0].set_edgecolor("gray")
            axA.scatter(x + np.random.default_rng(bi * 2 + j).uniform(-0.05, 0.05, len(vals)),
                        vals, s=12, c=col, alpha=0.7, zorder=4)
            coh = float(cc.loc[b, "delta_cohort_median"])
            axA.hlines(coh, x - 0.15, x + 0.15, color="k", lw=2.2, zorder=6)
            p = float(cc.loc[b, "seven_band_maxt_pfwer"])
            axA.text(x, axA.get_ylim()[1], _star(p), ha="center", va="top",
                     color=col if p < 0.05 else "gray", fontsize=13 if p < 0.05 else 8, weight="bold")
    axA.axhline(0, color="gray", lw=0.7)
    axA.set_xticks(range(len(BANDS))); axA.set_xticklabels([LAB[b] for b in BANDS])
    axA.set_ylabel("R3 grid-field concordance − all-contact null median\n(subject-level Δ)")
    axA.set_title("A · seven-band Δ: gradient (primary) vs endpoint axis\nstars = each axis' own seven-band maxT pFWER", fontsize=10)
    axA.legend(handles=[Patch(fc=COL_G, alpha=0.4, label="gradient (shared-else-own, primary)"),
                        Patch(fc=COL_E, alpha=0.4, label="endpoint (per-template A/B, sensitivity)"),
                        Line2D([0], [0], color="k", lw=2.2, label="cohort Δ median")],
               loc="lower left", fontsize=7.5, frameon=False)
    axA.spines[["top", "right"]].set_visible(False)

    # Panel B: direct per-band margin contrast; stars use the HOLM-CORRECTED p (0/7)
    for bi, b in enumerate(BANDS):
        d = merged[merged.band == b]["diff"].dropna().values
        p = float(pb.loc[b, "direct_holm_p"])
        if len(d) >= 2 and np.ptp(d) > 0:
            vp = axB.violinplot([d], positions=[bi], widths=0.7, showextrema=False)
            vp["bodies"][0].set_facecolor("#c44e52" if p < 0.05 else "#cfcfcf")
            vp["bodies"][0].set_alpha(0.4); vp["bodies"][0].set_edgecolor("gray")
        axB.scatter(bi + np.random.default_rng(bi).uniform(-0.08, 0.08, len(d)), d,
                    s=14, c="#333", alpha=0.7, zorder=4)
        axB.hlines(np.median(d), bi - 0.3, bi + 0.3, color="k", lw=2.4, zorder=6)
        # nominal (uncorrected) p shown small below the star for transparency
        raw = float(pb.loc[b, "direct_raw_wilcoxon_p"])
        axB.text(bi, axB.get_ylim()[1], _star(p), ha="center", va="top",
                 color="#c44e52" if p < 0.05 else "gray", fontsize=13 if p < 0.05 else 9, weight="bold")
        axB.text(bi, axB.get_ylim()[1] - 0.06 * (axB.get_ylim()[1] - axB.get_ylim()[0]),
                 f"nom {raw:.2f}", ha="center", va="top", fontsize=6.2, color="0.55")
    axB.axhline(0, color="gray", lw=0.7)
    axB.set_xticks(range(len(BANDS))); axB.set_xticklabels([LAB[b] for b in BANDS])
    axB.set_ylabel("margin difference  endpoint − gradient\n(per-subject, paired)")
    ov = summ["overall_band_to_subject_folded"]
    own = summ["closest_to_pure_axis_own_fallback_stratum"]
    axB.set_title("B · direct endpoint-package − gradient-primary contrast\n"
                  "stars = Holm-corrected direct p (0/7 significant; sign-flip maxT also 0/7)", fontsize=9.5)
    axB.text(0.02, 0.03, f"overall folded {ov['median']:+.4f}, {ov['n_endpoint_gt_gradient']}/{ov['n_subjects']}, p={ov['wilcoxon_p']:.2f} (n.s.)\n"
             f"own-fallback (closest to pure axis) {own['median']:+.5f}, p={own['wilcoxon_p']:.2f} (≈0)",
             transform=axB.transAxes, va="bottom", ha="left", fontsize=7, color="0.3")
    axB.spines[["top", "right"]].set_visible(False)

    fig.suptitle("endpoint-PACKAGE vs gradient-primary sensitivity (NOT axis-only) · seven bands · onset 0–10 s (n=17/167, N=161)",
                 fontsize=11.5)
    fig.text(0.5, 0.005, "endpoint changes axis + routing (per-template A/B vs shared-else-own) + sigma value together — a PACKAGE "
             "contrast, not the axis alone. No band survives the direct seven-band correction; the nominal β/α gains are confined to "
             "the shared stratum and vanish in own-fallback (closest to pure axis). Gradient stays primary; endpoint is a sensitivity.",
             ha="center", va="bottom", fontsize=7.4, color="0.3")
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    stage = Path(args.stage); stage.mkdir(parents=True, exist_ok=True)
    fig.savefig(stage / "endpoint_package_vs_gradient_primary.png", dpi=200, bbox_inches="tight")
    fig.savefig(stage / "endpoint_package_vs_gradient_primary.pdf", bbox_inches="tight")
    plt.close(fig)
    (stage / "endpoint_package_vs_gradient_primary_metadata.json").write_text(json.dumps({
        "figure": "endpoint_package_vs_gradient_primary",
        "not_axis_only": summ["NOT_axis_only"],
        "identical_pipeline_verified": summ["identical_pipeline_verified"],
        "panelA": "seven-band per-subject Δ, gradient (primary) vs endpoint; stars = each axis' own seven-band maxT pFWER",
        "panelB": "direct per-band endpoint-package − gradient-primary margin contrast; stars = Holm-corrected p (0/7); "
                  "nominal uncorrected p shown small; sign-flip maxT also 0/7",
        "overall_folded": summ["overall_band_to_subject_folded"],
        "own_fallback_stratum": summ["closest_to_pure_axis_own_fallback_stratum"],
        "shared_stratum": summ["routing_changed_shared_stratum"],
        "conclusion": summ["conclusion"]}, indent=2, default=str))
    print(f"[fig] wrote {stage/'endpoint_package_vs_gradient_primary.png'}")


if __name__ == "__main__":
    main()
