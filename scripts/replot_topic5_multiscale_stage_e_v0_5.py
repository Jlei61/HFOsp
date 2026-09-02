#!/usr/bin/env python3
"""Target-free visual-only repair of the Stage-E diagnostic figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
ARMS = (
    "L0_LOCAL_ONLY",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2M_MACRO_MATCHED_RANDOM_LR",
    "L3_LOCAL_PLUS_LEARNED_LR",
    "C_L3_ORDER_SHUFFLED",
)
LABELS = ("Local", "+ nearby", "+ matched\nnonlocal", "+ selected\nnonlocal", "Suffix null")
COLORS = ("#7d878c", "#6b97a5", "#b9904e", "#bd4449", "#b9bdbf")


def significance(p_value: float) -> str:
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < .05:
        return "*"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    if (out / "TARGET_UNSEAL_AUTHORIZATION.json").exists():
        raise RuntimeError("Stage-E visual repair must remain target-free")
    patient = pd.read_csv(out / "INTERICTAL_PER_PATIENT.csv")
    contrasts = pd.read_csv(out / "INTERICTAL_PATIENT_CONTRASTS.csv")
    summary = json.loads((out / "INTERICTAL_V0_5_SUMMARY.json").read_text())
    pivot = patient.pivot(index="subject", columns="arm")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11.0, "axes.labelsize": 12.5,
        "xtick.labelsize": 10.0, "ytick.labelsize": 10.0, "axes.linewidth": .85,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    figure, axes = plt.subplots(
        1, 4, figsize=(14.6, 3.35), gridspec_kw={"wspace": .66}, facecolor="white",
    )

    x = np.arange(len(ARMS))
    for subject in pivot.index:
        axes[0].plot(
            x, [pivot["test_contact_nll"][arm][subject] for arm in ARMS],
            color="#cbd0d2", lw=.55, alpha=.66,
        )
    medians = [float(pivot["test_contact_nll"][arm].median()) for arm in ARMS]
    axes[0].plot(x, medians, color="#171717", lw=1.8)
    axes[0].scatter(x, medians, c=COLORS, s=32, zorder=3)
    axes[0].set_xticks(x, LABELS, rotation=33, ha="right")
    axes[0].set_ylabel("Held-out contact NLL")

    contrast_labels = ("L3_vs_L0_distal", "L3_vs_L1_distal", "L3_vs_L2m_distal")
    contrast_names = ("Local", "+ nearby", "Matched\nnonlocal")
    for position, label in enumerate(contrast_labels):
        values = contrasts.loc[contrasts.contrast == label, "gain_nats"].to_numpy(float)
        axes[1].scatter(
            position + np.linspace(-.12, .12, len(values)), values,
            s=18, color="#9ca3a6", alpha=.78,
        )
        axes[1].plot(
            [position - .18, position + .18], [np.median(values)] * 2,
            color="#bd4449", lw=2.3,
        )
    axes[1].axhline(0, color="#62686b", lw=.8, ls="--")
    axes[1].set_xticks(range(3), contrast_names)
    axes[1].set_ylabel("Selected nonlocal gain\n(distal NLL, nats)")

    primary = pd.DataFrame(summary["primary_rows"])
    plotted_j = np.sqrt(np.maximum(primary.J_lat_exceedance_burden.to_numpy(float), 0))
    axes[2].scatter(
        plotted_j, primary.gain_nats, s=30,
        c=np.where(primary.geometry_2d, "#315f8a", "#d28b2d"),
        edgecolors="white", linewidths=.4,
    )
    axes[2].axhline(0, color="#62686b", lw=.8, ls="--")
    tick_values = np.asarray([0, .01, .05, .10, .25, .60])
    axes[2].set_xticks(np.sqrt(tick_values), ["0", ".01", ".05", ".10", ".25", ".60"])
    axes[2].set_xlim(-.025, np.sqrt(max(.6, float(primary.J_lat_exceedance_burden.max()))) + .035)
    axes[2].set_xlabel("Cross-fitted nonlocality J\n(sqrt scale)")
    axes[2].set_ylabel("Selected − matched\ndistal gain (nats)")

    suffix = contrasts.loc[
        contrasts.contrast == "L3_vs_suffix_all", "gain_nats"
    ].to_numpy(float)
    axes[3].scatter(
        np.linspace(-.12, .12, len(suffix)), suffix, s=19,
        color="#8f9699", alpha=.82,
    )
    axes[3].plot([-.20, .20], [np.median(suffix)] * 2, color="#171717", lw=2.3)
    axes[3].axhline(0, color="#62686b", lw=.8, ls="--")
    axes[3].set_xlim(-.35, .35)
    axes[3].set_xticks([0], ["True suffix\nvs reassigned"])
    axes[3].set_ylabel("Order-specific gain (nats)")
    order_p = summary["comparisons"]["L3_vs_suffix_all"]["wilcoxon_p_greater"]
    marker = significance(float(order_p))
    if marker:
        axes[3].text(
            .5, .98, marker, transform=axes[3].transAxes, ha="center", va="top",
            fontsize=13, fontweight="bold",
        )

    for label, axis in zip("ABCD", axes):
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            -.19, 1.08, label, transform=axis.transAxes, fontsize=15,
            fontweight="bold", va="top",
        )
    stem = out / "figures/stage_e_v0_5_interictal_multiscale_scaffold"
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight")
    plt.close(figure)
    record = {
        "status": "TARGET_FREE_VISUAL_REPAIR",
        "target_values_read": False,
        "scientific_estimands_unchanged": True,
        "changes": [
            "Panel C uses sqrt display transform with explicit original-value ticks",
            "Panel D displays significance stars from the frozen patient-level test",
            "axes and labels enlarged; no new statistic added",
        ],
    }
    (out / "STAGE_E_FIGURE_VISUAL_REPAIR.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
