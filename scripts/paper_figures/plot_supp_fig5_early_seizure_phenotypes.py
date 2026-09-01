#!/usr/bin/env python3
"""Build Supplementary Figure 5 from a cohort pie and the accepted gamma panel."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import plot_topic5_early_spectral_phenotypes as phenotype_plot  # noqa: E402


PHENOTYPE_ROOT = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/"
    "early_spectral_phenotype"
)
SUMMARY_CSV = PHENOTYPE_ROOT / "cohort_spectral_simple_summary.csv"
STATE_CSV = PHENOTYPE_ROOT / "per_seizure_spectral_overlap_state.csv"
OUT_ROOT = ROOT / "results/paper-ready-figure/supp_fig5_early_seizure_phenotypes"
FIG_DIR = OUT_ROOT / "figures"
GAMMA_PNG = FIG_DIR / "epilepsiae_635_seizure_07_raw_spectral_context.png"
GAMMA_JSON = FIG_DIR / "epilepsiae_635_seizure_07_raw_spectral_context_summary.json"
SHORT_LABELS = {
    "broadband_1_150": "Broadband\n1–150 Hz",
    "gamma_nonbroadband": "Gamma-dominant\n30–80 Hz",
    "low_frequency_only": "Low-frequency only\n1–13 Hz",
    "other": "Other patterns",
}


def _cohort_rows() -> list[dict]:
    frame = pd.read_csv(SUMMARY_CSV)
    frame = frame[frame["dataset"] == "combined_descriptive"].copy()
    by_category = frame.set_index("phenotype")
    rows = []
    for category in phenotype_plot.SIMPLE_CATEGORIES:
        row = by_category.loc[category]
        rows.append(
            {
                "category": category,
                "label": SHORT_LABELS[category],
                "n_seizures": int(row["n_seizures"]),
                "fraction": float(row["fraction_seizures"]),
                "color": phenotype_plot.SIMPLE_COLORS[category],
                "denominator_seizures": int(row["denominator_seizures"]),
                "denominator_subjects": int(row["denominator_subjects"]),
            }
        )
    if not np.isclose(sum(row["fraction"] for row in rows), 1.0):
        raise RuntimeError("Early spectral phenotype fractions do not sum to one")
    return rows


def _gamma_context() -> tuple[dict, dict]:
    if not GAMMA_PNG.exists() or not GAMMA_JSON.exists():
        raise FileNotFoundError(
            "Accepted gamma panel or summary is missing; rerun "
            "plot_fig3_raw_spectral_context.py with the gamma profile first"
        )
    summary = json.loads(GAMMA_JSON.read_text(encoding="utf-8"))
    states = pd.read_csv(STATE_CSV)
    matched = states[
        (states["subject"].astype(str) == str(summary["subject"]))
        & (states["seizure_idx"].astype(int) == int(summary["seizure_idx"]))
    ]
    if len(matched) != 1:
        raise RuntimeError(
            "Gamma example must map to exactly one per-seizure phenotype row"
        )
    state = matched.iloc[0].to_dict()
    if state["simple_phenotype"] != "gamma_nonbroadband":
        raise RuntimeError(
            "Accepted gamma panel is not classified as gamma_nonbroadband"
        )
    return summary, state


def _draw_composition_pie(ax: plt.Axes, rows: list[dict]) -> None:
    fractions = [row["fraction"] for row in rows]
    wedges, _ = ax.pie(
        fractions,
        colors=[row["color"] for row in rows],
        startangle=90,
        counterclock=False,
        radius=0.90,
        wedgeprops={"edgecolor": "white", "linewidth": 0.9},
    )
    label_positions = {
        "broadband_1_150": (1.03, 0.43, "left"),
        "gamma_nonbroadband": (0.48, -0.98, "left"),
        "low_frequency_only": (-1.02, -0.72, "right"),
        "other": (-1.02, 0.45, "right"),
    }
    for wedge, row in zip(wedges, rows):
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
        x, y = np.cos(angle), np.sin(angle)
        x_text, y_text, horizontal_alignment = label_positions[row["category"]]
        ax.annotate(
            f"{row['label']}\n{row['n_seizures']} ({100 * row['fraction']:.1f}%)",
            xy=(0.78 * x, 0.78 * y),
            xytext=(x_text, y_text),
            ha=horizontal_alignment,
            va="center",
            fontsize=6.7,
            linespacing=1.05,
            color="black",
            arrowprops={
                "arrowstyle": "-",
                "color": "0.35",
                "lw": 0.55,
                "shrinkA": 0,
                "shrinkB": 0,
                "connectionstyle": "arc3,rad=0.0",
            },
            annotation_clip=False,
            clip_on=False,
        )
    ax.text(
        0.0,
        -1.31,
        f"{rows[0]['denominator_seizures']} seizures · "
        f"{rows[0]['denominator_subjects']} patients",
        ha="center",
        va="top",
        fontsize=7.0,
        color="black",
    )
    ax.text(
        0.0,
        1.22,
        "Early ictal spectral phenotypes",
        ha="center",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color="black",
    )
    ax.text(
        0.0,
        1.11,
        "mutually exclusive classification",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="0.25",
    )
    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.40, 1.32)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = _cohort_rows()
    gamma_summary, gamma_state = _gamma_context()
    gamma_image = plt.imread(GAMMA_PNG)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(14.25, 4.25), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[2.55, 11.70],
        wspace=0.035,
        left=0.025,
        right=0.992,
        bottom=0.055,
        top=0.945,
    )
    ax_pie = fig.add_subplot(grid[0, 0])
    ax_gamma = fig.add_subplot(grid[0, 1])
    _draw_composition_pie(ax_pie, rows)
    ax_gamma.imshow(gamma_image, interpolation="none")
    ax_gamma.set_aspect("equal", adjustable="box")
    ax_gamma.set_anchor("W")
    ax_gamma.axis("off")

    fig.canvas.draw()
    panel_y = 0.972
    for label, axis in (("A", ax_pie), ("B", ax_gamma)):
        pos = axis.get_position()
        fig.text(
            pos.x0 - 0.018,
            panel_y,
            label,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

    stem = FIG_DIR / "supp_fig5_phenotypes_and_gamma_example"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    denominator_seizures = int(rows[0]["denominator_seizures"])
    denominator_subjects = int(rows[0]["denominator_subjects"])
    category_counts = {row["category"]: int(row["n_seizures"]) for row in rows}
    category_fractions = {row["category"]: float(row["fraction"]) for row in rows}
    caption_title = "Early ictal spectral phenotypes and a gamma-dominant example."
    caption_body = (
        f"**A,** Mutually exclusive early ictal spectral phenotypes across "
        f"{denominator_seizures} seizures from {denominator_subjects} patients: "
        f"broadband 1\u2013150 Hz "
        f"({category_counts['broadband_1_150']}, "
        f"{100 * category_fractions['broadband_1_150']:.1f}%), "
        f"gamma-dominant 30\u201380 Hz "
        f"({category_counts['gamma_nonbroadband']}, "
        f"{100 * category_fractions['gamma_nonbroadband']:.1f}%), "
        f"low-frequency only 1\u201313 Hz "
        f"({category_counts['low_frequency_only']}, "
        f"{100 * category_fractions['low_frequency_only']:.1f}%) and other "
        f"patterns ({category_counts['other']}, "
        f"{100 * category_fractions['other']:.1f}%). **B,** Representative "
        f"gamma-dominant seizure (E20, seizure 7), showing "
        f"common-average-referenced SEEG traces, the baseline-normalized "
        f"time\u2013frequency representation from contact HRB1 and band-power "
        f"trajectories for 1\u201330, 30\u201380, 80\u2013150 and 1\u2013150 Hz. "
        f"Time is aligned to clinical onset at 0 s; blue shading marks the "
        f"\u2212120 to \u221290 s baseline and red shading the 0\u201310 s early "
        f"clinical window. This single-seizure panel illustrates the signal "
        f"morphology of the gamma-dominant category and was not used for "
        f"cohort-level inference."
    )
    metadata = {
        "figure": "Supplementary Figure 5",
        "caption": (
            f"Supplementary Fig. 5 | {caption_title} "
            f"{caption_body.replace('**', '')}"
        ),
        "panel_a": {
            "source": str(SUMMARY_CSV.relative_to(ROOT)),
            "classification": "mutually exclusive simple early spectral phenotype",
            "denominator_seizures": rows[0]["denominator_seizures"],
            "denominator_subjects": rows[0]["denominator_subjects"],
            "categories": rows,
        },
        "panel_b": {
            "source_panel": str(GAMMA_PNG.relative_to(ROOT)),
            "source_summary": str(GAMMA_JSON.relative_to(ROOT)),
            "classification_source": str(STATE_CSV.relative_to(ROOT)),
            "producer_contract": (
                "scripts/paper_figures/plot_fig3_raw_spectral_context.py; "
                "accepted panel embedded without altering its internal layout"
            ),
            "patient_id": gamma_summary["public_patient_label"],
            "seizure_idx": gamma_summary["seizure_idx"],
            "spectral_profile": "gamma",
            "simple_phenotype": gamma_state["simple_phenotype"],
            "classification_reason": gamma_state["classification_reason"],
            "role": "representative gamma_nonbroadband early-onset pattern",
            "source_pixel_dimensions": [
                int(gamma_image.shape[1]),
                int(gamma_image.shape[0]),
            ],
        },
        "layout_note": (
            "Panel A is a direct-labelled pie; Panel B preserves the accepted "
            "gamma PNG aspect ratio and internal geometry."
        ),
        "outputs": {
            "png": str(stem.with_suffix(".png").relative_to(ROOT)),
            "pdf": str(stem.with_suffix(".pdf").relative_to(ROOT)),
        },
    }
    (OUT_ROOT / "supp_fig5_phenotypes_gamma_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (FIG_DIR / "README.md").write_text(
        "### supp_fig5_phenotypes_and_gamma_example.png\n\n"
        f"**Supplementary Fig. 5 | {caption_title}**\n\n"
        f"{caption_body}\n\n"
        "**关注点**：A 是描述性分型；B 只说明 gamma_nonbroadband 类的可见"
        "信号形态，不承担队列统计或机制结论。\n",
        encoding="utf-8",
    )
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
