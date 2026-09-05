#!/usr/bin/env python3
"""Build Supplementary Figure 2: SOZ ROC and raw-vs-synchronized AUC.

Panel A reuses the accepted Figure-1 refined-event ROC contract and adds a
subject-level AUC inset.  Panel B returns the legacy paper's paired comparison,
but recomputes it from the current artifacts on a shared subject denominator.
Here "synchronized" means refined HFO counts inside accepted group-event
windows; it is not a separate detector.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_figures import plot_fig1_interictal_hfo_temporal_scaffold as fig1  # noqa: E402
from paper_figures.patient_public_labels import public_patient_label  # noqa: E402
import plot_refine_soz_validation as refine_plot  # noqa: E402
from src.plot_style import COL_YQ  # noqa: E402
from src.supplementary_figure_style import (  # noqa: E402
    ANNOTATION_SIZE,
    AXIS_LABEL_SIZE,
    PANEL_LETTER_SIZE,
    SIGNIFICANCE_SIZE,
    TICK_LABEL_SIZE,
    apply_supplementary_rcparams,
    normalize_axis_text,
)


OUT_ROOT = ROOT / "results/paper-ready-figure/supp_fig2_soz_auc"
FIG_DIR = OUT_ROOT / "figures"
PAIR_CSV = OUT_ROOT / "subject_auc_raw_vs_synchronized.csv"
COLORS = {"yuquan": COL_YQ, "epilepsiae": fig1.EPI_ROC_COLOR}
LABELS = {"yuquan": "Yuquan", "epilepsiae": "Epilepsiae"}


def _load_auc_pairs(dataset: str) -> list[dict]:
    """Load raw/refined AUC on the same configured, SOZ-labelled subjects."""
    soz_all = refine_plot.load_soz_channels(fig1.SOZ_JSON[dataset])
    params = refine_plot.load_subject_params(fig1.PARAMS_JSON)
    configured = {
        key for key in params.get(dataset, {}) if not str(key).startswith("_")
    }
    rows: list[dict] = []
    for subject in sorted(configured):
        subject_dir = fig1.HFO_ROOT / subject
        if not subject_dir.is_dir() or not soz_all.get(subject):
            continue
        raw_counts, raw_names = refine_plot.load_raw_counts(subject_dir)
        sync_counts, sync_names = refine_plot.load_refine_counts(subject_dir)
        if (
            raw_counts is None
            or raw_names is None
            or sync_counts is None
            or sync_names is None
        ):
            continue
        raw_soz, _ = refine_plot.classify_channels_soz(
            raw_names, soz_all[subject]
        )
        sync_soz, _ = refine_plot.classify_channels_soz(
            sync_names, soz_all[subject]
        )
        raw_auc, _, _ = refine_plot.compute_auc(
            raw_counts, raw_soz, len(raw_names)
        )
        sync_auc, _, _ = refine_plot.compute_auc(
            sync_counts, sync_soz, len(sync_names)
        )
        if np.isfinite(raw_auc) and np.isfinite(sync_auc):
            rows.append(
                {
                    "dataset": dataset,
                    "patient_id": public_patient_label(dataset, subject),
                    "raw_auc": float(raw_auc),
                    "synchronized_auc": float(sync_auc),
                    "delta_auc": float(sync_auc - raw_auc),
                    "n_raw_channels": int(len(raw_names)),
                    "n_synchronized_channels": int(len(sync_names)),
                    "n_raw_soz": int(len(raw_soz)),
                    "n_synchronized_soz": int(len(sync_soz)),
                }
            )
    return rows


def _auc_inset(ax: plt.Axes, rows: list[dict], color: str) -> None:
    """Compact vertical subject-AUC distribution inside a ROC axis."""
    aucs = np.asarray([row["auc"] for row in rows], dtype=float)
    inset = ax.inset_axes([0.68, 0.09, 0.27, 0.38])
    rng = np.random.default_rng(23)
    jitter = rng.uniform(-0.055, 0.055, aucs.size)
    inset.scatter(
        jitter,
        aucs,
        s=13,
        color=color,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    mean = float(np.mean(aucs))
    inset.hlines(mean, -0.13, 0.13, color="black", lw=1.6, zorder=4)
    inset.set_xlim(-0.16, 0.16)
    inset.set_ylim(0.0, 1.02)
    inset.set_xticks([])
    inset.set_yticks([0, 0.5, 1.0])
    inset.set_ylabel("AUC", fontsize=AXIS_LABEL_SIZE, labelpad=1)
    inset.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, length=2, pad=1)
    inset.spines[["top", "right", "bottom"]].set_visible(False)
    inset.spines["left"].set_linewidth(0.6)


def _lighten(color: str, amount: float = 0.58) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple(rgb + (1.0 - rgb) * amount)


def _plot_paired_auc(
    ax: plt.Axes, dataset: str, rows: list[dict]
) -> dict:
    color = COLORS[dataset]
    raw_color = _lighten(color)
    raw = np.asarray([row["raw_auc"] for row in rows], dtype=float)
    sync = np.asarray([row["synchronized_auc"] for row in rows], dtype=float)
    for idx, (before, after) in enumerate(zip(raw, sync)):
        linestyle = "-" if after >= before else "--"
        ax.plot(
            [0, 1],
            [before, after],
            color="0.35",
            lw=0.65,
            ls=linestyle,
            alpha=0.72,
            zorder=1,
        )
        ax.scatter(
            [0, 1],
            [before, after],
            s=18,
            color=[raw_color, color],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    means = [float(np.mean(raw)), float(np.mean(sync))]
    ax.bar(
        [0, 1],
        means,
        width=0.58,
        color=[raw_color, color],
        alpha=0.62,
        edgecolor="none",
        zorder=0,
    )
    ax.hlines(means, [-0.23, 0.77], [0.23, 1.23], color="black", lw=1.4)
    try:
        test = wilcoxon(sync, raw, alternative="two-sided", method="auto")
        p_value = float(test.pvalue)
        statistic = float(test.statistic)
    except ValueError:
        p_value = float("nan")
        statistic = float("nan")
    y = min(1.025, max(float(np.max(raw)), float(np.max(sync))) + 0.045)
    ax.plot([0, 0, 1, 1], [y - 0.012, y, y, y - 0.012], color="black", lw=0.8)
    if not np.isfinite(p_value):
        p_text = "n.s."
    elif p_value < 0.001:
        p_text = "***"
    elif p_value < 0.01:
        p_text = "**"
    elif p_value < 0.05:
        p_text = "*"
    else:
        p_text = "n.s."
    ax.text(
        0.5,
        y + 0.010,
        p_text,
        ha="center",
        va="bottom",
        fontsize=SIGNIFICANCE_SIZE,
        fontweight="bold" if "*" in p_text else "normal",
    )
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks([0, 1], ["Raw", "Synchronized"])
    ax.set_xlabel(LABELS[dataset], color=color, fontsize=AXIS_LABEL_SIZE, labelpad=4)
    ax.set_ylabel("Subject AUC", fontsize=AXIS_LABEL_SIZE, labelpad=4)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, length=2.5, width=0.7)
    fig1._style_axis(ax)
    ax.set_box_aspect(1.0)
    return {
        "dataset": dataset,
        "n_subjects_paired": int(len(rows)),
        "raw_mean_auc": means[0],
        "raw_median_auc": float(np.median(raw)),
        "synchronized_mean_auc": means[1],
        "synchronized_median_auc": float(np.median(sync)),
        "mean_delta_auc": float(np.mean(sync - raw)),
        "median_delta_auc": float(np.median(sync - raw)),
        "n_improved": int(np.sum(sync > raw)),
        "wilcoxon_two_sided_statistic": statistic,
        "wilcoxon_two_sided_p": p_value,
        "patients": [row["patient_id"] for row in rows],
    }


def _write_pair_csv(rows: list[dict]) -> None:
    with PAIR_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    fig1._apply_rcparams()
    apply_supplementary_rcparams()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    roc_rows = {
        dataset: fig1._load_refined_roc(dataset)
        for dataset in ("yuquan", "epilepsiae")
    }
    paired_rows = {
        dataset: _load_auc_pairs(dataset)
        for dataset in ("yuquan", "epilepsiae")
    }
    _write_pair_csv(paired_rows["yuquan"] + paired_rows["epilepsiae"])

    fig = plt.figure(figsize=(9.20, 2.85), facecolor="white")
    grid = fig.add_gridspec(
        1,
        5,
        width_ratios=[1.0, 1.0, 0.30, 1.0, 1.0],
        wspace=0.35,
        left=0.066,
        right=0.990,
        bottom=0.20,
        top=0.88,
    )
    axes_roc = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
    ]
    axes_pair = [
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[0, 4]),
    ]

    roc_summaries = {}
    for dataset, ax in zip(("yuquan", "epilepsiae"), axes_roc):
        roc_summaries[dataset] = fig1._plot_roc(
            ax, dataset, roc_rows[dataset]
        )
        for artist in list(ax.texts):
            if artist.get_text().startswith("mean AUC"):
                artist.remove()
        ax.set_title("")
        ax.set_xlabel("False-positive rate", fontsize=AXIS_LABEL_SIZE, labelpad=4)
        ax.set_ylabel("True-positive rate", fontsize=AXIS_LABEL_SIZE, labelpad=4)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, length=2.5, width=0.7)
        ax.set_box_aspect(1.0)
        ax.text(
            0.04,
            0.96,
            LABELS[dataset],
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=COLORS[dataset],
            fontsize=ANNOTATION_SIZE,
        )
        _auc_inset(ax, roc_rows[dataset], COLORS[dataset])
        roc_subjects = roc_summaries[dataset].pop("subjects")
        roc_summaries[dataset]["patients"] = [
            public_patient_label(dataset, subject) for subject in roc_subjects
        ]
    axes_roc[1].set_ylabel("")

    paired_summaries = {}
    for dataset, ax in zip(("yuquan", "epilepsiae"), axes_pair):
        paired_summaries[dataset] = _plot_paired_auc(
            ax, dataset, paired_rows[dataset]
        )
    axes_pair[1].set_ylabel("")
    fig.canvas.draw()
    first_axes = (axes_roc[0], axes_pair[0])
    panel_y = max(axis.get_position().y1 for axis in first_axes) + 0.050
    for label, ax in zip(("A", "B"), first_axes):
        pos = ax.get_position()
        fig.text(
            pos.x0 - 0.048,
            panel_y,
            label,
            ha="left",
            va="top",
            fontsize=PANEL_LETTER_SIZE,
            fontweight="bold",
        )
    for axis in (*axes_roc, *axes_pair):
        normalize_axis_text(axis)
    stem = FIG_DIR / "supp_fig2_soz_auc_raw_vs_synchronized"
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    fig.savefig(png, dpi=400, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    yq = paired_summaries["yuquan"]
    epi = paired_summaries["epilepsiae"]
    caption_title = (
        "SOZ localization of raw and group-event-filtered interictal HFO burden."
    )
    caption_body = (
        f"**A,** Subject-level receiver operating characteristic curves for "
        f"distinguishing clinically labelled seizure onset zone (SOZ) from "
        f"non-SOZ contacts using contact-wise group-event-filtered "
        f"high-frequency oscillation (HFO) burden in the Yuquan "
        f"({yq['n_subjects_paired']} patients) and Epilepsiae "
        f"({epi['n_subjects_paired']} patients) cohorts; grey lines denote "
        f"individual patients, coloured lines the cohort mean, shading the "
        f"s.e.m., and inset points the patient-level area under the curve "
        f"(AUC), with the black line indicating the mean. Mean AUC was "
        f"{yq['synchronized_mean_auc']:.3f} in Yuquan and "
        f"{epi['synchronized_mean_auc']:.3f} in Epilepsiae. "
        f"**B,** Paired patient-level AUC before (Raw) and after restricting "
        f"refined HFO detections to accepted population-event windows "
        f"(Synchronized); bars and black horizontal lines indicate cohort "
        f"means, and points joined by lines denote paired patients "
        f"(Yuquan, {yq['raw_mean_auc']:.3f} versus "
        f"{yq['synchronized_mean_auc']:.3f}; Epilepsiae, "
        f"{epi['raw_mean_auc']:.3f} versus "
        f"{epi['synchronized_mean_auc']:.3f}). Statistical significance was "
        f"assessed using a two-sided paired Wilcoxon test; *P < 0.05; n.s., "
        f"P \u2265 0.05."
    )
    metadata = {
        "figure": "Supplementary Figure 2",
        "caption": (
            f"Supplementary Fig. 2 | {caption_title} "
            f"{caption_body.replace('**', '')}"
        ),
        "panel_a": {
            "question": (
                "Does synchronized/refined interictal HFO burden distinguish "
                "clinical SOZ from non-SOZ contacts within subjects?"
            ),
            "score": "refined events_count per contact",
            "roc": roc_summaries,
            "auc_inset": "one dot per subject; black bar is subject-mean AUC",
        },
        "panel_b": {
            "question": (
                "Does restricting raw HFO detections to accepted group-event "
                "windows improve subject-level SOZ AUC?"
            ),
            "paired_subject_contract": paired_summaries,
            "test": "two-sided paired Wilcoxon, matching the accepted SOZ validation producer",
            "terminology": (
                "Synchronized = refined HFO count inside accepted group-event "
                "windows; identical to the current _refineGpu event-count artifact."
            ),
        },
        "source_artifacts": [
            "results/hfo_detection/<subject>/*_gpu.npz",
            "results/hfo_detection/<subject>/_refineGpu.npz",
            "results/yuquan_soz_core_channels.json",
            "results/epilepsiae_soz_core_channels.json",
            "config/subject_params.json",
        ],
        "subject_table": str(PAIR_CSV.relative_to(ROOT)),
        "outputs": {
            "png": str(png.relative_to(ROOT)),
            "pdf": str(pdf.relative_to(ROOT)),
        },
    }
    (OUT_ROOT / "supp_fig2_soz_auc_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (FIG_DIR / "README.md").write_text(
        "### supp_fig2_soz_auc_raw_vs_synchronized.png\n\n"
        f"**Supplementary Fig. 2 | {caption_title}**\n\n"
        f"{caption_body}\n\n"
        f"**关注点**：Yuquan 配对 n={yq['n_subjects_paired']}，"
        f"mean AUC {yq['raw_mean_auc']:.3f}→"
        f"{yq['synchronized_mean_auc']:.3f}，双侧 Wilcoxon "
        f"P={yq['wilcoxon_two_sided_p']:.3g}；Epilepsiae 配对 "
        f"n={epi['n_subjects_paired']}，mean AUC "
        f"{epi['raw_mean_auc']:.3f}→{epi['synchronized_mean_auc']:.3f}，"
        f"P={epi['wilcoxon_two_sided_p']:.3g}。该图支持空间定位增益，不证明传播机制。\n",
        encoding="utf-8",
    )
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
