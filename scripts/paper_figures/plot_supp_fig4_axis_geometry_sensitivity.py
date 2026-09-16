#!/usr/bin/env python3
"""Build Supplementary Figure 4 without duplicating the main 2-D field panel.

Panel A compares the TA--TB axis-angle distributions from Timing-only and
Timing + 3D direction clustering. Panel B shows their paired held-out
direction scores. Panel C compares each score against its own frozen-model
direction-shuffle null and compares the two methods directly. Panel D adds
clinical-SOZ contact compactness relative to patient-specific all-contact
implantation nulls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_figures import plot_fig2_template_axis_direction as angle_fig  # noqa: E402
from paper_figures import plot_interictal_spatial_information_gain as gain_fig  # noqa: E402
import run_soz_spatial_compactness as soz_compact_fig  # noqa: E402
from src.plot_style import COL_YQ  # noqa: E402
from src.supplementary_figure_style import (  # noqa: E402
    ANNOTATION_SIZE,
    AXIS_LABEL_SIZE,
    LEGEND_SIZE,
    PANEL_LETTER_SIZE,
    SIGNIFICANCE_SIZE,
    TICK_LABEL_SIZE,
    apply_supplementary_rcparams,
    normalize_axis_text,
)


OUT_ROOT = ROOT / "results/paper-ready-figure/supp_fig4_axis_geometry"
FIG_DIR = OUT_ROOT / "figures"
COMPACTNESS_ROOT = ROOT / "results/spatial_modulation/soz_contact_compactness"
GAIN_ROOT = ROOT / "results/interictal_propagation_masked/spatial_information_gain_all_events"
NEW_FIELD_ROOT = (
    ROOT / "results/interictal_propagation_masked/"
    "template_gradient_fields_all_events_timing_plus_space"
)
GAIN_NULL_NPZ = GAIN_ROOT / "cohort_direction_shuffle_null.npz"
COL_EPI_PURPLE = "#7A3E87"


def _angle_rows() -> list[dict]:
    rows = []
    for path in sorted((NEW_FIELD_ROOT / "per_subject").glob("*.json")):
        record = json.loads(path.read_text())
        relation = ((record.get("axis_pair") or {}).get("relation") or {})
        cosine = relation.get("cosine")
        if record.get("status") != "ok" or cosine is None:
            continue
        cosine = float(cosine)
        if not np.isfinite(cosine):
            continue
        rows.append({
            "subject_id": str(record["subject_id"]),
            "dataset": str(record["dataset"]),
            "cos_ta_tb": cosine,
            "directed_angle_deg": float(
                np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            ),
            "geometry_2d_supported": bool(
                ((record.get("direction_validity") or {}).get("pair") or {}).get(
                    "geometry_2d_supported"
                )
            ),
            "strict_stability_pass": bool(
                ((record.get("direction_validity") or {}).get("pair") or {}).get(
                    "strict_stability_pass"
                )
            ),
        })
    if len(rows) != 28:
        raise RuntimeError(f"expected 28 all-event Timing+Space axes, found {len(rows)}")
    return rows


def _compactness_payload() -> tuple[list[dict], dict]:
    summary_path = COMPACTNESS_ROOT / "cohort_summary.json"
    per_subject_dir = COMPACTNESS_ROOT / "per_subject"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing compactness summary: {summary_path}")
    records = [
        json.loads(path.read_text())
        for path in sorted(per_subject_dir.glob("*.json"))
    ]
    summary = json.loads(summary_path.read_text())
    n_primary = summary["all_contact_primary"]["overall"]["n"]
    if len(records) != summary["n_labelled_subjects"] or n_primary != 29:
        raise RuntimeError(
            "unexpected SOZ compactness cohort: "
            f"records={len(records)}, primary={n_primary}"
        )
    return records, summary


def _direction_gain_payload() -> tuple[list[dict], dict, dict]:
    subject_csv = GAIN_ROOT / "subject_spatial_information_gain.csv"
    summary_path = GAIN_ROOT / "spatial_information_gain_summary.json"
    rows = gain_fig._load_rows(subject_csv)
    payload = json.loads(summary_path.read_text())
    summary = payload["cohort_statistics"]
    if len(rows) != int(summary["n"]) or int(summary["n"]) != 26:
        raise RuntimeError(
            "unexpected spatial-information cohort: "
            f"rows={len(rows)}, summary_n={summary['n']}"
        )
    if int(summary["n_positive_gain"]) != 22:
        raise RuntimeError(
            "unexpected spatial-information direction: "
            f"n_positive={summary['n_positive_gain']}"
        )
    return rows, summary, payload


def _angle_summary(angles: np.ndarray) -> dict:
    q25, median, q75 = np.percentile(angles, [25, 50, 75])
    return {
        "n": int(angles.size),
        "mean_deg": float(np.mean(angles)),
        "sd_deg": float(np.std(angles, ddof=1)),
        "median_deg": float(median),
        "iqr_deg": [float(q25), float(q75)],
        "n_negative_cosine": int(np.sum(angles > 90.0)),
        "n_angle_ge_120": int(np.sum(angles >= 120.0)),
    }


def _draw_compact_half_rose(
    ax: plt.Axes,
    timing_rows: list[dict],
    hybrid_rows: list[dict],
) -> dict:
    timing_by_subject = {
        row["subject_id"]: float(row["directed_angle_deg"])
        for row in timing_rows
    }
    hybrid_by_subject = {
        row["subject_id"]: float(row["directed_angle_deg"])
        for row in hybrid_rows
    }
    subjects = sorted(set(timing_by_subject) & set(hybrid_by_subject))
    if len(subjects) != 28:
        raise RuntimeError(
            f"expected 28 paired axis-angle subjects, found {len(subjects)}"
        )
    timing = np.asarray([timing_by_subject[subject] for subject in subjects])
    hybrid = np.asarray([hybrid_by_subject[subject] for subject in subjects])
    edges_deg = np.linspace(0.0, 180.0, angle_fig.N_BINS + 1)
    edges = np.deg2rad(edges_deg)
    centers = edges[:-1] + np.diff(edges) / 2
    bin_width = np.diff(edges)
    timing_counts = np.histogram(timing, bins=edges_deg)[0]
    hybrid_counts = np.histogram(hybrid, bins=edges_deg)[0]
    timing_prop = timing_counts / timing.size
    hybrid_prop = hybrid_counts / hybrid.size
    all_proportions = [timing_prop, hybrid_prop]
    bin_counts = {
        "timing_only": timing_counts.astype(int).tolist(),
        "timing_plus_3d": hybrid_counts.astype(int).tolist(),
    }
    radial_max = max(
        0.40,
        np.ceil(np.max(all_proportions) / 0.05) * 0.05 * 1.20,
    )
    timing_rgb = np.asarray(to_rgba(gain_fig.TEMPORAL_COLOR)[:3])
    hybrid_rgb = np.asarray(to_rgba(gain_fig.HYBRID_COLOR)[:3])
    overlap_color = tuple(0.5 * (timing_rgb + hybrid_rgb))
    full_width = bin_width * 0.88
    for center, width, timing_height, hybrid_height in zip(
        centers, full_width, timing_prop, hybrid_prop
    ):
        overlap = float(min(timing_height, hybrid_height))
        if overlap > 0:
            ax.bar(
                center, overlap, width=width,
                facecolor=to_rgba(overlap_color, 0.48),
                edgecolor="none", zorder=2,
            )
        if timing_height > overlap:
            ax.bar(
                center, timing_height - overlap, width=width, bottom=overlap,
                facecolor=to_rgba(gain_fig.TEMPORAL_COLOR, 0.30),
                edgecolor="none", zorder=2,
            )
        if hybrid_height > overlap:
            ax.bar(
                center, hybrid_height - overlap, width=width, bottom=overlap,
                facecolor=to_rgba(gain_fig.HYBRID_COLOR, 0.30),
                edgecolor="none", zorder=2,
            )
        ax.bar(
            center, timing_height, width=width,
            facecolor="none", edgecolor=gain_fig.TEMPORAL_COLOR,
            linewidth=0.85, zorder=3,
        )
        ax.bar(
            center, hybrid_height, width=width,
            facecolor="none", edgecolor=gain_fig.HYBRID_COLOR,
            linewidth=0.85, zorder=3,
        )
    for boundary in (0.0, np.pi):
        ax.plot(
            [boundary, boundary],
            [0, radial_max],
            color=angle_fig.BOUNDARY_COLOR,
            lw=0.9,
            zorder=5,
        )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_ylim(0, radial_max)
    ax.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]))
    ax.set_xticklabels(
        ["0° same", "", "", "90°", "", "", "180° opposite"],
        fontsize=TICK_LABEL_SIZE,
    )
    radial_ticks = [value for value in (0.10, 0.20, 0.30) if value < radial_max]
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([])
    for value in radial_ticks:
        ax.text(
            np.deg2rad(20),
            value,
            f"{int(100 * value)}%",
            ha="left",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            color="0.35",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.3},
            zorder=6,
        )
    ax.grid(color=angle_fig.GRID_COLOR, lw=0.5)
    ax.spines["polar"].set_color("0.55")
    ax.spines["polar"].set_linewidth(0.6)
    for index, (values, color) in enumerate((
        (timing, gain_fig.TEMPORAL_COLOR),
        (hybrid, gain_fig.HYBRID_COLOR),
    )):
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        mean_radius = radial_max * (0.93 + 0.055 * index)
        ax.scatter(
            [np.deg2rad(mean)],
            [mean_radius],
            marker="D",
            s=20,
            facecolor=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=7,
            clip_on=False,
        )
        arc = np.linspace(
            np.deg2rad(max(0.0, mean - sd)),
            np.deg2rad(min(180.0, mean + sd)),
            100,
        )
        ax.plot(
            arc,
            np.full_like(arc, radial_max * (0.93 + 0.055 * index)),
            color=color,
            lw=2.4,
            solid_capstyle="butt",
            zorder=6,
        )
    ax.legend(
        handles=[
            Line2D(
                [0], [0], color=gain_fig.TEMPORAL_COLOR, lw=4.0,
                label="Timing-only",
            ),
            Line2D(
                [0], [0], color=gain_fig.HYBRID_COLOR, lw=4.0,
                label="Timing + 3D direction",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(1.02, 1.05),
        ncol=1,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=1.7,
        handletextpad=0.35,
        borderaxespad=0,
    )
    return {
        "n": int(len(subjects)),
        "bin_edges_deg": edges_deg.tolist(),
        "bin_counts": bin_counts,
        "timing_only": _angle_summary(timing),
        "timing_plus_3d_direction": _angle_summary(hybrid),
        "paired_median_angle_increase_deg": float(np.median(hybrid - timing)),
        "n_angle_increased": int(np.sum(hybrid > timing)),
        "paired_wilcoxon_greater_p": float(
            wilcoxon(hybrid, timing, alternative="greater").pvalue
        ),
    }


def _score_null_payload(rows: list[dict]) -> dict:
    if not GAIN_NULL_NPZ.exists():
        raise FileNotFoundError(f"missing direction-shuffle null: {GAIN_NULL_NPZ}")
    null = np.load(GAIN_NULL_NPZ)
    subject_ids = [str(value) for value in null["subject_ids"]]
    by_subject = {str(row["subject_id"]): row for row in rows}
    if set(subject_ids) != set(by_subject):
        raise RuntimeError("direction-score rows and null subject IDs do not match")
    timing = np.asarray([
        float(by_subject[subject]["timing_only_score"]) for subject in subject_ids
    ])
    hybrid = np.asarray([
        float(by_subject[subject]["timing_plus_space_score"]) for subject in subject_ids
    ])
    timing_null = np.median(
        np.asarray(null["patient_null_timing_only_score"], float), axis=1
    )
    hybrid_null = np.median(
        np.asarray(null["patient_null_timing_plus_space_score"], float), axis=1
    )

    def summarize(values: np.ndarray) -> dict:
        q25, q75 = np.percentile(values, [25, 75])
        return {
            "median": float(np.median(values)),
            "iqr": [float(q25), float(q75)],
        }

    return {
        "subject_ids": subject_ids,
        "timing_only": timing,
        "timing_only_null": timing_null,
        "timing_plus_3d_direction": hybrid,
        "timing_plus_3d_direction_null": hybrid_null,
        "summary": {
            "n": int(len(subject_ids)),
            "timing_only": summarize(timing),
            "timing_only_null": summarize(timing_null),
            "timing_plus_3d_direction": summarize(hybrid),
            "timing_plus_3d_direction_null": summarize(hybrid_null),
            "timing_only_vs_null_p": float(
                wilcoxon(timing, timing_null, alternative="greater").pvalue
            ),
            "timing_plus_3d_direction_vs_null_p": float(
                wilcoxon(hybrid, hybrid_null, alternative="greater").pvalue
            ),
            "timing_plus_3d_direction_vs_timing_only_p": float(
                wilcoxon(hybrid, timing, alternative="greater").pvalue
            ),
        },
    }


def _draw_score_null_comparison(ax: plt.Axes, payload: dict) -> dict:
    timing = payload["timing_only"]
    timing_null = payload["timing_only_null"]
    hybrid = payload["timing_plus_3d_direction"]
    hybrid_null = payload["timing_plus_3d_direction_null"]
    positions = [0.0, 0.42, 1.12, 1.54]
    groups = (
        (timing_null, positions[0], gain_fig.TEMPORAL_COLOR, False),
        (timing, positions[1], gain_fig.TEMPORAL_COLOR, True),
        (hybrid_null, positions[2], gain_fig.HYBRID_COLOR, False),
        (hybrid, positions[3], gain_fig.HYBRID_COLOR, True),
    )
    rng = np.random.default_rng(44)
    for values, xpos, color, observed in groups:
        violin = ax.violinplot(
            [values],
            positions=[xpos],
            widths=0.30,
            showextrema=False,
            showmedians=False,
        )["bodies"][0]
        violin.set_facecolor(color if observed else "white")
        violin.set_edgecolor(color)
        violin.set_linewidth(0.9)
        violin.set_alpha(0.25 if observed else 0.85)
        jitter = rng.uniform(-0.035, 0.035, len(values))
        ax.scatter(
            xpos + jitter,
            values,
            s=12,
            facecolor=color if observed else "white",
            edgecolor=color,
            linewidth=0.55,
            alpha=0.85,
            zorder=4,
        )
        q25, q75 = np.percentile(values, [25, 75])
        median = float(np.median(values))
        ax.vlines(xpos, q25, q75, color="black", lw=0.9, zorder=5)
        ax.hlines(median, xpos - 0.10, xpos + 0.10, color="black", lw=1.6, zorder=5)
    for left, right, values_left, values_right in (
        (positions[0], positions[1], timing_null, timing),
        (positions[2], positions[3], hybrid_null, hybrid),
    ):
        for index in range(len(values_left)):
            ax.plot(
                [left, right],
                [values_left[index], values_right[index]],
                color="0.78",
                lw=0.45,
                alpha=0.55,
                zorder=1,
            )

    summary = payload["summary"]
    comparisons = (
        (
            positions[0],
            positions[1],
            0.78,
            summary["timing_only_vs_null_p"],
        ),
        (
            positions[2],
            positions[3],
            0.78,
            summary["timing_plus_3d_direction_vs_null_p"],
        ),
        (
            positions[1],
            positions[3],
            0.92,
            summary["timing_plus_3d_direction_vs_timing_only_p"],
        ),
    )
    for left, right, top, p_value in comparisons:
        ax.plot(
            [left, left, right, right],
            [top - 0.025, top, top, top - 0.025],
            color="black",
            lw=0.8,
            clip_on=False,
        )
        ax.text(
            0.5 * (left + right),
            top + 0.012,
            gain_fig._p_stars(float(p_value)),
            ha="center",
            va="bottom",
            fontsize=SIGNIFICANCE_SIZE,
            fontweight="bold",
        )
    ax.axhline(0.0, color="0.65", lw=0.65, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(-0.30, 1.84)
    ax.set_ylim(-0.05, 1.02)
    ax.set_xticks(positions, ["Null", "Data", "Null", "Data"])
    ax.set_ylabel("Held-out direction score", fontsize=AXIS_LABEL_SIZE, labelpad=3)
    ax.text(
        np.mean(positions[:2]), -0.18, "Timing-only",
        transform=ax.get_xaxis_transform(), ha="center", va="top",
        color=gain_fig.TEMPORAL_COLOR, fontsize=TICK_LABEL_SIZE,
    )
    ax.text(
        np.mean(positions[2:]), -0.18, "Timing + 3D direction",
        transform=ax.get_xaxis_transform(), ha="center", va="top",
        color=gain_fig.HYBRID_COLOR, fontsize=TICK_LABEL_SIZE,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92", lw=0.6)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    return summary


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    timing_angle_rows = angle_fig._load_rows()
    hybrid_angle_rows = _angle_rows()
    gain_rows, gain_summary, gain_payload = _direction_gain_payload()
    score_null_payload = _score_null_payload(gain_rows)
    compactness_records, compactness_summary = _compactness_payload()
    apply_supplementary_rcparams()
    fig = plt.figure(figsize=(7.45, 5.20), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.00, 1.03],
        height_ratios=[1.00, 1.00],
        wspace=0.42,
        hspace=0.50,
        left=0.090,
        right=0.985,
        bottom=0.105,
        top=0.945,
    )
    ax_rose = fig.add_subplot(grid[0, 0], projection="polar")
    ax_gain = fig.add_subplot(grid[0, 1])
    ax_null = fig.add_subplot(grid[1, 0])
    ax_soz = fig.add_subplot(grid[1, 1])
    rose_summary = _draw_compact_half_rose(
        ax_rose, timing_angle_rows, hybrid_angle_rows
    )
    gain_fig.draw_paired_scores(
        ax_gain,
        gain_rows,
        gain_summary,
        title=None,
        ylabel="Held-out direction score",
    )
    ax_gain.set_xticklabels(["Timing-only", "Timing + 3D direction"])
    for tick, color in zip(
        ax_gain.get_xticklabels(),
        (gain_fig.TEMPORAL_COLOR, gain_fig.HYBRID_COLOR),
    ):
        tick.set_color(color)
        tick.set_fontweight("normal")
    null_summary = _draw_score_null_comparison(ax_null, score_null_payload)
    soz_compact_fig.draw_compactness_panel(
        ax_soz,
        compactness_records,
        "all_contact_null",
        compactness_summary["all_contact_primary"],
        dataset_colors={"yuquan": COL_YQ, "epilepsiae": COL_EPI_PURPLE},
        point_size=18.0,
        annotation_fontsize=ANNOTATION_SIZE,
        tick_fontsize=TICK_LABEL_SIZE,
        line_scale=0.72,
    )
    ax_soz.set_ylabel("SOZ radius / null median", fontsize=AXIS_LABEL_SIZE, labelpad=2)
    ax_soz.set_xticklabels(["Yuquan", "Epilepsiae"])
    for tick, color in zip(ax_soz.get_xticklabels(), (COL_YQ, COL_EPI_PURPLE)):
        tick.set_color(color)
        tick.set_fontweight("normal")
    for annotation in ax_soz.texts:
        if "P=" in annotation.get_text():
            annotation.set_text("***")
    fig.canvas.draw()
    for label, x, y in (
        ("A", 0.025, 0.975),
        ("B", 0.545, 0.975),
        ("C", 0.025, 0.505),
        ("D", 0.545, 0.505),
    ):
        fig.text(
            x,
            y,
            label,
            ha="left",
            va="top",
            fontsize=PANEL_LETTER_SIZE,
            fontweight="bold",
        )
    for axis in (ax_rose, ax_gain, ax_null, ax_soz):
        normalize_axis_text(axis)
    for tick, color in zip(
        ax_gain.get_xticklabels(),
        (gain_fig.TEMPORAL_COLOR, gain_fig.HYBRID_COLOR),
    ):
        tick.set_color(color)
    for tick, color in zip(ax_soz.get_xticklabels(), (COL_YQ, COL_EPI_PURPLE)):
        tick.set_color(color)
    stem = FIG_DIR / "supp_fig4_axis_direction_and_heldout_readback"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    compact_all = compactness_summary["all_contact_primary"]
    compact_overall = compact_all["overall"]
    compact_ci = compact_overall["median_ratio_bootstrap_95ci"]
    compact_p = compact_overall["wilcoxon_log2_ratio_less_than_0"]["p_value"]
    compact_yq_p = compact_all["yuquan"]["wilcoxon_log2_ratio_less_than_0"]["p_value"]
    compact_epi_p = compact_all["epilepsiae"]["wilcoxon_log2_ratio_less_than_0"]["p_value"]
    gain_ci = gain_summary["median_gain_bootstrap_ci95"]
    timing_angle = rose_summary["timing_only"]
    hybrid_angle = rose_summary["timing_plus_3d_direction"]
    caption_title = (
        "Three-dimensional direction shifts paired template axes and improves "
        "held-out directional organization."
    )
    caption_body = (
        f"**A,** Paired half-rose distributions of the directed angular "
        f"separation between the full-fit TA and TB axes obtained using "
        f"Timing-only (blue) and Timing + 3D direction (orange) clustering in "
        f"{rose_summary['n']} patients. "
        f"0\u00b0 denotes the same direction and 180\u00b0 the opposite direction; "
        f"overlapping bars show the fraction of patients in 30\u00b0 bins, with "
        f"the blended region denoting the shared fraction. Coloured diamonds "
        f"mark means and outer arcs show mean \u00b1 1 s.d. The mean separation "
        f"was {timing_angle['mean_deg']:.1f}\u00b0 \u00b1 "
        f"{timing_angle['sd_deg']:.1f}\u00b0 for Timing-only and "
        f"{hybrid_angle['mean_deg']:.1f}\u00b0 \u00b1 "
        f"{hybrid_angle['sd_deg']:.1f}\u00b0 for Timing + 3D direction. "
        f"**B,** Paired patient-level direction scores for Timing-only and "
        f"Timing + 3D direction clustering, evaluated by alternating-recording-block "
        f"cross-validation. All interictal events contributed timing features, "
        f"whereas events with finite direction estimates additionally "
        f"contributed the masked spatial view. The score is the equal-fold, "
        f"equal-template mean signed cosine "
        f"between each held-out event direction and its assigned training-template "
        f"axis. Points denote patients, lines preserve patient pairing, violins "
        f"show the distributions, boxes show IQRs and black horizontal lines mark "
        f"medians. The median score increased from "
        f"{gain_summary['timing_only_median']:.3f} to "
        f"{gain_summary['timing_plus_space_median']:.3f}; "
        f"{gain_summary['n_positive_gain']} of {gain_summary['n']} patients "
        f"improved, with a median paired gain of {gain_summary['median_gain']:.3f} "
        f"(95% bootstrap CI, {gain_ci[0]:.3f}–{gain_ci[1]:.3f}; one-sided "
        f"paired Wilcoxon P = {gain_summary['paired_wilcoxon_greater_p']:.3g}). "
        f"**C,** The same held-out direction scores compared with the "
        f"patient-specific median of each method's direction-shuffle null. "
        f"Open and filled distributions denote null and observed scores, "
        f"respectively; grey lines preserve within-patient pairing. Timing-only "
        f"and Timing + 3D direction scores both exceeded their respective nulls "
        f"({null_summary['n']} of {null_summary['n']} patients for each method; "
        f"one-sided paired Wilcoxon P = "
        f"{null_summary['timing_only_vs_null_p']:.3g} and "
        f"{null_summary['timing_plus_3d_direction_vs_null_p']:.3g}, respectively). "
        f"The upper bracket compares the two observed methods (one-sided paired "
        f"Wilcoxon P = "
        f"{null_summary['timing_plus_3d_direction_vs_timing_only_p']:.3g}). "
        f"**D,** Clinical SOZ-contact compactness relative to each patient's "
        f"mapped invasive-contact geometry. The SOZ RMS radius was divided by "
        f"the median radius of {compactness_summary['n_null_per_subject']:,} "
        f"equally sized random contact subsets from the same patient; the "
        f"dashed line marks the null ratio of 1. Points denote patients "
        f"(blue, Yuquan, n = {compact_all['yuquan']['n']}; purple, Epilepsiae, "
        f"n = {compact_all['epilepsiae']['n']}), filled points indicate a "
        f"patient-level empirical P < 0.05, and horizontal and vertical black "
        f"lines mark medians and IQRs. Brackets show one-sided within-dataset "
        f"Wilcoxon signed-rank tests of log2 ratios against 0 (Yuquan, "
        f"P = {compact_yq_p:.2g}; Epilepsiae, P = {compact_epi_p:.2g}; "
        f"*** denotes P < 0.001). Across both cohorts, the median ratio was "
        f"{compact_overall['median_observed_to_null_ratio']:.3f} "
        f"(95% bootstrap CI, {compact_ci[0]:.3f}–{compact_ci[1]:.3f}; "
        f"n = {compact_overall['n']} patients; one-sided Wilcoxon "
        f"P = {compact_p:.2g})."
    )
    metadata = {
        "figure": "Supplementary Figure 4",
        "caption": (
            f"Supplementary Fig. 4 | {caption_title} "
            f"{caption_body.replace('**A,**', 'A,').replace('**B,**', 'B,').replace('**C,**', 'C,').replace('**D,**', 'D,')}"
        ),
        "main_figure_field_panel_repeated": False,
        "panel_a": {
            "timing_only_source": (
                "results/interictal_propagation_masked/template_gradient_fields/"
                "axis_cohort.csv"
            ),
            "timing_plus_3d_direction_source": (
                "results/interictal_propagation_masked/"
                "template_gradient_fields_all_events_timing_plus_space/"
                "axis_cohort.csv"
            ),
            "definition": (
                "Paired directed TA-TB axis-angle distributions from Timing-only "
                "and Timing + 3D direction full fits"
            ),
            **rose_summary,
        },
        "panel_b": {
            "source_table": str(
                (GAIN_ROOT / "subject_spatial_information_gain.csv").relative_to(ROOT)
            ),
            "source_summary": str(
                (GAIN_ROOT / "spatial_information_gain_summary.json").relative_to(ROOT)
            ),
            "definition": (
                "Two-way alternating-recording-block cross-fit comparison of "
                "Timing-only and Timing + 3D direction clustering; held-out score "
                "is the equal-fold, equal-template mean signed cosine between "
                "each event direction and its assigned training-template axis"
            ),
            "cohort_flow": gain_payload["cohort_flow"],
            "summary": gain_summary,
        },
        "panel_c": {
            "source": str(GAIN_NULL_NPZ.relative_to(ROOT)),
            "definition": (
                "Patient-level observed held-out direction scores versus each "
                "method's own within-recording-block direction-shuffle null "
                "median, plus the paired comparison between observed methods"
            ),
            "summary": null_summary,
        },
        "panel_d": {
            "source_summary": str(
                (COMPACTNESS_ROOT / "cohort_summary.json").relative_to(ROOT)
            ),
            "source_table": str(
                (COMPACTNESS_ROOT / "subject_compactness.csv").relative_to(ROOT)
            ),
            "definition": (
                "Clinical SOZ-contact RMS radius divided by the median radius "
                "of equal-size subsets drawn from all mapped invasive contacts "
                "within subject"
            ),
            "n_null_per_subject": compactness_summary["n_null_per_subject"],
            "summary": compact_all,
            "claim_boundary": compactness_summary["claim_boundary"],
        },
        "outputs": {
            "png": str(stem.with_suffix(".png").relative_to(ROOT)),
            "pdf": str(stem.with_suffix(".pdf").relative_to(ROOT)),
        },
    }
    (OUT_ROOT / "supp_fig4_axis_geometry_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (FIG_DIR / "README.md").write_text(
        "### supp_fig4_axis_direction_and_heldout_readback.png\n\n"
        f"**Supplementary Fig. 4 | {caption_title}**\n\n"
        f"{caption_body}\n\n"
        f"**关注点**：A 直接比较两种聚类的 TA/TB 轴夹角分布，均值 \u00b1 s.d. "
        f"由 {timing_angle['mean_deg']:.1f}\u00b0 \u00b1 "
        f"{timing_angle['sd_deg']:.1f}\u00b0 移至 "
        f"{hybrid_angle['mean_deg']:.1f}\u00b0 \u00b1 "
        f"{hybrid_angle['sd_deg']:.1f}\u00b0。B 显示真实三维方向信息带来的"
        f"留出方向增益："
        f"{gain_summary['n_positive_gain']}/{gain_summary['n']} 名患者提高，"
        f"中位增益 {gain_summary['median_gain']:.3f}（95% CI "
        f"{gain_ci[0]:.3f}–{gain_ci[1]:.3f}）。C 显示两种方法均在 "
        f"{null_summary['n']}/{null_summary['n']} 名患者中超过各自的方向置换 "
        f"null，并单独给出两种观测方法间的配对比较。A\u2013C 沿用主图的 "
        f"Timing-only / Timing + 3D direction 名称及蓝/橙语义色。D 共 n="
        f"{compact_overall['n']}，SOZ/null RMS 半径比中位数="
        f"{compact_overall['median_observed_to_null_ratio']:.3f}，95% CI "
        f"{compact_ci[0]:.3f}–{compact_ci[1]:.3f}。D 只支持 clinical SOZ "
        "触点相对患者自身植入几何更紧凑，不证明群体事件局限于 SOZ。"
        "已进入主图的 2D field panel 不在补图重复；完整患者数值进入补充表。\n",
        encoding="utf-8",
    )
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
