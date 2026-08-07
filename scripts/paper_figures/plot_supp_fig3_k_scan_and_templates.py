#!/usr/bin/env python3
"""Build Supplementary Figure 3: K=2--10 sensitivity and patient examples.

The K scan extends the accepted masked-feature K=2--8 analysis to K=9 and K=10
without overwriting canonical per-subject JSON files.  Selection maximizes
silhouette among solutions passing both the AMI stability and minimum-cluster
fraction gates.  The second panel set keeps Figure-1E's heatmap/rank-profile
grammar while showing two K=4 solutions and one K=6 solution.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.interictal_propagation import (  # noqa: E402
    _kmeans_stability_for_k,
    _valid_event_indices,
    load_subject_propagation_events,
)
from src.lagpat_rank_audit import build_masked_kmeans_features  # noqa: E402
from src.plot_style import COL_EPI, COL_YQ  # noqa: E402
from paper_figures import plot_fig1_interictal_hfo_temporal_scaffold as fig1  # noqa: E402
from paper_figures.patient_public_labels import (  # noqa: E402
    artifact_subject_from_public,
    public_patient_label,
)
import plot_interictal_propagation as propagation_plot  # noqa: E402


RESULT_ROOT = ROOT / "results/interictal_propagation_masked"
OUT_ROOT = ROOT / "results/paper-ready-figure/supp_fig3_k_scan_templates"
FIG_DIR = OUT_ROOT / "figures"
YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
AMI_THRESHOLD = 0.70
MIN_CLUSTER_FRACTION = 0.10
K_VALUES = tuple(range(2, 11))
TA_COLOR = "#D62728"
TB_COLOR = "#2878B5"
EXTRA_TEMPLATE_COLORS = ("#2A9D8F", "#E9A23B", "#7A5195", "#8C564B")


def _subject_dir(dataset: str, subject: str) -> Path:
    if dataset == "yuquan":
        return YUQUAN_ROOT / subject
    legacy = EPILEPSIAE_ROOT / subject / "all_recs"
    return legacy if legacy.exists() else EPILEPSIAE_ROOT / subject


def _load_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((RESULT_ROOT / "per_subject").glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        dataset, subject = path.stem.split("_", 1)
        record["dataset"] = dataset
        record["subject"] = subject
        records.append(record)
    if len(records) != 40:
        raise RuntimeError(f"expected the accepted 40-subject cohort, found {len(records)}")
    return records


def _extend_scan(record: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    adaptive = record["adaptive_cluster"]
    existing = {int(row["k"]): dict(row) for row in adaptive["scan"]}
    missing = [k for k in K_VALUES if k not in existing]
    if missing:
        loaded = load_subject_propagation_events(
            _subject_dir(record["dataset"], record["subject"])
        )
        valid_events = _valid_event_indices(loaded["bools"], min_participating=3)
        features = build_masked_kmeans_features(
            loaded["ranks"][:, valid_events],
            loaded["bools"][:, valid_events],
            impute="event_median",
        )
        for k in missing:
            result = _kmeans_stability_for_k(
                features,
                k,
                n_seeds=10,
                min_cluster_fraction=MIN_CLUSTER_FRACTION,
                stability_threshold=AMI_THRESHOLD,
            )
            existing[k] = {
                key: value for key, value in result.items() if key != "best_labels"
            }

    scan = [existing[k] for k in K_VALUES]
    passing = [row for row in scan if bool(row["passes_both"])]
    selected = (
        int(max(passing, key=lambda row: float(row["median_silhouette"]))["k"])
        if passing
        else 2
    )
    return scan, selected


def _scan_table(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    selected: Counter[int] = Counter()
    for record in records:
        scan, chosen_k = _extend_scan(record)
        selected[chosen_k] += 1
        for entry in scan:
            rows.append(
                {
                    "dataset": record["dataset"],
                    "subject": record["subject"],
                    "k": int(entry["k"]),
                    "median_silhouette": float(entry["median_silhouette"]),
                    "median_ami": float(entry["median_ami"]),
                    "worst_min_cluster_fraction": float(
                        entry["worst_min_cluster_fraction"]
                    ),
                    "passes_stability": bool(entry["passes_stability"]),
                    "passes_fraction": bool(entry["passes_fraction"]),
                    "passes_both": bool(entry["passes_both"]),
                    "selected_k_2_to_10": int(chosen_k),
                    "canonical_selected_k_2_to_8": int(adaptive_k(record)),
                }
            )
    return rows, {str(k): int(selected.get(k, 0)) for k in K_VALUES}


def adaptive_k(record: dict[str, Any]) -> int:
    return int(record["adaptive_cluster"]["chosen_k"])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    public_rows = []
    for row in rows:
        public_row = dict(row)
        subject = public_row.pop("subject")
        public_row["patient_id"] = public_patient_label(
            public_row["dataset"], subject
        )
        public_rows.append(public_row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)


def _read_completed_scan(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if len(raw) != 40 * len(K_VALUES):
        return None
    rows: list[dict[str, Any]] = []
    for row in raw:
        subject = (
            artifact_subject_from_public(row["dataset"], row["patient_id"])
            if "patient_id" in row
            else row["subject"]
        )
        rows.append(
            {
                "dataset": row["dataset"],
                "subject": subject,
                "k": int(row["k"]),
                "median_silhouette": float(row["median_silhouette"]),
                "median_ami": float(row["median_ami"]),
                "worst_min_cluster_fraction": float(row["worst_min_cluster_fraction"]),
                "passes_stability": row["passes_stability"] == "True",
                "passes_fraction": row["passes_fraction"] == "True",
                "passes_both": row["passes_both"] == "True",
                "selected_k_2_to_10": int(row["selected_k_2_to_10"]),
                "canonical_selected_k_2_to_8": int(row["canonical_selected_k_2_to_8"]),
            }
        )
    subject_selected = {
        (row["dataset"], row["subject"]): int(row["selected_k_2_to_10"])
        for row in rows
    }
    counts = Counter(subject_selected.values())
    distribution = {str(k): int(counts.get(k, 0)) for k in K_VALUES}
    return rows, distribution


def _scan_payload(
    rows: list[dict[str, Any]], distribution: dict[str, int]
) -> dict[str, np.ndarray]:
    by_subject: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_subject.setdefault((row["dataset"], row["subject"]), []).append(row)

    sil = np.asarray(
        [
            [next(r for r in scan if int(r["k"]) == k)["median_silhouette"] for k in K_VALUES]
            for scan in by_subject.values()
        ],
        dtype=float,
    )
    delta_sil = sil - sil[:, [0]]
    median = np.nanmedian(delta_sil, axis=0)
    q1, q3 = np.nanpercentile(delta_sil, [25, 75], axis=0)
    ami_pass = np.asarray(
        [
            np.mean([bool(r["passes_stability"]) for r in rows if int(r["k"]) == k])
            for k in K_VALUES
        ]
    )
    frac_pass = np.asarray(
        [
            np.mean([bool(r["passes_fraction"]) for r in rows if int(r["k"]) == k])
            for k in K_VALUES
        ]
    )
    both_pass = np.asarray(
        [
            np.mean([bool(r["passes_both"]) for r in rows if int(r["k"]) == k])
            for k in K_VALUES
        ]
    )
    counts = np.asarray([distribution[str(k)] for k in K_VALUES], dtype=int)
    return {
        "sil": sil,
        "delta_sil": delta_sil,
        "median": median,
        "q1": q1,
        "q3": q3,
        "ami_pass": ami_pass,
        "frac_pass": frac_pass,
        "both_pass": both_pass,
        "counts": counts,
    }


def _draw_scan_panels(
    axes: list[plt.Axes] | np.ndarray,
    rows: list[dict[str, Any]],
    distribution: dict[str, int],
) -> None:
    payload = _scan_payload(rows, distribution)
    delta_sil = payload["delta_sil"]
    median = payload["median"]
    q1 = payload["q1"]
    q3 = payload["q3"]
    both_pass = payload["both_pass"]
    counts = payload["counts"]

    ax = axes[0]
    colors = [COL_YQ if k == 2 else "0.66" for k in K_VALUES]
    bars = ax.bar(K_VALUES, counts, color=colors, width=0.72)
    for bar, value in zip(bars, counts):
        if value:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.55,
                str(int(value)),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_ylim(0, max(counts) * 1.15)
    ax.set_xticks(K_VALUES)
    ax.set_xlabel("Selected K")
    ax.set_ylabel("Subjects")

    ax = axes[1]
    for curve in delta_sil:
        ax.plot(K_VALUES, curve, color="0.83", lw=0.50, alpha=0.46, zorder=1)
    ax.fill_between(K_VALUES, q1, q3, color=COL_YQ, alpha=0.18, lw=0, zorder=2)
    ax.plot(
        K_VALUES,
        median,
        color=COL_YQ,
        lw=2.0,
        marker="o",
        ms=3.6,
        zorder=3,
    )
    ax.axhline(0.0, color="0.35", lw=0.75, zorder=0)
    ax.set_xlabel("Number of clusters, K")
    ax.set_ylabel("Δ silhouette\n(relative to K=2)")
    ax.set_xticks(K_VALUES)

    ax = axes[2]
    ax.plot(
        K_VALUES,
        100.0 * both_pass,
        marker="o",
        ms=3.8,
        lw=1.8,
        color=COL_YQ,
    )
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 50, 100])
    ax.set_xticks(K_VALUES)
    ax.set_xlabel("Number of clusters, K")
    ax.set_ylabel("Passing both gates (%)")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7.4, length=2.5)
        axis.xaxis.label.set_size(8.0)
        axis.yaxis.label.set_size(8.0)
        axis.grid(axis="y", color="0.92", lw=0.6, zorder=0)


def _plot_scan(rows: list[dict[str, Any]], distribution: dict[str, int]) -> Path:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.8, 3.10),
        gridspec_kw={"width_ratios": [0.92, 1.25, 1.05]},
        facecolor="white",
    )
    _draw_scan_panels(axes, rows, distribution)

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.19, top=0.91, wspace=0.42)
    png = FIG_DIR / "supp_fig3a_k2_to_k10_scan.png"
    pdf = FIG_DIR / "supp_fig3a_k2_to_k10_scan.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def _pair_corr(record: dict[str, Any]) -> float:
    matrix = np.asarray(record["adaptive_cluster"]["inter_cluster_corr_matrix"], dtype=float)
    return float(matrix[0, 1]) if matrix.shape == (2, 2) else float("nan")


def _select_examples(
    records: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return fixed public examples spanning two K=4 and one K=6 solutions."""
    targets = (
        ("epilepsiae", "916", 4),
        ("epilepsiae", "818", 4),
        ("yuquan", "zhaojinrui", 6),
    )
    by_key = {(row["dataset"], row["subject"]): row for row in records}
    extended = {
        (row["dataset"], row["subject"]): int(row["selected_k_2_to_10"])
        for row in rows
    }
    selected: list[dict[str, Any]] = []
    for dataset, subject, expected_k in targets:
        record = by_key[(dataset, subject)]
        observed_k = extended[(dataset, subject)]
        if observed_k != expected_k or adaptive_k(record) != expected_k:
            raise RuntimeError(
                f"{dataset}:{subject} no longer has the locked K={expected_k} "
                f"solution (extended={observed_k}, canonical={adaptive_k(record)})"
            )
        selected.append(record)
    return selected


def _template_colors(n_clusters: int) -> list[str]:
    palette = [TA_COLOR, TB_COLOR, *EXTRA_TEMPLATE_COLORS]
    return [palette[idx % len(palette)] for idx in range(n_clusters)]


def _draw_multi_k_row(
    fig: plt.Figure,
    row_spec,
    arr: dict[str, Any],
    *,
    display_label: str,
    show_xlabel: bool,
) -> dict[str, Any]:
    """Draw a Figure-1E-compatible row without forcing non-K=2 cases into TA/TB."""
    row_grid = row_spec.subgridspec(
        1, 2, width_ratios=[8.7, 1.35], wspace=0.095
    )
    ax_heat = fig.add_subplot(row_grid[0, 0])
    ax_mean = fig.add_subplot(row_grid[0, 1])
    ranks = arr["ranks"]
    bools = arr["bools"]
    channel_order = arr["channel_order"]
    events = arr["clustered_events"]
    labels = arr["clustered_labels"]
    unique_labels = np.unique(labels)
    colors = _template_colors(len(unique_labels))
    image = propagation_plot._plot_rank_heatmap(
        ax_heat,
        ranks[channel_order][:, events],
        arr["ordered_names"],
        title="",
        display_bools=bools[channel_order][:, events],
        ytick_fontsize=5.6,
        xtick_fontsize=6.2,
    )
    gap_half_width = max(2, int(round(0.006 * len(events))))
    cursor = 0
    cluster_counts: dict[str, int] = {}
    for display_idx, (cluster_id, color) in enumerate(zip(unique_labels, colors)):
        count = int(np.sum(labels == cluster_id))
        full_count = int(np.sum(arr["labels"] == cluster_id))
        center = cursor + count / 2
        if cursor > 0:
            ax_heat.axvspan(
                cursor - gap_half_width,
                cursor + gap_half_width,
                facecolor="white",
                edgecolor="0.66",
                hatch="////",
                linewidth=0.0,
                zorder=12,
            )
        if arr["chosen_k"] == 2:
            cluster_name = "TA" if display_idx == 0 else "TB"
        else:
            cluster_name = f"T{display_idx + 1}"
        ax_heat.text(
            center,
            1.015,
            cluster_name,
            transform=ax_heat.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.7,
            fontweight="bold",
            color=color,
            clip_on=False,
        )
        cluster_counts[cluster_name] = full_count
        cursor += count

    ax_heat.text(
        0.0,
        1.13,
        f"{display_label}   K={arr['chosen_k']}",
        transform=ax_heat.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color="black",
        clip_on=False,
    )
    if show_xlabel:
        ax_heat.set_xlabel("Population events (clustered)", fontsize=8.0)
    else:
        ax_heat.set_xlabel("")
        ax_heat.tick_params(axis="x", bottom=False, labelbottom=False)

    y_pos = np.arange(len(channel_order), dtype=float)
    for display_idx, (cluster_id, color) in enumerate(zip(unique_labels, colors)):
        event_idx = arr["valid_events"][arr["labels"] == cluster_id]
        means = np.full(len(channel_order), np.nan)
        stds = np.full(len(channel_order), np.nan)
        for plot_idx, raw_idx in enumerate(channel_order):
            values = np.asarray(ranks[raw_idx, event_idx], dtype=float)
            valid = np.asarray(bools[raw_idx, event_idx], dtype=bool) & np.isfinite(
                values
            )
            if np.any(valid):
                means[plot_idx] = float(np.mean(values[valid]))
                stds[plot_idx] = float(np.std(values[valid]))
        finite = np.isfinite(means)
        ax_mean.fill_betweenx(
            y_pos[finite],
            (means - stds)[finite],
            (means + stds)[finite],
            color=color,
            alpha=0.11,
            linewidth=0,
        )
        ax_mean.plot(
            means[finite],
            y_pos[finite],
            "-o",
            color=color,
            lw=1.25,
            ms=2.1,
            zorder=3,
        )
    ax_mean.set_xlim(-0.5, len(channel_order) - 0.5)
    ax_mean.set_ylim(-0.5, len(channel_order) - 0.5)
    ax_mean.set_yticks([])
    ax_mean.tick_params(axis="x", labelsize=6.2, length=2.2)
    ax_mean.spines[["top", "right", "left"]].set_visible(False)
    if show_xlabel:
        ax_mean.set_xlabel("Rank", fontsize=8.0)
    else:
        ax_mean.set_xlabel("")
        ax_mean.tick_params(axis="x", bottom=False, labelbottom=False)
    return {
        "image": image,
        "axes": {"heatmap": ax_heat, "mean_rank": ax_mean},
        "cluster_counts": cluster_counts,
        "displayed_events": int(len(events)),
    }


def _plot_combined(
    records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    distribution: dict[str, int],
) -> tuple[Path, list[dict[str, Any]]]:
    examples = _select_examples(records, rows)
    fig = plt.figure(figsize=(7.8, 7.05), facecolor="white")
    main = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.0, 2.30],
        hspace=0.24,
        left=0.085,
        right=0.985,
        bottom=0.070,
        top=0.952,
    )
    scan = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=main[0],
        width_ratios=[0.92, 1.25, 1.05],
        wspace=0.43,
    )
    scan_axes = [fig.add_subplot(scan[0, idx]) for idx in range(3)]
    _draw_scan_panels(scan_axes, rows, distribution)
    example_grid = gridspec.GridSpecFromSubplotSpec(
        4,
        2,
        subplot_spec=main[1],
        width_ratios=[8.7, 1.35],
        height_ratios=[1, 1, 1, 0.10],
        hspace=0.50,
        wspace=0.095,
    )
    example_meta: list[dict[str, Any]] = []
    first_example_axis: plt.Axes | None = None

    for row_index, record in enumerate(examples):
        arr = fig1._load_exemplar_arrays(record, max_events=2500)
        public = public_patient_label(record["dataset"], record["subject"])
        row_spec = example_grid[row_index, :].subgridspec(
            1, 1, wspace=0.0
        )[0, 0]
        draw = _draw_multi_k_row(
            fig,
            row_spec,
            arr,
            display_label=public,
            show_xlabel=row_index == len(examples) - 1,
        )
        if first_example_axis is None:
            first_example_axis = draw["axes"]["heatmap"]
        corr = _pair_corr(record)
        example_meta.append(
            {
                "dataset": record["dataset"],
                "patient_id": public,
                "selected_k": int(arr["chosen_k"]),
                "inter_template_spearman_r": corr,
                "n_valid_events": int(arr["valid_events"].size),
                "n_displayed_events": int(draw["displayed_events"]),
                "n_channels": int(len(arr["channel_names"])),
                "channel_order": arr["ordered_names"],
                "cluster_counts": draw["cluster_counts"],
            }
        )

    cax = fig.add_subplot(example_grid[3, 0])
    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="viridis")
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([0.0, 1.0], labels=["First", "Last"])
    cbar.ax.tick_params(labelsize=7, length=2, pad=1)
    cbar.outline.set_linewidth(0.55)
    cbar.set_label("Recruitment order", fontsize=7.5, labelpad=1)
    fig.add_subplot(example_grid[3, 1]).axis("off")

    fig.canvas.draw()
    if first_example_axis is None:
        raise RuntimeError("No example axis was created")
    for label, axis in (("A", scan_axes[0]), ("B", first_example_axis)):
        pos = axis.get_position()
        fig.text(
            pos.x0 - 0.060,
            pos.y1 + 0.040,
            label,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

    png = FIG_DIR / "supp_fig3_k_scan_and_multipatient_templates.png"
    pdf = FIG_DIR / "supp_fig3_k_scan_and_multipatient_templates.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, example_meta


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    records = _load_records()
    scan_csv = OUT_ROOT / "k2_to_k10_subject_scan.csv"
    cached = _read_completed_scan(scan_csv)
    if cached is None:
        rows, distribution = _scan_table(records)
    else:
        rows, distribution = cached
    _write_csv(scan_csv, rows)
    scan_png = _plot_scan(rows, distribution)
    combined_png, examples = _plot_combined(records, rows, distribution)

    canonical_distribution = Counter(adaptive_k(record) for record in records)
    changes = [
        {
            "dataset": row["dataset"],
            "patient_id": public_patient_label(
                row["dataset"], row["subject"]
            ),
            "canonical_k": int(row["canonical_selected_k_2_to_8"]),
            "extended_k": int(row["selected_k_2_to_10"]),
        }
        for row in rows[:: len(K_VALUES)]
        if int(row["canonical_selected_k_2_to_8"]) != int(row["selected_k_2_to_10"])
    ]
    caption_title = (
        "Model-order scanning identifies two temporal templates in most "
        "patients while retaining higher-order solutions."
    )
    caption_body = (
        f"**A,** Unsupervised clustering was evaluated for K = 2\u201310 using "
        f"event-by-contact recruitment-rank features after masking "
        f"non-participating contacts in {len(records)} patients; for each "
        f"patient, K was selected as the solution with the highest median "
        f"silhouette among those with median adjusted mutual information "
        f"(AMI) \u2265 {AMI_THRESHOLD:.2f} across seeds and a smallest-cluster "
        f"fraction \u2265 {MIN_CLUSTER_FRACTION:.2f}. The left panel shows the "
        f"selected-K distribution ({distribution['2']}/{len(records)} patients "
        f"selected K = 2), the middle panel shows each patient's silhouette "
        f"difference relative to K = 2 with the cohort median and interquartile "
        f"range, and the right panel shows the percentage of patients passing "
        f"both gates at each K. **B,** Recruitment-order maps for two K = 4 "
        f"solutions (E11 and E19) and one K = 6 solution (Y10), with 2,500 "
        f"displayed population events per patient; colour denotes relative "
        f"within-event recruitment order from first to last, light grey denotes "
        f"a non-participating contact and hatched gaps separate templates. "
        f"Profiles at right show the contact-wise mean rank for each template, "
        f"with shading indicating \u00b11 s.d."
    )
    metadata = {
        "figure": "Supplementary Figure 3",
        "caption": (
            f"Supplementary Fig. 3 | {caption_title} "
            f"{caption_body.replace('**', '')}"
        ),
        "feature_contract": (
            "masked lagPatRank features with non-participating channels imputed "
            "by each event's participating-channel median"
        ),
        "scan_range": [2, 10],
        "n_subjects": len(records),
        "selection_rule": (
            "highest median silhouette among K passing median AMI >= 0.70 and "
            "worst minimum cluster fraction >= 0.10"
        ),
        "extended_selected_k_distribution": distribution,
        "canonical_k2_to_k8_distribution": {
            str(k): int(canonical_distribution[k]) for k in sorted(canonical_distribution)
        },
        "selection_changes_after_extending_to_k10": changes,
        "example_selection": (
            "fixed manuscript-labelled cases comprising two K=4 solutions "
            "and one K=6 solution"
        ),
        "example_painter": (
            "Figure-1E-compatible masked event heatmap and mean-rank profile; "
            "non-K=2 cases retain all selected clusters"
        ),
        "examples": examples,
        "outputs": {
            "scan_png": str(scan_png.relative_to(ROOT)),
            "combined_png": str(combined_png.relative_to(ROOT)),
        },
    }
    (OUT_ROOT / "supp_fig3_k_scan_templates_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (FIG_DIR / "README.md").write_text(
        "### supp_fig3a_k2_to_k10_scan.png\n\n"
        "该文件是 Supplementary Fig. 3A 的独立导出，统计定义与下方完整图注一致。"
        "AMI 在这里表示聚类种子间稳定性，而不是正文的事件-模板 matching index。\n\n"
        "**关注点**：完整投稿图请使用下方的 A+B 合并版本。\n\n"
        "### supp_fig3_k_scan_and_multipatient_templates.png\n\n"
        f"**Supplementary Fig. 3 | {caption_title}**\n\n"
        f"{caption_body}\n\n"
        "**关注点**：B 说明 K=2 是队列主导结构但不是先验强制结果；队列结论仍由"
        "上排全部 40 名患者的扫描承担。\n",
        encoding="utf-8",
    )
    print(scan_png)
    print(combined_png)


if __name__ == "__main__":
    main()
