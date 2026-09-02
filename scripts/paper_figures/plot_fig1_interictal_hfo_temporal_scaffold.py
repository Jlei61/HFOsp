#!/usr/bin/env python3
"""Build paper Figure 1 as independent panel-level outputs.

Figure scope is intentionally temporal: group-event observation, refined-HFO
SOZ anchor, a masked representative temporal-template example, and cohort-level
MI/uplift. Spatial-axis evidence belongs to the next main figure.

The manuscript's Figure 1A is a hand-drawn schematic and is intentionally not
produced or retained here.  Code-generated panels follow the manuscript panel
letters exactly so they can be assembled externally without aliases:

    fig1-panelb1  legacy manually annotated HFO morphology set (n=178)
    fig1-panelb2  group-event phenomenon (reused Y3 demo, copied verbatim)
    fig1-panelc   time-ordered masked rank heatmap + rank distributions
    fig1-paneld   MI data vs permutation null (40 subjects)
    fig1-panele   TA/TB clustered heatmap + mean-rank profiles
    fig1-panelf   within-template matching-index uplift (40 subjects)

No composite is emitted. The across-time reproducibility (split-half/odd-even)
panel is intentionally not part of Figure 1.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import plot_interictal_propagation as propagation_plot  # noqa: E402
import plot_refine_soz_validation as refine_plot  # noqa: E402
from src.interictal_propagation import _valid_event_indices  # noqa: E402
from src.plot_style import COL_EPI, COL_YQ  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results/paper-ready-figure/fig1/figures"
DEFAULT_GROUP_EVENT = (
    ROOT
    / "results/paper-ready-figure/archive/2026-08-09_fig1_source_material/fig1_hfo_group_event_demo/figures"
    / "yuquan_y3_hfo_group_event_demo.png"
)
DEFAULT_SINGLE_HFO = (
    ROOT
    / "results/paper-ready-figure/archive/2026-08-09_fig1_source_material/fig1_hfo_group_event_demo/figures"
    / "legacy_hfo_n178_schematic.png"
)
MASKED_ROOT = ROOT / "results/interictal_propagation_masked"
HFO_ROOT = ROOT / "results/hfo_detection"
PARAMS_JSON = ROOT / "config/subject_params.json"
SOZ_JSON = {
    "yuquan": ROOT / "results/yuquan_soz_core_channels.json",
    "epilepsiae": ROOT / "results/epilepsiae_soz_core_channels.json",
}
EPI_ROC_COLOR = "#7A3E87"
EPI_STAT_COLOR = "#B07A74"
FIG1E_TEMPLATE_COLORS = {"TA": "#B2182B", "TB": "#2166AC"}
FIG1_RANK_COLORBAR_TITLE = "Heatmap rank\nFirst → Last"
FIG1F_INSET_BOUNDS = [0.68, 0.13, 0.28, 0.36]
FIG1F_INSET_YLABEL_FONTSIZE = 9.0
FIG1F_INSET_TICK_FONTSIZE = 8.0


def _panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8, length=2.5)


def _apply_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def _save_panel(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return [str(png.relative_to(ROOT)), str(pdf.relative_to(ROOT))]


# ---------------------------------------------------------------------------
# Panel B: legacy HFO morphology set (B1) + Y3 group-event demo (B2)
# ---------------------------------------------------------------------------
def _render_panel_b_sources(output_dir: Path, single_hfo_png: Path, group_event_png: Path) -> dict:
    def _copy(src_png: Path, stem: str) -> list[str]:
        src_pdf = src_png.with_suffix(".pdf")
        if src_pdf.exists():
            dst_pdf = output_dir / f"{stem}.pdf"
            shutil.copyfile(src_pdf, dst_pdf)
            subprocess.run(
                ["pdftoppm", "-png", "-singlefile", "-r", "600", str(dst_pdf),
                 str(output_dir / stem)],
                check=True,
            )
            return [
                str((output_dir / f"{stem}.png").relative_to(ROOT)),
                str(dst_pdf.relative_to(ROOT)),
            ]
        if not src_png.exists():
            raise FileNotFoundError(f"Panel {stem} source not found: {src_png}")
        dst_png = output_dir / f"{stem}.png"
        shutil.copyfile(src_png, dst_png)
        return [str(dst_png.relative_to(ROOT))]

    return {
        "b1": {
            "panel_id": "b1",
            "files": _copy(single_hfo_png, "fig1-panelb1"),
            "producer": "scripts/paper_figures/plot_fig1_single_hfo_schematic.py",
            "reused_from": str(single_hfo_png.relative_to(ROOT)),
            "source_set": "legacy manually annotated HFO morphology artifact",
            "n_hfo_snippets": 178,
            "artifacts": [
                "ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef/zhangkexuan_pickSigs.npz",
                "ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef/zhangkexuan_annot_v4.pik",
            ],
            "content": "178 overlaid HFO snippets + yellow mean, raw and baseline-normalized mean spectrograms",
        },
        "b2": {
            "panel_id": "b2",
            "files": _copy(group_event_png, "fig1-panelb2"),
            "producer": "scripts/paper_figures/plot_fig1_hfo_group_event_legacy_style.py",
            "reused_from": str(group_event_png.relative_to(ROOT)),
            "public_patient_label": "Yuquan Y3",
            "artifacts": [
                "/mnt/yuquan_data/yuquan_24h_edf/chengshuai/FC10477Q.edf",
                "/mnt/yuquan_data/yuquan_24h_edf/chengshuai/FC10477Q_gpu.npz",
                "/mnt/yuquan_data/yuquan_24h_edf/chengshuai/FC10477Q_packedTimes.npy",
            ],
        },
    }


# ---------------------------------------------------------------------------
# Panel b: refined-HFO count -> SOZ ROC, one file per cohort
# ---------------------------------------------------------------------------
def _load_refined_roc(dataset: str) -> list[dict]:
    soz_all = refine_plot.load_soz_channels(SOZ_JSON[dataset])
    params = refine_plot.load_subject_params(PARAMS_JSON)
    configured = {
        key for key in params.get(dataset, {}) if not str(key).startswith("_")
    }
    rows: list[dict] = []
    for subject in sorted(configured):
        subject_dir = HFO_ROOT / subject
        if not subject_dir.is_dir():
            continue
        counts, names = refine_plot.load_refine_counts(subject_dir)
        soz = soz_all.get(subject, [])
        if counts is None or names is None or not soz:
            continue
        soz_idx, _ = refine_plot.classify_channels_soz(names, soz)
        auc_value, fpr, tpr = refine_plot.compute_auc(counts, soz_idx, len(names))
        if np.isfinite(auc_value) and fpr.size:
            rows.append(
                {
                    "subject": subject,
                    "auc": float(auc_value),
                    "fpr": np.asarray(fpr, dtype=float),
                    "tpr": np.asarray(tpr, dtype=float),
                    "n_channels": int(len(names)),
                    "n_soz": int(len(soz_idx)),
                }
            )
    return rows


def _plot_roc(ax: plt.Axes, dataset: str, rows: list[dict]) -> dict:
    color = COL_YQ if dataset == "yuquan" else EPI_ROC_COLOR
    label = "Yuquan" if dataset == "yuquan" else "Epilepsiae"
    grid = np.linspace(0.0, 1.0, 201)
    curves = []
    for row in rows:
        ax.plot(row["fpr"], row["tpr"], color="0.78", lw=0.6, alpha=0.55, zorder=1)
        curves.append(np.interp(grid, row["fpr"], row["tpr"]))
    curve_array = np.asarray(curves, dtype=float)
    mean_curve = np.mean(curve_array, axis=0)
    sem_curve = (
        np.std(curve_array, axis=0, ddof=1) / np.sqrt(len(rows))
        if len(rows) > 1
        else np.zeros_like(grid)
    )
    ax.fill_between(
        grid,
        np.clip(mean_curve - sem_curve, 0.0, 1.0),
        np.clip(mean_curve + sem_curve, 0.0, 1.0),
        color=color,
        alpha=0.22,
        linewidth=0,
        zorder=2,
    )
    ax.plot(grid, mean_curve, color=color, lw=2.2, zorder=3)
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="0.55", zorder=0)
    aucs = np.asarray([row["auc"] for row in rows], dtype=float)
    ax.text(
        0.97,
        0.05,
        f"mean AUC = {np.mean(aucs):.3f}\nn = {len(rows)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=color,
    )
    ax.set_title(label, color=color, fontsize=12, fontweight="bold", pad=6)
    ax.set_xlabel("FPR", fontsize=11)
    ax.set_ylabel("TPR", fontsize=11)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    _style_axis(ax)
    return {
        "dataset": dataset,
        "n_subjects": int(len(rows)),
        "mean_subject_auc": float(np.mean(aucs)),
        "median_subject_auc": float(np.median(aucs)),
        "axis_limits": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]},
        "curve_band": "SEM across subject ROC curves after interpolation to 201 FPR points",
        "subjects": [row["subject"] for row in rows],
    }


def _render_roc_panel(output_dir: Path, dataset: str, panel_id: str) -> dict:
    fig, ax = plt.subplots(figsize=(3.9, 3.9), facecolor="white")
    summary = _plot_roc(ax, dataset, _load_refined_roc(dataset))
    _panel_label(ax, panel_id, x=-0.22, y=1.12)
    files = _save_panel(fig, output_dir, f"fig1-panel{panel_id}")
    return {
        "panel_id": panel_id,
        "files": files,
        "producer_source": "scripts/plot_refine_soz_validation.py",
        "artifact_root": "results/hfo_detection/<subject>/{*_gpu.npz,_refineGpu.npz}",
        "label_sources": [str(p.relative_to(ROOT)) for p in SOZ_JSON.values()],
        "score": "refined events_count per contact",
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Panel c: masked exemplar, order (c1) and TA/TB template (c2)
# ---------------------------------------------------------------------------
def _load_temporal_records() -> list[dict]:
    records = []
    for path in sorted((MASKED_ROOT / "per_subject").glob("*.json")):
        with path.open() as handle:
            record = json.load(handle)
        if "error" not in record and record.get("adaptive_cluster"):
            records.append(record)
    return records


def _assert_masked_mi_records(records: list[dict]) -> None:
    invalid = []
    for record in records:
        if record.get("legacy_mi", {}).get("masked") is not True:
            invalid.append(f"{record.get('dataset')}:{record.get('subject')}")
    if invalid:
        raise ValueError(
            "Panel D requires masked MI records; unmasked/missing flag for "
            + ", ".join(invalid)
        )


def _load_exemplar_arrays(record: dict, max_events: int) -> dict:
    dataset = str(record["dataset"])
    subject = str(record["subject"])
    subject_dir = propagation_plot._resolve_subject_dir(dataset, subject)
    loaded = propagation_plot._load_lagpat(subject_dir)
    ranks = np.asarray(loaded["ranks"], dtype=float)
    bools = np.asarray(loaded["bools"], dtype=bool)
    channel_names = list(loaded["channel_names"])
    event_abs_times = np.asarray(loaded["event_abs_times"], dtype=float)
    valid_events = _valid_event_indices(bools, min_participating=3)
    display_events = propagation_plot._sample_event_indices(valid_events, max_events=max_events)
    day_mask = propagation_plot._formal_day_mask(dataset, event_abs_times[display_events])
    channel_order = propagation_plot._fixed_channel_order(ranks, bools)
    ordered_names = [channel_names[idx] for idx in channel_order]
    adaptive = record["adaptive_cluster"]
    labels = np.asarray(adaptive["labels"], dtype=int)
    if labels.size != valid_events.size:
        raise ValueError(f"{dataset}:{subject}: adaptive labels do not align with valid events")
    display_mask = np.isin(valid_events, display_events)
    display_labels = labels[display_mask]
    clustered_order = np.argsort(display_labels, kind="stable")
    clustered_events = display_events[clustered_order]
    clustered_labels = display_labels[clustered_order]
    clustered_order_all = np.argsort(labels, kind="stable")
    clustered_events_all = valid_events[clustered_order_all]
    clustered_labels_all = labels[clustered_order_all]
    corr = np.asarray(adaptive.get("inter_cluster_corr_matrix", []), dtype=float)
    inter_corr = float(corr[0, 1]) if corr.shape == (2, 2) else float("nan")
    p_value = float(record.get("legacy_mi", {}).get("p_value", np.nan))
    p_text = "p < 0.001" if np.isfinite(p_value) and p_value < 0.001 else f"p = {p_value:.3f}"
    return {
        "dataset": dataset,
        "subject": subject,
        "ranks": ranks,
        "bools": bools,
        "channel_names": channel_names,
        "valid_events": valid_events,
        "display_events": display_events,
        "day_mask": day_mask,
        "channel_order": channel_order,
        "ordered_names": ordered_names,
        "labels": labels,
        "clustered_events": clustered_events,
        "clustered_labels": clustered_labels,
        "clustered_events_all": clustered_events_all,
        "clustered_labels_all": clustered_labels_all,
        "inter_corr": inter_corr,
        "mi_mean": float(record.get("legacy_mi", {}).get("mi_mean", np.nan)),
        "p_text": p_text,
        "chosen_k": int(adaptive["chosen_k"]),
        "within_cluster_tau": float(adaptive["within_cluster_tau_mean"]),
        "overall_tau": float(adaptive["overall_tau"]),
    }


def _panel_c_row_axes(fig: plt.Figure, outer: gridspec.GridSpec, row: int) -> tuple:
    """Create identical heatmap/colorbar/summary geometry for one Panel-c row."""
    left = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[row, 0], height_ratios=[20, 1], hspace=0.06,
    )
    colorbar_column = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[row, 1], height_ratios=[20, 1], hspace=0.06,
    )
    right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[row, 2], height_ratios=[20, 1], hspace=0.06,
    )
    ax_main = fig.add_subplot(left[0])
    ax_bottom = fig.add_subplot(left[1], sharex=ax_main)
    ax_cbar = fig.add_subplot(colorbar_column[0])
    ax_cbar_dummy = fig.add_subplot(colorbar_column[1])
    ax_cbar_dummy.axis("off")
    ax_right = fig.add_subplot(right[0])
    ax_right_dummy = fig.add_subplot(right[1], sharex=ax_right)
    ax_right_dummy.axis("off")
    return ax_main, ax_bottom, ax_cbar, ax_right


def _place_panel_c_colorbar(fig: plt.Figure, image, ax_cbar: plt.Axes) -> None:
    cbar = fig.colorbar(image, cax=ax_cbar, orientation="vertical")
    pos = ax_cbar.get_position()
    ax_cbar.set_position([pos.x0 - 0.018, pos.y0, pos.width, pos.height])
    cbar.set_label("")
    cbar.ax.text(
        0.5,
        1.035,
        FIG1_RANK_COLORBAR_TITLE,
        transform=cbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        linespacing=0.95,
        clip_on=False,
    )
    cbar.ax.tick_params(labelsize=8.5, length=2)


def _draw_fig1e_cluster_row(
    fig: plt.Figure,
    outer,
    row: int,
    arr: dict,
    *,
    display_label: str = "",
    panel_label: str = "",
    show_colorbar: bool = True,
    show_heatmap_xlabel: bool = True,
    show_mean_xlabel: bool = True,
    display_label_fontweight: str = "bold",
    display_label_fontsize: float = 10,
    display_label_y: float = 1.17,
    heatmap_ytick_fontsize: float = 9.5,
    cluster_label_fontsize: float = 10,
    mean_label_fontsize: float = 10.5,
    mean_xtick_fontsize: float = 8.5,
    column_indices: tuple[int, int, int] = (0, 1, 2),
    gap_half_width_events: int | None = None,
    cluster_label_names: list[str] | None = None,
    cluster_colors: list[str] | None = None,
    mean_profile_label_names: list[str] | None = None,
) -> dict:
    """Draw one clustered-template row with the exact Figure-1E painter.

    This is the single accepted drawing entry point for the clustered event
    heatmap, TA/TB separator, masked phantom cells, colorbar, and cluster mean
    rank profile.  Supplementary multi-patient examples call this function
    directly so they cannot silently drift into a different visual contract.
    """
    heatmap_col, colorbar_col, mean_col = column_indices
    left = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[row, heatmap_col],
        height_ratios=[20, 1], hspace=0.06,
    )
    colorbar_column = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[row, colorbar_col],
        height_ratios=[20, 1], hspace=0.06,
    )
    right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[row, mean_col],
        height_ratios=[20, 1], hspace=0.06,
    )
    ax_cluster = fig.add_subplot(left[0])
    ax_cluster_dummy = fig.add_subplot(left[1], sharex=ax_cluster)
    ax_cbar = fig.add_subplot(colorbar_column[0])
    ax_cbar_dummy = fig.add_subplot(colorbar_column[1])
    ax_cbar_dummy.axis("off")
    ax_mean = fig.add_subplot(right[0])
    ax_mean_dummy = fig.add_subplot(right[1], sharex=ax_mean)
    ax_mean_dummy.axis("off")
    ax_cluster_dummy.axis("off")
    semantic_names = cluster_label_names or ["TA", "TB"]
    semantic_colors = cluster_colors or [
        FIG1E_TEMPLATE_COLORS[name] for name in semantic_names
    ]
    ranks = arr["ranks"]
    bools = arr["bools"]
    channel_order = arr["channel_order"]
    clustered_events = arr["clustered_events_all"]
    im = propagation_plot._plot_rank_heatmap(
        ax_cluster,
        ranks[channel_order][:, clustered_events],
        arr["ordered_names"],
        title="",
        display_bools=bools[channel_order][:, clustered_events],
        ytick_fontsize=heatmap_ytick_fontsize,
        title_fontsize=12,
        xtick_fontsize=8,
    )
    cluster_boundary = int(np.sum(arr["clustered_labels_all"] == 0))
    gap_half_width = (
        max(24, int(round(0.006 * clustered_events.size)))
        if gap_half_width_events is None else int(gap_half_width_events)
    )
    if gap_half_width < 0:
        raise ValueError("gap_half_width_events must be non-negative")
    ax_cluster.axvspan(
        cluster_boundary - gap_half_width,
        cluster_boundary + gap_half_width,
        facecolor="white",
        edgecolor="0.62",
        hatch="////",
        linewidth=0.0,
        zorder=12,
    )
    ax_cluster.plot(
        [cluster_boundary - gap_half_width, cluster_boundary + gap_half_width],
        [0.0, 0.0],
        transform=ax_cluster.get_xaxis_transform(),
        color="white",
        lw=4,
        solid_capstyle="butt",
        clip_on=False,
        zorder=13,
    )
    propagation_plot._plot_cluster_boundaries(
        ax_cluster,
        arr["clustered_labels_all"],
        ranks.shape[0],
        line_color="#d00000",
        line_width=0.0,
        line_style="-",
        label_fontsize=cluster_label_fontsize,
        label_box=False,
        boundary_band=False,
        label_names=semantic_names,
        label_colors=semantic_colors,
        label_y_offset=0.5,
    )
    for text in ax_cluster.texts:
        if text.get_text().split(maxsplit=1)[0] in semantic_names:
            text.set_fontweight("bold")
    if show_heatmap_xlabel:
        ax_cluster.set_xlabel("Population events (clustered)", fontsize=10.5)
    else:
        ax_cluster.set_xlabel("")
        ax_cluster.tick_params(axis="x", bottom=False, labelbottom=False)
    propagation_plot._plot_cluster_rank_fig4(
        ax_mean,
        ranks,
        bools,
        arr["valid_events"],
        arr["labels"],
        channel_order,
        arr["channel_names"],
        title="",
        label_fontsize=mean_label_fontsize,
        title_fontsize=10,
        xtick_fontsize=mean_xtick_fontsize,
        legend_fontsize=7,
        show_legend=False,
        invert_yaxis=False,
        show_ylabels=False,
        marker_size=3.5,
        line_colors=semantic_colors,
        label_names=mean_profile_label_names or semantic_names,
    )
    if show_mean_xlabel:
        ax_mean.set_xlabel("Rank", fontsize=10.5)
    else:
        ax_mean.set_xlabel("")
        ax_mean.tick_params(axis="x", bottom=False, labelbottom=False)
    if show_colorbar:
        _place_panel_c_colorbar(fig, im, ax_cbar)
    else:
        ax_cbar.axis("off")
    if display_label:
        ax_cluster.text(
            0.0,
            display_label_y,
            display_label,
            transform=ax_cluster.transAxes,
            ha="left",
            va="bottom",
            fontsize=display_label_fontsize,
            fontweight=display_label_fontweight,
            clip_on=False,
        )
    if panel_label:
        _panel_label(ax_cluster, panel_label, x=-0.06, y=1.25)
    return {
        "axes": {
            "heatmap": ax_cluster,
            "colorbar": ax_cbar,
            "mean_rank": ax_mean,
        },
        "cluster_boundary": cluster_boundary,
        "gap_half_width": gap_half_width,
        "image": im,
        "template_semantic_colors": dict(zip(semantic_names, semantic_colors)),
    }


def _render_temporal_order_panel(
    output_dir: Path,
    arr: dict,
    display_label: str,
) -> dict:
    fig = plt.figure(figsize=(11.6, 3.35), facecolor="white")
    outer = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[8.4, 0.16, 1.35],
        wspace=0.13,
        left=0.065,
        right=0.985,
        bottom=0.14,
        top=0.88,
    )
    ax_raw, ax_strip, ax_cbar, ax_dist = _panel_c_row_axes(fig, outer, 0)
    ranks = arr["ranks"]
    bools = arr["bools"]
    channel_order = arr["channel_order"]
    display_events = arr["display_events"]
    image = propagation_plot._plot_rank_heatmap(
        ax_raw,
        ranks[channel_order][:, display_events],
        arr["ordered_names"],
        title="",
        display_bools=bools[channel_order][:, display_events],
        ytick_fontsize=9.5,
        title_fontsize=12,
        xtick_fontsize=8,
    )
    ax_raw.tick_params(axis="x", labelbottom=False)
    ax_raw.set_xlabel("")
    propagation_plot._plot_daynight_strip(ax_strip, arr["day_mask"])
    ax_strip.set_xlabel("Population events (time-ordered)", fontsize=10.5)
    propagation_plot._plot_rank_histogram(
        ax_dist,
        ranks,
        bools,
        arr["valid_events"],
        channel_order,
        arr["channel_names"],
        title="Rank dist.",
        show_ylabels=False,
        label_fontsize=10.5,
        title_fontsize=10,
        xtick_fontsize=8.5,
    )
    _place_panel_c_colorbar(fig, image, ax_cbar)
    ax_raw.text(
        0.006,
        1.07,
        f"{display_label}  |  n={arr['valid_events'].size:,}",
        transform=ax_raw.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )
    ax_raw.legend(
        handles=[
            Patch(facecolor="white", edgecolor="0.25", linewidth=0.8, label="Day"),
            Patch(facecolor="black", edgecolor="black", linewidth=0.8, label="Night"),
        ],
        loc="lower right",
        bbox_to_anchor=(0.78, 1.025),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    files = _save_panel(fig, output_dir, "fig1-panelc")
    return {
        "panel_id": "c",
        "files": files,
        "producer_source": "scripts/plot_interictal_propagation.py --masked-features --pr3 --paper-style",
        "layout": "time-ordered heatmap, day/night strip, colorbar, and rank distributions",
        "figure_size_inches": [11.6, 3.35],
        "axis_label_fontsize_points": 10.5,
        "channel_label_fontsize_points": 9.5,
        "colorbar_label_fontsize_points": 10.5,
        "rank_colorbar": {
            "title": FIG1_RANK_COLORBAR_TITLE,
            "placement": "horizontal title above colorbar",
            "side_label_removed": True,
        },
        "panel_column_order": ["event_heatmap", "colorbar", "rank_summary"],
        "record": f"results/interictal_propagation_masked/per_subject/{arr['dataset']}_{arr['subject']}.json",
        "public_patient_label": display_label,
        "n_valid_events": int(arr["valid_events"].size),
        "displayed_events": int(arr["display_events"].size),
        "masked_features": True,
        "daynight_strip": True,
        "daynight_legend": {
            "labels": ["Day", "Night"],
            "placement": "same title row as patient label, upper-right of heatmap",
            "removed_from_xlabel": True,
        },
        "masked_mi_mean": arr["mi_mean"],
        "rank_distribution_helper": "scripts/plot_interictal_propagation.py::_plot_rank_histogram",
    }


def _render_clustered_template_panel(
    output_dir: Path,
    arr: dict,
) -> dict:
    fig = plt.figure(figsize=(11.6, 3.35), facecolor="white")
    outer = gridspec.GridSpec(
        1, 3, figure=fig, width_ratios=[8.4, 0.16, 1.35], wspace=0.13,
        left=0.065, right=0.985, bottom=0.14, top=0.88,
    )
    drawn = _draw_fig1e_cluster_row(fig, outer, 0, arr)
    cluster_boundary = int(drawn["cluster_boundary"])
    gap_half_width = int(drawn["gap_half_width"])
    files = _save_panel(fig, output_dir, "fig1-panele")
    return {
        "panel_id": "e",
        "files": files,
        "producer_source": "scripts/plot_interictal_propagation.py --masked-features --pr3 --paper-style",
        "record": f"results/interictal_propagation_masked/per_subject/{arr['dataset']}_{arr['subject']}.json",
        "n_valid_events": int(arr["valid_events"].size),
        "displayed_events": int(arr["clustered_events_all"].size),
        "cluster_counts": {
            str(cluster_id): int(np.sum(arr["labels"] == cluster_id))
            for cluster_id in np.unique(arr["labels"])
        },
        "all_valid_events_displayed": True,
        "cluster_separator": {
            "style": "white gap with gray diagonal hatch and interrupted x-axis spine",
            "boundary_event_index": cluster_boundary,
            "gap_half_width_events": gap_half_width,
        },
        "masked_features": True,
        "template_semantic_colors": drawn["template_semantic_colors"],
        "template_labels_bold": True,
        "rank_colorbar": {
            "title": FIG1_RANK_COLORBAR_TITLE,
            "placement": "horizontal title above colorbar",
            "side_label_removed": True,
        },
        "chosen_k": arr["chosen_k"],
        "within_cluster_tau": arr["within_cluster_tau"],
        "overall_tau": arr["overall_tau"],
        "inter_template_spearman_r": arr["inter_corr"],
    }


# ---------------------------------------------------------------------------
# Panels D and F: cohort MI and within-template uplift
# ---------------------------------------------------------------------------
def _plot_mi(ax: plt.Axes, records: list[dict]) -> dict:
    import scipy.stats as st

    colors = {"yuquan": COL_YQ, "epilepsiae": EPI_STAT_COLOR}
    positions = {"yuquan": (0.0, 0.6), "epilepsiae": (1.8, 2.4)}
    summary = {}
    bracket_tops = []
    for group_index, dataset in enumerate(("yuquan", "epilepsiae")):
        subset = [r for r in records if r.get("dataset") == dataset]
        data = np.asarray([r["legacy_mi"]["mi_mean"] for r in subset], dtype=float)
        null = np.asarray([r["legacy_mi"]["permuted_mean_median"] for r in subset], dtype=float)
        x_data, x_null = positions[dataset]
        propagation_plot._violin_with_scatter(
            ax, data, x_data, colors[dataset], width=0.5,
            scatter_size=26, rng_seed=42 + group_index,
        )
        propagation_plot._violin_with_scatter(
            ax, null, x_null, "#BBBBBB", width=0.5,
            scatter_size=13, alpha_body=0.15, rng_seed=99 + group_index,
        )
        _, p_value = st.mannwhitneyu(data, null, alternative="greater")
        bracket_y = float(max(np.max(data), np.max(null)) + 0.025)
        propagation_plot._add_significance_bracket(
            ax, x_data, x_null, bracket_y, float(p_value),
            dy=0.015, fontsize=11,
        )
        bracket_tops.append(bracket_y + 0.045)
        summary[dataset] = {
            "n": len(subset),
            "n_significant": int(sum(bool(r["legacy_mi"]["significant"]) for r in subset)),
            "data_median": float(np.median(data)),
            "null_median": float(np.median(null)),
            "p_value_mannwhitney_greater": float(p_value),
            "all_mi_records_masked": bool(all(r["legacy_mi"].get("masked") is True for r in subset)),
        }
    ax.set_xticks([0.0, 0.6, 1.8, 2.4])
    ax.set_xticklabels(["Data", "Null", "Data", "Null"], fontsize=8.5)
    ax.text(0.3, -0.125, "Yuquan", transform=ax.get_xaxis_transform(), ha="center", fontsize=9.5)
    ax.text(2.1, -0.125, "Epilepsiae", transform=ax.get_xaxis_transform(), ha="center", fontsize=9.5)
    ax.set_ylabel("MI", fontsize=10.5)
    ax.set_title("MI: data vs permutation null", fontsize=10.5, pad=8)
    ax.set_ylim(0.0, max(0.58, max(bracket_tops)))
    _style_axis(ax)
    return summary


def _render_mi_panel(output_dir: Path, records: list[dict]) -> dict:
    fig, ax = plt.subplots(figsize=(5.2, 3.9), facecolor="white")
    summary = _plot_mi(ax, records)
    files = _save_panel(fig, output_dir, "fig1-paneld")
    return {
        "panel_id": "d",
        "files": files,
        "producer_source": "scripts/plot_interictal_propagation.py --masked-features",
        "records": "results/interictal_propagation_masked/per_subject/*.json",
        "matching_index": summary,
        "status": "masked_shared_participant",
        "masked_mi_hard_check": True,
        "summary_display": "shared violin_with_scatter helper: violin + box/IQR + whiskers + subject points",
        "significance_display": "shared add_significance_bracket helper; Mann-Whitney U, data > null",
        "y_axis_starts_at_zero": True,
    }


def _plot_uplift_distribution_inset(
    ax: plt.Axes,
    records: list[dict],
    overall_arr: np.ndarray,
    within_arr: np.ndarray,
) -> dict:
    """Draw the compact paired MI summary used by the supplementary HFO-AUC panel."""
    import scipy.stats as st

    colors = {"yuquan": COL_YQ, "epilepsiae": EPI_STAT_COLOR}

    def _lighten(color: str, amount: float = 0.62) -> tuple[float, float, float]:
        rgb = np.asarray(matplotlib.colors.to_rgb(color), dtype=float)
        return tuple(rgb + (1.0 - rgb) * amount)

    inset = ax.inset_axes(FIG1F_INSET_BOUNDS, zorder=6)
    inset.set_facecolor("white")
    for record, single_value, multi_value in zip(records, overall_arr, within_arr):
        dataset_color = colors[str(record["dataset"])]
        inset.plot(
            [0, 1],
            [single_value, multi_value],
            color="0.38",
            lw=0.45,
            alpha=0.46,
            zorder=1,
        )
        inset.scatter(
            [0, 1],
            [single_value, multi_value],
            s=8.5,
            color=[_lighten(dataset_color), dataset_color],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.86,
            zorder=3,
        )

    means = [float(np.mean(overall_arr)), float(np.mean(within_arr))]
    inset.bar(
        [0, 1],
        means,
        width=0.58,
        color=["#D9D9D9", "#8AA0AA"],
        alpha=0.62,
        edgecolor="none",
        zorder=0,
    )
    inset.hlines(means, [-0.23, 0.77], [0.23, 1.23], color="black", lw=1.05, zorder=4)

    try:
        test = st.wilcoxon(within_arr, overall_arr, alternative="two-sided", method="auto")
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    except ValueError:
        statistic = float("nan")
        p_value = float("nan")
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

    bracket_y = float(max(np.max(overall_arr), np.max(within_arr)) + 0.035)
    cap = 0.012
    inset.plot(
        [0, 0, 1, 1],
        [bracket_y - cap, bracket_y, bracket_y, bracket_y - cap],
        color="black",
        lw=0.75,
        clip_on=False,
        zorder=5,
    )
    inset.text(
        0.5,
        bracket_y + 0.008,
        p_text,
        ha="center",
        va="bottom",
        fontsize=8.0,
        fontweight="bold" if "*" in p_text else "normal",
        clip_on=False,
    )
    inset.set_xlim(-0.42, 1.42)
    inset.set_ylim(0.0, max(0.90, bracket_y + 0.055))
    inset.set_xticks([0, 1])
    inset.set_xticklabels(["Single", "Multi"])
    inset.set_yticks([0.0, 0.4, 0.8])
    inset.set_ylabel(
        "MI", fontsize=FIG1F_INSET_YLABEL_FONTSIZE, labelpad=2.0,
    )
    inset.tick_params(
        axis="both",
        labelsize=FIG1F_INSET_TICK_FONTSIZE,
        length=2.4,
        width=0.75,
        pad=1.5,
    )
    inset.spines[["top", "right"]].set_visible(False)
    inset.spines[["left", "bottom"]].set_linewidth(0.65)

    return {
        "n_paired": int(len(records)),
        "single_template_mean": means[0],
        "single_template_median": float(np.median(overall_arr)),
        "multi_cluster_mean": means[1],
        "multi_cluster_median": float(np.median(within_arr)),
        "mean_delta": float(np.mean(within_arr - overall_arr)),
        "median_delta": float(np.median(within_arr - overall_arr)),
        "n_improved": int(np.sum(within_arr > overall_arr)),
        "wilcoxon_two_sided_statistic": statistic,
        "wilcoxon_two_sided_p": p_value,
        "significance_label": p_text,
        "display": "paired subject points and lines, mean bars, and paired Wilcoxon bracket",
        "reference_grammar": "Supplementary Fig. 2 raw-vs-synchronized HFO AUC",
        "layout_bounds_axes_fraction": FIG1F_INSET_BOUNDS,
        "layout_aspect": "narrow portrait inset, not square",
        "ylabel_fontsize_points": FIG1F_INSET_YLABEL_FONTSIZE,
        "tick_label_fontsize_points": FIG1F_INSET_TICK_FONTSIZE,
        "x_tick_labels": ["Single", "Multi"],
        "x_tick_label_meanings": {
            "Single": "single-template MI",
            "Multi": "multi-cluster MI",
        },
        "x_tick_labels_single_line": True,
    }


def _plot_uplift(ax: plt.Axes, records: list[dict]) -> dict:
    colors = {"yuquan": COL_YQ, "epilepsiae": EPI_STAT_COLOR}
    overall = []
    within = []
    for record in records:
        x = float(record["adaptive_cluster"]["overall_tau"])
        y = float(record["adaptive_cluster"]["within_cluster_tau_mean"])
        overall.append(x)
        within.append(y)
        ax.scatter(x, y, s=24, color=colors[record["dataset"]], alpha=0.82, edgecolors="white", linewidths=0.4)
    overall_arr = np.asarray(overall, dtype=float)
    within_arr = np.asarray(within, dtype=float)
    hi = max(0.85, float(np.nanmax([overall_arr.max(), within_arr.max()])) + 0.03)
    ax.fill_between([0, hi], [0, hi], [0, 0], color="#F0F0F0", zorder=0)
    ax.plot([0, hi], [0, hi], ls="--", lw=0.9, color="0.55", zorder=1)
    ax.set_xlim(0.0, hi)
    ax.set_ylim(0.0, hi)
    ax.set_xlabel("Overall MI", fontsize=18)
    ax.set_ylabel("Within-template MI", fontsize=18)
    median_uplift = float(np.median(within_arr - overall_arr))
    n_above = int(np.sum(within_arr > overall_arr))
    paired_distribution = _plot_uplift_distribution_inset(
        ax, records, overall_arr, within_arr,
    )
    legend = ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=colors["yuquan"], markeredgecolor="white", markersize=5, label="Yuquan"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=colors["epilepsiae"], markeredgecolor="white", markersize=5, label="Epilepsiae"),
        ],
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="0.55",
        framealpha=0.92,
        fancybox=False,
        fontsize=12,
        handlelength=0.8,
        handletextpad=0.25,
        labelspacing=0.25,
        borderpad=0.25,
        borderaxespad=0.35,
    )
    legend.get_frame().set_linewidth(0.8)
    ax.set_aspect("equal", adjustable="box")
    _style_axis(ax)
    ax.tick_params(axis="both", labelsize=16, length=6.0, width=1.3)
    for spine in ax.spines.values():
        if spine.get_visible():
            spine.set_linewidth(max(1.2, float(spine.get_linewidth())))
    return {
        "n": int(len(records)),
        "median_uplift": median_uplift,
        "median_matching_index_uplift": median_uplift,
        "n_above_diagonal": n_above,
        "display_labels": ["Overall MI", "Within-template MI"],
        "underlying_fields": ["adaptive_cluster.overall_tau", "adaptive_cluster.within_cluster_tau_mean"],
        "gray_below_diagonal_region": True,
        "gray_summary_text_removed": True,
        "paired_distribution_inset": paired_distribution,
        "axis_limits_start_at_zero": True,
        "dataset_legend": True,
        "dataset_legend_frame": {
            "visible": True,
            "facecolor": "white",
            "edgecolor": "0.55",
            "linewidth": 0.8,
            "producer_fontsize_points": 12.0,
            "rendered_fontsize_points": 12.0,
            "marker_size_points": 5.0,
        },
    }


def _render_uplift_panel(output_dir: Path, records: list[dict]) -> dict:
    fig, ax = plt.subplots(figsize=(3.9, 3.9), facecolor="white")
    summary = _plot_uplift(ax, records)
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.19, top=0.84)
    files = _save_panel(fig, output_dir, "fig1-panelf")
    return {
        "panel_id": "f",
        "files": files,
        "producer_source": "scripts/plot_interictal_propagation.py --masked-features",
        "records": "results/interictal_propagation_masked/per_subject/*.json",
        "uplift": summary,
    }


def _write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        """# Figure 1 panel 与完整排版输出

Figure 1A 是作者手绘示意图，不由代码生成，也不保存在本目录。独立 panel 文件不写左上角 panel 字母；字母只出现在 `fig1-complete-layout` 完整排版中。

### fig1-panelb1.png / .pdf

严格复用 legacy 人工标注的 178 段 HFO，展示黑色叠加波形、黄色均值及 raw/normalized 平均谱。三行 x 轴均铺满完整 0–0.6 s，首末频谱 cell 仅延展绘图边界、不修改谱值。

**关注点**：标题应为红色 `HFO n = 178`，两张谱在 x 轴左右均不应出现白带。

### fig1-panelb2.png / .pdf

展示 Yuquan Y3 的三个真实群体 HFO 事件及 normalized spectrogram。B1/B2 的谱量统一为 Gaussian-smoothed magnitude；B2 保留原 50 ms Hamming 窗以维持群体事件的时间分辨率，红点取主峰 ≥70% 连通增强区的同图加权质心。

**关注点**：每个红点应落在对应通道的高频能量增强团内，左右外边界无白带，只有事件之间保留白色分隔线。

### fig1-panelc.png / .pdf

展示 Epilepsiae E7 的 masked 时间顺序热图、原始 overlapping rank ridgeline 与 day/night strip。

**关注点**：非参与触点必须保持空白；day/night strip 与事件时间顺序严格对齐；Day/Night 使用黑白方块在患者标题同一行单独画 legend，xlabel 只保留 `Population events (time-ordered)`；colorbar 使用顶部水平标题 `Heatmap rank / First → Last`。

### fig1-paneld.png / .pdf

患者内 masked shared-participant MI data vs permutation null；严格复用原 cohort producer 的 violin + box/IQR + whiskers + subject points，并恢复 data-vs-null 显著性括号。phantom ranks 已排除。

**关注点**：producer 对 40 个输入执行 `legacy_mi.masked=true` 硬检查；y 轴从 0 开始，括号显示 cohort-level data > null 检验。

### fig1-panele.png / .pdf

将同一位 Epilepsiae E7 的全量 6,556 个有效事件按 masked KMeans k=2 的 TA/TB 标签重排，并展示两类 mean-rank 轮廓。

**关注点**：TA/TB 两个 n 之和必须等于 6,556；TA/TB 顶部标签必须粗体显示，TA 固定为红色 `#B2182B`，TB 固定为蓝色 `#2166AC`，并与右侧 mean-rank 曲线一致；colorbar 标题固定放在色条上方。

### fig1-panelf.png / .pdf

Overall 与 within-template MI 配对散点，量化分模板后的 matching uplift。底层数值仍来自 masked `overall_tau` / `within_cluster_tau_mean` rank-concordance fields，但图面统一使用 MI 简写。右下小 panel 复用补充图 HFO AUC 的配对语法，以患者连线、均值柱和配对 Wilcoxon 括号直接比较 single-template 与 multi-cluster MI。

**关注点**：两轴从 0 开始；对角线下方保留灰区；右上角 Yuquan/Epilepsiae 图例必须使用较小字号并带白底细边框；右下不再放灰色摘要字，而应以窄竖向、非方形布局显示 40 名患者的配对 MI 分布和显著性括号。x 轴用居中的单行短标签 `Single` / `Multi`，分别表示 single-template MI / multi-cluster MI。

### fig1-complete-layout.png / .pdf

将代码生成的 B–F panel 排为完整 Figure 1，并在完整画布上添加 B–F 字母。A 为作者手绘内容，因此本版保留 A 的外部拼入边界。

**关注点**：独立 panel 内不应重复出现字母；完整排版中的字母位置和字号应统一。
""",
        encoding="utf-8",
    )


def build(
    output_dir: Path,
    single_hfo_png: Path,
    group_event_png: Path,
    c1_exemplar_subject: str,
    c1_exemplar_label: str,
    max_events: int,
) -> dict:
    propagation_plot._apply_masked_paths()
    records = _load_temporal_records()
    if len(records) != 40:
        raise ValueError(f"Expected 40 masked temporal records, found {len(records)}")
    _assert_masked_mi_records(records)
    c1_exemplar = next(
        record
        for record in records
        if record.get("dataset") == "epilepsiae"
        and str(record.get("subject")) == c1_exemplar_subject
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_rcparams()

    panel_b = _render_panel_b_sources(output_dir, single_hfo_png, group_event_png)
    c1_arr = _load_exemplar_arrays(c1_exemplar, max_events=max_events)
    panel_c = _render_temporal_order_panel(output_dir, c1_arr, c1_exemplar_label)
    panel_d = _render_mi_panel(output_dir, records)
    panel_e = _render_clustered_template_panel(output_dir, c1_arr)
    panel_f = _render_uplift_panel(output_dir, records)

    panels = {
        **panel_b,
        "c": panel_c,
        "d": panel_d,
        "e": panel_e,
        "f": panel_f,
    }
    outputs = [f for panel in panels.values() for f in panel["files"]]

    metadata = {
        "schema_version": "paper_figure1_independent_panels_v5",
        "panelf_canonical_contract": {
            "contract_id": "fig1f_single_template_vs_multi_cluster_paired_inset_v1",
            "locked_on": "2026-09-02",
            "required_visual": "paired subject lines and points, mean bars, and paired Wilcoxon bracket in the lower-right inset",
            "forbidden_visual": "gray median-delta summary text in the lower-right region",
            "statistics": "two-sided paired Wilcoxon on adaptive_cluster.overall_tau vs adaptive_cluster.within_cluster_tau_mean",
        },
        "panele_canonical_contract": {
            "contract_id": "fig1e_ta_red_tb_blue_semantic_colors_v1",
            "template_semantic_colors": FIG1E_TEMPLATE_COLORS,
            "required_visual": "TA labels and mean-rank profile are red; TB labels and mean-rank profile are blue",
        },
        "claim_scope": "Interictal HFO population events exhibit recurrent patient-specific temporal organization.",
        "forbidden_upgrade": "This figure alone does not establish a shared 3D propagation axis.",
        "producer": "scripts/paper_figures/plot_fig1_interictal_hfo_temporal_scaffold.py",
        "panel_id_stamped": {
            "individual_panels": False,
            "note": "panel letters are added only by fig1-complete-layout; Figure 1A is hand-drawn and absent",
        },
        "figure1a": "hand-drawn; intentionally not generated or retained in paper-ready-figure",
        "composite_emitted": False,
        "split_half_included": False,
        "paneld_statistic": "masked shared-participant MI (phantom ranks excluded); 40/40 significant, cohort median 0.228.",
        "outputs": outputs,
        "panels": panels,
    }
    metadata_path = output_dir.parent / "figure1_panel_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(output_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group-event-png", type=Path, default=DEFAULT_GROUP_EVENT)
    parser.add_argument("--single-hfo-png", type=Path, default=DEFAULT_SINGLE_HFO)
    parser.add_argument("--c1-exemplar-subject", default="442")
    parser.add_argument("--c1-exemplar-label", default="Epilepsiae E7")
    parser.add_argument("--max-events", type=int, default=2000)
    args = parser.parse_args()
    metadata = build(
        output_dir=args.output_dir.resolve(),
        single_hfo_png=args.single_hfo_png.resolve(),
        group_event_png=args.group_event_png.resolve(),
        c1_exemplar_subject=str(args.c1_exemplar_subject),
        c1_exemplar_label=str(args.c1_exemplar_label),
        max_events=int(args.max_events),
    )
    print(json.dumps(metadata["outputs"], ensure_ascii=False))


if __name__ == "__main__":
    main()
