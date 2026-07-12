#!/usr/bin/env python3
"""Build paper Figure 1 as individual temporal-scaffold panels.

Figure scope is intentionally temporal: group-event observation, refined-HFO
SOZ anchor, a masked representative temporal-template example, and cohort-level
MI/uplift. Spatial-axis evidence belongs to the next main figure.

Each scientific message is rendered as its own file under the strict
``fig1-panel<id>`` naming so the panels can be assembled externally:

    fig1-panela   group-event phenomenon (reused Y3 demo, copied verbatim)
    fig1-panelb1  Yuquan refined-HFO count -> clinical SOZ ROC
    fig1-panelb2  Epilepsiae refined-HFO count -> clinical SOZ ROC
    fig1-panelc1  exemplar temporal order heatmap + rank distribution
    fig1-panelc2  exemplar TA/TB clustered heatmap + mean rank
    fig1-paneld1  MI data vs permutation null (40 subjects)   [masked shared-participant]
    fig1-paneld2  within-template Kendall-tau uplift (40 subjects)

No composite is emitted. The across-time reproducibility (split-half/odd-even)
panel is intentionally not part of Figure 1.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import plot_interictal_propagation as propagation_plot  # noqa: E402
import plot_refine_soz_validation as refine_plot  # noqa: E402
from src.interictal_propagation import _valid_event_indices  # noqa: E402
from src.plot_style import COL_EPI, COL_YQ  # noqa: E402


DEFAULT_OUTPUT = ROOT / "results/paper-ready-figure/fig1_interictal_hfo_temporal_scaffold/figures"
DEFAULT_GROUP_EVENT = (
    ROOT
    / "results/paper-ready-figure/fig1_hfo_group_event_demo/figures"
    / "yuquan_y3_hfo_group_event_demo.png"
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
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return [str(png.relative_to(ROOT)), str(pdf.relative_to(ROOT))]


# ---------------------------------------------------------------------------
# Panel a: reuse the Y3 group-event demo verbatim
# ---------------------------------------------------------------------------
def _render_panel_a(output_dir: Path, group_event_png: Path) -> dict:
    files: list[str] = []
    for suffix in (".png", ".pdf"):
        src = group_event_png.with_suffix(suffix)
        if not src.exists():
            continue
        dst = output_dir / f"fig1-panela{suffix}"
        shutil.copyfile(src, dst)
        files.append(str(dst.relative_to(ROOT)))
    if not files:
        raise FileNotFoundError(f"Panel a source not found: {group_event_png}")
    return {
        "panel_id": "a",
        "files": files,
        "producer": "scripts/paper_figures/plot_fig1_hfo_group_event_legacy_style.py",
        "reused_from": str(group_event_png.relative_to(ROOT)),
        "public_patient_label": "Yuquan Y3",
        "artifacts": [
            "/mnt/yuquan_data/yuquan_24h_edf/chengshuai/FC10477Q.edf",
            "/mnt/yuquan_data/yuquan_24h_edf/chengshuai/FC10477Q_gpu.npz",
            "/mnt/yuquan_data/yuquan_24h_edf/chengshuai/FC10477Q_packedTimes.npy",
        ],
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
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    _style_axis(ax)
    return {
        "dataset": dataset,
        "n_subjects": int(len(rows)),
        "mean_subject_auc": float(np.mean(aucs)),
        "median_subject_auc": float(np.median(aucs)),
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
        "inter_corr": inter_corr,
        "p_text": p_text,
        "chosen_k": int(adaptive["chosen_k"]),
        "within_cluster_tau": float(adaptive["within_cluster_tau_mean"]),
        "overall_tau": float(adaptive["overall_tau"]),
    }


def _render_exemplar_order(output_dir: Path, arr: dict, display_label: str) -> dict:
    ranks = arr["ranks"]
    bools = arr["bools"]
    channel_order = arr["channel_order"]
    display_events = arr["display_events"]
    fig = plt.figure(figsize=(9.8, 4.0), facecolor="white")
    outer = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[6.0, 1.15], wspace=0.05,
        left=0.075, right=0.9, bottom=0.14, top=0.8,
    )
    left = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0, 0], height_ratios=[20, 1], hspace=0.06,
    )
    ax_raw = fig.add_subplot(left[0])
    ax_strip = fig.add_subplot(left[1], sharex=ax_raw)
    ax_dist = fig.add_subplot(outer[0, 1])
    im = propagation_plot._plot_rank_heatmap(
        ax_raw,
        ranks[channel_order][:, display_events],
        arr["ordered_names"],
        title="Events over time",
        display_bools=bools[channel_order][:, display_events],
        ytick_fontsize=8,
        title_fontsize=12,
        xtick_fontsize=8,
    )
    ax_raw.tick_params(axis="x", labelbottom=False)
    ax_raw.set_xlabel("")
    propagation_plot._plot_daynight_strip(ax_strip, arr["day_mask"])
    ax_strip.set_xlabel("Population events (time-ordered)  ·  strip: day (white) / night (black)", fontsize=8)
    propagation_plot._plot_rank_histogram(
        ax_dist,
        ranks,
        bools,
        arr["valid_events"],
        np.arange(ranks.shape[0], dtype=int),
        arr["channel_names"],
        title="Rank dist.",
        show_ylabels=False,
        label_fontsize=8,
        title_fontsize=10,
        xtick_fontsize=7,
    )
    cbar = fig.colorbar(im, ax=ax_dist, orientation="vertical", fraction=0.14, pad=0.2)
    cbar.set_label("First → Last", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)
    ax_raw.text(
        0.006,
        1.07,
        (
            f"{display_label}  |  n={arr['valid_events'].size:,}  |  "
            f"inter-template r={arr['inter_corr']:.2f}  |  MI {arr['p_text']}"
        ),
        transform=ax_raw.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )
    _panel_label(ax_raw, "c1", x=-0.06, y=1.22)
    files = _save_panel(fig, output_dir, "fig1-panelc1")
    return {
        "panel_id": "c1",
        "files": files,
        "producer_source": "scripts/plot_interictal_propagation.py --masked-features --pr3 --paper-style",
        "record": f"results/interictal_propagation_masked/per_subject/{arr['dataset']}_{arr['subject']}.json",
        "public_patient_label": display_label,
        "n_valid_events": int(arr["valid_events"].size),
        "displayed_events": int(arr["display_events"].size),
        "masked_features": True,
        "daynight_strip": True,
        "inter_template_spearman_r": arr["inter_corr"],
        "mi_p_display": arr["p_text"],
    }


def _render_exemplar_template(output_dir: Path, arr: dict) -> dict:
    ranks = arr["ranks"]
    bools = arr["bools"]
    channel_order = arr["channel_order"]
    clustered_events = arr["clustered_events"]
    fig = plt.figure(figsize=(9.8, 4.0), facecolor="white")
    outer = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[6.0, 1.15], wspace=0.05,
        left=0.075, right=0.9, bottom=0.15, top=0.74,
    )
    ax_cluster = fig.add_subplot(outer[0, 0])
    ax_mean = fig.add_subplot(outer[0, 1])
    im = propagation_plot._plot_rank_heatmap(
        ax_cluster,
        ranks[channel_order][:, clustered_events],
        arr["ordered_names"],
        title="",
        display_bools=bools[channel_order][:, clustered_events],
        ytick_fontsize=8,
        title_fontsize=12,
        xtick_fontsize=8,
    )
    propagation_plot._plot_cluster_boundaries(
        ax_cluster,
        arr["clustered_labels"],
        ranks.shape[0],
        line_color="#d00000",
        line_width=4.0,
        line_style="-",
        label_fontsize=10,
        label_box=False,
        boundary_band=True,
        label_names=["TA", "TB"],
        label_y_offset=0.5,
    )
    ax_cluster.set_xlabel("Population events (clustered)", fontsize=9)
    propagation_plot._plot_cluster_rank_fig4(
        ax_mean,
        ranks,
        bools,
        arr["valid_events"],
        arr["labels"],
        channel_order,
        arr["channel_names"],
        title="Mean rank",
        label_fontsize=8,
        title_fontsize=10,
        xtick_fontsize=7,
        legend_fontsize=7,
        show_legend=False,
        invert_yaxis=False,
        show_ylabels=False,
    )
    cbar = fig.colorbar(im, ax=ax_mean, orientation="vertical", fraction=0.14, pad=0.2)
    cbar.set_label("First → Last", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)
    fig.text(
        0.075,
        0.95,
        (
            f"KMeans k={arr['chosen_k']}  |  within-template τ={arr['within_cluster_tau']:.3f}  |  "
            f"overall τ={arr['overall_tau']:.3f}  |  inter-corr={arr['inter_corr']:.2f}"
        ),
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
    )
    _panel_label(ax_cluster, "c2", x=-0.06, y=1.18)
    files = _save_panel(fig, output_dir, "fig1-panelc2")
    return {
        "panel_id": "c2",
        "files": files,
        "producer_source": "scripts/plot_interictal_propagation.py --masked-features --pr3 --paper-style",
        "record": f"results/interictal_propagation_masked/per_subject/{arr['dataset']}_{arr['subject']}.json",
        "masked_features": True,
        "chosen_k": arr["chosen_k"],
        "within_cluster_tau": arr["within_cluster_tau"],
        "overall_tau": arr["overall_tau"],
    }


# ---------------------------------------------------------------------------
# Panel d: cohort MI (d1) and within-template uplift (d2)
# ---------------------------------------------------------------------------
def _plot_mi(ax: plt.Axes, records: list[dict]) -> dict:
    colors = {"yuquan": COL_YQ, "epilepsiae": EPI_STAT_COLOR}
    positions = {"yuquan": (0.0, 0.75), "epilepsiae": (2.0, 2.75)}
    summary = {}
    for dataset in ("yuquan", "epilepsiae"):
        subset = [r for r in records if r.get("dataset") == dataset]
        data = np.asarray([r["legacy_mi"]["mi_mean"] for r in subset], dtype=float)
        null = np.asarray([r["legacy_mi"]["permuted_mean_median"] for r in subset], dtype=float)
        x_data, x_null = positions[dataset]
        vp = ax.violinplot([data, null], positions=[x_data, x_null], widths=0.55, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(colors[dataset])
            body.set_edgecolor("none")
            body.set_alpha(0.22)
        jitter = np.linspace(-0.10, 0.10, len(data)) if len(data) else np.array([])
        ax.scatter(x_data + jitter, data, s=12, color=colors[dataset], alpha=0.8, edgecolors="white", linewidths=0.25)
        ax.scatter(x_null + jitter, null, s=10, color="0.55", alpha=0.65, edgecolors="none")
        ax.plot([x_data - 0.18, x_data + 0.18], [np.median(data)] * 2, color="black", lw=1.2)
        ax.plot([x_null - 0.18, x_null + 0.18], [np.median(null)] * 2, color="black", lw=1.2)
        summary[dataset] = {
            "n": len(subset),
            "n_significant": int(sum(bool(r["legacy_mi"]["significant"]) for r in subset)),
            "data_median": float(np.median(data)),
            "null_median": float(np.median(null)),
        }
    ax.set_xticks([0.0, 0.75, 2.0, 2.75])
    ax.set_xticklabels(["Data", "Null", "Data", "Null"], fontsize=7)
    ax.text(0.375, -0.2, "Yuquan", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, fontweight="bold", color=colors["yuquan"])
    ax.text(2.375, -0.2, "Epilepsiae", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, fontweight="bold", color=colors["epilepsiae"])
    ax.set_ylabel("Matching index", fontsize=9)
    ax.set_title("Temporal organization exceeds null", fontsize=10, pad=6)
    n_sig = sum(v["n_significant"] for v in summary.values())
    n_tot = sum(v["n"] for v in summary.values())
    ax.text(0.98, 0.94, f"{n_sig}/{n_tot} significant", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    _style_axis(ax)
    return summary


def _render_mi_panel(output_dir: Path, records: list[dict]) -> dict:
    fig, ax = plt.subplots(figsize=(4.7, 3.7), facecolor="white")
    summary = _plot_mi(ax, records)
    _panel_label(ax, "d1", x=-0.16, y=1.14)
    files = _save_panel(fig, output_dir, "fig1-paneld1")
    return {
        "panel_id": "d1",
        "files": files,
        "producer_source": "scripts/plot_interictal_propagation.py --masked-features",
        "records": "results/interictal_propagation_masked/per_subject/*.json",
        "matching_index": summary,
        "status": "masked_shared_participant",
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
        ax.scatter(x, y, s=17, color=colors[record["dataset"]], alpha=0.78, edgecolors="white", linewidths=0.3)
    overall_arr = np.asarray(overall, dtype=float)
    within_arr = np.asarray(within, dtype=float)
    hi = max(0.85, float(np.nanmax([overall_arr.max(), within_arr.max()])) + 0.03)
    ax.plot([0, hi], [0, hi], ls="--", lw=0.8, color="0.55")
    ax.set_xlim(-0.02, hi)
    ax.set_ylim(-0.02, hi)
    ax.set_xlabel("Overall Kendall τ", fontsize=9)
    ax.set_ylabel("Within-template τ", fontsize=9)
    median_uplift = float(np.median(within_arr - overall_arr))
    n_above = int(np.sum(within_arr > overall_arr))
    ax.set_title("Template-aware stereotypy", fontsize=10, pad=6)
    ax.text(
        0.04,
        0.93,
        f"median Δτ = {median_uplift:+.3f}\n{n_above}/{len(records)} above diagonal",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    ax.set_aspect("equal", adjustable="box")
    _style_axis(ax)
    return {"n": int(len(records)), "median_uplift": median_uplift, "n_above_diagonal": n_above}


def _render_uplift_panel(output_dir: Path, records: list[dict]) -> dict:
    fig, ax = plt.subplots(figsize=(3.9, 3.9), facecolor="white")
    summary = _plot_uplift(ax, records)
    _panel_label(ax, "d2", x=-0.2, y=1.12)
    files = _save_panel(fig, output_dir, "fig1-paneld2")
    return {
        "panel_id": "d2",
        "files": files,
        "producer_source": "scripts/plot_interictal_propagation.py --masked-features",
        "records": "results/interictal_propagation_masked/per_subject/*.json",
        "uplift": summary,
    }


def _write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        """### Figure 1 独立 panel（temporal scaffold）

论文 Figure 1 拆成独立 panel 文件，命名 `fig1-panel<id>`，不再拼成一张 composite；跨时间复现（split-half）已移出本图。

- `fig1-panela` — 真实 Yuquan 群体 HFO 事件：80–250 Hz 波形、normalized spectrogram、时频质心顺序（复用 Y3 demo 成品，直接复制）。
- `fig1-panelb1` / `fig1-panelb2` — Yuquan / Epilepsiae 用当前 refined-count pipeline 现场重算的 SOZ ROC。
- `fig1-panelc1` — Epilepsiae E3（内部 artifact 958，phantom-rank 修复后）的时间顺序热图 + rank 分布。
- `fig1-panelc2` — 同一例患的 TA/TB 聚类顺序 + mean rank。
- `fig1-paneld1` — 40 例患者内 masked（shared-participant）MI data vs permutation null（40/40 significant，masked median 0.228；仅共同参与触点，phantom 伪秩已排除）。
- `fig1-paneld2` — 40 例 within-template stereotypy uplift。

本图只支持“interictal HFO population events exhibit recurrent patient-specific temporal organization”。三维 SEEG 接触点空间轴不在本图中。完整输入与统计定义见同目录 `figure1_interictal_hfo_temporal_scaffold_metadata.json`。

**关注点**：先看 panela 的真实群体事件与红色质心轨迹；再确认 panelc1/c2 非参与 cell 为灰色、TA/TB 分界清楚；最后核对 paneld1 的 40/40 masked MI 与 paneld2 的 40/40 uplift。
""",
        encoding="utf-8",
    )


def build(
    output_dir: Path,
    group_event_png: Path,
    exemplar_subject: str,
    exemplar_label: str,
    max_events: int,
) -> dict:
    propagation_plot._apply_masked_paths()
    records = _load_temporal_records()
    if len(records) != 40:
        raise ValueError(f"Expected 40 masked temporal records, found {len(records)}")
    exemplar = next(
        record
        for record in records
        if record.get("dataset") == "epilepsiae" and str(record.get("subject")) == exemplar_subject
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_rcparams()

    panel_a = _render_panel_a(output_dir, group_event_png)
    panel_b1 = _render_roc_panel(output_dir, "yuquan", "b1")
    panel_b2 = _render_roc_panel(output_dir, "epilepsiae", "b2")
    arr = _load_exemplar_arrays(exemplar, max_events=max_events)
    panel_c1 = _render_exemplar_order(output_dir, arr, display_label=exemplar_label)
    panel_c2 = _render_exemplar_template(output_dir, arr)
    panel_d1 = _render_mi_panel(output_dir, records)
    panel_d2 = _render_uplift_panel(output_dir, records)

    panels = {
        "a": panel_a,
        "b1": panel_b1,
        "b2": panel_b2,
        "c1": panel_c1,
        "c2": panel_c2,
        "d1": panel_d1,
        "d2": panel_d2,
    }
    outputs = [f for panel in panels.values() for f in panel["files"]]

    metadata = {
        "schema_version": "paper_figure1_temporal_scaffold_panels_v2",
        "claim_scope": "Interictal HFO population events exhibit recurrent patient-specific temporal organization.",
        "forbidden_upgrade": "This figure alone does not establish a shared 3D propagation axis.",
        "producer": "scripts/paper_figures/plot_fig1_interictal_hfo_temporal_scaffold.py",
        "panel_id_stamped": True,
        "composite_emitted": False,
        "split_half_included": False,
        "paneld1_statistic": "masked shared-participant MI (phantom ranks excluded); 40/40 significant, cohort median 0.228.",
        "outputs": outputs,
        "panels": panels,
    }
    metadata_path = output_dir / "figure1_interictal_hfo_temporal_scaffold_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(output_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group-event-png", type=Path, default=DEFAULT_GROUP_EVENT)
    parser.add_argument("--exemplar-subject", default="958")
    parser.add_argument("--exemplar-label", default="Epilepsiae E3")
    parser.add_argument("--max-events", type=int, default=2000)
    args = parser.parse_args()
    metadata = build(
        output_dir=args.output_dir.resolve(),
        group_event_png=args.group_event_png.resolve(),
        exemplar_subject=str(args.exemplar_subject),
        exemplar_label=str(args.exemplar_label),
        max_events=int(args.max_events),
    )
    print(json.dumps(metadata["outputs"], ensure_ascii=False))


if __name__ == "__main__":
    main()
