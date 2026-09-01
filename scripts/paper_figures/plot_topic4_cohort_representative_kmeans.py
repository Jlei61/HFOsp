#!/usr/bin/env python3
"""Clustering check for the cohort's representative subject.

Layout and painter follow `docs/topic4_data_driven_snn_figure_spec.md` §3:
clustered event heatmap, per-contact rank distribution, cluster rank profile
and the model-versus-patient matrix, drawn with the accepted Figure 1E painter
rather than a look-alike renderer.  The subject is whichever one sits closest
to the cohort median, chosen in the aggregator before anything is drawn.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import plot_interictal_propagation as propagation_plot  # noqa: E402
from scripts.paper_figures.plot_fig1_interictal_hfo_temporal_scaffold import (  # noqa: E402
    _draw_fig1e_cluster_row,
)
from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (  # noqa: E402
    TA_COLOR,
    TB_COLOR,
    normalize_event_ranks,
)
from src.topic4_cohort_formal_scoring import score_readout  # noqa: E402

DEFAULT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_formal_v1.json"
MODE_COLORS = (TA_COLOR, TB_COLOR)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _representative_seed(row: dict) -> int:
    """The confirmation network closest to this subject's own median."""
    deltas = np.asarray([
        seed["delta_null_median_minus_observed"] for seed in row["per_seed"]
    ], float)
    median = float(np.median(deltas))
    order = np.lexsort((
        [seed["seed"] for seed in row["per_seed"]], np.abs(deltas - median),
    ))
    return int(row["per_seed"][int(order[0])]["seed"])


def _column_mean(values: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        counts = np.isfinite(values).sum(axis=0)
        total = np.nansum(values, axis=0)
    return np.divide(total, counts, out=np.full(values.shape[1], np.nan),
                     where=counts > 0)


def _column_quantile(values: np.ndarray, quantile: float) -> np.ndarray:
    output = np.full(values.shape[1], np.nan)
    for column in range(values.shape[1]):
        finite = values[np.isfinite(values[:, column]), column]
        if finite.size:
            output[column] = float(np.quantile(finite, quantile))
    return output


def _spearman_matrix(model: np.ndarray, patient: np.ndarray) -> np.ndarray:
    from scipy.stats import spearmanr

    matrix = np.full((2, 2), np.nan)
    for row in range(2):
        for column in range(2):
            finite = np.isfinite(model[row]) & np.isfinite(patient[column])
            if int(finite.sum()) >= 3:
                matrix[row, column] = spearmanr(
                    model[row, finite], patient[column, finite],
                ).statistic
    return matrix


def build(config: dict, result: dict) -> dict:
    from scripts.aggregate_topic4_data_driven_snn_cohort_formal import Cohort

    cohort = Cohort(config)
    subject_id = result["representative_subject"]["subject_id"]
    row = next(
        item for item in result["canonical_subjects"]
        if item["subject_id"] == subject_id
    )
    index = next(
        position for position, subject in enumerate(cohort.subjects)
        if subject["subject_id"] == subject_id
    )
    seed = _representative_seed(row)
    npz_path = (
        cohort.output_root / "workers" / f"{row['candidate_id']}_seed_{seed}.npz"
    )
    ranks = cohort.worker_ranks(npz_path, "canonical", index)
    kwargs = cohort.scorer_kwargs(cohort.subjects[index], "heldout")
    score = score_readout(ranks, **kwargs)
    if score.get("status") != "EVALUABLE":
        raise RuntimeError(f"representative subject is not evaluable: {score.get('status')}")

    names = kwargs["contact_names"]
    n_contacts = len(names)
    readable = np.isfinite(ranks).sum(axis=1) >= int(kwargs["minimum_contacts"])
    model_display = normalize_event_ranks(ranks[readable]) * (n_contacts - 1)
    labels = np.asarray(score["natural_kmeans"]["aligned_labels"], int)

    with np.load(
        cohort.target_root / f"{subject_id}_target.npz", allow_pickle=False,
    ) as loaded:
        train = np.vstack([
            np.asarray(loaded["train_ta_rank_samples"], float),
            np.asarray(loaded["train_tb_rank_samples"], float),
        ])
        heldout = {
            mode: np.asarray(loaded[f"heldout_{mode}_rank_samples"], float)
            for mode in ("ta", "tb")
        }
    channel_order = propagation_plot._fixed_channel_order(train.T, np.isfinite(train.T))
    patient_profiles = np.asarray([
        _column_mean(heldout[mode]) for mode in ("ta", "tb")
    ]) * (n_contacts - 1)
    patient_low = np.asarray([
        _column_quantile(heldout[mode], 0.05) for mode in ("ta", "tb")
    ]) * (n_contacts - 1)
    patient_high = np.asarray([
        _column_quantile(heldout[mode], 0.95) for mode in ("ta", "tb")
    ]) * (n_contacts - 1)
    model_profiles = np.asarray([
        _column_mean(model_display[labels == mode]) for mode in (0, 1)
    ])
    return {
        "subject_id": subject_id,
        "candidate_id": row["candidate_id"],
        "seed": seed,
        "npz": npz_path,
        "names": names,
        "n_contacts": n_contacts,
        "channel_order": channel_order,
        "model_display": model_display,
        "labels": labels,
        "patient_profiles": patient_profiles,
        "patient_low": patient_low,
        "patient_high": patient_high,
        "matrix": _spearman_matrix(model_profiles, patient_profiles),
        "score": score,
        "row": row,
    }


def render(data: dict, output: Path) -> dict:
    names = np.asarray(data["names"])
    labels = data["labels"]
    ranks = data["model_display"].T
    order = np.argsort(labels, kind="stable")
    arr = {
        "ranks": ranks,
        "bools": np.isfinite(ranks),
        "channel_order": data["channel_order"],
        "ordered_names": names[data["channel_order"]].tolist(),
        "clustered_events_all": order,
        "clustered_labels_all": labels[order],
        "valid_events": np.arange(len(labels), dtype=int),
        "labels": labels,
        "channel_names": names.tolist(),
    }
    fig = plt.figure(figsize=(19.2, 5.15), facecolor="white")
    outer = fig.add_gridspec(
        1, 5, width_ratios=(4.35, 0.16, 1.05, 1.82, 1.42),
        left=0.047, right=0.945, bottom=0.17, top=0.87, wspace=0.34,
    )
    draw = _draw_fig1e_cluster_row(
        fig, outer, 0, arr, column_indices=(0, 1, 3),
        gap_half_width_events=max(1, int(round(0.012 * len(labels)))),
        cluster_label_names=["MTA", "MTB"],
        cluster_colors=list(MODE_COLORS),
        mean_profile_label_names=["MTA", "MTB"],
        heatmap_ytick_fontsize=11.5, cluster_label_fontsize=12.5,
        mean_label_fontsize=13, mean_xtick_fontsize=11,
    )
    heatmap_ax = draw["axes"]["heatmap"]
    profile_ax = draw["axes"]["mean_rank"]
    heatmap_ax.set_title("")
    heatmap_ax.set_ylabel("electrode contact", fontsize=13)
    heatmap_ax.tick_params(axis="x", labelsize=11)
    profile_ax.set_title("cluster rank profile", fontsize=14, weight="bold", pad=8)
    profile_ax.set_xlabel(
        f"mean rank position (0 = first, {data['n_contacts'] - 1} = last)",
        fontsize=13,
    )
    profile_ax.tick_params(axis="x", labelsize=11)

    rank_grid = outer[0, 2].subgridspec(2, 1, height_ratios=(20, 1), hspace=0.06)
    rank_ax = fig.add_subplot(rank_grid[0])
    fig.add_subplot(rank_grid[1]).axis("off")
    propagation_plot._plot_rank_histogram(
        rank_ax, arr["ranks"], arr["bools"], arr["valid_events"],
        data["channel_order"], names.tolist(), title="rank distribution",
        show_ylabels=False, label_fontsize=13, title_fontsize=14,
        xtick_fontsize=11, ridge_spacing=0.10, smooth_sigma_bins=0.72,
        smooth_ridge_height=0.12,
    )

    y = np.arange(data["n_contacts"], dtype=float)
    ordered = data["channel_order"]
    for mode in (0, 1):
        finite = np.isfinite(data["patient_profiles"][mode, ordered])
        profile_ax.fill_betweenx(
            y[finite], data["patient_low"][mode, ordered][finite],
            data["patient_high"][mode, ordered][finite],
            color=MODE_COLORS[mode], alpha=0.08, linewidth=0,
        )
        profile_ax.plot(
            data["patient_profiles"][mode, ordered][finite], y[finite], "--",
            color=MODE_COLORS[mode], lw=1.7,
        )
    profile_ax.legend(
        handles=[
            Line2D([0], [0], color=TA_COLOR, lw=2.3, marker="o", ms=4.5, label="MTA"),
            Line2D([0], [0], color=TB_COLOR, lw=2.3, marker="o", ms=4.5, label="MTB"),
            Line2D([0], [0], color=TA_COLOR, lw=1.7, ls="--", label="TA"),
            Line2D([0], [0], color=TB_COLOR, lw=1.7, ls="--", label="TB"),
        ],
        frameon=False, fontsize=10, ncol=2, loc="upper right",
        bbox_to_anchor=(1.0, 1.0), columnspacing=0.8, handlelength=1.5,
        borderaxespad=0.2,
    )

    matrix_grid = outer[0, 4].subgridspec(2, 1, height_ratios=(20, 1), hspace=0.06)
    matrix_ax = fig.add_subplot(matrix_grid[0])
    fig.add_subplot(matrix_grid[1]).axis("off")
    matrix = data["matrix"]
    counts = np.bincount(labels, minlength=2)
    valid = bool(np.all(counts >= 3) and np.isfinite(matrix).all())
    matrix_ax.set_xticks((0, 1), ("TA", "TB"), fontsize=13, weight="bold")
    matrix_ax.set_yticks((0, 1), ("MTA", "MTB"), fontsize=13, weight="bold")
    for tick, color in zip(matrix_ax.get_xticklabels(), MODE_COLORS):
        tick.set_color(color)
    for tick, color in zip(matrix_ax.get_yticklabels(), MODE_COLORS):
        tick.set_color(color)
    matrix_ax.set_aspect("equal")
    matrix_ax.set_title("cluster vs patient", fontsize=14, weight="bold", pad=8)
    colorbar_ax = matrix_ax.inset_axes([1.08, 0.0, 0.065, 1.0])
    if valid:
        image = matrix_ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        for row in range(2):
            for column in range(2):
                shade = "white" if abs(matrix[row, column]) >= 0.55 else "#111111"
                matrix_ax.text(column, row, f"{matrix[row, column]:+.2f}",
                               ha="center", va="center", fontsize=13,
                               color=shade, weight="bold")
        bar = fig.colorbar(image, cax=colorbar_ax)
        bar.set_label("Spearman rho", fontsize=12)
        bar.ax.tick_params(labelsize=10.5)
    else:
        colorbar_ax.axis("off")
        matrix_ax.text(0.5, 0.5, "N/A", transform=matrix_ax.transAxes,
                       ha="center", va="center", fontsize=20, color="#9B2F2A",
                       weight="bold")

    natural = data["score"]["natural_kmeans"]
    verdict = data["row"]
    fig.text(
        0.5, 0.985,
        f"{data['subject_id']} | one network, {len(labels)} events | "
        f"beat its own within-shaft shuffle: "
        f"{'yes' if verdict['subject_endpoint_pass'] else 'no'}",
        ha="center", va="top", fontsize=12.5, fontweight="bold",
        color=TA_COLOR if verdict["subject_endpoint_pass"] else "#9B2F2A",
    )
    silhouette = natural["silhouette"]
    silhouette_text = "n/a" if silhouette is None else f"{float(silhouette):.2f}"
    fig.text(
        0.5, 0.045,
        f"cluster sizes {counts.tolist()} | seed AMI "
        f"{natural['seed_ami_median']:.2f} | silhouette {silhouette_text} | "
        f"events outside the patient's own event cloud "
        f"{data['score']['ood_fraction']:.0%} | "
        f"matrix is the pooled cluster-versus-patient profile correlation",
        ha="center", va="bottom", fontsize=10, color="#333333",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    return {
        "png": _relative(output.with_suffix(".png")),
        "png_sha256": _sha256(output.with_suffix(".png")),
        "pdf": _relative(output.with_suffix(".pdf")),
        "pdf_sha256": _sha256(output.with_suffix(".pdf")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output_root = ROOT / config["output_root"]
    result_path = output_root / "cohort_result.json"
    if not result_path.exists():
        print(json.dumps({"status": "COHORT_RESULT_ABSENT_FIGURE_SKIPPED"}))
        return
    result = json.loads(result_path.read_text())
    data = build(config, result)
    figures = output_root / "figures"
    files = render(data, figures / "topic4_cohort_representative_kmeans")
    metadata = {
        "schema_version": "topic4_cohort_representative_kmeans_v1",
        "science_status": {
            "cohort_status": result["status"],
            "verdict": result["verdict"],
            "result_json_sha256": _sha256(result_path),
        },
        "subject_id": data["subject_id"],
        "candidate_id": data["candidate_id"],
        "confirmation_seed": data["seed"],
        "worker_npz": _relative(data["npz"]),
        "worker_npz_sha256": _sha256(data["npz"]),
        "cluster_counts": np.bincount(data["labels"], minlength=2).tolist(),
        "cluster_vs_patient_matrix": data["matrix"].tolist(),
        "supervised_vs_patient_matrix": np.asarray(
            data["score"]["supervised_profile_matrix"], float,
        ).tolist(),
        "natural_kmeans": {
            key: value for key, value in data["score"]["natural_kmeans"].items()
            if not isinstance(value, np.ndarray)
        },
        "ood_fraction": data["score"]["ood_fraction"],
        "subject_endpoint_pass": data["row"]["subject_endpoint_pass"],
        "files": files,
        "scientific_boundary": config["claim_boundary"],
    }
    (figures / "topic4_cohort_representative_kmeans_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({"subject": data["subject_id"], **files}, indent=2))


if __name__ == "__main__":
    main()
