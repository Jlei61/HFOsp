#!/usr/bin/env python3
"""Render the rev20-DC dual-core canary KMeans and endpoint audit."""
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
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_d6_natural_kmeans import (  # noqa: E402
    best_binary_alignment, natural_kmeans, normalize_event_ranks,
    patient_profiles,
)
from src.topic4_rev20_dual_core_endpoint import _prototype_matrix  # noqa: E402
from src.topic4_shaft_aware import contract_groups  # noqa: E402
from src.topic4_shaft_aware_direction import assign_direction_modes  # noqa: E402
from src.topic4_rev20_dual_core_endpoint import (  # noqa: E402
    embedding_from_training_arrays,
)


MODE_COLORS = ("#C43C39", "#277DA1")
SHAFT_COLORS = {"ICL": "#E67E22", "SCL": "#159EAE"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, path: str) -> Path:
    local = ROOT / path
    return local if local.exists() else artifact_root / path


def _npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]).copy() for key in loaded.files}


def _load_contract(config: dict, artifact_root: Path):
    training_path = _resolve(
        artifact_root, config["inputs"]["patient_training_target"]["path"],
    )
    if _sha256(training_path) != config["inputs"]["patient_training_target"][
            "sha256"]:
        raise RuntimeError("patient training target changed")
    training = _npz(training_path)
    support_path = _resolve(
        artifact_root, config["inputs"]["patient_support_config"]["path"],
    )
    support = json.loads(support_path.read_text())
    contract_path = _resolve(
        artifact_root, support["inputs"]["contact_contract"]["path"],
    )
    classifier_path = _resolve(
        artifact_root, support["inputs"]["old_ab_train_only_classifier"]["path"],
    )
    contract = json.loads(contract_path.read_text())
    classifier = json.loads(classifier_path.read_text())["direction_classifier"]
    return training, contract, classifier


def _contact_overlay(ax, xy, names):
    names = np.asarray(names).astype(str)
    for shaft, color in SHAFT_COLORS.items():
        selected = np.char.startswith(names, shaft)
        ax.plot(xy[selected, 0], xy[selected, 1], color=color, lw=1.0, zorder=4)
        ax.scatter(
            xy[selected, 0], xy[selected, 1], s=22, color=color,
            edgecolor="white", linewidth=0.6, zorder=5,
        )


def _field_panel(ax, arrays):
    positions = np.asarray(arrays["positions_E"], float)
    h = np.asarray(arrays["h"], float)
    edges = np.linspace(0.0, 20.0, 101)
    weighted, _, _ = np.histogram2d(
        positions[:, 1], positions[:, 0], bins=(edges, edges), weights=h,
    )
    count, _, _ = np.histogram2d(
        positions[:, 1], positions[:, 0], bins=(edges, edges),
    )
    field = np.divide(weighted, count, out=np.zeros_like(weighted), where=count > 0)
    image = ax.imshow(
        gaussian_filter(field, 0.7), origin="lower", extent=(0, 20, 0, 20),
        cmap="plasma", vmin=0, vmax=1, interpolation="bilinear",
    )
    _contact_overlay(
        ax, np.asarray(arrays["contact_xy_mm"], float),
        np.asarray(arrays["contact_names"]),
    )
    ax.set(
        xlabel="sheet x (mm)", ylabel="sheet y (mm)",
        xlim=(0, 20), ylim=(0, 20),
    )
    ax.set_title("A  Frozen binary Node field", weight="bold", loc="left")
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    bar = plt.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    bar.set_label("Node field h", fontsize=8)
    bar.ax.tick_params(labelsize=7)


def _patient_band(training, names):
    ranks = normalize_event_ranks(training["patient_train_ranks"])
    labels = np.asarray(training["patient_train_old_labels"], int)
    blocks = np.asarray(training["patient_train_block_ids"], int)
    profiles = patient_profiles(training["patient_train_ranks"], labels)
    low = np.full((2, len(names)), np.nan)
    high = np.full_like(low, np.nan)
    for mode in (0, 1):
        block_profiles = []
        for block in np.unique(blocks[labels == mode]):
            selected = (labels == mode) & (blocks == block)
            if np.any(selected):
                values = ranks[selected]
                count = np.sum(np.isfinite(values), axis=0)
                block_profiles.append(np.divide(
                    np.nansum(values, axis=0), count,
                    out=np.full(values.shape[1], np.nan), where=count > 0,
                ))
        values = np.asarray(block_profiles, float)
        low[mode] = np.nanquantile(values, 0.05, axis=0)
        high[mode] = np.nanquantile(values, 0.95, axis=0)
    return profiles, low, high


def _endpoint_panel(ax, rows):
    distance = np.asarray([
        row["selection"]["complete_distribution_distance_training"]
        for row in rows
    ], float)
    floors = np.asarray([
        [row["training_patient_floor"]["q05"],
         row["training_patient_floor"]["q95"]]
        for row in rows
    ], float)
    alignment = np.asarray([
        row["validation"]["direction_balanced_alignment"] for row in rows
    ], float)
    ood = np.asarray([
        row["validation"]["ood_all_returned"] for row in rows
    ], float)
    insets = [
        ax.inset_axes([0.00, 0.08, 0.29, 0.82]),
        ax.inset_axes([0.355, 0.08, 0.29, 0.82]),
        ax.inset_axes([0.71, 0.08, 0.29, 0.82]),
    ]
    rng = np.random.default_rng(20260902)
    x = rng.normal(0.0, 0.025, len(rows))
    insets[0].fill_between(
        [-0.18, 0.18], np.min(floors[:, 0]), np.max(floors[:, 1]),
        color="#BDBDBD", alpha=0.35, linewidth=0,
    )
    for axis, values, title, ylim in zip(
            insets, (distance, alignment, ood),
            ("Complete distribution", "KMeans alignment", "OOD"),
            ((0, max(1.45, 1.08 * np.max(distance))), (0, 1), (0, 1))):
        axis.scatter(x, values, s=25, facecolor="white", edgecolor="#303030", zorder=3)
        axis.errorbar(
            0, np.mean(values),
            yerr=[[np.mean(values) - np.min(values)],
                  [np.max(values) - np.mean(values)]],
            fmt="o", color="#111111", capsize=3, ms=4.5, zorder=4,
        )
        axis.set(xlim=(-0.20, 0.20), ylim=ylim, title=title)
        axis.set_xticks([])
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7)
        axis.title.set_fontsize(8.5)
    insets[0].set_ylabel("distance", fontsize=8)
    insets[1].set_ylabel("balanced fraction", fontsize=8)
    insets[2].set_ylabel("fraction", fontsize=8)
    ax.set_axis_off()
    ax.set_title("B  Separated baseline endpoints", loc="left", weight="bold")
    return {
        "complete_distribution_distance": distance.tolist(),
        "patient_floor_ranges": floors.tolist(),
        "kmeans_balanced_alignment": alignment.tolist(),
        "ood_all_returned": ood.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--candidate-id", default="dualcore_s39_reference")
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    summary_path = args.summary.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    summary = json.loads(summary_path.read_text())
    if summary.get("validation_endpoint_status") != "CANARY_BASELINE_DIAGNOSTIC":
        raise RuntimeError("canary validation diagnostic is unavailable")
    rows = [
        row for row in summary["per_network"]
        if row["candidate_id"] == args.candidate_id
    ]
    if not rows:
        raise RuntimeError("candidate missing from canary summary")
    training, contract, classifier = _load_contract(config, artifact_root)
    groups = contract_groups(contract)
    embedding = embedding_from_training_arrays(training)

    all_ranks, all_onsets, network_ids, all_ood = [], [], [], []
    first_arrays = None
    worker_hashes = {}
    worker_root = artifact_root / config["output_root"] / "canary" / "workers"
    for row in rows:
        path = worker_root / f"{args.candidate_id}_seed_{row['seed']}.npz"
        arrays = _npz(path)
        first_arrays = arrays if first_arrays is None else first_arrays
        returned = np.asarray(arrays["event_returned"], bool)
        onsets = np.asarray(arrays["onsets"], float)[returned]
        ranks = np.asarray(arrays["ranks"], float)[returned]
        assigned = assign_direction_modes(
            onsets, groups=groups, embedding=embedding, classifier=classifier,
        )
        readable = np.sum(np.isfinite(onsets), axis=1) >= 3
        all_ranks.append(ranks[readable])
        all_onsets.append(onsets[readable])
        network_ids.extend([int(row["seed"])] * int(np.sum(readable)))
        all_ood.extend(np.asarray(assigned["ood"], bool)[readable].tolist())
        worker_hashes[str(row["seed"])] = _sha256(path)
    ranks = np.concatenate(all_ranks)
    onsets = np.concatenate(all_onsets)
    network_ids = np.asarray(network_ids, int)
    all_ood = np.asarray(all_ood, bool)
    direction = assign_direction_modes(
        onsets, groups=groups, embedding=embedding, classifier=classifier,
    )
    natural = natural_kmeans(
        ranks, np.asarray(direction["labels"], int),
        random_state=int(config["validation"]["natural_kmeans_seed"]),
    )
    if natural["status"] != "OK":
        raise RuntimeError("pooled canary does not support natural KMeans")
    valid = np.asarray(natural["valid_event_mask"], bool)
    ranks = ranks[valid]
    network_ids = network_ids[valid]
    all_ood = all_ood[valid]
    alignment = best_binary_alignment(
        np.asarray(natural["cluster_labels"], int),
        np.asarray(direction["labels"], int)[valid],
    )
    labels = np.asarray(alignment["mapped_labels"], int)
    normalized = normalize_event_ranks(ranks)
    order = np.lexsort((network_ids, labels))
    patient_profile, patient_low, patient_high = _patient_band(
        training, training["contact_names"],
    )
    model_profile = np.asarray([
        np.nanmean(normalized[labels == mode], axis=0) for mode in (0, 1)
    ])
    matrix = _prototype_matrix(
        ranks, labels, training["patient_train_ranks"],
        training["patient_train_old_labels"],
    )

    fig = plt.figure(figsize=(15.8, 7.7), facecolor="white")
    grid = fig.add_gridspec(
        2, 12, height_ratios=(0.90, 1.10), left=0.05, right=0.975,
        bottom=0.09, top=0.94, hspace=0.38, wspace=0.65,
    )
    ax_field = fig.add_subplot(grid[0, 0:4])
    ax_endpoints = fig.add_subplot(grid[0, 4:12])
    ax_heat = fig.add_subplot(grid[1, 0:7])
    ax_profile = fig.add_subplot(grid[1, 7:10])
    ax_matrix = fig.add_subplot(grid[1, 10:12])
    _field_panel(ax_field, first_arrays)
    endpoint_metadata = _endpoint_panel(ax_endpoints, rows)

    shown = np.ma.masked_invalid(normalized[order].T)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#D4D4D4")
    image = ax_heat.imshow(
        shown, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap=cmap,
    )
    split = int(np.sum(labels == 0))
    ax_heat.axvline(split - 0.5, color="white", lw=3.0)
    ood_positions = np.flatnonzero(all_ood[order])
    ax_heat.scatter(
        ood_positions, np.full(len(ood_positions), -0.85), marker="v", s=9,
        color="#111111", clip_on=False,
    )
    names = np.asarray(training["contact_names"]).astype(str)
    ax_heat.set_yticks(np.arange(len(names)), names, fontsize=7)
    ax_heat.set(
        xlabel=f"{len(labels)} families across {len(rows)} networks",
        ylabel="virtual contact",
    )
    ax_heat.set_title(
        "C  Natural KMeans on all readable causal families",
        weight="bold", loc="left",
    )
    ax_heat.text(
        max(0, split / 2), -1.35, f"cluster 1  {split / len(labels):.0%}",
        ha="center", color=MODE_COLORS[0], fontsize=8,
    )
    ax_heat.text(
        split + max(0, (len(labels) - split) / 2), -1.35,
        f"cluster 2  {(len(labels) - split) / len(labels):.0%}",
        ha="center", color=MODE_COLORS[1], fontsize=8,
    )
    cax = ax_heat.inset_axes([1.012, 0, 0.018, 1])
    bar = fig.colorbar(image, cax=cax)
    bar.set_ticks((0, 1), labels=("first", "last"))
    bar.ax.tick_params(labelsize=7)

    y = np.arange(len(names))
    for mode in (0, 1):
        ax_profile.fill_betweenx(
            y, patient_low[mode], patient_high[mode],
            color=MODE_COLORS[mode], alpha=0.12, lw=0,
        )
        ax_profile.plot(
            model_profile[mode], y, "-o", color=MODE_COLORS[mode], lw=1.6,
            ms=2.8, label=f"model {mode + 1}",
        )
        ax_profile.plot(
            patient_profile[mode], y, "--", color=MODE_COLORS[mode], lw=1.2,
            label=f"patient {mode + 1}",
        )
    ax_profile.set(
        xlabel="mean normalized rank", xlim=(-0.05, 1.05),
        ylim=(len(y) - 0.5, -0.5),
    )
    ax_profile.set_title("D  Cluster rank profile", weight="bold", loc="left")
    ax_profile.set_yticks(y, [])
    ax_profile.legend(frameon=False, fontsize=7, ncol=2, loc="upper center")
    ax_profile.spines[["top", "right"]].set_visible(False)

    matrix_image = ax_matrix.imshow(
        matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal",
    )
    for row in (0, 1):
        for column in (0, 1):
            value = matrix[row, column]
            ax_matrix.text(
                column, row, f"{value:+.2f}", ha="center", va="center",
                color="white" if abs(value) > 0.55 else "#222222", weight="bold",
            )
    ax_matrix.set_xticks((0, 1), ("patient 1", "patient 2"), fontsize=7)
    ax_matrix.set_yticks((0, 1), ("model 1", "model 2"), fontsize=7)
    ax_matrix.set_title("E  Model vs patient", weight="bold", loc="left")
    matrix_cax = ax_matrix.inset_axes([1.08, 0, 0.06, 1])
    fig.colorbar(matrix_image, cax=matrix_cax)
    matrix_cax.tick_params(labelsize=7)

    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "dualcore_s39_canary_fig4_audit"
    fig.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    metadata = {
        "schema_id": "topic4_rev20_dc_canary_figure_v1",
        "candidate_id": args.candidate_id,
        "network_seeds": [int(row["seed"]) for row in rows],
        "worker_npz_sha256": worker_hashes,
        "n_kmeans_families": int(len(labels)),
        "cluster_counts": np.bincount(labels, minlength=2).tolist(),
        "direction_balanced_alignment": alignment["balanced_alignment"],
        "direction_purity": alignment["purity"],
        "kmeans_seed_ami_median": natural["kmeans_seed_ami_median"],
        "silhouette": natural["silhouette"],
        "prototype_spearman_matrix": matrix.tolist(),
        "endpoints": endpoint_metadata,
        "config_sha256": _sha256(config_path),
        "summary_sha256": _sha256(summary_path),
        "claim_boundary": config["claim_boundary"],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "README.md").write_text("""### dualcore_s39_canary_fig4_audit.png

冻结的二值双-core Node 场与三张新网络的因果事件族验收。A 显示实际二值场和虚拟触点；B 将无标签完整事件分布距离、自然 KMeans 与冻结方向的一致性、以及所有 returned families 的 OOD 分开；C-E 展示自然 KMeans 热图、患者训练 prototype 和完整相关矩阵。灰带是同样事件数的患者自采样地板，热图上方黑三角标出 OOD 事件。

**关注点**：双簇稳定不能替代完整分布恢复；先看 B 左侧模型距离是否进入患者地板，再看两个方向是否同时对齐、OOD 是否通过压低事件数换来。
""")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
