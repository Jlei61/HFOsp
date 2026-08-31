#!/usr/bin/env python3
"""Render the static Node-only dual-core OOD and KMeans confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_d6_natural_kmeans import (  # noqa: E402
    best_binary_alignment,
    natural_kmeans,
    normalize_event_ranks,
)


DEFAULT_CONFIG = ROOT / "config/topic4_dual_core_ood_node_pathways.json"
MODE_COLORS = ("#c43c39", "#277da1")
CORE_COLOR = "#756bb1"


def _even_sample(indices: np.ndarray, maximum: int) -> np.ndarray:
    indices = np.asarray(indices, int)
    if len(indices) <= maximum:
        return indices
    local = np.linspace(0, len(indices) - 1, maximum).round().astype(int)
    return indices[local]


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.1,
        "ytick.labelsize": 6.1,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _patient_profiles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        ranks = normalize_event_ranks(np.asarray(loaded["patient_train_ranks"], float))
        labels = np.asarray(loaded["patient_train_old_labels"], int)
        profiles = np.asarray([
            np.nanmean(ranks[labels == mode], axis=0) for mode in (0, 1)
        ])
        return np.asarray(loaded["contact_names"]).astype(str), profiles


def render(config_path: Path, output_dir: Path) -> dict:
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"] / "confirmation"
    aggregate_path = root / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text())
    if aggregate.get("status") != "DUAL_CORE_OOD_PHASE_COMPLETE":
        raise RuntimeError("confirmation aggregate is incomplete")
    candidate = aggregate["ranking"][0]
    target_path = ROOT / config["inputs"]["shaft_aware_target_npz"]["path"]
    contact_names, patient_profiles = _patient_profiles(target_path)
    pooled_ranks, pooled_labels = [], []
    positions = h = contact_xy = None
    for row in candidate["per_network"]:
        with np.load(ROOT / row["worker_npz"], allow_pickle=False) as loaded:
            ranks = np.asarray(loaded["ranks"], float)
            if positions is None:
                positions = np.asarray(loaded["positions_E"], float)
                h = np.asarray(loaded["h"], float)
                contact_xy = np.asarray(loaded["contact_xy_mm"], float)
                if not np.array_equal(
                    np.asarray(loaded["contact_names"]).astype(str), contact_names,
                ):
                    raise RuntimeError("contact order changed")
        support = np.asarray([event["in_support"] for event in row["events"]], bool)
        labels = np.asarray([event["mode"] for event in row["events"]], int)
        pooled_ranks.append(ranks[support])
        pooled_labels.append(labels[support])
    pooled_ranks = np.concatenate(pooled_ranks)
    pooled_labels = np.concatenate(pooled_labels)
    natural = natural_kmeans(pooled_ranks, pooled_labels, random_state=20260830)
    if natural.get("status") != "OK":
        raise RuntimeError("pooled natural KMeans is not evaluable")
    valid = natural["valid_event_mask"]
    ranks = pooled_ranks[valid]
    labels = pooled_labels[valid]
    alignment = best_binary_alignment(natural["cluster_labels"], labels)
    mapped = alignment["mapped_labels"]
    sampled = np.concatenate([
        _even_sample(np.flatnonzero(mapped == mode), 160) for mode in (0, 1)
    ])
    split = len(_even_sample(np.flatnonzero(mapped == 0), 160))
    normalized = normalize_event_ranks(ranks) * 14.0
    model_profiles = np.asarray([
        np.nanmean(normalize_event_ranks(ranks[mapped == mode]), axis=0)
        for mode in (0, 1)
    ])
    _style()
    fig = plt.figure(figsize=(7.2, 4.5))
    layout = fig.add_gridspec(
        2, 3, width_ratios=(1.0, 1.55, 1.05), height_ratios=(1.0, 1.0),
        hspace=0.48, wspace=0.52,
    )
    field_axis = fig.add_subplot(layout[:, 0])
    heat_axis = fig.add_subplot(layout[0, 1:])
    profile_axis = fig.add_subplot(layout[1, 1])
    ood_axis = fig.add_subplot(layout[1, 2])
    inactive = h < 0.5
    field_axis.scatter(
        positions[inactive, 0], positions[inactive, 1], s=0.15,
        color="#d9d9d9", alpha=0.28, rasterized=True,
    )
    centers = np.asarray(candidate["node_field"]["centers_mm"], float)
    distance = np.linalg.norm(
        positions[:, None, :] - centers[None, :, :], axis=2,
    )
    membership = np.argmin(distance, axis=1)
    for core in (0, 1):
        selected_core = (h >= 0.5) & (membership == core)
        field_axis.scatter(
            positions[selected_core, 0], positions[selected_core, 1], s=0.35,
            color=CORE_COLOR, alpha=0.58, rasterized=True,
        )
    field_axis.scatter(
        contact_xy[:, 0], contact_xy[:, 1], s=17, facecolor="white",
        edgecolor="#202020", linewidth=0.45, zorder=3,
    )
    field_axis.scatter(
        centers[:, 0], centers[:, 1], s=32, marker="x", color="#111111",
        linewidth=1.0, zorder=4,
    )
    for core, center in enumerate(centers, start=1):
        field_axis.text(
            center[0], center[1] + 0.8, f"C{core}", ha="center", va="bottom",
            fontsize=6.0, color="#333333", fontweight="bold",
        )
    field_axis.set(
        xlim=(0, 20), ylim=(0, 20), xlabel="sheet x (mm)", ylabel="sheet y (mm)",
    )
    field_axis.set_aspect("equal")
    field_axis.set_title("Fitted two-core Node field", loc="left", fontweight="bold")

    image = heat_axis.imshow(
        normalized[sampled].T, aspect="auto", interpolation="nearest",
        cmap="viridis", vmin=0, vmax=14,
    )
    heat_axis.axvline(split - 0.5, color="white", lw=0.8)
    heat_axis.set_yticks(np.arange(len(contact_names)))
    heat_axis.set_yticklabels(contact_names)
    heat_axis.set_xlabel("formal-clean natural KMeans events  |  Mode 1      Mode 2")
    heat_axis.set_ylabel("contact")
    heat_axis.set_title(
        f"Natural K=2 within patient support  |  balanced match {alignment['balanced_alignment']:.2f}",
        loc="left", fontweight="bold",
    )
    colorbar = fig.colorbar(image, ax=heat_axis, pad=0.012, fraction=0.025)
    colorbar.set_label("rank", labelpad=1)

    y = np.arange(len(contact_names))
    for mode in (0, 1):
        profile_axis.plot(
            model_profiles[mode] * 14.0, y, color=MODE_COLORS[mode],
            lw=1.35, marker="o", ms=2.0, label=f"model Mode {mode + 1}",
        )
        profile_axis.plot(
            patient_profiles[mode] * 14.0, y, color=MODE_COLORS[mode],
            lw=1.0, ls="--", label=f"patient Mode {mode + 1}",
        )
    profile_axis.invert_yaxis()
    profile_axis.set_yticks(y)
    profile_axis.set_yticklabels(contact_names, fontsize=5.5)
    profile_axis.set_xlim(-0.4, 14.4)
    profile_axis.set_xlabel("mean rank (0 = first)")
    profile_axis.set_title("Cluster rank profiles", loc="left", fontweight="bold")
    profile_axis.legend(frameon=False, fontsize=5.3, ncol=2, loc="upper left")

    ood = np.asarray([
        row["ood_all_returned"] * 100.0 for row in candidate["per_network"]
    ])
    rng = np.random.default_rng(20260830)
    sampled_means = np.mean(
        ood[rng.integers(0, len(ood), size=(4096, len(ood)))], axis=1,
    )
    mean = float(np.mean(ood))
    low, high = np.quantile(sampled_means, (0.05, 0.95))
    ood_axis.scatter(
        0.03 * np.linspace(-1, 1, len(ood)), ood, s=13, facecolor="white",
        edgecolor="#555555", linewidth=0.45,
    )
    ood_axis.errorbar(
        0, mean, yerr=[[mean - low], [high - mean]], fmt="o", ms=4.5,
        color="#202020", capsize=2.5, elinewidth=1.1,
    )
    ood_axis.set_xlim(-0.2, 0.2)
    ood_axis.set_xticks([0], ["Node"])
    ood_axis.set_ylim(0, 105)
    ood_axis.set_ylabel("OOD among all returned events (%)")
    ood_axis.set_title("Independent-network OOD", loc="left", fontweight="bold")
    fig.suptitle(
        "Dual-core Node recovery against the frozen patient event distribution",
        y=0.99, fontsize=9.0, fontweight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.09, top=0.91)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "dual_core_node_ood_kmeans_confirmation"
    fig.savefig(
        stem.with_suffix(".png"), dpi=600, facecolor="white",
        bbox_inches="tight", pad_inches=0.025,
    )
    fig.savefig(
        stem.with_suffix(".pdf"), facecolor="white",
        bbox_inches="tight", pad_inches=0.025,
    )
    plt.close(fig)
    metadata = {
        "status": "DUAL_CORE_NODE_OOD_KMEANS_FIGURE_RENDERED",
        "candidate_id": candidate["candidate_id"],
        "node_field": candidate["node_field"],
        "n_confirmation_networks": candidate["n_networks"],
        "networks_with_both_modes": candidate["networks_with_both_modes"],
        "ood_percent": {
            "equal_network_mean": mean, "network_bootstrap_q05": float(low),
            "network_bootstrap_q95": float(high),
        },
        "pooled_natural_kmeans": {
            key: value for key, value in natural.items()
            if key not in {"valid_event_mask", "cluster_labels"}
        },
        "natural_kmeans_population": (
            "returned, readable, cross-shaft events inside frozen patient support"
        ),
        "pooled_alignment": {
            key: value for key, value in alignment.items()
            if key != "mapped_labels"
        },
        "aggregate": str(aggregate_path.relative_to(ROOT)),
        "outputs": [
            str(stem.with_suffix(".png").relative_to(ROOT)),
            str(stem.with_suffix(".pdf").relative_to(ROOT)),
        ],
    }
    (output_dir / "dual_core_node_ood_kmeans_confirmation_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=lambda value: value.tolist()) + "\n"
    )
    readme = output_dir / "README.md"
    existing = readme.read_text() if readme.exists() else ""
    entry = (
        "### dual_core_node_ood_kmeans_confirmation.png\n"
        "严格双 core Node 候选在 12 张独立网络上的静态验收。图中同时给出实际 core 位置、"
        "patient-support 内 formal-clean events 的 pooled natural KMeans、患者/模型 rank "
        "profile 和以网络为单位的 OOD 分布；两个 core 使用同一中性色，不代表两种模式。\n\n"
        "**关注点**：KMeans 是否自然形成两簇、两簇是否同时对齐患者原型，以及 OOD 是否由少数"
        "异常网络驱动。\n"
    )
    if "### dual_core_node_ood_kmeans_confirmation.png" not in existing:
        readme.write_text(existing + ("\n" if existing else "") + entry)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output_dir or (
        ROOT / config["output_root"] / "confirmation/figures"
    )
    metadata = render(args.config.resolve(), output.resolve())
    print(json.dumps({
        "status": metadata["status"], "outputs": metadata["outputs"],
    }, indent=2))


if __name__ == "__main__":
    main()
