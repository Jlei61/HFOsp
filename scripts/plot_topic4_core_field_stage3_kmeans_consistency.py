"""Plot Fig. 4C-style KMeans-to-patient consistency for Stage 3 candidates.

Plotting only.  Patient held-out events and every model candidate were clustered
independently; rows are aligned afterward to frozen patient-training modes by a
label-invariant Hungarian match performed by the confirmation producer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/topic4_sef_hfo/data_driven_core_field_stage3")
CONFIRMATION = ROOT / "joint_confirmation_pilot_rev6.json"
PROFILES = ROOT / "joint_confirmation/joint_confirmation_event_profiles_rev6.npz"
OUT = ROOT / "joint_kmeans_consistency/figures"
MODE_COLORS = ("#C73E35", "#2F80A7")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_indices(labels, max_events=240):
    """Cluster-grouped deterministic display sample; statistics use all rows."""
    labels = np.asarray(labels, int)
    if labels.ndim != 1 or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be a one-dimensional two-mode vector")
    if len(labels) <= int(max_events):
        return np.concatenate([np.flatnonzero(labels == mode) for mode in (0, 1)])
    counts = np.bincount(labels, minlength=2)
    targets = np.maximum(1, np.rint(int(max_events) * counts / counts.sum()).astype(int))
    targets[np.argmax(targets)] += int(max_events) - int(targets.sum())
    selected = []
    for mode in (0, 1):
        available = np.flatnonzero(labels == mode)
        n_take = min(int(targets[mode]), len(available))
        local = np.floor(
            (np.arange(n_take) + 0.5) * len(available) / n_take
        ).astype(int)
        selected.append(available[local])
    return np.concatenate(selected)


def _prototype(curves, labels, mode):
    selected = np.asarray(curves, float)[np.asarray(labels) == int(mode)]
    if not len(selected):
        return np.full(np.asarray(curves).shape[1], np.nan), np.full(
            np.asarray(curves).shape[1], np.nan)
    return selected.mean(axis=0), selected.std(axis=0)


def _row_title(label, metric, verdict):
    counts = metric["cluster_counts"]
    text = (
        f"{label}  |  n={sum(counts)} ({counts[0]}/{counts[1]})  |  "
        f"matched mean={metric['matched_mean']:.3f}  |  "
        f"contrast={metric['matrix_contrast']:.3f}"
    )
    return text if verdict is None else f"{text}  |  {verdict}"


def render(confirmation_path=CONFIRMATION, profiles_path=PROFILES, out_dir=OUT):
    confirmation_path = Path(confirmation_path)
    profiles_path = Path(profiles_path)
    out_dir = Path(out_dir)
    payload = json.loads(confirmation_path.read_text(encoding="utf-8"))
    expected = payload.get("event_profiles", {}).get("sha256")
    observed = _sha256(profiles_path)
    if expected != observed:
        raise RuntimeError("confirmation JSON and event-profile NPZ hashes differ")

    profiles = np.load(profiles_path)
    grid = np.asarray(profiles["grid"], float)
    patient_train = np.asarray(profiles["patient_train_prototypes"], float)
    rows = [dict(
        label="patient held-out benchmark",
        curves=np.asarray(profiles["patient_heldout_curves"], float),
        labels=np.asarray(profiles["patient_heldout_labels"], int),
        metric=payload["kmeans_controls"]["patient_heldout"],
        verdict=None,
    )]
    for index, candidate in enumerate(payload["candidates"]):
        role = ", ".join(candidate["roles"])
        rows.append(dict(
            label=role,
            curves=np.asarray(profiles[f"candidate_{index}_curves"], float),
            labels=np.asarray(profiles[f"candidate_{index}_labels"], int),
            metric=candidate["confirm"]["kmeans_data_consistency"],
            verdict=candidate["confirm"]["verdict"],
        ))

    all_curves = np.vstack([row["curves"] for row in rows])
    vmax = float(np.quantile(np.abs(all_curves), 0.995))
    fig = plt.figure(figsize=(15.2, 11.0), constrained_layout=True)
    outer = fig.add_gridspec(
        len(rows), 3, width_ratios=(2.25, 1.30, 0.82),
        hspace=0.13, wspace=0.10,
    )
    heat_image = matrix_image = None
    for row_index, row in enumerate(rows):
        curves = row["curves"]
        labels = row["labels"]
        metric = row["metric"]
        if len(curves) != len(labels):
            raise RuntimeError(f"{row['label']}: curve/label length mismatch")
        shown = display_indices(labels)
        shown_labels = labels[shown]

        heat = fig.add_subplot(outer[row_index, 0])
        heat_image = heat.imshow(
            curves[shown].T, origin="lower", aspect="auto",
            extent=(-0.5, len(shown) - 0.5, grid[0], grid[-1]),
            cmap="viridis", vmin=-vmax, vmax=vmax, interpolation="nearest",
        )
        boundary = int(np.sum(shown_labels == 0)) - 0.5
        heat.axvline(boundary, color="#D62728", lw=1.8)
        heat.set_ylabel("axis position (mm)")
        heat.set_xlabel(
            "events grouped by independent KMeans"
            + (f" (display {len(shown)}/{len(curves)})" if len(shown) < len(curves) else "")
        )
        heat.set_title(_row_title(row["label"], metric, row["verdict"]),
                       loc="left", fontsize=10.5, weight="bold")
        heat.text(
            max(boundary / 2, 0), grid[-1] + 0.04 * np.ptp(grid), "mode A",
            color=MODE_COLORS[0], ha="center", va="bottom", weight="bold",
            clip_on=False,
        )
        heat.text(
            boundary + max((len(shown) - boundary) / 2, 0),
            grid[-1] + 0.04 * np.ptp(grid), "mode B",
            color=MODE_COLORS[1], ha="center", va="bottom", weight="bold",
            clip_on=False,
        )

        profile = fig.add_subplot(outer[row_index, 1], sharey=heat)
        for mode, color in enumerate(MODE_COLORS):
            mean, std = _prototype(curves, labels, mode)
            profile.fill_betweenx(
                grid, mean - std, mean + std, color=color, alpha=0.14, lw=0)
            profile.plot(mean, grid, color=color, lw=2.0)
            profile.plot(
                patient_train[mode], grid, color=color, lw=1.6,
                ls=(0, (3, 2)),
            )
        profile.axvline(0.0, color="#777777", lw=0.8, ls=":")
        profile.set_xlabel("normalized rank profile")
        profile.tick_params(axis="y", labelleft=False)
        if row_index == 0:
            profile.set_title("solid: row mode   dashed: patient train",
                              fontsize=9.5)

        matrix_ax = fig.add_subplot(outer[row_index, 2])
        matrix = np.asarray(metric["similarity_matrix"], float)
        matrix_image = matrix_ax.imshow(
            matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
        matrix_ax.set_xticks((0, 1), ("data A", "data B"))
        matrix_ax.set_yticks((0, 1), ("mode A", "mode B"))
        matrix_ax.set_title("KMeans mode vs data", fontsize=9.5)
        for i in range(2):
            for j in range(2):
                value = matrix[i, j]
                matrix_ax.text(
                    j, i, f"{value:+.2f}", ha="center", va="center",
                    color="white" if abs(value) > 0.55 else "black",
                    fontsize=11, weight="bold",
                )
        passed_pattern = bool(
            metric.get("support_eligible", False)
            and metric.get("matrix_sign_consistent", False))
        border = "#2A8C68" if passed_pattern else "#C73E35"
        for spine in matrix_ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(2.2)

    fig.suptitle(
        "Stage 3 | independent KMeans modes matched to patient data",
        fontsize=16, weight="bold",
    )
    rigid = float(payload["kmeans_rigid_benchmark_matched_mean"])
    fig.supxlabel(
        "Confirmation-only read-back. Green matrix border = supported Fig. 4C sign pattern; "
        f"rigid-control matched-mean benchmark = {rigid:.3f}. "
        "No grid-point permutation p-values (interpolated positions are autocorrelated).",
        fontsize=9.5,
    )
    if heat_image is not None:
        cbar = fig.colorbar(heat_image, ax=fig.axes[0::3], location="left",
                            shrink=0.42, pad=0.025)
        cbar.set_label("normalized event rank curve")
    if matrix_image is not None:
        cbar = fig.colorbar(matrix_image, ax=fig.axes[2::3], location="right",
                            shrink=0.42, pad=0.025)
        cbar.set_label("Spearman rho")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "stage3_joint_kmeans_data_consistency"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure": stem.name,
        "confirmation": str(confirmation_path),
        "confirmation_sha256": _sha256(confirmation_path),
        "event_profiles": str(profiles_path),
        "event_profiles_sha256": observed,
        "plotting_only": True,
        "kmeans_contract": (
            "patient train, patient held-out, controls, and candidates independently fit K=2 "
            "in the frozen embedding; candidate modes are Hungarian-matched afterward"
        ),
        "display_sampling": "all model events; deterministic cluster-stratified cap of 240 patient held-out events",
        "acceptance": {
            "minimum_events_per_mode": payload["opposition_min_cluster_events"],
            "matrix_sign_pattern": "matched cells > 0 and crossed cells < 0",
            "rigid_control_matched_mean_benchmark": rigid,
        },
        "rows": [
            {
                "label": row["label"],
                "n_events": int(len(row["curves"])),
                "metric": row["metric"],
                "verdict": row["verdict"],
            }
            for row in rows
        ],
        "statistical_boundary": (
            "The 31 curve positions are interpolated and autocorrelated; matrix entries are effect "
            "sizes, not 31 independent observations, so no channel-shuffle p-value is shown."
        ),
        "script_sha256": _sha256(Path(__file__)),
    }
    metadata_path = stem.with_name(stem.name + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    readme = out_dir / "README.md"
    readme.write_text(
        "### stage3_joint_kmeans_data_consistency.png\n\n"
        "这张图按 Fig. 4C 的逻辑，把病人留出段和三个未见网络候选分别独立做 KMeans=2，"
        "再与病人训练段的两个冻结模板做标签置换不变的一一匹配。左侧是按聚类排列的事件曲线，"
        "中间是模型实线与病人训练模板虚线，右侧是模型簇对病人簇的 2x2 Spearman 矩阵。"
        "绿色矩阵边框只表示双簇支持和正对角/负交叉结构通过，最终还必须超过 rigid benchmark 与距离门。\n\n"
        "**关注点**：先看每行两簇事件数是否足够，再看右侧矩阵是否呈正对角、负交叉；不要只看两个模型原型彼此负相关。\n",
        encoding="utf-8",
    )
    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", default=str(CONFIRMATION))
    parser.add_argument("--profiles", default=str(PROFILES))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    stem = render(args.confirmation, args.profiles, args.out)
    print(f"wrote {stem}.png / .pdf")


if __name__ == "__main__":
    main()
