"""Stage-C visual QA for smoke training, decoder and graph contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "SMOKE_M0_NO_REC": "#9d9da1",
    "SMOKE_M3_FIXED_LOCAL": "#4c78a8",
    "SMOKE_M6_SPATIAL_MID": "#e45756",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    units = []
    for path in sorted((out_root / "per_subject").glob("*/SMOKE_*__rnn/seed0/metrics.json")):
        metrics = json.loads(path.read_text())
        history = json.loads((path.parent / "history.json").read_text())
        decoder = json.loads((path.parent / "rollout_decoder_history.json").read_text())
        units.append((path, metrics, history, decoder))
    if len(units) != 9:
        raise RuntimeError(f"expected 9 smoke units, found {len(units)}")

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.7,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    for _, metrics, history, _ in units:
        model_id = metrics["model_id"].split("__", 1)[0]
        x = [row["epoch"] for row in history]
        y = [row["val"] for row in history]
        axes[0].plot(x, y, color=COLORS[model_id], alpha=0.50, linewidth=1.0)
    axes[0].axvline(50, color="#555555", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Training epoch")
    axes[0].set_ylabel("Validation loss")
    axes[0].set_title("Rewire → freeze", loc="left", fontweight="bold")

    fits = sorted({metrics["fit_id"] for _, metrics, _, _ in units})
    width = 0.23
    for offset, model_id in enumerate(COLORS):
        values = []
        for fit_id in fits:
            match = next(metrics for _, metrics, _, _ in units
                         if metrics["fit_id"] == fit_id and metrics["model_id"].startswith(model_id))
            values.append(match["test"]["contact_nll"])
        axes[1].bar(np.arange(3) + (offset - 1) * width, values, width,
                    color=COLORS[model_id], label=model_id.split("_", 2)[-1])
    axes[1].set_xticks(range(3), ["small", "medium", "large"])
    axes[1].set_ylabel("Held-out contact NLL")
    axes[1].set_title("Three montage scales", loc="left", fontweight="bold")

    representative = max(units, key=lambda item: item[1]["n_nodes"])
    base = representative[0].parents[2]
    fit_id = representative[1]["fit_id"]
    for model_id, color in (("SMOKE_M3_FIXED_LOCAL", "#4c78a8"),
                            ("SMOKE_M6_SPATIAL_MID", "#e45756")):
        graph_path = out_root / "per_subject" / fit_id / f"{model_id}__rnn" / "seed0" / "graph.npz"
        graph = np.load(graph_path)
        mask = graph["mask"].astype(bool)
        strength = graph["strength"][mask]
        distance = graph["D_mm"][mask]
        axes[2].scatter(distance, strength, s=6, alpha=0.35, color=color,
                        linewidths=0, label=model_id.split("_", 2)[-1])
    axes[2].set_xlabel("Edge length (mm)")
    axes[2].set_ylabel("Effective weight")
    axes[2].set_title("Same edge budget", loc="left", fontweight="bold")
    axes[2].legend(loc="upper right", frameon=False, handletextpad=0.4)

    for label, axis in zip("abc", axes):
        axis.text(-0.17, 1.04, label, transform=axis.transAxes, fontsize=12,
                  fontweight="bold", va="bottom")
    figure_dir = out_root / "figures"
    figure_dir.mkdir(exist_ok=True)
    stem = figure_dir / "stage_c_smoke_training_and_decoder"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    audit = {
        "status": "ALIGNED_ENGINEERING_SMOKE",
        "n_units": 9,
        "purpose": "shape, optimization path, rollout decoder, memory and graph-contract validation",
        "scientific_inference_allowed": False,
        "target_values_read": False,
    }
    (out_root / "stage_c_scientific_drift_audit.json").write_text(json.dumps(audit, indent=2))
    readme = figure_dir / "README.md"
    with readme.open("a") as handle:
        handle.write(
            "\n### stage_c_smoke_training_and_decoder.png\n\n"
            "阶段 C 工程 smoke。a 检查 rewiring 到固定图后的优化轨迹；b 检查三种 montage "
            "规模都能完成 held-out scoring；c 检查固定局部图与空间重连图在相同 edge budget 下"
            "确实形成不同的距离—权重分布。\n\n"
            "**关注点**：本图只决定正式训练能否安全启动，不承载科学阳性或阴性。\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
