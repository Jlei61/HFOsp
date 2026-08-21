#!/usr/bin/env python3
"""Plot the rev12-ND zero-simulation historical Node rescore."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path("/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/") / (
    "data_driven_node_dualmode_rev12/historical_rescore/historical_rescore.json"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "figures"
COLORS = {"baseline": "#B72C39", "initialization": "#2070A0", "other": "#A6A6A6"}


def _role(row: dict, *, anchor_hash: str, best_distinct_id: str | None) -> str:
    if row["field_sha256"] == anchor_hash:
        return "baseline"
    if row["field_id"] == best_distinct_id:
        return "initialization"
    return "other"


def _finite_or_nan(value) -> float:
    return float("nan") if value is None else float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    rows = [
        row for row in payload["field_rows"]
        if row["substrate_stratum"] == "current_spatial_ou_node_only"
    ]
    anchor_hash = payload["anchor_field_sha256"]
    best_distinct_id = payload["best_distinct_field_id"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.0,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7,
        "ytick.major.width": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)

    for role in ("other", "baseline", "initialization"):
        selected = [
            row for row in rows
            if _role(row, anchor_hash=anchor_hash, best_distinct_id=best_distinct_id) == role
        ]
        axes[0].scatter(
            [row["mean_patient_objective"] for row in selected],
            [row["model_prototype_r2_on_heldout"] for row in selected],
            s=13 if role == "other" else 32, color=COLORS[role],
            alpha=0.55 if role == "other" else 1.0, edgecolor="white",
            linewidth=0.35, zorder=2 if role == "other" else 4,
            label={
                "other": "historical fields", "baseline": "frozen Node",
                "initialization": "best distinct field",
            }[role],
        )
    axes[0].axhline(0.0, color="#555555", lw=0.7, ls="--")
    axes[0].set(xlabel="Complete-event objective", ylabel=r"Held-out event-cloud $R^2$")
    axes[0].legend(frameon=False, fontsize=6.2, loc="lower left")

    complete = np.asarray([
        _finite_or_nan(row["estimands"]["complete_returned"]["model_prototype_r2_on_heldout"])
        for row in rows
    ])
    clean = np.asarray([
        _finite_or_nan(row["estimands"]["formal_clean"]["model_prototype_r2_on_heldout"])
        for row in rows
    ])
    finite = np.isfinite(complete) & np.isfinite(clean)
    lower = float(min(np.min(complete[finite]), np.min(clean[finite]))) - 0.03
    upper = float(max(np.max(complete[finite]), np.max(clean[finite]), 0.0)) + 0.03
    axes[1].plot([lower, upper], [lower, upper], color="#666666", lw=0.7, ls="--")
    for row, x, y in zip(np.asarray(rows, object)[finite], complete[finite], clean[finite]):
        role = _role(row, anchor_hash=anchor_hash, best_distinct_id=best_distinct_id)
        axes[1].scatter(
            x, y, s=12 if role == "other" else 32, color=COLORS[role],
            alpha=0.5 if role == "other" else 1.0, edgecolor="white",
            linewidth=0.35, zorder=2 if role == "other" else 4,
        )
    axes[1].set(
        xlim=(lower, upper), ylim=(lower, upper),
        xlabel=r"Complete returned $R^2$", ylabel=r"Formal-clean $R^2$",
    )

    mode0 = np.asarray([row["mean_mode_0_loss"] for row in rows])
    mode1 = np.asarray([row["mean_mode_1_loss"] for row in rows])
    minimum, maximum = float(min(mode0.min(), mode1.min())), float(max(mode0.max(), mode1.max()))
    axes[2].plot([minimum, maximum], [minimum, maximum], color="#666666", lw=0.7, ls="--")
    for row, x, y in zip(rows, mode0, mode1):
        role = _role(row, anchor_hash=anchor_hash, best_distinct_id=best_distinct_id)
        axes[2].scatter(
            x, y, s=12 if role == "other" else 32, color=COLORS[role],
            alpha=0.5 if role == "other" else 1.0, edgecolor="white",
            linewidth=0.35, zorder=2 if role == "other" else 4,
        )
    axes[2].set(xlabel="Mode 1 loss", ylabel="Mode 2 loss")

    for label, axis in zip("ABC", axes):
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.18, 1.05, label, transform=axis.transAxes,
                  fontsize=10, fontweight="bold", va="top")

    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "node_historical_rescore"
    figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    (args.output / "README.md").write_text(
        "### node_historical_rescore.png\n\n"
        "A 比较与当前 spatial-OU 合同相容的唯一连续 Node 场在完整 returned-event 目标与患者 held-out "
        "event-cloud R2 上的位置；相同 field hash 已跨运行库合并，网络 seed 只计一次。红色为当前冻结 Node，蓝色为最佳不同场。"
        "B 将完整事件口径与旧 Fig.4 formal-clean 口径直接对照，显示删除 OOD 和单杆事件会系统性抬高 R2。"
        "C 分开显示两个患者模式的损失，检查平均分是否掩盖较弱模式。\n\n"
        "**关注点**：所有历史候选的完整 held-out R2 仍为负；蓝色候选只能作为下一轮初始化，不能视为已恢复患者事件分布。\n"
    )
    print(json.dumps({"status": "OK", "png": str(stem.with_suffix('.png'))}, indent=2))


if __name__ == "__main__":
    main()
