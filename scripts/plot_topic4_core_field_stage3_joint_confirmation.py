"""Plot the bounded unseen-network confirmation screen for Stage 3 rev6."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
INPUT = f"{ROOT}/joint_confirmation_pilot_rev6.json"
OUT = f"{ROOT}/joint_confirmation/figures/stage3_joint_confirmation_screen"


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def plot_confirmation(payload, stem):
    controls = payload["optimization_controls_n20"]
    candidates = payload["candidates"]
    labels = ["patient floor", "Stage 2 filament", "hand-placed cores",
              "training global best", "final-generation best", "CMA mean"]
    rows = [controls["patient_heldout"], controls["stage2_filament"],
            controls["hand_placed_two_cores"]]
    rows.extend(row["confirm"]["bootstrap_distance_patient_train"]
                for row in candidates)
    colors = ["#333333", "#3a9d78", "#2d7fa3", "#c4473a", "#d28b26", "#6f5a9e"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), constrained_layout=True)
    ax = axes[0]
    yy = np.arange(len(labels))[::-1]
    ax.axvspan(controls["patient_heldout"]["p05"],
               controls["patient_heldout"]["p95"], color="#333333", alpha=0.09,
               label="patient 5-95% floor")
    for y, label, row, color in zip(yy, labels, rows, colors):
        median = float(row["median"])
        ax.errorbar(median, y,
                    xerr=[[median - float(row["p05"])],
                          [float(row["p95"]) - median]],
                    fmt="o", color=color, capsize=3, ms=6, lw=1.4)
    for index, candidate in enumerate(candidates):
        heldout = candidate["confirm"]["bootstrap_distance_patient_heldout"]
        y = yy[3 + index] - 0.16
        median = float(heldout["median"])
        ax.errorbar(median, y,
                    xerr=[[median - float(heldout["p05"])],
                          [float(heldout["p95"]) - median]],
                    fmt="D", mfc="white", mec=colors[3 + index],
                    ecolor=colors[3 + index], capsize=3, ms=5, lw=1.1)
    ax.set_yticks(yy, labels)
    ax.set_xlabel("joint profile distance (lower is closer)")
    ax.set_title("A  Unseen-network distance", loc="left", weight="bold")
    ax.text(0.99, 0.98, "circles: patient-train  |  diamonds: patient held-out",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.35")
    ax.spines[["right", "top"]].set_visible(False)

    ax = axes[1]
    ax.fill_betweenx(
        [-1.0, -0.2], payload["opposition_min_cluster_events"], 40,
        color="#3a9d78", alpha=0.13)
    ax.axvline(payload["opposition_min_cluster_events"], color="0.35", ls="--", lw=1)
    ax.axhline(-0.2, color="0.35", ls="--", lw=1)
    for index, (candidate, color) in enumerate(zip(candidates, colors[3:])):
        diagnostic = candidate["confirm"]["posthoc_prototypes"]
        counts = diagnostic["cluster_counts"]
        x = diagnostic["min_cluster_count"]
        y = diagnostic["prototype_correlation"]
        ax.scatter(x, y, s=72, color=color, edgecolor="white", linewidth=0.8,
                   zorder=3)
        offset = (7, -28) if index == 0 else ((6, 8) if index == 1 else (7, -16))
        ax.annotate(f"{labels[3 + index]}\n{counts[0]} / {counts[1]}",
                    (x, y), xytext=offset, textcoords="offset points", fontsize=8,
                    color=color)
    ax.text(38.5, -0.94, "required quadrant", ha="right", va="bottom",
            fontsize=8, color="#287a59")
    ax.set_xlim(0, 40)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("events in the smaller post-hoc cluster")
    ax.set_ylabel("correlation between the two prototypes")
    ax.set_title("B  Opposition requires support", loc="left", weight="bold")
    ax.spines[["right", "top"]].set_visible(False)

    os.makedirs(os.path.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".png", dpi=220, facecolor="white")
    fig.savefig(stem + ".pdf", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    payload = json.load(open(args.input))
    if payload["status"] != "UNSEEN_NETWORK_CONFIRMATION_NO_CANDIDATE_PASSES":
        raise SystemExit("confirmation payload has not reached the frozen negative verdict")
    plot_confirmation(payload, args.out)
    metadata = dict(
        status=payload["status"], input=args.input, input_sha256=_sha256(args.input),
        git_commit=_git_commit(), objective_event_count=payload["objective_event_count"],
        plot_contract=("distance intervals are event bootstraps; opposition support uses "
                       "the frozen minimum of 10 events in each post-hoc cluster"),
    )
    with open(args.out + "_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    readme = """# Joint confirmation 图说明

### stage3_joint_confirmation_screen

这张图汇总三代 K=3 pilot 的未见网络确认。A 用固定 20 条事件比较患者地板、rigid controls 和三个预冻结候选；B 同时检查两个后验簇是否各有至少 10 条事件、且两原型相关不高于 -0.2。

**关注点**：距离较低的两个候选都没有相反原型；唯一负相关的 CMA mean 实际是 1 对 66 的 singleton 分组。当前没有候选同时满足距离和双簇结构，不能解释为恢复了 core。
"""
    with open(os.path.join(os.path.dirname(args.out), "README.md"), "w") as fh:
        fh.write(readme)
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
