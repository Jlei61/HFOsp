#!/usr/bin/env python3
"""Plot the model-internal qualification boundary for the negative rev5 fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
BLUE = "#2C6CA3"
ORANGE = "#E97932"
INK = "#252525"


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_payload(records):
    points = []
    for row in records:
        qualification = row.get("model_ictal_qualification") or {}
        if qualification.get("joint_duty") is None:
            continue
        points.append({
            "candidate_id": row["candidate_id"],
            "full_learned_edges": bool(row.get("primary_zm_only")),
            "edge_expression_comparator": bool(row.get("edge_dose_comparator")),
            "joint_duty": float(qualification["joint_duty"]),
            "frequency_shift_hz": float(qualification["contact_centroid_shift_hz"]),
            "frequency_ratio": float(qualification["contact_centroid_ratio"]),
            "frequency_ratio_pass": bool(
                qualification["contact_centroid_ratio"] >= 1.25),
            "parameters": row.get("parameters") or {},
        })
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/topic4_data_driven_zm_target_informed_bridge_v1.json")
    args = parser.parse_args()
    config = _load(ROOT / args.config)
    result_root = ROOT / config["output_root"]
    rows = _load(result_root / "existing_candidate_rescore.json")["records"]
    points = build_payload(rows)
    primary = [row for row in points if row["full_learned_edges"]]
    comparators = [row for row in points if row["edge_expression_comparator"]]
    if not primary:
        raise RuntimeError("no full learned-edge Z/M candidates")

    out = ROOT / "results/paper-ready-figure/fig5_zm_qualification_boundary/figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.0,
        "axes.linewidth": 0.7, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "svg.fonttype": "none",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.65, 3.1), constrained_layout=True)
    all_x = [row["frequency_shift_hz"] for row in points]
    xmax = max(40.0, max(all_x) + 3.0)
    xmin = min(-6.0, min(all_x) - 2.0)
    ax.fill_between([5.0, xmax], [0.8, 0.8], [1.035, 1.035],
                    color="#E7EEE8", zorder=0)
    ax.axvline(5.0, color="#8A8A8A", lw=0.7, ls="--", zorder=1)
    ax.axhline(0.8, color="#8A8A8A", lw=0.7, ls="--", zorder=1)

    for group, color, marker, size in (
            (primary, BLUE, "o", 25), (comparators, ORANGE, "D", 29)):
        for row in group:
            ax.scatter(
                row["frequency_shift_hz"], row["joint_duty"], s=size,
                marker=marker,
                facecolor=color if row["frequency_ratio_pass"] else "white",
                edgecolor=color, linewidth=0.75, alpha=0.88, zorder=3)

    frequency_pass = [row for row in primary
                      if row["frequency_shift_hz"] >= 5.0
                      and row["frequency_ratio_pass"]]
    duty_pass = [row for row in primary if row["joint_duty"] >= 0.8]
    closest_frequency = max(frequency_pass, key=lambda row: row["joint_duty"])
    closest_duty = max(duty_pass, key=lambda row: row["frequency_shift_hz"])
    for row in (closest_frequency, closest_duty):
        ax.scatter(row["frequency_shift_hz"], row["joint_duty"], s=58,
                   facecolor="none", edgecolor=INK, linewidth=0.9, zorder=4)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.15, 1.035)
    ax.set_xlabel("Contact-frequency shift (Hz)")
    ax.set_ylabel("Broad-recruitment duty (1 s)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", color=BLUE,
               markerfacecolor=BLUE, markersize=4.5,
               label="Full learned EE/EI"),
        Line2D([], [], marker="D", ls="none", color=ORANGE,
               markerfacecolor=ORANGE, markersize=4.2,
               label="Reduced E-to-I expression"),
        Line2D([], [], marker="o", ls="none", color="#777777",
               markerfacecolor="white", markersize=4.5,
               label="Frequency-ratio < 1.25"),
    ], frameon=False, loc="lower right", fontsize=6.2)

    stem = out / "fig5-zm-qualification-boundary"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    metadata = {
        "status": "MODEL_INTERNAL_QUALIFICATION_BOUNDARY_RENDERED",
        "source": str((result_root / "existing_candidate_rescore.json").relative_to(ROOT)),
        "n_full_learned_edge_candidates": len(primary),
        "n_edge_expression_comparators": len(comparators),
        "qualification_region": {
            "joint_duty_min": 0.8,
            "contact_frequency_shift_min_hz": 5.0,
            "contact_frequency_ratio_min": 1.25,
        },
        "closest_frequency_passing_full_edge": closest_frequency,
        "closest_duty_passing_full_edge": closest_duty,
        "claim_boundary": (
            "model-internal development diagnostic; patient target is not an axis "
            "and no patient bridge score was assigned to failed full-edge candidates"),
    }
    (out / "fig5-zm-qualification-boundary-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "### fig5-zm-qualification-boundary.png\n"
        "比较完整保留 learned EE/EI 的 Z/M 候选与降低 E→I 表达的历史对照。"
        "绿色区域要求一秒内广泛招募 duty 不低于 0.8、触点频率质心至少增加 5 Hz；"
        "空心点另表示频率比低于 1.25。黑色外圈标出 full-edge 候选在两条单独边界上的最近点。\n\n"
        "**关注点**：full learned EE/EI 的 Z/M 候选没有进入联合合格区域；"
        "这发生在读取患者能量损失之前。\n",
        encoding="utf-8")
    print(json.dumps({"status": metadata["status"], "output": str(stem)}))


if __name__ == "__main__":
    main()
