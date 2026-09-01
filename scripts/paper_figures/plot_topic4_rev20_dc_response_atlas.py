#!/usr/bin/env python3
"""Render the frozen dual-core one-factor mechanism response atlas."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUPS = (
    ("Node substrate", (
        ("node_gain", "Node gain", "#C43C39"),
        ("core_budget_scale", "Core size", "#277DA1"),
        ("signed_depth_shrinkage", "Depth heterogeneity", "#3A923A"),
    )),
    ("Learned pathways", (
        ("g_EE", "E→E", "#C43C39"),
        ("g_EtoI", "E→I", "#277DA1"),
        ("both_scale", "Joint", "#222222"),
    )),
    ("EE orientation", (
        ("ellipse_angle_deg", "Long-axis angle", "#B66D0D"),
    )),
    ("EE anisotropy", (
        ("ellipse_aspect_ratio", "Long/short ratio", "#6F4E9C"),
    )),
)
METRICS = (
    ("heldout_complete_distribution", "Held-out complete distribution", "distance", (0, None)),
    ("kmeans_balanced_alignment", "Two-template concordance", "balanced alignment", (0, 1)),
    ("ood_all_returned", "Out-of-distribution events", "fraction", (0, 1)),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coordinate(levels, reference, grouped):
    values = np.asarray(levels, float)
    if not grouped:
        return values
    deviation = values - float(reference)
    scale = max(float(np.max(np.abs(deviation))), 1e-12)
    return deviation / scale


def _metric(row, phase, metric):
    values = row.get(phase)
    if values is None:
        return None
    return values["metrics"].get(metric)


def _floor_range(payload):
    values = []
    for candidate in payload["candidates"].values():
        floor = candidate["screen"].get("heldout_floor")
        if floor is None:
            continue
        low, high = floor["q05"], floor["q95"]
        if low is not None and high is not None:
            values.append((low["mean"], high["mean"]))
    if not values:
        return None
    return float(min(row[0] for row in values)), float(max(row[1] for row in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary_path = args.summary.resolve()
    payload = json.loads(summary_path.read_text())
    if payload.get("status") != "REV20_DC_RESPONSE_ATLAS_COMPLETE":
        raise RuntimeError("response atlas summary is incomplete")

    figure, axes = plt.subplots(
        3, 4, figsize=(16.2, 8.7), facecolor="white", sharey="row",
        gridspec_kw={"left": 0.06, "right": 0.985, "bottom": 0.10,
                     "top": 0.92, "hspace": 0.33, "wspace": 0.23},
    )
    patient_floor = _floor_range(payload)
    figure_metadata = {"panels": {}, "patient_floor_range": patient_floor}
    for column, (group_title, families) in enumerate(GROUPS):
        grouped = len(families) > 1
        for row_index, (metric, row_title, ylabel, limits) in enumerate(METRICS):
            axis = axes[row_index, column]
            if row_index == 0 and patient_floor is not None:
                axis.axhspan(
                    patient_floor[0], patient_floor[1], color="#BDBDBD",
                    alpha=0.25, linewidth=0, zorder=0,
                )
            panel_rows = []
            for family, label, color in families:
                curve = payload["family_curves"][family]
                rows = curve["rows"]
                levels = [float(item["level"]) for item in rows]
                x = _coordinate(levels, curve["reference_level"], grouped)
                screen = [_metric(item, "screen", metric) for item in rows]
                valid = np.asarray([item is not None for item in screen], bool)
                if np.any(valid):
                    mean = np.asarray([
                        np.nan if item is None else item["mean"] for item in screen
                    ], float)
                    low = np.asarray([
                        np.nan if item is None else item["bootstrap_q05"]
                        for item in screen
                    ], float)
                    high = np.asarray([
                        np.nan if item is None else item["bootstrap_q95"]
                        for item in screen
                    ], float)
                    axis.plot(
                        x[valid], mean[valid], color=color, lw=1.25,
                        marker="o", ms=4.0, markerfacecolor="white",
                        markeredgewidth=1.0, label=label, zorder=2,
                    )
                    axis.fill_between(
                        x[valid], low[valid], high[valid], color=color,
                        alpha=0.10, linewidth=0, zorder=1,
                    )
                confirmations = []
                for item, position in zip(rows, x):
                    confirm = _metric(item, "confirmation", metric)
                    if confirm is None:
                        continue
                    axis.errorbar(
                        position, confirm["mean"],
                        yerr=[[confirm["mean"] - confirm["bootstrap_q05"]],
                              [confirm["bootstrap_q95"] - confirm["mean"]]],
                        fmt="D", color=color, markeredgecolor="white",
                        markeredgewidth=0.65, ms=5.6, capsize=2.5,
                        lw=1.0, zorder=4,
                    )
                    confirmations.append({
                        "candidate_id": item["candidate_id"],
                        "level": item["level"],
                        "mean": confirm["mean"],
                        "q05": confirm["bootstrap_q05"],
                        "q95": confirm["bootstrap_q95"],
                    })
                panel_rows.append({
                    "family": family, "levels": levels,
                    "normalized_x": x.tolist(),
                    "selected_candidate_id": curve["selected_candidate_id"],
                    "confirmation": confirmations,
                })
            if row_index == 0:
                axis.set_title(group_title, weight="bold", fontsize=11)
                axis.legend(frameon=False, fontsize=7.4, loc="best")
            if column == 0:
                axis.set_ylabel(f"{row_title}\n{ylabel}", fontsize=9)
            if row_index == len(METRICS) - 1:
                if grouped:
                    axis.set_xlabel("normalized change from reference", fontsize=8.5)
                    axis.set_xticks((-1, 0, 1), ("lower", "reference", "higher"))
                elif families[0][0] == "ellipse_angle_deg":
                    axis.set_xlabel("long-axis angle (degrees)", fontsize=8.5)
                else:
                    axis.set_xlabel("long/short ratio", fontsize=8.5)
            if grouped:
                axis.axvline(0, color="#888888", lw=0.8, ls="--", zorder=0)
                axis.set_xlim(-1.12, 1.12)
            if limits[1] is not None:
                axis.set_ylim(*limits)
            else:
                current = axis.get_ylim()
                axis.set_ylim(0, max(current[1], 1.05))
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=8)
            figure_metadata["panels"][f"r{row_index}_c{column}"] = panel_rows

    axes[0, 0].text(
        0.02, 0.03, "gray: patient matched floor", transform=axes[0, 0].transAxes,
        fontsize=7, color="#666666", va="bottom",
    )
    figure.text(
        0.06, 0.965,
        "Frozen dual-core mechanism atlas",
        fontsize=14, weight="bold", ha="left", va="top",
    )
    figure.text(
        0.985, 0.965,
        "open circles: 4-network screen   diamonds: 12-network confirmation",
        fontsize=8, ha="right", va="top", color="#444444",
    )

    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "dualcore_mechanism_response_atlas"
    figure.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    metadata = {
        "schema_id": "topic4_rev20_dc_response_atlas_figure_v1",
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "figure": figure_metadata,
        "selection_boundary": payload["selection_boundary"],
        "claim_boundary": payload["claim_boundary"],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "README.md").write_text("""### dualcore_mechanism_response_atlas.png

冻结二值双-core Node 场后逐次只改变一个模型坐标。每列分别汇总 Node 底物、learned E→E/E→I/Joint 通路表达、EE 长轴方向和长短轴比；三行依次是 held-out 完整事件分布距离、自然 KMeans 与冻结患者方向的 balanced alignment、以及以全部 returned causal families 为分母的 OOD。空心圆和浅带是 4-network screen；菱形是仅对训练分布选中的水平做的 12-network confirmation；灰带是患者同样事件数的自采样地板。

**关注点**：先找能在不压低事件产率的前提下把完整分布距离拉向灰带的参数，再看同一变化是否保留双模板一致性并降低 OOD；只改善其中一项属于 trade-off。
""")
    print(json.dumps({
        "status": "REV20_DC_RESPONSE_ATLAS_FIGURE_COMPLETE",
        "png": str(stem.with_suffix(".png")),
        "pdf": str(stem.with_suffix(".pdf")),
    }, indent=2))


if __name__ == "__main__":
    main()
