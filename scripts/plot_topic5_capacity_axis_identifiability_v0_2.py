#!/usr/bin/env python3
"""Extended Data candidate: can this method find a direction at all?

Three questions the six-panel figure does not answer, kept out of it on purpose:

a  How much of the "patient-trained axis" is the shape of the implantation?
b  On data whose true axis is known, does the frozen estimator recover it — and
   does the rest of the machinery work when handed the true axis instead?
c  How far does partial electrode coverage move that answer?

Panel b was designed to separate "the method cannot detect an aligned structure"
from "the frozen axis estimator did not find the axis".  In this run it does not
separate them: neither arm beats chance (per cell of 24 montages, binomial p from
0.31 to 1.00), so the structure layer is uninformative rather than negative.  The
oracle arm exists only in simulation; real data never has a true axis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_topic5_capacity_supp_figure_v0_2 import visual_qa  # noqa: E402
from scripts.plot_topic5_capacity_supp_figure_v0_2 import COLOURS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"
FIG_ROOT = ROOT / "results/paper-ready-figure/supp_fig7_axis_identifiability_v0_2/figures"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.0, "axes.labelsize": 7.0,
    "axes.titlesize": 7.0, "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0, "axes.linewidth": 0.6, "savefig.dpi": 600,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def draw_axis_vs_geometry(ax, table: pd.DataFrame, summary: dict) -> None:
    frame = table[table["basis_fraction"] == 100]
    ax.scatter(frame["cloud_aspect_2d"], frame["gap_to_contact_cloud_axis_deg"], s=16,
               facecolor=COLOURS["patient"], edgecolor="white", linewidth=0.3, zorder=3,
               label="vs contact-cloud long axis")
    ax.scatter(frame["cloud_aspect_2d"], frame["gap_to_dominant_shaft_axis_deg"], s=16,
               facecolor=COLOURS["shaft"], edgecolor="white", linewidth=0.3, zorder=3,
               marker="s", label="vs dominant shaft axis")
    ax.axhline(20, color="#888888", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.text(ax.get_xlim()[1], 21, "20°", ha="right", va="bottom", fontsize=7.0, color="#888888")
    ax.set_xscale("log")
    ax.set_xlabel("how elongated the implantation is\n(long / short axis of the contact cloud)")
    ax.set_ylabel("angle between the trained axis\nand a purely geometric axis (°)")
    ax.set_ylim(-4, 100)
    ax.set_title(
        f"{summary['gap_to_contact_cloud_axis_deg']['n_within_20_deg']}/"
        f"{summary['gap_to_contact_cloud_axis_deg']['n']} within 20°"
        f", rank corr {summary['spearman_aspect_vs_gap_to_cloud']:+.2f}",
        fontsize=7.0, color="#666666", pad=3, loc="left")
    ax.legend(frameon=False, loc="upper center", handlelength=1.2, borderaxespad=0.1,
              bbox_to_anchor=(0.55, 1.02))


def draw_oracle_contrast(ax, cells: pd.DataFrame) -> None:
    """Same teacher, same data; the only difference is where the axis comes from."""
    frame = cells[cells["block"].astype(str).str.startswith("S1_power")]
    if not len(frame):
        frame = cells[cells["block"].astype(str).str.startswith("S0_")]
    frame = frame.dropna(subset=["U_FULL_SET_auto_structure_effect"])
    groups = []
    for effect, block in sorted({(e, b) for e, b in zip(frame["effect"], frame["block"])}):
        subset = frame[(frame["effect"] == effect) & (frame["block"] == block)]
        oracle = "oracle" in str(block)
        groups.append((effect, oracle, subset["U_FULL_SET_auto_structure_effect"].to_numpy()))
    effects = sorted({entry[0] for entry in groups})
    rng = np.random.default_rng(5)
    for oracle, colour, offset, label in ((False, COLOURS["unordered"], -0.13,
                                           "axis estimated"),
                                          (True, COLOURS["patient"], 0.13,
                                           "true axis given")):
        centres, medians = [], []
        for index, effect in enumerate(effects):
            values = np.concatenate([g[2] for g in groups
                                     if g[0] == effect and g[1] == oracle] or [np.array([])])
            if values.size == 0:
                continue
            x = index + offset
            ax.scatter(x + rng.uniform(-0.045, 0.045, values.size), values, s=9,
                       facecolor=colour, edgecolor="white", linewidth=0.2, alpha=0.85, zorder=3)
            boot = np.median(rng.choice(values, size=(4000, values.size), replace=True), axis=1)
            ax.plot([x, x], np.percentile(boot, [2.5, 97.5]), color="#222222", linewidth=1.3,
                    zorder=4)
            ax.plot([x - 0.09, x + 0.09], [np.median(values)] * 2, color="#222222",
                    linewidth=1.5, zorder=5)
            centres.append(x)
            medians.append(float(np.median(values)))
        if centres:
            ax.plot(centres, medians, color=colour, linewidth=1.0, alpha=0.6, zorder=2,
                    label=label)
    ax.axhline(0.0, color="#444444", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xticks(np.arange(len(effects)))
    ax.set_xticklabels([f"{value:g}" for value in effects])
    ax.set_xlabel("how strongly the events follow the axis")
    ax.set_ylabel("direction-rotated minus\npatient-trained error")
    ax.set_title("known answer", fontsize=7.0, color="#666666", pad=3, loc="left")
    ax.legend(frameon=False, loc="upper left", handlelength=1.0, borderaxespad=0.15,
              fontsize=7.0, labelspacing=0.25)
    ax.margins(y=0.34)


def draw_coverage(ax, cells: pd.DataFrame) -> None:
    frame = cells[cells["block"].astype(str).str.startswith("S2")].dropna(
        subset=["U_FULL_SET_auto_structure_effect", "observed_fraction"])
    if not len(frame):
        ax.text(0.5, 0.5, "no coverage cells yet", ha="center", va="center",
                transform=ax.transAxes, color="#888888")
        ax.axis("off")
        return
    kinds = sorted(set(frame["mask_kind"].astype(str)))
    palette = {"none": COLOURS["unordered"], "random": COLOURS["geometry"],
               "shaft_like": COLOURS["shaft"], "source_avoiding": COLOURS["rotated"]}
    for kind in kinds:
        subset = frame[frame["mask_kind"].astype(str) == kind]
        ax.scatter(subset["observed_fraction"], subset["U_FULL_SET_auto_structure_effect"],
                   s=14, facecolor=palette.get(kind, "#888888"), edgecolor="white",
                   linewidth=0.25, alpha=0.9, zorder=3,
                   label={"none": "all kept", "random": "random",
                          "shaft_like": "shaft-like",
                          "source_avoiding": "source-avoiding"}.get(kind, kind))
    ax.axhline(0.0, color="#444444", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xlabel("fraction of contacts observed")
    ax.set_ylabel("direction-rotated minus\npatient-trained error")
    ax.set_title(f"partial coverage, n={len(frame)}", fontsize=7.0, color="#666666",
                 pad=3, loc="left")
    ax.legend(frameon=False, loc="lower left", handlelength=1.0, ncol=1, fontsize=7.0,
              borderaxespad=0.15, labelspacing=0.25)
    ax.margins(y=0.30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(FIG_ROOT))
    arguments = parser.parse_args()
    out_root = Path(arguments.out)
    out_root.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(RESULT_ROOT / "PER_PATIENT_AXIS_VS_IMPLANTATION.csv")
    summary = json.loads((RESULT_ROOT / "AXIS_VS_IMPLANTATION_SUMMARY.json").read_text())
    cells_path = RESULT_ROOT / "synthetic" / "SYNTHETIC_CELLS.csv"
    cells = pd.read_csv(cells_path) if cells_path.exists() else pd.DataFrame()

    figure = plt.figure(figsize=(7.28, 3.10))
    axes = [figure.add_axes(box) for box in
            ([0.085, 0.230, 0.240, 0.630], [0.425, 0.230, 0.240, 0.630],
             [0.745, 0.230, 0.230, 0.630])]
    draw_axis_vs_geometry(axes[0], table, summary)
    if len(cells):
        draw_oracle_contrast(axes[1], cells)
        draw_coverage(axes[2], cells)
    for ax, label in zip(axes, "abc"):
        ax.text(-0.30, 1.18, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=10, fontweight="bold", color="#111111")
        ax.spines[["top", "right"]].set_visible(False)

    stem = out_root / "supp_fig7_topic5_axis_identifiability_v0_2"
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=600, facecolor="white")
    plt.close(figure)

    source = out_root / "source_data"
    source.mkdir(exist_ok=True)
    table.to_csv(source / "panela_axis_vs_implantation.csv", index=False)
    if len(cells):
        cells.to_csv(source / "panelbc_synthetic_cells.csv", index=False)
    (out_root / "README.md").write_text("\n".join([
        "### supp_fig7_topic5_axis_identifiability_v0_2.png / .pdf / .svg",
        "",
        "a 问：被称作「患者训练轴」的方向，有多少只是电极本身的形状？横轴是该患者电极云的长宽比，"
        "纵轴是训练轴与两条纯几何轴（电极云长轴、主要电极杆方向）的夹角。植入越狭长，训练轴越贴着"
        "电极云长轴。",
        "",
        "b 问：在答案已知的仿真上，这套方法能不能找到那条轴？同一份数据、同一套机器，只换方向的来源——"
        "灰色是按真实数据那条路自己估计出来的轴，深青是直接把真轴交给它（仿真才有）。"
        "**实测：两条线没有分开**——即使把真轴交给它、事件以最强强度沿该轴推进，"
        "赢过旋转对照的比例也只有 14/24（二项 p=0.54）。所以这一层的结论是"
        "「看不清」，不是「不存在方向」，也不能说成「机器能检出、只是估计器不行」。",
        "",
        "c 问：只观察到一部分触点时，上面的答案会移动多少？按遮挡方式分色。",
        "",
        "**关注点**：a 的 20° 参考线与秩相关；b 两条线在三个强度档上都压在零线附近、"
        "散点跨度远大于中位差（＝这套方法在本设计下对已知强轴向结构没有检出力）；"
        "c 的零线。三个面板都不进入 28 人主队列的统计面板。",
        "",
    ]))
    metadata = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_axis_identifiability_figure",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "asset_role": "Extended Data candidate; separate from the six-panel supplementary figure "
                      "because it answers questions none of those panels answers",
        "axis_summary": summary,
        "n_synthetic_cells": int(len(cells)),
        "assets_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(out_root.glob("supp_fig7_*"))},
        # same machine checks the six-panel figure gets, so a reader can tell the
        # three exports carry the same state and that nothing is clipped
        "VISUAL_QA": {
            **visual_qa(stem, {
                "a": int(len(table[table.basis_fraction == 100])),
                "b": int(cells.block.astype(str).str.startswith("S1_power").sum()),
                "c": int(cells.block.astype(str).str.startswith("S2_").sum())}),
            "panel_a_is_real_patients_panels_b_c_are_synthetic": True},
    }
    (out_root / "SUPP_FIG7_METADATA.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"stem": str(stem), "n_patients": int(len(table[table.basis_fraction == 100])),
                      "n_synthetic_cells": int(len(cells))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
