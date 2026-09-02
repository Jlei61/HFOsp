#!/usr/bin/env python3
"""Plot the frozen rev21 Z/M timescale trade-off after aggregation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm


def _matrix(rows, x_levels, y_levels, getter):
    output = np.full((len(y_levels), len(x_levels)), np.nan)
    for row in rows:
        level = row["level"]
        y = y_levels.index(float(level["tau_z_ms"]))
        x = x_levels.index(float(level["tau_adp_ms"]))
        output[y, x] = getter(row)
    return output


def _annotate(ax, values, formatter, *, threshold=None, white_above=True):
    finite = values[np.isfinite(values)]
    midpoint = float(np.nanmedian(finite)) if len(finite) else 0.0
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if np.isfinite(value):
                dark = value > midpoint if threshold is None else (
                    value > threshold if white_above else value < threshold
                )
                ax.text(x, y, formatter(value), ha="center", va="center",
                        fontsize=7, color="white" if dark else "black")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.aggregate.read_text())
    rows = payload["candidate_summaries"]
    cells = payload["per_cell"]
    x_levels = sorted({float(row["level"]["tau_adp_ms"]) for row in rows})
    y_levels = sorted({float(row["level"]["tau_z_ms"]) for row in rows})

    eligible = _matrix(
        rows, x_levels, y_levels,
        lambda row: float(row["model_ictal_eligible_cells"]),
    )
    duty = _matrix(
        rows, x_levels, y_levels,
        lambda row: float(np.median([
            cell["model_ictal"]["recruitment"]["joint_duty"]
            for cell in cells if cell["candidate_id"] == row["candidate_id"]
        ])),
    )
    frequency = _matrix(
        rows, x_levels, y_levels,
        lambda row: float(np.median([
            cell["model_ictal"]["contact_frequency"]["primary_shift_hz"]
            for cell in cells if cell["candidate_id"] == row["candidate_id"]
        ])),
    )
    retained = _matrix(
        rows, x_levels, y_levels,
        lambda row: float(row["interictal_substrate_retained"]),
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.05), constrained_layout=True)
    panels = (
        (eligible, "Ictal qualification", "Blues", BoundaryNorm(
            np.arange(-0.5, 5.5, 1), plt.get_cmap("Blues").N), None),
        (duty, "Broad recruitment", "magma", None, 0.55),
        (frequency, "Frequency shift", "RdBu_r",
         TwoSlopeNorm(vmin=-5, vcenter=0, vmax=30), 8),
        (retained, "Interictal retention", "YlGn",
         BoundaryNorm([-0.5, 0.5, 1.5], plt.get_cmap("YlGn").N), 0.5),
    )
    letters = "ABCD"
    for index, (ax, panel) in enumerate(zip(axes, panels)):
        values, title, cmap, norm, text_threshold = panel
        image = ax.imshow(values, origin="lower", cmap=cmap, norm=norm,
                          aspect="equal")
        ax.set_title(title, pad=5, fontweight="bold")
        ax.text(-0.16, 1.06, letters[index], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom")
        ax.set_xticks(
            range(len(x_levels)), [f"{value/1000:g}" for value in x_levels],
        )
        ax.set_yticks(range(len(y_levels)), [f"{value/1000:g}" for value in y_levels])
        ax.set_xlabel(r"$\tau_m$ (s)")
        if index == 0:
            ax.set_ylabel(r"$\tau_z$ (s)")
        else:
            ax.tick_params(axis="y", labelleft=False)
        if index == 0:
            _annotate(ax, values, lambda value: f"{int(value)}/4", threshold=2.0)
        elif index == 1:
            _annotate(ax, values, lambda value: f"{value:.2f}",
                      threshold=0.40, white_above=False)
        elif index == 2:
            _annotate(ax, values, lambda value: f"{value:+.1f}",
                      threshold=text_threshold)
        else:
            for y, tau_z in enumerate(y_levels):
                for x, tau_m in enumerate(x_levels):
                    row = next(item for item in rows if
                               float(item["level"]["tau_z_ms"]) == tau_z and
                               float(item["level"]["tau_adp_ms"]) == tau_m)
                    matched = row["event_count_matched_retention"]
                    counts = matched["observed"]["frozen_direction_counts"]
                    label = f"{'yes' if matched['retained'] else 'no'}\n{counts[0]}:{counts[1]}"
                    ax.text(x, y, label, ha="center", va="center", fontsize=6.5,
                            color="white" if matched["retained"] else "black")
        if index in (1, 2):
            colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
            colorbar.ax.tick_params(labelsize=6.5, width=0.6, length=2)
            colorbar.set_label("fraction" if index == 1 else "Hz", fontsize=7)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / "rev21_zm_timescale_boundary"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "schema_id": "topic4_rev21_zm_timescale_boundary_figure_v1",
        "source": str(args.aggregate.resolve()),
        "source_sha256": hashlib.sha256(args.aggregate.read_bytes()).hexdigest(),
        "source_schema_id": payload["schema_id"],
        "status": payload["status"],
        "panel_d_annotation": "retained yes/no and frozen direction counts mode0:mode1",
        "patient_ictal_inputs_read": False,
    }
    (args.out_dir / "rev21_zm_timescale_boundary.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "README.md").write_text(
        "### rev21_zm_timescale_boundary.png\n\n"
        "这张图汇总冻结双-core 底物上的 Z/M 时间常数实验。A 显示每个点在 4 个 "
        "topology×dynamics 单元中有几个达到模型发作资格；B/C 分别显示广泛持续占比和触点频率变化；"
        "D 的 yes/no 表示间期底物是否保留，下一行是两个冻结方向的事件数。\n\n"
        "**关注点**：快速失抑制能抬高频率，但没有稳定形成广泛持续状态，并同时压缩发作前双方向事件；"
        "较慢时间尺度可保留间期结构，却没有达到模型发作资格。\n"
    )
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
