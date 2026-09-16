#!/usr/bin/env python3
"""Render all geometry-supported shared-axis patients for Fig2E visual review.

This is a review atlas, not the canonical four-example Figure 2E.  It consumes
the same frozen all-event Timing+Space records and the same field painter, and
never refits an axis, plane, field, support, rank or display kernel.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig3f_ab_dominance_cohort import (  # noqa: E402
    _pretty as manuscript_id,
)
from scripts.plot_topic5_interictal_template_ab_fields import (  # noqa: E402
    DEFAULT_DISPLAY_SIGMA_MM,
    TA_COLOR,
    TB_COLOR,
    build_interictal_ab_panel_payloads,
    draw_interictal_rank_field_panel,
)


INPUT_ROOT = (
    ROOT
    / "results/interictal_propagation_masked"
    / "template_gradient_fields_all_events_timing_plus_space"
)
OUTPUT_DIR = INPUT_ROOT / "figures/fig2e_review_all_shared_axis"
EXPECTED_N = 18
SUBJECTS_PER_SHEET = 6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _load_metric_rows(input_root: Path) -> dict[str, dict]:
    path = input_root / "shared_field_similarity_subjects.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {str(row["subject_id"]): row for row in csv.DictReader(handle)}
    if len(rows) != EXPECTED_N:
        raise RuntimeError(f"expected {EXPECTED_N} cohort rows, found {len(rows)}")
    return rows


def load_review_rows(input_root: Path) -> tuple[list[dict], Path]:
    metric_path = input_root / "shared_field_similarity_subjects.csv"
    metrics = _load_metric_rows(input_root)
    rows: list[dict] = []
    for path in sorted((input_root / "per_subject").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        subject_id = str(record.get("subject_id"))
        pair = record.get("axis_pair") or {}
        field = record.get("interictal_field") or {}
        models = field.get("field_models") or {}
        if (
            record.get("status") != "ok"
            or not pair.get("geometry_2d_supported")
            or "shared_a" not in models
            or "shared_b" not in models
        ):
            continue
        if subject_id not in metrics:
            raise RuntimeError(f"{subject_id}: absent from frozen cohort CSV")
        dat_a, dat_b, mode = build_interictal_ab_panel_payloads(
            record, display_sigma_mm=DEFAULT_DISPLAY_SIGMA_MM,
        )
        if mode != "shared":
            raise RuntimeError(f"{subject_id}: shared fields exist but renderer mode={mode}")
        metric = metrics[subject_id]
        rows.append(
            {
                "record": record,
                "dat_a": dat_a,
                "dat_b": dat_b,
                "display_id": manuscript_id(subject_id),
                "subject_id": subject_id,
                "n_contacts": int(len(field["contact_order"])),
                "r": float(metric["observed_shared_field_r"]),
                "channel_p_negative": float(metric["channel_p_negative"]),
                "channel_q_bh": float(metric["channel_q_bh"]),
                "strict_stability_pass": bool(pair.get("strict_stability_pass")),
                "x_span_mm": float(dat_a["frame"]["xlim"][1] - dat_a["frame"]["xlim"][0]),
                "y_span_mm": float(dat_a["frame"]["ylim"][1] - dat_a["frame"]["ylim"][0]),
            }
        )
    if len(rows) != EXPECTED_N or {row["subject_id"] for row in rows} != set(metrics):
        raise RuntimeError(
            "shared-axis review cohort does not exactly match the frozen 18-subject CSV"
        )
    return sorted(rows, key=lambda row: (float(row["r"]), str(row["display_id"]))), metric_path


def _style_field_axis(
    ax: plt.Axes,
    *,
    template: str,
    show_y: bool,
    show_x: bool,
    contact_size: float,
) -> None:
    color = TA_COLOR if template == "TA" else TB_COLOR
    ax.text(
        0.035, 0.965, template, transform=ax.transAxes,
        ha="left", va="top", fontsize=11.5, fontweight="bold", color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.7},
        zorder=9,
    )
    ax.tick_params(axis="both", labelsize=8.5, length=2.4, width=0.75, pad=1.8)
    if show_y:
        ax.set_ylabel("Y (mm)", fontsize=10.0, labelpad=2.0)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    if show_x:
        ax.set_xlabel("Shared TA axis (mm)", fontsize=10.0, labelpad=3.0)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)


def render_atlas(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    *,
    stem: str,
    subject_columns: int,
    overview: bool,
) -> tuple[Path, Path]:
    n_groups = int(np.ceil(len(rows) / subject_columns))
    fig_height = (4.85 if overview else 4.45) * n_groups + (
        0.25 if overview else 0.0
    )
    fig = plt.figure(figsize=(18.0, fig_height), facecolor="white")
    grid = fig.add_gridspec(
        2 * n_groups,
        subject_columns,
        left=0.055,
        right=0.965,
        bottom=0.075 if n_groups == 1 else 0.045,
        top=0.925,
        wspace=0.16,
        hspace=0.34 if overview else 0.22,
    )
    field_axes: list[plt.Axes] = []
    for index, row in enumerate(rows):
        group = index // subject_columns
        column = index % subject_columns
        row_a = 2 * group
        row_b = row_a + 1
        ax_a = fig.add_subplot(grid[row_a, column])
        ax_b = fig.add_subplot(grid[row_b, column], sharex=ax_a, sharey=ax_a)
        draw_interictal_rank_field_panel(
            ax_a,
            row["dat_a"],
            "TA",
            compact=True,
            panel_title="",
            contact_outline_lw=0.95 if overview else 1.15,
            contact_size=42 if overview else 55,
            show_template_tag=False,
        )
        draw_interictal_rank_field_panel(
            ax_b,
            row["dat_b"],
            "TB",
            compact=True,
            panel_title="",
            contact_outline_lw=0.95 if overview else 1.15,
            contact_size=42 if overview else 55,
            show_template_tag=False,
        )
        ax_a.set_title(
            f"{row['display_id']}   n={row['n_contacts']}   r={row['r']:+.2f}",
            fontsize=11.5,
            fontweight="bold",
            color="#222222",
            pad=3.5,
        )
        _style_field_axis(
            ax_a, template="TA", show_y=column == 0, show_x=False,
            contact_size=42 if overview else 55,
        )
        _style_field_axis(
            ax_b, template="TB", show_y=column == 0, show_x=True,
            contact_size=42 if overview else 55,
        )
        field_axes.extend((ax_a, ax_b))

    total_slots = n_groups * subject_columns
    for index in range(len(rows), total_slots):
        group = index // subject_columns
        column = index % subject_columns
        for grid_row in (2 * group, 2 * group + 1):
            axis = fig.add_subplot(grid[grid_row, column])
            axis.axis("off")

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0.0, 1.0), cmap="viridis"),
        ax=field_axes,
        orientation="vertical",
        fraction=0.012,
        pad=0.012,
        shrink=0.72 if overview else 0.88,
        aspect=32,
    )
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_ticklabels(["0  early", "0.5", "1  late"])
    colorbar.ax.tick_params(labelsize=9.5, length=3.0)
    colorbar.ax.set_title("Normalized\nranks", fontsize=10.0, pad=5.0, loc="left")
    if overview:
        fig.text(
            0.055,
            0.975,
            f"All geometry-supported shared-axis patients (n={len(rows)}); ordered by TA–TB field r",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#222222",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def build(input_root: Path, output_dir: Path) -> dict:
    rows, metric_path = load_review_rows(input_root)
    outputs: dict[str, list[str]] = {}
    overview = render_atlas(
        rows,
        output_dir,
        stem="fig2e-all-shared-axis-overview",
        subject_columns=SUBJECTS_PER_SHEET,
        overview=True,
    )
    outputs["overview"] = [_portable(path) for path in overview]
    for sheet_index, start in enumerate(range(0, len(rows), SUBJECTS_PER_SHEET), 1):
        sheet_rows = rows[start : start + SUBJECTS_PER_SHEET]
        paths = render_atlas(
            sheet_rows,
            output_dir,
            stem=f"fig2e-all-shared-axis-sheet{sheet_index}",
            subject_columns=SUBJECTS_PER_SHEET,
            overview=False,
        )
        outputs[f"sheet{sheet_index}"] = [_portable(path) for path in paths]

    metadata = {
        "schema_version": "fig2e_all_shared_axis_visual_review_v1",
        "status": "visual-review-only; canonical four-example Fig2E is unchanged",
        "input_root": _portable(input_root),
        "cohort_csv": _portable(metric_path),
        "cohort_csv_sha256": _sha256(metric_path),
        "cohort_contract": "shared_field_available and geometry_2d_supported",
        "n_subjects": len(rows),
        "ordering": "ascending observed TA-TB shared-field correlation",
        "display_contract": (
            "TA/TB share one frozen plane and frame within each subject; frames are adaptive "
            "across subjects so no contact is cropped; 6-mm display kernel; normalized ranks"
        ),
        "canvas_identity_contract": "manuscript E/Y labels only; raw IDs remain metadata provenance",
        "subjects": [
            {
                key: row[key]
                for key in (
                    "display_id", "subject_id", "n_contacts", "r",
                    "channel_p_negative", "channel_q_bh", "strict_stability_pass",
                    "x_span_mm", "y_span_mm",
                )
            }
            for row in rows
        ],
        "outputs": outputs,
    }
    metadata_path = output_dir / "fig2e-all-shared-axis-review_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    readme = """# Fig2E 全部 shared-axis 患者视觉审阅

### fig2e-all-shared-axis-overview.png / .pdf

纳入最新 all-event Timing+Space 冻结场中全部 18 名同时具备 shared axis 和受支持二维几何的患者。每名患者上行为 TA、下行为 TB，标题给出脱敏编号、触点数与 contact-evaluated TA–TB field correlation；按 r 从最负到最正排列。

**关注点**：这是挑选 Fig2E 形态案例的审阅总览，不替代 Fig2F 的 cohort 推断。TA/TB 在患者内共用冻结平面与坐标框；患者间自适应显示范围以保留全部触点，因此不要用 panel 面积比较实际空间范围。

### fig2e-all-shared-axis-sheet1.png / .pdf

总览排序中的第 1–6 名放大版。

**关注点**：优先检查是否存在多个连续触点、是否跨多根电极杆，以及 TA/TB 反向是否由整体场而非单个离群触点驱动。

### fig2e-all-shared-axis-sheet2.png / .pdf

总览排序中的第 7–12 名放大版。

**关注点**：同上；画面只负责形态审阅，显著性与精确统计保留在 metadata。

### fig2e-all-shared-axis-sheet3.png / .pdf

总览排序中的第 13–18 名放大版，包含弱反向和正相关案例。

**关注点**：这些案例能帮助识别“看似反向但主要由少数触点驱动”以及不适合放入 Fig2E 的形态。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    metadata = build(args.input_root.resolve(), args.output_dir.resolve())
    print(json.dumps(metadata["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
