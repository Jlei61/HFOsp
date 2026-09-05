#!/usr/bin/env python3
"""Render Figure 2E from all-event Timing+Space shared template fields."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig2_shared_field_reversal_row import (  # noqa: E402
    apply_common_display_window,
    _restore_compact_axis_ticks,
)
from scripts.paper_figures.plot_fig3f_ab_dominance_cohort import (  # noqa: E402
    _pretty as manuscript_id,
)
from scripts.paper_figures.paper_figure_source_registry import (  # noqa: E402
    registered_path,
)
from scripts.plot_topic5_interictal_template_ab_fields import (  # noqa: E402
    DEFAULT_DISPLAY_SIGMA_MM,
    TA_COLOR,
    TB_COLOR,
    build_interictal_ab_panel_payloads,
    draw_interictal_rank_field_panel,
)


DEFAULT_INPUT = registered_path("fig2", "e", "analysis_root")
DEFAULT_OUTPUT = registered_path("fig2", "e", "staging_root")
N_EXAMPLES = 4
MIN_CONTACTS = 7
MIN_SHAFTS = 2
MIN_CONTIGUOUS_CONTACTS = 5
DISPLAY_X_SPAN_MM = 70.0
DISPLAY_Y_SPAN_MM = 70.0
DISPLAY_ORDER = ("E1", "E12", "E5", "Y9")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _max_contiguous_contacts(contact_order: list[str], shafts_by_name: dict[str, str]) -> int:
    """Return the longest consecutive contact-number run on one shaft."""
    numbers_by_shaft: dict[str, set[int]] = {}
    for name in contact_order:
        match = re.search(r"(\d+)$", name)
        if match is None:
            continue
        numbers_by_shaft.setdefault(shafts_by_name[name], set()).add(int(match.group(1)))
    longest = 0
    for numbers in numbers_by_shaft.values():
        current = 0
        previous = None
        for number in sorted(numbers):
            current = current + 1 if previous is not None and number == previous + 1 else 1
            longest = max(longest, current)
            previous = number
    return longest


def _eligible_rows(input_root: Path) -> list[dict]:
    with (input_root / "shared_field_similarity_subjects.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        statistics = {row["subject_id"]: row for row in csv.DictReader(handle)}
    rows = []
    for path in sorted((input_root / "per_subject").glob("*.json")):
        record = json.loads(path.read_text())
        pair = record.get("axis_pair") or {}
        field = record.get("interictal_field") or {}
        models = field.get("field_models") or {}
        if (
            record.get("status") != "ok"
            or pair.get("geometry_2d_supported") is not True
            or not {"shared_a", "shared_b"}.issubset(models)
        ):
            continue
        ta = np.asarray(models["shared_a"]["template_field"], float)
        tb = np.asarray(models["shared_b"]["template_field"], float)
        if ta.shape != tb.shape or ta.shape != (len(field["contact_order"]),):
            raise ValueError(f"field/contact mismatch: {record['subject_id']}")
        payload_a, payload_b, mode = build_interictal_ab_panel_payloads(
            record, display_sigma_mm=DEFAULT_DISPLAY_SIGMA_MM
        )
        if mode != "shared":
            raise ValueError(f"shared field routed to {mode}: {record['subject_id']}")
        x = np.concatenate([np.asarray(payload_a["xs"]), np.asarray(payload_b["xs"])])
        y = np.concatenate([np.asarray(payload_a["ys"]), np.asarray(payload_b["ys"])])
        contact_indices = {
            name: index for index, name in enumerate(record["names"])
        }
        shafts_by_name = {
            name: record["shafts"][index]
            for name, index in contact_indices.items()
        }
        used_shafts = {
            shafts_by_name[name]
            for name in field["contact_order"]
        }
        subject_statistics = statistics[str(record["subject_id"])]
        rows.append({
            "subject_id": str(record["subject_id"]),
            "display_id": manuscript_id(str(record["subject_id"])),
            "record": record,
            "n_contacts": int(len(ta)),
            "r": float(np.corrcoef(ta, tb)[0, 1]),
            "x_span_mm": float(np.ptp(x)),
            "y_span_mm": float(np.ptp(y)),
            "n_shafts": int(len(used_shafts)),
            "max_contiguous_contacts": _max_contiguous_contacts(
                field["contact_order"], shafts_by_name
            ),
            "channel_p_negative": float(subject_statistics["channel_p_negative"]),
            "channel_q_bh": float(subject_statistics["channel_q_bh"]),
            "channel_fdr_significant_negative": (
                subject_statistics["channel_fdr_significant_negative"] == "True"
            ),
        })
    return rows


def _select(rows: list[dict]) -> list[dict]:
    candidates = {
        row["display_id"]: row for row in rows
        if row["n_contacts"] >= MIN_CONTACTS
        and row["n_shafts"] >= MIN_SHAFTS
        and row["max_contiguous_contacts"] >= MIN_CONTIGUOUS_CONTACTS
        and row["r"] < 0
        and row["channel_p_negative"] < 0.05
        and row["x_span_mm"] <= DISPLAY_X_SPAN_MM
        and row["y_span_mm"] <= DISPLAY_Y_SPAN_MM
    }
    missing = [display_id for display_id in DISPLAY_ORDER if display_id not in candidates]
    if missing:
        raise RuntimeError(f"missing display-eligible examples: {missing}")
    selected = [candidates[display_id] for display_id in DISPLAY_ORDER]
    return selected


def render(input_root: Path, output_root: Path) -> dict:
    rows = _eligible_rows(input_root)
    selected = _select(rows)
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(7.15, 2.60), facecolor="white")
        grid = fig.add_gridspec(
            2, N_EXAMPLES + 1,
            width_ratios=(*([1] * N_EXAMPLES), 0.055),
            left=0.075, right=0.945, top=0.91, bottom=0.18,
            wspace=0.06, hspace=0.22,
        )
        axes = []
        for column, row in enumerate(selected):
            payload_a, payload_b, mode = build_interictal_ab_panel_payloads(
                row["record"], display_sigma_mm=DEFAULT_DISPLAY_SIGMA_MM
            )
            if mode != "shared":
                raise RuntimeError(f"{row['subject_id']}: expected shared plane")
            apply_common_display_window(
                payload_a,
                payload_b,
                x_span_mm=DISPLAY_X_SPAN_MM,
                y_span_mm=DISPLAY_Y_SPAN_MM,
            )
            ax_a = fig.add_subplot(grid[0, column])
            ax_b = fig.add_subplot(grid[1, column], sharex=ax_a, sharey=ax_a)
            draw_interictal_rank_field_panel(
                ax_a, payload_a, "TA", compact=True,
                panel_title=row["display_id"], contact_outline_lw=0.78,
                contact_size=28, show_template_tag=False,
            )
            draw_interictal_rank_field_panel(
                ax_b, payload_b, "TB", compact=True,
                contact_outline_lw=0.78, contact_size=28,
                show_template_tag=False,
            )
            ax_a.set_title(row["display_id"], fontsize=8.5, fontweight="bold", pad=2.0)
            ax_b.set_title("")
            _restore_compact_axis_ticks(ax_a)
            _restore_compact_axis_ticks(ax_b)
            if column == 0:
                ax_a.set_ylabel("y (mm)", fontsize=6.4, labelpad=0.5)
                ax_b.set_ylabel("y (mm)", fontsize=6.4, labelpad=0.5)
                ax_a.text(
                    -0.47, 0.5, "TA field", transform=ax_a.transAxes,
                    ha="center", va="center", rotation=90,
                    fontsize=7.4, fontweight="bold", color=TA_COLOR,
                )
                ax_b.text(
                    -0.47, 0.5, "TB field", transform=ax_b.transAxes,
                    ha="center", va="center", rotation=90,
                    fontsize=7.4, fontweight="bold", color=TB_COLOR,
                )
            axes.extend([ax_a, ax_b])

        cax = fig.add_subplot(grid[:, N_EXAMPLES])
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis"),
            cax=cax,
        )
        colorbar.set_ticks([0, 1], labels=["early", "late"])
        colorbar.ax.tick_params(labelsize=6.1, length=2, pad=1.3)
        fig.canvas.draw()
        bottom = min(ax.get_position().y0 for ax in axes)
        top = max(ax.get_position().y1 for ax in axes)
        pos = cax.get_position()
        cax.set_position([pos.x0, bottom, pos.width, top - bottom])
        fig.text(
            0.5 * (pos.x0 + pos.x1), top + 0.018, "rank",
            ha="center", va="bottom", fontsize=6.5,
        )
        fig.text(
            0.5 * (min(ax.get_position().x0 for ax in axes) + max(ax.get_position().x1 for ax in axes)),
            0.035, "Shared TA axis (mm)", ha="center", va="bottom", fontsize=7.2,
        )

        png = figures / "fig2-panele.png"
        pdf = figures / "fig2-panele.pdf"
        fig.savefig(png, dpi=600, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)

    metadata = {
        "schema_version": "fig2e_all_event_timing_plus_space_v4",
        "input_root": _portable(input_root),
        "input_summary_sha256": _sha256(input_root / "cohort_summary.json"),
        "cohort_shared_2d_n": len(rows),
        "selection_rule": (
            "four fixed illustrative shared 2D fields with a negative TA-TB correlation "
            f"at channel-permutation p<0.05, >= {MIN_CONTACTS} contacts across >= "
            f"{MIN_SHAFTS} shafts, >= {MIN_CONTIGUOUS_CONTACTS} consecutive contacts "
            "on at least one shaft, and full support inside one "
            f"{DISPLAY_X_SPAN_MM:.0f}x{DISPLAY_Y_SPAN_MM:.0f} mm display window; "
            "E1 and Y9 pass channel-level BH-FDR, while E12 and E5 are nominally "
            "significant continuous multi-contact morphology examples; cohort inference "
            "remains in Fig. 2F"
        ),
        "examples": [
            {key: row[key] for key in (
                "subject_id", "display_id", "n_contacts", "n_shafts",
                "max_contiguous_contacts", "r",
                "channel_p_negative", "channel_q_bh", "x_span_mm", "y_span_mm"
            )}
            for row in selected
        ],
        "outputs": {"png": _portable(png), "pdf": _portable(pdf)},
    }
    (figures / "fig2-panele-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    (figures / "README.md").write_text(
        "# Figure 2 all-event Timing+Space panels\n\n"
        "### fig2-panele.png / .pdf\n\n"
        "四名患者的 TA/TB 均来自 all-event Timing+Space 聚类后冻结的 shared plane；"
        "上下图使用同一患者、同一坐标、同一 70×70 mm 显示窗。示例按预先写入 metadata 的"
        "多杆、连续多触点、负相关与可读性规则选择：E1、Y9 通过 channel-level BH-FDR，"
        "E12、E5 为名义显著的连续多触点形态示例；统计结论仍使用 Fig. 2F 的完整 18 人分母。\n\n"
        "**关注点**：逐列比较同一患者 TA 与 TB 的早晚场是否翻转。\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(render(args.input_root.resolve(), args.output_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
