#!/usr/bin/env python3
"""Render the three stable core-evidence figures for Group-Event State.

Current H1/H2a observations are read from the v0.3 pilot summary.  H2b/H3
panels use explicit empty interfaces until those analyses are run; the producer
never invents demonstration data.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_group_event_state_core_figures")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_epi_prssm.figure_style import (  # noqa: E402
    DOUBLE_COLUMN_MM,
    MM,
    apply_style,
)
from src.topic5_group_event_state.v03.core_evidence import (  # noqa: E402
    HORIZON_MINUTES,
    build_payload,
    load_payload,
    validate_payload,
)


STATE = "#A35E48"       # Morandi rust: candidate state / cohort median
HISTORY = "#6B7280"     # interpretable history / neutral comparator
SHIFT = "#2F5D8A"       # wrong-time state
SUPPORT = "#DDE9E5"     # favourable region
NEUTRAL = "#C7C7C7"
TEXT = "#2B2B2B"
SUBJECT_COLORS = ("#2F5D8A", "#8A5A2F", "#7E6E84", "#5B8A72", "#A35E48")
FIGURE_STEMS = {
    "h1": "group_event_state_h1_future_blocks",
    "h2a": "group_event_state_h2a_repertoire",
    "transfer_feedback": "group_event_state_h2b_h3_transfer_feedback",
}


def _style() -> None:
    apply_style()
    # The older shared helper uses 6.8 pt ticks; this package honours the
    # project-level >=7 pt readability floor requested for final figures.
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.4,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
        }
    )


def _panel(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.14,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="top",
        color=TEXT,
    )


def _finish_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, pad=2)


def _subject_palette(payload: dict[str, Any]) -> dict[str, str]:
    aliases = [row["alias"] for row in payload["training"]]
    return {
        alias: SUBJECT_COLORS[i % len(SUBJECT_COLORS)]
        for i, alias in enumerate(aliases)
    }


def _rows_by_subject(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["alias"]].append(row)
    return grouped


def _tick_labels(rows: list[dict[str, Any]], field: str) -> list[str]:
    labels = []
    for horizon in HORIZON_MINUTES:
        n = sum(
            row["horizon_minutes"] == horizon and row.get(field) is not None
            for row in rows
        )
        name = "2 h" if horizon == 120 else f"{horizon} min"
        labels.append(f"{name}\nn={n}")
    return labels


def _gain_panel(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    field: str,
    title: str,
    ylabel: str,
    colors: dict[str, str],
    *,
    shared_ylim: tuple[float, float] | None = None,
    show_legend: bool = False,
) -> None:
    x = np.arange(len(HORIZON_MINUTES), dtype=float)
    values: list[float] = []
    for alias, subject_rows in _rows_by_subject(rows).items():
        lookup = {row["horizon_minutes"]: row.get(field) for row in subject_rows}
        y = np.asarray([lookup.get(h) for h in HORIZON_MINUTES], dtype=object)
        finite = np.asarray([v is not None and np.isfinite(float(v)) for v in y])
        yf = np.asarray([float(v) if ok else np.nan for v, ok in zip(y, finite)])
        values.extend(yf[np.isfinite(yf)].tolist())
        ax.plot(
            x,
            yf,
            color=colors[alias],
            lw=0.85,
            alpha=0.78,
            marker="o",
            ms=3.4,
            label=alias,
            zorder=3,
        )
    medians = []
    for horizon in HORIZON_MINUTES:
        vals = [
            float(row[field])
            for row in rows
            if row["horizon_minutes"] == horizon and row.get(field) is not None
        ]
        medians.append(float(np.median(vals)) if vals else np.nan)
    ax.plot(x, medians, color=STATE, lw=2.1, marker="D", ms=3.7, zorder=5, label="median")
    if shared_ylim is None:
        max_abs = max([abs(v) for v in values] + [0.02])
        ylim = (-1.16 * max_abs, 1.16 * max_abs)
    else:
        ylim = shared_ylim
    ax.set_ylim(*ylim)
    ax.axhspan(0, ylim[1], color=SUPPORT, alpha=0.55, zorder=-3)
    ax.axhline(0, color="#4D4D4D", lw=0.75, ls=(0, (3, 2)), zorder=1)
    ax.text(
        0.98,
        0.96,
        "supports state  ↑",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#4D766B",
        fontsize=7.0,
    )
    ax.set_xticks(x, _tick_labels(rows, field))
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    if show_legend:
        handles = [
            Line2D([0], [0], color=colors[a], marker="o", lw=0.8, ms=3.2, label=a)
            for a in colors
        ] + [Line2D([0], [0], color=STATE, marker="D", lw=2, ms=3.2, label="median")]
        ax.legend(handles=handles, loc="lower left", ncol=2, handlelength=1.4, columnspacing=0.8)
    _finish_axis(ax)


def _draw_h1_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    event_x = [0.9, 2.2, 3.5, 4.8]
    event_colors = ["#5D7C93", "#8A5A2F", "#5D7C93", "#7E6E84"]
    for x0, color in zip(event_x, event_colors):
        ys = [3.7, 4.15, 4.6]
        ax.scatter([x0] * 3, ys, s=[18, 28, 13], color=color, edgecolor="white", lw=0.45)
        ax.plot([x0, x0], [3.65, 4.65], color=color, lw=0.55, alpha=0.65)
    ax.text(0.75, 5.05, "Past group events", fontsize=8.0, fontweight="bold")
    state_x = np.linspace(0.7, 5.4, 100)
    state_y = 1.9 + 0.45 * np.sin(state_x * 0.95) + 0.13 * state_x
    ax.plot(state_x, state_y, color=STATE, lw=2.0)
    ax.scatter(event_x, np.interp(event_x, state_x, state_y), color=STATE, s=18, zorder=4)
    ax.text(0.75, 0.75, "Candidate state  S(t)", color=STATE, fontsize=8.2, fontweight="bold")
    ax.axvline(5.65, color="#4D4D4D", lw=0.85, ls=(0, (3, 2)))
    ax.add_patch(plt.Rectangle((5.8, 1.0), 3.6, 3.8, facecolor="#EEF1F2", edgecolor="none"))
    future_x = [6.4, 7.05, 8.1, 8.7]
    for x0, n in zip(future_x, [2, 3, 1, 3]):
        ax.scatter([x0] * n, np.linspace(3.7, 4.5, n), s=18, color="#A8B1B6")
    ax.annotate(
        "open-loop",
        xy=(8.9, 2.15),
        xytext=(6.05, 2.15),
        arrowprops=dict(arrowstyle="-|>", color=STATE, lw=1.2),
        color=STATE,
        fontsize=7.4,
        va="center",
    )
    ax.text(6.08, 1.25, "Future event block\n5 / 30 / 120 min", fontsize=8.0, fontweight="bold")
    ax.text(0.0, 5.82, "State must predict a future block", fontsize=8.2, fontweight="bold")


def render_h1(payload: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    _style()
    fig = plt.figure(figsize=(DOUBLE_COLUMN_MM * MM, 68 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.42)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    _draw_h1_schematic(axes[0])
    rows = payload["h1_future_block"]["rows"]
    colors = _subject_palette(payload)
    _gain_panel(
        axes[1], rows, "count_gain_over_multiscale", "Beyond multiscale history",
        "Relative count\nlog-score gain", colors, show_legend=True,
    )
    _gain_panel(
        axes[2], rows, "correct_time_gain_over_shifted", "Time-specific state",
        "Correct-time\ncount gain", colors,
    )
    for letter, ax in zip("ABC", axes):
        _panel(ax, letter)
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.18, top=0.91)
    return _save(fig, out_dir, FIGURE_STEMS["h1"])


def _draw_h2a_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    xy = np.asarray([[1.0, 3.5], [2.3, 4.25], [3.6, 3.2], [5.0, 4.65], [6.4, 3.05], [8.0, 4.2]])
    ax.plot(xy[:3, 0], xy[:3, 1], color="#5F6B70", lw=2.0)
    ax.scatter(xy[:3, 0], xy[:3, 1], s=32, facecolor="#5F6B70", edgecolor="white", lw=0.5, zorder=3)
    ax.text(0.45, 5.15, "Matched early prefix", fontsize=7.1, fontweight="bold")
    ax.annotate("", xy=(5.0, 5.25), xytext=(3.75, 3.35), arrowprops=dict(arrowstyle="-|>", color=STATE, lw=1.1))
    ax.annotate("", xy=(5.0, 1.65), xytext=(3.75, 3.25), arrowprops=dict(arrowstyle="-|>", color=SHIFT, lw=1.1))
    ax.text(7.45, 6.30, "state A", color=STATE, fontsize=7.2, fontweight="bold")
    ax.text(4.7, 0.72, "state B", color=SHIFT, fontsize=7.2, fontweight="bold")
    upper = np.asarray([[5.0, 5.25], [6.35, 5.8], [7.7, 5.1], [8.9, 5.65]])
    lower = np.asarray([[5.0, 1.65], [6.35, 2.15]])
    ax.plot(upper[:, 0], upper[:, 1], color=STATE, lw=1.7)
    ax.scatter(upper[:, 0], upper[:, 1], s=28, color=STATE, edgecolor="white", lw=0.45)
    ax.plot(lower[:, 0], lower[:, 1], color=SHIFT, lw=1.7)
    ax.scatter(lower[:, 0], lower[:, 1], s=28, color=SHIFT, edgecolor="white", lw=0.45)
    ax.plot([6.8, 8.9], [1.8, 1.8], color="#5F6B70", lw=0.75, ls=(0, (2, 2)))
    ax.text(7.85, 1.28, "STOP", ha="center", color="#5F6B70", fontsize=7.4)
    ax.text(
        0.0,
        6.72,
        "State-dependent\nevent path",
        fontsize=7.3,
        fontweight="bold",
        linespacing=0.95,
        va="top",
    )


def render_h2a(payload: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    _style()
    fig = plt.figure(figsize=(DOUBLE_COLUMN_MM * MM, 70 * MM))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1, 1, 1], wspace=0.43)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    _draw_h2a_schematic(axes[0])
    rows = payload["h2a_repertoire"]["rows"]
    colors = _subject_palette(payload)
    all_values = [
        abs(float(row["gain_over_shifted"]))
        for row in rows
        if row.get("gain_over_shifted") is not None
    ]
    limit = max(max(all_values, default=0.01) * 1.12, 0.01)
    shared_ylim = (-limit, limit)
    specs = (
        ("continue", "Continue vs STOP", "Wrong-time − correct-time\nmark NLL"),
        ("positive_size", "Size | continue", ""),
        ("subset", "Contact subset | size", ""),
    )
    for ax, (endpoint, title, ylabel) in zip(axes[1:], specs):
        subset = [row for row in rows if row["endpoint"] == endpoint]
        _gain_panel(
            ax, subset, "gain_over_shifted", title, ylabel, colors,
            shared_ylim=shared_ylim, show_legend=endpoint == "continue",
        )
    for letter, ax in zip("ABCD", axes):
        _panel(ax, letter)
    fig.subplots_adjust(left=0.042, right=0.99, bottom=0.18, top=0.91)
    return _save(fig, out_dir, FIGURE_STEMS["h2a"])


def _empty_or_curves(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    x_field: str,
    y_field: str,
    x_values: list[float],
    x_labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    positive_support: bool = True,
) -> None:
    ax.axhline(0, color="#4D4D4D", lw=0.75, ls=(0, (3, 2)))
    positions = np.arange(len(x_values), dtype=float)
    position_of = {float(value): position for value, position in zip(x_values, positions)}
    ax.set_xlim(-0.12, len(x_values) - 0.88)
    ax.set_xticks(positions, x_labels)
    if rows:
        groups = defaultdict(list)
        for row in rows:
            groups[row.get("alias", row.get("subject", row.get("event_type", "group")))].append(row)
        all_y = []
        for i, (name, group) in enumerate(groups.items()):
            group = sorted(group, key=lambda r: r[x_field])
            x = [position_of[float(r[x_field])] for r in group]
            y = [float(r[y_field]) for r in group]
            all_y.extend(y)
            ax.plot(x, y, marker="o", ms=3.2, lw=0.9, color=SUBJECT_COLORS[i % len(SUBJECT_COLORS)], label=name)
        max_abs = max([abs(v) for v in all_y] + [0.02])
        ax.set_ylim(-1.15 * max_abs, 1.15 * max_abs)
    else:
        ax.set_ylim(-0.12, 0.12)
        ax.text(
            0.5, 0.49, "not yet run", transform=ax.transAxes,
            ha="center", va="center", color="#777777", fontsize=8.2, fontweight="bold",
        )
    if positive_support:
        ax.axhspan(0, ax.get_ylim()[1], color=SUPPORT, alpha=0.55, zorder=-3)
        ax.text(0.98, 0.96, "supports transfer  ↑", transform=ax.transAxes, ha="right", va="top", color="#4D766B", fontsize=7.0)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _finish_axis(ax)


def _feedback_model_panel(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    families = ("count/rate", "mark-specific")
    ax.axhline(0, color="#4D4D4D", lw=0.75, ls=(0, (3, 2)))
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1], families)
    if rows:
        for i, family in enumerate(families):
            vals = [float(r["future_block_log_score_gain_over_no_feedback"]) for r in rows if r["feedback_family"] == family]
            ax.scatter(np.full(len(vals), i), vals, s=22, color=SUBJECT_COLORS[i], alpha=0.8)
            if vals:
                ax.plot([i - 0.16, i + 0.16], [np.median(vals)] * 2, color=STATE, lw=2)
        all_y = [float(r["future_block_log_score_gain_over_no_feedback"]) for r in rows]
        max_abs = max([abs(v) for v in all_y] + [0.02])
        ax.set_ylim(-1.15 * max_abs, 1.15 * max_abs)
    else:
        ax.set_ylim(-0.12, 0.12)
        ax.text(0.5, 0.49, "not yet run", transform=ax.transAxes, ha="center", va="center", color="#777777", fontsize=8.2, fontweight="bold")
    ax.axhspan(0, ax.get_ylim()[1], color=SUPPORT, alpha=0.55, zorder=-3)
    ax.text(0.98, 0.96, "supports feedback  ↑", transform=ax.transAxes, ha="right", va="top", color="#4D766B", fontsize=7.0)
    ax.set_title("IED feedback beyond common state", loc="left", fontweight="bold")
    ax.set_ylabel("Future-block\nlog-score gain")
    _finish_axis(ax)


def render_transfer_feedback(payload: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_MM * MM, 104 * MM))
    risk = payload["h2b_transfer"]["risk_rows"]
    field = payload["h2b_transfer"]["field_rows"]
    _empty_or_curves(
        axes[0, 0], risk, x_field="lead_minutes", y_field="brier_skill_state_over_history",
        x_values=[5, 15, 30, 60, 120, 360], x_labels=["5", "15", "30", "60", "120", "360"],
        title="Seizure-risk transfer", xlabel="Lead to seizure (min)", ylabel="Brier skill\nover history",
    )
    _empty_or_curves(
        axes[0, 1], field, x_field="lead_minutes", y_field="early_ictal_field_gain",
        x_values=[5, 30, 120, 360], x_labels=["5", "30", "120", "360"],
        title="Early-ictal field transfer", xlabel="Lead to seizure (min)", ylabel="Spatial prediction\ngain",
    )
    _feedback_model_panel(axes[1, 0], payload["h3_feedback"]["model_rows"])
    _empty_or_curves(
        axes[1, 1], payload["h3_feedback"]["impulse_rows"],
        x_field="lag_minutes", y_field="functional_state_change",
        x_values=[0, 30, 120, 360], x_labels=["0", "30", "120", "360"],
        title="Event-type impulse response", xlabel="Time after IED (min)", ylabel="Functional-state\nchange",
        positive_support=False,
    )
    for letter, ax in zip("ABCD", axes.flat):
        _panel(ax, letter)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.12, top=0.94, wspace=0.34, hspace=0.48)
    return _save(fig, out_dir, FIGURE_STEMS["transfer_feedback"])


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> dict[str, Any]:
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / f"{stem}.png"
    pdf = figure_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return {
        "png": str(png),
        "pdf": str(pdf),
        "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    tmp.replace(path)


def _write_readme(out_dir: Path) -> Path:
    path = out_dir / "figures" / "README.md"
    path.write_text(
        "# Group-Event State core evidence\n\n"
        "### group_event_state_h1_future_blocks.png\n\n"
        "这张图回答 H1：群体间期事件历史形成的状态，能否在停止读取未来事件后预测未来 5、30、120 分钟的事件数。B 比较状态与可解释 multiscale history，C 比较正确时刻与保留自相关的错时状态；纵轴均为正值支持状态。\n\n"
        "**关注点**：最终承重结果应同时在 B、C 的较长 horizon 位于零线上方，并以患者为分母；当前 pilot 未达到。\n\n"
        "### group_event_state_h2a_repertoire.png\n\n"
        "这张图回答 H2a：给定相同或相近的事件开头，状态是否改变事件继续/停止、继续时的招募规模以及具体触点集合。三个统计 panel 共用 y 轴，避免把接近零的小数方向误读成强效应。\n\n"
        "**关注点**：最终需要 contact subset 与 same-prefix continuation 的患者级增量稳定高于零；只改善 STOP 只能称 extent state。\n\n"
        "### group_event_state_h2b_h3_transfer_feedback.png\n\n"
        "这张图预先固定跨任务与反馈机制的最终接口。A/B 分别放冻结间期状态对发作风险和发作早期空间场的增量；C 比较 no-feedback、count/rate feedback 与 mark-specific feedback；D 显示不同 IED 类型的有符号状态冲击。当前这些实验尚未运行，因此只显示坐标、对照方向和 not yet run，不填模拟数据。\n\n"
        "**关注点**：H2b 必须以 held-out seizure 为分母；H3 必须先控制共同 pre-event state，且冲击允许正负方向。\n"
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("/data/hfosp_group_event_state_v0_3/pilot/summary_main.json"),
    )
    parser.add_argument("--payload", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/group_event_state/core_evidence",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.payload is not None:
        payload = load_payload(args.payload)
        source_summary = None
    else:
        summary = json.loads(args.summary.read_text())
        payload = build_payload(summary)
        source_summary = str(args.summary)
    validate_payload(payload)
    payload_path = args.output_root / "core_evidence_payload.json"
    _atomic_json(payload_path, payload)
    files = {
        "h1": render_h1(payload, args.output_root),
        "h2a": render_h2a(payload, args.output_root),
        "h2b_h3": render_transfer_feedback(payload, args.output_root),
    }
    readme = _write_readme(args.output_root)
    metadata = {
        "asset_id": "group_event_state_core_evidence",
        "paper_slot": "TBD",
        "status": "CANDIDATE_FRAMEWORK_CURRENT_PILOT_NOT_ESTABLISHED",
        "payload_format": payload["format"],
        "payload": str(payload_path),
        "source_summary": source_summary,
        "source_commit": payload["source"]["source_commit"],
        "figures": files,
        "readme": str(readme),
        "font_floor_pt": 7.0,
        "statistical_grammar": "patient-first; positive gain supports hypothesis; missing is not zero",
        "no_synthetic_observations_in_unrun_panels": True,
        "claim_boundary": payload["claim_boundary"],
    }
    metadata_path = args.output_root / "core_evidence_metadata.json"
    _atomic_json(metadata_path, metadata)
    print(json.dumps({"metadata": str(metadata_path), "figures": files}, indent=2))


if __name__ == "__main__":
    main()
