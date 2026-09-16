#!/usr/bin/env python3
"""Render the four v0.3.7 scientific decision figures from one machine summary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_group_event_state_v037_figures")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.topic5_epi_prssm.figure_style import apply_style


GREEN = "#245C3F"
GREEN_LIGHT = "#DCE9E1"
BLUE = "#355C7D"
RUST = "#A35E48"
AMBER = "#D4A72C"
GREY = "#8A8F93"
BLACK = "#252525"
CORE = ("epilepsiae_253", "epilepsiae_958", "epilepsiae_1077", "epilepsiae_1125")
LABEL = {f"epilepsiae_{number}": f"E{number}" for number in (253, 958, 1077, 1125, 916)}
HORIZON_LABEL = {1800: "0.5 h", 7200: "2 h", 21600: "6 h", 28800: "8 h"}


def _style() -> None:
    apply_style()
    plt.rcParams.update({
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8.2,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 6.8,
        "savefig.transparent": False,
    })


def _axis(ax: plt.Axes, title: str, letter: str) -> None:
    ax.axhline(0, color="#555555", lw=0.7, ls=(0, (3, 2)), zorder=1)
    lo, hi = ax.get_ylim()
    if hi > 0:
        ax.axhspan(0, hi, color=GREEN_LIGHT, alpha=0.55, zorder=-5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(-0.15, 1.08, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def _save(fig: plt.Figure, out_dir: Path, stem: str, metadata: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    (out_dir / f"{stem}.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plt.close(fig)


def _horizon_panel(ax: plt.Axes, rows: list[dict[str, Any]], field: str, title: str, letter: str) -> None:
    horizons = (1800, 7200, 21600, 28800); x = np.arange(len(horizons))
    values = []
    for subject in sorted(set(row["subject"] for row in rows)):
        by_h = {row["horizon_seconds"]: row.get(field) for row in rows if row["subject"] == subject}
        y = np.asarray([by_h.get(h, np.nan) if by_h.get(h) is not None else np.nan for h in horizons], float)
        if np.isfinite(y).any():
            style = "-" if subject in CORE else "--"
            ax.plot(x, y, style, lw=0.8, color=GREY, alpha=0.7, marker="o", ms=2.8)
            values.extend(y[np.isfinite(y)].tolist())
    medians = []
    for h in horizons:
        vals = [float(row[field]) for row in rows if row["horizon_seconds"] == h
                and row["subject"] in CORE and row.get(field) is not None]
        medians.append(np.median(vals) if vals else np.nan)
    ax.plot(x, medians, color=GREEN, lw=2.0, marker="o", ms=4.0, label="core median")
    bound = max([abs(v) for v in values] + [0.02]) * 1.16
    ax.set_ylim(-bound, bound); ax.set_xticks(x, [HORIZON_LABEL[h] for h in horizons])
    ax.set_xlabel("Future horizon"); ax.set_ylabel("Held-out score gain")
    _axis(ax, title, letter)


def _horizon_group_panel(
    ax: plt.Axes, rows: list[dict[str, Any]], field: str, title: str, letter: str,
) -> None:
    horizons = (1800, 7200, 21600, 28800); x = np.arange(len(horizons))
    styles = (("burden", BLUE, "burden"), ("grammar", RUST, "conditional grammar"))
    all_values = []
    for suffix, color, label in styles:
        key = f"{field}_{suffix}"
        for subject in sorted(set(row["subject"] for row in rows)):
            by_h = {row["horizon_seconds"]: row.get(key) for row in rows if row["subject"] == subject}
            y = np.asarray([by_h.get(h, np.nan) if by_h.get(h) is not None else np.nan for h in horizons], float)
            if np.isfinite(y).any():
                ax.plot(x, y, "--" if subject not in CORE else "-", lw=0.55,
                        color=color, alpha=0.22 if subject in CORE else 0.5,
                        marker="o", ms=2.0)
                all_values.extend(y[np.isfinite(y)].tolist())
        medians = []
        for horizon in horizons:
            vals = [float(row[key]) for row in rows if row["horizon_seconds"] == horizon
                    and row["subject"] in CORE and row.get(key) is not None]
            medians.append(np.median(vals) if vals else np.nan)
        ax.plot(x, medians, color=color, lw=1.8, marker="o", ms=3.6, label=label)
    bound = max([abs(value) for value in all_values] + [0.02]) * 1.16
    ax.set_ylim(-bound, bound); ax.set_xticks(x, [HORIZON_LABEL[h] for h in horizons])
    ax.set_xlabel("Future horizon"); ax.set_ylabel("Held-out score gain")
    _axis(ax, title, letter)


def render_h1(summary: dict[str, Any], out_dir: Path) -> None:
    _style(); fig, axes = plt.subplots(2, 4, figsize=(7.09, 4.55))
    event = summary["h1"]["event_by_horizon"]
    grid = summary["h1"]["grid_by_horizon"]
    dual = summary["h1"]["dual_by_horizon"]
    panels = (
        (event, "gain_over_mark_ewma", "Event SSM vs mark history"),
        (grid, "gain_over_mark_ewma", "5-min slow SSM vs history"),
        (dual, "persistent_background_over_current", "Persistent background"),
        (dual, "event_after_background", "Event gain after background"),
        (event, "dynamic_over_constant", "Event SSM: dynamic vs level"),
        (grid, "dynamic_over_constant", "Slow SSM: dynamic vs level"),
        (event, "correct_time_over_shifted", "Event SSM: correct time"),
        (dual, "correct_time_over_shifted", "Dual state: correct vs shifted"),
    )
    for ax, (rows, field, title), letter in zip(axes.flat, panels, "ABCDEFGH"):
        _horizon_group_panel(ax, rows, field, title, letter)
    axes[0, 0].legend(frameon=False, loc="best")
    fig.tight_layout(w_pad=1.2, h_pad=1.5)
    _save(fig, out_dir, "group_event_state_v037_h1_shared_horizons", {
        "question": "Does one causal state retain time-specific information across 0.5, 2, 6 and 8 h?",
        "positive_direction": "supports the named increment", "statistical_unit": "patient",
    })


def _subject_endpoint_panel(
    ax: plt.Axes, rows: list[dict[str, Any]], fields: list[tuple[str, str]], title: str, letter: str,
) -> None:
    x = np.arange(len(fields)); all_values = []
    for subject_row in rows:
        if subject_row["subject"] not in CORE:
            continue
        y = [subject_row["median"].get(field) for field, _label in fields]
        finite = [float(v) for v in y if v is not None]
        all_values.extend(finite)
        ax.plot(x, [np.nan if v is None else v for v in y], color=GREY, alpha=0.45, lw=0.7)
        ax.scatter(x, [np.nan if v is None else v for v in y], s=18, color=BLUE, zorder=3)
    med = []
    for field, _label in fields:
        vals = [row["median"].get(field) for row in rows if row["subject"] in CORE]
        vals = [float(v) for v in vals if v is not None]
        med.append(np.median(vals) if vals else np.nan)
    ax.scatter(x, med, marker="_", s=170, linewidths=2.2, color=GREEN, zorder=4)
    bound = max([abs(v) for v in all_values] + [0.02]) * 1.15
    ax.set_ylim(-bound, bound); ax.set_xticks(x, [label for _field, label in fields], rotation=28, ha="right")
    ax.set_ylabel("Held-out score gain"); _axis(ax, title, letter)


def render_h2a(summary: dict[str, Any], out_dir: Path) -> None:
    _style(); fig, axes = plt.subplots(2, 2, figsize=(7.09, 4.5))
    core_fields = [
        ("state_gain_over_static_contact", "contacts"),
        ("state_gain_over_static_stop", "STOP"),
        ("state_gain_over_static_grammar", "grammar"),
    ]
    _subject_endpoint_panel(axes[0, 0], summary["h2a"]["event"], core_fields,
                            "Event-only state", "A")
    _subject_endpoint_panel(axes[0, 1], summary["h2a"]["grid"], core_fields,
                            "5-min slow state", "B")
    _subject_endpoint_panel(axes[1, 0], summary["h2a"]["dual"], core_fields,
                            "Background + event state", "C")
    joint_rows = [
        {"subject": row["subject"], "median": row.get("h2a_median", {})}
        for row in summary["h2a"].get("joint_sensitivity", {}).get("rows", [])
    ]
    joint_fields = [
        ("joint_gain_over_static_grammar", "joint grammar"),
        ("constant_unexplained_grammar", "beyond level"),
        ("correct_time_paired_grammar", "correct time"),
    ]
    _subject_endpoint_panel(
        axes[1, 1], joint_rows, joint_fields,
        "Joint interictal sensitivity", "D",
    )
    fig.tight_layout(w_pad=1.4, h_pad=1.4)
    _save(fig, out_dir, "group_event_state_v037_h2a_frozen_decoder", {
        "question": "Does pre-event state change the continuation of a matched group event?",
        "decoder": "strict time-split frozen contact-sequence decoder",
        "positive_control": "leaked current-event oracle", "statistical_unit": "patient",
        "joint_panel_boundary": "supportive interictal-only producer update; H1 is mandatorily re-scored",
    })


def render_h2b(summary: dict[str, Any], out_dir: Path) -> None:
    _style(); rows = summary["h2b"]["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.45), gridspec_kw={"width_ratios": [1.05, 1.2, 1.2]})
    subjects = [row["subject"] for row in rows]
    risk_level = {"NOT_ESTIMABLE": 0, "SINGLE_HELD_OUT_SEIZURE_DESCRIPTIVE_ONLY": 1,
                  "REPEATED_HELD_OUT_SEIZURES": 2}
    matrix = np.asarray([[
        risk_level.get(row.get("scientific_repeatability_status", "NOT_ESTIMABLE"), 0),
        2 if any(v == "ESTIMATED" for lead in row["field_status"].values() for v in lead.values()) else 0,
    ] for row in rows], dtype=float) if rows else np.zeros((0, 2))
    axes[0].imshow(matrix, cmap=matplotlib.colors.ListedColormap(["#E5E7E9", AMBER, GREEN]),
                   vmin=0, vmax=2, aspect="auto")
    axes[0].set_xticks([0, 1], ["risk", "early field"]); axes[0].set_yticks(range(len(subjects)), [LABEL.get(s, s) for s in subjects])
    for i, row in enumerate(rows):
        counts = row.get("seizures_by_phase", {})
        axes[0].text(0, i, f"{counts.get('FIT', 0)}/{counts.get('INNER', 0)}/{counts.get('SELECTION', 0)}",
                     ha="center", va="center", color="white" if matrix[i, 0] == 2 else BLACK, fontsize=6.2)
    axes[0].set_title("Estimability", loc="left", fontweight="bold")
    axes[0].text(-0.18, 1.08, "A", transform=axes[0].transAxes, fontsize=10, fontweight="bold", va="top")
    for ax, keys, title, letter in (
        (axes[1], [("event_only_state_gain_over_mark_history", "event state"),
                   ("grid_state_gain_over_mark_history", "slow state"),
                   ("state_gain_over_background_censored_logscore", "dual state")],
         "Seizure-risk transfer", "B"),
        (axes[2], [("event_correct_time_gain_over_shift", "event time"),
                   ("event_dynamic_gain_over_fit_period_mean", "event level"),
                   ("dual_correct_time_gain_over_shift", "dual time"),
                   ("dual_dynamic_gain_over_fit_period_mean", "dual level")],
         "Temporal specificity", "C"),
    ):
        x = np.arange(len(keys)); vals = []
        for offset, row in enumerate(rows):
            if row["hazard_status"] != "ESTIMATED": continue
            for xi, (key, _label) in enumerate(keys):
                value = row["hazard_contrasts"].get(key)
                if value is not None:
                    repeated = row.get("scientific_repeatability_status") == "REPEATED_HELD_OUT_SEIZURES"
                    vals.append(float(value)); ax.scatter(
                        xi + (offset-len(rows)/2)*0.04, value, s=24,
                        color=BLUE if repeated else AMBER, marker="o" if repeated else "^",
                    )
                    # Only two subjects have an estimable hazard in this pilot,
                    # and their effects sit near zero.  Put repeated-seizure
                    # and one-seizure labels on opposite sides of the point so
                    # the exact overlap is visible rather than overprinted.
                    ax.annotate(
                        LABEL.get(row["subject"], row["subject"]),
                        (xi + (offset-len(rows)/2)*0.04, value),
                        xytext=(3, 4 if repeated else -9), textcoords="offset points",
                        fontsize=5.8, ha="left", va="bottom" if repeated else "top",
                    )
        bound = max([abs(v) for v in vals] + [0.02]) * 1.25
        ax.set_ylim(-bound, bound)
        rotation = 20 if len(keys) > 3 else 0
        ax.set_xticks(
            x, [label for _key, label in keys], rotation=rotation,
            ha="right" if rotation else "center",
        )
        ax.set_ylabel("Censored log-score gain"); _axis(ax, title, letter)
    fig.tight_layout(w_pad=1.3)
    _save(fig, out_dir, "group_event_state_v037_h2b_frozen_transfer", {
        "question": "Does an interictal-only frozen state transfer to seizure risk or early ictal fields?",
        "missing_rule": "not-estimable cells are grey and are never plotted as zero",
        "statistical_unit": "held-out seizure within patient, then patient",
    })


def render_h3(summary: dict[str, Any], out_dir: Path) -> None:
    _style(); fig, axes = plt.subplots(2, 3, figsize=(7.09, 4.45))
    axes = axes.flat
    names = ("common_drive", "count_feedback", "mark_feedback")
    labels = ("common\ndrive", "count\nedge", "mark\nedge")
    x = np.arange(3)

    one = summary["h3"].get("one_step_instrument") or {}
    truth = one.get("by_truth_median", {})
    count = [truth.get(name, {}).get("count_feedback_gain_on_future_background", np.nan) for name in names]
    mark = [truth.get(name, {}).get("mark_feedback_gain_on_future_background", np.nan) for name in names]
    axes[0].bar(x - 0.16, count, width=0.3, color=BLUE, label="count")
    axes[0].bar(x + 0.16, mark, width=0.3, color=RUST, label="mark")
    axes[0].set_xticks(x, labels); axes[0].set_ylabel("Recovery gain")
    axes[0].legend(frameon=False); _axis(axes[0], "One-step instrument", "A")

    persistent = summary["h3"].get("persistent_instrument") or {}
    truth = persistent.get("by_truth_median", {})
    count = [truth.get(name, {}).get("persistent_count_over_one_step_background", np.nan) for name in names]
    mark = [truth.get(name, {}).get("persistent_mark_over_one_step_background", np.nan) for name in names]
    axes[1].bar(x - 0.16, count, width=0.3, color=BLUE, label="count")
    axes[1].bar(x + 0.16, mark, width=0.3, color=RUST, label="mark")
    axes[1].set_xticks(x, labels); axes[1].set_ylabel("Recovery gain")
    _axis(axes[1], "Persistent instrument", "B")

    humans = summary["h3"].get("persistent_human", {}).get("rows", [])
    human_x = np.arange(len(humans))

    def human_panel(ax: plt.Axes, fields: tuple[tuple[str, str, str], ...], title: str, letter: str) -> None:
        vals = []
        offsets = np.linspace(-0.12, 0.12, len(fields)) if len(fields) > 1 else np.zeros(1)
        for offset, (field, label, color) in zip(offsets, fields):
            y = []
            for row in humans:
                value = row["median"].get(field)
                y.append(np.nan if value is None else float(value))
                if value is not None: vals.append(float(value))
            ax.scatter(human_x + offset, y, s=24, color=color, label=label)
        ax.set_xticks(human_x, [LABEL.get(row["subject"], row["subject"]) for row in humans], rotation=25, ha="right")
        bound = max([abs(v) for v in vals] + [0.02]) * 1.2
        ax.set_ylim(-bound, bound); ax.set_ylabel("Held-out gain")
        ax.legend(frameon=False, loc="best"); _axis(ax, title, letter)

    human_panel(
        axes[2], (("persistent_count_over_one_step_background", "count history", BLUE),),
        "Count history to background", "C",
    )
    human_panel(
        axes[3], (("persistent_mark_over_one_step_background", "mark history", RUST),),
        "Mark history to background", "D",
    )
    # The un-floored contrast is not reportable: a fitted same-capacity placebo
    # can generalise worse than having no edge at all, and then this panel plots
    # the placebo's overfitting rather than the real edge's skill (E1125 raw
    # +0.757 vs floored +0.089).  Panel E therefore reads the floored keys and
    # only falls back to the raw ones for cards written before the fix.
    floored_keys = (
        ("persistent_count_real_over_floored_wrong_time_background", "count correct-time", BLUE),
        ("persistent_mark_real_over_floored_wrong_time_background", "mark correct-time", RUST),
    )
    legacy_keys = (
        ("persistent_count_real_over_fitted_wrong_time_background", "count correct-time (UNFLOORED)", BLUE),
        ("persistent_mark_real_over_fitted_wrong_time_background", "mark correct-time (UNFLOORED)", RUST),
    )
    has_floor = any(
        key in (row.get("median") or {})
        for row in humans for key, _, _ in floored_keys
    )
    human_panel(
        axes[4], floored_keys if has_floor else legacy_keys,
        "Correct vs wrong-time (null floored at no-edge)" if has_floor
        else "Correct vs fitted wrong-time (NULL NOT FLOORED)", "E",
    )

    scales = ("7200", "21600", "86400", "172800")
    matrix = np.asarray([
        [float(row.get("eligible_physical_scales", {}).get(scale, {}).get("eligible_all_seeds", False)) for scale in scales]
        for row in humans
    ]) if humans else np.zeros((0, len(scales)))
    axes[5].imshow(matrix, cmap=matplotlib.colors.ListedColormap(["#E5E7E9", GREEN]), vmin=0, vmax=1, aspect="auto")
    axes[5].set_xticks(range(4), ["2 h", "6 h", "24 h", "48 h"])
    axes[5].set_yticks(range(len(humans)), [LABEL.get(row["subject"], row["subject"]) for row in humans])
    axes[5].set_title("Physical-scale estimability", loc="left", fontweight="bold")
    axes[5].text(-0.15, 1.08, "F", transform=axes[5].transAxes, fontsize=10, fontweight="bold", va="top")

    fig.tight_layout(w_pad=1.35, h_pad=1.35)
    _save(fig, out_dir, "group_event_state_v037_h3_independent_generation", {
        "question": "Do count or mark histories add persistent, correctly timed prediction beyond one-step common drive?",
        "panel_boundary": "Synthetic recovery validates the instrument only; human panels require correct-time and physical support",
        "claim_boundary": summary["h3"]["model_boundary"], "observer_state_used_as_jump": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True); args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8")); args.out_dir.mkdir(parents=True, exist_ok=True)
    render_h1(summary, args.out_dir); render_h2a(summary, args.out_dir)
    render_h2b(summary, args.out_dir); render_h3(summary, args.out_dir)
    (args.out_dir / "README.md").write_text(
        """# v0.3.7 核心证据图

### group_event_state_v037_h1_shared_horizons.png

检验同一个因果状态是否在 0.5、2、6、8 小时预测未来事件负荷与条件传播组成。图同时比较逐事件 SSM、五分钟慢链、背景流、常数慢水平和错时状态；蓝线是负荷，棕线是控制负荷后的 grammar。

**关注点**：事件历史是否在持续背景和透明多尺度历史之外仍有正确时刻的动态增量。

### group_event_state_v037_h2a_frozen_decoder.png

冻结成熟 contact-sequence decoder 后，只训练状态调制接口，检查相同事件开头之后的 STOP、触点和传播 grammar 是否因事件前状态而改变。第四格是允许间期形态损失更新 observer 的支持性实验，并必须回头重评 H1。

**关注点**：冻结跨读出是否成立；若只有 joint sensitivity 阳性，说明状态需要形态目标训练，不能冒充原 H1 状态。

### group_event_state_v037_h2b_frozen_transfer.png

显示完全由间期任务学习并冻结的状态，能否迁移到发作风险与早期发作空间场。灰格是不可估，黄色是仅一场 held-out 发作的描述，绿色才有重复 held-out 发作；风险增量还必须超过错时状态和训练期常数。

**关注点**：结论分母必须是独立发作，不是重叠时间格；水平偏移不算时刻特异易感状态。

### group_event_state_v037_h3_independent_generation.png

分别显示一步和长时反馈仪器的合成恢复，再展示人体中负荷或形态历史能否预测更远的非事件背景、能否胜过同容量错时暴露，以及各物理尺度是否可估。observer hidden update 从未作为 H3 证据。

**关注点**：只有具备物理支持、超过一步模型并胜过错时对照的结果才是反馈式方向依赖候选；仍不能写成干预意义的因果。
""",
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), "figures": 4}, indent=2))


if __name__ == "__main__":
    main()
