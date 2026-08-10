"""Plot the zero-simulation rev9 review corrections."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_AUDIT = Path(
    "results/topic4_sef_hfo/data_driven_core_field_rev9/"
    "review_audit_20260810/rev9_review_audit.json")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()


def _panel_label(axis, label):
    axis.text(-0.14, 1.08, label, transform=axis.transAxes,
              fontsize=10, fontweight="bold", va="top")


def _interval_error(row):
    estimate = float(row["estimate"])
    low, high = map(float, row["interval_95"])
    return estimate, np.asarray([[estimate - low], [high - estimate]])


def _plot_optimization_diagnosis(audit, out_dir):
    diagnosis = audit["optimization_diagnosis"]
    candidates = [row for row in diagnosis["candidates"]
                  if row["support_eligible"] and row["distance"] is not None]
    phase_colors = ("#999999", "#0072B2", "#009E73")
    phase_labels = ("rev8 K=2", "rev8 K=3", "rev8.1 K=3")
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.5), constrained_layout=True)

    axis = axes[0]
    for phase in range(3):
        rows = [row for row in candidates if row["phase_index"] == phase]
        axis.scatter(
            [row["distance"] for row in rows], [row["worst_mode"] for row in rows],
            s=22, color=phase_colors[phase], alpha=0.68, label=phase_labels[phase])
    pareto = sorted(
        [row for row in candidates if row["pareto_distance_worst_mode"]],
        key=lambda row: row["distance"])
    axis.plot([row["distance"] for row in pareto],
              [row["worst_mode"] for row in pareto],
              color="#222222", linewidth=1.0, linestyle="--", label="archive Pareto")
    markers = {
        "selected_candidate": ("selected", "D", "#E69F00"),
        "final_unseen_candidate": ("final unseen", "X", "#D55E00"),
    }
    for key, (label, marker, color) in markers.items():
        row = diagnosis[key]
        axis.scatter(row["distance"], row["worst_mode"], s=75, marker=marker,
                     color=color, edgecolor="black", linewidth=0.5, label=label, zorder=5)
    for key, label, marker in (
        ("hand_placed_two_cores", "hand dual-core", "s"),
        ("stage2_filament", "filament", "P"),
    ):
        row = diagnosis["rigid_controls"][key]
        axis.scatter(row["distance"], row["worst_mode"], s=68, marker=marker,
                     facecolor="white", edgecolor="#333333", linewidth=1.0,
                     label=label, zorder=5)
    axis.axvspan(0.0, diagnosis["patient_floor_p95"], color="#E6E6E6", alpha=0.8)
    axis.set_xlabel("global event-cloud distance (lower is better)")
    axis.set_ylabel("weakest-mode correlation (higher is better)")
    axis.set_title("candidate archive exposes a trade-off", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=6.3, ncol=2, loc="lower left")
    _panel_label(axis, "a")

    axis = axes[1]
    for phase in range(3):
        rows = [row for row in candidates if row["phase_index"] == phase]
        axis.scatter([row["mode_a"] for row in rows], [row["mode_b"] for row in rows],
                     s=22, color=phase_colors[phase], alpha=0.68)
    for key, (_, marker, color) in markers.items():
        row = diagnosis[key]
        axis.scatter(row["mode_a"], row["mode_b"], s=75, marker=marker,
                     color=color, edgecolor="black", linewidth=0.5, zorder=5)
    for key, marker in (("hand_placed_two_cores", "s"), ("stage2_filament", "P")):
        row = diagnosis["rigid_controls"][key]
        axis.scatter(row["mode_a"], row["mode_b"], s=68, marker=marker,
                     facecolor="white", edgecolor="#333333", linewidth=1.0, zorder=5)
    axis.axhline(0.0, color="#AAAAAA", linewidth=0.6)
    axis.axvline(0.0, color="#AAAAAA", linewidth=0.6)
    axis.scatter(1.0, 1.0, marker="*", s=85, color="#222222", zorder=6)
    axis.annotate(
        "patient target", xy=(1.0, 1.0), xytext=(0.64, 0.86),
        textcoords="data", fontsize=6.5,
        arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.6})
    axis.set_xlim(-1.02, 1.05)
    axis.set_ylim(-1.02, 1.05)
    axis.set_xlabel("mode A correlation")
    axis.set_ylabel("mode B correlation")
    axis.set_title("mode B saturates; mode A is unprotected", loc="left", fontweight="bold")
    _panel_label(axis, "b")

    axis = axes[2]
    offset = 0
    tick_positions, tick_labels = [], []
    for phase, execution in enumerate(diagnosis["executions"]):
        for generation in execution["generations"]:
            rows = [row for row in diagnosis["candidates"]
                    if row["phase_index"] == phase and row["generation"] == generation]
            supported = [row for row in rows if row["support_eligible"]]
            position = offset + generation
            tick_positions.append(position)
            tick_labels.append(f"P{phase + 1}:G{generation}")
            if supported:
                axis.scatter(position, max(row["worst_mode"] for row in supported),
                             color=phase_colors[phase], s=36, zorder=3)
                axis.vlines(
                    position, min(row["worst_mode"] for row in supported),
                    max(row["worst_mode"] for row in supported),
                    color=phase_colors[phase], alpha=0.35, linewidth=3)
        offset += max(execution["generations"]) + 2
    axis.axhline(
        diagnosis["rigid_controls"]["hand_placed_two_cores"]["worst_mode"],
        color="#333333", linestyle="--", linewidth=0.8, label="hand dual-core")
    axis.set_xticks(tick_positions, tick_labels, rotation=55, ha="right")
    axis.set_ylabel("supported-candidate worst mode")
    axis.set_title("all phases stopped at max generations", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="lower right")
    _panel_label(axis, "c")

    stem = out_dir / "rev9_optimizer_objective_diagnosis"
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return str(stem.with_suffix(".png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--out-dir")
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).read_text())
    out_dir = Path(args.out_dir or Path(args.audit).parent / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "Null": "#8C8C8C", "Node": "#0072B2", "Edge": "#D55E00",
        "Node+Edge": "#009E73", "mode_a": "#CC79A7", "mode_b": "#56B4E9",
    }
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.6), constrained_layout=True)

    sites = audit["local_response"]["site_table"]
    names = [row["site_id"].replace("component_", "C").replace("control_", "X")
             for row in sites]
    x = np.arange(len(sites))
    axis = axes[0, 0]
    source = np.asarray([
        np.nan if row["source_gain_ratio_median"] is None
        else row["source_gain_ratio_median"] for row in sites], float)
    downstream = np.asarray([
        np.nan if row["downstream_gain_ratio_median"] is None
        else row["downstream_gain_ratio_median"] for row in sites], float)
    axis.axhspan(0.8, 1.25, color="#E6E6E6", zorder=0, label="post-hoc reference")
    axis.scatter(x - 0.12, source, color="#CC79A7", label="source", zorder=3)
    axis.scatter(x + 0.12, downstream, color="#009E73", label="downstream", zorder=3)
    for index, row in enumerate(sites):
        axis.text(index, -0.22, f"{row['n_valid_pairs']}/12", ha="center", fontsize=6.5)
    axis.axhline(1.0, color="#555555", linewidth=0.7)
    axis.set_xticks(x, names)
    axis.set_ylim(-0.35, max(1.9, np.nanmax([source, downstream]) + 0.15))
    axis.set_ylabel("Edge / Node positive-response gain")
    axis.set_title("equivalence is unresolved", loc="left", fontweight="bold")
    axis.legend(frameon=False, ncol=3, loc="upper left")
    _panel_label(axis, "a")

    axis = axes[0, 1]
    map_rho = np.asarray([
        np.nan if row["positive_map_spearman_median"] is None
        else row["positive_map_spearman_median"] for row in sites], float)
    r90 = np.asarray([
        np.nan if row["r90_edge_minus_node_median_mm"] is None
        else row["r90_edge_minus_node_median_mm"] for row in sites], float)
    axis.bar(x, map_rho, color=["#CC79A7"] * 3 + ["#999999"] * 3,
             edgecolor="white")
    axis.axhline(0.8, color="#555555", linestyle="--", linewidth=0.8,
                 label="map rho reference")
    twin = axis.twinx()
    twin.plot(x, r90, color="#D55E00", marker="o", linewidth=1.0,
              label="r90 Edge-Node")
    twin.axhspan(-1.0, 1.0, color="#F5C6AA", alpha=0.18)
    axis.set_xticks(x, names)
    axis.set_ylim(0.0, 1.0)
    twin.set_ylim(-1.2, 1.2)
    axis.set_ylabel("positive-map Spearman rho")
    twin.set_ylabel("r90 difference (mm)")
    axis.set_title("width matches; maps do not", loc="left", fontweight="bold")
    _panel_label(axis, "b")

    axis = axes[0, 2]
    sensitivity = audit["common_detector"]["sensitivities"]
    multiplier = np.asarray([row["multiplier"] for row in sensitivity], float)
    for arm in ("Null", "Node", "Edge", "Node+Edge"):
        values = [row["arm_summaries"][arm]["event_rate_hz"] for row in sensitivity]
        estimate = np.asarray([row["estimate"] for row in values], float)
        low = np.asarray([row["interval_95"][0] for row in values], float)
        high = np.asarray([row["interval_95"][1] for row in values], float)
        axis.plot(multiplier, estimate, marker="o", color=colors[arm], label=arm)
        axis.fill_between(multiplier, low, high, color=colors[arm], alpha=0.12)
    axis.set_xticks(multiplier)
    axis.set_xlabel("common threshold multiplier")
    axis.set_ylabel("event rate (Hz)")
    axis.set_title("common-detector sensitivity", loc="left", fontweight="bold")
    axis.legend(frameon=False, ncol=2, loc="upper left")
    _panel_label(axis, "c")

    axis = axes[1, 0]
    threshold_free = audit["common_detector"]["threshold_free"]
    arms = list(ARM_ORDER := ("Null", "Node", "Edge", "Node+Edge"))
    for index, arm in enumerate(arms):
        estimate, error = _interval_error(threshold_free[arm]["mean_active_fraction"])
        axis.errorbar(index, estimate, yerr=error, fmt="o", color=colors[arm],
                      capsize=3)
    axis.set_xticks(np.arange(4), arms, rotation=20, ha="right")
    axis.set_ylabel("mean active-neuron fraction")
    axis.set_title("threshold-free activity burden", loc="left", fontweight="bold")
    _panel_label(axis, "d")

    mode_audit = audit["factorial_mode_audit"]["arms"]
    axis = axes[1, 1]
    width = 0.34
    for mode_index, (mode_key, label) in enumerate((("mode_a", "mode A"), ("mode_b", "mode B"))):
        estimates, lows, highs = [], [], []
        for arm in ("Node", "Node+Edge"):
            row = mode_audit[arm]["network_mode_repertoire"][
                f"{mode_key}_rate_hz"]
            estimates.append(row["estimate"])
            lows.append(row["interval_95"][0])
            highs.append(row["interval_95"][1])
        estimates = np.asarray(estimates, float)
        positions = np.arange(2) + (mode_index - 0.5) * width
        error = np.vstack((estimates - lows, np.asarray(highs) - estimates))
        axis.bar(positions, estimates, width=width, color=colors[mode_key],
                 yerr=error, capsize=3, label=label)
    axis.set_xticks(np.arange(2), ["Node", "Node+Edge"])
    axis.set_ylabel("in-distribution event rate (Hz)")
    axis.set_title("both modes occur within every network", loc="left", fontweight="bold")
    axis.legend(frameon=False)
    _panel_label(axis, "e")

    axis = axes[1, 2]
    positions = np.arange(2)
    for arm_index, arm in enumerate(("Node", "Node+Edge")):
        values = []
        errors = []
        for mode in (0, 1):
            row = mode_audit[arm]["hierarchical_profile_spearman_95"][mode]
            values.append(row["estimate"])
            errors.append(row["interval_95"])
        values = np.asarray(values, float)
        errors = np.asarray(errors, float)
        offset = (arm_index - 0.5) * 0.16
        axis.errorbar(
            positions + offset, values,
            yerr=np.vstack((values - errors[:, 0], errors[:, 1] - values)),
            fmt="o", color=colors[arm], capsize=3, label=arm)
    axis.axhline(0.0, color="#999999", linewidth=0.6)
    axis.set_xticks(positions, ["mode A", "mode B"])
    axis.set_ylim(-0.05, 1.05)
    axis.set_ylabel("patient prototype Spearman rho")
    axis.set_title("mode A remains the limiting phenotype", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="center right")
    _panel_label(axis, "f")

    stem = out_dir / "rev9_review_corrections"
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    metadata = dict(
        status="REV9_REVIEW_CORRECTION_FIGURE_COMPLETE",
        scientific_role=(
            "post-execution correction figure; no new SNN simulation and no patient held-out readout"),
        source=dict(path=args.audit, sha256=_sha256(args.audit)),
        git_commit=_git_commit(),
    )
    stem.with_name(stem.name + "_metadata").with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    optimization_figure = _plot_optimization_diagnosis(audit, out_dir)
    metadata["optimization_figure"] = optimization_figure
    stem.with_name(stem.name + "_metadata").with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    (out_dir / "README.md").write_text(
        "### rev9_review_corrections.png\n\n"
        "这张图汇总 rev9 审阅后的零仿真纠错。上排检查 alpha=0.75 的逐位置正响应增益、空间响应与统一 detector 敏感性；下排展示 threshold-free 活动负荷、网络层 mode-specific event rate 和 hierarchical bootstrap 的患者 prototype 一致性。\n\n"
        "图中 response reference band 是审阅后诊断，不是事前通过门；Null 和 Edge-only 因高 OOD、低 in-support 事件数不进入患者模式排名。\n\n"
        "**关注点**：source gain 与 map rho 未闭合，Edge-only 不点火，Node+Edge 增强活动负荷但没有修复 mode A。\n"
        "\n### rev9_optimizer_objective_diagnosis.png\n\n"
        "这张图对已有 rev8/rev8.1 候选做零仿真重评分。左图比较全局 event-cloud distance 与 weakest-mode correlation，中图展开 mode A/B，右图显示每一代 supported candidates 的范围；手放双核和细丝只作既有 benchmark。\n\n"
        "**关注点**：训练候选中存在比最终候选更好的 weakest-mode 点，但这不能证明 selection-seed 上存在漏选；L0 已直接证明旧目标不保护 mode A，优化不足和 family limitation 仍需后续容量实验区分。\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
