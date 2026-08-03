#!/usr/bin/env python3
"""Paper-ready cohort and representative figures for Topic 5 v0.4."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "M0": "#6B7280",
    "M1": "#4C78A8",
    "M2": "#E69F00",
    "M3": "#B2182B",
    "positive": "#B2182B",
    "negative": "#2166AC",
    "null": "#B9BEC6",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _clean(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, pad=2)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def _paired_delta_panel(
    ax: plt.Axes,
    table: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    colors: list[str],
    comparisons: list[dict],
) -> None:
    rng = np.random.default_rng(20260803)
    for index, (column, color, result) in enumerate(zip(columns, colors, comparisons)):
        values = table[column].dropna().to_numpy(float)
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=17,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        median = np.median(values)
        ax.plot([index - 0.22, index + 0.22], [median, median], color="black", lw=1.4, zorder=4)
        ax.text(
            index,
            ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0,
            "",
        )
        labels[index] = (
            f"{labels[index]}\nmed {median:+.3f}\n"
            f"P={result['p_two_sided_exact']:.3g}, n={result['n_patients']}"
        )
    ax.axhline(0, color="#333333", lw=0.8, ls="--", zorder=1)
    ax.set_xticks(range(len(columns)), labels)
    ax.set_ylabel(r"Patient $\Delta$ maxAB")
    _clean(ax)


def _draw_task(ax: plt.Axes) -> None:
    ax.set_axis_off()
    x = np.linspace(0.08, 0.72, 8)
    gradients = [np.linspace(0, 1, 8), np.linspace(1, 0, 8)]
    for row, values in enumerate(gradients):
        y = 0.74 - row * 0.25
        ax.plot([x.min(), x.max()], [y, y], color="#D1D5DB", lw=1, zorder=0)
        ax.scatter(x, np.full_like(x, y), c=values, cmap="viridis", vmin=0, vmax=1, s=28, zorder=2)
        ax.text(0.01, y, f"Static {'A' if row == 0 else 'B'}", va="center", ha="left")
    ax.annotate(
        "+ causal history residual",
        xy=(0.52, 0.31),
        xytext=(0.18, 0.16),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": COLORS["M3"]},
        color=COLORS["M3"],
        ha="center",
    )
    ax.text(0.79, 0.61, r"$\{\widehat F^A,\widehat F^B\}$", fontsize=9, ha="center")
    ax.text(0.79, 0.43, "maxAB against\nearly ictal field", ha="center", va="center")
    ax.text(
        0.50,
        0.005,
        "Set-valued, sign-free refinement\n(no unique direction label)",
        ha="center",
        va="bottom",
        fontsize=5.9,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _draw_models(ax: plt.Axes) -> None:
    ax.set_axis_off()
    rows = [
        ("M0", "Static A/B only", "no learned history", COLORS["M0"]),
        ("M1", "Frozen history state + head", "state readability", COLORS["M1"]),
        ("M2", "Fixed time summary + head", "nonrecurrent content", COLORS["M2"]),
        ("M3", "Joint history RNN + head", "history-conditioned refinement", COLORS["M3"]),
    ]
    for index, (name, title, question, color) in enumerate(rows):
        y = 0.85 - index * 0.22
        ax.add_patch(
            mpl.patches.FancyBboxPatch(
                (0.02, y - 0.075),
                0.95,
                0.14,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor=mpl.colors.to_rgba(color, 0.09),
                edgecolor=color,
                lw=0.9,
            )
        )
        ax.text(0.06, y, name, color=color, fontweight="bold", va="center", fontsize=8.5)
        ax.text(0.20, y + 0.023, title, va="center", fontweight="bold", fontsize=7.1)
        ax.text(0.20, y - 0.033, question, va="center", color="#4B5563", fontsize=6.4)
    ax.text(
        0.50,
        0.005,
        "Same static A/B basis and target\nPatient-level LOSO",
        ha="center",
        va="bottom",
        fontsize=5.9,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _main_figure(root: Path, patient: pd.DataFrame, summary: dict, output: Path) -> None:
    _style()
    fig = plt.figure(figsize=(7.35, 7.15), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.985,
        bottom=0.085,
        top=0.955,
        wspace=0.46,
        hspace=0.52,
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]

    _draw_task(axes[0])
    axes[0].set_title("Task: bounded history residual\non frozen static A/B fields", loc="left", pad=5)

    _draw_models(axes[1])
    axes[1].set_title("Models: state readability, time summary\nand joint recurrence", loc="left", pad=5)

    ax = axes[2]
    x = patient.m0_static_ab_1_45.to_numpy(float)
    y = patient.m3_joint_rnn_1_45.to_numpy(float)
    improved = y > x + 1e-9
    limits = [max(0, min(x.min(), y.min()) - 0.04), min(1, max(x.max(), y.max()) + 0.04)]
    ax.plot(limits, limits, color="#777777", lw=0.8, ls="--", zorder=1)
    ax.scatter(x[~improved], y[~improved], s=24, color=COLORS["negative"], edgecolor="white", lw=0.4, label="no gain")
    ax.scatter(x[improved], y[improved], s=24, color=COLORS["positive"], edgecolor="white", lw=0.4, label="M3 gain")
    ax.set(xlim=limits, ylim=limits, xlabel="M0 static maxAB", ylabel="M3 joint-RNN maxAB")
    primary = summary["comparisons"]["primary_m3_minus_m0"]
    ax.text(
        0.04,
        0.96,
        f"median Δ={primary['median_delta']:+.3f}\n"
        f"{primary['n_positive']} improved / {primary['n_negative']} worse / "
        f"{primary['n_tie']} ties\nP={primary['p_two_sided_exact']:.3g}",
        transform=ax.transAxes,
        va="top",
        fontsize=6.5,
    )
    ax.set_title("Primary: does M3 improve\nthe frozen static field?", loc="left", pad=5)
    _clean(ax)

    ax = axes[3]
    _paired_delta_panel(
        ax,
        patient,
        ["delta_m3_minus_m1_1_45", "delta_m3_minus_m2_1_45"],
        ["M3−M1", "M3−M2"],
        [COLORS["M1"], COLORS["M2"]],
        [summary["comparisons"]["m3_minus_m1"], summary["comparisons"]["m3_minus_m2"]],
    )
    ax.set_title("Does joint recurrence add value beyond\nfrozen state or time summary?", loc="left", pad=5)

    ax = axes[4]
    _paired_delta_panel(
        ax,
        patient,
        ["delta_true_minus_order_shuffle_1_45", "delta_correct_minus_history_swap_1_45"],
        ["True−shuffle", "Correct−swap"],
        [COLORS["M3"], "#7B3294"],
        [
            summary["comparisons"]["true_minus_order_shuffle"],
            summary["comparisons"]["correct_minus_history_swap"],
        ],
    )
    ax.set_title("Is any effect specific to true order\nand seizure-matched history?", loc="left", pad=5)

    ax = axes[5]
    columns = ["m0_static_ab_minus_channel_null_median_1_45", "m3_joint_rnn_minus_channel_null_median_1_45"]
    for row in patient.itertuples(index=False):
        values = [getattr(row, column) for column in columns]
        ax.plot([0, 1], values, color="#D1D5DB", lw=0.65, zorder=1)
    for index, (column, color, label) in enumerate(zip(columns, [COLORS["M0"], COLORS["M3"]], ["M0", "M3"])):
        values = patient[column].to_numpy(float)
        ax.scatter(np.full(len(values), index), values, s=20, color=color, edgecolor="white", lw=0.35, zorder=3)
        ax.plot([index - 0.20, index + 0.20], [np.median(values)] * 2, color="black", lw=1.3, zorder=4)
        model_name = "M0_STATIC_AB" if label == "M0" else "M3_JOINT_RNN"
        null = summary["matched_channel_null"][model_name]
        ax.text(index, ax.get_ylim()[1] if ax.get_ylim()[1] else 0, "", fontsize=1)
    sensitivity = summary["comparisons"]["sensitivity_1_150_m3_minus_m0"]
    ax.axhline(0, color="#333333", lw=0.8, ls="--")
    ax.set_xticks([0, 1], ["M0 static", "M3 joint RNN"])
    ax.set_ylabel("Observed − channel-null median")
    ax.text(
        0.02,
        0.98,
        "Above p95: M0 5/15; M3 5/15\n"
        f"1–150 Hz: M3−M0 med {sensitivity['median_delta']:+.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
    )
    ax.text(
        0.98,
        0.02,
        "Target-blind static A/B;\nretrospective full record",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.6,
        color="#555555",
    )
    ax.set_title("Absolute information relative\nto matched channel null", loc="left", pad=5)
    _clean(ax)

    for label, ax in zip("ABCDEF", axes):
        _panel_label(ax, label)
    fig.savefig(output.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def _sign_aligned_rank(candidate: np.ndarray, target: np.ndarray) -> np.ndarray:
    candidate_rank = rankdata(candidate, method="average")
    candidate_rank = (candidate_rank - candidate_rank.mean()) / max(candidate_rank.std(), 1e-12)
    target_rank = rankdata(target, method="average")
    target_rank = (target_rank - target_rank.mean()) / max(target_rank.std(), 1e-12)
    correlation = spearmanr(candidate, target).statistic
    return candidate_rank * (1 if correlation >= 0 else -1)


def _representative_figure(root: Path, patient: pd.DataFrame, output: Path) -> dict:
    median_delta = float(patient.delta_m3_minus_m0_1_45.median())
    selected = patient.iloc[(patient.delta_m3_minus_m0_1_45 - median_delta).abs().argmin()]
    subject = selected.subject
    frames = []
    for seed in (11, 29, 47):
        frames.append(
            pd.read_csv(root / "per_subject" / f"seed_{seed}" / subject / "heldout_candidate_predictions.csv.gz")
        )
    raw = pd.concat(frames, ignore_index=True)
    true = raw.loc[(raw.draw == -1) & raw.model.isin(["M0_STATIC_AB", "M3_JOINT_RNN"])]
    keys = ["subject", "seizure_id", "seizure_idx", "contact", "model"]
    ensemble = true.groupby(keys, as_index=False, sort=False).agg(
        prediction_a=("prediction_a", "mean"),
        prediction_b=("prediction_b", "mean"),
        target=("target_1_45", "first"),
    )
    seizure_deltas = []
    for seizure, group in ensemble.groupby("seizure_id", sort=False):
        scores = {}
        for model, model_group in group.groupby("model", sort=False):
            scores[model] = max(
                abs(spearmanr(model_group.prediction_a, model_group.target).statistic),
                abs(spearmanr(model_group.prediction_b, model_group.target).statistic),
            )
        seizure_deltas.append((seizure, scores["M3_JOINT_RNN"] - scores["M0_STATIC_AB"]))
    patient_median = float(np.median([value for _, value in seizure_deltas]))
    seizure = min(seizure_deltas, key=lambda item: abs(item[1] - patient_median))[0]
    group = ensemble.loc[ensemble.seizure_id == seizure]
    order = (
        raw.loc[
            (raw.seed == 11) & (raw.draw == -1) & (raw.model == "M0_STATIC_AB") & (raw.seizure_id == seizure),
            "contact",
        ]
        .drop_duplicates()
        .tolist()
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharey=True, constrained_layout=True)
    target_group = group.loc[group.model == "M0_STATIC_AB"].set_index("contact").loc[order]
    target = rankdata(target_group.target.to_numpy(), method="average")
    target = (target - target.mean()) / max(target.std(), 1e-12)
    x = np.arange(len(order))
    for branch_index, (branch, ax) in enumerate(zip(("a", "b"), axes)):
        for model, color, label, style, offset in [
            ("M0_STATIC_AB", COLORS["M0"], "Static", "--", -0.035),
            ("M3_JOINT_RNN", COLORS["M3"], "History-refined", "-", 0.035),
        ]:
            model_group = group.loc[group.model == model].set_index("contact").loc[order]
            aligned = _sign_aligned_rank(model_group[f"prediction_{branch}"].to_numpy(), model_group.target.to_numpy())
            ax.plot(
                x + offset,
                aligned,
                color=color,
                lw=1.5,
                ls=style,
                marker="o",
                ms=3,
                label=label,
            )
        ax.plot(x, target, color="black", lw=1.2, marker="s", ms=2.8, label="Early-ictal target")
        ax.set_title(f"Candidate {branch.upper()} (sign-aligned for display)", loc="left")
        ax.set_xticks(x, order, rotation=60, ha="right")
        ax.set_xlabel("Contact")
        ax.axhline(0, color="#D1D5DB", lw=0.6)
        _clean(ax)
        _panel_label(ax, "AB"[branch_index])
    axes[0].set_ylabel("Contact rank (z-scored)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)
    fig.suptitle(f"Median-effect held-out example: {subject}, seizure {seizure}", y=1.11, fontsize=8.5, fontweight="bold")
    fig.savefig(output.with_suffix(".png"), dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {
        "selection_rule": "patient closest to cohort-median M3-M0 delta; seizure closest to that patient's median delta",
        "subject": subject,
        "seizure_id": seizure,
        "patient_delta": float(selected.delta_m3_minus_m0_1_45),
        "cohort_median_delta": median_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "results/topic5_history_conditioned_field_refinement_v0_4",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    summary = json.loads((root / "HISTORY_CONDITIONED_FIELD_SUMMARY.json").read_text())
    patient = pd.read_csv(root / "history_conditioned_field_patient_metrics.csv")
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    main_stem = figures / "history_conditioned_field_refinement_six_panel"
    representative_stem = figures / "representative_history_refinement"
    _main_figure(root, patient, summary, main_stem)
    selection = _representative_figure(root, patient, representative_stem)
    figure_summary = {
        "contract": summary["contract"],
        "main_figure": str(main_stem.with_suffix(".png")),
        "representative": selection,
        "primary_endpoint": summary["primary_endpoint"],
        "static_boundary": summary["static_boundary"],
    }
    (figures / "FIGURE_SUMMARY.json").write_text(
        json.dumps(figure_summary, ensure_ascii=False, indent=2) + "\n"
    )
    readme = f"""# 图件说明

### history_conditioned_field_refinement_six_panel.png

六联图依次展示集合值任务、四个嵌套模型、M3 相对静态 M0 的患者级变化、M3 相对冻结状态 M1 与非递归 M2 的增量、真实历史相对完整顺序打乱及同患者 history-swap 的变化，以及 M0/M3 超出 matched channel null 的绝对信息。主 endpoint 固定为 clinical onset 后 0–10 s、1–45 Hz contact-energy field；1–150 Hz 仅为 no-retrain sensitivity。静态 A/B 未读取发作早期 target，但来自全记录间期事件，因此整体分析是回顾性的，不是完全前瞻预测器。

**关注点**：先看 C 的 M3−M0，再用 D/E 区分增量来自 recurrent dynamics、简单历史汇总、真实顺序还是发作匹配历史；F 检查模型是否只是在弱静态锚点附近做相对改善。

### representative_history_refinement.png

按预先固定的中位效应规则选择患者和发作，分别展示 A、B 两个候选场在静态 M0 与 history-refined M3 下的 contact rank，并与真实 early-ictal target 对照。由于正式 endpoint 使用绝对 Spearman，曲线仅为显示而逐候选做了符号对齐；它不表示模型预测了唯一传播方向。

**关注点**：观察残差修正是局部调整静态 A/B，还是把候选场整体改写；该病例只作模型行为展示，不替代 15 人统计。
"""
    (figures / "README.md").write_text(readme, encoding="utf-8")
    reproduction_path = root / "REPRODUCIBILITY_MANIFEST.json"
    if reproduction_path.exists():
        reproduction = json.loads(reproduction_path.read_text())
        for path in [
            main_stem.with_suffix(".png"),
            main_stem.with_suffix(".pdf"),
            representative_stem.with_suffix(".png"),
            representative_stem.with_suffix(".pdf"),
            figures / "FIGURE_SUMMARY.json",
            figures / "README.md",
        ]:
            reproduction["files"][str(path.relative_to(ROOT))] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        reproduction_path.write_text(
            json.dumps(reproduction, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps(figure_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
