#!/usr/bin/env python3
"""Render the frozen Topic 5 history-RNN result without changing inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_COLOR = "#4C566A"
EPILEPSIAE = "#B2182B"
YUQUAN = "#2166AC"
HISTORY = "#A35E48"
NULL = "#9B9B9B"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_label(axis, label: str) -> None:
    axis.text(
        -0.16,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def _diagram(axis) -> None:
    axis.axis("off")
    boxes = [
        (0.01, 0.48, 0.20, 0.23, "one-event\nrank field"),
        (0.27, 0.48, 0.20, 0.23, "EventRNN\n$u_e$; reset"),
        (0.53, 0.48, 0.20, 0.23, "history state\n$z(t)$\n+ IEI decay"),
        (0.79, 0.48, 0.20, 0.23, "next-event\ncontact field"),
    ]
    for x, y, width, height, label in boxes:
        axis.add_patch(
            plt.Rectangle(
                (x, y), width, height, transform=axis.transAxes,
                facecolor="#F4F1EC", edgecolor=ROOT_COLOR, linewidth=1.0,
            )
        )
        axis.text(
            x + width / 2, y + height / 2, label,
            ha="center", va="center", fontsize=7.2,
        )
    arrows = [
        ((0.21, 0.605), (0.27, 0.605)),
        ((0.47, 0.605), (0.53, 0.605)),
        ((0.73, 0.605), (0.79, 0.605)),
    ]
    for start, stop in arrows:
        axis.annotate(
            "", xy=stop, xytext=start, xycoords=axis.transAxes,
            arrowprops={"arrowstyle": "->", "color": ROOT_COLOR, "lw": 1.2},
        )
    axis.text(
        0.63, 0.82, "candidate state is carried with real-time decay",
        ha="center", transform=axis.transAxes, color=ROOT_COLOR, fontsize=7.5,
    )
    axis.text(
        0.01, 0.12,
        "v0.1: target-blind next-event proxy   |   early-ictal target not evaluated",
        transform=axis.transAxes, fontweight="bold", fontsize=7.3,
    )
    axis.text(
        0.01, 0.02, "Event reset vs continuous-segment history state",
        transform=axis.transAxes, color=ROOT_COLOR, fontsize=7.2,
    )
    axis.set_title("Causal cross-event state model", loc="left")


def _patient_contrast(axis, frame: pd.DataFrame, column: str, title: str, ylabel: str) -> None:
    datasets = [name for name in ("epilepsiae", "yuquan") if name in set(frame.dataset)]
    colors = {"epilepsiae": EPILEPSIAE, "yuquan": YUQUAN}
    rng = np.random.default_rng(20260801)
    for position, dataset in enumerate(datasets):
        values = frame.loc[frame.dataset == dataset, column].to_numpy(float)
        jitter = rng.uniform(-0.13, 0.13, len(values))
        axis.scatter(
            position + jitter, values, s=18, facecolor=colors[dataset],
            edgecolor="white", linewidth=0.35, alpha=0.82, zorder=2,
        )
        median = float(np.median(values))
        axis.plot([position - 0.22, position + 0.22], [median, median], color="black", lw=1.8, zorder=3)
    axis.axhline(0.0, color=NULL, ls="--", lw=0.9)
    axis.set_xticks(range(len(datasets)), ["Epilepsiae", "Yuquan"])
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left")


def _stat_annotation(axis, median: float, p_value: float, positive: int, total: int) -> None:
    lower, upper = axis.get_ylim()
    span = max(upper - lower, 1e-12)
    axis.set_ylim(lower, upper + 0.28 * span)
    # Never round a small p to 0.000; the effects here live at 1e-4 nats, so the
    # median needs enough digits to stay readable.
    p_text = f"p={p_value:.3f}" if p_value >= 1e-3 else f"p={p_value:.1e}"
    axis.text(
        0.98, 0.98,
        f"median={median:+.5f}\n{positive}/{total} positive\none-sided {p_text}",
        transform=axis.transAxes, ha="right", va="top", color=ROOT_COLOR,
        fontsize=7.2,
    )


def _locked(axis, title: str, message: str) -> None:
    axis.axis("off")
    axis.add_patch(
        plt.Rectangle(
            (0.05, 0.12), 0.9, 0.72, transform=axis.transAxes,
            facecolor="#F2F2F2", edgecolor="#B8B8B8", linewidth=0.9,
        )
    )
    axis.text(0.5, 0.55, "LOCKED", transform=axis.transAxes, ha="center", va="center", color="#777777", fontweight="bold")
    axis.text(0.5, 0.36, message, transform=axis.transAxes, ha="center", va="center", color="#777777", wrap=True)
    axis.set_title(title, loc="left")


def _g2_models(axis, input_root: Path) -> None:
    patient = pd.read_csv(input_root / "g2_patient_metrics.csv")
    x = np.arange(3)
    for row in patient.itertuples(index=False):
        axis.plot(x, [row.rho_M0, row.rho_M1, row.rho_M2], color="#B8B8B8", lw=0.6, alpha=0.7)
    medians = [patient.rho_M0.median(), patient.rho_M1.median(), patient.rho_M2.median()]
    axis.plot(x, medians, color=HISTORY, lw=2.2, marker="o", ms=4, zorder=3)
    axis.set_xticks(x, ["M0\nstatic", "M1\nunordered", "M2\nchronological"])
    axis.set_ylabel("held-out Spearman $\\rho$")
    axis.set_title("Early-ictal field prediction", loc="left")


def _g3(axis, input_root: Path) -> None:
    pairing = pd.read_csv(input_root / "g3_patient_pairing_metrics.csv")
    for row in pairing.itertuples(index=False):
        axis.plot([0, 1], [row.wrong_rho, row.correct_rho], color="#B8B8B8", lw=0.8, zorder=1)
        axis.scatter([0, 1], [row.wrong_rho, row.correct_rho], c=[NULL, HISTORY], s=18, edgecolor="white", linewidth=0.3, zorder=2)
    axis.set_xticks([0, 1], ["wrong state", "correct state"])
    axis.set_ylabel("held-out Spearman $\\rho$")
    axis.set_title("State-seizure specificity", loc="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("results/topic5_history_rnn_early_ictal_field"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    formal = root / "g1_sequential_formal_v0_1"
    multi_summary = formal / "G1_MULTI_SEED_SUMMARY.json"
    seed_summary = formal / "seed_20260725" / "G1_SUMMARY.json"
    if multi_summary.exists():
        gate = json.loads(multi_summary.read_text())
        multiseed_metrics = formal / "g1_multiseed_patient_metrics.csv"
        if multiseed_metrics.exists():
            metrics_path = multiseed_metrics
        elif gate.get("status") == "G1_SEED1_FAIL_KEEP_ICTAL_TARGET_SEALED":
            metrics_path = formal / "seed_20260725" / "g1_patient_metrics.csv"
        else:
            raise RuntimeError("multi-seed inference is incomplete")
        primary_inference = {
            "n_patients": int(gate["n_primary_patients"]),
            "median_chronological_increment": float(
                gate["median_chronological_increment"]
            ),
            "chronological_increment_one_sided_wilcoxon_p": float(
                gate["chronological_increment_one_sided_wilcoxon_p"]
            ),
            "n_chronological_positive": int(gate["n_chronological_positive"]),
            "median_order_shuffle_cost": float(
                gate["median_prefix_matched_order_shuffle_cost"]
            ),
            "order_shuffle_cost_one_sided_wilcoxon_p": float(
                gate["prefix_matched_order_shuffle_one_sided_wilcoxon_p"]
            ),
            "n_order_shuffle_positive": int(
                gate["n_prefix_matched_order_shuffle_positive"]
            ),
        }
    elif seed_summary.exists():
        gate = json.loads(seed_summary.read_text())
        if gate.get("status") == "G1_PASS_OPEN_G2":
            raise RuntimeError(
                "G1 seed 1 passed but multi-seed confirmation is incomplete"
            )
        metrics_path = formal / "seed_20260725" / "g1_patient_metrics.csv"
        primary_inference = gate["primary"]
    else:
        raise RuntimeError("G1 inference is incomplete; do not render a partial figure")
    frame = pd.read_csv(metrics_path)
    primary = frame.loc[
        ~frame.subject.isin(
            ["epilepsiae_1073", "epilepsiae_1146", "yuquan_chenziyang"]
        )
    ].copy()
    g2_root = root / "g2_early_ictal_loso_v0_1"
    g2_summary_path = g2_root / "G2_G3_SUMMARY.json"
    g2 = json.loads(g2_summary_path.read_text()) if g2_summary_path.exists() else None
    if gate.get("status") == "G1_MULTI_SEED_PASS_OPEN_G2" and g2 is None:
        raise RuntimeError("G1 passed but G2/G3 inference is incomplete")

    _style()
    if g2 is None:
        figure, row_axes = plt.subplots(
            1, 3, figsize=(10.6, 3.25), constrained_layout=True,
            gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]},
        )
        axes = np.asarray(row_axes).reshape(1, 3)
    else:
        figure, axes = plt.subplots(
            2, 3, figsize=(10.2, 5.7), constrained_layout=True,
            gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]},
        )
    _diagram(axes[0, 0])
    _patient_contrast(
        axes[0, 1], primary, "chronological_increment",
        "Chronological-state increment",
        "BCE(M1) - BCE(M2)",
    )
    _stat_annotation(
        axes[0, 1],
        primary_inference["median_chronological_increment"],
        primary_inference["chronological_increment_one_sided_wilcoxon_p"],
        primary_inference["n_chronological_positive"],
        primary_inference["n_patients"],
    )
    _patient_contrast(
        axes[0, 2], primary, "prefix_matched_order_shuffle_cost",
        "Causal-prefix order control",
        "BCE(shuffle) - BCE(true)",
    )
    _stat_annotation(
        axes[0, 2],
        primary_inference["median_order_shuffle_cost"],
        primary_inference["order_shuffle_cost_one_sided_wilcoxon_p"],
        primary_inference["n_order_shuffle_positive"],
        primary_inference["n_patients"],
    )
    if g2 is not None:
        _g2_models(axes[1, 0], g2_root)
        patient = pd.read_csv(g2_root / "g2_patient_metrics.csv")
        primary_g2 = patient.loc[patient.subject != "epilepsiae_1146"].copy()
        primary_g2["dataset"] = "epilepsiae"
        _patient_contrast(
            axes[1, 1], primary_g2, "rho_increment_M2_minus_M1",
            "Chronological-state increment",
            "$\\rho$(M2) - $\\rho$(M1)",
        )
        _g3(axes[1, 2], g2_root)
    for label, axis in zip("ABCDEF", axes.flat):
        _panel_label(axis, label)
    figure.suptitle(
        (
            "Target-blind next-event evaluation of an inter-event history model"
            if g2 is None
            else "Time-accumulated interictal history and early-ictal spatial fields"
        ),
        x=0.01, ha="left", fontsize=11, fontweight="bold",
    )
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "topic5_history_rnn_early_ictal_field_v0_1"
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "contract": "topic5_history_rnn_early_ictal_field_v0_1",
        "g1_status": gate["status"],
        "g2_status": g2["status"] if g2 is not None else "LOCKED_NOT_RUN",
        "partial_results_rendered": False,
        "n_g1_primary_patients": int(len(primary)),
        "n_panels": int(axes.size),
        "g1_primary": primary_inference,
        "g1_dataset_median_chronological_increment": gate.get(
            "dataset_median_chronological_increment",
            {
                key: value["median_chronological_increment"]
                for key, value in gate.get("primary", {}).get(
                    "dataset_direction", {}
                ).items()
            },
        ),
        "g2_g3": g2,
    }
    (stem.with_suffix(".json")).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if g2 is None:
        readme = f"""### topic5_history_rnn_early_ictal_field_v0_1.png

这是 v0.1 next-event proxy 的三联暂定阴性图。Panel A 定义候选计算图：单事件编码器在每场事件内重置，而候选 HistoryRNN state 按真实事件间隔跨事件传递。Panel B 是预注册主对比——带真实时间顺序的模型，是否比看到同一批事件、同一个 last event、同一份静态先验但不知道顺序的模型，更能预测下一场间期事件。Panel C 换一个问法——把同一段因果前缀里较早事件的先后打乱（事件集合、时间槽、last event 全部不动），带顺序的模型自己会不会变差。

Panel B 与 Panel C 回答的不是同一个问题，必须分开读：Panel C 为正只说明这个 state 确实依赖输入顺序（也可能只是它没见过乱序输入），Panel B 才是"顺序是否带来了无序模型拿不到的信息"，而 Panel B 为零。因此不能用 Panel C 反推顺序有预测价值。

图中不绘制 early-ictal target，因为 v0.1 合同曾把 G1 设为硬门。该空缺是旧停止规则的结果，不是 early-ictal 阴性结果；direct transfer 已转入独立 v0.2 合同。

**关注点**：3-seed patient-first 的 M2−M1 中位为 {primary_inference['median_chronological_increment']:+.6f}（{primary_inference['n_chronological_positive']}/{primary_inference['n_patients']} 人为正，单侧 p={primary_inference['chronological_increment_one_sided_wilcoxon_p']:.4g}），严格顺序置换代价中位为 {primary_inference['median_order_shuffle_cost']:+.6f}（{primary_inference['n_order_shuffle_positive']}/{primary_inference['n_patients']} 人为正，单侧 p={primary_inference['order_shuffle_cost_one_sided_wilcoxon_p']:.4g}）。该结果只约束 next-event chronology proxy；不能把当前 state 命名为已验证的跨事件病理状态，也不能据此否定 direct early-ictal transfer。
"""
    else:
        readme = f"""### topic5_history_rnn_early_ictal_field_v0_1.png

Panel A 定义真正的计算问题：单事件编码器在每场事件内重置，而 HistoryRNN 按真实事件间隔跨事件持续。Panel B 检验 chronological state 是否比使用相同事件集合、last event 和静态信息的无序模型更能预测下一场间期事件；Panel C 用完全相同 causal prefix 的顺序置换排除集合信息捷径。

Panel D–F 只有在 G1 预注册门通过后才显示 early-ictal target 结果，依次回答 latent state 能否预测发作早期接触点能量场、是否超过 M1，以及正确 state–seizure 配对是否优于同患者错误配对。若 G1 未通过，这三块明确显示为 locked，不能据此讨论 early-ictal 阴性或阳性。

**关注点**：G1 状态为 `{gate['status']}`，G2/G3 状态为 `{g2['status']}`。主结论由 patient-first 的 M2−M1、严格顺序对照和 gated early-ictal LOSO 共同决定；图中不把工程完成、状态低维或单病例表现当作跨状态机制证据。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
