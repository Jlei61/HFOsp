#!/usr/bin/env python3
"""Figure B -- H1 generator evidence.

asset_id: epi_prssm_generator_evidence
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.topic5_epi_prssm.contracts import OUTPUT_ROOT, code_revision, package_hash
from src.topic5_epi_prssm.figure_style import (
    COLOR, DOUBLE_COLUMN_MM, FS_AXIS, FS_TICK, FS_TITLE, arm_color, arm_label,
    unmapped_arms,
    LW_INDIVIDUAL, LW_MAIN, LW_REFERENCE, figure, panel_letter, save_asset, zero_line,
)

ASSET = "epi_prssm_generator_evidence"
LADDER = OUTPUT_ROOT / "generator_ladder"
FIG_ROOT = OUTPUT_ROOT / "figures"


def _read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a table that a stage may legitimately have left empty."""
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        pass
    return pd.DataFrame(columns=columns or [])

SHOW = ["frozen_state_node", "ct_ewma_g0", "g1_graph_clds", "g2_graph_gru_ode", "g3_resource"]


def _dataset_colour(dataset: str) -> str:
    return COLOR["epilepsiae"] if dataset == "epilepsiae" else COLOR["yuquan"]


def panel_a(ax, inventory: pd.DataFrame):
    ax.set_title("Support per patient", fontsize=FS_TITLE, loc="left", pad=5)
    frame = inventory.sort_values("n_events").reset_index(drop=True)
    y = np.arange(len(frame))
    for i, row in frame.iterrows():
        ax.plot([row["n_train"], row["n_events"]], [i, i], color="#DDDDDD", lw=1.4, zorder=1)
        ax.plot(row["n_train"], i, "o", ms=2.6, color=_dataset_colour(row["dataset"]), zorder=3)
        ax.plot(row["n_events"], i, "|", ms=4.0, color="#9A9A9A", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("events (train dot, total tick)")
    ax.set_yticks([])
    ax.set_ylabel(f"{len(frame)} patients")
    ax.plot([], [], "o", ms=2.6, color=COLOR["epilepsiae"], label="Epilepsiae")
    ax.plot([], [], "o", ms=2.6, color=COLOR["yuquan"], label="Yuquan")
    ax.legend(loc="lower right", handlelength=0.8, borderpad=0.2)


def panel_b(ax, variance: pd.DataFrame):
    ax.set_title("Fixed versus dynamic variance", fontsize=FS_TITLE, loc="left", pad=5)
    frame = variance[variance.status == "ok"].sort_values("dynamic_share_mean")
    y = np.arange(len(frame))
    ax.barh(y, 1.0 - frame["dynamic_share_mean"], color=COLOR["scaffold"], height=0.72,
            label=r"fixed $\mu_p$")
    ax.barh(y, frame["dynamic_share_mean"], left=1.0 - frame["dynamic_share_mean"],
            color=COLOR["G2"], height=0.72, label="dynamic residual")
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.0, len(frame) + 4.5)
    ax.set_xlabel("share of participation variance")
    ax.axvline(1.0 - float(frame["dynamic_share_mean"].median()), color="#333333",
               lw=LW_REFERENCE, ls=(0, (3, 2)))
    ax.text(0.02, 0.5,
            f"median dynamic share {float(frame['dynamic_share_mean'].median()):.3f}",
            transform=ax.transAxes, fontsize=FS_TICK - 1.0, color="#FFFFFF",
            ha="left", va="center")
    ax.legend(loc="upper left", ncol=2, handlelength=0.9, borderpad=0.2,
              columnspacing=1.0, fontsize=FS_TICK - 0.6)


def panel_c(ax, open_loop: pd.DataFrame):
    ax.set_title("Open-loop horizon", fontsize=FS_TITLE, loc="left", pad=5)
    cohort = open_loop[(open_loop.subject == "__cohort__") & (open_loop.endpoint == "event_nll")]
    reference_arm = "frozen_state_node" if (cohort.arm == "frozen_state_node").any() else "static"
    reference = cohort[cohort.arm == reference_arm].set_index("horizon")["delta_vs_static"]
    horizons = sorted(cohort.horizon.unique())
    for arm in SHOW:
        rows = cohort[cohort.arm == arm].set_index("horizon")
        if rows.empty:
            continue
        values = [rows["delta_vs_static"].get(h, np.nan) - reference.get(h, 0.0) for h in horizons]
        low = [rows["ci_low"].get(h, np.nan) - reference.get(h, 0.0) for h in horizons]
        high = [rows["ci_high"].get(h, np.nan) - reference.get(h, 0.0) for h in horizons]
        ax.plot(horizons, values, "-o", ms=3.0, lw=LW_MAIN, color=arm_color(arm),
                label=arm_label(arm))
        ax.fill_between(horizons, low, high, color=arm_color(arm), alpha=0.13, lw=0)
    zero_line(ax)
    ax.set_xscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
    ax.set_xlabel("events after the anchor")
    ax.set_ylabel("Δ event NLL (nats/event)")
    ax.text(0.98, 0.97, "arm colours: see E", transform=ax.transAxes,
            fontsize=FS_TICK - 1.2, ha="right", va="top", color="#8A8A8A")


def panel_d(ax, effects: pd.DataFrame, per_patient: pd.DataFrame):
    ax.set_title("Ladder increments", fontsize=FS_TITLE, loc="left", pad=14)
    steps = [("ct_ewma_g0", "frozen_state_node"), ("g1_graph_clds", "ct_ewma_g0"),
             ("g2_graph_gru_ode", "g1_graph_clds"), ("g3_resource", "g2_graph_gru_ode"),
             ("g3_resource_on_g1", "g1_graph_clds"),
             ("g1_graph_clds", "nuisance_timing_baseline")]
    labels = ["G0 −\nfrozen", "G1 −\nG0", "G2 −\nG1", "G3 −\nG2",
              "G1+r −\nG1", "G1 −\ntiming"]
    wide = per_patient[per_patient.endpoint == "event_nll"].pivot_table(
        index="subject", columns="arm", values="value")
    rng = np.random.default_rng(7)
    for i, ((better, worse), label) in enumerate(zip(steps, labels)):
        if better not in wide or worse not in wide:
            continue
        delta = (wide[better] - wide[worse]).dropna()
        jitter = rng.normal(0, 0.055, len(delta))
        ax.plot(i + jitter, delta.to_numpy(), "o", ms=2.4, color=arm_color(better),
                alpha=0.65, mew=0, zorder=3)
        row = effects[(effects.endpoint == "event_nll")
                      & (effects.contrast == f"{better} - {worse}")]
        if not row.empty:
            median = float(row.median_delta.iloc[0])
            ax.plot([i - 0.26, i + 0.26], [median, median], color="#1A1A1A", lw=1.6, zorder=4)
            ax.plot([i, i], [float(row.ci_low.iloc[0]), float(row.ci_high.iloc[0])],
                    color="#1A1A1A", lw=1.0, zorder=4)
            ax.text(i, 1.01, f"{int(row.n_favourable.iloc[0])}/{int(row.n_patients.iloc[0])}",
                    transform=ax.get_xaxis_transform(),
                    fontsize=FS_TICK - 0.8, ha="center", va="bottom", color="#333333")
    zero_line(ax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=FS_TICK - 1.2)
    ax.set_ylabel("Δ held-out event NLL")
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.axvline(3.5, color="#BBBBBB", lw=0.7, ls=(0, (2, 2)))
    low_y, high_y = ax.get_ylim()
    ax.set_ylim(low_y - 0.06 * (high_y - low_y), high_y)
    ax.text(3.62, 0.02, "gate", transform=ax.get_xaxis_transform(),
            fontsize=FS_TICK - 1.2, ha="left", va="bottom", color="#777777")


def panel_e(ax, reset: pd.DataFrame, runs: pd.DataFrame):
    ax.set_title("State reset recovery", fontsize=FS_TITLE, loc="left", pad=5)
    for arm in SHOW:
        rows = reset[reset.arm == arm]
        if rows.empty:
            continue
        grouped = rows.groupby(["horizon", "subject"]).reset_penalty_nll.median().reset_index()
        curve = grouped.groupby("horizon").reset_penalty_nll.median()
        ax.plot(curve.index, curve.to_numpy(), "-o", ms=2.6, lw=LW_MAIN,
                color=arm_color(arm), label=arm_label(arm))
    zero_line(ax)
    ax.set_xscale("log")
    ax.set_xlabel("events observed after the wipe")
    ax.set_ylabel("NLL penalty of a wiped state")
    ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
    ax.legend(loc="best", handlelength=1.1, borderpad=0.2, labelspacing=0.25,
              fontsize=FS_TICK - 0.6, title="arms (shared by C, E, F)",
              title_fontsize=FS_TICK - 0.8)


def panel_f(ax, runs: pd.DataFrame):
    ax.set_title("Correction budget", fontsize=FS_TITLE, loc="left", pad=5)
    frame = runs[runs.status == "COMPLETE"]
    for arm in SHOW:
        rows = frame[frame.arm == arm]
        if rows.empty:
            continue
        ax.plot(rows["correction_energy"], rows["diag_generator_tau_median_seconds"],
                "o", ms=4.0, color=arm_color(arm), label=arm_label(arm), mew=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("correction energy per event")
    ax.set_ylabel("generator time constant (s)")
    ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
    for seconds, name in ((60, "1 min"), (3600, "1 h"), (86400, "1 day")):
        ax.axhline(seconds, color="#D0D0D0", lw=LW_REFERENCE, zorder=0)
        ax.text(ax.get_xlim()[0], seconds, f" {name}", fontsize=FS_TICK - 1.0,
                va="bottom", color="#8A8A8A")
    ax.text(0.98, 0.97, "arm colours: see E", transform=ax.transAxes,
            fontsize=FS_TICK - 1.2, ha="right", va="top", color="#8A8A8A")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all34")
    args = parser.parse_args()

    inventory = _read_csv(OUTPUT_ROOT / "data_audit/support_inventory.csv")
    variance = _read_csv(OUTPUT_ROOT / "baseline/patient_repertoire_variance.csv")
    runs = _read_csv(LADDER / "model_runs.csv")
    effects = _read_csv(LADDER / "patient_effects.csv",
                        ["endpoint", "contrast", "n_patients", "median_delta", "ci_low",
                         "ci_high", "n_favourable", "sign_test_p", "wilcoxon_p"])
    per_patient = _read_csv(LADDER / "patient_filtered_scores.csv")
    open_loop = _read_csv(LADDER / "open_loop_horizon.csv")
    reset = _read_csv(LADDER / "state_reset.csv")
    card = json.loads((LADDER / "GENERATOR_EVIDENCE_CARD.json").read_text())

    fig, axes = figure(DOUBLE_COLUMN_MM, 128.0, nrows=2, ncols=3)
    fig.subplots_adjust(left=0.070, right=0.985, top=0.895, bottom=0.095, wspace=0.42, hspace=0.52)
    panel_a(axes[0, 0], inventory)
    panel_b(axes[0, 1], variance)
    panel_c(axes[0, 2], open_loop)
    panel_d(axes[1, 0], effects, per_patient)
    panel_e(axes[1, 1], reset, runs)
    panel_f(axes[1, 2], runs)
    for ax, letter in zip(axes.ravel(), "ABCDEF"):
        panel_letter(ax, letter, dx=-0.21, dy=1.20)

    files = save_asset(fig, ASSET, FIG_ROOT, metadata={
        "asset_id": ASSET, "provisional_role": "Figure B",
        "hypothesis": "H1", "status": "EXPLORATORY", "split": "development validation partition",
        "supported_layer": card["supported_layer"], "verdict": card["verdict"],
        "observable_timing_gate": card.get("observable_timing_gate"),
        "observable_timing_gate_meaning": card.get("observable_timing_gate_meaning"),
        "denominators": card["denominators"], "ladder_notes": card["ladder_notes"],
        "primary_reference_arm": "frozen_state_node (strictest capacity match: node-resolved "
                                 "but time-constant state; every adapter parameter present)",
        "colour_mapping": {k: arm_color(k) for k in SHOW},
        "arms_without_assigned_colour": unmapped_arms(),
        "claim_boundary": card["claim_boundary"],
        "code_revision": code_revision(), "package_hash": package_hash(),
    }, readme_entries=[{
        "filename": f"{ASSET}.png",
        "body": "H1 的证据卡。A 给出每位患者的事件分母（点=训练段，刻度=总量），"
                "B 把每位患者的参与度方差拆成固定 repertoire 与动态残差两块，"
                "C 是关闭 observer 后 5/10/20/40 个事件的开环预测差（相对容量配平的冻结状态臂），"
                "D 是阶梯上逐患者的成对增量，粗横线是中位数、竖线是自助 95% 区间、"
                "顶部数字是方向有利的患者数比总患者数，E 是把状态清零后随观测恢复的曲线，"
                "F 把 observer 校正能量与生成器时间常数放在一起，用来看 observer 是否吞掉了生成器。",
        "focus": "D 里虚线右边那两列是承重的：一个是资源锚加在最优递归族上有没有增量，"
                 "另一个是这套慢状态有没有胜过「只用可观测时间量」搭的基线——"
                 "后者不过，前面几级就只能读成把放电密度换了个说法。",
    }])
    print(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()
