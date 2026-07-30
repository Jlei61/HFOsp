#!/usr/bin/env python
"""Paper-grade intermediate Figure 6 feasibility and no-go gate audit."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_fig6_mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig6_state_conditioned_predictor import (
    BLUE,
    GREY,
    LIGHT,
    ORANGE,
    RED,
    TEAL,
    draw_contract,
    panel_label,
    subject_aliases,
)


def load_json(path):
    return json.loads(Path(path).read_text())


def draw_gate0(ax, attr, aliases, cfg):
    passed = attr.gate0_pass.astype(str).str.lower().isin(("true", "1", "yes"))
    for ok, color, marker, label in (
        (False, GREY, "x", "excluded"),
        (True, TEAL, "o", "eligible"),
    ):
        rows = attr[passed == ok]
        ax.scatter(
            rows.prefix_seed_ami,
            rows.prefix_split_axis_correlation,
            c=color,
            marker=marker,
            s=28,
            label=label,
        )
        for row in rows.itertuples():
            if np.isfinite(row.prefix_seed_ami) and np.isfinite(row.prefix_split_axis_correlation):
                if not ok:
                    ax.text(
                        row.prefix_seed_ami + 0.007,
                        row.prefix_split_axis_correlation,
                        aliases[row.subject],
                        fontsize=5.5,
                        va="center",
                    )
    ax.axvline(float(cfg["cohort"]["calibration_min_seed_ami"]), color=GREY, ls="--", lw=0.7)
    ax.axhline(
        float(cfg["cohort"]["calibration_min_split_axis_correlation"]),
        color=GREY,
        ls="--",
        lw=0.7,
    )
    ax.set(xlim=(0, 1.03), ylim=(0, 1.03), xlabel="seed stability (AMI)", ylabel="split-prefix axis stability")
    ax.legend(frameon=False, fontsize=6.3, loc="lower left")
    ax.text(
        0.98,
        0.04,
        f"{int(passed.sum())}/{len(attr)} eligible",
        transform=ax.transAxes,
        ha="right",
        fontsize=6.5,
        color=TEAL,
    )
    ax.set_title("Gate 0: prefix-only qualification", loc="left", fontsize=9, fontweight="bold")


def draw_static_gate(ax, table, aliases):
    order = table.sort_values("static_loso_r").subject.tolist()
    y = np.arange(len(order))
    lookup = table.set_index("subject")
    for i, subject in enumerate(order):
        geo = lookup.loc[subject, "geometry_support_loso_r"]
        static = lookup.loc[subject, "static_loso_r"]
        ax.plot([geo, static], [i, i], color=LIGHT, lw=1)
        ax.scatter(geo, i, color=GREY, s=18)
        ax.scatter(static, i, color=RED, s=22)
    ax.axvline(0, color="black", lw=0.7, ls="--")
    ax.set_yticks(y, [aliases[s] for s in order], fontsize=5.8)
    ax.set_xlabel("held-out patient field correlation")
    ax.scatter([], [], color=GREY, label="geometry/support")
    ax.scatter([], [], color=RED, label="TA/TB scaffold")
    ax.legend(frameon=False, fontsize=6.2, loc="lower right")
    ax.set_title("Gate 1: static scaffold readout", loc="left", fontsize=9, fontweight="bold")


def sensitivity_rows(cfg):
    roots = {
        6: ROOT / "results/topic5_state_conditioned_predictor/sensitivity/calibration_6h",
        12: ROOT / cfg["outputs"]["dataset"],
        24: ROOT / "results/topic5_state_conditioned_predictor/sensitivity/calibration_24h",
    }
    rows = []
    for hours, root in roots.items():
        attr = pd.read_csv(root / "gate0_attrition.csv")
        passed = attr.gate0_pass.astype(str).str.lower().isin(("true", "1", "yes"))
        verdict_path = root / "gate1_static_scaffold_loso/gate1_verdict.json"
        verdict = load_json(verdict_path) if verdict_path.exists() else {}
        rows.append(
            {
                "hours": hours,
                "subjects": int(passed.sum()),
                "seizures": int(attr.loc[passed, "n_primary_targets"].sum()),
                "static_r": verdict.get("static_scaffold_cohort_median_r", np.nan),
                "geometry_r": verdict.get("geometry_support_cohort_median_r", np.nan),
                "gate1_pass": verdict.get("gate1_pass", False),
            }
        )
    return pd.DataFrame(rows)


def draw_sensitivity(ax, table):
    x = np.arange(len(table))
    width = 0.34
    ax.bar(x - width / 2, table.subjects, width, color=TEAL, label="patients")
    ax.bar(x + width / 2, table.seizures, width, color=BLUE, alpha=0.75, label="seizures")
    ax.set_xticks(x, [f"{h} h" for h in table.hours])
    ax.set_ylabel("eligible count")
    ax.set_xlabel("cumulative calibration exposure")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left")
    right = ax.twinx()
    right.plot(x, table.static_r, color=RED, marker="o", lw=1.1, label="static r")
    right.plot(x, table.geometry_r, color=GREY, marker="o", lw=1.0, label="geometry r")
    right.axhline(0, color=GREY, lw=0.6, ls="--")
    right.set_ylabel("LOSO median r")
    right.legend(frameon=False, fontsize=6.0, loc="lower right")
    right.spines["top"].set_visible(False)
    ax.set_title("Calibration-duration sensitivity", loc="left", fontsize=9, fontweight="bold")


def draw_smoke(ax, record):
    true_logs = record["epoch_logs"]
    shuffle_logs = record["shuffle_control_epoch_logs"]
    ax.plot(
        [row["epoch"] for row in true_logs],
        [row["total"] for row in true_logs],
        color=TEAL,
        marker="o",
        ms=2.8,
        label="true-order training",
    )
    ax.plot(
        [row["epoch"] for row in shuffle_logs],
        [row["total"] for row in shuffle_logs],
        color=GREY,
        marker="o",
        ms=2.8,
        label="order-shuffle control",
    )
    heldout = record["pretext"]
    ax.text(
        0.98,
        0.96,
        f"held-out Δloss\n(shuffle−true) = {heldout['shuffle_minus_true']:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
    )
    ax.set(xlabel="pretraining epoch", ylabel="self-supervised loss")
    ax.legend(frameon=False, fontsize=6.2)
    ax.set_title("GPU smoke: Stage A is executable", loc="left", fontsize=9, fontweight="bold")


def draw_decision(ax, gate0, gate1):
    ax.axis("off")
    boxes = [
        (0.02, 0.62, "Gate 0\n13 patients\n41 seizures", TEAL, True),
        (0.375, 0.62, "Gate 1\nstatic\nscaffold", RED, False),
        (0.73, 0.62, "Gates 2–5\nnot entered", GREY, None),
    ]
    for x, y, text, color, status in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), 0.25, 0.23, facecolor=color, alpha=0.15, edgecolor=color, lw=1.1)
        )
        ax.text(x + 0.125, y + 0.115, text, ha="center", va="center", fontsize=5.9, fontweight="bold")
        if status is not None:
            ax.text(
                x + 0.225,
                y + 0.195,
                "✓" if status else "×",
                color=color,
                fontsize=13,
                fontweight="bold",
                ha="center",
                va="center",
            )
    ax.annotate("", xy=(0.375, 0.735), xytext=(0.27, 0.735), arrowprops=dict(arrowstyle="->", color=GREY))
    ax.annotate("", xy=(0.73, 0.735), xytext=(0.625, 0.735), arrowprops=dict(arrowstyle="-|>", color=GREY, ls="--"))
    ax.text(
        0.5,
        0.34,
        "NO-GO for formal recurrent prediction",
        ha="center",
        fontsize=10,
        color=RED,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.17,
        "Do not rescue with seed selection, larger rank, or full fine-tuning.",
        ha="center",
        fontsize=7,
        color=GREY,
    )
    ax.set_title("Pre-registered decision", loc="left", fontsize=9, fontweight="bold")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "config/topic5_state_conditioned_predictor.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    dataset = ROOT / cfg["outputs"]["dataset"]
    attr = pd.read_csv(dataset / "gate0_attrition.csv")
    aliases = subject_aliases(attr.subject.astype(str))
    static_table = pd.read_csv(dataset / "gate1_static_scaffold_loso/subject_level.csv")
    gate1 = load_json(dataset / "gate1_static_scaffold_loso/gate1_verdict.json")
    sensitivity = sensitivity_rows(cfg)
    smoke_record = load_json(
        ROOT
        / "results/topic5_state_conditioned_predictor/runs/partial_smoke_v3_20260724/"
        "checkpoints/primary/epilepsiae_1077/rank_1/seed_20260724/DONE.json"
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(7.15, 6.15), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.78, 1.12, 1.02])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]
    draw_contract(axes[0])
    draw_gate0(axes[1], attr, aliases, cfg)
    draw_static_gate(axes[2], static_table, aliases)
    draw_sensitivity(axes[3], sensitivity)
    draw_smoke(axes[4], smoke_record)
    draw_decision(axes[5], attr, gate1)
    for label, ax in zip("ABCDEF", axes):
        panel_label(ax, label)
        if ax.axison:
            ax.spines[["top", "right"]].set_visible(False)
    out = ROOT / cfg["outputs"]["paper_ready"] / "figures"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "fig6_feasibility_gate_audit_intermediate"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    summary = {
        "contract": cfg["contract"]["name"],
        "status": "pre-registered no-go at Gate 1",
        "gate0": {
            "candidate_subjects": int(len(attr)),
            "eligible_subjects": int(
                attr.gate0_pass.astype(str).str.lower().isin(("true", "1", "yes")).sum()
            ),
            "eligible_seizures": int(attr.n_primary_targets.sum()),
        },
        "gate1": gate1,
        "calibration_sensitivity": sensitivity.to_dict(orient="records"),
        "smoke_scope": "one held-out patient, rank 1, one seed; engineering feasibility only",
        "decision": "formal Gates 2-5 and full LOSO RNN training were not entered",
    }
    stem.with_name(stem.name + "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    readme = (
        "# Figure 6 feasibility gate audit\n\n"
        "### fig6_feasibility_gate_audit_intermediate.png / .pdf / .svg\n\n"
        "A 固定无泄漏任务；B 给出 prefix-only Gate 0；C 比较 nested-LOSO TA/TB 静态 scaffold 与"
        " geometry/support；D 展示 6/12/24 小时 calibration 敏感性；E 仅证明 event-driven 低秩 RNN"
        " 代码可在 GPU 上训练；F 按预注册规则在 Gate 1 停止。该图是负向 feasibility 结果，不是最终"
        "预测性能图，也不支持发作预测或机制因果表述。\n\n"
        "**关注点**：C 中静态 scaffold 未超过 geometry/support 和 rank-shuffle null，因此不能进入"
        "正式 Gates 2–5；E 的单患者 smoke 不得用于科学主张。\n"
    )
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
