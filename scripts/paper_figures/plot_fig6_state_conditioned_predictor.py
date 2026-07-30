#!/usr/bin/env python
"""Render the Figure 6 intermediate scientific result sheet."""
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

RED = "#B2182B"
BLUE = "#2166AC"
ORANGE = "#D97706"
TEAL = "#0F766E"
GREY = "#6B7280"
LIGHT = "#E5E7EB"


def latest_run(runs_root: Path):
    candidates = [p.parent for p in runs_root.glob("*/DONE.json")]
    if not candidates:
        raise FileNotFoundError(f"no completed training run under {runs_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def subject_aliases(subjects):
    return {subject: f"S{i+1:02d}" for i, subject in enumerate(sorted(subjects))}


def panel_label(ax, label):
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def draw_contract(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.4, 2.7)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((2, 1.15), 30, 0.55, color=TEAL, alpha=0.18, lw=0))
    ax.text(17, 1.43, "interictal calibration prefix", ha="center", va="center", fontsize=8)
    ax.add_patch(plt.Rectangle((48, 1.15), 25, 0.55, color=BLUE, alpha=0.18, lw=0))
    ax.text(60.5, 1.43, "event history", ha="center", va="center", fontsize=8)
    ax.add_patch(plt.Rectangle((73, 1.15), 6, 0.55, color="white", ec=GREY, hatch="//", lw=0.7))
    ax.text(76, 1.88, "5 min cutoff", ha="center", va="bottom", fontsize=7, color=GREY)
    ax.axvline(79, 0.17, 0.84, color=RED, lw=1.4)
    ax.text(79, 0.08, "EEG onset", ha="center", va="top", fontsize=7.5, color=RED)
    ax.add_patch(plt.Rectangle((79, 1.15), 10, 0.55, color=RED, alpha=0.15, lw=0))
    ax.text(84, 1.43, "1–8 Hz\n[0,10] s", ha="center", va="center", fontsize=7.5)
    ax.annotate(
        "fixed TA/TB axis",
        xy=(32, 1.42),
        xytext=(40, 2.02),
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=GREY),
        fontsize=7.5,
        color=GREY,
        ha="center",
    )
    ax.annotate(
        "event-driven\nLR-EI-CT-RNN",
        xy=(61, 1.15),
        xytext=(61, 0.25),
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=ORANGE),
        fontsize=8,
        color=ORANGE,
        ha="center",
        fontweight="bold",
    )
    ax.text(2, 2.48, "Leakage-safe task contract", fontsize=9, fontweight="bold")


def draw_gate0(ax, attr, aliases, cfg):
    passed = attr.gate0_pass.astype(str).str.lower().isin(("true", "1", "yes"))
    for ok, color, marker, label in (
        (False, GREY, "x", "excluded"),
        (True, TEAL, "o", "Gate 0 eligible"),
    ):
        sub = attr[passed == ok]
        ax.scatter(
            sub.prefix_seed_ami,
            sub.prefix_split_axis_correlation,
            c=color,
            marker=marker,
            s=30,
            label=label,
            zorder=3,
        )
        for row in sub.itertuples():
            if np.isfinite(row.prefix_seed_ami) and np.isfinite(row.prefix_split_axis_correlation):
                ax.text(
                    row.prefix_seed_ami + 0.006,
                    row.prefix_split_axis_correlation,
                    aliases.get(row.subject, ""),
                    fontsize=6,
                    va="center",
                )
    ax.axvline(float(cfg["cohort"]["calibration_min_seed_ami"]), color=GREY, ls="--", lw=0.8)
    ax.axhline(
        float(cfg["cohort"]["calibration_min_split_axis_correlation"]),
        color=GREY,
        ls="--",
        lw=0.8,
    )
    ax.set_xlabel("seed stability (AMI)")
    ax.set_ylabel("split-prefix axis stability")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax.set_title("Prefix-only axis qualification", loc="left", fontsize=9, fontweight="bold")


def draw_targets(ax, targets, aliases):
    finite = targets[np.isfinite(targets.target_low_1_8)].copy()
    order = (
        finite.groupby("subject").target_low_1_8.median().sort_values().index.tolist()
    )
    for y, subject in enumerate(order):
        values = finite.loc[finite.subject == subject, "target_low_1_8"].to_numpy()
        ax.scatter(values, np.full(len(values), y), s=11, color=BLUE, alpha=0.5)
        ax.plot(
            [np.min(values), np.max(values)],
            [y, y],
            color=BLUE,
            lw=0.7,
            alpha=0.6,
            zorder=0,
        )
        ax.scatter(np.median(values), y, s=25, color=RED, zorder=3)
    ax.axvline(0, color=GREY, lw=0.8, ls="--")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([aliases[s] for s in order], fontsize=6.5)
    ax.set_xlabel("signed early-ictal axis coefficient")
    ax.set_title("Seizure-specific 1–8 Hz readout", loc="left", fontsize=9, fontweight="bold")


def pretext_table(run_dir):
    rows = []
    for path in run_dir.glob("checkpoints/*/*/rank_*/seed_*/DONE.json"):
        record = json.loads(path.read_text())
        rows.append(
            {
                "outer_subject": record["outer_subject"],
                "rank": record["rank"],
                "seed": record["seed"],
                **record["pretext"],
            }
        )
    return pd.DataFrame(rows)


def draw_pretext(ax, pretext):
    if pretext.empty:
        ax.text(0.5, 0.5, "pretraining pending", ha="center", va="center", transform=ax.transAxes)
        return
    ranks = sorted(pretext["rank"].unique())
    for i, rank in enumerate(ranks):
        rows = pretext[pretext["rank"] == rank]
        for row in rows.itertuples():
            ax.plot(
                [i - 0.12, i + 0.12],
                [row.true_order_trained_loss, row.shuffled_order_trained_loss],
                color=LIGHT,
                lw=0.8,
                zorder=0,
            )
        ax.scatter(
            np.full(len(rows), i - 0.12),
            rows.true_order_trained_loss,
            color=TEAL,
            s=18,
            label="true order" if i == 0 else None,
        )
        ax.scatter(
            np.full(len(rows), i + 0.12),
            rows.shuffled_order_trained_loss,
            color=GREY,
            s=18,
            label="event-order shuffle" if i == 0 else None,
        )
    ax.set_xticks(range(len(ranks)), [str(r) for r in ranks])
    ax.set_xlabel("effective recurrent rank")
    ax.set_ylabel("held-out pretext loss")
    ax.set_title("Interictal temporal learning", loc="left", fontsize=9, fontweight="bold")
    ax.legend(frameon=False, fontsize=6.5)


def draw_model_performance(ax, selected):
    order = [
        "geometry_support",
        "static_scaffold",
        "last_event",
        "ab_count_imbalance",
        "ewma",
        "linear_state_space",
        "ridge_history",
        "matched_gru",
    ]
    rnn = [x for x in selected.model.unique() if x.startswith("lr_ei_ct_rnn")]
    order += sorted(rnn)
    labels = {
        "geometry_support": "geometry",
        "static_scaffold": "static",
        "last_event": "last",
        "ab_count_imbalance": "A/B\nimbalance",
        "ewma": "EWMA",
        "linear_state_space": "linear\nstate-space",
        "ridge_history": "ridge",
        "matched_gru": "GRU",
    }
    for i, model in enumerate(order):
        rows = selected[selected.model == model]
        per_subject = rows.groupby("subject").absolute_error.mean()
        color = ORANGE if model.startswith("lr_") else GREY
        jitter = np.linspace(-0.08, 0.08, max(len(per_subject), 1))
        ax.scatter(
            np.full(len(per_subject), i) + jitter[: len(per_subject)],
            per_subject,
            color=color,
            alpha=0.65,
            s=17,
        )
        if len(per_subject):
            ax.plot([i - 0.18, i + 0.18], [per_subject.median()] * 2, color="black", lw=1.4)
    ax.set_xticks(
        range(len(order)),
        [labels.get(model, f"LR-RNN\nr={int(model.rsplit('rank', 1)[1])}") for model in order],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("subject MAE")
    ax.set_title("Frozen-core held-out prediction", loc="left", fontsize=9, fontweight="bold")


def draw_increment(ax, selected):
    rnn_rows = selected[selected.model.str.startswith("lr_ei_ct_rnn")]
    baseline = selected[
        selected.model.isin(["ewma", "linear_state_space", "ridge_history"])
    ]
    values = []
    for (outer, seed), rnn in rnn_rows.groupby(["outer_subject", "seed"]):
        b = baseline[(baseline.outer_subject == outer) & (baseline.seed == seed)]
        if b.empty:
            continue
        rnn_mae = float(rnn.absolute_error.mean())
        best = min(
            float(group.absolute_error.mean()) for _, group in b.groupby("model")
        )
        values.append((outer, seed, best - rnn_mae))
    if values:
        y = np.asarray([v[2] for v in values])
        ax.scatter(np.arange(len(y)), y, c=np.where(y > 0, TEAL, GREY), s=26)
        ax.plot(np.arange(len(y)), y, color=LIGHT, lw=0.7, zorder=0)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks([])
        ax.set_ylabel("MAE improvement over\nbest non-RNN history model")
        ax.text(
            0.02,
            0.96,
            f"median Δ = {np.median(y):.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
        )
    else:
        ax.text(0.5, 0.5, "dynamic comparison pending", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Dynamic increment", loc="left", fontsize=9, fontweight="bold")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "config/topic5_state_conditioned_predictor.yaml")
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    dataset = ROOT / cfg["outputs"]["dataset"]
    runs = ROOT / cfg["outputs"]["runs"]
    run_dir = args.run_dir or latest_run(runs)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    attr = pd.read_csv(dataset / "gate0_attrition.csv")
    targets = pd.read_csv(dataset / "seizure_targets.csv")
    selected = pd.read_csv(run_dir / "selected_rank_predictions.csv")
    pretext = pretext_table(run_dir)
    aliases = subject_aliases(attr.subject.astype(str))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(7.15, 6.1), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 1.1, 1.1])
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
    draw_targets(axes[2], targets, aliases)
    draw_pretext(axes[3], pretext)
    draw_model_performance(axes[4], selected)
    draw_increment(axes[5], selected)
    for label, ax in zip("ABCDEF", axes):
        panel_label(ax, label)
        if ax.axison:
            ax.spines[["top", "right"]].set_visible(False)

    out = ROOT / cfg["outputs"]["paper_ready"] / "figures"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "fig6_state_conditioned_predictor_intermediate"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    summary = {
        "contract": cfg["contract"]["name"],
        "status": "intermediate feasibility result; not a final Figure 6 claim",
        "training_run": str(run_dir.relative_to(ROOT)),
        "n_candidate_subjects": int(len(attr)),
        "n_gate0_subjects": int(
            attr.gate0_pass.astype(str).str.lower().isin(("true", "1", "yes")).sum()
        ),
        "n_primary_targets": int(np.isfinite(targets.target_low_1_8).sum()),
        "panels": {
            "A": "frozen leakage-safe task contract",
            "B": "prefix-only stability and Gate-0 attrition",
            "C": "signed 1-8 Hz seizure-specific labels",
            "D": "held-out interictal pretext loss: true order versus shuffle",
            "E": "patient-level frozen-core LOSO MAE versus preregistered baselines",
            "F": "MAE increment over the best non-RNN history baseline",
        },
        "claim_boundary": (
            "Panel F is predictive performance, not biological causality. "
            "Mechanism claims require Gate 4 subspace stability and targeted lesions."
        ),
    }
    stem.with_name(stem.name + "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    readme = (
        "# Figure 6：state-conditioned predictor（中间结果）\n\n"
        "### fig6_state_conditioned_predictor_intermediate.png / .pdf / .svg\n\n"
        "A–C 固定并审计 calibration prefix、`[-65,-5] min` 历史窗和 EEG-onset `1–8 Hz` 有符号标签；"
        "D 检查纯间期预训练是否优于 event-order shuffle；E–F 比较冻结 recurrent core 与静态、EWMA、"
        "ridge 等预注册 baseline。当前文件是 feasibility / intermediate result，只有完成全部 LOSO、"
        "多 seed、null、Dale 确认与 mode lesion 后才能升级为最终主图。\n\n"
        "**关注点**：先看 B 的 Gate-0 分母，再看 D 的顺序增量和 F 是否在患者层面稳定高于 0；"
        "不能把单个 seed 或 pooled seizure 当独立队列证据。\n"
    )
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(stem.with_suffix(".png"), flush=True)


if __name__ == "__main__":
    main()
