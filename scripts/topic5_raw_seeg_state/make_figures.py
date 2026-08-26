#!/usr/bin/env python
"""The four R0.1 figures (execution plan section 6).

    R1  model structure and data flow
    R2  forecast error vs horizon for all five predictors  (load-bearing)
    R3  representative patient: observed vs open-loop field, and mode timescales
    R4  matched state swap and state consistency

Each figure writes PNG + vector PDF + ``<name>_metadata.json`` and the script
rewrites ``figures/README.md`` in Chinese. The canvases are paper-grade and
self-contained: plain-language predictor names, real units, no internal code
names, no section references. Patients are the unit of every cohort statistic.

Example
-------
    LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:$LD_LIBRARY_PATH \
    /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
      scripts/topic5_raw_seeg_state/make_figures.py --figure all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_raw_seeg_state import contract  # noqa: E402

#: Plain-language names. Internal arm keys never reach the canvas.
ARM_LABEL = {
    "patient_mean": "this patient's average",
    "persistence": "no change from now",
    "feature_ar": "linear spectral extrapolation",
    "identity_dynamics": "state held fixed",
    "model": "evolving state",
}
ARM_COLOR = {
    "patient_mean": "#9E9E9E",
    "persistence": "#4D4D4D",
    "feature_ar": "#2166AC",
    "identity_dynamics": "#F4A582",
    "model": "#B2182B",
}
ARM_ORDER = ("patient_mean", "persistence", "feature_ar", "identity_dynamics", "model")

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth": 1.2,
    "savefig.dpi": 400,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def _save(fig, out_dir: Path, name: str, inputs, command: str, notes: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = out_dir / f"{name}.png", out_dir / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    meta = {
        "figure": name,
        "generating_command": command,
        "input_paths": [str(p) for p in inputs],
        "code_revision": contract.code_revision(),
        "package_hash": contract.package_hash(contract.r0_1_source_files()),
        "contract_version": contract.CONTRACT_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "png": str(png), "pdf": str(pdf),
        "notes": notes,
    }
    contract.atomic_write_json(out_dir / f"{name}_metadata.json", meta)
    return meta


def _load_tables(inputs_dir: Path):
    import pandas as pd

    def maybe(name):
        p = inputs_dir / name
        return (pd.read_csv(p), p) if p.exists() else (None, p)

    horizon, p_h = maybe("cohort_horizon_metrics.csv")
    swap, p_s = maybe("cohort_state_swap.csv")
    cons, p_c = maybe("cohort_consistency.csv")
    traj_path = inputs_dir / "representative_trajectory.npz"
    traj = np.load(traj_path, allow_pickle=True) if traj_path.exists() else None
    return {"horizon": horizon, "swap": swap, "consistency": cons, "trajectory": traj,
            "paths": {"horizon": p_h, "swap": p_s, "consistency": p_c,
                      "trajectory": traj_path}}


# ---------------------------------------------------------------------------
# R1 -- model structure and data flow
# ---------------------------------------------------------------------------


def _stage(ax, x, y, w, h, title, detail, colour, edge="#5A6B7A"):
    """One pipeline box: bold title line, smaller detail block underneath."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.6",
        linewidth=0.8, edgecolor=edge, facecolor=colour, zorder=2))
    cx = x + w / 2.0
    ax.text(cx, y + h - 3.4, title, ha="center", va="center",
            fontsize=7.0, fontweight="bold", color="#1F2A33", zorder=3)
    ax.text(cx, y + (h - 6.0) / 2.0, detail, ha="center", va="center",
            fontsize=6.1, color="#33414D", linespacing=1.35, zorder=3)


def _arrow(ax, p0, p1, colour="#5A6B7A", lw=0.9):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8,
                                 lw=lw, color=colour, zorder=4,
                                 shrinkA=0, shrinkB=0))


def figure_r1(out_dir: Path, command: str) -> dict:
    """What the state is built from, and what it is asked to predict.

    Redrawn twice on 2026-08-21 after visual inspection. Version 1 let every
    caption overflow its frame and stacked the dynamics and readout captions on
    top of each other. Version 2 fixed the boxes but ran the encoder-to-state
    connector diagonally across the whole canvas, through the annotation, and
    clipped the last box off the right edge. Version 3 puts the cut-off on a
    horizontal rule -- everything above it reads EEG, everything below it does
    not -- which is both the honest reading and the one that leaves no crossing
    lines.
    """
    ENC = "#DCE7F1"      # reads EEG
    STATE = "#F6DCD8"    # the state itself
    OPEN = "#FBEBD7"     # runs with no new EEG
    OUT = "#ECECEC"      # what is scored
    RULE = "#B2182B"

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9.6, 4.6))
        ax.set_xlim(0, 100)
        ax.set_ylim(-11, 53)
        ax.axis("off")

        # ---- above the rule: the encoder, the only part that reads EEG ------
        w, gap, x0, ya, h = 14.2, 2.6, 1.0, 33.0, 15.0
        row_a = [
            ("Continuous SEEG", "past 10 minutes" "\n" "256 Hz" "\n" "bipolar pairs", ENC),
            ("Waveform patches", "0.25 s each" "\n" "shared" "\n" "convolution", ENC),
            ("Within a contact", "attention over" "\n" "the 20 patches of" "\n" "a 5 s window", ENC),
            ("Across contacts", "attention over all" "\n" "contacts; position =" "\n"
                                "coordinate + shaft", ENC),
            ("Minute tokens", "12 windows pooled," "\n" "then causal attention" "\n"
                              "over the 10 minutes", ENC),
            # NOT "State": whether these 32 numbers constitute a state is the
            # thing under test, and the figure must not settle it by wording.
            ("Candidate latent code", "32 numbers", STATE),
        ]
        xs = [x0 + i * (w + gap) for i in range(len(row_a))]
        for x, (title, detail, colour) in zip(xs, row_a):
            _stage(ax, x, ya, w, h, title, detail, colour)
        for i in range(len(row_a) - 1):
            _arrow(ax, (xs[i] + w, ya + h / 2), (xs[i + 1], ya + h / 2))

        # ---- the rule ------------------------------------------------------
        y_rule = 27.0
        ax.plot([0, 100], [y_rule, y_rule], color=RULE, lw=1.0, ls=(0, (4, 2.5)), zorder=1)
        ax.text(0.5, y_rule + 1.6, "above: reads intracranial EEG", ha="left", va="bottom",
                fontsize=6.6, style="italic", color="#5A6B7A")
        ax.text(0.5, y_rule - 1.8, "below: the input is cut off", ha="left", va="top",
                fontsize=6.8, fontweight="bold", color=RULE)
        ax.text(0.5, y_rule - 5.2,
                "no new EEG is read; all four horizons are" "\n"
                "decoded from that one state, and only the" "\n"
                "elapsed time changes",
                ha="left", va="top", fontsize=6.2, color=RULE, linespacing=1.35)

        # ---- below the rule: autonomous evolution and readout ---------------
        yb, hb = 5.0, 15.0
        dx, dw = 38.0, 26.0
        px, pw = 73.0, 26.0
        _stage(ax, dx, yb, dw, hb, "Damped rotation",
               "16 two-dimensional modes;" "\n" "time constants held" "\n"
               "between 1 minute and 48 hours", OPEN)
        _stage(ax, px, yb, pw, hb, "Predicted field",
               "log power in 12 bands," "\n" "1-100 Hz," "\n" "at every contact", OUT)

        # state -> dynamics, as an elbow that crosses the rule exactly once
        sx = xs[-1] + w / 2.0
        ax.plot([sx, sx], [ya, 23.0], color="#5A6B7A", lw=0.9, zorder=4)
        ax.plot([sx, dx + dw / 2.0], [23.0, 23.0], color="#5A6B7A", lw=0.9, zorder=4)
        _arrow(ax, (dx + dw / 2.0, 23.0), (dx + dw / 2.0, yb + hb))

        _arrow(ax, (dx + dw, yb + hb / 2), (px, yb + hb / 2))
        ax.text((dx + dw + px) / 2.0, yb - 1.0, "h = 1, 5, 10, 100 min",
                ha="center", va="top", fontsize=6.3, color="#33414D")

        # ---- what goes in, and what may never go in -------------------------
        ax.text(0.5, -6.0,
                "Inputs: waveform, contact coordinates, shaft identity, artifact mask.",
                ha="left", va="top", fontsize=6.4, color="#33414D")
        ax.text(0.5, -8.8,
                "Never an input: spike marks, seizure labels, seizure-onset zone, "
                "contact rankings.",
                ha="left", va="top", fontsize=6.4, color=RULE)

        meta = _save(fig, out_dir, "r1_model_and_data_flow", [], command,
                     "Schematic only; carries no measurement and no statistic.")
    return meta


# ---------------------------------------------------------------------------
# R2 -- forecast error vs horizon (load bearing)
# ---------------------------------------------------------------------------


def figure_r2(tables, out_dir: Path, command: str) -> dict:
    frame = tables["horizon"]
    if frame is None or frame.empty:
        raise FileNotFoundError(f"missing {tables['paths']['horizon']}")
    horizons = sorted(frame["horizon_min"].unique())
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

        ax = axes[0]
        offsets = np.linspace(-0.055, 0.055, len(ARM_ORDER))
        for k, arm in enumerate(ARM_ORDER):
            sub = frame[frame["arm"] == arm]
            if sub.empty:
                continue
            med, lo, hi, xs = [], [], [], []
            for h in horizons:
                vals = sub[sub["horizon_min"] == h]["mse"].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                med.append(np.median(vals))
                lo.append(np.percentile(vals, 25))
                hi.append(np.percentile(vals, 75))
                xs.append(h * (1.0 + offsets[k]))
            ax.errorbar(xs, med, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
                        color=ARM_COLOR[arm], marker="o", ms=3.0, lw=1.2, capsize=1.8,
                        elinewidth=0.8, label=ARM_LABEL[arm],
                        zorder=3 if arm == "model" else 2)
        ax.axhline(1.0, color="#BBBBBB", lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.text(horizons[0], 1.02, "level of this patient's own average", fontsize=6.3,
                color="#888888", va="bottom", ha="left")
        ax.set_xscale("log")
        ax.set_xticks(horizons)
        ax.set_xticklabels([str(int(h)) for h in horizons])
        ax.set_xlabel("time ahead of the last observation (minutes)")
        ax.set_ylabel("error in band power\n(fraction of this patient's variance)")
        ax.set_ylim(bottom=0.0)
        ax.legend(frameon=False, loc="lower right", handlelength=1.6)

        ax = axes[1]
        pivot = frame.pivot_table(index=["subject", "horizon_min"], columns="arm",
                                  values="mse")
        baselines = [a for a in ARM_ORDER if a != "model" and a in pivot.columns]
        subjects = sorted({s for s, _ in pivot.index})
        per_subject = {}
        for subject in subjects:
            xs, ys = [], []
            for h in horizons:
                if (subject, h) not in pivot.index:
                    continue
                row = pivot.loc[(subject, h)]
                best = np.nanmin([row[a] for a in baselines]) if baselines else np.nan
                if not np.isfinite(best) or best <= 0 or not np.isfinite(row.get("model", np.nan)):
                    continue
                xs.append(h)
                ys.append(1.0 - float(row["model"]) / float(best))
            if xs:
                per_subject[subject] = (xs, ys)
                ax.plot(xs, ys, color="#C9C9C9", lw=0.7, zorder=1)
        med = []
        for h in horizons:
            vals = [y for xs, ys in per_subject.values() for x, y in zip(xs, ys) if x == h]
            med.append(np.median(vals) if vals else np.nan)
            if vals:
                ax.scatter([h] * len(vals), vals, s=7, color="#B2182B", alpha=0.45,
                           linewidths=0, zorder=2)
        ax.plot(horizons, med, color="#B2182B", lw=1.8, marker="o", ms=3.5, zorder=3,
                label=f"median of {len(per_subject)} patients")
        ax.axhline(0.0, color="#555555", lw=0.9, zorder=1)
        ax.set_xscale("log")
        ax.set_xticks(horizons)
        ax.set_xticklabels([str(int(h)) for h in horizons])
        ax.set_xlabel("time ahead of the last observation (minutes)")
        ax.set_ylabel("improvement over the best\ncomparison predictor")
        ax.legend(frameon=False, loc="best", handlelength=1.6)
        for h in horizons:
            n = sum(1 for xs, _ in per_subject.values() if h in xs)
            ax.annotate(f"n={n}", (h, ax.get_ylim()[1]), fontsize=6.0, color="#888888",
                        ha="center", va="top")
        fig.tight_layout(w_pad=2.0)
        meta = _save(fig, out_dir, "r2_forecast_error_vs_horizon",
                     [tables["paths"]["horizon"]], command,
                     "Each patient contributes one value per horizon; error bars are the "
                     "inter-quartile range across patients, never across minute windows.")
    return meta


# ---------------------------------------------------------------------------
# R3 -- representative patient
# ---------------------------------------------------------------------------


def figure_r3(tables, out_dir: Path, command: str) -> dict:
    traj = tables["trajectory"]
    if traj is None:
        raise FileNotFoundError(f"missing {tables['paths']['trajectory']}")
    h_grid = np.asarray(traj["h_grid"]).astype(int)
    observed = np.asarray(traj["observed"], dtype=float)
    predicted = np.asarray(traj["predicted"], dtype=float)
    pick = int(np.argmax(h_grid))
    edges = np.asarray(traj["freq_edges_hz"], dtype=float) if "freq_edges_hz" in traj \
        else contract.FREQ_EDGES
    centres = np.sqrt(edges[:-1] * edges[1:])
    tau = np.asarray(traj["mode_tau_minutes"], dtype=float)
    period = np.asarray(traj["mode_period_minutes"], dtype=float)
    loading = np.asarray(traj["mode_loading"], dtype=float)
    weight = loading.reshape(loading.shape[0], -1).sum(axis=1)
    weight = weight / weight.max() if weight.max() > 0 else weight

    vmax = float(np.nanpercentile(np.abs(np.concatenate([observed[pick], predicted[pick]])), 98))
    vmax = vmax if vmax > 0 else 1.0
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6),
                                 gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})
        for ax, data, title in ((axes[0], observed[pick], "measured"),
                                (axes[1], predicted[pick], "reconstructed from the state alone")):
            im = ax.pcolormesh(np.arange(data.shape[0] + 1), edges, data.T,
                               cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
            ax.set_yscale("log")
            ax.set_ylim(edges[0], edges[-1])
            ax.set_yticks([1, 3, 10, 30, 100])
            ax.set_yticklabels(["1", "3", "10", "30", "100"])
            ax.set_xlabel("recording contact")
            ax.set_title(title, pad=3)
        axes[0].set_ylabel("frequency (Hz)")
        axes[1].set_yticklabels([])
        cbar = fig.colorbar(im, ax=axes[:2].tolist(), fraction=0.035, pad=0.02)
        cbar.ax.set_title("band power\n(patient SD)", fontsize=6.3, pad=4)
        cbar.outline.set_linewidth(0.6)
        axes[0].text(0.0, 1.16, f"{int(h_grid[pick])} minutes after the input was cut off",
                     transform=axes[0].transAxes, fontsize=7.2, fontweight="bold")

        ax = axes[2]
        rotating = np.isfinite(period)
        ax.scatter(tau[~rotating], weight[~rotating], s=22, facecolor="#4D4D4D",
                   edgecolor="none", label="pure decay")
        sc = ax.scatter(tau[rotating], weight[rotating], s=22, c=period[rotating],
                        cmap="viridis", norm=matplotlib.colors.LogNorm(),
                        edgecolor="none", label="decay with rotation")
        ax.set_xscale("log")
        ax.set_xlabel("time constant of the mode (minutes)")
        ax.set_ylabel("share of the readout (largest mode = 1)")
        ax.set_xlim(0.8, 48 * 60 * 1.3)
        ax.set_ylim(0.0, 1.08)
        for minutes, text in ((60, "1 h"), (60 * 24, "1 day")):
            ax.axvline(minutes, color="#CCCCCC", lw=0.7, ls=(0, (3, 3)), zorder=0)
            ax.text(minutes, 1.06, text, fontsize=6.0, color="#999999", ha="center", va="top")
        if rotating.any():
            cb2 = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
            cb2.ax.set_title("rotation\nperiod (min)", fontsize=6.3, pad=4)
            cb2.outline.set_linewidth(0.6)
        ax.legend(frameon=False, loc="upper left", handlelength=1.0, scatterpoints=1)
        meta = _save(fig, out_dir, "r3_open_loop_field_and_modes",
                     [tables["paths"]["trajectory"]], command,
                     "One patient, one moment. Illustrative; it carries no cohort claim "
                     "and no statistical test.")
    return meta


# ---------------------------------------------------------------------------
# R4 -- matched state swap and state consistency
# ---------------------------------------------------------------------------


def figure_r4(tables, out_dir: Path, command: str) -> dict:
    swap, cons = tables["swap"], tables["consistency"]
    if swap is None or cons is None:
        raise FileNotFoundError("missing cohort_state_swap.csv or cohort_consistency.csv")
    horizons = sorted(swap["horizon_min"].unique())
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8),
                                 gridspec_kw={"width_ratios": [1.0, 1.25]})

        ax = axes[0]
        rng = np.random.default_rng(0)
        for i, h in enumerate(horizons):
            vals = swap[swap["horizon_min"] == h]["median_dmse"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            ax.scatter(i + rng.uniform(-0.13, 0.13, vals.size), vals, s=12,
                       color="#B2182B", alpha=0.55, linewidths=0, zorder=2)
            ax.hlines(np.median(vals), i - 0.28, i + 0.28, color="#B2182B", lw=2.0, zorder=3)
            ax.annotate(f"n={vals.size}", (i, ax.get_ylim()[1]), fontsize=6.0,
                        color="#888888", ha="center", va="top")
        ax.axhline(0.0, color="#555555", lw=0.9, zorder=1)
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([str(int(h)) for h in horizons])
        ax.set_xlabel("time ahead of the last observation (minutes)")
        ax.set_ylabel("extra error when the state comes from\nanother minute that looks the same")
        ax.set_title("one point per patient", fontsize=7.0, pad=3)

        ax = axes[1]
        frame = cons.dropna(subset=["e_cons_median"]).sort_values("e_cons_median")
        y = np.arange(len(frame))
        ax.hlines(y, frame["e_cons_q25"], frame["e_cons_q75"], color="#9E9E9E", lw=1.4)
        ax.scatter(frame["e_cons_median"], y, s=14, color="#2166AC", zorder=3, linewidths=0)
        ax.axvline(1.0, color="#B2182B", lw=1.0, ls=(0, (4, 3)))
        ax.text(1.02, len(frame) - 0.5, "state re-read one minute later is\n"
                                        "as far away as the step itself",
                fontsize=6.2, color="#B2182B", va="top", ha="left")
        ax.set_yticks(y)
        ax.set_yticklabels(frame["subject"].tolist(), fontsize=5.6)
        ax.set_xlabel("mismatch between the re-read state and the predicted one\n"
                      "(relative to how far the state moved)")
        ax.set_xlim(left=0.0)
        ax.set_ylim(-0.8, len(frame) - 0.2)
        fig.tight_layout(w_pad=2.2)
        meta = _save(fig, out_dir, "r4_state_swap_and_consistency",
                     [tables["paths"]["swap"], tables["paths"]["consistency"]], command,
                     "Left panel: patients are the unit. Right panel: one row per patient, "
                     "bar is the inter-quartile range over that patient's minutes.")
    return meta


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

README_SECTIONS = {
    "r1_model_and_data_flow": (
        "这张图画的是信息怎么从原始颅内脑电走到未来的频谱预测：先把 10 分钟波形切成 0.25 秒小块，"
        "在触点内、触点间各做一次注意力，压成每分钟一个摘要，再压成 32 个数字的状态；"
        "红色虚线以下就不再读任何新脑电了，只让这 32 个数字按固定的衰减+旋转往前走，"
        "解码出 1 / 5 / 10 / 100 分钟后每个触点、每个频段的能量。"
        "看虚线下面那一层——四个时间点共用同一个解码器，差别只在走了多久。\n"
        "**关注点**：它只是结构示意，不含任何测量结果，也不能用来判断模型好不好。"),
    "r2_forecast_error_vs_horizon": (
        "这是本轮承重的图。左边是四个时间尺度上五种预测方式的误差，纵轴已按该患者自己的方差归一，"
        "所以 1.0 那条虚线就是“跟这个人平时的平均水平一样”；误差棒是**跨患者**的四分位距，不是跨分钟窗。"
        "右边把同一患者内的模型误差和它自己最强的那个对照相除，一条细线一个患者，0 以上表示模型更好。"
        "看两件事：左边模型线在哪个 horizon 开始离开对照，右边有多少患者的线稳定在 0 以上。\n"
        "**关注点**：它只能说“在这个时间尺度上有没有超出三种什么都不懂的对照”，"
        "不能读成癫痫易感性、发作预测，也与 100 Hz 以上的活动无关。"),
    "r3_open_loop_field_and_modes": (
        "代表患者的一个时刻：左、中两块是同一时刻真实测到的、和只靠状态推出来的触点×频段能量场，"
        "共用同一个色标（红=高于该患者均值，蓝=低于）；右边一块是 16 个二维模态各自的时间常数和它在解码器里占的比重，"
        "颜色表示旋转周期，灰点是不旋转的纯衰减模态。"
        "看左中两块的图案是否对得上，以及右边比重大的模态落在几分钟还是几小时的量程上。\n"
        "**关注点**：单患者单时刻的示例，不是队列证据；模态的时间常数只描述读出的时间尺度，"
        "不代表兴奋性、抑制性或任何生理机制。"),
    "r4_state_swap_and_consistency": (
        "左边回答“状态里有没有超出此刻快照的信息”：在同一患者内找一个当前频谱场几乎一样、但相隔两小时以上的时刻，"
        "把状态换成那一刻的再解码，纵轴是换掉之后多出来的误差，一个点一个患者，0 以上说明状态确实还记着别的东西。"
        "右边回答另一个独立问题：一分钟后编码器自己编出来的状态，跟从上一分钟推一步得到的状态差多远（已用状态本身的位移量归一）；"
        "红虚线 1.0 是分界，明显小于 1 才说明两者在同一条轨迹上。\n"
        "**关注点**：左右两块是两层独立结论，不得合并成一句总评；右边接近或超过 1 时，"
        "左边的正结果只能称为“可外推的潜在编码”，不能称为“统一的可演化状态”。"),
}


def write_readme(out_dir: Path, produced) -> Path:
    lines = [
        "# Raw-SEEG 可演化预测状态模型 — 图说明",
        "",
        "本目录每张图都同时有 PNG、矢量 PDF 和 `<name>_metadata.json`（生成命令 / 输入路径 / code revision / 时间戳）。",
        "所有队列统计以**患者**为单位，分钟窗口不当作独立生物学样本。",
        "",
    ]
    for name in produced:
        section = README_SECTIONS.get(name)
        if section is None:
            continue
        lines.append(f"### {name}.png")
        lines.append("")
        lines.append(section)
        lines.append("")
    path = out_dir / "README.md"
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figure", default="all", choices=["R1", "R2", "R3", "R4", "all"])
    p.add_argument("--inputs-dir", default=None,
                   help="directory holding the cohort CSVs and representative_trajectory.npz")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--skip-missing", action="store_true",
                   help="skip a figure whose inputs are absent instead of failing")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    inputs_dir = Path(args.inputs_dir) if args.inputs_dir else contract.RESULT_ROOT
    out_dir = Path(args.out_dir) if args.out_dir else contract.FIGURE_DIR
    command = "python scripts/topic5_raw_seeg_state/make_figures.py " + " ".join(
        sys.argv[1:] if argv is None else list(argv))
    tables = _load_tables(inputs_dir)

    wanted = ["R1", "R2", "R3", "R4"] if args.figure == "all" else [args.figure]
    makers = {"R1": lambda: figure_r1(out_dir, command),
              "R2": lambda: figure_r2(tables, out_dir, command),
              "R3": lambda: figure_r3(tables, out_dir, command),
              "R4": lambda: figure_r4(tables, out_dir, command)}
    produced, skipped = [], []
    for key in wanted:
        try:
            produced.append(makers[key]()["figure"])
        except FileNotFoundError as exc:
            if not args.skip_missing:
                raise
            skipped.append(f"{key}: {exc}")
    existing = [p.stem for p in sorted(out_dir.glob("*.png"))]
    readme = write_readme(out_dir, [n for n in existing if n in README_SECTIONS])
    print(json.dumps({"produced": produced, "skipped": skipped,
                      "readme": str(readme), "out_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
