#!/usr/bin/env python3
"""Figure 6: source-conditioned shared-scaffold RNN, interictal to early ictal.

Panels follow the frozen A--E order.  Every number drawn here is read back
from a frozen artifact; nothing is recomputed from the ictal target, and the
representative patient is fixed to ``epilepsiae_1146`` before any target
value was unsealed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_shared_scaffold_rnn_unit_v0_2 import (  # noqa: E402
    load_one_patient_record,
)
from src.topic5_patient_specific_rnn_bridge import (  # noqa: E402
    chronological_60_20_20,
)

REPRESENTATIVE = "epilepsiae_1146"
MODEL_ORDER = ("static", "ordinary_gru", "structured", "structured_rank_shuffle")
MODEL_LABEL = {
    "static": "Static",
    "ordinary_gru": "Ordinary GRU",
    "structured": "Structured",
    "structured_rank_shuffle": "Structured,\nshuffled order",
}
MODEL_COLOR = {
    "static": "#9E9E9E",
    "ordinary_gru": "#4D908E",
    "structured": "#B2182B",
    "structured_rank_shuffle": "#D6A0A6",
}
SIDE_LABEL = {"minus": "Source end 1", "plus": "Source end 2"}

plt.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "savefig.bbox": "tight",
    }
)


# --------------------------------------------------------------- data access
def load_representative(output: Path, dataset_root: Path):
    """Frozen structured fields plus the patient's own observed test events."""

    freeze = output / "field_freeze" / "per_subject" / REPRESENTATIVE
    with np.load(freeze / "structured_fields.npz", allow_pickle=False) as data:
        frozen = {key: np.asarray(data[key]) for key in data.files}
    source_definition = json.loads((freeze / "source_definition.json").read_text())
    record = load_one_patient_record(dataset_root, REPRESENTATIVE)
    _, _, test20 = chronological_60_20_20(record)
    names = np.asarray(frozen["contact_names"]).astype(str)
    if not np.array_equal(names, np.asarray(record.contact_names).astype(str)):
        raise RuntimeError("frozen field and dataset contact order differ")
    return frozen, source_definition, record, test20, names


def observed_arrival_by_side(record, test20, frozen):
    """Split held-out events by which frozen source end they actually started on.

    Grouping uses only the observed first rank set, never an empirical A/B
    label and never the ictal target.
    """

    groups = np.asarray(record.group_ids, dtype=np.int64)[np.asarray(test20)]
    horizon = int(frozen["horizon"])
    pools = {
        side: np.asarray(frozen[f"source_{side}_indices"], dtype=int)
        for side in ("minus", "plus")
    }
    out = {}
    for side, own in pools.items():
        other = pools["plus" if side == "minus" else "minus"]
        first = groups == 0
        overlap_own = first[:, own].sum(axis=1)
        overlap_other = first[:, other].sum(axis=1)
        selected = groups[overlap_own > overlap_other]
        arrival = np.zeros((horizon, groups.shape[1]), dtype=float)
        if len(selected):
            for step in range(1, horizon + 1):
                arrival[step - 1] = (selected == step).mean(axis=0)
        out[side] = {"arrival": arrival, "n_events": int(len(selected))}
    return out


def median_early_ictal_field(readout: dict, names: np.ndarray):
    """Patient-median early-ictal broadband field on the exact joined contacts."""

    target_root = Path(readout["target_cache_root"]).resolve()
    files = sorted((target_root / f"outer_{REPRESENTATIVE}").glob(f"{REPRESENTATIVE}__*.npz"))
    stacked, joined = [], None
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            target_names = np.asarray(data["contact_names"]).astype(str)
            values = np.asarray(data[str(readout["target_key"])], dtype=float)
        lookup = dict(zip(target_names, values))
        present = [name for name in names if name in lookup]
        if joined is None:
            joined = present
        elif present != joined:
            raise RuntimeError("seizures disagree on the joined contact set")
        stacked.append([lookup[name] for name in present])
    if joined is None:
        raise RuntimeError(f"no early-ictal files for {REPRESENTATIVE}")
    return np.asarray(joined), np.median(np.asarray(stacked, dtype=float), axis=0)


# ------------------------------------------------------------------ panel A
def panel_a(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, text, face, edge="#333333"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.008",
                linewidth=0.7, facecolor=face, edgecolor=edge, zorder=2,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=6.3, zorder=3)

    def arrow(x0, y0, x1, y1, color="#333333", style="-|>"):
        ax.add_patch(
            FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                            mutation_scale=7, linewidth=0.7, color=color, zorder=1)
        )

    top, mid, low, h = 0.72, 0.42, 0.12, 0.17
    c1, c2, c3, c4 = 0.005, 0.255, 0.515, 0.775
    w = 0.205

    box(c1, top, w, h, "Contacts seen at\nthis rank step", "#EDEDED")
    box(c1, mid, w, h, "One learned\nsigned axis", "#FFF3E0")
    box(c1, low, w, h, "First rank set\nsets the sign", "#FCE4EC")

    box(c2, top, w, h, "Symmetric\nscaffold", "#E3F2FD")
    box(c2, mid - 0.06, w, h, "Signed flow\n(same two ends)", "#E3F2FD")

    box(c3, top, w, h, "Propagation\nscaffold + flow", "#E8F5E9")
    box(c3, mid - 0.06, w, h, "Restraint\nscaffold only", "#E8F5E9")

    box(c4, (top + mid) / 2 - 0.02, w, h, "Next contact,\nstop, set size", "#EDEDED")

    arrow(c1 + w, top + h / 2, c2, top + h / 2)
    arrow(c1 + w, mid + h / 2, c2, top + 0.03)
    arrow(c1 + w, mid + h / 2, c2, mid - 0.06 + h / 2)
    arrow(c1 + w, low + h / 2, c2, mid - 0.06 + 0.03, color="#B2182B")
    arrow(c1 + w / 2, top, c1 + w / 2, low + h)

    arrow(c2 + w, top + h / 2, c3, top + h / 2)
    arrow(c2 + w, top + 0.03, c3, mid - 0.06 + h / 2)
    arrow(c2 + w, mid - 0.06 + h / 2, c3, top + 0.03, color="#B2182B")

    arrow(c3 + w, top + h / 2, c4, (top + mid) / 2 + h / 2 - 0.04)
    arrow(c3 + w, mid - 0.06 + h / 2, c4, (top + mid) / 2 + 0.02)

    ax.text(0.0, -0.12,
            "One scaffold; its sign flips with the observed start.\n"
            "Patient-specific, no empirical direction labels.",
            ha="left", va="top", fontsize=6.0, color="#555555")
    ax.set_title("Structured propagation model", loc="left", fontweight="bold")


# ------------------------------------------------------------------ panel B
def panel_b(axes, cbar_ax, frozen, observed, names, coordinate):
    order = np.argsort(coordinate)
    horizon = int(frozen["horizon"])
    images = []
    for row, side in enumerate(("minus", "plus")):
        rollout = np.asarray(frozen[f"first_arrival_mass_{side}"], dtype=float)
        panels = (
            ("Observed held-out events", observed[side]["arrival"]),
            ("Model rollout", rollout),
        )
        for column, (title, matrix) in enumerate(panels):
            ax = axes[row][column]
            image = ax.imshow(
                matrix[:, order].T, aspect="auto", origin="lower",
                cmap="viridis", vmin=0.0, vmax=max(0.05, float(matrix.max())),
                extent=(0.5, horizon + 0.5, -0.5, len(order) - 0.5),
            )
            images.append(image)
            ax.set_xticks([1, horizon])
            ax.set_xticklabels(["First", "Last"])
            if column == 0:
                ax.set_ylabel(SIDE_LABEL[side], fontsize=6.5)
                ax.set_yticks([0, len(order) - 1])
                ax.set_yticklabels(["end 1", "end 2"])
            else:
                ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
            if row == 1:
                ax.set_xlabel("Rank step")
            if column == 0:
                ax.text(0.97, 0.05, f"n={observed[side]['n_events']}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=5.8, color="white")
    bar = plt.colorbar(images[0], cax=cbar_ax)
    bar.set_label("Arrival probability", fontsize=6.3)
    bar.ax.tick_params(labelsize=5.8)
    axes[0][0].annotate(
        "Same patient, same model", xy=(0.0, 1.30), xycoords="axes fraction",
        fontsize=7.5, fontweight="bold", annotation_clip=False,
    )


# ------------------------------------------------------------------ panel C
def panel_c(ax, patient: pd.DataFrame, stats: dict):
    models = [m for m in MODEL_ORDER if m in set(patient.model)]
    wide = patient.pivot(index="subject", columns="model", values="contact_nll")
    wide = wide.dropna(subset=models)
    positions = np.arange(len(models), dtype=float)
    rng = np.random.default_rng(4)
    for _, row in wide.iterrows():
        ax.plot(positions, [row[m] for m in models], color="#BBBBBB",
                linewidth=0.35, alpha=0.6, zorder=1)
    for index, model in enumerate(models):
        values = wide[model].to_numpy(float)
        jitter = rng.uniform(-0.07, 0.07, size=len(values))
        ax.scatter(np.full(len(values), positions[index]) + jitter, values, s=6,
                   color=MODEL_COLOR[model], edgecolor="none", zorder=3)
        ax.hlines(np.median(values), positions[index] - 0.22, positions[index] + 0.22,
                  color="#222222", linewidth=1.1, zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_LABEL[m] for m in models])
    ax.set_ylabel("Held-out next-contact NLL\n(lower is better)")
    ax.set_xlim(-0.45, len(models) - 0.55)
    ax.set_title(
        f"Held-out interictal prediction ({len(wide)} patients)",
        loc="left", fontweight="bold",
    )
    lines = []
    for comparator in ("ordinary_gru", "static", "structured_rank_shuffle"):
        entry = stats["comparisons"].get(f"structured_vs_{comparator}__contact_nll", {})
        if entry.get("status") != "COMPLETE":
            continue
        lines.append(
            f"vs {MODEL_LABEL[comparator].replace(chr(10), ' ')}: "
            f"median {entry['median_delta']:+.4f} "
            f"[{entry['bootstrap_95ci'][0]:+.4f}, {entry['bootstrap_95ci'][1]:+.4f}], "
            f"P={entry['wilcoxon_two_sided_p']:.3g}, "
            f"{entry['n_positive']}+/{entry['n_negative']}-/{entry['n_tied']}="
        )
    ax.text(0.015, -0.40, "Structured minus comparator, positive favours structured\n" + "\n".join(lines),
            transform=ax.transAxes, fontsize=5.9, va="top", ha="left", color="#333333")


# ------------------------------------------------------------------ panel D
def panel_d(axes, cbar_axes, frozen, names, coordinate, joined_names, ictal):
    order = np.argsort(coordinate)
    ordered_names = names[order]
    rows = [
        ("Model field, start at end 1", np.asarray(frozen["field_minus"], float)[order], "viridis_r"),
        ("Model field, start at end 2", np.asarray(frozen["field_plus"], float)[order], "viridis_r"),
    ]
    lookup = dict(zip(joined_names, ictal))
    ictal_row = np.asarray(
        [lookup.get(name, np.nan) for name in ordered_names], dtype=float
    )
    rows.append(("Early-ictal broadband power", ictal_row, "magma_r"))
    handles = []
    for ax, (title, values, cmap) in zip(axes, rows):
        finite = np.isfinite(values)
        image = ax.imshow(
            values[None, :], aspect="auto", cmap=cmap,
            vmin=np.nanmin(values[finite]), vmax=np.nanmax(values[finite]),
            extent=(-0.5, len(values) - 0.5, 0, 1),
        )
        handles.append(image)
        ax.set_yticks([])
        ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=6.2)
        ax.set_xticks([0, len(values) - 1])
        ax.set_xticklabels(["end 1", "end 2"])
        if title != rows[-1][0]:
            ax.set_xticklabels([])
    axes[-1].set_xlabel("Contacts ordered along the learned axis")
    first = plt.colorbar(handles[0], cax=cbar_axes[0])
    first.set_label("Model arrival\nLast → First", fontsize=6.0)
    first.set_ticks([])
    second = plt.colorbar(handles[2], cax=cbar_axes[1])
    second.set_label("Power\nlow → high", fontsize=6.0)
    second.set_ticks([])
    axes[0].set_title(
        "Frozen model fields and the same patient's early-ictal field",
        loc="left", fontweight="bold",
    )


# ------------------------------------------------------------------ panel E
def panel_e(ax, inset, patient: pd.DataFrame, cohort: dict, supportive: str):
    models = [m for m in ("static", "ordinary_gru", "structured") if m in set(patient.model)]
    positions = np.arange(len(models), dtype=float)
    for mode, target_ax in (("all_contact", ax), ("within_shaft", inset)):
        column = f"{mode}_margin"
        wide = patient.pivot(index="subject", columns="model", values=column)
        wide = wide.dropna(subset=models)
        primary = wide.drop(index=supportive, errors="ignore")
        for _, row in primary.iterrows():
            target_ax.plot(positions, [row[m] for m in models], color="#BBBBBB",
                           linewidth=0.4, alpha=0.7, zorder=1)
        for index, model in enumerate(models):
            values = primary[model].to_numpy(float)
            target_ax.scatter(np.full(len(values), positions[index]), values, s=9,
                              color=MODEL_COLOR[model], edgecolor="none", zorder=3)
            target_ax.hlines(np.median(values), positions[index] - 0.2, positions[index] + 0.2,
                             color="#222222", linewidth=1.1, zorder=4)
        if supportive in wide.index:
            target_ax.plot(positions, [wide.loc[supportive, m] for m in models],
                           color="#B2182B", linewidth=0.6, linestyle="--", zorder=2)
            target_ax.scatter(positions, [wide.loc[supportive, m] for m in models],
                              s=14, facecolor="none", edgecolor="#B2182B",
                              linewidth=0.8, zorder=5)
        target_ax.axhline(0.0, color="#666666", linewidth=0.5, linestyle=":")
        target_ax.set_xticks(positions)
        target_ax.set_xlim(-0.4, len(models) - 0.6)
    ax.set_xticklabels([MODEL_LABEL[m].replace("\n", " ") for m in models])
    ax.set_ylabel("Correspondence above its own\nshuffled-contact null")
    inset.set_xticklabels([MODEL_LABEL[m].split()[0] for m in models], fontsize=5.2)
    inset.set_title("Within-shaft null", fontsize=5.8)
    inset.tick_params(labelsize=5.2)

    n_primary = cohort["model_statistics"]["structured"]["n_primary_patients"]
    ax.set_title(
        f"Interictal model field vs early-ictal field ({n_primary} patients)",
        loc="left", fontweight="bold",
    )
    paired = cohort["paired_comparisons"]["structured_vs_ordinary_all_contact"]
    per_model = cohort["model_statistics"]
    lines = [
        "Structured minus ordinary: "
        f"median {paired['median_delta']:+.4f} "
        f"[{paired['bootstrap_95ci'][0]:+.4f}, {paired['bootstrap_95ci'][1]:+.4f}], "
        f"P={paired['exact_wilcoxon_greater_p']:.3g}, "
        f"{paired['n_positive']}+/{paired['n_negative']}-/{paired['n_tied']}=",
    ]
    for model in models:
        entry = per_model[model]["all_contact"]
        lines.append(
            f"{MODEL_LABEL[model].replace(chr(10),' ')}: median margin "
            f"{entry['median_margin']:+.4f}, P={entry['exact_wilcoxon_greater_p']:.3g}, "
            f"{entry['n_exceeds_patient_p95']}/{n_primary} above own 95th percentile"
        )
    lines.append("Open red marker: supportive patient, excluded from every P value")
    ax.text(0.015, -0.34, "\n".join(lines), transform=ax.transAxes,
            fontsize=5.9, va="top", ha="left", color="#333333")


# --------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--readout-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    training = yaml.safe_load((ROOT / args.training_config).read_text())
    readout = yaml.safe_load((ROOT / args.readout_config).read_text())
    output = (ROOT / args.output_root).resolve()
    dataset_root = Path(training["dataset_artifact_root"]).resolve() / training["dataset_root"]

    patient_interictal = pd.read_csv(output / "interictal_patient_metrics.csv")
    interictal_stats = json.loads((output / "interictal_cohort_statistics.json").read_text())
    patient_ictal = pd.read_csv(output / "early_ictal" / "patient_scores.csv")
    ictal_stats = json.loads((output / "early_ictal" / "cohort_statistics.json").read_text())

    frozen, source_definition, record, test20, names = load_representative(output, dataset_root)
    coordinate = np.asarray(frozen["diffusion_coordinate"], dtype=float)
    observed = observed_arrival_by_side(record, test20, frozen)
    joined_names, ictal_field = median_early_ictal_field(readout, names)

    figure = plt.figure(figsize=(7.2, 8.4))
    grid = figure.add_gridspec(
        4, 6, height_ratios=[1.15, 1.35, 1.25, 1.35],
        hspace=0.95, wspace=0.75,
        left=0.11, right=0.965, top=0.965, bottom=0.055,
    )

    a_ax = figure.add_subplot(grid[0, 0:3])
    panel_a(a_ax)

    b_axes = [
        [figure.add_subplot(grid[0, 3]), figure.add_subplot(grid[0, 4])],
        [figure.add_subplot(grid[1, 3]), figure.add_subplot(grid[1, 4])],
    ]
    b_cbar = figure.add_subplot(grid[0:2, 5])
    b_cbar.set_position([0.905, 0.60, 0.012, 0.20])
    panel_b(b_axes, b_cbar, frozen, observed, names, coordinate)

    c_ax = figure.add_subplot(grid[1, 0:3])
    panel_c(c_ax, patient_interictal, interictal_stats)

    d_axes = [figure.add_subplot(grid[2, i]) for i in range(3)]
    for index, ax in enumerate(d_axes):
        ax.set_position([0.30, 0.375 - index * 0.036, 0.46, 0.028])
    d_cbars = [figure.add_axes([0.79, 0.339, 0.011, 0.064]),
               figure.add_axes([0.865, 0.339, 0.011, 0.028])]
    panel_d(d_axes, d_cbars, frozen, names, coordinate, joined_names, ictal_field)

    e_ax = figure.add_subplot(grid[3, 0:4])
    e_inset = figure.add_axes([0.795, 0.155, 0.155, 0.10])
    panel_e(e_ax, e_inset, patient_ictal, ictal_stats, str(readout["supportive_subject"]))

    for label, axis, offset in (("a", a_ax, (-0.06, 1.06)), ("b", b_axes[0][0], (-0.55, 1.30)),
                                ("c", c_ax, (-0.13, 1.10)), ("d", d_axes[0], (-0.44, 1.70)),
                                ("e", e_ax, (-0.13, 1.10))):
        axis.annotate(label, xy=offset, xycoords="axes fraction",
                      fontsize=10, fontweight="bold", annotation_clip=False)

    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = figures_dir / "topic5_figure6_source_conditioned_rnn"
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".svg"))
    figure.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(figure)

    # ---------------------------------------------------------- source data
    patient_interictal.to_csv(figures_dir / "figure6_panelC_source_data.csv", index=False)
    patient_ictal.to_csv(figures_dir / "figure6_panelE_source_data.csv", index=False)
    order = np.argsort(coordinate)
    pd.DataFrame(
        {
            "contact": names[order],
            "learned_axis_coordinate": coordinate[order],
            "model_field_end1": np.asarray(frozen["field_minus"], float)[order],
            "model_field_end2": np.asarray(frozen["field_plus"], float)[order],
            "early_ictal_median": [
                dict(zip(joined_names, ictal_field)).get(name, np.nan)
                for name in names[order]
            ],
        }
    ).to_csv(figures_dir / "figure6_panelD_source_data.csv", index=False)

    statistics = {
        "representative_subject": REPRESENTATIVE,
        "representative_fixed_before_target_unseal": True,
        "source_pool_rule": source_definition["source_pool_rule"],
        "panel_b_event_counts": {
            side: observed[side]["n_events"] for side in ("minus", "plus")
        },
        "panel_c": interictal_stats,
        "panel_e": ictal_stats,
        "not_implemented": [
            "rollout-vs-test20 participation / pairwise precedence / expected-rank "
            "distance consistency statistics are not computed by any current "
            "script, so panel C shows held-out likelihood and accuracy only"
        ],
    }
    (figures_dir / "figure6_statistics.json").write_text(
        json.dumps(statistics, indent=2, allow_nan=False, default=float) + "\n"
    )

    (figures_dir / "README.md").write_text(
        "# Figure 6 图说明\n\n"
        "### topic5_figure6_source_conditioned_rnn.png / .pdf / .svg\n\n"
        "五个面板讲一条链：模型长什么样、它能不能复现同一患者观察到的两种相反传播、"
        "全队列预测得分、冻结下来的两张方向场与该患者发作早期能量场并排、"
        "以及跨状态对应在整个主分析队列上的统计。\n\n"
        "a 是结构示意：每位患者只学一条有方向的轴，同一对端点归属同时生成"
        "对称支架和有符号的流；每场事件由最先放电的那几个触点定下方向，"
        "方向在该事件内不再变。b 固定用 "
        f"{REPRESENTATIVE}，上下两行是模型自己认定的两个起点端，"
        "左列是留出集里真实观察到的到达时间分布，右列是模型自主推演的到达分布，"
        "触点顺序只按冻结的学习轴排。c 是全队列留出集下一个触点的预测难度"
        "（越低越好），四个模型同患者连线。d 把两张冻结的模型场和同一患者"
        "发作早期宽带能量场放在同一套触点排布上，两个方向都必须画出来。"
        "e 是主分析患者的跨状态对应强度，已减去各自的随机重排基线；"
        "空心红点是辅助患者，不进入任何 P 值。\n\n"
        "**关注点**：b 左右两列的亮带走向是否一致、且上下两行方向相反；"
        "c 中结构化模型与打乱顺序对照之间是否分开；"
        "d 中两张模型场是否真的相反而不是几乎重合；"
        "e 中结构化是否高于随机基线，以及它与普通模型的配对差是否跨过零。\n"
    )
    print(json.dumps({"status": "COMPLETE", "figure": str(stem.with_suffix(".png"))}))


if __name__ == "__main__":
    main()
