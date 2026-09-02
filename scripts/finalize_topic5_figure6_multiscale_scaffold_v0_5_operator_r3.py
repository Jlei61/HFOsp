#!/usr/bin/env python3
"""Three-row Figure-6 review candidate with a visual perturbation story."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import finalize_topic5_figure6_multiscale_scaffold_v0_5_operator_r1 as r1  # noqa: E402
import finalize_topic5_figure6_multiscale_scaffold_v0_5_operator_r2 as r2  # noqa: E402
from scripts.paper_figures import (  # noqa: E402
    plot_topic5_figure6_lbss_full_tissue_v0_3 as tissue_plot,
)
from scripts.paper_figures import (  # noqa: E402
    plot_topic5_figure6_multiscale_scaffold_v0_5 as base,
)


DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
DEFAULT_OLD = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3"
DEFAULT_CANONICAL = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_RESPONSE = (
    ROOT / "results/topic5_latent_propagation_landscape_v0_2"
    / "spatial_control_field/patch_operator"
)
DEFAULT_FIGURE = (
    ROOT / "results/paper-ready-figure"
    / "fig6_interictal_crossstate_response_r3_candidate/figures"
)
STEM = "topic5_figure6_interictal_crossstate_response_r3_candidate"
REAL_ARMS = ("L0", "L1", "L2m", "L3")
ARM_TITLES = ("Nearby only", "Extra nearby", "Random distant", "Learned distant")
ARM_COLORS = ("#7b858b", "#6f98a7", "#b99051", base.RED)
RESPONSE_CMAP = "RdBu_r"


def paired_rank_key(name: str) -> tuple[str, int]:
    match = re.match(r"^(.*?)(\d+)$", str(name))
    return (match.group(1), int(match.group(2))) if match else (str(name), 0)


def clean_network_panel(ax: plt.Axes, old: Path, canonical: Path) -> dict:
    base.draw_full_tissue_graph(ax, old, base.FIT_ID, canonical)

    # Local structure should be visible, but not dominate the contacts.
    for line in ax.lines:
        if str(line.get_color()).lower() == "#d2d6d8":
            line.set_color("#969fa4")
            line.set_alpha(.60)
            line.set_linewidth(.56)

    # One actual fitted distant connection is enough to explain the encoding.
    # Keeping all three strongest arcs made the representative network unreadable.
    for patch in list(ax.patches)[1:]:
        patch.remove()
    for patch in ax.patches:
        patch.set_alpha(.72)
        patch.set_linewidth(.95)

    provenance = json.loads((old / "cache" / base.FIT_ID / "provenance.json").read_text())
    names = [str(value) for value in provenance["contacts"]]
    with np.load(old / "cache" / base.FIT_ID / "events.npz", allow_pickle=False) as events:
        test = np.flatnonzero(events["split"] == 2)
        example = int(test[np.flatnonzero(events["mode"][test] == 0)[0]])
        observed = np.asarray(events["ranks"][example], int)
    shown = np.full(len(observed), np.nan)
    use = (observed >= 0) & (observed <= 2)
    shown[use] = observed[use] / 2.0
    physical_order = np.asarray(sorted(range(len(names)), key=lambda index: paired_rank_key(names[index])))
    input_axis = ax.child_axes[1]
    input_axis.images[0].set_data(shown[physical_order, None])

    # Move the hidden-activity scale away from the requested upper-right legend.
    bbox = ax.get_position()
    ax.child_axes[0].set_axes_locator(None)
    ax.child_axes[0].set_position([
        bbox.x0 + .025 * bbox.width, bbox.y0 + .70 * bbox.height,
        .022 * bbox.width, .20 * bbox.height,
    ])
    handles = [
        Line2D([], [], color="#969fa4", lw=1.1, label="Nearby links"),
        Line2D([], [], color=base.RED, lw=1.3, label="One distant link"),
        Line2D([], [], marker="o", lw=0, markerfacecolor="white",
               markeredgecolor=base.DARK, markersize=5.2, label="Recorded contacts"),
    ]
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, .98),
              fontsize=6.9, frameon=False, handlelength=1.35,
              labelspacing=.25, borderpad=0)
    ax.text(-.058, .18, "3 observed\nsteps", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.8, color=base.DARK)
    ax.set_title("Recurrent network on the tissue plane", fontsize=10.4,
                 fontweight="bold", pad=5)
    with np.load(old / "per_fit" / base.FIT_ID / base.L3 / "seed0/graph.npz",
                 allow_pickle=False) as graph:
        added_edges = int(np.asarray(graph["added_mask"]).astype(bool).sum())
    return {"observed_steps_shown": 3, "distant_links_drawn": 1,
            "distant_links_in_fit": added_edges}


def effect_axis(
    ax: plt.Axes,
    groups: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    p_values: list[float],
    ylabel: str,
    title: str,
    *,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    pooled: list[float] = []
    for index, (raw, color) in enumerate(zip(groups, colors)):
        values = np.asarray(raw, float)
        values = values[np.isfinite(values)]
        pooled.extend(values.tolist())
        jitter = rng.uniform(-.13, .13, len(values))
        ax.scatter(index + jitter, values, s=22, color=color, alpha=.62,
                   edgecolor="white", lw=.25, zorder=3)
        r1.cohort_marker(ax, index, values, seed + 10 + index)
        ax.text(index, .02, f"n={len(values)}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.8, color="#666666")
    ax.axhline(0, color="#858b8e", lw=.8, ls="--", zorder=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-.52, len(labels) - .48)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.1, fontweight="bold", pad=5)
    ax.spines[["top", "right"]].set_visible(False)
    low, high = ax.get_ylim()
    span = max(high - low, 1e-9)
    ax.set_ylim(low, high + .13 * span)
    for index, p_value in enumerate(p_values):
        mark = base.stars(float(p_value)) or "n.s."
        ax.text(index, high + .018 * span, mark, ha="center", va="bottom",
                fontsize=10.5, fontweight="bold")


def draw_interictal_cohort(fig: plt.Figure, spec, out: Path) -> tuple[dict, pd.DataFrame]:
    sub = spec.subgridspec(1, 2, wspace=.35, width_ratios=(.88, 1.12))
    axes = [fig.add_subplot(sub[0, index]) for index in range(2)]

    sequence = pd.read_csv(out / "INTERICTAL_PER_PATIENT.csv").pivot(
        index="subject", columns="arm", values="test_contact_nll"
    )
    sequence_gain = (
        sequence["C_L3_ORDER_SHUFFLED"]
        - sequence["L3_LOCAL_PLUS_LEARNED_LR"]
    ).dropna()
    p_sequence = float(wilcoxon(sequence_gain, alternative="greater").pvalue)
    effect_axis(
        axes[0], [sequence_gain.to_numpy(float)], ["True event order"], [base.RED],
        [p_sequence], "Reduction in prediction error", "True order improves prediction",
        seed=6301,
    )

    fields = pd.read_csv(out / "MODEL_FIELD_PATIENT_METRICS.csv")
    fields = fields[fields.arm.eq("L3_LOCAL_PLUS_LEARNED_LR")].groupby(
        "subject", as_index=True
    ).median(numeric_only=True)
    full = fields["canonical_empirical_r"].dropna()
    start_removed = fields["seed_removed_empirical_r"].dropna()
    p_full = float(wilcoxon(full, alternative="greater").pvalue)
    p_removed = float(wilcoxon(start_removed, alternative="greater").pvalue)
    effect_axis(
        axes[1], [full.to_numpy(float), start_removed.to_numpy(float)],
        ["Whole field", "Starting contact\nremoved"], [base.BLUE, "#82a9c4"],
        [p_full, p_removed], "Generated vs observed field (Spearman r)",
        "Interictal propagation field", seed=6302,
    )
    source = pd.DataFrame({
        "patient": sequence_gain.index,
        "prediction_gain_true_order": sequence_gain.values,
    }).merge(
        pd.DataFrame({"patient": full.index, "whole_field_r": full.values,
                      "starting_contact_removed_r": start_removed.reindex(full.index).values}),
        on="patient", how="outer",
    )
    return {
        "n": int(len(sequence_gain)),
        "sequence_gain_median": float(np.median(sequence_gain)),
        "whole_field_median_r": float(np.median(full)),
        "starting_contact_removed_median_r": float(np.median(start_removed)),
    }, source


def draw_early_seizure_cohort(ax: plt.Axes, out: Path) -> tuple[dict, pd.DataFrame]:
    source = pd.read_csv(out / "early_ictal/POSTHOC_SIGN_SENSITIVITY_PER_PATIENT.csv")
    source = source[source.control.eq("all_contacts")]
    shape = source[source.orientation.eq("sign_free")].set_index("subject")["margin"]
    direction = source[source.orientation.eq("signed")].set_index("subject")["margin"]
    common = shape.index.intersection(direction.index)
    shape, direction = shape.loc[common], direction.loc[common]
    p_shape = float(wilcoxon(shape, alternative="greater").pvalue)
    p_direction = float(wilcoxon(direction, alternative="greater").pvalue)
    effect_axis(
        ax, [shape.to_numpy(float), direction.to_numpy(float)],
        ["Same shape\n(direction ignored)", "Same early-to-late\ndirection"],
        ["#6a88b7", base.RED], [p_shape, p_direction],
        "Match above shuffled contacts", "Interictal field in early seizures",
        seed=6310,
    )
    return {
        "n": int(len(common)), "shape_p": p_shape, "direction_p": p_direction,
        "shape_median_margin": float(np.median(shape)),
        "direction_median_margin": float(np.median(direction)),
    }, pd.DataFrame({"patient": common, "shape_margin": shape.values,
                     "direction_margin": direction.values})


def load_response_maps(response_root: Path, out: Path, canonical: Path):
    fit_id = base.FIT_ID
    root = response_root / "per_cell" / fit_id
    by_arm: list[np.ndarray] = []
    node_xy = None
    for arm in REAL_ARMS:
        seeds = []
        for seed in range(3):
            with np.load(root / arm / f"seed{seed}/patch_operator.npz",
                         allow_pickle=False) as handle:
                tensor = np.asarray(handle["mean_contact_operator"], float)
                node_xy = np.asarray(handle["node_xy_mm"], float)
            # Average phase and future steps 1--3 at the frozen primary dose.
            seeds.append(np.nanmean(tensor[:, :, 1, 1:4, :], axis=(0, 2)))
        by_arm.append(np.nanmedian(np.stack(seeds), axis=0))
    response = np.stack(by_arm)  # design, tissue location, future contact

    provenance = json.loads((out / "cache" / fit_id / "provenance.json").read_text())
    names = [str(value) for value in provenance["joint_contacts"]]
    with np.load(out / "cache" / fit_id / "plane.npz", allow_pickle=False) as plane:
        contacts_xy = np.asarray(plane["contacts_xy_mm"], float)
        source_nodes = np.asarray(plane["nodes_xy_mm"], float)
    centre = contacts_xy.mean(axis=0)
    location = int(np.argmin(np.linalg.norm(source_nodes - centre, axis=1)))

    record = json.loads((
        canonical / "results/interictal_propagation_masked/template_gradient_fields"
        / "per_subject" / f"{base.SUBJECT}.json"
    ).read_text())["interictal_field"]
    order = [str(value) for value in record["contact_order"]]
    take = np.asarray([names.index(name) for name in order], int)
    values = response[:, location, :][:, take]
    aligned_nodes, _ = tissue_plot._align_tissue_plane_to_frozen_display(
        source_nodes, contacts_xy, names, canonical
    )
    points, xlim, ylim = base.field_geometry(record)
    return values, aligned_nodes[location], points, xlim, ylim, location


def draw_response_maps(
    fig: plt.Figure, spec, response_root: Path, out: Path, canonical: Path,
) -> tuple[dict, pd.DataFrame]:
    sub = spec.subgridspec(1, 6, width_ratios=(.52, 1, 1, 1, 1, .055), wspace=.16)
    sketch = fig.add_subplot(sub[0, 0])
    axes = [fig.add_subplot(sub[0, index]) for index in range(1, 5)]
    cax = fig.add_subplot(sub[0, 5])
    values, location_xy, points, xlim, ylim, location = load_response_maps(
        response_root, out, canonical
    )
    limit = float(np.nanpercentile(np.abs(values), 98))
    limit = max(limit, 1e-9)

    theta = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    sketch.scatter(np.cos(theta), np.sin(theta), s=12, color="#aab2b6")
    sketch.scatter([0], [0], marker="*", s=105, color=base.RED,
                   edgecolor="white", lw=.7, zorder=4)
    sketch.annotate("", xy=(1.55, 0), xytext=(.35, 0),
                    arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": base.DARK})
    sketch.text(0, -1.38, "Same tissue\nlocation", ha="center", va="top",
                fontsize=7.2)
    sketch.set_xlim(-1.25, 1.8); sketch.set_ylim(-1.65, 1.25)
    sketch.axis("off")

    image = None
    for ax, vector, title, color in zip(axes, values, ARM_TITLES, ARM_COLORS):
        image = base.draw_field(
            ax, points, vector, np.ones_like(vector), xlim, ylim,
            cmap=RESPONSE_CMAP, vmin=-limit, vmax=limit, title=title,
            title_color=color, show_y=False,
        )
        ax.scatter([location_xy[0]], [location_xy[1]], marker="*", s=55,
                   color=base.RED, edgecolor="white", lw=.6, zorder=5)
    bar = fig.colorbar(image, cax=cax, orientation="vertical")
    bar.set_ticks([-limit, 0, limit], labels=["Less", "0", "More"])
    bar.ax.tick_params(labelsize=6.8, pad=1)
    axes[1].text(.5, 1.28, "Same nudge, four later-contact maps",
                 transform=axes[1].transAxes, ha="center", va="bottom",
                 fontsize=10.2, fontweight="bold")
    source = pd.DataFrame({
        "contact": np.tile(np.arange(values.shape[1]), values.shape[0]),
        "network_design": np.repeat(ARM_TITLES, values.shape[1]),
        "later_contact_change": values.reshape(-1),
        "tissue_location_index": location,
    })
    return {"patient": base.SUBJECT, "tissue_location_index": location,
            "network_designs": 4}, source


def draw_convergence(ax: plt.Axes, response_root: Path) -> tuple[dict, pd.DataFrame]:
    summary, convergence, _loo, _alignment = r1.load_operator_tables(response_root)
    effect = (
        convergence["real_pair_similarity_corrected"]
        - convergence["real_to_shuffled_similarity_corrected"]
    ).to_numpy(float)
    p_value = float(summary["topology_convergence"]["endpoints"]
                    ["reliability_corrected_margin"]["p_holm"])
    effect_axis(
        ax, [effect], ["True event order"], [base.RED], [p_value],
        "Extra similarity across designs", "Four designs agree", seed=6320,
    )
    return summary["topology_convergence"], pd.DataFrame({
        "patient": convergence["patient"], "extra_similarity": effect,
    })


def draw_data_link(ax: plt.Axes, response_root: Path) -> tuple[dict, pd.DataFrame]:
    summary, _convergence, _loo, alignment = r1.load_operator_tables(response_root)
    within = alignment["within_shaft_margin"].to_numpy(float)
    distance = alignment["distance_bin_margin"].to_numpy(float)
    endpoints = summary["data_link"]["endpoints"]
    effect_axis(
        ax, [within, distance], ["Within each\nelectrode", "Same contact\ndistances"],
        [base.BLUE, "#8fb3ca"],
        [float(endpoints["within_shaft_margin"]["p_holm_primary_family"]),
         float(endpoints["distance_bin_margin"]["p_one_sided"])],
        "Match above spatial controls", "Matches unseen events", seed=6330,
    )
    return summary["data_link"], alignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--response-root", type=Path, default=DEFAULT_RESPONSE)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()
    out = args.out_root.resolve()
    old = args.old_root.resolve()
    canonical = args.canonical_root.resolve()
    response_root = args.response_root.resolve()
    destination = args.figure_dir.resolve()

    required = [
        out / "INTERICTAL_PER_PATIENT.csv",
        out / "MODEL_FIELD_PATIENT_METRICS.csv",
        out / "early_ictal/POSTHOC_SIGN_SENSITIVITY_PER_PATIENT.csv",
        response_root / "PATCH_OPERATOR_SUMMARY.json",
        response_root / "per_cell" / base.FIT_ID / "L0/seed0/patch_operator.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Figure-6 R3 inputs missing: {missing}")

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.0,
        "axes.labelsize": 10.4, "axes.titlesize": 10.4,
        "xtick.labelsize": 8.7, "ytick.labelsize": 8.7,
        "axes.linewidth": .8, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(16.8, 11.2), facecolor="white")
    grid = fig.add_gridspec(
        3, 12, height_ratios=(.94, .70, .76), left=.043, right=.985,
        bottom=.06, top=.972, wspace=.80, hspace=.46,
    )

    a_stats = clean_network_panel(fig.add_subplot(grid[0, 0:3]), old, canonical)
    b_stats = r2.plain_sequence_panel(fig, grid[0, 3:9], out, old, canonical)
    c_stats = r2.draw_interictal_fields(fig, grid[0, 9:12], out, canonical)
    d_stats, d_source = draw_interictal_cohort(fig, grid[1, 0:7], out)
    e_stats, e_source = draw_early_seizure_cohort(fig.add_subplot(grid[1, 7:12]), out)
    f_stats, f_source = draw_response_maps(fig, grid[2, 0:8], response_root, out, canonical)
    g_stats, g_source = draw_convergence(fig.add_subplot(grid[2, 8:10]), response_root)
    h_stats, h_source = draw_data_link(fig.add_subplot(grid[2, 10:12]), response_root)

    cells = (
        grid[0, 0:3], grid[0, 3:9], grid[0, 9:12], grid[1, 0:7],
        grid[1, 7:12], grid[2, 0:8], grid[2, 8:10], grid[2, 10:12],
    )
    for label, cell in zip("ABCDEFGH", cells):
        base.grid_letter(fig, cell, label)

    destination.mkdir(parents=True, exist_ok=True)
    source_dir = destination / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("panel_d_interictal_cohort.csv", d_source),
        ("panel_e_early_seizure_cohort.csv", e_source),
        ("panel_f_example_response_maps.csv", f_source),
        ("panel_g_cross_design_agreement.csv", g_source),
        ("panel_h_unseen_event_match.csv", h_source),
    ):
        frame.to_csv(source_dir / name, index=False)

    stem = destination / STEM
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    assets = {path.name: r1.sha256_file(path) for path in (
        stem.with_suffix(".png"), stem.with_suffix(".pdf"), stem.with_suffix(".svg"),
    )}
    metadata = {
        "contract": "topic5_figure6_interictal_crossstate_response_r3_candidate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CANDIDATE_PENDING_USER_REVIEW",
        "panels": {
            "A": a_stats, "B": b_stats, "C": c_stats, "D": d_stats,
            "E": e_stats, "F": f_stats, "G": g_stats, "H": h_stats,
        },
        "assets_sha256": assets,
    }
    r1.write_json(destination / "FIGURE6_R3_METADATA.json", metadata)
    r1.write_json(destination / "FIGURE6_R3_COMPLETE.json", {
        "status": "COMPLETE_PENDING_USER_REVIEW", "assets_sha256": assets,
    })
    (destination / "README.md").write_text(
        "### topic5_figure6_interictal_crossstate_response_r3_candidate.png / .pdf / .svg\n\n"
        "A--C 展示代表患者的组织平面循环网络、真实与生成的间期事件，以及模型生成的两类间期"
        "传播场。A 的输入条显示前三个已观察步骤；网络图只画一条实际远距离连接，避免连线遮挡。\n\n"
        "D 在28位患者中展示真实事件顺序带来的预测改善，以及生成场与未见间期事件的对应。E 在"
        "17位患者中分别展示忽略方向的空间形状对应和保留早晚方向的对应。\n\n"
        "F 对同一组织位置施加相同小扰动，展示四种连接设计产生的实际后续触点变化图。G 将每位"
        "患者的跨设计相似性直接减去顺序被破坏的对照。H 检验这些变化是否对应未见间期事件，并"
        "保留电极归属或触点距离结构。\n\n"
        "**关注点**：上排是生成示例，中排是队列学习与跨状态结果，下排是连接方式不同但功能"
        "响应收敛的直接证据。\n"
    )
    print(json.dumps({"figure": str(stem.with_suffix('.png')), "assets": assets}, indent=2))


if __name__ == "__main__":
    main()
