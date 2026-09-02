#!/usr/bin/env python3
"""Plain-language, interictal-only Figure-6 review candidate.

The figure removes all early-seizure maps and statistics.  Two generated
interictal fields are stacked in the former cohort-statistic position, while
the cohort sequence result and two functional-response results occupy the
second row.  User-facing labels avoid internal model and analysis names.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import finalize_topic5_figure6_multiscale_scaffold_v0_5_operator_r1 as r1  # noqa: E402
import finalize_topic5_figure6_multiscale_scaffold_v0_5_r3 as r3  # noqa: E402
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
    / "fig6_interictal_recurrent_response_r2_candidate/figures"
)
STEM = "topic5_figure6_interictal_recurrent_response_r2_candidate"
GOLD = "#c99a37"
LIGHT_BLUE = "#7b9fc1"


def add_bracket(ax: plt.Axes, left: float, right: float, label: str) -> None:
    low, high = ax.get_ylim()
    span = max(high - low, 1e-9)
    y = high + .035 * span
    ax.set_ylim(low, high + .17 * span)
    ax.plot([left, left, right, right], [y - .02 * span, y, y, y - .02 * span],
            color="#252525", lw=.9, clip_on=False)
    ax.text((left + right) / 2, y + .012 * span, label, ha="center", va="bottom",
            fontsize=12, fontweight="bold")


def plain_network_panel(ax: plt.Axes, old: Path, canonical: Path):
    wrapped = r3.annotated_full_tissue_graph(base.draw_full_tissue_graph)
    wrapped(ax, old, base.FIT_ID, canonical)
    ax.set_title("How the network predicts later contacts", fontsize=10.6,
                 fontweight="bold", pad=5)
    legend = ax.get_legend()
    if legend is not None:
        replacements = (
            "Nearby links",
            "Learned distant links",
            "Recorded contacts",
        )
        for text, replacement in zip(legend.get_texts(), replacements):
            text.set_text(replacement)
    # The flanking bars and arrows already show input and output.  Removing the
    # two labels avoids a collision with panel B and keeps the mechanism panel
    # readable without a paragraph of annotations.
    for text in list(ax.texts):
        if text.get_text() in {"Input\nrank", "Generated\nrank"}:
            text.remove()
    return wrapped.stats


def plain_sequence_panel(fig: plt.Figure, spec, out: Path, old: Path, canonical: Path) -> dict:
    before = len(fig.axes)
    wrapped = r3.annotated_event_reproduction(base.draw_event_reproduction)
    wrapped(fig, spec, out, old, canonical)
    created = fig.axes[before:]
    # Four heat maps followed by the colourbar.
    created[0].set_title("Recorded events", fontsize=10.8, pad=4)
    created[1].set_title("Model output", fontsize=10.8, pad=4)
    created[0].set_ylabel("Pattern 1", color=base.RED, rotation=0, labelpad=18,
                          fontsize=10.5, fontweight="bold", va="center")
    created[2].set_ylabel("Pattern 2", color=base.BLUE, rotation=0, labelpad=18,
                          fontsize=10.5, fontweight="bold", va="center")
    created[3].set_xlabel(
        "30 unseen events per row; repeated outputs reflect identical starting contacts",
        fontsize=7.4, labelpad=3,
    )
    return wrapped.stats


def draw_interictal_fields(fig: plt.Figure, spec, out: Path, canonical: Path) -> dict:
    sub = spec.subgridspec(2, 2, width_ratios=(1, .075), hspace=.20, wspace=.08)
    axes = [fig.add_subplot(sub[row, 0]) for row in range(2)]
    cax = fig.add_subplot(sub[:, 1])
    field = json.loads((
        canonical / "results/interictal_propagation_masked/template_gradient_fields"
        / "per_subject" / f"{base.SUBJECT}.json"
    ).read_text())["interictal_field"]
    order = list(map(str, field["contact_order"]))
    with np.load(
        out / "model_fields/intact/per_patient" / base.SUBJECT
        / "L3_LOCAL_PLUS_LEARNED_LR.npz",
        allow_pickle=False,
    ) as model:
        names = model["contacts"].astype(str).tolist()
        take = np.asarray([names.index(name) for name in order])
        values = [
            1.0 - np.asarray(model["A_canonical_full"], float)[take],
            1.0 - np.asarray(model["B_canonical_full"], float)[take],
        ]
        supports = [
            np.asarray(model["A_participation"], float)[take],
            np.asarray(model["B_participation"], float)[take],
        ]
    points, xlim, ylim = base.field_geometry(field)
    image = None
    for index, (ax, value, support, color) in enumerate(
        zip(axes, values, supports, (base.RED, base.BLUE)), start=1
    ):
        image = base.draw_field(
            ax, points, value, support, xlim, ylim, cmap=base.TIMING_CMAP,
            vmin=0, vmax=1, title=f"Pattern {index}", title_color=color,
            show_y=False,
        )
        ax.set_ylabel("")
    bar = fig.colorbar(image, cax=cax, orientation="vertical")
    bar.set_ticks([0, 1], labels=["Earlier", "Later"])
    bar.ax.tick_params(labelsize=7.5, pad=1)
    axes[0].text(.5, 1.28, "Two interictal propagation patterns",
                 transform=axes[0].transAxes,
                 ha="center", va="bottom", fontsize=10.7, fontweight="bold")
    return {"patient": base.SUBJECT, "fields": 2, "seizure_values_read": False}


def draw_order_result(ax: plt.Axes, out: Path) -> tuple[dict, pd.DataFrame]:
    frame = pd.read_csv(out / "INTERICTAL_PER_PATIENT.csv")
    pivot = frame.pivot(index="subject", columns="arm", values="test_contact_nll")
    real = pivot["L3_LOCAL_PLUS_LEARNED_LR"].sort_index()
    shuffled = pivot["C_L3_ORDER_SHUFFLED"].reindex(real.index)
    difference = shuffled.to_numpy(float) - real.to_numpy(float)
    p_value = base.paired_test(difference, "greater")
    base.paired_axis(
        ax, real.to_numpy(float), shuffled.to_numpy(float),
        ("Real event\nsequence", "Later contacts\nreassigned"),
        (base.RED, base.GRAY), "Prediction error on unseen events\n(lower is better)",
        p_value,
    )
    ax.set_title("True order lowers prediction error", fontsize=10.5,
                 fontweight="bold", pad=5)
    return {
        "patients": int(len(real)), "median_improvement": float(np.median(difference)),
        "patients_improved": int(np.sum(difference > 1e-9)), "p_one_sided": float(p_value),
    }, pd.DataFrame({"patient": real.index, "real_sequence": real.values,
                     "later_contacts_reassigned": shuffled.values})


def paired_response_plot(
    ax: plt.Axes,
    control: np.ndarray,
    real: np.ndarray,
    labels: tuple[str, str],
    color: str,
    title: str,
) -> None:
    control = np.asarray(control, float)
    real = np.asarray(real, float)
    finite = np.isfinite(control) & np.isfinite(real)
    control, real = control[finite], real[finite]
    for low, high in zip(control, real):
        ax.plot([0, 1], [low, high], color=color, alpha=.28, lw=.75, zorder=1)
    ax.scatter(np.zeros(len(control)), control, s=22, color=base.GRAY, alpha=.65,
               edgecolor="white", lw=.3, zorder=3)
    ax.scatter(np.ones(len(real)), real, s=22, color=color, alpha=.68,
               edgecolor="white", lw=.3, zorder=3)
    r1.cohort_marker(ax, 0, control, 5320)
    r1.cohort_marker(ax, 1, real, 5321)
    ax.set_xticks([0, 1], labels)
    ax.set_xlim(-.42, 1.42)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=5)
    ax.text(.5, .02, f"n={len(real)}", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7.0, color="#666666")
    ax.spines[["top", "right"]].set_visible(False)
    add_bracket(ax, 0, 1, "***")


def draw_response_agreement(fig: plt.Figure, spec, response_root: Path) -> tuple[dict, pd.DataFrame]:
    summary, convergence, heldout, _alignment = r1.load_operator_tables(response_root)
    sub = spec.subgridspec(1, 2, wspace=.38)
    axes = [fig.add_subplot(sub[0, index]) for index in range(2)]
    paired_response_plot(
        axes[0], convergence["real_to_shuffled_similarity_corrected"],
        convergence["real_pair_similarity_corrected"],
        ("One model learned\nreassigned endings", "All learned the\ntrue event order"), base.RED,
        "Four network designs agree",
    )
    axes[0].set_ylabel("Similarity of predicted contact changes")
    paired_response_plot(
        axes[1], heldout["consensus_predicts_shuffled"],
        heldout["consensus_predicts_heldout_real"],
        ("Predict the reassigned-\nending model", "Predict a fourth\ntrue-order model"), base.BLUE,
        "Three designs predict the fourth",
    )
    axes[1].set_ylabel("")
    axes[0].text(
        1.00, 1.25, "Tissue nudge  →  later predictions",
        transform=axes[0].transAxes, ha="center", va="bottom",
        fontsize=7.7, color="#4f565b",
    )
    source = convergence.merge(heldout, on="patient", how="outer")
    return summary["topology_convergence"], source


def add_category_marks(ax: plt.Axes, marks: list[str]) -> None:
    low, high = ax.get_ylim()
    span = max(high - low, 1e-9)
    ax.set_ylim(low, high + .12 * span)
    for index, mark in enumerate(marks):
        ax.text(index, high + .015 * span, mark, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#252525")


def draw_unseen_event_result(ax: plt.Axes, response_root: Path) -> tuple[dict, pd.DataFrame]:
    summary, _convergence, _heldout, alignment = r1.load_operator_tables(response_root)
    series = [
        alignment["within_shaft_margin"].to_numpy(float),
        alignment["distance_bin_margin"].to_numpy(float),
        alignment["smoothing_matched_identity_margin"].to_numpy(float),
    ]
    r1.strip_axis(
        ax, series,
        ["Shuffle only within\neach electrode",
         "Keep contact\ndistances",
         "Own patient\nvs other patients"],
        [base.BLUE, LIGHT_BLUE, GOLD],
    )
    ax.set_ylabel("Extra match with unseen events")
    ax.set_title("Predicted changes match unseen events",
                 fontsize=10.5, fontweight="bold", pad=5)
    add_category_marks(ax, ["**", "**", "n.s."])
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
        out / "PIPELINE_COMPLETE.json", out / "INTERICTAL_PER_PATIENT.csv",
        response_root / "PATCH_OPERATOR_SUMMARY.json",
        response_root / "OPERATOR_TOPOLOGY_CONVERGENCE.csv",
        response_root / "OPERATOR_LEAVE_ONE_OUT_CONSENSUS.csv",
        response_root / "OPERATOR_DATA_ALIGNMENT.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"interictal Figure-6 inputs missing: {missing}")

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.labelsize": 11.0,
        "axes.titlesize": 11.0, "xtick.labelsize": 9.2, "ytick.labelsize": 9.2,
        "axes.linewidth": .8, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(16.2, 8.55), facecolor="white")
    grid = fig.add_gridspec(
        2, 12, height_ratios=(1.00, .88), left=.045, right=.985,
        bottom=.085, top=.965, wspace=.82, hspace=.45,
    )
    a_stats = plain_network_panel(fig.add_subplot(grid[0, 0:3]), old, canonical)
    b_stats = plain_sequence_panel(fig, grid[0, 3:9], out, old, canonical)
    c_stats = draw_interictal_fields(fig, grid[0, 9:12], out, canonical)
    d_stats, d_source = draw_order_result(fig.add_subplot(grid[1, 0:3]), out)
    e_stats, e_source = draw_response_agreement(fig, grid[1, 3:8], response_root)
    f_stats, f_source = draw_unseen_event_result(fig.add_subplot(grid[1, 8:12]), response_root)

    cells = (grid[0, 0:3], grid[0, 3:9], grid[0, 9:12],
             grid[1, 0:3], grid[1, 3:8], grid[1, 8:12])
    for label, cell in zip("ABCDEF", cells):
        base.grid_letter(fig, cell, label)

    destination.mkdir(parents=True, exist_ok=True)
    source = destination / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    d_source.to_csv(source / "panel_d_real_sequence_vs_reassigned_later_contacts.csv", index=False)
    e_source.to_csv(source / "panel_e_response_agreement.csv", index=False)
    f_source.to_csv(source / "panel_f_unseen_event_agreement.csv", index=False)

    stem = destination / STEM
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    assets = {path.name: r1.sha256_file(path) for path in (
        stem.with_suffix(".png"), stem.with_suffix(".pdf"), stem.with_suffix(".svg"),
    )}
    metadata = {
        "contract": "topic5_figure6_interictal_recurrent_response_r2_candidate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CANDIDATE_PENDING_USER_REVIEW",
        "early_seizure_panels_included": False,
        "panels": {
            "A": "one example recurrent network",
            "B": "recorded and generated interictal event sequences",
            "C": "two generated interictal spatial fields",
            "D": "real event sequence versus reassigned later contacts",
            "E": "agreement across differently connected networks and prediction of a held-out design",
            "F": "agreement with unseen events under spatial and cross-patient controls",
        },
        "panel_a": a_stats, "panel_b": b_stats, "panel_c": c_stats,
        "panel_d": d_stats, "panel_e": e_stats, "panel_f": f_stats,
        "claim_boundary": (
            "Real event sequences produce a reproducible future-contact response across differently "
            "connected recurrent networks, and that response agrees with unseen within-patient events "
            "after coarse spatial controls. Patient identity and necessity remain unconfirmed."
        ),
        "assets_sha256": assets,
    }
    r1.write_json(destination / "FIGURE6_R2_METADATA.json", metadata)
    r1.write_json(destination / "FIGURE6_R2_COMPLETE.json", {
        "status": "COMPLETE_PENDING_USER_REVIEW", "assets_sha256": assets,
    })
    (destination / "README.md").write_text(
        "### topic5_figure6_interictal_recurrent_response_r2_candidate.png / .pdf / .svg\n\n"
        "A 展示一个冻结循环网络如何在组织平面上接收已出现触点并预测后续触点。红线只画出实际"
        "107条远距离连接中的3条示例，不表示这些连接已被证明必要。B 展示代表患者两类真实间期事件及模型"
        "仅从起始触点生成的结果。C 只保留模型生成的两类间期空间场，不再展示发作早期场。\n\n"
        "D 是28位患者的正式顺序检验：真实事件的前后对应关系被保留时，模型对未见事件的预测"
        "误差更低。E 左图比较真实顺序训练的不同连接网络与顺序打乱对照；右图用三个网络的平均"
        "响应预测未参与平均的第四个网络。星号表示患者级单侧检验。\n\n"
        "F 比较模型预测的触点变化与未见事件的真实后续触点。保留电极归属或保留触点距离后仍有"
        "正向结果；本患者相对其他患者的优势未确认。\n\n"
        "**关注点**：本图只支持真实事件顺序产生可重复的功能响应，并与同患者未见事件一致。"
        "它不证明某组长连接、患者身份或该响应对预测是必要的。\n"
    )
    print(json.dumps({"figure": str(stem.with_suffix('.png')), "assets": assets}, indent=2))


if __name__ == "__main__":
    main()
