#!/usr/bin/env python3
"""Build the paper-ready v2.4 axis/readout closeout figure."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/topic5_rnn_axis_positive_static_transfer_v2_4"
OUT = (
    ROOT
    / "results/paper-ready-figure/"
    "fig6_rnn_axis_static_transfer_v2_4/figures"
)
EXAMPLE_SUBJECT = "epilepsiae_1084"

COLORS = {
    "empirical": "#007C91",
    "full": "#B2182B",
    "axis": "#D6604D",
    "source": "#8073AC",
    "transition": "#4D4D4D",
    "rnn": "#B2182B",
    "null": "#9E9E9E",
    "node": "#BDBDBD",
    "no_history": "#969696",
    "isotropic": "#636363",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 6.3,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        linespacing=1.25,
        fontsize=fontsize,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color="#555555",
            transform=ax.transAxes,
        )
    )


def draw_workflow(ax: plt.Axes) -> None:
    ax.set_axis_off()
    panel_label(ax, "A")
    ax.set_title("Frozen analysis chain", loc="left", pad=8)
    box(
        ax,
        (0.01, 0.35),
        0.205,
        0.34,
        "Interictal\nrank sets",
        facecolor="#E8F1F5",
        edgecolor="#3C7D91",
    )
    box(
        ax,
        (0.265, 0.35),
        0.205,
        0.34,
        "Self-supervised\nnext-contact\nprediction",
        facecolor="#F5ECEA",
        edgecolor=COLORS["full"],
    )
    box(
        ax,
        (0.52, 0.35),
        0.205,
        0.34,
        "Frozen node\nrank\ndistribution",
        facecolor="#F5ECEA",
        edgecolor=COLORS["full"],
    )
    box(
        ax,
        (0.775, 0.35),
        0.205,
        0.34,
        "Early-ictal\nenergy\nfield",
        facecolor="#EAF1F8",
        edgecolor="#2C6BA0",
    )
    arrow(ax, (0.217, 0.52), (0.26, 0.52))
    arrow(ax, (0.472, 0.52), (0.515, 0.52))
    arrow(ax, (0.727, 0.52), (0.77, 0.52))
    ax.text(
        0.5,
        0.08,
        "Train/validation axis selection → freeze → target read",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#555555",
        fontsize=6.7,
    )


def draw_cohort(ax: plt.Axes, audit: dict[str, Any]) -> None:
    panel_label(ax, "B")
    ax.set_title("Pre-existing axis subgroup and target cohort", loc="left")
    categories = [
        ("Physical-axis\nformal", audit["physical_axis_formal_n"], "#4D4D4D"),
        ("A/B axis-\npositive", audit["axis_positive_primary_n"], "#B2182B"),
        ("Target-\nready", audit["target_metadata_eligible_n"], "#2C6BA0"),
        (
            "Axis +\ntarget",
            len(audit["axis_positive_target_metadata_intersection"]),
            "#7B3294",
        ),
    ]
    x = np.arange(len(categories))
    values = [item[1] for item in categories]
    ax.bar(
        x,
        values,
        color=[item[2] for item in categories],
        width=0.68,
        edgecolor="white",
        linewidth=0.6,
    )
    for index, value in enumerate(values):
        ax.text(
            index,
            value + 0.6,
            str(value),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_xticks(x, [item[0] for item in categories])
    ax.set_ylabel("Patients")
    ax.set_ylim(0, max(values) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)


def paired_columns(
    ax: plt.Axes,
    first: np.ndarray,
    second: np.ndarray,
    labels: tuple[str, str],
    colors: tuple[str, str],
    ylabel: str,
) -> None:
    for index in range(len(first)):
        ax.plot(
            [0, 1],
            [first[index], second[index]],
            color="#D0D0D0",
            linewidth=0.7,
            zorder=1,
        )
    ax.scatter(
        np.zeros(len(first)),
        first,
        s=17,
        color=colors[0],
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    ax.scatter(
        np.ones(len(second)),
        second,
        s=17,
        color=colors[1],
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    for position, values, color in (
        (0, first, colors[0]),
        (1, second, colors[1]),
    ):
        median = float(np.median(values))
        ax.plot(
            [position - 0.18, position + 0.18],
            [median, median],
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=3,
        )
    ax.axhline(0, color="#777777", linewidth=0.7, linestyle="--")
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)


def draw_axis_alignment(
    ax: plt.Axes, a0: pd.DataFrame, a1: pd.DataFrame
) -> None:
    panel_label(ax, "C")
    ax.set_title("Axis alignment in the A/B-positive subgroup", loc="left")
    shared = sorted(set(a0.subject) & set(a1.subject))
    first = (
        a0.set_index("subject").loc[shared, "alignment_margin"].to_numpy(float)
    )
    second = (
        a1.set_index("subject").loc[shared, "alignment_margin"].to_numpy(float)
    )
    paired_columns(
        ax,
        first,
        second,
        ("Transition-\nselected", "RNN-\nselected"),
        (COLORS["transition"], COLORS["rnn"]),
        "Alignment above candidate median",
    )
    ax.text(
        0.98,
        0.04,
        f"n={len(shared)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#555555",
    )


def draw_predictive_contribution(
    ax: plt.Axes, a1: pd.DataFrame
) -> None:
    panel_label(ax, "D")
    ax.set_title("Held-out axis and source contributions", loc="left")
    axis_values = a1.axis_over_isotropic_heldout_benefit.to_numpy(float)
    reversed_frame = a1.loc[a1.relation == "reversed"]
    source_values = (
        reversed_frame.source_over_no_source_heldout_benefit.to_numpy(float)
    )
    rng = np.random.default_rng(20260727)
    for position, values, color in (
        (0, axis_values, COLORS["axis"]),
        (1, source_values, COLORS["source"]),
    ):
        jitter = rng.normal(0, 0.035, len(values))
        ax.scatter(
            position + jitter,
            values,
            s=18,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        median = float(np.median(values))
        ax.plot(
            [position - 0.20, position + 0.20],
            [median, median],
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
        )
    ax.axhline(0, color="#777777", linewidth=0.7, linestyle="--")
    ax.set_xticks(
        [0, 1],
        [f"Axis term\n(n={len(axis_values)})", f"Source term\n(n={len(source_values)})"],
    )
    ax.set_ylabel("Held-out NLL benefit")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)


def expected_rank(distribution: np.ndarray) -> np.ndarray:
    participation = distribution[:, 1:].sum(axis=1)
    bins = (np.arange(10, dtype=float) + 0.5) / 10.0
    numerator = distribution[:, 1:] @ bins
    return np.divide(
        numerator,
        participation,
        out=np.ones_like(numerator),
        where=participation > 0,
    )


def draw_rank_distributions(
    fig: plt.Figure,
    spec: Any,
    _fidelity: dict[str, Any],
) -> None:
    inner = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=spec, width_ratios=[1, 1], wspace=0.12
    )
    axes = [fig.add_subplot(inner[0, index]) for index in range(2)]
    representation_path = (
        BASE / "representations/per_subject" / f"{EXAMPLE_SUBJECT}.npz"
    )
    with np.load(representation_path, allow_pickle=False) as data:
        empirical = np.asarray(data["empirical_train80"], dtype=float)
        full = np.asarray(data["full_fixed_axis"], dtype=float)
    order = np.lexsort(
        (
            -empirical[:, 1:].sum(axis=1),
            expected_rank(empirical),
        )
    )
    arrays = (empirical[order], full[order])
    titles = ("Empirical train ranks", "RNN free rollouts")
    maximum = float(max(array.max() for array in arrays))
    image = None
    for index, (ax, array, title) in enumerate(zip(axes, arrays, titles)):
        image = ax.imshow(
            array,
            aspect="auto",
            cmap="viridis",
            norm=Normalize(0, maximum),
            interpolation="nearest",
        )
        ax.set_title(title, pad=4)
        ax.set_xlabel("Rank-distribution bin")
        ax.set_xticks([0, 3, 6, 10], ["Ø", "early", "middle", "late"])
        if index == 0:
            ax.set_ylabel("Contacts")
            ax.set_yticks([0, len(array) - 1], ["1", str(len(array))])
        else:
            ax.set_yticks([])
    panel_label(axes[0], "E")
    assert image is not None
    colorbar = fig.colorbar(
        image,
        ax=axes,
        fraction=0.035,
        pad=0.025,
        location="right",
    )
    colorbar.set_label("Probability")


def draw_static_readout(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    panel_label(ax, "F")
    ax.set_title("Source-free early-ictal static readout", loc="left")
    order = [
        "empirical_train80",
        "full_fixed_axis",
        "no_history",
        "local_isotropic",
        "node_only",
    ]
    labels = [
        "Empirical\nranks",
        "Full\nRNN",
        "No\nhistory",
        "Isotropic",
        "Node\nonly",
    ]
    colors = [
        COLORS["empirical"],
        COLORS["full"],
        COLORS["no_history"],
        COLORS["isotropic"],
        COLORS["node"],
    ]
    rng = np.random.default_rng(20260728)
    for position, (model, color) in enumerate(zip(order, colors)):
        values = metrics.loc[
            metrics.model == model, "all_contact_margin"
        ].to_numpy(float)
        jitter = rng.normal(0, 0.045, len(values))
        ax.scatter(
            position + jitter,
            values,
            s=13,
            facecolor=color,
            edgecolor="white",
            linewidth=0.25,
            alpha=0.85,
            zorder=2,
        )
        median = float(np.median(values))
        q1, q3 = np.quantile(values, [0.25, 0.75])
        ax.vlines(position, q1, q3, color=color, linewidth=2.3, zorder=3)
        ax.plot(
            [position - 0.18, position + 0.18],
            [median, median],
            color=color,
            linewidth=2.3,
            solid_capstyle="round",
            zorder=4,
        )
    ax.axhline(0, color="#777777", linewidth=0.7, linestyle="--")
    ax.set_xticks(np.arange(len(order)), labels)
    ax.set_ylabel("Spearman ρ above contact-shuffle median")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.text(
        0.98,
        0.96,
        "Patients, n=14",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#555555",
    )


def main() -> None:
    required = {
        "audit": BASE / "input_audit/INPUT_AUDIT_STATUS.json",
        "a0_status": BASE / "axis_readback_stage_a0/STAGE_A0_STATUS.json",
        "a0_metrics": BASE / "axis_readback_stage_a0/patient_metrics.csv",
        "a1_status": BASE / "formal/AXIS_SELECTION_GATE_STATUS.json",
        "a1_metrics": BASE / "formal/axis_selected_patient_metrics.csv",
        "static_status": BASE / "static_readout/STATIC_READOUT_GATE_STATUS.json",
        "static_metrics": BASE / "static_readout/patient_model_metrics.csv",
        "diagnostic": BASE / "static_readout/STATIC_READOUT_DIAGNOSTICS.json",
        "fidelity": BASE / "representations/RANK_DISTRIBUTION_FIDELITY.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise SystemExit("missing accepted inputs:\n" + "\n".join(missing))
    audit = read_json(required["audit"])
    a0_status = read_json(required["a0_status"])
    a1_status = read_json(required["a1_status"])
    static_status = read_json(required["static_status"])
    diagnostic = read_json(required["diagnostic"])
    fidelity = read_json(required["fidelity"])
    if any(
        payload.get("status") != "COMPLETE"
        for payload in (
            a0_status,
            a1_status,
            static_status,
            diagnostic,
            fidelity,
        )
    ):
        raise SystemExit("one or more accepted inputs are incomplete")

    a0 = pd.read_csv(required["a0_metrics"])
    a1 = pd.read_csv(required["a1_metrics"])
    static_metrics = pd.read_csv(required["static_metrics"])
    set_style()
    fig = plt.figure(figsize=(7.2, 7.0), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        2,
        left=0.11,
        right=0.975,
        bottom=0.075,
        top=0.955,
        hspace=0.82,
        wspace=0.38,
        height_ratios=[0.88, 1.0, 1.12],
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    ax_f = fig.add_subplot(grid[2, 1])
    draw_workflow(ax_a)
    draw_cohort(ax_b, audit)
    draw_axis_alignment(ax_c, a0, a1)
    draw_predictive_contribution(ax_d, a1)
    draw_rank_distributions(fig, grid[2, 0], fidelity)
    draw_static_readout(ax_f, static_metrics)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "fig6_rnn_axis_static_transfer_v2_4"
    fig.savefig(stem.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    metadata = {
        "contract": "fig6_rnn_axis_static_transfer_v2_4",
        "status": "COMPLETE",
        "source_files": {key: str(path.relative_to(ROOT)) for key, path in required.items()},
        "panels": {
            "A": "frozen analysis chain",
            "B": "cohort denominators",
            "C": "transition-selected and RNN-selected axis alignment margins",
            "D": "selected-axis and source-term heldout NLL benefits",
            "E": (
                "display-only node rank distributions for the first sorted "
                "axis-positive target-ready patient"
            ),
            "F": "source-free early-ictal static readout margins",
        },
        "example_subject_internal": EXAMPLE_SUBJECT,
        "example_selection_rule": (
            "first sorted patient in the frozen axis-positive/target-ready intersection"
        ),
        "gate_a": a1_status["gate_a_axis_positive_construct_validity"],
        "gate_s": static_status["gate_s_source_free_static_readout"],
        "gate_h": static_status["gate_h_history_contribution"],
        "gate_x": static_status["gate_x_axis_contribution"],
        "dynamic_source_conditioned_rollout": static_status[
            "dynamic_source_conditioned_rollout"
        ],
        "target_values_read": True,
        "claim_scope": (
            "bounded model closeout; does not negate the empirical A/B axis or "
            "the existing empirical interictal-to-ictal field result"
        ),
    }
    stem.with_name(stem.name + "_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
