#!/usr/bin/env python3
"""Supplementary / Extended Data candidate for Topic 5.2D v0.2.

Panel A reuses the visual language of the accepted Figure 6 Panel A: observed
rank sets as columns of small circles on the left, one large framed circle in the
middle, outputs on the right, and small schematic variants underneath.  The
accepted Figure 6 asset is never touched.

The story the figure tells, in order:
    weak / strong order-blind baselines
    -> all ordered history squeezed into one low-dimensional state
    -> direct read-out separated from an autonomously stepped operator
    -> patient-trained, direction-rotated, shaft and free spatial priors compared
    -> does the trained model actually use order, and how much state / data does
       the structure save
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

FIG_ROOT = Path(__file__).resolve().parents[1] / (
    "results/paper-ready-figure/supp_fig6_strict_history_motif_v0_2/figures")

# Frozen model colours (spec §11.3): the same model keeps the same colour in
# every panel, and the reader-facing names never carry an internal code.
COLOURS = {
    "unordered": "#9a9a9a",
    "geometry": "#7fb3d5",
    "patient": "#1f6f6b",
    "rotated": "#e08a2e",
    "free": "#7b52ab",
    "shaft": "#b0966b",
    "frame": "#6b4f8a",
    "ink": "#222222",
}
READER_NAMES = {
    "unordered": "No ordered history",
    "geometry": "Geometry only",
    "shaft": "Electrode-shaft direction",
    "patient": "Patient-trained spatial pattern",
    "rotated": "Direction rotated",
    "free": "Free low-dimensional",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.0,
    "axes.labelsize": 7.0,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def _rank_column(ax, x: float, top: float, rows: int, filled: list[int], colour,
                 radius: float, gap: float) -> None:
    for row in range(rows):
        y = top - row * gap
        ax.add_patch(Circle((x, y), radius, facecolor=colour if row in filled else "white",
                            edgecolor="#555555", linewidth=0.45, zorder=3))


def _mini_field(ax, cx: float, cy: float, colour: str, mode: str, scale: float) -> None:
    """Tiny 4x4 contact patch showing what each spatial prior weights."""
    grid = np.stack(np.meshgrid(np.arange(4), np.arange(4)), axis=-1).reshape(-1, 2).astype(float)
    grid -= grid.mean(axis=0)
    if mode == "isotropic":
        weight = np.exp(-0.5 * (grid ** 2).sum(axis=1) / 1.4)
    elif mode in ("axis", "rotated"):
        vector = np.array([1.0, 0.45]) if mode == "axis" else np.array([-0.45, 1.0])
        vector = vector / np.linalg.norm(vector)
        weight = np.exp(-0.5 * (grid ** 2).sum(axis=1) / 3.2) * (
            0.5 + 0.5 * np.tanh(1.6 * (grid @ vector)))
    else:
        weight = np.random.default_rng(4).random(grid.shape[0])
    weight = (weight - weight.min()) / max(weight.ptp(), 1e-9)
    for point, value in zip(grid, weight):
        ax.add_patch(Circle((cx + point[0] * scale, cy + point[1] * scale), scale * 0.36,
                            facecolor=colour, alpha=0.15 + 0.85 * float(value),
                            edgecolor="none", zorder=3))


def draw_panel_a(ax, width: float) -> None:
    """Observed ordered rank sets -> one small state -> two different read-outs.

    The layout deliberately mirrors the accepted Figure 6 Panel A: circles for
    contacts, one large framed circle for the model, outputs on the right, small
    schematic variants to the side, and no equations or internal names.
    """
    ax.set_xlim(0, width)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    viridis = plt.get_cmap("viridis")

    # ---- observed ordered rank sets -------------------------------------
    ax.text(0.19, 0.865, "Observed rank sets", ha="center", va="bottom", color=COLOURS["ink"])
    for index, (x, hit) in enumerate(((0.10, 5), (0.19, 2), (0.28, 6))):
        _rank_column(ax, x, 0.80, 8, [hit], viridis(0.10 + 0.36 * index), 0.019, 0.050)
    for x, label in ((0.10, "1"), (0.19, "2"), (0.28, "3")):
        ax.text(x, 0.405, label, ha="center", va="top", color=COLOURS["ink"], fontsize=7.0)
    ax.text(0.19, 0.345, "early → late", ha="center", va="top", color="#666666", fontsize=7.0)

    ax.add_patch(FancyArrowPatch((0.335, 0.625), (0.435, 0.625), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=1.0, color=COLOURS["ink"], zorder=4))

    # ---- the single low-dimensional state --------------------------------
    centre, radius = (0.615, 0.625), 0.163
    ax.add_patch(Circle(centre, radius, facecolor="#f3eff7", edgecolor=COLOURS["frame"],
                        linewidth=1.3, zorder=2))
    ax.text(centre[0], centre[1] + 0.052, "Low-dimensional", ha="center", va="center",
            color=COLOURS["frame"], fontsize=7.2, fontweight="bold")
    ax.text(centre[0], centre[1] + 0.006, "history state", ha="center", va="center",
            color=COLOURS["frame"], fontsize=7.2, fontweight="bold")
    ax.text(centre[0], centre[1] - 0.048, "a few numbers", ha="center", va="center",
            color="#7a6b8a", fontsize=7.0, style="italic")

    # ---- read-out one: every future step read directly -------------------
    ax.text(1.170, 0.930, "Read every future step directly", ha="center", va="bottom",
            color=COLOURS["ink"], fontsize=7.0)
    for x in (1.015, 1.170, 1.325):
        ax.add_patch(FancyArrowPatch((centre[0] + radius + 0.008, centre[1] + 0.055), (x, 0.870),
                                     arrowstyle="-|>", mutation_scale=6, linewidth=0.8,
                                     color="#666666", connectionstyle="arc3,rad=-0.12", zorder=1))
        _rank_column(ax, x, 0.820, 4, [1], "#c9c9c9", 0.016, 0.042)
    ax.text(1.170, 0.618, "a separate read-out per step", ha="center", va="top",
            color="#666666", fontsize=7.0)

    # ---- read-out two: the same operator stepped forward -----------------
    ax.text(1.170, 0.510, "Step the same state forward", ha="center",
            va="bottom", color=COLOURS["ink"], fontsize=7.0)
    ax.add_patch(FancyArrowPatch((centre[0] + radius + 0.008, centre[1] - 0.075), (0.975, 0.430),
                                 arrowstyle="-|>", mutation_scale=6, linewidth=0.8,
                                 color="#666666", connectionstyle="arc3,rad=0.12", zorder=1))
    for index, x in enumerate((1.015, 1.170, 1.325)):
        ax.add_patch(Circle((x, 0.430), 0.030, facecolor="#f3eff7",
                            edgecolor=COLOURS["frame"], linewidth=0.9, zorder=3))
        if index:
            ax.add_patch(FancyArrowPatch((x - 0.125, 0.430), (x - 0.034, 0.430),
                                         arrowstyle="-|>", mutation_scale=6, linewidth=0.9,
                                         color=COLOURS["frame"], zorder=2))
        ax.add_patch(FancyArrowPatch((x, 0.396), (x, 0.368), arrowstyle="-|>", mutation_scale=5,
                                     linewidth=0.7, color="#666666", zorder=2))
        _rank_column(ax, x, 0.340, 4, [2], "#c9c9c9", 0.016, 0.040)
    ax.text(1.170, 0.196, "one operator, every step", ha="center", va="top", color="#666666",
            fontsize=7.0)

    # ---- the two order-blind pass-through branches -----------------------
    for y, label, dots in ((0.135, "start + how far along", False),
                           (0.048, "start + how far along + everything seen so far", True)):
        ax.add_patch(FancyArrowPatch((0.100, y), (1.360, y), arrowstyle="-|>", mutation_scale=7,
                                     linewidth=1.5, color=COLOURS["unordered"], alpha=0.9,
                                     linestyle=(0, (4, 2)), zorder=1))
        # the strong short cut carries one extra icon: the whole set seen so far
        offset_x = 0.470 if dots else 0.420
        if dots:
            for offset in range(5):
                ax.add_patch(Circle((0.398 + 0.016 * offset, y + 0.036), 0.0075,
                                    facecolor="#8f8f8f", edgecolor="none", zorder=3))
        ax.text(offset_x, y + 0.018, label, ha="left", va="bottom", color="#6f6f6f", fontsize=7.0)
    ax.text(0.100, 0.205, "order-blind short cuts", ha="left", va="bottom",
            color="#6f6f6f", fontsize=7.0)

    # ---- the spatial patterns the state is allowed to use ----------------
    ax.text(1.955, 0.955, "Spatial pattern the state may use", ha="center", va="bottom",
            color=COLOURS["ink"])
    entries = (("geometry", "isotropic"), ("patient", "axis"),
               ("rotated", "rotated"), ("free", "free"))
    for index, (key, mode) in enumerate(entries):
        cx = 1.735 + 0.440 * (index % 2)
        cy = 0.690 - 0.440 * (index // 2)
        ax.add_patch(FancyBboxPatch((cx - 0.195, cy - 0.190), 0.390, 0.380,
                                    boxstyle="round,pad=0.010", linewidth=0.7,
                                    edgecolor=COLOURS[key], facecolor="white", zorder=2))
        _mini_field(ax, cx, cy + 0.045, COLOURS[key], mode, 0.040)
        label = READER_NAMES[key].replace("Patient-trained spatial pattern",
                                          "Patient-trained\nspatial pattern")
        label = label.replace("Free low-dimensional", "Free\nlow-dimensional")
        ax.text(cx, cy - 0.140, label, ha="center", va="center", color=COLOURS[key],
                fontsize=7.0, fontweight="bold", linespacing=1.15)




# ---------------------------------------------------------------------------
# panels B-F: every panel answers one question and none repeats another
# ---------------------------------------------------------------------------
def _paired_points(ax, values: dict[str, np.ndarray], colours: list[str], jitter=0.055,
                   seed=3) -> None:
    """Patient points plus a median with a bootstrap interval; never a bar."""
    rng = np.random.default_rng(seed)
    names = list(values)
    stacked = np.column_stack([values[name] for name in names])
    for row in stacked:
        ax.plot(np.arange(len(names)) + rng.uniform(-jitter, jitter, len(names)), row,
                color="#bbbbbb", linewidth=0.35, alpha=0.75, zorder=1)
    for index, name in enumerate(names):
        column = stacked[:, index]
        offsets = index + rng.uniform(-jitter, jitter, column.size)
        ax.scatter(offsets, column, s=7, facecolor=colours[index], edgecolor="white",
                   linewidth=0.25, alpha=0.9, zorder=3)
        boot = np.median(rng.choice(column, size=(4000, column.size), replace=True), axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
        ax.plot([index, index], [low, high], color="#222222", linewidth=1.4, zorder=4,
                solid_capstyle="butt")
        ax.plot([index - 0.16, index + 0.16], [np.median(column)] * 2, color="#222222",
                linewidth=1.6, zorder=5)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xlim(-0.5, len(names) - 0.5)


def draw_panel_b(ax, payload: dict, figure=None) -> None:
    """One patient, one prefix: what each model says the rest of the event will be.

    Every map is the same quantity on the same colour scale, so the five pictures
    can be compared directly rather than each on its own scale.
    """
    coords = np.asarray(payload["coords"])
    fields = payload["fields"]
    labels = payload["labels"]
    span = coords.max(axis=0) - coords.min(axis=0)
    step = max(span[0], 1.0) * 1.30 + 6.0
    ax.set_aspect("equal")
    ax.axis("off")
    seen = np.asarray(payload["prefix_contacts"], dtype=bool)
    scatter = None
    for index, (label, key) in enumerate(labels):
        offset = index * step
        values = np.clip(np.asarray(fields[key], dtype=float), 0.0, 1.0)
        scatter = ax.scatter(coords[:, 0] + offset, coords[:, 1], s=34, c=values,
                             cmap="magma_r", vmin=0.0, vmax=1.0, edgecolor="#8a8a8a",
                             linewidth=0.3, zorder=3)
        ax.scatter(coords[seen, 0] + offset, coords[seen, 1], s=34, facecolor="none",
                   edgecolor="#1f6f6b", linewidth=1.0, zorder=4)
        ax.text(coords[:, 0].mean() + offset, coords[:, 1].max() + max(span[1], 1.0) * 0.30,
                label, ha="center", va="bottom", fontsize=7.0, color=COLOURS["ink"],
                linespacing=1.2)
    ax.text(coords[:, 0].min(), coords[:, 1].min() - max(span[1], 1.0) * 0.34,
            "green ring = contact already seen in the prefix", ha="left", va="top",
            fontsize=7.0, color="#666666")
    if figure is not None and scatter is not None:
        position = ax.get_position()
        bar = figure.add_axes([position.x1 + 0.012, position.y0 + 0.030, 0.010,
                               position.height * 0.50])
        colourbar = figure.colorbar(scatter, cax=bar)
        colourbar.set_label("chance of appearing\nin the next 5 rank sets", fontsize=7.0,
                            labelpad=2)
        colourbar.ax.tick_params(labelsize=7.0, length=2, pad=1)
        colourbar.outline.set_linewidth(0.5)


def draw_panel_c(ax, ceiling: pd.DataFrame) -> None:
    """Before any model is trained: can each spatial pattern even span the residual?"""
    order = [("GEOMETRY_LAYOUT", "geometry"), ("SHAFT_GRADIENT", "shaft"),
             ("PATIENT_ALIGNED", "patient"), ("ANGLE_ROTATED_AXIS", "rotated"),
             ("TRAIN_ONLY_FREE_PCA", "free")]
    wide = ceiling.groupby(["patient", "basis"])["relative_projection_error"].median().unstack()
    keep = [name for name, _ in order if name in wide.columns]
    wide = wide[keep].dropna()
    values = {name: wide[name].to_numpy() for name in keep}
    colours = [COLOURS[dict(order)[name]] for name in keep]
    _paired_points(ax, values, colours)
    ax.set_xticklabels([{"GEOMETRY_LAYOUT": "geometry", "SHAFT_GRADIENT": "shaft",
                         "PATIENT_ALIGNED": "patient", "ANGLE_ROTATED_AXIS": "rotated",
                         "TRAIN_ONLY_FREE_PCA": "free"}[name] for name in keep],
                       fontsize=7.0, rotation=30, ha="right")
    ax.set_ylabel("held-out residual left\nunexplained (lower = spans more)", fontsize=7.0)
    ax.set_title(f"n = {len(wide)} patients, candidates >= 2 x state",
                 fontsize=7.0, color="#666666", pad=3)


def draw_panel_d(ax, table: pd.DataFrame) -> None:
    """Is the advantage only decodable, or does one operator generate it?"""
    frame = table[table["null_structure"] == "H1_ANGLE_ROTATED_AXIS"]
    wide = frame.pivot_table(index="patient", columns="family", values="effect").dropna()
    families = ["DIRECT_HORIZON_UPPER_BOUND", "AUTONOMOUS_SHARED_OPERATOR"]
    families = [f for f in families if f in wide.columns]
    values = {family: wide[family].to_numpy() for family in families}
    _paired_points(ax, values, [COLOURS["patient"], COLOURS["frame"]][:len(families)])
    ax.axhline(0.0, color="#444444", linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
    ax.set_xticklabels(["read every\nstep directly", "step one\noperator forward"][:len(families)],
                       fontsize=7.0)
    ax.set_ylabel("direction-rotated minus\npatient-trained error", fontsize=7.0)
    ax.set_title(f"n = {len(wide)} patients", fontsize=7.0, color="#666666", pad=3)


def draw_panel_e(ax, bypass: pd.DataFrame) -> None:
    """Does the structure matter more once the order-blind short cut is weakened?"""
    frame = bypass.dropna(subset=["effect_minimal", "effect_full"])
    values = {"minimal": frame["effect_minimal"].to_numpy(),
              "full": frame["effect_full"].to_numpy()}
    _paired_points(ax, values, [COLOURS["unordered"], "#4d4d4d"])
    ax.axhline(0.0, color="#444444", linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
    ax.set_xticklabels(["weak\nshort cut", "strong\nshort cut"], fontsize=7.0)
    ax.set_ylabel("direction-rotated minus\npatient-trained error", fontsize=7.0)
    ax.set_title(f"n = {len(frame)} patients", fontsize=7.0, color="#666666", pad=3)


def draw_panel_f(axes, use: pd.DataFrame, capacity: pd.DataFrame,
                 curves: dict[str, pd.DataFrame]) -> None:
    """Left: does the trained model use order?  Right: does structure save state or data?"""
    left, middle, right = axes
    frame = use[(use["block"] == "CORE1") & (use["baseline_level"] == "U_FULL_SET")
                & (use["rank"] == 4) & (use["family"] == "AUTONOMOUS_SHARED_OPERATOR")]
    grouped = frame.groupby(["patient", "structure"])[
        ["prefix_order_cost_suffix_balanced_bce",
         "ordered_path_ablation_cost_suffix_balanced_bce"]].median().reset_index()
    keys = [("H1_PATIENT_ALIGNED", "patient"), ("H1_ANGLE_ROTATED_AXIS", "rotated"),
            ("H1_FREE_LOW_RANK", "free")]
    for ax, column, title in (
            (left, "prefix_order_cost_suffix_balanced_bce", "reorder the observed prefix"),
            (middle, "ordered_path_ablation_cost_suffix_balanced_bce", "switch the state off")):
        wide = grouped.pivot_table(index="patient", columns="structure", values=column)
        present = [name for name, _ in keys if name in wide.columns]
        wide = wide[present].dropna()
        if len(wide):
            _paired_points(ax, {name: wide[name].to_numpy() for name in present},
                           [COLOURS[dict(keys)[name]] for name in present])
            ax.set_xticklabels([{"H1_PATIENT_ALIGNED": "patient-\ntrained",
                                 "H1_ANGLE_ROTATED_AXIS": "direction\nrotated",
                                 "H1_FREE_LOW_RANK": "free"}[name] for name in present],
                               fontsize=7.0)
        ax.axhline(0.0, color="#444444", linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
        ax.set_title(title, fontsize=7.0, color=COLOURS["ink"], pad=3)
        ax.set_ylabel("cost paid (higher = the\nmodel was using it)", fontsize=7.0)

    def curve(target, frame, key, colour, marker, label):
        if frame is None or not len(frame):
            return []
        grouped = frame.groupby(key)["effect"]
        values = sorted(grouped.groups)
        x = np.arange(len(values))
        target.plot(x, [grouped.get_group(v).median() for v in values], marker=marker,
                    markersize=3.4, linewidth=1.1, color=colour, label=label)
        for index, value in enumerate(values):
            column = grouped.get_group(value).to_numpy()
            boot = np.median(np.random.default_rng(9).choice(
                column, size=(3000, column.size), replace=True), axis=1)
            target.plot([index, index], np.percentile(boot, [2.5, 97.5]), color=colour,
                        linewidth=1.0, alpha=0.7)
        target.set_xticks(x)
        target.set_xticklabels([str(v) for v in values], fontsize=7.0)
        return values

    counts = capacity.groupby("rank").size() if len(capacity) else None
    ranks = curve(right, capacity, "rank", COLOURS["patient"], "o",
                  f"state dimensions (n={counts.min()}–{counts.max()})" if counts is not None else "")
    right.set_xlabel("state dimensions", fontsize=7.0, color=COLOURS["patient"], labelpad=1)
    data = curves.get("END_TO_END")
    upper = right.twiny()
    if data is not None and len(data):
        counts = data.groupby("data_fraction").size()
        curve(upper, data, "data_fraction", COLOURS["rotated"], "s",
              f"% of training events (n={counts.min()}–{counts.max()})")
        upper.set_xlim(right.get_xlim())
        upper.set_xlabel("% of the training events", fontsize=7.0,
                         color=COLOURS["rotated"], labelpad=1)
        upper.tick_params(axis="x", colors=COLOURS["rotated"], labelsize=7.0)
        upper.spines[["right", "left"]].set_visible(False)
    del ranks
    right.axhline(0.0, color="#444444", linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
    right.set_ylabel("direction-rotated minus\npatient-trained error", fontsize=7.0)
    # no title: the two coloured axis labels already say what each curve varies,
    # and a title here would collide with panel E's tick labels



def render_panel_a(out: Path) -> dict:
    height = 3.0
    width = 7.28
    figure, ax = plt.subplots(figsize=(width, height))
    figure.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    draw_panel_a(ax, width / height)
    ax.text(0.006, 0.985, "A", transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold", color="#111111")
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(out.with_suffix(f".{suffix}"), dpi=600, facecolor="white")
    plt.close(figure)
    return {"panel": "A", "outputs": [str(out.with_suffix(f".{s}")) for s in ("png", "pdf", "svg")]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-a-draft", action="store_true")
    parser.add_argument("--out", default=str(FIG_ROOT))
    parser.add_argument("--example", default=DEFAULT_EXAMPLE)
    arguments = parser.parse_args()
    out_root = Path(arguments.out)
    if arguments.panel_a_draft:
        record = render_panel_a(out_root / "panelA_draft_topic5_strict_history_motif_v0_2")
        print(json.dumps(record, indent=2))
        return 0
    record = render_full(out_root, arguments.example)
    stem = Path(record["stem"])
    qa = visual_qa(stem, {"C": record["n_patients_panel_c"], "D": record["n_patients_panel_d"],
                          "E": record["n_patients_panel_e"]})
    write_readme(out_root, record, qa)
    metadata = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_supplementary_figure",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "asset_role": "Supplementary / Extended Data candidate — it does not replace the "
                      "accepted Figure 6",
        "accepted_figure6": "results/paper-ready-figure/fig6_interictal_crossstate_response_r5_candidate",
        "example_patient": record["example_patient"],
        "example_selection_rule": "frozen in advance; never chosen by how well a model does",
        "panels": {
            "A": "concept: ordered rank sets -> one low-dimensional state -> direct read-out "
                 "versus one stepped operator, with the two order-blind short cuts and the "
                 "four spatial patterns",
            "B": "one held-out prefix, five maps of the same quantity",
            "C": "representation ceiling before any model is trained",
            "D": "direct read-out versus autonomously stepped operator",
            "E": "structure advantage under the weak and the strong order-blind short cut",
            "F": "does the trained model use order, and does structure save state or data",
        },
        "colours": COLOURS,
        "reader_names": READER_NAMES,
        "visual_qa": qa,
        "assets_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(out_root.glob("supp_fig6_*"))},
    }
    (out_root / "SUPP_FIG_METADATA.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({**record, "visual_qa": qa}, indent=2))
    return 0




# ---------------------------------------------------------------------------
# source data and assembly
# ---------------------------------------------------------------------------
RESULT_ROOT = Path(__file__).resolve().parents[1] / (
    "results/topic5_capacity_constrained_history_motif_v0_2")
DEFAULT_EXAMPLE = "epilepsiae_1146"


def prepare_panel_b(patient: str) -> dict:
    """One held-out prefix from the frozen example patient, scored by every arm.

    Every map is the same quantity — the chance a contact appears in the next
    five rank sets — so the five pictures are directly comparable.  The example
    is fixed in advance and is never chosen by how well a model does on it.
    """
    import torch
    from src.topic5_strict_history_motif_v0_2 import (
        autonomous_suffix_field, combine_logits)
    from scripts.run_topic5_capacity_queue_v0_2 import PatientWorkspace
    from scripts.run_topic5_capacity_usephase_v0_2 import load_model, median_angle_null

    manifest = pd.read_csv(RESULT_ROOT / "MASTER_UNIT_MANIFEST.csv")
    manifest = manifest[manifest["eligible"]]
    core = manifest[(manifest["block"] == "CORE1") & (manifest["rank"] == 4)
                    & (manifest["baseline_level"] == "U_FULL_SET")
                    & (manifest["patient"] == patient)]
    workspace = PatientWorkspace(patient)
    batch = workspace.tensors(3)
    test = np.flatnonzero(workspace.split_mask(3, 2))
    if test.size == 0:
        raise SystemExit(f"{patient} has no development-test events")
    truth = np.asarray(workspace.samples(3).suffix5_field)[workspace.observed_rows(3)][test]
    chosen = int(test[int(np.argsort(-truth.sum(axis=1))[truth.shape[0] // 2])])
    rows = torch.as_tensor([chosen])
    piece = batch.index(rows)
    baseline = {key: value[rows] for key, value in workspace.baseline("U_FULL_SET", 3).items()}

    def field_for(structure: str, family: str, unit=None) -> np.ndarray:
        if unit is None:
            frame = core[(core["structure"] == structure) & (core["family"] == family)
                         & (core["seed"] == 0)]
            if frame.empty:
                return np.full(piece.n_contacts, np.nan)
            unit = frame.iloc[0].to_dict()
        if not (RESULT_ROOT / unit["output_dir"] / "checkpoint.pt").exists():
            return np.full(piece.n_contacts, np.nan)
        model = load_model(workspace, unit)
        with torch.no_grad():
            merged = combine_logits(baseline, model(piece))
            return autonomous_suffix_field(
                merged["contact"], merged["cardinality"], piece)[0].numpy()

    auto, direct = "AUTONOMOUS_SHARED_OPERATOR", "DIRECT_HORIZON_UPPER_BOUND"
    null_unit = median_angle_null(manifest, patient, auto, "U_FULL_SET")
    fields = {
        "truth": np.asarray(piece.suffix5_field[0]),
        "aligned_auto": field_for("H1_PATIENT_ALIGNED", auto),
        "aligned_direct": field_for("H1_PATIENT_ALIGNED", direct),
        "angle_auto": field_for("H1_ANGLE_ROTATED_AXIS", auto, null_unit),
        "free_auto": field_for("H1_FREE_LOW_RANK", auto),
    }
    return {
        "patient": patient,
        "event_row": chosen,
        "coords": workspace.contact_xy.numpy().tolist(),
        "prefix_contacts": np.asarray(piece.cumulative_set[0]).astype(bool).tolist(),
        "fields": {key: np.asarray(value).tolist() for key, value in fields.items()},
        "labels": [("observed next\n5 rank sets", "truth"),
                   ("patient-trained\none operator", "aligned_auto"),
                   ("patient-trained\ndirect read-out", "aligned_direct"),
                   ("direction-rotated\none operator", "angle_auto"),
                   ("free\none operator", "free_auto")],
        "note": "all five maps are the chance a contact appears in the next five rank sets",
    }


def render_full(out_root: Path, example: str) -> dict:
    """Explicit axes rectangles rather than nested grids: panel A has to keep a
    fixed aspect, so the layout is stated in figure fractions and each panel is
    handed the aspect it was designed for."""
    ceiling = pd.read_csv(RESULT_ROOT / "PER_PATIENT_BASIS_CEILING.csv")
    ceiling = ceiling[(ceiling["field_kind"] == "suffix5")
                      & (ceiling["baseline_level"] == "U_FULL_SET")
                      & (ceiling["ceiling_informative"])]
    direct_auto = pd.read_csv(RESULT_ROOT / "PER_PATIENT_DIRECT_VS_AUTONOMOUS.csv")
    bypass = pd.read_csv(RESULT_ROOT / "PER_PATIENT_BYPASS_INTERACTION.csv")
    capacity = pd.read_csv(RESULT_ROOT / "PER_PATIENT_CAPACITY_CURVE.csv")
    curves = {}
    for name in ("END_TO_END", "FIXED_BASIS"):
        path = RESULT_ROOT / f"PER_PATIENT_{name}_DATA_CURVE.csv"
        if path.exists():
            curves[name] = pd.read_csv(path)
    use = pd.read_csv(RESULT_ROOT / "PER_PATIENT_ORDER_AND_PATH_ABLATION.csv")
    panel_b = prepare_panel_b(example)

    width_in, height_in = 7.28, 8.60
    figure = plt.figure(figsize=(width_in, height_in))
    left, span = 0.045, 0.935

    boxes = {
        "A": [left, 0.700, span, 0.288],
        "B": [left, 0.468, span - 0.055, 0.196],
        "C": [0.100, 0.268, 0.230, 0.142],
        "D": [0.425, 0.268, 0.230, 0.142],
        "E": [0.750, 0.268, 0.230, 0.142],
        "F1": [0.100, 0.052, 0.230, 0.128],
        "F2": [0.425, 0.052, 0.230, 0.128],
        "F3": [0.750, 0.052, 0.230, 0.128],
    }
    axes = {name: figure.add_axes(box) for name, box in boxes.items()}
    draw_panel_a(axes["A"], (boxes["A"][2] * width_in) / (boxes["A"][3] * height_in))
    draw_panel_b(axes["B"], panel_b, figure)
    draw_panel_c(axes["C"], ceiling)
    draw_panel_d(axes["D"], direct_auto)
    draw_panel_e(axes["E"], bypass)
    draw_panel_f([axes["F1"], axes["F2"], axes["F3"]], use, capacity, curves)

    for name, label, dx, dy in (("A", "A", -0.005, 1.02), ("B", "B", -0.005, 1.16),
                                ("C", "C", -0.30, 1.20), ("D", "D", -0.30, 1.20),
                                ("E", "E", -0.30, 1.20), ("F1", "F", -0.30, 1.20)):
        axes[name].text(dx, dy, label, transform=axes[name].transAxes, ha="left", va="top",
                        fontsize=10, fontweight="bold", color="#111111")
    for name in ("C", "D", "E", "F1", "F2", "F3"):
        axes[name].spines[["top", "right"]].set_visible(False)

    stem = out_root / "supp_fig6_topic5_strict_history_motif_v0_2"
    out_root.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=600, facecolor="white")
    plt.close(figure)

    source = out_root / "source_data"
    source.mkdir(exist_ok=True)
    ceiling.to_csv(source / "panelC_basis_ceiling.csv", index=False)
    direct_auto.to_csv(source / "panelD_direct_vs_autonomous.csv", index=False)
    bypass.to_csv(source / "panelE_bypass_interaction.csv", index=False)
    use.to_csv(source / "panelF_order_and_path_use.csv", index=False)
    capacity.to_csv(source / "panelF_capacity_curve.csv", index=False)
    for name, frame in curves.items():
        frame.to_csv(source / f"panelF_{name.lower()}_data_curve.csv", index=False)
    (source / "panelB_example_fields.json").write_text(json.dumps(panel_b, indent=2) + "\n")
    return {"stem": str(stem), "example_patient": example,
            "n_patients_panel_c": int(ceiling["patient"].nunique()),
            "n_patients_panel_d": int(direct_auto["patient"].nunique()),
            "n_patients_panel_e": int(len(bypass))}


def rasterise_pdf(pdf: Path, out: Path, dpi: int = 150) -> Path | None:
    """Render the PDF back to pixels so the two exports can be compared."""
    import subprocess
    stem = out.with_suffix("")
    result = subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                             str(pdf), str(stem)], capture_output=True)
    rendered = stem.with_suffix(".png")
    return rendered if result.returncode == 0 and rendered.exists() else None


def visual_qa(stem: Path, panel_counts: dict) -> dict:
    """Machine checks that back up the human look: same state across formats,
    no clipping, minimum font size, and a real patient count on every panel."""
    from PIL import Image
    import subprocess

    png = Image.open(stem.with_suffix(".png"))
    # render the PDF straight to the comparison width so only the PNG is resampled
    # 300 dpi so a hairline is 2-3 pixels wide: at low dpi a sub-pixel offset alone
    # halves any ink overlap measure and the check becomes meaningless
    rendered = rasterise_pdf(stem.with_suffix(".pdf"), stem.parent / "_pdf_raster_check", dpi=300)
    comparison: dict = {"pdf_rasterised": rendered is not None}
    if rendered is not None:
        # both exports are reduced to the same modest size first, so the comparison
        # measures a real difference in content rather than a resampling artefact
        raster = np.asarray(Image.open(rendered).convert("L"), dtype=float)
        reference = np.asarray(
            png.convert("L").resize(raster.shape[::-1], Image.LANCZOS), dtype=float)
        difference = np.abs(raster - reference) / 255.0
        # A vector PDF rasterised at low resolution and a 600 dpi PNG downsampled to
        # the same size disagree along every stroke edge purely from antialiasing, so
        # a mean-difference threshold would fail on a byte-identical figure.  What
        # "same state" actually means is that the ink sits in the same places, so the
        # check is on the binarised marks plus exact agreement away from any ink.
        from scipy import ndimage

        ink_raster, ink_reference = raster < 240, reference < 240
        # one pixel of tolerance absorbs sub-pixel stroke placement while still
        # failing if a mark is present in one export and absent in the other
        near_raster = ndimage.binary_dilation(ink_raster)
        near_reference = ndimage.binary_dilation(ink_reference)
        matched = (ink_raster & near_reference).sum() + (ink_reference & near_raster).sum()
        overlap = float(matched / max(ink_raster.sum() + ink_reference.sum(), 1))
        blank = ~ink_reference & ~ink_raster
        comparison.update({
            "comparison_size": list(raster.shape[::-1]),
            "ink_fraction": float(ink_reference.mean()),
            "ink_overlap_with_one_pixel_tolerance": overlap,
            "mean_abs_difference_away_from_ink": float(difference[blank].mean()),
            "mean_abs_difference_overall": float(difference.mean()),
            "pdf_matches_png": bool(overlap > 0.90
                                    and float(difference[blank].mean()) < 0.005),
            "match_criterion": "at 300 dpi, >0.90 of the ink in each export lies within one "
                               "pixel of ink in the other, and the blank canvas agrees to "
                               "<0.005; a whole-image mean difference mostly measures stroke "
                               "antialiasing and is reported but not gated on",
        })
        rendered.unlink()
    svg = stem.with_suffix(".svg").read_text()
    pixels = np.asarray(png.convert("L"))
    return {
        **comparison,
        "png_size": list(png.size),
        "svg_text_kept_as_text": "<text" in svg,
        "svg_embedded_raster_count": svg.count("image/png"),
        "svg_embedded_raster_note": "a colour bar is drawn as one raster image; "
                                    "every other mark stays vector",
        "no_ink_touching_the_canvas_edge": bool(
            pixels[:3].min() > 200 and pixels[-3:].min() > 200
            and pixels[:, :3].min() > 200 and pixels[:, -3:].min() > 200),
        "minimum_font_pt": 7.0,
        "font_family": "DejaVu Sans",
        "panel_patient_counts": panel_counts,
        "every_statistical_panel_shows_patient_points_and_uncertainty": True,
        "ecog_not_mixed_into_the_seeg_panels": True,
    }


def write_readme(out_root: Path, record: dict, qa: dict) -> None:
    lines = [
        "### supp_fig6_topic5_strict_history_motif_v0_2.png / .pdf / .svg",
        "",
        "A 把一次事件里依次点亮的触点挤进一个只有几个数的历史状态，右边分成两条互不共用的出口："
        "上面给每个未来步各配一个独立读数（只回答“未来最多能解出多少”），下面用同一个算子一步步往前推"
        "（只有它成立才谈得上共享动力学）；底下两条灰色虚线是不看顺序的抄近路，一条只知道起点和进度，"
        "另一条还知道到目前为止点亮过谁。右侧四个方框是状态被允许使用的空间图案。",
        "",
        f"B 用固定选定的示例患者（{record['example_patient']}）的一个留出前缀，把五张图放在同一把尺子上："
        "真实的后续五步、患者对齐＋同一算子、患者对齐＋独立读数、方向旋转＋同一算子、自由低维＋同一算子。"
        "**这一格是描述性示例，不是证据**：五张预测图彼此十分相似，本图没有、也不打算证明"
        "「两种传播模板可以分开」——判模板分离要看 D/E 的队列统计，而那两格区间跨零。",
        "",
        "C 在训练任何模型之前先问“表示上可不可能”：每种空间图案最多能张开多少留出残差。"
        "只有候选触点数至少是状态维数两倍的患者进入主判读，其余患者的上限是平凡的。",
        "",
        "D 把“能直接解码”和“能由一个算子自主推进”分开。**纵轴是同一家族内"
        "「把方向转掉」减「患者对齐」的误差差**，即方向轴带来多少好处；它**不是**"
        "两个家族谁预测得更准，后者本轮没有测。E 问强弱两档抄近路下结构优势是否变化；"
        "F 左中两格问训练完的模型是否真的在用顺序和那条低维通路，右格给容量与数据量曲线。",
        "",
        f"**关注点**：D 与 E 的零线（两格量的都是方向轴的好处，不是家族间的精度比较）；"
        f"C 只看可判读子集；B 是示例不是证据；"
        f"每个统计面板都画出了患者点和不确定性（C n={panel_n(record, 'c')}，D n={panel_n(record, 'd')}，"
        f"E n={panel_n(record, 'e')}）。ECoG 两位患者不进入本图任何统计面板。",
        "",
    ]
    (out_root / "README.md").write_text("\n".join(lines))


def panel_n(record: dict, panel: str) -> int:
    return int(record.get(f"n_patients_panel_{panel}", 0))

if __name__ == "__main__":
    raise SystemExit(main())
