#!/usr/bin/env python3
"""Figure 6 candidate: within-patient RNN fields and connectivity constraints.

A: patient geometry and the actual linear-state recurrent computation.
B: E1146 data versus same-start free RNN rollouts for TA/TB events.
C: full-cohort interictal propagation statistic.
D: E1146 RNN TA/TB fields versus an exact 1-150 Hz early-ictal field.
E: 17-patient frozen-field cross-state statistic.
F-H: spatial recurrent constraints, interictal sufficiency, and the prior
     connectivity-motif early-ictal comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_topic5_rnn_full_cohort_field_transfer_v0_1 import (  # noqa: E402
    BLUE,
    DARK,
    ENERGY_CMAP,
    GREY,
    RED,
    TIMING_CMAP,
    _field_geometry,
    _minmax,
)
from scripts.plot_contact_plane_static import _smooth_rank_field_mm  # noqa: E402
from scripts.plot_topic5_rnn_motif_figures_v0_4 import (  # noqa: E402
    COLORS as MOTIF_COLORS,
    MODEL_LABEL as MOTIF_LABELS,
    graph_data,
)
from scripts.run_topic5_rnn_full_cohort_field_transfer_v0_1 import (  # noqa: E402
    SEEDS,
    _mode_to_ab_mapping,
)
from src.topic5_constructive_readback import (  # noqa: E402
    evaluate_mode_readback,
    fit_train_mode_readback,
)


SUBJECT = "epilepsiae_1146"
DISPLAY_SIGMA_MM = 6.0
N_EVENT_COLUMNS = 18
MOTIF_MODELS = (
    "M1_DENSE",
    "M2_UNIFORM_SET",
    "M3_FIXED_LOCAL",
    "M6_SPATIAL_MID",
)


def _load_core(canonical_root: Path, analysis: Path, fig3_metadata: Path):
    field_path = (
        canonical_root
        / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
        / f"{SUBJECT}.json"
    )
    record = json.loads(field_path.read_text())
    with np.load(analysis / "model_fields" / f"{SUBJECT}.npz", allow_pickle=True) as z:
        model = {key: np.asarray(z[key]) for key in z.files}
    meta = json.loads(fig3_metadata.read_text())
    expected = list(record["interictal_field"]["contact_order"])
    if list(meta["contact_order"]) != expected:
        raise RuntimeError("fig3b_e1146_contact_order_mismatch")
    extraction = meta["ictal_extraction"]
    if extraction["clinical_window_sec"] != [0.0, 10.0] or not extraction["is_exact_1_150"]:
        raise RuntimeError("fig3b_target_is_not_exact_0_10s_1_150hz")
    activation = np.asarray(meta["raw_ictal_robust_z_mean"], float)
    return record, model, activation, int(meta["seizure_idx"]), extraction


def _event_display(group_ids: np.ndarray, take: np.ndarray) -> np.ndarray:
    groups = np.asarray(group_ids, int)[:, take]
    out = np.full(groups.shape, np.nan, float)
    for event in range(len(groups)):
        present = groups[event] >= 0
        if not np.any(present):
            continue
        maximum = int(np.max(groups[event, present]))
        out[event, present] = groups[event, present] / max(maximum, 1)
    return out.T


def _sequence_payload(canonical_root: Path, record: dict):
    dataset_path = (
        canonical_root
        / "results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject"
        / f"{SUBJECT}.npz"
    )
    with np.load(dataset_path, allow_pickle=True) as data:
        groups = np.asarray(data["event_group_ids"], int)
        split = np.asarray(data["event_split"], int)
        names = np.asarray(data["contact_names"], str)
    readback = fit_train_mode_readback(groups[split == 0], random_state=0)
    mapping, _ = _mode_to_ab_mapping(readback.templates, names, record)
    heldout = groups[split == 1]
    observed_eval = evaluate_mode_readback(readback, heldout)
    rollout_path = (
        canonical_root
        / "results/topic5_rnn_training_sufficiency_v0_1/formal/converged_teacher_forced"
        / f"seed_{SEEDS[0]}" / SUBJECT / "rollouts.npz"
    )
    with np.load(rollout_path, allow_pickle=True) as rollout:
        generated = np.asarray(rollout["native_model__event_group_ids"], int)
    field_names = np.asarray(record["interictal_field"]["contact_order"], str)
    index = {name: i for i, name in enumerate(names)}
    take = np.asarray([index[name] for name in field_names], int)
    order = np.argsort(np.asarray(record["interictal_field"]["rank_a"], float))
    take = take[order]
    payload = {"names": field_names[order], "mapping": mapping}
    for label in ("a", "b"):
        mode = mapping[label]
        observed_idx = np.flatnonzero(np.asarray(observed_eval["labels"]) == mode)[:N_EVENT_COLUMNS]
        if len(observed_idx) < N_EVENT_COLUMNS:
            raise RuntimeError(f"insufficient_{label}_events_for_panel_b")
        payload[f"observed_{label}"] = _event_display(heldout[observed_idx], take)
        payload[f"generated_{label}"] = _event_display(generated[observed_idx], take)
        payload[f"event_indices_{label}"] = observed_idx
    return payload


def _panel_letter(ax: plt.Axes, letter: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(
        x, y, letter, transform=ax.transAxes, fontsize=16, fontweight="bold",
        va="top", ha="left", color="black", clip_on=False,
    )


def _vertical_rank_bar(ax: plt.Axes, x: float, y0: float, height: float,
                       values: np.ndarray, *, label: str) -> None:
    values = np.asarray(values, float)
    n = len(values)
    cell_h = height / n
    cmap = plt.get_cmap(TIMING_CMAP)
    for idx, value in enumerate(values):
        y = y0 + (n - idx - 1) * cell_h
        color = "#E7E9EB" if not np.isfinite(value) else cmap(float(np.clip(value, 0, 1)))
        ax.add_patch(patches.Rectangle(
            (x - 0.026, y), 0.052, cell_h * 0.92,
            facecolor=color, edgecolor="white", linewidth=0.35, zorder=3,
        ))
    ax.add_patch(patches.Rectangle(
        (x - 0.028, y0 - 0.002), 0.056, height + 0.004,
        fill=False, edgecolor="#525A60", linewidth=0.75, zorder=4,
    ))
    ax.text(x, y0 - 0.035, label, ha="center", va="top", fontsize=9.2)


def _draw_architecture(ax: plt.Axes, record: dict, sequence: dict) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    points, _, _ = _field_geometry(record)
    px = (points[:, 0] - points[:, 0].min()) / max(float(np.ptp(points[:, 0])), 1e-9)
    py = (points[:, 1] - points[:, 1].min()) / max(float(np.ptp(points[:, 1])), 1e-9)
    layout = np.column_stack((0.31 + 0.38 * px, 0.22 + 0.42 * py))
    names = list(record["interictal_field"]["contact_order"])

    input_values = sequence["observed_a"][:, 0]
    output_values = sequence["generated_a"][:, 0]
    _vertical_rank_bar(ax, 0.08, 0.22, 0.42, input_values, label="Rank input")
    _vertical_rank_bar(ax, 0.86, 0.22, 0.42, output_values, label="Rank output")

    nodes = np.asarray([
        [0.34, 0.27], [0.43, 0.20], [0.56, 0.22], [0.68, 0.30],
        [0.70, 0.45], [0.61, 0.58], [0.47, 0.62], [0.35, 0.52],
        [0.29, 0.39], [0.43, 0.38], [0.55, 0.34], [0.58, 0.48],
    ])
    recurrent_edges = (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
        (7, 8), (8, 0), (0, 9), (9, 10), (10, 3), (7, 9), (9, 11),
        (11, 4), (6, 11), (10, 5),
    )
    for source, target in recurrent_edges:
        ax.add_patch(patches.FancyArrowPatch(
            nodes[source], nodes[target], arrowstyle="-|>", mutation_scale=6.5,
            connectionstyle="arc3,rad=0.10", color="#8A969D", lw=0.65,
            alpha=0.72, zorder=1,
        ))
    for x, y in nodes:
        ax.add_patch(patches.Circle(
            (x, y), 0.031, facecolor="#DCEAF1", edgecolor="#356B8C",
            linewidth=0.9, zorder=2,
        ))
    # The real E1146 shaft geometry is superimposed on the recurrent state.
    cmap = plt.get_cmap(TIMING_CMAP)
    output_lookup = dict(zip(np.asarray(sequence["names"], str), output_values))
    order_values = np.asarray([output_lookup.get(name, np.nan) for name in names], float)
    for prefix in sorted({"".join(ch for ch in name if not ch.isdigit()) for name in names}):
        ids = [i for i, name in enumerate(names) if name.startswith(prefix)]
        if len(ids) > 1:
            order = sorted(ids, key=lambda i: int("".join(ch for ch in names[i] if ch.isdigit()) or 0))
            ax.plot(layout[order, 0], layout[order, 1], color="#424B50", lw=1.25,
                    alpha=0.75, zorder=4)
    ax.scatter(layout[:, 0], layout[:, 1], s=31, c=order_values, cmap=TIMING_CMAP,
               vmin=0, vmax=1, edgecolor="white", linewidth=0.75, zorder=5)
    ax.scatter(layout[:, 0], layout[:, 1], s=38, facecolor="none",
               edgecolor="#3F484D", linewidth=0.55, zorder=6)

    for y in (0.29, 0.40, 0.51):
        ax.add_patch(patches.FancyArrowPatch(
            (0.12, 0.42), (0.28, y), arrowstyle="-|>", mutation_scale=7,
            color="#9CA6AC", lw=0.65, alpha=0.75,
        ))
        ax.add_patch(patches.FancyArrowPatch(
            (0.72, y), (0.82, 0.42), arrowstyle="-|>", mutation_scale=7,
            color="#9CA6AC", lw=0.65, alpha=0.75,
        ))
    ax.text(0.50, 0.69, "Recurrent state", ha="center", va="bottom", fontsize=9.4)
    cax = ax.inset_axes([0.955, 0.22, 0.022, 0.42])
    cbar = plt.colorbar(ScalarMappable(Normalize(0, 1), cmap=cmap), cax=cax)
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(["Early", "Late"])
    cbar.ax.tick_params(labelsize=7.3, length=2, pad=1)
    cbar.set_label("Rank", fontsize=7.6, labelpad=1)
    ax.set_title("Patient-specific RNN", fontsize=11.5, fontweight="bold", pad=4)
    _panel_letter(ax, "A", x=-0.04, y=1.055)


def _draw_heatmap(ax: plt.Axes, matrix: np.ndarray, *, title: str,
                  names: np.ndarray, show_y: bool, color: str | None = None) -> None:
    cmap = plt.get_cmap(TIMING_CMAP).copy()
    cmap.set_bad("#ECEDEF")
    ax.imshow(matrix, origin="upper", aspect="auto", cmap=cmap, vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_title(title, fontsize=9.4, color=color or DARK, fontweight="bold", pad=3)
    ax.set_xticks([])
    ticks = [0, len(names) // 2, len(names) - 1]
    if show_y:
        ax.set_yticks(ticks, [names[i] for i in ticks], fontsize=8.0)
    else:
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7); spine.set_color("#5E666B")


def _draw_sequences(subspec, payload: dict) -> None:
    sub = subspec.subgridspec(
        3, 3, height_ratios=[0.12, 1, 1], width_ratios=[1, 1, 0.035],
        wspace=0.08, hspace=0.14,
    )
    title_ax = plt.subplot(sub[0, :]); title_ax.axis("off")
    title_ax.text(-0.035, 0.78, "B", transform=title_ax.transAxes, fontsize=16,
                  fontweight="bold", va="top", clip_on=False)
    title_ax.text(0.02, 0.78, "E1146", transform=title_ax.transAxes,
                  fontsize=11.5, fontweight="bold", va="top")
    oa = plt.subplot(sub[1, 0]); ra = plt.subplot(sub[1, 1])
    ob = plt.subplot(sub[2, 0]); rb = plt.subplot(sub[2, 1])
    _draw_heatmap(oa, payload["observed_a"], title="TA · data",
                  names=payload["names"], show_y=True, color=RED)
    _draw_heatmap(ra, payload["generated_a"], title="TA · RNN",
                  names=payload["names"], show_y=False, color=RED)
    _draw_heatmap(ob, payload["observed_b"], title="TB · data",
                  names=payload["names"], show_y=True, color=BLUE)
    _draw_heatmap(rb, payload["generated_b"], title="TB · RNN",
                  names=payload["names"], show_y=False, color=BLUE)
    cax = plt.subplot(sub[1:, 2])
    cbar = plt.colorbar(
        ScalarMappable(Normalize(0, 1), cmap=TIMING_CMAP), cax=cax,
    )
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(["Early", "Late"])
    cbar.ax.tick_params(labelsize=8.0, length=2, pad=1)


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _paired_compact(ax: plt.Axes, left: np.ndarray, right: np.ndarray,
                    labels: tuple[str, str], colors: tuple[str, str],
                    ylabel: str, title: str, p_value: float) -> None:
    left = np.asarray(left, float); right = np.asarray(right, float)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    x0, x1 = 0.0, 0.42
    for a, b in zip(left, right):
        ax.plot([x0, x1], [a, b], color="#B8BDC1", lw=0.65, alpha=0.62, zorder=1)
    ax.scatter(np.full(len(left), x0), left, s=18, color=colors[0],
               edgecolor="white", linewidth=0.45, zorder=3)
    ax.scatter(np.full(len(right), x1), right, s=18, color=colors[1],
               edgecolor="white", linewidth=0.45, zorder=3)
    for x, values in ((x0, left), (x1, right)):
        median = float(np.nanmedian(values))
        ax.plot([x - 0.085, x + 0.085], [median, median], color="#111111", lw=1.55, zorder=4)
    pooled = np.concatenate((left, right))
    lo, hi = float(np.nanmin(pooled)), float(np.nanmax(pooled))
    span = max(hi - lo, 0.1)
    stars = _stars(float(p_value))
    if stars:
        y = hi + 0.10 * span
        ax.plot([x0, x0, x1, x1], [y - 0.025 * span, y, y, y - 0.025 * span],
                color="#222222", lw=0.75)
        ax.text((x0 + x1) / 2, y + 0.015 * span, stars, ha="center", va="bottom",
                fontsize=11.5, fontweight="bold")
        hi = y + 0.14 * span
    ax.set_xlim(-0.18, 0.60); ax.set_ylim(lo - 0.08 * span, hi + 0.04 * span)
    ax.set_xticks([x0, x1], labels, fontsize=9.0)
    ax.set_ylabel(ylabel, fontsize=10.0)
    ax.set_title(title, fontsize=11.0, fontweight="bold", pad=5)
    ax.tick_params(axis="y", labelsize=9.0, length=3)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_compact_field(ax: plt.Axes, points: np.ndarray, values: np.ndarray,
                        support: np.ndarray, xlim, ylim, *, cmap: str, title: str,
                        title_color: str, show_y: bool, show_x: bool) -> None:
    X, Y, field, _, _ = _smooth_rank_field_mm(
        points[:, 0], points[:, 1], np.asarray(values, float), np.asarray(support, float),
        xlim, ylim, DISPLAY_SIGMA_MM,
    )
    ax.imshow(field, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()],
              aspect="equal", cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
    ok = np.isfinite(values)
    ax.scatter(points[ok, 0], points[ok, 1], c=np.asarray(values)[ok], cmap=cmap,
               vmin=0, vmax=1, s=20, edgecolor="white", linewidth=0.55, zorder=3)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(title, fontsize=10.0, color=title_color, fontweight="bold", pad=3)
    ax.tick_params(labelsize=8.0, length=2)
    if show_x:
        ax.set_xlabel("Propagation axis (mm)", fontsize=9.0)
    else:
        ax.tick_params(labelbottom=False)
    if show_y:
        ax.set_ylabel("Transverse (mm)", fontsize=9.0)
    else:
        ax.tick_params(labelleft=False)


def _draw_fields(subspec, record: dict, model: dict, activation: np.ndarray,
                 seizure_idx: int) -> dict:
    sub = subspec.subgridspec(
        2, 6, height_ratios=[0.13, 1],
        width_ratios=[1, 0.035, 1, 0.035, 1, 0.042],
        wspace=0.11, hspace=0.06,
    )
    title_ax = plt.subplot(sub[0, :]); title_ax.axis("off")
    title_ax.text(-0.02, 0.76, "D", transform=title_ax.transAxes, fontsize=16,
                  fontweight="bold", va="top", clip_on=False)
    title_ax.text(0.025, 0.76, "E1146", transform=title_ax.transAxes,
                  fontsize=11.5, fontweight="bold", va="top")
    ta = plt.subplot(sub[1, 0]); ta_cax = plt.subplot(sub[1, 1])
    tb = plt.subplot(sub[1, 2]); tb_cax = plt.subplot(sub[1, 3])
    en = plt.subplot(sub[1, 4]); en_cax = plt.subplot(sub[1, 5])
    points, xlim, ylim = _field_geometry(record)
    ma, mb = _minmax(model["rank_a"]), _minmax(model["rank_b"])
    energy = _minmax(activation)
    support_a = np.asarray(model["support_a"], float)
    support_b = np.asarray(model["support_b"], float)
    _draw_compact_field(ta, points, ma, support_a, xlim, ylim, cmap=TIMING_CMAP,
                        title="RNN TA", title_color=RED, show_y=True, show_x=False)
    _draw_compact_field(tb, points, mb, support_b, xlim, ylim, cmap=TIMING_CMAP,
                        title="RNN TB", title_color=BLUE, show_y=False, show_x=False)
    _draw_compact_field(en, points, energy, support_a, xlim, ylim, cmap=ENERGY_CMAP,
                        title=f"Early ictal · seizure {seizure_idx}", title_color=DARK,
                        show_y=False, show_x=False)
    for cax in (ta_cax, tb_cax):
        timing = plt.colorbar(
            ScalarMappable(Normalize(0, 1), cmap=TIMING_CMAP), cax=cax,
        )
        timing.set_ticks([0, 1]); timing.set_ticklabels(["Early", "Late"])
        timing.ax.tick_params(labelsize=7.6, length=2, pad=1)
    energy_bar = plt.colorbar(
        ScalarMappable(Normalize(float(np.nanmin(activation)), float(np.nanmax(activation))),
                       cmap=ENERGY_CMAP),
        cax=en_cax,
    )
    energy_bar.ax.tick_params(labelsize=8.0, length=2, pad=1)
    energy_bar.ax.set_title("z", fontsize=8.0, pad=2)
    field = record["interictal_field"]
    return {
        "ta_spearman": float(spearmanr(np.asarray(model["rank_a"], float), field["rank_a"]).statistic),
        "tb_spearman": float(spearmanr(np.asarray(model["rank_b"], float), field["rank_b"]).statistic),
    }


def _draw_spatial_graph(ax: plt.Axes, graph: dict[str, np.ndarray],
                        plane: dict[str, np.ndarray], title: str,
                        norm: Normalize, *, dense: bool) -> None:
    xy = np.asarray(plane["nodes_xy_mm"], float)
    contacts = np.asarray(plane["contacts_xy_mm"], float)
    edges = np.argwhere(np.asarray(graph["mask"], bool))
    segments = np.asarray([[xy[i], xy[j]] for i, j in edges], float)
    lengths = np.linalg.norm(segments[:, 0] - segments[:, 1], axis=1)
    collection = LineCollection(
        segments, cmap="magma", norm=norm,
        linewidths=0.28 if dense else 0.62,
        alpha=0.32 if dense else 0.72, zorder=1,
    )
    collection.set_array(lengths)
    ax.add_collection(collection)
    ax.scatter(xy[:, 0], xy[:, 1], s=4.0, color="#565D61", alpha=0.75, zorder=2)
    ax.scatter(contacts[:, 0], contacts[:, 1], s=21, facecolor="white",
               edgecolor="#111111", linewidth=0.75, zorder=3)
    ax.set_title(title, fontsize=9.2, pad=2.5)
    ax.set_aspect("equal"); ax.autoscale_view(); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_motif_layouts(subspec, motif_root: Path) -> None:
    sub = subspec.subgridspec(
        2, 5, height_ratios=[0.15, 1], width_ratios=[1, 1, 1, 1, 0.035],
        wspace=0.05, hspace=0.02,
    )
    title_ax = plt.subplot(sub[0, :]); title_ax.axis("off")
    title_ax.text(-0.02, 0.76, "F", transform=title_ax.transAxes, fontsize=16,
                  fontweight="bold", va="top", clip_on=False)
    title_ax.text(0.035, 0.76, "Recurrent connectivity", transform=title_ax.transAxes,
                  fontsize=11.5, fontweight="bold", va="top")
    payloads = []
    all_lengths = []
    for model_id in MOTIF_MODELS:
        _, graph, plane = graph_data(motif_root, SUBJECT, model_id)
        xy = np.asarray(plane["nodes_xy_mm"], float)
        edges = np.argwhere(np.asarray(graph["mask"], bool))
        all_lengths.extend(np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1).tolist())
        payloads.append((model_id, graph, plane))
    norm = Normalize(float(np.min(all_lengths)), float(np.max(all_lengths)))
    title_map = {
        "M1_DENSE": "Dense · 100%",
        "M2_UNIFORM_SET": "Sparse · 10%",
        "M3_FIXED_LOCAL": "Local · 10%",
        "M6_SPATIAL_MID": "Spatial + cost · 10%",
    }
    for index, (model_id, graph, plane) in enumerate(payloads):
        ax = plt.subplot(sub[1, index])
        _draw_spatial_graph(
            ax, graph, plane, title_map[model_id], norm,
            dense=model_id == "M1_DENSE",
        )
    cax = plt.subplot(sub[1, 4])
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap="magma"), cax=cax)
    cbar.ax.set_title("mm", fontsize=8.0, pad=2)
    cbar.ax.tick_params(labelsize=7.5, length=2, pad=1)


def _strip_models(ax: plt.Axes, frame: pd.DataFrame, models: tuple[str, ...],
                  metric: str, *, ylabel: str, title: str, zero: bool = True) -> None:
    rng = np.random.default_rng(20260812)
    for x, model_id in enumerate(models):
        values = frame.loc[frame.model == model_id, metric].dropna().to_numpy(float)
        jitter = rng.uniform(-0.13, 0.13, len(values))
        ax.scatter(x + jitter, values, s=16, color=MOTIF_COLORS[model_id], alpha=0.65,
                   edgecolor="none", zorder=2)
        if len(values):
            median = float(np.median(values))
            ax.plot([x - 0.20, x + 0.20], [median, median], color="#111111", lw=1.45, zorder=3)
    if zero:
        ax.axhline(0, color="#8C9398", lw=0.75, zorder=0)
    ax.set_xticks(range(len(models)), [MOTIF_LABELS[m] for m in models],
                  rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(title, fontsize=11.0, fontweight="bold", pad=5)
    ax.tick_params(axis="y", labelsize=8.5, length=3)
    ax.spines[["top", "right"]].set_visible(False)


def _add_group_stars(ax: plt.Axes, models: tuple[str, ...], p_values: dict[str, float]) -> None:
    low, high = ax.get_ylim()
    span = max(high - low, 1e-6)
    top = high + 0.10 * span
    ax.set_ylim(low, top)
    for x, model_id in enumerate(models):
        label = _stars(float(p_values.get(model_id, 1.0)))
        if label:
            ax.text(x, high + 0.018 * span, label, ha="center", va="bottom",
                    fontsize=10.0, fontweight="bold")


def _draw_motif_statistics(ax_g: plt.Axes, ax_h: plt.Axes, motif_root: Path) -> dict:
    inter = pd.read_csv(motif_root / "interictal_per_patient.csv")
    inter = inter[inter.cell == "rnn"].copy()
    baseline = inter[inter.model == "M0_NO_REC"][["subject", "contact_nll"]].rename(
        columns={"contact_nll": "baseline_nll"}
    )
    inter = inter.merge(baseline, on="subject", how="inner")
    inter["recurrence_gain"] = inter.baseline_nll - inter.contact_nll
    g_models = MOTIF_MODELS + ("C_ORDER_SHUFFLED",)
    _strip_models(
        ax_g, inter, g_models, "recurrence_gain",
        ylabel="Next-contact gain\n(Δ NLL)", title="Interictal learning · n=21",
    )
    inter_summary = json.loads((motif_root / "INTERICTAL_SUMMARY.json").read_text())
    comparisons = inter_summary["statistics"]["rnn"]["comparisons"]
    _add_group_stars(ax_g, g_models, {
        model_id: comparisons[f"{model_id}_vs_M0"]["p_two_sided"]
        for model_id in g_models
    })
    _panel_letter(ax_g, "G", x=-0.20, y=1.065)

    early = pd.read_csv(motif_root / "early_ictal_per_patient_model.csv")
    primary = early.primary.astype(str).str.lower().isin({"true", "1", "1.0"})
    early = early[primary & (early.cell == "rnn") & (early.endpoint == "canonical_full")]
    h_models = ("M0_NO_REC", "M1_DENSE", "M3_FIXED_LOCAL", "M6_SPATIAL_MID", "C_ORDER_SHUFFLED")
    _strip_models(
        ax_h, early, h_models, "all_contact_margin",
        ylabel="Early-ictal margin", title="Frozen-field alignment · n=10",
    )
    _panel_letter(ax_h, "H", x=-0.20, y=1.065)
    return {
        "motif_interictal_patients": int(inter.subject.nunique()),
        "motif_early_ictal_patients": int(early.subject.nunique()),
    }


def plot(canonical_root: Path, analysis: Path, out_dir: Path,
         motif_root: Path, fig3_metadata: Path) -> dict:
    record, model, activation, seizure_idx, extraction = _load_core(
        canonical_root, analysis, fig3_metadata,
    )
    sequence = _sequence_payload(canonical_root, record)
    inter = pd.read_csv(analysis / "interictal_patient_statistics.csv").sort_values("subject")
    ictal = pd.read_csv(analysis / "ictal_patient_statistics.csv")
    ictal = ictal[ictal.group_id == "all_phenotype_matched"].sort_values("subject")
    cohort = pd.read_csv(analysis / "ictal_cohort_statistics.csv")
    manifest = json.loads((analysis / "MODEL_FIELD_MANIFEST.json").read_text())
    if len(inter) != 34 or len(ictal) != 17 or int(ictal.n_seizures.sum()) != 167:
        raise RuntimeError("combined_figure_denominator_mismatch")

    fig = plt.figure(figsize=(14.6, 11.4), facecolor="white")
    outer = fig.add_gridspec(
        3, 12, height_ratios=[0.96, 0.92, 0.86],
        left=0.045, right=0.985, bottom=0.065, top=0.97,
        hspace=0.29, wspace=0.72,
    )
    ax_a = fig.add_subplot(outer[0, 0:3])
    _draw_architecture(ax_a, record, sequence)
    _draw_sequences(outer[0, 3:10], sequence)
    ax_c = fig.add_subplot(outer[0, 10:12])
    _paired_compact(
        ax_c, inter.native_model, inter.static_only,
        ("RNN", "Baseline"), (RED, GREY),
        "Propagation correlation", "Interictal · n=34",
        float(manifest["interictal"]["wilcoxon_one_sided_native_gt_static_p"]),
    )
    _panel_letter(ax_c, "C", x=-0.24, y=1.065)

    field_stats = _draw_fields(outer[1, 0:10], record, model, activation, seizure_idx)
    ax_e = fig.add_subplot(outer[1, 10:12])
    stat = cohort[cohort.group_id == "all_phenotype_matched"].iloc[0]
    _paired_compact(
        ax_e, ictal.data, ictal.channel_null_median,
        ("RNN", "Shuffle"), (RED, GREY),
        "Field concordance |r|", "Early ictal · n=17",
        float(stat.wilcoxon_one_sided_data_gt_null_p),
    )
    _panel_letter(ax_e, "E", x=-0.24, y=1.065)

    _draw_motif_layouts(outer[2, 0:6], motif_root)
    ax_g = fig.add_subplot(outer[2, 6:9])
    ax_h = fig.add_subplot(outer[2, 9:12])
    motif_stats = _draw_motif_statistics(ax_g, ax_h, motif_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "topic5_figure6_rnn_full_cohort"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    summary = {
        "contract": "topic5_figure6_rnn_full_cohort_v0_2_candidate",
        "panel_a_model": "patient-specific linear-state RNN; hidden_size=32",
        "panel_b_subject": SUBJECT,
        "panel_b_events_per_mode_per_source": N_EVENT_COLUMNS,
        "panel_c_subjects": 34,
        "panel_d_subject": SUBJECT,
        "panel_d_seizure_idx": seizure_idx,
        "panel_d_exact_target": {
            "window_sec": extraction["clinical_window_sec"],
            "band_hz": extraction["band_hz"],
        },
        "panel_d_rnn_empirical_ta_spearman": field_stats["ta_spearman"],
        "panel_d_rnn_empirical_tb_spearman": field_stats["tb_spearman"],
        "panel_e_subjects": 17,
        "panel_e_seizures": 167,
        "panel_e_one_sided_p": float(stat.wilcoxon_one_sided_data_gt_null_p),
        "panel_f_h_source": str(motif_root),
        **motif_stats,
    }
    (analysis / "FIGURE6_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        analysis / "e1146_panel_b_sequence_source.npz",
        contact_order=sequence["names"],
        observed_ta=sequence["observed_a"], generated_ta=sequence["generated_a"],
        observed_tb=sequence["observed_b"], generated_tb=sequence["generated_b"],
        event_indices_ta=sequence["event_indices_a"],
        event_indices_tb=sequence["event_indices_b"],
    )
    assets = {}
    for suffix in (".png", ".pdf", ".svg"):
        path = stem.with_suffix(suffix)
        assets[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (analysis / "FIGURE6_COMPLETE.json").write_text(json.dumps({
        "status": "COMPLETE",
        "contract": summary["contract"],
        "figure_assets_sha256": assets,
        "source_artifact": "e1146_panel_b_sequence_source.npz",
    }, indent=2) + "\n")
    readme = out_dir / "README.md"
    existing = readme.read_text() if readme.exists() else "# Topic 5 RNN figures\n"
    marker = "### topic5_figure6_rnn_full_cohort.png"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n\n"
    existing += f"""### topic5_figure6_rnn_full_cohort.png / .pdf / .svg

Figure 6 八面板候选图。A 将 E1146 真实电极杆布局直接叠加在 patient-specific linear-state RNN 示意上，叠加触点颜色来自同一 RNN rollout 的输出 rank；左右竖向 rank bars 分别表示输入和输出，内部节点按非规则循环网络排布。B 用 E1146 的 TA/TB data 与 same-start RNN rollout 逐列比较，并在面板右侧独立放置 rank 色条。C 是 34 人间期自由生成相对静态基线的患者级统计。D 只保留 RNN TA、RNN TB 与 canonical Fig3-B 的 E1146 seizure {seizure_idx} early-ictal field；每个方形 field 后均使用独立竖直色条，后者严格为 clinical onset 0–10 s、1–150 Hz broadband energy。E 是 17 人/167 seizures 的冻结线性状态 RNN field 相对全通道同步 shuffle。

F–H 单独回顾 v0.4 connectivity-motif 结果：F 为 E1146 的 Dense、Sparse、Local、Spatial+cost 连接布局，统一用 edge-length 色标突出局部与非局部连接差异；G 为 21 人 next-contact recurrence gain；H 为旧 v0.4 exact-join 10 人的 early-ictal null-relative margin。H 不能与 E 的 17 人分母混用，也没有显示某一种空间连接显著优于其他 recurrent motifs。

**关注点**：主证据链把“患者内间期传播可生成”和“冻结场跨状态对应”放在前两行；第三行把多种连接约束的计算充分性与尚未分辨的 early-ictal motif specificity 分开呈现。
"""
    readme.write_text(existing)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument("--analysis", type=Path,
                        default=ROOT / "results/topic5_rnn_full_cohort_field_transfer_v0_1")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "results/paper-ready-figure/fig6_rnn_full_cohort_field_transfer/figures")
    parser.add_argument(
        "--motif-root", type=Path,
        default=Path("/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-rnn-motif-cross-state-v0-4/results/topic5_rnn_motif_cross_state_benchmark_v0_4"),
    )
    parser.add_argument(
        "--fig3-metadata", type=Path,
        default=Path("/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig3b_interictal_ictal_shared_field/figures/epilepsiae_1146_seizure_02_interictal_ictal_shared_field_metadata.json"),
    )
    args = parser.parse_args()
    print(json.dumps(plot(
        args.canonical_root, args.analysis, args.out_dir,
        args.motif_root, args.fig3_metadata,
    ), indent=2))


if __name__ == "__main__":
    main()
