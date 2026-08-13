#!/usr/bin/env python3
"""Paper-ready Figure 6 candidate for the full-tissue LBSS-RNN v0.3.

The first two rows connect patient-specific interictal generation to the frozen
early-ictal field benchmark.  The bottom row asks the spatial mechanism
question: whether task-selected nonlocal shortcuts add distal propagation and
whether attenuating them changes the frozen cross-state field.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.plot_contact_plane_static import _smooth_rank_field_mm  # noqa: E402
from scripts.plot_topic5_interictal_template_ab_fields import (  # noqa: E402
    _canonical_transverse_sign,
)
from scripts.plot_topic5_lbss_figure6_v0_2 import (  # noqa: E402
    BLUE,
    DARK,
    GRAY,
    RED,
    add_scale_bar,
    assert_no_label_overlap,
    grid_letter,
)


SUBJECT = "epilepsiae_1146"
FIT_ID = f"{SUBJECT}__shared"
L0 = "L0_LOCAL_ONLY"
L1 = "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL"
L2 = "L2_LOCAL_PLUS_RANDOM_LR"
L3 = "L3_LOCAL_PLUS_LEARNED_LR"
SHUFFLE = "C_L3_ORDER_SHUFFLED"
ARMS = (L0, L1, L2, L3)
ARM_LABEL = {L0: "Local", L1: "+ local", L2: "+ random", L3: "+ selected", SHUFFLE: "Shuffle"}
ARM_COLOR = {L0: "#7c858b", L1: "#6592a2", L2: "#9b8468", L3: RED, SHUFFLE: "#b7b7b7"}
TIMING_CMAP = "viridis"
ENERGY_CMAP = "Blues"
DISPLAY_SIGMA_MM = 6.0


def normalized_event_matrix(rows: list[np.ndarray], n_contacts: int) -> np.ndarray:
    matrix = np.full((n_contacts, len(rows)), np.nan)
    for column, rank in enumerate(rows):
        finite = rank >= 0
        if finite.any():
            top = max(1.0, float(rank[finite].max()))
            matrix[finite, column] = rank[finite] / top
    return matrix


def generated_rank(sequence: list[list[int]], n_contacts: int) -> np.ndarray:
    rank = np.full(n_contacts, -1, int)
    for index, contacts in enumerate(sequence):
        rank[np.asarray(contacts, int)] = index
    return rank


def _shaft_key(contact: str) -> tuple[str, int]:
    match = re.match(r"^(.*?)(\d+)$", str(contact))
    return (match.group(1), int(match.group(2))) if match else (str(contact), 0)


@torch.no_grad()
def _heldout_node_activity(out: Path, fit_id: str, arm: str, seed: int = 0) -> np.ndarray:
    """Mean absolute latent activity on held-out interictal rank steps.

    This is an input-output diagnostic for the representative panel only.  It
    is not used for selecting a model or a node subset, and it is evaluated on
    every tissue node, including nodes with zero direct SEEG support.
    """
    from src.topic5_lbss_rnn_v0_2 import LBSSConfig, LBSSModel, build_pool_contract
    from src.topic5_wiring_economy_rnn import build_event_tensors

    unit = out / "per_fit" / fit_id / arm / f"seed{seed}"
    metrics = json.loads((unit / "metrics.json").read_text())
    provenance = json.loads((out / "cache" / fit_id / "provenance.json").read_text())
    with np.load(out / "cache" / fit_id / "plane.npz", allow_pickle=False) as plane:
        h_operator = np.asarray(plane["H"], np.float32)
        distance = np.asarray(plane["D_mm"], np.float32)
    config = metrics["config"]
    pools = build_pool_contract(
        distance, float(config["density"]), float(config["added_fraction"]),
        float(config.get("r_local_multiplier", 2.0)),
    )
    model = LBSSModel(LBSSConfig(
        arm=arm,
        n_contacts=int(provenance["n_contacts"]),
        n_nodes=int(provenance["n_nodes"]),
        observation_operator=h_operator,
        node_distance_mm=distance,
        local_mask=pools.local_mask,
        extra_local_pool=pools.extra_local_pool,
        nonlocal_pool=pools.nonlocal_pool,
        k_added=pools.k_added,
        seed=int(seed),
        state_dim=int(config["state_dim"]),
    ))
    model.load_state_dict(torch.load(unit / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()
    with np.load(out / "cache" / fit_id / "events.npz", allow_pickle=False) as events:
        keep = np.asarray(events["split"] >= 0)
        ranks = np.asarray(events["ranks"])[keep]
        split = np.asarray(events["split"])[keep]
    tensors = build_event_tensors(ranks)
    test = np.flatnonzero(split == 2)
    total = torch.zeros(model.n_nodes, dtype=torch.float64)
    denominator = 0.0
    for begin in range(0, len(test), 256):
        chosen = torch.as_tensor(test[begin:begin + 256])
        batch_x = tensors["x"][chosen]
        batch_valid = tensors["valid"][chosen]
        hidden = torch.zeros(len(chosen), model.n_nodes * model.state_dim)
        for step in range(batch_x.shape[1]):
            hidden = model._step(hidden, batch_x[:, step])
            active = batch_valid[:, step].double().view(-1, 1, 1)
            state = hidden.reshape(len(chosen), model.n_nodes, model.state_dim).abs().double()
            total += (state * active).sum(dim=(0, 2)) / model.state_dim
            denominator += float(active.sum())
    return (total / max(denominator, 1.0)).numpy()


def _align_tissue_plane_to_frozen_display(
    nodes: np.ndarray,
    contacts: np.ndarray,
    contact_names: list[str],
    canonical_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly align the latent plane to the frozen Figure-2/3 orientation.

    The latent cache and the accepted field renderer use the same propagation
    plane, but the transverse SVD axis can differ by a sign.  The transform is
    fitted from contact geometry only; activity, rank and ictal values are not
    read.  Reflections are allowed because the transverse sign is a display
    convention rather than a biological direction.
    """
    record = json.loads((
        canonical_root
        / "results/interictal_propagation_masked/template_gradient_fields/"
        "per_subject/epilepsiae_1146.json"
    ).read_text())["interictal_field"]
    order = [str(value) for value in record["contact_order"]]
    plane = record["planes"]["shared"]
    target = np.asarray(plane["points"], float) * float(plane["scale_mm"])
    target[:, 1] *= _canonical_transverse_sign(plane["w"])
    target = target[np.asarray([order.index(name) for name in contact_names], int)]
    source_center = contacts.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _, right = np.linalg.svd(
        (contacts - source_center).T @ (target - target_center),
        full_matrices=False,
    )
    rotation = left @ right
    aligned_contacts = (contacts - source_center) @ rotation + target_center
    aligned_nodes = (nodes - source_center) @ rotation + target_center
    if float(np.sqrt(np.mean((aligned_contacts - target) ** 2))) > 1e-3:
        raise RuntimeError("latent plane cannot be aligned to frozen Figure-2/3 geometry")
    return aligned_nodes, aligned_contacts


def draw_full_tissue_graph(
    ax: plt.Axes,
    out: Path,
    fit_id: str,
    canonical_root: Path,
) -> None:
    plane = np.load(out / "cache" / fit_id / "plane.npz", allow_pickle=False)
    graph = np.load(out / "per_fit" / fit_id / L3 / "seed0" / "graph.npz", allow_pickle=False)
    xy = np.asarray(plane["nodes_xy_mm"], float)
    contacts = np.asarray(plane["contacts_xy_mm"], float)
    provenance = json.loads((out / "cache" / fit_id / "provenance.json").read_text())
    contact_names = [str(value) for value in provenance["contacts"]]
    xy, contacts = _align_tissue_plane_to_frozen_display(
        xy, contacts, contact_names, canonical_root,
    )
    activity = _heldout_node_activity(out, fit_id, L3, seed=0)
    local = graph["local_mask"].astype(bool)
    added = graph["added_mask"].astype(bool)
    strength = np.asarray(graph["strength"], float)
    local_pairs = np.argwhere(local & np.triu(np.ones_like(local, bool), 1))
    local_pair_strength = np.asarray([
        max(strength[target, source], strength[source, target])
        for target, source in local_pairs
    ])
    n_local_show = min(len(local_pairs), max(40, int(round(0.08 * len(local_pairs)))))
    local_show = local_pairs[np.argsort(local_pair_strength, kind="stable")[-n_local_show:]]
    for target, source in local_show:
        ax.plot(xy[[source, target], 0], xy[[source, target], 1], color="#d2d6d8",
                lw=0.42, alpha=0.34, zorder=1)
    selected = np.argwhere(added)
    # Show only the strongest few shortcuts; the full graph remains in every
    # calculation.  More arrows obscure the real SEEG layout without adding
    # scientific information.
    order = np.argsort(strength[added], kind="stable")[::-1][:min(3, len(selected))]
    centre = np.nanmean(xy, axis=0)
    for rank, index in enumerate(order):
        target, source = selected[index]
        midpoint = 0.5 * (xy[source] + xy[target])
        segment = xy[target] - xy[source]
        perpendicular = np.asarray([-segment[1], segment[0]])
        # Arc away from the graph centre.  ``arc3`` bends opposite to the
        # sign convention of the perpendicular used above, hence the minus.
        bend_sign = -1.0 if np.dot(midpoint - centre, perpendicular) >= 0 else 1.0
        ax.add_patch(FancyArrowPatch(
            xy[source], xy[target], arrowstyle="-|>", mutation_scale=4.5,
            connectionstyle=f"arc3,rad={bend_sign * (0.23 + 0.045 * rank):.3f}",
            color=RED, lw=1.05, alpha=0.80, zorder=2,
        ))
    low, high = np.nanpercentile(activity, [2, 98])
    if not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(activity)), float(np.nanmax(activity) + 1e-12)
    activity_display = np.clip((activity - low) / max(high - low, 1e-12), 0, 1)
    nodes = ax.scatter(
        xy[:, 0], xy[:, 1], s=13, c=activity_display, cmap="cividis", vmin=0, vmax=1,
        edgecolor="white", linewidth=0.18, alpha=0.93, zorder=3,
    )
    shafts: dict[str, list[tuple[int, int]]] = {}
    for index, name in enumerate(contact_names):
        shaft, number = _shaft_key(name)
        shafts.setdefault(shaft, []).append((number, index))
    for members in shafts.values():
        indices = [index for _, index in sorted(members)]
        ax.plot(contacts[indices, 0], contacts[indices, 1], color=DARK, lw=1.25, zorder=4)
    # The representative E1146 first rank contains one contact.  That contact
    # is highlighted on the input bar and at the SEEG port, while the outlined
    # tissue nodes show that H^T distributes the input across a local node set
    # rather than injecting it into a single recurrent unit.
    with np.load(out / "cache" / fit_id / "events.npz", allow_pickle=False) as events:
        test = np.flatnonzero(events["split"] == 2)
        example = int(test[np.flatnonzero(events["mode"][test] == 0)[0]])
        observed_rank = np.asarray(events["ranks"][example], int)
        input_contacts = np.flatnonzero(observed_rank == 0)
    input_color = mpl.colormaps[TIMING_CMAP](0.0)
    input_support = np.flatnonzero(np.asarray(plane["H"], float)[input_contacts].sum(axis=0) > 0)
    ax.scatter(
        xy[input_support, 0], xy[input_support, 1], s=27, facecolor="none",
        edgecolor=input_color, linewidth=1.05, alpha=0.95, zorder=4,
    )
    contact_face = np.full((len(contacts), 4), mpl.colors.to_rgba("white"))
    contact_face[input_contacts] = input_color
    ax.scatter(
        contacts[:, 0], contacts[:, 1], s=31, facecolor=contact_face, edgecolor=DARK,
        linewidth=0.9, zorder=5,
    )
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    add_scale_bar(ax, xy)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        handles=[
            Line2D([], [], color="#d2d6d8", lw=1.0, label="Local backbone"),
            Line2D([], [], color=RED, lw=1.4, label="Selected shortcut"),
            Line2D([], [], marker="o", lw=0, markerfacecolor="white", markeredgecolor=DARK,
                   markersize=5.5, label="SEEG contact"),
        ],
        loc="upper left", bbox_to_anchor=(-0.02, -0.01), fontsize=7.2,
        frameon=False, handlelength=1.45, labelspacing=0.28, borderpad=0.0,
    )
    color_axis = ax.inset_axes([0.905, 0.68, 0.022, 0.22])
    colorbar = ax.figure.colorbar(nodes, cax=color_axis, orientation="vertical")
    colorbar.set_ticks([0, 1], labels=["0", "1"])
    colorbar.ax.tick_params(labelsize=6.8, length=1.5, pad=1)
    colorbar.ax.set_title("|h|", fontsize=7.4, pad=2)

    # A literal first-rank input and a model-generated rank output make the
    # recurrent computation visible without mistaking the SEEG contacts for
    # the full set of tissue nodes.
    with gzip.open(
        out / "per_fit" / fit_id / L3 / "seed0" / "heldout_rollouts.json.gz", "rt"
    ) as stream:
        generated_lookup = {
            int(item["kept_event_index"]): item["generated_rank_sets"]
            for item in json.load(stream)
        }
    output_rank = generated_rank(generated_lookup[example], len(contact_names))
    physical_order = np.asarray(sorted(range(len(contact_names)), key=lambda i: _shaft_key(contact_names[i])))
    seed_bar = np.full(len(contact_names), np.nan)
    seed_bar[observed_rank == 0] = 0.0
    output_bar = output_rank.astype(float)
    output_bar[output_rank < 0] = np.nan
    if np.isfinite(output_bar).any():
        output_bar /= max(1.0, float(np.nanmax(output_bar)))
    rank_cmap = mpl.colormaps[TIMING_CMAP].copy(); rank_cmap.set_bad("#e7e7e7")
    input_axis = ax.inset_axes([-0.075, 0.25, 0.035, 0.52])
    output_axis = ax.inset_axes([0.965, 0.25, 0.035, 0.52])
    for bar_axis, values in ((input_axis, seed_bar), (output_axis, output_bar)):
        bar_axis.imshow(values[physical_order, None], aspect="auto", cmap=rank_cmap, vmin=0, vmax=1)
        bar_axis.set_xticks([]); bar_axis.set_yticks([])
        for spine in bar_axis.spines.values(): spine.set_visible(False)
    ax.annotate("", xy=(0.05, 0.51), xytext=(-0.025, 0.51), xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "lw": 0.85, "color": DARK})
    ax.annotate("", xy=(1.035, 0.51), xytext=(0.93, 0.51), xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "lw": 0.85, "color": DARK})


def draw_event_reproduction_v3(
    fig: plt.Figure,
    spec,
    out: Path,
    fit_id: str,
    canonical_root: Path,
) -> None:
    # Reserve a narrow blank column after the vertical rank bar so its labels
    # cannot collide with the cohort y-axis in panel C.
    sub = spec.subgridspec(
        2, 4, width_ratios=(1, 1, 0.028, 0.11), wspace=0.09, hspace=0.13,
    )
    axes = np.asarray([[fig.add_subplot(sub[i, j]) for j in range(2)] for i in range(2)])
    events = np.load(out / "cache" / fit_id / "events.npz", allow_pickle=False)
    provenance = json.loads((out / "cache" / fit_id / "provenance.json").read_text())
    keep = events["split"] >= 0
    ranks, split, modes = events["ranks"][keep], events["split"][keep], events["mode"][keep]
    test = np.flatnonzero(split == 2)
    with gzip.open(out / "per_fit" / fit_id / L3 / "seed0" / "heldout_rollouts.json.gz", "rt") as stream:
        rollouts = json.load(stream)
    by_index = {int(row["kept_event_index"]): row for row in rollouts}
    empirical = json.loads((
        canonical_root / "results/interictal_propagation_masked/template_gradient_fields/"
        f"per_subject/{provenance['subject']}.json"
    ).read_text())["interictal_field"]
    empirical_names = [str(value) for value in empirical["contact_order"]]
    take = np.asarray([empirical_names.index(str(value)) for value in provenance["contacts"]], int)
    order_a = np.argsort(np.asarray(empirical["rank_a"], float)[take], kind="stable")
    order_b = np.argsort(np.asarray(empirical["rank_b"], float)[take], kind="stable")
    image = None
    # Both templates must use the same physical contact order.  Sorting each
    # row by its own template would visually straighten both directions and
    # could falsely make a midpoint solution look like bidirectional replay.
    common_contact_order = order_a
    for row_index, (template, color) in enumerate(
        (("A", RED), ("B", BLUE))
    ):
        chosen = [
            int(index) for index in test
            if str(provenance["mode_to_template"].get(str(int(modes[index])), "")).upper() == template
        ][:30]
        observed = [ranks[index] for index in chosen]
        generated = [generated_rank(by_index[index]["generated_rank_sets"], ranks.shape[1]) for index in chosen]
        for column, payload in enumerate((observed, generated)):
            matrix = normalized_event_matrix(payload, ranks.shape[1])[common_contact_order]
            cmap = mpl.colormaps[TIMING_CMAP].copy(); cmap.set_bad("#e7e7e7")
            image = axes[row_index, column].imshow(
                matrix, aspect="auto", interpolation="nearest",
                cmap=cmap, vmin=0, vmax=1, origin="upper",
            )
            axes[row_index, column].set_xticks([]); axes[row_index, column].set_yticks([])
            for spine in axes[row_index, column].spines.values():
                spine.set_visible(False)
        axes[row_index, 0].set_ylabel(
            f"T{template}", color=color, rotation=0, labelpad=12,
            fontsize=11.5, fontweight="bold", va="center",
        )
    axes[0, 0].set_title("Data", fontsize=11, pad=4)
    axes[0, 1].set_title("Generated", fontsize=11, pad=4)
    axes[0, 0].text(
        0.0, 1.20, "E1146", transform=axes[0, 0].transAxes,
        fontsize=10.5, fontweight="bold", ha="left", va="bottom",
    )
    bar_ax = fig.add_subplot(sub[:, 2])
    bar = fig.colorbar(image, cax=bar_ax, orientation="vertical")
    bar.set_ticks([0, 1], labels=["First", "Last"])
    bar.ax.set_title("Rank", fontsize=8.5, pad=3)
    bar.ax.tick_params(labelsize=8, length=2, pad=2)


def paired_test(values: np.ndarray, alternative: str = "greater") -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values) & (np.abs(values) > 1e-9)]
    if not len(values):
        return 1.0
    return float(wilcoxon(values, alternative=alternative, method="auto").pvalue)


def stars(p: float) -> str:
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


def holm(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def paired_axis(
    ax: plt.Axes,
    left: np.ndarray,
    right: np.ndarray,
    labels: tuple[str, str],
    colors: tuple[str, str],
    ylabel: str,
    p_value: float,
) -> None:
    left, right = np.asarray(left, float), np.asarray(right, float)
    keep = np.isfinite(left) & np.isfinite(right)
    left, right = left[keep], right[keep]
    for a, b in zip(left, right):
        ax.plot([0, 1], [a, b], color="#c2c7ca", lw=0.65, alpha=0.72, zorder=1)
    ax.scatter(np.zeros(len(left)), left, s=19, color=colors[0], edgecolor="white", lw=0.4, zorder=3)
    ax.scatter(np.ones(len(right)), right, s=19, color=colors[1], edgecolor="white", lw=0.4, zorder=3)
    for x, values in ((0, left), (1, right)):
        median = float(np.nanmedian(values))
        ax.plot([x - 0.18, x + 0.18], [median, median], color="#141414", lw=1.8, zorder=4)
    lo = float(np.nanmin(np.r_[left, right])); hi = float(np.nanmax(np.r_[left, right]))
    span = max(hi - lo, 0.08)
    label = stars(p_value)
    if label:
        y = hi + 0.10 * span
        ax.plot([0, 0, 1, 1], [y - 0.02 * span, y, y, y - 0.02 * span], color="#1d1d1d", lw=0.8)
        ax.text(0.5, y + 0.02 * span, label, ha="center", va="bottom", fontsize=12, fontweight="bold")
        hi = y + 0.12 * span
    ax.set_xlim(-0.36, 1.36); ax.set_ylim(lo - 0.08 * span, hi + 0.04 * span)
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)


def patient_strip(
    ax: plt.Axes,
    groups: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    ylabel: str,
    p_values: list[float] | None = None,
) -> None:
    rng = np.random.default_rng(20260812)
    pooled = []
    for index, (group, color) in enumerate(zip(groups, colors)):
        values = np.asarray(group, float); values = values[np.isfinite(values)]
        pooled.extend(values.tolist())
        jitter = rng.uniform(-0.13, 0.13, len(values))
        ax.scatter(index + jitter, values, s=16, color=color, alpha=0.72, edgecolor="none")
        if len(values):
            median = float(np.median(values))
            ax.plot([index - 0.20, index + 0.20], [median, median], color="#161616", lw=1.7)
    ax.axhline(0, color="#8a9297", lw=0.75, zorder=0)
    ax.set_xticks(range(len(labels)), labels, rotation=28, ha="right")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    if p_values is not None and pooled:
        lo, hi = ax.get_ylim(); span = max(hi - lo, 1e-6)
        ax.set_ylim(lo, hi + 0.10 * span)
        for index, p_value in enumerate(p_values):
            label = stars(float(p_value))
            if label:
                ax.text(index, hi + 0.015 * span, label, ha="center", va="bottom", fontsize=11, fontweight="bold")


def field_geometry(field: dict) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    plane = field["planes"]["shared"]
    points = np.asarray(plane["points"], float) * float(plane["scale_mm"])
    # Reuse the frozen Figure-3 display orientation.  The transverse SVD axis
    # otherwise has an arbitrary sign, which can silently mirror only the RNN
    # panel even when the scored vectors are identical.
    points[:, 1] *= _canonical_transverse_sign(plane["w"])
    pad = 5.0
    return points, (float(points[:, 0].min() - pad), float(points[:, 0].max() + pad)), (
        float(points[:, 1].min() - pad), float(points[:, 1].max() + pad)
    )


def draw_field(
    ax: plt.Axes,
    points: np.ndarray,
    values: np.ndarray,
    support: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str,
    title_color: str,
    show_y: bool,
) -> mpl.image.AxesImage:
    X, Y, smooth, _, _ = _smooth_rank_field_mm(
        points[:, 0], points[:, 1], values, support, xlim, ylim, DISPLAY_SIGMA_MM
    )
    image = ax.imshow(
        smooth, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()],
        aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear",
    )
    finite = np.isfinite(values) & np.isfinite(support) & (support > 0)
    ax.scatter(
        points[finite, 0], points[finite, 1], c=values[finite], cmap=cmap,
        vmin=vmin, vmax=vmax, s=20, edgecolor="white", lw=0.65, zorder=3,
    )
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(title, fontsize=11, color=title_color, fontweight="bold", pad=4)
    ax.set_xticks([])
    if show_y:
        ax.set_ylabel("Transverse (mm)")
    else:
        ax.set_yticks([])
    return image


def draw_cross_state_fields(fig: plt.Figure, spec, out: Path, canonical_root: Path) -> dict:
    # Give the vertical colourbar tick labels their own gutters.  On the final
    # 15.4-inch canvas, the old 0.035 bars placed Early/Late over the next map.
    sub = spec.subgridspec(
        1, 6, width_ratios=(1, 0.075, 1, 0.075, 1, 0.055), wspace=0.24
    )
    axes = [fig.add_subplot(sub[0, index]) for index in (0, 2, 4)]
    bars = [fig.add_subplot(sub[0, index]) for index in (1, 3, 5)]
    record = json.loads((canonical_root / "results/interictal_propagation_masked/template_gradient_fields/per_subject/epilepsiae_1146.json").read_text())
    field = record["interictal_field"]
    order = [str(value) for value in field["contact_order"]]
    model_path = out / "model_fields/intact/per_patient/epilepsiae_1146/L3_LOCAL_PLUS_LEARNED_LR.npz"
    with np.load(model_path, allow_pickle=False) as model:
        names = model["contacts"].astype(str).tolist()
        lookup_a = dict(zip(names, np.asarray(model["A_canonical_full"], float)))
        lookup_b = dict(zip(names, np.asarray(model["B_canonical_full"], float)))
        support_lookup_a = dict(zip(names, np.asarray(model["A_participation"], float)))
        support_lookup_b = dict(zip(names, np.asarray(model["B_participation"], float)))
    rank_a = 1.0 - np.asarray([lookup_a[name] for name in order], float)
    rank_b = 1.0 - np.asarray([lookup_b[name] for name in order], float)
    support_a = np.asarray([support_lookup_a[name] for name in order], float)
    support_b = np.asarray([support_lookup_b[name] for name in order], float)
    with np.load(out / "early_ictal/e1146_early_ictal_broadband_1_150.npz", allow_pickle=False) as target:
        target_names = target["contact_order"].astype(str).tolist()
        target_lookup = dict(zip(target_names, np.asarray(target["activation"], float)))
        energy = np.asarray([target_lookup.get(name, np.nan) for name in order], float)
        n_seizures = int(np.asarray(target["n_seizures"]).item())
    points, xlim, ylim = field_geometry(field)
    timing_images = [
        draw_field(axes[0], points, rank_a, support_a, xlim, ylim,
                   cmap=TIMING_CMAP, vmin=0, vmax=1, title="RNN TA", title_color=RED, show_y=True),
        draw_field(axes[1], points, rank_b, support_b, xlim, ylim,
                   cmap=TIMING_CMAP, vmin=0, vmax=1, title="RNN TB", title_color=BLUE, show_y=False),
    ]
    e_min, e_max = float(np.nanmin(energy)), float(np.nanmax(energy))
    energy_image = draw_field(
        axes[2], points, energy, np.isfinite(energy).astype(float), xlim, ylim,
        cmap=ENERGY_CMAP, vmin=e_min, vmax=e_max,
        title="Early-ictal broadband", title_color=DARK, show_y=False,
    )
    for image, cax in zip(timing_images, bars[:2]):
        bar = fig.colorbar(image, cax=cax)
        bar.set_ticks([0, 1], labels=["Early", "Late"])
        bar.ax.tick_params(labelsize=8, length=2, pad=1)
    ebar = fig.colorbar(energy_image, cax=bars[2])
    ebar.ax.set_title("z", fontsize=8, pad=2)
    ebar.ax.tick_params(labelsize=8, length=2, pad=1)
    return {"subject": SUBJECT, "n_strict_broadband_seizures": n_seizures}


def draw_contact_space_interictal(ax: plt.Axes, analysis: Path) -> dict:
    frame = pd.read_csv(analysis / "interictal_patient_statistics.csv").sort_values("subject")
    p_value = paired_test(frame.native_model.to_numpy() - frame.static_only.to_numpy(), "greater")
    paired_axis(
        ax, frame.native_model, frame.static_only, ("RNN", "Static"), (RED, GRAY),
        "Propagation correlation", p_value,
    )
    ax.set_title("Interictal · n=34", fontsize=11.5, fontweight="bold", pad=5)
    return {"n": int(frame.subject.nunique()), "p_one_sided": p_value}


def draw_cross_state_statistics(fig: plt.Figure, spec, out: Path, analysis: Path) -> dict:
    sub = spec.subgridspec(1, 2, wspace=0.70)
    ax_full, ax_spatial = (fig.add_subplot(sub[0, i]) for i in range(2))
    contact = pd.read_csv(analysis / "ictal_patient_statistics.csv")
    contact = contact[contact.group_id.eq("all_phenotype_matched")].sort_values("subject")
    # The frozen full-cohort estimand is a patient-level observed-minus-null
    # margin.  Raw paired coordinates are shown only to keep the visual link to
    # the synchronized channel-label shuffle explicit.
    contact_p = paired_test(contact.margin.to_numpy(float), "greater")
    paired_axis(
        ax_full, contact.data, contact.channel_null_median,
        ("RNN", "Shuffle"), (RED, GRAY), "Field concordance |r|", contact_p,
    )
    ax_full.set_title("Contact-space · n=17", fontsize=10.8, fontweight="bold", pad=4)

    early = pd.read_csv(out / "early_ictal/early_ictal_per_patient_condition.csv")
    spatial = early[
        early.family.eq("intact") & early.endpoint.eq("canonical_full")
        & early.arm.isin((L0, L1, L2, L3, SHUFFLE))
    ].copy()
    spatial["margin"] = spatial.observed - spatial.all_contact_null_median
    arm_order = (L0, L1, L2, L3, SHUFFLE)
    groups = [spatial.loc[spatial.arm.eq(arm), "margin"].to_numpy(float) for arm in arm_order]
    # Spatial-arm stars are a model-family statement, so keep them aligned to
    # the formal two-sided patient-level inference rather than using a more
    # permissive directional display-only test.
    spatial_p = [paired_test(values, "two-sided") for values in groups]
    spatial_q = holm(spatial_p)
    patient_strip(
        ax_spatial, groups, ["Local", "+local", "+random", "+selected", "Order\nshuffle"],
        [ARM_COLOR[arm] for arm in arm_order], "Field margin over channel shuffle", spatial_q,
    )
    ax_spatial.set_title("Spatial · n=12", fontsize=10.8, fontweight="bold", pad=4)
    return {
        "contact_space_n": int(contact.subject.nunique()), "contact_space_p": contact_p,
        "spatial_n": int(spatial.subject.nunique()),
        "spatial_p_vs_channel_shuffle": dict(zip(arm_order, spatial_p)),
        "spatial_holm_q_vs_channel_shuffle": dict(zip(arm_order, spatial_q)),
        "spatial_model_margins": {
            arm: float(np.nanmedian(values)) for arm, values in zip(arm_order, groups)
        },
    }


def draw_distal_contrasts(fig: plt.Figure, spec, out: Path) -> dict:
    sub = spec.subgridspec(1, 2, wspace=0.82)
    axes = [fig.add_subplot(sub[0, i]) for i in range(2)]
    frame = pd.read_csv(out / "interictal_patient_contrasts.csv")
    controls = (L0, L1, L2, SHUFFLE)
    contrast_stems = {
        L0: "L3_vs_L0",
        L1: "L3_vs_L1",
        L2: "L3_vs_L2",
        SHUFFLE: "L3_vs_shuffle",
    }
    output = {}
    for axis, suffix, title in zip(axes, ("all", "distal"), ("All transitions", "Distal transitions")):
        names = [f"{contrast_stems[arm]}_{suffix}" for arm in controls]
        groups = [frame.loc[frame.contrast.eq(name), "gain_nats"].to_numpy(float) for name in names]
        if any(len(values) == 0 or not np.isfinite(values).all() for values in groups):
            raise RuntimeError(
                f"panel G contrast inventory is missing or nonfinite for {suffix}: {names}"
            )
        # The three matched topology contrasts form one prespecified family.
        # The order-shuffle contrast is an information control and remains a
        # separate two-sided test.
        raw = [paired_test(values, "two-sided") for values in groups[:3]]
        adjusted = holm(raw) + [paired_test(groups[3], "two-sided")]
        patient_strip(
            axis, groups, [ARM_LABEL[arm] for arm in controls], [ARM_COLOR[arm] for arm in controls],
            "Selected-shortcut gain (nats)", adjusted,
        )
        if suffix == "distal":
            # The two facets share units; one y label is sufficient and avoids
            # placing the second label over the first facet's data region.
            axis.set_ylabel("")
        else:
            # Keep the shared y label inside panel G's own column rather than
            # over the rightmost summary axis of panel F.
            axis.yaxis.set_label_coords(-0.10, 0.5)
        axis.set_title(title, fontsize=10.8, fontweight="bold", pad=4)
        output[suffix] = {name: {"median": float(np.nanmedian(values)), "p": float(p)}
                          for name, values, p in zip(names, groups, adjusted)}
    return output


def draw_pathway_panel(fig: plt.Figure, spec, out: Path) -> dict:
    sub = spec.subgridspec(1, 3, width_ratios=(1, 1, 1.05), wspace=0.58)
    axes = [fig.add_subplot(sub[0, index]) for index in range(3)]
    plane = np.load(out / "cache" / FIT_ID / "plane.npz", allow_pickle=False)
    xy = np.asarray(plane["contacts_xy_mm"], float)
    n_contacts = len(xy)
    payloads = []
    for arm in (L3, SHUFFLE):
        payloads.append(np.load(
            out / "pathway_analysis/per_patient" / SUBJECT / f"{arm}.npz",
            allow_pickle=False,
        ))
    for ax, payload, title in zip(axes[:2], payloads, ("True order", "Order shuffle")):
        pattern = np.asarray(payload["effective_pattern"], float)
        source = np.clip(pattern[:n_contacts], 0, None)
        target = np.clip(pattern[n_contacts:], 0, None)
        source = source / max(float(np.nanmax(source)), 1e-12)
        target = target / max(float(np.nanmax(target)), 1e-12)
        ax.scatter(xy[:, 0], xy[:, 1], s=20 + 310 * source, color=BLUE, alpha=0.70)
        ax.scatter(xy[:, 0], xy[:, 1], s=12 + 240 * target,
                   facecolor="none", edgecolor=RED, lw=1.0)
        ax.set_title(title, fontsize=10.8, fontweight="bold", pad=4)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        add_scale_bar(ax, xy)
        for spine in ax.spines.values():
            spine.set_visible(False)
    comparison = pd.read_csv(out / "pathway_analysis/true_vs_shuffle_patient_patterns.csv")
    groups = [
        comparison.endpoint_dissimilarity_beyond_proposal.to_numpy(float),
        comparison.effective_dissimilarity_beyond_proposal.to_numpy(float),
    ]
    p_values = [paired_test(values, "two-sided") for values in groups]
    q_values = holm(p_values)
    patient_strip(
        axes[2], groups, ["Endpoints", "Influence"], [GRAY, RED],
        "Order-specific change", q_values,
    )
    return {
        "n_patients": int(comparison.subject.nunique()),
        "endpoint_p": p_values[0], "influence_p": p_values[1],
        "endpoint_holm_q": q_values[0], "influence_holm_q": q_values[1],
    }


def draw_attenuation(fig: plt.Figure, spec, out: Path) -> dict:
    sub = spec.subgridspec(1, 2, wspace=0.92)
    ax_inter, ax_ictal = (fig.add_subplot(sub[0, i]) for i in range(2))
    auc = pd.read_csv(out / "attenuation/attenuation_patient_auc.csv")
    selected = auc[
        auc.target.eq("L3_ADDED") & auc.inferential_eligible.astype(bool)
    ].sort_values("subject")
    matched = auc[
        auc.target.eq("L3_MATCHED_LOCAL") & auc.inferential_eligible.astype(bool)
    ].sort_values("subject")
    common = sorted(set(selected.subject) & set(matched.subject))
    selected = selected.set_index("subject").loc[common]
    matched = matched.set_index("subject").loc[common]
    dd_p = paired_test(
        selected.auc_distal_selectivity.to_numpy() - matched.auc_distal_selectivity.to_numpy(),
        "two-sided",
    )
    paired_axis(
        ax_inter, selected.auc_distal_selectivity, matched.auc_distal_selectivity,
        ("Nonlocal", "Local"), (RED, "#7c858b"), "Distal-selectivity AUC", dd_p,
    )
    # The bottom-row panels are intentionally compact. Anchor this rotated
    # label to panel H instead of letting Matplotlib place it over panel G.
    ax_inter.yaxis.set_label_coords(-0.10, 0.5)
    ax_inter.set_title(
        f"Interictal attenuation · n={len(common)}",
        fontsize=10.8, fontweight="bold", pad=4,
    )

    early = pd.read_csv(out / "early_ictal/early_ictal_per_patient_condition.csv")
    intact = early[
        early.family.eq("intact") & early.arm.eq(L3) & early.endpoint.eq("seed_removed")
    ].set_index("subject").all_contact_margin
    alphas = (0.25, 0.50, 0.75, 1.00)
    attenuated = []
    early_subjects = intact.index
    for alpha in alphas:
        atten = early[
            early.family.eq("attenuated") & early.target.eq("L3_ADDED")
            & np.isclose(early.alpha, alpha) & early.endpoint.eq("seed_removed")
        ].set_index("subject").all_contact_margin
        attenuated.append(atten)
        early_subjects = early_subjects.intersection(atten.index)
    early_subjects = early_subjects.sort_values()
    doses = [
        (intact.loc[early_subjects] - atten.loc[early_subjects]).to_numpy(float)
        for atten in attenuated
    ]
    for patient_index in range(min(len(values) for values in doses)):
        ax_ictal.plot(alphas, [values[patient_index] for values in doses], color="#c5c9cc", lw=0.55, alpha=0.65)
    medians = [float(np.nanmedian(values)) for values in doses]
    ax_ictal.plot(alphas, medians, color=RED, lw=2.0, marker="o", ms=4.2)
    ax_ictal.axhline(0, color="#8a9297", lw=0.75)
    ax_ictal.set_xticks(alphas, ["25", "50", "75", "100"])
    ax_ictal.set_xlabel("Nonlocal attenuation (%)")
    ax_ictal.set_ylabel("Cross-state loss")
    ax_ictal.set_title("Early-ictal field", fontsize=10.8, fontweight="bold", pad=4)
    ax_ictal.spines[["top", "right"]].set_visible(False)
    dose_auc = np.trapz(np.column_stack(doses), x=np.asarray(alphas), axis=1)
    dose_p = paired_test(dose_auc, "two-sided")
    if stars(dose_p):
        ax_ictal.text(0.98, 0.97, stars(dose_p), transform=ax_ictal.transAxes,
                      ha="right", va="top", fontsize=12, fontweight="bold")
    return {
        "interictal_double_dissociation_n": len(common), "interictal_p": dd_p,
        "early_dose_n": int(len(early_subjects)), "early_dose_auc_p": dose_p,
        "early_dose_medians": dict(zip([str(value) for value in alphas], medians)),
    }


def required_outputs(out: Path, analysis: Path) -> None:
    required = (
        out / "PIPELINE_COMPLETE.json",
        out / "interictal_patient_contrasts.csv",
        out / "PATHWAY_ANALYSIS_COMPLETE.json",
        out / "attenuation/attenuation_patient_auc.csv",
        out / "early_ictal/early_ictal_per_patient_condition.csv",
        out / "early_ictal/e1146_early_ictal_broadband_1_150.npz",
        analysis / "interictal_patient_statistics.csv",
        analysis / "ictal_patient_statistics.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Figure 6 inputs are incomplete:\n" + "\n".join(missing))


def plot_pretarget_preview(
    out: Path,
    analysis: Path,
    canonical_root: Path,
    destination: Path,
) -> Path:
    """Render the target-free A--C row for layout review."""
    required = (
        out / "FORMAL_TRAINING_COMPLETE.json",
        out / "per_fit" / FIT_ID / L3 / "seed0" / "DONE.json",
        analysis / "interictal_patient_statistics.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("pretarget Figure 6 preview inputs are incomplete:\n" + "\n".join(missing))
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.7,
        "axes.labelsize": 11.5, "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(15.4, 3.35), facecolor="white")
    grid = fig.add_gridspec(
        1, 12, left=0.045, right=0.985, bottom=0.17, top=0.93, wspace=0.78,
    )
    ax_a = fig.add_subplot(grid[0, 0:3])
    draw_full_tissue_graph(ax_a, out, FIT_ID, canonical_root)
    draw_event_reproduction_v3(fig, grid[0, 3:10], out, FIT_ID, canonical_root)
    ax_c = fig.add_subplot(grid[0, 10:12])
    draw_contact_space_interictal(ax_c, analysis)
    for label, cell in zip("ABC", (grid[0, 0:3], grid[0, 3:10], grid[0, 10:12])):
        grid_letter(fig, cell, label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination


def plot(out: Path, analysis: Path, canonical_root: Path, destination: Path) -> dict:
    required_outputs(out, analysis)
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.7,
        "axes.labelsize": 11.5, "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(15.4, 11.5), facecolor="white")
    grid = fig.add_gridspec(
        3, 12, height_ratios=(0.92, 0.88, 0.90),
        left=0.045, right=0.985, bottom=0.07, top=0.97,
        wspace=0.78, hspace=0.40,
    )

    ax_a = fig.add_subplot(grid[0, 0:3])
    draw_full_tissue_graph(ax_a, out, FIT_ID, canonical_root)
    draw_event_reproduction_v3(fig, grid[0, 3:10], out, FIT_ID, canonical_root)
    ax_c = fig.add_subplot(grid[0, 10:12])
    c_stats = draw_contact_space_interictal(ax_c, analysis)

    d_stats = draw_cross_state_fields(fig, grid[1, 0:9], out, canonical_root)
    e_stats = draw_cross_state_statistics(fig, grid[1, 9:12], out, analysis)

    f_stats = draw_pathway_panel(fig, grid[2, 0:4], out)
    g_stats = draw_distal_contrasts(fig, grid[2, 4:8], out)
    h_stats = draw_attenuation(fig, grid[2, 8:12], out)

    for label, cell in zip("ABCDEFGH", (
        grid[0, 0:3], grid[0, 3:10], grid[0, 10:12], grid[1, 0:9],
        grid[1, 9:12], grid[2, 0:4], grid[2, 4:8], grid[2, 8:12],
    )):
        grid_letter(fig, cell, label)

    assert_no_label_overlap(fig)
    destination.mkdir(parents=True, exist_ok=True)
    stem = destination / "topic5_figure6_lbss_full_tissue_rnn"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "contract": "topic5_figure6_lbss_full_tissue_rnn_v0_3_candidate",
        "representative": SUBJECT,
        "panel_c": c_stats,
        "panel_d": d_stats,
        "panel_e": e_stats,
        "panel_f": f_stats,
        "panel_g": g_stats,
        "panel_h": h_stats,
        "geometry_status": "FULL_TISSUE_OFFSET_HULL_V0_3",
        "contact_role": "local H readout only",
        "target_role": "frozen external benchmark; not used for training or model selection",
    }
    (destination / "FIGURE6_METADATA.json").write_text(json.dumps(summary, indent=2) + "\n")
    assets = {}
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        assets[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (destination / "FIGURE6_COMPLETE.json").write_text(json.dumps({
        "status": "COMPLETE", "contract": summary["contract"], "assets_sha256": assets,
    }, indent=2) + "\n")
    (destination / "README.md").write_text(
        "### topic5_figure6_lbss_full_tissue_rnn.png / .pdf / .svg\n\n"
        "A 在 E1146 冻结组织平面内显示 full-tissue latent nodes、局部 recurrent backbone、task-selected nonlocal shortcuts 和只作为局部读出的真实 SEEG contacts；坐标方向与 Figure 2/3 的冻结 E1146 shared plane 一致。紫色输入 contact 及其周围带紫色外圈的 tissue nodes 显示 H^T 将第一 rank 输入分配到局部节点集合。为避免遮挡，仅显示权重最强的部分局部边与 3 条 nonlocal shortcuts，全部边仍参与统计和生成。"
        "B 比较 E1146 留出 TA/TB data 与只给第一 rank 后的自由生成。C 为 34 人 contact-space RNN 的患者内间期传播生成统计。"
        "D 为同一 E1146 full-tissue LBSS-RNN 生成的 TA/TB 场与 15 次 strict-broadband seizures 的 clinical onset 后 0–10 s、1–150 Hz early-ictal energy。"
        "E 并列 17 人 contact-space 跨状态结果与 12 人/141 seizures spatial exact-join 结果。"
        "F 只显示真实顺序与 order-shuffle 的粗 contact-space effective pathway，不把精确边当作真实 connectome。"
        "G 检验 selected nonlocal 相对 local、等容量 extra-local、固定 random nonlocal 与 shuffle 的总体/远端传播增量。"
        "H 检验 nonlocal attenuation 的 distal specificity 及其对 seed-removed early-ictal concordance 的剂量效应。\n\n"
        "**关注点**：只有 selected nonlocal 同时超过 matched controls，且 attenuation 选择性损害 distal propagation 或跨状态对应，才支持 selective-shortcut contribution；否则结论停在 full-tissue recurrent sufficiency。\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3")
    parser.add_argument("--contact-analysis", type=Path, default=ROOT / "results/topic5_rnn_full_cohort_field_transfer_v0_1")
    parser.add_argument("--canonical-root", type=Path, default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/paper-ready-figure/fig6_lbss_full_tissue_rnn/figures")
    parser.add_argument("--pretarget-preview", type=Path, default=None)
    args = parser.parse_args()
    if args.pretarget_preview is not None:
        path = plot_pretarget_preview(
            args.out_root.resolve(), args.contact_analysis.resolve(),
            args.canonical_root.resolve(), args.pretarget_preview.resolve(),
        )
        print(json.dumps({"pretarget_preview": str(path)}, indent=2))
        return
    summary = plot(
        args.out_root.resolve(), args.contact_analysis.resolve(),
        args.canonical_root.resolve(), args.out_dir.resolve(),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
