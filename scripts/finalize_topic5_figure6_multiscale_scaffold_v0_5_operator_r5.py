#!/usr/bin/env python3
"""Figure-6 R5 review candidate: sequence learning to shared functional response.

This candidate is intentionally visual.  It uses representative held-out TA/TB
events, an actual validation-loss curve, an orientation-free early-seizure
comparison, four explicit recurrent graph designs, their finite-time contact
response maps, a design-by-design similarity matrix, and a direct model-vs-null
scatter.  The accepted Figure 6 is never overwritten.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import finalize_topic5_figure6_multiscale_scaffold_v0_5_operator_r1 as r1  # noqa: E402
from build_topic5_multiscale_fields_v0_5 import empirical_candidates  # noqa: E402
from build_topic5_rnn_motif_fields_v0_4 import safe_corr  # noqa: E402
from analyze_topic5_patch_operator_v0_2 import (  # noqa: E402
    contact_space_operator,
    empirical_transition_operator,
)
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
    / "fig6_interictal_crossstate_response_r5_candidate/figures"
)
STEM = "topic5_figure6_interictal_crossstate_response_r5_candidate"

REAL_ARMS = ("L0", "L1", "L2m", "L3")
ALL_RESPONSE_ARMS = (*REAL_ARMS, "C-suffix")
LONG_ARM_NAMES = {
    "L0": base.L0,
    "L1": base.L1,
    "L2m": base.L2M,
    "L3": base.L3,
}
ARM_LABELS = {
    "L0": "Local only",
    "L1": "Local + nearby",
    "L2m": "Local + random distant",
    "L3": "Local + data-selected distant",
    "C-suffix": "Shuffled endings",
}
ARM_COLORS = {
    "L0": "#778188",
    "L1": "#6d97a7",
    "L2m": "#bd9045",
    "L3": base.RED,
    "C-suffix": "#9a9a9a",
}
RESPONSE_CMAP = "RdBu_r"
N_REPRESENTATIVE_EVENTS = 12
N_FIELD_NULL = 512


def paired_rank_key(name: str) -> tuple[str, int]:
    match = re.match(r"^(.*?)(\d+)$", str(name))
    return (match.group(1), int(match.group(2))) if match else (str(name), 0)


def stable_seed(*parts: object) -> int:
    token = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32 - 1)


def add_panel_letter(fig: plt.Figure, spec, label: str) -> None:
    box = spec.get_position(fig)
    fig.text(box.x0 - .020, box.y1 + .012, label, fontsize=15.5,
             fontweight="bold", ha="left", va="bottom")


def clean_network_panel(ax: plt.Axes, old: Path, canonical: Path) -> dict:
    """Observed rank sets -> full-tissue RNN -> next-contact SEEG readout."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    with np.load(old / "cache" / base.FIT_ID / "plane.npz", allow_pickle=False) as plane:
        nodes = np.asarray(plane["nodes_xy_mm"], float)
        contacts = np.asarray(plane["contacts_xy_mm"], float)
    provenance = json.loads((old / "cache" / base.FIT_ID / "provenance.json").read_text())
    names = [str(value) for value in provenance.get("contacts", provenance.get("joint_contacts"))]
    nodes, contacts = tissue_plot._align_tissue_plane_to_frozen_display(
        nodes, contacts, names, canonical,
    )
    with np.load(
        old / "per_fit" / base.FIT_ID / base.L3 / "seed0/graph.npz", allow_pickle=False,
    ) as graph:
        local = np.asarray(graph["local_mask"], bool)

    # Preserve the real SEEG/tissue geometry, but map it into one circular RNN
    # boundary.  No activity colouring and no long-range edges are shown here.
    all_xy = np.vstack([nodes, contacts])
    centre_data = all_xy.mean(axis=0)
    span = np.ptp(all_xy, axis=0)
    scale = max(float(span.max()), 1e-6)
    mapped_nodes = (nodes - centre_data) / scale * .58 + np.asarray([.535, .51])
    mapped_contacts = (contacts - centre_data) / scale * .58 + np.asarray([.535, .51])
    ax.add_patch(mpl.patches.Circle(
        (.535, .51), .345, facecolor="#f5f2f7", edgecolor="#846f89",
        lw=1.35, zorder=0,
    ))
    local_edges = unique_edges(local)
    if len(local_edges) > 135:
        local_edges = local_edges[np.linspace(0, len(local_edges) - 1, 135).astype(int)]
    for left, right in local_edges:
        ax.plot(mapped_nodes[[left, right], 0], mapped_nodes[[left, right], 1],
                color="#9ea5a8", lw=.40, alpha=.50, zorder=1)
    ax.scatter(mapped_nodes[:, 0], mapped_nodes[:, 1], s=8.5,
               facecolor="#d5dadd", edgecolor="#8b9498", lw=.24, zorder=2)
    shafts: dict[str, list[tuple[int, int]]] = {}
    for index, name in enumerate(names):
        shaft, number = tissue_plot._shaft_key(name)
        shafts.setdefault(shaft, []).append((number, index))
    for members in shafts.values():
        indices = [index for _, index in sorted(members)]
        ax.plot(mapped_contacts[indices, 0], mapped_contacts[indices, 1],
                color=base.DARK, lw=1.00, zorder=3)
    ax.scatter(mapped_contacts[:, 0], mapped_contacts[:, 1], s=21,
               facecolor="white", edgecolor=base.DARK, lw=.72, zorder=4)
    ax.text(.535, .82, "RNN", ha="center", va="center", fontsize=11.0,
            fontweight="bold", color="#5d4763")

    with np.load(old / "cache" / base.FIT_ID / "events.npz", allow_pickle=False) as source:
        events = {key: np.asarray(source[key]) for key in source.files}
    test = np.flatnonzero(events["split"] == 2)
    candidates = [int(index) for index in test
                  if int(np.max(events["ranks"][index])) >= 2]
    example = candidates[0] if candidates else int(test[0])
    observed = np.asarray(events["ranks"][example], int)
    physical_order = np.asarray(sorted(range(len(names)), key=lambda i: tissue_plot._shaft_key(names[i])))
    input_x = (.055, .100, .145)
    active_color = "#527c9a"
    for step, x in enumerate(input_x):
        for row, contact_index in enumerate(physical_order):
            y = .73 - row * (.44 / max(1, len(physical_order) - 1))
            active = observed[contact_index] == step
            ax.scatter(x, y, s=12.5, facecolor=active_color if active else "white",
                       edgecolor=base.DARK, lw=.48, zorder=5)
        ax.text(x, .245, str(step + 1), ha="center", va="top", fontsize=7.6,
                color=base.DARK)
    ax.text(.10, .80, "Observed ranks", ha="center", va="bottom", fontsize=8.7,
            color=base.DARK)
    ax.annotate("", xy=(.205, .51), xytext=(.165, .51),
                arrowprops={"arrowstyle": "-|>", "lw": 1.05, "color": base.DARK})

    with gzip.open(
        old / "per_fit" / base.FIT_ID / base.L3 / "seed0/heldout_rollouts.json.gz",
        "rt", encoding="utf-8",
    ) as stream:
        by_source = {int(row["event_source_index"]): row for row in json.load(stream)}
    sequence = by_source[int(events["event_source_index"][example])]["generated_rank_sets"]
    predicted = set(sequence[1]) if len(sequence) > 1 else set()
    for row, contact_index in enumerate(physical_order):
        y = .73 - row * (.44 / max(1, len(physical_order) - 1))
        ax.scatter(.925, y, s=14.5,
                   facecolor=base.RED if int(contact_index) in predicted else "white",
                   edgecolor=base.DARK, lw=.52, zorder=5)
    ax.annotate("", xy=(.885, .51), xytext=(.855, .51),
                arrowprops={"arrowstyle": "-|>", "lw": 1.05, "color": base.DARK})
    ax.text(.925, .80, "Next contact", ha="center", va="bottom", fontsize=8.7,
            color=base.DARK)
    ax.text(.925, .245, "or STOP", ha="center", va="top", fontsize=7.8,
            color=base.DARK)
    ax.legend(
        handles=[
            Line2D([], [], color="#9ea5a8", lw=1.1, label="Local recurrence"),
            Line2D([], [], marker="o", lw=0, markerfacecolor="white",
                   markeredgecolor=base.DARK, markersize=5.2, label="SEEG contact"),
        ],
        loc="upper right", bbox_to_anchor=(.86, .995), frameon=False,
        fontsize=7.4, handlelength=1.2, labelspacing=.22, borderpad=0,
    )
    return {
        "observed_steps": 3,
        "display": "real_SEEG_geometry_inside_circular_full_tissue_RNN",
        "long_links_drawn": 0,
        "TA_TB_supplied_to_model": False,
        "model_inputs": ["ordered rank-set contacts"],
        "model_outputs": ["next-contact logits", "STOP"],
    }


def generated_rank(sequence: list[list[int]], n_contacts: int) -> np.ndarray:
    result = np.full(n_contacts, -1, dtype=int)
    for step, contacts in enumerate(sequence):
        result[np.asarray(contacts, int)] = step
    return result


def normalized_event_matrix(rows: list[np.ndarray], n_contacts: int) -> np.ndarray:
    matrix = np.full((n_contacts, len(rows)), np.nan)
    for column, raw in enumerate(rows):
        rank = np.asarray(raw, int)
        finite = rank >= 0
        if finite.any():
            matrix[finite, column] = rank[finite] / max(1.0, float(rank[finite].max()))
    return matrix


def select_representative_events(
    out: Path, old: Path, canonical: Path,
) -> tuple[dict[str, dict[str, object]], list[str], np.ndarray]:
    with np.load(out / "cache" / base.FIT_ID / "events.npz", allow_pickle=False) as source:
        events = {key: np.asarray(source[key]) for key in source.files}
    provenance = json.loads((out / "cache" / base.FIT_ID / "provenance.json").read_text())
    contacts = [str(value) for value in provenance["joint_contacts"]]
    mapping = base.train_mode_to_ab(
        out / "cache" / base.FIT_ID, base.SUBJECT, np.asarray(contacts),
        canonical / "results/interictal_propagation_masked/template_gradient_fields/per_subject",
    )
    field = json.loads((
        canonical / "results/interictal_propagation_masked/template_gradient_fields"
        / "per_subject" / f"{base.SUBJECT}.json"
    ).read_text())["interictal_field"]
    order = [str(value) for value in field["contact_order"]]
    take = np.asarray([order.index(name) for name in contacts])
    templates = {
        "A": np.asarray(field["rank_a"], float)[take],
        "B": np.asarray(field["rank_b"], float)[take],
    }
    display_order = np.argsort(templates["A"], kind="stable")

    with gzip.open(
        old / "per_fit" / base.FIT_ID / base.L3 / "seed0/heldout_rollouts.json.gz",
        "rt", encoding="utf-8",
    ) as stream:
        rollouts = json.load(stream)
    by_source = {int(row["event_source_index"]): row for row in rollouts}

    selected: dict[str, dict[str, object]] = {}
    for template in ("A", "B"):
        candidates = []
        indices = [
            int(index) for index in np.flatnonzero(events["split"] == 2)
            if mapping[int(events["mode"][index])] == template
            and int(events["event_source_index"][index]) in by_source
        ]
        participation = np.asarray([
            np.sum(np.asarray(events["ranks"][index], int) >= 0) for index in indices
        ], float)
        typical = float(np.median(participation))
        for index, n_contact in zip(indices, participation):
            observed = np.asarray(events["ranks"][index], int)
            valid = (observed >= 0) & np.isfinite(templates[template])
            rho = float(spearmanr(observed[valid], templates[template][valid]).statistic) \
                if int(valid.sum()) >= 4 else -1.0
            if not np.isfinite(rho):
                rho = -1.0
            score = rho - .12 * abs(float(n_contact) - typical) / max(len(contacts), 1)
            candidates.append((score, rho, -abs(float(n_contact) - typical), index))
        candidates.sort(reverse=True)
        chosen = [item[-1] for item in candidates[:N_REPRESENTATIVE_EVENTS]]
        observed_rows = [np.asarray(events["ranks"][index], int) for index in chosen]
        generated_rows = [
            generated_rank(
                by_source[int(events["event_source_index"][index])]["generated_rank_sets"],
                len(contacts),
            ) for index in chosen
        ]
        selected[template] = {
            "indices": chosen,
            "observed": observed_rows,
            "generated": generated_rows,
            "median_template_rho": float(np.median([item[1] for item in candidates[:N_REPRESENTATIVE_EVENTS]])),
        }
    return selected, contacts, display_order


def draw_sequences(
    fig: plt.Figure, spec, out: Path, old: Path, canonical: Path,
) -> tuple[dict, mpl.image.AxesImage]:
    sub = spec.subgridspec(2, 2, hspace=.10, wspace=.075)
    axes = np.asarray([[fig.add_subplot(sub[row, col]) for col in range(2)] for row in range(2)])
    selected, contacts, display_order = select_representative_events(out, old, canonical)
    image = None
    cmap = mpl.colormaps[base.TIMING_CMAP].copy()
    cmap.set_bad("#e4e7e8")
    for row, (template, color) in enumerate((("A", base.RED), ("B", base.BLUE))):
        payload = selected[template]
        for col, key in enumerate(("observed", "generated")):
            matrix = normalized_event_matrix(payload[key], len(contacts))[display_order]
            image = axes[row, col].imshow(
                matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1,
            )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].margins(0)
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(
            f"T{template}", color=color, rotation=0, labelpad=15,
            fontsize=11.0, fontweight="bold", va="center",
        )
    axes[0, 0].text(.5, 1.06, "Recorded events", transform=axes[0, 0].transAxes,
                    ha="center", va="bottom", fontsize=10.0)
    axes[0, 1].text(.5, 1.06, "RNN rollouts", transform=axes[0, 1].transAxes,
                    ha="center", va="bottom", fontsize=10.0)
    return {
        "patient": base.SUBJECT,
        "events_per_pattern": N_REPRESENTATIVE_EVENTS,
        "selection": "highest held-out template concordance with typical participation",
        "TA_median_template_rho": selected["A"]["median_template_rho"],
        "TB_median_template_rho": selected["B"]["median_template_rho"],
        "TA_event_indices": selected["A"]["indices"],
        "TB_event_indices": selected["B"]["indices"],
        "TA_TB_used_as_model_input": False,
        "TA_TB_role": "train-only post-hoc grouping of held-out events and their rollouts",
    }, image


def draw_interictal_fields_and_shared_bar(
    fig: plt.Figure, spec, out: Path, canonical: Path, sequence_image,
) -> dict:
    sub = spec.subgridspec(2, 2, width_ratios=(1, .045), hspace=.10, wspace=.010)
    axes = [fig.add_subplot(sub[row, 0]) for row in range(2)]
    cax = fig.add_subplot(sub[:, 1])
    field = json.loads((
        canonical / "results/interictal_propagation_masked/template_gradient_fields"
        / "per_subject" / f"{base.SUBJECT}.json"
    ).read_text())["interictal_field"]
    order = [str(value) for value in field["contact_order"]]
    with np.load(
        out / "model_fields/intact/per_patient" / base.SUBJECT / f"{base.L3}.npz",
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
    for ax, value, support, label, color in zip(
        axes, values, supports, ("TA", "TB"), (base.RED, base.BLUE),
    ):
        base.draw_field(
            ax, points, value, support, xlim, ylim, cmap=base.TIMING_CMAP,
            vmin=0, vmax=1, title="", title_color=color, show_y=False,
        )
        ax.text(.04, .93, label, transform=ax.transAxes, color=color,
                fontsize=10.2, fontweight="bold", ha="left", va="top")
        ax.set_xlabel("")
        ax.set_ylabel("")
    bar = fig.colorbar(sequence_image, cax=cax, orientation="vertical")
    bar.set_ticks([0, 1], labels=["Early", "Late"])
    bar.ax.tick_params(labelsize=7.6, pad=1)
    bar.ax.set_ylabel("Event order", rotation=90, labelpad=4, fontsize=8.2)
    return {"patient": base.SUBJECT, "fields": ["TA", "TB"], "shared_colorbar_with_B": True}


def history_root(out: Path, old: Path, arm: str) -> Path:
    if arm == base.L2M:
        return out / "formal_units" / base.FIT_ID / arm
    return old / "per_fit" / base.FIT_ID / arm


def load_learning_history(
    out: Path, old: Path, arm: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    histories = []
    for seed in range(3):
        rows = json.loads((
            history_root(out, old, arm) / f"seed{seed}/history.json"
        ).read_text())
        series = pd.Series(
            [float(row["validation_contact_nll"]) for row in rows],
            index=[int(row["epoch"]) for row in rows], dtype=float,
        ).rolling(9, center=True, min_periods=1).mean()
        histories.append(series)
    epochs = np.arange(max(int(series.index.max()) for series in histories) + 1)
    matrix = np.full((len(histories), len(epochs)), np.nan)
    for row, series in enumerate(histories):
        matrix[row, series.index.to_numpy(int)] = series.to_numpy(float)
    return epochs, np.nanmedian(matrix, axis=0), np.nanpercentile(matrix, 25, axis=0), np.nanpercentile(matrix, 75, axis=0)


def draw_learning_curve(ax: plt.Axes, out: Path, old: Path) -> dict:
    curves = []
    for arm, public_arm, label, color, linestyle in (
        (base.L0, "L0", "Local only", ARM_COLORS["L0"], "-"),
        (base.L1, "L1", "Local + nearby", ARM_COLORS["L1"], "-"),
        (base.L2M, "L2m", "Local + random distant", ARM_COLORS["L2m"], "-"),
        (base.L3, "L3", "Local + data-selected distant", ARM_COLORS["L3"], "-"),
        (base.SUFFIX, "C-suffix", "Shuffled endings", ARM_COLORS["C-suffix"], "--"),
    ):
        epoch, median, low, high = load_learning_history(out, old, arm)
        ax.fill_between(epoch, low, high, color=color, alpha=.16, linewidth=0)
        ax.plot(epoch, median, color=color, lw=1.65, ls=linestyle, label=label)
        curves.append((public_arm, label, float(median[0]), float(np.nanmin(median)), int(epoch[np.nanargmin(median)])))
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Held-out next-contact NLL")
    ax.set_xlim(0, max(
        len(load_learning_history(out, old, arm)[0])
        for arm in (base.L0, base.L1, base.L2M, base.L3, base.SUFFIX)
    ) - 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.1, loc="upper right", handlelength=1.45,
              labelspacing=.25, borderpad=0)
    return {"patient": base.SUBJECT, "curves": curves, "seeds_per_curve": 3}


def build_interictal_field_null(out: Path, canonical: Path) -> pd.DataFrame:
    metrics = pd.read_csv(out / "MODEL_FIELD_PATIENT_METRICS.csv")
    subjects = metrics.loc[metrics.arm.eq(base.L3), "subject"].astype(str).sort_values().unique()
    field_root = canonical / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
    rows = []
    for subject in subjects:
        with np.load(
            out / "model_fields/intact/per_patient" / subject / f"{base.L3}.npz",
            allow_pickle=False,
        ) as model:
            contacts = np.asarray(model["contacts"]).astype(str)
            generated = {
                "full_A": np.asarray(model["A_canonical_full"], float),
                "full_B": np.asarray(model["B_canonical_full"], float),
                "later_A": np.asarray(model["A_seed_removed"], float),
                "later_B": np.asarray(model["B_seed_removed"], float),
            }
        empirical = empirical_candidates(field_root, subject, contacts)
        observed_full = float(np.nanmean([
            safe_corr(generated["full_A"], empirical["A"]),
            safe_corr(generated["full_B"], empirical["B"]),
        ]))
        observed_later = float(np.nanmean([
            safe_corr(generated["later_A"], empirical["A"]),
            safe_corr(generated["later_B"], empirical["B"]),
        ]))
        rng = np.random.default_rng(stable_seed("figure6_r4_field_null", subject))
        null_full, null_later = [], []
        for _ in range(N_FIELD_NULL):
            permutation = rng.permutation(len(contacts))
            null_full.append(np.nanmean([
                safe_corr(generated["full_A"], empirical["A"][permutation]),
                safe_corr(generated["full_B"], empirical["B"][permutation]),
            ]))
            null_later.append(np.nanmean([
                safe_corr(generated["later_A"], empirical["A"][permutation]),
                safe_corr(generated["later_B"], empirical["B"][permutation]),
            ]))
        rows.append({
            "patient": subject,
            "complete_event": observed_full,
            "complete_event_shuffled": float(np.nanmedian(null_full)),
            "later_contacts": observed_later,
            "later_contacts_shuffled": float(np.nanmedian(null_later)),
        })
    return pd.DataFrame(rows)


def add_bracket(ax: plt.Axes, left: float, right: float, y: float, text: str) -> None:
    span = np.diff(ax.get_ylim())[0]
    ax.plot([left, left, right, right], [y - .015 * span, y, y, y - .015 * span],
            color=base.DARK, lw=.85, clip_on=False)
    ax.text((left + right) / 2, y + .006 * span, text, ha="center", va="bottom",
            fontsize=10.0, fontweight="bold")


def violin_pair(ax: plt.Axes, values: list[np.ndarray], positions: list[float], colors: list[str]) -> None:
    parts = ax.violinplot(values, positions=positions, widths=.56,
                          showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(.24)
    rng = np.random.default_rng(6417)
    for values_i, pos, color in zip(values, positions, colors):
        values_i = np.asarray(values_i, float)
        jitter = rng.uniform(-.12, .12, len(values_i))
        ax.scatter(pos + jitter, values_i, s=15, color=color, alpha=.55,
                   edgecolor="white", lw=.25, zorder=3)
        q1, med, q3 = np.nanpercentile(values_i, [25, 50, 75])
        ax.plot([pos, pos], [q1, q3], color=color, lw=2.6, solid_capstyle="round", zorder=4)
        ax.plot([pos - .12, pos + .12], [med, med], color=base.DARK, lw=1.8, zorder=5)


def draw_interictal_field_recovery(ax: plt.Axes, out: Path, canonical: Path) -> tuple[dict, pd.DataFrame]:
    source = build_interictal_field_null(out, canonical)
    values = [
        source["later_contacts"].to_numpy(float),
        source["later_contacts_shuffled"].to_numpy(float),
    ]
    positions = [0, 1]
    colors = [base.BLUE, "#a7adaf"]
    violin_pair(ax, values, positions, colors)
    ax.axhline(0, color="#8a9093", lw=.75, ls="--", zorder=0)
    ax.set_xlim(-.48, 1.48)
    ax.set_xticks(positions, ["Generated\ncontinuation", "Contacts\nshuffled"])
    ax.set_ylabel("Mean TA/TB match to\nrecorded interictal fields")
    ax.spines[["top", "right"]].set_visible(False)
    p_later = float(wilcoxon(values[0] - values[1], alternative="greater").pvalue)
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + .14 * (high - low))
    y = high + .025 * (high - low)
    add_bracket(ax, positions[0], positions[1], y, base.stars(p_later))
    return {
        "n": int(len(source)), "p_generated_continuation": p_later,
        "median_generated_continuation": float(np.nanmedian(values[0])),
        "median_contact_shuffle_null": float(np.nanmedian(values[1])),
        "main_figure_endpoint": "first observed contact removed; freely generated continuation only",
    }, source


def early_profile(out: Path, canonical: Path) -> dict[str, object]:
    field = json.loads((
        canonical / "results/interictal_propagation_masked/template_gradient_fields"
        / "per_subject" / f"{base.SUBJECT}.json"
    ).read_text())["interictal_field"]
    order = [str(value) for value in field["contact_order"]]
    arm_patterns: dict[str, dict[str, np.ndarray]] = {}
    for public_arm in REAL_ARMS:
        arm_name = LONG_ARM_NAMES[public_arm]
        with np.load(
            out / "model_fields/intact/per_patient" / base.SUBJECT / f"{arm_name}.npz",
            allow_pickle=False,
        ) as model:
            names = model["contacts"].astype(str).tolist()
            take = np.asarray([names.index(name) for name in order])
            arm_patterns[public_arm] = {
                "TA": 1.0 - np.asarray(model["A_canonical_full"], float)[take],
                "TB": 1.0 - np.asarray(model["B_canonical_full"], float)[take],
            }
    with np.load(
        out / "early_ictal/per_patient_targets" / f"{base.SUBJECT}.npz",
        allow_pickle=False,
    ) as target:
        target_names = target["contacts"].astype(str).tolist()
        lookup = dict(zip(target_names, np.asarray(target["median_broadband_energy"], float)))
        energy = np.asarray([lookup[name] for name in order], float)
    correlations = {
        label: safe_corr(pattern, energy)
        for label, pattern in arm_patterns["L3"].items()
    }
    label = max(correlations, key=lambda key: abs(correlations[key]))
    model_profiles = {arm: patterns[label] for arm, patterns in arm_patterns.items()}
    energy_profile = rankdata(energy, method="average")
    energy_profile = (energy_profile - 1) / max(len(energy_profile) - 1, 1)
    if correlations[label] < 0:
        energy_profile = 1.0 - energy_profile
    points, _xlim, _ylim = base.field_geometry(field)
    return {
        "x": np.asarray(points[:, 0], float),
        "models": model_profiles,
        "seizure": energy_profile,
        "pattern": label,
        "absolute_r": abs(float(correlations[label])),
    }


def draw_early_seizure_case(
    ax_profile: plt.Axes, out: Path, canonical: Path,
) -> tuple[dict, pd.DataFrame]:
    profile = early_profile(out, canonical)
    order = np.argsort(profile["x"], kind="stable")
    for arm in REAL_ARMS:
        ax_profile.plot(
            np.asarray(profile["x"])[order], np.asarray(profile["models"][arm])[order],
            color=ARM_COLORS[arm], lw=1.35, marker="o", ms=2.3,
            label=ARM_LABELS[arm], alpha=.90,
        )
    ax_profile.plot(np.asarray(profile["x"])[order], np.asarray(profile["seizure"])[order],
                    color=base.DARK, lw=2.15, marker="o", ms=3.1, label="Early seizure")
    ax_profile.set_xlabel("Contact position (mm)")
    ax_profile.set_ylabel("Normalized field")
    ax_profile.set_ylim(-.05, 1.05)
    ax_profile.spines[["top", "right"]].set_visible(False)
    ax_profile.legend(
        handles=[Line2D([], [], color=base.DARK, marker="o", markersize=3.2,
                        lw=2.0, label="Early seizure")],
        frameon=False, fontsize=7.4, loc="upper right", bbox_to_anchor=(1.0, 1.015),
        handlelength=1.25, borderpad=0,
    )
    ax_profile.text(.03, .96, f"E1146 · {profile['pattern']}", transform=ax_profile.transAxes,
                    fontsize=7.5, ha="left", va="top", color="#4f5558")
    source = pd.DataFrame({
        "contact_position_mm": np.asarray(profile["x"])[order],
        **{
            f"{arm}_interictal_field": np.asarray(profile["models"][arm])[order]
            for arm in REAL_ARMS
        },
        "early_seizure_field_orientation_aligned": np.asarray(profile["seizure"])[order],
    })
    return {
        "patient": base.SUBJECT,
        "representative_pattern": profile["pattern"],
        "representative_absolute_r": profile["absolute_r"],
        "mode_fixed_from": "L3 max-absolute TA/TB; the same mode is shown for all four networks",
        "direction_used": False,
    }, source


def draw_early_seizure_cohort(
    ax: plt.Axes, out: Path,
) -> tuple[dict, pd.DataFrame]:
    source = pd.read_csv(out / "early_ictal/POSTHOC_SIGN_SENSITIVITY_PER_PATIENT.csv")
    source = source[(source.control == "all_contacts") & (source.orientation == "sign_free")].copy()
    values = [source["observed"].to_numpy(float), source["null_median"].to_numpy(float)]
    positions = [0, 1]
    violin_pair(ax, values, positions, [base.BLUE, "#a7adaf"])
    ax.set_xlim(-.48, 1.48)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(positions, ["Interictal\nfield", "Contacts\nshuffled"])
    ax.set_ylabel("Best absolute TA/TB match\nto early seizure field")
    ax.spines[["top", "right"]].set_visible(False)
    p_value = float(wilcoxon(source["margin"], alternative="greater").pvalue)
    add_bracket(ax, positions[0], positions[1], .965, base.stars(p_value))
    return {
        "n": int(len(source)), "p_vs_shuffled_contacts": p_value,
        "median_margin": float(np.median(source["margin"])),
        "direction_used": False,
    }, source


def load_response_maps(response_root: Path, out: Path, canonical: Path):
    root = response_root / "per_cell" / base.FIT_ID
    by_arm_phase = []
    counts_by_arm_phase = []
    source_nodes = None
    for arm in REAL_ARMS:
        seed_phase = []
        seed_counts = []
        for seed in range(3):
            with np.load(root / arm / f"seed{seed}/patch_operator.npz", allow_pickle=False) as handle:
                tensor = np.asarray(handle["mean_contact_operator"], float)
                counts = np.asarray(handle["valid_counts"], int)
                source_nodes = np.asarray(handle["node_xy_mm"], float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                seed_phase.append(np.nanmean(tensor[:, :, 1, 1:4, :], axis=2))
            seed_counts.append(counts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            by_arm_phase.append(np.nanmedian(np.stack(seed_phase), axis=0))
        counts_by_arm_phase.append(np.stack(seed_counts))
    response_by_phase = np.stack(by_arm_phase)

    provenance = json.loads((out / "cache" / base.FIT_ID / "provenance.json").read_text())
    names = [str(value) for value in provenance["joint_contacts"]]
    with np.load(out / "cache" / base.FIT_ID / "plane.npz", allow_pickle=False) as plane:
        contacts_xy = np.asarray(plane["contacts_xy_mm"], float)
        nodes_xy = np.asarray(plane["nodes_xy_mm"], float)
    centre = contacts_xy.mean(axis=0)
    location = int(np.argmin(np.linalg.norm(nodes_xy - centre, axis=1)))
    record = json.loads((
        canonical / "results/interictal_propagation_masked/template_gradient_fields"
        / "per_subject" / f"{base.SUBJECT}.json"
    ).read_text())["interictal_field"]
    order = [str(value) for value in record["contact_order"]]
    take = np.asarray([names.index(name) for name in order])
    values_by_phase = response_by_phase[:, :, location, :][:, :, take]
    counts_by_phase = np.asarray([
        counts[:, :, location, 1, 0].sum(axis=0)
        for counts in counts_by_arm_phase
    ], int)
    aligned_nodes, aligned_contacts = tissue_plot._align_tissue_plane_to_frozen_display(
        nodes_xy, contacts_xy, names, canonical,
    )
    points, xlim, ylim = base.field_geometry(record)
    return (
        values_by_phase, counts_by_phase, aligned_nodes, aligned_contacts,
        points, xlim, ylim, location,
    )


def graph_path(out: Path, old: Path, arm: str) -> Path:
    if arm == "L2m":
        return out / "formal_units" / base.FIT_ID / base.L2M / "seed0/graph.npz"
    return old / "per_fit" / base.FIT_ID / LONG_ARM_NAMES[arm] / "seed0/graph.npz"


def unique_edges(mask: np.ndarray) -> np.ndarray:
    undirected = np.asarray(mask, bool) | np.asarray(mask, bool).T
    return np.argwhere(np.triu(undirected, 1))


def draw_graph_inset(
    ax: plt.Axes, nodes: np.ndarray, contacts: np.ndarray, path: Path,
    arm: str, location: int,
) -> dict:
    with np.load(path, allow_pickle=False) as graph:
        local = np.asarray(graph["local_mask"], bool)
        added = np.asarray(graph["added_mask"], bool)
        strength = np.asarray(graph["strength"], float)
        distance = np.asarray(graph["D_mm"], float)
    local_edges = unique_edges(local)
    if len(local_edges) > 180:
        local_edges = local_edges[np.linspace(0, len(local_edges) - 1, 180).astype(int)]
    for left, right in local_edges:
        ax.plot(nodes[[left, right], 0], nodes[[left, right], 1],
                color="#8f999e", lw=.32, alpha=.30, zorder=1)
    added_edges = unique_edges(added)
    if len(added_edges):
        scores = np.asarray([
            max(abs(strength[left, right]), abs(strength[right, left]))
            for left, right in added_edges
        ])
        added_edges = added_edges[np.argsort(scores)[::-1][:18]]
        for left, right in added_edges:
            ax.plot(nodes[[left, right], 0], nodes[[left, right], 1],
                    color=ARM_COLORS[arm], lw=.72, alpha=.76, zorder=2)
    ax.scatter(nodes[:, 0], nodes[:, 1], s=5, color="#879196", alpha=.78, zorder=3)
    ax.scatter(contacts[:, 0], contacts[:, 1], s=15, facecolor="white",
               edgecolor=base.DARK, lw=.55, zorder=4)
    ax.scatter(nodes[location, 0], nodes[location, 1], marker="*", s=45,
               color=base.RED, edgecolor="white", lw=.5, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(nodes[:, 0].min() - 2, nodes[:, 0].max() + 2)
    ax.set_ylim(nodes[:, 1].min() - 2, nodes[:, 1].max() + 2)
    ax.axis("off")
    ax.text(.02, .98, ARM_LABELS[arm], transform=ax.transAxes, ha="left", va="top",
            fontsize=7.3, fontweight="bold", color=ARM_COLORS[arm])
    ax.annotate("", xy=(1.09, .50), xytext=(.94, .50), xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "lw": .9, "color": base.DARK},
                annotation_clip=False)
    added_distance = float(np.median(distance[added])) if added.any() else float("nan")
    return {"added_edges": int(added_edges.shape[0]), "median_added_distance_mm": added_distance}


def draw_network_designs(fig: plt.Figure, spec) -> dict:
    cartoons = spec.subgridspec(2, 2, hspace=.24, wspace=.18)
    x, y = [], []
    for row in range(4):
        for col in range(5):
            x.append(col + .42 * (row % 2))
            y.append(2.7 - .90 * row)
    cartoon_xy = np.column_stack([x, y])
    distance = np.linalg.norm(cartoon_xy[:, None, :] - cartoon_xy[None, :, :], axis=2)
    input_node = 7
    nearest = np.argsort(distance[input_node])[1:7]
    nearby = [0, 4, 15, 19]
    random_distant = [3, 14, 16]
    selected_distant = [0, 15, 19]
    added_edges = {
        "L0": [],
        "L1": [(input_node, node) for node in nearby],
        "L2m": [(input_node, node) for node in random_distant],
        "L3": [(input_node, node) for node in selected_distant],
    }
    for index, arm in enumerate(REAL_ARMS):
        ax = fig.add_subplot(cartoons[index // 2, index % 2])
        for right in nearest:
            left = input_node
            ax.plot(cartoon_xy[[left, right], 0], cartoon_xy[[left, right], 1],
                    color="#7e888d", lw=1.05, alpha=.82, zorder=1)
        for edge_index, (left, right) in enumerate(added_edges[arm]):
            bend = .06 * (-1 if edge_index % 2 else 1)
            patch = mpl.patches.FancyArrowPatch(
                cartoon_xy[left], cartoon_xy[right], arrowstyle="-",
                connectionstyle=f"arc3,rad={bend}", color=ARM_COLORS[arm],
                lw=1.45, alpha=.95, zorder=2,
            )
            ax.add_patch(patch)
        ax.scatter(cartoon_xy[:, 0], cartoon_xy[:, 1], s=11,
                   facecolor="white", edgecolor="#5f686c", lw=.55, zorder=3)
        ax.scatter(cartoon_xy[input_node, 0], cartoon_xy[input_node, 1],
                   marker="o", s=42, color=base.RED, edgecolor="white", lw=.65, zorder=4)
        display_label = {
            "L0": "Local only",
            "L1": "Local + nearby",
            "L2m": "Local +\nrandom distant",
            "L3": "Local +\ndata-selected distant",
        }[arm]
        ax.text(.5, 1.015, display_label, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.8, fontweight="bold",
                color=ARM_COLORS[arm])
        ax.set_xlim(-.25, 4.75)
        ax.set_ylim(-.25, 3.05)
        ax.set_aspect("equal")
        ax.axis("off")
    return {
        "same_red_input_node": True,
        "display": "schematic_spokes_from_one_common_input_node",
        "designs": [ARM_LABELS[arm] for arm in REAL_ARMS],
    }


def draw_combined_model_panel(
    fig: plt.Figure, spec, old: Path, canonical: Path,
) -> dict:
    """One panel: observable input/output computation plus four graph constraints."""
    sub = spec.subgridspec(1, 2, width_ratios=(1.22, 1.0), wspace=.045)
    computation = clean_network_panel(fig.add_subplot(sub[0, 0]), old, canonical)
    designs = draw_network_designs(fig, sub[0, 1])
    return {"computation": computation, "connection_designs": designs}


def draw_repeated_response_matrix(
    fig: plt.Figure, spec, response_root: Path, out: Path, canonical: Path,
) -> tuple[dict, pd.DataFrame]:
    outer = spec.subgridspec(1, 2, width_ratios=(1.0, .05), wspace=.09)
    (
        values_by_phase, counts_by_phase, _nodes, _contacts,
        _points, _xlim, _ylim, location,
    ) = load_response_maps(response_root, out, canonical)
    field = json.loads((
        canonical / "results/interictal_propagation_masked/template_gradient_fields"
        / "per_subject" / f"{base.SUBJECT}.json"
    ).read_text())["interictal_field"]
    contact_names = [str(value) for value in field["contact_order"]]
    values = np.concatenate([
        np.vstack([values_by_phase[arm_index], np.nanmean(values_by_phase[arm_index], axis=0)])
        for arm_index in range(len(REAL_ARMS))
    ], axis=0)
    response_ax = fig.add_subplot(outer[0, 0])
    vmin = min(float(np.nanpercentile(values, 2)), -.10)
    vmax = max(float(np.nanpercentile(values, 98)), .10)
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    image = response_ax.imshow(
        values, aspect="auto", interpolation="nearest", cmap=RESPONSE_CMAP, norm=norm,
    )
    response_ax.axvline(3.5, color="white", lw=1.8)
    for boundary in (3.5, 7.5, 11.5):
        response_ax.axhline(boundary, color="white", lw=2.0)
    response_ax.set_xticks(range(len(contact_names)), contact_names,
                           rotation=90, ha="center", fontsize=7.2)
    phase_names = ("early", "middle", "late")
    row_names = {
        "L0": "local",
        "L1": "+ nearby",
        "L2m": "+ random",
        "L3": "+ data-selected",
    }
    ylabels = []
    for arm in REAL_ARMS:
        ylabels.extend([f"{row_names[arm]} · early", "middle", "late", "all phases"])
    response_ax.set_yticks(range(16), ylabels, fontsize=7.0)
    for label, arm in zip(response_ax.get_yticklabels(), np.repeat(REAL_ARMS, 4)):
        label.set_color(ARM_COLORS[str(arm)])
    response_ax.tick_params(length=0, pad=2.5, colors=base.DARK)
    response_ax.text(1.5, 1.035, "SCL", transform=response_ax.get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    response_ax.text(9.0, 1.035, "ICL", transform=response_ax.get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    response_ax.set_xlabel("Future SEEG contact", labelpad=3, color=base.DARK)
    for spine in response_ax.spines.values():
        spine.set_linewidth(.65)

    cax = fig.add_subplot(outer[0, 1])
    bar = fig.colorbar(image, cax=cax, orientation="vertical")
    bar.set_ticks([vmin, 0, vmax], labels=["Less", "0", "More"])
    bar.ax.tick_params(labelsize=7.4, pad=1, colors=base.DARK)
    bar.ax.set_title("Mean output\nchange", fontsize=7.6, pad=3, color=base.DARK)

    source_rows = []
    for arm_index, arm in enumerate(REAL_ARMS):
        for phase_index, phase in enumerate(phase_names):
            for contact_index, value in enumerate(values_by_phase[arm_index, phase_index]):
                source_rows.append({
                    "network_design": ARM_LABELS[arm], "event_phase": phase,
                    "contact_index": contact_index, "contact": contact_names[contact_index],
                    "later_contact_change": float(value), "tissue_location_index": location,
                    "eligible_reference_states_across_3_seeds": int(counts_by_phase[arm_index, phase_index]),
                })
        pooled = np.nanmean(values_by_phase[arm_index], axis=0)
        for contact_index, value in enumerate(pooled):
            source_rows.append({
                "network_design": ARM_LABELS[arm], "event_phase": "3-phase mean",
                "contact_index": contact_index, "contact": contact_names[contact_index],
                "later_contact_change": float(value), "tissue_location_index": location,
                "eligible_reference_states_across_3_seeds": int(counts_by_phase[arm_index].sum()),
            })
    return {
        "patient": base.SUBJECT, "tissue_location_index": location,
        "same_gaussian_tissue_patch_for_all_designs": True,
        "dose": "0.5 local hidden-state SD",
        "future_steps_averaged": [1, 2, 3],
        "event_phases": [0.25, 0.5, 0.75],
        "seeds": 3,
        "eligible_reference_states_by_design_and_phase": counts_by_phase.tolist(),
        "response_display": "actual_uninterpolated_contact_output_matrix_by_event_phase",
    }, pd.DataFrame(source_rows)


def load_fit_operator(response_root: Path, fit_id: str, arm: str) -> np.ndarray:
    seeds = []
    for seed in range(3):
        with np.load(
            response_root / "per_cell" / fit_id / arm / f"seed{seed}/patch_operator.npz",
            allow_pickle=False,
        ) as source:
            mean = np.asarray(source["mean_contact_operator"], float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pooled = np.nanmean(np.nanmean(mean[:, :, 1][:, :, 1:4, :], axis=2), axis=0)
        seeds.append(pooled.T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(np.stack(seeds), axis=0)


def pattern_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, float).ravel(), np.asarray(right, float).ravel()
    use = np.isfinite(a) & np.isfinite(b)
    if int(use.sum()) < 8:
        return float("nan")
    a, b = a[use] - a[use].mean(), b[use] - b[use].mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else float("nan")


def build_design_matrix(response_root: Path) -> tuple[np.ndarray, pd.DataFrame]:
    by_patient: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    fit_dirs = sorted(path for path in (response_root / "per_cell").iterdir() if path.is_dir())
    for fit_dir in fit_dirs:
        fit_id = fit_dir.name
        patient = fit_id.rsplit("__", 1)[0]
        operators = {arm: load_fit_operator(response_root, fit_id, arm) for arm in ALL_RESPONSE_ARMS}
        for left_index, left in enumerate(ALL_RESPONSE_ARMS):
            for right_index in range(left_index, len(ALL_RESPONSE_ARMS)):
                right = ALL_RESPONSE_ARMS[right_index]
                value = 1.0 if left == right else pattern_similarity(operators[left], operators[right])
                by_patient[patient][(left, right)].append(value)
    matrix = np.full((len(ALL_RESPONSE_ARMS), len(ALL_RESPONSE_ARMS)), np.nan)
    rows = []
    for left_index, left in enumerate(ALL_RESPONSE_ARMS):
        for right_index, right in enumerate(ALL_RESPONSE_ARMS):
            key = (left, right) if left_index <= right_index else (right, left)
            patient_values = [float(np.nanmedian(values[key])) for values in by_patient.values()]
            value = float(np.nanmedian(patient_values))
            matrix[left_index, right_index] = value
            rows.append({
                "design_1": ARM_LABELS[left], "design_2": ARM_LABELS[right],
                "cohort_median_similarity": value, "n_patients": len(patient_values),
            })
    return matrix, pd.DataFrame(rows)


def draw_similarity_matrix(ax: plt.Axes, response_root: Path) -> tuple[dict, pd.DataFrame]:
    matrix, source = build_design_matrix(response_root)
    image = ax.imshow(matrix, cmap="Blues", vmin=.48, vmax=1.0, interpolation="nearest")
    labels = ["Local", "+ nearby", "+ random\ndistant", "+ learned\ndistant", "Shuffled\nendings"]
    ax.set_xticks(range(5), labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(5), labels)
    ax.tick_params(length=0, labelsize=7.8, pad=2)
    tick_colors = [ARM_COLORS[arm] for arm in ALL_RESPONSE_ARMS]
    for label, color in zip(ax.get_xticklabels(), tick_colors):
        label.set_color(color)
    for label, color in zip(ax.get_yticklabels(), tick_colors):
        label.set_color(color)
    for row in range(5):
        for col in range(5):
            color = "white" if matrix[row, col] > .73 else base.DARK
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center",
                    fontsize=7.3, color=color)
    ax.add_patch(mpl.patches.Rectangle((3.5, -.5), 1, 5, fill=False,
                                       edgecolor="#72787b", lw=1.2))
    ax.add_patch(mpl.patches.Rectangle((-.5, 3.5), 5, 1, fill=False,
                                       edgecolor="#72787b", lw=1.2))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Network 2", color=base.DARK, labelpad=4)
    ax.set_ylabel("Network 1", color=base.DARK, labelpad=4)
    real = matrix[:4, :4][np.triu_indices(4, 1)]
    control = matrix[:4, 4]
    return {
        "median_real_design_similarity": float(np.median(real)),
        "median_similarity_to_reassigned_endings": float(np.median(control)),
        "random_distant_arm": "L2M_MACRO_MATCHED_RANDOM_LR",
        "same_random_distant_arm_as_training_curve": True,
    }, source


def draw_model_data_link(
    fig: plt.Figure, spec, response_root: Path,
) -> tuple[dict, pd.DataFrame]:
    outer = spec.subgridspec(2, 1, height_ratios=(.38, .62), hspace=.46)
    diagram = fig.add_subplot(outer[0, 0])
    diagram.axis("off")
    consensus = np.nanmedian(np.stack([
        load_fit_operator(response_root, base.FIT_ID, arm) for arm in REAL_ARMS
    ]), axis=0)
    response_contact = contact_space_operator(consensus, base.FIT_ID, "L3", 0)
    transition, _n_events = empirical_transition_operator(base.FIT_ID)
    left = diagram.inset_axes([.03, .08, .36, .84])
    right = diagram.inset_axes([.61, .08, .36, .84])
    for matrix_ax, matrix in ((left, response_contact), (right, transition)):
        finite = matrix[np.isfinite(matrix)]
        bound = max(float(np.nanpercentile(np.abs(finite), 95)), 1e-6)
        matrix_ax.imshow(matrix / bound, cmap=RESPONSE_CMAP,
                         norm=mpl.colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
                         interpolation="nearest")
        matrix_ax.set_xticks([])
        matrix_ax.set_yticks([])
        for spine in matrix_ax.spines.values():
            spine.set_linewidth(.6)
            spine.set_color("#83898c")
    diagram.annotate("", xy=(.58, .50), xytext=(.42, .50), xycoords="axes fraction",
                     arrowprops={"arrowstyle": "<->", "lw": 1.1, "color": base.DARK})
    diagram.text(.21, .99, "Perturbation response",
                 transform=diagram.transAxes, ha="center", va="bottom",
                 fontsize=8.7, color=base.DARK)
    diagram.text(.79, .99, "Held-out contact transitions",
                 transform=diagram.transAxes, ha="center", va="bottom",
                 fontsize=8.7, color=base.DARK)

    ax = fig.add_subplot(outer[1, 0])
    source = pd.read_csv(response_root / "OPERATOR_DATA_ALIGNMENT.csv")
    patient = source.groupby("patient", as_index=False).median(numeric_only=True)
    null = patient["within_shaft_null_median"].to_numpy(float)
    observed = patient["consensus_alignment"].to_numpy(float)
    finite = np.isfinite(null) & np.isfinite(observed)
    null, observed = null[finite], observed[finite]
    violin_pair(ax, [observed, null], [0, 1], [base.BLUE, "#a7adaf"])
    ax.axhline(0, color="#8a9093", lw=.75, ls="--", zorder=0)
    ax.set_xlim(-.48, 1.48)
    ax.set_xticks([0, 1], ["Model response", "Contacts shuffled\nwithin electrode"])
    ax.set_ylabel("Spearman similarity to\nheld-out contact transitions", color=base.DARK)
    ax.tick_params(colors=base.DARK)
    ax.spines[["top", "right"]].set_visible(False)
    above = int(np.sum(observed > null))
    p_value = float(wilcoxon(observed - null, alternative="greater").pvalue)
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + .15 * (high - low))
    add_bracket(ax, 0, 1, high + .025 * (high - low), base.stars(p_value))
    patient = patient.loc[finite].copy()
    return {
        "n": int(len(observed)), "patients_above_spatial_null": above,
        "median_margin": float(np.median(observed - null)),
        "p_one_sided": p_value,
        "comparison": "finite-time perturbation response versus held-out empirical contact-following matrix",
        "not_a_prediction_accuracy_test": True,
        "matrix_icons_independently_normalized_for_shape_display": True,
    }, patient


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
        old / "per_fit" / base.FIT_ID / base.L3 / "seed0/history.json",
        out / "formal_units" / base.FIT_ID / base.L2M / "seed0/history.json",
        response_root / "PATCH_OPERATOR_SUMMARY.json",
        response_root / "per_cell" / base.FIT_ID / "L0/seed0/patch_operator.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Figure-6 R4 inputs missing: {missing}")

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.0,
        "axes.labelsize": 10.0, "axes.titlesize": 10.0,
        "xtick.labelsize": 8.7, "ytick.labelsize": 8.7,
        "text.color": base.DARK, "axes.labelcolor": base.DARK,
        "xtick.color": base.DARK, "ytick.color": base.DARK,
        "axes.linewidth": .75, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(15.4, 10.0), facecolor="white")
    grid = fig.add_gridspec(
        3, 1, height_ratios=(.96, .70, .94), left=.050, right=.975,
        bottom=.067, top=.970, hspace=.42,
    )

    top = grid[0, 0].subgridspec(1, 2, width_ratios=(1.58, 2.42), wspace=.075)
    a_spec, b_parent = top[0, 0], top[0, 1]
    b_group = b_parent.subgridspec(1, 2, width_ratios=(3.25, .75), wspace=.005)
    b_spec, b_field_spec = b_group[0, 0], b_group[0, 1]

    middle = grid[1, 0].subgridspec(
        1, 4, width_ratios=(1.18, 1.0, 1.0, 1.0), wspace=.46,
    )
    c_spec, d_spec, e_spec, f_spec = [middle[0, index] for index in range(4)]

    bottom = grid[2, 0].subgridspec(
        1, 3, width_ratios=(1.65, 1.03, 1.20), wspace=.38,
    )
    g_spec, h_spec, i_spec = [bottom[0, index] for index in range(3)]

    a_stats = draw_combined_model_panel(fig, a_spec, old, canonical)
    b_stats, sequence_image = draw_sequences(fig, b_spec, out, old, canonical)
    b_field_stats = draw_interictal_fields_and_shared_bar(
        fig, b_field_spec, out, canonical, sequence_image,
    )

    c_stats = draw_learning_curve(fig.add_subplot(c_spec), out, old)
    d_stats, d_source = draw_interictal_field_recovery(
        fig.add_subplot(d_spec), out, canonical,
    )
    e_stats, e_source = draw_early_seizure_case(fig.add_subplot(e_spec), out, canonical)
    f_stats, f_source = draw_early_seizure_cohort(fig.add_subplot(f_spec), out)

    g_stats, g_source = draw_repeated_response_matrix(
        fig, g_spec, response_root, out, canonical,
    )
    h_stats, h_source = draw_similarity_matrix(fig.add_subplot(h_spec), response_root)
    i_stats, i_source = draw_model_data_link(fig, i_spec, response_root)

    for label, spec in zip(
        "ABCDEFGHI",
        (a_spec, b_spec, c_spec, d_spec, e_spec, f_spec, g_spec, h_spec, i_spec),
    ):
        add_panel_letter(fig, spec, label)

    destination.mkdir(parents=True, exist_ok=True)
    source_dir = destination / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    d_source.to_csv(source_dir / "panel_d_interictal_generated_continuation.csv", index=False)
    e_source.to_csv(source_dir / "panel_e_interictal_vs_early_seizure_case.csv", index=False)
    f_source.to_csv(source_dir / "panel_f_early_seizure_direction_ignored.csv", index=False)
    g_source.to_csv(source_dir / "panel_g_repeated_patch_responses.csv", index=False)
    h_source.to_csv(source_dir / "panel_h_design_similarity_matrix.csv", index=False)
    i_source.to_csv(source_dir / "panel_i_response_vs_heldout_transitions.csv", index=False)

    stem = destination / STEM
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)

    assets = {path.name: r1.sha256_file(path) for path in (
        stem.with_suffix(".png"), stem.with_suffix(".pdf"), stem.with_suffix(".svg"),
    )}
    metadata = {
        "contract": "topic5_figure6_interictal_crossstate_response_r5_candidate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CANDIDATE_PENDING_USER_REVIEW",
        "panels": {
            "A": a_stats,
            "B": {"held_out_events": b_stats, "generated_interictal_fields": b_field_stats},
            "C": c_stats, "D": d_stats, "E": e_stats, "F": f_stats,
            "G": g_stats, "H": h_stats, "I": i_stats,
        },
        "assets_sha256": assets,
    }
    r1.write_json(destination / "FIGURE6_R5_METADATA.json", metadata)
    r1.write_json(destination / "FIGURE6_R5_COMPLETE.json", {
        "status": "COMPLETE_PENDING_USER_REVIEW", "assets_sha256": assets,
    })
    (destination / "README.md").write_text(
        "### topic5_figure6_interictal_crossstate_response_r5_candidate.png / .pdf / .svg\n\n"
        "A 把真实 SEEG/组织平面上的连续 rank-set 输入、圆框内 RNN、next-contact/STOP 输出和四种连接约束合为一体；B 展示代表性留出 TA/TB 事件、从相同第一 rank-set 开始的自由 rollout 及生成间期场，其中 TA/TB 只用于训练后分组，从未输入模型。\n\n"
        "C–F 依次展示下一触点留出 loss、删除起始触点后的间期续写场恢复、四种网络与同一患者发作早期场的直观比较，以及17位患者中不计方向的 TA/TB–发作早期场对应。\n\n"
        "G–I 依次展示同一组织小片在早中晚阶段多次扰动后的平均未来触点响应、不同训练条件的队列中位响应相似度，以及该响应形状与留出事件 contact-following 矩阵的比较；I 不是扰动前后预测准确率比较。\n\n"
        "**关注点**：真实顺序提高预测并恢复间期场；早期发作只支持不计方向的空间形状对应；精确连接"
        "不同的网络仍产生相似的有限时域响应，而且该响应超过保留电极杆结构的随机基线。\n"
    )
    print(json.dumps({"figure": str(stem.with_suffix('.png')), "assets": assets}, indent=2))


if __name__ == "__main__":
    main()
