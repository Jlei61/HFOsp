#!/usr/bin/env python3
"""Stage and final paper figures for the locked Topic 5 RNN motif benchmark."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_topic5_interictal_template_ab_fields import (  # noqa: E402
    build_interictal_ab_panel_payloads,
)
from plot_topic5_field_vs_ictal_swap import draw_topic5_field_panel  # noqa: E402
from paper_figures.plot_fig3b_interictal_ictal_shared_field import (  # noqa: E402
    _draw_field,
    _normalize_minmax,
)


MODEL_ORDER = [
    "M0_NO_REC", "M1_DENSE", "M2_UNIFORM_SET", "M3_FIXED_LOCAL",
    "M4_SPATIAL_GROWTH", "M6_SPATIAL_MID", "M8_UNIFORM_COST_MID",
    "C_ORDER_SHUFFLED",
]
MODEL_LABEL = {
    "M0_NO_REC": "No rec.", "M1_DENSE": "Dense", "M2_UNIFORM_SET": "Sparse",
    "M3_FIXED_LOCAL": "Local", "M4_SPATIAL_GROWTH": "Spatial",
    "M6_SPATIAL_MID": "Sp.+cost", "M8_UNIFORM_COST_MID": "Unif.+cost",
    "C_ORDER_SHUFFLED": "Shuffle",
}
COLORS = {
    "M0_NO_REC": "#9b9b9b", "M1_DENSE": "#252525", "M2_UNIFORM_SET": "#7f7f7f",
    "M3_FIXED_LOCAL": "#3b75af", "M4_SPATIAL_GROWTH": "#55a7a1",
    "M6_SPATIAL_MID": "#d64c4c", "M8_UNIFORM_COST_MID": "#e58a2d",
    "C_ORDER_SHUFFLED": "#6c65b8",
}
REPRESENTATIVE = "epilepsiae_1146"
REPRESENTATIVE_MODEL = "M6_SPATIAL_MID"


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        output = list(csv.DictReader(handle))
    for row in output:
        for key, value in list(row.items()):
            if key in {"subject", "fit_id", "scope", "model", "cell", "lesion", "endpoint",
                       "fit_aggregation", "producer_A", "producer_B", "seizure_id"}:
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_early(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    use = np.isfinite(values)
    out = np.full(values.shape, np.nan)
    if use.any():
        lo, hi = float(values[use].min()), float(values[use].max())
        out[use] = 0.0 if hi <= lo else 1.0 - (values[use] - lo) / (hi - lo)
    return out


def selected_metrics(out_root: Path, subject: str, model: str, cell: str = "rnn") -> Path:
    candidates = []
    for path in (out_root / "per_subject").glob(f"{subject}*" + f"/{model}__{cell}/seed*/metrics.json"):
        data = json.loads(path.read_text())
        candidates.append((path, float(data["validation"]["contact_nll"])))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint for {subject} {model} {cell}")
    median = float(np.median([value for _, value in candidates]))
    return min(candidates, key=lambda item: (abs(item[1] - median), str(item[0])))[0]


def empirical_record(out_root: Path, subject: str) -> dict[str, Any]:
    manifest = json.loads((out_root / "INPUT_MANIFEST.json").read_text())
    return json.loads((Path(manifest["input_roots"]["field"]) / f"{subject}.json").read_text())


def model_field_payloads(out_root: Path, subject: str, model: str, cell: str = "rnn"):
    record = empirical_record(out_root, subject)
    a_payload, b_payload, _ = build_interictal_ab_panel_payloads(record)
    path = out_root / "model_fields" / "per_patient" / subject / f"{model}__{cell}.npz"
    with np.load(path, allow_pickle=False) as data:
        names = np.asarray(data["A_contacts"]).astype(str).tolist()
        order = [str(value) for value in record["interictal_field"]["contact_order"]]
        for template, payload in (("A", a_payload), ("B", b_payload)):
            values = np.asarray(data[f"{template}_canonical_full"], float)
            lookup = dict(zip(names, values))
            payload["vals"] = normalize_early(np.asarray([lookup.get(name, np.nan) for name in order]))
    return record, a_payload, b_payload


def graph_data(out_root: Path, subject: str, model: str):
    metrics = selected_metrics(out_root, subject, model)
    graph = dict(np.load(metrics.parent / "graph.npz"))
    plane = dict(np.load(out_root / "cache" / json.loads(metrics.read_text())["fit_id"] / "plane.npz"))
    return metrics, graph, plane


def draw_graph(ax, graph: dict[str, np.ndarray], plane: dict[str, np.ndarray], title: str,
               influence: dict[str, np.ndarray] | None = None):
    xy = np.asarray(plane["nodes_xy_mm"], float)
    contacts = np.asarray(plane["contacts_xy_mm"], float)
    mask = np.asarray(graph["mask"], bool)
    edges = np.argwhere(mask)
    # Preserve the actual ten-fold density contrast.  Subsampling the dense
    # mask to approximately the sparse edge count makes distinct constraints
    # look falsely alike; a faint LineCollection keeps every edge readable as
    # a density cloud without producing a black hairball.
    segments = np.asarray([[xy[i], xy[j]] for i, j in edges], float)
    dense = len(edges) > 2 * xy.shape[0]
    ax.add_collection(LineCollection(
        segments, colors="#969696",
        linewidths=0.14 if dense and len(edges) > 1000 else 0.28,
        alpha=0.030 if dense and len(edges) > 1000 else 0.25,
        zorder=0, rasterized=len(edges) > 1000,
    ))
    if influence is not None:
        for key, color, width in (("local_backbone_mask", "#3b75af", 0.8),
                                  ("long_high_mask", "#d64c4c", 1.15)):
            for i, j in np.argwhere(np.asarray(influence[key], bool)):
                ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=color, lw=width, alpha=0.78, zorder=2)
        nodes = np.flatnonzero(np.asarray(influence["connector_nodes"], bool))
        if len(nodes):
            ax.scatter(xy[nodes, 0], xy[nodes, 1], s=24, facecolors="none",
                       edgecolors="#d64c4c", linewidths=1.1, zorder=4)
    ax.scatter(xy[:, 0], xy[:, 1], s=3.5, color="#626262", alpha=0.75, zorder=3)
    ax.scatter(contacts[:, 0], contacts[:, 1], s=17, facecolor="white", edgecolor="#111111",
               linewidth=0.65, zorder=5)
    ax.set_title(title, fontsize=8.5, pad=2.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)


def draw_motif_ladder(parent, out_root: Path):
    grid = parent.subgridspec(1, 4, wspace=0.04)
    for index, model in enumerate(("M1_DENSE", "M2_UNIFORM_SET", "M3_FIXED_LOCAL", "M6_SPATIAL_MID")):
        ax = parent.get_gridspec().figure.add_subplot(grid[0, index])
        _, graph, plane = graph_data(out_root, REPRESENTATIVE, model)
        draw_graph(ax, graph, plane, MODEL_LABEL[model])


def rollout_matrices(out_root: Path):
    metrics_path = selected_metrics(out_root, REPRESENTATIVE, REPRESENTATIVE_MODEL)
    metrics = json.loads(metrics_path.read_text())
    cache = out_root / "cache" / metrics["fit_id"]
    events = dict(np.load(cache / "events.npz"))
    keep = events["split"] >= 0
    ranks = np.asarray(events["ranks"])[keep]
    mode = np.asarray(events["mode"])[keep]
    split = np.asarray(events["split"])[keep]
    provenance = json.loads((cache / "provenance.json").read_text())
    with gzip.open(metrics_path.parent / "heldout_rollouts.json.gz", "rt") as handle:
        generated = json.load(handle)
    by_index = {int(row["kept_event_index"]): row["generated_rank_sets"] for row in generated}
    output = {}
    record = empirical_record(out_root, REPRESENTATIVE)["interictal_field"]
    contact_order = [str(value) for value in record["contact_order"]]
    contacts = [str(value) for value in provenance["contacts"]]
    reorder = np.asarray([contacts.index(name) for name in contact_order], int)
    sort_y = np.argsort(np.asarray(record["rank_a"], float))
    for template in ("A", "B"):
        indices = [index for index in np.flatnonzero(split == 2)
                   if provenance["mode_to_template"].get(str(int(mode[index]))) == template.lower()
                   and index in by_index]
        indices = indices[:min(28, len(indices))]
        observed, predicted = [], []
        for index in indices:
            obs = ranks[index].astype(float); obs[obs < 0] = np.nan
            pred = np.full(len(contacts), np.nan)
            for rank, rank_set in enumerate(by_index[index]): pred[np.asarray(rank_set, int)] = rank
            for source, store in ((obs, observed), (pred, predicted)):
                value = source[reorder][sort_y]
                use = np.isfinite(value)
                if use.any() and np.nanmax(value) > 0: value[use] /= np.nanmax(value)
                store.append(value)
        output[template] = (np.asarray(observed).T, np.asarray(predicted).T)
    return output


def draw_rollout_example(parent, out_root: Path):
    matrices = rollout_matrices(out_root)
    grid = parent.subgridspec(2, 2, hspace=0.10, wspace=0.08)
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#d9d9d9")
    seed_cmap = ListedColormap(["#c43d4d"])
    for row, template in enumerate(("A", "B")):
        for col, matrix in enumerate(matrices[template]):
            ax = parent.get_gridspec().figure.add_subplot(grid[row, col])
            ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
            seed = np.where(np.isfinite(matrix) & np.isclose(matrix, 0.0), 1.0, np.nan)
            ax.imshow(
                seed, aspect="auto", interpolation="nearest", cmap=seed_cmap,
                vmin=0, vmax=1,
            )
            if row == 0:
                ax.set_title(
                    ("Observed", "Given seed → free rollout")[col],
                    fontsize=8.5, pad=2,
                )
            if col == 0: ax.set_ylabel(f"T{template}\ncontacts", fontsize=8)
            else: ax.set_yticks([])
            ax.set_xticks([])
            if row == 1: ax.set_xlabel("held-out events", fontsize=7.5)
            for spine in ax.spines.values(): spine.set_linewidth(0.6)


def strip(ax, data: dict[str, np.ndarray], ylabel: str, zero: bool = False):
    models = [model for model in MODEL_ORDER if model in data]
    for x, model in enumerate(models):
        values = np.asarray(data[model], float); values = values[np.isfinite(values)]
        jitter = np.linspace(-0.13, 0.13, len(values)) if len(values) else np.asarray([])
        ax.scatter(x + jitter, values, s=12, alpha=0.6, color=COLORS[model], linewidths=0)
        if len(values): ax.plot([x - 0.22, x + 0.22], [np.median(values)] * 2, color="#111111", lw=1.25)
    if zero: ax.axhline(0, color="#8d8d8d", lw=0.7)
    ax.set_xticks(
        range(len(models)), [MODEL_LABEL[model] for model in models],
        rotation=45, ha="right", rotation_mode="anchor",
    )
    ax.tick_params(axis="x", labelsize=7.2, pad=2)
    ax.set_ylabel(ylabel)
    return models


def draw_interictal_sufficiency(parent, out_root: Path):
    frame = [row for row in rows(out_root / "interictal_per_patient.csv") if row["cell"] == "rnn"]
    lookup = {(row["subject"], row["model"]): row for row in frame}
    subjects = sorted({row["subject"] for row in frame if row["model"] == "M0_NO_REC"})
    models = [model for model in MODEL_ORDER if all((subject, model) in lookup for subject in subjects)]
    gain = {model: [lookup[(subject, "M0_NO_REC")]["contact_nll"]
                    - lookup[(subject, model)]["contact_nll"] for subject in subjects]
            for model in models if model != "M0_NO_REC"}
    rollout = {model: [lookup[(subject, model)]["rollout_spearman"] for subject in subjects]
               for model in models}
    grid = parent.subgridspec(1, 2, wspace=0.48)
    figure = parent.get_gridspec().figure
    ax1 = figure.add_subplot(grid[0, 0]); ax2 = figure.add_subplot(grid[0, 1])
    strip(ax1, gain, "Recurrence gain\n(Δ NLL)", zero=True)
    strip(ax2, rollout, "Free-rollout\nrank correlation", zero=True)


def draw_stage_interictal(parent, out_root: Path):
    """Diagnostic six-panel readout; the final figure uses the compact subset."""
    frame = rows(out_root / "interictal_per_patient.csv")
    rnn = [row for row in frame if row["cell"] == "rnn"]
    lookup = {(row["subject"], row["model"]): row for row in rnn}
    subjects = sorted({row["subject"] for row in rnn if row["model"] == "M0_NO_REC"})
    models = [model for model in MODEL_ORDER
              if all((subject, model) in lookup for subject in subjects)]
    # Repeating the same long, rotated model labels on the upper row makes
    # constrained_layout reserve a large empty band between rows.  The lower
    # row carries the complete model labels, while the upper row shows only
    # the measurements.
    grid = parent.subgridspec(2, 3, hspace=0.16, wspace=0.42)
    figure = parent.get_gridspec().figure
    axes = [figure.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]

    for axis, metric, label in (
        (axes[0], "contact_nll", "Next-contact gain\n(M0 NLL − model NLL)"),
        (axes[1], "stop_bce", "STOP gain\n(M0 BCE − model BCE)"),
    ):
        data = {model: [lookup[(subject, "M0_NO_REC")][metric]
                        - lookup[(subject, model)][metric] for subject in subjects]
                for model in models if model != "M0_NO_REC"}
        strip(axis, data, label, zero=True)
    strip(axes[2], {model: [lookup[(subject, model)]["rollout_spearman"]
                            for subject in subjects] for model in models},
          "Seed-removed rollout\nrank correlation", zero=True)
    strip(axes[3], {model: [lookup[(subject, model)]["length_ratio"]
                            for subject in subjects] for model in models},
          "Generated / observed\npost-seed contacts", zero=False)
    axes[3].axhline(1, color="#8d8d8d", lw=0.7)

    retention_rows = rows(out_root / "dense_benefit_retention.csv")
    retention = {
        model: [row["dense_benefit_retention"] for row in retention_rows
                if row["cell"] == "rnn" and row["model"] == model
                and np.isfinite(float(row["dense_benefit_retention"]))]
        for model in models if model not in {"M0_NO_REC", "M1_DENSE"}
    }
    strip(axes[4], retention, "Dense benefit retained", zero=True)
    axes[4].axhline(1, color="#8d8d8d", lw=0.7)

    gru = {(row["subject"], row["model"]): row for row in frame if row["cell"] == "gru"}
    shared_models = [model for model in models
                     if all((subject, model) in gru for subject in subjects)]
    architecture = {
        model: [gru[(subject, model)]["rollout_spearman"]
                - lookup[(subject, model)]["rollout_spearman"] for subject in subjects]
        for model in shared_models
    }
    strip(axes[5], architecture, "GRU − leaky RNN\nrollout correlation", zero=True)
    for axis in axes[:3]:
        axis.tick_params(axis="x", bottom=False, labelbottom=False)
    for label, axis in zip("abcdef", axes):
        axis.text(-0.18, 1.04, label, transform=axis.transAxes, fontsize=11,
                  fontweight="bold", ha="right", va="bottom")


def draw_pareto(ax, out_root: Path):
    inter = [row for row in rows(out_root / "interictal_per_patient.csv") if row["cell"] == "rnn"]
    fidelity = rows(out_root / "model_field_patient_metrics.csv")
    f_lookup = {(row["subject"], row["model"], row["cell"]): row for row in fidelity}
    for model in MODEL_ORDER:
        points = [(row["c_wiring"], f_lookup[(row["subject"], model, "rnn")]["matched_empirical_r"])
                  for row in inter if row["model"] == model and (row["subject"], model, "rnn") in f_lookup]
        if not points: continue
        values = np.asarray(points, float)
        ax.scatter(values[:, 0], values[:, 1], s=8, color=COLORS[model], alpha=0.18, linewidths=0)
        ax.scatter(np.nanmedian(values[:, 0]), np.nanmedian(values[:, 1]), s=45,
                   color=COLORS[model], edgecolor="white", linewidth=0.7, label=MODEL_LABEL[model], zorder=4)
    ax.set_xlabel("Normalized wiring cost")
    ax.set_ylabel("Interictal field fidelity")
    ax.axhline(0, color="#aaaaaa", lw=0.6)


def early_activation(out_root: Path, subject: str):
    record = empirical_record(out_root, subject)
    order = [str(value) for value in record["interictal_field"]["contact_order"]]
    values = []
    for path in locked_early_target_paths(out_root, subject):
        with np.load(path, allow_pickle=False) as data:
            lookup = dict(zip(np.asarray(data["contact_names"]).astype(str),
                              np.asarray(data["target_1_150"], float)))
        values.append([lookup.get(name, np.nan) for name in order])
    if not values:
        raise RuntimeError(f"no frozen early-ictal artifacts for {subject}")
    return np.nanmedian(np.asarray(values, float), axis=0)


def locked_early_target_paths(out_root: Path, subject: str) -> list[Path]:
    """Resolve the exact target bytes frozen before unseal for figure rendering."""
    inventory = rows(out_root / "early_ictal_metadata_inventory.csv")
    selected = []
    for row in inventory:
        if str(row["subject"]) != subject:
            continue
        path = Path(str(row["artifact_path"])).resolve()
        if not path.is_file():
            raise RuntimeError(f"frozen early-ictal artifact is missing: {path}")
        if sha256(path) != str(row["artifact_sha256"]):
            raise RuntimeError(f"frozen early-ictal artifact hash changed: {path}")
        selected.append(path)
    return sorted(selected)


def draw_cross_state(parent, out_root: Path, include_stats: bool = True,
                     model_subset: tuple[str, ...] | None = None):
    _, pa, pb = model_field_payloads(out_root, REPRESENTATIVE, REPRESENTATIVE_MODEL)
    activation = early_activation(out_root, REPRESENTATIVE)
    # Use the exact empirical record frozen in INPUT_MANIFEST.  A worktree-local
    # ``results/interictal_propagation_masked`` tree is gitignored and may not
    # exist in a clean execution worktree; resolving it implicitly would make
    # rendering depend on an unmanifested ambient file.
    fz = empirical_record(out_root, REPRESENTATIVE)
    columns = 3
    grid = parent.subgridspec(2 if include_stats else 1, columns,
                              height_ratios=[1.0, 0.72] if include_stats else [1.0],
                              hspace=0.42, wspace=0.13)
    for col, (payload, title) in enumerate(((pa, "Model TA"), (pb, "Model TB"))):
        ax = parent.get_gridspec().figure.add_subplot(grid[0, col])
        draw_topic5_field_panel(ax, payload, payload["vals"], title, "", compact=True,
                                labels=False, cbar=False, contact_size=20, contact_outline_lw=0.55)
    ax = parent.get_gridspec().figure.add_subplot(grid[0, 2])
    _draw_field(ax, fz, _normalize_minmax(activation), np.asarray(fz["support_a"], float),
                cmap="magma_r", colorbar_values=activation, title="Early ictal",
                title_color="#111111", show_y=False)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    if include_stats:
        stat_ax = parent.get_gridspec().figure.add_subplot(grid[1, :])
        frame = [row for row in rows(out_root / "early_ictal_per_patient_model.csv")
                 if row["primary"] in (True, "True", "1", 1.0) and row["cell"] == "rnn"]
        models = []
        for model in (model_subset or tuple(MODEL_ORDER)):
            canonical = [row["all_contact_margin"] for row in frame
                         if row["model"] == model and row["endpoint"] == "canonical_full"]
            seed_removed = [row["all_contact_margin"] for row in frame
                            if row["model"] == model and row["endpoint"] == "seed_removed"]
            if not canonical or not seed_removed:
                continue
            models.append(model)
            for offset, values, marker, filled in (
                    (-0.12, canonical, "o", True), (0.12, seed_removed, "D", False)):
                values = np.asarray(values, float)
                jitter = np.linspace(-0.055, 0.055, len(values))
                stat_ax.scatter(
                    len(models) - 1 + offset + jitter, values, s=12, marker=marker,
                    facecolor=COLORS[model] if filled else "white",
                    edgecolor=COLORS[model], linewidth=0.65, alpha=0.72,
                )
                stat_ax.plot(
                    [len(models) - 1 + offset - 0.08, len(models) - 1 + offset + 0.08],
                    [np.nanmedian(values)] * 2, color="#111111", lw=1.15,
                )
        stat_ax.axhline(0, color="#8d8d8d", lw=0.7)
        stat_ax.set_xticks(
            range(len(models)), [MODEL_LABEL[model] for model in models],
            rotation=45, ha="right", rotation_mode="anchor",
        )
        stat_ax.tick_params(axis="x", labelsize=7.2, pad=2)
        stat_ax.set_ylabel("Early-ictal\nnull-relative margin")
        stat_ax.legend(
            handles=[
                Line2D([], [], marker="o", color="#555555", markerfacecolor="#555555",
                       lw=0, markersize=4.5, label="Canonical full"),
                Line2D([], [], marker="D", color="#555555", markerfacecolor="white",
                       lw=0, markersize=4.5, label="Seed removed"),
            ],
            loc="upper left", frameon=False, ncol=2, fontsize=7.0,
            handletextpad=0.35, columnspacing=0.8, borderaxespad=0.1,
        )


def influence_for_selected(out_root: Path):
    metrics = selected_metrics(out_root, REPRESENTATIVE, REPRESENTATIVE_MODEL)
    path = (out_root / "effective_influence" / metrics.parents[2].name
            / metrics.parents[1].name / metrics.parent.name / "influence.npz")
    return metrics, dict(np.load(path, allow_pickle=False))


def patient_level_effective_reach(out_root: Path) -> dict[str, np.ndarray]:
    """Collapse fit and seed rows before exposing the patient denominator."""
    frame = [row for row in rows(out_root / "effective_influence_fit_seed.csv")
             if row["model"] == REPRESENTATIVE_MODEL and row["cell"] == "rnn"]
    output: dict[str, np.ndarray] = {}
    for subject in sorted({row["subject"] for row in frame}):
        selected = [row for row in frame if row["subject"] == subject]
        output[subject] = np.asarray([
            np.nanmedian([row[f"lag{lag}_reach_mm"] for row in selected])
            for lag in (1, 2, 3)
        ], float)
    return output


def lesion_display_values(out_root: Path) -> list[tuple[str, np.ndarray]]:
    """Return the preassigned lesion components that have cohort support.

    Local-backbone damage is required because it is one half of the frozen
    motif.  Long-range high-influence edges and connector nodes are then
    admitted in a fixed order when each has at least five valid patient-level
    estimates.  This rule depends only on estimability, never effect size.
    """
    lesion_path = out_root / "matched_lesion_patient_metrics.csv"
    lesion_rows = rows(lesion_path) if lesion_path.exists() else []
    lesion_order = (
        ("local_backbone_edges", "Local\nbackbone"),
        ("long_range_high_influence_edges", "Long-range\nedges"),
        ("connector_nodes", "Connector\nincident edges"),
    )
    available: dict[str, np.ndarray] = {}
    for lesion, _ in lesion_order:
        by_subject: dict[str, list[float]] = {}
        for row in lesion_rows:
            value = float(row["specificity_contact_nll"])
            if (row["model"] == REPRESENTATIVE_MODEL and row["cell"] == "rnn"
                    and row["lesion"] == lesion
                    and row["all_inference_available"] in (True, "True", "1", 1.0)
                    and np.isfinite(value)):
                by_subject.setdefault(str(row["subject"]), []).append(value)
        available[lesion] = np.asarray([
            np.median(by_subject[subject]) for subject in sorted(by_subject)
        ], float)
    if len(available["local_backbone_edges"]) < 5:
        return []
    selected = [(lesion_order[0][1], available[lesion_order[0][0]])]
    selected.extend(
        (label, available[lesion]) for lesion, label in lesion_order[1:]
        if len(available[lesion]) >= 5
    )
    return selected if len(selected) >= 2 else []


def draw_reach_or_lesion(ax, out_root: Path):
    """Use lesion specificity only when the frozen motif is estimable.

    The display rule uses only the number of valid matched controls, not effect
    size or significance.  Otherwise it falls back to patient-level open-loop
    reach, which is defined for every patient and does not overstate an
    underpowered lesion analysis.
    """
    lesion_display = lesion_display_values(out_root)
    if lesion_display:
        for x, (_, values) in enumerate(lesion_display):
            jitter = np.linspace(-0.10, 0.10, len(values))
            ax.scatter(x + jitter, values, s=17, color=COLORS[REPRESENTATIVE_MODEL],
                       alpha=0.68, linewidths=0)
            ax.plot([x - 0.17, x + 0.17], [np.median(values)] * 2,
                    color="#111111", lw=1.4)
        ax.axhline(0, color="#8d8d8d", lw=0.7)
        ax.set_xticks(
            range(len(lesion_display)), [label for label, _ in lesion_display]
        )
        ax.set_ylabel("Damage beyond\nmatched lesion (Δ NLL)")
        ax.set_title("Matched perturbation", fontsize=8.5, pad=2.5)
        return

    reach = patient_level_effective_reach(out_root)
    values = np.asarray(list(reach.values()), float)
    for value in values:
        ax.plot([1, 2, 3], value, color="#b8b8b8", lw=0.55, alpha=0.55)
        ax.scatter([1, 2, 3], value, s=8, color="#b8b8b8", alpha=0.55, linewidths=0)
    median = np.nanmedian(values, axis=0)
    ax.plot([1, 2, 3], median, color=COLORS[REPRESENTATIVE_MODEL], lw=2.0,
            marker="o", markersize=4.2, zorder=4)
    ax.set_xticks([1, 2, 3], ["1", "2", "3"])
    ax.set_xlabel("Rank steps after pulse")
    ax.set_ylabel("Effective reach (mm)")
    ax.set_title(f"Open-loop reach (n={len(values)} patients)", fontsize=8.5, pad=2.5)


def draw_effective_motif(parent, out_root: Path):
    metrics, influence = influence_for_selected(out_root)
    graph = dict(np.load(metrics.parent / "graph.npz"))
    fit_id = json.loads(metrics.read_text())["fit_id"]
    plane = dict(np.load(out_root / "cache" / fit_id / "plane.npz"))
    grid = parent.subgridspec(1, 2, wspace=0.30)
    ax = parent.get_gridspec().figure.add_subplot(grid[0, 0])
    draw_graph(ax, graph, plane, "Effective motif", influence)
    ax = parent.get_gridspec().figure.add_subplot(grid[0, 1])
    draw_reach_or_lesion(ax, out_root)


def panel_label(fig, subplot_spec, label: str):
    box = subplot_spec.get_position(fig)
    fig.text(box.x0 - 0.018, box.y1 + 0.006, label, fontsize=13, fontweight="bold",
             ha="right", va="bottom")


def render_stage(out_root: Path, stage: str):
    size = (11.4, 5.5) if stage == "interictal" else (9.2, 3.2)
    fig = plt.figure(figsize=size, layout="constrained", facecolor="white")
    root = fig.add_gridspec(1, 1)[0, 0]
    if stage == "interictal":
        draw_stage_interictal(root, out_root)
    elif stage == "fields":
        record = empirical_record(out_root, REPRESENTATIVE)
        empirical_a, empirical_b, _ = build_interictal_ab_panel_payloads(record)
        _, model_a, model_b = model_field_payloads(out_root, REPRESENTATIVE, REPRESENTATIVE_MODEL)
        grid = root.subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.15], wspace=0.15)
        for col, (payload, title) in enumerate(((empirical_a, "Data TA"), (model_a, "RNN TA"),
                                                (empirical_b, "Data TB"), (model_b, "RNN TB"))):
            ax = fig.add_subplot(grid[0, col])
            draw_topic5_field_panel(ax, payload, payload["vals"], title, "", compact=True,
                                    contact_size=23, contact_outline_lw=0.6)
        ax = fig.add_subplot(grid[0, 4])
        frame = rows(out_root / "model_field_patient_metrics.csv")
        data = {model: [row["matched_empirical_r"] for row in frame
                        if row["model"] == model and row["cell"] == "rnn"] for model in MODEL_ORDER}
        strip(ax, {key: value for key, value in data.items() if value}, "Field fidelity", zero=True)
    elif stage == "early":
        draw_cross_state(root, out_root, include_stats=True)
    elif stage == "motif":
        draw_effective_motif(root, out_root)
    else:
        raise ValueError(stage)
    stem = out_root / "figures" / f"stage_{stage}_scientific_readout"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_final(out_root: Path):
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "axes.linewidth": 0.7})
    fig = plt.figure(figsize=(11.2, 8.4), facecolor="white")
    outer = fig.add_gridspec(3, 2, hspace=0.34, wspace=0.22,
                             left=0.055, right=0.985, top=0.975, bottom=0.07)
    draw_motif_ladder(outer[0, 0], out_root); panel_label(fig, outer[0, 0], "A")
    draw_rollout_example(outer[0, 1], out_root); panel_label(fig, outer[0, 1], "B")
    draw_interictal_sufficiency(outer[1, 0], out_root); panel_label(fig, outer[1, 0], "C")
    ax_d = fig.add_subplot(outer[1, 1]); draw_pareto(ax_d, out_root); panel_label(fig, outer[1, 1], "D")
    draw_cross_state(
        outer[2, 0], out_root, include_stats=True,
        model_subset=("M0_NO_REC", "M1_DENSE", "M3_FIXED_LOCAL",
                      "M6_SPATIAL_MID", "C_ORDER_SHUFFLED"),
    ); panel_label(fig, outer[2, 0], "E")
    draw_effective_motif(outer[2, 1], out_root); panel_label(fig, outer[2, 1], "F")
    handles, labels = ax_d.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(8, len(labels)), frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=7.5, handletextpad=0.4, columnspacing=1.0)
    figure_dir = out_root / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "topic5_figure6_rnn_connectivity_motifs"
    for suffix, dpi in ((".png", 600), (".pdf", None), (".svg", None)):
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if dpi: kwargs["dpi"] = dpi
        fig.savefig(stem.with_suffix(suffix), **kwargs)
    plt.close(fig)
    readme = figure_dir / "README.md"
    marker = "<!-- topic5-rnn-motif-v0.4-stage-and-final-figures -->"
    existing = readme.read_text() if readme.exists() else "# Topic 5 RNN connectivity-motif figures\n"
    base = existing.split(marker, 1)[0].rstrip()
    section = """<!-- topic5-rnn-motif-v0.4-stage-and-final-figures -->

### topic5_figure6_rnn_connectivity_motifs.png / .pdf / .svg

六联图依次展示真实患者几何上的连接约束、同患者留出间期事件的真实与自由生成 A/B 时序、全队列间期预测充分性、传播场拟合与布线成本、冻结模型场与临床发作早期能量场，以及 target-free 有效连接组织。Panel B 的红格是观察到并提供给模型的第一 rank seed，其余 viridis 格才显示后续观察或自由推演。若预先指定的 local-backbone lesion 以及 long-range-edge 或 connector-node lesion 达到最低患者分母，Panel F 展示所有可估计成分的 matched-lesion 特异损害；否则展示患者级 lag-1/2/3 open-loop effective reach。所有统计先在患者内合并，所有场使用冻结触点顺序和几何；发作数值只在模型与场完全冻结后进入 Panel E。

**关注点**：图的承重顺序是“能生成间期传播 → 哪些结构更经济 → 哪些冻结场跨状态对应 → 哪些有效 motif 经干预承担该计算”，不把预测性能直接写成真实连接组恢复。

### stage_interictal_scientific_readout.png / .pdf

正式间期模型矩阵的六项阶段验收：next-contact、STOP、删除起点后的自由推演、生成长度、dense-benefit retention，以及 GRU 相对 leaky RNN 的复现；每个点是一位患者。

**关注点**：只有同时改善局部预测并保持自由推演，才将某种连接约束称为足以表示患者内传播；这仍不是解剖连接恢复。

### stage_fields_scientific_readout.png / .pdf

目标未解封前的间期场验收图：同一代表患者的数据 TA/TB 场与冻结 RNN 生成 TA/TB 场，并列全队列场一致性。

**关注点**：先确认模型场来自留出间期推演且保留 A/B，而不是由发作 target 反向选择。

### stage_early_scientific_readout.png / .pdf

冻结模型 TA/TB 场、患者平均 clinical-onset 0–10 s broadband 1–150 Hz 能量场和主队列 null-relative margin。

**关注点**：个体场只作直观例子，队列患者点才承担跨状态统计；canonical full 是主量，seed-removed 为机制性次量。

### stage_motif_scientific_readout.png / .pdf

代表患者的局部高影响骨架与少量长程 connector。右侧只有在 local backbone 与至少一种预先指定的 long-range/connector lesion 均达到最低可估计患者数时，才展示所有达到分母的 matched-lesion 特异损害；connector 操作切断所选 tissue node 的全部入/出 recurrent edges，但保留直接输入与 observation readout，因此不是完整 node ablation。否则固定展示全队列患者级 open-loop effective reach。

**关注点**：只有结构富集、任务关系和 matched-lesion 同向时，才把该组织写成更容易支持传播的计算 motif。

### stage_c_smoke_training_and_decoder.png / .pdf

三位开发患者上的工程 smoke：检查模型前向、梯度、冻结 free-rollout decoder、显存与 checkpoint schema。该图不承担患者队列的科学统计，也没有读取 early-ictal target。

**关注点**：这里只证明同一训练与解码合同可以稳定执行，不能把 smoke 性能写成科学筛选。

### stage_d_interictal_model_matrix.png / .pdf

正式间期分析的原始阶段诊断图，展示各模型 next-contact、STOP、自由推演和生成长度等患者级分布；与 `stage_interictal_scientific_readout` 使用同一冻结汇总，但保留较完整的工程读数。

**关注点**：用于核对每个模型没有靠事件长度或 STOP 单项获得表面优势。

### stage_e_target_free_model_fields.png / .pdf

early-ictal 解封前生成的完整模型场诊断图，展示代表患者两种起点场与全队列 field-fidelity 分布。它与 `stage_fields_scientific_readout` 使用同一 field manifest，未使用发作数值挑选模型或场。

**关注点**：确认共享和 non-collinear fit 的 A/B 聚合路径正确，并保留 canonical full 与 seed-removed 两个冻结端点。
"""
    readme.write_text(base + "\n\n" + section.lstrip())
    representative_metrics = selected_metrics(out_root, REPRESENTATIVE, REPRESENTATIVE_MODEL)
    representative_fit = json.loads(representative_metrics.read_text())["fit_id"]
    representative_plane = out_root / "cache" / representative_fit / "plane.npz"
    representative_events = out_root / "cache" / representative_fit / "events.npz"
    representative_provenance = out_root / "cache" / representative_fit / "provenance.json"
    input_manifest = json.loads((out_root / "INPUT_MANIFEST.json").read_text())
    representative_empirical = (
        Path(input_manifest["input_roots"]["field"]) / f"{REPRESENTATIVE}.json"
    )
    representative_influence = (
        out_root / "effective_influence" / representative_metrics.parents[2].name
        / representative_metrics.parents[1].name / representative_metrics.parent.name
        / "influence.npz"
    )
    representative_targets = locked_early_target_paths(out_root, REPRESENTATIVE)
    sources = {
        "A": [item for model in (
                  "M1_DENSE", "M2_UNIFORM_SET", "M3_FIXED_LOCAL", "M6_SPATIAL_MID"
              ) for item in (
                  selected_metrics(out_root, REPRESENTATIVE, model),
                  selected_metrics(out_root, REPRESENTATIVE, model).parent / "graph.npz",
              )] + [representative_plane],
        "B": [selected_metrics(out_root, REPRESENTATIVE, REPRESENTATIVE_MODEL).parent
              / "heldout_rollouts.json.gz", representative_metrics,
              representative_events, representative_provenance, representative_empirical],
        "C": [out_root / "interictal_per_patient.csv", out_root / "interictal_per_event.csv"],
        "D": [out_root / "accuracy_wiring_pareto.csv", out_root / "model_field_patient_metrics.csv"],
        "E": [out_root / "early_ictal_per_patient_model.csv",
              out_root / "early_ictal_null_matrices.npz",
              out_root / "MODEL_FIELD_MANIFEST.json",
              out_root / "model_fields/per_patient" / REPRESENTATIVE
              / f"{REPRESENTATIVE_MODEL}__rnn.npz", representative_empirical]
             + representative_targets,
        "F": [out_root / "effective_influence_fit_seed.csv",
              out_root / "matched_lesion_patient_metrics.csv",
              representative_metrics, representative_metrics.parent / "graph.npz",
              representative_plane,
              representative_influence],
    }
    panel_sources = {
        panel: [{"path": str(path), "sha256": sha256(path)} for path in paths]
        for panel, paths in sources.items()
    }
    manifest = {
        "_contract": "topic5_figure6_source_manifest_v0_4",
        "_representative_selection": {
            "patient": REPRESENTATIVE,
            "role": "target-free preassigned supportive visualization; excluded from primary p-values",
            "checkpoint_rule": (
                "within patient/model/cell, choose validation contact NLL nearest the seed median; "
                "ties resolved by lexical path"
            ),
            "selected_metrics": {
                model: str(selected_metrics(out_root, REPRESENTATIVE, model))
                for model in ("M1_DENSE", "M2_UNIFORM_SET", "M3_FIXED_LOCAL", "M6_SPATIAL_MID")
            },
        },
        **panel_sources,
    }
    (figure_dir / "figure6_source_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("interictal", "fields", "early", "motif", "final"),
                        required=True)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    if args.stage == "final": render_final(out_root)
    else: render_stage(out_root, args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
