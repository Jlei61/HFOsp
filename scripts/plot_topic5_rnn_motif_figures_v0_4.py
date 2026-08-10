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
from matplotlib.colors import Normalize
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
from plot_topic5_interictal_event_envelope_field import load_frozen  # noqa: E402


MODEL_ORDER = [
    "M0_NO_REC", "M1_DENSE", "M2_UNIFORM_SET", "M3_FIXED_LOCAL",
    "M4_SPATIAL_GROWTH", "M6_SPATIAL_MID", "M8_UNIFORM_COST_MID",
    "C_ORDER_SHUFFLED",
]
MODEL_LABEL = {
    "M0_NO_REC": "No rec.", "M1_DENSE": "Dense", "M2_UNIFORM_SET": "Sparse",
    "M3_FIXED_LOCAL": "Local", "M4_SPATIAL_GROWTH": "Spatial",
    "M6_SPATIAL_MID": "Sp. + cost", "M8_UNIFORM_COST_MID": "Unif. + cost",
    "C_ORDER_SHUFFLED": "Order shuffle",
    "EMPIRICAL_REFERENCE": "Empirical field",
}
COLORS = {
    "M0_NO_REC": "#9b9b9b", "M1_DENSE": "#252525", "M2_UNIFORM_SET": "#7f7f7f",
    "M3_FIXED_LOCAL": "#3b75af", "M4_SPATIAL_GROWTH": "#55a7a1",
    "M6_SPATIAL_MID": "#d64c4c", "M8_UNIFORM_COST_MID": "#e58a2d",
    "C_ORDER_SHUFFLED": "#6c65b8",
    "EMPIRICAL_REFERENCE": "#b83b3b",
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
    if len(edges) > 450:
        edges = edges[np.linspace(0, len(edges) - 1, 450).astype(int)]
    for i, j in edges:
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color="#b9b9b9", lw=0.28, alpha=0.25, zorder=0)
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
    for row, template in enumerate(("A", "B")):
        for col, matrix in enumerate(matrices[template]):
            ax = parent.get_gridspec().figure.add_subplot(grid[row, col])
            ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
            if row == 0: ax.set_title(("Observed", "Generated")[col], fontsize=8.5, pad=2)
            if col == 0: ax.set_ylabel(f"T{template}\ncontacts", fontsize=8)
            else: ax.set_yticks([])
            ax.set_xticks([])
            if row == 1: ax.set_xlabel("held-out events", fontsize=7.5)
            for spine in ax.spines.values(): spine.set_linewidth(0.6)


def strip(ax, data: dict[str, np.ndarray], ylabel: str, zero: bool = False,
          model_order: list[str] | None = None):
    models = [model for model in (model_order or MODEL_ORDER) if model in data]
    for x, model in enumerate(models):
        values = np.asarray(data[model], float); values = values[np.isfinite(values)]
        jitter = np.linspace(-0.13, 0.13, len(values)) if len(values) else np.asarray([])
        ax.scatter(x + jitter, values, s=12, alpha=0.6, color=COLORS[model], linewidths=0)
        if len(values): ax.plot([x - 0.22, x + 0.22], [np.median(values)] * 2, color="#111111", lw=1.25)
    if zero: ax.axhline(0, color="#8d8d8d", lw=0.7)
    ax.set_xticks(range(len(models)), [MODEL_LABEL[model] for model in models], rotation=35, ha="right")
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
    grid = parent.subgridspec(2, 3, hspace=0.58, wspace=0.42)
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
    ax.set_xlabel("Mean edge strength × distance / 10 mm")
    ax.set_ylabel("Interictal field fidelity")
    ax.axhline(0, color="#aaaaaa", lw=0.6)


def early_activation(out_root: Path, subject: str):
    metadata = json.loads((out_root / "EARLY_ICTAL_METADATA_INVENTORY.json").read_text())
    target_root = Path(metadata["target_cache_root"])
    record = empirical_record(out_root, subject)
    order = [str(value) for value in record["interictal_field"]["contact_order"]]
    values = []
    for path in sorted((target_root / f"outer_{subject}").glob(f"{subject}__*.npz")):
        with np.load(path, allow_pickle=False) as data:
            lookup = dict(zip(np.asarray(data["contact_names"]).astype(str),
                              np.asarray(data["target_1_150"], float)))
        values.append([lookup.get(name, np.nan) for name in order])
    return np.nanmedian(np.asarray(values, float), axis=0)


def draw_cross_state(parent, out_root: Path, include_stats: bool = True):
    _, pa, pb = model_field_payloads(out_root, REPRESENTATIVE, REPRESENTATIVE_MODEL)
    activation = early_activation(out_root, REPRESENTATIVE)
    fz = load_frozen(REPRESENTATIVE)
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
                 if row["primary"] in (True, "True", "1", 1.0)
                 and row["endpoint"] == "canonical_full"]
        display = ["EMPIRICAL_REFERENCE", "M0_NO_REC", "M1_DENSE", "M3_FIXED_LOCAL",
                   "M6_SPATIAL_MID", "M8_UNIFORM_COST_MID", "C_ORDER_SHUFFLED"]
        data = {}
        for model in display:
            expected_cell = "reference" if model == "EMPIRICAL_REFERENCE" else "rnn"
            values = [row["all_contact_margin"] for row in frame
                      if row["cell"] == expected_cell and row["model"] == model]
            if values: data[model] = values
        strip(stat_ax, data, "Early-ictal margin", zero=True, model_order=display)


def influence_for_selected(out_root: Path):
    metrics = selected_metrics(out_root, REPRESENTATIVE, REPRESENTATIVE_MODEL)
    path = (out_root / "effective_influence" / metrics.parents[2].name
            / metrics.parents[1].name / metrics.parent.name / "influence.npz")
    return metrics, dict(np.load(path, allow_pickle=False))


def draw_effective_motif(parent, out_root: Path):
    metrics, influence = influence_for_selected(out_root)
    graph = dict(np.load(metrics.parent / "graph.npz"))
    fit_id = json.loads(metrics.read_text())["fit_id"]
    plane = dict(np.load(out_root / "cache" / fit_id / "plane.npz"))
    grid = parent.subgridspec(1, 2, width_ratios=[1.05, 0.95], wspace=0.32)
    ax = parent.get_gridspec().figure.add_subplot(grid[0, 0])
    draw_graph(ax, graph, plane, "Effective motif", influence)
    summary = [row for row in rows(out_root / "effective_motif_patient.csv")
               if row["model"] == REPRESENTATIVE_MODEL and row["cell"] == "rnn"]
    right = grid[0, 1].subgridspec(2, 1, hspace=0.52)
    ax = parent.get_gridspec().figure.add_subplot(right[0, 0])
    local = np.asarray([row["local_effective_ratio"] - 1.0 for row in summary], float)
    ax.scatter(np.linspace(-0.09, 0.09, len(local)), local, s=15,
               color=COLORS[REPRESENTATIVE_MODEL], alpha=0.65)
    if len(local): ax.plot([-0.18, 0.18], [np.nanmedian(local)] * 2, color="#111111", lw=1.3)
    ax.axhline(0, color="#8d8d8d", lw=0.7); ax.set_xlim(-0.30, 0.30); ax.set_xticks([])
    ax.set_ylabel("Local influence\nenrichment")

    ax = parent.get_gridspec().figure.add_subplot(right[1, 0])
    for index, key in enumerate(("effective_operator_seed_stability",
                                 "effective_operator_split_half_stability")):
        values = np.asarray([row[key] for row in summary], float)
        values = values[np.isfinite(values)]
        ax.scatter(index + np.linspace(-0.08, 0.08, len(values)), values, s=15,
                   color=("#3b75af", "#55a7a1")[index], alpha=0.65)
        if len(values): ax.plot([index - 0.18, index + 0.18], [np.median(values)] * 2,
                                color="#111111", lw=1.3)
    ax.axhline(0, color="#8d8d8d", lw=0.7)
    ax.set_xticks([0, 1], ["Seeds", "Train halves"])
    ax.set_ylabel("Operator stability (ρ)")


def render_lesion_supplement(out_root: Path):
    frame = [row for row in rows(out_root / "matched_lesion_patient_metrics.csv")
             if row["model"] == REPRESENTATIVE_MODEL and row["cell"] == "rnn"]
    lesions = [("local_backbone_edges", "Local backbone"),
               ("connector_nodes", "Connector nodes")]
    fig, ax = plt.subplots(figsize=(3.9, 3.1), layout="constrained", facecolor="white")
    labels = []
    for index, (lesion, label) in enumerate(lesions):
        values = np.asarray([row["specificity_contact_nll"] for row in frame
                             if row["lesion"] == lesion
                             and row["all_inference_available"] in (True, "True", "1", 1.0)], float)
        values = values[np.isfinite(values)]
        ax.scatter(index + np.linspace(-0.09, 0.09, len(values)), values, s=20,
                   color=COLORS[REPRESENTATIVE_MODEL], alpha=0.65)
        if len(values): ax.plot([index - 0.2, index + 0.2], [np.median(values)] * 2,
                                color="#111111", lw=1.4)
        labels.append(f"{label}\n(n={len(values)})")
    ax.axhline(0, color="#8d8d8d", lw=0.7)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Damage beyond matched\nperturbation (Δ NLL)")
    stem = out_root / "figures" / "topic5_matched_lesion_exploratory"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_label(fig, subplot_spec, label: str):
    box = subplot_spec.get_position(fig)
    fig.text(box.x0 - 0.018, box.y1 + 0.006, label, fontsize=13, fontweight="bold",
             ha="right", va="bottom")


def render_stage(out_root: Path, stage: str):
    size = (11.4, 6.2) if stage == "interictal" else (9.2, 3.2)
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
    fig = plt.figure(figsize=(11.2, 9.1), facecolor="white")
    outer = fig.add_gridspec(3, 2, hspace=0.52, wspace=0.22,
                             left=0.055, right=0.985, top=0.975, bottom=0.115)
    draw_motif_ladder(outer[0, 0], out_root); panel_label(fig, outer[0, 0], "A")
    draw_rollout_example(outer[0, 1], out_root); panel_label(fig, outer[0, 1], "B")
    draw_interictal_sufficiency(outer[1, 0], out_root); panel_label(fig, outer[1, 0], "C")
    ax_d = fig.add_subplot(outer[1, 1]); draw_pareto(ax_d, out_root); panel_label(fig, outer[1, 1], "D")
    draw_cross_state(outer[2, 0], out_root, include_stats=True); panel_label(fig, outer[2, 0], "E")
    draw_effective_motif(outer[2, 1], out_root); panel_label(fig, outer[2, 1], "F")
    handles, labels = ax_d.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(8, len(labels)), frameon=False,
               bbox_to_anchor=(0.5, 0.025), fontsize=7.5, handletextpad=0.4, columnspacing=1.0)
    figure_dir = out_root / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "topic5_figure6_rnn_connectivity_motifs"
    for suffix, dpi in ((".png", 600), (".pdf", None), (".svg", None)):
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if dpi: kwargs["dpi"] = dpi
        fig.savefig(stem.with_suffix(suffix), **kwargs)
    plt.close(fig)
    render_lesion_supplement(out_root)
    readme = figure_dir / "README.md"
    marker = "<!-- topic5-rnn-motif-v0.4-stage-and-final-figures -->"
    existing = readme.read_text() if readme.exists() else "# Topic 5 RNN connectivity-motif figures\n"
    base = existing.split(marker, 1)[0].rstrip()
    section = """<!-- topic5-rnn-motif-v0.4-stage-and-final-figures -->

### topic5_figure6_rnn_connectivity_motifs.png / .pdf / .svg

六联图依次展示真实患者几何上的连接约束、同患者留出间期事件的真实与自由生成 A/B 时序、全队列间期预测充分性、传播场拟合与平均 active-edge 布线代价、冻结模型场与临床发作早期能量场，以及局部有效影响和跨 seed / train-half 的算子稳定性。所有统计先在患者内合并，所有场使用冻结触点顺序和几何；发作数值只在模型与场完全冻结后进入 Panel E。

**关注点**：图的承重顺序是“能生成间期传播 → 哪些结构更经济 → 哪些冻结场跨状态对应 → 哪些局部有效组织可重复”，不把预测性能直接写成真实连接组恢复。

### topic5_matched_lesion_exploratory.png / .pdf

Spatial + cost 模型中 local-backbone 与 connector 的 targeted perturbation 相对匹配随机扰动。横轴直接标出实际可匹配患者数；由于只有 n=5 和 n=7，该图只作探索性旁证，不承担 cohort 阴性或阳性结论。

**关注点**：当前 matched-lesion 状态是 `INCONCLUSIVE_DUE_TO_MATCHING_ELIGIBILITY`，不是“lesion 失败”。

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

代表患者的局部高影响组织，以及其患者级局部富集、跨 seed 和 train-half 稳定性。

**关注点**：当前只支持稳定的局部有效组织；长程 connector 与特异扰动仍未建立。
"""
    readme.write_text(base + "\n\n" + section.lstrip())
    sources = {
        "A": [selected_metrics(out_root, REPRESENTATIVE, model).parent / "graph.npz"
              for model in ("M1_DENSE", "M2_UNIFORM_SET", "M3_FIXED_LOCAL", "M6_SPATIAL_MID")],
        "B": [selected_metrics(out_root, REPRESENTATIVE, REPRESENTATIVE_MODEL).parent
              / "heldout_rollouts.json.gz", out_root / "cache" / "epilepsiae_1146__shared" / "events.npz"],
        "C": [out_root / "interictal_per_patient.csv", out_root / "interictal_per_event.csv"],
        "D": [out_root / "accuracy_wiring_pareto.csv", out_root / "model_field_patient_metrics.csv"],
        "E": [out_root / "early_ictal_per_patient_model.csv", out_root / "early_ictal_null_matrices.npz"],
        "F": [out_root / "effective_influence_fit_seed.csv", out_root / "effective_motif_patient.csv"],
    }
    manifest = {
        panel: [{"path": str(path), "sha256": sha256(path)} for path in paths]
        for panel, paths in sources.items()
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
