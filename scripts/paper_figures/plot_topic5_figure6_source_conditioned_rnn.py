#!/usr/bin/env python3
"""Figure 6: source-conditioned structured RNN, interictal to early ictal.

Panel geometry and colour follow the established paper-ready idioms rather
than being reinvented here:

* b reuses the Fig-1 rank-heatmap idiom (contacts x events, viridis First to
  Last, grey for a contact that did not participate);
* d reuses the frozen shared contact plane and the Fig-3b field renderer, so
  the model fields and the early-ictal field sit on the same real geometry.

Nothing is recomputed from the ictal target and the representative patient
was fixed before any target value was unsealed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_shared_scaffold_rnn_unit_v0_2 import (  # noqa: E402
    load_one_patient_record,
)
from scripts.freeze_topic5_shared_scaffold_rollout_subject_v0_2 import (  # noqa: E402
    load_model,
)
from src.topic5_patient_specific_rnn_bridge import (  # noqa: E402
    chronological_60_20_20,
)
from src.topic5_shared_scaffold_rollout import (  # noqa: E402
    exact_conditional_k_subset_sample,
)

REPRESENTATIVE = "epilepsiae_1146"
MODEL_ORDER = ("static", "ordinary_gru", "structured", "structured_rank_shuffle")
MODEL_LABEL = {
    "static": "Static",
    "ordinary_gru": "Dense GRU",
    "structured": "Structured",
    "structured_rank_shuffle": "Shuffled order",
}
MODEL_COLOR = {
    "static": "#9E9E9E",
    "ordinary_gru": "#4D908E",
    "structured": "#B2182B",
    "structured_rank_shuffle": "#E8B4B8",
}
# The patient's own two interictal propagation modes, read back from the
# frozen clustering.  The model never saw these labels.
MODE_KEYS = ("A", "B")
MODE_COLOR = {"A": "#B2182B", "B": "#2166AC"}
MODE_LABEL = {"A": "Mode A", "B": "Mode B"}
PANEL_C_MODELS = ("static", "structured", "structured_rank_shuffle")
PANEL_E_MODELS = ("static", "structured")
N_DISPLAY_EVENTS = 220
PROPAGATION_RECORD = Path(
    "/home/honglab/leijiaxin/HFOsp/results/interictal_propagation_masked/per_subject"
)

plt.rcParams.update(
    {
        "font.size": 6.5,
        "axes.labelsize": 6.5,
        "axes.titlesize": 7,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 1.8,
        "ytick.major.size": 1.8,
        "savefig.bbox": "tight",
    }
)


# ------------------------------------------------------------------ panel a
def _style(ax):
    """Repo convention: keep only the left and bottom spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_a(ax):
    """Contact sequence in, recurrent network, contact sequence out."""

    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cmap = plt.get_cmap("viridis")
    n_contacts = 8
    strip_w, strip_h = 0.085, 0.058

    def strip(x0, y0, values, label):
        for index, value in enumerate(values):
            y = y0 - index * strip_h
            face = "#DDDDDD" if value is None else cmap(value)
            ax.add_patch(
                plt.Rectangle((x0, y), strip_w, strip_h * 0.86, facecolor=face,
                              edgecolor="white", linewidth=0.5, zorder=3)
            )
        ax.text(x0 + strip_w / 2, y0 + strip_h * 1.05, label, ha="center",
                va="bottom", fontsize=6.2)
        return [(x0, y0 - i * strip_h + strip_h * 0.43) for i in range(len(values))]

    seen_now = [0.0, 0.12, None, 0.30, None, None, None, None]
    predicted = [0.0, 0.12, 0.55, 0.30, 0.72, 0.88, None, None]
    left = strip(0.045, 0.735, seen_now, "observed\nso far")
    right = strip(0.870, 0.735, predicted, "next\ncontacts")

    # recurrent network inside one boundary
    cx, cy, radius = 0.475, 0.50, 0.235
    ax.add_patch(
        plt.Circle((cx, cy), radius, facecolor="#F7F7F7", edgecolor="#333333",
                   linewidth=0.9, zorder=1)
    )
    n_units = 9
    angles = np.linspace(0.5 * np.pi, 2.5 * np.pi, n_units, endpoint=False)
    ux = cx + 0.155 * np.cos(angles)
    uy = cy + 0.155 * np.sin(angles)
    axis_value = np.linspace(0.0, 1.0, n_units)
    for i in range(n_units):
        for j in range(i + 1, n_units):
            gap = min(abs(i - j), n_units - abs(i - j))
            weight = float(np.exp(-(gap / 2.4) ** 2))
            if weight < 0.15:
                continue
            ax.plot([ux[i], ux[j]], [uy[i], uy[j]], color="#B8B8B8",
                    linewidth=0.25 + 1.1 * weight, alpha=0.75, zorder=2)
    for i, j in ((0, n_units // 2), (1, n_units // 2 + 1)):
        ax.add_patch(
            FancyArrowPatch((ux[i], uy[i]), (ux[j], uy[j]), arrowstyle="-|>",
                            mutation_scale=6, connectionstyle="arc3,rad=0.30",
                            linewidth=1.0, color=MODE_COLOR["A"], zorder=4)
        )
    ax.scatter(ux, uy, s=46, marker="o", c=axis_value, cmap="coolwarm",
               edgecolors="#333333", linewidths=0.6, zorder=5)
    ax.text(cx, cy + radius + 0.020, "recurrent contact network", ha="center",
            va="bottom", fontsize=6.5)

    for y in (0.62, 0.50, 0.38):
        ax.add_patch(
            FancyArrowPatch((0.145, y), (cx - radius - 0.012, y), arrowstyle="-|>",
                            mutation_scale=6, linewidth=0.6, color="#555555", zorder=6)
        )
        ax.add_patch(
            FancyArrowPatch((cx + radius + 0.012, y), (0.860, y), arrowstyle="-|>",
                            mutation_scale=6, linewidth=0.6, color="#555555", zorder=6)
        )


# ------------------------------------------------------------------ panel b
def _rank_image(groups: np.ndarray, order: np.ndarray, n_events: int, seed: int):
    """Contacts x events image of within-event rank, grey where absent."""

    if len(groups) > n_events:
        rng = np.random.default_rng(seed)
        groups = groups[np.sort(rng.choice(len(groups), size=n_events, replace=False))]
    image = np.asarray(groups, dtype=float)[:, order].T
    return np.where(image < 0, np.nan, image)


def _rank_profile(groups: np.ndarray, order: np.ndarray):
    """Median and interquartile spread of each contact's within-event rank."""

    values = np.where(groups < 0, np.nan, groups).astype(float)[:, order]
    with np.errstate(invalid="ignore"):
        low, mid, high = np.nanpercentile(values, [25, 50, 75], axis=0)
    return mid, low, high


def panel_b(axes, profile_axes, observed, rollout, order, cbar_ax):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#DDDDDD")
    vmax = max(
        np.nanmax(image)
        for key in MODE_KEYS
        for image in (observed[key]["image"], rollout[key]["image"])
    )
    handle = None
    for row, key in enumerate(MODE_KEYS):
        for column, (source, title) in enumerate(
            ((observed, "Observed held-out events"), (rollout, "Model, same starts"))
        ):
            ax = axes[row][column]
            handle = ax.imshow(source[key]["image"], aspect="auto", cmap=cmap,
                               vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            if column == 0:
                ax.set_yticks(range(len(order)))
                ax.set_yticklabels(source[key]["names"], fontsize=4.4)
                ax.set_ylabel(MODE_LABEL[key], color=MODE_COLOR[key], fontsize=7)
            else:
                ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=7)
        ax = profile_axes[row]
        rows = np.arange(len(order))
        for source, style, width in ((observed, "-", 1.0), (rollout, "--", 1.0)):
            mid, low, high = source[key]["profile"]
            ax.fill_betweenx(rows, low, high, color=MODE_COLOR[key], alpha=0.13,
                             linewidth=0)
            ax.plot(mid, rows, style, color=MODE_COLOR[key], linewidth=width)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_yticks([])
        ax.tick_params(labelsize=5)
        _style(ax)
        ax.spines["left"].set_visible(False)
    profile_axes[-1].set_xlabel("rank", fontsize=6.5)
    axes[1][0].set_xlabel("Events", fontsize=6.5)
    axes[1][1].set_xlabel("Events", fontsize=6.5)
    bar = plt.colorbar(handle, cax=cbar_ax)
    bar.set_label("First $\\rightarrow$ Last", fontsize=6)
    bar.ax.tick_params(labelsize=5)


# ------------------------------------------------------------------ panel c/e
def _bracket(ax, left, right, level, text):
    """Significance bracket drawn in axes coordinates above the data."""

    ax.plot([left, left, right, right],
            [level, level + 0.022, level + 0.022, level],
            transform=ax.get_xaxis_transform(), color="#333333",
            linewidth=0.5, clip_on=False)
    ax.text((left + right) / 2, level + 0.030, text,
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=5.6, color="#333333", clip_on=False)


def _paired(ax, wide, models, ylabel, seed):
    positions = np.arange(len(models), dtype=float)
    rng = np.random.default_rng(seed)
    for _, row in wide.iterrows():
        ax.plot(positions, [row[m] for m in models], color="#CCCCCC",
                linewidth=0.3, alpha=0.7, zorder=1)
    for index, model in enumerate(models):
        values = wide[model].to_numpy(float)
        jitter = rng.uniform(-0.07, 0.07, size=len(values))
        ax.scatter(positions[index] + jitter, values, s=5,
                   color=MODEL_COLOR[model], edgecolor="none", zorder=3)
        ax.hlines(np.median(values), positions[index] - 0.22, positions[index] + 0.22,
                  color="#111111", linewidth=1.0, zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_LABEL[m] for m in models], fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6.5)
    ax.set_xlim(-0.5, len(models) - 0.5)
    _style(ax)


def panel_c(ax, patient: pd.DataFrame, stats: dict):
    models = [m for m in PANEL_C_MODELS if m in set(patient.model)]
    wide = patient.pivot(index="subject", columns="model", values="contact_nll").dropna(
        subset=models
    )
    _paired(ax, wide, models, "Next-contact NLL", 4)
    ax.set_title(f"Interictal prediction, {len(wide)} patients", loc="left",
                 fontweight="bold", fontsize=7.5, pad=22)
    index = {model: position for position, model in enumerate(models)}
    for comparator, level in (("static", 1.01), ("structured_rank_shuffle", 1.13)):
        entry = stats["comparisons"].get(f"structured_vs_{comparator}__contact_nll")
        if entry and entry.get("status") == "COMPLETE" and comparator in index:
            _bracket(ax, index["structured"], index[comparator], level,
                     f"P={entry['wilcoxon_two_sided_p']:.1e}")


def panel_e(ax, patient: pd.DataFrame, cohort: dict, supportive: str):
    models = [m for m in PANEL_E_MODELS if m in set(patient.model)]
    wide = patient.pivot(index="subject", columns="model", values="all_contact_margin")
    wide = wide.dropna(subset=models)
    primary = wide.drop(index=supportive, errors="ignore")
    _paired(ax, primary, models, "Above null", 5)
    if supportive in wide.index:
        ax.scatter(np.arange(len(models)), [wide.loc[supportive, m] for m in models],
                   s=12, facecolor="none", edgecolor="#B2182B", linewidth=0.7, zorder=5)
    ax.axhline(0.0, color="#777777", linewidth=0.5, linestyle=":")
    n_primary = cohort["model_statistics"]["structured"]["n_primary_patients"]
    ax.set_title(f"Cross-state, {n_primary} patients", loc="left",
                 fontweight="bold", fontsize=7.5, pad=12)
    index = {model: position for position, model in enumerate(models)}
    paired = cohort["paired_comparisons"].get("structured_vs_static_all_contact")
    if paired and "static" in index and "structured" in index:
        _bracket(ax, index["structured"], index["static"], 1.02,
                 f"P={paired['exact_wilcoxon_greater_p']:.3g}")


# ------------------------------------------------------------------ panel d
def panel_d(axes, cbar_axes, frozen_plane, fields, event_field):
    """Model fields and the early-ictal field on the frozen contact plane."""

    points = np.asarray(frozen_plane["points_mm"], dtype=float)
    support = np.asarray(frozen_plane["support_a"], dtype=float)
    handles = []
    for ax, (title, values, cmap, colour) in zip(axes, fields):
        display = (values - np.nanmin(values)) / max(
            np.nanmax(values) - np.nanmin(values), 1e-12
        )
        x_grid, y_grid, field, _, _ = event_field(frozen_plane, display, support)
        image = ax.imshow(
            field, origin="lower",
            extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
            aspect="equal", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="bilinear",
        )
        handles.append(image)
        ax.scatter(points[:, 0], points[:, 1], c=display, cmap=cmap, vmin=0.0, vmax=1.0,
                   s=11, edgecolors="white", linewidths=0.45, zorder=3)
        ax.set_xlim(float(x_grid.min()), float(x_grid.max()))
        ax.set_ylim(float(y_grid.min()), float(y_grid.max()))
        ax.set_title(title, fontsize=6.8, color=colour, fontweight="bold", pad=3)
        ax.set_xlabel("shared axis (mm)", fontsize=6)
        ax.tick_params(labelsize=5)
    axes[0].set_ylabel("transverse (mm)", fontsize=6)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)
    first = plt.colorbar(handles[0], cax=cbar_axes[0])
    first.set_label("model arrival\nlate $\\rightarrow$ early", fontsize=5.6)
    first.set_ticks([])
    second = plt.colorbar(handles[2], cax=cbar_axes[1])
    second.set_label("ictal power\nlow $\\rightarrow$ high", fontsize=5.6)
    second.set_ticks([])


# --------------------------------------------------------------------- data
def load_mode_labels(dataset_root: Path, record) -> np.ndarray:
    """Join the frozen A/B cluster labels onto this dataset's events.

    The clustering ran on the patient's full event set; the RNN dataset keeps
    a chosen subset of blocks.  The two are joined through block identity and
    the join is verified on event count and on every block's own time window,
    because aligning these two arrays by position would silently mislabel
    every event.
    """

    with np.load(dataset_root / "per_subject" / f"{REPRESENTATIVE}.npz",
                 allow_pickle=False) as data:
        selected = np.asarray(data["selected_block_ids"], dtype=int)
        abs_time = np.asarray(data["event_abs_time"], dtype=float)
    frozen = json.loads((PROPAGATION_RECORD / f"{REPRESENTATIVE}.json").read_text())
    boundaries = {
        int(block["block_id"]): block
        for block in frozen["event_metadata"]["block_boundaries"]
    }
    labels = np.asarray(frozen["adaptive_cluster"]["labels"], dtype=int)
    joined, offset = [], 0
    for block_id in selected:
        block = boundaries[int(block_id)]
        count = int(block["n_events"])
        window = abs_time[offset:offset + count]
        if window.min() < float(block["block_start_epoch"]) - 1.0 or (
            window.max() > float(block["block_end_epoch"]) + 1.0
        ):
            raise RuntimeError(f"block {block_id}: dataset events fall outside it")
        joined.append(labels[int(block["start_event_idx"]):int(block["end_event_idx"])])
        offset += count
    joined = np.concatenate(joined)
    if joined.size != abs_time.size:
        raise RuntimeError("mode labels do not align with the dataset events")
    return joined


@torch.no_grad()
def event_matched_rollouts(model, first_sets, *, horizon, seed, batch_size=512):
    """Roll the model out from each observed event's own first rank set.

    Display only.  Seeding every rollout from the event it is compared with
    is what makes the two columns of panel b comparable; a fixed source pool
    would force a constant first rank set the observed events do not have.
    """

    device = model.participation_bias.device
    n_contacts = int(model.n_contacts)
    sources = torch.as_tensor(np.asarray(first_sets), device=device, dtype=torch.bool)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    model.eval()
    produced = []
    for start in range(0, int(sources.shape[0]), int(batch_size)):
        current = sources[start:start + int(batch_size)].clone()
        batch = int(current.shape[0])
        seen = current.clone()
        state = model.reset_state(batch_size=batch)
        state = model.observe(
            state, current,
            active=torch.ones(batch, dtype=torch.bool, device=device),
        )
        groups = torch.full((batch, n_contacts), -1, dtype=torch.int16, device=device)
        groups[current] = 0
        alive = torch.ones(batch, dtype=torch.bool, device=device)
        for step in range(1, int(horizon) + 1):
            decision = model.decision(state, seen)
            stop_probability = torch.sigmoid(decision["stop_logit"])
            draw = torch.rand(batch, device=device, dtype=stop_probability.dtype,
                              generator=generator)
            stop = alive & ((draw < stop_probability) | ~decision["eligible"].any(dim=1))
            continuing = alive & ~stop
            next_set = torch.zeros_like(seen)
            if torch.any(continuing):
                rows = torch.where(continuing)[0]
                probability = torch.softmax(decision["cardinality_logits"][rows], dim=1)
                cardinality = torch.multinomial(
                    probability, 1, generator=generator
                ).squeeze(1) + 1
                next_set[rows] = exact_conditional_k_subset_sample(
                    node_logits=decision["node_logits"][rows],
                    eligible=decision["eligible"][rows],
                    cardinality=cardinality,
                    generator=generator,
                )
                groups[next_set] = int(step)
            state = model.observe(state, next_set, active=continuing)
            seen = seen | next_set
            alive = continuing
            if not torch.any(alive):
                break
        produced.append(groups.cpu().numpy().astype(np.int64))
    return np.concatenate(produced, axis=0)


def build_panel_b_inputs(output: Path, dataset_root: Path, order_by: np.ndarray):
    record = load_one_patient_record(dataset_root, REPRESENTATIVE)
    _, _, test20 = chronological_60_20_20(record)
    modes = load_mode_labels(dataset_root, record)[np.asarray(test20)]
    groups = np.asarray(record.group_ids, dtype=np.int64)[np.asarray(test20)]
    freeze = output / "field_freeze" / "per_subject" / REPRESENTATIVE
    with np.load(freeze / "structured_fields.npz", allow_pickle=False) as data:
        names = np.asarray(data["contact_names"]).astype(str)
        horizon = int(data["horizon"])
    order = np.argsort(order_by)

    checkpoint_path = (
        output / "per_subject" / REPRESENTATIVE / "structured" / "seed_11" / "checkpoint.pt"
    )
    _, model = load_model(checkpoint_path, device=torch.device("cpu"))

    observed, rollout = {}, {}
    for key, label in zip(MODE_KEYS, (0, 1)):
        picked = groups[modes == label]
        generated = event_matched_rollouts(
            model, picked == 0, horizon=horizon, seed=90_001 + label
        )
        for store, source in ((observed, picked), (rollout, generated)):
            store[key] = {
                "image": _rank_image(source, order, N_DISPLAY_EVENTS, 11 + label),
                "profile": _rank_profile(source, order),
                "n_events": int(len(source)),
                "names": names[order],
            }
    return observed, rollout, order, names


def median_early_ictal_field(readout: dict, names: np.ndarray):
    target_root = Path(readout["target_cache_root"]).resolve()
    files = sorted((target_root / f"outer_{REPRESENTATIVE}").glob(f"{REPRESENTATIVE}__*.npz"))
    stacked = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            target_names = np.asarray(data["contact_names"]).astype(str)
            values = np.asarray(data[str(readout["target_key"])], dtype=float)
        lookup = dict(zip(target_names, values))
        stacked.append([lookup.get(name, np.nan) for name in names])
    return np.nanmedian(np.asarray(stacked, dtype=float), axis=0)


def directional_field_opposition(output: Path):
    from scipy.stats import spearmanr

    rows = []
    for path in sorted((output / "field_freeze" / "per_subject").glob("*/structured_fields.npz")):
        with np.load(path, allow_pickle=False) as data:
            minus = np.asarray(data["field_minus"], dtype=float)
            plus = np.asarray(data["field_plus"], dtype=float)
        rows.append({"subject": path.parent.name,
                     "rho_minus_plus": float(spearmanr(minus, plus).statistic)})
    values = np.asarray([row["rho_minus_plus"] for row in rows], dtype=float)
    return rows, {
        "n_patients": int(len(rows)),
        "median_rho": float(np.median(values)),
        "n_opposite_below_minus_0p5": int(np.count_nonzero(values < -0.5)),
    }


# --------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--readout-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    from scripts.plot_topic5_interictal_event_envelope_field import (
        _event_field,
        load_frozen,
    )

    training = yaml.safe_load((ROOT / args.training_config).read_text())
    readout = yaml.safe_load((ROOT / args.readout_config).read_text())
    output = (ROOT / args.output_root).resolve()
    dataset_root = Path(training["dataset_artifact_root"]).resolve() / training["dataset_root"]

    patient_interictal = pd.read_csv(output / "interictal_patient_metrics.csv")
    interictal_stats = json.loads((output / "interictal_cohort_statistics.json").read_text())
    patient_ictal = pd.read_csv(output / "early_ictal" / "patient_scores.csv")
    ictal_stats = json.loads((output / "early_ictal" / "cohort_statistics.json").read_text())

    freeze = output / "field_freeze" / "per_subject" / REPRESENTATIVE
    with np.load(freeze / "structured_fields.npz", allow_pickle=False) as data:
        coordinate = np.asarray(data["diffusion_coordinate"], dtype=float)
        field_minus = np.asarray(data["field_minus"], dtype=float)
        field_plus = np.asarray(data["field_plus"], dtype=float)

    observed, rollout, order, names = build_panel_b_inputs(output, dataset_root, coordinate)
    frozen_plane = load_frozen(REPRESENTATIVE)
    if list(map(str, frozen_plane["names"])) != list(names):
        raise RuntimeError("frozen plane and model contact order differ")
    ictal_field = median_early_ictal_field(readout, names)
    opposition = directional_field_opposition(output)

    figure = plt.figure(figsize=(7.09, 9.3))

    # a is square: 2.4 in on both sides of a 7.09 x 9.3 in canvas
    a_ax = figure.add_axes([0.331, 0.700, 0.338, 0.258])
    panel_a(a_ax)

    b_left, b_wide, b_gap = 0.105, 0.300, 0.017
    b_rows = ((0.505, 0.150), (0.330, 0.150))
    b_axes, b_profiles = [], []
    for bottom, height in b_rows:
        b_axes.append([
            figure.add_axes([b_left, bottom, b_wide, height]),
            figure.add_axes([b_left + b_wide + b_gap, bottom, b_wide, height]),
        ])
        b_profiles.append(
            figure.add_axes([b_left + 2 * (b_wide + b_gap), bottom, 0.098, height])
        )
    b_cbar = figure.add_axes([0.930, 0.375, 0.011, 0.150])
    panel_b(b_axes, b_profiles, observed, rollout, order, b_cbar)
    b_axes[0][0].annotate(
        f"{REPRESENTATIVE.replace('epilepsiae_', 'E')}   the patient's two interictal propagation modes",
        xy=(0.0, 1.26), xycoords="axes fraction", fontsize=7.5, fontweight="bold",
        annotation_clip=False,
    )

    d_axes = [figure.add_axes([0.105 + i * 0.205, 0.175, 0.175, 0.100]) for i in range(3)]
    d_cbars = [figure.add_axes([0.735, 0.175, 0.010, 0.100]),
               figure.add_axes([0.805, 0.175, 0.010, 0.100])]
    panel_d(
        d_axes, d_cbars, frozen_plane,
        [
            ("Model field, start 1", field_minus, "viridis_r", MODE_COLOR["A"]),
            ("Model field, start 2", field_plus, "viridis_r", MODE_COLOR["B"]),
            ("Early-ictal power", ictal_field, "Blues", "#111111"),
        ],
        _event_field,
    )
    d_axes[0].annotate("Frozen model fields and the early-ictal field",
                       xy=(0.0, 1.13), xycoords="axes fraction", fontsize=7.5,
                       fontweight="bold", annotation_clip=False)
    figure.text(
        0.855, 0.245,
        "the two model fields\nare not opposites:\n"
        f"$\\rho$={opposition[1]['median_rho']:+.2f} median,\n"
        f"below $-0.5$ in "
        f"{opposition[1]['n_opposite_below_minus_0p5']}/{opposition[1]['n_patients']}",
        ha="left", va="top", fontsize=5.5, color="#B2182B",
    )

    c_ax = figure.add_axes([0.105, 0.030, 0.300, 0.068])
    panel_c(c_ax, patient_interictal, interictal_stats)
    e_ax = figure.add_axes([0.655, 0.030, 0.195, 0.068])
    panel_e(e_ax, patient_ictal, ictal_stats, str(readout["supportive_subject"]))

    for label, axis, offset in (
        ("a", a_ax, (-0.120, 1.00)), ("b", b_axes[0][0], (-0.235, 1.26)),
        ("c", c_ax, (-0.210, 1.42)), ("d", d_axes[0], (-0.290, 1.16)),
        ("e", e_ax, (-0.300, 1.42)),
    ):
        axis.annotate(label, xy=offset, xycoords="axes fraction", fontsize=9,
                      fontweight="bold", annotation_clip=False)

    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = figures_dir / "topic5_figure6_source_conditioned_rnn"
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".svg"))
    figure.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(figure)

    patient_interictal.to_csv(figures_dir / "figure6_panelC_source_data.csv", index=False)
    patient_ictal.to_csv(figures_dir / "figure6_panelE_source_data.csv", index=False)
    pd.DataFrame(
        {
            "contact": names,
            "learned_axis_coordinate": coordinate,
            "model_field_start1": field_minus,
            "model_field_start2": field_plus,
            "early_ictal_median": ictal_field,
        }
    ).to_csv(figures_dir / "figure6_panelD_source_data.csv", index=False)
    pd.DataFrame(
        [
            {"mode": MODE_LABEL[key], "source": source,
             "n_events": payload[key]["n_events"],
             **{f"median_rank__{name}": value
                for name, value in zip(payload[key]["names"], payload[key]["profile"][0])}}
            for key in MODE_KEYS
            for source, payload in (("observed", observed), ("model_rollout", rollout))
        ]
    ).to_csv(figures_dir / "figure6_panelB_source_data.csv", index=False)

    statistics = {
        "representative_subject": REPRESENTATIVE,
        "representative_fixed_before_target_unseal": True,
        "panel_b_event_counts": {
            MODE_LABEL[key]: {"observed": observed[key]["n_events"],
                              "model_rollout": rollout[key]["n_events"]}
            for key in MODE_KEYS
        },
        "panel_b_mode_labels": (
            "read back from the frozen interictal clustering; the model never "
            "saw them, and model rollouts are seeded from each observed event's "
            "own first rank set"
        ),
        "panel_c": interictal_stats,
        "panel_d_directional_field_opposition": {
            "summary": opposition[1], "per_patient": opposition[0],
            "note": "measured on the emitted fields, not assumed from the direction state",
        },
        "panel_e": ictal_stats,
        "not_implemented": [
            "rollout-vs-test20 participation / pairwise precedence / expected-rank "
            "distance consistency statistics are not computed by any current script"
        ],
    }
    (figures_dir / "figure6_statistics.json").write_text(
        json.dumps(statistics, indent=2, allow_nan=False, default=float) + "\n"
    )

    counts = statistics["panel_b_event_counts"]
    opposition_summary = opposition[1]
    (figures_dir / "README.md").write_text(
        "# Figure 6 图说明\n\n"
        "### topic5_figure6_source_conditioned_rnn.png / .pdf / .svg\n\n"
        "a 模型结构。左边一列是到这一步为止已经观察到的触点，颜色就是下面各图同一套"
        "先后次序色标（深紫最早、黄最晚），灰色表示还没出现；中间圆圈是触点之间的循环网络，"
        "灰线是所有触点共用的对称连接、红线是唯一那条有方向的连接；右边一列是模型对下一步"
        "各触点的预测。\n\n"
        f"b 固定用 {REPRESENTATIVE.replace('epilepsiae_', 'E')}。上下两行是**这位患者自己的"
        "两种间期传播模式**，模式标签来自已冻结的间期聚类，模型训练时从未见过它们。"
        "每一列是一场事件，颜色是该触点在这场事件里第几个放电，灰色表示没参与。"
        "左列是留出集里真实观察到的事件；右列是模型推演，且**每一场推演都从对应那场观察事件"
        "自己的第一批触点出发**，所以两列起点一致、可以逐行对照走向。"
        "最右是各触点先后次序的中位数与四分位区间（实线观察、虚线模型）。\n\n"
        "d 两张冻结的模型场与同一患者发作早期宽带能量场，画在同一套真实电极几何上"
        "（沿用既有间期-发作共享场图的平面与插值），圆点为真实触点位置。\n\n"
        "c 与 e 是队列统计，横线为中位数，括号内为配对检验的 P 值。\n\n"
        "**读图注意**：\n\n"
        "1. c 中结构化模型同时优于静态基线与打乱顺序对照（两个 P 值均在图上），"
        "说明它确实在用事件内的先后顺序。\n"
        "2. e 的 P 值是「结构化优于静态」的单侧检验，接近 1 表示**方向相反**——"
        "在跨状态这一步是静态基线更好，不要读成不显著。\n"
        f"3. d 中两张模型场实测高度相似而非相反（全队列秩相关中位 "
        f"{opposition_summary['median_rho']:+.2f}，真正相反仅 "
        f"{opposition_summary['n_opposite_below_minus_0p5']}/"
        f"{opposition_summary['n_patients']} 人）。\n\n"
        "**关注点**：b 右列与左列的走向是否一致、实线与虚线是否贴合；"
        "b 上下两行是否确实呈现两种不同的次序；"
        "c 三者的高低与两个 P 值；e 中静态基线高于结构化模型这一事实。\n\n"
        f"事件数：模式 A 观察 {counts['Mode A']['observed']:,} / 模型 "
        f"{counts['Mode A']['model_rollout']:,}；模式 B 观察 "
        f"{counts['Mode B']['observed']:,} / 模型 {counts['Mode B']['model_rollout']:,}。\n"
    )

    print(json.dumps({"status": "COMPLETE", "figure": str(stem.with_suffix(".png"))}))


if __name__ == "__main__":
    main()
