#!/usr/bin/env python3
"""Figure 6 panels: source-conditioned structured RNN, interictal to early ictal.

Each panel is written as its own file, which is this repository's convention
for paper-ready output (see fig1-panela1 / fig1-panelc and the paper-ready
README).  Typography follows the accepted panels: bold panel title 11-13,
axis labels 9.5-11, tick labels 8-8.5, colourbar label 10.5 with 8.5 ticks,
top and right spines hidden, PNG at 300 dpi beside a PDF.

Reused rather than reinvented:

* panel b keeps the Fig-1 rank-heatmap grammar (contacts x events, viridis
  First to Last, grey where a contact did not participate);
* panel d uses the frozen shared contact plane and the Fig-3b field renderer;
* panels c and e call the accepted Fig-3 paired violin/box/points painter,
  whose bracket already reports significance as asterisks.

No ictal value is recomputed here and the representative patient was fixed
before any target value was unsealed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_topic5_shared_scaffold_rollout_subject_v0_2 import (  # noqa: E402
    load_model,
)
from scripts.paper_figures.plot_fig3_field_concordance_cohort_stat import (  # noqa: E402
    plot_paired_data_null_groups,
)
from scripts.run_topic5_shared_scaffold_rnn_unit_v0_2 import (  # noqa: E402
    load_one_patient_record,
)
from src.topic5_patient_specific_rnn_bridge import (  # noqa: E402
    chronological_60_20_20,
)
from src.topic5_shared_scaffold_rollout import (  # noqa: E402
    exact_conditional_k_subset_sample,
)

REPRESENTATIVE = "epilepsiae_1146"
MODE_KEYS = ("A", "B")
MODE_COLOR = {"A": "#B2182B", "B": "#2166AC"}
MODE_LABEL = {"A": "Mode A", "B": "Mode B"}
N_DISPLAY_EVENTS = 220
PROPAGATION_RECORD = Path(
    "/home/honglab/leijiaxin/HFOsp/results/interictal_propagation_masked/per_subject"
)

TITLE_SIZE = 12.0
LABEL_SIZE = 10.0
TICK_SIZE = 8.5
CBAR_LABEL_SIZE = 10.5
CBAR_TICK_SIZE = 8.5


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white",
                bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, length=2.5)


# ------------------------------------------------------------------ panel a
def panel_a(stem: Path) -> None:
    """Contact sequence in, one recurrent network, contact sequence out."""

    fig, ax = plt.subplots(figsize=(3.9, 3.9), facecolor="white")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cmap = plt.get_cmap("viridis")
    strip_w, strip_h = 0.088, 0.062

    def strip(x0, y0, values, label):
        for index, value in enumerate(values):
            y = y0 - index * strip_h
            ax.add_patch(
                plt.Rectangle((x0, y), strip_w, strip_h * 0.86,
                              facecolor="#DDDDDD" if value is None else cmap(value),
                              edgecolor="white", linewidth=0.6, zorder=3)
            )
        ax.text(x0 + strip_w / 2, y0 + strip_h * 1.15, label, ha="center",
                va="bottom", fontsize=9.0)

    observed_so_far = [0.0, 0.12, None, 0.30, None, None, None, None]
    predicted_next = [0.0, 0.12, 0.55, 0.30, 0.72, 0.88, None, None]
    strip(0.030, 0.735, observed_so_far, "observed\nso far")
    strip(0.882, 0.735, predicted_next, "next\ncontacts")

    cx, cy, radius = 0.478, 0.485, 0.242
    ax.add_patch(plt.Circle((cx, cy), radius, facecolor="#F7F7F7",
                            edgecolor="#333333", linewidth=1.1, zorder=1))
    n_units = 9
    angles = np.linspace(0.5 * np.pi, 2.5 * np.pi, n_units, endpoint=False)
    ux, uy = cx + 0.158 * np.cos(angles), cy + 0.158 * np.sin(angles)
    for i in range(n_units):
        for j in range(i + 1, n_units):
            gap = min(abs(i - j), n_units - abs(i - j))
            weight = float(np.exp(-(gap / 2.4) ** 2))
            if weight >= 0.15:
                ax.plot([ux[i], ux[j]], [uy[i], uy[j]], color="#B8B8B8",
                        linewidth=0.3 + 1.3 * weight, alpha=0.75, zorder=2)
    for i, j in ((0, n_units // 2), (1, n_units // 2 + 1)):
        ax.add_patch(FancyArrowPatch((ux[i], uy[i]), (ux[j], uy[j]),
                                     arrowstyle="-|>", mutation_scale=8,
                                     connectionstyle="arc3,rad=0.30",
                                     linewidth=1.2, color=MODE_COLOR["A"], zorder=4))
    ax.scatter(ux, uy, s=62, marker="o", c=np.linspace(0, 1, n_units),
               cmap="coolwarm", edgecolors="#333333", linewidths=0.7, zorder=5)
    ax.text(cx, cy + radius + 0.022, "recurrent contact network", ha="center",
            va="bottom", fontsize=9.5)
    for y in (0.60, 0.485, 0.37):
        ax.add_patch(FancyArrowPatch((0.132, y), (cx - radius - 0.014, y),
                                     arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.8, color="#555555", zorder=6))
        ax.add_patch(FancyArrowPatch((cx + radius + 0.014, y), (0.872, y),
                                     arrowstyle="-|>", mutation_scale=8,
                                     linewidth=0.8, color="#555555", zorder=6))
    _save(fig, stem)


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


# ------------------------------------------------------------------ panel b
def panel_b(stem: Path, observed, rollout, order) -> None:
    fig = plt.figure(figsize=(9.6, 5.4), facecolor="white")
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.34],
                            left=0.085, right=0.905, top=0.90, bottom=0.10,
                            hspace=0.16, wspace=0.10)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#DDDDDD")
    vmax = max(np.nanmax(source[key]["image"])
               for key in MODE_KEYS for source in (observed, rollout))
    handle = None
    for row, key in enumerate(MODE_KEYS):
        for column, (source, title) in enumerate(
            ((observed, "Observed held-out events"), (rollout, "Model, same starts"))
        ):
            ax = fig.add_subplot(grid[row, column])
            handle = ax.imshow(source[key]["image"], aspect="auto", cmap=cmap,
                               vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            if column == 0:
                ax.set_yticks(range(len(order)))
                ax.set_yticklabels(source[key]["names"], fontsize=7.0)
                ax.set_ylabel(MODE_LABEL[key], color=MODE_COLOR[key],
                              fontsize=LABEL_SIZE, fontweight="bold")
            else:
                ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=TITLE_SIZE, pad=6)
            if row == 1:
                ax.set_xlabel("Population events", fontsize=LABEL_SIZE)
        ax = fig.add_subplot(grid[row, 2])
        rows = np.arange(len(order))
        for source, style in ((observed, "-"), (rollout, "--")):
            mid, low, high = source[key]["profile"]
            ax.fill_betweenx(rows, low, high, color=MODE_COLOR[key], alpha=0.15,
                             linewidth=0)
            ax.plot(mid, rows, style, color=MODE_COLOR[key], linewidth=1.4,
                    label="observed" if style == "-" else "model")
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_yticks([])
        _style(ax)
        ax.spines["left"].set_visible(False)
        if row == 0:
            ax.legend(frameon=False, fontsize=8.2, handlelength=1.3, loc="upper right")
        if row == 1:
            ax.set_xlabel("Rank", fontsize=LABEL_SIZE)
    cbar_ax = fig.add_axes([0.925, 0.30, 0.014, 0.40])
    bar = fig.colorbar(handle, cax=cbar_ax)
    bar.set_label("First $\\rightarrow$ Last", fontsize=CBAR_LABEL_SIZE)
    bar.ax.tick_params(labelsize=CBAR_TICK_SIZE, length=2)
    fig.suptitle(f"{REPRESENTATIVE.replace('epilepsiae_', 'E')}", x=0.015,
                 ha="left", fontsize=13.5, fontweight="bold")
    _save(fig, stem)


# ------------------------------------------------------------------ panel d
def panel_d(stem: Path, frozen_plane, fields, event_field) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    points = np.asarray(frozen_plane["points_mm"], dtype=float)
    support = np.asarray(frozen_plane["support_a"], dtype=float)
    fig = plt.figure(figsize=(7.25, 3.20), layout="constrained", facecolor="white")
    grid = fig.add_gridspec(1, 6, width_ratios=[1.0, 1.0, 0.05, 1.0, 0.05, 0.10],
                            wspace=0.06)
    axes = [fig.add_subplot(grid[0, i]) for i in (0, 1, 3)]
    handles = []
    for position, (ax, (title, values, cmap, colour)) in enumerate(zip(axes, fields)):
        display = (values - np.nanmin(values)) / max(
            np.nanmax(values) - np.nanmin(values), 1e-12
        )
        x_grid, y_grid, field, _, _ = event_field(frozen_plane, display, support)
        handles.append(ax.imshow(
            field, origin="lower",
            extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
            aspect="equal", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="bilinear",
        ))
        ax.scatter(points[:, 0], points[:, 1], c=display, cmap=cmap, vmin=0.0,
                   vmax=1.0, s=34, edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_xlim(float(x_grid.min()), float(x_grid.max()))
        ax.set_ylim(float(y_grid.min()), float(y_grid.max()))
        ax.set_title(title, fontsize=11.0, pad=7, color=colour, fontweight="bold")
        ax.set_xlabel("shared axis (mm)", fontsize=9.5)
        ax.tick_params(axis="both", labelsize=8, length=2.2)
        if position == 0:
            ax.set_ylabel("transverse (mm)", fontsize=9.5)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
    left = fig.colorbar(handles[0], cax=fig.add_subplot(grid[0, 2]))
    left.ax.set_title("arrival", fontsize=7.5, pad=5)
    left.set_ticks([0.0, 1.0])
    left.ax.set_yticklabels(["late", "early"], fontsize=7.5)
    right = fig.colorbar(handles[2], cax=fig.add_subplot(grid[0, 4]))
    right.ax.set_title("power", fontsize=7.5, pad=5)
    right.set_ticks([0.0, 1.0])
    right.ax.set_yticklabels(["low", "high"], fontsize=7.5)
    fig.suptitle(f"{REPRESENTATIVE.replace('epilepsiae_', 'E')}", x=0.010,
                 ha="left", fontsize=13.5, fontweight="bold")
    _save(fig, stem)


# --------------------------------------------------------------- panels c/e
def _paired_groups(wide, reference, comparators, labels, stats_p):
    """Rows in the accepted painter's schema: reference vs each comparator."""

    groups = []
    for comparator, label in zip(comparators, labels):
        rows = [
            {"data": float(row[reference]), "null": float(row[comparator])}
            for _, row in wide.iterrows()
        ]
        groups.append({
            "label": label,
            "rows": rows,
            "summary": {"n": len(rows)},
            "display_p": float(stats_p[comparator]),
        })
    return groups


def panel_c(stem: Path, patient: pd.DataFrame, stats: dict) -> None:
    models = ["structured", "static", "structured_rank_shuffle"]
    wide = patient.pivot(index="subject", columns="model", values="contact_nll")
    wide = wide.dropna(subset=models)
    p_values = {
        comparator: stats["comparisons"][f"structured_vs_{comparator}__contact_nll"][
            "wilcoxon_two_sided_p"
        ]
        for comparator in ("static", "structured_rank_shuffle")
    }
    groups = _paired_groups(
        wide, "structured", ["static", "structured_rank_shuffle"],
        ["vs static baseline", "vs shuffled event order"], p_values,
    )
    plot_paired_data_null_groups(
        groups,
        stem.with_suffix(".png"),
        stem.with_suffix(".pdf"),
        ylabel="Held-out next-contact NLL\n(lower is better)",
        pair_tick_labels=("Structured", "Baseline"),
        figsize=(5.6, 4.1),
        ylim=(0.0, float(np.nanmax(wide.to_numpy(float))) + 0.55),
    )


def panel_e(stem: Path, patient: pd.DataFrame, cohort: dict, supportive: str) -> None:
    from scipy.stats import wilcoxon

    models = ["structured", "static"]
    wide = patient.pivot(index="subject", columns="model", values="all_contact_margin")
    wide = wide.dropna(subset=models).drop(index=supportive, errors="ignore")
    paired = cohort["paired_comparisons"]["structured_vs_static_all_contact"]

    # The frozen statistic is one-sided for structured over static.  Asterisks
    # drawn from it would print "n.s." while static is in fact the better arm,
    # which reads as "no difference".  The bracket therefore asks the symmetric
    # question, is there any difference, on the same stored paired deltas, and
    # the caption names which arm is higher.
    deltas = np.asarray(paired["paired_delta"], dtype=float)
    two_sided = float(wilcoxon(deltas, alternative="two-sided").pvalue)
    higher = "static higher" if float(np.median(deltas)) < 0 else "structured higher"

    groups = _paired_groups(
        wide, "structured", ["static"], ["structured vs static"],
        {"static": two_sided},
    )
    groups[0]["caption"] = (
        f"structured vs static, {higher}\n"
        f"n={len(deltas)}, two-sided p={two_sided:.3g}"
    )
    span = float(np.nanmax(np.abs(wide.to_numpy(float))))
    plot_paired_data_null_groups(
        groups,
        stem.with_suffix(".png"),
        stem.with_suffix(".pdf"),
        ylabel="Correspondence above null",
        pair_tick_labels=("Structured", "Static"),
        figsize=(4.8, 4.1),
        ylim=(-span - 0.12, span + 0.30),
        zero_reference=True,
    )


# --------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--readout-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--figure-root", type=Path,
        default=Path("results/paper-ready-figure/fig6_source_conditioned_rnn/figures"),
    )
    args = parser.parse_args()

    from scripts.plot_topic5_interictal_event_envelope_field import (
        _event_field,
        load_frozen,
    )

    training = yaml.safe_load((ROOT / args.training_config).read_text())
    readout = yaml.safe_load((ROOT / args.readout_config).read_text())
    output = (ROOT / args.output_root).resolve()
    figures = (ROOT / args.figure_root).resolve()
    figures.mkdir(parents=True, exist_ok=True)
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

    panel_a(figures / "fig6-panela")
    panel_b(figures / "fig6-panelb", observed, rollout, order)
    panel_c(figures / "fig6-panelc", patient_interictal, interictal_stats)
    panel_d(
        figures / "fig6-paneld", frozen_plane,
        [
            ("Model field, start 1", field_minus, "viridis_r", MODE_COLOR["A"]),
            ("Model field, start 2", field_plus, "viridis_r", MODE_COLOR["B"]),
            ("Early-ictal broadband power", ictal_field, "Blues", "#111111"),
        ],
        _event_field,
    )
    panel_e(figures / "fig6-panele", patient_ictal, ictal_stats,
            str(readout["supportive_subject"]))

    # ---------------------------------------------------------- source data
    patient_interictal.to_csv(figures / "fig6-panelc_source_data.csv", index=False)
    patient_ictal.to_csv(figures / "fig6-panele_source_data.csv", index=False)
    pd.DataFrame({
        "contact": names,
        "learned_axis_coordinate": coordinate,
        "model_field_start1": field_minus,
        "model_field_start2": field_plus,
        "early_ictal_median": ictal_field,
    }).to_csv(figures / "fig6-paneld_source_data.csv", index=False)
    pd.DataFrame([
        {"mode": MODE_LABEL[key], "source": source,
         "n_events": payload[key]["n_events"],
         **{f"median_rank__{name}": value
            for name, value in zip(payload[key]["names"], payload[key]["profile"][0])}}
        for key in MODE_KEYS
        for source, payload in (("observed", observed), ("model_rollout", rollout))
    ]).to_csv(figures / "fig6-panelb_source_data.csv", index=False)

    counts = {
        MODE_LABEL[key]: {"observed": observed[key]["n_events"],
                          "model_rollout": rollout[key]["n_events"]}
        for key in MODE_KEYS
    }
    statistics = {
        "representative_subject": REPRESENTATIVE,
        "representative_fixed_before_target_unseal": True,
        "panel_b_event_counts": counts,
        "panel_b_mode_labels": (
            "read back from the frozen interictal clustering; the model never saw "
            "them, and every model rollout is seeded from the first rank set of the "
            "observed event beside it"
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
    (figures / "fig6_statistics.json").write_text(
        json.dumps(statistics, indent=2, allow_nan=False, default=float) + "\n"
    )

    summary = opposition[1]
    (figures / "README.md").write_text(
        "# Figure 6 分面板说明\n\n"
        "每个 panel 单独成文件（本仓库 paper-ready 惯例），字号与轴/色条画法沿用已接受的"
        "Fig1、Fig3-B、Fig3 队列统计三张图。\n\n"
        "### fig6-panela\n\n"
        "模型结构。左列是到这一步为止已观察到的触点，配色与下面各图同一套先后次序色标"
        "（深紫最早、黄最晚，灰色=尚未出现）；中间圆圈内是触点之间的循环网络，灰线为所有触点"
        "共用的对称连接、红箭头为唯一那条有方向的连接；右列是模型对下一步的输出。\n\n"
        "**关注点**：输入与输出用的是同一套触点排布和同一套色标，中间只有一个网络。\n\n"
        "### fig6-panelb\n\n"
        f"{REPRESENTATIVE.replace('epilepsiae_', 'E')} 的两种间期传播模式，模式标签取自已冻结"
        "的间期聚类，模型训练时从未见过。每列一场事件，颜色为该触点在这场事件里第几个放电。"
        "右列模型推演**逐场从对应观察事件自己的第一批触点出发**，因此两列起点一致、可逐行对读；"
        "最右为各触点次序的中位数与四分位区间（实线观察、虚线模型）。\n\n"
        "**关注点**：上下两行是否呈现两种不同次序，以及每一行左右两列走向是否一致。\n\n"
        "### fig6-panelc\n\n"
        "全队列留出集下一个触点的预测难度（越低越好），结构化模型分别对静态基线与打乱事件"
        "顺序对照配对比较。小提琴+箱线+配对连线沿用已接受的队列统计画法，括号内为显著性星号。\n\n"
        "**关注点**：结构化模型是否同时低于两个对照。\n\n"
        "### fig6-paneld\n\n"
        "两张冻结模型场与同一患者发作早期宽带能量场，画在同一套真实电极几何上"
        "（沿用间期-发作共享场图的平面、插值与色条语义）。\n\n"
        f"**关注点**：两张模型场实测高度相似而非相反（全队列秩相关中位 {summary['median_rho']:+.2f}，"
        f"真正相反仅 {summary['n_opposite_below_minus_0p5']}/{summary['n_patients']} 人），"
        "不要按「两个相反方向场」去读。\n\n"
        "### fig6-panele\n\n"
        "主分析患者的跨状态对应强度，已减去各自的随机重排基线。\n\n"
        "**关注点**：星号方向——这一步是**静态基线高于结构化模型**，"
        "即方向信息没有转化成跨状态优势。\n\n"
        f"事件数：模式 A 观察 {counts['Mode A']['observed']:,} / 模型 "
        f"{counts['Mode A']['model_rollout']:,}；模式 B 观察 {counts['Mode B']['observed']:,}"
        f" / 模型 {counts['Mode B']['model_rollout']:,}。\n"
    )
    print(json.dumps({"status": "COMPLETE", "figure_root": str(figures)}))


if __name__ == "__main__":
    main()
