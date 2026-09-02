#!/usr/bin/env python3
"""Paper-ready Figure 6: what the RNN learns, and what motifs do not add."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1"
REPAIR = RESULT / "repair_v0_2"
OLD_SOURCE = (ROOT / "results/paper-ready-figure/fig6_multiscale_scaffold_v0_5/"
              "figures/source_data")
DESTINATION = ROOT / "results/paper-ready-figure/fig6_dynamical_motif_rnn_v0_2/figures"
SOURCE = DESTINATION / "source_data"

INK = "#25282B"
GREY = "#AAB0B5"
LIGHT = "#DDE1E4"
MOTIF = {
    "DM0_ISOTROPIC": "#7F878D",
    "DM1_FREE_AXIS": "#79B8C2",
    "DM2_LOCAL_DIRECTIONAL": "#237D9B",
    "DM3_AXIS_FEEDFORWARD_TRANSIENT": "#C95B3E",
}
MOTIF_LABEL = {
    "DM0_ISOTROPIC": "Even local\nspread",
    "DM1_FREE_AXIS": "Elongated\nspread",
    "DM2_LOCAL_DIRECTIONAL": "Direction-biased\nspread",
    "DM3_AXIS_FEEDFORWARD_TRANSIENT": "Forward\nrelay",
}
ORDER = list(MOTIF)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.4,
    "axes.labelsize": 7.8,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 450,
})


def bootstrap_ci(values: np.ndarray, seed: int = 20260816) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(values, (10000, values.size), replace=True), axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(np.median(values)), float(low), float(high)


def p_one_sided(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values) & (values != 0)]
    return (float(wilcoxon(values, alternative="greater").pvalue)
            if values.size else 1.0)


def panel_letter(axis, letter: str, x: float = -0.16, y: float = 1.10) -> None:
    axis.text(x, y, letter, transform=axis.transAxes, fontsize=14,
              fontweight="bold", ha="left", va="bottom", color="black")


def clean_axis(axis) -> None:
    axis.spines["left"].set_color(INK)
    axis.spines["bottom"].set_color(INK)
    axis.tick_params(colors=INK, width=0.8, length=3)


def draw_motif_icon(axis, center: tuple[float, float], model: str,
                    radius: float = 0.050) -> None:
    x0, y0 = center
    colour = MOTIF[model]
    points = np.array([
        [-1, 0], [-0.5, 0.75], [0, 0], [0.5, -0.75], [1, 0],
        [0, 0.78], [0, -0.78],
    ], float)
    points[:, 0] = x0 + radius * points[:, 0]
    points[:, 1] = y0 + radius * points[:, 1]
    centre = 2
    if model == "DM0_ISOTROPIC":
        targets = [0, 1, 3, 4, 5, 6]
    elif model == "DM1_FREE_AXIS":
        targets = [0, 4]
    elif model == "DM2_LOCAL_DIRECTIONAL":
        targets = [4, 3]
    else:
        targets = [4]
    for target in targets:
        axis.annotate("", xy=points[target], xytext=points[centre],
                      arrowprops=dict(arrowstyle="-|>", lw=1.0, color=colour,
                                      mutation_scale=6, shrinkA=3, shrinkB=3))
    if model == "DM3_AXIS_FEEDFORWARD_TRANSIENT":
        chain = np.array([[x0 - radius, y0], [x0, y0], [x0 + radius, y0]])
        for left, right in zip(chain[:-1], chain[1:]):
            axis.annotate("", xy=right, xytext=left,
                          arrowprops=dict(arrowstyle="-|>", lw=1.3, color=colour,
                                          mutation_scale=7, shrinkA=3, shrinkB=3))
    axis.scatter(points[:, 0], points[:, 1], s=10, facecolor="white",
                 edgecolor=colour, linewidth=0.8, zorder=4)
    axis.scatter([x0], [y0], s=18, facecolor=colour, edgecolor="white",
                 linewidth=0.5, zorder=5)


def architecture_panel(axis) -> None:
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    cmap = plt.get_cmap("viridis")

    # Observed SEEG ranks, with more than one rank shown.
    axis.text(0.075, 0.86, "Observed ranks", ha="center", fontsize=7.8)
    for rank, y in enumerate([0.73, 0.63, 0.53]):
        for contact in range(4):
            active = contact in ({0, 2}, {1}, {3})[rank]
            axis.add_patch(plt.Rectangle(
                (0.02 + contact * 0.032, y), 0.022, 0.055,
                facecolor=cmap(rank / 3) if active else "white",
                edgecolor=INK, linewidth=0.55))
    axis.annotate("", xy=(0.255, 0.64), xytext=(0.17, 0.64),
                  arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))

    # Tissue RNN inside one large circle; SEEG readout contacts sit on the rim.
    circle = plt.Circle((0.42, 0.65), 0.17, facecolor="#F8F9F9",
                        edgecolor=INK, linewidth=1.1)
    axis.add_patch(circle)
    rng = np.random.default_rng(4)
    nodes = rng.normal(size=(34, 2))
    nodes /= np.maximum(np.linalg.norm(nodes, axis=1, keepdims=True), 1)
    nodes *= rng.uniform(0.02, 0.145, size=(34, 1))
    nodes += np.array([0.42, 0.65])
    distance = np.linalg.norm(nodes[:, None] - nodes[None, :], axis=-1)
    for i, j in zip(*np.where((distance > 0) & (distance < 0.065))):
        if i < j:
            axis.plot(nodes[[i, j], 0], nodes[[i, j], 1], color="#BBC0C4",
                      lw=0.45, zorder=1)
    axis.scatter(nodes[:, 0], nodes[:, 1], s=9, color="#727B82", zorder=2)
    angles = np.linspace(0.25, 2 * np.pi - 0.4, 9)
    contacts = np.column_stack([0.42 + 0.17 * np.cos(angles),
                                0.65 + 0.17 * np.sin(angles)])
    axis.scatter(contacts[:, 0], contacts[:, 1], s=22, facecolor="white",
                 edgecolor=INK, linewidth=0.8, zorder=4)
    axis.text(0.42, 0.65, "Tissue RNN", ha="center", va="center",
              fontsize=8.2, fontweight="bold",
              bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=5)

    axis.annotate("", xy=(0.73, 0.64), xytext=(0.60, 0.64),
                  arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    axis.text(0.84, 0.86, "Predicted ranks", ha="center", fontsize=7.8)
    for rank, y in enumerate([0.73, 0.63, 0.53]):
        for contact in range(4):
            active = contact in ({1}, {3}, set())[rank]
            axis.add_patch(plt.Rectangle(
                (0.77 + contact * 0.032, y), 0.022, 0.055,
                facecolor=cmap((rank + 1) / 3) if active else "white",
                edgecolor=INK, linewidth=0.55))
    axis.text(0.84, 0.47, "+ STOP", ha="center", fontsize=7.5)

    x_positions = [0.11, 0.36, 0.61, 0.86]
    for x, model in zip(x_positions, ORDER):
        draw_motif_icon(axis, (x, 0.23), model, 0.065)
        short = {
            "DM0_ISOTROPIC": "Even",
            "DM1_FREE_AXIS": "Elongated",
            "DM2_LOCAL_DIRECTIONAL": "Direction-\nbiased",
            "DM3_AXIS_FEEDFORWARD_TRANSIENT": "Forward\nrelay",
        }[model]
        axis.text(x, 0.08, short, ha="center", va="top",
                  fontsize=6.8, color=MOTIF[model])
    panel_letter(axis, "A", x=-0.05, y=1.01)


def event_heatmaps(container, source: pd.DataFrame) -> None:
    sub = container.subgridspec(2, 3, width_ratios=[1, 1, 0.035],
                                hspace=0.10, wspace=0.08)
    axes = [[plt.subplot(sub[row, col]) for col in range(2)] for row in range(2)]
    bar_axis = plt.subplot(sub[:, 2])
    image = None
    for row, template in enumerate(["TA", "TB"]):
        for col, field in enumerate(["data", "generated"]):
            axis = axes[row][col]
            block = source[(source.template == template) & (source.field_type == field)]
            matrix = block.pivot_table(index="display_row", columns="event_column",
                                       values="normalized_rank", aggfunc="first").to_numpy()
            masked = np.ma.masked_invalid(matrix)
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad("#ECEEEF")
            image = axis.imshow(masked, aspect="auto", interpolation="nearest",
                                cmap=cmap, vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title("Data" if col == 0 else "RNN generated", fontsize=8.8, pad=4)
            if col == 0:
                axis.set_ylabel(template, rotation=0, ha="right", va="center",
                                labelpad=7, fontsize=9.0,
                                color="#B42836" if template == "TA" else "#286DA8",
                                fontweight="bold")
    bar = plt.colorbar(image, cax=bar_axis)
    bar.set_ticks([0, 1])
    bar.set_ticklabels(["Early", "Late"])
    bar_axis.set_title("Rank", fontsize=7.2, pad=3)
    axes[0][0].text(-0.05, 1.10, "B", transform=axes[0][0].transAxes,
                    fontsize=14, fontweight="bold", va="bottom")
    axes[0][0].text(0.08, 1.10, "E1146", transform=axes[0][0].transAxes,
                    fontsize=8.2, fontweight="bold", va="bottom")


def history_panel(axis, table: pd.DataFrame, stats: dict) -> None:
    values = table["reassigned_minus_true_gain_nats"].to_numpy(float)
    violin = axis.violinplot(values, positions=[0], widths=0.55,
                             showmeans=False, showmedians=False, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor("#B8D7DF")
        body.set_edgecolor("#4B7D88")
        body.set_alpha(0.85)
    rng = np.random.default_rng(6)
    axis.scatter(rng.uniform(-0.10, 0.10, values.size), values, s=13,
                 facecolor="#32798A", edgecolor="white", linewidth=0.35, zorder=3)
    median, low, high = bootstrap_ci(values)
    axis.plot([0, 0], [low, high], color=INK, lw=2.0, zorder=4)
    axis.scatter([0], [median], s=38, color=INK, zorder=5)
    axis.axhline(0, color=GREY, lw=0.8, ls=(0, (3, 2)))
    axis.set_xlim(-0.55, 0.55)
    axis.set_xticks([0])
    axis.set_xticklabels(["Continuation\nshuffled"])
    axis.set_ylabel("Error increase after shuffle\n(nats)")
    p = p_one_sided(values)
    axis.text(0.72, 0.96, f"***\n{int((values > 0).sum())}/{len(values)}",
              transform=axis.transAxes, ha="center", va="top", fontsize=7.6)
    clean_axis(axis)
    panel_letter(axis, "C")
    stats["history_shuffle"] = {
        "n": len(values), "median": median, "ci95": [low, high],
        "positive": int((values > 0).sum()), "p_one_sided": p,
    }


def dose_axis(axis, profile: pd.DataFrame, sweep: str, colour: str,
              xlabel: str, stats: dict) -> None:
    block = profile[profile.sweep == sweep].copy()
    best = (block.groupby(["subject", "value"], as_index=False)
            .calibration_contact_nll.min())
    zero = best[best.value == 0].set_index("subject").calibration_contact_nll
    best["delta"] = [row.calibration_contact_nll - zero[row.subject]
                     for row in best.itertuples()]
    grouped = []
    for value, sample in best.groupby("value"):
        med, low, high = bootstrap_ci(sample.delta.to_numpy(), seed=17)
        grouped.append((float(value), med, low, high))
    curve = np.asarray(grouped)
    axis.fill_between(curve[:, 0], curve[:, 2], curve[:, 3], color=colour, alpha=0.20,
                      linewidth=0)
    axis.plot(curve[:, 0], curve[:, 1], color=colour, lw=2.0)
    axis.axhline(0, color=GREY, lw=0.8, ls=(0, (3, 2)))
    axis.axvline(0, color=GREY, lw=0.8, ls=(0, (3, 2)))
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Next-contact error change\n(nats)")
    clean_axis(axis)
    stats[f"dose_{sweep}"] = [
        {"value": row[0], "median": row[1], "ci95": [row[2], row[3]]}
        for row in grouped]


def hard_transition_panel(axis, table: pd.DataFrame, stats: dict) -> None:
    # This panel asks only where the event goes.  Select checkpoints by
    # next-contact loss alone so STOP accuracy cannot choose the model.
    table = table[table.tag == "contact_selected"].copy()
    subsets = ["all", "late_after_two_predictions", "distal_train_q75"]
    labels = ["All transitions", "Later transitions", "Longer spatial steps"]
    symbols = ["o", "s", "^"]
    pivot = table.pivot_table(index=["subject", "subset"], columns="model_id",
                              values="contact_nll")
    for subset, label, marker in zip(subsets, labels, symbols):
        if subset not in pivot.index.get_level_values("subset"):
            continue
        block = pivot.xs(subset, level="subset")
        base = block[ORDER[0]]
        medians, lows, highs = [], [], []
        for model in ORDER:
            delta = (base - block[model]).dropna().to_numpy()
            med, low, high = bootstrap_ci(delta, seed=25)
            medians.append(med); lows.append(low); highs.append(high)
        x = np.arange(4)
        axis.plot(x, medians, marker=marker, ms=4.5, lw=1.35, label=label,
                  color={"All transitions": INK, "Later transitions": "#397F8C",
                         "Longer spatial steps": "#B76A38"}[label])
        axis.fill_between(x, lows, highs,
                          color={"All transitions": INK, "Later transitions": "#397F8C",
                                 "Longer spatial steps": "#B76A38"}[label],
                          alpha=0.10, linewidth=0)
        stats[f"hard_{subset}"] = {
            model: {"median_gain_vs_even": medians[i], "ci95": [lows[i], highs[i]]}
            for i, model in enumerate(ORDER)}
    axis.axhline(0, color=GREY, lw=0.8, ls=(0, (3, 2)))
    axis.set_xticks(range(4))
    axis.set_xticklabels(["Even", "Elongated", "Biased", "Relay"],
                         rotation=15, ha="right")
    for tick, model in zip(axis.get_xticklabels(), ORDER):
        tick.set_color(MOTIF[model])
    axis.set_ylabel("Gain vs even spread\n(nats)")
    axis.legend(frameon=False, loc="upper left", ncol=1)
    clean_axis(axis)
    panel_letter(axis, "E")


def implementation_panel(container, capacity: pd.DataFrame,
                         ablation: pd.DataFrame, stats: dict) -> None:
    """Separate use of the recurrent path from task-level necessity."""
    sub = container.subgridspec(1, 2, wspace=0.48)
    axes = [plt.subplot(sub[0, column]) for column in range(2)]
    contact = ablation.pivot(index="subject", columns="ablation", values="contact_nll")
    stop = ablation.pivot(index="subject", columns="ablation", values="stop_bce")
    samples = {
        "Next contact": [
            (contact["history_without_spatial_mixing"]
             - contact["history_with_spatial_mixing"]).to_numpy(float),
            capacity.contact_gain_rnn_minus_static.to_numpy(float),
        ],
        "Event ending": [
            (stop["history_without_spatial_mixing"]
             - stop["history_with_spatial_mixing"]).to_numpy(float),
            capacity.stop_gain_rnn_minus_static.to_numpy(float),
        ],
    }
    labels = ["Remove local\nrecurrence", "History-only\nmodel"]
    colours = ["#C95B3E", "#397F8C"]
    rng = np.random.default_rng(15)
    for endpoint_index, (axis, (endpoint, endpoint_samples)) in enumerate(
            zip(axes, samples.items())):
        for position, (sample, colour) in enumerate(zip(endpoint_samples, colours)):
            violin = axis.violinplot(
                sample, positions=[position], widths=0.62,
                showmeans=False, showmedians=False, showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(colour)
                body.set_edgecolor(colour)
                body.set_alpha(0.18)
            axis.scatter(position + rng.uniform(-0.10, 0.10, sample.size), sample,
                         s=8, color=colour, alpha=0.65, edgecolor="none")
            med, low, high = bootstrap_ci(sample, seed=30 + 2 * endpoint_index + position)
            axis.plot([position, position], [low, high], color=INK, lw=1.5)
            axis.scatter([position], [med], color=INK, s=22, zorder=4)
            stats[f"implementation_{endpoint}_{position}"] = {
                "comparison": labels[position].replace("\n", " "),
                "n": sample.size, "median": med, "ci95": [low, high],
                "positive": int((sample > 0).sum()), "p_one_sided": p_one_sided(sample),
            }
        axis.axhline(0, color=GREY, lw=0.8, ls=(0, (3, 2)))
        axis.set_xticks([0, 1])
        axis.set_xticklabels(labels, rotation=18, ha="right")
        axis.set_title(endpoint, fontsize=8.0, pad=4)
        axis.set_ylabel("Held-out error increase\n(nats)" if endpoint_index == 0 else "")
        clean_axis(axis)
        if endpoint == "Event ending":
            # The between-model STOP effect is much smaller than the within-RNN
            # lesion, so show its full patient distribution at its own scale.
            inset = axis.inset_axes([0.57, 0.52, 0.40, 0.42])
            sample = endpoint_samples[1]
            violin = inset.violinplot(
                sample, positions=[0], widths=0.65,
                showmeans=False, showmedians=False, showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(colours[1])
                body.set_edgecolor(colours[1])
                body.set_alpha(0.22)
            inset.scatter(rng.uniform(-0.10, 0.10, sample.size), sample,
                          s=6, color=colours[1], alpha=0.65, edgecolor="none")
            med, low, high = bootstrap_ci(sample, seed=77)
            inset.plot([0, 0], [low, high], color=INK, lw=1.2)
            inset.scatter([0], [med], color=INK, s=15, zorder=4)
            inset.axhline(0, color=GREY, lw=0.6, ls=(0, (3, 2)))
            inset.set_xlim(-0.45, 0.45)
            inset.set_ylim(-0.005, max(0.10, float(sample.max()) * 1.10))
            inset.set_xticks([])
            inset.tick_params(labelsize=5.8, length=2)
            inset.set_title("History-only zoom", fontsize=6.0, pad=2)
            inset.spines["top"].set_visible(False)
            inset.spines["right"].set_visible(False)
    panel_letter(axes[0], "F", x=-0.38)


def persistence_panel(axis, table: pd.DataFrame, stats: dict) -> None:
    rng = np.random.default_rng(16)
    for position, model in enumerate(ORDER):
        sample = (table.observed_cosine - table[f"{model}_cosine_mean"]).to_numpy(float)
        violin = axis.violinplot(sample, positions=[position], widths=0.64,
                                 showmeans=False, showmedians=False, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(MOTIF[model]); body.set_edgecolor(MOTIF[model]); body.set_alpha(0.22)
        axis.scatter(position + rng.uniform(-0.10, 0.10, sample.size), sample,
                     s=9, color=MOTIF[model], alpha=0.65, edgecolor="none")
        med, low, high = bootstrap_ci(sample, seed=41 + position)
        axis.plot([position, position], [low, high], color=INK, lw=1.6)
        axis.scatter([position], [med], color=INK, s=24, zorder=4)
        stats[f"persistence_{model}"] = {
            "median_data_minus_model": med, "ci95": [low, high],
            "positive": int((sample > 0).sum()), "n": len(sample)}
    axis.axhline(0, color=GREY, lw=0.8, ls=(0, (3, 2)))
    axis.set_xticks(range(4))
    axis.set_xticklabels(["Even", "Elongated", "Biased", "Relay"],
                         rotation=20, ha="right")
    for tick, model in zip(axis.get_xticklabels(), ORDER):
        tick.set_color(MOTIF[model])
    axis.set_ylabel("Data − model\ncontinuity")
    clean_axis(axis)
    panel_letter(axis, "G", x=-0.06, y=1.02)


def main() -> None:
    required = {
        "events": OLD_SOURCE / "panel_b_e1146_data_generated_events.csv",
        "history": OLD_SOURCE / "panel_c_interictal_v0_5_28_patients.csv",
        "dose": RESULT / "DOSE_RESPONSE_PROFILE.csv",
        "hard": REPAIR / "HARD_TRANSITION_METRICS_PER_PATIENT.csv",
        "capacity": REPAIR / "CAPACITY_MATCHED_STATIC_PER_PATIENT.csv",
        "ablation": REPAIR / "M0_STATE_PATH_ABLATION_PER_PATIENT.csv",
        "persistence": RESULT / "PERSISTENCE_MODEL_GAP_PER_PATIENT.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing final figure inputs: " + ", ".join(missing))
    data = {key: pd.read_csv(path) for key, path in required.items()}
    if data["capacity"].subject.nunique() != 28:
        raise RuntimeError("capacity-matched cohort is incomplete")
    if data["hard"].query("tag == 'contact_selected'").subject.nunique() != 28:
        raise RuntimeError("contact-selected hard-transition cohort is incomplete")

    # Render at the final two-column print width.  A larger canvas that is
    # subsequently shrunk would silently make the labels unreadable.
    figure = plt.figure(figsize=(7.15, 8.85), constrained_layout=False)
    grid = GridSpec(4, 12, figure=figure,
                    height_ratios=[1.08, 0.90, 0.90, 0.64],
                    hspace=0.70, wspace=1.00,
                    left=0.105, right=0.940, top=0.975, bottom=0.070)
    architecture = figure.add_subplot(grid[0, 0:6])
    architecture_panel(architecture)
    event_heatmaps(grid[0, 6:12], data["events"])

    history = figure.add_subplot(grid[1, 0:3])
    dose_eta = figure.add_subplot(grid[1, 3:6])
    dose_beta = figure.add_subplot(grid[1, 6:9])
    dose_gamma = figure.add_subplot(grid[1, 9:12])
    stats: dict[str, object] = {}
    history_panel(history, data["history"], stats)
    dose_axis(dose_eta, data["dose"], "eta", MOTIF["DM1_FREE_AXIS"],
              "Strength of elongation", stats)
    dose_axis(dose_beta, data["dose"], "beta", MOTIF["DM2_LOCAL_DIRECTIONAL"],
              "Directional bias", stats)
    dose_axis(dose_gamma, data["dose"], "gamma",
              MOTIF["DM3_AXIS_FEEDFORWARD_TRANSIENT"],
              "Forward-relay strength", stats)
    dose_beta.set_ylabel("")
    dose_gamma.set_ylabel("")
    panel_letter(dose_eta, "D", x=-0.18)

    hard = figure.add_subplot(grid[2, 0:5])
    capacity_slot = grid[2, 5:12]
    persistence = figure.add_subplot(grid[3, 0:12])
    hard_transition_panel(hard, data["hard"], stats)
    implementation_panel(capacity_slot, data["capacity"], data["ablation"], stats)
    persistence_panel(persistence, data["persistence"], stats)

    DESTINATION.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    stem = DESTINATION / "topic5_figure6_dynamical_motif_rnn_v0_2"
    for extension in ("png", "pdf", "svg"):
        figure.savefig(stem.with_suffix(f".{extension}"), facecolor="white")
    plt.close(figure)

    for key, path in required.items():
        shutil.copy2(path, SOURCE / f"panel_source_{key}.csv")
    (DESTINATION / "FIGURE6_STATS.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    metadata = {
        "asset_id": "topic5_figure6_dynamical_motif_rnn_v0_2",
        "n_patients": 28,
        "case_subject": "epilepsiae_1146",
        "panels": {
            "A": "ordered SEEG input, full-tissue RNN, output, and four nested local rules",
            "B": "representative data and frozen order-aware RNN-generated interictal sequences",
            "C": "held-out cost of disrupting the true event history",
            "D": "contact-prediction dose curves for elongation and directional bias",
            "E": "contact-selected motif increments on all, late, and spatially longer transitions",
            "F": "within-RNN recurrence ablation beside a parameter-capped history-only model",
            "G": "directional continuity remaining unexplained by each motif",
        },
        "excluded_from_main": {
            "seizure": "absolute incremental prediction remained negative after parity restriction",
            "old_synthetic": "gamma was absent from its generator",
            "m3_controls": "all collapsed to the zero-motif parent",
        },
        "panel_B_model_note": (
            "Panel B reuses the frozen v0.5 order-aware RNN case source; "
            "TA/TB are post-training display labels and were not training targets."),
        "panel_C_model_note": (
            "Panel C reuses the frozen v0.5 true-history versus shuffled-continuation control "
            "to establish that ordered history is informative before testing motif restrictions."),
    }
    (DESTINATION / "FIGURE6_METADATA.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    (DESTINATION / "README.md").write_text(
        "### topic5_figure6_dynamical_motif_rnn_v0_2.png\n\n"
        "A–B 从真实 SEEG rank-set 输入开始，展示完整组织 RNN、四种局部传播规则，以及代表患者中真实与生成的两类间期序列。C 说明打乱真实前后段对应会稳定增加留出误差，证明模型确实使用了有序历史。D–E 检验把局部传播拉长、加方向偏置或单向接力是否进一步改善 next-contact 预测，并分别查看全部、较晚和空间跨度较大的转移。F 用参数量不超过 RNN 的 history-only 对照区分 contact 预测与事件终止；G 显示真实事件仍比四种模型生成的事件保持更多方向连续性。\n\n"
        "**关注点**：承重结论是有序历史和普通递归计算有用，但当前四种几何 motif 没有获得额外的 held-out 支持；因此不能把阴性写成大脑不存在方向传播，只能说该 rank-only 任务没有识别出这些具体规则。\n",
        encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
