#!/usr/bin/env python3
"""Figures for the Topic 5 RNN training- and objective-sufficiency audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ANALYSIS = ROOT / "results/topic5_rnn_training_sufficiency_v0_1/analysis"
FIGURES = ROOT / "results/topic5_rnn_training_sufficiency_v0_1/figures"
PLATEAU_THRESHOLD = 0.002

BUDGET_COLOURS = {
    (8, 32): "#4C72B0",
    (32, 32): "#C44E52",
    (8, 64): "#7BA7D7",
    (32, 64): "#E0918F",
}
OBJECTIVE_COLOURS = {
    "teacher_forced_one_step": "#4C72B0",
    "self_fed_2step": "#DD8452",
    "self_fed_3step": "#C44E52",
    "scheduled_sampling": "#55A868",
    "static_only": "#8C8C8C",
}
OBJECTIVE_LABELS = {
    "teacher_forced_one_step": "Teacher forcing",
    "self_fed_2step": "Self-fed 2-step",
    "self_fed_3step": "Self-fed 3-step",
    "scheduled_sampling": "Scheduled sampling",
    "static_only": "Static scaffold only",
}
CONDITION_LABELS = {
    "current_teacher_forced_reference": "Previous frozen budget",
    "converged_teacher_forced": "Extended-training budget",
    "best_rollout_aware": "Rollout-aware objective",
    "static_only": "Static scaffold only",
}
ENDPOINT_LABELS = {
    "transition_correlation": "Adjacent-contact\ntransition correlation",
    "suffix_rank_wasserstein": "Event rank\ndistribution error",
    "suffix_precedence_correlation": "Pairwise order\ncorrelation",
    "suffix_precedence_mae": "Pairwise order\nerror",
    "suffix_participation_mae": "Participation\nerror",
    "event_length_wasserstein": "Event length\nerror",
    "stop_hazard_mae": "Termination\nhazard error",
    "likelihood_contact_choice_nll": "One-step next-contact\nNLL (nats/decision)",
}
HIGHER_IS_BETTER = {
    "transition_correlation",
    "suffix_precedence_correlation",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
        }
    )


def _save(fig, name: str, metadata: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    (FIGURES / f"{name}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    plt.close(fig)
    written = FIGURES / f"{name}.png"
    print(
        json.dumps(
            {
                "figure": name,
                "png": str(
                    written.relative_to(ROOT)
                    if written.is_relative_to(ROOT)
                    else written
                ),
            }
        )
    )


def _budget_label(updates: int, hidden: int) -> str:
    return f"{updates} updates/patient, hidden {hidden}"


def figure_convergence(args) -> None:
    frames = [pd.read_csv(ANALYSIS / "b1_config_cycle_summary.csv")]
    extended = ANALYSIS / "b1x_config_cycle_summary.csv"
    if extended.is_file():
        frames.append(pd.read_csv(extended))
    summary = pd.concat(frames, ignore_index=True)
    summary = (
        summary.sort_values(["cfg_updates_per_patient", "cfg_hidden_size", "coverage_cycle"])
        .drop_duplicates(
            ["cfg_updates_per_patient", "cfg_hidden_size", "coverage_cycle"],
            keep="last",
        )
    )
    patient_frames = [pd.read_csv(ANALYSIS / "b1_patient_values.csv")]
    if (ANALYSIS / "b1x_patient_values.csv").is_file():
        patient_frames.append(pd.read_csv(ANALYSIS / "b1x_patient_values.csv"))
    patients = pd.concat(patient_frames, ignore_index=True)

    _style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))

    # -- A: does more training improve held-out one-step prediction? --------
    ax = axes[0, 0]
    for (updates, hidden), group in summary.groupby(
        ["cfg_updates_per_patient", "cfg_hidden_size"]
    ):
        group = group.sort_values("coverage_cycle")
        colour = BUDGET_COLOURS.get((int(updates), int(hidden)), "#666666")
        seeds = [
            np.asarray(json.loads(value), dtype=float)
            for value in group.seed_patient_median_values
        ]
        low = np.asarray([values.min() for values in seeds])
        high = np.asarray([values.max() for values in seeds])
        ax.fill_between(
            group.coverage_cycle, low, high, color=colour, alpha=0.16, linewidth=0
        )
        ax.plot(
            group.coverage_cycle,
            group.patient_median_contact_choice_nll,
            "-o",
            color=colour,
            markersize=3,
            linewidth=1.4,
            label=_budget_label(int(updates), int(hidden)),
        )
    frozen = summary[
        (summary.cfg_updates_per_patient == 8)
        & (summary.cfg_hidden_size == 32)
        & (summary.coverage_cycle == 1)
    ]
    if not frozen.empty:
        value = float(frozen.patient_median_contact_choice_nll.iloc[0])
        ax.plot([1], [value], marker="*", markersize=11, color="black", zorder=5)
        ax.annotate(
            "previously frozen training budget",
            xy=(1, value),
            xytext=(1.5, value + 0.020),
            fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=0.6, color="black"),
        )
    ax.set_xlabel("Passes over every training event")
    ax.set_ylabel("Held-out next-contact NLL\n(nats/decision, patient median)")
    ax.set_title("A  Held-out prediction across training budgets", loc="left")
    ax.set_xticks(sorted(summary.coverage_cycle.unique()))
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)

    # -- B: has the optimisation converged? --------------------------------
    ax = axes[0, 1]
    for (updates, hidden), group in patients.groupby(
        [
            patients.config_key.str.extract(r"^u(\d+)_")[0].astype(int),
            patients.config_key.str.extract(r"_h(\d+)_")[0].astype(int),
        ]
    ):
        wide = group.pivot_table(index="subject", columns="coverage_cycle", values="value")
        cycles = sorted(wide.columns)
        gains, positions = [], []
        for previous, current in zip(cycles, cycles[1:]):
            delta = (wide[previous] - wide[current]).dropna()
            gains.append(float(np.median(delta)))
            positions.append(current)
        colour = BUDGET_COLOURS.get((int(updates), int(hidden)), "#666666")
        ax.plot(
            positions, gains, "-o", color=colour, markersize=3, linewidth=1.4
        )
    # a signed linear axis: an extra pass can also make prediction slightly
    # worse, which a log axis would silently hide
    ax.axhspan(
        -PLATEAU_THRESHOLD,
        PLATEAU_THRESHOLD,
        color="#BBBBBB",
        alpha=0.45,
        linewidth=0,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.text(
        0.98,
        PLATEAU_THRESHOLD * 1.3,
        "no-further-gain band",
        ha="right",
        va="bottom",
        transform=ax.get_yaxis_transform(),
        fontsize=7,
    )
    ax.set_xlabel("Passes over every training event")
    ax.set_ylabel("Improvement over previous pass\n(nats/decision, patient median)")
    ax.set_title("B  Improvement per pass versus the plateau rule", loc="left")
    ax.set_xticks(sorted(patients.coverage_cycle.unique())[1:])

    # -- C: is the gain generalisation rather than memorisation? -----------
    ax = axes[1, 0]
    for (updates, hidden), group in summary.groupby(
        ["cfg_updates_per_patient", "cfg_hidden_size"]
    ):
        group = group.sort_values("coverage_cycle")
        colour = BUDGET_COLOURS.get((int(updates), int(hidden)), "#666666")
        ax.plot(
            group.coverage_cycle,
            group.patient_median_train_validation_gap,
            "-o",
            color=colour,
            markersize=3,
            linewidth=1.4,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Passes over every training event")
    ax.set_ylabel("Held-out minus training NLL\n(nats/decision, patient median)")
    ax.set_title("C  Held-out minus training likelihood", loc="left")
    ax.set_xticks(sorted(summary.coverage_cycle.unique()))
    ax.text(
        0.03,
        0.92,
        "below zero: no overfitting",
        transform=ax.transAxes,
        fontsize=6.8,
        color="#444444",
        va="top",
    )

    # -- D: does the conclusion depend on the learning rate? ---------------
    ax = axes[1, 1]
    b2_path = ANALYSIS / "b2_config_cycle_summary.csv"
    if b2_path.is_file():
        b2 = pd.read_csv(b2_path)
        b2 = b2[b2.coverage_cycle == b2.coverage_cycle.max()]
        markers = {"adamw": "o", "adam": "s"}
        arms = sorted(b2.groupby(["cfg_optimizer", "cfg_weight_decay"]).groups)
        # the arms coincide to well under the marker size, so nudge them apart
        # on the log axis; the point of the panel is precisely that they agree
        offsets = np.linspace(-0.045, 0.045, len(arms))
        spread = []
        for offset, (optimizer, weight_decay) in zip(offsets, arms):
            group = b2[
                (b2.cfg_optimizer == optimizer)
                & (b2.cfg_weight_decay == weight_decay)
            ].sort_values("cfg_learning_rate")
            label = (
                f"{'AdamW' if optimizer == 'adamw' else 'Adam'}"
                f", weight decay {weight_decay:g}"
            )
            ax.errorbar(
                group.cfg_learning_rate * (10.0**offset),
                group.patient_median_contact_choice_nll,
                yerr=group.seed_patient_median_sd,
                marker=markers.get(optimizer, "o"),
                markersize=4,
                linewidth=1.1,
                capsize=2,
                alpha=0.9,
                label=label,
            )
            spread.append(group.set_index("cfg_learning_rate").patient_median_contact_choice_nll)
        agreement = float(
            pd.concat(spread, axis=1).max(axis=1).sub(
                pd.concat(spread, axis=1).min(axis=1)
            ).max()
        )
        ax.set_xscale("log")
        ax.set_xticks(sorted(b2.cfg_learning_rate.unique()))
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}")
        )
        ax.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Held-out next-contact NLL\n(nats/decision, patient median)")
        ax.legend(frameon=False, loc="lower right", fontsize=6.8)
        ax.text(
            0.03,
            0.95,
            f"arms are nudged apart to be visible;\nthey agree to {agreement:.5f} nats/decision",
            transform=ax.transAxes,
            fontsize=6.3,
            va="top",
            color="#444444",
        )
    else:
        ax.text(0.5, 0.5, "learning-rate sweep pending", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("D  Learning rate and optimiser sensitivity", loc="left")

    fig.tight_layout(w_pad=1.6, h_pad=1.6)
    _save(
        fig,
        "topic5_rnn_training_sufficiency_convergence",
        {
            "figure": "training sufficiency convergence audit",
            "panels": {
                "A": "held-out next-contact NLL versus training passes",
                "B": "paired per-patient improvement versus the 0.002 plateau rule",
                "C": "held-out minus training NLL",
                "D": "learning-rate and optimiser sensitivity at the selected budget",
            },
            "primary_endpoint": "validation contact-choice NLL, nats per decision",
            "aggregation": "seeds averaged inside a patient, patient median across patients",
            "data": "train80 inner training and inner validation only",
            "outer_heldout_read": False,
            "ictal_target_read": False,
        },
    )


def _patient_pivot(frame: pd.DataFrame, endpoint: str, index: str = "condition"):
    subset = frame[frame.endpoint == endpoint]
    return subset.pivot_table(index="subject", columns=index, values="value")


def figure_generation(args) -> None:
    """Free-running generation, one endpoint per panel, patients paired.

    Each panel answers a different question about a generated event, and every
    panel shows the same patients connected across conditions so that a level
    difference and a paired change are read from the same marks.
    """
    phase = args.phase
    generator = args.rollout
    patients = pd.read_csv(ANALYSIS / f"{phase}_patient_metrics.csv")
    tests = json.loads((ANALYSIS / f"{phase}_paired_tests.json").read_text())
    reference = tests["reference_condition"]
    rollout = patients[patients.rollout_condition == generator]
    if rollout.empty:
        raise RuntimeError(f"no rollout metrics for generator {generator}")
    likelihood = patients[patients.rollout_condition == "none"]
    static = patients[patients.rollout_condition == "static_only"]

    if phase == "c":
        order = [
            "objective_teacher_forced_one_step",
            "objective_self_fed_2step",
            "objective_self_fed_3step",
            "objective_scheduled_sampling",
        ]
        labels = {key: OBJECTIVE_LABELS[key.replace("objective_", "")] for key in order}
        colours = {
            key: OBJECTIVE_COLOURS[key.replace("objective_", "")] for key in order
        }
    else:
        order = [
            "current_teacher_forced_reference",
            "converged_teacher_forced",
            "best_rollout_aware",
        ]
        labels = {key: CONDITION_LABELS[key] for key in order}
        colours = dict(zip(order, ["#8C8C8C", "#4C72B0", "#55A868"]))
    order = [key for key in order if key in set(rollout.condition)]

    panels = [
        ("A", "transition_correlation", "Local step-to-step order", rollout),
        ("B", "suffix_rank_wasserstein", "Rank distribution of the whole event", rollout),
        ("C", "suffix_precedence_correlation", "Which contact precedes which", rollout),
        ("D", "suffix_participation_mae", "Which contacts take part", rollout),
        ("E", "event_length_wasserstein", "How long the event runs", rollout),
        ("F", "likelihood_contact_choice_nll", "One-step accuracy (guard)", likelihood),
    ]

    _style()
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.4))
    static_label = "Static\nscaffold"
    for axis, (tag, endpoint, question, source) in zip(axes.ravel(), panels):
        pivot = _patient_pivot(source, endpoint)
        columns = [key for key in order if key in pivot.columns]
        if not columns:
            axis.set_axis_off()
            continue
        include_static = not endpoint.startswith("likelihood_")
        static_pivot = _patient_pivot(static, endpoint) if include_static else None
        names, series, faces = [], [], []
        if include_static and static_pivot is not None and not static_pivot.empty:
            names.append(static_label)
            series.append(static_pivot.mean(axis=1))
            faces.append("#8C8C8C")
        for key in columns:
            names.append(labels[key].replace(" ", "\n", 1))
            series.append(pivot[key])
            faces.append(colours[key])
        table = pd.concat(series, axis=1).dropna()
        table.columns = range(table.shape[1])
        positions = np.arange(table.shape[1])
        for _, row in table.iterrows():
            axis.plot(positions, row.to_numpy(float), color="#BBBBBB", lw=0.35, zorder=1)
        for position, colour in zip(positions, faces):
            values = table[position].to_numpy(float)
            axis.scatter(
                np.full(values.size, position),
                values,
                s=8,
                color=colour,
                alpha=0.75,
                linewidths=0,
                zorder=2,
            )
            axis.plot(
                [position - 0.3, position + 0.3],
                [np.median(values)] * 2,
                color="black",
                lw=1.6,
                zorder=3,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [name.replace("\n", " ") for name in names],
            fontsize=6.5,
            rotation=32,
            ha="right",
        )
        axis.set_xlim(-0.5, table.shape[1] - 0.5)
        better = "higher is better" if endpoint in HIGHER_IS_BETTER else "lower is better"
        axis.set_ylabel(f"{ENDPOINT_LABELS[endpoint]}\n({better})", fontsize=7)
        axis.set_title(f"{tag}  {question}", loc="left")
    generator_label = (
        "constructive generator (static scaffold + ordered residual + empirical STOP)"
        if generator == "full_constructive"
        else "the model's own joint next-contact and STOP distribution"
    )
    fig.suptitle(
        f"Free generation from a revealed first contact — {generator_label}",
        fontsize=8.5,
        y=1.005,
    )
    fig.tight_layout(w_pad=1.5, h_pad=2.1)
    suffix = "" if generator == "full_constructive" else "_native"
    _save(
        fig,
        f"topic5_rnn_{'objective' if phase == 'c' else 'formal'}_sufficiency_generation{suffix}",
        {
            "figure": f"phase {phase} free-generation audit",
            "generator": generator,
            "generator_description": generator_label,
            "reference_condition": reference,
            "conditions": order,
            "panels": {tag: question for tag, _, question, _ in panels},
            "marks": (
                "one dot per patient, grey lines connect the same patient across "
                "conditions, black bar is the patient median"
            ),
            "aggregation": "seeds averaged inside a patient, patient as the unit",
            "outer_heldout_read": phase == "d",
            "ictal_target_read": False,
        },
    )


def figure_cohort(args) -> None:
    """Do Epilepsiae and Yuquan move in the same direction?"""
    phase = args.phase
    tests = json.loads((ANALYSIS / f"{phase}_paired_tests.json").read_text())
    reference = tests["reference_condition"]
    endpoints = [
        "transition_correlation",
        "suffix_rank_wasserstein",
        "suffix_precedence_correlation",
        "suffix_participation_mae",
        "event_length_wasserstein",
        "likelihood_contact_choice_nll",
    ]
    comparisons = [key for key in tests["paired_vs_reference"] if key.endswith(reference)]
    _style()
    fig, axes = plt.subplots(
        1, max(len(comparisons), 1), figsize=(3.4 * max(len(comparisons), 1), 3.2),
        squeeze=False,
    )
    strata = ["epilepsiae", "yuquan", "combined"]
    stratum_colours = {"epilepsiae": "#4C72B0", "yuquan": "#DD8452", "combined": "#333333"}
    for axis, comparison in zip(axes[0], comparisons):
        offsets = np.linspace(-0.26, 0.26, len(strata))
        for offset, stratum in zip(offsets, strata):
            centres, lows, highs = [], [], []
            for endpoint in endpoints:
                entry = tests["paired_vs_reference"][comparison].get(endpoint, {}).get(stratum, {})
                if "median_gain" not in entry:
                    centres.append(np.nan)
                    lows.append(np.nan)
                    highs.append(np.nan)
                    continue
                scale = abs(entry.get("reference_median") or 1.0) or 1.0
                low, high = entry["bootstrap_ci_median_gain"]
                centres.append(entry["median_gain"] / scale)
                lows.append(low / scale)
                highs.append(high / scale)
            positions = np.arange(len(endpoints)) + offset
            centres = np.asarray(centres, float)
            axis.errorbar(
                positions,
                centres,
                yerr=[centres - np.asarray(lows, float), np.asarray(highs, float) - centres],
                fmt="o",
                ms=3.5,
                lw=1.0,
                capsize=1.8,
                color=stratum_colours[stratum],
                label=stratum.capitalize() if stratum != "combined" else "Combined",
            )
        axis.axhline(0.0, color="black", lw=0.8)
        axis.set_xticks(range(len(endpoints)))
        axis.set_xticklabels(
            [ENDPOINT_LABELS[endpoint].replace("\n", " ") for endpoint in endpoints],
            rotation=28,
            ha="right",
            fontsize=6.5,
        )
        axis.set_ylabel("Relative gain over the reference\n(patient median, bootstrap CI)")
        axis.set_title(
            comparison.replace("__vs__", " vs ").replace("objective_", ""),
            loc="left",
            fontsize=8,
        )
        axis.legend(frameon=False, fontsize=7)
    fig.tight_layout(w_pad=1.6)
    _save(
        fig,
        f"topic5_rnn_{'objective' if phase == 'c' else 'formal'}_sufficiency_cohort",
        {
            "figure": f"phase {phase} cohort stratification",
            "reference_condition": reference,
            "strata": strata,
            "endpoints": endpoints,
            "normalisation": "median paired gain divided by the reference median level",
            "outer_heldout_read": phase == "d",
            "ictal_target_read": False,
        },
    )


def _normalised_rank_matrix(groups: np.ndarray, counts: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups, int)
    denominator = np.maximum(np.asarray(counts, int) - 1, 1)
    return np.where(groups >= 0, groups / denominator[:, None], np.nan)


def _precedence(groups: np.ndarray) -> np.ndarray:
    from src.topic5_rank_distribution import pairwise_precedence

    return pairwise_precedence(groups)


def figure_representative(args) -> None:
    """One typical patient: does a free-run event look like a real event?"""
    phase = args.phase
    root = ROOT / "results/topic5_rnn_training_sufficiency_v0_1"
    condition_root = (
        root / "development/c_objectives"
        if phase == "c"
        else root / "formal"
    )
    generator = args.rollout
    patients = pd.read_csv(ANALYSIS / f"{phase}_patient_metrics.csv")
    rollout = patients[
        (patients.rollout_condition == generator)
        & (patients.endpoint == "transition_correlation")
    ]
    condition = args.condition or sorted(rollout.condition.unique())[0]
    subset = rollout[rollout.condition == condition]
    if subset.empty:
        raise RuntimeError(f"condition {condition} has no rollout metrics")
    median = float(subset.value.median())
    subject = str(
        subset.iloc[(subset.value - median).abs().to_numpy().argmin()].subject
    )
    seed_dirs = sorted((condition_root / condition).glob("seed_*"))
    npz_path = seed_dirs[0] / subject / "rollouts.npz"
    if not npz_path.is_file():
        raise RuntimeError(f"missing rollout archive: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        observed = archive["observed_group_ids"]
        observed_count = archive["observed_group_count"]
        generated = archive[f"{generator}__event_group_ids"]
        generated_count = archive[f"{generator}__event_group_count"]

    observed_rank = _normalised_rank_matrix(observed, observed_count)
    generated_rank = _normalised_rank_matrix(generated, generated_count)
    order = np.argsort(np.nanmean(observed_rank, axis=0))
    participation = [
        np.mean(observed >= 0, axis=0)[order],
        np.mean(generated >= 0, axis=0)[order],
    ]
    mean_rank = [
        np.nanmean(observed_rank, axis=0)[order],
        np.nanmean(generated_rank, axis=0)[order],
    ]
    observed_precedence = _precedence(observed)[np.ix_(order, order)]
    generated_precedence = _precedence(generated)[np.ix_(order, order)]

    _style()
    fig = plt.figure(figsize=(7.2, 4.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.62, wspace=0.55)
    positions = np.arange(len(order))

    ax = fig.add_subplot(grid[0, 0])
    ax.plot(positions, participation[0], "-o", ms=3, lw=1.3, color="#333333", label="Observed")
    ax.plot(positions, participation[1], "-s", ms=3, lw=1.3, color="#C44E52", label="Generated")
    ax.set_xlabel("Contact (by observed mean rank)", fontsize=7)
    ax.set_ylabel("Participation probability")
    ax.set_ylim(0, 1)
    ax.set_title("A  Which contacts take part", loc="left", fontsize=8.5)
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(positions, mean_rank[0], "-o", ms=3, lw=1.3, color="#333333")
    ax.plot(positions, mean_rank[1], "-s", ms=3, lw=1.3, color="#C44E52")
    ax.set_xlabel("Contact (by observed mean rank)", fontsize=7)
    ax.set_ylabel("Mean normalised rank\n(0 = first, 1 = last)")
    ax.set_ylim(0, 1)
    ax.set_title("B  When they fire", loc="left", fontsize=8.5)

    ax = fig.add_subplot(grid[0, 2])
    ax.scatter(
        observed_precedence.ravel(),
        generated_precedence.ravel(),
        s=7,
        color="#C44E52",
        alpha=0.55,
        linewidths=0,
    )
    ax.plot([0, 1], [0, 1], color="black", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Observed P(i before j)")
    ax.set_ylabel("Generated P(i before j)")
    ax.set_title("C  Which precedes which", loc="left", fontsize=8.5)

    matrices = [observed_precedence, generated_precedence]
    titles = ["D  Observed order", "E  Generated order"]
    images = []
    for column, (matrix, title) in enumerate(zip(matrices, titles)):
        ax = fig.add_subplot(grid[1, column])
        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
        images.append(image)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Contact j")
        if column == 0:
            ax.set_ylabel("Contact i")
    bar = fig.colorbar(images[0], ax=fig.axes[-2:], fraction=0.03, pad=0.02)
    bar.set_label("P(i before j)")

    ax = fig.add_subplot(grid[1, 2])
    difference = generated_precedence - observed_precedence
    limit = float(np.nanmax(np.abs(difference))) or 1.0
    image = ax.imshow(difference, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_title("F  Generated − observed", loc="left", fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Contact j")
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("Difference")

    _save(
        fig,
        f"topic5_rnn_representative_patient_{'development' if phase == 'c' else 'formal'}"
        + ("" if generator == "full_constructive" else "_native"),
        {
            "figure": "representative patient, observed versus freely generated events",
            "phase": phase,
            "generator": generator,
            "condition": condition,
            "subject": subject,
            "subject_selection": (
                "the patient whose transition correlation is closest to the "
                "cohort median for this condition; not the best patient"
            ),
            "seed_directory": str(seed_dirs[0].relative_to(ROOT)),
            "n_events": int(observed.shape[0]),
            "outer_heldout_read": phase == "d",
            "ictal_target_read": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=("convergence", "generation", "cohort", "representative"),
        required=True,
    )
    parser.add_argument("--phase", choices=("c", "d"), default="c")
    parser.add_argument("--condition", default=None)
    parser.add_argument(
        "--rollout",
        choices=("full_constructive", "native_model"),
        default="full_constructive",
    )
    args = parser.parse_args()
    {
        "convergence": figure_convergence,
        "generation": figure_generation,
        "cohort": figure_cohort,
        "representative": figure_representative,
    }[args.figure](args)


if __name__ == "__main__":
    main()
