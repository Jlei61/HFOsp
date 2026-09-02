#!/usr/bin/env python3
"""Supplementary figure for the frozen-RNN shared-component lesion test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULT = (
    ROOT
    / "results/topic5_latent_propagation_landscape_v0_2"
    / "shared_functional_computation_necessity_v0_2"
)
FIGURES = (
    ROOT
    / "results/paper-ready-figure/fig6_interictal_crossstate_response_r5_candidate/figures"
)
SOURCE = FIGURES / "source_data"
STEM = "supplement_topic5_shared_component_necessity_v0_2"
MAIN_STEM = "topic5_figure6_interictal_crossstate_response_r5_candidate"

COLORS = {
    "SHARED": "#B24A3A",
    "SHARED_MINUS_ORTHOGONAL": "#2F6B9A",
    "SHARED_MINUS_C_SUFFIX": "#6F5A9C",
    "SHARED_MINUS_PCA": "#4D8066",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def scatter_distribution(
    ax: plt.Axes,
    patient: pd.DataFrame,
    inference: pd.DataFrame,
) -> None:
    metrics = (
        "SHARED",
        "SHARED_MINUS_ORTHOGONAL",
        "SHARED_MINUS_C_SUFFIX",
    )
    labels = (
        "Shared\nresponse",
        "vs unrelated\ndirection",
        "vs shuffled\nendings",
    )
    rng = np.random.default_rng(160816)
    all_values: list[float] = []
    for x, metric in enumerate(metrics):
        values = patient[
            patient.phase.eq("ALL") & patient.metric.eq(metric)
        ]["dose_auc_nll"].dropna().to_numpy(float)
        all_values.extend(values.tolist())
        violin = ax.violinplot(
            values,
            positions=[x],
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(COLORS[metric])
            body.set_edgecolor("none")
            body.set_alpha(0.18)
        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        ax.scatter(
            x + jitter,
            values,
            s=12,
            facecolor=COLORS[metric],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.70,
            zorder=3,
        )
        stat = inference[inference.metric.eq(metric)].iloc[0]
        ax.errorbar(
            x,
            stat.median_dose_auc_nll,
            yerr=np.asarray([
                [stat.median_dose_auc_nll - stat.ci95_low],
                [stat.ci95_high - stat.median_dose_auc_nll],
            ]),
            fmt="o",
            color="#171717",
            markerfacecolor="white",
            markeredgewidth=0.9,
            markersize=4.4,
            linewidth=1.1,
            capsize=2.3,
            zorder=5,
        )
    limit = max(abs(np.nanpercentile(all_values, 1)), abs(np.nanpercentile(all_values, 99))) * 1.18
    limit = max(limit, 0.004)
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color="#555555", linewidth=0.8, zorder=0)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Future-contact prediction loss\n(dose AUC, nats/decision)")
    ax.tick_params(axis="x", length=0, pad=5)
    ax.spines[["top", "right"]].set_visible(False)


def rank_sensitivity(ax: plt.Axes, inference: pd.DataFrame) -> None:
    metrics = (
        "SHARED",
        "SHARED_MINUS_C_SUFFIX",
        "SHARED_MINUS_PCA",
    )
    labels = (
        "Shared response",
        "vs shuffled endings",
        "vs high-variance",
    )
    offsets = (-0.18, 0.0, 0.18)
    for metric, label, offset in zip(metrics, labels, offsets):
        part = inference[inference.metric.eq(metric)].sort_values("rank")
        x = part["rank"].to_numpy(float) + offset
        y = part["median_dose_auc_nll"].to_numpy(float)
        low = part["ci95_low"].to_numpy(float)
        high = part["ci95_high"].to_numpy(float)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - low, high - y]),
            fmt="o-",
            color=COLORS[metric],
            markerfacecolor="white",
            markeredgewidth=1.0,
            markersize=4.3,
            linewidth=1.2,
            capsize=2.2,
            label=label,
        )
    ax.axhline(0, color="#555555", linewidth=0.8, zorder=0)
    ax.set_xticks((1, 2, 3))
    ax.set_xlim(0.55, 3.45)
    ax.set_xlabel("Shared response directions removed")
    ax.set_ylabel("Future-contact prediction loss\n(dose AUC, nats/decision)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        handlelength=1.7,
        borderaxespad=0,
        ncol=3,
    )


def main() -> None:
    set_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    patient = pd.read_csv(RESULT / "PATIENT_AUC_EFFECTS.csv")
    primary = pd.read_csv(RESULT / "PRIMARY_INFERENCE.csv")
    subspace = pd.read_csv(RESULT / "SUBSPACE_INFERENCE.csv")

    patient[
        patient.phase.eq("ALL")
        & patient.metric.isin(("SHARED", "SHARED_MINUS_ORTHOGONAL", "SHARED_MINUS_C_SUFFIX"))
    ].to_csv(SOURCE / f"{STEM}_panel_a.csv", index=False)
    subspace.to_csv(SOURCE / f"{STEM}_panel_b.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(7.20, 3.05), constrained_layout=False)
    scatter_distribution(axes[0], patient, primary)
    rank_sensitivity(axes[1], subspace)
    for label, ax in zip(("a", "b"), axes):
        x_position, y_position = ((-0.18, 1.06) if label == "a" else (0.01, 0.99))
        ax.text(
            x_position,
            y_position,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
        )
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.20, top=0.78, wspace=0.40)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(
            FIGURES / f"{STEM}.{extension}",
            dpi=400 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)

    main_hashes = {
        extension: sha256(FIGURES / f"{MAIN_STEM}.{extension}")
        for extension in ("png", "pdf", "svg")
    }
    output_hashes = {
        extension: sha256(FIGURES / f"{STEM}.{extension}")
        for extension in ("png", "pdf", "svg")
    }
    decision = {
        "contract": "topic5_figure6_shared_component_necessity_decision_v0_2",
        "main_figure_changed": False,
        "main_panel_eligible": False,
        "main_figure_hashes": main_hashes,
        "supplementary_figure": f"figures/{STEM}.png",
        "supplementary_hashes": output_hashes,
        "reason": (
            "Deleting the leave-one-network shared response direction, or its first "
            "three directions, did not selectively worsen held-out future-contact prediction."
        ),
    }
    (FIGURES / "FIGURE6_NECESSITY_DECISION.json").write_text(
        json.dumps(decision, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
