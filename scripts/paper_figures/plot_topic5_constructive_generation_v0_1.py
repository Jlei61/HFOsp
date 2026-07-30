#!/usr/bin/env python3
"""Plot the paper-ready constructive event-generation sufficiency audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FULL = "#2B6CB0"
STATIC = "#A0AEC0"
TERM = "#C05621"
MODE = "#6B46C1"
AXIS = "#2F855A"
INK = "#202124"


def _benefit(
    patient: pd.DataFrame,
    metric: str,
    reference: str,
    *,
    higher_is_better: bool = False,
    eligible: bool = False,
) -> np.ndarray:
    frame = patient
    if eligible:
        frame = frame[frame.global_readback_eligible.astype(bool)]
    pivot = frame.pivot(index="subject", columns="condition", values=metric)
    left = pivot["full_constructive"].to_numpy(float)
    right = pivot[reference].to_numpy(float)
    valid = np.isfinite(left) & np.isfinite(right)
    return (left[valid] - right[valid]) if higher_is_better else (
        right[valid] - left[valid]
    )


def _strip(ax, values, x, color, seed):
    values = np.asarray(values, dtype=float)
    jitter = np.random.default_rng(seed).uniform(-0.10, 0.10, values.size)
    ax.scatter(
        np.full(values.size, x) + jitter,
        values,
        s=20,
        color=color,
        alpha=0.70,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )
    if values.size:
        median = float(np.nanmedian(values))
        ax.plot([x - 0.22, x + 0.22], [median, median], color=INK, lw=2.2)


def _status_box(ax, gates):
    entries = [
        ("Engineering", gates["gate_a"]["status"], "Engineering passed"),
        ("Local generation", gates["gate_b"]["status"], "Local gate failed"),
        ("Global modes", gates["gate_c"]["status"], "Locked"),
        ("SNN bridge", gates["snn_fingerprint"]["status"], "Locked"),
    ]
    colors = {
        "PASS": "#D7F0E3",
        "FAIL": "#F8D7DA",
        "LOCKED_NOT_EVALUATED": "#ECEFF3",
        "LOCKED_BY_HUMAN_SUFFICIENCY_GATE": "#ECEFF3",
        "OPEN_FOR_INVENTORY": "#D7F0E3",
    }
    ax.axis("off")
    for row, (label, status, display) in enumerate(entries):
        y = 0.82 - row * 0.20
        ax.text(0.02, y, label, ha="left", va="center", fontsize=8.5)
        ax.text(
            0.98,
            y,
            display,
            ha="right",
            va="center",
            fontsize=7.2,
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": colors.get(status, "#ECEFF3"),
                "edgecolor": "none",
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=ROOT
        / "results/topic5_constructive_event_generation/analysis_v0_1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/topic5_constructive_event_generation/figures",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patient = pd.read_csv(args.analysis_root / "patient_condition_metrics.csv")
    inventory = pd.read_csv(args.analysis_root / "readback_subject_inventory.csv")
    gates = json.loads((args.analysis_root / "gate_verdict.json").read_text())

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(10.8, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[0.92, 1.08])
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]

    ax = axes[0]
    ax.axis("off")
    boxes = [
        (0.02, 0.64, 0.27, 0.22, "Static contact\nscaffold", STATIC),
        (0.36, 0.64, 0.27, 0.22, "Short ordered\ntransition", FULL),
        (0.70, 0.64, 0.27, 0.22, "Progress / STOP", TERM),
    ]
    for x, y, width, height, text, color in boxes:
        ax.add_patch(
            mpl.patches.FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.02",
                facecolor=mpl.colors.to_rgba(color, 0.16),
                edgecolor=color,
                linewidth=1.1,
            )
        )
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center")
    ax.text(
        0.50,
        0.43,
        "revealed first rank  →  free-running suffix",
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
    )
    ax.annotate(
        "",
        xy=(0.84, 0.29),
        xytext=(0.16, 0.29),
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.2},
    )
    ax.text(
        0.50,
        0.12,
        "Train80 defines components; A/B, physical axis and ictal data are sealed",
        ha="center",
        va="center",
        fontsize=7.3,
        color="#4A5568",
    )
    ax.set_title("Constructive within-event test", loc="left", weight="bold")

    ax = axes[1]
    rank = _benefit(patient, "suffix_rank_wasserstein", "static_only")
    precedence = _benefit(
        patient,
        "suffix_precedence_correlation",
        "static_only",
        higher_is_better=True,
    )
    transition = _benefit(
        patient,
        "transition_correlation",
        "static_only",
        higher_is_better=True,
    )
    ax.axhline(0, color="#718096", lw=0.8, ls="--")
    _strip(ax, rank, 0, FULL, 1)
    _strip(ax, precedence, 1, "#3182CE", 2)
    _strip(ax, transition, 2, "#2C7A7B", 7)
    ax.set_xticks(
        [0, 1, 2],
        ["Whole-event\nrank error", "Whole-event\nprecedence", "First-order\ntransition"],
    )
    ax.set_ylabel("History benefit\n(positive = better)")
    ax.set_title("Local versus whole-event generation", loc="left", weight="bold")

    ax = axes[2]
    full = patient[patient.condition == "full_constructive"].drop_duplicates(
        "subject"
    )
    counts = full.n_local_endpoints_within_empirical.value_counts().reindex(
        [0, 1, 2, 3], fill_value=0
    )
    ax.bar(counts.index, counts.values, color=[STATIC, "#90CDF4", FULL, "#1A365D"])
    ax.axvline(1.5, color=INK, lw=0.9, ls="--")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Endpoints within human split-half range")
    ax.set_ylabel("Patients")
    ax.set_title("Absolute posterior fidelity", loc="left", weight="bold")

    ax = axes[3]
    constant_length = _benefit(
        patient, "event_length_wasserstein", "constant_stop"
    )
    no_stop_length = _benefit(
        patient, "event_length_wasserstein", "no_termination"
    )
    ax.axhline(0, color="#718096", lw=0.8, ls="--")
    _strip(ax, constant_length, 0, TERM, 3)
    _strip(ax, no_stop_length, 1, "#9C4221", 4)
    ax.set_xticks([0, 1], ["Constant STOP", "No STOP"])
    ax.set_ylabel("Progress-hazard benefit\n(length Wasserstein)")
    ax.set_title("Termination is a separate component", loc="left", weight="bold")

    ax = axes[4]
    template = _benefit(
        patient,
        "template_error",
        "static_only",
        eligible=True,
    )
    axis = _benefit(
        patient,
        "signed_axis_wasserstein",
        "static_only",
        eligible=True,
    )
    ax.axhline(0, color="#718096", lw=0.8, ls="--")
    _strip(ax, template, 0, MODE, 5)
    _strip(ax, axis, 1, AXIS, 6)
    ax.set_xticks([0, 1], ["Two-mode\ntemplates", "Signed physical\naxis"])
    ax.set_ylabel("History fidelity benefit\n(positive = better)")
    ax.set_title(
        f"Global read-back (eligible n={int(inventory.global_readback_eligible.sum())})",
        loc="left",
        weight="bold",
    )

    ax = axes[5]
    _status_box(ax, gates)
    ax.set_title("Pre-registered gate outcome", loc="left", weight="bold")

    letters = "ABCDEF"
    for letter, ax in zip(letters, axes):
        ax.text(
            -0.14,
            1.08,
            letter,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    basename = "topic5_constructive_event_generation_sufficiency_v0_1"
    png = args.output_dir / f"{basename}.png"
    pdf = args.output_dir / f"{basename}.pdf"
    fig.savefig(png, dpi=350, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure": basename,
        "analysis_root": str(args.analysis_root.resolve()),
        "patient_n": int(patient.subject.nunique()),
        "global_readback_eligible_n": int(
            inventory.global_readback_eligible.sum()
        ),
        "gate_a": gates["gate_a"]["status"],
        "gate_b": gates["gate_b"]["status"],
        "gate_c": gates["gate_c"]["status"],
        "ictal_target_read": False,
    }
    (args.output_dir / f"{basename}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
