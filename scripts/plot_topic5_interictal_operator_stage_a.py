#!/usr/bin/env python3
"""Plot patient-level Stage-A gains for an engineering or formal gate summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "strongest static / first-order": "#4C78A8",
    "rank-shuffle": "#E07B39",
}


def _draw_metric(
    ax: plt.Axes,
    subject: pd.DataFrame,
    *,
    static_column: str,
    shuffle_column: str,
    ylabel: str,
) -> None:
    labels = list(COLORS)
    columns = [static_column, shuffle_column]
    x_positions = np.arange(len(labels), dtype=float)
    jitter = np.linspace(-0.09, 0.09, max(len(subject), 1))
    for patient_index, (_, row) in enumerate(subject.iterrows()):
        values = np.asarray([row[column] for column in columns], float)
        ax.plot(
            x_positions + jitter[patient_index],
            values,
            color="#B8B8B8",
            linewidth=0.65,
            alpha=0.65,
            zorder=1,
        )
        for x, value, label in zip(x_positions, values, labels):
            ax.scatter(
                x + jitter[patient_index],
                value,
                s=29,
                color=COLORS[label],
                edgecolor="white",
                linewidth=0.45,
                zorder=2,
            )
    for x, column, label in zip(x_positions, columns, labels):
        values = subject[column].to_numpy(float)
        values = values[np.isfinite(values)]
        if values.size:
            median = float(np.median(values))
            low, high = np.quantile(values, [0.25, 0.75])
            ax.errorbar(
                x,
                median,
                yerr=np.asarray([[median - low], [high - median]]),
                fmt="_",
                markersize=16,
                markeredgewidth=2.2,
                color="#202020",
                capsize=3,
                linewidth=1.4,
                zorder=3,
            )
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xticks(x_positions, ["Static / first-order", "Rank-shuffle"])
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.24)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--status-label", default="Engineering pilot")
    args = parser.parse_args()

    cell = pd.read_csv(args.cell_metrics)
    gain_columns = [
        "next_gain_vs_static",
        "suffix_gain_vs_static",
        "next_gain_vs_rank_shuffle",
        "suffix_gain_vs_rank_shuffle",
    ]
    missing = sorted(set(gain_columns + ["subject", "seed"]) - set(cell.columns))
    if missing:
        raise ValueError(f"missing Stage-A columns: {missing}")
    subject = (
        cell.groupby("subject", as_index=False)[gain_columns]
        .median()
        .sort_values("subject")
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.20,
        top=0.76,
        wspace=0.34,
    )
    _draw_metric(
        axes[0],
        subject,
        static_column="next_gain_vs_static",
        shuffle_column="next_gain_vs_rank_shuffle",
        ylabel="Next-set NLL gain\n(positive = GRU better)",
    )
    _draw_metric(
        axes[1],
        subject,
        static_column="suffix_gain_vs_static",
        shuffle_column="suffix_gain_vs_rank_shuffle",
        ylabel="Suffix concordance gain\n(positive = GRU better)",
    )
    axes[0].set_title("Next recruitment set", loc="left", fontweight="bold")
    axes[1].set_title("Remaining recruitment order", loc="left", fontweight="bold")
    figure.suptitle(
        f"{args.status_label} | n={len(subject)}, seeds={cell.seed.nunique()}",
        fontsize=8.5,
        y=0.965,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / "stage_a_event_dynamics_gain"
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    payload = {
        "status_label": args.status_label,
        "n_patients": int(subject.subject.nunique()),
        "n_seeds": int(cell.seed.nunique()),
        "patient_median": {
            column: float(np.nanmedian(subject[column])) for column in gain_columns
        },
        "positive_patient_count": {
            column: int(np.sum(subject[column] > 0)) for column in gain_columns
        },
        "source": str(args.cell_metrics),
        "formal_gate_claim": (
            False
            if any(
                token in args.status_label.lower()
                for token in ("pilot", "engineering", "screen")
            )
            else None
        ),
    }
    (args.out_dir / "stage_a_event_dynamics_gain.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
