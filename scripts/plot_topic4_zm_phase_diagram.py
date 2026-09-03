#!/usr/bin/env python3
"""Render the empirical spatial Z/M SNN branch and phase screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle


COLORS = {
    "low_start": "#35618D",
    "high_start": "#C94C4C",
    "LOW_MONOSTABLE_CANDIDATE": "#7FA6C9",
    "BISTABLE_CANDIDATE": "#8564A8",
    "HIGH_MONOSTABLE_CANDIDATE": "#D8756B",
    "MIXED_OR_UNRESOLVED": "#B8B8B8",
    "REVERSE_SPLIT": "#E0B44C",
}
ORDER = [
    "LOW_MONOSTABLE_CANDIDATE",
    "BISTABLE_CANDIDATE",
    "HIGH_MONOSTABLE_CANDIDATE",
    "MIXED_OR_UNRESOLVED",
    "REVERSE_SPLIT",
]
SHORT = {
    "LOW_MONOSTABLE_CANDIDATE": "low",
    "BISTABLE_CANDIDATE": "two-state",
    "HIGH_MONOSTABLE_CANDIDATE": "tonic high",
    "MIXED_OR_UNRESOLVED": "unresolved",
    "REVERSE_SPLIT": "reverse split",
}
BRANCH_SHORT = {
    "LOW": "L",
    "INTERMEDIATE": "I",
    "TONIC_HIGH": "H",
    "UNSTABLE": "U",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _panel_letter(ax, value, *, x=-0.14):
    ax.text(x, 1.08, value, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="left")


def _families(pairs):
    grouped = defaultdict(list)
    for row in pairs:
        grouped[(float(row["q_clamp"]), float(row["eta_m"]))].append(row)
    return grouped


def _majority_label(rows):
    counts = Counter(row["pair_label"] for row in rows)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "MIXED_OR_UNRESOLVED", counts
    return top[0][0], counts


def render(payload, output_stem, *, title=None):
    pairs = payload["pairs"]
    if not pairs:
        raise ValueError("aggregate has no paired phase points")
    families = _families(pairs)
    q_values = sorted({key[0] for key in families}, reverse=True)
    eta_values = sorted({key[1] for key in families})
    d_values = np.asarray([1.0 - q for q in q_values], float)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure = plt.figure(figsize=(8.1, 5.25))
    grid = figure.add_gridspec(
        2, 2, height_ratios=(2.15, 0.95), hspace=0.42, wspace=0.34,
        left=0.09, right=0.98, bottom=0.27, top=0.82)
    ax_rate = figure.add_subplot(grid[0, 0])
    ax_active = figure.add_subplot(grid[0, 1], sharex=ax_rate)
    ax_phase = figure.add_subplot(grid[1, :])

    # Primary plots show every paired observation lightly, with the across-seed
    # median overlaid. Lines only connect sampled values and are not analytic
    # stable/unstable continuation branches.
    branch_summaries = []
    for eta_index, eta_m in enumerate(eta_values):
        selected = [row for row in pairs if float(row["eta_m"]) == eta_m]
        alpha = 0.25 if len(eta_values) > 1 else 0.36
        for row in selected:
            d = 1.0 - float(row["q_clamp"])
            ax_rate.plot(
                [d, d], [row["low_median_rate_hz"], row["high_median_rate_hz"]],
                color="#B8B8B8", lw=0.65, alpha=0.45, zorder=1)
            ax_rate.scatter(d, row["low_median_rate_hz"], s=17,
                            color=COLORS["low_start"], alpha=alpha, zorder=2)
            ax_rate.scatter(d, row["high_median_rate_hz"], s=20, marker="s",
                            color=COLORS["high_start"], alpha=alpha, zorder=2)
            ax_active.plot(
                [d, d], [row["low_active_fraction"], row["high_active_fraction"]],
                color="#B8B8B8", lw=0.65, alpha=0.45, zorder=1)
            ax_active.scatter(d, row["low_active_fraction"], s=17,
                              color=COLORS["low_start"], alpha=alpha, zorder=2)
            ax_active.scatter(d, row["high_active_fraction"], s=20, marker="s",
                              color=COLORS["high_start"], alpha=alpha, zorder=2)

        low_rate, high_rate, low_active, high_active = [], [], [], []
        available_d = []
        for q, d in zip(q_values, d_values):
            rows = families.get((q, eta_m), [])
            if not rows:
                continue
            available_d.append(d)
            low_rate.append(np.median([row["low_median_rate_hz"] for row in rows]))
            high_rate.append(np.median([row["high_median_rate_hz"] for row in rows]))
            low_active.append(np.median([row["low_active_fraction"] for row in rows]))
            high_active.append(np.median([row["high_active_fraction"] for row in rows]))
        line_alpha = 1.0 if len(eta_values) == 1 else 0.45 + 0.45 * (
            eta_index / max(len(eta_values) - 1, 1))
        ax_rate.plot(available_d, low_rate, "o-", color=COLORS["low_start"],
                     ms=4.2, lw=1.45, alpha=line_alpha,
                     label="low-start" if eta_index == 0 else None, zorder=3)
        ax_rate.plot(available_d, high_rate, "s-", color=COLORS["high_start"],
                     ms=4.0, lw=1.45, alpha=line_alpha,
                     label="high-start" if eta_index == 0 else None, zorder=3)
        ax_active.plot(available_d, low_active, "o-", color=COLORS["low_start"],
                       ms=4.2, lw=1.45, alpha=line_alpha, zorder=3)
        ax_active.plot(available_d, high_active, "s-", color=COLORS["high_start"],
                       ms=4.0, lw=1.45, alpha=line_alpha, zorder=3)
        branch_summaries.append({
            "eta_m": eta_m,
            "D": list(map(float, available_d)),
            "low_start_median_rate_hz": list(map(float, low_rate)),
            "high_start_median_rate_hz": list(map(float, high_rate)),
            "low_start_median_active_fraction": list(map(float, low_active)),
            "high_start_median_active_fraction": list(map(float, high_active)),
        })

    ax_rate.axhspan(0, 80, color="#DCEAF4", alpha=0.55, lw=0)
    ax_rate.axhspan(300, 500, color="#F5DEDE", alpha=0.48, lw=0)
    ax_rate.axhline(80, color="#7596B2", ls="--", lw=0.8)
    ax_rate.axhline(300, color="#C87A7A", ls="--", lw=0.8)
    ax_rate.set_ylim(0, 510)
    ax_rate.set_ylabel("Population E rate (Hz)")
    ax_rate.set_title("Stationary firing-rate branch", loc="left", fontweight="bold")
    ax_rate.legend(frameon=False, loc="upper left", ncol=2,
                   handlelength=1.7, columnspacing=1.1)
    _panel_letter(ax_rate, "A")

    ax_active.axhspan(0, 0.5, color="#DCEAF4", alpha=0.55, lw=0)
    ax_active.axhspan(0.85, 1.0, color="#F5DEDE", alpha=0.48, lw=0)
    ax_active.axhline(0.5, color="#7596B2", ls="--", lw=0.8)
    ax_active.axhline(0.85, color="#C87A7A", ls="--", lw=0.8)
    ax_active.set_ylim(0, 1.03)
    ax_active.set_ylabel("Active E neurons (20-ms fraction)")
    ax_active.set_title("Global recruitment branch", loc="left", fontweight="bold")
    _panel_letter(ax_active, "B")

    if len(d_values) == 1:
        d_edges = np.asarray([d_values[0] - 0.005, d_values[0] + 0.005])
    else:
        mids = 0.5 * (d_values[:-1] + d_values[1:])
        d_edges = np.r_[d_values[0] - (mids[0] - d_values[0]), mids,
                        d_values[-1] + (d_values[-1] - mids[-1])]
    if len(eta_values) == 1:
        eta_edges = np.asarray([eta_values[0] - 0.005, eta_values[0] + 0.005])
    else:
        eta_array = np.asarray(eta_values, float)
        mids = 0.5 * (eta_array[:-1] + eta_array[1:])
        eta_edges = np.r_[eta_array[0] - (mids[0] - eta_array[0]), mids,
                          eta_array[-1] + (eta_array[-1] - mids[-1])]
    # Missing cells are not scientific outcomes.  Keep them masked/hatched
    # rather than silently painting them as MIXED_OR_UNRESOLVED.
    matrix = np.full((len(eta_values), len(q_values)), np.nan, float)
    outcome_audit = []
    cell_labels = []
    missing_cells = []
    for yi, eta_m in enumerate(eta_values):
        for xi, q in enumerate(q_values):
            rows = families.get((q, eta_m), [])
            if not rows:
                missing_cells.append((xi, yi, q, eta_m))
                outcome_audit.append({
                    "q_clamp": q, "D": 1.0 - q, "eta_m": eta_m,
                    "coverage": "missing", "n_seeds": 0,
                })
                continue
            label, counts = _majority_label(rows)
            matrix[yi, xi] = ORDER.index(label)
            branch_pairs = Counter(
                (row["low_start_label"], row["high_start_label"])
                for row in rows)
            common_branch_pair = branch_pairs.most_common(1)[0][0]
            cell_labels.append(
                (q, eta_m, label, len(rows), common_branch_pair))
            outcome_audit.append({
                "q_clamp": q, "D": 1.0 - q, "eta_m": eta_m,
                "majority_pair_label": label, "counts": dict(counts),
                "n_seeds": len(rows),
            })
    cmap = ListedColormap([COLORS[name] for name in ORDER])
    cmap.set_bad("white")
    ax_phase.pcolormesh(d_edges, eta_edges, matrix, cmap=cmap,
                        vmin=-0.5, vmax=len(ORDER) - 0.5, shading="flat")
    for xi, yi, _q, _eta_m in missing_cells:
        ax_phase.add_patch(Rectangle(
            (d_edges[xi], eta_edges[yi]),
            d_edges[xi + 1] - d_edges[xi],
            eta_edges[yi + 1] - eta_edges[yi],
            facecolor="white", edgecolor="#D6D6D6", linewidth=0.35,
            hatch="////", zorder=2))
    for q, eta_m, label, n_rows, branch_pair in cell_labels:
        branch_text = "/".join(BRANCH_SHORT.get(value, "?")
                               for value in branch_pair)
        ax_phase.text(1.0 - q, eta_m,
                      f"{branch_text}\nn={n_rows}",
                      ha="center", va="center", fontsize=7.2,
                      color="white" if label in {
                          "BISTABLE_CANDIDATE", "HIGH_MONOSTABLE_CANDIDATE"
                      } else "#202020")
    ax_phase.set_ylabel(r"Adaptation $\eta_m$")
    ax_phase.set_xlabel(
        r"Disinhibition $D = 1-q_{clamp}$  (ticks label $q_{clamp}$)")
    ax_phase.set_title("Paired-initial-state outcome", loc="left", fontweight="bold")
    ax_phase.set_yticks(eta_values)
    ax_phase.set_xticks(d_values)
    ax_phase.set_xticklabels([f"q={q:.3f}" for q in q_values])
    if len(d_values) > 1 and float(np.min(np.diff(d_values))) < 0.012:
        plt.setp(ax_phase.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    ax_phase.text(
        1.0, 1.03, "L / I / H = low / intermediate / tonic-high",
        transform=ax_phase.transAxes, ha="right", va="bottom",
        fontsize=6.7, color="#555555")
    _panel_letter(ax_phase, "C", x=-0.075)

    legend_labels = [
        ("LOW_MONOSTABLE_CANDIDATE", "low from both starts"),
        ("BISTABLE_CANDIDATE", "low/high split"),
        ("HIGH_MONOSTABLE_CANDIDATE", "tonic high from both starts"),
        ("MIXED_OR_UNRESOLVED", "mixed or unresolved"),
        ("REVERSE_SPLIT", "reverse split (audit fail)"),
    ]
    figure.legend(
        handles=[Patch(facecolor=COLORS[key], label=label)
                 for key, label in legend_labels],
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.018),
        ncol=3, columnspacing=1.2, handlelength=1.0, fontsize=6.8)

    for ax in (ax_rate, ax_active):
        ax.set_xlim(d_edges[0], d_edges[-1])
        ax.set_xticks(d_values)
        ax.set_xticklabels([])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D7D7D7", lw=0.5, alpha=0.65)
    ax_phase.set_xlim(d_edges[0], d_edges[-1])
    n_seeds = sorted({int(row["noise_seed"]) for row in pairs})
    main_title = title or "Empirical spatial Z/M SNN branch screen"
    figure.suptitle(main_title, x=0.09, ha="left", fontsize=12,
                    fontweight="bold", y=0.975)
    figure.text(
        0.09, 0.91,
        f"paired future noise; {len(n_seeds)} seed{'s' if len(n_seeds) != 1 else ''}; "
        "finite stochastic network (not an analytic bifurcation diagram)",
        ha="left", va="center", fontsize=8.2, color="#555555")

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output_stem.with_suffix(f".{extension}"), dpi=350,
                       facecolor="white")
    plt.close(figure)
    return {
        "status": "SPATIAL_ZM_EMPIRICAL_PHASE_FIGURE_COMPLETE",
        "source_aggregate_status": payload["status"],
        "source_scientific_contract_sha256": payload.get(
            "scientific_contract_sha256", payload["phase_config_sha256"]),
        "source_phase_config_sha256": payload["phase_config_sha256"],
        "n_pairs": len(pairs),
        "n_missing_phase_cells": len(missing_cells),
        "noise_seeds": n_seeds,
        "branch_summaries": branch_summaries,
        "outcome_audit": outcome_audit,
        "outputs": {
            extension: {
                "path": str(output_stem.with_suffix(f".{extension}").resolve()),
                "sha256": _sha256(output_stem.with_suffix(f".{extension}")),
            } for extension in ("png", "pdf", "svg")
        },
        "claim_boundary": (
            "Observed low/high initial-condition endpoints in a finite stochastic "
            "SNN. Lines connect sampled medians; no unstable analytic branch is shown."),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--out", required=True, help="Output stem without extension")
    parser.add_argument("--title")
    args = parser.parse_args()
    aggregate = Path(args.aggregate).resolve()
    payload = json.loads(aggregate.read_text())
    if payload.get("status") != "SPATIAL_ZM_PHASE_DIAGRAM_AGGREGATED":
        raise RuntimeError("input is not a spatial Z/M phase aggregate")
    metadata = render(payload, args.out, title=args.title)
    metadata["source_aggregate"] = {
        "path": str(aggregate), "sha256": _sha256(aggregate)}
    _atomic_json(metadata, Path(args.out).with_suffix(".metadata.json"))
    print(json.dumps({
        "status": metadata["status"],
        "png": metadata["outputs"]["png"]["path"],
        "n_pairs": metadata["n_pairs"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
