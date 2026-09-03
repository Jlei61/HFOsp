#!/usr/bin/env python3
"""Render a direct, three-panel validation of the frozen-q saddle-node."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#355C85"
PURPLE = "#76528F"
RED = "#C84A4A"
GOLD = "#C48A2B"
GREY = "#62676C"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _panel(ax, letter):
    ax.text(-0.15, 1.07, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", ha="left", va="bottom")


def render(continuation, audit, arrays, output_stem):
    output_stem = Path(output_stem).resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    target = min(
        continuation["families"], key=lambda row: abs(row["eta_m"] - 0.02))
    regular = target["regular_high_branch"]
    arc = target["arc"]
    fold_q = float(audit["fold"]["q_from_eigenvalue_zero"])
    fold_rate = float(audit["fold"]["mean_rate_e_hz_at_eigenvalue_zero"])
    probe = audit["two_root_probe"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.4,
        "axes.titlesize": 9.4, "axes.labelsize": 8.9,
        "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
        "legend.fontsize": 7.2, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.25))
    ax_full, ax_zoom, ax_eigen = axes
    figure.subplots_adjust(
        left=0.075, right=0.985, bottom=0.24, top=0.79, wspace=0.37)

    # A: the full fixed-point geometry. No stability labels are assigned here.
    regular_q = np.asarray([row["q"] for row in regular])
    regular_rate = np.asarray([row["mean_rate_e_hz"] for row in regular])
    arc_q = np.asarray([row["q"] for row in arc])
    arc_rate = np.asarray([row["mean_rate_e_hz"] for row in arc])
    turn_index = int(np.argmax(arc_q))
    ax_full.plot(regular_q, regular_rate, color=PURPLE, lw=2.0,
                 label="pre-turn branch")
    ax_full.plot(arc_q[:turn_index + 1], arc_rate[:turn_index + 1],
                 color=PURPLE, lw=2.0)
    ax_full.plot(arc_q[turn_index:], arc_rate[turn_index:],
                 color=GOLD, lw=2.0, label="returned branch")
    low_rate = float(target["low_anchor_at_fold_q"]["mean_rate_e_hz"])
    ax_full.plot([0.775, 0.90], [low_rate, low_rate], color=BLUE, lw=1.5,
                 label="near-silent root")
    ax_full.scatter(fold_q, fold_rate, s=75, marker="*", color=RED,
                    edgecolor="white", linewidth=0.6, zorder=5,
                    label="saddle-node")
    ax_full.axvspan(0.800, 0.825, color="#D5D5D5", alpha=0.48, lw=0)
    ax_full.text(0.8125, 375, "OU-SNN\nedge", ha="center", va="top",
                 fontsize=7.0, color=GREY)
    ax_full.annotate("branch turns back", xy=(fold_q, fold_rate),
                     xytext=(0.852, 235), fontsize=7.2, color=RED,
                     arrowprops={"arrowstyle": "->", "color": RED,
                                 "lw": 0.9})
    ax_full.set(xlim=(0.77, 0.90), ylim=(-8, 420),
                xlabel=r"Frozen inhibitory efficacy $q$",
                ylabel="Mean E rate (Hz)")
    ax_full.set_title("Full fixed-point branch", loc="left", fontweight="bold")
    ax_full.legend(frameon=False, loc="lower left", handlelength=1.5,
                   labelspacing=0.35)
    _panel(ax_full, "A")

    # B: intuitive local fold geometry and the two distinct roots at one q.
    q_micro = arrays["micro_q"]
    rate_micro = arrays["micro_rate_e_hz"]
    q_fine = np.asarray([row["q"] for row in target["fold_refinement"]])
    rate_fine = np.asarray(
        [row["mean_rate_e_hz"] for row in target["fold_refinement"]])
    q_local = np.r_[arc_q[arc_rate < rate_fine.min()], q_fine]
    rate_local = np.r_[arc_rate[arc_rate < rate_fine.min()], rate_fine]
    local_order = np.argsort(rate_local)[::-1]
    q_local = q_local[local_order]
    rate_local = rate_local[local_order]
    local_turn = int(np.argmin(np.abs(rate_local - fold_rate)))
    x_local = 1e4 * (q_local - fold_q)
    ax_zoom.plot(x_local[:local_turn + 1], rate_local[:local_turn + 1],
                 "o-", color=PURPLE, ms=2.6, lw=1.35)
    ax_zoom.plot(x_local[local_turn:], rate_local[local_turn:],
                 "o-", color=GOLD, ms=2.6, lw=1.35)
    micro_turn = int(np.argmax(q_micro))
    x_micro = 1e4 * (q_micro - fold_q)
    ax_zoom.plot(x_micro[:micro_turn + 1], rate_micro[:micro_turn + 1],
                 color=PURPLE, lw=1.7)
    ax_zoom.plot(x_micro[micro_turn:], rate_micro[micro_turn:],
                 color=GOLD, lw=1.7)
    upper = float(probe["upper_root"]["mean_rate_e_hz"])
    lower = float(probe["returned_root"]["mean_rate_e_hz"])
    probe_q = float(probe["q"])
    probe_x = 1e4 * (probe_q - fold_q)
    ax_zoom.plot([probe_x, probe_x], [lower, upper], color="#777777",
                 lw=0.8, ls="--")
    ax_zoom.scatter([probe_x], [upper], s=36, color=PURPLE,
                    edgecolor="white", linewidth=0.5, zorder=5)
    ax_zoom.scatter([probe_x], [lower], s=36, color=GOLD,
                    edgecolor="white", linewidth=0.5, zorder=5)
    ax_zoom.scatter(0.0, fold_rate, s=90, marker="*", color=RED,
                    edgecolor="white", linewidth=0.6, zorder=6)
    ax_zoom.text(probe_x - 0.12, upper + 0.8, "root 1", ha="right",
                 color=PURPLE, fontsize=7.1)
    ax_zoom.text(probe_x - 0.12, lower - 0.8, "root 2", ha="right",
                 va="top", color=GOLD, fontsize=7.1)
    ax_zoom.annotate("merge", xy=(0.0, fold_rate),
                     xytext=(-0.55, fold_rate - 0.4),
                     ha="right", fontsize=7.2, color=RED,
                     arrowprops={"arrowstyle": "->", "color": RED,
                                 "lw": 0.9})
    ax_zoom.set(xlim=(-2.75, 0.35), ylim=(122.5, 132.0),
                xlabel=r"Distance from fold $(q-q_{SN})\times10^4$",
                ylabel="Mean E rate (Hz)")
    ax_zoom.set_title("Fold zoom: same q, two fixed points", loc="left",
                      fontweight="bold")
    _panel(ax_zoom, "B")

    # C: the same ordered arclength points carry a simple real zero mode.
    eigen = arrays["micro_eigen_real"]
    order = np.argsort(rate_micro)
    ax_eigen.plot(rate_micro[order], eigen[order], "o-", color=RED,
                  ms=3.0, lw=1.35)
    ax_eigen.axhline(0.0, color="#333333", lw=0.8, ls="--")
    ax_eigen.axvline(fold_rate, color=PURPLE, lw=0.9, ls=":")
    ax_eigen.scatter([fold_rate], [0.0], marker="*", s=75, color=RED,
                     edgecolor="white", linewidth=0.5, zorder=5)
    ax_eigen.set(xlabel="Mean E rate along branch (Hz)",
                 ylabel=r"Real eigenvalue nearest zero")
    ax_eigen.set_title("Jacobian zero mode", loc="left", fontweight="bold")
    normal = audit["normal_form"]
    closest = audit["closest_corrected_fixed_point"]
    ax_eigen.text(
        0.04, 0.96,
        "fixed-point residual " + rf"$={closest['residual_inf']:.1e}$" + "\n"
        + "simple mode: " + rf"$|\lambda_1|={abs(closest['nearest_eigenvalue']['real']):.1e}$" + "\n"
        + rf"$|\lambda_2|={closest['second_mode_magnitude']:.3f}$" + "\n"
        + rf"$|w^T F_q|={normal['transversality_abs']:.2f}\ne 0$" + "\n"
        + rf"$|\frac{{1}}{{2}}w^T F_{{xx}}[v,v]|={normal['quadratic_abs']:.3f}\ne 0$",
        transform=ax_eigen.transAxes, ha="left", va="top", fontsize=7.0,
        color=GREY,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white",
              "ec": "#D0D0D0", "lw": 0.6})
    _panel(ax_eigen, "C")

    figure.suptitle(
        "Patient-matched spatial Z/M: saddle-node validation",
        x=0.075, ha="left", fontsize=11.3, fontweight="bold")
    figure.text(
        0.075, 0.075,
        r"Deterministic frozen-$q$ fast subsystem, 1-mm coarse grid, "
        r"$\eta_m=0.02$.  The gray OU-SNN edge is an empirical comparison, "
        "not the same control parameter boundary.",
        ha="left", va="bottom", fontsize=7.3, color=GREY)

    outputs = {}
    for extension in ("png", "pdf", "svg"):
        path = output_stem.with_suffix("." + extension)
        figure.savefig(path, dpi=300 if extension == "png" else None,
                       bbox_inches="tight", facecolor="white")
        outputs[extension] = {"path": str(path), "sha256": _sha256(path)}
    plt.close(figure)
    metadata = {
        "status": "PATIENT_ZM_SADDLE_NODE_VALIDATION_FIGURE_COMPLETE",
        "panel_semantics": {
            "A": "complete fixed-point geometry including the arclength return",
            "B": "two fold-participating roots at one q coalescing at the turn",
            "C": "simple real fixed-point Jacobian eigenvalue crossing zero plus normal-form audit",
        },
        "source_sha256": {
            "continuation": _sha256(continuation["_source_path"]),
            "audit_json": _sha256(audit["_source_path"]),
            "audit_npz": _sha256(audit["arrays"]["path"]),
        },
        "claim_boundary": audit["boundary"],
        "outputs": outputs,
    }
    _atomic_json(metadata, output_stem.with_suffix(".metadata.json"))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continuation",
        default=("/data/hfosp_topic4_fig45_artifacts/fig5/"
                 "data_driven_zm_phase_diagram/deterministic_meanfield/"
                 "patient_zm_bifurcation_ngrid20.json"))
    parser.add_argument(
        "--audit",
        default=("/data/hfosp_topic4_fig45_artifacts/fig5/"
                 "data_driven_zm_phase_diagram/deterministic_meanfield/"
                 "patient_zm_saddle_node_validation_ngrid20.json"))
    parser.add_argument("--output-stem", default=None)
    args = parser.parse_args()
    continuation_path = Path(args.continuation).resolve()
    audit_path = Path(args.audit).resolve()
    continuation = json.loads(continuation_path.read_text())
    audit = json.loads(audit_path.read_text())
    continuation["_source_path"] = str(continuation_path)
    audit["_source_path"] = str(audit_path)
    arrays = np.load(audit["arrays"]["path"])
    output_stem = (Path(args.output_stem).resolve() if args.output_stem else
                   continuation_path.parent.parent / "figures" /
                   "patient_zm_saddle_node_validation")
    print(json.dumps(
        render(continuation, audit, arrays, output_stem), indent=2))


if __name__ == "__main__":
    main()
