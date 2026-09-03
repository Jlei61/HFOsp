#!/usr/bin/env python3
"""Plot empirical OU-on endpoints beside the deterministic Z/M skeleton."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#315D88"
RED = "#C84A4A"
PURPLE = "#74518E"
GOLD = "#C28A2C"
GREY = "#666B70"


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
    ax.text(-0.16, 1.08, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", ha="left", va="bottom")


def _hopf_locator(rows, eta_m):
    selected = sorted(
        [row for row in rows if np.isclose(row["eta_m"], eta_m)
         and row.get("branch", "high") == "high"],
        key=lambda row: row["q"])
    for left, right in zip(selected[:-1], selected[1:]):
        lval = left["maximum_real_part_per_ms"]
        rval = right["maximum_real_part_per_ms"]
        if lval <= 0.0 < rval:
            fraction = -lval / (rval - lval)
            return left["q"] + fraction * (right["q"] - left["q"])
    return None


def render(empirical, continuation, output_stem):
    output_stem = Path(output_stem).resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2,
        "axes.titlesize": 9.2, "axes.labelsize": 8.8,
        "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
        "legend.fontsize": 7.2, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(2, 2, figsize=(8.15, 5.55))
    ax_empirical, ax_branch, ax_fold, ax_eigen = axes.flat
    figure.subplots_adjust(
        left=0.09, right=0.98, bottom=0.16, top=0.88,
        wspace=0.34, hspace=0.46)

    # A: actual finite SNN, persistent OU on, paired initial conditions.
    pairs = empirical["pairs"]
    for eta_m, marker_alpha in ((0.02, 1.0), (0.0, 0.72)):
        rows = sorted(
            [row for row in pairs if np.isclose(row["eta_m"], eta_m)],
            key=lambda row: row["q_clamp"])
        if not rows:
            continue
        q = np.asarray([row["q_clamp"] for row in rows])
        low = np.asarray([row["low_median_rate_hz"] for row in rows])
        high = np.asarray([row["high_median_rate_hz"] for row in rows])
        if eta_m:
            ax_empirical.plot(
                q, low, "o-", color=BLUE, lw=1.25, ms=4.0,
                alpha=marker_alpha, label="low-start")
            ax_empirical.plot(
                q, high, "s-", color=RED, lw=1.25, ms=4.0,
                alpha=marker_alpha, label="high-start")
        else:
            ax_empirical.plot(
                q, low, "o", color=BLUE, mfc="white", mew=1.0, ms=5.0,
                alpha=marker_alpha, label=r"q-only probe ($\eta_m=0$)")
            ax_empirical.plot(
                q, high, "s", color=RED, mfc="white", mew=1.0, ms=5.0,
                alpha=marker_alpha)
        for q_value, lo, hi in zip(q, low, high):
            ax_empirical.plot([q_value, q_value], [lo, hi], color="#B8B8B8",
                              lw=0.65, zorder=0)
    ax_empirical.axhspan(0, 80, color="#DCEAF4", alpha=0.55, lw=0)
    ax_empirical.axhspan(300, 500, color="#F4DDDD", alpha=0.5, lw=0)
    ax_empirical.axhline(80, color=BLUE, ls="--", lw=0.7)
    ax_empirical.axhline(300, color=RED, ls="--", lw=0.7)
    ax_empirical.set(xlim=(0.765, 0.90), ylim=(0, 430),
                     xlabel=r"Frozen inhibitory efficacy $q$",
                     ylabel="Population E rate (Hz)")
    ax_empirical.set_title("OU-on SNN: paired finite-time endpoints", loc="left",
                           fontweight="bold")
    ax_empirical.legend(frameon=False, ncol=3, loc="upper center",
                        columnspacing=0.9, handlelength=1.3,
                        bbox_to_anchor=(0.5, 1.0))
    ax_empirical.text(
        0.98, 0.04, "one seed per sampled point; not robust bistability",
        transform=ax_empirical.transAxes, ha="right", va="bottom",
        fontsize=7.1, color=GREY)
    _panel(ax_empirical, "A")

    # B: same graph/threshold fixed-point skeleton, stationary OU mean only.
    colours = {0.0: PURPLE, 0.02: GOLD}
    for family in continuation["families"]:
        eta_m = float(family["eta_m"])
        if eta_m not in colours:
            continue
        colour = colours[eta_m]
        regular = family["regular_high_branch"]
        arc = family["arc"]
        fold = family["fold"]
        ax_branch.plot(
            [row["q"] for row in regular],
            [row["mean_rate_e_hz"] for row in regular],
            color=colour, lw=2.4 if eta_m == 0.0 else 1.45,
            ls="-" if eta_m == 0.0 else "--",
            label=rf"high branch, $\eta_m={eta_m:g}$")
        # Keep only the after-fold segment to avoid overdrawing the regular arm.
        fold_index = int(np.argmax([row["q"] for row in arc]))
        after = arc[fold_index:]
        ax_branch.plot(
            [row["q"] for row in after],
            [row["mean_rate_e_hz"] for row in after],
            color=colour, lw=1.35,
            ls=":" if eta_m == 0.0 else "-.")
        ax_branch.scatter(
            fold["q"], fold["mean_rate_e_hz"], s=42, marker="*",
            color=colour, edgecolor="white", linewidth=0.5, zorder=4)
        low = family["low_anchor_at_fold_q"]["mean_rate_e_hz"]
        ax_branch.plot([0.775, fold["q"]], [low, low], color=colour,
                       lw=0.9, alpha=0.55)
    ax_branch.axvspan(0.800, 0.825, color="#D5D5D5", alpha=0.42, lw=0,
                      label="empirical SNN edge")
    hopf = _hopf_locator(continuation["stability_sensitivity"], 0.0)
    if hopf is not None:
        ax_branch.axvline(hopf, color="#333333", ls=":", lw=1.0)
        ax_branch.text(hopf + 0.0015, 405, "zero-delay\nHopf locator",
                       fontsize=6.9, color="#333333", va="top")
    ax_branch.set(xlim=(0.765, 0.90), ylim=(-8, 430),
                  xlabel=r"Frozen inhibitory efficacy $q$",
                  ylabel="Mean-field E rate (Hz)")
    ax_branch.set_title("Patient-matched deterministic skeleton", loc="left",
                        fontweight="bold")
    ax_branch.legend(frameon=False, loc="lower left", handlelength=1.7,
                     bbox_to_anchor=(0.0, 0.055))
    ax_branch.text(
        0.02, 0.97, "OU mean only",
        transform=ax_branch.transAxes, fontsize=7.0, color=GREY,
        va="top")
    _panel(ax_branch, "B")

    # C: two-dimensional q x M fold locus.
    eta = np.asarray([row["eta_m"] for row in continuation["families"]])
    q_fold = np.asarray([row["fold"]["q"] for row in continuation["families"]])
    order = np.argsort(eta)
    ax_fold.plot(eta[order], q_fold[order], "o-", color=PURPLE,
                 lw=1.6, ms=4.5)
    ax_fold.set_xlabel(r"Adaptation coupling $\eta_m$")
    ax_fold.set_ylabel(r"Fold location $q_\mathrm{fold}$")
    ax_fold.set_ylim(q_fold.min() - 0.00018, q_fold.max() + 0.00018)
    ax_fold.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax_fold.set_title("Fold locus in the frozen-q fast subsystem", loc="left",
                      fontweight="bold")
    ax_fold.text(
        0.03, 0.08,
        rf"$\Delta q_{{fold}}={q_fold.max()-q_fold.min():.1e}$ over tested $\eta_m$",
        transform=ax_fold.transAxes, fontsize=7.3, color=GREY)
    _panel(ax_fold, "C")

    # D: local hard evidence for a saddle-node at eta=0.02.
    target = min(
        continuation["families"], key=lambda row: abs(row["eta_m"] - 0.02))
    fine = target["fold_refinement"]
    rate = np.asarray([row["mean_rate_e_hz"] for row in fine])
    eigen = np.asarray([
        row["fixed_point_eigenvalue_near_zero"]["real"] for row in fine])
    ax_eigen.plot(rate, eigen, "o-", color=RED, ms=3.2, lw=1.25)
    ax_eigen.axhline(0.0, color="#333333", lw=0.8, ls="--")
    fold = target["fold"]
    ax_eigen.axvline(fold["mean_rate_e_hz"], color=PURPLE, lw=0.9, ls=":")
    ax_eigen.set_xlabel("E rate along arclength near fold (Hz)")
    ax_eigen.set_ylabel(r"Jacobian eigenvalue nearest zero")
    ax_eigen.set_title("Fold audit: real zero mode", loc="left",
                       fontweight="bold")
    ax_eigen.text(
        0.03, 0.95,
        rf"$q_{{fold}}={fold['q']:.6f}$, $\eta_m=0.02$",
        transform=ax_eigen.transAxes, va="top", fontsize=7.3, color=GREY)
    _panel(ax_eigen, "D")

    figure.suptitle(
        "Spatial Z/M phase analysis: stochastic endpoints and deterministic fold",
        x=0.09, ha="left", fontsize=11.5, fontweight="bold")
    figure.text(
        0.09, 0.055,
        "The fold is established for the 1-mm coarse deterministic reduction; "
        "it is not a thermodynamic phase transition of the finite OU-driven SNN.",
        ha="left", va="bottom", fontsize=7.3, color=GREY)
    outputs = {}
    for extension in ("png", "pdf", "svg"):
        path = output_stem.with_suffix("." + extension)
        figure.savefig(path, dpi=300 if extension == "png" else None,
                       bbox_inches="tight", facecolor="white")
        outputs[extension] = {"path": str(path), "sha256": _sha256(path)}
    plt.close(figure)
    metadata = {
        "status": "PATIENT_ZM_PHASE_BIFURCATION_FIGURE_COMPLETE",
        "panel_semantics": {
            "A": "finite stochastic SNN endpoints under persistent stationary OU",
            "B": "fixed points of the patient-matched deterministic OU-mean reduction",
            "C": "saddle-node fold locus over adaptation coupling at n_grid=20",
            "D": "real fixed-point Jacobian eigenvalue crossing zero near the fold",
        },
        "claim_boundary": (
            "Panel A cannot label a mathematical bifurcation. Panels B-D establish a "
            "saddle-node only in the reduced deterministic fast subsystem. Neither is "
            "finite-size scaling evidence for a thermodynamic phase transition."),
        "source_sha256": {
            "empirical": _sha256(empirical["_source_path"]),
            "continuation": _sha256(continuation["_source_path"]),
        },
        "outputs": outputs,
    }
    _atomic_json(metadata, output_stem.with_suffix(".metadata.json"))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--empirical", default=("/data/hfosp_topic4_fig45_artifacts/fig5/"
                                "data_driven_zm_phase_diagram/"
                                "stage0_sparse_2d_aggregate.json"))
    parser.add_argument(
        "--continuation", default=("/data/hfosp_topic4_fig45_artifacts/fig5/"
                                   "data_driven_zm_phase_diagram/"
                                   "deterministic_meanfield/"
                                   "patient_zm_bifurcation_ngrid20.json"))
    parser.add_argument("--output-stem", default=None)
    args = parser.parse_args()
    empirical_path = Path(args.empirical).resolve()
    continuation_path = Path(args.continuation).resolve()
    empirical = json.loads(empirical_path.read_text())
    continuation = json.loads(continuation_path.read_text())
    empirical["_source_path"] = str(empirical_path)
    continuation["_source_path"] = str(continuation_path)
    output_stem = (Path(args.output_stem).resolve() if args.output_stem else
                   continuation_path.parent.parent / "figures" /
                   "spatial_zm_phase_bifurcation_analysis")
    print(json.dumps(render(empirical, continuation, output_stem), indent=2))


if __name__ == "__main__":
    main()
