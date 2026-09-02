#!/usr/bin/env python3
"""Render the nine-panel Topic 5.2 latent-landscape closeout candidate.

Panel map follows the claim ladder in order (C1 -> C7).  Every statistical panel
plots the quantity that carries its claim: paired real-minus-order-shuffled
differences where the endpoint is a contrast, and the preregistered orientation
only where a post-hoc reorientation would be an exact sign mirror of the same
numbers.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import atomic_write_csv, atomic_write_json, sha256_file  # noqa: E402


OUT = ROOT / "results" / "topic5_latent_propagation_landscape_v0_2"
SYSTEM = OUT / "system_identification"
TRANSPORT = OUT / "dynamical_transport"
RESPONSE = OUT / "axis_perturbation" / "responses"
DATA = OUT / "spatial_control_field" / "data_alignment"
SPATIAL = OUT / "spatial_control_field"
EARLY = OUT / "early_ictal_exploratory"
FIGURES = OUT / "paper-ready-figure" / "latent_landscape_candidate" / "figures"
SOURCE = FIGURES / "source_data"
STEM = "topic5_latent_landscape_v0_2_candidate"
REAL_ARMS = ("L0", "L1", "L2m", "L3")
BLUE = "#3E6C99"
RED = "#B5544C"
GOLD = "#C99A37"
GRAY = "#727272"
# Spatial-null families need enough patients to read as a distribution; families
# below this denominator stay in the source table and the figure README.
MIN_NULL_FAMILY_PATIENTS = 10


def panel_label(axis: mpl.axes.Axes, label: str) -> None:
    axis.text(-0.20, 1.16, label, transform=axis.transAxes, fontsize=12, fontweight="bold", va="top")


def panel_note(axis: mpl.axes.Axes, text: str) -> None:
    """Explanatory note inside the axes, so it can never collide with the title."""
    axis.text(0.015, 0.985, text, transform=axis.transAxes, ha="left", va="top",
              fontsize=6.2, color="#5A5A5A", zorder=6)


def bootstrap_median_ci(values: np.ndarray, seed: int, draws: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    medians = np.median(values[rng.integers(0, len(values), size=(draws, len(values)))], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def strip_points(
    axis: mpl.axes.Axes,
    data: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    *,
    annotate_n: bool = True,
    rotation: float = 0.0,
) -> None:
    """Subject points, median, and a bootstrap median CI for every strip."""
    rng = np.random.default_rng(5202)
    for position, (values, color) in enumerate(zip(data, colors), start=1):
        finite = np.asarray(values, float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            continue
        jitter = rng.uniform(-0.15, 0.15, len(finite))
        axis.scatter(position + jitter, finite, s=13, color=color, alpha=0.55, edgecolor="none", zorder=2)
        median = float(np.median(finite))
        low, high = bootstrap_median_ci(finite, seed=5202 + position)
        axis.plot([position, position], [low, high], color="black", lw=1.1, zorder=3, solid_capstyle="butt")
        axis.plot([position - 0.24, position + 0.24], [median, median], color="black", lw=1.9, zorder=4)
        if annotate_n:
            axis.annotate(
                f"n={len(finite)}", (position, 0.0), xycoords=("data", "axes fraction"),
                xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                fontsize=5.6, color="#666666",
            )
    axis.axhline(0, color="#A0A0A0", lw=0.8, ls="--", zorder=1)
    axis.set_xticks(range(1, len(labels) + 1), labels,
                    rotation=rotation, ha="right" if rotation else "center")
    axis.set_xlim(0.4, len(labels) + 0.6)


def response_matrix() -> tuple[pd.DataFrame, np.ndarray]:
    values = pd.read_csv(RESPONSE / "C3_CELL_PHASE_RESPONSE.csv")
    metrics = [
        "R_progress_from_progress", "R_field_from_progress",
        "R_progress_from_field", "R_field_from_field",
    ]
    values = values.groupby(
        ["patient", "fit_id", "public_arm", "seed"], as_index=False
    )[metrics].mean()
    values = values.groupby(
        ["patient", "fit_id", "public_arm"], as_index=False
    )[metrics].median()
    values = values.groupby(["patient", "public_arm"], as_index=False)[metrics].median()
    values = values[values["public_arm"].isin(REAL_ARMS)].groupby("patient", as_index=False)[metrics].median()
    matrix = np.asarray([
        [np.nanmedian(values["R_progress_from_progress"]), np.nanmedian(values["R_progress_from_field"])],
        [np.nanmedian(values["R_field_from_progress"]), np.nanmedian(values["R_field_from_field"])],
    ])
    return values, matrix


def main() -> None:
    required = [
        SYSTEM / "C1_PATIENT_EFFECTS.csv", SYSTEM / "C1_EMERGENCE_CURVES.csv",
        TRANSPORT / "C2_PATIENT_EFFECTS.csv", RESPONSE / "C3_CELL_PHASE_RESPONSE.csv",
        RESPONSE / "C4_TOPOLOGY_FIELD_EFFECTS.csv",
        DATA / "C5_PATIENT_EFFECTS.csv", DATA / "PRIMARY_FIT_RESPONSE_FIELDS.csv",
        OUT / "C5_SPATIAL_NULL_FAMILY_PATIENT_EFFECTS.csv",
        OUT / "C5_SMOOTHING_MATCHED_IDENTITY.csv",
        OUT / "SNN_INPUT_ELIGIBILITY.json", EARLY / "EARLY_ICTAL_PER_PATIENT.csv",
        OUT / "CLAIM_LADDER_ADJUDICATION.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"figure inputs missing: {missing}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)

    c1 = pd.read_csv(SYSTEM / "C1_PATIENT_EFFECTS.csv")
    c1 = c1[c1["tier"].eq("generic_all_identifiable")].copy()
    emergence = pd.read_csv(SYSTEM / "C1_EMERGENCE_CURVES.csv")
    emergence = emergence[
        emergence["tier"].eq("generic_all_identifiable")
        & emergence["endpoint"].eq("incremental_r2_oh_minus_o")
    ].copy()
    c2 = pd.read_csv(TRANSPORT / "C2_PATIENT_EFFECTS.csv")
    c2 = c2[c2["tier"].eq("generic_all_identifiable")].copy()
    c3_patient, matrix = response_matrix()
    c4 = pd.read_csv(RESPONSE / "C4_TOPOLOGY_FIELD_EFFECTS.csv")
    c5 = pd.read_csv(DATA / "C5_PATIENT_EFFECTS.csv")
    c5 = c5[c5["tier"].eq("generic_all_identifiable")].copy()
    c5_nulls = pd.read_csv(OUT / "C5_SPATIAL_NULL_FAMILY_PATIENT_EFFECTS.csv")
    c5_nulls = c5_nulls[c5_nulls["tier"].eq("generic_all_identifiable")].copy()
    c5_identity = pd.read_csv(OUT / "C5_SMOOTHING_MATCHED_IDENTITY.csv")
    c5_identity = c5_identity.groupby(["patient", "axis"], as_index=False)[
        "smoothing_matched_identity_margin"
    ].median()
    snn = json.loads((OUT / "SNN_INPUT_ELIGIBILITY.json").read_text())
    early = pd.read_csv(EARLY / "EARLY_ICTAL_PER_PATIENT.csv")

    def c5_null(axis: str, family: str) -> np.ndarray:
        part = c5_nulls[c5_nulls["axis"].eq(axis) & c5_nulls["null_family"].eq(family)]
        return part["preregistered_margin"].to_numpy(float)

    def c5_ident(axis: str) -> np.ndarray:
        return c5_identity[c5_identity["axis"].eq(axis)][
            "smoothing_matched_identity_margin"
        ].to_numpy(float)

    def early_column(axis: str, column: str) -> np.ndarray:
        values = early[early["axis"].eq(axis)][column].to_numpy(float)
        values = values[np.isfinite(values)]
        return values if len(values) >= MIN_NULL_FAMILY_PATIENTS else np.asarray([])

    atomic_write_json(SOURCE / "panel_a_schematic.json", {
        "decoder_state": "q=(h,r,k)",
        "coordinates": ["progress", "future_field"],
        "role": "conceptual schematic; not a data panel",
    })
    atomic_write_csv(SOURCE / "panel_b_geometry.csv", c1)
    atomic_write_csv(SOURCE / "panel_c_emergence.csv", emergence)
    atomic_write_csv(SOURCE / "panel_d_transport.csv", c2)
    atomic_write_csv(SOURCE / "panel_e_response_matrix_patients.csv", c3_patient)
    atomic_write_json(SOURCE / "panel_e_response_matrix.json", {"matrix": matrix.tolist()})
    atomic_write_csv(SOURCE / "panel_f_topology_convergence.csv", c4)
    atomic_write_csv(SOURCE / "panel_g_data_alignment.csv", c5_nulls.merge(
        c5_identity, on=["patient", "axis"], how="outer"
    ))
    atomic_write_json(SOURCE / "panel_h_snn_eligibility.json", snn)
    atomic_write_csv(SOURCE / "panel_i_early_ictal.csv", early)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.0, "axes.titlesize": 9.0,
        "axes.labelsize": 8.0, "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "svg.fonttype": "none",
        "legend.frameon": False,
    })
    figure, axes = plt.subplots(3, 3, figsize=(13.6, 11.2), constrained_layout=True)

    # A: full-state and two-coordinate hypothesis.
    ax = axes[0, 0]
    panel_label(ax, "a")
    ax.set_axis_off()
    ax.annotate("ordered\nprefix", (0.11, 0.80), xycoords="axes fraction", ha="center", va="center",
                fontsize=7.2,
                bbox={"boxstyle": "round,pad=0.30", "fc": "#EEF2F6", "ec": GRAY})
    ax.annotate("frozen decoder state\nq = (h, r, k)", (0.50, 0.80), xycoords="axes fraction",
                ha="center", va="center", fontsize=7.2,
                bbox={"boxstyle": "round,pad=0.34", "fc": "#F5F1E8", "ec": GOLD})
    ax.annotate("future\ncontact field", (0.89, 0.80), xycoords="axes fraction", ha="center", va="center",
                fontsize=7.2,
                bbox={"boxstyle": "round,pad=0.30", "fc": "#EEF2F6", "ec": GRAY})
    ax.annotate("", (0.30, 0.80), (0.20, 0.80), xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.annotate("", (0.79, 0.80), (0.70, 0.80), xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.annotate("progress", (0.34, 0.26), xycoords="axes fraction", ha="center", color=BLUE,
                fontweight="bold", fontsize=7.6)
    ax.annotate("future field", (0.70, 0.26), xycoords="axes fraction", ha="center", color=RED,
                fontweight="bold", fontsize=7.6)
    ax.plot([0.20, 0.50], [0.45, 0.45], color=BLUE, lw=2, transform=ax.transAxes)
    ax.annotate("", (0.54, 0.45), (0.48, 0.45), xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": BLUE})
    ax.plot([0.50, 0.76], [0.45, 0.62], color=RED, lw=2, transform=ax.transAxes)
    ax.plot([0.50, 0.76], [0.45, 0.28], color=RED, lw=2, transform=ax.transAxes)
    ax.annotate("no training and no parameter change;\nevery intervention moves h only",
                (0.5, 0.07), xycoords="axes fraction", ha="center", va="center",
                fontsize=6.6, color="#555555")
    ax.set_title("Frozen recurrent state-control hypothesis")

    # B: C1 held-out geometry.
    ax = axes[0, 1]
    panel_label(ax, "b")
    strip_points(ax, [c1["progress_r2_P_minus_O"], c1["field_r2_PF_minus_P"],
                      c1["field_r2_PF_minus_PF_null"]],
                 ["phase over\nobservables", "future field\nover phase", "future field\nover label-shuffled"],
                 [BLUE, RED, RED])
    ax.set_ylabel(r"held-out $\Delta R^2$")
    ax.set_title("Task-aligned geometry is decodable")

    # C: the endpoint is the paired real-minus-shuffled difference, not two curves.
    ax = axes[0, 2]
    panel_label(ax, "c")
    bins = np.sort(emergence["phase_bin"].unique())
    grouped = emergence.groupby("phase_bin")["real_minus_C_suffix"]
    median = grouped.median().reindex(bins)
    q1 = grouped.quantile(0.25).reindex(bins)
    q3 = grouped.quantile(0.75).reindex(bins)
    ax.fill_between(bins, q1, q3, color=RED, alpha=0.16, linewidth=0)
    ax.plot(bins, median, marker="o", ms=3.6, color=RED)
    ax.axhline(0, color="#A0A0A0", lw=0.8, ls="--")
    ax.set_xticks(bins, [f"{value:.0f}" for value in bins])
    ax.set_xlabel("event phase bin (early to late)")
    ax.set_ylabel(r"paired $\Delta R^2$, true order $-$ shuffled")
    panel_note(ax, "median and interquartile range over 28 patients")
    ax.set_title("True order buys no earlier commitment")

    # D: transport relative to the same patient's order-shuffled arm.
    ax = axes[1, 0]
    panel_label(ax, "d")
    control_columns = [
        ("progress_transport_cosine", "progress\ntransport", BLUE),
        ("field_transport_cosine", "field\ntransport", RED),
        ("transverse_contraction", "transverse\ncontraction", GOLD),
        ("event_to_PF_manifold_convergence", "manifold\nconvergence", GRAY),
    ]
    strip_points(
        ax,
        [c2[f"{name}_real_minus_C_suffix"] for name, _, _ in control_columns],
        [f"{label}\nabs {np.nanmedian(c2[name]):+.2f}" for name, label, _ in control_columns],
        [color for _, _, color in control_columns],
    )
    ax.set_ylabel(r"paired difference, true order $-$ shuffled")
    ax.set_title("Transport is not order-specific")
    panel_note(ax, "abs = absolute median in the true-order arms")

    # E: preregistered response matrix, with the output orientation stated.
    ax = axes[1, 1]
    panel_label(ax, "e")
    limit = max(0.15, float(np.nanmax(np.abs(matrix))))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{matrix[row, column]:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(matrix[row, column]) > 0.55 * limit else "black")
    ax.set_xticks([0, 1], ["progress\nperturbation", "future-field\nperturbation"])
    ax.set_yticks([0, 1], ["progress output\n(earliness-signed)", "future-field\noutput"])
    ax.set_title("Preregistered response matrix")
    ax.text(0.5, -0.19,
            "advancing phase favours later contacts, so the earliness row is negative by construction",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color="#555555")
    colorbar = figure.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.ax.set_title("response", fontsize=6.5, pad=3)

    # F: the one supported claim gets its own panel.
    ax = axes[1, 2]
    panel_label(ax, "f")
    offsets = {"PROGRESS": 0, "FIELD": 1}
    for axis_name, base in offsets.items():
        part = c4[c4["perturbation_axis"].eq(axis_name)]
        pair = part["real_arm_pair_cosine"].to_numpy(float)
        control = part["real_arm_to_C_suffix_cosine"].to_numpy(float)
        color = BLUE if axis_name == "PROGRESS" else RED
        left, right = 1 + 2.4 * base, 1.9 + 2.4 * base
        for low, high in zip(control, pair):
            ax.plot([left, right], [low, high], color=color, lw=0.6, alpha=0.32, zorder=2)
        ax.scatter(np.full(len(control), left), control, s=13, color=GRAY, alpha=0.6,
                   edgecolor="none", zorder=3)
        ax.scatter(np.full(len(pair), right), pair, s=13, color=color, alpha=0.6,
                   edgecolor="none", zorder=3)
        for position, values in ((left, control), (right, pair)):
            ax.plot([position - 0.22, position + 0.22], [np.median(values)] * 2,
                    color="black", lw=1.9, zorder=4)
        ax.annotate(f"n={len(pair)}", ((left + right) / 2, 0.01), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=5.8, color="#666666")
    ax.set_xticks([1, 1.9, 3.4, 4.3],
                  ["vs\nshuffled", "real\npairs", "vs\nshuffled", "real\npairs"])
    ax.set_xlim(0.5, 4.8)
    ax.set_ylabel("response-field similarity")
    ax.annotate("progress axis", (1.45, -0.155), xycoords=("data", "axes fraction"), ha="center",
                fontsize=7.0, color=BLUE, va="top")
    ax.annotate("future-field axis", (3.85, -0.155), xycoords=("data", "axes fraction"), ha="center",
                fontsize=7.0, color=RED, va="top")
    ax.set_title("Topology convergence: future-field axis only")

    # G: held-out interictal alignment, preregistered orientation, all null families.
    ax = axes[2, 0]
    panel_label(ax, "g")
    g_series = [
        (c5_null("PROGRESS", "ALL_CONTACT_SYNCHRONIZED"), "all-contact", BLUE),
        (c5_null("PROGRESS", "WITHIN_SHAFT"), "within-shaft", "#7B9FC1"),
        (c5_null("PROGRESS", "DISTANCE_BIN_LOCAL"), "distance-bin", "#7B9FC1"),
        (c5_null("PROGRESS", "GRAPH_SPECTRAL_AUTOCORRELATION"), "spectral", "#7B9FC1"),
        (c5_ident("PROGRESS"), "identity", BLUE),
        (c5_null("FIELD", "ALL_CONTACT_SYNCHRONIZED"), "all-contact", RED),
        (c5_ident("FIELD"), "identity", RED),
    ]
    strip_points(ax, [values for values, _, _ in g_series],
                 [label for _, label, _ in g_series],
                 [color for _, _, color in g_series], rotation=32)
    ax.set_ylabel("null-relative margin")
    ax.set_title("Preregistered orientation fails; size depends on the null")
    panel_note(ax, "preregistered earliness sign only;\nthe post-hoc laterness audit is this panel $\\times$ $-1$")
    ax.annotate("progress axis", (3, 0.055), xycoords=("data", "axes fraction"), ha="center",
                fontsize=7.0, color=BLUE, va="bottom")
    ax.annotate("future field", (6.5, 0.055), xycoords=("data", "axes fraction"), ha="center",
                fontsize=7.0, color=RED, va="bottom")
    ax.margins(y=0.10)

    # H: why the cross-model comparison was never opened.
    ax = axes[2, 1]
    panel_label(ax, "h")
    requirements = [
        ("explicit runtime mode", "RUNTIME_MODE_NOT_EXPLICIT"),
        ("long-run (>=20 s) audit", "SIMULATION_SHORTER_THAN_20S_LONG_RUN_CONTRACT"),
        ("fresh-network replication", "FRESH_NETWORK_REPLICATION_NOT_CLOSED"),
        ("natural-mode benchmark", "NATURAL_MODE_PATIENT_BENCHMARK_NOT_PASSED"),
        ("more than one patient", "SINGLE_PATIENT_E1146"),
        ("pre-locked field mapping", "NO_LOCKED_RNN_TO_SNN_FIELD_MAPPING_OR_CORE_DEFINITION"),
    ]
    candidates = snn["candidates"]
    grid = np.asarray([
        [0.0 if reason in candidate["ineligibility_reasons"] else 1.0
         for candidate in candidates]
        for _, reason in requirements
    ])
    ax.imshow(grid, cmap=mpl.colors.ListedColormap(["#E7D7D5", "#D8E4D6"]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            ax.text(column, row, "met" if grid[row, column] > 0.5 else "not met",
                    ha="center", va="center", fontsize=6.6,
                    color="#3F6B3B" if grid[row, column] > 0.5 else "#8A3B33")
    ax.set_xticks(range(len(candidates)), [
        f"source {index + 1}\n{int(candidate['simulation_duration_ms'] / 1000)} s, "
        f"{candidate['n_networks']} networks"
        for index, candidate in enumerate(candidates)
    ])
    ax.set_yticks(range(len(requirements)), [label for label, _ in requirements])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Cross-model inputs: eligibility not met", pad=14)
    ax.text(0.5, 1.008, "spiking-model field values were never opened",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.2, color="#555555")

    # I: locked early-ictal exploratory margins, preregistered orientation.
    ax = axes[2, 2]
    panel_label(ax, "i")
    i_series = [
        (early_column("PROGRESS", "all_contact_margin"), "all-contact", BLUE),
        (early_column("PROGRESS", "within_shaft_margin"), "within-shaft", "#7B9FC1"),
        (early_column("PROGRESS", "distance_bin_margin"), "distance-bin", "#7B9FC1"),
        (early_column("PROGRESS", "identity_margin"), "identity", BLUE),
        (early_column("FIELD", "all_contact_margin"), "all-contact", RED),
        (early_column("FIELD", "identity_margin"), "identity", RED),
    ]
    strip_points(ax, [values for values, _, _ in i_series],
                 [label for _, label, _ in i_series],
                 [color for _, _, color in i_series], rotation=32)
    ax.set_ylabel("null-relative margin")
    ax.set_title("Early-ictal correspondence: exploratory, unconfirmed")
    panel_note(ax, "preregistered earliness sign only; the target had already\nbeen viewed, so this analysis cannot confirm")
    ax.annotate("progress axis", (2.5, 0.055), xycoords=("data", "axes fraction"), ha="center",
                fontsize=7.0, color=BLUE, va="bottom")
    ax.annotate("future field", (5.5, 0.055), xycoords=("data", "axes fraction"), ha="center",
                fontsize=7.0, color=RED, va="bottom")
    ax.margins(y=0.10)

    for ax in axes.flat:
        if ax.axison and ax not in (axes[1, 1], axes[2, 1]):
            ax.spines[["top", "right"]].set_visible(False)
    figure.savefig(FIGURES / f"{STEM}.png", dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(FIGURES / f"{STEM}.pdf", bbox_inches="tight", facecolor="white")
    figure.savefig(FIGURES / f"{STEM}.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)

    inputs = {str(path.relative_to(ROOT)): sha256_file(path) for path in required}
    outputs = {
        suffix: sha256_file(FIGURES / f"{STEM}.{suffix}") for suffix in ("png", "pdf", "svg")
    }
    atomic_write_json(FIGURES / f"{STEM}_metadata.json", {
        "contract": "topic5_latent_landscape_figure_candidate_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CANDIDATE_PENDING_VISUAL_QA",
        "panels": {
            "a": "frozen decoder state and the two-coordinate hypothesis",
            "b": "held-out raw-state geometry versus matched baselines",
            "c": "paired true-order minus order-shuffled future-field emergence",
            "d": "transport and contraction relative to the order-shuffled arm",
            "e": "preregistered perturbation response matrix with stated output orientation",
            "f": "response-field topology convergence, real pairs versus order-shuffled",
            "g": "held-out data-field margins across all spatial null families and the identity null",
            "h": "cross-model input eligibility criteria",
            "i": "locked early-ictal exploratory alignment",
        },
        "orientation_contract": (
            "Panels g and i show the preregistered earliness orientation only; the post-hoc "
            "laterness audit is an exact sign mirror of the same patient values and is reported "
            "numerically in CONTROL_REFERENCED_ADDENDUM.json rather than redrawn."
        ),
        "min_null_family_patients": MIN_NULL_FAMILY_PATIENTS,
        "inputs": inputs, "outputs": outputs,
        "scientific_status": "MIXED_CLOSEOUT_NOT_A_POSITIVE_MECHANISM_FIGURE",
    })
    print(FIGURES / f"{STEM}.png")


if __name__ == "__main__":
    main()
