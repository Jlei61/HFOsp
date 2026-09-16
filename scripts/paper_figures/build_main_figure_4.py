#!/usr/bin/env python3
"""Build the large-type Figure 4 A--G package with reserved Panel B.

The upstream simulations and cohort statistics are frozen.  This producer
redraws their paper-facing views with a single typography/layout contract and
does not rerun an SNN.  Author-accepted Panel A combines the local E/I circuit
with the patient substrate.  Panel B is intentionally left empty for a future
data-driven parameter-sensitivity analysis.  The accepted spatial modes, rank
profiles, cross-fit, direct readout and cohort result continue as Panels C--G.
The KMeans heatmap/rank-distribution panel remains in Fig. S7.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAPER_ROOT = ROOT / "results/paper-ready-figure"
FIG4_ROOT = PAPER_ROOT / "fig4"
FIGURES = FIG4_ROOT / "figures"
SUPERSEDED_PACKAGE = PAPER_ROOT / "archive/2026-08-19_rejected_a_g_fig4/fig4"
PRE_REVISION_PACKAGE = PAPER_ROOT / "archive/2026-08-19_pre_a_g_fig4/fig4"
PRE_AE_REORDER_PACKAGE = (
    PAPER_ROOT / "archive/2026-09-03_pre_fig4_a_e_reorder/fig4"
)
PRE_RESERVED_B_PACKAGE = (
    PAPER_ROOT / "archive/2026-09-03_pre_fig4_reserved_panel_b/fig4"
)

NLC_CONFIG = ROOT / "config/topic4_rev11_nlc_frozen_substrate_confirmation.json"
NLC_OUTPUT = (
    ROOT
    / "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc"
    / "frozen_substrate_confirmation"
)
NLC_FIGURES = NLC_OUTPUT / "figures"
DIRECT_METADATA = NLC_FIGURES / "fig4a_nlc_direct_readout_metadata.json"
KMEANS_METADATA = NLC_FIGURES / "fig4b_nlc_kmeans_consistency_metadata.json"
NULL_CALIBRATION = NLC_OUTPUT / "null_calibration.json"
PANELE_PAIRWISE_STATS = (
    FIG4_ROOT / "fig4_panele_pairwise_similarity_statistics.json"
)
LATEST_GH_ROOT = (
    PAPER_ROOT
    / "archive/2026-09-02_noncanonical_staging_qa_cleanup"
    / "fig4gh_all_event_timing_plus_space"
)
LATEST_GH_FIGURES = LATEST_GH_ROOT / "figures"
LATEST_GH_METADATA = LATEST_GH_FIGURES / "fig4gh-all-event-space-metadata.json"
LATEST_GH_STATS = LATEST_GH_ROOT / "fig4_panelh_pairwise_similarity_statistics.json"
COHORT_RESULT = (
    ROOT
    / "results/topic4_sef_hfo/data_driven_snn_cohort_v1/formal"
    / "cohort_result.json"
)

TA_COLOR = "#C43C39"
TB_COLOR = "#277DA1"
SHAFT_COLORS = {"ICL": "#E67E22", "SCL": "#159EAE"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_pdf(image: Image.Image, path: Path, *, title: str) -> None:
    """Save a deterministic high-resolution PDF from the accepted PNG."""
    fixed_time = time.struct_time((2026, 8, 19, 0, 0, 0, 2, 231, 0))
    image.save(
        path,
        "PDF",
        resolution=600.0,
        title=title,
        creationDate=fixed_time,
        modDate=fixed_time,
    )


def _save_figure(fig: plt.Figure, panel_id: str) -> list[str]:
    png = FIGURES / f"fig4-panel{panel_id}.png"
    pdf = FIGURES / f"fig4-panel{panel_id}.pdf"
    fig.savefig(
        png,
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.045,
    )
    plt.close(fig)
    with Image.open(png) as source:
        _save_pdf(
            source.convert("RGB"), pdf,
            title=f"Figure 4 panel {panel_id.upper()}",
        )
    return [str(png.relative_to(ROOT)), str(pdf.relative_to(ROOT))]


def _load_frozen_bundle():
    from scripts.paper_figures import (
        plot_fig4_spatial_edge_flow_validation as spatial,
    )

    bundle = spatial._load_bundle(NLC_CONFIG, NLC_OUTPUT)
    return spatial, bundle


def _build_panel_a(spatial, bundle) -> tuple[list[str], dict]:
    """Keep the local circuit mechanism as its own conceptual panel."""
    from scripts.paper_figures.plot_fig4_subject_snn_grouped import (
        _draw_integrated_mechanism,
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.15), facecolor="white")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.01, top=0.92)
    _draw_integrated_mechanism(ax)
    for text in ax.texts:
        if text.get_text() == "z↓":
            # ``z↓`` is a compact symbol inside a fixed dashed circle, not a
            # regular annotation.  Applying the global 14-pt minimum makes the
            # wider composite glyph push the visible z left of the circle.
            # Keep the symbol compact and shift its anchor just enough for the
            # z glyph itself to remain visually centred on the slow-state node.
            text.set_position((5.61, 2.82))
            text.set_fontsize(13.0)
            continue
        text.set_fontsize(max(14.0, 1.35 * text.get_fontsize()))
    ax.set_title(
        "Local E/I circuit", fontsize=20, fontweight="bold", pad=-2,
    )
    outputs = _save_figure(fig, "a")
    return outputs, {
        "component": "local E/I circuit and slow-variable concept",
        "independent_of_data_driven_core_fit": True,
        "slow_variable_symbol": {
            "text": "z↓",
            "visual_anchor": [5.61, 2.82],
            "font_size_pt": 13.0,
            "reason": "keep z visually centred inside the fixed dashed node",
        },
    }


def _promote_axis_type(ax: plt.Axes, *, tick=14, label=17, title=19) -> None:
    ax.tick_params(labelsize=tick, width=1.0, length=4)
    ax.xaxis.label.set_size(label)
    ax.yaxis.label.set_size(label)
    ax.title.set_fontsize(title)


def _add_data_driven_candidate_region(ax, bundle, display: dict) -> dict:
    """Show a possible core range with low-saturation field contours."""
    from scipy.ndimage import gaussian_filter

    positions = np.asarray(bundle["static"]["positions_E"], float)
    field = np.asarray(bundle["static"]["h"], float)
    positions = (
        positions @ np.asarray(display["matrix"], float)
        + np.asarray(display["offset"], float)
    )
    xlim = tuple(float(value) for value in display["xlim"])
    ylim = tuple(float(value) for value in display["ylim"])
    x_edges = np.linspace(xlim[0], xlim[1], 181)
    y_edges = np.linspace(ylim[0], ylim[1], 181)
    weighted, _, _ = np.histogram2d(
        positions[:, 1], positions[:, 0], bins=(y_edges, x_edges), weights=field,
    )
    counts, _, _ = np.histogram2d(
        positions[:, 1], positions[:, 0], bins=(y_edges, x_edges),
    )
    sigma_bins = 4.2
    support = gaussian_filter(counts, sigma_bins)
    smooth = gaussian_filter(weighted, sigma_bins) / np.maximum(support, 1e-12)
    smooth[support < 0.03 * float(np.max(support))] = np.nan
    quantiles = (0.85, 0.92, 0.97)
    levels = np.asarray([np.quantile(field, value) for value in quantiles])
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    contour_color = "#8D8397"
    ax.contour(
        x_centers, y_centers, smooth,
        levels=levels, colors=contour_color,
        linewidths=(0.85, 1.05, 1.25), alpha=0.72, zorder=4.0,
    )
    ax.set_aspect("equal")
    ax.text(
        0.0, ylim[0] + 0.65, "possible data driven core",
        ha="center", va="bottom", color="#776F80",
        fontsize=8.8, fontweight="bold", zorder=15,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.9),
    )
    return {
        "source": "complete frozen v62_density_t050 Node field",
        "display": "low-saturation contours of the high-field range",
        "label": "possible data driven core",
        "discrete_core_markers": False,
        "field_quantiles": list(quantiles),
        "contour_levels": levels.tolist(),
        "contour_color": contour_color,
        "contour_alpha": 0.72,
        "gaussian_sigma_bins": sigma_bins,
    }


def _build_panel_b(spatial, bundle) -> tuple[list[str], dict]:
    """Retain E/I information and mark a possible core with soft contours."""
    from scripts.paper_figures.plot_fig4_subject_snn_grouped import DEFAULT_TAG
    from scripts.paper_figures.plot_fig_subject_snn import (
        _registered_axis_display,
    )
    from scripts.paper_figures.plot_fig_subject_snn_mechanism import (
        _load_figdata,
        _plot_mechanism,
        _reconstruct_posI,
    )

    fd, source_path = _load_figdata(DEFAULT_TAG)
    updated = dict(fd)
    reg = dict(fd["reg"].item())
    # Do not retain the historical early-contact highlights: they visually
    # recreate two discrete cores even after the circles are removed.
    reg["source_names"] = []
    reg["sink_names"] = []
    updated["reg"] = np.asarray(reg, dtype=object)
    pos_i, pos_i_meta = _reconstruct_posI(fd, DEFAULT_TAG)
    plot_seed = int((pos_i_meta.get("seed") or 0) + 101)

    fig, ax = plt.subplots(figsize=(5.45, 4.15), facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    display = _registered_axis_display(fd)
    setup = _plot_mechanism(
        updated, ax, clean=True, posI=pos_i, plot_seed=plot_seed,
        display=display, homogeneous_cores=True,
        semantic_core_colors=False, show_basic_labels=True, show_title=False,
    )
    # The legacy painter needs two foci to construct its anisotropic E-to-E
    # footprint, but the paper-facing panel must not display those foci as
    # biological cores.  Remove only the two circle marks, their labels and the
    # line joining them; the E/I population, contacts and E-to-E corridor stay.
    for patch in list(ax.patches):
        if isinstance(patch, Circle):
            patch.remove()
    for line in list(ax.lines):
        if float(line.get_zorder()) == 8.0:
            line.remove()
    for text in list(ax.texts):
        if text.get_text().startswith("Core "):
            text.remove()
    candidate_region = _add_data_driven_candidate_region(ax, bundle, display)
    for text in ax.texts:
        text.set_fontsize(max(13.0, 1.5 * text.get_fontsize()))
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(12.5)
    outputs = _save_figure(fig, "b")
    return outputs, {
        "source_schematic": str(source_path.relative_to(ROOT)),
        "preserved_information": [
            "E and I neurons", "local E-to-I and I-to-E context",
            "anisotropic E-to-E scaffold", "patient contact geometry",
        ],
        "removed_components": [
            "Core 1 marker", "Core 2 marker", "core-to-core line",
            "core contact highlights",
        ],
        "candidate_region": candidate_region,
        "setup": setup,
    }


def _build_panel_c(spatial, bundle, panel_id: str = "c") -> tuple[list[str], dict]:
    """Redraw the same-network virtual-contact readout with readable type."""
    fig, ax = plt.subplots(figsize=(10.9, 4.0), facecolor="white")
    fig.subplots_adjust(left=0.115, right=0.99, bottom=0.18, top=0.86)
    pair = spatial._same_network_pair(bundle)
    readout = spatial._plot_readout(
        ax, bundle, pair,
        shade_contract="recruitment_onset_span", shade_pad_ms=12.0,
        show_onset_markers=False, show_scale_bar=False,
    )
    ax.set_xlabel("Time (ms)", fontsize=17)
    ax.set_ylabel("30–80 Hz activity", fontsize=17)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=13)
    for text in ax.texts:
        text.set_fontsize(max(13.0, text.get_fontsize()))
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(14)
    outputs = _save_figure(fig, panel_id)
    return outputs, readout


def _build_panel_d(spatial, bundle, panel_id: str = "d") -> tuple[list[str], dict]:
    """Show the data-driven field and both spatial modes without gray copy."""
    fig = plt.figure(figsize=(12.8, 4.35), facecolor="white")
    grid = fig.add_gridspec(
        1, 5, width_ratios=(1.26, 0.035, 0.22, 1.0, 1.0),
        left=0.025, right=0.992, bottom=0.13, top=0.90, wspace=0.07,
    )
    ax_field = fig.add_subplot(grid[0, 0], projection="3d")
    axes = [fig.add_subplot(grid[0, index]) for index in (3, 4)]
    context = spatial._plot_landscape(ax_field, bundle)
    ax_field.text2D(
        0.02, 0.975, "Data-driven Node field",
        transform=ax_field.transAxes, ha="left", va="top",
        fontsize=19, fontweight="bold", color="#243238",
    )
    colorbar_ax = fig.axes[-1]
    field_box = ax_field.get_position()
    colorbar_slot = grid[0, 1].get_position(fig)
    colorbar_ax.set_position([
        colorbar_slot.x0,
        field_box.y0 + 0.24 * field_box.height,
        colorbar_slot.width,
        0.44 * field_box.height,
    ])
    colorbar_ax.set_ylabel("")
    colorbar_ax.set_title("h", fontsize=16, pad=7)
    colorbar_ax.yaxis.set_ticks_position("left")
    colorbar_ax.tick_params(labelsize=13, pad=2)
    modes = (
        (spatial.TA_MODE, "Model TA", True),
        (spatial.TB_MODE, "Model TB", False),
    )
    n_networks = len(bundle["config"]["search"][bundle["network_seed_key"]])
    for ax, (mode, title, show_ylabel) in zip(axes, modes):
        spatial._plot_mode_density(
            ax, bundle, mode, title, show_ylabel=show_ylabel,
        )
        ax.set_xlabel("x (mm)", fontsize=19)
        ax.set_ylabel("y (mm)" if show_ylabel else "", fontsize=19)
        ax.set_title(title, fontsize=20, color=spatial.MODE_COLORS[mode],
                     fontweight="bold", pad=8)
        ax.tick_params(labelsize=14)
        for text in ax.texts:
            if text.get_text().startswith("clean events"):
                text.set_text(
                    f"n={int(bundle['clean_counts'][mode])} · {n_networks} nets"
                )
                text.set_fontsize(13.5)
    # The model maps should not visually dominate the Node-field context.
    # Preserve their square geometry and typography while reducing each axes
    # box around its own centre; centring also keeps the two maps aligned.
    mode_axes_scale = 0.74
    for ax in axes:
        box = ax.get_position()
        width = mode_axes_scale * box.width
        height = mode_axes_scale * box.height
        ax.set_position([
            box.x0 + 0.5 * (box.width - width),
            box.y0 + 0.5 * (box.height - height),
            width,
            height,
        ])
    outputs = _save_figure(fig, panel_id)
    return outputs, {
        "mode_counts": {
            "MTA": int(bundle["clean_counts"][spatial.TA_MODE]),
            "MTB": int(bundle["clean_counts"][spatial.TB_MODE]),
        },
        "pooled_networks": n_networks,
        "gray_status_subtitle_removed": True,
        "data_driven_landscape_retained": True,
        "field_colorbar_uses_dedicated_grid_column": True,
        "field_colorbar_ticks_on_left_and_title_on_top": True,
        "mode_axis_labels": {"x": "x (mm)", "y": "y (mm)"},
        "mode_axes_scale_relative_to_grid_cell": mode_axes_scale,
        "geometry_context": {
            key: context[key] for key in (
                "coord_space", "n_context_contacts", "n_selected_contacts",
                "registration_max_abs_error_mm",
            )
        },
    }


def _build_panel_e(panel_id: str = "e") -> tuple[list[str], dict]:
    from scripts.paper_figures import plot_fig4e_data_driven_cohort as cohort

    metadata = cohort.build(
        output_dir=FIGURES,
        result_path=COHORT_RESULT,
        stem_name=f"fig4-panel{panel_id}",
    )
    # The cohort producer writes its own PDF.  Replace it with the same
    # deterministic 600-dpi packaging contract used by every other panel.
    png = FIGURES / f"fig4-panel{panel_id}.png"
    pdf = FIGURES / f"fig4-panel{panel_id}.pdf"
    with Image.open(png) as source:
        _save_pdf(source.convert("RGB"), pdf, title=f"Figure 4 panel {panel_id.upper()}")
    metadata["files"]["pdf_sha256"] = _sha256(pdf)
    metadata_path = FIGURES / f"fig4-panel{panel_id}-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return [metadata["files"]["png"], metadata["files"]["pdf"]], metadata


def _kmeans_display_payload(spatial, bundle) -> dict:
    from scripts import plot_interictal_propagation as propagation_plot

    canonical = spatial._canonical_rank_kmeans(bundle)
    selected = canonical["clean_global_index"]
    n_contacts = len(bundle["static"]["contact_names"])
    display_ranks = spatial.normalize_event_ranks(bundle["ranks"][selected]) * (
        n_contacts - 1
    )
    frozen_labels, mapping = spatial._map_kmeans_clusters_to_modes(
        canonical["labels"], canonical["direction_contingency"],
    )
    # Display order is always MTA then MTB.
    labels = np.where(frozen_labels == spatial.TA_MODE, 0, 1).astype(int)
    expected = json.loads(KMEANS_METADATA.read_text(encoding="utf-8"))[
        "display_cluster_counts_MTA_MTB"
    ]
    observed = np.bincount(labels, minlength=2).tolist()
    if observed != expected:
        raise RuntimeError(
            f"frozen KMeans display counts drifted: {observed} != {expected}"
        )
    names = np.asarray(bundle["static"]["contact_names"], str)
    patient, _, _ = spatial._patient_profiles(bundle)
    patient = patient * (n_contacts - 1)
    patient_rank_matrix = np.asarray(
        bundle["patient"]["patient_train_ranks"], float,
    ).T
    order = propagation_plot._fixed_channel_order(
        patient_rank_matrix, np.isfinite(patient_rank_matrix),
    )
    model = np.asarray([
        spatial._column_stats(display_ranks[labels == mode])[0]
        for mode in (0, 1)
    ])
    return {
        "ranks_event_contact": display_ranks,
        "labels": labels,
        "names": names,
        "order": order,
        "model_profiles": model,
        "patient_profiles": patient,
        "cluster_mapping": mapping.tolist(),
        "counts": observed,
        "n_contacts": n_contacts,
    }


def _build_panel_f(payload: dict, panel_id: str = "f") -> tuple[list[str], dict]:
    """Heatmap, rank distribution, and one close shared colorbar."""
    from scripts import plot_interictal_propagation as propagation_plot

    from matplotlib import gridspec

    ranks = payload["ranks_event_contact"].T
    bools = np.isfinite(ranks)
    labels = payload["labels"]
    order = payload["order"]
    names = payload["names"]
    clustered = np.argsort(labels, kind="stable")
    counts = payload["counts"]

    fig = plt.figure(figsize=(11.4, 4.65), facecolor="white")
    outer = gridspec.GridSpec(
        1, 3, figure=fig, width_ratios=(6.5, 1.38, 0.16),
        left=0.075, right=0.985, bottom=0.17, top=0.87, wspace=0.075,
    )
    ax_heat = fig.add_subplot(outer[0, 0])
    ax_rank = fig.add_subplot(outer[0, 1])
    ax_cbar = fig.add_subplot(outer[0, 2])
    image = propagation_plot._plot_rank_heatmap(
        ax_heat,
        ranks[order][:, clustered],
        names[order].tolist(),
        title="",
        display_bools=bools[order][:, clustered],
        ytick_fontsize=13.5,
        title_fontsize=18,
        xtick_fontsize=13,
    )
    boundary = counts[0]
    gap = max(2, int(round(0.008 * len(labels))))
    ax_heat.axvspan(
        boundary - gap, boundary + gap,
        facecolor="white", edgecolor="0.62", hatch="////",
        linewidth=0.0, zorder=12,
    )
    centers = ((boundary - gap) / 2, boundary + gap + (len(labels) - boundary - gap) / 2)
    for center, label, count, color in zip(
        centers, ("MTA", "MTB"), counts, (TA_COLOR, TB_COLOR),
    ):
        ax_heat.text(
            center, 1.035, f"{label}  n={count}",
            transform=ax_heat.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=17, fontweight="bold", color=color, clip_on=False,
        )
    ax_heat.set_xlabel("Events", fontsize=16)
    ax_heat.set_ylabel("Electrode contact", fontsize=16)
    ax_heat.tick_params(axis="x", labelsize=13)

    propagation_plot._plot_rank_histogram(
        ax_rank, ranks, bools, np.arange(len(labels)), order,
        names.tolist(), title="Rank distribution", show_ylabels=False,
        label_fontsize=15, title_fontsize=17, xtick_fontsize=13,
        ridge_spacing=0.10, smooth_sigma_bins=0.72,
        smooth_ridge_height=0.12,
    )
    ax_rank.set_xlabel("Rank", fontsize=16)
    ax_rank.tick_params(labelsize=13)
    colorbar = fig.colorbar(image, cax=ax_cbar)
    colorbar.set_label("First → last", fontsize=16, labelpad=7)
    colorbar.ax.tick_params(labelsize=13, pad=2)
    outputs = _save_figure(fig, panel_id)
    return outputs, {
        "event_counts_MTA_MTB": counts,
        "rank_distribution_and_heatmap_share_colorbar": True,
        "component_wspace": 0.075,
        "rank_contract": "masked normalized event ranks; >=3 contacts",
    }


def _build_panel_g(payload: dict) -> tuple[list[str], dict]:
    """Replace the crowded four-band profile with four clean lines."""
    order = payload["order"]
    names = payload["names"]
    model = payload["model_profiles"]
    patient = payload["patient_profiles"]
    y = np.arange(len(order), dtype=float)

    fig, ax = plt.subplots(figsize=(5.45, 5.1), facecolor="white")
    fig.subplots_adjust(left=0.23, right=0.975, bottom=0.23, top=0.90)
    specifications = (
        (model[0], TA_COLOR, "-", "o", "MTA model"),
        (patient[0], TA_COLOR, "--", None, "TA patient"),
        (model[1], TB_COLOR, "-", "o", "MTB model"),
        (patient[1], TB_COLOR, "--", None, "TB patient"),
    )
    for values, color, linestyle, marker, label in specifications:
        selected = np.asarray(values)[order]
        finite = np.isfinite(selected)
        ax.plot(
            selected[finite], y[finite], linestyle,
            color=color, lw=2.6 if linestyle == "-" else 2.2,
            marker=marker, ms=5.4, label=label,
        )
    ax.set_yticks(y, names[order], fontsize=13)
    for tick, contact in zip(ax.get_yticklabels(), names[order]):
        shaft = "".join(character for character in contact if not character.isdigit())
        tick.set_color(SHAFT_COLORS.get(shaft, "#333333"))
    ax.invert_yaxis()
    ax.set_xlim(-0.4, payload["n_contacts"] - 0.6)
    ax.set_xticks([0, 4, 8, 12, 14])
    ax.set_xlabel("Mean rank", fontsize=16)
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(axis="x", color="#E3E7E9", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Rank profiles", fontsize=19, fontweight="bold", pad=8)
    # Keep the full plot height.  A single compact legend lives below the
    # x-label instead of pushing the axes down from above.
    fig.legend(
        handles=[
            Line2D([0], [0], color=TA_COLOR, lw=2.6, marker="o",
                   ms=5.0, label="MTA"),
            Line2D([0], [0], color=TA_COLOR, lw=2.2, ls="--", label="TA"),
            Line2D([0], [0], color=TB_COLOR, lw=2.6, marker="o",
                   ms=5.0, label="MTB"),
            Line2D([0], [0], color=TB_COLOR, lw=2.2, ls="--", label="TB"),
        ],
        frameon=False, fontsize=12.5, ncol=4, loc="lower center",
        bbox_to_anchor=(0.60, 0.015), columnspacing=0.9,
        handlelength=1.6, handletextpad=0.35,
    )
    outputs = _save_figure(fig, "g")
    return outputs, {
        "model_lines": ["MTA", "MTB"],
        "patient_lines": ["TA", "TB"],
        "uncertainty_bands_removed_for_legibility": True,
        "contact_order": names[order].tolist(),
    }


def _build_panel_h() -> tuple[list[str], dict]:
    """Show two separate matched-similarity tests in the square matrix."""
    metadata = json.loads(KMEANS_METADATA.read_text(encoding="utf-8"))
    pairwise = json.loads(PANELE_PAIRWISE_STATS.read_text(encoding="utf-8"))
    matrix = np.asarray(metadata["displayed_matrix"], float)
    audited_matrix = np.asarray(pairwise["displayed_equal_network_matrix"], float)
    if not np.allclose(matrix, audited_matrix, rtol=0.0, atol=1e-12):
        raise RuntimeError("Fig4H pairwise tests do not match the displayed matrix")
    tests_by_cell = {
        (0, 0): pairwise["tests"]["MTA_vs_TA"],
        (1, 1): pairwise["tests"]["MTB_vs_TB"],
    }
    fig = plt.figure(figsize=(4.6, 4.7), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, width_ratios=(1.0, 0.075),
        left=0.19, right=0.94, bottom=0.15, top=0.84, wspace=0.10,
    )
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_aspect("equal")
    ax.set_xticks((0, 1), ("TA", "TB"), fontsize=16, fontweight="bold")
    ax.set_yticks((0, 1), ("MTA", "MTB"), fontsize=16, fontweight="bold")
    for tick, color in zip(ax.get_xticklabels(), (TA_COLOR, TB_COLOR)):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), (TA_COLOR, TB_COLOR)):
        tick.set_color(color)
    for row in range(2):
        for column in range(2):
            foreground = (
                "white" if abs(matrix[row, column]) >= 0.55 else "#111111"
            )
            value_y = row - 0.10 if (row, column) in tests_by_cell else row
            ax.text(
                column, value_y, f"{matrix[row, column]:+.2f}",
                ha="center", va="center", fontsize=19, fontweight="bold",
                color=foreground,
            )
            if (row, column) in tests_by_cell:
                ax.text(
                    column, row + 0.22, tests_by_cell[(row, column)]["stars"],
                    ha="center", va="center", fontsize=16.5,
                    fontweight="bold", color=foreground,
                )
    ax.set_title("Cross-fit similarity", fontsize=19, fontweight="bold", pad=12)
    colorbar = fig.colorbar(image, cax=cax, ticks=(-1, -0.5, 0, 0.5, 1))
    colorbar.set_label("ρ", fontsize=17, labelpad=5)
    colorbar.ax.tick_params(labelsize=13)
    outputs = _save_figure(fig, "h")
    return outputs, {
        "matrix": matrix.tolist(),
        "contract": metadata["displayed_matrix_contract"],
        "separate_similarity_tests": pairwise["tests"],
        "statistics_source": str(PANELE_PAIRWISE_STATS.relative_to(ROOT)),
        "unified_diagonal_margin_test_displayed": False,
        "square_matrix": True,
        "same_complete_row_height_as_f_g": True,
    }


def _install_latest_rank_and_crossfit_as_d_e() -> tuple[dict[str, list[str]], dict[str, dict]]:
    """Install the accepted rank profile and cross-fit as panels D and E."""
    refresh = json.loads(LATEST_GH_METADATA.read_text(encoding="utf-8"))
    pairwise = json.loads(LATEST_GH_STATS.read_text(encoding="utf-8"))
    if refresh.get("schema_version") != "figure4gh_all_event_timing_plus_space_v1":
        raise ValueError("unexpected Figure 4G/H refresh schema")
    if pairwise.get("schema_version") != "fig4_panelh_pairwise_similarity_all_event_space_v1":
        raise ValueError("unexpected Figure 4H refreshed-statistics schema")
    outputs: dict[str, list[str]] = {}
    for source_id, target_id in (("g", "d"), ("h", "e")):
        paths: list[str] = []
        for suffix in (".png", ".pdf"):
            source = LATEST_GH_FIGURES / f"fig4-panel{source_id}{suffix}"
            target = FIGURES / f"fig4-panel{target_id}{suffix}"
            shutil.copy2(source, target)
            paths.append(str(target.relative_to(ROOT)))
        outputs[target_id] = paths
    shutil.copy2(LATEST_GH_STATS, PANELE_PAIRWISE_STATS)
    details = {
        "d": {
            "model_lines": ["MTA", "MTB"],
            "patient_lines": ["TA", "TB"],
            "patient_template_source": refresh["template_source"],
            "label_change": refresh["label_change"],
            "model_events_and_kmeans_frozen": True,
        },
        "e": {
            "matrix": pairwise["displayed_equal_network_matrix"],
            "contract": "12-network equal-weight contact-split cross-fit Spearman",
            "separate_similarity_tests": pairwise["tests"],
            "statistics_source": str(PANELE_PAIRWISE_STATS.relative_to(ROOT)),
            "unified_diagonal_margin_test_displayed": False,
            "square_matrix": True,
        },
    }
    return outputs, details


def _compose_complete_layout() -> list[str]:
    from scripts.paper_figures.build_main_figures_1_2 import (
        _compose_complete_layout,
    )

    outputs = _compose_complete_layout(
        figures_dir=FIGURES,
        stem="fig4-complete-layout",
        canvas_size=(10500, 7500),
        placements={
            "a": (FIGURES / "fig4-panela.png", (140, 150, 5500, 2800)),
            "c": (FIGURES / "fig4-panelc.png", (140, 3000, 6000, 5000)),
            "d": (FIGURES / "fig4-paneld.png", (6200, 3000, 7700, 5000)),
            "e": (FIGURES / "fig4-panele.png", (8000, 3000, 10360, 5000)),
            "f": (FIGURES / "fig4-panelf.png", (140, 5300, 6500, 7350)),
            "g": (FIGURES / "fig4-panelg.png", (6800, 5300, 10360, 7350)),
        },
        labels={
            "A": (25, 20),
            "B": (5580, 20),
            "C": (25, 2870), "D": (6065, 2870),
            "E": (7865, 2870),
            "F": (25, 5170), "G": (6665, 5170),
        },
        anchors={
            "a": "top-left", "c": "top-left", "d": "top",
            "e": "top", "f": "top-left", "g": "top",
        },
        label_font_size=132,
    )
    complete_png = FIGURES / "fig4-complete-layout.png"
    complete_pdf = FIGURES / "fig4-complete-layout.pdf"
    with Image.open(complete_png) as source:
        _save_pdf(
            source.convert("RGB"), complete_pdf,
            title="Figure 4 complete layout",
        )
    return outputs


def _write_readme() -> None:
    (FIGURES / "README.md").write_text(
        f"""# Figure 4 panel 与完整排版

本目录是 Figure 4 的 canonical paper-facing 入口。独立 panel 不带左上角字母；只有 `fig4-complete-layout` 带 A–G。Panel B 按作者要求留空，等待后续补入 data-driven 参数对患者间期事件复现的影响，因此当前没有 `fig4-panelb` 独立文件。其余 PNG 均为 600 dpi，并提供同画面的 PDF。

### fig4-panela.png / .pdf

作者确认的组合 panel。左侧为 local E/I circuit 与慢变量示意，右侧为同一冻结 figdata 的 patient-specific E/I substrate 和患者触点几何；右图删除 `anisotropic E→E` corridor 及 `possible data driven core` 覆盖层。左框与右侧 SCL9 局部框使用相同的约 `1.68 × 1.31 mm` 视野；左图 scale bar 为 `0.5 mm`，右图显示 `−10–10 mm` 仿真坐标。每个触点周围的低透明度深绿色 halo 表示 Fig4F firing-density readout 的 Gaussian sampling footprint（`σ=0.25 mm`；理论 95% 权重半径 `0.61 mm`）。

**关注点**：两个框尺度匹配，但左侧仍是机制示意而非从右图裁出的组织图；绿色 halo 不是解剖边界。

### Panel B（预留，无独立文件）

完整拼板右上区域仅保留 B 角标和空白画布，后续用于展示 data-driven 模型中不同参数对患者间期事件复现的影响。当前版本不填入占位文字、示意数据或临时结果。

**关注点**：B 的空白是明确的作者布局合同，不代表缺失文件或构建失败。

### fig4-panelc.png / .pdf

冻结 data-driven Node field，以及 12 张网络 pooled clean events 的 Model TA 与 Model TB 空间模式。Node-field 色条使用独立列，刻度在左、`h` 在顶部，不与场图或 `y (mm)` 重叠；模型图坐标为 `x (mm)` / `y (mm)`，两个模型方图缩至各自网格单元的 74%。

**关注点**：MTA/MTB 是 development-level 模型模式，不是解剖 core。

### fig4-paneld.png / .pdf

模型 MTA/MTB 与患者 TA/TB 的 mean-rank profile。删除拥挤的不确定性带；legend 放到 panel 底部，不再从上方压低主体。

**关注点**：这是 E10 development 对照，不是 cohort 级模板恢复。患者 TA/TB 使用 all-event Timing+Space 标签；模型 MTA/MTB 与 SNN KMeans 保持冻结。

### fig4-panele.png / .pdf

12 张网络等权的 contact-split cross-fit Spearman 矩阵；患者 TA/TB 使用 all-event Timing+Space 标签，模型端保持冻结。两个 `***` 分别检验 MTA–TA 与 MTB–TB 是否高于各自的 within-shaft contact-permutation null。

**关注点**：这是 development-case 的 post-hoc null calibration，不支持 patient-blind 或真实几何泛化。

### fig4-panelf.png / .pdf

同一网络窗口的 30–80 Hz virtual-contact firing-density readout。MTA/MTB 阴影按参与触点的 recruitment-onset span（两侧各加 12 ms）绘制，不再用过长的 detector-event duration；比例文字和逐通道黑色 onset 点均已删除。

**关注点**：这不是 current、LFP 或临床 SEEG；MTB 阴影只标招募时间范围。

### fig4-panelg.png / .pdf

34 人 held-out cohort。左侧完整列出 eligible、loss 胜过 matched null、same-network two-mode pass 及两项同时满足的人数和比例；右侧为被试内 paired held-out loss。

**关注点**：`P=0.043` 只对应连续 loss 的配对检验；11/34 是描述性交集。

### fig4-complete-layout.png / .pdf

三行 A–G 完整排版：A 为组合机制/底物；B 使用既有右上留白并明确预留；C–E 为模型空间模式与患者模板 rank/cross-fit；F–G 为同网络 readout 与 cohort。panel 字母只在此文件中出现。原 KMeans heatmap/rank-distribution panel 保留在 Supplementary Fig. 7E。

**关注点**：当前安全口径仍是 development-level partial interictal substrate support；不能升级为临床因果机制、解剖 core、patient-blind real-geometry generalization 或 ictal lifecycle 证明。
""",
        encoding="utf-8",
    )


def _remove_superseded_current_files() -> None:
    # Removed states remain recoverable in the versioned Figure 4 archives.
    for name in (
        "fig4-panelb.png",
        "fig4-panelb.pdf",
        "fig4-paneld-metadata.json",
        "fig4-panele-metadata.json",
        "fig4-panelh.png",
        "fig4-panelh.pdf",
        "fig4-panelab-candidate.png",
        "fig4-panelab-candidate.pdf",
        "fig4-panelab-candidate-metadata.json",
        "fig4_panelh_pairwise_similarity_statistics.json",
        "fig4_panelg_pairwise_similarity_statistics.json",
        "fig4_paneld_pairwise_similarity_statistics.json",
        "fig4-panelf.png",
        "fig4-panelf.pdf",
        "fig4-panelg.png",
        "fig4-panelg.pdf",
    ):
        (FIGURES / name).unlink(missing_ok=True)
        (FIG4_ROOT / name).unlink(missing_ok=True)


def build() -> dict:
    FIGURES.mkdir(parents=True, exist_ok=True)
    _remove_superseded_current_files()
    spatial, bundle = _load_frozen_bundle()
    panels: dict[str, list[str]] = {}
    panel_details: dict[str, dict] = {}
    from scripts.paper_figures.build_fig4_panel_a_combined import compose

    panel_details["a"] = compose()
    panels["a"] = panel_details["a"]["outputs"]
    panels["b"] = []
    panel_details["b"] = {
        "status": "AUTHOR_RESERVED_EMPTY",
        "intended_content": (
            "effects of data-driven parameter variation on reproducing "
            "patient interictal events"
        ),
        "complete_layout_region": "top-right",
        "standalone_output": False,
        "placeholder_text_drawn": False,
    }
    panels["c"], panel_details["c"] = _build_panel_d(
        spatial, bundle, panel_id="c",
    )
    refreshed_panels, refreshed_details = _install_latest_rank_and_crossfit_as_d_e()
    panels.update(refreshed_panels)
    panel_details.update(refreshed_details)
    panels["f"], panel_details["f"] = _build_panel_c(
        spatial, bundle, panel_id="f",
    )
    panels["g"], panel_details["g"] = _build_panel_e(panel_id="g")
    complete = _compose_complete_layout()
    _write_readme()

    direct_metadata = json.loads(DIRECT_METADATA.read_text(encoding="utf-8"))
    kmeans_metadata = json.loads(KMEANS_METADATA.read_text(encoding="utf-8"))
    registry = {
        "schema_version": "paper_figure4_reserved_b_a_g_v14",
        "status": "LAYOUT_INCOMPLETE_RESERVED_PANEL_B",
        "producer": "scripts/paper_figures/build_main_figure_4.py",
        "simulation_rerun": False,
        "plotting_recomputed_from_frozen_arrays": True,
        "panel_letters_in_individual_files": False,
        "panel_letters_in_complete_layout": True,
        "individual_panel_png_dpi": 600,
        "complete_layout_dpi": 600,
        "panels": panels,
        "panel_details": panel_details,
        "complete_layout": complete,
        "superseded_package_archive": str(SUPERSEDED_PACKAGE.relative_to(ROOT)),
        "pre_revision_package_archive": str(PRE_REVISION_PACKAGE.relative_to(ROOT)),
        "pre_a_e_reorder_package_archive": str(PRE_AE_REORDER_PACKAGE.relative_to(ROOT)),
        "pre_reserved_b_package_archive": str(PRE_RESERVED_B_PACKAGE.relative_to(ROOT)),
        "author_revision_contract": {
            "locked_on": "2026-09-03",
            "final_panel_letters": "A-G",
            "former_a_and_b_combined_as_panel_a": True,
            "panel_a_is_author_accepted": True,
            "panel_a_slow_variable_z_visual_center_restored": True,
            "panel_a_right_side_preserves_e_i_substrate_and_contact_geometry": True,
            "panel_a_right_side_discrete_core_markers_removed": True,
            "panel_a_right_side_anisotropic_corridor_removed": True,
            "panel_a_right_side_possible_core_overlay_removed": True,
            "panel_a_seeg_sampling_footprints_are_low_alpha_dark_green": True,
            "panel_b_reserved_empty": True,
            "panel_b_intended_for_parameter_sensitivity": True,
            "panel_b_placeholder_text_drawn": False,
            "old_d_gray_subtitle_removed": True,
            "panel_c_colorbar_separated_and_axis_labels_simplified": True,
            "panel_c_model_maps_reduced_to_74_percent": True,
            "panel_d_profile_redrawn_without_uncertainty_bands": True,
            "panel_d_legend_does_not_reduce_axes_height": True,
            "panel_d_all_event_timing_plus_space_patient_templates": True,
            "panel_e_crossfit_restored_to_main_figure": True,
            "panel_f_shading_uses_recruitment_onset_span": True,
            "panel_f_scale_bar_and_onset_dots_removed": True,
            "kmeans_heatmap_moved_to_supplementary_figure7_e": True,
            "reserved_panel_b_uses_existing_top_right_whitespace": True,
            "panel_c_scale_in_complete_layout_unchanged": True,
            "large_type_contract": "13 pt minimum for data/tick text; 15-20 pt labels and titles",
        },
        "sources": {
            "nlc_config": {
                "path": str(NLC_CONFIG.relative_to(ROOT)),
                "sha256": _sha256(NLC_CONFIG),
            },
            "nlc_output": str(NLC_OUTPUT.relative_to(ROOT)),
            "direct_metadata": {
                "path": str(DIRECT_METADATA.relative_to(ROOT)),
                "sha256": _sha256(DIRECT_METADATA),
            },
            "kmeans_metadata": {
                "path": str(KMEANS_METADATA.relative_to(ROOT)),
                "sha256": _sha256(KMEANS_METADATA),
            },
            "cohort": {
                "path": str(COHORT_RESULT.relative_to(ROOT)),
                "sha256": _sha256(COHORT_RESULT),
            },
            "panel_d_source_metadata": {
                "path": str(LATEST_GH_METADATA.relative_to(ROOT)),
                "sha256": _sha256(LATEST_GH_METADATA),
            },
            "panel_e_pairwise_statistics": {
                "path": str(PANELE_PAIRWISE_STATS.relative_to(ROOT)),
                "sha256": _sha256(PANELE_PAIRWISE_STATS),
            },
        },
        "panel_contract": {
            "a": "combined local E/I circuit and patient substrate with scale-matched inset",
            "b": "reserved for data-driven parameter sensitivity; intentionally empty",
            "c": "data-driven Node field with separated colorbar and pooled Model TA/MTB modes",
            "d": "frozen model and all-event Timing+Space patient mean-rank profiles",
            "e": "all-event Timing+Space equal-network contact-split model-patient matrix",
            "f": "same-network 30-80 Hz virtual-contact activity without scale bar or onset dots",
            "g": "34-patient cohort gates and paired matched-null loss",
        },
        "scientific_claim_boundary": direct_metadata["claim_boundary"],
        "nlc_source_status": direct_metadata["source_status"],
        "kmeans_event_subset": kmeans_metadata["kmeans_event_subset_contract"],
        "kmeans_display_counts_MTA_MTB": kmeans_metadata[
            "display_cluster_counts_MTA_MTB"
        ],
        "kmeans_panel_destination": "Supplementary Fig. 7E",
        "displayed_crossfit_matrix": panel_details["e"]["matrix"],
        "cohort_panel_g": panel_details["g"],
    }
    (FIG4_ROOT / "figure4_panel_registry.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return registry


def main() -> None:
    print(json.dumps(build(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
