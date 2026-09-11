#!/usr/bin/env python3
"""Render paper-facing Figure 5C/D candidates from the verified spatial-Z result.

Panel C is the smallest scale supported by the deterministic reduction: the
per-neuron mean E rate inside data-driven core A along the symmetric spatial-Z
continuation path.  It is deliberately not labelled as a literal single-cell
bifurcation because each 2-mm coarse unit represents a local E/I population.

Panel D preserves the accepted state-response semantic: identical frozen
random perturbation sites are probed in the low state and early-ictal state,
then paired probe-minus-sham responses are averaged across sites.  The spatial
zero mode and independent-core root catalog are rendered separately as a
mechanism supplement rather than substituted for Panel D.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, PowerNorm
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_dual_core_spatial_z_bifurcation import load_z_map  # noqa: E402
from src.topic4_patient_zm_meanfield import load_patient_coarse_model  # noqa: E402


LOW = "#355C8A"
HIGH = "#C7472F"
GLOBAL_BRANCH = "#B8794B"
LOCAL_BRANCH = "#777777"
FOLD_MARKER = "#333333"
OU = "#D62745"
COEXIST = "#D8D2E8"
RECRUITED_ONLY = "#E88963"
CORE_A = "#F28E2B"
CORE_B = "#2FA7B8"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem + ".", suffix=".json")
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _panel_label(axis, label: str, *, x: float = -0.16, y: float = 1.10) -> None:
    axis.text(
        x, y, label, transform=axis.transAxes, ha="left", va="top",
        fontsize=17, fontweight="bold", clip_on=False,
    )


def _style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7.6, width=0.8, length=3.0)


def _grid_mean(xy, values, extent, n=100):
    xy = np.asarray(xy, float)
    values = np.asarray(values, float)
    lo, hi = map(float, extent)
    edges = np.linspace(lo, hi, int(n) + 1)
    total, _, _ = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=(edges, edges), weights=values)
    count, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(edges, edges))
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = total / count
    return gaussian_filter(np.nan_to_num(grid), sigma=1.15)


def _plot_response(axis, contacts, response_grid, extent, title, vmax,
                   *, show_ylabel=True):
    image = axis.imshow(
        response_grid.T, origin="lower", extent=(*extent, *extent),
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear",
        aspect="equal",
    )
    axis.scatter(
        contacts[:, 0], contacts[:, 1], s=16, fc="white", ec="0.2",
        lw=0.45, alpha=0.85, zorder=4,
    )
    axis.set_xlim(*extent)
    axis.set_ylim(*extent)
    axis.set_xlabel("sheet x (mm)", fontsize=8.4)
    axis.set_ylabel("sheet y (mm)" if show_ylabel else "", fontsize=8.4)
    axis.set_title(title, fontsize=9.4, fontweight="bold", pad=7)
    axis.tick_params(labelsize=7.1, width=0.8, length=2.7)
    axis.spines[["top", "right"]].set_visible(False)
    return image


def _plot_panel_c(axis, arrays, payload, atlas, atlas_payload,
                  stability_payload) -> dict:
    folds = payload["folds"]
    empirical = payload["empirical_ou_on_projection"]
    recovery = float(folds["recruited_recovery"]["s"])
    entry = float(folds["runaway_entry"]["s"])
    onset_median = float(empirical["s_from_core_median"])
    z_q10, z_q90 = map(float, empirical["z_core_q10_q90"])
    onset_q10, onset_q90 = 1.0 - z_q90, 1.0 - z_q10

    axis.axvspan(
        recovery, entry, color=COEXIST, alpha=0.42, lw=0,
        label="low/high coexistence",
    )
    axis.axvspan(
        onset_q10, onset_q90, color=OU, alpha=0.10, lw=0,
        label="OU-on onset q10–q90",
    )
    global_s = np.asarray(atlas["global_recruited__s"], float)
    global_rate = np.asarray(atlas["global_recruited__core_a_hz"], float)
    local_s = np.asarray(atlas["core_a_entry__s"], float)
    local_rate = np.asarray(atlas["core_a_entry__core_a_hz"], float)
    global_folds = np.asarray(
        atlas["global_recruited__fold_indices"], int)
    local_folds = np.asarray(atlas["core_a_entry__fold_indices"], int)
    axis.plot(
        global_s, global_rate, color=GLOBAL_BRANCH, lw=0.72, alpha=0.66,
        solid_capstyle="round", label="global-recruited family",
    )
    axis.plot(
        local_s, local_rate, color=LOCAL_BRANCH, lw=0.72, alpha=0.64,
        solid_capstyle="round", label="core-A-localized family",
    )
    axis.scatter(
        global_s[global_folds], global_rate[global_folds], s=13,
        facecolor="white", edgecolor=GLOBAL_BRANCH, lw=0.65, zorder=5)
    axis.scatter(
        local_s[local_folds], local_rate[local_folds], s=13,
        facecolor="white", edgecolor=FOLD_MARKER, lw=0.65, zorder=5)
    axis.plot(
        arrays["low_branch__s"], arrays["low_branch__core_a_hz"],
        color=LOW, lw=2.15, solid_capstyle="round", label="low outer root",
    )
    axis.plot(
        arrays["outer_high__s"], arrays["outer_high__core_a_hz"],
        color=HIGH, lw=2.15, solid_capstyle="round", label="tonic outer root",
    )
    axis.axvline(onset_median, color=OU, lw=1.3, ls=":")

    recovery_rate = float(
        folds["recruited_recovery"]["regional_e_rate_hz"]["core_a"])
    entry_rate = float(
        folds["runaway_entry"]["regional_e_rate_hz"]["core_a"])
    axis.scatter(
        [recovery, entry], [recovery_rate, entry_rate], s=34,
        c=[GLOBAL_BRANCH, LOW], edgecolor="white", lw=0.7, zorder=6,
    )
    axis.annotate(
        "low fold", xy=(entry, entry_rate), xytext=(0.305, 1.45),
        fontsize=7.0, color=LOW, ha="right",
        arrowprops={"arrowstyle": "->", "color": LOW, "lw": 0.75},
    )
    operating_roots = stability_payload["root_catalog"]
    operating_rates = np.asarray([
        root["regional_e_rate_hz"]["core_a"] for root in operating_roots
    ], float)
    axis.scatter(
        np.full(operating_rates.size, onset_median), operating_rates,
        s=20, facecolor="white", edgecolor="#262626", lw=0.75, zorder=7,
    )
    axis.text(
        onset_median + 0.006, 13.0,
        f"{operating_rates.size} coexisting full states\n"
        f"({np.unique(np.round(operating_rates, 2)).size} projected levels)",
        fontsize=6.7, color="#262626", ha="left", va="center")
    # The upper-left quadrant is empty in the computed atlas.  Use it as an
    # unboxed key so neither the paths nor the fold markers are covered.
    key_rows = (
        (430.0, LOW, 2.15, "low outer"),
        (255.0, HIGH, 2.15, "tonic outer"),
        (150.0, GLOBAL_BRANCH, 0.9, "global-recruited family"),
        (88.0, LOCAL_BRANCH, 0.9, "core-A-localized family"),
    )
    for y_value, color, width, label in key_rows:
        axis.plot([0.010, 0.027], [y_value, y_value], color=color, lw=width,
                  solid_capstyle="round", clip_on=False)
        axis.text(0.032, y_value, label, color=color, fontsize=6.3,
                  ha="left", va="center")
    axis.scatter([0.0185], [52.0], s=13, facecolor="white",
                 edgecolor=FOLD_MARKER, lw=0.65)
    axis.text(0.032, 52.0, "continuation fold", color=FOLD_MARKER,
              fontsize=6.3, ha="left", va="center")
    axis.text(
        onset_median - 0.004, 0.19, "OU-on median", color=OU,
        fontsize=6.5, rotation=90, ha="right", va="bottom")
    axis.set_yscale("log")
    axis.set_xlim(0.0, 0.405)
    axis.set_ylim(0.15, 520.0)
    axis.set_xlabel(r"Core disinhibition  $D_A=1-Z_A$", fontsize=9.0)
    axis.set_ylabel("Mean E rate within core A (Hz)", fontsize=9.0)
    axis.set_title("Core-A fixed-point branch atlas", fontsize=10.2,
                   fontweight="bold", pad=8)
    _style_axis(axis)
    return {
        "semantic": (
            "core-A per-neuron mean E rate along the symmetric spatial-Z "
            "fixed-point continuation; not a literal single-neuron bifurcation"
        ),
        "x_definition": "D_A=1-Z_A=s on Z_A=Z_B=1-s, Z_surround=1-0.70s",
        "low_state_fold_s": entry,
        "runaway_entry_s": entry,  # deprecated metadata alias
        "recruited_recovery_s": recovery,
        "ou_on_median_s": onset_median,
        "ou_on_q10_q90_s": [onset_q10, onset_q90],
        "y_scale": "log",
        "branch_line_style": (
            "all continuation loci are solid; line style does not encode "
            "stability"
        ),
        "branch_families": atlas_payload,
        "operating_root_catalog_n": int(len(operating_roots)),
        "operating_root_delay_classification": [
            {
                "root_index": int(root["root_index"]),
                "mean_e_rate_hz": float(root["mean_e_rate_hz"]),
                "core_a_rate_hz": float(
                    root["regional_e_rate_hz"]["core_a"]),
                "classification": root[
                    "delay_aware_native_dt"]["classification"],
                "growth_rate_per_ms": float(root[
                    "delay_aware_native_dt"]["growth_rate_per_ms"]),
            }
            for root in operating_roots
        ],
        "operating_ou_residence": {
            "duration_ms": float(
                stability_payload["ou_contract"]["duration_ms"]),
            "retained_initial_root_n": int(sum(
                record["retained_initial_root"]
                for record in stability_payload["ou_residence"])),
            "trajectory_n": int(len(stability_payload["ou_residence"])),
            "contract": stability_payload["ou_contract"],
        },
        "stability_boundary": (
            "delay-aware power assay and nonlinear OU residence are reported "
            "only for the tested operating section, not extrapolated along "
            "the continuation loci"
        ),
    }


def _plot_mode_map(axis, arrays, payload, model, z_map):
    mode = np.asarray(arrays["entry_fold__critical_mode_e"], float)
    energy = np.square(mode)
    energy /= max(float(np.max(energy)), 1e-12)
    image = axis.imshow(
        energy, origin="lower", extent=[0.0, model.sheet_l_mm] * 2,
        cmap="Reds", norm=PowerNorm(gamma=0.52, vmin=0.0, vmax=1.0),
        interpolation="nearest",
        aspect="equal",
    )
    core_fields = (
        np.asarray(z_map.core_a_fraction_e).reshape(model.n_grid, model.n_grid),
        np.asarray(z_map.core_b_fraction_e).reshape(model.n_grid, model.n_grid),
    )
    cell_width = float(model.sheet_l_mm / model.n_grid)
    centers = (np.arange(model.n_grid) + 0.5) * cell_width
    for index, (field, color) in enumerate(zip(core_fields, (CORE_A, CORE_B))):
        axis.contour(
            centers, centers, field, levels=[0.10], colors=[color],
            linewidths=1.35,
        )
        center = np.asarray(z_map.centers_mm[index], float)
        axis.text(
            center[0], center[1] + 1.7, f"core {'AB'[index]}", color=color,
            fontsize=7.0, fontweight="bold", ha="center", va="bottom",
        )
    fold_mode = payload["folds"]["runaway_entry_critical_mode"]
    axis.scatter(
        *fold_mode["peak_xy_mm"], marker="+", s=52, color="white",
        lw=1.4, zorder=6,
    )
    axis.set_xlim(0.0, model.sheet_l_mm)
    axis.set_ylim(0.0, model.sheet_l_mm)
    axis.set_xlabel("sheet x (mm)", fontsize=8.4)
    axis.set_ylabel("sheet y (mm)", fontsize=8.4)
    axis.set_title("Low-state fold zero mode", fontsize=9.4,
                   fontweight="bold", pad=7)
    axis.text(
        0.03, 0.97,
        f"{100.0 * fold_mode['regional_e_mode']['core_a']['fraction_of_e_mode_energy']:.1f}% in core A",
        transform=axis.transAxes, ha="left", va="top", fontsize=7.0,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.48,
              "edgecolor": "none", "pad": 2.2},
    )
    axis.tick_params(labelsize=7.1, width=0.8, length=2.7)
    axis.spines[["top", "right"]].set_visible(False)
    return image


def _plot_phase_map(axis, arrays, payload):
    z_a = np.asarray(arrays["phase__z_a"], float)
    z_b = np.asarray(arrays["phase__z_b"], float)
    state = np.asarray(arrays["phase__state_code"], int)
    if not set(np.unique(state)).issubset({2, 3}):
        raise RuntimeError("paper Panel D expects recruited-only/coexistence states")
    cmap = ListedColormap([RECRUITED_ONLY, COEXIST])
    norm = BoundaryNorm([1.5, 2.5, 3.5], cmap.N)
    half_a = float(np.diff(z_a).mean() / 2.0)
    half_b = float(np.diff(z_b).mean() / 2.0)
    axis.imshow(
        state, origin="lower", interpolation="nearest", cmap=cmap, norm=norm,
        extent=[z_a[0] - half_a, z_a[-1] + half_a,
                z_b[0] - half_b, z_b[-1] + half_b],
        aspect="equal",
    )
    empirical = payload["empirical_ou_on_projection"]
    z_median = float(empirical["z_core_median"])
    axis.plot([z_a[0], z_a[-1]], [z_a[0], z_a[-1]], color="white",
              ls="--", lw=1.0, alpha=0.9)
    axis.scatter(
        z_median, z_median, marker="D", s=45, color=OU,
        edgecolor="white", linewidth=0.8, zorder=5,
    )
    axis.text(
        z_median + 0.012, z_median - 0.014, "OU-on median",
        color=OU, fontsize=6.9, ha="left", va="top",
    )
    axis.set_xlim(z_a[0] - half_a, z_a[-1] + half_a)
    axis.set_ylim(z_b[0] - half_b, z_b[-1] + half_b)
    axis.set_xlabel(r"core-A efficacy  $Z_A$", fontsize=8.4)
    axis.set_ylabel(r"core-B efficacy  $Z_B$", fontsize=8.4)
    axis.set_title(r"Independent cores ($Z_{surround}=0.80$)", fontsize=9.4,
                   fontweight="bold", pad=7)
    axis.legend(
        handles=[
            Patch(facecolor=RECRUITED_ONLY, label="tonic root only"),
            Patch(facecolor=COEXIST, label="low + tonic roots"),
        ],
        loc="upper right", frameon=False, fontsize=6.8,
        borderaxespad=0.3, handlelength=1.3,
    )
    axis.text(
        0.02, 0.02, "lower Z = weaker inhibition",
        transform=axis.transAxes, fontsize=6.7, ha="left", va="bottom",
    )
    _style_axis(axis)


def _save_all(figure, stem: Path) -> dict:
    outputs = {}
    for suffix, options in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        path = stem.with_suffix("." + suffix)
        figure.savefig(
            path, bbox_inches="tight", pad_inches=0.025,
            facecolor="white", **options,
        )
        outputs[suffix] = {"path": str(path), "sha256": _sha256(path)}
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    base = ("/data/hfosp_topic4_fig45_artifacts/fig5/"
            "data_driven_dual_core_spatial_z")
    parser.add_argument(
        "--result", default=base
        + "/bifurcation/dualcore_spatial_z_bifurcation.json")
    parser.add_argument(
        "--state-contrast", default=base
        + "/perturbation/dualcore_rev21_state_contrast.json")
    parser.add_argument(
        "--branch-atlas", default=base
        + "/bifurcation/branch_atlas/dualcore_spatial_z_branch_atlas.json")
    parser.add_argument(
        "--stability-assay", default=base
        + "/bifurcation/stability_assay/delay_ou_operating_section.json")
    parser.add_argument(
        "--out-dir",
        default="results/paper-ready-figure/fig5_dual_core_spatial_z/figures",
    )
    args = parser.parse_args()

    result_path = Path(args.result).resolve()
    arrays_path = result_path.with_suffix(".npz")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("status") != "DUAL_CORE_SPATIAL_Z_FOLD_CHAIN_ESTABLISHED":
        raise RuntimeError("Figure 5C/D requires the verified spatial-Z fold result")
    arrays = np.load(arrays_path, allow_pickle=False)
    atlas_path = Path(args.branch_atlas).resolve()
    atlas_payload = json.loads(atlas_path.read_text(encoding="utf-8"))
    if atlas_payload.get("status") != (
            "DUAL_CORE_SPATIAL_Z_MULTIBRANCH_ATLAS_COMPLETE"):
        raise RuntimeError("Figure 5C requires the verified multibranch atlas")
    if atlas_payload["cross_family_match"]["match_established"]:
        raise RuntimeError("Figure contract unexpectedly found a branch-family match")
    atlas_arrays_path = atlas_path.with_suffix(".npz")
    atlas = np.load(atlas_arrays_path, allow_pickle=False)
    stability_path = Path(args.stability_assay).resolve()
    stability_payload = json.loads(stability_path.read_text(encoding="utf-8"))
    if stability_payload.get("status") != (
            "DUAL_CORE_SPATIAL_Z_DELAY_OU_ASSAY_COMPLETE"):
        raise RuntimeError("Figure 5C requires the delay/OU operating assay")
    if not np.isclose(
            stability_payload["operating_parameter"]["s"],
            payload["empirical_ou_on_projection"]["s_from_core_median"]):
        raise RuntimeError("branch atlas and delay/OU assay use different sections")
    model_path = Path(payload["substrate"]["model"]["path"])
    z_map_path = Path(payload["substrate"]["z_map"]["path"])
    model = load_patient_coarse_model(model_path)
    z_map = load_z_map(z_map_path)
    z_map.validate(model)
    if not np.isclose(model.sheet_l_mm / model.n_grid, 2.0):
        raise RuntimeError("Figure contract expects the audited 2-mm reduction")

    contrast_path = Path(args.state_contrast).resolve()
    contrast_arrays_path = contrast_path.with_suffix(".npz")
    contrast_meta = json.loads(contrast_path.read_text(encoding="utf-8"))
    if contrast_meta.get("status") != "REV21_FIG5_LOW_EARLY_ICTAL_CONTRAST_COMPLETE":
        raise RuntimeError("Figure 5D requires the frozen low/early-ictal contrast")
    if contrast_meta.get("substrate") != "dualcore_s39 + Joint=1.25":
        raise RuntimeError("Figure 5C and D use different substrates")
    if contrast_meta.get("candidate_id") != "rev21_ts_tz3000_ta500":
        raise RuntimeError("Figure 5D uses an unexpected slow-state candidate")
    if (int(contrast_meta.get("topology_seed", -1)) != 2542
            or int(contrast_meta.get("dynamics_seed", -1)) != 2642):
        raise RuntimeError("Figure 5D uses an unexpected realized trajectory")
    if int(contrast_meta["site_contract"]["n_total"]) != 16:
        raise RuntimeError("Figure 5D requires all 16 frozen random sites")
    if not contrast_meta.get("all_sites_retained"):
        raise RuntimeError("Figure 5D may not discard perturbation sites")
    contrast = np.load(contrast_arrays_path, allow_pickle=False)
    positions = np.asarray(contrast["positions_E"], float)
    contacts = np.asarray(contrast["contact_xy_mm"], float)
    extent = (0.0, float(model.sheet_l_mm))
    if (np.min(positions) < extent[0] - 1e-9
            or np.max(positions) > extent[1] + 1e-9):
        raise RuntimeError("Figure 5D positions fall outside the 20-mm sheet")
    low_grid = _grid_mean(
        positions, contrast["low_response_early_mean"], extent)
    early_grid = _grid_mean(
        positions, contrast["early_ictal_response_early_mean"], extent)
    nonzero = np.abs(np.concatenate([low_grid.ravel(), early_grid.ravel()]))
    nonzero = nonzero[nonzero > 1e-12]
    if nonzero.size == 0:
        raise RuntimeError("Figure 5D has no nonzero signed response")
    response_vmax = max(float(np.percentile(nonzero, 99.5)), 1e-6)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    output_dir = (ROOT / args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Standalone C for direct insertion into the main figure.
    fig_c, axis_c = plt.subplots(figsize=(4.35, 3.35))
    panel_c_meta = _plot_panel_c(
        axis_c, arrays, payload, atlas, atlas_payload, stability_payload)
    _panel_label(axis_c, "C", x=-0.18, y=1.13)
    fig_c.tight_layout(pad=0.5)
    outputs_c = _save_all(
        fig_c, output_dir / "fig5-panel-c-core-a-bifurcation")
    plt.close(fig_c)

    # Standalone D preserves the accepted paired state-response semantic.
    fig_d = plt.figure(figsize=(7.9, 3.35), facecolor="white")
    grid_d = fig_d.add_gridspec(
        1, 2, width_ratios=[1.0, 1.0], left=0.08, right=0.92,
        bottom=0.16, top=0.88, wspace=0.28,
    )
    axis_d1 = fig_d.add_subplot(grid_d[0, 0])
    axis_d2 = fig_d.add_subplot(grid_d[0, 1])
    response_image = _plot_response(
        axis_d1, contacts, low_grid, extent,
        "Low-activity mean response", response_vmax)
    _plot_response(
        axis_d2, contacts, early_grid, extent,
        "Early-ictal mean response", response_vmax, show_ylabel=False)
    _panel_label(axis_d1, "D", x=-0.24, y=1.15)
    colorbar = fig_d.colorbar(
        response_image, ax=[axis_d1, axis_d2], fraction=0.034, pad=0.035)
    colorbar.set_label(
        "mean signed probe effect\n(0–50 ms excess spikes per local E cell)",
        fontsize=7.3,
    )
    colorbar.ax.tick_params(labelsize=6.8, width=0.7, length=2.4)
    outputs_d = _save_all(
        fig_d, output_dir / "fig5-panel-d-state-response")
    plt.close(fig_d)

    # One proof sheet at the intended bottom-row proportions.
    combined = plt.figure(figsize=(12.2, 3.45), facecolor="white")
    grid = combined.add_gridspec(
        1, 3, width_ratios=[1.28, 1.0, 1.0], left=0.055, right=0.925,
        bottom=0.17, top=0.88, wspace=0.30,
    )
    axis_c = combined.add_subplot(grid[0, 0])
    axis_d1 = combined.add_subplot(grid[0, 1])
    axis_d2 = combined.add_subplot(grid[0, 2])
    _plot_panel_c(
        axis_c, arrays, payload, atlas, atlas_payload, stability_payload)
    response_image = _plot_response(
        axis_d1, contacts, low_grid, extent,
        "Low-activity mean response", response_vmax)
    _plot_response(
        axis_d2, contacts, early_grid, extent,
        "Early-ictal mean response", response_vmax, show_ylabel=False)
    _panel_label(axis_c, "C", x=-0.16, y=1.15)
    _panel_label(axis_d1, "D", x=-0.23, y=1.15)
    colorbar = combined.colorbar(
        response_image, ax=[axis_d1, axis_d2], fraction=0.034, pad=0.035)
    colorbar.set_label(
        "mean signed probe effect\n(0–50 ms excess spikes per local E cell)",
        fontsize=7.0,
    )
    colorbar.ax.tick_params(labelsize=6.7, width=0.7, length=2.4)
    outputs_cd = _save_all(
        combined, output_dir / "fig5-panels-cd-dual-core-spatial-z")
    plt.close(combined)

    # The zero mode and independent-core root catalog remain useful mechanism
    # diagnostics, but they are not Figure 5D and therefore receive no D label.
    supplement = plt.figure(figsize=(7.9, 3.35), facecolor="white")
    supplement_grid = supplement.add_gridspec(
        1, 2, width_ratios=[1.0, 1.12], left=0.08, right=0.985,
        bottom=0.16, top=0.88, wspace=0.28,
    )
    axis_s1 = supplement.add_subplot(supplement_grid[0, 0])
    axis_s2 = supplement.add_subplot(supplement_grid[0, 1])
    mode_image = _plot_mode_map(axis_s1, arrays, payload, model, z_map)
    _plot_phase_map(axis_s2, arrays, payload)
    colorbar = supplement.colorbar(
        mode_image, ax=axis_s1, fraction=0.046, pad=0.035)
    colorbar.set_label("normalized zero-mode energy", fontsize=7.3)
    colorbar.ax.tick_params(labelsize=6.8, width=0.7, length=2.4)
    outputs_supplement = _save_all(
        supplement, output_dir / "fig5-supp-spatial-z-mechanism")
    plt.close(supplement)

    phase = payload["phase_map"]
    metadata = {
        "status": "FIG5_CD_DUAL_CORE_SPATIAL_Z_CANDIDATE_RENDERED",
        "figure_role": (
            "paper-facing Figure 5C/D candidate; it must be paired only with "
            "A/B rendered from the same dualcore_s39 + Joint=1.25 substrate"
        ),
        "source": {
            "result_json": {"path": str(result_path),
                            "sha256": _sha256(result_path)},
            "arrays_npz": {"path": str(arrays_path),
                           "sha256": _sha256(arrays_path)},
            "branch_atlas_json": {"path": str(atlas_path),
                                  "sha256": _sha256(atlas_path)},
            "branch_atlas_npz": {"path": str(atlas_arrays_path),
                                 "sha256": _sha256(atlas_arrays_path)},
            "stability_assay_json": {"path": str(stability_path),
                                     "sha256": _sha256(stability_path)},
            "stability_assay_npz": {
                "path": str(stability_path.with_suffix(".npz")),
                "sha256": _sha256(stability_path.with_suffix(".npz"))},
            "model": {"path": str(model_path), "sha256": _sha256(model_path)},
            "z_map": {"path": str(z_map_path), "sha256": _sha256(z_map_path)},
            "state_contrast_json": {
                "path": str(contrast_path), "sha256": _sha256(contrast_path)},
            "state_contrast_npz": {
                "path": str(contrast_arrays_path),
                "sha256": _sha256(contrast_arrays_path)},
        },
        "substrate": payload["substrate"],
        "panel_C": panel_c_meta,
        "panel_D": {
            "left": "mean paired probe-minus-sham response in low activity",
            "right": "mean paired probe-minus-sham response at early ictal state",
            "state_times_ms": contrast_meta["state_times_ms"],
            "site_contract": contrast_meta["site_contract"],
            "dose_contract": contrast_meta["dose_contract"],
            "aggregation": contrast_meta["aggregation"],
            "response_window": contrast_meta["response_window"],
            "all_sites_retained": contrast_meta["all_sites_retained"],
            "low_n_e1_evaluable": contrast_meta["low_n_e1_evaluable"],
            "early_ictal_n_e1_evaluable": contrast_meta[
                "early_ictal_n_e1_evaluable"],
            "site_response_summary": contrast_meta["site_response_summary"],
            "shared_vmax": response_vmax,
            "claim_boundary": contrast_meta["claim_boundary"],
        },
        "mechanism_supplement": {
            "role": "diagnostic supplement; not Figure 5D",
            "phase_scan": phase,
            "critical_mode": payload["folds"]["runaway_entry_critical_mode"],
        },
        "outputs": {
            "panel_C": outputs_c,
            "panel_D": outputs_d,
            "combined_CD": outputs_cd,
            "mechanism_supplement": outputs_supplement,
        },
        "claim_boundary": payload["claim_boundary"],
    }
    metadata_path = output_dir / "fig5-dual-core-spatial-z-cd-metadata.json"
    _atomic_json(metadata, metadata_path)
    (output_dir / "README.md").write_text(
        "### fig5-panel-c-core-a-bifurcation.png / .pdf / .svg\n\n"
        "Fig.5C 候选。横轴是对称空间路径上的 core-A 失抑制 `D_A=1-Z_A`，纵轴是 core A 内每个 E 神经元的平均发放率。蓝/橙粗线是低态与 tonic 外根；棕色细线是从 tonic 外根实际续接的 global-recruited family，灰色细线是从低支零模配对根独立续接的 core-A-localized family，空心圆是全部 continuation folds。粉色线/带是 100 个 OU-on SNN 的 operational onset 中位数及 q10–q90。\n\n"
        "这不是字面意义上的单个 LIF 神经元分岔：当前确定性模型最细是 2 mm E/I population unit。图中使用 core-A per-neuron mean，是现有证据允许的局部尺度。\n\n"
        "**关注点**：主图没有 inset、没有手工补线，也没有用线型外推稳定性。global-recruited 与 core-A-localized 两族在相同参数处最近仍相差 7.29 Hz full-state RMS，所以保持分开；这不等于证明它们在未续接区间永不相连。OU-on 中位截面找到 6 个完整空间根，但投影到 core-A 均值只有 4 个高度。含全部实际 delay bins、mean gain 与 diffusion-variance gain 的稳定性和 nonlinear OU residence 只在这个工作截面报告，不扩展为整条分支定理。\n\n"
        "### fig5-panel-d-state-response.png / .pdf / .svg\n\n"
        "Fig.5D 候选。同一条冻结 dual-core SNN 轨迹上，在低态 1000 ms 与 early-ictal 2615.4 ms 使用完全相同的 16 个分层随机位置和相同 16-cell 弱脉冲。每个位置均做 exact-resume paired probe–sham，图中分别对 0–50 ms descendant-only signed response 做等权位置平均。\n\n"
        "**关注点**：左右图比较同一网络两个时点的 incremental response，不是比较两个不同网络，也不丢弃强响应位置。两侧均由少数 hotspot 主导，且 early-ictal sham 已处于高态（0/16 可再作 ignition test），所以该图不是跨 seed 易感性或触发概率估计。\n\n"
        "### fig5-panels-cd-dual-core-spatial-z.png / .pdf / .svg\n\n"
        "C/D 同行 proof sheet，尺寸比例按 Fig.5 下排版准备。C 是多分支 fixed-point atlas，D 保持同位置扰动的 low/early-ictal 状态响应。只允许与同一 `dualcore_s39 + Joint=1.25` 底物重画的 A/B 合并；不能直接与旧 `joint_04_control seed1801` A/B 拼成同一实验。\n\n"
        "**关注点**：C 回答 core-A 局部群体快系统有什么分支，D 回答相同局部扰动在 runaway 前后如何产生不同空间响应。\n\n"
        "### fig5-supp-spatial-z-mechanism.png / .pdf / .svg\n\n"
        "机制补图。左图是 low-state fixed-point fold 零模在 20 mm 双核 sheet 上的位置；右图是固定 `Z_surround=0.80` 后独立扫描 `Z_A` 与 `Z_B` 的有限 multi-start root catalog。它们解释低根消失如何落到空间，但不再冒充 stable/runaway 分界或正式 Fig.5D。\n\n"
        "**关注点**：零模 93.7% 的能量位于 core A；相图只报告有限 root catalog，不把未找到的根写成数学不存在。\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": metadata["status"],
        "out_dir": str(output_dir),
        "metadata": str(metadata_path),
    }, indent=2))


if __name__ == "__main__":
    main()
