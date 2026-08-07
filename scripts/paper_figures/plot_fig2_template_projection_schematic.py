#!/usr/bin/env python3
"""Paper-ready E1146 schematic: 3-D template axes to shared 2-D fields.

This figure is intentionally built from the frozen interictal-only E1146
artifact used by the downstream field panels.  It does not refit coordinates,
axes, a plane, support, or template ranks.  The two final rank fields reuse the
canonical Topic 5 field renderer with the locked 6-mm display bandwidth.

The left MRI surface is context only.  Epilepsiae contacts are displayed in
the MNI152 1-mm coordinate system documented by ``seeg_coord_loader``; they
are not described as subject-native coordinates.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
from skimage.measure import marching_cubes


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_topic5_interictal_template_ab_fields import (  # noqa: E402
    build_interictal_ab_panel_payloads,
    draw_interictal_rank_field_panel,
)
from src.seeg_coord_loader import load_subject_coords  # noqa: E402


SUBJECT_ID = "epilepsiae_1146"
DISPLAY_LABEL = "E1146"
INPUT_ARTIFACT = (
    ROOT
    / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
    / f"{SUBJECT_ID}.json"
)
OUTPUT_DIR = (
    ROOT / "results/paper-ready-figure/fig2_template_projection_schematic/figures"
)
DISPLAY_SIGMA_MM = 6.0
TA_COLOR = "#B2182B"
TB_COLOR = "#2166AC"
INK = "#202428"
MID_GREY = "#6F777D"
LIGHT_GREY = "#D5DADD"
PLANE_COLOR = "#C8D4DD"


def _load_case() -> tuple[dict, dict, dict, str, Path]:
    record = json.loads(INPUT_ARTIFACT.read_text())
    if record.get("subject_id") != SUBJECT_ID or record.get("status") != "ok":
        raise ValueError(f"unexpected or unavailable record: {record.get('subject_id')}")
    pair = record.get("axis_pair") or {}
    field = record.get("interictal_field") or {}
    relation = pair.get("relation") or {}
    if not bool(pair.get("geometry_2d_supported")):
        raise ValueError("E1146 does not meet the frozen 2-D geometry contract")
    if not bool(relation.get("collinear")):
        raise ValueError("E1146 no longer meets the frozen shared-plane criterion")
    if not all(key in (field.get("field_models") or {}) for key in ("shared_a", "shared_b")):
        raise ValueError("E1146 frozen artifact is missing shared TA/TB fields")

    dat_a, dat_b, mode = build_interictal_ab_panel_payloads(
        record, display_sigma_mm=DISPLAY_SIGMA_MM,
    )
    if mode != "shared":
        raise ValueError(f"expected shared-plane E1146 artifact, found {mode!r}")

    # The frozen coordinates remain the plotting truth.  The loader is used
    # only to resolve and verify MRI provenance for the contextual surface.
    coord = load_subject_coords(
        "epilepsiae", "1146", list(field["contact_order"]),
    )
    if coord.coord_space != "mni152_1mm":
        raise ValueError(f"unexpected E1146 coordinate space: {coord.coord_space}")
    mri_path = Path(str(coord.provenance["affine_path"]))
    if not mri_path.exists():
        raise FileNotFoundError(mri_path)
    return record, dat_a, dat_b, coord.coord_space, mri_path


def _contact_number(name: str) -> int:
    match = re.search(r"(\d+)$", str(name))
    return int(match.group(1)) if match else 0


def _shaft_indices(names: Sequence[str], shafts: Sequence[str]) -> list[np.ndarray]:
    names_arr = np.asarray(names, dtype=object)
    shafts_arr = np.asarray(shafts, dtype=object)
    groups = []
    for shaft in sorted(set(shafts_arr.tolist())):
        idx = np.where(shafts_arr == shaft)[0]
        idx = idx[np.argsort([_contact_number(str(names_arr[i])) for i in idx])]
        groups.append(idx)
    return groups


def _brain_surface(mri_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a light contextual surface from the patient MRI on the MNI grid."""
    img = nib.load(str(mri_path))
    volume = np.asarray(img.dataobj).squeeze()
    if volume.ndim != 3:
        raise ValueError(f"expected one 3-D MRI volume, found shape={volume.shape}")
    positive = volume[np.isfinite(volume) & (volume > 0)]
    if positive.size == 0:
        raise ValueError("MRI contains no positive voxels")
    # A low, data-relative threshold gives a stable outer brain context.  Keep
    # only the largest connected component so isolated bright noise cannot
    # enter the paper-facing surface.
    threshold = float(np.percentile(positive, 8.0))
    mask = np.asarray(volume >= threshold, dtype=bool)
    labels, n_labels = ndimage.label(mask)
    if n_labels < 1:
        raise ValueError("MRI threshold produced no connected component")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    mask = labels == int(np.argmax(sizes))
    mask = ndimage.binary_fill_holes(mask)
    vertices_vox, faces, _, _ = marching_cubes(
        mask.astype(np.uint8), level=0.5, step_size=4, allow_degenerate=False,
    )
    vertices_mm = nib.affines.apply_affine(img.affine, vertices_vox)
    return vertices_mm, np.asarray(faces, int), threshold


def _projected_basis(record: Mapping[str, object], transverse_sign: int) -> dict:
    plane = record["interictal_field"]["planes"]["shared"]
    u = np.asarray(plane["u"], float)
    w = int(transverse_sign) * np.asarray(plane["w"], float)
    normal = np.cross(u, w)
    normal /= np.linalg.norm(normal)
    return {
        "origin": np.asarray(plane["origin"], float),
        "u": u,
        "w": w,
        "normal": normal,
        "scale_mm": float(plane["scale_mm"]),
        "analysis_sigma_mm": float(plane["sigma"]) * float(plane["scale_mm"]),
    }


def _axis_2d(axis: Sequence[float], basis: Mapping[str, np.ndarray]) -> np.ndarray:
    vector = np.asarray(axis, float)
    out = np.asarray([vector @ basis["u"], vector @ basis["w"]], float)
    return out / np.linalg.norm(out)


def _crop_brain_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    centers = vertices[faces].mean(axis=1)
    use = np.all(np.abs(centers - center[None, :]) <= radius * 1.08, axis=1)
    return vertices, faces[use]


def _draw_native_geometry(
    ax,
    record: Mapping[str, object],
    dat_a: Mapping[str, object],
    basis: Mapping[str, np.ndarray],
    brain_vertices: np.ndarray,
    brain_faces: np.ndarray,
) -> None:
    coords = np.asarray(record["coords"], float)
    names = [str(x) for x in record["names"]]
    shafts = [str(x) for x in record["shafts"]]
    origin = np.asarray(basis["origin"], float)
    u, w, normal = basis["u"], basis["w"], basis["normal"]
    xs = np.asarray(dat_a["xs"], float)
    ys = np.asarray(dat_a["ys"], float)
    projected = origin + np.outer(xs, u) + np.outer(ys, w)

    radius = float(max(34.0, 0.5 * np.max(np.ptp(coords, axis=0)) + 22.0))
    vertices, faces = _crop_brain_faces(brain_vertices, brain_faces, coords.mean(0), radius)
    if len(faces):
        mesh = Poly3DCollection(
            vertices[faces],
            facecolor=(0.70, 0.72, 0.75, 0.075),
            edgecolor=(0.48, 0.50, 0.53, 0.030),
            linewidth=0.08,
            rasterized=True,
            zorder=0,
        )
        ax.add_collection3d(mesh)

    xlim = dat_a["frame"]["xlim"]
    ylim = dat_a["frame"]["ylim"]
    corners = np.asarray(
        [
            origin + xlim[0] * u + ylim[0] * w,
            origin + xlim[1] * u + ylim[0] * w,
            origin + xlim[1] * u + ylim[1] * w,
            origin + xlim[0] * u + ylim[1] * w,
        ]
    )
    plane = Poly3DCollection(
        [corners], facecolor=PLANE_COLOR, edgecolor="#87939C",
        linewidth=0.65, alpha=0.24, zorder=2,
    )
    ax.add_collection3d(plane)

    for idx in _shaft_indices(names, shafts):
        ax.plot(*coords[idx].T, color="#49545B", lw=1.05, alpha=0.92, zorder=5)
    for point, foot in zip(coords, projected):
        ax.plot(
            [point[0], foot[0]], [point[1], foot[1]], [point[2], foot[2]],
            color="#9BA3A8", lw=0.45, ls=(0, (1.6, 1.4)), alpha=0.9, zorder=3,
        )
    ax.scatter(
        *projected.T, s=12, facecolor="#C7CDD1", edgecolor="white",
        linewidth=0.45, alpha=0.88, depthshade=False, zorder=4,
    )
    ax.scatter(
        *coords.T, s=28, facecolor="white", edgecolor=INK,
        linewidth=0.8, depthshade=False, zorder=6,
    )

    axis_a = np.asarray(record["axis_pair"]["axis_a"]["u"], float)
    axis_b = np.asarray(record["axis_pair"]["axis_b"]["u"], float)
    arrow_length = 27.0
    for vector, color, label, offset in (
        (axis_a, TA_COLOR, "TA", 1.8 * normal),
        (axis_b, TB_COLOR, "TB", -1.8 * normal),
    ):
        start = origin - 0.48 * arrow_length * vector + offset
        delta = arrow_length * vector
        ax.quiver(
            *start, *delta, color=color, lw=2.4, arrow_length_ratio=0.14,
            normalize=False, zorder=8,
        )
        tip = start + delta
        ax.text(*tip, label, color=color, fontsize=7.5, fontweight="bold", zorder=9)

    shared_extent = 20.5
    line = np.vstack((origin - shared_extent * u, origin + shared_extent * u))
    ax.plot(*line.T, color="#4F565B", lw=0.8, ls=(0, (2.2, 2.2)), zorder=7)

    # Use an oblique orthographic view: the patient plane remains readable but
    # its 3-D embedding and the tiny projection residual are still visible.
    camera = normal + 0.34 * w + 0.18 * u
    camera /= np.linalg.norm(camera)
    elev = float(np.degrees(np.arcsin(np.clip(camera[2], -1.0, 1.0))))
    azim = float(np.degrees(np.arctan2(camera[1], camera[0])))
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    center = coords.mean(0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()

    relation = record["axis_pair"]["relation"]
    ax.text2D(
        0.02, 0.04,
        f"line angle {float(relation['line_angle_deg']):.0f}°   |cos|={float(relation['abs_cosine']):.2f}   arrows: early → late",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color=MID_GREY,
    )


def _draw_projected_plane(
    ax,
    record: Mapping[str, object],
    dat_a: Mapping[str, object],
    basis: Mapping[str, np.ndarray],
) -> int:
    xs = np.asarray(dat_a["xs"], float)
    ys = np.asarray(dat_a["ys"], float)
    names = [str(x) for x in dat_a["names"]]
    shafts = [str(x) for x in record["shafts"]]
    xlim = dat_a["frame"]["xlim"]
    ylim = dat_a["frame"]["ylim"]

    # Show one true Gaussian footprint without turning this middle panel into a
    # third data field.  The highlighted point is chosen deterministically near
    # the center of the shared plane.
    kernel_index = int(np.argmin(xs**2 + ys**2))
    sigma = DISPLAY_SIGMA_MM
    gx = np.linspace(*xlim, 220)
    gy = np.linspace(*ylim, 220)
    xx, yy = np.meshgrid(gx, gy)
    gaussian = np.exp(
        -((xx - xs[kernel_index]) ** 2 + (yy - ys[kernel_index]) ** 2) / (2.0 * sigma**2)
    )
    masked = np.ma.masked_where(gaussian < 0.055, gaussian)
    ax.imshow(
        masked, origin="lower", extent=[*xlim, *ylim], cmap="Greys",
        vmin=0.0, vmax=1.0, alpha=0.17, interpolation="bilinear", zorder=0,
    )
    ax.contour(
        xx, yy, gaussian, levels=[np.exp(-2.0), np.exp(-0.5)],
        colors=["#AAB2B7", "#737C82"], linewidths=[0.65, 0.8],
        linestyles=[(0, (2.3, 1.8)), (0, (3.2, 1.8))], zorder=1,
    )

    for idx in _shaft_indices(names, shafts):
        ax.plot(xs[idx], ys[idx], color="#80888D", lw=1.0, zorder=2)

    axis_a = np.asarray(record["axis_pair"]["axis_a"]["u"], float)
    axis_b = np.asarray(record["axis_pair"]["axis_b"]["u"], float)
    vec_a = _axis_2d(axis_a, basis)
    vec_b = _axis_2d(axis_b, basis)
    length = 20.0
    for vector, color, label in ((vec_a, TA_COLOR, "TA"), (vec_b, TB_COLOR, "TB")):
        offset = np.asarray([0.0, 4.0])
        start = -0.44 * length * vector + offset
        end = 0.56 * length * vector + offset
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=8.5,
                lw=1.65, color=color, alpha=0.95, zorder=3,
            )
        )
        label_xy = end + 0.7 * vector + np.asarray([0.0, 1.5])
        ax.text(
            label_xy[0], label_xy[1], label,
            color=color, fontsize=7.2, fontweight="bold", ha="center", va="center", zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
        )
    ax.axhline(0.0, color="#5E666B", lw=0.7, ls=(0, (2.4, 2.1)), zorder=0)
    ax.axvline(0.0, color="#B9BFC3", lw=0.55, zorder=0)
    ax.scatter(
        xs, ys, s=37, facecolor="white", edgecolor=INK,
        linewidth=0.75, zorder=4,
    )
    ax.scatter(
        [xs[kernel_index]], [ys[kernel_index]], s=69, facecolor="none",
        edgecolor="#616A70", linewidth=1.0, zorder=5,
    )
    ax.annotate(
        "Gaussian\nkernel", xy=(xs[kernel_index] + sigma, ys[kernel_index] + sigma),
        xytext=(xlim[1] - 1.6, ylim[0] + 2.6), ha="right", va="bottom",
        fontsize=6.2, color=MID_GREY,
        arrowprops={"arrowstyle": "-", "color": "#8E969B", "lw": 0.65},
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-20, 0, 20])
    ax.set_yticks([-10, 0, 10, 20])
    ax.tick_params(labelsize=6.0, length=2.2, pad=1.5)
    ax.set_xlabel("shared TA axis (mm)", fontsize=7.0, labelpad=2.0)
    ax.set_ylabel("transverse (mm)", fontsize=7.0, labelpad=2.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#60676B")
    ax.set_title("")
    ax.text(
        0.02, 0.97, "contact-wise rank or activation", transform=ax.transAxes,
        ha="left", va="top", fontsize=6.3, color=MID_GREY,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.7},
        zorder=7,
    )
    return kernel_index


def _draw_field_pair(fig, ax_a, ax_b, cax, dat_a: Mapping[str, object], dat_b: Mapping[str, object]) -> None:
    draw_interictal_rank_field_panel(
        ax_a, dat_a, "TA", compact=True, panel_title="TA rank field",
        contact_outline_lw=1.0, contact_size=30, show_template_tag=False,
    )
    draw_interictal_rank_field_panel(
        ax_b, dat_b, "TB", compact=True, panel_title="TB rank field",
        contact_outline_lw=1.0, contact_size=30, show_template_tag=False,
    )
    ax_a.set_title("TA rank field", color=TA_COLOR, fontsize=7.4, fontweight="bold", pad=2.0)
    ax_b.set_title("TB rank field", color=TB_COLOR, fontsize=7.4, fontweight="bold", pad=2.0)
    scalar = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0.0, 14.0))
    colorbar = fig.colorbar(scalar, cax=cax)
    colorbar.set_ticks([0.0, 7.0, 14.0])
    colorbar.set_ticklabels(["0\nearly", "7", "14\nlate"])
    colorbar.ax.tick_params(labelsize=5.8, length=2.0, pad=1.5)
    colorbar.ax.set_title("rank", fontsize=6.6, pad=3.0)
    colorbar.outline.set_linewidth(0.65)


def _flow_annotation(fig, left_ax, right_ax, label: str, *, y: float) -> None:
    left = left_ax.get_position()
    right = right_ax.get_position()
    x0 = left.x1 + 0.010
    x1 = right.x0 - 0.010
    fig.add_artist(
        FancyArrowPatch(
            (x0, y), (x1, y), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=8.0, lw=0.85,
            color="#7C8489", clip_on=False,
        )
    )
    fig.text(
        0.5 * (x0 + x1), y + 0.028, label,
        ha="center", va="bottom", fontsize=6.0, color=MID_GREY,
    )


def _metadata(
    record: Mapping[str, object],
    dat_a: Mapping[str, object],
    basis: Mapping[str, np.ndarray],
    coord_space: str,
    mri_path: Path,
    mri_threshold: float,
    kernel_index: int,
) -> dict:
    coords = np.asarray(record["coords"], float)
    origin = np.asarray(basis["origin"], float)
    residual = (coords - origin) @ np.asarray(basis["normal"], float)
    relation = record["axis_pair"]["relation"]
    pair_bootstrap = record["axis_pair"]["pair_bootstrap"]
    return {
        "schema_version": "fig2_template_projection_schematic_v1",
        "subject_id": SUBJECT_ID,
        "display_label": DISPLAY_LABEL,
        "input_artifact": str(INPUT_ARTIFACT.resolve()),
        "input_contract": record.get("contract"),
        "input_fingerprint_algorithm": record["interictal_field"]["fingerprint_algorithm"],
        "input_fingerprint_sha256": record["interictal_field"]["fingerprint_sha256"],
        "coordinate_space": coord_space,
        "mri_context_path": str(mri_path.resolve()),
        "mri_surface_threshold": mri_threshold,
        "n_contacts": len(record["names"]),
        "contact_order": record["names"],
        "shafts": record["shafts"],
        "axis_contract": {
            "definition": record["axis_definition"],
            "direction": record["axis_direction_convention"],
            "u_ta": record["axis_pair"]["axis_a"]["u"],
            "u_tb": record["axis_pair"]["axis_b"]["u"],
            "cos_ta_tb": relation["cosine"],
            "abs_cos_ta_tb": relation["abs_cosine"],
            "directed_angle_deg": float(np.degrees(np.arccos(relation["cosine"]))),
            "line_angle_deg": relation["line_angle_deg"],
            "point_estimate_collinear": relation["collinear"],
            "pair_bootstrap_p_collinear": pair_bootstrap["p_collinear"],
            "pair_bootstrap_p_sign_stable": pair_bootstrap["p_sign_stable"],
            "pair_bootstrap_robust_collinear": pair_bootstrap["robust_collinear"],
        },
        "shared_plane": {
            "u": np.asarray(basis["u"]).tolist(),
            "w_after_display_sign": np.asarray(basis["w"]).tolist(),
            "normal": np.asarray(basis["normal"]).tolist(),
            "origin_mm": np.asarray(basis["origin"]).tolist(),
            "transverse_sign": int(dat_a["transverse_sign"]),
            "normal_residual_max_abs_mm": float(np.max(np.abs(residual))),
            "normal_residual_rms_mm": float(np.sqrt(np.mean(residual**2))),
        },
        "kernel": {
            "analysis_sigma_mm_frozen": basis["analysis_sigma_mm"],
            "display_sigma_mm": DISPLAY_SIGMA_MM,
            "display_kernel_exemplar_contact": record["names"][kernel_index],
            "normalized_weighted_field": "F(p)=sum_i support_i*K_sigma(p-p_i)*value_i / sum_i support_i*K_sigma(p-p_i)",
        },
        "rendering": {
            "final_field_renderer": "scripts.plot_topic5_interictal_template_ab_fields.draw_interictal_rank_field_panel",
            "colormap": "viridis",
            "rank_scale": [0, 14],
            "png_dpi": 400,
            "pdf_fonttype": 42,
        },
        "claim_boundary": (
            "Patient-specific illustration of a frozen interictal coordinate transform and rank-field "
            "display. The continuous surface is a support-limited interpolation, not measurement in "
            "unsampled tissue and not cohort or mechanism evidence. The same frozen plane can later "
            "receive exact-name-aligned contact activation values without refitting geometry."
        ),
    }


def plot() -> tuple[Path, Path, Path, Path]:
    record, dat_a, dat_b, coord_space, mri_path = _load_case()
    basis = _projected_basis(record, int(dat_a["transverse_sign"]))
    brain_vertices, brain_faces, mri_threshold = _brain_surface(mri_path)

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(7.25, 3.24), facecolor="white")
        gs = gridspec.GridSpec(
            2, 6, figure=fig,
            width_ratios=[2.48, 0.24, 1.72, 0.26, 1.38, 0.085],
            height_ratios=[1.0, 1.0],
            left=0.015, right=0.965, top=0.79, bottom=0.17,
            wspace=0.08, hspace=0.24,
        )
        ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
        ax_plane = fig.add_subplot(gs[:, 2])
        ax_field_a = fig.add_subplot(gs[0, 4])
        ax_field_b = fig.add_subplot(gs[1, 4])
        cax = fig.add_subplot(gs[:, 5])

        _draw_native_geometry(
            ax_3d, record, dat_a, basis, brain_vertices, brain_faces,
        )
        kernel_index = _draw_projected_plane(ax_plane, record, dat_a, basis)
        _draw_field_pair(fig, ax_field_a, ax_field_b, cax, dat_a, dat_b)

        fig.canvas.draw()
        _flow_annotation(fig, ax_3d, ax_plane, "project", y=0.81)
        _flow_annotation(fig, ax_plane, ax_field_a, "smooth", y=0.48)
        left_box = ax_3d.get_position()
        plane_box = ax_plane.get_position()
        right_box = ax_field_a.get_position()
        header_y = 0.955
        fig.text(
            left_box.x0, header_y, "1  Template axes in 3-D",
            ha="left", va="top", fontsize=9.0, fontweight="bold", color=INK,
        )
        fig.text(
            left_box.x0, header_y - 0.055, f"{DISPLAY_LABEL} · 15 contacts, 2 shafts",
            ha="left", va="top", fontsize=6.8, color=MID_GREY,
        )
        fig.text(
            plane_box.x0, header_y, "2  Shared patient plane",
            ha="left", va="top", fontsize=9.0, fontweight="bold", color=INK,
        )
        fig.text(
            right_box.x0, header_y, "3  Continuous fields",
            ha="left", va="top", fontsize=9.0, fontweight="bold", color=INK,
        )
        fig.text(
            0.5 * (ax_plane.get_position().x0 + ax_field_b.get_position().x1),
            0.055,
            r"$F(\mathbf{p})=\frac{\sum_i s_i K_\sigma(\mathbf{p}-\mathbf{p}_i)v_i}"
            r"{\sum_i s_i K_\sigma(\mathbf{p}-\mathbf{p}_i)}$",
            ha="center", va="center", fontsize=7.0, color="#42484C",
        )
        fig.text(
            0.018, 0.055, "E1146 · patient MRI on MNI152 grid",
            ha="left", va="center", fontsize=6.0, color="#858C91",
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        png = OUTPUT_DIR / "fig2_E1146_template_projection_schematic.png"
        pdf = OUTPUT_DIR / "fig2_E1146_template_projection_schematic.pdf"
        svg = OUTPUT_DIR / "fig2_E1146_template_projection_schematic.svg"
        metadata_path = OUTPUT_DIR / "fig2_E1146_template_projection_schematic_metadata.json"
        fig.savefig(png, dpi=400, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        fig.savefig(svg, facecolor="white")
        plt.close(fig)

    metadata = _metadata(
        record, dat_a, basis, coord_space, mri_path, mri_threshold, kernel_index,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return png, pdf, svg, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for path in plot():
        print(f"[done] {path}")


if __name__ == "__main__":
    main()
