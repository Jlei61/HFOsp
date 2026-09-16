#!/usr/bin/env python3
"""Spatial-method panel: implantation overview, local T1, and 2-D projection.

The upper-left overview is the author-supplied Y9 implantation image.  The
remaining three stages use the frozen E1146 MNI152 1-mm spatial bundle and
show local T1 anatomy, the 3-D contact-to-plane operation, and the
support-limited 2-D contact coverage.  The explicit subject switch prevents
the overview from being mistaken for a same-subject zoom sequence.
"""
from __future__ import annotations

import argparse
import hashlib
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

from scipy.ndimage import map_coordinates
from matplotlib import gridspec
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paper_figure_typography import (  # noqa: E402
    ILLUSTRATIVE_PANEL_TYPOGRAPHY,
    LOCKED_PANEL_TYPOGRAPHY_POLICY,
    apply_panel_aware_figure_typography,
)
from src.seeg_coord_loader import (  # noqa: E402
    enumerate_subject_all_channels,
    load_subject_coords,
)
from scripts.paper_figures.patient_public_labels import public_patient_label  # noqa: E402


SUBJECT_ID = "epilepsiae_1146"
DISPLAY_LABEL = public_patient_label(*SUBJECT_ID.split("_", 1))
INPUT_ARTIFACT = (
    ROOT
    / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
    / f"{SUBJECT_ID}.json"
)
MNI_BUNDLE = ROOT / "exports/epilepsiae_1146_mni_bundle"
T1_PATH = MNI_BUNDLE / "epilepsiae_1146_T1_mni152_1mm.nii.gz"
T1_MANIFEST = MNI_BUNDLE / "manifest.json"
IMPLANTATION_OVERVIEW = (
    ROOT / "scripts/paper_figures/assets/fig2_y9_implantation_overview.png"
)
IMPLANTATION_OVERVIEW_SHA256 = (
    "631a4b737ed9c779ed74ce8d799a17b53b6f83b704c7e79ff0ece01955b717a2"
)
OUTPUT_DIR = (
    ROOT
    / "results/paper-ready-figure/fig2/figures"
)

DISPLAY_SIGMA_MM = 6.0
NORMAL_DISPLAY_EXAGGERATION = 3.0
DISPLAY_PLANE_Z = -1.15
GEOMETRY_PADDING_FRACTION = 0.18
GEOMETRY_BOX_ZOOM = 1.00
PROJECTION_PANEL_SCALE = 0.84

INK = "#273238"
MID_GREY = "#77848A"
CONTEXT_CONTACT = "#D9DEE0"
SELECTED_CONTACT = "#2B7C87"
SELECTED_DARK = "#225F68"
PLANE_FACE = "#D9ECEE"
PLANE_EDGE = "#7E9CA2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _draw_implantation_overview(ax) -> None:
    """Render the frozen author-supplied Y9 implantation overview."""
    if not IMPLANTATION_OVERVIEW.exists():
        raise FileNotFoundError(IMPLANTATION_OVERVIEW)
    digest = _sha256(IMPLANTATION_OVERVIEW)
    if digest != IMPLANTATION_OVERVIEW_SHA256:
        raise ValueError(
            "Y9 implantation overview differs from the accepted source asset: "
            f"{digest}"
        )
    image = plt.imread(IMPLANTATION_OVERVIEW)
    if image.shape[:2] != (392, 488):
        raise ValueError(f"unexpected Y9 overview dimensions: {image.shape[:2]}")
    ax.imshow(image, interpolation="lanczos")
    height, width = image.shape[:2]
    ax.set_xlim(-0.045 * width, 1.045 * width)
    ax.set_ylim(1.045 * height, -0.045 * height)
    ax.set_axis_off()


def _contact_number(name: str) -> int:
    match = re.search(r"(\d+)$", str(name))
    return int(match.group(1)) if match else 0


def _shaft_indices(names: Sequence[str], shafts: Sequence[str]) -> list[np.ndarray]:
    names_arr = np.asarray(names, dtype=object)
    shafts_arr = np.asarray(shafts, dtype=object)
    groups: list[np.ndarray] = []
    for shaft in sorted(set(shafts_arr.tolist())):
        idx = np.where(shafts_arr == shaft)[0]
        idx = idx[np.argsort([_contact_number(str(names_arr[i])) for i in idx])]
        groups.append(idx)
    return groups


def _shaft_label_position(
    display_points: np.ndarray,
    ordering_points: np.ndarray,
    idx: np.ndarray,
    *,
    offset: float,
) -> np.ndarray:
    """Place a shaft name just beyond its minimum projected-x endpoint."""
    endpoint = int(idx[np.argmin(ordering_points[idx, 0])])
    others = idx[idx != endpoint]
    nearest = int(
        others[
            np.argmin(
                np.linalg.norm(
                    ordering_points[others] - ordering_points[endpoint], axis=1,
                )
            )
        ]
    )
    outward = display_points[endpoint] - display_points[nearest]
    outward /= np.linalg.norm(outward)
    return display_points[endpoint] + float(offset) * outward


def _load_case() -> tuple[dict, list[str], np.ndarray, str]:
    record = json.loads(INPUT_ARTIFACT.read_text())
    if record.get("subject_id") != SUBJECT_ID or record.get("status") != "ok":
        raise ValueError(f"unexpected or unavailable record: {record.get('subject_id')}")
    pair = record.get("axis_pair") or {}
    field = record.get("interictal_field") or {}
    if not bool(pair.get("geometry_2d_supported")):
        raise ValueError("E1146 no longer meets the frozen two-dimensional geometry contract")
    if not bool((pair.get("relation") or {}).get("collinear")):
        raise ValueError("E1146 no longer meets the frozen shared-plane criterion")
    if "shared" not in (field.get("planes") or {}):
        raise ValueError("E1146 frozen artifact is missing its shared patient plane")

    all_names = enumerate_subject_all_channels("epilepsiae", "1146")
    coord = load_subject_coords("epilepsiae", "1146", all_names)
    if coord.coord_space != "mni152_1mm" or coord.coord_units != "mm":
        raise ValueError(
            f"unexpected E1146 coordinate contract: {coord.coord_space}/{coord.coord_units}"
        )
    if not bool(np.all(coord.mapped_mask_in_requested_order)):
        raise ValueError("one or more E1146 invasive contacts lack coordinates")
    selected = set(str(x) for x in field["contact_order"])
    if not selected.issubset(set(all_names)):
        raise ValueError("frozen E1146 field contacts are not contained in the implantation")
    return (
        record,
        all_names,
        np.asarray(coord.coords_array_in_requested_order, float),
        coord.coord_space,
    )


def _load_t1() -> tuple[nib.Nifti1Image, np.ndarray, dict]:
    if not T1_PATH.exists() or not T1_MANIFEST.exists():
        raise FileNotFoundError("E1146 MNI spatial bundle is incomplete")
    manifest = json.loads(T1_MANIFEST.read_text())
    if manifest.get("coord_space") != "mni152_1mm":
        raise ValueError("E1146 T1 manifest is not on the MNI152 1-mm grid")
    image = nib.load(str(T1_PATH))
    data = np.asarray(image.dataobj, dtype=float)
    expected_shape = tuple(int(x) for x in manifest["mni152_1mm_shape"])
    if data.shape != expected_shape:
        raise ValueError(f"T1 shape mismatch: {data.shape} != {expected_shape}")
    if not np.allclose(image.affine, manifest["mni152_1mm_affine"], atol=1e-6):
        raise ValueError("T1 affine differs from its frozen bundle manifest")
    return image, data, manifest


def _canonical_transverse_sign(vector: Sequence[float]) -> int:
    """Resolve the arbitrary plane-basis sign from geometry alone."""
    vec = np.asarray(vector, float)
    if vec.shape != (3,) or not np.isfinite(vec).all() or np.linalg.norm(vec) <= 0:
        raise ValueError("transverse basis must be one finite nonzero 3-D vector")
    dominant = int(np.argmax(np.abs(vec)))
    return 1 if vec[dominant] >= 0 else -1


def _basis(record: Mapping[str, object], transverse_sign: int) -> dict[str, np.ndarray]:
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
    }


def _to_basis(points: np.ndarray, basis: Mapping[str, np.ndarray]) -> np.ndarray:
    rel = np.asarray(points, float) - np.asarray(basis["origin"], float)
    return np.column_stack(
        [rel @ basis["u"], rel @ basis["w"], rel @ basis["normal"]]
    )


def _fit_straight_shaft_points(
    coords: np.ndarray, names: Sequence[str], shafts: Sequence[str],
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Place displayed carrier contacts on each shaft's best-fit 3-D line."""
    points = np.asarray(coords, float)
    fitted = np.empty_like(points)
    stats: dict[str, dict[str, float]] = {}
    for idx in _shaft_indices(names, shafts):
        shaft = str(shafts[int(idx[0])])
        local = points[idx]
        origin = local.mean(axis=0)
        _, _, vt = np.linalg.svd(local - origin, full_matrices=False)
        direction = vt[0]
        along = (local - origin) @ direction
        fit = origin + np.outer(along, direction)
        fitted[idx] = fit
        residual = np.linalg.norm(local - fit, axis=1)
        stats[shaft] = {
            "n_contacts": int(len(idx)),
            "max_residual_mm": float(np.max(residual)),
            "rms_residual_mm": float(np.sqrt(np.mean(residual**2))),
        }
    return fitted, stats


def _local_geometry(
    record: Mapping[str, object],
    basis: Mapping[str, np.ndarray],
    context_names: Sequence[str],
    context_coords: np.ndarray,
) -> dict[str, object]:
    all_names = np.asarray([str(x) for x in context_names], dtype=object)
    all_shafts = np.asarray(
        [re.sub(r"\d+$", "", str(x)) for x in all_names], dtype=object,
    )
    local_mask = np.isin(all_shafts, ["ICL", "SCL"])
    names = all_names[local_mask]
    shafts = all_shafts[local_mask]
    world = np.asarray(context_coords, float)[local_mask]
    measured = _to_basis(world, basis)
    fitted, fit_stats = _fit_straight_shaft_points(
        measured, names.tolist(), shafts.tolist(),
    )
    world_fitted, world_fit_stats = _fit_straight_shaft_points(
        world, names.tolist(), shafts.tolist(),
    )
    selected_set = set(str(x) for x in record["interictal_field"]["contact_order"])
    selected = np.asarray([str(name) in selected_set for name in names], bool)
    return {
        "names": names,
        "shafts": shafts,
        "measured": measured,
        "fitted": fitted,
        "world": world,
        "world_fitted": world_fitted,
        "selected": selected,
        "fit_stats": fit_stats,
        "world_fit_stats": world_fit_stats,
    }


def _selected_support(
    record: Mapping[str, object], geometry: Mapping[str, object],
) -> np.ndarray:
    field = record["interictal_field"]
    source_names = [str(x) for x in field["contact_order"]]
    mean_support = 0.5 * (
        np.asarray(field["support_a"], float) + np.asarray(field["support_b"], float)
    )
    by_name = dict(zip(source_names, mean_support, strict=True))
    return np.asarray(
        [float(by_name.get(str(name), 0.0)) for name in geometry["names"]],
        dtype=float,
    )


def _limits_2d(xy: np.ndarray, pad_fraction: float = 0.12) -> tuple[tuple[float, float], tuple[float, float]]:
    xy = np.asarray(xy, float)
    mins = np.nanmin(xy, axis=0)
    maxs = np.nanmax(xy, axis=0)
    span = np.maximum(maxs - mins, 1.0)
    pad = pad_fraction * span
    return (
        (float(mins[0] - pad[0]), float(maxs[0] + pad[0])),
        (float(mins[1] - pad[1]), float(maxs[1] + pad[1])),
    )


def _sample_t1(
    image: nib.Nifti1Image, data: np.ndarray, world_points: np.ndarray,
) -> np.ndarray:
    points = np.asarray(world_points, float)
    voxels = nib.affines.apply_affine(
        np.linalg.inv(image.affine), points.reshape(-1, 3),
    )
    sampled = map_coordinates(
        data, voxels.T, order=1, mode="constant", cval=0.0,
    )
    return sampled.reshape(points.shape[:-1])


def _t1_display_range(data: np.ndarray) -> tuple[float, float]:
    tissue = np.asarray(data, float)
    tissue = tissue[np.isfinite(tissue) & (tissue > 5.0)]
    if tissue.size == 0:
        raise ValueError("E1146 T1 contains no non-background tissue")
    low, high = np.percentile(tissue, [2.0, 99.0])
    return float(low), float(high)


def _t1_facecolors(
    values: np.ndarray, display_range: tuple[float, float], alpha: float,
) -> np.ndarray:
    low, high = display_range
    normalized = np.clip((np.asarray(values, float) - low) / (high - low), 0.0, 1.0)
    rgba = plt.cm.gray(normalized**0.78)
    rgba[..., 3] = float(alpha) * (np.asarray(values) > 5.0)
    return rgba


def _draw_subject_t1_cutaway(
    ax,
    image: nib.Nifti1Image,
    data: np.ndarray,
    geometry: Mapping[str, object],
) -> None:
    """Render three real E1146 T1 planes around the local ICL/SCL contacts."""
    ax.computed_zorder = False
    world = np.asarray(geometry["world"], float)
    fitted = np.asarray(geometry["world_fitted"], float)
    names = np.asarray(geometry["names"], dtype=object)
    shafts = np.asarray(geometry["shafts"], dtype=object)
    selected = np.asarray(geometry["selected"], bool)
    measured = np.asarray(geometry["measured"], float)
    center = np.mean(world, axis=0)
    mins = np.min(world, axis=0) - np.asarray([14.0, 15.0, 14.0])
    maxs = np.max(world, axis=0) + np.asarray([14.0, 15.0, 14.0])
    display_range = _t1_display_range(data)

    x = np.linspace(mins[0], maxs[0], 76)
    y = np.linspace(mins[1], maxs[1], 68)
    z = np.linspace(mins[2], maxs[2], 68)
    yy, zz = np.meshgrid(y, z)
    xx_sag = np.full_like(yy, center[0])
    sag_world = np.stack([xx_sag, yy, zz], axis=-1)
    xx, zz_cor = np.meshgrid(x, z)
    yy_cor = np.full_like(xx, center[1])
    cor_world = np.stack([xx, yy_cor, zz_cor], axis=-1)
    xx_ax, yy_ax = np.meshgrid(x, y)
    zz_ax = np.full_like(xx_ax, center[2])
    ax_world = np.stack([xx_ax, yy_ax, zz_ax], axis=-1)

    planes = (
        (xx_sag, yy, zz, sag_world, 0.70),
        (xx, yy_cor, zz_cor, cor_world, 0.68),
        (xx_ax, yy_ax, zz_ax, ax_world, 0.64),
    )
    for px, py, pz, points, alpha in planes:
        values = _sample_t1(image, data, points)
        ax.plot_surface(
            px, py, pz,
            facecolors=_t1_facecolors(values, display_range, alpha),
            rstride=1, cstride=1, shade=False, linewidth=0.0,
            antialiased=False, zorder=1,
        )

    for idx in _shaft_indices(names.tolist(), shafts.tolist()):
        direction = fitted[idx[-1]] - fitted[idx[0]]
        direction /= np.linalg.norm(direction)
        carrier = np.vstack(
            [fitted[idx[0]] - 0.9 * direction, fitted[idx[-1]] + 0.9 * direction]
        )
        ax.plot(
            *carrier.T, color="#45545B", lw=1.25, alpha=0.92,
            solid_capstyle="round", zorder=7,
        )
    ax.scatter(
        *world[~selected].T, s=15.0, facecolor=CONTEXT_CONTACT,
        edgecolor=MID_GREY, linewidth=0.50, depthshade=False, zorder=8,
    )
    ax.scatter(
        *world[selected].T, s=21.0, facecolor=SELECTED_CONTACT,
        edgecolor="white", linewidth=0.60, depthshade=False, zorder=9,
    )
    for idx in _shaft_indices(names.tolist(), shafts.tolist()):
        label_position = _shaft_label_position(
            fitted, measured, idx, offset=2.2,
        )
        ax.text(
            *label_position, str(shafts[int(idx[0])]),
            color="#66757B", fontsize=6.0, fontweight="bold",
            ha="right", va="center", zorder=10,
        )
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(maxs - mins, zoom=1.30)
    ax.set_proj_type("ortho")
    ax.view_init(elev=22.0, azim=-105.0)
    ax.set_axis_off()


def _draw_contact_geometry(ax, geometry: Mapping[str, object]) -> None:
    """Render real E1146 local 3-D geometry and its display projection plane."""
    ax.computed_zorder = False
    measured = np.asarray(geometry["measured"], float)
    fitted = np.asarray(geometry["fitted"], float).copy()
    names = np.asarray(geometry["names"], dtype=object)
    shafts = np.asarray(geometry["shafts"], dtype=object)
    selected = np.asarray(geometry["selected"], bool)

    fitted[:, 2] *= NORMAL_DISPLAY_EXAGGERATION
    display_measured = measured.copy()
    display_measured[:, 2] *= NORMAL_DISPLAY_EXAGGERATION
    xlim, ylim = _limits_2d(
        measured[:, :2], pad_fraction=GEOMETRY_PADDING_FRACTION,
    )
    corners = np.asarray(
        [
            [xlim[0], ylim[0], DISPLAY_PLANE_Z],
            [xlim[1], ylim[0], DISPLAY_PLANE_Z],
            [xlim[1], ylim[1], DISPLAY_PLANE_Z],
            [xlim[0], ylim[1], DISPLAY_PLANE_Z],
        ]
    )
    ax.add_collection3d(
        Poly3DCollection(
            [corners], facecolor=PLANE_FACE, edgecolor=PLANE_EDGE,
            linewidth=0.7, alpha=0.48, zorder=0,
        )
    )

    for idx in _shaft_indices(names.tolist(), shafts.tolist()):
        direction = fitted[idx[-1]] - fitted[idx[0]]
        direction /= np.linalg.norm(direction)
        carrier = np.vstack(
            [fitted[idx[0]] - 0.8 * direction, fitted[idx[-1]] + 0.8 * direction]
        )
        ax.plot(
            *carrier.T, color=MID_GREY, lw=2.2, alpha=0.92,
            solid_capstyle="round", zorder=3,
        )

    for point in display_measured[selected]:
        ax.plot(
            [point[0], point[0]], [point[1], point[1]],
            [point[2], DISPLAY_PLANE_Z], color=PLANE_EDGE,
            lw=0.42, alpha=0.42, zorder=2,
        )
    feet = display_measured[:, :2]
    ax.scatter(
        feet[selected, 0], feet[selected, 1],
        np.full(int(np.sum(selected)), DISPLAY_PLANE_Z),
        s=8.0, facecolor=SELECTED_CONTACT, edgecolor="none",
        alpha=0.42, depthshade=False, zorder=2.7,
    )
    ax.scatter(
        *fitted[~selected].T, s=18.0, facecolor=CONTEXT_CONTACT,
        edgecolor=MID_GREY, linewidth=0.58, depthshade=False, zorder=4.5,
    )
    ax.scatter(
        *fitted[selected].T, s=25.0, facecolor=SELECTED_CONTACT,
        edgecolor=SELECTED_DARK, linewidth=0.65, depthshade=False, zorder=5.0,
    )
    for idx in _shaft_indices(names.tolist(), shafts.tolist()):
        label_position = _shaft_label_position(
            fitted, measured, idx, offset=1.8,
        )
        ax.text(
            *label_position, str(shafts[int(idx[0])]),
            color="#66757B", fontsize=6.0, fontweight="bold",
            ha="right", va="center", zorder=6.0,
        )

    zlim = (
        float(min(fitted[:, 2].min(), DISPLAY_PLANE_Z) - 0.9),
        float(max(fitted[:, 2].max(), 0.7) + 0.9),
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect(
        (xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]),
        zoom=GEOMETRY_BOX_ZOOM,
    )
    ax.set_anchor("N")
    ax.set_proj_type("ortho")
    ax.view_init(elev=29.0, azim=-98.0)
    ax.set_axis_off()


def _draw_projection(
    ax, record: Mapping[str, object], geometry: Mapping[str, object],
) -> str:
    """Draw contact projection and the exact 6-mm Gaussian display coverage."""
    measured = np.asarray(geometry["measured"], float)
    names = np.asarray(geometry["names"], dtype=object)
    shafts = np.asarray(geometry["shafts"], dtype=object)
    selected = np.asarray(geometry["selected"], bool)
    support = _selected_support(record, geometry)
    xy = measured[:, :2]
    xlim, ylim = _limits_2d(xy, pad_fraction=0.24)

    ax.set_facecolor("#F7FAFA")
    gx = np.linspace(*xlim, 300)
    gy = np.linspace(*ylim, 260)
    xx, yy = np.meshgrid(gx, gy)
    d2 = (xx[..., None] - xy[:, 0]) ** 2 + (yy[..., None] - xy[:, 1]) ** 2
    weights = np.exp(-d2 / (2.0 * DISPLAY_SIGMA_MM**2))
    density = np.sum(weights * support[None, None, :], axis=2)
    density /= float(np.nanmax(density))
    rgba = np.empty((*density.shape, 4), float)
    rgba[..., :3] = np.asarray([0.22, 0.54, 0.58])
    rgba[..., 3] = 0.28 * density * (density >= 0.03)
    ax.imshow(
        rgba, origin="lower", extent=[*xlim, *ylim],
        interpolation="bilinear", zorder=0,
    )
    for idx in _shaft_indices(names.tolist(), shafts.tolist()):
        points = xy[idx]
        ax.plot(
            points[:, 0], points[:, 1], color="#9AA8AD", lw=0.75,
            alpha=0.78, zorder=1,
        )
        shaft = str(shafts[int(idx[0])])
        anchor = points[int(np.argmin(points[:, 0]))]
        ax.text(
            anchor[0] - 0.8, anchor[1], shaft, color="#66757B",
            fontsize=5.8, ha="right", va="center", zorder=5,
        )

    ax.scatter(
        xy[~selected, 0], xy[~selected, 1], s=20.0,
        facecolor=CONTEXT_CONTACT, edgecolor=MID_GREY,
        linewidth=0.60, zorder=3, label="local context",
    )
    ax.scatter(
        xy[selected, 0], xy[selected, 1], s=28.0,
        facecolor=SELECTED_CONTACT, edgecolor="white",
        linewidth=0.72, zorder=4, label="analysis contacts",
    )

    kernel_index = int(np.argmax(support))
    kernel_name = str(names[kernel_index])
    cx, cy = float(xy[kernel_index, 0]), float(xy[kernel_index, 1])
    ax.add_patch(
        Circle(
            (cx, cy), DISPLAY_SIGMA_MM, facecolor="none",
            edgecolor=SELECTED_DARK, linewidth=0.85, zorder=5,
        )
    )
    angle = np.deg2rad(135.0)
    ex = cx + DISPLAY_SIGMA_MM * np.cos(angle)
    ey = cy + DISPLAY_SIGMA_MM * np.sin(angle)
    ax.plot([cx, ex], [cy, ey], color=SELECTED_DARK, lw=0.72, zorder=6)
    ax.text(
        0.5 * (cx + ex) - 0.3, 0.5 * (cy + ey) + 0.6,
        r"$\sigma=6$ mm", ha="center", va="bottom", fontsize=5.7,
        color=SELECTED_DARK, zorder=6,
    )

    scale_length = 10.0
    scale_x0 = xlim[0] + 0.09 * (xlim[1] - xlim[0])
    scale_y = ylim[0] + 0.10 * (ylim[1] - ylim[0])
    ax.plot(
        [scale_x0, scale_x0 + scale_length], [scale_y, scale_y],
        color=INK, lw=1.1, solid_capstyle="butt", zorder=6,
    )
    ax.text(
        scale_x0 + 0.5 * scale_length, scale_y + 0.65, "10 mm",
        color=INK, fontsize=5.6, ha="center", va="bottom", zorder=6,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(PLANE_EDGE)
        spine.set_linewidth(0.62)
    return kernel_name


def _metadata(
    record: Mapping[str, object],
    basis: Mapping[str, np.ndarray],
    geometry: Mapping[str, object],
    coord_space: str,
    transverse_sign: int,
    t1_manifest: Mapping[str, object],
    kernel_contact: str,
) -> dict:
    measured = np.asarray(geometry["measured"], float)
    selected = np.asarray(geometry["selected"], bool)
    selected_names = [str(x) for x in record["interictal_field"]["contact_order"]]
    local_names = [str(x) for x in np.asarray(geometry["names"], dtype=object)]
    local_residual = measured[:, 2]
    residual = measured[selected, 2]
    t1_entry = t1_manifest["files"][T1_PATH.name]
    return {
        "schema_version": "fig2_mixed_subject_implant_projection_2x2_v7",
        "subject_id": SUBJECT_ID,
        "display_label": DISPLAY_LABEL,
        "overview_subject_id": "yuquan_zhaochenxi",
        "overview_display_label": "Y9",
        "subject_continuity_across_panels": False,
        "input_artifact": str(INPUT_ARTIFACT.resolve()),
        "input_contract": record.get("contract"),
        "input_fingerprint_algorithm": record["interictal_field"]["fingerprint_algorithm"],
        "input_fingerprint_sha256": record["interictal_field"]["fingerprint_sha256"],
        "coordinate_space": coord_space,
        "visual_story": [
            "author_supplied_y9_implantation_overview",
            "subject_t1_local_cutaway",
            "local_electrode_geometry_to_plane",
            "contact_projection_with_gaussian_support_coverage",
        ],
        "implantation_overview": {
            "source_image": str(IMPLANTATION_OVERVIEW.resolve()),
            "source_image_sha256": IMPLANTATION_OVERVIEW_SHA256,
            "source_image_shape_px": [392, 488],
            "subject_id": "yuquan_zhaochenxi",
            "display_label": "Y9",
            "role": "representative implantation overview",
            "same_subject_as_projection_stages": False,
        },
        "anatomical_context": {
            "source_t1": str(T1_PATH.resolve()),
            "source_t1_sha256": t1_entry["sha256"],
            "source_t1_shape": t1_manifest["mni152_1mm_shape"],
            "source_t1_affine": t1_manifest["mni152_1mm_affine"],
            "source_t1_description": "subject-specific skull-stripped T1",
            "local_cutaway": "three orthogonal T1 planes through local-contact centroid",
            "world_coordinate_convention": t1_manifest["world_coordinate_convention"],
            "normalization_certainty": t1_manifest["normalization_certainty"],
        },
        "implantation": {
            "n_all_invasive_contacts": len(
                enumerate_subject_all_channels("epilepsiae", "1146")
            ),
            "local_context_shafts": ["ICL", "SCL"],
            "n_local_context_contacts": len(local_names),
            "local_context_contacts": local_names,
            "n_analysis_contacts": len(selected_names),
            "analysis_contacts": selected_names,
            "local_context_not_selected": [
                name for name in local_names if name not in set(selected_names)
            ],
            "selection_contract": record["interictal_field"]["field_contact_policy"],
            "selected_subset_of_local_context": set(selected_names).issubset(set(local_names)),
        },
        "projection_plane": {
            "source": "frozen_patient_shared_plane",
            "u": np.asarray(basis["u"]).tolist(),
            "w_after_display_sign": np.asarray(basis["w"]).tolist(),
            "normal": np.asarray(basis["normal"]).tolist(),
            "origin_mm": np.asarray(basis["origin"]).tolist(),
            "transverse_sign": int(transverse_sign),
            "selected_normal_residual_max_abs_mm": float(np.max(np.abs(residual))),
            "selected_normal_residual_rms_mm": float(np.sqrt(np.mean(residual**2))),
            "local_context_normal_residual_max_abs_mm": float(
                np.max(np.abs(local_residual))
            ),
            "projection_contact_marker_contract": (
                "grey local-context contacts may lie farther from the frozen plane than "
                "selected analysis contacts; the measured 2-D coordinates are unchanged"
            ),
        },
        "rendering": {
            "layout": "2x2",
            "projection_focus_shafts": ["ICL", "SCL"],
            "straight_shaft_display_fit": geometry["fit_stats"],
            "normal_display_exaggeration": NORMAL_DISPLAY_EXAGGERATION,
            "normal_display_exaggeration_scope": (
                "3-D electrode-geometry stage only; measured x/y projection is unchanged"
            ),
            "display_plane_normal_offset_units": DISPLAY_PLANE_Z,
            "display_plane_offset_scope": (
                "3-D display separation only; frozen 2-D coordinates are unchanged"
            ),
            "pipeline_arrows_rendered": False,
            "panel_grid_equal_width_height": True,
            "narrative_panel_titles_rendered": False,
            "view_identity_source": "figure legend",
            "legend_rendered": False,
            "geometry_padding_fraction": GEOMETRY_PADDING_FRACTION,
            "geometry_box_zoom": GEOMETRY_BOX_ZOOM,
            "geometry_panel_vertical_shift_figure_fraction": 0.015,
            "projection_panel_scale_in_cell": PROJECTION_PANEL_SCALE,
            "shaft_labels_rendered": {
                "e1146_anatomy": ["ICL", "SCL"],
                "electrodes_projection": ["ICL", "SCL"],
                "local_field": ["ICL", "SCL"],
            },
            "figure_size_inches": [5.80, 3.85],
            "propagation_direction_glyphs_rendered": True,
            "propagation_direction_glyphs_scope": (
                "present only inside the author-supplied Y9 implantation overview; "
                f"the {DISPLAY_LABEL} anatomy, projection geometry, and Gaussian-support stages "
                "remain direction-free"
            ),
            "e1146_projection_direction_glyphs_rendered": False,
            "template_labels_rendered": False,
            "template_rank_fields_rendered": False,
            "continuous_template_rank_interpolation_rendered": False,
            "rank_colormap_rendered": False,
            "gaussian_support_coverage_rendered": True,
            "gaussian_support_source": "mean_of_frozen_support_a_and_support_b",
            "gaussian_display_sigma_mm": DISPLAY_SIGMA_MM,
            "gaussian_kernel_exemplar_contact": kernel_contact,
            "gaussian_support_is_measured_tissue_field": False,
            "context_contact_color": CONTEXT_CONTACT,
            "analysis_contact_color": SELECTED_CONTACT,
            "png_dpi": 600,
            "pdf_fonttype": 42,
        },
        "claim_boundary": (
            f"Spatial-method illustration only. The Y9 overview and the {DISPLAY_LABEL} projection "
            "stages are representative examples from different subjects and are labelled as "
            f"such; the ordered 2x2 layout is not a same-subject zoom. {DISPLAY_LABEL} "
            "anatomy and contact locations use its frozen MNI-grid bundle, whose historical "
            "warp type is unverified. The 6-mm Gaussian layer shows display support coverage, "
            "not measured tissue activity or the analysis scoring kernel. The direction "
            f"glyphs visible in the supplied Y9 overview are not propagated into the {DISPLAY_LABEL} "
            "projection stages, and no rank field is shown."
        ),
    }


def plot(
    output_dir: Path = OUTPUT_DIR,
    stem: str = "fig2-panela",
) -> tuple[Path, Path, Path, Path]:
    record, context_names, context_coords, coord_space = _load_case()
    transverse_sign = _canonical_transverse_sign(
        record["interictal_field"]["planes"]["shared"]["w"]
    )
    basis = _basis(record, transverse_sign)
    geometry = _local_geometry(record, basis, context_names, context_coords)
    t1_image, t1_data, t1_manifest = _load_t1()

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(5.80, 3.85), facecolor="white")
        gs = gridspec.GridSpec(
            2, 2, figure=fig,
            width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0],
            left=0.020, right=0.990, top=0.980, bottom=0.035,
            wspace=0.05, hspace=0.06,
        )
        ax_overview = fig.add_subplot(gs[0, 0])
        ax_brain = fig.add_subplot(gs[0, 1], projection="3d")
        ax_geometry = fig.add_subplot(gs[1, 0], projection="3d")
        ax_projection = fig.add_subplot(gs[1, 1])

        _draw_implantation_overview(ax_overview)
        _draw_subject_t1_cutaway(ax_brain, t1_image, t1_data, geometry)
        _draw_contact_geometry(ax_geometry, geometry)
        kernel_contact = _draw_projection(ax_projection, record, geometry)

        geometry_position = ax_geometry.get_position()
        ax_geometry.set_position(
            [
                geometry_position.x0,
                geometry_position.y0 + 0.015,
                geometry_position.width,
                geometry_position.height,
            ]
        )
        projection_cell = gs[1, 1].get_position(fig)
        scaled_width = projection_cell.width * PROJECTION_PANEL_SCALE
        scaled_height = projection_cell.height * PROJECTION_PANEL_SCALE
        ax_projection.set_position(
            [
                projection_cell.x0 + 0.5 * (projection_cell.width - scaled_width),
                projection_cell.y1 - scaled_height,
                scaled_width,
                scaled_height,
            ]
        )
        ax_projection.set_anchor("N")
        # The figure legend explains these two lower views.  Nature-style main
        # figures do not repeat narrative subplot titles inside the canvas.
        apply_panel_aware_figure_typography(
            fig,
            spec=ILLUSTRATIVE_PANEL_TYPOGRAPHY,
            policy=LOCKED_PANEL_TYPOGRAPHY_POLICY,
            enforce_atomic_axis_gate=False,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        png = output_dir / f"{stem}.png"
        pdf = output_dir / f"{stem}.pdf"
        svg = output_dir / f"{stem}.svg"
        metadata_path = output_dir / f"{stem}_metadata.json"
        fig.savefig(png, dpi=600, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        fig.savefig(svg, facecolor="white")
        plt.close(fig)

    metadata_path.write_text(
        json.dumps(
            _metadata(
                record, basis, geometry, coord_space, transverse_sign,
                t1_manifest, kernel_contact,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return png, pdf, svg, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--stem", default="fig2-panela")
    args = parser.parse_args()
    for path in plot(args.output_dir, args.stem):
        print(f"[done] {path}")


if __name__ == "__main__":
    main()
