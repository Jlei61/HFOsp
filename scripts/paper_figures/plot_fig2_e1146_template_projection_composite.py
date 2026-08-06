#!/usr/bin/env python3
"""E1146 method composite: high-HFO contacts, shared plane, and TA/TB fields.

The figure uses the frozen E1146 interictal field artifact without refitting
axes, plane, ranks, support, or contact order.  It separates three objects:

1. the two local implantation shafts fitted as straight 3-D rods, with the
   frozen TA/TB directions defining the shared plane and its principal axis;
2. the selected contacts on that plane, together with their 6-mm Gaussian
   display support;
3. the canonical support-limited TA/TB rank fields on that same plane.
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
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_topic5_interictal_template_ab_fields import (  # noqa: E402
    build_interictal_ab_panel_payloads,
    draw_interictal_rank_field_panel,
)
from src.seeg_coord_loader import (  # noqa: E402
    enumerate_subject_all_channels,
    load_subject_coords,
)


SUBJECT_ID = "epilepsiae_1146"
DISPLAY_LABEL = "E1146"
INPUT_ARTIFACT = (
    ROOT
    / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
    / f"{SUBJECT_ID}.json"
)
OUTPUT_DIR = (
    ROOT
    / "results/paper-ready-figure/fig2_e1146_template_projection_composite/figures"
)
DISPLAY_SIGMA_MM = 6.0
NORMAL_DISPLAY_EXAGGERATION = 3.0
N_EARLY_SPATIAL_CONTACTS = 3
# Display-only separation along the shared-plane normal.  The frozen plane's
# x/y coordinates are unchanged; lowering its rendered z position exposes the
# near-coplanar ICL shaft and the SCL shaft's plane crossing.
DISPLAY_PLANE_Z = -1.2

TA_COLOR = "#B2182B"
TB_COLOR = "#2166AC"
ROD_CONTACT_FACE = "#F2F4F5"
ROD_CONTACT_EDGE = "#4F5A60"
PROJECTION_COLOR = "#587F8B"
INK = "#24292D"
MID_GREY = "#737C81"
LIGHT_GREY = "#C8CED2"
PLANE_FACE = "#E1E9ED"
PLANE_EDGE = "#9AA7AE"


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


def _load_case() -> tuple[dict, dict, dict, list[str], np.ndarray, str]:
    record = json.loads(INPUT_ARTIFACT.read_text())
    if record.get("subject_id") != SUBJECT_ID or record.get("status") != "ok":
        raise ValueError(f"unexpected or unavailable record: {record.get('subject_id')}")
    pair = record.get("axis_pair") or {}
    field = record.get("interictal_field") or {}
    if not bool(pair.get("geometry_2d_supported")):
        raise ValueError("E1146 no longer meets the frozen two-dimensional geometry contract")
    if not bool((pair.get("relation") or {}).get("collinear")):
        raise ValueError("E1146 no longer meets the frozen shared-plane criterion")
    if not all(key in (field.get("field_models") or {}) for key in ("shared_a", "shared_b")):
        raise ValueError("E1146 frozen artifact is missing shared TA/TB field models")

    dat_a, dat_b, mode = build_interictal_ab_panel_payloads(
        record, display_sigma_mm=DISPLAY_SIGMA_MM,
    )
    if mode != "shared":
        raise ValueError(f"expected shared-plane E1146 artifact, found {mode!r}")

    all_names = enumerate_subject_all_channels("epilepsiae", "1146")
    coord = load_subject_coords("epilepsiae", "1146", all_names)
    if coord.coord_space != "mni152_1mm" or coord.coord_units != "mm":
        raise ValueError(
            f"unexpected E1146 coordinate contract: {coord.coord_space}/{coord.coord_units}"
        )
    if not bool(np.all(coord.mapped_mask_in_requested_order)):
        raise ValueError("one or more E1146 ICL/SCL context contacts lack coordinates")
    selected = set(str(x) for x in field["contact_order"])
    if not selected.issubset(set(all_names)):
        raise ValueError("frozen E1146 field contacts are not contained in the implantation")
    return (
        record,
        dat_a,
        dat_b,
        all_names,
        np.asarray(coord.coords_array_in_requested_order, float),
        coord.coord_space,
    )


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


def _vector_2d(vector: Sequence[float], basis: Mapping[str, np.ndarray]) -> np.ndarray:
    vec = np.asarray(vector, float)
    out = np.asarray([vec @ basis["u"], vec @ basis["w"]], float)
    return out / np.linalg.norm(out)


def _selected_support(record: Mapping[str, object]) -> np.ndarray:
    support_a = np.asarray(record["interictal_field"]["support_a"], float)
    support_b = np.asarray(record["interictal_field"]["support_b"], float)
    return 0.5 * (support_a + support_b)


def _earliest_contact_names(
    record: Mapping[str, object], rank_key: str, n_contacts: int,
) -> list[str]:
    field = record["interictal_field"]
    names = np.asarray([str(x) for x in field["contact_order"]], dtype=object)
    ranks = np.asarray(field[rank_key], float)
    valid = np.where(np.isfinite(ranks))[0]
    order = valid[np.argsort(ranks[valid], kind="stable")]
    return names[order[:n_contacts]].tolist()


def _fit_straight_shaft_points(
    coords: np.ndarray, names: Sequence[str], shafts: Sequence[str],
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Place each displayed contact on its shaft's best-fit 3-D line."""
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


def _draw_local_geometry(
    ax,
    record: Mapping[str, object],
    dat_a: Mapping[str, object],
    basis: Mapping[str, np.ndarray],
    context_names: Sequence[str],
    context_coords: np.ndarray,
) -> None:
    # Honor explicit artist layers so early-contact colors remain visible even
    # when the 3-D painter would otherwise place them behind the display plane.
    ax.computed_zorder = False
    all_names = np.asarray([str(x) for x in context_names], dtype=object)
    all_shafts = np.asarray([re.sub(r"\d+$", "", str(x)) for x in all_names], dtype=object)
    all_coords = _to_basis(context_coords, basis)
    xlim = tuple(float(x) for x in dat_a["frame"]["xlim"])
    ylim = tuple(float(x) for x in dat_a["frame"]["ylim"])
    local_two_shaft = np.isin(all_shafts, ["ICL", "SCL"])
    names = all_names[local_two_shaft].tolist()
    shafts = all_shafts[local_two_shaft].tolist()
    coords_true = all_coords[local_two_shaft]
    # E1146's selected contacts genuinely lie very close to the plane
    # (max residual 0.82 mm).  A fixed normal-axis display exaggeration makes
    # the rod-to-plane operation legible without changing any x/y projection.
    fitted_true, _ = _fit_straight_shaft_points(coords_true, names, shafts)
    # The physical carrier and its displayed contacts form one ideal straight
    # SEEG shaft.  Plane footprints retain the measured E1146 x/y coordinates;
    # fitting those footprints would incorrectly force the field inputs onto a
    # straight line and break consistency with the downstream panels.
    measured_coords = coords_true.copy()
    measured_coords[:, 2] *= NORMAL_DISPLAY_EXAGGERATION
    rod_contacts = fitted_true.copy()
    rod_contacts[:, 2] *= NORMAL_DISPLAY_EXAGGERATION
    feet = measured_coords.copy()
    feet[:, 2] = DISPLAY_PLANE_Z
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
            linewidth=0.55, alpha=0.42, zorder=0,
        )
    )

    groups = _shaft_indices(names, shafts)
    for idx in groups:
        direction = rod_contacts[idx[-1]] - rod_contacts[idx[0]]
        direction /= np.linalg.norm(direction)
        carrier = np.vstack(
            [rod_contacts[idx[0]] - 1.15 * direction,
             rod_contacts[idx[-1]] + 1.15 * direction]
        )
        relative_z = carrier[:, 2] - DISPLAY_PLANE_Z
        if float(relative_z[0] * relative_z[1]) < 0.0:
            fraction = float(relative_z[0] / (relative_z[0] - relative_z[1]))
            crossing = carrier[0] + fraction * (carrier[1] - carrier[0])
            segments = [
                (np.vstack([carrier[0], crossing]), relative_z[0] >= 0.0),
                (np.vstack([crossing, carrier[1]]), relative_z[1] >= 0.0),
            ]
            ax.scatter(
                *crossing, s=10, facecolor="#5A666C", edgecolor="white",
                linewidth=0.4, depthshade=False, zorder=4.2,
            )
        else:
            segments = [(carrier, bool(np.mean(relative_z) >= 0.0))]
        for segment, above_plane in segments:
            alpha = 0.96 if above_plane else 0.34
            linestyle = "-" if above_plane else (0, (2.0, 1.5))
            ax.plot(
                *segment.T, color="#A9B1B5", lw=2.8, alpha=alpha,
                linestyle=linestyle, zorder=2.9, solid_capstyle="round",
            )
            ax.plot(
                *segment.T, color="#5A666C", lw=0.72,
                alpha=0.92 if above_plane else 0.38,
                linestyle=linestyle, zorder=3.0, solid_capstyle="round",
            )
        for contact in idx:
            contact = int(contact)
            ax.plot(
                [rod_contacts[contact, 0], feet[contact, 0]],
                [rod_contacts[contact, 1], feet[contact, 1]],
                [rod_contacts[contact, 2], DISPLAY_PLANE_Z],
                color="#A1AAAE", lw=0.34, alpha=0.28, zorder=2.2,
            )

    ax.scatter(
        *feet.T, s=5.5, facecolor="#7F8B90", edgecolor="none",
        linewidth=0.0, alpha=0.32, depthshade=False, zorder=2.8,
    )
    ax.scatter(
        *rod_contacts.T, s=21.0, facecolor=ROD_CONTACT_FACE,
        edgecolor=ROD_CONTACT_EDGE, linewidth=0.72, depthshade=False, zorder=5,
    )

    early_ta = set(
        _earliest_contact_names(record, "rank_a", N_EARLY_SPATIAL_CONTACTS)
    )
    early_tb = set(
        _earliest_contact_names(record, "rank_b", N_EARLY_SPATIAL_CONTACTS)
    )
    name_array = np.asarray(names, dtype=object)
    ta_only = np.asarray(
        [str(name) in early_ta and str(name) not in early_tb for name in name_array],
        bool,
    )
    tb_only = np.asarray(
        [str(name) in early_tb and str(name) not in early_ta for name in name_array],
        bool,
    )
    overlap = np.asarray(
        [str(name) in early_ta and str(name) in early_tb for name in name_array],
        bool,
    )
    for mask, color in ((ta_only, TA_COLOR), (tb_only, TB_COLOR)):
        ax.scatter(
            *rod_contacts[mask].T, s=21.0, facecolor=color,
            edgecolor=ROD_CONTACT_EDGE,
            linewidth=0.72, depthshade=False, zorder=6.2,
        )
    for point in rod_contacts[overlap]:
        ax.plot(
            [point[0]], [point[1]], [point[2]], linestyle="None", marker="o",
            markersize=4.55, fillstyle="left", markerfacecolor=TA_COLOR,
            markerfacecoloralt=TB_COLOR, markeredgecolor=ROD_CONTACT_EDGE,
            markeredgewidth=0.72, zorder=6.4,
        )

    _draw_template_directions_3d(ax, record, basis)

    zlim = (
        float(min(rod_contacts[:, 2].min(), DISPLAY_PLANE_Z) - 1.2),
        float(max(rod_contacts[:, 2].max(), 1.2) + 1.2),
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect(
        (xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]),
        zoom=1.08,
    )
    ax.set_proj_type("ortho")
    ax.view_init(elev=32.0, azim=-95.0)
    ax.set_axis_off()


def _draw_template_directions_3d(
    ax,
    record: Mapping[str, object],
    basis: Mapping[str, np.ndarray],
) -> None:
    ua = _vector_2d(record["axis_pair"]["axis_a"]["u"], basis)
    ub = _vector_2d(record["axis_pair"]["axis_b"]["u"], basis)
    length = 8.2
    direction_specs = (
        (ua, TA_COLOR, "TA", np.asarray([-14.0, 2.8, DISPLAY_PLANE_Z + 0.15])),
        (ub, TB_COLOR, "TB", np.asarray([14.0, 0.2, DISPLAY_PLANE_Z + 0.15])),
    )
    for vector, color, label, origin in direction_specs:
        vector3 = np.asarray([vector[0], vector[1], 0.0])
        ax.quiver(
            *origin, *(length * vector3), color=color, lw=0.95,
            arrow_length_ratio=0.12, normalize=False, zorder=8,
        )
        label_xy = origin + 0.48 * length * vector3
        label_xy[2] += 0.54
        ax.text(
            *label_xy, label, color=color, fontsize=6.4,
            fontweight="normal", ha="center", va="center", zorder=9,
        )


def _draw_projected_support(
    ax,
    record: Mapping[str, object],
    payload: Mapping[str, object],
) -> int:
    xs = np.asarray(payload["xs"], float)
    ys = np.asarray(payload["ys"], float)
    names = [str(x) for x in payload["names"]]
    shafts = [str(x) for x in record["interictal_field"]["shafts"]]
    support = _selected_support(record)
    xlim = tuple(float(x) for x in payload["frame"]["xlim"])
    ylim = tuple(float(x) for x in payload["frame"]["ylim"])

    gx = np.linspace(*xlim, 280)
    gy = np.linspace(*ylim, 280)
    xx, yy = np.meshgrid(gx, gy)
    d2 = (xx[..., None] - xs) ** 2 + (yy[..., None] - ys) ** 2
    weights = np.exp(-d2 / (2.0 * DISPLAY_SIGMA_MM**2))
    density = np.sum(weights * support[None, None, :], axis=2)
    density /= float(np.nanmax(density))
    rgba = np.empty((*density.shape, 4), float)
    rgba[..., :3] = np.asarray([0.34, 0.53, 0.59])
    rgba[..., 3] = 0.22 * density * (density >= 0.03)
    ax.imshow(
        rgba, origin="lower", extent=[*xlim, *ylim],
        interpolation="bilinear", zorder=0,
    )

    sizes = 28.0 + 16.0 * support
    ax.scatter(
        xs, ys, s=sizes, facecolor=PROJECTION_COLOR, edgecolor="white",
        linewidth=0.82, zorder=4,
    )

    kernel_index = int(np.argmax(support))
    cx, cy = float(xs[kernel_index]), float(ys[kernel_index])
    ax.add_patch(
        Circle(
            (cx, cy), DISPLAY_SIGMA_MM, facecolor="none", edgecolor="#526E77",
            linewidth=0.78, zorder=3,
        )
    )
    angle = np.deg2rad(135.0)
    ex = cx + DISPLAY_SIGMA_MM * np.cos(angle)
    ey = cy + DISPLAY_SIGMA_MM * np.sin(angle)
    ax.plot([cx, ex], [cy, ey], color="#526E77", lw=0.72, zorder=5)
    ax.text(
        0.5 * (cx + ex) - 0.4, 0.5 * (cy + ey) + 0.65, "6 mm",
        ha="center", va="bottom", fontsize=5.7, color="#46555B", zorder=6,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#AAB3B7")
        spine.set_linewidth(0.55)
    return kernel_index


def _draw_field(ax, payload: Mapping[str, object], template: str) -> None:
    draw_interictal_rank_field_panel(
        ax, payload, template, compact=True, panel_title="",
        contact_outline_lw=0.85, contact_size=27,
    )
    ax.set_title("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _flow_arrow(fig, left_ax, right_ax, *, y_from_right: bool = False) -> None:
    left = left_ax.get_position()
    right = right_ax.get_position()
    y_box = right if y_from_right else left
    y = 0.5 * (y_box.y0 + y_box.y1)
    fig.add_artist(
        FancyArrowPatch(
            (left.x1 + 0.0035, y), (right.x0 - 0.0035, y),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8.0,
            lw=1.05, color="#7E888D", clip_on=False,
        )
    )


def _metadata(
    record: Mapping[str, object],
    payload: Mapping[str, object],
    basis: Mapping[str, np.ndarray],
    context_names: Sequence[str],
    context_coords: np.ndarray,
    coord_space: str,
    kernel_contact: str,
) -> dict:
    selected_names = [str(x) for x in record["interictal_field"]["contact_order"]]
    selected_coords = np.asarray(record["interictal_field"]["coords"], float)
    residual = _to_basis(selected_coords, basis)[:, 2]
    all_names = np.asarray([str(x) for x in context_names], dtype=object)
    all_shafts = np.asarray([re.sub(r"\d+$", "", str(x)) for x in all_names], dtype=object)
    local_mask = np.isin(all_shafts, ["ICL", "SCL"])
    local_names = all_names[local_mask].tolist()
    local_shafts = all_shafts[local_mask].tolist()
    local_coords = _to_basis(np.asarray(context_coords, float), basis)[local_mask]
    straight_local, straight_fit_stats = _fit_straight_shaft_points(
        local_coords, local_names, local_shafts,
    )
    straight_local_display = straight_local.copy()
    straight_local_display[:, 2] *= NORMAL_DISPLAY_EXAGGERATION
    display_crossing: dict[str, bool] = {}
    for idx in _shaft_indices(local_names, local_shafts):
        shaft = str(local_shafts[int(idx[0])])
        z = straight_local_display[idx, 2]
        display_crossing[shaft] = bool(
            float(np.min(z)) <= DISPLAY_PLANE_Z <= float(np.max(z))
        )
    selected_set = set(selected_names)
    not_selected = [x for x in local_names if x not in selected_set]
    relation = record["axis_pair"]["relation"]
    pair_bootstrap = record["axis_pair"]["pair_bootstrap"]
    return {
        "schema_version": "fig2_e1146_template_projection_composite_v15",
        "subject_id": SUBJECT_ID,
        "display_label": DISPLAY_LABEL,
        "input_artifact": str(INPUT_ARTIFACT.resolve()),
        "input_contract": record.get("contract"),
        "input_fingerprint_algorithm": record["interictal_field"]["fingerprint_algorithm"],
        "input_fingerprint_sha256": record["interictal_field"]["fingerprint_sha256"],
        "coordinate_space": coord_space,
        "implantation": {
            "n_all_invasive_contacts": len(
                enumerate_subject_all_channels("epilepsiae", "1146")
            ),
            "local_context_shafts": ["ICL", "SCL"],
            "n_local_context_contacts": len(local_names),
            "local_context_contacts": local_names,
            "n_lagpat_selected_contacts": len(selected_names),
            "lagpat_selected_contacts": selected_names,
            "local_context_not_selected": not_selected,
            "selection_contract": record["interictal_field"]["field_contact_policy"],
            "support_source": record.get("support_source"),
            "selected_subset_of_local_context": selected_set.issubset(set(local_names)),
        },
        "axis_contract": {
            "definition": record["axis_definition"],
            "direction": record["axis_direction_convention"],
            "u_ta": record["axis_pair"]["axis_a"]["u"],
            "u_tb": record["axis_pair"]["axis_b"]["u"],
            "cos_ta_tb": relation["cosine"],
            "line_angle_deg": relation["line_angle_deg"],
            "relation": relation["relation"],
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
            "transverse_sign": int(payload["transverse_sign"]),
            "selected_normal_residual_max_abs_mm": float(np.max(np.abs(residual))),
            "selected_normal_residual_rms_mm": float(np.sqrt(np.mean(residual**2))),
        },
        "rendering": {
            "display_sigma_mm": DISPLAY_SIGMA_MM,
            "kernel_exemplar_contact": kernel_contact,
            "continuous_field_renderer": (
                "scripts.plot_topic5_interictal_template_ab_fields."
                "draw_interictal_rank_field_panel"
            ),
            "contact_style_contract": {
                "straight_3d_rod_contacts": ROD_CONTACT_FACE,
                "left_panel_measured_plane_footprints": "small low-alpha grey with no edge",
                "middle_panel_field_nodes": PROJECTION_COLOR,
                "yellow_highlight_rings": False,
            },
            "flow_arrow_contract": {
                "length": "span the compact inter-panel gap",
                "line_width_pt": 1.05,
                "mutation_scale": 8.0,
            },
            "spatial_early_contact_overlay": {
                "n_per_template": N_EARLY_SPATIAL_CONTACTS,
                "ta_contacts": _earliest_contact_names(
                    record, "rank_a", N_EARLY_SPATIAL_CONTACTS,
                ),
                "tb_contacts": _earliest_contact_names(
                    record, "rank_b", N_EARLY_SPATIAL_CONTACTS,
                ),
                "shared_contact_style": "half TA red and half TB blue",
                "edge_style": "same dark edge and size as all other 3-D shaft contacts",
                "scope": "left-panel 3-D shaft contacts only",
            },
            "direction_glyph_contract": {
                "origins": "separate template-specific origins adjacent to opposite early-contact sides",
                "placement": (
                    "TA occupies the left inter-shaft gap and points toward the TB-early blue side; "
                    "TB occupies the right inter-shaft gap and points toward the TA-early red side"
                ),
                "line_width_pt": 0.95,
                "font_weight": "normal",
                "shared_axis_rendered": False,
                "angle_wedge_rendered": False,
            },
            "projection_focus_shafts": ["ICL", "SCL"],
            "straight_shaft_display_fit": straight_fit_stats,
            "left_panel_camera": {"elevation_deg": 32.0, "azimuth_deg": -95.0},
            "normal_display_exaggeration": NORMAL_DISPLAY_EXAGGERATION,
            "normal_display_exaggeration_scope": (
                "left 3-D method illustration only; x/y projection, axes, plane and fields unchanged"
            ),
            "display_plane_normal_offset_units": DISPLAY_PLANE_Z,
            "display_plane_offset_scope": (
                "left-panel z separation only; frozen shared-plane x/y coordinates and all fields unchanged"
            ),
            "display_plane_crossing_by_shaft": display_crossing,
            "projection_contract": (
                "each implanted carrier and its displayed 3-D contacts lie on the same "
                "best-fit straight line; plane footprints retain the measured E1146 x/y "
                "projection coordinates, are shown as small blue-grey points, and are not "
                "joined into a projected rod"
            ),
            "colormap": "viridis",
            "rank_scale": [
                float(np.nanmin(record["interictal_field"]["rank_a"])),
                float(np.nanmax(record["interictal_field"]["rank_a"])),
            ],
            "png_dpi": 400,
            "pdf_fonttype": 42,
        },
        "claim_boundary": (
            "Patient-specific method illustration. Highlighted contacts are the exact frozen "
            "lagPat-selected, joint-valid, positive-support TA/TB field contacts; no visual "
            "threshold is introduced. The shared plane is allowed by the frozen point-estimate "
            "collinearity route, but its paired-bootstrap robust_collinear flag is false. "
            "Continuous surfaces are support-limited display interpolation, not measurements in "
            "unsampled tissue."
        ),
    }


def plot() -> tuple[Path, Path, Path, Path]:
    record, dat_a, dat_b, context_names, context_coords, coord_space = _load_case()
    basis = _basis(record, int(dat_a["transverse_sign"]))

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(7.18, 2.74), facecolor="white")
        gs = gridspec.GridSpec(
            2, 4, figure=fig,
            width_ratios=[2.24, 1.34, 1.0, 0.055],
            height_ratios=[1.0, 1.0],
            left=0.012, right=0.940, top=0.965, bottom=0.055,
            wspace=0.085, hspace=0.075,
        )
        ax_geometry = fig.add_subplot(gs[:, 0], projection="3d")
        ax_projection = fig.add_subplot(gs[:, 1])
        ax_field_a = fig.add_subplot(gs[0, 2])
        ax_field_b = fig.add_subplot(gs[1, 2])
        cax = fig.add_subplot(gs[:, 3])

        _draw_local_geometry(
            ax_geometry, record, dat_a, basis, context_names, context_coords,
        )
        kernel_index = _draw_projected_support(ax_projection, record, dat_a)
        _draw_field(ax_field_a, dat_a, "TA")
        _draw_field(ax_field_b, dat_b, "TB")

        rank_min = float(
            min(np.nanmin(dat_a["rank_values"]), np.nanmin(dat_b["rank_values"]))
        )
        rank_max = float(
            max(np.nanmax(dat_a["rank_values"]), np.nanmax(dat_b["rank_values"]))
        )
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(
                cmap="viridis", norm=plt.Normalize(rank_min, rank_max),
            ),
            cax=cax,
        )
        colorbar.set_ticks([rank_min, rank_max])
        colorbar.set_ticklabels(["early", "late"])
        colorbar.ax.tick_params(labelsize=5.8, length=1.8, pad=1.2)
        colorbar.outline.set_linewidth(0.55)

        fig.canvas.draw()
        _flow_arrow(fig, ax_geometry, ax_projection)
        _flow_arrow(fig, ax_projection, ax_field_a, y_from_right=True)
        _flow_arrow(fig, ax_projection, ax_field_b, y_from_right=True)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = "fig2_E1146_template_projection_composite"
        png = OUTPUT_DIR / f"{stem}.png"
        pdf = OUTPUT_DIR / f"{stem}.pdf"
        svg = OUTPUT_DIR / f"{stem}.svg"
        metadata_path = OUTPUT_DIR / f"{stem}_metadata.json"
        fig.savefig(png, dpi=400, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        fig.savefig(svg, facecolor="white")
        plt.close(fig)

    kernel_contact = str(dat_a["names"][kernel_index])
    metadata_path.write_text(
        json.dumps(
            _metadata(
                record, dat_a, basis, context_names, context_coords,
                coord_space, kernel_contact,
            ),
            indent=2,
        )
        + "\n"
    )
    return png, pdf, svg, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for path in plot():
        print(f"[done] {path}")


if __name__ == "__main__":
    main()
