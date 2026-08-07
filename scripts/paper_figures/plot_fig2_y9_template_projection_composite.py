#!/usr/bin/env python3
"""Compact Y9 geometry-to-field composite for the lower part of Figure 2.

The figure contains three scientific readouts of one frozen interictal case:

1. the true three-dimensional contact geometry and two fitted template axes;
2. one contact-centred 6-mm Gaussian display kernel on that plane;
3. the corresponding support-limited TA/TB rank fields.

No axis, plane, contact order, rank, support, or field model is refitted here.
The continuous panels reuse the canonical Topic 5 renderer with the locked
6-mm display bandwidth.  Shaft H is omitted only from this method illustration
at the user's request; the frozen all-contact axes, plane and source artifact
are not changed.
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
from scripts.plot_contact_plane_static import _limits_with_padding  # noqa: E402


SUBJECT_ID = "yuquan_zhaochenxi"
DISPLAY_LABEL = "Y9"
INPUT_ARTIFACT = (
    ROOT
    / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
    / f"{SUBJECT_ID}.json"
)
OUTPUT_DIR = (
    ROOT
    / "results/paper-ready-figure/fig2_y9_template_projection_composite/figures"
)
DISPLAY_SIGMA_MM = 6.0

TA_COLOR = "#B2182B"
TB_COLOR = "#2166AC"
INK = "#202428"
MID_GREY = "#737B80"
LIGHT_GREY = "#D6DBDE"
PLANE_FACE = "#DCE5EA"
PLANE_EDGE = "#9CA8AF"


def _load_case() -> tuple[dict, dict, dict]:
    record = json.loads(INPUT_ARTIFACT.read_text())
    if record.get("subject_id") != SUBJECT_ID or record.get("status") != "ok":
        raise ValueError(f"unexpected or unavailable record: {record.get('subject_id')}")

    pair = record.get("axis_pair") or {}
    relation = pair.get("relation") or {}
    field = record.get("interictal_field") or {}
    if not bool(pair.get("geometry_2d_supported")):
        raise ValueError("Y9 no longer meets the frozen two-dimensional geometry contract")
    if not bool(pair.get("strict_stability_pass")):
        raise ValueError("Y9 no longer passes the frozen strict-stability contract")
    if not bool(relation.get("collinear")) or relation.get("relation") != "reversed":
        raise ValueError("Y9 is no longer a frozen collinear, reversed TA/TB example")
    if not bool((pair.get("pair_bootstrap") or {}).get("robust_collinear")):
        raise ValueError("Y9 no longer passes the paired-bootstrap collinearity check")
    if not all(key in (field.get("field_models") or {}) for key in ("shared_a", "shared_b")):
        raise ValueError("Y9 frozen artifact is missing shared TA/TB field models")

    dat_a, dat_b, mode = build_interictal_ab_panel_payloads(
        record, display_sigma_mm=DISPLAY_SIGMA_MM,
    )
    if mode != "shared":
        raise ValueError(f"expected shared-plane Y9 artifact, found {mode!r}")
    if not np.allclose(dat_a["xs"], dat_b["xs"]) or not np.allclose(
        dat_a["ys"], dat_b["ys"],
    ):
        raise ValueError("shared-plane TA/TB contact coordinates are not display-aligned")
    return record, dat_a, dat_b


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
    centered = np.asarray(points, float) - np.asarray(basis["origin"], float)
    return np.column_stack(
        [centered @ basis["u"], centered @ basis["w"], centered @ basis["normal"]]
    )


def _vector_to_basis(vector: Sequence[float], basis: Mapping[str, np.ndarray]) -> np.ndarray:
    vec = np.asarray(vector, float)
    return np.asarray([vec @ basis["u"], vec @ basis["w"], vec @ basis["normal"]])


def _draw_geometry(
    ax,
    record: Mapping[str, object],
    dat_a: Mapping[str, object],
    basis: Mapping[str, np.ndarray],
) -> None:
    all_names = np.asarray([str(x) for x in record["names"]], dtype=object)
    all_shafts = np.asarray([str(x) for x in record["shafts"]], dtype=object)
    all_coords = _to_basis(np.asarray(record["coords"], float), basis)
    projected = np.column_stack(
        [
            np.asarray(dat_a["xs"], float),
            np.asarray(dat_a["ys"], float),
            np.zeros(len(all_names)),
        ]
    )
    if not np.allclose(all_coords[:, :2], projected[:, :2], atol=1e-7):
        raise ValueError("frozen 3-D coordinates and shared-plane payload do not agree")

    # H remains part of the frozen axis/plane fit but is intentionally hidden
    # from this paper-facing projection demonstration, as requested.  The
    # omission is recorded explicitly in the sidecar and README.
    keep = all_shafts != "H"
    names = all_names[keep].tolist()
    shafts = all_shafts[keep].tolist()
    coords = all_coords[keep]

    xlim = tuple(float(x) for x in dat_a["frame"]["xlim"])
    ylim = tuple(float(x) for x in dat_a["frame"]["ylim"])
    plane_corners = np.asarray(
        [
            [xlim[0], ylim[0], 0.0],
            [xlim[1], ylim[0], 0.0],
            [xlim[1], ylim[1], 0.0],
            [xlim[0], ylim[1], 0.0],
        ]
    )
    ax.add_collection3d(
        Poly3DCollection(
            [plane_corners], facecolor=PLANE_FACE, edgecolor=PLANE_EDGE,
            linewidth=0.55, alpha=0.34, zorder=0,
        )
    )

    shaft_groups = _shaft_indices(names, shafts)
    for idx in shaft_groups:
        ax.plot(*coords[idx].T, color="#4C555A", lw=1.0, alpha=0.94, zorder=4)

    # Three arrowed orthogonal drops reveal the projection operation without a
    # second, mismatched layer of grey contact markers.
    shaft_candidates = []
    for idx in shaft_groups:
        local = int(idx[np.argmax(coords[idx, 2])])
        shaft_candidates.append(local)
    residual_idx = sorted(
        (i for i in shaft_candidates if coords[i, 2] > 0.5),
        key=lambda i: float(coords[i, 2]), reverse=True,
    )[:3]
    for i in residual_idx:
        ax.plot(
            [coords[i, 0], coords[i, 0]],
            [coords[i, 1], coords[i, 1]],
            [coords[i, 2], 0.0],
            color="#3F6C78", lw=1.05, alpha=0.88, zorder=6.1,
        )
        ax.quiver(
            coords[i, 0], coords[i, 1], coords[i, 2],
            0.0, 0.0, -coords[i, 2],
            color="#3F6C78", lw=1.2, alpha=0.94,
            arrow_length_ratio=0.22, normalize=False, zorder=6.2,
        )

    ax.scatter(
        *coords.T, s=26, facecolor="white", edgecolor=INK,
        linewidth=0.75, depthshade=False, zorder=5,
    )
    if residual_idx:
        source = coords[residual_idx]
        feet = source.copy()
        feet[:, 2] = 0.0
        for points, size in ((source, 34), (feet, 31)):
            ax.scatter(
                *points.T, s=size, facecolor="#3F6C78", edgecolor="white",
                linewidth=0.8, depthshade=False, zorder=5.5,
            )

    # The neutral line is the frozen sign-aligned TA/TB bisector.  Drawing it
    # through the centre of the plane makes the plane construction visible:
    # the two coloured axes straddle the shared line, while electrode spread
    # supplies the transverse dimension of the plane.
    shared_half = 18.5
    axis_half = 15.5
    for key, color, label in (
        ("axis_a", TA_COLOR, "TA"),
        ("axis_b", TB_COLOR, "TB"),
    ):
        vector = _vector_to_basis(record["axis_pair"][key]["u"], basis)
        line = np.vstack((-axis_half * vector, axis_half * vector))
        ax.plot(*line.T, color=color, lw=0.8, alpha=0.56, zorder=6)
        ax.quiver(
            0.0, 0.0, 0.0, *(axis_half * vector),
            color=color, lw=2.0, arrow_length_ratio=0.14,
            normalize=False, zorder=7,
        )
        tip = axis_half * vector
        ax.text(
            *tip, label, color=color, fontsize=7.2, fontweight="bold",
            ha="center", va="center", zorder=8,
        )
    ax.plot(
        [-shared_half, shared_half], [0.0, 0.0], [0.0, 0.0],
        color="#4D565B", lw=0.95, zorder=7.5,
    )

    z_pad = 3.0
    zlim = (
        float(min(all_coords[:, 2].min(), -1.0) - z_pad),
        float(max(all_coords[:, 2].max(), 1.0) + z_pad),
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect(
        (xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    )
    ax.set_proj_type("ortho")
    ax.view_init(elev=24.0, azim=-58.0)
    ax.set_axis_off()
    ax.text2D(
        0.015, 0.965, DISPLAY_LABEL, transform=ax.transAxes,
        ha="left", va="top", fontsize=9.2, fontweight="bold", color=INK,
    )


def _filter_projection_payload(
    payload: Mapping[str, object], shafts: Sequence[str], *, hidden_shaft: str = "H",
) -> tuple[dict, list[str]]:
    shafts_arr = np.asarray([str(x) for x in shafts], dtype=object)
    keep = shafts_arr != hidden_shaft
    out = dict(payload)
    for key in ("names", "xs", "ys", "vals", "rank_values", "sup", "soz"):
        out[key] = np.asarray(payload[key])[keep]
    out["names"] = [str(x) for x in out["names"]]
    out["src_a"] = set()
    out["src_b"] = set()
    out["frame"] = dict(payload["frame"])
    out["frame"]["xlim"] = _limits_with_padding(
        np.asarray(out["xs"], float), include_zero=True, min_span=35.0,
    )
    out["frame"]["ylim"] = _limits_with_padding(
        np.asarray(out["ys"], float), include_zero=True, min_span=35.0,
    )
    return out, shafts_arr[keep].tolist()


def _draw_kernel_projection(
    ax, payload: Mapping[str, object], shafts: Sequence[str],
) -> int:
    xs = np.asarray(payload["xs"], float)
    ys = np.asarray(payload["ys"], float)
    names = [str(x) for x in payload["names"]]
    xlim = tuple(float(x) for x in payload["frame"]["xlim"])
    ylim = tuple(float(x) for x in payload["frame"]["ylim"])

    for idx in _shaft_indices(names, shafts):
        ax.plot(xs[idx], ys[idx], color="#A5ADB1", lw=0.72, zorder=2)

    kernel_index = int(np.argmin(xs**2 + ys**2))
    cx, cy = float(xs[kernel_index]), float(ys[kernel_index])
    sigma = DISPLAY_SIGMA_MM
    gx = np.linspace(*xlim, 260)
    gy = np.linspace(*ylim, 260)
    xx, yy = np.meshgrid(gx, gy)
    gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
    rgba = np.empty((*gaussian.shape, 4), float)
    rgba[..., :3] = np.asarray([0.29, 0.47, 0.53])
    rgba[..., 3] = 0.30 * gaussian * (gaussian >= 0.045)
    ax.imshow(
        rgba, origin="lower", extent=[*xlim, *ylim],
        interpolation="bilinear", zorder=0,
    )
    ax.add_patch(
        Circle(
            (cx, cy), sigma, facecolor="none", edgecolor="#536E77",
            linewidth=0.8, alpha=0.9, zorder=1,
        )
    )

    ax.scatter(
        xs, ys, s=24, facecolor="white", edgecolor="#42494D",
        linewidth=0.62, zorder=3,
    )
    ax.scatter(
        [cx], [cy], s=38, facecolor="#3F6C78", edgecolor="white",
        linewidth=0.9, zorder=5,
    )
    angle = np.deg2rad(24.0)
    ex, ey = cx + sigma * np.cos(angle), cy + sigma * np.sin(angle)
    ax.plot([cx, ex], [cy, ey], color="#536E77", lw=0.8, zorder=4)
    ax.text(
        0.5 * (cx + ex), 0.5 * (cy + ey) + 0.8, "6 mm",
        ha="center", va="bottom", fontsize=5.8, color="#46545A", zorder=5,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#AEB6BA")
        spine.set_linewidth(0.55)
    return kernel_index


def _draw_field(ax, payload: Mapping[str, object], template: str) -> None:
    draw_interictal_rank_field_panel(
        ax, payload, template, compact=True, panel_title="",
        contact_outline_lw=0.8, contact_size=27, show_template_tag=True,
    )
    ax.set_title("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _flow_arrow(fig, left_ax, right_ax, *, use_left_center: bool = True) -> None:
    left = left_ax.get_position()
    right = right_ax.get_position()
    y_box = left if use_left_center else right
    y = 0.5 * (y_box.y0 + y_box.y1)
    fig.add_artist(
        FancyArrowPatch(
            (left.x1 + 0.008, y), (right.x0 - 0.008, y),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=6.8,
            lw=0.72, color="#8B9296", clip_on=False,
        )
    )


def _metadata(
    record: Mapping[str, object],
    dat_a: Mapping[str, object],
    basis: Mapping[str, np.ndarray],
    kernel_contact: str,
) -> dict:
    coords = _to_basis(np.asarray(record["coords"], float), basis)
    relation = record["axis_pair"]["relation"]
    pair_bootstrap = record["axis_pair"]["pair_bootstrap"]
    rank_values = np.concatenate(
        [np.asarray(dat_a["rank_values"], float)]
    )
    return {
        "schema_version": "fig2_y9_template_projection_composite_v2",
        "subject_id": SUBJECT_ID,
        "display_label": DISPLAY_LABEL,
        "input_artifact": str(INPUT_ARTIFACT.resolve()),
        "input_contract": record.get("contract"),
        "input_fingerprint_algorithm": record["interictal_field"]["fingerprint_algorithm"],
        "input_fingerprint_sha256": record["interictal_field"]["fingerprint_sha256"],
        "n_contacts": len(record["names"]),
        "n_shafts": len(set(record["shafts"])),
        "contact_order": record["names"],
        "shafts": record["shafts"],
        "axis_contract": {
            "definition": record["axis_definition"],
            "direction": record["axis_direction_convention"],
            "u_ta": record["axis_pair"]["axis_a"]["u"],
            "u_tb": record["axis_pair"]["axis_b"]["u"],
            "cos_ta_tb": relation["cosine"],
            "abs_cos_ta_tb": relation["abs_cosine"],
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
            "transverse_sign": int(dat_a["transverse_sign"]),
            "normal_residual_max_abs_mm": float(np.max(np.abs(coords[:, 2]))),
            "normal_residual_rms_mm": float(np.sqrt(np.mean(coords[:, 2] ** 2))),
        },
        "rendering": {
            "display_sigma_mm": DISPLAY_SIGMA_MM,
            "projection_display_policy": {
                "hidden_shaft": "H",
                "n_hidden_contacts": int(sum(str(x) == "H" for x in record["shafts"])),
                "n_display_contacts": int(sum(str(x) != "H" for x in record["shafts"])),
                "reason": "user-requested visual omission in this method illustration",
                "axis_and_plane_source": "unchanged frozen all-contact artifact",
                "display_field_contacts": "all frozen field contacts except shaft H",
                "canonical_all_contact_artifact_modified": False,
            },
            "kernel_exemplar_contact": kernel_contact,
            "continuous_field_renderer": (
                "scripts.plot_topic5_interictal_template_ab_fields."
                "draw_interictal_rank_field_panel"
            ),
            "colormap": "viridis",
            "rank_scale": [float(np.nanmin(rank_values)), float(np.nanmax(rank_values))],
            "png_dpi": 400,
            "pdf_fonttype": 42,
            "brain_leader_anchor": "top edge of the left geometry group",
        },
        "claim_boundary": (
            "Patient-specific method illustration of a frozen coordinate transform. The TA/TB "
            "axes and shared plane remain the all-contact frozen solution. Shaft H is hidden only "
            "from the paper-facing projection and display-field illustration; the canonical "
            "artifact is unchanged. The continuous surfaces are support-limited display "
            "interpolations, not measurements in unsampled tissue or a replacement analysis."
        ),
    }


def plot() -> tuple[Path, Path, Path, Path]:
    record, dat_a, dat_b = _load_case()
    basis = _basis(record, int(dat_a["transverse_sign"]))
    shafts = [str(x) for x in record["shafts"]]
    display_a, display_shafts = _filter_projection_payload(dat_a, shafts)
    display_b, display_shafts_b = _filter_projection_payload(dat_b, shafts)
    if display_shafts != display_shafts_b:
        raise ValueError("TA/TB display subsets are not shaft-aligned")

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(7.18, 2.72), facecolor="white")
        gs = gridspec.GridSpec(
            2, 6, figure=fig,
            width_ratios=[2.38, 0.14, 1.16, 0.14, 1.0, 0.055],
            height_ratios=[1.0, 1.0],
            left=0.018, right=0.940, top=0.965, bottom=0.055,
            wspace=0.10, hspace=0.075,
        )
        ax_geometry = fig.add_subplot(gs[:, 0], projection="3d")
        ax_kernel = fig.add_subplot(gs[:, 2])
        ax_field_a = fig.add_subplot(gs[0, 4])
        ax_field_b = fig.add_subplot(gs[1, 4])
        cax = fig.add_subplot(gs[:, 5])

        _draw_geometry(ax_geometry, record, dat_a, basis)
        kernel_index = _draw_kernel_projection(ax_kernel, display_a, display_shafts)
        _draw_field(ax_field_a, display_a, "TA")
        _draw_field(ax_field_b, display_b, "TB")

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
        _flow_arrow(fig, ax_geometry, ax_kernel)
        _flow_arrow(fig, ax_kernel, ax_field_a, use_left_center=False)
        _flow_arrow(fig, ax_kernel, ax_field_b, use_left_center=False)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = "fig2_Y9_template_projection_composite"
        png = OUTPUT_DIR / f"{stem}.png"
        pdf = OUTPUT_DIR / f"{stem}.pdf"
        svg = OUTPUT_DIR / f"{stem}.svg"
        metadata_path = OUTPUT_DIR / f"{stem}_metadata.json"
        fig.savefig(png, dpi=400, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        fig.savefig(svg, facecolor="white")
        plt.close(fig)

    kernel_contact = str(display_a["names"][kernel_index])
    metadata_path.write_text(
        json.dumps(_metadata(record, dat_a, basis, kernel_contact), indent=2) + "\n"
    )
    return png, pdf, svg, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for path in plot():
        print(f"[done] {path}")


if __name__ == "__main__":
    main()
