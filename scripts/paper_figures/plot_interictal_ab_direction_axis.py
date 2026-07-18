#!/usr/bin/env python3
"""Paper-facing interictal Template A/B propagation-direction figure.

The figure deliberately separates three quantities that are easy to conflate:

1. each template's own 3-D propagation axis ``u_A`` / ``u_B``;
2. a shared *line* used only when both axes pass frozen QC and are collinear;
3. propagation direction, whose positive sign is explicitly EARLY -> LATE
   (the negative of the fitted earliness-gradient vector).

The default exemplar is the Yuquan subject ``zhaochenxi`` because the current
masked main-pool artifact has two QC-passing, robustly collinear, reversed axes
and subject-native MRI/CT coordinates plus FreeSurfer anatomy.  The public
manuscript label is intentionally supplied separately with ``--display-label``;
the private ID crosswalk is not inferred from repository folder names.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Noto Sans CJK JP"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.propagation_skeleton_geometry import parse_shaft  # noqa: E402
from src.seeg_coord_loader import load_subject_coords  # noqa: E402
from src.topic5_template_axis_field import (  # noqa: E402
    compute_template_axis_pair,
    z_earliness,
)


RANKDISP_ROOTS = {
    "main": REPO / "results/interictal_propagation_masked/rank_displacement/per_subject",
    "broad": REPO / "results/interictal_propagation_masked_broad/rank_displacement/per_subject",
}
YUQUAN_RECON_ROOT = Path(
    "/mnt/yuquan_data/yuquan_images/nii格式及点电极坐标/"
    "caseAndMRI/yuquan_24h_mriCT/recons"
)
LEGACY_REGION_ROOT = (
    REPO / "ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef/elecs_regionLabel"
)

TEMPLATE_COLORS = {"A": "#B2182B", "B": "#2166AC"}
SHAFT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*"]
UNRESOLVED_REGIONS = {"Deep / white matter", "Unassigned"}


def _normalize_rank(rank: Sequence[float]) -> np.ndarray:
    rank = np.asarray(rank, float)
    lo, hi = float(np.nanmin(rank)), float(np.nanmax(rank))
    if not np.isfinite(lo + hi) or hi <= lo:
        raise ValueError("template rank has no usable range")
    return (rank - lo) / (hi - lo)


def _parse_region_label(label: str) -> str:
    """Turn a FreeSurfer/legacy label into a compact reader-facing name."""
    if not label or label == "Unknown":
        return "Unassigned"
    if "Cerebral-White-Matter" in label:
        return "Deep / white matter"
    text = re.sub(r"^ctx-[lr]h-", "", label)
    replacements = {
        "rostralmiddlefrontal": "Rostral middle frontal",
        "caudalmiddlefrontal": "Caudal middle frontal",
        "superiorfrontal": "Superior frontal",
        "lateralorbitofrontal": "Lateral orbitofrontal",
        "medialorbitofrontal": "Medial orbitofrontal",
        "parsopercularis": "Pars opercularis",
        "parstriangularis": "Pars triangularis",
        "parsorbitalis": "Pars orbitalis",
        "inferiorparietal": "Inferior parietal",
        "supramarginal": "Supramarginal",
        "middletemporal": "Middle temporal",
        "inferiortemporal": "Inferior temporal",
        "superiortemporal": "Superior temporal",
        "parahippocampal": "Parahippocampal",
        "posteriorcingulate": "Posterior cingulate",
        "caudalanteriorcingulate": "Caudal anterior cingulate",
        "rostralanteriorcingulate": "Rostral anterior cingulate",
        "postcentral": "Postcentral",
        "precentral": "Precentral",
        "fusiform": "Fusiform",
        "insula": "Insula",
    }
    if text in replacements:
        return replacements[text]
    return text.replace("-", " ").replace("_", " ").strip().title()


def _contact_region(labels: Mapping[str, Sequence[str]], channel: str) -> str:
    match = re.fullmatch(r"([A-Za-z]+\'?)(\d+)", channel.strip())
    if match is None:
        return "Unassigned"
    shaft, ordinal = match.group(1), int(match.group(2))
    values = labels.get(shaft, [])
    if ordinal < 1 or ordinal > len(values):
        return "Unassigned"
    return _parse_region_label(str(values[ordinal - 1]))


def _load_regions(subject: str, channels: Sequence[str]) -> tuple[np.ndarray, str]:
    path = LEGACY_REGION_ROOT / f"{subject}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"anatomical region labels not found for {subject}: {path}; "
            "do not infer regions from shaft names"
        )
    labels = np.load(path, allow_pickle=True).item()
    regions = np.asarray([_contact_region(labels, ch) for ch in channels], dtype=object)
    return regions, str(path.resolve())


def _load_brain_mesh(subject: str, *, step_size: int = 3) -> tuple[np.ndarray, np.ndarray, str]:
    path = YUQUAN_RECON_ROOT / subject / "mri/brainmask.mgz"
    if not path.exists():
        raise FileNotFoundError(f"subject-native brain mask not found: {path}")
    img = nib.load(str(path))
    mask = np.asarray(img.dataobj) > 0
    verts_vox, faces, _, _ = marching_cubes(
        mask.astype(np.uint8), level=0.5, step_size=step_size, allow_degenerate=False
    )
    verts_ras = nib.affines.apply_affine(img.affine, verts_vox)
    return verts_ras, faces, str(path.resolve())


def _fit_order_on_axis(along: np.ndarray, order: np.ndarray) -> Dict[str, float | int]:
    slope, intercept = np.polyfit(along, order, 1)
    pred = intercept + slope * along
    ss_tot = float(np.sum((order - np.mean(order)) ** 2))
    ss_res = float(np.sum((order - pred) ** 2))
    r = float(np.corrcoef(along, order)[0, 1])
    return {
        "slope_per_mm": float(slope),
        "intercept": float(intercept),
        "pearson_r": r,
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        # rank/order grows from early to late; this is the propagation sign on u_shared.
        "propagation_sign_on_shared_axis": int(1 if slope > 0 else -1),
    }


def _route_endpoints(regions: np.ndarray, order: np.ndarray, *, min_contacts: int = 2) -> Dict[str, object]:
    """Descriptive region-level early/late endpoints; never used to fit an axis."""
    rows = []
    for region in sorted(set(regions.tolist())):
        if region in UNRESOLVED_REGIONS:
            continue
        values = order[regions == region]
        if len(values) >= min_contacts:
            rows.append((region, int(len(values)), float(np.median(values))))
    if len(rows) < 2:
        return {"status": "not_resolved", "eligible_regions": rows}
    rows.sort(key=lambda x: x[2])
    return {
        "status": "ok",
        "early_region": rows[0][0],
        "late_region": rows[-1][0],
        "eligible_regions": rows,
    }


def load_case(dataset: str, subject: str, pool: str, *, n_axis_boot: int, n_pair_boot: int) -> Dict[str, object]:
    if dataset != "yuquan":
        raise ValueError(
            "this manuscript figure currently requires Yuquan subject-native FreeSurfer anatomy; "
            "the Epilepsiae MNI/atlas rendering branch is not interchangeable"
        )
    source = RANKDISP_ROOTS[pool] / f"{dataset}_{subject}.json"
    if not source.exists():
        raise FileNotFoundError(source)
    record = json.loads(source.read_text())
    pair = record["pairs"][0]
    names = np.asarray(pair["channel_names"], dtype=object)
    joint = np.asarray(pair["joint_valid"], dtype=bool)
    rank_a = np.asarray(pair["rank_a_dense_full"], float)[joint]
    rank_b = np.asarray(pair["rank_b_dense_full"], float)[joint]
    joint_names = names[joint].tolist()

    coord = load_subject_coords(dataset, subject, joint_names)
    mapped = np.asarray(coord.mapped_mask_in_requested_order, dtype=bool)
    channels = np.asarray(joint_names, dtype=object)[mapped]
    coords = np.asarray(coord.coords_array_in_requested_order, float)[mapped]
    rank_a, rank_b = rank_a[mapped], rank_b[mapped]
    shafts = np.asarray([parse_shaft(ch)[0] for ch in channels], dtype=object)
    if len(channels) < 6:
        raise ValueError(f"only {len(channels)} joint-valid, coordinate-mapped contacts")

    axes = compute_template_axis_pair(
        coords,
        rank_a,
        rank_b,
        shafts,
        n_axis_boot=n_axis_boot,
        n_pair_boot=n_pair_boot,
        seed=0,
    )
    if axes.get("status") != "ok":
        raise ValueError(f"template axes undefined: {axes.get('status')}")
    if not axes.get("axis_pair_qc_pass"):
        raise ValueError("template A/B axes do not both pass the frozen axis-QC contract")
    if not axes["relation"]["collinear"]:
        raise ValueError(
            f"template axes are not collinear (line angle={axes['relation']['line_angle_deg']:.1f} deg); "
            "a shared direction line would be misleading"
        )
    if axes["shared_axis"].get("status") != "ok":
        raise ValueError(f"shared line undefined: {axes['shared_axis'].get('status')}")

    u_shared = np.asarray(axes["shared_axis"]["u"], float)
    xbar = coords.mean(0)
    along = (coords - xbar) @ u_shared
    # Make the displayed shared line point along Template-A propagation
    # (early->late).  This removes the harmless sign ambiguity of a line.
    e_a = z_earliness(rank_a)
    e_b = z_earliness(rank_b)
    if np.corrcoef(along, rank_a)[0, 1] < 0:
        u_shared = -u_shared
        along = -along

    order_a, order_b = _normalize_rank(rank_a), _normalize_rank(rank_b)
    fit_a = _fit_order_on_axis(along, order_a)
    fit_b = _fit_order_on_axis(along, order_b)
    regions, region_source = _load_regions(subject, channels.tolist())
    mesh_v, mesh_f, mesh_source = _load_brain_mesh(subject)
    soz = set(record.get("soz_channels", []))

    return {
        "dataset": dataset,
        "subject": subject,
        "pool": pool,
        "source": str(source.resolve()),
        "channels": channels,
        "coords": coords,
        "shafts": shafts,
        "rank_a": rank_a,
        "rank_b": rank_b,
        "order_a": order_a,
        "order_b": order_b,
        "earliness_a": e_a,
        "earliness_b": e_b,
        "dab": e_a - e_b,
        "axes": axes,
        "u_shared": u_shared,
        "xbar": xbar,
        "along": along,
        "fit_a": fit_a,
        "fit_b": fit_b,
        "regions": regions,
        "region_source": region_source,
        "route_a": _route_endpoints(regions, order_a),
        "route_b": _route_endpoints(regions, order_b),
        "mesh_vertices": mesh_v,
        "mesh_faces": mesh_f,
        "mesh_source": mesh_source,
        "soz": soz,
        "coord_space": coord.coord_space,
        "coord_source": coord.provenance.get("source_path", ""),
    }


def _panel_label(ax, label: str, *, is3d: bool = False) -> None:
    if is3d:
        ax.text2D(-0.12, 1.02, label, transform=ax.transAxes, fontsize=12, fontweight="bold")
    else:
        ax.text(-0.18, 1.10, label, transform=ax.transAxes, fontsize=12, fontweight="bold")


def _plot_gradient(ax, case: Mapping[str, object], template: str, *, show_ylabel: bool) -> None:
    along = np.asarray(case["along"], float)
    order = np.asarray(case[f"order_{template.lower()}"], float)
    fit = case[f"fit_{template.lower()}"]
    shafts = np.asarray(case["shafts"], object)
    unique_shafts = sorted(set(shafts.tolist()))
    marker_by_shaft = {s: SHAFT_MARKERS[i % len(SHAFT_MARKERS)] for i, s in enumerate(unique_shafts)}
    norm = plt.Normalize(0.0, 1.0)
    for shaft in unique_shafts:
        mask = shafts == shaft
        ax.scatter(
            along[mask],
            order[mask],
            c=order[mask],
            cmap="viridis",
            norm=norm,
            marker=marker_by_shaft[shaft],
            s=39,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    xx = np.linspace(float(np.min(along)), float(np.max(along)), 100)
    yy = float(fit["intercept"]) + float(fit["slope_per_mm"]) * xx
    ax.plot(xx, yy, color=TEMPLATE_COLORS[template], lw=2.0, zorder=2)
    ax.set_ylim(1.04, -0.04)  # early at the top, late at the bottom
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlim(float(np.min(along)) - 1.0, float(np.max(along)) + 1.0)
    ax.set_xlabel("position on shared line (mm)", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("temporal order\n0 early, 1 late", fontsize=8)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(labelsize=7, length=2.5)
    ax.grid(color="0.9", lw=0.6, zorder=0)
    direction = "+" if int(fit["propagation_sign_on_shared_axis"]) > 0 else "−"
    ax.set_title(
        f"Template {template}   r={float(fit['pearson_r']):+.2f}, "
        f"R²={float(fit['R2']):.2f}, direction={direction}axis",
        color=TEMPLATE_COLORS[template],
        fontsize=8.2,
        pad=4,
    )
    lo, hi = np.percentile(along, [5, 95])
    start, end = (lo, hi) if int(fit["propagation_sign_on_shared_axis"]) > 0 else (hi, lo)
    ax.annotate(
        "",
        xy=(end, 0.08),
        xytext=(start, 0.08),
        arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": TEMPLATE_COLORS[template]},
    )
    ax.text((start + end) / 2, 0.02, "early → late", ha="center", va="bottom", fontsize=6.7)


def _plot_anatomy(ax, case: Mapping[str, object]) -> None:
    regions = np.asarray(case["regions"], object)
    order_a = np.asarray(case["order_a"], float)
    order_b = np.asarray(case["order_b"], float)
    along = np.asarray(case["along"], float)
    unique = sorted(set(regions.tolist()), key=lambda r: float(np.median(along[regions == r])))
    y = np.arange(len(unique))
    rng = np.random.default_rng(0)
    for yi, region in enumerate(unique):
        mask = regions == region
        # Raw contacts are shown faintly; the large symbols are region medians.
        jitter = rng.uniform(-0.07, 0.07, int(mask.sum()))
        ax.scatter(order_a[mask], yi + 0.12 + jitter, s=10, c=TEMPLATE_COLORS["A"], alpha=0.22)
        ax.scatter(order_b[mask], yi - 0.12 + jitter, s=10, c=TEMPLATE_COLORS["B"], alpha=0.22)
        med_a = float(np.median(order_a[mask]))
        med_b = float(np.median(order_b[mask]))
        ax.plot([med_a, med_b], [yi, yi], color="0.78", lw=1.0, zorder=1)
        ax.scatter(med_a, yi + 0.08, s=39, c=TEMPLATE_COLORS["A"], marker="o", zorder=3)
        ax.scatter(med_b, yi - 0.08, s=39, c=TEMPLATE_COLORS["B"], marker="s", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r}  (n={int(np.sum(regions == r))})" for r in unique], fontsize=6.7)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("regional median temporal order   (early 0 → 1 late)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TEMPLATE_COLORS["A"], label="Template A"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=TEMPLATE_COLORS["B"], label="Template B"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7, ncol=2)
    ax.set_title("Anatomical labels along the interictal timing gradient", loc="left", fontsize=9)

    def _route_text(route: Mapping[str, object], template: str) -> str:
        if route.get("status") != "ok":
            return f"{template}: region-level route not resolved"
        return f"{template}: {route['early_region']} → {route['late_region']}"

    ax.text(
        0.0,
        -0.22,
        _route_text(case["route_a"], "A") + "     " + _route_text(case["route_b"], "B")
        + "\nDescriptive anatomy overlay only; regions are not used to fit or orient the axes.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color="0.25",
    )


def _set_3d_equal(ax, vertices: np.ndarray, *, padding_mm: float = 27.0) -> None:
    """Zoom around the implanted field while retaining the transparent brain context."""
    mins, maxs = np.nanmin(vertices, axis=0), np.nanmax(vertices, axis=0)
    center = (mins + maxs) / 2
    radius = float(np.max(maxs - mins) / 2 + padding_mm)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _plot_brain(ax, case: Mapping[str, object]) -> None:
    vertices = np.asarray(case["mesh_vertices"], float)
    faces = np.asarray(case["mesh_faces"], int)
    mesh = Poly3DCollection(
        vertices[faces],
        facecolor=(0.68, 0.70, 0.74, 0.035),
        edgecolor=(0.45, 0.47, 0.50, 0.025),
        linewidth=0.05,
    )
    ax.add_collection3d(mesh)

    coords = np.asarray(case["coords"], float)
    channels = np.asarray(case["channels"], object)
    shafts = np.asarray(case["shafts"], object)
    dab = np.asarray(case["dab"], float)
    vmax = float(np.max(np.abs(dab)))
    # Shaft trajectories are sampling geometry, not propagation arrows.
    for shaft in sorted(set(shafts.tolist())):
        idx = np.where(shafts == shaft)[0]
        nums = np.asarray([int(re.search(r"(\d+)$", str(ch)).group(1)) for ch in channels[idx]])
        idx = idx[np.argsort(nums)]
        ax.plot(*coords[idx].T, color="0.45", lw=0.75, alpha=0.8, zorder=3)
    edge = ["black" if str(ch) in case["soz"] else "white" for ch in channels]
    widths = [1.45 if str(ch) in case["soz"] else 0.35 for ch in channels]
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=dab,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        s=58,
        edgecolors=edge,
        linewidths=widths,
        depthshade=False,
        zorder=5,
    )

    u = np.asarray(case["u_shared"], float)
    xbar = np.asarray(case["xbar"], float)
    along = np.asarray(case["along"], float)
    lo, hi = np.percentile(along, [3, 97])
    pad = max(3.0, 0.08 * float(hi - lo))
    lo, hi = float(lo - pad), float(hi + pad)
    resid = coords - xbar - np.outer(along, u)
    transverse = np.linalg.svd(resid, full_matrices=False)[2][0]
    offset_mm = 4.5
    ax.plot(
        *(np.vstack((xbar + lo * u, xbar + hi * u)).T),
        color="0.25",
        lw=1.1,
        ls=(0, (2, 2)),
        zorder=6,
    )
    for template, offset_sign in (("A", 1.0), ("B", -1.0)):
        fit = case[f"fit_{template.lower()}"]
        start_s, end_s = (lo, hi) if int(fit["propagation_sign_on_shared_axis"]) > 0 else (hi, lo)
        start = xbar + start_s * u + offset_sign * offset_mm * transverse
        end = xbar + end_s * u + offset_sign * offset_mm * transverse
        vec = end - start
        ax.quiver(
            start[0],
            start[1],
            start[2],
            vec[0],
            vec[1],
            vec[2],
            color=TEMPLATE_COLORS[template],
            lw=3.8,
            arrow_length_ratio=0.16,
            normalize=False,
            zorder=8,
        )

    # Orthographic view normal to the contact cloud's best-fit plane.  This
    # maximizes visible contact/axis spread instead of collapsing depth shafts.
    camera = np.linalg.svd(coords - coords.mean(0), full_matrices=False)[2][-1]
    if camera[0] < 0:
        camera = -camera
    elev = float(np.degrees(np.arcsin(np.clip(camera[2], -1.0, 1.0))))
    azim = float(np.degrees(np.arctan2(camera[1], camera[0])))
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    _set_3d_equal(ax, coords)
    ax.set_axis_off()
    relation = case["axes"]["relation"]
    ax.set_title(
        "Template-specific early → late directions in subject-native SEEG space\n"
        f"line angle={float(relation['line_angle_deg']):.1f}°, relation={relation['relation']}; "
        "SOZ=black ring (overlay only)",
        fontsize=9,
        pad=0,
    )
    cax = ax.inset_axes([0.22, 0.03, 0.56, 0.022])
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-vmax, vmax))
    cb = ax.figure.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([-vmax, 0.0, vmax])
    cb.set_ticklabels(["B-leading", "0", "A-leading"])
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cb.set_label("relative earliness contrast", fontsize=6.5, labelpad=1)
    ax.text2D(
        0.5,
        0.11,
        "red arrow = Template A early → late     blue arrow = Template B early → late",
        transform=ax.transAxes,
        ha="center",
        fontsize=6.6,
        color="0.25",
    )


def _closed_path_patch(
    points: Sequence[Sequence[float]],
    *,
    facecolor: str,
    alpha: float,
    edgecolor: str = "none",
    linewidth: float = 0.0,
    zorder: float = 1.0,
) -> PathPatch:
    """Return a smooth closed Bezier patch in normalized brain-panel coordinates."""
    vertices = np.asarray(points, float)
    if len(vertices) < 4 or (len(vertices) - 1) % 3 != 0:
        raise ValueError("closed Bezier path requires 1 + 3k vertices")
    # The supplied final CURVE4 endpoint already returns to the first vertex.
    # CLOSEPOLY still needs its own dummy vertex; replacing that final CURVE4
    # would break the cubic grouping and can produce giant raster wedges.
    vertices = np.vstack((vertices, vertices[0]))
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * (len(vertices) - 2) + [MplPath.CLOSEPOLY]
    return PathPatch(
        MplPath(vertices, codes),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )


def _brain_schematic_patches(ax) -> None:
    """Draw the Illustrator-like lateral brain context used by the legacy figure.

    This is deliberately a schematic visual layer.  Contact positions and arrows
    are projected from subject-native coordinates and never inferred from these
    pastel regions.
    """
    # Outer lateral silhouette: a clean vector redraw of the visual language in
    # legacy Supplementary Fig. S6, not a raster crop of that panel.
    outline = [
        (0.12, 0.48),
        (0.07, 0.63), (0.11, 0.80), (0.25, 0.88),
        (0.32, 0.94), (0.39, 0.94), (0.44, 0.89),
        (0.58, 0.96), (0.79, 0.91), (0.89, 0.77),
        (0.98, 0.64), (0.93, 0.51), (0.93, 0.40),
        (0.89, 0.30), (0.79, 0.28), (0.67, 0.27),
        (0.60, 0.18), (0.50, 0.14), (0.44, 0.23),
        (0.38, 0.30), (0.31, 0.21), (0.22, 0.25),
        (0.16, 0.27), (0.16, 0.39), (0.12, 0.48),
    ]
    outline_patch = _closed_path_patch(
        outline,
        facecolor="white",
        alpha=1.0,
        edgecolor="none",
        zorder=0.5,
    )
    ax.add_patch(outline_patch)

    # Soft regional fields reproduce the restrained S6 palette.  They provide
    # orientation only; the actual region labels remain in panel b.
    region_specs = [
        (
            [(0.12, 0.48), (0.09, 0.60), (0.13, 0.73), (0.23, 0.79),
             (0.31, 0.80), (0.35, 0.70), (0.31, 0.58),
             (0.28, 0.47), (0.25, 0.32), (0.22, 0.25),
             (0.16, 0.27), (0.16, 0.39), (0.12, 0.48)],
            "#E6A3EA", 0.48,
        ),
        (
            [(0.25, 0.59), (0.31, 0.73), (0.43, 0.82), (0.56, 0.78),
             (0.63, 0.72), (0.58, 0.58), (0.49, 0.53),
             (0.43, 0.47), (0.42, 0.36), (0.34, 0.32),
             (0.28, 0.38), (0.23, 0.49), (0.25, 0.59)],
            "#C9C6F3", 0.46,
        ),
        (
            [(0.17, 0.48), (0.20, 0.57), (0.27, 0.64), (0.34, 0.64),
             (0.40, 0.61), (0.38, 0.51), (0.32, 0.46),
             (0.28, 0.40), (0.23, 0.38), (0.20, 0.41),
             (0.18, 0.42), (0.17, 0.45), (0.17, 0.48)],
            "#A9ECF3", 0.56,
        ),
        (
            [(0.50, 0.44), (0.55, 0.53), (0.65, 0.58), (0.73, 0.54),
             (0.79, 0.48), (0.76, 0.37), (0.69, 0.33),
             (0.63, 0.29), (0.56, 0.30), (0.52, 0.35),
             (0.49, 0.37), (0.48, 0.40), (0.50, 0.44)],
            "#A7EAF3", 0.51,
        ),
        (
            [(0.20, 0.31), (0.31, 0.28), (0.38, 0.31), (0.44, 0.27),
             (0.51, 0.22), (0.59, 0.23), (0.67, 0.27),
             (0.60, 0.18), (0.50, 0.14), (0.44, 0.23),
             (0.38, 0.30), (0.29, 0.20), (0.20, 0.31)],
            "#E7A1E9", 0.44,
        ),
    ]
    for points, color, alpha in region_specs:
        region_patch = _closed_path_patch(points, facecolor=color, alpha=alpha, zorder=1.0)
        region_patch.set_clip_path(outline_patch)
        ax.add_patch(region_patch)

    # Re-stroke the silhouette after the filled fields so the contour stays crisp.
    ax.add_patch(
        _closed_path_patch(
            outline,
            facecolor="none",
            alpha=1.0,
            edgecolor="0.30",
            linewidth=1.05,
            zorder=2.2,
        )
    )

    # The internal curve gives the same sparse hand-drawn anatomical cue as S6.
    ax.plot(
        [0.44, 0.39, 0.31, 0.29, 0.25, 0.23],
        [0.89, 0.82, 0.70, 0.57, 0.48, 0.26],
        color="0.30",
        lw=0.9,
        solid_capstyle="round",
        zorder=2,
    )


def _lateral_brain_coordinates(case: Mapping[str, object]) -> tuple[np.ndarray, Dict[str, float]]:
    """Map subject-native RAS contacts to a lateral anterior-posterior/superior-inferior view."""
    mesh = np.asarray(case["mesh_vertices"], float)
    coords = np.asarray(case["coords"], float)
    y_min, y_max = np.nanmin(mesh[:, 1]), np.nanmax(mesh[:, 1])
    z_min, z_max = np.nanmin(mesh[:, 2]), np.nanmax(mesh[:, 2])
    if y_max <= y_min or z_max <= z_min:
        raise ValueError("brain bounds are degenerate in the AP-SI projection")
    # Neurological lateral convention used here: anterior is left, superior is up.
    x = 0.12 + 0.81 * (y_max - coords[:, 1]) / (y_max - y_min)
    y = 0.17 + 0.72 * (coords[:, 2] - z_min) / (z_max - z_min)
    return np.column_stack((x, y)), {
        "y_min": float(y_min),
        "y_max": float(y_max),
        "z_min": float(z_min),
        "z_max": float(z_max),
    }


def _lateral_propagation_direction(u_propagation: Sequence[float]) -> tuple[np.ndarray, float]:
    """Project a 3-D early-to-late propagation axis in the AP-SI view."""
    u = np.asarray(u_propagation, float)
    if u.shape != (3,) or not np.all(np.isfinite(u)):
        raise ValueError("propagation axis must be one finite 3-D vector")
    norm = float(np.linalg.norm(u))
    if norm <= 0:
        raise ValueError("propagation axis has zero length")
    u = u / norm
    # Screen x is -RAS anterior/posterior; screen y is RAS superior/inferior.
    projected = np.asarray([-u[1], u[2]], float)
    projection_fraction = float(np.linalg.norm(projected))
    if projection_fraction <= 0:
        raise ValueError("axis has no AP-SI component")
    return projected / projection_fraction, projection_fraction


def _plot_brain_schematic(ax, case: Mapping[str, object]) -> None:
    """Plot current A/B propagation axes on an S6-like lateral brain schematic."""
    _brain_schematic_patches(ax)
    xy, _ = _lateral_brain_coordinates(case)
    channels = np.asarray(case["channels"], object)
    shafts = np.asarray(case["shafts"], object)
    dab = np.asarray(case["dab"], float)
    vmax = float(np.max(np.abs(dab)))

    for shaft in sorted(set(shafts.tolist())):
        idx = np.where(shafts == shaft)[0]
        nums = np.asarray([int(re.search(r"(\d+)$", str(ch)).group(1)) for ch in channels[idx]])
        idx = idx[np.argsort(nums)]
        ax.plot(xy[idx, 0], xy[idx, 1], color="0.38", lw=0.8, alpha=0.78, zorder=3)
    edge = ["black" if str(ch) in case["soz"] else "white" for ch in channels]
    widths = [1.25 if str(ch) in case["soz"] else 0.35 for ch in channels]
    scatter = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=dab,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        s=42,
        edgecolors=edge,
        linewidths=widths,
        zorder=5,
    )

    center = np.mean(xy, axis=0)
    arrow_length = 0.24
    arrow_info: Dict[str, Dict[str, float]] = {}
    for template, offset in (("A", 0.020), ("B", -0.020)):
        u = np.asarray(case["axes"][f"axis_{template.lower()}"]["u"], float)
        direction, projection_fraction = _lateral_propagation_direction(u)
        if projection_fraction < 0.35:
            raise ValueError(
                f"Template {template} AP-SI projection retains only "
                f"{projection_fraction:.2f} of the 3-D axis; lateral arrow would be misleading"
            )
        normal = np.asarray([-direction[1], direction[0]])
        start = center - 0.5 * arrow_length * direction + offset * normal
        end = center + 0.5 * arrow_length * direction + offset * normal
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "mutation_scale": 17,
                "lw": 3.1,
                "color": TEMPLATE_COLORS[template],
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=8,
        )
        label_pos = end + 0.055 * direction + 0.035 * normal
        ax.text(
            label_pos[0],
            label_pos[1],
            f"{template}  early→late",
            color=TEMPLATE_COLORS[template],
            fontsize=7.2,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            zorder=9,
        )
        arrow_info[template] = {
            "ap_si_projection_fraction": projection_fraction,
            "screen_dx": float(direction[0]),
            "screen_dy": float(direction[1]),
        }
    case["schematic_arrow_info"] = arrow_info

    ax.text(0.09, 0.93, "anterior", ha="left", va="bottom", fontsize=6.4, color="0.35")
    ax.text(0.94, 0.93, "posterior", ha="right", va="bottom", fontsize=6.4, color="0.35")
    ax.annotate(
        "",
        xy=(0.13, 0.91),
        xytext=(0.31, 0.91),
        arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "0.35"},
    )
    ax.set_xlim(0.04, 1.00)
    ax.set_ylim(0.09, 0.99)
    ax.set_aspect("equal")
    ax.set_axis_off()
    relation = case["axes"]["relation"]
    ax.set_title(
        "Template directions on a lateral brain schematic\n"
        f"AP–SI projection of subject-native 3-D axes; relation={relation['relation']}",
        fontsize=9,
        pad=4,
    )
    cax = ax.inset_axes([0.23, 0.02, 0.54, 0.025])
    cb = ax.figure.colorbar(scatter, cax=cax, orientation="horizontal")
    cb.set_ticks([-vmax, 0.0, vmax])
    cb.set_ticklabels(["B-leading", "0", "A-leading"])
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cb.set_label("relative earliness contrast", fontsize=6.5, labelpad=1)
    ax.text(
        0.50,
        0.10,
        "Pastel anatomy is display-only; contacts and arrows come from subject-native coordinates.",
        transform=ax.transAxes,
        ha="center",
        fontsize=6.3,
        color="0.30",
    )


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def plot_case(
    case: Mapping[str, object],
    display_label: str,
    out_png: Path,
    *,
    brain_style: str = "schematic",
) -> None:
    fig = plt.figure(figsize=(12.2, 5.9))
    gs = gridspec.GridSpec(
        2,
        3,
        width_ratios=[1.05, 1.05, 2.0],
        height_ratios=[1.0, 1.15],
        left=0.15,
        right=0.985,
        top=0.83,
        bottom=0.16,
        wspace=0.32,
        hspace=0.48,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharex=ax_a, sharey=ax_a)
    ax_anat = fig.add_subplot(gs[1, :2])
    if brain_style == "surface":
        ax_brain = fig.add_subplot(gs[:, 2], projection="3d")
    elif brain_style == "schematic":
        ax_brain = fig.add_subplot(gs[:, 2])
    else:
        raise ValueError(f"unknown brain style: {brain_style}")

    _plot_gradient(ax_a, case, "A", show_ylabel=True)
    _plot_gradient(ax_b, case, "B", show_ylabel=False)
    _plot_anatomy(ax_anat, case)
    if brain_style == "surface":
        _plot_brain(ax_brain, case)
    else:
        _plot_brain_schematic(ax_brain, case)
    _panel_label(ax_a, "a")
    _panel_label(ax_anat, "b")
    _panel_label(ax_brain, "c", is3d=brain_style == "surface")

    relation = case["axes"]["relation"]
    pair_boot = case["axes"]["pair_bootstrap"]
    fig.suptitle(
        f"{display_label} | interictal Template A/B propagation directions",
        x=0.15,
        y=0.97,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.15,
        0.92,
        f"Two independently fitted early-to-late propagation axes; shared line shown only after dual-axis QC "
        f"and collinearity (|cos|={float(relation['abs_cosine']):.2f}, "
        f"paired-bootstrap sign stability={float(pair_boot['p_sign_stable']):.2f}).",
        ha="left",
        fontsize=8,
        color="0.25",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)

    metadata = {
        "schema_version": "interictal_ab_direction_figure_v2",
        "figure_png": str(out_png.resolve()),
        "figure_pdf": str(out_pdf.resolve()),
        "display_label": display_label,
        "dataset": case["dataset"],
        "internal_subject_id": case["subject"],
        "pool": case["pool"],
        "input_artifact": case["source"],
        "coord_space": case["coord_space"],
        "coord_source": case["coord_source"],
        "region_source": case["region_source"],
        "brain_surface_source": case["mesh_source"],
        "brain_panel_style": brain_style,
        "brain_panel_projection": (
            "subject_native_3d" if brain_style == "surface" else "lateral_AP_SI_from_subject_native_RAS"
        ),
        "schematic_arrow_info": case.get("schematic_arrow_info", {}),
        "axis_contract": {
            "template_axes": "template_propagation_axis_v2",
            "scalar": "e_T=-z(rank_T)",
            "solver": "truncated least squares, relative singular-value floor 0.05",
            "shared_line": "aligned angular bisector, only after dual-axis QC and |cos(uA,uB)|>=0.5",
            "direction": "u_T=-normalize(gradient(e_T)); positive is early_to_late",
            "anatomy_role": "display_only_not_axis_input",
        },
        "n_contacts": int(len(case["channels"])),
        "n_shafts": int(len(set(np.asarray(case["shafts"], object).tolist()))),
        "channels": case["channels"],
        "regions": case["regions"],
        "template_a_fit_on_shared_line": case["fit_a"],
        "template_b_fit_on_shared_line": case["fit_b"],
        "route_a": case["route_a"],
        "route_b": case["route_b"],
        "axis_pair": case["axes"],
    }
    out_png.with_name(out_png.stem + "_metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png.with_name(out_png.stem + '_metadata.json')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yuquan", choices=["yuquan"])
    parser.add_argument("--subject", default="zhaochenxi")
    parser.add_argument("--pool", default="main", choices=sorted(RANKDISP_ROOTS))
    parser.add_argument(
        "--display-label",
        default="Yuquan example",
        help="Reader-facing label; do not infer Y-number from the artifact folder.",
    )
    parser.add_argument("--n-axis-boot", type=int, default=500)
    parser.add_argument("--n-pair-boot", type=int, default=1000)
    parser.add_argument(
        "--brain-style",
        default="schematic",
        choices=["schematic", "surface"],
        help="Use the S6-like vector schematic or the transparent subject-native 3-D surface.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO
        / "results/paper-ready-figure/fig_interictal_ab_direction_axis/figures/"
        "yuquan_example_interictal_ab_direction_axis.png",
    )
    args = parser.parse_args()
    case = load_case(
        args.dataset,
        args.subject,
        args.pool,
        n_axis_boot=args.n_axis_boot,
        n_pair_boot=args.n_pair_boot,
    )
    plot_case(case, args.display_label, args.out, brain_style=args.brain_style)


if __name__ == "__main__":
    main()
