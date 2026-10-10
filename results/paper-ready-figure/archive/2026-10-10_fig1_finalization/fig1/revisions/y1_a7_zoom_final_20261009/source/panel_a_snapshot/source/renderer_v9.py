#!/usr/bin/env python3
"""Build a data-derived Y3 implantation / K-shaft / recording Fig1A candidate.

The cortex and all contact centers are patient data. Connector lines and marker
diameters are display aids, not measured cable paths or manufacturer dimensions.
The Fig1B2 excerpts and filter are preserved; only K-shaft channels are shown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
import pyvista as pv
from matplotlib.patches import Ellipse, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from scipy.signal import butter, filtfilt, iirnotch, resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures.build_main_figures_1_2 import _compose_complete_layout
from src.seeg_coord_loader import YUQUAN_ELEC_ROOT, load_subject_coords

CANONICAL = ROOT / "results/paper-ready-figure/fig1"
DEFAULT_OUTPUT = CANONICAL / "candidates/recording_chain_connector_cap_20261009"
RECON = YUQUAN_ELEC_ROOT.parent / "recons/chengshuai"
PURPLE = "#8b70ba"
BLUE = "#24a9ce"
CHANNEL_COLORS = {"K5-K6": PURPLE, "K9-K10": BLUE}
GRAY = "#929292"
INK = "#292929"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_channel(name: str) -> str:
    name = re.sub(r"^(EEG|POL)\s+", "", name.strip())
    return re.sub(r"-(Ref|REF)$", "", name).replace(" ", "")


def load_recordings(source: dict) -> tuple[np.ndarray, dict]:
    """Same numerical operations as Fig1B2's _load_bipolar_event_snippets."""
    channels = source["selection"]["selected_channels"]
    windows = np.array(source["selection"]["event_windows_sec"], float)
    packed_path = Path(source["source_paths"]["packed_times"])
    packed_windows = np.load(packed_path)[source["selection"]["event_indices"]]
    np.testing.assert_array_equal(windows, packed_windows)
    fs_out = source["plot"]["fs_out"]
    duration = source["plot"]["window_sec"]
    band = source["plot"]["band_hz"]
    raw = mne.io.read_raw_edf(
        source["source_paths"]["edf"], preload=False, encoding="latin1", verbose="ERROR"
    )
    fs_in = float(raw.info["sfreq"])
    mapping = {normalize_channel(n): i for i, n in enumerate(raw.ch_names)}
    pairs = [name.split("-") for name in channels]
    needed = sorted({mapping[n] for pair in pairs for n in pair})
    to_row = {pick: row for row, pick in enumerate(needed)}
    segments, sample_bounds, raw_hashes = [], [], []
    target = int(round(duration * fs_out))
    for start, end in windows:
        crop_start = max(0.0, 0.5 * (start + end) - duration / 2)
        i0, i1 = (int(round(t * fs_in)) for t in (crop_start, crop_start + duration))
        data = raw.get_data(picks=needed, start=i0, stop=i1)
        raw_hashes.append(hashlib.sha256(data.tobytes()).hexdigest())
        bipolar = np.array([data[to_row[mapping[pair[0]]]] - data[to_row[mapping[pair[1]]]]
                            if len(pair) == 2 else data[to_row[mapping[pair[0]]]]
                            for pair in pairs], dtype=np.float64)
        if fs_out != fs_in:
            bipolar = resample_poly(bipolar, int(round(fs_out)), int(round(fs_in)), axis=-1)
        for freq in (50, 100, 150, 200, 250):
            if freq < fs_out / 2:
                b, a = iirnotch(float(freq), Q=30.0, fs=fs_out)
                bipolar = filtfilt(b, a, bipolar, axis=-1)
        b, a = butter(3, np.array(band) / (fs_out / 2), btype="bandpass")
        bipolar = filtfilt(b, a, bipolar, axis=-1)
        assert bipolar.shape == (len(channels), target)
        segments.append(bipolar)
        sample_bounds.append([i0, i1])
    raw.close()
    traces = np.concatenate(segments, axis=1)
    assert np.isfinite(traces).all()
    return traces, {
        "edf_path": source["source_paths"]["edf"],
        "packed_times_path": str(packed_path),
        "packed_times_sha256": sha256(packed_path),
        "event_windows_match_source_artifact": True,
        "sample_rate_in_hz": fs_in, "sample_rate_out_hz": fs_out,
        "edf_sample_bounds_stop_exclusive": sample_bounds,
        "edf_crop_bounds_sec": (np.array(sample_bounds) / fs_in).tolist(),
        "raw_selected_samples_sha256": raw_hashes,
        "reference_mode": source["selection"]["reference_mode"],
        "preprocessing": source["selection"]["reference_mode"] + "; resample_poly; 50:50:250 Hz IIR notch Q30; third-order Butterworth 80-250 Hz filtfilt, same filtering as Fig1B2",
        "edge_policy": "preserve the existing B2 excerpt-local filtering including its edge transients; no new event detection or temporal inference",
        "shape": list(traces.shape), "unit": "V",
        "filtered_samples_sha256": hashlib.sha256(traces.tobytes()).hexdigest(),
        "channels": channels, "event_indices": source["selection"]["event_indices"],
        "excerpt_duration_sec": duration,
        "display_time": "original B2 Time (s) axis; three concatenated 0.32 s excerpts with original dashed separators",
    }


def load_geometry(subject: str = "chengshuai") -> tuple[dict, list, dict]:
    coord_path = YUQUAN_ELEC_ROOT / subject / "chnXyzDict.npy"
    recon = YUQUAN_ELEC_ROOT.parent / "recons" / subject
    raw = np.load(coord_path, allow_pickle=True).item()
    names = [f"{shaft}{i + 1}" for shaft in sorted(raw) for i in range(len(raw[shaft]))]
    resolved = load_subject_coords("yuquan", subject, names)
    assert resolved.mapped_mask_in_requested_order.all()
    assert resolved.coord_units == "mm"
    coords = np.asarray(resolved.coords_array_in_requested_order)
    assert np.isfinite(coords).all() and len(coords) == sum(map(len, raw.values()))
    by_name = dict(zip(names, coords))
    # Pial is surface/tkregister RAS; contacts use scanner RAS, as established
    # by legacy region lookup via inv(orig.affine). Use the full affine mapping.
    orig = nib.load(recon / "mri/orig.mgz")
    tkr_to_scanner = orig.affine @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    meshes, sources = [], []
    for hemi in ("lh", "rh"):
        path = recon / f"surf/{hemi}.pial"
        vertices, faces = nib.freesurfer.read_geometry(str(path))
        vertices = nib.affines.apply_affine(tkr_to_scanner, vertices)
        cells = np.column_stack([np.full(len(faces), 3), faces])
        mesh = pv.PolyData(vertices, cells.ravel())
        meshes.append(mesh)
        sources.append({"path": str(path), "sha256": sha256(path),
                        "vertices": len(vertices), "triangles": len(faces)})
    return by_name, meshes, {
        "contact_file": str(coord_path), "contact_sha256": sha256(coord_path),
        "mri_file": str(recon / "mri/orig.mgz"),
        "mri_sha256": sha256(recon / "mri/orig.mgz"),
        "surface_sources": sources,
        "contact_space": "subject-native scanner RAS, mm",
        "pial_original_space": "FreeSurfer surface/tkregister RAS, mm",
        "surface_to_contact_transform": tkr_to_scanner.tolist(),
        "n_contacts": len(names), "n_shafts": len(raw),
        "names": names, "coordinates_mm": coords.tolist(),
        "loader_version": resolved.schema_version,
    }


def render_brain(by_name: dict, meshes: list, output: Path, *,
                 shaft_name: str = "K", n_readouts: int = 12,
                 channel_colors: dict | None = None,
                 camera_focus: np.ndarray | None = None,
                 parallel_scale_mm: float = 99,
                 camera_offset: tuple[float, float, float] = (-420, 180, 155),
                 camera_roll_degrees: float = 38) -> dict:
    channel_colors = CHANNEL_COLORS if channel_colors is None else channel_colors
    pv.OFF_SCREEN = True
    width, height = 1600, 1600
    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    plotter.enable_depth_peeling(number_of_peels=8)
    for mesh in meshes:
        plotter.add_mesh(mesh, color="#bdc0c2", opacity=0.055, smooth_shading=True,
                         ambient=0.55, diffuse=0.45, specular=0.0)
    shafts = sorted({re.sub(r"\d+$", "", n) for n in by_name})
    for shaft in shafts:
        xyz = np.array([v for n, v in by_name.items() if re.sub(r"\d+$", "", n) == shaft])
        line = pv.lines_from_points(xyz).tube(radius=0.17 if shaft == shaft_name else 0.09)
        plotter.add_mesh(line, color="#90959a" if shaft == shaft_name else "#c0c2c4",
                         smooth_shading=True, ambient=0.7, diffuse=0.3, specular=0.0)
    normal = np.array(list(by_name.values()))
    glyphs = pv.PolyData(normal).glyph(geom=pv.Sphere(radius=0.85, theta_resolution=16,
                                                   phi_resolution=12), scale=False, orient=False)
    plotter.add_mesh(glyphs, color="#a0a4a8", smooth_shading=True,
                     ambient=0.7, diffuse=0.3, specular=0.0)
    midpoint_coords = {f"{shaft_name}{i}-{shaft_name}{i+1}":
                       (by_name[f"{shaft_name}{i}"]+by_name[f"{shaft_name}{i+1}"])/2
                       for i in range(1, n_readouts+1)}
    for name, color in channel_colors.items():
        plotter.add_mesh(pv.Sphere(radius=1.3, center=midpoint_coords[name]), color=color,
                         smooth_shading=True, ambient=0.7, diffuse=0.3, specular=0.0)
    focus = np.array([0.0, -3.0, 26.0]) if camera_focus is None else np.asarray(camera_focus)
    # View from left/anterior/superior: anatomical anterior (+Y) projects
    # toward the lower left, matching the author's original brain orientation.
    plotter.camera_position = [focus + camera_offset, focus, [0, 0, 1]]
    plotter.camera.Roll(camera_roll_degrees)
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = parallel_scale_mm
    plotter.render()
    screen = {}
    for name, point in by_name.items():
        plotter.renderer.SetWorldPoint(*point, 1)
        plotter.renderer.WorldToDisplay()
        x, y, _ = plotter.renderer.GetDisplayPoint()
        screen[name] = [float(x), float(height - y)]
    midpoint_screen = {}
    for name, point in midpoint_coords.items():
        plotter.renderer.SetWorldPoint(*point, 1)
        plotter.renderer.WorldToDisplay()
        x, y, _ = plotter.renderer.GetDisplayPoint()
        midpoint_screen[name] = [float(x), float(height-y)]
    orientation_pixels = []
    for point in (focus, focus + [0, 60, 0]):
        plotter.renderer.SetWorldPoint(*point, 1)
        plotter.renderer.WorldToDisplay()
        x, y, _ = plotter.renderer.GetDisplayPoint()
        orientation_pixels.append([float(x), float(height-y)])
    anterior_delta = np.diff(orientation_pixels, axis=0)[0]
    assert anterior_delta[0] < 0 and anterior_delta[1] > 0
    plotter.screenshot(str(output))
    camera = [list(v) for v in plotter.camera_position]
    plotter.close()
    return {"image_size": [width, height], "contact_pixels": screen,
            "channel_midpoint_colors": channel_colors,
            "channel_midpoint_pixels": midpoint_screen,
            "channel_midpoint_coordinates_mm": {k: v.tolist() for k,v in midpoint_coords.items()},
            "camera_position_focal_viewup": camera,
            "anterior_screen_delta_right_down_pixels": anterior_delta.tolist(),
            "anterior_projects_lower_left": True,
            "additional_camera_roll_degrees": camera_roll_degrees,
            "camera_offset_mm": list(camera_offset),
            "parallel_scale_mm": parallel_scale_mm, "surface_opacity": 0.055,
            "contact_radius_mm_display_only": 0.85,
            "selected_contact_radius_mm_display_only": 1.3}


def draw_candidate(output: Path, source: dict, traces: np.ndarray, by_name: dict,
                   projection: dict) -> dict:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    figures = output / "figures"
    compact_in = float(source["plot"].get("brain_electrode_gap_reduction_mm", 0))/25.4
    canvas_width = 13.2-compact_in

    def rect(bounds, shift=False):
        x, y, width, height = bounds
        return [(13.2*x-(compact_in if shift else 0))/canvas_width,
                y, 13.2*width/canvas_width, height]

    def right_point(x, y):
        return [(13.2*x-compact_in)/canvas_width, y]

    channel_colors = source["plot"].get("channel_colors", CHANNEL_COLORS)
    fig = plt.figure(figsize=(canvas_width, 6.15), facecolor="white")
    brain = fig.add_axes(rect([0.0, 0.075, 0.51, 0.86]))
    brain.imshow(plt.imread(output / "source" / source["plot"].get("brain_image", "y3_brain.png")))
    brain.set_xlim(100, 1510)
    brain.set_ylim(1490, 210)
    brain.axis("off")
    channels = source["selection"]["selected_channels"]
    labels = source["selection"]["display_labels"]
    shaft_names = {re.sub(r"\d+$", "", n) for c in channels for n in c.split("-")}
    assert len(shaft_names) == 1
    shaft_name = next(iter(shaft_names))
    bipolar = source["selection"]["reference_mode"] == "adjacent_bipolar"
    n_contacts = len(channels) + int(bipolar)
    closeup_names = [f"{shaft_name}{i}" for i in range(1, n_contacts+1)]

    # The enclosure covers exactly the physical contacts used by this montage.
    xy = np.array([projection["contact_pixels"][n] for n in closeup_names])
    center = xy.mean(0)
    _, _, vh = np.linalg.svd(xy-center, full_matrices=False)
    unit = vh[0]
    normal = np.array([-unit[1], unit[0]])
    half_long = float(np.max(np.abs((xy-center)@unit))) + 25
    half_short = float(np.max(np.abs((xy-center)@normal))) + 24
    ellipse = Ellipse(center, width=2*half_long, height=2*half_short,
                      angle=np.degrees(np.arctan2(unit[1], unit[0])),
                      facecolor="none", edgecolor="#acafb2", linewidth=0.8)
    brain.add_patch(ellipse)
    highlighted = np.array([projection["channel_midpoint_pixels"][n] for n in channel_colors])
    brain.scatter(*highlighted.T, s=23, c=list(channel_colors.values()),
                  edgecolors="white", linewidths=0.35, zorder=4)

    points = np.array([by_name[n] for n in closeup_names])
    vertical = points[-1]-points[0]
    vertical /= np.linalg.norm(vertical)
    horizontal = np.cross(vertical, [0, 0, 1])
    horizontal /= np.linalg.norm(horizontal)
    local = np.column_stack([(points-points[0])@horizontal, (points-points[0])@vertical])
    theta = np.deg2rad(-11)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    local = local@rotation.T
    tangent = np.gradient(local, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    readout = (local[:-1]+local[1:])/2 if bipolar else local.copy()
    shaft = fig.add_axes(rect([0.49, 0.14, 0.13, 0.725], shift=True))
    shaft.set_aspect("equal")
    shaft.set_xlim(-4.5, 13.5)
    shaft.set_ylim(-5, local[-1, 1]+5)
    shaft.axis("off")
    body = np.vstack([local[0]-2.1*tangent[0], local, local[-1]+1.8*tangent[-1]])
    fig.canvas.draw()
    unit_points = np.linalg.norm(shaft.transData.transform([1, 0])-
                                 shaft.transData.transform([0, 0]))*72/fig.dpi
    body_width, collar_width, collar_length = 1.7, 1.95, 1.2
    shaft.plot(*body.T, color="#767b80", lw=body_width*unit_points+1.2,
               solid_capstyle="round", solid_joinstyle="round", zorder=3)
    shaft.plot(*body.T, color="#fafbfb", lw=body_width*unit_points,
               solid_capstyle="round", solid_joinstyle="round", zorder=4)
    marker_tangent = np.gradient(readout, axis=0)
    marker_tangent /= np.linalg.norm(marker_tangent, axis=1)[:, None]
    for i, (point, direction) in enumerate(zip(readout, marker_tangent)):
        perpendicular = np.array([direction[1], -direction[0]])
        transform = Affine2D(np.array([[perpendicular[0], direction[0], point[0]],
                                      [perpendicular[1], direction[1], point[1]],
                                      [0, 0, 1]])) + shaft.transData
        collar = FancyBboxPatch((-collar_width/2, -collar_length/2),
                                collar_width, collar_length,
                                boxstyle="round,pad=0,rounding_size=0.16",
                                transform=transform,
                                facecolor=channel_colors.get(channels[i], "#e4e7e9"),
                                edgecolor="#767b80", linewidth=0.6, zorder=5)
        shaft.add_patch(collar)

    # Keep the implanted lower-left end as the original simple rounded body.
    # The silver connector belongs at the external, upper-right end. Its
    # silhouette is illustrative, not a measured patient connector model.
    perpendicular = np.array([tangent[-1, 1], -tangent[-1, 0]])
    cap_transform = Affine2D(np.array([
        [perpendicular[0], tangent[-1, 0], local[-1, 0]],
        [perpendicular[1], tangent[-1, 1], local[-1, 1]], [0, 0, 1]
    ])) + shaft.transData
    cap_width, cap_base, cap_length = 2.3, 0.35, 2.7
    connector_cap = FancyBboxPatch((-cap_width/2, cap_base), cap_width, cap_length,
                                  boxstyle="round,pad=0,rounding_size=0.42",
                                  transform=cap_transform, facecolor="#bcc3c8",
                                  edgecolor="#727a80", linewidth=0.7, zorder=6)
    shaft.add_patch(connector_cap)
    shaft.plot([-0.56, -0.56], [cap_base+0.5, cap_base+cap_length-0.5],
               transform=cap_transform, color="#edf0f2", lw=0.85,
               solid_capstyle="round", zorder=7)

    # A continuous external lead exits the cap along the rod's tangent, then
    # drifts right above the label column. No arrow points at an individual row.
    to_figure = fig.transFigure.inverted()
    cable_start = to_figure.transform(cap_transform.transform([0, cap_base+cap_length]))
    cable_tangent = to_figure.transform(cap_transform.transform([0, cap_base+cap_length+3.8]))
    cable_vertices = np.array([
        cable_start, cable_tangent, right_point(0.610, 0.945), right_point(0.649, 0.913),
        right_point(0.680, 0.888), right_point(0.705, 0.886), right_point(0.738, 0.911),
    ])
    cable_path = MplPath(cable_vertices, [MplPath.MOVETO]+[MplPath.CURVE4]*6)
    cable = PathPatch(cable_path, transform=fig.transFigure, facecolor="none",
                      edgecolor="#737b82", linewidth=1.15, capstyle="round",
                      joinstyle="round", clip_on=False, zorder=5)
    fig.add_artist(cable)

    trace = fig.add_axes(rect([0.676, 0.14, 0.304, 0.725], shift=True))
    fs = source["plot"]["fs_out"]
    dur = source["plot"]["window_sec"]
    gap = float(source["plot"]["trace_spacing_V"])
    x = np.arange(traces.shape[1])/fs
    for row in range(len(channels)):
        trace.plot(x, traces[row]+row*gap, color="black", lw=0.42)
    for boundary in (dur, 2*dur):
        trace.axvline(boundary, color="#b9b9b9", ls="--", lw=0.55, alpha=0.9)
    trace.set_xlim(0, traces.shape[1]/fs)
    trace.set_ylim((len(channels)-0.2)*gap, -0.8*gap)
    trace.set_yticks(np.arange(len(channels))*gap)
    trace.set_yticklabels(labels, fontsize=12)
    if source["plot"].get("color_highlighted_labels", False):
        for label_artist, name in zip(trace.get_yticklabels(), channels):
            label_artist.set_color(channel_colors.get(name, "black"))
    trace.tick_params(axis="y", length=0, pad=23)
    tick_step = float(source["plot"].get("time_tick_step_sec", 0.2))
    trace.set_xticks(np.arange(0, traces.shape[1]/fs+1e-9, tick_step))
    trace.set_xlabel("Time (s)", fontsize=13, labelpad=6)
    trace.tick_params(axis="x", labelsize=11.5, length=3, pad=2.5)
    for key in ("top", "right"):
        trace.spines[key].set_visible(False)
    for key in ("left", "bottom"):
        trace.spines[key].set_linewidth(0.8)
    guide_x = -0.065
    trace.plot([guide_x, guide_x], trace.get_ylim(),
               transform=trace.get_yaxis_transform(), color="#b4b7ba", lw=0.65, ls=(0, (2, 2)), clip_on=False)
    marker_size = float(source["plot"].get("highlight_marker_size_pt2", 25))
    for row, name in enumerate(channels):
        trace.scatter([guide_x], [row*gap], transform=trace.get_yaxis_transform(),
                      s=marker_size if name in channel_colors else 13,
                      color=channel_colors.get(name, "#a7abae"), clip_on=False, zorder=6)
    trace.set_title("80-250Hz", fontsize=14, pad=7)
    fig.text(*right_point(0.828, 0.947), source["selection"]["display_label"],
             ha="center", fontsize=17, weight="bold")
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    center_fig = to_fig.transform(brain.transData.transform(center))
    axes_matrix = np.column_stack([
        to_fig.transform(brain.transData.transform(center+unit*half_long))-center_fig,
        to_fig.transform(brain.transData.transform(center+normal*half_short))-center_fig])

    def tangent_points(target):
        v = np.linalg.solve(axes_matrix, target-center_fig)
        norm2 = float(v@v)
        assert norm2 > 1
        perp = np.array([-v[1], v[0]])
        return [center_fig + axes_matrix@(v/norm2 + sign*perp*np.sqrt(norm2-1)/norm2)
                for sign in (-1, 1)]

    # The distal guide can be attached beside the true deepest contact instead
    # of using two tangents that both happen to originate near the shallow end.
    anchor_mode = source["plot"].get("zoom_anchor_mode", "ellipse_tangents")
    leader_segments = []
    leader_contacts = []
    for idx, choose in ((-1, max), (0, min)):
        endpoint = local[idx] + [-collar_width/2-0.6, 0]
        target = to_fig.transform(shaft.transData.transform(endpoint))
        if anchor_mode == "physical_ends" and idx == 0:
            contact_fig = to_fig.transform(brain.transData.transform(xy[idx]))
            radial = np.linalg.solve(axes_matrix, contact_fig-center_fig)
            anchor = center_fig + axes_matrix@(radial/np.linalg.norm(radial))
            anchor_pixel = brain.transData.inverted().transform(fig.transFigure.transform(anchor))
            assert int(np.argmin(np.linalg.norm(xy-anchor_pixel, axis=1))) == idx
            anchor_rule = "ellipse boundary beside deepest physical contact"
        else:
            anchor = choose(tangent_points(target), key=lambda p: p[1])
            anchor_rule = "ellipse tangent"
        line = np.linspace(anchor, target, 101)
        normalized = np.linalg.solve(axes_matrix, (line-center_fig).T).T
        assert np.min(np.sum(normalized**2, axis=1)) >= 1-1e-9
        leader_segments.append([anchor.tolist(), target.tolist()])
        leader_contacts.append({"contact": closeup_names[idx], "rule": anchor_rule,
                                "brain_contact_pixel": xy[idx].tolist()})
        fig.add_artist(plt.Line2D(*np.array([anchor, target]).T, transform=fig.transFigure,
                                  color="#b4b7ba", lw=0.65, zorder=1.5))
    def side(a, b, p):
        v, w = np.asarray(b)-a, np.asarray(p)-a
        return v[0]*w[1]-v[1]*w[0]
    (a,b), (c,d) = leader_segments
    crossed = bool(side(a,b,c)*side(a,b,d)<0 and side(c,d,a)*side(c,d,b)<0)
    assert not crossed

    # Reversing the display orientation of the rod is intentional: K1 is at
    # the lower-left rounded tip. Color identity carries correspondence without
    # crossing wires over the labels or between differently ordered columns.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gray_line_x = trace.get_yaxis_transform().transform([guide_x, 0])[0]
    labels_before_line = all(t.get_window_extent(renderer).x1 < gray_line_x
                             for t in trace.get_yticklabels())
    assert labels_before_line
    tip_fig = to_fig.transform(shaft.transData.transform(body[0]))
    proximal_fig = to_fig.transform(shaft.transData.transform(body[-1]))
    assert tip_fig[0] < proximal_fig[0] and tip_fig[1] < proximal_fig[1]
    margin = shaft.get_position()
    assert margin.y0 < tip_fig[1] < proximal_fig[1] < margin.y1
    cap_bbox = connector_cap.get_window_extent(renderer)
    assert fig.bbox.contains(cap_bbox.x0, cap_bbox.y0)
    assert fig.bbox.contains(cap_bbox.x1, cap_bbox.y1)
    cable_bbox = cable.get_window_extent(renderer)
    assert fig.bbox.contains(cable_bbox.x0, cable_bbox.y0)
    assert fig.bbox.contains(cable_bbox.x1, cable_bbox.y1)
    assert not cable_bbox.overlaps(trace.title.get_window_extent(renderer))
    assert all(not cable_bbox.overlaps(t.get_window_extent(renderer))
               for t in trace.get_yticklabels())
    display_marker_mapping = [{"label": labels[i], "channel": name,
                               "source_contacts": name.split("-"),
                               "marker_kind": "bipolar_pair_midpoint",
                               "brain_pixel": projection["channel_midpoint_pixels"][name],
                               "closeup_xy_mm": readout[i].tolist(),
                               "color": channel_colors.get(name, "#a7abae")}
                              for i, name in enumerate(channels)]
    for ext in ("png", "pdf", "svg"):
        fig.savefig(figures/f"fig1-panela.{ext}", dpi=350, facecolor="white")
    plt.close(fig)
    return {"selected_shaft": shaft_name, "source_contact_names": closeup_names,
            "display_marker_names": channels, "display_marker_mapping": display_marker_mapping,
            "one_marker_per_channel": True,
            "n_source_physical_contacts": len(closeup_names),
            "n_display_channel_markers": len(channels),
            "linked_channels": channel_colors,
            "brain_electrode_gap_reduction_mm": compact_in*25.4,
            "canvas_size_inches": [canvas_width, 6.15],
            "closeup_coordinates_mm": local.tolist(),
            "readout_locations_mm": readout.tolist(),
            "closeup_geometry": f"channel-level schematic at measured bipolar-pair midpoints; rounded {shaft_name}1 end at lower left; silver external connector cap at upper right with continuous curved lead",
            "trace_spacing_V": gap, "amplitude_scale": source["plot"].get("amplitude_scale", "frozen original B2 shared gain"),
            "highlight_marker_size_pt2": marker_size,
            "highlighted_labels_colored": bool(source["plot"].get("color_highlighted_labels", False)),
            "display_window_sec": dur, "time_tick_step_sec": tick_step,
            "trace_labels": labels, "trace_xlabel": "Time (s)",
            "trace_title": "80-250Hz", "extra_electrode_or_recording_titles": False,
            "brain_anterior_lower_left": projection["anterior_projects_lower_left"],
            "labels_before_gray_dashed_line": bool(labels_before_line),
            "gray_marker_line_style": "dashed",
            "tip_points_lower_left": True, "distal_tip_style": "simple rounded body",
            "complete_proximal_cap": True, "metal_connector_at_proximal_end": True,
            "external_lead": {"kind": "illustrative external cable, no arrow",
                              "path_figure_coordinates": cable_vertices.tolist(),
                              "continuous_with_cap": True,
                              "clear_of_channel_labels_and_title": True,
                              "reference_url": "https://adtechmedical.com/connection-systems",
                              "reference_scope": "electrode-to-cable connection only; cap geometry is schematic"},
            "magnification_leaders": leader_segments,
            "magnification_anchor_mode": anchor_mode,
            "magnification_contact_correspondence": leader_contacts,
            "leaders_cross": crossed, "leaders_outside_ellipse": True,
            "readout_links": [],
            "generic_acquisition_arrow": False,
            "claim_boundary": "one symbol per bipolar channel at its measured pair midpoint, not a one-contact monopolar claim; node widths, external connector cap and cable path are schematic"}


def compose_candidate(output: Path) -> dict:
    figures = output / "figures"
    before = {p.name: sha256(p) for p in (CANONICAL / "figures").glob("fig1-panel*.*")}
    files = _compose_complete_layout(
        figures_dir=figures, stem="fig1-complete-layout", canvas_size=(6000, 4800),
        placements={
            "a": (figures / "fig1-panela.png", (120, 190, 3190, 1500)),
            "b1": (CANONICAL / "figures/fig1-panelb1.png", (3360, 190, 3910, 1500)),
            "b2": (CANONICAL / "figures/fig1-panelb2.png", (4030, 190, 5850, 1500)),
            "c": (CANONICAL / "figures/fig1-panelc.png", (180, 1710, 4230, 3060)),
            "d": (CANONICAL / "figures/fig1-paneld.png", (4450, 1710, 5840, 3060)),
            "e": (CANONICAL / "figures/fig1-panele.png", (180, 3290, 4230, 4640)),
            "f": (CANONICAL / "figures/fig1-panelf.png", (4450, 3290, 5840, 4640)),
        },
        labels={"A": (45, 35), "B": (3230, 35), "C": (45, 1560),
                "D": (4310, 1560), "E": (45, 3140), "F": (4310, 3140)},
        anchors={"a": "top", "c": "top", "d": "top", "e": "top", "f": "top"},
    )
    after = {p.name: sha256(p) for p in (CANONICAL / "figures").glob("fig1-panel*.*")}
    assert before == after
    return {"files": files, "canonical_panel_hashes": before,
            "canonical_panels_unchanged": True,
            "layout_change": "only top-row A/B placement; C-F placements and source images unchanged"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-brain", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    for directory in (output / "figures", output / "source"):
        directory.mkdir(parents=True, exist_ok=True)
    source_path = CANONICAL / "figures/fig1-panelb2_metadata.json"
    source = json.loads(source_path.read_text())
    assert source["selection"]["display_label"] == "Yuquan Y3"
    source["selection"]["selected_channels"] = [f"K{i}-K{i+1}" for i in range(1, 13)]
    source["selection"]["display_labels"] = [f"K{i}" for i in range(1, 13)]
    source["selection"]["reference_mode"] = "adjacent_bipolar"
    original_trace_data = np.load(
        CANONICAL / "candidates/recording_chain_20261009/source/recording_and_geometry.npz"
    )
    source["plot"]["trace_spacing_V"] = float(np.nanstd(original_trace_data["signals_V"])*8.0)
    traces, signal_meta = load_recordings(source)
    by_name, meshes, geometry_meta = load_geometry()
    for channel in signal_meta["channels"]:
        assert all(n in by_name for n in channel.split("-"))
    projection_path = output / "source/brain_projection.json"
    if args.reuse_brain:
        projection = json.loads(projection_path.read_text())
    else:
        projection = render_brain(by_name, meshes, output / "source/y3_brain.png")
        projection_path.write_text(json.dumps(projection, indent=2) + "\n")
    display = draw_candidate(output, source, traces, by_name, projection)
    np.savez_compressed(output / "source/recording_and_geometry.npz", signals_V=traces,
                        channels=np.array(signal_meta["channels"]),
                        names=np.array(list(by_name)), coords_mm=np.array(list(by_name.values())))
    complete = compose_candidate(output)
    metadata = {"schema": "fig1a_data_recording_chain_connector_cap_v5", "display_label": "Yuquan Y3",
                "producer": str(Path(__file__).resolve()),
                "status": "CANDIDATE", "human_visual_acceptance": "PENDING",
                "formal_current_figure_replaced": False,
                "b2_metadata_source": str(source_path), "b2_metadata_sha256": sha256(source_path),
                "geometry": geometry_meta, "signals": signal_meta,
                "display": display, "complete_layout": complete}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    (output / "figures/README.md").write_text(
        "# Fig1A：同一患者的植入定位、K杆放大与多通道记录\n\n"
        "状态：候选，待作者目视检查；正式 Fig1 未覆盖。\n\n"
        "### fig1-panela.png / .pdf / .svg\n"
        "左侧为 Yuquan Y3 本人的双侧 FreeSurfer pial 脑表面和140个触点，表面经完整 affine 从 tkregister RAS 转到触点的 scanner RAS。"
        "保留已认可的脑图方向和中央电极向左下倾斜的姿态，下端恢复简洁圆头，上端外露尾部改成银灰色连接帽，并接一条向右舒缓弯曲的连续细导线。"
        "中央12个环状符号代表12个双极通道，符号中心由真实K1–K13触点的相邻中点计算；紫色K5–K6、蓝色K9–K10在脑图、示意和右侧各为一个同色标记。"
        "右侧保持K1至K12的原双极波形，按名称、灰色虚线与色点、波形的顺序排布。\n"
        "**关注点**：保留原EDF三个0.32秒片段、双极参考、滤波、幅值尺度和Time (s)。本图中央是通道层级的示意，并非12枚实物触点；K1简称K1–K2，依次至K12–K13。圈选区域的外切引导线不相交、不穿入椭圆；外接线不指向任何单通道且避开文字。杆宽、节点尺寸、连接帽和线缆路径均为示意，不宣称器械实测尺寸或患者器械型号；连接关系参考[Ad-Tech连接系统](https://adtechmedical.com/connection-systems)。\n\n"
        "### fig1-complete-layout.png / .pdf\n"
        "把本次 A 候选放入完整 Figure 1，B1/B2 整体向右移以容纳新的连续示意。"
        "B–F 直接读取正式源文件，C–F 的拼版位置不变，源文件哈希保持不变。\n"
        "**关注点**：本图用于检查整页比例和字体；A独立图及整图均待作者目视验收。\n",
        encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
