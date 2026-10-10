#!/usr/bin/env python3
"""Build a data-derived Y3 implantation / K-shaft / recording Fig1A candidate.

The cortex and all contact centers are patient data. Connector lines and marker
diameters are display aids, not measured cable paths or manufacturer dimensions.
The existing Fig1B2 channel order, excerpts and preprocessing are preserved.
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
from matplotlib.patches import ConnectionPatch, Ellipse, FancyArrowPatch, PathPatch, Polygon
from matplotlib.path import Path as MplPath
from scipy.signal import butter, filtfilt, iirnotch, resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures.build_main_figures_1_2 import _compose_complete_layout
from src.seeg_coord_loader import YUQUAN_ELEC_ROOT, load_subject_coords

CANONICAL = ROOT / "results/paper-ready-figure/fig1"
DEFAULT_OUTPUT = CANONICAL / "candidates/recording_chain_compact_20261009"
RECON = YUQUAN_ELEC_ROOT.parent / "recons/chengshuai"
PURPLE = "#8b70ba"
BLUE = "#24a9ce"
CONTACT_COLORS = {"K5": PURPLE, "K6": PURPLE, "K9": BLUE, "K10": BLUE}
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
        bipolar = np.array([data[to_row[mapping[a]]] - data[to_row[mapping[b]]]
                            for a, b in pairs], dtype=np.float64)
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
        "preprocessing": "adjacent bipolar; resample_poly; 50:50:250 Hz IIR notch Q30; third-order Butterworth 80-250 Hz filtfilt, identical to Fig1B2",
        "edge_policy": "preserve the existing B2 excerpt-local filtering including its edge transients; no new event detection or temporal inference",
        "shape": list(traces.shape), "unit": "V",
        "filtered_samples_sha256": hashlib.sha256(traces.tobytes()).hexdigest(),
        "channels": channels, "event_indices": source["selection"]["event_indices"],
        "excerpt_duration_sec": duration,
        "display_time": "original B2 Time (s) axis; three concatenated 0.32 s excerpts with original dashed separators",
    }


def load_geometry() -> tuple[dict, list, dict]:
    coord_path = YUQUAN_ELEC_ROOT / "chengshuai/chnXyzDict.npy"
    raw = np.load(coord_path, allow_pickle=True).item()
    names = [f"{shaft}{i + 1}" for shaft in sorted(raw) for i in range(len(raw[shaft]))]
    resolved = load_subject_coords("yuquan", "chengshuai", names)
    assert resolved.mapped_mask_in_requested_order.all()
    assert resolved.coord_units == "mm"
    coords = np.asarray(resolved.coords_array_in_requested_order)
    assert np.isfinite(coords).all() and len(coords) == 140
    by_name = dict(zip(names, coords))
    # Pial is surface/tkregister RAS; contacts use scanner RAS, as established
    # by legacy region lookup via inv(orig.affine). Use the full affine mapping.
    orig = nib.load(RECON / "mri/orig.mgz")
    tkr_to_scanner = orig.affine @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    meshes, sources = [], []
    for hemi in ("lh", "rh"):
        path = RECON / f"surf/{hemi}.pial"
        vertices, faces = nib.freesurfer.read_geometry(str(path))
        vertices = nib.affines.apply_affine(tkr_to_scanner, vertices)
        cells = np.column_stack([np.full(len(faces), 3), faces])
        mesh = pv.PolyData(vertices, cells.ravel())
        meshes.append(mesh)
        sources.append({"path": str(path), "sha256": sha256(path),
                        "vertices": len(vertices), "triangles": len(faces)})
    return by_name, meshes, {
        "contact_file": str(coord_path), "contact_sha256": sha256(coord_path),
        "mri_file": str(RECON / "mri/orig.mgz"),
        "mri_sha256": sha256(RECON / "mri/orig.mgz"),
        "surface_sources": sources,
        "contact_space": "subject-native scanner RAS, mm",
        "pial_original_space": "FreeSurfer surface/tkregister RAS, mm",
        "surface_to_contact_transform": tkr_to_scanner.tolist(),
        "n_contacts": len(names), "n_shafts": len(raw),
        "names": names, "coordinates_mm": coords.tolist(),
        "loader_version": resolved.schema_version,
    }


def render_brain(by_name: dict, meshes: list, output: Path) -> dict:
    pv.OFF_SCREEN = True
    width, height = 1600, 1600
    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    plotter.enable_depth_peeling(number_of_peels=8)
    for mesh in meshes:
        plotter.add_mesh(mesh, color="#b7babd", opacity=0.075, smooth_shading=True,
                         ambient=0.35, diffuse=0.65, specular=0.12)
    shafts = sorted({re.sub(r"\d+$", "", n) for n in by_name})
    for shaft in shafts:
        xyz = np.array([v for n, v in by_name.items() if re.sub(r"\d+$", "", n) == shaft])
        line = pv.lines_from_points(xyz).tube(radius=0.23 if shaft == "K" else 0.15)
        plotter.add_mesh(line, color="#707070" if shaft == "K" else "#a6a6a6",
                         smooth_shading=True)
    normal = np.array([v for n, v in by_name.items() if n not in CONTACT_COLORS])
    glyphs = pv.PolyData(normal).glyph(geom=pv.Sphere(radius=1.04, theta_resolution=16,
                                                   phi_resolution=12), scale=False, orient=False)
    plotter.add_mesh(glyphs, color="#858585", smooth_shading=True, specular=0.25)
    for name, color in CONTACT_COLORS.items():
        plotter.add_mesh(pv.Sphere(radius=1.6, center=by_name[name]), color=color,
                         smooth_shading=True, specular=0.25)
    focus = np.array([0.0, -3.0, 26.0])
    # View from left/anterior/superior: anatomical anterior (+Y) projects
    # toward the lower left, matching the author's original brain orientation.
    plotter.camera_position = [focus + [-420, 180, 155], focus, [0, 0, 1]]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = 99
    plotter.render()
    screen = {}
    for name, point in by_name.items():
        plotter.renderer.SetWorldPoint(*point, 1)
        plotter.renderer.WorldToDisplay()
        x, y, _ = plotter.renderer.GetDisplayPoint()
        screen[name] = [float(x), float(height - y)]
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
            "camera_position_focal_viewup": camera,
            "anterior_screen_delta_right_down_pixels": anterior_delta.tolist(),
            "anterior_projects_lower_left": True,
            "parallel_scale_mm": 99, "surface_opacity": 0.075,
            "contact_radius_mm_display_only": 1.04,
            "selected_contact_radius_mm_display_only": 1.6}


def draw_candidate(output: Path, source: dict, traces: np.ndarray, by_name: dict,
                   projection: dict) -> dict:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    figures = output / "figures"
    fig = plt.figure(figsize=(13.2, 6.15), facecolor="white")
    brain = fig.add_axes([0.0, 0.075, 0.535, 0.88])
    brain.imshow(plt.imread(output / "source/y3_brain.png"))
    brain.set_xlim(55, 1545)
    brain.set_ylim(1490, 220)
    brain.axis("off")

    # One local magnification marker surrounds both actual bipolar pairs.
    xy = np.array([projection["contact_pixels"][n] for n in CONTACT_COLORS])
    center = xy.mean(0)
    delta = xy[-1] - xy[0]
    unit = delta / np.linalg.norm(delta)
    normal = np.array([-unit[1], unit[0]])
    half_long = np.linalg.norm(delta)/2 + 35
    half_short = 35
    ellipse = Ellipse(center, width=2*half_long, height=2*half_short,
                      angle=np.degrees(np.arctan2(delta[1], delta[0])),
                      facecolor="none", edgecolor="#a1a1a1", linewidth=0.85)
    brain.add_patch(ellipse)
    # Keep the selected contacts legible through both transparent hemispheres.
    # Centers are the renderer's exact 3-D-to-screen projections.
    brain.scatter(*xy.T, s=24, c=list(CONTACT_COLORS.values()),
                  edgecolors="white", linewidths=0.35, zorder=4)

    shaft = fig.add_axes([0.498, 0.14, 0.108, 0.75])
    points = np.array([by_name[f"K{i}"] for i in range(1, 17)])
    origin = points[0]
    vertical = points[-1] - origin
    vertical /= np.linalg.norm(vertical)
    horizontal = np.cross(vertical, [0.0, 0.0, 1.0])
    horizontal /= np.linalg.norm(horizontal)
    local = np.column_stack([(points-origin) @ horizontal, (points-origin) @ vertical])
    theta = np.deg2rad(-9)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    local = local @ rotation.T
    tangent = np.gradient(local, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    body = np.vstack([local[0]-1.5*tangent[0], local, local[-1]+1.5*tangent[-1]])
    shaft.set_aspect("equal")
    shaft.set_xlim(-4, 15)
    shaft.set_ylim(-4, 59)
    shaft.axis("off")
    fig.canvas.draw()
    # Rounded insulated body, outlined contact collars: appearance follows the
    # supplied schematic; all 16 collar centers retain measured K coordinates.
    body_width_mm = 1.65
    collar_width_mm = 2.5
    collar_length_mm = 1.3
    unit_points = np.linalg.norm(shaft.transData.transform([1, 0])-
                                 shaft.transData.transform([0, 0])) * 72/fig.dpi
    shaft.plot(*body.T, color="#252525", lw=body_width_mm*unit_points+2.0,
               solid_capstyle="round", solid_joinstyle="round", zorder=3)
    shaft.plot(*body.T, color="#fafafa", lw=body_width_mm*unit_points,
               solid_capstyle="round", solid_joinstyle="round", zorder=4)
    for i, (point, direction) in enumerate(zip(local, tangent)):
        perpendicular = np.array([direction[1], -direction[0]])
        corners = np.array([point + a*collar_width_mm/2*perpendicular +
                            b*collar_length_mm/2*direction
                            for a, b in ((-1,-1), (1,-1), (1,1), (-1,1))])
        patch = Polygon(corners, closed=True,
                        facecolor=CONTACT_COLORS.get(f"K{i+1}", "#fafafa"),
                        edgecolor="#252525", linewidth=1.05, joinstyle="round", zorder=5)
        shaft.add_patch(patch)
    # Guides attach to the boundaries of the local patch, not guessed pixels.
    for sign, idx in ((-1, 9), (1, 4)):
        anchor = center + sign*half_short*normal + 0.4*half_long*unit
        endpoint = local[idx] + np.array([-collar_width_mm/2-0.6, 0])
        fig.add_artist(ConnectionPatch(anchor, endpoint, "data", "data",
                       axesA=brain, axesB=shaft, color="#adadad", lw=0.7, zorder=1.5))

    # Restore the original B2 trace grammar: labels, polarity, common gain,
    # time extent, ticks and dashed excerpt borders are unchanged.
    trace = fig.add_axes([0.678, 0.13, 0.304, 0.715])
    fs = source["plot"]["fs_out"]
    dur = source["plot"]["window_sec"]
    channels = source["selection"]["selected_channels"]
    labels = source["selection"]["display_labels"]
    gap = np.nanstd(traces)*8.0
    x = np.arange(traces.shape[1])/fs
    for row in range(len(channels)):
        trace.plot(x, traces[row]+row*gap, color="black", lw=0.42)
    for boundary in (dur, 2*dur):
        trace.axvline(boundary, color="#b9b9b9", linestyle="--", linewidth=0.55, alpha=0.9)
    trace.set_xlim(0, traces.shape[1]/fs)
    trace.set_ylim(-0.8*gap, (len(channels)-0.2)*gap)
    trace.invert_yaxis()
    trace.set_yticks(np.arange(len(channels))*gap)
    trace.set_yticklabels(labels, fontsize=12)
    trace.tick_params(axis="y", length=0, pad=23)
    trace.set_xticks(np.arange(0, traces.shape[1]/fs+1e-9, 0.2))
    trace.set_xlabel("Time (s)", fontsize=13, labelpad=6)
    trace.tick_params(axis="x", labelsize=11.5, length=3, pad=2.5)
    trace.spines["top"].set_visible(False)
    trace.spines["right"].set_visible(False)
    trace.spines["left"].set_linewidth(0.8)
    trace.spines["bottom"].set_linewidth(0.8)
    guide_x = -0.065
    trace.plot([guide_x, guide_x], trace.get_ylim(),
               transform=trace.get_yaxis_transform(), color="#999999", lw=0.65, clip_on=False)
    for row, name in enumerate(channels):
        trace.scatter([guide_x], [row*gap], transform=trace.get_yaxis_transform(),
                      s=26 if name in CHANNEL_COLORS else 14,
                      color=CHANNEL_COLORS.get(name, "#919191"),
                      clip_on=False, zorder=6)
    trace.set_title("80-250Hz", fontsize=14, pad=7)
    fig.text(0.83, 0.927, "Yuquan Y3", ha="center", fontsize=17, weight="bold")

    fig.canvas.draw()
    lead_start = fig.transFigure.inverted().transform(shaft.transData.transform(body[-1]))
    elbow = np.array([0.584, 0.925])
    path = MplPath([lead_start, lead_start+[0.002, 0.035], elbow-[0.014, 0.0], elbow],
                   [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    fig.add_artist(PathPatch(path, transform=fig.transFigure, fill=False,
                            color="#252525", lw=1.7, capstyle="round"))
    wire = MplPath([elbow, [0.61, 0.937], [0.647, 0.935], [0.679, 0.909]],
                   [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    fig.add_artist(FancyArrowPatch(path=wire, transform=fig.transFigure,
                                  arrowstyle="-|>", mutation_scale=11,
                                  color="#979797", lw=0.9))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(figures / f"fig1-panela.{ext}", dpi=350, facecolor="white")
    plt.close(fig)
    return {"selected_shaft": "K", "selected_contacts": list(CONTACT_COLORS),
            "contact_colors": CONTACT_COLORS, "linked_bipolar_channels": CHANNEL_COLORS,
            "closeup_coordinates_mm": local.tolist(),
            "closeup_geometry": "all K1-K16 measured centers, rigid orthogonal projection; outlined shaft and contact collars are schematic glyphs",
            "three_dimensional_intercontact_distances_mm": np.linalg.norm(np.diff(points, axis=0), axis=1).tolist(),
            "trace_spacing_V": float(gap), "one_amplitude_gain_for_all_channels": True,
            "trace_labels": labels, "trace_xlabel": "Time (s)",
            "trace_title": "80-250Hz", "extra_electrode_or_recording_titles": False,
            "channel_marker_vertical_line": True,
            "brain_anterior_lower_left": projection["anterior_projects_lower_left"],
            "highlighted_brain_contact_overlay": "exact renderer screen coordinates, same contact colors as the close-up and trace markers",
            "diagram_only": ["magnification leaders", "acquisition lead", "contact glyph dimensions", "shaft display thickness and cap extensions"],
            "claim_boundary": "representative K electrode; E traces remain separate E-shaft pairs; simplified channel names use the original B2 left-contact aliases"}


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
    metadata = {"schema": "fig1a_data_recording_chain_compact_v2", "display_label": "Yuquan Y3",
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
        "脑前额方向朝左下，中央按参考图使用白色杆体、深色轮廓和环状触点，16个触点中心仍来自实测坐标。"
        "紫色K5/K6与蓝色K9/K10分别对应右侧原标签K5与K9，右侧恢复Fig1B2的通道简称、波形、虚线边界和Time (s)，补上灰色节点竖线。\n"
        "**关注点**：中间不加标题；杆体和触点环的宽度、长度、端帽及导线仅为显示示意，不宣称实测器械尺寸。右侧仍是三个0.32秒片段的拼接；通道简称沿用左触点名，完整双极对应见metadata，E杆信号不归到K杆。\n\n"
        "### fig1-complete-layout.png / .pdf\n"
        "把本次 A 候选放入完整 Figure 1，B1/B2 整体向右移以容纳新的连续示意。"
        "B–F 直接读取正式源文件，C–F 的拼版位置不变，源文件哈希保持不变。\n"
        "**关注点**：本图用于检查整页比例和字体；A独立图及整图均待作者目视验收。\n",
        encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
