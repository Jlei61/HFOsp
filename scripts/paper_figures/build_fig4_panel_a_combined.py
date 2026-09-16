#!/usr/bin/env python3
"""Build the accepted combined Figure 4 panel A.

The left local-circuit material is copied from the archived former panel A.
The right patient-substrate material is rendered from the same frozen figdata
as the archived former panel B, but its anisotropic E-to-E corridor and
possible-core display layers are omitted.  No simulation or scientific value
is recomputed.  The dashed callout avoids implying that the conceptual local
circuit is a coordinate-exact crop of the patient substrate.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIGURES = ROOT / "results/paper-ready-figure/fig4/figures"
ARCHIVE_FIGURES = (
    ROOT
    / "results/paper-ready-figure/archive/2026-09-03_pre_combined_fig4_ab"
    / "fig4/figures"
)
PANEL_A = ARCHIVE_FIGURES / "fig4-panela.png"
PANEL_B = ARCHIVE_FIGURES / "fig4-panelb.png"
OUTPUT_PNG = FIGURES / "fig4-panela.png"
OUTPUT_PDF = FIGURES / "fig4-panela.pdf"
OUTPUT_METADATA = FIGURES / "fig4-panela-metadata.json"

CANVAS_SIZE = (6000, 3000)
A_CROP = (187, 0, 2773, 2218)
A_BOX = (120, 470, 2660, 2540)
B_BOX = (3260, 90, 5840, 2670)

CALLOUT_COLOR = "#4A4A4A"
TITLE_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
REGULAR_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
SIMULATION_XY_LIM_MM = (-10.0, 10.0)
EE_BASE_LENGTH_MM = 0.380
EE_ASPECT_RATIO = 2.0
EE_LONG_LENGTH_MM = EE_BASE_LENGTH_MM * np.sqrt(EE_ASPECT_RATIO)
# The outer ellipse in A is assigned to the e^-1 contour of the frozen
# elliptical-exponential E->E kernel: full long-axis diameter = 2*l_par.
EE_OUTER_ELLIPSE_DIAMETER_MM = 2.0 * EE_LONG_LENGTH_MM
LEFT_SCALE_BAR_MM = 0.5
ZOOM_CENTER_MM = (-4.502, 6.197)  # SCL9 in the frozen registered plane
SEEG_KERNEL_SIGMA_MM = 0.25
SEEG_KERNEL_R95_MM = SEEG_KERNEL_SIGMA_MM * np.sqrt(-2.0 * np.log(0.05))
SEEG_FOOTPRINT_COLOR = "#245C3F"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[Image.Image, tuple[int, int]]:
    x0, y0, x1, y1 = box
    scale = min((x1 - x0) / image.width, (y1 - y0) / image.height)
    size = (round(image.width * scale), round(image.height * scale))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    pos = (x0 + ((x1 - x0) - size[0]) // 2, y0 + ((y1 - y0) - size[1]) // 2)
    return resized, pos


def _dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: str,
    width: int,
    dash: int = 34,
    gap: int = 24,
) -> None:
    import math

    x0, y0 = start
    x1, y1 = end
    distance = math.hypot(x1 - x0, y1 - y0)
    if distance == 0:
        return
    ux, uy = (x1 - x0) / distance, (y1 - y0) / distance
    cursor = 0.0
    while cursor < distance:
        stop = min(cursor + dash, distance)
        draw.line(
            (
                round(x0 + ux * cursor),
                round(y0 + uy * cursor),
                round(x0 + ux * stop),
                round(y0 + uy * stop),
            ),
            fill=fill,
            width=width,
        )
        cursor += dash + gap


def _dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    width: int,
) -> None:
    x0, y0, x1, y1 = box
    _dashed_line(draw, (x0, y0), (x1, y0), fill=fill, width=width)
    _dashed_line(draw, (x1, y0), (x1, y1), fill=fill, width=width)
    _dashed_line(draw, (x1, y1), (x0, y1), fill=fill, width=width)
    _dashed_line(draw, (x0, y1), (x0, y0), fill=fill, width=width)


def _draw_xy_frame(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
) -> None:
    """Add the frozen -10..10 mm simulation coordinates around panel B."""
    x0, y0, x1, y1 = box
    frame_width = 9
    draw.rectangle(box, outline="black", width=frame_width)
    tick_font = ImageFont.truetype(str(REGULAR_FONT), 82)
    label_font = ImageFont.truetype(str(REGULAR_FONT), 98)
    tick_len = 28
    tick_values = (-10, 0, 10)

    for value in tick_values:
        frac = (value - SIMULATION_XY_LIM_MM[0]) / (
            SIMULATION_XY_LIM_MM[1] - SIMULATION_XY_LIM_MM[0]
        )
        x = round(x0 + frac * (x1 - x0))
        draw.line((x, y1, x, y1 + tick_len), fill="black", width=frame_width)
        draw.text((x, y1 + 66), str(value), font=tick_font, fill="black", anchor="mm")

        y = round(y1 - frac * (y1 - y0))
        draw.line((x0 - tick_len, y, x0, y), fill="black", width=frame_width)
        draw.text((x0 - 48, y), str(value), font=tick_font, fill="black", anchor="rm")

    draw.text(
        ((x0 + x1) // 2, y1 + 170),
        "x (mm)",
        font=label_font,
        fill="black",
        anchor="mm",
    )
    y_label = Image.new("RGBA", (430, 150), (255, 255, 255, 238))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text(
        (215, 75), "y (mm)", font=label_font, fill="black", anchor="mm",
    )
    y_label = y_label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    canvas.paste(
        y_label,
        (x0 - 245, (y0 + y1 - y_label.height) // 2),
        y_label,
    )


def _left_scale_bar_length_px(
    placed_a_width: int,
    source_a_width: int,
) -> int:
    """Calibrate the bar to the frozen E-to-E kernel's e^-1 contour."""
    # The source A axes span 10 schematic units.  With the fixed 5.2 x 4.15 in
    # canvas and equal aspect, the 10-unit x span is 4.720625 in at 600 dpi.
    source_px_per_schematic_unit = 4.720625 * 600.0 / 10.0
    resize_scale = placed_a_width / source_a_width
    return round(
        LEFT_SCALE_BAR_MM
        / EE_OUTER_ELLIPSE_DIAMETER_MM
        * 6.0
        * source_px_per_schematic_unit
        * resize_scale
    )


def _draw_left_scale_bar(
    draw: ImageDraw.ImageDraw,
    bar_px: int,
    left_box: tuple[int, int, int, int],
) -> None:
    x1 = left_box[2] - 95
    x0 = x1 - bar_px
    y = left_box[3] - 105
    width = 13
    draw.line((x0, y, x1, y), fill="black", width=width)
    draw.line((x0, y - 17, x0, y + 17), fill="black", width=width)
    draw.line((x1, y - 17, x1, y + 17), fill="black", width=width)
    font = ImageFont.truetype(str(REGULAR_FONT), 82)
    draw.text(
        ((x0 + x1) // 2, y - 58),
        f"{LEFT_SCALE_BAR_MM:g} mm",
        font=font,
        fill="black",
        anchor="mm",
    )


def _data_box_to_canvas(
    data_box: tuple[float, float, float, float],
    placed_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Map an x/y box in the frozen -10..10 mm plane to panel-B pixels."""
    x0, y0, x1, y1 = data_box
    px0, py0, px1, py1 = placed_box
    lo, hi = SIMULATION_XY_LIM_MM
    left = px0 + (x0 - lo) / (hi - lo) * (px1 - px0)
    right = px0 + (x1 - lo) / (hi - lo) * (px1 - px0)
    top = py0 + (hi - y1) / (hi - lo) * (py1 - py0)
    bottom = py0 + (hi - y0) / (hi - lo) * (py1 - py0)
    return tuple(round(v) for v in (left, top, right, bottom))


def _render_clean_panel_b() -> tuple[Image.Image, dict]:
    """Render the frozen patient substrate without redundant overlays."""
    from scripts.paper_figures.plot_fig4_subject_snn_grouped import DEFAULT_TAG
    from scripts.paper_figures.plot_fig_subject_snn import _registered_axis_display
    from scripts.paper_figures.plot_fig_subject_snn_mechanism import (
        _load_figdata,
        _plot_mechanism,
        _reconstruct_posI,
    )

    fd, source_path = _load_figdata(DEFAULT_TAG)
    updated = dict(fd)
    reg = dict(fd["reg"].item())
    reg["source_names"] = []
    reg["sink_names"] = []
    updated["reg"] = np.asarray(reg, dtype=object)
    pos_i, pos_i_meta = _reconstruct_posI(fd, DEFAULT_TAG)
    plot_seed = int((pos_i_meta.get("seed") or 0) + 101)

    fig = plt.figure(figsize=(5.0, 5.0), facecolor="white")
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    display = _registered_axis_display(fd)
    _plot_mechanism(
        updated,
        ax,
        clean=True,
        posI=pos_i,
        plot_seed=plot_seed,
        display=display,
        homogeneous_cores=True,
        semantic_core_colors=False,
        show_basic_labels=True,
        show_title=False,
    )
    for patch in list(ax.patches):
        if isinstance(patch, (Circle, Ellipse)):
            patch.remove()
    for line in list(ax.lines):
        if float(line.get_zorder()) == 8.0:
            line.remove()
    for text in list(ax.texts):
        if text.get_text().startswith("Core ") or text.get_text() == "anisotropic E→E":
            text.remove()
        else:
            text.set_fontsize(max(13.0, 1.5 * text.get_fontsize()))
    contacts_display = (
        np.asarray(fd["contacts"], float) @ np.asarray(display["matrix"], float)
        + np.asarray(display["offset"], float)
    )
    contact_span_mm = np.ptp(contacts_display, axis=0)
    for xy in contacts_display:
        ax.add_patch(
            Circle(
                xy,
                SEEG_KERNEL_R95_MM,
                facecolor=SEEG_FOOTPRINT_COLOR,
                edgecolor=SEEG_FOOTPRINT_COLOR,
                linewidth=0.65,
                alpha=0.12,
                zorder=0.5,
            )
        )
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(12.5)

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=600,
        facecolor="white",
    )
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        clean_panel = rendered.convert("RGB")
    return clean_panel, {
        "figdata_path": str(source_path.relative_to(ROOT)),
        "figdata_sha256": _sha256(source_path),
        "removed_display_layers": [
            "anisotropic E-to-E corridor and label",
            "possible data driven core contours and label",
            "historical discrete core outlines, labels, and connecting line",
        ],
        "preserved_information": [
            "frozen E-neuron positions",
            "deterministically reconstructed I-neuron positions",
            "patient contact geometry",
            "E/I neuron legend",
        ],
        "seeg_sampling_overlay": {
                "readout": "virtual-contact firing-density envelope used by Figure 4F",
            "kernel": "normalized isotropic Gaussian without hard cutoff",
            "sigma_mm": SEEG_KERNEL_SIGMA_MM,
            "displayed_radius": "theoretical 95% mass radius",
            "displayed_radius_mm": SEEG_KERNEL_R95_MM,
            "contact_span_xy_mm": contact_span_mm.tolist(),
            "color": SEEG_FOOTPRINT_COLOR,
            "alpha": 0.12,
            "not_the_current_lfp_cutoff": "LFPRecorder Rr=0.278 mm is a separate path",
        },
    }


def compose() -> dict:
    with Image.open(PANEL_A) as source_a:
        panel_a = source_a.convert("RGB").crop(A_CROP)
    # Remove only the embedded title band; all circuit artwork begins below it.
    ImageDraw.Draw(panel_a).rectangle((0, 0, panel_a.width, 250), fill="white")
    panel_b, clean_b_details = _render_clean_panel_b()

    canvas = Image.new("RGB", CANVAS_SIZE, "white")
    placed_a, pos_a = _fit(panel_a, A_BOX)
    placed_b, pos_b = _fit(panel_b, B_BOX)
    canvas.paste(placed_a, pos_a)
    canvas.paste(placed_b, pos_b)

    placed_a_box = (
        pos_a[0], pos_a[1], pos_a[0] + placed_a.width, pos_a[1] + placed_a.height,
    )
    placed_b_box = (
        pos_b[0], pos_b[1], pos_b[0] + placed_b.width, pos_b[1] + placed_b.height,
    )
    left_callout_box = (
        placed_a_box[0] - 30,
        placed_a_box[1] + 160,
        placed_a_box[2] + 30,
        placed_a_box[3] + 30,
    )
    left_scale_bar_px = _left_scale_bar_length_px(
        placed_a.width, panel_a.width,
    )
    left_fov_width_mm = (
        (left_callout_box[2] - left_callout_box[0])
        / left_scale_bar_px
        * LEFT_SCALE_BAR_MM
    )
    left_fov_height_mm = (
        (left_callout_box[3] - left_callout_box[1])
        / left_scale_bar_px
        * LEFT_SCALE_BAR_MM
    )
    zoom_data_box = (
        ZOOM_CENTER_MM[0] - 0.5 * left_fov_width_mm,
        ZOOM_CENTER_MM[1] - 0.5 * left_fov_height_mm,
        ZOOM_CENTER_MM[0] + 0.5 * left_fov_width_mm,
        ZOOM_CENTER_MM[1] + 0.5 * left_fov_height_mm,
    )
    callout_box = _data_box_to_canvas(zoom_data_box, placed_b_box)

    draw = ImageDraw.Draw(canvas)
    line_width = 10
    _dashed_rectangle(
        draw, left_callout_box, fill=CALLOUT_COLOR, width=line_width,
    )
    _dashed_rectangle(
        draw, callout_box, fill=CALLOUT_COLOR, width=line_width,
    )
    # Connect the two dashed frames corner-to-corner so the left panel reads as
    # one bounded inset instead of a floating annotation.
    _dashed_line(
        draw,
        (left_callout_box[2], left_callout_box[1]),
        (callout_box[0], callout_box[1]),
        fill=CALLOUT_COLOR,
        width=line_width,
    )
    _dashed_line(
        draw,
        (left_callout_box[2], left_callout_box[3]),
        (callout_box[0], callout_box[3]),
        fill=CALLOUT_COLOR,
        width=line_width,
    )
    title_font = ImageFont.truetype(str(TITLE_FONT), 112)
    draw.text(
        (
            (left_callout_box[0] + left_callout_box[2]) // 2,
            left_callout_box[1] - 105,
        ),
        "Local E/I circuit",
        font=title_font,
        fill="black",
        anchor="mm",
    )
    _draw_left_scale_bar(draw, left_scale_bar_px, left_callout_box)
    _draw_xy_frame(canvas, draw, placed_b_box)

    canvas.save(OUTPUT_PNG, dpi=(600, 600), optimize=True)
    fixed_time = time.struct_time((2026, 9, 3, 0, 0, 0, 3, 246, 0))
    canvas.save(
        OUTPUT_PDF,
        "PDF",
        resolution=600.0,
        title="Figure 4 panel A",
        creationDate=fixed_time,
        modDate=fixed_time,
    )

    metadata = {
        "status": "AUTHOR_ACCEPTED_CANONICAL",
        "producer": str(Path(__file__).resolve().relative_to(ROOT)),
        "composition_only": False,
        "simulation_rerun": False,
        "scientific_values_recomputed": False,
        "panel_a_redrawn": False,
        "right_substrate_rendering": "same frozen figdata with redundant display layers suppressed",
        "sources": {
            "local_circuit": {
                "path": str(PANEL_A.relative_to(ROOT)),
                "sha256": _sha256(PANEL_A),
            },
            "patient_substrate": {
                "path": str(PANEL_B.relative_to(ROOT)),
                "sha256": _sha256(PANEL_B),
                "role": "accepted visual lineage; clean right panel uses the same frozen figdata",
            },
            "clean_patient_substrate": clean_b_details,
        },
        "layout": {
            "canvas_px": list(CANVAS_SIZE),
            "local_circuit_role": "smaller left-side conceptual detail",
            "patient_substrate_role": "larger right-side main panel",
            "a_crop_px": list(A_CROP),
            "a_placed_box_px": list(placed_a_box),
            "a_dashed_inset_box_px": list(left_callout_box),
            "a_title": {
                "text": "Local E/I circuit",
                "position": "centered above dashed inset box",
                "font_px_at_600_dpi": 112,
            },
            "a_scale_bar": {
                "label_mm": LEFT_SCALE_BAR_MM,
                "length_px": left_scale_bar_px,
                "calibration": (
                    "outer A ellipse is assigned to the e^-1 contour of the "
                    "frozen elliptical-exponential E-to-E kernel; its full "
                    "long-axis diameter is 2*l_par"
                ),
                "l_EE_mm": EE_BASE_LENGTH_MM,
                "AR": EE_ASPECT_RATIO,
                "l_par_mm": EE_LONG_LENGTH_MM,
                "outer_ellipse_diameter_mm": EE_OUTER_ELLIPSE_DIAMETER_MM,
            },
            "b_placed_box_px": list(placed_b_box),
            "b_xy_frame": {
                "style": "solid",
                "xlim_mm": list(SIMULATION_XY_LIM_MM),
                "ylim_mm": list(SIMULATION_XY_LIM_MM),
                "ticks_mm": [-10, 0, 10],
                "labels": ["x (mm)", "y (mm)"],
            },
            "zoom_center": {"contact": "SCL9", "xy_mm": list(ZOOM_CENTER_MM)},
            "zoom_field_of_view_mm": [left_fov_width_mm, left_fov_height_mm],
            "zoom_data_box_mm": list(zoom_data_box),
            "representative_callout_canvas_box_px": list(callout_box),
            "connector_style": "dashed",
        },
        "semantic_boundary": (
            "The callout indicates a representative local E/I neighbourhood. "
            "Panel A is a conceptual mechanism schematic, not a coordinate-exact "
            "crop of panel B and not a recovered anatomical core. The right panel "
            "retains only E/I positions and patient contact geometry."
        ),
        "outputs": [
            str(OUTPUT_PNG.relative_to(ROOT)),
            str(OUTPUT_PDF.relative_to(ROOT)),
        ],
    }
    OUTPUT_METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


if __name__ == "__main__":
    print(json.dumps(compose(), indent=2, ensure_ascii=False))
