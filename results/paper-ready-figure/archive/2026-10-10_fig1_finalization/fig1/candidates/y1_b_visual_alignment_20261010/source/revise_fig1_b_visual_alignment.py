#!/usr/bin/env python3
"""Align the current Figure 1 B/D/F axes without changing their data.

The accepted A and current 18-channel C/E are retained as original PDF objects
and exact PNG pixels. B uses the current full-window S**3 spectrum contract.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/tmp/fig1_pdf_deps")

from scripts.paper_figures import build_fig1_y1_local_rank as previous
from scripts.paper_figures import restore_fig1_legacy_spectrum as restored
from matplotlib.axes import _base
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox, ScaledTranslation
from PIL import Image
from pypdf import PdfReader, PdfWriter

s, plt, np = previous.s, previous.plt, previous.np
current, layout, TYPE = previous.current, previous.layout, previous.TYPE
BASE = s.CANON / "candidates/y1_b_compact_third_event_20261010"
OUT = s.CANON / "candidates/y1_b_visual_alignment_20261010"
FIG = OUT / "figures"
WIDTH, HEIGHT = previous.WIDTH, previous.HEIGHT
RIGHT_LEFT, RIGHT_RIGHT = 14.98, 18.92
RIGHT_WIDTH = RIGHT_RIGHT - RIGHT_LEFT
ERASE = [(10.65, 10.20, 13.94, 15.87), (13.94, .12, 19.46, 15.87)]


def merge_pdf(base, overlay, output):
    page = PdfReader(base).pages[0]
    page.merge_page(PdfReader(overlay).pages[0])
    writer = PdfWriter()
    writer.add_page(page)
    with output.open("wb") as handle:
        writer.write(handle)


def inch_box(fig, artist):
    return artist.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.dpi_scale_trans.inverted())


def main():
    (OUT / "source").mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    pointer = s.CANON / "current_revision.json"
    pointer_before = pointer.read_bytes()
    retained = {str(p): s.sha(p) for p in BASE.rglob("*") if p.is_file()}
    metadata = json.loads((BASE / "metadata.json").read_text())
    contract = json.loads((BASE / "spectrum_contract.json").read_text())
    choices = [(e["record"], e["event_index"]) for e in contract["display_events"]]
    events = restored.display_events(copy.deepcopy(contract), write_metadata=False,
                                     event_selection=choices)
    shutil.copy2(BASE / "spectrum_contract.json", OUT / "spectrum_contract.json")
    for suffix in ("png", "pdf"):
        name = f"fig1-spectrum-full-window-check.{suffix}"
        shutil.copy2(BASE / "figures" / name, FIG / name)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": TYPE.annotation,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.unicode_minus": False})
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    white_patches = []
    for x0, y0, x1, y1 in ERASE:
        patch = Rectangle((x0/WIDTH, y0/HEIGHT), (x1-x0)/WIDTH, (y1-y0)/HEIGHT,
                          transform=fig.transFigure, facecolor="white",
                          edgecolor="none", zorder=-10)
        fig.add_artist(patch)
        white_patches.append(patch)
    groups = {}
    old_out = layout.OUT
    try:
        layout.OUT = OUT
        groups["B_hfo"], hfo_meta = layout.draw_hfo(fig, 10.75)
    finally:
        layout.OUT = old_out
    for i, ax in enumerate(groups["B_hfo"]):
        ax.set_position(current.rect(fig, previous.MID_X, 14.0-i*1.53,
                                     previous.MID_WIDTH, 1.10))
        previous.style_axis(ax)
        ax.set_xticks([0, .5]); ax.set_xticklabels(["0.0", "0.5"])
        if i < 2:
            ax.tick_params(axis="x", labelbottom=False)
    hfo = groups["B_hfo"]
    hfo[0].set_title("")
    hfo[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    hfo[0].yaxis.get_offset_text().set_fontsize(TYPE.tick_label)
    hfo[1].set_title("Raw", fontsize=TYPE.condition_label)
    hfo[2].set_title("Normalized", fontsize=TYPE.condition_label)
    for ax in hfo[1:]:
        ax.set_yticks([0, 200])
    with np.load(previous.OUT / "source/hfo_showcase.npz") as old, \
            np.load(OUT / "source/hfo_showcase.npz") as new:
        for key in old.files:
            np.testing.assert_array_equal(old[key], new[key])

    y0 = hfo[-1].get_position().y0 * HEIGHT
    y1 = hfo[0].get_position().y1 * HEIGHT
    gutter = .08
    column_width = (RIGHT_WIDTH-2*gutter)/3
    positions = [[RIGHT_LEFT+i*(column_width+gutter), y0, column_width, y1-y0]
                 for i in range(3)]
    colorbar = [RIGHT_RIGHT+.12, y0, .10, y1-y0]
    groups["B_tfr"] = restored.draw_spectrum(
        fig, events, positions, colorbar, previous.SUBTITLE_BASELINE, large=True)
    spectrum_title = fig.texts[-1]
    spectrum_title.set_x((RIGHT_LEFT+RIGHT_RIGHT)/2/WIDTH)
    spectrum_title.set_ha("center")
    hfo_title = fig.text((previous.MID_X+previous.MID_WIDTH/2)/WIDTH,
                         previous.SUBTITLE_BASELINE/HEIGHT, "HFO n = 178",
                         fontsize=TYPE.identity_label, color="red", ha="center",
                         va="baseline")
    # Keep the shared label on the middle event, centered on the three-event
    # frame. Independent event windows remain independent x coordinates.
    fig.canvas.draw()
    left_label = hfo[-1].xaxis.label
    right_label = groups["B_tfr"][1].xaxis.label
    label_delta = inch_box(fig, left_label).y0 - inch_box(fig, right_label).y0
    right_label.set_transform(right_label.get_transform() +
                             ScaledTranslation(0, label_delta, fig.dpi_scale_trans))

    s.old.propagation_plot._apply_masked_paths()
    records = s.old._load_temporal_records()
    s.old._assert_masked_mi_records(records)
    mechanism = metadata["mechanism_typography"]["source"]
    start = len(fig.axes)
    d_summary, _ = current.draw_d_aligned(fig, 15., 5.7, records, mechanism)
    groups["D"] = fig.axes[start:]
    diagram, stats = groups["D"]
    previous.enlarge_mi_diagram(diagram)
    diagram.set_position(current.rect(fig, 15., 8.48, 3.80, 1.55))
    stats.set_position(current.rect(fig, 15., 5.95, 3.80, 2.34))
    previous.style_axis(stats)
    for text in stats.texts:
        if text.get_text() in ("Yuquan", "Epilepsiae"):
            text.set_y(0)
            text.set_transform(stats.get_xaxis_transform() +
                               ScaledTranslation(0, -.50, fig.dpi_scale_trans))
            text.set_va("top")
    layout.fit_visual(fig, groups["D"], (14.35, 5.28, 19.20, 10.18))
    # Preserve D's vertical spacing and statistical grammar; match the full
    # three-column TFR x-axis span, not the extent of its colorbar labels.
    for ax in groups["D"]:
        p = ax.get_position()
        ax.set_position(current.rect(fig, RIGHT_LEFT, p.y0*HEIGHT,
                                     RIGHT_WIDTH, p.height*HEIGHT))

    start = len(fig.axes)
    f_summary, fax = current.draw_f_aligned(fig, RIGHT_LEFT, previous.HEAT_Y["E"], records)
    groups["F"] = fig.axes[start:]
    previous.style_axis(fax)
    fax.set_position(current.rect(fig, RIGHT_LEFT, previous.HEAT_Y["E"],
                                  RIGHT_WIDTH, previous.HEAT_HEIGHT))
    inset = fax.child_axes[0]
    inset.set_axes_locator(_base._TransformedBoundsLocator([.58, .12, .37, .40], fax.transAxes))
    # Transparent full-page export clears Axes.patch. Preserve the inset's
    # original white background as an explicit artist above the parent field.
    inset.add_patch(Rectangle((0, 0), 1, 1, transform=inset.transAxes,
                              facecolor="white", edgecolor="none", zorder=-1))
    inset.set_yticks([0, .8])
    inset.yaxis.label.set_fontsize(TYPE.annotation)
    handles = fax.get_legend().legend_handles
    fax.legend(handles=handles, loc="upper right", frameon=True, facecolor="white",
               edgecolor=".55", framealpha=.92, fancybox=False, fontsize=TYPE.legend,
               markerscale=1.3, handlelength=.8, handletextpad=.25, labelspacing=.2,
               borderpad=.25, borderaxespad=.35)
    assert d_summary == metadata["summaries"]["D"]
    assert f_summary == metadata["summaries"]["F"]

    letters = [fig.text(x/WIDTH, y/HEIGHT, label, fontsize=TYPE.panel_letter,
                        fontweight="bold", va="top")
               for label, x, y in [("B", 10.72, 15.80), ("D", 14.18, 10.42), ("F", 14.18, 5.39)]]
    fig.canvas.draw()
    extents = {key: layout.bounds(fig, axes).extents.tolist() for key, axes in groups.items()}
    # Figure-owned titles must be included in standalone panel crops.
    b_box = Bbox.union([Bbox.from_extents(*extents[k]) for k in ("B_hfo", "B_tfr")] +
                      [inch_box(fig, hfo_title), inch_box(fig, spectrum_title)])
    cropped = {"b": b_box.padded(.04), "d": Bbox.from_extents(*extents["D"]).padded(.04),
               "f": Bbox.from_extents(*extents["F"]).padded(.04)}
    frame_axes = groups["B_tfr"][:3]
    b_frame = Bbox.union([inch_box(fig, ax) for ax in frame_axes])
    left_frame = Bbox.union([inch_box(fig, ax) for ax in hfo])
    d_frame, f_frame = inch_box(fig, stats), inch_box(fig, fax)
    np.testing.assert_allclose([b_frame.y0, b_frame.y1], [left_frame.y0, left_frame.y1], atol=1e-9)
    for frame in (d_frame, f_frame):
        np.testing.assert_allclose([frame.x0, frame.x1], [b_frame.x0, b_frame.x1], atol=1e-9)
    np.testing.assert_allclose([f_frame.y0, f_frame.y1],
                              [previous.HEAT_Y["E"], previous.HEAT_Y["E"]+previous.HEAT_HEIGHT], atol=1e-9)
    np.testing.assert_allclose(inch_box(fig, left_label).extents[[1, 3]],
                              inch_box(fig, right_label).extents[[1, 3]], atol=1e-9)
    baselines = [t.get_transform().transform(t.get_position())[1]/fig.dpi
                 for t in (hfo_title, spectrum_title)]
    np.testing.assert_allclose(baselines, [previous.SUBTITLE_BASELINE]*2, atol=1e-9)
    ticks = [inch_box(fig, t) for ax in frame_axes for t in ax.get_xticklabels()]
    assert all(a.x1+.01 < b.x0 for a, b in zip(ticks, ticks[1:]))
    assert extents["B_tfr"][2] < 19.46
    assert min(extents[key][0] for key in ("D", "F")) > 13.94
    diagram_boxes = [inch_box(fig, t) for t in diagram.texts]
    assert not any(a.overlaps(b) for i, a in enumerate(diagram_boxes) for b in diagram_boxes[i+1:])
    assert inch_box(fig, inset).y1 < inch_box(fig, fax.get_legend()).y0
    alignment = dict(
        title_baselines_inches=baselines,
        Time_s_label_bounds_inches=[inch_box(fig, t).extents.tolist() for t in (left_label, right_label)],
        B_left_three_axes_union_inches=left_frame.extents.tolist(),
        B_right_three_axes_union_inches=b_frame.extents.tolist(),
        D_data_axis_inches=d_frame.extents.tolist(), F_data_axis_inches=f_frame.extents.tolist(),
        colorbar_axis_inches=inch_box(fig, groups["B_tfr"][-1]).extents.tolist(),
        colorbar_included_in_visible_bounds=True,
        independent_event_windows_preserved=True, B_title_centered_on_three_column_frame=True,
        shared_right_axis_width_inches=RIGHT_WIDTH,
        F_top_matches_E_heatmap_top_inches=previous.HEAT_Y["E"]+previous.HEAT_HEIGHT)
    print("Rendered B titles, X labels, plot frames and F/E top edges aligned", flush=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"source/layout_overlay.{suffix}", dpi=300, transparent=True)
    original = Image.open(BASE / "figures/fig1-complete-layout.png").convert("RGBA")
    overlay = Image.open(OUT / "source/layout_overlay.png").convert("RGBA")
    result = Image.alpha_composite(original, overlay)
    result.convert("RGB").save(FIG / "fig1-complete-layout.png")
    changed = np.any(np.asarray(result) != np.asarray(original), axis=2)
    yy, xx = np.where(changed)
    allowed = np.zeros(len(xx), dtype=bool)
    for x0, y0, x1, y1 in ERASE:
        allowed |= ((xx >= x0*300-1) & (xx <= x1*300+1) &
                    (yy >= (HEIGHT-y1)*300-1) & (yy <= (HEIGHT-y0)*300+1))
    assert allowed.all(), "Changes outside B/D/F replacement regions"
    merge_pdf(BASE / "figures/fig1-complete-layout.pdf", OUT / "source/layout_overlay.pdf",
              FIG / "fig1-complete-layout.pdf")
    for letter in "ace":
        for suffix in ("png", "pdf"):
            name = f"fig1-panel{letter}.{suffix}"
            shutil.copy2(BASE / "figures" / name, FIG / name)
            assert (BASE / "figures" / name).read_bytes() == (FIG / name).read_bytes()
    for text in letters:
        text.set_visible(False)
    for patch in white_patches:
        patch.set_visible(False)
    for letter, box in cropped.items():
        for suffix in ("png", "pdf"):
            fig.savefig(FIG / f"fig1-panel{letter}.{suffix}", dpi=300, facecolor="white", bbox_inches=box)
    plt.close(fig)
    assert retained == {str(p): s.sha(p) for p in BASE.rglob("*") if p.is_file()}
    assert pointer.read_bytes() == pointer_before
    metadata.update(producer=str(Path(__file__).resolve()), source_revision=str(BASE),
        status="PENDING_AUTHOR_VISUAL_REVIEW", human_visual_acceptance="PENDING",
        changed_panels=["B layout", "D horizontal axis layout", "F axis layout"],
        spectrum_contract=str(OUT / "spectrum_contract.json"), alignment=alignment,
        preservation=dict(A_C_E_standalone_files_byte_identical=True,
            whole_image_pixels_outside_B_D_F_identical=True,
            changed_pixels=int(changed.sum()), replacement_rectangles_inches=ERASE),
        validation=str(OUT / "validation.json"), hfo_source=hfo_meta)
    metadata["visible_bounds_inches"].update(extents)
    metadata["visible_bounds_inches"]["B_with_titles"] = b_box.extents.tolist()
    metadata.pop("layout_change", None)
    metadata["panel_b"]["layout"] = alignment
    metadata["panel_b"].pop("horizontal_shift_inches", None)
    metadata["panel_b"].pop("previous_spectrum_width_inches", None)
    metadata["panel_b"]["spectrum_width_inches"] = column_width
    metadata["summaries"]["B"] = metadata["panel_b"]
    metadata["outputs"] = {str(p.relative_to(OUT)): s.sha(p) for p in FIG.glob("*")
                           if p.suffix in (".png", ".pdf")}
    source_files = [Path(__file__), Path(previous.__file__), Path(restored.__file__),
                    BASE / "metadata.json", BASE / "spectrum_contract.json"]
    metadata["input_hashes"] = {str(p): s.sha(p) for p in source_files}
    for path in source_files[:3]:
        shutil.copy2(path, OUT / "source" / path.name)
    restored.write_json(OUT / "metadata.json", metadata)
    restored.write_json(OUT / "validation.json", dict(status="PASS",
        B_titles_and_Time_s_labels_aligned=True, B_left_right_data_heights_equal=True,
        B_D_F_data_axis_horizontal_bounds_equal=True, F_E_data_axis_vertical_bounds_equal=True,
        B_colorbar_space_included=True, adjacent_TFR_ticks_separated=True,
        data_and_original_spectrum_contract_unchanged=True, all_three_events_unchanged=True,
        D_F_statistics_unchanged=True, A_C_E_standalones_byte_identical=True,
        whole_image_changes_restricted_to_B_D_F=True, previous_outputs_unchanged=True,
        human_visual_acceptance="PENDING"))
    descriptions = {
        "complete-layout": "B左右标题共基线，Time (s)共基线，左侧三轴总高度与右侧三列TFR等高。B三列总横轴与D、F主轴同宽、同左右边界，色条另占一列；F上下边界与E热图对齐。",
        "panelb": "沿用当前三个Y1 A3–A9事件和完整事件S³质心，保留±150 ms显示范围。三列仍是三个独立事件，共用一个Time (s)标签和左侧通道标签，标题与左侧HFO标题对齐。",
        "paneld": "仅调整横向坐标轴位置和宽度，与B三列TFR总宽度一致。原40人MI数据、null、散点、括号和permutation示意数值保持。",
        "panelf": "主轴横向与B、D一致，上下边界与E热图一致。原40人散点、参考对角线、灰色区域和Single/Multi配对inset的数据及统计保持。",
        "panela": "采用作者已接受的Y1 A7/A9脑模型、电极和真实波形。独立PNG/PDF逐字节保持上一版。",
        "panelc": "保留18个显示通道、全部18,190事件和显示内1–18 rank。逐像素保持热图和每行峰高归一的离散rank分布，不改变原始lagPat。",
        "panele": "保留冻结TA/TB分组及18通道的1–18显示rank。热图及均值±总体标准差逐像素保持，与C共享事件和通道。",
        "spectrum-full-window-check": "沿用上一版相同三个事件的完整500 ms核对图。虚线仅标出主图±150 ms显示范围，质心始终用完整事件计算。",
    }
    (FIG / "README.md").write_text("# Figure 1：B左右与B/D/F列的视觉对齐\n\n" +
        "保留上一版，当前候选待作者目视检查。\n\n" + "\n\n".join(
            f"### fig1-{name}.png / .pdf\n\n{text}\n\n**关注点**：按最终图中的文字、刻度和色条检查视觉对齐；仅改排版，数据与算法保持。"
            for name, text in descriptions.items()) + "\n")
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
