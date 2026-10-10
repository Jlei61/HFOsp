#!/usr/bin/env python3
"""Align Figure 3 by complete visible groups and tighten its three rows."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
import pymupdf

from scripts.paper_figures import build_fig3_phenotype_field_layout as layout
from scripts.paper_figures import render_fig3_y1_matched_row as y1

SOURCE = layout.PAPER / "fig3/revisions/y1_gamma_matched_row_20261010"
OUT = layout.PAPER / "fig3/revisions/visual_alignment_compact_rows_20261010"
ROW_GAP = .16
COLUMN_GAP = .12
LEFT, RIGHT = .24, .12
TOP, BOTTOM = .12, .12


def rel(path):
    return str(path.relative_to(layout.ROOT))


def raster(path, dpi=130):
    with pymupdf.open(path) as doc:
        pix = doc[0].get_pixmap(dpi=dpi)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)


def bounds(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return Bbox.union([ax.get_tightbbox(renderer) for ax in axes]).transformed(fig.dpi_scale_trans.inverted())


def positions(fig):
    w, h = fig.get_size_inches()
    return [np.array(ax.get_position().bounds) * [w, h, w, h] for ax in fig.axes]


def set_canvas(fig, width):
    old = positions(fig)
    fig.set_size_inches(width, fig.get_size_inches()[1])
    for ax, box in zip(fig.axes, old):
        layout.position(ax, fig, *box)


def shift(fig, axes, dx):
    w, h = fig.get_size_inches()
    for ax in axes:
        box = np.array(ax.get_position().bounds) * [w, h, w, h]
        box[0] += dx
        layout.position(ax, fig, *box)


def data_signature(fig):
    """Data, limits, colors, text and type sizes; excludes permitted positions."""
    digest = hashlib.sha256()

    def add(value):
        if isinstance(value, np.ndarray):
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        else:
            digest.update(repr(value).encode())

    for ax in fig.axes:
        add(ax.get_xlim()); add(ax.get_ylim())
        add(ax.get_xticks()); add(ax.get_yticks())
        for line in ax.lines:
            add(np.asarray(line.get_xdata())); add(np.asarray(line.get_ydata()))
            add(line.get_color()); add(line.get_linewidth()); add(line.get_alpha())
        for collection in ax.collections:
            add(np.asarray(collection.get_offsets()))
            if collection.get_array() is not None:
                add(np.asarray(collection.get_array()))
            for path in collection.get_paths():
                add(path.vertices); add(path.codes)
            add(collection.get_clim()); add(collection.get_cmap().name)
        for image in ax.images:
            add(np.asarray(image.get_array())); add(image.get_extent())
            add(image.get_clim()); add(image.get_cmap().name)
        texts = list(ax.texts) + [ax.title, ax._left_title, ax._right_title,
                                ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
        if ax.get_legend() is not None:
            texts += list(ax.get_legend().get_texts())
        for text in texts:
            add(text.get_text()); add(text.get_fontsize()); add(text.get_color())
    return digest.hexdigest()


def fit_bottom(fig, width):
    """Expand/contract only the data width; retain fonts and colorbar width."""
    set_canvas(fig, width + .16)
    axes = fig.axes
    for _ in range(5):
        box = bounds(fig, axes)
        difference = width - box.width
        if abs(difference) < 1e-8:
            break
        data_box = positions(fig)[0]
        data_box[2] += difference
        assert data_box[2] > 1.8
        layout.position(axes[0], fig, *data_box)
        if len(axes) > 1:
            shift(fig, axes[1:], difference)
    box = bounds(fig, axes)
    np.testing.assert_allclose(box.width, width, atol=1e-7)
    shift(fig, axes, .08 - box.x0)
    return bounds(fig, axes)


def save_readme(out, audit):
    (out / "figures/README.md").write_text(f"""# Figure 3：色条纳入视觉列宽，收紧三行留白

沿用作者已认可的 E10/SZ3 broadband 与 Y1/SZ6 gamma 三列结构；本次只改排版。重新运行原 producer 后，A–E 原位 PDF 渲染与上一版逐像素一致，随后才调整位置和下排数据轴宽度。

### fig3-panela.png / .pdf / .svg
E10/SZ3 原始波形/TFR、早期发作场及间期 TA 场保持原图内容。三列整体平移，列宽计算同时包含坐标标签、标题、色条及其刻度。
**关注点**：场、触点、空间坐标、色阶和字体均保持；不再仅以数据轴框作为列宽。

### fig3-panelb.png / .pdf / .svg
Y1/SZ6 仍为左列原始波形/TFR、中列固定 0–10 s 的 30–80 Hz 相对能量场、右列 Y1 TA rank 场。B 与 A 的同列元素执行相同水平平移，原 X 范围、Y 轴 −20～20 mm、EEG marker 和 TA 自身投影保持。
**关注点**：没有重新选窗口、通道、模板或颜色；TA 相关仍为 +0.933796。

### fig3-panelc.png / .pdf / .svg
原三组配对统计完整保留，数据轴横向拓宽，使包含纵轴标签的整个 C 与上方波形/TFR及色条组合的视觉列宽对应。小提琴、箱线、配对点、显著性括号和原字体保持。
**关注点**：变化是版面宽度，统计量与 n=17/16/11 未变。

### fig3-paneld.png / .pdf / .svg
E10 signed q 时程与原中位数、IQR、单次轨迹保持。按上方能量场连同色条的完整宽度调整 D 的横轴长度，纵轴 −1～1 及右上纵向图例保持。
**关注点**：D 的完整可见左右边界与该列的共同边界对齐。

### fig3-panele.png / .pdf / .svg
17 人热图、原排序、灰色分隔带与色条不变。热图的数据轴宽度与色条位置一起调整，整个 E 的可见宽度对应上方 TA 场及其色条；底排三个数据轴上下边界保持一致。
**关注点**：完整可见边界包含色条标签，不以热图轴框单独定宽。

### fig3-complete-layout.png / .pdf / .svg
三列用 A/B 对应组的可见边界并集确定；C/D/E 的完整可见左右边界分别对齐这些列。三行之间的可见留白统一为 {audit['row_gap_mm']:.2f} mm，画布随之缩短，面板字母同步移动。
**关注点**：新间距及视觉对齐待作者目视确认；数据、色阶、字体和统计语法检查已通过。

### fig3-complete-layout-preview.png
完整版的 130 dpi 预览，与 PDF 内容相同。高分辨率 PNG 为 600 dpi。
**关注点**：用于检查三行留白、完整列宽和标签间距。
""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--set-current", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    source_registry = json.loads((SOURCE / "figure3_panel_registry.json").read_text())
    source_hashes = {rel(SOURCE / f"figures/fig3-panel{k}.pdf"):
                     layout.digest(SOURCE / f"figures/fig3-panel{k}.pdf") for k in "abcde"}
    source_hashes.update({rel(SOURCE / "figure3_panel_registry.json"):
                         layout.digest(SOURCE / "figure3_panel_registry.json")})
    figures, signatures, reproduced = {}, {}, {}
    original_save = layout.save
    with tempfile.TemporaryDirectory(prefix="fig3_visual_alignment_") as temporary:
        tmp = Path(temporary)
        (tmp / "figures").mkdir()

        def capture(fig, stem, dpi):
            key = stem.name[-1]
            figures[key] = fig
            original_save(fig, stem, dpi)

        layout.save = capture
        try:
            layout.example_row("a", layout.CASES["a"], tmp, 130)
            y1.render(tmp, dpi=130)
            layout.cohort_panel(tmp, 130)
            layout.signed_panel(tmp, 130)
            layout.heatmap_panel(tmp, 130)
        finally:
            layout.save = original_save
        for key, fig in figures.items():
            old = raster(SOURCE / f"figures/fig3-panel{key}.pdf")
            new = raster(tmp / f"figures/fig3-panel{key}.pdf")
            assert old.shape == new.shape
            reproduced[key] = int(np.any(old != new, axis=2).sum())
            assert reproduced[key] == 0, f"Original {key} changed before layout: {reproduced[key]} pixels"
            signatures[key] = data_signature(fig)
    print("A–E reproduced pixel-identically before layout", flush=True)

    # Wider visible-group spacing now permits the same Y-label padding in A/B.
    figures["b"].axes[5].yaxis.labelpad = figures["a"].axes[5].yaxis.labelpad
    groups = {key: [figures[key].axes[:3], figures[key].axes[3:5], figures[key].axes[5:7]] for key in "ab"}
    group_bounds = {key: [bounds(figures[key], axes) for axes in groups[key]] for key in "ab"}
    union = [Bbox.union([group_bounds[key][j] for key in "ab"]) for j in range(3)]
    widths = [box.width for box in union]
    lefts = [LEFT]
    for width in widths[:-1]:
        lefts.append(lefts[-1] + width + COLUMN_GAP)
    canvas_width = lefts[-1] + widths[-1] + RIGHT
    for key in "ab":
        fig = figures[key]
        set_canvas(fig, canvas_width)
        for j, axes in enumerate(groups[key]):
            shift(fig, axes, lefts[j] - union[j].x0)

    bottom_bounds = {}
    for j, key in enumerate("cde"):
        bottom_bounds[key] = fit_bottom(figures[key], widths[j])

    aligned_bounds = {key: [bounds(figures[key], axes) for axes in groups[key]] for key in "ab"}
    upper_y = Bbox.union([box for key in "ab" for box in aligned_bounds[key]])
    lower_y = Bbox.union(list(bottom_bounds.values()))
    upper_height, lower_height = upper_y.height, lower_y.height
    tops = [TOP, TOP + upper_height + ROW_GAP, TOP + 2 * (upper_height + ROW_GAP)]
    canvas_height = tops[-1] + lower_height + BOTTOM

    # Fields and signal plots move rigidly. Only bottom data widths may change.
    for key, fig in figures.items():
        fig.canvas.draw()
        assert data_signature(fig) == signatures[key], f"Data/style drift after layout in {key}"
        if key in "ab":
            old_boxes = source_registry["panels"][key]["display"]["axes_bounds_fraction"]
            for name, index in (("raw", 0), ("tfr", 1), ("ictal", 3), ("template", 5)):
                current = positions(fig)[index]
                previous = np.array(old_boxes[name]) * [11.85, 3.65, 11.85, 3.65]
                np.testing.assert_allclose(current[1:], previous[1:], atol=1e-9)
        original_save(fig, out / f"figures/fig3-panel{key}", args.dpi)
    np.testing.assert_allclose([positions(figures[k])[0][1::2] for k in "cde"],
                               [[.55, 1.95]] * 3, atol=1e-9)

    doc = pymupdf.open()
    page = doc.new_page(width=canvas_width * 72, height=canvas_height * 72)
    placements = {}
    for key in "abcde":
        j = "cde".find(key)
        if key in "ab":
            top = tops["ab".index(key)]
            clip = pymupdf.Rect(0, (3.65 - upper_y.y1) * 72, canvas_width * 72,
                               (3.65 - upper_y.y0) * 72)
            dest = pymupdf.Rect(0, top * 72, canvas_width * 72, (top + upper_height) * 72)
            letter_x = .035
        else:
            top = tops[2]
            # Keep a small clipping allowance for stroke caps at the axis edge.
            clip = pymupdf.Rect(.06 * 72, (2.85 - lower_y.y1) * 72,
                               (.10 + widths[j]) * 72, (2.85 - lower_y.y0) * 72)
            dest = pymupdf.Rect((lefts[j] - .02) * 72, top * 72, (lefts[j] + widths[j] + .02) * 72,
                               (top + lower_height) * 72)
            letter_x = lefts[j] - .205
        np.testing.assert_allclose([clip.width, clip.height], [dest.width, dest.height], atol=1e-9)
        with pymupdf.open(out / f"figures/fig3-panel{key}.pdf") as source:
            page.show_pdf_page(dest, source, 0, clip=clip)
        page.insert_text((letter_x * 72, (top + .22) * 72), key.upper(), fontsize=19, fontname="hebo")
        placements[key] = {"clip_points": list(clip), "destination_points": list(dest),
                           "scale": 1., "letter_inches": [letter_x, top + .22]}
    stem = out / "figures/fig3-complete-layout"
    doc.save(stem.with_suffix(".pdf"), deflate=True, garbage=4)
    page.get_pixmap(dpi=args.dpi).save(stem.with_suffix(".png"))
    page.get_pixmap(dpi=130).save(out / "figures/fig3-complete-layout-preview.png")
    stem.with_suffix(".svg").write_text(page.get_svg_image(text_as_path=True))
    doc.close()

    for path, sha in source_hashes.items():
        assert layout.digest(layout.ROOT / path) == sha
    with pymupdf.open(stem.with_suffix(".pdf")) as document:
        page = document[0]
        outside_text = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if not (page.rect + (-.5, -.5, .5, .5)).contains(pymupdf.Rect(span["bbox"])):
                        outside_text.append(span["text"])
        assert not outside_text, outside_text

    columns = []
    for j in range(3):
        ab = Bbox.union([aligned_bounds[k][j] for k in "ab"])
        np.testing.assert_allclose([ab.x0, ab.x1], [lefts[j], lefts[j] + widths[j]], atol=1e-9)
        bb = bottom_bounds["cde"[j]]
        np.testing.assert_allclose(bb.width, widths[j], atol=1e-7)
        columns.append({"left_inches": lefts[j], "right_inches": lefts[j] + widths[j],
                        "width_inches": widths[j], "upper_A_bbox_inches": list(aligned_bounds['a'][j].extents),
                        "upper_B_bbox_inches": list(aligned_bounds['b'][j].extents),
                        "bottom_local_bbox_inches": list(bb.extents)})
    old_row_gaps = [3.85 + (3.65 - union[1].y1) - (3.65 - min(b.y0 for b in group_bounds['a'])),
                    7.9 + (2.85 - lower_y.y1) - (3.85 + 3.65 - min(b.y0 for b in group_bounds['b']))]
    audit = {"status": "PASS", "source_reproduction_changed_pixels": reproduced,
             "data_limits_colors_text_fonts_unchanged": True, "upper_plot_sizes_unchanged": True,
             "source_files_unchanged": True, "full_composite_scale": 1.,
             "alignment_includes_colorbars_ticks_titles_and_axis_labels": True,
             "template_field_ylabel_padding_matched_AB": True,
             "bottom_visible_widths_match_upper_column_union": True,
             "row_gap_mm": ROW_GAP * 25.4, "previous_row_gaps_mm": [g * 25.4 for g in old_row_gaps],
             "column_gap_mm": COLUMN_GAP * 25.4,
             "canvas_inches": [canvas_width, canvas_height], "previous_canvas_inches": [11.85, 10.75],
             "columns": columns, "bottom_data_axis_widths_inches": {k: positions(figures[k])[0][2] for k in 'cde'},
             "pdf_text_inside_canvas": True, "human_visual_acceptance": "PENDING_NEW_SPACING_REVIEW"}
    registry = copy.deepcopy(source_registry)
    registry.update({"schema": "figure3_visual_alignment_compact_rows_v1", "producer": rel(Path(__file__).resolve()),
                     "predecessor": rel(SOURCE), "status": "PENDING_AUTHOR_VISUAL_REVIEW",
                     "human_visual_acceptance": "PENDING_NEW_SPACING_REVIEW",
                     "source_content_acceptance": "AUTHOR_ACCEPTED_THREE_COLUMN_CASES_AND_CONTENT",
                     "canvas_inches": [canvas_width, canvas_height], "placements_inches": None,
                     "placements_points": placements, "visual_alignment": audit})
    registry.pop("preserved_panel_assets", None)
    registry["layout_contract"] = {"alignment": "complete visible group including colorbar, ticks, titles and labels",
                                   "row_gap_inches": ROW_GAP, "column_gap_inches": COLUMN_GAP,
                                   "columns": columns}
    for key in "abcde":
        prior = registry["panels"][key].get("display")
        if prior:
            registry["panels"][key]["prior_display"] = prior
        registry["panels"][key]["display"] = {"canvas_inches": figures[key].get_size_inches().tolist(),
                                               "axes_bounds_inches": [list(box) for box in positions(figures[key])],
                                               "data_and_font_signature": signatures[key]}
        layout.dump(out / f"panel_{key}_metadata.json", registry["panels"][key])
    layout.dump(out / "figure3_panel_registry.json", registry)
    layout.dump(out / "validation.json", audit)
    layout.dump(out / "source_snapshot/manifest.json", source_hashes)
    save_readme(out, audit)
    if args.set_current:
        layout.dump(layout.PAPER / "fig3/current_revision.json", {
            "schema": "paper_figure_current_revision_v1", "figure": "Figure 3", "layout": "A–E",
            "layout_version": "visual_alignment_compact_rows_20261010", "revision_dir": rel(out),
            "producer": registry["producer"], "registry": rel(out / "figure3_panel_registry.json"),
            "complete_pdf": rel(stem.with_suffix('.pdf')),
            "preview_png": rel(out / "figures/fig3-complete-layout-preview.png"),
            "status": registry["status"], "human_visual_acceptance": registry["human_visual_acceptance"],
            "source_content_acceptance": registry["source_content_acceptance"],
            "predecessor": rel(SOURCE), "row_gap_mm": ROW_GAP * 25.4,
            "alignment_includes_colorbars": True})
    print(json.dumps({"output": str(out), "canvas_inches": audit["canvas_inches"],
                      "row_gap_mm": audit["row_gap_mm"], "previous_row_gaps_mm": audit["previous_row_gaps_mm"],
                      "columns": [c['width_inches'] for c in columns]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
