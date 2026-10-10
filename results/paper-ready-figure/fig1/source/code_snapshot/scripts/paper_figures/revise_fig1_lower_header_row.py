#!/usr/bin/env python3
"""Lower Figure 1 A/B annotations to the existing scientific-offset row.

Preserve all plot geometry, samples and statistical figures. Only annotation
artists change, in the full PNG/PDF and the matching standalone A/B exports.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revise_fig1_short_patient_labels import (
    CANON, FontProperties, Image, PdfReader, PdfWriter, Rectangle, TextToPath,
    np, plt, sha,
)

BASE = CANON / "revisions/y1_final_short_labels_20261010"
OUT = CANON / "revisions/y1_lower_header_row_20261010"
FULL_OFFSET_X, FULL_BASELINE = 838.8, 1090.2
FULL_FREQUENCY_X = 412.526350398
FULL_IDENTITY_BASELINE = 1117.44


def text_metrics(text, size, weight):
    return TextToPath().get_text_width_height_descent(
        text, FontProperties(family="DejaVu Sans", size=size, weight=weight), False)


def revise(name):
    page = PdfReader(BASE / f"figures/{name}.pdf").pages[0]
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    labels = []

    def collect(text, cm, tm, font, size):
        text = text.strip()
        if text in ("Y1", "80–250 Hz", "HFO n = 178", "1e-4"):
            item = (text, float(cm[4]), float(cm[5]), float(size))
            if 0 < cm[4] < width and 0 < cm[5] < height and item not in labels:
                labels.append(item)

    page.extract_text(visitor_text=collect)
    identities = [v for v in labels if v[0] == "Y1"]
    assert len(identities) == (2 if name == "fig1-complete-layout" else 1)
    shift_y = identities[0][2]-FULL_IDENTITY_BASELINE
    baseline = FULL_BASELINE+shift_y
    old_labels = [v for v in labels if v[0] != "1e-4"]
    fig = plt.figure(figsize=(width/72, height/72))
    rectangles, artists, details = [], [], []
    for text, x, y, size in old_labels:
        weight = "bold" if text == "Y1" else "normal"
        tw, th, descent = text_metrics(text, size, weight)
        # Erase the actual old glyph span, leaving the nearby electrode cable
        # intact. The new rendered extent is added separately to the QA mask.
        box = [x-1.5, y-descent-1.5, x+tw+1.5, y+th-descent+1.5]
        rectangles.append(box)
        fig.add_artist(Rectangle((box[0]/width, box[1]/height),
            (box[2]-box[0])/width, (box[3]-box[1])/height,
            transform=fig.transFigure, facecolor="white", edgecolor="none", zorder=-1))
        if text == "Y1":
            new, anchor, ha, fontsize, color = text, x+tw/2, "center", size, "black"
        elif text == "80–250 Hz":
            new, anchor, ha, fontsize, color = text, 756+(x-FULL_FREQUENCY_X), "right", size, "black"
        else:
            offset = next(v for v in labels if v[0] == "1e-4")
            new, anchor, ha, fontsize, color = "HFO n=178", 986.4+(offset[1]-FULL_OFFSET_X), "right", offset[3], "red"
        artist = fig.text(anchor/width, baseline/height, new, va="baseline", ha=ha,
                          fontfamily="DejaVu Sans", fontsize=fontsize, weight=weight, color=color)
        artists.append(artist)
        details.append(dict(old=text, new=new, baseline_pt=baseline,
            anchor_x_pt=anchor, horizontal_alignment=ha, fontsize_pt=fontsize))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_boxes = []
    for artist in artists:
        b = artist.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
        box = np.asarray(b.extents)*72
        text_boxes.append(box)
        rectangles.append((box+np.array([-1,-1,1,1])).tolist())
    for i, a in enumerate(text_boxes):
        for b in text_boxes[i+1:]:
            assert a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]
    if name != "fig1-panela":
        offset = next(v for v in labels if v[0] == "1e-4")
        count = next(v for v in details if v["new"] == "HFO n=178")
        gap = count["anchor_x_pt"]-text_metrics(count["new"], count["fontsize_pt"], "normal")[0] \
              - offset[1]-text_metrics(offset[0], offset[3], "normal")[0]
        assert gap > 5, gap
        np.testing.assert_allclose(baseline, offset[2], atol=1e-6)
    else:
        gap = None
    overlay_pdf = OUT / f"source/{name}_header.pdf"
    overlay_png = OUT / f"source/{name}_header.png"
    fig.savefig(overlay_pdf, transparent=True)
    fig.savefig(overlay_png, dpi=300, transparent=True)
    plt.close(fig)
    original = Image.open(BASE / f"figures/{name}.png").convert("RGBA")
    overlay = Image.open(overlay_png).convert("RGBA")
    assert max(abs(a-b) for a,b in zip(original.size, overlay.size)) <= 1
    if overlay.size != original.size:
        aligned = Image.new("RGBA", original.size, (255,255,255,0))
        aligned.paste(overlay, (0,0)); overlay = aligned
    result = Image.alpha_composite(original, overlay)
    result.convert("RGB").save(OUT / f"figures/{name}.png")
    yy, xx = np.where(np.any(np.asarray(result) != np.asarray(original), axis=2))
    allowed = np.zeros(len(xx), dtype=bool)
    for x0, y0, x1, y1 in rectangles:
        allowed |= ((xx >= x0*300/72-1) & (xx <= x1*300/72+1) &
                    (yy >= (height-y1)*300/72-1) & (yy <= (height-y0)*300/72+1))
    assert allowed.all(), name
    page.merge_page(PdfReader(overlay_pdf).pages[0])
    writer = PdfWriter(); writer.add_page(page)
    with (OUT / f"figures/{name}.pdf").open("wb") as handle:
        writer.write(handle)
    return dict(labels=details, count_offset_gap_pt=gap, changed_pixels=len(xx),
                allowed_annotation_rectangles_pt=rectangles,
                pixels_outside_annotations_identical=True)


def main():
    (OUT / "source").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    before = {str(p): sha(p) for p in BASE.rglob("*") if p.is_file()}
    plt.rcParams.update({"font.family":"DejaVu Sans", "pdf.fonttype":42})
    audits = {name: revise(f"fig1-{name}") for name in ("complete-layout", "panela", "panelb")}
    for name in ("panelc", "paneld", "panele", "panelf", "spectrum-full-window-check"):
        for suffix in ("png", "pdf"):
            source = BASE / f"figures/fig1-{name}.{suffix}"
            shutil.copy2(source, OUT / "figures" / source.name)
            assert sha(source) == sha(OUT / "figures" / source.name)
    shutil.copy2(BASE / "spectrum_contract.json", OUT / "spectrum_contract.json")
    assert before == {str(p): sha(p) for p in BASE.rglob("*") if p.is_file()}
    meta = json.loads((BASE / "metadata.json").read_text())
    meta.update(status="ANNOTATION_ROW_REVISION_PENDING_VISUAL_REVIEW",
        source_revision=str(BASE), previous_layout=str(BASE), producer=str(Path(__file__).resolve()),
        changed_panels=["A header annotations", "B header annotations"],
        human_visual_acceptance="PRIOR_PLOT_LAYOUT_ACCEPTED_NEW_HEADER_PENDING_CHECK",
        header_annotation_audit=audits, spectrum_contract=str(OUT / "spectrum_contract.json"),
        validation=str(OUT / "validation.json"),
        preservation=dict(plot_pixels_and_geometry_unchanged=True,
                          C_D_E_F_standalone_files_byte_identical=True),
        input_hashes={str(p):sha(p) for p in (Path(__file__), BASE / "metadata.json")},
        outputs={str(p.relative_to(OUT)):sha(p) for p in (OUT / "figures").glob("*")})
    meta["subtitle_alignment"] = dict(baseline_inches=FULL_BASELINE/72,
        labels=["Y1","HFO n=178","Y1"], reference="existing 1e-4 offset baseline",
        HFO_count_fontsize_pt=17.5, HFO_count_right_aligned=True,
        frequency_position="right-aligned above A waveform axis", frequency_font_size_pt=17.5)
    meta["alignment"]["title_baselines_inches"] = [FULL_BASELINE/72]*2
    for panel in (meta["panel_b"], meta["summaries"]["B"]):
        panel["layout"]["title_baselines_inches"] = [FULL_BASELINE/72]*2
    meta["annotation_positions_supersede_previous_visible_bounds"] = True
    for key in ("title_edit_audit", "PNG_PDF_visual_self_review"):
        meta.pop(key, None)
    audit = dict(status="PASS", header_annotation_audit=audits,
        all_scientific_data_and_plot_geometry_unchanged=True,
        C_D_E_F_files_byte_identical=True, previous_outputs_unchanged=True,
        new_header_human_visual_acceptance="PENDING")
    for name, data in (("metadata",meta),("validation",audit)):
        (OUT / f"{name}.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    shutil.copy2(Path(__file__), OUT / "source" / Path(__file__).name)
    descriptions = {
        "complete-layout":"A/B的Y1标题下移至既有1e-4标注的基线。HFO n=178放在该行右端，字号与1e-4一致；80–250 Hz移至A波形轴右上方并右对齐。",
        "panela":"Y1标题下移，保留原水平中心。80–250 Hz与标题共基线并对齐波形轴右边界，脑模型、电极、引线及波形保持。",
        "panelb":"HFO n=178缩为与1e-4相同的17.5 pt，并在该行右对齐。右侧Y1同基线下移，真实谱图、质心、轴框和Time (s)保持。",
        "panelc":"原18通道热图与峰高归一的rank分布保持。标题仍为Y1 | n=18,190，独立文件逐字节保留。",
        "paneld":"原40人MI对照及permutation示意保持。独立PNG/PDF逐字节保留。",
        "panele":"原18通道TA/TB热图及rank均值和总体标准差保持。独立PNG/PDF逐字节保留。",
        "panelf":"原40人散点、Single/Multi配对inset及与E对齐的轴框保持。独立PNG/PDF逐字节保留。",
        "spectrum-full-window-check":"原三个事件完整500 ms核对图保持。独立PNG/PDF逐字节保留。",
    }
    (OUT / "figures/README.md").write_text("# Figure 1：下移顶部标注\n\n"+
        "\n\n".join(f"### fig1-{name}.png / .pdf\n\n{text}\n\n**关注点**：仅修改顶部文字的位置，绘图区及数据保持；新标注行待作者目视检查。"
                    for name,text in descriptions.items())+"\n")
    print("DONE",OUT,flush=True)


if __name__ == "__main__":
    main()
