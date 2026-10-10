#!/usr/bin/env python3
"""Left-align A's patient label and shorten its illustrative external lead.

Reuse the current frozen scientific panels. Only the patient annotation and
the cable above the connector cap change; patient geometry and signals do not.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revise_fig1_short_patient_labels import (
    CANON, Image, PdfReader, PdfWriter, Rectangle, np, plt, sha,
)
from revise_fig1_lower_header_row import text_metrics
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

BASE = CANON / "revisions/y1_legend_label_spacing_20261010"
OUT = CANON / "revisions/y1_a_left_label_clear_lead_20261010"
SNAPSHOT = CANON / "revisions/y1_a7_zoom_final_20261009/source/panel_a_snapshot"
BACKGROUND = CANON / "candidates/y1_local_rank_peak_profiles_20261010/source/a_geometry_background.png"
TITLE_LEFT = 499.376038
TITLE_BASELINE = 1090.2
OLD_TITLE_LEFT = 609.380540


def lead_geometry():
    meta = json.loads((SNAPSHOT / "metadata.json").read_text())
    vertices = np.array(meta["display"]["external_lead"]["path_figure_coordinates"])
    with Image.open(BACKGROUND) as im:
        image_size = np.array(im.size)
    # Same frozen 350-dpi crop and placement as build_fig1_y1_local_rank.draw_a.
    old = (vertices * image_size - [331, image_size[1]-2022]) * (10.2*72/4037)
    old += [.30*72, 10.60*72]
    # Retain the first quarter of the existing Bezier and its outgoing tangent.
    # Only the free, illustrative end bends into a shorter rightward wave.
    t = .25
    a = (1-t)*old[:3] + t*old[1:4]
    b = (1-t)*a[:2] + t*a[1:]
    split = (1-t)*b[0] + t*b[1]
    new = np.array([old[0], a[0], b[0], split, b[1], [442,1100],
                    [454,1096], [466,1092], [474,1091], [483,1096]])
    np.testing.assert_allclose(new[4]-new[3], 3*(new[3]-new[2]))
    np.testing.assert_allclose(new[7]-new[6], new[6]-new[5])
    return old, new, 1.15*10.2/(4037/350)


def revise(name):
    page = PdfReader(BASE / f"figures/{name}.pdf").pages[0]
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    labels = []

    def collect(text, cm, tm, font, size):
        if text.strip() == "Y1" and 0 < cm[4] < width and 0 < cm[5] < height:
            labels.append((float(cm[4]), float(cm[5]), float(size)))

    page.extract_text(visitor_text=collect)
    full = name == "fig1-complete-layout"
    if full:
        labels = [v for v in labels if v[0] < 800 and v[1] > 1000]
    old_x, baseline, size = min(labels, key=lambda v:v[1])
    offset = np.array([OLD_TITLE_LEFT-old_x, TITLE_BASELINE-baseline])
    if full:
        np.testing.assert_allclose(offset, [0,0], atol=1e-5)
    old_lead, new_lead, linewidth = lead_geometry()
    fig = plt.figure(figsize=(width/72, height/72))
    rectangles = []

    def erase(box):
        x0,y0,x1,y1 = box
        rectangles.append(list(box))
        patch = Rectangle((x0/width,y0/height),(x1-x0)/width,(y1-y0)/height,
            transform=fig.transFigure,facecolor="white",edgecolor="none",zorder=-1)
        fig.add_artist(patch)
        return patch

    tw,th,descent = text_metrics("Y1",size,"bold")
    erase([old_x-1.5,baseline-descent-1.5,old_x+tw+1.5,baseline+th-descent+1.5])
    # The cap and the lower lead are below this region and remain pixel exact.
    cable_box = np.array([419,1077,554,1112])-np.tile(offset,2)
    clip = erase(cable_box)
    path = MplPath((new_lead-offset)/[width,height],
                   [MplPath.MOVETO]+[MplPath.CURVE4]*9)
    cable = PathPatch(path,transform=fig.transFigure,facecolor="none",
        edgecolor="#737b82",linewidth=linewidth,capstyle="round",joinstyle="round")
    cable.set_clip_path(clip)
    fig.add_artist(cable)
    title = fig.text((TITLE_LEFT-offset[0])/width,baseline/height,"Y1",
        ha="left",va="baseline",fontsize=size,fontfamily="DejaVu Sans",weight="bold")
    fig.canvas.draw()
    bbox = title.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    rectangles.append((np.array(bbox.extents)*72+[-1,-1,1,1]).tolist())
    clearance = TITLE_LEFT-new_lead[:,0].max()-linewidth/2
    assert clearance > 15, clearance
    for suffix in ("png","pdf"):
        fig.savefig(OUT / f"source/{name}_a_header.{suffix}",dpi=300,transparent=True)
    plt.close(fig)
    original = Image.open(BASE / f"figures/{name}.png").convert("RGBA")
    overlay = Image.open(OUT / f"source/{name}_a_header.png").convert("RGBA")
    assert max(abs(a-b) for a,b in zip(original.size,overlay.size)) <= 1
    if overlay.size != original.size:
        aligned = Image.new("RGBA",original.size,(255,255,255,0))
        aligned.paste(overlay,(0,0)); overlay=aligned
    result = Image.alpha_composite(original,overlay)
    result.convert("RGB").save(OUT / f"figures/{name}.png")
    yy,xx = np.where(np.any(np.asarray(result)!=np.asarray(original),axis=2))
    allowed = np.zeros(len(xx),dtype=bool)
    for x0,y0,x1,y1 in rectangles:
        allowed |= ((xx>=x0*300/72-1)&(xx<=x1*300/72+1)&
                    (yy>=(height-y1)*300/72-1)&(yy<=(height-y0)*300/72+1))
    assert allowed.all(),name
    page.merge_page(PdfReader(OUT / f"source/{name}_a_header.pdf").pages[0])
    writer = PdfWriter(); writer.add_page(page)
    with (OUT / f"figures/{name}.pdf").open("wb") as f: writer.write(f)
    return dict(old_title_xy_pt=[old_x,baseline],new_title_xy_pt=[TITLE_LEFT-offset[0],baseline],
        title_fontsize_pt=size,lead_title_horizontal_clearance_pt=clearance,
        lead_old_full_canvas_vertices_pt=old_lead.tolist(),lead_new_full_canvas_vertices_pt=new_lead.tolist(),
        lead_linewidth_pt=linewidth,lead_is_illustrative_not_measured=True,
        cap_shaft_brain_waveforms_pixel_identical=True,changed_pixels=len(xx),
        outside_annotation_regions_pixel_identical=True,allowed_rectangles_pt=rectangles)


def main():
    (OUT / "source").mkdir(parents=True,exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    before = {str(p):sha(p) for p in BASE.rglob("*") if p.is_file()}
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42})
    audits = {name:revise(f"fig1-{name}") for name in ("complete-layout","panela")}
    for name in ("panelb","panelc","paneld","panele","panelf","spectrum-full-window-check"):
        for suffix in ("png","pdf"):
            path = BASE / f"figures/fig1-{name}.{suffix}"
            shutil.copy2(path,OUT / "figures" / path.name)
            assert sha(path)==sha(OUT / "figures" / path.name)
    shutil.copy2(BASE / "spectrum_contract.json",OUT / "spectrum_contract.json")
    assert before == {str(p):sha(p) for p in BASE.rglob("*") if p.is_file()}
    meta = json.loads((BASE / "metadata.json").read_text())
    meta.update(status="A_LABEL_AND_LEAD_PENDING_VISUAL_REVIEW",source_revision=str(BASE),
        previous_layout=str(BASE),producer=str(Path(__file__).resolve()),
        changed_panels=["A patient label and illustrative external lead"],
        human_visual_acceptance="PRIOR_LAYOUT_RETAINED_A_LABEL_AND_LEAD_PENDING_CHECK",
        a_header_audit=audits,spectrum_contract=str(OUT / "spectrum_contract.json"),
        validation=str(OUT / "validation.json"),
        preservation=dict(all_scientific_data_and_plot_pixels_unchanged=True,B_to_F_files_byte_identical=True),
        input_hashes={str(p):sha(p) for p in (Path(__file__),BASE / "metadata.json",SNAPSHOT / "metadata.json",BACKGROUND)},
        outputs={str(p.relative_to(OUT)):sha(p) for p in (OUT / "figures").glob("*")})
    meta.pop("PNG_PDF_visual_self_review",None)
    lead = meta["summaries"]["A"]["external_lead"]
    lead["source_path_figure_coordinates"] = lead.pop("path_figure_coordinates")
    lead["current_path_full_canvas_pt"] = audits["complete-layout"]["lead_new_full_canvas_vertices_pt"]
    lead["revision"] = "shorter illustrative end; cap connection and initial tangent retained"
    meta["summaries"]["A"]["patient_label_alignment"] = "left on waveform data axis"
    meta["annotation_positions_supersede_previous_visible_bounds"] = True
    validation = dict(status="PASS",a_header_audit=audits,
        all_scientific_data_and_plot_pixels_unchanged=True,B_to_F_files_byte_identical=True,
        previous_outputs_unchanged=True,human_visual_acceptance="PENDING")
    for name,value in (("metadata",meta),("validation",validation)):
        (OUT / f"{name}.json").write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    shutil.copy2(Path(__file__),OUT / "source" / Path(__file__).name)
    descriptions = {
        "complete-layout":"A的Y1贴齐波形轴左边界，保持已认可的标题基线与字号；外部弯曲引线缩短，和文字留出间隙。其余面板及所有数据保持。",
        "panela":"Y1左对齐，电极帽子上方的示意引线收短，保留连续连接。脑模型、电极杆、帽子、彩色触点和真实双极波形逐像素保持，80–250 Hz仍右对齐。",
        "panelb":"沿用已认可的B布局。真实波形、谱图及完整事件S³质心保持。",
        "panelc":"沿用18通道显示及扁长Day/Night图例。rank分布仍无标题，数据及逐行峰高缩放保持。",
        "paneld":"保留已分开的Orig Pats与Mean Pat。示意数字与40位患者统计保持。",
        "panele":"18通道热图、冻结TA/TB标签和显示rank保持。完整原始lagPat不变。",
        "panelf":"40位患者统计、散点和配对inset保持。坐标轴对齐保持。",
        "spectrum-full-window-check":"三个完整窗口核对图保持。谱图与完整事件S³质心不变。",
    }
    (OUT / "figures/README.md").write_text("# Figure 1：A标签左对齐与引线避让\n\n"+
        "\n\n".join(f"### fig1-{name}.png / .pdf\n\n{text}\n\n**关注点**：A标题与引线无交叠；其余内容保持，新标注待作者目视检查。"
                      for name,text in descriptions.items())+"\n")
    print("DONE",OUT,flush=True)


if __name__ == "__main__": main()
