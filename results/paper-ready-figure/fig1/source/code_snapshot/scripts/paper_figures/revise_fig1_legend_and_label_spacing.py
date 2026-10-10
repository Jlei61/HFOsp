#!/usr/bin/env python3
"""Refine Figure 1 labels and legend shapes in the current frozen layout."""
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
from revise_fig1_lower_header_row import text_metrics

BASE = CANON / "revisions/y1_lower_header_row_20261010"
OUT = CANON / "revisions/y1_legend_label_spacing_20261010"


def revise(name):
    page = PdfReader(BASE / f"figures/{name}.pdf").pages[0]
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    labels = []

    def collect(text, cm, tm, font, size):
        if 0 < cm[4] < width and 0 < cm[5] < height:
            labels.append((text.strip(), float(cm[4]), float(cm[5]), float(size)))

    page.extract_text(visitor_text=collect)
    fig = plt.figure(figsize=(width/72, height/72))
    rectangles, changes, artists = [], [], []

    def erase(box):
        rectangles.append(box)
        x0,y0,x1,y1 = box
        fig.add_artist(Rectangle((x0/width,y0/height),(x1-x0)/width,(y1-y0)/height,
            transform=fig.transFigure,facecolor="white",edgecolor="none",zorder=-1))

    def replace(label, dx=None, x_new=None, weight="normal", remove=False):
        text,x,y,size = label
        tw,th,descent = text_metrics(text,size,weight)
        erase([x-1.5,y-descent-1.5,x+tw+1.5,y+th-descent+1.5])
        if remove:
            changes.append(dict(label=text,action="remove heading only"))
            return
        nx = x_new if x_new is not None else x+dx
        artist = fig.text(nx/width,y/height,text,ha="left",va="baseline",
            fontsize=size,fontfamily="DejaVu Sans",weight=weight,color="black")
        artists.append(artist)
        changes.append(dict(label=text,old_xy_pt=[x,y],new_xy_pt=[nx,y],fontsize_pt=size))

    full = name == "fig1-complete-layout"
    if full or name == "fig1-panelb":
        identities = [t for t in labels if t[0] == "Y1"]
        baseline = min(t[2] for t in identities)
        old = max((t for t in identities if abs(t[2]-baseline)<1e-5),key=lambda t:t[1])
        # Only the right-hand spectrum's patient label moves. A's title is
        # retained to keep the accepted electrode lead/header relationship.
        left = 14.98*72 if full else 310.
        replace(old,x_new=left,weight="bold")
    legend_gap = None
    if full or name == "fig1-panelc":
        peak = [t for t in labels if t[0] == "Peak-scaled"][-1]
        replace(peak,remove=True)
        legend = {key:[t for t in labels if t[0] == key][-1] for key in ("Day","Night")}
        handle_boxes = {}
        for key in ("Day","Night"):
            _,x,y,size = legend[key]
            # Original handles: 14 x 12.25 pt; preserve text and handle right
            # edges, expand to 24 pt and flatten to 5 pt at the same center.
            old_left, old_right = x-21,x-7
            new_left, new_right = old_right-24,old_right
            center = y+12.25/2
            erase([new_left-1,y-1,old_right+1,y+12.25+1])
            patch = Rectangle((new_left/width,(center-2.5)/height),24/width,5/height,
                transform=fig.transFigure,facecolor="white" if key=="Day" else "black",
                edgecolor="black",linewidth=.8,zorder=2)
            fig.add_artist(patch)
            handle_boxes[key] = [new_left,center-2.5,new_right,center+2.5]
            changes.append(dict(label=key+" legend marker",size_pt=[24,5],
                                text_position_preserved=True,rectangle_pt=handle_boxes[key]))
        legend_gap = handle_boxes["Night"][0]-(legend["Day"][1]+text_metrics("Day",17.5,"normal")[0])
        assert legend_gap > 5, legend_gap
    diagram_gap = None
    if full or name == "fig1-paneld":
        headings = {key:[t for t in labels if t[0] == key][-1] for key in ("Orig","Pats","Mean","Pat")}
        for key,dx in (("Orig",-6),("Pats",-6),("Mean",6),("Pat",6)):
            replace(headings[key],dx=dx)
        diagram_gap = headings["Mean"][1]+6-(headings["Orig"][1]-6+
                            text_metrics("Orig",headings["Orig"][3],"normal")[0])
        assert diagram_gap > 15, diagram_gap
        # The widened label gap must not push Mean Pat outside D's data width.
        mean_right = headings["Mean"][1]+6+text_metrics("Mean",15,"normal")[0]
        axis_right = 18.92*72 if full else width-2.88
        assert mean_right < axis_right, (mean_right,axis_right)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in artists:
        box = artist.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
        rectangles.append((np.asarray(box.extents)*72+[-1,-1,1,1]).tolist())
    for suffix in ("png","pdf"):
        fig.savefig(OUT / f"source/{name}_refinements.{suffix}",dpi=300,transparent=True)
    plt.close(fig)
    original = Image.open(BASE / f"figures/{name}.png").convert("RGBA")
    overlay = Image.open(OUT / f"source/{name}_refinements.png").convert("RGBA")
    assert max(abs(a-b) for a,b in zip(original.size,overlay.size)) <= 1
    if overlay.size != original.size:
        aligned = Image.new("RGBA",original.size,(255,255,255,0))
        aligned.paste(overlay,(0,0));overlay=aligned
    result = Image.alpha_composite(original,overlay)
    result.convert("RGB").save(OUT / f"figures/{name}.png")
    yy,xx = np.where(np.any(np.asarray(result)!=np.asarray(original),axis=2))
    allowed = np.zeros(len(xx),dtype=bool)
    for x0,y0,x1,y1 in rectangles:
        allowed |= ((xx>=x0*300/72-1)&(xx<=x1*300/72+1)&
                    (yy>=(height-y1)*300/72-1)&(yy<=(height-y0)*300/72+1))
    assert allowed.all(),name
    page.merge_page(PdfReader(OUT / f"source/{name}_refinements.pdf").pages[0])
    writer=PdfWriter();writer.add_page(page)
    with (OUT / f"figures/{name}.pdf").open("wb") as handle:writer.write(handle)
    return dict(changes=changes,allowed_annotation_rectangles_pt=rectangles,
                changed_pixels=len(xx),outside_annotations_pixel_identical=True,
                legend_inter_item_gap_pt=legend_gap,Orig_Mean_gap_pt=diagram_gap)


def main():
    (OUT / "source").mkdir(parents=True,exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    before={str(p):sha(p) for p in BASE.rglob("*") if p.is_file()}
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42})
    audits={name:revise(f"fig1-{name}") for name in ("complete-layout","panelb","panelc","paneld")}
    for name in ("panela","panele","panelf","spectrum-full-window-check"):
        for suffix in ("png","pdf"):
            path=BASE / f"figures/fig1-{name}.{suffix}"
            shutil.copy2(path,OUT / "figures" / path.name)
            assert sha(path)==sha(OUT / "figures" / path.name)
    shutil.copy2(BASE / "spectrum_contract.json",OUT / "spectrum_contract.json")
    assert before=={str(p):sha(p) for p in BASE.rglob("*") if p.is_file()}
    meta=json.loads((BASE / "metadata.json").read_text())
    meta.update(status="ANNOTATION_REFINEMENTS_PENDING_VISUAL_REVIEW",source_revision=str(BASE),
        previous_layout=str(BASE),producer=str(Path(__file__).resolve()),
        changed_panels=["B right Y1 alignment","C legend markers and rank heading","D Orig/Mean heading spacing"],
        human_visual_acceptance="PRIOR_LAYOUT_ACCEPTED_NEW_ANNOTATIONS_PENDING_CHECK",
        annotation_refinement_audit=audits,spectrum_contract=str(OUT / "spectrum_contract.json"),
        validation=str(OUT / "validation.json"),
        preservation=dict(all_scientific_pixels_and_coordinates_unchanged=True,A_E_F_files_byte_identical=True),
        input_hashes={str(p):sha(p) for p in (Path(__file__),BASE / "metadata.json")},
        outputs={str(p.relative_to(OUT)):sha(p) for p in (OUT / "figures").glob("*")})
    for d in (meta["alignment"],meta["panel_b"]["layout"],meta["summaries"]["B"]["layout"]):
        d["B_title_centered_on_three_column_frame"]=False
        d["B_title_left_aligned_on_first_column"]=True
    meta["summaries"]["C"]["ridge_display"]["heading_visible"]=False
    meta.pop("PNG_PDF_visual_self_review",None)
    audit=dict(status="PASS",annotation_refinement_audit=audits,
        all_scientific_data_and_plot_pixels_unchanged=True,A_E_F_files_byte_identical=True,
        rank_profile_scaling_unchanged=True,D_illustrative_numbers_and_40_patient_statistics_unchanged=True,
        previous_outputs_unchanged=True,human_visual_acceptance="PENDING")
    for name,value in (("metadata",meta),("validation",audit)):
        (OUT / f"{name}.json").write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    shutil.copy2(Path(__file__),OUT / "source" / Path(__file__).name)
    descriptions={
        "complete-layout":"右侧谱图Y1贴齐第一列谱图左边界，保留下移后的基线。C的Day/Night标记改为24×5 pt扁长矩形，去除rank分布上方标题；D的Orig Pats、Mean Pat分别外移6 pt。",
        "panelb":"右侧Y1改为左对齐；HFO n=178、1e-4及Time (s)对齐保持。真实谱图、质心和轴框逐像素保持。",
        "panelc":"Day/Night文字保持原位置，仅将方形标记改成扁长矩形。rank分布上方不再显示Peak-scaled，分布数值和每行峰高归一方法保持。",
        "paneld":"Orig Pats与Mean Pat分别向外移动6 pt，保留字号、示意数字、箭头和下方统计。两标题留白显著增加，未重新生成统计或重绘示意结构。",
        "panela":"已认可的A图原样保留。标题、80–250 Hz、电极引线与波形不变。",
        "panele":"已认可的E图原样保留。显示通道、冻结TA/TB标签和rank模板不变。",
        "panelf":"已认可的F图原样保留。40人MI散点、配对inset及统计不变。",
        "spectrum-full-window-check":"三个完整窗口核对图原样保留。谱图及完整事件S³质心不变。",
    }
    (OUT / "figures/README.md").write_text("# Figure 1：标签、图例和示意标题间距\n\n"+
        "\n\n".join(f"### fig1-{name}.png / .pdf\n\n{text}\n\n**关注点**：仅修改可视标注，图中数据和科学内容保持；新标注待作者目视检查。"
                    for name,text in descriptions.items())+"\n")
    (OUT / "D_heading_history.md").write_text(
        "# D标题排版来源核对\n\n"
        "y1_a7_zoom_final_20261009的D上方直接使用原始TIFF示意裁图，Orig Pats与Mean Pat单行显示且有明显留白。"
        "y1_display18_large_type_20261010首次由enlarge_mi_diagram按原示意数字重排为矢量图，"
        "标题变为两行15 pt，间距不足；local_rank版沿用同一D PNG。"
        "b_visual_alignment版仅收窄D横向轴框，这处拥挤继续保留；之后两轮标题调整的D独立文件逐字节保持。"
        "本次仅将这两组标题各外移6 pt；原示意数字及D的40人统计一直保持。\n")
    print("DONE",OUT,flush=True)


if __name__=="__main__":main()
