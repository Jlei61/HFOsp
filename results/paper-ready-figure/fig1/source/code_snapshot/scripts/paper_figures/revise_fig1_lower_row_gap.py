#!/usr/bin/env python3
"""Reduce Figure 1 C/D-to-E/F spacing by a rigid, unscaled row translation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revise_fig1_short_patient_labels import CANON, Image, PdfReader, PdfWriter, np, sha
from pypdf import Transformation
from pypdf.generic import DecodedStreamObject, NameObject

BASE = CANON / "revisions/y1_a_left_label_clear_lead_20261010"
OUT = CANON / "revisions/y1_tighter_lower_row_20261010"
DPI = 300
SHIFT_PX = 75  # 0.25 inch = 6.35 mm, exactly 18 PDF points.
# Separate the panel letters from the data/title block: D's dataset labels
# occupy the same horizontal strip as the E/F letters, in different columns.
BOXES_PX = [(0,3310,5850,4770), (0,3140,134,3270), (4241,3140,4371,3270)]


def clipped_page(page, boxes_pt, inverse=False):
    result = copy.copy(page)
    path = []
    if inverse:
        path.append(f"0 0 {float(page.mediabox.width)} {float(page.mediabox.height)} re")
    path.extend(f"{x0} {y0} {x1-x0} {y1-y0} re" for x0,y0,x1,y1 in boxes_pt)
    clip = "\n".join(path) + ("\nW* n\n" if inverse else "\nW n\n")
    stream = DecodedStreamObject()
    stream.set_data(b"q\n"+clip.encode()+page.get_contents().get_data()+b"\nQ\n")
    result[NameObject("/Contents")] = stream
    return result


def main():
    (OUT / "figures").mkdir(parents=True,exist_ok=True)
    (OUT / "source").mkdir(exist_ok=True)
    before = {str(p):sha(p) for p in BASE.rglob("*") if p.is_file()}
    image = Image.open(BASE / "figures/fig1-complete-layout.png").convert("RGB")
    assert image.size == (5850,4770)
    old = np.asarray(image)
    fixed = old.copy()
    for x0,y0,x1,y1 in BOXES_PX:
        fixed[y0:y1,x0:x1] = 255
    moved = fixed.copy()
    for x0,y0,x1,y1 in BOXES_PX:
        # The moved regions must never overwrite any stationary plot or label.
        assert np.all(fixed[y0-SHIFT_PX:y1-SHIFT_PX,x0:x1] == 255)
        moved[y0-SHIFT_PX:y1-SHIFT_PX,x0:x1] = old[y0:y1,x0:x1]
    result = moved[:-SHIFT_PX]
    for x0,y0,x1,y1 in BOXES_PX:
        np.testing.assert_array_equal(result[y0-SHIFT_PX:y1-SHIFT_PX,x0:x1],old[y0:y1,x0:x1])
    affected = np.zeros(result.shape[:2],dtype=bool)
    for x0,y0,x1,y1 in BOXES_PX:
        affected[max(0,y0-SHIFT_PX):min(len(result),y1),x0:x1] = True
    np.testing.assert_array_equal(result[~affected],old[:len(result)][~affected])
    Image.fromarray(result).save(OUT / "figures/fig1-complete-layout.png",dpi=(DPI,DPI))

    page = PdfReader(BASE / "figures/fig1-complete-layout.pdf").pages[0]
    width,height = float(page.mediabox.width),float(page.mediabox.height)
    shift_pt = SHIFT_PX*72/DPI
    boxes_pt = [[x0*72/DPI,height-y1*72/DPI,x1*72/DPI,height-y0*72/DPI]
                for x0,y0,x1,y1 in BOXES_PX]
    writer = PdfWriter()
    output = writer.add_blank_page(width=width,height=height-shift_pt)
    # The canvas loses the same height as the row translation. In bottom-up
    # PDF coordinates, E/F therefore stay fixed; A-D shift down by 18 points.
    output.merge_transformed_page(clipped_page(page,boxes_pt,inverse=True),
                                   Transformation().translate(ty=-shift_pt))
    output.merge_page(clipped_page(page,boxes_pt))
    # Older PDF layers contain covered plots. At a new clipping boundary,
    # antialiasing can expose subpixel seams through their white masks. Clear
    # only a verified empty four-point strip around this boundary.
    seam_y = boxes_pt[0][3]
    seam_top = round((height-shift_pt-seam_y-2)*DPI/72)
    seam_bottom = round((height-shift_pt-seam_y+2)*DPI/72)
    assert np.all(result[seam_top:seam_bottom] == 255)
    clean = DecodedStreamObject()
    clean.set_data(output.get_contents().get_data()+
        f"\nq 1 1 1 rg 0 {seam_y-2} {width} 4 re f Q\n".encode())
    output[NameObject("/Contents")] = writer._add_object(clean)
    with (OUT / "figures/fig1-complete-layout.pdf").open("wb") as f: writer.write(f)
    for name in ("panela","panelb","panelc","paneld","panele","panelf","spectrum-full-window-check"):
        for suffix in ("png","pdf"):
            p = BASE / f"figures/fig1-{name}.{suffix}"
            shutil.copy2(p,OUT / "figures" / p.name)
            assert sha(p)==sha(OUT / "figures" / p.name)
    shutil.copy2(BASE / "spectrum_contract.json",OUT / "spectrum_contract.json")
    assert before == {str(p):sha(p) for p in BASE.rglob("*") if p.is_file()}
    audit = dict(status="PASS",row_shift_up_mm=6.35,row_shift_up_px=SHIFT_PX,
        canvas_size_px=list(Image.fromarray(result).size),canvas_size_pt=[width,height-shift_pt],
        source_row_regions_pixels=BOXES_PX,source_row_regions_pt=boxes_pt,
        E_F_rigid_translation_pixel_exact=True,stationary_content_pixel_exact=True,
        overlap_with_stationary_content_pixels=0,standalone_A_to_F_byte_identical=True,
        scientific_data_and_all_panel_sizes_preserved=True,previous_outputs_unchanged=True,
        human_visual_acceptance="PENDING")
    meta = json.loads((BASE / "metadata.json").read_text())
    meta.update(status="LOWER_ROW_GAP_PENDING_VISUAL_REVIEW",source_revision=str(BASE),
        previous_layout=str(BASE),producer=str(Path(__file__).resolve()),
        changed_panels=["E/F placement in complete layout only"],
        human_visual_acceptance="CURRENT_ROW_SPACING_PENDING_CHECK",
        lower_row_gap_revision=audit,spectrum_contract=str(OUT / "spectrum_contract.json"),
        validation=str(OUT / "validation.json"),
        preservation=dict(scientific_data_and_all_panel_sizes_preserved=True,standalone_A_to_F_byte_identical=True),
        input_hashes={str(p):sha(p) for p in (Path(__file__),BASE / "metadata.json",BASE / "figures/fig1-complete-layout.png",BASE / "figures/fig1-complete-layout.pdf")})
    meta.pop("PNG_PDF_visual_self_review",None)
    meta.pop("visual_self_review",None)
    # These bounds are in bottom-up canvas inches. The bottom row is stationary
    # relative to the cropped page bottom, while A-D move down by 0.25 inch.
    for key,bounds in meta["visible_bounds_inches"].items():
        if key not in ("E_heat","E_rank","F"):
            bounds[1]-=.25; bounds[3]-=.25
    alignment = meta["alignment"]
    alignment["title_baselines_inches"] = [v-.25 for v in alignment["title_baselines_inches"]]
    for bounds in alignment["Time_s_label_bounds_inches"]:
        bounds[1]-=.25; bounds[3]-=.25
    for key in ("B_left_three_axes_union_inches","B_right_three_axes_union_inches","D_data_axis_inches","colorbar_axis_inches"):
        alignment[key][1]-=.25; alignment[key][3]-=.25
    meta["source_CE_visible_gap_inches"] = meta["CE_visible_gap_inches"]
    meta["CE_visible_gap_inches"] -= .25
    meta["prior_annotation_coordinates"] = "Earlier annotation audits use the source canvas; apply the rigid row/page transform recorded in lower_row_gap_revision."
    descriptions = {
        "complete-layout":"将E/F整行及面板字母整体上移6.35 mm，收紧与C/D之间的留白；画布同步缩短，底部留白保持。各图、文字、色条及横向对齐均不缩放。",
        "panela":"保留Y1左对齐及收短的外部弯曲引线。脑模型、电极触点与真实双极波形保持。",
        "panelb":"独立B图保持。原始HFO谱、群体事件谱及完整事件S³质心不变。",
        "panelc":"独立C图保持。18通道、Day/Night扁长图例及无标题rank分布不变。",
        "paneld":"独立D图保持。Orig Pats和Mean Pat的间距及40人统计不变。",
        "panele":"独立E图保持，仅在完整拼版中整体上移。原始lagPat、冻结模板标签及18通道显示rank不变。",
        "panelf":"独立F图保持，仅在完整拼版中整体上移。统计、图例、配对inset以及与E的上下对齐不变。",
        "spectrum-full-window-check":"完整窗口核对图保持。三个事件的谱图与S³质心不变。",
    }
    (OUT / "figures/README.md").write_text("# Figure 1：收紧C/D与E/F行距\n\n"+
        "\n\n".join(f"### fig1-{name}.png / .pdf\n\n{text}\n\n**关注点**：上下行文字和图例无交叠；本次行距待作者目视检查。"
                      for name,text in descriptions.items())+"\n")
    meta["outputs"] = {str(p.relative_to(OUT)):sha(p) for p in (OUT / "figures").glob("*")}
    for name,value in (("metadata",meta),("validation",audit)):
        (OUT / f"{name}.json").write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    shutil.copy2(Path(__file__),OUT / "source" / Path(__file__).name)
    print("DONE",OUT,flush=True)


if __name__ == "__main__": main()
