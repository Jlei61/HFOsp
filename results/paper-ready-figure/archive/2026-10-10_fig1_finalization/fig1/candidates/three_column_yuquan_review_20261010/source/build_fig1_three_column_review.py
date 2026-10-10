#!/usr/bin/env python3
"""Three visually aligned columns and an unfiltered Yuquan C/E review atlas."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import build_fig1_y1_selection as current
from scripts.paper_figures import plot_fig1_single_hfo_schematic as hfo
from matplotlib.transforms import Bbox
from PIL import Image, ImageDraw, ImageFont

s = current.shared
np, plt = s.np, s.plt
OUT = s.CANON / "candidates/three_column_yuquan_review_20261010"
FIG = OUT / "figures"
ATLAS = s.CANON / "candidates/yuquan_patient_review_20261009"
COLS = [(0.30, 10.50), (11.05, 12.80), (13.40, 17.70)]
ROW_BOTTOMS = [10.75, 5.75, 0.75]
ROW_HEIGHT = 4.40


def bounds(fig, axes):
    renderer = fig.canvas.get_renderer()
    return Bbox.union([ax.get_tightbbox(renderer) for ax in axes]).transformed(
        fig.dpi_scale_trans.inverted())


def fit_visual(fig, axes, target, *, vertical=True):
    """Fit visible extents, including text, legends, colorbars and child axes.

    Text retains its font size; only data rectangles and internal spacing move.
    Iteration accounts for the fixed physical extent of ticks and labels.
    """
    x0, y0, x1, y1 = target
    for _ in range(12):
        fig.canvas.draw()
        box = bounds(fig, axes)
        err = max(abs(box.x0-x0), abs(box.x1-x1),
                  abs(box.y0-y0) if vertical else 0,
                  abs(box.y1-y1) if vertical else 0)
        if err < .003:
            break
        sx = (x1-x0)/box.width
        sy = (y1-y0)/box.height if vertical else 1.
        for ax in axes:
            pos = ax.get_position()
            old = np.array([pos.x0*fig.get_figwidth(), pos.y0*fig.get_figheight(),
                            pos.width*fig.get_figwidth(), pos.height*fig.get_figheight()])
            new = [x0+(old[0]-box.x0)*sx,
                   y0+(old[1]-box.y0)*sy if vertical else old[1], old[2]*sx, old[3]*sy]
            ax.set_position(current.rect(fig, *new))
    fig.canvas.draw()
    box = bounds(fig, axes)
    assert max(abs(box.x0-x0), abs(box.x1-x1)) < .012, (box.bounds, target)
    if vertical:
        assert max(abs(box.y0-y0), abs(box.y1-y1)) < .012, (box.bounds, target)
    return box


def draw_hfo(fig, bottom):
    metadata = json.loads((s.CANON / "figures/fig1-panelb1_metadata.json").read_text())
    paths = metadata["source_paths"]
    snippets = hfo._load_annotated_hfos(Path(paths["snippets_npz"]), Path(paths["annotations_pickle"]))
    raw, norm, t, f = hfo._mean_spectrograms(snippets, 1000.)
    te = hfo._full_extent_edges(t, 0., .6)
    fe = hfo._frequency_edges(f)
    axes = [fig.add_axes(current.rect(fig, 11.65, bottom+.48+i*1.39, 1.12, 1.02))
            for i in (2, 1, 0)]
    a, b, c = axes
    time = np.arange(snippets.shape[1])/1000
    a.plot(time, snippets.T, color="black", lw=.28, alpha=.23)
    a.plot(time, snippets.mean(0), color="#FFD000", lw=1.05)
    a.set_title("HFO n = 178", color="red", fontsize=12.5, pad=4)
    lim = np.nanpercentile(raw, [1., 99.])
    norm_lim = float(np.nanpercentile(np.abs(norm), 99.))
    b.pcolormesh(te, fe, raw, cmap="coolwarm", vmin=lim[0], vmax=lim[1], rasterized=True)
    c.pcolormesh(te, fe, norm, cmap="coolwarm", vmin=-norm_lim, vmax=norm_lim, rasterized=True)
    for ax, name in ((b, "raw Spec"), (c, "normalized Spec")):
        ax.set_title(name, fontsize=10.5, pad=3)
        ax.set_ylabel("Freq (Hz)", fontsize=9.5, labelpad=5)
        ax.set(ylim=(0, 240), yticks=[0, 100, 200])
    c.set_xlabel("Time (s)", fontsize=10, labelpad=5)
    for ax in axes:
        ax.set(xlim=(0, .6), xticks=[0, .25, .5])
        ax.set_xticklabels(["0.00", "0.25", "0.50"])
        ax.tick_params(labelsize=9, length=2.5, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(.55)
    np.savez_compressed(OUT / "source/hfo_showcase.npz", snippets_V=snippets,
                        raw_spec=raw, normalized_spec=norm, times=t, frequencies=f)
    return axes, metadata


def atlas():
    rows = json.loads((ATLAS / "patient_inventory.json").read_text())
    assert [r["patient"] for r in rows] == [f"Y{i}" for i in range(1,21)]
    # Recheck the existing gallery against the live frozen records, without refitting.
    for row in rows:
        r = json.loads(Path(row["source_record"]).read_text())
        a = r["adaptive_cluster"]
        assert len(r["channel_names"]) == row["n_channels"]
        assert a["chosen_k"] == row["k"]
        assert a["n_valid_events"] == row["n_events"]
        assert [c["n_events"] for c in a["clusters"]] == row["cluster_counts"]
        assert [c["raw_tau"] for c in a["clusters"]] == row["within_mode_tau"]
        assert sum(row["cluster_counts"]) == row["n_events"]
    shutil.copy2(ATLAS / "figures/yuquan_all20_CE.pdf", FIG / "yuquan_all20_CE.pdf")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25)
    gallery = []
    for start in range(0,20,4):
        canvas = Image.new("RGB", (3200,2280), "white")
        draw = ImageDraw.Draw(canvas)
        for i,row in enumerate(rows[start:start+4]):
            x, y = (i%2)*1600, (i//2)*1140
            label = row["patient"]
            draw.text((x+25,y+15), f"{label} | {row['n_channels']} channels | K={row['k']} | n={row['n_events']:,}",
                      fill="#111111", font=font)
            with Image.open(ATLAS/f"figures/per_patient/{label}-CE.png") as im:
                im = im.convert("RGB")
                im.thumbnail((1580,1060),Image.Resampling.LANCZOS)
                canvas.paste(im,(x+(1600-im.width)//2,y+65+(1060-im.height)//2))
        name=f"yuquan_CE_Y{start+1}_Y{start+4}.png"
        canvas.save(FIG/name)
        gallery.append(name)
    # A focused comparison complements, rather than filters, the all-patient atlas.
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(FIG / "yuquan_comparison_Y1_Y12_Y8_Y3_Y2_Y16.pdf") as pdf:
        for label in ["Y1","Y12","Y8","Y3","Y2","Y16"]:
            path=ATLAS/f"figures/per_patient/{label}-CE.png"
            with Image.open(path) as im:
                fig=plt.figure(figsize=(14,14*im.height/im.width))
            s.image_in_rect(fig,[0,0,1,1],path)
            pdf.savefig(fig,dpi=250)
            plt.close(fig)
    s.write_json(OUT / "patient_inventory.json", rows)
    shutil.copy2(ATLAS / "patient_inventory.csv", OUT / "patient_inventory.csv")
    return rows, gallery


def main():
    (OUT / "source").mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    pointer_path = s.CANON / "current_revision.json"
    pointer_before = pointer_path.read_bytes()
    base = Path(json.loads(pointer_before)["revision_root"])
    assert base == current.OUT
    accepted = json.loads((base / "metadata.json").read_text())
    assert accepted["panel_a"]["human_visual_acceptance"] == "ACCEPTED"
    base_hashes = {str(p):s.sha(p) for p in base.rglob("*") if p.is_file()}
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "pdf.fonttype":42,"svg.fonttype":"none","axes.unicode_minus":False})
    rows, gallery = atlas()
    print("All 20 Yuquan patients verified; comparison atlas ready", flush=True)
    s.old.propagation_plot._apply_masked_paths()
    records=s.old._load_temporal_records()
    s.old._assert_masked_mi_records(records)
    record=next(r for r in records if s.public_patient_label(r["dataset"],r["subject"])=="Y1")
    arr=s.old._load_exemplar_arrays(record,max_events=10**9)
    assert len(arr["valid_events"])==18190 and len(arr["channel_names"])==26
    spectrum=current.load_y1_spectrum()
    fig=plt.figure(figsize=(18.,16.))
    groups, summaries = {}, {}

    # A is uniformly enlarged from its accepted rendering: no camera, waveform,
    # electrode-shape or connector changes. Crop only external white margins.
    with Image.open(base/"figures/fig1-panela.png") as im:
        rgb=np.asarray(im.convert("RGB"))
        yy,xx=np.where(rgb.min(2)<250)
        crop_box=[max(0,int(xx.min())-4),max(0,int(yy.min())-4),
                  min(im.width,int(xx.max())+5),min(im.height,int(yy.max())+5)]
        cropped=im.crop(crop_box)
        cropped.save(OUT/"source/panel_a_content.png")
    a_width=COLS[0][1]-COLS[0][0]
    a_height=a_width*cropped.height/cropped.width
    a_bottom=ROW_BOTTOMS[0]+ROW_HEIGHT/2-a_height/2
    s.image_in_rect(fig,current.rect(fig,COLS[0][0],a_bottom,a_width,a_height),
                    OUT/"source/panel_a_content.png")
    groups["A"]=fig.axes[-1:]
    a_scale=350*a_width/cropped.width
    # The original A patient title is 17 pt. Match its final displayed size.
    patient_title_size=17*a_scale

    groups["B_hfo"],hfo_meta=draw_hfo(fig,ROW_BOTTOMS[0])
    start=len(fig.axes); text_start=len(fig.texts)
    summaries["B"]=current.draw_b_compact(fig,11.,ROW_BOTTOMS[0]-.5,spectrum)
    temp_axes=fig.axes[start:]
    temp_axes[0].remove()  # B1 is rendered natively above, using the same data.
    for text in fig.texts[text_start:]:
        text.remove()
    groups["B_tfr"]=temp_axes[1:]
    tfr=groups["B_tfr"]
    for i,ax in enumerate(tfr[:3]):
        ax.set_position(current.rect(fig,14.+i*1.10,ROW_BOTTOMS[0]+.48,1.02,3.6))
    tfr[3].set_position(current.rect(fig,17.32,ROW_BOTTOMS[0]+.48,.10,3.6))
    tfr[0].text(0,1.045,"Yuquan Y1",transform=tfr[0].transAxes,
                fontsize=patient_title_size,fontweight="bold",va="bottom")

    for panel,clustered,y in [("C",False,ROW_BOTTOMS[1]),("E",True,ROW_BOTTOMS[2])]:
        start=len(fig.axes)
        summaries[panel]=s.draw_ce(fig,current.rect(fig,.2,y,12.5,4.4),arr,"Yuquan Y1",clustered)
        heat,bar,profile,*strip=fig.axes[start:]
        heat.set_position(current.rect(fig,.85,y+.58,9.1,3.45))
        bar.set_position(current.rect(fig,10.15,y+.58,.13,3.45))
        profile.set_position(current.rect(fig,11.22,y+.58,1.45,3.45))
        heat.tick_params(axis="y",labelsize=9)
        groups[panel+"_heat"]=[heat,bar]+strip
        groups[panel+"_rank"]=[profile]
        if strip:
            strip[0].set_position(current.rect(fig,.85,y+.40,9.1,.10))
        fit_visual(fig,groups[panel+"_heat"],(*[COLS[0][0],y],COLS[0][1],y+ROW_HEIGHT))
        # Align the actual rank data vertically with the heatmap after fitting.
        p=heat.get_position()
        profile.set_position([profile.get_position().x0,p.y0,profile.get_position().width,p.height])
        fit_visual(fig,[profile],(COLS[1][0],y,COLS[1][1],y+ROW_HEIGHT),vertical=False)
        summaries[panel]["rank_data_height_matches_heatmap"]=True

    start=len(fig.axes)
    summaries["D"],_=current.draw_d_aligned(fig,14.,ROW_BOTTOMS[1]+.65,records,accepted["mechanism"])
    groups["D"]=fig.axes[start:]
    start=len(fig.axes)
    summaries["F"],_=current.draw_f_aligned(fig,14.,ROW_BOTTOMS[2]+.65,records)
    groups["F"]=fig.axes[start:]
    assert summaries["D"]==accepted["panel_d"]
    assert summaries["F"]==accepted["panel_f"]

    for name,col,row in [("B_hfo",1,0),("B_tfr",2,0),("D",2,1),("F",2,2)]:
        y=ROW_BOTTOMS[row]
        fit_visual(fig,groups[name],(COLS[col][0],y,COLS[col][1],y+ROW_HEIGHT))
    for letter,x,y in [("A",.08,15.76),("B",10.86,15.76),("C",.08,10.48),
                        ("D",13.20,10.48),("E",.08,5.48),("F",13.20,5.48)]:
        fig.text(x/18,y/16,letter,fontsize=23,fontweight="bold",va="top")
    fig.canvas.draw()
    extents={name:bounds(fig,axes).extents.tolist() for name,axes in groups.items()}
    left_widths=[extents[n][2]-extents[n][0] for n in ("A","C_heat","E_heat")]
    middle_widths=[extents[n][2]-extents[n][0] for n in ("B_hfo","C_rank","E_rank")]
    right_sizes=[[extents[n][2]-extents[n][0],extents[n][3]-extents[n][1]] for n in ("B_tfr","D","F")]
    assert np.ptp(left_widths)<.025 and np.ptp(middle_widths)<.025
    assert np.max(np.ptp(right_sizes,axis=0))<.025
    assert all(b[0]>=0 and b[1]>=0 and b[2]<=18 and b[3]<=16 for b in extents.values())
    label_checks=current.check_channel_labels(fig,arr)
    stem=FIG/"fig1-complete-layout"
    fig.savefig(stem.with_suffix(".png"),dpi=300,facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"),dpi=300,facecolor="white")
    # Export panels from the same figure state, with their complete visual bounds.
    for panel,names in {"a":["A"],"b":["B_hfo","B_tfr"],"c":["C_heat","C_rank"],
                        "d":["D"],"e":["E_heat","E_rank"],"f":["F"]}.items():
        box=Bbox.union([Bbox.from_extents(*extents[n]) for n in names]).padded(.04)
        for ext in ("png","pdf"):
            fig.savefig(FIG/f"fig1-panel{panel}.{ext}",dpi=300,facecolor="white",bbox_inches=box)
    plt.close(fig)
    assert pointer_path.read_bytes()==pointer_before
    assert base_hashes=={str(p):s.sha(p) for p in base.rglob("*") if p.is_file()}
    s.write_json(OUT/"metadata.json",dict(
        status="PENDING_AUTHOR_VISUAL_REVIEW",producer=str(Path(__file__).resolve()),
        source_revision=str(base),patient="Y1",patient_reselection="OPEN; no replacement made",
        n_patients_reviewed=20,all_patients_and_frozen_clusters_retained=True,
        brain_camera_and_waveform_samples_unchanged=True,panel_a_uniform_enlargement=True,
        panel_a_source_crop_pixels=crop_box,panel_a_scale_from_native_render=a_scale,
        panel_a_enlargement_vs_previous=a_scale/(8.65/12.72755905511811),
        patient_title_points_A_and_B=patient_title_size,figure_size_inches=[18,16],
        visible_bounds_inches=extents,left_column_widths_inches=left_widths,
        middle_column_widths_inches=middle_widths,right_column_sizes_inches=right_sizes,
        alignment_includes="ticks, labels, titles, colorbars, legends and F inset",
        no_anisotropic_image_stretch=True,statistics_unchanged=True,
        summaries=summaries,hfo_source=hfo_meta,label_checks=label_checks,
        original_revision_retained=True,human_visual_acceptance="PENDING"))
    s.write_json(OUT/"validation.json",dict(status="PASS",all20_records_match_gallery=True,
        all_event_counts_conserved=True,frozen_k_preserved=True,old_revision_unchanged=True,
        current_pointer_unchanged=True,left_middle_right_visible_bounds_aligned=True,
        whole_figure_requires_author_review=True))
    shutil.copy2(__file__,OUT/"source"/Path(__file__).name)
    lines=["# Figure 1：三列视觉对齐与Yuquan全患者审阅", "",
           "本次为待目视检查的候选，C/E病例重新开放选择；整图暂保留Y1，不替换已定稿入口。", ""]
    descriptions={
        "fig1-complete-layout.png / .pdf":"按A/C/E、HFO/rank、TFR/D/F三列重排。对齐范围含刻度、标签和色条；A由已接受图等比例放大，脑朝向、电极造型与真实波形均保持。",
        "yuquan_all20_CE.pdf":"Y1至Y20每人一页，上排按时间、下排按冻结类别排列。保留全部有效事件、全部通道、真实未参与位置及总体标准差；K>2的患者不强制变成两类。",
        "yuquan_comparison_Y1_Y12_Y8_Y3_Y2_Y16.pdf":"补充比较原Y1与五位通道较少或双模式较清楚的候选。只用于示例目视选择，不改变聚类、D/F统计或患者身份。"}
    for name in gallery:
        descriptions[name]="四位患者的原C/E图等比例缩略总览。高通道病例请结合全20页PDF放大检查，未删除较弱病例。"
    for panel in "abcdef":
        descriptions[f"fig1-panel{panel}.png / .pdf"]="从同一完整画布导出独立面板，无左上字母。保持原数据及统计，只调整显示大小和列间对齐。"
    for name,desc in descriptions.items():
        lines += [f"### {name}","",desc,"","**关注点**：比较两类各自的时序稳定性与全图视觉对齐；当前候选待作者目视检查。",""]
    (FIG/"README.md").write_text("\n".join(lines))
    print("DONE",OUT,flush=True)


if __name__=="__main__":
    main()
