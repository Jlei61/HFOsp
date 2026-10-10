#!/usr/bin/env python3
"""Wider Y1 heatmaps and a shared A/B subtitle baseline; frozen data."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import build_fig1_y1_selection as current
from scripts.paper_figures import build_fig1_three_column_review as layout
from scripts.paper_figures import build_fig1a_recording_chain as chain
from src.paper_figure_typography import DENSE_MULTIPANEL_TYPOGRAPHY
from matplotlib.transforms import Bbox, ScaledTranslation
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch
from matplotlib.textpath import TextPath
from PIL import Image

s = current.shared
np, plt = s.np, s.plt
OUT = s.CANON / "candidates/y1_wide_heatmaps_aligned_titles_20261010"
FIG = OUT / "figures"
PREVIOUS = s.CANON / "candidates/three_column_yuquan_review_20261010"
PREVIOUS_DISPLAY = s.CANON / "candidates/y1_display18_large_type_20261010"
TYPE = DENSE_MULTIPANEL_TYPOGRAPHY.scaled(1.25)
WIDTH, HEIGHT = 19.5, 15.9
MID_X, MID_WIDTH = 11.65, 2.05
HEAT_Y = {"C": 6.07, "E": .98}
HEAT_HEIGHT = 3.30
HEAT_GROUP_RIGHT = 11.30
SUBTITLE_BASELINE = 15.52


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def select_display(arr):
    events, labels = arr["valid_events"], arr["labels"]
    rows = []
    for i, name in enumerate(arr["channel_names"]):
        parts, means = [], []
        for k in (0, 1):
            ids = events[labels == k]
            mask = arr["bools"][i, ids] & np.isfinite(arr["ranks"][i, ids])
            parts.append(float(mask.mean()))
            means.append(float(arr["ranks"][i, ids][mask].mean()))
        participation = float(arr["bools"][i, events].mean())
        low = participation < .65
        late = participation < .75 and min(means)/25 > .60
        rows.append(dict(channel=name, index=i, participation=participation,
                         participation_TA=parts[0], participation_TB=parts[1],
                         mean_rank_TA=means[0], mean_rank_TB=means[1],
                         hidden=bool(low or late),
                         reason="participation < 0.65" if low else
                         "participation < 0.75 and both mean ranks > 0.60*25" if late else "retained"))
    hidden = {r["index"] for r in rows if r["hidden"]}
    order = np.array([i for i in arr["channel_order"] if i not in hidden], dtype=int)
    assert len(order) == 18
    assert {"A7", "A9"}.issubset({arr["channel_names"][i] for i in order})
    view = dict(arr, channel_order=order,
                ordered_names=[arr["channel_names"][i] for i in order])
    # The full arrays, event eligibility and labels remain the original objects.
    for key in ("ranks", "bools", "labels", "valid_events", "clustered_events_all"):
        assert view[key] is arr[key]
    with (OUT/"display_channel_selection.csv").open("w") as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    s.write_json(OUT/"display_channel_selection.json",dict(
        patient="Y1", statistical_unit="channel; participation denominator is all 18190 valid events",
        rule="hide participation < .65 OR (participation < .75 AND both participating-event mean ranks > .60*25)",
        display_only=True, original_channel_count=26, displayed_channel_count=18,
        hidden_channels=[r["channel"] for r in rows if r["hidden"]],
        display_order_bottom_to_top=view["ordered_names"], original_rank_range=[0,25], channels=rows))
    return view, rows


def style_axis(ax, *, dense=False, colorbar=False):
    ax.tick_params(labelsize=TYPE.dense_tick if dense else TYPE.colorbar_tick if colorbar else TYPE.tick_label,
                   length=4, pad=3, width=.85)
    ax.xaxis.label.set_fontsize(TYPE.axis_label)
    ax.yaxis.label.set_fontsize(TYPE.axis_label)
    ax.title.set_fontsize(TYPE.colorbar_label if colorbar else TYPE.condition_label)
    ax.xaxis.labelpad=6
    ax.yaxis.labelpad=6
    for text in ax.texts:
        text.set_fontsize(TYPE.significance if text.get_text().strip() in ("*","**","***","n.s.") else TYPE.annotation)
    legend=ax.get_legend()
    if legend:
        for text in legend.get_texts(): text.set_fontsize(TYPE.legend)
    for child in ax.child_axes:
        style_axis(child, dense=True)


def render_a_background(base):
    src=base/"source/panel_a_snapshot"
    for name in ("selection.json", "brain_projection.json", "y1_brain.png", "recording_and_geometry.npz"):
        shutil.copy2(src/"source"/name,OUT/"source"/name)
    source=json.loads((src/"source/selection.json").read_text())
    projection=json.loads((src/"source/brain_projection.json").read_text())
    with np.load(src/"source/recording_and_geometry.npz") as z:
        traces=z["signals_V"].copy()
        by_name=dict(zip(z["names"].tolist(),z["coords_mm"]))
    with plt.rc_context():
        fig,meta=chain.draw_candidate(OUT,source,traces,by_name,projection,return_figure=True)
        trace_pos=fig.axes[-1].get_position().bounds
        fig.axes[-1].remove()
        for text in list(fig.texts): text.remove()
        fig.savefig(OUT/"source/a_geometry_background.png",dpi=350,facecolor="white")
        plt.close(fig)
    accepted=json.loads((src/"metadata.json").read_text())
    for key in ("closeup_coordinates_mm","readout_locations_mm","linked_channels",
                "magnification_leaders","external_lead"):
        assert meta[key]==accepted["display"][key],key
    crop=json.loads((PREVIOUS/"metadata.json").read_text())["panel_a_source_crop_pixels"]
    with Image.open(OUT/"source/a_geometry_background.png") as im:
        full_size=im.size
        im.crop(crop).save(OUT/"source/a_geometry_cropped.png")
    return source,traces,trace_pos,crop,full_size,meta


def draw_a(fig, inputs):
    source,traces,pos,crop,full_size,meta=inputs
    left,bottom,width=.30,10.60,10.2
    crop_width,crop_height=crop[2]-crop[0],crop[3]-crop[1]
    height=width*crop_height/crop_width
    s.image_in_rect(fig,current.rect(fig,left,bottom,width,height),OUT/"source/a_geometry_cropped.png")
    background=fig.axes[-1]
    px,py,pw,ph=pos
    trace_rect=[left+(px*full_size[0]-crop[0])/crop_width*width,
                bottom+(crop[3]-(py+ph)*full_size[1])/crop_height*height,
                pw*full_size[0]/crop_width*width,ph*full_size[1]/crop_height*height]
    # Convert the image's top-down crop coordinates to the figure's bottom-up axes.
    trace_rect[1]=bottom+(full_size[1]*py-(full_size[1]-crop[3]))/crop_height*height
    ax=fig.add_axes(current.rect(fig,*trace_rect))
    channels=source["selection"]["selected_channels"]
    colors=source["plot"]["channel_colors"]
    fs=source["plot"]["fs_out"]; gap=source["plot"]["trace_spacing_V"]
    for i,row in enumerate(traces): ax.plot(np.arange(len(row))/fs,row+i*gap,c="black",lw=.60)
    for boundary in (.16,.32): ax.axvline(boundary,color=".7",ls="--",lw=.7)
    ax.set(xlim=(0,.48),ylim=(11.8*gap,-.8*gap),xticks=[0,.2,.4],
           yticks=np.arange(12)*gap,xlabel="Time (s)")
    ax.set_yticklabels(source["selection"]["display_labels"])
    style_axis(ax)
    ax.tick_params(axis="y",labelsize=TYPE.dense_tick,pad=26,length=0)
    ax.spines[["top","right"]].set_visible(False)
    for text,name in zip(ax.get_yticklabels(),channels): text.set_color(colors.get(name,"black"))
    gx=-.065
    ax.plot([gx,gx],ax.get_ylim(),transform=ax.get_yaxis_transform(),color="#b4b7ba",lw=.75,ls=(0,(2,2)),clip_on=False)
    for i,name in enumerate(channels):
        ax.scatter([gx],[i*gap],transform=ax.get_yaxis_transform(),s=49 if name in colors else 13,
                   color=colors.get(name,"#a7abae"),clip_on=False,zorder=6)
    ax.set_title("80–250 Hz",fontsize=TYPE.condition_label,pad=7)
    identity=ax.text(.5,1.125,"Yuquan Y1",transform=ax.transAxes,ha="center",va="bottom",
                     fontsize=TYPE.identity_label,fontweight="bold")
    for i,line in enumerate(ax.lines[:12]):
        np.testing.assert_array_equal(line.get_ydata(),traces[i]+i*gap)
    return [background,ax],dict(meta, native_text=True, waveform_samples_unchanged=True,
                               signal_samples_sha256=array_hash(traces))


def draw_ce(fig, panel, arr, view):
    y=HEAT_Y[panel]; clustered=panel=="E"
    start=len(fig.axes)
    summary=s.draw_ce(fig,current.rect(fig,0,y-.6,13.9,4.8),view,"Yuquan Y1",clustered)
    heat,bar,profile,*strip=fig.axes[start:]
    heat.set_position(current.rect(fig,.85,y,8.6,HEAT_HEIGHT))
    bar.set_position(current.rect(fig,9.76,y,.14,HEAT_HEIGHT))
    profile.set_position(current.rect(fig,MID_X,y,MID_WIDTH,HEAT_HEIGHT))
    heat.collections[0].set_clim(0,25)
    bar.set_yticks([0,25]);bar.set_yticklabels(["0","25"])
    style_axis(heat,dense=True);heat.tick_params(axis="x",labelsize=TYPE.tick_label)
    style_axis(bar,colorbar=True)
    bar.set_title("Rank",fontsize=TYPE.colorbar_label,pad=7)
    for text in heat.texts:
        text.set_fontsize(TYPE.condition_label if clustered else TYPE.identity_label)
    if not clustered:
        heat.legend(handles=[Patch(facecolor="white",edgecolor="black",label="Day"),
                             Patch(facecolor="black",label="Night")],
                    loc="lower right",bbox_to_anchor=(.97,1.025),ncol=2,frameon=False,
                    fontsize=TYPE.legend,borderaxespad=0,handlelength=.8,
                    handletextpad=.4,columnspacing=.9)
        strip[0].set_position(current.rect(fig,.85,y-.19,8.6,.10))
        style_axis(strip[0])
        profile.clear()
        full_positions={int(ci):i for i,ci in enumerate(arr["channel_order"])}
        row_colors=[plt.cm.viridis(full_positions[int(ci)]/25) for ci in view["channel_order"]]
        s.old.propagation_plot._plot_rank_histogram(
            profile,arr["ranks"],arr["bools"],arr["valid_events"],view["channel_order"],arr["channel_names"],"",
            show_ylabels=False,rank_axis_n_channels=26,row_colors=row_colors,ridge_spacing=.30)
        profile.set_ylim(-.15,(18-.5)*.30)
        # Every retained histogram includes all original rank bins, including 18-25.
        for ci,container in zip(view["channel_order"],profile.containers):
            ids=arr["valid_events"]
            values=arr["ranks"][ci,ids][arr["bools"][ci,ids]]
            expected=np.histogram(values,bins=np.arange(27)-.5)[0]/len(values)
            np.testing.assert_allclose([p.get_height() for p in container.patches],expected)
            assert abs(expected.sum()-1)<1e-12
        assert max(p.get_y()+p.get_height() for p in profile.containers[-1].patches) <= profile.get_ylim()[1]
    else:
        for k,line in enumerate(profile.lines):
            ids=arr["valid_events"][arr["labels"]==k]
            means=[arr["ranks"][i,ids][arr["bools"][i,ids]].mean() for i in view["channel_order"]]
            np.testing.assert_allclose(line.get_xdata(),means)
    profile.set(xlim=(-.5,25.5),xticks=[0,12,25])
    style_axis(profile)
    profile.tick_params(axis="y",left=False,labelleft=False)
    heat.set_xticks([0,5000,10000,15000])
    group=[heat,bar]+strip
    # Left-column visible width retains the A/C/E relationship; the middle data
    # axes use their exact common x and width, independently of their labels.
    layout.fit_visual(fig,group,(.30,0,HEAT_GROUP_RIGHT,1),vertical=False)
    fig.canvas.draw()
    hp=heat.get_position()
    profile.set_position([MID_X/WIDTH,hp.y0,MID_WIDTH/WIDTH,hp.height])
    heat_y=heat.transData.transform(np.column_stack([np.zeros(18),np.arange(18)+.5]))[:,1]
    rank_y=profile.transData.transform(np.column_stack([np.zeros(18),np.arange(18)*(1 if clustered else .30)]))[:,1]
    np.testing.assert_allclose(heat_y,rank_y,atol=1e-8)
    return group,[profile],dict(summary,displayed_channels=18,original_rank_range=[0,25],
                               same_events_and_frozen_labels=True,rank_rows_aligned=True)


def align_subtitles(fig, groups):
    """Move only annotation artists; retain the accepted data-axis positions."""
    wave=groups["A"][-1]
    hfo=groups["B_hfo"][0]
    tfr=groups["B_tfr"][0]
    for ax in (wave,tfr):
        for text in list(ax.texts):
            if text.get_text()=="Yuquan Y1":text.remove()
    wave.set_title("");hfo.set_title("")
    titles=[]
    for ax,label,x,ha,color,weight in [
        (wave,"Yuquan Y1",wave.get_position().x0+wave.get_position().width/2,"center","black","bold"),
        (hfo,"HFO n = 178",hfo.get_position().x0+hfo.get_position().width/2,"center","red","normal"),
        (tfr,"Yuquan Y1",tfr.get_position().x0,"left","black","bold")]:
        titles.append(ax.text(x,SUBTITLE_BASELINE/HEIGHT,label,transform=fig.transFigure,
                             ha=ha,va="baseline",fontsize=TYPE.identity_label,color=color,fontweight=weight))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    channel_labels=Bbox.union([t.get_window_extent(renderer) for t in wave.get_yticklabels()])
    center_x=(channel_labels.x0+channel_labels.x1)/2/(fig.dpi*WIDTH)
    frequency=wave.text(center_x,wave.get_position().y1+.34/HEIGHT,"80–250 Hz",
                       transform=fig.transFigure,ha="center",va="bottom",fontsize=TYPE.annotation)
    baselines=[t.get_transform().transform(t.get_position())[1] for t in titles]
    np.testing.assert_allclose(baselines,[baselines[0]]*3,atol=1e-8)
    return dict(baseline_inches=SUBTITLE_BASELINE,labels=[t.get_text() for t in titles],
                baselines_pixels=baselines,frequency_position="above channel-label column",
                frequency_font_size_pt=TYPE.annotation)


def check_channel_label_ink(fig, view):
    """Check actual uppercase/digit glyph heights without shrinking 15 pt type.

    Matplotlib's line boxes reserve space for ascenders/descenders that these
    single-line contact labels do not use. Check the glyph ink plus 1 px margin
    at the 300 dpi export scale, and inspect the exported PNG separately.
    """
    checked=0;minimum=float("inf")
    for ax in fig.axes:
        labels=[t for t in ax.get_yticklabels() if t.get_visible()]
        if [t.get_text() for t in labels] != view["ordered_names"]:continue
        assert all(t.get_text().isascii() and t.get_text().isalnum() for t in labels)
        heights=[TextPath((0,0),t.get_text(),prop=t.get_fontproperties()).get_extents().height*300/72
                 for t in labels]
        centers=sorted(t.get_transform().transform(t.get_position())[1]*300/fig.dpi for t in labels)
        gap=min(np.diff(centers))-max(heights)
        assert gap>1.,gap
        minimum=min(minimum,float(gap));checked+=1
    assert checked==2
    return dict(heatmaps_checked=checked,unchanged_fontsize_pt=TYPE.dense_tick,
                minimum_glyph_gap_at_300dpi=minimum)


def enlarge_mi_diagram(ax):
    """Typeset the exact legacy rank example without shrinking raster text.

    All three input-column pairs, result columns and median notation come from
    the accepted TIFF crop. These are illustrative ranks, not new patient data.
    """
    ax.clear()
    ax.set_aspect("auto")
    ax.set(xlim=(0,1),ylim=(0,1))
    ax.axis("off")
    colors={1:"#440154",2:"#21918c",3:"#fde725"}
    examples=[(.025,[[3,1,2],[2,3,1]],[3,2,1],"Perm 0","Perm\nMed 0"),
              (.265,[[1,3,2],[3,2,1]],[2,3,1],"Perm 1","Perm\nMed 1"),
              (.705,[[1,2,3],[1,2,3]],[1,2,3],"Orig\nPats","Median (1,1)\n= 1")]
    for x,inputs,result,title,footer in examples:
        ax.add_patch(FancyBboxPatch((x,.32),.105,.41,boxstyle="round,pad=.008,rounding_size=.025",
                                   facecolor="#ededed",edgecolor="none"))
        for column,values in enumerate(inputs):
            for row,value in enumerate(values):
                ax.text(x+.028+column*.05,.675-row*.14,str(value),color=colors[value],
                        fontsize=TYPE.dense_tick,ha="center",va="center",weight="bold")
        for row,value in enumerate(result):
            ax.text(x+.20,.675-row*.14,str(value),color=colors[value],
                    fontsize=TYPE.dense_tick,ha="center",va="center",weight="bold")
        ax.add_patch(FancyArrowPatch((x+.114,.56),(x+.18,.56),arrowstyle="->",mutation_scale=10,
                                    color=".75",linewidth=1.1))
        for contact_x in (x+.028,x+.078):
            ax.add_patch(FancyArrowPatch((contact_x,.305),(x+.20,.305),
                         connectionstyle="arc3,rad=.40",arrowstyle="<->",mutation_scale=8,
                         color="#9fc9a2",linewidth=1.2))
        ax.text(x+.052,1.025,title,fontsize=TYPE.dense_tick,ha="center",va="top",linespacing=.9)
        ax.text(x+.11,.20,footer,fontsize=TYPE.dense_tick,ha="center",va="top",linespacing=.9,
                color=".5" if x<.7 else "#c84d4a")
    ax.text(.905,1.025,"Mean\nPat",fontsize=TYPE.dense_tick,ha="center",va="top",linespacing=.9)
    ax.text(.588,.56,"···",fontsize=TYPE.condition_label,ha="center",va="center",color=".7")


def main():
    (OUT/"source").mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
    pointer_path=s.CANON/"current_revision.json"
    pointer_before=pointer_path.read_bytes()
    base=Path(json.loads(pointer_before)["revision_root"])
    accepted=json.loads((base/"metadata.json").read_text())
    retained_hashes={str(p):s.sha(p) for root in (base,PREVIOUS,PREVIOUS_DISPLAY) for p in root.rglob("*") if p.is_file()}
    s.old.propagation_plot._apply_masked_paths()
    records=s.old._load_temporal_records();s.old._assert_masked_mi_records(records)
    record=next(r for r in records if s.public_patient_label(r["dataset"],r["subject"])=="Y1")
    arr=s.old._load_exemplar_arrays(record,10**9)
    original_hashes={k:array_hash(arr[k]) for k in ("ranks","bools","labels","valid_events","clustered_events_all")}
    previous_display_meta=json.loads((PREVIOUS_DISPLAY/"metadata.json").read_text())
    assert original_hashes==previous_display_meta["original_array_hashes"]
    view,selection=select_display(arr)
    print('Display 18; hidden:',[r['channel'] for r in selection if r['hidden']],flush=True)
    a_inputs=render_a_background(base)
    spectrum=current.load_y1_spectrum()
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":TYPE.annotation,
                         "pdf.fonttype":42,"svg.fonttype":"none","axes.unicode_minus":False})
    fig=plt.figure(figsize=(WIDTH,HEIGHT))
    groups,summary={},{}
    groups["A"],summary["A"]=draw_a(fig,a_inputs)
    prior_out=layout.OUT
    try:
        layout.OUT=OUT
        groups["B_hfo"],hfo_meta=layout.draw_hfo(fig,10.75)
    finally:
        layout.OUT=prior_out
    for i,ax in enumerate(groups["B_hfo"]):
        ax.set_position(current.rect(fig,MID_X,14.0-i*1.53,MID_WIDTH,1.10))
        style_axis(ax)
        ax.set_xticks([0,.5]);ax.set_xticklabels(["0.0","0.5"])
        if i<2: ax.tick_params(axis="x",labelbottom=False)
    groups["B_hfo"][0].set_title("HFO n = 178",fontsize=TYPE.condition_label)
    groups["B_hfo"][0].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
    groups["B_hfo"][0].yaxis.get_offset_text().set_fontsize(TYPE.tick_label)
    groups["B_hfo"][1].set_title("Raw",fontsize=TYPE.condition_label)
    groups["B_hfo"][2].set_title("Normalized",fontsize=TYPE.condition_label)
    for ax in groups["B_hfo"][1:]: ax.set_yticks([0,200])
    with np.load(PREVIOUS/"source/hfo_showcase.npz") as old,np.load(OUT/"source/hfo_showcase.npz") as new:
        for key in old.files: np.testing.assert_array_equal(old[key],new[key])

    start=len(fig.axes); ts=len(fig.texts)
    summary["B"]=current.draw_b_compact(fig,12.,10.,spectrum)
    axes=fig.axes[start:];axes[0].remove()
    for text in list(fig.texts[ts:]):text.remove()
    groups["B_tfr"]=axes[1:]
    for i,ax in enumerate(groups["B_tfr"][:3]):
        ax.set_position(current.rect(fig,15.+i*1.27,11.2,1.17,3.7))
        style_axis(ax,dense=True);ax.tick_params(axis="x",labelsize=TYPE.tick_label)
        ax.set_xticks([-40,40])
        ax.get_xticklabels()[0].set_ha("left")
        ax.get_xticklabels()[-1].set_ha("right")
    groups["B_tfr"][3].set_position(current.rect(fig,18.93,11.2,.10,3.7))
    style_axis(groups["B_tfr"][3],colorbar=True)
    groups["B_tfr"][0].text(0,1.04,"Yuquan Y1",transform=groups["B_tfr"][0].transAxes,
                           fontsize=TYPE.identity_label,fontweight="bold",va="bottom")
    summary["B"]["x_ticks_sec"]=[-.04,.04]
    summary["B"]["title_alignment"]="left edge of first spectrogram; included in right-column visible bounds"
    summary["B"].pop("title_baseline_from_panel_bottom_inches",None)
    for panel in ("C","E"):
        groups[panel+"_heat"],groups[panel+"_rank"],summary[panel]=draw_ce(fig,panel,arr,view)
    print('C/E rendered with original rank scale and enlarged labels',flush=True)
    start=len(fig.axes)
    summary["D"],_=current.draw_d_aligned(fig,15.,5.7,records,accepted["mechanism"])
    groups["D"]=fig.axes[start:]
    enlarge_mi_diagram(groups["D"][0])
    groups["D"][0].set_position(current.rect(fig,15.,8.48,3.80,1.55))
    groups["D"][-1].set_position(current.rect(fig,15.,5.95,3.80,2.34))
    style_axis(groups["D"][-1])
    for text in groups["D"][-1].texts:
        if text.get_text() in ("Yuquan","Epilepsiae"):
            text.set_y(0)
            text.set_transform(groups["D"][-1].get_xaxis_transform()+ScaledTranslation(0,-.50,fig.dpi_scale_trans))
            text.set_va("top")
    start=len(fig.axes)
    summary["F"],_=current.draw_f_aligned(fig,15.,.8,records)
    groups["F"]=fig.axes[start:]
    fax=groups["F"][0]; style_axis(fax)
    inset=fax.child_axes[0]
    inset.set_axes_locator(None)
    # Re-anchor the larger inset to its parent; numerical pairs and bars remain.
    from matplotlib.axes import _base
    inset.set_axes_locator(_base._TransformedBoundsLocator([.58,.12,.37,.40],fax.transAxes))
    inset.set_yticks([0,.8]);inset.yaxis.label.set_fontsize(TYPE.annotation)
    legend=fax.get_legend()
    handles=legend.legend_handles
    fax.legend(handles=handles,loc="upper right",frameon=True,facecolor="white",
               edgecolor=".55",framealpha=.92,fancybox=False,fontsize=TYPE.legend,
               markerscale=1.3,handlelength=.8,handletextpad=.25,labelspacing=.2,
               borderpad=.25,borderaxespad=.35)
    assert summary["D"]==accepted["panel_d"] and summary["F"]==accepted["panel_f"]
    right_boxes={"B_tfr":(14.35,10.55,19.20,15.45),"D":(14.35,5.28,19.20,10.18),"F":(14.35,.20,19.20,5.10)}
    for name,target in right_boxes.items():layout.fit_visual(fig,groups[name],target)
    subtitle_alignment=align_subtitles(fig,groups)
    letters=[]
    for letter,x,y in [("A",.06,15.80),("B",10.72,15.80),("C",.06,10.42),
                        ("D",14.18,10.42),("E",.06,5.39),("F",14.18,5.39)]:
        letters.append(fig.text(x/WIDTH,y/HEIGHT,letter,fontsize=TYPE.panel_letter,fontweight="bold",va="top"))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    extents={name:layout.bounds(fig,axes).extents.tolist() for name,axes in groups.items()}
    fig.savefig(OUT/"source/layout_preview.png",dpi=120,facecolor="white")
    data_axes=groups["B_hfo"]+groups["C_rank"]+groups["E_rank"]
    axis_bounds=[[ax.get_position().x0*WIDTH,ax.get_position().width*WIDTH] for ax in data_axes]
    np.testing.assert_allclose(axis_bounds,[[MID_X,MID_WIDTH]]*5,atol=1e-10)
    label_checks=check_channel_label_ink(fig,view)
    for group in groups.values():
        for ax in group:
            labels=[t.get_window_extent(renderer) for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
            for a,b in zip(labels,labels[1:]):assert a.x1 <= b.x0+1, (ax.get_xlabel(),a.bounds,b.bounds)
    tfr_ticks=[t.get_window_extent(renderer) for ax in groups["B_tfr"][:3] for t in ax.get_xticklabels()]
    for a,b in zip(tfr_ticks,tfr_ticks[1:]):assert a.x1+1 < b.x0, (a.bounds,b.bounds)
    for panel in ("C","E"):
        heat,bar=groups[panel+"_heat"][:2]
        header_boxes=[t.get_window_extent(renderer) for t in heat.texts]
        if heat.get_legend():header_boxes.append(heat.get_legend().get_window_extent(renderer))
        bar_box=bar.title.get_window_extent(renderer)
        assert all(not b.overlaps(bar_box) for b in header_boxes)
    diagram_boxes=[t.get_window_extent(renderer) for t in groups["D"][0].texts]
    diagram_overlaps=[(groups["D"][0].texts[i].get_text(),groups["D"][0].texts[j].get_text())
                      for i,a in enumerate(diagram_boxes) for j,b in enumerate(diagram_boxes) if j>i and a.overlaps(b)]
    assert not diagram_overlaps,diagram_overlaps
    dataset_boxes=[t.get_window_extent(renderer) for t in groups["D"][-1].texts
                   if t.get_text() in ("Yuquan","Epilepsiae")]
    tick_boxes=[t.get_window_extent(renderer) for t in groups["D"][-1].get_xticklabels()]
    assert all(not a.overlaps(b) for a in dataset_boxes for b in tick_boxes)
    ce_gap=extents["C_heat"][1]-extents["E_heat"][3]
    assert ce_gap > .02,(ce_gap,extents)
    assert ce_gap < .85
    heat_rank_gaps={panel:extents[panel+"_rank"][0]-extents[panel+"_heat"][2] for panel in ("C","E")}
    assert all(.2<gap<.4 for gap in heat_rank_gaps.values()),heat_rank_gaps
    assert all(b[0]>=-.02 and b[1]>=0 and b[2]<=WIDTH+.02 and b[3]<=HEIGHT for b in extents.values()),extents
    print('Alignment and channel labels checked; exporting',flush=True)
    fig.savefig(FIG/"fig1-complete-layout.png",dpi=300,facecolor="white")
    fig.savefig(FIG/"fig1-complete-layout.pdf",dpi=300,facecolor="white")
    # Hide panel letters before the standalone exports, retaining exact axes.
    for text in letters:text.set_visible(False)
    for panel,names in {"a":["A"],"b":["B_hfo","B_tfr"],"c":["C_heat","C_rank"],
                        "d":["D"],"e":["E_heat","E_rank"],"f":["F"]}.items():
        box=Bbox.union([Bbox.from_extents(*extents[name]) for name in names]).padded(.04)
        for suffix in ("png","pdf"):
            fig.savefig(FIG/f"fig1-panel{panel}.{suffix}",dpi=300,facecolor="white",bbox_inches=box)
    plt.close(fig)
    assert original_hashes=={k:array_hash(arr[k]) for k in original_hashes}
    assert retained_hashes=={str(p):s.sha(p) for root in (base,PREVIOUS,PREVIOUS_DISPLAY) for p in root.rglob("*") if p.is_file()}
    assert pointer_path.read_bytes()==pointer_before
    source_record=s.old.MASKED_ROOT/f"per_subject/{record['dataset']}_{record['subject']}.json"
    metadata=dict(status="PENDING_AUTHOR_VISUAL_REVIEW",patient="Y1",author_patient_choice="Y1 retained",
        producer=str(Path(__file__).resolve()),source_revision=str(base),previous_layout=str(PREVIOUS_DISPLAY),
        source_record=str(source_record),original_channel_count=26,displayed_channel_count=18,
        hidden_channels=[r["channel"] for r in selection if r["hidden"]],
        n_events=len(arr["valid_events"]),cluster_counts=[int(sum(arr["labels"]==k)) for k in (0,1)],
        display_only=True,raw_lagpat_modified=False,clusters_refit=False,ranks_recomputed=False,
        original_rank_range=[0,25],original_array_hashes=original_hashes,
        font_family="DejaVu Sans",typography_source="DENSE_MULTIPANEL_TYPOGRAPHY x 1.25",
        font_sizes_pt=TYPE.as_dict(),visible_bounds_inches=extents,
        middle_column_axis_left_and_width_inches=axis_bounds,CE_visible_gap_inches=ce_gap,
        heatmap_height_inches=HEAT_HEIGHT,heatmap_rank_visible_gaps_inches=heat_rank_gaps,
        heatmap_axis_widths_inches={panel:groups[panel+"_heat"][0].get_position().width*WIDTH for panel in ("C","E")},
        subtitle_alignment=subtitle_alignment,
        label_checks=label_checks,summaries=summary,hfo_source=hfo_meta,
        mechanism_typography=dict(source=accepted["mechanism"],native_text=True,
            original_rank_examples_preserved=True,illustrative_not_patient_data=True),
        human_visual_acceptance="PENDING",prior_versions_retained=True)
    s.write_json(OUT/"metadata.json",metadata)
    s.write_json(OUT/"validation.json",dict(status="PASS",original_data_unchanged=True,
        all_18190_events_retained=True,frozen_labels_retained=True,full_26_channel_rank_scale_retained=True,
        rank_histograms_include_all_26_bins=True,template_means_unchanged_for_retained_channels=True,
        rank_heatmap_rows_aligned=True,mid_column_axes_aligned=True,tick_labels_do_not_overlap=True,
        adjacent_tfr_ticks_separated=True,heatmap_headers_clear_of_colorbar_titles=True,
        A_B_subtitles_share_baseline=True,heatmaps_wider_and_flatter=True,
        heatmap_rank_gap_reduced=True,channel_label_fontsize_preserved=True,
        diagram_text_does_not_overlap=True,dataset_labels_clear_of_ticks=True,
        cohort_statistics_unchanged=True,previous_outputs_unchanged=True,human_visual_acceptance="PENDING"))
    for path in (Path(__file__),Path(chain.__file__),Path(s.old.propagation_plot.__file__)):
        shutil.copy2(path,OUT/"source"/path.name)
    lines=["# Figure 1：Y1热图加长压扁与A/B副标题对齐", "",
           "作者继续采用Y1；本次只改变可视化，原始lagPat、全部18,190事件、TA/TB标签（13,160/5,030）和40人统计均保留。原版保留，新版待作者目视检查。", ""]
    for name in ["complete-layout","panela","panelb","panelc","paneld","panele","panelf"]:
        text=("C/E沿用18通道显示及原26通道定义的0–25 rank标度，热图加长并降低高度，扩大宽高比；热图占用与rank之间原有空白。中列HFO的三个坐标轴仍与C/E rank坐标轴严格同左边界、同宽；A的Yuquan Y1、B的HFO n = 178和Yuquan Y1共用一条副标题基线，80–250 Hz移至A通道标签正上方。"
              if name=="complete-layout" else
              "只显示18个通道，所有事件和原rank数值保持。热图与右侧rank图使用同一显示顺序；直方图保留0–25全部bin，模板带仍为参与事件均值±总体标准差。" if name in ("panelc","panele") else
              "沿用已接受数据、事件及图形结构，放大坐标标签、刻度和图例。A的脑朝向、电极造型与引线保持，波形按同一已滤波样本原样重绘；D/F仍为原40人统计。")
        lines += [f"### fig1-{name}.png / .pdf","",text,"","**关注点**：字体可读性、行间距和坐标轴对齐；显示通道删减不改变原始数据或科学统计。",""]
    (FIG/"README.md").write_text("\n".join(lines))
    for panel in ("d","f"):
        with Image.open(PREVIOUS_DISPLAY/"figures"/f"fig1-panel{panel}.png") as previous_image, Image.open(FIG/f"fig1-panel{panel}.png") as current_image:
            np.testing.assert_array_equal(np.asarray(previous_image),np.asarray(current_image))
    validation=json.loads((OUT/"validation.json").read_text())
    validation["D_F_standalone_png_pixels_unchanged"]=True
    s.write_json(OUT/"validation.json",validation)
    print('DONE',OUT,flush=True)


if __name__=="__main__":main()
