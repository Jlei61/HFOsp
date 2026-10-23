#!/usr/bin/env python3
"""Move the projection arrow, enlarge roses, and match axis fonts to B."""
from __future__ import annotations
import argparse
import copy
import json
import re
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pymupdf as pdf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.paper_figures import build_fig2_aligned_columns as renderer
from scripts.paper_figures import plot_fig2b_rank_reconstruction_holdout as roses
from scripts.paper_figures.build_fig2_split_holdout_panels import raster,sha,write_json
from scripts.paper_figures.build_fig2_bottom_alignment import text_span

SOURCE=ROOT/'results/paper-ready-figure/fig2/candidates/aligned_bottom_row_20261023'
OUT=SOURCE.parent/'projection_typography_20261023'
AXIS=13.6
TICK=11.05
TITLE=14.
D_SCALE=530./710.
AXES_TOP=28.*D_SCALE
AXES_BOTTOM=310.*D_SCALE
XLABEL_BASELINE=AXES_BOTTOM+34.140625
LOWER_HEIGHT=AXES_BOTTOM+48.
REGULAR='/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'
BOLD='/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf'


def spans(page):
    return [dict(s,direction=line['dir'])
            for block in page.get_text('dict',flags=pdf.TEXTFLAGS_TEXT)['blocks']
            for line in block.get('lines',[]) for s in line['spans']]


def remove_text(page,items):
    # Remove glyphs only. Keep images, scientific paths, and clipping intact.
    for s in items:
        rect=pdf.Rect(s['bbox'])
        if s['direction']==(1.,0.):
            mid=(rect.y0+rect.y1)/2;rect.y0=mid-.15;rect.y1=mid+.15
        else:
            mid=(rect.x0+rect.x1)/2;rect.x0=mid-.15;rect.x1=mid+.15
        page.add_redact_annot(rect,fill=None)
    page.apply_redactions(images=0,graphics=0,text=0)


def remove_form_text(document):
    """Remove text objects from native content streams, retaining all graphics.

    A contains nested, repeatedly clipped MRI forms. Redacting their text by
    geometry repeatedly traverses the same large graphics trees. Its visible
    text is recorded first, then reinserted after this content-only edit.
    """
    content_xrefs={xref for page in document for xref in page.get_contents()}
    count=0
    for xref in range(1,document.xref_length()):
        if document.xref_get_key(xref,'Subtype')!=('name','/Form') and xref not in content_xrefs:
            continue
        if not document.xref_is_stream(xref):continue
        data=document.xref_stream(xref)
        # PDF name delimiters are legal directly after BT (e.g. BT/F1).
        cleaned,n=re.subn(rb'(?<!\S)BT(?=$|[\s/]).*?(?<!\S)ET(?=$|[\s/])',b'',data,flags=re.S)
        if n:document.update_stream(xref,cleaned);count+=n
    return count


def writer(page):
    fonts={}
    for key,path in [('FigureText',REGULAR),('FigureBold',BOLD)]:
        page.insert_font(fontname=key,fontfile=path);fonts[key]=pdf.Font(fontfile=path)
    def write(text,x,y,size,*,align='center',bold=False,color=(0,0,0),rotate=0):
        key='FigureBold' if bold else 'FigureText'
        length=fonts[key].text_length(text,fontsize=size)
        if rotate:y+=length/2
        else:x-=length*{'left':0,'center':.5,'right':1}[align]
        page.insert_text((x,y),text,fontname=key,fontsize=size,color=color,rotate=rotate)
    return write


def rgb(value):
    return tuple(((value>>shift)&255)/255 for shift in (16,8,0))


def export(document,path,dpi,preview):
    document.save(path,garbage=4,deflate=True)
    document.close()
    raster(path,dpi,preview)
    image=pdf.Pixmap(str(path.with_suffix('.png')))
    image.set_dpi(dpi,dpi);image.save(path.with_suffix('.png'))


def build_a(source,figures,dpi):
    doc=pdf.open(source/'figures/fig2-panela.pdf');page=doc[0]
    items=spans(page)
    selected=[s for s in items if s['size']>=9. and
              not s['text'].startswith(('ICL','SCL'))]
    removed_objects=remove_form_text(doc)
    # Erase the old arrow in its empty inter-row gap, without touching axes.
    page.draw_rect(pdf.Rect(327,154,363,164),color=None,fill=(1,1,1),overlay=True)
    write=writer(page);changed=[]
    for s in items:
        text=s['text'];box=pdf.Rect(s['bbox']);x,y=s['origin'];align='center'
        bold='Bold' in s['font'];size=TICK
        if s not in selected:
            if s['direction']==(0.,-1.):
                write(text,x,(box.y0+box.y1)/2,s['size'],bold=bold,color=rgb(s['color']),rotate=90)
            else:write(text,x,y,s['size'],align='left',bold=bold,color=rgb(s['color']))
            continue
        if text=='Early Projection':text='2D projection';size=TICK
        elif text in ('Time (ms)','shared TA axis (mm)','Normalized magnitude','y (mm)','ranks'):size=AXIS
        elif text.startswith(('TA ','TB ')):size=TITLE
        if s['direction']==(0.,-1.):
            write(text,x,(box.y0+box.y1)/2,size,bold=bold,color=rgb(s['color']),rotate=90)
        else:
            # Colorbar numbers retain their left anchor; field y ticks their right anchor.
            if x>490 or (150<x<178 and text in ('0','1')) or text=='ranks':
                align='left'
            elif 345<x<367 and text in ('0','20'):
                x=box.x1;align='right'
            else:x=(box.x0+box.x1)/2
            write(text,x,y,size,align=align,bold=bold,color=rgb(s['color']))
        changed.append(dict(old=s['text'],new=text,size_pt=size))
    # Centered inside the dashed projection region, above its title and frame.
    arrow=[(260.,164.),(284.,164.)]
    page.draw_line(*arrow,color=(.42,.49,.51),width=1.3)
    page.draw_polyline([(284.,160.5),(291.,164.),(284.,167.5)],
                       closePath=True,color=(.42,.49,.51),fill=(.42,.49,.51),width=.5)
    export(doc,figures/'fig2-panela.pdf',dpi,1400)
    return dict(source_sha256=sha(source/'figures/fig2-panela.pdf'),
                edited_text=changed,arrow_bounds_pt=[260,160.5,291,167.5],
                projection_frame_top_pt=266*320/430,
                text_objects_replaced=removed_objects,
                dense_contact_labels_preserved_for_row_spacing=True,
                scientific_images_and_paths_preserved=True)


def build_d(source,figures,dpi):
    old=pdf.open(source/'figures/fig2-paneld.pdf');items=spans(old[0]);remove_text(old[0],items)
    doc=pdf.open();page=doc.new_page(width=530,height=LOWER_HEIGHT)
    page.show_pdf_page(pdf.Rect(0,0,530,350*D_SCALE),old,0)
    # Reserve enough room for full-size early/late labels at the right edge.
    bar_region=pdf.Rect(499.7,20.4,507.,232.)
    page.draw_rect(bar_region,color=None,fill=(1,1,1),overlay=True)
    page.show_pdf_page(bar_region+(-4.,0.,-4.,0.),old,0,clip=bar_region*(1/D_SCALE))
    old.close()
    write=writer(page);edits=[]
    for s in items:
        text=s['text'];box=pdf.Rect(s['bbox'])*D_SCALE
        x,y=np.asarray(s['origin'])*D_SCALE
        bold='Bold' in s['font'];color=rgb(s['color']);size=TICK;align='center'
        if text=='Shared TA axis (mm)':
            size=AXIS;y=XLABEL_BASELINE;x=(box.x0+box.x1)/2
        elif text in ('y (mm)','TA TRAJ.','TB TRAJ.'):
            size=AXIS
            if text in ('TA TRAJ.','TB TRAJ.'):x=14.
            write(text,x,(box.y0+box.y1)/2,size,bold=bold,color=color,rotate=90)
            edits.append(dict(text=text,size_pt=size));continue
        elif text in ('E1','E5','Y9','Y14'):
            size=TITLE;x=(box.x0+box.x1)/2
        elif text=='rank':
            size=AXIS;x=(box.x0+box.x1)/2-4.
        elif text in ('early','late'):
            x-=4.
            y=(box.y0+box.y1)/2+.347*TICK;align='left'
        elif abs(s['origin'][1]-161.703125)<.01 or abs(s['origin'][1]-323.703125)<.01:
            x=(box.x0+box.x1)/2
            bottom=(148. if s['origin'][1]<200 else 310.)*D_SCALE
            y=bottom+14.
        else:
            x=box.x1;align='right';y=(box.y0+box.y1)/2+.347*TICK
        write(text,x,y,size,align=align,bold=bold,color=color)
        edits.append(dict(text=text,size_pt=size))
    export(doc,figures/'fig2-paneld.pdf',dpi,1400)
    return dict(source_sha256=sha(source/'figures/fig2-paneld.pdf'),edited_text=edits,
                data_scale=D_SCALE,scientific_images_and_paths_preserved=True)


def build_c(figures,out,dpi):
    payloads=[roses.rose_data(sid) for sid in roses.EXAMPLES]
    rmax=float(np.ceil(max(roses.prior._histogram_proportions(d['all_event_angles'],d['edges']).max()
                          for d in payloads)/.05)*.05)
    width,height,diameter=180.,320.,124.
    records=[]
    with plt.rc_context({'font.family':'Liberation Sans','pdf.fonttype':42,'axes.linewidth':.8}):
        fig=plt.figure(figsize=(width/72,height/72));axes=[]
        for data,top in zip(payloads,(17.,177.)):
            ax=fig.add_axes([28/width,1-(top+diameter)/height,diameter/width,diameter/height],projection='polar')
            axes.append(ax)
            roses.prior.draw_combined_probability_rose(ax,data,edges=data['edges'],rmax=rmax,row_label=roses.label(data['subject_id']))
            for line in list(ax.lines):
                if line.is_dashed():line.remove()
            ax.tick_params(labelsize=TICK,pad=1);ax.set_rlabel_position(132)
            for text in ax.texts:text.set_fontsize(15.3)
            bins=roses.prior._histogram_proportions(data['all_event_angles'],data['edges'])
            np.testing.assert_allclose([p.get_height() for p in ax.patches],bins,atol=1e-14)
            assert len(ax.lines)==2 and all(not line.is_dashed() for line in ax.lines)
            records.append(dict(subject=roses.label(data['subject_id']),histogram_probability=bins,
                                edges=data['edges'],rays=[float(line.get_xdata()[0]) for line in ax.lines]))
        handles=[Patch(facecolor='#DDDDDD',edgecolor='#777777',label='Event directions'),
                 Line2D([],[],color='#333333',lw=1.8,label='Spatial TRAJ.')]
        legend=axes[0].legend(handles=handles,loc='center',bbox_to_anchor=(.5,.23),
            fontsize=8.5,handlelength=1.3,labelspacing=.35,frameon=True,facecolor='white',
            edgecolor='none',framealpha=.95,borderpad=.25,handletextpad=.55)
        fits=False
        for legend_y in np.arange(.23,.341,.01):
            legend.set_bbox_to_anchor((.5,legend_y))
            fig.canvas.draw();box=legend.get_window_extent(fig.canvas.get_renderer())
            corners=axes[0].transAxes.inverted().transform(box.corners())
            fits=np.all(np.linalg.norm(corners-.5,axis=1)<.5) and not any(
                p.get_path().transformed(p.get_transform()).intersects_bbox(box,filled=True)
                for p in axes[0].patches)
            if fits:break
        assert fits,'Rose legend must fit without covering histogram bars'
        # Transparent outer margins allow the larger circles to use B's blank margin.
        fig.savefig(figures/'fig2-panelc.pdf',transparent=True);plt.close(fig)
    raster(figures/'fig2-panelc.pdf',dpi,1000)
    write_json(out/'rose_values.json',renderer.plain(records))
    return dict(size_pt=[width,height],diameter_pt=diameter,circle_gap_pt=36.,
                source_renderer=str(Path(roses.__file__)),transparent_outer_margin=True,
                same_event_histograms_and_template_rays=True,legend_inside_upper_rose=True)


def build(source,out,dpi):
    figures=out/'figures';figures.mkdir(parents=True,exist_ok=True)
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    old=json.loads((source/'figure2_layout_metadata.json').read_text())
    frozen={}
    for path in (source/'figures').glob('fig2-panelb*'):
        shutil.copy2(path,figures/path.name);frozen[path.name]=sha(path)
    d=build_d(source,figures,dpi);print('D typography ready',flush=True)
    reference=dict(axes_top_pt=AXES_TOP,axes_bottom_pt=AXES_BOTTOM,
        xlabel_baseline_pt=XLABEL_BASELINE,xlabel_fontsize=AXIS,xtick_fontsize=TICK,
        xtick_length_pt=3.,xtick_pad_pt=3.)
    e=renderer.build_e(figures,out,dpi,width=404.7,height=LOWER_HEIGHT,type_scale=.85,axis_alignment=reference)
    for key in ('distribution','display_raw_cohort_median_null','significance_display','patient_order'):
        assert e[key]==old['panel_e'][key],key
    print('E typography ready',flush=True)
    c=build_c(figures,out,dpi);print('C enlarged roses ready',flush=True)
    a=build_a(source,figures,dpi);print('A projection and typography ready',flush=True)
    layout=copy.deepcopy(old['complete_layout'])
    layout['c']=dict(source_clip_pt=[0,0,180,320],placed_rect_pt=[796.7,40,976.7,360],scale=1.)
    layout['d']=dict(source_clip_pt=[0,0,530,LOWER_HEIGHT],placed_rect_pt=[24,386,554,386+LOWER_HEIGHT],scale=1.)
    layout['e']=dict(source_clip_pt=[0,0,404.7,LOWER_HEIGHT],placed_rect_pt=[572,386,976.7,386+LOWER_HEIGHT],scale=1.)
    canvas=[1000.7,386+LOWER_HEIGHT+16]
    doc=pdf.open();page=doc.new_page(width=canvas[0],height=canvas[1])
    for letter,p in layout.items():
        with pdf.open(figures/f'fig2-panel{letter}.pdf') as part:
            page.show_pdf_page(pdf.Rect(p['placed_rect_pt']),part,0,clip=pdf.Rect(p['source_clip_pt']))
    for label,x,y in old['panel_labels']:page.insert_text((x,y),label,fontsize=23,fontname='hebo')
    export(doc,figures/'fig2-complete-layout.pdf',dpi,1900)
    meta=copy.deepcopy(old)
    meta.update(schema_version='fig2_projection_typography_v1',producer=str(Path(__file__)),
        source_candidate=str(source),complete_layout=layout,canvas_pt=canvas,frozen_panel_exports=frozen,
        panel_a_typography=a,panel_c_display=c,panel_d_typography=d,panel_e=e,
        typography=dict(reference='panel B',axis_label_pt=AXIS,numeric_ticks_pt=TICK,
                        dense_A_contact_labels_pt=8.93028),dpi=dpi,
        agent_visual_review='PENDING',status='PENDING_AUTHOR_VISUAL_REVIEW')
    meta.pop('agent_visual_review_notes',None)
    meta['alignment'].update(D_E_bottom_pt=386+LOWER_HEIGHT,D_E_height_pt=LOWER_HEIGHT,
                            D_E_xlabel_baseline_pt=386+XLABEL_BASELINE)
    meta['compaction'].update(C_circle_diameter_pt=124,C_circle_gap_pt=36,
                             C_native_whitespace_clip_pt=[0,0,180,320])
    meta['panel_c']['rose_layout']=dict(bottoms_pt=[179,19],diameter_pt=124,circle_gap_pt=36)
    write_json(out/'figure2_layout_metadata.json',renderer.plain(meta))
    for path in (Path(__file__),Path(renderer.__file__),Path(roses.__file__)):
        shutil.copy2(path,snapshot/path.name)
    shutil.copy2(source/'figure2_layout_metadata.json',snapshot/'previous_layout_metadata.json')
    (figures/'README.md').write_text(
        '# Figure 2：投影箭头、放大rose及统一字号\n\n状态：PENDING_AUTHOR_VISUAL_REVIEW。'
        '以B的显示字号为准，坐标标签13.6 pt、数字刻度11.05 pt；A密集通道名称保留原行距对应字号。'
        'A改为2D projection，箭头移至虚线投影区域内；C直径从108 pt增至124 pt，图例保留圆内。'
        'D/E绘图区高度和xlabel基线继续一致。\n\n'
        '### fig2-complete-layout.pdf / fig2-complete-layout.png\n完整A–E保持收窄宽度，D/E底部标签增加必要间距；PDF与PNG来自同一状态。\n**关注点**：字体一致性、放大rose与箭头位置，待作者目视验收。\n\n'
        '### fig2-panela.pdf / fig2-panela.png\n仅更改文字大小、2D projection名称与箭头位置，MRI、场和序列原生图层保留。\n**关注点**：箭头在虚线区域内，不触碰实线或文字。\n\n'
        '### fig2-panelb.pdf / fig2-panelb.png\nB原始导出保持，作为字号参照。\n**关注点**：留出误差、null及星号不变。\n\n'
        '### fig2-panelc.pdf / fig2-panelc.png\n两个rose直径124 pt，圆周间距36 pt，利用上排下方空白；透明外边缘使用B右侧空白。灰柱及模板直线仍用同一留出折原始读出。\n**关注点**：灰柱、两条实线与圆内图例均保留。\n\n'
        '### fig2-paneld.pdf / fig2-paneld.png\n仅重排文字以匹配B，八个空间场及轨迹维持原显示大小。\n**关注点**：放大的刻度不压住空间场，xlabel与E对齐。\n\n'
        '### fig2-panele.pdf / fig2-panele.png\n坐标标签和刻度匹配B，重新布置底部标签；原r、IQR、null与星号逐项一致。\n**关注点**：与D等高、xlabel对齐、患者名称无重叠。\n')
    return meta


def validate(out):
    meta=json.loads((out/'figure2_layout_metadata.json').read_text())
    for name,digest in meta['frozen_panel_exports'].items():assert sha(out/'figures'/name)==digest
    labels={}
    for letter,label in [('b','Held-out error (°)'),('d','Shared TA axis (mm)'),('e','Cohort median r vs spatial null')]:
        with pdf.open(out/f'figures/fig2-panel{letter}.pdf') as doc:
            s=text_span(doc[0],label);assert abs(s['size']-AXIS)<.01
            labels[letter]=s['origin'][1]
            for word in doc[0].get_text('words'):
                assert doc[0].rect.contains(pdf.Rect(word[:4])),(letter,word[4])
    assert abs(labels['d']-labels['e'])<.03
    for letter in 'ac':
        with pdf.open(out/f'figures/fig2-panel{letter}.pdf') as doc:
            text=doc[0].get_text()
            if letter=='a':assert '2D projection' in text and 'Early Projection' not in text
    report=dict(status='PASS',B_unchanged=True,E_statistics_unchanged=True,
        C_histograms_and_rays_preserved=True,axis_label_font_pt=AXIS,numeric_tick_font_pt=TICK,
        D_E_xlabel_baseline_difference_pt=labels['d']-labels['e'],
        human_visual_acceptance='PENDING_AUTHOR_VISUAL_REVIEW')
    write_json(out/'validation.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--dpi',type=int,default=600)
    args=parser.parse_args();build(args.source,args.output,args.dpi)
    print(json.dumps(validate(args.output),indent=2),flush=True)
