#!/usr/bin/env python3
"""Reflow the author's A–I Fig4 at its complete-figure display size.

Reuse the frozen renderers and observations. This is a typography candidate,
not a new fit, event selection, or promotion of the old canonical package.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
WT = ROOT / '.worktrees/topic4-continuous-core-state-r1'
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np
from PIL import Image

from scripts.paper_figures import build_fig4_geometry_patient_comparison as compare
from scripts.paper_figures import build_fig4_panel_a_combined as combined
from scripts.paper_figures import plot_topic4_cohort_loss_improvement as cohort

# The two plotting sources are the frozen copies packaged with the candidate.
# Their non-plotting dependencies still live in the original worktree.
import src, scripts, scripts.paper_figures
src.__path__.append(str(WT / 'src'))
scripts.__path__.append(str(WT / 'scripts'))
scripts.paper_figures.__path__.append(str(WT / 'scripts/paper_figures'))
sys.path.append(str(WT / 'src/snn_engine'))

OLD = compare.OLD
PREVIOUS = compare.OUT
EXPANDED = ROOT / 'results/paper-ready-figure/fig4/candidates/geometry_prior_expanded_response_20260920'
OUT = ROOT / 'results/paper-ready-figure/fig4/candidates/complete_readability_20260923'
FIG = OUT / 'figures'
SOURCE = OUT / 'source'
LOWER_ROW_EXTRA_GAP_MM = 6.5
W, H = 285., 258. + LOWER_ROW_EXTRA_GAP_MM
TOP_SHIFT = 23.
ROW_HEIGHT = 40.
MID_ROWS = {'D':132., 'E':73.}
FIELD_SIDE = 19.5
FIELD_STEP = ROW_HEIGHT - FIELD_SIDE
LOWER_Y = 13.
MM = 1 / 25.4
RED, BLUE = compare.RED, compare.BLUE
FONT = dict(label=10., tick=9., dense=7.5, legend=8.8, title=10., letter=15.)
MIDDLE_LABEL_PT = 12.
ITERATION_LABEL = 'Iteration order'
COHORT_XLABEL = 'Epochs'
ERROR_COLORS = dict(rank_error='#3D5C6F', within_rod_order_error='#E47159',
                   participation_error='#6B7546')
GROUPS = {}
SOURCES = {}
CHECKS = {}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_current_version(record):
    """Keep the author-designated current Fig4 discoverable after rebuilding."""
    path=ROOT/'results/paper-ready-figure/fig4/current_version.json'
    if path.exists() and read(path).get('producer') != record['producer']:
        return  # Reproducing a historical package must not replace a newer designation.
    pointer=dict(
        schema_version=1,figure='Fig4',panel_letters=list('ABCDEFGHI'),
        status='CURRENT_AUTHOR_DESIGNATED',author_designated_on='2026-09-28',
        asset_id='patient_geometry_prior_snn_complete_a_i',
        package=str(OUT.relative_to(ROOT)),producer=record['producer'],
        documentation='docs/current_figure4.md',
        registry=str((OUT/'figure4_candidate_registry.json').relative_to(ROOT)),
        visual_qa=str((OUT/'visual_qa.json').relative_to(ROOT)),
        complete_layout={ext:str((FIG/f'fig4-complete-layout.{ext}').relative_to(ROOT))
                         for ext in ['png','pdf','svg']},
        preview=str((FIG/'fig4-complete-layout-preview.png').relative_to(ROOT)),
        outputs_sha256={str((OUT/path).relative_to(ROOT)):digest
                        for path,digest in record['outputs'].items()},
        palette={'B_iterations':record['checks']['B_iteration_palette'],
                 'I':record['checks']['I_palette']},
        legacy_package='results/paper-ready-figure/fig4/figures',
        legacy_status='HISTORICAL_SOURCE_ONLY',
        scientific_claims='Unchanged; current-version designation does not expand scientific acceptance.')
    path=ROOT/'results/paper-ready-figure/fig4/current_version.json'
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(pointer,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def freeze(path):
    path = Path(path)
    SOURCES[str(path)] = sha(path)
    return path


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, freeze(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = module('scripts.paper_figures.plot_topic4_compact_english_acceptance',
              OLD / 'source/plot_topic4_compact_english_acceptance.py')
plot = module('scripts.paper_figures.plot_topic4_shared_plane_position_response',
              OLD / 'source/plot_topic4_shared_plane_position_response.py')
base.SOURCE = OLD / 'source'
base.OUT = SOURCE
base.HEIGHT = 190.


def rect(x, y, w, h):
    return [x / W, y / H, w / W, h / H]


def pos(ax, x, y, w, h):
    ax.set_position(rect(x, y, w, h))


def label(fig, x, y, text, **kwargs):
    return fig.text(x/W, y/H, text, fontsize=kwargs.pop('fontsize', FONT['label']), **kwargs)


@contextmanager
def group(fig, letter):
    old_axes, old_text, old_artists = set(fig.axes), set(fig.texts), set(fig.artists)
    yield
    GROUPS[letter] = dict(axes=[a for a in fig.axes if a not in old_axes],
                         texts=[t for t in fig.texts if t not in old_text],
                         artists=[a for a in fig.artists if a not in old_artists])


def axis_type(ax, dense=False):
    ax.tick_params(labelsize=FONT['dense'] if dense else FONT['tick'], length=2.4, pad=2)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.label.set_fontsize(FONT['label'])
        axis.labelpad = 3
    if dense:
        ax.tick_params(axis='x', labelsize=FONT['tick'])
    for t in ax.texts:
        t.set_fontsize(FONT['tick'])


def panel_a(fig):
    # Preserve the local circuit at precisely its previous physical scale.
    # Only the external title, coordinate frame and neuron legend are enlarged.
    archived=ROOT/'results/paper-ready-figure/archive/2026-09-29_pre_template_recovery_fig4/fig4/figures'
    # The original raster remains an input after the A-I package is published.
    source_dir=archived if archived.exists() else combined.OUTPUT_PNG.parent
    source=freeze(source_dir/combined.OUTPUT_PNG.name)
    with Image.open(source) as im:
        original=np.asarray(im.convert('RGB'))
    layout=read(freeze(source_dir/combined.OUTPUT_METADATA.name))['layout']
    scale=126/6000
    def image_box(box):
        x0,y0,x1,y1=box
        return (4+x0*scale,165+TOP_SHIFT+(3000-y1)*scale,
                (x1-x0)*scale,(y1-y0)*scale)
    frame=layout['a_dashed_inset_box_px']
    # The source stroke straddles the nominal border by five pixels. Retain
    # the entire stroke, then overlay a vector outline for full-plate clarity.
    left=[frame[0]-12,frame[1]-12,frame[2]+12,frame[3]+12]
    la=fig.add_axes(rect(*image_box(left)))
    # Show only the untouched interior raster. Excluding its old border avoids
    # superimposing two dash patterns beneath the clean vector frame.
    inner=[frame[0]+6,frame[1]+6,frame[2]-6,frame[3]-6]
    circuit=original[inner[1]:inner[3],inner[0]:inner[2]]
    la.imshow(circuit,extent=(inner[0],inner[2],inner[3],inner[1]),interpolation='lanczos')
    la.set(xlim=(left[0],left[2]),ylim=(left[3],left[1]));la.set_axis_off()
    fw,fh=left[2]-left[0],left[3]-left[1]
    la.add_patch(matplotlib.patches.Rectangle(
        ((frame[0]-left[0])/fw,(left[3]-frame[3])/fh),
        (frame[2]-frame[0])/fw,(frame[3]-frame[1])/fh,
        transform=la.transAxes,fill=False,edgecolor=combined.CALLOUT_COLOR,
        linewidth=.65,linestyle=(0,(2,1.4)),clip_on=False,zorder=5))
    lx,ly,lw,lh=image_box(left)
    label(fig,lx+lw/2,ly+lh+2,'Local E/I circuit',ha='center',va='bottom',
          fontsize=11,weight='bold')
    right=layout['b_placed_box_px']
    ax=fig.add_axes(rect(*image_box(right)))
    ax.imshow(original[right[1]:right[3],right[0]:right[2]],
              extent=(-10,10,-10,10),interpolation='lanczos')
    ax.set(xlabel='x (mm)',ylabel='y (mm)',xticks=[-10,0,10],yticks=[-10,0,10])
    axis_type(ax)
    for spine in ax.spines.values():spine.set_visible(True)
    ax.legend(handles=[Line2D([],[],marker=m,color=c,ls='',ms=3,label=s)
                       for m,c,s in [('^','#ef6868','E neuron'),('o','#72aadb','I neuron')]],
              loc='upper right',fontsize=8.8,frameon=True,framealpha=1,
              facecolor='white',edgecolor='#888888',handlelength=.7,
              handletextpad=.3,labelspacing=.12,borderpad=.25,borderaxespad=.25)
    zoom=layout['representative_callout_canvas_box_px']
    # Connect the preserved frames in figure coordinates, behind the axes.
    for start,end in [((frame[2],frame[1]),(zoom[0],zoom[1])),
                      ((frame[2],frame[3]),(zoom[0],zoom[3]))]:
        points=[(4+p[0]*scale,165+TOP_SHIFT+(3000-p[1])*scale) for p in [start,end]]
        fig.add_artist(Line2D([p[0]/W for p in points],[p[1]/H for p in points],
                            transform=fig.transFigure,color='#4a4a4a',lw=.6,ls='--',zorder=-1))
    CHECKS['A_local_circuit_source_pixels_unchanged']=True
    CHECKS['A_local_circuit_mm_per_source_pixel']=scale
    CHECKS['A_external_typography_pt']=dict(title=11,labels=10,ticks=9,legend=8.8)
    CHECKS['A_original_png_sha256']=sha(source)
    CHECKS['A_circuit_frame_vector_stroke_pt']=.65
    CHECKS['A_circuit_frame_source_crop_padding_px']=12


def panel_b(fig, rows, cases, *, profile_only=False):
    purple_gradient=matplotlib.colors.LinearSegmentedColormap.from_list(
        'reference_purples',['#B1A9DA','#7367BE','#483D8B'])
    stage_colors=purple_gradient(np.linspace(0,1,16))
    stage_cmap=matplotlib.colors.ListedColormap(stage_colors,name='iteration_purples')
    with patch.object(base,'COLORS',stage_colors):p=base.profile(fig,rows,[])
    pos(p,148,178+TOP_SHIFT,42,45)
    axis_type(p); p.set_yticks([0,.2,.4]); p.set_ylim(0,.5)
    p.set_ylabel('Error')
    p.set_xticklabels(['Mean rank\n'+r'$\varepsilon_r$',
                       'Order\n'+r'$\varepsilon_o$',
                       'Participation\n'+r'$\varepsilon_p$'])
    # Explicit coincident axes-coordinate origins eliminate the corner gap.
    for name in ['left','bottom']:
        p.spines[name].set_position(('axes',0)); p.spines[name].set_bounds(None,None)
    p.spines['bottom'].set_bounds(*p.get_xlim())
    p.spines['left'].set_bounds(*p.get_ylim())
    for t,key in zip(p.get_xticklabels(),base.KEYS):t.set_color(ERROR_COLORS[key])
    for c in cases:
        p.annotate(c['panel'],(2,c['participation_error']),xytext=(4,0),textcoords='offset points',fontsize=9,va='center')
    if profile_only:
        bar=fig.add_axes(rect(196,178+TOP_SHIFT,1.6,45))
        fig.colorbar(plt.cm.ScalarMappable(norm=base.NORM,cmap=stage_cmap),
                     cax=bar,orientation='vertical',ticks=[1,5,10,16])
        axis_type(bar);bar.set_ylabel(ITERATION_LABEL,fontsize=FONT['label'],labelpad=3)
        CHECKS['B_workpoints']=len(rows)
        CHECKS['B_stages']=sorted({r['stage'] for r in rows})
        CHECKS['B_profile_only']=True
        CHECKS['B_error_label_colors']=ERROR_COLORS.copy()
        for line,r in zip(p.lines,sorted(rows,key=lambda r:(r['stage'],r['candidate']))):
            np.testing.assert_allclose(line.get_color(),stage_colors[r['stage']-1])
        np.testing.assert_allclose(bar._colorbar.cmap.colors,stage_colors)
        CHECKS['B_iteration_palette']='Reference purple gradient #B1A9DA -> #7367BE -> #483D8B across the same 16 stages'
        return p
    with patch.object(base,'COLORS',stage_colors),patch.object(base,'CMAP',stage_cmap):
        tr=base.trajectory(fig,rows,[])
    bar=fig.axes[-1]
    pos(tr,220,175+TOP_SHIFT,51,51); pos(bar,196,178+TOP_SHIFT,1.6,45)
    axis_type(bar); bar.set_ylabel(ITERATION_LABEL,fontsize=10,labelpad=3)
    tr.tick_params(labelsize=8.5,pad=0)
    for a in [tr.xaxis,tr.yaxis,tr.zaxis]: a.set_label_text('')
    tr.set_xlabel(r'Mean rank error $\mathbf{\varepsilon_r}$',fontsize=9,
                  color=ERROR_COLORS['rank_error'],labelpad=0,weight='bold')
    tr.xaxis.set_rotate_label(True)
    tr.set_ylabel(r'Order error $\mathbf{\varepsilon_o}$',fontsize=8.5,
                  color=ERROR_COLORS['within_rod_order_error'],labelpad=0,weight='bold')
    tr.yaxis.set_rotate_label(True)
    error_labels=[
        tr.xaxis.label,
        tr.yaxis.label,
        label(fig,281,201+TOP_SHIFT,r'Participation error $\mathbf{\varepsilon_p}$',fontsize=9,ha='center',va='center',rotation=90,color=ERROR_COLORS['participation_error'],weight='bold')]
    for t,three_d,key in zip(p.get_xticklabels(),error_labels,base.KEYS):
        assert t.get_color()==three_d.get_color()==ERROR_COLORS[key]
    CHECKS['B_error_label_colors']=ERROR_COLORS.copy()
    CHECKS['B_profile_and_3D_label_colors_match']=True
    assert all(t.get_fontweight()=='bold' for t in error_labels)
    CHECKS['B_3D_error_axis_labels_bold']=True
    ordered_rows=sorted(rows,key=lambda r:(r['stage'],r['candidate']))
    for line,r in zip(p.lines,ordered_rows):
        np.testing.assert_allclose(line.get_color(),stage_colors[r['stage']-1])
    np.testing.assert_allclose(tr.collections[0].cmap.colors,stage_colors)
    np.testing.assert_allclose(bar._colorbar.cmap.colors,stage_colors)
    CHECKS['B_iteration_palette']='Reference purple gradient #B1A9DA -> #7367BE -> #483D8B across the same 16 stages'
    CHECKS['B_profile_scatter_colorbar_palette_match']=True
    CHECKS['BC_rank_error_name']='Mean rank error'
    CHECKS['B_error_symbols']={'rank':'epsilon_r','within_rod_order':'epsilon_o','participation':'epsilon_p'}
    CHECKS['B_restored_source_axis_sizes_mm']=dict(profile=[42,45],trajectory=[51,51])
    fig.canvas.draw()
    for c in cases:
        x,y,_=base.proj3d.proj_transform(*[c[k] for k in base.KEYS],tr.get_proj())
        tr.annotate(c['panel'],(x,y),xytext=(4,4),textcoords='offset points',fontsize=9)
    CHECKS['B_workpoints']=len(rows)
    CHECKS['B_stages']=sorted({r['stage'] for r in rows})
    fig.canvas.draw()
    corner=p.transAxes.transform((0,0))
    left=p.spines['left'].get_transform().transform(p.spines['left'].get_path().vertices)[0]
    bottom=p.spines['bottom'].get_transform().transform(p.spines['bottom'].get_path().vertices)[0]
    np.testing.assert_allclose(left,corner,atol=1e-6)
    np.testing.assert_allclose(bottom,corner,atol=1e-6)
    CHECKS['B_axis_corner_gap_px']=float(np.linalg.norm(left-bottom))
    boxes=[t.get_window_extent() for t in p.get_xticklabels()]
    assert all(a.x1<b.x0 for a,b in zip(boxes,boxes[1:]))
    CHECKS['B_category_labels_do_not_overlap']=True
    CHECKS['B_profile_ylabel']=p.get_ylabel()
    CHECKS['B_mean_rank_error_native_axis_label']=True
    CHECKS['B_order_error_native_axis_label']=True
    CHECKS['B_order_error_label_rotation_deg']=float(tr.yaxis.label.get_rotation())


def panel_c(fig, curves, geometry):
    rank_color,order_color=ERROR_COLORS['rank_error'],ERROR_COLORS['within_rod_order_error']
    order_text=order_color
    palette={'rank_error':rank_color,'within_rod_order_error':order_color}
    cmaps={key:matplotlib.colors.LinearSegmentedColormap.from_list(
        'fig4_'+key,[low,color]) for key,low,color in [
            ('rank_error','#F3F6F7',rank_color),
            ('within_rod_order_error','#FFF2EF',order_color)]}
    na,nt=len(fig.axes),len(fig.texts)
    with patch.object(base,'BLUE',rank_color),patch.object(base,'ORANGE',order_color),\
            patch.dict(plot.CMAPS,cmaps),patch.dict(plot.LABEL_COLORS,palette):
        plot.parameters(fig,curves,geometry)
    axes=fig.axes[na:]
    for a,r in zip(axes,[(13,MID_ROWS['D'],40,ROW_HEIGHT),(13,MID_ROWS['D'],40,ROW_HEIGHT),
                         (67,MID_ROWS['D'],40,ROW_HEIGHT),(67,MID_ROWS['D'],40,ROW_HEIGHT),
                         (13,MID_ROWS['E'],40,ROW_HEIGHT),(55,MID_ROWS['E'],1.4,ROW_HEIGHT),
                         (67,MID_ROWS['E'],40,ROW_HEIGHT),(109,MID_ROWS['E'],1.4,ROW_HEIGHT)]):
        pos(a,*r);axis_type(a)
    for t,x in zip(fig.texts[nt:],[33,87]):
        t.set_position((x/W,(MID_ROWS['E']+ROW_HEIGHT+2)/H));t.set_fontsize(9.5)
    axes[0].set_xticklabels(['1.0','1.25','1.5'])
    axes[0].set_xlabel('Outward E→E gain')
    axes[2].set_xlabel('E→E angle (°)')
    axes[0].set_ylabel(r'Mean rank error $\varepsilon_r$',color=rank_color)
    axes[3].set_ylabel(r'Order error $\varepsilon_o$',color=order_text)
    for a in [axes[1],axes[3]]:a.tick_params(axis='y',labelcolor=order_text)
    for a in axes:
        for axis in [a.xaxis,a.yaxis]:
            axis.label.set_fontsize(MIDDLE_LABEL_PT)
            axis.labelpad=2
    CHECKS['C_axis_label_font_pt']=MIDDLE_LABEL_PT
    axes[1].tick_params(axis='y',labelright=True)
    axes[2].tick_params(axis='y',labelleft=True)
    for a in axes[:4]:a.set_yticks([0,.2,.4])
    # Keep all four numeric scales explicit, with clear separation in the gap.
    for a in axes[:4]:a.tick_params(axis='y',pad=1)
    for t,s in zip(fig.texts[nt:],[r'$\varepsilon_r$',r'$\varepsilon_o$']):
        t.set_text(s);t.set_fontsize(11)
    fig.texts[nt+1].set_color(order_text)
    axes[5].set_yticks([0,.3]);axes[7].set_yticks([0,.3])
    CHECKS['C_position_conditions']=geometry['conditions']
    for a,color in zip(axes[:4],[rank_color,order_color]*2):
        assert matplotlib.colors.to_hex(a.lines[0].get_color()).upper()==color
    for a,key in [(axes[4],'rank_error'),(axes[6],'within_rod_order_error')]:
        for im in a.images:
            assert im.get_clim()==(0.,.3)
            assert matplotlib.colors.to_hex(im.get_cmap()(1.)).upper()==palette[key]
    CHECKS['C_palette']={**palette,'order_text':order_text,
        'map_limits':[0.,.3],'map_gradient':'light to the corresponding curve color'}


def panels_de(fig,cases):
    freeze(WT/'src/topic4_streaming_spike_readout.py')
    CHECKS['DE_native_colorbar_quantity']={
        'observable':'E neurons firing at least once within each spatial/time bin',
        'spatial_bin_mm2':1.,'time_bin_ms':2.,'display_range':[0,75]}
    na,nt=len(fig.axes),len(fig.texts)
    whole_size=fig.get_size_inches().copy()
    fig.set_size_inches(238*MM,190*MM)
    manifests=plot.compact_propagation(fig,cases)
    reference=read(freeze(OLD/'source/propagation_manifest.json'))['cases']
    for new,old in zip(manifests,reference):
        np.testing.assert_equal({k:v for k,v in new.items() if k!='panel'},
                                {k:v for k,v in old.items() if k!='panel'})
    axes,texts=fig.axes[na:],fig.texts[nt:]
    original={a:np.array(a.get_position().bounds)*[238,190,238,190] for a in axes}
    fig.set_size_inches(whole_size)
    # Use the original artist roles and regroup the two rows before reflow.
    for i,(letter,old_y,new_y) in enumerate([('D',55,MID_ROWS['D']),('E',10,MID_ROWS['E'])]):
        ga=[a for a in axes if (original[a][1]>=50)==(i==0)]
        gt=[t for t in texts if (t.get_position()[1]*190>=50)==(i==0)]
        GROUPS[letter]=dict(axes=ga,texts=gt,artists=[])
        for a in ga:
            ox,oy,ow,oh=original[a]
            if a.images and ox<110:  # contact envelope
                x=130 if ox<60 else 166
                pos(a,x,new_y,24,ROW_HEIGHT);axis_type(a,dense=True)
                a.set_xlabel('Time (ms)' if i else '',fontsize=MIDDLE_LABEL_PT)
                a.tick_params(axis='x',labelbottom=True)
                a.tick_params(axis='y',labelsize=8.5)
                # Shared channel identity on the first envelope; fixed slots
                # and tick marks remain on the second one.
                if x==166:a.tick_params(axis='y',labelleft=False)
            elif a.images:  # native snapshots, equal-aspect and unmodified
                col=round((ox-135)/20);row=1 if oy>old_y+1 else 0
                pos(a,211+20.5*col,new_y+FIELD_STEP*row,FIELD_SIDE,FIELD_SIDE);axis_type(a)
                a.tick_params(labelsize=8,labelleft=col==0 and row==0,
                              labelbottom=col==0 and row==0)
                if col==0 and row==0:
                    a.set_xlabel('x (mm)' if i else '',fontsize=MIDDLE_LABEL_PT,labelpad=1)
                    if i:
                        # Preserve label height below the visible ticks, but
                        # center its x anchor across all three snapshot columns.
                        a.xaxis.label.set_x(.5+20.5/FIELD_SIDE)
            elif ox<120:  # one envelope scale for the two same-scale plots
                if ox<80:a.set_visible(False);continue
                pos(a,192,new_y,1.4,ROW_HEIGHT);axis_type(a)
                a.set_ylabel('')
                a.set_title(r'$\widetilde{E}$',fontsize=11,pad=4)
            else:  # one native scale per row, matching the source
                row=1 if oy>old_y+1 else 0
                pos(a,273,new_y+FIELD_STEP*row,1.3,FIELD_SIDE);axis_type(a)
                a.tick_params(labelsize=8);a.set_ylabel('')
        for t in gt:
            ox,oy=np.array(t.get_position())*[238,190]
            s=t.get_text()
            if s in ['TA','TB']:
                t.set_text('M'+s);t.set_color(RED if s=='TA' else BLUE)
                t.set_fontweight('bold');t.set_fontsize(10)
                if ox<110:
                    t.set_position(((142 if s=='TA' else 178)/W,(new_y+ROW_HEIGHT+2)/H))
                else:
                    t.set_position((209/W,(new_y+FIELD_SIDE/2+(FIELD_STEP if s=='TA' else 0))/H))
                    t.set_ha('center');t.set_fontsize(9);t.set_rotation(90)
            elif s in ['0 ms','30 ms','60 ms']:
                col=['0 ms','30 ms','60 ms'].index(s)
                t.set_position(((211+FIELD_SIDE/2+20.5*col)/W,(new_y+ROW_HEIGHT+2)/H));t.set_fontsize(9)
            elif s=='y (mm)':
                t.set_position((204/W,(new_y+ROW_HEIGHT/2)/H));t.set_fontsize(MIDDLE_LABEL_PT)
                t.set_rotation(90)
            elif s=='Activity / 2 ms':
                t.set_text('E cells / mm² / 2 ms')
                t.set_position((283/W,(new_y+ROW_HEIGHT/2)/H));t.set_fontsize(9)
    CHECKS['DE_frozen_event_ids']=[[e['event'] for e in m['events']] for m in manifests]
    CHECKS['DE_frozen_events_and_scales_match']=True
    CHECKS['DE_axis_label_font_pt']=MIDDLE_LABEL_PT
    CHECKS['DE_channel_tick_font_pt']=8.5


def panels_fgh(fig):
    z=np.load(freeze(PREVIOUS/'source/comparison_arrays.npz'))
    meta=read(freeze(PREVIOUS/'comparison_metadata.json'))
    order=z['display_order']
    with group(fig,'G'),patch.object(plt,'figure',return_value=fig),patch.object(compare,'save'):
        compare.rank_panel(order,z['model_profiles_raw_mode_0_1'],z['patient_profiles_raw_mode_0_1'])
    g=GROUPS['G']['axes'][0];pos(g,131,LOWER_Y,35,ROW_HEIGHT);axis_type(g,dense=True)
    g.set_xticks([0,7,14])
    leg=fig.legends[-1];leg.remove()
    # Figure legends have to be explicitly associated with a panel for export.
    previous_legend_margin_mm=.15*7/72*25.4
    g.legend(handles=[Line2D([],[],color=c,ls=ls,marker=m,lw=1.2,ms=2.8,label=s)
        for c,ls,m,s in [(RED,'-','o','MTA'),(RED,'--',None,'TA'),(BLUE,'-','o','MTB'),(BLUE,'--',None,'TB')]],
        ncol=2,loc='upper right',bbox_to_anchor=(.99-previous_legend_margin_mm/35,1.075),frameon=True,
        facecolor='white',edgecolor='none',framealpha=.9,
        fontsize=7,handlelength=1.3,handletextpad=.3,columnspacing=.6,
        labelspacing=.2,borderpad=.2,borderaxespad=0)
    CHECKS['G_legend_position']='upper right, straddling the upper edge; model MTA/MTB solid, patient TA/TB dashed; raised 3 mm without changing axis bounds'
    CHECKS['G_legend_upward_shift_mm']=ROW_HEIGHT*.075
    CHECKS['G_legend_font_pt']=7.
    with group(fig,'H'),patch.object(plt,'figure',return_value=fig),patch.object(compare,'save'):
        compare.matrix_panel({'matrix':z['crossfit_matrix_raw_mode_0_1']},meta['diagonal_tests'])
    h,cb=GROUPS['H']['axes'];pos(h,179,LOWER_Y,40,ROW_HEIGHT);pos(cb,221,LOWER_Y,1.6,ROW_HEIGHT)
    h.set_aspect('equal');axis_type(h);axis_type(cb)
    cb.set_yticks([-1,0,1]);cb.set_ylabel('')
    h.set_title('Spearman ρ',fontsize=10,pad=7)
    for t in h.get_xticklabels()+h.get_yticklabels():t.set_fontweight('bold')
    CHECKS['H_patient_template_labels_bold']=True
    CHECKS['H_model_template_labels_bold']=True
    for t in h.texts:t.set_fontsize(10 if t.get_text().startswith(('+','-')) else 10)
    trajectory=Path(meta['workpoint']['source']);run=read(freeze(trajectory))
    needed=['contact_names','contact_envelope','contact_envelope_dt_ms','centroid_ms',
            'recruitment_ms','primary_event_indices','event_mode','event_time_ms']
    with np.load(freeze(trajectory.with_suffix('.npz'))) as raw:a={k:raw[k] for k in needed}
    with group(fig,'F'),patch.object(plt,'figure',return_value=fig),patch.object(compare,'save'),patch.object(compare,'SOURCE',SOURCE):
        wave=compare.waveform_panel(run,a,z['model_event_ids'],order)
    np.testing.assert_equal(wave,meta['waveform'])
    with np.load(PREVIOUS/'source/waveform_arrays.npz') as old, np.load(SOURCE/'waveform_arrays.npz') as new:
        for k in old.files: np.testing.assert_array_equal(old[k],new[k])
    f=GROUPS['F']['axes'][0];pos(f,15,LOWER_Y,100,ROW_HEIGHT);axis_type(f,dense=True);f.set_xticks([0,200,400,600,800])
    # Complete the shading for visible events using the existing model labels
    # and the same recruitment-onset span plus 12 ms as the source renderer.
    shaded_ids={span['event'] for span in wave['spans']}
    extra_spans=[]
    start=wave['absolute_window_ms'][0]
    for event in wave['all_primary_ids_visible']:
        if event in shaded_ids:continue
        mode=int(a['event_mode'][event]);assert mode in [0,1]
        valid=a['recruitment_ms'][event];valid=valid[np.isfinite(valid)]
        lo,hi=float(valid.min()-start-12),float(valid.max()-start+12)
        color=RED if mode==1 else BLUE
        f.axvspan(lo,hi,color=color,alpha=.14,lw=0,zorder=-2)
        extra_spans.append(dict(event=event,mode='MTA' if mode==1 else 'MTB',
            center_relative_ms=float(a['event_time_ms'][event]-start),
            shaded_span_relative_ms=[lo,hi],color=color,alpha=.14))
    CHECKS['F_additional_event_shading']=extra_spans
    f.legend_.remove()
    f.legend(handles=[matplotlib.patches.Patch(fc=c,alpha=.22,label=s) for c,s in [(RED,'MTA'),(BLUE,'MTB')]],
             ncol=2,loc='upper right',bbox_to_anchor=(.995,.995),frameon=True,
             facecolor='white',edgecolor='none',framealpha=.9,
             fontsize=8,borderaxespad=.15,borderpad=.2,
             handlelength=1.1,columnspacing=.8,handletextpad=.4)
    CHECKS['F_legend_position']='inside upper right, overlaying traces without adding y-axis headroom'
    CHECKS['F_legend_font_pt']=8.
    CHECKS['F_extra_legend_headroom_fraction']=0.
    CHECKS['F_waveform_exact_array_parity']=True
    CHECKS['GH_frozen_matrix']=z['crossfit_matrix_raw_mode_0_1'].tolist()


def panel_i(fig):
    path=freeze(cohort.DEST/'figure_data.json');data=read(path)
    patient_color,median_color='#D7B08A','#8C4B20'
    ax=fig.add_axes(rect(242,LOWER_Y,32,ROW_HEIGHT))
    cohort.trajectory(ax,data['records'],legend_loc='upper right',legend_frame=True)
    axis_type(ax);ax.set_yticks([0,.5,1]);ax.set_yticklabels(['0','0.5','1.0'])
    ax.set_xlabel(COHORT_XLABEL,fontsize=10,labelpad=2)
    ax.set_ylabel('Loss / baseline',fontsize=10,labelpad=3)
    for ln in ax.lines:
        individual=ln.get_zorder()==1
        ln.set_color(patient_color if individual else median_color)
        ln.set_alpha(.85 if individual else 1.)
        ln.set_linewidth(.55 if individual else 1.5)
        if ln.get_marker()=='o':ln.set_markersize(3)
    handles=list(ax.legend_.legend_handles)
    for handle,color in zip(handles,[patient_color,median_color]):handle.set_color(color)
    ax.legend(handles,['Subject','Median'],loc='upper right',frameon=True,
              fontsize=8.5,handlelength=1.3,handletextpad=.5,borderpad=.3,
              labelspacing=.25,borderaxespad=.3)
    CHECKS['I_legend_position']='inside upper right, original framed placement'
    CHECKS['I_paired_patients']=data['n_paired']
    CHECKS['I_condition_counts']=data['condition_counts']
    CHECKS['I_palette']={'Subject':patient_color,'Median':median_color,'subject_alpha':.85}
    CHECKS['I_xaxis_label']=COHORT_XLABEL
    CHECKS['I_xaxis_definition']='Author-requested Epochs display label; values remain cumulative evaluated-condition indices at batch boundaries: 1, 12, 20, 28, not minibatch or gradient-training epochs. B colors identify proposal stages 1 through 16.'


def visible_texts(fig):
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()
            and (t.axes is None or t.axes.get_visible())]


def verify_alignment(fig, *, panel_letters=None, extra_top_groups=(), parameter_rows=None):
    """Audit actual rendered axis edges, including equal-aspect adjustment."""
    fig.canvas.draw()
    def bounds(a):
        b=a.get_window_extent()
        return np.array([b.x0,b.y0,b.x1,b.y1])/fig.dpi*25.4
    def assert_edges(boxes):
        edges=np.array([[b[1],b[3]] for b in boxes])
        np.testing.assert_allclose(edges,np.broadcast_to(edges[0],edges.shape),rtol=0,atol=1e-7)
        np.testing.assert_allclose(edges[:,1]-edges[:,0],ROW_HEIGHT,rtol=0,atol=1e-7)
        return edges.tolist()
    lower={s:bounds(GROUPS[s]['axes'][0]) for s in 'FGHI'}
    assert_edges(list(lower.values()))
    CHECKS['FGHI_axis_bounds_mm']={s:b.tolist() for s,b in lower.items()}
    CHECKS['FGHI_equal_height_and_edges']=True
    hb=lower['H'];np.testing.assert_allclose(hb[2]-hb[0],ROW_HEIGHT,rtol=0,atol=1e-7)
    c=GROUPS['C']['axes']
    middle={}
    curve_rows=parameter_rows if parameter_rows is not None else {'D':[c[0],c[2]],'E':[c[4],c[6]]}
    for s,caxes in curve_rows.items():
        axes=GROUPS[s]['axes']
        env=[a for a in axes if a.images and a.get_xlim()[0]<0]
        fields=[a for a in axes if a.images and a.get_xlim()[0]>=0]
        assert len(env)==2 and len(fields)==6
        fb=np.array([bounds(a) for a in fields])
        combined=np.array([fb[:,0].min(),fb[:,1].min(),fb[:,2].max(),fb[:,3].max()])
        boxes=[bounds(a) for a in caxes+env]+[combined]
        edges=assert_edges(boxes)
        for x in sorted(set(np.round(fb[:,0],7))):
            pair=fb[np.isclose(fb[:,0],x)]
            assert len(pair)==2
            np.testing.assert_allclose([pair[:,1].min(),pair[:,3].max()],edges[0],atol=1e-7)
        middle[s]=dict(C_axes=[bounds(a).tolist() for a in caxes],
                       envelopes=[bounds(a).tolist() for a in env],native_two_row_bounds_mm=combined.tolist())
        if s=='E':
            xlabel=next(a.xaxis.label for a in fields if a.get_xlabel()=='x (mm)')
            lb=xlabel.get_window_extent()
            label_center=(lb.x0+lb.x1)/2/fig.dpi*25.4
            field_center=(combined[0]+combined[2])/2
            np.testing.assert_allclose(label_center,field_center,rtol=0,atol=1e-7)
            CHECKS['E_native_shared_xlabel_center_mm']=float(label_center)
            CHECKS['E_native_xlabel_centered_across_three_columns']=True
    CHECKS['CDE_axis_bounds_mm']=middle
    CHECKS['CDE_equal_row_height_and_edges']=True
    CHECKS['common_axis_height_mm']=ROW_HEIGHT
    for s in ['F','G','I']:
        a=GROUPS[s]['axes'][0]
        if s=='I' and a.get_legend() is None:
            CHECKS['I_legend_present']=False
            continue
        lb=a.legend_.get_window_extent(); ab=a.get_window_extent()
        assert lb.x0>=ab.x0 and lb.x1<=ab.x1 and lb.y0>=ab.y0
        if s=='G':
            np.testing.assert_allclose((lb.y1-ab.y1)/fig.dpi*25.4,3.,atol=1e-7)
            assert lb.y0<ab.y1
            CHECKS['G_legend_straddles_axes_upper_edge']=True
        else:
            assert lb.y1<=ab.y1
            CHECKS[f'{s}_legend_inside_axes_verified']=True
    ht=GROUPS['H']['axes'][0].get_xticklabels()
    assert [t.get_text() for t in ht]==['TA','TB']
    assert all(t.get_fontweight()=='bold' for t in ht)
    hy=GROUPS['H']['axes'][0].get_yticklabels()
    assert [t.get_text() for t in hy]==['MTA','MTB']
    assert all(t.get_fontweight()=='bold' for t in hy)
    i_legend=GROUPS['I']['axes'][0].get_legend()
    CHECKS['I_legend_labels']=[] if i_legend is None else [t.get_text() for t in i_legend.get_texts()]
    if i_legend is not None:
        assert CHECKS['I_legend_labels']==['Subject','Median']
    tick_key='C_single_yaxis_tick_labels' if parameter_rows is not None else 'C_four_yaxis_tick_labels'
    error_axes=[a for row in curve_rows.values() for a in row] if parameter_rows is not None else c[:4]
    CHECKS[tick_key]=[
        [t.get_text() for t in a.get_yticklabels() if t.get_visible()]
        for a in error_axes]
    assert all(len(t)==3 for t in CHECKS[tick_key])
    if parameter_rows is not None:
        CHECKS['CDE_equal_height_scope']='Parameter error axes, envelopes, and combined snapshot rows; position-density map is deliberately smaller.'
    renderer=fig.canvas.get_renderer()
    panel_letters=panel_letters or {}
    def row_box(keys):
        boxes=[]
        for s in keys:
            g=GROUPS[s]
            boxes += [a.get_tightbbox(renderer) for a in g['axes'] if a.get_visible()]
            boxes += [t.get_window_extent(renderer) for t in g['texts'] if t.get_visible()]
        displayed_letters={panel_letters.get(s,s) for s in keys}
        boxes += [t.get_window_extent(renderer) for t in fig.texts if t.get_text() in displayed_letters]
        return Bbox.union([b for b in boxes if b is not None])
    top,middle,bottom=[row_box(keys) for keys in [list('AB')+list(extra_top_groups),'CDE','FGHI']]
    gaps=np.array([top.y0-middle.y1,middle.y0-bottom.y1])/fig.dpi*25.4
    assert np.all(gaps>0) and abs(gaps[0]-gaps[1])<.25, gaps.tolist()
    CHECKS['visible_row_gaps_mm']=dict(AB_to_CDE=float(gaps[0]),CDE_to_FGHI=float(gaps[1]))
    CHECKS['FGHI_downward_shift_mm']=LOWER_ROW_EXTRA_GAP_MM
    CHECKS['H_leftward_shift_mm']=5.
    CHECKS['I_leftward_shift_mm']=9.
    CHECKS['I_axes_right_margin_mm']=float(W-lower['I'][2])
    for left,right in [('G','H'),('H','I')]:
        gap=(row_box(right).x0-row_box(left).x1)/fig.dpi*25.4
        assert gap>0,(left,right,gap)
        CHECKS[f'{left}{right}_visible_horizontal_gap_mm']=float(gap)


def main():
    FIG.mkdir(parents=True,exist_ok=True);SOURCE.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,
        'xtick.labelsize':9,'ytick.labelsize':9,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})
    rows=list(csv.DictReader(freeze(OLD/'source/all_workpoints_source.csv').open()))
    for r in rows:
        for k in base.KEYS:r[k]=float(r[k])
        r['stage']=int(r['stage']);r['N']=int(r['N'])
    cases=read(freeze(PREVIOUS/'source/selected_workpoints.json'))['cases']
    curves=list(csv.DictReader(freeze(OLD/'source/connection_response_source.csv').open()))
    geometry=read(freeze(EXPANDED/'source/position_map_source.json'))
    assert geometry['conditions']==83
    fig=plt.figure(figsize=(W*MM,H*MM),dpi=160,facecolor='white')
    for letter,fn in [('A',lambda:panel_a(fig)),('B',lambda:panel_b(fig,rows,cases)),
                       ('C',lambda:panel_c(fig,curves,geometry))]:
        with group(fig,letter):fn()
        print('Rendered',letter,flush=True)
    panels_de(fig,cases);print('Rendered D/E',flush=True)
    panels_fgh(fig)
    with group(fig,'I'):panel_i(fig)
    letters=[]
    for s,x,y in [('A',2,225+TOP_SHIFT),('B',137,225+TOP_SHIFT),('C',2,179),('D',122,178),('E',122,119),
                   ('F',2,63),('G',120,63),('H',170,63),('I',230,63)]:
        letters.append(label(fig,x,y,s,fontsize=FONT['letter'],weight='bold',va='top'))
    # Extend the canvas below C/D/E so the lower row gains the same visible
    # inter-row gap as A/B -> C/D/E, without shrinking any axes or text.
    # Twin axes share positions: read all boxes before moving either twin.
    lower_axes={a for s in 'FGHI' for a in GROUPS[s]['axes']}
    lower_texts={t for s in 'FGHI' for t in GROUPS[s]['texts']}
    lower_texts.update(t for t in letters if t.get_text() in 'FGHI')
    before_shift={a:a.get_position().frozen() for a in fig.axes}
    for a,b in before_shift.items():
        x,y,w,h=b.bounds
        shift=5 if a in lower_axes else 5+LOWER_ROW_EXTRA_GAP_MM
        a.set_position([x,y+shift/H,w,h])
    for t in fig.texts:
        shift=5 if t in lower_texts else 5+LOWER_ROW_EXTRA_GAP_MM
        x,y=t.get_position();t.set_position((x,y+shift/H))
    for a in GROUPS['A']['artists']:
        a.set_ydata(np.array(a.get_ydata())+(5+LOWER_ROW_EXTRA_GAP_MM)/H)
    fig.canvas.draw()
    verify_alignment(fig)
    texts=visible_texts(fig);rr=fig.canvas.get_renderer()
    outside=[]
    for t in texts:
        b=t.get_window_extent(rr)
        if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.x1+.5 or b.y1>fig.bbox.y1+.5:
            outside.append(t.get_text())
    CHECKS['text_outside_canvas']=outside
    print('Outside text',outside,flush=True)
    assert not outside, outside
    CHECKS['minimum_visible_font_pt']=min(t.get_fontsize() for t in texts)
    CHECKS['dense_font_at_180mm_page_width_pt']=FONT['dense']*180/W
    for s in ['D','E']:
        modes=[t for t in GROUPS[s]['texts'] if t.get_text() in ['MTA','MTB']]
        assert len(modes)==4
        assert all(t.get_color()==(RED if t.get_text()=='MTA' else BLUE) for t in modes)
    CHECKS['DE_semantic_titles_verified']=True
    positions={a:a.get_position().frozen() for a in fig.axes}
    for ext in ['png','pdf','svg']:
        fig.savefig(FIG/f'fig4-complete-layout.{ext}',dpi=320)
        for a,p in positions.items():a.set_position(p)
    fig.savefig(FIG/'fig4-complete-layout-preview.png',dpi=135)
    # Export the exact same artists as separate panels, without letters. No
    # panel is resized again when composed, so all font sizes stay comparable.
    visibility={a:a.get_visible() for a in fig.axes}
    for t in letters:t.set_visible(False)
    for s in 'ABCDEFGHI':
        for k,g in GROUPS.items():
            for a in g['axes']:a.set_visible(k==s and visibility[a])
            for a in g['texts']+g['artists']:a.set_visible(k==s)
        # D and E share units on the complete plate. Restore them in D's
        # standalone export so the independently placed panel is self-contained.
        restored=[]
        if s=='D':
            for a in GROUPS[s]['axes']:
                if a.images and a.get_visible() and any(t.get_visible() for t in a.get_xticklabels()):
                    restored.append((a,a.get_xlabel()))
                    a.set_xlabel('Time (ms)' if a.get_xlim()[0]<0 else 'x (mm)',fontsize=MIDDLE_LABEL_PT)
        fig.canvas.draw();rr=fig.canvas.get_renderer();g=GROUPS[s]
        boxes=[a.get_tightbbox(rr) for a in g['axes'] if a.get_visible()]
        boxes += [t.get_window_extent(rr) for t in g['texts'] if t.get_visible()]
        # Include projected 3D axis titles omitted by some Matplotlib versions.
        boxes += [t.get_window_extent(rr) for a in g['axes'] if a.get_visible()
                  for t in a.findobj(Text) if t.get_visible() and t.get_text()]
        box=Bbox.union([b for b in boxes if b is not None]).transformed(fig.dpi_scale_trans.inverted()).padded(1.2*MM)
        for ext in ['png','pdf','svg']:
            fig.savefig(FIG/f'fig4-panel{s.lower()}.{ext}',dpi=320,bbox_inches=box)
            for a,p in positions.items():a.set_position(p)
        for a,old_label in restored:a.set_xlabel(old_label)
    plt.close(fig)
    freeze(Path(__file__))
    for m in [compare,combined,cohort]:freeze(m.__file__)
    record=dict(status='CURRENT_AUTHOR_DESIGNATED',author_designated_on='2026-09-28',
        current_version_document='docs/current_figure4.md',producer=str(Path(__file__).relative_to(ROOT)),
        whole_canvas_mm=[W,H],font_pt=FONT,source_files=SOURCES,checks=CHECKS,
        typography_exception='A preserves the local circuit pixels and physical scale; only its external title, coordinate labels/ticks and neuron legend are enlarged.',
        layout_contract='F/G/H/I share identical rendered y bounds; C upper row/D and C lower row/E each share rendered y bounds with both contact envelopes and the combined two-row native fields. All data-axis heights are 40 mm. H and spatial maps retain equal x/y aspect. The bottom row is 6.5 mm lower relative to the upper panels, matching the visible row gaps; H moves left 5 mm and I left 9 mm, with all axis sizes preserved.',
        data_changes=False,new_simulations=0,reference='User-provided complete A–I plate, 2026-09-23',
        source_version_mix='B: September 18, 226 workpoints / 16 stages; C: September 20 expanded 83-condition position maps, with unchanged connection curves. Matches the attachment.',
        semantic_colors=dict(MTA=RED,MTB=BLUE,**ERROR_COLORS),canonical_assets_overwritten=False,
        outputs={str(p.relative_to(OUT)):sha(p) for p in FIG.iterdir() if p.suffix in ['.png','.pdf','.svg']})
    (OUT/'figure4_candidate_registry.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    write_current_version(record)
    titles={'A':'局部 E/I 回路与患者电极空间基底','B':'参数搜索中的三类误差与搜索历程',
        'C':'EE 强度、方向及核位置的误差响应','D':'较高误差工作点的传播示例（lf_bo04_07）',
        'E':'较低误差工作点的传播示例（xy_left_20）','F':'连续 30–80 Hz 虚拟接触活动',
        'G':'模型与患者的平均传播 rank','H':'模型—患者触点交叉匹配','I':'跨患者最佳训练 loss 随评估条件数的变化'}
    notes=['### fig4-complete-layout.png\nA 放大外部文字并恢复完整矢量虚线框，左侧 E→E/I→E 等内部文字保持原样；B 保留原始轴框比例，迭代色条、搜索轨迹和三维散点统一为参考紫色渐变（#B1A9DA → #7367BE → #483D8B），三维误差标签加粗；B/C 的 rank 误差名称统一为 Mean rank error。横轴及三维误差标签共用语义色：rank 深青灰、order 珊瑚红、participation 深橄榄绿。C 的 rank/order 误差改为深青灰 #3D5C6F 与珊瑚红 #E47159，曲线、轴文字和地图色条同步。C 两张曲线均有完整左右刻度，C/D/E 轴标签增至 12 pt、D/E 通道刻度 8.5 pt；F 图例在图内右上角并允许局部覆盖曲线，约 396 ms 的 MTB 事件补上相同蓝色阴影；G 图例再上移 3 mm，跨在绘图区上边缘，字号 7 pt、F 字号 8 pt；I 图例为 Subject / Median，横轴为 Epochs；个体轨迹采用浅棕 #D7B08A、中位数采用暖深棕 #8C4B20，H 使用 Spearman ρ 标题且 TA/TB、MTA/MTB 均加粗，D/E 使用红色 MTA 和蓝色 MTB。F/G/H/I 整行下移 6.5 mm、与中排之间的可见留白匹配 A/B 至 C/D 的间距；H 左移 5 mm、I 左移 9 mm，收紧 H 两侧留白。F/G/H/I 绘图区上下边界一致；C 上下两行分别与 D/E 的包络、右侧两行快照总框严格对齐，均高 40 mm。\n**关注点**：整图 PNG/PDF/SVG 同次导出，H 与空间图保持正方形；本版为作者于 2026-09-28 指定的当前 Fig4；后续改图仍需目视检查。']
    for s,title in titles.items():
        detail='局部回路直接复用原始像素并保持物理尺寸，外部文字为放大的可编辑文字。' if s=='A' else '独立 PNG/PDF/SVG 与整图使用相同绘图对象和数据，独立图不带角标。'
        notes.append(f'### fig4-panel{s.lower()}.png\n{s}：{title}。{detail}\n**关注点**：沿用源数据的解释边界，版式调整不构成新的科学验收。')
    (FIG/'README.md').write_text('\n\n'.join(notes)+'\n')
    mapping={
        'A':ROOT/'results/paper-ready-figure/fig4/figures/fig4-panela.png',
        'B':PREVIOUS/'figures/fig4-panelb.png',
        'C':EXPANDED/'figures/fig4-panela.png',
        **{s:PREVIOUS/f'figures/fig4-panel{s.lower()}.png' for s in 'DEFGH'},
        'I':cohort.DEST/'figures/cohort_loss_trajectory.png',
    }
    table=['# Figure 4 A–I 对应关系','',
        '当前身份：CURRENT_AUTHOR_DESIGNATED（用户指定，2026-09-28）。统一入口见 docs/current_figure4.md 与 fig4/current_version.json。','',
        '按 2026-09-23 用户提供的完整拼图核对。B 使用 9/18 的 226 个工作点 / 16 个阶段；C 使用 9/20 的 83 条件位置扫描，上方连接曲线未变。I 的原图在队列分析目录，本次一并导入 paper-ready 候选。','',
        '| 编号 | 子图标题 / 内容 | 原图 | 本次子图 |','|---|---|---|---|']
    for s,title in titles.items():
        table.append(f'| {s} | {title} | [来源]({mapping[s]}) | [PNG]({FIG}/fig4-panel{s.lower()}.png) |')
    table += ['', f'A 保留局部回路内部文字，仅放大外部标题、坐标和 neuron 图例，并恢复完整的 0.65 pt 矢量虚线框；整图 {W:g} × {H:g} mm 画布默认轴标签 10 pt，C/D/E 坐标标签增至 12 pt；普通刻度 9 pt、D/E 通道名 8.5 pt、F/G 通道名 7.5 pt。B 平行坐标恢复 42 × 45 mm，三维轴框恢复 51 × 51 mm；左侧纵轴为 Error、不带箭头，三维 Order error ε_o 使用随轴投影方向旋转的原生轴标签、紧邻刻度。G 图例以 7 pt 置于右上角，本次再上移 3 mm、跨在绘图区上边缘，其中 MTA/MTB 为模型实线、TA/TB 为患者虚线；F 图例以 8 pt 置于图内右上角覆盖少量曲线，取消此前为图例增加的纵轴留白；I 图例在图内，个体标签为 Subject，横轴按作者要求显示 Epochs；B 色条仍为 Iteration order；H 的 TA/TB、MTA/MTB 标签均加粗。', '',
        'F/G/H/I 整行下移 6.5 mm，画布高度相应增加；与中排的可见留白匹配 A/B 至 C/D 的间距。H 左移 5 mm、I 左移 9 mm，收紧 H 两侧留白，I 绘图区距画布右边缘 11 mm。各绘图区尺寸保持不变，F/G/H/I 统一 40 mm 高且上下边界一致，H 保持正方形。C 上下两行分别与 D/E 包络和右侧两行原生快照总框等高并对齐，均为 40 mm；空间图保持等比例。验收数值来自渲染后的轴框，见 registry checks。', '',
        '完整拼图与独立子图均提供 PNG/PDF/SVG；新增及放大的外部文字可编辑。D/E 在整图中共用 E 下方的横轴单位，E 右侧 x (mm) 居中于三列快照总框下方；独立 D 补回单位。旧正式图和原候选保留；本版为作者于 2026-09-28 指定的当前 Fig4；后续改图仍需目视检查。', '',
        '短标记定义：B/C 的名称统一为 Mean rank error，共用 ε_r（平均 rank 误差）、ε_o（杆内先后概率误差），B 的 ε_p 为触点参与概率误差。B 平行坐标类别名下面和三维误差轴文字后面均标出对应符号，两处标签颜色统一：ε_r 深青灰 #3D5C6F、ε_o 珊瑚红 #E47159、ε_p 深橄榄绿 #6B7546；C 上方曲线与下方地图共用相同标记和误差配色：rank 为深青灰 #3D5C6F，order 为珊瑚红 #E47159，轴文字和曲线颜色一致。两个地图均从浅色渐变至对应参考色，范围保持 0–0.3。Outward E→E gain 是核内 E 神经元向核外 E 神经元的连接权重相对基线的乘数，1 表示基线。D/E 的 Ẽ 表示逐触点最大值归一化的模型包络；使用正文包络 E 的字母习惯，但此处是模型发放密度读出，不是患者电压或高频能量。当前本地 Fig2 canonical 仍为完整英文标签，未将本次短记号虚称为既有已锁定符号。', '',
        'D/E 右侧 0–75 色标表示每 1 mm² 空间格、每 2 ms 时间窗内至少发放一次的 E 神经元数，色标上限仍为 75。来源 StreamingSpikes 使用 count > 0 后按网格计数；不是 Hz，也不是窗口内总 spike 次数。B 色条为参考紫色渐变的 Iteration order，曲线和三维散点共享该映射；B 刻度代表 16 个提案阶段。I 横轴按作者要求显示 Epochs，保留的刻度仍代表累计已评估条件的序号，不改写为梯度训练轮次；1/12/20/28 分别是基线、初始批次结束、两批自适应提案结束。I 的 Loss / baseline 保持原定义：每位患者截至该条件数的最低训练 loss 除以自身基线；Subject 为浅棕色（#D7B08A）个体轨迹，Median 为暖深棕色（#8C4B20）跨受试者中位数，图例同步。', '',
        'F 约 396 ms 的 event 9 为模型 MTB（mode 0），补充与原 MTB 一致的蓝色背景 #287FA1、alpha 0.14；阴影为显示窗相对时间 336–426 ms，即参与触点招募起止 348–414 ms 的两端各加 12 ms。既有时间窗、事件分类和波形数组未改变。', '',
        '科学读出保持原合同：F 为虚拟接触发放密度的带通读出；G 为冻结标签的平均 rank；H 是另一种触点交叉分组的匹配统计；I 是训练最优值轨迹，不据此增加机制恢复或优化器优越性的结论。']
    (OUT/'panel_map.md').write_text('\n'.join(table)+'\n')
    print(FIG,flush=True)


if __name__=='__main__':main()
