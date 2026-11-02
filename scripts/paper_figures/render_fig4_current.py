#!/usr/bin/env python3
"""Redraw accepted Figure 4 from bundled arrays into a separate output directory."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.transforms import Bbox
from matplotlib.text import Text
import numpy as np

PARENT = ROOT / 'results/paper-ready-figure/fig4'
OUT = None
SOURCE = PARENT / 'source'
FIGURES = None
W, H = 320., 232.52
RED, BLUE = '#D84840', '#347FB3'
FONT = dict(label=10.5, tick=9.5, dense=8., legend=9., title=10., letter=15.)
BOXES = {
    'A': (5, 154.52, 76, 70), 'B': (93, 160.52, 59, 59),
    'C': (164.43, 144.765915178571, 151, 80.790178571429),
    'D': (9, 71.05, 144, 78), 'E': (171, 71.05, 144, 78),
    'F': (19, 15, 53, 51), 'G': (95, 15, 48, 51),
    'H': (174, 15, 53, 51), 'I': (255, 15, 51, 51),
}
LETTERS = {'A': (4, 227.52), 'B': (80, 227.52), 'C': (164, 227.52),
           'D': (4, 149.33), 'E': (164, 149.33),
           'F': (4, 73), 'G': (80, 73), 'H': (163, 73), 'I': (246, 73)}
SNAPSHOTS = ['overview_neuron_samples.npz', 'rigid_contact_geometry.npz',
             'D_position_density.npz', 'D_position_contours.json',
             'D_position_electrodes.json', 'D_position_straight_electrodes.json',
             'D_position_selection.json', 'B_visible_workpoints.csv',
             'core_count_plot_arrays.npz', 'core_count_input_snapshot.json',
             'core_count_mean_std_curves.csv']



def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rect(box):
    x, y, w, h = box
    return [x/W, y/H, w/W, h/H]


def axis_style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9.5, length=2.5, width=.6, pad=2)
    ax.xaxis.label.set_fontsize(10.5)
    ax.yaxis.label.set_fontsize(10.5)
    ax.xaxis.labelpad = ax.yaxis.labelpad = 2


def load_helper(name):
    path = SOURCE / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'fig4_revision_{name}', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def draw_a(fig):
    return load_helper('ab_panel').draw_a(fig,rect(BOXES['A']))

def draw_b(fig):
    return load_helper('ab_panel').draw_b(fig,rect(BOXES['B']))


def draw_f(fig):
    arrays=dict(np.load(SOURCE/'core_count_plot_arrays.npz'))
    summary=read(SOURCE/'core_count_input_snapshot.json')['four_core_case']['summary']
    case={'summary':summary,'arrays':{i:arrays[f'core{i}'] for i in range(1,5)}}
    ax=fig.add_axes(rect(BOXES['F']))
    progress.draw(ax,case,show_title=False,paper_style=FONT,show_legend=True,summary_style='mean_std')
    ax.set(yticks=[0,2,4,6,8],xticks=[1,11,21,31]);axis_style(ax)
    rows=list(csv.DictReader(open(SOURCE/'core_count_mean_std_curves.csv')))
    for k,line in enumerate(ax.lines,1):
        expected=[r for r in rows if int(r['core_count'])==k and r['figure']=='e1146_one_two_three_four_core_loss']
        np.testing.assert_allclose(line.get_ydata(),[float(r['mean']) for r in expected],atol=1e-12)
        np.testing.assert_allclose(case['arrays'][k][:,:31,0].std(0,ddof=1),[float(r['std']) for r in expected],atol=1e-12)
    return {'old_panel':'C','core_counts':[1,2,3,4],'replicates':4,'epochs':31,'band':'sample SD ddof=1','values_unchanged':True}


def draw_position_density(fig):
    arrays=dict(np.load(SOURCE/'H_position_density.npz'))
    arrays['core_components_per_mm2']=arrays['per_core_density_per_mm2']/2
    geometry=read(SOURCE/'D_position_electrodes.json')
    straight=read(SOURCE/'D_position_straight_electrodes.json')
    inds=[straight['names'].index(n) for n in geometry['contact_names']]
    geometry['straight_contact_xy_mm']=np.array(straight['rod_sheet_xy'])[inds].tolist()
    geometry['density_contours']=read(SOURCE/'H_position_contours.json')['contours']
    pixels,cmap,norm,labels=parameters.completed_density_layer(arrays,geometry)
    ax=fig.add_axes(rect(BOXES['H']))
    ax.imshow(pixels,extent=(1,19,1,19),origin='upper',aspect='equal')
    ax.set(xlim=(1,19),ylim=(1,19),xticks=[5,10,15],yticks=[5,10,15],xlabel='x (mm)',ylabel='y (mm)');axis_style(ax)
    for xy,label in zip(labels,['Core A','Core B']):ax.text(*xy,label,fontsize=9.5,ha='center',va='bottom',color='#403949')
    # Relief height shifts the apparent y coordinate of a projected summit.
    # Explicitly register each density maximum on the physical ground plane.
    ground_peaks=[]
    for component,top in zip(arrays['per_core_density_per_mm2'],labels):
        iy,ix=np.unravel_index(np.argmax(component),component.shape)
        ground=np.array([arrays['x_mm'][ix],arrays['y_mm'][iy]])
        ground_peaks.append(ground.tolist())
        ax.plot([ground[0],ground[0]],[ground[1],top[1]-.25],color='#655D70',ls=(0,(2,2)),lw=.5,zorder=5)
        ax.plot(*ground,marker='+',ms=4.2,mew=.8,color='#403949',zorder=6)
    handles=[Line2D([],[],marker='o',ls='none',mfc='#B6B2BF',mec='none',ms=2.5),
             Line2D([],[],marker='o',ls='none',mfc='none',mec='#65509A',mew=.65,ms=3.2),
             Line2D([],[],color='#9784B6',lw=.6),
             Line2D([],[],marker='+',color='#403949',ls='none',ms=4,mew=.8)]
    ax.legend(handles,['Search','Outlier','90% mass','Peak (ground)'],loc='upper right',fontsize=8.2,frameon=False,borderpad=.2,labelspacing=.1,handlelength=.7,handletextpad=.25,borderaxespad=.1)
    bar=fig.add_axes(rect((229,15,1.5,51)))
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=bar,ticks=[0,.2,.4])
    cb.outline.set_visible(False);axis_style(bar);bar.tick_params(pad=1)
    cb.set_label(r'Position density (mm$^{-2}$)',fontsize=9,labelpad=1)
    selection=read(SOURCE/'H_position_selection.json')
    np.testing.assert_array_equal(ground_peaks,selection['ground_peaks_mm'])
    return {'old_panel':'E','renderer':'unchanged completed_density_layer with ground-coordinate peak registration',
        'selected':selection['selection']['count'],'scorable':selection['pool_scorable'],
        'selection_metric':'J','formula':selection['formula'],'values_unchanged':False,
        'selection_source':'source/H_position_selection.json','validation':'source/h_loss_density_validation.json',
        'cutoff_J':selection['selection']['cutoff_J'],'retained_from_previous':selection['retained_from_previous'],
        'ground_peaks_mm':ground_peaks,'relief_note':'Raised summits shift under oblique projection; added crosses mark true XY density modes and dashed stems distinguish height from position.'}


def draw_loss_profile(fig):
    rows=list(csv.DictReader((SOURCE/'g_training_loss_profiles.csv').open()))
    assert len(rows)==194
    stages=sorted({int(row['stage']) for row in rows})
    assert stages==list(range(6,17))
    palette=LinearSegmentedColormap.from_list('visible_stage_purples',
        ['#E0D6F2','#B59ADB','#8260B4','#4D287F'])(np.linspace(0,1,len(stages)))
    ax=fig.add_axes(rect(BOXES['G']))
    ordered=sorted(rows,key=lambda r:(int(r['stage']),r['candidate']))
    for row in ordered:
        values=[float(row[k]) for k in ('L_r','L_o','L_p')]
        np.testing.assert_allclose(np.mean(values),float(row['J']),rtol=0,atol=1e-12)
        color_index=int(row['stage'])-stages[0]
        line,=ax.plot([0,1,2],values,color='#A0A0AA',lw=.50,alpha=.75,zorder=1)
        line.set_gid('g_profile_line_'+row['candidate'])
    # Draw every colored marker above every neutral connector.
    for row in ordered:
        values=[float(row[k]) for k in ('L_r','L_o','L_p')]
        color_index=int(row['stage'])-stages[0]
        points,=ax.plot([0,1,2],values,ls='none',marker='o',ms=2.7,
            mfc=palette[color_index],mec='#65508A',mew=.3,alpha=1.,zorder=3+color_index/20)
        points.set_gid('g_profile_points_'+row['candidate'])
    axis_style(ax)
    ax.set(xlim=(-.18,2.12),ylim=(0,9.5),yticks=[0,3,6,9],xticks=[0,1,2],ylabel='Loss')
    ax.set_xticklabels(['Rank\n'+r'$\mathcal{L}_r$','Order\n'+r'$\mathcal{L}_o$','Participation\n'+r'$\mathcal{L}_p$'],fontsize=10.0)
    for t,col in zip(ax.get_xticklabels(),['#3D5C6F','#E47159','#6B7546']):t.set_color(col)
    # Display stages start at zero; original stage IDs stay frozen in the CSV.
    display_stages=np.arange(len(stages))
    cmap=ListedColormap(palette);norm=BoundaryNorm(np.arange(-.5,len(stages)),len(stages))
    bar=fig.add_axes(rect((145,15,1.5,51)))
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=bar,ticks=[0,2,4,6,8,10])
    cb.outline.set_visible(False);axis_style(bar);bar.tick_params(pad=1)
    cb.set_label('Search stage',fontsize=9,labelpad=1)
    validation=read(SOURCE/'g_loss_validation.json')
    assert validation['status']=='PASS'
    return {'old_panel':'B','profiles':194,'stages':list(range(6,17)),
        'metric':'Actual normalized training components L_r/L_o/L_p',
        'formula':'J = (L_r + L_o + L_p) / 3',
        'source':'source/g_training_loss_profiles.csv','validation':'source/g_loss_validation.json',
        'display_stages':display_stages.tolist(),'display_stage_rule':'original stage minus 6',
        'color_order':{'stages':stages,'normalized_positions':np.linspace(0,1,len(stages)).tolist(),
            'rgba':palette.tolist(),'mapping':'(stage - 6) / (16 - 6)',
            'marker_alpha':1.0,'marker_face_equals_colorbar':True,'draw_order':'all gray connectors, then markers light to dark',
            'marker_size_pt':2.7,'marker_edge_width_pt':.3,'marker_edge_color':'#65508A',
            'connector_color':'#A0A0AA','connector_width_pt':.50,'connector_alpha':.75,
            'x_axis_limits':[-.18,2.12],'first_category_inset_mm':48*.18/2.30},
        'stage_interpretation':validation['stage_interpretation']}


def placeholder(fig,key):
    ax=fig.add_axes(rect(BOXES[key]));ax.set_axis_off()
    ax.text(.5,.5,f'{key}: source preparation in progress',ha='center',va='center',color='#888888',fontsize=10)


def build(draft=False):
    global progress, parameters
    FIGURES.mkdir(parents=True,exist_ok=True)
    formal_hashes={str(p):sha(p) for p in PARENT.rglob('*') if p.is_file() and '__pycache__' not in str(p) and 'candidates' not in p.parts}
    progress=load_helper('fgh_progress_source')
    parameters=load_helper('fgh_density_source')
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':.65,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=140)
    checks={};groups={}
    funcs={'A':draw_a,'B':draw_b,'F':draw_f,'G':draw_loss_profile,'H':draw_position_density}
    if not draft or (SOURCE/'c_panel_data.npz').exists():
        cm=load_helper('c_panel')
        funcs['C']=lambda f:cm.render_c(f,rect(BOXES['C']))
    if not draft or (SOURCE/'i_angle_response.csv').exists():
        im=load_helper('i_panel')
        funcs['I']=lambda f:im.render_i(f,rect(BOXES['I']))
    for key in 'ABCFGHI':
        before=set(fig.axes);nt=len(fig.texts);na=len(fig.artists)
        checks[key]=funcs[key](fig) if key in funcs else placeholder(fig,key)
        groups[key]={'axes':[a for a in fig.axes if a not in before],'texts':fig.texts[nt:],'artists':fig.artists[na:]}
    before=set(fig.axes);nt=len(fig.texts)
    if not draft:
        dm=load_helper('render_native_de')
        checks['DE']=dm.render_de(fig,rect(BOXES['D']),rect(BOXES['E']))
    else:
        for key in 'DE':placeholder(fig,key)
    for key in 'DE':
        x,y,w,h=rect(BOXES[key]);axes=[a for a in fig.axes if a not in before and x-.04<a.get_position().x0<x+w]
        groups[key]={'axes':axes,'texts':[t for t in fig.texts[nt:] if x-.04<t.get_position()[0]<x+w]}
    panel_letters=[]
    for key,(x,y) in LETTERS.items():panel_letters.append(fig.text(x/W,y/H,key,fontsize=15,weight='bold',va='top'))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    # Measure complete row bounds, including titles, labels, colorbars and letters.
    row_bounds=[]
    for keys in ['ABC','DE','FGHI']:
        bounds=[]
        for key in keys:
            group=groups[key]
            bounds.extend(a.get_tightbbox(renderer) for a in group['axes'])
            bounds.extend(t.get_window_extent(renderer) for t in group['texts'])
            bounds.append(panel_letters['ABCDEFGHI'.index(key)].get_window_extent(renderer))
        bb=Bbox.union([b for b in bounds if b is not None])
        row_bounds.append(dict(panels=keys,bounds_mm=(np.array(bb.extents)/fig.dpi*25.4).tolist()))
    row_gaps=[row_bounds[i]['bounds_mm'][1]-row_bounds[i+1]['bounds_mm'][3] for i in range(2)]
    if not draft:assert all(gap>1.8 for gap in row_gaps),row_gaps
    (OUT/'row_layout_audit.json').write_text(json.dumps(dict(size_mm=[W,H],row_bounds=row_bounds,
        clear_gaps_mm=row_gaps,layout_operation='Preserve all data, backgrounds and geometry; enlarge D/E raster E/I labels, increase padding and use A semantic colors'),indent=2)+'\n')
    text_bounds=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():continue
        if t.axes is not None and not t.axes.axison and t not in t.axes.texts:continue
        bb=t.get_window_extent(renderer)
        text_bounds.append({'text':t.get_text(),'bounds_mm':(np.array(bb.extents)/fig.dpi*25.4).tolist()})
    outside=[t for t in text_bounds if min(t['bounds_mm'][:2])<-.3 or t['bounds_mm'][2]>W+.3 or t['bounds_mm'][3]>H+.3]
    (OUT/'text_layout_audit.json').write_text(json.dumps({'size_mm':[W,H],'outside_canvas':outside,'text_bounds':text_bounds},ensure_ascii=False,indent=2)+'\n')
    name='fig4-layout-work-in-progress' if draft else 'fig4-complete-layout'
    for ext in ['png','pdf','svg']:
        fig.savefig(FIGURES/f'{name}.{ext}',dpi=300 if ext=='png' else 180)
    fig.savefig(FIGURES/f'{name}-preview.png',dpi=145)
    if not draft:
        # Export the revised D/E row directly from the same native figure.
        de_axes=groups['D']['axes']+groups['E']['axes']
        de_texts=groups['D']['texts']+groups['E']['texts']+[panel_letters[3],panel_letters[4]]
        for a in fig.axes:a.set_visible(a in de_axes)
        for t in fig.texts:t.set_visible(t in de_texts)
        for artist in fig.artists:artist.set_visible(False)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        de_bounds=Bbox.union([a.get_tightbbox(renderer) for a in de_axes]+[t.get_window_extent(renderer) for t in de_texts]).expanded(1.015,1.045)
        for ext in ['png','pdf','svg']:
            fig.savefig(FIGURES/f'fig4-panels-de.{ext}',bbox_inches=de_bounds.transformed(fig.dpi_scale_trans.inverted()),dpi=220 if ext=='png' else 180)
        for t in panel_letters:t.set_visible(False)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        for key,group in groups.items():
            # Standalone exports contain only this panel, without neighbouring labels or zoom connectors.
            for a in fig.axes:a.set_visible(a in group['axes'])
            for t in fig.texts:t.set_visible(t in group['texts'])
            for artist in fig.artists:artist.set_visible(key=='C' and artist in group.get('artists',[]))
            fig.canvas.draw();renderer=fig.canvas.get_renderer()
            bboxes=[a.get_tightbbox(renderer) for a in group['axes']]
            bboxes += [t.get_window_extent(renderer) for t in group['texts']]
            bounds=Bbox.union([b for b in bboxes if b is not None]).expanded(1.025,1.045)
            inches=bounds.transformed(fig.dpi_scale_trans.inverted())
            for ext in ['png','pdf','svg']:fig.savefig(FIGURES/f'fig4-panel{key.lower()}.{ext}',bbox_inches=inches,dpi=300 if ext=='png' else 180)
    plt.close(fig)
    for path,digest in formal_hashes.items():assert sha(path)==digest,path
    if isinstance(checks.get('C'),dict):checks['C']=checks['C']['metadata']
    if 'DE' in checks:checks['DE']={k:v for k,v in checks['DE'].items() if k!='axes'}
    if (SOURCE/'i_metadata.json').exists():checks['I']=read(SOURCE/'i_metadata.json')
    metadata={'version':'snn_layout_ei_labels_20261031','status':'WORK_IN_PROGRESS' if draft else 'REDRAW_NOT_A_NEW_AUTHOR_ACCEPTANCE','figure':'Fig4','size_mm':[W,H],'panel_mapping':{'A':'new 3D local circuit','B':'spatial sheet with local Gaussian electrode sampling inset','C':'model/patient TFR and loss','D':'native SNN without core','E':'native SNN with one core','F':'old C','G':'old B workpoints, actual three training-loss components','H':'same old E pool, position density reselected by three-component training J','I':'old D angles, actual three training-loss components and total J with SD'},'checks':checks,'formal_package_unchanged':True,'sources':{name:sha(SOURCE/name) for name in SNAPSHOTS},'source_script':str(Path(__file__).resolve()),'panel_boxes_mm':BOXES}
    metadata.update(previous_candidate='snn_layout_core_only_fill_20261031', changed_panels=['D','E'], scientific_change=None, display_changes={'raster_group_labels':'Larger bold E/I labels with increased axis padding; E red and I blue match panel A'}, caption_file='captions_fig4_fig5_zh.tex')
    metadata['text_outside_canvas']=outside
    metadata['helper_sources']={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.glob('*') if p.suffix in ('.py','.npz','.json','.csv')}
    (OUT/'candidate_registry.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,default=str)+'\n')
    print(FIGURES/f'{name}-preview.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    OUT=args.output_dir.resolve()
    if OUT==PARENT.resolve() or PARENT.resolve() in OUT.parents:
        raise ValueError('Redraw into a directory outside the accepted package.')
    if OUT.exists() and any(OUT.iterdir()):
        raise ValueError('The output directory must be empty.')
    FIGURES=OUT/'figures'
    build(False)
