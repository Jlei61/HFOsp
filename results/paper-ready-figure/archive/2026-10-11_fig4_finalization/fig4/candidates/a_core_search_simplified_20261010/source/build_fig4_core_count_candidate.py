#!/usr/bin/env python3
"""Assemble the A–J candidate: B profile and new C core-count loss panel.

Reuse the author-designated Fig4 renderers and a fixed progress snapshot.
Never advance the live experiment or change current_version.json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.paper_figures import build_fig4_readable_complete as plate
from scripts.paper_figures import plot_core_count_optimization_progress as progress
from scripts.paper_figures import build_fig4_template_recovery_by_mode as recovery
from scripts.paper_figures import fig4_parameter_response_assets as parameters
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

OUT=ROOT/'results/paper-ready-figure/fig4/candidates/core_count_panel_c_20260929'
SNAPSHOT=progress.DEFAULT_OUT/'snapshots/20260929_162510/input_snapshot.json'
REFLOW_COLUMNS=True
LETTERS={'A':'A','B':'B','core_count':'C',**dict(zip('CDEFGHI','DEFGHIJ'))}
TITLES={
    'A':'局部 E/I 回路与患者电极空间基底',
    'B':'参数搜索中的三类误差（仅原左侧 panel）',
    'C':'E1146 单核与双核的最佳训练 loss',
    'D':'向外 EE／方向响应、低误差位置密度及核内 EE 三误差响应',
    'E':'较高误差工作点的传播示例（原 D；lf_bo04_07）',
    'F':'较低误差工作点的传播示例（原 E；xy_left_20）',
    'G':'连续 30–80 Hz 虚拟接触活动（原 F）',
    'H':'模型与患者的平均传播 rank（原 G）',
    'I':'模型—患者触点交叉匹配（原 H）',
    'J':'逐患者 TA–MTA 与 TB–MTB 模板相似度（沿用当前图例）',
}


def frozen_case(snapshot_path=SNAPSHOT,core_counts=(1,2),summary_style='median_iqr'):
    """Restore the exact attached curves, without reading later live scores."""
    path=plate.freeze(snapshot_path)
    shutil.copyfile(path,plate.SOURCE/'core_count_input_snapshot.json')
    snapshot=plate.read(path)
    reducer_path=plate.freeze(snapshot['reducer'])
    assert plate.sha(reducer_path)==snapshot['reducer_sha256']
    reduce=progress.original_reducer()
    case=next(c for c in snapshot['cases'] if c['summary']['subject']=='epilepsiae_1146')
    if 4 in core_counts:case=snapshot['four_core_case']
    elif 3 in core_counts:case=snapshot['three_core_case']
    summary=case['summary'];arrays={}
    for k in core_counts:
        reduced=[reduce(rows,summary['planned_epochs']) for rows in case['records'][str(k)]]
        assert [n for _,n in reduced]==summary['prefix_by_restart'][str(k)]
        arrays[k]=np.asarray([a for a,_ in reduced])
        # Independent selection from raw records checks the plotted best-so-far values.
        for r,rows in enumerate(case['records'][str(k)]):
            for epoch in range(1,summary['prefix_by_restart'][str(k)][r]+1):
                scores=[x['J'] for x in rows if x['epoch']<=epoch and x['J'] is not None]
                if scores:np.testing.assert_allclose(arrays[k][r,epoch-1,0],min(scores),atol=0,rtol=0)
        np.testing.assert_allclose(np.median(arrays[k][:,summary['common_epoch']-1,0]),
                                   summary['median_best_loss'][str(k)],atol=1e-12,rtol=0)
    np.savez_compressed(plate.SOURCE/'core_count_plot_arrays.npz',**{f'core{k}':a for k,a in arrays.items()})
    if summary_style=='mean_std':
        summary['mean_best_loss']={str(k):float(np.mean(a[:,summary['common_epoch']-1,0])) for k,a in arrays.items()}
        summary['std_best_loss']={str(k):float(np.std(a[:,summary['common_epoch']-1,0],ddof=1)) for k,a in arrays.items()}
        summary['std_ddof']=1
    return dict(summary=summary,records=case['records'],arrays=arrays,summary_style=summary_style),snapshot


def validate_new_panel(fig,case):
    fig.canvas.draw()
    b=plate.GROUPS['B']['axes'][0];c=plate.GROUPS['core_count']['axes'][0]
    bb,cb=b.get_window_extent(),c.get_window_extent()
    np.testing.assert_allclose([bb.y0,bb.y1],[cb.y0,cb.y1],atol=1e-7,rtol=0)
    assert not any(a.name=='3d' for a in fig.axes)
    assert c.get_title()=='' and not plate.GROUPS['core_count']['texts']
    legend=c.legend_.get_window_extent()
    assert cb.x0<=legend.x0<legend.x1<=cb.x1 and cb.y0<=legend.y0<legend.y1<=cb.y1
    assert [t.get_text() for t in c.legend_.get_texts()]==[f'{k} core'+('s' if k>1 else '') for k in sorted(case['arrays'])]
    assert c.xaxis.label.get_fontsize()==c.yaxis.label.get_fontsize()==plate.FONT['label']
    assert all(t.get_fontsize()==plate.FONT['legend'] for t in c.legend_.get_texts())
    # Matplotlib artists must retain the exact original data despite new point sizes.
    offset=0
    for k in sorted(case['arrays']):
        block=case['arrays'][k][:,:,0]
        if case.get('summary_style')=='mean_std':
            x,mean,sd=progress.mean_std_trajectory(block,case['summary']['common_epoch'])
            np.testing.assert_array_equal(c.lines[offset].get_xdata(),x)
            np.testing.assert_array_equal(c.lines[offset].get_ydata(),mean)
            band,=[b for b in c.collections if b.get_gid()==f'core{k}_sample_sd']
            vertices=np.concatenate([p.vertices for p in band.get_paths()])
            for epoch,low,high in zip(x,mean-sd,mean+sd):
                values=vertices[vertices[:,0]==epoch,1]
                np.testing.assert_array_equal([values.min(),values.max()],[low,high])
            assert not c.lines[offset].get_path().transformed(c.lines[offset].get_transform()).intersects_bbox(legend,filled=False)
            assert not any(p.transformed(band.get_transform()).intersects_bbox(legend,filled=True) for p in band.get_paths())
            offset+=1
            continue
        for line,row in zip(c.lines[offset:offset+len(block)],block):
            finite=np.flatnonzero(np.isfinite(row))
            np.testing.assert_array_equal(line.get_xdata(),finite+1)
            np.testing.assert_array_equal(line.get_ydata(),row[finite])
        common=case['summary']['common_epoch']
        np.testing.assert_array_equal(c.lines[offset+len(block)].get_xdata(),np.arange(1,common+1))
        np.testing.assert_array_equal(c.lines[offset+len(block)].get_ydata(),np.median(block[:,:common],axis=0))
        offset+=len(block)+1
    assert len(c.lines)==offset
    renderer=fig.canvas.get_renderer()
    b_right=max(a.get_tightbbox(renderer).x1 for a in plate.GROUPS['B']['axes'])
    c_left=c.get_tightbbox(renderer).x0
    assert c_left>b_right
    corner=b.transAxes.transform((0,0))
    for spine in ('left','bottom'):
        s=b.spines[spine]
        np.testing.assert_allclose(s.get_transform().transform(s.get_path().vertices)[0],corner,atol=1e-6)
    outside=[]
    for t in plate.visible_texts(fig):
        box=t.get_window_extent(renderer)
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.x1+.5 or box.y1>fig.bbox.y1+.5:
            outside.append(t.get_text())
    assert not outside,outside
    return dict(BC_equal_y_edges=True,C_axis_size_mm=[cb.width/fig.dpi*25.4,cb.height/fig.dpi*25.4],C_no_title_or_caption=True,
                C_legend_inside=True,C_artist_data_exact=True,C_original_reducer_independently_checked=True,
                BC_visible_gap_mm=(c_left-b_right)/fig.dpi*25.4,B_axis_corner_gap_px=0,
                no_3d_axes=True,text_outside_canvas=outside)


def panel_d(fig,curves,geometry,workpoints):
    """Three single-axis error responses, with position density at lower right."""
    selected_cases=plate.read(plate.freeze(plate.PREVIOUS/'source/selected_workpoints.json'))['cases']
    references=parameters.workpoint_references(plate,{LETTERS[c['panel']]:c for c in selected_cases})
    lookup={r['candidate']:r for r in workpoints}
    assert len(lookup)==len(workpoints)
    sweep_reference=np.array(json.loads(lookup['g1_axis3_plus']['vector']))
    plotted=[];upper=[]
    for axis,xpos,xlabel,xticks,parameter in [
            (4,13,'Outward E→E gain',[1,1.25,1.5],'EE_core_to_out_scale'),
            (5,67,'E→E angle (°)',[-20,0,20],'EE_angle_offset_deg')]:
        selected=sorted((r for r in curves if int(r['axis'])==axis and r['observable']=='local_order'),
                        key=lambda r:float(r['parameter_value']))
        rank={r['candidate']:r for r in curves if int(r['axis'])==axis and r['observable']=='rank'}
        assert len(selected)==13
        samples=[]
        for r in selected:
            source=lookup[r['candidate']]
            x=float(r['parameter_value'])
            assert int(r['N'])==source['N']
            np.testing.assert_allclose(json.loads(source['vector'])[axis],x,atol=1e-12,rtol=0)
            np.testing.assert_allclose(np.delete(json.loads(source['vector']),axis),
                                       np.delete(sweep_reference,axis),atol=1e-12,rtol=0)
            values=[float(rank[r['candidate']]['value']),float(r['value']),source['participation_error']]
            np.testing.assert_allclose(values,[source[k] for k in plate.base.KEYS],atol=1e-12,rtol=0)
            samples.append(dict(axis=axis,parameter_value=x,candidate=r['candidate'],
                                **dict(zip(parameters.METRICS,values)),N=source['N'],
                                source=source['source'],source_sha256=source['source_sha256']))
        ax=fig.add_axes(plate.rect(xpos,plate.MID_ROWS['D'],40,40))
        parameters.error_curves(plate,ax,[r['parameter_value'] for r in samples],
                                [[r[k] for r in samples] for k in parameters.METRICS],
                                xlabel,xticks,references[parameter])
        if axis==4:ax.set_xticklabels(['1.0','1.25','1.5'])
        upper.append(ax);plotted.extend(samples)
    with (plate.SOURCE/'D_connection_response.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(plotted[0]));writer.writeheader();writer.writerows(plotted)
    plate.CHECKS['D_connection_response']=dict(metrics=parameters.METRICS,condition_count=26,
        rendered_values_equal_frozen_workpoints=True,shared_y_limits=[0,.42],
        fixed_other_vector_coordinates_verified=True,sweep_reference_candidate='g1_axis3_plus',
        sweep_reference_vector=sweep_reference.tolist())
    plate.CHECKS['D_error_palette']=plate.ERROR_COLORS.copy()
    error,position,bar=parameters.draw_lower(plate,fig,references)
    # Artist roles replace the historical positional assumptions about twins.
    return {'D':upper,'E':[error]}


def validate_parameter_layout(fig):
    axes=plate.GROUPS['C']['axes']
    assert len(axes)==5  # Three error axes, one spatial axis, one colourbar.
    gain,angle,error,position,bar=axes
    def box(a):
        return np.array(a.get_position().bounds)*[plate.W,plate.H,plate.W,plate.H]
    np.testing.assert_allclose(box(gain)[[0,2]],box(error)[[0,2]],atol=1e-7,rtol=0)
    full_right_column='D_right_column_layout' in plate.CHECKS
    if full_right_column:
        np.testing.assert_allclose(box(angle)[0],box(position)[0],atol=1e-7,rtol=0)
        right=bar.get_tightbbox(fig.canvas.get_renderer()).x1/fig.dpi*25.4
        np.testing.assert_allclose(box(angle)[0]+box(angle)[2],right,atol=1e-7,rtol=0)
    else:
        np.testing.assert_allclose(box(angle)[0]+box(angle)[2]/2,
                                   box(position)[0]+box(position)[2]/2,atol=1e-7,rtol=0)
    density_size=plate.CHECKS['D_lower_axis_height_mm']['position_density']
    np.testing.assert_allclose(box(position)[2:],[density_size,density_size],atol=1e-7,rtol=0)
    error_size=box(gain)[2:]
    np.testing.assert_allclose(box(error)[2:],error_size,atol=1e-7,rtol=0)
    if full_right_column:
        np.testing.assert_allclose(box(angle)[3],error_size[1],atol=1e-7,rtol=0)
    else:
        np.testing.assert_allclose(box(angle)[2:],error_size,atol=1e-7,rtol=0)
    for a in (gain,angle,error):
        assert a.get_ylabel()=='Error' and not a.spines['right'].get_visible()
        assert all(not t.label2.get_visible() for t in a.yaxis.get_major_ticks())
        assert [t.get_text() for t in a.legend_.get_texts()]==list(parameters.ERROR_LABELS)
        ab=a.get_window_extent();lb=a.legend_.get_window_extent()
        assert ab.contains(lb.x0,lb.y0) and ab.contains(lb.x1,lb.y1)
        for line in a.lines[:3]:
            assert np.all((np.asarray(line.get_ydata())>=0)&(np.asarray(line.get_ydata())<=.42))
            assert not line.get_path().transformed(line.get_transform()).intersects_bbox(lb,filled=False)
    references=plate.CHECKS['D_EF_workpoint_references']['reference_lines']
    for a,key in zip((gain,angle,error),parameters.REFERENCE_PARAMETERS):
        expected=references[key]
        assert len(a.lines)==3+len(expected)
        assert [t.get_text() for t in a.texts]==[label for label,_ in expected]
        for line,(label,value) in zip(a.lines[3:],expected):
            np.testing.assert_array_equal(line.get_xdata(),[value,value])
    plate.CHECKS['D_EF_workpoint_references']['rendered_lines_match_executed_parameters']=True
    plate.CHECKS['D_parameter_layout']=dict(single_left_error_axes=3,error_axis_size_mm=error_size.tolist(),
        left_column_x_and_width_equal=True,right_column_centres_equal=not full_right_column,
        right_column_including_colourbar_aligned=full_right_column,angle_axis_size_mm=box(angle)[2:].tolist(),
        density_axis_size_mm=[density_size,density_size],legend_labels=parameters.ERROR_LABELS,legends_inside=True,
        legends_do_not_cover_observed_curves=True,
        artist_bounds_mm={name:box(a).tolist() for name,a in zip(
            ['outward_EE','EE_angle','within_core_EE','position_density','density_colourbar'],axes)})


def reflow_columns(fig):
    """Fit A to D's visible column and align B with E/F/H's left axes."""
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    def visible_box(key):
        g=plate.GROUPS[key]
        boxes=[a.get_tightbbox(renderer) for a in g['axes']]
        boxes += [t.get_window_extent(renderer) for t in g['texts']]
        return Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
    target=visible_box('C');before=visible_box('A')
    group=plate.GROUPS['A'];local,spatial=group['axes']
    original=[a.get_position().frozen() for a in group['axes']]
    left=target.x0/plate.MM
    lx,ly,lw,lh=np.array(original[0].bounds)*[plate.W,plate.H,plate.W,plate.H]
    sx,sy,sw,sh=np.array(original[1].bounds)*[plate.W,plate.H,plate.W,plate.H]
    tight=spatial.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
    label_width=sx-tight.x0/plate.MM
    right_overhang=tight.x1/plate.MM-(sx+sw)
    new_sx=left+lw+label_width+1.2
    side=target.x1/plate.MM-right_overhang-new_sx
    assert 40<side<sw
    plate.pos(local,left,ly,lw,lh)
    plate.pos(spatial,new_sx,sy,side,side)
    for t in group['texts']:
        x,y=t.get_position();t.set_position((x+(left-lx)/plate.W,y))
    # Each connector endpoint stays attached to its own image when A changes.
    updated=[a.get_position().frozen() for a in group['axes']]
    for line in group['artists']:
        points=np.column_stack([line.get_xdata(),line.get_ydata()])
        for i,(old,new) in enumerate(zip(original,updated)):
            points[i]=np.array([new.x0,new.y0])+(
                points[i]-[old.x0,old.y0])*[new.width/old.width,new.height/old.height]
        line.set_data(points[:,0],points[:,1])
    b=plate.GROUPS['B']['axes'][0]
    bx,by,bw,bh=np.array(b.get_position().bounds)*[plate.W,plate.H,plate.W,plate.H]
    bar=plate.GROUPS['B']['axes'][1]
    bar_x,bar_y,bar_w,bar_h=np.array(bar.get_position().bounds)*[plate.W,plate.H,plate.W,plate.H]
    c=plate.GROUPS['core_count']['axes'][0]
    cx,cy,cw,ch=np.array(c.get_position().bounds)*[plate.W,plate.H,plate.W,plate.H]
    # Equal B/C boxes share the available width, retaining the colourbar's
    # spacing. All three top-row data axes now use A's bottom and top edges.
    separation=cx-(bx+bw)
    bc_width=(cx+cw-130-separation)/2
    new_cx=130+bc_width+separation
    new_bar_x=130+bc_width+(bar_x-bx-bw)
    plate.pos(b,130,sy,bc_width,side)
    plate.pos(bar,new_bar_x,sy,bar_w,side)
    plate.pos(c,new_cx,sy,bc_width,side)
    h=plate.GROUPS['G']['axes'][0]  # Displayed H.
    hx,hy,hw,hh=np.array(h.get_position().bounds)*[plate.W,plate.H,plate.W,plate.H]
    plate.pos(h,130,hy,hw,hh)
    plate.CHECKS['column_reflow']=dict(A_previous_visible_x_mm=[before.x0/plate.MM,before.x1/plate.MM],
        AD_target_visible_x_mm=[target.x0/plate.MM,target.x1/plate.MM],
        A_spatial_axis_size_mm=[side,side],A_local_circuit_size_unchanged=True,
        A_external_fonts_unchanged=True,A_connectors_follow_both_images=True,
        B_axis_width_mm=bc_width,C_axis_width_mm=bc_width,
        BC_added_width_mm={'B':bc_width-bw,'C':bc_width-cw},
        ABC_common_axis_height_mm=side,C_left_axis_x_mm=new_cx,C_letter_x_mm=new_cx-11,
        B_colourbar_left_mm=new_bar_x,BEFH_left_axis_x_mm=130.,BEFH_letter_x_mm=122.)


def validate_columns(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    def horizontal_bounds(key):
        g=plate.GROUPS[key]
        bb=Bbox.union([a.get_tightbbox(renderer) for a in g['axes']]+
                      [t.get_window_extent(renderer) for t in g['texts']])
        return np.array([bb.x0,bb.x1])/fig.dpi*25.4
    np.testing.assert_allclose(horizontal_bounds('A'),horizontal_bounds('C'),atol=1e-7,rtol=0)
    for key in ('B','D','E','G'):
        a=plate.GROUPS[key]['axes'][0]
        np.testing.assert_allclose(a.get_position().x0*plate.W,130.,atol=1e-7,rtol=0)
    letters=[t for t in fig.texts if t.get_text() in ('B','E','F','H')]
    assert len(letters)==4
    np.testing.assert_allclose([t.get_position()[0]*plate.W for t in letters],122.,atol=1e-7,rtol=0)
    b=plate.GROUPS['B']['axes'][0];c=plate.GROUPS['core_count']['axes'][0]
    np.testing.assert_allclose([b.get_position().width*plate.W,c.get_position().width*plate.W],
                               [55.5,55.5],atol=1e-7,rtol=0)
    np.testing.assert_allclose(c.get_position().x1*plate.W,274,atol=1e-7,rtol=0)
    local,spatial=plate.GROUPS['A']['axes']
    top_axes=[spatial,b,c,plate.GROUPS['B']['axes'][1]]
    edges=np.array([[a.get_window_extent().y0,a.get_window_extent().y1] for a in top_axes])
    np.testing.assert_allclose(edges,np.broadcast_to(edges[0],edges.shape),atol=1e-7,rtol=0)
    # Inspect rendered bottom spines as well as axis boxes: this is the
    # baseline the reader sees, independent of the text below each axis.
    spine_y=[a.spines['bottom'].get_transform().transform(a.spines['bottom'].get_path().vertices)[:,1]
             for a in (spatial,b,c)]
    np.testing.assert_allclose(spine_y,edges[0,0],atol=1e-7,rtol=0)
    local_right=local.get_window_extent().x1
    spatial_label_left=spatial.get_tightbbox(renderer).x0
    np.testing.assert_allclose((spatial_label_left-local_right)/fig.dpi*25.4,1.2,atol=1e-7,rtol=0)
    plate.CHECKS['column_reflow'].update(AD_visible_width_mm=float(np.diff(horizontal_bounds('A'))[0]),
        AD_equal_visible_width_and_edges=True,BEFH_axes_and_letters_aligned=True,
        BC_equal_width_and_height=True,ABC_bottom_spines_and_top_edges_aligned=True,
        ABC_axis_y_edges_mm=(edges/fig.dpi*25.4).tolist())


def export(fig,letters):
    positions={a:a.get_position().frozen() for a in fig.axes}
    def save(stem,box=None,dpi=320,extensions=('png','pdf','svg')):
        for ext in extensions:
            fig.savefig(plate.FIG/f'{stem}.{ext}',dpi=dpi,bbox_inches=box)
            for a,p in positions.items():a.set_position(p)
    save('fig4-complete-layout')
    save('fig4-complete-layout-preview',dpi=135,extensions=('png',))
    visibility={a:a.get_visible() for a in fig.axes}
    for t in letters:t.set_visible(False)
    for key,display in LETTERS.items():
        for k,g in plate.GROUPS.items():
            for a in g['axes']:a.set_visible(k==key and visibility[a])
            for a in g['texts']+g['artists']:a.set_visible(k==key)
        restored=[]
        if key=='D':  # Former D / candidate E shares units with the lower row.
            for a in plate.GROUPS[key]['axes']:
                if a.images and a.get_visible() and any(t.get_visible() for t in a.get_xticklabels()):
                    restored.append((a,a.get_xlabel()))
                    a.set_xlabel('Time (ms)' if a.get_xlim()[0]<0 else 'x (mm)',fontsize=plate.MIDDLE_LABEL_PT)
        fig.canvas.draw();renderer=fig.canvas.get_renderer();g=plate.GROUPS[key]
        boxes=[a.get_tightbbox(renderer) for a in g['axes'] if a.get_visible()]
        boxes.extend(t.get_window_extent(renderer) for t in g['texts'] if t.get_visible())
        box=Bbox.union([b for b in boxes if b is not None]).transformed(fig.dpi_scale_trans.inverted()).padded(1.2*plate.MM)
        save(f'fig4-panel{display.lower()}',box=box)
        for a,old in restored:a.set_xlabel(old)


def main():
    pointer=ROOT/'results/paper-ready-figure/fig4/current_version.json'
    current=plate.read(pointer)
    # Use the fixed A-I source package even after the current pointer advances.
    base_package=recovery.OUT
    # A promoted base package can be fig4/ itself, which also contains this
    # candidate. Protect its published files without including our own outputs.
    protected=[pointer,*[ROOT/p for p in current['outputs_sha256']],
               ROOT/current['registry'],ROOT/current['visual_qa'],
               *[p for p in (base_package/'source').rglob('*') if p.is_file()]]
    before={str(p):plate.sha(p) for p in protected}
    plate.OUT=OUT;plate.FIG=OUT/'figures';plate.SOURCE=OUT/'source';plate.base.OUT=plate.SOURCE
    plate.FIG.mkdir(parents=True,exist_ok=True);plate.SOURCE.mkdir(exist_ok=True)
    plate.GROUPS.clear();plate.CHECKS.clear()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,
        'xtick.labelsize':9,'ytick.labelsize':9,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})
    case,snapshot=frozen_case()
    rows=list(csv.DictReader(plate.freeze(plate.OLD/'source/all_workpoints_source.csv').open()))
    for r in rows:
        for k in plate.base.KEYS:r[k]=float(r[k])
        r['stage']=int(r['stage']);r['N']=int(r['N'])
    cases=plate.read(plate.freeze(plate.PREVIOUS/'source/selected_workpoints.json'))['cases']
    curves=list(csv.DictReader(plate.freeze(plate.OLD/'source/connection_response_source.csv').open()))
    geometry=plate.read(plate.freeze(plate.EXPANDED/'source/position_map_source.json'))
    assert geometry['conditions']==83
    fig=plt.figure(figsize=(plate.W*plate.MM,plate.H*plate.MM),dpi=160,facecolor='white')
    # Only B's cross-references change; the propagation renderers retain source identities.
    b_cases=[dict(c,panel=LETTERS[c['panel']]) for c in cases]
    parameter_rows=None
    for key,fn in [('A',lambda:plate.panel_a(fig)),
                   ('B',lambda:plate.panel_b(fig,rows,b_cases,profile_only=True)),
                   ('C',lambda:panel_d(fig,curves,geometry,rows))]:
        with plate.group(fig,key):
            roles=fn()
            if key=='C':parameter_rows=roles
        print('Rendered',LETTERS[key],flush=True)
    with plate.group(fig,'core_count'):
        ax=fig.add_axes(plate.rect(223,178+plate.TOP_SHIFT,51,45))
        progress.draw(ax,case,show_title=False,paper_style=plate.FONT,show_legend=True,
                      summary_style=case.get('summary_style','median_iqr'))
        ax.set_yticks([0,2,4,6,8])
    plate.panels_de(fig,cases);plate.panels_fgh(fig)
    with plate.group(fig,'I'):
        recovery_source=plate.freeze(base_package/'source/summary.json')
        recovery_summary=plate.read(recovery_source)
        shutil.copyfile(recovery_source,plate.SOURCE/'template_recovery_summary.json')
        ax=fig.add_axes(plate.rect(242,plate.LOWER_Y,32,plate.ROW_HEIGHT))
        plate.axis_type(ax)
        recovery_metadata=recovery.draw_modes(ax,recovery_summary,
            legend_loc=current.get('panel_j',{}).get('legend_location'))
    if parameter_rows is not None and REFLOW_COLUMNS:reflow_columns(fig)
    letters=[];lower_letters=[]
    for key,x,y in [('A',2,248),('B',137,248),('core_count',212,248),('C',2,179),
                    ('D',122,178),('E',122,119),('F',2,63),('G',120,63),('H',170,63),('I',230,63)]:
        if parameter_rows is not None and REFLOW_COLUMNS and key in ('B','G'):x=122
        if parameter_rows is not None and REFLOW_COLUMNS and key=='core_count':x=plate.CHECKS['column_reflow']['C_letter_x_mm']
        t=plate.label(fig,x,y,LETTERS[key],fontsize=plate.FONT['letter'],weight='bold',va='top')
        letters.append(t)
        if key in 'FGHI':lower_letters.append(t)
    lower_axes={a for k in 'FGHI' for a in plate.GROUPS[k]['axes']}
    lower_texts={t for k in 'FGHI' for t in plate.GROUPS[k]['texts']}|set(lower_letters)
    before_shift={a:a.get_position().frozen() for a in fig.axes}
    for a,b in before_shift.items():
        x,y,w,h=b.bounds;shift=5 if a in lower_axes else 5+plate.LOWER_ROW_EXTRA_GAP_MM
        a.set_position([x,y+shift/plate.H,w,h])
    for t in fig.texts:
        shift=5 if t in lower_texts else 5+plate.LOWER_ROW_EXTRA_GAP_MM
        x,y=t.get_position();t.set_position((x,y+shift/plate.H))
    for a in plate.GROUPS['A']['artists']:
        a.set_ydata(np.array(a.get_ydata())+(5+plate.LOWER_ROW_EXTRA_GAP_MM)/plate.H)
    try:
        plate.verify_alignment(fig,panel_letters=LETTERS,extra_top_groups=('core_count',),
                               parameter_rows=parameter_rows)
        if parameter_rows is not None:
            validate_parameter_layout(fig)
            if REFLOW_COLUMNS:validate_columns(fig)
    except AssertionError:
        (OUT/'qa').mkdir(exist_ok=True)
        fig.savefig(OUT/'qa/layout-draft.png',dpi=135)
        raise
    checks=validate_new_panel(fig,case)
    export(fig,letters);plt.close(fig)
    changed=[p for p,digest in before.items() if plate.sha(p)!=digest]
    assert not changed,changed
    for mod in (plate,progress,recovery,parameters,plate.compare,plate.combined):plate.freeze(mod.__file__)
    plate.freeze(__file__)
    record=dict(status='CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW',producer=str(Path(__file__).relative_to(ROOT)),
        base_current_version=str(pointer.relative_to(ROOT)),base_package=str(base_package.relative_to(ROOT)),
        panel_letters=list('ABCDEFGHIJ'),base_group_to_candidate_letter=LETTERS,
        titles=TITLES,whole_canvas_mm=[plate.W,plate.H],font_pt=plate.FONT,
        changes='A matches D in visible width. B/C have equal widths and heights; their bottom spines, top edges and the B colourbar align with the A spatial axis. Both share the added horizontal space while preserving the readable gap. B/E/F/H remain vertically aligned. D retains its three error responses, position density, and verified E/F parameter annotations.',
        source_files=plate.SOURCES,source_snapshot_time=snapshot['updated_at'],core_count_summary=case['summary'],
        curve_semantics=snapshot['semantics'],current_package_and_pointer_unchanged=True,
        scientific_scope='E1146 initial-sampling training progress, not completed optimization or independent validation.',
        original_panel_checks=plate.CHECKS,checks=checks,new_simulations=0,
        candidate_J=recovery_metadata,
        outputs={str(p.relative_to(OUT)):plate.sha(p) for p in plate.FIG.iterdir() if p.suffix in ('.png','.pdf','.svg')})
    progress.write(OUT/'figure4_candidate_registry.json',record)
    progress.write(OUT/'visual_qa.json',dict(status='NUMERICAL_AND_LAYOUT_PASS_VISUAL_PENDING',
        checks=checks,current_package_and_pointer_unchanged=True,human_visual_acceptance='PENDING'))
    lines=['# Figure 4：新 C 候选（A–J）','',
        '状态：B/C共享新增空间、A/B/H列对齐与D修订候选，待作者目视检查；当前指定 A–J 的文件与机器入口保持不变。',
        '新 C 放在 B 右侧，替换原 B 的三维 panel；B 保留左侧误差曲线及紫色迭代色条，原 C–I 顺延为 D–J。B 中工作点引用已同步为 E/F。','',
        '| 候选编号 | 内容 |','|---|---|']
    lines += [f'| {letter} | {title} |' for letter,title in TITLES.items()]
    lines += ['', 'B/C共同使用新增18 mm横向空间并调整为等宽等高：两者均55.5 × 48.288 mm，相比原42/51 mm宽度分别增加13.5/4.5 mm。A空间图与B/C的横轴线、上边缘和B色条全部对齐，实际上下边界为206.43/254.718 mm。B色条位于191.5 mm，C绘图区与角标分别位于218.5/207.5 mm；保留两图可见间隙和整图右边界。轴标签10 pt、刻度9 pt、图例8.8 pt；C图内仅保留Loss、Epochs、1 core / 2 cores。', '',
        'A 按实际绘图边界收窄到与D等宽，左右边缘一致；局部回路的物理尺寸及内部字体保留，右侧空间图等比例缩小并保持外部字号，虚线端点随两张图重新定位。B/E/F/H绘图区左边界统一130 mm，四个panel角标统一122 mm。整图行间距及其余轴高保留。', '',
        'D 三张参数曲线都只保留左侧 Error 轴，统一0–0.42范围及40 × 40 mm轴框；图内三色图例为 Mean rank、Order、Participation，沿用B的语义色。上排两张各保留原13个条件，rank/order沿用原曲线值，participation从冻结工作点按candidate ID提取并交叉核对；实际26条件、78个误差值见 source/D_connection_response.csv。', '',
        'D 虚线按 E/F 传播示例的实际 design.json 和 applied_physics.json 定位并标注：E（lf_bo04_07）的向外EE为1.0183399664、EE角度偏移16.8874761835°；F（xy_left_20）分别为1.25、0°。两者核内EE均为0.85，故左下共用一条E/F虚线。旧无标注线是扫描基准，不曾分别代表两个示例。当前虚线只标出该参数坐标，曲线交点不是E/F的实际误差：上排扫描固定核中心为(3.4492,12.1289)/(16.4792,4.7155) mm，而E/F采用各自不同的中心，核内EE扫描也另有自身背景。实际参数、执行来源及核验见 source/D_EF_workpoint_references.json。', '',
        'D 右下复用用户参考图：257个可评分配置中 J_joint 最低20%的52个配置，核中心以0.35 mm高斯核构建等权位置密度；高度和紫色色条为 Position density (mm⁻²)，不是误差值、参数后验或独立重复的不确定性。地图缩为32 × 32 mm，其水平中心与上排E→E angle对齐。52个配置均来自兼容历史池；快照、阈值、实际坐标及密度数组均保留。', '',
        'D 左下为已完成的 E1146 核内 E→E 权重单参数扫描，与上排Outward E→E gain左边界及宽度对齐。J_EE,core = 0.60、0.70、0.80、0.85、0.95、1.10、1.25；七条60秒轨迹固定位置、拓扑2511、噪声847401及其余物理。三条曲线分别为平均rank、杆内顺序、参与误差；从原primary事件时刻重算并验证，非训练kernel loss。背景是队列上一轮E1146训练提名点，与上排的历史单患者扫描背景不同，不能将三张曲线拼成同一工作点的全参数切片。实际数据及审计见 source/D_recurrent_EE_response.csv 和 D_recurrent_EE_audit.json；仅为FIT描述性误差，不增加独立验证或机制验收结论。', '',
        '数据冻结于 2026-09-29 16:25:10 CST，与用户截图一致：E1146 每组四次随机搜索，共同完成前 7 个 epoch；较快的重复细线显示至第 8 个 epoch。粗线和色带分别为共同前缀的最佳训练 loss 中位数及四分位范围；第 7 个 epoch 中位数为 4.1394 和 1.9557。仍属于初始采样，未完成 96 epoch 计划；不表示独立验证。', '',
        '可复现命令：`python scripts/paper_figures/build_fig4_core_count_candidate.py`。只读取冻结数据；不重跑仿真，也不跟随后台训练自动更新输入。', '',
        '本次额外调整A/B/H的列对齐；D沿用三误差及E/F虚线修订，其余数据与示例保持冻结。保留此前整图等高、行间距和图例位置。详情沿用基底版本 panel_map.md。']
    (OUT/'panel_map.md').write_text('\n'.join(lines)+'\n')
    notes=['### fig4-complete-layout.png\n新 C 单核／双核 loss 曲线放入 B 右侧，B 仅保留原左侧误差 panel 和色条，原 C–I 顺延为 D–J。A/D等宽，B/C调整为等宽等高并共享腾出的空间；A空间图、B、C的横轴线及上边缘对齐，B色条同步等高。另有同源 PDF/SVG 和整图预览。\n**关注点**：按整图检查字号与布局；候选待作者目视检查。']
    for letter,title in TITLES.items():
        detail=('E1146 每组四次搜索，共同前7个epoch的中位数及四分位范围；个别细线延伸至第8个epoch。' if letter=='C' else
                '三张参数曲线均为单一左侧Error轴，图例为Mean rank、Order、Participation；核内EE位于左下并与Outward E→E gain等宽对齐，缩小的位置密度图位于右下并与E→E angle居中对齐。虚线E/F对应传播示例的实际参数，核内EE的0.85为两者重合值；仅表示参数坐标，曲线交点并非示例实际误差。数据来源及不同背景工作点见上级 panel_map.md。' if letter=='D' else
                '与完整拼图使用相同绘图对象；独立导出不带 panel 角标。')
        notes.append(f'### fig4-panel{letter.lower()}.png\n{title}。{detail}另有同源 PDF/SVG。\n**关注点**：'+('仅为冻结初始采样进度，尚非完整优化或独立验证。' if letter=='C' else '沿用原数据合同；候选编号见上级 panel_map.md。'))
    (plate.FIG/'README.md').write_text('\n\n'.join(notes)+'\n')
    print(plate.FIG,flush=True)


if __name__=='__main__':main()
