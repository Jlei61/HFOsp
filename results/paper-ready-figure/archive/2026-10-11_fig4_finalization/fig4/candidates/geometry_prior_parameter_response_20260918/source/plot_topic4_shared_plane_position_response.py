"""Place measured core-center response grids on the actual 20 mm sheet."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src/snn_engine')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse, Circle, Rectangle
from scripts.paper_figures import plot_topic4_compact_english_acceptance as base

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
                     'font.size':8,'svg.fonttype':'none','pdf.fonttype':42})

OUT = base.ROUND / 'key_results/data_driven_metric_tradeoffs/shared_plane_position_response'
FIG = OUT / 'figures'
OLD = base.OUT
CMAPS = dict(rank_error='Blues', within_rod_order_error='Oranges', participation_error='Blues')
LABEL_COLORS = dict(rank_error=base.BLUE, within_rod_order_error=base.ORANGE, participation_error=base.BLUE)
ERROR_NORM = Normalize(0, .30)
METRICS = list(zip(base.KEYS[:2], ['Mean rank\nerror', 'Within-rod\norder error']))


def geometry(rows, xy):
    ref = next(r for r in rows if r['candidate'] == 'dense_axis3_12')
    source = Path(ref['source'])
    assert base.sha(source) == ref['source_sha256']
    physics = base.read(source.parents[1] / 'applied_physics.json')
    centers = np.array(physics['threshold']['centers_mm'])
    with np.load(source.with_suffix('.npz')) as a:
        contacts = a['contact_xy_mm'].copy()
        names = a['contact_names'].astype(str)
    assert contacts.shape == (15, 2)
    index = {r['candidate']: r for r in rows}
    grids = []
    for core in (0, 1):
        cells = [r for r in xy if r['core'] == core]
        xs, ys = (np.unique([r[k] for r in cells]) for k in ('x', 'y'))
        assert len(xs) == len(ys) == 5
        assert np.allclose(np.diff(xs), .25) and np.allclose(np.diff(ys), .25)
        assert np.allclose([xs[2], ys[2]], centers[core])
        for r in cells:
            actual = base.read(Path(index[r['candidate']]['source']).parents[1] / 'applied_physics.json')
            assert np.allclose(actual['threshold']['centers_mm'][core], [r['x'], r['y']])
            assert np.allclose(actual['threshold']['centers_mm'][1-core], centers[1-core])
            assert actual['candidate']['parameters'] == physics['candidate']['parameters']
            for k in base.KEYS:
                assert np.isclose(r[k], index[r['candidate']][k], rtol=0, atol=1e-12)
        grids.append(dict(core=core, x=xs.tolist(), y=ys.tolist(),
            extent=[float(xs[0]-.125), float(xs[-1]+.125), float(ys[0]-.125), float(ys[-1]+.125)],
            matrices={k: [[next(r[k] for r in cells if r['x']==x and r['y']==y) for x in xs] for y in ys] for k in base.KEYS}))
    manifest = dict(reference=ref['candidate'], reference_source=str(source),
        applied_physics_sha256=base.sha(source.parents[1] / 'applied_physics.json'),
        centers_mm=centers.tolist(), threshold_geometry=physics['threshold'],
        contact_names=names.tolist(), contact_xy_mm=contacts.tolist(), grids=grids,
        meaning='Cell location is a tested core CENTER, color is the whole-run observable error; not a local activity field or error at tissue nodes.',
        conditional_design='Move A in XY with B fixed; move B in XY with A fixed. Both sweeps share the same reference and remaining parameters. Not a joint A/B response surface.',
        dimensions_mm=[20,20], tested_center_offsets_mm=[-.5,-.25,0,.25,.5],
        display_rule='0.25 mm square cells centered on actual sample coordinates; full field shown at equal aspect, no interpolation, dilation, smoothing or extrapolation. Gray background is unsampled, not zero error.',
        outlines='Solid outlines are actual reference core boundaries. Dotted rectangle encloses displayed sample cells; small cross marks reference center in the enlarged view only.',
        conditions=49, grid_cells_per_map=50, source_metric_values=150, displayed_metric_values=100,
        displayed_metrics=[k for k,_ in METRICS], omitted_from_A='Participation error removed at user request; source values and B remain unchanged.',
        color_limits=[0,.3], colormaps_by_metric=CMAPS,
        uncertainty='One paired topology/noise unit per condition; no replicate uncertainty inferred.')
    base.write(OUT / 'position_map_source.json', manifest)
    return manifest


def wide_geometry(rows, directory):
    # Establish the same verified reference geometry before loading the new grid.
    xy=base.table(base.SOURCE/'xy_grid_source.csv')
    for r in xy:
        r['core']=int(r['core'])
        for k in ['x','y']+base.KEYS: r[k]=float(r[k])
    manifest=geometry(rows,xy)
    plan=base.read(directory/'plan.json')
    table={r['candidate']:r for r in base.table(directory/'analysis/observables.csv') if r['mode']=='ALL'}
    ref=base.read(directory/'candidates'/f"{plan['reference_id']}.json")
    from scripts import run_topic4_xy_wide_response as run
    centers=np.array(manifest['centers_mm']); grids=[]; count=0; overflow={k:0 for k in base.KEYS}
    for grid in plan['grids']:
        core=grid['core']; dx=np.array(grid['x_offsets_mm']); dy=np.array(grid['y_offsets_mm'])
        step=float(np.diff(dx)[0]); assert np.allclose(np.diff(dx),step) and np.allclose(np.diff(dy),step)
        xs=centers[core,0]+dx; ys=centers[core,1]+dy
        matrices={k:[[None]*len(xs) for _ in ys] for k in base.KEYS}; states=[]
        for iy,cids in enumerate(grid['ids']):
            state_row=[]
            for ix,cid in enumerate(cids):
                candidate=base.read(directory/'candidates'/f'{cid}.json'); run.validate(candidate,ref)
                assert np.allclose(candidate['centers_mm'][core],[xs[ix],ys[iy]])
                assert np.allclose(candidate['centers_mm'][1-core],centers[1-core])
                score_path=directory/'scores'/f'{cid}.json'
                if cid not in table or not score_path.exists():
                    state_row.append('PENDING'); continue
                score=base.read(score_path)
                if score['J'] is None:
                    state_row.append(score['status']); continue
                actual=base.read(Path(score['source']).parents[1]/'applied_physics.json')
                assert run.old.physics(actual['candidate'])==run.old.physics(candidate)
                assert score['objective_sha256']==plan['objective']['sha256']
                for key in base.KEYS:
                    value=float(table[cid][key]); assert np.isfinite(value)
                    matrices[key][iy][ix]=value; overflow[key]+=int(value>.30)
                state_row.append('MEASURED'); count+=1
            states.append(state_row)
        grids.append(dict(core=core,x=xs.tolist(),y=ys.tolist(),
            extent=[float(xs[0]-step/2),float(xs[-1]+step/2),float(ys[0]-step/2),float(ys[-1]+step/2)],
            matrices=matrices,ids=grid['ids'],states=states))
    manifest.update(grids=grids,tested_center_offsets_mm=dx.tolist(),conditions=49,
        completed_unique_conditions=len(table),grid_cells_per_map=50,measured_cells_per_map=count,
        displayed_metric_values=2*count,source_metric_values=3*count,
        display_rule='0.75 mm square cells at true centers; missing/unscorable trajectories remain blank, with no interpolation or extrapolation.',
        experiment=str(directory),source_table_sha256=base.sha(directory/'analysis/observables.csv'),
        source_plan_sha256=base.sha(directory/'plan.json'),above_color_limit_by_metric=overflow)
    base.write(OUT/'position_map_source.json',manifest)
    return manifest


def sheet(ax, manifest, key, first=False, detailed=False):
    ax.set_facecolor('#F3F3F3')
    centers = np.array(manifest['centers_mm'])
    th = manifest['threshold_geometry']
    for grid in manifest['grids']:
        ax.imshow(np.asarray(grid['matrices'][key],dtype=float), extent=grid['extent'], origin='lower',
                  interpolation='nearest', cmap=CMAPS[key], norm=ERROR_NORM, aspect='equal', zorder=2)
        x0,x1,y0,y1=grid['extent']
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#555555',lw=.45,ls=':',zorder=4))
    ax.add_patch(Ellipse(centers[0], 2*th['ellipse_A_semiaxes_mm'][0], 2*th['ellipse_A_semiaxes_mm'][1],
                        angle=th['ellipse_A_angle_deg'],fill=False,edgecolor='#555555',lw=.65,zorder=3))
    ax.add_patch(Circle(centers[1],th['radii_mm'][1],fill=False,edgecolor='#555555',lw=.65,zorder=3))
    for core, (x,y) in enumerate(centers):
        if detailed: ax.plot(x,y,marker='+',ms=2,mew=.5,color='white',zorder=5)
        ax.text(x,y+th['radii_mm'][core]+.9, 'Core '+('A' if core==0 else 'B') if detailed else ('A' if core==0 else 'B'),
                ha='center',va='bottom',fontsize=9 if detailed else 8)
    contacts=np.array(manifest['contact_xy_mm']); names=np.array(manifest['contact_names'])
    for shaft in ['SCL','ICL']:
        order=sorted(np.flatnonzero(np.char.startswith(names,shaft)),key=lambda i:int(names[i][len(shaft):]),reverse=True)
        ax.plot(*contacts[order].T,color='#737373',lw=.7,zorder=3)
        ax.scatter(*contacts[order].T,s=8 if detailed else 3.5,facecolors='white',edgecolors='#515151',lw=.5,zorder=4)
        if detailed:
            px,py=contacts[order[0]]
            ax.text(px,py+1.2,shaft,fontsize=9,ha='center',color='#555555')
    ax.set(xlim=(0,20),ylim=(0,20),xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)',aspect='equal')
    if first: ax.set_ylabel('y (mm)',labelpad=1)
    ax.tick_params(length=2,pad=1.5,labelleft=first,labelsize=9 if detailed else 8)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color('#B3B3B3'); spine.set_linewidth(.5)


def parameters(fig, curves, manifest):
    for j,x,xlabel,xticks in [(4,14,'Outward EE (×)',[1,1.25,1.5]),
                             (5,62,'EE angle (°)',[-20,0,20])]:
        ax=base.axes(fig,x,156,28,26); twin=ax.twinx()
        for target,key,color,marker in [(ax,'rank',base.BLUE,'o'),(twin,'local_order',base.ORANGE,'s')]:
            r=sorted([r for r in curves if int(r['axis'])==j and r['observable']==key],key=lambda r:float(r['parameter_value']))
            assert len(r)==13
            target.plot([float(q['parameter_value']) for q in r],[float(q['value']) for q in r],color=color,marker=marker,ms=2.2,lw=1)
            target.set(ylim=(0,.42),yticks=[0,.2,.4]); target.tick_params(axis='y',colors=color,pad=1.5,length=2)
        ax.spines['left'].set_color(base.BLUE); twin.spines['right'].set_visible(True); twin.spines['right'].set_color(base.ORANGE)
        twin.spines['left'].set_visible(False); twin.spines['bottom'].set_visible(False)
        ax.axvline(1.25 if j==4 else 0,color='#999999',ls=':',lw=.7)
        ax.set(xlabel=xlabel,xticks=xticks); ax.tick_params(axis='x',pad=2,length=2)
        if j==4:
            ax.set_ylabel('Rank error ↓',labelpad=2,color=base.BLUE)
            twin.tick_params(labelright=False)
        if j==5:
            twin.set_ylabel('Within-rod error ↓',labelpad=2,color=base.ORANGE)
            ax.tick_params(labelleft=False)
    for i, (key,title) in enumerate(METRICS):
        x=14+48*i
        ax=base.axes(fig,x,105,28,28)
        sheet(ax,manifest,key,first=i==0)
        base.label(fig,x+14,135,title,ha='center',va='bottom',color=LABEL_COLORS[key])
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=ERROR_NORM,cmap=CMAPS[key]),
                       cax=base.axes(fig,x+30,105,1.2,28),ticks=[0,.15,.3],orientation='vertical',
                       extend='max' if manifest.get('above_color_limit_by_metric',{}).get(key,0) else 'neither')
        cb.outline.set_visible(False); cb.ax.tick_params(length=2,pad=1.5)


def export(fig, stem):
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); fonts=[]; outside=[]
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            fonts.append(t.get_fontsize()); bb=t.get_window_extent(renderer)
            assert not any('\u4e00'<=c<='\u9fff' for c in t.get_text())
            if bb.x0<-.5 or bb.y0<-.5 or bb.x1>fig.bbox.x1+.5 or bb.y1>fig.bbox.y1+.5: outside.append(t.get_text())
    assert min(fonts)>=8 and not outside, outside
    fig.savefig(FIG / (stem+'.png'),dpi=400)
    fig.savefig(FIG / (stem+'.svg'))
    plt.close(fig)
    from PIL import Image
    import xml.etree.ElementTree as ET
    with Image.open(FIG / (stem+'.png')) as image: image.load()
    assert any(e.tag.endswith('text') for e in ET.parse(FIG / (stem+'.svg')).getroot().iter())
    return dict(minimum_font_pt=min(fonts),text_outside_canvas=outside,decoded=True)


def search_panels(fig, rows, cases):
    # Selection names organize the examples; they do not alter color or salience.
    profile=base.profile(fig,rows,[])
    profile.set_position([110/base.WIDTH,136/base.HEIGHT,42/base.WIDTH,45/base.HEIGHT])
    for r in cases:
        profile.annotate(r['panel'],(2,r['participation_error']),xytext=(4,0),
                         textcoords='offset points',fontsize=8,weight='normal',va='center')
    trajectory=base.trajectory(fig,rows,[])
    trajectory.set_position([175/base.WIDTH,133/base.HEIGHT,51/base.WIDTH,51/base.HEIGHT])
    bar=fig.axes[-1]
    bar.set_position([159/base.WIDTH,136/base.HEIGHT,1.5/base.WIDTH,45/base.HEIGHT])
    bar.set_ylabel('Iteration order',labelpad=3)
    fig.canvas.draw()
    for r in cases:
        x,y,_=base.proj3d.proj_transform(*[r[k] for k in base.KEYS],trajectory.get_proj())
        trajectory.annotate(r['panel'],(x,y),xytext=(4,5),textcoords='offset points',
                            fontsize=8,weight='normal',zorder=6)
    assert len(profile.lines)==len(rows)
    assert np.isclose(bar.get_position().height,profile.get_position().height)


def compact_propagation(fig, cases):
    """Reflow the existing artists; keep arrays, events and field aspect unchanged."""
    na,nt=len(fig.axes),len(fig.texts)
    events=base.propagation(fig,cases)
    new_axes=fig.axes[na:]; new_text=fig.texts[nt:]
    assert len(new_axes)==18 and len(new_text)==16
    readout_h,field_size=36.,17.5
    text_positions={}
    for case_index,(old_bottom,bottom) in enumerate(zip([74,12],[55,10])):
        for mode_index,(old_x,x,row) in enumerate([(14,14,1),(56,68,0)]):
            idx=case_index*8+mode_index*4
            new_axes[idx].set_position([x/base.WIDTH,bottom/base.HEIGHT,36/base.WIDTH,readout_h/base.HEIGHT])
            field_y=bottom+(field_size+1)*row
            for col in range(3):
                new_axes[idx+col+1].set_position([(135+20*col)/base.WIDTH,field_y/base.HEIGHT,field_size/base.WIDTH,field_size/base.HEIGHT])
            text_positions[(old_x+13.5,old_bottom+51)]=(x+18,bottom+38)
            text_positions[(103,old_bottom+25*row+12)]=(133,field_y+field_size/2)
        for col in range(3):
            text_positions[(119+26*col,old_bottom+51)]=(135+20*col+field_size/2,bottom+38)
        text_positions[(97,old_bottom+24.5)]=(127,bottom+readout_h/2)
    for text in new_text:
        x,y=text.get_position();key=(round(x*base.WIDTH,4),round(y*base.HEIGHT,4))
        assert key in text_positions,key
        nx,ny=text_positions[key];text.set_position((nx/base.WIDTH,ny/base.HEIGHT))
    new_axes[-2].set_position([108/base.WIDTH,10/base.HEIGHT,2/base.WIDTH,readout_h/base.HEIGHT])
    new_axes[-1].set_position([195/base.WIDTH,10/base.HEIGHT,1.4/base.WIDTH,field_size/base.HEIGHT])
    new_axes[-1].set_ylabel('')
    new_axes[-1].set_yticks([0,75])
    for tick,va in zip(new_axes[-1].get_yticklabels(),['bottom','top']): tick.set_verticalalignment(va)
    assert np.isclose(new_axes[-2].get_position().height,new_axes[8].get_position().height)
    assert np.isclose(new_axes[-1].get_position().height,new_axes[15].get_position().height)
    # Preserve readout bars; native fields use one fixed-range bar per row.
    for mode_index,x in enumerate([52,108]):
        cb=fig.colorbar(new_axes[mode_index*4].images[0],
                       cax=base.axes(fig,x,55,1.4,readout_h),ticks=[0,1],orientation='vertical')
        cb.outline.set_visible(False);cb.ax.tick_params(length=2,pad=1.5)
        if mode_index==1: cb.set_label('Normalized envelope',labelpad=3)
        assert np.isclose(cb.ax.get_position().height,new_axes[mode_index*4].get_position().height)
    for case_index,bottom in enumerate([55,10]):
        for mode_index,row in enumerate([1,0]):
            if case_index==1 and row==0: continue  # Existing D/TB bar reused above.
            fy=bottom+(field_size+1)*row
            cb=fig.colorbar(plt.cm.ScalarMappable(cmap='inferno',norm=Normalize(0,75)),
                           cax=base.axes(fig,195,fy,1.4,field_size),ticks=[0,75],orientation='vertical')
            cb.outline.set_visible(False); cb.ax.tick_params(length=2,pad=1.5)
            for tick,va in zip(cb.ax.get_yticklabels(),['bottom','top']): tick.set_verticalalignment(va)
            assert np.isclose(cb.ax.get_position().height,new_axes[case_index*8+mode_index*4+3].get_position().height)
        base.label(fig,207,bottom+readout_h/2,'Activity / 2 ms',rotation=90,
                   rotation_mode='anchor',ha='center',va='center')
    return events


def main(wide_dir=None):
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(exist_ok=True)
    base.OUT=OUT; base.FIG=FIG; base.HEIGHT=190.
    rows,cases,curves,xy=base.load('mean')
    # User-requested display order: previous D first, previous C second.
    cases=[dict(cases[1],panel='C'),dict(cases[0],panel='D')]
    selection=base.read(OUT / 'selected_workpoints.json')
    selection.update(cases=cases,display_order='Higher-error workpoint C above lower-error workpoint D; same selected workpoints and events.')
    base.write(OUT / 'selected_workpoints.json',selection)
    manifest=geometry(rows,xy) if wide_dir is None else wide_geometry(rows,Path(wide_dir))
    contract=base.read(OUT / 'figure_contract.json')
    contract['panels']['A']='EE strength/direction curves above two 20x20 mm sheets: mean-rank and within-rod order errors; participation removed only from A at user request.'
    contract['core_conclusion']='Position-response values become spatially interpretable when registered to the actual core and electrode geometry; no new propagation-recovery claim.'
    contract['position_map']=manifest
    contract['new_simulations']=0 if wide_dir is None else dict(planned=40,source=str(wide_dir),completed_conditions=manifest['completed_unique_conditions'])
    contract['omissions']='Participation position map removed from A at user request; participation source values, all 226 B profiles/3D points and the existing C/D selection rule remain intact.'
    contract['panels']['C']='Higher-error workpoint above; same TA/TB events, two envelope bars and one native-field bar per row.'
    contract['panels']['D']='Lower-error workpoint below; same TA/TB events, envelope scale and one native-field bar per row.'
    contract['panels']['B']='All 226 equal-weight profiles/points; stage colors unchanged. Iteration order bar next to left profile; plain C/D annotations without overplotted lines or stars.'
    contract['panel_order']={r['panel']:r['candidate'] for r in cases}
    contract['source_files'][str(Path(__file__))]=base.sha(Path(__file__))
    contract['display_change']='A columns aligned; shorter axis labels. B has no selected-candidate emphasis; chronology bar next to left profile. C/D each have one native field colorbar per TA/TB row.'
    contract['layout_mm']=dict(width=238,height=190,B_profile_height=45,B_colorbar_height=45,CD_readout_height=36,field_side=17.5,envelope_colorbar_height=36,native_colorbar_height=17.5)
    base.write(OUT / 'figure_contract.json',contract)
    fig=plt.figure(figsize=(base.WIDTH/25.4,base.HEIGHT/25.4))
    parameters(fig,curves,manifest)
    search_panels(fig,rows,cases); events=compact_propagation(fig,cases)
    old_events={e['candidate']:{k:v for k,v in e.items() if k!='panel'} for e in base.read(OLD / 'propagation_manifest.json')['cases']}
    assert {e['candidate']:{k:v for k,v in e.items() if k!='panel'} for e in events}==old_events
    for s,x,y in [('A',2,185),('B',103,185),('C',2,94),('D',2,49)]: base.label(fig,x,y,s,fontsize=11,weight='bold')
    qa=export(fig,'parameter_search_propagation_shared_plane')
    # The two retained observables keep all 100 measured values and true coordinates.
    fig=plt.figure(figsize=np.array([164,79])/25.4)
    for i,(key,title) in enumerate(METRICS):
        ax=fig.add_axes([(13+74*i)/164,13/79,54/164,54/79])
        sheet(ax,manifest,key,first=i==0,detailed=True)
        ax.set_title(title.replace('\n',' '),fontsize=10,pad=8,color=LABEL_COLORS[key])
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=ERROR_NORM,cmap=CMAPS[key]),
                       cax=fig.add_axes([(70+74*i)/164,13/79,1.8/164,54/79]),ticks=[0,.15,.3],orientation='vertical',
                       extend='max' if manifest.get('above_color_limit_by_metric',{}).get(key,0) else 'neither')
        cb.outline.set_visible(False); cb.ax.tick_params(length=2,pad=2,labelsize=9)
    qa['detail']=export(fig,'core_position_two_observables')
    qa.update(profiles=226,spatial_maps=2,grids_per_map=2,grid_cells_per_map=50,total_displayed_metric_values=manifest['displayed_metric_values'],source_metric_values_retained=manifest['source_metric_values'],
        actual_geometry_verified=True,paired_fixed_background_verified=True,interpolation=False,
        candidate_and_event_selection_unchanged=True,panel_order=contract['panel_order'],B_labels_match_display_panels=True,shared_color_limits=[0,.3],colormaps_by_metric=CMAPS,colorbar_axis_height_checks='PASS',layout_mm=contract['layout_mm'],
        C_individual_colorbars=dict(envelopes=2,native_rows=2,envelope_limits=[0,1],native_limits=[0,75]),
        D_native_row_colorbars=2,B_uniform_profile_linewidth=.5,B_selected_stars=False,A_column_left_mm=[14,62],
        agent_visual_review='PENDING',human_visual_review='PENDING')
    base.write(OUT / 'qa.json',qa)
    (FIG / 'README.md').write_text('### parameter_search_propagation_shared_plane.png\nA的两列上下对齐，位置响应使用真实20×20 mm平面；上排EE曲线仍保留原有背景，见figure_contract.json。B的全部曲线等宽、点等大，浅蓝到深蓝表示真实提案阶段；C/D仅用普通字母定位，色条位于左侧曲线旁。C、D各有两行原生场，每行最右侧共用一根0–75色条；触点包络和原生帧、固定坐标及选例不变。\n**关注点**：未采样位置不插值；颜色表示运行误差或搜索阶段，不代表局部组织活动，也不保证优化逐步单调变好。\n\n### core_position_two_observables.png\nA中两张位置响应图的放大版；蓝色是平均rank误差，橙色是杆内顺序误差。实际采样范围、已完成数量和原始轨迹见position_map_source.json；每次只移动一核，另一核及背景固定。\n**关注点**：新增范围只显示已完成且可估计的真实轨迹，空白保持为缺测；单种子条件响应不代表跨种子确认。\n')
    print(FIG / 'parameter_search_propagation_shared_plane.png')


if __name__=='__main__': main()
