"""Reconnect the measured local-search history without inventing a legacy score."""
from pathlib import Path
import csv, json, sys, warnings
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src/snn_engine')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import proj3d
from scripts.paper_figures import plot_topic4_compact_optimization_abcd as layout
from scripts.paper_figures.plot_topic4_workpoint_waveforms_trajectory import Arrow3D
from scripts.paper_figures.plot_topic4_data_driven_metric_tradeoffs import recompute

BASE = layout.BASE
EARLY = BASE/'geometry_ee_axis_three_observable_optimization_20260914'
OLD, XY, ROUND = layout.OLD, layout.XY, layout.ROUND
OUT = layout.SOURCE/'full_search_history'
FIG = OUT/'figures'
KEYS = layout.KEYS
read, write, sha = layout.read, layout.write, layout.sha
COLORS = plt.get_cmap('Blues')(np.linspace(.22, .90, 16))
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(np.arange(.5, 17), 16)
plt.rcParams.update({'font.family':'sans-serif', 'font.size':6, 'svg.fonttype':'none', 'pdf.fonttype':42})


def history():
    phases, identity = [], {}
    def add(name, ids, source, timestamp, objective, adaptive=False, training=None):
        number = len(phases)+1
        assert not set(ids) & set(identity), 'reused trajectories must not be counted again'
        for cid in ids: identity[cid] = number
        phases.append(dict(stage=number, name=name, ids=ids, N=len(ids), source=str(source),
                           source_sha256=sha(source), timestamp=timestamp, objective=objective,
                           adaptive=adaptive, training_ids=training))
    p = read(EARLY/'plan.json')
    add('历史参考与初期探测', p['initial_ids'], EARLY/'proposals/initial.json', p['created_unix'], 'mode_conditioned')
    prior = set(p['initial_ids'])
    for path in sorted((EARLY/'proposals').glob('adaptive_*.json')):
        d = read(path); seen = {r['candidate'] for r in d['training_data']}
        assert seen == prior and not set(d['ids']) & seen
        add(f'早期自适应 {d["batch"]}', d['ids'], path, d['time'], 'mode_conditioned', True, sorted(seen))
        prior.update(d['ids'])
    assert [len(x['training_ids']) for x in phases[1:]] == [16,20,24,28]
    p = read(OLD/'plan.json')
    assert set(p['warm_ids']) == prior
    add('加密单参数探测', p['dense_ids'], OLD/'plan.json', p['created_unix'], 'label_free')
    prior_bo = set()
    for path in sorted((OLD/'proposals').glob('batch_*.json')):
        d = read(path); seen = {r['candidate'] for r in d['training_snapshot']}
        assert prior <= seen and prior_bo <= seen and not set(d['ids']) & seen
        add(f'无标签自适应 {d["batch"]}', d['ids'], path, d['time'], 'label_free', True, sorted(seen))
        prior_bo.update(d['ids'])
    p = read(XY/'plan.json')
    add('双核二维网格', [cid for cid in p['initial_ids'] if cid not in identity], XY/'plan.json', p['created_unix'], 'label_free')
    d = read(XY/'proposals/recombination.json')
    add('按实测结果组合双核', d['ids'], XY/'proposals/recombination.json', d['time'], 'label_free',
        training=[x['candidate'] for x in d['training_snapshot']])
    p = read(ROUND/'plan.json'); assert set(p['historical_ids']) == set(identity)
    add('位置与连接交互探测', p['interaction_new_ids'], ROUND/'plan.json', p['created_unix'], 'joint_label_free')
    for path in sorted((ROUND/'proposals').glob('batch_*.json')):
        d = read(path); seen = {r['candidate'] for r in read(d['training_snapshot'])['rows']}
        assert set(identity) == seen and not set(d['ids']) & seen
        add(f'联合目标自适应 {d["batch"]}', d['ids'], path, d['time'], 'joint_label_free', True, sorted(seen))
    assert len(identity)==226 and len(phases)==16
    assert all(b['timestamp'] > a['timestamp'] for a,b in zip(phases,phases[1:]))
    assert [p['N'] for p in phases]==[16,4,4,4,4,70,8,8,8,8,46,4,18,8,8,8]
    write(OUT/'history.json', dict(phases=phases, candidate_stage=identity,
        source_scope='Complete 226-condition local-search lineage from September 14 onward; earlier substrate revisions are a separate provenance bridge, not a continuous fixed-model search.',
        order='Proposal/admission stage, never worker completion order. Reused initial controls enter stage 1 but were generated before it; dense probes were proposed together and ran alongside subsequent BO.',
        objective_breaks=[5.5,12.5], early_training_N=[16,20,24,28],
        objective_contract='Stages 1-5 used mode-conditioned training with two noise runs; stages 6-12 used one-noise label-free J; stages 13-16 used label-free J and J_joint. Plotted summaries use only topology2511/noise847401 throughout.',
        missing_legacy_score='Original hand-placed runs saved half-peak first-crossing dense ranks, not full contact envelopes/centroids. Do not assign them current centroid errors or substitute the later local reference.'))
    return phases, identity


def legacy_anchor():
    p = Path('/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/field_swap_subject_snn')
    stems=['epilepsiae_1146_twoend_equal_tsrc_s3','epilepsiae_1146_gradient_shared_corefrozen_cr1p5_s4_20260722']
    records=[]
    for stem in stems:
        jf=p/f'readout_{stem}.json'; nf=p/f'figdata_{stem}.npz'; j=read(jf)
        with np.load(nf,allow_pickle=True) as z:  # trusted project-owned historical artifact
            v=z['vth']; reg=z['reg'].item()
            records.append(dict(identity=stem, readout=str(jf), readout_sha256=sha(jf), arrays=str(nf), arrays_sha256=sha(nf),
                seed=j['seed'], contacts=z['contacts'].tolist(), contact_names=z['names'].tolist(),
                centers_mm=z['foci'].tolist(), radius_mm=float(z['core_r']), core_mean_mV=float(z['core_mean']),
                duration_ms=float(z['times'][-1]+np.diff(z['times'][:2])[0]), axis_deg=float(z['theta_deg']),
                threshold_lowered=int((v<18).sum()),threshold_raised=int((v>18).sum()),
                source_names=reg['source_names'],sink_names=reg['sink_names'],
                raw_spikes_saved=False, full_contact_envelopes_saved=False,
                current_three_metrics=None, compatibility='UNAVAILABLE: first-half-peak-crossing dense ranks with 2 ms ties cannot recover centroid order; detector/participation and noise/topology also differ.',
                configuration_status='Saved geometry and realized thresholds verified; artifact does not contain complete applied graph/noise configuration. Current defaults are not substituted.'))
    write(OUT/'handplaced_anchor_audit.json',records)
    return records


def measured_rows(identity):
    table={r['candidate']:r for r in csv.DictReader(open(ROUND/'analysis/observables.csv')) if r['mode']=='ALL'}
    patient=read(ROUND/'analysis/diagnostic_observables.json')['patient']['ALL']
    assert patient['N']==19770 and set(table)==set(identity)
    rows=[]; frozen=None
    for cid in sorted(identity,key=lambda cid:(identity[cid],cid)):
        score=read(ROUND/'scores'/f'{cid}.json'); src=Path(score['source']); tr=read(src)
        assert sha(src)==score['source_sha256']
        assert tr['physical_status']=='COMPLETE_NO_RUNAWAY' and tr['actual_duration_ms']==60000
        app=read(src.parents[1]/'applied_physics.json'); c=app['candidate']
        assert app['topology_seed']==2511 and app['dynamics_seed']==847401 and app['threshold']['n_raised']==0
        fixed={k:v for k,v in c['parameters'].items() if k not in ['EE_core_to_out_scale','EE_angle_offset_deg']}
        fixed.update(radii=c['radii_mm'],core_mean_rate_scale=c['core_mean_rate_scale'],core_ou_correlation=c['core_ou_correlation'],shape=c['shape'],outgoing=c['outgoing'])
        if frozen is None:frozen=fixed
        assert fixed==frozen
        actual=[*np.asarray(c['centers_mm']).ravel(),app['graph']['stage_audits']['weights']['factors']['EE_core_to_out_scale'],app['graph']['kernel']['angle_offset_deg']]
        np.testing.assert_allclose(actual,score['vector'],atol=1e-10,rtol=0)
        with np.load(src.with_suffix('.npz')) as z:
            ids=np.array([int(i) for i in z['primary_event_indices'] if tr['events'][i]['window_ms'][0]>=1500 and tr['events'][i]['window_ms'][1]<=60000],int)
            names=z['contact_names'].astype(str).tolist(); times=z['centroid_ms'][ids]
        assert len(ids)==score['N']==int(table[cid]['N'])
        values=recompute(times,names,patient)
        np.testing.assert_allclose(values,[float(table[cid][k]) for k in KEYS],atol=1e-12,rtol=0)
        rows.append(dict(candidate=cid,stage=identity[cid],N=len(ids),**dict(zip(KEYS,map(float,values))),
            J=score['J'],J_joint=score['J_joint'],vector=score['vector'],source=str(src),source_sha256=sha(src)))
    with (OUT/'all_workpoints_source.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(OUT/'fixed_background.json',frozen)
    return rows


def colorbar(fig,rect,small=True):
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=NORM,cmap=CMAP),cax=fig.add_axes(rect),orientation='horizontal',ticks=[1,5,10,16])
    bar.outline.set_visible(False);bar.ax.tick_params(length=0,pad=2,labelsize=6)
    for boundary in [5.5,12.5]:bar.ax.axvline(boundary,color='#222222',lw=.6)
    bar.set_label('入池／提案阶段：早 → 晚',fontsize=6,labelpad=2)
    return bar


def profiles(fig,rows,cases):
    ax=fig.add_axes([.684,.625,.282,.31])
    for r in rows:
        ax.plot(range(3),[r[k] for k in KEYS],color=COLORS[r['stage']-1],lw=.48,marker='o',ms=1.1,mew=0,zorder=1+r['stage']/20)
    lookup={r['candidate']:r for r in rows}
    ref=lookup['bridge_circle_out125_xminus075']
    ax.plot(range(3),[ref[k] for k in KEYS],color='#292929',ls='--',lw=1.05,marker='D',ms=2.3,label='局部搜索参考',zorder=4)
    for case in cases:
        r=lookup[case['candidate']]
        ax.plot(range(3),[r[k] for k in KEYS],color=COLORS[r['stage']-1],lw=1.5,marker='o',ms=3,mec='#555555',mew=.35,zorder=5)
        ax.annotate(case['panel'],(2,r['participation_error']),xytext=(9,0),textcoords='offset points',fontsize=7,weight='bold',va='center')
    ax.set(xlim=(-.12,2.28),ylim=(0,.5),yticks=np.arange(0,.51,.1),xticks=range(3),xticklabels=['平均\nrank','杆内\n顺序','触点\n参与'],ylabel='误差 ↓')
    ax.grid(axis='y',color='#EBEBEB',lw=.5);ax.set_axisbelow(True)
    ax.legend(loc='upper left',fontsize=6,handlelength=2)
    colorbar(fig,[.684,.560,.282,.012])
    fig.text(.825,.515,'1–5 分型目标｜6–12 无标签｜13–16 联合',fontsize=6,ha='center')


def save(fig,stem):
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); outside=[]; sizes=[]
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            bb=t.get_window_extent(renderer);sizes.append(t.get_fontsize())
            if bb.x0<-.5 or bb.y0<-.5 or bb.x1>fig.bbox.x1+.5 or bb.y1>fig.bbox.y1+.5:outside.append(t.get_text())
    assert not outside,outside
    assert min(sizes)>=6
    for ext in ['png','svg']:fig.savefig(FIG/f'{stem}.{ext}',dpi=400)
    plt.close(fig)
    return dict(stem=stem,min_font_pt=min(sizes),outside_canvas=outside)


def composite(rows,cases,grids,ref,response):
    fig=plt.figure(figsize=(183/25.4,206/25.4))
    layout.xy_panels(fig,grids,ref);layout.connections(fig,response);profiles(fig,rows,cases)
    for label,x,y,title,tx in [('A',.018,.974,'位置与连接参数响应',.080),('B',.615,.974,'完整局部搜索（226）',.684),
                            ('C',.018,.497,'平均顺序示例 · 自适应第5批',.069),('D',.514,.497,'参与概率示例 · 自适应第7批',.565)]:
        fig.text(x,y,label,fontsize=9,weight='bold');fig.text(tx,y,title,fontsize=7.5)
    examples=layout.examples(fig,cases)
    for x,cmap,norm,title,ticks in [(.145,'magma',plt.Normalize(0,1),'发放包络',[0,1]),(.657,'inferno',plt.Normalize(0,75),'原生活动 / 2 ms / 网格',[0,75])]:
        bar=fig.colorbar(plt.cm.ScalarMappable(cmap=cmap,norm=norm),cax=fig.add_axes([x,.024,.19,.009]),orientation='horizontal',ticks=ticks)
        bar.outline.set_visible(False);bar.ax.tick_params(length=0,pad=1,labelsize=6);bar.ax.set_title(title,fontsize=6,pad=2)
    # Reserve a narrow provenance strip without crowding the accepted A-D plate.
    scale=195/206
    positions=[(ax,ax.get_position().bounds) for ax in fig.axes]
    for ax,(x,y,w,h) in positions:
        ax.set_position([x,y*scale,w,h*scale])
    for t in fig.texts:
        x,y=t.get_position();t.set_position((x,y*scale))
    fig.text(.5,.985,'手放双核 → 基底修订 → 初期探测 → 早期自适应 → 后续优化',ha='center',va='top',fontsize=7)
    fig.text(.5,.963,'原手放读出与当前质心指标不可换算；黑虚线为修订后的局部搜索参考',ha='center',va='top',fontsize=6,color='#555555')
    for e in examples:
        e['local_search_stage']={'C':7,'D':9}[e['panel']]
        e['adaptive_batch_from_first_recorded_bo']={'C':5,'D':7}[e['panel']]
        e['old_label_free_only_batch']=e.pop('chronological_batch')
    write(OUT/'model_example_manifest.json',examples)
    return save(fig,'optimization_response_full_history_abcd')


def trajectory(rows,phases,anchors):
    lookup={r['candidate']:r for r in rows}; old_scores={}
    for path in (EARLY/'scores').glob('*.json'):
        d=read(path)
        if d.get('stage') in ['initial','adaptive'] and d.get('topology')==2511 and d.get('noise') in [847401,847402]:
            old_scores.setdefault(d['candidate'],[]).append(float(d['J']))
    records=[]
    for phase in phases:
        if not phase['adaptive']:continue
        def key(cid):
            if phase['objective']=='mode_conditioned':
                assert len(old_scores[cid])==2
                return float(np.mean(old_scores[cid]))
            return lookup[cid]['J']
        cid=min(phase['ids'],key=lambda cid:(key(cid),cid))
        records.append(dict(stage=phase['stage'],objective=phase['objective'],candidate=cid,
            original_selection_score=key(cid),xyz=[lookup[cid][k] for k in KEYS]))
    assert len(records)==11
    fig=plt.figure(figsize=(183/25.4,116/25.4));ax=fig.add_axes([.015,.14,.71,.81],projection='3d',computed_zorder=False)
    xyz=np.array([[r[k] for k in KEYS] for r in rows]);ax.set_proj_type('ortho');ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-53)
    ax.scatter(*xyz.T,c=[r['stage'] for r in rows],cmap=CMAP,norm=NORM,s=13,alpha=.88,depthshade=False,edgecolors='none',zorder=1)
    for objective,ls in [('mode_conditioned',':'),('label_free','--'),('joint_label_free','-')]:
        rr=[r for r in records if r['objective']==objective]
        for a,b in zip(rr,rr[1:]):
            start=np.array(a['xyz']);end=np.array(b['xyz'])
            ax.plot(*np.array([start,end]).T,color='#333333',ls=ls,lw=.9,zorder=3)
            ax.add_artist(Arrow3D(start+.27*(end-start),start+.73*(end-start),mutation_scale=7,arrowstyle='-|>',lw=.8,color='#333333',zorder=4))
    for r in records:ax.scatter(*r['xyz'],c=[COLORS[r['stage']-1]],marker='o',s=20,edgecolor='#333333',lw=.45,depthshade=False,zorder=4)
    ref=lookup['bridge_circle_out125_xminus075'];ax.scatter(*[ref[k] for k in KEYS],c='#222222',marker='D',s=29,depthshade=False,zorder=5)
    for panel,cid in [('C','lf_bo01_03'),('D','lf_bo03_07')]:
        r=lookup[cid];ax.scatter(*[r[k] for k in KEYS],c=[COLORS[r['stage']-1]],s=38,edgecolors='#222222',lw=.7,depthshade=False,zorder=5)
    for axis,lim,label in zip([ax.xaxis,ax.yaxis,ax.zaxis],[.4,.5,.4],['平均 rank 误差 ↓','杆内顺序误差 ↓','触点参与误差 ↓']):
        axis.set_ticks(np.arange(0,lim+.01,.1));axis.set_pane_color((1,1,1,0));axis._axinfo['grid'].update(color=(.89,.90,.92,1),linewidth=.5)
        axis.set_label_text(label,fontsize=7);axis.labelpad=4
    ax.set(xlim=(0,.4),ylim=(0,.5),zlim=(0,.4));ax.tick_params(pad=0,labelsize=6)
    fig.canvas.draw()
    for panel,cid in [('C','lf_bo01_03'),('D','lf_bo03_07')]:
        r=lookup[cid];x,y,_=proj3d.proj_transform(*[r[k] for k in KEYS],ax.get_proj())
        ax.annotate(panel,(x,y),xytext=(5,7),textcoords='offset points',fontsize=7,weight='bold',zorder=6)
    offsets={2:(-42,35),5:(-48,-6),7:(70,65),10:(7,-33),14:(80,-3),16:(42,-19)}
    for record in records:
        if record['stage'] not in offsets:continue
        x,y,_=proj3d.proj_transform(*record['xyz'],ax.get_proj())
        ax.annotate(str(record['stage']),(x,y),xytext=offsets[record['stage']],textcoords='offset points',
            fontsize=6,ha='center',va='center',zorder=7,
            arrowprops=dict(arrowstyle='-',color='#666666',lw=.4))
    colorbar(fig,[.14,.066,.47,.023])
    fig.text(.74,.925,'手放几何起点',fontsize=7,weight='bold')
    geo=fig.add_axes([.76,.62,.19,.30]);a=anchors[1];xy=np.array(a['contacts'])
    for shaft in ['SCL','ICL']:
        ix=[i for i,n in enumerate(a['contact_names']) if n.startswith(shaft)];geo.plot(*xy[ix].T,'o-',color='#89939D',ms=1.6,lw=.5)
    for i,c in enumerate(a['centers_mm']):
        geo.add_patch(Circle(c,a['radius_mm'],ec='#222222',fc='none',lw=.85));geo.text(*c,'AB'[i],ha='center',va='center',fontsize=6)
    geo.set(xlim=(0,20),ylim=(0,20),xticks=[0,20],yticks=[0,20],aspect='equal',xlabel='x (mm)',ylabel='y (mm)');geo.tick_params(pad=1,length=2,labelsize=6)
    fig.text(.74,.55,'原位、半径 1.5 mm\n旧读出：当前三指标不可换算',fontsize=6,va='top',linespacing=1.6)
    handles=[Line2D([],[],color='#333333',ls=ls,lw=1,label=s) for ls,s in [(':','2–5：分型目标'),('--','7–10：无标签'),('-','14–16：联合目标')]]
    handles.append(Line2D([],[],color='#222222',marker='D',ms=3,ls='none',label='局部搜索参考'))
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.735,.44),fontsize=6,handlelength=1.8,labelspacing=.8)
    fig.text(.74,.225,'箭头：各自适应批次的代表点\n跨目标不连线；批内无先后\n全部 226 个条件均保留',fontsize=6,linespacing=1.6,va='top')
    write(OUT/'trajectory_representatives.json',dict(records=records,
        rule='Minimum original mode-conditioned two-noise J in early batches; minimum frozen label-free J in later batches, including the J/J_joint phase. Descriptive batch representatives, not Pareto incumbents or parent-child ancestry.',
        edges='Only consecutive adaptive batches inside each objective regime. Grid/recombination proposals are shown as points, not given an invented BO trajectory.',manual_baseline_metric_point=False))
    return save(fig,'three_observable_full_search_history')


def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True)
    layout.OUT=OUT;layout.FIG=FIG
    _,cases,grids,ref,response=layout.load()
    phases,identity=history();anchors=legacy_anchor();rows=measured_rows(identity)
    contract=read(OUT/'figure_contract.json')
    contract.update(question='Reconnect early probes and adaptive feedback to the current local-search history; preserve the true hand-placed model as a separate origin when no compatible metric is recoverable.',
        panels=dict(A='Retain measured conditional XY grids and EE/angle curves.',B='All 226 compatible local-search conditions, 16 ordered proposal/admission stages, two objective switches; black dashed line is the revised local-search reference, never the original hand-placed baseline.',C='Same former P1 events, now correctly adaptive batch5/stage7.',D='Same former P3 events, now correctly adaptive batch7/stage9.'),
        stages=16,conditions=226,adaptive_batches=11,new_simulations=0,
        raw_recomputation='All 226 centroid arrays recomputed with the same frozen patient FIT summaries; SCL and ICL equally weighted. Every pair required readable in this comparison.',
        baseline='Original hand-placed saved geometry/threshold arrays verified. Current centroid observables unavailable; baseline cannot appear as a numeric line or 3-D point.',
        exclusions='No compatible local-search candidate excluded. Pre-September14 substrate campaigns are an explicit provenance bridge with different physics/readouts, not a reconstructed fixed-loss optimization trajectory.',
        optimization='Early proposals used TA/TB and two noise runs. Later proposals are label-free. Unified display does not retroactively make early selection label-free.',
        output=dict(width_mm=183,height_mm=206,dpi=400,formats=['png','svg'],min_font_pt=6))
    write(OUT/'figure_contract.json',contract)
    plots=[composite(rows,cases,grids,ref,response),trajectory(rows,phases,anchors)]
    from PIL import Image
    import xml.etree.ElementTree as ET
    for f in FIG.glob('*.png'):
        with Image.open(f) as im:im.load()
    for f in FIG.glob('*.svg'):
        tree=ET.parse(f);assert any(x.tag.endswith('text') for x in tree.getroot().iter())
    write(OUT/'qa.json',dict(plots=plots,conditions=226,centroid_metrics_recomputed=226,all_three_metrics_match_atol=1e-12,
        initial_N=16,early_adaptive_N=16,early_training_N=[16,20,24,28],phase_counts=[p['N'] for p in phases],
        proposal_feedback_checked=True,chronology_checked=True,fixed_physics_checked=True,all_outputs_decoded=True,
        original_handplaced_numeric_comparability='NOT_ESTABLISHED',agent_visual_review='PENDING',human_visual_review='PENDING'))
    (FIG/'README.md').write_text('### optimization_response_full_history_abcd.png\nA保留二维XY网格与连接响应，B纳入完整局部搜索谱系的226个条件，按16个入池/提案阶段使用同一蓝色色标；黑虚线为修订后的局部搜索参考，不能称原手放基线。1–5阶段训练含模式信息，6–12为无标签目标，13–16为无标签双目标；C/D沿用原事件，分别为自适应第5/7批。\n原手放记录为半峰首次越阈顺序，不能转换为当前质心观测，故仅保留历史起点，不伪造其同尺度指标；A的XY与连接曲线仍有各自固定背景。\n**关注点**：早期自适应已被纳入；三项描述性观测的取舍，不是同一损失的连续收敛或患者完整双模式恢复。\n\n### three_observable_full_search_history.png\n226个实测点使用与合图一致的16阶段色标，箭头只连接同一目标阶段内各自适应批次的实测代表点，不跨目标边界，也不按worker结束时间排序。右上为原手放几何在shared-plane记录中的实际位置与半径；它没有可比的三指标数值点。\n**关注点**：历史起点、后续参数搜索、目标改变分别解释；早期代表按当时两噪声平均J选择，后期按原无标签J选择，轨迹仅为批次摘要而非累计最优。\n')
    (OUT/'scientific_note.md').write_text('本次纠正的是搜索历史和图示，不修改优化目标、候选或物理仿真。\n\n16个初期参考/探测点之后，四批早期自适应训练集分别为16/20/24/28；再接70个加密探测、32个无标签自适应、46个新二维网格、4个反馈组合、18个交互探测和24个双目标自适应，共226个不同条件。复用条件只计一次，早期原16中也含更早已有参考。\n\n早期选择用两噪声和TA/TB条件目标；统一展示只取已有拓扑2511/噪声847401的60秒轨迹，不能将重评分说成早期没有用标签。全部226个点从保存的质心重算三个观测，沿用同一个19770事件患者FIT目标，不重做训练。\n\n原手放的首次半峰越阈时间与当前质心时间不相同，旧NPZ没有保存完整发放/接触包络，无法离线还原当前三指标。旧artifact保存的几何、阈值可核查，完整应用连接/噪声配置未随artifact保存，故不使用当前默认值补齐，也不把局部搜索参考冒充原手放基线。手放历史模型只能作为独立来源节点；绘制严格匹配基线差值的证据目前缺失。\n\n本图范围为9月14日起226条件的完整局部搜索，加上此前手放/基底修订的来源说明；不是把全部历史自由场及噪声/阈值修订都重写为同一次优化。早期候选和后期候选的同尺度摘要可比较，但跨目标轨迹不是固定损失收敛，也不能据此归因优化算法优越。\n')
    print(json.dumps(dict(out=str(OUT),plots=plots,conditions=len(rows),stages=len(phases)),ensure_ascii=False))


if __name__=='__main__':main()
