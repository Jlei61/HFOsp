#!/usr/bin/env python3
"""Fig4 candidate: D/E workpoints, continuous readout F, patient comparisons G/H.

No simulations or refitting. Patient labels come from all-event Timing+Space;
model labels in G/F are the frozen labels already used in the propagation plate.
H retains the legacy alternating-contact cross-fit, with fresh permutation tests.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import rankdata, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from src.topic4_d6_natural_kmeans import (
    normalize_event_ranks, patient_profiles, contact_split_folds,
    crossfit_patient_readout,
)
from src.topic4_nlc_null_calibration import (
    crossfit_matrix, contact_permutation_matrix_draws,
)

OLD = ROOT/'results/paper-ready-figure/fig4/candidates/geometry_prior_parameter_response_20260918'
OUT = ROOT/'results/paper-ready-figure/fig4/candidates/geometry_prior_patient_comparison_20260918'
FIG = OUT/'figures'
SOURCE = OUT/'source'
PATIENT = ROOT/'results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/shaft_aware_patient_training_target.npz'
CONTRACT = PATIENT.with_name('contact_shaft_contract.json')
FIELD = ROOT/'results/interictal_propagation_masked/template_gradient_fields_all_events_timing_plus_space/per_subject/epilepsiae_1146.json'
DISPLAY = [f'SCL{i}' for i in range(9,5,-1)]+[f'ICL{i}' for i in range(11,0,-1)]
RED, BLUE = '#C63D3A', '#287FA1'
SHAFT = {'SCL':'#159CAC', 'ICL':'#E08229'}
MM = 1/25.4


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def style():
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'],
        'font.size':9, 'axes.labelsize':9, 'xtick.labelsize':9, 'ytick.labelsize':9,
        'legend.fontsize':9, 'legend.frameon':False, 'axes.linewidth':.65,
        'axes.spines.top':False, 'axes.spines.right':False,
        'svg.fonttype':'none', 'savefig.facecolor':'white'})


def save(fig, letter, bbox=None):
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    assert min(t.get_fontsize() for t in texts) >= 8
    assert all(not any('\u4e00' <= c <= '\u9fff' for c in t.get_text()) for t in texts)
    if bbox is None:
        outside = [t.get_text() for t in texts if not fig.bbox.contains(*t.get_window_extent(rr).p0)
                   or not fig.bbox.contains(*t.get_window_extent(rr).p1)]
        assert not outside, outside
    positions = {ax:ax.get_position().frozen() for ax in fig.axes}
    for ext in ['png','svg']:
        fig.savefig(FIG/f'fig4-panel{letter.lower()}.{ext}', dpi=400, bbox_inches=bbox)
        for ax,pos in positions.items(): ax.set_position(pos)
    plt.close(fig)


def load():
    selection = json.loads((OLD/'source/selected_workpoints.json').read_text())
    cases = [dict(c, panel=p) for c,p in zip(selection['cases'], ['D','E'])]
    case = cases[1]
    assert case['candidate'] == 'xy_left_20'
    trajectory = Path(case['source'])
    assert sha(trajectory) == case['source_sha256']
    run = json.loads(trajectory.read_text())
    assert run['status'] == 'COMPLETE' and run['actual_duration_ms'] == 60000
    assert sha(trajectory.with_suffix('.npz')) == run['arrays_sha256']
    needed = ['contact_names','contact_envelope','contact_envelope_dt_ms','centroid_ms',
              'recruitment_ms','primary_event_indices','event_mode','event_time_ms']
    with np.load(trajectory.with_suffix('.npz')) as z: a = {k:z[k] for k in needed}
    ids = np.array([i for i in a['primary_event_indices']
        if run['events'][i]['window_ms'][0] >= 1500
        and run['events'][i]['window_ms'][1] <= 60000], int)
    assert len(ids) == case['N'] == 151
    for i in ids:
        assert run['events'][i]['event_index'] == i and run['events'][i]['primary_eligible']
        assert run['events'][i]['mode'] == a['event_mode'][i]
    names = a['contact_names'].astype(str).tolist()
    order = np.array([names.index(n) for n in DISPLAY])
    with np.load(PATIENT) as z:
        p = {k:z[k] for k in ['contact_names','patient_train_ranks','patient_train_event_indices',
                             'patient_train_old_labels','patient_train_block_ids']}
    assert names == p['contact_names'].astype(str).tolist()
    field = json.loads(FIELD.read_text())
    assert sha(FIELD) == 'b5b0d65823f13caed389c890295887bcea5ca3f79fbfd27c4814962a2f962c1b'
    discovery = field['template_discovery']
    assert np.array_equal(discovery['sampled_event_indices'], np.arange(46683))
    # Frozen semantic mapping: spatial cluster 0 is TA; model mode 1 is TA.
    patient_labels = 1-np.asarray(discovery['event_labels'], int)[p['patient_train_event_indices']]
    assert len(patient_labels) == 30049
    assert np.array_equal(np.bincount(patient_labels), [9196,20853])
    assert np.sum(patient_labels != p['patient_train_old_labels']) == 1861
    ranks = np.full((len(ids),15), np.nan)
    for out,i in zip(ranks,ids):
        row = a['centroid_ms'][i]; valid = np.isfinite(row)
        out[valid] = rankdata(row[valid], method='average')-1
    normalized = normalize_event_ranks(ranks)
    assert np.array_equal(np.isfinite(normalized), np.isfinite(a['centroid_ms'][ids]))
    labels = a['event_mode'][ids]
    assert set(labels) == {0,1}
    profiles = patient_profiles(p['patient_train_ranks'], patient_labels)
    model_profiles = np.array([np.nanmean(normalized[labels == k],axis=0) for k in [0,1]])
    contract = json.loads(CONTRACT.read_text())
    assert [r['contact_name'] for r in contract['contacts']] == names
    folds = contact_split_folds(contract)
    cf = crossfit_patient_readout(ranks,p['patient_train_ranks'],patient_labels,folds)
    fast = crossfit_matrix(normalized, profiles, folds)
    assert np.allclose(fast,cf['matrix'],rtol=0,atol=1e-12,equal_nan=True)
    null = contact_permutation_matrix_draws(ranks,p['patient_train_ranks'],patient_labels,folds,
        draws=1000, seed=20260918, shaft_ids=[n[:3] for n in names])
    tests = {}
    for m,name in [(1,'MTA_vs_TA'),(0,'MTB_vs_TB')]:
        draws = null[:,m,m]; assert np.all(np.isfinite(draws))
        pv = (1+np.sum(draws >= cf['matrix'][m,m]-1e-12))/1001
        tests[name] = dict(observed=cf['matrix'][m,m],p_one_sided=pv,n_draws=1000,
            null_quantiles=np.quantile(draws,[.05,.5,.95]),
            stars='***' if pv<=.001 else '**' if pv<=.01 else '*' if pv<=.05 else 'n.s.')
    np.savez_compressed(SOURCE/'comparison_arrays.npz',contact_names=names,
        display_order=order,model_event_ids=ids,model_ranks=ranks,model_labels=labels,
        model_profiles_raw_mode_0_1=model_profiles,patient_profiles_raw_mode_0_1=profiles,
        patient_global_event_ids=p['patient_train_event_indices'],patient_spatial_labels=patient_labels,
        crossfit_matrix_raw_mode_0_1=cf['matrix'],permutation_matrices_raw_mode_0_1=null)
    provenance = dict(workpoint=case,topology_seed=2511,dynamics_seed=847401,n_networks=1,
        duration_ms=60000,burnin_ms=1500,n_model_events=len(ids),
        model_counts_TA_TB=[int(np.sum(labels==k)) for k in [1,0]],
        patient_n_events=len(patient_labels),patient_counts_TA_TB=[20853,9196],
        patient_n_blocks=len(np.unique(p['patient_train_block_ids'])),
        patient_pool='Existing 30049 qualified development events, with exact global-ID relabeling from the all-event Timing+Space discovery. Not an independent validation patient or block split.',
        patient_discovery_n_events=46683,patient_label_change_vs_timing_only=1861,
        model_labels='Frozen trajectory event_mode (1=TA,0=TB); same labels as D/E. No reclustering for G or F.',
        profile_definition='Finite-participant event ranks rescaled to 0..1; each contact averages over events participating at that contact, then multiplied by 14 for display. No missing-contact rank imputation.',
        G_grouping='Frozen model labels; patient labels are Timing+Space. This differs from legacy G using frozen KMeans model clusters.',
        H_grouping='Assign each model event to the patient profiles using one alternating-contact fold; evaluate Spearman on the disjoint fold; swap and average. H groups need not equal G frozen labels.',
        H_statistical_unit='One topology/noise trajectory. Conditional post-selection correspondence check, not search-corrected inference or out-of-patient generalization.',
        H_null='1000 within-shaft contact-identity permutations; each permutation is shared by all events and repeats both assignment/evaluation steps. Fresh one-sided diagonal tests, unadjusted for two tests and workpoint selection.',
        contact_order_top_to_bottom=DISPLAY,crossfit=cf,diagonal_tests=tests,
        sources={str(x):sha(x) for x in [trajectory,trajectory.with_suffix('.npz'),PATIENT,CONTRACT,FIELD]})
    return cases,run,a,ids,order,model_profiles,profiles,cf,tests,provenance


def rank_panel(order, model, patient):
    fig = plt.figure(figsize=(87*MM,98*MM))
    ax = fig.add_axes([.195,.225,.765,.745])
    for mode,color in [(1,RED),(0,BLUE)]:
        for values,ls,marker in [(model[mode],'-','o'),(patient[mode],'--',None)]:
            for sl in [slice(0,4),slice(4,15)]:
                pos = np.arange(15)[sl]
                ax.plot(values[order][sl]*14,pos,ls=ls,marker=marker,color=color,
                        markersize=2.8,lw=1.15)
    ax.axhline(3.5,color='#BABABA',lw=.6)
    ax.set(ylim=(14.65,-.65),xlim=(-.3,14.3),xticks=[0,4,8,12,14],
           yticks=np.arange(15),yticklabels=DISPLAY,xlabel='Mean rank')
    ax.grid(axis='x',color='#E1E3E5',lw=.6); ax.set_axisbelow(True)
    ax.tick_params(length=2.5,pad=3)
    for t in ax.get_yticklabels(): t.set_color(SHAFT[t.get_text()[:3]])
    handles = [Line2D([],[],color=c,ls=ls,marker=mk,lw=1.15,ms=3,label=label)
               for c,ls,mk,label in [(RED,'-','o','Model TA'),(RED,'--',None,'Patient TA'),
                                      (BLUE,'-','o','Model TB'),(BLUE,'--',None,'Patient TB')]]
    fig.legend(handles=handles,ncol=2,loc='lower center',bbox_to_anchor=(.57,.005),
               columnspacing=1.3,handlelength=2.1,handletextpad=.5)
    save(fig,'G')


def matrix_panel(cf,tests):
    fig = plt.figure(figsize=(92*MM,69*MM))
    ax = fig.add_axes([15/92,12/69,49/92,49/69])
    matrix = cf['matrix'][np.ix_([1,0],[1,0])]
    im = ax.imshow(matrix,cmap='RdBu_r',vmin=-1,vmax=1,aspect='equal')
    for i in range(2):
        for j in range(2):
            color = 'white' if abs(matrix[i,j])>.45 else '#222222'
            ax.text(j,i-(.09 if i==j else 0),f'{matrix[i,j]:+.2f}',ha='center',va='center',
                    color=color,fontsize=12)
            if i==j:
                key = ['MTA_vs_TA','MTB_vs_TB'][i]
                ax.text(j,i+.24,tests[key]['stars'],ha='center',va='center',color=color,fontsize=9)
    ax.set(xticks=[0,1],xticklabels=['TA','TB'],yticks=[0,1],yticklabels=['MTA','MTB'])
    for t,c in zip(ax.get_xticklabels(),[RED,BLUE]): t.set_color(c)
    for t,c in zip(ax.get_yticklabels(),[RED,BLUE]): t.set_color(c)
    ax.tick_params(length=2.5,pad=4)
    cb = fig.colorbar(im,cax=fig.add_axes([68/92,12/69,2.5/92,49/69]),ticks=[-1,-.5,0,.5,1])
    cb.outline.set_visible(False); cb.set_label('Spearman ρ',labelpad=3)
    cb.ax.tick_params(length=2,pad=3)
    save(fig,'H')


def waveform_panel(run,a,ids,order):
    # Legacy pair rule adapted to current primary events; no patient-distance choice.
    events = run['events']; labels = a['event_mode']
    def center(i): return float(np.mean(events[i]['qualifying_interval_ms']))
    choices = []
    for i in ids[labels[ids]==1]:
        for j in ids[labels[ids]==0]:
            n1,n2 = [int(np.isfinite(a['recruitment_ms'][k]).sum()) for k in [i,j]]
            gap = abs(center(i)-center(j))
            choices.append(dict(ta=int(i),tb=int(j),gap=gap,n1=n1,n2=n2))
    pool = [r for r in choices if 250 <= r['gap'] <= 1200] or choices
    pair = min(pool,key=lambda r:(-min(r['n1'],r['n2']),-(r['n1']+r['n2']),abs(r['gap']-550),r['ta'],r['tb']))
    i,j = pair['ta'],pair['tb']
    width = max(760,pair['gap']+280)
    dt = float(a['contact_envelope_dt_ms']); duration = run['actual_duration_ms']
    center_pair = .5*(center(i)+center(j))
    start = max(1500,min(duration-width,center_pair-width/2)); stop = start+width
    t = np.arange(a['contact_envelope'].shape[1])*dt
    filt = sosfiltfilt(butter(4,[30,80],btype='bandpass',fs=1000/dt,output='sos'),
                      a['contact_envelope'],axis=1)
    select = (t>=start)&(t<=stop); trace = filt[order][:,select]
    scale = max(float(np.quantile(abs(trace),.99)),1e-9)
    fig = plt.figure(figsize=(180*MM,67*MM))
    ax = fig.add_axes([.094,.17,.895,.73])
    plotted = (14-np.arange(15))[:,None]*1.25+.72*trace/scale
    for row,name in enumerate(DISPLAY):
        ax.plot(t[select]-start,plotted[row],
                color=SHAFT[name[:3]],lw=.7)
    spans = []
    for k,c in [(i,RED),(j,BLUE)]:
        valid = a['recruitment_ms'][k]; valid = valid[np.isfinite(valid)]
        lo,hi = float(valid.min()-12),float(valid.max()+12)
        ax.axvspan(lo-start,hi-start,color=c,alpha=.14,lw=0,zorder=-2)
        spans.append(dict(event=k,mode='TA' if labels[k]==1 else 'TB',
                          recruitment_onset_span_ms=[lo+12,hi-12],shaded_span_ms=[lo,hi]))
    ax.axhline((14-3.5)*1.25,color='#C8C8C8',ls=':',lw=.55,zorder=-1)
    ax.set(xlim=(0,width),ylim=(float(plotted.min()-.3),float(plotted.max()+.3)),yticks=(14-np.arange(15))*1.25,yticklabels=DISPLAY,
           xlabel='Time (ms)',ylabel='30–80 Hz activity',xticks=np.arange(0,width+1,100))
    ax.tick_params(length=2.5,pad=2)
    for text in ax.get_yticklabels(): text.set_color(SHAFT[text.get_text()[:3]])
    ax.legend(handles=[Patch(facecolor=RED,alpha=.22,label='MTA'),Patch(facecolor=BLUE,alpha=.22,label='MTB')],
              ncol=2,loc='lower right',bbox_to_anchor=(1,1.01),borderaxespad=0,
              handlelength=1.5,columnspacing=1.4)
    save(fig,'F')
    np.savez_compressed(SOURCE/'waveform_arrays.npz',time_ms=t[select]-start,
        absolute_time_ms=t[select],filtered_contact_activity=trace,contact_names=DISPLAY,
        common_amplitude_scale=scale)
    return dict(selection=pair,selection_rule='All eligible TA/TB pairs, prefer 250–1200 ms gap; maximize min contact count, then sum contact count, then minimize gap distance to 550 ms; event-ID tie break. No patient-distance criterion.',
        absolute_window_ms=[start,stop],spans=spans,common_amplitude_scale=scale,
        all_primary_ids_visible=[int(k) for k in ids if start<=center(k)<=stop],
        quantity='30–80 Hz fourth-order Butterworth zero-phase filtering of the full continuous virtual-contact firing-density envelope before cropping. Not raw SEEG, LFP, or patient HFO.',
        dt_ms=dt,normalization='One q99 absolute amplitude for all contacts in the displayed continuous segment; no per-contact rescaling.',
        onset_shading='Actual participating-contact recruitment-onset span plus 12 ms on each side; not centroid times.')


def update_b_and_copy(cases):
    # Reuse the accepted renderer; only the two case-reference letters change.
    import src
    src.__path__.insert(0,str(ROOT/'.worktrees/topic4-continuous-core-state-r1/src'))
    from scripts.paper_figures import build_fig4_geometry_prior_candidate as old
    plot = old.plot
    rows = list(csv.DictReader(open(OLD/'source/all_workpoints_source.csv')))
    for r in rows:
        for k in plot.base.KEYS+['J','J_joint']: r[k] = float(r[k])
        r['stage'] = int(r['stage']); r['N'] = int(r['N'])
    assert len(rows) == 226
    plot.base.HEIGHT = 190
    fig = plt.figure(figsize=(plot.base.WIDTH*MM,190*MM))
    plot.search_panels(fig,rows,cases)
    assert len(fig.axes)==3
    all_text = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert all_text.count('D')==2 and all_text.count('E')==2
    assert not any(t in ['C'] for t in all_text)
    bbox = old.panel_bbox(fig,fig.axes,fig.texts)
    save(fig,'B',bbox=bbox)
    for new,previous in [('a','a'),('d','c'),('e','d')]:
        for ext in ['png','svg']:
            shutil.copy2(OLD/'figures'/f'fig4-panel{previous}.{ext}',FIG/f'fig4-panel{new}.{ext}')
    return {str(Path(plot.__file__)):sha(Path(plot.__file__)),
            str(Path(plot.base.__file__)):sha(Path(plot.base.__file__))}


def main():
    for p in [FIG,SOURCE]: p.mkdir(parents=True,exist_ok=True)
    style()
    cases,run,a,ids,order,model,patient,cf,tests,provenance = load()
    print(json.dumps(clean(dict(n=151,counts=provenance['model_counts_TA_TB'],matrix=cf['matrix'],tests=tests))),flush=True)
    rank_panel(order,model,patient)
    matrix_panel(cf,tests)
    provenance['waveform'] = waveform_panel(run,a,ids,order)
    renderer_sources = update_b_and_copy(cases)
    for filename in ['selected_workpoints.json','propagation_manifest.json','figure_contract.json']:
        shutil.copy2(OLD/'source'/filename,SOURCE/('previous_'+filename))
    write(SOURCE/'selected_workpoints.json',dict(cases=cases,remap={'C':'D','D':'E'},comparison_workpoint='xy_left_20'))
    write(OUT/'comparison_metadata.json',provenance)
    records = {}
    for letter in 'ABDEFGH':
        png = FIG/f'fig4-panel{letter.lower()}.png'; svg = png.with_suffix('.svg')
        with Image.open(png) as im: im.load(); pixels=list(im.size)
        root = ET.parse(svg).getroot()
        assert any(e.tag.endswith('text') for e in root.iter())
        records[letter] = dict(pixels=pixels,files={str(p.relative_to(OUT)):sha(p) for p in [png,svg]})
    registry = dict(schema='paper_fig4_geometry_patient_comparison_v2',
        asset_id='patient_geometry_prior_snn_patient_comparison',status='CANDIDATE',
        paper_slot='Fig4-A/B/D/E/F/G/H candidate',producer=str(Path(__file__).relative_to(ROOT)),
        previous_candidate=str(OLD.relative_to(ROOT)),panel_remap={'old C':'D','old D':'E'},
        panel_cases={c['panel']:c['candidate'] for c in cases},FGH_workpoint='xy_left_20',
        panels=records,new_simulations=0,previous_candidate_preserved=True,
        patient_template_contract='all-event Timing+Space; profiles on existing 30049 development events',
        statistical_unit='One topology2511/noise847401 trajectory with 151 eligible events',
        scientific_status='DESCRIPTIVE_PATIENT_TEMPLATE_COMPARISON; NOT_MECHANISM_RECOVERY',
        source_files={str(Path(__file__)):sha(Path(__file__)),**renderer_sources,**provenance['sources']},
        source_array_hashes={str(p.relative_to(OUT)):sha(p) for p in SOURCE.glob('*.npz')},
        agent_visual_review='PENDING',human_visual_review='PENDING',
        full_layout='Standalone panels supplied. Panel C composition is not defined by this update; no invented C or full A–H layout.')
    write(OUT/'figure4_candidate_registry.json',registry)
    notes = [
        ('fig4-panela.png','沿用上一候选的参数响应与配准双核位置图，数据和画法均未改变。','仍为已完成的原范围扫描，不混入后台扩大范围的结果。'),
        ('fig4-panelb.png','226 个工作点及 16 个历史提案阶段保持不变；原 C/D 标记改为 D/E，普通字重和迭代色标不变。','D=lf_bo04_07，E=xy_left_20；历史上有目标变更，不解释为单一损失连续收敛。'),
        ('fig4-paneld.png','原候选 C 原样更名为 D，仍为较高误差 lf_bo04_07 的模型模式代表事件及同步原生场。','事件未重选，颜色和时间尺度保持原合同。'),
        ('fig4-panele.png','原候选 D 原样更名为 E，仍为较低误差 xy_left_20 的模型模式代表事件及同步原生场。','E 是本次 F/G/H 的唯一模型工作点。'),
        ('fig4-panelf.png',f"E 工作点的同一连续记录：事件 {provenance['waveform']['selection']['ta']}（TA）和 {provenance['waveform']['selection']['tb']}（TB）；SCL/ICL 行序固定。全记录发放密度包络经 30–80 Hz 滤波后截窗，阴影取实际招募起始跨度，使用共同幅度尺度。",'这是虚拟接触发放密度的带通读出，不是模拟或真实 HFO 电压；连续中间活动完整保留，选例规则见 metadata。'),
        ('fig4-panelg.png',f"E 工作点 151 个合格事件，模型 TA/TB 各 {provenance['model_counts_TA_TB'][0]}/{provenance['model_counts_TA_TB'][1]}；与 30049 个患者开发事件在 Timing+Space 标签下的平均 rank 对比。参与触点的事件内 rank 归一化后按触点取均值，并换算到 0–14；实线模型、虚线患者，红色 TA、蓝色 TB。",'模型沿用传播图的冻结标签，未重新聚类；缺失触点不填 rank，杆间不连线。曲线接近不代表参与率、传播时差或动力学机制全部恢复。'),
        ('fig4-panelh.png','E 工作点的触点交叉匹配：在各杆交替的一半触点按患者模板分组，另一半触点评估 Spearman，交换后平均。对角格标记来自本次重新计算的 1000 次杆内触点置换（***≤0.001，**≤0.01，*≤0.05，n.s.>0.05）。','只有 1 张网络；置换检验未校正搜索择优和两个对角检验。H 的每折模板分组不同于 G 的冻结标签，不能视为同一分组的独立患者验证或机制证据。')]
    (FIG/'README.md').write_text('\n\n'.join(f'### {n}\n{d}\n**关注点**：{f} 同名 SVG 保留可编辑文字。' for n,d,f in notes)+'\n')
    print(FIG,flush=True)


if __name__=='__main__': main()
