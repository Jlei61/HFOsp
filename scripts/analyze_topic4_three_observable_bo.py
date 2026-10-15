"""Patient-calibrated objective, incremental scoring, and fixed-layout reports."""
from pathlib import Path
import argparse,csv,fcntl,hashlib,json,pickle,sys,time,warnings
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import run_topic4_three_observable_bo as run
from src.topic4_three_observable_objective import ThreeObservableObjective,GROUPS,normalized_ranks
rt=run.rt;OUT=run.OUT;A=OUT/'analysis';F=A/'figures'
DISPLAY=['SCL9','SCL8','SCL7','SCL6']+[f'ICL{i}' for i in range(11,0,-1)]

def point_label(cid):
    plan=rt.read(OUT/'plan.json');ref=rt.read(OUT/'candidates'/f"{plan['reference_id']}.json")
    if cid==ref['id']:return '左移固定参考'
    x=run.vector(rt.read(OUT/'candidates'/f'{cid}.json'));base=run.vector(ref)
    names=['左核X','左核Y','右核X','右核Y','向外EE','轴偏移']
    text=[f'{names[j]}{x[j]-base[j]:+.2f}'+('mm' if j<4 else '°' if j==5 else '') for j in range(6) if abs(x[j]-base[j])>1e-9]
    return ' / '.join(text)



from scripts.topic4_three_observable_plot_labels import plot_label, parameter_table

def dump_pickle(path,value):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:pickle.dump(value,f,pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)

def patient():
    parent=rt.read(run.physical.base.PARENT);ev=rt.load_evaluator(parent)
    names=rt.load_observation_contract(parent)['contact_names'];return ev,names,parent

def chronology(ev):
    target=run.MAIN/'results/topic4_sef_hfo/data_driven_core_field_rev10_sa/shaft_aware_target/shaft_aware_patient_training_target.npz'
    clock=run.MAIN/'results/topic5_preseizure_template_share/epilepsiae_1146/event_index.npz'
    with np.load(target) as z:
        raw=z['patient_train_event_indices'];blocks=z['patient_train_block_ids'];t=z['patient_train_onsets']
    # Upstream shaft target stores seconds; the frozen SNN contract stores ms.
    contract=run.MAIN/'results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability/objective_qualification/patient_training_contract_v1.npz'
    with np.load(contract) as z:
        assert np.array_equal(z['patient_train_onsets_ms'],ev.patient,equal_nan=True)
        assert np.array_equal(z['patient_train_block_ids'],blocks)
    assert np.array_equal(blocks,ev.blocks)
    np.testing.assert_allclose(t.astype(np.float64)*1000,ev.patient,rtol=1e-7,atol=2e-5,equal_nan=True)
    with np.load(clock) as z:
        mapping={(int(b),int(i)):float(tm) for b,i,tm in zip(z['source_block_id'],z['source_event_index'],z['event_abs_time'])}
    times=np.array([mapping[(int(b),int(i))] for b,i in zip(blocks,raw)])
    cal=np.asarray(ev.index['CAL']);order=np.lexsort((times[cal],blocks[cal]));cal=cal[order]
    for b in np.unique(blocks[cal]):assert np.all(np.diff(times[cal][blocks[cal]==b])>=0)
    rt.atomic_npz(A/'patient_event_identity.npz',raw_event_indices=raw,blocks=blocks,event_abs_time=times,cal_chronological_indices=cal)
    rt.write(A/'patient_provenance.json',dict(evaluator=rt.read(run.physical.base.PARENT)['sources']['evaluator'],target=str(target),target_sha256=rt.sha(target),clock=str(clock),clock_sha256=rt.sha(clock),
        event_identity_matches=True,upstream_time_unit='seconds',frozen_training_time_unit='milliseconds',unit_factor=1000,
        canonical_contract=str(contract),canonical_contract_sha256=rt.sha(contract),
        cal_order='chronological within original block',clinical_cohort='original frozen FIT/CAL retained, not a new clinical cohort',
        labels='SNN frozen classifier TA=1 TB=0; no current patient-state label bank consumed'))
    return cal

def sanity(obj,ev,device):
    rng=np.random.default_rng(2026091411);results=[]
    for rep in range(8):
        ids=rng.choice(len(ev.fit),128,replace=False);x=ev.fit[ids].copy();lab=obj.labels(x)
        variants={'patient_resample':x}
        collapsed=x.copy()
        for k in [0,1]:
            ix=np.flatnonzero(lab==k)
            if len(ix):collapsed[ix]=x[ix[0]]
        variants['mode_collapsed']=collapsed
        shifted=x.copy();shifted[:,obj.scl]+=35;variants['SCL_plus_35ms']=shifted
        shuffled=x.copy()
        for row in shuffled:
            ix=obj.scl[np.isfinite(row[obj.scl])];row[ix]=rng.permutation(row[ix])
        variants['SCL_order_shuffled']=shuffled
        joint=x.copy()
        for j in range(joint.shape[1]):joint[:,j]=rng.permutation(joint[:,j])
        if np.all(np.isfinite(joint).sum(1)>=2):variants['independent_contact_columns']=joint
        for name,z in variants.items():
            sc=obj.score(z,device=device)
            results.append(dict(repetition=rep,control=name,J=sc['J'],N=sc['N'],mode_counts=sc['mode_counts'],components=sc['components'],
                participation_marginals_preserved=bool(np.array_equal(np.isfinite(x).sum(0),np.isfinite(z).sum(0)))))
    def median(name,i=None):
        vals=[r['J'] if i is None else r['components'][i] for r in results if r['control']==name]
        return float(np.median(vals)) if vals else None
    cpu=obj.features(ev.fit[:32],device='cpu');gpu=obj.features(ev.fit[:32],device=device)
    parity=max(float(abs(cpu[g]-gpu[g]).max()) for g in GROUPS)
    checks=dict(collapse_worse=median('mode_collapsed')>median('patient_resample'),
        rod_shift_worsens_timing=median('SCL_plus_35ms',1)>median('patient_resample',1),
        local_shuffle_worsens_timing=median('SCL_order_shuffled',1)>median('patient_resample',1),
        break_joint_participation_worse=median('independent_contact_columns',2)>median('patient_resample',2),
        cpu_gpu_features_match=parity<1e-9)
    result=dict(status='PASS' if all(checks.values()) else 'NEEDS_OBJECTIVE_REVIEW',checks=checks,median_by_control={n:{'J':median(n),'components':[median(n,i) for i in range(3)]} for n in sorted({r['control'] for r in results})},
        results=results,cpu_gpu_max_error=parity,note='These targeted controls do not establish full patient propagation recovery.')
    rt.write(A/'objective_sanity.json',result);return result

def calibrate(device='cuda:0'):
    A.mkdir(exist_ok=True,parents=True)
    with (A/'calibration.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (A/'objective_frozen.json').exists():return
        ev,names,parent=patient();calids=chronology(ev)
        obj=ThreeObservableObjective(names,ev.km,np.bincount(ev.fit_labels,minlength=2)/len(ev.fit))
        print('FITTING_PATIENT_MAPS',flush=True);obj.fit_maps(ev.fit)
        print('FITTING_PATIENT_TARGETS',flush=True);obj.fit_targets(ev.fit,ev.fit_labels,device)
        print('CALIBRATING_POSITIVE_SCALES',flush=True);cal=obj.calibrate(ev.patient[calids],ev.blocks[calids],device)
        cal['row_order']='chronological within verified source blocks';cal['parent_indices']=calids.tolist();rt.write(A/'positive_scale_calibration.json',cal)
        result=sanity(obj,ev,device)
        dump_pickle(A/'training_objective.pkl',obj)
        rt.write(A/'patient_reference.json',dict(names=names,fit_N=len(ev.fit),cal_N=len(calids),proportions=dict(TA=float(obj.proportions[1]),TB=float(obj.proportions[0])),
            time_scales_ms=obj.time_scales,normalizers=obj.scales,maps={k:dict(bandwidth=v['bandwidth'],linear_scale=v['linear_scale'],degenerate=v['degenerate']) for k,v in obj.maps.items()},
            source=parent['sources']['evaluator'],mode_labels_used_for_training=True))
        rt.write(A/'objective_frozen.json',dict(status='FROZEN' if result['status']=='PASS' else 'FROZEN_DIAGNOSTICS_FAILED',
            adaptive_dispatch_allowed=result['status']=='PASS',objective_sha256=rt.sha(A/'training_objective.pkl'),
            source_sha256=rt.sha(ROOT/'src/topic4_three_observable_objective.py'),scales=obj.scales,device=device,time=time.time()))
        print(json.dumps(result['checks']),flush=True)

def load_objective():
    meta=rt.read(A/'objective_frozen.json')
    assert rt.sha(A/'training_objective.pkl')==meta['objective_sha256']
    assert rt.sha(ROOT/'src/topic4_three_observable_objective.py')==meta['source_sha256']
    with (A/'training_objective.pkl').open('rb') as f:return pickle.load(f)

def load_small(path):
    r=rt.read(path)
    with np.load(path.with_suffix('.npz')) as z:
        names=z['contact_names'].tolist();times=z['centroid_ms'];stored=z['event_mode'];rawids=z['primary_event_indices']
        ids=np.array([i for i in rawids if r['events'][i]['window_ms'][0]>=1500 and r['events'][i]['window_ms'][1]<=r['actual_duration_ms']],int)
    return r,times,stored,ids,names

def raw_summary(t,names):
    from scripts.analyze_topic4_core_connectivity_search import measures
    ev,_,_=patient();obj=load_objective();labs=obj.labels(t) if len(t) else np.array([],int);out={}
    for name,k in [('ALL',None),('TA',1),('TB',0)]:
        x=t if k is None else t[labs==k];ref=ev.fit if k is None else ev.fit[ev.fit_labels==k]
        if not len(x):out[name]=dict(n=0);continue
        out[name]=measures(x,ref,np.asarray(names))
    return out

def score_path(path,device='cuda:0'):
    stage=path.parents[4].name;cid=path.parents[2].name;key=hashlib.sha256(str(path).encode()).hexdigest()[:20];dest=OUT/'scores'/f'{key}.json'
    if dest.exists():return rt.read(dest)
    obj=load_objective();r,t,labels,ids,names=load_small(path)
    assert names==obj.names
    assert rt.sha(path.with_suffix('.npz'))==r['arrays_sha256']
    applied=rt.read(path.parents[1]/'applied_physics.json');c=rt.read(OUT/'candidates'/f'{cid}.json')
    assert run.physics(c)==run.physics(applied['candidate'])
    audit_applied(applied,c)
    assert applied['threshold']['n_raised']==0 and all(applied['input'][k]=='off' for k in ['ZM','slow_I_state','spatial_ou','kick'])
    assert np.array_equal(obj.labels(t[ids]),labels[ids]) if len(ids) else True
    sc=obj.score(t[ids],labels[ids],device)
    if r['physical_status']=='RUNAWAY':sc.update(status='PHYSICAL_RUNAWAY',J=None,components=None)
    intervals=[]
    for lo in np.arange(1500,r['actual_duration_ms'],6000):
        selected=[i for i in ids if lo<=r['events'][i]['event_time_ms']<lo+6000]
        tm=np.array([r['events'][i]['event_time_ms'] for i in selected])
        intervals.append(dict(start_ms=float(lo),end_ms=min(float(lo+6000),r['actual_duration_ms']),N=len(selected),
            TA=int((labels[selected]==1).sum()),TB=int((labels[selected]==0).sum()),median_interval_ms=float(np.median(np.diff(tm))) if len(tm)>1 else None))
    sc.update(candidate=cid,stage=stage,topology=r['job']['topology_seed'],noise=r['job']['dynamics_seed'],source=str(path),source_sha256=rt.sha(path),
        objective_sha256=rt.read(A/'objective_frozen.json')['objective_sha256'],physical_status=r['physical_status'],vector=run.vector(c).tolist(),
        raw=raw_summary(t[ids],names),six_second_segments=intervals,static_identity=applied['identity'],updated_unix=time.time())
    rt.write(dest,sc);print(json.dumps(dict(candidate=cid,noise=sc['noise'],J=sc['J'],N=sc['N'],status=sc['status'])),flush=True);return sc

def audit_applied(applied,c):
    th=applied['threshold'];g=applied['graph'];inp=applied['input'];par=c['parameters']
    np.testing.assert_allclose(th['centers_mm'],c['centers_mm'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(th['radii_mm'],c['radii_mm'],rtol=0,atol=1e-12)
    assert th['n_raised']==0 and not any(th['boundary_clipped'])
    assert inp['mode']=='core_poisson_outside_expected' and inp['n_stochastic']==th['n_members']
    assert inp['core_mean_rate_scale']==c['core_mean_rate_scale'] and inp['core_ou_correlation']==c['core_ou_correlation']
    assert inp['OU_tau_ms']==150 and inp['OU_sigma_n']==3.3
    assert all(inp[k]=='off' for k in ['ZM','slow_I_state','spatial_ou','kick'])
    assert g['kernel']['angle_offset_deg']==par['EE_angle_offset_deg']
    assert g['kernel']['perp_scale']==par['EE_kernel_perp_scale'] and g['kernel']['parallel_scale']==par['EE_kernel_parallel_scale']
    for key,block in g['stage_audits']['weights']['blocks'].items():
        assert block['factor']==par[key]
        np.testing.assert_allclose(block['weight_after'],block['factor']*block['weight_before'],rtol=1e-10,atol=1e-7)
    assert g['stage_audits']['kernel']['indegree_preserved']

def records():return [rt.read(p) for p in sorted((OUT/'scores').glob('*.json'))]

def report():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.topic4_pdf_font_guard import install
    install();plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':3})
    F.mkdir(exist_ok=True,parents=True);rr=records();ev,names,_=patient();obj=load_objective();order=[names.index(n) for n in DISPLAY]
    descriptions=['整体rank分布误差','杆内顺序与杆间时差误差','参与结构误差']
    ids=list(dict.fromkeys(r['candidate'] for r in rr));lookup={(r['candidate'],r['topology'],r['noise']):r for r in rr}
    plan=rt.read(OUT/'plan.json');refid=plan['reference_id']
    byid={c:np.mean([r['J'] for r in rr if r['candidate']==c and r['stage']!='confirmation' and r['J'] is not None]) for c in ids if any(r['candidate']==c and r['stage']!='confirmation' and r['J'] is not None for r in rr)}
    top=[refid]+[c for c in sorted(byid,key=byid.get) if c!=refid][:2]
    fig,axes=plt.subplots(2,1+len(top),figsize=(4*(1+len(top)),9),sharey=True,squeeze=False)
    for row,(mode,k) in enumerate([('TA',1),('TB',0)]):
        for col,cid in enumerate([None]+top):
            if cid is None:x=ev.fit[ev.fit_labels==k]
            else:
                group=[]
                for r in rr:
                    if r['candidate']!=cid or r['topology']!=2511 or r['stage']=='confirmation':continue
                    _,t,lab,ix,_=load_small(Path(r['source']));group.append(t[ix[lab[ix]==k]])
                x=np.concatenate(group) if group else np.empty((0,len(names)))
            ax=axes[row,col];ax.set(yticks=np.arange(15),yticklabels=DISPLAY,ylim=(14.5,-.5),xlim=(-.05,1.05),xlabel='事件内归一化rank')
            if len(x):
                ranks,mask=normalized_ranks(x);ranks=np.where(mask,ranks,np.nan)[:,order]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning);mu=np.nanmean(ranks,0);q=np.nanquantile(ranks,[.05,.95],axis=0)
                ax.fill_betweenx(np.arange(15),q[0],q[1],color='#387fb2',alpha=.2)
                for sel in [slice(0,4),slice(4,15)]:ax.plot(mu[sel],np.arange(15)[sel],'o-',ms=4,color='#21649c')
            ax.axhline(3.5,color='gray',ls='--');ax.grid(alpha=.15)
            ax.set_title(f'{"患者FIT" if cid is None else plot_label(cid)}\n{mode}，事件 n={len(x)}',fontsize=9)
    fig.suptitle('患者TA/TB模板与模型：均值及5–95%范围；分杆Y轴固定\n模型图汇总同拓扑两条训练噪声，优化仍按运行等权',fontsize=12);fig.tight_layout(rect=(0,.18,1,.94));parameter_table(fig,top)
    for ext in ['png','pdf']:fig.savefig(F/f'patient_model_rank_templates.{ext}',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for j,ax in enumerate(axes):
        for i,cid in enumerate(ids):
            for r in rr:
                if r['candidate']==cid and r['components'] is not None:ax.scatter(i,r['components'][j],marker='o' if r['noise']==847401 else '^',c='#256fa8' if r['topology']==2511 else '#b77732',s=22)
        ax.set(xticks=range(len(ids)),xticklabels=ids,ylabel=descriptions[j]+' ↓');ax.tick_params(axis='x',rotation=85,labelsize=6);ax.grid(alpha=.15)
    fig.suptitle('三组观测分别评分；点为一次网络/噪声运行；不连接多参数混合候选');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(F/f'three_observable_scores.{ext}',dpi=150)
    plt.close(fig)
    # Initial single-coordinate probes support genuine paired response lines.
    fig,axes=plt.subplots(3,6,figsize=(21,11),squeeze=False)
    for axis in range(6):
        cids=[refid]+[m['candidate'] for m in plan['initial_meta'] if m['axis']==axis]
        cids=sorted(set(cids),key=lambda c:run.vector(rt.read(OUT/'candidates'/f'{c}.json'))[axis])
        for j in range(3):
            ax=axes[j,axis]
            for noise,color in zip(plan['seeds'],['#2576ad','#d77821']):
                x=[run.vector(rt.read(OUT/'candidates'/f'{c}.json'))[axis] for c in cids];y=[]
                for c in cids:
                    v=lookup.get((c,2511,noise));y.append(np.nan if v is None or v['components'] is None else v['components'][j])
                ax.plot(x,y,'o-',color=color,label=f'噪声{noise}')
            ax.set(xlabel=['左核X (mm)','左核Y (mm)','右核X (mm)','右核Y (mm)','向外EE倍率','轴偏移 (°)'][axis],ylabel=descriptions[j] if axis==0 else '')
            ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=7);fig.suptitle('共用拓扑/噪声种子：初始单参数配对响应；角度改变会重建EE边和时延\n患者参考由独立模板图与原量表提供；纵轴为患者尺度化误差，缺失点不填补');fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(F/f'initial_paired_parameter_response.{ext}',dpi=150)
    plt.close(fig)
    (F/'README.md').write_text('# 三组观测优化：实时图\n\n'+''.join(f'### {name}.png\n{description}\n**关注点**：区分模板、事件散布和逐运行证据；低分不等于原生传播已恢复。\n\n' for name,description in [
        ('patient_model_rank_templates','患者TA/TB均值模板和5–95%范围放左列，与参考及当前训练候选并列；固定SCL/ICL行序，未参与行保留。'),
        ('three_observable_scores','三个信息组分别展示所有已评分运行；蓝色为训练拓扑，棕色为确认拓扑，圆点为训练噪声847401，其余为三角。不将不同物理条件串成单参数响应。'),
        ('initial_paired_parameter_response','初始六个标量的正负扰动；共用拓扑/噪声种子，仅改变横轴参数，但角度改变会按模型规则重建EE边和时延，同一条件内两噪声的连接图才完全固定；尚未完成点保留缺失。')]))
    flat=[dict(candidate=r['candidate'],stage=r['stage'],topology=r['topology'],noise=r['noise'],N=r['N'],TA=r['mode_counts']['TA'],TB=r['mode_counts']['TB'],J=r['J'],status=r['status'],**{g:None if r['components'] is None else r['components'][i] for i,g in enumerate(GROUPS)}) for r in rr]
    if flat:
        with (A/'scores.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    rt.write(A/'status.json',dict(scored_units=len(rr),scorable_units=sum(r['J'] is not None for r in rr),patient_reference=rt.read(A/'patient_reference.json'),updated_unix=time.time()))
    if rr:
        from scripts.report_topic4_three_observable_raw import report as raw_report
        raw_report(sys.modules[__name__],rr,top)

def observe(device):
    while not (A/'objective_frozen.json').exists():time.sleep(10)
    last=-1
    while True:
        paths=sorted(OUT.glob('*/units/*/*/workers/trajectory.json'))
        for path in paths:
            if rt.read(path).get('status')=='COMPLETE':score_path(path,device)
        now=len(records())
        if now!=last:report();last=now
        if (OUT/'optimization_complete.json').exists():report();return
        time.sleep(15)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['calibrate','observe','report','score']);p.add_argument('--device',default='cuda:0');p.add_argument('--path');a=p.parse_args()
    if a.action=='calibrate':calibrate(a.device)
    elif a.action=='report':report()
    elif a.action=='score':score_path(Path(a.path),a.device)
    else:
        A.mkdir(exist_ok=True)
        with (A/'observer.lock').open('w') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);observe(a.device)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        kind='calibration' if 'calibrate' in sys.argv else 'observer'
        rt.write(A/f'{kind}_failure.json',dict(error=repr(exc),time=time.time()))
        raise
