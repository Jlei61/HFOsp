#!/usr/bin/env python3
"""Give native Z time to recover by slowing only low-activity K decay.

Fixed state-dependent law from t=0. No Z/M reset or observer-driven switch.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import copy
import fcntl
import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import psutil
import run_topic4_rhythm_preserving_feedback as parent

SOURCE=parent.OUT
OUT=Path('/data/hfosp/topic4_sef_hfo/fig5_recovery_window_20260917')
SEED=parent.SEED
SEEDS=[SEED,SEED+1,SEED+2]
OFF_TIMES=[.5,1.,2.]
REGIONS=parent.REGIONS
base=parent.base
audit=parent.audit
fixed=parent.fixed
budget=parent.budget
write=parent.write
load=audit.old.load
rate_data=parent.rate_data
ParentAnalyze=parent.analyze
ParentPrefix=parent.prefix_check


def assert_same_state(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_same_state(a[key],b[key])
    elif isinstance(a,np.ndarray):assert a.dtype==b.dtype and np.array_equal(a,b)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_same_state(x,y)
    else:assert a==b


class RecoveryWindowSlow(parent.RhythmSlow):
    off_tau_ms=500.
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.window_records=[]

    def apply_currents(self,*args,**kwargs):
        value=super().apply_currents(*args,**kwargs)
        if self._step_index%200==0:
            groups=[np.arange(self.NE),*self.region_groups()]
            dist=np.array([[self.z[ix].mean(),*np.quantile(self.z[ix],[.1,.5,.9]),
                            np.mean(self.z[ix]<.7),np.mean(self.z[ix]<.8)] for ix in groups])
            self.window_records.append((self._step_index*.1,
                self.sahp_tau_ms if self.gate>0 else self.off_tau_ms,dist))
        return value

    def step(self,spk,labels,dt):
        # The gate was calculated from the causal rate before this step's spikes.
        # K gain remains normalized by the original high-activity tau=0.5s.
        high_tau=self.sahp_tau_ms
        if self.gate==0:self.sahp_tau_ms=self.off_tau_ms
        try:super().step(spk,labels,dt)
        finally:self.sahp_tau_ms=high_tau


def configure():
    parent.OUT=OUT;parent.prepare=prepare;parent.reference=reference
    parent.previous.OUT=OUT;budget.OUT=OUT


def reference(seed):
    path=OUT/'references'/f'native_s{seed}.npz'
    if not path.exists():raise FileNotFoundError(path)
    return path


def make_job(off,seed,index):
    j=parent.make_job(15.,40.,.5,seed,index,horizon=120.)
    j.update(name=f'off{off:g}_s{seed}',off_tau_s=off,on_tau_s=.5,
        stage='new_condition' if off!=.5 else 'paired_control',
        mechanism='Fixed state-dependent K decay: tau=0.5s at R>200Hz; tau=off_tau_s otherwise. Spike increment remains0.16*q.',
        full_state_snapshots=True)
    return j


def copy_control(job):
    src=SOURCE/'runs'/f'G15_K40_tau0.5_s{job["seed"]}'
    dest=OUT/'runs'/job['name'];dest.mkdir(parents=True,exist_ok=True)
    old_result=base.read(src/'result.json')
    assert old_result['identity']==base.read(SOURCE/'protocol.json')['identity']
    assert old_result['job']['sahp_tau_s']==.5 and old_result['job']['K500']==40 and old_result['job']['G500']==15
    # Committed observation chunks are immutable; checkpoint and metadata get independent copies.
    for name in ['chunks','mechanism_chunks','intrinsic_adaptation_chunks','actual_current_chunks',
                 'regional_chunks','z_budget_chunks','feedback_chunks']:
        if (src/name).exists():shutil.copytree(src/name,dest/name,copy_function=os.link)
    for name in ['rhythm_preservation.json','feedback_activation.json']:
        if (src/name).exists():shutil.copy2(src/name,dest/name)
    applied=base.read(src/'applied_configuration.json');applied['job']=job
    applied['recovery_window_tau_off_s']=job['off_tau_s'];write(dest/'applied_configuration.json',applied)
    with (src/'checkpoint.pkl').open('rb') as h:saved=pickle.load(h)
    assert saved['job']==old_result['job']
    before=copy.deepcopy(saved['engine'])
    saved['job']=job
    assert saved['engine']['slow']['kind']=='RhythmSlow'
    saved['engine']['slow']['kind']='RecoveryWindowSlow'
    saved['tracker']['stop_reason']=None
    base.save_pickle(dest/'checkpoint.pkl',saved)
    before['slow']['kind']='RecoveryWindowSlow'
    assert_same_state(saved['engine'],before)
    write(dest/'reuse_provenance.json',dict(source=str(src),source_result=old_result,
        inherited_end_s=old_result['end_s'],source_checkpoint_sha256=base.sha(src/'checkpoint.pkl'),
        physical_state_unchanged=True,slow_kind_change='RhythmSlow -> RecoveryWindowSlow; off_tau=on_tau=0.5s verified by QA',
        new_independent_realization=False))
    if old_result['end_s']>=job['horizon_s']:
        result=copy.deepcopy(old_result);result['job']=job;result['reused_from']=str(src)
        write(dest/'result.json',result);write(dest/'progress.json',result)


def prepare():
    if (OUT/'protocol.json').exists():return base.read(OUT/'protocol.json')
    OUT.mkdir(parents=True,exist_ok=True)
    p0=base.read(SOURCE/'protocol.json')
    for path,h in p0['source_hashes'].items():assert base.sha(path)==h,path
    # Dispatch both new kinetics first, then the two short control continuations.
    settings=[(1.,SEED),(2.,SEED),(.5,SEED+1),(.5,SEED+2),
              (1.,SEED+1),(2.,SEED+1),(1.,SEED+2),(2.,SEED+2),(.5,SEED)]
    jobs=[make_job(off,seed,i) for i,(off,seed) in enumerate(settings)]
    shutil.copy2(SOURCE/'geometry.npz',OUT/'geometry.npz')
    refs=OUT/'references';refs.mkdir(exist_ok=True)
    for seed in SEEDS:
        if (SOURCE/'references'/f'native_s{seed}.npz').exists():
            shutil.copy2(SOURCE/'references'/f'native_s{seed}.npz',refs/f'native_s{seed}.npz')
            shutil.copy2(SOURCE/'references'/f'native_s{seed}.json',refs/f'native_s{seed}.json')
        else:
            folder=SOURCE/'runs'/f'G0_K0_tau1_s{seed}'
            d=load(folder,keys=['spikes_1ms','regions_1ms','raster','slow_time_ms','Z','M','inputs','field_5ms'])
            assert len(d['spikes_1ms'])==8000
            np.savez_compressed(refs/f'native_s{seed}.npz',**d)
            write(refs/f'native_s{seed}.json',dict(source=str(folder),duration_s=8.,identity=p0['identity']))
    for j in jobs:
        write(OUT/'jobs'/f'{j["name"]}.json',j)
        if j['off_tau_s']==.5:copy_control(j)
    p={k:copy.deepcopy(p0[k]) for k in ['identity','source_hashes','baseline','reference_current_scale']}
    p.update(created_epoch=time.time(),deadline_epoch=time.time()+12*3600,
        source_round=str(SOURCE),initial_jobs=jobs,branch_jobs=[],max_workers=4,
        min_available_memory_GiB=70.,disk_reserve_GiB=50.,wall_budget_hours=12.,
        producer_sha256=base.sha(parent.carrier.__file__),wrapper_sha256=base.sha(fixed.__file__),
        exploration_sha256=base.sha(__file__),protected_global_authorized=False,
        new_fixed_equation=dict(q='clip((R_G-200)/300,0,1); existing15ms causal global E rate',
            G_raw='15*q',G_applied='Z_i*G_raw',J='Original II_i + (18-EG)*G_raw for E',
            K_increment='Each E spike adds0.16*q; K500=40 at q=1,500Hz with tau_on=0.5s',
            K_decay='exp(-dt/tau): tau=0.5s when the pre-step R_G>200Hz, otherwise tau_off in[0.5,1,2]s'),
        question='Does longer low-activity K retention let native Z recover enough for recurrent interictal population events?',
        authorization='User2026-09-17: start next round; Z has not sufficiently recovered. Prioritize autonomous recovery-window test.',
        experiment='Three off times x three paired seeds, common120s horizon.6new cold starts,1complete old control reused,2controls continued from104/102s. No automatic expansion.',
        causal_branches='No artificial Z/M replacement in this autonomous batch. Retain full snapshots for later diagnostics if the direct test remains ambiguous.',
        preserved='Native local inhibition, native Z/M equations and parameters, graph, threshold field, noise, high-rate gate and K increment.',
        prefix_gate='Every cold start must reproduce first8s of same-seed native reference bitwise; compare longer prefix before first activation where reference exists.',
        acceptance='Preserved native interictal events -> high entry -> autonomous low state and native Z recovery -> native-like brief events. Quiet alone does not pass.',
        return_rules=dict(legacy='>=5brief events spanning>=2s and brief fraction>=0.8',
            sustained='>=10brief events spanning>=5s; fraction>=0.8; median duration/IEI/global peak within0.5-2x same-seed native reference. Spatial/core recruitment reported and human reviewed.'),
        observation='Every20ms native-Z budget and regional Z distribution; fixed native raster and field; snapshot every10s and first confirmed entry/exit states.',
        time_boundary='12h maximum to allow matched120s observations under shared GPU load. Checkpoint stop may lag deadline by one2s simulation block; any truncation is explicit.',
        human_review='PENDING',initial_rhythm_visual_reference='User accepted repaired initial rhythm in preceding round; new complete trajectories remain candidates.')
    for f in [__file__,parent.__file__,budget.__file__,parent.previous.__file__,fixed.__file__,parent.carrier.__file__,audit.__file__]:
        p['source_hashes'][str(Path(f).resolve())]=base.sha(f)
    impl=OUT/'implementation';impl.mkdir(exist_ok=True)
    for f in [__file__,parent.__file__]:shutil.copy2(f,impl/Path(f).name)
    write(OUT/'protocol.json',p)
    (OUT/'design.md').write_text('''# Z恢复窗口实验

本轮检验：保留原Fig5的起始间期群体活动，延长高活动之后的反馈尾部，能否让原生Z恢复并重新出现同类间期事件。统计单位为条件×噪声轨迹；三个种子按条件配对，固定同一拓扑，不宣称新拓扑泛化。

只改K低活动消退：已有15ms全E率R_G>200Hz时，K仍按0.5s消退，每E spike仍增加0.16*q，q=clip((R_G-200)/300,0,1)。R_G<=200Hz时，消退时间取0.5、1、2s。所有规律从t=0固定；初始K=0，低率间期段不积累K。此为状态依赖消退的新动力学假设，不是Liou原式复现，也不是检测到退出后切参数。高率指神经元群体的模型内部状态，不是临床分类。

G500=15、K500=40、高活动tauK=0.5s不变；原Z/M、局部抑制、阈值底物、连接及噪声不变。G仍由原Z门控，资源负荷沿用上一轮；不启用protected通路。Z没有人工恢复或重置。

三种消退时间×三个配对种子，统一120s：新增六条；0.5s主种子完整120s结果复用，另两条从原104/102s完整状态续到120s，旧文件保留。当前优先直接做自主恢复窗口检验，原提案中的人工Z/M置换不作为本批启动前提；每10s和首次进入/退出的完整状态另存，必要时再诊断。主批最多4并发，12h墙钟上限；不自动扩展强度、全局慢变量或新方程。

运行前核对0.5s新旧更新逐位一致、从既有完整状态恢复一致，以及低率消退、相同高率增量和原Z收支。每条冷启动首先核对8s原生spikes、区域计数、固定raster、空间场、Z/M和输入；提前改变原间期段按失败停止后续派发。

原短事件标准保留：20–200ms、全E峰>=20Hz、两核与全E同时<5Hz达20ms作为事件边界。原最低返回筛查为>=5事件跨>=2s；本轮额外要求>=10事件跨>=5s、短事件占比>=80%，持续时间/间隔/全E峰中位数在同种子原生参考0.5–2倍内，才标为持续返回候选。并列报告两核招募、空间覆盖和局部Z分布；不以Z达到某个均值自动认定回到间期。

若长尾只有延长静默、随后直接高活动，判为无间期返回；若初始事件消失，先判保留门失败；若终止后反复短事件返回，才进入最新版Fig5逐轨迹目视。所有配对种子报告，单种子成功不冒称稳健。A raster和两核放大、B原Z/M、C真实空间状态、D meanZ/IE/E率、E本轮条件×种子；F患者能量仍待密集核验。
''')
    return p


def qa():
    p=prepare();configure()
    cfg=base.old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000.,I_th_EI=95.19851312666987,tau_adp=1000.,eta_m=.0005)
    def make(cls,off=.5,n=12,ne=10):
        cls.C_R=0.;cls.feedback_form='conductance';cls.sahp_gain=16.;cls.sahp_tau_ms=500.;cls.off_tau_ms=off*1000
        obj=cls(n,18,cfg,NE=ne,mode='native',gamma=0.,global_gain=15.,global_resource='native_z',phi_jump=0.)
        obj.global_reversal=parent.EG;obj.voltage=np.full(n,5.)
        if n==12:obj._groups=[np.arange(3),np.arange(3,6),np.arange(6,10)]
        return obj
    rng=np.random.default_rng(17091701);a=make(parent.RhythmSlow);b=make(RecoveryWindowSlow)
    for i in range(1200):
        ie,ii=rng.uniform(0,1800,(2,12));sp=rng.random(12)<.04
        a.r_global=b.r_global=[0.,140.,200.,201.,350.,500.][i%6]
        assert np.array_equal(a.apply_currents(ie,ii),b.apply_currents(ie,ii))
        a.step(sp,None,.1);b.step(sp,None,.1)
        for key in ['z','m','g_k','g_global']:assert np.array_equal(getattr(a,key),getattr(b,key)),key
        assert a.r_global==b.r_global
    checks=[]
    for off in OFF_TIMES:
        o=make(RecoveryWindowSlow,off)
        for rate in [0.,140.,200.,201.,350.,500.]:
            o.r_global=rate;o.g_k[:]=2.
            o.apply_currents(np.ones(12)*100,np.ones(12)*10)
            q=o.gate;tau=500. if rate>200 else off*1000
            o.step(np.ones(12,bool),None,.1)
            assert np.array_equal(o.g_k,np.full(10,2*np.exp(-.1/tau)+.16*q))
        o.g_k[:]=0.;o.r_global=140.;o.apply_currents(np.ones(12),np.ones(12))
        o.step(np.ones(12,bool),None,.1);assert not o.uses_shunt() and np.all(o.g_k==0)
        checks.append(dict(off_tau_s=off,decay_and_gain=True,initial_low_rate_zero=True))
    # Restore the real high-state checkpoint into old and new objects and drive
    # identical currents/spikes. The integrator, delay buffers and RNGs are unchanged.
    with (SOURCE/'runs/G15_K40_tau0.5_s9108402/checkpoint.pkl').open('rb') as h:saved=pickle.load(h)
    a=make(parent.RhythmSlow,n=40000,ne=32000);b=make(RecoveryWindowSlow,n=40000,ne=32000)
    import checkpoint
    for obj in [a,b]:
        state=copy.deepcopy(saved['engine']);state['slow']['kind']=type(obj).__name__
        checkpoint.restore_slow(state,obj)
        obj.r_global=state['termination_mechanism']['r_global']
        obj.g_k[:]=state['termination_mechanism']['sahp_g'];obj.phi[:]=state['termination_mechanism']['phi']
    for i in range(400):
        ie=saved['engine']['I_E'];ii=saved['engine']['I_I'];sp=rng.random(40000)<.03
        assert np.array_equal(a.apply_currents(ie,ii),b.apply_currents(ie,ii))
        a.step(sp,None,.1);b.step(sp,None,.1)
        for key in ['z','m','g_k']:assert np.array_equal(getattr(a,key),getattr(b,key))
        assert a.r_global==b.r_global
    for path,h in p['source_hashes'].items():assert base.sha(path)==h,path
    write(OUT/'mechanism_qa.json',dict(status='PASS',same_tau_bitwise_1200_steps=True,
        real_checkpoint_slow_state_parity_400_steps=True,checks=checks,
        no_added_physical_state=True,Z_budget_balance_error=b.max_balance_error))
    print('PASS: paired law, low-rate decay, high-rate normalization, real checkpoint slow-state parity',flush=True)


def event_features(events,rates,field=None):
    if not events:return dict(duration_ms=None,interval_ms=None,peak_Hz=None,core_peak_Hz=None,active_cell_fraction=None)
    peaks=np.array([rates[round(e['start_s']*100):round(e['end_s']*100),:3].max(0) for e in events])
    coverage=[]
    if field is not None:
        for e in events:
            x=field[round(e['start_s']*200):round(e['end_s']*200)]
            coverage.append(float(np.mean(x.sum(0)>0)))
    return dict(duration_ms=float(np.median([e['duration_s']*1000 for e in events])),
        interval_ms=float(np.median(np.diff([e['start_s'] for e in events]))*1000) if len(events)>1 else None,
        peak_Hz=float(np.median(peaks[:,0])),core_peak_Hz=np.median(peaks[:,1:3],axis=0),
        active_cell_fraction=float(np.median(coverage)) if coverage else None)


def analyze(name):
    configure();row=ParentAnalyze(name)
    if row is None:return None
    folder=OUT/'runs'/name;rr=parent.rate_data(folder);end=len(rr)*.01
    pp=row['primary'];nr=row['native_rhythm'];events=pp['events']
    zd=load(folder,'z_budget_chunks',keys=['time_ms','values'])
    field=load(folder,keys=['field_5ms'])['field_5ms']
    with np.load(reference(row['job']['seed'])) as ref:
        with np.load(OUT/'geometry.npz') as geo:counts=np.r_[32000,geo['region_counts'][:3]]
        rref=np.column_stack([ref['spikes_1ms'][:8000,0],ref['regions_1ms'][:8000,:3]]).reshape(800,10,4).sum(1)/counts/.01
        eref=audit.interval_events(parent.strict_events(rref,8.),.5,8.)['brief_events']
        baseline=event_features(eref,rref,ref['field_5ms'])
    episodes=[]
    for ex in pp['low_activity_exits']:
        lo=ex['confirmation_s'];hi=next((e['onset_s'] for e in pp['entries'] if e['onset_s']>lo),end)
        part=audit.interval_events(events,lo,hi);features=event_features(part['brief_events'],rr,field)
        ratios={k:features[k]/baseline[k] if features[k] is not None and baseline[k] else None for k in ['duration_ms','interval_ms','peak_Hz']}
        matched=all(v is not None and .5<=v<=2 for v in ratios.values())
        zs=zd['time_ms']/1000;v=zd['values']
        ia=np.argmin(abs(zs-ex['start_s']));ib=np.argmin(abs(zs-hi))
        zrise=v[ib,1:3,1]-v[ia,1:3,1]
        shared_rise=float(np.max(np.min(v[ia:ib+1,1:3,1]-v[ia,1:3,1],axis=1)))
        sustained=bool(audit.qualifies(part,minimum_n=10,minimum_span=5.) and matched and shared_rise>=.02)
        episodes.append(dict(exit=ex,interval=part,features=features,reference_ratios=ratios,
            Z_at_exit_window_start=v[ia,:,1],Z_at_window_end=v[ib,:,1],core_Z_rise=zrise,maximum_shared_core_Z_rise=shared_rise,
            legacy_return=bool(audit.qualifies(part) and matched),sustained_return=sustained))
    quiet=rr[:,:3].max(1)<5
    gaps=[dict(start_s=a*.01,end_s=b*.01,duration_s=(b-a)*.01) for a,b in audit.old.spans(quiet)
          if (b-a)*.01>=.5 and pp['entries'] and a*.01>pp['entries'][0]['confirmation_s']]
    success=bool(nr['preservation']['status']=='PASS' and nr['strict_pre_pass'] and any(e['sustained_return'] for e in episodes))
    row['recovery_window']=dict(off_tau_s=row['job']['off_tau_s'],on_tau_s=.5,reference_features=baseline,
        episodes=episodes,joint_low_activity_intervals=gaps,sustained_return_screen=success,
        spatial_acceptance='PENDING_NATIVE_REVIEW')
    row['full_sequence_screen']=success
    row['classification']=('SUSTAINED_NATIVE_RETURN_CANDIDATE' if success else 'EXIT_WITHOUT_SUSTAINED_NATIVE_RETURN' if episodes else
                           'HIGH_WITHOUT_EXIT' if pp['entries'] else 'NO_HIGH_OBSERVED')
    if nr['preservation']['status']!='PASS':row['classification']='PRESERVATION_'+nr['preservation']['status']
    write(OUT/'analysis'/f'{name}.json',row)
    write(folder/'live_status.json',dict(observed_s=end,classification=row['classification'],
        off_tau_s=row['job']['off_tau_s'],preservation=nr['preservation']['status'],
        entries=len(pp['entries']),exits=len(episodes),pre_brief=nr['strict_pre']['brief_count'],
        max_post_brief=max([e['interval']['brief_count'] for e in episodes],default=0),updated_epoch=time.time()))
    return row


def flush_window(folder,step):
    parent.flush(folder,step);obj=budget.RecoveryBudgetSlow.instance
    if obj and obj.window_records:
        rec=obj.window_records;dest=folder/'window_chunks';dest.mkdir(exist_ok=True)
        path=dest/f'{round(rec[0][0]*10):010d}_{step:010d}.npz';tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,time_ms=[r[0] for r in rec],effective_tau_ms=[r[1] for r in rec],
            Z_distribution=np.array([r[2] for r in rec]),keys=['mean','q10','q50','q90','fraction_below0p7','fraction_below0p8'])
        tmp.replace(path);obj.window_records.clear()


def worker(name):
    p=prepare();configure();assert base.sha(__file__)==p['exploration_sha256']
    assert base.read(OUT/'mechanism_qa.json')['status']=='PASS'
    job=base.read(OUT/'jobs'/f'{name}.json');folder=OUT/'runs'/name
    start_s=0.
    if (folder/'checkpoint.pkl').exists():
        with (folder/'checkpoint.pkl').open('rb') as h:start_s=pickle.load(h)['engine']['step']*.0001
    RecoveryWindowSlow.C_R=0.;RecoveryWindowSlow.feedback_form='conductance'
    RecoveryWindowSlow.sahp_gain=16.;RecoveryWindowSlow.sahp_tau_ms=500.;RecoveryWindowSlow.off_tau_ms=job['off_tau_s']*1000
    RecoveryWindowSlow.record_regions=True;RecoveryWindowSlow.z_gate_off=False;RecoveryWindowSlow.k_freeze=False
    old=(fixed.OUT,fixed.prepare,fixed.TerminationSlow,fixed.observation_sink);sink0=fixed.observation_sink
    def factory(sink,j,deadline):
        full=sink0(sink,j,deadline)
        def observe(step,state):
            try:return full(step,state)
            finally:
                config=base.read(folder/'applied_configuration.json')
                config.update(native_local_GABA_unchanged=True,native_Z_function_preserved=True,
                    native_Z_input='Original local GABA + (18-EG)*15*q; unchanged from parent round',
                    native_GABA_and_Z_unchanged=False,recovery_window_tau_off_s=job['off_tau_s'],
                    new_fixed_feedback_law=p['new_fixed_equation'])
                write(folder/'applied_configuration.json',config)
                flush_window(folder,step);ParentPrefix(name);row=analyze(name)
                # Retain actual complete states; no reconstruction from summary traces.
                snap=folder/'states';snap.mkdir(exist_ok=True)
                wanted=[]
                if step%100000==0:wanted.append(f't{step*.0001:g}s.pkl')
                if row and any(e['confirmation_s']>start_s for e in row['primary']['entries']):wanted.append('first_new_entry_checkpoint.pkl')
                if row and any(e['confirmation_s']>start_s for e in row['primary']['low_activity_exits']):wanted.append('first_new_exit_checkpoint.pkl')
                for label in wanted:
                    if not (snap/label).exists():shutil.copy2(folder/'checkpoint.pkl',snap/label)
        return observe
    fixed.OUT=OUT;fixed.prepare=lambda:p;fixed.TerminationSlow=RecoveryWindowSlow;fixed.observation_sink=factory
    try:fixed.worker(name)
    finally:fixed.OUT,fixed.prepare,fixed.TerminationSlow,fixed.observation_sink=old
    result=base.read(folder/'result.json');result['display_stop_s']=result['end_s']
    if result['status']!='CENSORED_WALL_DEADLINE':result['tracker']['stop_reason']='SIMULATION_HORIZON'
    result['new_fixed_feedback_law']=p['new_fixed_equation'];write(folder/'result.json',result);write(folder/'progress.json',result)
    analyze(name)


def report():
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    with (logs/'report.log').open('a') as h:
        code=subprocess.call([sys.executable,str(Path(__file__).with_name('report_topic4_recovery_window.py'))],stdout=h,stderr=subprocess.STDOUT)
    write(OUT/'report_status.json',dict(exit_code=code,time=time.time(),human_review='PENDING'))


def supervise():
    p=prepare();lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert base.read(OUT/'mechanism_qa.json')['status']=='PASS'
    if 'launched_epoch' not in p:
        p['launched_epoch']=time.time();p['deadline_epoch']=time.time()+p['wall_budget_hours']*3600;write(OUT/'protocol.json',p)
    running={};failures=[];logs=OUT/'logs';logs.mkdir(exist_ok=True);last_report=-1
    while True:
        for name,(proc,h) in list(running.items()):
            if proc.poll() is not None:
                h.close();del running[name]
                if proc.returncode:failures.append(dict(name=name,exit_code=proc.returncode))
        done=[j['name'] for j in p['initial_jobs'] if (OUT/'runs'/j['name']/'result.json').exists()]
        failed={f['name'] for f in failures}
        pending=[j for j in p['initial_jobs'] if j['name'] not in done and j['name'] not in running and j['name'] not in failed]
        can_start=time.time()<p['deadline_epoch']-600 and not failures
        while pending and len(running)<p['max_workers'] and can_start:
            if psutil.virtual_memory().available/2**30<p['min_available_memory_GiB'] or shutil.disk_usage(OUT).free/2**30<p['disk_reserve_GiB']:break
            j=pending.pop(0);h=(logs/f'{j["name"]}.log').open('a')
            proc=subprocess.Popen([sys.executable,'-u',__file__,'worker','--name',j['name']],stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
            running[j['name']]=(proc,h);print('START',j['name'],proc.pid,flush=True);time.sleep(2)
        write(OUT/'status.json',dict(updated_epoch=time.time(),pid=os.getpid(),running={n:pr.pid for n,(pr,h) in running.items()},
            queued=[j['name'] for j in pending],finished=done,failures=failures,deadline_epoch=p['deadline_epoch']))
        # Report completed conditions; prefix figures are also available via manual report.
        if len(done)!=last_report:
            for name in done:
                if name not in running and not (OUT/'analysis'/f'{name}.json').exists():analyze(name)
            report();last_report=len(done)
        if not running and (not pending or failures or not can_start):break
        time.sleep(15)
    missing=[j['name'] for j in p['initial_jobs'] if not (OUT/'runs'/j['name']/'result.json').exists()]
    censored=[j['name'] for j in p['initial_jobs'] if (OUT/'runs'/j['name']/'result.json').exists()
              and base.read(OUT/'runs'/j['name']/'result.json')['end_s']<j['horizon_s']]
    write(OUT/'batch_complete.json',dict(status='FAILED' if failures else 'CENSORED' if missing or censored else 'FINISHED',
        finished_epoch=time.time(),failures=failures,missing_results=missing,censored=censored,human_review='PENDING'))
    report()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','qa','worker','supervise','analyze']);ap.add_argument('--name');a=ap.parse_args()
    try:globals()[a.action](a.name) if a.action in ['worker','analyze'] else globals()[a.action]()
    except Exception as exc:
        dest=OUT/'runs'/a.name/'failure.json' if a.name else OUT/f'{a.action}_failure.json'
        write(dest,dict(error=repr(exc),time=time.time()));raise
