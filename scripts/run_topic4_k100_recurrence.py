#!/usr/bin/env python3
"""k100-parameterized slow-K matrix on the fixed Fig5 Z/M two-core SNN.

Same model family as fig5_interictal_recurrence_20260915. The only change is
the parameter conversion K_gain = k100 / tau_K_s (per-spike increment
0.01*k100/tau_K_s), so that the steady-state gK/gL at a sustained 100 Hz
single-neuron rate equals k100 independent of tau_K. No new equation, state
variable, filter, resupply term, reset or parameter switching.

Actions: prepare | qa | worker --name | supervise | branch ... | reconcile
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
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import psutil
import analyze_topic4_interictal_recurrence as audit
import run_topic4_zm_matched_spatial_termination as matched

fixed=matched.fixed
carrier=fixed.carrier
base=carrier.base
PREVIOUS=audit.OUT
OUT=Path('/data/hfosp/topic4_sef_hfo/fig5_interictal_recurrence_k100_matrix_20260916')
K100=[.1,.25,.5]
TAU=[1.,2.5,5.]
SEED=9108401
HORIZON_S=120.
SNAPSHOT_STRIDE_STEPS=100000   # keep a full-state copy every 10 s for causal branches
DEADLINE_LOCAL='2026-09-16 07:30:00'


def sha(path):return base.sha(path)


def derive(k100,tau_s):
    gain=k100/tau_s
    return dict(k100=float(k100),sahp_tau_s=float(tau_s),sahp_gain=float(gain),
                per_spike_increment_gK_over_gL=.01*gain,
                steady_state_gK_over_gL_at_100Hz=.01*gain*tau_s*100.)


def job_name(k100,tau,seed):return f'k100_{k100:g}_tau{tau:g}_s{seed}'


class K100Slow(matched.MatchedSlow):
    """MatchedSlow plus regional observation and explicit branch modifications.

    With every modification at its default the equations are identical to
    MatchedSlow (verified in qa). Modifications are only used by declared
    causal-diagnostic branches and never by autonomous matrix runs.
    """
    z_gate_off=False
    k_freeze=False
    record_regions=True
    instance=None
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        K100Slow.instance=self
        self.regional_records=[];self._groups=None
    def region_groups(self):
        if self._groups is None:
            with np.load(OUT/'geometry.npz') as g:
                pos=g['positions_e'];centers=g['centers_mm'];counts=g['region_counts'][:3]
            d=np.linalg.norm(pos[:,None]-centers[None],axis=2);grp=np.full(len(pos),2)
            grp[d[:,0]<1.75]=0;grp[(d[:,1]<1.75)&(d[:,1]<d[:,0])]=1
            assert np.array_equal(np.bincount(grp,minlength=3),counts)
            self._groups=[np.flatnonzero(grp==k) for k in range(3)]
        return self._groups
    def apply_currents(self,ie,ii,labels=None,rec=None):
        if self.z_gate_off:
            kept=self.z;self.z=np.ones_like(kept)
            try:value=super().apply_currents(ie,ii,labels,rec)
            finally:self.z=kept
        else:
            value=super().apply_currents(ie,ii,labels,rec)
        if self.record_regions and self._step_index%200==0 and self.voltage is not None:
            ne=self.NE;z=np.ones(ne) if self.z_gate_off else self.z[:ne]
            local=ii[:ne]*(1-self.spatial_fraction);v=self.voltage[:ne]
            g_k=self.g_k if self.sahp_gain else np.zeros(ne)
            row=[self._step_index*.1]
            for ix in self.region_groups():
                row+=[g_k[ix].mean(),self.g_global[ix].mean(),v[ix].mean(),
                      (g_k[ix]*(v[ix]-self.sahp_reversal_mV)).mean(),
                      (self.g_global[ix]*(v[ix]-self.global_reversal)).mean(),
                      ie[ix].mean(),(z[ix]*local[ix]).mean(),self.cfg.eta_m*self.m[ix].mean(),
                      self.z[ix].mean(),self.m[ix].mean()]
            self.regional_records.append(row)
        return value
    def step(self,spk,labels,dt):
        if self.k_freeze:
            kept=self.g_k.copy();super().step(spk,labels,dt);self.g_k[:]=kept
        else:
            super().step(spk,labels,dt)


REGIONAL_KEYS=['gK_over_gL','gG_over_gL','V_mV','I_K_mV','I_G_mV','I_E_mV','Z_local_I_mV','I_M_mV','Z','M']
REGION_NAMES=['coreA','coreB','other']


def flush_regional(folder,step):
    o=K100Slow.instance
    if o is None or not o.regional_records:return
    rec=np.asarray(o.regional_records);dest=folder/'regional_chunks';dest.mkdir(exist_ok=True)
    path=dest/f'{round(rec[0,0]*10):010d}_{int(step):010d}.npz'
    out=dict(time_ms=rec[:,0],region_names=np.array(REGION_NAMES),keys=np.array(REGIONAL_KEYS),
             values=rec[:,1:].reshape(len(rec),3,len(REGIONAL_KEYS)))
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**out);tmp.replace(path)
    o.regional_records.clear()


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        return base.read(OUT/'protocol.json')
    parent=base.read(PREVIOUS/'protocol.json')
    template=next(j for j in parent['initial_jobs'] if j['name']=='g0.166667_k0.1_tau1_s9108401')
    assert abs(template['gamma']-1/6)<1e-9 and template['feedback_form']=='conductance'
    jobs=[];reused=[];queue=[]
    priority=[(.5,1.),(.5,2.5),(.25,1.),(.25,2.5),(.25,5.),(.1,2.5),(.1,5.),(.1,1.),(.5,5.)]
    for k100,tau in priority:
        d=derive(k100,tau);j=copy.deepcopy(template)
        j.update(name=job_name(k100,tau,SEED),seed=SEED,horizon_s=HORIZON_S,checkpoint_s=2.,
                 device=len(jobs)%2,stop_after_second_entry=False,qa=False,**d)
        old=next((o for o in parent['initial_jobs'] if abs(o['gamma']-j['gamma'])<1e-12
                  and o['sahp_gain']==j['sahp_gain'] and o['sahp_tau_s']==j['sahp_tau_s']),None)
        if old is not None:
            folder=PREVIOUS/'runs'/old['name']
            applied=base.read(folder/'applied_configuration.json');result=base.read(folder/'result.json')
            physical=['seed','eta_m','tau_M_s','tau_Z_s','threshold','mode','gamma','global_gain',
                      'global_resource','phi_jump','feedback_form','C_R','reference_voltage_mV',
                      'reference_gap_mV','global_reversal_mV','sahp_gain','sahp_tau_s']
            mismatch=[k for k in physical if applied['job'][k]!=j[k]]
            assert not mismatch,mismatch
            assert applied['identity']==parent['identity'] and applied['added_sahp_gain']==j['sahp_gain']
            assert applied['added_sahp_tau_ms']==1000.*j['sahp_tau_s'] and applied['native_Z_on'] and applied['native_M_on']
            assert applied['eta_M']==j['eta_m'] and applied['tau_M_ms']==1000.*j['tau_M_s']
            reused.append(dict(name=j['name'],previous_name=old['name'],previous_run=str(folder),
                               previous_analysis=str(PREVIOUS/'analysis'/f"{old['name']}.json"),
                               observed_s=result['end_s'],runtime_status=result['status'],
                               identity_verified=True,executed_parameters_verified=physical,**d))
            continue
        jobs.append(j);queue.append(j['name'])
    assert len(jobs)==7 and len(reused)==2
    p=copy.deepcopy(parent)
    p.update(initial_jobs=jobs,reused_previous_conditions=reused,branch_jobs=[],
             created_epoch=time.time(),
             deadline_epoch=time.mktime(time.strptime(DEADLINE_LOCAL,'%Y-%m-%d %H:%M:%S')),
             status='PROSPECTIVE_K100_MATRIX_SCREEN',
             producer_sha256=sha(carrier.__file__),wrapper_sha256=sha(fixed.__file__),
             matched_producer_sha256=sha(matched.__file__),audit_producer_sha256=sha(audit.__file__),
             k100_producer_sha256=sha(__file__),
             max_workers=6,min_available_memory_GiB=60.,disk_reserve_GiB=40.,
             source_round=str(PREVIOUS),
             question='With K strength (k100) and K time scale (tau_K) separated, does any condition give high activity -> exit -> recurrent brief events -> high activity again in one autonomous trajectory, and what limits the others?',
             k_parameterization=dict(rule='K_gain = k100 / tau_K_s; per-spike increment = 0.01*k100/tau_K_s; k100 = steady-state gK/gL of one neuron firing 100 Hz',
                 same_model_family=True,new_mechanism=False,
                 previous_round_confound='Fixed K_gain with different tau_K changed the mean feedback in proportion to tau_K, so the old tau 1 s versus 5 s contrast mixed time scale and strength.',
                 source_reference='Liou et al. 2020 eLife 50927: dgK = -gK/tau_K dt + gK_max/tau_K per spike, i.e. the per-spike increment already scales with 1/tau_K and the steady state gK_max*rate is tau_K independent; the local implementation had fixed the increment at 0.01*gain irrespective of tau_K.'),
             matrix=dict(k100=K100,tau_K_s=TAU,gamma=1/6,seed=SEED,horizon_s=HORIZON_S,
                 triage='Every new job is inspected at about 30 s; the two most informative conditions are then continued toward 90-120 s and receive additional independent noise seeds, others are stopped at a checkpoint.'),
             unchanged=dict(eta_M=template['eta_m'],tau_M_s=template['tau_M_s'],tau_Z_s=template['tau_Z_s'],
                 threshold=template['threshold'],gamma=template['gamma'],global_gain=template['global_gain'],
                 global_reversal_mV=template['global_reversal_mV'],C_R=template['C_R'],
                 sahp_reversal_mV=matched.MatchedSlow.sahp_reversal_mV,topology_field_seed=6101,noise_seed=SEED),
             no_state_or_parameter_reset=True,no_new_filter=True,no_extra_state_variable=True,
             no_10s_pool=True,no_added_Z_recovery=True,no_external_reset=True,
             observation_stop='Full horizon or wall deadline. A verified temporal loop does not stop the run; it is flagged and the run continues so that further loops and noise robustness can be inspected.',
             causal_branches='Optional branch jobs restart from a stored full-state checkpoint of a matrix run with exactly one feedback term modified (K increment scale, K frozen, Z gating removed). They are diagnostics and are never counted as autonomous loops.',
             snapshot_stride_s=SNAPSHOT_STRIDE_STEPS*1e-4,
             temporal_rule=audit.RULE,queue=queue,
             engine_dependency_note='The checkpoint/state module is imported from the worktree topic4-substrate-autapse-fix (path recorded in source_hashes); unchanged during this batch.')
    import checkpoint as ckpt_module
    p['source_hashes']=dict(parent['source_hashes']);p['source_hashes'][str(Path(ckpt_module.__file__).resolve())]=sha(ckpt_module.__file__)
    versions=OUT/'implementation';versions.mkdir(parents=True,exist_ok=True)
    for path in [Path(__file__),Path(audit.__file__),Path(matched.__file__),Path(fixed.__file__),Path(carrier.__file__)]:
        shutil.copy2(path,versions/path.name)
    for j in jobs:base.write(OUT/'jobs'/f"{j['name']}.json",j)
    shutil.copy2(PREVIOUS/'geometry.npz',OUT/'geometry.npz')
    base.write(OUT/'queue.json',dict(names=queue))
    base.write(OUT/'protocol.json',p)
    return p


def qa():
    p=prepare()
    audit.OUT=OUT;audit.qa()
    matched.OUT=OUT;matched.MatchedSlow=K100Slow
    K100Slow.sahp_tau_ms=5000.;K100Slow.sahp_gain=0.;K100Slow.record_regions=False
    matched.qa()
    cfg=base.old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000,I_th_EI=95.19851312666987,tau_adp=1000,eta_m=.0005)
    rng=np.random.default_rng(1601);report=dict(status='PASS',checks=[])
    def make(cls,gain,tau_s,**mods):
        cls.C_R=p['reference_current_scale'];cls.feedback_form='conductance'
        cls.sahp_gain=gain;cls.sahp_tau_ms=1000.*tau_s
        obj=cls(12,18,cfg,NE=10,mode='native',gamma=1/6,global_gain=p['unchanged']['global_gain'],global_resource='native_z',phi_jump=0.)
        obj.global_reversal=p['unchanged']['global_reversal_mV']
        for k,v in mods.items():setattr(obj,k,v)
        return obj
    # 1. converted old parameters reproduce the old update exactly (bitwise)
    for old_gain,old_tau,k100 in [(.1,1.,.1),(.1,5.,.5)]:
        d=derive(k100,old_tau);assert d['sahp_gain']==old_gain
        old_obj=make(OldMatched,old_gain,old_tau);new_obj=make(K100Slow,d['sahp_gain'],d['sahp_tau_s'],record_regions=False)
        for obj in [old_obj,new_obj]:
            obj.voltage=rng.uniform(-20,18,12);obj.r_global=40.;obj.g_k[:]=rng.uniform(0,1,10)
        new_obj.g_k[:]=old_obj.g_k;new_obj.voltage=old_obj.voltage.copy()
        for _ in range(50):
            ie,ii=rng.uniform(0,400,(2,12));sp=rng.random(12)<.3
            assert np.array_equal(old_obj.apply_currents(ie,ii),new_obj.apply_currents(ie,ii))
            old_obj.step(sp,None,.1);new_obj.step(sp,None,.1)
            assert np.array_equal(old_obj.g_k,new_obj.g_k) and np.array_equal(old_obj.z,new_obj.z) and np.array_equal(old_obj.m,new_obj.m)
        report['checks'].append(dict(check='converted parameters reproduce old K update bitwise',old_gain=old_gain,old_tau_s=old_tau,k100=k100,derived=d))
    # 2. same k100, different tau_K: equal long-run mean, different build-up / decay
    scalar=[]
    for k100 in K100:
        row=[]
        for tau in TAU:
            d=derive(k100,tau);inc=d['per_spike_increment_gK_over_gL'];dec=np.exp(-.1/(1000.*tau));P=100
            # 100 Hz regular spiking: one increment every P=100 steps of 0.1 ms; exact per-period recursion
            g=0.;minima=[]
            for period in range(4000):
                g=g*dec**P+inc;minima.append(g*dec**(P-1))
            minima=np.asarray(minima)
            time_average=g*(1-dec**P)/(P*(1-dec))          # exact cycle average once stationary
            rise=np.flatnonzero(minima>=.632*minima[-1])[0]*.01
            row.append(dict(tau_K_s=tau,K_gain=d['sahp_gain'],per_spike=inc,stationary_time_average=float(time_average),
                            relative_error_vs_k100=float(abs(time_average-k100)/k100),time_to_63pct_s=float(rise)))
            assert abs(time_average-k100)/k100<1e-3,(k100,tau,time_average)
            assert abs(rise-tau)/tau<.03,(k100,tau,rise)
        means=[r['stationary_time_average'] for r in row];rises=[r['time_to_63pct_s'] for r in row]
        assert max(means)-min(means)<1e-3*k100 and rises==sorted(rises) and rises[-1]>4*rises[0]
        scalar.append(dict(k100=k100,rows=row))
    report['checks'].append(dict(check='same k100 gives the same 100 Hz stationary time-average gK/gL for every tau_K; 63% rise time equals tau_K',results=scalar))
    # 3. dt / ms / s conversion: 1 s of decay at tau_K=1 s multiplies by exp(-1)
    dec=np.exp(-.1/1000.);g=1.
    for _ in range(10000):g*=dec
    assert abs(g-np.exp(-1))<1e-9
    for tau in TAU:
        obj=make(K100Slow,derive(.25,tau)['sahp_gain'],tau,record_regions=False);obj.voltage=np.full(12,0.);obj.r_global=0.
        obj.g_k[:]=1.;obj.apply_currents(np.zeros(12),np.zeros(12));obj.step(np.zeros(12,bool),None,.1)
        assert np.allclose(obj.g_k,np.exp(-.1/(1000.*tau)),rtol=0,atol=1e-15)
        sp=np.zeros(12,bool);sp[3]=True;obj.g_k[:]=0.;obj.step(sp,None,.1)
        assert obj.g_k[3]==.01*.25/tau and obj.g_k[4]==0.
    report['checks'].append(dict(check='dt=0.1 ms with tau_K in ms: 10000 steps give exp(-1); per-spike increment equals 0.01*k100/tau_K_s',passed=True))
    # 4. Liou source normalization: per-spike increment gK_max/tau_K, steady state tau_K independent
    gkmax,gL,unit=40.,4.,.2
    liou_inc=lambda tau_ms:gkmax/tau_ms/(gL*unit)
    assert abs(liou_inc(5000.)-.01)<1e-15
    liou_k100=[liou_inc(tau)*tau*100/1000. for tau in [1000.,2500.,5000.]]
    assert np.allclose(liou_k100,5.)
    report['checks'].append(dict(check='Liou 2020 sAHP normalisation',per_spike_increment_gK_over_gL=dict(tau_1s=liou_inc(1000.),tau_2p5s=liou_inc(2500.),tau_5s=liou_inc(5000.)),
        k100_at_gain1=liou_k100,statement='Source increment is gK_max/tau_K per spike, so the source steady state gK_max*rate is tau_K independent; k100 of the source at gain 1 is 5.0 for every tau_K. This batch spans k100 0.1-0.5, i.e. 10-50 times weaker than the source scale.'))
    # 5. K100Slow with default modifications equals MatchedSlow; modifications act as declared
    for _ in range(3):
        a=make(OldMatched,.2,2.5);b=make(K100Slow,.2,2.5,record_regions=False)
        for obj in [a,b]:obj.voltage=rng.uniform(-20,18,12);obj.r_global=60.
        b.voltage=a.voltage.copy();a.g_k[:]=rng.uniform(0,2,10);b.g_k[:]=a.g_k;a.z[:10]=rng.uniform(.1,1,10);b.z[:]=a.z;a.m[:10]=rng.uniform(0,300,10);b.m[:]=a.m
        for _ in range(20):
            ie,ii=rng.uniform(0,900,(2,12));sp=rng.random(12)<.4
            assert np.array_equal(a.apply_currents(ie,ii),b.apply_currents(ie,ii))
            a.step(sp,None,.1);b.step(sp,None,.1)
            assert np.array_equal(a.g_k,b.g_k) and np.array_equal(a.z,b.z) and np.array_equal(a.m,b.m) and a.r_global==b.r_global
    b=make(K100Slow,.2,2.5,record_regions=False,z_gate_off=True);b.voltage=np.full(12,5.);b.r_global=60.;b.z[:10]=.3;z_before=b.z.copy()
    c=make(OldMatched,.2,2.5);c.voltage=np.full(12,5.);c.r_global=60.;c.z[:]=1.
    ie,ii=rng.uniform(0,900,(2,12))
    assert np.array_equal(b.apply_currents(ie,ii),c.apply_currents(ie,ii)) and np.array_equal(b.z,z_before)
    sp=np.zeros(12,bool);b.step(sp,None,.1);assert np.all(b.z[:10]!=z_before[:10])  # depletion dynamics continue
    f=make(K100Slow,.2,2.5,record_regions=False,k_freeze=True);f.voltage=np.full(12,5.);f.g_k[:]=.7;f.apply_currents(ie,ii);f.step(np.ones(12,bool),None,.1)
    assert np.all(f.g_k==.7)
    report['checks'].append(dict(check='K100Slow default == MatchedSlow bitwise; z_gate_off applies Z=1 to delivery only while Z keeps evolving; k_freeze holds gK',passed=True))
    K100Slow.sahp_gain=0.;K100Slow.sahp_tau_ms=5000.;K100Slow.record_regions=True;K100Slow.z_gate_off=False;K100Slow.k_freeze=False
    for path,h in p['source_hashes'].items():assert sha(path)==h,path
    base.write(OUT/'k_reparameterization_qa.json',report)
    base.write(OUT/'kinetic_qa.json',dict(status='PASS',source='k_reparameterization_qa.json'))
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))


class OldMatched(matched.MatchedSlow):
    """Unmodified parent for parity checks (separate class so class attributes do not leak)."""


def snapshot_and_live(folder,name,step):
    flush_regional(folder,step)
    if step%SNAPSHOT_STRIDE_STEPS==0:
        dest=folder/'checkpoints';dest.mkdir(exist_ok=True)
        tmp=dest/f'step_{step:010d}.tmp';shutil.copy2(folder/'checkpoint.pkl',tmp);tmp.replace(dest/f'step_{step:010d}.pkl')
    try:
        row=audit.analyze_folder(folder,OUT/'geometry.npz',sensitivities=False)
        if row:
            audit.old.write(OUT/'analysis'/f'{name}_live.json',row)
            pp=row['primary'];k=audit.old.load(folder,'intrinsic_adaptation_chunks');d=audit.old.load(folder,keys=['Z'])
            audit.old.write(folder/'live_status.json',dict(time_s=step*1e-4,classification=pp['classification'],
                 entries=pp['entries'],exits=pp['low_activity_exits'],preentry_brief=pp['preentry']['brief_count'],
                 temporal_loop_pass=pp['temporal_loop_pass'],n_events=len(pp['events']),
                 Z=float(d['Z'][-1,0]) if len(d) else None,gK=float(k['sahp_mean_conductance_ratio'][-1]) if k else None,
                 updated_epoch=time.time()))
    except Exception as exc:   # observation only; never interrupt the simulation
        audit.old.write(folder/'live_status_error.json',dict(error=repr(exc),step=int(step)))


def worker(name):
    p=prepare();assert p['k100_producer_sha256']==sha(__file__)
    assert base.read(OUT/'k_reparameterization_qa.json')['status']=='PASS'
    assert base.read(OUT/'temporal_observer_qa.json')['status']=='PASS'
    job=base.read(OUT/'jobs'/f'{name}.json');folder=OUT/'runs'/name
    branch=job.get('branch')
    K100Slow.z_gate_off=False;K100Slow.k_freeze=False;K100Slow.record_regions=True
    if branch:
        mod=branch['modification']
        K100Slow.z_gate_off=bool(mod.get('z_gate_off',False));K100Slow.k_freeze=bool(mod.get('k_freeze',False))
    sink0=fixed.observation_sink
    def sink_factory(sink,job,deadline):
        full=sink0(sink,job,deadline)
        def observe(step,state):
            try:full(step,state)
            finally:snapshot_and_live(folder,name,step)
        return observe
    fixed.observation_sink=sink_factory;matched.OUT=OUT;matched.MatchedSlow=K100Slow
    try:matched.worker(name)
    finally:fixed.observation_sink=sink0
    result=base.read(folder/'result.json')
    result['display_stop_s']=result['end_s']
    result['tracker']['stop_reason']='WALL_DEADLINE' if result['status']=='CENSORED_WALL_DEADLINE' else 'SIMULATION_HORIZON'
    if branch:result['branch']=branch
    audit.old.write(folder/'result.json',result);audit.old.write(folder/'progress.json',result)
    row=audit.analyze_folder(folder,OUT/'geometry.npz')
    audit.old.write(OUT/'analysis'/f'{name}.json',row)


def reconcile(names=None):
    """Write result.json for runs stopped at a checkpoint (triage), then final analysis."""
    p=prepare();done=[]
    for job in p['initial_jobs']+p['branch_jobs']:
        if names and job['name'] not in names:continue
        folder=OUT/'runs'/job['name']
        if (folder/'result.json').exists() or not (folder/'checkpoint.pkl').exists():continue
        if any('worker' in (x.info['cmdline'] or []) and job['name'] in (x.info['cmdline'] or []) for x in psutil.process_iter(['cmdline'])):
            continue
        with (folder/'checkpoint.pkl').open('rb') as h:saved=pickle.load(h)
        end_step=0
        for file in sorted((folder/'chunks').glob('*.npz')):
            if '.tmp.' in file.name:continue
            with np.load(file) as a:
                assert int(a['start_step'])==end_step;end_step=int(a['end_step'])
        assert saved['engine']['step']==end_step and saved['job']==job
        result=dict(status='STOPPED_AT_CHECKPOINT_FOR_TRIAGE',job=job,identity=p['identity'],tracker=saved['tracker'],
                    end_s=end_step*1e-4,display_stop_s=end_step*1e-4,no_external_intervention=True,M_reset=False,Z_reset=False,
                    source_hashes=p['source_hashes'],stopped_epoch=time.time())
        result['tracker']['stop_reason']='STOPPED_FOR_TRIAGE'
        if job.get('branch'):result['branch']=job['branch']
        audit.old.write(folder/'result.json',result);audit.old.write(folder/'progress.json',result)
        row=audit.analyze_folder(folder,OUT/'geometry.npz');audit.old.write(OUT/'analysis'/f"{job['name']}.json",row)
        done.append(job['name'])
    print(json.dumps(done))


def add_seed(k100,tau,seed):
    p=prepare();name=job_name(k100,tau,seed)
    assert not any(j['name']==name for j in p['initial_jobs'])
    template=next(j for j in p['initial_jobs'] if j['k100']==k100 and j['sahp_tau_s']==tau)
    j=copy.deepcopy(template);j.update(name=name,seed=int(seed),device=len(p['initial_jobs'])%2,
                                        independent_noise_seed_of=template['name'])
    p['initial_jobs'].append(j);base.write(OUT/'jobs'/f'{name}.json',j);base.write(OUT/'protocol.json',p)
    q=base.read(OUT/'queue.json');q['names'].append(name);base.write(OUT/'queue.json',q)
    print(name)


def branch(source,step,modification,horizon_s,label):
    p=prepare();src_job=next(j for j in p['initial_jobs']+p['branch_jobs'] if j['name']==source) if not source.startswith('/') else None
    if src_job is None:   # branch from a previous-round run folder
        src_folder=Path(source);src_job=base.read(src_folder/'applied_configuration.json')['job'];snap=src_folder/'checkpoint.pkl'
        with snap.open('rb') as h:saved=pickle.load(h)
        assert saved['engine']['step']==step
    else:
        src_folder=OUT/'runs'/source;snap=src_folder/'checkpoints'/f'step_{step:010d}.pkl'
        with snap.open('rb') as h:saved=pickle.load(h)
    assert saved['engine']['step']==step and saved['identity']==p['identity']
    name=f"{Path(source).name}_br{step*1e-4:g}s_{label}"
    j=copy.deepcopy(src_job)
    if 'k100' not in j:j.update(**derive(j['sahp_gain']*j['sahp_tau_s'],j['sahp_tau_s']))
    scale=float(modification.get('k_increment_scale',1.))
    if scale!=1.:
        j['base_sahp_gain']=j['sahp_gain'];j['sahp_gain']=j['sahp_gain']*scale;j['k_increment_scale_applied']=scale
    j.update(name=name,horizon_s=step*1e-4+horizon_s,checkpoint_s=1.,device=len(p['branch_jobs'])%2,stop_after_second_entry=False,qa=False,
             branch=dict(source=str(source),source_checkpoint=str(snap),source_checkpoint_sha256=sha(snap),start_s=step*1e-4,
                         modification=modification,label=label,diagnostic_only=True,
                         note='Full engine state, synaptic/delay history, slow variables and random streams continue from the stored checkpoint; exactly the declared feedback modification differs. Not an autonomous loop.'))
    folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    for subdir in ['chunks','mechanism_chunks','intrinsic_adaptation_chunks','actual_current_chunks','regional_chunks']:
        srcdir=src_folder/subdir
        if not srcdir.exists():continue
        (folder/subdir).mkdir(exist_ok=True)
        for path in sorted(srcdir.glob('*.npz')):
            if '.tmp.' in path.name or int(path.stem.split('_')[-1])>step:continue
            target=folder/subdir/path.name
            if target.exists():continue
            try:os.link(path,target)
            except OSError:shutil.copy2(path,target)
    saved['job']=j;saved['engine']['slow']['kind']='K100Slow'
    base.save_pickle(folder/'checkpoint.pkl',saved)
    p['branch_jobs'].append(j);base.write(OUT/'jobs'/f'{name}.json',j);base.write(OUT/'protocol.json',p)
    q=base.read(OUT/'queue.json');q['names'].append(name);base.write(OUT/'queue.json',q)
    print(name)


def supervise():
    p=prepare()
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    running={};failures=[];stopped=[];logs=OUT/'logs';logs.mkdir(exist_ok=True)
    def all_jobs():
        pp=base.read(OUT/'protocol.json');return {j['name']:j for j in pp['initial_jobs']+pp['branch_jobs']}
    while True:
        jobs=all_jobs();queue=base.read(OUT/'queue.json')['names']
        stops=base.read(OUT/'stop_requests.json')['names'] if (OUT/'stop_requests.json').exists() else []
        hold=base.read(OUT/'hold.json')['names'] if (OUT/'hold.json').exists() else []
        for name,(proc,h) in list(running.items()):
            if name in stops and proc.poll() is None:
                proc.send_signal(signal.SIGTERM);stopped.append(name)
            if proc.poll() is not None:
                h.close();del running[name]
                if proc.returncode and name not in stopped:failures.append(dict(name=name,exit_code=proc.returncode))
        pending=[n for n in queue if n not in running and n not in stops and n not in hold and not (OUT/'runs'/n/'result.json').exists()
                 and n not in [f['name'] for f in failures]]
        max_workers=base.read(OUT/'protocol.json')['max_workers']
        while pending and len(running)<max_workers and time.time()<p['deadline_epoch']-1800:
            if psutil.virtual_memory().available/2**30<p['min_available_memory_GiB'] or shutil.disk_usage(OUT).free/2**30<p['disk_reserve_GiB']:break
            name=pending.pop(0);h=(logs/f'{name}.log').open('a')
            proc=subprocess.Popen([sys.executable,'-u',__file__,'worker','--name',name],stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
            running[name]=(proc,h);print('START',name,proc.pid,flush=True);time.sleep(30)
        base.write(OUT/'status.json',dict(updated_epoch=time.time(),running={n:proc.pid for n,(proc,h) in running.items()},
                 queued=pending,stopped=stopped,failures=failures,deadline_epoch=p['deadline_epoch'],
                 finished=[n for n in jobs if (OUT/'runs'/n/'result.json').exists()],
                 memory_available_GiB=psutil.virtual_memory().available/2**30,load=os.getloadavg()[0]))
        if not running and not pending and ((OUT/'finish.flag').exists() or time.time()>=p['deadline_epoch']):break
        time.sleep(20)
    base.write(OUT/'batch_complete.json',dict(updated_epoch=time.time(),failures=failures,stopped=stopped,
               full_Fig5_acceptance='NOT_ESTABLISHED',next='Scientific review; do not automatically expand or freeze.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','qa','worker','supervise','reconcile','add_seed','branch'])
    ap.add_argument('--name');ap.add_argument('--names',nargs='*');ap.add_argument('--k100',type=float);ap.add_argument('--tau',type=float);ap.add_argument('--seed',type=int)
    ap.add_argument('--source');ap.add_argument('--step',type=int);ap.add_argument('--modification');ap.add_argument('--horizon',type=float,default=6.);ap.add_argument('--label')
    args=ap.parse_args()
    try:
        if args.action=='worker':worker(args.name)
        elif args.action=='reconcile':reconcile(args.names)
        elif args.action=='add_seed':add_seed(args.k100,args.tau,args.seed)
        elif args.action=='branch':branch(args.source,args.step,json.loads(args.modification),args.horizon,args.label)
        else:globals()[args.action]()
    except Exception as exc:
        target=OUT/'runs'/args.name/'failure.json' if args.name else OUT/f'{args.action}_failure.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        audit.old.write(target,dict(error=repr(exc),time=time.time()));raise
