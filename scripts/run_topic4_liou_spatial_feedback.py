#!/usr/bin/env python3
"""Liou's uniform E-output projection with fast synapses on the fixed Fig5 SNN.

Reproduces the global spatial operator, not the entire conductance LAS model.
Native local explicit-I circuitry, Z/M, two cores and noise are retained.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import copy
import fcntl
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import psutil
import run_topic4_autonomous_recovery as carrier
import checkpoint
from run_topic4_nativeZ_continuation import checkpoint_policy

ROOT = carrier.ROOT
PREVIOUS = carrier.OUT
OUT = ROOT/'results/topic4_sef_hfo/liou_spatial_feedback_20260915'
LITERATURE = PREVIOUS/'literature/LAS-Model'


class SpatialFeedbackSlow(carrier.RecoverySlow):
    def __init__(self, *args, spatial_gamma=1/6, current_per_Hz=1., synapse_ms=15., **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_gamma = float(spatial_gamma)
        self.current_per_Hz = float(current_per_Hz)
        self.synapse_ms = float(synapse_ms)
        assert 0 <= self.spatial_gamma <= 1 and self.current_per_Hz > 0 and self.synapse_ms > 0
        self.global_synaptic_rate = 0.
        self.spatial_records = []

    def apply_currents(self, ie, ii, labels=None, rec=None):
        self.raw_mean = float(ii[:self.NE].mean())
        self.global_current = self.spatial_gamma*self.current_per_Hz*self.global_synaptic_rate
        if self.spatial_gamma == 0:
            self.delivered = ii
        else:
            self.delivered = ii.copy()
            self.delivered[:self.NE] = (1-self.spatial_gamma)*ii[:self.NE]+self.global_current
        value = carrier.base.old.MZSlowVars.apply_currents(self, ie, self.delivered, labels, rec)
        if self._step_index % 10 == 0:
            self.spatial_records.append([
                self._step_index*.1, self.global_synaptic_rate, self.global_current,
                self.global_current*float(self.z[:self.NE].mean()), self.raw_mean,
                (1-self.spatial_gamma)*self.raw_mean,
                np.mean(self.delivered[:self.NE] < self.cfg.I_th_EI)])
        return value

    def step(self, spk, labels, dt):
        super().step(spk, labels, dt)
        # sum(E spikes)/NE is exactly the archived uniform spatial projection.
        # Synaptic impulse has area1/NE in continuous time; rate unit is Hz.
        self.global_synaptic_rate = (
            self.global_synaptic_rate*np.exp(-float(dt)/self.synapse_ms)
            + np.count_nonzero(spk[:self.NE])/self.NE*1000/self.synapse_ms)


def calibration():
    rows = []
    for seed, branch in [(9108401, 'paired_recurrence_confirmation_round8'),
                         (9108402, 'native_global_confirmation_round9')]:
        folder = PREVIOUS/branch/'runs'/f'resource_rho0_k200_tau10_s{seed}'
        path = sorted((folder/'chunks').glob('*.npz'))[0]
        with np.load(path) as a:
            take = (a['time_ms'] >= 500) & (a['time_ms'] < 8000)
            r_e = a['spikes_1ms'][take, 0].sum()/32000/7.5
            take = (a['slow_time_ms'] >= 500) & (a['slow_time_ms'] < 8000)
            local_i = a['currents'][take, 1].mean()
        # Verify the old added pool is inactive throughout the calibration.
        with np.load(sorted((folder/'pool_chunks').glob('*.npz'))[0]) as a:
            take = (a['time_ms'] >= 500) & (a['time_ms'] < 8000)
            assert np.all(a['raw_global_current'][take] == 0)
        rows.append(dict(seed=seed, source=str(path), window_s=[.5, 8.],
                         E_mean_Hz=float(r_e), raw_local_II_mean=float(local_i),
                         current_per_Hz=float(local_i/r_e)))
    return dict(rows=rows, current_per_Hz=float(np.mean([r['current_per_Hz'] for r in rows])),
        method='Fixed preentry mean raw localII / meanE rate, averaged across both existing development noises. Never recalibrated per candidate or fitted to recovery.',
        limitation='Maps normalized local/global strengths to this current-LIF carrier. It is not a physical nS-to-mV conversion or an assertion that instantaneous component currents remain5:1.')


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists():
        return carrier.base.read(OUT/'protocol.json')
    p = copy.deepcopy(carrier.base.read(PREVIOUS/'protocol.json'))
    cal = calibration()
    jobs = []
    for tag, gamma in [('g0', 0.), ('g1of6', 1/6), ('g1of2', .5)]:
        for seed in [9108401, 9108402]:
            jobs.append(dict(name=f'liou_{tag}_s{seed}', mode='native', gamma=0.,
                spatial_gamma=gamma, current_per_Hz=cal['current_per_Hz'], synapse_ms=15.,
                eta_m=.005, tau_M_s=2., tau_Z_s=5., threshold=carrier.base.old.THRESHOLD,
                seed=seed, horizon_s=60., device=len(jobs) % 2, checkpoint_s=10.))
    p.update(status='DEFINED_BEFORE_RUNS', created_at=time.time(), start_epoch=time.time(),
        deadline_epoch=time.time()+4*3600, initial_jobs=jobs, max_workers=6,
        maximum_combined_workers=30, min_available_memory_GiB=80., disk_reserve_GiB=40.,
        maximum_production_conditions=6, calibration=cal,
        producer_sha256=carrier.base.sha(carrier.__file__), spatial_producer_sha256=carrier.base.sha(__file__),
        approval='User asks to attempt the original Liou global spatial feedback, replacing the added10s filter.',
        global_operator='K_global(x,y)=1/NE; input is allE spike output, not mean postsynaptic GABA or a core label.',
        temporal_operator='Only tau_syn=15ms exponential synapse, no10s state and no50Hz rectification threshold. Same causal one-step spike delivery as the carrier.',
        delivered_E_inhibition='J_i=(1-gamma)*rawLocalII_i+gamma*C*R_synE. Both membrane Zi*J_i and nativeZi depletion receive J_i. I cells retain their original input.',
        spatial_ratio='gamma1/6 gives normalized local/global5:1; gamma1/2 gives1:1, matching paper Fig4 motifs. Gamma0 is the native control.',
        exact_and_adapted='Exact uniform presynapticE spatial operator and15ms fast impulse response. Current magnitude is reference-calibrated; local explicitI kernel, currentLIF, nativeZ/M and noise remain the current model. Not a full LAS conductance/ionic reproduction.',
        no_intervention=True, new_resource_term=False, recurrent_stop='Observe full60s even after return/reentry; no short positive stopping.',
        decision='Require true allE/core termination, quiet b=1 and Z recovery, finite local events and reentry. Suppression, a lower platform or recurrent whole-field bursts do not establish recovered interictal dynamics.',
        literature_files={str(x):carrier.base.sha(x) for x in [LITERATURE/'StandardRecurrentConnection.m',LITERATURE/'SpikingModel1DStandardTemplate.m',LITERATURE/'@SpikingModel/SpikingModel.m']})
    for obsolete in ['window_hours', 'max_total_realizations', 'stage_policy', 'deadline', 'inhibition', 'native_M', 'limits']:
        p.pop(obsolete, None)
    carrier.base.write(OUT/'protocol.json', p)
    carrier.base.write(OUT/'calibration.json', cal)
    for job in jobs:
        carrier.base.write(OUT/'jobs'/(job['name']+'.json'), job)
    for name, gamma, split in [('qa_zero', 0., True), ('qa_spatial_full', 1/6, False),
                              ('qa_spatial_resume', 1/6, True)]:
        job = copy.deepcopy(jobs[0])
        job.update(name=name, spatial_gamma=gamma, eta_m=.0005, tau_M_s=1., horizon_s=1.,
                   checkpoint_s=.5 if split else 1., qa=split)
        carrier.base.write(OUT/'jobs'/(name+'.json'), job)
    return p


def worker(name):
    p = prepare()
    assert p['spatial_producer_sha256'] == carrier.base.sha(__file__)
    job = carrier.base.read(OUT/'jobs'/(name+'.json'))
    folder = OUT/'runs'/name
    capture0, restore0, wrap0 = checkpoint.capture, checkpoint.restore_slow, carrier.wrap_simulator
    def factory(*args, **kwargs):
        return SpatialFeedbackSlow(*args, **kwargs, spatial_gamma=job['spatial_gamma'],
            current_per_Hz=job['current_per_Hz'], synapse_ms=job['synapse_ms'])
    def capture(**kwargs):
        state = capture0(**kwargs)
        obj = kwargs['slow']
        state['liou_global_synaptic_rate_Hz'] = obj.global_synaptic_rate
        records = np.asarray(obj.spatial_records)
        if len(records):
            out = folder/'spatial_feedback_chunks'; out.mkdir(parents=True, exist_ok=True)
            dest = out/f'{round(records[0,0]*10):010d}_{int(kwargs["step"]):010d}.npz'
            tmp = dest.with_suffix('.tmp.npz')
            names = ['time_ms', 'synaptic_E_rate_Hz', 'raw_global_current',
                     'effective_global_current', 'raw_local_II_mean', 'scaled_local_II_mean', 'Z_recovery_drive_fraction']
            np.savez_compressed(tmp, **{k:records[:,i] for i,k in enumerate(names)})
            tmp.replace(dest); obj.spatial_records.clear()
        return state
    def restore(state, obj):
        restore0(state, obj)
        obj.global_synaptic_rate = float(state['liou_global_synaptic_rate_Hz'])
    ignored = []
    def wrap(fn, device_index):
        simulator = wrap0(fn, device_index=device_index)
        def invoke(*args, **kwargs):
            if not job.get('qa'):
                kwargs['checkpoint_sink'] = checkpoint_policy(kwargs['checkpoint_sink'],
                    round(job['horizon_s']*10000), p['deadline_epoch'], ignored)
            return simulator(*args, **kwargs)
        return invoke
    old = carrier.OUT, carrier.prepare, carrier.RecoverySlow, carrier.DEADLINE
    carrier.OUT, carrier.prepare, carrier.RecoverySlow, carrier.DEADLINE = OUT, lambda:p, factory, p['deadline_epoch']
    checkpoint.capture, checkpoint.restore_slow, carrier.wrap_simulator = capture, restore, wrap
    try:
        carrier.worker(name)
    finally:
        carrier.OUT, carrier.prepare, carrier.RecoverySlow, carrier.DEADLINE = old
        checkpoint.capture, checkpoint.restore_slow, carrier.wrap_simulator = capture0, restore0, wrap0
    if (folder/'result.json').exists():
        result = carrier.base.read(folder/'result.json')
        if not job.get('qa'):
            result['display_stop_s'] = result['end_s']
            if result['status'] == 'COMPLETE':
                assert result['end_s'] == job['horizon_s']
                result['tracker']['stop_reason'] = 'FULL_PRESPECIFIED_HORIZON'
            result['suppressed_observation_stops'] = ignored
            carrier.base.write(folder/'result.json', result)
            carrier.base.write(folder/'progress.json', result)


def equation_checks():
    cfg = carrier.base.old.MZSlowVarsConfig(use_z=True, use_m=True, tau_z=5000,
        I_th_EI=95.2, tau_adp=1000, eta_m=.0005)
    a = SpatialFeedbackSlow(12, 18, cfg, NE=10, spatial_gamma=1/6, current_per_Hz=6.)
    b = SpatialFeedbackSlow(12, 18, cfg, NE=10, spatial_gamma=1/6, current_per_Hz=6.)
    # Different spatial locations and I activity, identical E total => same global projection.
    s1 = np.zeros(12, bool); s1[[0,1]] = True
    s2 = np.zeros(12, bool); s2[[7,9,10,11]] = True
    a.step(s1,None,.1);b.step(s2,None,.1)
    assert a.global_synaptic_rate == b.global_synaptic_rate == 2/10*1000/15
    before = a.global_synaptic_rate
    for _ in range(1500):a.step(np.zeros(12,bool),None,.1)
    assert np.isclose(a.global_synaptic_rate/before,np.exp(-10),rtol=1e-11)
    b.z[:10] = .7; ie=np.full(12,150.);ii=np.arange(12,dtype=float)+80
    result=b.apply_currents(ie,ii)
    expected=ii.copy();expected[:10]=5/6*ii[:10]+b.global_synaptic_rate
    assert np.allclose(b.delivered,expected,rtol=0,atol=2e-14)
    assert np.array_equal(result[10:],ie[10:]-ii[10:])
    assert b.global_current>0 and b.global_synaptic_rate<50
    oldz=b.z.copy();b.step(np.zeros(12,bool),None,.1)
    assert np.array_equal(b.z[:10],oldz[:10]+.1/5000*((expected[:10]<95.2)-oldz[:10]))
    # Uniform output is invariant to population size at constant active fraction.
    c=SpatialFeedbackSlow(24,18,cfg,NE=20,spatial_gamma=1/6,current_per_Hz=6.)
    s=np.zeros(24,bool);s[:4]=True;c.step(s,None,.1)
    assert c.global_synaptic_rate==before
    return dict(uniform_projection_spatial_permutation_invariant=True,
        E_only_and_population_size_normalized=True, fast_synapse_150ms_tail=float(np.exp(-10)),
        no50Hz_threshold=True, same_delivered_current_drives_native_Z=True, I_cells_unchanged=True)


def verify():
    checks=equation_checks()
    import analyze_topic4_autonomous_recovery as obs
    keys=['raster','spikes_1ms','regions_1ms','field_5ms','Z','M','currents','inputs','lfp_raw']
    zero=obs.load(OUT/'runs/qa_zero',keys)
    ref=obs.load(PREVIOUS/'runs/qa_s9108401',keys)
    for k in keys:assert np.array_equal(zero[k],ref[k]),k
    a=obs.load(OUT/'runs/qa_spatial_full',keys)
    b=obs.load(OUT/'runs/qa_spatial_resume',keys)
    for k in keys:assert np.array_equal(a[k],b[k]),k
    x=carrier.base.load_pickle(OUT/'runs/qa_spatial_full/checkpoint.pkl')['engine']
    y=carrier.base.load_pickle(OUT/'runs/qa_spatial_resume/checkpoint.pkl')['engine']
    def equal(a,b,path='engine'):
        assert type(a) is type(b),path
        if isinstance(a,np.ndarray):assert np.array_equal(a,b,equal_nan=True),path
        elif isinstance(a,dict):
            assert a.keys()==b.keys(),path
            for k in a:equal(a[k],b[k],path+'.'+str(k))
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b),path
            for i,(v,w) in enumerate(zip(a,b)):equal(v,w,path+'.'+str(i))
        else:assert a==b,path
    equal(x,y)
    carrier.base.write(OUT/'qa.json',dict(status='PASS',native_zero_gamma_bitwise=True,
        resumed_spatial_observations_and_entire_engine_bitwise=True,**checks))


def qa():
    prepare();equation_checks()
    for names in [['qa_zero','qa_spatial_full','qa_spatial_resume'],['qa_zero','qa_spatial_resume']]:
        children=[]
        for name in names:
            folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
            with (folder/'worker.log').open('a') as log:
                children.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--name',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT))
        for child in children:
            if child.wait():raise RuntimeError('Short native spatial feedback QA failed')
    verify()


def supervise():
    p=prepare();assert carrier.base.read(OUT/'qa.json')['status']=='PASS'
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending=[j['name'] for j in p['initial_jobs'] if not (OUT/'runs'/j['name']/'result.json').exists()]
    children={};failed=[]
    while pending or children:
        for name,child in list(children.items()):
            if child.poll() is None:continue
            del children[name]
            if child.returncode or not (OUT/'runs'/name/'result.json').exists():failed.append(name)
        total=0
        for proc in psutil.process_iter(['cmdline','status']):
            try:
                cmd=proc.info['cmdline'] or []
                if 'worker' in cmd and any('topic4' in s for s in cmd) and proc.info['status']!=psutil.STATUS_ZOMBIE:total+=1
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        available=psutil.virtual_memory().available/2**30
        free=shutil.disk_usage(OUT).free/2**30
        while pending and not failed and len(children)<p['max_workers'] and total<p['maximum_combined_workers'] and available>=80 and free>=40 and time.time()<p['deadline_epoch']-1800:
            name=pending.pop(0);folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
            with (folder/'worker.log').open('a') as log:
                children[name]=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--name',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            total+=1;available-=4;free-=1
        completed=sum((OUT/'runs'/j['name']/'result.json').exists() for j in p['initial_jobs'])
        carrier.base.write(OUT/'status.json',dict(status='RUNNING' if not failed else 'DRAINING_FAILURE',pid=os.getpid(),updated_at=time.time(),completed=completed,total=6,running={k:v.pid for k,v in children.items()},pending=pending,failed=failed,combined_workers=total,available_GiB=available))
        if not children and (failed or time.time()>=p['deadline_epoch']-1800):break
        time.sleep(15)
    if not failed and not pending:
        subprocess.run([sys.executable,str(ROOT/'scripts/analyze_topic4_liou_spatial_feedback.py')],cwd=ROOT,check=True)
    carrier.base.write(OUT/'status.json',dict(status='READY_FOR_REVIEW' if not failed and not pending else 'INCOMPLETE_REVIEW',updated_at=time.time(),completed=completed,total=6,running={},pending=pending,failed=failed))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','qa','verify','worker','supervise']);parser.add_argument('--name');args=parser.parse_args()
    try:
        if args.mode=='worker':worker(args.name)
        else:globals()[args.mode]()
    except Exception as exc:
        carrier.base.write(OUT/('runs/'+args.name+'/failure.json' if args.name else args.mode+'_failure.json'),dict(error=repr(exc),time=time.time()))
        raise
