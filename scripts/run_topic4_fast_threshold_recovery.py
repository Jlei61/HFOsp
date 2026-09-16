#!/usr/bin/env python3
"""Explicit fast threshold adaptation added to the unchanged round-one carrier.

This is a new mechanism, motivated by Liou2020 Eq2/spiking Methods. It does not
turn the current-based SNN into the paper's conductance-based spiking model.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import argparse,copy,json,time,subprocess,sys
from pathlib import Path
import numpy as np
import run_topic4_autonomous_recovery as carrier
import checkpoint

ROOT=carrier.ROOT;PARENT=carrier.OUT;OUT=PARENT/'fast_threshold_round2'
OriginalSlow=carrier.RecoverySlow

class FastThresholdSlow(OriginalSlow):
    def __init__(self,*args,delta_phi=0.,tau_phi_ms=100.,**kwargs):
        super().__init__(*args,**kwargs)
        self.delta_phi=float(delta_phi);self.tau_phi_ms=float(tau_phi_ms)
        assert self.delta_phi>=0 and self.tau_phi_ms>0
        self.phi=np.zeros(self.N,dtype=np.float64)
        self.phi_records=[]

    def threshold(self,base):
        if self.delta_phi==0:return base
        # Never mutate the fixed two-core threshold field.
        return np.asarray(base)+self.phi

    def step(self,spk,labels,dt):
        super().step(spk,labels,dt)
        if self.delta_phi>0:
            self.phi[:self.NE]*=np.exp(-float(dt)/self.tau_phi_ms)
            self.phi[:self.NE][spk[:self.NE]]+=self.delta_phi

    def apply_currents(self,*args,**kwargs):
        value=super().apply_currents(*args,**kwargs)
        if self._step_index%50==0:
            self.phi_records.append([self._step_index*.1,float(self.phi[:self.NE].mean()),
                float(self.phi[:self.NE].max())])
        return value

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return carrier.base.read(OUT/'protocol.json')
    p=copy.deepcopy(carrier.base.read(PARENT/'protocol.json'))
    p.update(status='PREPARED_NOT_DISPATCHED',round=2,parent=str(PARENT),created_at=time.time(),
        producer_sha256=carrier.base.sha(carrier.__file__),extension_producer_sha256=carrier.base.sha(__file__),
        new_mechanism='E-only fast threshold state phi: decay exp(-dt/tau_phi), jump delta_phi per spike; Vth_effective=Vth_static+phi. Static threshold/core geometry and I-cell thresholds untouched.',
        source='https://elifesciences.org/articles/50927 ; @SpikingModel/SpikingModel.m. Current-based LIF, explicit I and existing M retained, not a full conductance-model reproduction.',
        controls='delta_phi=0 reproduces the native carrier exactly; nonzero phi checkpoint resume verified against uninterrupted run.',
        dispatch_gate='No production runs until explicit round-one scientific review is written locally. QA is allowed now.',
        limitations='Fast phi may bound the high-rate plateau without restoring intermittent events. Do not move the high-state threshold to make conditions pass; inspect spatially broad activity below200Hz separately.')
    jobs=[]
    for gamma in [0,.25,.5]:
        for jump in [.25,1.,2.5]:
            if gamma==.5 and jump==.25:continue
            jobs.append(dict(name=f'phi{jump:g}_g{gamma:g}_s9108401',round=2,mode='native' if gamma==0 else 'mix',
                gamma=gamma,eta_m=.005,tau_M_s=2.,tau_Z_s=5.,threshold=carrier.base.old.THRESHOLD,
                seed=9108401,horizon_s=60.,device=len(jobs)%2,checkpoint_s=10.,delta_phi=jump,tau_phi_ms=100.))
    p['initial_jobs']=jobs
    carrier.base.write(OUT/'protocol.json',p)
    for job in jobs:carrier.base.write(OUT/'jobs'/(job['name']+'.json'),job)
    for name,jump,gamma,pause in [('qa_zero',0.,0.,True),('qa_phi_resume',1.,.25,True),('qa_phi_full',1.,.25,False)]:
        job=copy.deepcopy(jobs[0]);job.update(name=name,delta_phi=jump,gamma=gamma,mode='native' if gamma==0 else 'mix',
            eta_m=.0005 if jump==0 else .005,tau_M_s=1. if jump==0 else 2.,horizon_s=1.,checkpoint_s=.5 if pause else 1.,qa=pause)
        carrier.base.write(OUT/'jobs'/(name+'.json'),job)
    return p

def worker(name):
    p=prepare();assert carrier.base.sha(__file__)==p['extension_producer_sha256']
    job=carrier.base.read(OUT/'jobs'/(name+'.json'));folder=OUT/'runs'/name
    capture0=checkpoint.capture;restore0=checkpoint.restore_slow
    slow_ref=[]
    def factory(*args,**kwargs):
        obj=FastThresholdSlow(*args,delta_phi=job['delta_phi'],tau_phi_ms=job['tau_phi_ms'],**kwargs)
        slow_ref.append(obj);return obj
    def capture(**kwargs):
        state=capture0(**kwargs);obj=kwargs['slow'];state['slow_phi']=obj.phi.copy()
        records=np.asarray(obj.phi_records)
        if len(records):
            dest=folder/'phi_chunks';dest.mkdir(parents=True,exist_ok=True)
            first=round(records[0,0]*10);last=int(kwargs['step']);path=dest/f'{first:010d}_{last:010d}.npz'
            tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,time_ms=records[:,0],phi_mean=records[:,1],phi_max=records[:,2]);tmp.replace(path)
            obj.phi_records.clear()
        return state
    def restore(state,obj):
        restore0(state,obj);phi=state['slow_phi'];assert phi.shape==(obj.N,)
        obj.phi[:]=phi
    old_out,old_prepare,old_class=carrier.OUT,carrier.prepare,carrier.RecoverySlow
    carrier.OUT=OUT;carrier.prepare=lambda:p;carrier.RecoverySlow=factory
    checkpoint.capture=capture;checkpoint.restore_slow=restore
    try:
        carrier.worker(name)
        if slow_ref:
            assert np.all(slow_ref[0].phi[32000:]==0)
    finally:
        carrier.OUT=old_out;carrier.prepare=old_prepare;carrier.RecoverySlow=old_class
        checkpoint.capture=capture0;checkpoint.restore_slow=restore0

def verify():
    p=prepare();rng=np.random.default_rng(316)
    cfg=carrier.base.old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000,I_th_EI=95.2,tau_adp=2000,eta_m=.005)
    obj=FastThresholdSlow(12,18,cfg,NE=10,delta_phi=1.,tau_phi_ms=100.)
    static=np.linspace(12,18,12);copy_static=static.copy()
    for k in range(100):
        sp=rng.random(12)<.15;before=obj.phi.copy();obj.apply_currents(np.ones(12)*100,np.ones(12)*110);obj.step(sp,None,.1)
        expected=before.copy();expected[:10]*=np.exp(-.1/100);expected[:10][sp[:10]]+=1.
        assert np.array_equal(obj.phi,expected)
        assert np.array_equal(obj.threshold(static),static+expected) and np.array_equal(static,copy_static)
        assert np.all(obj.phi[10:]==0)
    def load(folder,keys):
        data={k:[] for k in keys}
        for path in sorted((folder/'chunks').glob('*.npz')):
            if '.tmp.' in path.name:continue
            with np.load(path) as a:
                for k in keys:data[k].append(a[k])
        return {k:np.concatenate(v) for k,v in data.items()}
    keys=['raster','spikes_1ms','regions_1ms','field_5ms','Z','M','currents','inputs','lfp_raw']
    a=load(OUT/'runs/qa_zero',keys);b=load(PARENT/'runs/qa_s9108401',keys)
    for k in keys:assert np.array_equal(a[k],b[k]),k
    a=load(OUT/'runs/qa_phi_resume',keys);b=load(OUT/'runs/qa_phi_full',keys)
    for k in keys:assert np.array_equal(a[k],b[k]),k
    x=carrier.base.load_pickle(OUT/'runs/qa_phi_resume/checkpoint.pkl')['engine']
    y=carrier.base.load_pickle(OUT/'runs/qa_phi_full/checkpoint.pkl')['engine']
    for k in ['V','ref','s_E','I_E','s_I','I_I','ring_sE','ring_sI','slow_phi']:assert np.array_equal(x[k],y[k]),k
    assert x['rng_state']==y['rng_state']
    carrier.base.write(OUT/'qa.json',dict(status='PASS',zero_phi_native_bitwise=True,nonzero_phi_continuation_bitwise=True,
        fast_state_delay_history_noise_carried=True,phi_equation_and_static_threshold_immutability=True,actual_phi_max=float(x['slow_phi'].max())))

def qa():
    prepare()
    children=[]
    for name in ['qa_zero','qa_phi_resume','qa_phi_full']:
        folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'worker.log').open('a') as log:
            children.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--name',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT))
    for c in children:
        if c.wait():raise RuntimeError('Extension QA worker failed')
    children=[]
    for name in ['qa_zero','qa_phi_resume']:
        with (OUT/'runs'/name/'worker.log').open('a') as log:
            children.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--name',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT))
    for c in children:
        if c.wait():raise RuntimeError('Extension QA resume failed')
    verify()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','worker','qa','verify']);ap.add_argument('--name');args=ap.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='worker':worker(args.name)
        elif args.mode=='qa':qa()
        else:verify()
    except Exception as e:
        carrier.base.write(OUT/('runs/'+args.name+'/failure.json' if args.name else args.mode+'_failure.json'),dict(error=repr(e),time=time.time()));raise
