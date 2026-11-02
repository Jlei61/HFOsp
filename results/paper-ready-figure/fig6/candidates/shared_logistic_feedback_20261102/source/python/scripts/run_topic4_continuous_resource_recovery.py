#!/usr/bin/env python3
"""Explicit Z-equation hypothesis: recovery also operates during strong input.

This is NOT Liou's generalized Equation 8, nor an ionic chloride model.
With b=1[J<Ith], tauZ dz=b-z+rho*(1-b)*(1-z). rho=0 is the
unchanged native equation. There is no clock-, episode- or detector-driven term.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import argparse,copy,subprocess,sys,time
from pathlib import Path
import numpy as np
import run_topic4_activity_global_pool as pool
import checkpoint

carrier=pool.carrier;ROOT=pool.ROOT;PARENT=pool.PARENT
OUT=PARENT/'continuous_resource_recovery_round5'
OriginalPool=pool.GlobalPoolSlow

class ResourceRecoverySlow(OriginalPool):
    def __init__(self,*args,recovery_ratio=0.,**kwargs):
        super().__init__(*args,**kwargs)
        self.recovery_ratio=float(recovery_ratio)
        assert self.recovery_ratio>=0
        self.recovery_records=[]

    def apply_currents(self,ie,ii,labels=None,rec=None):
        value=super().apply_currents(ie,ii,labels,rec)
        if self._step_index%200==0:
            b=(self._I_I_last[:self.NE]<self.cfg.I_th_EI)
            z=self.z[:self.NE];scale=1000/self.cfg.tau_z
            native=(b-z)*scale
            extra=self.recovery_ratio*(~b)*(1-z)*scale
            self.recovery_records.append([self._step_index*.1,native.mean(),extra.mean(),(native+extra).mean()])
        return value

    def step(self,spk,labels,dt):
        if self.recovery_ratio==0 or not self.cfg.use_z:
            return super().step(spk,labels,dt)
        assert 0<float(dt)/self.cfg.tau_z*(1+self.recovery_ratio)<=1
        # Use pre-step Z for BOTH terms of the same explicit Euler update.
        extra=(float(dt)/self.cfg.tau_z*self.recovery_ratio)*(
            self._I_I_last[:self.NE]>=self.cfg.I_th_EI)*(1-self.z[:self.NE])
        super().step(spk,labels,dt)
        self.z[:self.NE]+=extra
        assert np.min(self.z[:self.NE])>=0 and np.max(self.z[:self.NE])<=1+1e-14

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return carrier.base.read(OUT/'protocol.json')
    p=copy.deepcopy(carrier.base.read(pool.OUT/'protocol.json'))
    p.update(round=5,status='PREPARED_PENDING_QA',parent=str(pool.OUT),created_at=time.time(),
        recovery_producer_sha256=carrier.base.sha(__file__),
        equation='b_i=1[J_i<Ith]; tauZ*dz_i/dt=b_i-z_i+rho*(1-b_i)*(1-z_i). Native Euler; rho=0 uses exact unchanged parent code. M and all fast/noise states unchanged.',
        scientific_difference='Explicit new phenomenological recovery term under high inhibitory input. Motivated by concurrent clearance in Liou chloride Equation3, but NOT an ionic derivation and NOT the published generalized Equation8. No claim that native Z is erroneous or that this alteration is necessary.',
        strong_input_equilibrium='z*=rho/(1+rho); effective relaxation tauZ/(1+rho). This is an equilibrium of the revised equation, not a state clamp. At z=1 the extra term is zero; at weak input all rho share native recovery.',
        question='Does persistent loss of all Z under above-threshold global feedback prevent autonomous termination in this current-LIF carrier? Contrast preserved-resource dynamics with completed/ongoing rho0 controls.',
        controls='Native moderate-M gamma0 old0-60s and round3 kappa50/moderateM both rho0; new rho1 kappa0 versus50 plus intermediate rho.25 kappa50. No timed reset, kick, clamp, event-specific input, or modified detector.',
        dose_caution='Changing rho changes both high-input equilibrium and relaxation, and may prevent entry. Such a negative is not recovery. Success requires actual high-return-high and independent noise; no parameter is fitted to patient data here.')
    template=copy.deepcopy(next(j for j in p['initial_jobs'] if j['name']=='pool_k50_r50_tau2_mmoderate_s9108401'))
    jobs=[]
    for gain,rho in [(50.,1.),(50.,.25),(0.,1.)]:
        j=copy.deepcopy(template);j.update(name=f'resource_rho{rho:g}_k{gain:g}_s9108401',round=5,
            recovery_ratio=rho,pool_gain=gain,device=len(jobs)%2,horizon_s=60.)
        jobs.append(j)
    p['initial_jobs']=jobs;carrier.base.write(OUT/'protocol.json',p)
    for j in jobs:carrier.base.write(OUT/'jobs'/(j['name']+'.json'),j)
    qa0=carrier.base.read(pool.OUT/'jobs/qa_pool_full.json')
    for name,rho,split in [('qa_zero',0.,False),('qa_full',1.,False),('qa_resume',1.,True)]:
        j=copy.deepcopy(qa0);j.update(name=name,recovery_ratio=rho,qa=split,checkpoint_s=.5 if split else 1.)
        carrier.base.write(OUT/'jobs'/(name+'.json'),j)
    return p

def worker(name):
    p=prepare();assert carrier.base.sha(__file__)==p['recovery_producer_sha256']
    j=carrier.base.read(OUT/'jobs'/(name+'.json'));folder=OUT/'runs'/name
    orig_out,orig_prepare,orig_class=pool.OUT,pool.prepare,pool.GlobalPoolSlow
    capture0=checkpoint.capture
    def factory(*args,**kwargs):return ResourceRecoverySlow(*args,**kwargs,recovery_ratio=j['recovery_ratio'])
    def capture(**kwargs):
        state=capture0(**kwargs);obj=kwargs['slow'];a=np.asarray(obj.recovery_records)
        if len(a):
            d=folder/'resource_chunks';d.mkdir(parents=True,exist_ok=True)
            path=d/f'{round(a[0,0]*10):010d}_{int(kwargs["step"]):010d}.npz';tmp=path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,time_ms=a[:,0],native_derivative_per_s=a[:,1],extra_recovery_per_s=a[:,2],net_derivative_per_s=a[:,3]);tmp.replace(path);obj.recovery_records.clear()
        return state
    pool.OUT=OUT;pool.prepare=lambda:p;pool.GlobalPoolSlow=factory;checkpoint.capture=capture
    try:pool.worker(name)
    finally:pool.OUT=orig_out;pool.prepare=orig_prepare;pool.GlobalPoolSlow=orig_class;checkpoint.capture=capture0

def verify():
    cfg=carrier.base.old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000,I_th_EI=95.2,tau_adp=1000,eta_m=.0005)
    for rho in [0.,.25,1.]:
        obj=ResourceRecoverySlow(4,18,cfg,NE=3,recovery_ratio=rho,pool_gain=0)
        obj.z[:3]=[0,.4,1];old=obj.z.copy();ii=np.array([100.,0,95.2,100.]);sp=np.array([True,False,True,True]);obj.apply_currents(np.ones(4)*100,ii)
        obj.step(sp,None,.1);b=ii[:3]<95.2
        expected=old[:3]+.1/5000*(b-old[:3]+rho*(~b)*(1-old[:3]))
        assert np.allclose(obj.z[:3],expected,rtol=0,atol=1e-16)
        assert obj.z[3]==1 and obj.m[3]==0 and np.array_equal(obj.m[:3],[1,0,1])
        zstar=rho/(1+rho);z=1.;factor=1-.1/5000*(1+rho)
        for _ in range(100000):z+=.1/5000*(-z+rho*(1-z))
        assert abs(z-(zstar+(1-zstar)*factor**100000))<1e-12
    def equal(a,b,path='root'):
        assert type(a) is type(b),path
        if isinstance(a,np.ndarray):assert np.array_equal(a,b,equal_nan=True),path
        elif isinstance(a,dict):
            assert a.keys()==b.keys(),path
            for k in a:equal(a[k],b[k],path+'.'+str(k))
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b),path
            for i,(v,w) in enumerate(zip(a,b)):equal(v,w,path+'.'+str(i))
        else:assert a==b,path
    keys=['raster','spikes_1ms','regions_1ms','field_5ms','Z','M','currents','regional_currents','inputs','lfp_raw']
    def load(folder):
        parts={k:[] for k in keys}
        for f in sorted((folder/'chunks').glob('*.npz')):
            if '.tmp.' in f.name:continue
            with np.load(f) as a:
                for k in keys:parts[k].append(a[k])
        return {k:np.concatenate(v) for k,v in parts.items()}
    equal(load(OUT/'runs/qa_zero'),load(pool.OUT/'runs/qa_pool_full'))
    equal(load(OUT/'runs/qa_full'),load(OUT/'runs/qa_resume'))
    for lhs,rhs in [(OUT/'runs/qa_zero',pool.OUT/'runs/qa_pool_full'),(OUT/'runs/qa_full',OUT/'runs/qa_resume')]:
        a=carrier.base.load_pickle(lhs/'checkpoint.pkl')['engine'];b=carrier.base.load_pickle(rhs/'checkpoint.pkl')['engine']
        if lhs.name=='qa_zero':
            assert a['slow']['kind']=='ResourceRecoverySlow' and b['slow']['kind']=='GlobalPoolSlow'
            # Only class-name metadata differs; every numerical state and RNG is tested.
            a['slow']['kind']=b['slow']['kind']
        equal(a,b)
    carrier.base.write(OUT/'qa.json',dict(status='PASS',zero_rho_all_observations_and_numerical_checkpoint_bitwise=True,
        zero_rho_only_metadata_difference='slow.kind: ResourceRecoverySlow versus GlobalPoolSlow; explicitly checked, not a physical state difference.',
        nonzero_rho_resume_all_observations_and_full_checkpoint_bitwise=True,explicit_Euler_equation_verified=True,
        constant_input_analytic_relaxation_verified=True,I_Z1_M0_preserved=True,no_external_intervention=True))

def qa():
    prepare()
    for name in ['qa_zero','qa_full','qa_resume','qa_resume']:
        folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'worker.log').open('a') as log:
            r=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--name',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'QA failed: {name}')
    verify()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','qa','verify','worker']);ap.add_argument('--name');ap.add_argument('--producer-script');args=ap.parse_args()
    try:
        if args.mode=='worker':worker(args.name)
        else:globals()[args.mode]()
    except Exception as e:
        carrier.base.write(OUT/('runs/'+args.name+'/failure.json' if args.name else args.mode+'_failure.json'),dict(error=repr(e),time=time.time()));raise
