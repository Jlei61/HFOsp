#!/usr/bin/env python3
"""New autonomous global inhibitory feedback; no event labels or timed actions.

The population spike filter is an explicit added state. This tests an effective
global recruitment motif, not a claim that the original Z/M already contained it.
Global current remains Z-depleted and contributes to the same depletion drive.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import argparse,copy,subprocess,sys,time
from pathlib import Path
import numpy as np
import run_topic4_autonomous_recovery as carrier
import checkpoint
ROOT=carrier.ROOT;PARENT=carrier.OUT;OUT=PARENT/'activity_global_pool_round3'
OriginalSlow=carrier.RecoverySlow

class GlobalPoolSlow(OriginalSlow):
    def __init__(self,*args,pool_gain=0.,pool_threshold_Hz=50.,pool_tau_s=2.,**kwargs):
        super().__init__(*args,**kwargs)
        self.pool_gain=float(pool_gain);self.pool_threshold_Hz=float(pool_threshold_Hz);self.pool_tau_s=float(pool_tau_s)
        assert self.pool_gain>=0 and self.pool_threshold_Hz>=0 and self.pool_tau_s>0
        self.pool_rate=0.;self.pool_current=0.;self.pool_records=[]
    def apply_currents(self,ie,ii,labels=None,rec=None):
        self.raw_mean=float(ii[:self.NE].mean())
        self.pool_current=self.pool_gain*max(self.pool_rate-self.pool_threshold_Hz,0.)
        if self.pool_current==0:self.delivered=ii
        else:
            self.delivered=ii.copy();self.delivered[:self.NE]+=self.pool_current
        value=carrier.base.old.MZSlowVars.apply_currents(self,ie,self.delivered,labels,rec)
        if self._step_index%50==0:
            self.pool_records.append([self._step_index*.1,self.pool_rate,self.pool_current,
                self.pool_current*float(self.z[:self.NE].mean()),self.raw_mean])
        return value
    def step(self,spk,labels,dt):
        super().step(spk,labels,dt)
        self.pool_rate=self.pool_rate*np.exp(-float(dt)/(self.pool_tau_s*1000))+np.count_nonzero(spk[:self.NE])/(self.NE*self.pool_tau_s)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return carrier.base.read(OUT/'protocol.json')
    p=copy.deepcopy(carrier.base.read(PARENT/'protocol.json'))
    p.update(status='PREPARED_NOT_DISPATCHED',round=3,parent=str(PARENT),created_at=time.time(),
        producer_sha256=carrier.base.sha(carrier.__file__),extension_producer_sha256=carrier.base.sha(__file__),
        new_mechanism='Added global causal spike-rate filter Rg, tauG dRg/dt=-Rg+population spike train. Jg=kappa*max(Rg-r0,0); delivered J_i=rawII_i+Jg for E. Membrane uses Zi*J_i and native Zi depletion is driven by the same J_i. I cells unchanged.',
        source='Liou2020 activity-driven global inhibitory recruitment, and historical A1c global EMA feedback. Rectified slow pool is an explicit new effective-circuit hypothesis, not an exact Liou implementation or preexisting native Z/M mechanism.',
        controls='kappa=0 is bitwise native baseline. r0=0 versus50Hz at identical gain/tau separates broad tonic suppression from thresholded global recruitment. r0 is a fixed model parameter; no online event classifier, Z/M reset, time schedule or episode-dependent intervention.',
        units='Rg Hz; kappa mV-equivalent/Hz; r0 Hz; tauG seconds. Local M eta0.0005/tau1s and Z tau5s/Ith95.1985 retain the current baseline.',
        dispatch_gate='Prepare/QA only. No production until a written scientific review compares saved current first/second-round evidence and names the final bounded jobs.',
        limits='Thresholded recruitment may stabilize a moderate plateau or suppress onset instead of generating recovery; none counts as an autonomous high-return-high cycle. Global term is not protected from Z depletion; diagnostic current column4 of inherited chunks remains gamma*rawII=0, actual added current is in pool_chunks.')
    jobs=[]
    for tau in [2.,.5]:
        for gain in [10.,5.]:
            for threshold in [50.,0.]:
                jobs.append(dict(name=f'pool_k{gain:g}_r{threshold:g}_tau{tau:g}_s9108401',round=3,
                    mode='native',gamma=0.,eta_m=.0005,tau_M_s=1.,tau_Z_s=5.,threshold=carrier.base.old.THRESHOLD,
                    seed=9108401,horizon_s=60.,device=len(jobs)%2,checkpoint_s=10.,
                    pool_gain=gain,pool_threshold_Hz=threshold,pool_tau_s=tau))
    p['initial_jobs']=jobs;carrier.base.write(OUT/'protocol.json',p)
    for job in jobs:carrier.base.write(OUT/'jobs'/(job['name']+'.json'),job)
    for name,gain,pause in [('qa_zero',0.,True),('qa_pool_resume',10.,True),('qa_pool_full',10.,False)]:
        j=copy.deepcopy(jobs[0]);j.update(name=name,pool_gain=gain,pool_threshold_Hz=0.,pool_tau_s=.5,
            horizon_s=1.,checkpoint_s=.5 if pause else 1.,qa=pause)
        carrier.base.write(OUT/'jobs'/(name+'.json'),j)
    return p

def worker(name):
    p=prepare();assert carrier.base.sha(__file__)==p['extension_producer_sha256']
    job=carrier.base.read(OUT/'jobs'/(name+'.json'));folder=OUT/'runs'/name
    capture0,restore0=checkpoint.capture,checkpoint.restore_slow
    def factory(*args,**kwargs):return GlobalPoolSlow(*args,**kwargs,pool_gain=job['pool_gain'],pool_threshold_Hz=job['pool_threshold_Hz'],pool_tau_s=job['pool_tau_s'])
    def capture(**kwargs):
        state=capture0(**kwargs);obj=kwargs['slow'];state['slow_global_pool_rate']=obj.pool_rate
        records=np.asarray(obj.pool_records)
        if len(records):
            d=folder/'pool_chunks';d.mkdir(parents=True,exist_ok=True)
            path=d/f'{round(records[0,0]*10):010d}_{int(kwargs["step"]):010d}.npz';tmp=path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,time_ms=records[:,0],rate_Hz=records[:,1],raw_global_current=records[:,2],effective_global_current=records[:,3],raw_local_II_mean=records[:,4]);tmp.replace(path);obj.pool_records.clear()
        return state
    def restore(state,obj):restore0(state,obj);obj.pool_rate=float(state['slow_global_pool_rate'])
    old_out,old_prepare,old_class=carrier.OUT,carrier.prepare,carrier.RecoverySlow
    carrier.OUT=OUT;carrier.prepare=lambda:p;carrier.RecoverySlow=factory;checkpoint.capture=capture;checkpoint.restore_slow=restore
    try:carrier.worker(name)
    finally:
        carrier.OUT=old_out;carrier.prepare=old_prepare;carrier.RecoverySlow=old_class;checkpoint.capture=capture0;checkpoint.restore_slow=restore0

def verify():
    prepare()
    cfg=carrier.base.old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000,I_th_EI=95.2,tau_adp=1000,eta_m=.0005)
    obj=GlobalPoolSlow(12,18,cfg,NE=10,pool_gain=10,pool_threshold_Hz=2,pool_tau_s=.5)
    obj.pool_rate=5;obj.z[:10]=.7
    ie=np.arange(12)+100.;ii=np.arange(12)+80.;v=obj.apply_currents(ie,ii)
    assert np.array_equal(obj.delivered[:10],ii[:10]+30)
    assert np.array_equal(v[:10],ie[:10]-.7*(ii[:10]+30))
    assert np.array_equal(v[10:],ie[10:]-ii[10:])
    assert np.array_equal(obj._I_I_last,obj.delivered)
    sp=np.arange(12)%2==0;before=obj.pool_rate;obj.step(sp,None,.1)
    assert obj.pool_rate==before*np.exp(-.1/500)+5/(10*.5)
    def load(folder,keys):
        data={k:[] for k in keys}
        for path in sorted((folder/'chunks').glob('*.npz')):
            if '.tmp.' in path.name:continue
            with np.load(path) as a:
                for k in keys:data[k].append(a[k])
        return {k:np.concatenate(v) for k,v in data.items()}
    keys=['raster','spikes_1ms','regions_1ms','field_5ms','Z','M','currents','regional_currents','inputs','lfp_raw']
    a=load(OUT/'runs/qa_zero',keys);b=load(PARENT/'runs/qa_s9108401',keys)
    for k in keys:assert np.array_equal(a[k],b[k]),k
    a=load(OUT/'runs/qa_pool_resume',keys);b=load(OUT/'runs/qa_pool_full',keys)
    for k in keys:assert np.array_equal(a[k],b[k]),k
    x=carrier.base.load_pickle(OUT/'runs/qa_pool_resume/checkpoint.pkl')['engine'];y=carrier.base.load_pickle(OUT/'runs/qa_pool_full/checkpoint.pkl')['engine']
    def equal(a,b,path='checkpoint'):
        assert type(a) is type(b),path
        if isinstance(a,np.ndarray):assert np.array_equal(a,b,equal_nan=True),path
        elif isinstance(a,dict):
            assert a.keys()==b.keys(),path
            for k in a:equal(a[k],b[k],path+'.'+str(k))
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b),path
            for i,(v,w) in enumerate(zip(a,b)):equal(v,w,path+'.'+str(i))
        else:assert a==b,path
    equal(x,y)
    carrier.base.write(OUT/'qa.json',dict(status='PASS',zero_gain_native_bitwise=True,entire_checkpoint_resume_bitwise=True,
        added_global_inhibition_also_Z_depleted=True,same_total_current_used_for_Z_depletion=True,
        pool_is_causal_spike_filter=True,no_clock_or_event_label_in_physics=True,pool_rate_Hz=float(x['slow_global_pool_rate'])))

def qa():
    prepare()
    for names in [['qa_zero','qa_pool_resume','qa_pool_full'],['qa_zero','qa_pool_resume']]:
        children=[]
        for n in names:
            folder=OUT/'runs'/n;folder.mkdir(parents=True,exist_ok=True)
            with (folder/'worker.log').open('a') as log:
                children.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--name',n],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT))
        for c in children:
            if c.wait():raise RuntimeError('Global-pool QA worker failed')
    verify()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','qa','worker','verify']);ap.add_argument('--name');args=ap.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='qa':qa()
        elif args.mode=='verify':verify()
        else:worker(args.name)
    except Exception as e:
        carrier.base.write(OUT/('runs/'+args.name+'/failure.json' if args.name else args.mode+'_failure.json'),dict(error=repr(e),time=time.time()));raise
