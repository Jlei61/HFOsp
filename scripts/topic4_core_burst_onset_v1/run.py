"""Bounded native noise-source/finite-perturbation assay; no slow variables."""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import argparse, inspect, json, sys, time, types, subprocess
import numpy as np
from scipy.special import ndtr

ROOT=Path('/home/honglab/leijiaxin/HFOsp')
sys.path.insert(0,str(ROOT/'scripts/topic4_burst_regime'))
import runtime as native
OUT=ROOT/'results/topic4_sef_hfo/core_burst_onset_brunel_v1_20260915'
OLD=native.OUT
SWITCH=6000.
PROBE=12000.

class Drive:
    def __init__(self,sub,core,seed,switch):
        self.signal=float(sub.params.nu_ext_ratio*native.compute_nu_theta(sub.params)[0])
        self.ou=native.CoreOUMixture(core,self.signal,.95,0.,sub.params.dt,sub.params.tau_n,sub.params.sigma_n,seed)
        mu=.95*self.signal;sd=self.ou.stationary_std
        self.matched=mu if sd==0 else mu*ndtr(mu/sd)+sd*np.exp(-.5*(mu/sd)**2)/np.sqrt(2*np.pi)
        self.constant=np.where(core>=0,self.matched-self.signal,0.)
        self.switch=switch
    def step(self,t):
        v=self.ou.step(t)
        return v if t<self.switch else self.constant

class Observer(native.Observer):
    def __init__(self,sub,groups,duration,path):
        super().__init__(sub,groups,duration,path)
        self.i_groups=[np.asarray(groups['coreAI']),np.asarray(groups['coreBI'])]
        self.i_count=np.zeros(2,np.uint32)
        self.i_counts=np.zeros((len(self.counts),2),np.uint32)
        self.aux=np.zeros((len(self.counts),6),np.float64)
        self.external_counts=np.zeros((len(self.counts),2),np.float64)
        self.external_accum=np.zeros(2,np.float64)
        self.rate_modulation=np.zeros((len(self.counts),2),np.float64)
        self.exact_t=[];self.exact_i=[]
    def observe(self,t,tm,xi,nu,ext,delta,V,I_E,I_I,spk):
        active=self.sample[spk[self.sample]]
        self.exact_t.extend([tm+self.dt]*len(active));self.exact_i.extend(active.tolist())
        for j,idx in enumerate(self.i_groups): self.i_count[j]+=np.count_nonzero(spk[idx])
        for j,key in enumerate(('coreAE','coreBE')): self.external_accum[j]+=ext[self.groups[key]].sum()
        super().observe(t,tm,xi,nu,ext,delta,V,I_E,I_I,spk)
        if (t+1)%self.stride==0:
            frame=(t+1)//self.stride-1
            self.i_counts[frame]=self.i_count;self.i_count.fill(0)
            self.external_counts[frame]=self.external_accum;self.external_accum.fill(0)
            for j,key in enumerate(('coreAE','coreBE')):
                idx=self.groups[key]
                self.aux[frame,3*j:3*j+3]=[I_E[idx].mean(),I_I[idx].mean(),V[idx].mean()]
                self.rate_modulation[frame,j]=delta[idx].mean()
    def arrays(self):
        a=super().arrays();n=self.nsteps//self.stride
        a.update(exact_spike_time_ms=np.asarray(self.exact_t,np.float64),exact_spike_cell=np.asarray(self.exact_i,np.int32),
            core_i_counts_2ms=self.i_counts[:n],core_i_sizes=np.array([len(x) for x in self.i_groups]),
            external_counts_2ms=self.external_counts[:n],core_rate_modulation_2ms=self.rate_modulation[:n],
            core_current_voltage_2ms=self.aux[:n],core_current_voltage_columns=np.array(['A_I_E','A_I_I','A_V','B_I_E','B_I_I','B_V']))
        return a

def engine(arm,switch):
    # Preserve the independently validated ordered scatter and add only a timed
    # external-count clamp. No voltage, synaptic, graph or delay equation changes.
    s=inspect.getsource(native.ORIGINAL)
    start=s.index('            if spE.size:\n',s.index('            spE = np.where'))
    end=s.index('\n        if verbose and',start)
    s=s[:start]+'''            if ee_std_on:
                raise RuntimeError("Plasticity must be off")
            if spE.size:
                ordered_scatter(ring_sE, spE, a_indptr, a_dst, a_dly, a_w, _abs_step)
            if spI.size:
                ordered_scatter(ring_sI, spI, g_indptr, g_dst, g_dly, g_w, _abs_step)
'''+s[end:]
    target='            ext[_stochastic_idx] = rng.poisson(ext[_stochastic_idx]).astype(np.float64)'
    assert s.count(target)==1
    s=s.replace(target,'            if not (_all_off and tm >= _switch_ms):\n    '+target)
    namespace=dict(native.ORIGINAL.__globals__,ordered_scatter=native.ordered_scatter,_all_off=arm=='all_off_probe',_switch_ms=switch)
    exec(compile(s,'<core_burst_onset_v1>', 'exec'),namespace)
    return namespace[native.ORIGINAL.__name__]

def simulate(ee,seed,arm,duration=20000.,switch=SWITCH,tag=None):
    name=tag or f'ee{ee:g}_s{seed}_{arm}'
    folder=OUT/'per_run'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists(): return
    start=time.time();native.write(folder/'progress.json',dict(status='BUILDING'))
    sub,groups,loading,det,applied,cores=native.setup(ee,1.,seed)
    original=json.loads((OLD/'per_run'/f'ee{ee:g}_d1_n1_t2511_s{seed}'/'applied_physics.json').read_text())
    assert applied['identity']==original['identity'],'changed baseline physics'
    sub.params.T=duration;sub.net['rng']=np.random.default_rng(seed)
    drive=Drive(sub,cores[0],seed,switch)
    observer=Observer(sub,groups,duration,folder/'progress.json')
    f=engine(arm,switch);proxy=native.NumpyProxy((round(duration/sub.params.dt),sub.n_e))
    f=types.FunctionType(f.__code__,dict(f.__globals__,np=proxy),f.__name__,f.__defaults__,f.__closure__)
    kwargs=dict(KICK_BOOST=0.,t_kick=1e9,V_th_per_neuron=sub.vtheta,slow=None,
        early_stop_runaway=True,es_thresh_hz=120.,es_dur_ms=100.,post_runaway_record_ms=500.,
        global_ou_loading=np.zeros_like(loading),deterministic_external_mask=det,
        external_e_rate_drive=drive,step_observer=observer)
    if arm=='all_off_probe' and duration>PROBE:
        mask=np.zeros(sub.n_e+sub.n_i,bool);mask[groups['coreAE']]=True
        kwargs.update(forced_spike_mask=mask,forced_spike_ms=PROBE)
    result=f(sub.params,sub.net,**kwargs)
    assert proxy.hits==1
    arrays=observer.arrays()
    arrays.update(positions_E=sub.positions_e,positions_I=sub.positions_i,vtheta=sub.vtheta,
        core_index_E=cores[0],core_index_I=cores[1])
    baseline=OLD/'per_run'/f'ee{ee:g}_d1_n1_t2511_s{seed}'/'trajectory.npz'
    prefix_ms=min(switch,observer.nsteps*sub.params.dt)
    prefix={}
    with np.load(baseline) as b:
        for k,step in [('spike_counts_2ms',2),('active_counts_2ms',2),('active_counts_10ms',10)]:
            prefix[k]=bool(np.array_equal(arrays[k][:round(prefix_ms/step)],b[k][:round(prefix_ms/step)]))
        masknew=arrays['raster_time_ms']<=prefix_ms;maskold=b['raster_time_ms']<=prefix_ms
        prefix['raster']=bool(np.array_equal(arrays['raster_time_ms'][masknew],b['raster_time_ms'][maskold]) and np.array_equal(arrays['raster_cell'][masknew],b['raster_cell'][maskold]))
    assert all(prefix.values()),prefix
    np.savez_compressed(folder/'trajectory.npz',**arrays)
    payload=dict(status='COMPLETE',name=name,ee=ee,depth=1.,seed=seed,topology=2511,arm=arm,
        switch_ms=switch,probe_ms=PROBE if arm=='all_off_probe' and duration>PROBE else None,
        duration_ms=observer.nsteps*sub.params.dt,requested_duration_ms=duration,dt_ms=sub.params.dt,
        baseline=str(baseline),baseline_sha256=native.sha(baseline),prefix_ms=prefix_ms,prefix_bitwise=prefix,
        constant_core_input_per_ms=drive.matched,ou_stationary_sd_per_ms=drive.ou.stationary_std,
        runaway_ms=result.get('runaway_early_stop_ms'),wall_s=time.time()-start,
        source_engine=str(Path(inspect.getfile(native.ORIGINAL))),source_engine_sha256=native.sha(inspect.getfile(native.ORIGINAL)),
        arrays_sha256=native.sha(folder/'trajectory.npz'),identity=applied['identity'],
        forced_spike_requested_count=result.get('forced_spike_requested_count'),forced_spike_collision_count=result.get('forced_spike_collision_count'))
    native.write(folder/'result.json',payload)
    print(json.dumps({k:payload[k] for k in ('name','status','duration_ms','wall_s','prefix_bitwise')}),flush=True)

def launch():
    from concurrent.futures import ThreadPoolExecutor,as_completed
    jobs=[(ee,seed,arm) for ee in (.7,.85,1.,1.2) for seed in (848101,848102) for arm in ('ou_off','all_off_probe')]
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    def run(job):
        ee,seed,arm=job;name=f'ee{ee:g}_s{seed}_{arm}'
        with (logs/f'{name}.log').open('w') as f:
            p=subprocess.run([sys.executable,__file__,'--ee',str(ee),'--seed',str(seed),'--arm',arm],stdout=f,stderr=subprocess.STDOUT)
        return dict(name=name,exit_code=p.returncode)
    done=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            d=future.result();done.append(d)
            native.write(OUT/'status.json',dict(status='RUNNING' if len(done)<len(jobs) else 'FINISHED',planned=len(jobs),finished=len(done),jobs=done))
            print(json.dumps(d),flush=True)
    assert all(d['exit_code']==0 for d in done),done

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--ee',type=float,default=.85)
    p.add_argument('--seed',type=int,default=848101);p.add_argument('--arm',choices=['ou_off','all_off_probe'],default='ou_off')
    p.add_argument('--duration',type=float,default=20000.);p.add_argument('--switch',type=float,default=SWITCH);p.add_argument('--tag')
    a=p.parse_args()
    if a.launch:launch()
    else:simulate(a.ee,a.seed,a.arm,a.duration,a.switch,a.tag)
