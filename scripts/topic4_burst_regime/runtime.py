"""Native, frozen core-input SNN with a bounded observer and ordered scatter.

No patient event filtering, classifier or loss is used in this experiment.
The imported scientific source and graph are not edited.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, copy, hashlib, inspect, json, sys, time, types
import numpy as np

MAIN = Path('/home/honglab/leijiaxin/HFOsp')
SOURCE = MAIN / '.worktrees/topic4-continuous-core-state-r1'
OUT = MAIN / 'results/topic4_sef_hfo/burst_regime_map_20260914'
sys.path[:0] = [str(SOURCE), str(SOURCE / 'src/snn_engine')]
from scripts import run_topic4_shape_output_response as builder
from src.topic4_core_ou_correlation import CoreOUMixture
from params import compute_nu_theta
from numba import njit

REFERENCE = Path('/data/hfosp/topic4_sef_hfo/core_multiseed_response_curves_20260913')
ORIGINAL = builder.base.simulate_kick

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@njit(cache=True)
def ordered_scatter(ring, sources, ptr, dst, delay, weight, step):
    slots = np.empty(ring.shape[0], np.int64)
    for lag in range(ring.shape[0]):
        slots[lag] = (step + lag) % ring.shape[0]
    for source in sources:
        for edge in range(ptr[source], ptr[source + 1]):
            ring[slots[delay[edge]], dst[edge]] += weight[edge]

def accelerated():
    source = inspect.getsource(ORIGINAL)
    start = source.index('            if spE.size:\n', source.index('            spE = np.where'))
    end = source.index('\n        if verbose and', start)
    replacement = '''            if ee_std_on:
                raise RuntimeError("This isolated backend requires short-term plasticity off")
            if spE.size:
                ordered_scatter(ring_sE, spE, a_indptr, a_dst, a_dly, a_w, _abs_step)
            if spI.size:
                ordered_scatter(ring_sI, spI, g_indptr, g_dst, g_dly, g_w, _abs_step)
'''
    namespace = dict(ORIGINAL.__globals__, ordered_scatter=ordered_scatter)
    exec(compile(source[:start] + replacement + source[end:], '<isolated_ordered_scatter>', 'exec'), namespace)
    return namespace[ORIGINAL.__name__]

class SpikeSink:
    def __init__(self, shape): self.shape, self.seen = shape, 0
    def __len__(self): return self.shape[0]
    def __setitem__(self, t, spikes):
        if t != self.seen: raise RuntimeError('noncontiguous native recorder')
        self.seen += 1
    def __getitem__(self, key):
        if not isinstance(key, slice) or key.start is not None or key.step is not None or key.stop != self.seen:
            raise RuntimeError('unexpected recorder slice')
        self.shape = (self.seen, self.shape[1])
        return self

class NumpyProxy:
    def __init__(self, shape): self.shape, self.hits = shape, 0
    def __getattr__(self, name): return getattr(np, name)
    def zeros(self, shape, dtype=float, *args, **kwargs):
        if isinstance(shape, tuple) and shape == self.shape and np.dtype(dtype) == np.dtype(bool):
            self.hits += 1
            if self.hits != 1: raise RuntimeError('ambiguous dense recorder interception')
            return SpikeSink(shape)
        return np.zeros(shape, dtype, *args, **kwargs)

class Observer:
    def __init__(self, sub, groups, duration_ms, path):
        self.dt, self.ne, self.n = sub.params.dt, sub.n_e, sub.n_e + sub.n_i
        self.groups = {k:np.asarray(groups[k]) for k in ('coreAE','coreBE','surroundE','allI')}
        self.groups['allE'] = np.arange(self.ne)
        self.groups['coreUnionE'] = np.r_[groups['coreAE'], groups['coreBE']]
        self.names = list(self.groups)
        self.stride = round(2. / self.dt)
        frames = int(np.ceil(duration_ms / 2.))
        self.counts = np.zeros((frames,len(self.names)), np.uint32)
        self.active = np.zeros_like(self.counts)
        self.window10 = np.zeros((int(np.ceil(duration_ms / 10.)),len(self.names)),np.uint32)
        self.count = np.zeros(self.n,np.uint16)
        self.seen10 = np.zeros(self.n,bool)
        self.sample = np.unique(np.concatenate([idx[np.linspace(0,len(idx)-1,min(len(idx),100),dtype=int)] for idx in self.groups.values() if len(idx)]))
        self.raster_t, self.raster_i = [], []
        self.nsteps = 0
        self.path = path
        self.started = time.time()
        self.total_spikes = np.zeros(self.n,np.uint32)
    def observe(self,t,tm,xi,nu,ext,delta,V,I_E,I_I,spk):
        self.nsteps = t+1
        self.count += spk
        self.total_spikes += spk
        self.seen10 |= spk
        if (t+1) % self.stride == 0:
            frame=(t+1)//self.stride-1
            for j,idx in enumerate(self.groups.values()):
                self.counts[frame,j]=self.count[idx].sum()
                self.active[frame,j]=np.count_nonzero(self.count[idx])
            sampled=self.sample[self.count[self.sample]>0]
            self.raster_t.extend([tm+self.dt]*len(sampled));self.raster_i.extend(sampled.tolist())
            self.count.fill(0)
        if (t+1) % round(10./self.dt) == 0:
            frame=(t+1)//round(10./self.dt)-1
            for j,idx in enumerate(self.groups.values()): self.window10[frame,j]=self.seen10[idx].sum()
            self.seen10.fill(False)
        if (t+1)%round(1000./self.dt)==0:
            write(self.path,dict(status='SIMULATING',simulated_ms=tm+self.dt,wall_s=time.time()-self.started))
    def arrays(self):
        frames=self.nsteps//self.stride
        return dict(group_names=np.array(self.names),group_sizes=np.array([len(x) for x in self.groups.values()]),
            spike_counts_2ms=self.counts[:frames],active_counts_2ms=self.active[:frames],
            active_counts_10ms=self.window10[:self.nsteps//round(10./self.dt)],
            raster_time_ms=np.asarray(self.raster_t,np.float32),raster_cell=np.asarray(self.raster_i,np.int32),
            raster_sample_ids=self.sample,total_spikes_per_cell=self.total_spikes)

def setup(ee, depth, seed, topology=2511):
    builder.OUT = OUT
    cache = OUT/'global_graph_cache'
    cache.mkdir(exist_ok=True)
    for old in (REFERENCE/'global_graph_cache').glob('*.pkl'):
        target=cache/old.name
        if not target.exists(): target.symlink_to(old)
    candidate=json.loads((REFERENCE/'candidates/curve_circle_out1_EI1.json').read_text())
    candidate['parameters'].update(EE_same_core_scale=float(ee),depth_A_scale=float(depth),depth_B_scale=float(depth))
    candidate['id']=f'ee{ee:g}_depth{depth:g}'
    record=None if topology==2511 else json.loads((REFERENCE/'confirmation/frozen_networks.json').read_text())
    return builder.build_unit(candidate,topology,seed,record)

def simulate(sub, groups, loading, deterministic, core, seed, duration, path, fast=True, noise=True, capture=None):
    sub.params.T=duration
    sub.net['rng']=np.random.default_rng(seed)
    signal=float(sub.params.nu_ext_ratio*compute_nu_theta(sub.params)[0])
    if noise:
        drive=CoreOUMixture(core,signal,.95,0.,sub.params.dt,sub.params.tau_n,sub.params.sigma_n,seed)
    else:
        from scipy.special import ndtr
        mu=.95*signal
        sigma=sub.params.sigma_n*1e-3*np.sqrt(sub.params.tau_n/2.)
        matched=mu if sigma==0 else mu*ndtr(mu/sigma)+sigma*np.exp(-.5*(mu/sigma)**2)/np.sqrt(2*np.pi)
        class Constant:
            def __init__(self): self.values=np.where(core>=0,matched-signal,0.)
            def step(self,t): return self.values
        drive=Constant()
        deterministic=np.ones_like(deterministic)
    observer=Observer(sub,groups,duration,path)
    function=accelerated() if fast else ORIGINAL
    proxy=NumpyProxy((round(duration/sub.params.dt),sub.n_e))
    namespace=dict(function.__globals__,np=proxy)
    function=types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    kwargs=dict(KICK_BOOST=0.,t_kick=1e9,V_th_per_neuron=sub.vtheta,slow=None,
        early_stop_runaway=True,es_thresh_hz=120.,es_dur_ms=100.,post_runaway_record_ms=500.,
        global_ou_loading=np.zeros_like(loading),deterministic_external_mask=deterministic,
        external_e_rate_drive=drive,step_observer=observer)
    if capture is not None:
        kwargs.update(checkpoint_steps=[round(duration/sub.params.dt)-1],checkpoint_sink=lambda *a,**k:capture.append((a,k)))
    result=function(sub.params,sub.net,**kwargs)
    assert proxy.hits==1
    return result,observer

def worker(ee,depth,seed,duration,topology=2511,noise=True,fast=True,tag=None):
    name=tag or f'ee{ee:g}_d{depth:g}_n{int(noise)}_t{topology}_s{seed}'
    folder=OUT/'per_run'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists(): return
    start=time.time();write(folder/'progress.json',dict(status='BUILDING',started_unix=start))
    sub,groups,loading,deterministic,applied,cores=setup(ee,depth,seed,topology)
    assert applied['threshold']['n_raised']==0
    applied['experiment']=dict(ee=ee,depth=depth,seed=seed,topology=topology,noise=noise,duration_ms=duration,
        backend='ordered_serial_scatter' if fast else 'original_numpy_scatter',patient_classifier_used=False)
    applied['experiment_input']=dict(core_mean_scale=.95,between_core_ou_correlation=0.,
        signal_per_ms=float(sub.params.nu_ext_ratio*compute_nu_theta(sub.params)[0]),OU_sigma_n=sub.params.sigma_n,
        OU_tau_ms=sub.params.tau_n,noise_off_mean='stationary expectation of rectified Gaussian core rate; both OU and Poisson randomness removed',
        outside='same deterministic signal input in all conditions',biological_normal_tissue_claim=False)
    write(folder/'applied_physics.json',applied)
    result,observer=simulate(sub,groups,loading,deterministic,cores[0],seed,duration,folder/'progress.json',fast,noise)
    arrays=observer.arrays()
    arrays.update(positions_E=sub.positions_e,positions_I=sub.positions_i,vtheta=sub.vtheta,core_index_E=cores[0],core_index_I=cores[1],
        rate_E=result['rate_E'],rate_I=result['rate_I'])
    np.savez_compressed(folder/'trajectory.npz',**arrays)
    write(folder/'result.json',dict(status='COMPLETE',name=name,**applied['experiment'],actual_duration_ms=observer.nsteps*sub.params.dt,
        dt_ms=sub.params.dt,burnin_ms=2000.,runaway_early_stop_ms=result.get('runaway_early_stop_ms'),
        wall_s=time.time()-start,source_engine=str(Path(inspect.getfile(ORIGINAL))),source_engine_sha256=sha(inspect.getfile(ORIGINAL)),
        arrays_sha256=sha(folder/'trajectory.npz'),n_e=sub.n_e,n_i=sub.n_i,external_rate_clipping=result.get('external_e_rate_drive')))
    print(json.dumps(dict(name=name,status='COMPLETE',wall_s=time.time()-start)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ee',type=float,default=.85);p.add_argument('--depth',type=float,default=1.)
    p.add_argument('--seed',type=int,default=848101);p.add_argument('--topology',type=int,default=2511)
    p.add_argument('--duration',type=float,default=20000.);p.add_argument('--no-noise',action='store_true');p.add_argument('--original',action='store_true');p.add_argument('--tag')
    a=p.parse_args();worker(a.ee,a.depth,a.seed,a.duration,a.topology,not a.no_noise,not a.original,a.tag)
