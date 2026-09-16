"""Native SNN near reduced bifurcations; exact spike rasters from a fixed sample."""
from pathlib import Path
import os,sys,json
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
V2=ROOT/'results/topic4_sef_hfo/core_burst_bifurcation_v2_20260915'
OUT=ROOT/'results/topic4_sef_hfo/core_network_bifurcation_v7_20260916'
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    path=OUT/p;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(d,indent=2)+'\n')
import argparse,hashlib,inspect,time,types,subprocess
import numpy as np
sys.path.insert(0,str(ROOT/'scripts/topic4_burst_regime'))
import runtime as native
OLD=native.OUT
native.OUT=OUT/'native';native.OUT.mkdir(exist_ok=True)
VALUES=[1.10,1.14,1.17623,1.17632,1.24,1.25,1.34,1.355,1.37,1.375,1.38,1.385,1.395,1.45,1.12181,1.12183]
class Observer(native.Observer):
    def __init__(self,sub,groups,duration_ms,path):
        super().__init__(sub,groups,duration_ms,path)
        ne=sub.n_e
        # The frozen projected graph supplies exactly the executor's six masks.
        z=np.load(V2/'projected_graph.npz');region=z['region']
        assert np.array_equal(z['vtheta'],sub.vtheta)
        assert np.array_equal(z['positions'],np.r_[sub.positions_e,sub.positions_i])
        self.region=region;self.groups6=[np.flatnonzero(region==j) for j in range(6)]
        self.counts6=np.zeros((len(self.counts),6),np.uint32)
        self.sample=np.sort(np.concatenate([g[np.linspace(0,len(g)-1,min(len(g),100),dtype=int)] for g in self.groups6]))
        self.exact_t=[];self.exact_i=[]
    def observe(self,t,tm,xi,nu,ext,delta,V,I_E,I_I,spk):
        sampled=self.sample[spk[self.sample]]
        self.exact_t.extend([tm]*len(sampled));self.exact_i.extend(sampled.tolist())
        if (t+1)%self.stride==0:
            c=self.count+spk
            for j,g in enumerate(self.groups6):self.counts6[(t+1)//self.stride-1,j]=c[g].sum()
        super().observe(t,tm,xi,nu,ext,delta,V,I_E,I_I,spk)
    def arrays(self):
        out=super().arrays();out.update(exact_spike_time_ms=np.array(self.exact_t,np.float32),exact_spike_cell=np.array(self.exact_i,np.int32),
            six_group_counts_2ms=self.counts6[:self.nsteps//self.stride],region=self.region)
        return out

def run(g,duration=12000,tag=None):
    name=tag or f'J{g:g}_s848101';folder=native.OUT/'per_run'/name
    if (folder/'result.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    original=read(OLD/'per_run/ee1_d1_n1_t2511_s848101/result.json')
    assert native.sha(inspect.getfile(native.ORIGINAL))==original['source_engine_sha256']
    native.Observer=Observer
    start=time.time();sub,groups,loading,det,applied,cores=native.setup(g,1,848101)
    assert applied['threshold']['n_raised']==0
    native.write(folder/'applied_physics.json',applied)
    result,observer=native.simulate(sub,groups,loading,det,cores[0],848101,duration,folder/'progress.json')
    arr=observer.arrays();arr.update(core_index_E=cores[0],core_index_I=cores[1],vtheta=sub.vtheta)
    # Aggregate spike counts must partition all actual E and I spikes.
    assert np.array_equal(arr['six_group_counts_2ms'][:,:3].sum(1),arr['spike_counts_2ms'][:,observer.names.index('allE')])
    assert np.array_equal(arr['six_group_counts_2ms'][:,3:].sum(1),arr['spike_counts_2ms'][:,observer.names.index('allI')])
    if tag=='prefix_validation':
        with np.load(OLD/'per_run/ee1_d1_n1_t2511_s848101/trajectory.npz') as z:
            for k in ['spike_counts_2ms','active_counts_2ms','active_counts_10ms']:
                assert np.array_equal(arr[k],z[k][:len(arr[k])]),k
    np.savez_compressed(folder/'trajectory.npz',**arr)
    native.write(folder/'result.json',dict(status='COMPLETE',g=g,ee=g,depth=1,seed=848101,topology=2511,noise=True,
        requested_duration_ms=duration,actual_duration_ms=observer.nsteps*sub.params.dt,burnin_ms=2000,
        runaway_early_stop_ms=result.get('runaway_early_stop_ms'),n_e=sub.n_e,n_i=sub.n_i,dt_ms=sub.params.dt,
        frozen_engine_sha256=original['source_engine_sha256'],arrays_sha256=native.sha(folder/'trajectory.npz'),
        threshold_raised=applied['threshold']['n_raised'],raster='exact native spikes at original integration time; fixed 100 cells per six groups',wall_s=time.time()-start))
    print('NATIVE_COMPLETE',g,time.time()-start,flush=True)

def batch():
    from concurrent.futures import ThreadPoolExecutor,as_completed
    assert (native.OUT/'per_run/prefix_validation/result.json').exists()
    def one(g):
        path=OUT/'solver_logs'/f'native_J{g:g}.log'
        with path.open('w') as f:code=subprocess.run([sys.executable,__file__,'--g',str(g)],stdout=f,stderr=subprocess.STDOUT).returncode
        return dict(g=g,returncode=code,log=str(path))
    jobs=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(one,g) for g in VALUES]):
            jobs.append(f.result());write('native_batch_status.json',dict(completed=len(jobs),total=len(VALUES),jobs=jobs));print(jobs[-1],flush=True)
    assert all(x['returncode']==0 for x in jobs)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--g',type=float,default=1);p.add_argument('--duration',type=float,default=12000);p.add_argument('--tag');p.add_argument('--batch',action='store_true');a=p.parse_args()
    if a.batch:batch()
    else:run(a.g,a.duration,a.tag)
