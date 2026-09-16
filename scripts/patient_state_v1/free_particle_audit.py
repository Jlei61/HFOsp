"""Independent particle checks of free-time ADF and Laplace solutions."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.special import logsumexp
from numba import cuda
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.gpu_two_timescale import dataset,evaluate,expand
from scripts.patient_state_v1.model import slice_data
OUT=RUN/'free_timescale_particle_audit_v1_14'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--replicates',type=int,default=256);args=ap.parse_args();cuda.select_device(args.gpu);OUT.mkdir(exist_ok=True)
    d=slice_data(dataset(False),0,30);t=np.array([[-.8,.7,np.log(.01),np.log(.4),np.log(.6)] for _ in range(4)])
    a=evaluate(d,t,6,seed=4199,batch=4);b=evaluate(d,np.column_stack([t,np.full(4,np.log(6))]),-1,seed=4199,batch=4)
    assert np.max(abs(a-b))<1e-10
    write_json(OUT/'kernel_parameter_canary.json',dict(status='PASS',max_absolute_difference=np.max(abs(a-b)),check='Optional per-row background time reproduces old scalar-time likelihood under identical random numbers'))
    best={}
    for path in (RUN/'two_timescale_free_tau_v1_13/fits').glob('full_*.json'):
        r=json.loads(path.read_text());j=r['job'];key=(j['carry'],j['history'],j['method'])
        if r['status']=='COMPLETE' and r['feasible'] and (key not in best or r['loglik']>best[key][0]['loglik']):best[key]=(r,path)
    for idx,(key,(r,path)) in enumerate(sorted(best.items())):
        carry,hist,method=key;dest=OUT/f'carry{int(carry)}_hist{int(hist)}_{method}.json'
        if dest.exists():continue
        start=time.time();theta=expand(np.tile(r['theta'],(args.replicates,1)),hist);ll=evaluate(dataset(carry),theta,1,seed=410000+idx*10000,batch=64);w=np.exp(ll-logsumexp(ll));v=logsumexp(ll)-np.log(len(ll))
        # Resample independent likelihood estimators, not log estimators.
        rng=np.random.default_rng(527+idx);scaled=np.exp(ll-ll.max());boot=np.log(np.mean(scaled[rng.integers(0,len(ll),(4000,len(ll)))],axis=1))+ll.max()
        result=dict(status='COMPLETE',fit_source=str(path),carry=carry,history=hist,method=method,theta=r['theta'],loglik_replicates=ll,particle_logmean=v,logmean_bootstrap_interval=np.quantile(boot,[.025,.975]),loglik_sd=np.std(ll),likelihood_ess=1/np.sum(w*w),replicates=args.replicates,particles=1024,adf_at_solution=r['adf_at_solution'],laplace_at_solution=r['laplace_at_solution'],elapsed=time.time()-start,uncertainty_note='Monte Carlo bootstrap may miss an unobserved likelihood tail; inspect likelihood ESS and replicate log variance')
        write_json(dest,result);print(json.dumps({k:result[k] for k in ['carry','history','method','particle_logmean','loglik_sd','likelihood_ess','elapsed']}),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_models=len(best)))
if __name__=='__main__':main()
