"""Can a stationary two-scale process mimic single-scale prefix instability?"""
import sys,json,time,traceback,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from numba import njit
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.gpu_two_timescale import dataset
from scripts.patient_state_v1.model import slice_data,laplace
from scripts.patient_state_v1.advanced_controls import history_data
OUT=RUN/'two_scale_prefix_calibration_v1_14'
@njit(cache=True)
def simulate(dt,reset,sdt,sreset,b,gamma,tf,sdf,ts,sds,seed):
    np.random.seed(seed);sf=0.;sb=0.;last=.5;y=np.empty(len(dt),np.int64)
    for i in range(len(dt)):
        if reset[i]:sf=np.random.normal()*sdf;h=0.
        else:
            a=np.exp(-dt[i]/tf);sf=a*sf+sdf*np.sqrt(-np.expm1(-2*dt[i]/tf))*np.random.normal();h=(last-.5)*np.exp(-dt[i]*3600)
        if sreset[i]:sb=np.random.normal()*sds
        else:
            a=np.exp(-sdt[i]/ts);sb=a*sb+sds*np.sqrt(-np.expm1(-2*sdt[i]/ts))*np.random.normal()
        p=1/(1+np.exp(-(b+gamma*h+sf+sb)));y[i]=int(np.random.random()<p);last=y[i]
    return y

def worker(j):
    path=OUT/'fits'/f"carry{int(j['carry'])}_{j['rep']:03d}_{j['scope']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        d=dataset(j['carry']);t=j['truth'];d['y']=simulate(d['dt'],d['reset'],d['slow_dt'],d['slow_reset'],t[0],t[1],np.exp(t[2]),np.exp(t[3]),j.get('slow_tau',6),np.exp(t[4]),j['seed']);d=history_data(slice_data(d,0,j['end']));p=d['y'].mean();b=np.log((p+1e-6)/(1-p+1e-6));rr=[]
        for tau in [.2,2.]:
            def objective(x):return -laplace(np.r_[x[:2],0.,x[2:]],d,True)
            o=minimize(objective,[b,.7,np.log(tau),np.log(.6)],method='L-BFGS-B',bounds=[(-8,8),(-8,8),(np.log(1/60),np.log(24)),(np.log(.01),np.log(5))],options={'maxiter':150,'ftol':1e-10,'eps':2e-5});rr.append(dict(theta=o.x,loglik=-o.fun,success=o.success,message=o.message))
        r=dict(max(rr,key=lambda x:x['loglik']));r.update(status='COMPLETE',job=j,initial_fits=rr,elapsed=time.time()-start)
    except Exception:r=dict(status='FAILED',job=j,traceback=traceback.format_exc(),elapsed=time.time()-start)
    write_json(path,r);return r

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--source',choices=['fixed6h','free_adf'],default='fixed6h');args=ap.parse_args()
    if args.source=='free_adf':OUT=RUN/'two_scale_free_prefix_calibration_v1_15'
    folds=json.loads((RUN/'splits.json').read_text());scopes=[('full',len(np.load(RUN/'observations.npz')['y']))]+[(f"fold{f['fold']}",f['train_end']) for f in folds];jobs=[];sources={}
    for carry in [False,True]:
        fits=[]
        directory='two_timescale_marks_v1_11' if args.source=='fixed6h' else 'two_timescale_free_tau_v1_13'
        for p in (RUN/directory/'fits').glob('full_*.json'):
            r=json.loads(p.read_text());j=r['job']
            wanted=j['history'] and (j['slow_tau']==6 and j['carry_slow']==carry if args.source=='fixed6h' else j['method']=='adf' and j['carry']==carry)
            if r['status']=='COMPLETE' and wanted:fits.append((r,p))
        r,path=max(fits,key=lambda x:x[0]['loglik']);sources[str(carry)]=str(path)
        for rep in range(128):
            for scope,end in scopes:jobs.append(dict(carry=carry,rep=rep,scope=scope,end=end,truth=r['theta'],slow_tau=6 if args.source=='fixed6h' else float(np.exp(r['theta'][-1])),seed=430000+rep+1000*carry+(20000 if args.source=='free_adf' else 0)))
    write_json(OUT/'contract.json',dict(question='Can fixed two-scale dynamics produce the apparent prefix-dependent time constant of a misspecified single-scale fit?',n_replicates_per_boundary=128,truth_sources=sources,truth_version=args.source,observation='Actual event times and exclusions; labels generated; all nested prefixes from the same generated full sequence',fit='Original single OU plus 1-second event-history model with same parameter bounds and two starts',limits='Successful mimicry would offer an alternative explanation, not prove two physiological components; failure limits the specific fitted stationary generator'))
    write_json(OUT/'queue.json',jobs)
    with ProcessPoolExecutor(max_workers=24) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),status=r['status'],carry=r['job']['carry'],scope=r['job']['scope'],elapsed=r['elapsed'])),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs)))
if __name__=='__main__':main()
