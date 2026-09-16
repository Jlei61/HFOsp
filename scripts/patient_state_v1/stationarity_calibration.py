"""Can fitting different recording prefixes explain the observed parameter shift?

Generate each complete marked sequence once at actual observation times under a
stationary fitted model, then refit the same nested prefixes. This retains the
dependence between prefix estimates; it is a model check, not a biological test.
"""
import sys,json,time,traceback,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from numba import njit
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import fit,laplace,slice_data
from scripts.patient_state_v1.advanced_controls import history_data
from scripts.patient_state_v1.analyze_first import best_fits
from scripts.patient_state_v1.review_predictions import selected

OUT=RUN/'prefix_stationarity_v1_8'

@njit(cache=True)
def simulate(dt,reset,b,gamma,tau,sd,seed):
    np.random.seed(seed);y=np.empty(len(dt),np.int64);s=0.;last=.5
    for i in range(len(dt)):
        if reset[i]:s=np.random.normal()*sd;h=0.
        else:
            a=np.exp(-dt[i]/tau);s=a*s+sd*np.sqrt(-np.expm1(-2*dt[i]/tau))*np.random.normal();h=(last-.5)*np.exp(-dt[i]*3600)
        p=1/(1+np.exp(-(b+s+gamma*h)));y[i]=1 if np.random.random()<p else 0;last=y[i]
    return y

def worker(job):
    path=OUT/'fits'/f"{job['model']}_{job['rep']:03d}_{job['scope']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        d=dict(np.load(RUN/'observations.npz'));t=job['truth'];gamma=t[1] if job['model']=='ou_history' else 0.;d['y']=simulate(d['dt'],d['reset'],t[0],gamma,np.exp(t[-2]),np.exp(t[-1]),job['seed']);d=slice_data(d,0,job['end']);b=np.log((d['y'].mean()+1e-6)/(1-d['y'].mean()+1e-6));results=[]
        if job['model']=='ou_history':d=history_data(d)
        for tau in (.2,2.):
            if job['model']=='ou':r=fit(d,'ou',np.r_[b,np.log(tau),t[-1]])
            else:
                def expand(x):return np.r_[x[:2],0.,x[2:]]
                opt=minimize(lambda x:-laplace(expand(x),d,True),np.r_[b,t[1],np.log(tau),t[-1]],method='L-BFGS-B',bounds=[(-8,8),(-8,8),(np.log(1/60),np.log(24)),(np.log(.01),np.log(5))],options={'maxiter':150,'ftol':1e-10,'eps':2e-5})
                r=dict(theta=opt.x,loglik=-opt.fun,success=opt.success,message=opt.message)
            results.append(r)
        r=dict(max(results,key=lambda x:x['loglik']));r.update(status='COMPLETE',job=job,elapsed=time.time()-start,initial_fits=results)
    except Exception:r=dict(status='FAILED',job=job,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,r);return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--replicates',type=int,default=128);ap.add_argument('--workers',type=int,default=24);args=ap.parse_args();data=np.load(RUN/'observations.npz');folds=json.loads((RUN/'splits.json').read_text());scopes=[('full',len(data['y']))]+[(f"fold{f['fold']}",f['train_end']) for f in folds];truth={'ou':best_fits()['full','ou']['theta'],'ou_history':selected(RUN/'advanced_controls_v1_2/fits','full','ou_history')[0]['theta']};jobs=[]
    for i,(model,t) in enumerate(truth.items()):
        for rep in range(args.replicates):
            for scope,end in scopes:jobs.append(dict(model=model,truth=t,scope=scope,end=end,rep=rep,seed=330000+i*10000+rep))
    write_json(OUT/'queue.json',jobs);write_json(OUT/'contract.json',dict(question='Does finite-prefix estimation under a stationary model account for parameter differences?',replicates=args.replicates,observations='Actual eligible event times and coverage/epoch starts, labels generated without seizure outcomes',statistic='Across nested prefix log-tau range, plus signed first-prefix minus full log-tau',interpretation='Model misspecification check, not proof of a unique time-varying physiological drift'))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),model=r['job']['model'],rep=r['job']['rep'],scope=r['job']['scope'],status=r['status'],elapsed=r['elapsed'])),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs),finished_unix=time.time()))

if __name__=='__main__':main()
