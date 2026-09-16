"""Versioned diagnostic wave: parameter profiles, bootstrap, and history/static-context controls."""
import sys,json,time,traceback,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.special import expit,betaln
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
from numba import njit
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import fit,laplace,slice_data,filter_adf,simulate
from scripts.patient_state_v1.analyze_first import best_fits,metrics

@njit(cache=True)
def beta_predict(y,n,group,alpha,beta):
    p=np.empty(len(y));a=alpha;b=beta
    for i in range(len(y)):
        if i==0 or group[i]!=group[i-1]:a=alpha;b=beta
        p[i]=a/(a+b);a+=y[i];b+=n[i]-y[i]
    return p

def beta_fit_predict(data,groups,end):
    ids=np.r_[0,np.flatnonzero(np.diff(groups[:end])!=0)+1]
    yy=np.add.reduceat(data['y'][:end],ids);nn=np.add.reduceat(data['n'][:end],ids)
    def fun(t):
        a,b=np.exp(t)
        return -(betaln(a+yy,b+nn-yy)-betaln(a,b)).sum()
    r=minimize(fun,np.log([3.,7.]),method='L-BFGS-B',bounds=[(-5,12)]*2)
    a,b=np.exp(r.x);return beta_predict(data['y'],data['n'],groups,a,b),dict(alpha=a,beta=b,train_loglik=-r.fun)

@njit(cache=True)
def ewma(y,n,dt,reset,tau,strength,p0):
    p=np.empty(len(y));a=0.;b=0.
    for i in range(len(y)):
        if reset[i]:a=0.;b=0.
        else:
            decay=np.exp(-dt[i]/tau);a*=decay;b*=decay
        p[i]=(a+strength*p0)/(a+b+strength)
        a+=y[i];b+=n[i]-y[i]
    return p

def ewma_fit_predict(data,end):
    p0=data['y'][:end].sum()/data['n'][:end].sum()
    def fun(t):
        pp=ewma(data['y'][:end],data['n'][:end],data['dt'][:end],data['reset'][:end],np.exp(t[0]),np.exp(t[1]),expit(t[2]))
        return -metrics(pp,data['y'][:end],data['n'][:end])['loglik']
    best=None
    for tau in (.01,.1,1.):
        r=minimize(fun,np.r_[np.log(tau),np.log(10),np.log(p0/(1-p0))],method='L-BFGS-B',bounds=[(np.log(1/3600),np.log(24)),(np.log(.01),np.log(10000)),(-8,8)])
        if best is None or r.fun<best.fun:best=r
    tau,strength=np.exp(best.x[:2]);p0=expit(best.x[2]);pp=ewma(data['y'],data['n'],data['dt'],data['reset'],tau,strength,p0)
    return pp,dict(tau_hours=tau,strength=strength,p0=p0,train_loglik=-best.fun)

def controls():
    data=dict(np.load(RUN/'observations.npz'));events=pd.read_csv(RUN/'events.csv');folds=json.loads((RUN/'splits.json').read_text());rows=[]
    for f in folds:
        for model in ('constant_within_coverage','constant_within_epoch','ewma'):
            end=f['train_end']
            if model=='ewma':pp,r=ewma_fit_predict(data,end)
            else:
                groups=events.coverage_segment.to_numpy() if model.endswith('coverage') else data['epoch']
                # Coverage membership can straddle seizures, so always split ictal exclusions.
                groups=groups*1000+data['epoch'];pp,r=beta_fit_predict(data,groups,end)
            a,b=f['test_start'],f['test_end'];rows.append(dict(fold=f['fold'],model=model,**metrics(pp[a:b],data['y'][a:b],data['n'][a:b]),**r))
    pd.DataFrame(rows).to_csv(RUN/'round2_controls.csv',index=False);print(pd.DataFrame(rows).to_string(index=False),flush=True)

def job_worker(job):
    path=RUN/'round2'/f"{job['id']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        data=slice_data(dict(np.load(RUN/'observations.npz')),0,job['end']);theta=np.array(job['theta']);cyclic=job['model']=='ou_cycle';nc=3 if cyclic else 1
        if job['kind']=='profile':
            logtau=np.log(job['tau']);free=np.r_[theta[:nc],theta[nc+1]]
            def build(t):return np.r_[t[:nc],logtau,t[nc]]
            def fun(t):return -laplace(build(t),data,cyclic)
            opt=minimize(fun,free,method='L-BFGS-B',bounds=[(-8,8)]*nc+[(np.log(.01),np.log(5))],options={'ftol':1e-10,'eps':2e-5,'maxiter':120})
            result=dict(theta=build(opt.x),loglik=-opt.fun,success=opt.success)
        elif job['kind']=='bootstrap':
            y,s=simulate(data,theta[0],np.exp(theta[nc]),np.exp(theta[nc+1]),job['seed'],theta[1:3] if cyclic else (0.,0.))
            data['y']=y;fits=[]
            for shift in (0.,-1.,1.):
                init=theta.copy();init[nc]=np.clip(init[nc]+shift,np.log(1/60),np.log(24));fits.append(fit(data,job['model'],init,maxiter=120))
            result=max(fits,key=lambda x:x['loglik']);result['truth']=theta
        else:raise ValueError(job['kind'])
        result.update(status='COMPLETE',job=job,elapsed=time.time()-start)
    except Exception:result=dict(status='FAILED',job=job,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=28);ap.add_argument('--controls-only',action='store_true');args=ap.parse_args()
    controls()
    if args.controls_only:return
    best=best_fits();folds=json.loads((RUN/'splits.json').read_text());length=len(np.load(RUN/'observations.npz')['y']);scopes=[('full',length)]+[(f"fold{f['fold']}",f['train_end']) for f in folds]
    jobs=[]
    for scope,end in scopes:
        for model in ('ou','ou_cycle'):
            theta=best[scope,model]['theta']
            for i,tau in enumerate(np.geomspace(1/60,24,61)):
                jobs.append(dict(id=f'profile_{scope}_{model}_{i:03d}',kind='profile',scope=scope,end=end,model=model,tau=tau,theta=theta))
    for model in ('ou','ou_cycle'):
        for seed in range(128):jobs.append(dict(id=f'bootstrap_full_{model}_{seed:03d}',kind='bootstrap',scope='full',end=length,model=model,seed=26092000+seed,theta=best['full',model]['theta']))
    write_json(RUN/'round2_queue.json',jobs);print(json.dumps(dict(total=len(jobs),workers=args.workers)),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs=[ex.submit(job_worker,j) for j in jobs]
        for i,f in enumerate(as_completed(fs)):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),id=r['job']['id'],status=r['status'],elapsed=r['elapsed'])),flush=True)
    write_json(RUN/'round2_status.json',dict(status='COMPLETE',jobs=len(jobs),finished_unix=time.time()))

if __name__=='__main__':main()
