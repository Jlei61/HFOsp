"""Bounded v1.2 alternatives: continuous-state plus fast history, and 2-state CT-HMM.

Hidden CT-HMM states are statistical regimes, not the observed TA/TB classes or proof of bistability.
"""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from numba import njit
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import laplace,filter_adf,slice_data
from scripts.patient_state_v1.analyze_first import metrics

OUT=RUN/'advanced_controls_v1_2'

@njit(cache=True)
def hmm_filter(theta,dt,reset,y,n):
    low=1/(1+np.exp(-theta[0]));high=1/(1+np.exp(-(theta[0]+np.exp(theta[1]))))
    k01=np.exp(theta[2]);k10=np.exp(theta[3]);pi=k01/(k01+k10);p=pi
    pred=np.empty(len(y));post=np.empty(len(y));terms=np.empty(len(y))
    for i in range(len(y)):
        if reset[i]:p=pi
        else:p=pi+(p-pi)*np.exp(-(k01+k10)*dt[i])
        pred[i]=(1-p)*low+p*high
        a=(1-p)*low**y[i]*(1-low)**(n[i]-y[i]);b=p*high**y[i]*(1-high)**(n[i]-y[i]);norm=max(a+b,1e-300)
        p=b/norm;post[i]=p;terms[i]=np.log(norm)
    return terms,pred,post

def history_data(d):
    d={k:np.array(v,copy=True) for k,v in d.items() if np.ndim(v)>0}
    h=np.r_[0.,d['y'][:-1]/d['n'][:-1]-.5]*np.exp(-d['dt']/(1/3600));h[d['reset']]=0
    d['x']=np.column_stack([np.ones(len(h)),h,np.zeros(len(h))]);return d

def worker(job):
    path=OUT/'fits'/f"{job['id']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        data=dict(np.load(RUN/'observations.npz'));d=slice_data(data,0,job['end']);p=np.clip(d['y'].mean(),.001,.999);b=np.log(p/(1-p))
        if job['model']=='cthmm2':
            t=np.array([b-1.,np.log(2),np.log(job['rate0']),np.log(job['rate0'])])
            def fun(t):return -hmm_filter(t,d['dt'],d['reset'],d['y'],d['n'])[0].sum()
            bounds=[(-8,5),(np.log(.02),np.log(12)),(np.log(1/48),np.log(3600))]*1
            bounds=[bounds[0],bounds[1],bounds[2],bounds[2]]
            r=minimize(fun,t,method='L-BFGS-B',bounds=bounds,options={'maxiter':250,'ftol':1e-11,'eps':1e-5})
            terms,pred,state=hmm_filter(r.x,data['dt'],data['reset'],data['y'],data['n'])
            params=dict(low_probability=expit(r.x[0]),high_probability=expit(r.x[0]+np.exp(r.x[1])),
                        low_mean_dwell_hours=np.exp(-r.x[2]),high_mean_dwell_hours=np.exp(-r.x[3]),
                        correlation_hours=1/np.exp(r.x[2:]).sum())
        else:
            d=history_data(d);data=history_data(data);t=np.array([b,1.,np.log(job['tau0']),np.log(.6)])
            def expand(t):return np.r_[t[:2],0,t[2:]]
            def fun(t):return -laplace(expand(t),d,True)
            r=minimize(fun,t,method='L-BFGS-B',bounds=[(-8,8),(-8,8),(np.log(1/60),np.log(24)),(np.log(.01),np.log(5))],options={'maxiter':200,'ftol':1e-10,'eps':2e-5})
            filtered=filter_adf(expand(r.x),data,True);pred=filtered['predict_tb'];state=filtered['mean'];params=dict(history_coefficient=r.x[1],tau_hours=np.exp(r.x[2]),sd=np.exp(r.x[3]))
        result=dict(status='COMPLETE',model=job['model'],theta=r.x,loglik=-r.fun,success=r.success,message=r.message,parameters=params,job=job,elapsed=time.time()-start)
        if job['scope']!='full':
            lo,hi=job['test_start'],job['test_end'];result['forward']=metrics(pred[lo:hi],data['y'][lo:hi],data['n'][lo:hi])
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),predict_tb=pred,state=state)
    except Exception:result=dict(status='FAILED',job=job,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,result);return result

def main():
    data=np.load(RUN/'observations.npz');folds=json.loads((RUN/'splits.json').read_text());scopes=[dict(scope='full',end=len(data['y']))]+[dict(scope=f"fold{f['fold']}",end=f['train_end'],test_start=f['test_start'],test_end=f['test_end']) for f in folds];jobs=[]
    for scope in scopes:
        for i,rate in enumerate((.2,2.,20.,200.)):jobs.append(dict(id=f"{scope['scope']}_cthmm2_{i}",model='cthmm2',rate0=rate,**scope))
        for i,tau in enumerate((.1,1.,6.)):jobs.append(dict(id=f"{scope['scope']}_ou_history_{i}",model='ou_history',tau0=tau,**scope))
    write_json(OUT/'queue.json',jobs)
    with ProcessPoolExecutor(max_workers=20) as ex:
        for f in as_completed([ex.submit(worker,j) for j in jobs]):
            r=f.result();print(json.dumps(dict(id=r['job']['id'],status=r['status'],elapsed=r['elapsed'],loglik=r.get('loglik'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs),finished_unix=time.time()))

if __name__=='__main__':main()
