"""Seizure-neighborhood and state-boundary sensitivity; fixed labels and comparisons."""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import fit,slice_data,laplace,filter_adf
from scripts.patient_state_v1.advanced_controls import history_data
from scripts.patient_state_v1.analyze_first import metrics

OUT=RUN/'boundary_sensitivity_v1_5'

def dataset(scenario):
    z=dict(np.load(RUN/'observations.npz'));ev=pd.read_csv(RUN/'events.csv');iv=json.loads((RUN/'seizures.json').read_text());keep=np.ones(len(ev),bool)
    post={'post15':15,'post60':60,'both60':60}.get(scenario,0);pre={'pre15':15,'pre60':60,'both60':60}.get(scenario,0)
    for s in iv:
        if post:keep&=~((ev.start_epoch.to_numpy()<s['offset']+post*60)&(ev.end_epoch.to_numpy()>s['offset']))
        if pre:keep&=~((ev.start_epoch.to_numpy()<s['onset'])&(ev.end_epoch.to_numpy()>s['onset']-pre*60))
    d={k:v[keep].copy() for k,v in z.items() if np.ndim(v)>0};d['dt']=np.r_[0,np.diff(d['t'])]
    if scenario=='carry_ictal':d['reset']=np.r_[True,np.zeros(len(d['t'])-1,bool)]
    elif scenario=='reset_coverage':
        group=ev.coverage_segment.to_numpy()[keep]*1000+d['epoch'];d['reset']=np.r_[True,np.diff(group)!=0]
    else:d['reset']=np.r_[True,np.diff(d['epoch'])!=0]
    d['dt'][d['reset']]=0;return d,keep

def worker(job):
    path=OUT/'fits'/f"{job['id']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        data,keep=dataset(job['scenario']);end=int(np.sum(keep[:job['original_end']]));d=slice_data(data,0,end)
        if len(d['y'])<500:raise ValueError('Insufficient retained training events')
        p=np.clip(d['y'].mean(),.001,.999);b=np.log(p/(1-p));tau=job['tau0']
        if job['model']=='ou':r=fit(d,'ou',np.r_[b,np.log(tau),np.log(.6)]);prediction=filter_adf(np.array(r['theta']),data)['predict_tb']
        else:
            d=history_data(d);data=history_data(data)
            def expand(t):return np.r_[t[:2],0,t[2:]]
            opt=minimize(lambda t:-laplace(expand(t),d,True),np.r_[b,1.,np.log(tau),np.log(.6)],method='L-BFGS-B',bounds=[(-8,8),(-8,8),(np.log(1/60),np.log(24)),(np.log(.01),np.log(5))],options={'maxiter':150,'ftol':1e-10,'eps':2e-5})
            r=dict(model='ou_history',theta=opt.x,loglik=-opt.fun,success=opt.success,tau_hours=np.exp(opt.x[-2]),stationary_sd=np.exp(opt.x[-1]));prediction=filter_adf(expand(opt.x),data,True)['predict_tb']
        r.update(status='COMPLETE',job=job,n_retained=len(data['y']),n_train=end,n_reset=int(data['reset'].sum()),elapsed=time.time()-start)
        if job['scope']!='full':
            lo=end;hi=int(np.sum(keep[:job['original_test_end']]));r['forward']=metrics(prediction[lo:hi],data['y'][lo:hi],data['n'][lo:hi])
    except Exception:r=dict(status='FAILED',job=job,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,r);return r

def main():
    z=np.load(RUN/'observations.npz');folds=json.loads((RUN/'splits.json').read_text());scopes=[dict(scope='full',original_end=len(z['y']))]+[dict(scope=f"fold{f['fold']}",original_end=f['train_end'],original_test_end=f['test_end']) for f in folds]
    jobs=[dict(id=f"{scenario}_{scope['scope']}_{model}_{i}",scenario=scenario,model=model,tau0=tau,**scope) for scenario in ('post15','post60','pre15','pre60','both60','carry_ictal','reset_coverage') for scope in scopes for model in ('ou','ou_history') for i,tau in enumerate((.15,2.))]
    write_json(OUT/'queue.json',jobs)
    with ProcessPoolExecutor(max_workers=24) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),id=r['job']['id'],status=r['status'],tau=r.get('tau_hours'),n_retained=r.get('n_retained'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs),finished_unix=time.time()))

if __name__=='__main__':main()
