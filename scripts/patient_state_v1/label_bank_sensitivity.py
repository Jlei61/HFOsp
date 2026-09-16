"""Repeat state inference with historical masked-rank adaptive-cluster labels.

Event identities, clinical exclusions, exposure and folds remain identical.
This is label-definition sensitivity; the primary Timing+Space bank is unchanged.
"""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import ROOT,RUN,write_json
from scripts.patient_state_v1.model import fit,laplace,filter_adf,slice_data
from scripts.patient_state_v1.advanced_controls import history_data
from scripts.patient_state_v1.free_timescale import fit_free
from scripts.patient_state_v1.two_timescale import filter2
OUT=RUN/'label_bank_sensitivity_v1_19'

def prepare():
    z=np.load(ROOT/'results/topic5_preseizure_template_share/epilepsiae_1146/event_index.npz');ev=pd.read_csv(RUN/'events.csv');order=np.argsort(z['event_abs_time']);t=z['event_abs_time'][order];ix=np.searchsorted(t,ev.start_epoch.to_numpy());assert np.max(abs(t[ix]-ev.start_epoch.to_numpy()))<1e-5 and len(np.unique(ix))==len(ev);y=z['template_label'][order[ix]];assert set(np.unique(y))=={0,1};d=dict(np.load(RUN/'observations.npz'));primary=d['y'].copy();assert np.all(d['n']==1);d['y']=y.astype(np.int64);OUT.mkdir(exist_ok=True);np.savez_compressed(OUT/'observations_timing_only.npz',**d);write_json(OUT/'label_crosswalk.json',dict(status='COMPLETE',n=len(y),agreement=float(np.mean(y==primary)),table=pd.crosstab(primary,y).to_dict(),primary='Current Timing+Space frozen labels',sensitivity='Historical masked-rank adaptive-cluster template_label',all_current_interictal_ids_matched=True))

def worker(j):
    p=OUT/'fits'/f"{j['id']}.json"
    if p.exists():return json.loads(p.read_text())
    start=time.time()
    try:
        d=dict(np.load(OUT/'observations_timing_only.npz'))
        if j['model']!='ou':d=history_data(d)
        if j['model']=='two_ou_history':d['slow_dt']=np.r_[0,np.diff(d['t'])];d['slow_reset']=np.r_[True,np.zeros(len(d['t'])-1,bool)]
        train=slice_data(d,0,j['end']);q=train['y'].mean();b=np.log(q/(1-q))
        if j['model']=='ou':r=fit(train,'ou',[b,np.log(j['tau0']),np.log(.6)]);f=filter_adf(np.array(r['theta']),d)
        elif j['model']=='ou_history':
            def expand(t):return np.r_[t[:2],0.,t[2:]]
            o=minimize(lambda t:-laplace(expand(t),train,True),[b,.8,np.log(j['tau0']),np.log(.6)],method='L-BFGS-B',bounds=[(-8,8),(-8,8),(np.log(1/60),np.log(24)),(np.log(.01),np.log(5))],options={'maxiter':150,'ftol':1e-10,'eps':2e-5});r=dict(theta=o.x,loglik=-o.fun,success=o.success,message=o.message);f=filter_adf(expand(o.x),d,True)
        else:
            r=fit_free(train,True,'adf',np.array([b,.7,np.log(.003 if j['tau0']<1 else .03),np.log(.4),np.log(.6),np.log(j['tau0']*5)]));f=filter2(np.array(r['theta'][:-1]),d,np.exp(r['theta'][-1]),True)
        r.update(status='COMPLETE',job=j,elapsed=time.time()-start);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p.with_suffix('.npz'),**f)
    except Exception:r=dict(status='FAILED',job=j,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(p,r);return r

def main():
    prepare();d=np.load(OUT/'observations_timing_only.npz');folds=json.loads((RUN/'splits.json').read_text());scopes=[dict(scope='full',end=len(d['y']))]+[dict(scope=f"fold{f['fold']}",end=f['train_end'],test_end=f['test_end']) for f in folds];jobs=[]
    for scope in scopes:
        for model in ['ou','ou_history','two_ou_history']:
            for i,tau in enumerate([.2,2.]):jobs.append(dict(id=f"{scope['scope']}_{model}_{i}",model=model,tau0=tau,**scope))
    write_json(OUT/'contract.json',dict(question='How sensitive are inferred state parameters to the frozen propagation-label definition?',primary_bank_unchanged=True,sensitivity_bank='Historical masked-rank adaptive-cluster labels, same event IDs and current clinical exclusions',models=['single OU','OU plus 1-second history','free two-OU plus history with background carry'],selection='Training likelihood only; no seizure type or SNN candidate performance used',scope='Definition sensitivity, not independent data replication',n_jobs=len(jobs)))
    with ProcessPoolExecutor(max_workers=24) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),id=r['job']['id'],status=r['status'],elapsed=r['elapsed'],success=r.get('success'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs)))
if __name__=='__main__':main()
