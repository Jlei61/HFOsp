"""Fit simple memory controls to the same future-label forecasting task."""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from numba import njit
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
OUT=RUN/'horizon_memory_controls_v1_17'
@njit(cache=True)
def predict(y,dt,reset,target,previous,elapsed,same,theta):
    tau=np.exp(theta[0]);strength=np.exp(theta[1]);p0=1/(1+np.exp(-theta[2]));aa=np.empty(len(y));bb=np.empty(len(y));a=0.;b=0.
    for i in range(len(y)):
        if reset[i]:a=0.;b=0.
        else:
            rho=np.exp(-dt[i]/tau);a*=rho;b*=rho
        a+=y[i];b+=1-y[i];aa[i]=a;bb[i]=b
    pp=np.empty(len(target));ll=0.
    for i in range(len(target)):
        if same[i]:rho=np.exp(-elapsed[i]/tau);a=aa[previous[i]]*rho;b=bb[previous[i]]*rho
        else:a=0.;b=0.
        p=(a+strength*p0)/(a+b+strength);p=min(max(p,1e-12),1-1e-12);pp[i]=p;ll+=y[target[i]]*np.log(p)+(1-y[target[i]])*np.log1p(-p)
    return pp,ll

def indices(d,ev,iv,lo,hi,horizon,kind,train_cut):
    starts=ev.start_epoch.to_numpy();ends=ev.end_epoch.to_numpy();offset=np.array([s['offset'] for s in iv]);target=np.arange(lo,hi);cut=starts[target]-horizon*60;valid=(cut>=train_cut)&(np.searchsorted(offset,cut,side='right')==d['epoch'][target]);target=target[valid];cut=cut[valid];prev=np.searchsorted(ends,cut,side='right')-1;valid=prev>=0;target=target[valid];cut=cut[valid];prev=prev[valid];elapsed=((starts[target] if kind=='decay' else cut)-starts[prev])/3600;same=d['epoch'][prev]==d['epoch'][target];return target,prev,elapsed,same

def worker(j):
    p=OUT/'fits'/f"{j['id']}.json"
    if p.exists():return json.loads(p.read_text())
    start=time.time()
    try:
        d=np.load(RUN/'observations.npz');ev=pd.read_csv(RUN/'events.csv');iv=json.loads((RUN/'seizures.json').read_text());train=indices(d,ev,iv,0,j['end'],j['horizon'],j['kind'],ev.end_epoch.iloc[0]);yy=d['y'][:j['end']];dt=d['dt'][:j['end']];reset=d['reset'][:j['end']];p0=yy.mean();initial=j.get('initial',[np.log(j['tau0']),np.log(10),np.log(p0/(1-p0))]);bounds=[(np.log(1/3600),np.log(24)),(np.log(.01),np.log(10000)),(-8,8)]
        o=minimize(lambda t:-predict(yy,dt,reset,*train,t)[1],initial,method='L-BFGS-B',bounds=bounds,options={'maxiter':120,'ftol':1e-10,'eps':1e-5});test=indices(d,ev,iv,j['end'],j['test_end'],j['horizon'],j['kind'],ev.end_epoch.iloc[j['end']-1]);pp,ll=predict(d['y'],d['dt'],d['reset'],*test,o.x);r=dict(status='COMPLETE',job=j,theta=o.x,train_loglik=-o.fun,success=o.success,message=o.message,test_loglik=ll,n_train=len(train[0]),n_test=len(test[0]),elapsed=time.time()-start);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p.with_suffix('.npz'),index=test[0],predict_tb=pp)
    except Exception:r=dict(status='FAILED',job=j,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(p,r);return r

def main():
    folds=json.loads((RUN/'splits.json').read_text());jobs=[]
    for fold in folds:
        for h in [0,1,5,15,60,120]:
            for kind in ['decay','frozen']:
                for i,tau in enumerate([.02,.2,2.]):jobs.append(dict(id=f"fold{fold['fold']}_h{h}_{kind}_{i}",fold=fold['fold'],end=fold['train_end'],test_end=fold['test_end'],horizon=h,kind=kind,tau0=tau))
    write_json(OUT/'contract.json',dict(question='Does a continuous latent model beat simple memory trained for the same forecast horizon?',horizons_minutes=[0,1,5,15,60,120],controls='Exponentially weighted previous mark counts with shrinkage; freeze at cutoff or decay to target',training='Only targets and prior events within each training prefix; three optimizer starts; no test outcome used',comparison='Same 13,184 held-out events eligible at every horizon; subsequent analysis pairs by event identity',not_claimed='The heuristic is not a physical drift model or an autonomous event generator'))
    with ProcessPoolExecutor(max_workers=24) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),id=r['job']['id'],status=r['status'],elapsed=r['elapsed'],success=r.get('success'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs)))
if __name__=='__main__':main()
