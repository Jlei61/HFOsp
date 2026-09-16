"""v1.3 effective renewal observation model for retained, nonoverlapping events.

The 250-ms exclusion is an observed packing constraint, not a biological refractory
period or an exact reconstruction of candidate rejection. Latent paths are constant
within quadrature bins; progressively refined 15/5/1-second integration is audited.
"""
import sys,json,time,traceback,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.joint_counts import prepare as prepare_bins,marginal,unpack

OUT=RUN/'renewal_observation_v1_3'

def prepare(seconds,deadtime):
    path=OUT/f'bins_{seconds}s_d{deadtime}.npz'
    if path.exists():return dict(np.load(path))
    d=prepare_bins(seconds);ev=pd.read_csv(RUN/'events.csv');starts=ev.start_epoch.to_numpy();ends=starts+deadtime
    def busy(x):
        k=np.searchsorted(ends,x,side='right');total=k*deadtime;valid=k<len(starts)
        total[valid]+=np.clip(x[valid]-starts[k[valid]],0,deadtime);return total
    busy_seconds=busy(d['hi'])-busy(d['lo']);physical=d['exposure'].copy();risk=physical-busy_seconds/3600
    assert risk.min()>-1e-8
    risk=np.maximum(risk,0);d.update(exposure=risk,physical_exposure=physical,deadtime_seconds=np.array(deadtime),bin_seconds=np.array(seconds))
    OUT.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**d);return d

def expand(t,model):return t if model=='mark' else np.r_[t,0.]

def observation(s,theta,d,model):
    b,a,c,tau,sd,k=unpack(theta,model);n,y,risk=d['n'],d['y'],d['exposure'];ll=np.zeros_like(s);g=np.zeros_like(s);h=np.zeros_like(s)
    if model!='rate':
        eta=b+s;p=expit(eta);ll+=y*eta-n*np.logaddexp(0,eta);g+=n*p-y;h+=n*p*(1-p)
    if model!='mark':
        eta=a+c*s;expected=risk*np.exp(np.clip(eta,-700,50));ll+=n*eta-expected;g+=c*(expected-n);h+=c*c*expected
    return ll,g,h

def fit_model(d,model,init,maxiter=180):
    tb=(np.log(1/60),np.log(24));sb=(np.log(.01),np.log(5));rb=(np.log(.01),np.log(1e7))
    bounds={'mark':[(-8,8),tb,sb],'rate':[rb,tb,sb],'shared':[(-8,8),rb,(-12,12),tb,sb]}[model]
    def fun(t):
        try:return -marginal(expand(t,model),d,model,observation_fn=observation)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError):return 1e20
    r=minimize(fun,init,method='L-BFGS-B',bounds=bounds,options={'maxiter':maxiter,'ftol':1e-10,'eps':1e-5,'maxls':25})
    return dict(model=model,theta=r.x,loglik=-r.fun,success=r.success,message=r.message,nfev=r.nfev)

def worker(job):
    path=OUT/'fits'/f"{job['id']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        full=prepare(job['seconds'],job['deadtime']);d={k:v[:job['end']].copy() for k,v in full.items() if np.ndim(v)>0};p=np.clip(d['y'].sum()/d['n'].sum(),.001,.999);b=np.log(p/(1-p));a=np.log(d['n'].sum()/d['exposure'].sum());tau,sd,c=job['tau'],job['sd'],job['c']
        init={'mark':[b,np.log(tau),np.log(sd)],'rate':[a,np.log(tau),np.log(sd)],'shared':[b,a,c,np.log(tau),np.log(sd)]}[job['model']]
        result=fit_model(d,job['model'],init);result.update(status='COMPLETE',job=job,elapsed=time.time()-start)
    except Exception:result=dict(status='FAILED',job=job,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=20);ap.add_argument('--wave',choices=['coarse','fine'],default='coarse');args=ap.parse_args();jobs=[];folds=json.loads((RUN/'splits.json').read_text());eventdata=np.load(RUN/'observations.npz')
    for seconds in ([15,5] if args.wave=='coarse' else [1]):
        for deadtime in ([0.,.25] if seconds==15 else [.25]):
            d=prepare(seconds,deadtime);scopes=[('full',len(d['n']))]
            if deadtime==.25:
                for f in folds:
                    cutoff=float(eventdata['origin_epoch'])+eventdata['t'][f['train_end']]*3600;scopes.append((f"fold{f['fold']}",int(np.searchsorted(d['hi'],cutoff,side='right'))))
            for scope,end in scopes:
                for model in ('mark','rate','shared'):
                    initial=[(.15,.6,2.),(1.,1.5,-2.)] if model=='shared' else [(.2,.6,1.),(2.,2.,1.)]
                    for i,(tau,sd,c) in enumerate(initial):jobs.append(dict(id=f'b{seconds}_d{deadtime}_{scope}_{model}_{i}',seconds=seconds,deadtime=deadtime,scope=scope,end=end,model=model,tau=tau,sd=sd,c=c))
    write_json(OUT/f'{args.wave}_queue.json',jobs)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),id=r['job']['id'],status=r['status'],elapsed=r['elapsed'],loglik=r.get('loglik'))),flush=True)
    write_json(OUT/f'{args.wave}_status.json',dict(status='COMPLETE',n_jobs=len(jobs),finished_unix=time.time()))

if __name__=='__main__':main()
