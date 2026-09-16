"""Known-truth test of treating state-dependent event times as external inputs.

Same stationary OU drives label probability and event rate. This is an
identifiability/observation-model assay, not another patient mechanism fit.
"""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.generate import simulate
from scripts.patient_state_v1.model import fit as fit_mark
from scripts.patient_state_v1.joint_counts import marginal
from scripts.patient_state_v1.renewal import observation
OUT=RUN/'informative_timing_calibration_v1_18'

def bins(ts,y,hours,deadtime=.25):
    edges=np.arange(int(hours*3600)+1,dtype=float);n=np.histogram(ts,edges)[0];yy=np.histogram(ts,edges,weights=np.asarray(y,np.int64))[0]
    def busy(x):
        k=np.searchsorted(ts+deadtime,x,side='right');v=k*deadtime;valid=k<len(ts);v[valid]+=np.clip(x[valid]-ts[k[valid]],0,deadtime);return v
    risk=1-np.diff(busy(edges));assert risk.min()>-1e-7;d=dict(dt=np.r_[0,np.full(len(n)-1,1/3600)],reset=np.r_[True,np.zeros(len(n)-1,bool)],n=n.astype(float),y=yy.astype(float),exposure=np.maximum(risk,0)/3600);assert n.sum()==len(ts) and yy.sum()==y.sum();return d

def worker(j):
    path=OUT/'fits'/f"c{j['c']}_rep{j['rep']:03d}_{j['kind']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        sd=.6;tau=.5;b=-.8;nominal=600.;hazard=nominal/(1-nominal*.25/3600);a=np.log(hazard)-.5*j['c']**2*sd**2;ts,y,_,_,_=simulate(hours=96,step=1.,seed=j['seed'],a=a,b=b,tau_r=1.,sd_r=0.,tau_s=tau,sd_s=sd,c=j['c'],kind=1,gamma=0.,deadtime=.25);hours=96 if j['kind']=='mark96' else 24;keep=ts<hours*3600;ts=ts[keep];y=y[keep].astype(np.int64);r=[];initial_b=np.log((y.mean()+1e-6)/(1-y.mean()+1e-6))
        if j['kind'].startswith('mark'):
            t=ts/3600;d=dict(t=t,dt=np.r_[0,np.diff(t)],reset=np.r_[True,np.zeros(len(t)-1,bool)],n=np.ones(len(t)),y=y,x=np.ones((len(t),1)))
            for t0 in [.2,2.]:r.append(fit_mark(d,'ou',[initial_b,np.log(t0),np.log(.6)]))
        else:
            d=bins(ts,y,hours)
            def fun(t):return -marginal(np.r_[t[0],a,j['c'],t[1:],0.],d,'shared',observation_fn=observation)
            for t0 in [.2,2.]:
                opt=minimize(fun,[initial_b,np.log(t0),np.log(.6)],method='L-BFGS-B',bounds=[(-8,8),(np.log(1/60),np.log(24)),(np.log(.01),np.log(5))],options={'maxiter':120,'ftol':1e-10,'eps':1e-5,'maxls':25});r.append(dict(theta=opt.x,loglik=-opt.fun,success=opt.success,message=opt.message))
        result=dict(max(r,key=lambda x:x['loglik']));result.update(status='COMPLETE',job=j,truth=dict(b=b,tau_hours=tau,sd=sd,rate_logbaseline=a,c=j['c']),n_events=len(ts),hours=hours,tb_fraction=float(y.mean()),initial_fits=r,elapsed=time.time()-start)
    except Exception:result=dict(status='FAILED',job=j,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,result);return result

def main():
    # Independent-rate factor must add a parameter-independent constant.
    ts=np.array([.1,1.3,2.8,5.1,5.6,8.2]);y=np.array([0,1,0,1,1,0]);d=bins(ts,y,10/3600);theta=np.array([-.8,np.log(.5),np.log(.6)]);a=np.log(600);m=marginal(theta,d,'mark',observation_fn=observation);s=marginal(np.r_[theta[0],a,0,theta[1:],0],d,'shared',observation_fn=observation);expected=d['n'].sum()*a-d['exposure'].sum()*np.exp(a);assert abs(s-m-expected)<1e-7
    write_json(OUT/'numerical_canary.json',dict(status='PASS',independent_rate_constant_error=abs(s-m-expected),observed_event_and_label_conservation=True));jobs=[dict(c=c,rep=rep,kind=kind,seed=530000+k*1000+rep) for k,c in enumerate([-1.5,0.,1.5]) for rep in range(32) for kind in ['mark24','mark96','joint24']]
    write_json(OUT/'contract.json',dict(question='Can informative event timing distort mark-only estimates of a stationary latent process?',truth='One stationary OU: tau=.5h, SD=.6, baseline log-odds=-.8; labels sigmoid(b+s); rate exp(a+c*s)',couplings=[-1.5,0,1.5],replicates=32,simulation='96h, 1-second exact OU updates and piecewise-constant conditional intensity; 250ms shifted-event support',fits='Mark-only exact-time OU on nested 24/96h prefixes; joint 24h count/mark likelihood with known true rate baseline and coupling',control='c=0 decouples event occurrence and mode state; joint likelihood adds a state-independent rate factor',limits='Synthetic observation-model calibration; no patient parameter or neural mechanism claim. Joint comparator deliberately knows the observation mapping to isolate omission of event-time information.',n_jobs=len(jobs)))
    with ProcessPoolExecutor(max_workers=24) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(complete=i+1,total=len(jobs),job=r['job'],status=r['status'],elapsed=r['elapsed'],success=r.get('success'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs)))
if __name__=='__main__':main()
