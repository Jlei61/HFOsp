"""Activity-dependent speed of a continuous mode state; not activity bias.

dx=-k(t)*x/tau dt + sigma*sqrt(2*k(t)/tau) dW,
k=exp(beta*clip(log(recent completed total-event rate/training rate),-4,4)).
The OU transition uses integrated operational time. State amplitude is unchanged.
Missing coverage uses neutral speed; events update the covariate, never reset x.
"""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.integrate import quad
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import laplace,filter_adf,slice_data
from scripts.patient_state_v1.advanced_controls import history_data
from scripts.patient_state_v1.analyze_first import best_fits
from scripts.patient_state_v1.review_predictions import selected
OUT=RUN/'activity_clock_v1_27'

@njit(cache=True)
def integral(q,length,hours,beta):
    if beta==0.:return length
    knots=np.sort(np.array([0.,min(max((q-4)*hours,0.),length),min(max((q+4)*hours,0.),length),length]));out=0.
    for k in range(3):
        lo,hi=knots[k],knots[k+1]
        if hi<=lo:continue
        middle=q-(lo+hi)/2/hours
        if middle>=4:out+=(hi-lo)*np.exp(4*beta)
        elif middle<=-4:out+=(hi-lo)*np.exp(-4*beta)
        else:out+=np.exp(beta*(q-lo/hours))*(-np.expm1(-beta*(hi-lo)/hours))*hours/beta
    return out

@njit(cache=True)
def before_rates(t,group,baseline,hours):
    rate=np.empty(len(t));rate[0]=baseline;delay=.25/3600
    for i in range(1,len(t)):
        if group[i]!=group[i-1]:rate[i]=baseline
        else:
            dt=t[i]-t[i-1];rate[i]=max(rate[i-1]*np.exp(-dt/hours)+np.exp(-(dt-delay)/hours)/hours,1e-300)
    return rate

@njit(cache=True)
def operational_dt(dt,reset,group,rate,baseline,hours,beta):
    if beta==0.:return dt.copy()
    out=dt.copy();delay=.25/3600
    for i in range(1,len(dt)):
        if reset[i] or group[i]!=group[i-1]:continue
        length=dt[i];first=min(length,delay);q=np.log(rate[i-1]/baseline);out[i]=integral(q,first,hours,beta)
        if length>delay:
            post=rate[i-1]*np.exp(-delay/hours)+1/hours;out[i]+=integral(np.log(post/baseline),length-delay,hours,beta)
    return out

def dataset(end,history):
    d=dict(np.load(RUN/'observations.npz'));ev=pd.read_csv(RUN/'events.csv');ex=pd.read_csv(RUN/'exposure.csv');cut=ev.end_epoch.iloc[end-1];hours=np.maximum(0,np.minimum(ex.end_epoch.to_numpy(),cut)-ex.start_epoch.to_numpy()).sum()/3600;baseline=end/hours;group=ev.coverage_segment.to_numpy()*1000+d['epoch'];rate=before_rates(d['t'],group,baseline,1/60);d=history_data(d) if history else d;d['group']=group;d['recent_rate']=rate;return d,dict(training_rate_per_hour=baseline,history_window_hours=1/60,completion_delay_seconds=.25)

def warped(d,beta,info):
    q=dict(d);q['dt']=operational_dt(d['dt'],d['reset'],d['group'],d['recent_rate'],info['training_rate_per_hour'],info['history_window_hours'],beta);return q

def expand(t,history):return np.r_[t[:2],0.,t[2:4]] if history else t[:3]

def worker(j):
    path=OUT/'fits'/f"{j['id']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        d,info=dataset(j['end'],j['history']);train=slice_data(d,0,j['end']);base=selected(RUN/'advanced_controls_v1_2/fits',j['scope'],'ou_history')[0]['theta'] if j['history'] else best_fits()[j['scope'],'ou']['theta'];init=np.r_[base,j['beta0']];init[-3]=np.log(j['tau0']);bounds=[(-8,8)]*(2 if j['history'] else 1)+[(np.log(1/60),np.log(24)),(np.log(.01),np.log(5)),(-2,2)]
        def fun(t):return -laplace(expand(t,j['history']),warped(train,t[-1],info),j['history'])+.5*t[-1]**2
        opt=minimize(fun,init,method='L-BFGS-B',bounds=bounds,options={'maxiter':200,'ftol':1e-10,'eps':2e-5,'maxls':25});w=warped(d,opt.x[-1],info);f=filter_adf(expand(opt.x,j['history']),w,j['history']);r=dict(status='COMPLETE',job=j,theta=opt.x,penalized_loglik=-opt.fun,laplace_loglik=-opt.fun+.5*opt.x[-1]**2,success=bool(opt.success),message=str(opt.message),nfev=int(opt.nfev),elapsed=time.time()-start,covariate_info=info);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**f,operational_dt=w['dt']);write_json(path,r)
    except Exception:r=dict(status='FAILED',job=j,elapsed=time.time()-start,traceback=traceback.format_exc());write_json(path,r)
    return r

def canary():
    rng=np.random.default_rng(17291);errs=[]
    for i in range(40):
        q=rng.uniform(-6,6);length=np.exp(rng.uniform(np.log(1e-5),np.log(5)));beta=rng.uniform(-2,2);h=1/60;points=[v for v in [(q-4)*h,(q+4)*h] if 0<v<length];reference=quad(lambda t:np.exp(beta*np.clip(q-t/h,-4,4)),0,length,points=points,epsabs=1e-10)[0];err=abs(integral(q,length,h,beta)-reference)/max(abs(reference),1e-12);errs.append(err)
    assert max(errs)<1e-8;d,info=dataset(28125,True);a=warped(d,0.,info);assert np.array_equal(a['dt'],d['dt']);prefix=before_rates(d['t'][:1000],d['group'][:1000],info['training_rate_per_hour'],1/60);assert np.array_equal(prefix,d['recent_rate'][:1000]);write_json(OUT/'numerical_canary.json',dict(status='PASS',max_integral_relative_error=max(errs),zero_clock_exact_dt_identity=True,completed_history_prefix_identity=True))

def main():
    canary();d=np.load(RUN/'observations.npz');folds=json.loads((RUN/'splits.json').read_text());scopes=[dict(scope='full',end=len(d['y']))]+[dict(scope=f"fold{f['fold']}",end=f['train_end'],test_end=f['test_end']) for f in folds];jobs=[]
    for scope in scopes:
        for history in [False,True]:
            for i,(tau,beta) in enumerate([(.2,-.5),(.2,.5),(2.,-.5),(2.,.5)]):jobs.append(dict(id=f"{scope['scope']}_h{int(history)}_{i}",history=history,tau0=tau,beta0=beta,**scope))
    write_json(OUT/'contract.json',dict(question='Does variation in mode-state speed with recent total activity explain changing fitted physical-time constants?',equation='dx=-k(t)*x/tau dt + stationary_sd*sqrt(2*k(t)/tau) dW; k=exp(beta*clip(log(recent_rate/training_rate),-4,4))',observation='Same logistic marks, optionally same one-second mark history in physical time; activity changes speed, not baseline preference',extra_parameters=1,beta_prior='Normal(0,1) penalty; beta in [-2,2]',activity='One-minute exponentially decayed completed total-event count rate; training-only baseline; 250ms completion delay; exact integral of clipped-log rate between events',missing='Neutral operational speed across coverage gaps; activity initializes at training baseline when coverage resumes; ictal prior boundaries inherited, not physical reset',n_jobs=32,limits='Conditional event-time model with a predictable data covariate, not evidence of neural causal feedback or Z/M interaction; no seizure outcome fitted'))
    write_json(OUT/'queue.json',jobs)
    with ProcessPoolExecutor(max_workers=20) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(done=i+1,total=32,id=r['job']['id'],status=r['status'],success=r.get('success'),theta=r.get('theta'),elapsed=r['elapsed']),default=lambda x:x.tolist()),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=32))

if __name__=='__main__':main()
