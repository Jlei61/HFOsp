"""Two continuous latent drivers for retained event times and marks.

Mode state s and residual activity r have independent OU innovations.
lambda=exp(a+r+c*s), P(TB)=sigmoid(b+s). Coupling c is an observation
association, not neural feedback. The same observed 250-ms packing support
is approximated by a shifted renewal likelihood; ictal gaps are excluded.
"""
import sys,json,time,traceback,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.special import expit
from scipy.linalg import solveh_banded,cholesky_banded
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import prior,qmul
from scripts.patient_state_v1.renewal import prepare,observation
from scripts.patient_state_v1.joint_counts import marginal
OUT=RUN/'joint_two_state_v1_20'

def unpack(t,coupled):
    b,a=t[:2];c=t[2] if coupled else 0.;ts,ss,tr,sr=np.exp(t[3:] if coupled else t[2:]);return b,a,c,ts,ss,tr,sr

def objective_parts(state,t,d,coupled):
    b,a,c,*_=unpack(t,coupled);s,r=state.T;eta=b+s;p=expit(eta);u=a+r+c*s;mu=d['exposure']*np.exp(np.clip(u,-700,60));n,y=d['n'],d['y']
    value=np.sum(n*np.logaddexp(0,eta)-y*eta+mu-n*u)
    g=np.column_stack([n*p-y+c*(mu-n),mu-n]);h=np.column_stack([n*p*(1-p)+c*c*mu,mu,c*mu]);return value,g,h

def marginal2(t,d,coupled,state=False):
    b,a,c,ts,ss,tr,sr=unpack(t,coupled);_,_,ds,os,lds=prior(d['dt'],d['reset'],ts,ss);_,_,dr,orr,ldr=prior(d['dt'],d['reset'],tr,sr);N=len(d['n']);s=np.zeros((N,2));band=np.zeros((3,2*N));band[2,:-2]=np.column_stack([os,orr]).ravel()
    def value(x):return objective_parts(x,t,d,coupled)[0]+.5*(x[:,0]@qmul(x[:,0],ds,os)+x[:,1]@qmul(x[:,1],dr,orr))
    val=value(s);converged=False
    for it in range(80):
        _,g,h=objective_parts(s,t,d,coupled);g[:,0]+=qmul(s[:,0],ds,os);g[:,1]+=qmul(s[:,1],dr,orr);band[0]=np.column_stack([ds+h[:,0],dr+h[:,1]]).ravel();band[1,::2]=h[:,2];delta=solveh_banded(band,g.ravel(),lower=True,check_finite=False).reshape(N,2);step=1.
        for _ in range(25):
            trial=s-step*delta;v=value(trial)
            if v<=val+1e-8:break
            step*=.5
        change=abs(val-v);s=trial;val=v
        if np.max(abs(step*delta))<1e-6 or (change<1e-8 and np.max(abs(delta))<1e-3):converged=True;break
    _,_,h=objective_parts(s,t,d,coupled);band[0]=np.column_stack([ds+h[:,0],dr+h[:,1]]).ravel();band[1,::2]=h[:,2];chol=cholesky_banded(band,lower=True,check_finite=False);ll=-val+.5*(lds+ldr-2*np.log(chol[0]).sum())
    if state:return dict(loglik=float(ll),mode=s,converged=converged,iterations=it+1)
    return float(ll) if converged else float(ll)-1000

def fit_joint(d,coupled,initial):
    bounds=[(-8,8),(np.log(.01),np.log(1e7))]+([(-5,5)] if coupled else [])+[(np.log(1/60),np.log(24)),(np.log(.01),np.log(5)),(np.log(1/3600),np.log(24)),(np.log(.01),np.log(5))]
    def fun(t):
        try:return -marginal2(t,d,coupled)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError):return 1e20
    opt=minimize(fun,initial,method='L-BFGS-B',bounds=bounds,options={'maxiter':180,'ftol':1e-10,'eps':2e-5,'maxls':25});return dict(theta=opt.x,loglik=-opt.fun,success=bool(opt.success),message=str(opt.message),nfev=opt.nfev)

def canary():
    rng=np.random.default_rng(6621);N=31;dt=np.r_[0,rng.uniform(.001,.01,N-1)];reset=np.zeros(N,bool);reset[[0,14]]=True;dt[reset]=0;n=rng.integers(0,4,N);y=rng.binomial(n,.3);d=dict(dt=dt,reset=reset,n=n,y=y,exposure=np.full(N,.002));t=np.array([-.8,5.,np.log(.3),np.log(.6),np.log(.1),np.log(1.)]);j=marginal2(t,d,False);m=marginal(np.r_[t[0],t[2:4]],d,'mark',observation_fn=observation);r=marginal(np.r_[t[1],t[4:],0.],d,'rate',observation_fn=observation);assert abs(j-m-r)<1e-6
    tc=np.r_[t[:2],.7,t[2:]];x=rng.normal(0,.3,(N,2));v,g,h=objective_parts(x,tc,d,True);eps=1e-5;err=0.
    for i in range(N):
        for k in range(2):
            xp=x.copy();xm=x.copy();xp[i,k]+=eps;xm[i,k]-=eps;vp,gp,_=objective_parts(xp,tc,d,True);vm,gm,_=objective_parts(xm,tc,d,True);err=max(err,abs((vp-vm)/(2*eps)-g[i,k]));expected=np.array([h[i,0],h[i,2]]) if k==0 else np.array([h[i,2],h[i,1]]);err=max(err,float(np.max(abs((gp[i]-gm[i])/(2*eps)-expected))))
    assert err<1e-6;write_json(OUT/'numerical_canary.json',dict(status='PASS',independent_factorization_error=abs(j-m-r),gradient_hessian_max_error=err))

def worker(j):
    p=OUT/'fits'/f"{j['id']}.json"
    if p.exists():return json.loads(p.read_text())
    start=time.time()
    try:
        full=prepare(j['seconds'],.25);d={k:v[:j['end']].copy() for k,v in full.items() if np.ndim(v)>0};q=d['y'].sum()/d['n'].sum();b=np.log(q/(1-q));a=np.log(d['n'].sum()/d['exposure'].sum());init=np.r_[b,a,([j['c0']] if j['coupled'] else []),np.log(j['ts0']),np.log(.6),np.log(.1),np.log(1.5)];r=fit_joint(d,j['coupled'],init);s=marginal2(r['theta'],d,j['coupled'],True);r.update(status='COMPLETE',job=j,elapsed=time.time()-start,state_converged=s['converged']);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p.with_suffix('.npz'),**s)
    except Exception:r=dict(status='FAILED',job=j,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(p,r);return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=int,nargs='+',default=[5,15]);ap.add_argument('--workers',type=int,default=20);args=ap.parse_args();canary();folds=json.loads((RUN/'splits.json').read_text());inv=json.loads((RUN/'seizures.json').read_text());ev=np.load(RUN/'observations.npz');jobs=[]
    for sec in args.seconds:
        d=prepare(sec,.25);scopes=[dict(scope='full',end=len(d['n']))]
        for f in folds:
            cut=inv[int(ev['epoch'][f['test_start']])-1]['offset'];scopes.append(dict(scope=f"fold{f['fold']}",end=int(np.searchsorted(d['hi'],cut,side='right')),cutoff_epoch=cut))
        for scope in scopes:
            for coupled in [False,True]:
                for k,(ts,c) in enumerate([(.2,-.5),(.6,0.),(2.,.5)]):jobs.append(dict(id=f"b{sec}_{scope['scope']}_c{int(coupled)}_{k}",seconds=sec,coupled=coupled,ts0=ts,c0=c,**scope))
    write_json(OUT/f"contract_{'_'.join(map(str,args.seconds))}.json",dict(question='Does separating residual total activity from mode preference support a reproducible state-to-rate association?',equations='ds=-s/tau_s dt+sigma_s dW_s; dr=-r/tau_r dt+sigma_r dW_r; lambda=exp(a+r+c*s); pTB=sigmoid(b+s)',independent_control='c=0, factorized latent rate and mark likelihoods',likelihood='Effective shifted renewal event density plus binary marks; state constant inside the selected integration bins',bin_seconds=args.seconds,no_neural_claim='Independent latent innovations, observation coupling only; no SNN feedback or Z/M',fit='Laplace marginal likelihood; three training-only starts; strict pre-interval training cutoffs',limits='Packing approximation and Laplace errors require checking before accepting parameter interpretation; no seizure type fitted',n_jobs=len(jobs)))
    write_json(OUT/f"queue_{'_'.join(map(str,args.seconds))}.json",jobs)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(done=i+1,total=len(jobs),id=r['job']['id'],status=r['status'],elapsed=r['elapsed'],success=r.get('success'))),flush=True)
    write_json(OUT/f"status_{'_'.join(map(str,args.seconds))}.json",dict(status='COMPLETE',n_jobs=len(jobs)))
if __name__=='__main__':main()
