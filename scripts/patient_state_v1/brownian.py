"""Non-leaky drift-diffusion alternative between known ictal exclusions.

ds = mu dt + sigma dW, common initial N(0,sd0^2) at interval boundary.
No absorption threshold or seizure-time likelihood is fitted here.
"""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from scipy.linalg import solveh_banded,cholesky_banded
from numpy.polynomial.hermite import hermgauss
from numba import njit
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import qmul,slice_data
from scripts.patient_state_v1.analyze_first import metrics

OUT=RUN/'brownian_drift_v1_5'

def data_with_age():
    d=dict(np.load(RUN/'observations.npz'));iv=json.loads((RUN/'seizures.json').read_text());bound=np.r_[float(d['origin_epoch']),[s['offset'] for s in iv]]
    d['age']=(float(d['origin_epoch'])+d['t']*3600-bound[d['epoch']])/3600;assert d['age'].min()>=0;return d

def marginal(t,d):
    b,mu,ls,li=t;sigma,sd0=np.exp([ls,li]);a=np.ones(len(d['y']));a[d['reset']]=0
    q=sigma*sigma*d['dt'];q[d['reset']]=sd0*sd0+sigma*sigma*d['age'][d['reset']];diag=1/q;diag[:-1]+=a[1:]**2/q[1:];off=-a[1:]/q[1:];ldq=-np.log(q).sum()
    y,n=d['y'],d['n'];eta0=b+mu*d['age'];s=np.zeros(len(y));band=np.zeros((2,len(y)));band[1,:-1]=off
    def f(s):return np.sum(n*np.logaddexp(0,eta0+s)-y*(eta0+s))+.5*s@qmul(s,diag,off)
    val=f(s)
    for _ in range(60):
        p=expit(eta0+s);band[0]=diag+n*p*(1-p);grad=qmul(s,diag,off)+n*p-y;delta=solveh_banded(band,grad,lower=True,check_finite=False);step=1.
        for _ in range(25):
            trial=s-step*delta;v=f(trial)
            if v<=val+1e-8:break
            step*=.5
        s=trial;val=v
        if max(abs(step*delta))<1e-6:break
    p=expit(eta0+s);band[0]=diag+n*p*(1-p);chol=cholesky_banded(band,lower=True,check_finite=False)
    return -val+.5*(ldq-2*np.log(chol[0]).sum())

@njit(cache=True)
def filter_kernel(t,dt,reset,age,y,n,nodes,w):
    b,mu,ls,li=t;sigma=np.exp(ls);sd0=np.exp(li);m=0.;v=sd0*sd0;pred=np.empty(len(y));state=np.empty(len(y));terms=np.empty(len(y))
    for i in range(len(y)):
        if reset[i]:m=mu*age[i];v=sd0*sd0+sigma*sigma*age[i]
        else:m+=mu*dt[i];v+=sigma*sigma*dt[i]
        z=m+np.sqrt(2*v)*nodes;norm=0.;m1=0.;m2=0.;pp=0.
        for j in range(len(nodes)):
            eta=b+z[j];p=1/(1+np.exp(-eta));pp+=w[j]*p;lp=y[i]*eta-n[i]*(max(eta,0)+np.log1p(np.exp(-abs(eta))));weight=w[j]*np.exp(lp);norm+=weight;m1+=weight*z[j];m2+=weight*z[j]*z[j]
        norm=max(norm,1e-300);m=m1/norm;v=max(m2/norm-m*m,1e-10);pred[i]=pp;state[i]=m;terms[i]=np.log(norm)
    return pred,state,terms

def worker(j):
    path=OUT/'fits'/f"{j['id']}.json"
    if path.exists():return json.loads(path.read_text())
    started=time.time();data=data_with_age();d=slice_data(data,0,j['end']);p=d['y'].mean();init=[np.log(p/(1-p)),j['mu0'],np.log(j['sigma0']),np.log(.6)]
    opt=minimize(lambda t:-marginal(t,d),init,method='L-BFGS-B',bounds=[(-8,8),(-5,5),(np.log(.01),np.log(10)),(np.log(.01),np.log(5))],options={'maxiter':200,'ftol':1e-10,'eps':1e-5})
    nodes,w=hermgauss(48);w/=np.sqrt(np.pi);pred,state,terms=filter_kernel(opt.x,data['dt'],data['reset'],data['age'],data['y'],data['n'],nodes,w)
    r=dict(status='COMPLETE',model='brownian_drift',theta=opt.x,loglik=-opt.fun,adf_loglik=terms[:j['end']].sum(),success=opt.success,message=opt.message,job=j,elapsed=time.time()-started)
    if j['scope']!='full':
        lo,hi=j['end'],j['test_end'];r['forward']=metrics(pred[lo:hi],data['y'][lo:hi],data['n'][lo:hi])
    path.parent.mkdir(exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),predict_tb=pred,state=state);write_json(path,r);return r

def main():
    data=data_with_age();folds=json.loads((RUN/'splits.json').read_text());scopes=[dict(scope='full',end=len(data['y']))]+[dict(scope=f"fold{f['fold']}",end=f['train_end'],test_end=f['test_end']) for f in folds]
    jobs=[dict(id=f"{s['scope']}_{i}",mu0=mu,sigma0=sigma,**s) for s in scopes for i,(mu,sigma) in enumerate([(0.,.5),(0.,2.),(.2,1.),(-.2,1.)])];write_json(OUT/'queue.json',jobs)
    with ProcessPoolExecutor(max_workers=16) as ex:
        for f in as_completed([ex.submit(worker,j) for j in jobs]):
            r=f.result();print(json.dumps(dict(id=r['job']['id'],ll=r['loglik'],mu=float(r['theta'][1]),sigma=float(np.exp(r['theta'][2])),success=bool(r['success']))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs),finished_unix=time.time(),scope='conditional Brownian drift alternative; not a fitted DDM threshold or seizure hazard'))

if __name__=='__main__':main()
