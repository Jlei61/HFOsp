"""v1.12: Brownian slow background plus finite-time fast mode fluctuations.

No event-triggered reset, absorbing threshold or seizure hazard. Compare carrying
the Brownian background through ictal exclusions with independent interval priors.
Initial background SD is fixed: one continuous record cannot identify its law.
"""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss
from numba import njit
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import prior,slice_data
from scripts.patient_state_v1.two_timescale import laplace_blocks
from scripts.patient_state_v1.advanced_controls import history_data
from scripts.patient_state_v1.brownian import data_with_age
from scripts.patient_state_v1.analyze_first import metrics

OUT=RUN/'brownian_fast_marks_v1_12'

def dataset(carry):
    original=data_with_age();d=history_data(original)
    if carry:d['age']=d['t'].copy();d['slow_dt']=np.r_[0,np.diff(d['t'])];d['slow_reset']=np.r_[True,np.zeros(len(d['t'])-1,bool)]
    else:d['slow_dt']=d['dt'].copy();d['slow_reset']=d['reset'].copy()
    return d

def unpack(theta,history,drift):
    nc=2 if history else 1;beta=np.asarray(theta[:nc]);mu=theta[nc] if drift else 0.;tau,sdf,sigma=np.exp(theta[nc+int(drift):]);return beta,mu,tau,sdf,sigma

def brownian_prior(d,sigma,sd0):
    a=np.ones(len(d['y']));a[d['slow_reset']]=0;q=sigma*sigma*d['slow_dt'];q[d['slow_reset']]=sd0*sd0+sigma*sigma*d['age'][d['slow_reset']];diag=1/q;diag[:-1]+=a[1:]**2/q[1:];off=-a[1:]/q[1:];return diag,off,-np.log(q).sum()

def likelihood(theta,d,history,drift,sd0):
    beta,mu,tau,sdf,sigma=unpack(theta,history,drift);_,_,df,of,ldf=prior(d['dt'],d['reset'],tau,sdf);ds,os,lds=brownian_prior(d,sigma,sd0);eta=d['x'][:,:len(beta)]@beta+mu*d['age'];return laplace_blocks(eta,d['n'],d['y'],df,of,ldf,ds,os,lds)

@njit(cache=True)
def filtering(dt,reset,sdt,sreset,age,y,baseline,tau,sdf,sigma,sd0,nodes,w):
    n=len(y);m0=0.;m1=0.;p00=sdf*sdf;p11=sd0*sd0;p01=0.;pred=np.empty(n);means=np.empty((n,2));variance=np.empty((n,3));terms=np.empty(n)
    for i in range(n):
        a=np.exp(-dt[i]/tau)
        if reset[i]:m0=0.;p00=sdf*sdf
        else:m0*=a;p00=a*a*p00+sdf*sdf*(-np.expm1(-2*dt[i]/tau))
        if sreset[i]:m1=0.;p11=sd0*sd0+sigma*sigma*age[i]
        else:p11+=sigma*sigma*sdt[i]
        if reset[i] or sreset[i]:p01=0.
        else:p01*=a
        mean=m0+m1;v=max(p00+p11+2*p01,1e-14);norm=0.;first=0.;second=0.;pp=0.
        for j in range(len(nodes)):
            z=mean+np.sqrt(2*v)*nodes[j];eta=baseline[i]+z;p=1/(1+np.exp(-eta));pp+=w[j]*p;lp=y[i]*eta-(max(eta,0)+np.log1p(np.exp(-abs(eta))));weight=w[j]*np.exp(lp);norm+=weight;first+=weight*z;second+=weight*z*z
        norm=max(norm,1e-300);post=first/norm;pv=max(second/norm-post*post,1e-12);k0=(p00+p01)/v;k1=(p11+p01)/v;m0+=k0*(post-mean);m1+=k1*(post-mean);p00+=k0*k0*(pv-v);p11+=k1*k1*(pv-v);p01+=k0*k1*(pv-v);pred[i]=pp;means[i,0]=m0;means[i,1]=m1;variance[i,0]=p00;variance[i,1]=p11;variance[i,2]=p01;terms[i]=np.log(norm)
    return pred,means,variance,terms

def filter_model(theta,d,history,drift,sd0):
    beta,mu,tau,sdf,sigma=unpack(theta,history,drift);nodes,w=hermgauss(48);p,m,v,ll=filtering(d['dt'],d['reset'],d['slow_dt'],d['slow_reset'],d['age'],d['y'],d['x'][:,:len(beta)]@beta+mu*d['age'],tau,sdf,sigma,sd0,nodes,w/np.sqrt(np.pi));return dict(predict_tb=p,mean=m,variance=v,loglik_terms=ll,loglik=ll.sum())

def worker(job):
    path=OUT/'fits'/f"{job['id']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        data=dataset(job['carry']);d=slice_data(data,0,job['end']);p=d['y'].mean();nc=2 if job['history'] else 1;init=np.r_[np.log(p/(1-p)),np.ones(nc-1)*.8,([0.] if job['drift'] else []),np.log(job['tau0']),np.log(.4),np.log(.3)];bounds=[(-8,8)]*nc+([(-1,1)] if job['drift'] else [])+[(np.log(1/3600),np.log(3)),(np.log(.01),np.log(3)),(np.log(.001),np.log(3))]
        opt=minimize(lambda t:-likelihood(t,d,job['history'],job['drift'],job['sd0']),init,method='L-BFGS-B',bounds=bounds,options={'maxiter':180,'ftol':1e-10,'eps':1e-5,'maxls':25});f=filter_model(opt.x,data,job['history'],job['drift'],job['sd0']);r=dict(status='COMPLETE',job=job,theta=opt.x,loglik=-opt.fun,success=opt.success,message=opt.message,nfev=opt.nfev,elapsed=time.time()-start)
        if job['scope']!='full':
            lo,hi=job['end'],job['test_end'];r['forward']=metrics(f['predict_tb'][lo:hi],data['y'][lo:hi],data['n'][lo:hi])
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**f)
    except Exception:r=dict(status='FAILED',job=job,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,r);return r

def main():
    d=np.load(RUN/'observations.npz');folds=json.loads((RUN/'splits.json').read_text());scopes=[dict(scope='full',end=len(d['y']))]+[dict(scope=f"fold{f['fold']}",end=f['train_end'],test_end=f['test_end']) for f in folds];jobs=[]
    for scope in scopes:
        for carry in [False,True]:
            for history in [False,True]:
                for drift in [False,True]:
                    for i,tau in enumerate([.001,.03]):jobs.append(dict(id=f"{scope['scope']}_carry{int(carry)}_hist{int(history)}_drift{int(drift)}_{i}",carry=carry,history=history,drift=drift,sd0=.6,tau0=tau,**scope))
    write_json(OUT/'queue.json',jobs);write_json(OUT/'contract.json',dict(question='Does slow mean reversion matter once fast serial variability is separately represented?',slow_state='Brownian background, with or without common constant drift',units='time in hours, state in log-odds, diffusion in log-odds/sqrt(hour)',initial_sd=.6,regularization='Initial distribution width fixed; no mode-specific initial vectors; one common drift across all intervals',not_fitted='No seizure absorption boundary, seizure hazard, event-triggered reset or clinical type input',scope='Developed after two-OU sensitivity; compare chronological prediction rather than pre-seizure p values'))
    with ProcessPoolExecutor(max_workers=24) as ex:
        for f in as_completed([ex.submit(worker,j) for j in jobs]):
            r=f.result();print(json.dumps(dict(id=r['job']['id'],status=r['status'],loglik=r.get('loglik'),elapsed=r['elapsed'],forward=r.get('forward'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs),finished_unix=time.time()))

if __name__=='__main__':main()
