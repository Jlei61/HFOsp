"""v1.9: one scalar mode preference with two regularized temporal components.

The observation is sigmoid(b + s_fast + s_background + optional fast mark
history). Background correlation time is fixed at 6 or 24 hours as a sensitivity,
not inferred as a circadian mechanism. No seizure type enters the model.
"""
import sys,json,time,traceback,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.special import expit
from scipy.linalg import solveh_banded,cholesky_banded
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss
from numba import njit
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.model import prior,qmul,slice_data
from scripts.patient_state_v1.advanced_controls import history_data
from scripts.patient_state_v1.analyze_first import metrics

OUT=RUN/'two_timescale_marks_v1_9'

def parts(theta,slow_tau,history):
    nc=2 if history else 1;tau,sd,slow_sd=np.exp(theta[nc:nc+3]);return np.asarray(theta[:nc]),tau,sd,slow_tau,slow_sd

def laplace2(theta,d,slow_tau,history=False,return_state=False):
    beta,tau,sd,ts,ss=parts(theta,slow_tau,history);nc=len(beta);eta0=d['x'][:,:nc]@beta;n,y=d['n'],d['y'];length=len(y)
    _,_,df,of,ldf=prior(d['dt'],d['reset'],tau,sd);_,_,ds,os,lds=prior(d.get('slow_dt',d['dt']),d.get('slow_reset',d['reset']),ts,ss)
    return laplace_blocks(eta0,n,y,df,of,ldf,ds,os,lds,return_state)

def laplace_blocks(eta0,n,y,df,of,ldf,ds,os,lds,return_state=False):
    length=len(y);s=np.zeros((length,2))
    def objective(s):
        eta=eta0+s.sum(axis=1);return np.sum(n*np.logaddexp(0,eta)-y*eta)+.5*(s[:,0]@qmul(s[:,0],df,of)+s[:,1]@qmul(s[:,1],ds,os))
    value=objective(s);converged=False;band=np.zeros((3,length*2));band[2,:-2]=np.column_stack([of,os]).ravel()
    for iteration in range(60):
        p=expit(eta0+s.sum(axis=1));w=n*p*(1-p);g=n*p-y;grad=np.column_stack([qmul(s[:,0],df,of)+g,qmul(s[:,1],ds,os)+g]).ravel()
        band[0]=np.column_stack([df+w,ds+w]).ravel();band[1,::2]=w;delta=solveh_banded(band,grad,lower=True,check_finite=False).reshape(length,2);step=1.
        for _ in range(25):
            candidate=s-step*delta;v=objective(candidate)
            if v<=value+1e-8:break
            step*=.5
        change=abs(value-v);s=candidate;value=v
        if np.max(abs(step*delta))<1e-6 or (change<1e-8 and np.max(abs(delta))<1e-3):converged=True;break
    p=expit(eta0+s.sum(axis=1));w=n*p*(1-p);band[0]=np.column_stack([df+w,ds+w]).ravel();band[1,::2]=w;chol=cholesky_banded(band,lower=True,check_finite=False);ll=-value+.5*(ldf+lds-2*np.log(chol[0]).sum())
    if return_state:return dict(loglik=ll,mode=s,converged=converged,iterations=iteration+1)
    return float(ll) if converged else float(ll)-1000

@njit(cache=True)
def filter_kernel(dt,reset,slow_dt,slow_reset,y,n,baseline,tau,sd,ts,ss,nodes,weights):
    length=len(y);means=np.empty((length,2));variances=np.empty((length,3));prediction=np.empty(length);terms=np.empty(length);m0=0.;m1=0.;p00=sd*sd;p11=ss*ss;p01=0.
    for i in range(length):
        a=np.exp(-dt[i]/tau);b=np.exp(-slow_dt[i]/ts)
        if reset[i]:m0=0.;p00=sd*sd
        else:m0*=a;p00=a*a*p00+sd*sd*(-np.expm1(-2*dt[i]/tau))
        if slow_reset[i]:m1=0.;p11=ss*ss
        else:m1*=b;p11=b*b*p11+ss*ss*(-np.expm1(-2*slow_dt[i]/ts))
        if reset[i] or slow_reset[i]:p01=0.
        else:p01*=a*b
        mean=m0+m1;var=max(p00+p11+2*p01,1e-14);norm=0.;first=0.;second=0.;pred=0.
        for j in range(len(nodes)):
            z=mean+np.sqrt(2*var)*nodes[j];eta=baseline[i]+z;p=1/(1+np.exp(-eta));pred+=weights[j]*p;lp=y[i]*eta-n[i]*(max(eta,0)+np.log1p(np.exp(-abs(eta))));w=weights[j]*np.exp(lp);norm+=w;first+=w*z;second+=w*z*z
        norm=max(norm,1e-300);post=first/norm;pv=max(second/norm-post*post,1e-12);k0=(p00+p01)/var;k1=(p11+p01)/var;m0+=k0*(post-mean);m1+=k1*(post-mean);p00+=k0*k0*(pv-var);p11+=k1*k1*(pv-var);p01+=k0*k1*(pv-var)
        means[i,0]=m0;means[i,1]=m1;variances[i,0]=p00;variances[i,1]=p11;variances[i,2]=p01;prediction[i]=pred;terms[i]=np.log(norm)
    return means,variances,prediction,terms

def filter2(theta,d,slow_tau,history=False):
    beta,tau,sd,ts,ss=parts(theta,slow_tau,history);nodes,w=hermgauss(40);m,v,p,ll=filter_kernel(d['dt'],d['reset'],d.get('slow_dt',d['dt']),d.get('slow_reset',d['reset']),d['y'],d['n'],d['x'][:,:len(beta)]@beta,tau,sd,ts,ss,nodes,w/np.sqrt(np.pi));return dict(mean=m,variance=v,predict_tb=p,loglik_terms=ll,loglik=ll.sum())

def fit2(d,slow_tau,history,initial,min_tau=1/60):
    nc=2 if history else 1;bounds=[(-8,8)]*nc+[(np.log(min_tau),np.log(slow_tau/4)),(np.log(.01),np.log(5)),(np.log(.001),np.log(3))]
    def objective(t):
        try:return -laplace2(t,d,slow_tau,history)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError):return 1e20
    opt=minimize(objective,initial,method='L-BFGS-B',bounds=bounds,options={'maxiter':160,'ftol':1e-10,'eps':2e-5,'maxls':20})
    return dict(theta=opt.x,loglik=-opt.fun,success=opt.success,message=opt.message,nfev=opt.nfev)

def worker(job):
    path=Path(job.get('output',OUT))/'fits'/f"{job['id']}.json"
    if path.exists():return json.loads(path.read_text())
    start=time.time()
    try:
        data=dict(np.load(RUN/'observations.npz'))
        if job['history']:data=history_data(data)
        if job.get('carry_slow',False):data['slow_dt']=np.r_[0,np.diff(data['t'])];data['slow_reset']=np.r_[True,np.zeros(len(data['t'])-1,bool)]
        train=slice_data(data,0,job['end']);p=train['y'].mean();nc=2 if job['history'] else 1;initial=np.r_[np.log(p/(1-p)),np.ones(nc-1)*.8,np.log(job['tau0']),np.log(.5),np.log(job['slow_sd0'])]
        r=fit2(train,job['slow_tau'],job['history'],initial,job.get('min_tau',1/60));f=filter2(r['theta'],data,job['slow_tau'],job['history']);r.update(status='COMPLETE',model='two_ou_history' if job['history'] else 'two_ou',job=job,elapsed=time.time()-start)
        if job['scope']!='full':
            lo,hi=job['test_start'],job['test_end'];r['forward']=metrics(f['predict_tb'][lo:hi],data['y'][lo:hi],data['n'][lo:hi])
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**f)
    except Exception:r=dict(status='FAILED',job=job,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(path,r);return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=24);ap.add_argument('--min-tau-seconds',type=float,default=60);ap.add_argument('--compare-carry',action='store_true');ap.add_argument('--output-name',default='two_timescale_marks_v1_9');args=ap.parse_args();out=RUN/args.output_name;d=np.load(RUN/'observations.npz');folds=json.loads((RUN/'splits.json').read_text());scopes=[dict(scope='full',end=len(d['y']))]+[dict(scope=f"fold{f['fold']}",end=f['train_end'],test_start=f['test_start'],test_end=f['test_end']) for f in folds];jobs=[]
    for scope in scopes:
        for slow_tau in (6.,24.):
            for history in (False,True):
                for carry in ([False,True] if args.compare_carry else [False]):
                    initial=((.001,.2),(.02,.6),(.2,1.)) if args.min_tau_seconds<60 else ((.1,.2),(.5,.6),(1.,1.))
                    for i,(tau,slow_sd) in enumerate(initial):jobs.append(dict(id=f"{scope['scope']}_slow{int(slow_tau)}_hist{int(history)}"+(f'_carry{int(carry)}' if args.compare_carry else '')+f'_{i}',slow_tau=slow_tau,history=history,tau0=tau,slow_sd0=slow_sd,carry_slow=carry,min_tau=args.min_tau_seconds/3600,output=str(out),**scope))
    write_json(out/'queue.json',jobs);write_json(out/'contract.json',dict(question='Does a slowly varying background absorb the prefix-dependent single-OU time constant?',state='One scalar mode preference is the sum of independent fast and background OU components',regularization='Background tau fixed at 6/24h sensitivity; fast tau <= one quarter of background tau; background SD may shrink to .001',min_fast_tau_seconds=args.min_tau_seconds,compare_background_carry_across_ictal=args.compare_carry,not_a_claim='Two temporal components are not two propagation modes, bistability, or measured E/I resources',limits='Initial state and observation-end conditions remain modeling assumptions; no seizure hazard is fitted'))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(worker,j) for j in jobs]):
            r=f.result();print(json.dumps(dict(id=r['job']['id'],status=r['status'],elapsed=r['elapsed'],loglik=r.get('loglik'),forward=r.get('forward'))),flush=True)
    write_json(out/'status.json',dict(status='COMPLETE',n_jobs=len(jobs),finished_unix=time.time()))

if __name__=='__main__':main()
