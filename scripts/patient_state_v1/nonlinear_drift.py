"""A bounded nonlinear-force diagnostic, with matched linear-OU comparator.

ds=-(k*(s-mu)+(s-mu)^3/A^2)/tau dt + sqrt(2)*A/sqrt(tau)dW.
Negative k permits two potential wells; it is not imposed. Marks are binned
and event times conditioned on, so this does not identify seizure boundaries.
"""
import sys,json,time,traceback,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
from numba import njit
from concurrent.futures import ProcessPoolExecutor,as_completed
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.joint_counts import prepare
OUT=RUN/'nonlinear_drift_v1_21'

def generator(t,kind,points=192):
    grid=np.linspace(-8.,6.,points);h=grid[1]-grid[0];mu=t[0];tau,A=np.exp(t[1:3]);u=(grid-mu)/A
    potential=.5*u*u if kind=='ou' else .5*t[3]*u*u+.25*u**4
    logpi=-potential;pi=np.exp(logpi-logsumexp(logpi));D=A*A/tau;delta=np.diff(potential)
    def bernoulli(v):
        out=np.empty_like(v);small=abs(v)<1e-5;pos=v>50;neg=v< -50;mid=~(small|pos|neg);out[small]=1-v[small]/2+v[small]**2/12;out[pos]=v[pos]*np.exp(-v[pos]);out[neg]=-v[neg];out[mid]=v[mid]/np.expm1(v[mid]);return out
    up=np.r_[D/h**2*bernoulli(delta),0.];down=np.r_[0.,D/h**2*bernoulli(-delta)];return grid,pi,up,down

@njit(cache=True)
def forward(dt,reset,n,y,grid,pi,up,down,max_step):
    N=len(grid);p=pi.copy();diag=np.empty(N);rhs=np.empty(N);upper=np.empty(N);terms=np.empty(len(n));means=np.empty(len(n));edge=np.empty(len(n));pred=np.empty(len(n))
    for i in range(len(n)):
        if reset[i]:p=pi.copy()
        elif dt[i]>0:
            steps=max(1,int(np.ceil(dt[i]/max_step)));step=dt[i]/steps
            for _ in range(steps):
                for j in range(N):diag[j]=1+step*(up[j]+down[j]);rhs[j]=p[j];upper[j]=-step*down[j+1] if j<N-1 else 0.
                for j in range(1,N):
                    fac=-step*up[j-1]/diag[j-1];diag[j]-=fac*upper[j-1];rhs[j]-=fac*rhs[j-1]
                p[N-1]=rhs[N-1]/diag[N-1]
                for j in range(N-2,-1,-1):p[j]=(rhs[j]-upper[j]*p[j+1])/diag[j]
                p/=p.sum()
        eta_ll=y[i]*grid-n[i]*np.logaddexp(0.,grid);peak=np.max(eta_ll);w=p*np.exp(eta_ll-peak);z=w.sum();terms[i]=np.log(max(z,1e-300))+peak;pred[i]=np.sum(p/(1+np.exp(-grid)));p=w/max(z,1e-300);means[i]=np.sum(p*grid);edge[i]=p[0]+p[-1]
    return terms,means,edge,pred

def evaluate(t,d,kind,points=192,max_step_seconds=15.):
    grid,pi,up,down=generator(t,kind,points);ll,m,e,p=forward(d['dt'],d['reset'],d['n'],d['y'],grid,pi,up,down,max_step_seconds/3600);return dict(loglik=float(ll.sum()),loglik_terms=ll,mean=m,edge_mass=e,predict_tb=p,stationary_edge_mass=float(pi[0]+pi[-1]))

def fit_model(d,kind,initial,points=192,max_step_seconds=15.):
    bounds=[(-3.,1.),(np.log(1/3600),np.log(48)),(np.log(.05),np.log(5))]+([(-4.,20.)] if kind=='quartic' else [])
    def fun(t):return -evaluate(t,d,kind,points,max_step_seconds)['loglik']
    opt=minimize(fun,initial,method='L-BFGS-B',bounds=bounds,options={'maxiter':150,'ftol':1e-10,'eps':2e-5,'maxls':25});return dict(theta=opt.x,loglik=-opt.fun,success=bool(opt.success),message=str(opt.message),nfev=opt.nfev)

def canary():
    from scipy.linalg import expm
    theta=np.array([-.8,np.log(.3),np.log(.6)]);grid,pi,up,down=generator(theta,'ou',32);Q=np.diag(-up-down)+np.diag(up[:-1],-1)+np.diag(down[1:],1);assert np.max(abs(Q@pi))<1e-9 and np.max(abs(Q.sum(axis=0)))<1e-9
    p0=np.zeros(32);p0[15]=1.;dt=np.array([.001]);d=dict(dt=dt,reset=np.array([False]),n=np.array([0.]),y=np.array([0.]));exact=expm(Q*.001)@p0;approx=[]
    for step in [1.,.25,.0625]:
        _,m,e,_=forward(dt,d['reset'],d['n'],d['y'],grid,p0,up,down,step/3600);approx.append(float(abs(m[0]-exact@grid)))
    assert approx[-1]<approx[0] and approx[-1]<1e-4
    write_json(OUT/'numerical_canary.json',dict(status='PASS',stationary_balance_error=float(np.max(abs(Q@pi))),mass_generator_error=float(np.max(abs(Q.sum(axis=0)))),implicit_refinement_mean_errors=approx))

def worker(j):
    p=OUT/'fits'/f"{j['id']}.json"
    if p.exists():return json.loads(p.read_text())
    start=time.time()
    try:
        full=prepare(j['seconds']);d={k:v[:j['end']].copy() for k,v in full.items() if np.ndim(v)>0};q=d['y'].sum()/d['n'].sum();b=np.log(q/(1-q));initial=np.r_[b,np.log(j['tau0']),np.log(j['A0']),([j['k0']] if j['kind']=='quartic' else [])];r=fit_model(d,j['kind'],initial);f=evaluate(r['theta'],full,j['kind']);r.update(status='COMPLETE',job=j,elapsed=time.time()-start,max_filtered_edge_mass=float(f['edge_mass'].max()),stationary_edge_mass=f['stationary_edge_mass']);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p.with_suffix('.npz'),**f)
    except Exception:r=dict(status='FAILED',job=j,elapsed=time.time()-start,traceback=traceback.format_exc())
    write_json(p,r);return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=int,nargs='+',default=[15,60]);ap.add_argument('--workers',type=int,default=20);args=ap.parse_args();canary();folds=json.loads((RUN/'splits.json').read_text());inv=json.loads((RUN/'seizures.json').read_text());ev=np.load(RUN/'observations.npz');jobs=[]
    for sec in args.seconds:
        d=prepare(sec);scopes=[dict(scope='full',end=len(d['n']))]
        for f in folds:
            cut=inv[int(ev['epoch'][f['test_start']])-1]['offset'];scopes.append(dict(scope=f"fold{f['fold']}",end=int(np.searchsorted(d['hi'],cut,side='right')),cutoff_epoch=cut))
        for scope in scopes:
            for kind in ['ou','quartic']:
                initials=[(.3,.6,0.),(2.,.6,0.)] if kind=='ou' else [(.3,.7,-1.),(.3,.7,0.),(.3,1.,2.),(2.,1.5,8.)]
                for k,(tau,A,shape) in enumerate(initials):jobs.append(dict(id=f"b{sec}_{scope['scope']}_{kind}_{k}",seconds=sec,kind=kind,tau0=tau,A0=A,k0=shape,**scope))
    write_json(OUT/'contract.json',dict(question='Does allowing a cubic restoring force improve causal label-sequence prediction over a matched linear OU?',state='Absolute log-odds s; logistic label observation, no separate state per mode',equation='ds=-(k*(s-mu)+(s-mu)^3/A^2)/tau dt+sqrt(2)*A/sqrt(tau)dW',shape='k may be negative (two potential wells), zero, or positive (one well); no absorbing boundaries or event resets',comparator='Three-parameter OU using identical observation bins, state grid and time integration',regularization='Four shared quartic parameters; bounded drift shape, state scale and time; no learned per-event initial vectors',numerics='192-point reflected grid [-8,6] log-odds, detailed-balance birth-death approximation, implicit 15-second maximum propagation step; mandatory resolution and boundary-mass audit',scope='15/60-second binned mark likelihood conditions on event occurrence; no count likelihood, no seizure-type fitting',n_jobs=len(jobs)))
    write_json(OUT/'queue.json',jobs)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i,f in enumerate(as_completed([ex.submit(worker,j) for j in jobs])):
            r=f.result();print(json.dumps(dict(done=i+1,total=len(jobs),id=r['job']['id'],status=r['status'],elapsed=r['elapsed'],success=r.get('success'))),flush=True)
    write_json(OUT/'status.json',dict(status='COMPLETE',n_jobs=len(jobs)))
if __name__=='__main__':main()
