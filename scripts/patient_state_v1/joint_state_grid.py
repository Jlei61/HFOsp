"""Deterministic two-dimensional OU filtering with joint rate/mark observations.

Exact Gaussian transition mass is integrated into destination cells. Independent
OU transitions factor into two sparse matrix products; the posterior is fully
joint. Boundary mass is not renormalized away before the likelihood update.
"""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import cupy as cp
import scipy.sparse as sp
import cupyx.scipy.sparse as csp
from scipy.special import ndtr
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.grid_renewal import transition
from scripts.patient_state_v1.renewal import prepare
from scripts.patient_state_v1.joint_two_state import unpack
OUT=RUN/'joint_state_grid_v1_24'
WEIGHT=cp.ElementwiseKernel('float64 mass,float64 s,float64 r,float64 n,float64 y,float64 risk,float64 shift,float64 b,float64 a,float64 c','float64 out','double eta=a+c*s+r; double q=b+s; double soft=fmax(q,0.)+log1p(exp(-fabs(q))); out=mass*exp(n*eta-risk*exp(fmin(eta,50.))+y*q-n*soft-shift);','joint_observation_weight')

def grid(sd,points,width=8.):
    dx=2*width*sd/points;x=(np.arange(points)+.5)*dx-width*sd;edges=np.r_[x-dx/2,x[-1]+dx/2];return x,np.diff(ndtr(edges/sd)),dx

def matrix(x,dx,tau,sd,dt):
    rho=np.exp(-dt/tau);innovation=sd*np.sqrt(-np.expm1(-2*dt/tau));ptr,dest,val=transition(x,dx,rho,innovation);origin=np.repeat(np.arange(len(x)),np.diff(ptr));return sp.csr_matrix((val,(dest,origin)),shape=(len(x),len(x)))

def evaluate(d,theta,coupled,ns=256,nr=128,progress=True,return_terms=False):
    b,a,c,ts,ss,tr,sr=unpack(theta,coupled);xs,ps,dxs=grid(ss,ns);xr,pr,dxr=grid(sr,nr);dt=np.round(d['dt'],10);unique,indices=np.unique(dt,return_inverse=True);operators={};start=time.time()
    for k,t in enumerate(unique):
        if t>0:operators[k]=(csp.csr_matrix(matrix(xs,dxs,ts,ss,t)),csp.csr_matrix(matrix(xr,dxr,tr,sr,t)))
    initial=cp.asarray(np.outer(ps,pr));s=cp.asarray(xs[:,None]);r=cp.asarray(xr[None,:]);p=initial.copy();ll=cp.asarray(0.);maxlost=cp.asarray(0.);sumlost=cp.asarray(0.);maxedge=0.;sampled=0;terms=cp.empty(len(dt)) if return_terms else None
    for i in range(len(dt)):
        if d['reset'][i]:p=initial.copy()
        else:
            T,U=operators[indices[i]];p=(U@(T@p).T.copy()).T.copy();loss=cp.maximum(0.,1-p.sum());maxlost=cp.maximum(maxlost,loss);sumlost+=loss
        n=float(d['n'][i]);y=float(d['y'][i]);risk=float(d['exposure'][i]);shift=n*np.log(n/risk)-n if n>0 and risk>0 else (n*(a+abs(c)*max(abs(xs))+max(abs(xr))) if n>0 else 0.)
        WEIGHT(p,s,r,n,y,risk,shift,b,a,c,p);norm=p.sum();increment=cp.log(norm)+shift;ll+=increment;p/=norm
        if return_terms:terms[i]=increment
        if i%5000==0 or i==len(dt)-1:
            value=float(ll.get());assert np.isfinite(value);edge=float((p[0,:].sum()+p[-1,:].sum()+p[:,0].sum()+p[:,-1].sum()).get());maxedge=max(maxedge,edge);sampled+=1
            if progress:print(json.dumps(dict(bin=i+1,total=len(dt),loglik=value,elapsed=time.time()-start,ns=ns,nr=nr)),flush=True)
    result=dict(loglik=float(ll.get()),elapsed=time.time()-start,ns=ns,nr=nr,spacing_s=dxs,spacing_r=dxr,max_transition_mass_loss=float(maxlost.get()),sum_transition_mass_loss=float(sumlost.get()),max_sampled_posterior_edge_mass=maxedge,n_edge_checks=sampled)
    if return_terms:result['loglik_terms']=terms.get()
    return result

def canary():
    n=np.array([0,1,2,0,1,3,0,0,1,2.]*2);y=np.floor(n*.6);dt=np.tile([0.,.001,.002,.001,.002,.001,.002,.001,.002,.001],2);reset=dt==0;d=dict(n=n,y=y,dt=dt,reset=reset,exposure=np.full(len(n),.001));rows=[]
    for c in [0.,.7]:
        t=np.array([-.8,5.,c,np.log(.3),np.log(.6),np.log(.1),np.log(.8)]);xs,ps,dxs=grid(.6,48);xr,pr,dxr=grid(.8,40);mat={u:(matrix(xs,dxs,.3,.6,u).toarray(),matrix(xr,dxr,.1,.8,u).toarray()) for u in [.001,.002]};p=np.outer(ps,pr);ll=0.;separate_s=ps.copy();separate_r=pr.copy();sep_ll=0.
        for i in range(len(n)):
            if reset[i]:p=np.outer(ps,pr);separate_s=ps.copy();separate_r=pr.copy()
            else:
                T,U=mat[dt[i]];p=T@p@U.T;separate_s=T@separate_s;separate_r=U@separate_r
            eta=5+c*xs[:,None]+xr[None,:];mark=y[i]*(-.8+xs)-n[i]*np.logaddexp(0,-.8+xs);obs=n[i]*eta-.001*np.exp(eta)+mark[:,None];p*=np.exp(obs);norm=p.sum();ll+=np.log(norm);p/=norm
            if c==0:
                separate_s*=np.exp(mark);separate_r*=np.exp(n[i]*(5+xr)-.001*np.exp(5+xr));zs=separate_s.sum();zr=separate_r.sum();sep_ll+=np.log(zs)+np.log(zr);separate_s/=zs;separate_r/=zr
        result=evaluate(d,t,True,48,40,False);assert abs(result['loglik']-ll)<1e-8
        if c==0:assert abs(ll-sep_ll)<1e-8
        rows.append(dict(c=c,cpu_joint_loglik=ll,gpu_joint_loglik=result['loglik'],cpu_gpu_error=abs(result['loglik']-ll),factorization_error=abs(ll-sep_ll) if c==0 else None))
    write_json(OUT/'canary.json',dict(status='PASS',checks=rows,scope='Same finite-volume model: CPU dense versus GPU sparse filtering, plus exact independent-component factorization; includes zero-event bins and interval resets'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,default=1);ap.add_argument('--canary-only',action='store_true');ap.add_argument('--ns',type=int,default=128);ap.add_argument('--nr',type=int,default=128);ap.add_argument('--models',nargs='+',default=['laplace_c0','laplace_c1','adf_c0','adf_c1']);args=ap.parse_args();cp.cuda.Device(args.gpu).use();canary()
    if args.canary_only:return
    d=prepare(5,.25);write_json(OUT/f'contract_s{args.ns}_r{args.nr}.json',dict(method='Full joint posterior over mode and activity states; separable exact-OU finite-volume transition, joint observation update',grid=[args.ns,args.nr],width_stationary_sd=8,gaussian_transition_cutoff_sd=8,observation_bin_seconds=5,no_boundary_mass_renormalization=True,required='Resolution and mass-loss audit; coarse grids are not automatically accepted'))
    for model in args.models:
        path=OUT/f'{model}_s{args.ns}_r{args.nr}.json'
        if path.exists():continue
        method,ctxt=model.split('_');coupled=ctxt=='c1';folder='joint_two_state_v1_20' if method=='laplace' else 'joint_adf_refit_v1_23';pattern=f'b5_full_{ctxt}_*.json' if method=='laplace' else f'full_{ctxt}_*.json';fits=[json.loads(p.read_text()) for p in (RUN/folder/'fits').glob(pattern)];best=max(fits,key=lambda r:r['loglik']);print(json.dumps(dict(model=model,phase='START',ns=args.ns,nr=args.nr)),flush=True);r=evaluate(d,best['theta'],coupled,args.ns,args.nr);r.update(status='COMPLETE',model=model,source=best);write_json(path,r)
    write_json(OUT/f'status_s{args.ns}_r{args.nr}.json',dict(status='COMPLETE',models=args.models))
if __name__=='__main__':main()
