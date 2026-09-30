"""Adaptive pseudo-arclength continuation of the unchanged spatial equilibria."""
from common import *
from model import SpatialBrunel
from scipy.sparse.linalg import spsolve,eigs
import argparse

RS=.01
JS=.05

def tangent(s,r,J,previous=None,direction=1):
    A=s.jacobian(r,J);b=s.parameter_derivative(r,J)*JS/RS
    if previous is None:t=np.r_[-spsolve(A.tocsc(),b),1.]*direction
    else:
        mat=sparse.bmat([[A,sparse.csr_matrix(b[:,None])],[sparse.csr_matrix(previous[None,:-1]),sparse.csr_matrix([[previous[-1]]])]],format='csc')
        t=spsolve(mat,np.r_[np.zeros(s.P),1.])
    t/=np.linalg.norm(t)
    if previous is not None and t@previous<0:t=-t
    return t

def correct(s,pred,t):
    trial=pred.copy()
    if trial[:-1].min()*RS < -1e-10:return None,0
    trial[:-1]=np.maximum(trial[:-1],0.)
    for iteration in range(18):
        rr=trial[:-1]*RS;jj=trial[-1]*JS
        f=np.r_[s.residual(rr,jj)/RS,t@(trial-pred)]
        if abs(f).max()<1e-8:return trial,iteration
        A=s.jacobian(rr,jj);b=s.parameter_derivative(rr,jj)*JS/RS
        mat=sparse.bmat([[A,sparse.csr_matrix(b[:,None])],[sparse.csr_matrix(t[None,:-1]),sparse.csr_matrix([[t[-1]]])]],format='csc')
        change=spsolve(mat,-f);a=1.
        for back in range(16):
            new=trial+a*change;rnew=new[:-1]*RS;jnew=new[-1]*JS
            if rnew.min()>=-1e-10 and np.all(rnew<1/s.ref) and jnew>0:
                # Sub-tolerance negatives in effectively silent groups are
                # projected to zero, then the original residual is rechecked.
                # They must not force vanishing arc steps on high-rate branches.
                new[:-1]=np.maximum(new[:-1],0.);rnew=new[:-1]*RS
                ff=np.r_[s.residual(rnew,jnew)/RS,t@(new-pred)]
                if np.linalg.norm(ff)<np.linalg.norm(f):break
            a*=.5
        else:return None,iteration
        trial=new
    return None,iteration

def main(args):
    s=SpatialBrunel(response='calibrated_full');dest=OUT/'expanded'/args.label;dest.mkdir(parents=True,exist_ok=True)
    initial=np.load(args.seed);r=initial['rates'];J=float(initial['J']);r,ok,_=s.solve(J,r);assert ok
    t=tangent(s,r,J,direction=args.direction);ds=.2;rows=[];folds=[];stop='step_limit'
    for k in range(args.steps):
        ev,vec=eigs(s.jacobian(r,J),k=3,sigma=0,tol=1e-9)
        row=dict(step=k,J_EE_core=J,rates_hz=s.regional_rates(r),residual=float(abs(s.residual(r,J)).max()),
            tangent_J=float(t[-1]),static_eigenvalues=ev,ds=ds)
        np.savez_compressed(dest/f'step{k:04d}.npz',rates=r,J=J,tangent=t,eigenvalues=ev,eigenvectors=vec)
        if rows and rows[-1]['tangent_J']*t[-1]<0:
            folds.append(dict(before=k-1,after=k,J_EE_core=J,rates_hz=row['rates_hz']))
            print('FOLD BRACKET',folds[-1],flush=True)
        rows.append(row)
        if k%10==0:
            print(args.label,k,J,row['rates_hz'],ds,flush=True)
            write(dest/'result.json',dict(status='RUNNING',rows=rows,fold_brackets=folds))
        if k>5 and (J<args.minimum or J>args.maximum):stop='parameter_bound';break
        x=np.r_[r/RS,J/JS]
        for attempt in range(14):
            trial,it=correct(s,x+ds*t,t)
            if trial is not None:break
            ds*=.5
        else:stop='corrector_failure';break
        if ds<1e-5:stop='step_size_underflow';break
        rnew=trial[:-1]*RS;Jnew=trial[-1]*JS;tnew=tangent(s,rnew,Jnew,t)
        angle=float(tnew@t)
        r,J,t=rnew,Jnew,tnew
        if it<=4 and angle>.99:ds=min(ds*1.2,.8)
        elif it>7 or angle<.95:ds=max(ds*.65,.005)
    write(dest/'result.json',dict(status='COMPLETE' if stop=='parameter_bound' else 'BOUNDED_STOP',rows=rows,
        fold_brackets=folds,stop_reason=stop,meaning='Connected stationary branch, not a time trajectory; temporal stability is assessed separately.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',required=True);p.add_argument('--label',required=True)
    p.add_argument('--direction',type=int,default=-1);p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--minimum',type=float,default=.39);p.add_argument('--maximum',type=float,default=2.02);main(p.parse_args())
