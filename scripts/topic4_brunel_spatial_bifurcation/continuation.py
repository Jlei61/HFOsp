"""Pseudo-arclength continuation and augmented saddle-node equations."""
from common import *
from model import SpatialBrunel
from scipy.sparse.linalg import spsolve,eigs
import argparse

def tangent(s,r,J,previous=None):
    A=s.jacobian(r,J);b=s.parameter_derivative(r,J)
    if previous is None:
        dr=spsolve(A.tocsc(),-b);t=np.r_[dr*.02/.001,1.]
    else:
        mat=sparse.bmat([[.001*A,sparse.csr_matrix((.02*b)[:,None])],[sparse.csr_matrix(previous[None,:-1]),sparse.csr_matrix([[previous[-1]]])]],format='csc')
        t=spsolve(mat,np.r_[np.zeros(s.P),1.])
    t/=np.linalg.norm(t)
    if previous is not None and t@previous<0:t=-t
    return t

def main(args):
    s=SpatialBrunel(args.grid);dest=OUT/f'g{args.grid}'/'arclength_v2';dest.mkdir(parents=True,exist_ok=True)
    r,ok,tr=s.solve(args.start)
    assert ok;J=args.start;t=tangent(s,r,J);ds=.12;rows=[]
    for k in range(args.steps):
        row=dict(step=k,J_EE_core=J,rates_hz=s.regional_rates(r),residual=float(abs(s.residual(r,J)).max()),tangent_J=float(t[-1]))
        ev,vec=eigs(s.jacobian(r,J),k=3,sigma=0,tol=1e-10);row['static_eigenvalues']=ev
        np.savez_compressed(dest/f'step{k:04d}.npz',rates=r,J=J,tangent=t,eigenvalues=ev,eigenvectors=vec)
        rows.append(row);write(dest/'result.json',dict(status='RUNNING',rows=rows))
        print(row,flush=True)
        x=np.r_[r/.001,J/.02];pred=x+ds*t;trial=pred.copy();success=False
        for iteration in range(18):
            rr=trial[:-1]*.001;jj=trial[-1]*.02
            f=np.r_[s.residual(rr,jj),t@(trial-pred)]
            if abs(f).max()<1e-10:success=True;break
            A=s.jacobian(rr,jj);b=s.parameter_derivative(rr,jj)
            mat=sparse.bmat([[.001*A,sparse.csr_matrix((.02*b)[:,None])],[sparse.csr_matrix(t[None,:-1]),sparse.csr_matrix([[t[-1]]])]],format='csc')
            change=spsolve(mat,-f);a=1.
            while (trial[:-1]+a*change[:-1]).min()<0:a*=.5
            trial+=a*change
        if not success:raise RuntimeError((k,abs(f).max()))
        r=trial[:-1]*.001;J=trial[-1]*.02;t=tangent(s,r,J,t)
        if k>10 and J<args.start-.01:break
    write(dest/'result.json',dict(status='COMPLETE',rows=rows,meaning='Equilibrium branch with mathematical folds; temporal stability computed separately'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grid',type=int,default=20);p.add_argument('--start',type=float,default=1.);p.add_argument('--steps',type=int,default=130);main(p.parse_args())
