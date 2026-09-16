"""Bordered eigenpair Newton for a root of the exact map characteristic matrix."""
import numpy as np
from scipy.linalg import eig,solve


def root(c,guess,vector=None,maxiter=12):
    lam=complex(guess);A=c.matrix(lam)
    if vector is None:
        vals,vecs=eig(A,check_finite=False);v=vecs[:,np.argmin(abs(vals))]
    else:v=np.array(vector,complex,copy=True)
    v/=np.linalg.norm(v);reference=v.copy();history=[]
    for k in range(maxiter):
        A=c.matrix(lam);f=A@v;phase=np.vdot(reference,v)-1;err=float(max(np.linalg.norm(f),abs(phase)));history.append(err)
        if err<1e-8:return lam,v,err,history
        eps=1e-3;derivative=(c.matrix(lam+eps)-c.matrix(lam-eps))@v/(2*eps)
        B=np.empty((len(v)+1,len(v)+1),complex);B[:-1,:-1]=A;B[:-1,-1]=derivative;B[-1,:-1]=reference.conj();B[-1,-1]=0
        delta=solve(B,-np.r_[f,phase],check_finite=False)
        for back in range(8):
            alpha=2.**(-back);ll=lam+alpha*delta[-1];vv=v+alpha*delta[:-1]
            if abs(ll-lam)>100:continue
            newerr=float(max(np.linalg.norm(c.matrix(ll)@vv),abs(np.vdot(reference,vv)-1)))
            if np.isfinite(newerr) and newerr<err:lam,v=ll,vv;break
        else:break
    return lam,v,float(np.linalg.norm(c.matrix(lam)@v)/np.linalg.norm(v)),history
