"""Floquet multipliers from the full variational delay system over one orbit.

History interpolation and Heun time integration are refined independently of
Fourier orbit collocation. The autonomous neutral multiplier is a built-in check.
"""
from model import System,OUT
from periodic import Orbit
import numpy as np,json,argparse
from scipy.signal import resample
from scipy.sparse.linalg import LinearOperator,eigs
from numba import njit

@njit(cache=True)
def monodromy(x,dt,di,dj,lag,frac,val,u,var,tr,rise,decay,D):
    hist=x[12:].reshape(D+1,6).copy();r=x[:6].copy();h=x[6:12].copy();head=0
    for k in range(len(u)-1):
        mu0=np.zeros(6);mu1=np.zeros(6)
        for e in range(len(val)):
            a=(head+lag[e])%(D+1);b=(a+1)%(D+1)
            ap=(a-1)%(D+1);bp=a
            mu0[di[e]]+=val[e]*((1-frac[e])*hist[a,dj[e]]+frac[e]*hist[b,dj[e]])
            mu1[di[e]]+=val[e]*((1-frac[e])*hist[ap,dj[e]]+frac[e]*hist[bp,dj[e]])
        dr0=(-r+var[k]@r+u[k]*mu0)/tr
        dh0=(r-h)/rise;dc0=(h-hist[head])/decay
        rp=r+dt*dr0;hp=h+dt*dh0;cp=hist[head]+dt*dc0
        dr1=(-rp+var[k+1]@rp+u[k+1]*mu1)/tr
        dh1=(rp-hp)/rise;dc1=(hp-cp)/decay
        r+=.5*dt*(dr0+dr1);h+=.5*dt*(dh0+dh1);cn=hist[head]+.5*dt*(dc0+dc1)
        head=(head-1)%(D+1);hist[head]=cn
    out=np.empty_like(x);out[:6]=r;out[6:12]=h
    for d in range(D+1):out[12+6*d:18+6*d]=hist[(head+d)%(D+1)]
    return out

def compute(path,dtmax=.1):
    z=np.load(path);r=z['r'];T=float(z['T']);g=float(z['g']);N=len(r);s=System();o=Orbit(s,g,N)
    n=int(np.ceil(T/dtmax));dt=T/n;D=int(np.ceil(s.delay[-1]/dt))+1
    H,_,_,_=o.kernels(T);mu=s.ext_mu+o.mean(r,H);ve=s.ext_var+(r[:,:3]@o.Q[:,:3].T)*s.tm;vi=(r[:,3:]@o.Q[:,3:].T)*s.tm
    # Resample the orbit moments before differentiating the nonlinear transfer.
    vals=[resample(a,n,axis=0) for a in [mu,ve,vi]];gains=[]
    for j in range(3):
        step=1e-5*np.maximum(abs(vals[j]),1.);hi=vals.copy();lo=vals.copy();hi[j]=vals[j]+step;lo[j]=vals[j]-step
        gains.append((o.phi(*hi)-o.phi(*lo))/(2*step))
    u,v,h=gains;var=o.Q[None,:,:]*s.tm[None,:,None]*np.concatenate([np.repeat(v[:,:,None],3,axis=2),np.repeat(h[:,:,None],3,axis=2)],axis=2)
    u=np.r_[u,u[:1]];var=np.r_[var,var[:1]]
    dd,di,dj=np.nonzero(o.W);delays=s.delay[dd]/dt;lag=np.floor(delays).astype(np.int64);frac=delays-lag
    weights=o.W[dd,di,dj]*s.tm[di]*s.area[dj]*s.sign[dj]
    dim=12+6*(D+1)
    def mv(x):return monodromy(np.ascontiguousarray(x),dt,di,dj,lag,frac,weights,u,var,s.tr,s.rise,s.decay,D)
    op=LinearOperator((dim,dim),matvec=mv,dtype=float)
    vals,vec=eigs(op,k=5,which='LM',tol=1e-8,ncv=14,maxiter=150,v0=np.random.default_rng(41).normal(size=dim))
    order=np.argsort(abs(vals))[::-1];vals=vals[order];vec=vec[:,order]
    residual=[float(np.linalg.norm(mv(vec[:,j].real)+1j*mv(vec[:,j].imag)-vals[j]*vec[:,j])/np.linalg.norm(vec[:,j])) for j in range(len(vals))]
    row=dict(g=g,T_ms=T,N=N,dt_ms=dt,history_dimension=dim,multipliers=[[v.real,v.imag] for v in vals],residuals=residual)
    print('FLOQUET',json.dumps(row),flush=True)
    dest=OUT/'floquet';dest.mkdir(exist_ok=True);(dest/f'g{g:.8f}_dt{dtmax:g}.json').write_text(json.dumps(row,indent=2)+'\n')
    np.savez_compressed(dest/f'g{g:.8f}_dt{dtmax:g}.npz',vectors=vec,multipliers=vals)
    return row

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--dt',type=float,default=.1);a=p.parse_args();compute(a.path,a.dt)
