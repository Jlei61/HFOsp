"""Floquet growth without exponentially large monodromy products.

For v(t)=exp(alpha*t)q(t), a delayed perturbation gains exp(-alpha*d).
The shifted map has eigenvalues kappa=mu*exp(-alpha*T). No phase
projection is needed: the known neutral multiplier becomes exp(-alpha*T).
"""
from common import *
from periodic import Orbit
from scipy.signal import resample
from scipy.sparse.linalg import LinearOperator,eigs
from shifted_monodromy import shifted_monodromy4
import numpy as np,argparse

def compute(path,alpha_s,dtmax,nev=3):
    z=np.load(path);r=z['r'];T=float(z['T']);g=float(z['g']);N=len(r);s=System();o=Orbit(s,g,N)
    n=int(np.ceil(T/dtmax));dt=T/n;D=int(np.ceil(s.delay[-1]/dt))+3;alpha=alpha_s/1000
    H,_,_,_=o.kernels(T);mu=s.ext_mu+o.mean(r,H)
    ve=s.ext_var+(r[:,:3]@o.Q[:,:3].T)*s.tm;vi=(r[:,3:]@o.Q[:,3:].T)*s.tm
    vals=[resample(a,2*n,axis=0) for a in (mu,ve,vi)];gains=[]
    for j in range(3):
        step=1e-5*np.maximum(abs(vals[j]),1.);hi=vals.copy();lo=vals.copy();hi[j]=vals[j]+step;lo[j]=vals[j]-step
        gains.append((o.phi(*hi)-o.phi(*lo))/(2*step))
    u,v,h=gains;var=o.Q[None,:,:]*s.tm[None,:,None]*np.concatenate([np.repeat(v[:,:,None],3,axis=2),np.repeat(h[:,:,None],3,axis=2)],axis=2)
    u=np.r_[u,u[:1]];var=np.r_[var,var[:1]]
    dd,di,dj=np.nonzero(o.W);physical=s.delay[dd];delays=physical/dt
    weights=o.W[dd,di,dj]*s.tm[di]*s.area[dj]*s.sign[dj]*np.exp(-alpha*physical)
    dim=12+6*(D+1)
    def mv(x):return shifted_monodromy4(np.ascontiguousarray(x),dt,di,dj,delays,weights,u,var,s.tr,s.rise,s.decay,D,alpha)
    op=LinearOperator((dim,dim),matvec=mv,dtype=float)
    vals,vec=eigs(op,k=nev,which='LM',tol=1e-9,ncv=max(12,2*nev+5),maxiter=200,v0=np.random.default_rng(51).normal(size=dim))
    ix=np.argsort(abs(vals))[::-1];vals=vals[ix];vec=vec[:,ix]
    residual=[float(np.linalg.norm(mv(vec[:,j].real)+1j*mv(vec[:,j].imag)-vals[j]*vec[:,j])/max(abs(vals[j]),1e-300)) for j in range(len(vals))]
    row=dict(source=str(Path(path).resolve()),g=g,T_ms=T,N=N,dt_ms=dt,alpha_per_s=alpha_s,
        kappa=[[v.real,v.imag] for v in vals],growth_per_s=(alpha_s+np.log(abs(vals))/(T/1000)).tolist(),
        angle_rad=np.angle(vals).tolist(),relative_residual=residual,
        neutral_kappa=float(np.exp(-alpha*T)),method='shifted RK4, cubic delays, no phase projection')
    folder=OUT/'shifted_floquet'/Path(path).stem;folder.mkdir(parents=True,exist_ok=True)
    tag=f'alpha{alpha_s:g}_dt{dtmax:g}'
    write(folder/f'{tag}.json',row);np.savez_compressed(folder/f'{tag}.npz',kappa=vals,vectors=vec)
    print(json.dumps(row),flush=True);return row

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--alpha',type=float,required=True);p.add_argument('--dt',type=float,default=.025);p.add_argument('--nev',type=int,default=3);a=p.parse_args();compute(a.path,a.alpha,a.dt,a.nev)
