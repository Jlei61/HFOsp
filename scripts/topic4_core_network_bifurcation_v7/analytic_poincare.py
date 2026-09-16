"""Floquet spectrum on a section transverse to the known orbit tangent.

Uses analytic derivatives of the same frozen transfer quadrature.
The autonomous phase direction is removed geometrically, rather than selecting
an eigenvalue numerically closest to one in an ill-conditioned near-fold pair.
Step refinement and comparison with the unprojected spectrum are mandatory.
"""
from common import *
from periodic import Orbit
from floquet import monodromy
from scipy.signal import resample
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import LinearOperator,eigs
import numpy as np,argparse

def operator(path,dtmax,method='heun'):
    z=np.load(path);r=z['r'];T=float(z['T']);g=float(z['g']);N=len(r);s=System();o=Orbit(s,g,N)
    n=int(np.ceil(T/dtmax));dt=T/n;D=int(np.ceil(s.delay[-1]/dt))+(3 if method=='rk4' else 1)
    H,_,_,_=o.kernels(T);mu=s.ext_mu+o.mean(r,H);ve=s.ext_var+(r[:,:3]@o.Q[:,:3].T)*s.tm;vi=(r[:,3:]@o.Q[:,3:].T)*s.tm
    vals=[resample(a,n*(2 if method=='rk4' else 1),axis=0) for a in (mu,ve,vi)]
    from analytic_gains import gains
    u,v,h=gains(s,*vals)
    var=o.Q[None,:,:]*s.tm[None,:,None]*np.concatenate([np.repeat(v[:,:,None],3,axis=2),np.repeat(h[:,:,None],3,axis=2)],axis=2)
    u=np.r_[u,u[:1]];var=np.r_[var,var[:1]]
    dd,di,dj=np.nonzero(o.W);delays=s.delay[dd]/dt;lag=np.floor(delays).astype(np.int64);frac=delays-lag
    weights=o.W[dd,di,dj]*s.tm[di]*s.area[dj]*s.sign[dj];dim=12+6*(D+1)
    def mv(x):return monodromy(np.ascontiguousarray(x),dt,di,dj,lag,frac,weights,u,var,s.tr,s.rise,s.decay,D)
    if method=='rk4':
        from rk4_monodromy import monodromy4
        assert dt<=s.delay[0]/2+1e-12
        def mv(x):return monodromy4(np.ascontiguousarray(x),dt,di,dj,delays,weights,u,var,s.tr,s.rise,s.decay,D)
    freq=2j*np.pi*np.arange(N//2+1)/T;R=np.fft.rfft(r,axis=0)
    dr=np.fft.irfft(freq[:,None]*R,n=N,axis=0)
    dh=np.fft.irfft(freq[:,None]*R/(1+freq[:,None]*s.rise),n=N,axis=0)
    dc=np.fft.irfft(freq[:,None]*R/((1+freq[:,None]*s.rise)*(1+freq[:,None]*s.decay)),n=N,axis=0)
    C=freq[:,None]*R/((1+freq[:,None]*s.rise)*(1+freq[:,None]*s.decay));C[1:-1]*=2
    ch=(np.exp((-np.arange(D+1)*dt)[:,None]*freq[None,:])@C).real/N
    tangent=np.r_[dr[0],dh[0],ch.ravel()];tangent/=np.linalg.norm(tangent)
    return mv,tangent,dict(g=g,T_ms=T,N=N,dt_ms=dt,history_dimension=dim,method=method)

def compute(path,dtmax=.1,section='orthogonal',method='heun',nev=1):
    path=Path(path).resolve();mv,tan,row=operator(path,dtmax,method);dim=len(tan);normal=tan.copy()
    if section=='rate':
        normal[:]=0;i=np.argmax(abs(tan[:6]));normal[i]=1/tan[i]
    normal/=normal@tan
    def project(x):return x-tan*(normal@x)
    def pp(x):return project(mv(project(x)))
    mt=mv(tan);op=LinearOperator((dim,dim),matvec=pp,dtype=float)
    vals,vec=eigs(op,k=nev,which='LM',tol=1e-8,ncv=max(10,2*nev+4),maxiter=150,v0=np.random.default_rng(51).normal(size=dim))
    ix=np.argsort(abs(vals))[::-1];vals=vals[ix];vec=vec[:,ix]
    residual=[float(np.linalg.norm(pp(vec[:,j].real)+1j*pp(vec[:,j].imag)-vals[j]*vec[:,j])) for j in range(len(vals))]
    row.update(source=str(path),gain_derivative='analytic frozen transfer quadrature',section=section,requested_multipliers=nev,multipliers=[[v.real,v.imag] for v in vals],residuals=residual,
        orbit_tangent_defect=float(np.linalg.norm(mt-tan)),section_tangent_defect=float(np.linalg.norm(project(mt-tan))),
        max_transverse=float(max(abs(vals))))
    dest=OUT/'analytic_poincare'/path.parent.name/path.stem;dest.mkdir(parents=True,exist_ok=True)
    (dest/f'{method}_{section}_dt{dtmax:g}.json').write_text(json.dumps(row,indent=2)+'\n')
    np.savez_compressed(dest/f'{method}_{section}_dt{dtmax:g}.npz',multipliers=vals,vectors=vec,tangent=tan,normal=normal)
    print('POINCARE',json.dumps(row),flush=True);return row

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('paths',nargs='+');ap.add_argument('--dt',type=float,default=.1);ap.add_argument('--section',default='orthogonal',choices=['orthogonal','rate']);ap.add_argument('--method',default='heun',choices=['heun','rk4']);ap.add_argument('--nev',type=int,default=1);a=ap.parse_args()
    for path in a.paths:compute(path,a.dt,a.section,a.method,a.nev)
