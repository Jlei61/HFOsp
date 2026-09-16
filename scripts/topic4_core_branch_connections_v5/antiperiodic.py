"""Anti-periodic variational collocation: a zero mode implies Floquet -1.

Half-integer Fourier frequencies impose v(t+T)=-v(t) exactly, avoiding long
products of strongly expanding and contracting monodromy factors.
"""
from common import *
from periodic import Orbit
from scipy.sparse.linalg import LinearOperator,eigs
import numpy as np,argparse

def build(s,r,T,g):
    N=len(r);o=Orbit(s,g,N);y=np.r_[(r/.01).ravel(),np.log(T)]
    _,_,m=o.evaluate(y,r,np.zeros_like(r),True);u,v,h=m['gains']
    phase=np.exp(1j*np.pi*np.arange(N)/N)[:,None]
    freq=2j*np.pi*(np.fft.fftfreq(N)*N+.5)/T
    W=np.einsum('kd,dij->kij',np.exp(-freq[:,None]*s.delay),o.W)
    H=s.tm[None,:,None]*W*(s.area*s.sign)[None,None,:]/((1+freq[:,None,None]*s.rise[None,None,:])*(1+freq[:,None,None]*s.decay[None,None,:]))
    L=1/(1+freq[:,None]*s.tr)
    def filt(a,kernel):return (np.fft.ifft(np.fft.fft(a*np.conj(phase),axis=0)*kernel,axis=0)*phase).real
    def mean(a,adj=False):
        R=np.fft.fft(a*np.conj(phase),axis=0)
        b=np.einsum('kij,kj->ki',H,R) if not adj else np.einsum('kij,ki->kj',np.conj(H),R)
        return (np.fft.ifft(b,axis=0)*phase).real
    def kmv(x):
        a=x.reshape(N,6);var=v*((a[:,:3]@o.Q[:,:3].T)*s.tm)+h*((a[:,3:]@o.Q[:,3:].T)*s.tm)
        return filt(u*mean(a)+var,L).ravel()
    def kadj(x):
        q=filt(x.reshape(N,6),np.conj(L));out=mean(u*q,True)
        out[:,:3]+=(s.tm*v*q)@o.Q[:,:3];out[:,3:]+=(s.tm*h*q)@o.Q[:,3:]
        return out.ravel()
    return LinearOperator((N*6,N*6),matvec=kmv,rmatvec=kadj)

def spectrum(path,nev=12):
    path=Path(path);z=np.load(path);r=z['r'];T=float(z['T']);g=float(z['g']);K=build(System(),r,T,g)
    val,vec=eigs(K,k=nev,which='LM',tol=1e-9,ncv=max(30,2*nev+4),maxiter=500)
    order=np.argsort(abs(val-1));val=val[order];vec=vec[:,order]
    row=dict(source=str(path),g=g,T_ms=T,N=len(r),eigenvalues_K=[[v.real,v.imag] for v in val],distance_to_antiperiodic_null=float(abs(val[0]-1)),residual=float(np.linalg.norm(K@vec[:,0].real+1j*(K@vec[:,0].imag)-val[0]*vec[:,0])))
    dest=OUT/'antiperiodic'/path.parent.name;dest.mkdir(parents=True,exist_ok=True)
    (dest/(path.stem+'.json')).write_text(json.dumps(row,indent=2)+'\n');np.savez_compressed(dest/(path.stem+'.npz'),eigenvalues_K=val,vectors=vec)
    print('ANTIPERIODIC',json.dumps(row),flush=True);return row

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('paths',nargs='+');ap.add_argument('--nev',type=int,default=12);a=ap.parse_args()
    for p in a.paths:spectrum(p,a.nev)
