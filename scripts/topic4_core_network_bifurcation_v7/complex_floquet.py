"""Floquet-exponent BVP for unstable periodic orbits, avoiding huge monodromy products."""
from common import *
from periodic import Orbit
from scipy.sparse.linalg import LinearOperator,eigs
from scipy.optimize import root
import numpy as np,argparse
p=argparse.ArgumentParser();p.add_argument('reference');a=p.parse_args();row=read(Path(a.reference));z=np.load(row['source']);r=z['r'];N=len(r);T=float(z['T']);g=float(z['g']);s=System();o=Orbit(s,g,N)
_,_,meta=o.evaluate(np.r_[(r/.01).ravel(),np.log(T)],r,np.zeros_like(r),True);u,v,h=meta['gains'];cache={}
def at(x):
 key=tuple(x)
 if key not in cache:
  lam=complex(*x)/1000;freq=lam+2j*np.pi*np.fft.fftfreq(N)*N/T
  W=np.einsum('kd,dij->kij',np.exp(-freq[:,None]*s.delay),o.W)
  H=s.tm[None,:,None]*W*(s.area*s.sign)[None,None,:]/((1+freq[:,None,None]*s.rise[None,None,:])*(1+freq[:,None,None]*s.decay[None,None,:]));L=1/(1+freq[:,None]*s.tr)
  def mv(x):
   q=x.reshape(N,6);mu=np.fft.ifft(np.einsum('kij,kj->ki',H,np.fft.fft(q,axis=0)),axis=0)
   var=v*((q[:,:3]@o.Q[:,:3].T)*s.tm)+h*((q[:,3:]@o.Q[:,3:].T)*s.tm)
   return np.fft.ifft(np.fft.fft(u*mu+var,axis=0)*L,axis=0).ravel()
  K=LinearOperator((N*6,N*6),matvec=mv,dtype=complex);val,vec=eigs(K,k=64,ncv=140,which='LM',tol=2e-10,maxiter=600,v0=np.random.default_rng(7).normal(size=N*6).astype(complex));j=np.argmin(abs(val-1));cache[key]=(val[j],vec[:,j],K)
  print('COMPLEX_MODE',x,val[j],flush=True)
 return cache[key]
def fun(x):
 val=at(x)[0]-1;return [val.real,val.imag]
mu=complex(*row['multipliers'][0]);initial=np.array([np.log(abs(mu))*1000/T,np.angle(mu)*1000/T]);sol=root(fun,initial,tol=1e-8,options={'eps':1e-5});val,vec,K=at(sol.x)
res=float(np.linalg.norm(K@vec-vec)/np.linalg.norm(vec));out=dict(source=row['source'],reference=a.reference,g=g,T_ms=T,growth_per_s=float(sol.x[0]),frequency_rad_per_s=float(sol.x[1]),multiplier_modulus=float(np.exp(sol.x[0]*T/1000)),relative_mode_defect=res,solver_success=bool(sol.success),operator_eigenvalue=[val.real,val.imag],scope='An independently verified unstable Floquet exponent; not a claim to enumerate the full spectrum.')
assert res<1e-7 and sol.x[0]>0,out
name=Path(row['source']).stem;write('complex_floquet/'+name+'.json',out);np.savez_compressed(OUT/'complex_floquet'/f'{name}.npz',mode=vec.reshape(N,6),exponent_per_s=complex(*sol.x));print('VERIFIED_COMPLEX_MODE',out,flush=True)
