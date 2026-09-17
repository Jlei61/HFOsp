"""Verify expanding Floquet modes without multiplying large growth factors."""
from common import *
from periodic import Orbit
from scipy.sparse.linalg import LinearOperator,eigs
import numpy as np,argparse

def build(s,r,T,g,alpha,negative):
 N=len(r);o=Orbit(s,g,N);y=np.r_[(r/.01).ravel(),np.log(T)];_,_,meta=o.evaluate(y,r,np.zeros_like(r),True);u,v,h=meta['gains']
 half=.5 if negative else 0.;phase=np.exp(2j*np.pi*half*np.arange(N)/N)[:,None];freq=alpha+2j*np.pi*(np.fft.fftfreq(N)*N+half)/T
 W=np.einsum('kd,dij->kij',np.exp(-freq[:,None]*s.delay),o.W)
 H=s.tm[None,:,None]*W*(s.area*s.sign)[None,None,:]/((1+freq[:,None,None]*s.rise[None,None,:])*(1+freq[:,None,None]*s.decay[None,None,:]));L=1/(1+freq[:,None]*s.tr)
 def mv(x):
  a=x.reshape(N,6);ff=np.fft.fft(a*np.conj(phase),axis=0);mu=(np.fft.ifft(np.einsum('kij,kj->ki',H,ff),axis=0)*phase).real
  var=v*((a[:,:3]@o.Q[:,:3].T)*s.tm)+h*((a[:,3:]@o.Q[:,3:].T)*s.tm)
  return (np.fft.ifft(np.fft.fft((u*mu+var)*np.conj(phase),axis=0)*L,axis=0)*phase).real.ravel()
 return LinearOperator((N*6,N*6),matvec=mv)

def run(path):
 row=read(Path(path));mu=complex(*row['multipliers'][0]);assert abs(mu.imag)<1e-5;z=np.load(row['source']);T=float(z['T']);alpha=np.log(abs(mu))/T
 K=build(System(),z['r'],T,float(z['g']),alpha,mu.real<0);val,vec=eigs(K,k=64,ncv=140,which='LM',tol=1e-10,maxiter=600);j=np.argmin(abs(val-1));v=vec[:,j]
 defect=float(np.linalg.norm(K@v.real+1j*(K@v.imag)-v)/np.linalg.norm(v))
 out=dict(source=row['source'],g=float(z['g']),T_ms=T,reference_poincare=str(path),growth_rate_per_s=alpha*1000,negative_multiplier=mu.real<0,weighted_operator_eigenvalue=[val[j].real,val[j].imag],relative_periodic_mode_defect=defect)
 name=Path(row['source']).stem;write('weighted_modes/'+name+'.json',out);print(out,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');a=p.parse_args()
 for path in a.paths:run(path)
