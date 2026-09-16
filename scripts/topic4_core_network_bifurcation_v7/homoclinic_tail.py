"""Long-period continuation at fixed T, testing approach to a saddle equilibrium."""
from common import *
from folds import Chart
from scipy.sparse.linalg import gmres
from scipy.interpolate import CubicSpline
from periodic import Orbit
from scipy.signal import resample
import numpy as np,argparse
p=argparse.ArgumentParser();p.add_argument('--periods',type=float,nargs='+',default=[1000,1200,1500,1800,2200]);p.add_argument('--N',type=int,default=8192);p.add_argument('--source');a=p.parse_args()
rows=read(OUT/'arcs/low_fast/progress.json');source=a.source or rows[-1]['source'];old=np.load(source);r=old['r'];T=float(old['T']);g=float(old['g']);s=System();N=a.N;scale=.01;dest=OUT/'periodic/long_period_tail';dest.mkdir(parents=True,exist_ok=True);out=[]
for target in a.periods:
 if target<T-.01:continue
 eq,err,ok=s.solve(g,np.array([.5,.3,0,0,0,0])/1000);assert ok
 closest=np.argmin(np.linalg.norm(r-eq,axis=1));r=np.roll(r,-closest,axis=0)
 spl=CubicSpline(np.linspace(0,T,len(r)+1),np.r_[r,r[:1]],bc_type='periodic')
 tt=np.arange(N)*target/N;delta=target-T
 guess=spl(np.clip(tt-delta/2,0,T));q=np.r_[(guess/.01).ravel(),np.log(target),g/scale]
 normal=np.zeros(len(q));normal[-2]=1;chart=Chart(s,q,normal,N,scale)
 for k in range(18):
  F,B,*_=chart.evaluate(q,0);e=float(abs(F).max());print('LONG_NEWTON',target,k,e,flush=True)
  if e<2e-11:break
  step,info=gmres(B,-F,rtol=1e-5,atol=2e-12,restart=180,maxiter=8);alpha=1
  for _ in range(14):
   test=q+alpha*step
   if np.linalg.norm(chart.evaluate(test,0,False))<np.linalg.norm(F):break
   alpha*=.5
  else:raise RuntimeError(('long period line search',target,k,e))
  q=test
 else:raise RuntimeError('long period convergence')
 r=q[:-2].reshape(N,6)*.01;T=float(np.exp(q[-2]));g=float(q[-1]*scale);eq,err,ok=s.solve(g,eq);assert ok
 rr=resample(r,2*N,axis=0);F2=Orbit(s,g,2*N).evaluate(np.r_[(rr/.01).ravel(),np.log(T)],rr,np.zeros_like(rr));dist=np.linalg.norm(r-eq,axis=1)*1000
 path=dest/f'T{target:g}_N{N}.npz';np.savez_compressed(path,r=r,T=T,g=g,N=N,residual=e,equilibrium=eq)
 row=dict(g=g,T_ms=T,N=N,residual=e,offgrid_hz=float(abs(F2[:-1]).max()*10),nearest_saddle_distance_hz=float(dist.min()),fraction_within_01hz=float((dist<.1).mean()),equilibrium_hz=(eq*1000).tolist(),mean_hz=(r.mean(0)*1000).tolist(),source=str(path))
 out.append(row);write('long_period_tail.json',out);print('LONG_ACCEPT',row,flush=True)
