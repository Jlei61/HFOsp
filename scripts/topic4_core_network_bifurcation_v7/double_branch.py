"""Actual doubled orbits with amplitude constraint; no ill-conditioned tangent solve."""
from common import *
from folds import Chart,metric
from scipy.sparse.linalg import gmres
import numpy as np,argparse
p=argparse.ArgumentParser();p.add_argument('--source',default=str(OUT/'flips/surround_2T_flip_N4096.npz'));p.add_argument('--name',default='surround_period4');p.add_argument('--from-orbit');p.add_argument('--amps',type=float,nargs='+',default=[.00001,.00002,.00004,.00008]);a=p.parse_args()
z=np.load(a.source);base=z['r'];mode=z['mode'];N=2*len(base);scale=1e-6
ww=np.r_[mode,-mode];t=np.r_[ww.ravel(),0.,0.];t/=np.sqrt(metric(t,t,N))
z0=np.r_[(np.tile(base,(2,1))/.01).ravel(),np.log(2*float(z['T'])),float(z['g'])/scale]
chart=Chart(System(),z0,t,N,scale);last=z0;old=0;dest=OUT/'periodic'/a.name;dest.mkdir(parents=True,exist_ok=True)
if a.from_orbit:
 u=np.load(a.from_orbit);last=np.r_[(u['r']/.01).ravel(),np.log(float(u['T'])),float(u['g'])/scale];old=float(u['amplitude'])
for amp in a.amps:
 q=last+(amp-old)*t
 for k in range(18):
  F,B,*_=chart.evaluate(q,amp);e=float(abs(F).max());print('DOUBLE_NEWTON',amp,k,e,flush=True)
  if e<1e-12:break
  dq,info=gmres(B,-F,rtol=1e-4,atol=1e-13,restart=180,maxiter=6)
  alpha=1.
  for _ in range(14):
   trial=q+alpha*dq
   if np.linalg.norm(chart.evaluate(trial,amp,False))<np.linalg.norm(F):break
   alpha*=.5
  else:raise RuntimeError(('double line search',amp,k,e,info))
  q=trial
 else:raise RuntimeError('double convergence')
 r=q[:-2].reshape(N,6)*.01;T=np.exp(q[-2]);g=q[-1]*scale;path=dest/f'amp{amp:g}_N{N}.npz'
 np.savez_compressed(path,r=r,T=T,g=g,N=N,residual=e,amplitude=amp,gscale=scale,amplitude_direction=t)
 row=dict(source=str(path),parent=a.source,g=g,T_ms=T,N=N,residual=e,amplitude=amp,mean_hz=(r.mean(0)*1000).tolist(),half_period_difference=float(np.linalg.norm(r-np.roll(r,N//2,axis=0))/np.linalg.norm(r-r.mean(0))))
 path.with_suffix('.json').write_text(json.dumps(row,indent=2)+'\n');print('DOUBLED',row,flush=True);last=q;old=amp
