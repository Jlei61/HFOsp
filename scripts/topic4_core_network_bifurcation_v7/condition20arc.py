from common import *
from folds import Chart,metric
from scipy.signal import resample
from scipy.optimize import brentq
import numpy as np
rows=[r for r in read(OUT/'arcs/low_fast/progress.json') if r['tangent_g']>0];g=1.12183;N=4096;scale=.01
for left,right in zip(rows[:-1],rows[1:]):
    if left['g']<=g<=right['g']:break
else:raise ValueError('no bracket')
def load(p):
 z=np.load(p);return np.r_[(resample(z['r'],N,axis=0)/.01).ravel(),np.log(float(z['T'])),float(z['g'])/scale]
a=load(left['source']);b=load(right['source']);t=b-a;t/=np.sqrt(metric(t,t,N));span=metric(t,b-a,N);chart=Chart(System(),a,t,N,scale);cache={}
def at(x):
 if x not in cache:
  z,t,e,*_=chart.solve(a+(b-a)*x/span,x);cache[x]=(z,t,e);print('FIXED_ARC',x,z[-1]*scale,e,flush=True)
 return cache[x]
x=brentq(lambda x:at(x)[0][-1]*scale-g,0,span,xtol=1e-9);z,t,e=at(x);r=z[:-2].reshape(N,6)*.01;T=np.exp(z[-2]);dest=OUT/'periodic/condition_20b';dest.mkdir(parents=True,exist_ok=True)
np.savez_compressed(dest/'g1.12183000_N4096.npz',r=r,T=T,g=z[-1]*scale,residual=e,N=N)
write('condition20_arc.json',dict(g=z[-1]*scale,target=g,residual=e,T_ms=T,left=left['source'],right=right['source']))
