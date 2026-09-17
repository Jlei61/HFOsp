from common import *
from orbits import save_solve
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
import numpy as np
rows=read(OUT/'arcs/recruited_fast/progress.json');source=rows[-1]['source'];z=np.load(source);r=z['r'];T=float(z['T']);g=float(z['g']);N=len(r)
pk,_=find_peaks(r[:,0],height=.1,distance=round(80/T*N));candidates=[]
for a,b in zip(pk[:-1],pk[1:]):
 period=(b-a)*T/N
 if 100<period<250 and r[a:b,2].max()<.02:candidates.append((float(np.linalg.norm(r[a]-r[b])),a,b))
assert candidates
candidates.sort();err,a,b=candidates[0];tt=np.arange(N)*T/N;cs=CubicSpline(np.r_[tt,T],np.r_[r,r[:1]],bc_type='periodic');period=(b-a)*T/N;guess=cs(a*T/N+np.arange(2048)*period/2048)
print('SHORT_SEED',g,period,err,source,flush=True)
path,row=save_solve(System(),g,guess,period,2048,dict(source=source,window_indices=[int(a),int(b)],initial_boundary_difference=err),'second_saddle_cycle')
from poincare import compute
compute(path,.025,'orthogonal','rk4',3)
