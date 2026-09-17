from common import *
from eigen import spectrum
import numpy as np
s=System();rows=read(OUT/'arcs/low_fast/progress.json');out=[]
for row in rows[-1:]:
 z=np.load(row['source']);r=z['r'];g=float(z['g']);roots=[]
 for aa in [.2,.42,.5,.6,.8,1.,2.,5.,15.,30.]:
  for bb in [.3,.5,1.,10.,30.]:
   initial=np.array([aa,bb,0.,max(0,(aa-2)*.8),max(0,(bb-2)*.8),0.])/1000
   rr,err,ok=s.solve(g,initial)
   if not ok or any(np.linalg.norm(rr-v['r'])<1e-7 for v in roots):continue
   roots.append(dict(r=rr,err=err))
 for item in roots:
  rr=item['r'];dist=np.linalg.norm(r-rr,axis=1)*1000;sp=spectrum(s,rr,g)
  result=dict(g=g,T_ms=float(z['T']),r_hz=(rr*1000).tolist(),distance_hz=float(min(dist)),fraction_within_1hz=float((dist<1).mean()),roots=[[v['lam'].real,v['lam'].imag] for v in sp],source=row['source'])
  out.append(result);print(result,flush=True)
write('low_tail_equilibria.json',out)
