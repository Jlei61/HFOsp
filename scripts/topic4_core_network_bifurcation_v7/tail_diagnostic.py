from common import *
import numpy as np
s=System();eq=read(V2/'equilibrium_spectrum.json')
for family in ['low','recruited']:
 path=OUT/'arcs'/f'{family}_fast/progress.json';rows=read(path)
 for row in rows[-3:]:
  z=np.load(row['source']);r=z['r'];g=float(z['g']);out=[]
  for direction in [-1,1]:
   e=min([x for x in eq if x['direction']==direction],key=lambda a:abs(a['g']-g));rr,err,ok=s.solve(g,np.array(e['r_hz'])/1000)
   if ok:out.append(dict(direction=direction,equilibrium_hz=(rr*1000).tolist(),closest_distance_hz=float(np.linalg.norm(r-rr,axis=1).min()*1000)))
  print(family,g,float(z['T']),out,flush=True)
