from common import *
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
import numpy as np
base=np.load(OUT/'periodic/candidate_saddle_cycle/g1.17626000_N2048.npz');b=base['r'];Tb=float(base['T']);b=np.roll(b,-np.argmax(b[:,0]),axis=0)
bf=CubicSpline(np.linspace(0,Tb,len(b)+1),np.r_[b,b[:1]],bc_type='periodic');tt=np.arange(1024)*Tb/1024;reference=bf(tt);scale=np.sqrt(np.mean((reference-reference.mean(0))**2));out=[]
rows=read(V5/'arcs/surround_recruited_back/progress.json')[-1:]+read(OUT/'arcs/recruited_fast/progress.json')[::5]
for row in rows:
 z=np.load(row['source']);r=z['r'];T=float(z['T']);peaks,_=find_peaks(r[:,0],height=.05,distance=max(1,round(50/T*len(r))));rf=CubicSpline(np.linspace(0,T,len(r)+1),np.r_[r,r[:1]],bc_type='periodic');best=None
 for peak in peaks:
  offset=peak*T/len(r);window=rf((tt+offset)%T);rms=float(np.sqrt(np.mean((window-reference)**2))*1000)
  if best is None or rms<best['rms_hz']:best=dict(rms_hz=rms,relative_rms=rms/(scale*1000),offset_ms=offset)
 out.append(dict(source=row['source'],g=float(z['g']),T_ms=T,closest_periodic_window=best))
write('candidate_cycle_shadow.json',out);print(out,flush=True)
