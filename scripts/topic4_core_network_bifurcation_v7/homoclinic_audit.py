from common import *
from eigen import spectrum
from scipy.optimize import curve_fit
from scipy.signal import resample
from periodic import Orbit
import numpy as np
files=sorted((OUT/'periodic/long_period_tail').glob('T*_N8192.npz'),key=lambda p:float(np.load(p)['T']))
z=[np.load(p) for p in files];T=np.array([float(a['T']) for a in z])/1000;g=np.array([float(a['g']) for a in z]);ref=1.1218
fits=[]
for offset in (0,1,2):
 def fun(t,c,a,b):return c-a*np.exp(-b*t)
 popt,_=curve_fit(fun,T[offset:],(g[offset:]-ref)*1e6,p0=[37.147,1000,5.87],maxfev=20000)
 fits.append(dict(first_T_ms=T[offset]*1000,J_infinite_period=ref+popt[0]*1e-6,rate_per_s=popt[2],amplitude=popt[1]*1e-6,max_J_residual=float(abs(fun(T[offset:],*popt)-(g[offset:]-ref)*1e6).max()*1e-6)))
s=System();J=fits[-1]['J_infinite_period'];eq,err,ok=s.solve(J,z[-1]['equilibrium']);assert ok;sp=spectrum(s,eq,J)
right=sp[0]['v'];left=sp[0]['w'];np.savez_compressed(OUT/'homoclinic_saddle_modes.npz',r=eq,J=J,roots=np.array([x['lam'] for x in sp]),right=right,left=left)
rows=[]
for p,a in zip(files,z):
 r=a['r'];dist=np.linalg.norm(r-a['equilibrium'],axis=1)*1000
 rows.append(dict(source=str(p),g=float(a['g']),T_ms=float(a['T']),distance_to_saddle_hz=float(dist.min()),near_saddle_fraction=float((dist<.1).mean()),mean_hz=(r.mean(0)*1000).tolist()))
summary=dict(fits=fits,orbits=rows,saddle_hz=(eq*1000).tolist(),saddle_roots_per_s=[[v['lam'].real,v['lam'].imag] for v in sp],right_rate_mode_fraction=(abs(right)**2/np.sum(abs(right)**2)).tolist(),left_rate_mode_fraction=(abs(left)**2/np.sum(abs(left)**2)).tolist(),interpretation='Numerical homoclinic termination: increasing period, convergence to the same saddle, and exponential parameter convergence consistent with its unstable eigenvalue. Finite-period extrapolation, not an exact infinite-time connecting-orbit solve.')
if (OUT/'periodic/long_period_tail/T2200_N16384.npz').exists():
 b=np.load(OUT/'periodic/long_period_tail/T2200_N16384.npz');a=z[-1]
 # Phase aligns at the minimum-distance-to-saddle point in both grid runs.
 ra=a['r'];rb=b['r'];ra=np.roll(ra,-np.argmin(np.linalg.norm(ra-a['equilibrium'],axis=1)),axis=0);rb=np.roll(rb,-np.argmin(np.linalg.norm(rb-b['equilibrium'],axis=1)),axis=0)
 summary['grid']=dict(J_difference=abs(float(a['g'])-float(b['g'])),mean_difference_hz=float(abs(ra.mean(0)-rb.mean(0)).max()*1000))
write('homoclinic_audit.json',summary);print(summary,flush=True)
