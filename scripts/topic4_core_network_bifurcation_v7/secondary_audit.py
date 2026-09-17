from common import *
from periodic import Orbit
from scipy.signal import resample
import numpy as np
s=System();pa=OUT/'flips/surround_2T_flip_N4096.npz';pb=OUT/'flips/surround_2T_flip_N8192.npz';a=np.load(pa);b=np.load(pb);N=len(b['r']);row=dict(J_grid_difference=abs(float(a['g'])-float(b['g'])),period_grid_difference_ms=abs(float(a['T'])-float(b['T'])),waveform_difference_hz=float(abs(resample(a['r'],N,axis=0)-b['r']).max()*1000))
for key in ['mode','left_mode']:
 v=resample(np.r_[a[key],-a[key]],2*N,axis=0)[:N].ravel();w=b[key].ravel();row[key+'_cosine']=float(abs(v@w)/(np.linalg.norm(v)*np.linalg.norm(w)))
children=[]
for p in sorted((OUT/'periodic/surround_period4_refined').glob('amp*.npz')):
 z=np.load(p);r=z['r'];n=len(r);rr=resample(r,2*n,axis=0);F=Orbit(s,float(z['g']),2*n).evaluate(np.r_[(rr/.01).ravel(),np.log(float(z['T']))],rr,np.zeros_like(rr));dif=float(abs(r-np.roll(r,n//2,axis=0)).max()*1000)
 c=dict(source=str(p),J=float(z['g']),T_ms=float(z['T']),amplitude=float(z['amplitude']),offgrid_hz=float(abs(F[:-1]).max()*10),half_period_max_difference_hz=dif,residual=float(z['residual']))
 assert c['offgrid_hz']<1e-6 and dif>1000*c['offgrid_hz']
 children.append(c)
assert row['J_grid_difference']<1e-12 and row['mode_cosine']>.99999 and row['left_mode_cosine']>.99999
write('secondary_flip_validation.json',dict(status='PASS',critical=row,accepted_children=children));print(row,children,flush=True)
