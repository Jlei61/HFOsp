from common import *
from periodic import Orbit
from scipy.signal import resample
import numpy as np
s=System();rows=[]
for p in sorted((OUT/'periodic').glob('condition_*/*.npz')):
 z=np.load(p);r=z['r'];N=len(r);rr=resample(r,2*N,axis=0);o=Orbit(s,float(z['g']),2*N)
 F=o.evaluate(np.r_[(rr/.01).ravel(),np.log(float(z['T']))],rr,np.zeros_like(rr));row=dict(source=str(p),g=float(z['g']),N=N,residual=float(z['residual']),offgrid_hz=float(abs(F[:-1]).max()*10))
 assert row['residual']<1e-8 and row['offgrid_hz']<.001,row
 rows.append(row)
write('condition_orbit_validation.json',rows)
a=read(OUT/'native/per_run/prefix_validation/applied_physics.json');b=read(ROOT/'results/topic4_sef_hfo/burst_regime_map_20260914/per_run/ee1_d1_n1_t2511_s848101/applied_physics.json')
assert a['identity']==b['identity']
za=np.load(OUT/'native/per_run/prefix_validation/trajectory.npz');zb=np.load(ROOT/'results/topic4_sef_hfo/burst_regime_map_20260914/per_run/ee1_d1_n1_t2511_s848101/trajectory.npz')
for key in ('spike_counts_2ms','active_counts_2ms','active_counts_10ms'):
 assert np.array_equal(za[key],zb[key][:len(za[key])]),key
write('native_prefix_validation.json',dict(status='PASS',full_applied_identity_equal=True,bitwise_spike_counts_2ms=True,bitwise_active_counts_2ms=True,bitwise_active_counts_10ms=True,comparison_duration_ms=100,threshold_raised=a['threshold']['n_raised']))
print('AUDITED',len(rows),flush=True)
