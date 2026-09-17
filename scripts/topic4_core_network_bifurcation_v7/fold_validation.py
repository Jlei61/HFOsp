from common import *
from periodic import Orbit
from scipy.signal import resample
import numpy as np
s=System();rows=[]
for p in sorted((OUT/'folds').glob('*_N4096.npz')):
    z=np.load(p);r=z['r'];N=len(r);g=float(z['g']);T=float(z['T']);meta=read(p.with_suffix('.json'));v=z['right_null'];w=z['left_null']
    der=np.fft.irfft(2j*np.pi*np.arange(N//2+1)[:,None]*np.fft.rfft(r,axis=0),n=N,axis=0);phase=der/np.sum(der*der)*.01
    o=Orbit(s,g,N);q=np.r_[(r/.01).ravel(),np.log(T)];F0=o.evaluate(q,r,phase);curv=[]
    for eps in (.03,.01,.003):
        second=(o.evaluate(q+eps*v,r,phase)-2*F0+o.evaluate(q-eps*v,r,phase))/eps**2
        curv.append(dict(step=eps,left_Fvv=float(w@second),predicted_J_curvature=float(-(w@second)/meta['left_Fg'])))
    ref=read(p.with_name(p.name.replace('_N4096.npz','_N2048.json')));a=np.load(ref['source']);mode=[]
    for key in ('right_null','left_null'):
        old=np.r_[resample(a[key][:-1].reshape(len(a['r']),6),N,axis=0).ravel(),a[key][-1]];new=z[key];mode.append(float(abs(old@new)/(np.linalg.norm(old)*np.linalg.norm(new))))
    rr=resample(r,2*N,axis=0);F=Orbit(s,g,2*N).evaluate(np.r_[(rr/.01).ravel(),np.log(T)],rr,np.zeros_like(rr))
    row=dict(name=meta['name'],g=g,N=N,J_grid_difference=abs(g-ref['g']),period_grid_difference_ms=abs(T-ref['T_ms']),right_mode_cosine=mode[0],left_mode_cosine=mode[1],offgrid_hz=float(abs(F[:-1]).max()*10),curvature_checks=curv,measured_J_curvature=meta['measured_g_curvature'],null_residual=meta['fixed_parameter_null_residual'])
    assert row['J_grid_difference']<1e-10 and row['offgrid_hz']<1e-5 and min(mode)>.9999
    assert all(x['predicted_J_curvature']*meta['measured_g_curvature']>0 for x in curv)
    rows.append(row)
write('new_fold_validation.json',dict(status='PASS',folds=rows));print(rows,flush=True)
