"""Check orbit and left/right rate-mode convergence at the new critical points."""
from common import *
from periodic import Orbit
from scipy.signal import resample
import numpy as np

def main():
    pairs=[('LP0a',V5/'folds/surround_first_fold_N2048.npz',OUT/'folds/surround_first_fold_N4096.npz'),
        ('LP0b',V5/'folds/surround_second_fold_N2048.npz',OUT/'folds/surround_second_fold_N4096.npz'),
        ('LP0c',V5/'folds/surround_recruited_fold_N2048.npz',V5/'folds/surround_recruited_fold_N4096.npz'),
        ('PD0',OUT/'flips/surround_micro_flip_N2048.npz',OUT/'flips/surround_micro_flip_N4096.npz')]
    s=System();rows=[]
    for name,pa,pb in pairs:
        a=np.load(pa);b=np.load(pb);N=len(b['r']);r=resample(b['r'],2*N,axis=0)
        F=Orbit(s,float(b['g']),2*N).evaluate(np.r_[(r/.01).ravel(),np.log(float(b['T']))],r,np.zeros_like(r))
        row=dict(name=name,offgrid_defect_hz=float(abs(F[:-1]).max()*10),grid_orbit_difference_hz=float(abs(resample(a['r'],N,axis=0)-b['r']).max()*1000))
        for key in ['right','left']:
            def get(z):return z['mode' if key=='right' else 'left_mode'] if name=='PD0' else z[key+'_null'][:-1].reshape(len(z['r']),6)
            va=get(a)
            # A PD mode is antiperiodic over T. Interpolate on its true 2T
            # period so no artificial discontinuity is added at the boundary.
            x=(resample(np.r_[va,-va],2*N,axis=0)[:N] if name=='PD0' else resample(va,N,axis=0)).ravel()
            y=get(b).ravel();x/=np.linalg.norm(x);y/=np.linalg.norm(y)
            row[key+'_rate_mode_cosine']=float(abs(x@y))
        # Absolute comparison at the inherited phase; <0.001 Hz is already
        # <4e-6 of a typical peak. Do not replace a measured error by zero.
        assert row['offgrid_defect_hz']<1e-6 and row['grid_orbit_difference_hz']<.001
        assert row['right_rate_mode_cosine']>.99999 and row['left_rate_mode_cosine']>.99999
        rows.append(row);print(row,flush=True)
    write('critical_grid_and_modes.json',rows)

if __name__=='__main__':main()
