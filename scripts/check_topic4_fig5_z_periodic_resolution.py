#!/usr/bin/env python3
"""Independent denser phase-grid check of the largest displayed v1 cycles."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json,gc
import numpy as np
from scipy.signal import resample
from topic4_fig5_z_frozen_v1 import Orbit,OUT


def main():
    rows=[]
    for core,folder in [('A','periodic_from_crossing'),('B','periodic_from_crossing_b')]:
        candidates=[]
        for f in (OUT/folder).glob('amp*_N*.npz'):
            a=np.load(f)
            if float(a['residual_hz'])<1e-5:candidates.append((float(a['control_amplitude_hz']),len(a['r']),f))
        amp,N,f=max(candidates);a=np.load(f);r=resample(a['r'],2*N,axis=0);s=float(a['s']);T=float(a['T'])
        o=Orbit(s,2*N);defect=float(abs(o.evaluate_fixed(r,T)).max()*1000);m=o.m
        rr=resample(r,8*N,axis=0);g=((rr[:,:o.U]*m.w_u).reshape(8*N,m.n,m.K).sum(2)@m.count_e)/m.count_e.sum()*1000
        row=dict(core=core,control_amplitude_hz=amp,file=str(f),fitted_N=N,validation_N=2*N,
                 map_defect_hz_on_fitted_grid=float(a['residual_hz']),map_defect_hz_on_double_grid=defect,
                 unit_minimum_hz=float(rr.min()*1000),global_minimum_hz=float(g.min()),global_maximum_hz=float(g.max()),global_mean_hz=float(g.mean()),
                 acceptance=bool(defect<1e-3 and rr.min()*1000>=-.01))
        rows.append(row);print(row,flush=True)
        (OUT/'periodic_resolution_qa.json').write_text(json.dumps(rows,indent=2)+'\n')
        del o,r,rr,a;gc.collect()


if __name__=='__main__':main()
