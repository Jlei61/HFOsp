#!/usr/bin/env python3
"""Find equilibria near prior high-activity states, allowing the path parameter to move."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
import numpy as np
from scipy.linalg import lstsq
from topic4_fig5_z_bifurcation_preview import Equilibrium,SOURCE,OUT as PRIOR
from topic4_fig5_z_branch_dynamics import accelerate

OUT=PRIOR.parent/'fig5_z_branch_extension_20260915'


def main():
    eq=Equilibrium();accelerate(eq.m);rows=[]
    for t in [9000,9420,8000,9870]:
        a=np.load(SOURCE/f'approx/v1/runs/z{t}_h8000_W1/fields.npz');r=a['fields_hz'][-4000:].mean(0).ravel().astype(float);s=float(eq.ss[eq.times.index(t/1000)]);y=np.r_[r/1000,s];hist=[]
        for it in range(45):
            r=y[:-1]*1000;s=float(y[-1]);f,j=eq.evaluate(r,s,True);err=float(abs(f).max());hist.append(err);print('FREE ROOT',t,it,s,err,flush=True)
            if err<1e-6:break
            eps=1e-5;fs=(eq.evaluate(r,s+eps)-eq.evaluate(r,s-eps))/(2*eps)/1000
            A=np.c_[j,fs];delta=lstsq(A,-f/1000,lapack_driver='gelsy',check_finite=False)[0]
            for back in range(14):
                trial=y+2.**(-back)*delta
                if not 0<=trial[-1]<=eq.ss[-1]:continue
                ff=eq.evaluate(trial[:-1]*1000,trial[-1])
                if np.linalg.norm(ff)<np.linalg.norm(f):y=trial;break
            else:break
        r=y[:-1]*1000;s=float(y[-1]);err=float(abs(eq.evaluate(r,s)).max());ok=bool(err<1e-5 and r.min()>-1e-7)
        row=dict(seed_t_ms=t,s=s,mean_z=1-s,mean_e_hz=float(np.average(r[:400],weights=eq.m.count_e)),error_hz=err,valid=ok,history=hist);rows.append(row);print('FREE RESULT',row,flush=True)
        if ok:np.savez_compressed(OUT/f'free_parameter_root_t{t}.npz',r_hz=r,s=s,residual_hz=err)
        (OUT/'free_parameter_roots.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
