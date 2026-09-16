#!/usr/bin/env python3
"""Use prior v1 trajectories only as starting guesses for exact equilibria."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,pickle
import numpy as np
from topic4_fig5_z_bifurcation_preview import Equilibrium,SOURCE,OUT as PRIOR

OUT=PRIOR.parent/'fig5_z_branch_extension_20260915'


def main():
    eq=Equilibrium();m=eq.m;rows=[]
    for t in [9420,9870,8000,9000,9300,10370]:
        folder=SOURCE/f'approx/v1/runs/z{t}_h8000_W1'
        a=np.load(folder/'fields.npz');rmean=a['fields_hz'][-4000:].mean(0).ravel()
        with (folder/'end_state.pkl').open('rb') as f:st=pickle.load(f)
        rend=np.r_[(st['r_u']*m.w_u).reshape(m.n,m.K).sum(1),st['r_i']]*1000
        s=float(eq.ss[eq.times.index(t/1000)])
        for label,x in [('tail_mean',rmean),('terminal',rend)]:
            r,error,ok=eq.solve(x,s);ok=bool(ok);row=dict(t_ms=t,s=s,initialization=label,valid=ok,error_hz=error,mean_e_hz=float(np.average(r[:m.n],weights=m.count_e)))
            rows.append(row);print(row,flush=True)
            if ok:np.savez_compressed(OUT/f'physical_equilibrium_t{t}_{label}.npz',r_hz=r,s=s,residual_hz=error)
            (OUT/'physical_equilibrium_seeds.json').write_text(json.dumps(rows,indent=2)+'\n')
            if ok:break


if __name__=='__main__':main()
