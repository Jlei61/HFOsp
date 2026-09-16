#!/usr/bin/env python3
"""Adaptive secant continuation of the unchanged filtered-v1 periodic families."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.signal import resample
from topic4_fig5_z_cycle_from_crossing import Branch
from topic4_fig5_z_frozen_v1 import OUT as PRIOR,BASE

OUT=BASE.parent/'fig5_z_branch_extension_20260915'


def main():
    p=argparse.ArgumentParser();p.add_argument('--core',choices=['a','b'],required=True);p.add_argument('--N',type=int,default=64)
    p.add_argument('--max-points',type=int,default=45);p.add_argument('--max-amplitude',type=float,default=100.);p.add_argument('--step',type=float,default=.5);p.add_argument('--precondition',action='store_true')
    a=p.parse_args();dest=OUT/f'periodic_{a.core}_N{a.N}';dest.mkdir(parents=True,exist_ok=True)
    b=Branch(a.N,a.core);b.destination=dest;b.use_preconditioner=a.precondition
    folder=PRIOR/('periodic_from_crossing_b' if a.core=='b' else 'periodic_from_crossing')
    byamp={}
    for f in list(folder.glob('amp*_N*.npz'))+list(dest.glob('amp*_N*.npz')):
        z=np.load(f);amp=float(z['control_amplitude_hz'])
        if float(z['residual_hz'])<1e-5 and (amp not in byamp or len(z['r'])>len(byamp[amp]['r'])):byamp[amp]={k:z[k] for k in z.files}
    levels=sorted(byamp)[-2:];states=[]
    for amp in levels:
        z=byamp[amp];states.append((amp,resample(z['r'],a.N,axis=0),float(z['T']),float(z['s'])))
    rows=[];step=a.step;t0=time.time();attempt=0
    while len(rows)<a.max_points:
        old,cur=states[-2:];amp=cur[0]+step
        if amp>a.max_amplitude:break
        factor=(amp-cur[0])/(cur[0]-old[0])
        r=cur[1]+factor*(cur[1]-old[1]);T=np.exp(np.log(cur[2])+factor*np.log(cur[2]/old[2]));s=cur[3]+factor*(cur[3]-old[3])
        print('PREDICT',a.core,amp,s,T,'step',step,flush=True)
        rr,tt,ss,err=b.solve(r,T,s,amp,maxiter=10);attempt+=1
        history=np.load(dest/f'amp{amp:g}_N{a.N}.npz')['history']
        row=dict(attempt=attempt,amplitude_hz=amp,N=a.N,s=ss,period_ms=tt,error_hz=err,iterations=len(history),accepted=bool(err<1e-5),seconds=time.time()-t0)
        rows.append(row);(dest/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');print('EXTENSION',row,flush=True)
        if err>=1e-5:
            step*=.5
            if step<.025:break
            continue
        states.append((amp,rr,tt,ss));states=states[-2:]
        if ss>=b.o.eq.ss[-1] or ss<-.5:break
        # Small predictor increments around strongly curved parts, growing only
        # after fast correctors. No failed root is interpreted as a bifurcation.
        if len(history)<=4:step=min(step*1.35,3.)
        elif len(history)>=7:step=max(step*.65,.025)
    (dest/'status.json').write_text(json.dumps(dict(status='BOUNDED_EXTENSION_COMPLETE',accepted=sum(x['accepted'] for x in rows),attempts=len(rows),last_s=states[-1][3],last_amplitude_hz=states[-1][0]),indent=2)+'\n')


if __name__=='__main__':main()
