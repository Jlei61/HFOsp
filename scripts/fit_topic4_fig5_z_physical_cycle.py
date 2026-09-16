#!/usr/bin/env python3
"""Correct an observed deterministic repeating trajectory to an exact v1 cycle."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.signal import resample
from continue_topic4_fig5_z_periodic_arc import Arc,OUT


def main():
    p=argparse.ArgumentParser();p.add_argument('--N',type=int,default=256);p.add_argument('--s',type=float,default=0.);p.add_argument('--orbit');p.add_argument('--gpu',type=int,default=0);p.add_argument('--iterations',type=int,default=12);a=p.parse_args()
    arc=Arc(a.N,a.gpu);arc.deflate_phase=True;arc.gpu_krylov=True;m=arc.o.m;folder=OUT/'periodic_physical';folder.mkdir(exist_ok=True)
    from topic4_fig5_z_frequency_parallel import install
    install();arc.o.preconditioner_modes=129
    if a.orbit:
        z=np.load(a.orbit);r=resample(z['r'],a.N,axis=0);T=float(z['T']);seed=dict(source=a.orbit)
    else:
        z=np.load(OUT/'deterministic/s0.npz');raw=z['r_hz'].astype(float)/1000;t=np.arange(1,len(raw)+1,dtype=float)
        macro=(raw[:,:3200]*m.w_u).reshape(len(raw),400,8).sum(2);macro=np.c_[macro,raw[:,3200:]]
        interp=CubicSpline(t,macro);test=np.arange(2150.,2700.,2.)
        def mismatch(T):return np.linalg.norm(interp(test+T)-interp(test))**2
        fit=minimize_scalar(mismatch,bounds=(270,278),method='bounded');T=float(fit.x)
        g=macro[:,:400]@m.count_e/m.count_e.sum();end=float(t[-150+int(np.argmin(g[-150:]))]);start=end-T
        r=CubicSpline(t,raw)(start+np.arange(a.N)*T/a.N)
        seed=dict(source='deterministic/s0.npz',initial_period_ms=T,cycle_start_ms=start,cycle_end_ms=end,recurrence_mismatch=float(fit.fun))
    arc.checkpoint=folder/f'candidate_s{a.s:g}_N{a.N}.npz'
    y=arc.pack(r,T,a.s);tan=np.zeros_like(y);tan[-1]=.1;t0=time.time()
    result,error,h=arc.correct(y,tan,y,maxiter=a.iterations);r,T,s=arc.unpack(result)
    dense=resample(r,4*a.N,axis=0);row=dict(s=s,N=a.N,T_ms=T,residual_hz=error,unit_minimum_hz=float(dense.min()*1000),seconds=time.time()-t0,history=h,seed=seed)
    np.savez_compressed(folder/f's{s:g}_N{a.N}.npz',r=r,T=T,s=s,residual_hz=error,N=a.N,history=h)
    (folder/f's{s:g}_N{a.N}.json').write_text(json.dumps(row,indent=2)+'\n');print('PHYSICAL CYCLE',row,flush=True)


if __name__=='__main__':main()
