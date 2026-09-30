"""Scaled pseudo-arclength continuation through stationary Z folds.

The tangent follows the SAME physical-D residual, not independent solves
joined over a gap. Temporal stability is an independent later calculation.
"""
from zm_model import *
from scipy.sparse.linalg import spsolve,eigs
import argparse

RS=.01
DS=.1

def tangent(s,r,D,previous=None,direction=1):
    A=s.jacobian(r,D)*RS;b=s.parameter_derivative(r,D)*DS
    if previous is None:
        v=np.r_[spsolve(A.tocsc(),-b),1.]*direction
    else:
        A=sparse.bmat([[A,sparse.csr_matrix(b[:,None])],
            [sparse.csr_matrix(previous[None,:-1]),sparse.csr_matrix([[previous[-1]]])]],format='csc')
        v=spsolve(A,np.r_[np.zeros(s.P),1.])
    v/=np.linalg.norm(v)
    if previous is not None and v@previous<0:v=-v
    return v

def correct(s,pred,t,maxiter=16):
    x=pred.copy()
    for k in range(maxiter):
        r=x[:-1]*RS;D=x[-1]*DS
        if not 0<=D<=1:return x,False,k
        f=np.r_[s.residual(r,D),t@(x-pred)*RS]
        if abs(f).max()<2e-11:
            # Rates as small as 1e-70 occur in suppressed spatial groups.
            # Remove only roundoff-sized negatives, then recheck the equation.
            if np.min(x[:-1]*RS)<-1e-12:return x,False,k
            x[:-1]=np.maximum(x[:-1],0)
            if abs(s.residual(x[:-1]*RS,x[-1]*DS)).max()<2e-11:return x,True,k
        A=s.jacobian(r,D)*RS;b=s.parameter_derivative(r,D)*DS
        B=sparse.bmat([[A,sparse.csr_matrix(b[:,None])],
            [sparse.csr_matrix((t[:-1]*RS)[None,:]),sparse.csr_matrix([[t[-1]*RS]])]],format='csc')
        change=spsolve(B,-f);step=1.
        for j in range(22):
            trial=x+step*change;rr=trial[:-1]*RS;dd=trial[-1]*DS
            if 0<=dd<=1 and np.all(rr>=-1e-12) and np.all(rr<1/s.ref):
                rr=np.maximum(rr,0);trial[:-1]=rr/RS
                ff=np.r_[s.residual(rr,dd),t@(trial-pred)*RS]
                if np.linalg.norm(ff)<np.linalg.norm(f):x=trial;break
            step*=.5
        else:return x,False,k
    return x,False,maxiter

def main(a):
    s=ZMRate(a.grid,mode=a.mode,m_current=a.m_current)
    dest=DEST/f'g{a.grid}'/a.label;dest.mkdir(parents=True,exist_ok=False)
    initial=None if a.direction==1 else .85/s.ref
    # Descend from D=1 so that starting on the upper branch is reproducible.
    if a.direction==-1 and a.resume is None:
        for d in np.linspace(1,a.start,50):
            initial,ok,tr=s.solve_D(float(d),initial)
            if not ok:raise RuntimeError((d,tr[-1]))
    if a.resume is None:
        r,ok,tr=s.solve_D(a.start,initial);assert ok
        D=a.start;t=tangent(s,r,D,direction=a.direction)
    else:
        seed=np.load(a.resume);r=seed['r'];D=float(seed['D']);t=seed['tangent']
        if a.reverse_tangent:t=-t
        assert abs(s.residual(r,D)).max()<1e-9
    ds=a.ds;rows=[];status='RUNNING';folds=[];quality=None
    for k in range(a.steps):
        J=s.jacobian(r,D);ev,vec=eigs(J,k=3,sigma=0,tol=1e-10)
        row=dict(index=k,D=D,global_E_hz=s.global_rate(r),regional_hz=s.regional_rates(r),
            equilibrium_residual_hz=float(abs(s.residual(r,D)).max()*1000),
            stationary_eigenvalues=ev,tangent_D=float(t[-1]),step_size=ds,
            incoming_step_quality=quality)
        np.savez_compressed(dest/f'point{k:04d}.npz',r=r,D=D,tangent=t,Z=s.Z,eigenvalues=ev,eigenvectors=vec)
        rows.append(row)
        if k and rows[-1]['tangent_D']*rows[-2]['tangent_D']<0:
            folds.append([k-1,k]);print('FOLD BRACKET',folds[-1],flush=True)
        write(dest/'result.json',dict(status=status,rows=rows,fold_brackets=folds,mode=a.mode,
            temporal_stability='NOT_YET_ASSIGNED',bifurcation_types='FOLD_BRACKETS_REQUIRE_REFINEMENT'))
        print(k,D,s.global_rate(r),t[-1],flush=True)
        if a.stop_rate_above is not None and s.global_rate(r)>=a.stop_rate_above:
            status='REQUESTED_RATE_REACHED';break
        if a.stop_rate_below is not None and s.global_rate(r)<=a.stop_rate_below:
            status='REQUESTED_RATE_REACHED';break
        x=np.r_[r/RS,D/DS]
        for attempt in range(24):
            nextx,ok,nit=correct(s,x+ds*t,t)
            if ok:
                nt=tangent(s,nextx[:-1]*RS,float(nextx[-1]*DS),t)
                cosine=float(nt@t)
                correction=float(np.linalg.norm(nextx-(x+ds*t))/ds)
                quality=dict(tangent_cosine=cosine,correction_fraction=correction,
                    actual_step=float(np.linalg.norm(nextx-x)),accepted_ds=ds)
                if cosine>=a.min_tangent_cos and correction<=a.max_correction_fraction:break
                ok=False
            ds*=.5
        if not ok:status='CONTINUATION_STOPPED_NONCONVERGENCE';break
        r=nextx[:-1]*RS;D=float(nextx[-1]*DS);t=nt
        if nit<=4 and cosine>.98 and correction<.1:ds=min(ds*1.15,a.max_ds)
        if nit>=10:ds=max(ds*.65,1e-5)
        if not .000001<D<.999999:status='PHYSICAL_DOMAIN_REACHED';break
    else:status='REQUESTED_SEGMENT_COMPLETE'
    write(dest/'result.json',dict(status=status,rows=rows,fold_brackets=folds,mode=a.mode,
        temporal_stability='NOT_YET_ASSIGNED',bifurcation_types='FOLD_BRACKETS_REQUIRE_REFINEMENT'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grid',type=int,default=20);p.add_argument('--start',type=float,default=.01)
    p.add_argument('--mode',choices=['dynamic_M','frozen_M'],default='dynamic_M');p.add_argument('--m-current',type=float,default=0.)
    p.add_argument('--direction',type=int,choices=[-1,1],default=1);p.add_argument('--steps',type=int,default=130)
    p.add_argument('--ds',type=float,default=.015);p.add_argument('--max-ds',type=float,default=.08)
    p.add_argument('--min-tangent-cos',type=float,default=0.)
    p.add_argument('--max-correction-fraction',type=float,default=float('inf'))
    p.add_argument('--stop-rate-above',type=float);p.add_argument('--stop-rate-below',type=float)
    p.add_argument('--reverse-tangent',action='store_true')
    p.add_argument('--resume');p.add_argument('--label',default='D_arclength_lower');main(p.parse_args())
