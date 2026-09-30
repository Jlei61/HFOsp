"""Pseudo-arclength continuation of equilibria in D on the prescribed Z path, with fold refinement.

Static Jacobian only (temporal stability is a separate calculation with the dynamic closure).
Records every accepted point; folds (dD/ds sign change) are refined by bisection on the tangent.
"""
from model_v3 import *
from scipy.sparse.linalg import eigs
import argparse
RS=.01;DS=.1

def param_derivative(s,r,h=1e-6):
    D=s.D;hh=min(h,D,1-D) if 0<D<1 else h
    if D-hh<0:
        s.set_D(D);f0=s.residual(r);s.set_D(D+hh);f1=s.residual(r);s.set_D(D);return (f1-f0)/hh
    if D+hh>1:
        s.set_D(D);f0=s.residual(r);s.set_D(D-hh);f1=s.residual(r);s.set_D(D);return (f0-f1)/hh
    s.set_D(D+hh);fp=s.residual(r);s.set_D(D-hh);fm=s.residual(r);s.set_D(D);return (fp-fm)/(2*hh)

def tangent(s,r,D,previous=None,direction=1):
    s.set_D(D);A=s.jacobian(r)*RS;b=param_derivative(s,r)*DS
    if previous is None:v=np.r_[spsolve(A,-b),1.]*direction
    else:
        B=sparse.bmat([[A,sparse.csr_matrix(b[:,None])],[sparse.csr_matrix(previous[None,:-1]),sparse.csr_matrix([[previous[-1]]])]],format='csc')
        v=spsolve(B,np.r_[np.zeros(s.P),1.])
    v/=np.linalg.norm(v)
    if previous is not None and v@previous<0:v=-v
    return v

def correct(s,pred,t,maxiter=16,tol=2e-11):
    x=pred.copy()
    for k in range(maxiter):
        r=x[:-1]*RS;D=x[-1]*DS
        if not 0<=D<=1:return x,False,k
        s.set_D(D);f=np.r_[s.residual(r),t@(x-pred)*RS]
        if abs(f).max()<tol:
            if np.min(r)<-1e-12:return x,False,k
            x[:-1]=np.maximum(x[:-1],0)
            if abs(s.residual(x[:-1]*RS)).max()<tol:return x,True,k
        A=s.jacobian(r)*RS;b=param_derivative(s,r)*DS
        B=sparse.bmat([[A,sparse.csr_matrix(b[:,None])],[sparse.csr_matrix((t[:-1]*RS)[None,:]),sparse.csr_matrix([[t[-1]*RS]])]],format='csc')
        change=spsolve(B,-f);step=1.
        for j in range(22):
            trial=x+step*change;rr=trial[:-1]*RS;dd=trial[-1]*DS
            if 0<=dd<=1 and np.all(rr>=-1e-12) and np.all(rr<1/s.ref):
                rr=np.maximum(rr,0);trial[:-1]=rr/RS;s.set_D(dd)
                ff=np.r_[s.residual(rr),t@(trial-pred)*RS]
                if np.linalg.norm(ff)<np.linalg.norm(f):x=trial;break
            step*=.5
        else:return x,False,k
    return x,False,maxiter

def static_eigs(s,r,D,k=4):
    s.set_D(D);J=s.jacobian(r)
    try:ev,vec=eigs(J,k=k,sigma=0,tol=1e-10);return ev,vec
    except Exception as e:return np.array([np.nan]),None

def refine_fold(s,L,R):
    xl=np.r_[L['r']/RS,L['D']/DS];xr=np.r_[R['r']/RS,R['D']/DS];tl=L['tangent'];tr=R['tangent']
    for k in range(40):
        chord=xr-xl;chord/=np.linalg.norm(chord);xm,ok,nit=correct(s,(xl+xr)/2,chord)
        if not ok:return None
        r=xm[:-1]*RS;D=float(xm[-1]*DS);tm=tangent(s,r,D,tl)
        if abs(tm[-1])<1e-11 or np.linalg.norm(xr-xl)<1e-10:break
        if tm[-1]*tl[-1]>0:xl=xm;tl=tm
        else:xr=xm;tr=tm
    ev,V=static_eigs(s,r,D,k=4);sel=np.argmin(abs(ev));v=V[:,sel].real;v/=np.linalg.norm(v)
    s.set_D(D);J=s.jacobian(r);ew,W=eigs(J.T,k=4,sigma=0,tol=1e-12);w=W[:,np.argmin(abs(ew))].real;w/=w@v
    fd=param_derivative(s,r);trans=float(w@fd);quad=[]
    for h in [2e-5,1e-5,5e-6]:quad.append(float(w@((s.jacobian(r+h*v)-s.jacobian(r-h*v))@v)/(2*h)))
    energy=s.sizes*abs(v)**2*s.E;part=[float(energy[s.geo['group_region']==k].sum()/energy.sum()) for k in range(3)]
    q=np.array(quad);second=float(np.sort(abs(ev))[1])
    ok=abs(ev[sel])<1e-7 and second>1e-5 and abs(trans)>1e-8 and abs(q.mean())>1e-6 and np.ptp(q)<.02*abs(q.mean())
    return dict(D=D,r=r,v=v,w=w,global_E_hz=s.global_rate(r),regional_hz=s.regional_rates(r),zero_eigenvalue=[float(ev[sel].real),float(ev[sel].imag)],
        next_eigenvalue_distance=second,transversality=trans,quadratic=quad,mode_energy_A_B_surround=part,
        static_type='SN_static_conditions_met' if ok else 'TURNING_POINT_NOT_VERIFIED',tangent_D=float(tm[-1]))

def main(a):
    s=SpatialRateV3(grid=a.grid,quiet=True);dest=DEST/'equilibria'/a.label;dest.mkdir(parents=True,exist_ok=True)
    if a.resume:
        z=np.load(a.resume);r=z['r'];D=float(z['D']);t=z['tangent']
        if a.reverse_tangent:t=-t
    else:
        initial=None
        if a.direction==-1:
            initial=.85/s.ref
            for d in np.linspace(1,a.start,40):
                s.set_D(float(d));initial,ok,tr=s.solve(initial)
                if not ok:raise RuntimeError(('descent',d,tr[-1]))
        s.set_D(a.start);r,ok,tr=s.solve(initial);assert ok,tr[-1]
        D=a.start;t=tangent(s,r,D,direction=a.direction)
    ds=a.ds;rows=[];folds=[];status='RUNNING';quality=None
    for k in range(a.steps):
        ev,vec=static_eigs(s,r,D)
        row=dict(index=k,D=D,global_E_hz=s.global_rate(r),regional_hz=s.regional_rates(r),residual_hz=float(abs(s.residual(r)).max()*1000),
            static_eigenvalues=[[float(e.real),float(e.imag)] for e in ev],tangent_D=float(t[-1]),step=ds,quality=quality)
        np.savez_compressed(dest/f'point{k:04d}.npz',r=r,D=D,tangent=t,Z=s.Z,eigenvalues=ev)
        rows.append(row)
        if k and rows[-1]['tangent_D']*rows[-2]['tangent_D']<0:folds.append([k-1,k]);log('FOLD BRACKET',folds[-1])
        write(dest/'result.json',dict(status=status,rows=rows,fold_brackets=folds,scope='prescribed Z path, dynamic M equilibrium (m=0.5 r); static Jacobian only'))
        log(k,'D=%.6f'%D,'global %.3f Hz'%s.global_rate(r),'tD %.3f'%t[-1],'ds %.4f'%ds)
        if a.stop_rate_above is not None and s.global_rate(r)>=a.stop_rate_above:status='REQUESTED_RATE_REACHED';break
        if a.stop_rate_below is not None and s.global_rate(r)<=a.stop_rate_below:status='REQUESTED_RATE_REACHED';break
        x=np.r_[r/RS,D/DS]
        for attempt in range(26):
            nextx,ok,nit=correct(s,x+ds*t,t)
            if ok:
                nt=tangent(s,nextx[:-1]*RS,float(nextx[-1]*DS),t);cosine=float(nt@t);corr=float(np.linalg.norm(nextx-(x+ds*t))/ds)
                quality=dict(tangent_cosine=cosine,correction_fraction=corr,accepted_ds=ds)
                if cosine>=a.min_tangent_cos and corr<=a.max_correction_fraction:break
                ok=False
            ds*=.5
        if not ok:status='CONTINUATION_STOPPED_NONCONVERGENCE';break
        r=nextx[:-1]*RS;D=float(nextx[-1]*DS);t=nt
        if nit<=4 and cosine>.98 and corr<.1:ds=min(ds*1.15,a.max_ds)
        if nit>=10:ds=max(ds*.65,1e-5)
        if not 1e-7<D<1-1e-7:status='PHYSICAL_DOMAIN_REACHED';break
    else:status='REQUESTED_SEGMENT_COMPLETE'
    refined=[]
    for i,j in folds:
        L=dict(np.load(dest/f'point{i:04d}.npz'));R=dict(np.load(dest/f'point{j:04d}.npz'));L['D']=float(L['D']);R['D']=float(R['D'])
        q=refine_fold(s,L,R)
        if q is None:refined.append(dict(bracket=[i,j],status='REFINEMENT_FAILED'));continue
        np.savez_compressed(dest/f'fold_{len(refined)+1}.npz',r=q.pop('r'),D=q['D'],v=q.pop('v'),w=q.pop('w'),Z=s.Z);q['bracket']=[i,j];refined.append(q);log('FOLD',q)
    write(dest/'result.json',dict(status=status,rows=rows,fold_brackets=folds,folds=refined,scope='prescribed Z path, dynamic M equilibrium (m=0.5 r); static Jacobian only',model_identity=s.identity()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',type=float,default=.005);p.add_argument('--direction',type=int,choices=[-1,1],default=1)
    p.add_argument('--steps',type=int,default=400);p.add_argument('--ds',type=float,default=.015);p.add_argument('--max-ds',type=float,default=.08)
    p.add_argument('--min-tangent-cos',type=float,default=0.);p.add_argument('--max-correction-fraction',type=float,default=float('inf'))
    p.add_argument('--stop-rate-above',type=float);p.add_argument('--stop-rate-below',type=float);p.add_argument('--reverse-tangent',action='store_true')
    p.add_argument('--resume');p.add_argument('--label',default='lower');p.add_argument('--grid',type=int,default=20);main(p.parse_args())
