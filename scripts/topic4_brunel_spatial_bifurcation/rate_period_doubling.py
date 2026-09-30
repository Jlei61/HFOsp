"""Refine a cycle multiplier -1 using a bordered antiperiodic DDE BVP.

The scalar border is zero only when the antiperiodic variational operator
has a null vector. A converged orbit, null-vector residual, nonzero crossing
slope, temporal-mesh refinement and monodromy are separate checks.
"""
from rate_antiperiodic import *
from scipy.optimize import brentq


def main():
    from cupyx.scipy.sparse.linalg import LinearOperator as CL, gmres
    p=argparse.ArgumentParser()
    p.add_argument('first');p.add_argument('second');p.add_argument('--seed',required=True)
    p.add_argument('--label',default='PD_double_low');p.add_argument('--N',type=int,default=512)
    p.add_argument('--device',type=int,default=0);p.add_argument('--scan',type=float,nargs='+')
    p.add_argument('--cache-glob',help='Extra already-continued orbit seeds on this same branch')
    p.add_argument('--nearest-orbit-seed',action='store_true',
                   help='Seed each Newton solve from the nearest saved orbit, avoiding interpolation of arbitrary phases')
    p.add_argument('--low-memory',action='store_true')
    p.add_argument('--linear-normalize',action='store_true')
    p.add_argument('--host-krylov',action='store_true')
    p.add_argument('--stream-harmonics',action='store_true',help='Use checked exact frequency blocks for periodic and antiperiodic actions')
    p.add_argument('--gpu-antiperiodic',action='store_true',
                   help='Use GPU Krylov for the fixed antiperiodic border, independently of Newton Krylov placement')
    p.add_argument('--harmonic-chunk-size',type=int,default=64)
    p.add_argument('--cached-antiperiodic',action='store_true',help='Cache the fixed odd-harmonic matrices within each bordered solve; retain low-memory Newton derivatives')
    p.add_argument('--krylov-restart',type=int,default=180)
    p.add_argument('--min-half-mismatch',type=float,default=0.,help='Reject collapse onto a repeated parent orbit')
    p.add_argument('--seed-index',type=int,help='Explicit eigenmode index from the antiperiodic seed file')
    p.add_argument('--parameter-bracket',type=float,nargs=2,
                   help='Explicit same-branch J bracket for temporal mesh refinement')
    p.add_argument('--null-root-tol',type=float,default=0.,
                   help='Optional termination on both raw border and relative null residual; raw values remain recorded')
    p.add_argument('--slope-step',type=float,
                   help='Independent J increment for the crossing derivative, resolved above BVP stopping noise')
    a=p.parse_args();s=RateField();cache=[]
    assert 0<=a.null_root_tol<=1e-9
    if a.stream_harmonics:
        assert read(PERIODIC_OUT/'streamed_harmonic_operator_check.json')['status']=='PASS'
    for path in [a.first,a.second]:
        z=np.load(path);cache.append((float(z['J']),resample(z['r'],a.N,axis=0),float(z['T'])))
    bracket=sorted([cache[0][0],cache[1][0]])
    if a.parameter_bracket is not None:
        bracket=sorted(a.parameter_bracket)
    assert bracket[0]<bracket[1] or a.scan, 'A root search requires a nonzero J bracket'
    if a.cache_glob:
        for path in sorted((PERIODIC_OUT/'orbits').glob(a.cache_glob)):
            z=np.load(path)
            if float(z['residual'])<2e-8:
                cache.append((float(z['J']),resample(z['r'],a.N,axis=0),float(z['T'])))
    z=np.load(a.seed)
    if 'u' in z:
        q=z['u'].real
    else:
        vals=z['eigenvalues'];seed_index=int(np.argmin(abs(vals))) if a.seed_index is None else a.seed_index
        q=z['vectors'][:,seed_index].real
    q=q.reshape(-1,s.P)
    # Resample the full antiperiodic extension, not the discontinuous half.
    q=resample(np.r_[q,-q],2*a.N,axis=0)[:a.N].ravel();q/=np.linalg.norm(q)
    rows=[];last=None
    def evaluate(J):
        nonlocal last
        _,r,T=min(reversed(cache),key=lambda x:abs(x[0]-J))
        lower=[v for v in cache if v[0]<=J];upper=[v for v in cache if v[0]>=J]
        if lower and upper and not a.nearest_orbit_seed:
            lo=max(lower,key=lambda v:v[0]);hi=min(upper,key=lambda v:v[0])
            if hi[0]-lo[0]>1e-12:
                frac=(J-lo[0])/(hi[0]-lo[0]);r=(1-frac)*lo[1]+frac*hi[1];T=(1-frac)*lo[2]+frac*hi[2]
        o=Periodic(s,a.N,a.device)
        o.low_memory=a.low_memory;o.normalize_linear_rhs=a.linear_normalize
        o.host_krylov=a.host_krylov;o.krylov_restart=a.krylov_restart
        o.stream_harmonics=a.stream_harmonics;o.linear_target_aware=a.stream_harmonics
        o.harmonic_chunk_size=a.harmonic_chunk_size;o.derivative_chunk_size=a.harmonic_chunk_size
        r,T,J,err,hist=o.solve(r,T,J,maxiter=24,tol=2e-11 if a.stream_harmonics else 1e-9);assert err<2e-8
        half_mismatch=float(np.linalg.norm(r-np.roll(r,len(r)//2,axis=0))/np.linalg.norm(r-r.mean(axis=0)))
        assert half_mismatch>a.min_half_mismatch,('Repeated-parent orbit',half_mismatch)
        path=save_orbit(s,r,T,J,err,hist,f'{a.label}_eval_J{J:.14f}_N{a.N}')
        cp=o.cp;o.cache=None;del o;gc.collect()
        cp.get_default_memory_pool().free_all_blocks();cache.append((J,r,T))
        anti=Antiperiodic(s,path,a.N,a.device,low_memory=a.low_memory and not a.cached_antiperiodic,
                         harmonic_chunk_size=a.harmonic_chunk_size,stream_harmonics=a.stream_harmonics);qq=cp.asarray(q);dim=len(q)
        def mv(x):return cp.r_[anti.apply(x[:-1])+qq*x[-1],cp.vdot(qq,x[:-1]).real]
        op=CL((dim+1,dim+1),matvec=mv,dtype=float);rhs=cp.zeros(dim+1);rhs[-1]=1
        if a.host_krylov and not a.gpu_antiperiodic:
            from scipy.sparse.linalg import LinearOperator as HostOperator,gmres as host_gmres
            host_A=HostOperator(op.shape,matvec=lambda v:(op@cp.asarray(v)).get(),dtype=float)
            calls=[0]
            def progress(value):
                calls[0]+=1
                if calls[0]%20==0:print('ANTIPERIODIC HOST',calls[0],float(value),flush=True)
            yy,info=host_gmres(host_A,rhs.get(),rtol=2e-10,atol=1e-12,restart=a.krylov_restart,maxiter=20,
                callback=progress,callback_type='pr_norm')
            y=cp.asarray(yy)
        else:
            calls=[0];started=time.time()
            def progress(value):
                calls[0]+=1
                print('ANTIPERIODIC GPU RESTART',calls[0],float(value),'seconds',time.time()-started,flush=True)
            y,info=gmres(op,rhs,tol=2e-10,atol=1e-12,restart=a.krylov_restart,maxiter=3600,
                         callback=progress,callback_type='pr_norm')
        linerr=float(cp.linalg.norm(op@y-rhs));assert linerr<1e-7,(info,linerr)
        eta=float(y[-1]);u=y[:-1];defect=float(cp.linalg.norm(anti.apply(u))/cp.linalg.norm(u))
        row=dict(J_EE_core=J,T_ms=T,border_scalar=eta,linear_residual=linerr,
                 half_period_relative_mismatch=half_mismatch,
                 cached_antiperiodic=bool(a.cached_antiperiodic),
                 streamed_harmonic_actions=a.stream_harmonics,
                 antiperiodic_krylov='host' if a.host_krylov and not a.gpu_antiperiodic else 'gpu',
                 antiperiodic_relative_residual=defect,orbit=str(path),orbit_residual_hz=err)
        rows.append(row);write(PERIODIC_OUT/f'{a.label}_scan_N{a.N}.json',rows)
        last=(row,u.get().reshape(a.N,s.P));print('PD BORDER',row,flush=True)
        del op,anti;gc.collect();cp.get_default_memory_pool().free_all_blocks();return eta
    if a.scan:
        for J in a.scan:evaluate(J)
        return
    ja,jb=bracket
    # Sharp multiplier crossings can amplify a 1e-10 parameter error enough
    # to fail the independent null-mode residual. Refine the root itself;
    # do not loosen the residual acceptance threshold.
    def objective(J):
        eta=evaluate(J)
        if (a.null_root_tol and abs(eta)<=a.null_root_tol and
            last[0]['antiperiodic_relative_residual']<=a.null_root_tol):
            return 0.
        return eta
    J=brentq(objective,ja,jb,xtol=2e-13,rtol=1e-14)
    h=a.slope_step if a.slope_step is not None else min(2e-6,(jb-ja)*.001)
    assert h>0
    slope=(evaluate(J+h)-evaluate(J-h))/(2*h)
    evaluate(J);row,u=last
    assert row['antiperiodic_relative_residual']<1e-7,row
    assert abs(slope)>1e-6,slope
    row.update(label=a.label,type='period-doubling candidate verified by antiperiodic null mode',
               N=a.N,dborder_dJ=slope,multiplier=-1.,criticality='NOT_COMPUTED',
               root_stop_null_tolerance=a.null_root_tol,
               slope_step_J=h,
               validation='Requires mesh agreement and independent monodromy check before promotion')
    write(PERIODIC_OUT/f'{a.label}_N{a.N}.json',row)
    save_periodic_array(PERIODIC_OUT/f'{a.label}_mode_N{a.N}.npz',u=u,J=J,T=row['T_ms'])
    print('PD ROOT',row,flush=True)


if __name__=='__main__':main()
