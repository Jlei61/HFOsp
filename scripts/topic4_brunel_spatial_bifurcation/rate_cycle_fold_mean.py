"""Periodic fold refinement using a core mean as the local branch coordinate.

This remains regular when period and J both turn together (weakly coupled
core cycles). It uses the same phase-fixed bordered periodic BVP.
"""
from rate_periodic import *
from scipy.optimize import brentq


def main():
    from cupyx.scipy.sparse.linalg import gmres
    p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second');p.add_argument('--label',required=True)
    p.add_argument('--N',type=int,default=256);p.add_argument('--core',choices=['A','B'],default='B');p.add_argument('--device',type=int,default=0)
    p.add_argument('--radius',type=float,help='Local mean-coordinate bracket about an already refined first orbit')
    p.add_argument('--low-memory',action='store_true',help='Exact on-demand parameter derivatives instead of stored harmonic banks')
    p.add_argument('--linear-normalize',action='store_true',help='Normalize Newton linear RHS without changing the equations')
    p.add_argument('--host-krylov',action='store_true',help='Keep the large Newton Arnoldi basis in host memory')
    p.add_argument('--cached-host-krylov',action='store_true',help='Keep harmonic operators on GPU and both Newton/tangent Arnoldi bases on host; no automatic streaming')
    p.add_argument('--linear-rtol-cap',type=float,default=.02)
    p.add_argument('--tol',type=float,default=1e-9,help='Algebraic BVP tolerance in Hz; tighter solves can be required by filter positivity')
    p.add_argument('--tangent-predictor',action='store_true',help='Use the saved same-coordinate tangent to predict the initial bracket')
    p.add_argument('--tangent-warm-start',action='store_true',help='Seed tangent solves with a residual-screened prior tangent; linear acceptance is unchanged')
    a=p.parse_args();s=RateField();o=Periodic(s,a.N,a.device);cp=o.cp;core='AB'.index(a.core)
    o.low_memory=a.low_memory
    o.harmonic_chunk_size=64;o.derivative_chunk_size=64
    o.normalize_linear_rhs=a.linear_normalize
    o.linear_target_aware=a.tol<1e-9
    assert 0<a.linear_rtol_cap<=.02;o.linear_rtol_cap=a.linear_rtol_cap
    o.host_krylov=a.host_krylov or a.cached_host_krylov
    if a.cached_host_krylov:
        free=int(cp.cuda.runtime.memGetInfo()[0])
        bank=(a.N//2+1)*sum(len(v[0]) for v in s.raw)*16
        if free<=bank+int(2.5*1024**3):
            assert read(PERIODIC_OUT/'streamed_harmonic_operator_check.json')['status']=='PASS'
            o.stream_harmonics=True
            print('EXACT STREAMED ACTIONS / HOST ARNOLDI',dict(free_bytes=free,bank_bytes=bank),flush=True)
        else:
            print('CACHED GPU ACTIONS / HOST ARNOLDI',dict(free_bytes=free,bank_bytes=bank),flush=True)
    elif a.host_krylov:
        # Host Arnoldi was needed before frequency-block construction. With
        # that temporary-memory peak removed, use the identical GPU solver
        # when its bank, basis and work arrays fit with an explicit reserve.
        free=int(cp.cuda.runtime.memGetInfo()[0])
        bank=(a.N//2+1)*sum(len(v[0]) for v in s.raw)*16
        basis=(a.N*s.P+2)*8*161
        estimate=bank+basis+int(3.5*1024**3)
        if free>estimate:o.host_krylov=False
        elif free>basis+int(3.5*1024**3):
            verification=PERIODIC_OUT/'streamed_harmonic_operator_check.json'
            if verification.exists() and read(verification)['status']=='PASS':
                o.stream_harmonics=True;o.host_krylov=False
        print('NEWTON KRYLOV PLACEMENT',dict(host=o.host_krylov,free_bytes=free,
            streamed_harmonics=getattr(o,'stream_harmonics',False),
            required_free_for_cached_GPU=estimate,equations_changed=False),flush=True)
    mask=s.E&(s.geo['group_region']==core);ww=s.geo['group_size']*mask;ww=ww/ww.sum();c=np.r_[np.tile(ww/a.N,a.N),0.,0.]
    warm_check=PERIODIC_OUT/'warm_tangent_fold_solver_check.json'
    # The residual-screened initial guess is equally valid for GPU and host
    # GMRES; neither the bordered operator nor its acceptance test changes.
    warm_enabled=a.tangent_warm_start or (warm_check.exists() and read(warm_check)['status']=='PASS')
    warm_tangents=[]
    if warm_enabled:
        # A finer temporal mesh represents the same full-space tangent.
        # Screen this initial guess with the current bordered residual below,
        # exactly as for a tangent from a neighbouring branch coordinate.
        # This does not reuse an old derivative as a new numerical result.
        saved=[read(f) for f in PERIODIC_OUT.glob(a.label+'_N*.json')]
        saved=[q for q in saved if q['N']<=a.N and
               q.get('coordinate')==f'core_{a.core}_mean_Hz' and
               (PERIODIC_OUT/f'{a.label}_tangent_N{q["N"]}.npz').exists()]
        if saved:
            seed=max(saved,key=lambda q:q['N'])
            raw=np.load(PERIODIC_OUT/f'{a.label}_tangent_N{seed["N"]}.npz')['tangent']
            tr=resample(raw[:-2].reshape(seed['N'],s.P),a.N,axis=0)
            candidate=np.r_[tr.ravel(),raw[-2:]]
            assert abs(c@candidate-1)<1e-5
            warm_tangents.append((seed['coordinate_value'],candidate))
            print('COARSE-MESH TANGENT INITIAL GUESS',seed['N'],a.N,flush=True)
    cache=[]
    for path in [a.first,a.second]:
        z=np.load(path);r=resample(z['r'],a.N,axis=0);coord=float(r.mean(0)@ww*1000);cache.append((coord,r,float(z['T']),float(z['J'])))
    if a.radius:
        coord,r,T,J=cache[0];oldN=len(np.load(a.first)['r'])
        tangent_source=PERIODIC_OUT/f'{a.label}_tangent_N{oldN}.npz'
        root_source=PERIODIC_OUT/f'{a.label}_N{oldN}.json'
        if a.tangent_predictor and tangent_source.exists() and root_source.exists() and read(root_source).get('coordinate')==f'core_{a.core}_mean_Hz':
            tangent=np.load(tangent_source)['tangent']
            tr=resample(tangent[:-2].reshape(oldN,s.P),a.N,axis=0)/1000
            assert abs(float(tr.mean(0)@ww*1000)-1)<1e-5
            cache=[(coord+d,r+d*tr,T*np.exp(d*tangent[-2]),J+d*tangent[-1]/1000)
                   for d in [-a.radius,a.radius]]
            print('FOLD TANGENT PREDICTOR',str(tangent_source),flush=True)
        else:
            cache=[(coord+d,r+mask[None,:]*d/1000,T,J) for d in [-a.radius,a.radius]]
    solved={}
    def solve(coord):
        if coord in solved:return solved[coord]
        _,r,T,J=min(reversed(cache),key=lambda q:abs(q[0]-coord))
        lower=[q for q in cache if q[0]<=coord];upper=[q for q in cache if q[0]>=coord]
        if lower and upper:
            left=max(lower,key=lambda q:q[0]);right=min(upper,key=lambda q:q[0])
            if right[0]>left[0]+1e-10:
                t=(coord-left[0])/(right[0]-left[0]);r=left[1]*(1-t)+right[1]*t;T=left[2]*(1-t)+right[2]*t;J=left[3]*(1-t)+right[3]*t
        pred=np.r_[(r*1000).ravel(),np.log(T),J*1000]
        pred+=c*(coord-c@pred)/(c@c);r=pred[:-2].reshape(a.N,s.P)/1000
        arc=(pred,c,np.ones_like(c));r,T,J,err,history=o.solve(r,T,J,arc=arc,maxiter=24,tol=a.tol)
        if err>=2e-8:
            save_orbit(s,r,T,J,err,history,f'diagnostic_{a.label}_mean{coord:.10f}_N{a.N}_attempt{time.time_ns()}')
        assert err<2e-8
        if a.tol<1e-9:assert err<=a.tol*1.01,('Requested fine BVP tolerance not met',err,a.tol)
        y=cp.asarray(np.r_[(r*1000).ravel(),np.log(T),J*1000]);ref=cp.asarray(r)
        dr=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(ref,axis=0),n=a.N,axis=0);phase=dr/cp.sum(dr*dr)*.001
        F,A,_=o.evaluate(y,ref,phase,J,derivative=True,arc=tuple(cp.asarray(x) for x in arc))
        rhs=cp.zeros_like(y);rhs[-1]=1
        initial_tangent=None
        if warm_enabled and warm_tangents:
            candidate=min(warm_tangents,key=lambda item:abs(item[0]-coord))[1]
            candidate=cp.asarray(candidate)
            # The phase condition changes with the base cycle. Remove its
            # current phase component, then compare actual bordered residuals.
            phase_direction=cp.r_[(dr*1000).ravel(),0.,0.]
            response=A@phase_direction
            if abs(float(response[-2]))>1e-12:
                candidate=candidate-phase_direction*((A@candidate-rhs)[-2]/response[-2])
            initial_error=float(cp.linalg.norm(A@candidate-rhs))
            if initial_error<1:
                initial_tangent=candidate
                print('TANGENT WARM INITIAL RESIDUAL',initial_error,flush=True)
        if a.cached_host_krylov:
            from scipy.sparse.linalg import LinearOperator as HostOperator,gmres as host_gmres
            from threadpoolctl import threadpool_limits
            host_A=HostOperator(A.shape,matvec=lambda v:(A@cp.asarray(v)).get(),dtype=float)
            with threadpool_limits(limits=4,user_api='blas'):
                step,info=host_gmres(host_A,rhs.get(),x0=initial_tangent.get() if initial_tangent is not None else None,rtol=1e-9,atol=1e-12,
                    restart=160 if a.N>=512 else 100,maxiter=18)
            dy=cp.asarray(step)
            del host_A,step
        else:
            dy,info=gmres(A,rhs,x0=initial_tangent,tol=1e-9,atol=1e-12,restart=160 if a.N>=512 else 100,maxiter=1800 if a.N>=512 else 700)
        error=float(cp.linalg.norm(A@dy-rhs))
        if error>=1e-6:
            # A longer host Arnoldi basis avoids rejecting a well-defined
            # tangent merely because the short GPU restart stagnates.
            # Preserve the same Jacobian action and residual threshold.
            from scipy.sparse.linalg import LinearOperator as HostOperator,gmres as host_gmres
            from threadpoolctl import threadpool_limits
            print('TANGENT HOST RETRY',info,error,flush=True)
            host_A=HostOperator(A.shape,matvec=lambda v:(A@cp.asarray(v)).get(),dtype=float)
            with threadpool_limits(limits=4,user_api='blas'):
                step,info=host_gmres(host_A,rhs.get(),x0=dy.get(),rtol=1e-10,atol=1e-12,
                    restart=240,maxiter=15)
            dy=cp.asarray(step);error=float(cp.linalg.norm(A@dy-rhs))
            print('TANGENT HOST RESIDUAL',info,error,flush=True)
        assert error<1e-6,error
        if warm_enabled:
            warm_tangents.append((coord,dy.get()))
            if len(warm_tangents)>2:warm_tangents.pop(0)
        dJ=float(dy[-1]/1000)
        # A corrected endpoint replaces its initial guess. Brent's repeated
        # evaluation of an identical coordinate can reuse the checked tangent.
        cache[:]=[entry for entry in cache if entry[0]!=coord]
        cache.append((coord,r,T,J));print('MEAN FOLD',coord,J,T,dJ,flush=True)
        result=(dJ,r,T,J,dy.get(),err,history);solved[coord]=result
        return result
    # The J error is quadratic in the mean-coordinate error at a fold. Avoid
    # chasing tangent-solver roundoff many orders below the mesh uncertainty.
    ca,cb=sorted([cache[0][0],cache[1][0]])
    # A tiny mean-coordinate bracket can have very large curvature. A fixed
    # 2e-7 tolerance then leaves a visibly nonzero parameter derivative even
    # though J itself is accurate. Refine the coordinate, not the acceptance.
    coordinate_tol=min(2e-7 if a.N>=512 else 2e-9,max(2e-12,(cb-ca)*1e-6))
    coord=brentq(lambda x:solve(x)[0],ca,cb,xtol=coordinate_tol,rtol=2e-13)
    dJ,r,T,J,tangent,err,history=solve(coord);h=min(.005,(cb-ca)*.002);curv=(solve(coord+h)[0]-solve(coord-h)[0])/(2*h)
    assert abs(dJ)<1e-7, f'Parameter derivative did not vanish: {dJ}; no fold certified'
    assert abs(curv)>1e-8, f'Degenerate or unresolved curvature: {curv}'
    path=save_orbit(s,r,T,J,err,history,f'{a.label}_N{a.N}');save_periodic_array(PERIODIC_OUT/f'{a.label}_tangent_N{a.N}.npz',tangent=tangent)
    row=dict(label=a.label,J_EE_core=J,T_ms=T,N=a.N,orbit=str(path),coordinate=f'core_{a.core}_mean_Hz',coordinate_value=coord,
        dJ_dcoordinate=dJ,d2J_dcoordinate2=curv,residual_hz=err,type='fold of periodic orbits',
        check='phase-fixed bordered BVP null tangent; nonzero parameter curvature')
    write(PERIODIC_OUT/f'{a.label}_N{a.N}.json',row);print('CYCLE FOLD',row,flush=True)


if __name__=='__main__':main()
