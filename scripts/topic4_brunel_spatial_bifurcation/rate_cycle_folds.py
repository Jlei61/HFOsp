"""Refine a periodic-orbit parameter turning point with a bordered BVP.

Use period as the local coordinate and solve dJ/d(log T)=0. Save the
non-phase null tangent and a second-derivative nondegeneracy check.
"""
from rate_periodic import *
from scipy.optimize import brentq


def refine_fold(first,second,label,N=64,device=0,low_memory=False,radius=None,krylov_restart=None,linear_normalize=False,
                host_krylov=False,stream_harmonics=False,tol=2e-8,tangent_predictor=False):
    from cupyx.scipy.sparse.linalg import gmres
    s=RateField();o=Periodic(s,N,device);cp=o.cp;za=np.load(first);zb=np.load(second);cache=[]
    o.low_memory=low_memory
    o.harmonic_chunk_size=64;o.derivative_chunk_size=64
    o.normalize_linear_rhs=linear_normalize
    o.host_krylov=host_krylov
    o.stream_harmonics=stream_harmonics
    o.linear_target_aware=tol<1e-9
    if krylov_restart:o.krylov_restart=krylov_restart
    warm=[]
    if tangent_predictor:
        stem=Path(first).stem.rsplit('_N',1)[0]
        original_N=len(za['r'])
        source=PERIODIC_OUT/f'{stem}_N{original_N}.json'
        vector=PERIODIC_OUT/f'{stem}_tangent_N{original_N}.npz'
        if source.exists() and vector.exists():
            old=read(source)
            assert old.get('coordinate','logT')=='logT'
            assert Path(old['orbit']).resolve()==Path(first).resolve()
            v=np.load(vector)['tangent']
            initial=np.r_[resample(v[:-2].reshape(original_N,s.P),N,axis=0).ravel(),v[-2:]]
            assert abs(initial[-2]-1)<1e-5
            warm.append((float(za['T']),initial))
    for z in [za,zb]:cache.append((float(z['T']),resample(z['r'],N,axis=0),float(z['J'])))
    if radius:
        T,r,J=cache[0];cache=[(T-radius,r.copy(),J),(T+radius,r.copy(),J)]
        if warm:
            tangent=warm[0][1];dr=tangent[:-2].reshape(N,s.P)/1000
            cache=[(t,r+np.log(t/T)*dr,J+np.log(t/T)*tangent[-1]/1000)
                   for t in [T-radius,T+radius]]
    corrected=[]
    def solve(T):
        # Reuse only a converged state once one exists.  The initial radius
        # guesses otherwise win equal-distance searches even after correction.
        if corrected:
            seed_T,r,J,result=min(corrected,key=lambda q:abs(q[0]-T))
            if abs(seed_T-T)<1e-12:
                return result
            r=r.copy()
            if tangent_predictor:
                step=np.log(T/seed_T);v=result[3]
                r+=step*v[:-2].reshape(N,s.P)/1000
                J+=step*v[-1]/1000
        else:
            _,r,J=min(cache,key=lambda q:abs(q[0]-T))
        pred=np.r_[(r*1000).ravel(),np.log(T),J*1000]
        tan=np.zeros_like(pred);tan[-2]=1;weight=np.ones_like(pred)
        r,T,J,err,history=o.solve(r,T,J,arc=(pred,tan,weight),tol=tol,maxiter=24)
        assert err<2e-8 and err<=tol*1.01
        y=cp.asarray(np.r_[(r*1000).ravel(),np.log(T),J*1000]);ref=cp.asarray(r)
        dr=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(ref,axis=0),n=N,axis=0);phase=dr/cp.sum(dr*dr)*.001
        F,A,data=o.evaluate(y,ref,phase,J,derivative=True,arc=tuple(cp.asarray(x) for x in (pred,tan,weight)))
        rhs=cp.zeros_like(y);rhs[-1]=1
        initial=None
        if warm:
            candidate=cp.asarray(min(warm,key=lambda item:abs(item[0]-T))[1])
            phase_direction=cp.r_[(dr*1000).ravel(),0.,0.]
            response=A@phase_direction
            if abs(float(response[-2]))>1e-12:
                candidate-=phase_direction*((A@candidate-rhs)[-2]/response[-2])
            if float(cp.linalg.norm(A@candidate-rhs))<1:initial=candidate
        restart=krylov_restart or (160 if N>=512 else 100)
        if host_krylov:
            from scipy.sparse.linalg import LinearOperator as HostOperator,gmres as host_gmres
            from threadpoolctl import threadpool_limits
            host_A=HostOperator(A.shape,matvec=lambda v:(A@cp.asarray(v)).get(),dtype=float)
            with threadpool_limits(limits=4,user_api='blas'):
                step,info=host_gmres(host_A,rhs.get(),x0=initial.get() if initial is not None else None,
                    rtol=1e-9,atol=1e-12,restart=restart,maxiter=18)
            dy=cp.asarray(step)
            del host_A,step
        else:
            dy,info=gmres(A,rhs,x0=initial,tol=1e-9,atol=1e-12,restart=restart,maxiter=1800 if N>=512 else 700)
        linear_error=float(cp.linalg.norm(A@dy-rhs));print('FOLD TANGENT RESIDUAL',linear_error,info,flush=True)
        if linear_error>=1e-6:
            from scipy.sparse.linalg import LinearOperator as HostOperator,gmres as host_gmres
            host_A=HostOperator(A.shape,matvec=lambda v:(A@cp.asarray(v)).get(),dtype=float)
            step,info=host_gmres(host_A,rhs.get(),x0=dy.get(),rtol=1e-10,atol=1e-12,restart=240,maxiter=15)
            dy=cp.asarray(step);linear_error=float(cp.linalg.norm(A@dy-rhs))
            print('PERIOD TANGENT HOST RESIDUAL',linear_error,info,flush=True)
        assert linear_error<1e-6
        if tangent_predictor:
            warm.append((T,dy.get()))
            if len(warm)>2:warm.pop(0)
        dJ=float(dy[-1]/1000);print('FOLD DERIV',label,T,J,dJ,flush=True)
        result=(dJ,r,J,dy.get(),float(cp.max(cp.abs(F))),history)
        corrected.append((T,r,J,result))
        return result
    Ta,Tb=sorted([cache[0][0],cache[1][0]]);T=brentq(lambda T:solve(T)[0],Ta,Tb,xtol=2e-7,rtol=1e-11)
    dJ,r,J,tangent,err,history=solve(T);h=.01
    d1=solve(T-h)[0];d2=solve(T+h)[0];curvature=(d2-d1)/(2*h)*T
    assert abs(dJ)<1e-7, f'Parameter derivative did not vanish: {dJ}; no fold certified'
    assert abs(curvature)>1e-8, f'Degenerate or unresolved curvature: {curvature}'
    path=save_orbit(s,r,T,J,err,history,f'{label}_N{N}')
    save_periodic_array(PERIODIC_OUT/f'{label}_tangent_N{N}.npz',tangent=tangent)
    row=dict(label=label,J_EE_core=J,T_ms=T,N=N,orbit=str(path),dJ_dlogT=dJ,d2J_dlogT2=curvature,
        residual_hz=err,type='fold of periodic orbits',check='phase-fixed BVP has a non-phase null tangent; parameter curvature nonzero',
        floquet_status='check matching monodromy files separately')
    write(PERIODIC_OUT/f'{label}_N{N}.json',row);print('CYCLE FOLD',row,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second');p.add_argument('--label',required=True);p.add_argument('--N',type=int,default=64);p.add_argument('--device',type=int,default=0)
    p.add_argument('--low-memory',action='store_true');p.add_argument('--radius',type=float,help='Period bracket radius in ms around the first orbit')
    p.add_argument('--krylov-restart',type=int,help='Bound Krylov storage; convergence tolerances remain unchanged')
    p.add_argument('--linear-normalize',action='store_true')
    p.add_argument('--host-krylov',action='store_true',help='Keep both Newton and tangent Arnoldi bases in host memory')
    p.add_argument('--stream-harmonics',action='store_true',help='Use the verified exact bounded harmonic actions')
    p.add_argument('--tol',type=float,default=2e-8,help='Periodic BVP tolerance in Hz')
    p.add_argument('--tangent-predictor',action='store_true',help='Use the independently stored log-period tangent as an initial guess')
    a=p.parse_args();refine_fold(a.first,a.second,a.label,a.N,a.device,a.low_memory,a.radius,a.krylov_restart,a.linear_normalize,
        a.host_krylov,a.stream_harmonics,a.tol,a.tangent_predictor)
