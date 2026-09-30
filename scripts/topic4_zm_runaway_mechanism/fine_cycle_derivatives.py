"""Recompute tangent and directional curvature on an accepted fine-mesh orbit.

No coarse tangent is inherited. The original curvature agreement criterion is
unchanged; quadrature and directional step sizes are recorded separately.
"""
from native_path import *
from compact_periodic import CompactExactGalerkin, RS
from cycle_local_curvature import curvature
import argparse


def main(a):
    s=model();attach_rate_entry_path(s);z=np.load(a.orbit)
    assert float(z['residual'])<2e-8
    CompactExactGalerkin.harmonic_block=33
    o=CompactExactGalerkin(s,len(z['r']),a.M,a.device)
    o.cache_mean_operators=False;o.cp.fft.config.get_plan_cache().set_size(0)
    o.host_krylov=a.host_krylov
    from cupyx.scipy.sparse.linalg import gmres
    if a.host_krylov:
        from host_krylov import host_gmres as gmres
        assert read(OUT/'host_krylov_check.json')['status']=='PASS'
    cp=o.cp;r=cp.asarray(z['r']);n=r.size;T=float(z['T']);D=float(z['D'])
    y=cp.r_[(r/RS).ravel(),np.log(T),D*1000]
    phase=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(r,axis=0),n=o.N,axis=0)
    c=cp.zeros(n+2);c[-2]=1000.;arc=(y.copy(),c,cp.ones(n+2))
    f,A,_=o.evaluate(y,r,phase,D,arc=arc,derivative=True)
    residual=float(cp.max(cp.abs(f)))
    assert residual<2e-8,('Orbit must be corrected on this nonlinear grid first',residual)
    rhs=cp.zeros(n+2);rhs[-1]=1.
    tangent,info=gmres(A,rhs,tol=1e-9,atol=1e-12,restart=a.restart,maxiter=1500,
                      M=o.preconditioner(len(y)),
                      callback=lambda q:log('FINE TANGENT LINEAR',float(q)),callback_type='pr_norm')
    error=float(cp.linalg.norm(A@tangent-rhs));assert error<1e-6,error
    v=tangent.get()*1000.
    assert abs(v[-2]-1)<1e-6
    dest=OUT/'periodic'/a.label;dest.mkdir(parents=True,exist_ok=True)
    s.set_D(D)
    np.savez_compressed(dest/'center_with_tangent.npz',r=z['r'],T=T,D=D,Z=s.Z,
                        tangent=v,residual=residual)
    q=dict(status='FINE_TANGENT_COMPLETE',source=a.orbit,N=o.N,M=o.M,D=D,T_ms=T,
           dD_dT=float(v[-1])*.001/T,linear_residual=error,bvp_residual=residual)
    write(dest/'tangent.json',q);log('FINE TANGENT',q)
    del A,tangent,f,rhs;cp.get_default_memory_pool().free_all_blocks()
    result=curvature(o,dict(r=z['r'],T=T,D=D),v,restart=a.restart,steps=a.steps)
    result['source']=a.orbit
    result['parameterization']='same frozen spatial Z slice; M dynamic'
    write(dest/'curvature.json',result);log('FINE CURVATURE',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--label',required=True)
    p.add_argument('--M',type=int,default=65536);p.add_argument('--device',type=int,default=0)
    p.add_argument('--steps',nargs=2,type=float,default=[1e-4,5e-5])
    p.add_argument('--restart',type=int,default=40);p.add_argument('--host-krylov',action='store_true')
    main(p.parse_args())
