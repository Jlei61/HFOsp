"""Differentiate the fixed-period BVP twice along its accepted branch tangent.

The curvature is checked by halving the directional finite-difference step.
It concerns the local Galerkin branch; it does not supply transverse stability.
"""
from native_path import *
from compact_periodic import CompactExactGalerkin,RS
import argparse


def curvature(o,sol,v,restart=60,steps=(1e-6,5e-7),eliminate_period=False):
    from cupyx.scipy.sparse.linalg import gmres,LinearOperator
    if getattr(o,'host_krylov',False):
        from host_krylov import host_gmres as gmres
    cp=o.cp;r=cp.asarray(sol['r']);n=r.size;T=float(sol['T']);D=float(sol['D'])
    y=cp.r_[(r/RS).ravel(),np.log(T),D*1000]
    phase=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(r,axis=0),n=o.N,axis=0)
    c=cp.zeros(n+2);c[-2]=1000.;arc=(y.copy(),c,cp.ones(n+2))
    f,A,_=o.evaluate(y,r,phase,D,arc=arc,derivative=True)
    tangent=cp.asarray(v);target=cp.zeros(n+2);target[-1]=1000
    tangent_error=float(cp.linalg.norm(A@tangent-target)/1000)
    assert tangent_error<1e-6,tangent_error
    rows=[]
    assert len(steps)==2 and steps[0]>steps[1]>0
    for h in steps:
        fp=o.evaluate(y+h*tangent,r,phase,D,arc=arc)
        fm=o.evaluate(y-h*tangent,r,phase,D,arc=arc)
        second=(fp+fm-2*f)/(h*h);scale=cp.linalg.norm(second)
        rhs=-second/scale
        kwargs=dict(tol=1e-9,atol=1e-12,restart=restart,maxiter=1500,
                    callback=lambda q:log('CURVATURE LINEAR',h,float(q)),callback_type='pr_norm')
        if eliminate_period:
            # log(T) is the prescribed coordinate, so its second derivative
            # is exactly zero. Keep the original rate and phase equations;
            # verify the reconstructed solution against the full Jacobian.
            assert abs(float(rhs[-1])) < 1e-8, float(rhs[-1])
            def embed(x):return cp.r_[x[:n],0.,x[-1]]
            B=LinearOperator((n+1,n+1),matvec=lambda x:(A@embed(x))[:-1],dtype=np.float64)
            diagonal=cp.ones(n+1);diagonal[-1]=1/max(o.fixed_D_column_norm,1.)
            P=LinearOperator((n+1,n+1),matvec=lambda x:x*diagonal,dtype=np.float64)
            reduced,info=gmres(B,rhs[:-1],M=P,**kwargs)
            w=embed(reduced)
        else:
            w,info=gmres(A,rhs,M=o.preconditioner(len(y)),**kwargs)
        err=float(cp.linalg.norm(A@w-rhs));assert err<1e-6,err
        w*=scale
        D_l=float(tangent[-1])*.001;D_ll=float(w[-1])*.001
        curv=(D_ll-D_l)/(T*T)
        # A quadratic predictor must reduce the symmetric Taylor remainder.
        qp=o.evaluate(y+h*tangent+.5*h*h*w,r,phase,D,arc=arc)
        qm=o.evaluate(y-h*tangent+.5*h*h*w,r,phase,D,arc=arc)
        original=float(cp.linalg.norm(fp+fm-2*f))
        corrected=float(cp.linalg.norm(qp+qm-2*f))
        q=dict(h_logT=h,d2D_dT2=curv,linear_relative_residual=err,
               symmetric_remainder_before=original,symmetric_remainder_after=corrected)
        rows.append(q);log('LOCAL CURVATURE',q)
    rel=abs(rows[-1]['d2D_dT2']-rows[0]['d2D_dT2'])/max(abs(rows[-1]['d2D_dT2']),1e-30)
    passed=rel<.05 and all(q['d2D_dT2']<0 and q['symmetric_remainder_after']<q['symmetric_remainder_before'] for q in rows)
    return dict(status='LOCAL_CURVATURE_CONVERGED' if passed else 'LOCAL_CURVATURE_NEEDS_REFINEMENT',
                D=D,T_ms=T,N=o.N,M=o.M,tangent_relative_residual=tangent_error,
                relative_step_difference=rel,rows=rows,d2D_dT2=rows[-1]['d2D_dT2'],
                directional_steps_logT=list(steps),quadrature_time_step_ms=T/o.M,
                exact_log_period_elimination=eliminate_period)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--M',type=int,default=16384)
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--steps',nargs=2,type=float,default=[1e-6,5e-7])
    p.add_argument('--restart',type=int,default=60)
    p.add_argument('--host-krylov',action='store_true')
    p.add_argument('--family',choices=['rate','native'],default='rate')
    p.add_argument('--large-quadrature',action='store_true')
    p.add_argument('--gain-cache-gb',type=float,default=3.)
    p.add_argument('--eliminate-period',action='store_true')
    p.add_argument('--label',default='local_curvature');a=p.parse_args()
    z=np.load(a.orbit);s=model()
    (attach_rate_entry_path if a.family=='rate' else attach_native_path)(s)
    assert float(z['residual'])<2e-8
    cls=CompactExactGalerkin
    if a.large_quadrature:
        from large_quadrature_periodic import LargeQuadratureGalerkin
        assert read(OUT/'large_quadrature_periodic_check.json')['status']=='PASS'
        cls=LargeQuadratureGalerkin
    cls.harmonic_block=129 if a.large_quadrature else 33
    o=cls(s,len(z['r']),a.M,a.device)
    o.gain_cache_gb=a.gain_cache_gb
    o.cache_mean_operators=False;o.cp.fft.config.get_plan_cache().set_size(0)
    o.host_krylov=a.host_krylov
    if a.host_krylov:assert read(OUT/'host_krylov_check.json')['status']=='PASS'
    q=curvature(o,dict(r=z['r'],T=float(z['T']),D=float(z['D'])),z['tangent'],
                restart=a.restart,steps=a.steps,eliminate_period=a.eliminate_period)
    write(Path(a.orbit).with_name(Path(a.orbit).stem+'_'+a.label+'.json'),q)
