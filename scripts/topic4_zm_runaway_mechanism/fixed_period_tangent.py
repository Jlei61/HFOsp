"""Differentiate a converged periodic branch with period held as coordinate.

Eliminate the prescribed log-period direction exactly, as in the audited
fixed-period corrector. Scale only the D column of the linear solve.
"""
from native_path import *
from large_quadrature_periodic import LargeQuadratureGalerkin,RS
import argparse


def tangent_at(o,z,restart=80,linear_tol=1e-9,initial_tangent=None):
    from cupyx.scipy.sparse.linalg import LinearOperator
    from host_krylov import host_gmres
    cp=o.cp;s=o.s;r=cp.asarray(z['r']);n=r.size;T=float(z['T']);D=float(z['D'])
    y=cp.r_[(r/RS).ravel(),np.log(T),D*1000]
    phase=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(r,axis=0),n=o.N,axis=0)
    c=cp.zeros(n+2);c[-2]=1000.;arc=(y.copy(),c,cp.ones(n+2))
    f,A,_=o.evaluate(y,r,phase,D,arc=arc,derivative=True)
    residual=float(cp.max(cp.abs(f)));assert residual<2e-8,residual
    e=cp.zeros(n+2);e[-2]=1.;rhs=-(A@e)[:-1]
    def embed(w):return cp.r_[w[:n],0.,w[-1]]
    B=LinearOperator((n+1,n+1),matvec=lambda w:(A@embed(w))[:-1],dtype=np.float64)
    scale=cp.ones(n+1);scale[-1]=1/max(o.fixed_D_column_norm,1.)
    M=LinearOperator((n+1,n+1),matvec=lambda w:w*scale,dtype=np.float64)
    base=cp.zeros(n+1)
    rhs_norm=float(cp.linalg.norm(rhs))
    if initial_tangent is not None:
        iv=cp.asarray(initial_tangent)
        assert iv.shape==(n+2,) and abs(float(iv[-2])-1.)<1e-12
        base=cp.r_[iv[:n],iv[-1]]
    correction_rhs=rhs-B@base if initial_tangent is not None else rhs
    # Refine the saved linear solution by its independently recomputed defect.
    # The absolute correction tolerance is tied to the ORIGINAL right-hand
    # side, so a small defect is not needlessly solved to machine precision.
    atol=max(1e-12,rhs_norm*linear_tol*.1) if initial_tangent is not None else 1e-11
    change,info=host_gmres(B,correction_rhs,tol=linear_tol,atol=atol,restart=restart,maxiter=1500,M=M,
        callback=lambda q:log('FIXED PERIOD TANGENT LINEAR',float(q)),callback_type='pr_norm')
    w=base+change
    defect=B@w-rhs;relative=float(cp.linalg.norm(defect)/cp.linalg.norm(rhs))
    assert relative<1e-7,(info,relative)
    v=embed(w)+e
    full=A@v;full[-1]-=1000.
    q=dict(D=D,T_ms=T,dD_dT=float(v[-1])*.001/T,
        bvp_residual_hz=residual,linear_relative_residual=relative,
        augmented_relative_residual=float(cp.linalg.norm(full)/max(float(cp.linalg.norm(rhs)),1000.)),
        augmented_target_relative_residual=float(cp.linalg.norm(full)/1000.),
        original_rhs_norm=rhs_norm,requested_linear_tolerance=linear_tol,
        saved_tangent_defect_correction=initial_tangent is not None,
        N=o.N,M=o.M,parameterization='log period; complete normalized phase gauge; native spatial Z path',
        scope='Geometric branch derivative, not a Floquet stability certificate')
    return v.get(),q


def main(a):
    s=model();attach_native_path(s);z=np.load(a.orbit)
    assert float(z['residual'])<2e-8
    s.set_D(float(z['D']));assert np.max(abs(s.Z-z['Z']))<1e-12
    LargeQuadratureGalerkin.harmonic_block=129
    o=LargeQuadratureGalerkin(s,len(z['r']),a.M,a.device)
    o.cache_mean_operators=False;o.gain_cache_gb=a.gain_cache_gb
    o.cp.fft.config.get_plan_cache().set_size(8)
    initial=None
    guess_check=None
    if a.initial_tangent:
        old=np.load(a.initial_tangent)
        if a.nearby_tangent_guess:
            assert old['r'].shape==z['r'].shape and float(old['residual'])<2e-8
            dt=abs(float(old['T'])-float(z['T']));dd=abs(float(old['D'])-float(z['D']))
            dz=float(np.max(abs(old['Z']-z['Z'])))
            dr=float(np.linalg.norm(old['r']-z['r'])/np.linalg.norm(z['r']))
            assert dt<.002 and dd<1e-7 and dz<1e-5 and dr<.001,(dt,dd,dz,dr)
            guess_check=dict(period_difference_ms=dt,D_difference=dd,max_Z_difference=dz,
                             relative_waveform_difference=dr,
                             scope='Numerical initial guess only; target Jacobian and full tangent residual are independently evaluated.')
        else:
            assert np.array_equal(old['r'],z['r']) and np.array_equal(old['Z'],z['Z'])
            assert float(old['D'])==float(z['D']) and float(old['T'])==float(z['T'])
        initial=old['tangent']
    else:assert not a.nearby_tangent_guess
    v,q=tangent_at(o,z,a.restart,a.linear_tol,initial)
    if initial is not None:assert q['augmented_target_relative_residual']<1e-6,q
    dest=OUT/'periodic'/a.label;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'point_with_tangent.npz',r=z['r'],T=z['T'],D=z['D'],Z=z['Z'],
        residual=z['residual'],tangent=v)
    q.update(source=a.orbit,status='TANGENT_COMPLETE',initial_tangent_source=a.initial_tangent,
             nearby_initial_guess_check=guess_check)
    write(dest/'result.json',q)
    log('FIXED PERIOD TANGENT',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--label',required=True)
    p.add_argument('--M',type=int,default=65536);p.add_argument('--device',type=int,default=0)
    p.add_argument('--gain-cache-gb',type=float,default=3.);p.add_argument('--restart',type=int,default=80)
    p.add_argument('--linear-tol',type=float,default=1e-9)
    p.add_argument('--initial-tangent',help='Same-orbit saved derivative; solve its recomputed linear defect without changing the Jacobian')
    p.add_argument('--nearby-tangent-guess',action='store_true',help='Use a tightly checked neighboring root only as a linear-solver initial guess; target derivative and all residual gates remain independent')
    main(p.parse_args())
