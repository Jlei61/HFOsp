"""Keep GMRES vectors in host RAM; retain the identical GPU Jacobian action.

Explicit right preconditioning matches CuPy's convention. SciPy's default
left-preconditioned M is deliberately not used here.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator,gmres


def host_gmres(A,b,**kw):
    import cupy as cp
    M=kw.pop('M',None);tol=kw.pop('tol',1e-5);atol=kw.pop('atol',0.)
    restart=kw.pop('restart',20);limit=kw.pop('maxiter',1500)
    callback=kw.pop('callback',None);kw.pop('callback_type',None)
    assert not kw,kw
    def precondition(v):return M@v if M is not None else v
    def action(v):return (A@precondition(cp.asarray(v))).get()
    B=LinearOperator(A.shape,matvec=action,dtype=np.float64);iteration=[0]
    def progress(v):
        iteration[0]+=1
        if callback is not None and iteration[0]%restart==0:callback(v)
    z,info=gmres(B,b.get(),rtol=tol,atol=atol,restart=restart,
                 maxiter=int(np.ceil(limit/restart)),callback=progress,callback_type='pr_norm')
    return precondition(cp.asarray(z)),info


if __name__=='__main__':
    from compact_periodic import *
    from scipy.signal import resample
    from cupyx.scipy.sparse.linalg import gmres as gpu_gmres
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);a=p.parse_args()
    s=model();attach_rate_entry_path(s);z=np.load(OUT/'periodic/rate_seed_N1024.npz')
    CompactExactGalerkin.harmonic_block=33;o=CompactExactGalerkin(s,513,2048,a.device)
    o.cache_mean_operators=False;cp=o.cp;cp.fft.config.get_plan_cache().set_size(0)
    r=cp.asarray(resample(z['r'],513,axis=0));y=cp.r_[(r/RS).ravel(),np.log(float(z['T']))]
    phase=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(r,axis=0),n=o.N,axis=0)
    f,A,_=o.evaluate(y,r,phase,float(z['D']),derivative=True)
    kw=dict(tol=1e-7,atol=1e-12,restart=80,maxiter=480,M=o.preconditioner(len(y)))
    x,ix=gpu_gmres(A,-f,**kw);h,ih=host_gmres(A,-f,**kw)
    errors=[float(cp.linalg.norm(A@v+f)/cp.linalg.norm(f)) for v in [x,h]]
    difference=float(cp.linalg.norm(h-x)/cp.linalg.norm(x))
    assert max(errors)<2e-7 and difference<1e-4,(errors,difference)
    q=dict(status='PASS',gpu_host_relative_linear_residuals=errors,
           relative_correction_difference=difference,preconditioning='explicit identical right scaling',
           model_changed=False)
    write(OUT/'host_krylov_check.json',q);log('HOST KRYLOV CHECK',q)
