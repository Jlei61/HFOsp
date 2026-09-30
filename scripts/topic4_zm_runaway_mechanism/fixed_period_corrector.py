"""Exactly eliminate the prescribed period from a periodic BVP corrector.

The earlier augmented formulation permits an inexact Krylov solve to leak into
the nominally fixed period. Here Newton only solves for rates and D; the same
full residual/Jacobian supplies all retained rows and columns.
"""
from native_path import *
from periodic_v3 import RS


def solve_fixed_period(o,r,T,D,maxiter=12,tol=2e-8,restart=80,maxit_lin=1500):
    cp=o.cp;s=o.s;n=o.N*s.P
    from cupyx.scipy.sparse.linalg import LinearOperator,gmres
    if getattr(o,'host_krylov',False):
        from host_krylov import host_gmres as gmres
    reference=cp.asarray(r);phase=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(reference,axis=0),n=o.N,axis=0)
    x=cp.r_[(reference/RS).ravel(),D*1000];logT=float(np.log(T));history=[];start=time.time()
    def embed(v,direction=False):
        return cp.r_[v[:n],0. if direction else logT,v[-1]]
    c=cp.zeros(n+2);c[-2]=1000.;weight=cp.ones(n+2)
    def evaluate(v,derivative=False):
        y=embed(v);arc=(y.copy(),c,weight)
        if not derivative:return o.evaluate(y,reference,phase,D,arc=arc)[:-1]
        full,A,_=o.evaluate(y,reference,phase,D,arc=arc,derivative=True)
        return full[:-1],LinearOperator((n+1,n+1),matvec=lambda w:(A@embed(w,True))[:-1],dtype=np.float64)
    for it in range(maxiter):
        F,A=evaluate(x,True);err=float(cp.max(cp.abs(F)));history.append(err)
        log('FIXED T BVP',o.N,it,'D',float(x[-1]*.001),'T',T,'err',err,'sec',round(time.time()-start,1))
        if err<tol:break
        kw=dict(tol=min(.001,max(1e-8,err*.01)),atol=1e-11,restart=restart,maxiter=maxit_lin)
        floor=getattr(o,'linear_tolerance_floor',None)
        if floor is not None:kw['tol']=max(kw['tol'],floor)
        if getattr(o,'normalize_fixed_D_column',False):
            scale=cp.ones(n+1);scale[-1]=1/max(o.fixed_D_column_norm,1.)
            kw['M']=LinearOperator((n+1,n+1),matvec=lambda w:w*scale,dtype=np.float64)
            log('FIXED T D COLUMN SCALE',float(scale[-1]))
        if getattr(o,'log_linear_progress',False):
            kw.update(callback=lambda value:log('FIXED T GMRES',float(value)),callback_type='pr_norm')
        dx,info=gmres(A,-F,**kw)
        linear=float(cp.linalg.norm(A@dx+F)/cp.linalg.norm(F));log('FIXED T LINEAR',info,linear)
        del A;cp.get_default_memory_pool().free_all_blocks()
        oldnorm=float(cp.linalg.norm(F))
        for k in range(16):
            trial=x+dx*2.**-k
            if not 0<=float(trial[-1]*.001)<=1:continue
            try:ff=evaluate(trial)
            except ValueError:continue
            if float(cp.linalg.norm(ff))<oldnorm:x=trial;break
        else:break
    err=float(cp.max(cp.abs(evaluate(x))))
    return dict(r=x[:n].reshape(o.N,s.P).get()*RS,T=float(T),D=float(x[-1]*.001),
                residual=err,history=history,y=embed(x).get(),period_constraint='exact variable elimination')


if __name__=='__main__':
    from compact_periodic import CompactExactGalerkin
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);a=p.parse_args()
    s=model();attach_rate_entry_path(s)
    path=OUT/'periodic/rate_period_path_G2049_M8192/point0000.npz';z=np.load(path)
    CompactExactGalerkin.harmonic_block=33;o=CompactExactGalerkin(s,len(z['r']),8192,a.device)
    o.cache_mean_operators=False;o.cp.fft.config.get_plan_cache().set_size(0)
    sol=solve_fixed_period(o,z['r'],float(z['T']),float(z['D'])+1e-6,maxiter=10,restart=60)
    Derror=abs(sol['D']-float(z['D']));rerror=float(np.linalg.norm(sol['r']-z['r'])/np.linalg.norm(z['r']))
    q=dict(status='PASS' if sol['residual']<2e-8 and Derror<1e-9 and rerror<1e-6 else 'FAIL',
           source=str(path),initial_D_perturbation=1e-6,residual=sol['residual'],
           D_recovery_error=Derror,rate_relative_recovery_error=rerror,
           T_exactly_preserved=sol['T']==float(z['T']),history=sol['history'])
    write(OUT/'fixed_period_corrector_check.json',q);log('FIXED PERIOD CHECK',q)
    assert q['status']=='PASS'
