"""Continue the unchanged Galerkin cycle with period as a local coordinate.

Period is a numerical continuation coordinate only. The scientific parameter is
still the fixed spatial Z field indexed by D, with M dynamic.
"""
from exact_periodic import *
from native_cycles import save
import argparse


def main(a):
    s=model();{'rate':attach_rate_entry_path,'native':attach_native_path,
               'fine':attach_fine_rate_entry_path}[a.family](s)
    z=np.load(a.orbit);N=a.N or len(z['r']);cls=ExactGalerkin
    if a.compact:
        from compact_periodic import CompactExactGalerkin
        assert read(OUT/'compact_periodic_check.json')['status']=='PASS'
        cls=CompactExactGalerkin
    if a.large_quadrature:
        from large_quadrature_periodic import LargeQuadratureGalerkin
        assert read(OUT/'large_quadrature_periodic_check.json')['status']=='PASS'
        cls=LargeQuadratureGalerkin
    cls.harmonic_block=a.harmonic_block
    o=cls(s,N,a.M,a.device)
    o.linear_tolerance_floor=a.linear_tol_floor
    o.normalize_fixed_D_column=a.normalize_parameter
    o.gain_cache_gb=a.gain_cache_gb
    if a.normalize_parameter:assert read(OUT/'fixed_period_parameter_scaling_check.json')['status']=='PASS'
    if a.gain_cache_gb>0:assert read(OUT/'gain_cache_parity.json')['status']=='PASS'
    o.log_linear_progress=a.log_linear_progress
    o.host_krylov=a.host_krylov
    if a.host_krylov:assert read(OUT/'host_krylov_check.json')['status']=='PASS'
    if a.fft_cache_plans:
        o.cp.fft.config.get_plan_cache().set_size(a.fft_cache_plans)
    elif a.fft_cache_mb is not None:
        if a.fft_cache_mb==0:o.cp.fft.config.get_plan_cache().set_size(0)
        else:o.cp.fft.config.get_plan_cache().set_memsize(a.fft_cache_mb*1024**2)
    o.cache_mean_operators=False;dest=OUT/'periodic'/a.label;dest.mkdir(parents=True,exist_ok=True)
    write(dest/'contract.json',dict(source_orbit=str(Path(a.orbit).resolve()),
        source_previous=str(Path(a.previous).resolve()) if a.previous else None,
        family=a.family,N=N,M=a.M,prescribed_periods_ms=a.periods,
        Z='held spatial path',M_state='dynamic',
        scope='Numerical continuation with explicit parent solutions; mesh changes remain separately validated and do not inherit stability.'))
    o.iteration_checkpoint=dest/'current_iterate.npz'
    n=N*s.P;c=np.zeros(n+2);c[-2]=1000.;rows=[]
    from scipy.signal import resample
    last=dict(r=resample(z['r'],N,axis=0) if N!=len(z['r']) else z['r'],T=float(z['T']),D=float(z['D']));previous=None
    last_path=str(Path(a.orbit).resolve())
    if 'tangent' in z:
        v=z['tangent'];assert len(v)==z['r'].size+2
        vr=v[:z['r'].size].reshape(z['r'].shape)
        if N!=len(z['r']):vr=resample(vr,N,axis=0)
        assert abs(v[-2]-1)<1e-6,'Expected tangent with respect to log period'
        last['tangent_logT']=np.r_[vr.ravel(),v[-2:]]
    if a.previous:
        zz=np.load(a.previous);previous=dict(r=resample(zz['r'],N,axis=0) if N!=len(zz['r']) else zz['r'],T=float(zz['T']),D=float(zz['D']))
        assert previous['r'].shape==last['r'].shape
    def advance(target,depth=0):
        nonlocal last,previous,last_path
        guess_r=last['r'];guess_D=last['D'];predictor='constant'
        if 'tangent_logT' in last and abs(np.log(target/last['T']))<.01:
            step=np.log(target/last['T']);v=last['tangent_logT']
            guess_r=last['r']+.001*step*v[:n].reshape(last['r'].shape)
            guess_D=last['D']+.001*step*v[-1];predictor='analytic source tangent'
        elif previous is not None and abs(last['T']-previous['T'])>1e-8:
            ratio=(target-last['T'])/(last['T']-previous['T'])
            if 0<abs(ratio)<=3:
                guess_r=last['r']+ratio*(last['r']-previous['r'])
                guess_D=last['D']+ratio*(last['D']-previous['D']);predictor='secant'
        pred=np.r_[(guess_r/.001).ravel(),np.log(target),guess_D*1000]
        arc=(pred,c,np.ones(n+2))
        if a.check_predictors and predictor!='constant':
            cp=o.cp;choices=[]
            for rr,dd,kind in [(guess_r,guess_D,predictor),(last['r'],last['D'],'constant')]:
                reference=cp.asarray(rr)
                phase=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(reference,axis=0),n=N,axis=0)
                yy=cp.r_[(reference/.001).ravel(),np.log(target),dd*1000]
                aa=(yy.copy(),cp.asarray(c),cp.ones(n+2))
                try:error=float(cp.linalg.norm(o.evaluate(yy,reference,phase,dd,arc=aa)))
                except ValueError:error=float('inf')
                choices.append((error,rr,dd,kind))
            error,guess_r,guess_D,predictor=min(choices,key=lambda q:q[0])
            log('PREDICTOR RESIDUALS',[(q[3],q[0]) for q in choices],'selected',predictor)
            pred=np.r_[(guess_r/.001).ravel(),np.log(target),guess_D*1000]
            arc=(pred,c,np.ones(n+2))
        if a.eliminate_period:
            from fixed_period_corrector import solve_fixed_period
            assert read(OUT/'fixed_period_corrector_check.json')['status']=='PASS'
            sol=solve_fixed_period(o,guess_r,target,guess_D,maxiter=a.maxiter,tol=2e-8,
                                   restart=a.restart,maxit_lin=a.maxiter_linear)
        else:
            sol=o.solve(guess_r,target,guess_D,arc=arc,maxiter=a.maxiter,tol=2e-8,restart=a.restart,maxit_lin=a.maxiter_linear)
        if sol['residual']>=2e-8:
            log('PERIOD STEP REJECTED',target,sol['residual'],depth)
            if depth>=4:raise RuntimeError(('period step unresolved',target,sol['residual']))
            mid=(last['T']+target)/2
            advance(mid,depth+1);advance(target,depth+1);return
        assert abs(sol['T']-target)<1e-7
        q=save(s,sol,dest,f'point{len(rows):04d}')
        q.update(nonlinear_samples=a.M,parameter_columns='analytic',
                 source_orbit=last_path,
                 method='dealiased Fourier-Galerkin; prescribed period corrector',
                 inner_linear_tolerance_floor=a.linear_tol_floor,
                 period_step_ms=sol['T']-last['T'],D_step=sol['D']-last['D'],predictor=predictor,
                 period_constraint='exact variable elimination' if a.eliminate_period else 'augmented linear constraint')
        rows.append(q);previous=last;last=sol;last_path=q['path']
        write(dest/'result.json',dict(status='RUNNING',family=a.family,rows=rows))
        log('PERIOD POINT',q)
    for target in a.periods:advance(target)
    write(dest/'result.json',dict(status='SEGMENT_COMPLETE',family=a.family,rows=rows,
            bifurcation_type='NOT_INFERRED_FROM_NUMERICAL_TURNS'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--label',required=True)
    p.add_argument('--periods',type=float,nargs='+',required=True)
    p.add_argument('--family',choices=['native','rate','fine'],default='rate')
    p.add_argument('--M',type=int,default=8192);p.add_argument('--device',type=int,default=0)
    p.add_argument('--harmonic-block',type=int,default=129)
    p.add_argument('--fft-cache-mb',type=int)
    p.add_argument('--fft-cache-plans',type=int,default=0)
    p.add_argument('--maxiter',type=int,default=12)
    p.add_argument('--N',type=int,help='Refine the temporal harmonic representation without changing the model')
    p.add_argument('--restart',type=int,default=40)
    p.add_argument('--linear-tol-floor',type=float)
    p.add_argument('--maxiter-linear',type=int,default=1500)
    p.add_argument('--log-linear-progress',action='store_true')
    p.add_argument('--previous',help='Previous converged point for a secant predictor')
    p.add_argument('--compact',action='store_true')
    p.add_argument('--large-quadrature',action='store_true',help='Same BVP with host inputs and componentwise GPU contraction')
    p.add_argument('--gain-cache-gb',type=float,default=0.,help='Exact float64 gain blocks cached on GPU, remaining blocks on host')
    p.add_argument('--normalize-parameter',action='store_true',help='Right-scale the fixed-period D column to unit norm; unchanged root tolerance')
    p.add_argument('--host-krylov',action='store_true')
    p.add_argument('--eliminate-period',action='store_true')
    p.add_argument('--check-predictors',action='store_true',help='Compare constant and tangent/secant residuals before Newton; accepted-root tolerance is unchanged')
    main(p.parse_args())
