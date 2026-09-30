"""Fixed-D continuation on the actual fine spatial-Z path.

Accepted roots retain the unchanged model and strict BVP tolerance. Failed
correctors are recorded separately and do not identify a bifurcation. Period
is solved, rather than assumed to be a monotone local continuation coordinate.
"""
from native_path import *
from compact_periodic import CompactExactGalerkin
from native_cycles import save
from scipy.signal import resample
import argparse


def main(a):
    s=model();path={'fine':attach_fine_rate_entry_path,'native':attach_native_path,
                    'rate':attach_rate_entry_path}[a.family](s);z=np.load(a.orbit)
    assert float(z['residual'])<2e-8
    N=a.N or len(z['r']);s.set_D(float(z['D']))
    assert np.max(abs(s.Z-z['Z']))<1e-12
    last=dict(r=resample(z['r'],N,axis=0),T=float(z['T']),D=float(z['D']))
    previous=None;rows=[];attempts=[]
    if a.previous:
        old=np.load(a.previous);assert float(old['residual'])<2e-8
        s.set_D(float(old['D']));assert np.max(abs(s.Z-old['Z']))<1e-12
        previous=dict(r=resample(old['r'],N,axis=0),T=float(old['T']),D=float(old['D']))
        s.set_D(last['D'])
    cls=CompactExactGalerkin
    if a.large_quadrature:
        from large_quadrature_periodic import LargeQuadratureGalerkin
        assert read(OUT/'large_quadrature_periodic_check.json')['status']=='PASS'
        cls=LargeQuadratureGalerkin
    cls.harmonic_block=a.harmonic_block
    o=cls(s,N,a.M,a.device);o.gain_cache_gb=a.gain_cache_gb
    if a.gain_cache_gb:assert read(OUT/'gain_cache_parity.json')['status']=='PASS'
    o.linear_tolerance_floor=a.linear_tol_floor
    o.linear_tolerance_cap=a.linear_tol_cap
    o.cache_mean_operators=False
    if a.fft_cache_plans:o.cp.fft.config.get_plan_cache().set_size(a.fft_cache_plans)
    elif a.fft_cache_mb:o.cp.fft.config.get_plan_cache().set_memsize(a.fft_cache_mb*1024**2)
    else:o.cp.fft.config.get_plan_cache().set_size(0)
    o.host_krylov=True;assert read(OUT/'host_krylov_check.json')['status']=='PASS'
    dest=OUT/'periodic'/a.label;dest.mkdir(parents=True,exist_ok=True)
    o.iteration_checkpoint=dest/'current_iterate.npz'
    if a.verify_source:
        ff,_=o.compact_residual(o.cp.asarray(last['r']),last['T'],s.Z,False)
        error=float(o.cp.max(o.cp.abs(ff)));del ff
        assert error<2e-8,('Source root must pass unchanged-grid residual check',error)
        write(dest/'source_check.json',dict(status='PASS',source=a.orbit,residual_hz=error,N=N,M=a.M))
        log('SOURCE ROOT CHECK',error)
    def record(status):
        write(dest/'result.json',dict(status=status,family=a.family,rows=rows,
            attempts=attempts,source=a.orbit,source_path=path,
            bifurcation_type='NOT_INFERRED_FROM_CORRECTOR_FAILURE',Z='held',M='dynamic'))
    def advance(D,depth=0):
        nonlocal last,previous
        guess=last['r'];T=last['T'];predictor='constant'
        predictor_errors={}
        if previous is not None:
            ratio=(D-last['D'])/(last['D']-previous['D'])
            if 0<abs(ratio)<=3:
                guess=last['r']+ratio*(last['r']-previous['r'])
                T=last['T']+ratio*(last['T']-previous['T']);predictor='secant'
        if predictor=='secant':
            # Sharp bursts move slightly in phase between roots. Extrapolation
            # can amplify their truncation error; choose by the actual BVP
            # residual, without changing the target equation or acceptance.
            s.set_D(D)
            for name,rr,tt in [('secant',guess,T),('constant',last['r'],last['T'])]:
                ff,_=o.compact_residual(o.cp.asarray(rr),tt,s.Z,False)
                predictor_errors[name]=float(o.cp.linalg.norm(ff))
                del ff
            if predictor_errors['constant']<predictor_errors['secant']:
                guess=last['r'];T=last['T'];predictor='constant'
            log('PREDICTOR',D,predictor,predictor_errors)
        sol=o.solve(guess,T,D,maxiter=12,tol=2e-8,restart=a.restart,maxit_lin=900)
        attempts.append(dict(D=D,residual=sol['residual'],history=sol['history'],depth=depth,
                             predictor=predictor,predictor_residual_norms=predictor_errors))
        record('RUNNING')
        if sol['residual']>=2e-8:
            log('FIXED D STEP REJECTED',D,sol['residual'],depth)
            if depth>=3 or len(attempts)>=a.max_attempts:
                record('CORRECTOR_UNRESOLVED');return False
            return advance((D+last['D'])/2,depth+1) and advance(D,depth+1)
        q=save(s,sol,dest,f'point{len(rows):04d}')
        q.update(family=a.family,nonlinear_samples=a.M,D_step=D-last['D'],predictor=predictor,
                 method='Fixed D; dealiased Fourier-Galerkin; period and rates solved')
        rows.append(q);previous=last;last=sol;record('RUNNING');log('FIXED D POINT',q)
        return True
    for D in a.D:
        assert 0<=D<=1
        s.set_D(D)  # The selected spatial path performs its own domain check.
        if not advance(D):return
    record('SEGMENT_COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('--D',nargs='+',type=float,required=True)
    p.add_argument('--label',required=True);p.add_argument('--N',type=int)
    p.add_argument('--M',type=int,default=65536);p.add_argument('--device',type=int,default=1)
    p.add_argument('--restart',type=int,default=60);p.add_argument('--harmonic-block',type=int,default=129)
    p.add_argument('--max-attempts',type=int,default=20)
    p.add_argument('--large-quadrature',action='store_true')
    p.add_argument('--gain-cache-gb',type=float,default=0.)
    p.add_argument('--fft-cache-mb',type=int,default=0)
    p.add_argument('--fft-cache-plans',type=int,default=0)
    p.add_argument('--linear-tol-floor',type=float)
    p.add_argument('--verify-source',action='store_true')
    p.add_argument('--family',choices=['fine','native','rate'],default='fine')
    p.add_argument('--previous')
    p.add_argument('--linear-tol-cap',type=float)
    main(p.parse_args())
