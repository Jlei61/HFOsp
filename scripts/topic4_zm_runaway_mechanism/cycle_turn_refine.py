"""Refine a candidate periodic-branch turn using period as a local coordinate.

The result is a numerical branch-turn test until temporal refinement and
transverse Floquet evidence establish a physical cycle bifurcation.
"""
from native_path import *
from periodic_v3 import PeriodicV3,RS
from scipy.signal import resample
from scipy.optimize import brentq
import argparse


def main(a):
    from cupyx.scipy.sparse.linalg import gmres
    if a.host_krylov:
        from host_krylov import host_gmres as gmres
        assert read(OUT/'host_krylov_check.json')['status']=='PASS'
    s=model();(attach_rate_entry_path if a.family=='rate' else attach_native_path)(s)
    if a.large_quadrature:
        from large_quadrature_periodic import LargeQuadratureGalerkin
        assert a.M and read(OUT/'large_quadrature_periodic_check.json')['status']=='PASS'
        LargeQuadratureGalerkin.harmonic_block=a.harmonic_block
        o=LargeQuadratureGalerkin(s,a.N,a.M,a.device)
        o.gain_cache_gb=a.gain_cache_gb
        o.linear_tolerance_floor=a.linear_tol_floor
        o.log_linear_progress=True
        o.cp.fft.config.get_plan_cache().set_size(8)
    elif a.compact:
        from compact_periodic import CompactExactGalerkin
        assert a.M and read(OUT/'compact_periodic_check.json')['status']=='PASS'
        CompactExactGalerkin.harmonic_block=a.harmonic_block
        o=CompactExactGalerkin(s,a.N,a.M,a.device)
        o.linear_tolerance_floor=a.linear_tol_floor
        o.log_linear_progress=True
    elif a.exact_columns:
        from exact_periodic import ExactGalerkin
        assert a.M
        ExactGalerkin.harmonic_block=a.harmonic_block
        o=ExactGalerkin(s,a.N,a.M,a.device)
    elif a.M:
        if a.stream:
            from streaming_periodic import StreamGalerkin
            o=StreamGalerkin(s,a.N,a.M,a.device)
        else:
            from galerkin_cycles import Galerkin
            o=Galerkin(s,a.N,a.M,a.device)
    elif a.stream:
        from streaming_periodic import StreamPeriodic
        o=StreamPeriodic(s,a.N,a.device)
    else:o=PeriodicV3(s,a.N,a.device)
    if a.no_operator_cache:o.cache_mean_operators=False
    if a.no_fft_cache:o.cp.fft.config.get_plan_cache().set_size(0)
    o.host_krylov=a.host_krylov
    cp=o.cp;dest=OUT/'periodic'/a.label;dest.mkdir(parents=True,exist_ok=True)
    cache=[];results=[];evaluated={}
    for f in [a.first,a.second]:
        z=np.load(f);cache.append(dict(r=resample(z['r'],a.N,axis=0),T=float(z['T']),D=float(z['D'])))
    endpoints=[x['T'] for x in cache]
    if a.resume and (dest/'evaluations.json').exists():
        previous=read(dest/'evaluations.json')
        for k,q in enumerate(previous):
            z=np.load(dest/f'eval_{k:03d}.npz')
            assert len(z['r'])==a.N and float(z['residual'])<2e-8
            sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']),residual=float(z['residual']),
                     tangent_logT=z['tangent'])
            cache.append(sol);results.append(q)
            evaluated[q['T_ms']]=(q['dD_dT'],sol,z['tangent'])
        log('RESUMED TURN EVALUATIONS',len(results))
    n=a.N*s.P;c=np.zeros(n+2);c[-2]=1000.
    def solve(T):
        if T in evaluated:return evaluated[T]
        z=min(reversed(cache),key=lambda x:abs(x['T']-T))
        guess_r=z['r'];guess_D=z['D'];increment=np.log(T/z['T'])
        if 'tangent_logT' in z and abs(increment)<.01 and not a.eliminate_period:
            v=z['tangent_logT']
            guess_r=guess_r+RS*increment*v[:n].reshape(guess_r.shape)
            guess_D=guess_D+.001*increment*v[-1]
        pred=np.r_[(guess_r/RS).ravel(),np.log(T),guess_D*1000]
        arc=(pred,c,np.ones(n+2))
        if a.eliminate_period:
            from fixed_period_corrector import solve_fixed_period
            assert a.large_quadrature and a.family=='native'
            assert read(OUT/'fixed_period_corrector_check.json')['status']=='PASS'
            o.normalize_fixed_D_column=True
            sol=solve_fixed_period(o,guess_r,T,guess_D,maxiter=24,tol=2e-8,restart=a.restart)
        else:
            sol=o.solve(guess_r,T,guess_D,arc=arc,maxiter=24,tol=2e-8,restart=a.restart)
        assert sol['residual']<2e-8,('period-constrained BVP failed',T,sol['residual'])
        assert abs(sol['T']-T)<1e-6,('period constraint unresolved',T,sol['T'])
        s.set_D(sol['D'])
        np.savez_compressed(dest/'last_converged_BVP.npz',r=sol['r'],T=sol['T'],D=sol['D'],Z=s.Z,residual=sol['residual'])
        if a.eliminate_period:
            from fixed_period_tangent import tangent_at
            v,details=tangent_at(o,sol,a.restart)
            der=details['dD_dT'];err=details['linear_relative_residual']
        else:
            y=cp.asarray(sol['y']);rr=cp.asarray(sol['r'])
            dr=cp.fft.irfft(2j*np.pi*cp.arange(o.K)[:,None]*cp.fft.rfft(rr,axis=0),n=a.N,axis=0)
            phase=dr/cp.sum(dr*dr)*RS
            _,A,_=o.evaluate(y,rr,phase,sol['D'],arc=tuple(cp.asarray(x) for x in arc),derivative=True)
            rhs=cp.zeros(n+2);rhs[-1]=1.
            tangent,info=gmres(A,rhs,tol=1e-9,atol=1e-12,restart=a.restart,maxiter=1500,
                callback=lambda v:log('TANGENT residual',float(v)),callback_type='pr_norm',
                M=o.preconditioner(A.shape[0]) if a.exact_columns or a.compact else None)
            err=float(cp.linalg.norm(A@tangent-rhs));assert err<1e-6,err
            v=tangent.get()*1000.;der=float(v[-1]/1000/T)
        sol['tangent_logT']=v;cache.append(sol)
        q=dict(T_ms=T,D=sol['D'],dD_dT=der,bvp_residual_hz=sol['residual'],linear_residual=err,
               linear_residual_definition='relative, reduced fixed-period derivative equation' if a.eliminate_period else 'absolute, augmented derivative equation',
               N=a.N,M=a.M)
        s.set_D(sol['D'])
        np.savez_compressed(dest/f'eval_{len(results):03d}.npz',r=sol['r'],T=sol['T'],D=sol['D'],Z=s.Z,tangent=v,residual=sol['residual'])
        results.append(q);write(dest/'evaluations.json',results);log('TURN EVAL',q)
        evaluated[T]=(der,sol,v)
        return evaluated[T]
    lo,hi=sorted(endpoints);fl=solve(lo)[0];fh=solve(hi)[0]
    if fl*fh>=0:
        write(dest/'result.json',dict(status='DERIVATIVE_SIGN_BRACKET_MISSING',N=a.N,M=a.M,rows=results));return
    T=brentq(lambda x:solve(x)[0],lo,hi,xtol=a.period_tol,rtol=1e-11)
    der,sol,v=solve(T);h=min(.005,(hi-lo)/10)
    s.set_D(sol['D'])
    np.savez_compressed(dest/'turn_candidate.npz',r=sol['r'],T=sol['T'],D=sol['D'],Z=s.Z,tangent=v,residual=sol['residual'])
    if a.local_curvature:
        from cycle_local_curvature import curvature
        details=curvature(o,sol,v,restart=a.restart,steps=a.curvature_steps)
        write(dest/'local_curvature.json',details)
        assert details['status']=='LOCAL_CURVATURE_CONVERGED',details
        curv=details['d2D_dT2']
    else:
        curv=(solve(T+h)[0]-solve(T-h)[0])/(2*h)
    r=sol['r'];g=r[:,s.E]@s.mean_weights*1000;s.set_D(sol['D'])
    np.savez_compressed(dest/'turn.npz',r=r,T=sol['T'],D=sol['D'],Z=s.Z,tangent=v,residual=sol['residual'])
    row=dict(status='NUMERICAL_PERIODIC_TURN_REFINED',N=a.N,M=a.M,T_ms=sol['T'],D=sol['D'],
        dD_dT=der,d2D_dT2=curv,mean_hz=float(g.mean()),min_hz=float(g.min()),max_hz=float(g.max()),
        bvp_residual_hz=sol['residual'],minimum_group_rate_hz=float(r.min()*1000),
        period_root_tolerance_ms=a.period_tol,
        bifurcation_type='NOT_YET_ESTABLISHED; requires resolution and transverse Floquet checks',
        orbit=str(dest/'turn.npz'))
    write(dest/'result.json',row);log('TURN RESULT',row)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second');p.add_argument('--label',required=True)
    p.add_argument('--N',type=int,default=1024);p.add_argument('--M',type=int,default=0)
    p.add_argument('--family',choices=['native','rate'],default='native')
    p.add_argument('--exact-columns',action='store_true');p.add_argument('--no-operator-cache',action='store_true')
    p.add_argument('--compact',action='store_true')
    p.add_argument('--large-quadrature',action='store_true')
    p.add_argument('--gain-cache-gb',type=float,default=3.)
    p.add_argument('--eliminate-period',action='store_true',help='Use audited native fixed-period corrector and reduced branch tangent; same equations and root tolerance')
    p.add_argument('--linear-tol-floor',type=float)
    p.add_argument('--period-tol',type=float,default=2e-7)
    p.add_argument('--local-curvature',action='store_true',help='Use step-checked local BVP second derivative instead of two full offset continuations')
    p.add_argument('--curvature-steps',type=float,nargs=2,default=[1e-6,5e-7],
                   help='Directional log-period steps; choose jointly with nonlinear quadrature resolution for clamped response tables')
    p.add_argument('--host-krylov',action='store_true')
    p.add_argument('--resume',action='store_true',help='Reuse saved accepted BVP/tangent evaluations at these meshes')
    p.add_argument('--harmonic-block',type=int,default=129);p.add_argument('--restart',type=int,default=40)
    p.add_argument('--no-fft-cache',action='store_true')
    p.add_argument('--device',type=int,default=1);p.add_argument('--stream',action='store_true');main(p.parse_args())
