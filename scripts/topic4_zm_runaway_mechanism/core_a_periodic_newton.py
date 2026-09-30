"""Correct an individual local-burst near return in the full delayed model.

No low-dimensional fit: matrix-free Newton uses the verified variational
equations of all current spatial states, M and delay history.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build,regrid_state
from onset_poincare_corrector import SectionReturn,regional_rate_section
from onset_shooting_newton import SectionDerivative,gmres_step
from onset_period_return import errors,dynamical_state
import argparse,os,time

SEED=OUT/'core_a_bifurcation_type_20260924/near_returns/mid_lower/exact_seed'
DEST=SEED.parent/'periodic_newton'


def main(device,maximum,iterations,section='flow',resume=None,period_override=None,name=None,positive=False,dt=.05,source_dt=.05,interpolation='linear',target_D=None,destination=None,halfwidth=None,orientation=None,cached_derivative=False,projected_proposal=False,target_native_time=None):
    global DEST
    assert not projected_proposal or positive
    assert target_D is None or target_native_time is None
    if section!='flow':DEST=DEST.with_name(DEST.name+'_'+section)
    if name:DEST=DEST.with_name(name)
    if destination:
        from pathlib import Path
        DEST=Path(destination).resolve()
    DEST.mkdir(exist_ok=True);assert not (DEST/'jobs.json').exists()
    if resume:
        from pathlib import Path
        source=Path(resume);assert period_override is not None
        period=period_override
    else:
        candidate=read(SEED/'result.json');assert candidate['status']=='EXACT_REPLAY_PASS_RECURRENCE_ONLY'
        tm=candidate['times_ms'][0];period=candidate['period_ms'];source=SEED/f'state{tm}.npz'
    halfwidth=min(20.,period*.25) if halfwidth is None else halfwidth
    write(DEST/'contract.json',dict(question='Does the declared recurrent full spatial state correct to an actual periodic orbit of the unchanged model?',
        source=str(source),dt_ms=dt,source_dt_ms=source_dt,all_M_dynamic=True,all_Z_held=True,section=section,section_orientation=orientation,interpolation=interpolation,target_D_A=target_D,target_native_path_time_ms=target_native_time,cached_derivative=cached_derivative,
        positive_coordinate_proposals=positive,explicit_source_supplied=bool(resume),return_halfwidth_ms=halfwidth,
        projected_positive_proposal=projected_proposal,
        proposal_method='Only outward negative numerical coordinates projected to the nonnegative boundary, then exact section restoration; original physical flow is never clipped' if projected_proposal else 'Original rational positive retraction with unchanged admissible Newton steps',
        method='Actual full-state positive Poincare return, full variational derivative including return time, matrix-free Newton-GMRES. Fixed section through source. No rate-only recurrence or projected dynamics.',
        gates='First section derivative must match nonlinear full-state finite differences; no inadmissible states or clipping. Numerical orbit requires combined closure<1e-7 and every block<1e-6. Independent phase, mesh, fundamental period and Floquet checks remain necessary.',
        limits=dict(nonlinear_iterations=iterations,Krylov_per_step=maximum,admissible_linesearch_returns=4),model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid(),stage='initial_return');write(DEST/'jobs.json',jobs);start=time.time()
    Return,Derivative=SectionReturn,SectionDerivative
    if interpolation=='cubic':
        from onset_cubic_section import CubicSectionReturn,CubicSectionDerivative
        Return,Derivative=CubicSectionReturn,CubicSectionDerivative
    if cached_derivative:
        assert interpolation=='cubic' and dt in [.05,.025]
        tag='exact_cached_tangent'+('_fine' if dt==.025 else '')
        qa=read(OUT/'core_a_bifurcation_type_20260924/numerical_checks'/tag/'result.json')
        assert qa['status']=='PASS' and qa['dt_ms']==dt
        from onset_cached_tangent import CachedCubicSectionDerivative
        Derivative=CachedCubicSectionDerivative
    e=build(device,dt=dt);base=regrid_state(np.load(source),e,source_dt)
    if target_D is not None or target_native_time is not None:
        if target_native_time is not None:
            from core_a_parameter_path_audit import NativeTimeFamily
            family=NativeTimeFamily(e.s);z,display_D=family.field_at_time(target_native_time)
            tm=target_native_time
        else:
            from core_a_equilibrium_branch import Family
            family=Family(e.s);z,tm=family.field(target_D);display_D=target_D
        assert np.array_equal(z[~family.A],base['syn'][5,~family.A])
        base['syn'][5]=z
        write(DEST/'target_field.json',dict(D_A=display_D,native_field_coordinate_ms=tm,
            path_parameter='native_time' if target_native_time is not None else 'first_upcrossing_mean_D',
            outside_A_unchanged=True,all_M_dynamic=True))
    A=Return(base,e,period,halfwidth)
    if section in ['core_A','core_B']:
        # A change of transversal section is a numerical coordinate choice,
        # not a change to the physical dynamics or the retained state space.
        regional_rate_section(A,0 if section=='core_A' else 1,orientation)
    x=A.xref.copy();rows=[];status='ITERATION_LIMIT_NOT_AN_ORBIT';checks=[]
    try:
        for it in range(iterations):
            step=DEST/f'iteration{it:02d}';step.mkdir(exist_ok=True)
            y,meta=A(x);f=y-x;slope=A.last_time_slope.copy()
            err=errors(dynamical_state(A.state(x)),dynamical_state(A.state(y)),e.s.sizes/e.s.sizes.sum())
            row=dict(iteration=it,**meta,**err,coordinate_residual=float(np.linalg.norm(f)/np.linalg.norm(x)))
            rows.append(row);write(DEST/'iterations.json',rows);log('CORE A PERIODIC NEWTON',row)
            np.savez_compressed(DEST/'latest_state.npz',**A.state(x))
            if err['combined_relative_rms']<1e-7 and max(v['relative_rms'] for v in err['blocks'].values())<1e-6:
                status='NUMERICAL_PERIODIC_ROOT';break
            if it:
                del J
                import gc
                gc.collect();e.cp.get_default_memory_pool().free_all_blocks()
            J=Derivative(A,x,meta['period_ms'],slope)
            if it==0:
                derivative=J(f)
                # A fractional section return is piecewise differentiable in
                # its time bracket. Large perturbations may cross a mesh knot;
                # reduce epsilon before rejecting the actual local derivative.
                for eps in [1e-3,5e-4,1e-4,3e-5,1e-5,3e-6,1e-6]:
                    assert A.admissible(x+eps*f)
                    q,inf=A(x+eps*f);fd=(q-y)/eps
                    check=dict(epsilon=eps,relative_error=float(np.linalg.norm(fd-derivative)/max(np.linalg.norm(derivative),1e-30)),
                        phase_leakage=float(abs(A.normal@derivative)),period_ms=inf['period_ms'])
                    checks.append(check);write(DEST/'derivative_check.json',checks);log('CORE A PERIODIC DERIVATIVE',check)
                    if check['relative_error']<1e-3:break
                if checks[-1]['relative_error']>=1e-3 or checks[-1]['phase_leakage']>=1e-8:
                    status='SECTION_DERIVATIVE_GATE_FAILED';break
            jobs.update(iteration=it,stage='Newton_GMRES');write(DEST/'jobs.json',jobs)
            delta,linear=gmres_step(J,f,maximum,step)
            np.savez_compressed(step/'newton_direction.npz',delta=delta,source=x,normal=A.normal)
            attempts=0;trials=[];accepted=False
            for alpha in 2.**-np.arange(14):
                if positive:
                    from core_a_positive_newton_coordinates import retract
                    nxt=retract(A,x,delta,alpha,project_zero=projected_proposal,project_all=projected_proposal)
                else:nxt=x+alpha*delta
                if nxt is None or not A.admissible(nxt):trials.append(dict(alpha=float(alpha),status='INADMISSIBLE_NO_FLOW'));continue
                if attempts>=4:break
                attempts+=1
                try:
                    value,info=A(nxt);ratio=float(np.linalg.norm(value-nxt)/np.linalg.norm(f))
                    trial=dict(alpha=float(alpha),status='EVALUATED',residual_ratio=ratio,period_ms=info['period_ms'])
                except RuntimeError as exc:
                    trial=dict(alpha=float(alpha),status='SECTION_WINDOW_MISS',error=str(exc));ratio=np.inf
                trials.append(trial);write(step/'line_search.json',trials);log('CORE A PERIODIC LINE',trial)
                if ratio<1:
                    x=nxt;A.period=info['period_ms'];accepted=True
                    np.savez_compressed(step/'accepted_state.npz',**A.state(x))
                    write(step/'accepted_update.json',dict(period_ms=A.period,residual_ratio=ratio,
                        alpha=float(alpha),needs_next_residual_evaluation=True))
                    break
            row['line_search']=trials;row['linear_last']=linear[-1];write(DEST/'iterations.json',rows)
            if not accepted:status='NEWTON_STEP_NOT_ACCEPTED';break
        write(DEST/'result.json',dict(status=status,iterations=rows,derivative_checks=checks,dt_ms=dt,
            period_ms=rows[-1]['period_ms'],elapsed_seconds=time.time()-start,
            scope='Numerical full-state section root at most. No stability, period fold, crisis or native correspondence claimed.',model_promoted=False))
        jobs.update(status='COMPLETE',scientific_status=status);write(DEST/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(DEST/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);p.add_argument('--krylov',type=int,default=20);p.add_argument('--iterations',type=int,default=6)
    p.add_argument('--section',choices=['flow','core_A','core_B'],default='flow')
    p.add_argument('--resume');p.add_argument('--period',type=float);p.add_argument('--name');p.add_argument('--positive',action='store_true')
    p.add_argument('--dt',type=float,default=.05);p.add_argument('--source-dt',type=float,default=.05)
    p.add_argument('--interpolation',choices=['linear','cubic'],default='linear')
    p.add_argument('--target-D',type=float)
    p.add_argument('--target-native-time',type=float)
    p.add_argument('--destination');p.add_argument('--halfwidth',type=float)
    p.add_argument('--orientation',type=int,choices=[-1,1]);p.add_argument('--cached-derivative',action='store_true')
    p.add_argument('--projected-proposal',action='store_true')
    a=p.parse_args();main(a.device,a.krylov,a.iterations,a.section,a.resume,a.period,a.name,a.positive,a.dt,a.source_dt,a.interpolation,a.target_D,a.destination,a.halfwidth,a.orientation,a.cached_derivative,a.projected_proposal,a.target_native_time)
