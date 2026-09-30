"""Use period as a continuation coordinate for the full Core-A Z family.

An autonomous periodic root can turn back in D_A. The bordered physical
period equation permits that geometry without declaring a failed fixed-D
correction to be a bifurcation. All M remain dynamic in every flow call.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build
from onset_cubic_section import CubicSectionReturn
from onset_cached_tangent import CachedCubicSectionDerivative
from core_a_equilibrium_branch import Family
from core_a_positive_newton_coordinates import retract
from core_a_periodic_hookstep import krylov
from onset_period_return import errors,dynamical_state
from pathlib import Path
from types import SimpleNamespace
import argparse,os,time,gc


def main(a):
    out=Path(a.destination).resolve();out.mkdir(parents=True,exist_ok=True);assert not(out/'jobs.json').exists()
    assert read(OUT/'core_a_bifurcation_type_20260924/numerical_checks/exact_cached_tangent/result.json')['status']=='PASS'
    write(out/'contract.json',dict(source=str(Path(a.source).resolve()),target_period_ms=a.period,dt_ms=.05,
        question='Does a genuine full-state short periodic branch exist near the sustained-A state when D_A may vary as part of periodic continuation?',
        equations='Unchanged3479-group spatial model, native within-Core-A Z family only; all outside-A Z native9s, all M dynamic, every individual orbit has its complete Z field held. No physical parameter other than D_A is changed.',
        method='Bordered Newton-GMRES hookstep solves P(X,D_A)-X=0 and T_return(X,D_A)=target_period. Full actual variational derivative includes return time; parameter derivatives use two finite-difference steps of the actual native-A field. Exact float64 derivative caching has independent whole-state parity checks. Nonnegative bounds constrain numerical proposals only; full original residual and actual-step prediction gate each accepted update.',
        parameter_interval=[a.lower_D,a.upper_D],D_coordinate_weight=100.,period_residual_scale_ms=1.,
        maximum_iterations=a.iterations,Krylov_per_iteration=a.krylov,
        interpretation='A converged periodic root is not a cycle-fold or onset certificate. Need continuation, nontrivial actual activity, independent phase/mesh, critical multiplier/nondegeneracy and correspondence to the observed transition.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs);begin=time.time()
    e=build(a.device);base=dict(np.load(a.source));family=Family(e.s)
    D=float(1-np.average(base['syn'][5,family.A],weights=e.s.sizes[family.A]));z,_=family.field(D)
    assert np.max(abs(z-base['syn'][5]))<2e-12
    A=CubicSectionReturn(base,e,a.period,min(8.,a.period*.2));x=A.xref.copy()
    wd=100.;ts=1.;radius=a.radius*np.linalg.norm(x);rows=[];status='ITERATION_LIMIT_NOT_A_PERIODIC_ROOT'
    def field(d):
        z,tm=family.field(d);assert np.array_equal(z[~family.A],base['syn'][5,~family.A])
        A.base['syn'][5]=z;return tm
    def score(xx,d):
        field(d);y,meta=A(xx);f=np.r_[y-xx,(a.period-meta['period_ms'])/ts]
        err=errors(dynamical_state(A.state(xx)),dynamical_state(A.state(y)),e.s.sizes/e.s.sizes.sum())
        return f,meta,err,A.last_time_slope.copy()
    try:
        for iteration in range(a.iterations):
            step=out/f'iteration{iteration:02d}';step.mkdir(exist_ok=True)
            f,meta,err,slope=score(x,D);norm=float(np.linalg.norm(f))
            row=dict(iteration=iteration,D_A=D,Z_A=1-D,**meta,**err,augmented_residual=norm)
            rows.append(row);write(out/'iterations.json',rows);np.savez_compressed(out/'latest_state.npz',**A.state(x))
            jobs.update(iteration=iteration,D_A=D,last_residual=err['combined_relative_rms']);write(out/'jobs.json',jobs)
            log('FAST PERIOD FAMILY ROOT',row)
            if err['combined_relative_rms']<1e-7 and max(v['relative_rms'] for v in err['blocks'].values())<1e-6 and abs(meta['period_ms']-a.period)<1e-7:
                status='NUMERICAL_PERIODIC_ROOT';break
            parameter=[]
            for h in ([1e-6,5e-7] if iteration==0 else [5e-7]):
                values=[];times=[]
                for sign in [-1,1]:
                    field(D+sign*h);y,m=A(x);values.append(y);times.append(m['period_ms'])
                parameter.append(dict(h=h,p=(values[1]-values[0])/(2*h),t=(times[1]-times[0])/(2*h)))
            pd=parameter[-1]['p'];td=parameter[-1]['t'];field(D)
            if iteration==0:
                check=dict(relative_P_error=float(np.linalg.norm(parameter[0]['p']-pd)/np.linalg.norm(pd)),
                    relative_T_error=float(abs(parameter[0]['t']-td)/max(abs(td),1.)),T_D_ms_per_D=float(td))
                write(out/'parameter_derivative_check.json',check);assert max(check['relative_P_error'],check['relative_T_error'])<1e-3
            if iteration:
                del J;gc.collect();e.cp.get_default_memory_pool().free_all_blocks()
            J=CachedCubicSectionDerivative(A,x,meta['period_ms'],slope)
            def linear(v):
                # Trial flows may change A.base's parameter field. Every
                # derivative product must restore its own nominal D_A.
                field(D);jv=J(v[:-1]);tv=J.last_return_time_derivative;dd=v[-1]/wd
                return np.r_[v[:-1]-jv-pd*dd,(tv+td*dd)/ts]
            if iteration==0:
                rng=np.random.default_rng(92455);v=x*rng.standard_normal(x.size)
                free=np.zeros_like(v);free.reshape(-1,e.s.P)[11:47]=A.normal.reshape(-1,e.s.P)[11:47]
                v-=free*float(A.normal@v)/float(A.normal@free);v*=norm/np.linalg.norm(v)
                direction=np.r_[v,wd*1e-4];derivative=linear(direction);checks=[];fd_error=np.inf
                for eps in [1e-4,1e-5,1e-6,1e-7]:
                    if not A.admissible(x+eps*v):continue
                    ff,mm,ee,ss=score(x+eps*v,D+eps*direction[-1]/wd)
                    fd_error=float(np.linalg.norm((ff-f)/eps+derivative)/np.linalg.norm(derivative))
                    checks.append(dict(epsilon=eps,relative_error=fd_error));write(out/'bordered_derivative_check.json',checks)
                    if fd_error<1e-3:break
                assert fd_error<1e-3
            factor=1./(-np.expm1(-meta['period_ms']/float(e.transport.consts[6].get())))
            def right(v):
                u=v.copy();state=u[:-1].reshape(-1,e.s.P);state[4]*=factor
                u[:-1]-=A.normal*float(A.normal@u[:-1]);return u
            # krylov's generic interface builds I-Q; Q=v-linear(v) gives
            # precisely the bordered physical periodicity operator above.
            propose,progress=krylov(lambda v:v-linear(v),f,a.krylov,step,e.s.P,factor,
                right_override=right,operator_description='Bordered [I-DP,-P_D;T_X,T_D] periodicity equation with physical D scaling; not a monodromy matrix')
            trials=[];accepted=False
            for attempt in range(10):
                delta,small=propose(radius);dd=D+delta[-1]/wd;diagnostic={}
                xx=retract(A,x,delta[:-1],1.,project_zero=True,project_all=True,diagnostics=diagnostic)
                if not a.lower_D<dd<a.upper_D or xx is None:
                    trials.append(dict(status='OUTSIDE_PHYSICAL_PROPOSAL',D_A=float(dd),diagnostic=diagnostic,**small));radius*=.5;continue
                effective=np.r_[xx-x,wd*(dd-D)]
                if np.linalg.norm(effective)>1.05*radius:
                    trials.append(dict(status='PROJECTION_EXCEEDS_TRUST_RADIUS',**small));radius*=.5;continue
                if sum(t['status'] in ['EVALUATED','SECTION_WINDOW_MISS'] for t in trials)>=5:break
                try:ff,mm,ee,ss=score(xx,dd)
                except RuntimeError as exc:
                    trials.append(dict(status='SECTION_WINDOW_MISS',error=str(exc),**small));radius*=.5;continue
                pred=f-linear(effective);gain=1-float(ff@ff)/(norm*norm);pred_gain=1-float(pred@pred)/(norm*norm)
                agreement=gain/pred_gain if pred_gain>0 else -np.inf
                trial=dict(status='EVALUATED',D_A=float(dd),period_ms=mm['period_ms'],combined_relative_rms=ee['combined_relative_rms'],
                    residual_ratio=float(np.linalg.norm(ff)/norm),predicted_gain=pred_gain,agreement=agreement,diagnostic=diagnostic,**small)
                trials.append(trial);write(step/'trials.json',trials);log('FAST PERIOD FAMILY TRIAL',trial)
                if gain>0 and agreement>.1:
                    x=xx;D=float(dd);A.period=mm['period_ms'];field(D);accepted=True
                    np.savez_compressed(step/'accepted_state.npz',**A.state(x));write(step/'accepted_coordinates.json',dict(D_A=D,period_ms=A.period))
                    if agreement>.75:radius=min(2*radius,.2*np.linalg.norm(x))
                    break
                radius*=.5
            row.update(trials=trials,linear_last=progress[-1]);write(out/'iterations.json',rows)
            if not accepted:status='NO_ACCEPTED_BORDERED_STEP_NOT_A_BIFURCATION';break
        write(out/'result.json',dict(status=status,D_A=rows[-1]['D_A'],period_ms=rows[-1]['period_ms'],dt_ms=.05,
            iterations=rows,seconds=time.time()-begin,physical_Floquet='NOT_ESTABLISHED',model_promoted=False))
        jobs.update(status='COMPLETE',scientific_status=status);write(out/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(out/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--destination',required=True)
    p.add_argument('--period',type=float,required=True);p.add_argument('--device',type=int,default=0)
    p.add_argument('--krylov',type=int,default=80);p.add_argument('--iterations',type=int,default=8)
    p.add_argument('--radius',type=float,default=.04);p.add_argument('--lower-D',type=float,default=.30);p.add_argument('--upper-D',type=float,default=.405)
    main(p.parse_args())
