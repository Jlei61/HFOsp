"""Nonlinear criticality of the full numerical Poincare map near mu=-1.

Only the original spatial return map is evaluated. No raw cubic fit along a
mode is substituted for the quadratic response of all noncritical states.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build
from onset_cubic_section import CubicSectionReturn
from onset_segmented_poincare import cycle_coordinates,SegmentedPoincareDerivative
from onset_segmented_adjoint import SegmentedPoincareAdjoint
from onset_flip_normal_form import cubic_from_curved_pair
from core_a_periodic_hookstep import krylov
from core_a_flip_locator import root_state,negative_mode
from onset_period_return import errors,dynamical_state
from fine_rate_frozen_Z_fields import capture
from pathlib import Path
import argparse,os,time


def main(a):
    parent=Path(a.parent).resolve();source,T=root_state(parent);entry,modepath=negative_mode(parent)
    mu=entry['real'];assert abs(mu+1)<2e-4,'Locate numerical -1 before calling this a criticality diagnostic'
    assert read(OUT/'core_a_bifurcation_type_20260924/numerical_checks/flip_map_normal_form/result.json')['status']=='PASS'
    out=parent/'flip_cubic_coefficient';out.mkdir(exist_ok=True);assert not(out/'jobs.json').exists()
    contract=read(parent/'contract.json');steps=[a.amplitude,a.amplitude/2,a.amplitude/4]
    write(out/'contract.json',dict(source=str(parent),dt_ms=contract['dt_ms'],multiplier=mu,quadratic_amplitudes=steps,
        curved_amplitude_rule='Use min(initial_amplitude,0.05/norm(h2)) and two successive halvings, limiting the quadratic correction to2.5% of the linear displacement. This is a priori numerical sample-size selection, not a sign-dependent coefficient choice.',
        question='Is the numerical -1 crossing nondegenerate, and on which side does its doubled cycle locally emerge?',
        method='Compute full original Poincare J and its exact transpose; verify a left/right eigenpair. Estimate B(q,q) by centered full-state returns at three amplitudes. Solve (mu^2 I-J)h2=B(q,q) in the complete history state. Evaluate the odd cubic coefficient along x +/- hq + h^2*h2/2 and extrapolate the two smaller amplitudes.',
        formula='At mu=-1, c = p*C(q,q,q)/6 + p*B(q,(I-J)^-1 B(q,q))/2, with p*q=1. Map normal-form algebra, not an ODE-vector-field Lyapunov coefficient.',
        algebra_check='12 independent two-variable analytic maps with stable-variable quadratic feedback and nonorthogonal coordinate transforms.',
        gates='Original full derivative parity; two adjoint bilinear identities <1e-8; right residual<1e-6; left residual<1e-8; quadratic difference convergence<1%; h2 full linear residual<1e-6; cubic last-pair variation<5%, common sign and extrapolated magnitude>10 times extrapolation change. No clipping of any return-map input.',
        maximum_left_iterations=24,maximum_linear_Krylov=32,
        physical_scope='AllZheld, allE Mdynamic, same full locked spatial delayed rate. Existing phase/mesh failures retained. Numerical map criticality is not continuous-model qualification or onset attribution.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs);start=time.time()
    try:
        e=build(a.device,contract['dt_ms']);base=dict(np.load(source));A=CubicSectionReturn(base,e,T,5.)
        cycle_coordinates(A,T,6);x=A.xref;y0,meta=A(x)
        closure=errors(dynamical_state(A.state(x)),dynamical_state(A.state(y0)),e.s.sizes/e.s.sizes.sum())
        write(out/'initial_closure.json',closure);assert closure['combined_relative_rms']<1e-6
        J=SegmentedPoincareDerivative(A,x,meta['period_ms'],A.last_time_slope,6);JT=SegmentedPoincareAdjoint(J)
        write(out/'partition_check.json',J.partition_check)
        rng=np.random.default_rng(925953);bilinear=[]
        for k in range(2):
            v=rng.normal(size=x.size);w=rng.normal(size=x.size)
            for z in [v,w]:z.reshape(-1,e.s.P)[4,~e.s.E]=0;z/=np.linalg.norm(z)
            jv=J(v);before=capture(e);jtw=JT(w);after=capture(e)
            assert all(np.array_equal(val,after[key]) for key,val in before.items())
            lhs=float(w@jv);rhs=float(v@jtw);err=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-12)
            bilinear.append(err);write(out/'bilinear_checks.json',bilinear);assert err<1e-8
        m=np.load(modepath);q=(m['vector'].real.reshape(-1,e.s.P)*m['coordinate_scale']/m['coordinate_weight']*A.c.weight/A.c.scale).ravel()
        q/=np.linalg.norm(q);linear=J(q);right_error=float(np.linalg.norm(linear-mu*q)/max(1.,abs(mu)));assert right_error<1e-6
        if a.left_seed:
            z=np.load(a.left_seed)
            w=(z['left'].reshape(-1,e.s.P)*z['coordinate_weight']/z['coordinate_scale']*A.c.scale/A.c.weight).ravel()
        else:w=rng.normal(size=x.size)
        w.reshape(-1,e.s.P)[4,~e.s.E]=0;w/=np.linalg.norm(w);left=[]
        for k in range(24):
            v=JT(w);err=float(np.linalg.norm(v-mu*w)/max(1.,abs(mu)))
            left.append(dict(iteration=k,relative_residual=err));write(out/'left_progress.json',left);log('FLIP LEFT MODE',left[-1])
            if err<1e-8:break
            w=v/np.linalg.norm(v)
        assert err<1e-8,'Left eigenmode not resolved'
        overlap=float(w@q);assert abs(overlap)>1e-10;p=w/overlap
        B=[];second=[]
        for h in steps:
            for initial in [x+h*q,x-h*q]:assert A.admissible(initial),'No projected derivative inputs'
            plus,_=A(x+h*q);minus,_=A(x-h*q);b=(plus-2*y0+minus)/(h*h);B.append(b)
            second.append(dict(amplitude=h,norm=float(np.linalg.norm(b)),dual_projection=float(p@b)))
            write(out/'second_derivative_progress.json',second);log('FLIP QUADRATIC RESPONSE',second[-1])
        diff=float(np.linalg.norm(B[-1]-B[-2])/max(np.linalg.norm(B[-1]),1e-12))
        write(out/'quadratic_convergence.json',dict(last_pair_relative_change=diff));assert diff<.01
        b=(4*B[-1]-B[-2])/3;mu2=mu*mu
        factor=1/(1-np.exp(-meta['period_ms']/float(e.transport.consts[6].get()))/mu2)
        folder=out/'quadratic_linear_solve';folder.mkdir()
        propose,progress=krylov(lambda v:J(v)/mu2,b/mu2,32,folder,e.s.P,factor,
            linear_tolerance=1e-7,operator_description='(mu^2 I-DP)h2=B(q,q), noncritical-state quadratic response, not Floquet')
        h2,small=propose(1e10)
        linear_error=float(np.linalg.norm(mu2*h2-J(h2)-b)/np.linalg.norm(b))
        write(out/'quadratic_solve_check.json',dict(full_relative_residual=linear_error,phase_component=float(A.normal@h2),norm=float(np.linalg.norm(h2))))
        assert linear_error<1e-6
        curved_start=min(a.amplitude,.05/max(np.linalg.norm(h2),1e-15))
        curved_steps=[curved_start,curved_start/2,curved_start/4]
        write(out/'curved_amplitudes.json',dict(amplitudes=curved_steps,quadratic_to_linear_ratio_at_largest=.5*curved_start*float(np.linalg.norm(h2))))
        coefficients=[]
        for h in curved_steps:
            u=x+h*q+.5*h*h*h2;v=x-h*q+.5*h*h*h2
            assert A.admissible(u) and A.admissible(v),'No retraction of curved normal-form samples'
            plus,_=A(u);minus,_=A(v);c=cubic_from_curved_pair(plus,minus,h,linear,p)
            coefficients.append(dict(amplitude=h,coefficient=c));write(out/'cubic_progress.json',coefficients);log('FLIP CUBIC COEFFICIENT',coefficients[-1])
        values=[r['coefficient'] for r in coefficients];c=(4*values[-1]-values[-2])/3
        uncertainty=abs(c-values[-1]);variation=abs(values[-1]-values[-2])/max(abs(c),1e-12)
        passed=variation<.05 and min(values)*max(values)>0 and abs(c)>10*uncertainty
        np.savez_compressed(out/'normal_form_directions.npz',right=q,left=p,quadratic=h2,multiplier=mu,coefficient=c,
            coordinate_scale=A.c.scale,coordinate_weight=A.c.weight,section_normal=A.normal)
        write(out/'result.json',dict(status='NUMERICAL_MAP_CUBIC_SIGN_RESOLVED' if passed else 'CUBIC_COEFFICIENT_NOT_CONVERGED',
            multiplier=mu,right_residual=right_error,left_residual=err,quadratic_relative_change=diff,quadratic_solve_residual=linear_error,
            estimates=coefficients,extrapolated_coefficient=c,last_pair_relative_variation=variation,extrapolation_change=uncertainty,
            numerical_local_interpretation=('supercritical flip coefficient' if c>0 else 'subcritical flip coefficient') if passed else 'UNRESOLVED',
            physical_bifurcation_type='NOT_ESTABLISHED',onset_attribution='NOT_ESTABLISHED',seconds=time.time()-start,model_promoted=False))
        jobs.update(status='COMPLETE');write(out/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(out/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('--device',type=int,default=1)
    p.add_argument('--amplitude',type=float,default=.0005);p.add_argument('--left-seed')
    main(p.parse_args())
