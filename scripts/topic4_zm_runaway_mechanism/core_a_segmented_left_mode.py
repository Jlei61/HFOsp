"""Verify a numerical left/right mode for a closed full-state cycle.

This is a numerical section-map diagnostic. Existing continuous-flow phase
and mesh failures are retained; no physical Floquet or onset certificate.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build
from onset_cubic_section import CubicSectionReturn
from onset_segmented_poincare import cycle_coordinates,SegmentedPoincareDerivative
from onset_segmented_adjoint import SegmentedPoincareAdjoint
from onset_period_return import errors,dynamical_state
from fine_rate_frozen_Z_fields import capture
from pathlib import Path
import argparse,os,time


def main(a):
    parent=Path(a.parent).resolve();result=read(parent/'result.json');contract=read(parent/'contract.json')
    assert result['status'] in ['NUMERICAL_SEGMENTED_SINGLE_SHOOTING_ROOT','NUMERICAL_MULTIPLE_SHOOTING_ROOT']
    spec=parent/'section_spectrum_segmented';sr=read(spec/'result.json')
    candidates=[(i,v) for i,v in enumerate(sr['verified']) if v['status']=='VERIFIED_NUMERICAL_EIGENPAIR' and v['imag']==0 and v['real']<0]
    assert len(candidates)==1
    index,entry=candidates[0];mu=entry['real'];assert abs(mu)>.8
    out=parent/a.name;out.mkdir(exist_ok=True);assert not(out/'jobs.json').exists()
    write(out/'contract.json',dict(source=str(parent),mode_source=str(spec/f'mode{index:02d}.npz'),dt_ms=contract['dt_ms'],
        question='Construct a verified local scalar coordinate for the numerical negative return mode before nonlinear daughter-branch tests.',
        method='Exact transpose of every cached full fixed-time segment, in reverse order, after the transpose implicit return-time projection. Full physical state retained. Two bilinear identities and unchanged nominal state precede bounded left power iteration.',
        maximum_iterations=a.iterations,segments=a.segments,prior_physical_Floquet=sr['physical_Floquet'],
        scope='Numerical Poincare map only. Prior phase/mesh failures retained. This is not a separatrix, physical Floquet certificate, bifurcation label or native onset attribution.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs);start=time.time()
    try:
        row=result['iterations'][-1];T=row['period_ms']
        state=dict(np.load(parent/f"iteration{row['iteration']:02d}"/'node00.npz'))
        e=build(a.device,contract['dt_ms']);A=CubicSectionReturn(state,e,T,5.)
        cycle_coordinates(A,T,a.segments);x=A.xref;y,meta=A(x)
        closure=errors(dynamical_state(A.state(x)),dynamical_state(A.state(y)),e.s.sizes/e.s.sizes.sum())
        write(out/'initial_closure.json',dict(return_meta=meta,**closure));log('LEFT MODE INITIAL CLOSURE',meta['period_ms'],closure['combined_relative_rms'])
        assert closure['combined_relative_rms']<1e-6,closure
        J=SegmentedPoincareDerivative(A,x,meta['period_ms'],A.last_time_slope,a.segments)
        JT=SegmentedPoincareAdjoint(J);write(out/'partition_check.json',J.partition_check)
        rng=np.random.default_rng(925901);checks=[]
        for k in range(2):
            v=rng.normal(size=x.size);w=rng.normal(size=x.size)
            for q in [v,w]:q.reshape(-1,e.s.P)[4,~e.s.E]=0;q/=np.linalg.norm(q)
            jv=J(v);before=capture(e);jtw=JT(w);after=capture(e)
            assert all(np.array_equal(value,after[key]) for key,value in before.items())
            lhs=float(w@jv);rhs=float(v@jtw)
            err=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-12)
            checks.append(dict(direction=k,relative_bilinear_error=err,nominal_state_bitwise_unchanged=True))
            write(out/'bilinear_checks.json',checks);log('SEGMENTED ADJOINT CHECK',checks[-1]);assert err<1e-8
        mode=np.load(spec/f'mode{index:02d}.npz')
        u=(mode['vector'].real.reshape(-1,e.s.P)*mode['coordinate_scale']/mode['coordinate_weight']*A.c.weight/A.c.scale).ravel()
        u/=np.linalg.norm(u);right_error=float(np.linalg.norm(J(u)-mu*u)/max(1.,abs(mu)));assert right_error<1e-6
        w=rng.normal(size=x.size);w.reshape(-1,e.s.P)[4,~e.s.E]=0;w/=np.linalg.norm(w);rows=[]
        for k in range(a.iterations):
            q=JT(w);err=float(np.linalg.norm(q-mu*w)/max(1.,abs(mu)))
            rows.append(dict(iteration=k,relative_left_residual=err,rayleigh=float(w@q)))
            write(out/'left_progress.json',rows);log('SEGMENTED LEFT MODE',rows[-1])
            if err<1e-8:break
            w=q/np.linalg.norm(q)
        passed=err<1e-8
        if passed:
            overlap=float(w@u);assert abs(overlap)>1e-10;w/=overlap
            # Recheck after normalization in the actual full operator.
            left_error=float(np.linalg.norm(JT(w)-mu*w)/(max(1.,abs(mu))*np.linalg.norm(w)))
            assert left_error<1e-8
            np.savez_compressed(out/'local_coordinate.npz',left=w,right=u,multiplier=mu,
                coordinate_scale=A.c.scale,coordinate_weight=A.c.weight,base_state_coordinate=x,section_normal=A.normal)
        else:overlap=None;left_error=err
        write(out/'result.json',dict(status='VERIFIED_NUMERICAL_LEFT_RIGHT_MODE' if passed else 'LEFT_MODE_NOT_CONVERGED',
            multiplier=mu,right_relative_residual=right_error,left_relative_residual=left_error,raw_overlap=overlap,
            checks=checks,closure=closure,seconds=time.time()-start,prior_physical_Floquet=sr['physical_Floquet'],
            bifurcation_type='NOT_ESTABLISHED',model_promoted=False))
        jobs.update(status='COMPLETE');write(out/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(out/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('--device',type=int,default=1)
    p.add_argument('--segments',type=int,default=6);p.add_argument('--iterations',type=int,default=24)
    p.add_argument('--name',default='segmented_left_mode')
    main(p.parse_args())
