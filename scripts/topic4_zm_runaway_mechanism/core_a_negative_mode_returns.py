"""Original nonlinear section returns along a verified negative mode.

The returned states are full delayed states, not a fitted scalar map. This
bounded calculation nominates daughter-cycle seeds; numerical return-map
behavior is not promoted to a continuous-model or onset certificate.
"""
from common import np,read,write,log
from onset_state_continuation import build
from onset_cubic_section import CubicSectionReturn
from onset_period_return import errors,dynamical_state
from fine_rate_frozen_Z_fields import capture
from pathlib import Path
import argparse,os,time


def main(a):
    parent=Path(a.parent).resolve();result=read(parent/'result.json');contract=read(parent/'contract.json')
    assert result['status'] in ['NUMERICAL_SEGMENTED_SINGLE_SHOOTING_ROOT','NUMERICAL_PERIODIC_ROOT']
    assert 0<=a.returns<=50
    numerical_method=contract.get('numerical_method','old_endpoint')
    spec=parent/'section_spectrum_segmented';sr=read(spec/'result.json')
    candidates=[(i,v) for i,v in enumerate(sr['verified']) if v['status']=='VERIFIED_NUMERICAL_EIGENPAIR' and v['imag']==0 and v['real']<-1]
    assert len(candidates)==1;index,entry=candidates[0];mu=entry['real']
    out=parent/a.name;out.mkdir(exist_ok=True);assert not(out/'jobs.json').exists()
    write(out/'contract.json',dict(source=str(parent),mode_source=str(spec/f'mode{index:02d}.npz'),
        dt_ms=contract['dt_ms'],amplitude_in_saved_invertible_coordinates=a.amplitude,
        numerical_method=numerical_method,
        question='Does the verified negative numerical mode grow by alternating sign in the original nonlinear return, and does it nominate a nontrivial doubled-period state?',
        method='Use the original cubic full-state Poincare map with the exact saved spectrum coordinates and phase plane. Test both signs and half-amplitudes against the full verified eigenpair. Then follow both signed initial perturbations for a bounded number of returns. No scalar-map fit, physical-state clipping, parameter change, M clamp, or model change.',
        number_of_returns_each_sign=a.returns,halfwidth_ms=20.,
        acceptance='Initial physical states must be admissible without any retraction. Both signed half-amplitude nonlinear derivative errors must be below1%. Every subsequent input remains admissible. One-/two-return full-state residuals only nominate roots for subsequent correction; they do not certify them.',
        physical_Floquet=sr['physical_Floquet'],scope='Numerical return map only, phase and mesh qualification pending. Alternating growth does not itself establish a nonlinear criticality or onset mechanism.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs);start=time.time()
    try:
        row=result['iterations'][-1];T=row['period_ms']
        if numerical_method=='exponential_midpoint':
            from onset_exponential_midpoint import ExponentialMidpointEngine
            e=ExponentialMidpointEngine(dt=contract['dt_ms'],device=a.device);e.graph()
        else:
            assert numerical_method=='old_endpoint'
            e=build(a.device,contract['dt_ms'])
        source=(parent/f"iteration{row['iteration']:02d}"/'node00.npz' if result['status']=='NUMERICAL_SEGMENTED_SINGLE_SHOOTING_ROOT'
                else parent/('root_state.npz' if (parent/'root_state.npz').exists() else 'latest_state.npz'))
        base=dict(np.load(source))
        A=CubicSectionReturn(base,e,T,20.);m=np.load(spec/f'mode{index:02d}.npz')
        assert np.array_equal(A.c.weight,m['coordinate_weight'])
        A.c.scale=m['coordinate_scale'].copy();A.xref=A.c.pack(base)
        # Reconstruct the original spectrum phase plane after rescaling.
        from fine_rate_frozen_Z_fields import restore
        restore(e,base);points=[A.xref]
        for _ in range(2):e.step();e.cp.cuda.get_current_stream().synchronize();points.append(A.c.pack(capture(e)))
        velocity=(-3*points[0]+4*points[1]-points[2])/(2*e.dt);A.normal=velocity/np.linalg.norm(velocity)
        x=A.xref;u=m['vector'].real.copy();u/=np.linalg.norm(u);assert abs(A.normal@u)<1e-8
        y0,meta=A(x);weights=e.s.sizes/e.s.sizes.sum()
        rooterr=errors(dynamical_state(base),dynamical_state(A.state(y0)),weights)
        write(out/'initial_closure.json',dict(return_meta=meta,**rooterr))
        assert rooterr['combined_relative_rms']<1e-6
        checks=[];first={}
        for amplitude in [a.amplitude,a.amplitude/2]:
            for sign in [-1,1]:
                initial=x+sign*amplitude*u
                assert A.admissible(initial),'No projected or clipped perturbation seeds'
                y,meta=A(initial);delta=y-y0
                error=float(np.linalg.norm(delta-sign*amplitude*mu*u)/(amplitude*max(1.,abs(mu))))
                r=dict(amplitude=amplitude,sign=sign,nonlinear_derivative_relative_error=error,
                    along_right_vector_ratio=float(u@delta/(sign*amplitude)),period_ms=meta['period_ms'])
                checks.append(r);write(out/'initial_direction_check.json',checks);log('NEGATIVE MODE NONLINEAR CHECK',r)
                np.savez_compressed(out/f'check_a{amplitude:g}_sign{sign:+d}.npz',**A.state(y))
                if amplitude==a.amplitude:first[sign]=(initial,y,meta)
        assert all(r['nonlinear_derivative_relative_error']<.01 for r in checks if r['amplitude']==a.amplitude/2),checks
        branches=[]
        for sign in [-1,1]:
            branch=out/('negative' if sign<0 else 'positive');branch.mkdir(exist_ok=True)
            initial,y,meta=first[sign];np.savez_compressed(branch/'state00.npz',**A.state(initial))
            previous=[initial];rows=[];status='RETURN_BUDGET_COMPLETE'
            for k in range(1,a.returns+1):
                if k>1:
                    try:y,meta=A(previous[-1])
                    except (RuntimeError,AssertionError) as exc:
                        status='RETURN_LEFT_LOCAL_SEARCH_OR_ADMISSIBILITY_DOMAIN';write(branch/'local_return_stop.json',dict(error=repr(exc),meaning='No bifurcation or orbit nonexistence inferred from leaving the local numerical return definition.'));break
                state=A.state(y)
                assert np.array_equal(state['syn'][5],base['syn'][5])
                assert np.all(state['parameters'][19]==0) and np.all(state['parameters'][20]==1)
                r=dict(iteration=k,period_ms=meta['period_ms'],distance_to_parent=float(np.linalg.norm(y-x)),
                    right_vector_projection=float(u@(y-x)),
                    one_return=errors(dynamical_state(A.state(previous[-1])),dynamical_state(state),weights))
                if len(previous)>=2:r['two_return']=errors(dynamical_state(A.state(previous[-2])),dynamical_state(state),weights)
                rows.append(r);write(branch/'progress.json',rows);np.savez_compressed(branch/f'state{k:02d}.npz',**state)
                log('NEGATIVE MODE RETURN',sign,k,r['period_ms'],r['distance_to_parent'],r['right_vector_projection'],r.get('two_return',{}).get('combined_relative_rms'))
                previous.append(y);previous=previous[-2:]
                jobs.update(sign=sign,return_index=k);write(out/'jobs.json',jobs)
            branches.append(dict(sign=sign,status=status,rows=rows));write(out/'branches.json',branches)
        write(out/'result.json',dict(status='BOUNDED_NONLINEAR_RETURN_EXPERIMENT_COMPLETE',multiplier=mu,
            checks=checks,branches=branches,seconds=time.time()-start,
            nonlinear_daughter_root='NOT_CERTIFIED',physical_bifurcation_type='NOT_ESTABLISHED',onset_attribution='NOT_ESTABLISHED',model_promoted=False))
        jobs.update(status='COMPLETE');write(out/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(out/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('--device',type=int,default=0)
    p.add_argument('--amplitude',type=float,default=.001);p.add_argument('--returns',type=int,default=12)
    p.add_argument('--name',default='negative_mode_returns')
    main(p.parse_args())
