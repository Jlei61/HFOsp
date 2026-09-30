"""Prepare a neighboring-field or refined-mesh period guess from one root.

The full starting state is transplanted unchanged except for the explicitly
selected Core A Z field and optional physical-lag mesh interpolation.
"""
from common import np,read,write,log
from onset_state_continuation import build,regrid_state
from onset_cubic_section import CubicSectionReturn
from onset_period_return import errors,dynamical_state
from core_a_parameter_path_audit import NativeTimeFamily
from pathlib import Path
import argparse,os,time


def main(a):
    parent=Path(a.parent).resolve();out=Path(a.destination).resolve();out.mkdir(parents=True,exist_ok=True)
    assert not (out/'jobs.json').exists()
    result=read(parent/'result.json');contract=read(parent/'contract.json')
    assert result['status'] in ['NUMERICAL_MULTIPLE_SHOOTING_ROOT','NUMERICAL_SEGMENTED_SINGLE_SHOOTING_ROOT']
    row=result['iterations'][-1];T=row['period_ms'];source_dt=contract['dt_ms']
    dt=a.dt if a.dt is not None else source_dt
    source=parent/f"iteration{row['iteration']:02d}"/'node00.npz'
    write(out/'contract.json',dict(source=str(source),source_dt_ms=source_dt,dt_ms=dt,
        old_period_ms=T,target_native_time_ms=a.target_native_time,search_halfwidth_ms=a.halfwidth,
        question='Follow the actual three-burst return toward the prolonged-activity field, or refine that same full-state orbit. Is a nearby full-state return available as a phase-aligned Newton starting guess?',
        method='One original cubic Poincare return near the old period. Preserve every initial M, synapse, covariance and input-memory state; any finer delay history uses the existing physical-lag refinement. Optional parameter change selects only native within-CoreA Z on the continuous recorded-time path, outsideCoreA remains bitwise unchanged.',
        limits='This prepares a numerical guess only. A return is not a closed orbit or proof that the same branch survives. Subsequent whole-state Newton residual, subperiod, independent-flow, stability and mesh checks remain separate. Failed section search is not a bifurcation or nonexistence proof.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs);started=time.time()
    try:
        e=build(a.device,dt);original=dict(np.load(source));base=regrid_state(original,e,source_dt)
        A_mask=e.s.E&(e.s.geo['group_region']==0)
        if a.target_native_time is not None:
            family=NativeTimeFamily(e.s);Z,_=family.field_at_time(a.target_native_time)
            assert np.array_equal(base['syn'][5,~A_mask],Z[~A_mask]);base['syn'][5]=Z.copy()
        assert np.array_equal(base['syn'][:5],original['syn'][:5])
        assert np.array_equal(base['local'],original['local'])
        assert np.all(base['parameters'][19]==0) and np.all(base['parameters'][20]==1)
        coordinate=dict(native_time_ms=a.target_native_time,
            D_A=float(1-np.average(base['syn'][5,A_mask],weights=e.s.sizes[A_mask])))
        np.savez_compressed(out/'starting_state.npz',**base)
        write(out/'coordinates.json',coordinate)
        A=CubicSectionReturn(base,e,T,a.halfwidth);assert A.admissible(A.xref)
        y,meta=A(A.xref);returned=A.state(y)
        check=errors(dynamical_state(base),dynamical_state(returned),e.s.sizes/e.s.sizes.sum())
        np.savez_compressed(out/'returned_state.npz',**returned)
        write(out/'result.json',dict(status='PERIODIC_INITIAL_GUESS_NOT_A_ROOT',**coordinate,
            **meta,**check,old_period_ms=T,period_shift_ms=meta['period_ms']-T,
            seconds=time.time()-started,model_promoted=False))
        jobs.update(status='COMPLETE');write(out/'jobs.json',jobs);log('PERIODIC SEED ALIGNMENT',coordinate,meta,check)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(out/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('--destination',required=True)
    p.add_argument('--device',type=int,default=1);p.add_argument('--dt',type=float)
    p.add_argument('--target-native-time',type=float);p.add_argument('--halfwidth',type=float,default=50.)
    main(p.parse_args())
