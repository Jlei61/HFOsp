"""Two directly onset-relevant stationary seed tests, original spatial model.

Trajectory means are numerical guesses only. No frozen-M replacement or
equilibrium bifurcation claim follows from this screen.
"""
from common import OUT,np,read,write,log
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
import core_a_static_candidates as root
import os

BASE=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap'
DEST=BASE/'entry_stationary_screen';root.DEST=DEST


def main():
    DEST.mkdir(exist_ok=True);assert not (DEST/'jobs.json').exists()
    cases=[('short_midpoint','actual_native_entry_midpoint',None),('first_long_D0300','actual_D0300','first_long')]
    write(DEST/'contract.json',dict(
        question='Do the actual short-event field and the first seconds-long local-activity field have a stationary spatial counterpart that could nominate a relevant equilibrium mechanism, instead of the already excluded later surround fold?',
        equations='Same3479-group physical private-Q static equations, original graph, locked local response and nativeA spatialfields. EveryM obeys its stationary dynamic-law constraint M=.5*E*r; M is not clamped to an observed value. Locked transient correction and its first derivative vanish at stationary input.',
        cases=cases,seed_selection='Short field: actual last5s mean of every group. Long field: mean of every group in the1s window ending200ms before the first complete >=1s Core-A activity ends, selected from the unchanged5Hz/20ms audit.',
        numerical='Existing logit full-space Newton/Armijo/LSMR method, atmost70iterations percase and residual-stagnation stop. Numerical seed floor only. Original physical residual<1e-11/ms required.',
        scope='A trajectory mean is not a root. Failure is not nonexistence. A root remains a stationary candidate pending original-engine, temporal stability and actual-onset correspondence. No continuation or bifurcation label automatically follows.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid(),completed=[]);write(DEST/'jobs.json',jobs)
    s=PhysicalDelayConditionalDrift();rows=[]
    try:
        for label,name,selection in cases:
            source=BASE/name/'from_interictal_history';audit=read(source/'whole_record_audit.json')
            assert audit['status']=='AUDIT_PASS'
            if selection:
                events=next(r for r in audit['thresholds'] if r['threshold_hz']==5.)['activities']
                event=next(r for r in events if r['duration_ms']>=1000 and not r['left_censored'] and not r['right_censored'])
                end=event['end_ms']-200;start=end-1000;assert start>=event['start_ms']
            else:start=audit['observed_ms']-5000;end=audit['observed_ms']
            rates=[];Z=None
            for j in read(source/'jobs.json')['completed_blocks']:
                with np.load(source/f'block{j:02d}.npz') as data:
                    time=data['elapsed_time_ms'];mask=(time>start)&(time<=end)
                    if mask.any():rates.append(data['group_rate_hz'][mask].astype(float))
                    if Z is None:Z=data['Z'].copy()
                    assert np.array_equal(Z,data['Z'])
            initial=np.concatenate(rates).mean(0)/1000;s.set_Z(Z)
            write(DEST/f'{label}_source.json',dict(source=str(source),coordinates=audit['coordinates'],mean_window_ms=[start,end],equilibrium_M_law=True))
            row=root.solve(s,initial,label);rows.append(row);jobs['completed'].append(label);write(DEST/'jobs.json',jobs)
        write(DEST/'result.json',dict(status='COMPLETE',rows=rows,bifurcation_type='NOT_ESTABLISHED',model_promoted=False))
        jobs['status']='COMPLETE';write(DEST/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(DEST/'jobs.json',jobs);raise


if __name__=='__main__':main()
