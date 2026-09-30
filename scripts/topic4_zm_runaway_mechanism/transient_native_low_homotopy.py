"""Track the verified low root along observed native spatial Z fields.

Natural-parameter stepping is a numerical accessibility diagnostic. Failure
does not certify a fold; no stability or native-onset label is emitted.
"""
from common import OUT, np, read, write, log
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from datetime import datetime
import os

DEST=OUT/'transient_native_low_homotopy_20260923'


def main():
    previous=read(OUT/'transient_conditional_root_probe_20260923/result.json')
    assert previous['status']=='COMPLETE' and all(x['status']=='ROOT_FAIL' for x in previous['rows'])
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='Can the numerically verified Z1 low equilibrium be tracked continuously to native preentry/entry fields, avoiding the failed large-jump initial guesses?',
        difference_from_failed_roots='Use the previous converged equilibrium as the next seed and reduce only the Z-path step after failure. This changes the numerical route, not equations or parameters.',
        model='Current corrected private-Q conditional drift; constant original mean input, full spatial Z prescribed, M dynamic. Transient correction and its derivative vanish at equilibrium.',
        parameter='Original native observation time indexes a frozen spatial Z family; it is NOT integration time. Piecewise linear interpolation of the actual5/10ms native spatial fields. D is its cell-weighted display coordinate.',
        initial='Independently flow-verified Z1 low equilibrium.',
        budget='At most40 trial roots, each at most25 direct Newton iterations; initial path step250ms, maximum500ms, halve on failure until below2ms, terminate at9870ms. Every trial retained. No root-count, critical-type assignment or response fitting.',
        acceptance='Each accepted point original residual<1e-11perms, physical positive rates. Natural continuation stops do not establish SN or other bifurcation.',
        scope='Conditional mathematical seed construction only; autonomous Z/native correspondence failures retained; model not promoted.'))
    s=PhysicalDelayConditionalDrift();data=np.load(OUT/'transient_native_Z_path_20260923/native_Z_path.npz')
    times=data['time_ms'];fields=data['Z']
    def field(t):
        lo=int(np.clip(np.searchsorted(times,t,side='right')-1,0,len(times)-2));a=(t-times[lo])/(times[lo+1]-times[lo])
        return (1-a)*fields[lo]+a*fields[lo+1]
    r=np.load(OUT/'physical_delay_conditional_drift_interface/baseline_equilibrium.npz')['r']
    t=0.;step=250.;s.set_Z(field(t));assert abs(s.residual(r)).max()<1e-11
    rows=[];trials=[];jobs=dict(status='RUNNING',pid=os.getpid(),accepted_native_time_ms=t)
    write(DEST/'jobs.json',jobs)
    def save(t,r):
        row=dict(index=len(rows),native_Z_time_ms=t,D=s.D,global_rate_hz=s.global_rate(r),regional_rates_hz=s.regional_rates(r),
            residual_per_ms=float(abs(s.residual(r)).max()),stability='NOT_COMPUTED')
        np.savez_compressed(DEST/f'point{len(rows):03d}.npz',r=r,Z=s.Z,native_Z_time_ms=t,D=s.D)
        rows.append(row);write(DEST/'accepted.json',rows);log('NATIVE LOW ACCEPT',row)
    save(t,r)
    reason='TRIAL_BUDGET'
    for n in range(40):
        target=min(9870.,t+step);s.set_Z(field(target))
        candidate,ok,trace=s.solve(r,tol=1e-11,maxiter=25)
        err=float(abs(s.residual(candidate)).max())
        trial=dict(index=n,start_time_ms=t,target_time_ms=target,step_ms=step,converged=bool(ok),residual_per_ms=err,trace=trace)
        np.savez_compressed(DEST/f'trial{n:03d}.npz',r=candidate,Z=s.Z,initial_guess=r)
        trials.append(trial);write(DEST/'trials.json',trials);log('NATIVE LOW TRIAL',n,t,target,ok,err)
        if ok:
            r=candidate;t=target;save(t,r);step=min(500.,step*1.5)
            jobs.update(accepted_native_time_ms=t,accepted_D=s.D);write(DEST/'jobs.json',jobs)
            if t==9870.:reason='REACHED_TARGET';break
        else:
            step*=.5
            if step<2.:reason='MINIMUM_PATH_STEP';break
    result=dict(status='COMPLETE',stop_reason=reason,rows=rows,trials=len(trials),
        interpretation='Accepted static points only, stability uncomputed. A natural-parameter solver stop is not a certified bifurcation.',
        model_promoted=False,bifurcation_type='NOT_ESTABLISHED')
    write(DEST/'result.json',result);jobs.update(status='COMPLETE',stop_reason=reason);write(DEST/'jobs.json',jobs)


if __name__=='__main__':main()
