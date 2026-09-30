"""Check relative-time bins against original per-step flux, without feedback."""
from common import OUT,np,write,log
from onset_state_continuation import build
from onset_relative_rate_recorder import RelativeRateRecorder
from onset_segment_flow import split_step
from onset_period_return import dynamical_state,errors
from fine_rate_frozen_Z_fields import restore,capture
import os


def main():
    root=OUT/'core_a_bifurcation_type_20260924'
    out=root/'numerical_checks/relative_rate_recorder';out.mkdir(exist_ok=True)
    assert not (out/'jobs.json').exists();write(out/'jobs.json',dict(status='RUNNING',pid=os.getpid()))
    source=root/'reference_stability_gap/actual_D0270/recurrence/nine_burst_fixed_grid_seed/node00.npz'
    base=dict(np.load(source));e=build(0);restore(e,base);native=e.chunk();terminal=capture(e)
    restore(e,base);R=RelativeRateRecorder(e);relative=R.chunk();repeated=capture(e)
    weights=e.s.sizes/e.s.sizes.sum();parity=errors(dynamical_state(terminal),dynamical_state(repeated),weights)
    assert parity['combined_relative_rms']<1e-12 and max(v['relative_rms'] for v in parity['blocks'].values())<1e-11
    assert np.array_equal(terminal['clock'],repeated['clock']) and np.array_equal(terminal['parameters'],repeated['parameters'])
    restore(e,base);manual=[];acc=np.zeros((2,e.s.P))
    for j in range(round(10/e.dt)):
        e.step();e.cp.cuda.get_current_stream().synchronize()
        acc+=np.array([e.emitted.get(),e.local.rate.get()])*e.dt
        if (j+1)%round(1/e.dt)==0:manual.append(acc.copy()*1000);acc.fill(0)
    manual=np.array(manual);bin_error=float(np.max(abs(relative-manual)))
    assert np.all(abs(relative-manual)<=1e-10+1e-12*abs(manual))
    T=10.137;restore(e,base);rec=RelativeRateRecorder(e);rate,m,mean=rec.read_period(T)
    end=capture(e);restore(e,base);n,a=split_step(T,e.dt);integral=np.zeros((2,e.s.P))
    for j in range(n+1):
        e.step();e.cp.cuda.get_current_stream().synchronize()
        integral+=(1. if j<n else a)*e.dt*np.array([e.emitted.get(),e.local.rate.get()])
    expected_mean=1000*integral/T;mean_error=float(np.max(abs(mean-expected_mean)))
    assert np.all(abs(mean-expected_mean)<=1e-10+1e-12*abs(expected_mean))
    endpoint=errors(dynamical_state(end),dynamical_state(capture(e)),weights)
    assert endpoint['combined_relative_rms']<1e-12
    assert len(rate)==10 and np.all(abs(rate-manual[:,0])<=1e-10+1e-12*abs(manual[:,0]))
    result=dict(status='PASS',source=str(source),dt_ms=e.dt,absolute_start_time_ms=float(base['clock'][0]*e.dt),
        original_chunk_physical_parity=parity,original_chunk_all_buffers_bitwise=all(np.array_equal(v,repeated[k]) for k,v in terminal.items()),
        relative_bins_max_absolute_hz_error=bin_error,native_absolute_bins_max_difference_hz=float(np.max(abs(native-relative))),
        exact_T_ms=T,mean_flux_max_absolute_hz_error=mean_error,endpoint_parity=endpoint,
        interpretation='Relative1ms bins and exactT per-step flux quadrature match independent original stepping. No feedback into physical dynamics. The original absolute-clock output cannot be concatenated as relative time from this fractional-ms source.',model_promoted=False)
    np.savez_compressed(out/'readout_comparison.npz',native_absolute_clock=native,relative_clock=relative,manual_relative=manual,mean=mean,manual_mean=expected_mean)
    write(out/'result.json',result);write(out/'jobs.json',dict(status='COMPLETE',pid=os.getpid()));log('RELATIVE RATE RECORDER',result)


if __name__=='__main__':main()
