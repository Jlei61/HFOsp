"""Independently compare whole stochastic trajectories and complete events."""
from common import *
from scipy.ndimage import uniform_filter1d
from native_readouts import readouts, window_stats
from closure_network_sensitivity_audit import high_entry


def main():
    dest=OUT/'closure_stochastic_sensitivity'
    assert read(dest/'result.json')['status']=='DIAGNOSTIC_COMPLETE'
    assert read(dest/'baseline_prefix_audit.json')['status']=='BITWISE_PREFIX_PASS'
    c=read(OUT/'closure_network_sensitivity_contract.json');s=model()
    native=np.load(BASE/'native_reference/seed9108401_readouts.npz')
    nd=read(BASE/'native_reference/checkpoint_projections.json')
    rows=[]
    folders={'frozen':BASE/'runs/A4_stoch_seed9108401',
             **{label:dest/label for label in ['units','units_history']}}
    count=np.load(folders['frozen']/'trajectory.npz')['cell_counts']
    for label in ['native','frozen','units','units_history']:
        if label=='native':
            t=native['t'];field=native['rate_cells'];whole=field.astype(float)@(count/count.sum())
            assert np.allclose(whole,native['allE'],rtol=1e-6,atol=1e-4)
            D={key:float(v['D']) for key,v in nd.items()};qa={'source':'Original native reference, not mean-input replacement.'}
        else:
            raw=np.load(folders[label]/'trajectory.npz');t=raw['time_ms'];field=raw['field_E_hz']
            assert np.array_equal(count,raw['cell_counts'])
            whole=field.astype(float)@(count/count.sum())
            group=raw['group_rate_hz'].astype(float)[:,s.E]@s.mean_weights
            err=float(max(abs(whole-raw['global_E_hz']).max(),abs(group-raw['global_E_hz']).max()))
            assert err<1e-4,err
            dd=1-raw['Z'].astype(float)[:,s.E]@s.mean_weights
            de=float(abs(dd-raw['D']).max());assert de<1e-6
            assert raw['Z'].min()>=-1e-7 and raw['Z'].max()<=1+1e-7
            ts=(np.arange(len(dd))+1)*10.
            D={key:float(raw['D'][np.flatnonzero(ts==int(key))[0]]) for key in nd}
            qa=dict(weighted_rate_error_Hz=err,D_float32_error=de,Z_physical=True,
                    M_dynamic_max_mV=float(raw['M_current'].max()))
        sm=uniform_filter1d(whole,10,mode='nearest');entry=high_entry(t,sm)
        ev,_,_,_=readouts(t,field,count,label);complete=[]
        for event in ev:
            a=int(np.searchsorted(t,event['start_ms']));b=a+int(event['duration_ms'])
            if a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all():
                complete.append(event)
        win={f'{a}-{b}':window_stats(complete,a,b) for a,b in c['event_windows_ms']}
        quiet={f'{a}-{b}':float(np.mean(sm[(t>=a)&(t<b)]<5)) for a,b in c['event_windows_ms']}
        if label!='native':assert entry==read(folders[label]/'result.json')['high_onset_ms']
        rows.append(dict(label=label,high_onset_ms=entry,D_at_native_checkpoints=D,
                         complete_event_windows=win,quiet_fraction_by_window=quiet,qa=qa,
                         tail_global_mean_Hz=float(whole[-1000:].mean())))
    write(dest/'independent_comparison.json',dict(status='READOUT_AUDIT_PASS',rows=rows,
        statistical_unit='One original native realization and three corresponding closure diagnostics. Same rate noise seed/counter keys, state-dependent Poisson counts. No replicate-based inference.',
        scope='Recorded external drive and original finite-group noise protocol; all Z/M dynamic. No native recurrence spikes provided. Timing agreement alone is not model acceptance.',
        replacement_promoted=False))
    log('STOCHASTIC AUDIT',[(r['label'],r['high_onset_ms'],r['D_at_native_checkpoints']['9870']) for r in rows])


if __name__=='__main__':main()
