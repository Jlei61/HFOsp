"""Compare autonomous closure diagnostics on their common clock and initial history."""
from common import *
from scipy.ndimage import uniform_filter1d
from native_readouts import readouts, window_stats

DEST=OUT/'closure_network_sensitivity'


def high_entry(t,sm):
    spans=np.diff(np.r_[0,(sm>=200).astype(int),0])
    return next((float(t[a]) for a,b in zip(np.flatnonzero(spans==1),np.flatnonzero(spans==-1)) if b-a>=200),None)


def main():
    c=read(OUT/'closure_network_sensitivity_contract.json')
    assert read(DEST/'result.json')['status']=='DIAGNOSTIC_COMPLETE'
    s=model();reference=np.load(DEST/'frozen/initial.npz')
    native=np.load(BASE/'native_reference/seed9108401_readouts.npz')
    nd=read(BASE/'native_reference/checkpoint_projections.json')
    rows=[]
    for label in ['native']+c['variants']:
        if label=='native':
            t=native['t'];field=native['rate_cells'];count=np.load(DEST/'frozen/trajectory.npz')['cell_counts']
            whole=field.astype(float)@(count/count.sum())
            assert np.allclose(whole,native['allE'],rtol=1e-6,atol=1e-4)
            zrow={key:float(value['D']) for key,value in nd.items()}
            qa=dict(scope='Stored original seed9108401 native data, independently reaggregated.')
        else:
            initial=np.load(DEST/label/'initial.npz')
            assert np.array_equal(initial['state'],reference['state'])
            assert np.array_equal(initial['history'],reference['history'])
            raw=np.load(DEST/label/'trajectory.npz');t=raw['time_ms'];field=raw['field_E_hz'];count=raw['cell_counts']
            assert np.array_equal(t,np.arange(12500)+1.)
            whole=field.astype(float)@(count/count.sum())
            group=raw['group_rate_hz'].astype(float)[:,s.E]@s.mean_weights
            error=float(max(np.max(abs(whole-raw['global_E_hz'])),np.max(abs(group-raw['global_E_hz']))))
            assert error<1e-4,error
            d=1-raw['Z'].astype(float)[:,s.E]@s.mean_weights
            d_error=float(np.max(abs(d-raw['D'])));assert d_error<1e-6
            assert raw['Z'].min()>=-1e-7 and raw['Z'].max()<=1+1e-7
            zrow={key:float(raw['D'][np.flatnonzero(raw['state_time_ms']==int(key))[0]]) for key in nd}
            qa=dict(initial_state_bitwise=True,initial_history_bitwise=True,
                    weighted_rate_max_error_Hz=error,D_float32_reaggregation_error=d_error,
                    Z_physical_bounds=True,M_dynamic_peak_mV=float(raw['M_current'].max()))
        sm=uniform_filter1d(whole,10,mode='nearest');entry=high_entry(t,sm)
        event,_,_,_=readouts(t,field,count,label)
        complete=[]
        for ev in event:
            a=int(np.searchsorted(t,ev['start_ms']));b=a+int(ev['duration_ms'])
            if a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all():
                complete.append(ev)
        windows={f'{a}-{b}':window_stats(complete,a,b) for a,b in c['event_windows_ms']}
        quiet={f'{a}-{b}':float(np.mean(sm[(t>=a)&(t<b)]<5)) for a,b in c['event_windows_ms']}
        rows.append(dict(label=label,high_onset_ms=entry,complete_event_windows=windows,
                         quiet_fraction_by_window=quiet,D_at_native_checkpoints=zrow,qa=qa,
                         tail_global_mean_Hz=float(whole[-1000:].mean())))
        if label!='native':assert entry==read(DEST/label/'result.json')['high_onset_ms']
    write(DEST/'independent_comparison.json',dict(status='MATCHED_INITIAL_AND_READOUT_AUDIT_PASS',rows=rows,
          statistical_unit='One original native history and three deterministic rate variants; correlated model diagnostics, no replicate-based inference.',
          scope=c['scope'],replacement_promoted=False))
    log('CLOSURE SENSITIVITY AUDIT',[(r['label'],r['high_onset_ms'],r['D_at_native_checkpoints']['9870']) for r in rows])


if __name__=='__main__':main()
