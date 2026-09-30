"""Independent stored-rate/resource readout of the single variance diagnostic."""
from common import *
from scipy.ndimage import uniform_filter1d
from native_readouts import readouts,window_stats
from closure_network_sensitivity_audit import high_entry


def main():
    dest=OUT/'shared_variance_network_sensitivity'
    folder=dest/'units_history_delay_covariance';q=read(folder/'result.json')
    assert q['status']=='COMPLETE'
    s=model();z=np.load(folder/'trajectory.npz');counts=z['cell_counts']
    whole=z['field_E_hz'].astype(float)@(counts/counts.sum())
    group=z['group_rate_hz'].astype(float)[:,s.E]@s.mean_weights
    err=float(max(abs(whole-z['global_E_hz']).max(),abs(group-whole).max()))
    assert err<1e-4,err
    D=1-z['Z'].astype(float)[:,s.E]@s.mean_weights
    derr=float(abs(D-z['D']).max());assert derr<1e-6
    assert z['Z'].min()>=0 and z['Z'].max()<=1
    t=z['time_ms'];sm=uniform_filter1d(whole,10,mode='nearest')
    entry=high_entry(t,sm);assert entry==q['high_onset_ms']
    events,_,_,_=readouts(t,z['field_E_hz'],counts,'units_history_delay_covariance')
    complete=[]
    for ev in events:
        a=int(np.searchsorted(t,ev['start_ms']));b=a+int(ev['duration_ms'])
        if a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all():
            complete.append(ev)
    c=read(OUT/'closure_network_sensitivity_contract.json')
    wins={f'{a}-{b}':window_stats(complete,a,b) for a,b in c['event_windows_ms']}
    quiet={f'{a}-{b}':float(np.mean(sm[(t>=a)&(t<b)]<5)) for a,b in c['event_windows_ms']}
    nd=read(BASE/'native_reference/checkpoint_projections.json')
    d_at={key:float(z['D'][np.flatnonzero(z['state_time_ms']==int(key))[0]]) for key in nd}
    row=dict(label='units_history_delay_covariance',high_onset_ms=entry,
        D_at_native_checkpoints=d_at,complete_event_windows=wins,quiet_fraction_by_window=quiet,
        qa=dict(weighted_rate_error_Hz=err,D_float32_error=derr,Z_physical=True,
                M_dynamic_max_mV=float(z['M_current'].max())))
    previous=read(OUT/'closure_stochastic_sensitivity/independent_comparison.json')
    refs=[v for v in previous['rows'] if v['label'] in ['native','units_history']]
    result=dict(status='READOUT_AUDIT_PASS',rows=refs+[row],
        statistical_unit='One original realization and one paired variance-partition sensitivity; no replicate inference.',
        scope=read(OUT/'shared_variance_network_sensitivity_contract.json')['scope'],replacement_promoted=False)
    write(dest/'independent_comparison.json',result)
    log('VARIANCE SENSITIVITY COMPARISON',[(r['label'],r['high_onset_ms'],r['D_at_native_checkpoints']['9870']) for r in result['rows']])


if __name__=='__main__':main()
