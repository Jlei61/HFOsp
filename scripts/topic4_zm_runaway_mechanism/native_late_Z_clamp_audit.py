"""Audit original-state Z clamps against their unchanged reference future.

The different clamp clocks are paired interventions on one original history,
not replicates or a transplant isolating only the initial value of Z.
"""
from native_same_history_audit import *


def innovations(folder):
    parts=[];end=None
    for path in sorted((folder/'fields').glob('*.npz')):
        if '.tmp.' in path.name:continue
        z=np.load(path);start=int(z['start_step'])
        if end is not None:assert start==end
        end=int(z['end_step']);assert len(z['xi'])==end-start
        parts.append(z['xi'])
    return np.concatenate(parts)


def describe(d,start,counts):
    assert d['start_step']==start*10 and d['end_step']==125000
    field=d['field_1ms']/counts*1000
    cell=field.reshape(-1,10,400).mean(1)
    rate=R.rate_10ms(d['spikes_1ms'][:,0],N.NE)
    weights=counts/counts.sum()
    assert np.max(abs(rate-cell@weights))<1e-10
    seps,events=R.find_events(rate);high=R.high_rate_entry(rate)
    broad=(rate>=200)&((cell>50)@weights>=.75)
    broad_runs=[(int(lo),int(hi)) for lo,hi in R.runs_of(broad) if hi-lo>=20]
    ev=[dict(e,start_s=start/1000+e['start_bin']*.01,
             end_s=start/1000+e['end_bin']*.01) for e in events if e['qualifies']]
    tail=R.window_stats(rate,seps,events,cell,counts,
                       (11500-start)//10,(12500-start)//10,high)
    return dict(high_entry_s=None if high is None else start/1000+high['onset_bin']*.01,
        high_confirmation_s=None if high is None else start/1000+high['confirmation_bin']*.01,
        high_entry_left_censored=bool(high is not None and high['onset_bin']==0),
        broad_entry_s=None if not broad_runs else start/1000+broad_runs[0][0]*.01,
        broad_entry_left_censored=bool(broad_runs and broad_runs[0][0]==0),
        complete_events=ev,tail_window_ms=[11500,12500],tail=tail),field


def main():
    contract=N.read(OUT/'native_late_Z_clamp_contract.json')
    assert contract['clamp_times_ms']==[9420,9870] and contract['stop_ms']==12500
    assert N.read(PAIR/'dynamic_replay_qa.json')['status']=='PASS'
    N.check_reference_sources()
    ref=PAIR/'runs/native_t9000_Zdynamic'
    original=R.load_chunks(ref,keys=('spikes_1ms','field_1ms'))
    ref_xi=innovations(ref);ref_final=N.load_pickle(ref/'checkpoint.pkl')['engine']
    input_keys=['rng_state','external_drive','xi']
    geo=np.load(PAIR/'geometry.npz');counts=geo['cell_e_counts']
    rows=[];arrays={}
    for start in [9000]+contract['clamp_times_ms']:
        folder=PAIR/'runs'/f'native_t{start}_Zheld'
        assert N.read(folder/'result.json')['status']=='COMPLETE'
        job=N.read(PAIR/'jobs'/f'native_t{start}_Zheld.json')
        config=N.read(folder/'applied_configuration.json')
        origin=N.read(folder/'continuation.json')
        assert job['freeze_z'] and not job['freeze_m']
        assert config['frozen_Z_state_update'] and not config['frozen_M_state_update']
        assert config['Z_enabled'] and config['M_effective_feedback']
        assert origin['initial_state_bitwise_identical'] and not origin['clock_rebased']
        assert not origin['random_streams_replaced']
        initial=N.replay_checkpoint(start)
        final=N.load_pickle(folder/'checkpoint.pkl')['engine']
        z_unchanged=np.array_equal(initial['slow']['z'],final['slow']['z'])
        input_diff=N.compare_states({k:ref_final[k] for k in input_keys},
                                    {k:final[k] for k in input_keys})
        same_xi=np.array_equal(innovations(folder),ref_xi[(start-9000)*10:])
        assert z_unchanged and same_xi and input_diff==[]
        observed=R.load_chunks(folder,keys=('spikes_1ms','field_1ms'))
        reference={k:v[start-9000:] for k,v in original.items()
                   if k in ['spikes_1ms','field_1ms']}
        reference.update(start_step=start*10,end_step=125000)
        held,field=describe(observed,start,counts)
        dynamic,ref_field=describe(reference,start,counts)
        arrays[f'field_held_{start}_Hz']=field.astype(np.float32)
        if start==9000:arrays['field_dynamic_9000_Hz']=ref_field.astype(np.float32)
        rows.append(dict(clamp_time_ms=start,source=str(folder),
            global_Z_at_clamp=origin['initial_global_Z'],
            Z_bitwise_unchanged=z_unchanged,same_future_xi=same_xi,
            final_input_state_difference_keys=input_diff,held=held,dynamic=dynamic))
        print(start,'held',held['high_entry_s'],held['broad_entry_s'],
              held['tail']['mean_rate_hz'],held['tail']['persistent_fraction'],flush=True)
    q=dict(status='COMPLETE',contract=str(OUT/'native_late_Z_clamp_contract.json'),
        rows=rows,statistical_unit=contract['interpretation'],
        reference_source=str(ref),dynamic_replay_qa='PASS',
        Z='held separately from each original checkpoint; reference dynamic',M='dynamic',
        bifurcation_type='NOT_INFERRED',
        limitation='A high run starting at the clamp boundary is left-censored; its true onset cannot be recovered from this continuation alone. No 4s-tail state category is assigned to these shorter records.')
    np.savez_compressed(PAIR/'late_clamp_fields.npz',**arrays,
                        cell_counts=counts,centers_mm=geo['centers_mm'])
    N.write(PAIR/'late_clamp_result.json',q)


if __name__=='__main__':main()
