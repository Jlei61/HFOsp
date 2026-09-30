"""Measured native input balance in the existing same-history intervention.

Window averages are from the original 5-ms current observer. A separate final
snapshot calculation restores only Z in the membrane-current formula; that
algebraic counterfactual is not a further simulated trajectory.
"""
from native_same_history_audit import *


def main():
    assert N.read(PAIR/'result.json')['status']=='COMPLETE'
    assert N.read(PAIR/'dynamic_replay_qa.json')['status']=='PASS'
    N.check_reference_sources()
    initial=N.replay_checkpoint(9000);z_initial=initial['slow']['z'][:N.NE]
    windows=[(9000,9420),(9420,9870),(9870,10070),(10070,12500)]
    rows=[];snapshots=[]
    for condition in ['held','dynamic']:
        folder=PAIR/'runs'/f'native_t9000_Z{condition}'
        chunks=sorted(folder.joinpath('chunks').glob('*.npz'))
        arrays={key:[] for key in ['slow_time_ms','currents','M','Z']}
        for path in chunks:
            z=np.load(path)
            for key in arrays:arrays[key].append(z[key])
        arrays={key:np.concatenate(value) for key,value in arrays.items()}
        times=arrays['slow_time_ms'];assert np.all(np.diff(times)==5)
        job=N.read(PAIR/'jobs'/f'native_t9000_Z{condition}.json');eta=job['eta_m']
        assert eta==.0005 and job['freeze_m'] is False
        for start,end in windows:
            mask=(times>=start)&(times<end);assert mask.sum()==(end-start)//5
            c=arrays['currents'][mask];m=eta*arrays['M'][mask,0]
            exc,raw,effective=c.mean(0);feedback=float(m.mean())
            rows.append(dict(condition=condition,absolute_window_ms=[start,end],
                sampling_ms=5,samples=int(mask.sum()),unit='mV-equivalent membrane drive',
                mean_E_input=float(exc),mean_raw_I_input=float(raw),
                mean_Z_times_I_input=float(effective),mean_eta_M_M=feedback,
                mean_net_drive=float(exc-effective-feedback),
                ratio_of_window_means_effective_I_over_raw_I=float(effective/raw),
                ratio_of_window_means_effective_I_over_E=float(effective/exc),
                global_mean_Z=float(arrays['Z'][mask,0].mean())))
        final=N.load_pickle(folder/'checkpoint.pkl')['engine']
        assert final['absolute_time_ms']==12500
        exc=final['I_E'][:N.NE];raw=final['I_I'][:N.NE]
        z=final['slow']['z'][:N.NE];m=eta*final['slow']['m'][:N.NE]
        net=exc-z*raw-m;restored=exc-z_initial*raw-m
        gain=(z_initial-z)*raw
        assert np.max(abs((net-restored)-gain))<1e-10
        snapshots.append(dict(condition=condition,time_ms=12500,
            native_state_source=str(folder/'checkpoint.pkl'),
            mean_E_input=float(exc.mean()),mean_raw_I_input=float(raw.mean()),
            mean_effective_I=float((z*raw).mean()),mean_M_feedback=float(m.mean()),
            mean_net_drive=float(net.mean()),
            net_drive_with_original9s_Z_only=float(restored.mean()),
            extra_net_drive_from_Z_change_with_all_other_final_variables_fixed=float(gain.mean()),
            scope='Algebraic evaluation on the final state. Not a simulation, equilibrium or actual membrane-update record.'))
    out=dict(status='COMPLETE',rows=rows,final_snapshot_counterfactual=snapshots,
        measured_source='Existing currents[:,0:3] = E input, raw I input, mean(cell Z*I); M[:,0] is all-E mean M. Original observer samples every5ms before the native slow update.',
        current_equation='I_net = I_E - Z*I_I - eta_M*M',
        producer=str(ROOT/'scripts/run_topic4_m_parameter_modes.py'),
        current_equation_source=str(ROOT/'src/snn_engine/mz_slow_vars.py'),
        statistical_unit='One paired native history; 5ms samples are not independent replicates.',
        limitation='Increased absolute inhibition can coexist with reduced inhibition relative to excitation. Window balance alone is not a bifurcation or a region-specific causal test.')
    N.write(PAIR/'current_balance.json',out)
    for row in rows:
        print(row['condition'],row['absolute_window_ms'],
              {k:round(row[k],4) for k in ['mean_E_input','mean_raw_I_input','mean_Z_times_I_input',
                                           'mean_eta_M_M','mean_net_drive',
                                           'ratio_of_window_means_effective_I_over_E']},flush=True)


if __name__=='__main__':main()
