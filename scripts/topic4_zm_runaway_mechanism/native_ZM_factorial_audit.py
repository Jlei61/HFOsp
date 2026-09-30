"""Independent four-arm native feedback audit; no new state classification."""
from native_late_Z_clamp_audit import *
import csv

DEST=OUT/'native_ZM_factorial'


def main():
    contract=N.read(OUT/'native_ZM_factorial_contract.json')
    assert contract['new_arms']==['Zdynamic_Mheld','Zheld_Mheld']
    assert N.read(DEST/'M_only_application_check.json')['status']=='PASS'
    assert N.read(PAIR/'dynamic_replay_qa.json')['status']=='PASS'
    N.check_reference_sources()
    initial=N.replay_checkpoint(9000);ref=PAIR/'runs/native_t9000_Zdynamic'
    input_keys=['rng_state','external_drive','xi']
    ref_final=N.load_pickle(ref/'checkpoint.pkl')['engine'];ref_xi=innovations(ref)
    ref_origin=N.read(ref/'continuation.json')
    geometry=np.load(PAIR/'geometry.npz');counts=geometry['cell_e_counts']
    arms=[('Zdynamic_Mdynamic',PAIR,'native_t9000_Zdynamic',False,False),
          ('Zheld_Mdynamic',PAIR,'native_t9000_Zheld',True,False),
          ('Zdynamic_Mheld',DEST,'native_t9000_Zdynamic_Mheld',False,True),
          ('Zheld_Mheld',DEST,'native_t9000_Zheld_Mheld',True,True)]
    rows=[];arrays={};table=[]
    for label,parent,name,fz,fm in arms:
        folder=parent/'runs'/name
        assert N.read(folder/'result.json')['status']=='COMPLETE'
        job=N.read(parent/'jobs'/f'{name}.json')
        applied=N.read(folder/'applied_configuration.json')
        origin=N.read(folder/'continuation.json')
        assert job['freeze_z']==fz and job['freeze_m']==fm
        assert applied['frozen_Z_state_update']==fz and applied['frozen_M_state_update']==fm
        assert applied['Z_enabled'] and applied['M_effective_feedback']
        assert applied['frozen_values_still_applied_to_current']
        assert origin['initial_state_bitwise_identical']
        assert origin['source_sha256']==ref_origin['source_sha256']
        assert not origin['clock_rebased'] and not origin['random_streams_replaced']
        final=N.load_pickle(folder/'checkpoint.pkl')['engine']
        checks={}
        for key,frozen in [('z',fz),('m',fm)]:
            unchanged=np.array_equal(final['slow'][key],initial['slow'][key])
            checks[f'{key}_final_bitwise_unchanged']=unchanged
            if frozen:assert unchanged
        if fm:
            recorded=np.concatenate([np.load(f)['m'] for f in sorted((folder/'fields').glob('*.npz'))
                                     if '.tmp.' not in f.name])
            assert np.array_equal(recorded,np.broadcast_to(initial['slow']['m'][:N.NE],recorded.shape))
            checks['all_recorded_M_snapshots_bitwise_constant']=True
        checks['same_future_xi']=np.array_equal(innovations(folder),ref_xi)
        checks['final_input_difference_keys']=N.compare_states(
            {k:final[k] for k in input_keys},{k:ref_final[k] for k in input_keys})
        assert checks['same_future_xi'] and checks['final_input_difference_keys']==[]
        d=R.load_chunks(folder,keys=('spikes_1ms','field_1ms'))
        observed,field=describe(d,9000,counts)
        row=dict(arm=label,source=str(folder),Z_update='held' if fz else 'dynamic',
            M_update='held' if fm else 'dynamic',checks=checks,
            global_Z_initial=origin['initial_global_Z'],
            global_Z_final=float(final['slow']['z'][:N.NE].mean()),
            M_feedback_initial_mv=origin['initial_M_feedback_mv'],
            M_feedback_final_mv=float(job['eta_m']*final['slow']['m'][:N.NE].mean()),
            **observed)
        rows.append(row);arrays[label+'_field_Hz']=field.astype(np.float32)
        table.append(dict(arm=label,high_entry_s=row['high_entry_s'],broad_entry_s=row['broad_entry_s'],
            complete_events=len(row['complete_events']),tail_mean_E_hz=row['tail']['mean_rate_hz'],
            tail_persistent_fraction_50Hz_duty80=row['tail']['persistent_fraction']['50Hz_duty80'],
            global_Z_final=row['global_Z_final'],M_feedback_final_mv=row['M_feedback_final_mv']))
        print(table[-1],flush=True)
    q=dict(status='COMPLETE',contract=str(OUT/'native_ZM_factorial_contract.json'),rows=rows,
        original_dynamic_replay='PASS',same_complete_initial_state=True,same_future_inputs=True,
        statistical_unit='One original history with paired counterfactuals; four arms are not independent samples.',
        scope='Finite9.0-12.5s feedback dependence. No mathematical bifurcation type or all-history necessity claim.',
        main_readout='Original global200Hz for200ms; broad75percent spatial criterion is additional and separate.')
    N.write(DEST/'result.json',q)
    np.savez_compressed(DEST/'fields.npz',**arrays,cell_counts=counts,centers_mm=geometry['centers_mm'])
    with (DEST/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=table[0].keys());writer.writeheader();writer.writerows(table)


if __name__=='__main__':main()
