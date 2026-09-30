"""Native single-core Z intervention paired with the existing all-held9s arm.

Use the original native checkpoint, cell-level resource and future inputs.
This is a correspondence check, not a reduced-model bifurcation certificate.
"""
from native_same_history_feedback import native as N, ROOT
from pathlib import Path
from datetime import datetime
import copy
import argparse
from scipy.ndimage import uniform_filter1d

np=N.np
OUT=ROOT/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'
DEST=OUT/'core_a_native_clamp_20260924'
PAIR=OUT/'native_same_history_feedback'
NAME='native_t9000_coreA_Z10370_held'
GEOMETRY=ROOT/'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g20/geometry.npz'


def prepare():
    DEST.mkdir(exist_ok=True);(DEST/'jobs').mkdir(exist_ok=True)
    reference=N.check_reference_sources()
    assert N.read(PAIR/'dynamic_replay_qa.json')['status']=='PASS'
    assert N.read(PAIR/'runs/native_t9000_Zheld/result.json')['status']=='COMPLETE'
    geo=np.load(GEOMETRY);region=geo['group_region'][geo['cell_group'][:N.NE]]
    A=np.r_[region==0,np.zeros(N.NI,dtype=bool)];assert A.sum()==754
    original=N.replay_checkpoint(9000);late=N.replay_checkpoint(10370)
    state=copy.deepcopy(original);state['slow']['z'][A]=late['slow']['z'][A]
    assert np.array_equal(state['slow']['z'][~A],original['slow']['z'][~A])
    restored=copy.deepcopy(state);restored['slow']['z']=original['slow']['z'].copy()
    assert N.compare_states(restored,original)==[]
    job=N.make_job(NAME,10370,9000,'W1',duration_ms=3500)
    job.update(start_step=90000,anchor_ms=9000,horizon_s=12.5,freeze_z=True,freeze_m=False,
               state_construction='Original9s full history; only754 CoreA E-cell Z values copied from original10.37s. AllZheld, allMdynamic, original future inputs and clock.')
    folder=DEST/'runs'/NAME;folder.mkdir(parents=True,exist_ok=True)
    protocol=dict(identity=reference['identity'],source_hashes=reference['source_hashes'],
        original_checkpoint=str(N.REPLAY_RUN/'checkpoints/t9000ms.npz'),
        core_A_Z_source=str(N.REPLAY_RUN/'checkpoints/t10370ms.npz'),
        reference_arm=str(PAIR/'runs/native_t9000_Zheld'),geometry=str(GEOMETRY),
        question='Does Core A-only resource depletion also yield sustained local activity without global recruitment in the unchanged native SNN?',
        statistical_unit='One matched native history and one original future input sequence; no seed replication.',
        budget='One3500ms new native continuation, reuse the independently audited all-held9s control.',
        initial_state_difference_keys=N.compare_states(state,original),
        only_Core_A_Z_changed=True,Core_A_cells=int(A.sum()),M_dynamic=True,
        comparison='Same Z intervention as rate screen, original within-group heterogeneity preserved. Native uses its real9s state and noisy input; rate uses its own settled state and constant input. No claim of identical cross-model histories.')
    if (DEST/'protocol.json').exists():assert N.read(DEST/'protocol.json')==protocol
    else:N.write(DEST/'protocol.json',protocol)
    path=DEST/'jobs'/f'{NAME}.json'
    if path.exists():assert N.read(path)==job
    else:N.write(path,job)
    checkpoint=folder/'checkpoint.pkl'
    if not checkpoint.exists():
        tracker=N.core.fresh_tracker();tracker['stop_s']=1e9
        N.save_pickle(checkpoint,dict(job=job,identity=protocol['identity'],engine=state,tracker=tracker,restore_from=None))
        assert N.compare_states(N.load_pickle(checkpoint)['engine'],state)==[]
        N.write(folder/'continuation.json',dict(source=protocol['original_checkpoint'],source_sha256=N.sha(Path(protocol['original_checkpoint'])),
            initial_difference_keys=protocol['initial_state_difference_keys'],only_Core_A_Z_changed=True,
            initial_global_Z=float(state['slow']['z'][:N.NE].mean()),initial_Core_A_Z=float(state['slow']['z'][A].mean()),
            initial_M_feedback_mv=float(.0005*state['slow']['m'][:N.NE].mean()),
            clock_rebased=False,random_streams_replaced=False,M_dynamic=True))
    np.savez_compressed(DEST/'initial_resources.npz',Z=state['slow']['z'],reference_Z=original['slow']['z'],A=A,E_region=region)
    return protocol


def run(device):
    protocol=prepare();N.NATIVE=DEST;N.prepare=lambda:protocol
    N.seed_folder=lambda job,device:DEST/'runs'/NAME
    result=N.worker(NAME,device);print('NATIVE CORE A',result['status'],flush=True)


def audit():
    from native_late_Z_clamp_audit import R, innovations, describe
    geo=np.load(PAIR/'geometry.npz');resources=np.load(DEST/'initial_resources.npz')
    counts=geo['region_counts'][:3];reference=PAIR/'runs/native_t9000_Zheld'
    ref_final=N.load_pickle(reference/'checkpoint.pkl')['engine'];ref_xi=innovations(reference)
    original=N.replay_checkpoint(9000);rows=[];arrays={}
    for label,folder in [('reference',reference),('coreA_depleted',DEST/'runs'/NAME)]:
        assert N.read(folder/'result.json')['status']=='COMPLETE'
        config=N.read(folder/'applied_configuration.json')
        assert config['frozen_Z_state_update'] and not config['frozen_M_state_update'] and config['M_effective_feedback']
        final=N.load_pickle(folder/'checkpoint.pkl')['engine']
        expected=resources['reference_Z' if label=='reference' else 'Z']
        assert np.array_equal(final['slow']['z'],expected)
        assert not np.array_equal(final['slow']['m'],original['slow']['m'])
        assert np.array_equal(innovations(folder),ref_xi)
        keys=['rng_state','external_drive','xi']
        assert N.compare_states({k:final[k] for k in keys},{k:ref_final[k] for k in keys})==[]
        d=R.load_chunks(folder,keys=('spikes_1ms','field_1ms','regions_1ms'))
        assert np.array_equal(d['regions_1ms'][:,:3].sum(1),d['spikes_1ms'][:,0])
        observed,field=describe(d,9000,geo['cell_e_counts'])
        regional=d['regions_1ms'][:,:3]/counts*1000
        stats=[]
        for j,name in enumerate(['Core A','Core B','Surround']):
            sm=uniform_filter1d(regional[-1000:,j],10,mode='nearest')
            stats.append(dict(region=name,mean_rate_hz=float(regional[-1000:,j].mean()),
                quiet_fraction=float((sm<5).mean()),duty_above50=float((sm>50).mean()),
                range_10ms_hz=[float(sm.min()),float(sm.max())]))
        rows.append(dict(label=label,source=str(folder),regional=stats,**observed))
        arrays[label+'_field_Hz']=field.astype('f4');arrays[label+'_regional_Hz']=regional.astype('f4')
    np.savez_compressed(DEST/'fields.npz',**arrays,cell_counts=geo['cell_e_counts'],centers_mm=geo['centers_mm'])
    result=dict(status='AUDIT_PASS',rows=rows,same_future_inputs=True,held_Z_exact=True,all_M_dynamic=True,
        scope='Single native paired history,9–12.5s,final1s local readout. No asymptotic state or bifurcation type assigned.')
    N.write(DEST/'result.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','audit']);p.add_argument('--device',type=int,default=0)
    a=p.parse_args();{'prepare':prepare,'run':lambda:run(a.device),'audit':audit}[a.command]()
