"""Complete the same-native-history Z/M feedback factorial with two new arms.

Reuse the unchanged and Z-held arms already audited. Preserve each frozen
state's current effect; suppress only its update. No new topology or input.
"""
from native_same_history_feedback import native,ROOT
import argparse,time

OUT=ROOT/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'
DEST=OUT/'native_ZM_factorial'


def application_check():
    np=native.np;old=native.core.old
    cfg=old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000.,
        I_th_EI=old.THRESHOLD,tau_adp=1000.,eta_m=.0005)
    rng=np.random.default_rng(91971)
    live=old.ReleaseZ(8,18.,cfg,NE=6);held=old.ReleaseZ(8,18.,cfg,NE=6)
    z=rng.uniform(.5,.9,8);z[6:]=1.;m=rng.uniform(0,30,8);m[6:]=0.
    for slow in (live,held):slow.z[:]=z;slow.m[:]=m
    native.FrozenSlow(held,False,True)
    for k in range(3000):
        ie,ii=rng.uniform(0,300,(2,8));spikes=rng.random(8)<.1
        live.apply_currents(ie,ii);actual=held.apply_currents(ie,ii)
        assert np.array_equal(actual,ie-held.z*ii-cfg.eta_m*m)
        live.step(spikes,None,.1);held.step(spikes,None,.1)
        assert np.array_equal(held.m,m) and np.array_equal(held.z,live.z)
    assert not np.array_equal(live.m,m) and not np.array_equal(held.z,z)
    q=dict(status='PASS',steps=3000,M_state_frozen=True,
           frozen_M_still_subtracts_current=True,
           dynamic_Z_identical_under_identical_prescribed_inputs=True)
    native.write(DEST/'M_only_application_check.json',q)
    return q


def main(a):
    contract=native.read(OUT/'native_ZM_factorial_contract.json')
    assert contract['start_ms']==9000 and contract['stop_ms']==12500
    assert contract['new_arms']==['Zdynamic_Mheld','Zheld_Mheld']
    reference=native.check_reference_sources()
    assert native.read(native.REPLAY_RUN/'replay_qa.json')['status']=='PASS'
    assert native.read(OUT/'native_same_history_feedback/dynamic_replay_qa.json')['status']=='PASS'
    DEST.mkdir(parents=True,exist_ok=True);(DEST/'jobs').mkdir(exist_ok=True)
    application_check()
    source=native.REPLAY_RUN/'checkpoints/t9000ms.npz'
    state=native.replay_checkpoint(9000);assert int(state['step'])==90000
    protocol=dict(identity=reference['identity'],source_hashes=reference['source_hashes'],
        contract=str(OUT/'native_ZM_factorial_contract.json'),
        scope='Same complete native9s state, original future input and clock; freeze only named state updates',
        statistical_unit='one original history, not four independent replicates')
    if (DEST/'protocol.json').exists():assert native.read(DEST/'protocol.json')==protocol
    else:native.write(DEST/'protocol.json',protocol)
    name='native_t9000_'+a.condition
    job=native.make_job(name,9000,9000,'W1',duration_ms=3500)
    job.update(start_step=90000,anchor_ms=9000,horizon_s=12.5,
        freeze_z=a.condition=='Zheld_Mheld',freeze_m=True,
        state_construction='Complete original checkpoint, no clock/RNG/history or state transplant')
    path=DEST/'jobs'/f'{name}.json'
    if path.exists():assert native.read(path)==job
    else:native.write(path,job)
    folder=DEST/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/'checkpoint.pkl'
    if not checkpoint.exists():
        tracker=native.core.fresh_tracker();tracker['stop_s']=1e9
        native.save_pickle(checkpoint,dict(job=job,identity=protocol['identity'],
            engine=state,tracker=tracker,restore_from=None))
        differences=native.compare_states(native.load_pickle(checkpoint)['engine'],state)
        assert differences==[]
        native.write(folder/'continuation.json',dict(source=str(source),source_sha256=native.sha(source),
            seeded_at=time.time(),initial_state_bitwise_identical=True,
            initial_difference_keys=differences,initial_step=90000,
            initial_global_Z=float(state['slow']['z'][:native.NE].mean()),
            initial_M_feedback_mv=float(job['eta_m']*state['slow']['m'][:native.NE].mean()),
            clock_rebased=False,random_streams_replaced=False,
            freeze_Z=job['freeze_z'],freeze_M=True))
    else:assert native.load_pickle(checkpoint)['job']==job
    native.NATIVE=DEST;native.prepare=lambda:protocol;native.seed_folder=lambda job,device:folder
    result=native.worker(name,a.device);print(name,result['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True,
        choices=['Zdynamic_Mheld','Zheld_Mheld']);p.add_argument('--device',type=int,default=0)
    main(p.parse_args())
