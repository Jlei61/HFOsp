"""Native Fig.5 feedback intervention from its actual, untransplanted state.

The clock, membrane/synaptic/delay state, M, OU field and RNG streams all come
from one reference checkpoint. Only whether Z updates is changed. The dynamic
arm within the replay horizon must recover the previously recorded trajectory.
"""
import sys,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/topic4_fig5_z_state'))
import native_continue as native


def main(a):
    assert a.start_ms in [9000,9420,9870]
    assert a.stop_ms==12500, 'This paired experiment has a fixed replay horizon'
    reference=native.check_reference_sources()
    qa=native.read(native.REPLAY_RUN/'replay_qa.json');assert qa['status']=='PASS'
    source=native.REPLAY_RUN/'checkpoints'/f't{a.start_ms}ms.npz'
    state=native.replay_checkpoint(a.start_ms)
    assert int(state['step'])==native.ms_to_step(a.start_ms)
    out=ROOT/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918/native_same_history_feedback'
    out.mkdir(parents=True,exist_ok=True);(out/'jobs').mkdir(exist_ok=True)
    protocol=dict(identity=reference['identity'],source_hashes=reference['source_hashes'],
        replay_qa=str(native.REPLAY_RUN/'replay_qa.json'),
        scope='Native SNN same-state feedback intervention, original absolute clock and future innovations',
        Z='held or dynamic according to named arm',M='dynamic in both arms',
        statistical_unit='one paired reference history; checkpoints are not independent replicates',
        duration_scope='Until original absolute12.5s; absence of entry is finite-time censoring')
    if (out/'protocol.json').exists():assert native.read(out/'protocol.json')==protocol
    else:native.write(out/'protocol.json',protocol)
    name=f'native_t{a.start_ms}_Z{a.condition}'
    job=native.make_job(name,a.start_ms,a.start_ms,'W1',duration_ms=a.stop_ms-a.start_ms)
    job.update(start_step=int(state['step']),anchor_ms=a.start_ms,horizon_s=a.stop_ms/1000,
               freeze_z=a.condition=='held',freeze_m=False,
               state_construction='Complete original checkpoint; no clock, RNG, M or fast-history transplant')
    path=out/'jobs'/f'{name}.json'
    if path.exists():assert native.read(path)==job
    else:native.write(path,job)
    folder=out/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/'checkpoint.pkl'
    if not checkpoint.exists():
        tracker=native.core.fresh_tracker();tracker['stop_s']=1e9
        saved=dict(job=job,identity=protocol['identity'],engine=state,tracker=tracker,restore_from=None)
        native.save_pickle(checkpoint,saved)
        differences=native.compare_states(native.load_pickle(checkpoint)['engine'],state)
        assert differences==[]
        native.write(folder/'continuation.json',dict(source=str(source),source_sha256=native.sha(source),
            seeded_at=time.time(),initial_state_bitwise_identical=True,initial_difference_keys=differences,
            initial_global_Z=float(state['slow']['z'][:native.NE].mean()),
            initial_M_feedback_mv=float(job['eta_m']*state['slow']['m'][:native.NE].mean()),
            initial_step=int(state['step']),clock_rebased=False,random_streams_replaced=False,
            Z_only_intervention=True,M_dynamic=True))
    else:assert native.load_pickle(checkpoint)['job']==job
    native.NATIVE=out
    native.prepare=lambda:protocol
    native.seed_folder=lambda job,device:folder
    result=native.worker(name,a.device)
    print(name,result['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=['held','dynamic'],required=True)
    p.add_argument('--start-ms',type=int,default=9000);p.add_argument('--stop-ms',type=int,default=12500)
    p.add_argument('--device',type=int,default=1);main(p.parse_args())
