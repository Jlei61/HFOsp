"""Connect the conditional turn candidate to matched finite-time dynamics."""
from common import *
from native_post_transition_geometry import describe
import subprocess,json,argparse


def main(a):
    batch=read(OUT/f'endpoint_runs_{a.batch}.json');assert batch['status']=='COMPLETE'
    contract=read(OUT/'native_near_turn_matched_contract.json');rows=[];initial=None
    for D in contract['D']:
        folder=OUT/'runs'/f'endpoint_D{D:.7f}_dt{a.dt}'
        run=read(folder/'contract.json');z=np.load(folder/'trajectory.npz')
        current=Path(run['initial']).resolve()
        if initial is None:initial=current
        assert current==initial
        assert run['dt_ms']==a.dt and run['duration_ms']==30000
        assert run['Z']=='held' and run['M']=='dynamic'
        assert run['rate_history_scheme']=='instantaneous rate at the labelled endpoint'
        canonical=json.loads(subprocess.check_output([sys.executable,
            str(HERE/'canonical_case_readout.py'),str(folder)],text=True))
        spatial=describe(folder/'trajectory.npz')
        events=canonical['events']
        durations=[e['duration_ms'] for e in events]
        rows.append(dict(D=D,global_Z=1-D,canonical=canonical,spatial=spatial,
            complete_events_whole_30s=len(events),
            maximum_complete_event_ms=max(durations,default=None)))
        log('MATCHED NEAR TURN',D,canonical['category'],
            canonical['tail']['mean_rate_hz'],spatial['spatial_recurrence_peaks'][:1])
    q=dict(status='COMPLETE',contract=str(OUT/'native_near_turn_matched_contract.json'),
        batch=a.batch,dt_ms=a.dt,initial=str(initial),rows=rows,
        question=contract['question'],
        limitation='Finite 30 s conditional trajectories. No bifurcation, asymptotic survival, or branch connectivity is inferred solely from these readouts.')
    write(OUT/f'native_near_turn_matched_audit_dt{a.dt}.json',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dt',type=float,default=.05)
    p.add_argument('--batch',default='native_near_turn');main(p.parse_args())
