"""Register a same-Z comparison seeded by both onset endpoint histories.

This is state continuation, not equilibrium or periodic-orbit continuation.
It preserves the complete rate-model fast, refractory, delay and M states.
"""
from common import np, read, write, log, model
from onset_state_continuation import DEST, run, build, initialize
from fine_rate_frozen_Z_fields import capture, restore
from datetime import datetime
import argparse


def register(field, lower_source='lower_endpoint', upper_source='upper_endpoint'):
    assert field in np.load(DEST/'fields.npz').files
    audits = {side: read(DEST/f'{side}_endpoint/independent_audit.json')
              for side in ('lower', 'upper')}
    assert all(a['status'] == 'AUDIT_PASS' and a['observed_ms'] == 20000
               for a in audits.values())
    low = audits['lower']['windows'][-1]
    high = audits['upper']['windows'][-1]
    assert low['complete_events']['n'] > 0 and low['quiet_fraction'] > 0
    assert high['quiet_fraction'] == 0 and high['persistent_spatial_fraction'] > 0
    path = DEST/f'{field}_pair_contract.json'
    assert not path.exists()
    conditions = read(DEST/'conditions.json')
    new = []
    source_readouts={}
    for side,source in [('lower',lower_source),('upper',upper_source)]:
        source_audit=read(DEST/source/'independent_audit.json')
        assert source_audit['status']=='AUDIT_PASS'
        assert read(DEST/source/'jobs.json')['status']=='COMPLETE'
        source_readouts[side]=dict(label=source,last_window=source_audit['windows'][-1])
        label = f'{field}_from_{side}'
        assert label not in conditions
        c = dict(label=label, field=field,
                 initial=str(DEST/source/'final_state.npz'),
                 previous_elapsed_ms=0, duration_ms=10000)
        conditions[label] = c
        new.append(c)
    s = model(40)
    Z = np.load(DEST/'fields.npz')[field]
    write(path, dict(
        created_local=datetime.now().astimezone().isoformat(),
        question='At exactly the same spatial Z field, do settled self-limited and sustained complete histories approach the same activity or remain distinct?',
        field=field, D=float(1-Z[s.E]@s.mean_weights), conditions=new,
        equations='Exactly the endpoint conditional-drift equations; Z entire field held, M dynamic, constant original external mean, no future count innovations.',
        intervention='Only Z is replaced. Synapses, local response states, full delay/refractory history, M and clock are carried from the specified completed and audited source. Different M is part of the complete initial state, not a separately changed parameter.',
        readout='Same original event/quiet/spatial readouts; final5s comparison, full E/I spatial-rate recurrence and saved exact checkpoints.',
        budget='Two10s runs for this one target field. No adaptive extra run until these results have been audited.',
        interpretation='Agreement narrows the activity-transition bracket. Persistence of distinct results supports finite-time history dependence and motivates basin/longer-time tests; it does not by itself certify bistability or a separatrix.',
        endpoint_evidence={k: a['windows'][-1] for k,a in audits.items()},
        initial_source_evidence=source_readouts,
        model_promoted=False, bifurcation_type='NOT_ESTABLISHED'))
    write(DEST/'conditions.json', conditions)
    log('ONSET NEIGHBOR REGISTERED', field, new)


def check(field, device):
    contract = read(DEST/f'{field}_pair_contract.json')
    e = build(device)
    rows = []
    for c in contract['conditions']:
        Z = initialize(e, c)
        state = capture(e)
        a = e.chunk()
        end = capture(e)
        restore(e, state)
        b = e.chunk()
        repeated = capture(e)
        assert np.array_equal(a, b) and np.array_equal(b[:,0], b[:,1])
        assert all(np.array_equal(v, repeated[k]) for k,v in end.items())
        assert np.array_equal(repeated['syn'][5], Z)
        rows.append(dict(label=c['label'], full_state_replay_bitwise=True,
                         full_initial_history_and_M_preserved=True,
                         Z_fixed=True, M_dynamic=True))
    write(DEST/f'{field}_pair_implementation.json', dict(status='PASS', rows=rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['register', 'check', 'run'])
    p.add_argument('--field', required=True)
    p.add_argument('--side', choices=['lower', 'upper'])
    p.add_argument('--lower-source',default='lower_endpoint')
    p.add_argument('--upper-source',default='upper_endpoint')
    p.add_argument('--device', type=int, default=0)
    a = p.parse_args()
    if a.command == 'register':
        register(a.field,a.lower_source,a.upper_source)
    elif a.command == 'check':
        check(a.field, a.device)
    else:
        assert a.side is not None
        assert read(DEST/f'{a.field}_pair_implementation.json')['status'] == 'PASS'
        run(f'{a.field}_from_{a.side}', a.device)
