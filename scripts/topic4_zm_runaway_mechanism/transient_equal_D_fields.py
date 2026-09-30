"""Same mean depletion, different measured spatial fields, paired fast history."""
from common import OUT, model, np, read, write, log
from physical_delay_count_rate import PhysicalDelayCountEngine, projections
from transient_response_network import install, LABEL
from fine_rate_frozen_Z_fields import restore, capture
from native_readouts import readouts, window_stats
from datetime import datetime
import argparse
import os
import time

FREE = OUT/'transient_response_network_20260923'
PRESCRIBED = OUT/'transient_native_Z_path_20260923'
DEST = OUT/'transient_equal_D_fields_20260923'


def at_first_crossing(Z, t, s, target):
    D = 1-Z[:, s.E]@s.mean_weights
    i = int(np.flatnonzero(D >= target)[0])
    assert i > 0 and D[i] > D[i-1]
    a = (target-D[i-1])/(D[i]-D[i-1])
    z = (1-a)*Z[i-1]+a*Z[i]
    assert abs(1-z[s.E]@s.mean_weights-target) < 1e-12
    assert z.min() >= 0 and z.max() <= 1
    return z, dict(sample_bracket_ms=[float(t[i-1]), float(t[i])],
        interpolation_weight=float(a), interpolated_time_ms=float((1-a)*t[i-1]+a*t[i]))


def prepare():
    assert read(PRESCRIBED/'jobs.json')['status'] == 'COMPLETE'
    DEST.mkdir(exist_ok=True)
    assert not (DEST/'contract.json').exists()
    write(DEST/'contract.json', dict(created_local=datetime.now().astimezone().isoformat(),
        question='At exactly the same meanD=.20, does the excessive core depletion in the free rate trajectory change the conditional state relative to the native spatial field?',
        rationale='Native-Z-driven fixed rate model recovered entry9.844s versus native9.8685s and both lead directions; autonomousZ version enters7.674s and overdepletes cores at matchedmeanD. This is a prospective paired spatial-field intervention.',
        mean_D=.20, parameter_selection='One shared depletion within the previously identified preentry interval; native and free fields both cross it. No scanning for a favorable field.',
        field_construction='First observed upward crossing of meanD=.20 in each completed path. Linear interpolation of the entire groupZ between adjacent stored samples, using originalEcell weights; spatial pattern unaltered otherwise.',
        initialization='Exactly the same complete9000ms fast/local/refractory/delay/M state from the completed nativeZ-driven rate run. Intervene only in fullZ field and freeze it. InitialM and all fast histories identical.',
        future='Original externalinputclock9-12.5s and originalseed1 Philox(group,tick) innovation keys shared. M dynamic; Z held in both arms.',
        arms=['native_shape', 'free_rate_shape'], duration_ms=3500, dt_ms=.05,
        observable='Original10ms rate quiet<5Hz, completeevents requiring20ms quiet both sides, high>=200Hz for200ms, final1s globalmean and spatial persistent fraction >50Hz in>=90percentofbins.',
        interpretation='Different outcomes establish spatialZ-pattern dependence at fixedmean in this finite-time comparison. This alone is not a separatrix or a bifurcation type. Same outcomes would reject these two shapes as sufficient to explain the observed discrepancy under this chosen history.',
        budget='Two paired3.5s trajectories, checkpoint replay check, original readouts and one comparison figure. No fit, newseed, extension or branch.',
        model_promoted=False, original_local_failures_retained=True))
    s = model(40)
    native = np.load(PRESCRIBED/'native_Z_path.npz')
    free = np.load(FREE/LABEL/'trajectory.npz')
    fields, rows = {}, []
    for name, z, t in [('native_shape', native['Z'], native['time_ms']),
                       ('free_rate_shape', free['Z'].astype(float), free['state_time_ms'])]:
        field, info = at_first_crossing(z, t, s, .20)
        fields[name] = field
        info.update(arm=name, D=float(1-field[s.E]@s.mean_weights),
            regional_Z={name:float(np.average(field[s.E&(s.geo['group_region']==k)], weights=s.sizes[s.E&(s.geo['group_region']==k)]))
                for k, name in enumerate(['Core A', 'Core B', 'Surround'])})
        rows.append(info)
    np.savez_compressed(DEST/'fields.npz', **fields)
    write(DEST/'field_construction.json', dict(status='PASS', rows=rows,
        weighted_spatial_RMS=float(np.sqrt((fields['native_shape'][s.E]-fields['free_rate_shape'][s.E])**2@s.mean_weights))))
    log('EQUAL D FIELDS', rows)


def check(device):
    from transient_native_Z_path import PrescribedZEngine
    source = np.load(PRESCRIBED/LABEL/'checkpoint9000.npz')
    e = PrescribedZEngine(seed=1, device=device)
    install(e)
    e.graph()
    restore(e, source)
    reference = np.load(PRESCRIBED/LABEL/'trajectory.npz')
    x = e.chunk()
    assert np.array_equal(x[:, 0].astype('f4'), reference['group_rate_hz'][9000:9010])
    assert np.array_equal(x[:, 1].astype('f4'), reference['group_expected_rate_hz'][9000:9010])
    # Same complete history and seed; a constant Z prescription is the held-field flow.
    held = source['syn'][5]
    restore(e, source)
    e.Z_path[:] = e.cp.asarray(held)
    e.chunk()
    target = capture(e)
    base = PhysicalDelayCountEngine(seed=1, device=device)
    install(base)
    base.graph()
    restore(base, source)
    base.transport.pars[19].fill(0)
    base.chunk()
    actual = capture(base)
    errors = {k:float(np.max(abs(actual[k]-v))) for k,v in target.items()}
    assert max(errors.values()) < 1e-8
    write(DEST/'implementation_check.json', dict(status='PASS',
        prescribed_reference10ms_saved_precision_bitwise=True,
        constant_prescription_vs_held_flow_max_errors=errors,
        complete_initial_history_preserved=True, M_dynamic=True))
    log('EQUAL D IMPLEMENTATION PASS')


def run(device):
    c = read(DEST/'contract.json')
    assert read(DEST/'implementation_check.json')['status'] == 'PASS'
    assert not (DEST/'jobs.json').exists()
    jobs = dict(status='RUNNING', pid=os.getpid(), completed=[])
    write(DEST/'jobs.json', jobs)
    e = PhysicalDelayCountEngine(seed=1, device=device)
    install(e)
    e.graph()
    source = np.load(PRESCRIBED/LABEL/'checkpoint9000.npz')
    fields = np.load(DEST/'fields.npz')
    P, count = projections(e.s, e.coarse, e.parent)[20]
    results = []
    try:
        for arm in c['arms']:
            restore(e, source)
            e.syn[5] = e.cp.asarray(fields[arm])
            e.transport.pars[19].fill(0)
            assert np.array_equal(e.syn[4].get(), source['syn'][4])
            assert np.array_equal(e.transport.history.get(), source['history'])
            assert np.array_equal(e.local.state.get(), source['local'])
            e.cp.cuda.get_current_stream().synchronize()
            folder = DEST/arm
            folder.mkdir()
            R, M = [], []
            start = time.time()
            for k in range(350):
                x = e.chunk()
                assert np.isfinite(x).all() and x[:, 0].min() >= 0
                assert np.array_equal(e.syn[5].get(), fields[arm])
                R.append(x)
                M.append(e.syn[4].get())
                if (k+1) % 50 == 0:
                    jobs.update(arm=arm, elapsed_ms=(k+1)*10)
                    write(DEST/'jobs.json', jobs)
                    log('EQUAL D', arm, (k+1)*10, 'ms', round(time.time()-start, 1), 's')
            r = np.concatenate(R)
            field = (P@r[:, 0].T).T
            t = np.arange(9001, 12501.)
            events, summary, whole, sm = readouts(t, field, count, arm)
            complete = []
            for event in events:
                a = int(np.searchsorted(t, event['start_ms']))
                b = a+int(event['duration_ms'])
                if a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all():
                    complete.append(event)
            result = dict(arm=arm, D=.20, Z_held=True, M_dynamic=True,
                high_entry_ms=summary['high_onset_ms'], quiet_fraction=summary['quiet_fraction'],
                complete_events=window_stats(complete, 9000, 12500),
                tail_global_hz=float(whole[-1000:].mean()),
                tail_persistent_fraction=float((count/count.sum())[(field[-1000:]>50).mean(0)>=.9].sum()),
                seconds=time.time()-start, same_initial_fast_and_M=True, same_future_innovation_keys=True)
            np.savez_compressed(folder/'trajectory.npz', time_ms=t,
                group_rate_hz=r[:, 0].astype('f4'), group_expected_rate_hz=r[:, 1].astype('f4'),
                field_E_hz=field.astype('f4'), global_E_hz=whole, cell_counts=count,
                state_time_ms=np.arange(9010, 12501., 10), M_current=np.array(M), Z=fields[arm])
            np.savez_compressed(folder/'final_state.npz', **capture(e))
            write(folder/'result.json', result)
            results.append(result)
            jobs['completed'].append(arm)
            write(DEST/'jobs.json', jobs)
            log('EQUAL D RESULT', result)
        write(DEST/'result.json', dict(status='COMPLETE', rows=results,
            statistical_unit='One complete fast/M history, two paired fullZ fields; one shared future noise realization.',
            scope='Finite-time spatial-field sensitivity at fixedmeanD, not a certified attractor, separatrix or bifurcation.',
            model_promoted=False, bifurcation_type='NOT_ESTABLISHED'))
        jobs['status'] = 'COMPLETE'
        write(DEST/'jobs.json', jobs)
    except BaseException as error:
        jobs.update(status='FAILED', error=repr(error))
        write(DEST/'jobs.json', jobs)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'check', 'run'])
    p.add_argument('--device', type=int, default=1)
    a = p.parse_args()
    {'prepare':prepare, 'check':lambda:check(a.device), 'run':lambda:run(a.device)}[a.command]()
