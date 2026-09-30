"""Bounded autonomous conditional-drift probe at actual native spatial Z fields.

Counts innovations are suppressed after constructing the physical private-Q
engine. This is not the full-Q expected-rate arm or an exact ensemble mean.
No classification here certifies a bifurcation or repairs the free Z path.
"""
from common import OUT, model, np, read, write, log
from transient_response_network import install, LABEL
from physical_delay_count_rate import PhysicalDelayCountEngine, projections
from fine_rate_frozen_Z_fields import native_field, restore, capture
from native_readouts import readouts
from datetime import datetime
from scipy.signal import find_peaks
import argparse
import os
import time

SOURCE = OUT/'transient_native_Z_path_20260923'
DEST = OUT/'transient_autonomous_Z_probe_20260923'


def register():
    assert read(SOURCE/'scientific_comparison.json')['prescribed_Z_saved_max_error'] < 1e-7
    assert read(OUT/'transient_equal_D_fields_20260923/independent_audit.json')['status'] == 'PASS'
    DEST.mkdir(exist_ok=True)
    assert not (DEST/'contract.json').exists()
    write(DEST/'contract.json', dict(created_local=datetime.now().astimezone().isoformat(),
        question='At native preentry/entry fullZ fields, does the autonomous conditional drift show equilibrating or recurrent activity, providing actual same-equation seeds for subsequent bifurcation analysis?',
        mathematical_object='PhysicalDelayCountEngine with locked transient correction; suppress future count sampling but retain physical-delay private-Q. Replace external OU by original constant mean input. This is explicitly the finite-count model conditional drift, not the separate full-Q mean-field or stochastic ensemble mean.',
        empirical_status='NativeZ-driven fast/M correspondence passes the five applicable originalA4 criteria; autonomousZ path and original local response gates still FAIL. This mathematical exploration does not promote the model or certify native onset.',
        Z='Entire actualg40 native9420 and9870ms fields held. D=1-originalEcount-weightedZ. No uniformZ replacement.',
        M='Dynamic in both runs, sameinitialM and fullfast/local/delay/history from the completed nativeZ-driven9000ms checkpoint.',
        input='Both exogenous noise and future count innovations disabled explicitly; inherited noisy initial history retained and allowed to relax. Rates are continuous expected flux, with own renewal history.',
        fields_ms=[9420, 9870], duration_ms=5000, dt_ms=.05,
        readout='Final2s global and regional rates, complete spatial field, peak spacing, state variation and exact static residual at tailmean. Finite-time convergence/recurrent-waveform diagnostics only.',
        budget='Two5s fixed-field deterministic trajectories and numerical/field audits. No parameter scan, fit, Floquet or critical label under this registration.',
        followup='Only after actual residual and response-domain checks may a resulting state seed same-equation equilibrium or periodic continuation. Solver failure or loss of a finite event is never a named bifurcation.',
        model_promoted=False))
    s = model(40)
    np.savez_compressed(DEST/'fields.npz', **{str(tm):native_field(s, tm) for tm in [9420, 9870]})


def check(device):
    e = PhysicalDelayCountEngine(seed=1, device=device, count_sampling=False, constant_input=True)
    install(e)
    e.graph()
    checkpoint = np.load(SOURCE/LABEL/'checkpoint9000.npz')
    restore(e, checkpoint)
    e.syn[5] = e.cp.asarray(np.load(DEST/'fields.npz')['9420'])
    e.transport.pars[19].fill(0)
    assert not e.noise and not e.transport.drive_on
    state = capture(e)
    a = e.chunk()
    terminal = capture(e)
    restore(e, state)
    b = e.chunk()
    assert np.array_equal(a, b)
    assert np.array_equal(b[:, 0], b[:, 1])
    assert all(np.array_equal(v, capture(e)[k]) for k,v in terminal.items())
    assert np.array_equal(e.syn[5].get(), state['syn'][5])
    assert np.all(e.transport.pars[20].get() == 1)
    write(DEST/'implementation_check.json', dict(status='PASS',
        exact_checkpoint_replay=True, no_count_innovations=True, emitted_equals_expected=True,
        constant_original_mean_external_drive=True, physical_private_Q_retained=True,
        Z_held=True, M_dynamic=True))
    log('AUTONOMOUS CONDITIONAL PROBE CHECK PASS')


def run(device):
    c = read(DEST/'contract.json')
    assert read(DEST/'implementation_check.json')['status'] == 'PASS'
    assert not (DEST/'jobs.json').exists()
    jobs = dict(status='RUNNING', pid=os.getpid(), completed=[])
    write(DEST/'jobs.json', jobs)
    e = PhysicalDelayCountEngine(seed=1, device=device, count_sampling=False, constant_input=True)
    install(e)
    e.graph()
    source = np.load(SOURCE/LABEL/'checkpoint9000.npz')
    fields = np.load(DEST/'fields.npz')
    P, count = projections(e.s, e.coarse, e.parent)[20]
    rows = []
    try:
        for tm in c['fields_ms']:
            label = str(tm)
            folder = DEST/label
            folder.mkdir()
            restore(e, source)
            e.syn[5] = e.cp.asarray(fields[label])
            e.transport.pars[19].fill(0)
            e.cp.cuda.get_current_stream().synchronize()
            assert not e.noise and not e.transport.drive_on
            rates, M = [], []
            start = time.time()
            for k in range(500):
                x = e.chunk()
                assert np.isfinite(x).all() and x.min() > -1e-9
                assert np.array_equal(x[:, 0], x[:, 1])
                assert np.array_equal(e.syn[5].get(), fields[label])
                rates.append(x[:, 0])
                M.append(e.syn[4].get())
                if (k+1) % 100 == 0:
                    jobs.update(field_native_time_ms=tm, elapsed_ms=(k+1)*10)
                    write(DEST/'jobs.json', jobs)
                    log('AUTONOMOUS PROBE', tm, (k+1)*10, 'ms', round(time.time()-start, 1), 's')
            r = np.concatenate(rates)
            field = (P@r.T).T
            whole = r[:, e.s.E]@e.s.mean_weights
            t = np.arange(1, 5001.)
            peaks, _ = find_peaks(whole[-2000:], prominence=5, distance=10)
            intervals = np.diff(peaks)
            tail = r[-2000:]
            mean = tail.mean(0)
            variation = float(np.linalg.norm(tail-mean)/max(np.linalg.norm(tail), 1.))
            row = dict(native_Z_time_ms=tm, D=float(1-fields[label][e.s.E]@e.s.mean_weights),
                tail_global_mean_hz=float(whole[-2000:].mean()),
                tail_global_range_hz=[float(whole[-2000:].min()), float(whole[-2000:].max())],
                tail_group_relative_variation=variation, peaks_in_final2s=len(peaks),
                median_peak_interval_ms=float(np.median(intervals)) if len(intervals) else None,
                peak_interval_CV=float(np.std(intervals)/np.mean(intervals)) if len(intervals)>1 else None,
                tail_persistent_fraction=float((count/count.sum())[(field[-1000:]>50).mean(0)>=.9].sum()),
                scope='Five-second transient probe; no certified equilibrium/periodic/stability label.',
                seconds=time.time()-start)
            np.savez_compressed(folder/'trajectory.npz', elapsed_time_ms=t,
                group_rate_hz=r.astype('f4'), field_E_hz=field.astype('f4'), global_E_hz=whole,
                cell_counts=count, state_time_ms=np.arange(10, 5001., 10),
                M_current=np.array(M), Z=fields[label], mean_tail_rate_per_ms=mean/1000)
            np.savez_compressed(folder/'final_state.npz', **capture(e))
            write(folder/'result.json', row)
            rows.append(row)
            jobs['completed'].append(tm)
            write(DEST/'jobs.json', jobs)
            log('AUTONOMOUS PROBE RESULT', row)
        write(DEST/'result.json', dict(status='COMPLETE', rows=rows,
            model_promoted=False, bifurcation_type='NOT_ESTABLISHED'))
        jobs['status'] = 'COMPLETE'
        write(DEST/'jobs.json', jobs)
    except BaseException as error:
        jobs.update(status='FAILED', error=repr(error))
        write(DEST/'jobs.json', jobs)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['register', 'check', 'run'])
    p.add_argument('--device', type=int, default=1)
    a = p.parse_args()
    {'register':register, 'check':lambda:check(a.device), 'run':lambda:run(a.device)}[a.command]()
