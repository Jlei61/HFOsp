"""Unchanged native replay: group input moments, not a fitted rate trajectory."""
from native_same_history_feedback import native, ROOT
from datetime import datetime
from pathlib import Path
import argparse, os, time

np = native.np
OUT = ROOT / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'
DEST = OUT / 'native_input_bridge'
GEO = ROOT / 'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g20/geometry.npz'
NAMES = ['ampa', 'ampa2', 'gaba', 'gaba2', 'zgaba', 'zgaba2',
         'mcurrent', 'mcurrent2', 'net', 'net2', 'voltage', 'voltage2',
         'z', 'z2', 'ampa_gaba', 'ampa_zgaba', 'ampa_m', 'zgaba_m',
         'net_voltage', 'below_reset']


def register():
    DEST.mkdir(exist_ok=True)
    assert not (DEST / 'contract.json').exists()
    g = dict(np.load(GEO)); selected = [279, 141, 39, 594]
    for region in [0, 1]:
        ix = np.flatnonzero((g['population'] == 0) & (g['group_region'] == region) & (g['group_size'] >= 10))
        selected.append(int(ix[np.argmin(g['threshold_mv'][ix])]))
    native.write(DEST / 'contract.json', dict(created_local=datetime.now().astimezone().isoformat(),
        design=str(OUT / 'next_native_input_bridge.md'), design_sha256=native.sha(OUT / 'next_native_input_bridge.md'),
        status='REGISTERED_BEFORE_RECORDING', start_ms=8000, stop_ms=10370, dt_ms=.1,
        groups=935, geometry_sha256=native.sha(GEO), moment_names=NAMES,
        selected_groups=selected,
        selection='Four prior groups plus each core minimum threshold group having at least10original cells; geometry only.',
        primary_window_ms=[9000, 10370], preparation_ms=[8000, 9000],
        fixed_count_windows_ms=50, native_sampling='Single native trajectory; group/time windows are descriptive, not independent trials.',
        ordering='Current observer before membrane and slow updates; spike observer after this step threshold and reset. Pre-step Z/M apply to recorded currents.',
        acceptance='Four complete original500ms chunks and full10370ms engine bitwise identical; every group count sum agrees with native count records.',
        scope='Input/response localization only, no autonomous correspondence claim or new bifurcation classification.',
        budget='One2.370s replay. No parameter or RNG change; no new training or response fit.'))
    print('REGISTERED', selected, flush=True)


def run(device):
    c = native.read(DEST / 'contract.json'); assert native.sha(GEO) == c['geometry_sha256']
    assert not (DEST / 'replay_audit.json').exists()
    reference = native.check_reference_sources()
    assert native.read(native.REPLAY_RUN / 'replay_qa.json')['status'] == 'PASS'
    start, stop = c['start_ms'], c['stop_ms']; state = native.replay_checkpoint(start)
    geo = dict(np.load(GEO)); groups = geo['cell_group']; sizes = geo['group_size']; P = len(sizes)
    assert P == c['groups'] and np.array_equal(np.bincount(groups), sizes)
    def avg(v): return np.bincount(groups, weights=v, minlength=P) / sizes
    initial = {}
    for prefix, qkey, ikey in [('ampa', 's_E', 'I_E'), ('gaba', 's_I', 'I_I')]:
        q, i = state[qkey], state[ikey]
        initial.update({prefix+'_q': avg(q), prefix+'_i': avg(i), prefix+'_q2': avg(q*q),
                        prefix+'_qi': avg(q*i), prefix+'_i2': avg(i*i)})
    np.savez_compressed(DEST / 'initial_moments.npz', **initial,
        z=avg(state['slow']['z']), m=avg(state['slow']['m']), group_size=sizes, start_step=state['step'])
    name = 'native_t8000_inputs_observe'
    (DEST / 'jobs').mkdir(exist_ok=True)
    protocol = dict(identity=reference['identity'], source_hashes=reference['source_hashes'],
                    contract=str(DEST / 'contract.json'), geometry_sha256=native.sha(GEO))
    native.write(DEST / 'protocol.json', protocol)
    job = native.make_job(name, start, start, 'W1', duration_ms=stop-start)
    job.update(start_step=start*10, anchor_ms=start, horizon_s=stop/1000,
               freeze_z=False, freeze_m=False, state_construction='Complete native checkpoint, input observation only')
    native.write(DEST / 'jobs' / (name+'.json'), job)
    folder = DEST / 'runs' / name; folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'checkpoint.pkl'; assert not checkpoint.exists(), 'No silent replay resume'
    tracker = native.core.fresh_tracker(); tracker['stop_s'] = 1e9
    native.save_pickle(checkpoint, dict(job=job, identity=protocol['identity'], engine=state,
                                       tracker=tracker, restore_from=None))
    assert native.compare_states(native.load_pickle(checkpoint)['engine'], state) == []
    native.write(folder / 'continuation.json', dict(source=str(native.REPLAY_RUN/'checkpoints/t8000ms.npz'),
        initial_state_bitwise_identical=True, clock_rebased=False, random_streams_replaced=False,
        Z_dynamic=True, M_dynamic=True, metadata_written_after_simulation=False))
    native.write(DEST / 'progress.json', dict(status='RUNNING', pid=os.getpid(), time_ms=start,
        Z_dynamic=True, M_dynamic=True, source=str(native.REPLAY_RUN)))
    original = native.core.old.simulate_kick
    globals()['compute_nu_theta'] = original.__globals__['compute_nu_theta']

    def observer(params, net, *args, **kwargs):
        assert np.array_equal(net['pos'], geo['original_positions'])
        slow = kwargs['slow']; previous_current = kwargs.get('current_observer')
        previous_spike = kwargs['spike_observer']; previous_sink = kwargs['checkpoint_sink']
        previous_input = kwargs.get('input_observer')
        chunk_start = int(kwargs['resume_state']['step']); assert chunk_start == start*10
        n = 0; nsp = 0; ni = 0
        moments = np.empty((5000, len(NAMES), P), dtype=np.float64)
        counts = np.empty((5000, P), dtype=np.uint16)
        drive = np.empty((5000, P), dtype=np.float64)
        times = np.empty(5000); spike_times = np.empty(5000); input_times = np.empty(5000)
        theta = kwargs.get('V_th_per_neuron')
        np.savez_compressed(DEST / 'membership.npz', **geo,
                            actual_threshold_mv=np.broadcast_to(params.V_th if theta is None else theta, (len(groups),)))
        def current(tm, ie, ii, voltage):
            nonlocal n
            if previous_current is not None: previous_current(tm, ie, ii, voltage)
            z = slow.z; m = slow.cfg.eta_m * slow.m; zi = z*ii; neti = ie-zi-m
            vals = [ie, ie*ie, ii, ii*ii, zi, zi*zi, m, m*m, neti, neti*neti,
                    voltage, voltage*voltage, z, z*z, ie*ii, ie*zi, ie*m, zi*m,
                    neti*voltage, voltage < params.V_reset]
            for j, v in enumerate(vals): moments[n, j] = avg(v)
            times[n] = tm; n += 1
        def inputs(tm, nu, xi):
            nonlocal ni
            if previous_input is not None: previous_input(tm, nu, xi)
            drive[ni] = avg(nu); input_times[ni] = tm; ni += 1
        def spikes(tm, spk):
            nonlocal nsp
            previous_spike(tm, spk)
            x = np.bincount(groups, weights=spk, minlength=P)
            assert np.all(x <= sizes) and np.all(x == x.astype(np.uint16))
            counts[nsp] = x; spike_times[nsp] = tm; nsp += 1
        def sink(k, engine):
            nonlocal chunk_start, n, nsp, ni
            try: previous_sink(k, engine)
            finally:
                assert n == nsp == ni == k-chunk_start
                assert np.array_equal(times[:n], spike_times[:n]) and np.array_equal(times[:n], input_times[:n])
                dest = folder / 'inputs'; dest.mkdir(exist_ok=True)
                np.savez_compressed(dest / f'{chunk_start:010d}_{k:010d}.npz',
                    time_ms=times[:n], moments=moments[:n], spikes=counts[:n], external_rate_per_ms=drive[:n],
                    start_step=chunk_start, end_step=k)
                native.write(DEST / 'progress.json', dict(status='RUNNING', pid=os.getpid(), time_ms=k/10, chunks_closed=True))
                print('INPUT CHUNK', chunk_start, k, flush=True)
                n = nsp = ni = 0; chunk_start = k
        kwargs.update(current_observer=current, input_observer=inputs, spike_observer=spikes, checkpoint_sink=sink)
        return original(params, net, *args, **kwargs)

    native.core.old.simulate_kick = observer
    native.NATIVE = DEST; native.prepare = lambda: protocol; native.seed_folder = lambda job, device: folder
    result = native.worker(name, device); assert result['status'] == 'COMPLETE', result
    audit()


def audit():
    c=native.read(DEST/'contract.json');start,stop=c['start_ms'],c['stop_ms'];P=c['groups']
    folder=DEST/'runs/native_t8000_inputs_observe';checkpoint=folder/'checkpoint.pkl'
    result=native.read(folder/'result.json');assert result['status']=='COMPLETE'
    differences = native.compare_states(native.load_pickle(checkpoint)['engine'], native.replay_checkpoint(stop))
    assert differences == [], differences
    rows = []
    for f in sorted((folder / 'chunks').glob('*.npz')):
        source = native.REPLAY_RUN / 'chunks' / f.name
        if not source.exists(): continue
        with np.load(f) as x, np.load(source) as y:
            mismatch = [key for key in y.files if key not in x or x[key].dtype != y[key].dtype or not np.array_equal(x[key], y[key])]
            assert not mismatch, mismatch
            rows.append(dict(chunk=f.name, common_keys=len(y.files), bitwise=True))
    assert len(rows) == 4
    samples = 0
    for f in sorted((folder / 'inputs').glob('*.npz')):
        with np.load(f) as z:
            assert np.allclose(np.diff(z['time_ms']), .1, rtol=0, atol=1e-10)
            assert z['moments'].dtype == np.float64 and np.isfinite(z['moments']).all()
            samples += len(z['time_ms'])
    assert samples == (stop-start)*10
    native.write(DEST / 'replay_audit.json', dict(status='PASS', final_engine_difference_keys=differences,
        original_chunk_comparisons=rows, samples=samples, groups=P, selected_groups=c['selected_groups'],
        scope='Observer replay identity; downstream count/readout and moment reconstruction checks pending.'))
    native.write(DEST / 'progress.json', dict(status='COMPLETE', time_ms=stop, pid=os.getpid()))
    print('INPUT OBSERVER REPLAY PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['register', 'run', 'audit']); p.add_argument('--device', type=int, default=1)
    a = p.parse_args(); {'register':register,'audit':audit,'run':lambda:run(a.device)}[a.command]()
