"""Read-only native membrane observation with exact replay acceptance."""
from native_same_history_feedback import native, ROOT
from datetime import datetime
import argparse
import time

np = native.np
OUT = ROOT / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'
DEST = OUT / 'native_voltage_observation'
GEOMETRY = ROOT / 'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g20/geometry.npz'


def register():
    path = OUT / 'native_voltage_observation_contract.json'
    assert not path.exists()
    native.write(path, dict(created_local=datetime.now().astimezone().isoformat(),
        question='Does the unchanged native pre-onset trajectory show hyperpolarized recovery in the previously selected surround subgroup and in regional summaries?',
        start_ms=9000, stop_ms=10370, sample_interval_ms=1, selected_group=39,
        populations=['Core A E', 'Core B E', 'Surround E', 'Selected surround E group39'],
        observables=['Pre-membrane-update V and V squared', 'Fraction V below reset',
                     'AMPA, raw GABA, Z times GABA, eta_M times M, net current', 'Z', 'Spikes each native step'],
        ordering='Current observer sees this step currents and previous post-step V, with pre-slow-update Z/M. Spike observer sees this step spikes. Absolute clocks retained; no phase alignment or time shifting.',
        acceptance='All original arrays in first two complete 500ms chunks and complete final engine state at10370ms bitwise equal to reference; physical initial state also equal.',
        statistical_unit='One native history; neurons, time bins and successive events are descriptive repeated observations, not independent trajectories.',
        budget='One1.37s observe-only native replay; no parameter change, fitting, new model or bifurcation scan.',
        scope='Native membrane/input description. Does not validate a population closure or identify a bifurcation; no input waveform from a reduced model is supplied.'))


def main(a):
    c = native.read(OUT / 'native_voltage_observation_contract.json')
    reference = native.check_reference_sources()
    assert native.read(native.REPLAY_RUN / 'replay_qa.json')['status'] == 'PASS'
    start, stop = c['start_ms'], c['stop_ms']
    state = native.replay_checkpoint(start)
    geo = dict(np.load(GEOMETRY))
    groups = geo['cell_group'][:native.NE]
    region = geo['group_region'][groups]
    masks = [region == k for k in range(3)] + [groups == c['selected_group']]
    assert all(m.any() for m in masks)
    assert np.all(region[masks[-1]] == 2)
    name = 'native_t9000_voltage_observe'
    DEST.mkdir(exist_ok=True); (DEST / 'jobs').mkdir(exist_ok=True)
    protocol = dict(identity=reference['identity'], source_hashes=reference['source_hashes'],
                    contract=str(OUT / 'native_voltage_observation_contract.json'),
                    geometry=str(GEOMETRY), geometry_sha256=native.sha(GEOMETRY))
    native.write(DEST / 'protocol.json', protocol)
    job = native.make_job(name, start, start, 'W1', duration_ms=stop-start)
    job.update(start_step=start*10, anchor_ms=start, horizon_s=stop/1000,
               freeze_z=False, freeze_m=False, state_construction='Complete native checkpoint, observation only')
    job_path = DEST / 'jobs' / (name+'.json')
    if job_path.exists(): assert native.read(job_path) == job
    else: native.write(job_path, job)
    folder = DEST / 'runs' / name; folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'checkpoint.pkl'
    if not checkpoint.exists():
        tracker = native.core.fresh_tracker(); tracker['stop_s'] = 1e9
        native.save_pickle(checkpoint, dict(job=job, identity=protocol['identity'], engine=state,
                                           tracker=tracker, restore_from=None))
        assert native.compare_states(native.load_pickle(checkpoint)['engine'], state) == []
        native.write(folder / 'continuation.json', dict(source=str(native.REPLAY_RUN / 'checkpoints/t9000ms.npz'),
            initial_state_bitwise_identical=True, clock_rebased=False, random_streams_replaced=False,
            Z_dynamic=True, M_dynamic=True, initial_global_Z=float(state['slow']['z'][:native.NE].mean()),
            seeded_at=time.time()))
    original = native.core.old.simulate_kick
    globals()['compute_nu_theta'] = original.__globals__['compute_nu_theta']

    def observe_simulator(params, net, *args, **kwargs):
        assert np.array_equal(net['pos'], geo['original_positions'])
        slow = kwargs['slow']; previous_current = kwargs.get('current_observer')
        previous_spike = kwargs['spike_observer']; previous_sink = kwargs['checkpoint_sink']
        times, moments, spike_times, spike_counts = [], [], [], []
        chunk_start = int(kwargs['resume_state']['step'])
        assert chunk_start == start*10, 'Partial observation replay needs explicit recovery, not silent truncation'
        def current(tm, ie, ii, voltage):
            if previous_current is not None: previous_current(tm, ie, ii, voltage)
            k = round(tm / params.dt)
            if k % 10: return
            z = slow.z[:native.NE]; m = slow.cfg.eta_m * slow.m[:native.NE]
            e, i, v = ie[:native.NE], ii[:native.NE], voltage[:native.NE]
            arrays = [v, v*v, v < params.V_reset, e, i, z*i, m, e-z*i-m, z]
            moments.append([[float(x[mask].mean()) for x in arrays] for mask in masks])
            times.append(tm)
        def spikes(tm, spk):
            previous_spike(tm, spk)
            spike_times.append(tm)
            spike_counts.append([int(spk[:native.NE][mask].sum()) for mask in masks])
        def sink(k, engine):
            nonlocal chunk_start
            try:
                previous_sink(k, engine)
            finally:
                dest = folder / 'voltage'; dest.mkdir(exist_ok=True)
                np.savez_compressed(dest / f'{chunk_start:010d}_{k:010d}.npz',
                    time_ms=times, moments=moments, spike_time_ms=spike_times, spikes=spike_counts,
                    cell_counts=[int(mask.sum()) for mask in masks], start_step=chunk_start, end_step=k)
                times.clear(); moments.clear(); spike_times.clear(); spike_counts.clear(); chunk_start=k
        kwargs['current_observer'] = current
        kwargs['spike_observer'] = spikes
        kwargs['checkpoint_sink'] = sink
        return original(params, net, *args, **kwargs)

    native.core.old.simulate_kick = observe_simulator
    native.NATIVE = DEST; native.prepare = lambda: protocol; native.seed_folder = lambda job, device: folder
    result = native.worker(name, a.device)
    assert result['status'] == 'COMPLETE'
    final = native.load_pickle(checkpoint)['engine']
    differences = native.compare_states(final, native.replay_checkpoint(stop))
    assert differences == [], differences
    rows = []
    for f in sorted((folder / 'chunks').glob('*.npz')):
        source = native.REPLAY_RUN / 'chunks' / f.name
        if not source.exists(): continue
        x, y = np.load(f), np.load(source)
        mismatch = [key for key in y.files if key not in x or x[key].dtype != y[key].dtype or not np.array_equal(x[key], y[key])]
        assert not mismatch, mismatch
        rows.append(dict(chunk=f.name, common_keys=len(y.files), bitwise=True))
    assert len(rows) == 2
    data = [np.load(f) for f in sorted((folder / 'voltage').glob('*.npz'))]
    t = np.concatenate([z['time_ms'] for z in data])
    st = np.concatenate([z['spike_time_ms'] for z in data])
    assert len(t) == 1370 and len(st) == 13700
    assert np.allclose(np.diff(t), 1, rtol=0, atol=1e-10)
    assert np.allclose(np.diff(st), .1, rtol=0, atol=1e-10)
    native.write(DEST / 'replay_audit.json', dict(status='NATIVE_VOLTAGE_REPLAY_PASS',
        final_engine_difference_keys=differences, rows=rows, voltage_samples=len(t), spike_steps=len(st),
        neuron_counts=[int(m.sum()) for m in masks], source='Original native reference; no reduced-model input',
        scope=c['scope']))
    print('NATIVE VOLTAGE REPLAY PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--register', action='store_true')
    p.add_argument('--device', type=int, default=0); a = p.parse_args()
    register() if a.register else main(a)
