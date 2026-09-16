#!/usr/bin/env python3
"""Paired backend continuation of an actual revised-Z checkpoint."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import argparse, copy, time
import numpy as np
import run_topic4_continuous_resource_recovery as run


def main(name, duration_ms=200, output_tag=None):
    source = run.OUT / 'runs' / name / 'checkpoint.pkl'
    saved = run.carrier.base.load_pickle(source)
    job, state = saved['job'], saved['engine']
    first, duration = int(state['step']), round(duration_ms * 10)
    assert duration >= 2000
    last = first + duration
    setup, tr, frozen, identity = run.carrier.base.old.setup(job['seed'])
    assert identity == saved['identity']
    outputs, timings = {}, {}
    capture0, restore0 = run.checkpoint.capture, run.checkpoint.restore_slow
    def capture(**kw):
        value = capture0(**kw)
        value['slow_global_pool_rate'] = kw['slow'].pool_rate
        return value
    def restore(value, slow):
        restore0(value, slow)
        slow.pool_rate = float(value['slow_global_pool_rate'])
    run.checkpoint.capture, run.checkpoint.restore_slow = capture, restore
    try:
        for mode in ['cpu', 'gpu']:
            cfg = run.carrier.base.old.MZSlowVarsConfig(use_z=True, use_m=True,
                tau_z=job['tau_Z_s'] * 1000, I_th_EI=job['threshold'],
                tau_adp=job['tau_M_s'] * 1000, eta_m=job['eta_m'])
            slow = run.ResourceRecoverySlow(setup.n_e + setup.n_i, setup.params.V_th,
                cfg, NE=setup.n_e, mode=job['mode'], gamma=job['gamma'],
                recovery_ratio=job['recovery_ratio'], pool_gain=job['pool_gain'],
                pool_threshold_Hz=job['pool_threshold_Hz'], pool_tau_s=job['pool_tau_s'])
            setup.net['rng'] = np.random.default_rng(job['seed'])
            drive = run.carrier.base.old.make_external_drive(setup, tr['spatial_ou'], job['seed'])
            captured, spikes, times = {}, [], {}
            def observe(tm, spk):
                k = round(tm / .1) - first
                spikes.append(spk.copy())
                if k in [200, duration - 200]:
                    times[k] = time.perf_counter()
            def sink(k, engine):
                captured.update(engine)
            params = copy.deepcopy(setup.params)
            params.T = (duration + 1) * .1
            fn = run.carrier.base.old.simulate_kick
            if mode == 'gpu':
                fn = run.carrier.wrap_simulator(fn, device_index=job['device'])
            tic = time.perf_counter()
            fn(params, setup.net, KICK_BOOST=0., slow=slow, V_th_per_neuron=setup.vtheta,
               external_e_rate_drive=drive, early_stop_runaway=False, spike_observer=observe,
               record_dense_spikes=False, fast_scatter=True, resume_state=copy.deepcopy(state),
               time_offset_ms=first * .1, checkpoint_steps={last}, checkpoint_sink=sink, verbose=False)
            outputs[mode] = (captured, np.asarray(spikes), np.asarray(slow.pool_records),
                             np.asarray(slow.recovery_records))
            timings[mode] = dict(total_wall_s=time.perf_counter()-tic,
                                 middle_duration_ms=duration_ms-40,
                                 middle_wall_s=times[duration-200]-times[200])
            print(mode, timings[mode], flush=True)
    finally:
        run.checkpoint.capture, run.checkpoint.restore_slow = capture0, restore0
    def equal(a, b, path='checkpoint'):
        assert type(a) is type(b), path
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b, equal_nan=True), path
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), path
            for k in a:
                equal(a[k], b[k], path + '.' + str(k))
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b), path
            for k, (x, y) in enumerate(zip(a, b)):
                equal(x, y, path + '.' + str(k))
        else:
            assert a == b, path
    equal(outputs['cpu'], outputs['gpu'])
    spikes = outputs['cpu'][1][:duration, :32000]
    rate20 = spikes.reshape(-1, 200, 32000).sum(axis=(1, 2)) / 32000 / .02
    report = dict(status='PASS', source=str(source), source_step=first, duration_ms=duration_ms,
        job=job, entire_checkpoint_recursive_bitwise=True, full_40000_neuron_spike_matrix_bitwise=True,
        Z_M_pool_resource_flux_and_RNG_bitwise=True, global_pool_rate_restored=True,
        timings=timings, speed_ratio_gpu_over_cpu=timings['gpu']['middle_wall_s']/timings['cpu']['middle_wall_s'],
        mean_E_rate_Hz=float(spikes.sum()/32000/(duration_ms/1000)),
        E_rates_20ms_Hz=rate20.tolist(),
        scope='Actual saved-state continuation, backend QA only. No source worker or source checkpoint is changed; not an independent science replicate. Timing is state/load dependent; full population rates document whether the test included bursts.')
    suffix = f'_{output_tag}' if output_tag else ''
    run.carrier.base.write(run.OUT / 'backend_benchmarks' / (name + suffix + f'_{duration_ms:g}ms.json'), report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--duration-ms', type=float, default=200)
    p.add_argument('--output-tag', help='Keep a new operating-state benchmark separate from earlier evidence.')
    args = p.parse_args()
    main(args.name, args.duration_ms, args.output_tag)
