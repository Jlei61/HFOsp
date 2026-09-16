#!/usr/bin/env python3
"""M0: exact replay of the two reference Fig.5 trajectories (<=12.5 s) with per-cell Z/M, inputs and checkpoints.

The reference producer chain is imported unchanged; this file only adds observers and extra checkpoint
times. Bitwise identity with the stored reference chunks and the reference 12.5-s checkpoint is the M0 gate.
"""
import argparse
import subprocess
import numpy as np
import psutil
from common import *  # noqa: F401,F403


def base_dir(canary):
    return OUT / ('replay_canary' if canary else 'replay')


def prepare(canary=False):
    base = base_dir(canary); base.mkdir(parents=True, exist_ok=True)
    if not (base / 'protocol.json').exists():
        p = check_reference_sources()
        write(base / 'protocol.json', dict(
            status='REPLAY_OF_REFERENCE', identity=p['identity'], source_hashes=p['source_hashes'],
            reference_protocol=str(REF / 'protocol.json'), reference_protocol_sha256=sha(REF / 'protocol.json'),
            producer=p['producer'], producer_sha256=p['producer_sha256'], wrapper=p['wrapper'], wrapper_sha256=p['wrapper_sha256'],
            tracker='verbatim copy of run_topic4_fig5_half_m_first_entry.first_only',
            replay_end_ms=REPLAY_END_MS, extra_checkpoints_ms=Z_TIMES_MS + [REPLAY_END_MS],
            per_cell_record='Z, M (float64) and raw I_E, I_I (float32) every 10 ms; every 5 ms in 8.8-10.4 s',
            input_record='global OU xi every step; 20x20 cell mean/variance of the applied E external rate every 1 ms; I external rate every 1 ms',
            canary=canary, resolved_modules=resolved_module_identity()))
    return read(base / 'protocol.json')


def worker(seed, canary=False):
    p = prepare(canary); check_reference_sources()
    job = reference_job(seed); assert job['seed'] == seed
    base = base_dir(canary); core.OUT = base; core.tracker_step = first_only_tracker
    folder = base / 'runs' / job['name']; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        return read(folder / 'result.json')
    stop_step = 10000 if canary else ms_to_step(REPLAY_END_MS)
    extra = {ms_to_step(ms) for ms in Z_TIMES_MS} | {stop_step}
    if canary:
        extra = {5000, stop_step}
    from src.topic4_cuda_ordered_scatter import wrap_simulator
    gpu = wrap_simulator(core.old.simulate_kick, device_index=job['device'])
    dense_lo, dense_hi = ms_to_step(DENSE_WINDOW_MS[0]), ms_to_step(DENSE_WINDOW_MS[1])

    def simulate(params, net, *args, **kwargs):
        ne = net['NE']; assert ne == NE and params.dt == DT_MS
        slow = kwargs['slow']
        assert slow.cfg.use_m and slow.cfg.use_z and slow.cfg.eta_m == job['eta_m']
        assert slow.cfg.tau_adp == 1000 and slow.cfg.tau_z == 5000 and slow.cfg.I_th_EI == job['threshold']
        write(folder / 'applied_configuration.json', dict(
            eta_m=slow.cfg.eta_m, tau_M_s=1., Z_enabled=slow.cfg.use_z, M_observed=slow.cfg.use_m,
            M_effective_feedback=job['eta_m'] > 0, tau_Z_s=5., threshold=slow.cfg.I_th_EI,
            starting_step=slow._step_index, starts_from_own_checkpoint=kwargs.get('resume_state') is not None,
            device=job['device'], neurons=NE + NI, step_ms=DT_MS, topology_identity=p['identity'],
            source_hashes=p['source_hashes'], replay_of=str(REF / 'runs' / job['name']),
            extra_checkpoint_steps=sorted(extra), stop_step=stop_step))
        cells = core.old.spatial_cell_index(net['pos'][:ne], n_grid=20, sheet_l_mm=params.L)
        nc = np.bincount(cells, minlength=400).astype(float)
        fields = []; populations = []; observe = kwargs['spike_observer']; original_sink = kwargs['checkpoint_sink']
        refs = dict(zip(original_sink.__code__.co_freevars, original_sink.__closure__))
        assert {'block_start', 'clear', 'data', 'identity', 'prior_wall', 'started', 'tracker'} <= refs.keys()
        rec = dict(zm_step=[], z=[], m=[], ie=[], ii=[], xi=[], in_step=[], drive_mean=[], drive_var=[], glob=[])
        core_apply = slow.apply_currents

        def apply(ie, ii, labels=None, rec_=None):
            value = core_apply(ie, ii, labels, rec_); k = slow._step_index
            if k % 100 == 0 or (dense_lo <= k < dense_hi and k % 50 == 0):
                rec['zm_step'].append(k); rec['z'].append(slow.z[:ne].copy()); rec['m'].append(slow.m[:ne].copy())
                rec['ie'].append(ie[:ne].astype(np.float32)); rec['ii'].append(ii[:ne].astype(np.float32))
            return value
        slow.apply_currents = apply
        core_inputs = kwargs['input_observer']

        def inputs(tm, nu, xi):
            core_inputs(tm, nu, xi); k = round(tm / DT_MS); rec['xi'].append(xi)
            if k % 10 == 0:
                v = nu[:ne]; mean = np.bincount(cells, weights=v, minlength=400) / nc
                rec['in_step'].append(k); rec['drive_mean'].append(mean.astype(np.float32))
                rec['drive_var'].append(np.maximum(np.bincount(cells, weights=v * v, minlength=400) / nc - mean * mean, 0.).astype(np.float32))
                rec['glob'].append(float(nu[ne]))

        def observer(tm, spikes):
            counts = np.bincount(cells[spikes[:ne]], minlength=400).astype(np.uint16)
            fields.append(counts); populations.append([int(counts.sum()), int(spikes[ne:].sum())])
            observe(tm, spikes)

        def sink(k, engine):
            c = {n: v.cell_contents for n, v in refs.items()}
            start = c['block_start']
            if k in extra:
                (folder / 'checkpoints').mkdir(exist_ok=True)
                digest = ckpt.save(engine, folder / 'checkpoints' / f't{k // 10}ms.npz')
                write(folder / 'checkpoints' / f't{k // 10}ms.json', dict(step=k, time_ms=k * DT_MS, sha256=digest,
                      mean_Z_E=float(engine['slow']['z'][:ne].mean()), mean_M_E=float(engine['slow']['m'][:ne].mean())))
            if k % 5000 != 0:
                return
            assert 0 < k - start <= 5000
            data = {key: np.asarray(val) for key, val in c['data'].items()}
            raw = np.asarray(fields, dtype=np.uint16); pop = np.asarray(populations, dtype=np.uint16)
            assert len(raw) == len(data['raster']) == k - start
            assert np.array_equal(raw.reshape(-1, 10, 400).sum(1), data['field_1ms'])
            assert np.array_equal(pop.reshape(-1, 10, 2).sum(1), data['spikes_1ms'])
            assert np.array_equal(raw.sum(1), pop[:, 0])
            for key in ['spikes_1ms', 'regions_1ms', 'field_1ms']:
                data[key] = data[key].astype(np.uint16)
            data.update(field_0p1ms=raw, population_0p1ms=pop, start_step=start, end_step=k)
            chunks = folder / 'chunks'; chunks.mkdir(exist_ok=True)
            dest = chunks / f'{start:010d}_{k:010d}.npz'; tmp = dest.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, **data); tmp.replace(dest)
            fdir = folder / 'fields'; fdir.mkdir(exist_ok=True)
            fdest = fdir / f'{start:010d}_{k:010d}.npz'; ftmp = fdest.with_suffix('.tmp.npz')
            np.savez(ftmp, zm_step=np.asarray(rec['zm_step'], np.int64), z=np.asarray(rec['z']), m=np.asarray(rec['m']),
                     ie=np.asarray(rec['ie']), ii=np.asarray(rec['ii']), xi=np.asarray(rec['xi'], np.float32),
                     in_step=np.asarray(rec['in_step'], np.int64), drive_mean=np.asarray(rec['drive_mean']),
                     drive_var=np.asarray(rec['drive_var']), glob=np.asarray(rec['glob'], np.float32),
                     cell_e=cells, cell_e_counts=nc, start_step=start, end_step=k)
            ftmp.replace(fdest)
            tr = c['tracker']; tr['wall_s'] = c['prior_wall'] + time.time() - c['started']
            save_pickle(folder / 'checkpoint.pkl', dict(job=job, identity=c['identity'], engine=engine, tracker=tr, restore_from=slow.restore_from))
            refs['block_start'].cell_contents = k; c['clear'](); fields.clear(); populations.clear()
            for v in rec.values():
                v.clear()
            write(folder / 'progress.json', dict(status='RUNNING', pid=os.getpid(), create_time=psutil.Process().create_time(),
                  time_s=k * .0001, job=job, entries=tr['entries'], recoveries=tr['recoveries'], phase=tr['phase'],
                  stop_s=min(job['horizon_s'], tr['stop_s']), Z=float(slow.z[:ne].mean()),
                  adaptation_current=float(job['eta_m'] * slow.m[:ne].mean()), wall_s=tr['wall_s']))
            if k >= stop_step or k * .0001 >= min(job['horizon_s'], tr['stop_s']) - 1e-9:
                raise core.Stop()
        kwargs['spike_observer'] = observer; kwargs['checkpoint_sink'] = sink; kwargs['input_observer'] = inputs
        kwargs['checkpoint_steps'] = set(kwargs['checkpoint_steps']) | extra
        return gpu(params, net, *args, **kwargs)
    core.old.simulate_kick = simulate
    try:
        result = core.worker(job)
        result.update(post_second_confirmation_s=None, post_first_confirmation_s=2., manual_intervention=False,
                      native_spatial_bin_ms=.1, external_restore_never_clears_M=True,
                      replay_of=str(REF / 'runs' / job['name']), canary=canary, stop_step=stop_step)
        assert result['tracker']['restore_s'] is None and result['tracker']['release_s'] is None
        write(folder / 'result.json', result)
        return result
    except Exception as exc:
        write(folder / 'failure.json', dict(error=repr(exc), time=time.time(), pid=os.getpid()))
        raise


def qa(seed, canary=False):
    """Bitwise comparison with the reference chunks and (full replay) the reference 12.5-s engine state."""
    base = base_dir(canary); job = reference_job(seed); folder = base / 'runs' / job['name']
    ref_folder = REF / 'runs' / job['name']
    r = read(folder / 'result.json'); assert r['status'] == 'COMPLETE'
    keys = ['time_ms', 'spikes_1ms', 'regions_1ms', 'field_1ms', 'raster', 'slow_time_ms', 'Z', 'M', 'currents',
            'lfp_time_ms', 'lfp_raw', 'inputs', 'field_0p1ms', 'population_0p1ms', 'start_step', 'end_step']
    compared = []; mismatches = []
    for path in sorted((folder / 'chunks').glob('*.npz')):
        ref = ref_folder / 'chunks' / path.name
        if not ref.exists():
            mismatches.append(dict(chunk=path.name, error='no reference chunk')); continue
        with np.load(path) as a, np.load(ref) as b:
            for key in keys:
                if not np.array_equal(a[key], b[key]):
                    mismatches.append(dict(chunk=path.name, key=key))
        compared.append(path.name)
    report = dict(seed=seed, canary=canary, chunks_compared=compared, chunk_keys=keys, mismatches=mismatches,
                  chunks_bitwise_identical=not mismatches)
    if not canary:
        ref_ck = load_pickle(ref_folder / 'checkpoint.pkl')
        assert int(ref_ck['engine']['step']) == ms_to_step(REPLAY_END_MS)
        mine = ckpt.load(folder / 'checkpoints' / f't{REPLAY_END_MS}ms.npz')
        diffs = compare_states(mine, ref_ck['engine'])
        report.update(final_state_compared_step=ms_to_step(REPLAY_END_MS), final_state_differences=diffs,
                      final_engine_state_bitwise_identical=not diffs,
                      reference_tracker_entries=ref_ck['tracker']['entries'], replay_tracker_entries=r['tracker']['entries'],
                      tracker_entries_identical=ref_ck['tracker']['entries'] == r['tracker']['entries'])
        z_index = {}
        for ms in Z_TIMES_MS:
            st = ckpt.load(folder / 'checkpoints' / f't{ms}ms.npz'); z = st['slow']['z'][:NE]; m = st['slow']['m'][:NE]
            z_index[str(ms)] = dict(step=int(st['step']), mean_Z=float(z.mean()), std_Z=float(z.std()), min_Z=float(z.min()),
                                    mean_M=float(m.mean()), sha256=sha(folder / 'checkpoints' / f't{ms}ms.npz'))
        report['z_field_index'] = z_index
    report['status'] = 'PASS' if not mismatches and report.get('final_engine_state_bitwise_identical', True) else 'FAIL'
    write(folder / 'replay_qa.json', report)
    return report


def launch(seed, canary=False):
    env = dict(os.environ, LD_LIBRARY_PATH=LD)
    base = base_dir(canary); folder = base / 'runs' / f'eta0.0005_s{seed}'; folder.mkdir(parents=True, exist_ok=True)
    log = (folder / 'worker.log').open('a')
    cmd = [PY, '-u', str(Path(__file__).resolve()), 'worker', '--seed', str(seed)] + (['--canary'] if canary else [])
    child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close(); return child


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['prepare', 'worker', 'qa', 'launch'])
    ap.add_argument('--seed', type=int); ap.add_argument('--canary', action='store_true'); a = ap.parse_args()
    if a.mode == 'prepare':
        prepare(a.canary)
    elif a.mode == 'worker':
        worker(a.seed, a.canary)
    elif a.mode == 'qa':
        print(json.dumps(qa(a.seed, a.canary), indent=1)[:3000])
    else:
        child = launch(a.seed, a.canary); print('launched', child.pid)
