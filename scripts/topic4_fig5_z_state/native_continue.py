#!/usr/bin/env python3
"""M1: native SNN continuations with frozen per-cell Z (dynamic M), optional frozen M, extensions.

State construction (design section 5.1): physical fast state + M from the history checkpoint, clock/OU/RNG/drive
from the 10.370-s anchor checkpoint, Z from the source-time checkpoint; W2 replaces only the RNG streams.
Runs go through the unchanged reference producer (core.worker) with a seeded checkpoint and an observe-only tracker.
"""
import argparse
import shutil
import subprocess
import numpy as np
import psutil
from common import *  # noqa: F401,F403

NATIVE = OUT / 'native'
REPLAY_RUN = OUT / 'replay' / 'runs' / f'eta0.0005_s{MAIN_SEED}'


def replay_checkpoint(ms):
    path = REPLAY_RUN / 'checkpoints' / f't{ms}ms.npz'
    if not path.exists():
        raise FileNotFoundError(path)
    return ckpt.load(path)


def build_state(job):
    physical = replay_checkpoint(job['history_ms'])
    anchor = replay_checkpoint(ANCHOR_MS)
    z = replay_checkpoint(job['z_source_ms'])['slow']['z']
    assert z.shape == (NE + NI,) and np.all(z[NE:] == 1.)
    m = None
    if job.get('m_source_ms') is not None:
        m = replay_checkpoint(job['m_source_ms'])['slow']['m']; assert np.all(m[NE:] == 0.)
    state = transplant_state(physical, anchor, z_field=z, m_field=m)
    if job['future'] == 'W2':
        state = replace_future_innovations(state, W2_SEED)
    else:
        assert job['future'] == 'W1'
    assert int(state['step']) == ms_to_step(ANCHOR_MS)
    return state


def make_job(name, z_source_ms, history_ms, future, m_source_ms=None, duration_ms=CONTINUATION_MS):
    ref = reference_job(MAIN_SEED)
    start = ms_to_step(ANCHOR_MS); horizon_s = (start + ms_to_step(duration_ms)) * DT_MS / 1000.
    return dict(name=name, eta_m=ref['eta_m'], tau_M_s=ref['tau_M_s'], seed=MAIN_SEED, eta_index=ref['eta_index'],
                tau_index=ref['tau_index'], tau_z_ms=ref['tau_z_ms'], threshold=ref['threshold'], horizon_s=horizon_s,
                device=None, z_source_ms=int(z_source_ms), history_ms=int(history_ms), future=future,
                m_source_ms=None if m_source_ms is None else int(m_source_ms), freeze_z=True, freeze_m=m_source_ms is not None,
                anchor_ms=ANCHOR_MS, start_step=start, duration_ms=int(duration_ms), kind='continuation')


def make_extension_job(parent_name, duration_ms=EXTENSION_MS):
    parent = read(NATIVE / 'jobs' / (parent_name + '.json'))
    job = dict(parent); job['name'] = parent_name + '_ext'; job['kind'] = 'extension'; job['parent'] = parent_name
    end = ms_to_step(ANCHOR_MS) + ms_to_step(parent['duration_ms'])
    job['start_step'] = end; job['duration_ms'] = int(duration_ms)
    job['horizon_s'] = (end + ms_to_step(duration_ms)) * DT_MS / 1000.
    return job


def main_jobs():
    jobs = []
    for z in Z_TIMES_MS:
        for h in HISTORY_MS:
            for w in ('W1', 'W2'):
                jobs.append(make_job(f'z{z}_h{h}_{w}', z, h, w))
    return jobs


def prepare():
    NATIVE.mkdir(parents=True, exist_ok=True); (NATIVE / 'jobs').mkdir(exist_ok=True)
    if not (NATIVE / 'protocol.json').exists():
        p = check_reference_sources()
        qa = read(REPLAY_RUN / 'replay_qa.json'); assert qa['status'] == 'PASS', 'M0 replay QA must pass first'
        write(NATIVE / 'protocol.json', dict(status='DEFINED_BEFORE_RUNS', identity=p['identity'], source_hashes=p['source_hashes'],
              replay_qa=str(REPLAY_RUN / 'replay_qa.json'), tracker='observe_only (records entries/recoveries, never shortens horizon)',
              z_source_times_ms=Z_TIMES_MS, histories_ms=HISTORY_MS, anchor_ms=ANCHOR_MS, W2_seed=W2_SEED,
              W2_drive_seed=W2_SEED + W2_DRIVE_SEED_OFFSET, continuation_ms=CONTINUATION_MS,
              frozen_application_check=read(OUT / 'frozen_application_check.json'),
              statistical_unit='one complete continuation trajectory'))
    for job in main_jobs():
        if not (NATIVE / 'jobs' / (job['name'] + '.json')).exists():
            write(NATIVE / 'jobs' / (job['name'] + '.json'), job)
    return read(NATIVE / 'protocol.json')


def seed_folder(job, device):
    """Write the seeded checkpoint.pkl that core.worker resumes from."""
    folder = NATIVE / 'runs' / job['name']; folder.mkdir(parents=True, exist_ok=True)
    cp = folder / 'checkpoint.pkl'
    if cp.exists():
        saved = load_pickle(cp); assert saved['job'] == job, 'existing checkpoint belongs to a different job definition'
        return folder
    identity = read(NATIVE / 'protocol.json')['identity']
    if job['kind'] == 'extension':
        parent = NATIVE / 'runs' / job['parent']
        saved = load_pickle(parent / 'checkpoint.pkl'); r = read(parent / 'result.json'); assert r['status'] == 'COMPLETE'
        assert int(saved['engine']['step']) == job['start_step'], (saved['engine']['step'], job['start_step'])
        assert saved['job']['name'] == job['parent']
        tracker = saved['tracker']; tracker['stop_s'] = 1e9
        state = saved['engine']
        (folder / 'chunks').mkdir(exist_ok=True)
        for path in sorted((parent / 'chunks').glob('*.npz')):
            if '.tmp.' not in path.name:
                shutil.copy2(path, folder / 'chunks' / path.name)
        (folder / 'fields').mkdir(exist_ok=True)
        for path in sorted((parent / 'fields').glob('*.npz')):
            shutil.copy2(path, folder / 'fields' / path.name)
        origin = dict(parent=str(parent), parent_checkpoint_sha256=sha(parent / 'checkpoint.pkl'), parent_result_sha256=sha(parent / 'result.json'),
                      all_state_preserved=True, chunks_copied=True)
    else:
        state = build_state(job)
        tracker = core.fresh_tracker(); tracker['stop_s'] = 1e9
        origin = dict(history_checkpoint=str(REPLAY_RUN / 'checkpoints' / f"t{job['history_ms']}ms.npz"),
                      anchor_checkpoint=str(REPLAY_RUN / 'checkpoints' / f't{ANCHOR_MS}ms.npz'),
                      z_checkpoint=str(REPLAY_RUN / 'checkpoints' / f"t{job['z_source_ms']}ms.npz"),
                      m_checkpoint=None if job['m_source_ms'] is None else str(REPLAY_RUN / 'checkpoints' / f"t{job['m_source_ms']}ms.npz"),
                      initial_mean_Z_E=float(state['slow']['z'][:NE].mean()), initial_mean_M_E=float(state['slow']['m'][:NE].mean()),
                      delay_ring_rebased_by_steps=int((ms_to_step(ANCHOR_MS) - ms_to_step(job['history_ms'])) % state['ring_sE'].shape[0]),
                      future_rng=('anchor original stream' if job['future'] == 'W1' else f'seed {W2_SEED} / drive seed {W2_SEED + W2_DRIVE_SEED_OFFSET}'),
                      xi_at_start=float(state['xi']), all_state_preserved=True)
    save_pickle(cp, dict(job=job, identity=identity, engine=state, tracker=tracker, restore_from=None))
    write(folder / 'continuation.json', dict(origin, device=device, seeded_at=time.time(), checkpoint_sha256=sha(cp)))
    return folder


def worker(name, device):
    p = prepare(); check_reference_sources()
    job = read(NATIVE / 'jobs' / (name + '.json'))
    folder = seed_folder(job, device)
    if (folder / 'result.json').exists():
        return read(folder / 'result.json')
    core.OUT = NATIVE; core.tracker_step = observe_only_tracker
    last = round(job['horizon_s'] * 10000); extra = {last}
    from src.topic4_cuda_ordered_scatter import wrap_simulator
    gpu = wrap_simulator(core.old.simulate_kick, device_index=int(device))
    centers = np.asarray(read(SUBSTRATE / 'substrate.json')['centers_mm'])

    def simulate(params, net, *args, **kwargs):
        ne = net['NE']; assert ne == NE and params.dt == DT_MS
        slow = kwargs['slow']; state = kwargs['resume_state']; assert state is not None
        assert slow.cfg.use_m and slow.cfg.use_z and slow.cfg.eta_m == job['eta_m']
        assert slow.cfg.tau_adp == 1000 and slow.cfg.tau_z == 5000 and slow.cfg.I_th_EI == job['threshold']
        frozen = FrozenSlow(slow, freeze_z=job['freeze_z'], freeze_m=job['freeze_m'])
        z_expected = np.array(state['slow']['z'], copy=True); m_expected = np.array(state['slow']['m'], copy=True)
        write(folder / 'applied_configuration.json', dict(
            eta_m=slow.cfg.eta_m, tau_M_s=1., Z_enabled=slow.cfg.use_z, M_observed=slow.cfg.use_m, M_effective_feedback=True,
            tau_Z_s=5., threshold=slow.cfg.I_th_EI, starting_step=int(state['step']), starts_from_seeded_checkpoint=True,
            frozen_Z_state_update=job['freeze_z'], frozen_M_state_update=job['freeze_m'],
            frozen_values_still_applied_to_current=True, device=int(device), neurons=NE + NI, step_ms=DT_MS,
            topology_identity=p['identity'], source_hashes=p['source_hashes'], job=job))
        pos = net['pos'][:ne]; d = np.linalg.norm(pos[:, None] - centers[None], axis=2)
        g15 = np.full(ne, 2); g15[d[:, 0] < 1.5] = 0; g15[(d[:, 1] < 1.5) & (d[:, 1] < d[:, 0])] = 1
        cells = core.old.spatial_cell_index(pos, n_grid=20, sheet_l_mm=params.L)
        fields = []; populations = []; core15 = np.zeros(3, np.uint32); core15_1ms = []
        observe = kwargs['spike_observer']; original_sink = kwargs['checkpoint_sink']
        refs = dict(zip(original_sink.__code__.co_freevars, original_sink.__closure__))
        assert {'block_start', 'clear', 'data', 'identity', 'prior_wall', 'started', 'tracker'} <= refs.keys()
        rec = dict(m_step=[], m=[], xi=[])
        core_apply = slow.apply_currents

        def apply(ie, ii, labels=None, rec_=None):
            value = core_apply(ie, ii, labels, rec_); k = slow._step_index
            if k % 10000 == 0:
                rec['m_step'].append(k); rec['m'].append(slow.m[:ne].copy())
            return value
        slow.apply_currents = apply
        core_inputs = kwargs['input_observer']

        def inputs(tm, nu, xi):
            core_inputs(tm, nu, xi); rec['xi'].append(xi)

        def observer(tm, spikes):
            counts = np.bincount(cells[spikes[:ne]], minlength=400).astype(np.uint16)
            fields.append(counts); populations.append([int(counts.sum()), int(spikes[ne:].sum())])
            core15[:] += np.bincount(g15[spikes[:ne]], minlength=3).astype(np.uint32)
            k = round(tm / DT_MS)
            if (k + 1) % 10 == 0:
                core15_1ms.append(core15.astype(np.uint16)); core15.fill(0)
            observe(tm, spikes)

        def sink(k, engine):
            c = {n: v.cell_contents for n, v in refs.items()}
            start = c['block_start']
            if job['freeze_z']:
                assert np.array_equal(engine['slow']['z'], z_expected), 'frozen Z drifted'
            if job['freeze_m']:
                assert np.array_equal(engine['slow']['m'], m_expected), 'frozen M drifted'
            if k in extra:
                (folder / 'checkpoints').mkdir(exist_ok=True)
                digest = ckpt.save(engine, folder / 'checkpoints' / f't{k // 10}ms.npz')
                write(folder / 'checkpoints' / f't{k // 10}ms.json', dict(step=k, time_ms=k * DT_MS, sha256=digest))
            assert 0 < k - start <= 5000
            data = {key: np.asarray(val) for key, val in c['data'].items()}
            raw = np.asarray(fields, dtype=np.uint16); pop = np.asarray(populations, dtype=np.uint16)
            assert len(raw) == len(data['raster']) == k - start
            assert np.array_equal(raw.reshape(-1, 10, 400).sum(1), data['field_1ms'])
            assert np.array_equal(pop.reshape(-1, 10, 2).sum(1), data['spikes_1ms'])
            assert np.array_equal(raw.sum(1), pop[:, 0])
            c15 = np.asarray(core15_1ms, dtype=np.uint16); assert np.array_equal(c15.sum(1), data['spikes_1ms'][:, 0])
            for key in ['spikes_1ms', 'regions_1ms', 'field_1ms']:
                data[key] = data[key].astype(np.uint16)
            data.update(field_0p1ms=raw, population_0p1ms=pop, core15_1ms=c15, start_step=start, end_step=k)
            chunks = folder / 'chunks'; chunks.mkdir(exist_ok=True)
            dest = chunks / f'{start:010d}_{k:010d}.npz'; tmp = dest.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, **data); tmp.replace(dest)
            fdir = folder / 'fields'; fdir.mkdir(exist_ok=True)
            fdest = fdir / f'{start:010d}_{k:010d}.npz'; ftmp = fdest.with_suffix('.tmp.npz')
            np.savez(ftmp, m_step=np.asarray(rec['m_step'], np.int64), m=np.asarray(rec['m']).reshape(-1, ne),
                     xi=np.asarray(rec['xi'], np.float32), start_step=start, end_step=k, g15=g15, cell_e=cells)
            ftmp.replace(fdest)
            tr = c['tracker']; tr['wall_s'] = c['prior_wall'] + time.time() - c['started']
            save_pickle(folder / 'checkpoint.pkl', dict(job=job, identity=c['identity'], engine=engine, tracker=tr, restore_from=slow.restore_from))
            refs['block_start'].cell_contents = k; c['clear'](); fields.clear(); populations.clear(); core15_1ms.clear()
            for v in rec.values():
                v.clear()
            write(folder / 'progress.json', dict(status='RUNNING', pid=os.getpid(), create_time=psutil.Process().create_time(),
                  time_s=k * .0001, job=job, entries=tr['entries'], recoveries=tr['recoveries'], phase=tr['phase'],
                  stop_s=job['horizon_s'], Z=float(slow.z[:ne].mean()), adaptation_current=float(job['eta_m'] * slow.m[:ne].mean()),
                  wall_s=tr['wall_s']))
            if k >= last:
                raise core.Stop()
        kwargs['spike_observer'] = observer; kwargs['checkpoint_sink'] = sink; kwargs['input_observer'] = inputs
        kwargs['checkpoint_steps'] = set(kwargs['checkpoint_steps']) | extra
        return gpu(params, net, *args, **kwargs)
    core.old.simulate_kick = simulate
    try:
        result = core.worker(job)
        result.update(native_spatial_bin_ms=.1, kind=job['kind'], frozen_Z=job['freeze_z'], frozen_M=job['freeze_m'],
                      start_s=job['start_step'] * DT_MS / 1000., duration_ms=job['duration_ms'], device=int(device),
                      continuation=read(folder / 'continuation.json'))
        write(folder / 'result.json', result)
        return result
    except Exception as exc:
        write(folder / 'failure.json', dict(error=repr(exc), time=time.time(), pid=os.getpid()))
        raise


def launch(name, device):
    env = dict(os.environ, LD_LIBRARY_PATH=LD)
    folder = NATIVE / 'runs' / name; folder.mkdir(parents=True, exist_ok=True)
    log = (folder / 'worker.log').open('a')
    cmd = [PY, '-u', str(Path(__file__).resolve()), 'worker', '--name', name, '--device', str(device)]
    child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close(); return child


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['prepare', 'worker', 'launch'])
    ap.add_argument('--name'); ap.add_argument('--device', type=int, default=0); a = ap.parse_args()
    if a.mode == 'prepare':
        prepare()
    elif a.mode == 'worker':
        worker(a.name, a.device)
    else:
        print('launched', launch(a.name, a.device).pid)
