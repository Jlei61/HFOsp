#!/usr/bin/env python3
"""Fixed M-on manual substrate: bounded first-entry Z-kinetics response surface."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['TOPIC4_MANUAL_ARM'] = 'manual_hard'
import numpy as np
import run_topic4_weak_fast_recurrence as base

ROOT = base.ROOT
OUT = ROOT / 'results/topic4_sef_hfo/m_on_z_kinetics_20260912'
OLD = base.OUT / 'runs/weak_fast_z_refill_recurrence'
TH = base.THRESHOLD
SEEDS = [9108401, 9108402, 9108403]


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    base.write(p, value)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    tmp = p.with_suffix('.tmp')
    with tmp.open('wb') as f:
        pickle.dump(value, f, protocol=5)
    tmp.replace(p)


class Stop(Exception):
    pass


def confirmed_entry(counts, ne=32000):
    """Return first 20 consecutive high 10-ms bins, onset and confirmation."""
    mask = np.asarray(counts)[:, 0] / ne / .01 >= 200.
    run = 0
    for i, high in enumerate(mask):
        run = run + 1 if high else 0
        if run == 20:
            return (i - 19) * .01, (i + 1) * .01
    return None, None


def prepare():
    if (OUT / 'protocol.json').exists():
        return read(OUT / 'protocol.json')
    tau = np.r_[np.geomspace(2500, 5000, 4), np.geomspace(5000, 10000, 4)[1:]]
    th = np.r_[np.linspace(75, TH, 4), np.linspace(TH, 120, 4)[1:]]
    jobs = [dict(name=f'y{y}_x{x}_s{s}', x=x, y=y, seed=s, tau_z_ms=float(t),
                 threshold=float(h), duration_ms=180000., eta_m=.02, tau_adp_ms=2000.)
            for y, h in enumerate(th) for x, t in enumerate(tau) for s in SEEDS]
    sources = [Path(__file__), Path(base.__file__),
               ROOT/'scripts/validate_topic4_fixed_rate_base.py',
               ROOT/'scripts/topic4_historical_manual_z_common.py',
               ROOT/'src/topic4_raster_protocol_engine.py',
               ROOT/'src/snn_engine/mz_slow_vars.py',
               ROOT/'config/topic4_rate_model_dynamics_validation_v1.json',
               ROOT/'results/topic4_sef_hfo/data_driven_core_field/config/stage_config.json']
    p = dict(status='DEFINED_BEFORE_SIMULATIONS', tau_s=(tau/1000).tolist(),
             thresholds=th.tolist(), seeds=SEEDS, jobs=jobs, total=147, grid_shape=[7, 7],
             M_enabled=True, eta_m=.02, tau_M_s=2., horizon_s=180., topology_seed=6101,
             max_workers=32, minimum_available_memory_GiB=60, worker_memory_budget_GiB=4,
             minimum_disk_free_GiB=20, checkpoint_interval_s=20,
             substrate=str(base.PREVIOUS/'substrate.json'),
             identity=read(base.PREVIOUS/'substrate.json')['identity'],
             origin='Fresh initialization Z=1, M=0; same native OU and paired seeds at every grid node; no stimulus or manual reset.',
             endpoint='All-E 10-ms rate >=200 Hz for 20 consecutive bins. Store onset and confirmation separately; F uses confirmation, matching prior F.',
             post_confirmation_s=.5,
             estimator='Mean min(T_confirmation,180 s) over three paired noise seeds. Also show fraction entering by 180 s; no interpolation, no missing-to-censored conversion.',
             semantics='tau_Z changes BOTH depletion and recovery. I_th is the raw GABA current threshold that triggers depletion, not synaptic inhibitory strength or depletion per event.',
             equations='tau_Z dz_i/dt = 1[I_GABA_i<I_th]-z_i; dm_i/dt=-m_i/tau_M+spikes; applied current=IE-z_i*II-0.02*m_i, E only.',
             scope='First-entry timing response on historical manual dual core with M on; separate from the ongoing Z-reset recurrence diagnosis and patient propagation fitting.',
             acceptance=str(ROOT/'results/topic4_sef_hfo/reset_state_diagnosis_20260911/figure_acceptance.md'),
             stopping='Exactly 147 grid runs plus one 2-s split/resume canary; stop at scientific review. No automatic new grid, selection, or replacement of full Fig5.',
             source_hashes={str(q): sha(q) for q in sources})
    write(OUT/'protocol.json', p)
    for job in jobs:
        write(OUT/'jobs'/(job['name']+'.json'), job)
    return p


def check_sources(p):
    for path, digest in p['source_hashes'].items():
        assert sha(path) == digest, 'Changed source: '+path


def worker(job):
    import fcntl
    import resource
    folder = OUT/'runs'/job['name']; folder.mkdir(parents=True, exist_ok=True)
    lock = (folder/'worker.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder/'result.json').exists():
        return read(folder/'result.json')
    protocol = read(OUT/'protocol.json'); check_sources(protocol)
    started = time.time()
    write(folder/'progress.json', dict(status='BUILDING', pid=os.getpid(), job=job))
    s, tr, frozen, identity = base.setup(job['seed'])
    assert identity == protocol['identity']
    ne, ni = s.n_e, s.n_i; p = s.params
    assert (ne, ni, p.dt) == (32000, 8000, .1)
    v = s.vtheta[:ne]; assert int((v < 18).sum()) == 781 and not (v > 18).any()
    cfg = base.MZSlowVarsConfig(use_z=True, use_m=True, tau_z=job['tau_z_ms'],
          I_th_EI=job['threshold'], tau_adp=job['tau_adp_ms'], eta_m=job['eta_m'])
    # Unmodified native Z/M equations. The existing class records no built-in
    # dense trace; restore_ms remains None for the entire first-entry assay.
    slow = base.ReleaseZ(ne+ni, p.V_th, cfg, NE=ne)
    centers = np.asarray(frozen['candidate']['node_field']['centers_mm'])
    def groups(pos):
        d = np.linalg.norm(pos[:, None]-centers[None], axis=2)
        g = np.full(len(pos), 2); g[d[:, 0] < 1.75] = 0
        g[(d[:, 1] < 1.75) & (d[:, 1] < d[:, 0])] = 1
        return g
    ge, gi = groups(s.positions_e), groups(s.positions_i)
    idx = [np.flatnonzero(ge == g) for g in range(3)]
    nr = np.r_[np.bincount(ge, minlength=3), np.bincount(gi, minlength=3)]
    obs = dict(counts=[], regions=[], slow_time_s=[], Z=[], M=[], currents=[], inputs=[], input_digests=[])
    track = dict(seen=0, high_bins=0, confirmation_s=None, previous_wall_s=0.)
    resume = None; cp = folder/'checkpoint.pkl'
    if cp.exists():
        with cp.open('rb') as f: saved = pickle.load(f)
        assert saved['job'] == job and saved['identity'] == identity
        resume, obs, track = saved['engine'], saved['observations'], saved['tracker']
        del saved
    offset = track['seen']; end = round(job['duration_ms']/.1)
    count = np.zeros(2, dtype=np.int64); region = np.zeros(6, dtype=np.int64)
    digest = hashlib.sha256()
    original = slow.apply_currents
    def apply(ie, ii, labels=None, rec=None):
        value = original(ie, ii, labels, rec)
        k = slow._step_index
        if k % 1000 == 0:
            z, m = slow.z[:ne], slow.m[:ne]
            obs['slow_time_s'].append(k*.0001)
            obs['Z'].append([z.mean(), *[z[g].mean() for g in idx], np.mean(ii[:ne] >= cfg.I_th_EI)])
            obs['M'].append([m.mean(), *[m[g].mean() for g in idx]])
            obs['currents'].append([ie[:ne].mean(), ii[:ne].mean(), np.mean(z*ii[:ne])])
        return value
    slow.apply_currents = apply
    def inputs(tm, nu, xi):
        nonlocal digest
        k = round(tm/.1)
        if k % 1000 == 0:
            obs['inputs'].append([tm/1000, xi, nu[:ne].mean(), nu[ne:].mean()])
            digest.update(np.asarray(nu).tobytes()); digest.update(np.float64(xi).tobytes())
        if k % 10000 == 9000:
            obs['input_digests'].append(digest.hexdigest()); digest = hashlib.sha256()
    def progress(status='RUNNING'):
        write(folder/'progress.json', dict(status=status, pid=os.getpid(), job=job,
            time_s=track['seen']*.0001, confirmation_s=track['confirmation_s'],
            wall_s=track['previous_wall_s']+time.time()-started,
            mean_Z=float(slow.z[:ne].mean()), adaptation_current=float(.02*slow.m[:ne].mean())))
    def observe(tm, spikes):
        k = round(tm/.1); track['seen'] = k+1
        count[:] += [spikes[:ne].sum(), spikes[ne:].sum()]
        region[:3] += np.bincount(ge[spikes[:ne]], minlength=3)
        region[3:] += np.bincount(gi[spikes[ne:]], minlength=3)
        if (k+1) % 100 == 0:
            obs['counts'].append(count.copy()); obs['regions'].append(region.copy())
            rate = count[0]/ne/.01; count.fill(0); region.fill(0)
            track['high_bins'] = track['high_bins']+1 if rate >= 200 else 0
            if track['high_bins'] >= 20 and track['confirmation_s'] is None:
                track['confirmation_s'] = (k+1)*.0001
        if (k+1) % 10000 == 0:
            progress()
        if track['confirmation_s'] is not None and (k+1)*.0001 >= track['confirmation_s']+.5-1e-9:
            raise Stop()
    def checkpoint(k, state):
        assert k == track['seen'] and not np.any(count) and not np.any(region)
        h = dict(track, previous_wall_s=track['previous_wall_s']+time.time()-started)
        dump(cp, dict(job=job, identity=identity, engine=state, observations=obs, tracker=h))
        if job.get('qa_pause') and offset == 0:
            raise Stop()
    s.net['rng'] = np.random.default_rng(job['seed'])
    drive = base.make_external_drive(s, tr['spatial_ou'], job['seed'])
    p.T = job['duration_ms'] - offset*.1
    spacing = 10000 if job.get('qa_pause') else 200000
    try:
        base.simulate_kick(p, s.net, KICK_BOOST=0., V_th_per_neuron=s.vtheta,
            slow=slow, external_e_rate_drive=drive, early_stop_runaway=False,
            spike_observer=observe, input_observer=inputs, record_dense_spikes=False,
            fast_scatter=True, verbose=False, resume_state=resume, time_offset_ms=offset*.1,
            checkpoint_steps=range((offset//spacing+1)*spacing, end, spacing), checkpoint_sink=checkpoint)
    except Stop:
        pass
    if job.get('qa_pause') and offset == 0:
        progress('QA_CHECKPOINT_PAUSE'); return
    arrays = {k: np.asarray(v) for k, v in obs.items()}
    arrays.update(region_sizes=nr, final_z_e=slow.z[:ne], final_m_e=slow.m[:ne])
    counts = arrays['counts']; assert len(counts)*100 == track['seen']
    assert np.array_equal(counts[:, 0], arrays['regions'][:, :3].sum(1))
    assert np.array_equal(counts[:, 1], arrays['regions'][:, 3:].sum(1))
    onset, confirm = confirmed_entry(counts)
    assert confirm is None if track['confirmation_s'] is None else abs(confirm-track['confirmation_s']) < 1e-9
    assert np.isfinite(arrays['Z']).all() and np.isfinite(arrays['M']).all()
    assert np.all(slow.z[ne:] == 1) and np.all(slow.m[ne:] == 0) and slow.restore_ms is None
    match = None
    if job['seed'] == SEEDS[0] and job['tau_z_ms'] == 5000. and job['threshold'] == TH:
        # Before 75.5 s the saved reference has no intervention; verify the
        # actual native output, including replay after the canary checkpoint.
        with np.load(OLD.with_suffix('.npz')) as a:
            n = min(len(counts), 7550)
            ec = np.rint(a['rate_e_hz'][:n*100]*ne*.1/1000).astype(np.int64).reshape(n, 100).sum(1)
            ic = np.rint(a['rate_i_hz'][:n*100]*ni*.1/1000).astype(np.int64).reshape(n, 100).sum(1)
            assert np.array_equal(counts[:n], np.c_[ec, ic]), 'Original M-on native prefix mismatch'
            m = np.asarray(obs['slow_time_s']) < min(n*.01, 75.5)
            ix = np.rint(arrays['slow_time_s'][m]*200).astype(int)
            assert np.array_equal(arrays['Z'][m, :4], a['z_stats'][ix][:, [0, 5, 6, 7]])
            assert np.array_equal(arrays['M'][m], a['m_stats'][ix][:, [0, 5, 6, 7]])
            match = dict(status='PASS', duration_s=n*.01, counts_Z_M_bitwise=True)
        if not job.get('qa_pause'):
            assert abs(confirm - 73.68) < 1e-9
    tmp = folder/'observations.tmp.npz'; np.savez_compressed(tmp, **arrays)
    tmp.replace(folder/'observations.npz')
    duration = track['seen']*.0001
    result = dict(status='COMPLETE', job=job, duration_s=duration, onset_s=onset,
        confirmation_s=confirm, observed=confirm is not None,
        restricted_time_s=confirm if confirm is not None else job['duration_ms']/1000,
        censored=confirm is None, eta_m=.02, tau_M_s=2., M_enabled=True,
        reset_applied=False, frozen_identity=identity,
        Vth_E_counts=dict(lowered=int((v<18).sum()), equal=int((v==18).sum()), raised=int((v>18).sum())),
        reference_prefix_qa=match, spatial_count_conservation=True,
        peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        wall_s=track['previous_wall_s']+time.time()-started)
    write(folder/'result.json', result); progress('COMPLETE')
    return result


def qa():
    # Independent endpoint boundary checks: 19 high bins cannot trigger; 20 can.
    lo = np.zeros((5, 2)); hi = np.tile([64000, 0], (20, 1))
    assert confirmed_entry(hi[:19]) == (None, None)
    assert confirmed_entry(np.r_[lo, hi]) == (.05, .25)
    job = dict(name='qa_split_resume', seed=SEEDS[0], tau_z_ms=5000., threshold=TH,
               duration_ms=2000., eta_m=.02, tau_adp_ms=2000., qa_pause=True)
    for _ in range(2):
        result = worker(job)
    assert result['reference_prefix_qa']['duration_s'] == 2.
    # Direct parameter-application check against the Euler equations, including
    # both sides of the depletion threshold and spike-triggered adaptation.
    for tau in [2500., 5000., 10000.]:
        for threshold in [75., TH, 120.]:
            cfg = base.MZSlowVarsConfig(use_z=True, use_m=True, tau_z=tau,
                     I_th_EI=threshold, tau_adp=2000., eta_m=.02)
            s = base.ReleaseZ(4, 18., cfg, NE=3)
            s.z[:3] = [.8, .7, .6]; s.m[:3] = [2., 3., 4.]
            ie = np.full(4, 200.); ii = np.array([74., 95., 121., 50.]); sp = np.array([1, 0, 1, 0], bool)
            z = s.z.copy(); m = s.m.copy()
            assert np.array_equal(s.apply_currents(ie, ii), ie-z*ii-.02*m)
            s.step(sp, None, .1)
            z[:3] += .1/tau*((ii[:3]<threshold)-z[:3])
            m[:3] -= .1/2000*m[:3]; m[:3] += sp[:3]
            assert np.array_equal(s.z, z) and np.array_equal(s.m, m)
    write(OUT/'qa.json', dict(status='PASS', native_reference_prefix_s=2,
        split_resume_at_s=1, original_counts_Z_M_bitwise=True,
        parameter_equation_checks=9, endpoint_boundary_checks=True,
        reference_file=str(OLD.with_suffix('.npz'))))


def controller():
    import fcntl
    import psutil
    import shutil
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p = prepare(); check_sources(p)
    assert read(OUT/'qa.json')['status'] == 'PASS'
    # Center, then deterministic spatial coverage; no outcome-driven selection.
    points = [(3, 3)]; todo = [(x, y) for x in range(7) for y in range(7) if (x, y) != (3, 3)]
    while todo:
        q = max(todo, key=lambda q: min((q[0]-v[0])**2+(q[1]-v[1])**2 for v in points))
        points.append(q); todo.remove(q)
    jobs = sorted(p['jobs'], key=lambda j: (points.index((j['x'], j['y']))//4, SEEDS.index(j['seed']), points.index((j['x'], j['y']))))
    pending = [j for j in jobs if not (OUT/'runs'/j['name']/'result.json').exists()]
    running = {}; failed = []; started = time.time(); last_analysis = 0; previous_n = -1
    env = dict(os.environ, LD_LIBRARY_PATH='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib')
    analysis = ROOT/'scripts/analyze_topic4_m_on_z_kinetics.py'
    while pending or running:
        for name, (child, handle) in list(running.items()):
            code = child.poll()
            if code is None:
                continue
            handle.close(); del running[name]
            if code or not (OUT/'runs'/name/'result.json').exists():
                failed.append(dict(name=name, returncode=code))
        if not failed:
            try:
                check_sources(p)
            except Exception as exc:
                failed.append(dict(error=repr(exc)))
        # Reserve for each new process even before its RSS has grown.
        available = psutil.virtual_memory().available/1024**3
        reserve = 0.
        for child, _ in running.values():
            try:
                reserve += max(0., 4-psutil.Process(child.pid).memory_info().rss/1024**3)
            except psutil.NoSuchProcess:
                # The completed child is collected at the next status pass.
                pass
        while pending and not failed and len(running) < p['max_workers'] and available-reserve > 64:
            if shutil.disk_usage(OUT).free/1024**3 < 20:
                break
            job = pending.pop(0); name = job['name']
            log = OUT/'logs'/(name+'.log'); log.parent.mkdir(exist_ok=True)
            f = log.open('a')
            child = subprocess.Popen([sys.executable, '-u', __file__, 'worker', '--job', str(OUT/'jobs'/(name+'.json'))],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
            running[name] = (child, f); reserve += 4
        completed = len([j for j in jobs if (OUT/'runs'/j['name']/'result.json').exists()])
        write(OUT/'status.json', dict(status='DRAINING_AFTER_FAILURE' if failed else 'RUNNING',
             pid=os.getpid(), total=147, completed=completed, running={n:c.pid for n,(c,_) in running.items()},
             pending=len(pending), failed=failed, wall_s=time.time()-started,
             available_memory_GiB=available, max_workers=p['max_workers']))
        if completed != previous_n and time.time()-last_analysis > 60:
            with (OUT/'analysis.log').open('a') as f:
                code = subprocess.call([sys.executable, str(analysis)], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            if code:
                failed.append(dict(error='Analysis failed; see analysis.log'))
            previous_n = completed; last_analysis = time.time()
        if failed and not running:
            break
        time.sleep(15)
    with (OUT/'analysis.log').open('a') as f:
        code = subprocess.call([sys.executable, str(analysis)], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    write(OUT/'status.json', dict(status='FAILED' if failed or code else 'COMPLETE_PENDING_SCIENTIFIC_REVIEW',
        completed=len([j for j in jobs if (OUT/'runs'/j['name']/'result.json').exists()]), total=147,
        pending=len(pending), running={}, failed=failed, analysis_exit_code=code, wall_s=time.time()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['prepare', 'qa', 'controller', 'worker'])
    parser.add_argument('--job'); args = parser.parse_args()
    if args.mode == 'prepare': prepare()
    elif args.mode == 'qa': qa()
    elif args.mode == 'controller': controller()
    else: worker(read(args.job))
