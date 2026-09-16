#!/usr/bin/env python3
"""Persistent, resumable after-search Fig5 controller. Does not edit upstream."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic4_xy_fig5_followup import (
    COMPLETE, admission_slots, available_gib, choose_development_candidate,
    choose_mechanism_config, parameter_grid, proc_memory, read, sha,
    verify_hashes, verify_lock, write)

ART = Path('/home/honglab/leijiaxin/HFOsp')
UPSTREAM = ROOT / 'results/topic4_sef_hfo/vth_dual_core_xy_research'
CONFIG = ROOT / 'config/topic4_xy_round1_fig5_followup.json'
PYTHON = '/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python'
ENV = dict(os.environ, **{k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
ENV['LD_LIBRARY_PATH'] = '/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib' + (
    ':' + ENV['LD_LIBRARY_PATH'] if ENV.get('LD_LIBRARY_PATH') else '')
SOURCES = ['src/topic4_xy_fig5_followup.py', 'scripts/run_topic4_xy_fig5_worker.py',
           'scripts/chain_topic4_xy_round1_fig5.py',
           'src/snn_engine/mz_slow_vars.py', 'src/snn_engine/checkpoint.py',
           'scripts/paper_figures/plot_topic4_xy_fig5_followup.py',
           'config/topic4_xy_round1_fig5_followup.json']


def status(out, name, **kwargs):
    write(out / 'status.json', dict(status=name, updated_unix=time.time(), **kwargs))


def fixed_json(path, value):
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f'immutable downstream input differs: {path}')
    else:
        write(path, value)


def prepare(upstream, out, config):
    plan = read(config)
    lock_path = out / 'runtime_lock.json'
    if lock_path.exists():
        verify_lock(lock_path)
        if read(lock_path)['config_path'] != str(config.resolve()):
            raise RuntimeError('resumed with a different plan')
        fixed_json(out / 'analysis_plan.json', plan)
        return plan, lock_path
    if (upstream / 'selection/aggregate.json').exists():
        raise RuntimeError('create the automatic selection rule before selection finishes')
    hashes = {str(ROOT / name): digest for name, digest in read(upstream / 'source_lock.json')['source_hashes'].items()}
    verify_hashes(hashes)
    hashes.update(read(upstream / 'runtime_lock_direction_v2.json')['hashes'])
    round_contract = read(upstream / 'round1_relative_position_contract.json')
    hashes.update(round_contract['locked_inputs']['source_hashes'])
    hashes.update({str(ROOT / name): sha(ROOT / name) for name in SOURCES})
    # The passed config, lock and original contracts are immutable input records.
    for path in [config, upstream / 'source_lock.json', upstream / 'runtime_lock_direction_v2.json',
                 upstream / 'round1_relative_position_contract.json', upstream / 'search_design.json']:
        hashes[str(path.resolve())] = sha(path)
    verify_hashes(hashes, ROOT)
    # Canonicalize relative v2 keys, if a future producer uses them.
    hashes = {str(Path(p) if Path(p).is_absolute() else ROOT / p): h for p, h in hashes.items()}
    write(lock_path, {'version': 1, 'config_path': str(config.resolve()), 'created_unix': time.time(),
                      'hashes': hashes, 'claim_boundary': plan['claim_boundary'],
                      'selection_rule_fixed_before_selection': True,
                      'upstream_root': str(upstream.resolve()), 'author_acceptance': False})
    fixed_json(out / 'analysis_plan.json', plan)
    return plan, lock_path


def wait_upstream(upstream, out, plan, lock, once=False):
    while True:
        verify_lock(lock)
        path = upstream / 'status.json'
        current = read(path) if path.exists() else {'status': 'MISSING'}
        name = current['status']
        if name == COMPLETE:
            return True
        if name == 'FAILED':
            raise RuntimeError(f'upstream failed: {current.get("reason", current)}')
        if name == 'ROUND1_COMPLETE_NO_QUALIFIED_GEOMETRY':
            status(out, 'STOPPED_NO_QUALIFIED_GEOMETRY', upstream=current)
            return False
        service = subprocess.run(['systemctl', '--user', 'is-active', plan['upstream_service']],
                                 capture_output=True, text=True)
        # A completed status is evaluated above; do not start off a report that exists
        # while the upstream plot/final completion validation is still running.
        if service.returncode != 0 and not once:
            raise RuntimeError(f'upstream service is {service.stdout.strip()}; status={name}')
        status(out, 'WAITING_UPSTREAM', upstream_status=name,
               upstream_phase=current.get('phase'), upstream_complete=current.get('complete'),
               upstream_total=current.get('total'), upstream_service=plan['upstream_service'],
               dependency='all round1 phases, six-seed confirmation and final output completion',
               next_phase='handoff_then_native_ZM_scan', runtime_lock_sha256=sha(lock))
        if once:
            return False
        time.sleep(plan['runtime']['poll_seconds'])


def make_handoff(upstream, out, plan, lock):
    path = out / 'development_substrate.json'
    if path.exists():
        handoff = read(path)
        verify_hashes(handoff['input_hashes'])
        if handoff['runtime_lock_sha256'] != sha(lock):
            raise RuntimeError('resumed handoff runtime changed')
        return path
    if read(upstream / 'status.json')['status'] != COMPLETE:
        raise RuntimeError('upstream is not completely finished')
    final = read(upstream / 'final_search_report.json')
    if final['status'] != COMPLETE or final['round_contract_sha256'] != sha(upstream / 'round1_relative_position_contract.json'):
        raise RuntimeError('invalid final round1 report')
    confirmation = read(upstream / 'confirmation/aggregate.json')
    if final['confirmation'] != confirmation:
        raise RuntimeError('final report does not match confirmation aggregate')
    selection = read(upstream / 'selection/aggregate.json')
    nominees = read(upstream / 'confirmation_nominees.json')
    if not nominees['selection_complete_before_confirmation']:
        raise RuntimeError('confirmation selection contract missing')
    candidate, selected, confirmed = choose_development_candidate(
        selection, confirmation, nominees, plan['confirmation_seeds'])
    cfg = read(upstream / 'confirmation/execution_config.json')
    manifest = read(upstream / 'confirmation/candidate_manifest.json')
    snapshot = read(upstream / 'confirmation/runtime_snapshot.json')
    verify_hashes(snapshot['input_hashes'])
    if candidate != next(r for r in manifest['candidates'] if r['candidate_id'] == candidate['candidate_id']):
        raise RuntimeError('nominee differs from the executed confirmation candidate')
    if manifest['config_sha256'] != sha(upstream / 'confirmation/execution_config.json'):
        raise RuntimeError('confirmation manifest mismatch')
    completion = read(upstream / 'confirmation/completion.json')
    expected_jobs = len(nominees['candidates']) * len(plan['confirmation_seeds'])
    if completion['status'] != 'ALL_WORKERS_COMPLETE' or completion['jobs'] != expected_jobs:
        raise RuntimeError('confirmation incomplete')
    input_paths = [upstream / p for p in (
        'selection/aggregate.json', 'confirmation/aggregate.json', 'confirmation_nominees.json',
        'confirmation/execution_config.json', 'confirmation/candidate_manifest.json',
        'confirmation/runtime_snapshot.json', 'confirmation/completion.json', 'final_search_report.json')]
    # Check every confirmed candidate/seed against producer checksums before forwarding any one.
    for row in confirmation['candidates']:
        for unit in row['units']:
            worker_path = upstream / 'confirmation/workers' / f'{row["candidate_id"]}_seed_{unit["seed"]}.json'
            worker = read(worker_path)
            if (sha(worker_path) != unit['worker_sha256'] or worker['status'] != 'REV12ND_NODE_WORKER_COMPLETE'
                    or sha(worker['arrays']['path']) != worker['arrays']['sha256']
                    or worker['provenance']['source_hash_snapshot']['sha256'] != sha(upstream / 'confirmation/runtime_snapshot.json')):
                raise RuntimeError(f'confirmation worker is incomplete or changed: {worker_path}')
            input_paths.append(worker_path)
    networks = {str(s): cfg['corrected_networks'][str(s)] for s in plan['analysis_topology_seeds']}
    for seed, record in networks.items():
        if record['status'] != 'CORRECTED_GRAPH_VALIDATED' or sha(record['path']) != record['sha256']:
            raise RuntimeError(f'bad corrected graph seed {seed}')
        if any(p['self_edges'] or not p['exact_expected_degree'] for p in record['pathways'].values()):
            raise RuntimeError('graph structural audit failed')
        if record['sha256'] != next(u['graph_sha256'] for u in confirmed['units'] if u['seed'] == int(seed)):
            raise RuntimeError('handoff graph differs from confirmation')
    # Match the upstream worker's local-first artifact resolution exactly.
    transition = ROOT / cfg['inputs']['transition_config']['path']
    if not transition.exists():
        transition = ART / cfg['inputs']['transition_config']['path']
    if sha(transition) != cfg['inputs']['transition_config']['sha256']:
        raise RuntimeError('transition config changed')
    input_paths.append(transition)
    handoff = {'status': 'DEVELOPMENT_SUBSTRATE_QUALIFIED_FOR_FIG5',
               'candidate': candidate, 'selection_result': selected, 'confirmation_result': confirmed,
               'selection_rule': plan['candidate_rule'], 'runtime_lock_sha256': sha(lock),
               'input_hashes': {str(p.resolve()): sha(p) for p in input_paths},
               'transition_config': str(transition), 'artifact_root': str(ART),
               'network_cache': cfg['network_cache'], 'networks': networks,
               'source_search': str(upstream), 'final_substrate_frozen': False,
               'development_parameters_locked': True, 'author_acceptance': False,
               'patient_heldout_opened': False, 'ictal_opened': False,
               'claim_boundary': plan['claim_boundary']}
    write(path, handoff)
    return path


def jobs_for(configs, phase, plan):
    seeds = plan['pilot_dynamics_seeds'] if phase == 'pilot' else plan['validation_dynamics_seeds']
    return [dict(job_id=f'{phase}_{cfg["config_id"]}_t{topo}_d{dyn}', kind='trajectory', phase=phase,
                 config=cfg, topology_seed=topo, dynamics_seed=dyn, duration_ms=plan['duration_ms'])
            for cfg in configs for topo, dyn in zip(plan['analysis_topology_seeds'], seeds)]


def run_jobs(jobs, out, plan, lock, handoff, phase):
    paths = {}
    for job in jobs:
        p = out / 'jobs' / f'{job["job_id"]}.json'
        fixed_json(p, job)
        paths[job['job_id']] = p
    workers = out / 'workers'; workers.mkdir(exist_ok=True)
    logs = out / 'run_logs'; logs.mkdir(exist_ok=True)
    lock_hash, handoff_hash = sha(lock), sha(handoff)
    def done(job):
        p = workers / f'{job["job_id"]}.json'
        if not p.exists():
            return False
        d = read(p)
        if (d['status'] != 'FIG5_WORKER_COMPLETE' or d['job_sha256'] != sha(paths[job['job_id']])
                or d['runtime_lock_sha256'] != lock_hash or d['handoff_sha256'] != handoff_hash
                or sha(d['arrays']['path']) != d['arrays']['sha256']):
            raise RuntimeError(f'existing worker output is stale: {p}')
        return True
    pending = [j for j in jobs if not done(j)]
    active = {}
    completed = len(jobs) - len(pending)
    resources = []
    failure = None
    limits = plan['runtime']
    budget = float(limits['worker_address_space_gib'])
    try:
        while pending or active:
            verify_lock(lock)
            for name, worker in list(active.items()):
                sample = proc_memory(worker['process'].pid)
                for k, value in sample.items():
                    worker['peaks'][k] = max(value, worker['peaks'].get(k, 0.))
                code = worker['process'].poll()
                if code is None:
                    continue
                worker['stream'].close(); del active[name]
                resources.append(dict(job_id=name, exit_code=code, **worker['peaks']))
                if code != 0 or not done(worker['job']):
                    failure = f'worker failed: {name}, exit={code}; see {logs / (name + ".log")}'
                else:
                    completed += 1
            disk_free = shutil.disk_usage(out).free / 1024**3
            if disk_free < limits['minimum_free_disk_gib']:
                failure = f'free disk below reserve: {disk_free:.1f} GiB'
            if not failure:
                rss = [proc_memory(w['process'].pid).get('VmRSS', 0.) for w in active.values()]
                slots = admission_slots(available_gib(), limits['memory_reserve_gib'], budget,
                                        rss, limits['maximum_workers'])
                for _ in range(min(slots, len(pending))):
                    job = pending.pop(0); name = job['job_id']
                    stream = open(logs / f'{name}.log', 'a')
                    command = [PYTHON, str(ROOT / 'scripts/run_topic4_xy_fig5_worker.py'),
                               '--job', str(paths[name]), '--runtime-lock', str(lock), '--handoff', str(handoff),
                               '--out', str(workers / f'{name}.json'), '--memory-gib', str(budget)]
                    process = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
                    active[name] = dict(process=process, stream=stream, job=job, peaks={})
            status(out, 'DRAINING_AFTER_FAILURE' if failure else 'RUNNING_FIG5', phase=phase,
                   complete=completed, total=len(jobs), running=len(active), pending=len(pending),
                   memory_available_gib=available_gib(), worker_address_space_gib=budget,
                   memory_reserve_gib=limits['memory_reserve_gib'], maximum_workers=limits['maximum_workers'],
                   active=[{'job_id': k, 'pid': w['process'].pid} for k, w in active.items()], failure=failure)
            write(out / f'{phase}_worker_resources.json', {'completed_this_invocation': resources})
            if failure and not active:
                raise RuntimeError(failure)
            if pending or active:
                time.sleep(5)
    finally:
        # Never abandon our worker children or interfere with unrelated science jobs.
        for w in active.values():
            w['process'].wait(); w['stream'].close()
    result = [read(workers / f'{j["job_id"]}.json') for j in jobs]
    fixed_json(out / f'{phase}_results.json', {'status': 'ALL_WORKERS_COMPLETE', 'results': result})
    return result


def plot(out, lock):
    verify_lock(lock)
    subprocess.run([PYTHON, str(ROOT / 'scripts/paper_figures/plot_topic4_xy_fig5_followup.py'),
                    '--out', str(out)], cwd=ROOT, env=ENV, check=True)


def execute(upstream, out, plan, lock):
    try:
        handoff = make_handoff(upstream, out, plan, lock)
    except ValueError as exc:
        status(out, 'STOPPED_CANDIDATE_NOT_QUALIFIED', reason=str(exc), no_historical_fallback=True)
        return
    configs = parameter_grid(plan)
    fixed_json(out / 'parameter_design.json', {'configs': configs, 'fixed_before_simulation': True})
    pilot = run_jobs(jobs_for(configs, 'pilot', plan), out, plan, lock, handoff, 'pilot')
    plot(out, lock)
    selected = choose_mechanism_config(configs, pilot)
    fixed_json(out / 'mechanism_selection.json', {'selected_config': selected,
               'rule': plan['mechanism_selection_rule'], 'selection_uses_pilot_only': True})
    if selected is None:
        status(out, 'COMPLETE_SCAN_NO_QUALIFIED_TRANSITION',
               reason='No Z/M point has >=2/3 tonic trajectories with >=2 s preparation and >=2 returned population excursions.',
               scan_complete=True, spatial_probe_estimable=False)
        return
    # Paired controls use the selected time constants, threshold and adaptation gain.
    controls = [dict(selected, config_id='validation_slow_off', family='control', use_z=False, use_m=False),
                dict(selected, config_id='validation_clamp_z', family='control', use_z=False)]
    validation = run_jobs(jobs_for([selected] + controls, 'validation', plan), out, plan, lock,
                          handoff, 'validation')
    active = [r for r in validation if r['job']['config']['config_id'] == selected['config_id']]
    valid = [r for r in active if r['trajectory']['qualified_pretransition']]
    plot(out, lock)
    baseline = [r for r in validation if r['job']['config']['config_id'] == 'validation_slow_off']
    if any(r['trajectory']['classification'] != 'NO_TONIC_WITHIN_HORIZON'
           or r['trajectory']['pretransition_returned_population_excursions'] < 2 for r in baseline):
        status(out, 'COMPLETE_SCAN_BASELINE_NOT_REPLICATED', scan_complete=True,
               reason='Slow-off baseline did not retain bounded recurrent events in all three new noise seeds.',
               spatial_probe_estimable=False)
        return
    if len(valid) < 2:
        status(out, 'COMPLETE_SCAN_TRANSITION_NOT_REPLICATED', scan_complete=True,
               qualified_validation_seeds=len(valid), spatial_probe_estimable=False)
        return
    probes = []
    for row in valid:
        job = dict(row['job'])
        path = out / 'workers' / f'{job["job_id"]}.json'
        job.update(job_id='probe_' + job['job_id'], kind='probe', phase='probe', probe=plan['probe'],
                   parent_result={'path': str(path), 'sha256': sha(path)})
        probes.append(job)
    run_jobs(probes, out, plan, lock, handoff, 'probe')
    plot(out, lock)
    status(out, 'COMPLETE_FIG5_DEVELOPMENT_ANALYSIS', scan_complete=True,
           spatial_probe_estimable=True, qualified_validation_seeds=len(valid),
           author_acceptance=False, final_substrate_frozen=False,
           figures=str(out / 'figures'), claim_boundary=plan['claim_boundary'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, default=UPSTREAM)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--once', action='store_true', help='check dependency once without waiting')
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    out = (args.out or upstream / 'fig5_followup').resolve()
    out.mkdir(parents=True, exist_ok=True)
    guard = open(out / 'controller.lock', 'a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        plan, lock = prepare(upstream, out, args.config)
        if args.prepare_only:
            status(out, 'PREPARED_WAITING_UPSTREAM', runtime_lock_sha256=sha(lock))
            print(str(out)); return
        if wait_upstream(upstream, out, plan, lock, once=args.once):
            execute(upstream, out, plan, lock)
    except Exception as exc:
        status(out, 'FAILED', reason=str(exc))
        raise


if __name__ == '__main__':
    main()
