#!/usr/bin/env python3
"""Direction-aware amendment using immutable existing simulation contracts.

The original worker sources remain unchanged while their trajectory outputs
are reused. New analysis/controller sources have a separate immutable lock.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_topic4_xy_research as base
from scripts.prepare_topic4_xy_direction_objective import prepare, PATH as OBJECTIVE_PATH
from src.topic4_xy_direction import (onset_directions, direction_histogram,
    direction_distance, direction_summary, core_alignment_penalty, admission_slots)

OUT = base.OUT
RESERVE_GIB = 40.
MIN_BUDGET = {'canary': 8., 'global': 10., 'refinement': 10., 'selection': 15., 'confirmation': 25.}
NEW_SOURCES = ['src/topic4_xy_direction.py', 'scripts/prepare_topic4_xy_direction_objective.py',
               'scripts/run_topic4_xy_direction_research.py',
               'scripts/paper_figures/plot_topic4_xy_direction_target.py']


def proc_memory(pid):
    try:
        fields = {line.split(':')[0]: line.split(':', 1)[1].strip()
                  for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line}
    except (FileNotFoundError, ProcessLookupError):
        return None
    if fields['State'].startswith('Z'):
        return None
    return {k: float(fields.get(k, '0 kB').split()[0]) / 1024**2
            for k in ('VmRSS', 'VmHWM', 'VmSize', 'VmPeak')}


def lock_amendment():
    path = OUT / 'runtime_lock_direction_v2.json'
    expected = {str(ROOT / p): base.sha(ROOT / p) for p in NEW_SOURCES}
    expected[str(OBJECTIVE_PATH)] = base.sha(OBJECTIVE_PATH)
    if path.exists():
        if base.read(path)['hashes'] != expected:
            raise RuntimeError('direction amendment changed after launch')
    else:
        base.write(path, {'hashes': expected, 'created_unix': time.time(),
                          'worker_source_lock': str(OUT / 'source_lock.json')})
    return expected


def verify_amendment(lock):
    for path, expected in lock.items():
        if base.sha(path) != expected:
            raise RuntimeError(f'amendment file changed: {path}')


def finish_old_controller():
    """Only retire the paused controller after every old child finishes."""
    path = OUT / 'direction_amendment_handoff.json'
    handoff = base.read(path)
    if handoff.get('status') == 'DRAIN_COMPLETE_OLD_CONTROLLER_RETIRED':
        return
    pid = handoff['old_controller_pid']
    parent = Path(f'/proc/{pid}/status')
    if parent.exists() and '\nState:\tT ' not in parent.read_text():
        raise RuntimeError('old controller is not paused; refuse overlapping dispatch')
    while True:
        live = [p for p in handoff['child_pids'] if proc_memory(p) is not None]
        if not live:
            break
        base.write(OUT / 'status.json', {'status': 'DRAINING_OLD_WORKERS', 'running': len(live),
            'phase': 'global', 'active_pids': live, 'updated_unix': time.time()})
        time.sleep(5)
    current = int(subprocess.check_output(['systemctl', '--user', 'show', handoff['old_service'],
                                         '-p', 'MainPID', '--value'], text=True))
    if current not in (0, pid):
        raise RuntimeError('old service has been replaced; refuse to stop it')
    if current:
        # SIGKILL here reaches only an idle stopped controller and already
        # completed zombies. No active scientific worker remains.
        subprocess.run(['systemctl', '--user', 'kill', '--signal=SIGKILL', handoff['old_service']], check=True)
        subprocess.run(['systemctl', '--user', 'stop', handoff['old_service']], check=True)
    handoff.update(status='DRAIN_COMPLETE_OLD_CONTROLLER_RETIRED', retired_unix=time.time())
    base.write(path, handoff)


def phase_inputs(phase, rows, lock):
    directory = OUT / phase
    cp, mp, sp = [directory / p for p in ('execution_config.json', 'candidate_manifest.json', 'runtime_snapshot.json')]
    if sp.exists():
        if base.read(mp)['candidates'] != rows:
            raise RuntimeError(f'resumed {phase} candidate manifest differs')
        snap = base.read(sp)
        if snap['source_hashes'] != lock:
            raise RuntimeError('resumed worker source lock differs')
        for path in (cp, mp):
            if snap['input_hashes'].get(str(path.resolve())) != base.sha(path):
                raise RuntimeError('resumed worker inputs changed')
        cfg = base.read(cp)
        if cfg['search']['fit_network_seeds'] != base.SEEDS[phase]:
            raise RuntimeError('resumed seed pool differs')
        return cp, mp, sp
    return base.phase_contract(phase, rows, lock)


def run_phase(phase, rows, worker_lock, amendment_lock, maximum_workers):
    base.verify_sources(worker_lock); verify_amendment(amendment_lock)
    cp, mp, sp = phase_inputs(phase, rows, worker_lock)
    directory = OUT / phase; workers = directory / 'workers'; logs = directory / 'run_logs'
    workers.mkdir(exist_ok=True); logs.mkdir(exist_ok=True)
    snapshot_sha = base.sha(sp)
    jobs = [(r['candidate_id'], seed) for r in rows for seed in base.SEEDS[phase]]
    def done(job):
        cid, seed = job; p = workers / f'{cid}_seed_{seed}.json'
        if not p.exists() or not p.with_suffix('.npz').exists():
            return False
        d = base.read(p)
        return (d.get('status') == 'REV12ND_NODE_WORKER_COMPLETE'
                and d['candidate_id'] == cid and d['seed'] == seed
                and d['arrays']['sha256'] == base.sha(p.with_suffix('.npz'))
                and d['provenance']['source_hash_snapshot']['sha256'] == snapshot_sha)
    complete = [j for j in jobs if done(j)]
    pending = [j for j in jobs if j not in complete]; active = {}; failure = None
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    resource_path = directory / 'worker_resources_v2.json'
    resources = base.read(resource_path) if resource_path.exists() else {'completed': [], 'observed_peak_gib': 0.}
    peak_observed = resources['observed_peak_gib']
    last_status = 0
    try:
        while pending or active:
            for job, unit in list(active.items()):
                proc = unit['process']; mem = proc_memory(proc.pid)
                if mem:
                    for k, value in mem.items(): unit['peaks'][k] = max(unit['peaks'].get(k, 0.), value)
                    peak_observed = max(peak_observed, mem['VmPeak'])
                code = proc.poll()
                if code is None: continue
                unit['stream'].close(); del active[job]
                resources['completed'].append({'candidate_id': job[0], 'seed': job[1], 'exit_code': code,
                    'peaks_gib': unit['peaks'], 'address_space_limit_gib': unit['budget'],
                    'elapsed_seconds': time.time() - unit['start']})
                if code != 0 or not done(job):
                    failure = f'{phase} worker {job} failed, exit {code}; inspect log, no additional dispatch'
                else: complete.append(job)
            resources['observed_peak_gib'] = peak_observed
            base.write(resource_path, resources)
            disk_free = shutil.disk_usage(OUT).free / 1024**3
            if disk_free < 30:
                failure = 'Less than 30 GiB free disk; no additional dispatch'
            # Hard RLIMIT_AS protects each NEW worker from uncontrolled growth;
            # reserve the same full amount in admission, not its initial RSS.
            budget = max(MIN_BUDGET[phase], float(np.ceil(1.5 * peak_observed)))
            rss = [(proc_memory(u['process'].pid) or {}).get('VmRSS', 0.) for u in active.values()]
            budget = max([budget] + [u['budget'] for u in active.values()])
            available = base.available_gib()
            slots = admission_slots(available, rss, budget, maximum_workers, RESERVE_GIB)
            if not failure:
                for unused in range(min(slots, len(pending))):
                    cid, seed = pending.pop(0); stream = open(logs / f'{cid}_seed_{seed}.log', 'w')
                    command = ['/usr/bin/prlimit', f'--as={int(budget * 1024**3)}', '--', base.PYTHON,
                        str(ROOT / 'scripts/run_topic4_rev12_node_worker.py'), '--config', str(cp),
                        '--candidate-id', cid, '--seed', str(seed), '--expected-commit', commit,
                        '--runtime-manifest', str(sp), '--artifact-root', str(base.ART),
                        '--out-json', str(workers / f'{cid}_seed_{seed}.json'),
                        '--out-npz', str(workers / f'{cid}_seed_{seed}.npz')]
                    proc = subprocess.Popen(command, cwd=ROOT, env=base.ENV, stdout=stream, stderr=subprocess.STDOUT)
                    active[(cid, seed)] = {'process': proc, 'stream': stream, 'peaks': {},
                                          'budget': budget, 'start': time.time()}
            if time.time() - last_status >= 5:
                base.write(OUT / 'status.json', {'status': 'DRAINING_AFTER_FAILURE' if failure else 'RUNNING',
                    'phase': phase, 'objective_version': 'xy_direction_v2', 'total': len(jobs),
                    'complete': len(complete), 'running': len(active), 'pending': len(pending),
                    'memory_available_gib': available, 'system_reserve_gib': RESERVE_GIB,
                    'worker_budget_gib': budget, 'maximum_workers': maximum_workers,
                    'observed_worker_peak_virtual_gib': peak_observed, 'failure': failure,
                    'updated_unix': time.time(), 'active': [{'candidate_id': j[0], 'seed': j[1],
                        'pid': u['process'].pid, 'address_space_limit_gib': u['budget']}
                        for j, u in active.items()]})
                last_status = time.time()
            if failure and not active: raise RuntimeError(failure)
            if active or pending: time.sleep(2)
    except BaseException:
        # Ordinary controller errors must not discard already running work.
        for unit in active.values(): unit['process'].wait(); unit['stream'].close()
        raise
    base.verify_sources(worker_lock); verify_amendment(amendment_lock)
    base.write(directory / 'completion.json', {'status': 'ALL_WORKERS_COMPLETE', 'jobs': len(jobs),
        'runtime_snapshot_sha256': snapshot_sha, 'objective_version': 'xy_direction_v2'})


def score_views(tables, xy, contract):
    view = onset_directions(np.concatenate(tables), xy)
    hist = direction_histogram(view)
    return {'distance': direction_distance(hist, contract['patient_direction_histogram']),
            'histogram': None if hist is None else hist.tolist(), 'summary': direction_summary(view)}


def aggregate(phase, rows, contract):
    report = base.aggregate(phase, rows)
    base.write(OUT / phase / 'aggregate_cloud_v1.json', report)
    training, objective = base.training_contract()
    xy = np.asarray(contract['contact_xy_mm'])
    target_axis = contract['patient_direction_summary']['axial_angle_deg']
    for row in report['candidates']:
        tables = []; full_maps = []
        for unit in row['units']:
            p = OUT / phase / 'workers' / f"{row['candidate_id']}_seed_{unit['seed']}.npz"
            with np.load(p) as a:
                if not np.allclose(a['contact_xy_mm'], xy, atol=1e-10, rtol=0):
                    raise RuntimeError('model and patient direction coordinates differ')
                if list(a['contact_names'].astype(str)) != contract['contact_names']:
                    raise RuntimeError('direction contact order differs')
                returned = np.asarray(a['event_returned'], bool)
                table = np.asarray(a['onsets'], float)[returned]; tables.append(table)
                maps = np.asarray(a['source_onset_maps_ms'], float)
                if maps.shape[0] != len(returned): raise RuntimeError('source/contact event correspondence differs')
                maps = maps[returned]; bin_mm = float(a['source_bin_mm'])
                full_maps.append(maps.reshape(len(maps), -1) if len(maps) else np.empty((0, maps.shape[1]*maps.shape[2])))
                y, x = np.indices(maps.shape[1:])
                source_xy = np.column_stack([(x.ravel()+.5)*bin_mm, (y.ravel()+.5)*bin_mm])
            unit['contact_direction'] = score_views([table], xy, contract)
            unit['full_sheet_direction'] = direction_summary(onset_directions(full_maps[-1], source_xy))
            uv = objective.component_vector(table, training['reference'], training['groups'],
                training['pairs'], training['embedding'], composite=True, components=())
            unit['D_cloud'] = uv['D_cloud_composite']
            dd = unit['contact_direction']['distance']
            unit['J_direction'] = None if unit['D_cloud'] is None or dd is None else (
                unit['D_cloud'] / contract['normalization']['D_cloud'] + dd / contract['normalization']['D_direction'])
        direction = score_views(tables, xy, contract)
        row['direction'] = direction; row['D_direction'] = direction['distance']
        row['full_sheet_direction'] = direction_summary(onset_directions(np.concatenate(full_maps), source_xy))
        prior = core_alignment_penalty(row['node_field']['centers_mm'], target_axis)
        row['core_alignment_penalty'] = prior
        value = None if row['D_cloud'] is None or row['D_direction'] is None else (
            row['D_cloud'] / contract['normalization']['D_cloud'] +
            row['D_direction'] / contract['normalization']['D_direction'])
        row['J_direction'] = value
        row['J_weak_prior'] = None if value is None else value + contract['weak_prior_weight'] * prior
        row['selection_eligible'] = row['selection_eligible'] and value is not None
    feasible = [r for r in report['candidates'] if r['selection_eligible']]
    report['ranking_cloud_only'] = report['ranking']
    report['ranking'] = [r['candidate_id'] for r in sorted(feasible,key=lambda r:(r['J_direction'],r['candidate_id']))]
    report['ranking_weak_prior'] = [r['candidate_id'] for r in sorted(feasible,key=lambda r:(r['J_weak_prior'],r['candidate_id']))]
    report['direction_pareto_candidate_ids'] = [feasible[i]['candidate_id'] for i in base.pareto_indices(
        [[r['D_cloud'],r['D_direction']] for r in feasible])] if feasible else []
    report['prior_sensitivity_rankings'] = {str(w): [r['candidate_id'] for r in sorted(feasible,
        key=lambda r:(r['J_direction']+w*r['core_alignment_penalty'],r['candidate_id']))]
        for w in contract['prior_sensitivity_weights']}
    report.update(objective_version='xy_direction_v2', direction_objective_sha256=base.sha(OBJECTIVE_PATH),
                  primary_selection_uses_core_prior=False)
    if phase == 'confirmation':
        contrasts = []; rng = np.random.default_rng(20260909)
        byid = {r['candidate_id']:r for r in report['candidates']}
        for row in report['candidates']:
            for control in ('control_old_edge','control_historical_matched'):
                if row['candidate_id'] == control: continue
                ref = {u['seed']:u for u in byid[control]['units']}
                delta = [u['J_direction'] - ref[u['seed']]['J_direction'] for u in row['units']
                    if u['J_direction'] is not None and ref[u['seed']]['J_direction'] is not None]
                ci = None
                if len(delta) == len(base.SEEDS[phase]):
                    boot = np.asarray(delta)[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
                    ci = np.quantile(boot,[.025,.975]).tolist()
                contrasts.append({'candidate_id':row['candidate_id'],'control':control,
                    'paired_network_differences':delta,'mean_difference':float(np.mean(delta)) if delta else None,
                    'bootstrap_95_interval':ci,'n_networks':len(delta),
                    'interpretation':'negative favors candidate; network-unit descriptive uncertainty, not patient generalization'})
        report['paired_network_confirmation'] = contrasts
    base.write(OUT / phase / 'aggregate.json', report)
    return report


def shortlist(reports, rows):
    scored = {r['candidate_id']:r for p in reports for r in p['candidates'] if r['selection_eligible']}
    chosen = []
    for domain in ('whole_sheet','interior'):
        eligible = [r for r in scored.values() if r['domain'] == domain]
        if not eligible: continue
        picks = sorted(eligible,key=lambda r:(r['J_direction'],r['candidate_id']))[:2]
        picks += [min(eligible,key=lambda r:(r[k],r['candidate_id']))
                  for k in ('D_cloud','D_direction','J_weak_prior')]
        picks += [min(eligible,key=lambda r:(r['components'][k]['value'],r['candidate_id'])) for k in base.COMPONENTS]
        chosen.extend(r['candidate_id'] for r in picks)
    chosen += ['control_old_edge','control_historical_matched']
    byid = {r['candidate_id']:r for r in rows}
    return [byid[cid] for cid in dict.fromkeys(chosen)]


def refinement(report, rows):
    # Reuse the original geometric proposal function with explicitly chosen
    # primary + weak-prior anchors in each domain. All XY dimensions stay free.
    byid = {r['candidate_id']:r for r in report['candidates']}
    anchors = []
    for domain in ('whole_sheet','interior'):
        primary = [c for c in report['ranking'] if byid[c]['domain'] == domain]
        weak = [c for c in report['ranking_weak_prior'] if byid[c]['domain'] == domain]
        ids = list(dict.fromkeys(primary[:1] + weak[:1] + primary))[:2]
        anchors += ids
    result = base.refinement({**report,'ranking':anchors}, rows)
    base.write(OUT / 'refinement_design.json', {'candidates':result, 'anchors':anchors,
        'selection_objective':'direction primary plus prespecified weak-prior sensitivity',
        'all_four_XY_dimensions_free':True})
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--maximum-workers',type=int,default=24)
    parser.add_argument('--validate-only',action='store_true')
    args = parser.parse_args()
    if not 1 <= args.maximum_workers <= 24: raise ValueError('maximum workers must be 1..24')
    contract = prepare()
    worker_lock = base.read(OUT / 'source_lock.json')['source_hashes']
    base.verify_sources(worker_lock)
    if args.validate_only:
        print({'status':'VALIDATED','objective':str(OBJECTIVE_PATH),'original_sources_unchanged':True});return
    amendment_lock = lock_amendment()
    finish_old_controller()
    controller_lock = open(OUT / 'controller.lock','a')
    fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows = base.prepare_design()['candidates']
    base.write(OUT / 'active_search_contract.json', {'original_design':str(OUT / 'search_design.json'),
        'objective_amendment':str(OBJECTIVE_PATH),'objective_sha256':base.sha(OBJECTIVE_PATH),
        'controller_sha256':base.sha(Path(__file__)), 'maximum_workers':args.maximum_workers,
        'memory_reserve_gib':RESERVE_GIB,'worker_address_space_minimum_gib':MIN_BUDGET,
        'EE_EtoI_ZM_learning':'off','final_substrate_frozen':False,
        'old_s39_status':'HISTORICAL_CONTROL_ONLY_NOT_ACCEPTED_FINAL_SUBSTRATE',
        'confirmation_selection':'best primary and weak-prior candidate per domain plus two controls; no confirmation reselection'})
    def run(phase, candidates):
        run_phase(phase,candidates,worker_lock,amendment_lock,args.maximum_workers)
        return aggregate(phase,candidates,contract)
    global_report = run('global',rows)
    refined = refinement(global_report,rows); reports = [global_report]; all_rows = rows + refined
    if refined: reports.append(run('refinement',refined))
    selected_rows = shortlist(reports,all_rows)
    base.write(OUT / 'development_shortlist.json',{'candidates':selected_rows,'objective_version':'xy_direction_v2'})
    selected_report = run('selection',selected_rows)
    byid = {r['candidate_id']:r for r in selected_report['candidates']}
    nominees=[]
    for domain in ('whole_sheet','interior'):
        for ranking in ('ranking','ranking_weak_prior'):
            ids=[c for c in selected_report[ranking] if byid[c]['domain']==domain]
            nominees += ids[:1]
    if not nominees:
        base.write(OUT/'status.json',{'status':'SEARCH_COMPLETE_NO_QUALIFIED_GEOMETRY','final_substrate_frozen':False});return
    nominees += ['control_old_edge','control_historical_matched']
    source = {r['candidate_id']:r for r in all_rows}; confirmation = [source[c] for c in dict.fromkeys(nominees)]
    base.write(OUT/'confirmation_nominees.json',{'candidates':confirmation,
        'selection_complete_before_confirmation':True,'objective_version':'xy_direction_v2','final_substrate_frozen':False})
    final = run('confirmation',confirmation)
    base.write(OUT/'final_search_report.json',{'status':'XY_SEARCH_COMPLETE_AWAITING_SCIENTIFIC_REVIEW',
        'confirmation':final,'old_s39_status':'HISTORICAL_CONTROL_ONLY','final_substrate_frozen':False,
        'global_optimum_established':False,'patient_heldout_opened':False,
        'next_scientific_step':'Review independent-network event and signed-direction distributions; freeze VTH only after qualification, then test EE/EtoI/ZM.'})
    subprocess.run([base.PYTHON,str(ROOT/'scripts/paper_figures/plot_topic4_xy_direction_target.py')],cwd=ROOT,env=base.ENV,check=True)
    base.write(OUT/'status.json',{'status':'XY_SEARCH_COMPLETE_AWAITING_SCIENTIFIC_REVIEW',
                               'objective_version':'xy_direction_v2','final_substrate_frozen':False})


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        base.write(OUT/'status.json',{'status':'FAILED','reason':str(exc),'objective_version':'xy_direction_v2','updated_unix':time.time()})
        raise
