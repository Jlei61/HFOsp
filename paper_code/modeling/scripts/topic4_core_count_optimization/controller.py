"""Resumable, resource-bounded background dispatch and automatic final plots."""
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import fcntl
import shutil
import subprocess
import psutil
from modeling.scripts.topic4_core_count_optimization.common import *
pass
from modeling.scripts.topic4_core_count_optimization.feedback_schedule import training_queue, dispatch_rank
from modeling.scripts.topic4_core_count_optimization.resource_schedule import validated_limits, reservations
from modeling.scripts.topic4_core_count_optimization.arm_schedule import validated_core_counts, canary_queue
from modeling.scripts.topic4_core_count_optimization.priority_schedule import validated_tiers, completed_core_counts, dispatch_phase, partition_tasks
from modeling.scripts.topic4_core_count_optimization.stage_schedule import stage_settings, stage_progress, stage_dispatch_rank

def resource_limits(defaults):
    """A live concurrency ceiling leaves the frozen scientific plan unchanged."""
    path = OUT / 'resource_control.json'
    return validated_limits(defaults, read(path) if path.exists() else {}, len(os.sched_getaffinity(0)))

def target(task):
    kind, k, restart, value = task
    if kind == 'canary':
        return OUT / 'canary' / f'core{k}' / 'result.json'
    if kind == 'check':
        return OUT / 'canary_validation' / f'core{k}.json' if k else OUT / 'canary_validation.json'
    d = run_dir(k, restart)
    if kind == 'train':
        return d / 'scores' / f'{value}.json'
    if kind == 'propose':
        return d / 'proposals' / f'batch_{int(value):02d}.json'
    if kind == 'confirmation':
        cid, noise = value.rsplit(':', 1)
        return d / 'confirmation' / f'{cid}_{noise}' / 'result.json'
    raise ValueError(task)

def command(task):
    kind, k, restart, value = task
    if kind == 'check':
        return [PY, '-u', str(HERE / 'check.py'), '--canaries'] + (['--cores', str(k)] if k else [])
    if kind == 'propose':
        return [BO_PY, '-u', str(HERE / 'search.py'), '--cores', str(k), '--restart', str(restart), '--batch', str(value)]
    args = [PY, '-u', str(HERE / 'worker.py'), '--cores', str(k), '--restart', str(restart), '--phase', kind]
    if kind == 'confirmation':
        cid, noise = value.rsplit(':', 1)
        return args + ['--candidate', cid, '--noise', noise]
    return args + ['--candidate', value]

def main():
    verify_sources()
    plan = read(OUT / 'plan.json')
    res = plan['resources']
    active = {}
    policy_path = OUT / 'dispatch_policy.json'
    policy = read(policy_path) if policy_path.exists() else {}
    tiers = validated_tiers(plan['core_counts'], policy)
    stage = stage_settings(plan, policy)
    failures_path = OUT / 'execution_failures.json'
    failures = read(failures_path) if failures_path.exists() else []
    ledger_path = OUT / 'dispatch.jsonl'

    def key(task):
        return '|'.join(map(str, task))

    def attempts(task):
        return sum((f['key'] == key(task) for f in failures))
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            row = json.loads(line)
            task = tuple(row['task'])
            try:
                p = psutil.Process(row['pid'])
                if p.status() != psutil.STATUS_ZOMBIE and abs(p.create_time() - row['created']) < 0.1 and (p.cmdline() == row['command']):
                    active[key(task)] = dict(task=task, pid=p.pid, created=p.create_time(), process=None, log=None)
            except psutil.NoSuchProcess:
                pass
    launch = OUT / 'controller_launch.json'
    if not launch.exists():
        write(launch, dict(started_unix=time.time(), deadline=time.time() + res['dispatch_hours'] * 3600))
    deadline = read(launch)['deadline']
    last_report = 0.0
    last_report_count = -1
    last_launch = 0.0
    psutil.cpu_percent()

    def fail(a, reason):
        failures.append(dict(key=key(a['task']), task=a['task'], pid=a['pid'], reason=reason, time=time.time()))
        write(failures_path, failures)

    def submit(task):
        if key(task) in active or target(task).exists() or attempts(task) >= res['attempts']:
            return False
        cmd = command(task)
        log = (OUT / 'logs' / (key(task).replace('|', '_').replace(':', '_') + '.log')).open('a')
        proc = subprocess.Popen(cmd, cwd=REPO, env=ENV, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        created = psutil.Process(proc.pid).create_time()
        active[key(task)] = dict(task=task, pid=proc.pid, created=created, process=proc, log=log)
        with ledger_path.open('a') as f:
            f.write(json.dumps(dict(task=task, pid=proc.pid, created=created, command=cmd)) + '\n')
        return True
    while True:
        res = resource_limits(plan['resources'])
        memory = psutil.virtual_memory().available / 2 ** 30
        cpu = psutil.cpu_percent()
        rss = {}
        for ident, a in list(active.items()):
            try:
                p = psutil.Process(a['pid'])
                alive = p.is_running() and p.status() != psutil.STATUS_ZOMBIE and (abs(p.create_time() - a['created']) < 0.1)
                if alive:
                    tree = [p] + p.children(recursive=True)
                    rss[ident] = sum((q.memory_info().rss for q in tree if q.is_running())) / 2 ** 30
                    if rss[ident] > res['tree_limit_gib']:
                        for q in reversed(tree):
                            q.terminate()
                        a['guard'] = 'PROCESS_TREE_MEMORY_LIMIT'
            except psutil.NoSuchProcess:
                alive = False
            if not alive:
                if a['process'] is not None:
                    a['process'].poll()
                if a['log'] is not None:
                    a['log'].close()
                if a['task'][0] == 'train' and (not target(a['task']).exists()):
                    _, k, restart, cid = a['task']
                    u = run_dir(k, restart) / 'units' / cid
                    if (u / 'result.json').exists() and (u / 'score.json').exists():
                        write(target(a['task']), read(u / 'score.json'))
                if not target(a['task']).exists():
                    fail(a, a.get('guard', 'PROCESS_EXIT_WITHOUT_OUTPUT'))
                del active[ident]
        if memory < res['emergency_available_gib'] and rss:
            ident = max(rss, key=rss.get)
            if ident in active:
                a = active[ident]
                try:
                    p = psutil.Process(a['pid'])
                    for q in reversed([p] + p.children(recursive=True)):
                        q.terminate()
                    a['guard'] = 'SYSTEM_MEMORY_EMERGENCY'
                except psutil.NoSuchProcess:
                    pass
        workers = []
        helpers = []
        complete = 0
        conf_complete = 0
        streams = []
        validated = validated_core_counts(OUT, plan['core_counts'])
        gate = validated == set(plan['core_counts'])
        workers, helpers = canary_queue(OUT, plan['core_counts'], validated)
        for restart in range(plan['restarts']):
            for k in plan['core_counts']:
                d = run_dir(k, restart)
                rows = [read(p) for p in (d / 'scores').glob('*.json')]
                complete += len(rows)
                stream = dict(core_count=k, restart=restart, completed=len(rows), scorable=sum((r['J'] is not None for r in rows)), confirmation_completed=0, confirmation_required=len(plan['confirmation_noises']), optimization_and_confirmation_complete=False)
                streams.append(stream)
                if stage and k not in stage['core_counts']:
                    stream['stage_status'] = 'ACCEPTED_STAGE_PAUSED_BY_USER'
                    continue
                if k not in validated:
                    continue
                if stage:
                    stream.update(stage_progress(d, rows, stage))
                    if stream['stage_complete'] or stream['stage_review_required']:
                        continue
                    missing, batch = training_queue(d, dict(plan, epochs=stage['epochs']), rows)
                    workers.extend((('train', k, restart, cid) for cid in missing))
                    if batch is not None:
                        helpers.append(('propose', k, restart, str(batch)))
                    continue
                if len(rows) == plan['epochs']:
                    nomination = d / 'nomination.json'
                    if not nomination.exists():
                        good = [r for r in rows if r['J'] is not None]
                        best = min(good, key=lambda r: (r['J'], r['epoch'])) if good else None
                        write(nomination, dict(status='NOMINATED' if best else 'NO_SCORABLE_CANDIDATE', selected=None if best is None else best['candidate'], training_J=None if best is None else best['J'], source='training only', frozen_unix=time.time()))
                    chosen = read(nomination)['selected']
                    if chosen is not None:
                        for noise in plan['confirmation_noises']:
                            task = ('confirmation', k, restart, f'{chosen}:{noise}')
                            if target(task).exists():
                                conf_complete += 1
                                stream['confirmation_completed'] += 1
                            else:
                                workers.append(task)
                    else:
                        stream['confirmation_required'] = 0
                    stream['nomination_status'] = read(nomination)['status']
                    stream['optimization_and_confirmation_complete'] = stream['confirmation_completed'] == stream['confirmation_required']
                    continue
                missing, batch = training_queue(d, plan, rows)
                workers.extend((('train', k, restart, cid) for cid in missing))
                if batch is not None:
                    helpers.append(('propose', k, restart, str(batch)))
        completed_cores = completed_core_counts(streams, plan['core_counts'], plan['restarts'])
        phase_index, dispatch_cores = dispatch_phase(tiers, completed_cores)
        stage_streams = [s for s in streams if stage and s['core_count'] in stage['core_counts']]
        stage_done_cores = {k for k in stage['core_counts'] if k in validated and all((s.get('stage_complete', False) for s in stage_streams if s['core_count'] == k))} if stage else set()
        if stage:
            dispatch_cores = set(stage['core_counts']) - stage_done_cores
            phase_index = 0 if dispatch_cores else None
        dispatch_workers, deferred_workers = partition_tasks(workers, dispatch_cores)
        dispatch_helpers, deferred_helpers = partition_tasks(helpers, dispatch_cores)
        stopped = time.time() > deadline or (OUT / 'STOP_NEW_DISPATCH').exists()
        nworkers = sum((a['task'][0] in ('train', 'canary', 'confirmation') for a in active.values()))
        nhelpers = len(active) - nworkers
        disk = shutil.disk_usage(OUT).free / 2 ** 30
        forecasts, growth, nbuilding = reservations(OUT, active, rss, res, time.time())
        if not stopped and nhelpers < res['max_helpers'] and (memory > res['available_reserve_gib'] + growth + 4):
            for task in dispatch_helpers:
                if submit(task):
                    growth += 4.0
                    break
        if not stopped and nworkers < res['max_workers'] and (nbuilding < res['max_building_workers']) and (cpu + 100.0 / len(os.sched_getaffinity(0)) <= res['cpu_ceiling_percent']) and (disk > res['disk_reserve_gib']) and (memory > res['available_reserve_gib'] + growth + res['projected_worker_gib']) and (time.time() - last_launch > 3):

            def priority(task):
                _, k, restart, cid = task
                if task[0] == 'canary':
                    return (-1, 0, restart, k)
                row = next((s for s in streams if s['core_count'] == k and s['restart'] == restart), {'completed': 0})
                running = sum((a['task'][1:3] == (k, restart) for a in active.values()))
                if stage:
                    return (stage_dispatch_rank(cid), running, restart, k)
                return (dispatch_rank(cid), row['completed'] + running, restart, k)
            for task in sorted(dispatch_workers, key=priority):
                if submit(task):
                    last_launch = time.time()
                    break
        forecasts, growth, nbuilding = reservations(OUT, active, rss, res, time.time())
        pending = [t for t in workers + helpers if not target(t).exists() and key(t) not in active]
        dispatch_pending, deferred_pending = partition_tasks(pending, dispatch_cores)
        permanently_failed = [t for t in dispatch_pending if attempts(t) >= res['attempts']]
        full_finished = gate and complete == plan['train_simulations'] and (not workers) and (not helpers) and (not active)
        finished = stage_done_cores == set(stage['core_counts']) and (not active) if stage else full_finished
        status = 'COMPLETE' if finished else 'DRAINING_DISPATCH_WINDOW' if stopped else 'RUNNING' if gate else 'RUNNING_WITH_PENDING_CANARY' if validated else 'CANARY_RUNNING'
        if stage:
            status = 'STAGE_COMPLETE_REVIEW_PENDING' if finished else 'STAGE_DRAINING_DISPATCH_WINDOW' if stopped else 'STAGE_RUNNING'
            if not active and any((s.get('stage_review_required') for s in stage_streams)):
                status = 'STAGE_REVIEW_REQUIRED'
        if not active and dispatch_pending and (len(permanently_failed) == len(dispatch_pending)):
            status = 'EXECUTION_FAILED_PARTIAL'
        report_ok = True
        if finished or (complete + conf_complete != last_report_count and time.time() - last_report > 120):
            log = (OUT / 'logs/report.log').open('a')
            try:
                report = 'report_four.py' if 4 in plan['core_counts'] else 'report.py'
                subprocess.run([PY, str(HERE / report)], cwd=REPO, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)
                last_report_count = complete + conf_complete
            except Exception as e:
                report_ok = False
                write(OUT / 'report_error.json', dict(error=str(e), time=time.time()))
            finally:
                log.close()
                last_report = time.time()
        if finished and (not report_ok):
            status = 'STAGE_COMPLETE_REPORT_FAILED' if stage else 'SIMULATIONS_COMPLETE_REPORT_FAILED'
        write(OUT / 'status.json', dict(status=status, controller_pid=os.getpid(), train_complete=complete, train_planned=plan['train_simulations'], confirmation_complete=conf_complete, confirmation_planned=plan['confirmation_simulations'], canary_pass=gate, validated_core_counts=sorted(validated), pending_canary_core_counts=sorted(set(plan['core_counts']) - validated), dispatch_policy_path=str(policy_path), dispatch_tiers=tiers, dispatch_phase_index=phase_index, dispatch_core_counts=sorted(dispatch_cores), completed_core_counts=sorted(completed_cores), deferred_core_counts=sorted(set(plan['core_counts']) - completed_cores - stage_done_cores - dispatch_cores), execution_scope='bounded_feedback_stage' if stage else 'original_full_search', stage_id=stage['id'] if stage else None, stage_core_counts=stage['core_counts'] if stage else [], stage_epochs=stage['epochs'] if stage else None, stage_feedback_rounds=stage['feedback_rounds'] if stage else None, stage_train_complete=sum((s.get('stage_evaluated', 0) for s in stage_streams)), stage_train_planned=len(stage['core_counts']) * plan['restarts'] * stage['epochs'] if stage else None, stage_completed_core_counts=sorted(stage_done_cores), paused_core_counts=sorted(set(plan['core_counts']) - set(stage['core_counts'])) if stage else [], confirmations_deferred=bool(stage), pending_dispatch_phase=len(dispatch_pending), pending_deferred_phases=len(deferred_pending), streams=streams, active=[dict(task=a['task'], pid=a['pid'], rss_gib=rss.get(ident), memory_forecast=forecasts[ident]) for ident, a in active.items()], pending_current_batches=len(pending), failures=failures, available_gib=memory, cpu_percent=cpu, disk_free_gib=disk, max_workers=res['max_workers'], original_max_workers=plan['resources']['max_workers'], resource_control_path=str(OUT / 'resource_control.json'), growth_reserve_gib=growth, available_reserve_gib=res['available_reserve_gib'], projected_worker_gib=res['projected_worker_gib'], emergency_available_gib=res['emergency_available_gib'], cpu_ceiling_percent=res['cpu_ceiling_percent'], simulation_worker_gib=res['simulation_worker_gib'], building_workers=nbuilding, max_building_workers=res['max_building_workers'], next_worker_memory_required_gib=res['available_reserve_gib'] + growth + res['projected_worker_gib'], additional_worker_slots_by_memory=max(0, int((memory - res['available_reserve_gib'] - growth) // res['projected_worker_gib'])), dispatch_deadline=deadline, updated_unix=time.time()))
        if finished or status in ('EXECUTION_FAILED_PARTIAL', 'STAGE_REVIEW_REQUIRED') or (stopped and (not active)):
            if stage and finished:
                write(OUT / 'stage_completion.json', dict(status=status, time=time.time(), stage=stage, streams=stage_streams, original_full_plan_complete=full_finished, confirmations_deferred=True, scientific_review='PENDING', figures_human_acceptance='PENDING'))
            elif finished and report_ok:
                write(OUT / 'completion.json', dict(status='COMPUTATION_COMPLETE_SCIENTIFIC_REVIEW_PENDING', time=time.time(), training_complete=complete, confirmation_complete=conf_complete, confirmation_unavailable=plan['confirmation_simulations'] - conf_complete, figures_human_acceptance='PENDING'))
            break
        time.sleep(5)
if __name__ == '__main__':
    with (OUT / 'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
