"""Bounded dependency queue for the first evidence package.

The S_marks chain is prioritised so the frozen-state consumers (G2) can start
before the other four arms finish.  Budget stops, numerical failures and tasks
that never ran all stay visible in the task table (C54).
"""
from __future__ import annotations
from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch

from .train import (RunConfig, ROOT, PACKETS, ARMS, FINAL_SEEDS, run_paired_inner, run_final,
                    source_digest, atomic_json, tag)

SYNTHETIC_TASKS = ('conditional_zero', 'organization_positive')
SYNTHETIC_REALIZATIONS = (901, 902, 903)
SYNTHETIC_ARMS = ('S_stats', 'S_marks', 'S_marks-short')


def acquire_lock(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, 'a+')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    f.seek(0)
    f.truncate()
    f.write(json.dumps(dict(pid=os.getpid(), host=socket.gethostname(), start=time.time())))
    f.flush()
    return f


def arm_config(arm_name, short_minutes, **kw):
    for name, inputs, arm, hh in ARMS:
        if name == arm_name:
            return RunConfig(arm_name=name, inputs=inputs, arm=arm,
                             history_hours=(short_minutes / 60. if hh == 'H_SHORT' else None),
                             short_history_minutes=int(short_minutes), **kw)
    raise ValueError(arm_name)


def build_plan(path, root=ROOT, short_minutes=120, quick=False, subject='epilepsiae_1125'):
    root = Path(root)
    tasks = []
    order = ['S_marks', 'S_stats', 'S_marks-short', 'B_marks', 'B_stats']

    def add(kind, priority, **kw):
        task = dict(id=f't{len(tasks) + 1:03d}', kind=kind, priority=priority, depends=[], **kw)
        tasks.append(task)
        return task['id']

    small = dict(batch_size=4, microbatch=4, train_paths=2, eval_paths=4, eval_stride=180,
                 max_updates=2, extended_updates=2, eval_every=1, eval_chunk=6) if quick else {}
    human = {}
    for rank, arm_name in enumerate(order):
        cfg = arm_config(arm_name, short_minutes, subject=subject, out_dir=str(root / 'runs'), **small)
        recipe = str(root / 'recipes' / f'{arm_name}.json')
        pid = add('paired_inner', 10 * rank, config=asdict(cfg), output=recipe)
        human[arm_name] = dict(recipe=recipe, finals=[])
        for si, seed in enumerate(FINAL_SEEDS if not quick else FINAL_SEEDS[:1]):
            fcfg = replace(cfg, stage='outer', seed=int(seed), recipe_path=recipe)
            fid = add('final', 10 * rank + 1 + (0 if si == 0 else 100), config=asdict(fcfg))
            tasks[-1]['depends'] = [pid]
            selected = str(root / 'runs' / tag(fcfg) / 'selected.pt')
            human[arm_name]['finals'].append(dict(seed=int(seed), selected=selected, task=fid))
            if arm_name == 'S_marks':
                bid = add('consumers', 10 * rank + 2 + (0 if si == 0 else 100), selected=selected,
                          output=str(root / 'frozen' / tag(fcfg)), quick=quick)
                tasks[-1]['depends'] = [fid]
    # Synthetic diagnostics are independent of the human chain and run on the second GPU.
    if not quick:
        for task_name in SYNTHETIC_TASKS:
            for realization in SYNTHETIC_REALIZATIONS:
                subject_id = f'synthetic_{task_name}_{realization}'
                packets = str(root / 'synthetic' / 'packets')
                gid = add('synthetic_data', 500, output=packets, task=task_name, realization=realization)
                for arm_name in SYNTHETIC_ARMS:
                    scfg = arm_config(arm_name, short_minutes, subject=subject_id, stage='inner0',
                                      packets_root=packets, out_dir=str(root / 'synthetic' / 'runs'))
                    sid = add('synthetic_fit', 510, config=asdict(scfg), task=task_name,
                              realization=realization, output=str(root / 'synthetic' / 'scores'))
                    tasks[-1]['depends'] = [gid]
    plan = dict(version='v0.4.0', scope='E1125 development chain; one patient, five arms, first seed complete',
                source_digest=source_digest()[0], root=str(root), subject=subject,
                short_history_minutes=int(short_minutes), tasks=tasks, arms=human,
                created_epoch=time.time(), workers_per_gpu=1,
                budget=dict(human_paired_inner=len(order), human_final=len(order) * len(FINAL_SEEDS),
                            human_total=len(order) * (1 + len(FINAL_SEEDS)),
                            main_fits=len(order) * 2 + len(order) * len(FINAL_SEEDS),
                            synthetic_fits=len(SYNTHETIC_TASKS) * len(SYNTHETIC_REALIZATIONS) * len(SYNTHETIC_ARMS)),
                gpu_concurrency_reason='one training process per GPU; concurrent same-GPU FP64 Lyapunov solves are not admitted',
                deadline_policy='PAUSED preserves both INNER optimizers; no budget or clock relabelled as convergence')
    atomic_json(plan, path)
    return plan


def run_worker(plan_path, device, deadline, kinds=None, slot=0):
    plan = json.loads(Path(plan_path).read_text())
    root = Path(plan['root'])
    state = root / 'queue_state'
    if plan['source_digest'] != source_digest()[0]:
        raise ValueError('plan/source changed; freeze a new plan explicitly')
    worker = acquire_lock(root / 'resource_locks' / f'{device.replace(":", "_")}_slot{slot}.lock')
    if worker is None:
        raise RuntimeError('another worker owns this device in this queue')
    completed = []
    try:
        while time.time() < deadline:
            if source_digest()[0] != plan['source_digest']:
                raise RuntimeError('source changed during the queue; freeze a new plan before continuing')
            chosen = None
            live = False
            eligible = [t for t in plan['tasks'] if kinds is None or t['kind'] in kinds]
            for task in sorted(eligible, key=lambda t: t['priority']):
                path = state / (task['id'] + '.json')
                existing = json.loads(path.read_text()) if path.exists() else {}
                if existing.get('status') in ('COMPLETE', 'FAILED'):
                    continue
                if any(not (state / (d + '.json')).exists()
                       or json.loads((state / (d + '.json')).read_text()).get('status') != 'COMPLETE'
                       for d in task['depends']):
                    continue
                lock = acquire_lock(state / (task['id'] + '.lock'))
                if lock is None:
                    live = True
                    continue
                if path.exists() and json.loads(path.read_text()).get('status') in ('COMPLETE', 'FAILED'):
                    lock.close()
                    continue
                chosen = (task, path, lock)
                break
            if chosen is None:
                pending = any((not (state / (t['id'] + '.json')).exists()
                               or json.loads((state / (t['id'] + '.json')).read_text()).get('status')
                               not in ('COMPLETE', 'FAILED')) for t in eligible)
                if live or pending:
                    time.sleep(min(10, max(0, deadline - time.time())))
                    continue
                break
            task, path, lock = chosen
            atomic_json(dict(status='RUNNING', task=task['id'], kind=task['kind'], device=device,
                             pid=os.getpid(), started=time.time()), path)
            try:
                result = _dispatch(task, device, deadline, plan, path)
                atomic_json(dict(status=result['status'], task=task['id'], kind=task['kind'], device=device,
                                 finished=time.time(), source_digest=source_digest()[0]), path)
                completed.append(task['id'])
                if result['status'] == 'PAUSED':
                    break
            except Exception as exc:
                import traceback
                atomic_json(dict(status='FAILED', task=task['id'], kind=task['kind'],
                                 error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(),
                                 device=device), path)
                if 'device-side assert' in str(exc):
                    raise
            finally:
                lock.close()
                import gc
                gc.collect()
                if device.startswith('cuda'):
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError:
                        pass
        return dict(status='WINDOW_STOPPED', completed=completed,
                    remaining=[t['id'] for t in plan['tasks']
                               if not (state / (t['id'] + '.json')).exists()
                               or json.loads((state / (t['id'] + '.json')).read_text()).get('status') != 'COMPLETE'])
    finally:
        worker.close()


def _dispatch(task, device, deadline, plan, status_path):
    kind = task['kind']
    if kind == 'paired_inner':
        cfg = RunConfig(**(task['config'] | {'device': device}))
        recipe = run_paired_inner(cfg, deadline=deadline,
                                  progress=lambda row, action: atomic_json(
                                      dict(status='RUNNING', task=task['id'], kind=kind, device=device,
                                           pid=os.getpid(), heartbeat=time.time(), update=row['update'],
                                           mean=row.get('mean')), status_path))
        if recipe['status'] == 'FROZEN':
            atomic_json(recipe, task['output'])
            return dict(status='COMPLETE')
        return recipe
    if kind == 'final':
        cfg = RunConfig(**(task['config'] | {'device': device}))
        card = run_final(cfg, deadline=deadline,
                         progress=lambda row, action: atomic_json(
                             dict(status='RUNNING', task=task['id'], kind=kind, device=device, pid=os.getpid(),
                                  heartbeat=time.time(), update=row['update']), status_path))
        return card
    if kind == 'consumers':
        from .consumers import run_consumers
        return run_consumers(task['selected'], task['output'], device, quick=task.get('quick', False))
    if kind == 'synthetic_data':
        from .synthetic import generate
        return generate(task['output'], task['task'], task['realization'])
    if kind == 'synthetic_fit':
        from .synthetic import fit_and_score
        cfg = RunConfig(**(task['config'] | {'device': device}))
        return fit_and_score(cfg, task['task'], task['realization'], task['output'])
    raise ValueError(kind)
