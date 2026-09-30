#!/usr/bin/env python3
"""Aggregate execution status (counts, active processes, budgets) into status.json and jobs.json."""
import psutil
import numpy as np
from common import *  # noqa: F401,F403
import native_continue as nc
from approx_system import APPROX


def native_summary():
    runs = {}
    for jp in sorted((nc.NATIVE / 'jobs').glob('*.json')):
        job = read(jp); f = nc.NATIVE / 'runs' / job['name']
        st = 'COMPLETE' if (f / 'result.json').exists() else ('FAILED' if (f / 'failure.json').exists() else ('RUNNING' if (f / 'progress.json').exists() else 'PLANNED'))
        runs[job['name']] = dict(kind=job['kind'], freeze_m=job['freeze_m'], status=st, duration_ms=job['duration_ms'])
    return runs


def approx_summary():
    out = {}
    if not APPROX.exists():
        return out
    for vd in sorted(APPROX.glob('v*')):
        if not (vd / 'variant.json').exists():
            continue
        runs = {}
        for jp in sorted((vd / 'jobs').glob('*.json')):
            job = read(jp); f = vd / 'runs' / job['name']
            runs[job['name']] = 'COMPLETE' if (f / 'result.json').exists() else ('FAILED' if (f / 'failure.json').exists() else ('RUNNING' if (f / 'progress.json').exists() else 'PLANNED'))
        out[vd.name] = dict(variant=read(vd / 'variant.json'), runs=runs, completed=sum(v == 'COMPLETE' for v in runs.values()), total=len(runs))
    return out


def active_processes():
    found = []
    for proc in psutil.process_iter(['pid', 'cmdline', 'create_time']):
        cmd = ' '.join(proc.info['cmdline'] or [])
        if 'topic4_fig5_z_state' in cmd and 'deliver.py' not in cmd:
            found.append(dict(pid=proc.info['pid'], cmd=cmd[-160:]))
    return found


def main(final=False, execution_complete=None, notes=None):
    nat = native_summary(); app = approx_summary()
    counts = dict(native_replay_complete=sum((OUT / 'replay' / 'runs' / f'eta0.0005_s{s}' / 'result.json').exists() for s in SEEDS),
                  native_continuations={k: sum(1 for v in nat.values() if v['kind'] == 'continuation' and not v['freeze_m'] and v['status'] == k) for k in ('COMPLETE', 'RUNNING', 'FAILED', 'PLANNED')},
                  native_extensions={k: sum(1 for v in nat.values() if v['kind'] == 'extension' and v['status'] == k) for k in ('COMPLETE', 'RUNNING', 'FAILED', 'PLANNED')},
                  native_fixed_zm={k: sum(1 for v in nat.values() if v['kind'] == 'continuation' and v['freeze_m'] and v['status'] == k) for k in ('COMPLETE', 'RUNNING', 'FAILED', 'PLANNED')},
                  small_set_runs=len(list((APPROX / 'small_set').glob('*_s*_*.npz'))) // 2 if (APPROX / 'small_set').exists() else 0,
                  approx={k: dict(completed=v['completed'], total=v['total']) for k, v in app.items()})
    prev = read(OUT / 'status.json') if (OUT / 'status.json').exists() else {}
    prev.update(updated_at=time.time(), updated_human=time.strftime('%Y-%m-%d %H:%M:%S'), counts=counts, native_runs=nat, approx_versions={k: v['runs'] for k, v in app.items()},
                active_processes=active_processes(), independent_sample_meaning='one complete native continuation trajectory on one fixed network; replays, extensions and approximation units add no independent samples')
    if final:
        prev.update(status='EXECUTION_COMPLETE' if execution_complete else 'EXECUTION_INCOMPLETE', execution_complete=bool(execution_complete),
                    scientific_acceptance='PENDING_ORIGINAL_DESIGN_TASK_REVIEW', human_figure_review='PENDING', notes=notes or [])
    write(OUT / 'status.json', prev)
    return prev


def write_job_table():
    """Final job table: planned vs actual per stage (counts + per-job status)."""
    nat = native_summary(); app = approx_summary()
    jobs = read(OUT / 'jobs.json') if (OUT / 'jobs.json').exists() else []
    table = []
    for seed in SEEDS:
        table.append(dict(stage='M0_replay', name=f'replay_s{seed}', status='COMPLETE' if (OUT / 'replay' / 'runs' / f'eta0.0005_s{seed}' / 'result.json').exists() else 'MISSING',
                          duration_ms=REPLAY_END_MS, counts_as_independent_sample=False, qa=read(OUT / 'replay_qa.json')['status'] if (OUT / 'replay_qa.json').exists() else None))
    for name, v in nat.items():
        stage = 'M1_extension' if v['kind'] == 'extension' else ('M1_fixed_Z_M' if v['freeze_m'] else 'M1_frozen_Z_dynamic_M')
        table.append(dict(stage=stage, name=name, status=v['status'], duration_ms=v['duration_ms'], counts_as_independent_sample=(stage == 'M1_frozen_Z_dynamic_M')))
    for ver, v in app.items():
        for name, st in v['runs'].items():
            table.append(dict(stage='M2_approximation', version=ver, name=name, status=st, counts_as_independent_sample=False))
    small = len(list((APPROX / 'small_set').glob('*_s*_*.npz'))) // 2 if (APPROX / 'small_set').exists() else 0
    table.append(dict(stage='M2_small_set', name='lif_small_set', status='COMPLETE' if small >= 60 else 'PARTIAL', runs=small, cells=2048, duration_ms=2000))
    table.append(dict(stage='M3_bifurcation', name='branch_and_spectra', status='NOT_RUN_DEPENDENCY', branch_points=0, spectrum_workpoints=0))
    summary = dict(planned=dict(replay=2, frozen_Z_dynamic_M=24, extension_max=8, fixed_ZM_max=8, small_set_max=60, approx_units_per_version_max=48, versions_max=2, branch_points_max=400, spectrum_workpoints_max=6),
                   actual=dict(replay=2, frozen_Z_dynamic_M=sum(1 for t in table if t['stage'] == 'M1_frozen_Z_dynamic_M' and t['status'] == 'COMPLETE'),
                               extension=sum(1 for t in table if t['stage'] == 'M1_extension' and t['status'] == 'COMPLETE'),
                               fixed_ZM=sum(1 for t in table if t['stage'] == 'M1_fixed_Z_M' and t['status'] == 'COMPLETE'),
                               small_set=small, approx={ver: v['completed'] for ver, v in app.items()}, versions_used=1, branch_points=0, spectrum_workpoints=0),
                   failed=[t['name'] for t in table if t['status'] == 'FAILED'],
                   skipped=dict(fixed_ZM='not authorised by the section 5.3 rule (no consistent two sides after extension)', second_approx_version='not consumed (structural operating-point failure; see scientific_report 3.3)',
                                deterministic_OU_off='dependency (M2 failed)', bifurcation='dependency (M2 failed)'),
                   extended=[t['name'] for t in table if t['stage'] == 'M1_extension'])
    write(OUT / 'jobs.json', dict(summary=summary, jobs=table, written=time.strftime('%Y-%m-%d %H:%M:%S')))
    return summary


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--final', action='store_true'); ap.add_argument('--complete', action='store_true'); a = ap.parse_args()
    s = main(a.final, a.complete)
    print(json.dumps(write_job_table(), indent=1)[:1500])
