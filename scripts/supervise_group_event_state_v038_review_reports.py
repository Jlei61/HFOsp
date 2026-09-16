#!/usr/bin/env python3
"""Finalize this bounded repair batch after its registered workers finish."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repair-root', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    phases = ('dual_credit_batch', 'dual_credit_remaining_batch', 'h2a_suffix_batch', 'random_budget_batch')
    output = args.repair_root / 'reporting_status.json'
    while True:
        queues = {}
        for phase in phases:
            file = args.repair_root / phase / 'queue_status.json'
            state = json.loads(file.read_text()) if file.exists() else {'status': 'PENDING'}
            queues[phase] = {k: state.get(k) for k in ('status', 'complete', 'not_estimable', 'failed', 'running', 'pending')}
        done = all(row['status'] in ('COMPLETE', 'FAILED') for row in queues.values())
        payload = {'status': 'AGGREGATING' if done else 'WAITING_FOR_REGISTERED_BATCHES',
                   'queues': queues, 'updated_at': time.time(), 'entire_repair_goal_complete': False}
        tmp = output.with_suffix('.tmp'); tmp.write_text(json.dumps(payload, indent=2) + '\n'); os.replace(tmp, output)
        if done:
            break
        time.sleep(20)
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    commands = [
        [sys.executable, str(root / 'scripts/finalize_group_event_state_v038.py'), '--repair-root', str(args.repair_root),
         '--output-dir', str(args.repair_root / 'final_reports')],
        [sys.executable, str(root / 'scripts/paper_figures/plot_group_event_state_v038_core_closure.py'),
         '--summary', str(args.repair_root / 'final_reports/summary_main.json'),
         '--out-dir', str(args.repair_root / 'final_reports/figures'), '--allow-repair-diagnostic'],
    ]
    try:
        for command in commands:
            subprocess.run(command, cwd=root, env=env, check=True)
    except Exception as error:
        payload.update(status='FAILED', error=repr(error), updated_at=time.time())
        output.write_text(json.dumps(payload, indent=2) + '\n')
        raise
    payload.update(status='REGISTERED_BATCH_REPORTS_GENERATED_REVIEW_REQUIRED', updated_at=time.time(),
                   all_batches_successful=all(row['status'] == 'COMPLETE' for row in queues.values()),
                   whole_goal_still_requires='remaining optimization, provenance, H2b assay, and scientific hypothesis qualification')
    output.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
