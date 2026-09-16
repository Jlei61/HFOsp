#!/usr/bin/env python3
"""Entry point for the v0.4.0 first epilepsy-state evidence package."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('NVIDIA_TF32_OVERRIDE', '0')
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch


def deterministic():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['g0', 'plan', 'work', 'report'])
    p.add_argument('--config', default='config/group_event_state_v040_first_package.json')
    p.add_argument('--plan')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--kinds', help='comma-separated task kinds this worker may take')
    p.add_argument('--slot', type=int, default=0)
    p.add_argument('--hours', type=float, default=8.)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--out')
    a = p.parse_args()
    deterministic()
    cfg = json.loads(Path(a.config).read_text())
    root = cfg['results_root']
    if a.command == 'g0':
        from src.topic5_group_event_state.v040.contracts import run_g0
        result = run_g0(cfg, a.device)
        print(json.dumps({k: result[k] for k in ('status', 'short_history_minutes', 'checks_status')}, indent=2))
    elif a.command == 'plan':
        from src.topic5_group_event_state.v040.queue import build_plan
        contract = json.loads((Path(root) / 'contracts' / 'support.json').read_text())
        plan = build_plan(a.plan or str(Path(root) / 'plan.json'), root,
                          contract['short_history']['selected_short_history_minutes'],
                          a.quick, cfg['subject'])
        print(json.dumps(dict(status='PLAN_WRITTEN', tasks=len(plan['tasks']), budget=plan['budget']), indent=2))
    elif a.command == 'work':
        from src.topic5_group_event_state.v040.queue import run_worker
        kinds = a.kinds.split(',') if a.kinds else None
        print(json.dumps(run_worker(a.plan or str(Path(root) / 'plan.json'), a.device,
                                    time.time() + a.hours * 3600, kinds, a.slot), indent=2))
    else:
        from src.topic5_group_event_state.v040.report import summarize
        r = summarize(a.plan or str(Path(root) / 'plan.json'), a.out or str(Path(root) / 'report' / 'summary.json'))
        print(json.dumps({k: r[k] for k in ('status', 'n_complete', 'n_total')}, indent=2))
