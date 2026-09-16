#!/usr/bin/env python3
"""Wait for H1 units, then run frozen-decoder H2a without human supervision."""

from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
PY=Path('/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python')
RUN=REPO/'scripts/run_group_event_state_v037_h2a.py'
H1=Path('/data/hfosp_group_event_state_v0_3_7/h1_shared_equal_horizon')
OUT=Path('/data/hfosp_group_event_state_v0_3_7/h2a_frozen_decoder_equal_horizon')
SUBJECTS=('epilepsiae_253','epilepsiae_958','epilepsiae_1077','epilepsiae_1125')
SEEDS=(20260903,20260904,20260905,20260906,20260907)

def write(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--h1-root', type=Path, default=H1)
    parser.add_argument('--out-root', type=Path, default=OUT)
    parser.add_argument('--state-family', choices=('event','dual'), default='event')
    args = parser.parse_args()
    h1_root=Path(args.h1_root); out_root=Path(args.out_root)
    supervisor = out_root / 'supervisor'
    logs = supervisor / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    pending = [(subject, seed) for seed in SEEDS for subject in SUBJECTS]
    slots = [(gpu, slot) for gpu in (0, 1) for slot in range(args.workers_per_gpu)]
    running = {}
    failures = []
    complete = 0
    while pending or running:
        for key in list(running):
            process, subject, seed, handle = running[key]
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del running[key]
            if code == 0 and (out_root / subject / f'seed{seed}' / 'card.json').exists():
                complete += 1
            else:
                failures.append({
                    'subject': subject, 'seed': seed,
                    'returncode': code, 'physical_gpu': key[0],
                })
        h1_status_path = h1_root / 'supervisor' / 'queue_status.json'
        h1_complete = False
        if h1_status_path.exists():
            try:
                h1_complete = json.loads(h1_status_path.read_text()).get('status') == 'COMPLETE'
            except json.JSONDecodeError:
                h1_complete = False
        released = (supervisor / 'RELEASED').exists()
        if h1_complete and released:
            for key in slots:
                if key in running:
                    continue
                ready = next(
                    (i for i, (subject, seed) in enumerate(pending)
                     if (h1_root / subject / f'seed{seed}' / 'card.json').exists()),
                    None,
                )
                if ready is None:
                    continue
                subject, seed = pending.pop(ready)
                handle = (logs / f'{subject}__seed{seed}.log').open('a')
                env = dict(os.environ)
                env['CUDA_VISIBLE_DEVICES'] = str(key[0])
                env.setdefault('OMP_NUM_THREADS', '2')
                process = subprocess.Popen(
                    [str(PY), str(RUN), '--subject', subject, '--seed', str(seed),
                     '--device', 'cuda:0', '--out-root', str(out_root),
                     '--h1-root', str(h1_root), '--state-family', args.state_family],
                    cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT,
                )
                running[key] = (process, subject, seed, handle)
        state = (
            'WAITING_FOR_H1' if pending and not h1_complete
            else ('WAITING_FOR_INSTRUMENT_RELEASE' if pending and not released else 'RUNNING')
        )
        write(supervisor / 'queue_status.json', {
            'format': 'group_event_state_v0_3_7_h2a_equal_horizon_queue_v2',
            'status': state, 'total': 20, 'complete': complete,
            'pending': len(pending),
            'running': [
                {'subject': subject, 'seed': seed, 'pid': process.pid,
                 'physical_gpu': gpu, 'slot_on_gpu': slot}
                for (gpu, slot), (process, subject, seed, _handle) in sorted(running.items())
            ],
            'failures': failures, 'development_targets_read': False,
            'seizure_targets_read': False, 'sealed_partition_opened': False,
        })
        if pending or running:
            time.sleep(args.poll_seconds)
    write(supervisor / 'queue_status.json', {
        'format': 'group_event_state_v0_3_7_h2a_equal_horizon_queue_v2',
        'status': 'FAILED' if failures else 'COMPLETE', 'total': 20,
        'complete': complete, 'pending': 0, 'running': [], 'failures': failures,
        'development_targets_read': False, 'seizure_targets_read': False,
        'sealed_partition_opened': False,
    })
    raise SystemExit(1 if failures else 0)

if __name__=='__main__': main()
