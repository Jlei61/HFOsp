#!/usr/bin/env python3
"""Full-trainer per-card preflight: is a second worker per GPU actually faster?

Spec section 9 allows two processes on a card only when the measured aggregate
throughput gain is at least 20 percent, peak total memory stays under 18 GiB
and at least 6 GiB remain free.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.topic5_group_event_state.v035.contracts import atomic_json

DATA = '/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired/human_data_v2'


def launch(work, tag, gpu, family, width, rank, hidden, updates, physical):
    out = Path(work) / tag
    if out.exists():
        shutil.rmtree(out)
    argv = [sys.executable, str(ROOT / 'scripts/train_group_event_state_v0310_human.py'),
            '--data', f'{DATA}/epilepsiae_253.pt', '--output-dir', str(out), '--family', family,
            '--recipe-id', 'PROBE', '--state-width', str(width), '--transition-rank', str(rank),
            '--readout-hidden', str(hidden), '--history-hours', '8.0', '--device', 'cuda:0',
            '--physical-batch', str(physical), '--event-budget', str(updates),
            '--foundation-budget', '50', '--constant-budget', '50']
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    log = (Path(work) / f'{tag}.log').open('w')
    return subprocess.Popen(argv, env=env, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True), log


def peak_memory(gpu, processes, interval=2.):
    peak = 0
    while any(p.poll() is None for p in processes):
        try:
            used = int(subprocess.check_output(
                ['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                text=True).strip())
            peak = max(peak, used)
        except Exception:
            pass
        time.sleep(interval)
    return peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--updates', type=int, default=150)
    p.add_argument('--family', default='N')
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--physical-batch', type=int, default=128)
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, count in (('single', 1), ('double', 2)):
        started = time.time()
        launched = [launch(args.work, f'{label}_{i}', args.gpu, args.family, args.width, args.rank,
                           args.hidden, args.updates, args.physical_batch) for i in range(count)]
        processes = [p for p, _ in launched]
        peak = peak_memory(args.gpu, processes)
        codes = [p.wait() for p in processes]
        for _, log in launched:
            log.close()
        elapsed = time.time() - started
        results[label] = dict(workers=count, wall_seconds=elapsed, returncodes=codes,
                              peak_gpu_memory_mib=peak,
                              aggregate_updates=count * (args.updates + 100),
                              updates_per_second=count * (args.updates + 100) / elapsed,
                              all_ok=all(c == 0 for c in codes))
    gain = results['double']['updates_per_second'] / results['single']['updates_per_second'] - 1
    free_gib = (24576 - results['double']['peak_gpu_memory_mib']) / 1024
    accept = (gain >= 0.20 and results['double']['all_ok']
              and results['double']['peak_gpu_memory_mib'] / 1024 < 18 and free_gib >= 6)
    payload = dict(schema='v0310_throughput_probe_v1', timestamp=time.time(), probe=vars(args) | {},
                   results=results, aggregate_throughput_gain=gain,
                   peak_total_memory_gib=results['double']['peak_gpu_memory_mib'] / 1024,
                   free_after_peak_gib=free_gib, workers_per_gpu=2 if accept else 1,
                   rule='two workers per card only when the measured aggregate gain is >=20 percent, '
                        'peak total memory <18 GiB and >=6 GiB stay free (spec section 9)')
    payload['probe'] = {k: str(v) for k, v in vars(args).items()}
    atomic_json(args.output, payload)
    print(json.dumps({k: payload[k] for k in ('aggregate_throughput_gain', 'peak_total_memory_gib',
                                              'free_after_peak_gib', 'workers_per_gpu')}), flush=True)


if __name__ == '__main__':
    main()
