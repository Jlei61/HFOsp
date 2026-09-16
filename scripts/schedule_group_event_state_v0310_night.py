#!/usr/bin/env python3
"""Overnight scheduler for the frozen v0.3.10 dispatch order.

Admission is group-aware: a paired F/L/N group is only started when the
measured throughput says it can finish before the no-new-long-jobs deadline.
Nothing here reads SELECTION or seizure outcomes; the only feedback into
scheduling is elapsed time and completion.
"""
from __future__ import annotations

import argparse, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v0310 import queue_plan as Q
from src.topic5_group_event_state.v0310 import audit
from src.topic5_group_event_state.v035.contracts import atomic_json

PYTHON = sys.executable
DATA = Path('/data/hfosp_group_event_state_v0_3_9_transition_transfer_background_repaired/human_data_v2')
PRIOR_SECONDS = {('F', 8.0): 420., ('L', 8.0): 1200., ('N', 8.0): 2200.}


def prior(family, hours):
    scale = {0.5: .12, 2.0: .32, 8.0: 1., 16.0: 1.9, 24.0: 2.8}.get(hours, 1.)
    return PRIOR_SECONDS.get((family, 8.0), 1200.) * scale


class Scheduler:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.root)
        self.run_root = self.root / 'human_v0310'
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / 'queue_state.json'
        self.snapshot_path = self.root / 'resource_snapshots.jsonl'
        self.observed = {}
        self.running = {}
        self.finished = []
        self.quarantined = set()
        self.consecutive_failures = {}
        self.launched = set()
        self.admission_stops = []
        self.slots = [(gpu, k) for gpu in args.gpus for k in range(args.workers_per_gpu)]
        self.phase_index = 0
        self.common_recipe = {}
        self.common_family = {}
        self.started = time.time()
        self.last_snapshot = 0.

    # ------------------------------------------------------------------ plan
    def cells_for_phase(self, phase):
        cells = Q.phase_cells(phase, self.common_recipe, self.common_family)
        for c in cells:
            c['output_dir'] = str(self.run_root / phase / c['id'])
            c['data'] = str(DATA / f"{c['subject']}.pt")
        return cells

    def refresh_common_recipe(self):
        capacity = audit.load_cards(self.run_root / 'U1')
        extra = audit.load_cards(self.run_root / 'U2')
        report = {}
        for subject in Q.SUBJECTS:
            rows = [c for c in capacity if c['subject'] == subject]
            picked = audit.common_recipe_by_inner(rows)
            report[subject] = picked
            if picked['recipe']:
                self.common_recipe[subject] = picked['recipe']
                scored = [(c['stages']['event']['selected_inner'], c['family'])
                          for c in rows + [x for x in extra if x['subject'] == subject]
                          if c['recipe'] == picked['recipe'] and c.get('stages', {}).get('event')]
                if scored:
                    self.common_family[subject] = min(scored)[1]
        return report

    def estimate(self, cell):
        key = (cell['family'], cell['history_hours'])
        seen = self.observed.get(key)
        return max(seen) * 1.25 if seen else prior(*key)

    # ------------------------------------------------------------- dispatch
    def launch(self, cell, slot):
        gpu, _ = slot
        out = Path(cell['output_dir']); out.mkdir(parents=True, exist_ok=True)
        argv = [PYTHON, str(ROOT / 'scripts/train_group_event_state_v0310_human.py'),
                '--data', cell['data'], '--output-dir', str(out), '--family', cell['family'],
                '--recipe-id', cell['recipe'], '--state-width', str(cell['state_width']),
                '--transition-rank', str(cell['transition_rank']),
                '--readout-hidden', str(cell['readout_hidden']),
                '--history-hours', str(cell['history_hours']), '--source-mode', cell['source_mode'],
                '--view', cell['view'], '--seed', str(cell['seed']), '--device', 'cuda:0',
                '--physical-batch', str(self.args.physical_batch),
                '--deadline-epoch', str(self.args.checkpoint_stop)]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        log = (out / 'run.log').open('a')
        process = subprocess.Popen(argv, env=env, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        self.launched.add(cell['id'])
        self.running[slot] = dict(cell=cell, process=process, log=log, started=time.time())
        print(json.dumps(dict(event='STARTED', id=cell['id'], gpu=gpu, pid=process.pid)), flush=True)

    def reap(self):
        for slot in list(self.running):
            info = self.running[slot]
            code = info['process'].poll()
            if code is None:
                continue
            info['log'].close()
            elapsed = time.time() - info['started']
            card_path = Path(info['cell']['output_dir']) / 'card.json'
            status = 'FAILED'
            if card_path.exists():
                try:
                    status = json.loads(card_path.read_text()).get('status', 'FAILED')
                except json.JSONDecodeError:
                    status = 'FAILED'
            record = dict(id=info['cell']['id'], phase=info['cell']['phase'], group=info['cell']['group'],
                          status=status, returncode=code, elapsed_seconds=elapsed, gpu=slot[0],
                          output_dir=info['cell']['output_dir'])
            if status in ('COMPLETE', 'WALL_TIME_LIMITED'):
                self.observed.setdefault((info['cell']['family'], info['cell']['history_hours']), []).append(elapsed)
            else:
                record['tail'] = (Path(info['cell']['output_dir']) / 'run.log').read_text()[-1500:]
            self.finished.append(record)
            if status in ('COMPLETE', 'WALL_TIME_LIMITED'):
                self.consecutive_failures[slot[0]] = 0
            else:
                self.consecutive_failures[slot[0]] = self.consecutive_failures.get(slot[0], 0) + 1
                if self.consecutive_failures[slot[0]] >= 2:
                    self.quarantined.add(slot[0])
                    print(json.dumps(dict(event='GPU_QUARANTINED', gpu=slot[0],
                                          consecutive_failures=self.consecutive_failures[slot[0]],
                                          note='isolated after repeated failures; no automatic retry')),
                          flush=True)
            print(json.dumps({k: record[k] for k in ('event', 'id', 'status', 'elapsed_seconds')
                              if k in record} | dict(event='FINISHED', id=record['id'],
                                                     status=status, elapsed_seconds=round(elapsed))), flush=True)
            del self.running[slot]

    def already_done(self, cell):
        """A cell already launched in this session is NOT re-dispatchable.

        Only checking for a finished card would re-queue a cell that is still
        running, because a running cell has not written its card yet. The
        contract also forbids automatic retries, so one launch per id is right.
        """
        return cell['id'] in self.launched or (Path(cell['output_dir']) / 'card.json').exists()

    def save(self, status):
        atomic_json(self.state_path, dict(
            status=status, elapsed_seconds=time.time() - self.started,
            phase=Q.DISPATCH_ORDER[self.phase_index] if self.phase_index < len(Q.DISPATCH_ORDER) else 'DONE',
            common_recipe=self.common_recipe, common_family=self.common_family,
            not_implemented_phases=Q.NOT_IMPLEMENTED_PHASES,
            workers_per_gpu=self.args.workers_per_gpu,
            running=[dict(id=v['cell']['id'], gpu=slot[0], pid=v['process'].pid,
                          seconds=round(time.time() - v['started'])) for slot, v in self.running.items()],
            finished=self.finished, quarantined_gpus=sorted(self.quarantined),
            observed_seconds={f'{k[0]}_H{k[1]}': [round(x) for x in v] for k, v in self.observed.items()},
            admission_stops=self.admission_stops, consecutive_failures=self.consecutive_failures,
            deadlines=dict(no_new_long_jobs=self.args.no_new_long_jobs,
                           checkpoint_stop=self.args.checkpoint_stop)))

    def snapshot(self):
        if time.time() - self.last_snapshot < 300:
            return
        self.last_snapshot = time.time()
        try:
            smi = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'], text=True).strip().split('\n')
        except Exception as error:
            smi = [f'nvidia-smi failed: {error}']
        usage = shutil.disk_usage('/data')
        with self.snapshot_path.open('a') as handle:
            handle.write(json.dumps(dict(t=time.time(), gpus=smi,
                                         data_free_gib=round(usage.free / 2 ** 30, 1),
                                         load=os.getloadavg(),
                                         running=[v['cell']['id'] for v in self.running.values()])) + '\n')

    # ------------------------------------------------------------------- run
    def loop(self):
        pending = []
        while True:
            self.reap(); self.snapshot()
            now = time.time()
            if now >= self.args.checkpoint_stop:
                self.save('CHECKPOINT_STOP'); break
            if not pending and self.phase_index < len(Q.DISPATCH_ORDER):
                recipe_fixed = self.phase_index > 0 and bool(self.common_recipe)
                if not self.running or recipe_fixed:
                    phase = Q.DISPATCH_ORDER[self.phase_index]
                    if phase != 'U1':
                        self.refresh_common_recipe()
                    pending = [c for c in self.cells_for_phase(phase) if not self.already_done(c)]
                    if not pending:
                        self.phase_index += 1
                        self.save('RUNNING')
                        continue
                    print(json.dumps(dict(event='PHASE', phase=phase, cells=len(pending),
                                          common_recipe=self.common_recipe)), flush=True)
            if not pending and not self.running:
                if self.phase_index >= len(Q.DISPATCH_ORDER):
                    self.save('QUEUE_EXHAUSTED'); break
                self.phase_index += 1; continue
            free = [s for s in self.slots if s not in self.running and s[0] not in self.quarantined]
            while free and pending:
                cell = pending[0]
                group = [c for c in pending if c['group'] == cell['group']]
                serial = sum(self.estimate(c) for c in group)
                width = max(1, min(len(self.slots) - len(self.quarantined) * self.args.workers_per_gpu, len(group)))
                if now + serial / width * 1.25 > self.args.no_new_long_jobs and not self.args.ignore_admission:
                    print(json.dumps(dict(event='ADMISSION_STOP', group=cell['group'],
                                          phase=cell['phase'],
                                          estimated_seconds=round(serial / width * 1.25),
                                          remaining_seconds=round(self.args.no_new_long_jobs - now))), flush=True)
                    self.admission_stops.append(dict(group=cell['group'], phase=cell['phase'],
                                                     estimated_seconds=round(serial / width * 1.25),
                                                     remaining_seconds=round(self.args.no_new_long_jobs - now),
                                                     unrun_cells=[c['id'] for c in pending]))
                    pending = []
                    self.phase_index += 1
                    break
                self.launch(pending.pop(0), free.pop(0))
            self.save('RUNNING')
            time.sleep(self.args.poll_seconds)
        # graceful stop: the trainer saves latest.pt at every evaluation
        for slot, info in list(self.running.items()):
            try:
                os.killpg(info['process'].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                info['process'].wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(info['process'].pid, signal.SIGKILL); info['process'].wait()
            info['log'].close()
            self.finished.append(dict(id=info['cell']['id'], phase=info['cell']['phase'],
                                      group=info['cell']['group'], status='WALL_TIME_LIMITED',
                                      elapsed_seconds=time.time() - info['started'], gpu=slot[0],
                                      output_dir=info['cell']['output_dir']))
        self.running.clear()
        self.save('STOPPED')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    p.add_argument('--workers-per-gpu', type=int, default=1)
    p.add_argument('--physical-batch', type=int, default=128)
    p.add_argument('--poll-seconds', type=float, default=30.)
    p.add_argument('--no-new-long-jobs', type=float, required=True)
    p.add_argument('--checkpoint-stop', type=float, required=True)
    p.add_argument('--ignore-admission', action='store_true')
    args = p.parse_args()
    Scheduler(args).loop()


if __name__ == '__main__':
    main()
