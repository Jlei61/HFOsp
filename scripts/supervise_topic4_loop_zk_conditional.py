#!/usr/bin/env python3
"""Resource-bounded CPU dispatch of the frozen18 native conditional branches."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil

REPO = Path(__file__).resolve().parents[1]
OUT = Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924')
RUNNER = REPO/'scripts/run_topic4_loop_zk_conditional.py'
ANALYZE = REPO/'scripts/analyze_topic4_loop_zk_conditional.py'


def write(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2)+'\n');tmp.replace(path)


def main():
    lock = (OUT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((OUT/'protocol.json').read_text())
    for name in ['resume_high_cpu', 'resume_interictal_cpu', 'clamp_mechanism', 'paired_clamp_input']:
        assert json.loads((OUT/'qa'/f'{name}.json').read_text())['status'] == 'PASS', name
    names = json.loads((OUT/'queue.json').read_text())['names']
    assert len(names) == 18 and len(set(names)) == 18
    # Central pair first, then the other preregistered points. No adaptive additions.
    central = [n for n in names if n.startswith('z0.75_k2_')]
    names = central+[n for n in names if n not in central]
    complete = [n for n in names if (OUT/'runs'/n/'result.json').exists()]
    pending = [n for n in names if n not in complete]
    active, failed = {}, {}
    logs = OUT/'logs'; logs.mkdir(exist_ok=True)
    while pending or active:
        for name, (proc, handle) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            handle.close();del active[name]
            if code == 0 and (OUT/'runs'/name/'result.json').exists():
                complete.append(name)
                analyzed = subprocess.run([sys.executable, str(ANALYZE)], cwd=REPO,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                (logs/'latest_analysis.log').write_text(analyzed.stdout)
                if analyzed.returncode:
                    failed['analysis_after_'+name] = analyzed.stdout[-3000:]
                    pending.clear()  # Preserve running work; stop new dispatch on an integrity failure.
            else:
                failed[name] = dict(exit_code=code, log=str(logs/f'{name}.log'))
        available = psutil.virtual_memory().available/2**30
        while pending and len(active) < protocol['max_workers'] and available >= 80:
            name = pending.pop(0)
            handle = (logs/f'{name}.log').open('a')
            proc = subprocess.Popen([sys.executable, str(RUNNER), 'worker', name],
                                    cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
            active[name] = (proc, handle)
            available -= 8.  # Include not-yet-resident worker setup in resource accounting.
        progress = {}
        for name, (proc, _) in active.items():
            path = OUT/'runs'/name/'progress.json'
            row = json.loads(path.read_text()) if path.exists() else {}
            progress[name] = dict(pid=proc.pid, time_s=row.get('time_s'), state=row.get('status'))
        status = dict(stage='RUNNING' if pending or active else ('FINISHED_WITH_FAILURES' if failed else 'COMPLETE'),
                      supervisor_pid=os.getpid(), updated_epoch=time.time(), completed=complete,
                      pending=pending, active=progress, failed=failed, available_memory_GiB=available,
                      total=18, backend='cpu', new_gpu_processes=0, diagnostic_only=True)
        write(OUT/'status.json', status)
        if pending or active:
            time.sleep(15)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
