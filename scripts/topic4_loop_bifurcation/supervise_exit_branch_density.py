#!/usr/bin/env python3
"""Dispatch only the four registered correspondence runs, one per GPU."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import fcntl
import subprocess
import time
import psutil
from campaign import ROOT, REPO, PYTHON, read, write, sha
from exit_branch_density import OUT, NAMES


def main():
    lock = (OUT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(), 'Inspect previous terminal or live controller; no automatic resume/restart'
    contract = read(OUT/'contract.json')
    script = REPO/'scripts/topic4_loop_bifurcation/exit_branch_density.py'
    assert sha(script) == contract['producer_sha256']
    assert read(OUT/'forcing_qa.json')['status'] == read(OUT/'implementation_check.json')['status'] == 'PASS'
    pending = list(NAMES); active = {}; done = []; failed = []; launches = []
    dispatch = (ROOT/'native_dispatch.lock').open('a')
    while pending or active:
        for name, (proc, device) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            del active[name]
            path = OUT/name/'result.json'
            if code == 0 and path.exists() and read(path)['status'] == 'COMPLETE':
                done.append(name)
            else:
                failed.append(dict(name=name, pid=proc.pid, exit_code=code))
        fcntl.flock(dispatch, fcntl.LOCK_EX)
        try:
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
            free = {int(x.split(',')[0]):float(x.split(',')[1])/1024 for x in raw.strip().splitlines()}
            host = psutil.virtual_memory().available/2**30
            for device in sorted([0, 1], key=lambda d: free[d], reverse=True):
                if not pending or failed or any(d == device for _, d in active.values()) or free[device] < 10 or host < 90:
                    continue
                name = pending.pop(0)
                assert not (OUT/name).exists()
                cmd = [PYTHON, str(script), 'worker', '--name', name, '--device', str(device)]
                with (OUT/f'{name}.log').open('x') as handle:
                    proc = subprocess.Popen(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
                active[name] = (proc, device)
                launches.append(dict(name=name, pid=proc.pid, device=device, created_epoch=psutil.Process(proc.pid).create_time(), command=cmd))
                host -= 12
        finally:
            fcntl.flock(dispatch, fcntl.LOCK_UN)
        detail = []
        for name, (proc, device) in active.items():
            path = OUT/name/'progress.json'
            progress = read(path) if path.exists() else {}
            detail.append(dict(name=name, pid=proc.pid, device=device, elapsed_simulation_s=progress.get('elapsed_simulation_s'), stage=progress.get('status')))
        status = 'FAILED' if failed else 'COMPLETE' if len(done) == 4 else 'RUNNING' if active else 'WAITING_RESOURCES'
        write(OUT/'status.json', dict(status=status, supervisor_pid=os.getpid(), completed=done, pending=pending,
            active=detail, failed=failed, total=4, launches=launches, updated_epoch=time.time()))
        if failed and not active:
            break
        if pending or active:
            time.sleep(20)


if __name__ == '__main__':
    main()
