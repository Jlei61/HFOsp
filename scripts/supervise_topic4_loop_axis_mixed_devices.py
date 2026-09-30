#!/usr/bin/env python3
"""Same sixteen branches; six total, four CPU, two GPU0, one GPU1 maximum."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import fcntl
import hashlib
from pathlib import Path
import subprocess
import sys
import time
import psutil
import supervise_topic4_loop_axis_conditional as previous
from supervise_topic4_loop_zk_mixed_devices import alive
import topic4_loop_axis_checkpoint_handoff as checkpoint_handoff

run = previous.run
REPO = Path(__file__).resolve().parents[1]
GPU_RUNNER = REPO / 'scripts/run_topic4_loop_axis_cuda_override.py'
GATE = run.PRIMARY / 'qa/axis_cuda_device0_route/gate.json'
GPU1_GATE = run.PRIMARY / 'qa/axis_cuda_device1_route/gate.json'
MAX_TOTAL = 6
MAX_CPU = 4
MAX_GPU0 = 2
MAX_GPU1 = 1


def gpu_resources(active, device):
    row = subprocess.run(['nvidia-smi', f'--id={device}', '--query-gpu=memory.free,uuid',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if row.returncode:
        return 0., 0., 0.
    amount, uuid = [v.strip() for v in row.stdout.strip().split(',')]
    raw = float(amount) / 1024
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True)
    memory = {}
    if query.returncode == 0:
        for line in query.stdout.strip().splitlines():
            pid, gpu, used = [v.strip() for v in line.split(',')]
            if gpu == uuid and used.replace('.', '').isdigit():
                memory[int(pid)] = float(used) / 1024
    reserved = sum(max(0., 1.25 - memory.get(proc.pid, 0.))
                   for proc, _, backend in active.values()
                   if backend == f'cuda_ordered:{device}' and memory.get(proc.pid, 0.) < 1.)
    return raw, reserved, max(0., raw - reserved)


def verify_backend():
    for device, gate_path in [(0, GATE), (1, GPU1_GATE)]:
        gate = run.native.base.read(gate_path)
        assert gate['status'] == 'PASS' and gate['device'] == device
        for path, key in [(GPU_RUNNER, 'wrapper_sha256'),
                          (REPO / 'src/topic4_cuda_ordered_scatter.py', 'backend_sha256'),
                          (Path(run.__file__), 'producer_sha256')]:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == gate[key], path
    cpu = run.native.base.read(run.PRIMARY / 'qa/locality_native_gate.json')
    assert cpu['status'] == 'PASS'
    assert hashlib.sha256((REPO / 'scripts/topic4_loop_locality_cpu.py').read_bytes()).hexdigest() == cpu['source_sha256']


def launch(condition, name, backend, logs):
    handle = (logs / f'{condition}_{name}.log').open('a')
    if backend == 'locality_cpu':
        job = run.native.base.read(run.OUT / condition / 'jobs' / f'{name}.json')
        run.native.write(run.OUT / condition / 'runs' / name / 'runtime_backend.json',
            dict(actual_backend='locality_cpu', actual_device=None, threads=8,
                 planned_job_backend=job['backend'], planned_job_device=job['device'],
                 execution_override_only=True, physics_and_job_unchanged=True,
                 wrapper_sha256=run.native.base.sha(run.__file__),
                 backend_sha256=run.native.base.sha(REPO / 'scripts/topic4_loop_locality_cpu.py')))
    command = [sys.executable, str(GPU_RUNNER), 'worker', '--condition', condition,
               '--name', name, '--device', backend.rsplit(':', 1)[1]] if backend.startswith('cuda_ordered:') else [
                   sys.executable, str(Path(run.__file__)), 'worker', condition, name]
    return subprocess.Popen(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT), handle, backend


def main():
    lock = (run.OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_backend()
    names = run.native.base.read(run.OUT / 'rotated/queue.json')['names']
    central = [n for n in names if n.startswith('z0.75_k2_')]
    queue = [(c, n) for n in central + [n for n in names if n not in central]
             for c in ['rotated', 'isotropic']]
    assert len(queue) == len(set(queue)) == 16
    old = run.native.base.read(run.OUT / 'status.json')
    active = {}
    for row in old.get('active', []):
        try:
            proc = psutil.Process(row['pid'])
        except psutil.NoSuchProcess:
            continue
        if not alive(proc):
            continue
        c, n = row['condition'], row['name']
        command = proc.cmdline()
        assert (c, n) in queue and n in command and c in command
        assert any('run_topic4_loop_axis_' in part for part in command)
        backend = (f'cuda_ordered:{command[command.index("--device") + 1]}'
                   if str(GPU_RUNNER) in command else 'locality_cpu')
        active[c, n] = proc, None, backend
    complete = [key for key in queue if key not in active and
                (run.OUT / key[0] / 'runs' / key[1] / 'result.json').exists()]
    pending = [key for key in queue if key not in active and key not in complete]
    failed = old.get('failed', [])
    logs = run.OUT / 'logs'
    logs.mkdir(exist_ok=True)
    opened = False
    while pending or active:
        for key, (proc, handle, _) in list(active.items()):
            if alive(proc):
                continue
            if handle:
                handle.close()
            del active[key]
            c, n = key
            if not (run.OUT / c / 'runs' / n / 'result.json').exists():
                failed.append(dict(condition=c, name=n, reason='No durable result'))
                pending = []
            else:
                complete.append(key)
                analysis = subprocess.run([sys.executable,
                    str(REPO / 'scripts/analyze_topic4_loop_axis_conditional.py')],
                    cwd=REPO, capture_output=True, text=True)
                (logs / 'latest_analysis.log').write_text(analysis.stdout + analysis.stderr)
                if analysis.returncode:
                    failed.append(dict(stage='analysis', reason=analysis.stderr[-3000:]))
                    pending = []
        missing = previous.prerequisites() if not opened else []
        if not missing and not opened:
            verify_backend()
            opened = True
            run.native.write(run.OUT / 'dispatch_gate.json', dict(status='PASS', epoch=time.time(),
                original18_finished=True, native_graph_prefixes_saved=True,
                reference_route_exact=True, CUDA_reference_route_exact=True,
                maximum_scientific_jobs=16))
        available = psutil.virtual_memory().available / 2**30
        cpu = psutil.cpu_percent(interval=.2)
        raw, reserved, free = gpu_resources(active, 0)
        raw1, reserved1, free1 = gpu_resources(active, 1)
        # Prioritize an already-running, expensive lowZ/lowK CPU branch when
        # a validated GPU slot becomes available. Saved science is unchanged.
        for (c, n), (proc, handle, backend) in list(active.items()):
            if (backend != 'locality_cpu' or not n.startswith('z0.25_k0.02_')
                    or available < 88 or cpu > 65 or failed):
                continue
            if free >= 9.3 and sum(v[2] == 'cuda_ordered:0' for v in active.values()) < MAX_GPU0:
                target = 'cuda_ordered:0'
            elif free1 >= 9.3 and sum(v[2] == 'cuda_ordered:1' for v in active.values()) < MAX_GPU1:
                target = 'cuda_ordered:1'
            else:
                continue
            record = checkpoint_handoff.stop_at_checkpoint(c, n, proc, target)
            if record is None:
                continue
            if handle:
                handle.close()
            active[c, n] = launch(c, n, target, logs)
            record.update(status='RESUMED_FROM_VERIFIED_CHECKPOINT',
                          new_pid=active[c, n][0].pid, epoch=time.time())
            checkpoint_handoff.write(Path(record['record_path']), record)
            if target == 'cuda_ordered:0':
                free -= 1.25
                reserved += 1.25
            else:
                free1 -= 1.25
                reserved1 += 1.25
            available -= 8
        while opened and pending and len(active) < MAX_TOTAL and available >= 88 and cpu <= 65 and not failed:
            gpu_count = sum(item[2] == 'cuda_ordered:0' for item in active.values())
            cpu_count = sum(item[2] == 'locality_cpu' for item in active.values())
            if free >= 9.3 and gpu_count < MAX_GPU0:
                backend = 'cuda_ordered:0'
            elif cpu_count < MAX_CPU:
                backend = 'locality_cpu'
            else:
                break
            c, n = pending.pop(0)
            active[c, n] = launch(c, n, backend, logs)
            available -= 8
            if backend == 'cuda_ordered:0':
                free -= 1.25
                reserved += 1.25
                cpu += 2
            else:
                cpu += 10
        detail = []
        for (c, n), (proc, _, backend) in active.items():
            path = run.OUT / c / 'runs' / n / 'progress.json'
            p = run.native.base.read(path) if path.exists() else {}
            detail.append(dict(condition=c, name=n, pid=proc.pid, time_s=p.get('time_s'),
                               state=p.get('status'), backend=backend))
        stage = 'FAILED' if failed else 'COMPLETE' if not pending and not active else (
            'WAITING_PREREQUISITES' if missing else 'RUNNING' if active else 'WAITING_RESOURCE')
        run.native.write(run.OUT / 'status.json', dict(stage=stage, supervisor_pid=os.getpid(),
            updated_epoch=time.time(), missing_prerequisites=missing, completed=complete,
            pending=pending, active=detail, failed=failed, new_scientific_total=16,
            max_workers=MAX_TOTAL, max_CPU_workers=MAX_CPU, max_GPU0_workers=MAX_GPU0,
            max_GPU1_workers=MAX_GPU1,
            available_memory_GiB=available, CPU_busy_percent_at_dispatch=cpu,
            GPU0_free_GiB=raw, GPU0_pending_startup_reserved_GiB=reserved,
            GPU0_effective_free_for_dispatch_GiB=free,
            GPU1_free_GiB=raw1, GPU1_pending_startup_reserved_GiB=reserved1,
            GPU1_effective_free_for_dispatch_GiB=free1,
            execution_policy='At most6conditional total,4CPU,2GPU0,1GPU1; GPU1 reserved for lowZ/lowK CPU checkpoint continuation. Existing nativeGPU1 worker is separate and preserved. Each GPU needsfree>=9.3GiB, host>=88GiB/CPU<=65percent before dispatch. Carry startup reservations across polls. Original16jobs and prerequisites unchanged.'))
        if pending or active:
            time.sleep(30)


if __name__ == '__main__':
    main()
