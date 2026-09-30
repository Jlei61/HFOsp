"""Execution-only handoff of a complete structural-branch checkpoint."""
import hashlib
import json
import pickle
from pathlib import Path
import shutil
import time
import psutil

ROOT = Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924')
OUT = ROOT / 'axis_controls/conditional_runs'
REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def stop_at_checkpoint(condition, name, process, new_backend):
    """Return an archived checkpoint record, or leave the worker running.

    The caller owns dispatch and must queue or restart a stopped job. Complete
    saved observations are retained. At most the current2s block is recomputed.
    A prepared initial checkpoint is allowed only if no observations exist.
    """
    try:
        proc = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return None
    command = proc.cmdline()
    assert condition in command and name in command
    assert any(str(REPO / 'scripts' / file) in command for file in
               ['run_topic4_loop_axis_conditional.py', 'run_topic4_loop_axis_cuda_override.py'])
    folder = OUT / condition / 'runs' / name
    cp = folder / 'checkpoint.pkl'
    if not cp.exists() or (folder / 'result.json').exists():
        return None
    protocol = read(OUT / condition / 'protocol.json')
    for path, digest in protocol['source_hashes'].items():
        assert sha(Path(path)) == digest, path
    assert sha(REPO / 'scripts/run_topic4_loop_axis_conditional.py') == protocol['axis_conditional_sha256']
    assert sha(REPO / 'scripts/run_topic4_loop_zk_conditional.py') == protocol['runner_sha256']
    try:
        proc.suspend()
    except psutil.NoSuchProcess:
        return None
    stopped = False
    try:
        if (folder / 'result.json').exists():
            return None
        with cp.open('rb') as handle:
            saved = pickle.load(handle)
        job = read(OUT / condition / 'jobs' / f'{name}.json')
        assert saved['job'] == job and saved['identity'] == protocol['identity']
        step = int(saved['engine']['step'])
        initial_step = round(job['branch_start_s'] * 10000)
        assert step >= initial_step
        required = {'chunks', 'actual_current_chunks', 'conditional_drift_chunks',
                    'feedback_chunks', 'global_response_chunks',
                    'intrinsic_adaptation_chunks', 'mechanism_chunks', 'regional_chunks'}
        # Clamped branches record counterfactual drift instead of autonomous
        # z_budget_chunks. Include any additional stream only when produced.
        directories = required | {p.name for p in folder.iterdir() if p.is_dir() and p.name.endswith('_chunks')}
        committed = {}
        for subdir in sorted(directories):
            files = list((folder / subdir).glob('*.npz'))
            if any('.tmp.' in p.name for p in files):
                return None
            if step == initial_step:
                if files:
                    return None
                committed[subdir] = dict(files=0, end_step=initial_step)
            else:
                if not files or max(int(p.stem.split('_')[-1]) for p in files) != step:
                    return None
                committed[subdir] = dict(files=len(files), end_step=step)
        progress = read(folder / 'progress.json')
        if progress.get('time_s') is not None and round(progress['time_s'] * 10000) < step:
            return None
        archive = ROOT / 'backend_handoffs/axis_cpu_gpu' / condition / name / str(time.time_ns())
        archive.mkdir(parents=True)
        backup = archive / 'checkpoint.pkl'
        shutil.copy2(cp, backup)
        digest = sha(cp)
        assert sha(backup) == digest
        shutil.copy2(folder / 'progress.json', archive / 'progress_before.json')
        if (folder / 'runtime_backend.json').exists():
            shutil.copy2(folder / 'runtime_backend.json', archive / 'runtime_backend_before.json')
        record = dict(status='VERIFIED_BEFORE_STOP', condition=condition, name=name,
                      old_pid=proc.pid, old_command=command, step=step, time_s=step * .0001,
                      backup=str(backup), checkpoint_sha256=digest, job_sha256=sha(OUT / condition / 'jobs' / f'{name}.json'),
                      record_path=str(archive / 'handoff.json'), new_backend=new_backend,
                      committed_observation_endpoints=committed, complete_engine_unmodified=True,
                      maximum_uncommitted_time_to_recompute_s=job['checkpoint_s'],
                      persisted_observations_discarded=False, source_files_verified=len(protocol['source_hashes']),
                      prepared_initial_checkpoint=step == initial_step)
        write(archive / 'handoff.json', record)
        proc.terminate()
        proc.resume()
        proc.wait(timeout=10)
        stopped = True
        assert sha(cp) == digest
        record['status'] = 'STOPPED_AT_VERIFIED_CHECKPOINT_PENDING_RESUME'
        write(archive / 'handoff.json', record)
        return record
    finally:
        if not stopped and proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
            proc.resume()
