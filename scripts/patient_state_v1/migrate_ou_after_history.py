"""Move the existing OU continuation after GPU1's history chain finishes.

Only this run's exact PID/model is eligible. Freeze and validate its saved
checkpoint, archive it, terminate that owned process, then resume the same
chain and random stream on GPU1. Never runs two writers to the checkpoint.
"""
import sys, os, json, time, signal, shutil, subprocess, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.patient_state_v1.common import ROOT, RUN, write_json

OUT = RUN/'ou_gpu_migration'

def command(pid):
    try: return Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:-1]
    except FileNotFoundError: return []

def is_sampler(pid, model):
    args = command(pid)
    return any(b'patient_state_v1/particle_mcmc.py' in x for x in args) and b'--model' in args and args[args.index(b'--model')+1] == model.encode()

def main():
    OUT.mkdir(exist_ok=True)
    old_pid, history_pid = 3063892, 3058753
    deadline = 1788968996+8.5*3600
    write_json(OUT/'control.json', dict(status='WAITING_FOR_HISTORY_TERMINAL', pid=os.getpid(), old_ou_pid=old_pid, history_pid=history_pid, deadline_unix=deadline))
    while is_sampler(history_pid, 'ou_history') and time.time() < deadline: time.sleep(30)
    history = json.loads((RUN/'particle_posterior_v1_2/ou_history/checkpoint.json').read_text())
    if time.time() >= deadline or is_sampler(history_pid, 'ou_history') or history['status'] != 'COMPLETE' or not is_sampler(old_pid, 'ou'):
        write_json(OUT/'control.json', dict(status='NO_MIGRATION', reason='Time budget or required exact process/terminal condition not met', created_unix=time.time())); return
    args = command(old_pid)
    assert b'--gpu' in args and args[args.index(b'--gpu')+1] == b'0'
    checkpoint_root = RUN/'particle_posterior_v1_2/ou'
    validated = False
    for attempt in range(3):
        os.kill(old_pid, signal.SIGSTOP)
        try:
            meta = json.loads((checkpoint_root/'checkpoint.json').read_text())
            with np.load(checkpoint_root/'checkpoint.npz') as z:
                assert len(z['samples']) == meta['iteration']+1
                assert len(z['loglikes']) == meta['iteration']+1
                assert len(z['accepted']) == meta['iteration']
                assert np.isfinite(z['samples'][-1]).all() and np.isfinite(z['loglikes'][-1]).all()
                assert meta['iteration'] > meta['warmup']
            for filename in ['checkpoint.npz', 'checkpoint.json', 'contract.json']:
                shutil.copy2(checkpoint_root/filename, OUT/('before_'+filename))
            validated = True; break
        except Exception:
            os.kill(old_pid, signal.SIGCONT)
            if attempt == 2: raise
            time.sleep(2)
    assert validated
    write_json(OUT/'checkpoint_audit.json', dict(status='PASS', iteration=meta['iteration'], warmup=meta['warmup'], old_pid=old_pid, saved_state='Parameters, retained stochastic likelihood, proposal covariance/scale, accepted flags and proposal RNG state preserved', random_stream='Explicit0 on GPU1 retains old GPU0 particle-seed sequence', unsaved_work='At most one25-iteration checkpoint interval can be recomputed; no saved samples are discarded', hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('before_*')}, created_unix=time.time()))
    os.kill(old_pid, signal.SIGTERM); os.kill(old_pid, signal.SIGCONT)
    for _ in range(20):
        if not is_sampler(old_pid, 'ou'): break
        time.sleep(.5)
    assert not is_sampler(old_pid, 'ou'), 'Do not launch a second checkpoint writer while the original is live'
    cmd = [sys.executable, '-u', str(ROOT/'scripts/patient_state_v1/particle_mcmc.py'), '--gpu', '1', '--rng-stream', '0', '--model', 'ou', '--iterations', '6000']
    log = RUN/'logs/ou_pmmh_gpu1_continuation.log'
    with log.open('a') as handle:
        process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(OUT/'control.json', dict(status='RESUMED', old_pid=old_pid, new_pid=process.pid, history_terminal=history['status'], resumed_iteration=meta['iteration'], command=cmd, hard_end_unix=1788968996+8.75*3600, resumed_unix=time.time(), log=str(log)))
    print(json.dumps(dict(status='RESUMED', pid=process.pid, iteration=meta['iteration'])), flush=True)

if __name__ == '__main__':
    main()
