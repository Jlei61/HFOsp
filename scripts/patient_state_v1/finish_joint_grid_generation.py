"""Wait for this run's bounded grid fit, then run the already specified generation check."""
import sys, os, time, json, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.patient_state_v1.common import ROOT, RUN, write_json

def main():
    result = RUN/'joint_grid_refit_v1_26/result.json'
    destination = RUN/'joint_grid_refit_generation_v1_26'
    destination.mkdir(exist_ok=True)
    control = destination/'orchestration.json'
    deadline = 1788968996 + 8.6*3600
    if (destination/'status.json').exists():
        assert json.loads((destination/'status.json').read_text())['status'] == 'COMPLETE'
        print('ALREADY_COMPLETE', flush=True); return
    write_json(control, dict(status='WAITING_FOR_GRID_TERMINAL', pid=os.getpid(), deadline_unix=deadline))
    while not result.exists():
        if time.time() > deadline:
            write_json(control, dict(status='DEADLINE_WITHOUT_GRID_RESULT', pid=os.getpid(), deadline_unix=deadline)); return
        time.sleep(30)
    terminal = json.loads(result.read_text())
    assert terminal['status'] in ['OPTIMIZER_COMPLETE', 'OPTIMIZER_LIMIT_OR_FAILURE', 'BUDGET_LIMIT']
    command = [sys.executable, '-u', str(ROOT/'scripts/patient_state_v1/joint_grid_generation.py')]
    write_json(control, dict(status='GENERATING', pid=os.getpid(), source_fit_status=terminal['status'], command=command, started_unix=time.time()))
    completed = subprocess.run(command, cwd=ROOT, check=False)
    write_json(control, dict(status='COMPLETE' if completed.returncode == 0 else 'GENERATION_FAILED', returncode=completed.returncode, source_fit_status=terminal['status'], finished_unix=time.time()))

if __name__ == '__main__':
    main()
