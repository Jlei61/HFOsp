"""Bounded low-overhead monitor of already running samplers; never launches/resumes them."""
import sys, os, json, time, argparse, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.patient_state_v1.common import ROOT, RUN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minutes', type=float, default=20.)
    args = parser.parse_args()
    targets = [(3063892, 'ou', 'particle_posterior_v1_2/ou'), (3058753, 'ou_history', 'particle_posterior_v1_2/ou_history'), (3050472, 'two_ou_history', 'two_scale_particle_posterior_v1_15_16chains')]
    end = min(time.time()+60*args.minutes, 1789001396)
    next_review = 0.
    last = None
    while time.time() < end:
        rows = []
        for pid, name, folder in targets:
            migration_path = RUN/'ou_gpu_migration/control.json'
            if name == 'ou' and migration_path.exists():
                migration = json.loads(migration_path.read_text())
                if migration['status'] == 'RESUMED': pid = migration['new_pid']
            extension_path = RUN/'two_scale_terminal_extension/control.json'
            if name == 'two_ou_history' and extension_path.exists():
                extension = json.loads(extension_path.read_text())
                if extension['status'] == 'RESUMED': pid = extension['new_pid']
            proc = Path(f'/proc/{pid}/cmdline')
            try: command = proc.read_bytes().replace(b'\0', b' ').decode(); live = ('patient_state_v1/particle_mcmc.py' in command or 'patient_state_v1/two_scale_posterior.py' in command)
            except FileNotFoundError: live = False
            metadata = json.loads((RUN/folder/'checkpoint.json').read_text())
            rows.append(dict(pid=pid, model=name, process_live=live, iteration=metadata['iteration'], saved_status=metadata['status']))
        state = [(r['model'], r['process_live'], r['iteration']//100, r['saved_status']) for r in rows]
        if state != last:
            record = dict(created_unix=time.time(), local_time=time.strftime('%H:%M:%S'), samplers=rows)
            with (RUN/'sampler_terminal_monitor.jsonl').open('a') as handle: handle.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True); last = state
        if time.time() >= next_review:
            subprocess.run([sys.executable, str(ROOT/'scripts/patient_state_v1/review_posteriors.py')], cwd=ROOT, check=True)
            next_review = time.time()+300
        if not any(r['process_live'] for r in rows): break
        time.sleep(30)
    print(json.dumps(dict(monitor='COMPLETE', samplers_not_restarted=True, finished_unix=time.time())), flush=True)

if __name__ == '__main__':
    main()
