"""Read-only completion observer for the five already-running fixed-Z arms."""
from pathlib import Path
from datetime import datetime
import argparse, json, os, subprocess, sys, time
import psutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DEST=ROOT/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918/fine_rate_frozen_Z_fields'


def read(p):return json.loads(p.read_text())
def save(x):
    p=DEST/'collector.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)


def main(pid):
    worker=psutil.Process(pid);born=worker.create_time()
    assert 'fine_rate_frozen_Z_fields.py' in ' '.join(worker.cmdline()) and 'run' in worker.cmdline()
    assert not (DEST/'collector.json').exists()
    status=dict(status='WATCHING_VERIFIED_LIVE_WORKER',collector_pid=os.getpid(),worker_pid=pid,
        worker_create_time=born,created_local=datetime.now().astimezone().isoformat(),
        scope='Completed-output readout and figure only; no simulation, restart, fit, continuation or model promotion.',actions=[])
    save(status);start=time.monotonic();seen=[]
    def call(name,cmd):
        with (DEST/(name+'.log')).open('w') as f:
            result=subprocess.run([sys.executable,*map(str,cmd)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        assert result.returncode==0,(name,result.returncode)
        status['actions'].append(name);save(status);print('COLLECTED',name,flush=True)
    try:
        while time.monotonic()-start<10800:
            jobs=read(DEST/'jobs.json');completed=jobs['completed']
            if jobs['status']=='COMPLETE':
                assert len(completed)==5
                call('audit_full',[HERE/'audit_fine_rate_frozen_Z_fields.py'])
                call('plot_full',[HERE/'plot_fine_rate_frozen_Z_fields.py'])
                status.update(status='COMPLETE_AUDIT_AND_FIGURE_VISUAL_REVIEW_PENDING',updated_local=datetime.now().astimezone().isoformat());save(status);return
            if completed!=seen:
                call(f'audit_prefix{len(completed)}',[HERE/'audit_fine_rate_frozen_Z_fields.py','--partial'])
                seen=completed.copy()
            try:
                p=psutil.Process(pid);live=p.create_time()==born and p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:live=False
            if not live:
                jobs=read(DEST/'jobs.json')
                if jobs['status']=='COMPLETE':continue
                status.update(status='WORKER_EXITED_WITH_INCOMPLETE_OUTPUT',observed_jobs=jobs,updated_local=datetime.now().astimezone().isoformat());save(status);return
            time.sleep(20)
        status.update(status='OBSERVER_TIMEOUT_WORKER_NOT_STOPPED',updated_local=datetime.now().astimezone().isoformat());save(status)
    except Exception as error:
        status.update(status='COLLECTION_FAILED_NO_RESTART',error=repr(error),updated_local=datetime.now().astimezone().isoformat());save(status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker-pid',type=int,required=True);a=p.parse_args();main(a.worker_pid)
