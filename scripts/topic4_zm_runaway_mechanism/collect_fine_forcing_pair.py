"""Collect only the registered live pair; never launch or restart simulations."""
from pathlib import Path
from datetime import datetime
import argparse, json, os, subprocess, sys, time
import psutil

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'
DEST=OUT/'conditioned_refractory_fine_forcing'


def read(p):return json.loads(p.read_text())
def save(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)


def main(pid):
    worker=psutil.Process(pid);born=worker.create_time()
    assert 'refractory_fine_forcing_pair.py' in ' '.join(worker.cmdline())
    assert 'run' in worker.cmdline()
    assert not (DEST/'collector.json').exists()
    status=dict(status='WATCHING_VERIFIED_LIVE_WORKER',collector_pid=os.getpid(),worker_pid=pid,
        worker_create_time=born,created_local=datetime.now().astimezone().isoformat(),
        launches_allowed='Completed-output audits and figures only. No simulation, restart, fit or modelpromotion.',
        completed_actions=[])
    save(DEST/'collector.json',status);start=time.monotonic();seen=[]
    def call(name,cmd):
        with (DEST/(name+'.log')).open('w') as f:
            result=subprocess.run([sys.executable,*map(str,cmd)],cwd=HERE.parents[1],stdout=f,stderr=subprocess.STDOUT)
        assert result.returncode==0,(name,result.returncode)
        status['completed_actions'].append(name);save(DEST/'collector.json',status)
        print('COLLECTED',name,flush=True)
    try:
        while time.monotonic()-start<10800:
            jobs=read(DEST/'jobs.json');completed=jobs['completed']
            if jobs['status']=='COMPLETE':
                assert len(completed)==2
                call('audit_full',[HERE/'audit_fine_forcing_pair.py','audit'])
                call('plot_full',[HERE/'plot_refractory_spatial_diagnostic.py','--fine-forcing'])
                status.update(status='COMPLETE_AUDIT_AND_FIGURE_VISUAL_REVIEW_PENDING',updated_local=datetime.now().astimezone().isoformat())
                save(DEST/'collector.json',status);return
            if completed!=seen:
                call('audit_partial',[HERE/'audit_fine_forcing_pair.py','audit','--partial'])
                if 'recorded_drive_expected' in completed:
                    call('plot_expected',[HERE/'plot_refractory_spatial_diagnostic.py','--fine-forcing','--fine-expected'])
                seen=completed.copy()
            try:
                p=psutil.Process(pid);live=p.create_time()==born and p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:live=False
            if not live:
                # Re-read terminal file to avoid racing the final output write.
                jobs=read(DEST/'jobs.json')
                if jobs['status']=='COMPLETE':continue
                status.update(status='WORKER_EXITED_WITH_INCOMPLETE_OUTPUT',observed_jobs=jobs,updated_local=datetime.now().astimezone().isoformat())
                save(DEST/'collector.json',status);return
            time.sleep(20)
        status.update(status='OBSERVER_TIMEOUT_WORKER_NOT_STOPPED',updated_local=datetime.now().astimezone().isoformat())
        save(DEST/'collector.json',status)
    except Exception as error:
        status.update(status='COLLECTION_FAILED_NO_RESTART',error=repr(error),updated_local=datetime.now().astimezone().isoformat())
        save(DEST/'collector.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker-pid',type=int,required=True);main(p.parse_args().worker_pid)
