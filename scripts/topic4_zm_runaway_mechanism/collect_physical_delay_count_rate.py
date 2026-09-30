"""Read-only completion observer for the single registered corrected run."""
from pathlib import Path
from datetime import datetime
import json,os,sys,time,subprocess,argparse
import psutil

HERE=Path(__file__).resolve().parent
DEST=HERE.parents[1]/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918/physical_delay_count_rate'


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main(pid):
    p=psutil.Process(pid);born=p.create_time();assert 'physical_delay_count_rate.py' in ' '.join(p.cmdline()) and 'run' in p.cmdline()
    assert not (DEST/'collector.json').exists()
    status=dict(status='WATCHING_LIVE_WORKER',collector_pid=os.getpid(),worker_pid=pid,worker_create_time=born,
        created_local=datetime.now().astimezone().isoformat(),actions=[],scope='Completed-output audit andfigureonly; no simulation,restart,fit,branchor modelpromotion.')
    write(DEST/'collector.json',status);start=time.monotonic()
    try:
        while time.monotonic()-start<10800:
            jobs=json.loads((DEST/'jobs.json').read_text())
            if jobs['status']=='COMPLETE':
                assert jobs['completed']==['recorded_drive_binomial_seed1']
                for name,command in [('audit',['audit_physical_delay_count_rate.py','audit']),('plot',['plot_physical_delay_count_rate.py'])]:
                    with (DEST/(name+'.log')).open('w') as f:
                        result=subprocess.run([sys.executable,str(HERE/command[0]),*command[1:]],cwd=HERE.parents[1],stdout=f,stderr=subprocess.STDOUT)
                    assert result.returncode==0,(name,result.returncode)
                    status['actions'].append(name);write(DEST/'collector.json',status)
                status.update(status='COMPLETE_AUDIT_AND_FIGURE_VISUAL_REVIEW_PENDING',updated_local=datetime.now().astimezone().isoformat())
                write(DEST/'collector.json',status);return
            try:p=psutil.Process(pid);live=p.create_time()==born and p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:live=False
            if not live:
                if json.loads((DEST/'jobs.json').read_text())['status']=='COMPLETE':continue
                status['status']='WORKER_EXITED_WITH_INCOMPLETE_OUTPUT';write(DEST/'collector.json',status);return
            time.sleep(20)
        status['status']='OBSERVER_TIMEOUT_WORKER_NOT_STOPPED';write(DEST/'collector.json',status)
    except Exception as error:
        status.update(status='COLLECTION_FAILED_NO_RESTART',error=repr(error));write(DEST/'collector.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker-pid',type=int,required=True);main(p.parse_args().worker_pid)
