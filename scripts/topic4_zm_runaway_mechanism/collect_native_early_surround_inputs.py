"""Observe the fixed native replay, then perform registered diagnostic readouts."""
from pathlib import Path
from datetime import datetime
import json,os,sys,time,subprocess,argparse,psutil

HERE=Path(__file__).resolve().parent
DEST=HERE.parents[1]/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918/native_early_surround_inputs'


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main(pid,device):
    p=psutil.Process(pid);born=p.create_time();assert 'native_early_surround_inputs.py' in ' '.join(p.cmdline())
    assert not (DEST/'collector.json').exists()
    state=dict(status='WATCHING_LIVE_REPLAY',worker_pid=pid,worker_create_time=born,collector_pid=os.getpid(),
        actions=[],created_local=datetime.now().astimezone().isoformat(),
        scope='One native replay followedby fixedinput analysis,numericalauditandfigure. No replayrestart, responsefit,whole-networklaunch orbranch. Scientificreview remainsrootagent responsibility.')
    write(DEST/'collector.json',state);started=time.monotonic()
    try:
        while time.monotonic()-started<10800:
            progress=json.loads((DEST/'progress.json').read_text())
            if progress['status']=='COMPLETE_REPLAY_AUDIT_PASS':
                for name,cmd in [('analysis',['analyze_native_early_surround_inputs.py','--device',str(device)]),
                    ('response_audit',['audit_native_early_surround_response.py','--device',str(device)]),
                    ('plot',['plot_native_early_surround_response.py'])]:
                    with (DEST/(name+'.log')).open('w') as f:
                        result=subprocess.run([sys.executable,str(HERE/cmd[0]),*cmd[1:]],cwd=HERE.parents[1],stdout=f,stderr=subprocess.STDOUT)
                    assert result.returncode==0,(name,result.returncode)
                    state['actions'].append(name);write(DEST/'collector.json',state)
                state.update(status='COMPLETE_DIAGNOSTIC_AND_FIGURE_REVIEW_PENDING',updated_local=datetime.now().astimezone().isoformat());write(DEST/'collector.json',state);return
            try:p=psutil.Process(pid);live=p.create_time()==born and p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:live=False
            if not live:
                if json.loads((DEST/'progress.json').read_text())['status']=='COMPLETE_REPLAY_AUDIT_PASS':continue
                state['status']='REPLAY_EXITED_BEFORE_ACCEPTED_IDENTITY';write(DEST/'collector.json',state);return
            time.sleep(20)
        state['status']='OBSERVER_TIMEOUT_REPLAY_NOT_STOPPED';write(DEST/'collector.json',state)
    except Exception as error:
        state.update(status='COLLECTION_FAILED_NO_RESTART',error=repr(error));write(DEST/'collector.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker-pid',type=int,required=True);p.add_argument('--device',type=int,default=1);a=p.parse_args();main(a.worker_pid,a.device)
