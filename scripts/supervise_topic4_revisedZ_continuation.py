#!/usr/bin/env python3
"""One bounded same-state continuation, then actual-data review artifacts."""
import os,subprocess,sys,time
import psutil
import run_topic4_revisedZ_continuation as run
from supervise_topic4_reviewed_global_batch import resources

def main():
    out=run.OUT;p=run.prepare();write=run.carrier.base.write
    assert (out/'dispatch_authorization.json').exists()
    while True:
        total,gpu=resources()
        if total<28 and gpu<24 and psutil.virtual_memory().available/2**30>=120:break
        if time.time()>=p['deadline_epoch']-5400:
            write(out/'status.json',dict(status='NOT_DISPATCHED',reason='Resource/time limit'))
            return
        time.sleep(15)
    with (out/'worker.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,'-u',str(run.ROOT/'scripts/run_topic4_revisedZ_continuation.py'),
            'worker','--producer-script',str(run.ROOT/'scripts/run_topic4_autonomous_recovery.py')],
            cwd=run.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            write(out/'status.json',dict(status='RUNNING',pid=os.getpid(),updated_at=time.time(),
                total=1,completed=0,running={run.NAME:child.pid},pending=[],failed=[]))
            time.sleep(15)
    if child.returncode:
        write(out/'status.json',dict(status='FAILED_REVIEW',returncode=child.returncode,
            total=1,completed=0,running={},pending=[],failed=[run.NAME]))
        return
    commands=[[sys.executable,str(run.ROOT/'scripts'/script),'--root',str(out)] for script in
              ['analyze_topic4_autonomous_recovery.py','analyze_topic4_autonomous_events.py',
               'analyze_topic4_event_extent.py']]
    commands.append([sys.executable,str(run.ROOT/'scripts/analyze_topic4_autonomous_recovery.py'),
                    '--root',str(out),'--name',run.NAME])
    with (out/'analysis.log').open('ab') as log:
        for cmd in commands:
            completed=subprocess.run(cmd,cwd=run.ROOT,stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode:
                write(out/'analysis_failure.json',dict(command=cmd,returncode=completed.returncode))
                break
    write(out/'status.json',dict(status='REVIEW_MILESTONE',pid=os.getpid(),updated_at=time.time(),
        total=1,completed=1,running={},pending=[],failed=[]))

if __name__=='__main__':main()
