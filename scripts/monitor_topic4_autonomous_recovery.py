#!/usr/bin/env python3
"""Generate real-count reviews as the bounded experiment finishes chunks."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import fcntl,time,subprocess,sys
from pathlib import Path
import run_topic4_autonomous_recovery as run

def main():
    out=run.OUT;lock=(out/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    done=set();last=0
    while time.time()<run.DEADLINE+60:
        completed=[p.parent.name for p in (out/'runs').glob('*/result.json') if not p.parent.name.startswith('qa_')]
        new=[n for n in completed if n not in done]
        if new or time.time()-last>=600:
            cmd=[sys.executable,str(run.ROOT/'scripts/analyze_topic4_autonomous_recovery.py')]
            has_data=any(not p.parent.parent.name.startswith('qa_') for p in (out/'runs').glob('*/chunks/*.npz'))
            if has_data:
                with (out/'analysis.log').open('a') as log:
                    rc=subprocess.call(cmd,cwd=run.ROOT,stdout=log,stderr=subprocess.STDOUT)
                if rc:run.base.write(out/'monitor_failure.json',dict(time=time.time(),command=cmd,returncode=rc));return
                for name in new:
                    with (out/'analysis.log').open('a') as log:
                        rc=subprocess.call(cmd+['--name',name],cwd=run.ROOT,stdout=log,stderr=subprocess.STDOUT)
                    if rc:run.base.write(out/'monitor_failure.json',dict(time=time.time(),name=name,returncode=rc));return
                    done.add(name)
            last=time.time()
            run.base.write(out/'monitor_status.json',dict(updated_at=last,figures_completed=sorted(done)))
        time.sleep(20)

if __name__=='__main__':main()
