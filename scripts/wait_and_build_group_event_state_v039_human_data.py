#!/usr/bin/env python3
"""Start each patient's data build as soon as that patient's blocks finish."""
import argparse,json,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--subject',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
args=p.parse_args();previous=None
while True:
    state=json.loads((args.root/'measurement_batch/queue_status.json').read_text())
    jobs={key:value for key,value in state['jobs'].items() if key.startswith(args.subject+'_')}
    if not jobs:raise ValueError('Subject has no registered measurement jobs')
    if any(v['status']=='FAILED' for v in jobs.values()):raise RuntimeError('Subject measurement failed; do not build a partial data set')
    done=sum(v['status']=='COMPLETE' for v in jobs.values())
    if done!=previous:print(json.dumps(dict(subject=args.subject,measurement_complete=done,expected=len(jobs))),flush=True);previous=done
    if done==len(jobs):break
    time.sleep(10)
subprocess.run([sys.executable,str(Path(__file__).with_name('build_group_event_state_v039_human_data.py')),
                '--subject',args.subject,'--root',str(args.root),'--output',str(args.output)],check=True)
