#!/usr/bin/env python
"""Release claims for failed tasks so a worker retries them with the current source.

A task is retried at most `--max-retries` times; after that it stays FAILED and
is reported as such rather than silently disappearing.
"""
import argparse,json,glob,os,time
from pathlib import Path
ROOT=Path('/data/hfosp_group_event_state_rich_event_identification_v0311')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--max-retries',type=int,default=2)
    ap.add_argument('--deadline-epoch',type=float,required=True)
    a=ap.parse_args()
    tries={}
    log=ROOT/'retry_ledger.json'
    while time.time()<a.deadline_epoch:
        for f in glob.glob(str(ROOT/'claims'/'*.done.json')):
            d=json.load(open(f))
            if d['status']=='COMPLETE':continue
            k=Path(f).name.replace('.done.json','')
            tries[k]=tries.get(k,0)+1
            if tries[k]>a.max_retries:
                print(f'giving up on {k} after {tries[k]-1} retries',flush=True);continue
            os.remove(f)
            c=f.replace('.done.json','.claim.json')
            if os.path.exists(c):os.remove(c)
            print(f'retry {k} (attempt {tries[k]+1}) task={d["task"]["subject"]} '
                  f'{d["task"]["inputs"]} {d["task"]["family"]} {d["task"]["arm"]}',flush=True)
        log.write_text(json.dumps(tries,indent=1))
        time.sleep(180)
    print('watchdog deadline reached',flush=True)
