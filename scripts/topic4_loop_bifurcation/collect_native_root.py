#!/usr/bin/env python3
"""Wait for a declared native root, then use the same complete-only analysis."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time,fcntl
from pathlib import Path
from campaign import read,write
from analyze_native import main


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    lock=(root/'collector.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    names=read(root/'queue.json')['names']
    while True:
        complete=[]
        for n in names:
            result=root/'runs'/n/'result.json'
            if result.exists() and read(result)['status']=='COMPLETE':complete.append(n)
        write(root/'collector_status.json',dict(status='READY' if len(complete)==len(names) else 'WAITING_NATIVE',completed=complete,total=len(names),pid=os.getpid(),updated_epoch=time.time()))
        if len(complete)==len(names):break
        time.sleep(30)
    main(root)
    write(root/'collector_status.json',dict(status='COMPLETE',completed=complete,total=len(names),pid=os.getpid(),updated_epoch=time.time()))
