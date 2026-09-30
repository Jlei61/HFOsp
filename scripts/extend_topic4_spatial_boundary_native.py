#!/usr/bin/env python3
"""Four predeclared native controls around the history-dependent 8.8-s Z field."""
from run_topic4_spatial_boundary_native import run
from topic4_spatial_boundary_common import OUT, read, write
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import os
import time


def main():
    while True:
        p=OUT/'native_batch_status.json';status=read(p) if p.exists() else {}
        if status.get('status')=='COMPLETE':break
        if status.get('status')=='FAILED':raise RuntimeError(status)
        os.kill(read(OUT/'native_batch_process.json')['pid'],0);time.sleep(10)
    jobs=read(OUT/'native_adaptive_protocol.json')['jobs'];rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        iterator=iter(jobs);pending={pool.submit(run,next(iterator)) for _ in range(2)}
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:rows.append(future.result())
            write(OUT/'native_adaptive_status.json',{'status':'RUNNING','completed':len(rows),'total':len(jobs),'rows':rows})
            for _ in done:
                job=next(iterator,None)
                if job is not None:pending.add(pool.submit(run,job))
    write(OUT/'native_adaptive_status.json',{'status':'COMPLETE','completed':len(rows),'total':len(jobs),'rows':rows})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'native_adaptive_status.json',{'status':'FAILED','error':repr(exc)});raise
