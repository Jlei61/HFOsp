#!/usr/bin/env python3
"""Necessary high-side duration control after the intermediate state returned."""
from run_topic4_spatial_boundary_native import run
from topic4_spatial_boundary_common import OUT,read,write
from concurrent.futures import ProcessPoolExecutor,as_completed
import os
import time


def main():
    jobs=[{'name':f'z9400_history{h}_extend4s','z_profile_ms':9400,'history_ms':h,'duration_ms':4000,
           'resume_endpoint':str(OUT/'native_endpoints'/f'z9400_history{h}.npz')} for h in (8000,9400)]
    write(OUT/'high_duration_protocol.json',{'status':'DEFINED_BEFORE_EXTENSION','jobs':jobs,'workers':2,
        'reason':'The apparent persistent state at Z8.8 returned to self-limited events during the authorized extension. Two-second persistence is therefore insufficient; the claimed high-side Z9.4 condition must receive the same uninterrupted 6-s observation.',
        'amendment':'Add exactly two 4-s continuations to the prior native cap of fourteen runs (total sixteen), with no new Z fields, seeds, biology or parameter search. This checks the required sustained-state claim rather than selecting a better-looking condition.',
        'interpretation':'Return weakens a claim of a sustained high-side attractor; persistent activity in both histories supports a finite-time high regime, not proof of a classical bifurcation.'})
    while True:
        p=OUT/'native_adaptive_status.json';r=read(p) if p.exists() else {}
        if r.get('status')=='COMPLETE':break
        if r.get('status')=='FAILED':raise RuntimeError(r)
        os.kill(read(OUT/'native_adaptive_process.json')['pid'],0);time.sleep(10)
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            rows.append(future.result());write(OUT/'high_duration_status.json',{'status':'RUNNING','completed':len(rows),'rows':rows})
    write(OUT/'high_duration_status.json',{'status':'COMPLETE','rows':rows})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'high_duration_status.json',{'status':'FAILED','error':repr(exc)});raise
