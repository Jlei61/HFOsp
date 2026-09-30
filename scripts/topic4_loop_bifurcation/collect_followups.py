#!/usr/bin/env python3
"""Analyze only finished frozen follow-ups; preserve original 30-second prefixes."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import fcntl
import time
import traceback
import numpy as np
from campaign import ROOT,read,write,sha
from analyze_native import analyze


def main():
    lock=(ROOT/'followup_collector.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    roots=[ROOT/n for n in ['native_extensions','entry_spatial_probes','exit_return_probes']]
    cache={r:{} for r in roots};anchors={r:{} for r in roots}
    while True:
        progress=[]
        try:
            for root in roots:
                if not (root/'queue.json').exists():
                    progress.append(dict(root=str(root),status='WAITING_PREPARE'));continue
                names=read(root/'queue.json')['names']
                for name in names:
                    if name in cache[root]:continue
                    path=root/'runs'/name/'result.json'
                    if not path.exists() or read(path)['status']!='COMPLETE':continue
                    if (root/'preserved_prefix.json').exists():
                        prefix=read(root/'preserved_prefix.json');paths=prefix['hashes'][name]
                        for relative,digest in paths.items():
                            assert sha(root/'runs'/name/relative)==digest
                            assert sha(__import__('pathlib').Path(prefix['source'])/'runs'/name/relative)==digest
                        write(root/'prefix_verification'/f'{name}.json',dict(status='PASS',files=len(paths),new_and_original_prefix_hashes_unchanged=True))
                    row,inputs=analyze(root,name)
                    assert row['full_horizon']
                    key=(row['job'].get('external_noise_source','original t50s paired input'),row['observation_horizon_s'])
                    if key in anchors[root]:assert np.array_equal(inputs,anchors[root][key]),name
                    else:anchors[root][key]=inputs
                    cache[root][name]=row
                    write(root/'extended_analysis_summary.json',dict(status='COMPLETE' if len(cache[root])==len(names) else 'PARTIAL',completed=len(cache[root]),total=len(names),rows=[cache[root][n] for n in names if n in cache[root]],common_future_inputs_exact_within_noise_and_horizon=True,distinct_future_input_groups=len(anchors[root]),formal_bifurcation='NOT_ESTABLISHED',human_review='PENDING',producer_sha256=sha(__file__)))
                progress.append(dict(root=str(root),status='COMPLETE' if len(cache[root])==len(names) else 'WAITING_NATIVE',completed=len(cache[root]),total=len(names)))
            done=all(x['status']=='COMPLETE' for x in progress)
            write(ROOT/'followup_collector_status.json',dict(status='COMPLETE' if done else 'WAITING_NATIVE',pid=os.getpid(),updated_epoch=time.time(),roots=progress))
            if done:return
        except Exception:
            write(ROOT/'followup_collector_status.json',dict(status='ANALYSIS_ERROR',pid=os.getpid(),updated_epoch=time.time(),error=traceback.format_exc(),native_jobs_unaffected=True));raise
        time.sleep(30)


if __name__=='__main__':main()
