#!/usr/bin/env python3
"""Verify full replay state, then dispatch the frozen eight remaining probes."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import copy
import time
import subprocess
from campaign import ROOT,REPO,PYTHON,read,write,sha
import native_campaign as n
import spatial_probes
from run_topic4_recovery_window import assert_same_state


def main():
    qa=ROOT/'exit_state_continuation_qa';run=qa/'runs/source16p70_to20'
    while not (run/'result.json').exists():
        p=read(run/'progress.json') if (run/'progress.json').exists() else {}
        write(qa/'collector_status.json',dict(status='WAITING_REPLAY',time_s=p.get('time_s'),updated_epoch=time.time(),pid=os.getpid()))
        time.sleep(20)
    assert read(run/'result.json')['status']=='COMPLETE'
    source=n.native.SOURCE/'runs'/n.native.NAME/'states/t20s.pkl'
    a=n.native.read_pickle(run/'checkpoint.pkl')['engine'];b=n.native.read_pickle(source)['engine']
    classes=[a['slow']['kind'],b['slow']['kind']]
    assert classes==['ConditionalSlow','GlobalResponseSlow'],classes
    # The diagnostic class delegates the original update whenever clamp=False.
    assert not read(run/'result.json')['job']['conditional_clamp']
    a['slow']['kind']=b['slow']['kind']
    assert_same_state(a,b)
    write(qa/'gate.json',dict(status='PASS',whole_physical_engine_bitwise=True,source=str(source),
        replay=str(run/'checkpoint.pkl'),checkpoint_sha256=sha(run/'checkpoint.pkl'),
        verified_epoch=time.time(),only_metadata_normalization=dict(path='slow.kind',values=classes,
        reason='ConditionalSlow with clampFalse delegates original GlobalResponseSlow physical update; no numerical state omitted. Tracker is an observer outside engine.'),
        scope='Reconstructed16.7s natural state continued without intervention reaches the original20s full engine exactly. Together with168bitwise observation arrays over10to16.7s, validates the snapshot lineage. Not a new scientific replicate.'))
    write(qa/'collector_status.json',dict(status='FULL_STATE_PASS',updated_epoch=time.time()))
    spec=read(ROOT/'exit_return_probe_spec.json');root=ROOT/'exit_return_probes'
    if not (root/'protocol.json').exists():spatial_probes.prepare(root,spec)
    with (ROOT/'logs/exit_probe_supervisor.log').open('a') as handle:
        proc=subprocess.Popen([PYTHON,str(REPO/'scripts/topic4_loop_bifurcation/supervise_probes_v2.py'),'--root',str(root),'--max-workers','6'],cwd=REPO,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'exit_probe_launch.json',dict(pid=proc.pid,launch_epoch=time.time(),root=str(root),gate=str(qa/'gate.json')))


if __name__=='__main__':main()
