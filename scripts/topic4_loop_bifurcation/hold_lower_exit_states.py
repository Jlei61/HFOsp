#!/usr/bin/env python3
"""Next bounded held-state pair after both higher-K histories collapsed."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,subprocess,time
from campaign import ROOT,REPO,PYTHON,read,write,sha
import hold_carried_exit_states as runner
import analyze_carried_exit_holds as collector

OUT=ROOT/'carried_exit_lower_holds'
JOBS={'held_K9p2':9.2,'held_K9p35':9.35}


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    higher=[read(runner.OUT/n/'result.json') for n in runner.JOBS]
    assert all(max(r['terminal3s_rates_allE_A_B_surround_Hz'])<5 for r in higher)
    c=read(runner.OUT/'contract.json')
    c.update(status='REGISTERED_AFTER_BOTH_HIGHER_K_HOLDS_COLLAPSED',created_epoch=time.time(),
        question='Which lower K value preserves the carried active state after the transient recruitment shrinkage has time to unfold?',
        selection='K9.5 and9.65 both went quiet during their8s holds, so the prior slow-ramp K9.687 coreexit was a delayed dynamic location. Next exactlyK9.2 and9.35, spanning the first recruitment drop of that trajectory; no inference of root nonexistence from finite quiet histories.',
        design='Same complete K9 control state, same0.15K/s prescribed approach, then8s heldK at9.2or9.35. Actual Zfieldheldmean.21,40000x128densityparticles, unchanged constant expectedexternalmean and paired numericalRNG future. This pair is selected after the previous negative evidence, not an independent boundary confirmation.',
        jobs=JOBS,predecessor_results=[str(runner.OUT/n/'result.json') for n in runner.JOBS],
        wrapper_sha256=sha(__file__),producer_sha256=sha(runner.__file__),
        collector_sha256=sha(collector.__file__),formal_bifurcation_allowed=False)
    c.pop('pre_dispatch_amendment',None)
    write(OUT/'contract.json',c)


def verify():
    c=read(OUT/'contract.json')
    assert c['wrapper_sha256']==sha(__file__) and c['producer_sha256']==sha(runner.__file__)
    assert c['collector_sha256']==sha(collector.__file__)
    return c


def worker(name,device):
    verify();runner.OUT=OUT;runner.JOBS=JOBS;runner.worker(name,device)


def analyze():
    verify();collector.OUT=OUT;collector.JOBS=JOBS;collector.main(True)


def supervise():
    verify();assert not (OUT/'status.json').exists();active={};done=[];failed=[]
    for device,name in enumerate(JOBS):
        with (OUT/f'{name}.log').open('w') as log:
            active[name]=subprocess.Popen([PYTHON,__file__,'worker','--name',name,'--device',str(device)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    while active:
        for name,p in list(active.items()):
            code=p.poll()
            if code is None:continue
            if code==0 and (OUT/name/'result.json').exists():done.append(name)
            else:failed.append(dict(name=name,exit_code=code))
            del active[name]
        write(OUT/'status.json',dict(status='FAILED' if failed else 'RUNNING' if active else 'COMPLETE',
            supervisor_pid=os.getpid(),completed=done,failed=failed,
            active=[dict(name=n,pid=p.pid) for n,p in active.items()],updated_epoch=time.time()))
        if active:time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker','supervise','analyze'])
    p.add_argument('--name',choices=list(JOBS));p.add_argument('--device',type=int,default=0)
    a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='worker':worker(a.name,a.device)
    elif a.command=='analyze':analyze()
    else:supervise()
