#!/usr/bin/env python3
"""Bounded two-dose screen to address loss of brief interictal events."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import copy,fcntl,subprocess,sys,time,shutil
from pathlib import Path
import psutil
import run_topic4_fixed_zm_termination as fixed
import run_topic4_zm_matched_spatial_termination as matched

ROOT=fixed.OUT;OUT=ROOT/'low_fraction_round6'

def main():
    parent=fixed.carrier.base.read(ROOT/'sahp_bracket_round5/protocol.json')
    base=next(j for j in parent['initial_jobs'] if j['gamma']<.2);jobs=[]
    for i,gain in enumerate([.25,.5]):
        j=copy.deepcopy(base);j.update(name=f'sahp{gain:g}_g0.166667_s9108401',sahp_gain=gain,horizon_s=20.,device=i);jobs.append(j)
    p=copy.deepcopy(parent);p.update(initial_jobs=jobs,status='BOUNDED_BASELINE_PRESERVATION_SCREEN',
        wrapper_sha256=fixed.carrier.base.sha(fixed.__file__),matched_producer_sha256=fixed.carrier.base.sha(matched.__file__),
        question='At global fraction1/6, can a weaker source sAHP preserve brief self-limited events while still allowing entry and autonomous return?',
        motivation='Fraction1/2 and Kgain1.25-1.5 produced finite moving high-rate bands and exit, but prolonged pre-entry events. Fraction1/6 with Kgain1 has remained mostly sparse; zeroK enters without returning. Test two intervening doses.',
        stage_policy='Two20s one-noise runs, no new time constant, no forced reset or stimulation. Same shared4h deadline,12total workers,80GiB host reserve.')
    fixed.carrier.base.write(OUT/'protocol.json',p);shutil.copy2(ROOT/'geometry.npz',OUT/'geometry.npz')
    for j in jobs:fixed.carrier.base.write(OUT/'jobs'/f"{j['name']}.json",j)
    matched.OUT=OUT;matched.qa();pending=list(jobs);running={};handles={};lock=(ROOT/'followup_dispatch.lock').open('a')
    while pending or running:
        for name,proc in list(running.items()):
            if proc.poll() is not None:
                handles[name].close();assert proc.returncode==0,(name,proc.returncode);del running[name]
        if pending:
            assert time.time()<p['deadline_epoch']-2100,'Insufficient reserved time to dispatch this screen'
            fcntl.flock(lock,fcntl.LOCK_EX)
            active=0
            for proc in psutil.process_iter(['cmdline']):
                args=proc.info['cmdline'] or []
                if 'worker' in args and any(Path(a).name=='run_topic4_zm_matched_spatial_termination.py' for a in args):active+=1
            if active<12 and not fixed.carrier.base.read(ROOT/'combined_dispatch_status.json')['queued'] and psutil.virtual_memory().available/2**30>80:
                j=pending.pop(0);assert not (OUT/'runs'/j['name']).exists(),'Do not duplicate a started run'
                h=(OUT/(j['name']+'.log')).open('a');env=dict(os.environ,TOPIC4_TERMINATION_OUT=str(OUT))
                proc=subprocess.Popen([sys.executable,'-u',matched.__file__,'worker','--name',j['name']],env=env,stdout=h,stderr=subprocess.STDOUT)
                running[j['name']]=proc;handles[j['name']]=h;print('Started',j['name'],proc.pid,flush=True)
            fcntl.flock(lock,fcntl.LOCK_UN)
        fixed.carrier.base.write(OUT/'dispatch_status.json',dict(time=time.time(),running={k:v.pid for k,v in running.items()},queued=[j['name'] for j in pending]))
        if pending or running:time.sleep(10)
    producer=Path(__file__).with_name('analyze_topic4_fixed_zm_termination.py')
    for j in jobs:subprocess.run([sys.executable,str(producer),'plot','--root',str(OUT),'--name',j['name']],check=True)
    fixed.carrier.base.write(OUT/'complete.json',dict(time=time.time(),n_runs=2,human_review='PENDING'))

if __name__=='__main__':main()
