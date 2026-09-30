#!/usr/bin/env python3
"""Two paired first-entry controls at half the current M feedback; no refill."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse, copy
from pathlib import Path
import run_topic4_weaker_M_onset_pilot as base
OUT=base.ROOT/'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914'

def first_only(tr,rate,sec,rescue=True):
    tr['rates'].append(float(rate));tr['rates']=tr['rates'][-200:]
    tr['high_bins']=tr['high_bins']+1 if rate>=200 else 0
    if tr['high_bins']>=20 and not tr['entries']:
        tr['entries'].append(dict(onset_s=sec-.2,confirmation_s=sec))
        tr['phase']='HIGH';tr['last_entry_s']=sec;tr['stop_s']=sec+2.

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return
    p=copy.deepcopy(base.core.read(base.OUT/'protocol.json'))
    jobs=[dict(name=f'eta0.0005_s{s}',eta_m=.0005,tau_M_s=1.,seed=s,eta_index=-1,tau_index=0,
        tau_z_ms=5000.,threshold=base.core.old.THRESHOLD,horizon_s=40.,device=i,
        first_entry_horizon_s=40.,post_first_confirmation_s=2.) for i,s in enumerate(base.SEEDS)]
    p.update(jobs=jobs,total=2,eta_M=[.0005],tau_M_s=[1.],status='DEFINED_BEFORE_NEW_RUNS',
        approval='2026-09-14 user requests weaker adaptation and pre-entry event-frequency test.',
        first_entry_horizon_s=40,maximum_trajectory_s=40,max_workers=2,
        intervention='No reset, refill, or stimulation; stop 2 s after first high confirmation, or at 40 s.',
        stop='Exactly two paired-noise half-M runs; no automatic further parameter search.',
        wrapper=str(Path(__file__).resolve()),wrapper_sha256=base.core.sha(__file__),
        producer_sha256=base.core.sha(base.__file__),
        statistical_unit='One fixed topology, two paired noise realizations; descriptive comparison.',
        question='Does halving eta M shorten finite events, quiet intervals or first-entry latency? Test separately.',
        event_observables='Finite event count per second, duration, onset interval, postevent quiet gap, duty fraction; whole E and each core.',
        event_windows='Early 0.5-3.5 s; late onset-3.5 to onset-0.5 s; exclude ongoing transition from finite events.',
        comparisons='Current eta .001, new .0005, existing zero-feedback .0; exact same seeds and physical network.')
    base.core.write(OUT/'protocol.json',p)
    for j in jobs:base.core.write(OUT/'jobs'/(j['name']+'.json'),j)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name');args=ap.parse_args();prepare()
    if args.name:
        base.OUT=OUT;base.schedule=first_only;base.worker(args.name)
        folder=OUT/'runs'/args.name;r=base.core.read(folder/'result.json')
        r.update(post_second_confirmation_s=None,post_first_confirmation_s=2.,manual_intervention=False)
        assert r['tracker']['restore_s'] is None and r['tracker']['release_s'] is None
        base.core.write(folder/'result.json',r)
