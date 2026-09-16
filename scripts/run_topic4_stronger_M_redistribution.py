#!/usr/bin/env python3
"""Parameter-only extension of the frozen native/mix carrier.

No new state, coupling, or observation is introduced. The experiment asks whether
the local-inhibition reduction admits entry at M strengths that fail to ignite
the native geometry, and whether that existing M can then terminate activity.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import argparse,copy,time
from pathlib import Path
import run_topic4_autonomous_recovery as carrier

ROOT=carrier.ROOT;PARENT=carrier.OUT;OUT=PARENT/'stronger_M_redistribution_round4'

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return carrier.base.read(OUT/'protocol.json')
    p=copy.deepcopy(carrier.base.read(PARENT/'protocol.json'))
    jobs=[]
    for i,eta in enumerate([.05,.1]):
        jobs.append(dict(name=f'mix_g0.25_eta{eta:g}_s9108401',round=4,mode='mix',gamma=.25,
            eta_m=eta,tau_M_s=2.,tau_Z_s=5.,threshold=carrier.base.old.THRESHOLD,
            seed=9108401,horizon_s=60.,device=i,checkpoint_s=10.))
    p.update(round=4,parent=str(PARENT),status='PREPARED_REVIEWED_PARAMETER_EXTENSION',created_at=time.time(),
        initial_jobs=jobs,extension_producer_sha256=carrier.base.sha(__file__),
        change='Only etaM=.05/.1 on gamma=.25,tauM2s relative to the initial mix controls eta.005/.02. Exact frozen carrier, topology, Z, noise and observations retained.',
        rationale='Native stronger M can prevent entry, while gamma=.25 redistribution weakens local inhibition and earlier entry was observed. Their interaction has not been exhausted by the lower-M screen. Test whether stronger existing adaptation terminates activity before Z is fully depleted; no new global state is added.',
        controls='Compare the initial gamma.25 eta.005/.02 trajectories and historical native M controls. First verify finite preentry events and recruitment origin; startup sustained activity or simple ignition suppression does not establish the target cycle.',
        resource_gate='Dispatch only after round3 joint pending queue clears, with total<28,GPU<24,available RAM>=120GiB. Stop dispatch90min before the same9h deadline.',
        qa='Reuses the exact already-verified carrier and its frozen dependency hashes. This wrapper changes only the output namespace and explicit job parameters; no physics implementation is changed.')
    carrier.base.write(OUT/'protocol.json',p)
    for j in jobs:carrier.base.write(OUT/'jobs'/(j['name']+'.json'),j)
    return p

def worker(name):
    p=prepare();assert carrier.base.sha(__file__)==p['extension_producer_sha256']
    old_out,old_prepare=carrier.OUT,carrier.prepare
    carrier.OUT=OUT;carrier.prepare=lambda:p
    try:carrier.worker(name)
    finally:carrier.OUT=old_out;carrier.prepare=old_prepare

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','worker']);a.add_argument('--name')
    a.add_argument('--producer-script',help='Explicit frozen carrier identity for shared process/resource accounting.')
    v=a.parse_args()
    if v.producer_script:assert Path(v.producer_script).resolve()==Path(carrier.__file__).resolve()
    if v.mode=='prepare':prepare()
    else:
        try:worker(v.name)
        except Exception as e:
            carrier.base.write(OUT/'runs'/v.name/'failure.json',dict(error=repr(e),time=time.time()));raise
