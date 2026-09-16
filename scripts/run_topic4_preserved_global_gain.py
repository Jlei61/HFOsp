#!/usr/bin/env python3
"""Complete gain-by-timescale contrasts with the already-tested revised Z."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse,copy,time
from pathlib import Path
import run_topic4_continuous_resource_recovery as physical

carrier=physical.carrier
ROOT,PARENT=physical.ROOT,physical.PARENT
OUT=PARENT/'preserved_global_gain_round7'


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return carrier.base.read(OUT/'protocol.json')
    p=copy.deepcopy(physical.prepare())
    reference=carrier.base.read(physical.OUT/'jobs/resource_rho0.25_k50_s9108401.json')
    jobs=[]
    for i,(gain,tau) in enumerate([(200.,2.),(50.,10.),(200.,10.)]):
        job=copy.deepcopy(reference)
        job.update(name=f'resource_rho0.25_k{gain:g}_tau{tau:g}_s9108401',round=7,
                   pool_gain=gain,pool_tau_s=tau,device=i%2)
        jobs.append(job)
    p.update(round=7,created_at=time.time(),initial_jobs=jobs,
        parameter_producer_sha256=carrier.base.sha(__file__),
        parameter_extension='Same ResourceRecoverySlow equations and rho.25, same native M.005/tau2, tauZ5, r0=50. Complete gain50/200 by tauG2/10 contrast using existing R5 gain50/tau2 as its fourth cell.',
        parameter_question='After preserved Z prevents total inhibitory loss, does increasing global feedback capacity quench persistent core activity, and does slower recruitment retain entry before feedback builds?',
        observed_basis='R5 rho.25/gain50/tau2 actual15–20s coreA/B rates283/382Hz while all-E remains60–80Hz, so preserved Z alone has not terminated cores. The previous native-Z gain comparison cannot settle gain effects at a nonzero Z floor.',
        effect_size='At Rg=80Hz and z=.2, gain50/200 produces300/1200mV-equivalent global current. This illustration fixes Rg and Z only for comparing gain, not for simulating or predicting a self-consistent state.',
        hypothesis_boundary='Revised-Z phenomenology remains an explicit new hypothesis, not native-Z success, Liou Equation8 reproduction, or an ionic derivation. Global input remains Z-scaled and contributes to depletion; no protected pool is introduced.',
        acceptance='Same unchanged high/return markers plus finite-event, local-rate, native-field and independent-noise review. Suppressed entry or a narrowed persistent band are not recovery. No automatic expansion beyond these three parameter jobs.')
    carrier.base.write(OUT/'protocol.json',p)
    for job in jobs:carrier.base.write(OUT/'jobs'/(job['name']+'.json'),job)
    return p


def worker(name):
    p=prepare()
    assert carrier.base.sha(__file__)==p['parameter_producer_sha256']
    old_out,old_prepare=physical.OUT,physical.prepare
    physical.OUT,physical.prepare=OUT,lambda:p
    try:physical.worker(name)
    finally:physical.OUT,physical.prepare=old_out,old_prepare


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','worker'])
    parser.add_argument('--name');parser.add_argument('--producer-script');args=parser.parse_args()
    if args.producer_script:assert Path(args.producer_script).resolve()==Path(carrier.__file__).resolve()
    if args.mode=='prepare':prepare()
    else:
        try:worker(args.name)
        except Exception as exc:
            carrier.base.write(OUT/'runs'/args.name/'failure.json',dict(error=repr(exc),time=time.time()));raise
