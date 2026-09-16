#!/usr/bin/env python3
"""One second-noise confirmation of the observed native-Z global-pool return."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import copy
import time
from pathlib import Path
import run_topic4_continuous_resource_recovery as physical

carrier=physical.carrier
ROOT,PARENT=physical.ROOT,physical.PARENT
OUT=PARENT/'native_global_confirmation_round9'


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return carrier.base.read(OUT/'protocol.json')
    parent=PARENT/'paired_recurrence_confirmation_round8'
    p=copy.deepcopy(carrier.base.read(parent/'protocol.json'))
    job=carrier.base.read(parent/'jobs/resource_rho0_k200_tau10_s9108401.json')
    job.update(name='resource_rho0_k200_tau10_s9108402',round=9,seed=9108402,device=1)
    p.update(round=9,created_at=time.time(),initial_jobs=[job],
        parent_parameter_producer_sha256=p['parameter_producer_sha256'],
        parameter_producer_sha256=carrier.base.sha(__file__),
        parameter_extension='Only the second fixed noise seed of rho0/kappa200/tauG10; nativeZ, nativeM eta.005/tau2, tauZ5 and all fast parameters unchanged.',
        parameter_question='Does the native-Z return survive the second noise? The source is R8 rho0 rather than a revised-Z candidate. Preserved high-rate criterion alone does not establish restoration of the original interictal repertoire.',
        acceptance='Saved source recurrence must first pass actual temporal/raster/native-field review. No new physical equation; rho0 uses the already tested exact nativeZ branch. Retain60s or second-confirmation+2s stop and global wall deadline.',
        dispatch_gate='Prepared only. Require explicit saved-source scientific review and dispatch_authorization.json for this one condition.')
    carrier.base.write(OUT/'protocol.json',p)
    carrier.base.write(OUT/'jobs'/(job['name']+'.json'),job)
    return p


def worker(name):
    p=prepare();assert carrier.base.sha(__file__)==p['parameter_producer_sha256']
    old_out,old_prepare=physical.OUT,physical.prepare
    physical.OUT,physical.prepare=OUT,lambda:p
    try:physical.worker(name)
    finally:physical.OUT,physical.prepare=old_out,old_prepare


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker'])
    p.add_argument('--name');p.add_argument('--producer-script');args=p.parse_args()
    if args.producer_script:assert Path(args.producer_script).resolve()==Path(carrier.__file__).resolve()
    if args.mode=='prepare':prepare()
    else:
        try:worker(args.name)
        except Exception as exc:
            carrier.base.write(OUT/'runs'/args.name/'failure.json',dict(error=repr(exc),time=time.time()))
            raise
