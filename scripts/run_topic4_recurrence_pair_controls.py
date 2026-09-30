#!/usr/bin/env python3
"""Independent-noise, gain-zero and native-Z controls of observed returns."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import copy
import time
from pathlib import Path
import run_topic4_continuous_resource_recovery as physical

carrier = physical.carrier
ROOT, PARENT = physical.ROOT, physical.PARENT
OUT = PARENT / 'paired_recurrence_confirmation_round8'


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        return carrier.base.read(OUT / 'protocol.json')
    parent = PARENT / 'preserved_global_gain_round7'
    protocol = copy.deepcopy(carrier.base.read(parent / 'protocol.json'))
    reference = carrier.base.read(parent / 'jobs/resource_rho0.25_k50_tau10_s9108401.json')
    jobs = []
    cases = [(50., 9108402, .25), (200., 9108402, .25),
             (50., 9108401, 0.), (200., 9108401, 0.),
             (0., 9108401, .25), (0., 9108402, .25)]
    for i, (gain, seed, rho) in enumerate(cases):
        job = copy.deepcopy(reference)
        job.update(name=f'resource_rho{rho:g}_k{gain:g}_tau10_s{seed}', round=8,
                   pool_gain=gain, seed=seed, recovery_ratio=rho, device=i % 2)
        jobs.append(job)
    protocol.update(round=8, created_at=time.time(), initial_jobs=jobs,
        parent_parameter_producer_sha256=protocol['parameter_producer_sha256'],
        parameter_producer_sha256=carrier.base.sha(__file__),
        parameter_extension='Same frozen ResourceRecoverySlow equations: etaM.005/tauM2s, tauZ5s, pooltau10s/r0=50. Complete gains0/50/200 at two noise seeds with rho.25, reusing R7 gains50/200 seed1. Also test rho0 at gains50/200 seed1, which is the already verified exact native-Z equation.',
        parameter_question='Do the observed quiet/reentry and post-high finite-burst regimes survive another noise realization? What changes without global feedback? At the same slow pool kinetics, is the revisedZ term needed, or can nativeZ also return?',
        acceptance='Original high/return markers plus native local/global rates, finite events, field and raster review. Returning once is not recurrence; lower mean rate with persistent cores is not return. Neither revised-Z success nor two noise realizations establishes native-Z sufficiency or robustness across network realizations.',
        dispatch_gate='Prepare only until actual saved R7 return observations are reviewed. No automatic new conditions beyond these six jobs; retain60s horizon and existing wall deadline.')
    carrier.base.write(OUT / 'protocol.json', protocol)
    for job in jobs:
        carrier.base.write(OUT / 'jobs' / (job['name'] + '.json'), job)
    return protocol


def worker(name):
    protocol = prepare()
    assert carrier.base.sha(__file__) == protocol['parameter_producer_sha256']
    previous_out, previous_prepare = physical.OUT, physical.prepare
    physical.OUT, physical.prepare = OUT, lambda: protocol
    try:
        physical.worker(name)
    finally:
        physical.OUT, physical.prepare = previous_out, previous_prepare


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'worker'])
    parser.add_argument('--name')
    parser.add_argument('--producer-script')
    args = parser.parse_args()
    if args.producer_script:
        assert Path(args.producer_script).resolve() == Path(carrier.__file__).resolve()
    if args.mode == 'prepare':
        prepare()
    else:
        try:
            worker(args.name)
        except Exception as exc:
            carrier.base.write(OUT / 'runs' / args.name / 'failure.json', dict(error=repr(exc), time=time.time()))
            raise
