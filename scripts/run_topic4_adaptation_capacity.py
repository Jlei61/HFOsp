#!/usr/bin/env python3
"""Parameter-only tests of adaptation capacity and matched-capacity timing."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import argparse, copy, time
from pathlib import Path
import run_topic4_autonomous_recovery as carrier

ROOT, PARENT = carrier.ROOT, carrier.OUT
OUT = PARENT / 'adaptation_capacity_round6'


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        return carrier.base.read(OUT / 'protocol.json')
    p = copy.deepcopy(carrier.base.read(PARENT / 'protocol.json'))
    jobs = []
    for i, (eta, tau) in enumerate([(1., 2.), (2., 2.), (.4, 10.)]):
        jobs.append(dict(name=f'mix_g0.25_eta{eta:g}_tau{tau:g}_s9108401', round=6,
            mode='mix', gamma=.25, eta_m=eta, tau_M_s=tau, tau_Z_s=5.,
            threshold=carrier.base.old.THRESHOLD, seed=9108401, horizon_s=60.,
            device=i % 2, checkpoint_s=10.))
    p.update(round=6, parent=str(PARENT), created_at=time.time(), initial_jobs=jobs,
        status='REVIEWED_PARAMETER_ONLY_EXTENSION', extension_producer_sha256=carrier.base.sha(__file__),
        change='Only etaM/tauM; native Z and fixed25% redistribution retained. No added global state, fast threshold, resource recovery term, reset, clamp or event-driven input.',
        rationale='At eta.05/.1,tauM2s the actual high-state adaptation current is only about48/95mV-equivalent, far below order1000 recurrent excitation. Those failed trajectories do not test feedback with capacity sufficient to quench the saturated carrier.',
        capacity_comparison='At a sustained500Hz, eta*tauM*r is1000,2000,2000mV-equivalent. The eta2/tau2 and eta.4/tau10 pair have the same asymptotic current at a given firing rate; they differ in per-spike increment and adaptation/recovery time. This is an algebraic capacity comparison, not a predicted firing solution or guaranteed termination.',
        limits='Large feedback may prevent entry or stabilize a lower persistent rate. Neither is autonomous recovery. Actual finite preentry events, all-E/core quiet return and reentry remain separately assessed. Numerical strength is not a fitted human physiological parameter.',
        qa='Exact frozen carrier and hashes, already verified native/mix zero-effect and checkpoint continuation; only explicit job values and output namespace change.')
    carrier.base.write(OUT / 'protocol.json', p)
    for job in jobs:
        carrier.base.write(OUT / 'jobs' / (job['name'] + '.json'), job)
    return p


def worker(name):
    p = prepare()
    assert carrier.base.sha(__file__) == p['extension_producer_sha256']
    old_out, old_prepare = carrier.OUT, carrier.prepare
    carrier.OUT, carrier.prepare = OUT, lambda: p
    try:
        carrier.worker(name)
    finally:
        carrier.OUT, carrier.prepare = old_out, old_prepare


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
