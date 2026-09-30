#!/usr/bin/env python3
"""One four-second R=256 resolution check at the completed R=64 exit point."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import shutil
import time
import numpy as np
from campaign import ROOT, read, write, sha
import dynamic_mean_history_pair as history
from mean_boundary_correspondence import OUT as BOUNDARY

OUT = ROOT/'mean_boundary_resolution'


def prepare():
    assert read(BOUNDARY/'model/upper/result.json')['both_RNGs_paired_with_K9p35']
    arrays = []
    for p in sorted((BOUNDARY/'model/upper/chunks').glob('*.npz'))[:40]:
        with np.load(p) as z:arrays.append(z['global_R_Hz'])
    r = np.concatenate(arrays);assert len(r) == 4000 and (r[3000:] < 5).all()
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json', dict(status='REGISTERED_ONE_NUMERICAL_RESOLUTION_CHECK', created_epoch=time.time(),
        question='Does increasing the numerical replica count from64to256 materially shift the K9.5 high-state exit or its spatial collapse?',
        design='Exactly one4s R256 trajectory from the identical complete72s highstate, sameK9.5 heldfield/heldZ, originalgraph/delays/thresholds andfixedPoissonexternal. G/M and recurrentmean remain free. Compare alreadycompletedR64 first4s. No newnativebiologicalparameter or seed.',
        motivation='R64 falls below5Hz by2.031s and is quiet throughout3-4s. This bounded numerical-resolution test covers the transient and onequietsecond. R64 goodmean agreement atK9.35 does not establish numerical convergence of the nearby exit.',
        rng='Same original percell/replica stream key (cell<<32)+replica, seeds unchanged. Externalnu/countgeneration independent ofnetworkstate. More numericalcopies are not more native biologicalsamples; no claim of savedendpointRNG equality between differentdurations.',
        guards='First100ms-continuousR<=5 onset shift<=.1s; tail3-4s quiet allE/A/B<5Hz; tailweightedfieldRMS<=5Hz; corecounterfactualZdrift difference<=.005/s. Additionally report unaligned100ms spatialerror acrosswhole4s. Passing is resolution evidence at thispoint, not R-infinity proof or stability.',
        stop='One4s only, no automatic replica/order/time/parameter expansion.',
        producer_sha256=sha(__file__), history_sha256=sha(history.__file__), fields_sha256=sha(BOUNDARY/'held_fields.npz'),
        formal_bifurcation_allowed=False))
    shutil.copy2(__file__, OUT/'producer.py')


def run(device):
    c = read(OUT/'contract.json');assert c['producer_sha256'] == sha(__file__) and c['history_sha256'] == sha(history.__file__)
    assert c['fields_sha256'] == sha(BOUNDARY/'held_fields.npz');assert not (OUT/'progress.json').exists()
    started = time.time();write(OUT/'progress.json', dict(status='INITIALIZING', pid=os.getpid(), updated_epoch=time.time()))
    e = history.HistoryNetwork('high', 256, device)
    with np.load(BOUNDARY/'held_fields.npz') as f:
        assert np.array_equal(e.native_initial[:32000, 6], f['Z']);e.native_initial[:32000, 7] = f['K']
    e.initial_state = e.cp.asarray(np.repeat(e.native_initial[:, None, :], e.R, axis=1));e.reset()
    write(OUT/'implementation_qa.json', history.leading.qa(e));e.graph();chunks = OUT/'chunks';chunks.mkdir()
    groups = [];globals_ = []
    for offset in range(0, 4000, 10):
        groups.append(e.chunk().astype('f4'));globals_.append(e.global_output.get())
        if (offset+10) % 100 == 0:
            value = np.concatenate(groups);glob = np.concatenate(globals_)
            assert np.isfinite(value).all() and np.isfinite(glob).all()
            np.savez_compressed(chunks/f'{offset-90:05d}_{offset+10:05d}.npz', group_output=value, global_R_Hz=glob[:, 0], global_s=glob[:, 1])
            groups.clear();globals_.clear()
            write(OUT/'progress.json', dict(status='RUNNING', pid=os.getpid(), replicas=256, elapsed_simulation_ms=offset+10,
                elapsed_wall_s=time.time()-started, updated_epoch=time.time()))
            if (offset+10) % 500 == 0:print('R256', offset+10, flush=True)
    assert int(e.clock.get()[0]) == 40000
    assert np.array_equal(e.state.get()[:, :, 6:8], e.initial_state.get()[:, :, 6:8])
    np.savez_compressed(OUT/'final_state.npz', state=e.state.get(), ref=e.ref.get(), rng=e.rng.get(), external_rng=e.external_rng.get(),
        source_history=e.source_history.get(), history=e.history.get(), clock=e.clock.get(), global_state=e.global_state.get())
    write(OUT/'result.json', dict(status='COMPLETE_ONE_RESOLUTION_CHECK_ANALYSIS_PENDING', duration_ms=4000, replicas=256,
        held_fields_bitwise=True, elapsed_wall_s=time.time()-started, formal_bifurcation_allowed=False))
    write(OUT/'progress.json', dict(status='COMPLETE', updated_epoch=time.time()));print('R256 COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('command', choices=['prepare', 'run']);p.add_argument('--device', type=int, default=1);a = p.parse_args()
    if a.command == 'prepare':prepare()
    else:
        try:run(a.device)
        except Exception:
            write(OUT/'progress.json', dict(status='FAILED', pid=os.getpid(), updated_epoch=time.time()));raise
