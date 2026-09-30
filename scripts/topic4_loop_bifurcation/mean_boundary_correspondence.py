#!/usr/bin/env python3
"""One native/model upper-exit comparison, conditional on history correspondence."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import shutil
import time
import numpy as np
from campaign import ROOT, NATIVE, read, write, sha
import dynamic_mean_history_pair as model
import observe_source_aggregation as native
from run_topic4_recovery_window import assert_same_state

OUT = ROOT/'mean_boundary_correspondence'
MODEL = OUT/'model'
NATIVE_OUT = OUT/'native'
INITIAL = model.CASES['high'][0]
NAME = 'high_history_K9p5_constant_background'
K = 9.5


def native_configure():
    native.OUT = NATIVE_OUT;native.PARENT = INITIAL.parent;native.INITIAL = INITIAL;native.NAME = NAME
    native.configure()
    fields = dict(np.load(OUT/'held_fields.npz'))
    native.native.fields = lambda z, k: (fields['Z'].copy(), fields['K'].copy())


def prepare():
    assert read(ROOT/'dynamic_mean_history_pair/analysis/result.json')['both_histories_retained']
    assert read(ROOT/'mean_single_replica_identity_v2/result.json')['retained']
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert np.array_equal(np.load(model.base.PARAMETERS/'parameters.npz')['nu_per_ms'],
                          np.load(model.base.MATCHED/'fixed_external_per_ms.npy'))
    with np.load(INITIAL.parent/'held_fields.npz') as f:
        z = f['Z'];k = f['K']*(K/9.35)
    assert abs(k.mean()-K) < 1e-12
    np.savez_compressed(OUT/'held_fields.npz', Z=z, K=k)
    c = dict(status='REGISTERED_ONE_EXIT_UPPER_CONDITION', created_epoch=time.time(),
        question='At a nearby largerK, do native and leading-mean dynamics both lose the sustained high state and permit coreZ recovery?',
        motivation='Both10s high/asymmetric histories atK9.35 correspond, and R=1 nativeidentity passes. A second parameter tests disappearance of activity, rather than merely retaining an initializedhighstate. Prior native variable-background K9.5 evidence motivates this fixed-background test but does not replace its matched control.',
        design='One native10s and one64-replica10s at Kmean9.5 from the same complete72s highhistory used at9.35. Only the heldK field scales9.5/9.35; heldZ, fullinitialV/ref/synapse/M/G/pendingpulses, graph, thresholds and fixedpercellnu are unchanged. BothG/M are free. No timedquietwindow or neuralforcing.',
        pairing='Native future countstream paired with completedK9.35 highreference: sameinitialrng and exactnu. Model external/mainrng likewise paired with its10sK9.35 highreference. Native andR64 model share countlaw, not individual realizedcounts.',
        guards='Tail5-10s weightedspatialRMS<=10Hz; eachcoremeanrate error<=10Hz; coreZdrift error<=.01/s and same signs; both satisfy quiet allE/A/B mean<5Hz. First100ms-continuous causalR<=5 entry times must differ<=.25s. Report all guards even if failed. These are finiteconditional correspondence tests, not fold certification.',
        stop='Exactly the native/model pair at9.5 for10s. Analyze after both finish; no automatic K scan or extension. If discrepancy, inspect nativecount sensitivity/replica approximation before formalbranch work.',
        source=str(INITIAL), source_sha256=sha(INITIAL), fields_sha256=sha(OUT/'held_fields.npz'), producer_sha256=sha(__file__),
        formal_bifurcation_allowed=False)
    write(OUT/'contract.json', c);shutil.copy2(__file__, OUT/'producer.py')
    p = copy.deepcopy(read(NATIVE/'protocol.json'));p.update(stage='NATIVE_MEAN_MODEL_UPPER_EXIT', created_epoch=time.time(), deadline_epoch=time.time()+86400)
    write(NATIVE_OUT/'protocol.json', p);shutil.copy2(NATIVE/'geometry.npz', NATIVE_OUT/'geometry.npz')
    native_configure();native.native.make_job(NAME, str(INITIAL), 10., True, .21, K, False)
    jobpath = NATIVE_OUT/'jobs'/f'{NAME}.json';job = read(jobpath)
    job.update(held_fields_file=str(OUT/'held_fields.npz'), held_fields_sha256=sha(OUT/'held_fields.npz'),
        external_expected_rate_override=dict(path=str(model.base.MATCHED/'fixed_external_per_ms.npy'), applies='Every step before originalPoisson countdraw'),
        probe_reason='Matched upperexit correspondence, same72s completehistory as10sK9.35 reference')
    write(jobpath, job);path = NATIVE_OUT/'runs'/NAME/'checkpoint.pkl';saved = native.native.read_pickle(path);saved['job'] = job
    native.native.base.save_pickle(path, saved)
    expected = copy.deepcopy(native.native.read_pickle(INITIAL)['engine']);expected['termination_mechanism']['sahp_g'][:] = k
    assert_same_state(saved['engine'], expected)
    write(MODEL/'contract.json', dict(**c, references={'upper': dict(initial_sha256=sha(INITIAL))},
        dependencies={p: sha(p) for p in [__file__, model.__file__, model.leading.__file__, model.base.__file__, model.leading.previous.__file__]}))


def run_native(device):
    assert read(OUT/'contract.json')['producer_sha256'] == sha(__file__)
    assert not (NATIVE_OUT/'progress.json').exists();native_configure()
    nu = np.load(model.base.MATCHED/'fixed_external_per_ms.npy');backend = native.gpu.cuda_backend.wrap_simulator;calls = 0
    def wrap(original, device_index):
        fast = backend(original, device_index=device_index)
        def simulate(params, net, *args, **kw):
            old = kw.get('input_observer')
            def inputs(tm, actual, xi):
                nonlocal calls
                actual[:] = nu;calls += 1
                if old is not None:old(tm, actual, xi)
            kw['input_observer'] = inputs
            return fast(params, net, *args, **kw)
        return simulate
    native.gpu.cuda_backend.wrap_simulator = wrap
    write(NATIVE_OUT/'progress.json', dict(status='RUNNING', pid=os.getpid(), updated_epoch=time.time()))
    try:native.gpu.worker(NATIVE_OUT, NAME, device)
    finally:native.gpu.cuda_backend.wrap_simulator = backend
    assert calls == 100000
    folder = NATIVE_OUT/'runs'/NAME;runtime = read(folder/'runtime_backend.json')
    runtime.update(physics_and_job_unchanged=False, neuronal_equations_unchanged=True,
        registered_external_expected_rate_override=read(NATIVE_OUT/'jobs'/f'{NAME}.json')['external_expected_rate_override'])
    write(folder/'runtime_backend.json', runtime)
    for filename in ['result.json', 'progress.json']:
        a = read(folder/filename);a['runtime_backend'] = runtime;write(folder/filename, a)
    end = native.native.read_pickle(folder/'checkpoint.pkl')['engine']
    reference = native.native.read_pickle(model.base.MATCHED/'runs/high_history_constant_background/checkpoint.pkl')['engine']
    assert end['step'] == 820000 and end['rng_state'] == reference['rng_state']
    assert np.array_equal(end['slow']['z'][:32000], np.load(OUT/'held_fields.npz')['Z'])
    assert np.array_equal(end['termination_mechanism']['sahp_g'], np.load(OUT/'held_fields.npz')['K'])
    write(NATIVE_OUT/'result.json', dict(status='COMPLETE', steps=calls, final_external_RNG_paired=True, held_fields_exact=True, formal_bifurcation_allowed=False))
    write(NATIVE_OUT/'progress.json', dict(status='COMPLETE', updated_epoch=time.time()))


def run_model(device):
    assert read(OUT/'contract.json')['producer_sha256'] == sha(__file__)
    parent = model.HistoryNetwork
    class BoundaryNetwork(parent):
        def __init__(self, case, replicas, device):
            super().__init__(case, replicas, device)
            k = np.load(OUT/'held_fields.npz')['K']
            self.native_initial[:32000, 7] = k
            self.initial_state = self.cp.asarray(np.repeat(self.native_initial[:, None, :], replicas, axis=1));self.reset()
    model.OUT = MODEL;model.CASES = {'upper': (INITIAL, 720000, NAME)};model.HistoryNetwork = BoundaryNetwork
    model.run('upper', device)
    with np.load(MODEL/'upper/final_state.npz') as end, np.load(ROOT/'dynamic_mean_history_pair/high/final_state.npz') as ref:
        assert np.array_equal(end['rng'], ref['rng']) and np.array_equal(end['external_rng'], ref['external_rng'])
    a = read(MODEL/'upper/result.json');a['both_RNGs_paired_with_K9p35'] = True;write(MODEL/'upper/result.json', a)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('command', choices=['prepare', 'native', 'model']);p.add_argument('--device', type=int, default=0);a = p.parse_args()
    if a.command == 'prepare':prepare()
    elif a.command == 'native':run_native(a.device)
    else:run_model(a.device)
