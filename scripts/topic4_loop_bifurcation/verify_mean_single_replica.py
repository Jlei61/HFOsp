#!/usr/bin/env python3
"""Native/shared-count replay verifies the R=1 endpoint of the replica model."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import shutil
import time
import numpy as np
from campaign import ROOT, NATIVE, read, write, sha
import observe_source_aggregation as native
import dynamic_mean_history_pair as model
import dynamic_poisson_external_pilot as poisson
import density_spatial as physical
import target_density_exit as target
from run_topic4_recovery_window import assert_same_state

OUT = ROOT/'mean_single_replica_identity'
INITIAL = model.CASES['high'][0]
NAME = 'native_shared_count_100ms'
STEPS = 1000


def configure():
    native.OUT = OUT;native.PARENT = INITIAL.parent;native.INITIAL = INITIAL;native.NAME = NAME
    native.configure()


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json', dict(status='REGISTERED_SHARED_COUNT_IDENTITY_CHECK', created_epoch=time.time(),
        question='At one replica, does free recurrent-mean feedback reduce to the original native spike network with identical external counts?',
        design='One100ms native continuation from complete72s highhistory at heldZ/K and fixedpercell expectednu; record actual externalPoissoncounts and everycellspike. Replay those same externalcounts in the R=1 model with own freely generated recurrent spikes. No neural teacherforcing, parameter fit or spectrum approximation.',
        justification='At R=1 each source probability is the actual0/1 source spike; averaging over replicas becomes identity. The delayed meanoperator should then yield the actual native weighted synaptic increments. This tests local arithmetic, recurrentdelay timing, complete pendinghistory and global/M updates together beyond max35.8ms delay.',
        checks='Every native/model spike and ref exact; final V/sE/IE/sI/II/M/Z/K differences<1e-7 nativeunits, R/G<1e-8. Roundoff need not be bitwise because sparse addition order and exponential evaluation differ. Failure is investigated, not redefined as pass.',
        scope='One conditional100ms implementation equivalence assay. It does not prove largerR converges, native correspondence near a boundary, or a deterministic bifurcation. Native and model engines remain untouched.',
        source=str(INITIAL), source_sha256=sha(INITIAL), producer_sha256=sha(__file__), formal_bifurcation_allowed=False))
    p = copy.deepcopy(read(NATIVE/'protocol.json'));p.update(stage='NATIVE_SINGLE_REPLICA_IDENTITY', created_epoch=time.time(), deadline_epoch=time.time()+86400)
    write(OUT/'protocol.json', p);shutil.copy2(NATIVE/'geometry.npz', OUT/'geometry.npz')
    configure();native.native.make_job(NAME, str(INITIAL), .1, True, .21, 9.35, False)
    jobpath = OUT/'jobs'/f'{NAME}.json';job = read(jobpath)
    job['external_expected_rate_override'] = dict(path=str(model.base.MATCHED/'fixed_external_per_ms.npy'), applies='Fixed every0.1ms before originalPoissondraw')
    write(jobpath, job);path = OUT/'runs'/NAME/'checkpoint.pkl';saved = native.native.read_pickle(path);saved['job'] = job
    native.native.base.save_pickle(path, saved)
    assert_same_state(saved['engine'], native.native.read_pickle(INITIAL)['engine'])
    shutil.copy2(__file__, OUT/'producer.py')


def observe(device):
    c = read(OUT/'contract.json');assert sha(__file__) == c['producer_sha256'] and sha(INITIAL) == c['source_sha256']
    assert not (OUT/'native_progress.json').exists();configure()
    nu = np.load(model.base.MATCHED/'fixed_external_per_ms.npy');counts = np.empty((STEPS, 40000), dtype='u2');spikes = np.empty((STEPS, 40000), dtype=bool)
    backend = native.gpu.cuda_backend.wrap_simulator
    class RecordedRng:
        def __init__(self, rng):self.original = rng;self.n = 0
        def __getattr__(self, key):return getattr(self.original, key)
        def poisson(self, lam, size=None):
            assert np.array_equal(np.broadcast_to(lam, nu.shape), nu*.1)
            out = self.original.poisson(lam, size=size);assert out.max() < 65536
            counts[self.n] = out;self.n += 1;return out
    def wrap(original, device_index):
        fast = backend(original, device_index=device_index)
        def simulate(params, net, *args, **kw):
            proxy = RecordedRng(net['rng']);net = dict(net);net['rng'] = proxy;n = ni = 0
            oldin = kw.get('input_observer');oldsp = kw.get('spike_observer')
            def inputs(tm, actual, xi):
                nonlocal ni
                actual[:] = nu;ni += 1
                if oldin is not None:oldin(tm, actual, xi)
            def sp(tm, actual):
                nonlocal n
                spikes[n] = actual;n += 1
                if oldsp is not None:oldsp(tm, actual)
            kw.update(input_observer=inputs, spike_observer=sp)
            result = fast(params, net, *args, **kw)
            assert proxy.n == ni == n == STEPS
            np.savez_compressed(OUT/'native_record.npz', counts=counts, spikes=spikes)
            return result
        return simulate
    native.gpu.cuda_backend.wrap_simulator = wrap
    write(OUT/'native_progress.json', dict(status='RUNNING', pid=os.getpid(), updated_epoch=time.time()))
    try:native.gpu.worker(OUT, NAME, device)
    finally:native.gpu.cuda_backend.wrap_simulator = backend
    folder = OUT/'runs'/NAME
    runtime = read(folder/'runtime_backend.json');runtime.update(physics_and_job_unchanged=False, neuronal_equations_unchanged=True, registered_external_expected_rate_override=read(OUT/'jobs'/f'{NAME}.json')['external_expected_rate_override'])
    write(folder/'runtime_backend.json', runtime)
    for filename in ['result.json', 'progress.json']:
        result = read(folder/filename);result['runtime_backend'] = runtime;write(folder/filename, result)
    write(OUT/'native_progress.json', dict(status='COMPLETE', steps=STEPS, updated_epoch=time.time()))


def replay(device):
    assert read(OUT/'native_progress.json')['status'] == 'COMPLETE'
    assert not (OUT/'model_progress.json').exists()
    write(OUT/'model_progress.json', dict(status='RUNNING', pid=os.getpid(), updated_epoch=time.time()))
    record = dict(np.load(OUT/'native_record.npz'));e = model.HistoryNetwork('high', 1, device);cp = e.cp
    supplied = cp.asarray(record['counts'])
    code = poisson.CODE.replace('unsigned char* external_memory,double* draws,', 'unsigned char* external_memory,double* draws,const unsigned short* supplied,')
    old = 'double count=curand_poisson(&er,lambda);ex[id]=er;'
    assert old in code
    code = code.replace(old, 'double count=supplied[(long long)tick*N+i];')
    module = cp.RawModule(code=physical.CODE+target.EXTRA+target.CLAMP+code, options=('--fmad=false',), name_expressions=['fixed_particles'])
    kernel = module.get_function('fixed_particles')
    def with_counts(grid, block, args):
        return kernel(grid, block, (*args[:3], e.external_rng, e.draws, supplied, *args[3:]))
    e.k['fixed_particles'] = with_counts
    mismatch = 0;first = None
    for step in range(STEPS):
        e.step();actual = e.spikes.get()[:, 0].astype(bool);n = int(np.count_nonzero(actual != record['spikes'][step]))
        if n and first is None:first = step
        mismatch += n
    saved = native.native.read_pickle(OUT/'runs'/NAME/'checkpoint.pkl')['engine'];assert saved['step'] == 721000
    k = np.r_[saved['termination_mechanism']['sahp_g'], np.zeros(8000)]
    expected = np.stack([saved['V'], saved['s_E'], saved['I_E'], saved['s_I'], saved['I_I'], saved['slow']['m'], saved['slow']['z'], k], axis=1)
    delta = abs(e.state.get()[:, 0]-expected);errors = dict(zip(['V', 'sE', 'IE', 'sI', 'II', 'M', 'Z', 'K'], delta.max(0).tolist()))
    global_error = abs(e.global_state.get()-np.array([saved['termination_mechanism']['r_global'], saved['global_feedback_response']['global_state']]))
    refs = np.array_equal(e.ref.get()[:, 0], saved['ref'])
    gates = dict(spikes_exact=mismatch == 0, refs_exact=refs, state_tolerance=float(delta.max()) < 1e-7, global_tolerance=float(global_error.max()) < 1e-8)
    result = dict(status='COMPLETE_SINGLE_REPLICA_IDENTITY', guards=gates, retained=all(gates.values()), steps=STEPS,
        spike_mismatch_count=mismatch, first_mismatch_step=first, max_state_error=errors, global_error=global_error.tolist(),
        interpretation='Shared external counts only; recurrent spikes and their delayed feedback are free. One-replica equivalence supports an explicit graph-replication interpretation of R>1, without asserting any largeR/native near-boundary correspondence.', formal_bifurcation_allowed=False)
    write(OUT/'result.json', result);write(OUT/'model_progress.json', dict(status='COMPLETE', updated_epoch=time.time()));print(result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('command', choices=['prepare', 'observe', 'replay']);p.add_argument('--device', type=int, default=0);a = p.parse_args()
    if a.command == 'prepare':prepare()
    elif a.command == 'observe':observe(a.device)
    else:replay(a.device)
