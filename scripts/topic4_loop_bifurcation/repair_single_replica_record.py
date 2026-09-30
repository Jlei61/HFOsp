#!/usr/bin/env python3
"""Persist shared-count observations also on the native normal stop exception."""
import argparse
import os
import shutil
import time
import numpy as np
import verify_mean_single_replica as old
from campaign import ROOT, read, write, sha

FAILED = old.OUT
OUT = ROOT/'mean_single_replica_identity_v2'
old.OUT = OUT


def prepare():
    assert read(FAILED/'native_progress.json')['status'] == 'COMPLETE'
    assert not (FAILED/'native_record.npz').exists()
    assert not (FAILED/'result.json').exists()
    write(FAILED/'recording_failure.json', dict(status='NATIVE_COMPLETE_RECORD_NOT_SAVED_MODEL_NOT_STARTED',
        cause='The native horizon observer raises its normal completion exception. Post-return recording code was skipped; the wrapper incorrectly reported observation completion without checking the file. Native100ms result exists; model aborted at file loading before initialization or a physical step.',
        repair='Parallel v2 folder; same100ms replay, save in finally, require exactcount/spike lengths and record existence, compare final complete native engine with the retained first run. No physics, input or acceptance changes.'))
    old.prepare();c = read(OUT/'contract.json');c.update(recording_repair_sha256=sha(__file__), failed_recording=str(FAILED))
    write(OUT/'contract.json', c);shutil.copy2(__file__, OUT/'recording_repair_producer.py')


def observe(device):
    assert read(OUT/'contract.json')['recording_repair_sha256'] == sha(__file__)
    assert not (OUT/'native_progress.json').exists();old.configure()
    nu = np.load(old.model.base.MATCHED/'fixed_external_per_ms.npy')
    counts = np.empty((old.STEPS, 40000), dtype='u2');spikes = np.empty((old.STEPS, 40000), dtype=bool)
    backend = old.native.gpu.cuda_backend.wrap_simulator
    class RecordedRng:
        def __init__(self, rng):self.original = rng;self.n = 0
        def __getattr__(self, key):return getattr(self.original, key)
        def poisson(self, lam, size=None):
            assert np.array_equal(np.broadcast_to(lam, nu.shape), nu*.1)
            value = self.original.poisson(lam, size=size);assert value.max() < 65536
            counts[self.n] = value;self.n += 1;return value
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
            try:return fast(params, net, *args, **kw)
            finally:
                if proxy.n == ni == n == old.STEPS:
                    np.savez_compressed(OUT/'native_record.npz', counts=counts, spikes=spikes)
                else:write(OUT/'incomplete_record.json', dict(counts=proxy.n, inputs=ni, spikes=n, expected=old.STEPS))
        return simulate
    old.native.gpu.cuda_backend.wrap_simulator = wrap
    write(OUT/'native_progress.json', dict(status='RUNNING', pid=os.getpid(), updated_epoch=time.time()))
    try:old.native.gpu.worker(OUT, old.NAME, device)
    finally:old.native.gpu.cuda_backend.wrap_simulator = backend
    assert (OUT/'native_record.npz').exists() and not (OUT/'incomplete_record.json').exists()
    folder = OUT/'runs'/old.NAME
    assert read(folder/'result.json')['status'] == 'COMPLETE'
    old.assert_same_state(old.native.native.read_pickle(folder/'checkpoint.pkl')['engine'], old.native.native.read_pickle(FAILED/'runs'/old.NAME/'checkpoint.pkl')['engine'])
    runtime = read(folder/'runtime_backend.json');runtime.update(physics_and_job_unchanged=False, neuronal_equations_unchanged=True,
        registered_external_expected_rate_override=read(OUT/'jobs'/f'{old.NAME}.json')['external_expected_rate_override'])
    write(folder/'runtime_backend.json', runtime)
    for filename in ['result.json', 'progress.json']:
        result = read(folder/filename);result['runtime_backend'] = runtime;write(folder/filename, result)
    write(OUT/'native_progress.json', dict(status='COMPLETE', steps=old.STEPS, complete_engine_same_as_first_run=True,
        counts_and_spikes_record_saved=True, updated_epoch=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('command', choices=['prepare', 'observe', 'replay']);p.add_argument('--device', type=int, default=0);a = p.parse_args()
    if a.command == 'prepare':prepare()
    elif a.command == 'observe':observe(a.device)
    else:
        assert read(OUT/'contract.json')['recording_repair_sha256'] == sha(__file__)
        old.replay(a.device)
