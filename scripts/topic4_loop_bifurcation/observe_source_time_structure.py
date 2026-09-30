#!/usr/bin/env python3
"""Replay the unchanged 42--44s native state with read-only temporal records."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, copy, shutil, time
import numpy as np
from scipy import sparse
from campaign import ROOT, NATIVE, read, write, sha
import observe_source_aggregation as base
from run_topic4_recovery_window import assert_same_state

OUT = ROOT / 'native_K9p35_temporal_inputs_v2'
NAME = 'unchanged_K9p35_42to44_temporal_observation'
SOURCE = base.OUT


def configure():
    base.OUT = OUT; base.NAME = NAME
    return base.configure()


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'contract.json').exists()
    assert read(SOURCE / 'observer_audit.json')['status'] == 'PASS'
    write(OUT / 'contract.json', dict(status='REGISTERED_BEFORE_TEMPORAL_OBSERVATION', created_epoch=time.time(),
        question='Are the excessive white-source variances explained by source temporal regularity, or do cross-source dependencies remain necessary? Does remaining local rate error coincide with colored input or time-varying adaptation?',
        design='Replay exactly the completed unchanged42-44s native continuation. Readonly all-source spike indices at0.1ms and raw IE/II/V/M for the39 previously assayed targets. No new parameter, input, seed, or physical model. Compare whole final engine bitwise against the existing44s reference.',
        limits='One conditional2s trajectory selected for diagnosis. Periodic source-autospectrum estimates are finite-window approximations; they omit cross-source spectra. Neither observed spectra nor measured moments are an autonomous closure or new biological samples.',
        source_sha256=sha(base.INITIAL), reference_final_sha256=sha(SOURCE / 'runs' / base.NAME / 'checkpoint.pkl'),
        prior_observer_sha256=sha(SOURCE / 'cell_statistics.npz'), producer_sha256=sha(__file__),
        formal_bifurcation_allowed=False, counts_as_autonomous_loop=False))
    p = copy.deepcopy(read(NATIVE / 'protocol.json'))
    p.update(stage='NATIVE_READONLY_SOURCE_TEMPORAL_OBSERVATION', created_epoch=time.time(), deadline_epoch=time.time()+7200)
    write(OUT / 'protocol.json', p); shutil.copy2(NATIVE / 'geometry.npz', OUT / 'geometry.npz')
    configure(); base.native.make_job(NAME, str(base.INITIAL), 2., True, .21, 9.35, False)
    assert_same_state(base.native.read_pickle(OUT / 'runs' / NAME / 'checkpoint.pkl')['engine'],
                      base.native.read_pickle(base.INITIAL)['engine'])
    write(OUT / 'initial_gate.json', dict(status='PASS', whole_initial_engine_bitwise=True))


def worker(device):
    configure(); c = read(OUT / 'contract.json')
    assert c['producer_sha256'] == sha(__file__) and c['source_sha256'] == sha(base.INITIAL)
    assert not (OUT / 'observer_progress.json').exists()
    cells = np.load(SOURCE / 'local_response_factorial/inputs.npz')['cells']
    backend = base.gpu.cuda_backend.wrap_simulator
    def observing_backend(original, device_index):
        fast = backend(original, device_index=device_index)
        def wrapped(params, net, *args, **kw):
            slow = kw['slow']; assert kw['resume_state']['step'] == 420000
            traces = np.empty((20000, 4, len(cells))); events = []; global_state = np.empty((20000, 2))
            n = 0; oldcur = kw.get('current_observer'); oldsp = kw.get('spike_observer')
            def currents(tm, ie, ii, v):
                nonlocal n
                if oldcur is not None: oldcur(tm, ie, ii, v)
                traces[n] = np.stack([ie[cells], ii[cells], v[cells], slow.m[cells]])
                global_state[n] = [slow.r_global, slow.global_state]; n += 1
            def spikes(tm, sp):
                if oldsp is not None: oldsp(tm, sp)
                events.append(np.flatnonzero(sp).astype('i4'))
            kw.update(current_observer=currents, spike_observer=spikes)
            try:
                return fast(params, net, *args, **kw)
            finally:
                if n == len(events) == 20000:
                    counts = np.array([len(a) for a in events], dtype='i4')
                    pointers = np.r_[0, np.cumsum(counts)]
                    indices = np.concatenate(events)
                    matrix = sparse.csc_matrix((np.ones(len(indices), dtype='u1'), indices, pointers), shape=(40000, 20000))
                    sparse.save_npz(OUT / 'source_spikes_0p1ms.npz', matrix)
                    np.savez_compressed(OUT / 'target_traces.npz', cells=cells, IE_II_V_M=traces,
                                        causal_R_G=global_state, times_s=42+np.arange(20000)*.0001)
                else:
                    write(OUT / 'incomplete_observation.json', dict(currents=n, spike_samples=len(events), expected=20000))
        return wrapped
    base.gpu.cuda_backend.wrap_simulator = observing_backend
    write(OUT / 'observer_progress.json', dict(status='RUNNING', pid=os.getpid(), device=device, updated_epoch=time.time()))
    try: base.gpu.worker(OUT, NAME, device)
    finally: base.gpu.cuda_backend.wrap_simulator = backend
    reference = SOURCE / 'runs' / 'unchanged_K9p35_42to44_source_observation' / 'checkpoint.pkl'
    assert c['reference_final_sha256'] == sha(reference)
    assert_same_state(base.native.read_pickle(reference)['engine'],
                      base.native.read_pickle(OUT / 'runs' / NAME / 'checkpoint.pkl')['engine'])
    sp = sparse.load_npz(OUT / 'source_spikes_0p1ms.npz')
    old = dict(np.load(SOURCE / 'cell_statistics.npz'))
    assert np.array_equal(np.asarray(sp.sum(axis=1)).ravel(), old['per_cell_spike_counts'].sum(0))
    with np.load(OUT / 'target_traces.npz') as z:
        tr = z['IE_II_V_M']
        assert np.array_equal(z['causal_R_G'], old['global_R_and_s'])
        mean = tr.reshape(4, 5000, 4, len(cells)).mean(1)
        expected = old['per_cell_mean_moments'][:, [0, 1, 5, 6]][:, :, cells]
        assert np.allclose(mean, expected, rtol=1e-12, atol=1e-12)
    write(OUT / 'observer_audit.json', dict(status='PASS', initial_and_final_engine_bitwise=True,
        all_cell_spike_counts_exact=True, global_trace_bitwise=True, selected_current_moments_match=True,
        events=int(sp.nnz), samples=20000, targets=len(cells), producer_sha256=sha(__file__)))
    write(OUT / 'observer_progress.json', dict(status='COMPLETE_OBSERVATION_QA_PASS', updated_epoch=time.time()))
    print('COMPLETE TEMPORAL OBSERVER; full-engine replay PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare', 'worker'])
    p.add_argument('--device', type=int, default=1); a = p.parse_args()
    prepare() if a.command == 'prepare' else worker(a.device)
