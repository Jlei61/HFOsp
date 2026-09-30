#!/usr/bin/env python3
"""One second-history spectral response, before any more root refinement."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, shutil, subprocess, time
from pathlib import Path
import numpy as np
from scipy import fft, sparse
from campaign import ROOT, PYTHON, read, write, sha
from conditional_density_inputs import OPS
import run_spectral_closure_pilot as implementation
from individual_spectral_sampler import kernel, make_parameters, run
import observe_high_history_spectra as observer

OUT = ROOT / 'high_history_spectral_value'
SOURCE = ROOT / 'individual_source_spectral_pilot'
MATCHED = ROOT / 'native_K9p35_constant_background_pair_v2'
N = 20000
REPLICAS = 64


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'contract.json').exists()
    assert read(observer.OUT / 'observer_audit.json')['status'] == 'PASS'
    write(OUT / 'contract.json', dict(
        status='REGISTERED_BEFORE_SECOND_HISTORY_RESPONSE', created_epoch=time.time(),
        question='Does the unchanged individual-source spectral map retain the native double-core high-history state and the correct G-off segment at the same held Z/K and fixed external mean?',
        design='Exactly one response F(X_native_high), 40000 targets times64 numerical replicas, from the completed72-74s native source record. Reuse original weights, individual thresholds, filters and unchanged sampler. Same2s FFT,3-4s burn, dynamicM and stationaryG protocol. No damping, root solve, new parameter, or automatic next generation.',
        interpretation='Development-anchor local response, not independent self-consistency or physical stability. Native sources provide the initial candidate only; one returned output does not certify a self-generated solution. Compare both native variable-background2s anchor and completed constant-background10s tail explicitly.',
        decision='If either core, spatial recruitment, G segment or counterfactual coreZ drift fails correspondence, stop root refinement and diagnose that failure. Retained state permits further bounded work but is not a formal gate pass.',
        uncertainty='64 independent numerical target replicas do not constitute native seeds. Source cross spectra and shared-input correlations omitted; finite2s periodic drive remains an approximation.',
        unit='One second native conditional history at the existing Z0.21/K9.35 spatial field family.',
        replicas=REPLICAS, generations=1, source=str(observer.OUT),
        source_spikes_sha256=sha(observer.OUT / 'source_spikes_0p1ms.npz'),
        unchanged_map_sha256=sha(implementation.__file__),
        sampler_sha256=sha(Path(__file__).with_name('individual_spectral_sampler.py')),
        producer_sha256=sha(__file__), formal_bifurcation_allowed=False,
        counts_as_autonomous_loop=False))
    for name in ['original_ampa_jump.npz', 'original_gaba_jump.npz', 'filter_power.npy', 'threshold_identity.json']:
        (OUT / name).symlink_to(SOURCE / name)
    raw = dict(np.load(SOURCE / 'parameters.npz'))
    initial = observer.base.native.read_pickle(observer.INITIAL)['engine']
    stats = dict(np.load(observer.OUT / 'cell_statistics.npz'))
    assert np.array_equal(raw['Z'], initial['slow']['z'])
    assert np.array_equal(raw['K'][:32000], initial['termination_mechanism']['sahp_g'])
    nu = stats['per_cell_mean_external_per_ms'].mean(0)
    assert np.array_equal(nu, raw['nu_per_ms'])
    raw.update(initial_V=initial['V'], initial_ref=initial['ref'], initial_M=initial['slow']['m'],
               original_moments=stats['per_cell_mean_moments'].mean(0))
    np.savez_compressed(OUT / 'parameters.npz', **raw)
    sp = sparse.load_npz(observer.OUT / 'source_spikes_0p1ms.npz').tocsr()
    assert sp.shape == (40000, N)
    folder = OUT / 'generation_0'; folder.mkdir()
    psd = np.lib.format.open_memmap(folder / 'source_PSD.npy', mode='w+', dtype='f8', shape=(40000, N // 2 + 1))
    weights = np.full(N // 2 + 1, 2.); weights[[0, -1]] = 1
    error = 0.
    for lo in range(0, 40000, 128):
        x = sp[lo:lo + 128].toarray().astype(float); x -= x.mean(1, keepdims=True)
        p = abs(fft.rfft(x, axis=1, workers=2)) ** 2; p[:, 0] = 0
        error = max(error, float(abs(p @ weights / N**2 - x.var(1)).max()))
        psd[lo:lo + len(x)] = p
    psd.flush(); assert error < 1e-12
    rates = np.asarray(sp.sum(1)).ravel() / 2.
    np.save(folder / 'source_rate_Hz.npy', rates)
    write(folder / 'complete.json', dict(status='COMPLETE_NATIVE_HIGH_INITIAL_GUESS_ONLY'))
    write(OUT / 'preparation_qa.json', dict(status='PASS', source_Parseval_max_abs_error=error,
        original_graph_filters_thresholds_reused=True, same_full_Z_K=True, same_fixed_external_mean_exact=True,
        true_high_history_initial_V_ref_M=True, high_native_allE_rate_Hz=float(rates[:32000].mean())))
    shutil.copy2(__file__, OUT / 'producer.py')


def configure():
    c = read(OUT / 'contract.json')
    assert c['unchanged_map_sha256'] == sha(implementation.__file__)
    assert c['producer_sha256'] == sha(__file__)
    assert c['sampler_sha256'] == sha(Path(__file__).with_name('individual_spectral_sampler.py'))
    implementation.OUT = OUT; implementation.GENERATIONS = 1; implementation.REPLICAS = REPLICAS


def qa():
    import cupy as cp
    cp.cuda.Device(0).use(); fn = kernel(cp)
    raw = dict(np.load(OUT / 'parameters.npz')); params = read(OPS / 'prepared.json')['params']
    tr = dict(np.load(observer.OUT / 'target_traces.npz')); cells = tr['cells']
    sp = sparse.load_npz(observer.OUT / 'source_spikes_0p1ms.npz').tocsr()[cells].toarray().T
    assert tr['causal_R_and_s'][:, 0].max() < 200
    assert abs(tr['causal_R_and_s'][:, 1]).max() < 1e-20
    p, cfg = make_parameters(raw, cells, np.zeros(len(cells)), 0., params, True)
    flags, st = run(cp, fn, cp.asarray(np.ascontiguousarray(tr['IE_II_V_M'][:, 0, :])),
        cp.asarray(np.ascontiguousarray(tr['IE_II_V_M'][:, 1, :])), p, cfg, 1, 0,
        np.zeros(len(cells), 'i4'), 929531, 0, True)
    assert np.array_equal(flags.get(), sp)
    assert np.allclose(st.get()[:, 0, 4], tr['IE_II_V_M'][:, 3, :].mean(0), rtol=0, atol=1e-12)
    write(OUT / 'implementation_qa.json', dict(status='PASS', targets=len(cells),
        every_observed_high_history_spike_exact=True, native_M_mean_exact=True,
        original_graph_filter_projection_QA=str(SOURCE / 'implementation_qa.json'),
        residual_native_G_below_1e_minus20=True))


def supervise():
    configure(); assert not (OUT / 'supervisor.json').exists()
    while not (MATCHED / 'result.json').exists():
        if read(MATCHED / 'supervisor.json')['status'] == 'FAILED':
            raise RuntimeError('Native controls failed; inspect before GPU dispatch')
        write(OUT / 'supervisor.json', dict(status='WAITING_NATIVE_CONSTANT_BACKGROUND_CONTROLS',
            pid=os.getpid(), updated_epoch=time.time()))
        time.sleep(10)
    assert read(MATCHED / 'result.json')['status'] == 'COMPLETE_TWO_NATIVE_CONSTANT_BACKGROUND_CONTROLS'
    write(OUT / 'supervisor.json', dict(status='RUNNING', pid=os.getpid(), updated_epoch=time.time()))
    qa()
    folder = OUT / 'generation_1'; folder.mkdir()
    for name, shape in [('source_PSD.npy', (40000, N // 2 + 1)), ('source_rate_Hz.npy', (40000,))]:
        a = np.lib.format.open_memmap(folder / name, mode='w+', dtype='f8', shape=shape)
        a[:] = np.nan; a.flush(); del a
    jobs = []
    for part in range(2):
        log = (folder / f'worker_part{part}.log').open('w')
        p = subprocess.Popen([PYTHON, __file__, 'worker', '--part', str(part)], stdout=log, stderr=subprocess.STDOUT)
        jobs.append((p, log))
    codes = []
    for p, log in jobs: codes.append(p.wait()); log.close()
    if any(codes):
        write(OUT / 'supervisor.json', dict(status='FAILED', codes=codes)); raise RuntimeError(codes)
    implementation.collect(1)
    result = read(folder / 'complete.json')
    result.update(status='COMPLETE_SINGLE_HIGH_HISTORY_SPECTRAL_RESPONSE', self_consistency_established=False)
    write(OUT / 'result.json', result)
    write(OUT / 'supervisor.json', dict(status='COMPLETE', updated_epoch=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare', 'supervise', 'worker'])
    p.add_argument('--part', type=int, default=0); a = p.parse_args()
    if a.command == 'prepare': prepare()
    elif a.command == 'supervise': supervise()
    else: configure(); implementation.worker(1, a.part, a.part, 128)
