#!/usr/bin/env python3
"""Numerical representation only: one fixed-order temporal residual model.

No physical network, new biological mechanism, or bifurcation is run here.
Fit coefficients to existing source-derived input spectra, never to rate error.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import shutil
import time
import numpy as np
from scipy import sparse
from scipy.linalg import toeplitz
from campaign import ROOT, read, write, sha

OUT = ROOT/'dynamic_colored_memory_preparation'
SOURCE = ROOT/'high_history_phase_source_closure'
ORDER = 64
BATCH = 512


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(ROOT/'filtered_residual_variance_audit/result.json')['status'] == 'COMPLETE_READONLY_FILTERED_MEMORY_AUDIT'
    write(OUT/'contract.json', dict(status='REGISTERED_FIXED_ORDER_NOISE_REPRESENTATION_ONLY', created_epoch=time.time(),
        question='Can a stable finite-memory representation retain the source-derived residual current spectrum well enough for a subsequent physical-time diagnostic?',
        evidence='Same-step Bernoulli variance and native Poisson external counts did not fix spurious G activation. Native refractory identities and filtered variance budgets show independent-time residuals discard a substantial source memory effect; remaining cross-source covariance is still missing.',
        design='Original jumpweights squared times the existing native72-74s source residual PSD and original discrete synaptic filter. For all40000 physical targets and E/I pathways, fit exactly one AR64 representation atdt0.1ms by Yule-Walker covariance equations. Order64 is fixed before fit; no order/amplitude/period search or biological parameter adjustment. Preserve exact projected spectra for review.',
        checks='Levinson solution versus independent CPU Toeplitz solves; positive innovations and reflectionmagnitudes<1; compare resulting PSD with target PSD, including its total variance and fullfrequency discrepancy. Fixed64 numerical memory may fail, which is retained. No relaxation of the native scientific gate.',
        representation_guard='A pathway retains this proposed representation only if weighted relative spectralL1 error median<=0.05 and95thpercentile<=0.15 overtargets. This is numerical fidelity to an approximate conditional input spectrum, not original-network correspondence.',
        scope='Existing native statistics seed only a possible diagnostic noise model. No future neural mean or phaseperiod is prescribed, but no self-generated noise covariance, autonomous physicalnetwork, stablebranch or bifurcation is established by this preparation.',
        stop='Prepare one order and report; no automatic dynamic experiment or alternate order. A future physical model must address initial residual-memory state, actual synaptic state/pending inputs, and covariance self-consistency explicitly.',
        order=ORDER, dt_ms=.1, source=str(SOURCE), producer_sha256=sha(__file__), formal_bifurcation_allowed=False))
    shutil.copy2(__file__, OUT/'producer.py')


def levinson(cp, covariance):
    n = covariance.shape[0];r = covariance/covariance[:, :1]
    a = cp.zeros((n, ORDER));error = cp.ones(n);largest = cp.zeros(n)
    for k in range(1, ORDER+1):
        numerator = r[:, k]
        if k > 1:numerator = numerator-cp.sum(a[:, :k-1]*r[:, k-1:0:-1], axis=1)
        reflection = numerator/error;largest = cp.maximum(largest, abs(reflection))
        if k > 1:a[:, :k-1] = a[:, :k-1]-reflection[:, None]*a[:, k-2::-1]
        a[:, k-1] = reflection;error *= 1-reflection**2
    return a, error*covariance[:, 0], largest


def run(device):
    import cupy as cp
    import cupyx.scipy.sparse as cs
    cp.cuda.Device(device).use()
    assert read(OUT/'contract.json')['producer_sha256'] == sha(__file__)
    assert not (OUT/'result.json').exists();started = time.time()
    source = cp.asarray(np.load(SOURCE/'generation_0/residual_PSD.npy', mmap_mode='r'))
    H = cp.asarray(np.load(SOURCE/'filter_power.npy'));N = 2*(source.shape[1]-1)
    weights = cp.full(N//2+1, 2.);weights[0] = weights[-1] = 1
    spectrum = np.lib.format.open_memmap(OUT/'projected_current_residual_PSD.npy', mode='w+', dtype='f8', shape=(2, 40000, N//2+1))
    coefficients = np.zeros((2, 40000, ORDER));innovation = np.zeros((2, 40000));errors = np.zeros((2, 40000, 3))
    maxreflection = np.zeros((2, 40000));checks = []
    for q, (kind, ss) in enumerate([('ampa', slice(0, 32000)), ('gaba', slice(32000, 40000))]):
        a = sparse.load_npz(SOURCE/f'original_{kind}_jump.npz').tocsr();W2 = cs.csr_matrix(a.multiply(a));del a
        for lo in range(0, 40000, BATCH):
            hi = min(lo+BATCH, 40000);n = hi-lo
            write(OUT/'progress.json', dict(status='FITTING_FIXED_ORDER', pid=os.getpid(), device=device,
                pathway=kind, completed_targets=lo, updated_epoch=time.time()))
            P = (W2[lo:hi]@source[ss])*H[q]
            cov = cp.fft.irfft(P, n=N, axis=1)/N
            assert bool(cp.isfinite(cov).all()) and float(cov[:, 0].min()) >= 0
            active = cov[:, 0] > 0
            assert not bool((P[~active] != 0).any())
            coef = cp.zeros((n, ORDER));var = cp.zeros(n);reflection = cp.zeros(n)
            if bool(active.any()):
                aa, vv, rr = levinson(cp, cov[active, :ORDER+1])
                coef[active] = aa;var[active] = vv;reflection[active] = rr
                assert float(vv.min()) > 0 and float(rr.max()) < 1
            polynomial = cp.zeros((n, N));polynomial[:, 0] = 1;polynomial[:, 1:ORDER+1] = -coef
            predicted = N*var[:, None]/abs(cp.fft.rfft(polynomial, axis=1))**2
            norm = cp.where(active, P@weights, 1.);variance = cp.where(active, cov[:, 0], 1.)
            l1 = (abs(predicted-P)@weights)/norm
            variance_error = abs(predicted@weights/N**2-cov[:, 0])/variance
            covpred = cp.fft.irfft(predicted, n=N, axis=1)/N
            short_cov_error = cp.max(abs(covpred[:, :ORDER+1]-cov[:, :ORDER+1]), axis=1)/variance
            spectrum[q, lo:hi] = P.get();coefficients[q, lo:hi] = coef.get();innovation[q, lo:hi] = var.get()
            errors[q, lo:hi] = cp.stack([l1, variance_error, short_cov_error], axis=1).get();maxreflection[q, lo:hi] = reflection.get()
            if lo == 0:
                ids = cp.flatnonzero(active)[:6];cc = cov[ids, :ORDER+1].get();aa = coef[ids].get();audit = []
                for i, r in enumerate(cc):
                    truth = np.linalg.solve(toeplitz(r[:ORDER]), r[1:ORDER+1])
                    err = float(abs(truth-aa[i]).max());assert err < 2e-5
                    residual = float(abs(toeplitz(r[:ORDER])@aa[i]-r[1:ORDER+1]).max()/r[0]);assert residual < 1e-9
                    audit.append(dict(target=int(ids[i]), coefficient_max_abs_difference=err, normalized_Yule_Walker_residual=residual))
                checks.append(dict(pathway=kind, independent_CPU=audit))
            del P, cov, coef, var, reflection, polynomial, predicted, covpred
        del W2;cp.get_default_memory_pool().free_all_blocks();spectrum.flush()
    np.savez_compressed(OUT/'AR64_parameters.npz', coefficient=coefficients, innovation_variance=innovation,
        max_abs_reflection=maxreflection, errors=errors, error_names=['relative_PSD_L1', 'relative_variance', 'normalized_short_covariance'])
    rows = []
    for q, name in enumerate(['AMPA', 'GABA']):
        nonzero = innovation[q] > 0
        quantile = np.quantile(errors[q, nonzero, 0], [.5, .95, 1.])
        rows.append(dict(pathway=name, relative_PSD_L1_median_p95_max=quantile.tolist(),
            nonzero_spectrum_targets=int(nonzero.sum()), exact_zero_spectrum_targets=int((~nonzero).sum()),
            maximum_relative_variance_error=float(errors[q, :, 1].max()),
            maximum_normalized_short_covariance_error=float(errors[q, :, 2].max()),
            maximum_abs_reflection=float(maxreflection[q].max()), representation_guard=bool(quantile[0] <= .05 and quantile[1] <= .15)))
    result = dict(status='COMPLETE_ONE_AR64_REPRESENTATION', rows=rows, checks=checks,
        representation_retained=all(r['representation_guard'] for r in rows),
        physical_model_simulated=False, formal_bifurcation_allowed=False, elapsed_s=time.time()-started,
        producer_sha256=sha(__file__))
    write(OUT/'result.json', result);write(OUT/'progress.json', dict(status=result['status'], updated_epoch=time.time()));print(result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('command', choices=['prepare', 'run']);p.add_argument('--device', type=int, default=1);a = p.parse_args()
    if a.command == 'prepare':prepare()
    else:
        try:run(a.device)
        except Exception:
            write(OUT/'progress.json', dict(status='FAILED', pid=os.getpid(), updated_epoch=time.time()));raise
