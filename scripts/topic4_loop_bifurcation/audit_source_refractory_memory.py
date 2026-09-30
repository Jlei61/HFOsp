#!/usr/bin/env python3
"""Read-only covariance identity for native spikes and independent-time closure."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import shutil
import time
import numpy as np
from scipy import sparse
from campaign import ROOT, read, write, sha
import observe_high_history_spectra as observer

OUT = ROOT/'source_refractory_memory_audit'
SOURCES = ['dynamic_individual_source_pilot/individual_source', 'dynamic_bernoulli_source_pilot', 'dynamic_poisson_external_pilot']
LAGS = [0, 1, 5, 10, 19, 20, 22, 44]


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'result.json').exists();started = time.time()
    write(OUT/'contract.json', dict(status='READONLY_TEMPORAL_COVARIANCE_DIAGNOSTIC', created_epoch=time.time(),
        question='Does the independent-time residual assumption discard source refractory memory even after its same-bin variance and external count law are corrected?',
        data='Existing unchanged72-74s native source spikes, plus last359 steps of three completed physical-time density probability histories. The native observer has its original varying background, not the fixed-background control; no paired trajectory claim. The refractory identity is exact for either background.',
        identity='For each E source and lag1..19 atdt0.1ms, n(t)*n(t+lag)=0 because native refractory is20steps. Given marginal probabilities p(t), residual covariance must equal -p(t)*p(t+lag); independent-time residuals instead assume zero. This concerns the recurrent-input approximation, not a claim that the density cells themselves violate refractory updates.',
        phase_diagnostic='For the native finite record only, decompose n into its measured22step phase conditional mean and residual. Evaluate the finite-record covariance identity exactly. This period is a diagnostic from earlier native observation, not an imposed new drive or a new physical model.',
        scope='No simulation, parameter fitting, native seed or formal bifurcation. Nonzero covariance alone does not prove it causes the complete rate/spatial error.', producer_sha256=sha(__file__)))
    sp = sparse.load_npz(observer.OUT/'source_spikes_0p1ms.npz').tocsr();assert sp.shape == (40000, 20000)
    rows = np.repeat(np.arange(40000), np.diff(sp.indptr));tm = sp.indices
    isi = np.diff(tm);same = np.diff(rows) == 0;source = rows[:-1]
    assert isi[same & (source < 32000)].min() >= 20
    assert isi[same & (source >= 32000)].min() >= 10
    L = 22;T = sp.shape[1];exposure = np.bincount(np.arange(T) % L, minlength=L)
    phase = np.bincount(rows*L+tm % L, minlength=40000*L).reshape(40000, L)/exposure
    covariances = []
    for label, lo, hi in [('E', 0, 32000), ('I', 32000, 40000)]:
        use = (rows >= lo) & (rows < hi);cell, tick = rows[use]-lo, tm[use];p = phase[lo:hi];n = hi-lo
        matrix = sp[lo:hi];lag_rows = []
        for lag in LAGS:
            denom = n*(T-lag)
            if lag == 0:nn = float(matrix.nnz)/denom
            elif lag < (20 if label == 'E' else 10):nn = 0.
            else:nn = float(matrix[:, :T-lag].multiply(matrix[:, lag:]).sum())/denom
            left = tick < T-lag;right = tick >= lag
            np_ = float(p[cell[left], (tick[left]+lag) % L].sum()/denom)
            pn = float(p[cell[right], (tick[right]-lag) % L].sum()/denom)
            exposures = np.bincount(np.arange(T-lag) % L, minlength=L)
            pp = float((p*p[:, (np.arange(L)+lag) % L]*exposures).sum()/denom)
            lag_rows.append(dict(lag_steps=lag, lag_ms=lag*.1, native_joint_probability=nn,
                phase_mean_product=pp, residual_covariance=nn-np_-pn+pp))
        variance = lag_rows[0]['residual_covariance'];assert variance > 0
        for r in lag_rows:r['residual_correlation'] = r['residual_covariance']/variance
        covariances.append(dict(population=label, residual_variance=variance, lags=lag_rows))
    histories = []
    for name in SOURCES:
        with np.load(ROOT/name/'final_state.npz') as z:
            tick = int(z['clock'][0]);h = z['source_history'];D = len(h)
            p = h[(tick-D+np.arange(D)) % D]*.1
        assert p.min() >= 0 and p.max() <= 1
        rows_ = []
        for label, lo, hi, refractory in [('E', 0, 32000, 20), ('I', 32000, 40000, 10)]:
            v = p[:, lo:hi];var = float((v*(1-v)).mean());pairs = []
            for lag in [1, 5, refractory-1]:
                product = float((v[:-lag]*v[lag:]).mean())
                pairs.append(dict(lag_ms=lag*.1, required_residual_covariance=-product,
                    independent_time_assumption=0., magnitude_over_same_bin_variance=product/var))
            rows_.append(dict(population=label, same_bin_variance=var, forbidden_lag_identities=pairs))
        histories.append(dict(model=name, last_history_s=[(tick-D)*.0001, tick*.0001], populations=rows_))
    result = dict(status='COMPLETE_READONLY_REFRACTORY_MEMORY_AUDIT', native_minimum_ISI_steps=dict(
        E=int(isi[same & (source < 32000)].min()), I=int(isi[same & (source >= 32000)].min())),
        native_phase_residual_covariance=covariances, model_required_covariances=histories,
        conclusion='Source refractory memory requires nonzero negative short-lag residual covariance. The present independent-time recurrent Gaussian approximation discards it even with exact same-bin Bernoulli variance. Native observed phase-residual correlations are diagnostic only; the magnitude and causal role of the resulting network error still require a targeted temporal-memory closure test.',
        formal_bifurcation_allowed=False, producer_sha256=sha(__file__), elapsed_s=time.time()-started)
    write(OUT/'result.json', result);shutil.copy2(__file__, OUT/'producer.py')
    print(dict(status=result['status'], native=covariances), flush=True)


if __name__ == '__main__':main()
