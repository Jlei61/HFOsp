#!/usr/bin/env python3
"""Finite-window source-autospectrum input variance; no fitted noise factor."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, time
import numpy as np
from scipy import sparse, fft
from campaign import ROOT, read, write, sha
from observe_source_time_structure import OUT as SOURCE
from observe_source_aggregation import OUT as MOMENTS
from coupled_density_exit import ADAPTED
from conditional_density_inputs import OPS
from analyze_source_aggregation import filter_square_sum
import run_topic4_loop_zk_conditional as native

OUT = SOURCE / 'spectral_analysis'


def main(wait):
    OUT.mkdir(exist_ok=True); assert not (OUT / 'contract.json').exists()
    write(OUT / 'contract.json', dict(status='REGISTERED_BEFORE_SOURCE_SPECTRA', created_epoch=time.time(),
        question='How much of the independent-white current variance excess is removed by keeping each source own observed temporal autocorrelation while still neglecting correlations between distinct sources?',
        method='Use each originalsource0.1ms binarytrain, remove DC, periodic2s FFT, multiply by exact discrete native synaptic transfer and sum frequency power. Original squared targetweights project single-source filtered variances. Native external Poisson variance is retained from inferred actual counts. Check white variance against prior independent calculation and frequency/time Parseval identities.',
        limits='Data-conditioned source spectra, not predicted or autonomous. Periodic2s estimate has boundary and finite-window effects. Residual target variance includes cross-source/input correlations, nonstationarity and boundary errors; it is not a pure cross-covariance estimate. No fitted factor, no new network or bifurcation.',
        references=['https://doi.org/10.3389/fncom.2014.00104', 'https://doi.org/10.1103/PhysRevResearch.1.023024'],
        producer_sha256=sha(__file__), formal_bifurcation_allowed=False))
    while not (SOURCE / 'observer_audit.json').exists():
        write(OUT / 'progress.json', dict(status='WAITING_COMPLETE_TEMPORAL_RECORD', pid=os.getpid(), updated_epoch=time.time()))
        if not wait: return
        time.sleep(15)
    assert read(SOURCE / 'observer_audit.json')['status'] == 'PASS'
    sp = sparse.load_npz(SOURCE / 'source_spikes_0p1ms.npz').tocsr()
    assert sp.shape == (40000, 20000) and np.max(sp.data) == 1
    geo = dict(np.load(ADAPTED / 'geometry.npz')); p = read(OPS / 'prepared.json')['params']
    n = sp.shape[1]; f = fft.rfftfreq(n, d=.0001); phase = np.exp(-2j*np.pi*f*.0001)
    weights = np.full(len(f), 2.); weights[[0,-1]] = 1.
    filters = []
    for kind in ['AMPA', 'GABA']:
        ar, ad = np.exp(-.1/p['tau_r_'+kind]), np.exp(-.1/p['tau_d_'+kind])
        H = (1-ad)/((1-ar*phase)*(1-ad*phase))
        assert abs(np.dot(abs(H)**2, weights)/n - filter_square_sum(ar, ad)) < 1e-10
        filters.append(H)
    cellvar = np.empty((2, 40000)); cv = np.full(40000, np.nan); lag1 = cv.copy(); intervals = np.zeros(40000, dtype='i4')
    counts = np.diff(sp.indptr); parseval = []
    for lo in range(0, 40000, 256):
        write(OUT / 'progress.json', dict(status='SOURCE_AUTOSPECTRA', pid=os.getpid(), completed=lo, total=40000, updated_epoch=time.time()))
        x = sp[lo:lo+256].toarray().astype(float); x -= x.mean(1, keepdims=True)
        X = fft.rfft(x, axis=1, workers=2); power = abs(X)**2*weights/n**2
        assert np.allclose(power.sum(1), x.var(1), rtol=1e-10, atol=1e-12)
        for k, H in enumerate(filters):
            cellvar[k, lo:lo+len(x)] = power @ abs(H)**2
            if lo in [0, 32000]:
                a = fft.irfft(X[:4]*H, n=n, axis=1)
                err = float(abs(a.var(1)-cellvar[k,lo:lo+len(a)]).max())
                assert err < 1e-10; parseval.append(err)
        for i in range(lo, min(lo+256, 40000)):
            times = sp.indices[sp.indptr[i]:sp.indptr[i+1]]; isi = np.diff(times).astype(float)
            intervals[i] = len(isi)
            if len(isi) >= 20:
                cv[i] = isi.std()/isi.mean()
                if isi[:-1].std() > 0 and isi[1:].std() > 0:
                    lag1[i] = np.corrcoef(isi[:-1], isi[1:])[0,1]
    d = dict(np.load(MOMENTS / 'input_analysis/input_reconstruction.npz'))
    assert np.array_equal(counts/2, d['native_rate_Hz'])
    write(OUT / 'progress.json', dict(status='ORIGINAL_SQUARED_WEIGHT_PROJECTION', pid=os.getpid(), updated_epoch=time.time()))
    sim, _, _, identity = native.base.old.setup(9108405)
    assert identity == read(OPS / 'prepared.json')['graph_identity']
    predictions = np.zeros((2, 40000)); white = predictions.copy(); overlaps = []
    for k, (kind, sources) in enumerate([('ampa', slice(0,32000)), ('gaba', slice(32000,40000))]):
        matrices = sim.net[kind+'_by_delay']; combined = sum(matrices)
        overlap = int(sum(m.nnz for m in matrices) - combined.nnz)
        assert overlap == 0, 'Repeated source-target edges at distinct delays require coherent auto-spectrum terms.'
        overlaps.append(overlap)
        for m in matrices:
            if m.nnz:
                squared = m.multiply(m)
                predictions[k] += squared @ cellvar[k,sources]
                white[k] += squared @ (counts[sources]/n)
        ar, ad = np.exp(-.1/p['tau_r_'+kind.upper()]), np.exp(-.1/p['tau_d_'+kind.upper()])
        white[k] *= filter_square_sum(ar,ad)
    tm = np.r_[np.full(32000,p['tau_m_E']),np.full(8000,p['tau_m_I'])]
    jump = tm/p['tau_r_AMPA'] * np.r_[np.full(32000,p['J_ext_E']),np.full(8000,p['J_ext_I'])]
    ar, ad = np.exp(-.1/p['tau_r_AMPA']), np.exp(-.1/p['tau_d_AMPA'])
    extvar = np.rint(d['inferred_actual_external_counts'])/n*jump**2*filter_square_sum(ar,ad)
    predictions[0] += extvar; white[0] += extvar
    oldwhite = d['independent_white_variance_IE_II_cell_group'][:,:,0]
    relative = float(np.linalg.norm(white-oldwhite)/np.linalg.norm(oldwhite)); assert relative < 1e-12
    group = geo['cell_group']; region = geo['group_region'][group]; display = geo['group_cell'][group]; E = np.arange(40000)<32000
    edges = [r['cell'] for r in read(MOMENTS / 'contract.json')['edge_cells']]
    rows = []
    for label,mask in [('allE',E),('coreA',E&(region==0)),('coreB',E&(region==1)),
                       ('surroundE',E&(region==2)),('I',~E),('selected10edge_cells_E',E&np.isin(display,edges))]:
        usable = mask & np.isfinite(cv); serial = mask & np.isfinite(lag1)
        rows.append(dict(region=label,targets=int(mask.sum()),
            measured_IE_II_variance=d['observed_variance_IE_II'][:,mask].mean(1).tolist(),
            white_IE_II_variance=white[:,mask].mean(1).tolist(),
            source_auto_IE_II_variance=predictions[:,mask].mean(1).tolist(),
            source_CV_eligible=int(usable.sum()), median_source_CV=float(np.median(cv[usable])) if usable.any() else None,
            source_CV_q10_q90=np.quantile(cv[usable],[.1,.9]).tolist() if usable.any() else None,
            median_lag1_ISI_correlation=float(np.median(lag1[serial])) if serial.any() else None))
    np.savez_compressed(OUT/'source_spectra.npz', source_filtered_variance=cellvar, source_CV=cv,
        source_lag1_ISI_correlation=lag1, observed_intervals=intervals,
        predicted_IE_II_variance=predictions, white_IE_II_variance=white, external_variance=extvar)
    result = dict(status='COMPLETE_SOURCE_AUTOCORRELATION_DIAGNOSTIC',rows=rows,
        white_projection_relative_error=relative, filter_Parseval_errors=parseval,
        repeated_source_target_delay_edges=overlaps, data_prescribed_not_autonomous=True,
        formal_bifurcation_allowed=False, producer_sha256=sha(__file__))
    write(OUT/'result.json',result); write(OUT/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()))
    print(result,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
