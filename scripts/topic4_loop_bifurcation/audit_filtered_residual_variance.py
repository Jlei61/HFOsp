#!/usr/bin/env python3
"""Read-only effect of source temporal memory after the actual synaptic filters."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import shutil
import time
import numpy as np
from scipy import sparse
from campaign import ROOT, read, write, sha

OUT = ROOT/'filtered_residual_variance_audit'
SOURCE = ROOT/'high_history_phase_source_closure'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'result.json').exists();started = time.time()
    write(OUT/'contract.json', dict(status='READONLY_FILTERED_MEMORY_AUDIT', created_epoch=time.time(),
        question='How much does discarding the measured source residual temporal spectrum change target current variance after the unchanged graph and synaptic filters?',
        design='Reuse the already validated native72-74s phase residual PSD, original physical jumpweights and exact discretefilter power. Compare independent coloredsource prediction W2*integral(H2*S) with whiteprediction W2*var(epsilon)*filterenergy. Also report a finite2s zeroDC white comparison. No neuronal simulation or fittedparameter.',
        interpretation='This is a conditionalinput-statistics diagnostic using an existing native record, not autonomous covariance closure. Both approximations omit cross-source residual spectra; measured59target residualvariance is shown separately. Source auto spectra alone may still be insufficient.',
        producer_sha256=sha(__file__), formal_bifurcation_allowed=False))
    psd = np.load(SOURCE/'generation_0/residual_PSD.npy', mmap_mode='r')
    H = np.load(SOURCE/'filter_power.npy');N = 2*(psd.shape[1]-1);assert H.shape == (2, N//2+1)
    weights = np.full(N//2+1, 2.);weights[[0, -1]] = 1
    totalvar = np.zeros(40000);filtered = np.zeros(40000)
    for lo in range(0, 40000, 256):
        hi = min(lo+256, 40000);a = np.asarray(psd[lo:hi]);totalvar[lo:hi] = a@weights/N**2
        for q, lower, upper in [(0, 0, 32000), (1, 32000, 40000)]:
            aa, bb = max(lo, lower), min(hi, upper)
            if bb > aa:filtered[aa:bb] = a[aa-lo:bb-lo]@(weights*H[q])/N**2
    fields = [];white_dc = [];white_no_dc = []
    for q, (kind, ss) in enumerate([('ampa', slice(0, 32000)), ('gaba', slice(32000, 40000))]):
        W = sparse.load_npz(SOURCE/f'original_{kind}_jump.npz').tocsr();W2 = W.multiply(W)
        energy = float(weights@H[q]/N)
        energy_finite = float((weights@H[q]-H[q, 0])/(N-1))
        fields.append(W2@filtered[ss]);white_dc.append(W2@(totalvar[ss]*energy));white_no_dc.append(W2@(totalvar[ss]*energy_finite))
    colored, white, white_finite = map(np.asarray, [fields, white_dc, white_no_dc])
    raw = np.load(SOURCE/'parameters.npz');region = raw['region'];E = np.arange(40000) < 32000
    measured = np.load(ROOT/'high_history_coherent_phase_component/target_phase_inputs.npz')
    cells = measured['cells'];nativevar = measured['residual_variance']
    rows = []
    for label, mask in [('allE', E), ('coreA', E & (region == 0)), ('coreB', E & (region == 1)), ('otherE', E & (region == 2)), ('I', ~E)]:
        rows.append(dict(population=label, independent_colored_mean_IE_II_mV2=colored[:, mask].mean(1).tolist(),
            independent_white_mean_IE_II_mV2=white[:, mask].mean(1).tolist(),
            finite_zeroDC_white_mean_IE_II_mV2=white_finite[:, mask].mean(1).tolist(),
            white_to_colored_ratio=(white[:, mask].sum(1)/np.maximum(colored[:, mask].sum(1), 1e-30)).tolist()))
    selected = dict(targets=len(cells), native_actual_residual_IE_II_mean_mV2=nativevar.mean(1).tolist(),
        independent_colored_IE_II_mean_mV2=colored[:, cells].mean(1).tolist(),
        independent_white_IE_II_mean_mV2=white[:, cells].mean(1).tolist(),
        limitation='These59 targets were selected in earlier error diagnostics, not a random spatial sample; native residual includes sourcecross correlations and finitephase/filtereffects.')
    np.savez_compressed(OUT/'variance_inputs.npz', source_residual_variance=totalvar, source_filtered_variance=filtered,
        target_colored=colored, target_white=white, target_white_zeroDC=white_finite,
        selected_cells=cells, selected_native_residual=nativevar)
    result = dict(status='COMPLETE_READONLY_FILTERED_MEMORY_AUDIT', rows=rows, selected_target_diagnostic=selected,
        formal_bifurcation_allowed=False, producer_sha256=sha(__file__), elapsed_s=time.time()-started)
    write(OUT/'result.json', result);shutil.copy2(__file__, OUT/'producer.py');print(result, flush=True)


if __name__ == '__main__':main()
