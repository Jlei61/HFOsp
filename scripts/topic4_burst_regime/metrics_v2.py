"""Measurement amendment: exclude repeated multiplets from irregular labels.

V1's CV/CV2 screen cannot distinguish irregular timing from alternating short/long
intervals made by two offset clocks. A max-over-frequency ISI-permutation assay
detects that counterexample. Neither outcome identifies autonomous oscillators.
"""
from functools import lru_cache
import numpy as np
import metrics as v1

VERSION='native_burst_metrics_v2_frequency_standardized_ordering'

@lru_cache(maxsize=4096)
def _ordering_test(onsets_tuple,nframes,n_surrogates=199):
    onsets=np.array(onsets_tuple,float)
    if len(onsets)<8:return dict(ordering_p=None,ordering_peak_hz=None,ordering_peak_score=None)
    intervals=np.diff(onsets)
    frequencies=np.fft.rfftfreq(nframes,.01)
    selected=(frequencies>=max(.5,8/(nframes*.01)))&(frequencies<=15.)
    if not selected.any():return dict(ordering_p=None,ordering_peak_hz=None,ordering_peak_score=None)
    def spectrum(times):
        ix=np.rint(times/.01).astype(int)
        x=np.bincount(ix[(ix>=0)&(ix<nframes)],minlength=nframes).astype(float)
        return abs(np.fft.rfft(x-x.mean()))**2/max(len(times),1)
    observed=spectrum(onsets)[selected]
    rng=np.random.default_rng(994701)
    spectra=[observed]
    for _ in range(n_surrogates):
        times=np.r_[onsets[0],onsets[0]+np.cumsum(rng.permutation(intervals))]
        spectra.append(spectrum(times)[selected])
    spectra=np.asarray(spectra)
    # Identical lattice harmonics may dominate every raw maximum. Standardize
    # each frequency symmetrically over the observed+permutation ensemble before
    # taking a maximum, so preserved marginal/lattice peaks cannot hide ordering.
    mean=spectra.mean(0);sd=spectra.std(0)
    valid=sd>np.maximum(1e-9,abs(mean)*1e-9)
    if not valid.any():return dict(ordering_p=1.,ordering_peak_hz=None,ordering_peak_score=0.)
    standardized=(spectra[:,valid]-mean[valid])/sd[valid]
    maximum=standardized.max(1);stat=float(maximum[0])
    frequency=float(frequencies[selected][valid][np.argmax(standardized[0])])
    ge=int(np.count_nonzero(maximum[1:]>=stat-1e-10))
    return dict(ordering_p=(1+ge)/(1+n_surrogates),ordering_peak_hz=frequency,ordering_peak_score=stat)

def ordering_test(onsets,duration_s,n_surrogates=199):
    return dict(_ordering_test(tuple(np.round(onsets,8)),round(duration_s/.01),n_surrogates))

def amend(metric):
    onsets=[e['start_s'] for e in metric['events'] if not e['left_censored']]
    metric.update(ordering_test(onsets,metric['observed_s']))
    metric['v1_label']=metric['label']
    if metric['label']=='irregular_bursts' and metric['ordering_p'] is not None and metric['ordering_p']<=.05:
        metric['label']='patterned_bursts'
    return metric

def run_metrics(path,result):
    out=v1.run_metrics(path,result)
    with np.load(path) as z:
        names=z['group_names'].tolist()
        for group,metric in out.items():
            amend(metric)
            j=names.index(group);size=int(z['group_sizes'][j])
            alternatives=[]
            for threshold in (.075,.125):
                m=v1.summarize(z['active_counts_10ms'][:,j]/max(size,1),z['spike_counts_2ms'][:,j],size,
                    burnin_s=result['burnin_ms']/1000,runaway=result['runaway_early_stop_ms'] is not None,onset_threshold=threshold)
                alternatives.append(amend(m)['label'])
            metric['threshold_sensitivity_labels']=alternatives
            metric['threshold_stable']=all(x==metric['label'] for x in alternatives)
            metric['measurement_version']=VERSION
    return out
