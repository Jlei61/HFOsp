#!/usr/bin/env python3
"""Offline checks of baseline, detrending and the observed high-frequency component."""
import json
import numpy as np
from scipy.signal import periodogram
from analyze_topic4_fig5_native_transition import BASE, OUT, WINDOWS, BANDS, FS, write


def main():
    a=np.load(OUT/'spatial_analysis.npz');source=np.load(OUT/'native_fields.npz')
    x=np.add(source['cell_mean_ampa'],source['cell_mean_applied_gaba'],dtype=np.float64)
    weights=a['region_cell_weights'];regions=['Core A','Core B','Surround']
    rate=np.load(BASE/'runs/continuous_refill_release.npz')['rate_e_hz']
    bases=np.arange(.5,8,.25);activity=[];peak=[]
    for lo in bases:
        r=rate[round(lo*10000):round((lo+.25)*10000)]
        activity.append(r.mean());peak.append(r.reshape(-1,100).mean(1).max())
    activity=np.array(activity);peak=np.array(peak)
    quiet=np.flatnonzero(peak<1.)
    low=np.argsort(activity,kind='stable')[:3]
    baseline_sets={'all_interictal':np.arange(30),'lowest_activity_3_windows':low}
    if len(quiet):baseline_sets['quiet_250ms_windows']=quiet
    def power(lo,hi,detrend):
        f,p=periodogram(x[round(lo*FS):round(hi*FS)],fs=FS,window='hann',detrend=detrend,axis=0,scaling='density')
        return f,p,np.array([p[(f>=l)&(f<h)].sum(0)*(f[1]-f[0]) for l,h in BANDS])
    records=[];details=[]
    for detrend in ['linear','constant']:
        bp=np.array([power(lo,lo+.25,detrend)[2] for lo in bases])
        for key,ids in baseline_sets.items():
            baseline=bp[ids].mean(0)
            for k,(lo,hi) in enumerate(WINDOWS):
                f,p,v=power(lo,hi,detrend)
                for j,region in enumerate(regions):
                    w=weights[j]
                    ratio=10*np.log10(np.average(v,weights=w,axis=1)/np.average(baseline,weights=w,axis=1))
                    records.append(dict(detrend=detrend,baseline=key,window=[float(lo),float(hi)],region=region,band_change_dB=ratio.tolist()))
        if detrend=='linear':
            for k,(lo,hi) in enumerate(WINDOWS):
                f,p,v=power(lo,hi,detrend)
                xx=x[round(lo*FS):round(hi*FS)]
                for j,region in enumerate(regions):
                    w=weights[j];s=np.average(p,weights=w,axis=1)
                    valid=(f>=20)&(f<1000)
                    details.append(dict(window=[float(lo),float(hi)],region=region,
                        dominant_frequency_above20_Hz=float(f[valid][np.argmax(s[valid])]),
                        local_temporal_rms_over_mean=float(np.sqrt(np.average(xx.var(0),weights=w))/np.average(xx.mean(0),weights=w))))
    result=dict(primary_unchanged=True,bands_hz=BANDS,baseline_window_starts_s=bases.tolist(),
        baseline_window_E_rates_Hz=activity.tolist(),baseline_window_peak_10ms_E_rates_Hz=peak.tolist(),
        quiet_window_ids=quiet.tolist(),lowest_activity_window_ids=low.tolist(),
        baseline_selection='Quiet: every 10-ms all-E bin <1 Hz throughout a full 250-ms window. Low activity: three lowest all-E mean-rate windows, selected without any spectral criterion; not relabeled quiet.',
        comparisons=records,spectral_details=details,
        interpretation='Primary baseline remains all 30 interictal windows. Alternative baselines and detrending expose sensitivity, not a license to select the most positive map. Cell-mean power differs from mean individual-neuron power.')
    write('spectrum_robustness.json',result)
    print(json.dumps(dict(quiet_windows=len(quiet),lowest_activity_window_starts_s=bases[low].tolist(),lowest_E_rates=activity[low].tolist(),
        core_rows=[r for r in records if r['region']!='Surround' and r['window']==[10.5,10.75]],details=details),indent=2))


if __name__=='__main__':main()
