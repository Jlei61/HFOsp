#!/usr/bin/env python3
"""Offline spike regularity and a longer-window 1–150-Hz sensitivity check."""
import json
import numpy as np
from scipy.signal import periodogram
from analyze_topic4_fig5_native_transition import BASE, OUT, WINDOWS, FS, write


def main():
    native=np.load(OUT/'native_fields.npz');a=np.load(OUT/'spatial_analysis.npz')
    original=np.load(BASE/'runs/continuous_refill_release.npz')
    raster=original['sample_spikes'];ids=original['sample_ids'];ne=len(native['positions_e'])
    groups=np.full(len(ids),3)
    is_e=ids<ne
    distance=np.linalg.norm(native['positions_e'][ids[is_e],None]-native['centers_mm'][None],axis=2)
    eg=np.full(is_e.sum(),2);eg[distance[:,0]<=1.5]=0;eg[distance[:,1]<=1.5]=1
    groups[is_e]=eg
    spike_records=[];names=['Core A E','Core B E','Surround E','All I']
    for lo,hi in np.vstack(([.5,8.],WINDOWS)):
        segment=raster[round(lo*10000):round(hi*10000)]
        for g,name in enumerate(names):
            cv=[];freq=[]
            for col in np.flatnonzero(groups==g):
                isi=np.diff(np.flatnonzero(segment[:,col]))*.1
                if len(isi)>=5:
                    cv.append(isi.std()/isi.mean());freq.append(1000/np.median(isi))
            spike_records.append(dict(window_s=[float(lo),float(hi)],region=name,
                sampled_neurons=int((groups==g).sum()),eligible_neurons=len(cv),
                minimum_within_window_intervals=5,
                median_ISI_CV=float(np.median(cv)) if cv else None,
                median_inverse_median_ISI_Hz=float(np.median(freq)) if freq else None,
                fraction_CV_below_005=float(np.mean(np.array(cv)<.05)) if cv else None))

    # Clinical Figure 3 uses 1–150 Hz with a 1-s PSD window. A 250-ms
    # periodogram cannot resolve its low-frequency end. This diagnostic uses
    # 1-s windows and exposes that temporal-resolution tradeoff explicitly.
    x=np.add(native['cell_mean_ampa'],native['cell_mean_applied_gaba'],dtype=np.float64)
    baseline_starts=np.arange(.5,7.01,.5)
    windows=np.array([[9.,10.],[9.5,10.5],[10.,11.]])
    def band(lo,detrend='constant'):
        f,p=periodogram(x[round(lo*FS):round((lo+1)*FS)],fs=FS,window='hann',
            detrend=detrend,scaling='density',axis=0)
        return np.array([p[(f>=low)&(f<=high)].sum(0)*(f[1]-f[0])
            for low,high in [(1,150),(1,19),(20,150)]])
    baseline_all=np.array([band(lo) for lo in baseline_starts]);power_all=np.array([band(lo) for lo in windows[:,0]])
    baseline=baseline_all[:,0];power=power_all[:,0]
    bp=baseline.mean(0);db=10*np.log10(power/bp)
    logs=np.log10(baseline);med=np.median(logs,axis=0);mad=1.4826*np.median(abs(logs-med),axis=0)
    z=np.divide(np.log10(power)-med,mad,out=np.full_like(power,np.nan),where=mad>1e-12)
    rows=[]
    base_linear=np.array([band(lo,'linear')[0] for lo in baseline_starts]).mean(0)
    for k,w in enumerate(windows):
        for g,name in enumerate(['Core A','Core B','Surround']):
            weights=a['region_cell_weights'][g]
            low=np.average(power_all[k,1],weights=weights)
            middle=np.average(power_all[k,2],weights=weights)
            rows.append(dict(window_s=w.tolist(),region=name,
                band_1_150_dB=float(10*np.log10(np.average(power[k],weights=weights)/np.average(bp,weights=weights))),
                band_1_150_linear_detrend_dB=float(10*np.log10(np.average(band(w[0],'linear')[0],weights=weights)/np.average(base_linear,weights=weights))),
                low_1_19_fraction_of_1_150_power=float(low/(low+middle)),
                band_20_150_same_1s_window_dB=float(10*np.log10(middle/np.average(baseline_all[:,2].mean(0),weights=weights))),
                weighted_mean_cell_log_power_robust_z=float(np.average(z[k],weights=weights))))
    np.savez_compressed(OUT/'broadband_1s_diagnostic.npz',windows_s=windows,
        baseline_window_starts_s=baseline_starts,band_power=power,baseline_band_power=bp,
        band_change_dB=db,log_power_robust_z=z,centers_mm=a['centers_mm'])
    result=dict(spike_records=spike_records,spike_scope='Fixed original sampled neurons; reclassified with the physical 1.5-mm E cores. ISI CV is single-neuron regularity, not population phase synchrony. Baseline is longer and includes interevent gaps.',
        broadband_records=rows,broadband_band_hz=[1,150],broadband_window_s=1.,
        broadband_baseline_s=[.5,8.],broadband_baseline_windows=len(baseline),
        broadband_scope='Sensitivity diagnostic matching the clinical frequency limits and 1-s duration, not a clinical replication. Hann periodogram with mean removal; 14 baseline windows are fewer than the clinical robust-z eligibility requirement. Wider windows mix pre-onset and tonic activity; no selection on power sign.',
        primary_short_window_analysis_unchanged=True)
    write('spikes_broadband_audit.json',result);print(json.dumps(result,indent=2))
    rate_records=[]
    for lo,hi in WINDOWS:
        for key in ['rate_e_hz','rate_i_hz']:
            r=original[key][round(lo*10000):round(hi*10000)]
            f,p=periodogram(r,fs=10000,window='hann',detrend='constant',scaling='density')
            valid=(f>=20)&(f<1000);high=(f>=250)&(f<500)
            rate_records.append(dict(window_s=[float(lo),float(hi)],population=key,
                mean_Hz=float(r.mean()),raw_bin_CV=float(r.std()/r.mean()),
                peak_above20_Hz=float(f[valid][np.argmax(p[valid])]),
                band_250_500_RMS_over_mean=float(np.sqrt(p[high].sum()*(f[1]-f[0]))/r.mean())))
    write('population_rate_spectrum_audit.json',dict(records=rate_records,
        observable='All original E or I spikes in 0.1-ms bins. Band RMS is the integrated PSD fluctuation amplitude divided by mean rate. Raw bin CV includes finite-count noise and is not a synchrony estimator.'))


if __name__=='__main__':main()
