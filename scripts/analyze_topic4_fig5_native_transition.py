#!/usr/bin/env python3
"""Native rate and band-power maps for the actual state-2 to state-3 path."""
import csv
import json
from pathlib import Path
import warnings
import numpy as np
from scipy.signal import periodogram
from scipy.stats import rankdata
from scipy.ndimage import gaussian_filter1d
from replay_topic4_fig5_native_transition import BASE, OUT, write

BANDS=[(20,150),(20,40),(40,80),(80,150),(150,250),(30,80),(250,500)]
WINDOWS=np.array([[10.25,10.5],[10.5,10.75],[10.75,11.0]])
FS=2000.


def spectrum(x,lo,hi):
    segment=x[round(lo*FS):round(hi*FS)]
    f,p=periodogram(segment,fs=FS,window='hann',detrend='linear',scaling='density',axis=0)
    df=f[1]-f[0]
    power=np.stack([p[(f>=low)&(f<high)].sum(0)*df for low,high in BANDS])
    return f,p,power


def main():
    qa=json.loads((OUT/'qa.json').read_text());assert all(qa['checks'].values())
    source=np.load(OUT/'native_fields.npz')
    x=np.add(source['cell_mean_ampa'],source['cell_mean_applied_gaba'],dtype=np.float64)
    ecount=source['E_count_1ms'];icount=source['I_count_1ms'];nc=source['cell_n_E'];nic=source['cell_n_I']
    baseline=[];psd=[]
    for lo in np.arange(.5,8.,.25):
        f,p,power=spectrum(x,float(lo),float(lo+.25));baseline.append(power);psd.append(p)
    bp=np.mean(baseline,axis=0);basepsd=np.mean(psd,axis=0)
    baseline_rate=ecount[500:8000].sum(0)/nc/7.5
    powers=[];rates=[];irates=[];window_psd=[];means=[]
    for lo,hi in WINDOWS:
        f,p,power=spectrum(x,lo,hi);powers.append(power);window_psd.append(p)
        rates.append(ecount[round(lo*1000):round(hi*1000)].sum(0)/nc/(hi-lo))
        irates.append(np.divide(icount[round(lo*1000):round(hi*1000)].sum(0),nic*(hi-lo),out=np.full(1600,np.nan),where=nic>0))
        means.append(x[round(lo*FS):round(hi*FS)].mean(0))
    powers=np.array(powers);rates=np.array(rates);irates=np.array(irates)
    assert np.all(bp>0) and np.all(powers>0)
    db=10*np.log10(powers/bp[None])
    # Keep a like-for-like 10.48-10.73 s native map for comparison with v10.
    _,_,earlypower=spectrum(x,10.48,10.73)
    earlydb=10*np.log10(earlypower/bp)
    earlyrate=ecount[10480:10730].sum(0)/nc/.25

    # Same event identities as v10, now native 0.5-mm cells, without electrodes.
    with (BASE/'early_spatial_v1/events.csv').open() as stream:events=list(csv.DictReader(stream))
    family_ranks=[];family_support=[]
    for family in ['A','B']:
        ranks=[]
        for event in events:
            if event['family']!=family:continue
            lo,hi=float(event['start_s']),float(event['end_s'])
            y=ecount[round(lo*1000):round(hi*1000)].astype(float);total=y.sum(0)
            peak=gaussian_filter1d(y/nc*1000,2,axis=0).max(0)
            valid=(total>=5)&(peak>=20)
            arrival=np.argmax(y.cumsum(0)>=.1*total,axis=0)
            rank=np.full(1600,np.nan)
            rank[valid]=(rankdata(arrival[valid])-1)/(valid.sum()-1)
            ranks.append(rank)
        ranks=np.array(ranks);support=np.isfinite(ranks).mean(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);template=np.nanmedian(ranks,axis=0)
        template[support<.5]=np.nan
        family_ranks.append(template);family_support.append(support)
    grid=(np.arange(40)+.5)*.5;xx,yy=np.meshgrid(grid,grid);xy=np.c_[xx.ravel(),yy.ravel()]
    centers=source['centers_mm']
    # Exact physical core membership supplies cell weights, rather than assuming
    # a boundary cell is wholly core or wholly surround.
    dist=np.linalg.norm(source['positions_e'][:,None]-centers[None],axis=2)
    region=np.full(len(dist),2);region[dist[:,0]<=1.5]=0;region[dist[:,1]<=1.5]=1
    weights=np.array([np.bincount(source['cell_e'][region==k],minlength=1600) for k in range(3)])
    records=[]
    for k,(lo,hi) in enumerate(WINDOWS):
        for j,name in enumerate(['Core A','Core B','Surround']):
            w=weights[j]
            records.append(dict(window_start_s=float(lo),window_end_s=float(hi),region=name,
                E_rate_Hz=float(np.average(rates[k],weights=w)),
                current_mean=float(np.average(means[k],weights=w)),
                band_20_150_dB=float(10*np.log10(np.average(powers[k,0],weights=w)/np.average(bp[0],weights=w)))))
    np.savez_compressed(OUT/'spatial_analysis.npz',windows_s=WINDOWS,bands_hz=np.array(BANDS),
        E_rate_hz=rates,I_rate_hz=irates,baseline_E_rate_hz=baseline_rate,
        band_power=powers,baseline_band_power=bp,band_change_dB=db,
        mean_current=np.array(means),baseline_mean_current=x[1000:16000].mean(0),
        early_band_change_dB=earlydb,early_E_rate_hz=earlyrate,
        frequency_hz=f,baseline_psd=basepsd,window_psd=np.array(window_psd),
        family_native_rank=np.array(family_ranks),family_participation=np.array(family_support),
        region_cell_weights=weights,cell_xy=xy,cell_n_E=nc,centers_mm=centers,
        contact_xy=source['contact_xy'])
    with (OUT/'regional_readouts.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    summary=dict(status='ANALYSIS_COMPLETE',windows_s=WINDOWS.tolist(),baseline_s=[.5,8.],baseline_windows=30,
        spatial_grid=[40,40],cell_width_mm=.5,primary_band_hz=[20,150],bands_hz=BANDS,
        current_observable='Each cell mean |AMPA|+|Z*GABA| current on E neurons; temporal power, no spatial readout kernel.',
        spectral_estimator='250-ms Hann periodogram, linear detrend removes DC/ramp, 4-Hz frequency spacing. Sum PSD*df over [low,high).',
        power_change='10 log10(window band power / average baseline band power), per native cell. Baseline includes all interictal activity, not selected quiet bins.',
        positive_power_fraction=(db[:,0]>0).mean(1).tolist(),
        native_band_change_minmax=[[float(v.min()),float(v.max())] for v in db[:,0]],
        global_E_rates=np.average(rates,weights=nc,axis=1).tolist(),regional_readouts=records,
        native_fidelity_checks=qa['checks'],
        spatial_smoothing=False, limitations='Power of a model current proxy, not a validated biophysical SEEG forward model. Field maps distinguish tonic elevation from band activity; changes need not have the same sign. Regions summarize their native cells, not exact per-region coherent power.')
    write('analysis_summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
