#!/usr/bin/env python3
"""Native spike/rate audit of the recurrent-high case, with fixed stage windows."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import numpy as np
from scipy.signal import periodogram
import plot_topic4_m_parameter_modes as f
import matplotlib.pyplot as plt

WINDOW = f.ROOT / 'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913'
SOURCE = WINDOW / 'early_Z_lookup_dense_figures/runs/early_z_refill_s9108401'
OUT = WINDOW / 'recurrent_high_native_rhythm'


def spectral(signal, fs):
    mean = np.asarray(signal).mean(axis=0)
    freq, power = periodogram(signal, fs=fs, window='hann', detrend='linear', axis=0)
    band = (freq >= 1) & (freq <= 150)
    relative = np.sqrt(power[band].sum(axis=0) * (freq[1] - freq[0])) / np.maximum(mean, 1e-12)
    return freq, power, mean, relative


def cell_regularity(raster):
    rows = []
    for lo, hi, label in [(0,20,'Core A E'),(20,40,'Core B E'),(40,60,'Other E'),(60,80,'I')]:
        values = []
        for col in range(lo, hi):
            isi_ms = np.diff(np.flatnonzero(raster[:,col])) * .1
            if len(isi_ms) >= 3:
                values.append((float(isi_ms.std()/isi_ms.mean()),float(np.median(isi_ms))))
        a = np.asarray(values)
        rows.append(dict(region=label,sampled_cells=hi-lo,eligible_cells=len(values),
            median_ISI_CV=float(np.median(a[:,0])) if len(a) else None,
            median_ISI_ms=float(np.median(a[:,1])) if len(a) else None))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    blob = (SOURCE/'checkpoint.pkl').read_bytes()
    saved = pickle.loads(blob)
    tr, end = saved['tracker'], saved['engine']['step']
    assert saved['job']['eta_m'] == .005 and saved['job']['tau_M_s'] == 1.
    assert len(tr['entries']) >= 2 and tr['recoveries']
    windows = [('Finite events',1.,2.),
        ('First high',tr['restore_s']-1,tr['restore_s']),
        ('After Z refill',tr['recoveries'][0]['confirmation_s']+4,tr['recoveries'][0]['confirmation_s']+5),
        ('Second high',tr['entries'][1]['onset_s']+8,tr['entries'][1]['onset_s']+9)]
    design = dict(windows=[dict(label=n,start_s=lo,end_s=hi) for n,lo,hi in windows],
        selection='One-second windows fixed by protocol/stage timing, not selected on spectral power.',
        population_rate_bin_ms=1.,sampled_neuron_spike_bin_ms=.1,
        spectral_estimator='One-second Hann periodogram with linear detrending; no added oscillation threshold.',
        band_Hz=[1,150],sampled_population_units='Same fixed60 E and20 I neurons at both temporal resolutions.',
        sampling='20 Core A E,20 Core B E,20 Other E,20 I; stratified display sample, not proportional to the full network.',
        spatial_observable='Power of recorded1ms binned spike counts. Above-Nyquist harmonics may alias; not an exact continuous-time rate spectrum.',
        new_simulations=0,new_independent_samples=0,source=str(SOURCE))
    f.write(OUT/'design.json',design)
    if end*.0001 < windows[-1][2]:
        print(json.dumps(dict(status='WAITING_FOR_FIXED_WINDOW',closed_end_s=end*.0001,
                             required_end_s=windows[-1][2])));return
    a = f.load(SOURCE,end_step=end)
    ne,ni = 32000,8000
    with np.load(f.OUT/'geometry.npz') as g:cell_counts=g['cell_e_counts'].copy()
    rows=[];plot_data=[]
    for label,lo,hi in windows:
        one=(a['time_ms']/1000>=lo)&(a['time_ms']/1000<hi)
        rate=a['spikes_1ms'][one]/np.array([ne,ni])/.001
        assert len(rate)==1000
        raster=a['raster'][round(lo*10000):round(hi*10000)]
        assert raster.shape==(10000,80)
        sampled=np.c_[raster[:,:60].sum(1)/60/.0001,raster[:,60:].sum(1)/20/.0001]
        binned=sampled.reshape(1000,10,2).mean(1)
        freq,power,mean,relative=spectral(sampled,10000)
        bf,bp,bmean,brelative=spectral(binned,1000)
        pf,pp,pmean,prelative=spectral(rate,1000)
        field=a['field_1ms'][one]/np.maximum(cell_counts,1)/.001
        _,_,fm,fr=spectral(field,1000);valid=(cell_counts>0)&(fm>=1)
        zt=a['slow_time_ms']/1000;zmask=(zt>=lo)&(zt<hi)
        rows.append(dict(label=label,window_s=[lo,hi],all_population_mean_Hz=pmean.tolist(),
            all_population_1ms_relative_band_RMS=prelative.tolist(),
            sampled_0p1ms_mean_Hz=mean.tolist(),sampled_0p1ms_relative_band_RMS=relative.tolist(),
            same_sampled_1ms_relative_band_RMS=brelative.tolist(),
            sampled_resolution_band_RMS_ratio=(brelative/np.maximum(relative,1e-12)).tolist(),
            sampled_cell_regularity=cell_regularity(raster),
            native_1ms_field_valid_cells=int(valid.sum()),
            native_1ms_field_relative_band_RMS_quantiles=np.quantile(fr[valid],[.1,.5,.9]).tolist(),
            mean_Z=float(a['Z'][zmask,0].mean()),mean_M_current=float(.005*a['M'][zmask,0].mean())))
        plot_data.append((rate,raster,freq,power,bf,bp,mean))
    report=dict(status='COMPLETE_FIXED_WINDOW_AUDIT',design=design,rows=rows,
        source_checkpoint_step=end,source_checkpoint_sha256=hashlib.sha256(blob).hexdigest(),
        planned_full_followup_completed=(SOURCE/'result.json').exists(),
        low_band_power_is_not_limit_cycle_evidence=True,single_cell_regularity_is_not_population_synchrony=True,
        human_review='PENDING',agent_visual_review='PENDING')
    f.write(OUT/'analysis.json',report)
    plt.rcParams.update({'font.size':13,'axes.labelsize':15,'axes.titlesize':15,'pdf.fonttype':42})
    fig,axes=plt.subplots(3,4,figsize=(19,11),gridspec_kw={'height_ratios':[1,1.3,1.15]})
    maximum=max(x[0].max() for x in plot_data)*1.04
    for col,((label,lo,hi),data) in enumerate(zip(windows,plot_data)):
        rate,raster,freq,power,bf,bp,mean=data
        top,ras,spec=axes[:,col]
        top.plot((np.arange(1000)+.5)/1000,rate[:,0],c='#287da8',lw=.8,label='All E')
        top.plot((np.arange(1000)+.5)/1000,rate[:,1],c='#c87535',lw=.8,label='All I')
        top.set(xlim=[0,1],ylim=[0,maximum],title=f'{label}\n{lo:.2f}–{hi:.2f} s')
        if col==0:top.set_ylabel('Population rate (Hz)');top.legend(frameon=False,fontsize=10)
        for low,high,color in [(0,60,'#287da8'),(60,80,'#c87535')]:
            rr,cc=np.where(raster[:,low:high]);ras.scatter(rr*.0001,cc+low,s=.7,c=color,lw=0,rasterized=True)
        ras.set(xlim=[0,1],ylim=[-1,80],yticks=[10,30,50,70],
            yticklabels=['Core A E','Core B E','Other E','I'] if col==0 else [],xlabel='Window time (s)')
        for y in [19.5,39.5,59.5]:ras.axhline(y,color='#cccccc',lw=.5)
        for k,color in enumerate(['#287da8','#c87535']):
            ok=(freq>=1)&(freq<=2000);spec.loglog(freq[ok],np.maximum(power[ok,k]/mean[k]**2,1e-16),c=color,lw=1)
            ok=bf>=1;spec.loglog(bf[ok],np.maximum(bp[ok,k]/mean[k]**2,1e-16),c=color,lw=.7,ls='--',alpha=.6)
        spec.axvspan(1,150,color='#bbbbaa',alpha=.16);spec.axvline(500,color='#777777',ls=':',lw=.7)
        spec.set(xlim=[1,2000],ylim=[1e-12,10],xlabel='Frequency (Hz)',xticks=[1,10,100,1000])
        if col==0:spec.set_ylabel('Sampled rate PSD / mean² (1/Hz)')
    fig.subplots_adjust(left=.07,right=.98,top=.94,bottom=.12,wspace=.24,hspace=.35)
    fig.text(.5,.025,'Solid: same sampled neurons at 0.1 ms; dashed: their 1-ms bins. Shading: 1–150 Hz. Dotted line: 1-ms Nyquist.',ha='center',fontsize=12)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'native_rhythm_and_sampling.png',dpi=170)
    fig.savefig(folder/'native_rhythm_and_sampling.pdf');plt.close(fig)
    (folder/'README.md').write_text('### native_rhythm_and_sampling.png / .pdf\n'
        '用预先确定的四个1秒窗口对照有限事件、第一次高率、补Z后事件及第二次高率。上排为全部E/I的1ms计数率，中排为同一80个神经元的真实0.1ms raster，下排只比较同一批采样神经元在0.1ms和1ms时分辨率的频谱。\n'
        '**关注点**：逐神经元快速周期放电、低频群体包络及采样混叠必须分开；区域分层抽样并非全网比例，分辨率比较不把60个E或20个I冒充全部神经元，也不将任意功率峰称为极限环。\n')
    print(json.dumps(dict(status=report['status'],rows=[{k:r[k] for k in
        ['label','all_population_mean_Hz','sampled_0p1ms_relative_band_RMS','sampled_resolution_band_RMS_ratio']} for r in rows],
        figure=str(folder/'native_rhythm_and_sampling.png'))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    SOURCE=args.source.resolve();OUT=args.output.resolve()
    assert SOURCE.is_relative_to(WINDOW.resolve()) and OUT.is_relative_to(WINDOW.resolve())
    main()
