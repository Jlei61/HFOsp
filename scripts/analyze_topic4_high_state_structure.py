#!/usr/bin/env python3
"""Separate sustained rate plateaus from spike periodicity and finite events."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch
import analyze_topic4_autonomous_recovery as a


def main():
    root=a.OUT
    cases=[('Finite events',root,'native_weak_s9108401',.5,5.5),
           ('Native high',root,'native_weak_s9108401',55.,60.),
           ('Fast threshold',root/'fast_threshold_round2','phi2.5_g0.5_s9108401',35.,40.),
           ('Preserved Z',root/'continuous_resource_recovery_round5','resource_rho0.25_k50_s9108401',15.,20.)]
    plt.rcParams.update({'font.size':15,'axes.labelsize':17,'xtick.labelsize':13,'ytick.labelsize':13})
    fig,axes=plt.subplots(4,4,figsize=(23,15),gridspec_kw={'hspace':.40,'wspace':.30})
    rows=[]
    for col,(label,source,name,lo,hi) in enumerate(cases):
        folder=source/'runs'/name;job=a.read(source/'jobs'/(name+'.json'))
        data=a.load(folder,['time_ms','spikes_1ms','regions_1ms','raster'])
        assert data['time_ms'][-1]/1000>=hi-.001,(name,hi)
        geo=np.load(source/'geometry.npz');nr=geo['region_counts'][:3]
        rates=np.column_stack([data['spikes_1ms'][:,0]/32,
                               data['regions_1ms'][:,:3]*1000/nr])
        t=data['time_ms']/1000
        use=(t>=lo)&(t<hi);r=rates[use]
        it,ix=np.nonzero(data['raster'][round((hi-.3)*10000):round(hi*10000),:60])
        for low,high,color in [(0,20,'#b66296'),(20,40,'#3889b0'),(40,60,'#28536c')]:
            keep=(ix>=low)&(ix<high)
            axes[0,col].scatter(it[keep]*.0001,ix[keep],s=4,marker='|',c=color,lw=.55,rasterized=True)
        axes[0,col].set(xlim=(0,.3),ylim=(-1,60),yticks=[9.5,29.5,49.5],
                        yticklabels=['Core A E','Core B E','Other E'],xlabel='Time within zoom (s)')
        axes[0,col].set_title(label,pad=15,fontsize=19)
        spectra=[]
        for i,(color,region) in enumerate([('#333333','All E'),('#b66296','Core A'),('#3889b0','Core B')]):
            axes[1,col].plot(t[use]-lo,gaussian_filter1d(r[:,i],3),c=color,lw=.75,label=region)
            f,p=welch(r[:,i],fs=1000,nperseg=2000,detrend='constant')
            normalized=p/max(float(r[:,i].mean())**2,1e-12)
            axes[2,col].loglog(f[1:],np.maximum(normalized[1:],1e-15),c=color,lw=.9)
            bands={}
            for start,end in [(.5,30),(30,300),(300,500)]:
                q=(f>=start)&(f<end if end<500 else f<=end)
                bands[f'{start:g}-{end:g}Hz']=float(p[q].sum()*(f[1]-f[0]))
            spectra.append(dict(region=region,mean_Hz=float(r[:,i].mean()),
                count_rate_CV=float(r[:,i].std()/max(r[:,i].mean(),1e-12)),
                modulation_20ms_CV=float(r[:,i].reshape(-1,20).mean(1).std()/max(r[:,i].mean(),1e-12)),
                power_by_band_Hz2=bands,largest_non_DC_peak_Hz=float(f[1:][np.argmax(p[1:])]),
                quiet10ms_fraction=float((r[:,i].reshape(-1,10).mean(1)<5).mean())))
        axes[1,col].set(xlim=(0,hi-lo),ylim=(-5,505),xlabel='Time within window (s)')
        if col==0:axes[1,col].legend(loc='upper right',fontsize=11,framealpha=.9)
        axes[2,col].set(xlim=(.5,500),ylim=(1e-10,10),xlabel='Frequency (Hz)')
        raster=data['raster'][round(lo*10000):round(hi*10000),:60]
        isi=[];neuron_cv=[]
        for neuron in range(60):
            x=np.diff(np.flatnonzero(raster[:,neuron]))*.1
            if len(x):isi.extend(x)
            if len(x)>=2:neuron_cv.append(float(x.std()/x.mean()))
        isi=np.asarray(isi)
        if len(isi):
            ordered=np.sort(isi)
            axes[3,col].semilogx(ordered,np.arange(1,len(ordered)+1)/len(ordered),c='#3d5365',lw=1.5)
        axes[3,col].set(xlim=(1,2000),ylim=(-.02,1.02),xlabel='Sampled E ISI (ms)')
        rows.append(dict(label=label,source=str(folder),job=job,analysis_window_s=[lo,hi],
            raster_zoom_s=[hi-.3,hi],source_run_complete=(folder/'result.json').exists(),
            population_spectra=spectra,sampled_E_cells=60,available_ISIs=len(isi),
            silent_sampled_E_cells=int(np.sum(~raster.any(0))),
            sampled_E_cells_with_at_least_two_spikes=int(np.sum(raster.sum(0)>=2)),
            sampled_ISI_quantiles_ms=np.quantile(isi,[.1,.5,.9]).tolist() if len(isi) else None,
            sampled_fraction_ISI_above20ms=float(np.mean(isi>20)) if len(isi) else None,
            median_per_sampled_neuron_ISI_CV=float(np.median(neuron_cv)) if neuron_cv else None))
        for ax in axes[:,col]:ax.spines[['top','right']].set_visible(False)
    for i,label in enumerate(['Fixed sampled E cells','Rate (Hz)','PSD / mean rate² (s)','ISI cumulative fraction']):
        axes[i,0].set_ylabel(label)
        axes[i,0].text(-.28,1.06,'ABCD'[i],transform=axes[i,0].transAxes,fontsize=23,weight='bold')
    fig.subplots_adjust(left=.08,right=.98,top=.95,bottom=.075)
    out=root/'high_state_structure';figdir=out/'figures';figdir.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(figdir/f'high_state_structure.{ext}',dpi=170)
    plt.close(fig)
    a.write(out/'observed_structure.json',dict(updated_at=time.time(),rows=rows,
        spectra='Welch of actual1ms population counts (all cells in each indicated population), mean removed,2000-sample windows,0.5Hz resolution,500Hz Nyquist. Frequency bands summarize continuous power and are not seizure or limit-cycle acceptance thresholds.',
        raster='Unchanged60 preselected E cells (20 per region), native0.1ms spikes. ISI distribution is descriptive for these sampled neurons; neither intervals nor neurons are independent network replicates.',
        caution='A refractory-scale spike train can create fast population spectral power without recovered finite events or a network seizure oscillation. A small all-E CV may also hide persistent core activity. Neither spectra nor ISIs classify Hopf, stability, or attractors;1ms counts cannot exclude oscillations above500Hz.',
        human_review='PENDING'))
    (figdir/'README.md').write_text('### high_state_structure.png / .pdf\n以相同观测方式比较原生有限事件、原生高态、快速阈值分支高态和保留Z分支的局部持续态：固定E采样raster、全E与双核率、1ms原生计数频谱及采样神经元ISI分布。窗口与参数完整保留在同级JSON，频谱分段只作描述而非发作或振荡验收阈值。**关注点**：不应将不应期量级的规则脉冲、持续低全网均率下的高core率，或一点高频纹波直接称为恢复的间期事件或Hopf振荡。\n')
    print([(r['label'],[(v['region'],round(v['mean_Hz'],2),round(v['modulation_20ms_CV'],4)) for v in r['population_spectra']],r['sampled_ISI_quantiles_ms']) for r in rows])


if __name__=='__main__':main()
