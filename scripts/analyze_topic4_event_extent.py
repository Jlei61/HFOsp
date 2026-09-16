#!/usr/bin/env python3
"""Native spatial extent and quiet gaps beyond the fixed tonic-entry gate.

This observer does not relabel bursts as seizures. It exposes localized finite
events, widespread burst trains, and persistent high activity for direct review.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import analyze_topic4_autonomous_recovery as common

def episodes(rate):
    edges=np.diff(np.r_[False,rate<5,False].astype(int))
    quiet=[(lo,hi) for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if hi-lo>=3]
    bounds=[(0,0)]+quiet+[(len(rate),len(rate))]
    out=[]
    for i,(left,right) in enumerate(zip(bounds[:-1],bounds[1:])):
        lo,hi=left[1],right[0]
        if hi<=lo or rate[lo:hi].max()<20:continue
        out.append(dict(start_s=float(lo*.01),end_s=float(hi*.01),duration_s=float((hi-lo)*.01),
            preceding_quiet_s=float((left[1]-left[0])*.01),following_quiet_s=float((right[1]-right[0])*.01),
            left_bounded=i>0,right_bounded=i<len(bounds)-2,
            finite_event=bool(i>0 and i<len(bounds)-2 and hi-lo<=30),
            peak_time_s=float((lo+np.argmax(rate[lo:hi])+.5)*.01),peak_all_E_Hz=float(rate[lo:hi].max())))
    return out

def main(root,name=None):
    geo=np.load(root/'geometry.npz');nc=geo['cell_e_counts'];nr=geo['region_counts'];allrows=[]
    folders=[root/'runs'/name] if name else sorted((root/'runs').iterdir())
    for folder in folders:
        if folder.name.startswith('qa_') or not (folder/'chunks').exists():continue
        a=common.load(folder,['time_ms','spikes_1ms','regions_1ms','field_5ms','field_time_ms'])
        if a is None:continue
        n=len(a['spikes_1ms'])//10
        r=a['spikes_1ms'][:n*10,0].reshape(n,10).sum(1)/32000/.01
        field=a['field_5ms'].astype(float)
        assert np.array_equal(field.sum(0).sum(),a['spikes_1ms'][:,0].sum())
        prefix=np.cumsum(np.pad(field,((1,0),(0,0))),axis=0)
        r20=(prefix[4:]-prefix[:-4])/nc/.02
        ft=a['field_time_ms'][3:]/1000-.0075
        recruited=r20>=20;instant=recruited.mean(1);rows=episodes(r)
        for e in rows:
            selected=(ft>=e['start_s'])&(ft<e['end_s'])
            if not selected.any():selected[np.argmin(abs(ft-e['peak_time_s']))]=True
            union=float(recruited[selected].any(0).mean());simultaneous=float(instant[selected].max())
            assert 0<=simultaneous<=union<=1
            sensitivity={}
            for threshold in [20,50,100]:
                local=r20[selected]>=threshold
                sensitivity[str(threshold)]=dict(union_area=float(local.any(0).mean()),peak_simultaneous_area=float(local.mean(1).max()))
            lo,hi=round(e['start_s']*1000),round(e['end_s']*1000)
            count=a['regions_1ms'][lo:hi,:3].sum(0)
            e.update(native_recruited_area_union=union,native_peak_simultaneous_area=simultaneous,
                recruitment_threshold_sensitivity_Hz=sensitivity,mean_spikes_per_neuron_A_B_surround=(count/nr[:3]).tolist())
        finite=[e for e in rows if e['finite_event']]
        broad=[e for e in finite if e['native_recruited_area_union']>=.5]
        record=dict(name=folder.name,observed_s=n*.01,episodes=rows,finite_episode_count=len(finite),
            broad_finite_episode_count=len(broad),
            median_finite_union_area=float(np.median([e['native_recruited_area_union'] for e in finite])) if finite else None,
            interpretation='Broad finite episodes are a spatial diagnostic, not seizure labels. A train of such waves needs separate temporal and local-rate review; no entry/recovery threshold is changed.')
        common.write(folder/'event_extent_audit.json',record);allrows.append(record)
        if name:
            fig,axs=plt.subplots(3,1,figsize=(15,10),sharex=True,gridspec_kw={'hspace':.25})
            t=a['time_ms']/1000;rate=gaussian_filter1d(a['spikes_1ms'][:,0]*1000/32000,3)
            axs[0].plot(t,rate,c='#333333',lw=.8);axs[0].axhline(200,c='.6',ls=':',lw=.8);axs[0].set_ylabel('All E rate (Hz)')
            axs[1].plot(ft,instant,c='#4b8e99',lw=.8,label='Simultaneous area')
            for e in rows:axs[1].scatter(e['peak_time_s'],e['native_recruited_area_union'],s=30,c='#ba6851' if e['finite_event'] else '#555555',zorder=3)
            axs[1].set_ylabel('Recruited area\n(fraction)');axs[1].set_ylim(-.03,1.03);axs[1].legend(frameon=False,loc='upper right')
            for e in rows:
                if e['left_bounded']:axs[2].scatter(e['start_s'],e['preceding_quiet_s'],c='#4c7d9c',s=30)
            axs[2].set_ylabel('Preceding quiet\ninterval (s)');axs[2].set_xlabel('Time (s)')
            for i,ax in enumerate(axs):
                ax.set_xlim(0,n*.01);ax.tick_params(labelsize=16);ax.xaxis.label.set_fontsize(19);ax.yaxis.label.set_fontsize(19)
                ax.text(-.10,1.02,'ABC'[i],transform=ax.transAxes,weight='bold',fontsize=23)
            fig.subplots_adjust(left=.13,right=.97,top=.96,bottom=.10)
            d=folder/'event_extent/figures';d.mkdir(parents=True,exist_ok=True)
            for ext in ['png','pdf']:fig.savefig(d/f'native_event_extent.{ext}',dpi=170)
            plt.close(fig)
            (d/'README.md').write_text('### native_event_extent.png / .pdf\n连续全E放电与原生20ms网格招募面积对齐；中图曲线为同时激活面积，点为单个活动段内曾被招募的面积，末行是此前真实安静间隔。红点为满足原有限事件时长及双侧安静要求的事件，灰点为较长或边界不完整的活动段；这里的面积诊断没有修改高态或恢复判据。**关注点**：广泛的短爆发不能因未达200ms高态门而被忽略，也不能仅凭空间范围就归为持续发作或自主恢复。\n')
    output=folder/'event_extent_detailed_method.json' if name else root/'event_extent_audit.json'
    common.write(output,dict(updated_at=time.time(),rows=allrows,
        definitions='All-E10ms episodes separated by>=30ms<5Hz, peak>=20Hz. Finite events additionally<=300ms and bounded on both sides. Native1mm E grids use20ms windows every5ms, local>=20Hz for recruitment; union and simultaneous area reported separately with50/100Hz sensitivity. Broad diagnostic uses union>=.5; no seizure relabeling.',
        statistical_unit='One fixed parameter/topology/noise trajectory; events and spatial cells are not independent network replicates.'))
    print([(v['name'],v['finite_episode_count'],v['broad_finite_episode_count'],v['median_finite_union_area']) for v in allrows])

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=common.OUT);a.add_argument('--name');v=a.parse_args();main(v.root,v.name)
