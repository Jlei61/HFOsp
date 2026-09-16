#!/usr/bin/env python3
"""One continuous native SNN clip: field, all-cell rates, sampled raster.

No electrode interpolation, event warping, spike sorting by observed response,
or simulated-state change is used to make the movie.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.animation import FuncAnimation,PillowWriter
from scipy.ndimage import gaussian_filter1d
import analyze_topic4_autonomous_recovery as common

def main(root,name,start,end,label):
    folder=root/'runs'/name
    a=common.load(folder,['time_ms','spikes_1ms','field_5ms','raster'])
    geo=np.load(root/'geometry.npz')
    if start<.02 or end>len(a['spikes_1ms'])/1000-.02 or end<=start:raise ValueError('Clip must have a complete native20ms field window')
    times=np.arange(round(start*100),round(end*100)+1)/100
    fields=[]
    for tm in times:
        lo=round((tm-.01)/.005)
        fields.append(a['field_5ms'][lo:lo+4].sum(0)/geo['cell_e_counts']/.02)
    fields=np.asarray(fields)
    fig=plt.figure(figsize=(13,6));gs=fig.add_gridspec(2,2,width_ratios=[1,1.6],left=.06,right=.96,bottom=.12,top=.90,wspace=.32,hspace=.30)
    ax=fig.add_subplot(gs[:,0]);im=ax.imshow(fields[0].reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',norm=PowerNorm(.6,0,500),interpolation='nearest')
    for core,center in zip('AB',geo['centers_mm']):
        ax.add_patch(plt.Circle(center,float(geo['core_radius_mm']),fill=False,ec='#41c6c9',lw=1.5))
        ax.text(center[0],center[1]+2.0,core,color='#2e969b',weight='bold',ha='center',bbox=dict(fc='white',ec='none',alpha=.8,pad=.2))
    ax.set(xlabel='x (mm)',ylabel='y (mm)',xticks=[0,10,20],yticks=[0,10,20])
    cb=fig.colorbar(im,ax=ax,shrink=.6,pad=.03,ticks=[0,250,500]);cb.set_label('E rate (Hz)')
    rx=fig.add_subplot(gs[0,1]);t=a['time_ms']/1000;take=(t>=start)&(t<=end)
    rate=gaussian_filter1d(a['spikes_1ms'].astype(float)/np.array([32000,8000])*1000,2,axis=0)
    rx.plot(t[take],rate[take,0],c='#347da0',lw=1,label='All E');rx.plot(t[take],rate[take,1],c='#bf7a39',lw=.9,label='All I')
    rx.set(xlim=(start,end),ylabel='Rate (Hz)');rx.tick_params(labelbottom=False);rx.legend(frameon=False,loc='upper right')
    cursor1=rx.axvline(times[0],color='#a8304c',lw=1.3)
    sx=fig.add_subplot(gs[1,1],sharex=rx)
    first,last=round(start*10000),round(end*10000)
    it,ix=np.nonzero(a['raster'][first:last]);it=(it+first)*.0001
    sx.scatter(it,ix,s=7,marker='|',c=np.where(ix<60,'#347da0','#bf7a39'),linewidths=.7,rasterized=True)
    for y in [19.5,39.5,59.5]:sx.axhline(y,c='.7',lw=.6)
    sx.set(ylim=(-1,80),yticks=[9.5,29.5,49.5,69.5],yticklabels=['Core A E','Core B E','Other E','I'],xlabel='Time (s)')
    cursor2=sx.axvline(times[0],color='#a8304c',lw=1.3)
    stamp=fig.text(.07,.96,f'Time = {times[0]:.3f} s',fontsize=17,weight='bold')
    fig.text(.65,.96,'Native 20-ms activity · playback 5× slower',fontsize=12,ha='center')
    for item in fig.axes:item.tick_params(labelsize=11);item.xaxis.label.set_fontsize(13);item.yaxis.label.set_fontsize(13)
    def update(k):
        im.set_data(fields[k].reshape(20,20));cursor1.set_xdata([times[k],times[k]]);cursor2.set_xdata([times[k],times[k]])
        stamp.set_text(f'Time = {times[k]:.3f} s');return im,cursor1,cursor2,stamp
    out=folder/'native_movies';out.mkdir(exist_ok=True)
    movie=FuncAnimation(fig,update,frames=len(times),interval=50,blit=False)
    movie.save(out/f'{label}.gif',writer=PillowWriter(fps=20),dpi=105)
    plt.close(fig)
    # Fixed uniform time samples for reviewing what the animation actually shows.
    selected=np.linspace(0,len(times)-1,12).round().astype(int)
    fig,axs=plt.subplots(3,4,figsize=(12,9))
    for k,ax in zip(selected,axs.flat):
        ax.imshow(fields[k].reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',norm=PowerNorm(.6,0,500),interpolation='nearest')
        for center in geo['centers_mm']:ax.add_patch(plt.Circle(center,float(geo['core_radius_mm']),fill=False,ec='#41c6c9',lw=1))
        ax.set_title(f'{times[k]:.3f} s');ax.set(xticks=[0,10,20],yticks=[0,10,20])
    fig.tight_layout();fig.savefig(out/f'{label}_contact_sheet.png',dpi=140);plt.close(fig)
    metadata=dict(source=str(folder),clip_window_s=[start,end],frame_step_s=.01,field_window_s=.02,playback_fps=20,
        activity='Actual E spike counts divided by each1mm cell population and20ms, no electrode projection.',
        raster='Same fixed80 neuron sample and same saved row order; core rows E only, I separately shown.',
        display='PowerNorm gamma.6 at0–500Hz, nearest-neighbor display; fields are not spatially interpolated.',
        frames=len(times),uniform_contact_sheet_times_s=times[selected].tolist(),human_review='PENDING')
    (out/f'{label}_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with (out/'README.md').open('a') as f:f.write(f'\n### {label}.gif\n同一连续SNN轨迹{start:g}–{end:g}秒的原生空间场、全E/I放电率与固定样本raster，红线同步标出帧时刻。20ms细胞计数窗每10ms移动，播放慢5倍，不经电极投影或按发放顺序重排神经元。**关注点**：逐次事件是依次传播还是同时全局招募，能否自行结束，以及安静间隔是否真实存在。\n\n### {label}_contact_sheet.png\n动画中均匀选取的12帧，使用共同0–500Hz色标，便于核对实际时间与空间招募。**关注点**：均匀选帧可能错过短事件峰，完整过程以动画和原生计数为准。\n')
    print(out/f'{label}.gif',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=common.OUT);p.add_argument('--name',required=True)
    p.add_argument('--start',type=float,required=True);p.add_argument('--end',type=float,required=True);p.add_argument('--label',default='native_clip')
    a=p.parse_args();main(a.root,a.name,a.start,a.end,a.label)
