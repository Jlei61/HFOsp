#!/usr/bin/env python3
"""Auditable diagnostic plots, never a patient/ictal acceptance decision."""
from validate_topic4_fixed_rate_base import OUT, read, write
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


def render(name='prefix_7101'):
    snn = np.load(OUT / 'snn' / f'{name}.npz')
    dt = float(snn['dt_ms']); rate = snn['rate_e_hz']; t = np.arange(len(rate))*dt
    conditions = [('SNN', rate, snn['field_e_hz_20'], 'black')]
    for grid, closure, color in [(10,'legacy','#4477aa'), (10,'cascade','#ee7733'),
                                  (20,'cascade','#228833'), (20,'mesoscopic_seed9108001','#aa3377'),
                                  (20,'cascade_colored','#cc3311')]:
        path=OUT/'rate'/f'{name}_grid{grid}_{closure}.npz'
        if path.exists():
            a=np.load(path)
            conditions.append((f'{grid}×{grid} {closure.split("_seed")[0]}',a['rates_hz'][:,0],a['field_rates_hz'][:,0],color))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig=plt.figure(figsize=(13,8),layout='constrained');gs=fig.add_gridspec(2,3,height_ratios=[1,1.15])
    ax=fig.add_subplot(gs[0,:2]);bx=fig.add_subplot(gs[0,2])
    for label, values, field, color in conditions:
        y=uniform_filter1d(values.astype(float),100,mode='nearest')
        ax.plot(t,y,label=label,color=color,lw=1.7)
        bx.plot(t,np.maximum(y,1e-5),color=color,lw=1.3)
    ax.set(xlabel='Time (ms)',ylabel='Mean E rate (Hz)',title='Same frozen substrate and recorded external input')
    ax.set_ylim(-1, max(float(uniform_filter1d(rate.astype(float),100).max())*1.35,10))
    ax.legend(fontsize=9,ncol=2,loc='upper right');bx.set(yscale='log',xlabel='Time (ms)',ylabel='Mean E rate (Hz)',title='Log scale reveals low-state rates')
    peak=int(np.argmax(uniform_filter1d(rate.astype(float),100,mode='nearest')))
    frame=min(peak//20,len(snn['field_e_hz_20'])-1)
    picks=[conditions[0],conditions[2] if len(conditions)>2 else conditions[-1],conditions[-1]]
    vmax=max(1.,float(np.max(picks[0][2][frame])))
    axes=[]
    for j,(label,values,field,color) in enumerate(picks):
        axis=fig.add_subplot(gs[1,j]);axes.append(axis);grid=int(np.sqrt(field.shape[1]))
        im=axis.imshow(field[frame].reshape(grid,grid),origin='lower',extent=[0,20,0,20],vmin=0,vmax=vmax,cmap='magma')
        axis.set(title=f'{label} at {frame*2} ms',xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(im,ax=axes,label='Local E rate (Hz)',shrink=.8)
    fig.suptitle('Reduction validation: spontaneous short-prefix response\nCold start; Z/M off; this is not a seizure-transition assay',fontsize=15)
    dest=OUT/'figures';dest.mkdir(exist_ok=True);stem=dest/f'{name}_reduction_validation'
    fig.savefig(str(stem)+'.png',dpi=170);fig.savefig(str(stem)+'.pdf');plt.close(fig)
    rows=[]
    for label,values,field,color in conditions:
        y=uniform_filter1d(values.astype(float),100,mode='nearest')
        rows.append({'model':label,'full_window_mean_e_hz':float(np.mean(values)),
            'mean_e_hz_500_to_end':float(np.mean(values[int(500/dt):])),
            'max_10ms_mean_e_hz':float(y.max()),'peak_local_2ms_rate_hz':float(field.max())})
    write(stem.with_suffix('.json'),{'conditions':rows,'spatial_snapshot_time_ms':frame*2,
        'spatial_selection':'SNN maximum 10-ms mean in the full prefix, same timestamp for every model',
        'claim':'short-prefix dynamics only; event-free baseline and later stationary behavior not established',
        'local_time_sampling':'SNN averages spikes over 2-ms bins; rate samples instantaneous population rate every 2 ms. Not a precise latency comparison.',
        'author_visual_acceptance':False})
    path=dest/'README.md';old=path.read_text() if path.exists() else ''
    header='### '+stem.name+'.png'
    entry=header+'\n同一6101底物的短前缀SNN与rate近似比较，上排是10毫秒平滑的群体放电率及其对数尺度，下排在同一SNN峰值时刻、同一颜色尺度展示原生二维场。背景输入逐步取自真实SNN，包含全局与局部OU、裁剪及外源Poisson率；rate并未重放逐神经元spikes。SNN空间帧为2ms计数率，rate空间帧为每2ms瞬时率采样，不能据此报告精确传播时差；colored为开发修正，尚未通过网络验证。\n**关注点**：启动期短前缀中的事件与安静背景需分开，不能把整窗均值叫作安静基线，也不能将这张图当成长期多事件或发作证明。\n'
    if header in old:
        start=old.index(header);end=old.find('\n### ',start+len(header));end=len(old) if end<0 else end
        old=old[:start]+entry+old[end:]
    else:old+='\n'+entry
    path.write_text(old)
    print(rows)


if __name__=='__main__':render()
