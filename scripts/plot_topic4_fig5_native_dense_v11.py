#!/usr/bin/env python3
"""Fig5 with a direct native transition field and the simulated dense Z grid."""
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation
import plot_topic4_fig5_spatial_latency_layout_v10 as prior

ROOT=prior.ROOT;BASE=prior.BASE;OUT=BASE/'layout_v11';FIG=OUT/'figures'
NATIVE=BASE/'native_transition_v2';DENSE=BASE/'latency_dense_v1'
RATE_CMAP=prior.previous.previous.prior.old.ACTIVITY_CMAP
PREVIEW_README='''### native_rate_preview_1mm.png / .pdf
直接使用原连续SNN已保存的1毫米网格spikes，展示10.25–11.00秒的三个连续250毫秒窗。色标为全E神经元局部放电率，core边界为实际1.5毫米半径，没有经过电极投影或空间插值。
**关注点**：这是原生场记录完成前的直接放电预览；放电率图不替代新增0.5毫米场的频带功率分析。

'''
LAYOUT_PREVIEW_README='''
### fig5_manual_core_release_layout_v11_progress.png / .pdf
完整排版的过程预览，原生E2已完成，F仅对三条种子都完成的节点着色。未完成格子保持空白，不用插值或已有种子平均填补。
**关注点**：此文件不是完整参数结果，完成数见preview_metadata.json；整轮完成后另行生成不带progress的版本。
'''


def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.14)
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.14)
    plt.close(fig)


def write(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def rate_preview():
    a=np.load(BASE/'runs/continuous_refill_release.npz')
    windows=np.array([[10.25,10.5],[10.5,10.75],[10.75,11.0]])
    counts=a['field_e_count_1ms'];n=a['cell_e_counts']
    fig=plt.figure(figsize=(10,3.5));gs=fig.add_gridspec(1,4,width_ratios=[1,1,1,.045],wspace=.20)
    for k,(lo,hi) in enumerate(windows):
        rate=counts[round(lo*1000):round(hi*1000)].sum(0)/n/(hi-lo)
        ax=fig.add_subplot(gs[k]);im=ax.imshow(rate.reshape(20,20),origin='lower',extent=[0,20,0,20],
            cmap=RATE_CMAP,vmin=0,vmax=500,interpolation='nearest')
        prior.core_outlines(ax,a['centers_mm'],1.5,labels=k==0,color='#2bd3cf')
        ax.set(xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)');ax.tick_params(labelsize=11)
        if k==0:ax.set_ylabel('y (mm)')
        else:ax.set_yticklabels([])
        ax.set_title(f'{lo:.2f}–{hi:.2f} s',fontsize=13)
    cb=fig.colorbar(im,cax=fig.add_subplot(gs[3]),ticks=[0,250,500]);cb.set_label('E rate (Hz)',fontsize=12)
    save(fig,'native_rate_preview_1mm')
    readme=FIG/'README.md'
    if not readme.exists():readme.write_text(PREVIEW_README)
    elif '### native_rate_preview_1mm' not in readme.read_text():
        with readme.open('a') as stream:stream.write('\n'+PREVIEW_README)


def native_map(ax,v,a,norm,cmap,xlabel=True,ylabel=True,core_labels=False):
    im=ax.imshow(np.ma.masked_invalid(v.reshape(40,40)),origin='lower',extent=[0,20,0,20],
                 norm=norm,cmap=cmap,interpolation='nearest',rasterized=True)
    prior.core_outlines(ax,a['centers_mm'],1.5,labels=core_labels,color='#2bd3cf')
    ax.set(xlim=(0,20),ylim=(0,20),xticks=[0,10,20],yticks=[0,10,20])
    ax.set_xlabel('x (mm)' if xlabel else '',fontsize=12)
    ax.set_ylabel('y (mm)' if ylabel else '',fontsize=12)
    if not xlabel:ax.set_xticklabels([])
    if not ylabel:ax.set_yticklabels([])
    ax.tick_params(labelsize=11)
    return im


def native_transition(fig,spec,a,title='E2  Early recruitment'):
    outer=spec.subgridspec(2,1,height_ratios=[.065,1],hspace=.10)
    head=fig.add_subplot(outer[0]);head.axis('off');head.text(0,.7,title,fontsize=16,weight='bold')
    gs=outer[1].subgridspec(2,4,width_ratios=[1,1,1,.045],wspace=.18,hspace=.22)
    db=a['band_change_dB'][:,0]
    lim=max(5.,float(np.ceil(np.max(abs(db))/5)*5))
    bn=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim);rn=Normalize(0,500)
    for k,(lo,hi) in enumerate(a['windows_s']):
        ax=fig.add_subplot(gs[0,k]);native_map(ax,a['E_rate_hz'][k],a,rn,RATE_CMAP,xlabel=False,ylabel=k==0,core_labels=k==0)
        ax.set_title(f'{lo:.2f}–{hi:.2f} s',fontsize=12,pad=8)
        ax=fig.add_subplot(gs[1,k]);native_map(ax,db[k],a,bn,'RdBu_r',ylabel=k==0)
    cb=fig.colorbar(ScalarMappable(norm=rn,cmap=RATE_CMAP),cax=fig.add_subplot(gs[0,3]),ticks=[0,250,500])
    cb.set_label('E rate (Hz)',fontsize=12);cb.ax.tick_params(labelsize=10)
    cb=fig.colorbar(ScalarMappable(norm=bn,cmap='RdBu_r'),cax=fig.add_subplot(gs[1,3]),ticks=[-lim,0,lim])
    cb.set_label('20–150 Hz change (dB)',fontsize=12);cb.ax.tick_params(labelsize=10)
    return dict(window_s=a['windows_s'].tolist(),grid_shape=[40,40],cell_width_mm=.5,
                spatial_interpolation=False,E_rate_limits_Hz=[0,500],band_change_limits_dB=[-lim,lim])


def edges(x,log=False):
    v=np.log(x) if log else np.array(x)
    mid=(v[:-1]+v[1:])/2
    out=np.r_[v[0]-(mid[0]-v[0]),mid,v[-1]+(v[-1]-mid[-1])]
    return np.exp(out) if log else out


def latency(fig,spec,a,title='F  Z kinetics'):
    gs=spec.subgridspec(1,2,width_ratios=[1,.055],wspace=.13)
    ax=fig.add_subplot(gs[0]);xe=edges(a['tau_s'],True);ye=edges(a['threshold'])
    im=ax.pcolormesh(xe,ye,a['restricted_mean_s'],cmap='viridis',vmin=0,vmax=24,
                    edgecolors=(1,1,1,.30),linewidth=.4,rasterized=True)
    for y,x in zip(*np.where(a['transition_fraction']<1)):
        ax.add_patch(Rectangle((xe[x],ye[y]),xe[x+1]-xe[x],ye[y+1]-ye[y],
            fc='none',ec='#222222',hatch='///',lw=0))
    ax.plot(5.,95.19851312666987,marker='o',ms=7,mec='white',mfc='none',mew=1.5)
    ax.set_xscale('log',base=2);ax.set_xticks([2.5,5.,10.],labels=['2.5','5','10'])
    ax.set_yticks([75.,95.19851312666987,120.],labels=['75','95.2','120'])
    ax.set_xlabel(r'$\tau_Z$ (s)',fontsize=15);ax.set_ylabel('Depletion threshold $I_{th}$\n(mV equiv.)',fontsize=13)
    ax.tick_params(labelsize=12);ax.set_title(title,fontsize=16,weight='bold',loc='left',pad=12)
    cb=fig.colorbar(im,cax=fig.add_subplot(gs[1]),ticks=[0,6,12,18,24])
    cb.set_label('Restricted mean time (s)',fontsize=13);cb.ax.tick_params(labelsize=11)


def spectra(a):
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
    f=a['frequency_hz'];colors=['#d69524','#e26139','#a91e36']
    for j,(ax,name) in enumerate(zip(axes,['Core A','Core B','Surround'])):
        weights=a['region_cell_weights'][j]
        base=np.average(a['baseline_psd'],axis=1,weights=weights)
        ax.plot(f,base,c='black',lw=1.5,label='0.5–8 s')
        for k,(lo,hi) in enumerate(a['windows_s']):
            ax.plot(f,np.average(a['window_psd'][k],axis=1,weights=weights),c=colors[k],lw=1.2,label=f'{lo:.2f}–{hi:.2f} s')
        ax.set(xlim=(4,1000),xscale='log',yscale='log',xlabel='Frequency (Hz)',title=name)
        ax.set_xticks([10,30,100,300,1000],labels=['10','30','100','300','1000'])
        ax.axvline(20,color='black',lw=.6,ls=':');ax.axvline(150,color='black',lw=.6,ls=':')
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel('Local current PSD (mV²/Hz)');axes[-1].legend(frameon=False,fontsize=9)
    fig.tight_layout();save(fig,'native_local_spectra')


def ei_rates(a):
    fig=plt.figure(figsize=(11,7));gs=fig.add_gridspec(2,4,width_ratios=[1,1,1,.045],
        left=.065,right=.90,top=.92,bottom=.09,wspace=.18,hspace=.22)
    upper=max(500.,float(np.ceil(np.nanmax(a['I_rate_hz'])/100)*100))
    for row,(key,limit,label) in enumerate([('E_rate_hz',500.,'E rate (Hz)'),('I_rate_hz',upper,'I rate (Hz)')]):
        for k,(lo,hi) in enumerate(a['windows_s']):
            ax=fig.add_subplot(gs[row,k]);im=native_map(ax,a[key][k],a,Normalize(0,limit),RATE_CMAP,
                xlabel=row==1,ylabel=k==0,core_labels=(row==0 and k==0))
            if row==0:ax.set_title(f'{lo:.2f}–{hi:.2f} s',fontsize=12)
        cb=fig.colorbar(im,cax=fig.add_subplot(gs[row,3]),ticks=[0,limit/2,limit]);cb.set_label(label,fontsize=12)
    save(fig,'native_EI_rate_maps')


def family_comparison(a):
    fig=plt.figure(figsize=(17,3.8));gs=fig.add_gridspec(1,9,
        width_ratios=[1,1,.04,.18,1,.04,.18,1,.04],wspace=.30)
    for k,title in enumerate(['A-leading','B-leading']):
        ax=fig.add_subplot(gs[k]);im=native_map(ax,a['family_native_rank'][k],a,Normalize(0,1),'viridis',ylabel=k==0)
        ax.set_title(title,fontsize=13)
    cb=fig.colorbar(im,cax=fig.add_subplot(gs[2]),ticks=[0,1]);cb.ax.set_yticklabels(['Early','Late'])
    ax=fig.add_subplot(gs[4]);im=native_map(ax,a['early_E_rate_hz'],a,Normalize(0,500),RATE_CMAP,ylabel=False)
    ax.set_title('Early E rate',fontsize=13)
    cb=fig.colorbar(im,cax=fig.add_subplot(gs[5]),ticks=[0,250,500]);cb.set_label('Hz')
    ax=fig.add_subplot(gs[7]);db=a['early_band_change_dB'][0];lim=max(5.,np.ceil(abs(db).max()/5)*5)
    im=native_map(ax,db,a,TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),'RdBu_r',ylabel=False)
    ax.set_title('Early 20–150 Hz',fontsize=13)
    cb=fig.colorbar(im,cax=fig.add_subplot(gs[8]),ticks=[-lim,0,lim]);cb.set_label('dB')
    save(fig,'native_families_and_early_transition')


def frequency_comparison(a):
    fig=plt.figure(figsize=(11,7));gs=fig.add_gridspec(2,4,width_ratios=[1,1,1,.045],
        left=.065,right=.90,top=.92,bottom=.09,wspace=.18,hspace=.22)
    selected=[0,6]
    for row,band in enumerate(selected):
        db=a['band_change_dB'][:,band]
        limit=max(5.,float(np.ceil(abs(db).max()/5)*5))
        norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
        for k,(lo,hi) in enumerate(a['windows_s']):
            ax=fig.add_subplot(gs[row,k]);im=native_map(ax,db[k],a,norm,'RdBu_r',
                xlabel=row==1,ylabel=k==0,core_labels=(row==0 and k==0))
            if row==0:ax.set_title(f'{lo:.2f}–{hi:.2f} s',fontsize=12)
        cb=fig.colorbar(im,cax=fig.add_subplot(gs[row,3]),ticks=[-limit,0,limit])
        lo,hi=a['bands_hz'][band];cb.set_label(f'{lo}–{hi} Hz change (dB)',fontsize=12)
    save(fig,'native_frequency_comparison')


def broadband_comparison(a):
    source=NATIVE/'broadband_1s_diagnostic.npz'
    if not source.exists():return
    b=np.load(source)
    fig=plt.figure(figsize=(11,3.7));gs=fig.add_gridspec(1,4,width_ratios=[1,1,1,.045],
        left=.065,right=.90,top=.85,bottom=.17,wspace=.18)
    limit=max(5.,float(np.ceil(abs(b['band_change_dB']).max()/5)*5))
    norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    for k,(lo,hi) in enumerate(b['windows_s']):
        ax=fig.add_subplot(gs[k]);im=native_map(ax,b['band_change_dB'][k],a,norm,'RdBu_r',
            ylabel=k==0,core_labels=k==0)
        ax.set_title(f'{lo:.2f}–{hi:.2f} s',fontsize=12)
    cb=fig.colorbar(im,cax=fig.add_subplot(gs[3]),ticks=[-limit,0,limit])
    cb.set_label('1–150 Hz change (dB)',fontsize=12)
    save(fig,'native_broadband_1s_windows')


def depletion(a):
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for mode,ax in enumerate(axes):
        vals=a['tau_s'] if mode==0 else a['threshold']
        norm=Normalize(vals[0],vals[-1])
        for k,value in enumerate(vals):
            color=plt.get_cmap('viridis')(norm(value))
            y,x=(3,k) if mode==0 else (k,3)
            for s in range(3):
                prefix=f'y{y}_x{x}_s{s}'
                t=a[prefix+'_time_s'];d=a[prefix+'_depletion']
                ax.plot(t,d,c=color,lw=.9,alpha=.8)
                if a['observed'][y,x,s]:ax.plot(a['run_restricted_times_s'][y,x,s],1-a['Z_at_end'][y,x,s],marker='o',ms=3,c=color)
        ax.set(xlabel='Time (s)',xlim=(0,24),ylim=(0,.55))
        ax.set_title('Vary τZ' if mode==0 else 'Vary depletion threshold',fontsize=14)
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'),ax=ax,pad=.03)
        cb.set_label('τZ (s)' if mode==0 else 'Ith')
    axes[0].set_ylabel('Depletion, 1 − mean Z')
    fig.tight_layout();save(fig,'parameter_depletion_trajectories')


def completed_grid_preview():
    protocol=json.loads((DENSE/'protocol.json').read_text())
    tau=np.array(protocol['tau_z_ms']);threshold=np.array(protocol['threshold'])
    values=np.full((7,7,3),np.nan);observed=np.full_like(values,np.nan)
    paths=[Path(r['source']) for r in protocol['reused_runs']]
    paths += [DENSE/'runs'/(j['name']+'.json') for j in protocol['jobs']]
    for path in paths:
        if not path.exists():continue
        result=json.loads(path.read_text())
        if result.get('status')!='COMPLETE':continue
        job=result['job'];x=np.argmin(abs(tau-job['tau_z_ms']));y=np.argmin(abs(threshold-job['threshold']))
        s=protocol['seeds'].index(job['seed']);tm=result['first_trigger_ms']
        values[y,x,s]=min(tm/1000,24.) if tm is not None else 24.
        observed[y,x,s]=float(tm is not None and tm<=24000)
    # np.mean intentionally leaves any incomplete three-seed cell missing.
    return dict(tau_s=tau/1000,threshold=threshold,restricted_mean_s=values.mean(-1),
        transition_fraction=observed.mean(-1),runs=int(np.isfinite(values).sum()),
        complete_cells=int(np.isfinite(values).all(-1).sum()))


def main(native_only=False,preview=False):
    a=np.load(NATIVE/'spatial_analysis.npz')
    fig=plt.figure(figsize=(11,7.7));spec=fig.add_gridspec(1,1,left=.065,right=.90,top=.96,bottom=.09)[0]
    spatial=native_transition(fig,spec,a,title='Native early recruitment')
    save(fig,'native_early_transition');spectra(a);ei_rates(a);family_comparison(a);frequency_comparison(a);broadband_comparison(a)
    readme='''### native_early_transition.png / .pdf
同一手放双核 SNN 状态2到3附近的三个连续250毫秒窗，上排为原生E神经元放电率，下排为20–150 Hz局部电流功率相对0.5–8秒基线的dB变化。40×40网格覆盖整个平面，每格0.5毫米；不经过电极投影或空间插值。
**关注点**：放电率升高与特定频带增强分开判读，蓝色为频带功率降低。主图保留真实增强/减弱，不根据预期梯度选窗；core边界为1.5毫米。

### native_local_spectra.png / .pdf
Core A、Core B与核外区域的局部电流PSD，比较基线和相同三个早期窗。PSD先在每个原生网格计算，再按区域内真实E细胞数加权，故不是把区域电流先相干平均后的PSD。
**关注点**：250毫秒Hann窗、线性去趋势和4 Hz频率分辨率，20–150 Hz主带之外另保留分带结果。频带增强不能直接等同于患者SEEG或已建立的极限环。

### native_EI_rate_maps.png / .pdf
同一三个早期窗分别显示全平面的原生E、I神经元放电率，直接统计各0.5毫米网格的spikes并按该类细胞数归一化。无I细胞的网格保持缺失，不填零；青色圈在两排都指示E阈值core的位置。
**关注点**：区分E与I的局部招募强度和范围，不能把青圈解释为另外施加的I-core。两类使用各自明确的Hz色标，未做电极读出或空间插值。

### native_families_and_early_transition.png / .pdf
保留上一版的全部A先行/B先行事件身份，将招募rank改为原生0.5毫米网格，并与10.48–10.73秒的原生放电率、20–150 Hz功率变化直接并列。未参与网格保持缺失；不使用电极高斯插值。
**关注点**：同一早期高态场用于两类比较，rank、放电率与频带功率是不同观测，不预设两类一定共享相同梯度。

### native_frequency_comparison.png / .pdf
同样三个早期窗并列20–150 Hz与250–500 Hz功率变化，用于定位原生场频谱从低中频向更高频的转移。两个频带都保留负值，分别使用以零为中心、数值明确的dB色标；高频图作为补充诊断，不替换主频带。
**关注点**：局部高频功率增加和大幅群体振荡并不等价，需结合完整PSD、相对调制和逐神经元spikes。此处的20–150 Hz短窗诊断不是患者Figure 3精确1–150 Hz、1秒PSD与robust-z标准的直接复现。

### native_broadband_1s_windows.png / .pdf
按患者Figure 3的1–150 Hz频带和1秒窗时长做补充原生场诊断，依次显示9–10、9.5–10.5和10–11秒三个重叠窗。基线仍为0.5–8秒，图示平均功率比的dB变化；不是临床robust-z图的数值复制。
**关注点**：更长窗可见转变前的低频/宽频能量增强，但混合了点火、上升和高率平台，不能将其正值直接解释为持续振荡。14个基线窗少于临床合同要求，未以更正向的结果替换短窗诊断。
'''
    if (FIG/'native_rate_preview_1mm.png').exists():readme+='\n'+PREVIEW_README
    if (FIG/'fig5_manual_core_release_layout_v11_progress.png').exists():readme+=LAYOUT_PREVIEW_README
    if (FIG/'native_EI_transition.gif').exists():readme+='''
### native_EI_transition.gif
直接显示9.00–11.18秒的原生E/I空间放电过程，每帧统计40毫秒spikes、步长20毫秒，保留真实0.5毫米网格。与静态图一样按每格实际细胞数归一化，无I细胞的网格留白，物理core为1.5毫米。
**关注点**：动画放慢播放以检查招募次序与活动带扩张，不显示频带功率；画面时间标签为实际仿真时间。
'''
    (FIG/'README.md').write_text(readme)
    if native_only:
        write('native_figure_metadata.json',spatial)
        from report_topic4_fig5_native_dense import main as report
        report();return
    if preview:d=completed_grid_preview()
    else:
        d=np.load(DENSE/'analysis_arrays.npz')
        ds=json.loads((DENSE/'analysis_summary.json').read_text());assert ds['runs']==147
    metadata=json.loads((prior.previous.OUT/'figure_metadata.json').read_text())
    original,run,*_=prior.previous.previous.prior.old.previous.source.load_main()
    t,xyz,_=prior.previous.previous.current_coordinates(original)
    paths=prior.previous.previous.previous.previous.complete_paths(t,xyz,run,metadata['windows'])
    saved=np.load(BASE/'layout_v8/trajectory_arrays.npz')
    assert all(np.array_equal(p['coords'],saved[f'path{k}_Z_H_E']) and np.array_equal(p['time'],saved[f'path{k}_time_s']) for k,p in enumerate(paths))
    fig=plt.figure(figsize=(26,18));outer=fig.add_gridspec(1,2,width_ratios=[1.37,1],left=.054,right=.94,top=.945,bottom=.05,wspace=.18)
    before=len(fig.axes)
    left=prior.left_with_boundaries(fig,outer[0],original,run,metadata['windows'],1.5,[10.25,11.0])
    for ax in fig.axes[before:]:
        for text in list(ax.texts):
            if text.get_text()=='M off':text.remove()
    right=outer[1].subgridspec(3,1,height_ratios=[1.05,1.22,.90],hspace=.32)
    phase=right[0].subgridspec(1,3,width_ratios=[.08,1,.22])[1]
    ax=fig.add_subplot(phase,projection='3d')
    prior.previous.plot_summary(ax,t,xyz,paths,run,metadata['windows'],panel_letter=False)
    ax.set_ylabel(r'$H_E$ (mV equiv.)',fontsize=14,labelpad=8)
    for artist in list(ax.get_children()):
        if isinstance(artist,Annotation):
            if not artist.get_text().isdigit():artist.remove()
            else:
                artist.xyann=tuple(.68*np.array(artist.xyann));artist.set_fontsize(12)
    ax.set_title('E1  State trajectory',fontsize=16,weight='bold',loc='left',pad=10)
    native_transition(fig,right[1],a)
    latency(fig,right[2],d,title='F  Z kinetics (in progress)' if preview else 'F  Z kinetics')
    fig.suptitle('Slow inhibitory depletion and spatial recruitment in a dual-core SNN',fontsize=20,y=.985)
    if preview:
        save(fig,'fig5_manual_core_release_layout_v11_progress')
        write('preview_metadata.json',dict(completed_runs=d['runs'],complete_cells=d['complete_cells'],
            missing_cells='Shown blank; only complete three-seed cells are colored.',status='GRID_IN_PROGRESS'))
        if '### fig5_manual_core_release_layout_v11_progress' not in readme:
            with (FIG/'README.md').open('a') as stream:stream.write(LAYOUT_PREVIEW_README)
        return
    save(fig,'fig5_manual_core_release_layout_v11')
    fig=plt.figure(figsize=(8.5,6.5));spec=fig.add_gridspec(1,1,left=.15,right=.87,top=.91,bottom=.14)[0]
    latency(fig,spec,d,title='Z kinetics');save(fig,'z_parameter_transition_time_7x7');depletion(d)
    with (FIG/'README.md').open('a') as stream:stream.write('''
### fig5_manual_core_release_layout_v11.png / .pdf
左侧保留同一连续SEEG/raster/Z和空间快照，右侧改为简洁的原生三维轨迹、状态2→3的原生放电/频带场及真实7×7参数图。去除三维轨迹灰色补充文字、E2和F的段落注释；保留必要变量、单位和时间。
**关注点**：E2来自原生神经元网格，时窗10.25–11.00秒，与左侧对应；不是触点插值。参数图每格3条噪声，49格共147条，其中120条为此次加密新增。

### z_parameter_transition_time_7x7.png / .pdf
同一底物与方程下的τZ×耗竭电流阈值加密，颜色为截至24秒的限制平均进入高态时间。条纹表示该格至少一条运行未在24秒内进入；白圈为原工作点，网格之间不插值。
**关注点**：τZ同时控制耗竭与恢复，阈值控制耗竭条件而不是I→E连接强度。未进入者贡献24秒，保留原3条配对噪声种子，不按阳性结果补点。

### parameter_depletion_trajectories.png / .pdf
固定另一参数时显示7档τZ或耗竭阈值下的全部3条噪声轨迹，纵轴为1−平均Z，点标首次达到高态判据。曲线截止首次进入或24秒，不混入人工补回。
**关注点**：Z对应对耗竭驱动的时间滤波而非无恢复的总放电计数；不同参数的进入时刻与资源轨迹同时判读，不假定存在唯一临界平均Z。
''')
    write('figure_metadata.json',dict(source=str(prior.OUT),spatial=spatial,left=left,
        native_replay=str(NATIVE),dense_parameter_analysis=str(DENSE),
        removed_annotations=['Refill starts / Z released 3D text boxes','M off','E2 explanation paragraphs','F explanation paragraphs'],
        trajectory_unchanged=True,time_colorbar='right',parameter_nodes=49,parameter_runs=147,
        physical_core_radius_mm=1.5,human_acceptance='PENDING_USER_REVIEW'))
    from report_topic4_fig5_native_dense import main as report
    report()
    scripts=[Path(__file__)]+[ROOT/'scripts'/name for name in [
        'replay_topic4_fig5_native_transition.py','analyze_topic4_fig5_native_transition.py',
        'run_topic4_fig5_latency_dense.py','analyze_topic4_fig5_latency_dense.py',
        'audit_topic4_fig5_native_spectrum.py','audit_topic4_fig5_native_spikes_broadband.py',
        'animate_topic4_fig5_native_transition.py','report_topic4_fig5_native_dense.py']]
    write('producer_manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts})
    write('delivery_status.json',dict(status='RENDERED_PENDING_AGENT_VISUAL_REVIEW',human_acceptance='PENDING_USER_REVIEW'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--native-only',action='store_true');parser.add_argument('--preview-rate',action='store_true');parser.add_argument('--layout-preview',action='store_true');args=parser.parse_args()
    if args.preview_rate:rate_preview()
    else:main(args.native_only,preview=args.layout_preview)
