#!/usr/bin/env python3
"""Read-only figures from the source-equation Liou reference experiments."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/liou_original_design_20260915'
plt.rcParams.update({'font.size':13,'axes.labelsize':15,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':11,'pdf.fonttype':42})

def load(folder):
    p=json.loads((folder/'protocol.json').read_text())
    tr=np.load(folder/'traces.npz')
    state=np.load(folder/'state.npy',mmap_mode='r')
    if p['spiking']:
        s=np.load(folder/'spikes.npy',mmap_mode='r')
        ntime=len(s)//10
        # 10 ms by 20 neighboring neurons, literal spike counts, no interpolation.
        field=s[:ntime*10].reshape(ntime,10,p['n']//20,20).sum((1,3))/(.01*20)
    else:
        s=None
        r=np.load(folder/'output.npy',mmap_mode='r')
        ntime=len(r)//10
        field=r[:ntime*10].reshape(ntime,10,p['n']).mean(1)
    return p,tr,state,field,s

def decorate(ax,label=None):
    ax.spines[['top','right']].set_visible(False)
    if label:ax.text(-.12,1.03,label,transform=ax.transAxes,fontsize=18,fontweight='bold')

def metrics(folder):
    p,tr,state,field,s=load(folder)
    t=tr['trace_time_ms']/1000
    rates=tr['trace'][:,:4]
    active=np.any(field>20,axis=1)
    active_t=(np.arange(len(field))+.5)*.01
    post=active_t>=5
    tail=active_t>=active_t[-1]-5
    baseline=(t>=.5)&(t<2)
    result={'name':folder.name,'protocol':p,'baseline_E_Hz':float(rates[baseline,0].mean()),'tail5_E_Hz':float(rates[-5000:,0].mean()),'peak_100ms_global_Hz':float(gaussian_filter1d(rates[:,0],100).max()),'last_20Hz_local_activity_s':float(active_t[active][-1]) if active.any() else None,'post_stim_active_time_s':float(active[post].sum()*.01),'tail5_fraction_with_local_over20Hz':float(active[tail].mean()),'max_recruited_fraction_at20Hz':float((field>20).mean(1).max()),'final_Cl_mean_mM':float(state[-1,2].mean()),'final_gK_mean_nS':float(state[-1,3].mean()/.2),'readout_boundary':'Rate: native populations. Spiking: 10 ms/20-neuron bins; isolated sparse spikes can cross 20 Hz. These are descriptive and are not the native 40k model 200 Hz entry criterion.'}
    return result

def rate_figure(folder):
    p,tr,state,field,_=load(folder);t=tr['trace_time_ms']/1000;ts=tr['state_time_ms']/1000
    fig=plt.figure(figsize=(16,10));gs=fig.add_gridspec(3,2,left=.08,right=.94,bottom=.08,top=.94,hspace=.40,wspace=.24)
    ax=fig.add_subplot(gs[0,0]);im=ax.imshow(field.T,origin='lower',aspect='auto',extent=[0,len(field)*.01,0,1],cmap='magma',vmin=0,vmax=200,interpolation='nearest');ax.axvspan(2,5,color='#59b982',alpha=.18);ax.set(xlabel='Time (s)',ylabel='Position');decorate(ax,'A')
    cb=fig.colorbar(im,ax=ax,pad=.015);cb.set_label('Rate (Hz)')
    ax=fig.add_subplot(gs[0,1]);ax.imshow(field[1800:2100].T,origin='lower',aspect='auto',extent=[18,21,0,1],cmap='magma',vmin=0,vmax=200,interpolation='nearest');ax.set(xlabel='Time (s)',ylabel='Position');decorate(ax,'B')
    ax=fig.add_subplot(gs[1,0]);
    for j,(c,label) in enumerate([('#202020','All populations'),('#bc5878','Stimulated region'),('#48918c','Middle'),('#b08e35','Far end')]):ax.plot(t,tr['trace'][:,j],c=c,lw=.7,label=label)
    ax.legend(loc='upper right');ax.set(xlabel='Time (s)',ylabel='Rate (Hz)',xlim=(0,100));decorate(ax,'C')
    ax=fig.add_subplot(gs[1,1]);locations=[.075,.125,.25,.4]
    for x,c in zip(locations,['#777777','#bc5878','#48918c','#b08e35']):ax.plot(ts,state[:,2,int(x*p['n'])],c=c,lw=1,label=f'x = {x:g}')
    ax.legend();ax.set(xlabel='Time (s)',ylabel='Intracellular Cl (mM)',xlim=(0,100));decorate(ax,'D')
    ax=fig.add_subplot(gs[2,0]);
    for x,c in zip(locations,['#777777','#bc5878','#48918c','#b08e35']):ax.plot(ts,state[:,3,int(x*p['n'])]/.2,c=c,lw=1,label=f'x = {x:g}')
    ax.set(xlabel='Time (s)',ylabel='Slow K conductance (nS)',xlim=(0,100));decorate(ax,'E')
    ax=fig.add_subplot(gs[2,1]);
    for x,c in zip(locations,['#777777','#bc5878','#48918c','#b08e35']):ax.plot(ts,state[:,1,int(x*p['n'])],c=c,lw=1,label=f'x = {x:g}')
    ax.set(xlabel='Time (s)',ylabel='Spike threshold (mV)',xlim=(0,100));decorate(ax,'F')
    save(fig,'original_rate_exp2a')

def spiking_figure(seed):
    folders=[OUT/'reference_runs'/f'{e}_s{seed}' for e in ['exp4a','exp4b']]
    fig=plt.figure(figsize=(17,14));gs=fig.add_gridspec(5,2,left=.085,right=.94,bottom=.06,top=.96,hspace=.32,wspace=.24,height_ratios=[1.5,1.25,1,.8,.8])
    for col,folder in enumerate(folders):
        p,tr,state,field,s=load(folder);t=tr['trace_time_ms']/1000;ts=tr['state_time_ms']/1000
        ax=fig.add_subplot(gs[0,col]);it,idx=np.nonzero(s[:,::5]);ax.scatter((it+1)*.001,(idx*5+1)/p['n'],s=.5,c='#246c8a',rasterized=True,linewidths=0);ax.axvspan(2,5,color='#ce705c',alpha=.17);ax.set(xlim=(0,50),ylim=(0,1),xlabel='Time (s)',ylabel='Neuron position');ax.set_title(f'Original spiking model: γ = {p["w_global_i"]/300:.3g}',fontsize=17);decorate(ax,'A' if col==0 else 'B')
        ax=fig.add_subplot(gs[1,col]);sel=s[2000:8000,::2];it,idx=np.nonzero(sel);ax.scatter(2+(it+1)*.001,(idx*2+1)/p['n'],s=1,c='#246c8a',rasterized=True,linewidths=0);ax.axvspan(2,5,color='#ce705c',alpha=.17);ax.set(xlim=(2,8),ylim=(0,.55),xlabel='Time (s)',ylabel='Neuron position');decorate(ax,'C' if col==0 else 'D')
        ax=fig.add_subplot(gs[2,col]);
        for j,(c,label) in enumerate([('#202020','All neurons'),('#bc5878','Stimulated region'),('#48918c','Middle')]):ax.plot(t,gaussian_filter1d(tr['trace'][:,j],5),c=c,lw=.7,label=label)
        ax.set(xlim=(0,50),xlabel='Time (s)',ylabel='Rate (Hz)');ax.legend(loc='upper right');decorate(ax,'E' if col==0 else 'F')
        ax=fig.add_subplot(gs[3,col]);
        for x,c in zip([.075,.25,.5,.8],['#bc5878','#777777','#48918c','#b08e35']):ax.plot(ts,state[:,2,int(x*p['n'])],c=c,lw=1,label=f'x = {x:g}')
        ax.set(xlim=(0,50),xlabel='Time (s)',ylabel='Cl (mM)');ax.legend(loc='upper left',ncol=2);decorate(ax,'G' if col==0 else 'H')
        ax=fig.add_subplot(gs[4,col]);
        for x,c in zip([.075,.25,.5,.8],['#bc5878','#777777','#48918c','#b08e35']):ax.plot(ts,state[:,3,int(x*p['n'])]/.2,c=c,lw=1)
        ax.set(xlim=(0,50),xlabel='Time (s)',ylabel='Slow K (nS)');decorate(ax,'I' if col==0 else 'J')
    save(fig,f'original_spiking_s{seed}')

def save(fig,name):
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(dest/f'{name}.{ext}',dpi=160,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)

def spatial_ablation_figure():
    names=['exp2a_s20260915','exp2_global_removed','exp2_global_redistributed_local_extended200']
    fig=plt.figure(figsize=(18,10));gs=fig.add_gridspec(3,3,left=.07,right=.95,bottom=.08,top=.95,hspace=.35,wspace=.26,height_ratios=[1.2,1,.8])
    rows=[]
    for col,name in enumerate(names):
        folder=OUT/'reference_runs'/name
        p,tr,state,field,s=load(folder)
        end=p['duration_ms']/1000;t=tr['trace_time_ms']/1000
        rows.append(metrics(folder))
        ax=fig.add_subplot(gs[0,col]);im=ax.imshow(field.T,origin='lower',aspect='auto',extent=[0,len(field)*.01,0,1],cmap='magma',vmin=0,vmax=150,interpolation='nearest');ax.axvspan(2,5,color='#6caf87',alpha=.25);ax.set(xlabel='Time (s)',ylabel='Position',xlim=(0,end));ax.set_title(f'Local {p["w_local_i"]:g} / global {p["w_global_i"]:g}',fontsize=17);decorate(ax,'ABC'[col])
        ax=fig.add_subplot(gs[1,col]);
        for j,(c,lab) in enumerate([('#333333','All populations'),('#bb567a','Stimulated region'),('#4a9b98','Middle')]):ax.plot(t,gaussian_filter1d(tr['trace'][:,j],20),lw=.75,c=c,label=lab)
        ax.set(xlabel='Time (s)',ylabel='Rate (Hz)',xlim=(0,end),ylim=(0,150));decorate(ax,'DEF'[col])
        if col==0:ax.legend(loc='upper right')
        ax=fig.add_subplot(gs[2,col]);ax.plot(t,tr['trace'][:,8],color='#276c9b',lw=.7,label='Global inhibitory conductance');ax.set(xlabel='Time (s)',ylabel='Global inhibition (nS)',xlim=(0,end),ylim=(0,8));decorate(ax,'GHI'[col])
    cb=fig.add_axes([.961,.725,.008,.20]);fig.colorbar(im,cax=cb,label='Rate (Hz)')
    save(fig,'original_global_spatial_ablation')
    (OUT/'spatial_ablation_analysis.json').write_text(json.dumps({'rows':rows,'scope':'Single deterministic source rate model; the global-removed control also reduces total weight, while redistribution preserves the300total. These controls preserve all equations.'},indent=2)+'\n')

def extended_spiking_figure():
    names=['exp4a_s20260915_extended150','exp4a_s20260916_extended150','exp4b_s20260915_extended150','exp4b_s20260916']
    fig=plt.figure(figsize=(18,12));gs=fig.add_gridspec(4,4,left=.065,right=.97,bottom=.07,top=.95,hspace=.35,wspace=.28,height_ratios=[1.35,1,.8,.8])
    for col,name in enumerate(names):
        p,tr,state,field,s=load(OUT/'reference_runs'/name);t=tr['trace_time_ms']/1000;ts=tr['state_time_ms']/1000
        end=85. if p['duration_ms']>50000 else 40.
        ax=fig.add_subplot(gs[0,col]);it,ix=np.nonzero(s[:int(end*1000),::8]);ax.scatter((it+1)*.001,(ix*8+1)/p['n'],s=.8,c='#216d8a',linewidths=0,rasterized=True);ax.axvspan(2,5,color='#c37860',alpha=.22);ax.set(xlim=(0,end),ylim=(0,1),xlabel='Time (s)',ylabel='Neuron position');ax.set_title(f'γ = {p["w_global_i"]/300:.3g}; seed {p["seed"]}',fontsize=14);decorate(ax,'ABCD'[col])
        ax=fig.add_subplot(gs[1,col]);
        for j,(c,lab) in enumerate([('#333333','All neurons'),('#b84e7a','Stimulated region')]):ax.plot(t,gaussian_filter1d(tr['trace'][:,j],5),c=c,lw=.6,label=lab)
        ax.set(xlim=(0,end),ylim=(0,180),xlabel='Time (s)',ylabel='Rate (Hz)');decorate(ax,'EFGH'[col])
        if col==0:ax.legend(loc='upper right')
        for row,idx,label in [(2,2,'Cl (mM)'),(3,3,'Slow K (nS)')]:
            ax=fig.add_subplot(gs[row,col])
            for pos,c in zip([.075,.3,.6],['#b84e7a','#4a9b98','#b08c36']):ax.plot(ts,state[:,idx,int(pos*p['n'])]/(.2 if idx==3 else 1.),c=c,lw=.8)
            ax.set(xlim=(0,end),xlabel='Time (s)',ylabel=label);decorate(ax,'IJKL'[col] if row==2 else 'MNOP'[col])
    save(fig,'original_spiking_recovery_extended')

def two_dimensional_figure():
    folder=OUT/'reference_runs/exp1_2d';a=np.load(folder/'traces.npz');f=np.load(folder/'field_Hz.npy',mmap_mode='r');s=np.load(folder/'state.npy',mmap_mode='r');mask=a['mask'];t=a['time_ms']/1000;ft=a['field_time_ms']/1000
    times=[3.,20.,45.,70.,90.]
    fig=plt.figure(figsize=(17,12));gs=fig.add_gridspec(4,5,left=.07,right=.94,bottom=.07,top=.95,hspace=.4,wspace=.24,height_ratios=[.85,1,1,1])
    ax=fig.add_subplot(gs[0,:3]);ax.plot(t,a['trace'][:,0],lw=.7,c='#333333');ax.axvspan(2,5,color='#d28e68',alpha=.18);ax.set(xlabel='Time (s)',ylabel='Mean rate (Hz)',xlim=(0,100));decorate(ax,'A')
    for tm in times:ax.axvline(tm,lw=.6,ls=':',color='#777777')
    ax=fig.add_subplot(gs[0,3:]);ax.plot(t,a['trace'][:,6],lw=.7,c='#276e9d');ax.set(xlabel='Time (s)',ylabel='Global inhibition (nS)',xlim=(0,100));decorate(ax,'B')
    specs=[('Rate (Hz)',0,150,'magma'),('Cl (mM)',6,22,'viridis'),('Slow K (nS)',0,7,'plasma')]
    for row,(label,vmin,vmax,cmap) in enumerate(specs,1):
        for col,tm in enumerate(times):
            idx=int(np.argmin(abs(ft-tm)))
            val=f[idx] if row==1 else s[idx,row, :].reshape(100,100)/(.2 if row==3 else 1.)
            val=np.ma.masked_array(val,~mask)
            ax=fig.add_subplot(gs[row,col]);im=ax.imshow(val.T,origin='lower',extent=[0,1,0,1],cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest');ax.set(xlabel='x',xticks=[0,.5,1],yticks=[0,.5,1]);ax.set_aspect('equal')
            if col==0:ax.set_ylabel('y')
            else:ax.tick_params(labelleft=False)
            if row==1:ax.set_title(f'{tm:g} s',fontsize=16)
        pos=ax.get_position()
        fig.text(.016,pos.y0+pos.height/2,label,rotation=90,va='center',fontsize=15)
        cb=fig.add_axes([.954,pos.y0,.01,pos.height]);fig.colorbar(im,cax=cb,label=label)
    save(fig,'original_2d_spatial_recovery')
    active=np.any(f.reshape(len(f),-1)>20,axis=1)
    (OUT/'reference_2d_analysis.json').write_text(json.dumps({'source':'Exp1','last_local_rate_over20Hz_s':float(ft[active][-1]),'max_fraction_disk_over20Hz':float(((f>20).reshape(len(f),-1).sum(1)/mask.sum()).max()),'tail5_disk_E_Hz':float(a['trace'][-5000:,0].mean()),'snapshot_times_s':times,'human_visual_acceptance':'PENDING'},indent=2)+'\n')

def main():
    folders=sorted((OUT/'reference_runs').glob('*/result.json'))
    rows=[metrics(f.parent) for f in folders if f.parent.name!='exp1_2d']
    (OUT/'reference_analysis.json').write_text(json.dumps({'rows':rows,'human_visual_acceptance':'PENDING'},indent=2)+'\n')
    rate_figure(OUT/'reference_runs/exp2a_s20260915')
    for seed in [20260915,20260916]:spiking_figure(seed)
    if (OUT/'reference_runs/exp2_global_redistributed_local_extended200/result.json').exists():spatial_ablation_figure()
    if (OUT/'reference_runs/exp4b_s20260915_extended150/result.json').exists():extended_spiking_figure()
    if (OUT/'reference_runs/exp1_2d/result.json').exists():two_dimensional_figure()
    (OUT/'figures/README.md').write_text('### original_rate_exp2a.png / .pdf\n按作者 Exp2A 参数与更新顺序执行的 500 点一维 rate 参考模型，显示真实空间率、局部率、氯离子、慢钾电导和快阈值适应。外部刺激只在2–5秒，之后未重置状态。**关注点**：撤除输入后是否持续传播、转为阵发活动并自行终止；这不是双核 SNN。\n\n### original_spiking_s20260915.png / .pdf\n作者 Exp4A/B 参数下的两种局部/全局抑制比例，展示全段及起始区间的真实 spike raster 和局部状态。固定稀疏采样用于显示，全体率使用全部2000个神经元。**关注点**：起始模式、空间波前与末端状态是否改变，不能用全局均率掩盖局部持续发放。\n\n### original_spiking_s20260916.png / .pdf\n第二个配对随机种子的相同原文参数，不改变方程或刺激。**关注点**：第一种子的行为是否重现，区分进入失败与进入后终止。\n')
    with (OUT/'figures/README.md').open('a') as h:
        h.write('\n### original_global_spatial_ablation.png / .pdf\n原文rate基线、删除全局投射以及把全局权重全部分配回局部的对照；第三列延长至200秒，各列横轴如实标注观察长度。未更改细胞方程、初始状态或刺激。**关注点**：停止前的波前位置与空间边界，不能仅凭结束时刻判定机制。\n\n### original_spiking_recovery_extended.png / .pdf\n四条原文spiking轨迹的完整事件及自行恢复，三条从50秒延长至150秒且全部初始spike和已存状态逐位一致。图截去恢复后冗长的静息段，完整记录保留。**关注点**：是否先进入并传播，再在没有reset的情况下整体停下。\n\n### original_2d_spatial_recovery.png / .pdf\n作者100×100圆形rate场，逐时刻显示原生率、氯离子和慢钾电导；全局投射仍按作者完整方形节点数归一化。**关注点**：活动范围如何扩大、慢变量如何在空间上错开，以及最终停止。\n')
    print(json.dumps({'analyzed_1d_runs':len(rows)},indent=2))

if __name__=='__main__':main()
