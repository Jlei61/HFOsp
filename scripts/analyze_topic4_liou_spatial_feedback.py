#!/usr/bin/env python3
"""Native observations for the bounded Liou spatial-feedback comparison."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import run_topic4_liou_spatial_feedback as run
import analyze_topic4_autonomous_recovery as common
import analyze_topic4_event_extent as extent

OUT=run.OUT


def feedback(folder):
    values={}
    for p in sorted((folder/'spatial_feedback_chunks').glob('*.npz')):
        if '.tmp.' in p.name:continue
        with np.load(p) as a:
            for k in a.files:values.setdefault(k,[]).append(a[k])
    return {k:np.concatenate(v) for k,v in values.items()}


def audit(folder):
    old=common.OUT;common.OUT=OUT
    try:stats=common.classify(folder)
    finally:common.OUT=old
    a=common.load(folder,['time_ms','spikes_1ms','regions_1ms','slow_time_ms','Z'])
    q=feedback(folder);geo=np.load(OUT/'geometry.npz')
    n=len(a['spikes_1ms'])//10
    r=np.c_[a['spikes_1ms'][:n*10,0].reshape(n,10).sum(1)/32000/.01,
        a['regions_1ms'][:n*10,:3].reshape(n,10,3).sum(1)/geo['region_counts'][:3]/.01]
    quiet=np.all(r[:,:3]<5,axis=1)
    onset=stats['entries'][0]['onset_s'] if stats['entries'] else None
    ti=a['slow_time_ms']/1000
    ix=np.minimum((ti/.01).astype(int),n-1)
    sel=quiet[ix] & ((ti>onset) if onset is not None else np.zeros(len(ti),bool))
    z_up=(1-a['Z'][:,-1]-a['Z'][:,0])/stats['job']['tau_Z_s']
    stats['quiet_after_first_high']=dict(actual_s=float(quiet[np.arange(n)*.01>(onset or np.inf)].sum()*.01),
        sampled_slow_points=int(sel.sum()),
        mean_fraction_E_with_recovery_drive=float(np.mean(1-a['Z'][sel,-1])) if sel.any() else None,
        fraction_slow_points_with_positive_mean_Z_derivative=float(np.mean(z_up[sel]>0)) if sel.any() else None,
        note='Actual simultaneous allE/CoreA/CoreB<5Hz bins. PositiveZ derivative means b_fraction>meanZ under the unchanged native equation; not merely a low rate.')
    assert np.allclose(q['raw_global_current'],stats['job']['spatial_gamma']*stats['job']['current_per_Hz']*q['synaptic_E_rate_Hz'])
    stats['feedback_current_equation_QA']='PASS'
    stats['actual_feedback_ms']=15.
    return stats


def plots(rows,seed):
    rows=sorted([r for r in rows if r['job']['seed']==seed],key=lambda x:x['job']['spatial_gamma'])
    if len(rows)!=3:return
    end=min(r['observed_s'] for r in rows)
    if end<=0:return
    geo=np.load(OUT/'geometry.npz');nr=geo['region_counts'];centers=geo['centers_mm']
    fig=plt.figure(figsize=(19,18));gs=fig.add_gridspec(6,3,height_ratios=[1.7,1,.95,.8,.65,1.25],left=.09,right=.94,bottom=.06,top=.96,hspace=.38,wspace=.28)
    letters=iter('ABCDEFGHIJKLMNOPQR')
    for col,r in enumerate(rows):
        folder=OUT/'runs'/r['name'];a=common.load(folder,['time_ms','spikes_1ms','regions_1ms','raster','slow_time_ms','Z','M','field_5ms','field_time_ms']);q=feedback(folder)
        axes=[fig.add_subplot(gs[row,col]) for row in range(5)]
        for row,ax in enumerate(axes):
            ax.text(-.17,1.03,'ABCDE'[row]+str(col+1),transform=ax.transAxes,fontsize=19,weight='bold')
            ax.set_xlim(0,end);ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=13)
            if row<4:ax.tick_params(labelbottom=False)
            else:ax.set_xlabel('Time (s)',fontsize=17)
        axes[0].set_title(f'γ = {r["job"]["spatial_gamma"]:.3g}',fontsize=19,pad=14)
        raster=a['raster'][:round(end*10000)];it,ix=np.nonzero(raster)
        axes[0].scatter(it*.0001,ix,s=3.2,marker='|',linewidths=.6,c=np.where(ix<60,'#387d9d','#c7833b'),rasterized=True)
        for y in [19.5,39.5,59.5]:axes[0].axhline(y,c='.7',lw=.5)
        axes[0].set(ylim=(-1,80),yticks=[9.5,29.5,49.5,69.5],yticklabels=['Core A E','Core B E','Other E','I'] if col==0 else [])
        t=a['time_ms']/1000;ts=a['slow_time_ms']/1000
        rates=np.c_[a['spikes_1ms'][:,0]*1000/32000,a['regions_1ms'][:,:2]*1000/nr[:2]]
        for i,(color,label) in enumerate([('#333333','All E'),('#bd5e9e','Core A E'),('#4d9ecd','Core B E')]):
            axes[1].plot(t,gaussian_filter1d(rates[:,i],2),c=color,lw=.65,label=label)
        axes[1].set_ylim(-5,505)
        axes[2].plot(ts,a['Z'][:,0],c='#7e3ba0',lw=1.2,label='E mean Z')
        axes[2].fill_between(ts,a['Z'][:,2],a['Z'][:,4],color='#7e3ba0',alpha=.15)
        axes[2].set_ylim(0,1.04)
        twin=axes[2].twinx();twin.plot(ts,a['M'][:,0]*r['job']['eta_m'],c='#c37b35',lw=.9,label='M current')
        twin.tick_params(labelsize=12,labelcolor='#a45b25');twin.set_ylim(0,5.3)
        if col==2:twin.set_ylabel('M current (mV equiv.)',fontsize=15,color='#a45b25')
        qt=q['time_ms']/1000
        axes[3].plot(qt,q['raw_global_current'],c='#c29261',lw=.8,label='Global before Z')
        axes[3].plot(qt,q['effective_global_current'],c='#397e73',lw=.8,label='Global after Z')
        axes[3].set_ylim(-10,1750)
        axes[4].plot(qt,q['Z_recovery_drive_fraction'],c='#537864',lw=.65)
        axes[4].set_ylim(-.03,1.03)
        if col==0:
            for ax,label in zip(axes[1:],['E rate (Hz)','Resource Z','Global current\n(mV equiv.)','Z recovery\nfraction']):ax.set_ylabel(label,fontsize=16)
            axes[1].legend(fontsize=11,loc='upper right',frameon=True)
            axes[2].legend(fontsize=11,loc='upper right',frameon=True)
            axes[3].legend(fontsize=11,loc='upper right',frameon=True)
        for e in r['entries']:
            for ax in axes:ax.axvline(e['onset_s'],c='#b53e4b',ls=':',lw=.6)
        for e in r['recoveries']:
            for ax in axes:ax.axvline(e['confirmation_s'],c='#2c927c',ls=':',lw=.6)
        sub=gs[5,col].subgridspec(1,2,wspace=.2)
        ref=(r['entries'][0]['onset_s']+.1) if r['entries'] else (float(t[(t>1)&(t<end)][np.argmax(rates[(t>1)&(t<end),0])]))
        for j,tm in enumerate([min(ref,end-.025),end-.025]):
            ax=fig.add_subplot(sub[0,j]);lo=max(0,round((tm-.025)/.005));hi=lo+10
            field=a['field_5ms'][lo:hi].sum(0)/geo['cell_e_counts']/.05
            im=ax.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=500,interpolation='nearest')
            for center in centers:ax.add_patch(plt.Circle(center,float(geo['core_radius_mm']),fill=False,ec='#52d0d0',lw=1.2))
            ax.set(xticks=[0,10,20],yticks=[0,10,20]);ax.set_title(f'{tm:.2f} s',fontsize=15);ax.set_xlabel('x (mm)',fontsize=16)
            if col==0 and j==0:ax.set_ylabel('y (mm)',fontsize=16)
            else:ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=12)
    cb=fig.add_axes([.958,.07,.009,.115]);fig.colorbar(im,cax=cb,label='E rate (Hz)',ticks=[0,250,500])
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(dest/f'liou_spatial_feedback_s{seed}.{ext}',dpi=170,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


def main(partial=False):
    p=run.prepare();rows=[]
    for job in p['initial_jobs']:
        folder=OUT/'runs'/job['name']
        if not (folder/'result.json').exists() and not partial:raise RuntimeError('Final analysis requested before all six endpoints')
        if not list((folder/'chunks').glob('*.npz')):continue
        rows.append(audit(folder))
    if not rows:return
    for seed in [9108401,9108402]:plots(rows,seed)
    if not partial:extent.main(OUT)
    run.carrier.base.write(OUT/'analysis.json',dict(updated_at=time.time(),final=not partial,rows=rows,
        statistical_unit='One parameter/noise trajectory on the same topology; noise conditions are paired counterfactuals.',
        interpretation='Original uniform E-output spatial projection and15ms synapse on currentLIF. No10s memory or50Hz threshold. Full LAS biophysics is not reproduced.',human_review='PENDING'))
    lines=['# Liou 全局空间反馈：当前双核 SNN 对照','',
        '复现的是全局空间平均投射和15ms突触响应。局部显式I网络、原生Z/M、双核和噪声保留；强度按固定旧基线换算，不把电导数值直接当作电流。无人工reset、无新增rho、无10秒池或50Hz开启门。','',
        '| γ | 噪声 | 实际时长(s) | 高态进入(s) | 返回确认(s) | 末10s全E/较强核(Hz) | quiet中Z上升比例 |',
        '|---|---|---:|---|---|---|---|']
    for r in rows:
        quiet=r['quiet_after_first_high']['fraction_slow_points_with_positive_mean_Z_derivative']
        lines.append(f'|{r["job"]["spatial_gamma"]:.3g}|{r["job"]["seed"]}|{r["observed_s"]:.1f}|{[e["onset_s"] for e in r["entries"]]}|{[e["confirmation_s"] for e in r["recoveries"]]}|{r["tail10s_mean_Hz"][0]:.2f}/{max(r["tail10s_mean_Hz"][1:3]):.2f}|{quiet}|')
    lines += ['','高态门为全E≥200Hz连续200ms；返回门要求全E和两core都满足两秒低率及安静比例。必须结合完整raster、真实Z回补和原生空间活动判断，未进入不等于自主恢复，单个core停止也不等于全网恢复。','',
        '图的A行为固定80神经元样本raster，B为全群体率，C为Z及实际M电流，D为新增全局电流，E为满足原Z回补条件的E细胞比例；最下行为初次进入附近（无进入时取峰活动）和观察末端各50ms的原生场。纵虚线为高态起点和返回确认。','',
        '来源：[Liou2020](https://elifesciences.org/articles/50927)，[原始空间连接实现](https://github.com/elifesciences-publications/LAS-Model/blob/master/StandardRecurrentConnection.m)。','',
        '状态：'+('部分轨迹快照，实验仍在运行。' if partial else '六个完整端点均已分析，待科学及用户目视审阅；不自动扩展或冻结。')]
    (OUT/'scientific_review.md').write_text('\n'.join(lines)+'\n')
    d=OUT/'figures'
    if d.exists():
        (d/'README.md').write_text('### liou_spatial_feedback_s9108401.png / .pdf\n同一手放双核底物与第一条配对噪声下，对比原生控制、全局占比1/6及1/2；各行同步显示真实raster、全体/核内率、Z/M、全局电流及Z回补条件。下方为同一轨迹的原生50ms细胞场，所有空间图共用0–500Hz。**关注点**：去除慢全局尾巴后，网络是否真能终止并回补Z，还是仅防止进入或保持高平台。\n\n### liou_spatial_feedback_s9108402.png / .pdf\n第二条噪声的相同条件与观测协议，不是新增网络拓扑重复。只有完整时长和实际局部事件能支持恢复判断。**关注点**：第一噪声观察是否保留，以及全局均率是否掩盖持续core活动。\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');args=p.parse_args();main(args.partial)
