#!/usr/bin/env python3
"""Read only closed observation blocks; audit local versus global escalation."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_topic4_reset_state_diagnosis import OUT, PREV, load_series


def spans(mask):
    d=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))


def sampled_spikes(lo_s,hi_s,chunk_paths):
    lo=round(lo_s*10000);hi=round(hi_s*10000);times=[];ids=[]
    if lo<2380000:
        sel=np.load(OUT/'geometry.npz')['sample_source_indices']
        with np.load(PREV/'runs/weak_fast_z_refill_recurrence.npz') as f:
            sp=f['sample_spikes'][lo:min(hi,2380000),sel]
        tt,nn=np.where(sp);times.append((tt+lo)*.0001);ids.append(nn)
    for p in chunk_paths:
        start,end=[int(x) for x in p.stem.split('_')]
        a,b=max(lo,start),min(hi,end)
        if b<=a:continue
        with np.load(p) as f:sp=f['raster'][a-start:b-start]
        tt,nn=np.where(sp);times.append((tt+a)*.0001);ids.append(nn)
    return np.concatenate(times),np.concatenate(ids)


def main():
    folder=OUT/'runs/z_only_long'
    series,digests=load_series(folder)
    paths=[folder/'chunks'/name for name in sorted(digests)]
    t=series['time_s'];end=float(t[-1]+.005)
    assert np.allclose(np.diff(t),.005,atol=1e-10)
    t10=t[::2]
    global_e=series['rate_E'].reshape(-1,2).mean(1)
    regions=series['region_E'].reshape(-1,2,3).mean(1)
    rates=np.column_stack([global_e,regions])
    populations=['All E','Core A','Core B','Surround E']
    audit=[]
    for j,label in enumerate(populations):
        before=[];after=[]
        for lo,hi in spans(rates[:,j]>=200):
            item=dict(start_s=float(t10[lo]),end_s=float(t10[hi-1]+.01),duration_ms=int((hi-lo)*10))
            if item['start_s']<75.5:before.append(item)
            if item['start_s']>=76.5:after.append(item)
        audit.append(dict(population=label,
            pre_reset_sustained=[x for x in before if x['duration_ms']>=200],
            post_reset_sustained=[x for x in after if x['duration_ms']>=200],
            post_reset_longest=max(after,key=lambda x:x['duration_ms']) if after else None))
    assert audit[0]['pre_reset_sustained'][0]['start_s']==73.48
    core_b=audit[2]
    strongest=core_b['post_reset_longest']
    if strongest is None:raise ValueError('No post-reset core-B excursion to display')
    post_center=(strongest['start_s']+strongest['end_s'])/2
    windows=[(71.5,74.5,'First escalation'),(post_center-1.5,post_center+1.5,'Longest post-reset Core B excursion')]
    windows[1]=(max(76.5,windows[1][0]),min(end,windows[1][1]),windows[1][2])
    summary=[]
    for lo,hi,label in [(40,70,'Before first escalation'),(100,130,'After reset'),(end-30,end,'Latest 30 s')]:
        q=(t>=lo)&(t<hi)
        summary.append(dict(label=label,time_s=[lo,hi],Z=series['Z'][q][:,[0,5,6]].mean(0).tolist(),
            adaptation_current=series['M'][q,:3].mean(0).tolist(),
            region_E_hz=series['region_E'][q].mean(0).tolist(),mean_E_hz=float(series['rate_E'][q].mean())))
    artifact=OUT/'interim';figdir=artifact/'figures';figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':13,'axes.labelsize':15,'axes.titlesize':15,
                         'xtick.labelsize':12,'ytick.labelsize':12,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors=['#66517f','#ba466d','#2081a4'];labels=['All E','Core A','Core B']
    fig,axs=plt.subplots(3,1,figsize=(16,9),sharex=True)
    n_overview=len(t10)//50
    overview_t=t10[:n_overview*50].reshape(-1,50).mean(1)+.005
    overview_r=rates[:n_overview*50].reshape(-1,50,4).mean(1)
    for j in [0,1,2]:axs[0].plot(overview_t,overview_r[:,j],c=colors[j],lw=1,label=labels[j])
    axs[0].set(ylabel='E rate (Hz)\n500-ms mean',ylim=(0,550))
    for j,k in enumerate([0,5,6]):axs[1].plot(t,series['Z'][:,k],c=colors[j],lw=.8,label=labels[j])
    axs[1].set(ylabel='Resource Z',ylim=(.3,1.02))
    for j in range(3):axs[2].plot(t,series['M'][:,j],c=colors[j],lw=.8)
    axs[2].set(ylabel='ηM × M\n(mV equiv.)',xlabel='Simulation time (s)')
    for ax in axs:
        ax.axvspan(75.5,76.5,color='#409e87',alpha=.2,lw=0)
        ax.axvline(76.5,color='#409e87',ls='--',lw=1)
        ax.set_xlim(0,end)
    axs[0].legend(ncol=3,frameon=False,loc='upper right')
    axs[0].set_title(f'Z-only continuation · completed blocks through {end:g} s',loc='left')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(figdir/f'long_run_overview.{ext}',dpi=165,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(4,2,figsize=(15.5,11),sharey='row',gridspec_kw={'height_ratios':[1.05,1.15,.8,.8]})
    for col,(lo,hi,title) in enumerate(windows):
        q=(t>=lo)&(t<hi);q10=(t10>=lo)&(t10<hi)
        for j in range(3):axs[0,col].plot(t10[q10],rates[q10,j],c=colors[j],lw=1.3,label=labels[j])
        axs[0,col].axhline(200,ls=':',lw=.8,c='#969696')
        axs[0,col].set_ylim(0,550);axs[0,col].set_title(title,weight='bold')
        st,nn=sampled_spikes(lo,hi,paths)
        for a,b,c in [(0,20,colors[1]),(20,40,colors[2]),(40,60,'#708798'),(60,80,'#c47e45')]:
            m=(nn>=a)&(nn<b);axs[1,col].scatter(st[m],nn[m],s=1,c=c,marker='.',lw=0,rasterized=True)
        axs[1,col].set(ylim=(-1,80),yticks=[10,30,50,70],yticklabels=['Core A E','Core B E','Other E','I'])
        for cut in [19.5,39.5,59.5]:axs[1,col].axhline(cut,c='#cccccc',lw=.6)
        for j,k in enumerate([0,5,6]):axs[2,col].plot(t[q],series['Z'][q,k],c=colors[j],lw=1.4)
        axs[2,col].set_ylim(.4,.9)
        for j in range(3):axs[3,col].plot(t[q],series['M'][q,j],c=colors[j],lw=1.4)
        for row in range(4):
            axs[row,col].set_xlim(lo,hi)
            if row<3:axs[row,col].tick_params(labelbottom=False)
        axs[3,col].set_xlabel('Simulation time (s)')
    axs[0,0].set_ylabel('E rate (Hz)');axs[2,0].set_ylabel('Resource Z')
    axs[3,0].set_ylabel('ηM × M\n(mV equiv.)')
    axs[0,1].legend(ncol=3,frameon=False,fontsize=11,loc='upper right')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(figdir/f'local_escalation_comparison.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    evidence=dict(status='INTERIM_SINGLE_TRAJECTORY',saved_until_s=end,
        observation_after_release_s=end-76.5,statistical_unit='one continuous trajectory',
        diagnostic='200 Hz in 10-ms E-population bins; >=200 ms sustained, applied separately to each named population',
        populations=audit,window_summaries=summary,display_windows=windows,
        post_window_selection='The longest post-reset Core B run of >=200 Hz bins; first occurrence breaks ties',
        closed_chunk_paths=[str(p) for p in paths],
        global_endpoint_independently_recomputed=True,overview_rate_bin_ms=500,
        zoom_and_endpoint_rate_bin_ms=10,M_causality='NOT_ESTABLISHED',
        permanent_nonrecurrence='NOT_ESTABLISHED',agent_visual_review='PENDING',human_review='PENDING')
    (artifact/'local_vs_global_audit.json').write_text(json.dumps(evidence,indent=2)+'\n')
    text=(f'# 长程中期：局部升级与全局招募\n\n读取已完整保存至{end:g}秒的连续轨迹；释放Z后观察{end-76.5:g}秒。'
          '未改变网络、动力学参数、原生观测或正式高态判据。\n\n'
          f'按同一200Hz/200ms操作阈值检查各E群体，恢复后全局与两核持续高活动数量分别为'
          f'{len(audit[0]["post_reset_sustained"])}/{len(audit[1]["post_reset_sustained"])}/{len(audit[2]["post_reset_sustained"])}。'
          f'Core B恢复后的最长高放电片段为{strongest["duration_ms"]}ms（{strongest["start_s"]:.2f}秒），'
          '第一次转变前先出现330ms片段，随后进入更长的局部高活动并扩展至全局。\n\n'
          '因此当前记录中的缺口已出现在局部持续升级阶段，不只是群体平均掩盖局部runaway。'
          '这不是M抑制该阶段的因果证据，也不能说明该状态永远不可达；按原计划继续到1000秒或实际再次进入。'
          '局部操作阈值仅用于这次机制诊断，不作为新增训练目标或患者传播判据。\n')
    (artifact/'scientific_note.md').write_text(text)
    (figdir/'README.md').write_text(
        '### long_run_overview.png / .pdf\n显示完整已保存轨迹的全E及两核放电率、Z和适应电流；绿色标记唯一一次Z补充及释放。'
        '全程概览的放电率采用500ms均值，避免密集短峰被误看成高平台；局部对照和高态判据仍使用10ms分箱。'
        '只绘制闭合观测块，时间终点写在图内。\n**关注点**：长窗中期记录不构成永久不发作或稳定吸引子的证明。\n\n'
        '### local_escalation_comparison.png / .pdf\n以相同3秒宽度、相同纵轴及固定80个神经元抽样，对照首次局部升级与恢复后Core B最长高放电片段。'
        '同时展示真实raster、两核Z和适应电流；右侧按持续时间选窗，未按外观选择。\n'
        '**关注点**：60ms一类的短片段与持续升级分开判读；未据此认定M的因果作用，图待用户目视审阅。\n')
    print(json.dumps({k:evidence[k] for k in ['saved_until_s','observation_after_release_s','populations']},indent=2))


if __name__=='__main__':main()
