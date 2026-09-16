#!/usr/bin/env python3
"""Review figures for prescribed spatial Z; no native bifurcation assertion."""
from analyze_topic4_prescribed_z_phase import OUT,PREVIOUS,REFERENCE,TIMES,source,read,write
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse

plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})


def save(fig,name,description,focus):
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    fig.savefig(dest/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(dest/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    p=dest/'README.md';text=p.read_text() if p.exists() else '# 给定空间 Z 路径的诊断图\n\n'
    heading=f'### {name}\n';block=heading+'\n'+description+'\n\n**关注点**：'+focus+'\n\n'
    if heading in text:
        start=text.index(heading);end=text.find('\n### ',start+len(heading));text=text[:start]+block+(text[end+1:] if end>=0 else '')
    else:text+=block
    p.write_text(text)


def planes():
    assert read(OUT/'plane_status.json')['status']=='COMPLETE'
    native,src,z=source();a=np.load(PREVIOUS/'mixed_timescale/native_replay.npz')
    fig=plt.figure(figsize=(15,8.6),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[.8,1.4]);top=fig.add_subplot(grid[0,:])
    e=np.average(a['fields_hz'][:,0],axis=1,weights=a['count_e']);e=e[:10680].reshape(-1,10).mean(1)
    ne=native['rate_e_hz'][:106800].reshape(-1,100).mean(1)
    tt=(np.arange(len(e))+.5)*.01
    top.plot(tt,ne,c='0.65',lw=.9,label='Native SNN reference');top.plot(tt,e,c='#357ba1',lw=1,label='Corrected rate: prescribed native Z')
    for tm in TIMES:top.axvline(tm/1000,c='#cc9732',lw=1,ls='--')
    top.set(xlim=(0,10.68),ylim=(0,370),xlabel='Replay time (s)',ylabel='Mean E rate (Hz)')
    top.legend(loc='upper left',ncol=2,fontsize=10)
    for k,tm in enumerate(TIMES):
        ax=fig.add_subplot(grid[1,k]);a=np.load(OUT/f'plane_{tm}.npz');X,Y,U,V=[a[key] for key in ('X','Y','U','V')]
        tr=a['trajectory_hz'];center=a['center_hz'];d=a['center_drift_hz_per_s']
        endpoint=center+d*np.array([.005,.0025])
        xmax=min(500,max(tr[:,0].max(),center[0],endpoint[0])*1.3+10)
        ymax=min(800,max(tr[:,1].max(),center[1],endpoint[1])*1.3+10)
        speed=np.hypot(U/xmax,V/ymax);den=np.maximum(speed,1e-12)
        # Only draw arrows inside this displayed slice; lengths are normalized.
        mask=(X<=xmax)&(Y<=ymax)
        ax.quiver(X[::2,::2],Y[::2,::2],np.where(mask,U/den,np.nan)[::2,::2],np.where(mask,V/den,np.nan)[::2,::2],
            color='0.72',angles='xy',scale_units='xy',scale=18,width=.003)
        for drift,color in ((U,'#b13c76'),(V,'#24818b')):
            if drift.min()<0<drift.max():ax.contour(X,Y,drift,levels=[0],colors=[color],linewidths=2)
        ax.plot(tr[:,0],tr[:,1],c='#222222',lw=1.3)
        for j in (10,35,65,85):
            ax.annotate('',xy=tr[j+3],xytext=tr[j],arrowprops={'arrowstyle':'->','color':'#222222','lw':1.2})
        ax.scatter(*center,s=65,c='#e3a731',edgecolor='black',zorder=5)
        ax.set(xlim=(0,xmax),ylim=(0,ymax),xlabel='Mean E rate (Hz)',ylabel='Mean I rate (Hz)',
            title=f'{tm/1000:g} s | mean Z = {float(a["mean_Z"]):.3f}')
    handles=[Line2D([],[],c='#b13c76',lw=2,label='Conditional dE/dt = 0'),Line2D([],[],c='#24818b',lw=2,label='Conditional dI/dt = 0'),
        Line2D([],[],c='k',lw=1.3,label='Actual replay trajectory: +/-50 ms'),Line2D([],[],marker='o',ls='',mfc='#e3a731',mec='k',label='Exact snapshot')]
    fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False,fontsize=11)
    fig.suptitle('Prescribed spatial Z: instantaneous conditional nullclines, direction fields and trajectories',fontsize=15)
    save(fig,'prescribed_z_conditional_phase_planes',
        '上排为原SNN参照及修正rate model在真实空间Z和原OU输入回放下的群体率，下排取8.8、10.15和10.5秒的条件E–I相平面。每张同时显示两条零漂移线、归一化方向箭头、前后50毫秒实际轨迹及精确快照。',
        '每张场固定该时刻Z、已更新突触电流、延迟历史和外部输入；黑色轨迹上的隐藏状态仍在演化，因此黑线无需处处跟随灰箭头。零漂移交点不等于完整网络固定点；各相平面的坐标范围按轨迹分别显示，原SNN分岔归因未通过。')


def stability():
    branch=read(OUT/'branch_status.json')['rows'];spec=read(OUT/'spectrum_status.json')['rows'];_,src,z=source()
    fig,axs=plt.subplots(2,2,figsize=(13,8.5),layout='constrained');ts=np.arange(len(z))*.01;keep=ts<=10.68
    mean=np.average(z,axis=1,weights=src['count_e'])
    axs[0,0].fill_between(ts[keep],z.min(1)[keep],z.max(1)[keep],color='#87a9c5',alpha=.3,label='Cell minimum–maximum')
    axs[0,0].plot(ts[keep],mean[keep],c='#326f99',label='Neuron-weighted mean')
    axs[0,0].set(ylabel='E-target Z',title='Actual spatial resource path');axs[0,0].legend(fontsize=10)
    colors={'low':'#326f99','high':'#ab4277'}
    for name in ('low','high'):
        rr=sorted([r for r in branch if r['branch']==name],key=lambda r:r['time_ms'])
        tt=np.array([r['time_ms']/1000 for r in rr]);val=np.array([r.get('E_mean_hz',np.nan) if r['valid'] else np.nan for r in rr])
        axs[0,1].plot(tt,val,c=colors[name],lw=1.1,marker='.',ms=3,label=f'Continuation from {name}-rate seed')
        sr=sorted([r for r in spec if r['branch']==name],key=lambda r:r['time_ms'])
        if not sr:continue
        xx=[r['time_ms']/1000 for r in sr]
        growth=[max(v['real_per_s'] for v in r['spectrum']['roots']) for r in sr]
        count=[r['root_count']['unstable_roots'] if r['root_count']['status']=='PASS' else np.nan for r in sr]
        axs[1,0].plot(xx,count,'o',c=colors[name],label=f'{name}-seed branch')
        axs[1,1].plot(xx,growth,'o',c=colors[name],label=f'{name}-seed branch')
        for r in sr:
            stable=r['root_count']['status']=='PASS' and r['root_count']['unstable_roots']==0
            axs[0,1].scatter(r['time_ms']/1000,r['E_mean_hz'],s=55,edgecolors=colors[name],facecolors=colors[name] if stable else 'white',zorder=4)
    axs[0,1].set(ylabel='Equilibrium mean E rate (Hz; log scale)',yscale='log',title='Resolved fixed points: constant background');axs[0,1].legend(fontsize=9)
    axs[1,0].set(ylabel='Number of unstable modes',title='Full delayed-map count at tested profiles');axs[1,0].legend(fontsize=10)
    axs[1,1].axhline(0,c='0.5',lw=1);axs[1,1].set(ylabel='Largest sampled Re(lambda) (1/s)',title='Shift-targeted spectrum, checked in full map')
    for ax in axs.flat:
        ax.set(xlim=(0,10.68),xlabel='Z-profile position on recorded path (s)')
        for tm in TIMES:ax.axvline(tm/1000,c='#cc9732',ls='--',lw=.8,alpha=.6)
    crossing=read(OUT/'first_instability.json')
    for ax in axs.flat:ax.axvline(crossing['time_parameter_ms']/1000,c='#27836a',lw=1,ls=':')
    axs[1,1].annotate(f"Early oscillatory crossing\n{crossing['time_parameter_ms']/1000:.3f} s; {crossing['frequency_hz']:.2f} Hz",xy=(crossing['time_parameter_ms']/1000,0),xytext=(3.2,7),arrowprops={'arrowstyle':'->','color':'#27836a'},fontsize=10,color='#27836a')
    fig.suptitle('Frozen fast-system stability along prescribed native spatial Z',fontsize=15)
    save(fig,'prescribed_z_frozen_stability',
        '沿真实空间Z路径逐点冻结，固定参考背景输入，从低率和高率初值延续固定点。下排分别给出保留全部延迟历史的失稳模态计数和移位Arnoldi抽样谱，均以实际数值校验为依据。',
        '横轴是选择Z场的位置，不是此冻结系统的演化时间；背景条件与OU回放不同。空缺是未解析固定点而非已证实鞍结，空心标记表示未确认稳定，谱抽样最大实部不保证全谱最右根；本图分析简化模型，不宣称原SNN发生相同分岔。')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['planes','stability']);a=p.parse_args()
    {'planes':planes,'stability':stability}[a.part]()
