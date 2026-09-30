"""Fig.5 A/B/C layout for the autonomous spatial kinetic candidate.

Display conventions follow plot_topic4_fig5_clean_panels.py: same sample IDs,
group order, 50-ms maps, resource statistics, and shared spatial color scale.
All panels here come from the new candidate's single uninterrupted trajectory.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, ConnectionPatch, Rectangle
from audit_topic4_fig5_native_reduction_correspondence import runs, finite_events, first

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/fig5_spatial_kinetic_full_trajectory_20260916'
SOURCE=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1'
COL=['#616873','#267ba8','#dd871c','#ba263c']
REG=['#806398','#d45381','#4d9fc0']


def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def select(a):
    r=a['spikes_1ms'][:,0].reshape(-1,10).sum(1)/320
    end=len(r)*.01
    onset=first(r>=200,hold=20,start=0.)
    quiet,events=finite_events(r,start=0.)
    rest=[]
    for lo,hi in quiet:
        for b in range(lo,hi-4):
            t=b*.01+.025
            if .7<=t<=2.:rest.append(t)
    assert rest,'No early rest window; do not invent a Rest state.'
    rest_t=min(rest,key=lambda t:abs(t-1.235))
    early=[e for e in events if e['start_s']>=2. and e['end_s']<=min(6.,(onset or end)-.3)]
    assert early,'No early self-limited event; do not invent an Interictal state.'
    for e in early:
        e['peak_s']=(e['start_bin']+int(np.argmax(r[e['start_bin']:e['end_bin']])))*.01+.005
    event=min(early,key=lambda e:abs(e['peak_s']-4.025))
    event_t=round(min(event['start_s']+.020,event['end_s']-.005),3)
    if onset is not None:
        before=[(lo,hi) for lo,hi in quiet if hi*.01<onset]
        assert before
        late=before[-1][1]*.01
        last=min(onset+.100,end-.025)
        labels=['Rest','Interictal','Pre-entry','High-rate']
    else:
        late,last=end-1.,end-.1
        labels=['Rest','Interictal','Late activity','Late activity']
    snaps=[dict(number=i+1,time_s=float(t),label=labels[i],color=COL[i])
           for i,t in enumerate([rest_t,event_t,late,last])]
    assert all(0.025<=s['time_s']<=end-.025 for s in snaps)
    assert np.all(np.diff([s['time_s'] for s in snaps])>0)
    return r,snaps,dict(high_onset_s=onset,state2_event=event,
        state2_selection='Complete self-limited event in 2-6s with peak nearest4.025s; display onset+20ms. No spatial localization optimization.',
        state1_selection='Complete50ms quiet window in0.7-2s with center nearest1.235s.',
        state3_selection='End of the last >=20ms below5Hz quiet interval before high onset.',
        state4_selection='First sustained200Hz-for200ms onset+100ms; not a bifurcation or clinical seizure.',
        events=events)


def statistics(r,lo,hi):
    x=r[round(lo*100):round(hi*100)]
    _,ev=finite_events(x,start=lo)
    lengths=[e['duration_ms'] for e in ev]
    intervals=np.diff([e['start_s'] for e in ev])
    return dict(window_s=[lo,hi],finite_events=len(ev),quiet_fraction=float(np.mean(x<5)),
        mean_global_E_hz=float(x.mean()),duration_median_ms=float(np.median(lengths)) if lengths else None,
        duration_range_ms=[min(lengths),max(lengths)] if lengths else None,
        onset_intervals_ms=(intervals*1000).tolist(),
        onset_interval_CV=float(intervals.std()/intervals.mean()) if len(intervals)>1 else None,
        events=ev)


def render():
    assert json.loads((OUT/'run/status.json').read_text())['status']=='COMPLETE'
    a=dict(np.load(OUT/'run/trajectory.npz'))
    assert a['start_ms']==0
    r,snaps,selection=select(a);end=len(r)*.01
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':14,
        'axes.linewidth':1.,'xtick.labelsize':12,'ytick.labelsize':12,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(12.8,13.8))
    gs=fig.add_gridspec(6,1,left=.105,right=.885,top=.972,bottom=.065,
        height_ratios=[3.4,1.5,.14,2.1,.30,2.35],hspace=.38)
    ax=fig.add_subplot(gs[0]);it,ix=np.where(a['raster']);tt=it*.0001
    mapping=np.r_[np.linspace(0,33,20),np.linspace(36,69,20),np.linspace(72,84,20),np.linspace(87,99,20)]
    for lo,hi,color in [(0,20,REG[1]),(20,40,REG[2]),(40,60,'#225b7f'),(60,80,'#c17730')]:
        take=(ix>=lo)&(ix<hi)
        ax.scatter(tt[take],mapping[ix[take]],s=2.4,marker='o',c=color,lw=0,rasterized=True)
    for y in [34.5,70.5,85.5]:ax.axhline(y,color='#bbbbbb',lw=.7)
    ax.set(ylim=(-2,101),yticks=[16.5,52.5,78,93],yticklabels=['Core A E','Core B E','Other E','I'],xlim=(0,end))
    ax.tick_params(axis='x',labelbottom=False)
    ax.text(-.065,1.04,'A',transform=ax.transAxes,weight='bold',fontsize=21)
    previous=-10
    for s in snaps:
        label_y=.98 if s['time_s']-previous>.38 else .88
        ax.text(s['time_s'],label_y,str(s['number']),transform=ax.get_xaxis_transform(),ha='center',va='top',
            color=s['color'],weight='bold',fontsize=15,bbox=dict(fc='white',ec='none',alpha=.9,pad=.6))
        previous=s['time_s']
    zoomgs=gs[1].subgridspec(1,2,wspace=.30);zooms=[]
    for j,s in enumerate(snaps[1:3]):
        zax=fig.add_subplot(zoomgs[j]);lo=s['time_s']-.05;hi=lo+.3
        for left,right,color in [(0,20,REG[1]),(20,40,REG[2])]:
            take=(ix>=left)&(ix<right)&(tt>=lo)&(tt<hi)
            zax.scatter(tt[take],ix[take],s=12,marker='|',c=color,linewidth=.75,rasterized=True)
        zax.axhline(19.5,color='#bbbbbb',lw=.7)
        zax.axvspan(s['time_s']-.025,s['time_s']+.025,color=s['color'],alpha=.12,lw=0)
        zax.axvline(s['time_s'],color=s['color'],ls=':',lw=1.1)
        zax.set(ylim=(-1,40),xlim=(lo,hi),yticks=[9.5,29.5],yticklabels=['Core A E','Core B E'],xlabel='Time (s)')
        zax.set_xticks(np.array([s['time_s'],s['time_s']+.1,s['time_s']+.2]))
        zax.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%.2f'))
        zax.text(0,1.07,str(s['number']),transform=zax.transAxes,color=s['color'],fontsize=15)
        ax.add_patch(Rectangle((lo,-1),hi-lo,71,fc='none',ec=s['color'],lw=1.5,zorder=10))
        zooms.append(dict(number=s['number'],window_s=[lo,hi],same_samples_and_raw_spikes=True))
    za=fig.add_subplot(gs[3],sharex=ax);ma=za.twinx()
    zt=a['slow_time_ms']/1000;z=a['Z'];m=.0005*a['M']
    za.fill_between(zt,z[:,2],z[:,4],color=REG[0],alpha=.15,lw=0)
    for zi,mi,color in [(0,0,REG[0]),(5,1,REG[1]),(6,2,REG[2])]:
        za.plot(zt,z[:,zi],color=color,lw=1.6)
        ma.plot(zt,m[:,mi],color=color,ls='--',lw=1.3)
    za.set(ylabel='Resource Z',ylim=(0,1.05),xlabel='Time (s)',yticks=[0,.25,.5,.75,1.])
    za.set_xticks([0,2,4,6,8,10,12])
    upper=max(.5,np.ceil(m.max()*10)/10)
    ma.set(ylim=(0,upper),yticks=[0,upper/2,upper],ylabel=r'$\eta_M M$ (mV equiv.)')
    ma.spines['right'].set_visible(True)
    handles=[Line2D([],[],color=c,label=n) for c,n in zip(REG,['All E','Core A','Core B'])]
    handles += [Line2D([],[],color='black',label='Z'),Line2D([],[],color='black',ls='--',label=r'$\eta_M M$')]
    za.legend(handles=handles,ncol=5,loc='lower right',bbox_to_anchor=(1.,1.015),frameon=False,fontsize=11.5,
              handlelength=1.7,columnspacing=1.2,borderaxespad=0.)
    za.text(-.065,1.12,'B',transform=za.transAxes,weight='bold',fontsize=21)
    for panel in [ax,za]:
        if selection['high_onset_s'] is not None:panel.axvspan(selection['high_onset_s'],end,color=COL[3],alpha=.08,lw=0)
        for s in snaps:
            panel.axvspan(s['time_s']-.025,s['time_s']+.025,color=s['color'],alpha=.10,lw=0)
            panel.axvline(s['time_s'],color=s['color'],ls=':',lw=1.)
    maps=[];mapaxes=[];mapgs=gs[5].subgridspec(1,4,wspace=.20)
    for j,s in enumerate(snaps):
        q=fig.add_subplot(mapgs[j]);lo=round(s['time_s']*1000)-25
        field=a['field_1ms'][lo:lo+50].sum(0)/a['cell_e_counts']/.05
        im=q.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',
                    norm=PowerNorm(.6,0,500),interpolation='nearest')
        for k,xy in enumerate(a['centers_mm']):
            q.add_patch(Circle(xy,float(a['core_radius_mm']),ec='#2dd4cd',fc='none',lw=1.35))
            q.text(xy[0],xy[1]+2.1,'AB'[k],color='#10656a',fontsize=10,ha='center',weight='bold',
                   bbox=dict(fc='white',ec='none',pad=.15,alpha=.9))
        q.set(xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
        if j==0:q.set_ylabel('y (mm)')
        else:q.set_yticklabels([])
        q.set_title(f'{s["number"]}  {s["label"]}\n{s["time_s"]:.3f} s',color=s['color'],fontsize=12.5,pad=10)
        connector=ConnectionPatch(xyA=(s['time_s'],-.32),coordsA=za.get_xaxis_transform(),
            xyB=(.5,1.33),coordsB=q.transAxes,arrowstyle='-',color=s['color'],lw=1.,alpha=.75,clip_on=False,zorder=-1)
        fig.add_artist(connector)
        mapaxes.append(q)
        maps.append(dict(number=s['number'],time_window_s=[lo/1000,(lo+50)/1000],field_E_hz=field,
                        weighted_global_E_hz=float(np.average(field,weights=a['cell_e_counts']))))
    fig.canvas.draw()
    box=mapaxes[-1].get_position()
    cax=fig.add_axes([box.x1+.014,box.y0,.012,box.height])
    cb=fig.colorbar(im,cax=cax,ticks=[0,250,500]);cb.set_label('E rate (Hz)')
    fig.text(.055,mapaxes[0].get_position().y1+.047,'C',weight='bold',fontsize=21)
    for panel in fig.axes:
        if panel not in [ma,cax]:panel.spines[['top','right']].set_visible(False)
        panel.tick_params(direction='out',length=3.5)
    for ext in ('png','pdf','svg'):
        fig.savefig(figdir/f'fig_kinetic_full_trajectory.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    native=[]
    for path in sorted((SOURCE/'replay/runs/eta0.0005_s9108401/chunks').glob('*.npz')):
        with np.load(path) as chunk:native.append(chunk['spikes_1ms'][:,0])
    nr=np.concatenate(native).reshape(-1,10).sum(1)/320
    stats={name:{'candidate':statistics(r,*win),'native':statistics(nr,*win)}
           for name,win in [('early_interictal',(0.5,8.)),('pre_entry_window',(8.,9.42))]}
    native_onset=first(nr>=200,hold=20,start=0.)
    meta=dict(model='40x40 mean spatial kinetic candidate, 40000 particles',continuous_from_zero=True,
        seed=9108401,Z='dynamic',M='dynamic',eta_M=.0005,tau_M_ms=1000,tau_Z_ms=5000,
        no_native_state_transplants_after_zero=True,display_time_window_s=[0,end],snapshots=snaps,
        selection=selection,raster=dict(sample_ids=a['sample_ids'],groups=['Core A E','Core B E','Other E','I'],
            per_group=20,spike_time_resolution_ms=.1,unchanged_samples_from_reference=True),
        spatial_maps=maps,zooms=zooms,compute_grid_mm=.5,observed_grid_mm=1.,
        spatial_normalization=dict(cmap='magma',norm='PowerNorm',gamma=.6,vmin=0,vmax=500,window_ms=50),
        Z_envelope='10th-90th percentiles over E particles, not uncertainty',
        resource_core_radius_mm=1.75,outline_threshold_core_radius_mm=1.5,
        interictal_statistics=stats,native_high_onset_s=native_onset,scientific_equivalence='NOT_ESTABLISHED',
        producer=str(Path(__file__).resolve()),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source=str(OUT/'run/trajectory.npz'),source_sha256=hashlib.sha256((OUT/'run/trajectory.npz').read_bytes()).hexdigest(),
        human_visual_acceptance='PENDING')
    (OUT/'figure_metadata.json').write_text(json.dumps(safe(meta),indent=2,allow_nan=False)+'\n')
    readme='''### fig_kinetic_full_trajectory.png / .pdf / .svg
A展示当前0.5mm空间候选从初态连续运行的80个固定粒子栅格，分Core A E、Core B E、Other E和I，并放大②和③附近相同放电数据。B实线为全E及两核的Z，虚线为有效适应电流ηM M；阴影是E粒子Z的10–90%分位，Z和M均动态更新。C为本轨迹自身四个时刻的50ms二维放电率，计算网格0.5mm、显示网格1mm，编号和时间与A/B对应；High-rate只是全局200Hz持续200ms的操作性状态，尚非等效验收或分岔证据。**关注点**：早期间期事件能否自行结束、两核是否交替，以及③到④空间招募的变化；不与原SNN同编号时刻强行视为同一状态。
'''
    (figdir/'README.md').write_text(readme)
    c,n=stats['early_interictal']['candidate'],stats['early_interictal']['native']
    lines=['# 当前空间候选：从零开始的完整轨迹',
        '本图按用户参考的A栅格/放大、B资源、C空间快照布局绘制；不是原生SNN图，也未拼接原生历史。',
        '当前模型为40×40 mean空间通信、40000粒子；原参数不变，Z和M均动态。仅seed9108401一条展示轨迹，不能作为多噪声等效验收。',
        f'自主高率进入：候选 {selection["high_onset_s"]} s；原生 {native_onset} s。原先从8s原生状态移植续接的10.18s进入时间不适用于这条从零开始的新轨迹。',
        f'0.5–8s完整自限事件：候选{c["finite_events"]}次，原生{n["finite_events"]}次；静息占比{c["quiet_fraction"]:.3f}/{n["quiet_fraction"]:.3f}；事件时长中位数{c["duration_median_ms"]}/{n["duration_median_ms"]}ms。',
        '状态选取规则、全部事件、源文件和数值见figure_metadata.json。编号只表示同一图内时空对应，静态空间图不能独自证明传播一致性或同步。',
        '数值核查：8–8.1s被动观察器与原候选空间计数/核计数/外部输入逐位一致；本次完整轨迹外部随机状态与原生检查点逐项核验，空间与全局计数守恒。Agent图形自查记录见visual_qa.json；用户人工验收待定。']
    (OUT/'figure_report.md').write_text('\n\n'.join(lines)+'\n')
    print(json.dumps(safe(dict(snapshots=snaps,high_onset_s=selection['high_onset_s'],
        candidate_early={k:v for k,v in c.items() if k!='events'},native_early={k:v for k,v in n.items() if k!='events'})),indent=2))


if __name__=='__main__':render()
