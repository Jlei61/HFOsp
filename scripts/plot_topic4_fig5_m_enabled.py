#!/usr/bin/env python3
"""Left-side Fig5 M-on assay, with a fixed M-off reference and explicit refill."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, lfilter
from scipy.stats import linregress
import plot_topic4_fig5_manual_release as source
from plot_topic4_fig5_manual_release_layout_v3 import ACTIVITY_CMAP

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1'
OUT=BASE/'m_enabled_v1';FIG=OUT/'figures'
REF=BASE/'runs/continuous_refill_release'
NAMES=['M_off','m0p2_refill','m0p2_native','m0p8_native']
COLORS=['#26638c','#b5791b','#a33b54','#377b4f']
ANCHORS=[1.1325,10.3075,10.98,12.961,25.5]
REGCOLORS=['#603175','#aa3768','#17658f']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.labelsize':18,
    'xtick.labelsize':16,'ytick.labelsize':16,'axes.titlesize':19,'pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False})


def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load(name):
    path=REF if name=='M_off' else OUT/'runs'/name
    with np.load(path.with_suffix('.npz')) as f:a={k:f[k] for k in f.files}
    return a,json.loads(path.with_suffix('.json').read_text())


def intervals(mask):
    changes=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)))


def analyze(a,r):
    dt=float(a['dt_ms']);e=a['rate_e_hz'].reshape(-1,round(10/dt)).mean(1)
    t=(np.arange(len(e))+.5)*.01;zt=a['z_time_ms']/1000;z=a['z_stats'][:,0]
    q=a['z_stats'][:,8];m=a['z_stats'][:,10];eta=r.get('eta_m',0.)
    high=[(lo*.01,hi*.01) for lo,hi in intervals(e>=200) if hi-lo>=20]
    quiet=[(lo,hi) for lo,hi in intervals(e<5) if hi-lo>=2]
    episodes=[]
    for (_,start),(end,_) in zip(quiet[:-1],quiet[1:]):
        if end-start<2:continue
        peak=float(e[start:end].max())
        if peak>=20:episodes.append(dict(start_s=start*.01,end_s=end*.01,peak_hz=peak))
    late=(zt>=16)&(zt<26);late_e=(t>=16)&(t<26)
    slope=float(linregress(zt[late],z[late]).slope)
    blocks=[float(z[(zt>=lo)&(zt<hi)].mean()) for lo,hi in [(16,21),(21,26)]]
    late_stats=dict(E_mean_hz=float(e[late_e].mean()),quiet_bin_fraction=float(np.mean(e[late_e]<5)),
        high_bin_fraction=float(np.mean(e[late_e]>=200)),Z_mean=float(z[late].mean()),
        Z_sd=float(z[late].std()),Z_slope_per_s=slope,Z_block_means=blocks,
        Z_block_change=blocks[1]-blocks[0],mean_M=float(m[late].mean()),
        mean_adaptation_current=float(eta*m[late].mean()),
        depletion_duty=float(q[late].mean()),depletion=float((1-z[late]).mean()),
        mean_native_Z_drift_per_s=float(np.mean((1-q[late]-z[late])/5.)),
        finite_window_balance_diagnostic=bool(abs(slope*10)<.03 and abs(blocks[1]-blocks[0])<.03),
        self_limited_episodes=int(sum(v['start_s']>=16 for v in episodes)))
    m_error=None
    if r.get('M_enabled',False):
        increments=a['rate_e_hz']*dt/1000
        post=lfilter([1.],[1.,-(1-dt/r['tau_adp_ms'])],increments)
        expected=np.r_[0.,post[:-1]][np.rint(a['z_time_ms']/dt).astype(int)]
        m_error=float(np.max(abs(expected-m)))
        assert m_error<1e-9,m_error
    return dict(first_trigger_s=None if r['first_trigger_ms'] is None else r['first_trigger_ms']/1000,
        high_intervals_s=high,n_self_limited_episodes=len(episodes),episodes=episodes,
        late_16_26s=late_stats,refill_start_s=None if r['restore_start_ms'] is None else r['restore_start_ms']/1000,
        end_Z=float(z[-1]),max_mean_M=float(m.max()),max_mean_adaptation_current=float((eta*m).max()),
        M_reconstruction_max_error=m_error)


def snapshots(a):
    e=a['rate_e_hz'].reshape(-1,100).mean(1);t=(np.arange(len(e))+.5)*.01
    peaks=find_peaks(e,height=20,distance=8)[0]
    output=[];used=set()
    for target in ANCHORS:
        candidates=[i for i in peaks if abs(t[i]-target)<=.2 and i not in used]
        if candidates:
            idx=min(candidates,key=lambda i:abs(t[i]-target));used.add(idx);center=float(t[idx])
        else:center=target
        ms=round(center*1000);output.append(dict(time_s=center,lo_ms=ms-25,hi_ms=ms+25))
    return output


def markings(ax,r,snaps):
    ax.set(xlim=(0,26),xticks=[0,5,10,15,20,25])
    for k,s in enumerate(snaps,1):
        ax.axvline(s['time_s'],c='#575757',ls=':',lw=.8,alpha=.7)
    if r['restore_start_ms'] is not None:
        start=r['restore_start_ms']/1000;end=r['release_ms']/1000
        ax.axvspan(start,end,color='#2c8a70',alpha=.13,lw=0)
        ax.axvline(end,color='#2c8a70',ls='--',lw=1)


def left(fig,spec,a,r,name,reference,gain,mmax,snapshot_override=None):
    gs=spec.subgridspec(5,1,height_ratios=[1.22,1.22,1.35,.10,.86],hspace=.36)
    snaps=snapshots(a) if snapshot_override is None else snapshot_override
    t=a['lfp_time_ms']/1000;contacts=source.choose_contacts(reference)
    ref_t=reference['lfp_time_ms']/1000
    offset=np.arange(len(contacts))[::-1]
    center=np.median(reference['lfp_effective'][(ref_t>=.5)&(ref_t<1)][:,contacts],axis=0)
    displayed=(a['lfp_effective'][:,contacts]-center)/gain*.82+offset
    axes=[];ax=fig.add_subplot(gs[0]);axes.append(ax)
    for k,j in enumerate(contacts):
        color='#925227' if str(a['shaft_ids'][j]) in ('ICL','0','A') else '#1b626b'
        ax.plot(t,displayed[:,k],c=color,lw=.9,rasterized=True)
    ax.set(yticks=offset,yticklabels=a['contact_names'][contacts],ylim=(-.35,8.35),
           ylabel='Virtual SEEG\ncurrent proxy (a.u.)')
    condition='M off' if name=='M_off' else f'M on: ηM = {r["eta_m"]:g}, τM = 2 s'
    ax.set_title('A  Unfiltered contact readout',loc='left',weight='bold',pad=34)
    ax.text(0,1.04,condition,transform=ax.transAxes,fontsize=16,color='#222222')
    markings(ax,r,snaps)
    for k,s in enumerate(snaps,1):
        ax.text(s['time_s'],.97,str(k),transform=ax.get_xaxis_transform(),va='top',ha='center',
                fontsize=15,bbox=dict(fc='white',ec='none',alpha=.9,pad=.3))
    if r['restore_start_ms'] is not None:
        ax.text(r['restore_start_ms']/1000,1.04,'Z refill → release',color='#267762',transform=ax.get_xaxis_transform(),fontsize=14)
    else:ax.text(1,1.04,'No Z refill',color='#267762',transform=ax.transAxes,ha='right',fontsize=15)
    ra=fig.add_subplot(gs[1]);axes.append(ra)
    chosen=np.r_[np.arange(0,60,3),np.arange(60,120,3),np.arange(120,240,6),np.arange(240,300,3)]
    matrix=a['sample_spikes'][:,chosen];st,sn=np.where(matrix)
    ra.scatter(st*.0001,sn,s=1.6,marker='.',c=np.where(chosen[sn]<240,'#235878','#955323'),
               linewidths=0,rasterized=True)
    ra.set(ylim=(-1,80),yticks=[10,30,50,70],yticklabels=['Core A E','Core B E','Other E','I'])
    for y in [19.5,39.5,59.5]:ra.axhline(y,c='#c3c3c3',lw=.6)
    ra.set_title('B  Continuous spike raster',loc='left',weight='bold',pad=12);markings(ra,r,snaps)
    sg=gs[2].subgridspec(2,1,hspace=.18,height_ratios=[1,1])
    za=fig.add_subplot(sg[0]);axes.append(za);zt=a['z_time_ms']/1000;zs=a['z_stats']
    za.fill_between(zt,zs[:,2],zs[:,4],color=REGCOLORS[0],alpha=.15,lw=0)
    for j,(col,label) in enumerate(zip(REGCOLORS,['All E','Core A','Core B'])):
        za.plot(zt,zs[:,[0,5,6][j]],color=col,lw=1.5 if j==0 else 1,label=label)
    za.set(ylabel='Resource Z',ylim=(.25,1.04),yticks=[.4,.7,1])
    za.set_title('C  Z and spike adaptation',loc='left',weight='bold',pad=12)
    markings(za,r,snaps);za.tick_params(labelbottom=False)
    za.legend(loc='lower left',ncol=3,fontsize=13,frameon=False,handlelength=1.4)
    ma=fig.add_subplot(sg[1]);axes.append(ma)
    if 'm_stats' in a:
        ms=a['m_stats'];ma.fill_between(zt,ms[:,2],ms[:,4],color=REGCOLORS[0],alpha=.15,lw=0)
        for j,c in enumerate(REGCOLORS):ma.plot(zt,ms[:,[0,5,6][j]],c=c,lw=1.5 if j==0 else 1)
    else:ma.plot(zt,np.zeros_like(zt),color=REGCOLORS[0],lw=1.5)
    ma.set(xlabel='Time, t (s)',ylabel='Adaptation M\n(spike trace)',ylim=(-.03*mmax,mmax))
    markings(ma,r,snaps)
    head=fig.add_subplot(gs[3]);head.axis('off');head.text(0,.3,'D  Spatial activity (50-ms windows)',fontsize=18,weight='bold')
    maps=gs[4].subgridspec(1,5,wspace=.26)
    for k,s in enumerate(snaps):
        ax=fig.add_subplot(maps[k]);field=a['field_e_count_1ms'][s['lo_ms']:s['hi_ms']].sum(0)/a['cell_e_counts']/.05
        im=ax.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],
                     cmap=ACTIVITY_CMAP,norm=PowerNorm(.6,vmin=0,vmax=500),interpolation='nearest')
        for idx,xy in enumerate(a['centers_mm']):
            ax.add_patch(Circle(xy,1.5,fc='none',ec='#40d1ce',lw=1.2))
            if k==0:ax.text(xy[0],xy[1]+2,'AB'[idx],ha='center',color='#18666a',fontsize=13,
                           bbox=dict(fc='white',ec='none',alpha=.85,pad=.3))
        ax.set(xticks=[0,20],yticks=[0,20],xlabel='x (mm)')
        if k==0:ax.set_ylabel('y (mm)')
        else:ax.set_yticklabels([])
        ax.set_title(f'{k+1}   {s["time_s"]:.2f} s',fontsize=15,pad=10)
        if k==4:
            cax=ax.inset_axes([1.05,0,.045,1]);cb=fig.colorbar(im,cax=cax,ticks=[0,250,500]);cb.set_label('E rate (Hz)',fontsize=15);cax.tick_params(labelsize=14)
    bounds=np.array([[ax.get_position().x0,ax.get_position().x1] for ax in axes])
    assert np.allclose(bounds,bounds[0],atol=1e-12)
    assert int(matrix.sum())==len(st)
    rule='Nearest population E peak >=20Hz within +/-0.2s of fixed original anchor; fallback exact anchor. No spatial pattern selection.' if snapshot_override is None else 'Self-limited episodes nearest fixed 1,7,13,19,25s anchors, peak-centered; no spatial pattern selection. Fixed-anchor versions retained separately.'
    return dict(snapshot_windows=snaps,snapshot_rule=rule,
                raster_neurons=len(chosen),raster_spikes=len(st),temporal_thinning=False,
                axes_time_alignment_pass=True,readout_gain=gain,readout_center_reference='Original M-off 0.5-1 s')


def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True);fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.14)
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.14);plt.close(fig)


def main():
    datasets={name:load(name) for name in NAMES};ref=datasets['M_off'][0]
    lt=ref['lfp_time_ms']/1000;ids=source.choose_contacts(ref);raw=ref['lfp_effective'][:,ids]
    x=raw-np.median(raw[(lt>=.5)&(lt<1)],axis=0)
    gain=max(float(np.max(np.quantile(x,.995,axis=0)-np.quantile(x,.005,axis=0))),1e-12)
    mmax=max(np.max(a['m_stats'][:,[4,5,6]]) for a,r in datasets.values() if 'm_stats' in a)*1.08
    mmax=max(float(mmax),10.)
    metrics={name:analyze(*datasets[name]) for name in NAMES};layouts={}
    for name,(a,r) in datasets.items():
        fig=plt.figure(figsize=(14.4,16.7));spec=fig.add_gridspec(1,1,left=.14,right=.90,top=.95,bottom=.06)[0]
        layouts[name]=left(fig,spec,a,r,name,ref,gain,mmax);save(fig,name+'_left')
    # Event-centered companions show actual spatial episodes when the old high-state
    # sampling times fall in quiet gaps; the fixed-anchor comparison is retained.
    for name in ['m0p2_native','m0p8_native']:
        a,r=datasets[name];e=a['rate_e_hz'].reshape(-1,100).mean(1)
        events=metrics[name]['episodes'];times=[]
        for event in events:
            lo=round(event['start_s']*100);hi=round(event['end_s']*100)
            times.append((lo+int(np.argmax(e[lo:hi]))+.5)*.01)
        used=set();snaps=[]
        for target in [1.,7.,13.,19.,25.]:
            k=min((k for k in range(len(times)) if k not in used),key=lambda k:abs(times[k]-target))
            used.add(k);ms=round(times[k]*1000);snaps.append(dict(time_s=times[k],lo_ms=ms-25,hi_ms=ms+25))
        fig=plt.figure(figsize=(14.4,16.7));spec=fig.add_gridspec(1,1,left=.14,right=.90,top=.95,bottom=.06)[0]
        layouts[name+'_events']=left(fig,spec,a,r,name,ref,gain,mmax,snapshot_override=snaps)
        save(fig,name+'_events_left')
    for names,output in [(['M_off','m0p2_refill'],'M_off_vs_M_on_same_refill'),
                         (['m0p2_native','m0p8_native'],'M_on_without_Z_refill')]:
        fig=plt.figure(figsize=(29,17));gs=fig.add_gridspec(1,2,left=.065,right=.935,top=.95,bottom=.06,wspace=.29)
        for k,name in enumerate(names):left(fig,gs[k],*datasets[name],name,ref,gain,mmax)
        save(fig,output)
    fig,axs=plt.subplots(4,1,figsize=(13,12),sharex=True,gridspec_kw={'hspace':.13})
    for name,c in zip(NAMES,COLORS):
        if name=='m0p2_refill':continue  # exactly identical: refill never triggered
        a,r=datasets[name];t=a['z_time_ms']/1000;zs=a['z_stats'];eta=r.get('eta_m',0.)
        label='M off · refill' if name=='M_off' else f'ηM={eta:g} · '+('refill' if r['job']['refill'] else 'no refill')
        axs[0].plot(t,zs[:,0],c=c,lw=1.5,label=label)
        axs[1].plot(t,eta*zs[:,10],c=c,lw=1.3)
        axs[2].plot(t,uniform_filter1d(zs[:,8]-(1-zs[:,0]),200,mode='nearest'),c=c,lw=1.3)
        e=a['rate_e_hz'].reshape(-1,100).mean(1);te=(np.arange(len(e))+.5)*.01
        axs[3].plot(te,uniform_filter1d(e,50,mode='nearest'),c=c,lw=1.2)
    axs[0].legend(ncol=2,fontsize=13,frameon=False);axs[0].set_ylabel('Mean Z')
    axs[1].set_ylabel('Adaptation current\n(mV equiv.)')
    axs[2].set_ylabel('Depletion excess\n'+r'$q-(1-\overline{Z})$');axs[2].axhline(0,c='black',lw=.8,ls='--')
    axs[3].set_ylabel('E rate (Hz)\n0.5-s mean');axs[3].set_xlabel('Time (s)');axs[3].set(xlim=(0,26),xticks=[0,5,10,15,20,25])
    fig.subplots_adjust(left=.18,right=.96,top=.98,bottom=.07);save(fig,'M_Z_activity_comparison')
    # Matched M-on interventions must agree until the intervention actually starts.
    aa,ra=datasets['m0p2_refill'];bb,rb=datasets['m0p2_native']
    split=ra['restore_start_ms'];steps=260000 if split is None else round(split/.1)
    assert np.array_equal(aa['sample_spikes'][:steps],bb['sample_spikes'][:steps])
    assert np.array_equal(aa['rate_e_hz'][:steps],bb['rate_e_hz'][:steps])
    write('analysis_summary.json',dict(runs=metrics,paired_prefix_steps=steps,paired_prefix_spikes_exact=True,
        balance_rule='Finite-window diagnostic only: over16-26s abs(linear drift*10s)<0.03 and abs(mean Z21-26 minus16-21)<0.03. Not an attractor proof.',
        episode_rule='Contiguous activity between >=20ms quiet intervals (<5Hz E); duration>=20ms and peak>=20Hz. Operational self-limited population episodes, not patient TA/TB labels.',
        statistical_unit='One paired noise seed, one topology, three new trajectories',
        layouts=layouts,human_acceptance='PENDING_USER_REVIEW'))
    readme=''
    for name in NAMES:
        readme+=f'''### {name}_left.png / .pdf
同一手放双核底物的连续26秒SEEG、固定80个神经元raster、逐神经元Z/M摘要以及5个50毫秒空间快照。所有版本使用M-off参照的同一读出增益和空间色标；绿色带仅表示实际发生的外部Z补回。
**关注点**：M有每次spike加1的原生更新，不把零系数当作开启；M-off对照来自既有运行。数字仅索引采样时窗，不强行指称所有版本都有相同动力学阶段。

'''
    readme+='''### M_off_vs_M_on_same_refill.png / .pdf
左为既有M-off，右为M开启ηM=0.2、τM=2秒，使用相同的首次高态后500毫秒启动1秒补回规则。若未达到高态判据，规则不会触发，图中不添加虚构补回。
**关注点**：除了M外底物和输入均固定；外部操作只补Z，M在补回期间仍自然演化。

### M_on_without_Z_refill.png / .pdf
两档ηM=0.2、0.8均不施加任何外部Z补回，观察相同26秒内的事件、Z/M和原生空间活动。τM固定2秒，保留相同噪声和所有快状态连续演化。
**关注点**：这是剂量诊断，未按患者数据拟合；无进入高态只说明本观察窗，Z近似平台也不等于已证明稳定吸引子。

### M_Z_activity_comparison.png / .pdf
以统一时间轴比较平均Z、真实适应电流ηM×M、耗竭超额q−(1−平均Z)以及0.5秒平滑E放电率。耗竭超额为正时原生Z平均导数为负，第三行仅平滑显示，指标来自未平滑值。
**关注点**：减少高率同时保留自限事件才有助于稳定间期工作点；不能把整体静默或适应后的高率平台误称为间期恢复。

### m0p2_native_events_left.png / .pdf
同一ηM=0.2轨迹的事件中心快照版，A/B/C连续数据不变，D选取最接近1、7、13、19、25秒的自限活动峰值。采样时刻同时标在上方时间轴，未按传播方向或空间形状挑选。
**关注点**：该显示修订在发现原高态采样时刻落入静息间隔后增加；原固定锚点版本保留，可直接核对。数字仅为事件示例编号。

### m0p8_native_events_left.png / .pdf
同一ηM=0.8轨迹的事件中心快照版，沿用相同的均匀时间覆盖规则及原生50毫秒放电图。所有信号增益与M-off、较弱M版本一致。
**关注点**：用于显示更强适应下仍存在的自限活动，而不是患者TA/TB恢复验收。
'''
    (FIG/'README.md').write_text(readme)
    print(json.dumps({k:{x:y for x,y in v.items() if x!='episodes'} for k,v in metrics.items()},indent=2),flush=True)


if __name__=='__main__':main()
