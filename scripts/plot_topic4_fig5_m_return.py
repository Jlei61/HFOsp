#!/usr/bin/env python3
"""Report the bounded native M/Z timing assay without selecting desired outcomes."""
from pathlib import Path
import json
import hashlib
import numpy as np
from scipy.signal import lfilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
import plot_topic4_fig5_manual_release as original
from plot_topic4_fig5_manual_release_layout_v3 import ACTIVITY_CMAP

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1'
OUT=BASE/'m_runaway_return_v1';FIG=OUT/'figures'
NAMES=['m0_native','weak_fast','weak_20s','matched_gain_80s','slow_80s','slow_200s']
COLORS=['#252525','#687747','#b57524','#478a9b','#923b5d','#563a79']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.labelsize':18,
    'xtick.labelsize':16,'ytick.labelsize':16,'axes.titlesize':18,'pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False})

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def spans(mask):
    c=np.diff(np.r_[False,mask,False].astype(int))
    return [(int(a),int(b)) for a,b in zip(np.flatnonzero(c==1),np.flatnonzero(c==-1))]

def load(name):
    path=OUT/'runs'/name
    with np.load(path.with_suffix('.npz')) as f:a={k:f[k] for k in f.files}
    return a,json.loads(path.with_suffix('.json').read_text())

def analyze(a,r):
    dt=float(a['dt_ms']);rate=a['rate_e_hz'].reshape(-1,round(10/dt)).mean(1)
    high=[(lo,hi) for lo,hi in spans(rate>=200) if hi-lo>=20]
    high_mask=np.zeros(len(rate),bool)
    for lo,hi in high:high_mask[lo:hi]=True
    quiet=[(lo,hi) for lo,hi in spans(rate<5) if hi-lo>=2]
    events=[]
    for (_,lo),(hi,_) in zip(quiet[:-1],quiet[1:]):
        if hi-lo>=2 and rate[lo:hi].max()>=20 and not high_mask[lo:hi].any():
            peak=lo+int(np.argmax(rate[lo:hi]))
            events.append(dict(start_s=lo*.01,end_s=hi*.01,peak_s=(peak+.5)*.01,peak_hz=float(rate[peak])))
    regions=a['region_spikes_1ms'].reshape(-1,10,6).sum(1)/a['region_counts'][None,:]/.01
    recovery=None
    if high:
        for lo in range(((high[0][1]+9)//10)*10,len(rate)-200+1,10):
            x=rate[lo:lo+200].reshape(2,100)
            if high_mask[lo:lo+200].any():continue
            if np.all(x.mean(1)<50) and np.all((x<5).mean(1)>=.2):
                recovery=lo*.01;break
    zt=a['z_time_ms']/1000;z=a['z_stats'][:,0];m=a['m_stats'][:,0]
    post=None
    if recovery is not None:
        end=min(recovery+10,len(rate)*.01);sel=(zt>=recovery)&(zt<end)
        lo=round(recovery*100);hi=round(end*100)
        pre=(zt>=high[0][0]*.01)&(zt<recovery)
        region_quiet=(regions[lo:hi,:3]<5).mean(0)
        post=dict(start_s=recovery,end_s=end,
            E_mean_hz=float(rate[lo:hi].mean()),quiet_fraction=float((rate[lo:hi]<5).mean()),
            core_A_quiet_fraction=float(region_quiet[0]),core_B_quiet_fraction=float(region_quiet[1]),
            surround_quiet_fraction=float(region_quiet[2]),
            finite_event_count=sum(recovery<=v['start_s']<end for v in events),
            Z_at_recovery=float(np.interp(recovery,zt,z)),Z_10s_max=float(z[sel].max()),
            Z_increase_from_high_min=float(z[sel].max()-z[pre].min()),
            adaptation_at_recovery=float(r['eta_m']*np.interp(recovery,zt,m)),
            adaptation_at_end=float(r['eta_m']*np.interp(end,zt,m)),
            region_E_means=regions[lo:hi,:3].mean(0).tolist())
    pre_events=sum(v['end_s']<=high[0][0]*.01 for v in events) if high else len(events)
    cycle=bool(high and post and post['finite_event_count']>=2 and pre_events>=2 and
               post['core_A_quiet_fraction']>=.2 and post['core_B_quiet_fraction']>=.2 and
               post['Z_increase_from_high_min']>=.05)
    classification=('ENTRY_RETURN_WITH_EVENTS' if cycle else
        'GLOBAL_LOW_ACTIVITY_RETURN_NEEDS_REGIONAL_OR_EVENT_REVIEW' if recovery is not None else
        'HIGH_ENTRY_WITHOUT_RECOVERY_WITHIN_HORIZON' if high else 'NO_HIGH_ENTRY_WITHIN_HORIZON')
    post_m=lfilter([1.],[1.,-(1-dt/r['tau_adp_ms'])],a['rate_e_hz']*dt/1000)
    expected=np.r_[0.,post_m[:-1]][np.rint(a['z_time_ms']/dt).astype(int)]
    merror=float(abs(expected-m).max());assert merror<1e-8
    late=(zt>=zt[-1]-10);nr=min(1000,len(rate));tail=rate[-nr:]
    return dict(name=r['job']['name'],eta_m=r['eta_m'],tau_M_s=r['tau_adp_ms']/1000,
        K_M=r['eta_m']*r['tau_adp_ms']/1000,duration_s=len(rate)*.01,
        classification=classification,full_cycle_candidate=cycle,
        high_intervals_s=[[lo*.01,hi*.01] for lo,hi in high],
        high_time_total_s=float(high_mask.sum()*.01),
        first_high_start_s=high[0][0]*.01 if high else None,
        first_high_confirmed_s=r['first_trigger_ms']/1000 if r['first_trigger_ms'] is not None else None,
        first_low_return_s=recovery,post_return=post,pre_high_finite_events=pre_events,
        n_finite_events=len(events),events=events,
        late_10s=dict(E_mean_hz=float(tail.mean()),quiet_fraction=float((tail<5).mean()),
            Z_mean=float(z[late].mean()),Z_end=float(z[-1]),
            Z_slope_per_s=float(np.polyfit(zt[late]-zt[late][0],z[late],1)[0]),
            adaptation_current_mean=float(r['eta_m']*m[late].mean()),
            regional_E_mean=regions[-nr:,:3].mean(0).tolist()),
        max_mean_adaptation_current=float((r['eta_m']*m).max()),M_reconstruction_error=merror,
        no_intervention=bool(r['restore_start_ms'] is None and not a['z_rhs'][:,1].any()),
        baseline_rate_prefix=r.get('baseline_rate_prefix'),external_prefix_match=r['external_input_prefix_matches_M_off'])

def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/(name+'.png'),dpi=150,bbox_inches='tight',pad_inches=.12)
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.12);plt.close(fig)

def overview(datasets,metrics):
    fig,axes=plt.subplots(3,6,figsize=(27,10.2),sharex=True,sharey='row',gridspec_kw={'hspace':.16,'wspace':.18})
    for col,name in enumerate(NAMES):
        a,r=datasets[name];m=metrics[name];x=a['rate_e_hz'].reshape(-1,1000).mean(1)
        t=(np.arange(len(x))+.5)*.1;zt=a['z_time_ms']/1000
        axes[0,col].plot(t,x,lw=.85,color=COLORS[col]);axes[0,col].set(ylim=(0,520))
        axes[1,col].plot(zt,a['z_stats'][:,0],color=COLORS[col],lw=1.5);axes[1,col].set(ylim=(0,1.04))
        axes[2,col].plot(zt,r['eta_m']*a['m_stats'][:,0],color=COLORS[col],lw=1.5)
        for row in range(3):
            ax=axes[row,col];ax.set(xlim=(0,90),xticks=[0,30,60,90])
            for lo,hi in m['high_intervals_s']:ax.axvspan(lo,hi,color='#b9443f',alpha=.10,lw=0)
            if m['first_low_return_s'] is not None:ax.axvline(m['first_low_return_s'],color='#26806d',ls='--',lw=1.1)
        axes[0,col].set_title(f'ηM = {r["eta_m"]:g}\nτM = {r["tau_adp_ms"]/1000:g} s',fontsize=17,pad=12)
        axes[2,col].set_xlabel('Time (s)')
    for row,label in enumerate(['E rate (Hz)\n100-ms bins','Mean resource Z','Adaptation current\nηM × mean M']):axes[row,0].set_ylabel(label)
    fig.subplots_adjust(left=.065,right=.99,bottom=.09,top=.91)
    save(fig,'M_timing_native_comparison')

def snapshots(a,metric):
    events=metric['events'];dur=metric['duration_s'];high=metric['first_high_start_s'];ret=metric['first_low_return_s']
    if high is not None:
        before=[v for v in events if v['end_s']<high]
        one=before[len(before)//2]['peak_s'] if before else max(.05,high-1)
        end=ret if ret is not None else min(dur,high+4)
        mid=min(high+.5,(high+end)/2)
        targets=[one,max(.025,high-.15),mid,ret+.5 if ret is not None else dur*.65,dur-1]
        labels=['Finite event','Before high','High activity','After return' if ret is not None else 'Later','Late']
        if ret is not None:
            after=[v for v in events if v['start_s']>=ret]
            if after:targets[3]=after[0]['peak_s']
            if after:targets[4]=after[-1]['peak_s']
    else:
        targets=np.linspace(1,dur-1,5).tolist();labels=['Event']*5
        used=set()
        for k,tar in enumerate(targets):
            pool=[(i,v) for i,v in enumerate(events) if i not in used]
            if pool:
                idx,event=min(pool,key=lambda iv:abs(iv[1]['peak_s']-tar));used.add(idx);targets[k]=event['peak_s']
    return [dict(time_s=float(np.clip(t,.025,dur-.025)),label=label) for t,label in zip(targets,labels)]

def left(a,r,metric,reference,gain,time_limits=None,snapshot_override=None,contact_indices=None):
    name=metric['name'];dur=metric['duration_s'];snaps=snapshots(a,metric) if snapshot_override is None else snapshot_override
    xlim=(0,dur) if time_limits is None else tuple(time_limits)
    fig=plt.figure(figsize=(15.2,17 if contact_indices is None else 20));gs=fig.add_gridspec(5,1,left=.145,right=.90,top=.95,bottom=.07,
        height_ratios=[1.22 if contact_indices is None else 1.8,1.22,1.36,.10,.82],hspace=.36)
    axes=[];ax=fig.add_subplot(gs[0]);axes.append(ax)
    ids=original.choose_contacts(reference) if contact_indices is None else np.asarray(contact_indices,int);rt=reference['lfp_time_ms']/1000
    center=np.median(reference['lfp_effective'][(rt>=.5)&(rt<1)][:,ids],axis=0)
    lt=a['lfp_time_ms']/1000
    visible=np.ones(len(lt),bool) if time_limits is None else (lt>=xlim[0])&(lt<xlim[1])
    x=(a['lfp_effective'][visible][:,ids]-center)/gain*.82+np.arange(len(ids))[::-1];lt=lt[visible]
    for j,i in enumerate(ids):ax.plot(lt,x[:,j],c='#315f72' if j<4 else '#956537',lw=.65,rasterized=True)
    ax.set(yticks=np.arange(len(ids))[::-1],yticklabels=a['contact_names'][ids],ylim=(-.4,len(ids)+.4),ylabel='Virtual SEEG\ncurrent proxy (a.u.)')
    ax.set_title('A  Unfiltered contact readout',loc='left',weight='bold',pad=33)
    ax.text(0,1.03,f'ηM = {r["eta_m"]:g}, τM = {r["tau_adp_ms"]/1000:g} s',transform=ax.transAxes,fontsize=16)
    ax.text(1,1.03,'Native Z + M; no external reset',transform=ax.transAxes,ha='right',fontsize=16)
    ra=fig.add_subplot(gs[1]);axes.append(ra)
    chosen=np.r_[np.arange(0,60,3),np.arange(60,120,3),np.arange(120,240,6),np.arange(240,300,3)]
    ts,ns=np.where(a['sample_spikes'][:,chosen])
    if time_limits is not None:
        keep=(ts*.0001>=xlim[0])&(ts*.0001<xlim[1]);ts,ns=ts[keep],ns[keep]
    # Retain every spike for these fixed80 sampled cells; no temporal subsampling.
    for lo,hi,color in [(0,60,'#275d80'),(60,80,'#a36023')]:
        sel=(ns>=lo)&(ns<hi);ra.scatter(ts[sel]*.0001,ns[sel],s=1.6,c=color,marker='.',linewidths=0,rasterized=True)
    ra.set(ylim=(-1,80),yticks=[10,30,50,70],yticklabels=['Core A E','Core B E','Other E','I'])
    for y in [19.5,39.5,59.5]:ra.axhline(y,c='#bdbdbd',lw=.7)
    ra.set_title('B  Continuous spike raster',loc='left',weight='bold',pad=12)
    sub=gs[2].subgridspec(2,1,hspace=.18);za=fig.add_subplot(sub[0]);ma=fig.add_subplot(sub[1]);axes.extend([za,ma])
    zt=a['z_time_ms']/1000;zs=a['z_stats'];ms=a['m_stats']
    if time_limits is not None:
        visible=(zt>=xlim[0])&(zt<xlim[1]);zt=zt[visible];zs=zs[visible];ms=ms[visible]
    za.fill_between(zt,zs[:,2],zs[:,4],color='#63427e',alpha=.16,lw=0)
    ma.fill_between(zt,r['eta_m']*ms[:,2],r['eta_m']*ms[:,4],color='#63427e',alpha=.16,lw=0)
    for idx,color,label in [(0,'#63427e','All E'),(5,'#ab3d70','Core A'),(6,'#2581a6','Core B')]:
        za.plot(zt,zs[:,idx],c=color,lw=1.3,label=label)
        ma.plot(zt,r['eta_m']*ms[:,idx],c=color,lw=1.3)
    za.set(ylabel='Resource Z',ylim=(0,1.04),yticks=[0,.5,1]);za.set_title('C  Inhibition resource and adaptation',loc='left',weight='bold',pad=13)
    za.legend(loc='lower left',ncol=3,fontsize=13,frameon=False,handlelength=1.3)
    ma.set(ylabel='ηM × M\n(mV-equivalent)',xlabel='Time, t (s)');ma.set_ylim(bottom=0,top=max(1,ma.get_ylim()[1]))
    for axis in axes:
        axis.set_xlim(*xlim)
        if time_limits is None:axis.set_xticks(np.arange(0,dur+1,5 if dur<=30 else 15))
        else:axis.set_xticks(np.arange(np.ceil(xlim[0]),xlim[1]+.001,1.))
        for lo,hi in metric['high_intervals_s']:axis.axvspan(lo,hi,color='#b9443f',alpha=.11,lw=0)
        if time_limits is not None and metric.get('first_high_confirmed_s') is not None:
            axis.axvline(metric['first_high_confirmed_s'],c='#aa3043',ls='--',lw=1.1)
        if metric['first_low_return_s'] is not None:axis.axvline(metric['first_low_return_s'],c='#26806d',ls='--',lw=1)
        for k,s in enumerate(snaps,1):axis.axvline(s['time_s'],c='#555555',ls=':',lw=.85,alpha=.65)
        if axis is not ma:axis.tick_params(labelbottom=False)
    for k,s in enumerate(snaps,1):ax.text(s['time_s'],.98,str(k),transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=15,bbox=dict(fc='white',ec='none',pad=.3,alpha=.8))
    h=fig.add_subplot(gs[3]);h.axis('off');h.text(0,.3,'D  Native spatial activity (50-ms windows)',weight='bold',fontsize=18)
    maps=gs[4].subgridspec(1,5,wspace=.29)
    for k,s in enumerate(snaps):
        axis=fig.add_subplot(maps[k]);lo=round(s['time_s']*1000)-25;hi=lo+50
        field=a['field_e_count_1ms'][lo:hi].sum(0)/a['cell_e_counts']/.05
        im=axis.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],interpolation='nearest',cmap=ACTIVITY_CMAP,norm=PowerNorm(.6,vmin=0,vmax=500))
        for xy in a['centers_mm']:axis.add_patch(Circle(xy,1.5,fc='none',ec='#41cecd',lw=1.25))
        axis.set(xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)');axis.tick_params(labelsize=14)
        if k==0:axis.set_ylabel('y (mm)')
        else:axis.set_yticklabels([])
        axis.set_title(f'{k+1}  {s["time_s"]:.2f} s',fontsize=15,pad=9)
        s.update(lo_ms=lo,hi_ms=hi)
        if k==4:
            ca=axis.inset_axes([1.05,0,.05,1]);cb=fig.colorbar(im,cax=ca,ticks=[0,250,500]);cb.set_label('E rate (Hz)',fontsize=16);ca.tick_params(labelsize=14)
    bounds=np.array([[ax.get_position().x0,ax.get_position().x1] for ax in axes]);assert np.allclose(bounds,bounds[0])
    save(fig,name+'_left')
    return dict(snapshots=snaps,shared_time_axis=True,time_limits_s=list(xlim),readout_gain=gain,fixed_raster_neurons=80,contact_names_displayed=a['contact_names'][ids].tolist(),
                raster_spikes=int(len(ts)),temporal_thinning=False,spatial_circle_radius_mm=1.5)

def main(available=False):
    global NAMES
    if available:
        NAMES=[n for n in NAMES if (OUT/"runs"/(n+".json")).exists()]
    if not NAMES:
        return
    datasets={n:load(n) for n in NAMES};metrics={n:analyze(*datasets[n]) for n in NAMES}
    write(OUT/'analysis_summary.json',metrics)
    if len(NAMES)==6:overview(datasets,metrics)
    with np.load(BASE/'runs/continuous_refill_release.npz') as f:ref={k:f[k] for k in f.files}
    lt=ref['lfp_time_ms']/1000;ids=original.choose_contacts(ref);raw=ref['lfp_effective'][:,ids]
    x=raw-np.median(raw[(lt>=.5)&(lt<1)],axis=0)
    gain=float(np.max(np.quantile(x,.995,axis=0)-np.quantile(x,.005,axis=0)))
    qa={}
    prior=json.loads((OUT/'artifact_qa.json').read_text())['layouts'] if (OUT/'artifact_qa.json').exists() else {}
    for name in NAMES:
        if name in prior and (FIG/(name+'_left.png')).exists():qa[name]=prior[name]
        else:qa[name]=left(*datasets[name],metrics[name],ref,gain)
    write(OUT/'artifact_qa.json',dict(status='PASS',metrics_reconstructed=True,layouts=qa,user_visual_review='PENDING'))
    cycles=[n for n in NAMES if metrics[n]['full_cycle_candidate']]
    entered=[n for n in NAMES if metrics[n]['first_high_start_s'] is not None]
    no_entry=[n for n in NAMES if metrics[n]['first_high_start_s'] is None]
    finding=(f'当前已有{len(NAMES)}/6条完成。满足本轮进入—恢复并再现有限事件规则的候选：'+(', '.join(cycles) if cycles else '尚无')+'。')
    if no_entry:finding+=' 未进入高态：'+', '.join(no_entry)+'；这是其观察窗内的结果，不能推成无限时长结论。'
    if entered:finding+=' 已进入高态：'+', '.join(entered)+'；须结合恢复列区分持续高态与真正返回。'
    rows=['# Native M timing assay','',finding,'',
        '历史手放双核与既有C快网络固定，原生全局/空间OU及Poisson输入连续。只改变E细胞M的增量与消退时间，全程没有Z回填、clamp或状态重置。参数是探索值，尚未拟合患者。',
        '', 'M电流为ηM×M；K=ηM×τM(s)是定常每Hz适应增益。相同ηM的不同τM，初期单次放电效应相同，但长期积累强度不同；weak_20s和matched_gain_80s则K相同。','',
        '| 条件 | ηM | τM(s) | K | 首次高态确认(s) | 首次低活动恢复(s) | 最后10s E(Hz) | 分类 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for n in NAMES:
        m=metrics[n];rows.append(f'| {n} | {m["eta_m"]:g} | {m["tau_M_s"]:g} | {m["K_M"]:g} | {m["first_high_confirmed_s"]} | {m["first_low_return_s"]} | {m["late_10s"]["E_mean_hz"]:.2f} | {m["classification"]} |')
    rows+=['','高态沿用全E10ms分箱≥200Hz持续200ms。恢复要求至少2秒，两个1秒均值都<50Hz且低于5Hz分箱至少20%，期间无合格高态段；另外分别检查两核低活动间隔、恢复后有限事件和Z回升。下穿200Hz并不算恢复。',
        '', '单拓扑、单配对噪声；事件数是网络内部事件，不能当作独立模型样本。M电流关闭对照只观察30秒，M组90秒，未发生转变均是该观察窗内未见。强度降低/恢复的时间关联不能单独证明唯一失稳类型；这里不宣称Hopf或临床发作。',
        '', 'A为原空间加权电流proxy，保持旧增益与中心，不是新构造的生理EEG；B保留固定80神经元的每一个spike；C画Z和实际ηM×M（带宽为E细胞10–90%分位）；D直接由全体E神经元二维放电计数生成。红色底纹为操作性高态，绿虚线是首个低活动恢复窗起点。',
        '', '所有图已程序自查，仍待用户目视审阅；不替换正式图，不自动启动下一轮。']
    (OUT/'scientific_review.md').write_text('\n'.join(rows)+'\n')
    lines=['# 图件说明','']
    for n in NAMES:
        lines+=['### '+n+'_left.png',f'展示{n}条件的完整连续SEEG电流读出、原生raster、Z和实际M适应电流，以及五个50ms二维活动窗。保持真实1.5mm双核圈和统一空间色阶；无人工Z恢复。', '**关注点**：区分从未进入高态、进入后持续、低活动恢复，以及恢复后再次出现有限事件。','']
    if len(NAMES)==6:lines+=['### M_timing_native_comparison.png','六个条件分别展示E率、平均Z与平均M电流。红色区段按同一高态规则定义；无M电流对照只到30秒，其余到90秒。','**关注点**：同ηM改变τM，以及相同K改变积累速度的效果。','']
    (FIG/'README.md').write_text('\n'.join(lines))
    paths=[ROOT/'scripts/run_topic4_fig5_m_return.py',Path(__file__),ROOT/'src/topic4_raster_protocol_engine.py',ROOT/'.worktrees/topic4-substrate-autapse-fix/src/snn_engine/mz_slow_vars.py']
    paths.extend(p for p in [ROOT/'scripts/resume_topic4_fig5_m_return.py',ROOT/'scripts/supervise_topic4_fig5_m_return.py'] if p.exists())
    write(OUT/'producer_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    print(json.dumps({n:{k:v for k,v in m.items() if k not in ('events','high_intervals_s')} for n,m in metrics.items()},indent=2))

if __name__=='__main__':
    import argparse,fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--available',action='store_true');args=parser.parse_args()
    with (OUT/'analysis.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main(available=args.available)
