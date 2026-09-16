#!/usr/bin/env python3
"""Read-only scientific audit and figures for autonomous SNN recovery rounds."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,PowerNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks,welch
import run_topic4_autonomous_recovery as run

OUT=run.OUT

def read(path):return json.loads(path.read_text())
def write(path,data):run.base.write(path,data)

def load(folder,keys):
    out={k:[] for k in keys};expected=0
    for path in sorted((folder/'chunks').glob('*.npz')):
        if '.tmp.' in path.name:continue
        with np.load(path) as a:
            lo,hi=int(a['start_step']),int(a['end_step']);assert lo==expected,(path,expected,lo);expected=hi
            for key in keys:out[key].append(a[key])
    return {key:np.concatenate(v) for key,v in out.items()} if expected else None

def classify(folder):
    result=read(folder/'result.json') if (folder/'result.json').exists() else None
    keys=['time_ms','spikes_1ms','regions_1ms','field_5ms','slow_time_ms','Z','M','currents','regional_currents']
    a=load(folder,keys)
    if a is None:return None
    job=read(OUT/'jobs'/(folder.name+'.json'));geo=np.load(OUT/'geometry.npz');nr=geo['region_counts']
    counts=a['spikes_1ms'];regions=a['regions_1ms'];assert np.array_equal(counts[:,0],regions[:,:3].sum(1))
    assert np.array_equal(counts[:,1],regions[:,3:].sum(1))
    n=len(counts)//10
    rates=np.c_[counts[:n*10,0].reshape(n,10).sum(1)/32000/.01,
        regions[:n*10,:3].reshape(n,10,3).sum(1)/nr[:3]/.01]
    tr=run.fresh_tracker()
    for k,r in enumerate(rates):run.track(tr,r,(k+1)*.01)
    if result:
        assert tr['entries']==result['tracker']['entries']
        assert tr['recoveries']==result['tracker']['recoveries']
    kind='NO_HIGH_ENTRY'
    if tr['entries']:kind='HIGH_WITHOUT_VERIFIED_RECOVERY'
    if tr['recoveries']:kind='AUTONOMOUS_RETURN_NO_REENTRY'
    if len(tr['entries'])>=2 and tr['recoveries']:kind='AUTONOMOUS_RECURRENCE_CANDIDATE'
    tail=rates[-1000:];early=rates[50:500]
    modulation=(np.quantile(tail,.95,axis=0)-np.quantile(tail,.05,axis=0))/np.maximum(tail.mean(0),1e-9)
    rhythm=[]
    for i in range(4):
        freq,power=welch(tail[:,i],fs=100,nperseg=min(500,len(tail)),detrend='constant')
        band=(freq>=.5)&(freq<=40);total=float(np.trapz(power[band],freq[band]))
        rhythm.append(dict(envelope_band_Hz=[.5,40],envelope_power=total,
            peak_frequency_Hz=float(freq[band][np.argmax(power[band])]) if total>1e-12 else None,
            interpretation='Descriptive population envelope spectrum; noise-driven bursts or spatial cancellation can produce peaks, so not a limit-cycle test.'))
    nf=len(a['field_5ms'])//4
    f20=a['field_5ms'][:nf*4].reshape(nf,4,400).sum(1)/np.maximum(geo['cell_e_counts'],1)/.02
    area=(f20>=20).mean(1);r20=counts[:nf*20,0].reshape(nf,20).sum(1)/32000/.02
    broad=(area>=.5)&(r20>=50)
    edges=np.diff(np.r_[False,broad,False].astype(int))
    intervals=[dict(start_s=round(lo*.02,4),end_s=round(hi*.02,4),duration_s=round((hi-lo)*.02,4))
        for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if hi-lo>=50]
    stats=dict(name=folder.name,job=job,complete=result is not None,observed_s=n*.01,mode=kind,
        termination_status=result.get('status') if result else 'RUNNING',
        stop_reason=result.get('tracker',{}).get('stop_reason') if result else None,
        planned_horizon_s=job['horizon_s'],reached_planned_horizon=bool(n*.01>=job['horizon_s']-1e-9),
        entries=tr['entries'],recoveries=tr['recoveries'],no_intervention=True,
        tail10s_mean_Hz=tail.mean(0).tolist(),tail10s_quiet_fraction=(tail<5).mean(0).tolist(),
        early_mean_Hz=early.mean(0).tolist(),Z_final=a['Z'][-1].tolist(),M_feedback_final=(job['eta_m']*a['M'][-1]).tolist(),
        tail_modulation_p95_minus_p05_over_mean=modulation.tolist(),tail_envelope_spectrum=rhythm,
        broad_sustained_intervals=intervals,broad_definition='20ms bins: at least50%native1mm cells>=20Hz AND allE>=50Hz continuously1s. Independent diagnostic; not a replacement for the primary high-state criterion.',
        final_regional_currents=a['regional_currents'][-1].tolist(),counts_and_endpoint_audit='PASS',
        limit='Operational high/return detection, pending raster/native-field review; no claim of a bifurcation or patient seizure reproduction.')
    return stats

def summary():
    rows=[]
    for folder in sorted((OUT/'runs').glob('*')):
        if folder.name.startswith('qa_'):continue
        if (folder/'chunks').exists():
            v=classify(folder)
            if v:rows.append(v)
    write(OUT/'analysis.json',dict(updated_at=time.time(),rows=rows,completed=sum(v['complete'] for v in rows)))
    fig,axs=plt.subplots(1,3,figsize=(18,6),gridspec_kw={'width_ratios':[1.25,1,1]})
    labels=[r['name'].replace('_s9108401','') for r in rows];yy=np.arange(len(rows));colors={'NO_HIGH_ENTRY':'#727981','HIGH_WITHOUT_VERIFIED_RECOVERY':'#c7434b','AUTONOMOUS_RETURN_NO_REENTRY':'#e69c34','AUTONOMOUS_RECURRENCE_CANDIDATE':'#248671'}
    for i,r in enumerate(rows):
        col=colors[r['mode']];axs[0].plot([0,r['observed_s']],[i,i],color=col,lw=3)
        for v in r['entries']:axs[0].scatter(v['onset_s'],i,c='#c7434b',marker='^',s=55,zorder=3)
        for v in r['recoveries']:axs[0].scatter(v['confirmation_s'],i,c='#248671',marker='o',s=55,zorder=4)
        axs[1].scatter(r['tail10s_mean_Hz'][0],i,c=col,s=50);axs[1].scatter(max(r['tail10s_mean_Hz'][1:3]),i,edgecolors=col,facecolors='none',s=50)
        axs[2].scatter(r['Z_final'][0],i,c=col,s=50)
    for ax in axs:
        ax.set_yticks(yy);ax.invert_yaxis();ax.tick_params(labelsize=11);ax.spines[['top','right']].set_visible(False)
    axs[0].set_yticklabels(labels);axs[1].set_yticklabels([]);axs[2].set_yticklabels([])
    axs[0].set_xlabel('Time (s)',fontsize=16);axs[1].set_xlabel('Late E rate (Hz)',fontsize=16);axs[2].set_xlabel('Mean Z',fontsize=16)
    axs[1].set_xlim(0,550);axs[2].set_xlim(0,1.03)
    fig.tight_layout();d=OUT/'figures';d.mkdir(exist_ok=True)
    fig.savefig(d/'exploration_summary.png',dpi=180);fig.savefig(d/'exploration_summary.pdf');plt.close(fig)
    lines=['# 自主恢复：实时观测审阅','',f'已分析 {len(rows)} 条，完成 {sum(r["complete"] for r in rows)} 条。所有轨迹均无 Z/M reset、无时刻触发的外部抑制。','',
        '| 条件 | 观测时长(s) | 高态起点(s) | 自主恢复确认(s) | 末10s全E / 最强core均率(Hz) | 当前判定 |',
        '|---|---:|---|---|---|---|']
    for r in rows:
        lines.append(f'| {r["name"]} | {r["observed_s"]:.1f} | {[v["onset_s"] for v in r["entries"]]} | {[v["confirmation_s"] for v in r["recoveries"]]} | {r["tail10s_mean_Hz"][0]:.2f} / {max(r["tail10s_mean_Hz"][1:3]):.2f} | {r["mode"]} |')
    lines+=['','每个条件先用一条配对开发噪声定位机制；需要第二噪声确认。两秒低态是筛查门，不代替有限事件、Z重新积累和完整空间场验收。无进入只表示在保存时长内未达到固定200Hz/200ms定义，不是永不发作。','',
        '汇总图：红三角为高态进入，绿圆为全E与双核均通过的自主恢复；中图实心点为全E均率、空心点为较强核均率；右图为末时刻平均Z。']
    (OUT/'scientific_progress.md').write_text('\n'.join(lines)+'\n')
    (d/'README.md').write_text('### exploration_summary.png\n同一手放双核底物上，逐条件显示实际观测时长、高态进入和自主恢复位置，旁列末段全局/核内放电率与 Z。结果来自原生计数，未进入的轨迹保留完整观察上限。**关注点**：全局平均下降时，核是否仍持续放电；两秒返回检测尚需完整轨迹目视复核。\n\n### exploration_summary.pdf\n上述汇总的矢量版本，统计单位为一条固定参数和噪声轨迹。**关注点**：开发筛查与后续独立噪声确认需分开。\n')
    if len(rows)>=4 and all('pool_gain' in r['job'] for r in rows):
        global_feedback_comparison(rows)
        global_feedback_comparison(rows,joint=True)
    return rows

def global_feedback_comparison(rows,joint=False):
    if joint:
        selected=[r for r in rows if r['job']['pool_gain'] in [10,50] and r['job']['pool_threshold_Hz']==50
                  and r['job']['pool_tau_s']==2 and r['job']['seed']==9108401]
    else:
        selected=[r for r in rows if r['job']['pool_gain']==10 and r['job']['seed']==9108401
                  and r['job']['eta_m']==.0005 and r['job']['tau_M_s']==1]
    if len(selected)!=4 or min(r['observed_s'] for r in selected)<20:return
    selected.sort(key=lambda r:(r['job']['pool_gain'],r['job']['eta_m']) if joint else (r['job']['pool_threshold_Hz'],r['job']['pool_tau_s']))
    stem='global_M_joint_comparison' if joint else 'global_feedback_comparison'
    end=min(r['observed_s'] for r in selected)
    geo=np.load(OUT/'geometry.npz');nr=geo['region_counts'];centers=geo['centers_mm']
    fig,axs=plt.subplots(5,4,figsize=(21,17),gridspec_kw={'height_ratios':[1,1,1,1,1.2]})
    metadata=[]
    for col,r in enumerate(selected):
        folder=OUT/'runs'/r['name'];job=r['job']
        a=load(folder,['time_ms','spikes_1ms','regions_1ms','slow_time_ms','Z','currents','field_5ms','field_time_ms'])
        mask=a['time_ms']<=end*1000;ts=a['slow_time_ms']/1000;t=a['time_ms'][mask]/1000
        er=gaussian_filter1d(a['spikes_1ms'][mask,0]*1000/32000,3)
        axs[0,col].plot(t,er,c='#222222',lw=.8,label='All E')
        for i,color in enumerate(['#b778a5','#4b93b5']):
            y=gaussian_filter1d(a['regions_1ms'][mask,i]*1000/nr[i],3)
            axs[0,col].plot(t,y,c=color,lw=.6,alpha=.65,label=f'Core {"AB"[i]} E')
        axs[0,col].axhline(200,color='.5',ls=':',lw=.6);axs[0,col].set_ylim(-5,505)
        title=(f'κ = {job["pool_gain"]:g}; ηM = {job["eta_m"]:g}\nτM = {job["tau_M_s"]:g} s' if joint
               else f'r₀ = {job["pool_threshold_Hz"]:g} Hz; τG = {job["pool_tau_s"]:g} s')
        axs[0,col].set_title(title,fontsize=18)
        axs[1,col].plot(ts,a['Z'][:,0],c='#7e399d',lw=1.5)
        axs[1,col].fill_between(ts,a['Z'][:,2],a['Z'][:,4],color='#7e399d',alpha=.14)
        axs[1,col].set_ylim(0,1.04)
        q={k:[] for k in ['time_ms','raw_global_current','effective_global_current']}
        for path in sorted((folder/'pool_chunks').glob('*.npz')):
            if '.tmp.' in path.name:continue
            with np.load(path) as d:
                for k in q:q[k].append(d[k])
        q={k:np.concatenate(v) for k,v in q.items()}
        axs[2,col].plot(q['time_ms']/1000,q['raw_global_current'],c='#bc9870',lw=.8,label='Before Z')
        axs[2,col].plot(q['time_ms']/1000,q['effective_global_current'],c='#417f70',lw=1.3,label='After Z')
        axs[3,col].plot(ts,a['currents'][:,0],c='#c56350',lw=1,label='Excitatory')
        axs[3,col].plot(ts,a['currents'][:,2],c='#447ca4',lw=1,label='Applied inhibitory')
        for row in range(4):
            axs[row,col].set_xlim(0,end)
            if row<3:axs[row,col].tick_params(labelbottom=False)
            else:axs[row,col].set_xlabel('Time (s)')
            for event in r['entries']:
                if event['onset_s']<=end:axs[row,col].axvline(event['onset_s'],color='#bc3847',ls=':',lw=.9)
        ft=a['field_time_ms']/1000;chosen=(ft>end-.05)&(ft<=end)
        field=a['field_5ms'][chosen].sum(0)/geo['cell_e_counts']/(chosen.sum()*.005)
        im=axs[4,col].imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],vmin=0,vmax=500,cmap='magma')
        for center in centers:axs[4,col].add_patch(plt.Circle(center,1.5,fill=False,ec='#60d3d5',lw=1.2))
        axs[4,col].set_xlabel('x (mm)')
        metadata.append(dict(name=r['name'],window_s=[0,end],native_field_window_s=[end-.05,end],entries=r['entries'],recoveries=r['recoveries']))
    for row in [2,3]:
        upper=max(ax.get_ylim()[1] for ax in axs[row])
        for ax in axs[row]:ax.set_ylim(0,upper)
    for row,label in enumerate(['E rate (Hz)','Resource Z','Global inhibition\n(mV equiv.)','Input current\n(mV equiv.)','y (mm)']):axs[row,0].set_ylabel(label)
    for row in [0,2,3]:axs[row,-1].legend(frameon=False,fontsize=11,loc='upper right')
    for ax in axs.flat:ax.tick_params(labelsize=13);ax.xaxis.label.set_fontsize(17);ax.yaxis.label.set_fontsize(17)
    fig.subplots_adjust(left=.08,right=.93,bottom=.06,top=.95,wspace=.23,hspace=.24)
    cb=fig.add_axes([.95,.075,.012,.13]);fig.colorbar(im,cax=cb,label='E rate (Hz)')
    d=OUT/'figures'
    for ext in ['png','pdf']:fig.savefig(d/f'{stem}.{ext}',dpi=170)
    plt.close(fig)
    comparison=('Same topology/noise/Z/r0=50/tauG2; gain10/50 by weak/moderate M.' if joint else 'Same topology/noise/M/Z; r0 and tauG controls, fixed gain10.')
    write(OUT/f'{stem}.json',dict(conditions=metadata,comparison=comparison+' Curves are matched in observed duration; last50ms native fields share0–500Hz scale.',human_review='PENDING'))
    comparison_zh='全局增益与局部M的联合2×2' if joint else '全局反馈招募阈值与时间常数'
    with (d/'README.md').open('a') as f:f.write(f'\n### {stem}.png / .pdf\n同一噪声下比较{comparison_zh}，依次显示全局/双核放电、Z、全局输入乘Z前后以及实际总兴奋/抑制电流，末行是同一末50ms原生空间活动。各列使用共同的实际观测长度，未用尚未保存的进度值补画轨迹。**关注点**：反馈是阻止进入、暂时压低平台，还是在高活动后真正使双核和外围恢复；原始反馈增大但有效反馈被Z削弱须单独辨认。\n')

def figure(name):
    folder=OUT/'runs'/name;job=read(OUT/'jobs'/(name+'.json'));s=classify(folder)
    a=load(folder,['time_ms','spikes_1ms','regions_1ms','field_time_ms','field_5ms','raster','slow_time_ms','Z','M','currents'])
    if (folder/'result.json').exists():
        stop=read(folder/'result.json')['display_stop_s']*1000
        groups=[('time_ms',['time_ms','spikes_1ms','regions_1ms']),('field_time_ms',['field_time_ms','field_5ms']),('slow_time_ms',['slow_time_ms','Z','M','currents'])]
        for clock,keys in groups:
            selected=a[clock]<=stop
            for key in keys:a[key]=a[key][selected]
        a['raster']=a['raster'][:round(stop*10)]
    geo=np.load(OUT/'geometry.npz');nr=geo['region_counts'];nc=geo['cell_e_counts'];centers=geo['centers_mm']
    t=a['time_ms']/1000;ts=a['slow_time_ms']/1000;tf=a['field_time_ms']/1000
    rate=gaussian_filter1d(a['spikes_1ms'][:,0]*1000/32000,3)
    fig=plt.figure(figsize=(22,16));gs=fig.add_gridspec(3,2,width_ratios=[1.6,1],height_ratios=[1.45,1,1],hspace=.53,wspace=.30)
    raster_slot=gs[0,0].subgridspec(2,1,height_ratios=[2.3,1],hspace=.32)
    ax=fig.add_subplot(raster_slot[0]);xy=np.nonzero(a['raster']);it,ix=xy;spike_t=it*.0001
    mapping=np.r_[np.linspace(0,33,20),np.linspace(36,69,20),np.linspace(72,84,20),np.linspace(87,99,20)]
    for lo,hi,col in [(0,20,'#a46d99'),(20,40,'#357fa4'),(40,60,'#225b7f'),(60,80,'#c17730')]:
        selected=(ix>=lo)&(ix<hi)
        ax.scatter(spike_t[selected],mapping[ix[selected]],s=7,c=col,edgecolors='none',rasterized=True)
    for v in [34.5,70.5,85.5]:ax.axhline(v,color='#aaaaaa',lw=.7)
    ax.set_yticks([16.5,52.5,78,93],['Core A E','Core B E','Other E','I']);ax.set_xlim(0,t[-1]);ax.set_ylim(-2,101)
    ax.tick_params(labelbottom=False);ax.text(-.12,1.04,'A',transform=ax.transAxes,fontweight='bold',fontsize=24)
    bx=fig.add_subplot(gs[1,0],sharex=ax);bx.plot(ts,a['Z'][:,0],color='#7e399d',lw=1.7,label='E mean Z')
    bx.fill_between(ts,a['Z'][:,2],a['Z'][:,4],color='#7e399d',alpha=.12)
    bx.plot(ts,a['Z'][:,5],color='#de72b0',lw=1,label='Core A Z');bx.plot(ts,a['Z'][:,6],color='#4b9ed1',lw=1,label='Core B Z')
    br=bx.twinx();br.plot(ts,job['eta_m']*a['M'][:,0],color='#b86820',lw=1.2,label='M current')
    bx.set_ylabel('Resource Z');br.set_ylabel('M current (mV equiv.)',color='#b86820');bx.set_xlabel('Time (s)');bx.set_ylim(0,1.04)
    bx.legend(loc='upper right',frameon=True,facecolor='white',edgecolor='none',framealpha=.92,fontsize=12,ncol=1);bx.text(-.12,1.04,'B',transform=bx.transAxes,fontweight='bold',fontsize=24)
    bx.set_zorder(br.get_zorder()+1);bx.patch.set_visible(False)
    dx=fig.add_subplot(gs[0,1],projection='3d');rr=np.interp(ts,t,rate);pp=np.c_[a['Z'][:,0],a['currents'][:,2],rr]
    seg=np.stack([pp[:-1],pp[1:]],axis=1);norm=Normalize(ts[0],ts[-1]);lc=Line3DCollection(seg,cmap='viridis',norm=norm,linewidth=1.1);lc.set_array(ts[:-1]);dx.add_collection3d(lc)
    dx.set_xlim(a['Z'][:,0].min()-.02,1.02);dx.set_ylim(0,max(1,a['currents'][:,2].max()*1.05));dx.set_zlim(0,max(50,rr.max()*1.05));dx.view_init(24,135)
    dx.set_xlabel('Mean Z',labelpad=10);dx.set_ylabel('Applied inhibition',labelpad=12);dx.set_zlabel('')
    dx.text2D(-.13,.52,'E rate (Hz)',rotation=90,transform=dx.transAxes,fontsize=17)
    fig.colorbar(lc,ax=dx,shrink=.6,pad=.20,label='Time (s)');dx.text2D(-.05,1.04,'D',transform=dx.transAxes,fontweight='bold',fontsize=24)
    ex=fig.add_subplot(gs[1,1]);ex.plot(t,rate,color='#333333',lw=.8,label='All E')
    for i,col in enumerate(['#ae5d9c','#347ea0']):ex.plot(t,gaussian_filter1d(a['regions_1ms'][:,i]*1000/nr[i],3),color=col,lw=.6,alpha=.7,label=f'Core {"AB"[i]} E')
    ex.set_xlabel('Time (s)');ex.set_ylabel('Rate (Hz)');ex.legend(frameon=False,fontsize=11);ex.text(-.1,1.04,'E',transform=ex.transAxes,fontweight='bold',fontsize=24)
    # Select actual native windows by activity and detector timing; no field from a frozen-Z continuation is substituted.
    entries=s['entries'];rec=s['recoveries'];times=[]
    before=entries[0]['onset_s'] if entries else min(t[-1],10)
    rest_candidates=np.flatnonzero((t>.035)&(t<min(1.,before/3))&(rate<1))
    times.append(float(t[rest_candidates[len(rest_candidates)//2]]) if len(rest_candidates) else .035)
    # Use the same complete finite-event definition as the independent audit.
    from analyze_topic4_autonomous_events import finite_events
    nb=len(t)//10
    r10=a['spikes_1ms'][:nb*10,0].reshape(nb,10).sum(1)/320
    g10=a['regions_1ms'][:nb*10].reshape(nb,10,6).sum(1)
    actual_events=finite_events(r10,g10,nr)
    peaks=np.asarray([int(np.argmin(abs(t-e['peak_s']))) for e in actual_events
        if e['start_s']>.2 and e['end_s']<max(.3,before-.3)],dtype=int)
    verified_finite_event=bool(len(peaks))
    if len(peaks):
        def locality(idx):
            lo=max(0,idx-25);hi=min(len(t),idx+25);rr=a['regions_1ms'][lo:hi,:3].sum(0)/nr[:3]/((hi-lo)*.001)
            return max(rr[:2])/(1+rr[2])
        event=int(max(peaks,key=locality));times.append(float(t[event]))
    else:times.append(min(t[-1]-.025,max(.05,before/2)))
    # The resting example must PRECEDE the selected finite event, not be
    # a later quiet gap chosen independently on the same time axis.
    quiet10=(r10<5)&(g10[:,:2]/nr[:2]/.01<5).all(1)
    edges=np.diff(np.r_[False,quiet10,False].astype(int))
    quiet_windows=[]
    for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if hi-lo<5:continue
        tm=(lo+hi)*.005
        take=(tf>=tm-.025)&(tf<tm+.025)
        if take.sum()!=10:continue
        whole_rate=a['field_5ms'][take].sum()/32000/.05
        if whole_rate<1:quiet_windows.append(float(tm))
    quiet_before=[tm for tm in quiet_windows if tm<times[1]-.05]
    verified_rest=bool(quiet_before)
    times[0]=quiet_before[-1] if verified_rest else .025
    early=max(.025,before-.1)
    if entries:
        nf=len(a['field_5ms'])//4
        f20=a['field_5ms'][:nf*4].reshape(nf,4,400).sum(1)/np.maximum(nc,1)/.02
        hit=f20>=100;cs=np.cumsum(np.pad(hit.astype(int),((1,0),(0,0))),axis=0)
        sustained=(cs[10:]-cs[:-10])>=10
        first_latch=float(np.flatnonzero(sustained.any(1))[0]*.02) if sustained.any() else before-.1
        candidates=np.flatnonzero((tf>max(.025,first_latch-.1))&(tf<min(before,first_latch+.2)))
        early=max(.025,first_latch)
        for idx in candidates:
            q=np.abs(tf-tf[idx])<=.025;fm=a['field_5ms'][q].sum(0)/np.maximum(nc,1)/(q.sum()*.005)
            fraction=float((fm>=20).mean())
            if .1<=fraction<=.35:early=float(tf[idx]);break
    times.append(early);times.append(min(t[-1]-.025,before+.1))
    recovered_time=t[-1]-.025
    if rec:
        recovered_time=(rec[0]['start_s']+rec[0]['confirmation_s'])/2
        q=[tm for tm in quiet_windows if rec[0]['start_s']+.025<tm<rec[0]['confirmation_s']-.025]
        if q:recovered_time=min(q,key=lambda tm:abs(tm-recovered_time))
    times.append(recovered_time)
    labels=['Rest' if verified_rest else 'Initial','Interictal' if verified_finite_event else 'Early activity','Entry' if entries else 'No entry','High' if entries else 'Later','Recovered' if rec else 'Tail']
    if not entries:
        # No invented entry/high placeholders for a condition that never entered.
        late=[e for e in actual_events if e['start_s']>max(t[-1]/2,times[1]+.075)]
        last_event=float(max(late,key=lambda e:e['peak_all_E_Hz'])['peak_s']) if late else t[-1]-.025
        times=[times[0],times[1],last_event]
        labels=['Rest' if verified_rest else 'Initial','Finite event' if verified_finite_event else 'Early activity','Late finite event' if late else 'Tail']
        # A subthreshold persistent core state must remain visible even when
        # the whole-network200Hz gate was never met.
        if late and t[-1]-.025>last_event+.1:
            times.append(float(t[-1]-.025));labels.append('Tail activity')
    elif not rec:
        times=times[:4];labels=labels[:4]
        post_high=[e for e in actual_events if e['start_s']>before+.5]
        if post_high:
            times.append(float(post_high[-1]['peak_s']));labels.append('Post-high finite event')
    if len(entries)>=2:
        # A fixed delay after confirmation may already be quiet for a finite
        # second high episode. Select an actual50ms peak inside that episode.
        second=entries[1]
        candidates=np.flatnonzero((tf>=second['onset_s']+.025)&
                                  (tf<=min(second['confirmation_s']-.025,t[-1]-.025)))
        assert len(candidates), 'Second entry must contain a complete50ms snapshot'
        peak=max(candidates,key=lambda k:a['field_5ms'][(tf>=tf[k]-.025)&(tf<tf[k]+.025)].sum())
        times.append(float(tf[peak]));labels.append('Re-entry')
    label_colors={'Rest':'#397db0','Initial':'#727981','Interictal':'#bf842b',
        'Finite event':'#bf842b','Late finite event':'#bf842b','Early activity':'#bf842b',
        'Entry':'#d45a4f','High':'#bb3544','Re-entry':'#bb3544','Recovered':'#248671',
        'Tail':'#727981','Tail activity':'#727981','Post-high finite event':'#bf842b'}
    state_colors=[label_colors.get(label,'#727981') for label in labels]
    zoomgrid=raster_slot[1].subgridspec(1,2,wspace=.36);zoom_records=[]
    for i,tm in enumerate(times[1:3]):
        zx=fig.add_subplot(zoomgrid[i]);lo=float(np.clip(tm-.05,0,max(0,t[-1]-.30)));hi=min(t[-1],lo+.30)
        for lower,upper,col in [(0,20,'#a46d99'),(20,40,'#357fa4')]:
            selected=(ix>=lower)&(ix<upper)&(spike_t>=lo)&(spike_t<hi)
            zx.scatter(spike_t[selected],ix[selected],s=15,marker='|',c=col,linewidths=.9,rasterized=True)
        color=state_colors[i+1]
        zx.set(xlim=(lo,hi),ylim=(-1,40),yticks=[9.5,29.5],yticklabels=['Core A E','Core B E'],xlabel='Time (s)')
        zx.axhline(19.5,color='.7',lw=.6);zx.axvline(tm,color=color,ls=':',lw=1)
        zx.axvspan(tm-.025,tm+.025,color=color,alpha=.12)
        zx.set_xticks(np.linspace(lo+.05,hi-.05,3));zx.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.2f}'))
        zx.text(.01,.95,str(i+2),transform=zx.transAxes,color=color,va='top',weight='bold',fontsize=17)
        ax.add_patch(Rectangle((lo,-1),hi-lo,72,fill=False,ec=color,lw=1.6))
        zoom_records.append(dict(state=i+2,window_s=[lo,hi],same_fixed_spikes_and_row_order=True))
    sub=gs[2,:].subgridspec(1,len(times)+1,width_ratios=[1]*len(times)+[.06],wspace=.25)
    for i,(tm,label) in enumerate(zip(times,labels)):
        cx=fig.add_subplot(sub[0,i]);sel=(tf>=tm-.025)&(tf<tm+.025);duration=sel.sum()*.005
        field=a['field_5ms'][sel].sum(0)/np.maximum(nc,1)/duration
        im=cx.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',norm=PowerNorm(.6,0,500),interpolation='nearest')
        for core,xyc in zip('AB',centers):cx.add_patch(plt.Circle(xyc,1.5,fill=False,ec='#5fd8dc',lw=1.4))
        cx.set_xlabel('x (mm)');cx.set_title(f'{i+1} {label}\n{tm:.2f} s',fontsize=16)
        if i==0:cx.set_ylabel('y (mm)');cx.text(-.18,1.22,'C',transform=cx.transAxes,fontweight='bold',fontsize=24)
        else:cx.set_yticklabels([])
        color=state_colors[i]
        for aa in [ax,bx,ex]:aa.axvline(tm,color=color,ls=':',lw=1)
        ax.text(tm,.99,str(i+1),transform=ax.get_xaxis_transform(),va='top',ha='center',color=color,weight='bold',fontsize=17,bbox=dict(fc='white',ec='none',alpha=.9,pad=.2))
    fig.colorbar(im,cax=fig.add_subplot(sub[0,len(times)]),label='E rate (Hz)')
    for aa in fig.axes:
        aa.tick_params(labelsize=15)
        for label in [aa.xaxis.label,aa.yaxis.label]:label.set_fontsize(19)
    d=folder/'figures';d.mkdir(exist_ok=True)
    fig.savefig(d/'fig5_dynamics.png',dpi=180,bbox_inches='tight');fig.savefig(d/'fig5_dynamics.pdf',bbox_inches='tight');plt.close(fig)
    write(folder/'figure_metadata.json',dict(job=job,summary=s,state_times_s=times,state_labels=labels,state_colors=state_colors,verified_rest_before_event=verified_rest,state_times_chronological=bool(np.all(np.diff(times)>0)),verified_finite_preentry_event=verified_finite_event,finite_event_definition='Same<=300ms episodes bounded by>=30ms all-E<5Hz as the independent native-event audit.',rest_window_definition='Full50ms lies inside a>=50ms all-E and both-core<5Hz interval; actual50ms native-field whole-E mean<1Hz. Pointwise quiet alone is insufficient.',source='Same native continuous trajectory',raster_zoom_windows=zoom_records,core_rows_are_E_only=True,core_vertical_fraction=.70,spatial_display_normalization='PowerNorm gamma.6, actual0–500Hz; no rate data transformed',human_review='PENDING'))
    (d/'README.md').write_text('### fig5_dynamics.png\n连续 SNN raster、原生 Z/M、实测空间招募与 Z—抑制—放电率轨迹；双核E行占70%高度，编号2/3有同一脉冲的300ms局部放大。空间图取自同一轨迹的50ms窗口，未进入全局高态的条件仍显示真实有限事件及末段活动，以免漏掉局部持续放电；0–500Hz颜色用gamma=.6凸显低活动，未改变放电率数据。**关注点**：是否先进入高态，随后全局和双核均回到有间隔的事件，并在没有 reset 的情况下再次进入。\n\n### fig5_dynamics.pdf\n对应的矢量图，raster作为栅格嵌入以控制体积；核内raster仅E细胞，I另列。**关注点**：高态为操作性放电率定义，20ms慢量采样的3D图只是实测轨迹投影，不能称为已证实的持续振荡、nullcline或分岔。\n')
    feedback_figure(folder,job,a)

def feedback_figure(folder,job,a):
    kind='phi' if (folder/'phi_chunks').exists() else 'pool' if (folder/'pool_chunks').exists() else None
    if kind is None:return
    keys=['time_ms','phi_mean','phi_max'] if kind=='phi' else ['time_ms','rate_Hz','raw_global_current','effective_global_current']
    records={k:[] for k in keys}
    for path in sorted((folder/(kind+'_chunks')).glob('*.npz')):
        if '.tmp.' in path.name:continue
        with np.load(path) as q:
            for k in keys:records[k].append(q[k])
    if not records['time_ms']:return
    q={k:np.concatenate(v) for k,v in records.items()};ts=a['slow_time_ms']/1000;tq=q['time_ms']/1000
    fig,axs=plt.subplots(3,1,figsize=(13,10),sharex=True,gridspec_kw={'hspace':.25})
    axs[0].plot(ts,a['Z'][:,0],color='#7e399d',label='E mean Z');axs[0].set_ylabel('Resource Z');axs[0].set_ylim(0,1.04)
    rr=axs[0].twinx();rr.plot(ts,job['eta_m']*a['M'][:,0],color='#b86820');rr.set_ylabel('Local M current\n(mV equiv.)',color='#b86820')
    axs[1].plot(ts,a['currents'][:,0],color='#cf6454',label='Excitatory input')
    axs[1].plot(ts,a['currents'][:,2],color='#3575a1',label='Total applied inhibition')
    axs[1].set_ylabel('Current (mV equiv.)');axs[1].legend(frameon=False,loc='upper right')
    if kind=='phi':
        axs[2].plot(tq,q['phi_mean'],color='#39865b',label='E mean');axs[2].plot(tq,q['phi_max'],color='#91b79f',label='E maximum')
        axs[2].set_ylabel('Threshold shift, phi (mV)');axs[2].legend(frameon=False,loc='upper right')
    else:
        axs[2].plot(tq,q['rate_Hz'],color='#756196',label='Global rate filter')
        axs[2].axhline(job['pool_threshold_Hz'],color='#756196',ls=':',label='Recruitment threshold');axs[2].set_ylabel('Global filtered rate (Hz)')
        gr=axs[2].twinx();gr.plot(tq,q['raw_global_current'],color='#bc9870',lw=.8,label='Before Z')
        gr.plot(tq,q['effective_global_current'],color='#417f70',label='After Z');gr.set_ylabel('Added inhibition\n(mV equiv.)')
        axs[2].legend(frameon=False,loc='upper left');gr.legend(frameon=False,loc='upper right')
    axs[2].set_xlabel('Time (s)');axs[2].set_xlim(0,a['time_ms'][-1]/1000)
    for i,ax in enumerate(axs):ax.text(-.1,1.02,'ABC'[i],transform=ax.transAxes,weight='bold',fontsize=22)
    for ax in fig.axes:ax.tick_params(labelsize=13);ax.xaxis.label.set_fontsize(16);ax.yaxis.label.set_fontsize(16)
    fig.subplots_adjust(left=.12,right=.87,bottom=.08,top=.97)
    for ext in ['png','pdf']:fig.savefig(folder/'figures'/f'added_feedback.{ext}',dpi=170)
    plt.close(fig)
    with (folder/'figures/README.md').open('a') as fh:fh.write('\n### added_feedback.png / .pdf\n列出本分支实际新增的反馈状态，并与连续Z/M及总兴奋、实际有效抑制电流对齐。快速phi是阈值偏移；全局池图另区分未乘Z与实际乘Z的额外抑制。**关注点**：不得把新增机制的作用误归因于原生M，也不得把原始GABA输入当成已经应用到膜的电流。\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name');ap.add_argument('--root',type=Path);args=ap.parse_args()
    if args.root:OUT=args.root
    summary()
    if args.name:figure(args.name)
