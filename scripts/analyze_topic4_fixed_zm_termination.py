#!/usr/bin/env python3
"""Read actual current-model trajectories; never substitute LAS termination."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse,json
from pathlib import Path
import numpy as np
import run_topic4_fixed_zm_termination as run
import analyze_topic4_fig5_preentry_events as event_audit

OUT=run.OUT

def safe(x):
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.generic):return safe(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,Path):return str(x)
    return x

def write(path,x):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(safe(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

def figure_readme(dest):
    blocks=[]
    for p in sorted(dest.glob('fig5_*.png')):
        blocks.append(f'### {p.name}\n\n固定Fig5的Z/M双核模型，连续raster、原Z/M和原生空间快照使用同一条仿真；新增慢钾机制另画真实电导。D是实测状态投影，E是本轮离散条件，F只在基线足够时与固定Fig3C患者比较；不是分岔证明。\n\n**关注点**：区分抑制入态、降低平台与真正自主退出；尚待用户人工验收。\n\n')
    for p in sorted(dest.glob('native_storyboard_*.png')):
        blocks.append(f'### {p.name}\n\n来自原生20×20场的50ms计数快照，未经过电极投影或空间平滑。圆圈为物理core半径1.5mm，统一色标0–500Hz。\n\n**关注点**：看移动波前、局部高率终止与边界接触，不能把全E均值低于阈值自动解释成没有传播。\n\n')
    for p in sorted(dest.glob('native_*.gif')):
        blocks.append(f'### {p.name}\n\n原生空间率动画与全E/core率、Z及新增gK时间序列同步。每帧使用50ms实际计数，播放速度不同于仿真时间，画面显示真实时间。\n\n**关注点**：区分初始化暂态、后续自主事件以及波前到达边界后消失；该动画不替代患者传播模式验证。\n\n')
    for p in sorted(dest.glob('autonomous_cycle_*.png')):
        blocks.append(f'### {p.name}\n\n完整连续时间轴上的E/I率、原Z/M及新增gK，与第一次和第二次事件的绝对时间raster放大对应。两次进入之间没有任何参数或状态reset，第二次进入按全E计数核查。\n\n**关注点**：同一轨迹的自主退出及再进入，不等同跨噪声稳健性、患者模式恢复或临床复发率验证。\n\n')
    (dest/'README.md').write_text(''.join(blocks))

def load(folder,subdir='chunks',keys=None):
    parts={};end=0
    committed=max([int(p.stem.split('_')[-1]) for p in (folder/'chunks').glob('*.npz') if '.tmp.' not in p.name],default=0)
    for path in sorted((folder/subdir).glob('*.npz')):
        if '.tmp.' in path.name:continue
        # Extra observers finish before the native chunk is atomically committed.
        # Reading only this prefix avoids opening a still-compressing observer file.
        if subdir!='chunks' and int(path.stem.split('_')[-1])>committed:continue
        with np.load(path) as a:
            if subdir=='chunks':
                assert int(a['start_step'])==end;end=int(a['end_step'])
            for k in (keys or a.files):
                if k in ['start_step','end_step']:continue
                parts.setdefault(k,[]).append(a[k])
    return {k:np.concatenate(v) for k,v in parts.items()}

def spans(x):
    e=np.diff(np.r_[False,x,False].astype(int))
    return list(zip(np.flatnonzero(e==1),np.flatnonzero(e==-1)))

def analyze(name):
    folder=OUT/'runs'/name;job=run.carrier.base.read(OUT/'jobs'/f'{name}.json')
    a=load(folder,keys=['time_ms','spikes_1ms','regions_1ms','slow_time_ms','Z','M','field_time_ms','field_5ms'])
    if not a:return None
    with np.load(OUT/'geometry.npz') as g:geo={k:g[k] for k in g.files}
    n=len(a['spikes_1ms'])//10;end=n*.01
    counts=a['spikes_1ms'][:n*10].reshape(n,10,2).sum(1)
    local=a['regions_1ms'][:n*10].reshape(n,10,6).sum(1)
    rates=np.c_[counts[:,0]/320,local[:,:3]/geo['region_counts'][:3]/.01]
    tracker=run.carrier.fresh_tracker()
    for k,r in enumerate(rates):run.carrier.track(tracker,r,(k+1)*.01)
    ev=event_audit.events(rates[:,0],end=end)
    nf=len(a['field_5ms'])//10
    fields=a['field_5ms'][:nf*10].reshape(nf,10,400).sum(1)/geo['cell_e_counts']/.05
    area=(fields>=20).mean(1)
    mech=load(folder,'mechanism_chunks')
    late=min(5.,end);last=rates[-round(late/.01):]
    result_path=folder/'result.json'
    complete=result_path.exists() and run.carrier.base.read(result_path)['status']=='COMPLETE'
    category=3 if len(tracker['entries'])>1 and tracker['recoveries'] else 2 if tracker['recoveries'] else 1 if tracker['entries'] else 0
    broad_intervals=[dict(start_s=lo*.05,end_s=hi*.05,duration_s=(hi-lo)*.05) for lo,hi in spans(area>=.5)]
    summary=dict(name=name,job=job,complete=complete,observed_s=end,entries=tracker['entries'],recoveries=tracker['recoveries'],
        category=category,category_label=['No confirmed global high','High without confirmed return','High then return','High / return / high'][category],
        late_window_s=[end-late,end],late_mean_Hz=last.mean(0),late_quiet_fraction=(last<5).mean(0),
        population_order=['All E','Core A E','Core B E','Other E'],finite_events=ev,
        broad_intervals=broad_intervals,broad_definition='Descriptive only: at least50% of20x20native cells have50ms rate>=20Hz; not the high-state or recovery acceptance criterion.',
        max_E_10ms_Hz=rates[:,0].max(),mean_Z_end=a['Z'][-1,0],mean_Z_min=a['Z'][:,0].min(),
        M_current_end=job['eta_m']*a['M'][-1,0],Z_recovery_target_fraction_late=1-a['Z'][-round(late/.02):,8].mean(),
        no_intervention=True,human_review='PENDING')
    first=tracker['entries'][0]['onset_s'] if tracker['entries'] else end
    summary['finite_events_before_first_entry']=sum(e['end_s']<first for e in ev)
    summary['core_high_intervals']={label:[dict(start_s=lo*.01,end_s=hi*.01,duration_s=(hi-lo)*.01)
        for lo,hi in spans(rates[:,i]>=200) if hi-lo>=20] for i,label in [(1,'A'),(2,'B')]}
    if mech:
        summary['mechanism_tail']={k:mech[k][-min(len(mech[k]),round(late*1000)):].mean() for k in mech if k!='time_ms'}
    intrinsic=load(folder,'intrinsic_adaptation_chunks')
    if intrinsic:
        summary['sahp_tail']={k:intrinsic[k][-min(len(intrinsic[k]),round(late*1000)):].mean() for k in intrinsic if k!='time_ms'}
    write(OUT/'analysis'/f'{name}.json',summary)
    return summary

def collect():
    p=run.prepare();rows=[]
    for j in p['initial_jobs']:
        r=analyze(j['name'])
        if r:rows.append(r)
    result=dict(rows=rows,expected=len(p['initial_jobs']),complete=len(rows)==len(p['initial_jobs']) and all(r['complete'] for r in rows),
        scope=f"{len(p['initial_jobs'])} paired conditions on one native40k network and one noise; exploratory mechanism contrast, not a fitted phase diagram.")
    write(OUT/'measured_comparison.json',result)
    return result

def choose_states(a,geo,r):
    end=r['observed_s'];rate=a['spikes_1ms'][:,0].reshape(-1,10).sum(1)/320
    on=r['entries'][0]['onset_s'] if r['entries'] else end
    ev=[e for e in r['finite_events'] if .5<e['start_s']<min(5.,on-.5) and (not r['entries'] or e['end_s']<on)]
    ev=sorted(ev,key=lambda e:abs(e['start_s']-4.))
    initial_fallback=False
    if not ev and r['finite_events']:
        first_event=r['finite_events'][0]
        if first_event['start_s']<.2 and first_event['end_s']<on:
            ev=[first_event];initial_fallback=True
    quiet=[(lo,hi) for lo,hi in spans(rate<1) if lo*.01>.25 and hi*.01<min(3.,on) and hi-lo>=5]
    times=[]
    initial_long_wave=bool(r['finite_events'] and r['finite_events'][0]['start_s']<.2 and r['finite_events'][0]['duration_s']>.2)
    if initial_fallback:
        times.append((.025,'Initial','#657181'))
    elif quiet:
        lo,hi=quiet[0];times.append(((lo+hi)*.005,'Rest','#657181'))
    elif initial_long_wave:
        times.append((.025,'Initial','#657181'))
    if ev:
        local=a['regions_1ms'][:len(rate)*10,:2].reshape(-1,10,2).sum(1)/geo['region_counts'][:2]/.01
        begin=max(0,round((ev[0]['start_s']-.05)/.01));stop=min(len(rate),round((ev[0]['start_s']+.10)/.01))
        idx=np.flatnonzero(local[begin:stop].max(1)>=50)
        t=(begin+idx[0]+.5)*.01+.015 if len(idx) else ev[0]['start_s']+.005
        # Long self-terminated waves are not automatically interictal. The
        # frozen baseline's observed pre-entry events span0.04-0.20s; use this
        # only as a display distinction, never to change the high/return gate.
        event_label='Initial wave' if initial_fallback else 'Interictal' if ev[0]['duration_s']<=.20+1e-9 else 'Long event'
        times.append((round(t/.005)*.005,event_label,'#267ba8'))
    if r['entries']:
        # Show a short lead-in before the measured gate, then the first local
        # population maximum; do not move the numerical onset criterion.
        ent=max(0,on-.15)
        lo=round(on/.01);hi=min(len(rate),round((on+1)/.01))
        peak=(lo+np.argmax(rate[lo:hi])+.5)*.01
        times.extend([(ent,'Entry','#dd871c'),(min(peak,end-.05),'High','#ba263c')])
    else:
        if ev and ev[0]['duration_s']>.20+1e-9:
            times.append(((ev[0]['start_s']+ev[0]['end_s'])/2,'Propagation','#dd871c'))
        else:
            k=round(.25/.01)+np.argmax(rate[round(.25/.01):]);times.append(((k+.5)*.01,'Largest event','#dd871c'))
    if r['recoveries']:
        rr=r['recoveries'][0];t=next(float(rr[k]) for k in ['confirmation_s','time_s','recovery_s'] if k in rr)
        times.append((min(t+.1,end-.05),'Return','#268771'))
    elif end>on+2 and r['entries']:
        label='Late high' if rate[-min(len(rate),100):].mean()>=200 else 'Persistent'
        times.append((end-.1,label,'#ba263c'))
    elif not r['entries'] and r['finite_events']:
        last_event=r['finite_events'][-1]['end_s']
        later_quiet=[(lo,hi) for lo,hi in spans(rate<1) if lo*.01>=last_event-.01 and hi-lo>=200]
        if later_quiet:times.append((later_quiet[0][0]*.01+1.,'Quiescent','#268771'))
    times=sorted(times)
    out=[]
    for number,(t,label,col) in enumerate(times,1):
        ix=round((t-.025)/.005);ix=max(0,min(ix,len(a['field_5ms'])-10));t=ix*.005+.025
        field=a['field_5ms'][ix:ix+10].sum(0)/geo['cell_e_counts']/.05
        out.append(dict(number=number,time_s=t,label=label,color=col,field_Hz=field))
    return out

def early_energy(name,geo,r,anchor='high_gate'):
    import analyze_topic4_fig5_onset_z_field as energy
    from scipy.stats import spearmanr
    if not r['entries']:return None
    onset=r['entries'][0]['onset_s']
    if anchor=='episode_start':
        episode=next((e for e in r['finite_events'] if e['start_s']<=onset<e['end_s']),None)
        if episode is None:return None
        onset=episode['start_s']
        if onset<3.5:return None
    else:
        if onset<4:return None
    if r['observed_s']<onset+1:return None
    folder=OUT/'runs'/name
    manifest=folder/'dense_contact_source.json'
    sampling_status='PROVISIONAL_COARSE_SAMPLING_WITHOUT_EXPLICIT_ANTIALIAS_FILTER'
    sampling_source=str(folder/'actual_current_chunks')
    if manifest.exists():
        dense=run.carrier.base.read(manifest)
        qa=run.carrier.base.read(Path(dense['qa']))
        assert qa['status']=='PASS' and all(qa['fields'].values())
        a=load(Path(dense['folder']),'dense_contact_chunks');fs=float(dense['sample_rate_Hz'])
        sampling_status='NATIVE_TIME_RESOLUTION_PSD_WITH_BITWISE_DYNAMICS_PARITY'
        sampling_source=str(Path(dense['folder'])/'dense_contact_chunks')
    else:
        a=load(folder,'actual_current_chunks');fs=500.
    times=a['time_ms']/1000;raw=a['contact_current']
    assert np.allclose(np.diff(times),1/fs,atol=1e-10,rtol=0)
    record,fz,cases=energy.patient_cases()
    names=[str(v) for v in geo['contact_names']];target=list(fz['names']);index=[names.index(v) for v in target]
    signal=(raw-raw.mean(1,keepdims=True))[:,index]
    early=[round(onset*fs)/fs,round(onset*fs)/fs+1.]
    bp,bt=energy.segment_power(signal,times,fs,[.5,3.5]);ep,et=energy.segment_power(signal,times,fs,early)
    assert bp.shape==(15,5) and ep.shape==(15,1)
    med=np.median(bp,axis=1);mad=np.median(abs(bp-med[:,None]),axis=1)
    if np.any(mad<=0):return None
    model=(ep[:,0]-med)/(1.4826*mad)
    raw_bp,_=energy.segment_power(raw[:,index],times,fs,[.5,3.5]);raw_ep,_=energy.segment_power(raw[:,index],times,fs,early)
    raw_med=np.median(raw_bp,axis=1);raw_mad=np.median(abs(raw_bp-raw_med[:,None]),axis=1)
    raw_model=(raw_ep[:,0]-raw_med)/(1.4826*raw_mad)
    canonical=run.carrier.base.read(run.ROOT/'results/paper-ready-figure/fig3/fig3_panelc_metadata.json')
    patient=next(c for c in cases if c['seizure_idx']==canonical['seizure_idx'])
    assert np.allclose(patient['values'],canonical['raw_ictal_robust_z_mean'],rtol=0,atol=1e-12)
    out=dict(name=name,energy_anchor=anchor,contact_names=target,model_robust_z=model,patient_robust_z=patient['values'],
        patient_public_seizure=f'SZ{patient["seizure_idx"]+1}',patient_source=patient['source'],
        model_window_s=early,model_baseline_s=[.5,3.5],model_baseline_frames=5,model_sampling_Hz=fs,
        sampling_status=sampling_status,sampling_source=sampling_source,
        model_positive_contacts=int(np.sum(model>0)),no_CAR_diagnostic_robust_z=raw_model,
        no_CAR_positive_contacts=int(np.sum(raw_model>0)),no_CAR_scope='Diagnostic of reference cancellation only; does not replace the predefinedCAR patient comparison.',
        signal='Original contact weights applied to absolute actual AMPA, native Z-scaled GABA and new global shunt currents; CAR before spectrum.',
        estimator='Exact Fig3C spectrogram estimator:1s Tukey0.25,0.5s hop,constant detrend,summed1-150Hz PSD then natural log. Model uses5 baseline frames; patient retains>=50.',
        patient_selection='Locked existing Fig3C seizure, never reselected by current-model correlation.',
        contact_rho=float(spearmanr(model,patient['values']).statistic),
        boundary='Descriptive contact-field comparison; different baseline lengths and observation operators. No fitted amplitude equivalence, rescaled frequency, or held-out clinical mechanism validation.')
    suffix='' if anchor=='high_gate' else '_episode_start'
    write(OUT/'analysis'/f'{name}_early_energy{suffix}.json',out)
    return out,fz,energy.clinical

def plot(name,energy_anchor='high_gate'):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize,PowerNorm,ListedColormap
    from matplotlib.patches import Circle,Rectangle
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from scipy.ndimage import gaussian_filter1d
    comp=collect();r=next(v for v in comp['rows'] if v['name']==name)
    folder=OUT/'runs'/name;a=load(folder)
    with np.load(OUT/'geometry.npz') as g:geo={k:g[k] for k in g.files}
    end=r['observed_s'];intrinsic=load(folder,'intrinsic_adaptation_chunks')
    if r['recoveries']:
        end=min(end,r['recoveries'][0]['confirmation_s']+3.)
    elif not r['entries'] and r['finite_events'] and r['finite_events'][0]['duration_s']>.2:
        cluster_end=r['finite_events'][0]['end_s']
        for event in r['finite_events'][1:]:
            if event['start_s']-cluster_end>.5:break
            cluster_end=event['end_s']
        end=min(end,cluster_end+5.)
    end=min(r['observed_s'],np.ceil(end/.05)*.05)
    for key in ['spikes_1ms','regions_1ms','time_ms']:a[key]=a[key][:round(end*1000)]
    a['raster']=a['raster'][:round(end*10000)]
    for key in ['field_5ms','field_time_ms']:a[key]=a[key][:round(end*200)]
    slow_keep=a['slow_time_ms']<end*1000
    for key in ['slow_time_ms','Z','M','currents','regional_currents']:a[key]=a[key][slow_keep]
    if intrinsic:
        keep=intrinsic['time_ms']<end*1000;intrinsic={k:v[keep] for k,v in intrinsic.items()}
    shown={**r,'observed_s':end,'finite_events':[e for e in r['finite_events'] if e['end_s']<=end]}
    snaps=choose_states(a,geo,shown)
    if energy_anchor=='episode_start' and r['entries']:
        on=r['entries'][0]['onset_s'];episode=next(e for e in r['finite_events'] if e['start_s']<=on<e['end_s'])
        entry=next(s for s in snaps if s['label']=='Entry')
        ix=round((episode['start_s']+.02)/.005)
        entry.update(time_s=ix*.005+.025,label='Recruitment',field_Hz=a['field_5ms'][ix:ix+10].sum(0)/geo['cell_e_counts']/.05)
    plt.rcParams.update({'font.size':17,'axes.labelsize':20,'xtick.labelsize':17,'ytick.labelsize':17,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig=plt.figure(figsize=(23,17));gs=fig.add_gridspec(5,2,width_ratios=[1.36,1],height_ratios=[2,.9,1.25,.65 if intrinsic else .12,1.55],hspace=.38,wspace=.27,left=.07,right=.98,bottom=.07,top=.97)
    ax=fig.add_subplot(gs[0,0]);it,ix=np.where(a['raster']);tt=it*.0001
    mapping=np.r_[np.linspace(0,33,20),np.linspace(36,69,20),np.linspace(72,84,20),np.linspace(87,99,20)]
    for lo,hi,col in [(0,20,'#176ba1'),(20,40,'#168aa2'),(40,60,'#466675'),(60,80,'#c17730')]:
        take=(ix>=lo)&(ix<hi);ax.scatter(tt[take],mapping[ix[take]],s=7,c=col,lw=0,rasterized=True)
    for y in [34.5,70.5,85.5]:ax.axhline(y,c='#bbb',lw=.7)
    ax.set(ylim=(-2,112),yticks=[16.5,52.5,78,93],yticklabels=['Core A E','Core B E','Other E','I'],xlim=(0,end));ax.set_title('A',loc='left',weight='bold',pad=15)
    for i,s in enumerate(snaps):
        ax.axvline(s['time_s'],c=s['color'],ls=':',lw=1)
        close_next=i+1<len(snaps) and snaps[i+1]['time_s']-s['time_s']<.3
        ax.text(s['time_s'],108 if close_next else 102,str(s['number']),ha='center',c=s['color'],weight='bold')
    ax.tick_params(labelbottom=False)
    zs=gs[1,0].subgridspec(1,2,wspace=.30)
    zo=[next((s for s in snaps if s['label'] in labels),snaps[0]) for labels in [('Interictal','Long event','Initial wave'),('Entry','Recruitment','Propagation','Largest event')]]
    for i,s in enumerate(zo):
        zz=fig.add_subplot(zs[i]);lo=max(0,s['time_s']-.05);hi=min(end,lo+.3)
        for low,high,col in [(0,20,'#176ba1'),(20,40,'#168aa2')]:
            take=(ix>=low)&(ix<high)&(tt>=lo)&(tt<=hi);zz.scatter(tt[take],ix[take],s=18,c=col,marker='|',lw=1,rasterized=True)
        zz.set(xlim=(lo,hi),ylim=(-1,40),yticks=[9.5,29.5],yticklabels=['Core A E','Core B E'],xlabel='Time (s)');zz.axvline(s['time_s'],c=s['color'],ls=':')
        ax.add_patch(Rectangle((lo,0),hi-lo,69,fc='none',ec=s['color'],lw=1.5));zz.set_title(str(s['number']),loc='left',c=s['color'],weight='bold')
    b=fig.add_subplot(gs[2,0]);st=a['slow_time_ms']/1000
    b.plot(st,a['Z'][:,0],c='#723490',lw=2,label='E mean Z');b.fill_between(st,a['Z'][:,2],a['Z'][:,4],color='#723490',alpha=.12)
    for i,(label,col) in enumerate([('Core A','#d24b99'),('Core B','#2195bc')]):b.plot(st,a['Z'][:,5+i],lw=.9,c=col,label=label)
    b.set(xlim=(0,end),ylim=(0,1.05),ylabel='Resource Z',xlabel='Time (s)');b.set_title('B',loc='left',weight='bold')
    twin=b.twinx();twin.plot(st,r['job']['eta_m']*a['M'][:,0],c='#9f631c',lw=1.4,label='Adaptation');twin.set_ylabel(r'$\eta_M M$ (mV equiv.)',c='#9f631c');twin.spines['right'].set_visible(True)
    lines=b.get_lines()+twin.get_lines();b.legend(lines,[v.get_label() for v in lines],loc='upper right',fontsize=11,ncol=1)
    for s in snaps:b.axvline(s['time_s'],c=s['color'],lw=1,ls=':')
    if intrinsic:
        b.set_xlabel('');b.tick_params(labelbottom=False)
        bk=fig.add_subplot(gs[3,0]);kt=intrinsic['time_ms']/1000
        bk.plot(kt,intrinsic['sahp_mean_conductance_ratio'],c='#a95b25',lw=1.2)
        bk.set(xlim=(0,end),ylabel='Added '+r'$g_K/g_L$',xlabel='Time (s)');bk.yaxis.label.set_fontsize(17)
        for s in snaps:bk.axvline(s['time_s'],c=s['color'],lw=1,ls=':')
    cgrid=gs[4,0].subgridspec(1,len(snaps)+1,width_ratios=[1]*len(snaps)+[.06],wspace=.32)
    for i,s in enumerate(snaps):
        cc=fig.add_subplot(cgrid[i]);im=cc.imshow(s['field_Hz'].reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='magma',norm=PowerNorm(.6,0,500),interpolation='nearest')
        for label,xy in zip('AB',geo['centers_mm']):cc.add_patch(Circle(xy,float(geo['core_radius_mm']),fc='none',ec='#56d5d3',lw=1.5));cc.text(xy[0],xy[1]+2,label,fontsize=13,c='#158b95',ha='center')
        cc.set(xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20]);cc.set_title(f'{s["number"]} {s["label"]}\n{s["time_s"]:.2f} s',color=s['color'],fontsize=14)
        if i==0:cc.set_ylabel('y (mm)');cc.text(-.15,1.36,'C',transform=cc.transAxes,weight='bold',fontsize=21)
        else:cc.tick_params(labelleft=False)
    fig.colorbar(im,cax=fig.add_subplot(cgrid[-1]),label='E rate (Hz)')
    de=gs[:2,1].subgridspec(1,2,width_ratios=[2,1],wspace=.25)
    d=fig.add_subplot(de[0],projection='3d');rate=gaussian_filter1d(a['spikes_1ms'][:,0]/32,2)
    mech=load(folder,'mechanism_chunks')
    hz=a['currents'][:,2]+np.interp(st,mech['time_ms']/1000,mech['global_outward_current_mV_equiv'])
    if 'feedback_form' in r['job']:
        hz-=r['job']['gamma']*r['job']['C_R']*a['Z'][:,0]*np.interp(st,mech['time_ms']/1000,mech['global_E_rate_Hz'])
    if intrinsic:
        state_y=np.interp(st,intrinsic['time_ms']/1000,intrinsic['sahp_mean_conductance_ratio'])
        state_ylabel=r'Mean $g_K/g_L$'
    else:
        state_y=hz;state_ylabel='Applied inhibition\n(mV equiv.)'
    xyz=np.c_[a['Z'][:,0],state_y,np.interp(st,(np.arange(len(rate))+.5)*.001,rate)]
    pts=np.stack([xyz[:-1],xyz[1:]],axis=1);lc=Line3DCollection(pts,cmap='viridis',norm=Normalize(0,end),linewidth=.9);lc.set_array(st[:-1]);d.add_collection(lc)
    d.set(xlim=(max(0,xyz[:,0].min()-.03),1.02),ylim=(min(0,state_y.min()),max(1,state_y.max()*1.05)),zlim=(0,max(30,xyz[:,2].max()*1.05)),xlabel='Mean Z',ylabel=state_ylabel,zlabel='E rate (Hz)')
    d.view_init(elev=23,azim=-54);d.set_box_aspect((1.35,1,1.2));d.set_title('D',loc='left',weight='bold');d.tick_params(labelsize=11,pad=1)
    for axis in [d.xaxis,d.yaxis,d.zaxis]:axis.label.set_fontsize(13)
    d.xaxis.labelpad=2;d.yaxis.labelpad=3;d.zaxis.labelpad=2
    for s in snaps:
        k=np.argmin(abs(st-s['time_s']));d.scatter(*xyz[k],c='white',edgecolors='#444',s=80,depthshade=False);d.text(*xyz[k],str(s['number']),fontsize=11)
    cb=fig.colorbar(lc,ax=d,shrink=.36,pad=.12,fraction=.025);cb.ax.tick_params(labelsize=10);cb.ax.set_title('Time\n(s)',fontsize=11,pad=9)
    matched='feedback_form' in r['job']
    panel_kind=run.prepare().get('parameter_panel_rows');phi_rows=panel_kind=='phi';sahp_rows=panel_kind=='sahp'
    e=fig.add_subplot(de[1]);mat=np.full((len(comp['rows']),1) if sahp_rows else (3 if matched else 2,2),np.nan);labels={}
    for row in comp['rows']:
        if sahp_rows:iy=list(v['name'] for v in comp['rows']).index(row['name']);jx=0
        elif matched:
            iy=[0.,1.25,2.5].index(row['job']['phi_jump']) if phi_rows else 0 if row['job']['feedback_form']=='current' else 2 if row['job']['phi_jump'] else 1
            jx=int(row['job']['gamma']>.3)
        else:iy=int(row['job']['phi_jump']>0);jx=int(row['job']['global_gain']>0)
        mat[iy,jx]=row['category'];labels[iy,jx]=row
    e.imshow(mat,origin='lower',cmap=ListedColormap(['#dbe5eb','#cf6977','#54a88f','#707bbb']),vmin=-.5,vmax=3.5,interpolation='nearest',aspect='auto' if sahp_rows else 'equal')
    for (iy,jx),row in labels.items():
        first='No global entry' if not row['entries'] else f'High gate {row["entries"][0]["onset_s"]:.2f}s'
        second=('Return' if row['recoveries'] else 'No return') if row['entries'] else None
        condition=None
        if sahp_rows:
            gamma='1/6' if abs(row['job']['gamma']-1/6)<1e-6 else f"{row['job']['gamma']:g}"
            condition=f"γ={gamma}, K×{row['job']['sahp_gain']:g}"
        e.text(jx,iy,'\n'.join([s for s in [condition,first,second,f'{row["observed_s"]:.0f}s observed'] if s]),ha='center',va='center',fontsize=10.5 if sahp_rows else 11)
    if sahp_rows:
        e.set(xticks=[],yticks=[])
    elif matched:
        e.set(xticks=[0,1],xticklabels=['1/6','1/2'],yticks=[0,1,2],yticklabels=['0','1.25','2.5'] if phi_rows else ['Current','Shunt','Shunt\n+ fast φ'],xlabel='Global fraction γ');e.tick_params(labelsize=12)
        if phi_rows:e.set_ylabel(r'$\Delta\phi$ (mV/spike)',fontsize=13)
    else:e.set(xticks=[0,1],xticklabels=['0','.0625'],yticks=[0,1],yticklabels=['0','2.5'],xlabel=r'$k_G$ (Hz$^{-1}$)',ylabel=r'$\Delta\phi$ (mV/spike)');e.yaxis.label.set_fontsize(13)
    e.set_title('E',loc='left',weight='bold');e.set_box_aspect(1.8 if sahp_rows else 1.)
    energy=early_energy(name,geo,r,energy_anchor)
    if energy:
        from scripts.plot_contact_plane_static import _smooth_rank_field_mm
        ev,fz,clinical=energy;fg=gs[2:,1].subgridspec(1,4,width_ratios=[1,.045,1,.045],wspace=.42)
        model_heading='Model' if ev['model_sampling_Hz']==10000 else 'Model\n500 Hz, provisional'
        if energy_anchor=='episode_start':model_heading+=f'\n{ev["model_window_s"][0]:.2f}–{ev["model_window_s"][1]:.2f} s'
        for i,(values,heading) in enumerate([(ev['model_robust_z'],model_heading),(ev['patient_robust_z'],f'E10 | {ev["patient_public_seizure"]}')]):
            ff=fig.add_subplot(fg[i*2]);values=np.asarray(values)
            points=np.asarray(fz['points_mm']);bound=float(np.max(abs(values)))
            xx,yy,field,_,_=_smooth_rank_field_mm(points[:,0],points[:,1],values,np.asarray(fz['support_a']),fz['display_xlim_mm'],fz['display_ylim_mm'],fz['display_sigma_mm'])
            norm=Normalize(-bound,bound)
            mapped=ff.imshow(field,origin='lower',extent=[xx.min(),xx.max(),yy.min(),yy.max()],cmap='RdBu',norm=norm,interpolation='bilinear')
            ff.scatter(points[:,0],points[:,1],c=values,cmap='RdBu',norm=norm,s=38,edgecolors='white',lw=.8)
            ff.set_xlabel('Shared axis (mm)',fontsize=16);ff.set_ylabel('y (mm)' if i==0 else '',fontsize=16);ff.tick_params(labelsize=14)
            ff.set_title(heading,fontsize=17)
            if i==0:ff.text(-.15,1.2,'F',transform=ff.transAxes,weight='bold',fontsize=21)
            pos=ff.get_position();cc=fig.add_axes([pos.x1+.007,pos.y0,.009,pos.height]);bar=fig.colorbar(mapped,cax=cc)
            bar.set_label('Power change (robust z)',fontsize=12);bar.ax.tick_params(labelsize=11)
        ev['display']='Exact Fig3C patient values, geometry, support and6mm smoothing; signed zero-centered color mapping makes positive changes blue and negative changes red. Separate symmetric ranges remain explicitly labeled. Negative model power is not painted as enhancement.'
    else:
        ff=fig.add_subplot(gs[2:,1]);ff.set_axis_off();ff.text(0,1,'F',transform=ff.transAxes,weight='bold',fontsize=21)
        ff.text(.05,.8,'No confirmed entry with adequate\nbaseline and onset recording.',transform=ff.transAxes,fontsize=17,va='top')
    suffix='' if energy_anchor=='high_gate' else '_early_recruitment'
    dest=OUT/'figures';dest.mkdir(exist_ok=True);fig.savefig(dest/f'fig5_{name}{suffix}.png',dpi=160);fig.savefig(dest/f'fig5_{name}{suffix}.pdf');plt.close(fig)
    write(dest/f'fig5_{name}{suffix}_metadata.json',dict(source=folder,producer_sha256=run.carrier.base.sha(__file__),summary=r,display_window_s=[0,end],full_observed_s=r['observed_s'],states=snaps,energy_anchor=energy_anchor,measured_comparison=comp,trajectory='Projection of measured native trajectory, not a deterministic vector field or nullcline.',human_review='PENDING',clinical_panel=energy[0] if energy else 'NOT_ESTIMABLE_YET'))
    figure_readme(dest)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['collect','plot']);ap.add_argument('--name');ap.add_argument('--root');ap.add_argument('--energy-anchor',choices=['high_gate','episode_start'],default='high_gate');args=ap.parse_args()
    if args.root:OUT=Path(args.root).resolve();run.OUT=OUT
    if args.mode=='plot':plot(args.name,args.energy_anchor)
    else:
        r=collect();print(json.dumps(safe([dict(name=v['name'],observed_s=v['observed_s'],category=v['category_label'],late_Hz=v['late_mean_Hz'],Z=v['mean_Z_end']) for v in r['rows']]),ensure_ascii=False))
