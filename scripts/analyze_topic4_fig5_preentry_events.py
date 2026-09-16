#!/usr/bin/env python3
"""Independent event-frequency/duration audit, before displaying any trend."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plot_topic4_m_parameter_modes as f
import analyze_topic4_weaker_M_onset_pilot as pilot
OUT=f.ROOT/'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914'

def sources():
    yield from pilot.sources()
    for seed in [9108401,9108402]:
        name=f'eta0.0005_s{seed}';p=OUT/'runs'/name
        if (p/'result.json').exists():yield dict(name=name,eta_m=.0005,seed=seed,source=p)

def load_small(folder,keys=('spikes_1ms','regions_1ms','slow_time_ms','Z','M')):
    result=f.read(folder/'result.json');end=result['tracker']['entries'][0]['confirmation_s']+2 if result['tracker']['entries'] else result['end_s']
    pieces={k:[] for k in keys};prev=0
    for path in sorted((folder/'chunks').glob('*.npz')):
        if '.tmp.' in path.name:continue
        with np.load(path) as a:
            assert int(a['start_step'])==prev;prev=int(a['end_step'])
            for k in keys:pieces[k].append(a[k])
        if prev*.0001>=end:break
    a={k:np.concatenate(v) for k,v in pieces.items()}
    with np.load(f.OUT/'geometry.npz') as g:a.update({k:g[k] for k in g.files})
    return a,result

def events(rate,dt=.01,quiet=5.,quiet_duration=.02,peak=20.,end=np.inf):
    qs=[(lo,hi) for lo,hi in f.spans(rate<quiet) if (hi-lo)*dt>=quiet_duration-1e-9]
    ev=[]
    for (qlo,lo),(hi,_) in zip(qs[:-1],qs[1:]):
        if (hi-lo)*dt<.02-1e-9 or hi*dt>=end or rate[lo:hi].max()<peak:continue
        ev.append(dict(start_s=lo*dt,end_s=hi*dt,duration_s=(hi-lo)*dt,quiet_gap_s=(lo-qlo)*dt,
                       peak_s=(lo+int(np.argmax(rate[lo:hi]))+.5)*dt,peak_Hz=float(rate[lo:hi].max())))
    for k,e in enumerate(ev):
        e['onset_interval_s']=None if k==0 else e['start_s']-ev[k-1]['start_s']
        e['interevent_gap_s']=None if k==0 else e['start_s']-ev[k-1]['end_s']
    return ev

def summarize(ev,rate,dt,window,quiet,z,zt):
    lo,hi=window;take=[e for e in ev if lo<=e['start_s']<hi]
    def med(k):
        v=[e[k] for e in take if e[k] is not None];return float(np.median(v)) if v else None
    r=rate[round(lo/dt):round(hi/dt)]
    return dict(window_s=window,count=len(take),event_frequency_Hz=len(take)/(hi-lo),
        median_duration_ms=None if not take else med('duration_s')*1000,
        median_onset_interval_ms=None if med('onset_interval_s') is None else med('onset_interval_s')*1000,
        median_quiet_gap_ms=None if med('quiet_gap_s') is None else med('quiet_gap_s')*1000,
        active_fraction=float((r>=quiet).mean()),mean_rate_Hz=float(r.mean()),
        mean_Z=float(np.interp(np.arange(lo,hi,dt)+dt/2,zt,z).mean()))

def analyze(row):
    a,r=load_small(row['source']);first=r['tracker']['entries'][0]['onset_s'] if r['tracker']['entries'] else r['end_s']
    windows={'early':[.5,3.5],'late':[first-3.5,first-.5]};assert windows['late'][0]>=3.5
    variants=[]
    for dt,quiet in [(.01,5.),(.005,5.),(.01,1.)]:
        n=round(dt*1000);length=len(a['spikes_1ms'])//n*n
        allE=a['spikes_1ms'][:length,0].reshape(-1,n).sum(1)/32000/dt
        local=a['regions_1ms'][:length,:2].reshape(-1,n,2).sum(1)/a['region_counts'][:2]/dt
        pops={}
        for i,(name,rate) in enumerate([('All E',allE),('Core A',local[:,0]),('Core B',local[:,1])]):
            ev=events(rate,dt=dt,quiet=quiet,end=first)
            z=a['Z'][:,[0,5,6][i]];zt=a['slow_time_ms']/1000
            stats={k:summarize(ev,rate,dt,w,quiet,z,zt) for k,w in windows.items()}
            temporal=[]
            for end in np.arange(1.5,first-.25,.25):
                w=[max(.5,float(end)-2),float(end)]
                if w[1]-w[0]>=1:temporal.append(summarize(ev,rate,dt,w,quiet,z,zt))
            pops[name]=dict(events=ev,windows=stats,rolling=temporal)
        variants.append(dict(bin_ms=dt*1000,quiet_Hz=quiet,populations=pops))
    return dict(name=row['name'],source=str(row['source']),eta_m=row['eta_m'],seed=row['seed'],first_onset_s=first,
        variants=variants,mean_adaptation_before_first=float((row['eta_m']*a['M'][a['slow_time_ms']<first*1000,0]).mean()))

def plot(rows):
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(17,10),gridspec_kw={'wspace':.40,'hspace':.50})
    labels=[('event_frequency_Hz','Finite events (s⁻¹)'),('median_duration_ms','Event duration (ms)'),('active_fraction','Active fraction')]
    for i,seed in enumerate([9108401,9108402]):
        for j,(key,label) in enumerate(labels):
            ax=axes[i,j]
            for eta,col in [(.001,'#315c91'),(.0005,'#e07832'),(0.,'#777777')]:
                row=next((r for r in rows if r['seed']==seed and r['eta_m']==eta),None)
                if row is None:continue
                ws=row['variants'][0]['populations']['All E']['windows']
                ax.plot([0,1],[ws[k][key] for k in ['early','late']],'o-',c=col,lw=2,ms=8,label=f'ηM = {eta:g}')
            ax.set(xticks=[0,1],xticklabels=['Early','Pre-entry'],ylabel=label,xlim=(-.15,1.15))
            ax.set_title(f'{chr(65+i*3+j)}  Noise seed {i+1}',loc='left',fontsize=19,weight='bold')
            ax.tick_params(labelsize=17);ax.yaxis.label.set_fontsize(18)
            if j==0:ax.legend(fontsize=13,frameon=False)
    for ext in ['png','pdf']:fig.savefig(dest/f'preentry_event_audit.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    (dest/'README.md').write_text('### preentry_event_audit.png / .pdf\n固定双核、同噪声配对比较早期0.5–3.5秒与首次高态前3.5至0.5秒，分别展示有限事件频率、单次持续时间及活动时间比例。事件由全E 10ms率、至少20ms低于5Hz的静息间隔、峰率至少20Hz定义；不把持续高平台切成多个事件。\n**关注点**：每条线是一个条件/噪声实现，两个种子、同一个拓扑，只作模型内描述；时序关联不能单独证明Z的因果作用。\n')
    fig,axes=plt.subplots(2,2,figsize=(13,10),gridspec_kw={'wspace':.35,'hspace':.38})
    for i,seed in enumerate([9108401,9108402]):
        for j,(key,label) in enumerate([('median_duration_ms','Core event duration (ms)'),('active_fraction','Core active fraction')]):
            ax=axes[i,j]
            for eta,ls in [(.001,'-'),(.0005,'--')]:
                row=next((r for r in rows if r['seed']==seed and r['eta_m']==eta),None)
                if row is None:continue
                for core,col in [('Core A','#b33c6c'),('Core B','#197f9e')]:
                    windows=row['variants'][0]['populations'][core]['windows']
                    ax.plot([0,1],[windows[k][key] for k in ['early','late']],marker='o',ls=ls,
                        c=col,lw=2,ms=8,label=f'{core} · ηM={eta:g}')
            ax.set(xticks=[0,1],xticklabels=['Early','Pre-entry'],ylabel=label,xlim=(-.1,1.1))
            ax.set_title(f'{chr(65+i*2+j)}  Noise seed {i+1}',loc='left',weight='bold',fontsize=19)
            ax.tick_params(labelsize=17);ax.yaxis.label.set_fontsize(18)
            ax.set_ylim((25,100) if j==0 else (0,.40))
            if i==0 and j==0:ax.legend(fontsize=11,loc='upper left',frameon=False)
    for ext in ['png','pdf']:fig.savefig(dest/f'core_persistence_audit.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    with (dest/'README.md').open('a') as file:
        file.write('\n### core_persistence_audit.png / .pdf\n按Core A/B分别统计同样等长早晚窗口的有限事件中位持续时间和活动占时。颜色表示核心，实线为原ηM=0.001，虚线为减半ηM=0.0005，每行一个相同噪声种子。\n**关注点**：区分单次活动延长与次数增加，不能把来自同一条轨迹的事件视作独立网络重复；此图呈现关联，不把Z单独解释为已确定原因。\n')

def main():
    rows=[]
    for row in sources():
        if not (row['source']/'result.json').exists():continue
        rows.append(analyze(row));print(row['name'],flush=True)
    report=dict(question='Do finite events increase as Z falls before first sustained-high entry?',
        statistical_unit='One noise realization on one fixed topology; paired eta comparison; n=2 noise seeds.',
        primary_event='10ms population rate; quiet <5Hz for >=20ms; events between quiet intervals, >=20ms duration, peak >=20Hz, self-terminated before first high onset.',
        frequency='Count finite-event starts within a 3s window /3s. Both event boundaries established in full trace; edge-window truncation does not split events.',
        quiet_gap='Duration of the immediately preceding continuous below-threshold quiet interval. Interevent gaps between qualifying bursts are retained separately and may include smaller activity.',
        windows='0.5-3.5s versus first high onset minus3.5 to minus0.5s; no overlap; no entry plateau.',
        sensitivity='5ms bins at5Hz; 10ms bins at1Hz; same20ms quiet duration and peak20Hz.',
        scope='Rate events are not independently validated HFO detections. Z/time association is descriptive, not a frozen-Z causal experiment.',
        runs=rows,producer=str(Path(__file__).resolve()),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    f.write(OUT/'event_audit.json',f.safe(report))
    flat=[]
    for row in rows:
        for pop,val in row['variants'][0]['populations'].items():
            for phase,s in val['windows'].items():flat.append(dict(name=row['name'],seed=row['seed'],eta_m=row['eta_m'],population=pop,phase=phase,**s))
    with (OUT/'event_summary.csv').open('w') as file:
        w=csv.DictWriter(file,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    plot(rows)
    for row in rows:
        if row['eta_m']<=.001:
            w=row['variants'][0]['populations']['All E']['windows']
            print(row['name'],row['first_onset_s'],{k:{key:v[key] for key in ['count','event_frequency_Hz','median_duration_ms','median_quiet_gap_ms','active_fraction']} for k,v in w.items()},flush=True)
if __name__=='__main__':main()
