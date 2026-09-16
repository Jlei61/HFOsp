"""Display an existing rod-lag observable over record time; no new objective."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import analyze_topic4_three_observable_bo as a

def plot(cid, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.topic4_pdf_font_guard import install
    install();plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':3})
    rows=sorted([r for r in a.records() if r['candidate']==cid and r['stage']=='adaptive'],key=lambda r:r['noise'])
    assert len(rows)==2
    _,names,_=a.patient();scl=[i for i,n in enumerate(names) if n.startswith('SCL')];icl=[i for i,n in enumerate(names) if n.startswith('ICL')]
    patient=a.rt.read(a.A/'raw/observables.json')['patient']
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True)
    output=[]
    for row,r in enumerate(rows):
        meta,t,labels,ids,_=a.load_small(Path(r['source']))
        start=np.array([e['event_time_ms']/1000 for e in meta['events']])
        both=np.isfinite(t[:,scl]).any(1)&np.isfinite(t[:,icl]).any(1)
        all_lag=np.full(len(t),np.nan)
        all_lag[both]=np.nanmedian(t[both][:,scl],axis=1)-np.nanmedian(t[both][:,icl],axis=1)
        media=a.A/f'native_review/{cid}/2511_{r["noise"]}/patient_mean_native_multievent.json'
        selected=a.rt.read(media)['events'] if media.exists() else []
        for col,(mode,k) in enumerate([('TA',1),('TB',0)]):
            ax=axes[row,col];events=ids[labels[ids]==k];valid=events[np.isfinite(all_lag[events])]
            ax.scatter(start[valid],all_lag[valid],s=14,alpha=.7,color=['#3276a8','#db7c36'][row],label='全部两杆参与事件')
            gif=np.array([e['event'] for e in selected if e['mode']==mode],dtype=int)
            gif=gif[np.isfinite(all_lag[gif])]
            if len(gif):ax.scatter(start[gif],all_lag[gif],facecolors='none',edgecolors='#a12c70',marker='s',s=65,label='既定GIF示例')
            bins=[]
            for lo in np.arange(1.5,60,6):
                q=events[(start[events]>=lo)&(start[events]<lo+6)];v=all_lag[q];v=v[np.isfinite(v)]
                med=float(np.median(v)) if len(v) else None
                bins.append(dict(start_s=float(lo),end_s=float(min(lo+6,60)),mode_events=len(q),both_rods_events=len(v),rod_lag_median_ms=med))
            assert [b['mode_events'] for b in bins]==[b[mode] for b in r['six_second_segments']]
            med=[np.nan if b['rod_lag_median_ms'] is None else b['rod_lag_median_ms'] for b in bins]
            ax.plot([(b['start_s']+b['end_s'])/2 for b in bins],med,'k.-',lw=1,ms=5,label='既定6秒段内的中位数')
            ax.axhline(patient[mode]['rod_lag_ms']['median'],color='black',ls='--',label='患者FIT中位数')
            ax.axvspan(0,1.5,color='#999999',alpha=.2)
            ax.set(title=f'{mode} · 噪声 {r["noise"]}；事件 {len(events)}，两杆可读 {len(valid)}',xlabel='记录时间 (s)',ylabel='SCL−ICL 质心差 (ms)',xlim=(0,60))
            ax.grid(alpha=.15)
            if row==0 and col==0:ax.legend(fontsize=8,loc='best')
            output.append(dict(candidate=cid,noise=r['noise'],mode=mode,event_ids=events.tolist(),event_time_s=start[events].tolist(),rod_lag_ms=[float(x) if np.isfinite(x) else None for x in all_lag[events]],six_second_bins=bins,gif_example_ids=gif.tolist()))
    fig.suptitle(f'{cid}：同一观测量随记录时间的变化\n每点一个事件；灰色为排除的启动区间；缺少另一杆的事件不制造时差；此图不改变训练排序',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));dest=Path(dest);dest.mkdir(exist_ok=True,parents=True)
    for ext in ['png','pdf']:fig.savefig(dest/f'{cid}_rod_lag_timecourse.{ext}',dpi=150)
    plt.close(fig);a.rt.write(dest/f'{cid}_rod_lag_timecourse.json',dict(records=output,time_definition='saved event_time_ms; segments start at 1.5 seconds and match frozen observer counts, final segment is partial',interpretation='Descriptive temporal arrangement of an existing observable; not a state or stationarity test.'))
    from scripts.analyze_topic4_three_observable_core_timing import unit
    core_runs=[unit(r) for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharex=True,sharey=True)
    associations=[]
    for ax,r,color in zip(axes,core_runs,['#3276a8','#db7c36']):
        events=[e for e in r['events'] if e['mode']=='TB' and e['both_rods'] and e['B_minus_A_peak_ms'] is not None]
        xx=np.array([e['B_minus_A_peak_ms'] for e in events]);yy=np.array([e['SCL_minus_ICL_ms'] for e in events])
        ax.scatter(xx,yy,s=26,color=color,alpha=.7)
        ax.axvline(0,color='gray',lw=.8);ax.axhline(patient['TB']['rod_lag_ms']['median'],color='black',ls='--',lw=1)
        for e in events:
            if any(e['event'] in x['gif_example_ids'] for x in output if x['noise']==r['noise'] and x['mode']=='TB'):
                ax.annotate(str(e['event']),(e['B_minus_A_peak_ms'],e['SCL_minus_ICL_ms']),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.set(title=f'TB · 噪声 {r["noise"]}；N={len(events)}',xlabel='右核B−左核A聚合发放峰时间 (ms)',ylabel='SCL−ICL 质心差 (ms)');ax.grid(alpha=.15)
        associations.append(dict(noise=r['noise'],N=len(events),pearson_descriptive=float(np.corrcoef(xx,yy)[0,1]) if len(events)>1 and xx.std()>0 and yy.std()>0 else None))
    fig.suptitle('同一TB标签内：两核错相与杆间时差的逐事件联系\n横轴为正表示左核先达峰；只作关联描述，不证明核间因果驱动、双稳态或患者起源',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.88))
    for ext in ['png','pdf']:fig.savefig(dest/f'{cid}_core_peak_rod_lag.{ext}',dpi=150)
    plt.close(fig);a.rt.write(dest/f'{cid}_core_peak_rod_lag.json',dict(runs=core_runs,associations=associations,definition='Same primary 250 ms windows; aggregate core spike traces sampled at 1 ms, peaks in 2 ms bins; existing diagnostic, not training.'))
    print(dest/f'{cid}_rod_lag_timecourse.png')

if __name__=='__main__':plot(sys.argv[1],sys.argv[2])
