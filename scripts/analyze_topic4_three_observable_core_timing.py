"""Explain a timing/participation tradeoff from existing primary windows only."""
from pathlib import Path
import sys, warnings
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import analyze_topic4_three_observable_bo as a
from scripts.report_topic4_three_observable_raw import describe

DEST=a.OUT/'overnight_20260914/core_timing_tradeoff'
REFERENCE='bridge_circle_out125_xminus075'
CANDIDATE='g2_b01_p04'


def unit(row):
    path=Path(row['source']);r,t,lab,ids,names=a.load_small(path)
    with np.load(path.with_suffix('.npz')) as z:
        clock=z['trace_time_ms'];core=[z[f'trace_core{x}E_spikes'] for x in ['A','B']]
    scl=[i for i,n in enumerate(names) if n.startswith('SCL')];icl=[i for i,n in enumerate(names) if n.startswith('ICL')]
    events=[]
    for i in ids:
        lo,hi=r['events'][i]['window_ms'];ix=(clock>=lo)&(clock<hi);bins=np.arange(lo,hi+2,2)
        items=[]
        for v in core:
            vv=v[ix];tt=clock[ix];mass=float(vv.sum());h,_=np.histogram(tt,bins,weights=vv)
            items.append(dict(mass=mass,peak2ms=float(bins[np.argmax(h)]+1) if mass else None,
                              peak_count=float(h.max()),t10_ms=float(tt[np.searchsorted(vv.cumsum(),.1*mass)]) if mass else None,
                              t50_ms=float(tt[np.searchsorted(vv.cumsum(),.5*mass)]) if mass else None))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            lag=float(np.nanmedian(t[i,scl])-np.nanmedian(t[i,icl]))
        events.append(dict(event=int(i),mode='TA' if lab[i]==1 else 'TB',window_ms=[lo,hi],core=items,
                           A_mass_fraction=items[0]['mass']/max(1,items[0]['mass']+items[1]['mass']),
                           B_minus_A_peak_ms=items[1]['peak2ms']-items[0]['peak2ms'] if all(q['mass'] for q in items) else None,
                           SCL_minus_ICL_ms=lag,both_rods=bool(np.isfinite(lag)),
                           SCL_contacts=int(np.isfinite(t[i,scl]).sum())))
    return dict(candidate=row['candidate'],noise=row['noise'],topology=row['topology'],source=str(path),
                source_sha256=a.rt.sha(path),N=len(ids),events=events)


def main():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.topic4_pdf_font_guard import install
    install();plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':3})
    DEST.mkdir(exist_ok=True,parents=True);F=DEST/'figures';F.mkdir(exist_ok=True)
    rows=[r for r in a.records() if r['candidate'] in [REFERENCE,CANDIDATE] and r['topology']==2511 and r['stage'] in ['initial','adaptive']]
    runs=[unit(r) for r in rows]
    summaries=[]
    for r in runs:
        for both in [None,True,False]:
            events=[e for e in r['events'] if e['mode']=='TB' and (both is None or e['both_rods']==both)]
            summaries.append(dict(candidate=r['candidate'],noise=r['noise'],TB_both_rods_subset=both,N=len(events),
                                  core_A_mass_fraction=describe([e['A_mass_fraction'] for e in events]),
                                  core_B_minus_A_peak_ms=describe([np.nan if e['B_minus_A_peak_ms'] is None else e['B_minus_A_peak_ms'] for e in events]),
                                  rod_lag_ms=describe([e['SCL_minus_ICL_ms'] for e in events])))
    a.rt.write(DEST/'event_decomposition.json',dict(runs=runs,summaries=summaries,
       definition='Same frozen primary windows. Core traces contain aggregate spikes in 1ms bins; peaks use 2ms bins. Both-rods timing requires at least one participating contact on each rod.',
       scientific_scope='Descriptive within-run association, not causal attribution, proof of patient sources, or a changed objective. All six physical scalars changed in the joint candidate.'))
    colors={REFERENCE:'#377eae',CANDIDATE:'#d07a31'}
    labels={REFERENCE:'固定参考',CANDIDATE:'第一批联合探索点4'}
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    ev,names,_=a.patient();scl=[i for i,n in enumerate(names) if n.startswith('SCL')];icl=[i for i,n in enumerate(names) if n.startswith('ICL')]
    patient=ev.fit[ev.fit_labels==0];p_both=np.isfinite(patient[:,scl]).any(1)&np.isfinite(patient[:,icl]).any(1)
    plags=np.nanmedian(patient[p_both][:,scl],axis=1)-np.nanmedian(patient[p_both][:,icl],axis=1)
    for cid in [REFERENCE,CANDIDATE]:
        for r in sorted([x for x in runs if x['candidate']==cid],key=lambda x:x['noise']):
            events=[e for e in r['events'] if e['mode']=='TB'];noise=r['noise'];marker='o' if noise==847401 else '^';xx=(0 if cid==REFERENCE else 1)+(-.06 if noise==847401 else .06)
            both=[e for e in events if e['both_rods']];other=[e for e in events if not e['both_rods']]
            axes[0,0].scatter(xx,len(both)/len(events),color=colors[cid],marker=marker,s=55)
            axes[0,0].annotate(f'{len(both)}/{len(events)}',(xx,len(both)/len(events)),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
            lag=np.sort([e['SCL_minus_ICL_ms'] for e in both]);axes[0,1].step(lag,np.arange(1,len(lag)+1)/len(lag),where='post',color=colors[cid],ls='-' if noise==847401 else '--',label=f'{labels[cid]} / {noise}')
            for e in both:
                if e['B_minus_A_peak_ms'] is not None:axes[1,0].scatter(e['B_minus_A_peak_ms'],e['SCL_minus_ICL_ms'],color=colors[cid],marker=marker,s=19,alpha=.65)
            for subgroup,offset in [(both,-.1),(other,.1)]:
                if subgroup:axes[1,1].scatter(np.full(len(subgroup),xx+offset),[e['A_mass_fraction'] for e in subgroup],color=colors[cid],marker=marker,s=17,alpha=.5)
    axes[0,0].axhline(p_both.mean(),ls='--',color='black',label='患者FIT TB');axes[0,0].set(xticks=[0,1],xticklabels=list(labels.values()),ylim=(0,1.08),ylabel='同时招募SCL和ICL的TB比例',title='跨杆时间可读的分母有明显变化');axes[0,0].legend(fontsize=8)
    axes[0,1].step(np.sort(plags),np.arange(1,len(plags)+1)/len(plags),where='post',color='black',label='患者FIT TB');axes[0,1].set(xlabel='SCL−ICL质心差 (ms)',ylabel='累计事件比例',title='仅在两杆均参与的TB内比较时差');axes[0,1].legend(fontsize=7)
    axes[1,0].axvline(0,color='gray',lw=.7);axes[1,0].axhline(0,color='gray',lw=.7);axes[1,0].set(xlabel='右核B−左核A聚合发放峰时间 (ms)',ylabel='SCL−ICL质心差 (ms)',title='逐事件关联：横轴为正表示左核先达峰')
    axes[1,1].set(xticks=[-.1,.1,.9,1.1],xticklabels=['参考\n两杆参与','参考\n缺少SCL','联合点4\n两杆参与','联合点4\n缺少SCL'],ylabel='左核占两核总发放质量的比例',ylim=(-.03,1.03),title='发放质量另看；不以微小早期亮点判断起源')
    for ax in axes.ravel():ax.grid(alpha=.15)
    fig.suptitle('TB时差缩短，是否伴随招募缺失和两核错相？\n同一拓扑2511，两条噪声；蓝=固定参考，橙=联合点4，圆=847401，三角=847402',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(F/f'TB_lag_participation_core_timing.{ext}',dpi=150)
    plt.close(fig)
    (F/'README.md').write_text('### TB_lag_participation_core_timing.png\n比较固定参考与第一批联合探索点4的两条配对噪声：TB两杆参与、条件时差分布、两核聚合发放峰差及质量占比。患者参考保留；每点代表一个事件或一条完整运行，横轴为两核峰差时不称其为患者起源。\n**关注点**：短时差是否伴随丢失SCL或另一种两核相位关系；联合点同时改变六个标量，不能把差异归因于一个参数。\n')
    print('CORE_TIMING_DIAGNOSTIC_COMPLETE',len(runs),flush=True)


if __name__=='__main__':main()
