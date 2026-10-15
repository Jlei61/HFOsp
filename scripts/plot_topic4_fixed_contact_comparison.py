"""Fixed parent versus threshold-dose control: per-contact observations."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_topic4_geometry_threshold_refinement as a

def main():
    out=a.run.OUT/'first_refinement_response/paired_replays'
    plan=a.rt.read(a.run.OUT/'plan.json')
    evaluator=a.rt.load_evaluator(plan['parent_design'])
    names=list(a.rt.load_observation_contract(plan['parent_design'])['contact_names'])
    cases=['expand_A_4','A4_dose_preserved'];seeds=[847101,847102];rows=[];sources=[]
    def append(cid,seed,mode,table):
        ranks=a.p.ranks(table)
        for j,name in enumerate(names):
            values=ranks[:,j];values=values[np.isfinite(values)]
            rows.append(dict(candidate=cid,seed=seed,mode='TA' if mode==1 else 'TB',contact=name,
                n_events=len(table),n_participating=len(values),participation_probability=float(len(values)/len(table)),
                mean_rank=float(values.mean()) if len(values) else None,
                rank_q05_median_q95=np.percentile(values,[5,50,95]).tolist() if len(values) else None))
    for mode in [1,0]:
        append('patient_FIT',None,mode,np.asarray(evaluator.fit)[np.asarray(evaluator.fit_labels)==mode])
    for cid in cases:
        for seed in seeds:
            path=a.path_for(dict(id=cid,reference=cid=='expand_A_4'),seed);result=a.rt.read(path)
            assert result['status']=='COMPLETE' and result['actual_duration_ms']==90000
            with np.load(path.with_suffix('.npz')) as arrays:
                assert list(arrays['contact_names'])==names
                ids=np.array([i for i in arrays['primary_event_indices'] if result['events'][i]['window_ms'][0]>=1500 and result['events'][i]['window_ms'][1]<=90000],int)
                table=arrays['centroid_ms'];labels=arrays['event_mode']
                for mode in [1,0]:append(cid,seed,mode,table[ids[labels[ids]==mode]])
            sources.append(dict(candidate=cid,seed=seed,json_path=str(path),arrays_sha256=result['arrays_sha256']))
    a.rt.write(out/'contact_observations.json',rows)
    a.rt.write(out/'contact_observation_sources.json',dict(sources=sources,patient_design=plan['parent_design'],
        definition='Per-event normalized centroid rank among participating contacts; per-contact moments conditional on that contact participating. Participation probability uses all qualified events.',
        across_run_summary='Equal noise-run mean and min/max; not a confidence interval.',
        producer_sha256=a.rt.sha(Path(__file__))))
    display=[f'SCL{i}' for i in range(9,5,-1)]+[f'ICL{i}' for i in range(11,0,-1)]
    lookup={(r['candidate'],r['seed'],r['mode'],r['contact']):r for r in rows}
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    labels=['仅扩大左核到 4 mm','扩大左核＋保持双核阈值总量']
    for row,mode in enumerate(['TA','TB']):
        for ax,(metric,title) in zip(axes[row],[('participation_probability','触点参与概率'),('mean_rank','参与后的平均归一化质心顺序')]):
            ref=[lookup[('patient_FIT',None,mode,name)][metric] for name in display]
            ax.plot(ref,'ko-',label='患者 FIT',ms=4)
            for cid,label,color in zip(cases,labels,['#d68532','#397da8']):
                values=np.asarray([[lookup[(cid,seed,mode,name)][metric] for name in display] for seed in seeds],float)
                ax.plot(np.nanmean(values,axis=0),'o-',color=color,label=label,ms=4)
                ax.fill_between(range(len(display)),np.nanmin(values,axis=0),np.nanmax(values,axis=0),color=color,alpha=.18)
            ax.axvline(3.5,color='gray',ls=':',lw=.8)
            ax.set(title=f'{mode} · {title}',xticks=range(15),xticklabels=display,ylim=(-.02,1.02))
            ax.tick_params(axis='x',rotation=60);ax.grid(alpha=.12)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('同一网络、两条噪声：参与缺口与具体触点的顺序偏差分开显示\n线为运行等权均值，阴影为两噪声极值范围；不是置信区间；各触点顺序只在该触点参与的事件中计算')
    for ext in ['png','pdf']:fig.savefig(out/'figures'/f'contact_participation_and_order.{ext}',dpi=150,bbox_inches='tight')
    plt.close(fig)
    readme=out/'figures/README.md';text=readme.read_text()
    section='### contact_participation_and_order.png'
    if section not in text:
        text+='\n'+section+'\n\n固定的扩大左核与阈值总量对照，分别显示患者 TA/TB 的逐触点参与概率和参与后的平均归一化质心顺序。黑色为患者，橙色为出发点，蓝色为新条件；阴影是两条噪声之间的极值范围，不是置信区间。不同触点的条件均值不能直接等同同一批事件的成对先后概率。**关注点**：SCL 与 ICL 的具体残差，不能用总体顺序相关替代整条电极的恢复。\n'
        readme.write_text(text)
    print(json.dumps(dict(output=str(out),rows=len(rows),physical_runs=4)))

if __name__=='__main__':main()
