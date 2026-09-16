"""Patient block context for model propagation residuals; no resampling or gate.

Preserve each existing FIT source block and its actual event count. Model runs
remain separate. This is a descriptive comparison, not a matched-duration test.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import analyze_topic4_propagation_recovery_night as review
an=review.an;rt=review.rt


def main():
    out=review.night.OUT/'patient_block_context';f=out/'figures';f.mkdir(parents=True,exist_ok=True)
    source=rt.read(an.run.PARENT);ev=rt.load_evaluator(source)
    x=np.asarray(ev.fit);labels=np.asarray(ev.fit_labels);blocks=np.asarray(ev.blocks)[ev.index['FIT']]
    names=np.asarray(rt.load_observation_contract(source)['contact_names']);rows=[]
    for block in np.unique(blocks):
        take=blocks==block
        for lab,label in [('ALL',None),('TA',1),('TB',0)]:
            xx=x[take if label is None else take&(labels==label)]
            ref=x if label is None else x[labels==label]
            rows.append(dict(block=int(block),mode=lab,block_n=int(take.sum()),TA_fraction=float(labels[take].mean()),**an.measures(xx,ref,names)))
    an.writecsv(out/'patient_blocks.csv',rows)
    model=list(csv.DictReader((review.night.OUT/'analysis_long/run_observations.csv').open()))
    model=[r for r in model if r['base_id']=='refine_mid_EE075' and r['layer']=='primary']
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    for r in rows:
        if r['mode']=='ALL':
            axes[0].scatter(r['n'],r['TA_fraction'],s=20,c='#555555',zorder=4)
        elif r['lag_joint_n']:
            ax=axes[1 if r['mode']=='TA' else 2]
            ax.plot([r['lag_joint_n']]*2,[r['SCL_minus_ICL_lag_q05_ms'],r['SCL_minus_ICL_lag_q95_ms']],c='#bbbbbb',lw=.7)
            ax.scatter(r['lag_joint_n'],r['SCL_minus_ICL_lag_median_ms'],s=20,c='#555555',zorder=4)
    for i,seed in enumerate(['847101','847102']):
        rr={r['mode']:r for r in model if r['seed']==seed}
        color=['#138b80','#9b59a3'][i];marker=['s','^'][i]
        axes[0].scatter(int(rr['ALL']['n']),int(rr['TA']['n'])/int(rr['ALL']['n']),s=48,c=color,marker=marker,label='模型噪声 '+seed,zorder=5)
        for ax,mode in zip(axes[1:],['TA','TB']):
            r=rr[mode];n=int(r['lag_joint_n'])
            ax.plot([n]*2,[float(r['SCL_minus_ICL_lag_q05_ms']),float(r['SCL_minus_ICL_lag_q95_ms'])],c=color,lw=1.1)
            ax.scatter(n,float(r['SCL_minus_ICL_lag_median_ms']),s=48,c=color,marker=marker,zorder=5)
    axes[0].axhline(labels.mean(),c='black',ls='--',lw=.8,label='患者全FIT比例')
    axes[0].set(title='A  每个数据块的模式比例',ylabel='TA事件比例',xlabel='该数据块 / 模型运行的全部合格事件数',ylim=(-.03,1.03))
    axes[0].scatter([],[],s=20,c='#555555',label='患者：一个FIT数据块')
    axes[0].legend(fontsize=7,loc='lower right')
    for ax,lab in zip(axes[1:],['TA','TB']):
        ax.axhline(0,c='black',lw=.7,ls='--')
        ax.set(title=lab+'：SCL与ICL的时序',ylabel='SCL − ICL 质心时差 (ms)',xlabel='该类中两杆均参与的实际事件数')
    for ax in axes:
        ax.set_xscale('log');ax.grid(alpha=.13)
    fig.suptitle('患者块间变异能否解释当前模型残差？\n患者28个原FIT数据块；模型为端点上移3mm、EE×0.75，两条60秒运行的原合格集合\n点为比例或事件中位数，竖线为事件5–95%范围；不等同置信区间，也未匹配记录时长',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(f/f'patient_block_context.{ext}',dpi=180)
    plt.close(fig)
    rt.write(out/'manifest.json',dict(status='DESCRIPTIVE_CONTEXT_COMPLETE',producer=__file__,producer_sha256=rt.sha(__file__),
        patient_source=source['sources']['evaluator'],patient_events=len(x),blocks=len(np.unique(blocks)),
        definition='For each original FIT block, use every qualifying FIT event without replacing chronology by independent draws; block is existing evaluator identifier.',
        observable='Per-event median participating SCL centroid minus median participating ICL centroid; summarize within patient block or model run and mode.',
        model_source=str(review.night.OUT/'analysis_long/run_observations.csv'),model_candidate='refine_mid_EE075',
        no_claim='No matched-duration or independent-block significance test. Empirical source-block differences are not a calibrated acceptance interval. No new selection criterion or loss.'))
    (f/'README.md').write_text('### patient_block_context.png\n\n左图显示患者每个原FIT数据块的TA比例与事件数；右两图显示TA/TB各块的杆间时差中位数和事件5–95%范围。模型为端点上移3mm、EE0.75的两条60秒原合格集合，颜色和符号分别保留噪声，未将事件当作独立网络；不匹配记录时长，也不作显著性检验。**关注点**：患者比例确有块间变化，但模型SCL系统性偏晚是否仍超出多数患者块的中心位置；经验范围不构成新验收门。\n\n### patient_block_context.pdf\n\n同源矢量版，定义、样本与选择均同PNG。**关注点**：用于审阅患者块间变异和模型残差的联系，不是新的总体恢复率。\n')
    print(dict(output=str(out),blocks=len(np.unique(blocks))),flush=True)


if __name__=='__main__':main()
