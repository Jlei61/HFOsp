"""Show all autonomous replicates, including failed distributional predictions."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json

MODELS=['constant','mode_only','activity_only','independent','shared','independent_history']
NAMES=['Constant','Mode only','Activity only','Independent','One shared state','Independent + history']

def extract(r):
    w=r['windows']['5'];q=w['tb_share_quantiles']
    return dict(rate=r['rate_per_hour'],interval_median=r['interval_seconds_quantiles'][2],fano=w['count_fano'],adjacency_excess=r['adjacent_excess'],share_iqr=q[3]-q[1],share_lag1=w['tb_share_autocorrelation_lags124'][0],tb_fraction=r['tb_fraction'])

def main():
    rows=[];base=RUN/'autonomous_generator_v1_4'
    for seconds in (1,5):
        folder=base/f'fit_grid_{seconds}s'
        if not (folder/'status.json').exists():continue
        for model in MODELS:
            for p in folder.glob(model+'_*.json'):
                r=json.loads(p.read_text())
                if r.get('status')=='COMPLETE' and r['job']['model']==model:rows.append(dict(grid_seconds=seconds,model=model,rep=r['job']['rep'],**extract(r)))
    df=pd.DataFrame(rows);df.to_csv(base/'autonomous_summary_table.csv',index=False);real=extract(json.loads((base/'fit_grid_5s/real_summary.json').read_text()));summary=[]
    for (g,m),part in df.groupby(['grid_seconds','model']):
        for measure in real:
            q=part[measure].quantile([.025,.5,.975]);summary.append(dict(grid_seconds=g,model=m,measure=measure,lower=q.iloc[0],median=q.iloc[1],upper=q.iloc[2],patient=real[measure],n=len(part)))
    summary=pd.DataFrame(summary);summary.to_csv(base/'autonomous_distribution_comparison.csv',index=False)
    fig,axs=plt.subplots(2,3,figsize=(15,8));panels=[('rate','Events per observed hour'),('interval_median','Median within-coverage interval (s)'),('fano','5-min count Fano factor'),('adjacency_excess','Same-label excess over global proportion'),('share_iqr','5-min TB fraction: interquartile width'),('share_lag1','5-min TB fraction: lag-1 correlation')]
    for ax,(measure,title) in zip(axs.flat,panels):
        ax.axhline(real[measure],color='black',ls='--',lw=1.3,label='Patient')
        for grid,offset,color in [(5,-.13,'#4682b4'),(1,.13,'#d47829')]:
            for i,m in enumerate(MODELS):
                r=summary[(summary.grid_seconds==grid)&(summary.model==m)&(summary.measure==measure)]
                if not len(r):continue
                r=r.iloc[0];ax.errorbar(i+offset,r['median'],yerr=[[r['median']-r.lower],[r.upper-r['median']]],fmt='o',color=color,ms=4,capsize=2,label=f'{grid}s fit grid' if i==0 else None)
        ax.set_xticks(range(len(MODELS)),NAMES,rotation=30,ha='right',fontsize=8);ax.set_title(title,fontsize=10);ax.spines[['top','right']].set_visible(False)
    axs[0,0].legend(fontsize=8);fig.suptitle('Autonomous effective observed-event generation: discrepancies remain\n64 independent 24-hour runs per condition; intervals show run variation, not parameter uncertainty',fontsize=12);fig.tight_layout(rect=(0,0,1,.94));fp=RUN/'figures'
    for ext in ('png','pdf'):fig.savefig(fp/f'autonomous_generation_audit.{ext}',dpi=180)
    plt.close(fig)

    # Deterministic representatives: first 24 observed calendar hours, replicate 0.
    ev=pd.read_csv(RUN/'events.csv');z=np.load(RUN/'observations.npz');origin=float(z['origin_epoch']);ex=pd.read_csv(RUN/'exposure.csv');seiz=json.loads((RUN/'seizures.json').read_text());fig,axs=plt.subplots(3,2,figsize=(14,9));g=1 if (base/'fit_grid_1s/status.json').exists() else 5
    series=[('Patient: first 24 calendar hours',ev.start_epoch.to_numpy()-origin,ev.label_tb.to_numpy(),None)]
    for m in ('independent_history','shared'):
        r=np.load(base/f'fit_grid_{g}s'/f'{m}_000.npz');series.append((f'{NAMES[MODELS.index(m)]}: replicate 0',r['event_seconds'],r['label_tb'],r))
    edges=np.arange(0,86400+300,300)
    for row,(name,t,y,state) in enumerate(series):
        y=np.asarray(y,dtype=np.int64)
        n=np.histogram(t,edges)[0];tb=np.histogram(t,edges,weights=y)[0];duration=np.full(len(n),300.)
        assert np.all((tb>=0)&(tb<=n))
        if row==0:
            duration[:]=0
            for lo,hi in ex[['start_epoch','end_epoch']].to_numpy()-origin:duration+=np.maximum(0,np.minimum(edges[1:],hi)-np.maximum(edges[:-1],lo))
        good=duration>=285;rate=np.full(len(n),np.nan);share=rate.copy();rate[good]=n[good]*3600/duration[good];ok=good&(n>=10);share[ok]=tb[ok]/n[ok];time=(edges[1:]+edges[:-1])/7200
        axs[row,0].plot(time,rate,color='#555555',lw=.8);axs[row,0].set_ylabel('Events / hour (common symlog scale)');axs[row,0].set_title(name,fontsize=10);axs[row,0].set_yscale('symlog',linthresh=100);axs[row,0].set_ylim(0,15000)
        axs[row,1].plot(time,share,'.-',color='#336699',lw=.7,ms=2);axs[row,1].set_ylim(0,1);axs[row,1].set_ylabel('TB fraction');axs[row,1].axhline(real['tb_fraction'],c='gray',ls=':',lw=.7)
        if state is not None:
            from scipy.special import expit
            cfg=json.loads((base/f'fit_grid_{g}s/contract.json').read_text())['configs'][('independent_history','shared')[row-1]]
            axs[row,1].plot(state['state_time_seconds']/3600,expit(cfg['b']+state['s']),color='#b35806',alpha=.6,lw=.7,label='Slow-state emission; before fast history');axs[row,1].legend(fontsize=7)
        else:
            for seizure in seiz:
                a,b=(seizure['onset']-origin)/3600,(seizure['offset']-origin)/3600
                if 0<=a<=24:
                    for ax in axs[row]:ax.axvspan(a,b,color='#bb5555',alpha=.2)
        for ax in axs[row]:ax.set_xlim(0,24);ax.set_xlabel('Hours');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Representative continuous sequences, selected without outcome screening\n5-minute summaries; patient missing coverage remains missing; simulation does not replay seizure times',fontsize=12);fig.tight_layout(rect=(0,0,1,.94))
    for ext in ('png','pdf'):fig.savefig(fp/f'autonomous_representative_sequences.{ext}',dpi=180)
    plt.close(fig)
    readme=fp/'README.md';txt=readme.read_text()
    if '### autonomous_generation_audit.png' not in txt:
        with readme.open('a') as f:f.write('\n### autonomous_generation_audit.png\n比较各条件64条自主生成24小时序列与患者观测摘要，事件时刻由模型自行产生。标记及误差条为运行间中位数与2.5%–97.5%范围，患者统计来自78小时有效覆盖，因此不是严格匹配时长的后验预测检验。\n**关注点**：间隔、计数波动和标签聚集能否同时恢复；本轮生成器仍有实质偏差。\n\n### autonomous_representative_sequences.png\n展示患者最早24个日历小时，以及预先按编号选取的两种模型第0条生成序列；没有依据相似程度筛选。灰线是5分钟率、蓝线是至少10事件窗口的TB比例，橙线为慢状态的模式发射概率。\n**关注点**：这是有效保留事件的统计生成，不是SNN放电；患者缺失覆盖不填零，模拟不回放患者发作。\n')
    write_json(base/'review_status.json',dict(status='COMPLETE',n_runs=len(df),grids=df.grid_seconds.unique(),representatives='replicate 0, first 24 calendar hours',acceptance='NOT_ACCEPTED_AS_DISTRIBUTION_MATCH'))
    print(summary[(summary.grid_seconds==1)&(summary.measure.isin(['rate','interval_median','fano']))].to_string(index=False),flush=True)

if __name__=='__main__':main()
