"""Report the size, not just existence, of informative-observation bias."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
OUT=RUN/'observation_calibration_review_v1_18'
def main():
    OUT.mkdir(exist_ok=True);rows=[]
    for p in (RUN/'informative_timing_calibration_v1_18/fits').glob('*.json'):
        r=json.loads(p.read_text());assert r['status']=='COMPLETE';j=r['job'];t=r['theta'];rows.append(dict(c=j['c'],kind=j['kind'],rep=j['rep'],baseline=t[0],tau_hours=np.exp(t[1]),sd=np.exp(t[2]),success=r['success']))
    df=pd.DataFrame(rows);assert len(df)==288;df.to_csv(OUT/'informative_timing_summary.csv',index=False);summary=[];fig,axs=plt.subplots(1,3,figsize=(13,4.8));truth=dict(baseline=-.8,tau_hours=.5,sd=.6)
    for ax,(measure,value) in zip(axs,truth.items()):
        ax.axhline(value,c='k',ls='--',label='Generating value')
        for kind,off,c in [('mark24',-.18,'#2166ac'),('mark96',0,'#92c5de'),('joint24',.18,'#b35806')]:
            for i,coupling in enumerate([-1.5,0.,1.5]):
                g=df[(df.kind==kind)&(df.c==coupling)];q=g[measure].quantile([.025,.5,.975]);ax.errorbar(i+off,q.iloc[1],yerr=[[q.iloc[1]-q.iloc[0]],[q.iloc[2]-q.iloc[1]]],fmt='o',c=c,capsize=2,ms=4,label=kind if i==0 else None);summary.append(dict(c=coupling,kind=kind,measure=measure,truth=value,lower=q.iloc[0],median=q.iloc[1],upper=q.iloc[2],all_success=bool(g.success.all())))
        ax.set_xticks(range(3),['-1.5','0','+1.5']);ax.set_xlabel('State-to-log-rate coupling');ax.set_ylabel(measure);ax.spines[['top','right']].set_visible(False)
    axs[0].legend(fontsize=8);fig.suptitle('Known stationary state with state-dependent event occurrence\n32 sequences per condition; joint comparator knows the true rate mapping; synthetic calibration only',fontsize=11);fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'informative_event_timing_calibration.{ext}',dpi=180)
    plt.close(fig);pd.DataFrame(summary).to_csv(OUT/'informative_timing_intervals.csv',index=False)
    # Inspect generated correlations across model/profile choices.
    rows=[]
    for p in (RUN/'generation_model_sensitivity_v1_18/runs').glob('*.json'):
        r=json.loads(p.read_text());w=r['windows']['5'];rows.append(dict(model=r['job']['model'],rep=r['job']['rep'],adjacency_excess=r['adjacent_excess'],lag2=w['tb_share_autocorrelation_lags124'][1],lag4=w['tb_share_autocorrelation_lags124'][2]))
    df=pd.DataFrame(rows);assert len(df)==1152;df.to_csv(OUT/'generation_sensitivity_summary.csv',index=False);real=json.loads((RUN/'autonomous_generator_v1_4/fit_grid_1s/real_summary.json').read_text());panels=[('adjacency_excess',real['adjacent_excess']),('lag2',real['windows']['5']['tb_share_autocorrelation_lags124'][1]),('lag4',real['windows']['5']['tb_share_autocorrelation_lags124'][2])];fig,axs=plt.subplots(1,3,figsize=(14,6));models=sorted(df.model.unique());readouts=[]
    for ax,(measure,val) in zip(axs,panels):
        ax.axvline(val,c='k',ls='--')
        for i,m in enumerate(models):
            g=df[df.model==m];q=g[measure].quantile([.025,.5,.975]);ax.errorbar(q.iloc[1],i,xerr=[[q.iloc[1]-q.iloc[0]],[q.iloc[2]-q.iloc[1]]],fmt='o',c='#2166ac',capsize=2,ms=4);readouts.append(dict(model=m,measure=measure,lower=q.iloc[0],median=q.iloc[1],upper=q.iloc[2],patient=val))
        ax.set_yticks(range(len(models)),[m.replace('profile_history_','').replace('nohistory_adf_','No history / ') for m in models],fontsize=7);ax.set_xlabel(measure);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Generation sensitivity across supported background times and history choices\n128 sequences per point; intervals are simulation variation, not posterior intervals',fontsize=11);fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'profile_generation_sensitivity.{ext}',dpi=180)
    plt.close(fig);pd.DataFrame(readouts).to_csv(OUT/'generation_sensitivity_intervals.csv',index=False);write_json(OUT/'scientific_audit.json',dict(status='COMPLETE',informative_timing='Ignoring the state-dependent event-rate likelihood introduces modest baseline and amplitude biases in this calibration; correlation time is much less affected. Known-rate joint inference recovers the generating parameters. This does not explain the full observed prefix shift by itself.',generation='Longer-lag fraction correlation remains high across these profile points and after dropping the direct history term; not merely one handpicked bad parameter vector',limits='Synthetic calibration and model-conditional sensitivity, not biological identification; two optimizer warnings retained in the timing calibration'))
    readme=RUN/'figures/README.md'
    if '### informative_event_timing_calibration.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### informative_event_timing_calibration.png\n用已知、固定参数的OU同时生成事件时刻和标签，每种状态对事件率的耦合生成32条记录。比较只用标签的24/96小时拟合与已知真实事件率映射的24小时联合拟合。\n**关注点**：忽略事件时刻信息造成的偏差有多大；本例主要影响基线和幅度，不能据此把真实数据的全部不稳定性归因于观察过程。\n\n### profile_generation_sensitivity.png\n在背景时间profile支持网格和去掉直接标签历史的版本中，共生成1,152条条件标签序列。比较相邻聚集和分钟比例的较长滞后相关性，黑线为患者。\n**关注点**：真实数据生成偏差是否只发生在一个任意参数点；这些范围是模拟变动，不是后验区间。\n')
    print(pd.DataFrame(summary)[pd.DataFrame(summary).measure=='baseline'].to_string(index=False))
if __name__=='__main__':main()
