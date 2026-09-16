"""Separate conditional-mark checks from exposure-matched autonomous checks."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.review_generation import extract

def interval_table(df,real):
    rows=[]
    for m,part in df.groupby('model'):
        for measure in real:
            if measure not in part:continue
            q=part[measure].quantile([.025,.5,.975]);rows.append(dict(model=m,measure=measure,lower=q.iloc[0],median=q.iloc[1],upper=q.iloc[2],patient=real[measure],n=len(part)))
    return pd.DataFrame(rows)
def figure(table,panels,models,names,title,name):
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for ax,(measure,label) in zip(axs.flat,panels):
        part=table[table.measure==measure];ax.axhline(part.patient.iloc[0],c='k',ls='--',label='Patient')
        for i,m in enumerate(models):
            r=part[part.model==m].iloc[0];ax.errorbar(i,r['median'],yerr=[[r['median']-r.lower],[r.upper-r['median']]],fmt='o',color='#28668c',capsize=3)
        ax.set_xticks(range(len(models)),names,rotation=25,ha='right',fontsize=8);ax.set_title(label,fontsize=10);ax.spines[['top','right']].set_visible(False)
    axs.flat[0].legend(fontsize=8);fig.suptitle(title,fontsize=12);fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'{name}.{ext}',dpi=180)
    plt.close(fig)
def main():
    raw=json.loads((RUN/'autonomous_generator_v1_4/fit_grid_1s/real_summary.json').read_text());real=extract(raw);real['share_lag2']=raw['windows']['5']['tb_share_autocorrelation_lags124'][1];real['share_lag4']=raw['windows']['5']['tb_share_autocorrelation_lags124'][2]
    root=RUN/'conditional_mark_generation_v1_10';rows=[]
    for p in (root/'runs').glob('*.json'):
        r=json.loads(p.read_text());assert r['status']=='COMPLETE';w=r['windows']['5'];rows.append(dict(model=r['job']['model'],rep=r['job']['rep'],tb_fraction=r['tb_fraction'],adjacency_excess=r['adjacent_excess'],share_iqr=w['tb_share_quantiles'][3]-w['tb_share_quantiles'][1],share_lag1=w['tb_share_autocorrelation_lags124'][0],share_lag2=w['tb_share_autocorrelation_lags124'][1],share_lag4=w['tb_share_autocorrelation_lags124'][2]))
    df=pd.DataFrame(rows);assert len(df)==640;df.to_csv(root/'summary.csv',index=False);tab=interval_table(df,real);tab.to_csv(root/'distribution_comparison.csv',index=False)
    models=['constant','ou','ou_history','two_ou6h','two_ou6h_history'];names=['Constant','One OU','OU + history','Two OU (6h)','Two OU + history']
    figure(tab,[('tb_fraction','Overall TB fraction'),('adjacency_excess','Adjacent same-label excess'),('share_iqr','5-min TB fraction: interquartile width'),('share_lag1','5-min TB fraction: lag 1'),('share_lag2','5-min TB fraction: lag 2'),('share_lag4','5-min TB fraction: lag 4')],models,names,'Conditional label generation at actual event times\n128 sequences per model; timing and coverage are fixed inputs','conditional_mark_generation_check')
    root=RUN/'matched_exposure_generation_v1_12';rows=[]
    for p in (root/'runs').glob('*.json'):
        r=json.loads(p.read_text());assert r['status']=='COMPLETE';rows.append(dict(model=r['job']['version']+'_'+r['job']['model'],rep=r['job']['rep'],**extract(r)))
    df=pd.DataFrame(rows);assert len(df)==256;df.to_csv(root/'summary.csv',index=False);tab=interval_table(df,real);tab.to_csv(root/'distribution_comparison.csv',index=False)
    models=['laplace_activity_only','state_grid_activity_only','laplace_independent_history','state_grid_independent_history'];names=['Laplace rate','Grid-refit rate','Laplace + marks','Grid-refit + marks']
    figure(tab,[('rate','Events / observed hour'),('interval_median','Median within-coverage interval (s)'),('fano','5-min count Fano factor'),('adjacency_excess','Adjacent same-label excess'),('share_iqr','5-min TB fraction: interquartile width'),('share_lag1','5-min TB fraction: lag 1')],models,names,'Autonomous interictal generation with matched observation exposure\n64 sequences per model; actual gaps and ictal exclusions conditioned on','matched_exposure_generation_check')
    a=RUN/'generation_histogram_dtype_audit.json';r=json.loads(a.read_text());r.update(status='FIXED_AND_RECOMPUTED',verified_runs=dict(autonomous=768,conditional_marks=640,matched_exposure=256),old_results='Pre-fix folders preserved with pre_int8_histogram_fix suffix');write_json(a,r)
    with (RUN/'figures/README.md').open('a') as f:
        f.write('\n### conditional_mark_generation_check.png\n固定实际事件时刻、覆盖与发作排除边界，每种模型重新生成128条标签序列。显示总体TB比例、相邻聚集和5分钟比例的分布宽度及相关性；误差条为模拟间2.5%–97.5%范围。\n**关注点**：事件率与间隔在这里是输入，不能被当作生成成功；固定6小时背景模型仍是敏感性版本。\n\n### matched_exposure_generation_check.png\n在与患者相同的实际覆盖、缺失与发作排除区间上自主生成间期事件，每条件64条序列。比较Laplace拟合与确定性状态网格重新拟合后的事件率、间隔、计数波动及模式读出。\n**关注点**：改善似然数值近似是否解决真实分布偏差；发作时间与覆盖被条件化，不属于本模型产生的结果。\n')
    write_json(root/'review_status.json',dict(status='COMPLETE',acceptance='NOT_ACCEPTED_AS_JOINT_DISTRIBUTION_MATCH',interpretation='Matched exposure and improved rate likelihood do not by themselves recover event-rate and interval distributions'))
    print(tab[tab.measure.isin(['rate','interval_median','fano'])].to_string(index=False))
if __name__=='__main__':main()
