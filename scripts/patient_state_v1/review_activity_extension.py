"""Assess the causal-activity observation extension with predictions and generation."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.review_predictions import block_bootstrap
OUT=RUN/'activity_extension_review_v1_16'
def main():
    OUT.mkdir(exist_ok=True);table=pd.read_csv(RUN/'drift_version_review_v1_14/forward_predictions.csv.gz');table=table[table.model.isin(['constant','constant_within_coverage','ewma','ou','OU2_adf_carry0','OU2_adf_carry1'])];folds=json.loads((RUN/'splits.json').read_text());d=np.load(RUN/'observations.npz');best={};rows=[];params=[]
    for p in (RUN/'two_scale_activity_v1_16/fits').glob('*.json'):
        r=json.loads(p.read_text());j=r['job'];key=(j['scope'],j['carry'],j['minutes'])
        if r['status']=='COMPLETE' and r['feasible'] and (key not in best or r['loglik']>best[key][0]['loglik']):best[key]=(r,p)
    for (scope,carry,minutes),(r,p) in best.items():
        name=f'activity{minutes}m_carry{int(carry)}';t=np.array(r['theta']);params.append(dict(scope=scope,model=name,activity_coefficient=t[2],tau_background_hours=np.exp(t[-1]),tau_fast_seconds=np.exp(t[3])*3600,loglik=r['loglik'],success=r['success']))
        if scope=='full':continue
        f=next(f for f in folds if scope==f"fold{f['fold']}");ix=np.arange(f['test_start'],f['test_end']);pp=np.load(p.with_suffix('.npz'))['predict_tb'][ix];pp=np.clip(pp,1e-12,1-1e-12);y=d['y'][ix];rows.append(pd.DataFrame(dict(fold=f['fold'],model=name,index=ix,hour=d['t'][ix],p_tb=pp,y=y,score=y*np.log(pp)+(1-y)*np.log1p(-pp))))
    table=pd.concat([table,*rows],ignore_index=True);table.to_csv(OUT/'forward_predictions.csv.gz',index=False);params=pd.DataFrame(params);params.to_csv(OUT/'parameters.csv',index=False);ints=pd.DataFrame(block_bootstrap(table,6));ints.to_csv(OUT/'forward_block_uncertainty.csv',index=False);generated=[]
    for p in (RUN/'activity_mark_generation_v1_16/runs').glob('*.json'):
        r=json.loads(p.read_text());w=r['windows']['5'];generated.append(dict(model=r['job']['model'],rep=r['job']['rep'],adjacent_excess=r['adjacent_excess'],lag1=w['tb_share_autocorrelation_lags124'][0],lag2=w['tb_share_autocorrelation_lags124'][1],lag4=w['tb_share_autocorrelation_lags124'][2]))
    gen=pd.DataFrame(generated);assert len(gen)==256;gen.to_csv(OUT/'generated_summaries.csv',index=False);real=json.loads((RUN/'autonomous_generator_v1_4/fit_grid_1s/real_summary.json').read_text());models=['activity1m_carry0','activity5m_carry0','activity1m_carry1','activity5m_carry1'];fig,axs=plt.subplots(1,3,figsize=(14,4.8))
    for i,m in enumerate(models):
        r=ints[(ints.model==m)&(ints.baseline=='ewma')].iloc[0];axs[0].errorbar(i,r.mean_gain,yerr=[[r.mean_gain-r.lower],[r.upper-r.mean_gain]],fmt='o',c='#2166ac',capsize=2)
        g=params[params.model==m];full=g[g.scope=='full'].iloc[0];axs[1].plot(i,full.activity_coefficient,'o',c='#2166ac');axs[1].plot(np.full(3,i),g[g.scope!='full'].activity_coefficient,'x',c='gray')
        q=gen[gen.model==m].lag2.quantile([.025,.5,.975]);axs[2].errorbar(i,q.iloc[1],yerr=[[q.iloc[1]-q.iloc[0]],[q.iloc[2]-q.iloc[1]]],fmt='o',c='#2166ac',capsize=2)
    axs[0].axhline(0,c='gray',lw=.8);axs[0].set_title('Forward gain over recent-memory baseline');axs[0].set_ylabel('Log-score gain / event; 6h block interval');axs[1].axhline(0,c='gray',lw=.8);axs[1].set_title('Effect of causal past total activity');axs[1].set_ylabel('Log-odds / log relative activity');axs[2].axhline(real['windows']['5']['tb_share_autocorrelation_lags124'][1],c='k',ls='--');axs[2].set_title('Generated 5-min TB fraction: lag 2');axs[2].set_ylabel('Correlation; dashed line = patient')
    names=['1m / new interval','5m / new interval','1m / carry','5m / carry']
    for ax in axs:ax.set_xticks(range(4),names,rotation=25,ha='right',fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Observed activity adjustment: association does not ensure generative adequacy\nDots = full-record coefficients; crosses = training-prefix coefficients; generated intervals use 64 runs',fontsize=11);fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'activity_observation_extension.{ext}',dpi=180)
    plt.close(fig);write_json(OUT/'contract.json',dict(status='COMPLETE',question='Does causal past activity explain excess persistence in the mark model?',scope='One covariate added to the observation mapping; does not model feedback to the state equation or generate event times',acceptance='Review both forward prediction and generated correlations; full likelihood improvement alone is insufficient'))
    readme=RUN/'figures/README.md'
    if '### activity_observation_extension.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### activity_observation_extension.png\n在快慢状态标签概率中加入过去1或5分钟总事件活动量，比较前向评分、活动量系数及条件标签模拟的分钟尺度相关性。点为完整记录系数、叉为各训练前缀系数，模拟误差条来自64条序列。\n**关注点**：活动量关联是否能解释生成中过强的模式持续性；该协变量不是已识别的神经反馈力。\n')
    print(ints[(ints.model.isin(models))&(ints.baseline=='ewma')].to_string(index=False));print(gen.groupby('model').lag2.quantile([.025,.5,.975]).to_string())
if __name__=='__main__':main()
