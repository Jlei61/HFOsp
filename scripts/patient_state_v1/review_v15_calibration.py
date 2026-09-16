"""Review own-model recovery, prefix misspecification and conditional generation."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.review_generation_matched import interval_table,figure
from scripts.patient_state_v1.review_generation import extract
OUT=RUN/'calibration_review_v1_15'
def main():
    OUT.mkdir(exist_ok=True);root=RUN/'free_model_recovery_v1_15';assert (root/'status.json').exists();rows=[]
    for p in (root/'fits').glob('*.json'):
        r=json.loads(p.read_text());assert r['status']=='COMPLETE';j=r['job'];f=r['fit'];t=np.array(f['theta']);truth=np.array(j['theta']);rows.append(dict(carry=j['carry'],rep=j['rep'],success=f['success'],baseline=t[0],history=t[1],tau_fast_s=np.exp(t[2])*3600,sd_fast=np.exp(t[3]),sd_background=np.exp(t[4]),tau_background_h=np.exp(t[5]),true_tau_fast_s=np.exp(truth[2])*3600,true_tau_background_h=np.exp(truth[5]),true_sd_fast=np.exp(truth[3]),true_sd_background=np.exp(truth[4]),fast_lower_hit=bool(t[2]<np.log(1/3600)+.001),background_lower_hit=bool(t[5]<np.log(.25)+.001)))
    df=pd.DataFrame(rows);df.to_csv(OUT/'recovery_summary.csv',index=False);fig,axs=plt.subplots(2,2,figsize=(11,7));panels=[('tau_fast_s','Fast correlation time (s)'),('tau_background_h','Background correlation time (h)'),('sd_fast','Fast stationary SD'),('sd_background','Background stationary SD')];audit=[]
    for ax,(measure,label) in zip(axs.flat,panels):
        for carry,offset,c in [(False,0,'#2166ac'),(True,1,'#b35806')]:
            g=df[df.carry==carry];q=g[measure].quantile([.025,.5,.975]);truth=g['true_'+measure].iloc[0];ax.errorbar(offset,q.iloc[1],yerr=[[q.iloc[1]-q.iloc[0]],[q.iloc[2]-q.iloc[1]]],fmt='o',c=c,capsize=3);ax.plot(offset,truth,'x',c='k',ms=9);audit.append(dict(carry=carry,measure=measure,truth=truth,lower=q.iloc[0],median=q.iloc[1],upper=q.iloc[2],n=len(g),all_success=bool(g.success.all())))
        ax.set_xticks([0,1],['Independent interval prior','Carry background']);ax.set_ylabel(label);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Synthetic recovery under the fitted two-scale model\n64 full records per boundary rule; cross = generating value; intervals = refit variation',fontsize=12);fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'two_scale_parameter_recovery.{ext}',dpi=180)
    plt.close(fig)
    # Fixed-time mark generation uses only label summaries.
    root=RUN/'free_conditional_generation_v1_15';rows=[]
    for p in (root/'runs').glob('*.json'):
        r=json.loads(p.read_text());w=r['windows']['5'];rows.append(dict(model=r['job']['model'],rep=r['job']['rep'],tb_fraction=r['tb_fraction'],adjacency_excess=r['adjacent_excess'],share_iqr=w['tb_share_quantiles'][3]-w['tb_share_quantiles'][1],share_lag1=w['tb_share_autocorrelation_lags124'][0],share_lag2=w['tb_share_autocorrelation_lags124'][1],share_lag4=w['tb_share_autocorrelation_lags124'][2]))
    df=pd.DataFrame(rows);assert len(df)==512;df.to_csv(root/'summary.csv',index=False);raw=json.loads((RUN/'autonomous_generator_v1_4/fit_grid_1s/real_summary.json').read_text());real=extract(raw);real.update(share_lag2=raw['windows']['5']['tb_share_autocorrelation_lags124'][1],share_lag4=raw['windows']['5']['tb_share_autocorrelation_lags124'][2]);tab=interval_table(df,real);tab.to_csv(root/'distribution_comparison.csv',index=False)
    figure(tab,[('tb_fraction','Overall TB fraction'),('adjacency_excess','Adjacent same-label excess'),('share_iqr','5-min TB fraction: interquartile width'),('share_lag1','5-min TB fraction: lag 1'),('share_lag2','5-min TB fraction: lag 2'),('share_lag4','5-min TB fraction: lag 4')],['adf_carry0','adf_carry1','laplace_carry0','laplace_carry1'],['ADF / new interval','ADF / carry background','Laplace / new interval','Laplace / carry background'],'Free two-scale model: conditional label generation\n128 sequences per condition; actual event times and boundaries fixed','free_two_scale_generation_check')
    prefix=[]
    for directory in ['two_scale_prefix_calibration_v1_14','two_scale_free_prefix_calibration_v1_15']:
        assert (RUN/directory/'status.json').exists();rows=[]
        for p in (RUN/directory/'fits').glob('*.json'):
            r=json.loads(p.read_text());assert r['status']=='COMPLETE';j=r['job'];rows.append(dict(carry=j['carry'],rep=j['rep'],scope=j['scope'],logtau=r['theta'][2],success=r['success']))
        df=pd.DataFrame(rows);df.to_csv(RUN/directory/'summary.csv',index=False)
        for carry,g in df.groupby('carry'):
            x=g.pivot(index='rep',columns='scope',values='logtau');delta=x.fold0-x.full;actual=1.159211
            prefix.append(dict(version=directory,carry=carry,n=len(x),actual_difference=actual,synthetic_quantiles=np.quantile(delta,[.025,.5,.975]),n_as_extreme=int((delta>=actual).sum()),calibration_p=(1+(delta>=actual).sum())/(len(x)+1),all_optimizer_success=bool(g.success.all())))
    write_json(OUT/'scientific_audit.json',dict(status='COMPLETE',recovery=audit,prefix=prefix,interpretation='Synthetic recovery is conditional on the fitted model. Failure to reproduce minute-scale correlations or nested-prefix changes limits stationary two-scale adequacy; it does not identify the missing physiological cause.'))
    readme=RUN/'figures/README.md'
    if '### two_scale_parameter_recovery.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### two_scale_parameter_recovery.png\n以自由快慢状态拟合值生成完整标签记录，分别在两种背景边界规则下重拟合64次，每次使用两个优化起点。黑叉为生成真值，点和误差条为重拟合中位数及2.5%–97.5%范围。\n**关注点**：方法在自身模型下能否恢复时间尺度和幅度；这不证明真实患者或SNN遵循该模型。\n\n### free_two_scale_generation_check.png\n展示自由快慢状态在固定实际事件时刻上的512条标签模拟，分别比较两种似然近似和背景边界规则。患者值为黑色虚线，模拟区间反映随机序列间变化。\n**关注点**：相邻标签聚集恢复后，分钟尺度的比例相关性是否仍偏高。\n')
    print(json.dumps(dict(recovery=audit,prefix=prefix),indent=2,default=lambda x:x.tolist()))
if __name__=='__main__':main()
