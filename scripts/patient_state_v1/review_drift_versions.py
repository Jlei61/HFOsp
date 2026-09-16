"""Compare free slow OU and Brownian background using identical forward events."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
from scripts.patient_state_v1.review_predictions import block_bootstrap
OUT=RUN/'drift_version_review_v1_14'
def main():
    OUT.mkdir(exist_ok=True);best={}
    for directory,kind in [('two_timescale_free_tau_v1_13','free'),('brownian_fast_marks_v1_12','brownian')]:
        for p in (RUN/directory/'fits').glob('*.json'):
            r=json.loads(p.read_text());j=r['job']
            if r['status']!='COMPLETE' or not j['history']:continue
            name=f"OU2_{j['method']}_carry{int(j['carry'])}" if kind=='free' else f"Brownian_drift{int(j['drift'])}_carry{int(j['carry'])}";key=(j['scope'],name)
            if key not in best or r['loglik']>best[key][0]['loglik']:best[key]=(r,p,kind)
    d=np.load(RUN/'observations.npz');folds=json.loads((RUN/'splits.json').read_text());base=pd.read_csv(RUN/'all_forward_predictions.csv.gz');base=base[base.model.isin(['constant','constant_within_coverage','ewma','ou','ou_history'])];records=[base];params=[]
    for (scope,name),(r,path,kind) in best.items():
        t=np.array(r['theta']);j=r['job'];row=dict(scope=scope,model=name,theta=json.dumps(t.tolist()),loglik=r['loglik'],success=r['success'],baseline=t[0],history=t[1],source=str(path))
        if kind=='free':row.update(fast_tau_seconds=np.exp(t[2])*3600,background_tau_hours=np.exp(t[-1]),fast_sd=np.exp(t[3]),background_sd=np.exp(t[4]))
        else:row.update(drift_per_hour=t[2] if j['drift'] else 0,fast_tau_seconds=np.exp(t[2+int(j['drift'])])*3600,diffusion_per_sqrt_hour=np.exp(t[-1]))
        params.append(row)
        if scope=='full':continue
        f=next(f for f in folds if scope==f"fold{f['fold']}");lo,hi=f['test_start'],f['test_end'];p=np.load(path.with_suffix('.npz'))['predict_tb'][lo:hi];p=np.clip(p,1e-12,1-1e-12);y=d['y'][lo:hi];records.append(pd.DataFrame(dict(fold=f['fold'],model=name,index=np.arange(lo,hi),hour=d['t'][lo:hi],p_tb=p,y=y,score=y*np.log(p)+(1-y)*np.log1p(-p))))
    table=pd.concat(records,ignore_index=True);assert table.groupby('model')['index'].nunique().eq(16157).all();table.to_csv(OUT/'forward_predictions.csv.gz',index=False);pd.DataFrame(params).to_csv(OUT/'parameters.csv',index=False);ints=pd.DataFrame(block_bootstrap(table,6));ints.to_csv(OUT/'forward_block_uncertainty.csv',index=False)
    models=['ou_history','OU2_laplace_carry0','OU2_adf_carry0','OU2_laplace_carry1','OU2_adf_carry1','Brownian_drift0_carry0','Brownian_drift1_carry0','Brownian_drift0_carry1','Brownian_drift1_carry1'];names=['One OU + history','Two OU / Laplace / new interval','Two OU / ADF / new interval','Two OU / Laplace / carry background','Two OU / ADF / carry background','Brownian / no drift / new interval','Brownian / drift / new interval','Brownian / no drift / carry','Brownian / drift / carry']
    fig,axs=plt.subplots(1,2,figsize=(15,6))
    for ax,baseline in zip(axs,['ewma','ou']):
        for i,m in enumerate(models):
            r=ints[(ints.model==m)&(ints.baseline==baseline)].iloc[0];ax.errorbar(r.mean_gain,i,xerr=[[r.mean_gain-r.lower],[r.upper-r.mean_gain]],fmt='o',color='#28668c',capsize=2)
        ax.axvline(0,c='gray',lw=.7);ax.set_yticks(range(len(models)),names,fontsize=8);ax.set_xlabel('Forward log-score gain / event');ax.set_title('Compared with '+baseline);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Slow background alternatives: forward mode prediction\nSame 16,157 events; 6-hour block uncertainty; exploratory development comparisons',fontsize=12);fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'drift_versions_forward.{ext}',dpi=180)
    plt.close(fig)
    drift=[]
    for carry in [False,True]:
        r0=best['full',f'Brownian_drift0_carry{int(carry)}'][0];r1=best['full',f'Brownian_drift1_carry{int(carry)}'][0];drift.append(dict(carry=carry,common_drift_per_hour=r1['theta'][2],full_loglik_gain=r1['loglik']-r0['loglik']))
    write_json(OUT/'scientific_audit.json',dict(status='COMPLETE',constant_drift=drift,scope='Fitting marked interictal events without seizure outcomes; no threshold/reset model',interpretation='A shared constant directional drift adds almost no likelihood within the tested Brownian-background models; this does not exclude state-dependent restoring drift or unobserved time-varying inputs',approximation='Chronological predictions use Gaussian assumed-density filtering. Independent particle checks quantify likelihood approximation at full-data fits.'))
    readme=RUN/'figures/README.md'
    if '### drift_versions_forward.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### drift_versions_forward.png\n在同样16,157个前推测试事件上比较快慢OU与Brownian背景，所有模型都含1秒短时标签历史项。误差条按各折内6小时时间块重采样；背景跨发作延续与独立区间先验分别展示。\n**关注点**：哪类过程有前向模式信息；各版本为开发数据上的探索比较，不能据此宣布发作DDM机制或独立临床验证。\n')
    print(ints[(ints.baseline=='ewma')&ints.model.isin(models)].to_string(index=False));print(drift)
if __name__=='__main__':main()
