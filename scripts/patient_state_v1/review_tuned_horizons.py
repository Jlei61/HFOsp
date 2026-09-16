"""Paired comparison against memory optimized for each prediction distance."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
OUT=RUN/'tuned_horizon_review_v1_17'
def main():
    OUT.mkdir(exist_ok=True);base=pd.read_csv(RUN/'matched_horizon_review_v1_16/predictions.csv.gz');base=base[~base.model.str.startswith('ewma')];common=set(base['index']);d=np.load(RUN/'observations.npz');best={};extra=[];params=[]
    for p in (RUN/'horizon_memory_controls_v1_17/fits').glob('*.json'):
        r=json.loads(p.read_text());j=r['job'];key=(j['fold'],j['horizon'],j['kind'])
        if r['status']=='COMPLETE' and (key not in best or r['train_loglik']>best[key][0]['train_loglik']):best[key]=(r,p)
    for (fold,h,kind),(r,p) in best.items():
        z=np.load(p.with_suffix('.npz'));ix=z['index'];keep=np.isin(ix,list(common));ix=ix[keep];pp=z['predict_tb'][keep];y=d['y'][ix];extra.append(pd.DataFrame(dict(fold=fold,model='tuned_'+kind,horizon_minutes=h,index=ix,hour=d['t'][ix],p_tb=pp,y=y,score=y*np.log(pp)+(1-y)*np.log1p(-pp))));params.append(dict(fold=fold,horizon_minutes=h,kind=kind,tau_hours=np.exp(r['theta'][0]),strength=np.exp(r['theta'][1]),p0=1/(1+np.exp(-r['theta'][2])),success=r['success'],train_loglik=r['train_loglik']))
    table=pd.concat([base,*extra],ignore_index=True);assert table.groupby(['model','horizon_minutes'])['index'].nunique().eq(len(common)).all();table.to_csv(OUT/'predictions.csv.gz',index=False);pd.DataFrame(params).to_csv(OUT/'memory_parameters.csv',index=False);rng=np.random.default_rng(517017);rows=[]
    for h,g in table.groupby('horizon_minutes'):
        for baseline in ['constant','tuned_decay','tuned_frozen']:
            b=g[g.model==baseline]
            for model,m in g.groupby('model'):
                if model==baseline:continue
                x=m.merge(b,on=['fold','index'],suffixes=('_m','_b'),validate='one_to_one');assert len(x)==len(common);x['delta']=x.score_m-x.score_b;x['block']=np.floor(x.hour_m/6).astype(int);num=np.zeros(4000);den=np.zeros(4000)
                for _,part in x.groupby('fold'):
                    q=part.groupby('block').agg(delta=('delta','sum'),n=('index','size'));ix=rng.integers(0,len(q),(4000,len(q)));num+=q.delta.to_numpy()[ix].sum(1);den+=q.n.to_numpy()[ix].sum(1)
                ci=np.quantile(num/den,[.025,.975]);rows.append(dict(model=model,baseline=baseline,horizon_minutes=h,mean_gain=x.delta.mean(),lower=ci[0],upper=ci[1],n_events=len(x)))
    ints=pd.DataFrame(rows);ints.to_csv(OUT/'block_uncertainty.csv',index=False);fig,axs=plt.subplots(1,2,figsize=(12,4.8));models=['ou','ou2_carry0','ou2_carry1','brownian_carry0','brownian_carry1'];colors=['#666666','#2166ac','#b35806','#998ec3','#7b3294']
    for ax,b in zip(axs,['constant','tuned_frozen']):
        for m,c in zip(models,colors):
            g=ints[(ints.model==m)&(ints.baseline==b)].sort_values('horizon_minutes');ax.plot(g.horizon_minutes,g.mean_gain,'o-',c=c,label=m,ms=4);ax.fill_between(g.horizon_minutes,g.lower,g.upper,color=c,alpha=.07)
        ax.axhline(0,c='gray',lw=.8);ax.set_xscale('symlog',linthresh=1);ax.set_xticks([0,1,5,15,60,120],['0','1','5','15','60','120']);ax.set_xlabel('Forecast distance (minutes)');ax.set_ylabel('Gain / event over '+b);ax.spines[['top','right']].set_visible(False)
    axs[0].legend(fontsize=8);fig.suptitle('State models versus horizon-trained memory\nSame 13,184 events; each heuristic fitted only on its training prefix; exploratory 6h-block intervals',fontsize=11);fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'tuned_memory_forecast_comparison.{ext}',dpi=180)
    plt.close(fig);write_json(OUT/'contract.json',dict(status='COMPLETE',same_event_count=len(common),controls='Each horizon gets training-prefix fitted memory time, shrinkage and baseline; starts chosen by training loss only', primary_memory='tuned_frozen', equivalence='Decay and frozen families coincide after horizon-specific strength reparametrization within bounds; they are not independent controls',scope='Stronger predictive controls, still development-data analysis; no clinical type or future observation enters forecasts'))
    readme=RUN/'figures/README.md'
    if '### tuned_memory_forecast_comparison.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### tuned_memory_forecast_comparison.png\n为每个预测距离分别在训练前缀拟合简单指数记忆的时间常数、收缩强度与基线，再与状态模型比较。所有距离使用同样13,184个事件，未读取预测截点后的任何中间事件。\n**关注点**：慢状态优势是否超过针对同一任务优化的简单记忆；阴影为探索性6小时时间块区间。\n')
    print(ints[(ints.model=='ou2_carry1')&ints.baseline.str.startswith('tuned')].to_string(index=False))
if __name__=='__main__':main()
