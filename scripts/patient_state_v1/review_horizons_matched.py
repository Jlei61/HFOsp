"""Compare forecast horizons on the same events and against memory heuristics."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pandas as pd
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.patient_state_v1.common import RUN,write_json
OUT=RUN/'matched_horizon_review_v1_16'
@njit(cache=True)
def counts(y,dt,reset,tau):
    aa=np.empty(len(y));bb=np.empty(len(y));a=0.;b=0.
    for i in range(len(y)):
        if reset[i]:a=0.;b=0.
        else:a*=np.exp(-dt[i]/tau);b*=np.exp(-dt[i]/tau)
        a+=y[i];b+=1-y[i];aa[i]=a;bb[i]=b
    return aa,bb

def main():
    OUT.mkdir(exist_ok=True);df=pd.read_csv(RUN/'two_scale_horizon_v1_16/predictions.csv.gz');d=np.load(RUN/'observations.npz');ev=pd.read_csv(RUN/'events.csv');starts=ev.start_epoch.to_numpy();ends=ev.end_epoch.to_numpy();control=pd.read_csv(RUN/'round2_controls.csv');extra=[]
    for fold,group in df[df.model=='constant'].groupby('fold'):
        r=control[(control.fold==fold)&(control.model=='ewma')].iloc[0];a,b=counts(d['y'],d['dt'],d['reset'],r.tau_hours)
        for h,g in group.groupby('horizon_minutes'):
            target=g['index'].to_numpy();cut=starts[target]-h*60;previous=np.searchsorted(ends,cut,side='right')-1;same=d['epoch'][target]==d['epoch'][previous]
            for name,forecast_time in [('ewma_decay',starts[target]),('ewma_frozen',cut)]:
                elapsed=(forecast_time-starts[previous])/3600;rho=np.exp(-elapsed/r.tau_hours);aa=np.where(same,rho*a[previous],0);bb=np.where(same,rho*b[previous],0);p=(aa+r.strength*r.p0)/(aa+bb+r.strength);y=d['y'][target];extra.append(pd.DataFrame(dict(fold=fold,model=name,horizon_minutes=h,index=target,hour=d['t'][target],p_tb=p,y=y,score=y*np.log(p)+(1-y)*np.log1p(-p))))
    all_events=pd.concat([df,*extra],ignore_index=True);common=set(df[(df.model=='constant')&(df.horizon_minutes==120)]['index']);table=all_events[all_events['index'].isin(common)].copy();assert table.groupby(['model','horizon_minutes'])['index'].nunique().eq(len(common)).all();table.to_csv(OUT/'predictions.csv.gz',index=False);rng=np.random.default_rng(516017);rows=[]
    for horizon,g in table.groupby('horizon_minutes'):
        for baseline in ['constant','ewma_decay','ewma_frozen','ou']:
            base=g[g.model==baseline]
            for model,m in g.groupby('model'):
                if model==baseline:continue
                x=m.merge(base,on=['fold','index'],suffixes=('_m','_b'),validate='one_to_one');assert len(x)==len(common);x['delta']=x.score_m-x.score_b;x['block']=np.floor(x.hour_m/6).astype(int);num=np.zeros(4000);den=np.zeros(4000)
                for _,part in x.groupby('fold'):
                    b=part.groupby('block').agg(delta=('delta','sum'),n=('index','size'));ix=rng.integers(0,len(b),(4000,len(b)));num+=b.delta.to_numpy()[ix].sum(1);den+=b.n.to_numpy()[ix].sum(1)
                q=np.quantile(num/den,[.025,.975]);rows.append(dict(model=model,baseline=baseline,horizon_minutes=horizon,mean_gain=x.delta.mean(),lower=q[0],upper=q[1],n_events=len(x)))
    ints=pd.DataFrame(rows);ints.to_csv(OUT/'block_uncertainty.csv',index=False);fig,axs=plt.subplots(1,2,figsize=(12,4.8));models=['ou','ou2_carry0','ou2_carry1','brownian_carry1','ewma_frozen'];colors=['#666666','#2166ac','#b35806','#7b3294','#1b9e77']
    for ax,base in zip(axs,['constant','ewma_decay']):
        for m,c in zip(models,colors):
            g=ints[(ints.model==m)&(ints.baseline==base)].sort_values('horizon_minutes');ax.plot(g.horizon_minutes,g.mean_gain,'o-',c=c,label=m,ms=4);ax.fill_between(g.horizon_minutes,g.lower,g.upper,color=c,alpha=.07)
        ax.axhline(0,c='gray',lw=.8);ax.set_xscale('symlog',linthresh=1);ax.set_xticks([0,1,5,15,60,120],['0','1','5','15','60','120']);ax.set_xlabel('Minutes without intervening observations');ax.set_ylabel('Forward gain / event over '+base);ax.spines[['top','right']].set_visible(False)
    axs[0].legend(fontsize=8);fig.suptitle(f'Forecast horizons compared on the same {len(common):,} events\n6-hour block uncertainty; no direct previous-label models; future event occurrence not predicted',fontsize=11);fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(RUN/'figures'/f'matched_forecast_horizons.{ext}',dpi=180)
    plt.close(fig);write_json(OUT/'contract.json',dict(status='COMPLETE',n_common_events=len(common),matching='Same events eligible at every horizon including 120min; no changing sample composition across horizon',heuristics='EWMA decay advances old pseudo-counts to target; EWMA frozen advances to observation cutoff then holds its probability; both use training-only hyperparameters',scope='Mode forecast conditional on an event and uninterrupted interictal interval; no unobserved future event/mark consumed',uncertainty='Exploratory time-block intervals, not corrected across all version/horizon comparisons'))
    readme=RUN/'figures/README.md'
    if '### matched_forecast_horizons.png' not in readme.read_text():
        with readme.open('a') as f:f.write('\n### matched_forecast_horizons.png\n在所有预测间隔共同合格的13,184个事件上比较状态模型与近期比例记忆，预测前1–120分钟内不读取任何中间事件。阴影为按折内6小时时间块重采样的探索性区间。\n**关注点**：慢背景是否提供超出简单记忆的未来模式信息；事件发生及无中间发作是条件，不属于被预测的结果。\n')
    print(ints[(ints.model=='ou2_carry1')&ints.baseline.isin(['constant','ewma_decay','ewma_frozen'])].to_string(index=False))
if __name__=='__main__':main()
